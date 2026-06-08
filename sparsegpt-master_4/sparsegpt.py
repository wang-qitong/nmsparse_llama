import math
import time

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def _safe_cholesky(H, diag, upper=False, label="H"):
    last_error = None
    diag_abs_mean = torch.mean(torch.diag(H).abs())
    if not torch.isfinite(diag_abs_mean) or diag_abs_mean <= 0:
        diag_abs_mean = torch.ones((), device=H.device, dtype=H.dtype)

    jitters = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
    for attempt, jitter_scale in enumerate(jitters):
        H_try = H
        if jitter_scale > 0:
            H_try = H.clone()
            jitter = diag_abs_mean * jitter_scale
            H_try[diag, diag] += jitter
        try:
            chol = torch.linalg.cholesky(H_try, upper=upper)
            if attempt > 0:
                print(
                    f"[WARN] {label} Cholesky succeeded after adding "
                    f"jitter={float((diag_abs_mean * jitter_scale).item()):.6e}"
                )
            return chol
        except RuntimeError as err:
            last_error = err

    raise last_error


class SparseGPT:

    def __init__(self, layer, hessian_device=None, init_hessian=True):
        self.layer = layer
        self.dev = self.layer.weight.device
        self.hessian_device = hessian_device if hessian_device is not None else self.dev
        W = layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.hessian_device) if init_hessian else None
        self.nsamples = 0

    def add_batch(self, inp, out, blocksize=1024):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t()
        inp = inp.to(self.hessian_device)
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    def fasterprune(
        self, sparsity, prunen=0, prunem=0, blocksize=128, percdamp=.01
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        if hasattr(self, 'quantizer'):
            if not self.quantizer.ready():
                self.quantizer.find_params(W, weight=True)

        tick = time.time()

        H = self.H
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = 0.5 * (H + H.t())
        H_snapshot = H.clone()
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        Losses = torch.zeros(self.rows, device=self.dev)
        final_mask = torch.zeros_like(W, dtype=torch.bool)

        damp = percdamp * torch.mean(torch.diag(H))
        if not torch.isfinite(damp) or damp <= 0:
            damp = torch.mean(torch.diag(H).abs()) * 1e-6
        if not torch.isfinite(damp) or damp <= 0:
            damp = torch.ones((), device=H.device, dtype=H.dtype) * 1e-6
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = _safe_cholesky(H, diag, upper=False, label="H")
        H = torch.cholesky_inverse(H)
        H = torch.nan_to_num(H, nan=0.0, posinf=0.0, neginf=0.0)
        H = 0.5 * (H + H.t())
        H = _safe_cholesky(H, diag, upper=True, label="Hinv")
        Hinv = H

        mask = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            if prunen == 0: 
                if mask is not None:
                    mask1 = mask[:, i1:i2]
                else:
                    tmp = W1 ** 2 / (torch.diag(Hinv1).reshape((1, -1))) ** 2
                    thresh = torch.sort(tmp.flatten())[0][int(tmp.numel() * sparsity)]
                    mask1 = tmp <= thresh
            else:
                mask1 = torch.zeros_like(W1) == 1

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if prunen != 0 and i % prunem == 0:
                    tmp = W1[:, i:(i + prunem)] ** 2 / (torch.diag(Hinv1)[i:(i + prunem)].reshape((1, -1))) ** 2
                    mask1.scatter_(1, i + torch.topk(tmp, prunen, dim=1, largest=False)[1], True)

                q = w.clone()
                q[mask1[:, i]] = 0

                if hasattr(self, 'quantizer'):
                    q = quantize(
                        q.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()

                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            final_mask[:, i1:i2] = ~mask1
            W[:, i1:i2] = Q1
            Losses += torch.sum(Losses1, 1) / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = W[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
            final_mask = final_mask.t()
            H_snapshot = H_snapshot.t()
        self.layer.weight.data = W.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

        return {
            'mask': final_mask.reshape(self.layer.weight.shape),
            'hessian': H_snapshot,
            'hessian_diag': torch.diag(H_snapshot),
            'nsamples': self.nsamples,
        }

    def move_hessian_to(self, device):
        if self.H is not None:
            self.H = self.H.to(device)
            self.hessian_device = device

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        torch.cuda.empty_cache()
