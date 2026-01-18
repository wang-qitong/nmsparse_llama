#!/usr/bin/env python3

import argparse
import os
import random
import sys
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nmsparse_test_utils import convert_to_sparse_format
from sparsegpt_impl.sparsegpt import SparseGPT


LLAMA_MODEL_PATH = "/wangqitong/llama_2-7b"
DEFAULT_CACHE_FILE = "/wangqitong/sparse_params_sparsity0.5_bank16_mlp_qkvo_8_16_sparsegpt.pt"
LOCAL_WIKITEXT2_PATH = "/wangqitong/wikitext-2-raw-v1"

PROJ_NAMES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _get_projection_module(layer, proj_name: str):
    if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        return getattr(layer.self_attn, proj_name)
    if proj_name in ["gate_proj", "up_proj", "down_proj"]:
        return getattr(layer.mlp, proj_name)
    raise ValueError(f"unknown proj_name={proj_name}")


def _load_wikitext2_train_tokens(
    tokenizer,
    *,
    max_examples: int,
) -> torch.Tensor:
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Failed to import HuggingFace `datasets`. Install it via: pip install datasets\n"
            f"Original error: {e}"
        )

    if os.path.exists(LOCAL_WIKITEXT2_PATH):
        data_files = {
            "train": f"{LOCAL_WIKITEXT2_PATH}/train-00000-of-00001.parquet",
        }
        ds = load_dataset("parquet", data_files=data_files, split="train")
    else:
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    texts = ds["text"]
    if max_examples > 0:
        texts = texts[:max_examples]
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        raise RuntimeError("wikitext2 train split contained no usable text")

    enc = tokenizer(" ".join(texts), return_tensors="pt")
    return enc["input_ids"][0]


def _make_calib_batches(
    input_ids_1d: torch.Tensor,
    *,
    nsamples: int,
    seqlen: int,
    seed: int,
) -> List[torch.Tensor]:
    if int(input_ids_1d.numel()) < seqlen + 1:
        raise ValueError("tokenized calibration text too short")

    random.seed(seed)
    batches = []
    max_start = int(input_ids_1d.numel()) - seqlen - 1
    for _ in range(nsamples):
        i = random.randint(0, max_start)
        j = i + seqlen
        batches.append(input_ids_1d[i:j].unsqueeze(0))
    return batches


@torch.no_grad()
def _collect_layer_inputs(
    model,
    batches: List[torch.Tensor],
    *,
    dev: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    hidden_size = int(model.config.hidden_size)
    nsamples = len(batches)

    inps_cpu = torch.empty((nsamples, model.seqlen, hidden_size), dtype=torch.float16, device="cpu")
    cache: Dict[str, torch.Tensor] = {"i": 0, "attention_mask": None}

    class Catcher(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def __getattr__(self, name: str):
            try:
                return super().__getattr__(name)
            except AttributeError:
                return getattr(self.module, name)

        def forward(self, hidden_states, **kwargs):
            idx = cache["i"]
            hs = hidden_states.detach()
            if hs.dim() == 3 and int(hs.shape[0]) == 1:
                hs = hs[0]
            inps_cpu[idx].copy_(hs.to("cpu", dtype=torch.float16))
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])

    for b in batches:
        attention_mask = torch.ones_like(b, device=dev)
        try:
            model.model(
                input_ids=b.to(dev),
                attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
        except ValueError:
            pass

    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    model.config.use_cache = use_cache

    if cache["attention_mask"] is None:
        raise RuntimeError("failed to capture attention_mask during calibration")

    return inps_cpu, cache["attention_mask"]


def _validate_8_16(mask: torch.Tensor, num_bank_val: int = 16, keep: int = 8):
    if mask.is_cuda:
        mask = mask.cpu()
    K, N = mask.shape
    if K % num_bank_val != 0:
        raise ValueError(f"K={K} must be multiple of num_bank_val={num_bank_val}")

    nb = K // num_bank_val
    m3 = mask.reshape(nb, num_bank_val, N)
    nnz = m3.sum(dim=1)
    if not torch.all(nnz == keep):
        bad = torch.nonzero(nnz != keep, as_tuple=False)[0]
        bank_id = int(bad[0].item())
        col_id = int(bad[1].item())
        actual = int(nnz[bank_id, col_id].item())
        raise ValueError(
            f"8:16 validation failed at bank={bank_id} col={col_id}: expected keep={keep}, got {actual}"
        )


@torch.no_grad()
def build_sparsegpt_cache(
    *,
    model_path: str,
    cache_file: str,
    device: str,
    dtype: torch.dtype,
    nsamples: int,
    seqlen: int,
    seed: int,
    max_train_examples: int,
    prunen: int,
    prunem: int,
    percdamp: float,
    blocksize: int,
    num_bank_val: int,
):
    dev = torch.device(device)

    model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map="cpu")
    model.eval()
    model.seqlen = seqlen

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    input_ids_1d = _load_wikitext2_train_tokens(tokenizer, max_examples=max_train_examples)
    batches = _make_calib_batches(input_ids_1d, nsamples=nsamples, seqlen=seqlen, seed=seed)

    inps_cpu, attention_mask = _collect_layer_inputs(model, batches, dev=dev, dtype=dtype)

    layers = model.model.layers
    sparse_params: Dict[int, Dict[str, Dict]] = {}

    for layer_idx in range(len(layers)):
        layer = layers[layer_idx].to(dev)

        subset = {name: _get_projection_module(layer, name) for name in PROJ_NAMES}
        position_ids = torch.arange(int(seqlen), device=dev, dtype=torch.long).unsqueeze(0)

        originals: Dict[str, torch.Tensor] = {}
        gpts: Dict[str, SparseGPT] = {}

        for name, mod in subset.items():
            originals[name] = mod.weight.detach().to("cpu").clone()
            gpts[name] = SparseGPT(mod)

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name, mod in subset.items():
            handles.append(mod.register_forward_hook(add_batch(name)))

        outs_cpu = torch.empty_like(inps_cpu)
        for j in range(len(batches)):
            out = layer(
                inps_cpu[j].to(dev, dtype=dtype).unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            if out.dim() == 3 and int(out.shape[0]) == 1:
                out = out[0]
            outs_cpu[j].copy_(out.detach().to("cpu", dtype=torch.float16))

        for h in handles:
            h.remove()

        for name in PROJ_NAMES:
            gpts[name].fasterprune(
                sparsity=0.0,
                prunen=prunen,
                prunem=prunem,
                percdamp=percdamp,
                blocksize=blocksize,
            )
            gpts[name].free()

        for j in range(len(batches)):
            out = layer(
                inps_cpu[j].to(dev, dtype=dtype).unsqueeze(0),
                attention_mask=attention_mask,
                position_ids=position_ids,
            )[0]
            if out.dim() == 3 and int(out.shape[0]) == 1:
                out = out[0]
            outs_cpu[j].copy_(out.detach().to("cpu", dtype=torch.float16))

        sparse_params[layer_idx] = {}
        for name in PROJ_NAMES:
            mod = subset[name]

            original_kn = originals[name].t().contiguous().to(torch.float32)
            pruned_kn = mod.weight.detach().to("cpu").t().contiguous().to(torch.float32)

            mask = (pruned_kn != 0).to(torch.float32)
            _validate_8_16(mask, num_bank_val=num_bank_val, keep=(prunem - prunen))

            mat_data_cpu, mat_index_cpu = convert_to_sparse_format(pruned_kn, mask, num_bank_val)

            K, N = pruned_kn.shape
            num_bank = K // num_bank_val
            num_nonzeros_per_bank = int((mask[:num_bank_val, 0].sum().item()))
            if mat_data_cpu.shape != (num_bank * num_nonzeros_per_bank, N):
                raise RuntimeError(f"mat_data shape mismatch: got={tuple(mat_data_cpu.shape)} expected={(num_bank * num_nonzeros_per_bank, N)}")
            if mat_index_cpu.shape != (num_bank * num_nonzeros_per_bank, N):
                raise RuntimeError(f"mat_index shape mismatch: got={tuple(mat_index_cpu.shape)} expected={(num_bank * num_nonzeros_per_bank, N)}")
            if mat_data_cpu.dtype != torch.float32:
                raise RuntimeError(f"mat_data dtype mismatch: got={mat_data_cpu.dtype} expected=torch.float32")
            if mat_index_cpu.dtype != torch.int32:
                raise RuntimeError(f"mat_index dtype mismatch: got={mat_index_cpu.dtype} expected=torch.int32")

            sparse_params[layer_idx][name] = {
                "mat_data": mat_data_cpu,
                "mat_index": mat_index_cpu,
                "pruned_weight": pruned_kn,
                "original_weight": original_kn,
                "w": num_bank * num_nonzeros_per_bank,
                "h": N,
                "block_width": num_nonzeros_per_bank,
                "vec_width": num_bank_val,
                "K": K,
                "N": N,
            }

        layers[layer_idx] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps_cpu = outs_cpu

        if (layer_idx + 1) % 4 == 0:
            print(f"  ✓ Layer {layer_idx} pruned (SparseGPT N:M={prunem - prunen}:{prunem})")

    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    torch.save(sparse_params, cache_file)
    print(f"✓ Saved SparseGPT cache to {cache_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=LLAMA_MODEL_PATH)
    parser.add_argument("--cache_file", type=str, default=DEFAULT_CACHE_FILE)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float16", "float32"])
    parser.add_argument("--nsamples", type=int, default=16)
    parser.add_argument("--seqlen", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_train_examples", type=int, default=2000)
    parser.add_argument("--prunen", type=int, default=8)
    parser.add_argument("--prunem", type=int, default=16)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--num_bank_val", type=int, default=16)
    args = parser.parse_args()

    dtype = torch.float16 if args.dtype == "float16" else torch.float32

    build_sparsegpt_cache(
        model_path=args.model_path,
        cache_file=args.cache_file,
        device=args.device,
        dtype=dtype,
        nsamples=int(args.nsamples),
        seqlen=int(args.seqlen),
        seed=int(args.seed),
        max_train_examples=int(args.max_train_examples),
        prunen=int(args.prunen),
        prunem=int(args.prunem),
        percdamp=float(args.percdamp),
        blocksize=int(args.blocksize),
        num_bank_val=int(args.num_bank_val),
    )


if __name__ == "__main__":
    main()
