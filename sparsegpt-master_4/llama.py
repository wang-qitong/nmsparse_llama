import time
import os
import json
import math
from collections import deque

import torch
import torch.nn as nn
from transformers import StoppingCriteria, StoppingCriteriaList

from sparsegpt import *
from modelutils import *
from quant import *

try:
    import wandb
    has_wandb = True
except:
    has_wandb = False


def get_llama(model):
    import torch
    def skip(*args, **kwargs):
        pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model, torch_dtype='auto')
    model.seqlen = 2048
    return model


def _default_baseline_config_file(save_path: str, save_mask_path: str):
    base = save_path or save_mask_path
    if not base:
        return ""
    root, ext = os.path.splitext(base)
    if ext:
        return root + "_baseline_config.json"
    return base.rstrip("/") + "_baseline_config.json"


def _default_baseline_artifact_file(save_path: str, save_mask_path: str):
    base = save_path or save_mask_path
    if not base:
        return ""
    root, ext = os.path.splitext(base)
    if ext:
        return root + "_baseline_artifacts.pt"
    return base.rstrip("/") + "_baseline_artifacts.pt"


def _parse_int_csv(csv_text: str):
    values = set()
    for part in str(csv_text).split(","):
        part = part.strip()
        if part:
            values.add(int(part))
    return values


def _parse_proj_csv(csv_text: str):
    values = set()
    for part in str(csv_text).split(","):
        part = part.strip()
        if part:
            values.add(part)
    return values


def _should_save_full_hessian(layer_idx: int, proj_name: str, full_hessian_layers, full_hessian_projs):
    if not full_hessian_layers:
        return False
    if layer_idx not in full_hessian_layers:
        return False
    if not full_hessian_projs:
        return True
    return proj_name in full_hessian_projs


def _detach_to_cpu(obj):
    if torch.is_tensor(obj):
        return obj.detach().cpu()
    if isinstance(obj, tuple):
        return tuple(_detach_to_cpu(v) for v in obj)
    if isinstance(obj, list):
        return [_detach_to_cpu(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _detach_to_cpu(v) for k, v in obj.items()}
    return obj


def _move_to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, tuple):
        return tuple(_move_to_device(v, device) for v in obj)
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    return obj


def _add_token_normalized_batch(gpt, inp):
    """Accumulate H = 2 / N * sum x x^T where N is the number of token rows."""
    if inp.dim() == 3:
        inp = inp.reshape((-1, inp.shape[-1]))
    elif inp.dim() == 1:
        inp = inp.unsqueeze(0)
    tmp = inp.shape[0]
    if tmp <= 0:
        return
    inp = inp.to(gpt.hessian_device).t()
    gpt.H *= gpt.nsamples / (gpt.nsamples + tmp)
    gpt.nsamples += tmp
    inp = math.sqrt(2 / gpt.nsamples) * inp.float()
    gpt.H += inp.matmul(inp.t())


@torch.no_grad()
def export_weight_masks(model):
    """Export binary masks (1=kept, 0=pruned) for all prunable linear weights."""
    masks = {}
    for layer_idx, layer in enumerate(model.model.layers):
        full = find_layers(layer)
        for name, mod in full.items():
            key = f"model.layers.{layer_idx}.{name}.weight"
            masks[key] = (mod.weight.data != 0).to(torch.uint8).cpu()
    return masks


@torch.no_grad()
def llama_sequential(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    model.model.norm = model.model.norm.to(dev)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    inps = []
    cache = {"layer_kwargs": []}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps.append(inp.detach().cpu())
            cache["layer_kwargs"].append(_detach_to_cpu(kwargs))
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            if isinstance(batch, dict):
                model_inputs = {
                    k: v.to(dev)
                    for k, v in batch.items()
                    if k != "weights"
                }
                model(**model_inputs)
            else:
                model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module

    nbatches = len(inps)
    if nbatches == 0:
        raise ValueError("No calibration samples were captured for pruning.")
    nsamples = sum(inp.shape[0] for inp in inps)

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.cpu()
    torch.cuda.empty_cache()

    outs = [None] * nbatches

    print("Ready.")

    quantizers = {}
    baseline_artifacts = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "nsamples": nsamples,
        "calib_batches": nbatches,
        "seqlen": model.seqlen,
        "prunen": args.prunen,
        "prunem": args.prunem,
        "sparsity": args.sparsity,
        "percdamp": args.percdamp,
        "blocksize": args.blocksize,
        "calib_batch_size": args.calib_batch_size,
        "calib_tokens": args.calib_tokens,
        "true_sequential": bool(args.true_sequential),
        "mask_semantics": "1=kept,0=pruned",
        "masks": {},
        "hessian_diag": {},
        "hessian_nsamples": {},
        "hessian_full": {},
    }
    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
                ["self_attn.o_proj"],
                ["mlp.up_proj", "mlp.gate_proj"],
                ["mlp.down_proj"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names}

            gpts = {}
            for name in subset:
                if (
                    not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
                ) == (not args.invert):
                    continue
                gpts[name] = SparseGPT(subset[name])
                if args.wbits < 16:
                    gpts[name].quantizer = Quantizer()
                    gpts[name].quantizer.configure(
                        args.wbits, perchannel=True, sym=False, mse=False
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(nbatches):
                layer_in = inps[j].to(dev)
                layer_kwargs = _move_to_device(cache["layer_kwargs"][j], dev)
                out = layer(
                    layer_in,
                    **layer_kwargs,
                )
                if isinstance(out, (tuple, list)):
                    out = out[0]
                outs[j] = out.detach().cpu()
            for h in handles:
                h.remove()

            for name in subset:
                print(i, name)
                print("Pruning ...")
                sparsity = args.sparsity
                prune_result = gpts[name].fasterprune(
                    sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                artifact_key = f"model.layers.{i}.{name}.weight"
                baseline_artifacts["masks"][artifact_key] = prune_result["mask"].to(torch.uint8).cpu()
                baseline_artifacts["hessian_diag"][artifact_key] = prune_result["hessian_diag"].cpu()
                baseline_artifacts["hessian_nsamples"][artifact_key] = int(prune_result["nsamples"])
                if _should_save_full_hessian(i, name.split(".")[-1], args.full_hessian_layers, args.full_hessian_projs):
                    baseline_artifacts["hessian_full"][artifact_key] = prune_result["hessian"].cpu()
                gpts[name].free()

        for j in range(nbatches):
            layer_in = inps[j].to(dev)
            layer_kwargs = _move_to_device(cache["layer_kwargs"][j], dev)
            out = layer(
                layer_in,
                **layer_kwargs,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
            outs[j] = out.detach().cpu()

        layers[i] = layer.cpu()
        del layer
        del gpts
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache

    return quantizers, baseline_artifacts


@torch.no_grad()
def llama_sequential_ar(model, prompts, tokenizer, dev, args):
    """
    AR decoding-only calibration: run model.generate() on each prompt,
    collect activations only during decode steps (seq_len==1) via hooks,
    then prune layer-by-layer using the accumulated Hessians.
    """
    print("Starting AR decoding calibration ...")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = True
    model = model.to(dev)

    layers = model.model.layers

    # --- Phase 1: Create SparseGPT instances for ALL target Linears ---
    gpts = {}
    handles = []
    hessian_dev = dev

    for i, layer in enumerate(layers):
        full = find_layers(layer)
        for name in full:
            if (
                not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
            ) == (not args.invert):
                continue
            key = f"layers.{i}.{name}"
            gpts[key] = SparseGPT(full[name], hessian_device=hessian_dev)
            if args.wbits < 16:
                gpts[key].quantizer = Quantizer()
                gpts[key].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

    print(f"Tracking {len(gpts)} Linear layers for decoding-only calibration.")

    # --- Phase 1b: Register hooks (decode-only: skip prefill) ---
    collecting = [False]
    collect_mode = [None]
    decode_count = [0]  # generated decode tokens for current prompt
    prompt_collected_count = [0]
    counter_key = next(iter(gpts), None)
    first_k = getattr(args, 'ar_first_k_tokens', 0)  # 0 means collect all
    last_k = getattr(args, 'ar_last_k_tokens', 0)  # 0 means disabled
    mix_prefill_decode = getattr(args, 'ar_mix_prefill_decode_hessian', False)
    prefill_weight = float(getattr(args, 'ar_prefill_weight', 0.5))
    decode_weight = float(getattr(args, 'ar_decode_weight', 0.5))
    last_k_buffers = None
    if last_k > 0:
        last_k_buffers = {key: deque(maxlen=last_k) for key in gpts}

    def _make_hook(key):
        gpt = gpts[key]
        def _hook(mod, inp, out):
            if not collecting[0]:
                return
            mode = collect_mode[0]
            if mode is None:
                return
            x = inp[0].data
            if mix_prefill_decode and mode == "prefill":
                _add_token_normalized_batch(gpt, x)
                return
            if x.shape[1] != 1:
                return
            if last_k > 0:
                last_k_buffers[key].append(x.detach())
                return
            if first_k > 0 and decode_count[0] >= first_k:
                return
            if mix_prefill_decode:
                _add_token_normalized_batch(gpt, x)
            else:
                gpt.add_batch(x, out.data)
            if key == counter_key:
                prompt_collected_count[0] += x.shape[0]
        return _hook

    class _StopCollectingAfterK(StoppingCriteria):
        def __init__(self, k, prompt_len):
            self.k = k
            self.prompt_len = prompt_len
        def __call__(self, input_ids, scores, **kwargs):
            n_decoded = input_ids.shape[1] - self.prompt_len
            decode_count[0] = n_decoded
            if n_decoded >= self.k:
                collecting[0] = False
            return False

    for i, layer in enumerate(layers):
        full = find_layers(layer)
        for name in full:
            key = f"layers.{i}.{name}"
            if key in gpts:
                handles.append(full[name].register_forward_hook(_make_hook(key)))

    # --- Phase 1c: Generate on each prompt ---
    total_prefill_tokens = 0
    total_decode_tokens = 0
    total_collected_tokens = 0
    max_new = getattr(args, 'max_new_tokens', 32768)
    prefill_hessians = {}
    prefill_nsamples = {}
    decode_nsamples = {}

    if mix_prefill_decode:
        print(
            "Collecting prefill and decode Hessians separately, then mixing them "
            f"(prefill_weight={prefill_weight}, decode_weight={decode_weight}) ..."
        )
        for pi, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
            prompt_len = input_ids.shape[1]

            collecting[0] = True
            collect_mode[0] = "prefill"
            model(input_ids=input_ids, use_cache=False)
            collect_mode[0] = None
            collecting[0] = False

            total_prefill_tokens += prompt_len
            print(f"  prefill prompt {pi+1}/{len(prompts)}: prompt_len={prompt_len}, "
                  f"total_prefill_tokens={total_prefill_tokens}")

        print("Offloading prefill Hessians to CPU and resetting GPU accumulators for decode ...")
        for key, gpt in gpts.items():
            prefill_hessians[key] = gpt.H.detach().cpu()
            prefill_nsamples[key] = int(gpt.nsamples)
            gpt.H.zero_()
            gpt.nsamples = 0
        torch.cuda.empty_cache()

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(dev)
        prompt_len = input_ids.shape[1]

        decode_count[0] = 0
        prompt_collected_count[0] = 0
        if last_k > 0:
            for buf in last_k_buffers.values():
                buf.clear()
        collecting[0] = True
        collect_mode[0] = "decode"
        if first_k > 0:
            stopping_criteria = StoppingCriteriaList(
                [_StopCollectingAfterK(first_k, prompt_len)]
            )
        else:
            stopping_criteria = None
        generated = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stopping_criteria,
        )
        collect_mode[0] = None
        collecting[0] = False

        n_decoded = generated.shape[1] - prompt_len
        if last_k > 0:
            n_collected = len(next(iter(last_k_buffers.values()))) if last_k_buffers else 0
            for key, buf in last_k_buffers.items():
                if not buf:
                    continue
                x_batch = torch.cat(list(buf), dim=0)
                if mix_prefill_decode:
                    _add_token_normalized_batch(gpts[key], x_batch)
                else:
                    gpts[key].add_batch(x_batch, x_batch)
        else:
            n_collected = prompt_collected_count[0]
        total_decode_tokens += n_decoded
        total_collected_tokens += n_collected
        print(f"  prompt {pi+1}/{len(prompts)}: prompt_len={prompt_len}, "
              f"decoded={n_decoded}, collected={n_collected}, "
              f"total_decode_tokens={total_decode_tokens}")

    for h in handles:
        h.remove()

    print(f"Decoding calibration done: {total_decode_tokens} decode tokens generated, "
          f"{total_collected_tokens} collected for calibration"
          + (f" (first_k={first_k} per prompt)" if first_k > 0 else "")
          + (f" (last_k={last_k} per prompt)" if last_k > 0 else "") + ".")
    if mix_prefill_decode:
        decode_nsamples = {key: int(gpt.nsamples) for key, gpt in gpts.items()}

    # --- Phase 2: Prune layer-by-layer ---
    nsamples_example = next(iter(gpts.values())).nsamples if gpts else 0
    baseline_artifacts = {
        "model": args.model,
        "dataset": args.dataset,
        "seed": args.seed,
        "nsamples": nsamples_example,
        "calib_mode": "ar_prefill_decode_mixed" if mix_prefill_decode else "ar_decoding_only",
        "max_new_tokens": max_new,
        "num_prompts": len(prompts),
        "total_prefill_tokens": total_prefill_tokens,
        "total_decode_tokens": total_decode_tokens,
        "total_collected_tokens": total_collected_tokens,
        "ar_first_k_tokens": first_k,
        "ar_last_k_tokens": last_k,
        "ar_mix_prefill_decode_hessian": bool(mix_prefill_decode),
        "ar_prefill_weight": prefill_weight,
        "ar_decode_weight": decode_weight,
        "prunen": args.prunen,
        "prunem": args.prunem,
        "sparsity": args.sparsity,
        "percdamp": args.percdamp,
        "blocksize": args.blocksize,
        "mask_semantics": "1=kept,0=pruned",
        "masks": {},
        "hessian_diag": {},
        "hessian_nsamples": {},
        "hessian_prefill_nsamples": {},
        "hessian_decode_nsamples": {},
        "hessian_full": {},
    }

    for i in range(len(layers)):
        full = find_layers(layers[i])
        for name in full:
            key = f"layers.{i}.{name}"
            if key not in gpts:
                continue
            print(f"{i} {name}")
            print("Pruning ...")
            gpts[key].move_hessian_to(dev)
            if mix_prefill_decode:
                hp = prefill_hessians.pop(key).to(dev)
                gpts[key].H.mul_(decode_weight)
                gpts[key].H.add_(hp, alpha=prefill_weight)
                gpts[key].nsamples = int(prefill_nsamples.get(key, 0) + decode_nsamples.get(key, 0))
                del hp
            prune_result = gpts[key].fasterprune(
                args.sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                blocksize=args.blocksize,
            )
            artifact_key = f"model.layers.{i}.{name}.weight"
            baseline_artifacts["masks"][artifact_key] = prune_result["mask"].to(torch.uint8).cpu()
            baseline_artifacts["hessian_diag"][artifact_key] = prune_result["hessian_diag"].cpu()
            baseline_artifacts["hessian_nsamples"][artifact_key] = int(prune_result["nsamples"])
            if mix_prefill_decode:
                baseline_artifacts["hessian_prefill_nsamples"][artifact_key] = int(prefill_nsamples.get(key, 0))
                baseline_artifacts["hessian_decode_nsamples"][artifact_key] = int(decode_nsamples.get(key, 0))
            if _should_save_full_hessian(i, name.split(".")[-1], args.full_hessian_layers, args.full_hessian_projs):
                baseline_artifacts["hessian_full"][artifact_key] = prune_result["hessian"].cpu()
            gpts[key].free()

    model.config.use_cache = True
    return {}, baseline_artifacts


@torch.no_grad()
def llama_eval(model, testenc, dev,  dataset: str, log_wandb: bool = False):
    print("Evaluating ...")

    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs.get("attention_mask", None)
            raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        batch = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)].to(dev)
        try:
            model(batch)
        except ValueError:
            pass
    layers[0] = layers[0].module

    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    position_ids = torch.arange(model.seqlen, device=dev, dtype=torch.long).unsqueeze(0)
    cache_position = torch.arange(model.seqlen, device=dev, dtype=torch.long)

    if hasattr(model.model, "rotary_emb") and model.model.rotary_emb is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)

        if args.gmp:
            subset = find_layers(layer)
            for name in subset:
                W = subset[name].weight.data
                thresh = torch.sort(torch.abs(W.flatten()))[0][
                    int(W.numel() * args.sparsity)
                ]
                W.data[torch.abs(W.data) <= thresh] = 0

        for j in range(nsamples):
            layer_in = inps[j].unsqueeze(0)
            position_embeddings = model.model.rotary_emb(layer_in, position_ids)
            out = layer(
                layer_in,
                attention_mask=attention_mask,
                position_ids=position_ids,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
            outs[j] = out
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if model.model.norm is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)

    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        hidden_states = inps[i].unsqueeze(0)
        if model.model.norm is not None:
            hidden_states = model.model.norm(hidden_states)
        lm_logits = model.lm_head(hidden_states)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, (i * model.seqlen) : ((i + 1) * model.seqlen)][:, 1:]
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        neg_log_likelihood = loss.float() * model.seqlen
        nlls.append(neg_log_likelihood)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()

    parser.add_argument("model", type=str, help="LlaMA model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4", "longwriter_predjsonl"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument("--sparsity", type=float, default=0, help="Target sparsity")
    parser.add_argument("--prunen", type=int, default=0, help="N for N:M pruning.")
    parser.add_argument("--prunem", type=int, default=0, help="M for N:M pruning.")
    parser.add_argument(
        "--blocksize",
        type=int,
        default=128,
        help="Blocksize to use for adaptive mask selection.",
    )
    parser.add_argument(
        "--gmp", action="store_true", help="Whether to run the GMP baseline."
    )
    parser.add_argument(
        "--wbits", type=int, default=16, help="Whether to quantize as well."
    )
    parser.add_argument(
        "--minlayer", type=int, default=-1, help="Prune all layers with id >= this."
    )
    parser.add_argument(
        "--maxlayer", type=int, default=1000, help="Prune all layers with id < this."
    )
    parser.add_argument(
        "--prune_only",
        type=str,
        default="",
        help="Prune only layers that contain this text.",
    )
    parser.add_argument("--invert", action="store_true", help="Invert subset.")
    parser.add_argument("--save", type=str, default="", help="Path to saved model.")
    parser.add_argument(
        "--save_mask",
        type=str,
        default="",
        help="Path to save binary pruning masks (.pt). 1=kept, 0=pruned.",
    )
    parser.add_argument(
        "--baseline_config_file",
        type=str,
        default="",
        help="Path to save baseline config JSON. Defaults next to --save or --save_mask.",
    )
    parser.add_argument(
        "--baseline_artifact_file",
        type=str,
        default="",
        help="Path to save baseline artifacts PT. Defaults next to --save or --save_mask.",
    )
    parser.add_argument(
        "--save_hessian_diag",
        action="store_true",
        default=False,
        help="Save per-projection H_calib diagonal for the baseline.",
    )
    parser.add_argument(
        "--full_hessian_layers",
        type=str,
        default="",
        help="Comma-separated layer indices for which to save full H_calib matrices.",
    )
    parser.add_argument(
        "--full_hessian_projs",
        type=str,
        default="",
        help="Comma-separated projection suffixes (e.g. q_proj,o_proj) for full H_calib saving.",
    )
    parser.add_argument(
        "--skip_eval",
        action="store_true",
        help="Skip perplexity evaluation on wikitext2/ptb/c4.",
    )
    parser.add_argument(
        "--true-sequential",
        action="store_true",
        help="Whether to run in true sequential model.",
    )
    parser.add_argument(
        "--log_wandb", action="store_true", help="Whether to log to wandb."
    )
    parser.add_argument(
        "--longwriter_jsonl",
        type=str,
        default="/wangqitong/LongWriter/evaluation/models/llama3.1-8b/pred.jsonl",
        help="Path to LongWriter prompt/response JSONL used for RAC-style calibration.",
    )
    parser.add_argument(
        "--calib_batch_size",
        type=int,
        default=8,
        help="Calibration batch size for RAC-style padded LongWriter calibration.",
    )
    parser.add_argument(
        "--calib_tokens",
        type=int,
        default=0,
        help="Token budget for RAC-style LongWriter calibration. Defaults to nsamples * seqlen when 0.",
    )
    parser.add_argument(
        "--ar_decoding",
        action="store_true",
        help="Use AR decoding-only calibration: generate tokens and collect decode-step activations only.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=32768,
        help="Max tokens to generate per prompt in AR decoding calibration mode. "
             "LongWriter evaluation uses 32768 to cover targets up to 20000 words/chars.",
    )
    parser.add_argument(
        "--ar_first_k_tokens",
        type=int,
        default=0,
        help="In AR decoding calibration, only collect activations for the first K decode tokens "
             "per prompt. 0 means collect all tokens (default).",
    )
    parser.add_argument(
        "--ar_last_k_tokens",
        type=int,
        default=0,
        help="In AR decoding calibration, only collect activations for the last K decode tokens "
             "per prompt. 0 disables this mode. Mutually exclusive with --ar_first_k_tokens.",
    )
    parser.add_argument(
        "--ar_mix_prefill_decode_hessian",
        action="store_true",
        help="In AR decoding calibration, collect LongWriter prefill and decode Hessians separately, "
             "normalize each by its own token count, then mix them before pruning.",
    )
    parser.add_argument(
        "--ar_prefill_weight",
        type=float,
        default=0.5,
        help="Weight for prefill Hessian when --ar_mix_prefill_decode_hessian is enabled.",
    )
    parser.add_argument(
        "--ar_decode_weight",
        type=float,
        default=0.5,
        help="Weight for decode Hessian when --ar_mix_prefill_decode_hessian is enabled.",
    )

    args = parser.parse_args()
    if args.ar_first_k_tokens < 0 or args.ar_last_k_tokens < 0:
        parser.error("--ar_first_k_tokens and --ar_last_k_tokens must be non-negative.")
    if args.ar_first_k_tokens > 0 and args.ar_last_k_tokens > 0:
        parser.error("--ar_first_k_tokens and --ar_last_k_tokens are mutually exclusive.")
    if args.ar_last_k_tokens > 0 and not args.ar_decoding:
        parser.error("--ar_last_k_tokens requires --ar_decoding.")
    if args.ar_mix_prefill_decode_hessian and not args.ar_decoding:
        parser.error("--ar_mix_prefill_decode_hessian requires --ar_decoding.")
    if args.ar_prefill_weight < 0 or args.ar_decode_weight < 0:
        parser.error("--ar_prefill_weight and --ar_decode_weight must be non-negative.")
    if args.ar_mix_prefill_decode_hessian and args.ar_prefill_weight + args.ar_decode_weight <= 0:
        parser.error("At least one Hessian mix weight must be positive.")
    args.full_hessian_layers = _parse_int_csv(args.full_hessian_layers)
    args.full_hessian_projs = _parse_proj_csv(args.full_hessian_projs)
    if not args.baseline_config_file:
        args.baseline_config_file = _default_baseline_config_file(args.save, args.save_mask)
    if not args.baseline_artifact_file:
        args.baseline_artifact_file = _default_baseline_artifact_file(args.save, args.save_mask)

    # init W&B logging
    if args.log_wandb:
        assert has_wandb, "wandb not installed try `pip install wandb`"
        wandb.init(config=args)

    model = get_llama(args.model)
    model.eval()

    if args.ar_decoding:
        from datautils import get_longwriter_prompts_for_ar
        prompts, tokenizer = get_longwriter_prompts_for_ar(
            seed=args.seed,
            model=args.model,
            jsonl_path=args.longwriter_jsonl,
            calib_tokens=args.calib_tokens,
        )
        dataloader, testloader = None, None
    else:
        dataloader, testloader = get_loaders(
            args.dataset,
            nsamples=args.nsamples,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            longwriter_jsonl=args.longwriter_jsonl,
            calib_batch_size=args.calib_batch_size,
            calib_tokens=args.calib_tokens,
        )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        if args.ar_decoding:
            _, baseline_artifacts = llama_sequential_ar(model, prompts, tokenizer, DEV, args)
        else:
            _, baseline_artifacts = llama_sequential(model, dataloader, DEV)
        for n, p in model.named_parameters():
            print(n, torch.mean((p == 0).float()))
            if 'down_proj' in n:
                break
        print(time.time() - tick)
        if not args.save_hessian_diag:
            baseline_artifacts["hessian_diag"] = {}
        if args.baseline_artifact_file:
            save_dir = os.path.dirname(args.baseline_artifact_file)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            torch.save(baseline_artifacts, args.baseline_artifact_file)
            print(f"Saved baseline artifacts to {args.baseline_artifact_file}")
        if args.baseline_config_file:
            baseline_config = {
                "model": args.model,
                "dataset": args.dataset,
                "seed": args.seed,
                "nsamples": args.nsamples,
                "seqlen": model.seqlen,
                "percdamp": args.percdamp,
                "sparsity": args.sparsity,
                "prunen": args.prunen,
                "prunem": args.prunem,
                "blocksize": args.blocksize,
                "gmp": bool(args.gmp),
                "wbits": args.wbits,
                "minlayer": args.minlayer,
                "maxlayer": args.maxlayer,
                "prune_only": args.prune_only,
                "invert": bool(args.invert),
                "save": args.save,
                "save_mask": args.save_mask,
                "save_hessian_diag": bool(args.save_hessian_diag),
                "full_hessian_layers": sorted(args.full_hessian_layers),
                "full_hessian_projs": sorted(args.full_hessian_projs),
                "true_sequential": bool(args.true_sequential),
                "skip_eval": bool(args.skip_eval),
                "longwriter_jsonl": args.longwriter_jsonl,
                "calib_batch_size": args.calib_batch_size,
                "calib_tokens": args.calib_tokens,
                "ar_decoding": bool(args.ar_decoding),
                "max_new_tokens": args.max_new_tokens,
                "ar_first_k_tokens": args.ar_first_k_tokens,
                "ar_last_k_tokens": args.ar_last_k_tokens,
                "ar_mix_prefill_decode_hessian": bool(args.ar_mix_prefill_decode_hessian),
                "ar_prefill_weight": args.ar_prefill_weight,
                "ar_decode_weight": args.ar_decode_weight,
                "baseline_config_file": args.baseline_config_file,
                "baseline_artifact_file": args.baseline_artifact_file,
            }
            save_dir = os.path.dirname(args.baseline_config_file)
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
            with open(args.baseline_config_file, "w", encoding="utf-8") as f:
                json.dump(baseline_config, f, indent=2, sort_keys=True)
            print(f"Saved baseline config to {args.baseline_config_file}")

    if args.save_mask:
        mask_state = {
            "model": args.model,
            "prunen": args.prunen,
            "prunem": args.prunem,
            "sparsity": args.sparsity,
            "mask_dtype": "uint8",
            "mask_semantics": "1=kept,0=pruned",
            "masks": export_weight_masks(model),
        }
        save_dir = os.path.dirname(args.save_mask)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        torch.save(mask_state, args.save_mask)
        print(f"Saved masks to {args.save_mask}")

    if not args.skip_eval:
        for dataset in ["wikitext2", "ptb", "c4"]:
            try:
                dataloader, testloader = get_loaders(
                    dataset, seed=args.seed, model=args.model, seqlen=model.seqlen
                )
            except Exception as e:
                print(f"[WARN] Skip eval on {dataset}: {e}")
                continue
            print("Dataset:", dataset)
            llama_eval(model, testloader, DEV, dataset, args.log_wandb)

    if args.save:
        # Vicuna 等模型的 generation_config 可能 do_sample=False 却带 temperature/top_p，保存时校验会报错，先修正
        gc = getattr(model, "generation_config", None)
        if gc is not None and getattr(gc, "do_sample", True) is False:
            if getattr(gc, "temperature", None) is not None or getattr(gc, "top_p", None) is not None:
                import copy
                model.generation_config = copy.deepcopy(gc)
                model.generation_config.temperature = None
                model.generation_config.top_p = None
        model.save_pretrained(args.save)
