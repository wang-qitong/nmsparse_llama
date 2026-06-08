"""
AR-decoding calibration for Qwen3.5/3.6 VLM (Qwen3_5ForConditionalGeneration).

Adapted from llama_ar_token_weighted.py with three Qwen-specific changes:
  1. Nested config: use_cache lives in model.config.text_config, not model.config.
  2. No model.to(dev): 27B model loaded with device_map="auto" across GPUs.
  3. Hessians collected on CPU; moved to each layer's actual device before pruning.
"""

import copy
import time
import os
import json
import math
from collections import deque

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

import sparsegpt as _sparsegpt_module
from sparsegpt import *
from modelutils import *
from quant import *

# Reuse all shared helpers from the AR token-weighted module.
from llama_ar_token_weighted import (
    _safe_cholesky_with_wider_fallback,
    _should_save_full_hessian,
    _add_token_normalized_batch,
    _add_weighted_token_normalized_batch,
    _decode_token_weight,
    _default_baseline_config_file,
    _default_baseline_artifact_file,
    _parse_int_csv,
    _parse_proj_csv,
)

_sparsegpt_module._safe_cholesky = _safe_cholesky_with_wider_fallback

try:
    import wandb
    has_wandb = True
except Exception:
    has_wandb = False


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_qwen35(model_path):
    def skip(*a, **k): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map="auto",
    )
    model.seqlen = 2048
    return model


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _set_use_cache(model, value):
    """Set use_cache on both the top-level config and the nested text_config."""
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = value
    if hasattr(model.config, "text_config") and hasattr(model.config.text_config, "use_cache"):
        model.config.text_config.use_cache = value


def _hidden_size(model):
    cfg = model.config
    if hasattr(cfg, "text_config") and hasattr(cfg.text_config, "hidden_size"):
        return cfg.text_config.hidden_size
    return cfg.hidden_size


def _embed_device(model):
    """Device of the token embedding layer (= first device in device_map)."""
    return next(model.model.embed_tokens.parameters()).device


# ---------------------------------------------------------------------------
# Export masks (same key convention as llama_ar_token_weighted)
# ---------------------------------------------------------------------------

@torch.no_grad()
def export_weight_masks(model):
    masks = {}
    for li, layer in enumerate(model.model.layers):
        for name, mod in find_layers(layer).items():
            key = f"model.layers.{li}.{name}.weight"
            masks[key] = (mod.weight.data != 0).to(torch.uint8).cpu()
    return masks


# ---------------------------------------------------------------------------
# AR calibration — Qwen-adapted llama_sequential_ar
# ---------------------------------------------------------------------------

@torch.no_grad()
def qwen35_sequential_ar(model, prompts, tokenizer, dev, args):
    """
    AR decoding calibration for Qwen3.5/3.6 VLM.

    Differences from llama_sequential_ar:
      - use_cache set via _set_use_cache (handles nested text_config).
      - model.to(dev) is skipped; model is already distributed by device_map.
      - Hessians collected on CPU to be device-agnostic; moved to each layer's
        own device immediately before pruning.
      - input_ids placed on the embedding layer's device, not a fixed `dev`.
    """
    print("Starting Qwen3.5 AR decoding calibration ...")

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    _set_use_cache(model, True)
    # device_map="auto" already distributed layers across GPUs; no model.to(dev) needed.

    layers = model.model.layers
    embed_dev = _embed_device(model)

    # ---- Phase 1: create SparseGPT instances; Hessian on same GPU as each layer ----
    # hessian_dev is set per-layer below to avoid OOM on any single card.
    gpts = {}
    handles = []

    for i, layer in enumerate(layers):
        full = find_layers(layer)
        for name in full:
            if (
                not (args.minlayer <= i < args.maxlayer and args.prune_only in name)
            ) == (not args.invert):
                continue
            key = f"layers.{i}.{name}"
            layer_dev = next(full[name].parameters()).device
            gpts[key] = SparseGPT(full[name], hessian_device=layer_dev)
            if args.wbits < 16:
                gpts[key].quantizer = Quantizer()
                gpts[key].quantizer.configure(
                    args.wbits, perchannel=True, sym=False, mse=False
                )

    print(f"Tracking {len(gpts)} Linear layers for Qwen decoding-only calibration.")

    # ---- Phase 1b: register hooks ----
    collecting = [False]
    collect_mode = [None]
    decode_count = [0]
    prompt_collected_count = [0]
    counter_key = next(iter(gpts), None)
    first_k = getattr(args, "ar_first_k_tokens", 0)
    last_k = getattr(args, "ar_last_k_tokens", 0)
    mix_prefill_decode = getattr(args, "ar_mix_prefill_decode_hessian", False)
    prefill_weight = float(getattr(args, "ar_prefill_weight", 0.5))
    decode_weight = float(getattr(args, "ar_decode_weight", 0.5))
    token_weight_strategy = getattr(args, "ar_token_weight_strategy", "uniform")
    token_weight_min = float(getattr(args, "ar_token_min_weight", 0.8))
    token_weight_max = float(getattr(args, "ar_token_max_weight", 1.2))
    token_weight_horizon = int(getattr(args, "ar_token_weight_horizon", 256))
    token_weighting_enabled = token_weight_strategy != "uniform"
    decode_pos_by_key = {key: 0 for key in gpts}
    last_k_buffers = None
    if last_k > 0:
        last_k_buffers = {key: deque(maxlen=last_k) for key in gpts}

    print(
        f"AR token weighting: strategy={token_weight_strategy}, "
        f"min={token_weight_min}, max={token_weight_max}, horizon={token_weight_horizon}"
    )

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
            decode_pos = decode_pos_by_key[key]
            decode_pos_by_key[key] = decode_pos + 1
            if last_k > 0:
                last_k_buffers[key].append((decode_pos, x.detach()))
                return
            if first_k > 0 and decode_pos >= first_k:
                return
            if token_weighting_enabled:
                alpha = _decode_token_weight(
                    decode_pos, token_weight_strategy,
                    token_weight_min, token_weight_max, token_weight_horizon,
                )
                _add_weighted_token_normalized_batch(gpt, x, alpha)
            elif mix_prefill_decode:
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

    # ---- Phase 1c: generate on each prompt ----
    total_prefill_tokens = 0
    total_decode_tokens = 0
    total_collected_tokens = 0
    max_new = getattr(args, "max_new_tokens", 32768)
    prefill_hessians = {}
    prefill_nsamples = {}
    decode_nsamples = {}

    if mix_prefill_decode:
        print(
            f"Collecting prefill+decode Hessians separately "
            f"(prefill_weight={prefill_weight}, decode_weight={decode_weight}) ..."
        )
        for pi, prompt in enumerate(prompts):
            input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(embed_dev)
            prompt_len = input_ids.shape[1]
            collecting[0] = True
            collect_mode[0] = "prefill"
            model(input_ids=input_ids, use_cache=False)
            collect_mode[0] = None
            collecting[0] = False
            total_prefill_tokens += prompt_len
            print(
                f"  prefill {pi+1}/{len(prompts)}: prompt_len={prompt_len}, "
                f"total_prefill_tokens={total_prefill_tokens}"
            )

        print("Offloading prefill Hessians to CPU and resetting for decode ...")
        for key, gpt in gpts.items():
            prefill_hessians[key] = gpt.H.detach().cpu()
            prefill_nsamples[key] = int(gpt.nsamples)
            gpt.H.zero_()
            gpt.nsamples = 0
            if hasattr(gpt, "weighted_nsamples"):
                gpt.weighted_nsamples = 0.0
        torch.cuda.empty_cache()

    for pi, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(embed_dev)
        prompt_len = input_ids.shape[1]

        decode_count[0] = 0
        prompt_collected_count[0] = 0
        for key in decode_pos_by_key:
            decode_pos_by_key[key] = 0
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
            n_collected = (
                len(next(iter(last_k_buffers.values()))) if last_k_buffers else 0
            )
            for key, buf in last_k_buffers.items():
                if not buf:
                    continue
                if token_weighting_enabled:
                    for decode_pos, x_step in buf:
                        alpha = _decode_token_weight(
                            decode_pos, token_weight_strategy,
                            token_weight_min, token_weight_max, token_weight_horizon,
                        )
                        _add_weighted_token_normalized_batch(gpts[key], x_step, alpha)
                else:
                    x_batch = torch.cat([x for _, x in buf], dim=0)
                    if mix_prefill_decode:
                        _add_token_normalized_batch(gpts[key], x_batch)
                    else:
                        gpts[key].add_batch(x_batch, x_batch)
        else:
            n_collected = prompt_collected_count[0]

        total_decode_tokens += n_decoded
        total_collected_tokens += n_collected
        print(
            f"  prompt {pi+1}/{len(prompts)}: prompt_len={prompt_len}, "
            f"decoded={n_decoded}, collected={n_collected}, "
            f"total_decode={total_decode_tokens}"
        )

    for h in handles:
        h.remove()

    print(
        f"Decoding calibration done: {total_decode_tokens} decode tokens, "
        f"{total_collected_tokens} collected."
        + (f" (first_k={first_k})" if first_k > 0 else "")
        + (f" (last_k={last_k})" if last_k > 0 else "")
    )
    if mix_prefill_decode:
        decode_nsamples = {key: int(gpt.nsamples) for key, gpt in gpts.items()}

    # ---- Phase 2: prune layer-by-layer ----
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
        "ar_token_weight_strategy": token_weight_strategy,
        "ar_token_min_weight": token_weight_min,
        "ar_token_max_weight": token_weight_max,
        "ar_token_weight_horizon": token_weight_horizon,
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
        "hessian_weighted_nsamples": {},
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
            layer_dev = next(full[name].parameters()).device
            gpts[key].move_hessian_to(layer_dev)

            if mix_prefill_decode:
                hp = prefill_hessians.pop(key).to(layer_dev)
                gpts[key].H.mul_(decode_weight)
                gpts[key].H.add_(hp, alpha=prefill_weight)
                gpts[key].nsamples = int(
                    prefill_nsamples.get(key, 0) + decode_nsamples.get(key, 0)
                )
                del hp

            prune_result = gpts[key].fasterprune(
                args.sparsity,
                prunen=args.prunen,
                prunem=args.prunem,
                percdamp=args.percdamp,
                blocksize=args.blocksize,
            )
            artifact_key = f"model.layers.{i}.{name}.weight"
            baseline_artifacts["masks"][artifact_key] = (
                prune_result["mask"].to(torch.uint8).cpu()
            )
            baseline_artifacts["hessian_diag"][artifact_key] = (
                prune_result["hessian_diag"].cpu()
            )
            baseline_artifacts["hessian_nsamples"][artifact_key] = int(
                prune_result["nsamples"]
            )
            baseline_artifacts["hessian_weighted_nsamples"][artifact_key] = float(
                getattr(gpts[key], "weighted_nsamples", gpts[key].nsamples)
            )
            if mix_prefill_decode:
                baseline_artifacts["hessian_prefill_nsamples"][artifact_key] = int(
                    prefill_nsamples.get(key, 0)
                )
                baseline_artifacts["hessian_decode_nsamples"][artifact_key] = int(
                    decode_nsamples.get(key, 0)
                )
            if _should_save_full_hessian(
                i, name.split(".")[-1],
                args.full_hessian_layers, args.full_hessian_projs,
            ):
                baseline_artifacts["hessian_full"][artifact_key] = (
                    prune_result["hessian"].cpu()
                )
            gpts[key].free()

    _set_use_cache(model, True)
    return {}, baseline_artifacts


# ---------------------------------------------------------------------------
# Perplexity eval (text-only, adapted from qwen35.py in sparsegpt-master_2)
# ---------------------------------------------------------------------------

def _to_dev(kwargs, dev):
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(dev)
        elif isinstance(v, (tuple, list)):
            out[k] = type(v)(t.to(dev) if isinstance(t, torch.Tensor) else t for t in v)
        else:
            out[k] = v
    return out


def _fwd(layer, inp, kw, rotary):
    kw = dict(kw)
    if rotary is not None and "position_embeddings" in kw:
        pos_ids = kw.get("position_ids")
        if isinstance(pos_ids, torch.Tensor):
            pos_ids = pos_ids.to(inp.device)
        kw["position_embeddings"] = rotary(inp, pos_ids)
    out = layer(inp, **kw)
    return out[0] if isinstance(out, (tuple, list)) else out


@torch.no_grad()
def qwen35_eval(model, testenc, dev, dataset, log_wandb=False):
    print("Evaluating ...")
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen

    _set_use_cache(model, False)
    layers = model.model.layers
    hs = _hidden_size(model)

    embed_dev = _embed_device(model)
    model.model.embed_tokens = model.model.embed_tokens.to(embed_dev)
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(embed_dev)
    if getattr(model.model, "rotary_emb", None) is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(embed_dev)
    layers[0] = layers[0].to(embed_dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, hs), dtype=dtype, device=embed_dev)
    cache = {"i": 0, "kw": {}}

    class Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, **kw):
            inps[cache["i"]] = inp; cache["i"] += 1; cache["kw"] = kw; raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        try:
            model(testenc[:, i * model.seqlen:(i + 1) * model.seqlen].to(embed_dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    kw = _to_dev(cache["kw"], embed_dev)
    rotary = getattr(model.model, "rotary_emb", None)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(embed_dev)
        for j in range(nsamples):
            outs[j] = _fwd(layer, inps[j].unsqueeze(0), kw, rotary)
        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if getattr(model.model, "norm", None) is not None:
        model.model.norm = model.model.norm.to(embed_dev)
    model.lm_head = model.lm_head.to(embed_dev)
    testenc = testenc.to(embed_dev)
    nlls = []
    for i in range(nsamples):
        h = inps[i].unsqueeze(0)
        if getattr(model.model, "norm", None) is not None:
            h = model.model.norm(h)
        logits = model.lm_head(h)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, i * model.seqlen:(i + 1) * model.seqlen][:, 1:]
        loss = nn.CrossEntropyLoss()(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        nlls.append(loss.float() * model.seqlen)

    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.3f}")
    if log_wandb:
        wandb.log({f"{dataset}/perplexity": ppl.item()})
    _set_use_cache(model, True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    from datautils import get_longwriter_prompts_for_ar, get_loaders

    DEV = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    parser = argparse.ArgumentParser(description="SparseGPT AR calibration for Qwen3.5/3.6 VLM")
    parser.add_argument("model", type=str, help="Path to Qwen3.5/3.6 model")
    parser.add_argument("dataset", type=str,
                        choices=["wikitext2", "ptb", "c4", "longwriter_predjsonl"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--wbits", type=int, default=16)
    parser.add_argument("--minlayer", type=int, default=-1)
    parser.add_argument("--maxlayer", type=int, default=1000)
    parser.add_argument("--prune_only", type=str, default="")
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--save_mask", type=str, default="")
    parser.add_argument("--baseline_config_file", type=str, default="")
    parser.add_argument("--baseline_artifact_file", type=str, default="")
    parser.add_argument("--save_hessian_diag", action="store_true", default=False)
    parser.add_argument("--full_hessian_layers", type=str, default="")
    parser.add_argument("--full_hessian_projs", type=str, default="")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip perplexity eval (recommended for VLM).")
    parser.add_argument("--log_wandb", action="store_true")
    parser.add_argument("--longwriter_jsonl", type=str,
                        default="/wangqitong/LongWriter/evaluation/models/qwen3.6/pred.jsonl")
    parser.add_argument("--calib_tokens", type=int, default=0)
    parser.add_argument("--ar_decoding", action="store_true",
                        help="AR decoding calibration (required for --prunen/--prunem with Qwen).")
    parser.add_argument("--max_new_tokens", type=int, default=32768)
    parser.add_argument("--ar_first_k_tokens", type=int, default=0)
    parser.add_argument("--ar_last_k_tokens", type=int, default=0)
    parser.add_argument("--ar_mix_prefill_decode_hessian", action="store_true")
    parser.add_argument("--ar_prefill_weight", type=float, default=0.5)
    parser.add_argument("--ar_decode_weight", type=float, default=0.5)
    parser.add_argument("--ar_token_weight_strategy", type=str, default="uniform",
                        choices=["uniform", "linear_decrease", "linear_increase",
                                 "log_decrease", "log_increase"])
    parser.add_argument("--ar_token_min_weight", type=float, default=0.8)
    parser.add_argument("--ar_token_max_weight", type=float, default=1.2)
    parser.add_argument("--ar_token_weight_horizon", type=int, default=256)

    args = parser.parse_args()

    args.full_hessian_layers = _parse_int_csv(args.full_hessian_layers)
    args.full_hessian_projs = _parse_proj_csv(args.full_hessian_projs)
    if not args.baseline_config_file:
        args.baseline_config_file = _default_baseline_config_file(args.save, args.save_mask)
    if not args.baseline_artifact_file:
        args.baseline_artifact_file = _default_baseline_artifact_file(args.save, args.save_mask)

    if args.log_wandb:
        assert has_wandb
        wandb.init(config=args)

    model = get_qwen35(args.model)
    model.eval()

    if args.ar_decoding:
        prompts, tokenizer = get_longwriter_prompts_for_ar(
            seed=args.seed,
            model=args.model,
            jsonl_path=args.longwriter_jsonl,
            calib_tokens=args.calib_tokens,
        )
    else:
        dataloader, testloader = get_loaders(
            args.dataset, nsamples=args.nsamples, seed=args.seed,
            model=args.model, seqlen=model.seqlen,
        )

    if (args.sparsity or args.prunen) and not getattr(args, "gmp", False):
        tick = time.time()
        if args.ar_decoding:
            _, baseline_artifacts = qwen35_sequential_ar(
                model, prompts, tokenizer, DEV, args
            )
        else:
            raise ValueError(
                "Non-AR calibration for Qwen3.6 VLM: use sparsegpt-master_2/qwen35.py instead."
            )
        print(f"Pruning took {time.time() - tick:.1f}s")

        if not args.save_hessian_diag:
            baseline_artifacts["hessian_diag"] = {}
        if args.baseline_artifact_file:
            os.makedirs(os.path.dirname(args.baseline_artifact_file) or ".", exist_ok=True)
            torch.save(baseline_artifacts, args.baseline_artifact_file)
            print(f"Saved artifacts to {args.baseline_artifact_file}")
        if args.baseline_config_file:
            cfg_out = {k: (sorted(v) if isinstance(v, set) else v)
                       for k, v in vars(args).items()}
            os.makedirs(os.path.dirname(args.baseline_config_file) or ".", exist_ok=True)
            with open(args.baseline_config_file, "w", encoding="utf-8") as f:
                json.dump(cfg_out, f, indent=2, sort_keys=True)
            print(f"Saved config to {args.baseline_config_file}")

    if args.save_mask:
        mask_state = {
            "model": args.model, "prunen": args.prunen, "prunem": args.prunem,
            "sparsity": args.sparsity, "mask_dtype": "uint8",
            "mask_semantics": "1=kept,0=pruned", "masks": export_weight_masks(model),
        }
        os.makedirs(os.path.dirname(args.save_mask) or ".", exist_ok=True)
        torch.save(mask_state, args.save_mask)
        print(f"Saved masks to {args.save_mask}")

    if not args.skip_eval:
        for ds in ["wikitext2", "ptb", "c4"]:
            try:
                _, testloader = get_loaders(ds, seed=args.seed, model=args.model,
                                            seqlen=model.seqlen)
            except Exception as e:
                print(f"[WARN] Skip eval on {ds}: {e}"); continue
            print("Dataset:", ds)
            qwen35_eval(model, testloader, DEV, ds, args.log_wandb)

    if args.save:
        gc = getattr(model, "generation_config", None)
        if gc is not None and not getattr(gc, "do_sample", True):
            if getattr(gc, "temperature", None) or getattr(gc, "top_p", None):
                model.generation_config = copy.deepcopy(gc)
                model.generation_config.temperature = None
                model.generation_config.top_p = None
        model.save_pretrained(args.save)
        print(f"Saved pruned model to {args.save}")
