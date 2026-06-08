#!/usr/bin/env python3
import argparse
import json
import statistics
from collections import deque

import torch
from transformers import LlamaForCausalLM, StoppingCriteria, StoppingCriteriaList

from datautils import get_longwriter_prompts_for_ar
from modelutils import DEV, find_layers


def parse_csv_set(text):
    vals = set()
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.add(part)
    return vals


def parse_layer_set(text):
    vals = set()
    for part in str(text).split(","):
        part = part.strip()
        if part:
            vals.add(int(part))
    return vals


def add_energy(stats, key, mode, x):
    if x.dim() == 2:
        x = x.unsqueeze(0)
    tokens = x.numel() // x.shape[-1]
    stats[key][f"{mode}_sum"] += x.detach().float().pow(2).sum()
    stats[key][f"{mode}_tokens"] += int(tokens)


def energy_value(stats, key, mode):
    n = stats[key][f"{mode}_tokens"]
    if n <= 0:
        return None
    return (stats[key][f"{mode}_sum"] / n).item()


class DecodeCountUpdater(StoppingCriteria):
    def __init__(self, prompt_len, decode_count):
        self.prompt_len = prompt_len
        self.decode_count = decode_count

    def __call__(self, input_ids, scores, **kwargs):
        self.decode_count[0] = input_ids.shape[1] - self.prompt_len
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Compare LongWriter prefill vs decode activation energy per Llama linear module."
    )
    parser.add_argument("--model", required=True, help="HF model path.")
    parser.add_argument(
        "--longwriter_jsonl",
        default="/wangqitong/LongWriter/evaluation/models/llama3.1-8b/pred.jsonl",
        help="LongWriter JSONL containing prompts.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--calib_tokens",
        type=int,
        default=0,
        help="Optional token budget for selecting LongWriter prompts.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=8,
        help="Number of prompts to run. Use 0 for all selected prompts.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Decode tokens per prompt. Use 32768 for full LongWriter-style generation.",
    )
    parser.add_argument(
        "--decode_position",
        choices=["all", "first", "last"],
        default="all",
        help="Which decode activations to include.",
    )
    parser.add_argument(
        "--decode_k",
        type=int,
        default=50,
        help="K for --decode_position first/last. Ignored for all.",
    )
    parser.add_argument(
        "--modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module suffixes to track.",
    )
    parser.add_argument(
        "--layers",
        default="",
        help="Optional comma-separated layer ids. Empty means all layers.",
    )
    parser.add_argument("--out_json", default="", help="Optional path to save full results.")
    args = parser.parse_args()

    if args.decode_position in {"first", "last"} and args.decode_k <= 0:
        parser.error("--decode_k must be positive when --decode_position is first or last.")

    module_filter = parse_csv_set(args.modules)
    layer_filter = parse_layer_set(args.layers)
    dev = DEV

    print(f"Loading model: {args.model}")
    model = LlamaForCausalLM.from_pretrained(args.model, torch_dtype="auto").to(dev)
    model.eval()
    model.config.use_cache = True

    prompts, tokenizer = get_longwriter_prompts_for_ar(
        seed=args.seed,
        model=args.model,
        jsonl_path=args.longwriter_jsonl,
        calib_tokens=args.calib_tokens,
    )
    if args.max_prompts > 0:
        prompts = prompts[: args.max_prompts]
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Using {len(prompts)} prompts")

    stats = {}
    handles = []
    active_mode = [None]
    decode_count = [0]
    last_buffers = {}

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_filter and layer_idx not in layer_filter:
            continue
        full = find_layers(layer)
        for name, module in full.items():
            suffix = name.split(".")[-1]
            if module_filter and suffix not in module_filter:
                continue
            key = f"layer{layer_idx}.{name}"
            stats[key] = {
                "prefill_sum": torch.zeros((), device=dev),
                "decode_sum": torch.zeros((), device=dev),
                "prefill_tokens": 0,
                "decode_tokens": 0,
            }

    if args.decode_position == "last":
        last_buffers = {key: deque(maxlen=args.decode_k) for key in stats}

    def make_hook(key):
        def hook(mod, inp, out):
            mode = active_mode[0]
            if mode is None:
                return
            x = inp[0]
            if mode == "prefill":
                add_energy(stats, key, "prefill", x)
                return

            if x.dim() < 3 or x.shape[1] != 1:
                return
            if args.decode_position == "first" and decode_count[0] >= args.decode_k:
                return
            if args.decode_position == "last":
                last_buffers[key].append(x.detach())
                return
            add_energy(stats, key, "decode", x)

        return hook

    for layer_idx, layer in enumerate(model.model.layers):
        if layer_filter and layer_idx not in layer_filter:
            continue
        full = find_layers(layer)
        for name, module in full.items():
            suffix = name.split(".")[-1]
            if module_filter and suffix not in module_filter:
                continue
            key = f"layer{layer_idx}.{name}"
            handles.append(module.register_forward_hook(make_hook(key)))

    try:
        with torch.inference_mode():
            for pi, prompt in enumerate(prompts):
                input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(dev)
                prompt_len = input_ids.shape[1]

                active_mode[0] = "prefill"
                model(input_ids=input_ids, use_cache=False)
                active_mode[0] = None

                decode_count[0] = 0
                if args.decode_position == "last":
                    for buf in last_buffers.values():
                        buf.clear()

                active_mode[0] = "decode"
                generated = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    stopping_criteria=StoppingCriteriaList(
                        [DecodeCountUpdater(prompt_len, decode_count)]
                    ),
                )
                active_mode[0] = None

                n_decoded = generated.shape[1] - prompt_len
                if args.decode_position == "last":
                    for key, buf in last_buffers.items():
                        if buf:
                            add_energy(stats, key, "decode", torch.cat(list(buf), dim=0))
                    n_used = min(n_decoded, args.decode_k)
                elif args.decode_position == "first":
                    n_used = min(n_decoded, args.decode_k)
                else:
                    n_used = n_decoded

                print(
                    f"prompt {pi + 1}/{len(prompts)}: prompt_len={prompt_len} "
                    f"decoded={n_decoded} decode_used={n_used}"
                )
    finally:
        active_mode[0] = None
        for h in handles:
            h.remove()

    rows = []
    print("\nmodule,prefill_energy,decode_energy,ratio,pre_tokens,decode_tokens")
    for key in stats:
        ep = energy_value(stats, key, "prefill")
        ed = energy_value(stats, key, "decode")
        ratio = None if ep is None or ed is None else ed / (ep + 1e-12)
        row = {
            "module": key,
            "prefill_energy": ep,
            "decode_energy": ed,
            "ratio": ratio,
            "prefill_tokens": stats[key]["prefill_tokens"],
            "decode_tokens": stats[key]["decode_tokens"],
        }
        rows.append(row)
        print(
            f"{key},{ep:.6g},{ed:.6g},{ratio:.6g},"
            f"{row['prefill_tokens']},{row['decode_tokens']}"
        )

    ratios = [r["ratio"] for r in rows if r["ratio"] is not None]
    if ratios:
        print("\nSummary")
        print(f"modules={len(ratios)}")
        print(f"mean_ratio={statistics.mean(ratios):.6g}")
        print(f"median_ratio={statistics.median(ratios):.6g}")
        print(f"min_ratio={min(ratios):.6g}")
        print(f"max_ratio={max(ratios):.6g}")
        print(f"ratio_lt_0.5={sum(r < 0.5 for r in ratios)}")
        print(f"ratio_gt_2={sum(r > 2.0 for r in ratios)}")
        print(f"ratio_lt_0.2={sum(r < 0.2 for r in ratios)}")
        print(f"ratio_gt_5={sum(r > 5.0 for r in ratios)}")

    if args.out_json:
        payload = {
            "model": args.model,
            "longwriter_jsonl": args.longwriter_jsonl,
            "seed": args.seed,
            "max_prompts": args.max_prompts,
            "max_new_tokens": args.max_new_tokens,
            "decode_position": args.decode_position,
            "decode_k": args.decode_k,
            "results": rows,
        }
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved JSON to {args.out_json}")


if __name__ == "__main__":
    main()
