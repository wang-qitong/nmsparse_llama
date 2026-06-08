"""
Apply 2:4 masks from masks.pt to the original dense model and save as HF format.

Usage:
  python apply_mask_and_save.py \
    --base_model /wangqitong/llama3.1-8b-instruct \
    --masks /wangqitong/llama3.1-8b-instruct-sparsegpt-2of4-ar/masks.pt \
    --save /wangqitong/llama3.1-8b-instruct-sparsegpt-2of4-ar
"""
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--masks", type=str, required=True)
    parser.add_argument("--save", type=str, required=True)
    args = parser.parse_args()

    print(f"Loading masks from {args.masks} ...")
    mask_state = torch.load(args.masks, map_location="cpu")
    masks = mask_state["masks"]
    print(f"  {len(masks)} masks loaded (semantics: {mask_state.get('mask_semantics')})")

    print(f"Loading base model from {args.base_model} ...")
    model = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype="auto")

    applied = 0
    for name, param in model.named_parameters():
        key = name  # e.g. "model.layers.0.self_attn.q_proj.weight"
        if key in masks:
            mask = masks[key].to(param.device).to(param.dtype)
            param.data.mul_(mask)
            applied += 1

    print(f"Applied {applied}/{len(masks)} masks.")

    # Fix generation_config if needed
    import copy
    gc = getattr(model, "generation_config", None)
    if gc is not None and getattr(gc, "do_sample", True) is False:
        if getattr(gc, "temperature", None) is not None or getattr(gc, "top_p", None) is not None:
            model.generation_config = copy.deepcopy(gc)
            model.generation_config.temperature = None
            model.generation_config.top_p = None

    print(f"Saving pruned model to {args.save} ...")
    model.save_pretrained(args.save)

    # Also copy tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    tokenizer.save_pretrained(args.save)

    print("Done.")


if __name__ == "__main__":
    main()
