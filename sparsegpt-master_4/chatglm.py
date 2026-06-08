import os
import time

import torch
import torch.nn as nn

from sparsegpt import *
from modelutils import *
from quant import *


def get_chatglm(model_name_or_path):
    import torch

    def skip(*args, **kwargs):
        pass

    # avoid re-init overhead when loading HF checkpoints
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    # ChatGLM config seq_length can be huge (e.g. 1M); SparseGPT calibration keeps 2K.
    model.seqlen = min(2048, int(getattr(model.config, "seq_length", 2048)))
    return model


@torch.no_grad()
def export_weight_masks(model):
    """Export binary masks (1=kept, 0=pruned) for all prunable linear weights."""
    masks = {}
    for layer_idx, layer in enumerate(model.transformer.encoder.layers):
        full = find_layers(layer)
        for name, mod in full.items():
            key = f"transformer.encoder.layers.{layer_idx}.{name}.weight"
            masks[key] = (mod.weight.data != 0).to(torch.uint8).cpu()
    return masks


@torch.no_grad()
def chatglm_sequential(model, dataloader, dev):
    print("Starting...")

    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.transformer.encoder.layers

    model.transformer.embedding = model.transformer.embedding.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None, "rotary_pos_emb": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(
            self,
            hidden_states,
            attention_mask,
            rotary_pos_emb,
            kv_cache=None,
            use_cache=True,
        ):
            inps[cache["i"]] = hidden_states
            cache["i"] += 1
            cache["attention_mask"] = attention_mask
            cache["rotary_pos_emb"] = rotary_pos_emb
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev), use_cache=False)
        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.transformer.embedding = model.transformer.embedding.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]
    rotary_pos_emb = cache["rotary_pos_emb"]
    if isinstance(attention_mask, torch.Tensor):
        attention_mask = attention_mask.to(dev)
    if isinstance(rotary_pos_emb, torch.Tensor):
        rotary_pos_emb = rotary_pos_emb.to(dev)

    print("Ready.")

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)

        if args.true_sequential:
            sequential = [
                ["self_attention.query_key_value"],
                ["self_attention.dense"],
                ["mlp.dense_h_to_4h"],
                ["mlp.dense_4h_to_h"],
            ]
        else:
            sequential = [list(full.keys())]

        for names in sequential:
            subset = {n: full[n] for n in names if n in full}

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
                        args.wbits,
                        perchannel=True,
                        sym=False,
                        mse=False,
                    )

            def add_batch(name):
                def tmp(_, inp, out):
                    gpts[name].add_batch(inp[0].data, out.data)

                return tmp

            handles = []
            for name in subset:
                if name in gpts:
                    handles.append(subset[name].register_forward_hook(add_batch(name)))

            for j in range(args.nsamples):
                out = layer(
                    inps[j].unsqueeze(0),
                    attention_mask,
                    rotary_pos_emb,
                    kv_cache=None,
                    use_cache=False,
                )
                if isinstance(out, (tuple, list)):
                    out = out[0]
                outs[j] = out

            for h in handles:
                h.remove()

            for name in subset:
                if name not in gpts:
                    continue
                print(i, name)
                print("Pruning ...")
                gpts[name].fasterprune(
                    args.sparsity,
                    prunen=args.prunen,
                    prunem=args.prunem,
                    percdamp=args.percdamp,
                    blocksize=args.blocksize,
                )
                gpts[name].free()

        for j in range(args.nsamples):
            out = layer(
                inps[j].unsqueeze(0),
                attention_mask,
                rotary_pos_emb,
                kv_cache=None,
                use_cache=False,
            )
            if isinstance(out, (tuple, list)):
                out = out[0]
            outs[j] = out

        layers[i] = layer.cpu()
        del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import get_loaders

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str, help="ChatGLM model to load")
    parser.add_argument(
        "dataset",
        type=str,
        choices=["wikitext2", "ptb", "c4"],
        help="Where to extract calibration data from.",
    )
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
    parser.add_argument("--true-sequential", action="store_true")
    parser.add_argument("--save", type=str, default="")
    parser.add_argument("--save_mask", type=str, default="")
    parser.add_argument("--skip_eval", action="store_true")

    args = parser.parse_args()

    model = get_chatglm(args.model)
    model.eval()

    dataloader, _ = get_loaders(
        args.dataset,
        nsamples=args.nsamples,
        seed=args.seed,
        model=args.model,
        seqlen=model.seqlen,
    )

    if (args.sparsity or args.prunen):
        tick = time.time()
        chatglm_sequential(model, dataloader, DEV)
        print("Pruning finished in", time.time() - tick, "seconds")

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

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        model.save_pretrained(args.save)
        print(f"Saved pruned model to {args.save}")
