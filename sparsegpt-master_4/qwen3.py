import copy, time, os, json
import torch
import torch.nn as nn
from sparsegpt import *
from modelutils import *
from quant import *

try:
    import wandb; has_wandb = True
except: has_wandb = False


def get_qwen3(model_path):
    def skip(*a, **k): pass
    torch.nn.init.kaiming_uniform_ = skip
    torch.nn.init.uniform_ = skip
    torch.nn.init.normal_ = skip
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.seqlen = 2048
    return model


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
    if rotary is not None and 'position_embeddings' in kw:
        pos_ids = kw.get('position_ids')
        if isinstance(pos_ids, torch.Tensor):
            pos_ids = pos_ids.to(inp.device)
        kw['position_embeddings'] = rotary(inp, pos_ids)
    out = layer(inp, **kw)
    return out[0] if isinstance(out, (tuple, list)) else out


def _default_base_file(save, save_mask, suffix):
    base = save or save_mask
    if not base: return ""
    root, ext = os.path.splitext(base)
    return (root if ext else base.rstrip("/")) + suffix


def _parse_int_csv(csv_text):
    values = set()
    for part in str(csv_text).split(","):
        part = part.strip()
        if part: values.add(int(part))
    return values


def _parse_proj_csv(csv_text):
    values = set()
    for part in str(csv_text).split(","):
        part = part.strip()
        if part: values.add(part)
    return values


@torch.no_grad()
def export_weight_masks(model):
    masks = {}
    for li, layer in enumerate(model.model.layers):
        for name, mod in find_layers(layer).items():
            key = f"model.layers.{li}.{name}.weight"
            masks[key] = (mod.weight.data != 0).to(torch.uint8).cpu()
    return masks


@torch.no_grad()
def qwen3_sequential(model, dataloader, dev):
    print("Starting...")
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    hs = model.config.hidden_size
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if getattr(model.model, 'norm', None) is not None:
        model.model.norm = model.model.norm.to(dev)
    if getattr(model.model, 'rotary_emb', None) is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((args.nsamples, model.seqlen, hs), dtype=dtype, device=dev)
    cache = {"i": 0, "kw": {}}

    class Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, **kw):
            inps[cache["i"]] = inp; cache["i"] += 1; cache["kw"] = kw; raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try: model(batch[0].to(dev))
        except ValueError: pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    if getattr(model.model, 'norm', None) is not None:
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    kw = _to_dev(cache["kw"], dev)
    rotary = getattr(model.model, 'rotary_emb', None)

    print("Ready.")
    artifacts = {
        "model": args.model, "dataset": args.dataset, "seed": args.seed,
        "nsamples": args.nsamples, "seqlen": model.seqlen,
        "prunen": args.prunen, "prunem": args.prunem,
        "sparsity": args.sparsity, "percdamp": args.percdamp, "blocksize": args.blocksize,
        "mask_semantics": "1=kept,0=pruned",
        "masks": {}, "hessian_diag": {}, "hessian_nsamples": {}, "hessian_full": {},
    }

    for i in range(len(layers)):
        layer = layers[i].to(dev)
        full = find_layers(layer)
        gpts = {}
        for name in full:
            if (not (args.minlayer <= i < args.maxlayer and args.prune_only in name)) == (not args.invert):
                continue
            gpts[name] = SparseGPT(full[name])
            if args.wbits < 16:
                gpts[name].quantizer = Quantizer()
                gpts[name].quantizer.configure(args.wbits, perchannel=True, sym=False, mse=False)

        def add_batch(n):
            def tmp(_, inp, out): gpts[n].add_batch(inp[0].data, out.data)
            return tmp

        handles = [full[n].register_forward_hook(add_batch(n)) for n in full]
        for j in range(args.nsamples):
            outs[j] = _fwd(layer, inps[j].unsqueeze(0), kw, rotary)
        for h in handles: h.remove()

        for name in gpts:
            print(i, name, "Pruning...")
            res = gpts[name].fasterprune(args.sparsity, prunen=args.prunen, prunem=args.prunem,
                                         percdamp=args.percdamp, blocksize=args.blocksize)
            key = f"model.layers.{i}.{name}.weight"
            artifacts["masks"][key] = res["mask"].to(torch.uint8).cpu()
            artifacts["hessian_diag"][key] = res["hessian_diag"].cpu()
            artifacts["hessian_nsamples"][key] = int(res["nsamples"])
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = _fwd(layer, inps[j].unsqueeze(0), kw, rotary)

        layers[i] = layer.cpu()
        del layer, gpts
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    return {}, artifacts


@torch.no_grad()
def qwen3_eval(model, testenc, dev, dataset, log_wandb=False):
    print("Evaluating...")
    testenc = testenc.input_ids
    nsamples = testenc.numel() // model.seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False

    layers = model.model.layers
    hs = model.config.hidden_size
    model.model.embed_tokens = model.model.embed_tokens.to(dev)
    if getattr(model.model, 'rotary_emb', None) is not None:
        model.model.rotary_emb = model.model.rotary_emb.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((nsamples, model.seqlen, hs), dtype=dtype, device=dev)
    cache = {"i": 0, "kw": {}}

    class Catcher(nn.Module):
        def __init__(self, m): super().__init__(); self.module = m
        def forward(self, inp, **kw):
            inps[cache["i"]] = inp; cache["i"] += 1; cache["kw"] = kw; raise ValueError

    layers[0] = Catcher(layers[0])
    for i in range(nsamples):
        try: model(testenc[:, i*model.seqlen:(i+1)*model.seqlen].to(dev))
        except ValueError: pass
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    kw = _to_dev(cache["kw"], dev)
    rotary = getattr(model.model, 'rotary_emb', None)

    for i in range(len(layers)):
        print(i)
        layer = layers[i].to(dev)
        if args.gmp:
            for _, mod in find_layers(layer).items():
                W = mod.weight.data
                W.data[torch.abs(W.data) <= torch.sort(torch.abs(W.flatten()))[0][int(W.numel()*args.sparsity)]] = 0
        for j in range(nsamples):
            outs[j] = _fwd(layer, inps[j].unsqueeze(0), kw, rotary)
        layers[i] = layer.cpu(); del layer
        torch.cuda.empty_cache()
        inps, outs = outs, inps

    if getattr(model.model, 'norm', None) is not None:
        model.model.norm = model.model.norm.to(dev)
    model.lm_head = model.lm_head.to(dev)
    testenc = testenc.to(dev)
    nlls = []
    for i in range(nsamples):
        h = inps[i].unsqueeze(0)
        if getattr(model.model, 'norm', None) is not None:
            h = model.model.norm(h)
        logits = model.lm_head(h)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = testenc[:, i*model.seqlen:(i+1)*model.seqlen][:, 1:]
        loss = nn.CrossEntropyLoss()(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        nlls.append(loss.float() * model.seqlen)
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
    print(f"Perplexity: {ppl.item():.3f}")
    if log_wandb: wandb.log({f"{dataset}/perplexity": ppl.item()})
    model.config.use_cache = use_cache


if __name__ == "__main__":
    import argparse
    from datautils import *

    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=str)
    parser.add_argument("dataset", type=str, choices=["wikitext2", "ptb", "c4"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nsamples", type=int, default=128)
    parser.add_argument("--percdamp", type=float, default=0.01)
    parser.add_argument("--sparsity", type=float, default=0)
    parser.add_argument("--prunen", type=int, default=0)
    parser.add_argument("--prunem", type=int, default=0)
    parser.add_argument("--blocksize", type=int, default=128)
    parser.add_argument("--gmp", action="store_true")
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
    parser.add_argument("--skip_eval", action="store_true")
    parser.add_argument("--log_wandb", action="store_true")
    args = parser.parse_args()

    if not args.baseline_config_file:
        args.baseline_config_file = _default_base_file(args.save, args.save_mask, "_baseline_config.json")
    if not args.baseline_artifact_file:
        args.baseline_artifact_file = _default_base_file(args.save, args.save_mask, "_baseline_artifacts.pt")

    if args.log_wandb:
        assert has_wandb; wandb.init(config=args)

    DEV = torch.device("cuda:0")

    model = get_qwen3(args.model)
    model.eval()

    dataloader, testloader = get_loaders(
        args.dataset, nsamples=args.nsamples, seed=args.seed, model=args.model, seqlen=model.seqlen
    )

    if (args.sparsity or args.prunen) and not args.gmp:
        tick = time.time()
        _, artifacts = qwen3_sequential(model, dataloader, DEV)
        print(time.time() - tick)
        if not args.save_hessian_diag:
            artifacts["hessian_diag"] = {}
        if args.baseline_artifact_file:
            os.makedirs(os.path.dirname(args.baseline_artifact_file) or ".", exist_ok=True)
            torch.save(artifacts, args.baseline_artifact_file)
            print(f"Saved artifacts to {args.baseline_artifact_file}")
        if args.baseline_config_file:
            cfg_out = {k: v for k, v in vars(args).items()}
            cfg_out["seqlen"] = model.seqlen
            os.makedirs(os.path.dirname(args.baseline_config_file) or ".", exist_ok=True)
            with open(args.baseline_config_file, "w") as f:
                json.dump(cfg_out, f, indent=2, default=str)
            print(f"Saved config to {args.baseline_config_file}")

    if args.save_mask:
        mask_state = {"model": args.model, "prunen": args.prunen, "prunem": args.prunem,
                      "sparsity": args.sparsity, "mask_dtype": "uint8",
                      "mask_semantics": "1=kept,0=pruned", "masks": export_weight_masks(model)}
        os.makedirs(os.path.dirname(args.save_mask) or ".", exist_ok=True)
        torch.save(mask_state, args.save_mask)
        print(f"Saved masks to {args.save_mask}")

    if not args.skip_eval:
        for ds in ["wikitext2", "ptb", "c4"]:
            try:
                _, testloader = get_loaders(ds, seed=args.seed, model=args.model, seqlen=model.seqlen)
            except Exception as e:
                print(f"[WARN] Skip eval on {ds}: {e}"); continue
            print("Dataset:", ds)
            qwen3_eval(model, testloader, DEV, ds, args.log_wandb)

    if args.save:
        gc = getattr(model, "generation_config", None)
        if gc is not None and not getattr(gc, "do_sample", True):
            if getattr(gc, "temperature", None) is not None or getattr(gc, "top_p", None) is not None:
                model.generation_config = copy.deepcopy(gc)
                model.generation_config.temperature = None
                model.generation_config.top_p = None
        model.save_pretrained(args.save)
        print(f"Saved pruned model to {args.save}")
