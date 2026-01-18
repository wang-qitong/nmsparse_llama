#!/usr/bin/env python3

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nmsparse_test_utils import load_and_prune_weight, convert_to_sparse_format


LLAMA_MODEL_PATH = "/wangqitong/llama_2-7b"
DEVICE = "cuda"
DTYPE = torch.float32

SPARSITY = 0.5
NUM_BANK_VAL = 32
NUM_THREADS = 128

PRUNE_METHOD = os.getenv("PRUNE_METHOD", "magnitude").strip().lower()
if PRUNE_METHOD not in ["magnitude", "sparsegpt"]:
    raise ValueError(f"invalid PRUNE_METHOD={PRUNE_METHOD!r}, must be one of: magnitude, sparsegpt")

def get_projection_module(layer, proj_name):
    if proj_name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        return getattr(layer.self_attn, proj_name)
    if proj_name in ["gate_proj", "up_proj", "down_proj"]:
        return getattr(layer.mlp, proj_name)
    raise ValueError(f"unknown proj_name={proj_name}")


class SparsifiedLinear:
    def __init__(self, original_module, nmsparse_module, sparse_param):
        self.original_module = original_module
        self.nmsparse_module = nmsparse_module
        self.mat_data = sparse_param["mat_data"]
        self.mat_index = sparse_param["mat_index"]
        self.w = sparse_param["w"]
        self.h = sparse_param["h"]
        self.block_width = sparse_param["block_width"]
        self.vec_width = sparse_param["vec_width"]
        self.bias = original_module.bias

    def __call__(self, x):
        x_2d = x.reshape(-1, x.shape[-1])
        minibatch = x_2d.shape[0]
        vec_num = x_2d.shape[1]
        out = self.nmsparse_module.forward(
            x_2d,
            self.mat_data,
            self.mat_index,
            self.w,
            self.h,
            self.block_width,
            NUM_THREADS,
            self.vec_width,
            minibatch,
            vec_num,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.view(*x.shape[:-1], -1)


def apply_pruned_weights_to_model(model, sparse_params, proj_names):
    with torch.no_grad():
        for layer_idx in range(len(model.model.layers)):
            layer = model.model.layers[layer_idx]
            for proj_name in proj_names:
                proj_module = get_projection_module(layer, proj_name)
                pruned_w = sparse_params[layer_idx][proj_name]["pruned_weight"]
                proj_module.weight.copy_(pruned_w.t().contiguous().to(proj_module.weight.dtype))


def apply_original_weights_to_model(model, sparse_params, proj_names):
    with torch.no_grad():
        for layer_idx in range(len(model.model.layers)):
            layer = model.model.layers[layer_idx]
            for proj_name in proj_names:
                proj_module = get_projection_module(layer, proj_name)
                original_w = sparse_params[layer_idx][proj_name]["original_weight"]
                proj_module.weight.copy_(original_w.t().contiguous().to(proj_module.weight.dtype))


def restore_dense_forwards(model, original_forwards, original_block_forwards, proj_names):
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        layer.self_attn.forward = original_block_forwards[layer_idx]["self_attn"]
        layer.mlp.forward = original_block_forwards[layer_idx]["mlp"]
        for proj_name in proj_names:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = original_forwards[layer_idx][proj_name]


def apply_sparse_forwards(model, nmsparse_module, sparse_params, proj_names):
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        for proj_name in proj_names:
            proj_module = get_projection_module(layer, proj_name)
            proj_module.forward = SparsifiedLinear(proj_module, nmsparse_module, sparse_params[layer_idx][proj_name])


def load_nmsparse_module(script_dir: str):
    wrapper_cu_path = os.path.join(script_dir, "nmsparse_wrapper_with_initialdata.cu")
    if not os.path.exists(wrapper_cu_path):
        raise FileNotFoundError(wrapper_cu_path)

    conda_prefix = os.environ.get("CONDA_PREFIX", "")
    cuda_include_path = f"{conda_prefix}/targets/x86_64-linux/include"

    return load(
        name="nmsparse_fig9",
        sources=[wrapper_cu_path],
        extra_cuda_cflags=["-O3", "--use_fast_math", f"-I{cuda_include_path}"],
        verbose=True,
    )


def compute_ppl(model, input_ids_1d: torch.Tensor, block_size: int, max_blocks: int | None):
    total_nll = 0.0
    total_tokens = 0

    num_tokens = int(input_ids_1d.numel())
    if num_tokens < 2:
        raise ValueError("eval text too short after tokenization")

    blocks_done = 0
    for start in range(0, num_tokens - 1, block_size):
        if max_blocks is not None and blocks_done >= max_blocks:
            break

        end = min(start + block_size, num_tokens)
        chunk = input_ids_1d[start:end]
        if chunk.numel() < 2:
            break

        chunk = chunk.unsqueeze(0).to(DEVICE)
        labels = chunk.clone()

        with torch.no_grad():
            out = model(input_ids=chunk, labels=labels, use_cache=False, return_dict=True)
            loss = out.loss

        n_pred = int(chunk.numel() - 1)
        total_nll += float(loss.item()) * n_pred
        total_tokens += n_pred
        blocks_done += 1

    ppl = math.exp(total_nll / max(1, total_tokens))
    return {
        "loss": total_nll / max(1, total_tokens),
        "ppl": ppl,
        "tokens": total_tokens,
        "blocks": blocks_done,
    }


def compute_ppl_autoregressive(model, input_ids_1d: torch.Tensor, block_size: int, max_blocks: int | None):
    total_nll = 0.0
    total_tokens = 0

    num_tokens = int(input_ids_1d.numel())
    if num_tokens < 2:
        raise ValueError("eval text too short after tokenization")

    blocks_done = 0
    for start in range(0, num_tokens - 1, block_size):
        if max_blocks is not None and blocks_done >= max_blocks:
            break

        end = min(start + block_size, num_tokens)
        segment = input_ids_1d[start:end]
        if segment.numel() < 2:
            break

        past = None

        with torch.no_grad():
            for i in range(int(segment.numel()) - 1):
                cur_tok = segment[i : i + 1].unsqueeze(0).to(DEVICE)  # [1,1]
                tgt_tok = segment[i + 1].unsqueeze(0).to(DEVICE)      # [1]

                out = model(
                    input_ids=cur_tok,
                    past_key_values=past,
                    use_cache=True,
                    return_dict=True,
                )
                past = out.past_key_values

                logits = out.logits[:, -1, :]  # [1, vocab]
                loss = F.cross_entropy(logits, tgt_tok)
                total_nll += float(loss.item())
                total_tokens += 1

        blocks_done += 1

    ppl = math.exp(total_nll / max(1, total_tokens))
    return {
        "loss": total_nll / max(1, total_tokens),
        "ppl": ppl,
        "tokens": total_tokens,
        "blocks": blocks_done,
    }


def load_eval_text_from_file(eval_file: str) -> str:
    if not os.path.exists(eval_file):
        raise FileNotFoundError(
            f"eval_file not found: {eval_file} (prepare a local text file, e.g. a few 10k-100k chars)"
        )
    with open(eval_file, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_eval_text_from_wikitext2(split: str, hf_cache_dir: str | None, max_examples: int) -> str:

    local_path = "/wangqitong/wikitext-2-raw-v1"
    try:
        from datasets import load_dataset
    except Exception as e:
        raise RuntimeError(
            "Failed to import HuggingFace `datasets`. Install it via: pip install datasets\n"
            f"Original error: {e}"
        )

    cache_dir = hf_cache_dir if (hf_cache_dir is not None and hf_cache_dir != "") else None
    try:
        if os.path.exists(local_path):
            # WikiText 通常下载下来是 .parquet 或 .txt 文件
            # 这里指向具体的 subset 文件夹下的文件
            data_files = {
                "test": f"{local_path}/test-00000-of-00001.parquet",
                "train": f"{local_path}/train-00000-of-00001.parquet",
                "validation": f"{local_path}/validation-00000-of-00001.parquet",
            }
            ds = load_dataset("parquet", data_files=data_files, split=split)
        else:
            # 如果本地没找到，再尝试官方在线加载
            ds = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split=split,
                cache_dir=cache_dir,
            )
    except Exception as e:
        raise RuntimeError(
            "Failed to load WikiText2 from HuggingFace. This may require internet access on first run, "
            "or a pre-populated HF cache.\n"
            f"Original error: {e}"
        )

    texts = ds["text"]
    if max_examples > 0:
        texts = texts[:max_examples]

    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    return "\n\n".join(texts)


def load_or_build_sparse_params(model, cache_file: str, proj_names, *, build_if_missing: bool = True):
    if os.path.exists(cache_file):
        sparse_params = torch.load(cache_file, map_location="cpu")
    else:
        if not build_if_missing:
            raise FileNotFoundError(
                f"sparse cache not found: {cache_file}\n"
                "PRUNE_METHOD=sparsegpt requires a pre-generated cache. "
                "Run 16_32A800/prune_sparsegpt_cache.py first."
            )
        sparse_params = {}
        num_layers = len(model.model.layers)
        for layer_idx in range(num_layers):
            sparse_params[layer_idx] = {}
            for proj_name in proj_names:
                real_weight, pruned_weight, pruned_mask, K, N = load_and_prune_weight(
                    model, layer_idx, proj_name, SPARSITY, NUM_BANK_VAL
                )
                mat_data_cpu, mat_index_cpu = convert_to_sparse_format(pruned_weight, pruned_mask, NUM_BANK_VAL)
                num_bank = K // NUM_BANK_VAL
                num_nonzeros_per_bank = int(NUM_BANK_VAL * (1 - SPARSITY))
                w = num_bank * num_nonzeros_per_bank
                h = N
                sparse_params[layer_idx][proj_name] = {
                    "mat_data": mat_data_cpu,
                    "mat_index": mat_index_cpu,
                    "pruned_weight": pruned_weight,
                    "original_weight": real_weight,
                    "w": w,
                    "h": h,
                    "block_width": num_nonzeros_per_bank,
                    "vec_width": NUM_BANK_VAL,
                    "K": K,
                    "N": N,
                }
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        torch.save(sparse_params, cache_file)

    num_layers = len(model.model.layers)
    for layer_idx in range(num_layers):
        layer = model.model.layers[layer_idx]
        for proj_name in proj_names:
            entry = sparse_params[layer_idx][proj_name]
            if "original_weight" not in entry:
                proj_module = get_projection_module(layer, proj_name)
                entry["original_weight"] = proj_module.weight.data.detach().to("cpu").t().contiguous()

            K = int(entry["K"])
            N = int(entry["N"])
            num_bank = K // NUM_BANK_VAL
            num_nonzeros_per_bank = int(NUM_BANK_VAL * (1 - SPARSITY))
            entry["w"] = num_bank * num_nonzeros_per_bank
            entry["h"] = N
            entry["block_width"] = num_nonzeros_per_bank
            entry["vec_width"] = NUM_BANK_VAL

    for layer_idx in range(num_layers):
        for proj_name in proj_names:
            entry = sparse_params[layer_idx][proj_name]
            entry["mat_data"] = entry["mat_data"].to(DEVICE)
            entry["mat_index"] = entry["mat_index"].to(DEVICE)
            entry["pruned_weight"] = entry["pruned_weight"].to(DEVICE)

    return sparse_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="file", choices=["file", "wikitext2"])
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--eval_file", type=str, default=os.path.join(os.path.dirname(__file__), "eval.txt"))
    parser.add_argument("--block_size", type=int, default=1024)
    parser.add_argument("--max_blocks", type=int, default=0)
    parser.add_argument("--eval_style", type=str, default="teacher_forcing", choices=["teacher_forcing", "autoregressive"])
    parser.add_argument("--modes", type=str, default="dense_original,dense_pruned,sparse")
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    model = LlamaForCausalLM.from_pretrained(
        LLAMA_MODEL_PATH,
        torch_dtype=DTYPE,
        device_map=DEVICE,
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(LLAMA_MODEL_PATH)

    if args.dataset == "file":
        text = load_eval_text_from_file(args.eval_file)
    elif args.dataset == "wikitext2":
        text = load_eval_text_from_wikitext2(
            split=args.split,
            hf_cache_dir=args.hf_cache_dir,
            max_examples=int(args.max_examples),
        )
    else:
        raise ValueError(f"unknown dataset={args.dataset}")

    enc = tokenizer(text, return_tensors="pt")
    input_ids_1d = enc["input_ids"][0].to("cpu")

    proj_names = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if PRUNE_METHOD == "magnitude":
        pruned_cache_file = "/wangqitong/sparse_params_sparsity0.5_bank32_mlp_qkvo_16_32.pt"
        build_if_missing = True
    else:
        pruned_cache_file = "/wangqitong/sparse_params_sparsity0.5_bank32_mlp_qkvo_16_32_sparsegpt.pt"
        build_if_missing = False

    print(f"[PruneMethod] PRUNE_METHOD={PRUNE_METHOD} cache={pruned_cache_file}")
    sparse_params = load_or_build_sparse_params(model, pruned_cache_file, proj_names, build_if_missing=build_if_missing)

    _e = sparse_params[0]["q_proj"]
    _pw = _e["pruned_weight"].detach().to("cpu")
    _K, _N = _pw.shape
    _nnz = int((_pw.abs() > 0).sum(dim=0)[0].item())
    print(
        f"[CacheCheck] NUM_BANK_VAL={NUM_BANK_VAL} cache={pruned_cache_file} layer0.q_proj K={_K} N={_N} "
        f"vec_width={int(_e['vec_width'])} block_width={int(_e['block_width'])} nnz_per_col={_nnz}"
    )

    original_forwards = {}
    original_block_forwards = {}
    for layer_idx in range(len(model.model.layers)):
        layer = model.model.layers[layer_idx]
        original_block_forwards[layer_idx] = {
            "self_attn": layer.self_attn.forward,
            "mlp": layer.mlp.forward,
        }
        original_forwards[layer_idx] = {}
        for proj_name in proj_names:
            proj_module = get_projection_module(layer, proj_name)
            original_forwards[layer_idx][proj_name] = proj_module.forward

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    max_blocks = None if args.max_blocks <= 0 else int(args.max_blocks)

    nmsparse_module = None
    for mode in modes:
        restore_dense_forwards(model, original_forwards, original_block_forwards, proj_names)

        if mode == "dense_original":
            apply_original_weights_to_model(model, sparse_params, proj_names)
        elif mode in ["dense_pruned", "sparse"]:
            apply_pruned_weights_to_model(model, sparse_params, proj_names)
        else:
            raise ValueError(f"unknown mode={mode}")

        if mode == "sparse":
            if nmsparse_module is None:
                nmsparse_module = load_nmsparse_module(os.path.dirname(os.path.abspath(__file__)))
            apply_sparse_forwards(model, nmsparse_module, sparse_params, proj_names)

        torch.cuda.synchronize()
        eval_style = args.eval_style
        if mode == "sparse" and eval_style == "teacher_forcing":
            eval_style = "autoregressive"

        if eval_style == "teacher_forcing":
            stats = compute_ppl(model, input_ids_1d, block_size=args.block_size, max_blocks=max_blocks)
        else:
            stats = compute_ppl_autoregressive(model, input_ids_1d, block_size=args.block_size, max_blocks=max_blocks)
        print(
            f"[PPL] mode={mode} style={eval_style} block_size={args.block_size} blocks={stats['blocks']} tokens={stats['tokens']} "
            f"loss={stats['loss']:.6f} ppl={stats['ppl']:.4f}"
        )
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
