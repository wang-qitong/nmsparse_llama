#!/usr/bin/env python3
"""
nmSPARSE 测试工具函数
封装了加载权重、剪枝、转换、测试等核心功能，用于批量测试
"""

import os
import sys
import time
import torch
import numpy as np
from typing import Dict, Optional, Tuple


def load_and_prune_weight(
    model,
    layer_idx: int,
    weight_name: str,
    sparsity: float,
    num_bank_val: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    """
    从模型中提取权重并进行bank-aware剪枝

    Returns:
        (real_weight, pruned_weight, pruned_mask, K, N)
        - real_weight: [K, N] 原始权重
        - pruned_weight: [K, N] 剪枝后权重
        - pruned_mask: [K, N] 剪枝mask
        - K: 输入维度
        - N: 输出维度
    """
    if weight_name == "q_proj":
        weight_module = model.model.layers[layer_idx].self_attn.q_proj
    elif weight_name == "k_proj":
        weight_module = model.model.layers[layer_idx].self_attn.k_proj
    elif weight_name == "v_proj":
        weight_module = model.model.layers[layer_idx].self_attn.v_proj
    elif weight_name == "o_proj":
        weight_module = model.model.layers[layer_idx].self_attn.o_proj
    elif weight_name == "gate_proj":
        weight_module = model.model.layers[layer_idx].mlp.gate_proj
    elif weight_name == "up_proj":
        weight_module = model.model.layers[layer_idx].mlp.up_proj
    elif weight_name == "down_proj":
        weight_module = model.model.layers[layer_idx].mlp.down_proj
    else:
        raise ValueError(f"未知的权重名称: {weight_name}")

    W = weight_module.weight.data.cpu().clone()
    real_weight = W.t().contiguous()

    K, N = real_weight.shape
    assert K % num_bank_val == 0, f"K={K} 必须是 NUM_BANK_VAL={num_bank_val} 的倍数"

    num_keep_per_bank = int(num_bank_val * (1 - sparsity))
    num_bank = K // num_bank_val
    real_weight_3d = real_weight.reshape(num_bank, num_bank_val, N)
    topk_idx = torch.topk(
        real_weight_3d.abs(),
        k=num_keep_per_bank,
        dim=1,
        largest=True,
        sorted=True,
    ).indices
    pruned_mask_3d = torch.zeros_like(real_weight_3d)
    pruned_mask_3d.scatter_(dim=1, index=topk_idx, value=1.0)
    pruned_mask = pruned_mask_3d.reshape(K, N)

    pruned_weight = real_weight * pruned_mask
    return real_weight, pruned_weight, pruned_mask, K, N


def convert_to_sparse_format(
    weight: torch.Tensor,
    mask: torch.Tensor,
    num_bank_val: int = 16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将剪枝后的权重转换为bank-aware稀疏格式

    Returns:
        (mat_data, mat_index)
        - mat_data: [w, N] 稀疏矩阵值（CPU）
        - mat_index: [w, N] 稀疏矩阵索引（CPU）
    """
    if weight.is_cuda:
        weight = weight.cpu()
    if mask.is_cuda:
        mask = mask.cpu()

    K, N = weight.shape
    num_banks = K // num_bank_val
    num_nonzeros_per_bank = int((mask[:num_bank_val, 0].sum().item()))
    w = num_banks * num_nonzeros_per_bank

    mat_data = torch.zeros(w, N, dtype=torch.float32, device="cpu")
    mat_index = torch.zeros(w, N, dtype=torch.int32, device="cpu")

    weight_3d = weight.reshape(num_banks, num_bank_val, N)
    mask_3d = mask.reshape(num_banks, num_bank_val, N)

    nnz_per_bank_col = mask_3d.sum(dim=1)
    if not torch.all(nnz_per_bank_col == num_nonzeros_per_bank):
        bad = torch.nonzero(nnz_per_bank_col != num_nonzeros_per_bank, as_tuple=False)[0]
        bank_id = int(bad[0].item())
        col_idx = int(bad[1].item())
        actual = int(nnz_per_bank_col[bank_id, col_idx].item())
        raise ValueError(
            f"Bank {bank_id} 列 {col_idx} 的非零元素个数不匹配！\n"
            f"  期望: {num_nonzeros_per_bank}\n"
            f"  实际: {actual}\n"
            f"  这表明剪枝过程没有保证每个 bank 有相同数量的非零元素。\n"
            f"  请检查 load_and_prune_weight() 函数的剪枝逻辑。"
        )

    nonzero_local_idx = torch.topk(
        mask_3d.to(torch.float32),
        k=num_nonzeros_per_bank,
        dim=1,
        largest=True,
        sorted=False,
    ).indices
    nonzero_local_idx = nonzero_local_idx.sort(dim=1).values

    selected_data = weight_3d.gather(dim=1, index=nonzero_local_idx)
    bank_offsets = (
        torch.arange(num_banks, device=weight_3d.device, dtype=nonzero_local_idx.dtype) * num_bank_val
    ).view(num_banks, 1, 1)
    selected_index = nonzero_local_idx + bank_offsets

    mat_data.copy_(selected_data.reshape(w, N).to(dtype=torch.float32, device="cpu"))
    mat_index.copy_(selected_index.reshape(w, N).to(dtype=torch.int32, device="cpu"))

    return mat_data, mat_index


def benchmark_sparse_kernel(
    nmsparse_module,
    vec_gpu: torch.Tensor,
    mat_data_gpu: torch.Tensor,
    mat_index_gpu: torch.Tensor,
    w: int,
    h: int,
    block_width: int,
    num_threads: int,
    vec_width: int,
    minibatch: int,
    vecNum: int,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> Tuple[float, torch.Tensor]:
    for _ in range(num_warmup):
        _ = nmsparse_module.forward(
            vec_gpu,
            mat_data_gpu,
            mat_index_gpu,
            w,
            h,
            block_width,
            num_threads,
            vec_width,
            minibatch,
            vecNum,
        )
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_iters):
        output_gpu = nmsparse_module.forward(
            vec_gpu,
            mat_data_gpu,
            mat_index_gpu,
            w,
            h,
            block_width,
            num_threads,
            vec_width,
            minibatch,
            vecNum,
        )

    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / num_iters
    return avg_time, output_gpu


def benchmark_dense_gemv(
    vec_gpu: torch.Tensor,
    weight_gpu: torch.Tensor,
    num_warmup: int = 10,
    num_iters: int = 100,
) -> Tuple[float, torch.Tensor]:
    for _ in range(num_warmup):
        _ = vec_gpu @ weight_gpu
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()
    start_event.record()

    for _ in range(num_iters):
        output = vec_gpu @ weight_gpu

    end_event.record()
    torch.cuda.synchronize()

    avg_time = start_event.elapsed_time(end_event) / num_iters
    return avg_time, output


def run_one_test(
    model,
    nmsparse_module,
    tokenizer,
    layer_idx: int,
    weight_name: str,
    sparsity: float = 0.5,
    use_real_activation: bool = True,
    prompt_text: str = "The future of artificial intelligence is",
    num_iters: int = 100,
    device: str = "cuda",
    num_bank_val: int = 16,
    num_threads: int = 128,
    verbose: bool = False,
) -> Dict:
    result = {
        "layer_idx": layer_idx,
        "weight_name": weight_name,
        "sparsity": sparsity,
        "success": False,
        "error_msg": None,
    }

    try:
        real_weight, pruned_weight, pruned_mask, K, N = load_and_prune_weight(
            model, layer_idx, weight_name, sparsity, num_bank_val
        )

        if use_real_activation:
            vec_gpu = torch.randn(1, K, dtype=torch.float32, device=device)
        else:
            vec_gpu = torch.randn(1, K, dtype=torch.float32, device=device)

        mat_data_cpu, mat_index_cpu = convert_to_sparse_format(pruned_weight, pruned_mask, num_bank_val)
        mat_data_gpu = mat_data_cpu.to(device)
        mat_index_gpu = mat_index_cpu.to(device)

        num_bank = K // num_bank_val
        w = int(K * (1 - sparsity))
        h = N
        block_width = w // num_bank
        vec_width = K * block_width // w
        minibatch = 1
        vecNum = K

        sparse_time, sparse_output = benchmark_sparse_kernel(
            nmsparse_module,
            vec_gpu,
            mat_data_gpu,
            mat_index_gpu,
            w,
            h,
            block_width,
            num_threads,
            vec_width,
            minibatch,
            vecNum,
            num_warmup=10,
            num_iters=num_iters,
        )

        pruned_weight_gpu = pruned_weight.to(device)
        dense_time_pruned, dense_output_pruned = benchmark_dense_gemv(
            vec_gpu, pruned_weight_gpu, num_warmup=10, num_iters=num_iters
        )

        max_error = (sparse_output - dense_output_pruned).abs().max().item()

        result.update(
            {
                "shape_K": K,
                "shape_N": N,
                "sparse_time_ms": sparse_time,
                "dense_time_pruned_ms": dense_time_pruned,
                "speedup_vs_pruned": dense_time_pruned / sparse_time,
                "max_error": max_error,
                "success": True,
            }
        )

        del vec_gpu, mat_data_gpu, mat_index_gpu
        del pruned_weight_gpu
        del sparse_output, dense_output_pruned
        torch.cuda.empty_cache()

    except Exception as e:
        result["success"] = False
        result["error_msg"] = str(e)

    return result
