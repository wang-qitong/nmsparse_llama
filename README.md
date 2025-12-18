# nmSPARSE-based SpMV and LLaMA Decode Benchmark

This repository provides a **self-contained experimental framework** for evaluating  
**bank-aware structured sparse SpMV kernels** (implemented with CUDA) and their impact on  
**LLaMA-2 decoding performance**, at both:

- **kernel level (GEMV / SpMV micro-benchmarks)**  
- **end-to-end single-token decode latency**

The sparse CUDA kernel used in this repository is adapted from **nmSPARSE**, but the codebase
is organized and evaluated **independently**, without relying on external benchmark scripts.

---

## Repository Structure

.
├── nmsparse_wrapper_with_initialdata.cu # CUDA wrapper for bank-aware sparse SpMV
├── nmsparse_test_utils.py # Pruning & sparse-format utility functions
├── test_nmsparse.py # Kernel-level SpMV benchmark
└── e2e_decode_modular.py # End-to-end LLaMA decode benchmark


---

## 1. CUDA Kernel: Bank-Aware Sparse SpMV

### `nmsparse_wrapper_with_initialdata.cu`

This file contains a **CUDA extension** that exposes a bank-aware sparse SpMV kernel
to PyTorch via `torch.utils.cpp_extension.load`.

Key characteristics:

- Bank-aware structured sparsity (default: 16:32)
- Explicit sparse value (`mat_data`) and index (`mat_index`) representation
- Designed for **GEMV / small-N SpMM** workloads
- Adapted from the nmSPARSE kernel design

The kernel is compiled **just-in-time** when running the Python scripts.

---

## 2. Utility Functions

### `nmsparse_test_utils.py`

This file provides reusable utilities for:

- Loading LLaMA projection weights
- Applying magnitude-based structured pruning (bank granularity = 32)
- Converting pruned weights into bank-aware sparse format:
  - `mat_data`
  - `mat_index`
- Ensuring index validity and consistency with kernel assumptions

These utilities are shared by both the micro-benchmark and the end-to-end benchmark.

---

## 3. Kernel-Level Benchmark

### `test_nmsparse.py`

This script evaluates **pure kernel performance** of the sparse SpMV implementation.

### What it measures

- Sparse SpMV kernel latency (CUDA events)
- Dense GEMV latency (pruned vs original weights)
- Speedup ratios
- Numerical correctness against dense reference

### Supported modes

- Random synthetic data
- **Real LLaMA-2 projection weights**
- Optional **real decoding activation** (captured from model forward)

### Typical use


bash python test_nmsparse.py

Optional environment variables:

export WEIGHT_NAME=o_proj        # q_proj, k_proj, v_proj, o_proj,
                                # gate_proj, up_proj, down_proj
export PROMPT_TOKEN_LEN=512     # force fixed token length for activation capture


This script is intended to answer:

Which matrix shapes benefit most from bank-aware sparse SpMV?
