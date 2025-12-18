# nmSPARSE SpMV and LLaMA Decode Scripts

This repository contains several standalone scripts for experimenting with
**bank-aware structured sparse SpMV kernels** and their usage in **LLaMA decoding**.
The CUDA kernel implementation is adapted from **nmSPARSE**.

## Files

- **nmsparse_wrapper_with_initialdata.cu**  
  CUDA extension that exposes a bank-aware sparse SpMV kernel to PyTorch.

- **nmsparse_test_utils.py**  
  Helper functions for structured pruning and converting dense weights
  into the sparse format required by the kernel.

- **test_nmsparse.py**  
  Kernel-level benchmark for sparse SpMV, supporting random data or real
  LLaMA projection weights.

- **e2e_decode_modular.py**  
  End-to-end single-token decode benchmark for LLaMA, comparing sparse
  execution with dense baselines.
