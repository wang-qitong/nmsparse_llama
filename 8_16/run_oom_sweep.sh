#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-sparse}"              # sparse | dense_pruned | dense_original
GPU="${2:-0}"                   # CUDA_VISIBLE_DEVICES
REPEATS="${3:-20}"              # repeats per step
MAX_STEPS="${4:-4096}"          # maximum decode steps to try
STEP="${5:-20}"                # decode steps increment per sweep iteration

KVLEN_START="${6:-512}"         # sweep kvlen start
KVLEN_END="${7:-4096}"          # sweep kvlen end
KVLEN_STEP="${8:-128}"          # sweep kvlen step

export CUDA_VISIBLE_DEVICES="$GPU"

export PATH="/wangqitong/miniconda3/envs/myenv/bin:${PATH}"
export CONDA_PREFIX="/wangqitong/miniconda3/envs/myenv"

if [[ -z "${CUDA_HOME:-}" ]]; then
  if [[ -x /usr/local/cuda-12.4/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda-12.4
  elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
    export CUDA_HOME=/usr/local/cuda
  elif command -v nvcc >/dev/null 2>&1; then
    NVCC_PATH="$(command -v nvcc)"
    export CUDA_HOME="$(cd "$(dirname "$NVCC_PATH")/.." && pwd)"
  fi
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/bin" ]]; then
  export PATH="${CUDA_HOME}/bin:${PATH}"
fi
if [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib64" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
elif [[ -n "${CUDA_HOME:-}" && -d "${CUDA_HOME}/lib" ]]; then
  export LD_LIBRARY_PATH="${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
fi

export E2E_MODE="$MODE"
export E2E_OOM_SWEEP=1
export E2E_SWEEP_REPEATS="$REPEATS"
export E2E_SWEEP_MAX_STEPS="$MAX_STEPS"
export E2E_SWEEP_STEP="$STEP"
export E2E_SWEEP_KVLEN_START="$KVLEN_START"
export E2E_SWEEP_KVLEN_END="$KVLEN_END"
export E2E_SWEEP_KVLEN_STEP="$KVLEN_STEP"
export E2E_SWEEP_PRINT_ALL=1

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" && -x /wangqitong/miniconda3/envs/myenv/bin/python ]]; then
  PYTHON_BIN=/wangqitong/miniconda3/envs/myenv/bin/python
fi
if [[ -z "${PYTHON_BIN}" ]]; then
  PYTHON_BIN=python3
fi

"${PYTHON_BIN}" /wangqitong/8_16/e2e_decode_modular.py
