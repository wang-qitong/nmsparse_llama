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

if [[ -z "${CUDA_HOME:-}" ]]; then
  export CUDA_HOME=/usr/local/cuda-12.4
fi
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

export E2E_MODE="$MODE"
export E2E_OOM_SWEEP=1
export E2E_SWEEP_REPEATS="$REPEATS"
export E2E_SWEEP_MAX_STEPS="$MAX_STEPS"
export E2E_SWEEP_STEP="$STEP"
export E2E_SWEEP_KVLEN_START="$KVLEN_START"
export E2E_SWEEP_KVLEN_END="$KVLEN_END"
export E2E_SWEEP_KVLEN_STEP="$KVLEN_STEP"
export E2E_SWEEP_PRINT_ALL=1

python /wangqitong/32_64/e2e_decode_modular.py
