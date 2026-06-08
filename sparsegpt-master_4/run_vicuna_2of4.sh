#!/usr/bin/env bash
# 使用 SparseGPT 对 Vicuna-7B 做 2:4 剪枝，保存为 HF 格式
# 在 sparsegpt-master_2 目录下执行: bash run_vicuna_2of4.sh
#
# 依赖: 需能访问 lmsys/vicuna-7b-v1.5（或设置 VICUNA_PATH 为本地路径）
# 校准数据: 默认 c4；若无 c4 可改为 wikitext2：把下面 c4 改成 wikitext2

set -e
cd "$(dirname "$0")"

# 模型：HuggingFace 名或本地路径
VICUNA_PATH="${VICUNA_PATH:-/wangqitong/PMPD-main/downloads/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d}"
# 剪枝结果保存目录（HF 格式，可直接用于 from_pretrained）
SAVE_PATH="${SAVE_PATH:-/wangqitong/vicuna-7b-v1.5-sparsegpt-2of4}"
# 校准数据集: c4 | wikitext2 | ptb
DATASET="${DATASET:-c4}"
# 校准样本数
NSAMPLES="${NSAMPLES:-128}"
# 使用的 GPU
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export CUDA_VISIBLE_DEVICES

echo "=== SparseGPT 2:4 剪枝 Vicuna-7B ==="
echo "  model:   $VICUNA_PATH"
echo "  save:    $SAVE_PATH"
echo "  dataset: $DATASET  nsamples: $NSAMPLES"
echo ""

# 2:4 结构稀疏：--prunen 2 --prunem 4（每 4 个元素保留 2 个，50% 稀疏）
# --skip_eval 可省略 perplexity 评估以节省时间
python llama.py "$VICUNA_PATH" "$DATASET" \
  --prunen 2 \
  --prunem 4 \
  --nsamples "$NSAMPLES" \
  --save "$SAVE_PATH" \
  --skip_eval

echo ""
echo "Done. Pruned model saved to: $SAVE_PATH"
