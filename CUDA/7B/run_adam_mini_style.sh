#!/usr/bin/env bash
# Adam-mini 式主实验：fixed global batch=256，五路对照 + MDQ
#
#   CUDA_VISIBLE_DEVICES=4,5 bash run_adam_mini_style.sh
#
# 需先准备数据: bash run_prepare_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export LLAMA_INIT="${LLAMA_INIT:-random}"
export LLAMA_DATA_PATH="${LLAMA_DATA_PATH:-/workspace/data/openwebtext_llama7b_1024}"
export NUM_GPUS="${NUM_GPUS:-2}"
export FIXED_GLOBAL_BATCH="${FIXED_GLOBAL_BATCH:-256}"
export FIXED_G_OUTPUT_DIR="${FIXED_G_OUTPUT_DIR:-$ROOT/results/adam_mini_style_g${FIXED_GLOBAL_BATCH}}"

if [ ! -d "$LLAMA_DATA_PATH" ]; then
  echo "ERROR: 数据不存在: $LLAMA_DATA_PATH" >&2
  echo "请先运行: bash run_prepare_data.sh" >&2
  exit 1
fi

export OPTS="${OPTS:-AdamW-32bit 8bit-Adam-bnb Adam-mini GaLore MDQAdamW-Simple-FusedIO}"

echo "==> Adam-mini 式实验"
echo "==> data=$LLAMA_DATA_PATH init=$LLAMA_INIT Fixed_G=$FIXED_GLOBAL_BATCH GPUs=$NUM_GPUS"
echo "==> OPTS=$OPTS"
echo "==> OUTPUT=$FIXED_G_OUTPUT_DIR"

exec bash run_fixed_global_batch_sweep.sh
