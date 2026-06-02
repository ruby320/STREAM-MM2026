#!/usr/bin/env bash
# 验收：AdamW-32bit 至少能跑 micro=1（ZeRO-2 + 随机初始化 LLaMA-7B）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export LLAMA_INIT="${LLAMA_INIT:-random}"
export LLAMA_DATA_PATH="${LLAMA_DATA_PATH:-/workspace/data/openwebtext_llama7b_1024}"
export NUM_GPUS="${NUM_GPUS:-2}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4,5}"

echo "==> smoke: AdamW-32bit micro=1 accum=16 GPUs=$NUM_GPUS init=$LLAMA_INIT"

FIND_MAX_MICRO=0 \
  MICRO_LIST=1 \
  GRAD_ACCUM=16 \
  MICRO_BENCHMARK=0 \
  OPTS="AdamW-32bit" \
  bash run_microbatch_sweep.sh

echo "==> smoke 完成，检查 results/microbatch_sweep/sweep_AdamW-32bit.csv"
