#!/usr/bin/env bash
# 固定 grad_accum，各 optimizer 用 config 里 max micro 长训至滑动平均 loss 达标。
#
# 1) 先在 convergence_max_micro_config.json 里填好各 optimizer 的 micro_batch
# 2) 4 卡跑全部:
#      CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run_convergence_max_micro.sh
# 3) 单 optimizer + 覆盖 micro:
#      OPTS="MDQAdamW-Simple-FusedIO" MICRO_BATCH=44 bash run_convergence_max_micro.sh
#
# 环境变量:
#   CONV_CONFIG          配置文件路径
#   CONV_OUTPUT_DIR      输出目录
#   CONV_TARGET_LOSS     默认 5.0
#   CONV_SMOOTH_WINDOW   滑动窗口（log 点数），默认 5
#   CONV_SMOOTH_CONSEC   连续达标次数，默认 3
#   CONV_MAX_STEPS       默认 1500
#   CONV_GRAD_ACCUM      默认 16
#   MICRO_BATCH          覆盖当前 optimizer 的 micro（可选）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
CONV_CONFIG="${CONV_CONFIG:-$ROOT/convergence_max_micro_config.json}"
CONV_OUTPUT_DIR="${CONV_OUTPUT_DIR:-$ROOT/results/convergence_max_micro}"
CONV_TARGET_LOSS="${CONV_TARGET_LOSS:-5.0}"
CONV_SMOOTH_WINDOW="${CONV_SMOOTH_WINDOW:-5}"
CONV_SMOOTH_CONSEC="${CONV_SMOOTH_CONSEC:-3}"
CONV_MAX_STEPS="${CONV_MAX_STEPS:-1500}"
CONV_WARMUP_STEPS="${CONV_WARMUP_STEPS:-10}"
CONV_LOG_INTERVAL="${CONV_LOG_INTERVAL:-10}"
CONV_GRAD_ACCUM="${CONV_GRAD_ACCUM:-16}"
MICRO_BATCH="${MICRO_BATCH:-}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-1800}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "$NUM_GPUS" -le 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export CUDA_VISIBLE_DEVICES="$(printf '%s,' $(seq 0 $((NUM_GPUS - 1))) | sed 's/,$//')"
  fi
fi

ALL_OPTS=(
  "AdamW-32bit"
  "8bit-Adam-bnb"
  "MDQAdamW-Simple-FusedIO"
  "GaLore"
  "Adam-mini"
)
# shellcheck disable=SC2206
OPTS=(${OPTS:-${ALL_OPTS[@]}})

echo "==> Convergence max-micro training"
echo "==> GPUs=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS"
echo "==> CONFIG=$CONV_CONFIG OUTPUT=$CONV_OUTPUT_DIR"
echo "==> target_loss=$CONV_TARGET_LOSS smooth=${CONV_SMOOTH_WINDOW}x${CONV_SMOOTH_CONSEC}"
echo "==> max_steps=$CONV_MAX_STEPS grad_accum=$CONV_GRAD_ACCUM"
echo "==> OPTS=${OPTS[*]}"

if ! ls stquant_timetest_cpp*.so 1>/dev/null 2>&1; then
  echo "==> 编译 MDQ CUDA 扩展..."
  python setup.py build_ext --inplace
fi

mkdir -p "$CONV_OUTPUT_DIR"

run_one() {
  local opt="$1"
  local extra=()
  if [ -n "$MICRO_BATCH" ]; then
    extra+=(--micro-batch "$MICRO_BATCH")
  fi
  if [ "$NUM_GPUS" -le 1 ]; then
    python convergence_max_micro.py \
      --optimizer "$opt" \
      --config "$CONV_CONFIG" \
      --grad-accum "$CONV_GRAD_ACCUM" \
      --target-loss "$CONV_TARGET_LOSS" \
      --smooth-window "$CONV_SMOOTH_WINDOW" \
      --smooth-consecutive "$CONV_SMOOTH_CONSEC" \
      --max-steps "$CONV_MAX_STEPS" \
      --warmup-steps "$CONV_WARMUP_STEPS" \
      --log-interval "$CONV_LOG_INTERVAL" \
      --output-dir "$CONV_OUTPUT_DIR" \
      "${extra[@]}"
  else
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
      convergence_max_micro.py \
      --optimizer "$opt" \
      --config "$CONV_CONFIG" \
      --grad-accum "$CONV_GRAD_ACCUM" \
      --target-loss "$CONV_TARGET_LOSS" \
      --smooth-window "$CONV_SMOOTH_WINDOW" \
      --smooth-consecutive "$CONV_SMOOTH_CONSEC" \
      --max-steps "$CONV_MAX_STEPS" \
      --warmup-steps "$CONV_WARMUP_STEPS" \
      --log-interval "$CONV_LOG_INTERVAL" \
      --output-dir "$CONV_OUTPUT_DIR" \
      "${extra[@]}"
  fi
}

for opt in "${OPTS[@]}"; do
  echo ""
  echo "=========================================="
  echo "  Optimizer: $opt"
  echo "=========================================="
  run_one "$opt" || echo "==> 警告: $opt 失败，继续下一个"
done

echo ""
echo "==> 完成。合并汇总: $CONV_OUTPUT_DIR/convergence_summary_all.csv"
