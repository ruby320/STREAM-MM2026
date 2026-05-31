#!/usr/bin/env bash
# Micro-batch sweep：依次对每个 optimizer 探测可用的最大 micro batch（扫到 OOM）
#
# 默认: 4 卡, 指数跳跃+二分找 OOM (micro_max=64), probe 8 steps, benchmark 带 fallback
#
# 4 卡全量重跑:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run_microbatch_sweep.sh
#
# 仅补跑 MDQ（AdamW/bnb 已有结果时）:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 \
#     OPTS="MDQAdamW-Simple MDQAdamW-Simple-FusedIO" bash run_microbatch_sweep.sh
#
# 边界复测（32bit/bnb 在 probe max 附近）:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 FIND_MAX_MICRO=0 \
#     MICRO_LIST=35,36,37,38,39,40 OPTS="AdamW-32bit" bash run_microbatch_sweep.sh
#
# 等显存预算（如 32bit@micro4 的 Peak≈36GB）:
#   PEAK_BUDGET_GB=36 bash run_microbatch_sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

NUM_GPUS="${NUM_GPUS:-4}"
MICRO_MIN="${MICRO_MIN:-4}"
MICRO_MAX="${MICRO_MAX:-64}"
MICRO_LIST="${MICRO_LIST:-}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"
MICRO_WARMUP_STEPS="${MICRO_WARMUP_STEPS:-2}"
MICRO_PROBE_STEPS="${MICRO_PROBE_STEPS:-8}"
MICRO_BENCHMARK_STEPS="${MICRO_BENCHMARK_STEPS:-20}"
PEAK_BUDGET_GB="${PEAK_BUDGET_GB:-}"
MICRO_BENCHMARK="${MICRO_BENCHMARK:-1}"
MICRO_BENCHMARK_FALLBACK="${MICRO_BENCHMARK_FALLBACK:-1}"
MICRO_CONTINUE_ON_FAIL="${MICRO_CONTINUE_ON_FAIL:-1}"
FIND_MAX_MICRO="${FIND_MAX_MICRO:-1}"
MICRO_SEARCH_MODE="${MICRO_SEARCH_MODE:-exp_binary}"
OUTPUT_DIR="${MICRO_OUTPUT_DIR:-$ROOT/results/microbatch_sweep}"

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
  "MDQAdamW-Simple"
  "MDQAdamW-Simple-FusedIO"
)
# shellcheck disable=SC2206
OPTS=(${OPTS:-${ALL_OPTS[@]}})

echo "==> GPUs=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS"
if [ -n "$MICRO_LIST" ]; then
  echo "==> micro_list=$MICRO_LIST accum=$GRAD_ACCUM find_max=0"
else
  echo "==> micro=${MICRO_MIN}..${MICRO_MAX} accum=$GRAD_ACCUM find_max=$FIND_MAX_MICRO search=$MICRO_SEARCH_MODE"
fi
echo "==> warmup=$MICRO_WARMUP_STEPS probe=$MICRO_PROBE_STEPS benchmark_steps=$MICRO_BENCHMARK_STEPS"
echo "==> OUTPUT=$OUTPUT_DIR"

if ! ls stquant_timetest_cpp*.so 1>/dev/null 2>&1; then
  echo "==> 编译 MDQ CUDA 扩展..."
  python setup.py build_ext --inplace
fi

EXTRA_ARGS=()
if [ -n "$MICRO_LIST" ]; then
  EXTRA_ARGS+=(--micro-list "$MICRO_LIST")
else
  EXTRA_ARGS+=(--micro-min "$MICRO_MIN" --micro-max "$MICRO_MAX")
  if [ "$FIND_MAX_MICRO" = "1" ]; then
    EXTRA_ARGS+=(--find-max-micro --search-mode "$MICRO_SEARCH_MODE")
  fi
fi
if [ -n "$PEAK_BUDGET_GB" ]; then
  EXTRA_ARGS+=(--peak-budget-gb "$PEAK_BUDGET_GB")
fi
if [ "$MICRO_BENCHMARK" = "1" ]; then
  EXTRA_ARGS+=(--benchmark)
fi
if [ "$MICRO_BENCHMARK_FALLBACK" = "0" ]; then
  EXTRA_ARGS+=(--no-benchmark-fallback)
fi

mkdir -p "$OUTPUT_DIR"

run_sweep() {
  local opt="$1"
  if [ "$NUM_GPUS" -le 1 ]; then
    python microbatch_sweep.py \
      --optimizer "$opt" \
      --grad-accum "$GRAD_ACCUM" \
      --warmup-steps "$MICRO_WARMUP_STEPS" \
      --probe-steps "$MICRO_PROBE_STEPS" \
      --benchmark-steps "$MICRO_BENCHMARK_STEPS" \
      --output-dir "$OUTPUT_DIR" \
      "${EXTRA_ARGS[@]}"
  else
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
      microbatch_sweep.py \
      --optimizer "$opt" \
      --grad-accum "$GRAD_ACCUM" \
      --warmup-steps "$MICRO_WARMUP_STEPS" \
      --probe-steps "$MICRO_PROBE_STEPS" \
      --benchmark-steps "$MICRO_BENCHMARK_STEPS" \
      --output-dir "$OUTPUT_DIR" \
      "${EXTRA_ARGS[@]}"
  fi
}

for opt in "${OPTS[@]}"; do
  echo ""
  echo "=========================================="
  echo "  Optimizer: $opt"
  echo "=========================================="
  if [ "$MICRO_CONTINUE_ON_FAIL" = "1" ]; then
    run_sweep "$opt" || echo "==> 警告: $opt sweep 失败，继续下一个 optimizer"
  else
    run_sweep "$opt"
  fi
done

echo ""
echo "==> 全部完成。合并结果: $OUTPUT_DIR/sweep_all.csv"
