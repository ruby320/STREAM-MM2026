#!/usr/bin/env bash
# 固定 Global_Batch=256（micro×accum×GPUs=256），从大到小搜 max micro 并 benchmark。
#
# 默认 2 卡：P=256/2=128，合法 micro∈{128,64,32,16,8,4,2,1}（比 4 卡多 128/64 档）
# 默认 3 个 optimizer：AdamW-32bit / 8bit-Adam-bnb / MDQAdamW-Simple-FusedIO
#
# 2 卡（默认）:
#   CUDA_VISIBLE_DEVICES=4,5 NUM_GPUS=2 bash run_fixed_global_batch_sweep.sh
#
# 改回 4 卡:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 \
#     FIXED_G_OUTPUT_DIR=$PWD/results/fixed_global_batch bash run_fixed_global_batch_sweep.sh
#
# 单 optimizer:
#   OPTS="MDQAdamW-Simple-FusedIO" bash run_fixed_global_batch_sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

NUM_GPUS="${NUM_GPUS:-2}"
FIXED_GLOBAL_BATCH="${FIXED_GLOBAL_BATCH:-256}"
MICRO_WARMUP_STEPS="${MICRO_WARMUP_STEPS:-2}"
MICRO_PROBE_STEPS="${MICRO_PROBE_STEPS:-8}"
MICRO_BENCHMARK_WARMUP="${MICRO_BENCHMARK_WARMUP:-10}"
MICRO_BENCHMARK_STEPS="${MICRO_BENCHMARK_STEPS:-40}"
MICRO_BENCHMARK="${MICRO_BENCHMARK:-1}"
MICRO_BENCHMARK_FALLBACK="${MICRO_BENCHMARK_FALLBACK:-1}"
MICRO_CONTINUE_ON_FAIL="${MICRO_CONTINUE_ON_FAIL:-1}"
OUTPUT_DIR="${FIXED_G_OUTPUT_DIR:-$ROOT/results/fixed_global_batch_2gpu}"

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
)
# shellcheck disable=SC2206
OPTS=(${OPTS:-${ALL_OPTS[@]}})

if [ $((FIXED_GLOBAL_BATCH % NUM_GPUS)) -ne 0 ]; then
  echo "ERROR: FIXED_GLOBAL_BATCH=$FIXED_GLOBAL_BATCH 不能被 NUM_GPUS=$NUM_GPUS 整除" >&2
  exit 1
fi
PER_GPU_PRODUCT=$((FIXED_GLOBAL_BATCH / NUM_GPUS))
echo "==> Fixed Global Batch sweep"
echo "==> GPUs=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS FIXED_G=$FIXED_GLOBAL_BATCH (micro×accum=${PER_GPU_PRODUCT})"
echo "==> probe: warmup=$MICRO_WARMUP_STEPS steps=$MICRO_PROBE_STEPS"
echo "==> benchmark: warmup=$MICRO_BENCHMARK_WARMUP steps=$MICRO_BENCHMARK_STEPS"
echo "==> OUTPUT=$OUTPUT_DIR"
echo "==> OPTS=${OPTS[*]}"

if ! ls stquant_timetest_cpp*.so 1>/dev/null 2>&1; then
  echo "==> 编译 MDQ CUDA 扩展..."
  python setup.py build_ext --inplace
fi

EXTRA_ARGS=(
  --fixed-global-batch "$FIXED_GLOBAL_BATCH"
  --warmup-steps "$MICRO_WARMUP_STEPS"
  --probe-steps "$MICRO_PROBE_STEPS"
  --benchmark-warmup-steps "$MICRO_BENCHMARK_WARMUP"
  --benchmark-steps "$MICRO_BENCHMARK_STEPS"
  --output-dir "$OUTPUT_DIR"
)
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
      "${EXTRA_ARGS[@]}"
  else
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
      microbatch_sweep.py \
      --optimizer "$opt" \
      "${EXTRA_ARGS[@]}"
  fi
}

for opt in "${OPTS[@]}"; do
  echo ""
  echo "=========================================="
  echo "  Optimizer: $opt  (Fixed_G=$FIXED_GLOBAL_BATCH)"
  echo "=========================================="
  if [ "$MICRO_CONTINUE_ON_FAIL" = "1" ]; then
    run_sweep "$opt" || echo "==> 警告: $opt fixed-G sweep 失败，继续下一个"
  else
    run_sweep "$opt"
  fi
done

echo ""
echo "==> 全部完成。合并结果: $OUTPUT_DIR/g${FIXED_GLOBAL_BATCH}_sweep_all.csv"
