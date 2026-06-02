#!/usr/bin/env bash
# LLaMA-7B micro-batch sweep（DeepSpeed ZeRO-2，默认随机初始化）
#
# 4 卡:
#   CUDA_VISIBLE_DEVICES=4,5,6,7 NUM_GPUS=4 bash run_microbatch_sweep.sh
#
# 仅 MDQ:
#   OPTS="MDQAdamW-Simple-FusedIO" bash run_microbatch_sweep.sh
#
# 32bit smoke（固定 micro=1）:
#   FIND_MAX_MICRO=0 MICRO_LIST=1 GRAD_ACCUM=16 OPTS="AdamW-32bit" bash run_microbatch_sweep.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export LLAMA_INIT="${LLAMA_INIT:-random}"
export LLAMA_DATA_PATH="${LLAMA_DATA_PATH:-/workspace/data/openwebtext_llama7b_1024}"

NUM_GPUS="${NUM_GPUS:-4}"
MICRO_MIN="${MICRO_MIN:-1}"
MICRO_MAX="${MICRO_MAX:-32}"
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
  "GaLore"
  "Adam-mini"
)
# shellcheck disable=SC2206
OPTS=(${OPTS:-${ALL_OPTS[@]}})

echo "==> LLaMA-7B sweep | init=$LLAMA_INIT ZeRO-2"
echo "==> data=$LLAMA_DATA_PATH"
echo "==> GPUs=$CUDA_VISIBLE_DEVICES NUM_GPUS=$NUM_GPUS"
echo "==> OUTPUT=$OUTPUT_DIR"

if ! ls "$PARENT"/stquant_timetest_cpp*.so 1>/dev/null 2>&1; then
  echo "==> 编译 MDQ CUDA 扩展（父目录）..."
  (cd "$PARENT" && python setup.py build_ext --inplace)
fi

mkdir -p "$OUTPUT_DIR"

run_sweep() {
  local opt="$1"
  local common_args=(
    --optimizer "$opt"
    --grad-accum "$GRAD_ACCUM"
    --warmup-steps "$MICRO_WARMUP_STEPS"
    --probe-steps "$MICRO_PROBE_STEPS"
    --benchmark-steps "$MICRO_BENCHMARK_STEPS"
    --output-dir "$OUTPUT_DIR"
  )
  if [ -n "$PEAK_BUDGET_GB" ]; then
    common_args+=(--peak-budget-gb "$PEAK_BUDGET_GB")
  fi
  if [ "$MICRO_BENCHMARK" = "1" ]; then
    common_args+=(--benchmark)
  fi
  if [ "$MICRO_BENCHMARK_FALLBACK" = "0" ]; then
    common_args+=(--no-benchmark-fallback)
  fi

  if [ "$FIND_MAX_MICRO" = "1" ] && [ -z "$MICRO_LIST" ]; then
    python microbatch_sweep.py \
      --orchestrate \
      "${common_args[@]}" \
      --micro-min "$MICRO_MIN" \
      --micro-max "$MICRO_MAX" \
      --search-mode "$MICRO_SEARCH_MODE"
    return
  fi

  if [ -n "$MICRO_LIST" ]; then
    common_args+=(--micro-list "$MICRO_LIST")
  else
    common_args+=(--micro-min "$MICRO_MIN" --micro-max "$MICRO_MAX")
  fi

  if [ "$NUM_GPUS" -le 1 ]; then
    python microbatch_sweep.py "${common_args[@]}"
  else
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
      microbatch_sweep.py \
      "${common_args[@]}"
  fi
}

for opt in "${OPTS[@]}"; do
  echo ""
  echo "=========================================="
  echo "  Optimizer: $opt"
  echo "=========================================="
  if [ "$MICRO_CONTINUE_ON_FAIL" = "1" ]; then
    run_sweep "$opt" || echo "==> 警告: $opt sweep 失败，继续"
  else
    run_sweep "$opt"
  fi
done

echo ""
echo "==> 完成: $OUTPUT_DIR/sweep_all.csv"
