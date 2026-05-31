#!/usr/bin/env bash
# GPT2-XL optimizer timetest — 每 optimizer 独立进程 + warmup + 轮换顺序
#
# 默认: warmup=10, measure=40 optimizer steps（覆盖 2 次 MDQ stats 更新周期）
#   bash run_build_and_train.sh
#
# 4 卡:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 NUM_GPUS=4 bash run_build_and_train.sh
#
# 顺序: TIMETEST_ORDER=rotate|random|fixed  TIMETEST_ROTATE=0..3（4 个 optimizer）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export TIMETEST_DEBUG="${TIMETEST_DEBUG:-1}"
export TIMETEST_WARMUP_STEPS="${TIMETEST_WARMUP_STEPS:-10}"
export TIMETEST_MAX_STEPS="${TIMETEST_MAX_STEPS:-40}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

NUM_GPUS="${NUM_GPUS:-1}"
ORDER_MODE="${TIMETEST_ORDER:-rotate}"
ROTATE="${TIMETEST_ROTATE:-1}"

export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  if [ "$NUM_GPUS" -le 1 ]; then
    export CUDA_VISIBLE_DEVICES=0
  else
    export CUDA_VISIBLE_DEVICES="$(printf '%s,' $(seq 0 $((NUM_GPUS - 1))) | sed 's/,$//')"
  fi
fi

ALL_OPTS=(
  "MDQAdamW-Simple"
  "MDQAdamW-Simple-FusedIO"
  "AdamW-32bit"
  "8bit-Adam-bnb"
)

echo "==> 编译 stquant_timetest_cpp"
pip install -e . --no-build-isolation -q

mkdir -p results

run_one() {
  local opt="$1"
  local idx="$2"
  export TIMETEST_RUN_INDEX="$idx"
  echo ""
  echo "========== [$idx] optimizer=$opt =========="
  if [ "$NUM_GPUS" -le 1 ]; then
    python Cfinetuning_timetest.py --optimizer "$opt"
  else
    torchrun --standalone --nproc_per_node="$NUM_GPUS" \
      Cfinetuning_timetest.py --optimizer "$opt"
  fi
}

if [ "$ORDER_MODE" = "random" ]; then
  mapfile -t ORDER < <(printf '%s\n' "${ALL_OPTS[@]}" | shuf)
elif [ "$ORDER_MODE" = "rotate" ]; then
  ORDER=()
  n=${#ALL_OPTS[@]}
  r=$(( ROTATE % n ))
  for i in $(seq 0 $(( n - 1 ))); do
    ORDER+=( "${ALL_OPTS[$(( (i + r) % n ))]}" )
  done
else
  ORDER=( "${ALL_OPTS[@]}" )
fi

echo "==> NUM_GPUS=${NUM_GPUS} CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "==> WARMUP=${TIMETEST_WARMUP_STEPS} MEASURE=${TIMETEST_MAX_STEPS} DEBUG=${TIMETEST_DEBUG}"
echo "==> ORDER_MODE=${ORDER_MODE} ROTATE=${ROTATE}"
echo "==> 运行顺序: ${ORDER[*]}"

idx=0
for opt in "${ORDER[@]}"; do
  run_one "$opt" "$idx"
  idx=$((idx + 1))
  sleep 2
done

echo ""
echo "==> 汇总 CSV"
python aggregate_timing.py --dir results

echo "完成。查看 results/summary_metrics.csv"
