#!/usr/bin/env bash
# LLaMA-7B optimizer 计时（每 optimizer 独立 torchrun 进程）
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export LLAMA_INIT="${LLAMA_INIT:-random}"
export LLAMA_DATA_PATH="${LLAMA_DATA_PATH:-/workspace/data/openwebtext_llama7b_1024}"
NUM_GPUS="${NUM_GPUS:-4}"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  export CUDA_VISIBLE_DEVICES="$(printf '%s,' $(seq 0 $((NUM_GPUS - 1))) | sed 's/,$//')"
fi

if ! ls "$PARENT"/stquant_timetest_cpp*.so 1>/dev/null 2>&1; then
  (cd "$PARENT" && python setup.py build_ext --inplace)
fi

mkdir -p results

ALL_OPTS=(
  "MDQAdamW-Simple-FusedIO"
  "MDQAdamW-Simple"
  "AdamW-32bit"
  "8bit-Adam-bnb"
)
# shellcheck disable=SC2206
OPTS=(${OPTS:-${ALL_OPTS[@]}})

for opt in "${OPTS[@]}"; do
  echo ">>> $opt"
  torchrun --standalone --nproc_per_node="$NUM_GPUS" \
    llama7b_timetest.py --optimizer "$opt" \
    || echo "警告: $opt 失败"
done

echo "==> 结果: $ROOT/results/"
