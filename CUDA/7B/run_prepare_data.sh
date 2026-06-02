#!/usr/bin/env bash
# 预处理 OpenWebText（Llama tokenizer, seq=1024）→ LLAMA_DATA_PATH
#
# 全量:
#   bash run_prepare_data.sh
#
# 调试（1000 条）:
#   MAX_SAMPLES=1000 bash run_prepare_data.sh
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export LLAMA_DATA_PATH="${LLAMA_DATA_PATH:-/workspace/data/openwebtext_llama7b_1024}"
export OWT_RAW_PATH="${OWT_RAW_PATH:-/workspace/data/openwebtext/plain_text/}"
export LLAMA_MODEL_ID="${LLAMA_MODEL_ID:-meta-llama/Llama-2-7b-hf}"
export LLAMA_SEQ_LEN="${LLAMA_SEQ_LEN:-1024}"

ARGS=(--raw-path "$OWT_RAW_PATH" --save-path "$LLAMA_DATA_PATH" --tokenizer "$LLAMA_MODEL_ID" --seq-len "$LLAMA_SEQ_LEN")
if [ -n "${MAX_SAMPLES:-}" ]; then
  ARGS+=(--max-samples "$MAX_SAMPLES")
fi

echo "==> raw=$OWT_RAW_PATH"
echo "==> save=$LLAMA_DATA_PATH"
echo "==> tokenizer=$LLAMA_MODEL_ID seq=$LLAMA_SEQ_LEN"

python prepare_openwebtext_llama7b.py "${ARGS[@]}"
