#!/usr/bin/env bash
# Collect MDQ score distributions and plot threshold analysis figures.
#
# Usage:
#   bash run_all.sh                        # all 3 models, parallel on GPU 2+3
#   GPUS="0 1" bash run_all.sh             # custom physical GPU ids
#   bash run_all.sh gpt2-medium            # single model on GPUS[0]
#   GPU=3 bash run_all.sh vit-base         # single model on a specific GPU
#   SEED=43 MAX_STEPS=100 bash run_all.sh  # quick test
#
# Parallel schedule (MODEL=all, >=2 GPUs):
#   Wave 1: gpt2-medium @ GPU[0]  ||  vit-base @ GPU[1]
#   Wave 2: roberta-large @ GPU[0]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="${1:-all}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-}"
# Space-separated physical GPU indices, e.g. "2 3"
read -r -a GPUS <<< "${GPUS:-2 3}"
GPU="${GPU:-}"  # optional override for single-model runs

EXTRA=("--seed" "$SEED")
if [[ -n "$MAX_STEPS" ]]; then
  EXTRA+=("--max-steps" "$MAX_STEPS")
fi

LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p results "$LOG_DIR"

run_one() {
  local gpu="$1"
  local mdl="$2"
  local logfile="$LOG_DIR/${mdl}.log"
  echo ">>> [$mdl] GPU=$gpu  log=$logfile" >&2
  CUDA_VISIBLE_DEVICES="$gpu" python collect_score_distribution.py \
    --model "$mdl" "${EXTRA[@]}" > "$logfile" 2>&1
}

# Must launch background jobs in the current shell (not inside $()) so wait works.
launch_bg() {
  local gpu="$1"
  local mdl="$2"
  local logfile="$LOG_DIR/${mdl}.log"
  echo ">>> [$mdl] GPU=$gpu  log=$logfile  (background)" >&2
  CUDA_VISIBLE_DEVICES="$gpu" python collect_score_distribution.py \
    --model "$mdl" "${EXTRA[@]}" > "$logfile" 2>&1 &
}

wait_all() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      echo "ERROR: background job pid=$pid failed (check $LOG_DIR/*.log)" >&2
      failed=1
    fi
  done
  return "$failed"
}

echo "=== MDQ Score Distribution Experiment ==="
echo "Model=$MODEL  SEED=$SEED  GPUS=${GPUS[*]}  Output=$SCRIPT_DIR/results/"

echo ""
echo ">>> Collecting scores..."

if [[ "$MODEL" != "all" ]]; then
  gpu="${GPU:-${GPUS[0]}}"
  run_one "$gpu" "$MODEL"
else
  n_gpus=${#GPUS[@]}
  if [[ "$n_gpus" -ge 2 ]]; then
    gpu0="${GPUS[0]}"
    gpu1="${GPUS[1]}"

    launch_bg "$gpu0" "gpt2-medium"
    pid1=$!
    launch_bg "$gpu1" "vit-base"
    pid2=$!

    wait_all "$pid1" "$pid2"
    run_one "$gpu0" "roberta-large"
  else
    gpu="${GPUS[0]}"
    for mdl in gpt2-medium vit-base roberta-large; do
      run_one "$gpu" "$mdl"
    done
  fi
fi

echo ""
echo ">>> Plotting distributions..."
python plot_score_distributions.py

echo ""
echo "=== Done ==="
echo "Combined figure: $SCRIPT_DIR/results/score_distribution_combined.png"
echo "Summary CSV:     $SCRIPT_DIR/results/score_summary.csv"
echo "Per-model logs:  $LOG_DIR/"
