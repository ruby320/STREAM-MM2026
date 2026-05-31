#!/usr/bin/env bash
# MDQ threshold scale sensitivity sweep
#
# Usage:
#   bash run_sweep.sh                     # all tasks, parallel on GPU 2+3
#   bash run_sweep.sh gpt2-medium         # single task
#   GPUS="0 1" SEED=42 bash run_sweep.sh
#   MAX_STEPS=50 bash run_sweep.sh gpt2-medium   # quick smoke test
#
# Sweep: thresholds = s × {6.8, 12, 24},  s ∈ {0.75, 0.85, 1.0, 1.15, 1.25}
# Each task also runs AdamW-32bit baseline.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="${1:-all}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-}"
read -r -a GPUS <<< "${GPUS:-2 3}"

# "adamw32" or numeric scale
JOB_SPECS=(adamw32 0.75 0.85 1.0 1.15 1.25)

EXTRA=(--seed "$SEED")
if [[ -n "$MAX_STEPS" ]]; then
  EXTRA+=(--max-steps "$MAX_STEPS")
fi

LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p results/runs results/figures "$LOG_DIR"

run_job() {
  local gpu="$1"
  local task="$2"
  local spec="$3"
  local logfile="$LOG_DIR/${task}_${spec}.log"
  echo ">>> GPU=$gpu  task=$task  spec=$spec  log=$logfile" >&2

  if [[ "$spec" == "adamw32" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python pretrain_threshold_sweep.py \
      --task "$task" --mode adamw32 "${EXTRA[@]}" > "$logfile" 2>&1
  else
    CUDA_VISIBLE_DEVICES="$gpu" python pretrain_threshold_sweep.py \
      --task "$task" --mode mdq --threshold-scale "$spec" "${EXTRA[@]}" \
      > "$logfile" 2>&1
  fi
}

launch_job_bg() {
  local gpu="$1"
  local task="$2"
  local spec="$3"
  local logfile="$LOG_DIR/${task}_${spec}.log"
  echo ">>> GPU=$gpu  task=$task  spec=$spec  (background)  log=$logfile" >&2

  if [[ "$spec" == "adamw32" ]]; then
    CUDA_VISIBLE_DEVICES="$gpu" python pretrain_threshold_sweep.py \
      --task "$task" --mode adamw32 "${EXTRA[@]}" > "$logfile" 2>&1 &
  else
    CUDA_VISIBLE_DEVICES="$gpu" python pretrain_threshold_sweep.py \
      --task "$task" --mode mdq --threshold-scale "$spec" "${EXTRA[@]}" \
      > "$logfile" 2>&1 &
  fi
}

wait_all() {
  local failed=0
  local pid
  for pid in "$@"; do
    if ! wait "$pid"; then
      echo "ERROR: job pid=$pid failed (see $LOG_DIR/)" >&2
      failed=1
    fi
  done
  return "$failed"
}

run_task_sweep() {
  local task="$1"
  local gpu_a="$2"
  local gpu_b="$3"

  echo ""
  echo "=== Task: $task ==="

  local i=0
  local n=${#JOB_SPECS[@]}
  while [[ $i -lt $n ]]; do
    launch_job_bg "$gpu_a" "$task" "${JOB_SPECS[$i]}"
    local pid1=$!
    if [[ $((i + 1)) -lt $n ]]; then
      launch_job_bg "$gpu_b" "$task" "${JOB_SPECS[$((i + 1))]}"
      local pid2=$!
      wait_all "$pid1" "$pid2"
      i=$((i + 2))
    else
      wait_all "$pid1"
      i=$((i + 1))
    fi
  done
}

run_task_serial() {
  local task="$1"
  local gpu="$2"
  echo ""
  echo "=== Task: $task (serial) ==="
  for spec in "${JOB_SPECS[@]}"; do
    run_job "$gpu" "$task" "$spec"
  done
}

echo "=== MDQ Threshold Scale Sensitivity ==="
echo "Task=$TASK  SEED=$SEED  GPUS=${GPUS[*]}  Jobs=${JOB_SPECS[*]}"

if [[ "$TASK" == "all" ]]; then
  if [[ ${#GPUS[@]} -ge 2 ]]; then
    run_task_sweep "gpt2-medium" "${GPUS[0]}" "${GPUS[1]}"
    run_task_sweep "roberta-large" "${GPUS[0]}" "${GPUS[1]}"
  else
    run_task_serial "gpt2-medium" "${GPUS[0]}"
    run_task_serial "roberta-large" "${GPUS[0]}"
  fi
else
  if [[ ${#GPUS[@]} -ge 2 ]]; then
    run_task_sweep "$TASK" "${GPUS[0]}" "${GPUS[1]}"
  else
    run_task_serial "$TASK" "${GPUS[0]}"
  fi
fi

echo ""
echo ">>> Aggregating results..."
python aggregate_results.py

echo ""
echo "=== Done ==="
echo "Summary:  $SCRIPT_DIR/results/threshold_summary.csv"
echo "Figures:  $SCRIPT_DIR/results/figures/"
echo "Logs:     $LOG_DIR/"
