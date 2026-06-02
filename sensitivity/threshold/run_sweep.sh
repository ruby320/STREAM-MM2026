#!/usr/bin/env bash
# MDQ threshold scale sensitivity sweep
#
# Usage:
#   bash run_sweep.sh                        # all tasks, parallel on GPU 0-3
#   bash run_sweep.sh roberta-large          # single task
#   GPUS="0 1 2 3" bash run_sweep.sh roberta-large
#   GPUS="2 3" bash run_sweep.sh gpt2-medium # 2-GPU mode still works
#   NUM_EPOCHS=1 GPUS="0 1 2 3" bash run_sweep.sh roberta-large
#   NUM_EPOCHS=3 GPUS="0 1 2 3" bash run_sweep.sh roberta-large  # match finetuning_glue.py
#   MAX_STEPS=49088 GPUS="0 1 2 3" bash run_sweep.sh roberta-large  # override steps directly
#
# Sweep: thresholds = s × {6.8, 12, 24},  s ∈ {0.75, 0.85, 1.0, 1.15, 1.25}
# Each task also runs AdamW-32bit baseline (6 jobs total per task).
# With 4 GPUs: wave1 runs 4 jobs, wave2 runs remaining 2 jobs.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TASK="${1:-all}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-}"
NUM_EPOCHS="${NUM_EPOCHS:-}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3}"

# "adamw32" or numeric scale
JOB_SPECS=(adamw32 0.75 0.85 1.0 1.15 1.25)

EXTRA=(--seed "$SEED")
if [[ -n "$MAX_STEPS" ]]; then
  EXTRA+=(--max-steps "$MAX_STEPS")
fi
if [[ -n "$NUM_EPOCHS" ]]; then
  EXTRA+=(--num-epochs "$NUM_EPOCHS")
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

# Run all JOB_SPECS for one task, using up to N GPUs in parallel per wave.
run_task_sweep() {
  local task="$1"
  shift
  local -a gpus=("$@")
  local n_gpus=${#gpus[@]}

  if [[ "$n_gpus" -lt 1 ]]; then
    echo "ERROR: no GPUs specified" >&2
    return 1
  fi

  echo ""
  echo "=== Task: $task  (${n_gpus}-GPU parallel) ==="

  local i=0
  local n=${#JOB_SPECS[@]}
  local wave=1
  while [[ $i -lt $n ]]; do
    local -a pids=()
    local batch=0
    echo "--- Wave $wave ---" >&2
    while [[ $batch -lt $n_gpus && $((i + batch)) -lt $n ]]; do
      local spec="${JOB_SPECS[$((i + batch))]}"
      launch_job_bg "${gpus[$batch]}" "$task" "$spec"
      pids+=($!)
      batch=$((batch + 1))
    done
    wait_all "${pids[@]}"
    i=$((i + batch))
    wave=$((wave + 1))
  done
}

run_task_serial() {
  local task="$1"
  local gpu="$2"
  echo ""
  echo "=== Task: $task (serial on GPU $gpu) ==="
  for spec in "${JOB_SPECS[@]}"; do
    run_job "$gpu" "$task" "$spec"
  done
}

echo "=== MDQ Threshold Scale Sensitivity ==="
echo "Task=$TASK  SEED=$SEED  NUM_EPOCHS=${NUM_EPOCHS:-default}  MAX_STEPS=${MAX_STEPS:-default}  GPUS=${GPUS[*]}  Jobs=${JOB_SPECS[*]}"

if [[ "$TASK" == "all" ]]; then
  if [[ ${#GPUS[@]} -ge 2 ]]; then
    run_task_sweep "gpt2-medium" "${GPUS[@]}"
    run_task_sweep "roberta-large" "${GPUS[@]}"
  else
    run_task_serial "gpt2-medium" "${GPUS[0]}"
    run_task_serial "roberta-large" "${GPUS[0]}"
  fi
else
  if [[ ${#GPUS[@]} -ge 2 ]]; then
    run_task_sweep "$TASK" "${GPUS[@]}"
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
