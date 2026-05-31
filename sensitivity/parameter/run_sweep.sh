#!/usr/bin/env bash
# MDQ score-smoothing 参数敏感性 sweep
# 用法:
#   bash run_sweep.sh              # 跑全部（单卡）
#   bash run_sweep.sh 1            # torchrun 1 GPU
#   bash run_sweep.sh 4            # torchrun 4 GPU
#   SEED=43 bash run_sweep.sh      # 换 seed

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

NGPU="${1:-1}"
SEED="${SEED:-42}"
MAX_STEPS="${MAX_STEPS:-400}"

run_one() {
  local extra_args=("$@")
  if [[ "$NGPU" -gt 1 ]]; then
    torchrun --nproc_per_node="$NGPU" pretrain_sensitivity.py \
      --seed "$SEED" \
      --max-steps "$MAX_STEPS" \
      "${extra_args[@]}"
  else
    python pretrain_sensitivity.py \
      --seed "$SEED" \
      --max-steps "$MAX_STEPS" \
      "${extra_args[@]}"
  fi
}

echo "=== MDQ Parameter Sensitivity Sweep ==="
echo "GPUs=$NGPU  SEED=$SEED  MAX_STEPS=$MAX_STEPS"
echo "Output: $SCRIPT_DIR/results/"

mkdir -p results/runs ckpts

# 1) 32-bit baseline
echo ""
echo ">>> [1/8] AdamW 32-bit baseline"
run_one --mode adamw32

# 2) OAT sweeps
declare -A SWEEPS=(
  ["alpha"]="0.75 0.85 0.9 0.95 0.99"
  ["tau_scale"]="0.5 0.75 1.0 1.25 1.5"
  ["update_freq"]="5 10 20 40 80"
  ["score_bias"]="6.8 7.0 7.2 7.4 7.6"
  ["w_n"]="0.5 0.75 1.0 1.25 1.5"
  ["init_score"]="8 10 12 14 16"
)

idx=2
for param in alpha tau_scale update_freq score_bias w_n init_score; do
  echo ""
  echo ">>> [$idx/8] Sweep param: $param"
  for val in ${SWEEPS[$param]}; do
    echo "    -> $param=$val"
    run_one --mode mdq --sweep-param "$param" --sweep-value "$val"
  done
  idx=$((idx + 1))
done

# 3) Combined stress test
echo ""
echo ">>> [8/8] MDQ stress combo (alpha=0.75, tau_scale=1.5, update_freq=40)"
run_one --mode stress

# 4) Aggregate summary table
echo ""
echo ">>> Aggregating results into summary table..."
python aggregate_results.py

echo ""
echo "=== Sweep complete ==="
echo "Summary table: $SCRIPT_DIR/results/sensitivity_summary.csv"
