#!/bin/bash
# Shared launcher; called from either dataset-specific SLURM array script.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python environment not found: $PYTHON_BIN. Set HQQ_PYTHON in your existing .env." >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
dataset="${1:?dataset required}"
case "$dataset" in
  wikitext) config=configs/frozen_replication/llama31_8b_wikitext_v1.json ;;
  c4) config=configs/frozen_replication/llama31_8b_c4_v1.json ;;
  *) echo "Expected wikitext or c4" >&2; exit 1 ;;
esac
case "${SLURM_ARRAY_TASK_ID:-}" in
  0) seed=42 ;;
  1) seed=100 ;;
  2) seed=2026 ;;
  *) echo "Use array task 0, 1 or 2 (seeds 42, 100 and 2026)." >&2; exit 1 ;;
esac
extra=()
if [ -n "${2:-}" ]; then extra+=(--source-dir "$2"); fi
"$PYTHON_BIN" -m hds_quant.run_frozen_replication \
  --config "$config" --seed "$seed" \
  --output-dir "results_hqq_frozen_replication_${dataset}/job_${SLURM_ARRAY_JOB_ID:-manual}_${SLURM_ARRAY_TASK_ID}_seed${seed}_v1" \
  "${extra[@]}"
