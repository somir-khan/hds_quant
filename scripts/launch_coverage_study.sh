#!/bin/bash
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then echo "Set HQQ_PYTHON to your existing Python environment." >&2; exit 1; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
dataset="${1:?dataset required}"
mode="${2:?mode required}"
case "$dataset" in wikitext|c4) ;; *) echo "Unknown dataset" >&2; exit 1 ;; esac
task="${SLURM_ARRAY_TASK_ID:?Use the dataset-specific SLURM array script}"
if ! [[ "$task" =~ ^[0-9]+$ ]]; then echo "Invalid array index" >&2; exit 1; fi
extra=()
suffix=""
case "$mode" in
  quality)
    if [ "$task" -gt 2 ]; then echo "Quality indices are 0..2" >&2; exit 1; fi
    seed_index="$task"
    ;;
  timing)
    if [ "$task" -gt 17 ]; then echo "Timing indices are 0..17" >&2; exit 1; fi
    seed_index=$((task / 6))
    k_index=$(((task % 6) / 2))
    policy_index=$((task % 2))
    ks=(2 4 8)
    policies=(stratified boundary_inclusive)
    extra+=(--k-layers "${ks[$k_index]}" --policy "${policies[$policy_index]}")
    suffix="_k${ks[$k_index]}_${policies[$policy_index]}"
    ;;
  *) echo "Unknown mode" >&2; exit 1 ;;
esac
seeds=(42 100 2026)
seed="${seeds[$seed_index]}"
if [ -n "${3:-}" ]; then extra+=(--source-dir "$3"); fi
"$PYTHON_BIN" -m hds_quant.run_coverage_study \
  --config "configs/coverage_study/llama31_8b_${dataset}_v1.json" \
  --mode "$mode" --seed "$seed" \
  --output-dir "results_hqq_coverage_${dataset}_${mode}/job_${SLURM_ARRAY_JOB_ID:-manual}_${task}_seed${seed}${suffix}_v1" \
  "${extra[@]}"
