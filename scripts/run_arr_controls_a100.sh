#!/bin/bash
#SBATCH --job-name=hqq_arr_controls
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=6%1
#SBATCH --output=results_hqq_arr_controls/run_%A_%a.log
#SBATCH --error=results_hqq_arr_controls/run_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then echo "Set HQQ_PYTHON to your existing environment." >&2; exit 1; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
task="${SLURM_ARRAY_TASK_ID:?Submit this script with sbatch}"
if ! [[ "$task" =~ ^[0-9]+$ ]] || [ "$task" -gt 11 ]; then echo "Case indices are 0..11" >&2; exit 1; fi
"$PYTHON_BIN" -m hds_quant.run_arr_stage12 --mode controls --case-index "$task" \
  --output-dir "results_hqq_arr_controls/job_${SLURM_ARRAY_JOB_ID}_${task}_v1" "$@"
