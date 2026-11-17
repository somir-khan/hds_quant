#!/bin/bash
#SBATCH --job-name=hqq_arr_e_1b
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=results_hqq_arr_e_diagnostic/run_%j.log
#SBATCH --error=results_hqq_arr_e_diagnostic/run_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then echo "Set HQQ_PYTHON to your existing environment." >&2; exit 1; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
"$PYTHON_BIN" -m hds_quant.run_arr_stage12 --mode diagnostic --case-index 6 \
  --output-dir "results_hqq_arr_e_diagnostic/job_${SLURM_JOB_ID:-manual}_1b_wikitext_seed42_v1" "$@"
