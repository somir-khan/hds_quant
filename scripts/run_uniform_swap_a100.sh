#!/bin/bash
#SBATCH --job-name=hqq_uniform_swap
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=results_hqq_uniform_swap/uniform_swap_%j.log
#SBATCH --error=results_hqq_uniform_swap/uniform_swap_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
extra=()
if [ -n "${1:-}" ]; then extra+=(--source-dir "$1"); fi
if [ -n "${2:-}" ]; then extra+=(--fixed4-source-dir "$2"); fi
"$PYTHON_BIN" -m hds_quant.run_uniform_swap \
  --config configs/amq_diagnostics/llama31_8b_wikitext_uniform_swap_v1.json \
  --output-dir "results_hqq_uniform_swap/job_${SLURM_JOB_ID:-manual}_llama31_8b_wikitext_uniform_swap_v1" "${extra[@]}"
