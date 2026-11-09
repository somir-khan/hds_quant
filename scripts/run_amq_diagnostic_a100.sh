#!/bin/bash
#SBATCH --job-name=hqq_amq_diag
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results_hqq_amq_diagnostic/diagnostic_%j.log
#SBATCH --error=results_hqq_amq_diagnostic/diagnostic_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
source_dir="${1:-results_hqq_amq_comparison/job_1169695_llama31_8b_wikitext_amq_matched_pilot_v1}"
seed="${2:-42}"
out="results_hqq_amq_diagnostic/job_${SLURM_JOB_ID:-manual}_archive_seed_${seed}_v1"
"$PYTHON_BIN" -m hds_quant.run_amq_diagnostic --source-dir "$source_dir" --output-dir "$out" --seed "$seed"
echo "AMQ archive diagnostic completed: $out"
