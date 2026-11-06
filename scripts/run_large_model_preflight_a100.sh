#!/bin/bash
#SBATCH --job-name=hqq_large_preflight
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=results_hqq_large_preflight/preflight_%j.log
#SBATCH --error=results_hqq_large_preflight/preflight_%j.err

# mkdir -p results_hqq_large_preflight
# Default: Llama-3.1-8B-Instruct
# sbatch scripts/run_large_model_preflight_a100.sh
# Exact 7B alternative:
# sbatch scripts/run_large_model_preflight_a100.sh configs/generalization/llama2_7b_preflight.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
spec="${1:-configs/generalization/llama31_8b_preflight.json}"
bash scripts/launch_generalization.sh "$spec" results_hqq_large_preflight
