#!/bin/bash
#SBATCH --job-name=hqq_amq
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=results_hqq_amq_comparison/amq_%j.log
#SBATCH --error=results_hqq_amq_comparison/amq_%j.err

# mkdir -p results_hqq_amq_comparison
# sbatch scripts/run_amq_comparison_a100.sh [config.json]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
spec="${1:-configs/amq_comparison/llama31_8b_wikitext_pilot.json}"
bash scripts/launch_amq_comparison.sh "$spec" results_hqq_amq_comparison
