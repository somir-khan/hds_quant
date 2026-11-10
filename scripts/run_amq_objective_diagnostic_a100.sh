#!/bin/bash
#SBATCH --job-name=hqq_amq_objectives
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results_hqq_amq_objective_diagnostic/objectives_%j.log
#SBATCH --error=results_hqq_amq_objective_diagnostic/objectives_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/launch_amq_followup.sh configs/amq_diagnostics/llama31_8b_wikitext_objectives_v1.json results_hqq_amq_objective_diagnostic "${1:-}"
