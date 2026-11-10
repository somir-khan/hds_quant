#!/bin/bash
#SBATCH --job-name=hqq_amq_fixed4
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=results_hqq_amq_fixed4_diagnostic/fixed4_%j.log
#SBATCH --error=results_hqq_amq_fixed4_diagnostic/fixed4_%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/launch_amq_followup.sh configs/amq_diagnostics/llama31_8b_wikitext_fixed4_v1.json results_hqq_amq_fixed4_diagnostic "${1:-}"
