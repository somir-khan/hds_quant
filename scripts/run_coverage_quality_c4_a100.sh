#!/bin/bash
#SBATCH --job-name=hqq_cov_quality_c4
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%1
#SBATCH --output=results_hqq_coverage_c4_quality/coverage_%A_%a.log
#SBATCH --error=results_hqq_coverage_c4_quality/coverage_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/launch_coverage_study.sh c4 quality "${1:-}"
