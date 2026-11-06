#!/bin/bash
#SBATCH --job-name=hqq_c4_1b
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results_hqq_c4_1b/c4_%j.log
#SBATCH --error=results_hqq_c4_1b/c4_%j.err

# mkdir -p results_hqq_c4_1b
# sbatch scripts/run_c4_1b_a100.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
spec="${1:-configs/generalization/c4_1b.json}"
bash scripts/launch_generalization.sh "$spec" results_hqq_c4_1b
