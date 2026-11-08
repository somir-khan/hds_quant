#!/bin/bash
#SBATCH --job-name=hqq_controls
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --output=results_hqq_representation_controls/controls_%j.log
#SBATCH --error=results_hqq_representation_controls/controls_%j.err

# mkdir -p results_hqq_representation_controls
# sbatch scripts/run_representation_controls_a100.sh [config.json]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
spec="${1:-configs/representation_controls/llama31_8b_wikitext.json}"
bash scripts/launch_representation_controls.sh "$spec" results_hqq_representation_controls
