#!/bin/bash
#SBATCH --job-name=hqq_8b_wikitext
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=results_hqq_8b_wikitext/8b_%j.log
#SBATCH --error=results_hqq_8b_wikitext/8b_%j.err

# Submit from the project root after creating the scheduler log directory:
# mkdir -p results_hqq_8b_wikitext
# sbatch scripts/run_llama31_8b_wikitext_a100.sh
# Optional first argument: a previous resolved_experiment_config.json for replay.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
spec="${1:-configs/generalization/llama31_8b_wikitext.json}"
bash scripts/launch_generalization.sh "$spec" results_hqq_8b_wikitext
