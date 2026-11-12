#!/bin/bash
#SBATCH --job-name=hqq_frozen_c4
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=13
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=08:00:00
#SBATCH --array=0-2%1
#SBATCH --output=results_hqq_frozen_replication_c4/frozen_%A_%a.log
#SBATCH --error=results_hqq_frozen_replication_c4/frozen_%A_%a.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
bash scripts/launch_frozen_replication.sh c4 "${1:-}"
