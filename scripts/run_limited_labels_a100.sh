#!/bin/bash
#SBATCH --job-name=hqq_limited_labels
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=results_hqq_limited_labels/limited_%j.log
#SBATCH --error=results_hqq_limited_labels/limited_%j.err

# mkdir -p results_hqq_limited_labels
# sbatch scripts/run_limited_labels_a100.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ -f scripts/load_hf_env.sh ]; then
    source scripts/load_hf_env.sh
fi
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Set HQQ_PYTHON to your Python executable, or submit beside venv/bin/python." >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
# Equal CPU threading budgets for the regressions; randomized method order per seed/K.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name(0))'
out="results_hqq_limited_labels/job_${SLURM_JOB_ID:-manual}"
"$PYTHON_BIN" -m hds_quant.run_limited_labels \
    --seeds 42 100 2026 --k-layers 2 4 8 --budgets 3.0 3.5 \
    --methods HDS-Hybrid Coord-Rank-Hist Scalar-MSE-Ridge \
    --candidate-bits 1 2 3 4 --output-dir "$out"
"$PYTHON_BIN" scripts/summarize_limited_labels.py --input-dir "$out"
echo "Limited-label experiment completed: $out"
