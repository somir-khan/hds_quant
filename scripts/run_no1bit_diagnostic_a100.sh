#!/bin/bash
#SBATCH --job-name=hqq_no1bit_diag
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results_hqq_no1bit_diagnostic/no1bit_%j.log
#SBATCH --error=results_hqq_no1bit_diagnostic/no1bit_%j.err

# Submit from the main project:
# mkdir -p results_hqq_no1bit_diagnostic
# sbatch scripts/run_no1bit_diagnostic_a100.sh
# Optional: DIAG_SCOPE=all sbatch scripts/run_no1bit_diagnostic_a100.sh /path/to/limited_labels.json
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f scripts/load_hf_env.sh ]; then
    echo "Missing scripts/load_hf_env.sh; update the main Git checkout before submitting." >&2
    exit 1
fi
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Project Python unavailable: set HQQ_PYTHON or submit beside venv/bin/python." >&2
    exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
source_results="${1:-${SOURCE_RESULTS:-results_hqq_limited_labels/job_1163862/limited_labels.json}}"
test -f "$source_results"
scope="${DIAG_SCOPE:-focused}"
filters=()
case "$scope" in
    focused) filters=(--seeds 100 --k-layers 8 --budgets 3.0) ;;
    all) ;;
    *) echo "DIAG_SCOPE must be focused or all" >&2; exit 1 ;;
esac
out="results_hqq_no1bit_diagnostic/job_${SLURM_JOB_ID:-manual}_${scope}"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -m hds_quant.run_no1bit_diagnostic \
    --source "$source_results" --output-dir "$out" --replay-all "${filters[@]}"
echo "Allocation-only diagnostic completed: $out"
