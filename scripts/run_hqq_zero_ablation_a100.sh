#!/bin/bash
#SBATCH --job-name=hqq_zero_ablation
#SBATCH --partition=gpu2
#SBATCH --account=loni_driftaudit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=results_hqq_zero_ablation/zero_%j.log
#SBATCH --error=results_hqq_zero_ablation/zero_%j.err

# Before submitting from the project root:
# mkdir -p results_hqq_zero_ablation
# sbatch scripts/run_hqq_zero_ablation_a100.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
if [ ! -f hds_quant/run_experiment.py ] || [ ! -f tests/test_hqq_zero_policy.py ]; then
    echo "Submit from the updated project root containing hds_quant/, tests/, scripts/, and venv/." >&2
    echo "Merge the update folder's contents into that root before submitting." >&2
    exit 1
fi
# Use the project's interpreter explicitly: batch shells may have no python
# on PATH. Do not silently use an unrelated system Python or parent checkout.
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Python interpreter not found or not executable: $PYTHON_BIN" >&2
    echo "Merge the update into the existing project beside venv/ and submit there." >&2
    echo "For another environment, export HQQ_PYTHON=/absolute/path/to/bin/python before sbatch." >&2
    exit 1
fi
export PATH="$(dirname "$PYTHON_BIN"):$PATH"
echo "Project directory: $PWD"
echo "Python interpreter: $PYTHON_BIN"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA is required"; print(torch.__version__, torch.cuda.get_device_name(0))'
"$PYTHON_BIN" -m pytest tests/test_hqq_zero_policy.py -k packed_policy_parity -q

# Only zero-point rounding at 4 bits changes. Each policy gets fresh labels in
# its own directory; do not copy the earlier pilot cache into these folders.
for policy in floating round4; do
    out="results_hqq_zero_ablation/$policy"
    mkdir -p "$out/cache"
    "$PYTHON_BIN" -m hds_quant.validate_hqq \
        --device cuda --axes 0 --group-sizes 128 --bits 4 \
        --hqq-zero-policy "$policy" --max-eval-chunks 2 \
        --output "$out/packed_parity_diagnostic.json"
    "$PYTHON_BIN" -m hds_quant.run_experiment \
        --model-id meta-llama/Llama-3.2-1B-Instruct \
        --eval-dataset wikitext2 --quantizer hqq --hqq-axis 0 \
        --hqq-zero-policy "$policy" --quant-group-size 128 \
        --residual-anchor int4_aligned --coord-binding salience_bucket \
        --readout-grouping per_bit --target-transform signed_log1p \
        --seeds 42 --candidate-bits 1 2 3 4 --budgets 3.0 3.5 \
        --estimators HDS-Hybrid Coord-Rank-Hist Oracle-Loss \
        --n-calib-seqs 64 --calib-seq-len 1024 --max-encoded-tokens 4096 \
        --eval-seq-len 2048 --batch-size 8 \
        --cache-dir "$out/cache" --output-dir "$out" --run-tag "$policy"
done
"$PYTHON_BIN" scripts/compare_hqq_zero_policies.py --input-dir results_hqq_zero_ablation
echo "Zero-point ablation completed successfully."
