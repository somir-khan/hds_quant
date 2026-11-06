#!/bin/bash
# Shared launcher. Called from the project root by the two separate SLURM jobs.
set -euo pipefail
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Project Python unavailable; set HQQ_PYTHON or submit beside venv/bin/python." >&2
    exit 1
fi
spec="${1:?configuration JSON required}"
root="${2:?result root required}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; print(torch.__version__, torch.cuda.get_device_name(0))'
name="$("$PYTHON_BIN" -c 'import json,sys; from hds_quant.run_generalization import validate_spec; s=json.load(open(sys.argv[1])); validate_spec(s); print(s["experiment_id"])' "$spec")"
out="$root/job_${SLURM_JOB_ID:-manual}_${name}"
"$PYTHON_BIN" -m hds_quant.run_generalization --config "$spec" --output-dir "$out"
"$PYTHON_BIN" scripts/summarize_limited_labels.py --input-dir "$out"
echo "Generalization experiment completed: $out"
