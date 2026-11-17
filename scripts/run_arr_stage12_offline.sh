#!/bin/bash
# Run on a CPU allocation or in the existing environment; no HF model is loaded.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then echo "Set HQQ_PYTHON to your existing environment." >&2; exit 1; fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
"$PYTHON_BIN" -m hds_quant.run_arr_stage12 --mode offline "$@"
