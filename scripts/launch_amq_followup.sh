#!/bin/bash
# Shared environment and provenance entry point for separate follow-up jobs.
set -euo pipefail
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
    echo "Project Python unavailable; set HQQ_PYTHON or submit beside venv/bin/python." >&2
    exit 1
fi
spec="${1:?config required}"
root="${2:?result root required}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
name="$("$PYTHON_BIN" -c 'import json,sys; from hds_quant.amq_followup_plan import validate_spec; s=json.load(open(sys.argv[1])); validate_spec(s); print(s["experiment_id"])' "$spec")"
extra=()
if [ -n "${3:-}" ]; then extra+=(--source-dir "$3"); fi
out="$root/job_${SLURM_JOB_ID:-manual}_${name}"
"$PYTHON_BIN" -m hds_quant.run_amq_followup --config "$spec" --output-dir "$out" "${extra[@]}"
