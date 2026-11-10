# Saved-assignment objective diagnostic and fixed4 constraint check

These are two independent, bounded diagnostics of completed WikiText job 1173223.
They use separate configurations, launchers and result roots. Both use your
existing project `.env` through `scripts/load_hf_env.sh`, then `HQQ_PYTHON` or
`$PWD/venv/bin/python`. No new dependencies or token entry are required.

## 1. What each job answers

### Objective diagnostic

The fixed assignment list contains HDS/Hist/Exact winners at both budgets, the
20-round AMQ winners, both AMQ starting anchors, and uniform 3/4-bit references.
Duplicate assignments are evaluated only once and retain all their role labels.
For the supplied source, expect 11 unique assignments.

For every assignment it reports calibration JSD against the floating teacher,
calibration NLL/PPL, and held-out NLL/PPL. JSD uses the same FP32 arithmetic and
epsilon-clamped formula as the AMQ search. Cached-proxy calibration NLL is checked
against fresh shared-HQQ quantization (absolute tolerance 1e-6). Where an archived
JSD exists, it must reproduce within 1e-5. Historical held-out NLL, chunk NLLs and
token counts are replay-checked (NLL tolerance 1e-6).

The candidate list is frozen before GPU evaluation and uses saved assignments,
not a new ranking by test perplexity. There is no AMQ search, sensitivity
screening, label acquisition, feature extraction or regression. Held-out replay
is a reproducibility check, not a new independent test set.

Interpret JSD and NLL together. If a learned assignment has lower JSD than the AMQ
winner, check whether it satisfies AMQ's fixed4 rule before calling it a missed
feasible candidate. If objective rankings differ, that is evidence of metric
disagreement among these saved assignments, not proof about every possible model.

### Fixed4 constraint check

This reconstructs the original mixed measured/predicted sensitivity costs for
each HDS/Hist/Exact trial. The reconstructed objective must match the saved
source objective. It then applies the fixed4 set recorded by AMQ screening
(currently `model.layers.31.mlp.down_proj`) and reallocates the remaining weights
under {2,3,4}, preserving the original **integer parameter-weighted payload**.

The free projections use the existing exact-payload allocator after subtracting
the fixed weights' payload from the budget. The merged assignment is verified
against the original integer payload; infeasible payloads fail explicitly.
Original labels, predictions, target transform and monotonic cost processing are
retained. No new fitting or loss labels enter the allocation.

Both 3.0 and 3.5 budgets are checked for each learned method: six comparisons.
An original assignment already satisfying fixed4 is retained exactly, avoiding
an unrelated ILP tie-breaking change. All three original 3.5-bit assignments
should therefore be unchanged controls. At 3 bits, the forced upgrade can cause
compensating changes elsewhere; the complete change list is saved. The effect is
an allocation-level intervention, not the isolated causal effect of one matrix.

This job also scores original and constrained assignments with calibration JSD,
calibration NLL and held-out NLL/PPL. AMQ winners and uniform 3/4 references are
included for context. Expect 13 unique assignments for the supplied source
(deduplication may reduce this if constrained assignments coincide).

The AMQ 3-bit-cap winner uses about 2.964 bits. The reallocated learned controls
remain at exactly 3.0, so the diagnostic does not remove that payload difference.
The 3.5-bit comparison matches payload and the current fixed4 rule.

## 2. Source files required on LONI

Default source job:

```text
results_hqq_amq_budgetcap_20round_v2/
  job_1173223_llama31_8b_wikitext_amq_budgetcap_20round_v2/
```

Keep its `experiment_config.json`, `resolved_experiment_config.json`,
`experiment_manifest.json`, `data_manifest.json`, `limited_labels.json`, and
`amq_seed_42/archive.json`. Source files are read and hashed, never overwritten.
Validation checks completed status, pinned revisions, quantizer protocol,
archive geometry, payload accounting, screening constraints and saved AMQ
selection. Config and source-file hashes plus current code/package versions
are recorded in each diagnostic manifest.

## 3. Submit using your existing environment

Commit and push these code changes through your normal Git workflow. On LONI:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only

mkdir -p results_hqq_amq_objective_diagnostic
sbatch scripts/run_amq_objective_diagnostic_a100.sh

mkdir -p results_hqq_amq_fixed4_diagnostic
sbatch scripts/run_amq_fixed4_diagnostic_a100.sh
```

The jobs can be submitted independently; neither reads the other's output.
Each requests one A100, 13 CPUs, 128 GB host RAM and a four-hour scheduling
limit. These are limits, not runtime promises. Each builds its own host proxy
cache and disk-backed teacher once (roughly 39 GiB proxies and 15.6 GiB teacher
scratch plus other process memory). Teacher scratch is removed after evaluation,
including exceptions, and original weights are restored.

If the same source job is stored at a different cluster path, pass it as the
first argument to either dedicated SLURM launcher:

```bash
sbatch scripts/run_amq_objective_diagnostic_a100.sh /path/to/original/job_directory
sbatch scripts/run_amq_fixed4_diagnostic_a100.sh /path/to/original/job_directory
```

## 4. Output and completion checks

```text
results_hqq_amq_objective_diagnostic/
  objectives_<JOBID>.log / objectives_<JOBID>.err
  job_<JOBID>_llama31_8b_wikitext_objectives_v1/

results_hqq_amq_fixed4_diagnostic/
  fixed4_<JOBID>.log / fixed4_<JOBID>.err
  job_<JOBID>_llama31_8b_wikitext_fixed4_v1/
```

Both job directories contain:

- `experiment_config.json`: diagnostic config snapshot.
- `selection_plan.json`: frozen assignments, roles, source hashes, fixed4 set,
  and exact-payload reallocation plans where applicable.
- `diagnostic_manifest.json`: provenance, baseline verification and completion.
- `data_manifest.json`: reproduced token-stream identity.
- `diagnostic.json`: all calibration/test results and replay/parity checks.
- `objective_scores.csv`: one row per unique assignment with all role labels.
- `comparisons.json`: paired fixed4 effects; empty for objective-only mode.
- `SUMMARY.txt`: readable scores and constraint effects.

The constraint job additionally writes `constraint_comparisons.csv`, with
before/after payload, predicted costs, changed counts, PPL and loss differences.
Differences are **after minus before**; negative loss/PPL change favors fixed4.

Require `complete: true` in both diagnostic JSON and manifest, passing parity
and historical replay checks, and six fixed4 comparisons. Review changed
projections in the plan, especially the unchanged 3.5-bit controls.

Failures leave a manifest with `complete: false` and an error, plus available
partial rows in `amq_seed_42/diagnostic_partial.json`. Existing output directories
are refused; partial output is not automatic resume state. Re-submit with a new
job ID after diagnosing a failure.

Both jobs are post-hoc diagnostics of one calibration seed. Their elapsed time
is not a new calibration-speed result. Do not use their test outcomes to choose
a preferred assignment for a claimed independent benchmark.

## 5. Optional offline plan inspection

These commands use the existing Python packages but load no model, download no
data, and need no GPU. Use fresh output directories each time:

```bash
venv/bin/python -m hds_quant.run_amq_followup \
  --config configs/amq_diagnostics/llama31_8b_wikitext_objectives_v1.json \
  --output-dir results_hqq_amq_objective_diagnostic/plan_only_1173223 --plan-only

venv/bin/python -m hds_quant.run_amq_followup \
  --config configs/amq_diagnostics/llama31_8b_wikitext_fixed4_v1.json \
  --output-dir results_hqq_amq_fixed4_diagnostic/plan_only_1173223 --plan-only
```

A plan-only manifest records `plan_only: true`, `plan_complete: true` and
`complete: false`. It must not be mistaken for completed GPU validation.

## 6. Verification

```bash
python -m pytest tests/test_amq_followup.py tests/test_amq_comparison.py tests/test_amq_coverage.py tests/test_no1bit_diagnostic.py -q -p no:cacheprovider
```

Tests cover exact-payload reallocation, preserving unchanged assignments,
infeasible constraints, corrupt source costs/archive payloads, plan deduplication,
source immutability, real small-model HQQ objective scoring, historical replay
failures and weight restoration. Full 8B CUDA validation occurs in the two jobs.
