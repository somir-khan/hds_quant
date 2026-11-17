# Frozen ARR stages 1 and 2 — v1

This is a separate follow-up to the completed coverage study. It does not edit
historical code or results, launch jobs automatically, or install dependencies.
The frozen declaration is `configs/arr_stage12/frozen_v1.json`.

## Questions and decisions frozen before this run

- **A:** Are 1-bit probes redundant for the existing final {2,3,4} allocator?
  Refit the old {1,2,3}-probe models from saved features, then compare with models
  trained only on {2,3}. Keep the old `[bit==1, bit==2]` column geometry: the
  first indicator is identically zero in the reduced model. It requires no
  1-bit measurement. This preserves Hist/HDS metadata weighting, which otherwise
  changes when a marginal precision is removed. Compare predictions, processed
  costs and exact assignments, and separately replay the saved source models.
  Training labels fall from 42/168 to **28/112**. This stage does not re-time.
- **C:** For each fixed K/policy, compare model predictions everywhere with
  retaining the acquired projections' measured costs. Both versions use the
  same fit and labels. Predictions at acquired positions are in-sample model
  predictions, explicitly not out-of-fold accuracy measurements. This isolates
  direct retention conditional on the training set; it is not a pure causal
  comparison of training-set size across non-nested K sets.
- **D:** Add label-free `Residual-MSE` (recover `max(exp(logMSE)-1e-8,0)` from the
  saved FP32 log feature, not exact original MSE), `Family-Depth-Ridge` (seven
  family indicators + normalized depth, squared depth, first/last indicators),
  and `Meta-Depth-Ridge` (the nine existing metadata values + these four depth
  features). Both depth models use ordinary standardized, per-bit ridge with
  the existing signed target transform, alpha grid and block-grouped CV, no
  wide-vector weighting. They have retention and prediction-only variants.
  Residual-MSE has no measured-cost overwrite or supervised normalization.
  Raw costs across families are used as declared, without test-score tuning.
- **E:** One exploratory case only: **Llama-3.2-1B-Instruct, WikiText-2, seed42**.
  Complete its true 2/3-bit bank; replay endpoints; compare original frozen
  allocations to full-label raw and processed additive references; evaluate
  raw-singleton composition residuals on calibration tokens; measure all
  subsets of the first block's gate/up/down at 2 bits in a 4-bit context.

Both payload budgets 3.0 (secondary) and 3.5 (primary) are retained. All quality
allocations have exact parameter-weighted payload and final-down4 protection.
The existing preserve-already-protected rule is reused. Original family tie
sweeps are not repeated: C/D concern the regression controls and two structural
anchors only. None of A/C/D imports test scores to choose a model or allocation.

This is **not** A's fresh standalone timing, B's 1B same-K grid, intermediate
budgets, new-family validation, or an AMQ rerun. No new timing claim follows
from planning, scoring time or incremental bank completion.

## Source cases and isolation

Default source roots relative to the project root on LONI:

```
results_hqq_coverage_wikitext_quality/job_1177435_{0,1,2}_seed{42,100,2026}_v1
results_hqq_coverage_c4_quality/job_1177436_{0,1,2}_seed{42,100,2026}_v1
results_hqq_1b_focused/job_1182854_both_v1
```

The first two lines mean paired index/seed values, not a Cartesian product.
Their complete quality directories, including `feature_bank.pt`, are needed
for controls. The focused 1B root also needs its parent manifests, geometry,
dataset manifests, per-seed resolved config and quality files.
Sources are hash-checked and rechecked at completion; outputs refuse an existing
directory. Data/token pins, geometry and effective HQQ settings are checked
before GPU scoring. Saved feature banks are loaded on CPU with
`torch.load(..., weights_only=True)`.

| Array index | Model | Dataset | Seed |
|---|---|---|---|
| 0,1,2 | 8B | WikiText-2 | 42,100,2026 |
| 3,4,5 | 8B | C4 | 42,100,2026 |
| 6,7,8 | 1B | WikiText-2 | 42,100,2026 |
| 9,10,11 | 1B | C4 | 42,100,2026 |

The default controls submission runs **index 6 only**. It requests the existing
gpu2/A100/account/CPU/memory settings. Scheduling allowances are not runtime
estimates. No full-array experiment is submitted implicitly.

## Offline check first

From the cluster project root, after transferring the code through Git:

```bash
git pull
mkdir -p results_hqq_arr_controls results_hqq_arr_e_diagnostic
bash scripts/run_arr_stage12_offline.sh \
  --case-index 6 \
  --output-dir results_hqq_arr_controls/offline_case6_v1
```

This uses the existing environment and CPU only: no HF model, dataset download,
or held-out scoring. Run it on an appropriate CPU allocation; it performs ridge
fits and ILPs. Inspect `parity.json`. `passed=true` requires prediction/cost
agreement within 1e-8, legacy refit/saved prediction agreement within 1e-6,
and exact agreement of both reduced/legacy and legacy/saved allocations.
A failed parity is recorded, not hidden by substituting saved assignments.

Optional E declaration check (no HF model or feature-bank deserialization):

```bash
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
"$PYTHON_BIN" -m hds_quant.run_arr_stage12 --mode diagnostic --plan-only \
  --case-index 6 --output-dir results_hqq_arr_e_diagnostic/preflight_case6_v1
```

On this source, E needs **98 new labels** from seven missing blocks plus
**28 replay labels** from blocks 0 and 15. Those overlap checks must pass at
1e-6. It reuses known source labels and never describes incremental collection
as independently timed full-bank calibration.

For local Windows checks, use the existing `venv/Scripts/python.exe` from `code`
with `--results-base D:/TinyML`; downloaded results are siblings of `code`.
On LONI the default base is `$PWD`. A custom location is supported on every
launcher through `--results-base /absolute/path/to/results-parent`.

## GPU pilot submissions

After checking offline parity, submit the controls pilot:

```bash
sbatch scripts/run_arr_controls_a100.sh
```

This repeats the CPU check inside the job, then scores C/D only if A passes.
It evaluates duplicate assignments once. Existing baseline and assignment
scores are replay checks, not selection criteria. Inspect the pilot before
expanding the controls matrix:

```bash
# Optional later expansion: the other eleven cases, without repeating index 6.
sbatch --array=0-5,7-11%1 scripts/run_arr_controls_a100.sh
```

The second stage is the single-case E diagnostic:

```bash
sbatch scripts/run_arr_e_diagnostic_a100.sh
```

Run E after reviewing the controls pilot; no dependency or job is created by
these instructions. E loads only 1B WikiText seed42, uses the existing .env via
`scripts/load_hf_env.sh`, and selects `HQQ_PYTHON` or `$PWD/venv/bin/python`.
No environment is created and no credentials are printed.

## Diagnostic interpretation

For each original frozen assignment and each full-label additive reference:

```
actual = calibration_NLL(assignment) - calibration_NLL(all4)
true_sum = sum(raw true singleton delta_NLL at assigned precision)
predicted_sum = sum(raw estimated singleton delta_NLL)
aggregate_prediction_error = predicted_sum - true_sum
composition_residual = actual - true_sum
processing_shift = processed_allocator_objective - predicted_sum
```

These are exact arithmetic identities on the measured bank, not a complete
causal attribution of failure. Opposite errors can cancel. Report held-out
quality separately. Prediction metrics exclude acquired blocks and are reported
per precision and within each family; constant-input rank correlations are null.
The full-label raw reference preserves signed labels, while the processed
reference uses the existing nonnegative/monotone rules. Neither is an oracle
for true joint model quality.

Joint probes use eight subsets: reference, three singletons, three pairs and
the triple. Reference/singletons reuse measured labels; four joint calibration
evaluations are new. They intentionally vary payload, remain in the all4
context, do not score held-out data, and do not isolate the deployed 3-bit
context or the final-down protection effect.

## Outputs and summary

Controls: `run_manifest.json`, `parity.json`, frozen `selection_plan.json`,
`estimates.json`, GPU `diagnostic.json`, `comparisons.csv`,
`retention_ablation.csv`, data manifest, copied declaration.

E: declaration/preflight, `full_label_bank.json`, frozen `selection_plan.json`,
`diagnostic.json`, `composition_residuals.csv`,
`heldout_block_prediction_metrics.csv`, `joint_first_block.csv`, provenance.

Only completed GPU manifests support quality findings. Offline manifests say
`mode=offline`; a diagnostic `--plan-only` remains `complete=false`.

```bash
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
"$PYTHON_BIN" scripts/summarize_arr_stage12.py \
  results_hqq_arr_controls/job_CONTROLS_JOBID_6_v1 \
  results_hqq_arr_e_diagnostic/job_E_JOBID_1b_wikitext_seed42_v1 \
  --output-dir results_hqq_arr_summary/review_stage12_v1
```

Replace the two JOBID placeholders with actual job numbers. Duplicate case/mode
inputs and incomplete jobs are rejected. Summary rows retain case, method,
variant and evidence type; no seed replication or runtime is inferred.

## Local validation before handoff

- All twelve real saved cases passed the offline parity gate, including exact
  reduced-versus-legacy and legacy-versus-saved primary assignments.
- The default controls pilot produced 46 roles / 31 distinct candidate plans.
- The real one-case E preflight confirmed 98 new + 28 replay labels.
- Eight new tests cover legacy parity, retention, label-free/depth controls,
  signed error decomposition, held-out-block metrics and synthetic end-to-end E
  orchestration. Nine existing coverage/focused tests also passed.
- Bash syntax and Python AST checks passed. No cluster/GPU experiment was run
  locally. Synthetic E tests do not replace a real HQQ/CUDA run.

Temporary local checks were written outside the Git repository under
`D:/TinyML/tmp/arr_stage12_*`; these are validation artifacts, not new GPU results.
