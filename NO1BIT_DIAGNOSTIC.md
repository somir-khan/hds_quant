# Allocation-only diagnostic: forbid 1-bit choices

This is a separate experiment, implemented entirely in new files. It reads the
completed `limited_labels.json`; it does not change the limited-label runner,
source results, estimator implementation, or original quantization settings.

## Run using Git and the existing environment

Commit/push the new files, then update the existing LONI project. From that
project containing `venv/` and the project-root `.env`:

```bash
git pull --ff-only
mkdir -p results_hqq_no1bit_diagnostic
sbatch scripts/run_no1bit_diagnostic_a100.sh
```

The script sources `scripts/load_hf_env.sh` before starting the explicit project
Python interpreter. It respects HF_HOME and HQQ_PYTHON from the existing setup.
It requests one A100 for one hour. No calibration probes or regressions are run.

Default source:
`results_hqq_limited_labels/job_1163862/limited_labels.json`.

Default **focused** scope: seed 100, K=8, budget 3.0, all three saved methods.
This targets the observed HDS-Hybrid outlier, retaining the two controls. The
SLURM script replays all three original allocations before reporting results.

Override the source without editing code:

```bash
sbatch scripts/run_no1bit_diagnostic_a100.sh /absolute/path/to/limited_labels.json
```

To extend to every trial and budget in the saved source:

```bash
DIAG_SCOPE=all sbatch scripts/run_no1bit_diagnostic_a100.sh
```

Focused and full-scope outputs go into different job-specific directories:
`results_hqq_no1bit_diagnostic/job_<JOBID>_<scope>/`.
The Python runner refuses to reuse an existing output directory. Every source
file is hashed before execution and checked again afterward for modification.

## What is held fixed

The original checkpoint, quantizer configuration, calibration block selection,
measured labels, held-out predictions, cost mapping, and test evaluation
protocol are taken from the source. Labels and predictions are joined to
reconstruct the original monotone costs using the original marginal precisions
{1,2,3}; the reconstructed objective must match the saved original objective.
Only the allocation candidate set changes from {1,2,3,4} to {2,3,4}. Costs of
retained choices are unchanged. The restricted solve must match the original
integer bit-parameter payload exactly, not just approximately equal average bits.

If an original allocation has no 1-bit projections, that exact assignment is
retained as an unchanged control. It is not re-solved to avoid unrelated ILP
tie-breaking changes. The summary explicitly identifies evaluations replayed
on the current model versus copied from an unchanged source assignment. The
SLURM launcher uses `--replay-all`, so its controls are replayed as well.

The direct Python CLI defaults to all source trials and replays affected
allocations; use `--replay-all` to replay unchanged controls too. Filters can be
set with `--seeds`, `--k-layers`, `--methods`, and `--budgets`.

## Reproducibility checks

* Requires a complete `true_limited_labels_v1` source and the original candidate set.
* Rejects missing, overlapping, non-finite or unexpected prediction/label pairs.
* Confirms labels correspond precisely to the selected calibration blocks.
* Checks original objective values and payload metadata.
* Checks effective HQQ settings against the source.
* Replays FP16 and uniform 3-bit, including per-chunk NLL and token counts.
* Replays the original allocation before evaluating each affected intervention.
* Default replay tolerance is absolute NLL 1e-5, applied to pooled and per-chunk
  NLL. A mismatch stops the diagnostic; it does not silently compare changed environments.
* Records source SHA-256, evaluation-token SHA-256, source configuration,
  resolved model revision when available, package versions, and relevant source-code hashes.

## Outputs and interpretation

* `diagnostic.json`: source provenance, verification results, old/new allocations,
  original 1-bit projection names, all changed projections, and evaluation losses.
  Check `complete: true` before interpreting a finished experiment.
* `comparisons.csv`: paired results, percentage changes, one-bit and change counts.
* `SUMMARY.txt`: per-seed comparisons and across-seed means when applicable.

Negative PPL change means the restricted allocation is better. An unchanged
control should have zero change within the replay tolerance. A focused run uses
one seed, so it does not estimate seed variability. No calibration efficiency is
claimed by this diagnostic: its timings cover only reallocation and evaluation.

Forbidding 1-bit choices can change several other projections to preserve the
budget. The intervention tests the effect of the constraint and resulting
reallocation; it is not proof that one particular projection alone caused a loss.

If a job stops, partial `diagnostic.json` checkpoints and the error log remain.
Use a new job/output directory for another run rather than mixing attempts.
