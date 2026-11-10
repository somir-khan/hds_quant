# Uniform-3 matched-swap control before replication

This is a separate post-hoc control for the completed seed42 fixed4 diagnostic.
It evaluates all three previously observed two-projection swaps on Uniform-3:

| Paired learned method | Upgrade | Compensating downgrade |
| --- | --- | --- |
| HDS | Block 31 down: 3 to 4 | Block 28 gate: 3 to 2 |
| Coord-Hist | Block 31 down: 3 to 4 | Block 14 gate: 3 to 2 |
| Coord-Exact | Block 31 down: 3 to 4 | Block 1 gate: 3 to 2 |

All other target projections remain at 3 bits. Down/gate matrices have equal
parameter counts in this model, and the plan checks exact integer payload
preservation. Each variant remains exactly 3.0 target-weight payload bits.

These gate choices come from the previously observed learned reallocations.
They are frozen diagnostic controls, not an independent feature-free algorithm
or a rule to carry unchanged into other seeds/models. Report every row; do not
select the best gate by held-out perplexity. Comparing a constrained learned
assignment to its corresponding uniform swap measures what the remaining learned
allocation adds in that fixed context. It does not isolate the causal effect of
the features alone.

## Run using the existing environment

Commit and push the code locally. On LONI:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
mkdir -p results_hqq_uniform_swap
sbatch scripts/run_uniform_swap_a100.sh
```

The launcher sources your existing project `.env` through
`scripts/load_hf_env.sh` and uses `HQQ_PYTHON` or `$PWD/venv/bin/python`.
No new dependencies or credentials are needed. It requests one A100, 13 CPUs,
96 GB host RAM and a two-hour scheduling limit (not a runtime estimate).

Both completed source jobs must remain available:

```text
results_hqq_amq_budgetcap_20round_v2/job_1173223_llama31_8b_wikitext_amq_budgetcap_20round_v2/
results_hqq_amq_fixed4_diagnostic/job_1174274_llama31_8b_wikitext_fixed4_v1/
```

The first source supplies pinned revisions, token hashes and the saved learned
allocations. The second supplies its validated frozen plan and actual fixed4
scores. Source hashes, assignment identities, original-score replay and the
observed two-projection changes are checked before GPU evaluation. Different
locations for these same jobs can be passed in that order:

```bash
sbatch scripts/run_uniform_swap_a100.sh /path/to/20round_job /path/to/fixed4_job
```

## Scope and outputs

The runner replays FP16 and Uniform-3 on the same 141 evaluation chunks, then
evaluates the three new swap assignments. One assignment evaluation includes
all batches/chunks; it is not literally one model forward call. Fresh HQQ
quantization and the common evaluator are used. Original weights are restored
by the evaluator, including on failure. No teacher-logit bank, AMQ proxy cache,
sensitivity labels, feature extraction, regression or search is needed.

There is no calibration-based winner selection and no inference/calibration-speed
claim. The held-out outcomes are post-hoc comparisons on the same test stream.
Existing learned scores are reused with provenance and marked as source scores;
they are not presented as new independent observations.

```text
results_hqq_uniform_swap/
  uniform_swap_<JOBID>.log / uniform_swap_<JOBID>.err
  job_<JOBID>_llama31_8b_wikitext_uniform_swap_v1/
    experiment_config.json
    selection_plan.json
    diagnostic_manifest.json
    data_manifest.json
    diagnostic.json
    comparisons.csv
    SUMMARY.txt
```

Require `complete: true` in both the manifest and result, two passing baseline
replay checks, and three evaluated candidates. The comparison table includes
Uniform-3, swapped Uniform-3, original learned and fixed4 learned PPL for each
pair. Negative `learned_fixed4_vs_swap_ppl_percent` favors the learned assignment.
Failed GPU evaluation leaves incomplete status and available partial rows.
Output directories are never reused; there is no automatic resume.

Optional offline plan verification (existing Python packages, no GPU/model/data
download, fresh output directory required):

```bash
venv/bin/python -m hds_quant.run_uniform_swap \
  --config configs/amq_diagnostics/llama31_8b_wikitext_uniform_swap_v1.json \
  --output-dir results_hqq_uniform_swap/plan_only_seed42 --plan-only
```

## Decisions after this control

Inspect all three pairs, then proceed with a frozen seed100 replication before
deciding on seed2026 and C4. The seed42 diagnostic rule is not a prospective
protection policy. Per-seed screening sets obtained in AMQ runs are available for
matched-space controls, but a standalone HDS method using them must account for
their acquisition cost even when that data already exists in a comparison job.

Boundary-inclusive label acquisition is a separate future experiment, after the
frozen replication. One genuinely additional block adds 7 projections times 3
non-reference probes = 21 labels. The observed 574.13 s / 84 labels suggests
about 143.5 s per added block, before other overheads; measure actual time.
Including the last block measures its sensitivities but does not guarantee an
additive allocator will assign 4 bits. Deduplicate a boundary block already in
the selected set. Do not combine this change with the frozen replication.

## Timing correction for the research record

The 96.8-minute anchor checkpoint from job1173223 includes **250 initial-design
candidate evaluations**, precomputation, screening and anchors. Call it initial
design plus anchors. A standalone anchor-only method has not been measured.
The recorded precompute (78.48 s), screening (2970.37 s) and two-anchor increment
(23.28 s) sum to about 51.2 minutes. This is an accounting estimate, not a new
measured standalone runtime. Consequently, a 1.6x speed claim against a supposed
97-minute anchor-only baseline is unsupported.

Tests:

```bash
python -m pytest tests/test_uniform_swap.py tests/test_amq_followup.py -q -p no:cacheprovider
```
