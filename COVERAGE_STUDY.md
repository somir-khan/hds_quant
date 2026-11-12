# Five-arm label-coverage study, version 1

This is a new study, separate from the frozen K=4 replication. Its primary comparison is **stratified acquisition plus protection versus boundary-inclusive acquisition plus the same protection**, at matched label counts. It collects fresh calibration labels. It does not alter the previous result folders or fold a new pooling rule into the earlier replication.

## Declared matrix

- Model: the same pinned `meta-llama/Llama-3.1-8B-Instruct` revision.
- Datasets: pinned WikiText-2 and C4 selections, with their existing disjoint calibration/evaluation text and token hashes.
- Calibration seeds: **42, 100, 2026** on both datasets.
- Acquired blocks: **K=2, 4, 8**, corresponding to **42, 84, 168** labels per fit.
- Acquisition policies: `stratified` and `boundary_inclusive`.
- Primary arms: **Family-Mean, Family-Median, Scalar-Meta-Ridge, Coord-Rank-Hist, HDS-Hybrid**.
- Primary payload budgets: **3.0 and 3.5 bits**, using only 2/3/4-bit assignments.
- Protection: final-block `down_proj` at 4 bits for every arm and policy, with exact payload compensation.
- Family placement: depth ascending as primary, plus the same ten fixed SHA256 orders used in replication, for **both** pooling arms.

K=1, 2.5-bit budgets, metadata-plus-depth, Coord-Exact and new AMQ searches are outside this version. This keeps the primary study aligned with the agreed practical budgets. Structural references remain in the tables.

## Acquisition rule

`stratified` retains the existing policy: split block depth into K contiguous strata and select one block per stratum using the declared seed.

`boundary_inclusive` starts with that exact selection, replaces the first selected block with block 0, replaces the last selected block with block 31, and retains all K-2 interior selections. No extra labels are added.

- At K=2, the boundary set is exactly `[0,31]`.
- At K=4, seed 100 already selects `[0,8,19,31]`, so both policies are identical. Report this as a no-op control, not an independent acquisition improvement.
- The K=2/4/8 sets are **not nested**. Interpret the curves as a comparison of declared acquisition budgets and selections, not as sequentially adding labels to the same training set.

The acquisition plan is written before collection. Selection never uses held-out perplexity, newly observed sensitivity values or AMQ screening.

## Pooling and fitting

Family-Mean averages raw acquired delta-NLL separately by projection family and probe precision. Family-Median uses the median of those raw labels. Both retain the exact individual label on each acquired projection and predict only unmeasured projections. Both then use the existing nonnegative/monotone cost processing.

At K=2, the two-point median is defined using the identical arithmetic as the mean. The arms share the same allocation result when their cost banks are identical, avoiding a meaningless solver tie difference. No special interior-only fallback is needed.

The three feature arms retain their existing feature formulas, per-bit ridge fits, grouped inner CV, training-fold scaling, scaled signed-log target transform and alpha grid. HDS still uses 512 salience buckets, Hist 16; their normalization differences remain. This is a comparison of existing representation pipelines, not a newly matched encoding-only ablation.

The joint quality feature path computes residual magnitudes, Q99 and per-token rank frequencies once, then produces HDS, Hist and the exact nine metadata values. Standalone timing uses each arm's own extractor instead. Tests check that the shared and standalone feature values agree exactly.

Probe labels remain 1/2/3-bit single-projection interventions against a 4-bit background. Final allocations exclude 1 bit. Embeddings, norms and LM head remain outside the quantized target projection set. Payloads exclude packing overhead and quantizer metadata; this remains weight quantize/dequantize evaluation, not a packed inference benchmark.

## Protection and placement

For each arm/budget/ordering, build its unconstrained assignment first. If it already satisfies final-down4, preserve that entire assignment. Otherwise solve at the same exact integer payload with the final-down4 constraint. Family placement permutations preserve per-family precision counts and equal-cost/size groups, while keeping acquired projections and the constrained projection fixed.

The primary family ordering is depth ascending. Random order seeds are `17,29,43,59,71,89,101,127,149,173`, using SHA256 of `seed|projection_name`. Report their range and median without choosing a favorable ordering. The sensitivity remains conditional on the solver's chosen family precision counts. Orders are not calibration-seed replications.

## Two execution modes: quality and timing

**Quality mode:** one job per dataset/calibration seed covers all six K/policy cells and all five arms. It acquires the union of the predeclared block sets once, extracts joint features once, and masks the label bank separately for every fit. A K=2 fit sees exactly 42 labels, even though the physical study bank contains more. Identical acquisition sets share fits; identical cost banks share allocation results; identical assignments share PPL evaluations across roles.

The physical union contains 15 blocks for seed 42, 10 for seed 100 and 14 for seed 2026: 315, 210 and 294 labels respectively. **These are quality-study collection counts, not the label budget of an individual method.** Per-cell provenance records the actual 42/84/168 labels supplied to that method. No full-bank/oracle labels enter fitting.

Every allocation is frozen and hashed before held-out scoring. FP16, Uniform-3 and Uniform-4 are replay-checked against the pinned source. Any historical source allocation that independently matches a planned assignment is also replay-checked. The two protected simple patterns are measured references and are deduplicated rather than repeated across every K/policy role.

There are 300 primary-and-placement result roles per quality job before deduplication, plus four structural-reference roles. Across six quality jobs this is 1,800 study roles, not necessarily 1,800 unique PPL evaluations. The five primary arm curves themselves have 60 rows per job. Do not estimate evaluation cost from the five-arm count alone: random placements add roles, while identical assignments reduce actual work.

**Timing mode:** one job covers one dataset/seed/K/policy cell and runs all five arms in a fixed SHA256-derived order. Each arm independently recollects its labels, independently extracts its required features, fits, and selects its primary protected allocations. Label, feature and solver caches are not shared between arms. The timing mode includes both payload budgets but only the primary placement, not the ten sensitivity evaluations.

Timing excludes model/data loading, warmup, disk output and held-out PPL evaluation. `calibration_to_budget_sec` includes acquisition, features, fitting/cost construction and that budget's allocation; `calibration_wall_sec` includes both budget solves. Both are measured, not reconstructed from quality-stage component times. CUDA synchronization and existing peak-memory accounting are used.

The aggregate tool requires every timed primary assignment to match its quality-study counterpart before attaching the measured runtime to that PPL. A mismatch is an error to inspect, not an automatically accepted alternative tie solution.

## Existing environment and sources

All scripts source `scripts/load_hf_env.sh`, so they use the existing `.env`, credentials and cache settings. Python is `HQQ_PYTHON` when set, otherwise `$PWD/venv/bin/python`. No environment or package installation is performed.

The source directories provide pinned configuration, token hashes and historical baselines. **Their old labels and predictions are not used to fit the new coverage study.** Fresh overlapping labels are compared with the source labels as a calibration replay check.

Defaults, relative to the cluster checkout root:

```text
results_hqq_representation_controls/job_1169696_llama31_8b_wikitext_coordexact_lim_k4_v1
results_hqq_representation_controls/job_1171547_llama31_8b_c4_coordexact_lim_k4_v1
```

Each requires `experiment_config.json`, `resolved_experiment_config.json`, `experiment_manifest.json`, `limited_labels.json` and `data_manifest.json`. Supply another directory as the first SLURM script argument if needed. The completed duplicate C4 source job 1171551 is an acceptable alternative; do not combine duplicates as extra seeds.

## Run the quality study

Commit/push the new files locally, then update the cluster checkout. Run from the project root containing the existing environment and `.env`:

```bash
git pull
mkdir -p results_hqq_coverage_wikitext_quality results_hqq_coverage_c4_quality
sbatch scripts/run_coverage_quality_wikitext_a100.sh
sbatch scripts/run_coverage_quality_c4_a100.sh
```

Each quality array has indices `0=42`, `1=100`, `2=2026`. Each task covers all K values, both policies and all five arms. The default `%1` runs one seed at a time within each dataset array. Submitting both arrays can run two GPU jobs concurrently.

To check only one complete quality job first, use this **instead of** the full WikiText submission:

```bash
sbatch --array=0 scripts/run_coverage_quality_wikitext_a100.sh
```

Use the first job to check completion and validation, not to change the declared method after viewing PPL.

## Run standalone timing

Timing is a separate submission so the shared quality sweep cannot be confused with a method runtime benchmark:

```bash
mkdir -p results_hqq_coverage_wikitext_timing results_hqq_coverage_c4_timing
sbatch scripts/run_coverage_timing_wikitext_a100.sh
sbatch scripts/run_coverage_timing_c4_a100.sh
```

Each default timing array contains 18 tasks: three seeds × three K values × two policies. All five arms run independently within each task. Across both datasets the full timing grid is **36 cells / 180 standalone calibrations**. This is substantially more acquisition work than the shared quality phase; do not accidentally submit it twice.

Timing indices map as follows:

| Seed | K=2: stratified / boundary | K=4: stratified / boundary | K=8: stratified / boundary |
| --- | --- | --- | --- |
| 42 | 0 / 1 | 2 / 3 | 4 / 5 |
| 100 | 6 / 7 | 8 / 9 | 10 / 11 |
| 2026 | 12 / 13 | 14 / 15 | 16 / 17 |

For an initial timing subset at K=4 only, use these **instead of** the full timing submissions:

```bash
sbatch --array=2-3,8-9,14-15%1 scripts/run_coverage_timing_wikitext_a100.sh
sbatch --array=2-3,8-9,14-15%1 scripts/run_coverage_timing_c4_a100.sh
```

A subset supports timing claims only for its measured cells. The aggregate explicitly marks an incomplete timing matrix. If you later complete it, submit only the missing indices or select one intended job per cell during aggregation.

Every task requests one A100, 13 CPUs, 96 GB host memory and an eight-hour allowance. This is a scheduling limit, not a runtime estimate. Result roots and job folders are separate by dataset and mode. Existing folders are refused; failed jobs do not silently resume.

For a non-default source location:

```bash
sbatch scripts/run_coverage_quality_c4_a100.sh /absolute/path/to/c4_source_job
sbatch scripts/run_coverage_timing_c4_a100.sh /absolute/path/to/c4_source_job
```

## Offline acquisition preflight

```bash
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
"$PYTHON_BIN" -m hds_quant.run_coverage_study \
  --config configs/coverage_study/llama31_8b_c4_v1.json \
  --mode quality --seed 42 \
  --output-dir results_hqq_coverage_c4_quality/preflight_seed42_v1 \
  --plan-only
```

Add `--source-dir /absolute/path/to/source` if needed. For timing preflight add `--mode timing --k-layers 4 --policy boundary_inclusive` instead of `--mode quality`. No model or dataset is loaded. The output remains incomplete because fresh labels/features/allocations have not been acquired. Use a new output folder each time.

## Outputs and aggregation

Quality jobs contain:

```text
study_manifest.json          experiment_config.json
resolved_model_config.json   acquisition_plan.json
data_manifest.json           label_bank.json
feature_bank.pt              estimates.json
selection_plan.json          diagnostic.json
comparisons.csv              coverage_curves.csv
acquisition_pairs.csv        placement_sensitivity.csv
SUMMARY.txt
```

Timing jobs contain the same configuration/provenance setup plus `timing.json`, `standalone_timings.csv` and `SUMMARY.txt`, without a shared quality feature/label cache. Manifests record source/config/code hashes, Git revision/tracked changes, runtime packages, token provenance, effective HQQ settings and phase-specific artifact hashes. Require completed manifest and result flags. Partial data are not completed comparisons.

Aggregate completed quality jobs and any completed timing cells using the existing environment:

```bash
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
"$PYTHON_BIN" scripts/summarize_coverage_study.py \
  results_hqq_coverage_wikitext_quality/job_* \
  results_hqq_coverage_c4_quality/job_* \
  results_hqq_coverage_wikitext_timing/job_* \
  results_hqq_coverage_c4_timing/job_* \
  --output-dir results_hqq_coverage_summary/final_v1
```

If timing has not run, omit both timing globs. If reruns created duplicates, list the exact intended completed folders. The tool rejects duplicate cells, mixed code/package/quantizer snapshots, incomplete jobs and timing/quality assignment mismatches. It never selects jobs using PPL.

Outputs are `primary_curves.csv`, `paired_acquisition.csv`, `standalone_timing_with_quality.csv` when timing is supplied, and `input_manifest.json` with separate quality/timing completeness flags. Negative paired percentages favor boundary-inclusive acquisition. Preserve per-seed results, mark identical acquisition sets, and report the ten-order sensitivity rather than choosing its best member. The 1% practical margin is descriptive and does not establish statistical equivalence.

## Implementation separation

- Policies, pooling and allocation: `hds_quant/coverage_protocol.py`.
- Fresh collection, shared quality and independent timing: `hds_quant/coverage_engine.py`.
- Versioned entry point and provenance: `hds_quant/run_coverage_study.py`.
- Reporting and timing/quality parity: `hds_quant/coverage_summary.py`.
- Two dataset configs under `configs/coverage_study/` and four separate SLURM entry points.

The only extension to a historical estimator function is the opt-in `Coverage-Primary` joint feature branch. Existing standalone estimator formulas and historical experiment defaults are retained.
