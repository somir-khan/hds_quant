# Frozen K=4 allocation-control replication

This experiment replicates the family-mean and feature-control comparison using the saved Llama-3.1-8B-Instruct representation-control banks. It does not change the acquisition policy or collect additional sensitivity labels. It measures every resulting allocation on the same pinned evaluation data used by its source job.

## Run matrix

| Dataset | Calibration seeds | Role |
| --- | --- | --- |
| WikiText-2 | 100, 2026 | Requested seed replications |
| WikiText-2 | 42 | Optional exploratory reference refresh with ten random orders and standardized protection |
| C4 | 42, 100, 2026 | Requested dataset/seed replications |

The WikiText seed-42 refresh makes the three-seed reporting matrix complete under one protocol. It is not a new calibration seed or an independent confirmation of the exploratory findings. The default WikiText SLURM script runs only 100 and 2026; the commands below show how to include the refresh.

Every case uses K=4: 84 projection/precision labels from its original depth-stratified blocks. The selected blocks are 42: `[6,11,20,30]`; 100: `[0,8,19,31]`; 2026: `[1,14,18,24]`. Boundary-inclusive acquisition is not introduced here.

## Frozen comparison

- Budgets: **2.5, 3.0, 3.5** parameter-weighted bits on the seven decoder projection families. Every candidate must achieve the exact integer payload at its budget; allocations use only 2/3/4 bits.
- Both settings: unconstrained and final-block `down_proj` fixed to 4 bits. This protection is the previously observed post-hoc prior.
- Saved predictors: **HDS-Hybrid, Coord-Rank-Hist, Coord-Rank-Exact**, imported from the same source job/seed.
- New controls: **Family-Mean, Scalar-Meta-Ridge, Scalar-Meta-Depth-Ridge**. Metadata ridge uses exactly log-MSE, log-Q99 and seven family indicators. The separately named depth control adds normalized block index.
- Simple measured references: the existing last-half and alternating-odd block patterns at all three budgets, including deterministic payload compensation when protection requires it. There is no interpolated-PPL baseline.
- No additional AMQ screening, AMQ search, full-bank labels, label-coverage curve or family-count mechanism intervention.

Family-mean averages raw acquired delta-NLL labels separately by family and probe precision. Individual measured labels remain on the acquired projections, exactly as in HDS. The common nonnegative/monotone processing and exact-payload allocator are retained. Ridge fitting uses the existing block-grouped inner CV, training-fold scaling, target transform and alpha grid.

Saved probe labels include 1/2/3-bit interventions against the 4-bit background. One-bit predictions remain in the source banks and fitting protocol; final assignments exclude 1 bit.

The model, dataset revisions, HQQ axis 0 / group size 128 / floating zero points, calibration sizes and token-selection procedure are inherited and validated from the source. Each case explicitly restores its own feature-sampling seed: a multi-seed source JSON retains the final seed's configuration, which must not accidentally be reused for other seeds. Calibration and evaluation token hashes are checked before feature extraction.

## Ten random placement orders and primary result

The primary Family-Mean ordering remains **depth ascending**. The ten sensitivity orders use SHA256 of `SEED|projection_name` with fixed seeds:

```text
17, 29, 43, 59, 71, 89, 101, 127, 149, 173
```

Seeds 17 and 29 retain their previous definitions; eight additional orders are declared before new results. Each ordering places higher precision first within equal-cost, equal-size, unmeasured projections of the same family. Measured projections are not permuted. Integer payload, per-family precision counts and predicted objective are preserved.

These are **11 declared placements: one primary plus ten random**, not eleven calibration seeds. Identical assignments are evaluated once and can have multiple table roles. Summaries report all orders, the random-order median/range, unique random assignment count and how many orders HDS beats. No order is selected using test perplexity.

This sensitivity study is conditional on the MILP's primary per-family precision counts. It does not enumerate all alternative optimal family-count patterns. Canonical source geometry, saved assignments and runtime/code hashes identify the actual solution; mathematical uniqueness across solver versions is not claimed.

## Standardized protection rule

For **every method, budget and placement order**:

1. Construct the unconstrained assignment.
2. If its final `down_proj` is already 4 bits, copy that entire assignment unchanged into the protected setting. Do not re-solve or permute it again.
3. Otherwise solve with the fixed4 constraint at the same exact payload. Apply the same declared family placement order where relevant, excluding the protected projection from permutations.

This prevents an equal-cost re-solve from being misinterpreted as a protection effect. It also applies to newly fitted metadata models and to every random family ordering. Source HDS/coordinate assignments at 3.0/3.5 bits are retained for the unconstrained setting and replay-checked; 2.5-bit assignments use their saved budget-independent cost banks.

The old seed-42 diagnostic runner/config remain available with their historical defaults. Shared helpers gained optional arguments; their old defaults retain three tie orders and the old summary behavior. New behavior is isolated behind `frozen_replication_v1` configs and the new runner.

## Cost and evidence boundaries

These runs reuse sensitivity labels and learned predictions. The two metadata controls need one fresh shared feature pass and fresh fits; their component times are recorded. This is **not a fresh end-to-end timing benchmark**. Do not add the recorded component times to old acquisition times and present that as a directly measured family-mean runtime. Fresh method timing remains a separate next experiment.

Evaluation is the existing weight quantize/dequantize benchmark, not a packed inference-kernel speed test. Budgets exclude other model weights, quantization metadata and packing overhead. Held-out scores select neither allocations nor tie orders.

## Existing environment and source folders

The launcher sources `scripts/load_hf_env.sh`, using the existing trusted `.env`. It selects `HQQ_PYTHON` if defined, otherwise `$PWD/venv/bin/python`. It creates no environment, installs no dependencies and prints no credentials. Keep your existing model access and cache settings.

Default input folders, relative to the cluster project root:

```text
results_hqq_representation_controls/job_1169696_llama31_8b_wikitext_coordexact_lim_k4_v1
results_hqq_representation_controls/job_1171547_llama31_8b_c4_coordexact_lim_k4_v1
```

Each must contain the original `experiment_config.json`, `resolved_experiment_config.json`, `experiment_manifest.json`, `limited_labels.json` and `data_manifest.json`. The C4 representation-control job `1171551` is an alternative duplicate source if that is the copy retained on your cluster. Pass it explicitly rather than merging the duplicate runs.

Original 20-round AMQ, fixed4 and uniform-swap result folders are not required.

## Submit on the cluster

After committing/pushing the new files locally, update the existing cluster checkout and run from its project root:

```bash
git pull
mkdir -p results_hqq_frozen_replication_wikitext results_hqq_frozen_replication_c4
```

To start with WikiText seed 100 only:

```bash
sbatch --array=1 scripts/run_frozen_replication_wikitext_a100.sh
```

To run both new WikiText seeds (100 and 2026), use the script's default array:

```bash
sbatch scripts/run_frozen_replication_wikitext_a100.sh
```

To run the complete WikiText matrix including the optional seed-42 reference refresh:

```bash
sbatch --array=0-2%1 scripts/run_frozen_replication_wikitext_a100.sh
```

Run all three C4 seeds:

```bash
sbatch scripts/run_frozen_replication_c4_a100.sh
```

Choose one WikiText submission variant; do not submit the same seeds twice unless intentionally rerunning a failed job. Array indices are `0 -> 42`, `1 -> 100`, `2 -> 2026`. The default `%1` runs one task at a time **within each dataset array**. Submitting both arrays can run two GPU jobs concurrently. Each task requests one A100, 13 CPUs, 96 GB memory and an eight-hour allowance; this is not a measured runtime estimate.

If a source folder is elsewhere, pass it as the first script argument:

```bash
sbatch scripts/run_frozen_replication_c4_a100.sh /absolute/path/to/c4_source_job
sbatch --array=0-2%1 scripts/run_frozen_replication_wikitext_a100.sh /absolute/path/to/wikitext_source_job
```

Both arrays use the same environment loader and Python-selection logic. Failed jobs do not silently resume; submit a new job with a new output directory. Existing output directories are refused.

## Optional offline preflight

This validates the selected saved bank and constructs HDS/coordinate, family-mean and simple allocations without loading a model or datasets:

```bash
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
"$PYTHON_BIN" -m hds_quant.run_frozen_replication \
  --config configs/frozen_replication/llama31_8b_c4_v1.json \
  --seed 100 \
  --output-dir results_hqq_frozen_replication_c4/preflight_seed100_v1 \
  --plan-only
```

Use `--source-dir /absolute/path/to/source_job` if necessary. Metadata fits remain pending until the GPU job. The preflight intentionally records `complete=false`, `plan_complete=false`, `plan_only=true`; it is not a completed experiment. Use a fresh preflight directory on subsequent attempts.

## Outputs and joint report

Each dataset has a separate root and each array task has its own folder:

```text
results_hqq_frozen_replication_c4/
  frozen_ARRAYID_TASKINDEX.log
  frozen_ARRAYID_TASKINDEX.err
  job_ARRAYID_TASKINDEX_seedSEED_v1/
    experiment_config.json
    diagnostic_manifest.json
    data_manifest.json
    selection_plan.json
    estimates.json
    diagnostic.json
    comparisons.csv
    hds_vs_controls.csv
    placement_sensitivity.csv
    SUMMARY.txt
```

The WikiText root has the same structure. Require both manifest and diagnostic completion flags, a complete plan and `plan_only=false`. Source hashes, code/config hashes, available Git revision/tracked changes, packages, effective HQQ settings, token hashes and the frozen plan hash are recorded. The final plan is saved before held-out scoring. Partial rows are useful for diagnosis but are not a completed comparison.

When finished, aggregate the exact completed case folders using the existing environment:

```bash
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
"$PYTHON_BIN" scripts/summarize_frozen_replications.py \
  results_hqq_frozen_replication_wikitext/job_*_seed*_v1 \
  results_hqq_frozen_replication_c4/job_*_seed*_v1 \
  --output-dir results_hqq_frozen_replication_summary/final_v1
```

If reruns left multiple folders for one dataset/seed, replace the globs with the exact intended completed folders. The aggregator rejects duplicates, incomplete jobs, older diagnostic schemas and mixed runtime/code snapshots. It records the selected input hashes. It does not choose a job using PPL.

Outputs include `paired_seed_results.csv`, `decisions.csv` and `input_manifest.json`. An absent WikiText-42 refresh produces an explicitly incomplete three-seed summary rather than silently importing its old three-order diagnostic. The C4 matrix must include all three seeds for a complete C4 decision.

## Predeclared practical decision rule

For each dataset/budget/constraint separately, the primary paired difference is:

```text
100 * (HDS_PPL / primary_family_mean_PPL - 1)
```

Negative favors HDS. The practical margin is **1% using this denominator**. If all three seed differences are at most -1%, report a consistent material HDS advantage on the primary ordering; if all are at least +1%, report a consistent material family-mean advantage. If all absolute differences are below 1%, report that they fall below the practical margin, **not statistical equivalence**. Otherwise report mixed/inconclusive results. Missing seeds remain explicitly incomplete.

These are descriptive decisions, not p-values or a venue judgment. The ten-order sensitivity range must accompany the primary result: an apparent primary advantage can still be sensitive to placement. Never count orderings as independent seeds, select a favorable ordering/budget after scoring, or interpret one benchmark dataset as universal evidence.

After this matrix is complete, decide whether to proceed with fresh timing and a separately declared, label-budget-matched acquisition-coverage study. The replication code does not implement either method change.
