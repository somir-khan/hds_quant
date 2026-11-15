# One-job 1B replication on WikiText and C4

This version runs **both datasets in the same SLURM job and Python process**, loading the 1B model once. It creates a fresh current-pipeline replication rather than combining older WikiText outputs with newer C4 outputs.

The earlier WikiText experiment did use `meta-llama/Llama-3.2-1B-Instruct`. Its folder lacks the newer pinned provenance bundle, so its numerical scores are historical context. The existing C4 job supplies a checked model/data provenance bridge; it does not supply training labels, predictions or PPL targets to the new study.

## Frozen scope

- Model: `meta-llama/Llama-3.2-1B-Instruct`, revision `9213176726f574b556790deb65791e0c5aa438b6`.
- Datasets in this order: WikiText-2, then C4, both pinned to the same dataset revisions/selections as the current 8B pipeline.
- Calibration seeds: **42, 100, 2026** on both datasets.
- Focused comparison: **K=2 boundary-inclusive versus K=8 stratified**.
- All five arms: Family-Mean, Family-Median, Scalar-Meta-Ridge, Coord-Rank-Hist, HDS-Hybrid.
- **3.5 bits primary; 3.0 bits secondary.** Keep both in reporting; do not pool budgets into a quality-equivalence claim.
- Protection on: final block's `down_proj` at 4 bits, with exact integer payload compensation. On this model that is block **15**.
- Ten fixed random placements plus the depth-ascending primary for both family arms.
- HQQ axis 0, group size 128, floating zero points, official quantize/dequantize backend.
- Probe bits 1/2/3 against a 4-bit background; final allocations use only 2/3/4 bits.
- Calibration: 64 sequences × 1024 tokens; 4096 feature tokens. Evaluation: 141 chunks × 2048 tokens, 288627 scored next-token targets.

K=2 and K=8 mean 42 and 168 labels per fit, as on 8B. They cover **12.5% and 50% of this 16-block model**, versus 6.25% and 25% on 8B. Report counts and fractions.

This is a second model size in the same Llama family, not a new architecture-family validation. The experiment does not assume the 8B result will reproduce on 1B.

## Batch-size decision and fresh baselines

The old 1B experiments used batch size **8**. This new run explicitly uses batch size **1** for collection and evaluation to align with the current 8B pipeline.

Both datasets receive fresh FP16, Uniform-3 and Uniform-4 baselines under the new setting. Repeated baseline evaluations across seeds check internal consistency and weight restoration against that fresh reference. They are not claims of exact replay against the old batch-8 scores.

Historical numeric labels and PPL are deliberately not imported as replay targets. The generic collector may record a zero-overlap historical replay field; the run manifest explicitly marks historical numeric replay as not performed. The substantive new numerical check is the complete timing-versus-quality label/prediction/assignment comparison.

Do not forecast the new runtime from the old 68/227-second 1B measurements. Batch size and pipeline differ; the new standalone measurements are the relevant evidence.

## Reuse and timing

Each dataset/seed quality case collects a fresh union of just the two declared acquired sets and shares one joint feature pass. Each fit is strictly masked to its own 42 or 168 labels. Mean and median coincide at K=2 and share identical assignments. Duplicate assignments are evaluated once.

After quality scoring, the job independently times all five arms at each of the two settings. Each timed arm reacquires its labels, extracts its required features, fits and allocates without shared calibration caches. Timed primary assignments must match the quality-study counterparts exactly. Label differences must be within the declared numerical tolerance; prediction differences are recorded.

Calibration timing excludes model/data loading, warmup, I/O and held-out scoring. The ten placement sensitivity evaluations are excluded from primary-method timing. The model remains loaded while both datasets run, but the timed calibration operations are freshly executed.

Default full run: **six quality cases, twelve timing cells, sixty standalone arm calibrations and 120 budget-assignment checks**. A single-seed pilot has two quality cases, four timing cells and forty assignment checks.

## Source requirements

Default historical C4 provenance folder, relative to the cluster project root:

```text
results_hqq_c4_1b/job_1165143_c4_1b_allocation234_v1
```

Required files: `experiment_config.json`, `resolved_experiment_config.json`, `experiment_manifest.json`, `limited_labels.json`, `data_manifest.json`.

The adapter accepts the legitimate original `main` revisions together with their recorded resolved commit hashes. It checks completion, config hash, model/dataset pins, quantizer settings and token provenance. Newly prepared C4 text/token hashes and calibration windows must reproduce that source. Batch-size changes do not alter token selection.

The old WikiText folder is not required. WikiText is freshly prepared using the pinned Salesforce/wikitext `wikitext-2-raw-v1` train/test splits and the same tokenizer revision as C4. New manifests record all resulting text/token hashes.

No 8B source folder or AMQ output is required. The frozen 8B source guards remain unchanged. New topology validation derives all q/k/v/o/gate/up/down shapes from the loaded 1B model configuration, then checks all 112 discovered projections.

## Run from the cluster project root

Commit/push the code using Git as usual, then:

```bash
git pull
mkdir -p results_hqq_1b_focused
sbatch scripts/run_focused_1b_both_a100.sh
```

**One submission runs both datasets and all three seeds, including quality and standalone timing.** This script has no SLURM array. It requests one A100, 13 CPUs, 96 GB host memory and a 24-hour scheduling allowance. The allowance is not a runtime estimate.

It sources `scripts/load_hf_env.sh`, uses your existing `.env`, and selects `HQQ_PYTHON` or otherwise `$PWD/venv/bin/python`. No packages or environment files are installed or modified. Model/data caches and access credentials use your existing settings.

If the historical C4 source is elsewhere:

```bash
sbatch scripts/run_focused_1b_both_a100.sh \
  --historical-c4-source-dir /absolute/path/to/job_1165143_c4_1b_allocation234_v1
```

Optional smaller pilot: both datasets, seed 42, quality and timing:

```bash
sbatch scripts/run_focused_1b_both_a100.sh --seed 42
```

A later full run goes into its own new job folder; do not combine duplicate pilot/full cases as additional seeds. An optional `--quality-only` invocation runs both datasets without standalone timing. It cannot supply cost claims or be upgraded in place; the default full run is the intended replication.

## Offline preflight

```bash
source scripts/load_hf_env.sh
PYTHON_BIN="${HQQ_PYTHON:-$PWD/venv/bin/python}"
"$PYTHON_BIN" -m hds_quant.run_focused_1b \
  --config configs/coverage_generalization/llama32_1b_both_focused_v1.json \
  --output-root results_hqq_1b_focused/preflight_both_v1 \
  --plan-only
```

Add the historical-source override if needed. This checks the declaration, old provenance and 16-block acquisition plans without loading a model or downloading datasets. Actual checkpoint geometry and fresh baseline checks remain pending. Its `complete=false` and `plan_only=true` flags are intentional. Use a new output directory every time.

## Outputs

```text
results_hqq_1b_focused/
  run_JOBID.log
  run_JOBID.err
  job_JOBID_both_v1/
    run_manifest.json
    experiment_config.json
    acquisition_plan.json
    model_config.json
    projection_geometry.json
    wikitext2/
      data_manifest.json
      fresh_uniform_baselines.json
      seed_42/
        resolved_model_config.json
        quality/
        timing_k2_boundary_inclusive/
        timing_k8_stratified/
      seed_100/...
      seed_2026/...
    c4/
      ...same structure...
    summary/
      primary_quality.csv
      standalone_timing_with_quality.csv
      paired_seed_results.csv
      mean_results.csv
      validation.json
```

Each quality folder contains its case manifest, label/feature banks, estimates, frozen plan, baseline checks, per-chunk scores, primary results and placement sensitivity. Each timing folder contains its case manifest, measured calibration stages, raw labels/predictions, assignments and explicit quality-parity checks.

The summary is generated automatically after all requested cases pass. It separates primary and secondary budgets, records label coverage fractions, reports per-seed PPL differences and measured runtime ratios, and never estimates missing timings. The same baseline object is never reused between datasets.

To verify/rebuild a completed run's summary:

```bash
source scripts/load_hf_env.sh
"${HQQ_PYTHON:-$PWD/venv/bin/python}" scripts/summarize_focused_1b.py \
  results_hqq_1b_focused/job_JOBID_both_v1
```

Replace `JOBID`. Require the root run manifest and every requested case to be complete. A single-seed pilot remains explicitly marked as an incomplete three-seed matrix. Failed runs retain partial per-case outputs but do not silently resume; use a new output folder for a fresh run. New 1B data and summaries remain separate from old 1B and 8B result schemas.

## Code boundaries

- `focused_1b_protocol.py`: strict source bridge, pinned declaration and model-derived geometry.
- `run_focused_1b.py`: one-model, both-dataset orchestration and provenance.
- `focused_1b_summary.py`: current-baseline reporting, label/assignment parity and focused cost-quality summaries.
- `coverage_engine.py`: an optional acquisition-plan argument reuses the existing tested quality/timing engine; the 8B default remains its full K=2/4/8 coverage plan.

The unchanged 8B checkpoint/model guard is not relaxed to accept arbitrary models. The new route is explicit and versioned.
