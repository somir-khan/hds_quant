# C4 validation and larger-model experiments

Use the main Git checkout on LONI. All launchers load the existing project-root
`.env` through `scripts/load_hf_env.sh` before starting Python, and explicitly
use `venv/bin/python` (or HQQ_PYTHON). Model access must already be approved for
the account associated with the token. The scripts do not install packages.

## 1. C4 with the current 1B model

After committing/pushing the code and pulling the same branch on the cluster:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
mkdir -p results_hqq_c4_1b
sbatch scripts/run_c4_1b_a100.sh
```

Configuration: `configs/generalization/c4_1b.json`.

* Llama-3.2-1B-Instruct; axis 0, group size 128, floating zero points.
* Calibration probe choices {1,2,3,4}, reference 4; final allocations {2,3,4}.
* HDS-Hybrid, Coord-Rank-Hist, Scalar-MSE-Ridge; K=2/4/8; seeds 42/100/2026.
* Budgets 3.0/3.5; actual independent label acquisition/timing per method.
* Full measured-loss reference after limited-label decisions for each seed.
* Calibration: 64 sequences x 1024 tokens; 4096 tokens for residual features.
* First C4 English validation shard: document indices [0,20000) for calibration;
  [20000,21100) for held-out evaluation. Ranges are end-exclusive. No shuffle.
* Evaluation is the first **141 disjoint 2048-token chunks** of the held-out
  stream (288,627 scored next-token predictions), not the entire C4 corpus or
  all 1,100 held-out documents. The job rejects insufficient text rather than
  silently reducing the configured chunk count. Batch size 8.
* One A100, 64 GB host RAM, four-hour scheduler limit. Duration depends on
  data access and hardware. C4 is streamed from one pinned raw shard; text
  preparation and downloads are outside calibration timing.

This changes only the allocation feasible set relative to the original probe
protocol. Removing 1-bit probes/training rows is a separate ablation and is
not performed. Historical {1,2,3,4} allocations remain a separate experiment;
use the recorded no-1-bit diagnostic for its matched WikiText-2 comparison.

## 2. Larger-model preflight

Default (recommended larger instruction-tuned checkpoint):

```bash
mkdir -p results_hqq_large_preflight
sbatch scripts/run_large_model_preflight_a100.sh
```

This uses `configs/generalization/llama31_8b_preflight.json` and
`meta-llama/Llama-3.1-8B-Instruct`.

If you specifically want the 7B checkpoint instead:

```bash
sbatch scripts/run_large_model_preflight_a100.sh configs/generalization/llama2_7b_preflight.json
```

That config uses `meta-llama/Llama-2-7b-hf`, a base model. The 8B choice is
instruction-tuned. Results across these checkpoints do not isolate scale alone.
Choose one preflight first; both are provided as explicit independent configs.

Preflight scope:

* One seed (42), K=2, all three predictors, budgets 3.0/3.5.
* Same probe/allocator distinction and quantization settings as C4.
* WikiText-2 train calibration, test evaluation; only 8 x 2048 test-token chunks.
* Full-sized calibration workload (64 x 1024; 4096 feature tokens), batch size 1.
* Checks first q/up projection weights and outputs against packed HQQLinear at
  3/4 bits. These selected-projection checks are not a full packed-model PPL test.
* Measures the real calibration workload and its allocated GPU tensor peak.
* Skips full-bank reference acquisition and offline ranking diagnostics to
  keep the preflight bounded. It cannot establish calibration speedup versus
  a full-bank reference, and its PPL is not a full WikiText-2 benchmark.
* One A100, 96 GB host RAM, two-hour scheduler limit. It may reveal a need to
  adjust resources; GPU compatibility cannot be established by local CPU tests.

Inspect successful completion, parity checks, baseline PPL, GPU memory, and
trial duration before defining a full larger-model benchmark. Do not launch
the full 7B/8B x C4 grid based only on a model-loading check.

## 3. Full Llama-3.1-8B WikiText-2 benchmark

The 8B preflight job 1165151 completed successfully. This separate benchmark
uses the same model and dataset commit revisions recorded by that run, rather
than resolving `main` again. The preflight configuration remains available.

After committing/pushing the new files and pulling the same branch on LONI:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
mkdir -p results_hqq_8b_wikitext
sbatch scripts/run_llama31_8b_wikitext_a100.sh
```

Configuration: `configs/generalization/llama31_8b_wikitext.json`.

* Model: `meta-llama/Llama-3.1-8B-Instruct`, revision
  `0e9e39f249a16976918f6564b8830bc894c89659`.
* WikiText-2 raw, revision `b08601e04326c79dfdd32d625aee71d232d685c3`;
  train calibration and test evaluation.
* HDS-Hybrid, Coord-Rank-Hist, Scalar-MSE-Ridge; K=2/4/8 of 32 blocks;
  seeds 42/100/2026. These are 27 independently timed limited-label trials.
* Both 3.0- and 3.5-bit budgets, with exactly the same quantizer and estimator
  settings used in the preflight: axis 0, group 128, floating zero points;
  probe choices {1,2,3,4}, final allocation choices {2,3,4}, reference 4.
* Calibration: 64 x 1024 tokens; residual features use 4096 tokens.
* Evaluation: 141 disjoint 2048-token chunks, or 288,627 scored next-token
  predictions, at batch size 1. This is the fixed complete-chunk evaluation
  budget, not a sliding-window evaluation of every test token.
* FP16, uniform 3-bit and uniform 4-bit baselines use that same evaluation set.
* Full measured-loss acquisition over all 32 blocks runs after the limited-label
  decisions for each seed. The three references produce six evaluated budget
  allocations; limited trials produce 54. No preflight label/feature caches
  or results are reused. `Oracle-Loss` denotes this measured-loss allocator,
  not a proven optimum of end-to-end perplexity.
* One A100, 13 CPUs (matching the recorded preflight), 96 GB host RAM, and a
  24-hour time request. The full grid takes substantially longer than the
  preflight. The request is a scheduling limit, not a measured duration;
  actual runtime and peak memory must be checked from this benchmark.

The launcher sources the project-root `.env` through the existing helper and
uses `venv/bin/python` (or `HQQ_PYTHON`); no package upgrade is needed. Keep the
same environment as the successful preflight. Package versions, source hashes,
resolved revisions and token hashes are recorded with the results.

Monitor the SLURM job and logs:

```bash
squeue -u "$USER"
tail -f results_hqq_8b_wikitext/8b_<JOBID>.log
```

The completed job writes `SUMMARY.txt`, `summary.csv`, `trials.csv`, raw JSON
and manifests beneath
`results_hqq_8b_wikitext/job_<JOBID>_llama31_8b_wikitext_allocation234_v1/`.
Both `limited_labels.json` and `experiment_manifest.json` must say
`"complete": true`. There should be 27 limited trials, three full references,
60 evaluated allocations, and 141 evaluation chunks. Check that all four
selected-projection parity checks passed and all achieved budgets match.

Partial JSON checkpoints are diagnostic records, not automatic resume points.
An interrupted job needs a fresh output directory; the runner rejects reuse.
For a completed-run replay, pass its `resolved_experiment_config.json` as the
first argument to this SLURM script.

Use uniform 3-bit as the budget-matched baseline at 3.0 bits, and the measured
reference at both budgets. Assess seed variation as well as mean perplexity.
The preflight had HDS close to uniform at 3 bits; this benchmark tests whether
that pattern persists. Do not use the short preflight PPL as a directly
comparable full-benchmark number or tune configurations on the test results.

## 4. Full Llama-3.1-8B C4 benchmark

This is the next dataset check after the completed 8B WikiText-2 benchmark
(job 1165153). It reuses the same calibration, estimator, quantizer and
allocation implementations. Only the dataset specification and experiment
identity change relative to `llama31_8b_wikitext.json`.

After committing/pushing these changes and pulling the same branch on LONI:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
mkdir -p results_hqq_8b_c4
sbatch scripts/run_llama31_8b_c4_a100.sh
```

Configuration: `configs/generalization/llama31_8b_c4.json`.

* Llama-3.1-8B-Instruct revision
  `0e9e39f249a16976918f6564b8830bc894c89659`, matching the 8B WikiText-2 run.
* C4 revision `1588ec454efa1a09f29cd18ddd04fe05fc8653a2`, matching the completed
  1B C4 run. Stream the same first English validation shard, with document
  indices [0,20000) for calibration and [20000,21100) for held-out evaluation.
* Three methods, seeds 42/100/2026, K=2/4/8, budgets 3.0/3.5; retain the entire
  grid instead of selecting K=4 based on the WikiText-2 results.
* HQQ axis 0, group size 128, floating zero points; probe bits {1,2,3,4},
  reference bit 4, final allocations {2,3,4}.
* Calibration: 64 x 1024 tokens, with 4096 tokens for residual features.
* Evaluation: the first 141 disjoint 2048-token chunks of the held-out stream,
  with 288,627 scored next-token predictions; batch size 1.
* FP16, uniform 3-bit and uniform 4-bit baselines, plus independent full
  measured-loss references over all 32 blocks for each seed. There are 27
  limited trials, three references, and 60 evaluated budget allocations.
* One A100, 13 CPUs, 96 GB host RAM and a 24-hour scheduling limit, matching
  the 8B WikiText-2 launcher. This is a full benchmark, not a short preflight.

The existing shared launcher loads the project-root `.env` and uses
`venv/bin/python` (or `HQQ_PYTHON`). Keep the environment from the successful
8B run; no package upgrades are required. Each method acquires its own labels
and features for timing. Previous C4 or WikiText-2 labels/features are not reused.
Selected q/up projection parity checks and the existing source/package/data
manifests are included automatically.

```bash
squeue -u "$USER"
tail -f results_hqq_8b_c4/8b_c4_<JOBID>.log
```

Results are saved to
`results_hqq_8b_c4/job_<JOBID>_llama31_8b_c4_allocation234_v1/`.
Check `"complete": true` in both result and experiment manifests, 141 evaluated
chunks, all four parity checks passing, and the trial/reference counts above.
The existing runner rejects an existing output directory; partial checkpoints
do not provide automatic resume. Replay a completed run by passing its
`resolved_experiment_config.json` as the SLURM script's first argument.

Compare each method to the uniform and measured-loss baselines within C4.
Use the two 8B datasets to assess whether quality, seed stability and calibration
tradeoffs persist; raw perplexities across datasets are not directly comparable.

## Results remain separate

* C4: `results_hqq_c4_1b/job_<JOBID>_c4_1b_allocation234_v1/`
* 8B preflight: `results_hqq_large_preflight/job_<JOBID>_llama31_8b_wikitext_preflight_v1/`
* 7B preflight: `results_hqq_large_preflight/job_<JOBID>_llama2_7b_wikitext_preflight_v1/`
* 8B benchmark: `results_hqq_8b_wikitext/job_<JOBID>_llama31_8b_wikitext_allocation234_v1/`
* 8B C4 benchmark: `results_hqq_8b_c4/job_<JOBID>_llama31_8b_c4_allocation234_v1/`

The new config-driven runner rejects existing output directories. Historical
launchers keep their original defaults; new optional hooks reuse the tested
calibration engine rather than duplicating the estimator implementation.

Each job saves:

* `experiment_config.json`: the exact requested config.
* `resolved_experiment_config.json`: immutable model/dataset commit revisions
  resolved at startup, suitable as a later config input. `main` is only the
  initial lookup; actual model/tokenizer/data loads use the resolved revisions.
* `experiment_manifest.json`: purpose, config/source-code hashes, package
  versions, revisions, and selected execution settings. HF_TOKEN is not recorded.
* `data_manifest.json`: pinned shard/ranges or dataset split fingerprints,
  raw-text/token hashes, actual evaluation budget, and calibration window offsets.
* `limited_labels.json`: actual trial measurements, predictions, assignments,
  evaluation losses, probe choices and separate allocation choices; checkpoints
  after each trial. `complete: true` marks successful completion.
* `trials.csv`, `summary.csv`, `SUMMARY.txt`; plots if matplotlib is installed.
  Summary headings identify the checkpoint, dataset, policy and preflight status.

For exact data/model revision replay, pass a prior
`resolved_experiment_config.json` to the corresponding SLURM script as its
first argument. Use a new job directory and the same code/package environment.
Never present smoke-test or short preflight outputs as benchmark results.

## Code-only checks without downloading models

```bash
python -m pytest tests/test_generalization.py tests/test_limited_labels.py -q
```

Tests use synthetic tokenizers/models to verify disjoint C4 indices,
deterministic sampling, frozen evaluation sizes, retained 1-bit probe labels,
forbidden 1-bit allocations, selected-projection packed parity, manifest
creation, preflight reference skipping, and overwrite protection.
