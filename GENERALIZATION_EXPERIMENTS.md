# C4 validation and larger-model preflight

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

## Results remain separate

* C4: `results_hqq_c4_1b/job_<JOBID>_c4_1b_allocation234_v1/`
* 8B preflight: `results_hqq_large_preflight/job_<JOBID>_llama31_8b_wikitext_preflight_v1/`
* 7B preflight: `results_hqq_large_preflight/job_<JOBID>_llama2_7b_wikitext_preflight_v1/`

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
