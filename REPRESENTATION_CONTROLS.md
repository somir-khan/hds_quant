# Coord-Rank-Exact and LIM comparison

This is a separate, bounded follow-up to the completed 8B WikiText-2 and C4
experiments. Existing configs, historical results, and default method lists are
unchanged. New configs live in `configs/representation_controls/`; the new runner
reuses the existing data, HQQ, label acquisition, regression, allocation and
evaluation engines. It never reads or merges previous result directories.

## What runs

Each dataset job uses Llama-3.1-8B-Instruct, K=4, seeds 42/100/2026 and budgets
3.0/3.5 bits. K=4 is a focused diagnostic choice informed by earlier experiments,
not a new claim of independently optimized K. The controls are:

| Method | Representation / score | Measured sensitivity labels per seed | Allocation |
| --- | --- | --- | --- |
| HDS-Hybrid | 512 salience buckets, 32 rank levels, D=4096 hypervector + 9 metadata values | 84 (4 blocks x 7 projections x 3 probes) | Projection-wise {2,3,4} |
| Coord-Rank-Hist | 16 x 32 bucket-mean histogram + same metadata | 84 | Projection-wise {2,3,4} |
| Coord-Rank-Exact | 512 x 32 bucket-mean histogram + same metadata | 84 | Projection-wise {2,3,4} |
| LIM-Block-24 | Negative mean tokenwise cosine of each FP block's input/output | 0 | Whole-block {2,4} |

Exact is the existing full-resolution coordinate estimator, now supported by
the selective feature path. It matches HDS's salience-bucket resolution; it does
not preserve every original channel separately. It averages within buckets,
whereas HDS binds bucket sums and applies its existing normalization. This is
a resolution control with the same metadata, standardization, ridge tuning,
readout grouping and target transform, not an isolation of every encoding detail.
Both coordinate methods build no HDS codebooks. Raw feature lengths are 521,
16,393 and 4,105 for Hist, Exact and HDS respectively, before readout indicators.
Feature storage and total calibration time are measured separately.

LIM follows the score and descending-importance 2/4-bit rule in Dumitru et al.,
[Variable Layerwise Quantization: A Simple and Effective Approach to Quantize LLMs](https://aclanthology.org/2025.findings-acl.29/),
Findings of ACL 2025, sections 3.1--3.3. The linked
[authors' repository](https://github.com/RazvanDu/LayerwiseQuant) contained only
a README when inspected for this implementation. This is an **independent
paper-defined HQQ adaptation**, not verified parity with released author code.
The paper uses PG19 calibration and Quanto/GPTQ; here we use the same calibration
tokens as the controls and HQQ. Mean tokenwise cosine aggregation is explicitly
specified here because the paper does not fully specify its reduction across
tokens/documents. All calibration tokens have equal weight; no padded inputs.

LIM scores the original FP decoder backbone in one pass over calibration data
(multiple forward batches), without an LM-head forward or sensitivity probes.
It retains scalar sums, not an activation bank. Higher LIM scores receive 4 bits.
For 32 equal-sized blocks this gives 16 high/16 low blocks at 3 bits and 24 high/8
low at 3.5 bits. All seven target projections in a block receive the same bit
width. Unsupported unequal block sizes or unattainable exact budgets raise an
error. Tie scores are resolved by ascending block index. K=0 in its tables means
zero blocks with measured loss labels; it still uses calibration text.

The external comparison matches target-weight payload, quantizer, calibration
text and evaluation, but its feasible set differs from projection-wise {2,3,4}.
Do not interpret a difference against LIM as an isolated comparison of sensitivity
estimators. Weight bits exclude scales, zero points, packing, embeddings and the
LM head. Evaluation uses the existing dequantized-HQQ weight path; these jobs do
not measure packed inference speed or deployment memory.

## Fixed settings and cost

- Exact same pinned model/dataset revisions as `configs/generalization/llama31_8b_*.json`.
- HQQ axis 0, group size 128, floating zero points; reference 4, probes {1,2,3,4}.
  Final learned allocations exclude 1 bit. No zero-policy or quantizer change.
- Calibration: 64 x 1024 tokens; residual features use 4096 tokens. LIM uses all
  65,536 calibration tokens, since its score needs no expensive residual bank.
- Evaluation: 141 disjoint 2048-token chunks, batch size 1. WikiText train/test;
  C4 pinned first English validation shard with calibration documents [0,20000)
  and held-out documents [20000,21100). Token hashes are recorded.
- Learned methods independently reacquire the same K blocks per seed; there is
  no label/feature cache sharing. Learned execution order is shuffled per seed;
  LIM runs after these controls once per seed. CPU/GPU synchronization and memory
  accounting are shared with earlier jobs. LIM retains its own timing metadata.
- FP16 and uniform 3/4-bit evaluations are rerun. Full-bank reference acquisition
  and its offline ranking diagnostics are skipped to limit this run's cost.
- Each dataset: 9 learned trials + 3 LIM trials, 24 budget-allocation evaluations,
  and 3 uniform evaluations. No K=2/8 or new models. Each SLURM job requests one
  A100, 13 CPUs, 96 GB host RAM and 8 hours; that is a scheduling limit, not a
  measured runtime or guarantee. Exact's wider features may cost more CPU time.

## Submit using the existing environment

Commit and push these code changes from your local Git checkout, then on LONI:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
mkdir -p results_hqq_representation_controls

# First run the WikiText-2 comparison.
sbatch scripts/run_representation_controls_a100.sh

# Separate C4 job with identical methods and budgets.
sbatch scripts/run_representation_controls_a100.sh configs/representation_controls/llama31_8b_c4.json
```

The launcher sources **your existing project-root `.env`** through
`scripts/load_hf_env.sh` before Python starts. It uses `HQQ_PYTHON` if set,
otherwise `$PWD/venv/bin/python`. Keep the working cluster environment; no new
dependencies, package installation or token entry are required by this change.
Do not commit `.env`. The script records code/package versions without recording
its contents or credential values. Bash scripts are committed with LF endings.

The two jobs have separate output directories even if run concurrently:

```text
results_hqq_representation_controls/
  controls_<JOBID>.log / controls_<JOBID>.err
  job_<JOBID>_llama31_8b_wikitext_coordexact_lim_k4_v1/
  job_<JOBID>_llama31_8b_c4_coordexact_lim_k4_v1/
```

Each directory contains config snapshots, manifests, raw `limited_labels.json`,
`SUMMARY.txt`, `summary.csv`, `trials.csv`, `method_protocols.json`,
`paired_trials.csv`, `paired_summary.csv`, and optional cost/quality plots.
All score rankings, block selections and final projection assignments are saved.
Both manifest and raw JSON must have `complete: true`. Expect 12 trials, zero
full references, 24 allocations and four successful selected-projection HQQ
parity checks. The summarizer rejects missing/duplicate trials, unmatched budgets,
mismatched learned labels, invalid precisions or different evaluation counts.
Logs should end with `Representation controls completed: ...`.

Paired files use **method A minus method B**: negative PPL/NLL/time differences
favor A. Inspect per-seed differences and both datasets, not just the mean.
Seed standard deviations are not confidence intervals. This focused follow-up
does not by itself establish conference-level novelty or reproduce the LIM paper's
original numerical results.

To regenerate tables (the existing virtual environment is sufficient):

```bash
venv/bin/python scripts/summarize_representation_controls.py --input-dir results_hqq_representation_controls/job_<JOBID>_<EXPERIMENT_ID>
```

Pass a completed job's `resolved_experiment_config.json` as the SLURM argument
to replay it. The runner refuses an existing output directory. Partial JSON is
a diagnostic checkpoint, not an automatic resume point; rerun under a new job ID.
If a job ends during summarization, the raw completed JSON remains available.

## Local verification

```bash
python -m pytest tests/test_representation_controls.py tests/test_limited_labels.py tests/test_generalization.py -q -p no:cacheprovider
bash -n scripts/launch_representation_controls.sh
bash -n scripts/run_representation_controls_a100.sh
```

The offline integration test uses a synthetic four-block Llama with K=2 and a
short evaluation. It exercises HQQ, real selective feature acquisition/readout,
LIM scoring/allocation, result isolation, manifests and paired summaries without
downloading models. CPU smoke success is not a full 8B CUDA performance test.
