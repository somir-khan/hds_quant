# AMQ comparison

For the follow-up to pilot job1169695, use [AMQ_NEXT_RUNS.md](AMQ_NEXT_RUNS.md).
It documents the archive diagnostic and separately versioned budget-cap v2 pilot.
The commands below describe the original v1 protocol.

This experiment adds **AMQ-HQQ-Matched** alongside HDS-Hybrid, Coord-Rank-Hist,
and Coord-Rank-Exact at K=4. It does not modify or merge the pending
representation-controls/LIM jobs. CoopQ is excluded.

## What is implemented

The official source is https://github.com/dlwns147/amq at commit
`618334902a5b43813607caf73025fc9595a55026`. The small search core is vendored in
`hds_quant/amq_vendor/` with Apache-2.0 LICENSE and PROVENANCE.json. This avoids
installing AMQ's unrelated quantization kernels or replacing the working HQQ stack.
The RBF predictor, search space, NSGA-II operators, 20-generation proposal search
and subset selection come from that commit. The adapter uses those components,
not an independently invented evolutionary baseline.

Compatibility wrappers skip pySOT's unrelated legacy DOE import on modern Python
and temporarily restore NumPy's removed row_stack=vstack alias during the
single-threaded pymoo call. They do not change search mathematics or downgrade
the existing NumPy/torch environment.

Each seed independently:

1. Precomputes HQQ 2/3/4-bit projection weights and FP teacher logits.
2. Screens every projection at 2 bits with all others at 4, scoring JSD relative
   to FP outputs. Sensitivities above twice the median are fixed to 4 bits.
3. Evaluates initial configurations, fits the official RBF surrogate, proposes
   configurations with official NSGA-II, evaluates their JSD, and repeats.
4. Selects the lowest measured calibration JSD in `[budget-0.005, budget]`.
   Held-out perplexity NEVER selects configurations. A missing candidate is
   reported explicitly, without relaxing the budget or using a predicted score.
5. Evaluates selected assignments using the same held-out PPL evaluator as HDS.

AMQ's RBF linear tail needs enough samples for its active dimensionality. Do not
reduce the 250 initial samples to 20 or 50 for an 8B model. A pilot reduces the
number of refinement rounds, not this necessary initialization.

## Explicit differences from the paper and upstream scripts

- Same pinned Llama-3.1-8B-Instruct, dataset revisions, 64x1024 calibration tokens,
  and 141x2048 evaluation chunks as the existing 8B runs. Each method receives
  exactly the same calibration tokens for a seed; test tokens are separate.
- HQQ axis0, group128, floating zero points throughout. Upstream example proxy
  paths use axis1. Final quantization stays HQQ, not AMQ's final AWQ compression.
  This is an allocation/search comparison, not a paper numerical reproduction.
- JSD preserves upstream epsilon=1e-7 clamping, computed in FP32 in vocabulary
  chunks of 128 tokens to limit temporary GPU memory. Teacher logits are FP16.
- Official search retains its effective-bit cost (+32/group_size). Selection
  translates this constant to target-projection weight payload, which is the
  budget used by HDS. Report actual achieved payload; the 0.005 selection window
  means AMQ can slightly underfill while HDS uses its exact feasible payload.
  No packed size, inference speed or deployment-memory claim is made.
- Quantized dequantized weights are cached in host memory; only the active model
  resides on GPU. Teacher logits use a temporary disk-backed array, removed after
  the trial. CPU transfers differ from upstream's resident proxy models and may
  slow search. Report them as part of this implementation's cost, not intrinsic
  AMQ speed. Approximately 16 GiB teacher scratch is needed for this 8B protocol;
  the SLURM launcher requests 128 GiB host RAM and one A100.
- AMQ timing includes cache construction, teacher forward, screening, search,
  CPU transfers and archive I/O. It excludes model/data loading and held-out
  evaluation. The controls retain their existing timing (excluding JSON I/O).
  Do not compare upstream paper runtimes directly with these measurements.
- Failure/nonfinite JSD stops the run rather than silently clipping a failed
  candidate to the upstream score ceiling. No resume is implemented.

## Run on LONI

Commit and push the changes using Git, then:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
source scripts/load_hf_env.sh
"${HQQ_PYTHON:-$PWD/venv/bin/python}" -m pip install -r requirements-amq.txt
mkdir -p results_hqq_amq_comparison
sbatch scripts/run_amq_comparison_a100.sh
```

The default is the WikiText pilot: seed42, 250 initial candidates, 2 rounds,
up to 10 new candidates per round, plus 224 screening evaluations. HDS/Hist/Exact
are freshly rerun at K=4. This is a pipeline validation and preliminary search,
not a full-strength AMQ result. The launcher sources the existing project `.env`
through `load_hf_env.sh`; it uses HQQ_PYTHON or the existing venv/bin/python.
It does not install dependencies inside a GPU job or expose credentials.

Separate C4 pilot:

```bash
sbatch scripts/run_amq_comparison_a100.sh configs/amq_comparison/llama31_8b_c4_pilot.json
```

Inspect the pilot's timings and missing-budget status before a full search.
Full configs use 250 initial samples, 200 rounds, up to 50 candidates per round
and population200 (paper Llama-2-7B search settings transferred to this 8B model).
Full jobs are separated by seed so a single timeout does not discard other seeds.
The 24-hour scheduler request is a limit, not a runtime guarantee.

```bash
sbatch scripts/run_amq_comparison_a100.sh configs/amq_comparison/llama31_8b_wikitext_full.json
# Additional independently recorded seeds:
sbatch scripts/run_amq_comparison_a100.sh configs/amq_comparison/llama31_8b_wikitext_full_seed100.json
sbatch scripts/run_amq_comparison_a100.sh configs/amq_comparison/llama31_8b_wikitext_full_seed2026.json
# Analogous c4_full*.json configs are provided.
```

Do not pool pilot and full results. Equal token counts are not equal compute
budgets: the native AMQ workflow screens all projections and evaluates many
complete configurations, whereas HDS K=4 measures 84 individual interventions.

## Outputs and checks

`results_hqq_amq_comparison/job_<JOBID>_<EXPERIMENT_ID>/` contains config snapshots,
data/source/package manifests, `limited_labels.json`, `comparison.csv`,
`SUMMARY.txt`, and `amq_seed_<SEED>/archive.json`. The archive saves complete bit
assignments, calibration JSD, sensitivities, fixed4 projections, iteration and
elapsed time. Every refinement round overwrites an atomic checkpoint. Partial
archives are diagnostic, not completed results or automatic resume points.

Require `complete: true` in the experiment manifest, raw result and AMQ archive.
Inspect `missing_budgets`: completion alone does not mean AMQ found candidates
at both requested budgets. The summary never hides this distinction.
Regenerate tables using:

```bash
venv/bin/python scripts/summarize_amq_comparison.py --input-dir <job-directory>
python -m pytest tests/test_amq_comparison.py -q -p no:cacheprovider
```

The local test exercises the real RBF/NSGA-II search on a synthetic transformer,
HQQ proxy construction, JSD, restoration of original weights, and checkpoint
cleanup. It does not validate 8B CUDA throughput or reproduce AMQ paper scores.
