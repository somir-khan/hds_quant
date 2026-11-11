# Feature-value diagnostic: family means, metadata and simple allocations

This separate experiment tests what the HDS representation contributes beyond the projection family, local error magnitude and block placement. It uses the completed seed-42 Llama-3.1-8B-Instruct WikiText run as an immutable source. It does **not** run AMQ, recollect sensitivity labels, change acquisition coverage or replicate across calibration seeds.

## Frozen scope

- Source: `results_hqq_amq_budgetcap_20round_v2/job_1173223_llama31_8b_wikitext_amq_budgetcap_20round_v2`.
- Same model/dataset revisions, calibration/evaluation token hashes, 64 calibration sequences of 1024 tokens, 4096 local-feature tokens, 141 evaluation chunks of 2048 tokens, and common evaluator.
- HQQ axis 0, group size 128, floating zero points, weight-only quantize/dequantize evaluation. These are target-weight payload budgets, excluding other weights, quantization metadata and packing overhead; this is not an inference-kernel benchmark.
- Labels remain probes at 1/2/3 bits against a 4-bit background. Final allocations use only 2/3/4 bits.
- Acquired blocks remain **6, 11, 20 and 30**, with 84 measured projection/bit labels. No first/last block is added.
- Evaluate every budget **2.5, 3.0 and 3.5 bits**, with both `unconstrained` and `last_down4` settings.
- Every assignment achieves the exact same integer bit-parameter sum at its budget. Protection redistributes that payload; it does not add free bits.
- The final-block protection is a **post-hoc structural prior suggested by previous results**. It is not independently validated. The protected setting does not require a new AMQ screening, but the observation that motivated it came from prior screening and evaluation. Do not retroactively attach the original HDS runtime ratio to this variant.

The runner validates the source completion flags, configuration hash, revisions, archive payloads and selections through the existing follow-up loader. It additionally checks that all three learned methods acquired the same blocks and measured labels. Every assignment is fixed and saved before new held-out scoring. Held-out scores never select a gate, a family-mean ordering, a ridge penalty or a candidate allocation.

## Methods and why they are included

| Method | Prediction on an unmeasured projection | Purpose |
| --- | --- | --- |
| HDS-Hybrid | Reuse its saved prediction | Preserve the original learned predictor and isolate allocation/budget changes |
| Coord-Rank-Hist / Coord-Rank-Exact | Reuse their saved predictions | Keep the previous coordinate controls in the same comparison |
| Family-Mean | Arithmetic mean of raw acquired delta-NLL for that projection family and probe bit | Test whether family and precision explain the result without local residual features |
| Scalar-Meta-Ridge | Fit on log-MSE, log-Q99 and seven family indicators | Remove the hypervector while keeping its exact nine appended inputs |
| Scalar-Meta-Depth-Ridge | Those nine inputs plus block index divided by 31 | Separately test an explicit depth feature; depth was not in original HDS metadata |
| Simple-last_half | Higher precision in the last half of blocks | Measured allocation from a structural prior, with no labels/features |
| Simple-alternating_odd | Higher precision in odd-numbered blocks (zero-based) | A second measured, predeclared placement pattern |
| AMQ-screened-anchor | Reuse both saved starting anchors, evaluate again | Retain the actual measured 3.0/3.5-bit references; their construction depends on earlier AMQ screening |

For all learned controls, **individual acquired labels are retained on the measured projections**. Only unmeasured entries are predicted. For family means, take the mean of **raw** labels before the common nonnegative/monotone cost processing. Do not use the full-bank reference labels. Family means are computed separately for each probe bit; one-bit costs are retained for provenance but one-bit allocations remain prohibited.

The metadata controls use the existing per-bit ridge readout, the same alpha grid and block-grouped inner CV, the same training-fold standardization and scaled signed-log target transform, and the same cost postprocessing. Metadata-only models receive ordinary standardized columns, without the wide-vector block balancing factor used by HDS. The extra depth control is named separately. Both share one metadata feature extraction pass in this diagnostic; this is not standalone timing for either method.

The original HDS/coordinate assignments at 3.0 and 3.5 bits are retained when feasible and replay-checked. A protected assignment that needs a change is obtained by the existing fixed4 intervention using the unchanged saved cost bank. The newly introduced 2.5-bit assignments use the same saved cost banks and allocator. This is **not retraining the learned methods at 2.5 bits**: their predictions were already independent of the allocation budget.

## Family-mean tie rule

With family means, many unmeasured projections have identical costs and identical sizes. Adding tiny arbitrary noise to costs could change the primary solution, so this diagnostic does not do that.

1. Solve the existing exact-payload MILP in canonical block/family order to obtain an optimal primary cost and per-family bit counts.
2. Among unmeasured, unprotected projections with **exactly identical family, size and cost profile**, keep that precision multiset and place its higher bits first according to the declared ordering.
3. Primary order: block depth ascending. Sensitivity orders: SHA256 of `17|projection_name` and `29|projection_name`, with projection name resolving any hash collision. These are fixed pseudorandom permutations; they do not use calibration RNG state or test PPL.
4. Verify that payload and primary predicted cost do not change. Keep measured assignments and the protected projection fixed. Save all assignments and report all three outcomes.

This isolates **within-family block-placement sensitivity conditional on the MILP's chosen family precision counts**. It does not enumerate every optimal count pattern across families or every tie involving measured projections. Canonical input order, package versions and explicit assignment files make the actual result reviewable; different solver versions can still choose a different primary optimum. Do not claim a mathematically unique solution independent of solver version. The two random orders are placement controls, **not additional calibration seeds**. Do not choose the best ordering after looking at PPL.

## Simple baseline construction

Identical block geometry makes the half-block patterns exact in payload:

- **2.5 bits:** 2 bits in half the blocks, 3 bits in the selected half.
- **3.0 bits:** uniform 3 bits; the two unprotected patterns collapse to one assignment.
- **3.5 bits:** 3 bits in half the blocks, 4 bits in the selected half. These are actual models evaluated on the full corpus, not interpolated perplexities.

For `last_half`, the selected blocks are 16–31. For `alternating_odd`, they are 1, 3, …, 31. Both include the last block. At 3.5 bits they already protect the final down projection, so the two constraint settings yield identical assignments and share one evaluation.

At 2.5 or 3.0 bits, protecting that down projection requires a payload compensation. Raise it to 4 bits and lower the **first eligible gate projection in ascending block order**, using only equal-size matrices and never going below 2 bits. This is fixed before scoring and does not borrow gate choices from learned allocations. At 3.0 bits it is exactly uniform-3 plus final `down_proj` at 4 bits and **block 0 `gate_proj` at 2 bits**. At 2.5 bits the compensating gate is block 16 for the last-half pattern or block 1 for the odd-block pattern. This is still a simultaneous two-projection intervention; do not attribute its PPL change to either projection alone.

## Files and separation

- Core cost/allocation controls: `hds_quant/allocation_controls.py`.
- Source validation, candidate plans and metadata fitting: `hds_quant/allocation_control_plan.py`.
- Isolated runner: `hds_quant/run_allocation_controls.py`.
- Frozen config: `configs/allocation_controls/llama31_8b_wikitext_seed42_v1.json`.
- SLURM entry point: `scripts/run_allocation_controls_a100.sh`.
- Results only under `results_hqq_allocation_controls/`, ignored by Git.

The only shared implementation extension is selective extraction of the already defined `Scalar-Meta-Ridge` feature. It avoids rank histograms and hypervector binding; its values are tested for exact equality to the nine-column tail of the existing HDS feature. Historical feature formulas and old experiment configurations are unchanged.

## Run on the cluster

Commit and push your local changes with Git as usual, then update the cluster checkout. Run from its project root, where `hds_quant/`, `scripts/`, `.env` and your existing environment are located:

```bash
git pull
mkdir -p results_hqq_allocation_controls
sbatch scripts/run_allocation_controls_a100.sh
```

The launcher sources the existing `scripts/load_hf_env.sh` and therefore your existing `.env`. It uses `HQQ_PYTHON` when defined, otherwise `$PWD/venv/bin/python`. It does not create an environment, install packages, modify `.env` or print credentials. If your environment has a different path, use your existing `HQQ_PYTHON` setting. No additional AMQ dependencies are introduced by this diagnostic beyond what the existing follow-up modules already import.

The complete source job must still exist on the cluster. The default is the path listed above relative to the checkout root. If it is elsewhere, pass its directory explicitly:

```bash
sbatch scripts/run_allocation_controls_a100.sh /absolute/path/to/job_1173223_llama31_8b_wikitext_amq_budgetcap_20round_v2
```

Required files inside that directory are `experiment_config.json`, `resolved_experiment_config.json`, `experiment_manifest.json`, `limited_labels.json`, `data_manifest.json` and `amq_seed_42/archive.json`. The earlier fixed4 and uniform-swap result directories are not required for this run.

Optional offline preflight in the same existing environment:

```bash
source scripts/load_hf_env.sh
"${HQQ_PYTHON:-$PWD/venv/bin/python}" -m hds_quant.run_allocation_controls \
  --config configs/allocation_controls/llama31_8b_wikitext_seed42_v1.json \
  --output-dir results_hqq_allocation_controls/preflight_seed42_v1 \
  --plan-only
```

Add `--source-dir /absolute/path/to/source_job` if needed. The preflight constructs the source-based and family/simple allocations without loading a model or accessing datasets. Metadata fits necessarily wait for the GPU run. Its manifest intentionally has `complete=false`, `plan_complete=false`, `plan_only=true`. A preflight is not an experiment result. Use a new output directory on every invocation; existing folders are refused and incomplete jobs are not silently resumed.

Resource request: one A100, 13 CPUs, 96 GB host memory, eight-hour limit. The limit is an allowance, not a measured runtime prediction. The job extracts metadata once, replays FP16/uniform-3/uniform-4, and evaluates each unique allocation once. Source-based assignments are freshly evaluated here even when historical scores exist, permitting replay checks. It does not recreate teacher-logit caches or run another 20-round search.

## Outputs and interpretation

Each job has its own folder:

```text
results_hqq_allocation_controls/
  controls_JOBID.log
  controls_JOBID.err
  job_JOBID_llama31_8b_wikitext_allocation_controls_seed42_v1/
    experiment_config.json
    diagnostic_manifest.json
    data_manifest.json
    selection_plan.json
    estimates.json
    diagnostic.json
    comparisons.csv
    hds_vs_controls.csv
    SUMMARY.txt
```

The manifest records source/code/config hashes, available Git revision/tracked changes, package versions, effective quantizer settings, and the frozen plan hash. `selection_plan.json` links every unique assignment to all method/budget/constraint/tie-order roles. `estimates.json` contains the reused measured labels, raw family means and the newly fitted metadata predictions. `diagnostic.json` includes per-chunk NLLs, baseline replay checks and historical learned-assignment checks. Shared assignments have one candidate ID; repeated rows in the comparison table are **not independent evaluations**.

Only use a completed GPU result: manifest `complete=true`, `plan_complete=true`, `plan_only=false`, and result `complete=true`. Per-candidate results and summaries are written during scoring, but a partial run is not a finished comparison. Errors leave the manifest incomplete.

`hds_vs_controls.csv` reports HDS relative to each control **at the same exact budget and constraint**. Negative percentages favor HDS. Evaluate the primary family ordering and report its sensitivity range; do not select one favorable ordering or budget as the headline. Compare the exact nine-input ridge first, then the separately named depth variant. Keep screened-anchor provenance explicit.

This is a **seed-42 diagnostic**, not a fresh calibration timing benchmark. Saved labels and predictions are reused. Fresh metadata feature/fit component times are recorded for debugging; label acquisition, original HDS feature/fit time, screening and shared preprocessing are not newly timed here. No end-to-end calibration time or speedup is inferred. Family means use paid labels; a simple allocation has no label/feature acquisition, and screened anchors have screening provenance.

If HDS remains ahead of family means, metadata ridge and measured simple controls, freeze this comparison protocol before collecting new calibration seeds or C4 results. If controls match it, focus the next research question on limited-label acquisition and coverage. Boundary-inclusive labeling remains a separate, later method change with matched label budgets. Do not modify the method repeatedly based on this evaluation set or expand AMQ replication without a specific remaining question.
