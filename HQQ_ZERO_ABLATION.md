# Small HQQ zero-point ablation and configuration guide

## Submit on LONI

Copy the updated `hds_quant/`, `scripts/`, and `tests/` files to the project
directory containing your cluster virtual environment. Run there:

```bash
mkdir -p results_hqq_zero_ablation
sbatch scripts/run_hqq_zero_ablation_a100.sh
```

If extraction created a nested `hqq_zero_ablation_update/` folder, it is an
update bundle, not a standalone project. From the existing project root
(the folder containing `venv/`), merge its files first:

```bash
cp -r hqq_zero_ablation_update/hds_quant/. hds_quant/
cp -r hqq_zero_ablation_update/scripts/. scripts/
cp -r hqq_zero_ablation_update/tests/. tests/
./venv/bin/python -c 'import torch, hqq; print(torch.__version__)'
mkdir -p results_hqq_zero_ablation
sbatch scripts/run_hqq_zero_ablation_a100.sh
```

The launcher uses `venv/bin/python` explicitly and stops with an actionable
error if it is missing. For a differently located Linux environment, export
`HQQ_PYTHON=/absolute/path/to/bin/python` before submission. A Windows virtual
environment cannot be used on the Linux cluster. GPU availability is checked
inside the allocated job, not on the login node.

The script uses the existing LONI account/partition and requests one A100 for
one hour. It requires CUDA. It does not submit any additional jobs.

## What this experiment changes

| Setting | Both conditions |
| --- | --- |
| Model | meta-llama/Llama-3.2-1B-Instruct |
| Dataset | WikiText-2; train for calibration, test for final PPL |
| Quantizer / axis / group size | Official HQQ / 0 / 128 |
| Seed | 42 |
| Candidate projection precisions | 1, 2, 3, 4 bits, matching the previous axis-0 run |
| Average payload budgets | 3.0 and 3.5 bits |
| Feature configuration | int4_aligned / salience_bucket / per_bit / signed_log1p |
| Evaluated estimators | HDS-Hybrid, Coord-Rank-Hist, Oracle-Loss |
| Calibration | 64 sequences of 1,024 tokens; up to 4,096 tokens for residual features |
| Evaluation | Full test token stream, disjoint 2,048-token chunks, batch size 8 |
| Low-shot curve | Disabled |

The two conditions are `--hqq-zero-policy floating` (floating zero points at
all bit widths) and `--hqq-zero-policy round4` (integer zero points at 4 bits
only). This changes the location of the quantization grid, not the number of
levels. It does not mean zeroing weights. At 4 bits there are still 16 levels.

Each policy first gets a two-chunk diagnostic against packed HQQLinear at
4 bits. That diagnostic is not the full benchmark. Both main experiments then
evaluate the full test stream. They recompute their own calibration labels and
features in separate directories. The script intentionally omits `--reuse-cache`.

There are 12 main mixed-precision evaluations: 2 policies x 3 methods x 2 budgets.
Each condition also evaluates the five uniform baselines. Shared features and
readouts are still computed for all methods; `--estimators` reduces allocation
and held-out evaluation only. This is not an estimator runtime comparison.

Outputs:

* `results_hqq_zero_ablation/floating/run_floating.json`
* `results_hqq_zero_ablation/round4/run_round4.json`
* `results_hqq_zero_ablation/zero_policy_comparison.csv`
* `results_hqq_zero_ablation/zero_policy_comparison.txt`

The comparison checks that all controlled settings match, the FP16/1/2/3-bit
baselines remain unchanged (NLL tolerance 1e-6), and achieved payloads match.
A negative change percentage means rounded 4-bit zero points gave lower PPL.
One seed is a diagnostic, not evidence of statistical significance. Extend
across seeds if the difference merits investigation; do not select a policy
solely to optimize repeated test-set results.

## What the experiment does

1. Measure the language model's loss with all targeted projections at 4 bits.
2. Temporarily reduce one projection to each lower candidate precision and
   measure its additional loss. These are the sensitivity labels.
3. Extract features describing how quantization changes projection outputs.
4. Fit sensitivity predictors on other transformer layers.
5. Choose one precision for each projection under a common payload budget.
6. Apply all chosen precisions together and measure held-out perplexity.

A projection is one q/k/v/o/gate/up/down weight matrix, not a whole transformer
block. There are 16 blocks x 7 projections = 112 allocation units in this model.

## The historical configurations

| Run family | Checkpoint | Quantizer | Axis |
| --- | --- | --- | --- |
| Original results | Llama-3.2-1B Base | RTN | Contiguous row groups |
| HQQ axis-1 results | Llama-3.2-1B-Instruct | HQQ | 1 |
| HQQ axis-0 results | Llama-3.2-1B-Instruct | HQQ | 0 |

The original-to-HQQ change also changes the checkpoint; it does not isolate
the quantizer. The axis comparison holds those recorded settings fixed.

Within a run family, two feature presets were used:

| Setting | strict_baseline | salience_int4_tweak |
| --- | --- | --- |
| Residual anchor | fp16_strict | int4_aligned |
| Coordinate binding | raw_index_strict | salience_bucket |
| Readout grouping | pooled | per_bit |
| Target transform | signed_log1p | signed_log1p |

`strict_baseline` is an experimental preset, not the FP16 baseline and not the
name of a precision. Each preset contains all the estimator rows.

* **Anchor:** The reference output used to measure a local projection error.
  fp16_strict uses the original projection output; int4_aligned uses its 4-bit
  output. In both presets, task-loss sensitivity labels still use a model with
  all targeted projections at 4 bits. These are two different references.
* **Coordinate binding:** raw_index_strict identifies output channels by their
  index. salience_bucket groups identities according to ranked original-output
  column norms. Here salience is an output-strength proxy, not measured task loss.
* **Readout grouping:** pooled fits a predictor shared across candidate bit
  widths; per_bit fits separate predictors. per_family separates projection
  families; per_bit_family separates both, with pooled fallbacks for small groups.
  A readout is a ridge regression sensitivity predictor, not LLM fine-tuning.
* **Target transform:** signed_log1p compresses large positive and negative loss
  changes for regression and transforms predictions back afterward. raw uses
  loss changes directly; rank trains on relative ordering.
* **Axis:** HQQ reshapes a matrix to [-1, group_size] for axis 1 and
  [group_size, -1] for axis 0, reducing along the chosen axis. This determines
  which weights share a scale/zero point; it is not a model layer number.
* **Group size:** Number of weights sharing a quantization grid. Smaller groups
  provide more grids and more metadata. A payload-only bit budget excludes that
  metadata, untouched model parameters, and any deployment packing overhead.
* **Candidates versus budget:** Candidates are the choices for each projection.
  A 3-bit average can mix 1/2/3/4-bit projections. It is not the same allocation
  as uniform 3-bit. The average is weighted by projection parameter counts.
* **Seeds:** Control calibration samples and randomized representations. The
  held-out test stream stays fixed; seeds are not separate test datasets.
* **Four-fold layer-disjoint CV:** Predictions for a projection are fitted using
  labels from other blocks. For this 16-block model each outer fold holds out
  four blocks. It does not train or fine-tune language-model weights.
* **Low-shot K:** Acquire labels from K transformer blocks, fit the predictor,
  and predict the rest. Features are still needed for the target blocks. K is
  a calibration-label count, not the number of quantized blocks.

The strict-to-tweaked comparison changes three switches simultaneously. It
cannot attribute an improvement to any one of them without separate ablations.

## Reading the estimator rows

| Row | Information used to predict sensitivity |
| --- | --- |
| Scalar-MSE | Mean squared local output error, used directly |
| Scalar-L1 | Mean absolute local output error, used directly |
| Scalar-TrimmedMean | Mean error after removing both tails |
| Scalar-WinsorL2 | RMS error after clipping extreme values |
| Scalar-Q99 | 99th percentile of local output error |
| Scalar-MSE-Ridge | A learned mapping from log mean squared error |
| Scalar-Meta-Ridge | Learned log MSE, log Q99, and projection-family metadata |
| Ordinal-Hist | Error-magnitude histogram plus log MSE |
| RandProj-JL | Random projection of average within-token error ranks |
| Coord-Rank-Hist | Explicit channel-bucket/rank histogram plus metadata |
| Coord-Rank-Exact | Finer coordinate/rank representation plus metadata |
| HDS-Rank | Hyperdimensional representation of error-rank patterns |
| HDS-Mag-Strict | HDS rank representation plus log MSE |
| HDS-Hybrid | HDS rank representation plus log MSE, log Q99, family metadata |
| Oracle-Loss | Directly measured individual loss changes with monotone allocation costs |
| Oracle-Loss-Raw | Same measurements without that monotone adjustment |

The learned methods use ridge regression; Scalar-L1 and Scalar-MSE are error
statistics, not choices of regression regularization. Oracle-Loss is not a
guaranteed best whole-model allocation: simultaneous damage need not equal
the sum of the individually measured losses.

## Reading the report columns

* **PPL:** Whole-model next-token prediction quality on held-out text; lower is
  better. Compare methods at the same achieved budget and evaluation protocol.
* **Spearman / Kendall:** Agreement between predicted and measured sensitivity
  orderings; higher is better. These do not measure whole-model language quality.
* **Within-family correlation:** Ranking agreement within q/k/v/o/gate/up/down
  families, averaged over families. It is different from within-bit correlation.
* **Top-20% recall:** Fraction of the truly most damaging probe choices also
  identified among the predicted most damaging 20%; higher is better.
* **Cross-seed stability:** Agreement across randomized runs. Stable predictions
  or allocations can still be inaccurate.
* **Mean +/- standard deviation:** Variation across calibration seeds, not a
  confidence interval. One seed cannot estimate that variability.
* **Achieved bits / payload gap:** Precision actually allocated and distance
  from the requested budget. It is not a measured model-file size.
* **Calibration time:** Cost of obtaining information and fitting the predictor.
  Check the report's accounting before comparing it with full job wall time.

At 3 bits, the previous axis-0 tweaked HDS-Hybrid scored 32.83 PPL versus
36.93 for uniform 3-bit and 32.70 for Coord-Rank-Hist. This means it helped
relative to uniform allocation, while the coordinate-histogram control was
competitive. A global ranking correlation near 0.92 does not mean 92% accuracy.
