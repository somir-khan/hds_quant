# True limited-label calibration experiment

Extract the update bundle into the LONI project containing `hds_quant/`,
`scripts/`, `tests/`, and `venv/`. From that project directory:

```bash
mkdir -p results_hqq_limited_labels
sbatch scripts/run_limited_labels_a100.sh
```

The script follows the existing LONI account/partition, requests one A100 and
four hours, and uses `venv/bin/python` or an explicit `HQQ_PYTHON` override.
It loads the existing HF environment helper if present. No package installations
or new jobs are triggered. To generate plots, matplotlib must already be
installed in that environment; missing matplotlib does not prevent table output.

## Fixed configuration

Llama-3.2-1B-Instruct, WikiText-2, HQQ axis 0, group size 128, floating zero
points, candidate bits {1,2,3,4}, int4_aligned anchor, salience_bucket binding,
per_bit readouts, signed_log1p targets. Calibration uses 64 x 1024 tokens, with
4096 sampled tokens for residual features. Evaluation uses the full held-out
stream in disjoint 2048-token chunks, batch size 8.

K = 2/4/8 transformer blocks, seeds = 42/100/2026, budgets = 3.0/3.5 average
payload bits. The methods are HDS-Hybrid, Coord-Rank-Hist, Scalar-MSE-Ridge.
Each method/K/seed is independently executed with a fresh label acquisition
and feature extraction. There are 27 limited-label trials and 54 resulting
allocations. Three additional full-label references each provide two allocations.
Uniform 3-bit, uniform 4-bit and FP16 provide quality anchors; the 4-bit and
FP16 anchors do not match the mixed-precision budgets.

## What differs from the old low-shot curve

* Only the selected K blocks receive marginal loss probes. All model projections
  still use 4-bit reference weights when measuring those probes.
* Every method receives the same selected blocks and calibration tokens for
  each K/seed. K sets use the existing depth-stratified selection and need not
  be nested across K values.
* Each trial recomputes labels. No full-bank cache, cached features or estimated
  K/16 probe-time scaling is used. ILP payload caches are cleared before trials.
* Only the chosen feature and chosen readout are computed. Scalar/coordinate
  controls do not construct unused HDS codebooks.
* Features are required for all blocks, including blocks without labels.
* Acquired blocks use their measured costs; other blocks use predicted costs.
  Ridge tuning uses layer-grouped splits within the acquired blocks only.
* Method order is reproducibly shuffled per K/seed. One original-model forward
  warmup per seed is outside timing. GPU kernels are synchronized for timing.
* The full-label measured-loss baseline runs after the limited-label trials.
  Its labels are used only for an offline ranking diagnostic afterward and
  never enter the limited-label fitting or allocation.

## Time and memory accounting

Each trial measures the complete calibration wall time through solving both
budgets. For comparison at one budget, it reports measured shared calibration
(selected-label acquisition, encoder construction, method feature extraction,
design construction, regression) plus that budget's measured ILP time. This
does not divide or extrapolate probe time. Both quantities are retained so
multi-budget and single-budget costs cannot be confused.

Model/data loading, warmup, serialization, and held-out evaluation are excluded
from calibration time. Evaluation time is separately recorded. GPU peak allocated
bytes include the resident model and are measured only during calibration;
incremental bytes are relative to the allocation at trial start. These are
PyTorch allocated bytes, not total GPU process memory. CPU feature bytes count
stored numeric payload only, not Python containers or transient design matrices.
Encoder codebook bytes are reported separately. There is no claim of faster
packed inference or lower inference memory from these measurements.

Outputs are written to `results_hqq_limited_labels/job_<JOBID>/`:

* `limited_labels.json`: checkpoint after every completed trial, raw PPL/NLL,
  scored-token counts, assignments, measured/predicted labels, selected blocks,
  actual timing and memory. A final `complete: true` confirms completion.
* `trials.csv`: one row per seed/method/K/budget.
* `summary.csv` and `SUMMARY.txt`: means and population standard deviations
  across seeds, not confidence intervals.
* `cost_quality_3.00b.png/.pdf` and `cost_quality_3.50b.png/.pdf`, if matplotlib
  is installed: direct cost versus quality with across-seed standard deviations.

The runner does not resume incomplete trials or reuse their timing. A new
SLURM job gets a new output directory. Partial JSON is retained for diagnosis.

Offline CPU check, without model downloads:

```bash
python -m hds_quant.run_limited_labels --smoke-test --k-layers 2 --seeds 42 --output-dir tmp/limited_smoke
python scripts/summarize_limited_labels.py --input-dir tmp/limited_smoke
```
