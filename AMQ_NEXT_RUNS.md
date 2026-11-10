# C4 controls, AMQ archive diagnostic, and budget coverage v2

The objective/fixed4 diagnostics have completed. The next small control is the
frozen Uniform-3 matched swap in [UNIFORM_SWAP_CONTROL.md](UNIFORM_SWAP_CONTROL.md),
before a seed100 replication. Boundary-inclusive labeling remains a later,
separate method change.

The 20-round WikiText run has completed as job 1173223. The current follow-up is
the saved-assignment objective diagnostic and fixed4 constraint check described
in [AMQ_OBJECTIVE_FIXED4_DIAGNOSTICS.md](AMQ_OBJECTIVE_FIXED4_DIAGNOSTICS.md).

The diagnostic and two-round WikiText v2 pilot have now completed. For the next
bounded 20-round WikiText run, use [AMQ_20_ROUND.md](AMQ_20_ROUND.md) and the dedicated
`scripts/run_amq_budgetcap_20round_a100.sh` launcher. The instructions below retain
the earlier diagnostic/pilot sequence for reproducibility.

Run these as separate jobs with separate result directories. Existing completed
WikiText results and all v1 configurations remain unchanged. All launchers source
the project `.env` through `scripts/load_hf_env.sh` and use `HQQ_PYTHON` or
`$PWD/venv/bin/python`. No new dependencies beyond requirements-amq.txt.

## 1. Run the existing Coord/LIM protocol on C4

After committing/pushing locally and pulling on LONI:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
mkdir -p results_hqq_representation_controls
sbatch scripts/run_representation_controls_c4_a100.sh
```

This dedicated launcher selects the existing pinned C4 configuration. It runs
HDS, Coord-Hist, Coord-Exact at K=4 plus LIM, three seeds, budgets 3/3.5. It uses
the same model revision, calibration/evaluation document separation, token counts,
HQQ settings, and reporting as the completed WikiText controls. No new method or
changed allocation policy is introduced. Expect 12 trials and 24 allocations.
The result directory ends in `llama31_8b_c4_coordexact_lim_k4_v1`.

## 2. Diagnose the saved AMQ pilot without rerunning its search

Keep the original completed job directory on LONI. The default source is:
`results_hqq_amq_comparison/job_1169695_llama31_8b_wikitext_amq_matched_pilot_v1`.
The diagnostic requires its original JSON files and `amq_seed_42/archive.json`.
If only the ZIP is available on LONI, extract it into a separate directory and
pass that extracted job path as the first script argument. Do not overwrite the
original completed results.

```bash
mkdir -p results_hqq_amq_diagnostic
sbatch scripts/run_amq_diagnostic_a100.sh
# Or specify a different location for the same source job:
# sbatch scripts/run_amq_diagnostic_a100.sh /path/to/original/job_directory
```

The candidate-selection rule is fixed before any new held-out evaluation:

- Original narrow-window winner for each requested budget, where one exists.
- Two lowest archived calibration-JSD configurations below each budget cap.
- All-4-bit initialization as a reference; duplicates are evaluated once.

For job1169695 this gives six zero-based archive indices: **2, 74, 146, 154,
204, 241**. The original poor 3-bit candidate (146) is retained. No candidate is
chosen using held-out perplexity. This is explicitly post-hoc analysis, not an
independent benchmark establishing an AMQ or HDS win.

The runner verifies source completion/config/payload/fixed4 constraints, rebuilds
the exact pinned data, and checks calibration and evaluation token hashes. It
then rebuilds proxies and teacher logits once and, for each selected assignment:

1. Recomputes calibration JSD and compares with the archive (absolute tolerance
   1e-5).
2. Compares cached-weight calibration NLL with fresh HQQ quantization through the
   shared evaluator (absolute tolerance 1e-6).
3. Evaluates held-out perplexity, and checks historical NLL for the original
   selected candidate and all4 reference (absolute tolerance 1e-6).

Mismatch fails the diagnostic and leaves incomplete manifests plus available
partial results. Original FP weights are restored even on failure. Teacher-logit
scratch is removed after the diagnostic, including exception cleanup.

No sensitivity screening, RBF fitting, NSGA-II search, or HDS/Coord acquisition
is repeated. This job requests one A100, 128 GiB RAM and four hours; it should
be substantially smaller than a new full search, but the request is not a runtime
promise. It needs the same approximately 16 GiB teacher scratch as the pilot.

Outputs: `results_hqq_amq_diagnostic/job_<ID>_archive_seed_42_v1/` contains a frozen
`selection_plan.json`, `diagnostic_manifest.json`, `data_manifest.json`,
`diagnostic.json` and `SUMMARY.txt`. Require complete=true and all parity checks
passing. Diagnostic elapsed time is not a new AMQ calibration-speed result.

Offline selection inspection is also available (creates a new output directory,
does not load a model or download data):

```bash
venv/bin/python -m hds_quant.run_amq_diagnostic \
  --source-dir results_hqq_amq_comparison/job_1169695_llama31_8b_wikitext_amq_matched_pilot_v1 \
  --output-dir results_hqq_amq_diagnostic/selection_only_1169695 --plan-only
```

## 3. Revised AMQ budget coverage: new v2 pilot

After the diagnostic passes, submit the revised WikiText pilot:

```bash
mkdir -p results_hqq_amq_budgetcap_v2
sbatch scripts/run_amq_budgetcap_a100.sh
```

This uses `configs/amq_comparison/llama31_8b_wikitext_budgetcap_pilot_v2.json`.
The experiment is explicitly named **AMQ-HQQ-BudgetCap-v2**. It is a matched-HQQ
adaptation of the official search components, not unchanged upstream AMQ or a
reproduction of the paper's final AWQ results.

Two changes address the observed coverage problem:

1. After the original 250 initial configurations, evaluate one additional
   exact-payload starting assignment per requested budget. A deterministic MILP
   finds an assignment near uniform precision while preserving the projections
   fixed to 4 bits by AMQ screening. It uses geometry and the seed only, not loss
   labels or held-out data. Infeasible exact budgets fail explicitly. These
   evaluated anchors enter the archive before RBF search, guaranteeing candidate
   coverage at each supported budget. Identical existing candidates are reused.
2. Select the lowest **measured calibration JSD among all candidates at or below
   the memory cap**. Remove the artificial lower edge of the old 0.005 window.
   Report actual payload and the gap from the cap, even when the best candidate
   underfills it. Do not describe that result as exactly 3.0/3.5 bits or infer
   equal deployed memory.

The official surrogate, NSGA-II proposal search and subset selection remain
unchanged. Coverage does not guarantee high quality, convergence, or superiority
over HDS. The revised pilot still has one seed and two rounds of up to ten new
candidates, for at most 272 measured configurations plus 224 screening probes.
The exact starting assignments are saved in the trial and the archive contains
their measured scores. HDS/Hist/Exact are independently rerun in this job.

The C4 v2 config is provided for later use, after inspecting WikiText:

```bash
sbatch scripts/run_amq_budgetcap_a100.sh configs/amq_comparison/llama31_8b_c4_budgetcap_pilot_v2.json
```

Existing `run_amq_comparison_a100.sh` and `*_full.json` still use **v1**. Do not
submit those to obtain v2, and do not launch the old full search unchanged: the
pilot suggested roughly 33 hours per seed before growing predictor costs, above
the old 24-hour request. Decide the next search effort after these diagnostics.

## Verification

```bash
python -m pytest tests/test_amq_comparison.py tests/test_amq_coverage.py tests/test_representation_controls.py -q -p no:cacheprovider
```

Tests exercise exact anchors on actual 8B projection sizes, preservation of fixed4
projections, infeasible-budget rejection, old/new selection rules, real small-model
RBF/NSGA-II search and archive replay, weight restoration, and result summaries.
Local CPU tests do not replace the scheduled 8B CUDA diagnostic.
