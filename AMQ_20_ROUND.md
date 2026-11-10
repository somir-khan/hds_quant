# AMQ BudgetCap-v2: 20-round WikiText experiment

## Why this run

Job 1172697 successfully covered both budgets, but its two selected assignments
were the initialization anchors. Neither of its two refinement rounds improved
the best calibration JSD below the caps. This separate experiment tests whether
additional search improves quality and what that improvement costs.

The method remains AMQ-HQQ-BudgetCap-v2. Relative to the pilot configuration,
only the experiment ID and refinement-round count change:

| Setting | Value |
| --- | --- |
| Model | Pinned Llama-3.1-8B-Instruct |
| Dataset | Pinned WikiText-2 train calibration / test evaluation |
| Seed | 42 |
| Initial design | 250 configurations |
| Additional starting anchors | Up to two, at payloads 3.0 and 3.5 |
| Refinement | 20 rounds, up to ten new candidates each |
| NSGA-II population | 200 |
| Total configuration evaluations | Up to 452, plus 224 projection screens |
| Selection | Lowest measured calibration JSD under each payload cap |
| Controls | Independent HDS/Hist/Exact reruns, K=4 |
| HQQ | Axis 0, group 128, floating zero points |

This is a fresh, self-contained run. It rebuilds caches and teacher logits and
does not resume the pilot. The same search algorithm and selection policy are
used; progress reporting does not consume random samples or evaluate additional
candidates. Held-out PPL is evaluated only for the final selected assignments,
not after each round. Search duration is fixed in advance at 20 rounds.

## Submit using the existing environment

Commit and push the code through your normal Git workflow. Then on LONI:

```bash
cd /ddnB/work/somir49/hds_quant
git pull --ff-only
mkdir -p results_hqq_amq_budgetcap_20round_v2
sbatch scripts/run_amq_budgetcap_20round_a100.sh
```

The dedicated launcher defaults to
`configs/amq_comparison/llama31_8b_wikitext_budgetcap_20round_v2.json`.
The existing shared launcher sources project-root `.env` with
`scripts/load_hf_env.sh`, then uses `HQQ_PYTHON` or `$PWD/venv/bin/python`.
Use the environment that ran the pilot; no new dependencies or token entry
are required. Keep `.env` out of Git.

The job requests one A100, 13 CPUs, 128 GB host RAM, and a 24-hour scheduling
limit on the existing partition/account. The limit is not a runtime estimate.
Host proxy storage and teacher scratch requirements match the pilot (roughly
39 GiB proxies and 15.6 GiB teacher scratch, plus other process memory).

## Output and monitoring

```text
results_hqq_amq_budgetcap_20round_v2/
  amq_<JOBID>.log
  amq_<JOBID>.err
  job_<JOBID>_llama31_8b_wikitext_amq_budgetcap_20round_v2/
    experiment_config.json
    resolved_experiment_config.json
    experiment_manifest.json
    data_manifest.json
    limited_labels.json
    comparison.csv
    SUMMARY.txt
    amq_seed_42/
      archive.json
      search_progress.json
      search_progress.csv
```

```bash
squeue -u "$USER"
tail -f results_hqq_amq_budgetcap_20round_v2/amq_<JOBID>.log
```

Progress is recorded after the initial design, after anchors, and after every
completed refinement round: 22 snapshots / 44 budget rows for a completed run.
Each row includes elapsed calibration time, evaluated count, best measured JSD,
winning archive index and origin, actual payload and gap, whether JSD improved,
and the new candidates' payload range. The same history is stored in the archive
and final AMQ trial. CSV/JSON progress views are atomically replaced after each
finished stage. Intermediate candidate checkpoints are diagnostic evidence;
automatic resume is not supported. A failed stage may leave candidates in the
archive beyond the last finished progress row.

Progress files have no independent completion flag; check the final experiment
manifest, raw result, and archive for `complete: true`. Expect archive iteration
20, four trials, eight final budget evaluations, and no missing AMQ budgets.
Configuration count can be below 452 when proposed assignments are duplicates.
Both caps have anchors, but selection may choose an underfilled candidate;
report its actual payload.

Compare the initial-design, anchor, round-2, round-5, round-10, and round-20 rows
to see whether search improves its calibration objective. Keep the final held-out
PPL as an outcome, rather than using it to pick an earlier stopping round.
Timings include the small progress-reporting overhead. No inference speed or
converged full-paper AMQ performance claim is implied.

Do not use the two-round `run_amq_budgetcap_a100.sh` default or old v1 full-search
configs to submit this experiment. Completed result folders are preserved and a
repeated submission gets a fresh job directory.

## Local verification

```bash
python -m pytest tests/test_amq_comparison.py tests/test_amq_coverage.py tests/test_amq_progress.py -q -p no:cacheprovider
bash -n scripts/run_amq_budgetcap_20round_a100.sh
```

Small-model tests exercise actual HQQ/RBF/NSGA-II execution, restoration,
budget selection, and progress consistency. They do not replace the 8B CUDA run.
