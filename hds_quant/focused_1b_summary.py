"""Focused 1B reporting, strictly separated from old 1B and frozen 8B results."""
import json
import statistics
from pathlib import Path

from .amq_followup_plan import digest
from .coverage_protocol import ARMS
from .coverage_summary import quality_rows, write_csv
from .frozen_replication_plan import ORDERS
from .run_amq_followup import save


def summarize_quality(report, output):
    rows = quality_rows(report)
    for row in rows:
        row['block_coverage_fraction'] = row['k_layers']/report['num_layers']
        row['primary_budget'] = row['budget'] == 3.5
    write_csv(output/'comparisons.csv', rows)
    primary = [r for r in rows if r['method'] in ARMS and r['tie_order'] in ('native', 'depth_ascending')]
    write_csv(output/'primary_results.csv', primary)
    sensitivity = []
    for row in primary:
        if not row['method'].startswith('Family-'):
            continue
        random = [r for r in rows if (r['method'], r['k_layers'], r['policy'], r['budget']) ==
            (row['method'], row['k_layers'], row['policy'], row['budget']) and r['tie_order'].startswith('random_')]
        if len(random) == 10:
            sensitivity.append(dict(dataset=report['dataset'], seed=report['seed'], method=row['method'],
                k_layers=row['k_layers'], policy=row['policy'], budget=row['budget'], primary_ppl=row['ppl'],
                random_ppl_min=min(r['ppl'] for r in random), random_ppl_median=statistics.median(r['ppl'] for r in random),
                random_ppl_max=max(r['ppl'] for r in random), unique_random_assignments=len({r['candidate_id'] for r in random})))
    write_csv(output/'placement_sensitivity.csv', sensitivity)
    lines = [f'Complete: {report["complete"]}', f'Fresh 1B quality: {report["dataset"]}, seed {report["seed"]}.',
        'Batch size 1; fresh baselines. Older batch-8 PPL/labels are not historical replay targets.',
        '3.5 bits primary; 3.0 bits secondary. K=2 boundary (12.5%) vs K=8 stratified (50%).',
        '42 vs 168 labels per fit; shared quality bank is strictly masked.',
        'Family depth ordering primary; ten fixed random orders are placement sensitivity, not calibration seeds.']
    lines += [str(r) for r in primary]
    (output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def check_timing(quality, estimates, timing):
    if len(timing['trials']) != 5 or {t['method'] for t in timing['trials']} != set(ARMS):
        raise ValueError('Expected five complete standalone arms')
    checks = []
    cell = timing['cell']
    expected_cell = next(c for c in estimates['cells'] if (c['k_layers'], c['policy']) == (cell['k_layers'], cell['policy']))
    for trial in timing['trials']:
        measured = trial['measured_labels']; expected = expected_cell['measured_labels']
        if measured.keys() != expected.keys():
            raise ValueError('Standalone timing label mask differs from quality')
        label_gap = max(abs(measured[k]-expected[k]) for k in measured)
        pred = expected_cell['raw_predictions'][trial['method']]
        if pred.keys() != trial['raw_predictions'].keys():
            raise ValueError('Standalone prediction coverage differs')
        prediction_gap = max(abs(pred[k]-v) for k, v in trial['raw_predictions'].items())
        if label_gap > 1e-6:
            raise ValueError('Fresh timing labels differ from fresh quality labels')
        for budget, allocation in trial['allocations'].items():
            matches = [c for c in quality['candidates'] for r in c['roles'] if
                (r['method'], r['k_layers'], r['policy'], r['budget'], r['tie_order']) ==
                (trial['method'], cell['k_layers'], cell['policy'], float(budget),
                 'depth_ascending' if trial['method'].startswith('Family-') else 'native')]
            if len(matches) != 1 or matches[0]['bit_assignment'] != allocation['bit_assignment']:
                raise ValueError('Standalone timing assignment differs from the quality primary')
            checks.append(dict(method=trial['method'], budget=float(budget), assignment_parity=True,
                max_label_abs_difference=label_gap, max_prediction_abs_difference=prediction_gap))
    return checks


def aggregate(root, *, require_completed_run=True):
    root = Path(root); run = json.loads((root/'run_manifest.json').read_text())
    if run['schema'] != 'focused_1b_run_v1' or run['plan_only'] or (require_completed_run and not run['complete']):
        raise ValueError('Expected a completed focused 1B run')
    if digest(root/'experiment_config.json') != run['config_sha256']:
        raise ValueError('Run config hash differs')
    if digest(root/'model_config.json') != run['model_config_sha256']:
        raise ValueError('Pinned model config changed')
    rows = []; time_rows = []; checks = []; inputs = []
    for dataset in run['datasets']:
        baseline_path = root/dataset/'fresh_uniform_baselines.json'
        if digest(baseline_path) != run['baseline_sha256'][dataset]:
            raise ValueError('Fresh baseline reference changed')
        if digest(root/dataset/'data_manifest.json') != run['data_manifest_sha256'][dataset]:
            raise ValueError('Dataset token manifest changed')
        for seed in run['selected_seeds']:
            case = root/dataset/f'seed_{seed}'; qdir = case/'quality'
            qm = json.loads((qdir/'case_manifest.json').read_text())
            quality = json.loads((qdir/'diagnostic.json').read_text())
            estimates = json.loads((qdir/'estimates.json').read_text())
            if (not qm['complete'] or not quality['complete'] or
                qm['selection_plan_sha256'] != digest(qdir/'selection_plan.json') or
                qm['parent_config_sha256'] != run['config_sha256'] or
                qm['baseline_sha256'] != run['baseline_sha256'][dataset] or
                qm['label_bank_sha256'] != digest(qdir/'label_bank.json') or
                qm['feature_bank_sha256'] != digest(qdir/'feature_bank.pt')):
                raise ValueError('Incomplete or changed quality case')
            rows.extend(r for r in quality_rows(quality) if r['method'] in ARMS and r['tie_order'] in ('native', 'depth_ascending'))
            inputs.append(dict(directory=str(qdir.relative_to(root)), manifest_sha256=digest(qdir/'case_manifest.json')))
            if run['quality_only']:
                continue
            for k, policy in [(2, 'boundary_inclusive'), (8, 'stratified')]:
                directory = case/f'timing_k{k}_{policy}'
                tm = json.loads((directory/'case_manifest.json').read_text())
                timing = json.loads((directory/'timing.json').read_text())
                if (not tm['complete'] or not timing['complete'] or tm['timing_sha256'] != digest(directory/'timing.json') or
                    tm['parent_config_sha256'] != run['config_sha256']):
                    raise ValueError('Incomplete or changed timing case')
                if tm['quality_manifest_sha256'] != digest(qdir/'case_manifest.json'):
                    raise ValueError('Timing refers to a different quality manifest')
                found = check_timing(quality, estimates, timing)
                checks.extend(dict(dataset=dataset, seed=seed, k_layers=k, policy=policy, **c) for c in found)
                for trial in timing['trials']:
                    for budget, a in trial['allocations'].items():
                        qrow = next(r for r in rows if (r['dataset'], r['seed'], r['method'], r['budget'], r['k_layers']) ==
                            (dataset, seed, trial['method'], float(budget), k))
                        time_rows.append(dict(dataset=dataset, seed=seed, method=trial['method'], budget=float(budget),
                            k_layers=k, policy=policy, block_coverage_fraction=k/16, probe_label_count=21*k,
                            calibration_to_budget_sec=a['calibration_to_budget_sec'], ppl=qrow['ppl'], primary_budget=float(budget) == 3.5))
                inputs.append(dict(directory=str(directory.relative_to(root)), manifest_sha256=digest(directory/'case_manifest.json')))
    paired = []
    for row in rows:
        if row['k_layers'] != 2:
            continue
        other = next(r for r in rows if (r['dataset'], r['seed'], r['method'], r['budget'], r['k_layers']) ==
            (row['dataset'], row['seed'], row['method'], row['budget'], 8))
        def timer(k):
            return next((t['calibration_to_budget_sec'] for t in time_rows if
                (t['dataset'], t['seed'], t['method'], t['budget'], t['k_layers']) ==
                (row['dataset'], row['seed'], row['method'], row['budget'], k)), None)
        a, b = timer(2), timer(8)
        paired.append(dict(dataset=row['dataset'], seed=row['seed'], method=row['method'], budget=row['budget'],
            primary_budget=row['budget'] == 3.5, k2_boundary_ppl=row['ppl'], k8_stratified_ppl=other['ppl'],
            k2_vs_k8_ppl_percent=100*(row['ppl']/other['ppl']-1), k2_calibration_sec=a,
            k8_calibration_sec=b, measured_speedup=b/a if a is not None else None))
    means = []
    for dataset in run['datasets']:
        for method in ARMS:
            for budget in [3., 3.5]:
                group = [p for p in paired if (p['dataset'], p['method'], p['budget']) == (dataset, method, budget)]
                lo = statistics.mean(p['k2_calibration_sec'] for p in group) if time_rows else None
                hi = statistics.mean(p['k8_calibration_sec'] for p in group) if time_rows else None
                means.append(dict(dataset=dataset, method=method, budget=budget, primary_budget=budget == 3.5,
                    calibration_seed_count=len(group), mean_k2_ppl=statistics.mean(p['k2_boundary_ppl'] for p in group),
                    mean_k8_ppl=statistics.mean(p['k8_stratified_ppl'] for p in group),
                    mean_paired_ppl_percent=statistics.mean(p['k2_vs_k8_ppl_percent'] for p in group),
                    mean_k2_calibration_sec=lo, mean_k8_calibration_sec=hi,
                    ratio_of_mean_calibration_times=hi/lo if time_rows else None))
    out = root/'summary'; out.mkdir(exist_ok=True)
    write_csv(out/'primary_quality.csv', rows); write_csv(out/'standalone_timing_with_quality.csv', time_rows)
    write_csv(out/'paired_seed_results.csv', paired); write_csv(out/'mean_results.csv', means)
    save(out/'validation.json', dict(schema='focused_1b_summary_v1', inputs=inputs, checks=checks,
        quality_cases=len(run['datasets'])*len(run['selected_seeds']), timing_cases=0 if run['quality_only'] else 2*len(run['datasets'])*len(run['selected_seeds']),
        assignment_checks=len(checks), complete_three_seed_matrix=run['selected_seeds'] == [42, 100, 2026],
        complete_focused_timing_matrix=not run['quality_only'] and run['selected_seeds'] == [42, 100, 2026],
        primary_budget=3.5, batch_size=1, historical_batch8_scores_reused=False))
    return dict(quality_cases=2*len(run['selected_seeds']), timing_cases=0 if run['quality_only'] else 4*len(run['selected_seeds']), assignment_checks=len(checks))
