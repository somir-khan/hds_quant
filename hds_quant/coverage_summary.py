"""Coverage curves, matched acquisition comparisons and standalone timing parity."""
import csv
import json
import statistics
from pathlib import Path

from .coverage_protocol import ARMS
from .frozen_replication_plan import ORDERS


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader()
        writer.writerows({k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows)


def quality_rows(report):
    return [dict(dataset=report['dataset'], seed=report['seed'], candidate_id=c['candidate_id'],
        **r, ppl=c['evaluation']['ppl'], nll=c['evaluation']['nll'])
        for c in report['candidates'] for r in c['roles']]


def summarize_quality(report, output):
    rows = quality_rows(report); write_csv(output/'comparisons.csv', rows)
    primary = [r for r in rows if r['tie_order'] in ('depth_ascending', 'native')]
    write_csv(output/'coverage_curves.csv', primary)
    pairs = []; sensitivity = []
    for row in primary:
        if row['policy'] != 'stratified':
            continue
        other = [r for r in primary if (r['method'], r['k_layers'], r['budget'], r['policy']) ==
            (row['method'], row['k_layers'], row['budget'], 'boundary_inclusive')]
        if len(other) == 1:
            after = other[0]
            pairs.append(dict(dataset=report['dataset'], seed=report['seed'], method=row['method'],
                k_layers=row['k_layers'], budget=row['budget'], stratified_ppl=row['ppl'],
                boundary_ppl=after['ppl'], boundary_vs_stratified_percent=100*(after['ppl']/row['ppl']-1),
                identical_acquisition=row['acquired_layers'] == after['acquired_layers'],
                identical_assignment=row['candidate_id'] == after['candidate_id']))
    for row in primary:
        if row['method'] not in ARMS[:2]:
            continue
        group = [r for r in rows if (r['method'], r['k_layers'], r['budget'], r['policy']) ==
                 (row['method'], row['k_layers'], row['budget'], row['policy'])]
        if len(group) == len(ORDERS):
            random = [r for r in group if r['tie_order'] != 'depth_ascending']
            sensitivity.append(dict(dataset=report['dataset'], seed=report['seed'], method=row['method'],
                k_layers=row['k_layers'], budget=row['budget'], policy=row['policy'],
                primary_ppl=row['ppl'], random_order_count=10,
                unique_random_assignments=len({r['candidate_id'] for r in random}),
                random_ppl_min=min(r['ppl'] for r in random),
                random_ppl_median=statistics.median(r['ppl'] for r in random),
                random_ppl_max=max(r['ppl'] for r in random)))
    write_csv(output/'acquisition_pairs.csv', pairs); write_csv(output/'placement_sensitivity.csv', sensitivity)
    lines = [f'Complete: {report["complete"]}',
        f'Coverage quality sweep: {report["dataset"]}, seed {report["seed"]}.',
        'All five arms, K=2/4/8, both policies, protected exact budgets 3.0/3.5.',
        'One physical union label bank; every predictor sees only its declared 21*K labels.',
        'Quality feature/label sharing is not standalone calibration timing.',
        'Family mean and median: depth-ascending primary plus ten fixed random placements.',
        'K=2 mean and median are identical controls. Repeated roles are not independent observations.',
        'Boundary comparison includes the same structural protection in both policies.',
        'Negative boundary_vs_stratified_percent favors boundary-inclusive acquisition.']
    lines += [str(r) for r in pairs]
    (output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def summarize_timing(report, output):
    rows = []
    for trial in report['trials']:
        for budget, a in trial['allocations'].items():
            rows.append(dict(dataset=report['dataset'], seed=report['seed'], method=trial['method'],
                k_layers=trial['k_layers'], policy=trial['policy'], probe_label_count=trial['probe_label_count'],
                budget=float(budget), label_acquisition_sec=trial['label_acquisition_sec'],
                feature_extraction_sec=trial['feature_extraction_sec'], fit_and_cost_sec=trial['fit_and_cost_sec'],
                allocation_sec=a['allocation_sec'], calibration_to_budget_sec=a['calibration_to_budget_sec'],
                calibration_wall_both_budgets_sec=trial['calibration_wall_sec']))
    write_csv(output/'standalone_timings.csv', rows)
    (output/'SUMMARY.txt').write_text(f'Complete: {report["complete"]}\n'+report['timing_scope']+'\n', encoding='utf-8')


def aggregate(job_dirs, output):
    from .amq_followup_plan import digest
    from .run_amq_followup import save
    quality = {}; timing = {}; snapshots = set(); inputs = []
    for directory in job_dirs:
        directory = Path(directory); m = json.loads((directory/'study_manifest.json').read_text())
        spec = json.loads((directory/'experiment_config.json').read_text())
        if (m['schema'] != 'coverage_study_v1' or not m['complete'] or m['plan_only'] or
            m['config_sha256'] != digest(directory/'experiment_config.json')):
            raise ValueError('Expected completed coverage-study job')
        snapshots.add(json.dumps({k: m[k] for k in ['code_sha256', 'packages', 'effective_hqq_config']}, sort_keys=True))
        key = (m['dataset'], m['seed'])
        if m['mode'] == 'quality':
            d = json.loads((directory/'diagnostic.json').read_text())
            p = json.loads((directory/'selection_plan.json').read_text())
            if (not d['complete'] or not m['plan_complete'] or
                digest(directory/'selection_plan.json') != m['selection_plan_sha256'] or
                len(d['candidates']) != len(p['candidates'])):
                raise ValueError('Incomplete quality plan/result')
            if key in quality:
                raise ValueError('Duplicate dataset/seed quality job')
            quality[key] = d
        else:
            d = json.loads((directory/'timing.json').read_text())
            if (not d['complete'] or digest(directory/'timing.json') != m['timing_sha256'] or
                {t['method'] for t in d['trials']} != set(ARMS) or len(d['trials']) != 5):
                raise ValueError('Incomplete timing job')
            key = (*key, m['k_layers'], m['policy'])
            if key in timing:
                raise ValueError('Duplicate standalone timing cell')
            timing[key] = d
        inputs.append(dict(directory=str(directory.resolve()), mode=m['mode'], manifest_sha256=digest(directory/'study_manifest.json')))
    if len(snapshots) != 1:
        raise ValueError('Mixed code/package/quantizer snapshots')
    flat = [row for d in quality.values() for row in quality_rows(d)]
    primary = [r for r in flat if r['tie_order'] in ('depth_ascending', 'native')]
    time_rows = []
    for key, d in timing.items():
        if key[:2] not in quality:
            raise ValueError('Timing cell requires its matching quality job for assignment parity')
        q = quality[key[:2]]
        for trial in d['trials']:
            for budget, a in trial['allocations'].items():
                matches = [c for c in q['candidates'] for r in c['roles']
                    if (r['method'], r['k_layers'], r['policy'], r['budget'], r['tie_order']) ==
                    (trial['method'], key[2], key[3], float(budget),
                     'depth_ascending' if trial['method'] in ARMS[:2] else 'native')]
                if len(matches) != 1 or matches[0]['bit_assignment'] != a['bit_assignment']:
                    raise ValueError('Standalone timing assignment differs from quality primary assignment')
                time_rows.append(dict(dataset=key[0], seed=key[1], k_layers=key[2], policy=key[3],
                    method=trial['method'], budget=float(budget),
                    calibration_to_budget_sec=a['calibration_to_budget_sec'],
                    calibration_wall_both_budgets_sec=trial['calibration_wall_sec'],
                    primary_ppl=matches[0]['evaluation']['ppl'], assignment_parity=True))
    paired = []
    for row in primary:
        if row['policy'] != 'stratified':
            continue
        other = [r for r in primary if (r['dataset'], r['seed'], r['k_layers'], r['method'], r['budget'], r['policy']) ==
            (row['dataset'], row['seed'], row['k_layers'], row['method'], row['budget'], 'boundary_inclusive')]
        if len(other) != 1:
            raise ValueError('Incomplete acquisition-policy pair')
        paired.append(dict(dataset=row['dataset'], seed=row['seed'], k_layers=row['k_layers'],
            method=row['method'], budget=row['budget'], stratified_ppl=row['ppl'],
            boundary_ppl=other[0]['ppl'], boundary_vs_stratified_percent=100*(other[0]['ppl']/row['ppl']-1),
            identical_acquisition=row['acquired_layers'] == other[0]['acquired_layers']))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'primary_curves.csv', primary); write_csv(output/'paired_acquisition.csv', paired)
    write_csv(output/'standalone_timing_with_quality.csv', time_rows)
    expected_quality = {(d, s) for d in ['wikitext2', 'c4'] for s in [42, 100, 2026]}
    expected_timing = {(d, s, k, p) for d, s in expected_quality for k in [2, 4, 8] for p in ['stratified', 'boundary_inclusive']}
    save(output/'input_manifest.json', dict(schema='coverage_aggregate_v1', inputs=inputs,
        complete_quality_matrix=set(quality) == expected_quality,
        complete_timing_matrix=set(timing) == expected_timing,
        interpretation='Per-seed acquisition pairs. Random placement orders are not calibration seeds. Missing timings are not estimated.'))
    return dict(quality_jobs=len(quality), timing_cells=len(timing), timing_assignment_checks=len(time_rows))
