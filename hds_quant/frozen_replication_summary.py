"""Paired descriptive reporting: calibration seeds and placement orders stay separate."""
import csv
import json
import statistics
from pathlib import Path

from .frozen_replication_plan import ORDERS


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def materiality(delta, margin=1.0):
    if delta <= -margin:
        return 'HDS_lower_by_at_least_margin'
    if delta >= margin:
        return 'HDS_higher_by_at_least_margin'
    return 'below_practical_margin_not_equivalence'


def summarize_replication(report, output):
    rows = []
    for c in report['candidates']:
        for role in c['roles']:
            rows.append(dict(dataset=report['dataset'], seed=report['seed'],
                candidate_id=c['candidate_id'], **role, actual_bits=c['actual_payload_bits'],
                ppl=c['evaluation']['ppl'], nll=c['evaluation']['nll'],
                reuses_labels=True, end_to_end_calibration_sec=None))
    write_csv(output/'comparisons.csv', rows)
    pairs = []
    for row in rows:
        hds = [r for r in rows if r['method'] == 'HDS-Hybrid' and r['budget'] == row['budget'] and r['setting'] == row['setting']]
        if len(hds) == 1:
            delta = 100*(hds[0]['ppl']/row['ppl']-1)
            pairs.append(dict(dataset=report['dataset'], seed=report['seed'],
                method=row['method'], budget=row['budget'], setting=row['setting'],
                tie_order=row['tie_order'], control_ppl=row['ppl'], hds_ppl=hds[0]['ppl'],
                hds_vs_control_ppl_percent=delta, materiality=materiality(delta)))
    write_csv(output/'hds_vs_controls.csv', pairs)
    placement = []
    for budget in [2.5, 3., 3.5]:
        for setting in ['unconstrained', 'last_down4']:
            family = [r for r in rows if r['method'] == 'Family-Mean' and r['budget'] == budget and r['setting'] == setting]
            if len(family) != len(ORDERS):
                continue  # A partial evaluation is not a complete placement sensitivity result.
            primary = next(r for r in family if r['tie_order'] == 'depth_ascending')
            random = [r for r in family if r['tie_order'] != 'depth_ascending']
            hds = [r for r in rows if r['method'] == 'HDS-Hybrid' and r['budget'] == budget and r['setting'] == setting]
            if len(hds) != 1:
                continue
            placement.append(dict(dataset=report['dataset'], seed=report['seed'], budget=budget,
                setting=setting, primary_family_ppl=primary['ppl'], hds_ppl=hds[0]['ppl'],
                primary_hds_vs_family_percent=100*(hds[0]['ppl']/primary['ppl']-1),
                random_order_count=len(random), unique_random_assignments=len({r['candidate_id'] for r in random}),
                random_ppl_min=min(r['ppl'] for r in random), random_ppl_median=statistics.median(r['ppl'] for r in random),
                random_ppl_max=max(r['ppl'] for r in random),
                random_orders_hds_beats=sum(hds[0]['ppl'] < r['ppl'] for r in random)))
    write_csv(output/'placement_sensitivity.csv', placement)
    lines = [f'Complete: {report["complete"]}',
        f'Frozen replication: {report["dataset"]}, calibration seed {report["seed"]}, K=4, 84 reused labels.',
        'Primary Family-Mean ordering: depth ascending. Ten random SHA256 orderings are sensitivity controls, not seed replications.',
        'Negative hds_vs_control_ppl_percent favors HDS. Practical margin: 1%; below-margin is not equivalence.',
        'Already protected unconstrained assignments are preserved exactly for every method/order.',
        'Saved labels/predictions reused; metadata features and fitting are fresh. No end-to-end speed claim.',
        'No new AMQ, no new labels, no boundary-inclusive acquisition change.',
        'Every assignment was frozen before held-out scoring. Partial files are not completed experiments.']
    lines += [str(r) for r in placement]
    (output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def aggregate(directories, output):
    """Require one completed case per dataset/seed; never silently mix protocols/jobs."""
    from .amq_followup_plan import digest
    from .run_amq_followup import save
    from .frozen_replication_plan import validate_spec
    seen = set(); signatures = {}; rows = []; inputs = []; runtime_signature = None
    for directory in directories:
        directory = Path(directory)
        manifest = json.loads((directory/'diagnostic_manifest.json').read_text())
        report = json.loads((directory/'diagnostic.json').read_text())
        spec = json.loads((directory/'experiment_config.json').read_text())
        validate_spec(spec, report['seed'])
        if (manifest['schema'] != 'frozen_replication_v1' or not manifest['complete'] or
            not manifest['plan_complete'] or manifest['plan_only'] or not report['complete'] or
            report['schema'] != 'frozen_replication_result_v1' or report['dataset'] != spec['dataset'] or
            manifest['seed'] != report['seed'] or manifest['dataset'] != report['dataset'] or
            manifest['config_sha256'] != digest(directory/'experiment_config.json') or
            manifest['selection_plan_sha256'] != digest(directory/'selection_plan.json')):
            raise ValueError('Expected completed, hash-verified frozen replication')
        key = (report['dataset'], report['seed'])
        runtime = json.dumps(dict(code=manifest['code_sha256'], packages=manifest['packages'],
            quantizer=manifest['effective_hqq_config']), sort_keys=True)
        if runtime_signature is not None and runtime != runtime_signature:
            raise ValueError('Mixed code/package/quantizer snapshots; aggregate a matched protocol')
        runtime_signature = runtime
        if key in seen:
            raise ValueError('Duplicate dataset/calibration seed; choose one job explicitly')
        seen.add(key)
        signature = json.dumps(dict(spec, source_dir='PATH', experiment_id='ID'), sort_keys=True)
        old = signatures.setdefault(report['dataset'], signature)
        if signature != old:
            raise ValueError('Mixed frozen protocols within a dataset')
        inputs.append(dict(directory=str(directory.resolve()), dataset=key[0], seed=key[1],
            diagnostic_sha256=digest(directory/'diagnostic.json'), manifest_sha256=digest(directory/'diagnostic_manifest.json')))
        for c in report['candidates']:
            for role in c['roles']:
                rows.append(dict(dataset=key[0], seed=key[1], **role, candidate_id=c['candidate_id'], ppl=c['evaluation']['ppl']))
    pairs = []; placement = []
    for dataset in sorted({key[0] for key in seen}):
        for budget in [2.5, 3., 3.5]:
            for setting in ['unconstrained', 'last_down4']:
                deltas = []
                for seed in [42, 100, 2026]:
                    group = [r for r in rows if (r['dataset'], r['seed'], r['budget'], r['setting']) == (dataset, seed, budget, setting)]
                    if not group:
                        continue
                    hds = [r for r in group if r['method'] == 'HDS-Hybrid']
                    family = [r for r in group if r['method'] == 'Family-Mean']
                    if len(hds) != 1 or len(family) != len(ORDERS) or {r['tie_order'] for r in family} != set(ORDERS):
                        raise ValueError('Incomplete/duplicate method or family placement matrix')
                    primary = next(r for r in family if r['tie_order'] == 'depth_ascending')
                    random = [r['ppl'] for r in family if r['tie_order'] != 'depth_ascending']
                    delta = 100*(hds[0]['ppl']/primary['ppl']-1); deltas.append(delta)
                    pairs.append(dict(dataset=dataset, seed=seed, budget=budget, setting=setting,
                        hds_ppl=hds[0]['ppl'], primary_family_ppl=primary['ppl'],
                        hds_vs_family_percent=delta, materiality=materiality(delta),
                        random_family_ppl_min=min(random), random_family_ppl_median=statistics.median(random),
                        random_family_ppl_max=max(random)))
                if len(deltas) != 3:
                    decision = 'incomplete_three_seed_matrix'
                elif all(v <= -1 for v in deltas):
                    decision = 'consistent_material_HDS_advantage_on_primary'
                elif all(v >= 1 for v in deltas):
                    decision = 'consistent_material_FamilyMean_advantage_on_primary'
                elif all(abs(v) < 1 for v in deltas):
                    decision = 'all_primary_differences_below_margin_not_equivalence'
                else:
                    decision = 'mixed_or_inconclusive'
                placement.append(dict(dataset=dataset, budget=budget, setting=setting,
                    calibration_seed_count=len(deltas), mean_paired_percent=statistics.mean(deltas) if deltas else None,
                    decision=decision))
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    write_csv(output/'paired_seed_results.csv', pairs)
    write_csv(output/'decisions.csv', placement)
    save(output/'input_manifest.json', dict(schema='frozen_replication_aggregate_v1', inputs=inputs,
        complete_seed_matrix=all({s for d, s in seen if d == dataset} == {42, 100, 2026} for dataset in signatures),
        interpretation='Descriptive practical-margin decisions per dataset/budget/setting; no p-values, equivalence claim, best-order selection or pooling placement orders as seeds.'))
    return placement
