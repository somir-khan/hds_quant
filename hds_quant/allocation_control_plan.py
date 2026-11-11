"""Immutable source validation and assignment plans for the seed42 feature diagnostic."""
import math

from .allocation_controls import (SETTINGS, TIE_ORDERS, allocation_costs,
    family_mean_predictions, family_tie_order, last_down, simple_assignment, solve_control)
from .amq_followup_plan import METHODS, assignment, geometry, load_source, payload
from .config import ExperimentConfig
from .run_no1bit_diagnostic import decode_pairs, reconstruct_costs

METADATA_METHODS = ('Scalar-Meta-Ridge', 'Scalar-Meta-Depth-Ridge')


def validate_spec(spec):
    expected = dict(schema='allocation_controls_v1', seed=42, budgets=[2.5, 3.0, 3.5],
        settings=list(SETTINGS), family_tie_orders=list(TIE_ORDERS),
        metadata_methods=list(METADATA_METHODS), simple_patterns=['last_half', 'alternating_odd'])
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f'Frozen diagnostic policy differs: {key}')
    if not spec.get('experiment_id') or not spec.get('source_dir'):
        raise ValueError('Missing experiment ID/source')


def add_candidate(plan, bits, meta, role, historical=None):
    actual = payload(bits, meta)/sum(m['num_params'] for m in meta)
    if not math.isclose(actual, role['budget'], abs_tol=1e-12):
        raise ValueError('Candidate does not match the declared exact budget')
    if role['setting'] == 'last_down4' and bits[last_down(meta)] != 4:
        raise ValueError('Candidate violates final-down protection')
    for row in plan['candidates']:
        if row['bit_assignment'] == bits:
            break
    else:
        row = dict(candidate_id=f'candidate_{len(plan["candidates"]):03d}',
            bit_assignment=dict(bits), actual_payload_bits=actual,
            roles=[], historical_evaluations=[])
        plan['candidates'].append(row)
    row['roles'].append(role)
    if historical is not None:
        row['historical_evaluations'].append(historical)


def add_estimator(plan, method, costs, meta, spec, measured_names=(), trial=None):
    for budget in spec['budgets']:
        key = f'{budget:.2f}'
        for setting in spec['settings']:
            protected = setting == 'last_down4'
            historical = None
            if trial is not None and key in trial['allocations']:
                original = trial['allocations'][key]
                original_cost = math.fsum(costs[n, b] for n, b in original['bit_assignment'].items())
                if not math.isclose(original_cost, original['total_predicted_cost'], abs_tol=1e-9, rel_tol=1e-9):
                    raise ValueError('Reconstructed objective differs from source')
                if not protected or original['bit_assignment'][last_down(meta)] == 4:
                    bits = dict(original['bit_assignment'])
                    historical = original['evaluation']
                else:
                    from .amq_followup_plan import fixed4_intervention
                    bits = fixed4_intervention(trial, key, meta,
                        ExperimentConfig(**plan['experiment_config']), [last_down(meta)])['bit_assignment']
            else:
                bits = solve_control(costs, meta, budget, protected)
            orders = spec['family_tie_orders'] if method == 'Family-Mean' else ['native']
            for order in orders:
                chosen = family_tie_order(bits, costs, meta, measured_names, order,
                    [last_down(meta)] if protected else []) if method == 'Family-Mean' else bits
                objective = math.fsum(costs[n, b] for n, b in chosen.items())
                add_candidate(plan, chosen, meta, dict(method=method, budget=budget, setting=setting,
                    tie_order=order, predicted_cost=objective, source='saved_labels_and_predictions'
                    if trial else 'saved_labels_new_control'), historical)


def build_plan(spec, source):
    validate_spec(spec)
    src, raw, saved, hashes, screened_fixed = load_source(source, spec['seed'])
    cfg = ExperimentConfig(**raw['config']); meta = geometry()
    if (cfg.readout_grouping, cfg.target_transform, cfg.residual_anchor, cfg.coord_binding) != (
            'per_bit', 'signed_log1p', 'int4_aligned', 'salience_bucket'):
        raise ValueError('Unexpected source feature/readout policy')
    trials = {t['method']: t for t in raw['trials'] if t['method'] in METHODS}
    measured = decode_pairs(trials['HDS-Hybrid']['measured_labels'])
    acquired = trials['HDS-Hybrid']['acquired_layers']
    for trial in trials.values():
        other = decode_pairs(trial['measured_labels'])
        if trial['acquired_layers'] != acquired or other.keys() != measured.keys() or any(
                not math.isclose(other[k], measured[k], rel_tol=0, abs_tol=1e-9) for k in measured):
            raise ValueError('Learned methods must share acquired blocks and measured labels')
        reconstruct_costs(trial, meta, cfg)  # Enforce exact measured/held-out coverage.
    plan = dict(schema='allocation_controls_plan_v1', seed=spec['seed'],
        source_spec=src, source_hashes=hashes, experiment_config=raw['config'],
        acquired_layers=acquired, probe_label_count=len(measured),
        protection_projection=last_down(meta), screened_fixed4_source_context=screened_fixed,
        candidates=[], metadata_pending=True,
        timing_scope='Diagnostic reuses labels/predictions; no end-to-end calibration speed claim.',
        tie_policy='Within-family equal-cost/equal-size unmeasured placements; preserve ILP family bit counts. Primary depth ascending, SHA256 random seeds 17/29 sensitivity only. Report all, never select by test PPL.',
        constraint_scope='Final-block down4 is a declared post-hoc structural prior, not independently validated or newly screened.')
    for method in METHODS:
        add_estimator(plan, method, reconstruct_costs(trials[method], meta, cfg), meta, spec, trial=trials[method])
    family_raw = family_mean_predictions(measured, meta, cfg.marginal_bits)
    add_estimator(plan, 'Family-Mean', allocation_costs(family_raw, meta, cfg), meta, spec,
                  measured_names={n for n, b in measured})
    for budget in spec['budgets']:
        for pattern in spec['simple_patterns']:
            for setting in spec['settings']:
                bits = simple_assignment(meta, budget, pattern, setting == 'last_down4')
                add_candidate(plan, bits, meta, dict(method=f'Simple-{pattern}', budget=budget,
                    setting=setting, tie_order='declared', predicted_cost=None,
                    source='no_labels_no_features'))
    amq = next(t for t in raw['trials'] if t['method'] == 'AMQ-HQQ-BudgetCap-v2')
    for budget, arch in zip([3.0, 3.5], amq['initialization_anchors']):
        bits = assignment(arch)
        if not any(assignment(a) == bits for a, _, _ in saved['archive']):
            raise ValueError('Anchor not present in completed source archive')
        add_candidate(plan, bits, meta, dict(method='AMQ-screened-anchor', budget=budget,
            setting='last_down4', tie_order='source', predicted_cost=None,
            source='saved_anchor_depends_on_AMQ_screening'))
    return plan, raw, measured, family_raw


def fit_metadata_controls(features, measured, meta, cfg, acquired, seed):
    """Exact nine appended HDS values, plus an explicitly separate depth variant."""
    import torch
    from .readout_cv import evaluate_fixed_k_layer_acquisition
    output = {}
    if set(features) != {(m['name'], b) for m in meta for b in cfg.marginal_bits}:
        raise ValueError('Metadata features must cover every projection/probe bit')
    max_depth = max(m['layer_idx'] for m in meta)
    by_name = {m['name']: m for m in meta}
    for method in METADATA_METHODS:
        selected = {}
        for key, values in features.items():
            vector = values['Scalar-Meta-Ridge']
            if vector.numel() != 9:
                raise ValueError('Expected exactly the nine HDS metadata columns')
            if method == 'Scalar-Meta-Depth-Ridge':
                depth = by_name[key[0]]['layer_idx']/max(1, max_depth)
                vector = torch.cat([vector, vector.new_tensor([depth])])
            selected[key] = {'Scalar-Meta-Ridge': vector}
        fit = evaluate_fixed_k_layer_acquisition(selected, measured, meta, cfg,
            k_layers=len(acquired), seed=seed, estimator_names=('Scalar-Meta-Ridge',))
        if fit['acquired_layers'] != acquired:
            raise ValueError('Metadata fit changed the acquired block set')
        output[method] = dict(raw=fit['raw_predictions']['Scalar-Meta-Ridge'],
            costs=fit['alloc_costs']['Scalar-Meta-Ridge'],
            fit_sec=fit['estimator_readout_time_sec']['Scalar-Meta-Ridge'],
            feature_columns=9 if method == 'Scalar-Meta-Ridge' else 10)
    return output
