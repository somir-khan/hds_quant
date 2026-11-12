"""Declared coverage policies, raw pooling and protected allocation rules."""
import math
import statistics

from .allocation_control_plan import add_candidate
from .allocation_controls import allocation_costs, family_tie_order, last_down, solve_control
from .frozen_replication_plan import ORDERS
from .readout_cv import select_stratified_k_layers

ARMS = ('Family-Mean', 'Family-Median', 'Scalar-Meta-Ridge', 'Coord-Rank-Hist', 'HDS-Hybrid')
FEATURE_ARMS = ARMS[2:]
POLICIES = ('stratified', 'boundary_inclusive')


def validate_spec(spec, seed, mode, k=None, policy=None):
    expected = dict(schema='coverage_study_v1', seeds=[42, 100, 2026], k_layers=[2, 4, 8],
        acquisition_policies=list(POLICIES), methods=list(ARMS), budgets=[3., 3.5],
        family_tie_orders=list(ORDERS), protection='last_down4',
        acquisition_rule='replace_extreme_strata_with_endpoints', materiality_percent=1.0)
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f'Frozen coverage policy differs: {key}')
    if seed not in expected['seeds'] or mode not in ('quality', 'timing'):
        raise ValueError('Invalid seed/mode')
    if mode == 'timing' and (k not in expected['k_layers'] or policy not in POLICIES):
        raise ValueError('Timing requires one declared K and policy')
    if spec.get('dataset') not in ('wikitext2', 'c4') or not spec.get('source_dir'):
        raise ValueError('Missing dataset/source')


def acquired_layers(num_layers, k, seed, policy):
    if k < 2 or k > num_layers or policy not in POLICIES:
        raise ValueError('Invalid coverage acquisition')
    layers = select_stratified_k_layers(list(range(num_layers)), k, seed)
    if policy == 'boundary_inclusive':
        layers = sorted([0, *layers[1:-1], num_layers-1])
    if len(layers) != k or len(set(layers)) != k:
        raise ValueError('Acquisition did not preserve the label budget')
    return layers


def acquisition_plan(spec, seed, num_layers=32):
    cells = [dict(k_layers=k, policy=policy,
        acquired_layers=acquired_layers(num_layers, k, seed, policy), probe_label_count=21*k)
        for k in spec['k_layers'] for policy in POLICIES]
    return dict(schema='coverage_acquisition_plan_v1', seed=seed, cells=cells,
        union_layers=sorted({i for c in cells for i in c['acquired_layers']}),
        note='Separate K sets are not nested. Boundary policy changes only the extreme-stratum selections.')


def mask_labels(labels, meta, layers, marginal_bits):
    wanted = {(m['name'], b) for m in meta if m['layer_idx'] in layers for b in marginal_bits}
    if not wanted <= labels.keys():
        raise ValueError('Missing acquired labels')
    selected = {key: labels[key] for key in sorted(wanted)}
    if any(not math.isfinite(v) for v in selected.values()):
        raise ValueError('Non-finite acquired labels')
    return selected


def pooled_predictions(measured, meta, marginal_bits, method):
    if method not in ARMS[:2]:
        raise ValueError('Unknown pooling arm')
    pools = {}
    for m in meta:
        for b in marginal_bits:
            key = (m['name'], b)
            if key in measured:
                pools.setdefault((m['family_idx'], b), []).append(measured[key])
    raw = {}
    for m in meta:
        for b in marginal_bits:
            values = sorted(pools.get((m['family_idx'], b), []))
            if not values:
                raise ValueError('Every family/precision requires acquired labels')
            # Define the two-point median with identical arithmetic to the mean arm.
            value = math.fsum(values)/len(values) if method == 'Family-Mean' or len(values) <= 2 else statistics.median(values)
            raw[m['name'], b] = measured.get((m['name'], b), value)
    return raw


def protected_allocations(costs, meta, budget, measured_names=(), family=False, orders=None):
    """Preserve every already-feasible unconstrained placement; never add free payload."""
    down = last_down(meta)
    orders = tuple(orders or (ORDERS if family else ('native',)))
    base = solve_control(costs, meta, budget); constrained = None; out = {}
    for order in orders:
        before = family_tie_order(base, costs, meta, measured_names, order, allowed_orders=ORDERS) if family else dict(base)
        if before[down] == 4:
            after = dict(before)
        else:
            if constrained is None:
                constrained = solve_control(costs, meta, budget, True)
            after = family_tie_order(constrained, costs, meta, measured_names, order,
                [down], allowed_orders=ORDERS) if family else dict(constrained)
        if math.fsum(costs[n, b] for n, b in after.items()) < math.fsum(costs[n, b] for n, b in before.items())-1e-8:
            raise ValueError('Constrained objective unexpectedly improved')
        out[order] = dict(bit_assignment=after,
            protection_changed_projections=sum(before[n] != after[n] for n in before))
    return out


def append_allocations(plan, method, raw, meta, cfg, spec, cell, cache):
    costs = allocation_costs(raw, meta, cfg)
    family = method in ARMS[:2]
    names = {m['name'] for m in meta if m['layer_idx'] in cell['acquired_layers']}
    # Exact equal cost banks share the same solver result, including mean==median at K=2.
    signature = (tuple(cell['acquired_layers']), family,
        tuple((n, b, costs[n, b]) for n in sorted(m['name'] for m in meta) for b in (2, 3, 4)))
    for budget in spec['budgets']:
        key = (signature, budget)
        if key not in cache:
            cache[key] = protected_allocations(costs, meta, budget, names, family)
        for order, allocation in cache[key].items():
            bits = allocation['bit_assignment']
            add_candidate(plan, bits, meta, dict(method=method, budget=budget, setting='last_down4',
                tie_order=order, policy=cell['policy'], k_layers=cell['k_layers'],
                acquired_layers=cell['acquired_layers'], probe_label_count=cell['probe_label_count'],
                predicted_cost=math.fsum(costs[n, b] for n, b in bits.items()),
                protection_changed_projections=allocation['protection_changed_projections'],
                source='fresh_masked_coverage_labels'))
