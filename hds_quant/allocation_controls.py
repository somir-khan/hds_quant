"""Calibration-only controls; never use held-out losses to choose assignments."""
import hashlib
import math
from collections import defaultdict

from .allocator import solve_bit_allocation_ilp
from .amq_followup_plan import payload
from .readout_cv import _build_alloc_costs

BITS = (2, 3, 4)
TIE_ORDERS = ('depth_ascending', 'random_17', 'random_29')
SETTINGS = ('unconstrained', 'last_down4')


def family_mean_predictions(measured, meta, marginal_bits):
    """Mean raw acquired labels per family/bit; retain each individual known label."""
    expected_names = {m['name'] for m in meta}
    if any(n not in expected_names or b not in marginal_bits or not math.isfinite(v)
           for (n, b), v in measured.items()):
        raise ValueError('Invalid acquired labels')
    pools = defaultdict(list)
    by_name = {m['name']: m for m in meta}
    for (name, bit), value in measured.items():
        pools[by_name[name]['family_idx'], bit].append(value)
    result = {}
    for m in meta:
        for bit in marginal_bits:
            values = pools[m['family_idx'], bit]
            if not values:
                raise ValueError('Every family/bit requires acquired labels')
            result[m['name'], bit] = measured.get((m['name'], bit), math.fsum(values)/len(values))
    return result


def allocation_costs(raw, meta, cfg):
    return _build_alloc_costs({'control': raw}, meta, ('control',),
        marginal_bits=cfg.marginal_bits, reference_bit=cfg.reference_bit)['control']


def last_down(meta):
    last = max(m['layer_idx'] for m in meta)
    names = [m['name'] for m in meta if m['layer_idx'] == last and m['name'].endswith('.down_proj')]
    if len(names) != 1:
        raise ValueError('Expected one final-block down projection')
    return names[0]


def solve_control(costs, meta, budget, protection=False):
    """Common exact-payload ILP with an optional structural, explicitly post-hoc prior."""
    total = sum(m['num_params'] for m in meta)
    target_float = budget * total
    target = round(target_float)
    if abs(target_float-target) > 1e-6:
        raise ValueError('Budget is not an integer payload')
    fixed = {last_down(meta): 4} if protection else {}
    free = [m for m in meta if m['name'] not in fixed]
    fixed_payload = sum(m['num_params']*fixed[m['name']] for m in meta if m['name'] in fixed)
    free_total = sum(m['num_params'] for m in free)
    if not free_total:
        raise ValueError('No free projections')
    solved = solve_bit_allocation_ilp(costs, free, (target-fixed_payload)/free_total, candidate_bits=BITS)
    if not solved['success']:
        raise RuntimeError('Control allocation failed')
    bits = dict(solved['bit_assignment'], **fixed)
    if payload(bits, meta) != target or any(bits[n] != b for n, b in fixed.items()):
        raise ValueError('Control does not achieve the exact requested payload')
    return bits


def family_tie_order(bits, costs, meta, measured_names, order, protected=(), *, allowed_orders=None):
    """Permute equal-cost, equal-size unlabeled projections within each family.

    Keeps the ILP's per-family precision counts and primary objective exactly.
    This tests block-placement ties, not alternative optimal family-count solutions.
    SHA256 seeded orders avoid dependence on Python hash randomization or RNG state.
    """
    if order not in (TIE_ORDERS if allowed_orders is None else allowed_orders):
        raise ValueError('Unknown tie ordering')
    groups = defaultdict(list)
    for m in meta:
        n = m['name']
        if n not in measured_names and n not in protected:
            key = (m['family_idx'], m['num_params'], *(costs[n, b] for b in BITS))
            groups[key].append(m)
    result = dict(bits)
    for group in groups.values():
        if order == 'depth_ascending':
            group.sort(key=lambda m: (m['layer_idx'], m['name']))
        else:
            seed = order.split('_')[1]
            group.sort(key=lambda m: (hashlib.sha256(f'{seed}|{m["name"]}'.encode()).hexdigest(), m['name']))
        values = sorted((bits[m['name']] for m in group), reverse=True)
        for m, bit in zip(group, values):
            result[m['name']] = bit
    before = math.fsum(costs[n, b] for n, b in bits.items())
    after = math.fsum(costs[n, b] for n, b in result.items())
    if payload(bits, meta) != payload(result, meta) or not math.isclose(before, after, abs_tol=1e-12, rel_tol=1e-12):
        raise RuntimeError('Tie permutation changed payload/objective')
    return result


def simple_assignment(meta, budget, pattern, protection=False):
    """Declared depth patterns, no labels/features. Protection compensated at first eligible gate."""
    layers = sorted({m['layer_idx'] for m in meta})
    if len(layers) % 2 or layers != list(range(len(layers))):
        raise ValueError('Simple controls require an even, contiguous block count')
    sizes = [sum(m['num_params'] for m in meta if m['layer_idx'] == i) for i in layers]
    if len(set(sizes)) != 1 or budget not in (2.5, 3.0, 3.5):
        raise ValueError('Simple controls require identical block payloads and declared budgets')
    if pattern not in ('last_half', 'alternating_odd'):
        raise ValueError('Unknown simple pattern')
    low = math.floor(budget)
    high_layers = set(layers[len(layers)//2:] if pattern == 'last_half' else layers[1::2])
    bits = {m['name']: low + int(budget != 3 and m['layer_idx'] in high_layers) for m in meta}
    if protection:
        down = last_down(meta)
        size = next(m['num_params'] for m in meta if m['name'] == down)
        owed = 4-bits[down]
        bits[down] = 4
        for m in sorted(meta, key=lambda m: (m['layer_idx'], m['name'])):
            n = m['name']
            if n.endswith('.gate_proj') and m['num_params'] == size:
                take = min(owed, bits[n]-2)
                bits[n] -= take
                owed -= take
            if owed == 0:
                break
        if owed:
            raise ValueError('Cannot compensate simple protection within the budget')
    if payload(bits, meta) != round(budget*sum(m['num_params'] for m in meta)):
        raise ValueError('Simple control payload mismatch')
    return bits
