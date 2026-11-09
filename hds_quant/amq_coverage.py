"""Versioned budget-cap selection and exact-payload initialization anchors."""
import math
from functools import reduce

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def select_under_cap(archive, budget, group_size):
    candidates = [x for x in archive if 2 <= x[2]-32/group_size <= budget+1e-10]
    return min(candidates, key=lambda x: (x[1], -x[2])) if candidates else None


def exact_anchor(config, fixed, budget, seed):
    """Find a deterministic near-uniform assignment at exact payload, preserving fixed4.

    This is an explicitly identified initialization addition, not upstream AMQ.
    The objective uses geometry/seed only, never loss labels or evaluation data.
    """
    names = [(i, f) for f in config['linear'] for i in range(config['n_block'])]
    weights = [math.prod(config['linear_shape'][f]) for i, f in names]
    unit = reduce(math.gcd, weights)
    units = np.array(weights, dtype=float)/unit
    target = budget*sum(weights)/unit
    if not math.isclose(target, round(target), abs_tol=1e-7):
        raise ValueError('Exact anchor payload is unattainable')
    n = len(names)
    matrix = lil_matrix((n+1, 3*n))
    upper = np.ones(3*n)
    for j, (i, f) in enumerate(names):
        matrix[j, 3*j:3*j+3] = 1
        matrix[n, 3*j:3*j+3] = units[j]*np.array([2,3,4])
        if f'{i}.{f}' in fixed:
            upper[3*j:3*j+2] = 0
    rng = np.random.default_rng(seed)
    cost = np.concatenate([u*(np.array([2,3,4])-budget)**2 for u in units])
    cost += rng.random(3*n)*1e-5
    rhs = np.r_[np.ones(n), round(target)]
    solved = milp(cost, integrality=np.ones(3*n), bounds=Bounds(0, upper),
                  constraints=LinearConstraint(matrix.tocsr(), rhs, rhs),
                  options={'time_limit':60, 'mip_rel_gap':0})
    if not solved.success:
        raise RuntimeError(f'Exact budget anchor failed: {solved.message}')
    bits = np.argmax(solved.x.reshape(n,3), axis=1)+2
    if sum(w*int(b) for w,b in zip(weights,bits)) != round(budget*sum(weights)):
        raise RuntimeError('Anchor payload failed integer verification')
    result = {'linear': {f:[0]*config['n_block'] for f in config['linear']}}
    for (i,f), bit in zip(names,bits):
        result['linear'][f][i] = int(bit)
    return result


def diagnostic_selection(archive, budgets, group_size):
    """Freeze old-window winner and top two under-cap JSD candidates per budget."""
    selected = {}
    for budget in budgets:
        feasible = [i for i,x in enumerate(archive) if 2 <= x[2]-32/group_size <= budget+1e-10]
        ranked = sorted(feasible, key=lambda i:(archive[i][1],-archive[i][2]))
        narrow = [i for i in feasible if archive[i][2]-32/group_size >= budget-.005]
        policies = [(i,f'top{rank+1}_under_{budget}') for rank,i in enumerate(ranked[:2])]
        if narrow:
            policies.append((min(narrow,key=lambda i:archive[i][1]),f'old_window_{budget}'))
        for i, reason in policies:
            selected.setdefault(i, []).append(reason)
    # Uniform-4 initialization is a quantizer/teacher reference, not a budget candidate.
    four = next((i for i,x in enumerate(archive) if all(b==4 for bs in x[0]['linear'].values() for b in bs)), None)
    if four is not None:
        selected.setdefault(four,[]).append('all4_reference')
    return [{'archive_index':i, 'reasons':reasons, 'architecture':archive[i][0],
             'archived_jsd':archive[i][1], 'payload_bits':archive[i][2]-32/group_size}
            for i,reasons in sorted(selected.items())]
