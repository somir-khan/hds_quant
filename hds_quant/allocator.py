"""Exact multiple-choice knapsack bit allocation via ``scipy.optimize.milp``."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

# Module-level cache for estimator-independent optimal feasible payload (B_star, K_star)
_FEASIBLE_PAYLOAD_CACHE: Dict[Tuple[Tuple[float, ...], Tuple[int, ...], float], Tuple[float, int]] = {}


def solve_bit_allocation_ilp(
    alloc_costs: Dict[Tuple[str, int], float],
    meta_list: List[dict],
    target_avg_bits: float,
    candidate_bits: Tuple[int, ...] = (2, 3, 4),
    enforce_exact_payload: bool = True,
    payload_slack_steps: float = 1.0,
) -> dict:
    """Choose one bit-width per projection minimizing total predicted cost under a weight-payload budget.

    minimize   sum_{p,b} cost[p, b] * z[p, b]
    subject to sum_b z[p, b] == 1                                   for every projection p
               lb <= sum_{p,b} b * N_p * z[p, b] / sum_p N_p <= ub
               z binary

    ``enforce_exact_payload`` (default ``True``) solves a two-stage MILP:
      1. Stage 1 (estimator-independent, cached): maximizes the achieved bit-width subject to
         ``avg_bits <= target_avg_bits`` to determine the unique optimal feasible payload ``B_star``
         and total bit-parameter sum ``K_star = sum_{p,b} b * N_p * z_{p,b}``.
      2. Stage 2 (cost minimization): minimizes predicted cost subject to an exact equality constraint
         on the total bit-payload (``sum_{p,b} b * N_p * z_{p,b} == K_star``), with a tie-breaker
         ``-1e-7 * (N_p / sum N) * b``.
    This guarantees that at any budget ``target_avg_bits``, EVERY estimator achieves the exact same
    weight payload to the last bit, regardless of non-monotone or negative costs.

    ``enforce_exact_payload=False`` recovers the plain upper-bound-only multiple-choice knapsack.

    Returns ``bit_assignment`` ({name: bits}), ``achieved_avg_bits`` (parameter-weighted), ``total_predicted_cost``,
    ``bit_counts`` ({bits: #projections}), ``success`` and ``payload_lower_bound``.
    """
    P, B = len(meta_list), len(candidate_bits)
    if P == 0:
        raise ValueError("meta_list is empty")
    names = [m["name"] for m in meta_list]
    n_params = np.array([float(m["num_params"]) for m in meta_list])
    total_params = float(n_params.sum())
    bits = np.array(candidate_bits, dtype=np.float64)

    try:
        c = np.array([alloc_costs[(n, int(b))] for n in names for b in candidate_bits], dtype=np.float64)
    except KeyError as e:
        raise KeyError(f"alloc_costs is missing entry {e.args[0]}") from None

    target = float(target_avg_bits)
    if target < float(bits.min()):
        return {
            "bit_assignment": {},
            "achieved_avg_bits": float("nan"),
            "total_predicted_cost": float("nan"),
            "bit_counts": {int(b): 0 for b in candidate_bits},
            "success": False,
            "payload_lower_bound": float("nan"),
        }

    budget_row = np.outer(n_params / total_params, bits).reshape(1, P * B)
    param_bits_row = np.outer(n_params, bits).reshape(1, P * B)

    A_eq = np.zeros((P, P * B))
    for p in range(P):
        A_eq[p, p * B : (p + 1) * B] = 1.0

    c_obj = c
    constraints = [LinearConstraint(A_eq, lb=1.0, ub=1.0)]
    budget_lb = -np.inf

    if enforce_exact_payload:
        c_obj = c - 1e-7 * budget_row.reshape(-1)  # prefer more bits among (near-)tied solutions

        if target >= float(bits.max()):
            B_star = float(bits.max())
            K_star = int(round(B_star * total_params))
        else:
            cache_key = (tuple(n_params.tolist()), tuple(int(b) for b in candidate_bits), round(target, 8))
            if cache_key not in _FEASIBLE_PAYLOAD_CACHE:
                # Stage 1: find max feasible payload independent of costs
                res_stage1 = milp(
                    c=-budget_row.reshape(-1),
                    constraints=[
                        LinearConstraint(A_eq, lb=1.0, ub=1.0),
                        LinearConstraint(budget_row, lb=-np.inf, ub=target + 1e-12),
                    ],
                    integrality=np.ones(P * B),
                    bounds=Bounds(0.0, 1.0),
                    options={"mip_rel_gap": 0.0},
                )
                if res_stage1.x is None or not res_stage1.success:
                    return {
                        "bit_assignment": {},
                        "achieved_avg_bits": float("nan"),
                        "total_predicted_cost": float("nan"),
                        "bit_counts": {int(b): 0 for b in candidate_bits},
                        "success": False,
                        "payload_lower_bound": float("nan"),
                    }
                choice1 = res_stage1.x.reshape(P, B).argmax(axis=1)
                assigned1 = [int(candidate_bits[j]) for j in choice1]
                K_star_val = int(round(sum(b * n for b, n in zip(assigned1, n_params))))
                B_star_val = float(K_star_val / total_params)
                _FEASIBLE_PAYLOAD_CACHE[cache_key] = (B_star_val, K_star_val)

            B_star, K_star = _FEASIBLE_PAYLOAD_CACHE[cache_key]

        budget_lb = B_star
        # Stage 2: exact equality on integer bit-parameter sum
        constraints.append(LinearConstraint(param_bits_row, lb=float(K_star) - 0.5, ub=float(K_star) + 0.5))
    else:
        constraints.append(LinearConstraint(budget_row, lb=-np.inf, ub=target + 1e-12))

    res = milp(
        c=c_obj,
        constraints=constraints,
        integrality=np.ones(P * B),
        bounds=Bounds(0.0, 1.0),
        options={"mip_rel_gap": 0.0},
    )

    if res.x is None or not res.success:
        return {
            "bit_assignment": {},
            "achieved_avg_bits": float("nan"),
            "total_predicted_cost": float("nan"),
            "bit_counts": {int(b): 0 for b in candidate_bits},
            "success": False,
            "payload_lower_bound": float(budget_lb),
        }

    choice = res.x.reshape(P, B).argmax(axis=1)
    assigned = [int(candidate_bits[j]) for j in choice]
    achieved = float(sum(b * n for b, n in zip(assigned, n_params)) / total_params)
    cost = float(sum(alloc_costs[(n, b)] for n, b in zip(names, assigned)))
    return {
        "bit_assignment": dict(zip(names, assigned)),
        "achieved_avg_bits": achieved,
        "total_predicted_cost": cost,
        "bit_counts": {int(b): assigned.count(int(b)) for b in candidate_bits},
        "success": bool(res.success),
        "payload_lower_bound": float(budget_lb),
    }
