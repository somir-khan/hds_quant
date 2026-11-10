"""Calibration-only progress reporting; never evaluates or selects using test loss."""
import csv
import json

from .amq_coverage import select_under_cap


def progress_snapshot(archive, budgets, group_size, history, *, stage, iteration,
                      new_from, elapsed_sec):
    previous = history[-1]['budgets'] if history else {}
    new_bits = [x[2] - 32 / group_size for x in archive[new_from:]]
    rows = {}
    for budget in budgets:
        key = f'{budget:.2f}'
        best = select_under_cap(archive, budget, group_size)
        prior = previous.get(key, {}).get('best_calibration_jsd')
        index = next((i for i, x in enumerate(archive) if x is best), None)
        origin = next((h['stage'] for h in history
                       if index is not None and index < h['configuration_evaluations']), stage)
        bits = best[2] - 32 / group_size if best is not None else None
        rows[key] = dict(
            feasible_candidates=sum(2 <= x[2]-32/group_size <= budget+1e-10 for x in archive),
            best_archive_index=index, best_calibration_jsd=best[1] if best is not None else None,
            actual_payload_bits=bits, payload_gap=budget-bits if bits is not None else None,
            winner_origin=origin if best is not None else None,
            improved_vs_previous=best is not None and (prior is None or best[1] < prior))
    return dict(stage=stage, iteration=iteration, configuration_evaluations=len(archive),
                new_configurations=len(new_bits), elapsed_sec=elapsed_sec,
                new_candidate_payload_min=min(new_bits) if new_bits else None,
                new_candidate_payload_max=max(new_bits) if new_bits else None, budgets=rows)


def write_progress(directory, history):
    """Replace small progress views atomically, retaining all finished stages."""
    payload = dict(schema='amq_search_progress_v1',
                   scope='Best measured calibration JSD under each payload cap; no held-out selection',
                   history=history)
    temp = directory / 'search_progress.json.tmp'
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    temp.replace(directory / 'search_progress.json')
    rows = []
    for snapshot in history:
        base = {k: v for k, v in snapshot.items() if k != 'budgets'}
        rows.extend(dict(base, budget=float(budget), **values)
                    for budget, values in snapshot['budgets'].items())
    if rows:
        temp = directory / 'search_progress.csv.tmp'
        with temp.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(directory / 'search_progress.csv')
