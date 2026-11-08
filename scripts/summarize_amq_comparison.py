"""Summarize only a completed isolated AMQ comparison; never merge old jobs."""
import argparse
import csv
import json
import math
from pathlib import Path


def summarize(directory):
    root = Path(directory)
    data = json.loads((root/'limited_labels.json').read_text())
    spec = json.loads((root/'experiment_config.json').read_text())
    manifest = json.loads((root/'experiment_manifest.json').read_text())
    if spec['schema'] != 'amq_comparison_v1' or not data.get('complete') or not manifest.get('complete'):
        raise ValueError('Expected a completed AMQ comparison')
    expected = {(s, m) for s in spec['seeds'] for m in spec['methods']+['AMQ-HQQ-Matched']}
    keys = [(t['seed'], t['method']) for t in data['trials']]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Missing/duplicate trials')
    rows = []
    for trial in data['trials']:
        for budget in spec['budgets']:
            a = trial['allocations'].get(f'{budget:.2f}')
            row = dict(seed=trial['seed'], method=trial['method'], target_bits=budget,
                       status='ok' if a else 'no_candidate_in_budget_window',
                       actual_bits='', ppl='', nll='', calibration_sec=trial['calibration_wall_sec'])
            if a:
                if not a['success'] or not budget-0.005-1e-10 <= a['achieved_avg_bits'] <= budget+1e-10:
                    raise ValueError('Invalid achieved payload')
                if not set(a['bit_assignment'].values()) <= {2, 3, 4}:
                    raise ValueError('Invalid precision')
                ev = a['evaluation']
                if not math.isfinite(ev['ppl']) or len(ev['seq_nlls']) != data['evaluation_chunks']:
                    raise ValueError('Invalid held-out evaluation')
                row.update(actual_bits=a['achieved_avg_bits'], ppl=ev['ppl'], nll=ev['nll'],
                           calibration_sec=a['calibration_to_budget_sec'])
            elif budget not in trial.get('missing_budgets', []):
                raise ValueError('Unexpected missing allocation')
            rows.append(row)
    with (root/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['AMQ-HQQ-Matched comparison (not final-AWQ paper reproduction)',
             'Pilot settings are not full-search performance claims.',
             'AMQ time includes precompute/screening/search/I/O; no inference speed claim.',
             'Missing budget candidates are failures to reach the selection window, not substituted results.']
    lines += [str(row) for row in rows]
    (root/'SUMMARY.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return rows


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    summarize(p.parse_args().input_dir)
