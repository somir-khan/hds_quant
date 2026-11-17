"""Read completed ARR runs, preserving their case identity and evidence type."""
import argparse
import csv
import json
from pathlib import Path


def main(inputs, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    rows = []; parity = []; sources = []; seen = set()
    for directory in map(Path, inputs):
        m = json.loads((directory/'run_manifest.json').read_text())
        if m['schema'] != 'arr_stage12_run_v1' or not m['complete']:
            raise ValueError(f'Incomplete or incompatible run: {directory}')
        key = m['case']['case_id'], m['mode']
        if key in seen:
            raise ValueError(f'Duplicate case/mode: {key}')
        seen.add(key); sources.append(dict(directory=str(directory), case=m['case'], mode=m['mode']))
        if (directory/'parity.json').exists():
            p = json.loads((directory/'parity.json').read_text())
            parity.extend(dict(case_id=key[0], mode=key[1], **r) for r in p['checks'])
        filename = 'composition_residuals.csv' if m['mode'] == 'diagnostic' else 'comparisons.csv'
        if (directory/filename).exists():
            with (directory/filename).open(newline='', encoding='utf-8') as f:
                rows.extend(dict(case_id=key[0], mode=key[1], **r) for r in csv.DictReader(f))
    def write(name, data):
        if not data:
            return
        keys = list(dict.fromkeys(k for r in data for k in r))
        with (output/name).open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=keys); w.writeheader()
            w.writerows({k: json.dumps(v) if isinstance(v, (list, dict)) else v for k, v in r.items()} for r in data)
    write('results.csv', rows); write('parity.csv', parity)
    (output/'inputs.json').write_text(json.dumps(sources, indent=2), encoding='utf-8')
    print(f'{len(sources)} runs, {len(rows)} result roles. No standalone calibration time inferred.')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('job_dirs', nargs='+'); p.add_argument('--output-dir', required=True)
    a = p.parse_args(); main(a.job_dirs, a.output_dir)
