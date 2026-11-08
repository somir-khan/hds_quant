"""Isolated matched-protocol AMQ comparison; existing controls are unchanged."""
import argparse
import copy
import importlib.metadata
import json
from pathlib import Path

from .amq_baseline import run_amq_trial
from .run_generalization import run_config as shared_run, validate_spec as shared_validate


def validate_spec(spec):
    base = copy.deepcopy(spec)
    search = base.pop('amq_search', {})
    if base.get('schema') != 'amq_comparison_v1':
        raise ValueError('Expected amq_comparison_v1')
    base['schema'] = 'generalization_v1'
    shared_validate(base)
    if set(search) != {'initial_samples', 'iterations', 'candidates_per_iteration', 'population'}:
        raise ValueError('Invalid AMQ search fields')
    if any(type(v) is not int or v < 1 for v in search.values()):
        raise ValueError('AMQ search values must be positive integers')
    if search['initial_samples'] < 250 or search['population'] < search['candidates_per_iteration']:
        raise ValueError('Use at least 250 initial samples; population must cover candidate count')
    if spec['full_reference'] or spec['k_layers'] != [4] or spec['batch_size'] != 1:
        raise ValueError('AMQ comparison requires K=4, batch=1, no full reference')
    if spec['methods'] != ['HDS-Hybrid', 'Coord-Rank-Hist', 'Coord-Rank-Exact']:
        raise ValueError('Expected the three representation controls')
    if len(set(spec['seeds'])) != len(spec['seeds']):
        raise ValueError('Duplicate seeds')


def run_config(config_path, output_dir, smoke_test=False):
    spec = json.loads(Path(config_path).read_text())
    validate_spec(spec)
    # Dependency check before allocating a model or creating outputs.
    from .amq_vendor.search import Search
    packages = {p: importlib.metadata.version(p) for p in ('pymoo', 'pySOT', 'numpy', 'scipy')}
    protocol = {'method': 'AMQ-HQQ-Matched', 'upstream_commit': '618334902a5b43813607caf73025fc9595a55026',
        'packages': packages, 'search': spec['amq_search'],
        'adaptations': ['same calibration tokens and HQQ axis0/floating zeros as controls',
            'host proxy cache and disk-backed teacher logits; FP32 JSD arithmetic',
            'payload budget, selection within [target-0.005,target]; missing budgets reported',
            'no AWQ final compression; not paper numerical reproduction'],
        'timing': 'AMQ includes precompute, screening, search and checkpoint I/O; controls use existing direct timing'}
    def trial(**kwargs):
        return run_amq_trial(**kwargs, settings=spec['amq_search'], output=output_dir)
    sources = ['hds_quant/amq_baseline.py', 'hds_quant/run_amq_comparison.py',
               'scripts/summarize_amq_comparison.py', 'requirements-amq.txt',
               'hds_quant/config.py', 'hds_quant/data_utils.py', 'hds_quant/run_experiment.py',
               'scripts/launch_amq_comparison.sh', 'scripts/run_amq_comparison_a100.sh',
               'scripts/load_hf_env.sh']
    sources += [str(p.relative_to(Path(__file__).resolve().parents[1])).replace('\\', '/')
                for p in Path(__file__).with_name('amq_vendor').iterdir() if p.is_file()]
    return shared_run(config_path, output_dir, smoke_test, spec_validator=validate_spec,
        extra_trial_runner=trial, extra_source_files=sources, comparison_protocol=protocol)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--output-dir', required=True)
    args = p.parse_args()
    run_config(args.config, args.output_dir)
