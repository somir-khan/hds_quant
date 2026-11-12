"""Frozen K=4 replication: immutable source banks, exact payloads and placement controls."""
import json
import math
from pathlib import Path

from .allocation_control_plan import METADATA_METHODS, add_candidate
from .allocation_controls import (SETTINGS, allocation_costs, family_mean_predictions,
    family_tie_order, last_down, simple_assignment, solve_control)
from .amq_followup_plan import METHODS, digest, geometry, payload
from .config import ExperimentConfig
from .readout_cv import select_stratified_k_layers
from .run_no1bit_diagnostic import decode_pairs, reconstruct_costs

RANDOM_SEEDS = (17, 29, 43, 59, 71, 89, 101, 127, 149, 173)
ORDERS = ('depth_ascending',) + tuple(f'random_{s}' for s in RANDOM_SEEDS)
MODEL = 'meta-llama/Llama-3.1-8B-Instruct'
MODEL_REVISION = '0e9e39f249a16976918f6564b8830bc894c89659'
DATASETS = {
    'wikitext2': dict(name='wikitext2', revision='b08601e04326c79dfdd32d625aee71d232d685c3', eval_chunks=141),
    'c4': dict(name='c4', revision='1588ec454efa1a09f29cd18ddd04fe05fc8653a2',
               calibration_docs=20000, evaluation_docs=1100, eval_chunks=141),
}


def validate_spec(spec, seed):
    expected = dict(schema='frozen_replication_v1', seeds=[42, 100, 2026], k_layers=4,
        budgets=[2.5, 3.0, 3.5], settings=list(SETTINGS), family_tie_orders=list(ORDERS),
        saved_methods=METHODS, metadata_methods=list(METADATA_METHODS),
        simple_patterns=['last_half', 'alternating_odd'], materiality_percent=1.0,
        protection_rule='preserve_feasible_unconstrained_assignment',
        primary_family_order='depth_ascending')
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f'Frozen replication policy differs: {key}')
    if type(seed) is not int or seed not in expected['seeds']:
        raise ValueError('Seed must be 42, 100 or 2026')
    if spec.get('dataset') not in DATASETS or not spec.get('source_dir') or not spec.get('experiment_id'):
        raise ValueError('Missing/invalid dataset, source or experiment ID')


def load_replication_source(source, dataset, seed):
    """Only completed representation-controls jobs; no AMQ archive or full label bank."""
    source = Path(source)
    names = ['experiment_config.json', 'resolved_experiment_config.json',
             'experiment_manifest.json', 'limited_labels.json', 'data_manifest.json']
    hashes = {n: digest(source/n) for n in names}
    original, src, manifest, raw, data = [json.loads((source/n).read_text()) for n in names]
    if not raw.get('complete') or not manifest.get('complete') or manifest.get('smoke_test'):
        raise ValueError('Source must be a completed real representation-controls job')
    if hashes['experiment_config.json'] != manifest['config_sha256']:
        raise ValueError('Source configuration hash differs')
    if src.get('schema') != 'representation_controls_v1' or src.get('purpose') != 'benchmark':
        raise ValueError('Expected representation_controls_v1 benchmark source')
    if src.get('model_id') != MODEL or src.get('model_revision') != MODEL_REVISION:
        raise ValueError('Source model/revision differs')
    if src.get('dataset') != DATASETS[dataset] or src['dataset']['revision'] != manifest['resolved_dataset_revision']:
        raise ValueError('Source dataset/revision differs')
    if src['model_revision'] != manifest['resolved_model_revision']:
        raise ValueError('Source resolved model revision differs')
    if (src.get('seeds') != [42, 100, 2026] or src.get('k_layers') != [4] or
        src.get('methods') != METHODS or src.get('probe_bits') != [1, 2, 3, 4] or
        src.get('allocation_bits') != [2, 3, 4] or src.get('budgets') != [3.0, 3.5] or
        src.get('full_reference') is not False):
        raise ValueError('Source acquisition/precision protocol differs')
    expected_data = dict(calibration=dict(n_sequences=64, seq_len=1024, max_encoded_tokens=4096),
                         eval_seq_len=2048, batch_size=1)
    if any(src.get(k) != v for k, v in expected_data.items()):
        raise ValueError('Source calibration/evaluation sizes differ')
    if (original['model_id'] != src['model_id'] or original['dataset'] != src['dataset'] or
        original['model_revision'] != src['model_revision']):
        raise ValueError('Original/resolved source pin mismatch')
    if (raw['evaluation_token_sha256'] != data['evaluation_token_sha256'] or
        raw['evaluation_token_sha256'] != manifest['evaluation_token_sha256'] or
        raw['evaluation_chunks'] != 141 or data['scored_tokens'] != 288627):
        raise ValueError('Source evaluation token metadata differs')
    # Multi-seed source JSON retains the final seed's config. Explicitly restore this case's seed.
    config = dict(raw['config'], seeds=[seed])
    expected_cfg = dict(model_name=MODEL, num_layers=32, quantizer='hqq', hqq_axis=0,
        hqq_zero_policy='floating', quant_group_size=128, candidate_bits=[1, 2, 3, 4],
        reference_bit=4, calib_num_sequences=64, calib_seq_len=1024, max_encoded_tokens=4096,
        hd_dim=4096, num_levels=32, num_buckets=512, residual_anchor='int4_aligned',
        coord_binding='salience_bucket', readout_grouping='per_bit', target_transform='signed_log1p')
    if any(config.get(k) != v for k, v in expected_cfg.items()):
        raise ValueError('Source quantizer/feature/readout settings differ')
    cfg = ExperimentConfig(**config); meta = geometry(); total = sum(m['num_params'] for m in meta)
    trials = {}
    for method in METHODS:
        found = [t for t in raw['trials'] if t['method'] == method and t['seed'] == seed and t['k_layers'] == 4]
        if len(found) != 1:
            raise ValueError('Missing/duplicate seed-method source trial')
        trial = found[0]; costs = reconstruct_costs(trial, meta, cfg)
        expected_layers = select_stratified_k_layers(list(range(32)), 4, seed)
        if trial['acquired_layers'] != expected_layers or trial['probe_label_count'] != 84:
            raise ValueError('Source acquired blocks/label count differ')
        for budget in [3., 3.5]:
            a = trial['allocations'][f'{budget:.2f}']; bits = a['bit_assignment']
            objective = math.fsum(costs[n, b] for n, b in bits.items())
            if (not a['success'] or payload(bits, meta) != round(budget*total) or
                not math.isclose(a['achieved_avg_bits'], budget, abs_tol=1e-12) or
                not math.isclose(objective, a['total_predicted_cost'], abs_tol=1e-9, rel_tol=1e-9)):
                raise ValueError('Source assignment payload/objective differs')
        trials[method] = trial
    measured = decode_pairs(trials['HDS-Hybrid']['measured_labels'])
    for trial in trials.values():
        other = decode_pairs(trial['measured_labels'])
        if other.keys() != measured.keys() or any(abs(other[k]-measured[k]) > 1e-9 for k in measured):
            raise ValueError('Source methods have different measured labels')
    if str(seed) not in raw['calibration_token_sha256']:
        raise ValueError('Missing seed calibration token hash')
    return src, raw, hashes, cfg, trials, measured


def add_replication_estimator(plan, method, costs, meta, spec, measured_names=(), trial=None):
    """Same no-op protection rule for every method and every placement order."""
    down = last_down(meta)
    for budget in spec['budgets']:
        saved = trial['allocations'].get(f'{budget:.2f}') if trial else None
        original = dict(saved['bit_assignment']) if saved else solve_control(costs, meta, budget)
        orders = ORDERS if method == 'Family-Mean' else ('native',)
        constrained_base = None
        for order in orders:
            def place(bits, protected=False):
                if method != 'Family-Mean':
                    return dict(bits)
                return family_tie_order(bits, costs, meta, measured_names, order,
                    [down] if protected else (), allowed_orders=ORDERS)
            unconstrained = place(original)
            if unconstrained[down] == 4:
                protected = dict(unconstrained)
            else:
                if constrained_base is None:
                    constrained_base = solve_control(costs, meta, budget, True)
                protected = place(constrained_base, True)
            before_cost = math.fsum(costs[n, b] for n, b in unconstrained.items())
            after_cost = math.fsum(costs[n, b] for n, b in protected.items())
            if after_cost < before_cost-1e-8:
                raise ValueError('Protected solution unexpectedly improves the unconstrained optimum')
            for setting, chosen in [('unconstrained', unconstrained), ('last_down4', protected)]:
                changed = sum(chosen[n] != unconstrained[n] for n in chosen)
                historical = saved['evaluation'] if saved and chosen == saved['bit_assignment'] else None
                add_candidate(plan, chosen, meta, dict(method=method, budget=budget, setting=setting,
                    tie_order=order, predicted_cost=math.fsum(costs[n, b] for n, b in chosen.items()),
                    source='saved_labels_and_predictions' if trial else 'saved_labels_new_control',
                    protection_changed_projections=changed,
                    protection_was_noop=setting == 'last_down4' and changed == 0), historical)


def build_replication_plan(spec, source, seed):
    validate_spec(spec, seed)
    src, raw, hashes, cfg, trials, measured = load_replication_source(source, spec['dataset'], seed)
    meta = geometry()
    plan = dict(schema='frozen_replication_plan_v1', protocol=spec, seed=seed,
        dataset=spec['dataset'], source_spec=src, source_hashes=hashes,
        experiment_config=cfg.to_dict(), acquired_layers=trials['HDS-Hybrid']['acquired_layers'],
        probe_label_count=len(measured), protection_projection=last_down(meta),
        candidates=[], metadata_pending=True,
        timing_scope='Saved K=4 labels/predictions reused. Fresh metadata extraction/fit only; no end-to-end timing or speedup.',
        tie_scope='Preserve ILP family precision counts; only permute equal-cost/size unmeasured projections within family. One depth primary plus ten fixed SHA256 orders.',
        constraint_scope='Previously observed final-block down4 prior. Preserve each already-feasible unconstrained assignment exactly.')
    for method in METHODS:
        add_replication_estimator(plan, method, reconstruct_costs(trials[method], meta, cfg),
                                  meta, spec, trial=trials[method])
    family_raw = family_mean_predictions(measured, meta, cfg.marginal_bits)
    add_replication_estimator(plan, 'Family-Mean', allocation_costs(family_raw, meta, cfg),
                              meta, spec, measured_names={n for n, b in measured})
    for budget in spec['budgets']:
        for pattern in spec['simple_patterns']:
            before = simple_assignment(meta, budget, pattern)
            for setting in SETTINGS:
                chosen = before if setting == 'unconstrained' or before[last_down(meta)] == 4 else simple_assignment(meta, budget, pattern, True)
                changed = sum(chosen[n] != before[n] for n in chosen)
                add_candidate(plan, chosen, meta, dict(method=f'Simple-{pattern}', budget=budget,
                    setting=setting, tie_order='declared', predicted_cost=None,
                    source='no_labels_no_features', protection_changed_projections=changed,
                    protection_was_noop=setting == 'last_down4' and changed == 0))
    return plan, raw, measured, family_raw
