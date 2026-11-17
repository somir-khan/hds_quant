"""Frozen ARR controls: source guards, reduced-probe parity and allocation plans.

This module never loads an HF model, reads test scores to select a method, or
changes an older experiment. Features are deserialized on CPU from trusted banks.
"""
import copy
import json
import math
from pathlib import Path

import numpy as np

from .allocation_control_plan import add_candidate
from .allocation_controls import allocation_costs, simple_assignment
from .amq_followup_plan import digest, geometry
from .config import ExperimentConfig
from .coverage_protocol import FEATURE_ARMS, mask_labels, protected_allocations
from .focused_1b_protocol import MODEL as MODEL1, REVISION as REV1
from .frozen_replication_plan import DATASETS, MODEL as MODEL8, MODEL_REVISION as REV8
from .readout_cv import (_base_feature, _fit_predict_grouped,
                        evaluate_fixed_k_layer_acquisition)

CELLS = [(2, 'boundary_inclusive'), (8, 'stratified')]


def validate_spec(spec):
    expected = dict(schema='arr_stage12_v1', budgets=[3., 3.5], primary_budget=3.5,
        probe_bits=[2, 3], final_bits=[2, 3, 4], protection='last_down4',
        cells=[dict(k_layers=k, policy=p) for k, p in CELLS], predictors=list(FEATURE_ARMS),
        parity_tolerance=1e-8, replay_tolerance=1e-6,
        preserve_legacy_indicator_geometry=True,
        depth_features=['depth', 'depth_squared', 'is_first', 'is_last'],
        direct_residual='exp_saved_log_mse_minus_1e-8', e_case_index=6,
        e_joint_block=0, e_joint_bits=2, e_joint_context='all_other_projections_4bit',
        e_source_replay_blocks=[0, 15])
    for k, v in expected.items():
        if spec.get(k) != v:
            raise ValueError(f'Frozen ARR declaration differs: {k}')
    if set(spec.get('source_roots', {})) != {'8b_wikitext', '8b_c4', '1b'}:
        raise ValueError('Missing source roots')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def encode(values):
    return {f'{n}|{b}': float(v) for (n, b), v in sorted(values.items())}


def decode(values):
    return {(k.rsplit('|', 1)[0], int(k.rsplit('|', 1)[1])): float(v) for k, v in values.items()}


def case_definitions(spec):
    cases = []
    for model in ('8b', '1b'):
        for dataset in ('wikitext2', 'c4'):
            for i, seed in enumerate((42, 100, 2026)):
                if model == '8b':
                    key = '8b_wikitext' if dataset == 'wikitext2' else '8b_c4'
                    job = 1177435 if dataset == 'wikitext2' else 1177436
                    rel = f'{spec["source_roots"][key]}/job_{job}_{i}_seed{seed}_v1'
                else:
                    rel = f'{spec["source_roots"]["1b"]}/{dataset}/seed_{seed}/quality'
                cases.append(dict(index=len(cases), model=model, dataset=dataset, seed=seed,
                    case_id=f'{model}_{dataset}_seed{seed}', relative_quality=rel))
    return cases


def load_source(spec, base, index, *, features=True):
    if type(index) is not int or not 0 <= index < 12:
        raise ValueError('Case index must be 0..11')
    case = case_definitions(spec)[index]; q = (Path(base)/case['relative_quality']).resolve()
    one = case['model'] == '1b'
    manifest_name = 'case_manifest.json' if one else 'study_manifest.json'
    man = read(q/manifest_name); report = read(q/'diagnostic.json')
    if not man.get('complete') or not report.get('complete') or man.get('plan_only'):
        raise ValueError('Expected a completed quality source')
    for key in ('dataset', 'seed'):
        if man[key] != case[key] or report[key] != case[key]:
            raise ValueError('Source case differs')
    paths = [manifest_name, 'diagnostic.json', 'selection_plan.json', 'label_bank.json',
             'estimates.json', 'feature_bank.pt']
    for f in ('selection_plan', 'label_bank', 'feature_bank'):
        suffix = '.pt' if f == 'feature_bank' else '.json'
        if digest(q/(f+suffix)) != man[f+'_sha256']:
            raise ValueError(f'Source hash mismatch: {f}')
    selection = read(q/'selection_plan.json')
    planned = {c['candidate_id']: (c['bit_assignment'], c['roles']) for c in selection['candidates']}
    evaluated = {c['candidate_id']: (c['bit_assignment'], c['roles']) for c in report['candidates']}
    if planned != evaluated:
        raise ValueError('Source frozen plan and evaluated candidates differ')
    if one:
        root = Path(base)/spec['source_roots']['1b']; rm = read(root/'run_manifest.json')
        if (not rm.get('complete') or rm['model_id'] != MODEL1 or rm['model_revision'] != REV1
            or rm['batch_size'] != 1 or man['parent_config_sha256'] != rm['config_sha256']):
            raise ValueError('Invalid focused 1B parent')
        if digest(root/'experiment_config.json') != rm['config_sha256']:
            raise ValueError('1B parent configuration changed')
        data_path = q.parents[1]/'data_manifest.json'
        if digest(data_path) != rm['data_manifest_sha256'][case['dataset']]:
            raise ValueError('1B data manifest changed')
        baseline_path = q.parents[1]/'fresh_uniform_baselines.json'
        if digest(baseline_path) != rm['baseline_sha256'][case['dataset']] or man['baseline_sha256'] != rm['baseline_sha256'][case['dataset']]:
            raise ValueError('1B baseline provenance changed')
        gp = root/'projection_geometry.json'
        if digest(gp) != rm['projection_geometry_sha256']:
            raise ValueError('1B geometry changed')
        meta = read(gp); config_path = q.parent/'resolved_model_config.json'
        model_id, revision = MODEL1, REV1
        extra = [root/'run_manifest.json', root/'experiment_config.json', gp,
                 config_path, data_path, baseline_path]
    else:
        if digest(q/'experiment_config.json') != man['config_sha256']:
            raise ValueError('8B source configuration changed')
        config_path = q/'resolved_model_config.json'; data_path = q/'data_manifest.json'
        meta = geometry(); model_id, revision = MODEL8, REV8
        extra = [config_path, data_path, q/'experiment_config.json']
    cfg = ExperimentConfig(**read(config_path)); cfg.seeds = [case['seed']]
    expected = dict(model_name=model_id, num_layers=16 if one else 32, quantizer='hqq',
        hqq_axis=0, hqq_zero_policy='floating', quant_group_size=128, reference_bit=4,
        residual_anchor='int4_aligned', coord_binding='salience_bucket',
        readout_grouping='per_bit', target_transform='signed_log1p', max_encoded_tokens=4096,
        calib_num_sequences=64, calib_seq_len=1024)
    if any(getattr(cfg, k) != v for k, v in expected.items()) or tuple(cfg.candidate_bits) != (1, 2, 3, 4):
        raise ValueError('Source model or quantizer/readout protocol differs')
    data = read(data_path)
    if (data['resolved_revision'] != DATASETS[case['dataset']]['revision'] or
        data['evaluation_chunks'] != 141 or data['scored_tokens'] != 288627):
        raise ValueError('Source data protocol differs')
    labels = decode(read(q/'label_bank.json')['measured_labels'])
    estimates = read(q/'estimates.json')
    hashes = {str(p): digest(p) for p in [q/n for n in paths]+extra}
    bank = None
    if features:
        import torch
        bank = torch.load(q/'feature_bank.pt', map_location='cpu', weights_only=True)
        wanted = {(m['name'], b) for m in meta for b in (1, 2, 3)}
        if bank.keys() != wanted:
            raise ValueError('Feature bank projection/precision coverage differs')
    return dict(case=case, directory=q, cfg=cfg, meta=meta, labels=labels,
        estimates=estimates, report=report, features=bank, hashes=hashes,
        data_manifest=data, model_id=model_id, revision=revision,
        calibration_token_sha256=man.get('calibration_token_sha256'),
        hqq_config=rm['effective_hqq_config'] if one else man['effective_hqq_config'])


def verify_sources(source):
    if any(digest(p) != h for p, h in source['hashes'].items()):
        raise RuntimeError('An immutable source file changed')


def reduced_config(cfg):
    new = copy.deepcopy(cfg); new.candidate_bits = (2, 3, 4)
    return new


def fit_reduced(features, measured, meta, cfg, acquired, method, *, depth=None,
                preserve_geometry=True):
    """Predict ALL rows, including acquired ones; retention is a separate control.

    Legacy designs appended [bit==1, bit==2]. The first column is zero in each
    2/3-bit model but counts in sqrt(D/d_meta). Retain it without a 1-bit probe.
    Depth controls use ordinary standardized ridge with no wide-block weights.
    """
    meta = sorted(meta, key=lambda m: (m['layer_idx'], m['family_idx']))
    keys = [(m['name'], b) for m in meta for b in (2, 3)]
    layers = np.array([m['layer_idx'] for m in meta for _ in (2, 3)])
    X = []
    for m in meta:
        z = m['layer_idx']/max(1, cfg.num_layers-1)
        for b in (2, 3):
            if depth == 'family_only':
                base = np.eye(7)[m['family_idx']]
            else:
                base = _base_feature(features[m['name'], b], method)
            if depth:
                base = np.r_[base, z, z*z, float(z == 0), float(z == 1)]
            ind = [0., float(b == 2)] if preserve_geometry else [float(b == 2)]
            X.append(np.r_[base, ind])
    X = np.asarray(X, dtype=np.float64)
    mask = np.isin(layers, acquired)
    y = np.array([measured.get(k, np.nan) for k in keys])
    if not np.isfinite(y[mask]).all() or len(measured) != 14*len(acquired):
        raise ValueError('Reduced fit label mask differs')
    pred, _ = _fit_predict_grouped(X[mask], y[mask], [k[1] for k, yes in zip(keys, mask) if yes],
        X, [k[1] for k in keys], estimator=None if depth else method, hd_dim=None if depth else cfg.hd_dim,
        groups_tr=layers[mask], transform_mode=cfg.target_transform,
        alphas=np.asarray(cfg.ridge_alphas), inner_cv_folds=cfg.cv_folds,
        n_indicator_cols=2 if preserve_geometry else 1)
    pred = np.nan_to_num(pred, nan=0., posinf=1e30, neginf=-1e30)
    predicted = {k: float(v) for k, v in zip(keys, pred)}
    return predicted, predicted | measured


def direct_residual(features, meta):
    # The saved FP32 log-MSE is used, not an invented exact reconstruction.
    return {(m['name'], b): max(0., math.exp(float(features[m['name'], b]['Scalar-Meta-Ridge'][0]))-1e-8)
            for m in meta for b in (2, 3)}


def add_raw(plan, source, cfg, raw, method, cell, variant):
    costs = allocation_costs(raw, source['meta'], cfg)
    for budget in (3., 3.5):
        bits = protected_allocations(costs, source['meta'], budget)['native']['bit_assignment']
        old = next((c['evaluation'] for c in source['report']['candidates'] if c['bit_assignment'] == bits), None)
        add_candidate(plan, bits, source['meta'], dict(method=method, budget=budget,
            setting='last_down4', tie_order='native', k_layers=cell['k_layers'], policy=cell['policy'],
            acquired_layers=cell['acquired_layers'], probe_label_count=14*cell['k_layers'],
            variant=variant, source='saved_features_reduced_probes',
            predicted_cost=math.fsum(costs[n, b] for n, b in bits.items())), old)
    return costs


def build_controls(source, spec):
    cfg = reduced_config(source['cfg']); bank = source['features']; meta = source['meta']
    plan = dict(schema='arr_controls_plan_v1', seed=source['case']['seed'], candidates=[],
        acquired_layers=[], probe_label_count=None, timing_scope='Saved-bank fitting and allocation; not standalone calibration timing.')
    parity = []; estimates = []
    for k, policy in CELLS:
        cell = next(c for c in source['estimates']['cells'] if (c['k_layers'], c['policy']) == (k, policy))
        acquired = cell['acquired_layers']; measured = mask_labels(source['labels'], meta, acquired, (2, 3))
        legacy_measured = mask_labels(source['labels'], meta, acquired, (1, 2, 3))
        legacy = evaluate_fixed_k_layer_acquisition(bank, legacy_measured, meta, source['cfg'],
            k_layers=k, seed=source['case']['seed'], estimator_names=FEATURE_ARMS)['raw_predictions']
        for method in FEATURE_ARMS:
            predicted, retained = fit_reduced(bank, measured, meta, cfg, acquired, method)
            old = legacy[method]; saved = decode(cell['raw_predictions'][method])
            saved_gap = max(abs(old[key]-saved[key]) for key in retained)
            gap = max(abs(old[key]-retained[key]) for key in retained)
            costs = add_raw(plan, source, cfg, retained, method, cell, 'retained_measured')
            old_costs = allocation_costs(old, meta, source['cfg'])
            cost_gap = max(abs(costs[n, b]-old_costs[n, b]) for n in (m['name'] for m in meta) for b in (2, 3, 4))
            assignments = []
            for budget in spec['budgets']:
                new = protected_allocations(costs, meta, budget)['native']['bit_assignment']
                refit = protected_allocations(old_costs, meta, budget)['native']['bit_assignment']
                saved_bits = [c['bit_assignment'] for c in source['report']['candidates']
                    if any(r['method'] == method and r['budget'] == budget and r['k_layers'] == k
                        and r['policy'] == policy and r['tie_order'] == 'native' for r in c['roles'])]
                if len(saved_bits) != 1:
                    raise ValueError('Missing unique original primary role')
                assignments.append(dict(budget=budget, reduced_vs_legacy=new == refit,
                                        legacy_vs_saved=refit == saved_bits[0]))
            passed = (gap <= spec['parity_tolerance'] and cost_gap <= spec['parity_tolerance']
                and saved_gap <= spec['replay_tolerance']
                and all(a['reduced_vs_legacy'] and a['legacy_vs_saved'] for a in assignments))
            parity.append(dict(method=method, k_layers=k, policy=policy, passed=passed,
                reduced_vs_legacy_max_abs=gap, legacy_vs_saved_max_abs=saved_gap,
                final_cost_max_abs=cost_gap, assignments=assignments))
            add_raw(plan, source, cfg, predicted, method, cell, 'predict_all')
            estimates.append(dict(method=method, k_layers=k, policy=policy, acquired_layers=acquired,
                predicted_all=encode(predicted), retained=encode(retained), measured=encode(measured)))
        for name, depth in [('Family-Depth-Ridge', 'family_only'), ('Meta-Depth-Ridge', 'metadata')]:
            predicted, retained = fit_reduced(bank, measured, meta, cfg, acquired,
                'Scalar-Meta-Ridge', depth=depth)
            add_raw(plan, source, cfg, retained, name, cell, 'retained_measured')
            add_raw(plan, source, cfg, predicted, name, cell, 'predict_all')
            estimates.append(dict(method=name, k_layers=k, policy=policy, acquired_layers=acquired,
                predicted_all=encode(predicted), retained=encode(retained), measured=encode(measured)))
    # Label-free residual baseline: never overwrite a projection with a measured label.
    add_raw(plan, source, cfg, direct_residual(bank, meta), 'Residual-MSE',
        dict(k_layers=0, policy='none', acquired_layers=[]), 'no_labels')
    for budget in spec['budgets']:
        for pattern in ('last_half', 'alternating_odd'):
            bits = simple_assignment(meta, budget, pattern, True)
            old = next(c['evaluation'] for c in source['report']['candidates'] if c['bit_assignment'] == bits)
            add_candidate(plan, bits, meta, dict(method='Simple-'+pattern, budget=budget,
                setting='last_down4', tie_order='declared', k_layers=0, policy='reference',
                acquired_layers=[], probe_label_count=0, variant='no_labels_no_features'), old)
    return plan, dict(schema='arr_probe_parity_v1', passed=all(p['passed'] for p in parity), checks=parity), estimates
