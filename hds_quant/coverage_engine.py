"""Shared-label quality sweep and independently measured standalone calibration."""
import gc
import hashlib
import time

import torch

from .allocation_controls import allocation_costs, simple_assignment
from .allocation_control_plan import add_candidate
from .allocator import _FEASIBLE_PAYLOAD_CACHE
from .coverage_protocol import (ARMS, FEATURE_ARMS, acquisition_plan, append_allocations,
    mask_labels, pooled_predictions, protected_allocations)
from .hds_encoder import HDSEncoder
from .label_collector import (collect_features_single_pass, collect_marginal_loss_labels,
    discover_target_projections)
from .readout_cv import evaluate_fixed_k_layer_acquisition
from .run_amq_followup import save
from .run_experiment import _pair_dict_to_json
from .run_limited_labels import memory_finish, memory_start, sync


def extract_features(model, calib, cfg, seed, method):
    encoder = HDSEncoder(hd_dim=cfg.hd_dim, num_levels=cfg.num_levels,
        max_coords=cfg.max_coords, num_buckets=cfg.num_buckets, seed=seed,
        device=next(model.parameters()).device,
        build_codebooks=method in ('HDS-Hybrid', 'Coverage-Primary'))
    features = collect_features_single_pass(model, calib, cfg, encoder,
        batch_size=1, feature_estimator=method)['features']
    del encoder
    return features


def check_labels(labels, meta, layers, cfg):
    masked = mask_labels(labels, meta, layers, cfg.marginal_bits)
    if labels.keys() != masked.keys():
        raise ValueError('Collector returned extra or missing labels')


def label_replay(labels, expected):
    common = labels.keys() & expected.keys()
    gap = max((abs(labels[k]-expected[k]) for k in common), default=0.)
    if gap > 1e-6:
        raise ValueError(f'Source label replay failed: {gap}')
    return dict(overlapping_label_count=len(common), max_abs_difference=gap, passed=True)


def build_quality_plan(model, calib, cfg, spec, seed, output, source_labels):
    meta = discover_target_projections(model); device = next(model.parameters()).device
    acquisitions = acquisition_plan(spec, seed, cfg.num_layers)
    union = acquisitions['union_layers']
    labels = collect_marginal_loss_labels(model, calib, cfg, batch_size=1, target_layers=set(union))
    check_labels(labels['delta_loss'], meta, union, cfg)
    replay = label_replay(labels['delta_loss'], source_labels)
    save(output/'label_bank.json', dict(schema='coverage_label_bank_v1',
        union_layers=union, union_probe_label_count=len(labels['delta_loss']),
        measured_labels=_pair_dict_to_json(labels['delta_loss']), source_replay=replay,
        acquisition_time_sec=labels['label_time_sec'], timing_scope='Physical shared quality-bank acquisition, not standalone method calibration.'))
    sync(device); started = time.perf_counter()
    features = extract_features(model, calib, cfg, seed, 'Coverage-Primary')
    sync(device); feature_sec = time.perf_counter()-started
    torch.save(features, output/'feature_bank.pt')
    plan = dict(schema='coverage_quality_plan_v1', seed=seed, dataset=spec['dataset'],
        acquired_layers=union, probe_label_count=len(labels['delta_loss']),
        acquisition_plan=acquisitions, candidates=[],
        timing_scope='Quality sweep shares a union label bank and feature pass. Per-cell masks enforce 21*K labels. Not standalone calibration timing.')
    estimates = dict(schema='coverage_estimates_v1', shared_feature_extraction_sec=feature_sec,
        source_label_replay=replay, cells=[])
    cache = {}; fit_cache = {}
    for cell in acquisitions['cells']:
        measured = mask_labels(labels['delta_loss'], meta, cell['acquired_layers'], cfg.marginal_bits)
        fit_key = tuple(cell['acquired_layers'])
        if fit_key not in fit_cache:
            fit = evaluate_fixed_k_layer_acquisition(features, measured, meta, cfg,
                k_layers=cell['k_layers'], seed=seed, estimator_names=FEATURE_ARMS)
            if fit['acquired_layers'] != cell['acquired_layers']:
                raise ValueError('Readout changed the declared acquired blocks')
            raw = {m: fit['raw_predictions'][m] for m in FEATURE_ARMS}
            raw.update({m: pooled_predictions(measured, meta, cfg.marginal_bits, m) for m in ARMS[:2]})
            fit_cache[fit_key] = raw
        raw = fit_cache[fit_key]
        for method in ARMS:
            if any(raw[method][k] != v for k, v in measured.items()):
                raise ValueError('A coverage predictor changed acquired labels')
            append_allocations(plan, method, raw[method], meta, cfg, spec, cell, cache)
        estimates['cells'].append(dict(cell, measured_labels=_pair_dict_to_json(measured),
            raw_predictions={m: _pair_dict_to_json(raw[m]) for m in ARMS}))
        save(output/'estimates.json', estimates)
    for budget in spec['budgets']:
        for pattern in ['last_half', 'alternating_odd']:
            bits = simple_assignment(meta, budget, pattern, True)
            add_candidate(plan, bits, meta, dict(method=f'Simple-{pattern}', budget=budget,
                setting='last_down4', tie_order='declared', policy='reference', k_layers=0,
                acquired_layers=[], probe_label_count=0, predicted_cost=None,
                protection_changed_projections=None, source='no_labels_no_features'))
    del features, fit_cache, labels
    gc.collect()
    return plan


def time_one_arm(model, calib, cfg, spec, seed, cell, method):
    """No shared label/features/solver cache; inference timing and I/O excluded."""
    device = next(model.parameters()).device; meta = discover_target_projections(model)
    baseline = memory_start(device); _FEASIBLE_PAYLOAD_CACHE.clear(); sync(device)
    start = time.perf_counter()
    labels = collect_marginal_loss_labels(model, calib, cfg, batch_size=1,
        target_layers=set(cell['acquired_layers']))
    check_labels(labels['delta_loss'], meta, cell['acquired_layers'], cfg)
    sync(device); labels_done = time.perf_counter()
    features = None
    if method in FEATURE_ARMS:
        features = extract_features(model, calib, cfg, seed, method)
    sync(device); features_done = time.perf_counter()
    if method in FEATURE_ARMS:
        fit = evaluate_fixed_k_layer_acquisition(features, labels['delta_loss'], meta, cfg,
            k_layers=cell['k_layers'], seed=seed, estimator_names=(method,))
        raw = fit['raw_predictions'][method]
        if fit['acquired_layers'] != cell['acquired_layers']:
            raise ValueError('Timing fit changed acquired blocks')
    else:
        raw = pooled_predictions(labels['delta_loss'], meta, cfg.marginal_bits, method)
    costs = allocation_costs(raw, meta, cfg)
    sync(device); fit_done = time.perf_counter()
    allocations = {}
    names = {m['name'] for m in meta if m['layer_idx'] in cell['acquired_layers']}
    for budget in spec['budgets']:
        # Each per-budget measurement includes its own payload-feasibility solve.
        _FEASIBLE_PAYLOAD_CACHE.clear(); sync(device); before = time.perf_counter()
        selected = protected_allocations(costs, meta, budget, names, method in ARMS[:2],
            orders=('depth_ascending',) if method in ARMS[:2] else ('native',))
        sync(device); elapsed = time.perf_counter()-before
        allocation = next(iter(selected.values()))
        allocations[f'{budget:.2f}'] = dict(allocation, allocation_sec=elapsed,
            calibration_to_budget_sec=fit_done-start+elapsed)
    sync(device); wall = time.perf_counter()-start; memory = memory_finish(device, baseline)
    result = dict(method=method, seed=seed, **cell,
        label_acquisition_sec=labels_done-start, feature_extraction_sec=features_done-labels_done,
        fit_and_cost_sec=fit_done-features_done, calibration_shared_sec=fit_done-start,
        calibration_wall_sec=wall, allocations=allocations,
        measured_labels=_pair_dict_to_json(labels['delta_loss']),
        raw_predictions=_pair_dict_to_json(raw), **memory)
    del features, labels
    gc.collect()
    return result


def run_timing(model, calib, cfg, spec, seed, cell, output, source_labels):
    order = sorted(ARMS, key=lambda m: hashlib.sha256(f'{seed}|{cell["k_layers"]}|{cell["policy"]}|{m}'.encode()).hexdigest())
    report = dict(schema='coverage_timing_result_v1', complete=False, dataset=spec['dataset'],
        seed=seed, cell=cell, method_order=order, trials=[],
        timing_scope='Independent fresh acquisition, features, fitting and primary protected allocation per arm. Excludes model/data loading, warmup, disk output and held-out PPL. Ten placement sensitivities are not included in primary calibration timing.')
    save(output/'timing.json', report)
    for method in order:
        print(f'Timing {method}: K={cell["k_layers"]}, {cell["policy"]}, fresh acquisition', flush=True)
        trial = time_one_arm(model, calib, cfg, spec, seed, cell, method)
        measured = {(n.rsplit('|', 1)[0], int(n.rsplit('|', 1)[1])): v for n, v in trial['measured_labels'].items()}
        trial['source_label_replay'] = label_replay(measured, source_labels)
        report['trials'].append(trial); save(output/'timing.json', report)
    report['complete'] = True; save(output/'timing.json', report)
    return report
