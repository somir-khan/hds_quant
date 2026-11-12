"""Actual collection/fit/evaluation parity, masked labels and independent timing."""
import copy
import json
from pathlib import Path

import pytest
import torch

from hds_quant.amq_followup_plan import digest, geometry, payload
from hds_quant.config import ExperimentConfig
from hds_quant.coverage_protocol import (ARMS, FEATURE_ARMS, POLICIES, acquired_layers,
    acquisition_plan, mask_labels, pooled_predictions, validate_spec)
from hds_quant.estimators import compute_selected_estimator
from hds_quant.frozen_replication_plan import ORDERS
from hds_quant.hds_encoder import HDSEncoder

ROOT = Path(__file__).resolve().parents[1]


def spec_for(dataset='wikitext'):
    return json.loads((ROOT/f'configs/coverage_study/llama31_8b_{dataset}_v1.json').read_text())


def test_acquisition_policies_match_counts_and_replace_only_extremes():
    spec = spec_for()
    for seed in [42, 100, 2026]:
        validate_spec(spec, seed, 'quality')
        plan = acquisition_plan(spec, seed)
        assert len(plan['cells']) == 6
        for k in [2, 4, 8]:
            strat = acquired_layers(32, k, seed, 'stratified')
            bound = acquired_layers(32, k, seed, 'boundary_inclusive')
            assert len(strat) == len(bound) == k and bound[0] == 0 and bound[-1] == 31
            assert strat[1:-1] == bound[1:-1]
    assert acquired_layers(32, 4, 100, 'stratified') == acquired_layers(32, 4, 100, 'boundary_inclusive')
    for key, value in [('budgets', [2.5, 3., 3.5]), ('methods', list(ARMS[:-1])), ('protection', 'off')]:
        with pytest.raises(ValueError, match='Frozen'):
            validate_spec(dict(spec, **{key: value}), 42, 'quality')
    with pytest.raises(ValueError, match='Timing requires'):
        validate_spec(spec, 42, 'timing', 1, 'boundary_inclusive')


def test_median_is_defined_at_k2_and_robust_pooling_retains_measured_labels():
    meta = [dict(name=f'p{i}', layer_idx=i, family_idx=0) for i in range(6)]
    labels = {(f'p{i}', b): v*(4-b) for i, v in enumerate([1., 2., 3., 100.]) for b in [1, 2, 3]}
    mean = pooled_predictions(labels, meta, [1, 2, 3], 'Family-Mean')
    median = pooled_predictions(labels, meta, [1, 2, 3], 'Family-Median')
    assert mean['p5', 3] == 26.5 and median['p5', 3] == 2.5
    assert all(mean[k] == median[k] == v for k, v in labels.items())
    two = mask_labels(labels, meta, [0, 3], [1, 2, 3])
    assert len(two) == 6 and set(two) < set(labels)
    assert pooled_predictions(two, meta, [1, 2, 3], 'Family-Mean') == pooled_predictions(two, meta, [1, 2, 3], 'Family-Median')
    with pytest.raises(ValueError, match='Missing acquired'):
        mask_labels(labels, meta, [5], [1, 2, 3])


def test_joint_feature_values_match_every_standalone_arm_exactly():
    encoder = HDSEncoder(hd_dim=64, num_levels=4, max_coords=64, num_buckets=16, device='cpu')
    residual = torch.rand(8, 32, generator=torch.Generator().manual_seed(5)); norms = torch.arange(32).float()
    joint = compute_selected_estimator(residual, norms, 2, encoder, 'Coverage-Primary')
    assert set(joint) == set(FEATURE_ARMS)
    for method in FEATURE_ARMS:
        standalone = compute_selected_estimator(residual, norms, 2, encoder, method)[method]
        torch.testing.assert_close(joint[method], standalone, rtol=0, atol=0)


def test_real_quality_masks_labels_and_matches_independent_timing(tmp_path, monkeypatch):
    import hds_quant.coverage_engine as engine
    from hds_quant.run_allocation_controls import evaluate_candidates
    from hds_quant.run_experiment import _build_smoke_model
    from hds_quant.label_collector import discover_target_projections
    from hds_quant.evaluator import evaluate_perplexity
    from hds_quant.coverage_summary import summarize_quality, summarize_timing, aggregate
    from hds_quant.run_amq_followup import save
    torch.set_num_threads(1)
    model, ids = _build_smoke_model(); ids = ids[:1, :8]; calib = ids.clone()
    cfg = ExperimentConfig(num_layers=4, quantizer='hqq', hqq_axis=0, quant_group_size=64,
        candidate_bits=(1, 2, 3, 4), hd_dim=64, num_levels=4, num_buckets=16, max_coords=256,
        max_encoded_tokens=8, seeds=(42,), device='cpu', residual_anchor='int4_aligned',
        coord_binding='salience_bucket', readout_grouping='per_bit', target_transform='signed_log1p')
    spec = dict(spec_for(), k_layers=[2, 4])
    counts = dict(labels=0, features=0, fits=0)
    collector = engine.collect_marginal_loss_labels; extractor = engine.extract_features
    fitter = engine.evaluate_fixed_k_layer_acquisition
    def collect(*a, **kw):
        counts['labels'] += 1
        return collector(*a, **kw)
    def extract(*a, **kw):
        counts['features'] += 1
        return extractor(*a, **kw)
    def fit(features, measured, meta, config, **kw):
        counts['fits'] += 1
        layers = {int(k[0].split('.')[2]) for k in measured}
        assert len(layers) == kw['k_layers']
        assert set(measured) == {(m['name'], b) for m in meta if m['layer_idx'] in layers for b in config.marginal_bits}
        return fitter(features, measured, meta, config, **kw)
    monkeypatch.setattr(engine, 'collect_marginal_loss_labels', collect)
    monkeypatch.setattr(engine, 'extract_features', extract)
    monkeypatch.setattr(engine, 'evaluate_fixed_k_layer_acquisition', fit)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    quality = tmp_path/'quality'; quality.mkdir()
    plan = engine.build_quality_plan(model, calib, cfg, spec, 42, quality, {})
    assert counts['labels'] == counts['features'] == 1
    assert counts['fits'] <= 4
    meta = discover_target_projections(model)
    roles = [(c, r) for c in plan['candidates'] for r in c['roles'] if r['k_layers']]
    assert len(roles) == 4*2*(2*11+3)
    for c, r in roles:
        assert c['bit_assignment']['model.layers.3.mlp.down_proj'] == 4
        assert payload(c['bit_assignment'], meta) == round(r['budget']*sum(m['num_params'] for m in meta))
        if r['method'] == 'Family-Mean' and r['k_layers'] == 2:
            other = next(x for x, z in roles if (z['method'], z['k_layers'], z['budget'], z['policy'], z['tie_order']) ==
                ('Family-Median', 2, r['budget'], r['policy'], r['tie_order']))
            assert c['candidate_id'] == other['candidate_id']
    raw = dict(uniform_baselines={n: evaluate_perplexity(model, ids, meta, cfg, uniform_bit=b, batch_size=1)
        for n, b in [('FP16', None), ('Uniform-3bit', 3), ('Uniform-4bit', 4)]})
    report = evaluate_candidates(model, ids, cfg, plan, raw, quality, summary_writer=summarize_quality,
        result_fields=dict(schema='coverage_quality_result_v1', dataset='wikitext2'))
    assert report['complete'] and (quality/'coverage_curves.csv').exists()
    timing = tmp_path/'timing'; timing.mkdir()
    cell = next(c for c in plan['acquisition_plan']['cells'] if c['k_layers'] == 2 and c['policy'] == 'boundary_inclusive')
    timed = engine.run_timing(model, calib, cfg, spec, 42, cell, timing, {})
    summarize_timing(timed, timing)
    assert counts['labels'] == 6 and counts['features'] == 4
    assert timed['complete'] and len(timed['trials']) == 5
    for trial in timed['trials']:
        assert len(trial['measured_labels']) == 42 and trial['calibration_wall_sec'] > 0
        for budget, allocation in trial['allocations'].items():
            matching = [c for c, r in roles if (r['method'], r['k_layers'], r['policy'], r['budget'], r['tie_order']) ==
                (trial['method'], 2, 'boundary_inclusive', float(budget),
                 'depth_ascending' if trial['method'] in ARMS[:2] else 'native')]
            assert len(matching) == 1 and matching[0]['bit_assignment'] == allocation['bit_assignment']
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)
    # Check the public aggregation path and strict timing/quality assignment matching.
    common = dict(schema='coverage_study_v1', complete=True, plan_only=False, plan_complete=True,
        dataset='wikitext2', seed=42, code_sha256={'module': 'same'}, packages={'torch': 'same'}, effective_hqq_config={})
    for folder in [quality, timing]:
        save(folder/'experiment_config.json', spec)
    save(quality/'selection_plan.json', plan)
    save(quality/'study_manifest.json', dict(common, mode='quality',
        config_sha256=digest(quality/'experiment_config.json'), selection_plan_sha256=digest(quality/'selection_plan.json')))
    save(timing/'study_manifest.json', dict(common, mode='timing', k_layers=2, policy='boundary_inclusive',
        config_sha256=digest(timing/'experiment_config.json'), timing_sha256=digest(timing/'timing.json')))
    result = aggregate([quality, timing], tmp_path/'aggregate')
    assert result['timing_assignment_checks'] == 10
    altered = copy.deepcopy(timed); first = altered['trials'][0]['allocations']['3.00']['bit_assignment']
    name = next(iter(first)); first[name] = 4 if first[name] != 4 else 3
    save(timing/'timing.json', altered)
    m = json.loads((timing/'study_manifest.json').read_text()); m['timing_sha256'] = digest(timing/'timing.json'); save(timing/'study_manifest.json', m)
    with pytest.raises(ValueError, match='differs from quality'):
        aggregate([quality, timing], tmp_path/'bad')
