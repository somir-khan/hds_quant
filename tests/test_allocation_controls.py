"""Tests of control identity, label isolation, payloads, ties and real evaluation restoration."""
import copy
import json
import math

import pytest
import torch

from hds_quant.allocation_controls import (TIE_ORDERS, allocation_costs, family_mean_predictions,
    family_tie_order, last_down, simple_assignment, solve_control)
from hds_quant.allocation_control_plan import add_candidate, fit_metadata_controls, validate_spec
from hds_quant.amq_followup_plan import geometry, payload
from hds_quant.config import ExperimentConfig
from hds_quant.estimators import compute_estimators, compute_selected_estimator
from hds_quant.hds_encoder import HDSEncoder


def toy_meta():
    return [dict(name=f'model.layers.{i}.mlp.{f}_proj', layer_idx=i, family_idx=j,
                 num_params=10) for i in range(6) for j, f in enumerate(['gate', 'down'])]


def test_family_raw_mean_before_clamp_and_retains_individual_labels():
    meta = toy_meta(); cfg = ExperimentConfig(candidate_bits=(1, 2, 3, 4))
    measured = {(m['name'], b): (-1 if m['layer_idx'] == 0 else 3)*(4-b)
                for m in meta if m['layer_idx'] in (0, 5) for b in cfg.marginal_bits}
    before = copy.deepcopy(measured)
    raw = family_mean_predictions(measured, meta, cfg.marginal_bits)
    for key, value in measured.items():
        assert raw[key] == value
    for m in meta:
        if m['layer_idx'] not in (0, 5):
            assert raw[m['name'], 3] == 1  # Mean(-1,3), not mean(clamp(-1),clamp(3)).
    assert measured == before
    costs = allocation_costs(raw, meta, cfg)
    assert all(costs[m['name'], 2] >= costs[m['name'], 3] >= costs[m['name'], 4] == 0 for m in meta)
    missing = {k: v for k, v in measured.items() if not k[0].endswith('gate_proj')}
    with pytest.raises(ValueError, match='Every family'):
        family_mean_predictions(missing, meta, cfg.marginal_bits)


@pytest.mark.parametrize('budget', [2.5, 3.0, 3.5])
def test_exact_allocation_and_three_tie_orders_preserve_primary_objective(budget):
    meta = toy_meta(); costs = {(m['name'], b): float(4-b) for m in meta for b in (2, 3, 4)}
    for protected in [False, True]:
        original = solve_control(costs, meta, budget, protected)
        expected = math.fsum(costs[n, b] for n, b in original.items())
        for order in TIE_ORDERS:
            bits = family_tie_order(original, costs, meta, set(), order,
                [last_down(meta)] if protected else [])
            assert payload(bits, meta) == int(budget*120)
            assert math.fsum(costs[n, b] for n, b in bits.items()) == expected
            if protected:
                assert bits[last_down(meta)] == 4
            assert family_tie_order(original, costs, list(reversed(meta)), set(), order,
                [last_down(meta)] if protected else []) == bits


def test_tie_order_changes_placement_but_preserves_measured_projections():
    meta = toy_meta(); costs = {(m['name'], b): float(4-b) for m in meta for b in (2, 3, 4)}
    bits = {m['name']: 2+(m['layer_idx'] % 3) for m in meta}
    measured = {m['name'] for m in meta if m['layer_idx'] == 0}
    variants = [family_tie_order(bits, costs, meta, measured, o) for o in TIE_ORDERS]
    assert len({json.dumps(b, sort_keys=True) for b in variants}) > 1
    for variant in variants:
        assert all(variant[n] == bits[n] for n in measured)
        for family in (0, 1):
            names = [m['name'] for m in meta if m['family_idx'] == family]
            assert sorted(variant[n] for n in names) == sorted(bits[n] for n in names)


def test_real_geometry_simple_controls_and_label_free_compensation():
    meta = geometry(); total = sum(m['num_params'] for m in meta)
    for budget in [2.5, 3., 3.5]:
        for pattern in ['last_half', 'alternating_odd']:
            native = simple_assignment(meta, budget, pattern)
            protected = simple_assignment(meta, budget, pattern, True)
            assert payload(native, meta) == payload(protected, meta) == round(budget*total)
            assert protected[last_down(meta)] == 4
            if budget == 3:
                assert native == {m['name']: 3 for m in meta}
                changed = {n: b for n, b in protected.items() if b != 3}
                assert changed == {last_down(meta): 4, 'model.layers.0.mlp.gate_proj': 2}
            if budget == 3.5:
                assert native == protected


def test_selected_metadata_exactly_matches_hds_tail_without_rank_or_binding(monkeypatch):
    residual = torch.rand(10, 32, generator=torch.Generator().manual_seed(5))
    norms = torch.ones(32)
    encoder = HDSEncoder(hd_dim=512, max_coords=64, num_buckets=32, device='cpu')
    full = compute_estimators(residual, norms, 4, encoder)
    def forbidden(*args, **kwargs):
        raise AssertionError('Metadata must not rank or bind residuals')
    monkeypatch.setattr(HDSEncoder, 'level_frequencies', forbidden)
    monkeypatch.setattr(HDSEncoder, 'bind', forbidden)
    selected = compute_selected_estimator(residual, norms, 4, encoder, 'Scalar-Meta-Ridge')['Scalar-Meta-Ridge']
    torch.testing.assert_close(selected, full['Scalar-Meta-Ridge'], rtol=0, atol=0)
    torch.testing.assert_close(selected, full['HDS-Hybrid'][-9:], rtol=0, atol=0)


def test_metadata_fit_keeps_labels_and_depth_is_separate():
    meta = toy_meta(); cfg = ExperimentConfig(candidate_bits=(1, 2, 3, 4),
        readout_grouping='per_bit', target_transform='signed_log1p')
    features = {(m['name'], b): {'Scalar-Meta-Ridge': torch.tensor(
        [float(m['layer_idx']), float(b)]+[float(j == m['family_idx']) for j in range(7)])}
        for m in meta for b in cfg.marginal_bits}
    acquired = [0, 5]
    measured = {(m['name'], b): float((m['layer_idx']+1)*(4-b))
        for m in meta if m['layer_idx'] in acquired for b in cfg.marginal_bits}
    original = {k: v['Scalar-Meta-Ridge'].clone() for k, v in features.items()}
    fits = fit_metadata_controls(features, measured, meta, cfg, acquired, 42)
    assert [v['feature_columns'] for v in fits.values()] == [9, 10]
    for fit in fits.values():
        assert all(fit['raw'][k] == v for k, v in measured.items())
        assert len(fit['raw']) == len(meta)*3
    for k in features:
        torch.testing.assert_close(features[k]['Scalar-Meta-Ridge'], original[k], rtol=0, atol=0)


def test_real_small_model_controls_dedup_replay_and_restore(tmp_path):
    from hds_quant.run_allocation_controls import evaluate_candidates
    from hds_quant.run_experiment import _build_smoke_model
    from hds_quant.label_collector import discover_target_projections
    from hds_quant.evaluator import evaluate_perplexity
    torch.set_num_threads(1)
    model, ids = _build_smoke_model(); ids = ids[:1, :8]
    cfg = ExperimentConfig(num_layers=model.config.num_hidden_layers, quantizer='hqq',
        hqq_axis=0, quant_group_size=64, device='cpu')
    meta = discover_target_projections(model)
    raw = dict(uniform_baselines={n: evaluate_perplexity(model, ids, meta, cfg, uniform_bit=b, batch_size=1)
        for n, b in [('FP16', None), ('Uniform-3bit', 3), ('Uniform-4bit', 4)]})
    bits = simple_assignment(meta, 3.5, 'last_half', True)
    expected = evaluate_perplexity(model, ids, meta, cfg, bit_assignment=bits, batch_size=1)
    plan = dict(seed=42, acquired_layers=[0], probe_label_count=21, candidates=[], timing_scope='reused')
    for method in ['simple', 'HDS-Hybrid']:
        add_candidate(plan, bits, meta, dict(method=method, budget=3.5, setting='last_down4',
            tie_order='declared', source='test', predicted_cost=None), expected)
    assert len(plan['candidates']) == 1 and len(plan['candidates'][0]['roles']) == 2
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = evaluate_candidates(model, ids, cfg, plan, raw, tmp_path)
    assert result['complete'] and len(result['candidates']) == 1
    assert (tmp_path/'hds_vs_controls.csv').is_file()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)
    bad = tmp_path/'bad'; bad.mkdir()
    raw['uniform_baselines']['FP16']['nll'] += .1
    with pytest.raises(ValueError, match='replay failed'):
        evaluate_candidates(model, ids, cfg, plan, raw, bad)
    assert not json.loads((bad/'diagnostic.json').read_text())['complete']


def test_spec_rejects_budget_or_tie_policy_changes():
    from pathlib import Path
    spec = json.loads((Path(__file__).resolve().parents[1]/
        'configs/allocation_controls/llama31_8b_wikitext_seed42_v1.json').read_text())
    validate_spec(spec)
    for key, value in [('budgets', [3., 3.5]), ('family_tie_orders', ['random_42']), ('seed', 100)]:
        bad = dict(spec, **{key: value})
        with pytest.raises(ValueError, match='Frozen'):
            validate_spec(bad)
