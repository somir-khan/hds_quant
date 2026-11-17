"""Meaningful controls: preserve weighting, isolate retention, identify composition."""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from hds_quant.arr_stage12_diagnostics import composition_terms, joint_subsets, prediction_metrics
from hds_quant.arr_stage12_protocol import (case_definitions, direct_residual, encode,
    fit_reduced, read, reduced_config, validate_spec)
from hds_quant.config import ExperimentConfig, PROJ_FAMILIES
from hds_quant.readout_cv import evaluate_fixed_k_layer_acquisition

ROOT = Path(__file__).resolve().parents[1]


def spec():
    return read(ROOT/'configs/arr_stage12/frozen_v1.json')


def test_frozen_scope_and_no_cross_case_override():
    s = spec(); validate_spec(s)
    cases = case_definitions(s)
    assert len({c['case_id'] for c in cases}) == 12
    assert cases[6]['model'] == '1b' and cases[6]['dataset'] == 'wikitext2' and cases[6]['seed'] == 42
    assert [cases[i]['seed'] for i in range(3)] == [42, 100, 2026]
    for key, value in [('probe_bits', [1, 2, 3]), ('e_case_index', 7),
                       ('preserve_legacy_indicator_geometry', False), ('budgets', [3.25])]:
        with pytest.raises(ValueError, match='Frozen'):
            validate_spec(dict(s, **{key: value}))


def toy_bank():
    meta = [dict(name=f'model.layers.{l}.{f}', layer_idx=l, family_idx=j,
                 num_params=128, d_in=16, d_out=8) for l in range(4) for j, f in enumerate(PROJ_FAMILIES)]
    rng = np.random.RandomState(6); bank = {}; labels = {}
    for m in meta:
        for b in (1, 2, 3):
            scal = np.r_[rng.normal(size=2), np.eye(7)[m['family_idx']]].astype('float32')
            bank[m['name'], b] = {'Scalar-Meta-Ridge': torch.tensor(scal),
                'Coord-Rank-Hist': torch.tensor(np.r_[rng.normal(size=32), scal].astype('float32')),
                'HDS-Hybrid': torch.tensor(np.r_[rng.normal(size=64), scal].astype('float32'))}
            labels[m['name'], b] = .02*(4-b)+.001*m['layer_idx']+.002*m['family_idx']
    return meta, bank, labels


@pytest.mark.parametrize('method', ['Scalar-Meta-Ridge', 'Coord-Rank-Hist', 'HDS-Hybrid'])
def test_reduced_fit_matches_legacy_despite_changing_unused_one_bit_labels(method):
    meta, bank, labels = toy_bank(); acquired = [0, 3]
    old = ExperimentConfig(candidate_bits=(1, 2, 3, 4), num_layers=4, hd_dim=64,
        seeds=[42], readout_grouping='per_bit', target_transform='signed_log1p',
        ridge_alphas=(.01, 1., 100.))
    measured_old = {k: v for k, v in labels.items() if int(k[0].split('.')[2]) in acquired}
    legacy = evaluate_fixed_k_layer_acquisition(bank, measured_old, meta, old,
        k_layers=2, seed=42, estimator_names=[method])['raw_predictions'][method]
    measured = {k: v for k, v in measured_old.items() if k[1] in (2, 3)}
    pred, retained = fit_reduced(bank, measured, meta, reduced_config(old), acquired, method)
    assert max(abs(v-legacy[k]) for k, v in retained.items()) < 1e-10
    assert all(retained[k] == v for k, v in measured.items())
    assert any(abs(pred[k]-measured[k]) > 1e-8 for k in measured)
    changed = {k: 10000*v if k[1] == 1 else v for k, v in measured_old.items()}
    replay = evaluate_fixed_k_layer_acquisition(bank, changed, meta, old,
        k_layers=2, seed=42, estimator_names=[method])['raw_predictions'][method]
    assert max(abs(replay[k]-legacy[k]) for k in retained) < 1e-10


def test_direct_residual_has_no_label_access_and_depth_controls_need_no_wide_bank():
    meta, bank, labels = toy_bank()
    simple = {k: {'Scalar-Meta-Ridge': v['Scalar-Meta-Ridge']} for k, v in bank.items()}
    raw = direct_residual(simple, meta)
    key = next(iter(raw))
    assert raw[key] == max(math.exp(float(simple[key]['Scalar-Meta-Ridge'][0]))-1e-8, 0.)
    acquired = [0, 3]; measured = {k: v for k, v in labels.items() if k[1] in (2, 3)
                                 and int(k[0].split('.')[2]) in acquired}
    cfg = ExperimentConfig(num_layers=4, readout_grouping='per_bit', ridge_alphas=(.01, 1.))
    for depth in ('family_only', 'metadata'):
        pred, retained = fit_reduced(simple, measured, meta, cfg, acquired, 'Scalar-Meta-Ridge', depth=depth)
        assert len(pred) == 56 and all(retained[k] == v for k, v in measured.items())


def test_prediction_and_composition_errors_are_separate_signed_terms():
    bits = {'A': 2, 'B': 3, 'C': 4}
    truth = {('A', 2): .10, ('B', 3): .20}
    pred = {('A', 2): .12, ('B', 3): .18}
    processed = {('A', 2): .15, ('B', 3): .20, ('C', 4): 0.}
    r = composition_terms(bits, truth, pred, 2., 2.5, processed)
    assert r['aggregate_prediction_error'] == pytest.approx(0.)
    assert r['composition_residual'] == pytest.approx(.20)
    assert r['processing_shift'] == pytest.approx(.05)
    assert r['actual_minus_processed'] == pytest.approx(.15)
    assert len(joint_subsets(['gate', 'up', 'down'])) == 8
    assert sum(len(s) > 1 for s in joint_subsets(['gate', 'up', 'down'])) == 4


def test_metrics_exclude_training_blocks_and_do_not_invent_constant_rank():
    meta, _, labels = toy_bank(); pred = {k: v for k, v in labels.items() if k[1] in (2, 3)}
    for k in pred:
        pred[k] = 10. if int(k[0].split('.')[2]) in [0, 3] else 1.
    rows = prediction_metrics(pred, labels, meta, [0, 3])
    assert rows[0]['n'] == 14 and all(r['spearman'] is None for r in rows)


def test_e_orchestration_uses_same_calibration_context_and_freezes_full_bank(tmp_path, monkeypatch):
    """Synthetic loss has known joint residual; no HF model or data downloaded."""
    from hds_quant.allocation_controls import simple_assignment
    from hds_quant.run_arr_stage12 import diagnostic
    import hds_quant.label_collector as collector
    import hds_quant.evaluator as evaluator
    meta = [dict(name=f'model.layers.{l}.{f}', layer_idx=l, family_idx=j,
                 num_params=128, d_in=16, d_out=8) for l in range(16) for j, f in enumerate(PROJ_FAMILIES)]
    full = {(m['name'], b): .001*(4-b)*(1+m['family_idx']) for m in meta for b in (2, 3)}
    known = set(range(8)) | {15}
    cached = {k: v for k, v in full.items() if int(k[0].split('.')[2]) in known}
    cfg = ExperimentConfig(num_layers=16, seeds=[42], readout_grouping='per_bit')
    context_calls = []
    def evaluation(model, ids, meta, cfg, bit_assignment=None, uniform_bit=None, batch_size=1):
        bits = bit_assignment or {m['name']: uniform_bit for m in meta}
        joint = sum(bits[f'model.layers.0.mlp.{f}_proj'] == 2 for f in ('gate', 'up', 'down'))
        nll = 2.+sum(full[n, b] for n, b in bits.items() if b != 4)+(.03 if joint >= 2 else 0.)
        context_calls.append((ids, dict(bits)))
        return dict(nll=nll, ppl=math.exp(nll), seq_nlls=[nll], seq_token_counts=[7])
    def acquire(model, calib, cfg, batch_size, target_layers):
        assert set(target_layers) == (set(range(16))-known) | {0, 15}
        return dict(delta_loss={k: v for k, v in full.items() if int(k[0].split('.')[2]) in target_layers},
            base_loss_4bit=2., label_time_sec=1.)
    monkeypatch.setattr(collector, 'collect_marginal_loss_labels', acquire)
    monkeypatch.setattr(evaluator, 'evaluate_perplexity', evaluation)
    cells = []; candidates = []
    for k, p, layers in [(2, 'boundary_inclusive', [0, 15]), (8, 'stratified', list(range(8)))]:
        cell = dict(k_layers=k, policy=p, acquired_layers=layers, raw_predictions={})
        for method in ('Scalar-Meta-Ridge', 'Coord-Rank-Hist', 'HDS-Hybrid'):
            cell['raw_predictions'][method] = encode({key: v if int(key[0].split('.')[2]) in layers else .9*v
                                                    for key, v in full.items()})
            for budget in (3., 3.5):
                bits = simple_assignment(meta, budget, 'last_half', True)
                candidates.append(dict(bit_assignment=bits, evaluation=evaluation(None, 'heldout', meta, cfg, bits),
                    roles=[dict(method=method, budget=budget, k_layers=k, policy=p, tie_order='native')]))
        cells.append(cell)
    source = dict(meta=meta, labels=cached, estimates=dict(cells=cells), report=dict(candidates=candidates))
    diagnostic(None, 'calibration', 'heldout', cfg, source, spec(), tmp_path)
    report = read(tmp_path/'diagnostic.json')
    assert report['complete'] and len(report['joint_first_block']) == 8
    assert all(r['interaction_residual'] == pytest.approx(.03) for r in report['joint_first_block'] if r['count'] >= 2)
    bank = read(tmp_path/'full_label_bank.json')
    assert len(bank['measured_labels']) == 224 and bank['newly_acquired_labels'] == 98
    assert bank['replayed_labels'] == 28
    assert (tmp_path/'composition_residuals.csv').exists()
