"""Frozen source validation, ten-order sensitivity, no-op protection and paired reporting."""
import copy
import json
import math
from pathlib import Path

import pytest
import torch

from hds_quant.allocation_controls import family_tie_order, last_down, simple_assignment
from hds_quant.amq_followup_plan import METHODS, digest, geometry, payload
from hds_quant.config import ExperimentConfig
from hds_quant.frozen_replication_plan import (DATASETS, MODEL, MODEL_REVISION, ORDERS,
    add_replication_estimator, build_replication_plan, load_replication_source, validate_spec)
from hds_quant.frozen_replication_summary import aggregate, materiality, summarize_replication
from hds_quant.readout_cv import select_stratified_k_layers
from hds_quant.run_amq_followup import save

ROOT = Path(__file__).resolve().parents[1]


def spec_for(dataset='wikitext'):
    return json.loads((ROOT/f'configs/frozen_replication/llama31_8b_{dataset}_v1.json').read_text())


@pytest.fixture
def source(tmp_path):
    meta = geometry(); total = sum(m['num_params'] for m in meta)
    src = dict(schema='representation_controls_v1', purpose='benchmark', model_id=MODEL,
        model_revision=MODEL_REVISION, dataset=DATASETS['wikitext2'], seeds=[42, 100, 2026],
        k_layers=[4], methods=METHODS, probe_bits=[1, 2, 3, 4], allocation_bits=[2, 3, 4],
        budgets=[3., 3.5], full_reference=False,
        calibration=dict(n_sequences=64, seq_len=1024, max_encoded_tokens=4096),
        eval_seq_len=2048, batch_size=1)
    cfg = ExperimentConfig(model_name=MODEL, num_layers=32, quantizer='hqq', hqq_axis=0,
        hqq_zero_policy='floating', candidate_bits=(1, 2, 3, 4), seeds=(2026,),
        calib_num_sequences=64, calib_seq_len=1024, max_encoded_tokens=4096,
        residual_anchor='int4_aligned', coord_binding='salience_bucket',
        readout_grouping='per_bit', target_transform='signed_log1p')
    evaluation = dict(nll=2., ppl=math.exp(2), seq_nlls=[2.]*141, seq_token_counts=[2047]*141)
    raw_values = {f'{m["name"]}|{b}': (4-b)*m['num_params']/total for m in meta for b in [1, 2, 3]}
    trials = []
    for seed in [42, 100, 2026]:
        layers = select_stratified_k_layers(list(range(32)), 4, seed)
        measured = {k: v for k, v in raw_values.items() if int(k.split('.')[2]) in layers}
        predictions = {k: v for k, v in raw_values.items() if k not in measured}
        allocations = {}
        for b in [3., 3.5]:
            bits = simple_assignment(meta, b, 'last_half')
            objective = math.fsum((4-v)*next(m['num_params'] for m in meta if m['name'] == n)/total for n, v in bits.items())
            allocations[f'{b:.2f}'] = dict(success=True, bit_assignment=bits,
                achieved_avg_bits=b, total_predicted_cost=objective, evaluation=evaluation)
        for method in METHODS:
            trials.append(dict(seed=seed, method=method, k_layers=4, acquired_layers=layers,
                probe_label_count=84, measured_labels=measured, heldout_predictions=predictions,
                allocations=allocations))
    raw = dict(complete=True, config=cfg.to_dict(), trials=trials, evaluation_token_sha256='a'*64,
        evaluation_chunks=141, calibration_token_sha256={str(s): str(s)*10 for s in [42, 100, 2026]},
        uniform_baselines={n: evaluation for n in ['FP16', 'Uniform-3bit', 'Uniform-4bit']})
    for name, value in [('experiment_config.json', src), ('resolved_experiment_config.json', src),
        ('limited_labels.json', raw), ('data_manifest.json', dict(evaluation_token_sha256='a'*64, scored_tokens=288627))]:
        save(tmp_path/name, value)
    save(tmp_path/'experiment_manifest.json', dict(complete=True, smoke_test=False,
        config_sha256=digest(tmp_path/'experiment_config.json'), resolved_model_revision=MODEL_REVISION,
        resolved_dataset_revision=src['dataset']['revision'], evaluation_token_sha256='a'*64))
    return tmp_path


def test_ten_orders_are_frozen_and_old_diagnostic_rejects_new_orders():
    spec = spec_for()
    assert len(ORDERS) == 11 and len(set(ORDERS)) == 11
    assert len([o for o in ORDERS if o.startswith('random_')]) == 10
    for seed in [42, 100, 2026]:
        validate_spec(spec, seed)
    for key, value in [('materiality_percent', 2), ('family_tie_orders', list(ORDERS[:-1])),
                       ('protection_rule', 'always_resolve'), ('k_layers', 8)]:
        with pytest.raises(ValueError, match='Frozen'):
            validate_spec(dict(spec, **{key: value}), 100)
    meta = geometry(); costs = {(m['name'], b): float(4-b) for m in meta for b in [2, 3, 4]}
    bits = simple_assignment(meta, 3.5, 'last_half')
    with pytest.raises(ValueError, match='Unknown tie'):
        family_tie_order(bits, costs, meta, set(), 'random_173')
    chosen = family_tie_order(bits, costs, meta, set(), 'random_173', allowed_orders=ORDERS)
    assert payload(chosen, meta) == payload(bits, meta)


def test_all_seed_plans_restore_case_seed_keep_84_labels_and_do_not_modify_source(source):
    before = {p: p.read_bytes() for p in source.glob('*.json')}
    for seed in [42, 100, 2026]:
        plan, raw, measured, family_raw = build_replication_plan(spec_for(), source, seed)
        assert plan['experiment_config']['seeds'] == [seed]
        assert len(measured) == 84
        assert all(family_raw[k] == v for k, v in measured.items())
        roles = [r for c in plan['candidates'] for r in c['roles']]
        assert len([r for r in roles if r['method'] == 'Family-Mean']) == 66
        assert len(roles) == 96
        total = sum(m['num_params'] for m in geometry())
        for c in plan['candidates']:
            for role in c['roles']:
                assert payload(c['bit_assignment'], geometry()) == round(role['budget']*total)
                if role['setting'] == 'last_down4':
                    assert c['bit_assignment'][last_down(geometry())] == 4
        assert plan['metadata_pending']
    assert before == {p: p.read_bytes() for p in before}


@pytest.mark.parametrize('corruption', ['config_hash', 'completion', 'prediction', 'label_disagreement', 'dataset'])
def test_source_corruption_is_rejected(source, corruption):
    if corruption == 'config_hash':
        p = source/'experiment_config.json'; p.write_text(p.read_text()+' ')
    elif corruption == 'completion':
        p = source/'experiment_manifest.json'; r = json.loads(p.read_text()); r['complete'] = False; save(p, r)
    elif corruption in ['prediction', 'label_disagreement']:
        p = source/'limited_labels.json'; r = json.loads(p.read_text()); t = r['trials'][0]
        if corruption == 'prediction':
            t['heldout_predictions'].pop(next(iter(t['heldout_predictions'])))
        else:
            # A 1-bit acquired label cannot change any source {2,3,4} objective, but must still be checked.
            key = next(k for k in t['measured_labels'] if k.endswith('|1')); t['measured_labels'][key] += .1
        save(p, r)
    with pytest.raises((ValueError, KeyError)):
        load_replication_source(source, 'c4' if corruption == 'dataset' else 'wikitext2', 42)


def test_feasible_protection_never_resolves_or_changes_any_assignment(monkeypatch):
    meta = geometry(); costs = {(m['name'], b): float(4-b) for m in meta for b in [2, 3, 4]}
    allocations = {f'{b:.2f}': dict(bit_assignment=simple_assignment(meta, b, 'last_half', True), evaluation={}) for b in [2.5, 3., 3.5]}
    def forbidden(*args, **kwargs):
        raise AssertionError('Already feasible source assignment must be preserved')
    monkeypatch.setattr('hds_quant.frozen_replication_plan.solve_control', forbidden)
    plan = dict(candidates=[])
    add_replication_estimator(plan, 'HDS-Hybrid', costs, meta, spec_for(), trial=dict(allocations=allocations))
    assert len(plan['candidates']) == 3
    for c in plan['candidates']:
        assert len(c['roles']) == 2
        protected = next(r for r in c['roles'] if r['setting'] == 'last_down4')
        assert protected['protection_was_noop'] and protected['protection_changed_projections'] == 0


def test_actual_small_model_uses_new_reporting_and_restores_weights(tmp_path):
    from hds_quant.run_allocation_controls import evaluate_candidates
    from hds_quant.run_experiment import _build_smoke_model
    from hds_quant.label_collector import discover_target_projections
    from hds_quant.evaluator import evaluate_perplexity
    from hds_quant.allocation_control_plan import add_candidate
    torch.set_num_threads(1)
    model, ids = _build_smoke_model(); ids = ids[:1, :8]
    cfg = ExperimentConfig(num_layers=model.config.num_hidden_layers, quantizer='hqq',
        hqq_axis=0, quant_group_size=64, device='cpu')
    meta = discover_target_projections(model)
    raw = dict(uniform_baselines={n: evaluate_perplexity(model, ids, meta, cfg, uniform_bit=b, batch_size=1)
        for n, b in [('FP16', None), ('Uniform-3bit', 3), ('Uniform-4bit', 4)]})
    plan = dict(seed=100, acquired_layers=[0], probe_label_count=21, candidates=[], timing_scope='reused')
    bits = simple_assignment(meta, 3.5, 'last_half')
    for method, orders in [('HDS-Hybrid', ['native']), ('Family-Mean', ORDERS)]:
        for order in orders:
            for setting in ['unconstrained', 'last_down4']:
                add_candidate(plan, bits, meta, dict(method=method, budget=3.5, setting=setting, tie_order=order,
                    predicted_cost=None, source='test', protection_changed_projections=0,
                    protection_was_noop=setting == 'last_down4'))
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    result = evaluate_candidates(model, ids, cfg, plan, raw, tmp_path,
        summary_writer=summarize_replication,
        result_fields=dict(schema='frozen_replication_result_v1', dataset='c4'))
    assert result['complete'] and result['seed'] == 100 and len(result['candidates']) == 1
    assert (tmp_path/'placement_sensitivity.csv').is_file()
    assert 'seed 100' in (tmp_path/'SUMMARY.txt').read_text()
    for n, p in model.named_parameters():
        torch.testing.assert_close(p, before[n], rtol=0, atol=0)


def completed_case(root, seed):
    folder = root/f'seed{seed}'; folder.mkdir()
    spec = spec_for(); report = dict(schema='frozen_replication_result_v1', complete=True,
        dataset='wikitext2', seed=seed, candidates=[])
    for b in [2.5, 3., 3.5]:
        for setting in ['unconstrained', 'last_down4']:
            for method, orders, ppl in [('HDS-Hybrid', ['native'], 9.8), ('Family-Mean', ORDERS, 10.)]:
                for order in orders:
                    report['candidates'].append(dict(candidate_id=f'{b}-{setting}-{method}-{order}',
                        roles=[dict(method=method, budget=b, setting=setting, tie_order=order)], evaluation=dict(ppl=ppl)))
    save(folder/'experiment_config.json', spec); save(folder/'selection_plan.json', {})
    save(folder/'diagnostic.json', report)
    save(folder/'diagnostic_manifest.json', dict(schema='frozen_replication_v1', complete=True,
        plan_complete=True, plan_only=False, dataset='wikitext2', seed=seed,
        config_sha256=digest(folder/'experiment_config.json'), selection_plan_sha256=digest(folder/'selection_plan.json'),
        code_sha256={'module': 'same'}, packages={'torch': 'same'}, effective_hqq_config={'axis': 0}))
    return folder


def test_aggregation_counts_calibration_seeds_not_orders_and_rejects_duplicates(tmp_path):
    cases = [completed_case(tmp_path, s) for s in [42, 100, 2026]]
    decisions = aggregate(cases, tmp_path/'aggregate')
    assert all(r['calibration_seed_count'] == 3 for r in decisions)
    assert all(r['decision'] == 'consistent_material_HDS_advantage_on_primary' for r in decisions)
    partial = aggregate(cases[1:], tmp_path/'partial')
    assert all(r['decision'] == 'incomplete_three_seed_matrix' for r in partial)
    with pytest.raises(ValueError, match='Duplicate'):
        aggregate(cases+cases[:1], tmp_path/'duplicate')
    manifest = cases[0]/'diagnostic_manifest.json'; value = json.loads(manifest.read_text())
    value['packages']['torch'] = 'changed'; save(manifest, value)
    with pytest.raises(ValueError, match='Mixed code'):
        aggregate(cases, tmp_path/'mismatch')
    assert materiality(.2) == 'below_practical_margin_not_equivalence'
