"""Pinned legacy bridge, dataset isolation, model-derived geometry and focused reporting."""
import copy
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from hds_quant.amq_followup_plan import digest
from hds_quant.config import ExperimentConfig
from hds_quant.coverage_protocol import ARMS
from hds_quant.focused_1b_protocol import (MODEL, REVISION, focused_acquisition,
    load_historical_source, make_config, projection_geometry, validate_model, validate_spec)
from hds_quant.focused_1b_summary import aggregate, check_timing
from hds_quant.frozen_replication_plan import DATASETS
from hds_quant.run_amq_followup import save

ROOT = Path(__file__).resolve().parents[1]


def spec_for():
    return json.loads((ROOT/'configs/coverage_generalization/llama32_1b_both_focused_v1.json').read_text())


@pytest.fixture
def legacy(tmp_path):
    cfg = make_config(spec_for(), 42).to_dict()
    src = dict(schema='generalization_v1', model_id=MODEL, model_revision=REVISION,
        dataset=DATASETS['c4'], batch_size=8, seeds=[42, 100, 2026], eval_seq_len=2048,
        calibration=dict(n_sequences=64, seq_len=1024, max_encoded_tokens=4096))
    original = copy.deepcopy(src); original['model_revision'] = 'main'; original['dataset']['revision'] = 'main'
    raw = dict(complete=True, config=cfg, evaluation_token_sha256='a'*64,
        calibration_token_sha256={'42': 'b'*64}, hqq_config={'axis': 0}, uniform_baselines={'old': {'ppl': 999}})
    for name, value in [('experiment_config.json', original), ('resolved_experiment_config.json', src),
        ('limited_labels.json', raw), ('data_manifest.json', dict(evaluation_token_sha256='a'*64, scored_tokens=288627))]:
        save(tmp_path/name, value)
    save(tmp_path/'experiment_manifest.json', dict(complete=True, smoke_test=False,
        config_sha256=digest(tmp_path/'experiment_config.json'), resolved_model_revision=REVISION,
        resolved_dataset_revision=DATASETS['c4']['revision'], evaluation_token_sha256='a'*64))
    return tmp_path


def test_legacy_main_resolution_is_accepted_but_old_scores_are_not_returned(legacy):
    historical = load_historical_source(legacy)
    assert historical['historical_batch_size'] == 8
    assert historical['old_scores_are_not_replay_targets']
    assert 'uniform_baselines' not in historical and 'measured_labels' not in historical
    bad = legacy/'resolved_experiment_config.json'; value = json.loads(bad.read_text()); value['model_revision'] = 'main'; save(bad, value)
    with pytest.raises(ValueError, match='pin mismatch'):
        load_historical_source(legacy)


def test_frozen_config_both_datasets_correct_seed_and_16_block_fractions():
    spec = spec_for(); validate_spec(spec)
    assert [d['name'] for d in spec['datasets']] == ['wikitext2', 'c4']
    for seed in spec['seeds']:
        cfg = make_config(spec, seed, 'cpu'); assert cfg.seeds == [seed] and cfg.num_layers == 16
        acquisition = focused_acquisition(spec, seed)
        a, b = acquisition['cells']
        assert a['acquired_layers'] == [0, 15] and a['block_coverage_fraction'] == .125
        assert b['block_coverage_fraction'] == .5 and len(b['acquired_layers']) == 8
        assert a['probe_label_count'] == 42 and b['probe_label_count'] == 168
    for key, value in [('batch_size', 8), ('model_revision', 'main'), ('num_layers', 32), ('primary_budget', 3.)]:
        with pytest.raises(ValueError, match='Frozen'):
            validate_spec(dict(spec, **{key: value}))


def test_geometry_uses_checkpoint_heads_and_shapes_without_materializing_weights():
    from transformers import LlamaConfig, LlamaForCausalLM
    from hds_quant.label_collector import discover_target_projections
    config = LlamaConfig(hidden_size=2048, intermediate_size=8192, num_hidden_layers=16,
        num_attention_heads=32, num_key_value_heads=8, vocab_size=128256)
    with torch.device('meta'):
        model = LlamaForCausalLM(config)
    meta = discover_target_projections(model)
    expected = validate_model(model, meta)
    assert len(expected) == 112
    key = next(m for m in expected if m['name'] == 'model.layers.0.self_attn.k_proj')
    assert (key['d_out'], key['d_in']) == (512, 2048)
    corrupt = copy.deepcopy([{k: v for k, v in m.items() if k != 'module'} for m in meta])
    corrupt[0]['d_out'] += 1
    with pytest.raises(ValueError, match='geometry'):
        validate_model(model, corrupt)


def test_fresh_baselines_are_current_pipeline_and_restore_model():
    from hds_quant.run_focused_1b import fresh_baselines
    from hds_quant.run_experiment import _build_smoke_model
    torch.set_num_threads(1)
    model, ids = _build_smoke_model(); ids = ids[:1, :8]
    cfg = ExperimentConfig(quantizer='hqq', hqq_axis=0, quant_group_size=64, device='cpu')
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    first = fresh_baselines(model, ids, cfg); second = fresh_baselines(model, ids, cfg)
    assert set(first) == {'FP16', 'Uniform-3bit', 'Uniform-4bit'}
    for key in first:
        assert first[key]['seq_nlls'] == second[key]['seq_nlls']
    for name, p in model.named_parameters():
        torch.testing.assert_close(p, before[name], rtol=0, atol=0)


def write_case(root, dataset, seed, config_hash, baseline_hash):
    base = root/dataset/f'seed_{seed}'; qdir = base/'quality'; qdir.mkdir(parents=True)
    quality = dict(complete=True, dataset=dataset, seed=seed, candidates=[]); estimates = dict(cells=[])
    for k, policy in [(2, 'boundary_inclusive'), (8, 'stratified')]:
        raw = {'p|3': .1}; measured = {'p|3': .1}
        estimates['cells'].append(dict(k_layers=k, policy=policy, measured_labels=measured,
            raw_predictions={m: raw for m in ARMS}))
        timed = dict(complete=True, cell=dict(k_layers=k, policy=policy), trials=[])
        for method in ARMS:
            allocations = {}
            for budget in [3., 3.5]:
                bits = {'p': 3 if budget == 3 else 4}
                quality['candidates'].append(dict(candidate_id=f'{k}-{method}-{budget}', bit_assignment=bits,
                    roles=[dict(method=method, budget=budget, setting='last_down4', k_layers=k, policy=policy,
                        tie_order='depth_ascending' if method.startswith('Family-') else 'native')], evaluation=dict(ppl=10., nll=math.log(10.))))
                allocations[f'{budget:.2f}'] = dict(bit_assignment=bits, calibration_to_budget_sec=100.*k)
            timed['trials'].append(dict(method=method, measured_labels=measured, raw_predictions=raw, allocations=allocations))
        directory = base/f'timing_k{k}_{policy}'; directory.mkdir()
        save(directory/'timing.json', timed)
    save(qdir/'diagnostic.json', quality); save(qdir/'estimates.json', estimates)
    save(qdir/'selection_plan.json', {}); save(qdir/'label_bank.json', {})
    (qdir/'feature_bank.pt').write_bytes(b'fixture')
    save(qdir/'case_manifest.json', dict(complete=True, parent_config_sha256=config_hash, baseline_sha256=baseline_hash,
        selection_plan_sha256=digest(qdir/'selection_plan.json'), label_bank_sha256=digest(qdir/'label_bank.json'),
        feature_bank_sha256=digest(qdir/'feature_bank.pt')))
    for k, policy in [(2, 'boundary_inclusive'), (8, 'stratified')]:
        directory = base/f'timing_k{k}_{policy}'
        save(directory/'case_manifest.json', dict(complete=True, parent_config_sha256=config_hash,
            timing_sha256=digest(directory/'timing.json'), quality_manifest_sha256=digest(qdir/'case_manifest.json')))


def test_combined_summary_is_model_and_dataset_separated_and_checks_parity(tmp_path):
    save(tmp_path/'experiment_config.json', spec_for()); save(tmp_path/'model_config.json', {})
    manifest = dict(schema='focused_1b_run_v1', complete=True, plan_only=False,
        config_sha256=digest(tmp_path/'experiment_config.json'), model_config_sha256=digest(tmp_path/'model_config.json'),
        datasets=['wikitext2', 'c4'], selected_seeds=[42], quality_only=False, baseline_sha256={}, data_manifest_sha256={})
    for name in manifest['datasets']:
        path = tmp_path/name; path.mkdir()
        save(path/'fresh_uniform_baselines.json', dict(dataset=name)); save(path/'data_manifest.json', dict(dataset=name))
        manifest['baseline_sha256'][name] = digest(path/'fresh_uniform_baselines.json')
        manifest['data_manifest_sha256'][name] = digest(path/'data_manifest.json')
        write_case(tmp_path, name, 42, manifest['config_sha256'], manifest['baseline_sha256'][name])
    save(tmp_path/'run_manifest.json', manifest)
    assert aggregate(tmp_path) == dict(quality_cases=2, timing_cases=4, assignment_checks=40)
    validation = json.loads((tmp_path/'summary/validation.json').read_text())
    assert not validation['complete_three_seed_matrix'] and not validation['historical_batch8_scores_reused']
    timing_path = tmp_path/'c4/seed_42/timing_k2_boundary_inclusive/timing.json'
    timed = json.loads(timing_path.read_text()); timed['trials'][0]['allocations']['3.00']['bit_assignment']['p'] = 2
    save(timing_path, timed)
    mpath = timing_path.parent/'case_manifest.json'; m = json.loads(mpath.read_text()); m['timing_sha256'] = digest(timing_path); save(mpath, m)
    with pytest.raises(ValueError, match='differs from the quality'):
        aggregate(tmp_path)
