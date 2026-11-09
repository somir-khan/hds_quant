import json
from pathlib import Path

import numpy as np
import pytest
import torch

from hds_quant.amq_baseline import jsd_tokens, select_candidate, run_amq_trial
from hds_quant.run_amq_comparison import validate_spec
from hds_quant.run_experiment import _build_smoke_model
from hds_quant.config import ExperimentConfig

ROOT = Path(__file__).resolve().parents[1]


def test_budget_selection_never_overspends_or_uses_test_ppl():
    archive = [('over', 0.0, 3.251), ('under', 0.1, 3.246), ('exact', 0.2, 3.25)]
    assert select_candidate(archive, 3, 128)[0] == 'under'
    assert select_candidate(archive, 3.5, 128) is None


def test_jsd_matches_upstream_formula_and_chunking():
    torch.manual_seed(7)
    p, q = torch.randn(17, 23), torch.randn(17, 23)
    m = (0.5*(p.softmax(-1)+q.softmax(-1))).clamp_min(1e-7).log()
    kl = torch.nn.KLDivLoss(reduction='batchmean', log_target=True)
    expected = 0.5*(kl(m,p.log_softmax(-1))+kl(m,q.log_softmax(-1)))*len(p)
    torch.testing.assert_close(jsd_tokens(p,q),expected)
    torch.testing.assert_close(jsd_tokens(p,q),jsd_tokens(p[:8],q[:8])+jsd_tokens(p[8:],q[8:]))


@pytest.mark.parametrize('path', sorted((ROOT/'configs/amq_comparison').glob('*.json')))
def test_configs(path):
    spec = json.loads(path.read_text())
    validate_spec(spec)
    spec['amq_search']['initial_samples'] = 10
    with pytest.raises(ValueError):
        validate_spec(spec)


@pytest.mark.parametrize('policy', ['window','cap_v2'])
def test_real_search_and_hqq_restore_model(tmp_path, policy):
    torch.set_num_threads(1)
    model, ids = _build_smoke_model()
    before = {n: p.detach().clone() for n,p in model.named_parameters()}
    cfg = ExperimentConfig(num_layers=model.config.num_hidden_layers, quantizer='hqq',
                           quant_group_size=64, hqq_axis=0, device='cpu')
    calib = ids[:1,:16]
    result = run_amq_trial(model, calib, calib, cfg, [3.0,3.5], 42, 1,
        settings={'initial_samples':40,'iterations':1,'candidates_per_iteration':2,'population':20},
        output=tmp_path, budget_policy=policy)
    assert result['configuration_evaluations'] >= 40
    assert result['probe_label_count'] == 7*cfg.num_layers
    assert len(result['allocations'])+len(result['missing_budgets']) == 2
    for n,p in model.named_parameters():
        torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    checkpoint = json.loads((tmp_path/'amq_seed_42/archive.json').read_text())
    assert checkpoint['complete']
    assert not (tmp_path/'amq_seed_42/teacher_logits.bin').exists()
    if policy == 'cap_v2':
        assert result['missing_budgets'] == []
        assert len(result['initialization_anchors']) == 2
        assert all(a['achieved_avg_bits'] <= float(b)+1e-10 for b,a in result['allocations'].items())
    # Archive-only replay checks teacher/JSD and cache-vs-fresh calibration NLL.
    selected = [dict(archive_index=2, reasons=['test_all4'], architecture=checkpoint['archive'][2][0],
                     archived_jsd=checkpoint['archive'][2][1],payload_bits=4)]
    replay = tmp_path/'replay'
    replay.mkdir()
    diagnostic = run_amq_trial(model,calib,calib,cfg,[3.0,3.5],42,1,
                              settings={},output=replay,diagnostic=selected)
    assert diagnostic['complete'] and diagnostic['candidates'][0]['parity_passed']
    for n,p in model.named_parameters():
        torch.testing.assert_close(p,before[n],rtol=0,atol=0)


def test_scoring_failure_restores_original_weights(tmp_path, monkeypatch):
    import hds_quant.amq_baseline as baseline
    torch.set_num_threads(1)
    model, ids = _build_smoke_model()
    before = {n: p.detach().clone() for n,p in model.named_parameters()}
    cfg = ExperimentConfig(num_layers=4, quantizer='hqq', quant_group_size=64,
                           hqq_axis=0, device='cpu')
    def fail(*args):
        raise RuntimeError('injected scoring failure')
    monkeypatch.setattr(baseline, 'jsd_tokens', fail)
    with pytest.raises(RuntimeError, match='injected'):
        run_amq_trial(model, ids[:1,:8], ids[:1,:8], cfg, [3.0], 42, 1,
            settings={'initial_samples':40,'iterations':1,'candidates_per_iteration':2,'population':20},
            output=tmp_path)
    for n,p in model.named_parameters():
        torch.testing.assert_close(p,before[n],rtol=0,atol=0)


def test_summary_reports_missing_budget_without_fabricating_ppl(tmp_path):
    from scripts.summarize_amq_comparison import summarize
    spec = {'schema':'amq_comparison_v1','seeds':[42], 'methods':[], 'budgets':[3.0]}
    trial = {'seed':42, 'method':'AMQ-HQQ-Matched','allocations':{},
             'missing_budgets':[3.0], 'calibration_wall_sec':2.0}
    raw = {'complete':True,'trials':[trial],'evaluation_chunks':1}
    for name, value in [('limited_labels.json',raw), ('experiment_config.json',spec),
                         ('experiment_manifest.json',{'complete':True})]:
        (tmp_path/name).write_text(json.dumps(value))
    rows = summarize(tmp_path)
    assert rows[0]['status'] == 'no_candidate_in_budget_window'
    assert rows[0]['ppl'] == ''
    raw['trials'].append(trial)
    (tmp_path/'limited_labels.json').write_text(json.dumps(raw))
    with pytest.raises(ValueError, match='duplicate'):
        summarize(tmp_path)
