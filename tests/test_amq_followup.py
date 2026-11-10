import copy
import hashlib
import json
from pathlib import Path

import pytest

from hds_quant.amq_followup_plan import (METHODS, architecture, build_plan,
    fixed4_intervention, geometry, load_source, payload)
from hds_quant.config import ExperimentConfig


def trial_for(meta, bits):
    total=sum(m['num_params'] for m in meta)
    raw={f"{m['name']}|{b}":(4-b)*m['num_params']/total for m in meta for b in (1,2,3)}
    acquired=[0]
    measured={k:v for k,v in raw.items() if next(m['layer_idx'] for m in meta if k.rsplit('|',1)[0]==m['name']) in acquired}
    objective=sum((4-b)*next(m['num_params'] for m in meta if m['name']==n)/total for n,b in bits.items())
    return dict(method='HDS-Hybrid',acquired_layers=acquired,measured_labels=measured,
        heldout_predictions={k:v for k,v in raw.items() if k not in measured},
        allocations={'3.00':dict(success=True,bit_assignment=bits,total_predicted_cost=objective)})


def test_fixed4_preserves_cost_bank_and_exact_payload():
    meta=[dict(name=f'p{i}',num_params=10,layer_idx=0) for i in range(4)]
    trial=trial_for(meta,{m['name']:3 for m in meta});before=copy.deepcopy(trial)
    cfg=ExperimentConfig(candidate_bits=(1,2,3,4))
    result=fixed4_intervention(trial,'3.00',meta,cfg,['p0'])
    assert result['bit_assignment']['p0']==4
    assert payload(result['bit_assignment'],meta)==120
    assert len(result['changed_projections'])>=2
    assert trial==before
    assert result['constrained_objective']==pytest.approx(result['original_objective'])


def test_already_fixed_is_not_resolved(monkeypatch):
    meta=[dict(name=f'p{i}',num_params=10,layer_idx=0) for i in range(4)]
    bits=dict(p0=4,p1=2,p2=3,p3=3);trial=trial_for(meta,bits)
    def fail(*a,**kw): raise AssertionError('unchanged control must not be solved')
    monkeypatch.setattr('hds_quant.allocator.solve_bit_allocation_ilp',fail)
    result=fixed4_intervention(trial,'3.00',meta,ExperimentConfig(candidate_bits=(1,2,3,4)),['p0'])
    assert result['bit_assignment']==bits and result['changed_projections']==[]


def test_infeasible_integer_payload_and_corrupt_cost_rejected():
    meta=[dict(name='p0',num_params=2,layer_idx=0),dict(name='p1',num_params=3,layer_idx=0)]
    trial=trial_for(meta,dict(p0=3,p1=3));cfg=ExperimentConfig(candidate_bits=(1,2,3,4))
    with pytest.raises(ValueError,match='integer payload'):
        fixed4_intervention(trial,'3.00',meta,cfg,['p0'])
    trial['allocations']['3.00']['total_predicted_cost']+=1
    with pytest.raises(ValueError,match='objective'):
        fixed4_intervention(trial,'3.00',meta,cfg,['p0'])


@pytest.fixture
def source(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/amq_comparison/llama31_8b_wikitext_budgetcap_20round_v2.json').read_text())
    meta=geometry();names=[m['name'] for m in meta]
    from hds_quant.amq_coverage import exact_anchor
    from hds_quant.amq_vendor.helpers import get_bits_usage
    families={m['name'].split('.',3)[3]:[m['d_out'],m['d_in']] for m in meta}
    config=dict(n_block=32,linear=list(families),linear_shape=families,
                model_numel=sum(m['num_params'] for m in meta))
    anchors=[exact_anchor(config,['31.mlp.down_proj'],b,42) for b in [3,3.5]]
    tobits=lambda a:{f'model.layers.{i}.{f}':b for f,bs in a['linear'].items() for i,b in enumerate(bs)}
    def alloc(bits):
        total=config['model_numel'];objective=sum((4-b)*next(m['num_params'] for m in meta if m['name']==n)/total for n,b in bits.items())
        return dict(success=True,bit_assignment=bits,achieved_avg_bits=payload(bits,meta)/total,
                    total_predicted_cost=objective,evaluation=dict(nll=2,ppl=7.389,seq_nlls=[2],seq_token_counts=[2047]))
    trials=[]
    for method in METHODS:
        t=trial_for(meta,dict.fromkeys(names,3));t.update(seed=42,method=method,k_layers=4,probe_label_count=84,acquired_layers=list(range(4)))
        raw={**t['measured_labels'],**t['heldout_predictions']}
        t['measured_labels']={k:v for k,v in raw.items() if int(k.split('.')[2])<4}
        t['heldout_predictions']={k:v for k,v in raw.items() if int(k.split('.')[2])>=4}
        t['allocations']={'3.00':alloc(dict.fromkeys(names,3)),'3.50':alloc(tobits(anchors[1]))}
        trials.append(t)
    archive=[[arch,jsd,get_bits_usage(arch,config,128)] for arch,jsd in zip(anchors,[.1,.05])]
    allocations={f'{b:.2f}':dict(alloc(tobits(arch)),calibration_jsd=jsd) for b,arch,jsd in zip([3,3.5],anchors,[.1,.05])}
    trials.append(dict(method='AMQ-HQQ-BudgetCap-v2',seed=42,configuration_evaluations=2,
        fixed_4bit=['31.mlp.down_proj'],missing_budgets=[],initialization_anchors=anchors,allocations=allocations))
    cfg=ExperimentConfig(model_name=spec['model_id'],num_layers=32,quantizer='hqq',hqq_axis=0,candidate_bits=(1,2,3,4))
    raw=dict(complete=True,config=cfg.to_dict(),trials=trials,
             uniform_baselines={f'Uniform-{b}bit':alloc(dict.fromkeys(names,b))['evaluation'] for b in [3,4]})
    for name in ['experiment_config.json','resolved_experiment_config.json']:
        (tmp_path/name).write_text(json.dumps(spec))
    manifest=dict(complete=True,config_sha256=hashlib.sha256((tmp_path/'experiment_config.json').read_bytes()).hexdigest(),
        resolved_model_revision=spec['model_revision'],resolved_dataset_revision=spec['dataset']['revision'])
    (tmp_path/'experiment_manifest.json').write_text(json.dumps(manifest))
    (tmp_path/'limited_labels.json').write_text(json.dumps(raw))
    (tmp_path/'data_manifest.json').write_text('{}')
    (tmp_path/'amq_seed_42').mkdir()
    (tmp_path/'amq_seed_42/archive.json').write_text(json.dumps(dict(complete=True,iteration=20,seed=42,
        budget_policy='cap_v2',fixed_4bit=['31.mlp.down_proj'],archive=archive)))
    return tmp_path


def test_frozen_plans_deduplicate_and_preserve_sources(source):
    root=Path(__file__).resolve().parents[1]
    before={p:p.read_bytes() for p in source.rglob('*.json')}
    for mode in ['objectives','fixed4']:
        spec=json.loads((root/f'configs/amq_diagnostics/llama31_8b_wikitext_{mode}_v1.json').read_text())
        plan,_=build_plan(source,spec)
        assert all('evaluation' not in role for r in plan['candidates'] for role in r['roles'])
        assert len({json.dumps(r['architecture'],sort_keys=True) for r in plan['candidates']})==len(plan['candidates'])
        assert len(plan['comparisons'])==(6 if mode=='fixed4' else 0)
        if mode=='fixed4':
            assert all(not r['changed_projections'] for r in plan['comparisons'] if r['budget']==3.5)
    assert before=={p:p.read_bytes() for p in before}
    path=source/'amq_seed_42/archive.json';bad=json.loads(path.read_text());bad['archive'][0][2]+=.01
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='Invalid archive'):
        load_source(source,42)


def test_real_small_model_objective_scoring_and_replay_failure(tmp_path):
    import torch
    from hds_quant.run_experiment import _build_smoke_model
    from hds_quant.label_collector import discover_target_projections
    from hds_quant.evaluator import evaluate_perplexity
    from hds_quant.run_amq_followup import evaluate_plan
    torch.set_num_threads(1)
    model,ids=_build_smoke_model();ids=ids[:1,:8]
    cfg=ExperimentConfig(num_layers=model.config.num_hidden_layers,quantizer='hqq',hqq_axis=0,quant_group_size=64,device='cpu')
    meta=discover_target_projections(model)
    bits={m['name']:3 for m in meta}
    expected=evaluate_perplexity(model,ids,meta,cfg,bit_assignment=bits,batch_size=1)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    plan=dict(seed=42,candidates=[dict(candidate_id='test',architecture=architecture(bits,cfg.num_layers),
        payload_bits=3,archived_jsd=None,roles=[],historical_evaluations=[expected])])
    for directory,corrupt in [('valid',False),('bad_replay',True)]:
        out=tmp_path/directory;out.mkdir()
        if corrupt: plan['candidates'][0]['historical_evaluations'][0]['nll']+=.1
        result=evaluate_plan(model,ids,ids,cfg,plan,out)
        assert result['complete'] is not corrupt
        assert result['candidates'][0]['parity_passed']
        assert not (out/'amq_seed_42/teacher_logits.bin').exists()
        for n,p in model.named_parameters(): torch.testing.assert_close(p,before[n],rtol=0,atol=0)


def test_uniform_plan_freezes_all_observed_gates_and_checks_reference(source, tmp_path):
    from hds_quant.run_uniform_swap import build_plan as swap_plan, uniform_swap
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/amq_diagnostics/llama31_8b_wikitext_uniform_swap_v1.json').read_text())
    _,raw,_,hashes,_=load_source(source,42)
    reference=tmp_path/'fixed4';reference.mkdir()
    rows=[];pairs=[];meta=geometry()
    for i,method in enumerate(METHODS):
        original=next(t for t in raw['trials'] if t['method']==method)['allocations']['3.00']
        before=dict(candidate_id=f'b{i}',architecture=architecture(original['bit_assignment']),
                    evaluation=original['evaluation'],parity_passed=True)
        bits=uniform_swap(meta,'model.layers.31.mlp.down_proj',f"model.layers.{spec['gate_layers'][method]}.mlp.gate_proj")
        after=dict(candidate_id=f'a{i}',architecture=architecture(bits),evaluation=original['evaluation'],parity_passed=True)
        rows.extend([before,after]);pairs.append(dict(method=method,budget=3.0,before_id=f'b{i}',after_id=f'a{i}'))
    for name,value in [
        ('diagnostic_manifest.json',dict(complete=True,mode='fixed4',source_hashes=hashes)),
        ('diagnostic.json',dict(complete=True,mode='fixed4',candidates=rows)),
        ('selection_plan.json',dict(seed=42,source_hashes=hashes,candidates=rows,comparisons=pairs))]:
        (reference/name).write_text(json.dumps(value))
    before=copy.deepcopy(raw)
    plan,actual=swap_plan(spec,source,reference)
    assert len(plan['candidates'])==3 and actual==before
    assert {r['gate_projection'] for r in plan['candidates']}=={f'model.layers.{i}.mlp.gate_proj' for i in [28,14,1]}
    assert all(r['actual_payload_bits']==3 for r in plan['candidates'])
    spec['gate_layers']['HDS-Hybrid']=0
    with pytest.raises(ValueError,match='all three'):
        swap_plan(spec,source,reference)
