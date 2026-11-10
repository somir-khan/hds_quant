import copy

import pytest

from hds_quant.run_uniform_swap import uniform_swap, evaluate_controls, summarize


def test_swap_is_exact_and_rejects_unequal_weight_sizes():
    meta=[dict(name='down',num_params=10),dict(name='gate',num_params=10),dict(name='other',num_params=2)]
    assert uniform_swap(meta,'down','gate')==dict(down=4,gate=2,other=3)
    bad=copy.deepcopy(meta);bad[1]['num_params']=9
    with pytest.raises(ValueError,match='exact integer payload'):
        uniform_swap(bad,'down','gate')


def test_real_evaluation_replays_baselines_and_restores_weights(tmp_path):
    import torch
    from hds_quant.run_experiment import _build_smoke_model
    from hds_quant.config import ExperimentConfig
    from hds_quant.label_collector import discover_target_projections
    from hds_quant.evaluator import evaluate_perplexity
    torch.set_num_threads(1)
    model,ids=_build_smoke_model();ids=ids[:1,:8]
    cfg=ExperimentConfig(num_layers=model.config.num_hidden_layers,quantizer='hqq',hqq_axis=0,
                         quant_group_size=64,device='cpu')
    meta=discover_target_projections(model)
    baseline={n:evaluate_perplexity(model,ids,meta,cfg,uniform_bit=b,batch_size=1)
              for n,b in [('FP16',None),('Uniform-3bit',3)]}
    bits=uniform_swap(meta,'model.layers.3.mlp.down_proj','model.layers.1.mlp.gate_proj')
    candidate=dict(candidate_id='test',paired_method='HDS-Hybrid',bit_assignment=bits,actual_payload_bits=3,
                   learned_original_evaluation=baseline['Uniform-3bit'],learned_fixed4_evaluation=baseline['Uniform-3bit'])
    plan=dict(candidates=[candidate]);raw=dict(uniform_baselines=baseline)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    result=evaluate_controls(model,ids,cfg,plan,raw,tmp_path)
    assert result['complete'] and len(result['candidates'])==1
    summarize(result,tmp_path)
    assert (tmp_path/'comparisons.csv').exists()
    for n,p in model.named_parameters():torch.testing.assert_close(p,before[n],rtol=0,atol=0)
    bad=tmp_path/'bad';bad.mkdir();raw=copy.deepcopy(raw);raw['uniform_baselines']['Uniform-3bit']['nll']+=.1
    with pytest.raises(ValueError,match='replay failed'):
        evaluate_controls(model,ids,cfg,plan,raw,bad)
    import json
    assert not json.loads((bad/'diagnostic.json').read_text())['complete']
