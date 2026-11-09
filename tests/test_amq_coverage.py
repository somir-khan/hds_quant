import json
from pathlib import Path

import pytest
from hds_quant.amq_coverage import exact_anchor, select_under_cap, diagnostic_selection
from hds_quant.amq_vendor.helpers import get_bits_usage


def geometry():
    families = {'self_attn.q_proj':[4096,4096], 'self_attn.k_proj':[1024,4096],
        'self_attn.v_proj':[1024,4096], 'self_attn.o_proj':[4096,4096],
        'mlp.gate_proj':[14336,4096], 'mlp.up_proj':[14336,4096], 'mlp.down_proj':[4096,14336]}
    return dict(n_block=32,linear=list(families),linear_shape=families,
                model_numel=32*sum(a*b for a,b in families.values()))


@pytest.mark.parametrize('budget',[3,3.5])
def test_actual_8b_anchors_exact_and_protected(budget):
    cfg=geometry()
    a=exact_anchor(cfg,['31.mlp.down_proj'],budget,42)
    assert a['linear']['mlp.down_proj'][31]==4
    assert get_bits_usage(a,cfg,128)-.25 == budget
    assert a==exact_anchor(cfg,['31.mlp.down_proj'],budget,42)


def test_impossible_anchor_fails():
    cfg=geometry()
    with pytest.raises((ValueError,RuntimeError)):
        exact_anchor(cfg,[f'{i}.{f}' for i in range(32) for f in cfg['linear']],3,42)


def test_cap_selects_by_jsd_without_overspending():
    archive=[('near',.4,3.248),('better',.12,3.20),('over',.001,3.251)]
    assert select_under_cap(archive,3,128)[0]=='better'


def test_diagnostic_selection_is_frozen_and_deduplicated():
    cfg=geometry()
    def arch(bit): return {'linear':{f:[bit]*32 for f in cfg['linear']}}
    archive=[(arch(3),.4,3.25),(arch(3),.1,3.20),(arch(4),.01,4.25)]
    rows=diagnostic_selection(archive,[3,3.5],128)
    assert [x['archive_index'] for x in rows]==[0,1,2]
    assert len(rows[1]['reasons'])==2
    assert rows[2]['reasons']==['all4_reference']


def test_v2_summary_accepts_underfill_and_reports_gap(tmp_path):
    from scripts.summarize_amq_comparison import summarize
    allocation=dict(success=True,achieved_avg_bits=2.95,bit_assignment={'p':3},
        calibration_to_budget_sec=1,evaluation=dict(ppl=10,nll=2.302585,seq_nlls=[2.302585]))
    trial=dict(seed=42,method='AMQ-HQQ-BudgetCap-v2',allocations={'3.00':allocation},calibration_wall_sec=1)
    for name,value in [('experiment_config.json',dict(schema='amq_comparison_v2',seeds=[42],methods=[],budgets=[3])),
                       ('limited_labels.json',dict(complete=True,trials=[trial],evaluation_chunks=1)),
                       ('experiment_manifest.json',dict(complete=True))]:
        (tmp_path/name).write_text(json.dumps(value))
    rows=summarize(tmp_path)
    assert rows[0]['payload_gap']==pytest.approx(.05)


def test_archive_loader_rejects_corrupt_payload_and_incomplete_source(tmp_path):
    import hashlib
    from hds_quant.run_amq_diagnostic import load_source
    root=Path(__file__).resolve().parents[1]
    spec=json.loads((root/'configs/amq_comparison/llama31_8b_wikitext_pilot.json').read_text())
    source=json.dumps(spec).encode()
    (tmp_path/'experiment_config.json').write_bytes(source)
    (tmp_path/'resolved_experiment_config.json').write_bytes(source)
    manifest=dict(complete=True,config_sha256=hashlib.sha256(source).hexdigest(),
        resolved_model_revision=spec['model_revision'],resolved_dataset_revision=spec['dataset']['revision'])
    (tmp_path/'experiment_manifest.json').write_text(json.dumps(manifest))
    raw=dict(complete=True,config=dict(quantizer='hqq',hqq_axis=0,hqq_zero_policy='floating',quant_group_size=128),
             trials=[dict(seed=42,method='AMQ-HQQ-Matched',configuration_evaluations=1)])
    (tmp_path/'limited_labels.json').write_text(json.dumps(raw))
    arch={'linear':{f:[4]*32 for f in geometry()['linear']}}
    saved=dict(complete=True,seed=42,archive=[[arch,.02,4.25]],fixed_4bit=[])
    sub=tmp_path/'amq_seed_42'; sub.mkdir()
    (sub/'archive.json').write_text(json.dumps(saved))
    assert load_source(tmp_path,42)[2][0]['reasons']==['all4_reference']
    saved['archive'][0][2]=3.25
    (sub/'archive.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='payload'):
        load_source(tmp_path,42)
    saved['complete']=False
    (sub/'archive.json').write_text(json.dumps(saved))
    with pytest.raises(ValueError,match='complete'):
        load_source(tmp_path,42)
