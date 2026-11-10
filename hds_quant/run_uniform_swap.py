"""Frozen Uniform-3 swap controls paired with the saved seed42 fixed4 diagnostic."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from .amq_followup_plan import assignment, digest, geometry, load_source, payload, METHODS
from .run_amq_followup import save


def uniform_swap(meta, down, gate):
    bits={m['name']:3 for m in meta}
    if down==gate or down not in bits or gate not in bits:
        raise ValueError('Invalid swap projections')
    original=payload(bits,meta)
    bits[down]=4; bits[gate]=2
    if payload(bits,meta)!=original:
        raise ValueError('Swap does not preserve exact integer payload')
    return bits


def validate_spec(spec):
    if spec.get('schema')!='uniform_swap_control_v1' or spec.get('seed')!=42:
        raise ValueError('Expected frozen seed42 uniform-swap control')
    if spec.get('gate_layers')!=dict(zip(METHODS,[28,14,1])):
        raise ValueError('Evaluate all three previously observed gate swaps without test selection')


def build_plan(spec, source, fixed4_source):
    validate_spec(spec)
    src,raw,_,hashes,fixed=load_source(source,42)
    down='model.layers.31.mlp.down_proj'
    if fixed!=[down]: raise ValueError('Expected the observed single fixed4 projection')
    fixed4_source=Path(fixed4_source)
    names=['diagnostic_manifest.json','diagnostic.json','selection_plan.json']
    fixed_hashes={n:digest(fixed4_source/n) for n in names}
    manifest,result,oldplan=[json.loads((fixed4_source/n).read_text()) for n in names]
    if not manifest.get('complete') or not result.get('complete') or manifest.get('mode')!='fixed4' or result.get('mode')!='fixed4':
        raise ValueError('Expected completed fixed4 diagnostic')
    if manifest['source_hashes']!=hashes or oldplan['source_hashes']!=hashes or oldplan['seed']!=42:
        raise ValueError('Fixed4 diagnostic refers to a different source')
    byid={r['candidate_id']:r for r in result['candidates']}
    planned={r['candidate_id']:r for r in oldplan['candidates']}
    if len(byid)!=len(result['candidates']): raise ValueError('Duplicate diagnostic candidate')
    candidates=[];meta=geometry();total=sum(m['num_params'] for m in meta)
    from .run_no1bit_diagnostic import check_evaluation
    for method in METHODS:
        pairs=[p for p in oldplan['comparisons'] if p['method']==method and p['budget']==3.0]
        if len(pairs)!=1: raise ValueError('Missing/duplicate fixed4 comparison')
        pair=pairs[0];before,after=[byid[pair[k]] for k in ['before_id','after_id']]
        for r in [before,after]:
            if not r['parity_passed'] or r['architecture']!=planned[r['candidate_id']]['architecture']:
                raise ValueError('Fixed4 diagnostic assignment/parity mismatch')
        original=next(t for t in raw['trials'] if t['method']==method and t['seed']==42)['allocations']['3.00']
        if assignment(before['architecture'])!=original['bit_assignment']:
            raise ValueError('Original learned assignment differs from source')
        check_evaluation(before['evaluation'],original['evaluation'],1e-6)
        gate=f"model.layers.{spec['gate_layers'][method]}.mlp.gate_proj"
        b,a=assignment(before['architecture']),assignment(after['architecture'])
        changes={n:(b[n],a[n]) for n in b if b[n]!=a[n]}
        if changes!={down:(3,4),gate:(3,2)} or payload(b,meta)!=payload(a,meta):
            raise ValueError('Saved intervention differs from the frozen two-projection swap')
        bits=uniform_swap(meta,down,gate)
        candidates.append(dict(candidate_id=f"uniform3_gate{spec['gate_layers'][method]}",
            paired_method=method,down_projection=down,gate_projection=gate,
            bit_assignment=bits,actual_payload_bits=payload(bits,meta)/total,
            learned_original_evaluation=before['evaluation'],learned_fixed4_evaluation=after['evaluation']))
    return dict(schema='uniform_swap_plan_v1',seed=42,source_spec=src,
        source_hashes=hashes,fixed4_source_hashes=fixed_hashes,candidates=candidates,
        selection_rule='All three observed gate choices frozen; no new gate search or test-set winner selection'),raw


def evaluate_controls(model, ids, cfg, plan, raw, output):
    from .label_collector import discover_target_projections
    from .evaluator import evaluate_perplexity
    from .run_no1bit_diagnostic import check_evaluation
    meta=discover_target_projections(model)
    report=dict(schema='uniform_swap_result_v1',complete=False,baseline_checks={},candidates=[])
    def persist(): save(output/'diagnostic.json',report)
    persist()
    for name,bit in [('FP16',None),('Uniform-3bit',3)]:
        ev=evaluate_perplexity(model,ids,meta,cfg,uniform_bit=bit,batch_size=1)
        report['baseline_checks'][name]=dict(evaluation=ev,**check_evaluation(ev,raw['uniform_baselines'][name],1e-6))
        persist()
    for candidate in plan['candidates']:
        print(f"Evaluating {candidate['candidate_id']}",flush=True)
        ev=evaluate_perplexity(model,ids,meta,cfg,bit_assignment=candidate['bit_assignment'],batch_size=1)
        row=dict(candidate,evaluation=ev,
            swap_vs_uniform_ppl_percent=100*(ev['ppl']/raw['uniform_baselines']['Uniform-3bit']['ppl']-1),
            learned_fixed4_vs_swap_ppl_percent=100*(candidate['learned_fixed4_evaluation']['ppl']/ev['ppl']-1))
        report['candidates'].append(row);persist()
    report['complete']=True;persist()
    return report


def summarize(report, output):
    rows=[dict(candidate_id=r['candidate_id'],paired_method=r['paired_method'],
        actual_bits=r['actual_payload_bits'],uniform_ppl=report['baseline_checks']['Uniform-3bit']['evaluation']['ppl'],
        uniform_swap_ppl=r['evaluation']['ppl'],learned_original_ppl=r['learned_original_evaluation']['ppl'],
        learned_fixed4_ppl=r['learned_fixed4_evaluation']['ppl'],
        swap_vs_uniform_ppl_percent=r['swap_vs_uniform_ppl_percent'],
        learned_fixed4_vs_swap_ppl_percent=r['learned_fixed4_vs_swap_ppl_percent']) for r in report['candidates']]
    with (output/'comparisons.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['Post-hoc Uniform-3 matched-swap control; all three gate choices evaluated.',
        'Gate locations come from previous learned allocations; this is not an independently designed baseline.',
        'Negative learned_fixed4_vs_swap_ppl_percent favors the remaining learned allocation.',
        'No best-gate selection, new sensitivity labels, or calibration-speed claim.']
    lines += [str(r) for r in rows]
    (output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(config, output, source_override=None, fixed4_override=None, plan_only=False):
    config=Path(config);spec=json.loads(config.read_text())
    source=Path(source_override or spec['source_dir']).resolve()
    fixed4=Path(fixed4_override or spec['fixed4_source_dir']).resolve()
    plan,raw=build_plan(spec,source,fixed4)
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[1]
    code=list(Path(__file__).parent.glob('*.py'))+[project/'scripts/run_uniform_swap_a100.sh',project/'scripts/load_hf_env.sh']
    manifest=dict(schema='uniform_swap_control_v1',complete=False,plan_only=plan_only,plan_complete=True,
        source_directory=str(source),fixed4_source_directory=str(fixed4),
        source_hashes=plan['source_hashes'],fixed4_source_hashes=plan['fixed4_source_hashes'],
        code_sha256={str(p.relative_to(project)):digest(p) for p in code},config_sha256=digest(config))
    (output/'experiment_config.json').write_bytes(config.read_bytes());save(output/'selection_plan.json',plan)
    path=output/'diagnostic_manifest.json';save(path,manifest)
    if plan_only:return plan
    try:
        import torch
        from transformers import AutoModelForCausalLM,AutoTokenizer
        from .config import ExperimentConfig
        from .generalization_data import prepare_dataset,sha_tensor
        from .label_collector import discover_target_projections
        from .quantizer import get_effective_hqq_config
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; --plan-only is offline')
        manifest['packages']={n:importlib.metadata.version(n) for n in ['torch','transformers','hqq','datasets','scipy']}
        save(path,manifest)
        src=plan['source_spec'];tokenizer=AutoTokenizer.from_pretrained(src['model_id'],revision=src['model_revision'])
        args=SimpleNamespace(eval_chunks=src['dataset']['eval_chunks'],eval_seq_len=src['eval_seq_len'],
            calib_seq_len=src['calibration']['seq_len'],n_calib_seqs=src['calibration']['n_sequences'],seeds=[42])
        data=prepare_dataset(tokenizer,args,src['dataset'],src['dataset']['revision'])
        if sha_tensor(data['eval_ids'])!=raw['evaluation_token_sha256'] or sha_tensor(data['calib_by_seed'][42])!=raw['calibration_token_sha256']['42']:
            raise ValueError('Source token hashes differ')
        save(output/'data_manifest.json',data['manifest'])
        model=AutoModelForCausalLM.from_pretrained(src['model_id'],revision=src['model_revision'],torch_dtype=torch.float16).to('cuda').eval()
        cfg=ExperimentConfig(**raw['config']);meta=discover_target_projections(model)
        if {m['name']:m['num_params'] for m in meta}!={m['name']:m['num_params'] for m in geometry()}:
            raise ValueError('Loaded projection geometry differs')
        if get_effective_hqq_config(128,'cuda',torch.float16,axis=0,zero_policy='floating')!=raw['hqq_config']:
            raise ValueError('Effective HQQ settings differ')
        report=evaluate_controls(model,data['eval_ids'],cfg,plan,raw,output)
        for folder,hashes in [(source,plan['source_hashes']),(fixed4,plan['fixed4_source_hashes'])]:
            if any(digest(folder/n)!=h for n,h in hashes.items()):raise RuntimeError('Source changed during evaluation')
        summarize(report,output);manifest['complete']=True;save(path,manifest)
        print(f'Uniform-swap control completed: {output}',flush=True)
        return report
    except Exception as error:
        manifest.update(complete=False,error=f'{type(error).__name__}: {error}');save(path,manifest);raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output-dir',required=True)
    p.add_argument('--source-dir');p.add_argument('--fixed4-source-dir');p.add_argument('--plan-only',action='store_true')
    a=p.parse_args();run(a.config,a.output_dir,a.source_dir,a.fixed4_source_dir,a.plan_only)
