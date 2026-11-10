"""Independent objective/constraint diagnostics; no screening, fitting, or AMQ search."""
import argparse
import csv
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from .amq_followup_plan import build_plan, digest, geometry, validate_spec


def save(path, value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def evaluate_plan(model, calib, eval_ids, cfg, plan, output):
    from .amq_baseline import run_amq_trial
    from .run_no1bit_diagnostic import check_evaluation
    result=run_amq_trial(model,calib,eval_ids,cfg,[3,3.5],plan['seed'],1,
        settings={},output=output,diagnostic=plan['candidates'])
    for row in result['candidates']:
        row['historical_replay_checks']=[]
        for expected in row['historical_evaluations']:
            try:
                check=check_evaluation(row['evaluation'],expected,1e-6)
            except ValueError as error:
                check=dict(passed=False,error=str(error))
            row['historical_replay_checks'].append(check)
    result['complete']=all(row['parity_passed'] and all(x['passed'] for x in row['historical_replay_checks'])
                           for row in result['candidates'])
    return result


def summaries(plan, result, output):
    rows=[]
    by_id={row['candidate_id']:row for row in result['candidates']}
    for row in result['candidates']:
        rows.append(dict(candidate_id=row['candidate_id'],roles=json.dumps(row['roles']),
            actual_payload_bits=row['payload_bits'],satisfies_amq_fixed4=row['satisfies_amq_fixed4'],
            calibration_jsd=row['recomputed_jsd'],calibration_nll=row['fresh_calibration']['nll'],
            calibration_ppl=row['fresh_calibration']['ppl'],heldout_nll=row['evaluation']['nll'],
            heldout_ppl=row['evaluation']['ppl'],parity_passed=row['parity_passed']))
    def write_csv(name, values):
        if not values: return
        with (output/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]))
            writer.writeheader(); writer.writerows(values)
    write_csv('objective_scores.csv',rows)
    comparisons=[]
    for pair in plan['comparisons']:
        before,after=by_id[pair['before_id']],by_id[pair['after_id']]
        comparisons.append(dict(method=pair['method'],budget=pair['budget'],
            before_payload=before['payload_bits'],after_payload=after['payload_bits'],
            changed_projection_count=len(pair['changed_projections']),
            original_predicted_cost=pair['original_objective'],fixed4_predicted_cost=pair['constrained_objective'],
            before_ppl=before['evaluation']['ppl'],after_ppl=after['evaluation']['ppl'],
            ppl_change_percent=100*(after['evaluation']['ppl']/before['evaluation']['ppl']-1),
            calibration_jsd_change=after['recomputed_jsd']-before['recomputed_jsd'],
            calibration_nll_change=after['fresh_calibration']['nll']-before['fresh_calibration']['nll'],
            heldout_nll_change=after['evaluation']['nll']-before['evaluation']['nll']))
    write_csv('constraint_comparisons.csv',comparisons)
    save(output/'comparisons.json',comparisons)
    lines=['AMQ '+plan['mode']+' diagnostic; post-hoc analysis, not a new benchmark or speed result.',
        'Assignment list frozen before GPU evaluation; lower JSD/NLL/PPL is better.',
        'Payload covers target projection weights only. JSD uses the existing epsilon-clamped AMQ formula.',
        'Fixed4 comparison changes all required compensating assignments; it does not isolate one matrix.',
        '', 'Candidate | actual bits | calibration JSD | calibration NLL | held-out PPL | satisfies AMQ fixed4']
    for row in rows:
        lines.append(f"{row['candidate_id']} | {row['actual_payload_bits']:.6f} | {row['calibration_jsd']:.8f} | {row['calibration_nll']:.6f} | {row['heldout_ppl']:.6f} | {row['satisfies_amq_fixed4']}")
        lines.append('  '+row['roles'])
    if comparisons:
        lines+=['','Fixed4 effects: after minus before; negative PPL change favors the constrained allocation.']
        for row in comparisons:
            lines.append(f"{row['method']} budget={row['budget']}: {row['before_ppl']:.6f} -> {row['after_ppl']:.6f} ({row['ppl_change_percent']:+.3f}%), changed={row['changed_projection_count']}")
    (output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run(config, output, source_override=None, plan_only=False):
    config=Path(config)
    spec=json.loads(config.read_text()); validate_spec(spec)
    source=Path(source_override or spec['source_dir']).resolve()
    plan,raw=build_plan(source,spec)
    output=Path(output); output.mkdir(parents=True,exist_ok=False)
    project=Path(__file__).resolve().parents[1]
    sources=list(Path(__file__).parent.glob('*.py'))+[config,
        project/'scripts/launch_amq_followup.sh',project/'scripts/load_hf_env.sh',
        project/'scripts/run_amq_objective_diagnostic_a100.sh',project/'scripts/run_amq_fixed4_diagnostic_a100.sh']
    manifest=dict(schema='amq_followup_v1',experiment_id=spec['experiment_id'],mode=spec['mode'],
        complete=False,plan_only=plan_only,plan_complete=True,source_directory=str(source),
        source_hashes=plan['source_hashes'],config_sha256=digest(config),
        code_sha256={str(p.relative_to(project)) if p.is_relative_to(project) else str(p):digest(p) for p in sources},
        scope='Frozen assignment and fixed-cost allocation diagnostics; no new sensitivity labels or search')
    (output/'experiment_config.json').write_bytes(config.read_bytes())
    save(output/'selection_plan.json',plan)
    manifest_path=output/'diagnostic_manifest.json'; save(manifest_path,manifest)
    if plan_only: return plan
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .config import ExperimentConfig
        from .generalization_data import prepare_dataset, sha_tensor
        from .label_collector import discover_target_projections
        from .quantizer import get_effective_hqq_config
        from .evaluator import evaluate_perplexity
        from .run_no1bit_diagnostic import check_evaluation
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required; --plan-only is offline')
        manifest['packages']={p:importlib.metadata.version(p) for p in ('torch','transformers','hqq','numpy','scipy','datasets')}
        save(manifest_path,manifest)
        src=plan['source_spec']
        tokenizer=AutoTokenizer.from_pretrained(src['model_id'],revision=src['model_revision'])
        args=SimpleNamespace(eval_chunks=src['dataset']['eval_chunks'],eval_seq_len=src['eval_seq_len'],
            calib_seq_len=src['calibration']['seq_len'],n_calib_seqs=src['calibration']['n_sequences'],seeds=[spec['seed']])
        data=prepare_dataset(tokenizer,args,src['dataset'],src['dataset']['revision'])
        calib=data['calib_by_seed'][spec['seed']]; eval_ids=data['eval_ids']
        if sha_tensor(calib)!=raw['calibration_token_sha256'][str(spec['seed'])] or sha_tensor(eval_ids)!=raw['evaluation_token_sha256']:
            raise ValueError('Recreated calibration/evaluation token hashes differ')
        save(output/'data_manifest.json',data['manifest'])
        model=AutoModelForCausalLM.from_pretrained(src['model_id'],revision=src['model_revision'],torch_dtype=torch.float16).to('cuda').eval()
        cfg=ExperimentConfig(**raw['config'])
        meta=discover_target_projections(model)
        if {m['name']:m['num_params'] for m in meta}!={m['name']:m['num_params'] for m in geometry()}:
            raise ValueError('Loaded model geometry differs from source')
        hqq=get_effective_hqq_config(128,next(model.parameters()).device,next(model.parameters()).dtype,axis=0,zero_policy='floating')
        if hqq!=raw['hqq_config']: raise ValueError('Effective HQQ configuration differs from source')
        fp16=evaluate_perplexity(model,eval_ids,meta,cfg,batch_size=1)
        manifest['fp16_replay']=check_evaluation(fp16,raw['uniform_baselines']['FP16'],1e-6)
        save(manifest_path,manifest)
        result=evaluate_plan(model,calib,eval_ids,cfg,plan,output)
        result.update(schema='amq_followup_result_v1',mode=spec['mode'],fixed4_projections=plan['fixed4_projections'])
        save(output/'diagnostic.json',result)
        if not result['complete']: raise RuntimeError('Historical evaluation replay failed; inspect diagnostic.json')
        if any(digest(source/name)!=value for name,value in plan['source_hashes'].items()):
            raise RuntimeError('Source files changed during diagnostic')
        summaries(plan,result,output)
        manifest['complete']=True; save(manifest_path,manifest)
        print(f"AMQ {spec['mode']} diagnostic completed: {output}",flush=True)
        return result
    except Exception as error:
        manifest.update(complete=False,error=f'{type(error).__name__}: {error}')
        save(manifest_path,manifest)
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--source-dir'); p.add_argument('--plan-only',action='store_true')
    args=p.parse_args(); run(args.config,args.output_dir,args.source_dir,args.plan_only)
