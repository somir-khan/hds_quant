"""Reevaluate a frozen calibration-selected subset of an existing AMQ archive."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
from types import SimpleNamespace

from .amq_coverage import diagnostic_selection


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_source(source, seed):
    source = Path(source)
    spec = json.loads((source/'resolved_experiment_config.json').read_text())
    raw = json.loads((source/'limited_labels.json').read_text())
    manifest = json.loads((source/'experiment_manifest.json').read_text())
    archive_path = source/f'amq_seed_{seed}/archive.json'
    saved = json.loads(archive_path.read_text())
    if not all(x.get('complete') for x in (raw,manifest,saved)):
        raise ValueError('Source job and archive must be complete')
    if spec['schema'] != 'amq_comparison_v1' or saved['seed'] != seed or seed not in spec['seeds']:
        raise ValueError('Expected original v1 AMQ source and a matching seed')
    if any(raw['config'].get(k)!=v for k,v in {'quantizer':'hqq', 'hqq_axis':0,
            'hqq_zero_policy':'floating','quant_group_size':128}.items()):
        raise ValueError('Source quantizer settings differ from the diagnostic protocol')
    if digest(source/'experiment_config.json') != manifest['config_sha256']:
        raise ValueError('Source config hash mismatch')
    if spec['model_revision'] != manifest['resolved_model_revision'] or spec['dataset']['revision'] != manifest['resolved_dataset_revision']:
        raise ValueError('Source revision mismatch')
    trial = next(t for t in raw['trials'] if t['seed']==seed and t['method']=='AMQ-HQQ-Matched')
    if len(saved['archive']) != trial['configuration_evaluations']:
        raise ValueError('Archive count mismatch')
    # Validate every architecture and payload before selecting anything.
    families = {'self_attn.q_proj':4096*4096,'self_attn.k_proj':1024*4096,
        'self_attn.v_proj':1024*4096,'self_attn.o_proj':4096*4096,
        'mlp.gate_proj':14336*4096,'mlp.up_proj':14336*4096,'mlp.down_proj':4096*14336}
    if spec['model_id'] != 'meta-llama/Llama-3.1-8B-Instruct':
        raise ValueError('This diagnostic pins the existing Llama-3.1-8B experiment')
    import math
    for arch,jsd,effective in saved['archive']:
        if set(arch['linear']) != set(families) or not math.isfinite(jsd):
            raise ValueError('Invalid archive fields')
        if any(len(bs)!=32 or any(type(b) is not int or b not in (2,3,4) for b in bs) for bs in arch['linear'].values()):
            raise ValueError('Invalid archived precision')
        actual = sum(families[f]*sum(bs) for f,bs in arch['linear'].items())/(32*sum(families.values()))
        if not math.isclose(actual+.25,effective,abs_tol=1e-10):
            raise ValueError('Archived payload mismatch')
        for name in saved['fixed_4bit']:
            block,family=name.split('.',1)
            if arch['linear'][family][int(block)]!=4:
                raise ValueError('Archive violates fixed4 screening')
    selected = diagnostic_selection(saved['archive'],spec['budgets'],128)
    return spec,raw,selected,archive_path


def run(source, output, seed=42, plan_only=False):
    spec, raw, selected, archive_path = load_source(source,seed)
    output = Path(output)
    output.mkdir(parents=True,exist_ok=False)
    manifest = {'schema':'amq_archive_diagnostic_v1', 'complete':False,
        'source_directory':str(Path(source).resolve()), 'source_archive_sha256':digest(archive_path),
        'source_result_sha256':digest(Path(source)/'limited_labels.json'),
        'source_manifest_sha256':digest(Path(source)/'experiment_manifest.json'),
        'selection_rule':'old-window winner + two best under-cap calibration-JSD candidates per budget + all4 reference; deduplicated',
        'seed':seed, 'plan_only':plan_only, 'source_config':spec,
        'code_sha256':{p.name:digest(p) for p in Path(__file__).parent.glob('*.py')},
        'packages':{p:importlib.metadata.version(p) for p in ('torch','transformers','hqq','numpy','datasets','scipy')}}
    manifest['code_sha256']['scripts/run_amq_diagnostic_a100.sh'] = digest(Path(__file__).resolve().parents[1]/'scripts/run_amq_diagnostic_a100.sh')
    (output/'selection_plan.json').write_text(json.dumps(selected,indent=2))
    manifest_file = output/'diagnostic_manifest.json'
    manifest_file.write_text(json.dumps(manifest,indent=2))
    if plan_only:
        return selected
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .amq_baseline import run_amq_trial
    from .config import ExperimentConfig
    from .generalization_data import prepare_dataset, sha_tensor
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for archive evaluation; --plan-only is offline')
    tokenizer = AutoTokenizer.from_pretrained(spec['model_id'],revision=spec['model_revision'])
    args = SimpleNamespace(eval_chunks=spec['dataset']['eval_chunks'],eval_seq_len=spec['eval_seq_len'],
        calib_seq_len=spec['calibration']['seq_len'],n_calib_seqs=spec['calibration']['n_sequences'],seeds=[seed])
    data = prepare_dataset(tokenizer,args,spec['dataset'],spec['dataset']['revision'])
    calib = data['calib_by_seed'][seed]
    if sha_tensor(calib)!=raw['calibration_token_sha256'][str(seed)] or sha_tensor(data['eval_ids'])!=raw['evaluation_token_sha256']:
        raise ValueError('Recreated data hashes differ; refusing archive comparison')
    (output/'data_manifest.json').write_text(json.dumps(data['manifest'],indent=2))
    model = AutoModelForCausalLM.from_pretrained(spec['model_id'],revision=spec['model_revision'],torch_dtype=torch.float16).to('cuda').eval()
    cfg = ExperimentConfig(num_layers=32,quantizer='hqq',hqq_axis=0,hqq_zero_policy='floating',quant_group_size=128,device='cuda')
    result = run_amq_trial(model,calib,data['eval_ids'],cfg,spec['budgets'],seed,1,
        settings={},output=output,diagnostic=selected)
    prior = next(t for t in raw['trials'] if t['seed']==seed and t['method']=='AMQ-HQQ-Matched')
    for row in result['candidates']:
        bits={f'model.layers.{i}.{f}':b for f,bs in row['architecture']['linear'].items() for i,b in enumerate(bs)}
        expected = next((a['evaluation'] for a in prior['allocations'].values() if a['bit_assignment']==bits),None)
        if 'all4_reference' in row['reasons']:
            expected = raw['uniform_baselines']['Uniform-4bit']
        if expected is not None:
            row['historical_eval_nll_abs_difference'] = abs(row['evaluation']['nll']-expected['nll'])
            row['historical_eval_passed'] = row['historical_eval_nll_abs_difference'] <= 1e-6
    result['complete'] = all(r.get('historical_eval_passed',True) for r in result['candidates'])
    (output/'diagnostic.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    if not result['complete']:
        raise RuntimeError('Historical held-out evaluation mismatch; inspect diagnostic.json')
    lines = ['AMQ archive diagnostic; post-hoc analysis, not an independent method win.',
             'Archive index | actual payload | archived/recomputed JSD | held-out PPL | parity']
    for row in result['candidates']:
        lines.append(f"{row['archive_index']} | {row['payload_bits']:.6f} | {row['archived_jsd']:.6f}/{row['recomputed_jsd']:.6f} | {row['evaluation']['ppl']:.6f} | {row['parity_passed']}")
    (output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n')
    manifest['complete'] = True
    manifest_file.write_text(json.dumps(manifest,indent=2))
    return result


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-dir',required=True)
    p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--plan-only',action='store_true')
    a=p.parse_args()
    run(a.source_dir,a.output_dir,a.seed,a.plan_only)
