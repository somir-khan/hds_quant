"""Frozen assignment plans and allocation-only fixed4 intervention for saved AMQ v2 jobs."""
import hashlib
import json
import math
from pathlib import Path

FAMILIES = {'self_attn.q_proj': (4096,4096), 'self_attn.k_proj': (1024,4096),
    'self_attn.v_proj': (1024,4096), 'self_attn.o_proj': (4096,4096),
    'mlp.gate_proj': (14336,4096), 'mlp.up_proj': (14336,4096),
    'mlp.down_proj': (4096,14336)}
METHODS = ['HDS-Hybrid', 'Coord-Rank-Hist', 'Coord-Rank-Exact']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry():
    return [dict(name=f'model.layers.{i}.{f}', layer_idx=i, family_idx=j,
                 d_out=shape[0], d_in=shape[1], num_params=math.prod(shape))
            for i in range(32) for j,(f,shape) in enumerate(FAMILIES.items())]


def payload(bits, meta):
    if set(bits) != {m['name'] for m in meta} or any(type(b) is not int or b not in (2,3,4) for b in bits.values()):
        raise ValueError('Invalid assignment coverage or precision')
    return sum(m['num_params']*bits[m['name']] for m in meta)


def architecture(bits, layers=32):
    return {'linear': {f:[bits[f'model.layers.{i}.{f}'] for i in range(layers)] for f in FAMILIES}}


def assignment(arch):
    if set(arch.get('linear', {})) != set(FAMILIES) or any(len(bs)!=32 for bs in arch['linear'].values()):
        raise ValueError('Invalid archived geometry')
    return {f'model.layers.{i}.{f}':b for f,bs in arch['linear'].items() for i,b in enumerate(bs)}


def validate_spec(spec):
    if spec.get('schema') != 'amq_followup_v1' or spec.get('mode') not in ('objectives','fixed4'):
        raise ValueError('Expected an AMQ objectives/fixed4 follow-up config')
    if spec.get('seed') != 42 or spec.get('budgets') != [3.0,3.5] or spec.get('methods') != METHODS:
        raise ValueError('This follow-up pins seed42, both budgets, and three learned controls')
    if not spec.get('experiment_id') or not spec.get('source_dir'):
        raise ValueError('Missing experiment ID or source directory')


def load_source(source, seed):
    source = Path(source)
    files = ['experiment_config.json','resolved_experiment_config.json','experiment_manifest.json',
             'limited_labels.json','data_manifest.json',f'amq_seed_{seed}/archive.json']
    hashes = {f:digest(source/f) for f in files}
    spec = json.loads((source/files[1]).read_text())
    manifest = json.loads((source/files[2]).read_text())
    raw = json.loads((source/files[3]).read_text())
    saved = json.loads((source/files[5]).read_text())
    if not all(x.get('complete') for x in (manifest,raw,saved)):
        raise ValueError('Source job and archive must be complete')
    if spec['schema']!='amq_comparison_v2' or spec['amq_search']['iterations']!=20 or saved['iteration']!=20:
        raise ValueError('Expected completed 20-round AMQ BudgetCap-v2 source')
    if spec['model_id']!='meta-llama/Llama-3.1-8B-Instruct' or spec['dataset']['name']!='wikitext2':
        raise ValueError('Expected pinned 8B WikiText source')
    if hashes[files[0]]!=manifest['config_sha256'] or spec['model_revision']!=manifest['resolved_model_revision'] or spec['dataset']['revision']!=manifest['resolved_dataset_revision']:
        raise ValueError('Source config hash or resolved revisions differ')
    if any(raw['config'].get(k)!=v for k,v in dict(quantizer='hqq',hqq_axis=0,
            hqq_zero_policy='floating',quant_group_size=128,reference_bit=4,num_layers=32).items()):
        raise ValueError('Source quantizer/model configuration differs')
    if spec['probe_bits']!=[1,2,3,4] or spec['allocation_bits']!=[2,3,4] or spec['budgets']!=[3.0,3.5]:
        raise ValueError('Source precision/budget protocol differs')
    if saved['seed']!=seed or saved.get('budget_policy')!='cap_v2':
        raise ValueError('Source archive seed/policy mismatch')
    meta=geometry(); total=sum(m['num_params'] for m in meta)
    fixed=[f'model.layers.{name}' for name in saved['fixed_4bit']]
    if not fixed or not set(fixed) <= {m['name'] for m in meta}:
        raise ValueError('Invalid or empty fixed4 set')
    seen=set()
    for arch,jsd,effective in saved['archive']:
        bits=assignment(arch); actual=payload(bits,meta)/total
        key=json.dumps(bits,sort_keys=True)
        if key in seen or not math.isfinite(jsd) or not math.isclose(actual+.25,effective,abs_tol=1e-10) or any(bits[n]!=4 for n in fixed):
            raise ValueError('Invalid archive duplicate, JSD, payload, or fixed4 constraint')
        seen.add(key)
    trials=[t for t in raw['trials'] if t['seed']==seed]
    if len(trials)!=4 or {t['method'] for t in trials}!=set(METHODS+['AMQ-HQQ-BudgetCap-v2']):
        raise ValueError('Missing/duplicate source trials')
    amq=next(t for t in trials if t['method'].startswith('AMQ'))
    if amq['configuration_evaluations']!=len(saved['archive']) or amq['fixed_4bit']!=saved['fixed_4bit'] or amq['missing_budgets']:
        raise ValueError('Source AMQ metadata mismatch')
    from .amq_coverage import select_under_cap
    for t in trials:
        if t['method'] in METHODS and (t['k_layers']!=4 or t['probe_label_count']!=84):
            raise ValueError('Expected original K=4 learned trial')
        for b in spec['budgets']:
            row=t['allocations'][f'{b:.2f}']; actual=payload(row['bit_assignment'],meta)/total
            if not row['success'] or not math.isclose(actual,row['achieved_avg_bits'],abs_tol=1e-10) or actual>b+1e-10:
                raise ValueError('Invalid source allocation payload')
            if t['method'] in METHODS and not math.isclose(actual,b,abs_tol=1e-10):
                raise ValueError('Learned controls must have exact payload')
            if t is amq:
                best=select_under_cap(saved['archive'],b,128)
                if assignment(best[0])!=row['bit_assignment'] or best[1]!=row['calibration_jsd']:
                    raise ValueError('Source AMQ selection differs from archive')
    return spec,raw,saved,hashes,fixed


def fixed4_intervention(trial, budget, meta, cfg, fixed):
    from .run_no1bit_diagnostic import reconstruct_costs
    from .allocator import solve_bit_allocation_ilp, _FEASIBLE_PAYLOAD_CACHE
    costs=reconstruct_costs(trial,meta,cfg)
    original=trial['allocations'][budget]; before=original['bit_assignment']
    target=payload(before,meta)
    objective=sum(costs[n,b] for n,b in before.items())
    if not math.isclose(objective,original['total_predicted_cost'],abs_tol=1e-9,rel_tol=1e-9):
        raise ValueError('Reconstructed objective differs from source')
    if not set(fixed) <= set(before):
        raise ValueError('Unknown fixed projection')
    if all(before[n]==4 for n in fixed):
        after=dict(before); solved_objective=objective
    else:
        free=[m for m in meta if m['name'] not in fixed]
        remaining=target-4*sum(m['num_params'] for m in meta if m['name'] in fixed)
        if not free:
            raise ValueError('Fixed4 constraint cannot preserve the source payload')
        free_params=sum(m['num_params'] for m in free)
        if not 2*free_params <= remaining <= 4*free_params:
            raise ValueError('Fixed4 constraint cannot preserve the source payload')
        _FEASIBLE_PAYLOAD_CACHE.clear()
        result=solve_bit_allocation_ilp(costs,free,remaining/free_params,candidate_bits=(2,3,4))
        if not result['success']:
            raise RuntimeError('Fixed4 reallocation failed')
        after=dict(result['bit_assignment'],**{n:4 for n in fixed})
        solved_objective=sum(costs[n,b] for n,b in after.items())
    if payload(after,meta)!=target or any(after[n]!=4 for n in fixed):
        raise ValueError('Fixed4 intervention failed exact integer payload verification')
    if solved_objective < objective-1e-8:
        raise ValueError('Constrained solution unexpectedly improves the saved optimal objective')
    return dict(bit_assignment=after, matched_integer_payload=target,
        original_objective=objective, constrained_objective=solved_objective,
        changed_projections=[dict(projection=n,before=before[n],after=after[n]) for n in sorted(before) if before[n]!=after[n]])


def build_plan(source, spec):
    validate_spec(spec)
    src,raw,saved,hashes,fixed=load_source(source,spec['seed'])
    meta=geometry(); total=sum(m['num_params'] for m in meta)
    rows=[]; comparisons=[]
    archive_by_bits={json.dumps(assignment(a),sort_keys=True):(i,j) for i,(a,j,_) in enumerate(saved['archive'])}
    def add(bits,role,expected=None):
        for row in rows:
            if assignment(row['architecture'])==bits:
                row['roles'].append(role)
                if expected is not None: row['historical_evaluations'].append(expected)
                return row['candidate_id']
        index,jsd=archive_by_bits.get(json.dumps(bits,sort_keys=True),(None,None))
        cid=f'candidate_{len(rows):02d}'
        rows.append(dict(candidate_id=cid,architecture=architecture(bits),payload_bits=payload(bits,meta)/total,
            archived_jsd=jsd,archive_index=index,roles=[role],
            satisfies_amq_fixed4=all(bits[n]==4 for n in fixed),
            historical_evaluations=[] if expected is None else [expected]))
        return cid
    from .config import ExperimentConfig
    cfg=ExperimentConfig(**raw['config'])
    for method in METHODS:
        trial=next(t for t in raw['trials'] if t['seed']==spec['seed'] and t['method']==method)
        for b in spec['budgets']:
            key=f'{b:.2f}'; alloc=trial['allocations'][key]
            before=add(alloc['bit_assignment'],dict(method=method,budget=b,variant='original'),alloc['evaluation'])
            if spec['mode']=='fixed4':
                intervention=fixed4_intervention(trial,key,meta,cfg,fixed)
                after=add(intervention.pop('bit_assignment'),dict(method=method,budget=b,variant='fixed4'))
                comparisons.append(dict(method=method,budget=b,before_id=before,after_id=after,**intervention))
    amq=next(t for t in raw['trials'] if t['seed']==spec['seed'] and t['method'].startswith('AMQ'))
    for b in spec['budgets']:
        a=amq['allocations'][f'{b:.2f}']
        add(a['bit_assignment'],dict(method=amq['method'],budget=b,variant='selected'),a['evaluation'])
    if spec['mode']=='objectives':
        if len(amq['initialization_anchors'])!=2: raise ValueError('Missing saved initialization anchors')
        for b,arch in zip(spec['budgets'],amq['initialization_anchors']):
            bits=assignment(arch)
            if not math.isclose(payload(bits,meta)/total,b,abs_tol=1e-10) or any(bits[n]!=4 for n in fixed):
                raise ValueError('Invalid saved anchor')
            add(bits,dict(method='AMQ-anchor',budget=b,variant='initialization'))
    for b in [3,4]:
        add({m['name']:b for m in meta},dict(method=f'Uniform-{b}bit',budget=b,variant='reference'),raw['uniform_baselines'][f'Uniform-{b}bit'])
    return dict(schema='amq_followup_plan_v1',mode=spec['mode'],seed=spec['seed'],
        source_spec=src,source_hashes=hashes,fixed4_projections=fixed,
        selection_rule='Saved method winners and anchors; fixed-cost exact-payload intervention; no test selection',
        candidates=rows,comparisons=comparisons),raw
