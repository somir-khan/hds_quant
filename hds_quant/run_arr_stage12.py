"""Separate ARR offline controls, allocation scoring, and one-case diagnostic."""
import argparse
import csv
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .arr_stage12_protocol import (FEATURE_ARMS, add_raw, build_controls, case_definitions,
    decode, encode, load_source, read, reduced_config, validate_spec, verify_sources)
from .amq_followup_plan import digest
from .run_amq_followup import save


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)


def gpu_context(source, output):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from .generalization_data import prepare_dataset, sha_tensor
    from .label_collector import discover_target_projections
    from .quantizer import get_effective_hqq_config
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for scoring; offline mode loads no model')
    seed = source['case']['seed']; ds = DATASETS[source['case']['dataset']]
    tokenizer = AutoTokenizer.from_pretrained(source['model_id'], revision=source['revision'])
    args = SimpleNamespace(eval_chunks=141, eval_seq_len=2048, calib_seq_len=1024,
                           n_calib_seqs=64, seeds=[seed])
    data = prepare_dataset(tokenizer, args, ds, ds['revision'])
    old = source['data_manifest']; new = data['manifest']
    for k in ('calibration_text_sha256', 'evaluation_text_sha256', 'calibration_stream_sha256',
              'evaluation_stream_sha256', 'evaluation_token_sha256', 'scored_tokens'):
        if old[k] != new[k]:
            raise ValueError(f'Source token provenance differs: {k}')
    if old['calibration_window_starts'][str(seed)] != new['calibration_window_starts'][str(seed)]:
        raise ValueError('Calibration sampling differs')
    if source['calibration_token_sha256'] is not None and sha_tensor(data['calib_by_seed'][seed]) != source['calibration_token_sha256']:
        raise ValueError('Calibration token hash differs from source case')
    cfg = reduced_config(source['cfg']); cfg.device = 'cuda'
    model = AutoModelForCausalLM.from_pretrained(source['model_id'], revision=source['revision'],
        torch_dtype=torch.float16).to('cuda').eval()
    meta = discover_target_projections(model)
    if {m['name']: (m['d_out'], m['d_in'], m['num_params']) for m in meta} != {
        m['name']: (m['d_out'], m['d_in'], m['num_params']) for m in source['meta']}:
        raise ValueError('Loaded model geometry differs from source')
    effective = get_effective_hqq_config(128, 'cuda', torch.float16, axis=0, zero_policy='floating')
    if effective != source['hqq_config']:
        raise ValueError('HQQ runtime differs from source')
    with torch.no_grad():
        model(input_ids=data['calib_by_seed'][seed][:1].to('cuda'), use_cache=False)
    torch.cuda.synchronize()
    save(output/'data_manifest.json', new)
    return model, data['calib_by_seed'][seed], data['eval_ids'], cfg, dict(
        effective_hqq_config=effective, calibration_token_sha256=sha_tensor(data['calib_by_seed'][seed]),
        evaluation_token_sha256=sha_tensor(data['eval_ids']), packages={n: importlib.metadata.version(n)
        for n in ('torch', 'transformers', 'hqq', 'datasets', 'scipy', 'scikit-learn', 'numpy')})


def quality_summary(report, output):
    rows = [dict(candidate_id=c['candidate_id'], **role, nll=c['evaluation']['nll'],
        ppl=c['evaluation']['ppl']) for c in report['candidates'] for role in c['roles']]
    write_csv(output/'comparisons.csv', rows)
    crows = []
    for row in rows:
        if row.get('variant') != 'predict_all':
            continue
        ref = [x for x in rows if (x['method'], x['budget'], x['k_layers'], x['policy'], x['variant']) ==
            (row['method'], row['budget'], row['k_layers'], row['policy'], 'retained_measured')]
        if len(ref) == 1:
            crows.append(dict(method=row['method'], budget=row['budget'], k_layers=row['k_layers'],
                policy=row['policy'], retained_ppl=ref[0]['ppl'], predicted_all_ppl=row['ppl'],
                predict_all_vs_retained_percent=100*(row['ppl']/ref[0]['ppl']-1)))
    write_csv(output/'retention_ablation.csv', crows)


def evaluate_controls(model, ids, cfg, source, plan, output):
    from .run_allocation_controls import evaluate_candidates
    baselines = {n: v['evaluation'] for n, v in source['report']['baseline_checks'].items()}
    return evaluate_candidates(model, ids, cfg, plan, dict(uniform_baselines=baselines), output,
        summary_writer=quality_summary, result_fields=dict(schema='arr_controls_quality_v1',
        dataset=source['case']['dataset'], model=source['case']['model'],
        calibration_timing_measured=False, uses_saved_banks=True))


def diagnostic(model, calib, ids, cfg, source, spec, output):
    from .allocation_control_plan import add_candidate
    from .allocation_controls import allocation_costs, last_down
    from .arr_stage12_diagnostics import composition_terms, joint_subsets, prediction_metrics
    from .coverage_protocol import protected_allocations
    from .evaluator import evaluate_perplexity
    from .label_collector import collect_marginal_loss_labels
    from .run_no1bit_diagnostic import check_evaluation
    meta = source['meta']; names = {m['name'] for m in meta}
    truth = {k: v for k, v in source['labels'].items() if k[1] in (2, 3)}
    known = {m['layer_idx'] for m in meta if (m['name'], 2) in truth}
    missing = sorted(set(range(16))-known)
    targets = sorted(set(missing) | set(spec['e_source_replay_blocks']))
    # Only missing blocks and two declared replay blocks are freshly probed.
    fresh = collect_marginal_loss_labels(model, calib, cfg, batch_size=1, target_layers=set(targets))
    measured = fresh['delta_loss']; overlap = measured.keys() & truth.keys()
    gap = max(abs(measured[k]-truth[k]) for k in overlap)
    if len(overlap) != 28 or gap > spec['replay_tolerance']:
        raise ValueError(f'E source-label replay failed: {len(overlap)} labels, maxgap={gap}')
    for k, v in measured.items():
        if k not in truth:
            truth[k] = v
    if truth.keys() != {(n, b) for n in names for b in (2, 3)}:
        raise ValueError('E full bank is incomplete')
    save(output/'full_label_bank.json', dict(schema='arr_full_bank_v1', measured_labels=encode(truth),
        frozen_source_labels=len(source['labels'])//3*2, missing_blocks=missing,
        newly_acquired_labels=len(measured)-len(overlap), replayed_labels=len(overlap),
        replay_max_abs_difference=gap, fresh_probe_call_sec=fresh['label_time_sec'],
        base_loss_4bit=fresh['base_loss_4bit'],
        timing_scope='Incremental bank completion plus replay. Not standalone full-label calibration time.'))
    plan = dict(schema='arr_e_plan_v1', candidates=[]); role_raw = {}; role_processed = {}
    for cell in source['estimates']['cells']:
        if (cell['k_layers'], cell['policy']) not in CELLS:
            continue
        for method in FEATURE_ARMS:
            raw = {k: v for k, v in decode(cell['raw_predictions'][method]).items() if k[1] in (2, 3)}
            costs = allocation_costs(raw, meta, cfg)
            for budget in spec['budgets']:
                original = [c for c in source['report']['candidates'] if any(
                    r['method'] == method and r['budget'] == budget and r['k_layers'] == cell['k_layers']
                    and r['policy'] == cell['policy'] and r['tie_order'] == 'native' for r in c['roles'])]
                if len(original) != 1:
                    raise ValueError('Missing E original assignment')
                role = dict(method=method, budget=budget, setting='last_down4', tie_order='native',
                    k_layers=cell['k_layers'], policy=cell['policy'], variant='original_frozen')
                add_candidate(plan, original[0]['bit_assignment'], meta, role, original[0]['evaluation'])
                key = (method, cell['k_layers']); role_raw[key] = raw; role_processed[key] = costs
    processed = allocation_costs(truth, meta, cfg)
    raw_costs = {(n, b): (0. if b == 4 else truth[n, b]) for n in names for b in (2, 3, 4)}
    for method, costs in [('Full-Label-Processed', processed), ('Full-Label-Raw', raw_costs)]:
        for budget in spec['budgets']:
            bits = protected_allocations(costs, meta, budget)['native']['bit_assignment']
            add_candidate(plan, bits, meta, dict(method=method, budget=budget, setting='last_down4',
                tie_order='native', k_layers=16, policy='all_blocks', variant='full_label_additive_reference'))
        role_raw[method, 16] = truth; role_processed[method, 16] = costs
    save(output/'selection_plan.json', plan); frozen_hash = digest(output/'selection_plan.json')
    evals = []; terms = []; base = fresh['base_loss_4bit']
    base_cal = evaluate_perplexity(model, calib, meta, cfg, uniform_bit=4, batch_size=1)
    if abs(base_cal['nll']-base) > spec['replay_tolerance']:
        raise ValueError('Calibration loss implementations disagree')
    for i, c in enumerate(plan['candidates']):
        print(f'E candidate {i+1}/{len(plan["candidates"])}', flush=True)
        cal = evaluate_perplexity(model, calib, meta, cfg, bit_assignment=c['bit_assignment'], batch_size=1)
        ev = evaluate_perplexity(model, ids, meta, cfg, bit_assignment=c['bit_assignment'], batch_size=1)
        for old in c['historical_evaluations']:
            check_evaluation(ev, old, spec['replay_tolerance'])
        evals.append(dict(c, calibration_evaluation=cal, evaluation=ev))
        for role in c['roles']:
            key = role['method'], role['k_layers']
            terms.append(dict(candidate_id=c['candidate_id'], **role,
                **composition_terms(c['bit_assignment'], truth, role_raw[key], base, cal['nll'], role_processed[key]),
                heldout_nll=ev['nll'], heldout_ppl=ev['ppl']))
        save(output/'diagnostic.json', dict(schema='arr_e_result_v1', complete=False,
            base_calibration_evaluation=base_cal, candidates=evals))
        write_csv(output/'composition_residuals.csv', terms)
    metrics = []
    for cell in source['estimates']['cells']:
        for method in FEATURE_ARMS:
            pred = decode(cell['raw_predictions'][method])
            for row in prediction_metrics(pred, truth, meta, cell['acquired_layers']):
                metrics.append(dict(method=method, k_layers=cell['k_layers'], policy=cell['policy'], **row))
    write_csv(output/'heldout_block_prediction_metrics.csv', metrics)
    # All subsets of first-block gate/up/down at 2 bits in an otherwise 4-bit
    # model. Base and singletons reuse full-bank measurements; only four joint
    # forward evaluations are needed. These probes intentionally vary payload.
    first = [f'model.layers.0.mlp.{f}_proj' for f in ('gate', 'up', 'down')]
    joint = []
    for subset in joint_subsets(first):
        summed = math.fsum(truth[n, 2] for n in subset)
        if len(subset) < 2:
            nll = base+summed; origin = 'reference_or_measured_singleton'
        else:
            bits = {n: 2 if n in subset else 4 for n in names}
            nll = evaluate_perplexity(model, calib, meta, cfg, bit_assignment=bits, batch_size=1)['nll']
            origin = 'fresh_joint_calibration_evaluation'
        joint.append(dict(subset=';'.join(subset), count=len(subset), calibration_nll=nll,
            actual_delta=nll-base, singleton_sum=summed, interaction_residual=nll-base-summed,
            origin=origin, payload_matched=False))
    write_csv(output/'joint_first_block.csv', joint)
    if digest(output/'selection_plan.json') != frozen_hash:
        raise RuntimeError('E frozen candidate plan changed')
    save(output/'diagnostic.json', dict(schema='arr_e_result_v1', complete=True,
        base_calibration_evaluation=base_cal, candidates=evals, joint_first_block=joint,
        scope='One 1B WikiText seed. Raw singleton composition diagnostics, not a causal attribution of all failure.'))


def run(config, output, mode, index, results_base, plan_only=False):
    spec = read(config); validate_spec(spec)
    if mode == 'diagnostic' and index != spec['e_case_index']:
        raise ValueError('E is frozen to case 6: 1B WikiText seed42')
    source = load_source(spec, results_base, index, features=mode != 'diagnostic')
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    def git(*args):
        try:
            return subprocess.run(['git', '-c', f'safe.directory={project.as_posix()}', '-C', str(project),
                *args], capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    manifest = dict(schema='arr_stage12_run_v1', complete=False, plan_complete=False,
        mode=mode, plan_only=plan_only or mode == 'offline', case=source['case'],
        model_id=source['model_id'], model_revision=source['revision'],
        source_hashes=source['hashes'], config_sha256=digest(config),
        git_revision=git('rev-parse', 'HEAD'), git_tracked_changes=git('status', '--porcelain', '--untracked-files=no'),
        code_sha256={str(p.relative_to(project)): digest(p) for p in [
            *Path(__file__).parent.glob('*.py'),
            *project.glob('scripts/*arr*.sh'), project/'scripts/load_hf_env.sh',
            project/'scripts/summarize_arr_stage12.py']},
        standalone_calibration_timing_measured=False)
    path = output/'run_manifest.json'; save(path, manifest)
    (output/'experiment_config.json').write_bytes(Path(config).read_bytes())
    try:
        if mode == 'diagnostic':
            known = {int(n.split('.')[2]) for n, b in source['labels'] if b == 2}
            missing = sorted(set(range(16))-known)
            save(output/'diagnostic_declaration.json', dict(case_index=index, missing_blocks=missing,
                new_labels=14*len(missing), replay_labels=28, joint_new_evaluations=4,
                heldout_scoring_selects_no_assignments=True))
        else:
            plan, parity, estimates = build_controls(source, spec)
            save(output/'selection_plan.json', plan); save(output/'parity.json', parity)
            save(output/'estimates.json', dict(cells=estimates))
            manifest['parity_passed'] = parity['passed']
        manifest['plan_complete'] = True; save(path, manifest)
        if mode == 'offline' or plan_only:
            verify_sources(source); manifest['complete'] = mode == 'offline'
            save(path, manifest)
            print(f'CPU-only plan complete: {source["case"]["case_id"]}; no model loaded.', flush=True)
            return manifest
        if mode == 'controls' and not parity['passed']:
            raise RuntimeError('A parity gate failed; inspect parity.json. GPU scoring was not started.')
        model, calib, ids, cfg, runtime = gpu_context(source, output)
        manifest.update(runtime); save(path, manifest)
        if mode == 'controls':
            frozen = digest(output/'selection_plan.json')
            evaluate_controls(model, ids, cfg, source, plan, output)
            if digest(output/'selection_plan.json') != frozen:
                raise RuntimeError('Controls plan changed during scoring')
        else:
            diagnostic(model, calib, ids, cfg, source, spec, output)
        verify_sources(source); manifest['complete'] = True; save(path, manifest)
        print(f'Completed {mode}: {output}', flush=True)
        return manifest
    except Exception as error:
        manifest.update(complete=False, error=f'{type(error).__name__}: {error}'); save(path, manifest)
        raise


# Imported here to keep all dataset declarations in the existing pinned module.
from .frozen_replication_plan import DATASETS
from .arr_stage12_protocol import CELLS


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/arr_stage12/frozen_v1.json')
    p.add_argument('--mode', choices=['offline', 'controls', 'diagnostic'], required=True)
    p.add_argument('--case-index', type=int, default=6)
    p.add_argument('--results-base', default='.')
    p.add_argument('--output-dir', required=True)
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args()
    run(a.config, a.output_dir, a.mode, a.case_index, a.results_base, a.plan_only)
