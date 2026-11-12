"""Separate seed42 WikiText feature-value diagnostic using immutable saved labels."""
import argparse
import csv
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from .allocation_control_plan import (add_estimator, build_plan, fit_metadata_controls)
from .amq_followup_plan import digest, geometry
from .run_amq_followup import save


def write_summaries(report, output):
    rows = []
    for candidate in report['candidates']:
        for role in candidate['roles']:
            rows.append(dict(candidate_id=candidate['candidate_id'], **role,
                actual_bits=candidate['actual_payload_bits'], ppl=candidate['evaluation']['ppl'],
                nll=candidate['evaluation']['nll'],
                reuses_labels=True, end_to_end_calibration_sec=None))
    if rows:
        with (output/'comparisons.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    comparisons = []
    for row in rows:
        matches = [r for r in rows if r['method'] == 'HDS-Hybrid' and
                   r['budget'] == row['budget'] and r['setting'] == row['setting']]
        if len(matches) == 1:
            hds = matches[0]
            comparisons.append(dict(method=row['method'], budget=row['budget'],
                setting=row['setting'], tie_order=row['tie_order'],
                control_ppl=row['ppl'], hds_ppl=hds['ppl'],
                hds_vs_control_ppl_percent=100*(hds['ppl']/row['ppl']-1)))
    if comparisons:
        with (output/'hds_vs_controls.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(comparisons[0]))
            writer.writeheader(); writer.writerows(comparisons)
    lines = [f'Complete: {report["complete"]}',
        'Seed42 WikiText diagnostic. Saved K=4 labels and learned predictions reused.',
        'Not a fresh end-to-end calibration timing benchmark or seed replication.',
        'Exact metadata: log-MSE, log-Q99, seven family indicators; no depth in original HDS.',
        'Depth ridge adds normalized depth as a separate control.',
        'Family primary order is depth ascending; report both random orders too, no test winner selection.',
        'Protection is the previously observed final-block down4 prior; its compensation is budget matched.',
        'Simple references and both constraint settings are measured, not PPL interpolations.',
        'AMQ anchors depend on its earlier screening. No new AMQ search is run.',
        'Negative hds_vs_control_ppl_percent favors HDS.']
    lines += [str(r) for r in comparisons]
    (output/'SUMMARY.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def evaluate_candidates(model, ids, cfg, plan, raw, output, *, summary_writer=None, result_fields=None):
    from .evaluator import evaluate_perplexity
    from .label_collector import discover_target_projections
    from .run_no1bit_diagnostic import check_evaluation
    meta = discover_target_projections(model)
    report = dict(schema='allocation_controls_result_v1', complete=False,
        seed=plan['seed'], acquired_layers=plan['acquired_layers'],
        probe_label_count=plan['probe_label_count'], reuses_labels=True,
        timing_scope=plan['timing_scope'], baseline_checks={}, candidates=[])
    if result_fields:
        report.update(result_fields)
    write_summary = summary_writer or write_summaries
    save(output/'diagnostic.json', report)
    for name, bit in [('FP16', None), ('Uniform-3bit', 3), ('Uniform-4bit', 4)]:
        ev = evaluate_perplexity(model, ids, meta, cfg, uniform_bit=bit, batch_size=1)
        report['baseline_checks'][name] = dict(evaluation=ev,
            **check_evaluation(ev, raw['uniform_baselines'][name], 1e-6))
        save(output/'diagnostic.json', report)
    for i, candidate in enumerate(plan['candidates']):
        print(f'Evaluating {i+1}/{len(plan["candidates"])}: {candidate["candidate_id"]} '
              f'{[r["method"] for r in candidate["roles"]]}', flush=True)
        ev = evaluate_perplexity(model, ids, meta, cfg,
            bit_assignment=candidate['bit_assignment'], batch_size=1)
        checks = [check_evaluation(ev, old, 1e-6) for old in candidate['historical_evaluations']]
        report['candidates'].append(dict(candidate, evaluation=ev, historical_replay_checks=checks))
        save(output/'diagnostic.json', report)
        write_summary(report, output)
    report['complete'] = True
    save(output/'diagnostic.json', report); write_summary(report, output)
    return report


def run(config, output, source_override=None, plan_only=False):
    config = Path(config); spec = json.loads(config.read_text())
    source = Path(source_override or spec['source_dir']).resolve()
    plan, raw, measured, family_raw = build_plan(spec, source)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    code = list(Path(__file__).parent.glob('*.py')) + [
        project/'scripts/run_allocation_controls_a100.sh', project/'scripts/load_hf_env.sh']
    try:
        revision = subprocess.run(['git', '-c', f'safe.directory={project.as_posix()}',
            '-C', str(project), 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(['git', '-c', f'safe.directory={project.as_posix()}',
            '-C', str(project), 'status', '--porcelain', '--untracked-files=no'],
            capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = None; dirty = 'unavailable'
    manifest = dict(schema='allocation_controls_v1', complete=False, plan_only=plan_only,
        plan_complete=False, source_directory=str(source), source_hashes=plan['source_hashes'],
        config_sha256=digest(config), git_revision=revision, git_tracked_changes=dirty,
        code_sha256={str(p.relative_to(project)): digest(p) for p in code},
        timing_scope=plan['timing_scope'])
    path = output/'diagnostic_manifest.json'
    (output/'experiment_config.json').write_bytes(config.read_bytes())
    save(output/'selection_plan.json', plan); save(path, manifest)
    if plan_only:
        print(f'Offline partial plan: {len(plan["candidates"])} unique assignments; '
              'metadata controls require CUDA feature collection. Not a completed experiment.', flush=True)
        return plan
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .config import ExperimentConfig
        from .generalization_data import prepare_dataset, sha_tensor
        from .hds_encoder import HDSEncoder
        from .label_collector import collect_features_single_pass, discover_target_projections
        from .quantizer import get_effective_hqq_config
        from .run_experiment import _pair_dict_to_json
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; --plan-only constructs the offline partial plan')
        manifest['packages'] = {n: importlib.metadata.version(n) for n in
            ['torch', 'transformers', 'hqq', 'datasets', 'scipy', 'scikit-learn', 'numpy']}
        save(path, manifest)
        src = plan['source_spec']
        tokenizer = AutoTokenizer.from_pretrained(src['model_id'], revision=src['model_revision'])
        args = SimpleNamespace(eval_chunks=src['dataset']['eval_chunks'], eval_seq_len=src['eval_seq_len'],
            calib_seq_len=src['calibration']['seq_len'], n_calib_seqs=src['calibration']['n_sequences'],
            seeds=[spec['seed']])
        data = prepare_dataset(tokenizer, args, src['dataset'], src['dataset']['revision'])
        if (sha_tensor(data['eval_ids']) != raw['evaluation_token_sha256'] or
            sha_tensor(data['calib_by_seed'][spec['seed']]) != raw['calibration_token_sha256'][str(spec['seed'])]):
            raise ValueError('Source calibration/evaluation token hashes differ')
        save(output/'data_manifest.json', data['manifest'])
        model = AutoModelForCausalLM.from_pretrained(src['model_id'], revision=src['model_revision'],
            torch_dtype=torch.float16).to('cuda').eval()
        cfg = ExperimentConfig(**raw['config']); meta = discover_target_projections(model)
        if {m['name']: m['num_params'] for m in meta} != {m['name']: m['num_params'] for m in geometry()}:
            raise ValueError('Loaded projection geometry differs')
        effective = get_effective_hqq_config(128, 'cuda', torch.float16, axis=0, zero_policy='floating')
        if effective != raw['hqq_config']:
            raise ValueError('Effective HQQ settings differ')
        manifest['effective_hqq_config'] = effective; save(path, manifest)
        # No sensitivity labels are acquired here. Metadata uses the existing FP-input residual path.
        encoder = HDSEncoder(hd_dim=cfg.hd_dim, num_levels=cfg.num_levels, max_coords=cfg.max_coords,
            num_buckets=cfg.num_buckets, seed=spec['seed'], device='cuda', build_codebooks=False)
        torch.cuda.synchronize(); start = time.perf_counter()
        features = collect_features_single_pass(model, data['calib_by_seed'][spec['seed']], cfg,
            encoder, batch_size=1, feature_estimator='Scalar-Meta-Ridge')['features']
        torch.cuda.synchronize(); feature_sec = time.perf_counter()-start
        fits = fit_metadata_controls(features, measured, meta, cfg, plan['acquired_layers'], spec['seed'])
        estimates = dict(schema='allocation_control_estimates_v1', reuses_labels=True,
            feature_extraction_sec=feature_sec,
            measured_labels=_pair_dict_to_json(measured),
            family_raw=_pair_dict_to_json(family_raw), metadata={})
        for method, fit in fits.items():
            add_estimator(plan, method, fit['costs'], meta, spec)
            estimates['metadata'][method] = dict(raw_predictions=_pair_dict_to_json(fit['raw']),
                feature_columns=fit['feature_columns'], fit_sec=fit['fit_sec'])
        del features, fits, encoder
        plan['metadata_pending'] = False
        # Freeze every assignment before any new test evaluation; write hash into the manifest.
        save(output/'estimates.json', estimates); save(output/'selection_plan.json', plan)
        manifest.update(plan_complete=True, selection_plan_sha256=digest(output/'selection_plan.json'))
        save(path, manifest)
        report = evaluate_candidates(model, data['eval_ids'], cfg, plan, raw, output)
        if any(digest(source/n) != h for n, h in plan['source_hashes'].items()):
            raise RuntimeError('Source changed during the run')
        if digest(output/'selection_plan.json') != manifest['selection_plan_sha256']:
            raise RuntimeError('Frozen assignment plan changed during evaluation')
        manifest['complete'] = True; save(path, manifest)
        print(f'Allocation controls completed: {output}', flush=True)
        return report
    except Exception as error:
        manifest.update(complete=False, error=f'{type(error).__name__}: {error}')
        save(path, manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--source-dir')
    parser.add_argument('--plan-only', action='store_true')
    a = parser.parse_args()
    run(a.config, a.output_dir, a.source_dir, a.plan_only)
