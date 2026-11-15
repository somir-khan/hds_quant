"""Run both 1B datasets in one process with fresh quality and independent timing."""
import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .amq_followup_plan import digest
from .coverage_engine import build_quality_plan, run_timing
from .coverage_summary import summarize_timing
from .focused_1b_protocol import (MODEL, REVISION, focused_acquisition, load_historical_source,
    make_config, validate_model, validate_spec)
from .focused_1b_summary import aggregate, check_timing, summarize_quality
from .run_allocation_controls import evaluate_candidates
from .run_amq_followup import save


def fresh_baselines(model, eval_ids, cfg):
    from .evaluator import evaluate_perplexity
    from .label_collector import discover_target_projections
    meta = discover_target_projections(model)
    return {name: evaluate_perplexity(model, eval_ids, meta, cfg, uniform_bit=bit, batch_size=1)
            for name, bit in [('FP16', None), ('Uniform-3bit', 3), ('Uniform-4bit', 4)]}


def verify_c4_tokens(data, historical, seeds):
    from .generalization_data import sha_tensor
    old = historical['data_manifest']; new = data['manifest']
    fields = ['calibration_text_sha256', 'evaluation_text_sha256', 'calibration_stream_sha256',
              'evaluation_stream_sha256', 'evaluation_token_sha256', 'scored_tokens']
    if any(new[k] != old[k] for k in fields):
        raise ValueError('Fresh C4 data differs from pinned historical token provenance')
    for seed in seeds:
        if sha_tensor(data['calib_by_seed'][seed]) != historical['calibration_token_sha256'][str(seed)]:
            raise ValueError('Fresh C4 calibration tokens differ from historical provenance')


def run(config, output, *, historical_override=None, selected_seed=None, quality_only=False, plan_only=False):
    config = Path(config); spec = json.loads(config.read_text()); validate_spec(spec)
    seeds = spec['seeds'] if selected_seed is None else [selected_seed]
    if any(type(s) is not int or s not in spec['seeds'] for s in seeds):
        raise ValueError('Pilot seed must be 42, 100 or 2026')
    source = Path(historical_override or spec['historical_c4_source_dir']).resolve()
    historical = load_historical_source(source)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    code = list(Path(__file__).parent.glob('*.py')) + [project/f for f in (
        'scripts/run_focused_1b_both_a100.sh', 'scripts/load_hf_env.sh', 'scripts/summarize_focused_1b.py')]
    def git(*args):
        try:
            return subprocess.run(['git', '-c', f'safe.directory={project.as_posix()}', '-C', str(project), *args],
                text=True, capture_output=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    manifest = dict(schema='focused_1b_run_v1', complete=False, plan_only=plan_only,
        model_id=MODEL, model_revision=REVISION, num_layers=16, batch_size=1,
        primary_budget=3.5, secondary_budget=3., datasets=[d['name'] for d in spec['datasets']],
        selected_seeds=seeds, quality_only=quality_only, baseline_sha256={}, completed_cases=[],
        historical_c4_directory=str(source), historical_source_hashes=historical['hashes'],
        historical_numeric_replay_performed=False,
        historical_context='Old batch-8 source supplies model/data pins only. Fresh batch-1 baselines and fits on both datasets.',
        config_sha256=digest(config), git_revision=git('rev-parse', 'HEAD'),
        git_tracked_changes=git('status', '--porcelain', '--untracked-files=no'),
        code_sha256={str(p.relative_to(project)): digest(p) for p in code})
    (output/'experiment_config.json').write_bytes(config.read_bytes())
    save(output/'acquisition_plan.json', dict(schema='focused_1b_plans_v1',
        model_geometry_validation_pending=True, plans=[focused_acquisition(spec, s) for s in seeds]))
    path = output/'run_manifest.json'; save(path, manifest)
    if plan_only:
        print(f'Offline 1B preflight: both datasets, seeds={seeds}, batch=1, '
            'K2 boundary vs K8 stratified; model topology and fresh baselines pending. No model/data downloaded.', flush=True)
        return manifest
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .generalization_data import prepare_dataset, sha_tensor
        from .label_collector import discover_target_projections
        from .quantizer import get_effective_hqq_config
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; --plan-only validates the offline declaration/provenance')
        manifest['packages'] = {n: importlib.metadata.version(n) for n in
            ['torch', 'transformers', 'hqq', 'datasets', 'scipy', 'scikit-learn', 'numpy']}
        save(path, manifest)
        tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
        model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION,
            torch_dtype=torch.float16).to('cuda').eval()
        geometry = validate_model(model, discover_target_projections(model))
        save(output/'model_config.json', json.loads(model.config.to_json_string()))
        save(output/'projection_geometry.json', geometry)
        declared = json.loads((output/'acquisition_plan.json').read_text())
        declared.update(model_geometry_validation_pending=False,
            projection_geometry_sha256=digest(output/'projection_geometry.json'))
        save(output/'acquisition_plan.json', declared)
        effective = get_effective_hqq_config(128, 'cuda', torch.float16, axis=0, zero_policy='floating')
        if effective != historical['hqq_config']:
            raise ValueError('HQQ settings differ from the declared 1B protocol')
        manifest.update(effective_hqq_config=effective, model_geometry_validated=True,
            model_config_sha256=digest(output/'model_config.json'),
            projection_geometry_sha256=digest(output/'projection_geometry.json'),
            acquisition_plan_sha256=digest(output/'acquisition_plan.json'))
        save(path, manifest)
        args = SimpleNamespace(eval_chunks=141, eval_seq_len=2048, calib_seq_len=1024, n_calib_seqs=64, seeds=seeds)
        for dataset in spec['datasets']:
            name = dataset['name']; directory = output/name; directory.mkdir()
            print(f'Preparing pinned 1B dataset: {name}', flush=True)
            data = prepare_dataset(tokenizer, args, dataset, dataset['revision'])
            if name == 'c4':
                verify_c4_tokens(data, historical, seeds)
            save(directory/'data_manifest.json', data['manifest'])
            manifest.setdefault('data_manifest_sha256', {})[name] = digest(directory/'data_manifest.json')
            baselines = None
            for seed in seeds:
                case = directory/f'seed_{seed}'; case.mkdir()
                cfg = make_config(spec, seed)
                save(case/'resolved_model_config.json', cfg.to_dict())
                calib = data['calib_by_seed'][seed]
                with torch.no_grad():
                    warmup = model(input_ids=calib[:1].to('cuda'), use_cache=False)
                    del warmup
                torch.cuda.synchronize()
                acquisitions = focused_acquisition(spec, seed)
                quality_dir = case/'quality'; quality_dir.mkdir()
                common = dict(schema='focused_1b_case_v1', complete=False, dataset=name, seed=seed,
                    parent_config_sha256=manifest['config_sha256'], batch_size=1,
                    evaluation_token_sha256=sha_tensor(data['eval_ids']),
                    calibration_token_sha256=sha_tensor(calib), num_layers=16)
                qmanifest = dict(common, mode='quality')
                save(quality_dir/'case_manifest.json', qmanifest)
                print(f'1B {name}, seed={seed}: fresh shared-label quality sweep', flush=True)
                plan = build_quality_plan(model, calib, cfg, dict(spec, dataset=name), seed,
                    quality_dir, {}, acquisition_override=acquisitions)
                save(quality_dir/'selection_plan.json', plan)
                # Each allocation is frozen before its scores, including the first fresh reference baselines.
                if baselines is None:
                    baselines = fresh_baselines(model, data['eval_ids'], cfg)
                    save(directory/'fresh_uniform_baselines.json', dict(schema='fresh_1b_baselines_v1',
                        model_id=MODEL, model_revision=REVISION, dataset=name, batch_size=1,
                        evaluation_token_sha256=common['evaluation_token_sha256'], evaluations=baselines,
                        reference='Fresh current pipeline; old unpinned/batch-8 scores are not numerical targets.'))
                    manifest['baseline_sha256'][name] = digest(directory/'fresh_uniform_baselines.json')
                    save(path, manifest)
                qmanifest.update(selection_plan_sha256=digest(quality_dir/'selection_plan.json'),
                    baseline_sha256=manifest['baseline_sha256'][name],
                    label_bank_sha256=digest(quality_dir/'label_bank.json'), feature_bank_sha256=digest(quality_dir/'feature_bank.pt'))
                save(quality_dir/'case_manifest.json', qmanifest)
                quality = evaluate_candidates(model, data['eval_ids'], cfg, plan,
                    dict(uniform_baselines=baselines), quality_dir, summary_writer=summarize_quality,
                    result_fields=dict(schema='focused_1b_quality_v1', dataset=name, num_layers=16,
                        primary_budget=3.5, batch_size=1, old_numeric_replay_performed=False))
                if digest(quality_dir/'selection_plan.json') != qmanifest['selection_plan_sha256']:
                    raise RuntimeError('Frozen 1B quality plan changed')
                qmanifest['complete'] = True; save(quality_dir/'case_manifest.json', qmanifest)
                if not quality_only:
                    estimates = json.loads((quality_dir/'estimates.json').read_text())
                    for cell in acquisitions['cells']:
                        td = case/f'timing_k{cell["k_layers"]}_{cell["policy"]}'; td.mkdir()
                        tm = dict(common, mode='timing', k_layers=cell['k_layers'], policy=cell['policy'])
                        save(td/'case_manifest.json', tm)
                        timed = run_timing(model, calib, cfg, dict(spec, dataset=name), seed, cell, td, {})
                        parity = check_timing(quality, estimates, timed)
                        save(td/'quality_parity.json', parity); summarize_timing(timed, td)
                        tm.update(complete=True, timing_sha256=digest(td/'timing.json'),
                            quality_assignment_checks=len(parity), quality_manifest_sha256=digest(quality_dir/'case_manifest.json'))
                        save(td/'case_manifest.json', tm)
                manifest['completed_cases'].append(dict(dataset=name, seed=seed))
                save(path, manifest)
                print(f'Completed 1B {name}, seed={seed}', flush=True)
            del data
        if any(digest(source/n) != h for n, h in historical['hashes'].items()):
            raise RuntimeError('Historical provenance source changed during execution')
        summary = aggregate(output, require_completed_run=False)
        manifest.update(complete=True, summary=summary); save(path, manifest)
        print(f'Both-dataset 1B study completed: {output}; {summary}', flush=True)
        return manifest
    except Exception as error:
        manifest.update(complete=False, error=f'{type(error).__name__}: {error}'); save(path, manifest)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output-root', required=True)
    p.add_argument('--historical-c4-source-dir'); p.add_argument('--seed', type=int)
    p.add_argument('--quality-only', action='store_true'); p.add_argument('--plan-only', action='store_true')
    a = p.parse_args(); run(a.config, a.output_root, historical_override=a.historical_c4_source_dir,
        selected_seed=a.seed, quality_only=a.quality_only, plan_only=a.plan_only)
