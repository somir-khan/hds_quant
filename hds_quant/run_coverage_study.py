"""Versioned five-arm coverage study: shared quality sweep or independent timing cell."""
import argparse
import importlib.metadata
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

from .amq_followup_plan import digest, geometry
from .coverage_engine import build_quality_plan, run_timing
from .coverage_protocol import acquisition_plan, validate_spec
from .coverage_summary import summarize_quality, summarize_timing
from .frozen_replication_plan import load_replication_source
from .run_allocation_controls import evaluate_candidates
from .run_amq_followup import save


def run(config, output, seed, mode, source_override=None, k=None, policy=None, plan_only=False):
    config = Path(config); spec = json.loads(config.read_text())
    validate_spec(spec, seed, mode, k, policy)
    source = Path(source_override or spec['source_dir']).resolve()
    src, raw, hashes, cfg, trials, source_labels = load_replication_source(source, spec['dataset'], seed)
    acquired = acquisition_plan(spec, seed, cfg.num_layers)
    cell = next((c for c in acquired['cells'] if c['k_layers'] == k and c['policy'] == policy), None)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    code = list(Path(__file__).parent.glob('*.py')) + list((project/'scripts').glob('*coverage*.sh')) + [
        project/'scripts/load_hf_env.sh', project/'scripts/summarize_coverage_study.py']
    def git(*args):
        try:
            return subprocess.run(['git', '-c', f'safe.directory={project.as_posix()}', '-C', str(project), *args],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    manifest = dict(schema='coverage_study_v1', complete=False, plan_complete=False, plan_only=plan_only,
        mode=mode, dataset=spec['dataset'], seed=seed, k_layers=k, policy=policy,
        source_directory=str(source), source_hashes=hashes, config_sha256=digest(config),
        git_revision=git('rev-parse', 'HEAD'), git_tracked_changes=git('status', '--porcelain', '--untracked-files=no'),
        code_sha256={str(p.relative_to(project)): digest(p) for p in code})
    path = output/'study_manifest.json'
    (output/'experiment_config.json').write_bytes(config.read_bytes())
    save(output/'acquisition_plan.json', acquired); save(output/'resolved_model_config.json', cfg.to_dict()); save(path, manifest)
    if plan_only:
        print(f'{spec["dataset"]} seed={seed} mode={mode}: '
            f'{len(acquired["union_layers"])} union blocks for quality; selected timing cell={cell}. '
            'No model/data loaded, no labels/features acquired, experiment incomplete.', flush=True)
        return acquired
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .generalization_data import prepare_dataset, sha_tensor
        from .label_collector import discover_target_projections
        from .quantizer import get_effective_hqq_config
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required; --plan-only is an offline acquisition preflight')
        manifest['packages'] = {n: importlib.metadata.version(n) for n in
            ['torch', 'transformers', 'hqq', 'datasets', 'scipy', 'scikit-learn', 'numpy']}
        save(path, manifest)
        tokenizer = AutoTokenizer.from_pretrained(src['model_id'], revision=src['model_revision'])
        args = SimpleNamespace(eval_chunks=141, eval_seq_len=2048, calib_seq_len=1024, n_calib_seqs=64, seeds=[seed])
        data = prepare_dataset(tokenizer, args, src['dataset'], src['dataset']['revision'])
        if (sha_tensor(data['eval_ids']) != raw['evaluation_token_sha256'] or
            sha_tensor(data['calib_by_seed'][seed]) != raw['calibration_token_sha256'][str(seed)]):
            raise ValueError('Source token hashes differ')
        save(output/'data_manifest.json', data['manifest'])
        model = AutoModelForCausalLM.from_pretrained(src['model_id'], revision=src['model_revision'],
            torch_dtype=torch.float16).to('cuda').eval()
        meta = discover_target_projections(model)
        if {m['name']: m['num_params'] for m in meta} != {m['name']: m['num_params'] for m in geometry()}:
            raise ValueError('Loaded model geometry differs')
        effective = get_effective_hqq_config(128, 'cuda', torch.float16, axis=0, zero_policy='floating')
        if effective != raw['hqq_config']:
            raise ValueError('Effective HQQ settings differ')
        manifest['effective_hqq_config'] = effective; save(path, manifest)
        calib = data['calib_by_seed'][seed]
        with torch.no_grad():
            warmup = model(input_ids=calib[:1].to('cuda'), use_cache=False)
            del warmup
        torch.cuda.synchronize()
        if mode == 'quality':
            plan = build_quality_plan(model, calib, cfg, spec, seed, output, source_labels)
            # Replay any historical source assignment that independently appears in this plan.
            for candidate in plan['candidates']:
                for trial in trials.values():
                    for allocation in trial['allocations'].values():
                        if candidate['bit_assignment'] == allocation['bit_assignment']:
                            candidate['historical_evaluations'].append(allocation['evaluation'])
            save(output/'selection_plan.json', plan)
            manifest.update(plan_complete=True, selection_plan_sha256=digest(output/'selection_plan.json'),
                label_bank_sha256=digest(output/'label_bank.json'), feature_bank_sha256=digest(output/'feature_bank.pt'))
            save(path, manifest)
            result = evaluate_candidates(model, data['eval_ids'], cfg, plan, raw, output,
                summary_writer=summarize_quality,
                result_fields=dict(schema='coverage_quality_result_v1', dataset=spec['dataset'],
                    union_label_bank_is_shared=True, per_cell_label_counts=[42, 84, 168]))
            if digest(output/'selection_plan.json') != manifest['selection_plan_sha256']:
                raise RuntimeError('Frozen plan changed during scoring')
        else:
            result = run_timing(model, calib, cfg, spec, seed, cell, output, source_labels)
            summarize_timing(result, output)
            manifest.update(plan_complete=True, timing_sha256=digest(output/'timing.json'))
        if any(digest(source/n) != h for n, h in hashes.items()):
            raise RuntimeError('Immutable source changed during study')
        manifest['complete'] = True; save(path, manifest)
        print(f'Coverage {mode} completed: {output}', flush=True)
        return result
    except Exception as error:
        manifest.update(complete=False, error=f'{type(error).__name__}: {error}')
        save(path, manifest)
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--output-dir', required=True)
    p.add_argument('--seed', required=True, type=int); p.add_argument('--mode', choices=['quality', 'timing'], required=True)
    p.add_argument('--source-dir'); p.add_argument('--k-layers', type=int); p.add_argument('--policy')
    p.add_argument('--plan-only', action='store_true')
    a = p.parse_args(); run(a.config, a.output_dir, a.seed, a.mode, a.source_dir, a.k_layers, a.policy, a.plan_only)
