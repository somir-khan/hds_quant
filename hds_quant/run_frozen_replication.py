"""One dataset/calibration-seed case of the frozen K=4 control replication."""
import argparse
import importlib.metadata
import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from .allocation_control_plan import fit_metadata_controls
from .amq_followup_plan import digest, geometry
from .frozen_replication_plan import add_replication_estimator, build_replication_plan
from .frozen_replication_summary import summarize_replication
from .run_allocation_controls import evaluate_candidates
from .run_amq_followup import save


def run(config, output, seed, source_override=None, plan_only=False):
    config = Path(config); spec = json.loads(config.read_text())
    source = Path(source_override or spec['source_dir']).resolve()
    plan, raw, measured, family_raw = build_replication_plan(spec, source, seed)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    project = Path(__file__).resolve().parents[1]
    code = list(Path(__file__).parent.glob('*.py')) + [project/f for f in (
        'scripts/run_frozen_replication_wikitext_a100.sh', 'scripts/run_frozen_replication_c4_a100.sh',
        'scripts/launch_frozen_replication.sh', 'scripts/load_hf_env.sh',
        'scripts/summarize_frozen_replications.py')]
    def git(*args):
        try:
            return subprocess.run(['git', '-c', f'safe.directory={project.as_posix()}', '-C', str(project), *args],
                capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    manifest = dict(schema='frozen_replication_v1', complete=False, plan_complete=False,
        plan_only=plan_only, dataset=spec['dataset'], seed=seed,
        source_directory=str(source), source_hashes=plan['source_hashes'],
        config_sha256=digest(config), git_revision=git('rev-parse', 'HEAD'),
        git_tracked_changes=git('status', '--porcelain', '--untracked-files=no'),
        code_sha256={str(p.relative_to(project)): digest(p) for p in code},
        timing_scope=plan['timing_scope'])
    path = output/'diagnostic_manifest.json'
    (output/'experiment_config.json').write_bytes(config.read_bytes())
    save(output/'selection_plan.json', plan); save(path, manifest)
    if plan_only:
        print(f'{spec["dataset"]} seed={seed}: offline plan has {len(plan["candidates"])} unique assignments; '
              'metadata pending. This is not a completed experiment.', flush=True)
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
            raise RuntimeError('CUDA required. --plan-only builds the offline partial plan.')
        manifest['packages'] = {n: importlib.metadata.version(n) for n in
            ['torch', 'transformers', 'hqq', 'datasets', 'scipy', 'scikit-learn', 'numpy']}
        save(path, manifest)
        src = plan['source_spec']
        tokenizer = AutoTokenizer.from_pretrained(src['model_id'], revision=src['model_revision'])
        args = SimpleNamespace(eval_chunks=src['dataset']['eval_chunks'], eval_seq_len=src['eval_seq_len'],
            calib_seq_len=src['calibration']['seq_len'], n_calib_seqs=src['calibration']['n_sequences'], seeds=[seed])
        data = prepare_dataset(tokenizer, args, src['dataset'], src['dataset']['revision'])
        if (sha_tensor(data['eval_ids']) != raw['evaluation_token_sha256'] or
            sha_tensor(data['calib_by_seed'][seed]) != raw['calibration_token_sha256'][str(seed)]):
            raise ValueError('Source calibration/evaluation token hashes differ')
        save(output/'data_manifest.json', data['manifest'])
        model = AutoModelForCausalLM.from_pretrained(src['model_id'], revision=src['model_revision'],
            torch_dtype=torch.float16).to('cuda').eval()
        cfg = ExperimentConfig(**plan['experiment_config']); meta = discover_target_projections(model)
        if cfg.seeds != [seed]:
            raise ValueError('Wrong seed in the feature-extraction configuration')
        if {m['name']: m['num_params'] for m in meta} != {m['name']: m['num_params'] for m in geometry()}:
            raise ValueError('Loaded projection geometry differs')
        effective = get_effective_hqq_config(128, 'cuda', torch.float16, axis=0, zero_policy='floating')
        if effective != raw['hqq_config']:
            raise ValueError('Effective HQQ settings differ')
        manifest['effective_hqq_config'] = effective; save(path, manifest)
        encoder = HDSEncoder(hd_dim=cfg.hd_dim, num_levels=cfg.num_levels, max_coords=cfg.max_coords,
            num_buckets=cfg.num_buckets, seed=seed, device='cuda', build_codebooks=False)
        torch.cuda.synchronize(); start = time.perf_counter()
        features = collect_features_single_pass(model, data['calib_by_seed'][seed], cfg,
            encoder, batch_size=1, feature_estimator='Scalar-Meta-Ridge')['features']
        torch.cuda.synchronize(); feature_sec = time.perf_counter()-start
        fits = fit_metadata_controls(features, measured, meta, cfg, plan['acquired_layers'], seed)
        estimates = dict(schema='frozen_replication_estimates_v1', seed=seed, dataset=spec['dataset'],
            reuses_labels=True, feature_extraction_sec=feature_sec,
            measured_labels=_pair_dict_to_json(measured), family_raw=_pair_dict_to_json(family_raw), metadata={})
        for method, fit in fits.items():
            add_replication_estimator(plan, method, fit['costs'], meta, spec)
            estimates['metadata'][method] = dict(raw_predictions=_pair_dict_to_json(fit['raw']),
                feature_columns=fit['feature_columns'], fit_sec=fit['fit_sec'])
        del features, fits, encoder
        plan['metadata_pending'] = False
        save(output/'estimates.json', estimates); save(output/'selection_plan.json', plan)
        manifest.update(plan_complete=True, selection_plan_sha256=digest(output/'selection_plan.json'))
        save(path, manifest)
        result_fields = dict(schema='frozen_replication_result_v1', dataset=spec['dataset'],
            family_tie_orders=spec['family_tie_orders'], materiality_percent=1.0,
            exploratory_reference=spec['dataset'] == 'wikitext2' and seed == 42)
        report = evaluate_candidates(model, data['eval_ids'], cfg, plan, raw, output,
            summary_writer=summarize_replication, result_fields=result_fields)
        if any(digest(source/n) != h for n, h in plan['source_hashes'].items()):
            raise RuntimeError('Immutable source changed during the run')
        if digest(output/'selection_plan.json') != manifest['selection_plan_sha256']:
            raise RuntimeError('Frozen allocation plan changed during scoring')
        manifest['complete'] = True; save(path, manifest)
        print(f'Frozen replication completed: {output}', flush=True)
        return report
    except Exception as error:
        manifest.update(complete=False, error=f'{type(error).__name__}: {error}')
        save(path, manifest)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--source-dir')
    parser.add_argument('--plan-only', action='store_true')
    a = parser.parse_args()
    run(a.config, a.output_dir, a.seed, a.source_dir, a.plan_only)
