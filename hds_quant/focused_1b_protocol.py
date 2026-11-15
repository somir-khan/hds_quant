"""Pinned two-dataset 1B replication without weakening the frozen 8B source checks."""
import json
from pathlib import Path

from .amq_followup_plan import digest
from .config import ExperimentConfig, PROJ_FAMILIES
from .coverage_protocol import ARMS, acquired_layers
from .frozen_replication_plan import DATASETS, ORDERS

MODEL = 'meta-llama/Llama-3.2-1B-Instruct'
REVISION = '9213176726f574b556790deb65791e0c5aa438b6'
CELLS = [dict(k_layers=2, policy='boundary_inclusive'), dict(k_layers=8, policy='stratified')]


def validate_spec(spec):
    expected = dict(schema='focused_1b_both_v1', model_id=MODEL, model_revision=REVISION,
        num_layers=16, seeds=[42, 100, 2026], methods=list(ARMS), comparison_cells=CELLS,
        budgets=[3., 3.5], primary_budget=3.5, secondary_budget=3., batch_size=1,
        family_tie_orders=list(ORDERS), protection='last_down4',
        calibration=dict(n_sequences=64, seq_len=1024, max_encoded_tokens=4096),
        eval_seq_len=2048, datasets=[DATASETS['wikitext2'], DATASETS['c4']])
    for key, value in expected.items():
        if spec.get(key) != value:
            raise ValueError(f'Frozen 1B protocol differs: {key}')
    if not spec.get('historical_c4_source_dir'):
        raise ValueError('Historical C4 provenance source is required')


def load_historical_source(directory):
    """Accept the old generalization bundle, including legitimate main -> resolved pins.

    Old labels/PPL are not supplied to new fits or used as numerical replay targets:
    the old execution was batch 8 and the new execution is batch 1.
    """
    directory = Path(directory)
    names = ['experiment_config.json', 'resolved_experiment_config.json',
             'experiment_manifest.json', 'limited_labels.json', 'data_manifest.json']
    hashes = {n: digest(directory/n) for n in names}
    original, resolved, manifest, raw, data = [json.loads((directory/n).read_text()) for n in names]
    if (not manifest.get('complete') or not raw.get('complete') or manifest.get('smoke_test') or
        manifest.get('config_sha256') != hashes[names[0]]):
        raise ValueError('Incomplete or hash-mismatched historical 1B source')
    if resolved.get('schema') != 'generalization_v1' or resolved.get('model_id') != MODEL:
        raise ValueError('Expected the original 1B generalization source')
    if (resolved['model_revision'] != REVISION or manifest['resolved_model_revision'] != REVISION or
        resolved['dataset'] != DATASETS['c4'] or manifest['resolved_dataset_revision'] != DATASETS['c4']['revision']):
        raise ValueError('Historical model/dataset pin mismatch')
    if (original['model_id'] != MODEL or original['model_revision'] not in ('main', REVISION) or
        original['dataset']['revision'] not in ('main', DATASETS['c4']['revision']) or
        dict(original['dataset'], revision=resolved['dataset']['revision']) != resolved['dataset']):
        raise ValueError('Original/resolved historical provenance differs')
    cfg = raw['config']
    expected = dict(model_name=MODEL, num_layers=16, quantizer='hqq', hqq_axis=0,
        hqq_zero_policy='floating', quant_group_size=128, candidate_bits=[1, 2, 3, 4],
        reference_bit=4, residual_anchor='int4_aligned', coord_binding='salience_bucket',
        readout_grouping='per_bit', target_transform='signed_log1p')
    if any(cfg.get(k) != v for k, v in expected.items()):
        raise ValueError('Historical 1B quantizer/readout settings differ')
    if (resolved['batch_size'] != 8 or resolved['calibration'] != dict(n_sequences=64, seq_len=1024, max_encoded_tokens=4096)
        or resolved['eval_seq_len'] != 2048 or resolved['seeds'] != [42, 100, 2026]):
        raise ValueError('Unexpected historical acquisition/evaluation protocol')
    if (raw['evaluation_token_sha256'] != data['evaluation_token_sha256'] or
        raw['evaluation_token_sha256'] != manifest['evaluation_token_sha256'] or
        data['scored_tokens'] != 288627):
        raise ValueError('Historical C4 token provenance differs')
    return dict(hashes=hashes, data_manifest=data,
        calibration_token_sha256=raw['calibration_token_sha256'],
        hqq_config=raw['hqq_config'], historical_batch_size=8,
        old_scores_are_not_replay_targets=True)


def make_config(spec, seed, device='cuda'):
    return ExperimentConfig(model_name=MODEL, num_layers=16, quantizer='hqq', hqq_axis=0,
        hqq_zero_policy='floating', quant_group_size=128, candidate_bits=(1, 2, 3, 4),
        reference_bit=4, max_encoded_tokens=4096, calib_num_sequences=64, calib_seq_len=1024,
        hd_dim=4096, num_levels=32, num_buckets=512, max_coords=8192, seeds=[seed], device=device,
        residual_anchor='int4_aligned', coord_binding='salience_bucket',
        readout_grouping='per_bit', target_transform='signed_log1p')


def focused_acquisition(spec, seed, num_layers=16):
    cells = [dict(c, acquired_layers=acquired_layers(num_layers, c['k_layers'], seed, c['policy']),
        probe_label_count=21*c['k_layers'], block_coverage_fraction=c['k_layers']/num_layers)
        for c in spec['comparison_cells']]
    return dict(schema='focused_1b_acquisition_v1', seed=seed, num_layers=num_layers, cells=cells,
        union_layers=sorted({i for c in cells for i in c['acquired_layers']}),
        note='Focused non-nested K=2 boundary vs K=8 stratified. Fits receive only 42 or 168 labels.')


def projection_geometry(config):
    """Expected linear shapes derived from the checkpoint's model configuration."""
    hidden = int(config.hidden_size); intermediate = int(config.intermediate_size)
    heads = int(config.num_attention_heads); kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, 'head_dim', None) or hidden//heads)
    shapes = [(heads*head_dim, hidden), (kv_heads*head_dim, hidden),
              (kv_heads*head_dim, hidden), (hidden, heads*head_dim),
              (intermediate, hidden), (intermediate, hidden), (hidden, intermediate)]
    return [dict(name=f'model.layers.{i}.{family}', layer_idx=i, family_idx=j,
        d_out=shape[0], d_in=shape[1], num_params=shape[0]*shape[1])
        for i in range(config.num_hidden_layers) for j, (family, shape) in enumerate(zip(PROJ_FAMILIES, shapes))]


def validate_model(model, actual_meta):
    config = model.config
    if (config.num_hidden_layers, config.hidden_size, config.intermediate_size) != (16, 2048, 8192):
        raise ValueError('Loaded model is not the declared 1B architecture')
    expected = projection_geometry(config)
    actual = {m['name']: (m['num_params'], m['d_out'], m['d_in']) for m in actual_meta}
    if actual != {m['name']: (m['num_params'], m['d_out'], m['d_in']) for m in expected}:
        raise ValueError('Loaded projection geometry differs from pinned model configuration')
    return expected
