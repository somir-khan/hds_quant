"""Official AMQ search components with a matched-data, matched-HQQ evaluator.

This is AMQ-HQQ-Matched, not a reproduction of the paper's final AWQ results.
"""
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .config import PROJ_FAMILIES
from .evaluator import evaluate_perplexity
from .label_collector import discover_target_projections
from .quantizer import quantize_weight
from .run_limited_labels import sync, memory_start, memory_finish

METHOD = "AMQ-HQQ-Matched"


def select_candidate(archive, budget, group_size, tolerance=0.005):
    # Never overspend a payload budget. No test-set selection or silent fallback.
    feasible = [a for a in archive if budget-tolerance <= a[2]-32/group_size <= budget+1e-10]
    return min(feasible, key=lambda a: a[1]) if feasible else None


@torch.no_grad()
def jsd_tokens(p, q):
    """Upstream epsilon-clamped JSD, FP32 arithmetic, mean over scored tokens."""
    p, q = p.float(), q.float()
    lp, lq = p.log_softmax(-1), q.log_softmax(-1)
    m = (0.5 * (lp.exp() + lq.exp())).clamp_min(1e-7).log()
    return (0.5 * ((lp.exp() * (lp-m)).sum(-1) + (lq.exp() * (lq-m)).sum(-1))).sum()


def run_amq_trial(model, calib_ids, eval_ids, cfg, budgets, seed, batch_size, *, settings, output):
    from .amq_vendor.search import Search
    from .amq_vendor.space import SearchSpace
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    meta = discover_target_projections(model)
    directory = Path(output) / f"amq_seed_{seed}"
    directory.mkdir(exist_ok=False)
    baseline = memory_start(device)
    sync(device)
    start = time.perf_counter()
    originals, proxies, current = {}, {}, {}
    archive = []
    mapping = {(m['layer_idx'], '.'.join(m['name'].split('.')[-2:])): m for m in meta}
    config = {'n_block': cfg.num_layers, 'n_linear': 7, 'linear': list(PROJ_FAMILIES),
              'linear_shape': {f: [mapping[0, f]['d_out'], mapping[0, f]['d_in']] for f in PROJ_FAMILIES},
              'model_numel': sum(m['num_params'] for m in meta)}
    if len(mapping) != cfg.num_layers * 7 or any(
            [m['d_out'], m['d_in']] != config['linear_shape'][f] for (i, f), m in mapping.items()):
        raise ValueError('AMQ requires seven equally shaped projection families across blocks')
    teacher = None
    try:
        # Cache on host to fit one A100. Count cache construction in total time.
        for m in meta:
            name = m['name']
            originals[name] = m['module'].weight.detach().cpu().clone()
            for bit in (2, 3, 4):
                proxies[name, bit] = quantize_weight(originals[name].to(device), bits=bit,
                    group_size=cfg.quant_group_size, clip_ratio=cfg.clip_ratio, quantizer=cfg.quantizer,
                    hqq_axis=cfg.hqq_axis, hqq_zero_policy=cfg.hqq_zero_policy).cpu()
        # Disk-backed teacher logits avoid retaining all vocabulary logits on GPU.
        shape = (len(calib_ids), calib_ids.shape[1]-1, model.config.vocab_size)
        teacher = np.memmap(directory/'teacher_logits.bin', mode='w+', dtype=np.float16, shape=shape)
        with torch.inference_mode():
            for i, ids in enumerate(calib_ids):
                logits = model(input_ids=ids[None].to(device), use_cache=False).logits[0, :-1]
                teacher[i] = logits.cpu().half().numpy()
        teacher.flush()
        sync(device)
        proxy_sec = time.perf_counter()-start

        def assignment(arch):
            return {m['name']: int(arch['linear'][f][i]) for (i, f), m in mapping.items()}

        def score(arch):
            assigned = assignment(arch)
            with torch.inference_mode():
                for m in meta:
                    name, bit = m['name'], assigned[m['name']]
                    if current.get(name) != bit:
                        m['module'].weight.copy_(proxies[name, bit].to(device))
                        current[name] = bit
                total = 0.0
                for i, ids in enumerate(calib_ids):
                    logits = model(input_ids=ids[None].to(device), use_cache=False).logits[0, :-1]
                    for j in range(0, len(logits), 128):
                        ref = torch.from_numpy(np.array(teacher[i, j:j+128], copy=True)).to(device)
                        total += float(jsd_tokens(logits[j:j+128], ref))
                value = total/(shape[0]*shape[1])
            if not np.isfinite(value):
                raise RuntimeError('Nonfinite AMQ JSD; refusing to hide a failed candidate')
            return value

        all4 = {'linear': {f: [4]*cfg.num_layers for f in PROJ_FAMILIES}}
        sensitivity = {}
        screen_start = time.perf_counter()
        for (i, f), m in mapping.items():
            all4['linear'][f][i] = 2
            sensitivity[f'{i}.{f}'] = score(all4)
            all4['linear'][f][i] = 4
        fixed = [k for k, v in sensitivity.items() if v > 2*np.median(list(sensitivity.values()))]
        sync(device)
        screen_sec = time.perf_counter()-screen_start
        engine = Search()
        engine.config, engine.group_size, engine.predictor = config, cfg.quant_group_size, 'rbf'
        engine.ga_pop_size = settings['population']
        engine.subset_pop_size = 100
        engine.crossover_prob, engine.mut_prob = 0.9, 0.1
        engine.search_space = SearchSpace(config, cfg.num_layers, 7, cfg.quant_group_size, fixed, [2, 3, 4])
        active = len(meta)-len(fixed)
        if active == 0 or settings['initial_samples'] < active+1:
            raise ValueError(f'RBF linear tail needs at least {active+1} initial samples and active variables')

        def checkpoint(iteration):
            sync(device)
            data = {'iteration': iteration, 'archive': archive, 'sensitivity': sensitivity,
                    'fixed_4bit': fixed, 'elapsed_sec': time.perf_counter()-start,
                    'settings': settings, 'seed': seed, 'complete': False}
            temp = directory/'archive.tmp'
            temp.write_text(json.dumps(data, allow_nan=False), encoding='utf-8')
            temp.replace(directory/'archive.json')

        def evaluate(candidates):
            from .amq_vendor.helpers import get_bits_usage
            for arch in candidates:
                if any(arch == x[0] for x in archive):
                    continue
                archive.append((arch, score(arch), get_bits_usage(arch, config, cfg.quant_group_size)))
                if len(archive) % 25 == 0:
                    checkpoint(-1)  # In-progress batch; never a resume boundary.

        checkpoint(-1)
        evaluate(engine.search_space.initialize(settings['initial_samples']))
        if len(archive) < active+1:
            raise RuntimeError('Too few distinct initial candidates for the RBF linear tail')
        checkpoint(0)
        for iteration in range(1, settings['iterations']+1):
            predictor, _ = engine._fit_predictor(archive)
            candidates, _ = engine._next(archive, predictor, settings['candidates_per_iteration'])
            if not candidates:
                raise RuntimeError('AMQ produced no new candidates; inspect checkpoint')
            evaluate(candidates)
            checkpoint(iteration)
            print(f'AMQ seed={seed} iteration={iteration}: {len(archive)} evaluated configurations', flush=True)
        sync(device)
        elapsed = time.perf_counter()-start
        allocations, missing = {}, []
        for budget in budgets:
            candidate = select_candidate(archive, budget, cfg.quant_group_size)
            if candidate is None:
                missing.append(budget)
                continue
            bits = assignment(candidate[0])
            achieved = sum(m['num_params']*bits[m['name']] for m in meta)/config['model_numel']
            allocations[f'{budget:.2f}'] = {'success': True, 'bit_assignment': bits,
                'achieved_avg_bits': achieved, 'bit_counts': dict(Counter(bits.values())),
                'calibration_jsd': candidate[1], 'allocation_sec': 0.0,
                'calibration_to_budget_sec': elapsed}
        result = {'seed': seed, 'method': METHOD, 'k_layers': 0, 'acquired_layers': [],
            'trial_kind': 'external_search', 'probe_label_count': len(sensitivity),
            'configuration_evaluations': len(archive), 'fixed_4bit': fixed,
            'proxy_and_teacher_sec': proxy_sec, 'screening_sec': screen_sec,
            'calibration_shared_sec': elapsed, 'calibration_wall_sec': elapsed,
            'feature_payload_bytes': 0, 'encoder_codebook_bytes': 0,
            'proxy_host_bytes': sum(x.numel()*x.element_size() for x in proxies.values()),
            'teacher_disk_bytes': int(np.prod(shape))*2,
            'allocations': allocations, 'missing_budgets': missing,
            'timing_scope': 'Includes host proxy construction, teacher logits, screening, search and archive I/O; excludes held-out evaluation and model/data load.',
            **memory_finish(device, baseline)}
    finally:
        with torch.no_grad():
            for m in meta:
                if m['name'] in originals:
                    m['module'].weight.copy_(originals[m['name']].to(device))
        proxies.clear()
        if teacher is not None:
            teacher._mmap.close()
            (directory/'teacher_logits.bin').unlink()
    for a in allocations.values():
        a['evaluation'] = evaluate_perplexity(model, eval_ids, meta, cfg,
            bit_assignment=a['bit_assignment'], batch_size=batch_size)
    done = json.loads((directory/'archive.json').read_text())
    done.update(complete=True, missing_budgets=missing)
    (directory/'archive.json').write_text(json.dumps(done, allow_nan=False), encoding='utf-8')
    return result
