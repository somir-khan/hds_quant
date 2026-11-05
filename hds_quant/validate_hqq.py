"""Compare fake quantization with official packed HQQLinear on identical tokens.

No calibration or allocation is run. Use --smoke-test for an offline synthetic
model; real-model runs default to CUDA and WikiText-2 test, without truncation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from hqq.core.quantize import BaseQuantizeConfig, HQQLinear
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import ExperimentConfig
from .data_utils import get_eval_tokens
from .evaluator import evaluate_perplexity
from .label_collector import discover_target_projections
from .quantizer import get_effective_hqq_config, quantize_weight_hqq
from .run_experiment import _build_smoke_model


@torch.no_grad()
def validate(args):
    if args.device == "cuda" and not torch.cuda.is_available() and not args.smoke_test:
        raise RuntimeError("CUDA is unavailable; use --device cpu explicitly for CPU validation.")
    if args.max_eval_chunks < 0 or args.batch_size <= 0 or args.nll_atol < 0:
        raise ValueError("Invalid evaluation count, batch size, or tolerance")
    if args.smoke_test:
        model, ids = _build_smoke_model()
        model_id = "synthetic_smoke_model"
    else:
        model_id = args.model_id
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float16 if args.device == "cuda" else torch.float32
        ).to(args.device)
        ids = get_eval_tokens(tokenizer, args.eval_dataset, args.eval_seq_len)
    model.eval()
    if args.max_eval_chunks:
        ids = ids[:args.max_eval_chunks]
    meta = discover_target_projections(model)
    if not meta:
        raise ValueError("No supported transformer projections found")
    param = next(model.parameters())
    originals = {m["name"]: m["module"].weight.detach().cpu().clone() for m in meta}
    cfg = ExperimentConfig(quantizer="hqq")
    baseline = evaluate_perplexity(model, ids, meta, cfg, orig_weights_cpu=originals, batch_size=args.batch_size)
    report = {"model_id": model_id, "args": vars(args), "device": str(param.device),
              "dtype": str(param.dtype), "torch_version": torch.__version__,
              "n_eval_chunks": len(ids), "scored_tokens": sum(baseline["seq_token_counts"]),
              "projection_count": len(meta), "unquantized": baseline, "comparisons": []}
    for axis in args.axes:
        for group in args.group_sizes:
            if group <= 0:
                raise ValueError("group sizes must be positive")
            for bits in args.bits:
                cfg = ExperimentConfig(quantizer="hqq", hqq_axis=axis, quant_group_size=group)
                fake = evaluate_perplexity(model, ids, meta, cfg, uniform_bit=bits,
                                          orig_weights_cpu=originals, batch_size=args.batch_size)
                max_weight_error = 0.0
                try:
                    for m in meta:
                        layer = m["module"]
                        packed_config = BaseQuantizeConfig(nbits=bits, group_size=group, axis=axis)
                        # Match the experiment's explicit floating-zero policy.
                        # BaseQuantizeConfig otherwise defaults to round_zero=True at 4 bits.
                        packed_config["weight_quant_params"]["round_zero"] = False
                        official = HQQLinear(layer, packed_config,
                                             del_orig=False, compute_dtype=param.dtype, device=str(param.device))
                        expected = quantize_weight_hqq(originals[m["name"]].to(param.device), bits,
                                                       group_size=group, axis=axis)
                        max_weight_error = max(max_weight_error, float((official.dequantize().float() - expected.float()).abs().max()))
                        model.set_submodule(m["name"], official)
                    # Empty projection list: evaluate the installed HQQLinear modules directly.
                    packed = evaluate_perplexity(model, ids, [], cfg, batch_size=args.batch_size)
                finally:
                    for m in meta:
                        model.set_submodule(m["name"], m["module"])
                gap = abs(fake["nll"] - packed["nll"])
                row = {"bits": bits, "hqq_config": get_effective_hqq_config(group, param.device, param.dtype, axis=axis),
                       "fake": fake, "packed_hqqlinear": packed, "max_weight_abs_error": max_weight_error,
                       "nll_abs_difference": gap, "nll_parity_pass": gap <= args.nll_atol}
                report["comparisons"].append(row)
                print(f"axis={axis} group={group} bits={bits}: fake PPL={fake['ppl']:.6f}, packed PPL={packed['ppl']:.6f}, NLL gap={gap:.3g}", flush=True)
    report["all_nll_parity_pass"] = all(r["nll_parity_pass"] for r in report["comparisons"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if not report["all_nll_parity_pass"]:
        raise RuntimeError(f"HQQ NLL parity failed; inspect {output}")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-id", default="meta-llama/Llama-3.2-1B-Instruct")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--axes", nargs="+", type=int, choices=[0, 1], default=[0, 1])
    p.add_argument("--group-sizes", nargs="+", type=int, default=[128])
    p.add_argument("--bits", nargs="+", type=int, choices=[1, 2, 3, 4], default=[2, 3, 4])
    p.add_argument("--eval-dataset", choices=["wikitext2", "c4"], default="wikitext2")
    p.add_argument("--eval-seq-len", type=int, default=2048)
    p.add_argument("--max-eval-chunks", type=int, default=0, help="0 means all chunks; positive values are diagnostic subsets.")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--nll-atol", type=float, default=0.005, help="Absolute NLL tolerance; reported explicitly, not exact equality.")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--output", default="results_hqq_axis0/hqq_validation.json")
    validate(p.parse_args())


if __name__ == "__main__":
    main()
