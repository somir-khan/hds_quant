"""Focused representation controls plus a separately identified external LIM baseline."""
import argparse
import copy

from .lim_baseline import METHOD, SOURCE, run_lim_trial
from .run_generalization import run_config as run_shared_config, validate_spec as validate_shared_spec

METHODS = ("HDS-Hybrid", "Coord-Rank-Hist", "Coord-Rank-Exact")
EXTERNAL_SPEC = {"method": METHOD, "allocation_bits": [2, 4],
                 "aggregation": "mean_token_cosine", "calibration": "same_tokens_as_controls"}
PROTOCOL = {
    "schema": "representation_controls_v1",
    "external_baseline": EXTERNAL_SPEC,
    "source": SOURCE,
    "representation_comparison": "HDS 512 salience buckets compressed to D=4096; Exact 512x32 explicit bucket means; Hist 16x32 explicit bucket means. Same metadata/readout settings. This controls resolution, not every normalization detail.",
    "external_comparison": "Independent LIM implementation adapted to HQQ and this run's calibration text; whole-block {2,4} choices versus projection {2,3,4}. Same target-weight payload and held-out evaluation, different feasible sets.",
    "budget_scope": "Parameter-weighted average bits of the seven decoder projections only; excludes embeddings, LM head, scales/zeros, packing overhead. Not measured deployment memory.",
    "label_scope": "K counts blocks with measured sensitivity labels; LIM K=0 uses unlabeled calibration text, not zero calibration data.",
}


def validate_spec(spec):
    base = copy.deepcopy(spec)
    if base.get("schema") != "representation_controls_v1" or base.pop("external_baseline", None) != EXTERNAL_SPEC:
        raise ValueError("Expected representation_controls_v1 with the fixed LIM baseline policy")
    base["schema"] = "generalization_v1"
    validate_shared_spec(base)
    if len(spec["methods"]) != 3 or set(spec["methods"]) != set(METHODS):
        raise ValueError("This control requires HDS, Coord-Rank-Hist and Coord-Rank-Exact")
    if spec["full_reference"] or spec["purpose"] != "benchmark":
        raise ValueError("Focused controls skip full-bank reference acquisition")
    if len(spec["k_layers"]) != 1 or spec["budgets"] != [3.0, 3.5]:
        raise ValueError("Use one K and the fixed 3.0/3.5 budgets for focused controls")
    if len(set(spec["seeds"])) != len(spec["seeds"]):
        raise ValueError("Duplicate seeds")


def run_config(config_path, output_dir, smoke_test=False):
    return run_shared_config(config_path, output_dir, smoke_test,
        spec_validator=validate_spec, extra_trial_runner=run_lim_trial,
        comparison_protocol=copy.deepcopy(PROTOCOL), extra_source_files=(
            "hds_quant/run_representation_controls.py", "hds_quant/lim_baseline.py",
            "hds_quant/config.py", "hds_quant/data_utils.py", "hds_quant/run_experiment.py",
            "scripts/summarize_representation_controls.py", "scripts/summarize_limited_labels.py",
            "scripts/launch_representation_controls.sh", "scripts/run_representation_controls_a100.sh",
            "scripts/load_hf_env.sh"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    run_config(args.config, args.output_dir, args.smoke_test)
