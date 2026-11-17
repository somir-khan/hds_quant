"""Loss-space diagnostics; no fitting or held-out selection."""
import itertools
import math

import numpy as np
from scipy.stats import spearmanr


def composition_terms(bits, truth, predicted, baseline_nll, observed_nll, processed=None):
    """All loss terms must use the SAME calibration tokens and 4-bit context."""
    raw_sum = math.fsum(truth[n, b] for n, b in bits.items() if b != 4)
    predicted_sum = math.fsum(predicted[n, b] for n, b in bits.items() if b != 4)
    actual = observed_nll-baseline_nll
    out = dict(actual_calibration_delta_nll=actual, raw_singleton_sum=raw_sum,
        predicted_singleton_sum=predicted_sum,
        aggregate_prediction_error=predicted_sum-raw_sum,
        composition_residual=actual-raw_sum,
        actual_minus_predicted=actual-predicted_sum)
    if processed is not None:
        processed_sum = math.fsum(processed[n, b] for n, b in bits.items())
        out.update(processed_objective=processed_sum,
            processing_shift=processed_sum-predicted_sum,
            actual_minus_processed=actual-processed_sum)
    return out


def prediction_metrics(predicted, truth, meta, acquired):
    rows = []
    for b in (2, 3):
        for fam in [None, *range(7)]:
            keys = [(m['name'], b) for m in meta if m['layer_idx'] not in acquired
                    and (fam is None or m['family_idx'] == fam)]
            x = np.asarray([predicted[k] for k in keys]); y = np.asarray([truth[k] for k in keys])
            rho = None
            if len(x) >= 2 and np.ptp(x) > 0 and np.ptp(y) > 0:
                rho = float(spearmanr(x, y).statistic)
            rows.append(dict(probe_bit=b, family='all' if fam is None else fam,
                n=len(keys), mae=float(np.mean(np.abs(x-y))),
                rmse=float(np.sqrt(np.mean((x-y)**2))), spearman=rho))
    return rows


def joint_subsets(names):
    return [list(s) for size in range(len(names)+1) for s in itertools.combinations(names, size)]
