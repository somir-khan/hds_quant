"""Layer-disjoint cross-validation of sensitivity estimators (SPEC §6).

Every projection is predicted by a model that never saw labels from its own transformer layer or from any
layer in the same fold (``fold_id = layer_idx % n_folds``). The ridge penalty of every readout is also chosen with
*layer-grouped* inner CV (``GroupKFold`` over the training layers) with target transform and feature scaling
fitted strictly inside each inner split.

Input structures (as produced by ``label_collector``):
    features_by_proj_bit : {(proj_name, b): {estimator_name: float | torch.Tensor}}   (b in (2, 3))
    delta_loss_by_proj_bit: {(proj_name, b): float}
    meta_list            : [{"name", "layer_idx", "family_idx", "num_params", ...}]
"""

from __future__ import annotations

import math
import time
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.stats import rankdata
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig

MARGINAL_BITS = (2, 3)
REFERENCE_BIT = 4
N_FOLDS = 4
MONO_EPS = 1e-6
RIDGE_ALPHAS = np.logspace(-2, 4, 25)
INNER_CV_FOLDS = 4
MIN_GROUP_ROWS = 8  # (group, bit) models with fewer training rows fall back to the pooled model

UNSUPERVISED_SCALARS = ("Scalar-MSE", "Scalar-L1", "Scalar-TrimmedMean", "Scalar-WinsorL2", "Scalar-Q99")
# "Oracle-Loss" uses monotone-clamped costs; "Oracle-Loss-Raw" keeps the measured delta_loss untouched.
ORACLE_ESTIMATORS = ("Oracle-Loss", "Oracle-Loss-Raw")
SUPERVISED_ESTIMATORS = (
    "Scalar-MSE-Ridge",
    "Scalar-Meta-Ridge",
    "Ordinal-Hist",
    "RandProj-JL",
    "Coord-Rank-Hist",
    "Coord-Rank-Exact",
    "HDS-Rank",
    "HDS-Mag-Strict",
    "HDS-Hybrid",
)
ALL_ESTIMATORS = UNSUPERVISED_SCALARS + ORACLE_ESTIMATORS + SUPERVISED_ESTIMATORS

# Number of low-dimensional scalar / metadata columns (incl. the 2-bit indicator) at the END of the feature
# matrix of estimators that mix a wide block with scalars; see ``_get_column_weights``.
_META_COLS = {
    "HDS-Mag-Strict": 2,
    "HDS-Hybrid": 10,
    "Coord-Rank-Hist": 10,
    "Coord-Rank-Exact": 10,
}

_MAX_EXPM1_ARG = 30.0


def _fit_rank_transform(
    y_train: np.ndarray,
) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:
    """Fit average-rank target transform and its consistent forward/inverse functions.

    Returns ``(y_ranks, forward_fn, inverse_fn)``.
    """
    n = len(y_train)
    if n < 2:
        const_val = float(y_train[0]) if n == 1 else 0.0
        return (
            np.full(n, 0.5, dtype=np.float64),
            lambda v: np.full_like(np.asarray(v, dtype=np.float64), 0.5),
            lambda p: np.full_like(np.asarray(p, dtype=np.float64), const_val),
        )
    y_train_arr = np.asarray(y_train, dtype=np.float64)
    y_ranks = (rankdata(y_train_arr, method="average") - 1.0) / (n - 1.0)
    uniq_y, inv_idx = np.unique(y_train_arr, return_inverse=True)

    if len(uniq_y) == 1:
        const_y = float(uniq_y[0])
        return (
            np.full(n, 0.5, dtype=np.float64),
            lambda v: np.full_like(np.asarray(v, dtype=np.float64), 0.5),
            lambda p: np.full_like(np.asarray(p, dtype=np.float64), const_y),
        )

    uniq_r = np.array([float(y_ranks[inv_idx == u][0]) for u in range(len(uniq_y))], dtype=np.float64)

    def forward_fn(y_val: np.ndarray) -> np.ndarray:
        return np.interp(np.asarray(y_val, dtype=np.float64), uniq_y, uniq_r)

    def inverse_fn(p_pred: np.ndarray) -> np.ndarray:
        p_clipped = np.clip(np.asarray(p_pred, dtype=np.float64), uniq_r[0], uniq_r[-1])
        return np.interp(p_clipped, uniq_r, uniq_y)

    return y_ranks, forward_fn, inverse_fn


# ------------------------------------------------------------------ helpers
def _make_target_transform(mode: str, y_train: np.ndarray) -> Tuple[np.ndarray, Callable[[np.ndarray], np.ndarray]]:
    """Return ``(transformed y_train, inverse)`` fitted on the training fold only."""
    if mode == "raw":
        return y_train.copy(), lambda p: np.asarray(p, dtype=np.float64)
    if mode == "signed_log1p":
        scale = max(float(np.median(np.abs(y_train))), 1e-12)
        y_t = np.sign(y_train) * np.log1p(np.abs(y_train) / scale)

        def inv(p: np.ndarray) -> np.ndarray:
            p_arr = np.asarray(p, dtype=np.float64)
            a = np.minimum(np.abs(p_arr), _MAX_EXPM1_ARG)
            return np.sign(p_arr) * scale * np.expm1(a)

        return y_t, inv
    if mode == "rank":
        y_ranks, forward_fn, inverse_fn = _fit_rank_transform(y_train)
        return y_ranks, inverse_fn
    raise ValueError(f"unknown target_transform {mode!r}")


def _group_key(grouping: str, family_idx: int, b: int):
    if grouping == "pooled":
        return 0
    if grouping == "per_bit":
        return b
    if grouping == "per_family":
        return family_idx
    if grouping == "per_bit_family":
        return (family_idx, b)
    raise ValueError(f"unknown readout_grouping {grouping!r}")


def _get_column_weights(
    n_cols: int,
    estimator: Optional[str] = None,
    col_weights: Optional[np.ndarray] = None,
    hd_dim: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Calculate column upweighting factor sqrt(D / d_meta) for metadata/scalar columns.

    This ensures the wide block (hypervector / explicit histogram) and the scalar metadata block receive equal
    L2 penalty capacity.
    """
    if col_weights is not None:
        return np.asarray(col_weights, dtype=np.float64)
    d_meta = None
    if estimator is not None:
        d_meta = _META_COLS.get(estimator)
    else:
        if hd_dim is not None:
            if n_cols == hd_dim + 2:
                d_meta = 2
            elif n_cols == hd_dim + 10:
                d_meta = 10
        elif n_cols == 4098:
            d_meta = 2
        elif n_cols == 4106:
            d_meta = 10
        elif n_cols == 16394:
            d_meta = 10

    if d_meta is None:
        return None

    d_hd = n_cols - d_meta
    if d_hd <= 0:
        return None
    weights = np.ones(n_cols, dtype=np.float64)
    weights[d_hd:] = math.sqrt(d_hd / d_meta)
    return weights


def _inner_group_splits(
    X: np.ndarray,
    y: np.ndarray,
    groups: Optional[np.ndarray],
    inner_cv_folds: Optional[int] = None,
):
    """Layer-grouped inner CV splits for RidgeCV, or ``None`` if fewer than 2 distinct groups exist."""
    if groups is None:
        return None
    groups = np.asarray(groups)
    n_unique = len(np.unique(groups))
    if n_unique < 2:
        return None
    n_splits = INNER_CV_FOLDS if inner_cv_folds is None else int(inner_cv_folds)
    return list(GroupKFold(n_splits=min(n_splits, n_unique)).split(X, y, groups=groups))


def _fit_ridge(
    X: np.ndarray,
    y_raw: np.ndarray,
    col_weights: Optional[np.ndarray] = None,
    estimator: Optional[str] = None,
    hd_dim: Optional[int] = None,
    groups: Optional[np.ndarray] = None,
    transform_mode: str = "raw",
    alphas: Optional[np.ndarray] = None,
    inner_cv_folds: Optional[int] = None,
) -> Tuple[StandardScaler, Union[Ridge, RidgeCV], Callable[[np.ndarray], np.ndarray]]:
    """Standardize, upweight scalar/metadata columns, and fit Ridge/RidgeCV.

    When ``groups`` has >= 2 unique layers, the ridge penalty is chosen via ``GroupKFold`` over layers,
    with target transform and StandardScaler fitted strictly on each inner training split.
    Returns ``(scaler, ridge, inverse_full)``.
    """
    weights = _get_column_weights(X.shape[1], estimator=estimator, col_weights=col_weights, hd_dim=hd_dim)
    groups = None if groups is None else np.asarray(groups)
    n_unique = 0 if groups is None else len(np.unique(groups))
    alphas_arr = RIDGE_ALPHAS if alphas is None else np.asarray(alphas, dtype=np.float64)
    n_splits = INNER_CV_FOLDS if inner_cv_folds is None else int(inner_cv_folds)

    if groups is not None and n_unique >= 2:
        gkf = GroupKFold(n_splits=min(n_splits, n_unique))
        splits = list(gkf.split(X, y_raw, groups=groups))
        alpha_val_mses = np.zeros((len(splits), len(alphas_arr)), dtype=np.float64)

        for s_idx, (itr, ival) in enumerate(splits):
            y_itr = y_raw[itr]
            y_ival = y_raw[ival]

            # 1. Fit target transform strictly on inner training split
            if transform_mode == "raw":
                y_itr_t = y_itr.copy()
                y_ival_t = y_ival.copy()
            elif transform_mode == "signed_log1p":
                scale_itr = max(float(np.median(np.abs(y_itr))), 1e-12)
                y_itr_t = np.sign(y_itr) * np.log1p(np.abs(y_itr) / scale_itr)
                y_ival_t = np.sign(y_ival) * np.log1p(np.abs(y_ival) / scale_itr)
            elif transform_mode == "rank":
                y_itr_t, forward_itr, _ = _fit_rank_transform(y_itr)
                y_ival_t = forward_itr(y_ival)
            else:
                raise ValueError(f"unknown target_transform {transform_mode!r}")

            # 2. Fit StandardScaler strictly on inner training split
            scaler_inner = StandardScaler().fit(X[itr])
            scaler_inner.col_weights_ = weights
            X_itr_s = scaler_inner.transform(X[itr])
            X_ival_s = scaler_inner.transform(X[ival])
            if weights is not None:
                X_itr_s = X_itr_s * weights
                X_ival_s = X_ival_s * weights

            # 3. Fast exact SVD evaluation of all candidate alphas
            x_mean = X_itr_s.mean(axis=0)
            y_mean = float(y_itr_t.mean())
            X_c = X_itr_s - x_mean
            y_c = y_itr_t - y_mean
            U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
            UT_y = U.T @ y_c

            for a_idx, alpha in enumerate(alphas_arr):
                d = S / (S**2 + alpha)
                beta = Vt.T @ (d * UT_y)
                intercept = y_mean - float(x_mean @ beta)
                val_pred = X_ival_s @ beta + intercept
                alpha_val_mses[s_idx, a_idx] = float(np.mean((val_pred - y_ival_t) ** 2))

        best_alpha = float(alphas_arr[np.argmin(alpha_val_mses.mean(axis=0))])

        # Fit on full training set
        y_full_t, inverse_full = _make_target_transform(transform_mode, y_raw)
        scaler = StandardScaler().fit(X)
        scaler.col_weights_ = weights
        X_s = scaler.transform(X)
        if weights is not None:
            X_s = X_s * weights
        ridge = Ridge(alpha=best_alpha).fit(X_s, y_full_t)
        ridge.alpha_ = float(best_alpha)
        return scaler, ridge, inverse_full
    else:
        y_full_t, inverse_full = _make_target_transform(transform_mode, y_raw)
        scaler = StandardScaler().fit(X)
        scaler.col_weights_ = weights
        X_s = scaler.transform(X)
        if weights is not None:
            X_s = X_s * weights
        ridge = RidgeCV(alphas=alphas_arr).fit(X_s, y_full_t)
        return scaler, ridge, inverse_full


def _model_vector(scaler: StandardScaler, ridge: Union[Ridge, RidgeCV]) -> np.ndarray:
    alpha_val = float(getattr(ridge, "alpha_", getattr(ridge, "alpha", 0.0)))
    return np.concatenate(
        [scaler.mean_, scaler.scale_, np.ravel(ridge.coef_), [float(ridge.intercept_)], [alpha_val]]
    )


def _fit_predict_grouped(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    keys_tr: List,
    X_te: np.ndarray,
    keys_te: List,
    col_weights: Optional[np.ndarray] = None,
    estimator: Optional[str] = None,
    hd_dim: Optional[int] = None,
    groups_tr: Optional[np.ndarray] = None,
    transform_mode: str = "raw",
    alphas: Optional[np.ndarray] = None,
    inner_cv_folds: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fit one scaler+Ridge per group key and predict the test rows (inverted to original delta_loss scale).

    Groups with fewer than 8 training rows fall back to a pooled model fitted on all training rows.
    ``groups_tr`` holds the layer index of every training row (layer-grouped inner CV for alpha selection).
    Returns ``(predictions, flattened model parameters)``.
    """
    keys_tr_arr = np.empty(len(keys_tr), dtype=object)
    keys_tr_arr[:] = keys_tr
    groups_tr = None if groups_tr is None else np.asarray(groups_tr)
    models = {}
    vec_parts = []
    fallback = None
    preds = np.zeros(len(keys_te))
    weights = _get_column_weights(X_tr.shape[1], estimator=estimator, col_weights=col_weights, hd_dim=hd_dim)

    for key in sorted(set(keys_te), key=repr):
        te_idx = [i for i, k in enumerate(keys_te) if k == key]
        tr_idx = [i for i, k in enumerate(keys_tr) if k == key]
        if len(tr_idx) >= MIN_GROUP_ROWS:
            if key not in models:
                g = None if groups_tr is None else groups_tr[tr_idx]
                models[key] = _fit_ridge(
                    X_tr[tr_idx],
                    y_tr[tr_idx],
                    col_weights=weights,
                    estimator=estimator,
                    hd_dim=hd_dim,
                    groups=g,
                    transform_mode=transform_mode,
                    alphas=alphas,
                    inner_cv_folds=inner_cv_folds,
                )
                vec_parts.append(_model_vector(models[key][0], models[key][1]))
            scaler, ridge, inverse = models[key]
        else:
            if fallback is None:
                fallback = _fit_ridge(
                    X_tr,
                    y_tr,
                    col_weights=weights,
                    estimator=estimator,
                    hd_dim=hd_dim,
                    groups=groups_tr,
                    transform_mode=transform_mode,
                    alphas=alphas,
                    inner_cv_folds=inner_cv_folds,
                )
                vec_parts.append(_model_vector(fallback[0], fallback[1]))
            scaler, ridge, inverse = fallback

        X_te_s = scaler.transform(X_te[te_idx])
        if getattr(scaler, "col_weights_", None) is not None:
            X_te_s = X_te_s * scaler.col_weights_
        pred_t = ridge.predict(X_te_s)
        preds[te_idx] = inverse(pred_t)

    return preds, (np.concatenate(vec_parts) if vec_parts else np.zeros(0))


def _alloc_cost(raw2: float, raw3: float) -> Tuple[float, float]:
    """Non-negative, bit-monotone costs: s3 >= 0 and s2 >= s3 + 1e-6."""
    r2 = float(np.nan_to_num(raw2, nan=0.0, posinf=1e30, neginf=0.0))
    r3 = float(np.nan_to_num(raw3, nan=0.0, posinf=1e30, neginf=0.0))
    s3 = max(r3, 0.0)
    s2 = max(max(r2, 0.0), s3 + MONO_EPS)
    return s2, s3


def _base_feature(feats: dict, est: str) -> np.ndarray:
    if est == "Scalar-MSE-Ridge":
        return np.array([math.log(float(feats["Scalar-MSE"]) + 1e-8)], dtype=np.float64)
    return np.asarray(feats[est], dtype=np.float64).reshape(-1)


def _build_alloc_costs(raw: Dict[str, Dict[Tuple[str, int], float]], projs: List[dict], estimators) -> dict:
    """ILP input costs. ``Oracle-Loss-Raw`` is the unmodified measured delta_loss (no clipping, no monotone fix)."""
    alloc: Dict[str, Dict[Tuple[str, int], float]] = {}
    for e in estimators:
        costs: Dict[Tuple[str, int], float] = {}
        for m in projs:
            name = m["name"]
            if e == "Oracle-Loss-Raw":
                s2, s3 = float(raw[e][(name, 2)]), float(raw[e][(name, 3)])
            else:
                s2, s3 = _alloc_cost(raw[e][(name, 2)], raw[e][(name, 3)])
            costs[(name, 2)] = s2
            costs[(name, 3)] = s3
            costs[(name, REFERENCE_BIT)] = 0.0
        alloc[e] = costs
    return alloc


def _prepare_design(
    features_by_proj_bit: dict,
    delta_loss_by_proj_bit: dict,
    meta_list: List[dict],
    config: ExperimentConfig,
    require_all_labels: bool = True,
) -> dict:
    """Shared sample/design-matrix construction for the CV and the fixed-K acquisition experiments."""
    grouping, transform = config.readout_grouping, config.target_transform
    _group_key(grouping, 0, 2)  # validate modes early
    _make_target_transform(transform, np.array([1.0, 2.0]))

    projs = sorted(meta_list, key=lambda m: (m["layer_idx"], m["family_idx"]))
    if require_all_labels:
        projs = [
            m
            for m in projs
            if all((m["name"], b) in features_by_proj_bit and (m["name"], b) in delta_loss_by_proj_bit for b in MARGINAL_BITS)
        ]
        if not projs:
            raise ValueError("no projection has both features and labels for b in (2, 3)")
    else:
        projs = [
            m
            for m in projs
            if all((m["name"], b) in features_by_proj_bit for b in MARGINAL_BITS)
        ]
        if not projs:
            raise ValueError("no projection has features for b in (2, 3)")

    samples = [(m, b) for m in projs for b in MARGINAL_BITS]
    names = [(m["name"], b) for m, b in samples]
    layer = np.array([m["layer_idx"] for m, _ in samples])
    bits = np.array([b for _, b in samples])
    fam = np.array([m["family_idx"] for m, _ in samples])
    y = np.array([
        float(delta_loss_by_proj_bit[k])
        if (k in delta_loss_by_proj_bit and delta_loss_by_proj_bit[k] is not None and np.isfinite(delta_loss_by_proj_bit[k]))
        else np.nan
        for k in names
    ], dtype=np.float64)
    keys = [_group_key(grouping, int(f), int(b)) for f, b in zip(fam, bits)]
    indicator = (bits == 2).astype(np.float64)[:, None]
    X_all = {
        e: np.hstack([np.stack([_base_feature(features_by_proj_bit[k], e) for k in names]), indicator])
        for e in SUPERVISED_ESTIMATORS
    }
    return {
        "projs": projs,
        "names": names,
        "layer": layer,
        "bits": bits,
        "y": y,
        "keys": keys,
        "X_all": X_all,
        "grouping": grouping,
        "transform": transform,
    }


# ------------------------------------------------------------------ main entry point
def cross_validate_estimators(
    features_by_proj_bit: dict,
    delta_loss_by_proj_bit: dict,
    meta_list: List[dict],
    config: ExperimentConfig,
    num_train_layers: Optional[int] = None,
    seed: int = 42,
) -> dict:
    """Predict a sensitivity for every ``(projection, b in {2, 3})`` with layer-disjoint folds."""
    d = _prepare_design(features_by_proj_bit, delta_loss_by_proj_bit, meta_list, config, require_all_labels=True)
    projs, names, layer, y, keys, X_all = d["projs"], d["names"], d["layer"], d["y"], d["keys"], d["X_all"]
    grouping, transform = d["grouping"], d["transform"]

    num_layers = max(m["layer_idx"] for m in projs) + 1
    n_folds = min(int(config.cv_folds), num_layers)
    fold = layer % n_folds

    folds_present = sorted(set(fold.tolist()))
    if len(folds_present) < 2:
        raise ValueError("layer-disjoint CV needs projections from at least 2 folds")

    raw: Dict[str, Dict[Tuple[str, int], float]] = {e: {} for e in ALL_ESTIMATORS}
    readout_time: Dict[str, float] = {e: 0.0 for e in ALL_ESTIMATORS}

    # ---- label-free estimators and the oracles
    for e in UNSUPERVISED_SCALARS:
        for k in names:
            raw[e][k] = float(features_by_proj_bit[k][e])
    for k in names:
        raw["Oracle-Loss"][k] = float(delta_loss_by_proj_bit[k])
        raw["Oracle-Loss-Raw"][k] = float(delta_loss_by_proj_bit[k])

    # ---- supervised estimators
    train_layers_used: Dict[int, List[int]] = {}
    model_coefs: Dict[Tuple[str, int], np.ndarray] = {}
    for k_fold in folds_present:
        test_mask = fold == k_fold
        train_layers = sorted(set(layer[~test_mask].tolist()))
        if num_train_layers is not None and num_train_layers < len(train_layers):
            rng = np.random.RandomState(seed + k_fold)
            train_layers = sorted(int(v) for v in rng.choice(train_layers, size=num_train_layers, replace=False))
        train_layers_used[int(k_fold)] = train_layers
        train_mask = np.isin(layer, train_layers)
        te_idx = np.where(test_mask)[0]
        tr_idx = np.where(train_mask)[0]
        keys_tr = [keys[i] for i in tr_idx]
        keys_te = [keys[i] for i in te_idx]
        y_tr_raw = y[tr_idx]

        for e in SUPERVISED_ESTIMATORS:
            t0 = time.perf_counter()
            pred, vec = _fit_predict_grouped(
                X_all[e][tr_idx],
                y_tr_raw,
                keys_tr,
                X_all[e][te_idx],
                keys_te,
                estimator=e,
                hd_dim=config.hd_dim,
                groups_tr=layer[tr_idx],
                transform_mode=transform,
                alphas=np.asarray(config.ridge_alphas, dtype=np.float64),
                inner_cv_folds=int(config.cv_folds),
            )
            readout_time[e] += time.perf_counter() - t0
            pred = np.nan_to_num(pred, nan=0.0, posinf=1e30, neginf=-1e30)
            for i, p in zip(te_idx, pred):
                raw[e][names[i]] = float(p)
            model_coefs[(e, int(k_fold))] = vec

    alloc = _build_alloc_costs(raw, projs, ALL_ESTIMATORS)

    return {
        "raw_predictions": raw,
        "alloc_costs": alloc,
        "estimator_readout_time_sec": readout_time,
        "fold_of_layer": {int(l): int(l % n_folds) for l in sorted(set(layer.tolist()))},
        "n_folds": n_folds,
        "train_layers_used": train_layers_used,
        "model_coefs": model_coefs,
        "estimators": ALL_ESTIMATORS,
        "readout_grouping": grouping,
        "target_transform": transform,
    }


# ------------------------------------------------------------------ fixed K-layer acquisition
def _select_stratified_layers(layer_ids: List[int], k_layers: int, seed: int) -> List[int]:
    """One layer drawn (with ``seed``) from each of ``k_layers`` contiguous depth strata."""
    k = min(int(k_layers), len(layer_ids))
    rng = np.random.RandomState(seed)
    strata = np.array_split(np.asarray(sorted(layer_ids)), k)
    return sorted(int(rng.choice(s)) for s in strata)


def select_stratified_k_layers(layer_ids: List[int], k_layers: int, seed: int = 42) -> List[int]:
    """Select one layer from each of ``k_layers`` contiguous depth strata."""
    return _select_stratified_layers(layer_ids, k_layers, seed)


def evaluate_fixed_k_layer_acquisition(
    features_by_proj_bit: dict,
    delta_loss_by_proj_bit: dict,
    meta_list: List[dict],
    config: ExperimentConfig,
    k_layers: int,
    seed: int = 42,
) -> dict:
    """Label ONE fixed, depth-stratified subset of ``k_layers`` layers and predict all the others."""
    if int(k_layers) < 1:
        raise ValueError("k_layers must be >= 1")
    d = _prepare_design(features_by_proj_bit, delta_loss_by_proj_bit, meta_list, config, require_all_labels=False)
    projs, names, layer, y, keys, X_all = d["projs"], d["names"], d["layer"], d["y"], d["keys"], d["X_all"]
    transform = d["transform"]

    all_layer_ids = sorted(set(layer.tolist()))
    labeled_layer_ids = sorted({
        int(layer[i])
        for i, k in enumerate(names)
        if k in delta_loss_by_proj_bit and delta_loss_by_proj_bit[k] is not None and np.isfinite(delta_loss_by_proj_bit[k])
    })
    if len(labeled_layer_ids) == int(k_layers):
        acquired = labeled_layer_ids
    else:
        acquired = select_stratified_k_layers(all_layer_ids, int(k_layers), seed)

    for i, k in enumerate(names):
        if int(layer[i]) in acquired:
            if k not in delta_loss_by_proj_bit or delta_loss_by_proj_bit[k] is None or not np.isfinite(delta_loss_by_proj_bit[k]):
                raise ValueError(f"Acquired layer {layer[i]} is missing finite label for {k}")

    unlabeled = [l for l in all_layer_ids if l not in acquired]
    train_mask = np.isin(layer, acquired)
    tr_idx = np.where(train_mask)[0]
    te_idx = np.where(~train_mask)[0]

    keys_tr = [keys[i] for i in tr_idx]
    keys_te = [keys[i] for i in te_idx]
    y_tr_raw = y[tr_idx]

    heldout: Dict[str, Dict[Tuple[str, int], float]] = {e: {} for e in SUPERVISED_ESTIMATORS}
    full: Dict[str, Dict[Tuple[str, int], float]] = {e: {} for e in SUPERVISED_ESTIMATORS}
    readout_time: Dict[str, float] = {e: 0.0 for e in SUPERVISED_ESTIMATORS}

    for e in SUPERVISED_ESTIMATORS:
        for i in tr_idx:
            full[e][names[i]] = float(y[i])  # known measured labels on the acquired layers
        if te_idx.size == 0:
            continue
        t0 = time.perf_counter()
        pred, _ = _fit_predict_grouped(
            X_all[e][tr_idx],
            y_tr_raw,
            keys_tr,
            X_all[e][te_idx],
            keys_te,
            estimator=e,
            hd_dim=config.hd_dim,
            groups_tr=layer[tr_idx],
            transform_mode=transform,
            alphas=np.asarray(config.ridge_alphas, dtype=np.float64),
            inner_cv_folds=int(config.cv_folds),
        )
        readout_time[e] += time.perf_counter() - t0
        pred = np.nan_to_num(pred, nan=0.0, posinf=1e30, neginf=-1e30)
        for i, p in zip(te_idx, pred):
            heldout[e][names[i]] = float(p)
            full[e][names[i]] = float(p)

    return {
        "k_layers": int(k_layers),
        "acquired_layers": acquired,
        "unlabeled_layers": unlabeled,
        "n_labeled_layers": len(acquired),
        "num_layers": len(all_layer_ids),
        "acquisition_fraction": len(acquired) / len(all_layer_ids),
        "heldout_predictions": heldout,
        "raw_predictions": full,
        "alloc_costs": _build_alloc_costs(full, projs, SUPERVISED_ESTIMATORS),
        "estimator_readout_time_sec": readout_time,
        "estimators": SUPERVISED_ESTIMATORS,
        "readout_grouping": d["grouping"],
        "target_transform": transform,
    }
