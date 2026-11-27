"""ECOD: unsupervised outlier score from empirical cumulative distribution tails.

Li et al., "ECOD: Unsupervised Outlier Detection Using Empirical Cumulative
Distribution Functions", IEEE TKDE 2022.  Fitted on training-split normal
traffic only, so the score never sees attack labels.
"""
import numpy as np


def _feature_matrix(values, name, *, require_rows=False):
    """Validate raw feature layout before missing-value imputation can broadcast it."""
    if np.ma.isMaskedArray(values) and np.ma.getmaskarray(values).any():
        raise ValueError(name + ' must not contain masked values')
    try:
        array = np.asarray(values)
        if array.ndim != 2 or not array.shape[1] or array.dtype.kind not in 'biuf':
            raise ValueError
        array = array.astype(np.float64, copy=False)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(name + ' must be a real numeric feature matrix') from exc
    if require_rows and not len(array):
        raise ValueError(name + ' must contain at least one row')
    return array


def fit_ecod(x_normal, max_rows=20000, seed=42):
    x = _feature_matrix(x_normal, 'x_normal', require_rows=True)
    if isinstance(max_rows, (bool, np.bool_)) or not isinstance(max_rows, (int, np.integer)) or max_rows < 1:
        raise ValueError('max_rows must be a positive integer')
    if len(x) > max_rows:
        x = x[np.random.default_rng(seed).choice(len(x), max_rows, replace=False)]
    median = np.nanmedian(x, axis=0)
    median = np.where(np.isfinite(median), median, 0.0)
    x = np.where(np.isfinite(x), x, median)
    # Skewness picks the "natural" tail per feature (ECOD's automatic variant).
    centered = x - x.mean(axis=0)
    std = x.std(axis=0)
    skew = np.divide((centered ** 3).mean(axis=0), std ** 3, out=np.zeros_like(std), where=std > 0)
    return {'sorted': np.sort(x, axis=0), 'median': median, 'skew_sign': np.sign(skew)}


def tail_logs(model, x):
    """Per-feature -log left tail and -log right tail, with 1/(n+1) smoothing."""
    x = _feature_matrix(x, 'x')
    ref = np.asarray(model['sorted'])
    median = np.asarray(model['median'])
    if (ref.ndim != 2 or not len(ref) or ref.shape[1] != x.shape[1]
            or median.shape != (x.shape[1],)):
        raise ValueError('ECOD reference, median and query feature dimensions must agree')
    x = np.where(np.isfinite(x), x, median)
    n = len(ref)
    left = np.empty_like(x)
    right = np.empty_like(x)
    for j in range(x.shape[1]):
        left[:, j] = np.searchsorted(ref[:, j], x[:, j], side='right')
        right[:, j] = n - np.searchsorted(ref[:, j], x[:, j], side='left')
    return -np.log((left + 1) / (n + 1)), -np.log((right + 1) / (n + 1))


def _sum_tail_logs(left, right, skew_sign):
    auto = np.where(skew_sign < 0, left, right)
    return np.maximum.reduce([left.sum(axis=1), right.sum(axis=1), auto.sum(axis=1)])


def score_ecod(model, x):
    """O(x) = max(sum left, sum right, sum auto); larger means more outlying."""
    left, right = tail_logs(model, x)
    return _sum_tail_logs(left, right, model['skew_sign'])
