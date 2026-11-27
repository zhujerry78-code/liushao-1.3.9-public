"""Inductive Venn-Abers predictor (Vovk & Petej 2014): a calibrated probability interval.

For a test score s, isotonic regression is refitted with (s, 0) and with (s, 1)
added to the calibration set, giving [p0, p1].  The pair is precomputed on a grid
of calibration-score quantiles and interpolated at inference time.
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression

from .conformal import _finite_vector


def _probabilities(values, name):
    array = _finite_vector(values, name)
    if ((array < 0) | (array > 1)).any():
        raise ValueError(name + ' must contain probabilities in [0, 1]')
    return array


def fit_va(cal_scores, cal_y, grid_size=256):
    s, y = _probabilities(cal_scores, 'cal_scores'), _finite_vector(cal_y, 'cal_y')
    if not len(s) or len(s) != len(y) or not np.isin(y, [0, 1]).all():
        raise ValueError('Venn-Abers calibration needs aligned nonempty scores and binary labels')
    if isinstance(grid_size, (bool, np.bool_)) or not isinstance(grid_size, (int, np.integer)) or grid_size < 1:
        raise ValueError('grid_size must be a positive integer')
    grid = np.unique(np.concatenate([[0.0, 1.0], np.quantile(s, np.linspace(0, 1, grid_size))]))
    p0, p1 = np.empty_like(grid), np.empty_like(grid)
    for i, g in enumerate(grid):
        for label, out in ((0.0, p0), (1.0, p1)):
            iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
            iso.fit(np.append(s, g), np.append(y, label))
            out[i] = iso.predict([g])[0]
    return {'grid': grid, 'p0': p0, 'p1': p1}


def predict_va(model, scores):
    scores = _probabilities(scores, 'scores')
    p0 = np.interp(scores, model['grid'], model['p0'])
    p1 = np.interp(scores, model['grid'], model['p1'])
    return np.minimum(p0, p1), np.maximum(p0, p1)


TIE_RESOLUTION = float(np.finfo(np.float64).resolution)


def _grouped(normal_scores, attack_scores):
    """Unique calibration scores with counts and mean label (ties pooled, as isotonic regression does)."""
    normal = _probabilities(normal_scores, 'normal_scores')
    attack = _probabilities(attack_scores, 'attack_scores')
    scores = np.r_[normal, attack]
    labels = np.r_[np.zeros(len(normal)), np.ones(len(attack))]
    if not len(scores):
        raise ValueError('Invalid calibration scores')
    unique, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    return unique, counts, np.bincount(inverse, weights=labels) / counts


def exact_endpoints(unique, counts, means, score):
    """Exact inductive Venn-Abers pair: isotonic refits with (score, 0) and (score, 1) added."""
    x, weights = np.r_[unique, score], np.r_[counts, 1]
    values = []
    for label in (0., 1.):
        iso = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip')
        iso.fit(x, np.r_[means, label], sample_weight=weights)
        values.append(float(iso.predict([score])[0]))
    return min(values), max(values)


def fit_exact_table(normal_scores, attack_scores):
    """Precompute Venn-Abers endpoints at interval representatives and exact ties.

    In the order-based isotonic problem, an inserted point has constant endpoints
    between neighbouring calibration scores.  Scikit-learn additionally pools
    near-equal float64 scores, then interpolates at the query.  A nonidentical
    query within TIE_RESOLUTION of a calibration score can therefore differ from
    this table.  The application inference layer handles those queries by local
    exact refits from the unchanged B1 calibration sample.
    """
    unique, counts, means = _grouped(normal_scores, attack_scores)
    reps = np.r_[unique[0] - 1.0, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1.0]
    gap = np.array([exact_endpoints(unique, counts, means, s) for s in reps])
    tie = np.array([exact_endpoints(unique, counts, means, s) for s in unique])
    return {'method': 'exact-interval-table-v1', 'scores': unique,
            'gap_low': gap[:, 0], 'gap_high': gap[:, 1], 'tie_low': tie[:, 0], 'tie_high': tie[:, 1]}


def predict_exact_table(table, scores):
    values = _probabilities(scores, 'scores')
    unique = table['scores']
    pos = np.searchsorted(unique, values, side='left')
    # Preserve the stored table's near-tie lookup convention.  Snapping alone
    # does not reproduce sklearn's interpolation for nonidentical near queries;
    # the application inference layer uses the original B1 sample to refit them.
    left, right = np.maximum(pos - 1, 0), np.minimum(pos, len(unique) - 1)
    near_left = np.abs(values - unique[left]) < TIE_RESOLUTION
    near_right = np.abs(unique[right] - values) < TIE_RESOLUTION
    tie = near_right | near_left
    tie_pos = np.where(near_right, right, left)
    low = np.where(tie, table['tie_low'][tie_pos], table['gap_low'][pos])
    high = np.where(tie, table['tie_high'][tie_pos], table['gap_high'][pos])
    return low, high
