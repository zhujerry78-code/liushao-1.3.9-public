"""Kernel two-sample test with maximum mean discrepancy (Gretton et al. 2012)."""
import numpy as np
from scipy.spatial.distance import cdist

from .ecod import _feature_matrix


def _sq_dists(z):
    # Direct differences preserve small separations after a large common shift;
    # ||x||^2 + ||y||^2 - 2 x.y can cancel those separations completely.
    d = cdist(z, z, metric='sqeuclidean')
    if not np.isfinite(d).all():
        raise ValueError('MMD squared distances must be finite')
    return d


def mmd_test(x, y, n_perm=200, max_rows=2000, seed=42):
    """RBF kernel, median-heuristic bandwidth, permutation p-value of the biased MMD^2."""
    for name, value, minimum in (('n_perm', n_perm, 0), ('max_rows', max_rows, 1)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(name + ' must be an integer >= ' + str(minimum))
    x, y = _feature_matrix(x, 'x', require_rows=True), _feature_matrix(y, 'y', require_rows=True)
    if x.shape[1] != y.shape[1] or not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError('MMD samples must be finite and have matching feature dimensions')
    rng = np.random.default_rng(seed)
    if len(x) > max_rows:
        x = x[rng.choice(len(x), max_rows, replace=False)]
    if len(y) > max_rows:
        y = y[rng.choice(len(y), max_rows, replace=False)]
    z = np.vstack([x, y])
    d = _sq_dists(z)
    upper = d[np.triu_indices(len(z), k=1)]
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        bandwidth = float(np.sqrt(np.median(upper[upper > 0]) / 2)) if (upper > 0).any() else 1.0
        scale = 2 * np.float64(bandwidth) ** 2
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('MMD kernel bandwidth must have finite positive squared scale')
    k = np.exp(-d / scale)
    # MMD^2 = w' K w with w = +1/n on the first sample and -1/m on the second.
    base = np.concatenate([np.full(len(x), 1 / len(x)), np.full(len(y), -1 / len(y))])

    def stat(w):
        return float(w @ k @ w)

    observed = stat(base)
    null = np.array([stat(rng.permutation(base)) for _ in range(n_perm)])
    if not np.isfinite(observed) or not np.isfinite(null).all():
        raise ValueError('MMD observed and permutation statistics must be finite')
    return {'mmd2': float(observed), 'p_value': float(((null >= observed).sum() + 1) / (n_perm + 1)),
            'bandwidth': bandwidth, 'rows': [int(len(x)), int(len(y))], 'n_perm': int(n_perm)}
