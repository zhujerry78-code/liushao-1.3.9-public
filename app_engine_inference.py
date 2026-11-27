"""Exact Venn-Abers endpoints from the unchanged frozen B calibration sample.

The stored conformal arrays retain class-conditional B nonconformity scores.
For normal records alpha=BetP; for attacks alpha=1-BetP.  Recovering those
labelled scores permits exact Venn-Abers inference without retraining, choosing
a new split, or changing any calibration parameter.

Format 2 engines (1.3.8) ship the same exact endpoints precomputed for every
interval between calibration scores (liushao_engine.venn_abers.fit_exact_table);
Ordinary gaps and exact knots use table lookup.  Queries very close to a knot
can be interpolated differently after sklearn pools near-equal scores, so they
use the same cached per-score refit as format 1.
"""
from functools import lru_cache
import hashlib
import json
from threading import Lock

import numpy as np
from sklearn.isotonic import IsotonicRegression

from liushao_engine.conformal import _finite_vector

ENGINE_INFERENCE_VERSION = 'r13-exact-va-v1'
TABLE_INFERENCE_VERSION = 'r138-exact-va-table-v1'
EFFECTIVE_BATCH_COLUMN = '_engine_effective_batch'


def inference_version(engine):
    return TABLE_INFERENCE_VERSION if 'venn_abers_exact' in engine else ENGINE_INFERENCE_VERSION


class ExactVennAbers:
    def __init__(self, conformal):
        normal = _finite_vector(conformal['normal'], 'normal calibration')
        attack = 1 - _finite_vector(conformal['attack'], 'attack calibration')
        scores = np.r_[normal, attack]
        labels = np.r_[np.zeros(len(normal)), np.ones(len(attack))]
        if not len(scores) or not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
            raise ValueError('Invalid frozen calibration scores')
        self.scores, inverse, self.weights = np.unique(scores, return_inverse=True, return_counts=True)
        self.labels = np.bincount(inverse, weights=labels) / self.weights
        self.endpoints = lru_cache(maxsize=8192)(self._endpoints)

    def _endpoints(self, score):
        # Grouping identical X and using counts as sample_weight preserves the
        # two original isotonic regression problems, including exact ties.
        x = np.r_[self.scores, score]
        weights = np.r_[self.weights, 1]
        values = []
        for label in (0., 1.):
            iso = IsotonicRegression(y_min=0., y_max=1., out_of_bounds='clip')
            iso.fit(x, np.r_[self.labels, label], sample_weight=weights)
            values.append(float(iso.predict([score])[0]))
        return min(values), max(values)

    def predict(self, scores):
        values = _finite_vector(scores, 'Venn-Abers query scores')
        if values.ndim != 1 or not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError('Venn-Abers query scores must be finite probabilities')
        unique, inverse = np.unique(values, return_inverse=True)
        if not len(unique):
            return values.copy(), values.copy()
        pairs = np.asarray([self.endpoints(float(score)) for score in unique])
        return pairs[inverse, 0], pairs[inverse, 1]


_VA_CACHE = {}
_VA_CACHE_LOCK = Lock()


def _cached_va(conformal):
    # Validate before hashing too: a masked version must not reuse a valid cache entry.
    calibration = {name: _finite_vector(conformal[name], name + ' calibration')
                   for name in ('normal', 'attack')}
    digest = hashlib.sha256()
    for name, array in calibration.items():
        values = np.asarray(array, dtype='<f8')
        digest.update(name.encode()); digest.update(str(values.shape).encode()); digest.update(values.tobytes())
    key = digest.hexdigest()
    with _VA_CACHE_LOCK:
        if key not in _VA_CACHE:
            if len(_VA_CACHE) >= 2:
                _VA_CACHE.pop(next(iter(_VA_CACHE)))
            _VA_CACHE[key] = ExactVennAbers(calibration)
        return _VA_CACHE[key]


def exact_va(engine, scores):
    values = _finite_vector(scores, 'Venn-Abers query scores')
    if 'venn_abers_exact' in engine:
        from liushao_engine.venn_abers import TIE_RESOLUTION, predict_exact_table
        table = engine['venn_abers_exact']
        low, high = predict_exact_table(table, values)
        unique = np.asarray(table['scores'], dtype=np.float64)
        pos = np.searchsorted(unique, values, side='left')
        left, right = np.maximum(pos - 1, 0), np.minimum(pos, len(unique) - 1)
        exact = (values == unique[left]) | (values == unique[right])
        near = ((np.abs(values - unique[left]) < TIE_RESOLUTION) |
                (np.abs(unique[right] - values) < TIE_RESOLUTION)) & ~exact
        if near.any():
            calibration = engine.get('venn_abers_calibration')
            if calibration is None:
                # Historical format-2 legacy frames stored VA's original sample in conformal.
                # Open frames use a separate B1: their B2 sample is never a VA substitute.
                calibration = engine.get('conformal', {})
                if ('frame' in engine.get('fusion', {}) or 'scores' in calibration or
                        not {'normal', 'attack'} <= calibration.keys()):
                    raise ValueError('Near-knot Venn-Abers queries require the original B1 calibration sample')
            low[near], high[near] = _cached_va(calibration).predict(values[near])
        return low, high
    return _cached_va(engine['conformal']).predict(values)


def analyze_current_engine(pred, features, bundle, engine, *, alpha=None, return_used=False):
    import liushao_engine
    result, used = liushao_engine.analyze(pred, features, bundle, engine, alpha=alpha, return_used=True)
    low, high = exact_va(used, result.ds_betp.to_numpy(dtype=float))
    result['va_low'], result['va_high'] = low, high
    detected = used.get('_batch_drift')
    if detected is not None:
        digest = hashlib.sha256(b'liushao-effective-calibration-v1;')
        parameters = ('fusion', 'conflict_cut', 'conformal', 'venn_abers', 'venn_abers_calibration',
                      'venn_abers_exact')
        liushao_engine._canonical({name: used[name] for name in parameters}, digest)
        # These ordered inputs are exactly the three scores used for batch adaptation.
        scores = result[['rf_score', 'lr_score', 'typ_score']].to_numpy(dtype='<f8')
        provenance = {'method_version': 'effective-batch-v1', 'calibration_sha256': digest.hexdigest(),
                      'expert_scores_sha256': hashlib.sha256(scores.tobytes()).hexdigest(),
                      'discount_factors': detected['factors'], 'rows': len(result),
                      'drift': {name: detected[name] for name in
                                ('candidate', 'size_class', 'rows_used', 'delta', 'threshold')}}
        result[EFFECTIVE_BATCH_COLUMN] = json.dumps(provenance, sort_keys=True, allow_nan=False)
    elif EFFECTIVE_BATCH_COLUMN in result:
        del result[EFFECTIVE_BATCH_COLUMN]
    return (result, used) if return_used else result
