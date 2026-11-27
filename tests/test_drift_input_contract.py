"""Reject invalid drift inputs before dtype conversion can erase their meaning."""
import numpy as np
import pandas as pd
import pytest

from liushao_engine import drift


@pytest.fixture(scope='module')
def inputs():
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 300)
    z = np.column_stack([np.where(y, .9, .1), np.where(y, .8, .2), rng.random(600)])
    blocks = np.repeat(['a', 'b'], 300)
    return z, y, blocks


def fit(z, y, blocks, threshold=.5):
    return drift.fit_drift(z, y, blocks, rf_threshold=threshold,
                           candidates=('tail_h_adj',), sizes=(200,), reps=2)


@pytest.fixture(scope='module')
def model(inputs):
    return fit(*inputs)


@pytest.mark.parametrize('make_bad', [
    pytest.param(lambda z: np.ma.array(z, mask=np.ones_like(z, dtype=bool)), id='masked'),
    pytest.param(lambda z: z.astype(complex) + 1j, id='complex'),
    pytest.param(lambda z: z.astype(str), id='numeric-strings'),
])
@pytest.mark.parametrize('entry', ['fit', 'detect'])
def test_scores_are_rejected_before_lossy_conversion(inputs, model, make_bad, entry):
    z, y, blocks = inputs
    with pytest.raises(ValueError):
        if entry == 'fit':
            fit(make_bad(z), y, blocks)
        else:
            drift.detect_drift(model, make_bad(z))


@pytest.mark.parametrize('make_bad', [
    pytest.param(lambda y: y + .1, id='fractional'),
    pytest.param(lambda y: np.where(y, 2, 0), id='nonbinary'),
    pytest.param(lambda y: np.where(y, np.nan, 0), id='missing'),
    pytest.param(lambda y: np.ma.array(y, mask=y.astype(bool)), id='masked'),
    pytest.param(lambda y: y.astype(complex) + 1j, id='complex'),
    pytest.param(lambda y: y.astype(str), id='numeric-strings'),
    pytest.param(lambda y: y[:, None], id='column-vector'),
    pytest.param(lambda y: y[:-1], id='short'),
    pytest.param(lambda y: np.r_[y, 0], id='long'),
])
def test_reference_labels_are_binary_one_dimensional_and_aligned(inputs, make_bad):
    z, y, blocks = inputs
    with pytest.raises(ValueError):
        fit(z, make_bad(y), blocks)


@pytest.mark.parametrize('make_bad', [
    pytest.param(lambda b: b[:-1], id='short'),
    pytest.param(lambda b: np.r_[b, 'b'], id='long'),
    pytest.param(lambda b: b[:, None], id='column-vector'),
    pytest.param(lambda b: np.array([None] * 300 + ['b'] * 300, dtype=object), id='none'),
    pytest.param(lambda b: np.array([np.nan] * 300 + ['b'] * 300, dtype=object), id='nan'),
    pytest.param(lambda b: ['a'] * 300 + [np.nan] * 300, id='mixed-list-nan'),
    pytest.param(lambda b: np.array([pd.NA] * 300 + ['b'] * 300, dtype=object), id='pandas-na'),
    pytest.param(lambda b: np.ma.array(b, mask=np.arange(len(b)) < 300), id='masked'),
])
def test_time_blocks_are_one_dimensional_aligned_and_present(inputs, make_bad):
    z, y, blocks = inputs
    with pytest.raises(ValueError):
        fit(z, y, make_bad(blocks))


@pytest.mark.parametrize('threshold', [np.nan, np.inf, -np.inf, .5 + 1j, np.array([.5])])
def test_rf_threshold_must_be_a_finite_real_scalar(inputs, threshold):
    with pytest.raises(ValueError):
        fit(*inputs, threshold=threshold)


@pytest.mark.parametrize('make_bad', [lambda y: y[:-1], lambda y: y + .1])
def test_direct_reference_stats_validates_labels(inputs, make_bad):
    z, y, _ = inputs
    with pytest.raises(ValueError):
        drift.reference_stats(z, make_bad(y), .5)


@pytest.mark.parametrize('blocks', [np.array(['a', 'b'])[:, None], [None, 'b']])
def test_direct_usable_blocks_rejects_missing_or_nonvector_ids(blocks):
    with pytest.raises(ValueError):
        drift.usable_blocks(blocks, min_rows=1)


def test_valid_lists_binary_booleans_numeric_blocks_and_empty_queries_remain_supported(inputs):
    z, y, _ = inputs
    m = fit(z.tolist(), y.astype(bool), np.repeat([1, 2], 300))
    assert m['blocks'] == {'1': 300, '2': 300}
    assert m['reference']['pi'] == .5
    assert drift.detect_drift(m, z[:0]) == {
        'status': 'too_small', 'candidate': 'tail_h_adj', 'rows': 0, 'drift': None}
    assert drift.detect_drift(m, np.ma.array(z, mask=False)) == drift.detect_drift(m, z)
