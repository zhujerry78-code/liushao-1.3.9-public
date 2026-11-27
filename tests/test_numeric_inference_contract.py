"""Synthetic regressions for inference input contracts and MMD distance arithmetic."""
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

import liushao_engine as engine_api
from liushao_engine.ecod import fit_ecod, score_ecod, tail_logs
from liushao_engine.mmd import _sq_dists, mmd_test
from liushao_engine.typicality import aggregate, fit_typicality, p_values, score_typicality
from liushao_engine.venn_abers import fit_exact_table, fit_va, predict_exact_table, predict_va


def ecod_reference():
    return {'sorted': np.array([[0., 0.], [1., 2.], [2., 4.]]),
            'median': np.array([1., 2.]), 'skew_sign': np.array([1., -1.])}


@pytest.mark.parametrize('bad', [np.array([]), np.empty((0, 2)), np.empty((2, 0)),
                                np.ones((2, 2, 1)), [[1j]], [[1 + 0j]],
                                np.ma.array([[100.]], mask=True)])
def test_ecod_fit_rejects_invalid_reference(bad):
    with pytest.raises(ValueError):
        fit_ecod(bad)


@pytest.mark.parametrize('max_rows', [0, -1, 1.5, True])
def test_ecod_rejects_invalid_sample_cap(max_rows):
    with pytest.raises(ValueError):
        fit_ecod([[1.], [2.]], max_rows=max_rows)


@pytest.mark.parametrize('query', [[[1.5]], [[1., 2., 3.]], [1., 2.],
                                  [[1j, 2.]], np.ma.array([[100., 100.]], mask=True)])
def test_ecod_query_cannot_broadcast_or_discard_values(query):
    with pytest.raises(ValueError):
        score_ecod(ecod_reference(), query)


def test_ecod_preserves_nonfinite_feature_imputation_and_empty_query():
    model = ecod_reference()
    np.testing.assert_array_equal(score_ecod(model, [[np.nan, np.inf]]), score_ecod(model, [[1., 2.]]))
    np.testing.assert_array_equal(score_ecod(model, np.empty((0, 2))), np.empty(0))
    np.testing.assert_array_equal(score_ecod(model, [[1.5, 1.5]]), [1.3862943611198906])
    left, right = tail_logs(model, [[1.5, 1.5]])
    np.testing.assert_array_equal(left, -np.log([[3 / 4, 2 / 4]]))
    np.testing.assert_array_equal(right, -np.log([[2 / 4, 3 / 4]]))


@pytest.mark.parametrize('bad', [[np.nan], [np.inf], [1j], [[1.]],
                                np.ma.array([100.], mask=True)])
@pytest.mark.parametrize('location', ['calibration', 'query'])
def test_typicality_p_rejects_invalid_vectors(bad, location):
    args = (bad, [1.]) if location == 'calibration' else ([0., 1.], bad)
    with pytest.raises(ValueError):
        p_values(*args)


def test_typicality_p_keeps_ties_unsorted_calibration_and_empty_contract():
    np.testing.assert_array_equal(p_values([2., 0., 2., 3.], [-1., 0., 1., 2., 3., 4.]),
                                  [1., 1., .8, .8, .4, .2])
    np.testing.assert_array_equal(p_values([], [0., 2.]), [1., 1.])
    assert p_values([0.], []).shape == (0,)


@pytest.mark.parametrize('k', [0, -1, 1.5, True, np.bool_(True)])
def test_typicality_topk_rejects_nonpositive_or_noninteger_k(k):
    with pytest.raises(ValueError):
        aggregate(np.ones((1, 3)), np.zeros((1, 3)), np.ones(3), ('topk', k))


def test_typicality_topk_preserves_selected_rule_and_oversized_slice():
    left, right = np.array([[1., 3., 2., 4.]]), np.zeros((1, 4))
    np.testing.assert_array_equal(aggregate(left, right, np.ones(4), ('topk', 3)), [3.])
    np.testing.assert_array_equal(aggregate(left, right, np.ones(4), ('topk', 10)), [2.5])


def test_typicality_entrypoints_preserve_masks_until_validation():
    ref = np.arange(12.).reshape(6, 2)
    with pytest.raises(ValueError):
        fit_typicality(np.ma.array(ref, mask=True), max_rows=4)
    model = fit_typicality(ref, max_rows=4)
    for bad in [np.ma.array([[100., 100.]], mask=True), [[1j, 2.]], [[1.]]]:
        with pytest.raises(ValueError):
            score_typicality(model, bad)


def test_open_analysis_reuses_feature_tails_without_changing_scores(monkeypatch):
    ecod = ecod_reference()
    typicality = {'how': ('topk', 2), 'calibration': [1., 0., 2.]}
    engine = {'format': 2, 'model_version': 'test', 'fusion': {'frame': 'open'},
              'ecod': ecod, 'typicality': typicality}
    x = np.array([[1., 2.], [np.nan, np.inf], [-1., 5.]])
    pred = pd.DataFrame({'rf_score': [.2, .4, .6], 'lr_score': [.3, .5, .7]},
                        index=[6, 3, 8])
    expected_ecod = score_ecod(ecod, x)
    expected_typicality = score_typicality(dict(typicality, tail=ecod), x)
    searches = []
    searchsorted = np.searchsorted

    def count_reference_search(a, v, *args, **kwargs):
        if np.shares_memory(a, ecod['sorted']):
            searches.append(1)
        return searchsorted(a, v, *args, **kwargs)

    def fuse_scores(scores, engine, *, alpha):
        values = dict.fromkeys(engine_api.COLUMNS + engine_api.OPEN_COLUMNS, np.nan)
        values['typ_score'] = scores['typ']
        return values

    monkeypatch.setattr(np, 'searchsorted', count_reference_search)
    monkeypatch.setattr(engine_api, 'analyze_scores', fuse_scores)
    result = engine_api.analyze(pred, x, {'model_version': 'test'}, engine)
    assert len(searches) == 2 * x.shape[1]
    np.testing.assert_array_equal(result.ecod_score, expected_ecod)
    np.testing.assert_array_equal(result.typ_score, expected_typicality)
    pd.testing.assert_frame_equal(result[pred.columns], pred)


@pytest.mark.parametrize('bad', [[[np.nan]], [[np.inf]], [[1j]], [[1 + 0j]],
                                np.empty((0, 1)), np.empty((2, 0)), [1., 2.],
                                np.ma.array([[100.]], mask=True)])
def test_mmd_rejects_invalid_samples(bad):
    with pytest.raises(ValueError):
        mmd_test(bad, [[1.], [2.]], n_perm=1)


@pytest.mark.parametrize('name,value', [('n_perm', -2), ('n_perm', 1.5), ('n_perm', True),
                                      ('max_rows', 0), ('max_rows', -1), ('max_rows', False)])
def test_mmd_rejects_invalid_integer_options(name, value):
    with pytest.raises(ValueError):
        mmd_test([[0.], [1.]], [[3.], [4.]], **{name: value})


def test_mmd_rejects_dimension_mismatch_and_finite_overflow():
    with pytest.raises(ValueError):
        mmd_test([[0., 1.]], [[2.]], n_perm=1)
    with pytest.raises(ValueError, match='squared distances'):
        mmd_test([[0.], [1e200]], [[3e200], [4e200]], n_perm=1)


def test_mmd_zero_permutations_and_identical_samples_remain_valid():
    out = mmd_test([[1.], [1.]], [[1.], [1.]], n_perm=0)
    assert out == {'mmd2': 0., 'p_value': 1., 'bandwidth': 1., 'rows': [2, 2], 'n_perm': 0}


def test_mmd_distances_match_independent_exact_integer_oracle_after_translation():
    z = np.array([[0., 0.], [1., 2.], [3., 1.], [4., -2.]]) + 1e9
    exact = [[float(sum((Fraction(float(a)) - Fraction(float(b))) ** 2 for a, b in zip(x, y)))
              for y in z] for x in z]
    np.testing.assert_array_equal(_sq_dists(z), exact)
    np.testing.assert_array_equal(_sq_dists(z), _sq_dists(z - 1e9))


def test_mmd_full_test_is_translation_invariant_for_representable_offsets():
    x, y = np.array([[0.], [1.]]), np.array([[3.], [4.]])
    original = mmd_test(x, y, n_perm=19)
    assert mmd_test(x + 1e9, y + 1e9, n_perm=19) == original
    assert original['p_value'] == .6
    assert original['mmd2'] == pytest.approx(1.2941155174055756, abs=1e-15)


@pytest.mark.parametrize('bad', [[np.nan], [np.inf], [1j], [[.2]],
                                np.ma.array([.2], mask=True), [-.1], [1.1]])
@pytest.mark.parametrize('method', ['grid', 'exact'])
def test_va_prediction_rejects_invalid_queries(bad, method):
    model = fit_va([.2, .8], [0, 1], grid_size=2) if method == 'grid' else fit_exact_table([.2], [.8])
    predict = predict_va if method == 'grid' else predict_exact_table
    with pytest.raises(ValueError):
        predict(model, bad)


@pytest.mark.parametrize('scores,labels', [([[.2], [.8]], [[0], [1]]), ([.2, .8], [0, .5]),
                                         ([.2], [0, 1]), ([], []),
                                         (np.ma.array([.2, .8], mask=True), [0, 1]),
                                         ([.2, .8], np.ma.array([0, 1], mask=True))])
def test_va_fit_rejects_silent_flattening_hidden_values_and_nonbinary_labels(scores, labels):
    with pytest.raises(ValueError):
        fit_va(scores, labels, grid_size=2)


@pytest.mark.parametrize('size', [0, -1, 2.5, True])
def test_va_fit_rejects_invalid_grid_size(size):
    with pytest.raises(ValueError):
        fit_va([.2, .8], [0, 1], grid_size=size)


def test_va_exact_calibration_rejects_masks_and_retains_single_class_and_empty_queries():
    with pytest.raises(ValueError):
        fit_exact_table(np.ma.array([.2], mask=True), [.8])
    with pytest.raises(ValueError):
        fit_exact_table([], [])
    table = fit_exact_table([], [.2, .8])
    low, high = predict_exact_table(table, [])
    assert low.shape == high.shape == (0,)
    low, high = predict_va(fit_va([.2, .8], [1, 1], grid_size=2), [])
    assert low.shape == high.shape == (0,)
