"""Unweighted conformal input contracts, synthetic only; thresholds stay unchanged."""
import numpy as np
import pytest

from liushao_engine import conformal as cp


LAYOUTS = ['legacy', cp.CLASS_SPECIFIC, cp.ORTHOGONAL]


def fit_case(layout):
    betp = np.array([.1, .2, .8, .9])
    y = np.array([0, 0, 1, 1])
    kwargs = ({'betp_normal': 1 - betp} if layout == cp.CLASS_SPECIFIC else
              {'novelty_share': np.array([.1, .2, .5, .8])} if layout == cp.ORTHOGONAL else {})
    return cp.fit_conformal(betp, y, **kwargs)


BAD_VECTORS = [
    [np.nan, .8], [np.inf, .8], [-np.inf, .8],
    np.array([.2 + 1j, .8 + 0j]),
    np.ma.array([.2, .8], mask=[True, False]),
    [[.2], [.8]], .2, ['.2', '.8'], [None, .8],
]


@pytest.mark.parametrize('field', ['betp', 'y', 'betp_normal', 'novelty_share'])
@pytest.mark.parametrize('bad', BAD_VECTORS)
def test_fit_rejects_nonfinite_missing_complex_and_nonnumeric_vectors(field, bad):
    args = {'betp': [.2, .8], 'y': [0, 1]}
    args[field] = bad
    with pytest.raises(ValueError):
        cp.fit_conformal(**args)


@pytest.mark.parametrize('labels', [[.5, 1.], [0, 2], [-1, 1], [0, 1.1]])
def test_fit_does_not_truncate_or_silently_drop_unknown_labels(labels):
    with pytest.raises(ValueError):
        cp.fit_conformal([.2, .8], labels)


@pytest.mark.parametrize('field', ['y', 'betp_normal', 'novelty_share'])
@pytest.mark.parametrize('bad', [[0.], [], [0., 1., 0.]])
def test_fit_rejects_misaligned_arrays_before_boolean_indexing(field, bad):
    args = {'betp': [.2, .8], 'y': [0, 1]}
    args[field] = bad
    with pytest.raises(ValueError):
        cp.fit_conformal(**args)


@pytest.mark.parametrize('layout', LAYOUTS)
@pytest.mark.parametrize('bad', BAD_VECTORS)
def test_predict_rejects_invalid_query_vectors_on_every_layout(layout, bad):
    kwargs = ({'betp_normal': [.7, .3]} if layout == cp.CLASS_SPECIFIC else
              {'novelty_share': [.2, .3]} if layout == cp.ORTHOGONAL else {})
    with pytest.raises(ValueError):
        cp.predict_conformal(fit_case(layout), bad, **kwargs)


@pytest.mark.parametrize('layout,field', [(cp.CLASS_SPECIFIC, 'betp_normal'),
                                         (cp.ORTHOGONAL, 'novelty_share')])
@pytest.mark.parametrize('bad', BAD_VECTORS + [[.2], [], [.1, .2, .3]])
def test_predict_extra_scores_cannot_broadcast_or_discard_invalid_values(layout, field, bad):
    with pytest.raises(ValueError):
        cp.predict_conformal(fit_case(layout), [.3, .7], **{field: bad})


@pytest.mark.parametrize('eps', [np.nan, np.inf, 0., 1., -.1, '0.05', .05 + 0j, np.array([.05]),
                                 pytest.param(10**1000, id='huge-integer')])
def test_fit_and_both_predict_epsilon_sources_reject_invalid_thresholds(eps):
    with pytest.raises(ValueError):
        cp.fit_conformal([.2, .8], [0, 1], epsilon=eps)
    params = fit_case('legacy')
    with pytest.raises(ValueError):
        cp.predict_conformal(params, [.5], epsilon=eps)
    with pytest.raises(ValueError):
        cp.predict_conformal(dict(params, epsilon=eps), [.5])


@pytest.mark.parametrize('position', [0, 1])
@pytest.mark.parametrize('bad', BAD_VECTORS)
def test_low_level_p_values_rejects_invalid_calibration_and_query(position, bad):
    args = [[.1, .8], [.2, .7]]
    args[position] = bad
    with pytest.raises(ValueError):
        cp._p_values(*args)


def test_unsorted_calibration_is_rejected_without_mutating_the_bound_array():
    calibration = np.array([.8, .1])
    original = calibration.copy()
    # A direct count gives 2/3; searchsorted on the unsorted vector used to give 1/3.
    assert (np.count_nonzero(calibration >= .5) + 1) / (len(calibration) + 1) == 2 / 3
    with pytest.raises(ValueError):
        cp._p_values(calibration, [.5])
    np.testing.assert_array_equal(calibration, original)


@pytest.mark.parametrize('layout,key', [('legacy', 'attack'), ('legacy', 'normal'),
                                       (cp.CLASS_SPECIFIC, 'normal'),
                                       (cp.ORTHOGONAL, 'normal_supervised'),
                                       (cp.ORTHOGONAL, 'normal_typicality')])
@pytest.mark.parametrize('bad', [[.8, .1], [np.nan, .8], [[.1], [.8]],
                                np.ma.array([.1, .8], mask=[True, False])])
def test_prediction_rejects_invalid_stored_calibration_vectors(layout, key, bad):
    params = fit_case(layout)
    params[key] = bad
    kwargs = ({'betp_normal': [.7, .3]} if layout == cp.CLASS_SPECIFIC else
              {'novelty_share': [.2, .3]} if layout == cp.ORTHOGONAL else {})
    with pytest.raises(ValueError):
        cp.predict_conformal(params, [.3, .7], **kwargs)


def count_oracle(calibration, queries):
    return np.array([(np.count_nonzero(calibration >= q) + 1) / (len(calibration) + 1)
                     for q in queries], dtype=float)


@pytest.mark.parametrize('layout', LAYOUTS)
def test_valid_ties_and_threshold_boundaries_match_independent_count_oracle(layout):
    betp = np.r_[np.zeros(39), np.ones(19)]
    labels = np.r_[np.zeros(39), np.ones(19)]
    normal = 1 - betp
    share = betp.copy()
    fit_args = ({'betp_normal': normal} if layout == cp.CLASS_SPECIFIC else
                {'novelty_share': share} if layout == cp.ORTHOGONAL else {})
    params = cp.fit_conformal(betp, labels, **fit_args)
    query = np.array([0., np.nextafter(0., 1.), .5, np.nextafter(1., 0.), 1.])
    query_normal, query_share = 1 - query, query.copy()
    predict_args = ({'betp_normal': query_normal} if layout == cp.CLASS_SPECIFIC else
                    {'novelty_share': query_share} if layout == cp.ORTHOGONAL else {})
    actual = cp.predict_conformal(params, query, **predict_args)
    attack = count_oracle(1 - betp[labels == 1], 1 - query)
    if layout == cp.ORTHOGONAL:
        sup = count_oracle(betp[labels == 0], query)
        typ = count_oracle(share[labels == 0], query_share)
        expected_normal = np.minimum(1., 2 * np.minimum(sup, typ))
        np.testing.assert_array_equal(actual['cp_p_normal_supervised'], sup)
        np.testing.assert_array_equal(actual['cp_p_normal_typicality'], typ)
    else:
        cal_normal = 1 - normal[labels == 0] if layout == cp.CLASS_SPECIFIC else betp[labels == 0]
        alpha_normal = 1 - query_normal if layout == cp.CLASS_SPECIFIC else query
        expected_normal = count_oracle(cal_normal, alpha_normal)
    np.testing.assert_array_equal(actual['cp_p_attack'], attack)
    np.testing.assert_array_equal(actual['cp_p_normal'], expected_normal)
    np.testing.assert_array_equal(actual['cp_set_size'], (attack > .05).astype(int) + (expected_normal > .05).astype(int))


@pytest.mark.parametrize('layout', LAYOUTS)
def test_empty_calibration_and_empty_query_keep_defined_one_dimensional_results(layout):
    extra = {'betp_normal': []} if layout == cp.CLASS_SPECIFIC else {'novelty_share': []} if layout == cp.ORTHOGONAL else {}
    params = cp.fit_conformal([], [], **extra)
    output = cp.predict_conformal(params, [], **extra)
    assert all(np.asarray(values).shape == (0,) for values in output.values())
    extra = {'betp_normal': [.5]} if layout == cp.CLASS_SPECIFIC else {'novelty_share': [.5]} if layout == cp.ORTHOGONAL else {}
    output = cp.predict_conformal(params, [.5], **extra)
    assert output['cp_p_attack'][0] == output['cp_p_normal'][0] == 1.
    assert output['cp_set_size'][0] == 2


def test_mask_free_arrays_and_binary_float_or_boolean_labels_remain_supported():
    for labels in ([0., 1.], [False, True]):
        params = cp.fit_conformal(np.ma.array([.2, .8], mask=False), labels)
        actual = cp.predict_conformal(params, np.ma.array([.5], mask=False))
        assert actual['cp_p_attack'].shape == (1,)
