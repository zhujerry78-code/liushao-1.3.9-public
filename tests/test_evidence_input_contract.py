"""Synthetic score-boundary regressions; no fitted artifact or experiment is changed."""
import numpy as np
import pytest

from liushao_engine import evidence as ev


@pytest.fixture
def params():
    identity = {'x': np.array([0., 1.]), 'y': np.array([0., 1.])}
    return {'frame': 'ANU', 'theta': ev.THETA, 'sources': ['rf', 'lr', 'typ'],
            'kinds': {'rf': 'supervised', 'lr': 'supervised', 'typ': 'typicality'},
            'calibrators': {'rf': identity, 'lr': identity},
            'reliability': {'rf': .9, 'lr': .8, 'typ': .95},
            'atypical_map': ev.fit_atypical_map(np.arange(1., 100.), .05), 'atypical_focal': 'AU'}


BAD_SCORES = [pytest.param(np.array([np.nan]), id='nan'),
              pytest.param(np.array([np.inf]), id='positive-infinity'),
              pytest.param(np.array([-np.inf]), id='negative-infinity'),
              pytest.param(np.ma.array([.05], mask=[True]), id='hidden-missing'),
              pytest.param(np.array([[.05]]), id='column-vector'),
              pytest.param(.05, id='scalar')]


@pytest.mark.parametrize('source', ['rf', 'lr', 'typ'])
@pytest.mark.parametrize('bad', BAD_SCORES)
def test_open_fusion_and_source_reject_invalid_score_before_mass_assignment(params, source, bad):
    scores = {'rf': np.array([.05]), 'lr': np.array([.05]), 'typ': np.array([50.])}
    scores[source] = bad
    with pytest.raises(ValueError):
        ev.source_mass(source, bad, params)
    with pytest.raises(ValueError):
        ev.fuse_open(scores, params)


@pytest.mark.parametrize('bad', BAD_SCORES)
def test_legacy_fusion_rejects_invalid_score_before_interpolation(params, bad):
    legacy = {key: params[key] for key in ('sources', 'calibrators', 'reliability')}
    legacy['sources'] = ['rf', 'lr']
    with pytest.raises(ValueError):
        ev.fuse({'rf': bad, 'lr': np.array([.05])}, legacy)


@pytest.mark.parametrize('query', [np.array([1.]), np.array([])])
def test_fusion_rejects_misaligned_sources_instead_of_broadcasting(params, query):
    scores = {'rf': np.array([.05, .9]), 'lr': query, 'typ': np.array([50., 99.])}
    with pytest.raises(ValueError):
        ev.fuse_open(scores, params)
    legacy = {key: params[key] for key in ('sources', 'calibrators', 'reliability')}
    legacy['sources'] = ['rf', 'lr']
    with pytest.raises(ValueError):
        ev.fuse(scores, legacy)


@pytest.mark.parametrize('bad', BAD_SCORES)
def test_tail_reference_rejects_invalid_or_hidden_scores(bad):
    with pytest.raises(ValueError):
        ev.fit_atypical_map(bad)


@pytest.mark.parametrize('bad', BAD_SCORES)
def test_calibrator_fit_rejects_invalid_or_hidden_scores(bad):
    with pytest.raises(ValueError):
        ev.fit_calibrator(bad, np.array([0.]))


def test_calibrator_fit_rejects_hidden_labels():
    with pytest.raises(ValueError):
        ev.fit_calibrator(np.array([.1, .9]), np.ma.array([0, 1], mask=[0, 1]))


def test_finite_vector_validation_preserves_values_and_all_unmasked_data():
    scores = {'rf': np.array([.2, .3]), 'lr': [.1, .5],
              'typ': np.ma.array([10., 20.], mask=[False, False])}
    got = ev.validated_scores(scores, ('rf', 'lr', 'typ'))
    assert got['rf'] is scores['rf']
    assert all(value.dtype == np.float64 and value.shape == (2,) for value in got.values())
    assert all(np.array_equal(got[name], scores[name]) for name in got)
    empty = ev.validated_scores({'rf': [], 'lr': []}, ('rf', 'lr'))
    assert all(value.shape == (0,) for value in empty.values())


def test_finite_out_of_range_scores_keep_existing_endpoint_and_tail_semantics(params):
    query = np.array([-np.finfo(float).max, -.1, 0., .2, 1., 2., np.finfo(float).max])
    cal = params['calibrators']['rf']
    expected = np.clip(np.interp(query, cal['x'], cal['y']), ev.EPS, 1 - ev.EPS)
    assert np.array_equal(ev.calibrate(cal, query), expected)
    tail = params['atypical_map']
    p = (1 + len(tail['null']) - np.searchsorted(tail['null'], query, side='left')) / (len(tail['null']) + 1)
    expected_q = np.clip(1 - p / tail['alpha'], 0., 1.)
    assert np.array_equal(ev.atypical_degree(tail, query), expected_q)


def test_missing_typicality_cannot_become_a_valid_synthetic_prediction(params):
    from liushao_engine import STATE_RULES, analyze_scores
    from liushao_engine.conformal import fit_conformal
    from liushao_engine.venn_abers import fit_va
    score = np.r_[np.linspace(.005, .1, 100), np.linspace(.9, .995, 100)]
    labels = np.repeat([0, 1], 100)
    engine = {'fusion': params,
              'conformal': fit_conformal(score, labels, .05, novelty_share=np.zeros(200)),
              'venn_abers': fit_va(score, labels, grid_size=8), 'conflict_cut': 1., 'states': STATE_RULES}
    scores = {'rf': np.array([.05]), 'lr': np.array([.05]), 'typ': np.array([50.])}
    assert analyze_scores(scores, engine)['ow_state'].tolist() == ['正常']
    for missing in (np.array([np.nan]), np.array([np.inf]), np.array([-np.inf]),
                    np.ma.array([50.], mask=[True])):
        with pytest.raises(ValueError):
            analyze_scores({**scores, 'typ': missing}, engine)
