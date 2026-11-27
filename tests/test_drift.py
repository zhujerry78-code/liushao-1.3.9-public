"""liushao_engine.drift (T08) on synthetic expert scores; no real days, no artifacts."""
import numpy as np
import pytest

from liushao_engine import drift
from liushao_engine import typicality as ty


def scores(rng, n, attack_share, shift=0.0):
    """Two-mode output space: normals near 0, known attacks near 1; typ ~ exponential."""
    y = (rng.random(n) < attack_share).astype(int)
    rf = np.clip(np.where(y == 1, 0.95, 0.02) + rng.normal(0, 0.02, n), 0, 1)
    lr = np.clip(np.where(y == 1, 0.9, 0.05) + rng.normal(0, 0.03, n) + shift, 0, 1)
    typ = rng.exponential(1.0, n) + shift
    return np.column_stack([rf, lr, typ]), y


def reference(seed=0, n=8000, blocks=8, attack_share=0.2):
    rng = np.random.default_rng(seed)
    z, y = scores(rng, n, attack_share)
    return z, y, np.repeat([f'b{i}' for i in range(blocks)], n // blocks)


@pytest.fixture(scope='module')
def model():
    z, y, b = reference()
    return drift.fit_drift(z, y, b, rf_threshold=0.5, reps=20, seed=1)


def test_fit_reports_blocks_thresholds_and_drops_small_blocks():
    z, y, b = reference()
    b = b.astype(object)
    b[:50] = 'tiny'
    m = drift.fit_drift(z, y, b, rf_threshold=0.5, reps=5, sizes=(200,), candidates=('tail_h',))
    assert m['dropped_blocks'] == {'tiny': 50} and len(m['blocks']) == 8 and m['blocks']['b0'] == 950
    assert set(m['thresholds']) == {'tail_h'} and set(m['thresholds']['tail_h'][200]) == set(drift.NAMES)
    assert m['null_summary']['tail_h'][200]['draws'] == 40 and m['null_summary']['tail_h'][200]['units'] == 8
    with pytest.raises(ValueError):
        drift.fit_drift(z, y, np.full(len(z), 'one'), rf_threshold=0.5, reps=2)
    with pytest.raises(ValueError):
        drift.fit_drift(np.where(z > 5, np.nan, z), y, b, rf_threshold=0.5, reps=2)


def test_same_distribution_batch_is_quiet_and_statistics_stay_in_unit_interval(model):
    z, _ = scores(np.random.default_rng(10), 3000, 0.2)
    for c in drift.CANDIDATES:
        r = drift.detect_drift(model, z, c)
        assert r['status'] == 'ok' and r['size_class'] == 1000
        assert 0.0 <= r['delta'] <= 1.0 and all(0.0 <= v <= 1.0 for v in r['delta_s'].values())
        assert r['delta'] < 0.1, c
        assert r['rows_used'] == (1000 if c.startswith('whole_mmd') else 3000)


def test_shifted_batch_is_flagged_by_every_candidate_and_names_the_expert(model):
    z, _ = scores(np.random.default_rng(11), 3000, 0.2, shift=0.5)
    for c in drift.CANDIDATES:
        r = drift.detect_drift(model, z, c)
        assert r['drift'] and r['delta'] > r['threshold'], c
        assert r['drift_s']['lr'] and r['drift_s']['typ'], c
        assert r['delta_s']['rf'] < 0.01 < r['delta_s']['typ'] < r['delta_s']['lr'], c  # RF coordinate is unshifted


def test_prior_adjustment_ignores_a_pure_change_in_known_attack_share(model):
    z, _ = scores(np.random.default_rng(12), 3000, 0.6)  # same classes, three times the attacks
    fixed, adj = drift.detect_drift(model, z, 'whole_mmd'), drift.detect_drift(model, z, 'whole_mmd_adj')
    assert fixed['drift'] and fixed['delta'] > 0.2
    assert adj['delta'] < 0.05 and adj['w_hat'] == pytest.approx(0.6, abs=0.03)
    tail_fixed, tail_adj = drift.detect_drift(model, z, 'tail_h'), drift.detect_drift(model, z, 'tail_h_adj')
    assert tail_fixed['delta'] > 5 * tail_adj['delta'] and tail_fixed['w_hat'] is None


def test_tail_statistic_sees_a_small_contamination_the_whole_batch_index_misses(model):
    rng = np.random.default_rng(13)
    z, _ = scores(rng, 20000, 0.2)
    odd = rng.choice(len(z), 500, replace=False)  # 2.5% of the batch: RF silent, typ far in the tail
    z[odd, 0], z[odd, 2] = 0.02, z[odd, 2] + 6
    whole, tail = drift.detect_drift(model, z, 'whole_mmd'), drift.detect_drift(model, z, 'tail_h')
    assert tail['drift'] and tail['drift_s']['typ'] and not tail['drift_s']['rf']
    assert whole['delta'] < 0.05


def test_cohens_h_value_and_attack_share_closed_form():
    z, y, _ = reference()
    ref = drift.reference_stats(z, y, 0.5)
    assert ref['tpr'] == 1.0 and ref['fpr'] == 0.0 and ref['pi'] == pytest.approx(0.2, abs=0.02)
    batch = z[y == 0][:1000].copy()
    assert drift.attack_share(ref, batch) == 0.0
    s = drift.tail_statistic(ref, batch, adjusted=True, w=0.0)
    p = (batch[:, 2] > ref['cuts'][2, 1]).mean()
    p0 = ref['above_normal'][2, 1]
    assert p0 == pytest.approx(0.05, abs=1e-3)
    assert s['typ'] >= abs(2 * np.arcsin(np.sqrt(p)) - 2 * np.arcsin(np.sqrt(p0))) / np.pi - 1e-12
    no_attack = drift.reference_stats(z[y == 0], y[y == 0], 0.5)
    assert drift.attack_share(no_attack, batch) == 0.0 and np.isnan(no_attack['tpr'])


def test_mmd_index_saturates_for_disjoint_samples_and_is_zero_for_identical_ones():
    rng = np.random.default_rng(14)
    z, y, _ = reference(n=2000)
    space, ref = drift.fit_space(z), drift.reference_stats(z, y, 0.5)
    far = z[:500] + 100.0
    assert drift.mmd_statistic(space, ref, far, rng, adjusted=False)['joint'] == pytest.approx(1.0)
    same = drift.mmd_statistic(space, ref, z[rng.choice(2000, 1000, replace=False)], rng, adjusted=False)
    assert same['joint'] < 0.01 and set(same) == set(drift.NAMES)


def test_detection_is_deterministic_and_size_classes_follow_the_batch(model):
    z, _ = scores(np.random.default_rng(15), 700, 0.2)
    a, b = drift.detect_drift(model, z, 'whole_mmd'), drift.detect_drift(model, z, 'whole_mmd')
    assert a == b and a['size_class'] == 500 and a['rows_used'] == 500
    assert model['candidate'] == drift.SELECTED == 'tail_h_adj' and drift.detect_drift(model, z)['candidate'] == 'tail_h_adj'
    assert drift.detect_drift(model, z, 'whole_mmd', seed=7)['delta'] != a['delta']
    assert drift.detect_drift(model, z[:199], 'tail_h') == {'status': 'too_small', 'candidate': 'tail_h',
                                                           'rows': 199, 'drift': None}
    with pytest.raises(ValueError):
        drift.detect_drift(model, z, 'ks')
    with pytest.raises(ValueError):
        drift.detect_drift(model, z[:, :2], 'tail_h')


def test_null_draws_do_not_depend_on_which_candidates_are_computed_or_on_parallelism():
    z, y, b = reference(n=4000)
    one = drift.fit_drift(z, y, b, rf_threshold=0.5, reps=4, sizes=(200,), candidates=('whole_mmd_adj',), keep_null=True)
    both = drift.fit_drift(z, y, b, rf_threshold=0.5, reps=4, sizes=(200,), n_jobs=2, keep_null=True)
    assert one['thresholds']['whole_mmd_adj'] == both['thresholds']['whole_mmd_adj']
    assert np.array_equal(one['null']['whole_mmd_adj']['b3'][200], both['null']['whole_mmd_adj']['b3'][200])


def test_selected_typicality_model_matches_the_t06_grid_path():
    rng = np.random.default_rng(16)
    train, query = rng.lognormal(size=(3000, 6)), rng.lognormal(size=(400, 6))
    model = ty.fit_typicality(train, max_rows=2000)
    fit = ty.split_reference(3000, 2000)
    keys = np.full(3000, ty.GLOBAL, dtype=object)

    def score_fn(m, which, mask):
        z = train[~fit][mask] if which == 'cal' else query[mask]
        return {'top3': ty.all_aggregations(m, z, [('topk', 3)])[('topk', 3)]}
    p, _ = ty.bucketed_p(keys[fit], keys[~fit], {'q': np.full(400, ty.GLOBAL, dtype=object)},
                         lambda m: ty.fit_tail_model(train[fit][m]), score_fn)
    assert np.allclose(ty.score_typicality(model, query), -np.log(p['top3']['q']))
    assert len(model['calibration']) == 1000 and model['config'] == ty.SELECTED
