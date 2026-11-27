"""Exercise the fit, application inference and review persistence boundaries."""
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import pytest

import liushao_engine as engine_api
import app_service
from app_engine_evidence import batch_evidence_reference, build_engine_evidence, event_review_payload
from app_engine_inference import analyze_current_engine, exact_va, inference_version
from liushao import reviews


class ScoreModel:
    def predict_proba(self, features):
        probability = np.clip(features.iloc[:, 0].to_numpy() / 4 + .5, .01, .99)
        return np.column_stack((1 - probability, probability))


@pytest.fixture
def fit_inputs(monkeypatch):
    rng = np.random.default_rng(138)
    labels = np.tile([0, 1], 90)
    x = pd.DataFrame(rng.normal(size=(len(labels), 3)) / 3 + labels[:, None], columns=list('abc'))
    bundle = {'model_version': 'synthetic', 'rf': ScoreModel(), 'lr': ScoreModel(),
              'config': {'features': list(x), 'rf_threshold': .6}}
    fit_ecod, fit_typicality = engine_api.fit_ecod, engine_api.fit_typicality
    monkeypatch.setattr(engine_api, 'fit_ecod', lambda rows, **kw: fit_ecod(rows, max_rows=60, **kw))
    monkeypatch.setattr(engine_api, 'fit_typicality', lambda rows, **kw: fit_typicality(rows, max_rows=60, **kw))
    monkeypatch.setattr(engine_api, 'drift_space', lambda bundle, frame: frame.to_numpy())
    fit_drift = engine_api.drift.fit_drift
    monkeypatch.setattr(
        engine_api.drift, 'fit_drift',
        lambda *args, **kwargs: fit_drift(*args, **kwargs, sizes=(32,), min_block_rows=2, reps=4))
    blocks = np.repeat(['day-1', 'day-2', 'day-3'], 60)
    return (bundle, x, labels, x, labels), blocks


def test_joint_four_states_is_rejected_during_fit(fit_inputs):
    args, _ = fit_inputs
    with pytest.raises(ValueError, match='joint.*orthogonal'):
        engine_api.fit_engine(*args, normal_tests='joint', states=engine_api.STATE_RULES)


def test_joint_four_states_is_rejected_during_load(fit_inputs, tmp_path, monkeypatch):
    args, _ = fit_inputs
    bundle = args[0]
    # Joint calibration is a supported standalone method. Reject only the combination.
    joint = engine_api.fit_engine(*args, normal_tests='joint')
    path = tmp_path / 'engine.joblib'
    joblib.dump(joint, path)
    assert engine_api.load_engine(path, bundle)['conformal']['scores'] == joint['conformal']['scores']
    joint_four = {**joint, 'states': engine_api.STATE_RULES}
    joblib.dump(joint_four, path)
    with pytest.raises(ValueError, match='joint.*orthogonal'):
        engine_api.load_engine(path, bundle)

    descriptor = {'format': 2, 'base_model_version': bundle['model_version'], 'base_model_sha256': 'c' * 64,
                  'engine_version': engine_api.engine_version(joint_four),
                  'engine_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    (tmp_path / 'engine.json').write_text(json.dumps(descriptor), encoding='utf-8')
    # Integration checks a release allowlist before reading the synthetic artifact.
    if hasattr(app_service, 'matches_release'):
        monkeypatch.setattr(app_service, 'matches_release', lambda *_: True)
    loaded, status = app_service.load_trusted_engine_status(tmp_path, bundle)
    assert loaded is None and status.code == 'version_mismatch'
    assert 'joint' in status.message and 'orthogonal' in status.message


def test_joint_conformal_columns_cannot_enter_four_states(fit_inputs):
    args, _ = fit_inputs
    joint = engine_api.fit_engine(*args, normal_tests='joint')
    result = engine_api.analyze_scores(engine_api.expert_scores(args[0], joint, args[1].iloc[:2]), joint)
    with pytest.raises(ValueError, match='joint.*orthogonal'):
        engine_api.four_states(result, [False, False], joint['conformal']['epsilon'])


@pytest.mark.parametrize('missing', ['none', 'masked'])
def test_fit_validates_original_time_blocks(fit_inputs, missing):
    args, blocks = fit_inputs
    # Affect both validation halves and exceed the synthetic minimum block size.
    if missing == 'none':
        blocks = blocks.tolist()
        blocks[:20] = [None] * 20
    else:
        blocks = np.ma.array(blocks, mask=np.arange(len(blocks)) < 20)
    with pytest.raises(ValueError, match='漂移时间块.*缺失'):
        engine_api.fit_engine(*args, cal_blocks=blocks)


def test_discounted_batch_exact_va_and_persisted_evidence(fit_inputs, tmp_path):
    args, blocks = fit_inputs
    bundle = args[0]
    engine = engine_api.fit_engine(
        *args, cal_blocks=blocks, states=engine_api.STATE_RULES,
        drift_discount={'form': 'uniform', 'lambda': 10}, conflict_gate='q95',
        risk_control='class', events={'k': 3, 'reps': 3})
    frozen_version = engine_api.engine_version(engine)
    identity = {'engine_version': frozen_version, 'engine_sha256': 'a' * 64,
                'engine_descriptor_sha256': 'b' * 64, 'engine_inference_version': inference_version(engine)}
    engine['_artifact_identity'] = identity
    features = pd.DataFrame(np.full((36, 3), 9.), columns=bundle['config']['features'],
                            index=np.arange(11, 47))
    pred = pd.DataFrame({'rf_score': .99, 'lr_score': .99, 'rf_anomaly': True,
                         'source_file': 'synthetic.csv', 'source_row': features.index, 'source_sha256': 'c' * 64,
                         'record_id': [f'row-{i}' for i in features.index]}, index=features.index)
    for name in ('engine_sha256', 'engine_descriptor_sha256', 'engine_inference_version'):
        pred[name] = identity[name]
    scores = engine_api.expert_scores(bundle, engine, features, pred.rf_score, pred.lr_score)
    values, used = engine_api.analyze_batch(scores, engine)
    _, detected = engine_api.batch_engine(engine, scores)
    assert detected['drift'] and any(f != 1 for f in detected['factors'].values())
    expected_low, expected_high = exact_va(used, values['ds_betp'])
    old_low, old_high = exact_va(engine, values['ds_betp'])
    assert np.any((old_low != expected_low) | (old_high != expected_high))
    displayed = analyze_current_engine(pred, features, bundle, engine)
    np.testing.assert_array_equal(displayed.va_low, expected_low)
    np.testing.assert_array_equal(displayed.va_high, expected_high)
    reference = batch_evidence_reference(displayed, features, bundle, engine)
    assert displayed.engine_version.eq(frozen_version).all()

    db = tmp_path / 'reviews.sqlite'
    score_bytes = np.column_stack([scores[name] for name in engine_api.OPEN_SOURCES]).astype('<f8').tobytes()
    for i, (index, record) in enumerate(displayed.iterrows()):
        evidence = build_engine_evidence(record, features.loc[[index]], bundle, engine,
                                         reference=reference.loc[index])
        assert evidence['identity'] == identity
        assert evidence['results']['va_low'] == expected_low[i]
        assert evidence['results']['va_high'] == expected_high[i]
        assert evidence['parameters']['conflict_cut'] == used['conflict_cut']
        masses = engine_api.source_masses(record, used).to_dict(orient='records')
        assert evidence['source_masses'] == masses
        assert masses != engine_api.source_masses(record, engine).to_dict(orient='records')
        effective = evidence['effective_batch']
        assert effective['discount_factors'] == detected['factors']
        assert effective['rows'] == len(features)
        assert effective['expert_scores_sha256'] == hashlib.sha256(score_bytes).hexdigest()
        assert len(effective['calibration_sha256']) == 64
        snapshot = dict(schema_version=2, method_version='synthetic', record_id=record.record_id,
                        model_version=bundle['model_version'], scores=dict(rf=.99, lr=.99, rf_threshold=.6, lr_threshold=.6),
                        cards=[], selection={}, decomposition={}, input_sha256='c' * 64,
                        model_sha256='d' * 64, descriptor_sha256='e' * 64, engine=evidence)
        reviews.save_review(db, record.record_id, bundle['model_version'], '待复核', '', evidence_snapshot=snapshot)
        assert reviews.history(db, record_id=record.record_id)[0]['evidence_snapshot'] == snapshot

    meta = dict(model_version=bundle['model_version'], input_name='synthetic.csv', input_sha256='c' * 64,
                model_sha256='d' * 64, descriptor_sha256='e' * 64, ui_revision=1, rf_threshold=.6, lr_threshold=.6)
    inherited = event_review_payload(displayed.iloc[0], 11, meta, 1, [{'id': 1}],
                                     event_size=len(displayed), signature_fields=['a'])
    saved_batch = reviews.history(db, record_id=displayed.iloc[0].record_id)[0]['evidence_snapshot']['engine']
    assert inherited['event_review']['effective_batch'] == saved_batch['effective_batch']

    # Same per-row output under full discount still cannot certify a different batch.
    subset = displayed.iloc[:-1]
    wrong_reference = batch_evidence_reference(subset, features.iloc[:-1], bundle, engine)
    with pytest.raises(ValueError, match='批次'):
        build_engine_evidence(displayed.iloc[0], features.iloc[[0]], bundle, engine,
                              reference=wrong_reference.iloc[0])
    assert engine_api.engine_version(engine) == frozen_version
