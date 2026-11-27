import hashlib
import json

import joblib
import pandas as pd
import pytest

import app_engine_inference
import app_engine_trust as trust
import app_service
import liushao_engine


@pytest.fixture
def registered_engine(tmp_path, monkeypatch):
    content = b'Inert engine fixture, handled only by the test loader'
    digest = hashlib.sha256(content).hexdigest()
    registered = next(iter(trust.TRUSTED_ENGINES.values()))
    metadata = dict(registered, engine_sha256=digest)
    bundle = {'model_version': registered['base_model_version'],
              'model_sha256': registered['base_model_sha256'],
              'config': {'data_version': registered['data_version']}}
    (tmp_path / 'engine.joblib').write_bytes(content)
    (tmp_path / 'engine.json').write_text(json.dumps(metadata), encoding='utf-8')
    monkeypatch.setattr(trust, 'TRUSTED_ENGINES', {digest: registered})
    monkeypatch.setattr(app_service, '_ENGINE_CACHE', {})
    return tmp_path, content, metadata, bundle


def reject(registered_engine, monkeypatch, code):
    directory, _, _, bundle = registered_engine
    monkeypatch.setattr(joblib, 'load', lambda *_: pytest.fail('Rejected bytes reached deserialization'))
    predictions = pd.DataFrame({'rf_score': [0.125], 'lr_score': [0.375]})
    result, status = app_service.with_engine_status(predictions, None, bundle, directory)
    assert result is predictions
    assert status.code == code and 'RF/LR' in status.message


def test_matching_unregistered_description_cannot_authorize_loading(registered_engine, monkeypatch):
    monkeypatch.setattr(trust, 'TRUSTED_ENGINES', {})
    reject(registered_engine, monkeypatch, 'version_mismatch')


@pytest.mark.parametrize('field,value', [
    ('engine_sha256', '0' * 64), ('engine_version', 'different'), ('base_model_version', 'different'),
    ('base_model_sha256', '0' * 64), ('data_version', 'different'),
    ('format', True), ('format', 2.0), ('format', '2'), ('format', 3),
])
def test_release_identity_checked_before_loading(registered_engine, monkeypatch, field, value):
    directory, _, metadata, _ = registered_engine
    metadata[field] = value
    (directory / 'engine.json').write_text(json.dumps(metadata), encoding='utf-8')
    reject(registered_engine, monkeypatch, 'tampered' if field == 'engine_sha256' else 'version_mismatch')


@pytest.mark.parametrize('field', ['model_sha256', 'model_version', 'data_version'])
def test_running_model_identity_must_match(registered_engine, monkeypatch, field):
    bundle = registered_engine[3]
    target = bundle['config'] if field == 'data_version' else bundle
    target[field] = 'different'
    reject(registered_engine, monkeypatch, 'version_mismatch')


@pytest.mark.parametrize('duplicate', ['"format": 2', '"notes": {"a": 1, "a": 2}'])
def test_duplicate_fields_rejected_even_if_values_agree(registered_engine, monkeypatch, duplicate):
    directory, _, metadata, _ = registered_engine
    contents = json.dumps(metadata)[:-1] + ', ' + duplicate + '}'
    (directory / 'engine.json').write_text(contents, encoding='utf-8')
    reject(registered_engine, monkeypatch, 'tampered')


@pytest.mark.parametrize('name,limit', [('engine.joblib', 16 * 1024 * 1024), ('engine.json', 256 * 1024)])
def test_artifact_file_limit(registered_engine, monkeypatch, name, limit):
    directory, _, _, _ = registered_engine
    with (directory / name).open('r+b') as handle:
        handle.truncate(limit + 1)
    reject(registered_engine, monkeypatch, 'tampered')


def test_cached_load_still_checks_identity_and_inference_failure_degrades(registered_engine, monkeypatch):
    directory, content, metadata, bundle = registered_engine
    loaded = []

    def load(handle):
        loaded.append(handle.read())
        return {'format': metadata['format'], 'model_version': bundle['model_version']}

    monkeypatch.setattr(joblib, 'load', load)
    monkeypatch.setattr(liushao_engine, 'engine_version', lambda _: metadata['engine_version'])
    engine, status = app_service.load_trusted_engine_status(directory, bundle)
    assert loaded == [content] and status.code == 'ok'
    assert engine['_artifact_identity']['base_model_sha256'] == bundle['model_sha256']

    def fail_inference(*args):
        raise RuntimeError('synthetic inference error')

    monkeypatch.setattr(app_engine_inference, 'analyze_current_engine', fail_inference)
    predictions = pd.DataFrame({'rf_score': [0.125], 'lr_score': [0.375]})
    result, status = app_service.with_engine_status(predictions, None, bundle, directory)
    assert result is predictions and status.code == 'runtime_error' and '运行出错' in status.message
    assert loaded == [content]
    bundle['model_sha256'] = 'different'
    reject(registered_engine, monkeypatch, 'version_mismatch')
