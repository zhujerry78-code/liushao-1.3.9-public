"""Captured JSON must be valid, stable, and compatible with old snapshots."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
import sqlite3
from threading import Event, current_thread

import pytest

from app_engine_evidence import save_event_reviews
from liushao import reviews


@pytest.mark.parametrize('writer', ['single', 'event'])
@pytest.mark.parametrize('mutation', ['score', 'schema'])
@pytest.mark.parametrize('mutation_time', ['before_capture', 'after_capture'])
def test_snapshot_capture_rechecks_bytes_and_isolates_later_mutations(tmp_path, monkeypatch, writer, mutation, mutation_time):
    payload = dict(schema_version=2, method_version='synthetic-capture', record_id='record',
                   model_version='model', input_sha256='a' * 64, model_sha256='b' * 64,
                   descriptor_sha256='c' * 64, cards=[], selection={}, decomposition={},
                   scores=dict(rf=.2, lr=.3, rf_threshold=.5, lr_threshold=.5))
    expected = deepcopy(payload)
    ready, release = Event(), Event()
    original_dumps = reviews.json.dumps
    expected_json = original_dumps(expected, ensure_ascii=False, sort_keys=True,
                                   separators=(',', ':'), allow_nan=False)

    def controlled_dump(value, *args, **kwargs):
        if current_thread().name.startswith('snapshot-writer') and isinstance(value, dict) and value.get('record_id') == 'record':
            captured = original_dumps(value, *args, **kwargs) if mutation_time == 'after_capture' else None
            ready.set()
            assert release.wait(5), 'test did not release serialization'
            if captured is not None:
                return captured
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(reviews.json, 'dumps', controlled_dump)
    database = tmp_path / 'synthetic.sqlite'

    def write():
        if writer == 'single':
            reviews.save_review(database, 'record', 'model', '确认正常', '合成API并发验证', evidence_snapshot=payload)
        else:
            save_event_reviews(database, 'model', '确认正常', '合成API并发验证', {'record': 0}, [payload])

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix='snapshot-writer') as pool:
        future = pool.submit(write)
        try:
            assert ready.wait(5)
            if mutation == 'score':
                payload['scores']['rf'] = 1.5
            else:
                payload['schema_version'] = 3
        finally:
            release.set()
        if mutation_time == 'before_capture':
            # A value changed after the initial checks must be rechecked using
            # the captured bytes, before either write API opens the database.
            with pytest.raises(ValueError):
                future.result(timeout=5)
        else:
            future.result(timeout=5)
    if mutation_time == 'before_capture':
        assert not database.exists()
        return
    saved = reviews.history(database)[0]
    assert saved['status'] == '确认正常'
    assert saved['evidence_snapshot_status'] == 'saved'
    assert saved['evidence_schema_version'] == 2
    assert saved['evidence_snapshot'] == expected
    assert saved['evidence_sha256'] == hashlib.sha256(expected_json.encode('utf-8')).hexdigest()
    with sqlite3.connect(database) as db:
        assert db.execute('SELECT payload_json FROM review_evidence').fetchone()[0] == expected_json


def compatibility_snapshot():
    return dict(schema_version=2, method_version='synthetic-compatibility', record_id='record',
                model_version='model', input_sha256='a' * 64, model_sha256='b' * 64,
                descriptor_sha256='c' * 64, cards=[], selection={}, decomposition={},
                scores=dict(rf=.2, lr=.3, rf_threshold=.5, lr_threshold=.5))


@pytest.mark.parametrize('schema_version', [1, 2])
@pytest.mark.parametrize('reader', ['capture', 'history'])
def test_deep_legacy_optional_json_keeps_existing_compatibility(schema_version, reader):
    payload = compatibility_snapshot()
    payload['schema_version'] = schema_version
    nested = {'leaf': '合成旧版历史字段'}
    for _ in range(500):
        nested = {'child': nested}
    payload['legacy_optional'] = nested
    serialized = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                            separators=(',', ':'), allow_nan=False)
    assert serialized.count('"child"') == 500
    assert len(serialized.encode('utf-8')) < reviews.MAX_SNAPSHOT_BYTES

    if reader == 'capture':
        assert reviews._snapshot_json(payload, 'record', 'model') == serialized
    else:
        original = dict(record_id='record', model_version='model', status='待补证',
                        note='合成旧版记录，保留原判断', evidence_schema_version=schema_version,
                        evidence_json=serialized,
                        evidence_sha256=hashlib.sha256(serialized.encode('utf-8')).hexdigest())
        saved = reviews.review_with_evidence(original)
        assert saved['evidence_snapshot_status'] == 'saved'
        assert saved['status'] == original['status'] and saved['note'] == original['note']
        assert json.dumps(saved['evidence_snapshot'], ensure_ascii=False, sort_keys=True,
                          separators=(',', ':'), allow_nan=False) == serialized
        assert original['evidence_json'] == serialized


@pytest.mark.parametrize('field,value', [
    ('cards', ()),
    ('cards', ({'field': 'synthetic', 'reference': {}, 'rf': {}, 'lr': {}},)),
    ('scores', [('rf', .2), ('lr', .3), ('rf_threshold', .5), ('lr_threshold', .5)]),
    ('selection', []), ('decomposition', ()), ('identity', []),
    ('schema_version', True),
])
def test_snapshot_capture_does_not_normalize_invalid_contract_types(field, value):
    payload = compatibility_snapshot()
    payload[field] = value
    with pytest.raises(ValueError):
        reviews._snapshot_json(payload, 'record', 'model')
