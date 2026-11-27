"""Synthetic review persistence cases; no frozen models or experiment artifacts."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from copy import deepcopy
import hashlib
import json
import sqlite3
import threading

import pytest

from app_engine_evidence import event_review_state, save_event_reviews
from app_record_tools import (
    format_review_time, saved_engine_summary, saved_evidence_summary,
    valid_saved_event_review,
)
from liushao import reviews


def evidence(record='target'):
    return dict(schema_version=2, method_version='p0-2-synthetic',
                record_id=record, model_version='model', cards=[], selection={}, decomposition={},
                input_sha256='a' * 64, model_sha256='b' * 64, descriptor_sha256='c' * 64,
                scores=dict(rf=.2, lr=.3, rf_threshold=.5, lr_threshold=.5))


@pytest.mark.parametrize('field,value', [
    ('record_id', 7), ('record_id', b'target'), ('record_id', ''), ('record_id', ' \t'),
    ('model_version', 7), ('model_version', b'model'), ('model_version', ''),
    ('model_version', ' \t'), ('status', None), ('status', []), ('status', ''),
    ('status', b'pending'), ('note', None), ('note', b'bytes'), ('note', []),
])
def test_invalid_single_write_never_creates_database(tmp_path, field, value):
    args = dict(record_id='target', model_version='model', status='待补证', note='合成依据')
    args[field] = value
    path = tmp_path / 'absent' / 'reviews.sqlite'
    with pytest.raises(ValueError):
        reviews.save_review(path, **args)
    assert not path.parent.exists()


def test_single_save_reopen_and_duplicate_preserves_exact_snapshot(tmp_path):
    path = tmp_path / 'reviews.sqlite'
    payload = evidence()
    frozen = deepcopy(payload)
    reviews.save_review(path, 'target', 'model', '确认正常', '逐条依据',
                        expected_review_id=0, evidence_snapshot=payload)
    before = reviews.history(path)
    payload['scores']['rf'] = .9
    payload['cards'].append({'changed': True})
    with closing(reviews.connect(path)) as db:
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    assert reviews.history(path) == before
    assert before[0]['evidence_snapshot'] == frozen
    with pytest.raises(reviews.ReviewConflictError):
        reviews.save_review(path, 'target', 'model', '确认异常', '重复提交不得覆盖',
                            expected_review_id=0, evidence_snapshot=frozen)
    assert reviews.history(path) == before
    # A newly loaded revision can still append a legitimate subsequent judgment.
    reviews.save_review(path, 'target', 'model', '待补证', '后续复核',
                        expected_review_id=before[0]['id'], evidence_snapshot=frozen)
    assert reviews.history(path)[0] == before[0]


def test_simultaneous_single_submissions_keep_one_record(tmp_path):
    path = tmp_path / 'reviews.sqlite'
    reviews.history(path)
    ready = threading.Barrier(2)
    def submit(_):
        ready.wait(timeout=10)
        try:
            reviews.save_review(path, 'target', 'model', '确认正常', '并发逐条依据',
                                expected_review_id=0, evidence_snapshot=evidence())
            return 'saved'
        except reviews.ReviewConflictError:
            return 'conflict'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(submit, range(2))) == ['conflict', 'saved']
    assert len(reviews.history(path)) == 1


def test_event_save_reopen_duplicate_and_existing_valid_record(tmp_path):
    path = tmp_path / 'reviews.sqlite'
    for rid in ['rep-a', 'rep-b', 'already-reviewed']:
        reviews.save_review(path, rid, 'model', '确认正常', '直接复核依据',
                            expected_review_id=0, evidence_snapshot=evidence(rid))
    direct = reviews.current_reviews(path, 'model')
    members = list(direct) + ['target-a', 'target-b']
    representatives = ['rep-a', 'rep-b']
    state = event_review_state(members, representatives, direct)
    assert state.status == '确认正常' and not state.disagreement
    assert state.targets == ['target-a', 'target-b']
    refs = [direct[rid]['id'] for rid in representatives]
    payloads = [evidence(rid) for rid in state.targets]
    for payload in payloads:
        payload['event_review'] = dict(mode='representative_consensus', event_id=0,
                                      representative_review_ids=refs)
    frozen = deepcopy(payloads)
    expected = {rid: direct.get(rid, {}).get('id', 0) for rid in members}
    save_event_reviews(path, 'model', state.status, '整簇依据', expected, iter(payloads))
    before = reviews.history(path)
    payloads[0]['scores']['rf'] = .99
    with closing(reviews.connect(path)) as db:
        assert db.execute('SELECT COUNT(*) FROM review_evidence').fetchone()[0] == 5
    reopened = reviews.current_reviews(path, 'model')
    assert reviews.history(path) == before
    for payload in frozen:
        saved = reopened[payload['record_id']]
        assert saved['evidence_snapshot'] == payload
        assert saved['evidence_snapshot_status'] == 'saved'
        assert '本条未单独复核' in saved_evidence_summary(saved['evidence_snapshot'])
    for rid in direct:
        assert reopened[rid] == direct[rid]
    with pytest.raises(reviews.ReviewConflictError):
        save_event_reviews(path, 'model', state.status, '重复整簇不得覆盖', expected, frozen)
    assert reviews.history(path) == before
    assert event_review_state(members, representatives, reopened).targets == []


def test_disagreement_falls_back_to_single_review(tmp_path):
    path = tmp_path / 'reviews.sqlite'
    for rid, status in [('rep-a', '确认正常'), ('rep-b', '确认异常')]:
        reviews.save_review(path, rid, 'model', status, '代表分别复核',
                            expected_review_id=0, evidence_snapshot=evidence(rid))
    before = reviews.history(path)
    state = event_review_state(['rep-a', 'rep-b', 'target'], ['rep-a', 'rep-b'],
                               reviews.current_reviews(path, 'model'))
    assert state.disagreement and state.status is None and state.targets == ['target']
    with pytest.raises(ValueError):
        save_event_reviews(path, 'model', state.status, '有分歧不能整簇',
                           {'target': 0}, [evidence()])
    assert reviews.history(path) == before
    reviews.save_review(path, 'target', 'model', '确认异常', '分歧后逐条判断',
                        expected_review_id=0, evidence_snapshot=evidence())
    reopened = reviews.history(path)
    assert reopened[:2] == before and len(reopened) == 3
    assert 'event_review' not in reopened[-1]['evidence_snapshot']


def test_batch_failure_rolls_back_all_reviews_and_evidence(tmp_path):
    path = tmp_path / 'reviews.sqlite'
    reviews.history(path)
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TRIGGER reject_second BEFORE INSERT ON review_evidence "
                   "WHEN NEW.review_id=2 BEGIN SELECT RAISE(ABORT, 'second evidence failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='second evidence failure'):
        save_event_reviews(path, 'model', '确认正常', '原子整簇', {'a': 0, 'b': 0},
                           [evidence('a'), evidence('b')])
    assert reviews.history(path) == []
    with closing(sqlite3.connect(path)) as db:
        assert db.execute('SELECT COUNT(*) FROM review_evidence').fetchone()[0] == 0


def test_verified_evidence_is_deeply_detached_and_corruption_keeps_judgment():
    payload = evidence()
    raw = reviews._snapshot_json(payload, 'target', 'model')
    row = dict(record_id='target', model_version='model', status='确认正常', note='保留依据',
               evidence_snapshot=payload, evidence_sha256=hashlib.sha256(raw.encode()).hexdigest())
    verified = reviews.review_with_evidence(row)
    assert verified['evidence_snapshot_status'] == 'saved'
    verified['evidence_snapshot']['scores']['rf'] = .8
    assert payload['scores']['rf'] == .2
    payload['scores']['rf'] = .9
    invalid = reviews.review_with_evidence(row)
    assert invalid['evidence_snapshot_status'] == 'invalid'
    assert invalid['evidence_snapshot'] is None
    assert invalid['status'] == row['status'] and invalid['note'] == row['note']


def test_single_schema_tag_comes_from_frozen_bytes(tmp_path, monkeypatch):
    path = tmp_path / 'reviews.sqlite'
    payload = evidence()
    frozen = deepcopy(payload)
    original = reviews.connect
    def mutate_after_capture(path):
        payload['schema_version'] = 1
        payload['scores']['rf'] = .99
        return original(path)
    monkeypatch.setattr(reviews, 'connect', mutate_after_capture)
    reviews.save_review(path, 'target', 'model', '确认正常', '冻结依据', evidence_snapshot=payload)
    saved = reviews.history(path)[0]
    assert saved['evidence_schema_version'] == 2
    assert saved['evidence_snapshot'] == frozen


@pytest.mark.parametrize('stamp,beijing,expected', [
    ('2020-01-01T00:00:00.123456+09:00', False, '2019-12-31T15:00:00.123456+00:00'),
    ('2020-01-01T00:00:00+09:00', True, '2019-12-31 23:00:00'),
    ('2400-01-01T00:00:00+00:00', True, '2400-01-01 08:00:00'),
    ('0001-01-01T00:00:00+01:00', False, '无法换算为 UTC；原始时间 0001-01-01T00:00:00+01:00'),
    ('9999-12-31T20:00:00+00:00', True, '无法换算为北京时间；原始时间 9999-12-31T20:00:00+00:00'),
])
def test_timezone_safe_conversion(stamp, beijing, expected):
    assert format_review_time(stamp, beijing=beijing) == expected


def test_naive_time_is_rejected():
    with pytest.raises(ValueError, match='时区'):
        format_review_time('2020-01-01T00:00:00')


@pytest.mark.parametrize('update', [
    {'mode': 'other'}, {'event_id': True}, {'event_id': -1}, {'event_id': ' '},
    {'representative_review_ids': []}, {'representative_review_ids': [True]},
    {'representative_review_ids': [0]}, {'representative_review_ids': [1, 1]},
    {'representative_review_ids': ['1']}, {'representative_review_ids': (1,)},
])
def test_incomplete_event_never_claims_consensus(update):
    event = dict(mode='representative_consensus', event_id=0, representative_review_ids=[1, 2])
    assert valid_saved_event_review(event)
    event.update(update)
    payload = evidence()
    payload['event_review'] = event
    before = deepcopy(payload)
    assert not valid_saved_event_review(event)
    summary = saved_evidence_summary(payload)
    assert '事件复核信息不完整' in summary and '代表一致确认' not in summary
    assert payload == before


@pytest.mark.parametrize('world', [
    {}, None, [], {'state': [], 'alpha': True, 'guarantee_status': '有效'},
    {'state': '正常', 'alpha': float('nan'), 'guarantee_status': '有效'},
    {'state': '正常', 'alpha': 0, 'guarantee_status': '有效'},
    {'state': '正常', 'alpha': 1, 'guarantee_status': '有效'},
    {'state': '正常', 'alpha': 10**400, 'guarantee_status': '有效'},
    {'state': '正常', 'alpha': .05, 'guarantee_status': [], 'b2_mode': ['weighted']},
])
def test_invalid_optional_engine_fields_are_never_filled(world):
    payload = evidence()
    payload['engine'] = {'status': 'verified', 'open_world': world}
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    lines = saved_engine_summary(payload)
    assert any('不完整' in line for line in lines)
    assert not any('比例不超过' in line for line in lines)
    assert json.dumps(payload, ensure_ascii=False, sort_keys=True) == raw


def test_saved_event_context_takes_priority_without_borrowing_current_engine():
    payload = evidence()
    payload['engine'] = dict(status='verified', open_world=dict(
        state='攻击', guarantee_status='有效', alpha=.05))
    payload['event_review'] = dict(mode='representative_consensus', event_id=0,
                                   representative_review_ids=[1])
    assert saved_engine_summary(payload) == ['旧快照未记录引擎四态与批次保证状态。']
    payload['event_review']['open_world'] = dict(state='新异', guarantee_status='失效', alpha=.05)
    lines = saved_engine_summary(payload)
    assert '保存时引擎四态：新异' in lines
    assert not any('四态：攻击' in line for line in lines)
    assert any('不在已知攻击放行风险' in line for line in lines)
