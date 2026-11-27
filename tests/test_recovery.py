import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from uuid import uuid4

import pytest

from app_engine_evidence import IDENTITY_COLUMNS
from app_types import EngineStatus
from liushao.gate import GateBlocked, GateConfirmation, GateState, MemberState, RiskRequirement, evaluate_gate
from liushao.recovery import (
    EvidenceRef, RecoveryConflict, RecoveryState, current_decision, inheritance_history,
    initialize_recovery, invalidate, operation_history, preview_inheritance, preview_withdrawal,
    submit_inheritance, withdraw,
)
from liushao.reviews import _snapshot_json, schema_kind, upgrade_schema
from liushao.search import (
    ANOMALY, NORMAL, AnalysisContext, Candidate, CheckBasis, Finding, HumanReview, SearchResult, SearchScope,
)


CONTEXT = AnalysisContext('a' * 64, 'base-v1', 'b' * 64, 'c' * 64,
                          ('engine', 'descriptor', 'method'), .05)
GOOD_RISK = Finding('applicability', 'guarantee_status', 'available', '有效', '有效')
LOCAL = Finding('authorization', 'asset_log', 'available', '资产日志可核查')


def _connect(review_db, ledger_db):
    db = sqlite3.connect(review_db, uri=True)
    try:
        if schema_kind(db) != 'v2':
            upgrade_schema(db)
        db.execute('ATTACH DATABASE ? AS recovery', (ledger_db,))
        initialize_recovery(db)
        return db
    except BaseException:
        db.close()
        raise


@pytest.fixture
def db():
    with closing(_connect(':memory:', ':memory:')) as connection:
        yield connection


def _review(db, rid, status=NORMAL, *, provenance='direct'):
    note = '逐条核对原始记录'
    with db:
        cursor = db.execute('INSERT INTO reviews (record_id,model_version,status,note,created_at) VALUES (?,?,?,?,?)',
                            (rid, CONTEXT.model_version, status, note, '2026-11-25T00:00:00+00:00'))
        review_id = cursor.lastrowid
        row = dict(id=review_id, record_id=rid, model_version=CONTEXT.model_version, status=status, note=note)
        if provenance == 'missing':
            return row
        snapshot = dict(record_id=rid, model_version=CONTEXT.model_version, method_version='review-evidence-v1',
                        input_sha256=CONTEXT.input_sha256, model_sha256=CONTEXT.model_sha256,
                        descriptor_sha256=CONTEXT.descriptor_sha256, input_csv_row=2, cards=[],
                        scores=dict(rf=.1, lr=.1, rf_threshold=.5, lr_threshold=.5),
                        engine=dict(status='verified', identity=dict(zip(IDENTITY_COLUMNS, CONTEXT.engine_identity)),
                                    open_world=dict(alpha=CONTEXT.risk_alpha)))
        if provenance == 'legacy_inheritance':
            snapshot.update(method_version='o1-event-review-v1',
                            event_review={'event_id': 3, 'representative_review_ids': [1]})
        version = 9 if provenance == 'unknown' else 1
        snapshot['schema_version'] = version
        if provenance == 'unknown':
            # Simulate a legacy record stored before strict schema validation:
            # build the payload directly so Recovery's unknown-version rejection
            # can be exercised without the current serializer refusing it.
            payload = json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                 separators=(',', ':'), allow_nan=False)
        else:
            payload = _snapshot_json(snapshot, rid, CONTEXT.model_version)
        digest = hashlib.sha256(payload.encode()).hexdigest() if provenance != 'damaged' else '0' * 64
        db.execute('INSERT INTO review_evidence VALUES (?,?,?,?)', (review_id, version, payload, digest))
        return dict(row, evidence_json=payload, evidence_sha256=digest, evidence_schema_version=version)


def _gate(db, *, representative=None, targets=('a', 'b'), local=LOCAL, risk=GOOD_RISK):
    representative = _review(db, 'rep') if representative is None else representative
    human = HumanReview('rep', representative['id'], representative['status'], representative['note'], CONTEXT,
                        ('event',) if representative['status'] == ANOMALY else (),
                        ('event',) if representative['status'] == NORMAL else (), True)
    scope = SearchScope(3, NORMAL, frozenset({'rep'}), frozenset(targets))
    candidates = tuple(Candidate(rid, ('event',), '', index, index,
                                 CheckBasis(CONTEXT, ('event',), None, (local,)), ('局部证据',), priority=3)
                       for index, rid in enumerate(targets, 2))
    search = SearchResult(CONTEXT, scope, candidates, None,
                          (human,) if human.counterexample_paths else (), (human,) if human.supporting_paths else (),
                          (), (), True, (risk,))
    members = {'rep': MemberState('已直接确认', False)}
    members.update({rid: MemberState('待复核', True) for rid in targets})
    state = GateState(search, members, CONTEXT, 2, 7, EngineStatus('ok', ''),
                      risk_requirements=(RiskRequirement('guarantee_status', ('有效',)),))
    return state, evaluate_gate(state, {'rep': representative})


def _submit(db, state, preview, operation_id):
    confirmation = GateConfirmation(operation_id, NORMAL, preview.decision.targets, '确认当前授权范围', True)
    return submit_inheritance(db, preview, confirmation, load_current=lambda conn: state, actor='Iris')


def _authorize(db, state, gate, operation_id='batch'):
    preview = preview_inheritance(db, state.search, gate)
    return _submit(db, state, preview, operation_id)


def _current(db, rid):
    return current_decision(db, CONTEXT.model_version, rid)


def _withdraw(db, preview, operation_id='withdraw'):
    return withdraw(db, preview, operation_id=operation_id, reason='原批次证据需要复核', confirmed=True)


def test_pending_direct_and_new_direct_operation(db):
    assert _current(db, 'a').state == RecoveryState.PENDING
    first = _review(db, 'a')
    assert (_current(db, 'a').state, _current(db, 'a').review_id) == (RecoveryState.DIRECT, first['id'])
    second = _review(db, 'a', ANOMALY)
    assert (_current(db, 'a').conclusion, _current(db, 'a').review_id) == (ANOMALY, second['id'])
    _review(db, 'a', '待补证', provenance='missing')
    assert (_current(db, 'a').state, _current(db, 'a').action) == (RecoveryState.PENDING, 'supplement_evidence')


def test_gate_preview_remains_pending_and_exact_submission_replays(db):
    state, gate = _gate(db)
    preview = preview_inheritance(db, state.search, gate)
    assert all(_current(db, rid).state == RecoveryState.PENDING for rid in gate.targets)
    assert not _submit(db, state, preview, 'batch').replayed
    assert _submit(db, state, preview, 'batch').replayed
    assert all(_current(db, rid).state == RecoveryState.INHERITED for rid in gate.targets)
    rows = inheritance_history(db, 'batch')
    assert len({row['id'] for row in rows}) == 2
    assert all(EvidenceRef('representative', 'rep', True) in row['dependencies'] for row in rows)
    assert operation_history(db, 'batch')['actor'] == 'Iris'


@pytest.mark.parametrize('resolution', ['direct', 'new_gate', 'withdraw'])
def test_invalidated_member_can_be_resolved_without_reviving_original(db, resolution):
    state, gate = _gate(db)
    _authorize(db, state, gate)
    rep = _review(db, 'rep')
    failed, blocked = _gate(db, representative=rep, targets=('a',),
                           local=replace(LOCAL, state='invalid', reason='日志校验失败'))
    assert not blocked.passed
    invalidate(db, 'batch', EvidenceRef('asset_log', 'a'), failed.search, blocked,
               operation_id='invalid', reason='目标日志校验失败')
    assert _current(db, 'a').state == RecoveryState.INVALIDATED
    assert _current(db, 'a').conclusion is None
    assert _current(db, 'b').state == RecoveryState.INHERITED
    if resolution == 'direct':
        _review(db, 'a', ANOMALY)
        assert (_current(db, 'a').state, _current(db, 'a').conclusion) == (RecoveryState.DIRECT, ANOMALY)
    elif resolution == 'new_gate':
        new_state, new_gate = _gate(db, representative=rep, targets=('a',))
        preview_inheritance(db, new_state.search, new_gate)
        assert _current(db, 'a').state == RecoveryState.INVALIDATED
        _authorize(db, new_state, new_gate, 'replacement')
        assert _current(db, 'a').state == RecoveryState.INHERITED
        result = _withdraw(db, preview_withdrawal(db, 'batch'))
        assert [m['disposition'] for m in result['evidence']['impact']['members']] == ['other_authorization', 'affected']
        assert _current(db, 'a').state == RecoveryState.INHERITED
    else:
        result = _withdraw(db, preview_withdrawal(db, 'batch'))
        assert result['evidence']['validity'][0]['operation_id'] == 'invalid'
        assert _current(db, 'a').state == RecoveryState.PENDING
    expected = RecoveryState.WITHDRAWN if resolution == 'withdraw' else RecoveryState.INVALIDATED
    assert inheritance_history(db, 'batch')[0]['state'] == expected


@pytest.mark.parametrize('has_independent_review', [False, True])
def test_shared_gate_premise_invalidates_all_still_dependent_members(db, has_independent_review):
    state, gate = _gate(db)
    _authorize(db, state, gate)
    rep = _review(db, 'rep')
    failed, blocked = _gate(db, representative=rep, risk=replace(GOOD_RISK, state='invalid'))
    if has_independent_review:
        blocked = evaluate_gate(failed, {'rep': rep, 'b': _review(db, 'b', ANOMALY)})
    result = invalidate(db, 'batch', EvidenceRef('guarantee_status', shared=True), failed.search, blocked,
                        operation_id='invalid', reason='共同风险前提已失效')
    assert [member['record_id'] for member in result['evidence']['affected']] == (
        ['a'] if has_independent_review else ['a', 'b'])
    assert _current(db, 'a').state == RecoveryState.INVALIDATED
    assert _current(db, 'b').state == (RecoveryState.DIRECT if has_independent_review else RecoveryState.INVALIDATED)


def test_target_failure_cannot_be_used_as_a_same_named_batch_failure(db):
    local = replace(LOCAL, source=GOOD_RISK.source)
    rep = _review(db, 'rep')
    state, gate = _gate(db, representative=rep, local=local)
    _authorize(db, state, gate)
    failed, blocked = _gate(db, representative=rep, targets=('a',), local=replace(local, state='invalid'))
    with pytest.raises(ValueError, match='批次风险前提未变化'):
        invalidate(db, 'batch', EvidenceRef(local.source, shared=True), failed.search, blocked,
                   operation_id='invalid', reason='同名目标证据失效')
    invalidate(db, 'batch', EvidenceRef(local.source, 'a'), failed.search, blocked,
               operation_id='invalid', reason='目标证据失效')
    assert _current(db, 'a').state == RecoveryState.INVALIDATED
    assert _current(db, 'b').state == RecoveryState.INHERITED


def test_withdrawal_records_reason_links_evidence_and_skips_independent_review(db):
    state, gate = _gate(db)
    _authorize(db, state, gate)
    direct = _review(db, 'b', ANOMALY)
    preview = preview_withdrawal(db, 'batch')
    result = _withdraw(db, preview)
    assert (result['id'], result['batch_id'], result['reason'], result['actor']) == (
        'withdraw', 'batch', '原批次证据需要复核', '未记录')
    assert result['created_at'].endswith('+00:00')
    raw = db.execute('SELECT evidence FROM recovery.operations WHERE id=?', ('batch',)).fetchone()[0]
    assert result['evidence']['original'] == {'operation_id': 'batch',
                                            'evidence_sha256': hashlib.sha256(raw.encode()).hexdigest()}
    assert result['evidence']['reviews']['b']['evidence_sha256'] == direct['evidence_sha256']
    assert [member['disposition'] for member in result['evidence']['impact']['members']] == ['affected', 'direct_review']
    assert _current(db, 'a').state == RecoveryState.PENDING
    assert _current(db, 'b').conclusion == ANOMALY
    assert [row['state'] for row in inheritance_history(db, 'batch')] == [RecoveryState.WITHDRAWN, RecoveryState.INHERITED]
    assert _withdraw(db, preview) == _withdraw(db, preview, 'repeated') == result
    assert operation_history(db, 'repeated') is None


@pytest.mark.parametrize('resolution', ['direct', 'new_gate'])
def test_withdrawn_record_can_receive_a_new_conclusion(db, resolution):
    rep = _review(db, 'rep')
    state, gate = _gate(db, representative=rep)
    _authorize(db, state, gate)
    _withdraw(db, preview_withdrawal(db, 'batch'))
    if resolution == 'direct':
        _review(db, 'a')
        assert _current(db, 'a').state == RecoveryState.DIRECT
    else:
        _authorize(db, state, gate, 'new-batch')
        assert _current(db, 'a').state == RecoveryState.INHERITED
    assert inheritance_history(db, 'batch')[0]['state'] == RecoveryState.WITHDRAWN


def test_illegal_transitions_cannot_create_or_revive_inheritance(db):
    rep = _review(db, 'rep')
    state, gate = _gate(db, representative=rep)
    _authorize(db, state, gate)
    with pytest.raises(RecoveryConflict, match='已有有效结论'):
        preview_inheritance(db, state.search, gate)
    with pytest.raises(ValueError, match='未判定'):
        invalidate(db, 'batch', EvidenceRef('asset_log', 'a'), state.search, gate,
                   operation_id='invalid', reason='Gate 尚未判定失效')
    preview = preview_withdrawal(db, 'batch')
    with pytest.raises(ValueError, match='明确确认'):
        withdraw(db, preview, operation_id='withdraw', reason='复核', confirmed=False)
    with pytest.raises(RecoveryConflict, match='另一份确认'):
        _withdraw(db, preview, 'batch')
    assert operation_history(db, 'invalid') is None
    assert _current(db, 'a').state == RecoveryState.INHERITED


def test_gate_rejects_submit_after_a_concurrent_direct_review(db):
    state, gate = _gate(db)
    preview = preview_inheritance(db, state.search, gate)
    _review(db, 'a', ANOMALY)
    with pytest.raises(GateBlocked):
        _submit(db, state, preview, 'stale')
    assert _current(db, 'a').state == RecoveryState.DIRECT
    assert _current(db, 'b').state == RecoveryState.PENDING
    assert operation_history(db, 'stale') is None


def test_concurrent_recovery_revision_requires_repreview_even_after_withdrawal(db):
    state, gate = _gate(db)
    preview = preview_inheritance(db, state.search, gate)
    _authorize(db, state, gate)
    _withdraw(db, preview_withdrawal(db, 'batch'))
    with pytest.raises(RecoveryConflict, match='继承修订已变化'):
        _submit(db, state, preview, 'stale')
    assert _current(db, 'a').state == RecoveryState.PENDING
    assert operation_history(db, 'stale') is None


def test_withdrawal_rejects_stale_preview_across_two_connections():
    review_db = f'file:reviews-{uuid4().hex}?mode=memory&cache=shared'
    ledger_db = f'file:recovery-{uuid4().hex}?mode=memory&cache=shared'
    with closing(_connect(review_db, ledger_db)) as first, closing(_connect(review_db, ledger_db)) as second:
        state, gate = _gate(first)
        _authorize(first, state, gate)
        preview = preview_withdrawal(first, 'batch')
        _review(second, 'b', ANOMALY)
        with pytest.raises(RecoveryConflict, match='影响范围或复核修订已变化'):
            _withdraw(first, preview)
        assert _current(first, 'a').state == RecoveryState.INHERITED
        assert _current(first, 'b').state == RecoveryState.DIRECT
        assert operation_history(second, 'withdraw') is None
        _withdraw(first, preview_withdrawal(first, 'batch'))
        assert _current(second, 'a').state == RecoveryState.PENDING
        assert _current(second, 'b').conclusion == ANOMALY


def test_withdrawal_rolls_back_audit_and_first_update_if_second_update_fails(db):
    state, gate = _gate(db)
    _authorize(db, state, gate)
    db.execute("CREATE TRIGGER recovery.fail_withdrawal BEFORE UPDATE ON inheritances WHEN NEW.record_id='b' "
               "BEGIN SELECT RAISE(ABORT, 'test write failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match='test write failure'):
        _withdraw(db, preview_withdrawal(db, 'batch'))
    assert operation_history(db, 'withdraw') is None
    assert all(_current(db, rid).state == RecoveryState.INHERITED for rid in gate.targets)


def test_later_pending_review_does_not_reactivate_an_old_inheritance(db):
    state, gate = _gate(db)
    _authorize(db, state, gate)
    _review(db, 'a', ANOMALY)
    _withdraw(db, preview_withdrawal(db, 'batch'))
    _review(db, 'a', '待补证', provenance='missing')
    assert _current(db, 'a').state == RecoveryState.PENDING
    assert _current(db, 'a').conclusion is None


@pytest.mark.parametrize(('provenance', 'expected', 'snapshot_status'), [
    ('missing', RecoveryState.PENDING, 'missing'),
    ('damaged', RecoveryState.PENDING, 'invalid'),
    ('legacy_inheritance', RecoveryState.INVALIDATED, 'saved'),
    ('direct', RecoveryState.DIRECT, 'saved'),
])
def test_legacy_evidence_is_preserved_and_only_verified_direct_confirmation_survives(
        db, provenance, expected, snapshot_status):
    row = _review(db, 'a', provenance=provenance)
    view = _current(db, 'a')
    assert (view.state, view.review_id, view.snapshot_status) == (expected, row['id'], snapshot_status)
    raw = db.execute('SELECT payload_json FROM review_evidence WHERE review_id=?', (row['id'],)).fetchone()
    assert (raw[0] if raw else None) == row.get('evidence_json')
    if expected != RecoveryState.DIRECT:
        assert view.action in ('review_record', 'supplement_evidence')
        assert view.conclusion is None


def test_unknown_snapshot_version_is_rejected_without_rewriting_it(db):
    row = _review(db, 'a', provenance='unknown')
    with pytest.raises(ValueError, match='未知复核证据快照版本'):
        _current(db, 'a')
    assert json.loads(row['evidence_json'])['schema_version'] == 9


@pytest.mark.parametrize('provenance', ['missing', 'legacy_inheritance'])
def test_legacy_unverified_or_inherited_conclusion_cannot_authorize_as_representative(db, provenance):
    representative = _review(db, 'rep', provenance=provenance)
    state, gate = _gate(db, representative=representative)
    assert not gate.passed
    with pytest.raises(ValueError, match='Gate 未通过'):
        preview_inheritance(db, state.search, gate)
    assert _current(db, 'a').state == RecoveryState.PENDING
