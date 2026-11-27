import json
import sqlite3
from contextlib import closing
from dataclasses import replace
from uuid import uuid4

import pytest

from app_engine_evidence import IDENTITY_COLUMNS
from app_types import EngineStatus
from liushao import dynamic, recovery
from liushao.gate import GateConfirmation, GateState, MemberState, evaluate_gate
from liushao.reviews import review_with_evidence, upgrade_schema
from liushao.search import (
    ANOMALY, NORMAL, AnalysisContext, Candidate, CheckBasis, HumanReview, SearchResult, SearchScope,
)


CONTEXT = AnalysisContext('a' * 64, 'model', 'b' * 64, 'c' * 64,
                          ('engine', 'descriptor', 'method'), .05)
SCOPE = dynamic.Scope('event-workbench', CONTEXT.model_version, frozenset({'rep', 'a', 'b'}))
ROWS = {'rep': 2, 'a': 3, 'b': 4, 'c': 5}


def review(record_id, *, status=NORMAL, expected=0, context=CONTEXT, note='核对原始记录'):
    evidence = dict(record_id=record_id, model_version=context.model_version,
                    method_version='review-evidence-v1', input_sha256=context.input_sha256,
                    model_sha256=context.model_sha256, descriptor_sha256=context.descriptor_sha256,
                    input_csv_row=ROWS[record_id], cards=[],
                    scores=dict(rf=.1, lr=.1, rf_threshold=.5, lr_threshold=.5),
                    engine=dict(status='verified', identity=dict(zip(IDENTITY_COLUMNS, context.engine_identity)),
                                open_world=dict(alpha=context.risk_alpha)))
    return dynamic.ReviewChange(record_id, context.model_version, status, note, expected, evidence)


class AnalysisSource:
    def __init__(self):
        self.context = CONTEXT
        self.event = 3
        self.budget = None
        self.proposal = frozenset({'a', 'b'})
        self.conclusion = NORMAL
        self.pending = frozenset()
        self.analysis_revision = 1
        self.evidence_revision = 1
        self.unavailable = False

    def gate_state(self, scope, reviews, *, proposal, conclusion, budget):
        candidates, supporting, opposing, pending = [], [], [], []
        reps = frozenset({'rep'}) & scope.record_ids
        for rid in sorted(scope.record_ids):
            row = reviews.get(rid, {})
            saved = review_with_evidence(row)['evidence_snapshot']
            if saved and row['status'] in (NORMAL, ANOMALY):
                supports = conclusion == row['status']
                human = HumanReview(rid, row['id'], row['status'], row['note'], self.context,
                                    () if supports else ('event',), ('event',) if supports else (),
                                    not supports and rid in reps | proposal)
                (supporting if supports else opposing).append(human)
            else:
                candidate = Candidate(rid, ('event',), 'review', ROWS[rid], ROWS[rid],
                                      CheckBasis(self.context, ('event',), None, ()), ('待核查',), 3)
                (pending if rid in self.pending else candidates).append(candidate)
        search = SearchResult(self.context, SearchScope(self.event, conclusion, reps, proposal),
                              tuple(candidates), budget, tuple(opposing), tuple(supporting),
                              tuple(pending), (), True, ())
        return GateState(search, {rid: MemberState('待核查', True) for rid in scope.record_ids},
                         self.context, self.analysis_revision, self.evidence_revision,
                         EngineStatus('ok', ''), budget=budget)

    def __call__(self, scope, inputs):
        if self.unavailable:
            return None
        # Dynamic reads Recovery's conclusions from the same snapshot as reviews.
        assert all(isinstance(view, recovery.DecisionView) for view in inputs.decisions.values())
        state = self.gate_state(scope, inputs.reviews, proposal=self.proposal & scope.record_ids,
                                conclusion=self.conclusion, budget=self.budget)
        rechecks = {}
        for auth in inputs.authorizations:
            original = json.loads(auth.original)['gate']
            proposal = frozenset(original['search']['scope']['authorization_ids']) & scope.record_ids
            original_state = self.gate_state(scope, inputs.reviews, proposal=proposal,
                                             conclusion=original['conclusion'], budget=original['budget'])
            rechecks[auth.batch_id] = evaluate_gate(original_state, inputs.reviews)
        return dynamic.Analysis(state.search, evaluate_gate(state, inputs.reviews), rechecks)


@pytest.fixture
def session():
    suffix = uuid4().hex
    main = f'file:reviews-{suffix}?mode=memory&cache=shared'
    ledger = f'file:ledger-{suffix}?mode=memory&cache=shared'
    with closing(sqlite3.connect(main, uri=True)) as db, closing(sqlite3.connect(main, uri=True)) as other:
        upgrade_schema(db)
        for connection in (db, other):
            connection.execute('ATTACH DATABASE ? AS recovery', (ledger,))
            recovery.initialize_recovery(connection)
        yield db, other, AnalysisSource()


def refresh(db, source, *changes, scope=SCOPE, expected=None):
    return dynamic.rebuild(db, scope, source, reason='复核或研判材料更新',
                           reviews=changes, expected=expected)[0]


def authorize(db, source):
    initial = refresh(db, source, review('rep'))
    gate = initial.preview.gate
    preview = recovery.preview_inheritance(db, gate.search, gate)
    confirmation = GateConfirmation('batch', NORMAL, gate.targets, '确认事件范围', True)

    def load_current(connection):
        dynamic.revalidate(connection, initial.preview.revision, source)
        reviews = {'rep': recovery._read_current_reviews(connection, SCOPE.model_version, ['rep'])['rep']}
        return source.gate_state(SCOPE, reviews, proposal=source.proposal, conclusion=NORMAL, budget=None)

    recovery.submit_inheritance(db, preview, confirmation, load_current=load_current)
    return refresh(db, source)


def by_id(current):
    return {row.record_id: row for row in current.records}


@pytest.mark.parametrize('status', [ANOMALY, '待补证'])
def test_representative_change_persists_loss_and_withdrawal_entry(session, status):
    db, other, source = session
    previous = authorize(db, source)
    rep_id = by_id(previous)['rep'].decision.review_id
    result = refresh(db, source, review('rep', status=status, expected=rep_id))
    assert result.preview.gate.blockers
    uncovered = ('a', 'b', 'rep') if status == '待补证' else ('a', 'b')
    assert result.preview.gate.uncovered_members == uncovered
    result = by_id(dynamic.read_current(other, SCOPE, source))
    assert [result[rid].decision.state for rid in ('a', 'b')] == [recovery.RecoveryState.INVALIDATED] * 2
    assert [result[rid].withdrawal_batch for rid in ('a', 'b')] == ['batch', 'batch']
    assert all(recovery.current_decision(other, SCOPE.model_version, rid) == result[rid].decision
               for rid in ('a', 'b'))
    withdrawal = recovery.preview_withdrawal(other, result['b'].withdrawal_batch)
    assert [member.record_id for member in withdrawal.members if member.disposition == 'affected'] == ['a', 'b']
    entry = recovery.operation_history(other, withdrawal.members[0].validity_revision)
    assert entry['kind'] == 'invalidate'
    assert entry['reason'] and {row['record_id'] for row in entry['evidence']['affected'].values()} == {'a', 'b'}


def test_reliable_new_counterexample_invalidates_authorized_sibling(session):
    db, other, source = session
    authorize(db, source)
    now = refresh(db, source, review('a', status=ANOMALY))
    assert now.preview.search.confirmed_counterexamples[0].record_id == 'a'
    assert by_id(now)['a'].decision.state == recovery.RecoveryState.DIRECT
    assert by_id(now)['b'].decision.state == recovery.RecoveryState.INVALIDATED
    assert recovery.current_decision(other, SCOPE.model_version, 'b').conclusion is None


def test_added_support_expires_preview_without_invalidating_authorization(session):
    db, other, source = session
    previous = authorize(db, source)
    rep_id = by_id(previous)['rep'].decision.review_id
    result = refresh(db, source, review('rep', expected=rep_id, note='补充已核验的依据'))
    assert result.preview.gate.passed
    assert result.preview.gate.supporting_reviews[0].review_id != rep_id
    assert by_id(result)['a'].decision.state == recovery.RecoveryState.INHERITED
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.revalidate(other, previous.preview.revision, source)


def test_scope_rebuild_uses_stable_members_when_event_number_changes(session):
    db, other, source = session
    before = authorize(db, source)
    source.event = 73
    source.analysis_revision += 1
    rebuilt = refresh(db, source)
    assert rebuilt.preview.search.scope.event_id == 73
    assert by_id(rebuilt)['a'].decision.state == recovery.RecoveryState.INHERITED
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.revalidate(other, before.preview.revision, source)
    moved = replace(SCOPE, record_ids=frozenset({'rep', 'a', 'c'}))
    source.context = replace(CONTEXT, risk_alpha=.1)
    rep_id = by_id(rebuilt)['rep'].decision.review_id
    result = refresh(db, source, review('rep', expected=rep_id, context=source.context), scope=moved)
    assert result.preview.search.context.risk_alpha == .1
    assert {row.record_id for row in result.records} == moved.record_ids
    assert recovery.current_decision(other, SCOPE.model_version, 'b').state == recovery.RecoveryState.INVALIDATED
    assert by_id(result)['rep'].decision.state == recovery.RecoveryState.DIRECT
    assert by_id(result)['a'].decision.state == recovery.RecoveryState.INVALIDATED


@pytest.mark.parametrize('change', ['budget', 'targets', 'conclusion', 'checks', 'evidence'])
def test_preview_dependencies_are_revalidated_from_current_server_materials(session, change):
    db, other, source = session
    before = refresh(db, source, review('rep')).preview
    if change == 'budget':
        source.budget = 1
    elif change == 'targets':
        source.proposal = frozenset({'a'})
    elif change == 'conclusion':
        source.conclusion = ANOMALY
    elif change == 'evidence':
        source.evidence_revision += 1
    else:
        source.pending = frozenset({'b'})
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.revalidate(other, before.revision, source)
    latest = refresh(db, source).preview
    assert dynamic.revalidate(other, latest.revision, source) == latest


def test_budget_and_draft_do_not_revoke_saved_inheritance(session):
    db, other, source = session
    before = authorize(db, source)
    source.budget = 0
    source.proposal = frozenset({'a'})
    source.conclusion = ANOMALY
    current = dynamic.read_current(other, SCOPE, source)
    assert current.preview.gate.blockers
    assert current.preview.revision != before.preview.revision
    assert by_id(current)['b'].decision.state == recovery.RecoveryState.INHERITED
    assert by_id(refresh(db, source))['b'].decision.state == recovery.RecoveryState.INHERITED


def test_export_freezes_revisions_and_history_at_preparation(session):
    db, other, source = session
    before = authorize(db, source)
    first = dynamic.prepare_export(other, SCOPE, source)
    assert first.revision == before.preview.revision
    assert first.confirmed_count == 3
    assert first.checks[0].inheritance_source == 'batch'
    assert first.checks[0].physical_row == ROWS['a']
    assert dynamic.download_export(db, first, source) is first
    refresh(db, source, review('a', status=ANOMALY))
    second = dynamic.prepare_export(other, SCOPE, source)
    assert [row.record_id for row in second.conclusions] == ['a', 'rep']
    assert second.confirmed_count == 2 and first.confirmed_count == 3
    assert second.checks[1].decision.state == recovery.RecoveryState.INVALIDATED
    assert second.checks[1].decision.conclusion is None
    assert second.checks[1].decision.inheritance_id is None
    assert second.checks[1].inheritance_source is None
    assert json.loads(second.checks[0].evidence_json)['input_csv_row'] == ROWS['a']
    assert [json.loads(row).get('kind') for row in second.history][-2:] == ['authorize', 'invalidate']
    assert second.exported_at.endswith('+00:00')
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.download_export(db, first, source)


def test_another_window_withdrawal_invalidates_download_and_preview(session):
    db, other, source = session
    before = authorize(db, source)
    frozen = dynamic.prepare_export(db, SCOPE, source)
    recovery.withdraw(other, recovery.preview_withdrawal(other, 'batch'), operation_id='withdraw',
                      reason='撤回继承', confirmed=True)
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.download_export(db, frozen, source)
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.revalidate(db, before.preview.revision, source)
    refresh(db, source)
    result = dynamic.prepare_export(db, SCOPE, source)
    assert [row.record_id for row in result.conclusions] == ['rep']
    assert {row.decision.state for row in result.checks[:2]} == {recovery.RecoveryState.PENDING}
    assert json.loads(result.history[-1])['kind'] == 'withdraw'


@pytest.mark.parametrize('defer', ['marked', 'unavailable', 'outside_review'])
def test_unverified_reads_hide_inheritance_across_windows(session, defer):
    db, other, source = session
    before = authorize(db, source)
    if defer == 'marked':
        dynamic.mark_pending(other, SCOPE, reason='等待可信更新')
    elif defer == 'unavailable':
        source.unavailable = True
    else:
        # Simulate a legacy writer using the same underlying review store.
        with other:
            other.execute('INSERT INTO reviews (model_version,record_id,status,note,created_at) VALUES (?,?,?,?,?)',
                          (SCOPE.model_version, 'rep', '待补证', '等待来源证据', '2026-11-25'))
    result = dynamic.read_current(db, SCOPE, source)
    assert result.preview is None
    assert by_id(result)['a'].decision == recovery.DecisionView(
        recovery.RecoveryState.PENDING, action='review_record')
    assert by_id(result)['a'].inheritance_source is None
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.prepare_export(db, SCOPE, source)
    source.unavailable = False
    refreshed = refresh(db, source)
    expected = recovery.RecoveryState.INVALIDATED if defer == 'outside_review' else recovery.RecoveryState.INHERITED
    assert by_id(refreshed)['a'].decision.state == expected
    with pytest.raises(dynamic.RefreshRequired):
        dynamic.revalidate(db, before.preview.revision, source)


def test_review_validity_and_revision_roll_back_together(session):
    db, other, source = session
    before = authorize(db, source)
    original_rep = by_id(before)['rep'].decision
    calls = 0

    def fail_after_invalidation(scope, inputs):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert inputs.decisions['a'].state == recovery.RecoveryState.INVALIDATED
            raise RuntimeError('模拟生成预览失败')
        return source(scope, inputs)

    with pytest.raises(RuntimeError, match='模拟生成预览失败'):
        refresh(db, fail_after_invalidation, review('rep', status=ANOMALY, expected=original_rep.review_id))
    assert recovery.current_decision(other, SCOPE.model_version, 'rep') == original_rep
    assert recovery.current_decision(other, SCOPE.model_version, 'a').state == recovery.RecoveryState.INHERITED
    assert other.execute("SELECT COUNT(*) FROM recovery.operations WHERE kind='invalidate'").fetchone()[0] == 0
    assert dynamic.revalidate(other, before.preview.revision, source) == before.preview


def test_stale_editor_cannot_write_review_after_another_window_rebuilds(session):
    db, other, source = session
    before = authorize(db, source)
    source.budget = 2
    refresh(other, source)
    with pytest.raises(dynamic.RefreshRequired):
        refresh(db, source, review('a', status=ANOMALY), expected=before.preview.revision)
    assert recovery.current_decision(other, SCOPE.model_version, 'a').state == recovery.RecoveryState.INHERITED


def test_pending_authorization_is_shared_by_overlapping_scopes(session):
    db, other, source = session
    authorize(db, source)
    overlapping = replace(SCOPE, key='same-authorization-other-view')
    refresh(other, source, scope=overlapping)
    dynamic.mark_pending(db, SCOPE, reason='等待原范围重新核验')
    assert by_id(dynamic.read_current(other, overlapping, source))['a'].decision.state == recovery.RecoveryState.PENDING
    with pytest.raises(dynamic.RefreshRequired):
        refresh(other, source, scope=overlapping)
    assert by_id(refresh(db, source))['a'].decision.state == recovery.RecoveryState.INHERITED
    assert by_id(refresh(other, source, scope=overlapping))['a'].decision.state == recovery.RecoveryState.INHERITED


def test_review_change_outside_unrelated_scope_keeps_its_revision(session):
    db, other, source = session
    authorize(db, source)
    unrelated = replace(SCOPE, key='unrelated', record_ids=frozenset({'c'}))
    first = refresh(other, source, scope=unrelated).preview
    rep_id = recovery.current_decision(db, SCOPE.model_version, 'rep').review_id
    refresh(db, source, review('rep', expected=rep_id, status=ANOMALY))
    assert dynamic.revalidate(other, first.revision, source) == first
