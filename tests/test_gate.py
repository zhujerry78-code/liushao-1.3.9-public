import hashlib
import sqlite3
from dataclasses import replace

import pytest

from app_engine_evidence import IDENTITY_COLUMNS
from app_types import EngineStatus
from liushao.gate import (
    GateBlocked, GateConfirmation, GateState, MemberState, ReviewWork, RiskRequirement, WorkflowActivity,
    evaluate_gate, submit_gate,
)
from liushao.reviews import _snapshot_json, upgrade_schema
from liushao.search import (
    ANOMALY, NORMAL, AnalysisContext, Candidate, CheckBasis, Finding, HumanReview,
    PriorCheck, SearchResult, SearchScope,
)


CONTEXT = AnalysisContext('a' * 64, 'base-v1', 'b' * 64, 'c' * 64,
                          tuple(f'engine-{i}' for i in range(len(IDENTITY_COLUMNS))), .05)
DIAGNOSTICS = (Finding('applicability', 'guarantee_status', 'available', '有效', '有效'),)


def candidate(rid='member', *, path='event', evidence=(), priority=3, prior=None):
    return Candidate(rid, (path,), '', 4, 1, CheckBasis(CONTEXT, (path,), None, evidence),
                     ('搜索依据',), priority=priority, prior_check=prior)


def direct_review(rid, status=NORMAL, review_id=1):
    snapshot = dict(record_id=rid, model_version=CONTEXT.model_version, method_version='fixture',
                    input_sha256=CONTEXT.input_sha256, model_sha256=CONTEXT.model_sha256,
                    descriptor_sha256=CONTEXT.descriptor_sha256, input_csv_row=2, cards=[],
                    scores=dict(rf=.1, lr=.1, rf_threshold=.5, lr_threshold=.5),
                    identity=dict(source_file='synthetic.csv', source_sha256='d' * 64, source_row=2),
                    engine=dict(status='verified', identity=dict(zip(IDENTITY_COLUMNS, CONTEXT.engine_identity)),
                                open_world=dict(alpha=CONTEXT.risk_alpha)))
    row = dict(id=review_id, record_id=rid, model_version=CONTEXT.model_version, status=status,
               note='逐条检查了原始证据', created_at='2026-11-25', evidence_snapshot=snapshot)
    payload = _snapshot_json(snapshot, rid, CONTEXT.model_version)
    return dict(row, evidence_sha256=hashlib.sha256(payload.encode()).hexdigest())


def event_state(*, ordered=None, budget=None):
    rep = direct_review('rep')
    representative = HumanReview('rep', 1, NORMAL, rep['note'], CONTEXT, (), ('event',), False)
    scope = SearchScope(3, NORMAL, frozenset({'rep'}), frozenset({'member'}))
    search = SearchResult(CONTEXT, scope, (candidate(),) if ordered is None else ordered, budget,
                          (), (representative,), (), (), True, DIAGNOSTICS)
    members = {'rep': MemberState('已直接确认', False), 'member': MemberState('待复核', True)}
    return GateState(search, members, CONTEXT, 2, 7, EngineStatus('ok', '')), {'rep': rep}


def release_state():
    scope = SearchScope(pending_release_ids=frozenset({'release'}))
    search = SearchResult(CONTEXT, scope, (candidate('release', path='release'),), None,
                          (), (), (), (), True, DIAGNOSTICS)
    return GateState(search, {'release': MemberState('队列外', True)}, CONTEXT, 2, 7, EngineStatus('ok', ''))


def test_gate_blocks_search_confirmed_direct_conflict_with_review_link():
    state, reviews = event_state(ordered=())
    reviews['member'] = direct_review('member', ANOMALY, 2)
    conflict = HumanReview('member', 2, ANOMALY, reviews['member']['note'], CONTEXT,
                           ('event',), (), True)
    state = replace(state, search=replace(state.search, confirmed_counterexamples=(conflict,)))
    result = evaluate_gate(state, reviews)
    block = next(item for item in result.blockers if item.code == 'direct_conflict')
    assert not result.passed
    assert (block.entry.record_id, block.entry.review_id, block.entry.action) == ('member', 2, 'review_record')
    assert result.counterexamples == (conflict,)
    assert result.targets == ()


def test_gate_blocks_when_representative_has_no_current_direct_confirmation():
    state, _ = event_state()
    result = evaluate_gate(state, {})
    assert not result.passed
    assert any(item.code == 'required_check' and item.entry.record_id == 'rep' for item in result.blockers)


def test_out_of_scope_counterexample_is_retained_within_the_preview():
    state, reviews = event_state()
    reviews['excluded'] = direct_review('excluded', ANOMALY, 2)
    conflict = HumanReview('excluded', 2, ANOMALY, reviews['excluded']['note'], CONTEXT,
                           ('event',), (), False)
    state = replace(state, members=dict(state.members, excluded=MemberState('已直接确认', False)),
                    search=replace(state.search, confirmed_counterexamples=(conflict,)))
    result = evaluate_gate(state, reviews)
    assert result.passed
    assert result.targets == ('member',)
    assert result.counterexamples == (conflict,)
    assert ('excluded', '已有有效的直接判断') in result.exclusions


def test_gate_blocks_required_check_even_outside_the_search_budget():
    opposite = Finding('prediction_direction', '引擎四态', 'available', '指向异常', ANOMALY)
    required = candidate(evidence=(opposite,), priority=1)
    state, reviews = event_state(ordered=(required,), budget=0)
    result = evaluate_gate(state, reviews)
    block = next(item for item in result.blockers if item.code == 'required_check')
    assert not result.passed
    assert result.required_checks == (required,)
    assert not result.recommendations
    assert (block.entry.record_id, block.entry.evidence_source, block.entry.physical_row) == ('member', '引擎四态', 4)
    assert result.workload.additional_checks == 1


def test_gate_can_use_retained_check_with_matching_evidence_and_reason():
    opposite = Finding('prediction_direction', '引擎四态', 'available', '指向异常', ANOMALY)
    checked = candidate(evidence=(opposite,), priority=None)
    receipt = PriorCheck(checked.basis, 'checked', '排查了造成反向提示的流量特征')
    checked = replace(checked, prior_check=receipt)
    state, reviews = event_state(ordered=())
    state = replace(state, search=replace(state.search, retained_checks=(checked,)),
                    work=(ReviewWork('member', 'round-1', frozenset({'search_check'})),))
    assert evaluate_gate(state, reviews).completed_checks == (checked,)
    assert evaluate_gate(state, reviews).passed
    changed = replace(checked, basis=replace(checked.basis, evidence=(replace(opposite, reason='新证据'),)))
    state = replace(state, search=replace(state.search, retained_checks=(changed,)))
    assert not evaluate_gate(state, reviews).passed


def test_gate_blocks_material_heterogeneity_and_pending_evidence():
    evidence = Finding('event_heterogeneity', 'member_comparison', 'signal', '未覆盖形态')
    novel = candidate('other', evidence=(evidence,), priority=2)
    pending = candidate(priority=None)
    state, reviews = event_state(ordered=(novel,))
    state = replace(state, members=dict(state.members, other=MemberState('待复核', True)),
                    search=replace(state.search, pending_verification=(pending,)))
    result = evaluate_gate(state, reviews)
    assert not result.passed
    assert {(item.code, item.entry.record_id) for item in result.blockers} >= {
        ('required_check', 'other'), ('critical_evidence', 'member'),
    }


def test_gate_reports_each_unavailable_authorization_source():
    invalid = Finding('applicability', 'guarantee_status', 'invalid', '风险前提失效')
    evidence = (Finding('authorization', 'guarantee_status', 'unknown', '关键风险前提', invalid),
                Finding('authorization', 'asset_log', 'unknown', '缺少资产日志'))
    state, reviews = event_state(ordered=(candidate(evidence=evidence, priority=1),))
    result = evaluate_gate(state, reviews)
    assert not result.passed
    assert {(item.entry.record_id, item.entry.evidence_source, item.entry.action)
            for item in result.blockers if item.code == 'critical_evidence'} == {
        ('member', 'guarantee_status', 'supplement_evidence'),
        ('member', 'asset_log', 'supplement_evidence'),
    }


def test_gate_does_not_overwrite_direct_review_absent_from_search():
    state, reviews = event_state()
    reviews['member'] = direct_review('member', review_id=2)
    result = evaluate_gate(state, reviews)
    assert not result.passed
    assert any(item.code == 'critical_evidence' and item.entry.review_id == 2 for item in result.blockers)


def test_gate_checks_risk_premises_when_authorization_depends_on_them():
    state = release_state()
    state = replace(state, risk_requirements=(RiskRequirement('guarantee_status', ('有效',)),),
                    search=replace(state.search, batch_diagnostics=()))
    result = evaluate_gate(state, {}, release_id='release')
    assert not result.passed
    assert [(item.code, item.entry.record_id, item.entry.evidence_source) for item in result.blockers] == [
        ('critical_evidence', 'release', 'guarantee_status'),
    ]


@pytest.mark.parametrize('fallback', ['runtime', 'b2'])
def test_gate_blocks_engine_degradation_and_preserves_record_entry(fallback):
    state = release_state()
    if fallback == 'runtime':
        state = replace(state, engine_status=EngineStatus('runtime_error', '引擎运行失败'))
    else:
        mode = Finding('applicability', 'b2_mode', 'available', 'fallback', 'fallback')
        reason = Finding('applicability', 'b2_fallback_reason', 'available', '样本不足')
        state = replace(state, search=replace(state.search, batch_diagnostics=(mode, reason)))
    result = evaluate_gate(state, {}, release_id='release')
    assert not result.passed
    block, = result.blockers
    assert (block.code, block.entry.record_id, block.entry.action) == (
        'engine_degraded', 'release', 'supplement_evidence')


def test_gate_passes_event_with_unreviewed_member_and_one_exact_target():
    state, reviews = event_state()
    result = evaluate_gate(state, reviews)
    assert result.passed and result.requires_confirmation
    assert not result.blockers
    assert result.targets == ('member',)
    assert result.uncovered_members == ('member',)
    assert result.supporting_reviews == state.search.supporting_reviews
    assert result.recommendations == state.search.recommendations


def test_gate_passes_queue_external_release_without_event_representative():
    result = evaluate_gate(release_state(), {}, release_id='release')
    assert result.passed and result.requires_confirmation
    assert result.targets == ('release',)
    assert result.conclusion == NORMAL


def test_workload_merges_purposes_and_counts_repeat_checks_and_budget_overflow():
    state, reviews = event_state()
    state = replace(state, budget=2, work=(
        ReviewWork('rep', 'review:1', frozenset({'search_check'})),
        ReviewWork('member', 'round-1', frozenset({'search_check', 'gate_check'})),
        ReviewWork('member', 'round-1', frozenset({'evidence_supplement'})),
        ReviewWork('member', 'round-2', frozenset({'recovery_recheck'})),
    ), activities=(WorkflowActivity('explain', 12.5), WorkflowActivity('compute')))
    result = evaluate_gate(state, reviews)
    work = result.workload
    assert not result.passed
    assert (work.unique_records, work.repeat_checks, work.within_budget, work.overflow) == (2, 1, 2, 1)
    assert work.still_blocked == 1
    assert work.rounds[0].purposes == frozenset({'search_check', 'gate_check', 'evidence_supplement'})
    assert work.rounds[-1].purposes == frozenset({'search_check', 'direct_review', 'representative_review'})
    assert work.activities == state.activities


@pytest.fixture
def db():
    conn = sqlite3.connect(':memory:')
    upgrade_schema(conn)
    conn.execute('CREATE TABLE operations (id TEXT PRIMARY KEY, binding TEXT)')
    conn.commit()
    try:
        yield conn
    finally:
        conn.close()


def persist_review(db, review):
    db.execute('INSERT INTO reviews VALUES (?, ?, ?, ?, ?, ?)',
               tuple(review[key] for key in ('id', 'record_id', 'model_version', 'status', 'note', 'created_at')))
    if 'evidence_snapshot' in review:
        payload = _snapshot_json(review['evidence_snapshot'], review['record_id'], review['model_version'])
        db.execute('INSERT INTO review_evidence VALUES (?, ?, ?, ?)',
                   (review['id'], 1, payload, review['evidence_sha256']))


def find_operation(db, operation_id):
    assert db.in_transaction
    row = db.execute('SELECT binding FROM operations WHERE id=?', (operation_id,)).fetchone()
    return row[0] if row else None


def write_operation(db, decision, confirmation, binding):
    assert db.in_transaction
    db.execute('INSERT INTO operations VALUES (?, ?)', (confirmation.operation_id, binding))
    for index, rid in enumerate(decision.targets, 10):
        persist_review(db, direct_review(rid, decision.conclusion, index))


def confirm(preview):
    return GateConfirmation('operation-1', preview.conclusion, preview.targets, '人工确认依据', True)


def test_passed_gate_still_refuses_an_unconfirmed_write(db):
    preview = evaluate_gate(release_state(), {}, release_id='release')
    for confirmation in (None, replace(confirm(preview), confirmed=False),
                         replace(confirm(preview), reason=' '), replace(confirm(preview), targets=('other',))):
        with pytest.raises(GateBlocked) as caught:
            submit_gate(db, preview, confirmation, load_current=lambda _: pytest.fail('premature recheck'),
                        find_operation=find_operation, write_operation=write_operation)
        assert caught.value.blockers[0].code == 'confirmation_required'
    assert not db.in_transaction
    assert db.execute('SELECT COUNT(*) FROM reviews').fetchone() == (0,)


def test_confirmed_write_holds_write_transaction_and_is_idempotent(db):
    state, reviews = event_state()
    persist_review(db, reviews['rep'])
    db.commit()
    preview = evaluate_gate(state, reviews)
    trace = []
    db.set_trace_callback(trace.append)

    def load_current(conn):
        assert conn.in_transaction
        assert trace[0] == 'BEGIN IMMEDIATE'
        return state

    operation = confirm(preview)
    writers = dict(load_current=load_current, find_operation=find_operation, write_operation=write_operation)
    assert not submit_gate(db, preview, operation, **writers).replayed
    assert submit_gate(db, preview, operation, **writers).replayed
    with pytest.raises(GateBlocked, match='操作标识已用于另一份确认'):
        submit_gate(db, preview, replace(operation, reason='不同的确认依据'), **writers)
    assert db.execute('SELECT COUNT(*) FROM reviews').fetchone() == (2,)
    assert db.execute('SELECT COUNT(*) FROM review_evidence').fetchone() == (2,)
    assert db.execute('SELECT COUNT(*) FROM operations').fetchone() == (1,)


def test_submit_reads_persisted_analysis_identity_under_lock_even_with_unchanged_search(db):
    state = release_state()
    preview = evaluate_gate(state, {}, release_id='release')
    db.execute('CREATE TABLE analysis (model_sha256 TEXT)')
    db.execute('INSERT INTO analysis VALUES (?)', ('f' * 64,))
    db.commit()

    def load_current(conn):
        assert conn.in_transaction
        digest, = conn.execute('SELECT model_sha256 FROM analysis').fetchone()
        return replace(state, analysis_identity=replace(CONTEXT, model_sha256=digest))

    with pytest.raises(GateBlocked) as caught:
        submit_gate(db, preview, confirm(preview), load_current=load_current, find_operation=find_operation,
                    write_operation=lambda *_: pytest.fail('stale write'))
    block, = caught.value.blockers
    assert (block.code, block.entry.record_id) == ('version_identity', 'release')
    assert not db.in_transaction


@pytest.mark.parametrize('change,code,source', [
    ('version', 'version_identity', 'analysis_identity'),
    ('state', 'member_state', 'membership'),
    ('membership', 'member_state', 'membership'),
    ('analysis_revision', 'revision', 'analysis_revision'),
    ('evidence_revision', 'revision', 'analysis_revision'),
    ('review_revision', 'revision', 'review_revision'),
    ('recheck', 'required_check', 'search_candidate'),
])
def test_submit_blocks_when_live_revalidation_differs(db, change, code, source):
    state, reviews = event_state()
    persist_review(db, reviews['rep'])
    db.commit()
    preview = evaluate_gate(state, reviews)
    if change == 'version':
        context = replace(CONTEXT, model_sha256='f' * 64)
        state = replace(state, search=replace(state.search, context=context), analysis_identity=context)
    elif change == 'state':
        state = replace(state, members=dict(state.members, member=MemberState('待补证', False)))
    elif change == 'membership':
        state = replace(state, members={'rep': state.members['rep']})
    elif change == 'analysis_revision':
        state = replace(state, analysis_revision=3)
    elif change == 'evidence_revision':
        state = replace(state, evidence_revision=8)
    elif change == 'review_revision':
        persist_review(db, direct_review('rep', review_id=2))
        db.commit()
    else:
        state = replace(state, search=replace(state.search, ordered_candidates=(candidate(priority=1),)))

    with pytest.raises(GateBlocked) as caught:
        submit_gate(db, preview, confirm(preview), load_current=lambda _: state,
                    find_operation=find_operation, write_operation=lambda *_: pytest.fail('stale write'))
    assert (code, source) in {(item.code, item.entry.evidence_source) for item in caught.value.blockers}
    assert db.execute('SELECT COUNT(*) FROM operations').fetchone() == (0,)
    assert not db.in_transaction


def test_submit_rolls_back_all_rows_when_persistence_fails(db):
    state = release_state()
    preview = evaluate_gate(state, {}, release_id='release')

    def write_then_fail(*args):
        write_operation(*args)
        raise sqlite3.OperationalError('write failure')

    with pytest.raises(GateBlocked) as caught:
        submit_gate(db, preview, confirm(preview), load_current=lambda _: state,
                    find_operation=find_operation, write_operation=write_then_fail)
    assert caught.value.blockers[0].entry.record_id == 'release'
    for table in ('operations', 'reviews', 'review_evidence'):
        assert db.execute(f'SELECT COUNT(*) FROM {table}').fetchone() == (0,)
    assert not db.in_transaction
