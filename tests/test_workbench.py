from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app_types import EngineStatus
from liushao import dynamic, recovery, workbench
from liushao.gate import (
    GateBlock, GateBlocked, GateConfirmation, GateDecision, GateEntry, MemberRevision,
    MemberState, ReviewWork, RiskRequirement, Workload,
)
from liushao.search import (
    ANOMALY, NORMAL, AnalysisContext, Candidate, CheckBasis, Finding, HumanReview,
    SearchResult, SearchScope,
)


@pytest.fixture
def event():
    context = AnalysisContext('input', 'model', 'artifact', 'descriptor', ('engine',), .1)
    representatives = frozenset({'rep'})
    scope = SearchScope(9, NORMAL, representatives, frozenset({'member', 'supp'}),
                        frozenset({'outside'}))
    gap = Finding('authorization', 'reference', 'unknown', '参考值待验证')
    difference = Finding('event_heterogeneity', 'representative_comparison', 'signal',
                         '尾部形态不同', ('ecod_tails',))
    uncertainty = Finding('prediction_uncertainty', 'cp_set', 'signal', '共形双标签')
    conflict = Finding('model_conflict', 'RF/LR', 'signal', '两模型方向相反')
    inside = Candidate(
        'supp', ('event',), '模糊', 4, 8,
        CheckBasis(context, ('event',), (9, NORMAL, representatives, scope.authorization_ids),
                   (gap, difference, uncertainty, conflict)),
        ('缺少授权前提', '具有不同形态'), priority=1,
    )
    outside = Candidate('outside', ('release',), '', 5, 2,
                        CheckBasis(context, ('release',), None, (conflict,)),
                        ('队列外放行核查',), priority=1)
    pending = replace(inside, record_id='pending', physical_row=7, reasons=('续办旧核查',), priority=None)
    support = HumanReview('rep', 10, NORMAL, '代表源证据', context, (), ('event',), False)
    counter = HumanReview('counter', 11, ANOMALY, '人工核对发现反向证据', context, ('event',), (), False)
    risk = Finding('applicability', 'guarantee_status', 'available', '逐条风险保证：有效', '有效')
    search = SearchResult(context, scope, (outside, inside), 1, (counter,), (support,),
                          (pending,), (), True, (risk,))
    ids = ('rep', 'member', 'supp', 'pending', 'counter')
    entry = GateEntry('supp', 'reference', 'supplement_evidence', physical_row=4)
    block = GateBlock('critical_evidence', '授权参考值缺失', entry)
    members = tuple(MemberRevision(rid, MemberState('待复核', True), None, None, None)
                    for rid in ids)
    workload = Workload((ReviewWork('rep', 'review:10', frozenset({'direct_review'})),),
                        1, 0, 1, 0, 1, 2, ())
    gate = GateDecision(search, None, NORMAL, ('supp',), (('member', '已有有效授权'),),
                        members, 1, 2, EngineStatus('ok', ''), (RiskRequirement('guarantee_status', ('有效',)),),
                        3, (block,), (support,), (counter,), (inside,), (), (), ('supp', 'pending'), workload)
    revision = dynamic.Revision(dynamic.Scope('input-event-9', 'model', frozenset(ids)), 'ck1', 'fp1')
    preview = dynamic.Preview(search, gate, revision)
    state = recovery.RecoveryState
    rows = (
        dynamic.CurrentRecord('rep', recovery.DecisionView(state.DIRECT, NORMAL, 10)),
        dynamic.CurrentRecord('counter', recovery.DecisionView(state.DIRECT, ANOMALY, 11)),
        dynamic.CurrentRecord('member', recovery.DecisionView(state.INHERITED, NORMAL, inheritance_id='i-1'),
                              inheritance_source='original-batch'),
        dynamic.CurrentRecord('supp', recovery.DecisionView(state.PENDING, action='supplement_evidence')),
        dynamic.CurrentRecord('pending', recovery.DecisionView(state.PENDING, action='review_record')),
    )
    details = workbench.EventDetails(9, '疑似事件 · 模糊', ('同一四态', '共同尾部字段'),
                                     {'rep': '固定种子分层内抽样'})
    return SimpleNamespace(
        search=search, gate=gate, current=dynamic.CurrentScope(rows, preview), details=details,
        batch=workbench.BatchPreview(preview), gap=gap, difference=difference, block=block,
        export=dynamic.Export(revision, '2026-11-25T00:00:00Z', rows, ('{"id":"original-batch"}',)),
        withdrawal=recovery.WithdrawalPreview('original-batch', (
            recovery.WithdrawalMember('member', 'i-1', 'original-batch', 'original-batch',
                                      (None, None, None), 'affected'),
            recovery.WithdrawalMember('counter', 'i-2', 'original-batch', 'original-batch',
                                      (11, ANOMALY, 'hash'), 'direct_review'),
        )),
    )


def assemble(event, **kwargs):
    return workbench.build_workbench(event.search, event.gate, event.current, event.details, **kwargs)


def test_assembles_overview_from_gate_roster_and_current_decisions(event):
    overview = assemble(event).overview
    assert overview.event == event.details
    assert overview.identity == event.search.context
    assert overview.member_total == 5
    assert overview.status_counts == {'直接确认': 2, '继承有效': 1, '待补证': 1, '待复核': 1}
    assert overview.uncovered_members == ('supp', 'pending')
    assert overview.representatives == ('rep',)
    assert overview.risk_diagnostics == event.search.batch_diagnostics
    assert overview.authorization_batches == ('original-batch',)
    by_id = {member.record_id: member for member in overview.members}
    assert by_id['rep'].representative_reason == event.details.representative_reasons['rep']
    assert by_id['member'].current.decision.conclusion == NORMAL


def test_preserves_search_order_paths_and_candidate_basis(event):
    evidence = assemble(event).evidence
    outside, inside = evidence.candidates
    assert (outside.position, inside.position) == (1, 2)
    assert outside.candidate == event.search.ordered_candidates[0]
    assert outside.candidate.paths == ('release',)
    assert outside.candidate.native_queue == ''
    assert outside.candidate.basis.event_question is None
    assert inside.candidate.reasons == ('缺少授权前提', '具有不同形态')
    assert inside.differences == (event.difference,)
    assert inside.check_focus == ('授权参考值缺失', '尾部形态不同', '共形双标签', '两模型方向相反')
    assert inside.unavailable == (event.gap,)
    assert evidence.pending[0].position is None
    assert evidence.representative_support == event.search.supporting_reviews
    assert evidence.counterexamples == event.search.confirmed_counterexamples


def test_gate_workload_and_revision_bound_history(event):
    view = assemble(event, batch=event.batch, export=event.export).confirmation
    assert view.gate.blockers == (event.block,)
    assert view.gate.targets == ('supp',)
    assert view.gate.exclusions == (('member', '已有有效授权'),)
    assert (view.gate.budget, len(view.gate.workload.rounds), view.gate.workload.additional_checks) == (3, 1, 2)
    assert view.batch == event.batch
    assert not view.requires_reread
    assert view.history == event.export.history
    assert view.export.conclusions == event.export.conclusions


def test_expired_preview_requires_reread_of_affected_basis(event):
    gate = replace(event.gate, evidence_revision=3)
    revision = replace(event.current.preview.revision, checkpoint='ck2')
    current = replace(event.current, preview=dynamic.Preview(event.search, gate, revision))
    view = workbench.build_workbench(event.search, gate, current, event.details,
                                     batch=event.batch, export=event.export).confirmation
    assert view.requires_reread
    assert [block.entry.evidence_source for block in view.affected_basis] == ['evidence_revision']
    assert view.export is None
    assert view.history == ()


def test_submit_time_gate_blockers_are_the_reread_basis(event):
    view = assemble(event, batch=event.batch, batch_error=GateBlocked((event.block,))).confirmation
    assert view.requires_reread
    assert view.affected_basis == (event.block,)


def test_unverified_dynamic_projection_does_not_restore_inheritance(event):
    rows = tuple(
        dynamic.CurrentRecord(row.record_id, recovery.DecisionView(recovery.RecoveryState.PENDING))
        if row.record_id == 'member' else row for row in event.current.records
    )
    current = dynamic.CurrentScope(rows, None)
    view = workbench.build_workbench(event.search, event.gate, current, event.details, batch=event.batch)
    assert view.confirmation.requires_reread
    assert '继承有效' not in view.overview.status_counts
    assert view.overview.authorization_batches == ()


def test_withdrawal_is_bound_to_original_batch_and_uses_full_recovery_impact(event):
    rows = tuple(
        replace(row, decision=recovery.DecisionView(recovery.RecoveryState.INVALIDATED, inheritance_id='i-1'),
                inheritance_source=None, withdrawal_batch='original-batch')
        if row.record_id == 'member' else row for row in event.current.records
    )
    event.current = replace(event.current, records=rows)
    view = assemble(event, withdrawal=event.withdrawal)
    assert view.overview.status_counts['继承失效待重新确认'] == 1
    assert view.overview.authorization_batches == ('original-batch',)
    assert view.confirmation.withdrawal is event.withdrawal
    assert assemble(event, withdrawal=replace(event.withdrawal, batch_id='other')).confirmation.withdrawal is None


def test_original_batch_can_be_selected_from_verified_audit_history(event):
    export = replace(event.export, history=('{"id":"earlier-batch","kind":"authorize"}',
                                           '{"id":"withdraw-op","kind":"withdraw"}'))
    earlier = replace(event.withdrawal, batch_id='earlier-batch')
    view = assemble(event, export=export, withdrawal=earlier).confirmation
    assert view.withdrawal_batches == ('earlier-batch', 'original-batch')
    assert view.withdrawal is earlier


def test_preparing_and_submitting_batch_uses_upstream_guards(event, monkeypatch):
    gate = replace(event.gate, blockers=())
    preview = replace(event.current.preview, gate=gate)
    inheritance = recovery.InheritancePreview(gate, (('supp', None),))
    verified = Mock(return_value=preview)
    read_inheritance = Mock(return_value=inheritance)
    load = Mock()
    state = Mock()
    confirmation = GateConfirmation('op', NORMAL, gate.targets, '依据', True)
    db = object()

    def submit(connection, passed, confirmed, *, load_current, actor):
        assert connection is db and passed is inheritance and confirmed is confirmation
        assert load_current(connection) is state.return_value
        return 'submitted'

    monkeypatch.setattr(dynamic, 'revalidate', verified)
    monkeypatch.setattr(recovery, 'preview_inheritance', read_inheritance)
    monkeypatch.setattr(recovery, 'submit_inheritance', submit)
    batch = workbench.prepare_batch(db, preview, load=load)
    assert workbench.submit_batch(db, batch, confirmation, load=load, load_gate_state=state) == 'submitted'
    read_inheritance.assert_called_once_with(db, preview.search, gate)
    state.assert_called_once_with(db)
    assert verified.call_count == 2
    verified.assert_called_with(db, preview.revision, load)


def _render_fixture():
    import streamlit as st

    from liushao.workbench import WorkbenchActions, render_workbench

    def record(action, *args):
        st.session_state['last_action'] = (action, args)

    actions = WorkbenchActions(
        lambda value: record('event', value), lambda value: record('record', value),
        lambda: record('outside'), lambda value: record('resolve', value),
        lambda: record('reread'), lambda value: record('prepare', value),
        lambda *args: record('submit', *args), lambda value: record('withdraw_preview', value),
        lambda *args: record('withdraw', *args),
    )
    render_workbench(st.session_state['view'], actions, events=(9, 10), dirty=st.session_state['dirty'])


def _app(view, *, dirty=False):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_render_fixture)
    app.session_state['view'] = view
    app.session_state['dirty'] = dirty
    app.run()
    assert not app.exception
    return app


def _widget(widgets, label):
    return next(widget for widget in widgets if widget.label == label)


def test_streamlit_regions_and_draft_aware_navigation(event):
    app = _app(assemble(event, batch=event.batch, export=event.export), dirty=True)
    assert [header.value for header in app.header] == ['事件概览', '证据与成员', '确认与恢复']
    assert _widget(app.button, '生成批次预览').disabled
    assert any('预算已用 1/3 轮' in item.value for item in app.markdown)
    assert any('授权参考值缺失' in item.value for item in app.text)
    _widget(app.button, '补证').click().run()
    assert app.session_state['last_action'] == ('resolve', (event.block.entry,))
    _widget(app.button, '查看逐条证据、来源与历史').click().run()
    assert app.session_state['last_action'] == ('record', ('outside',))
    _widget(app.button, '搜索同批队列外记录').click().run()
    assert app.session_state['last_action'] == ('outside', ())
    _widget(app.selectbox, '疑似事件').select(10).run()
    _widget(app.button, '切换事件').click().run()
    assert app.session_state['last_action'] == ('event', (10,))


def test_streamlit_expired_batch_blocks_submit_and_offers_reread(event):
    app = _app(assemble(event, batch=event.batch, batch_error=GateBlocked((event.block,))))
    assert any('请重读依据' in item.value for item in app.warning)
    assert any('reference' in item.value and '授权参考值缺失' in item.value for item in app.text)
    assert all(button.label not in ('生成批次预览', '确认当前批次') for button in app.button)
    _widget(app.button, '重读依据').click().run()
    assert app.session_state['last_action'] == ('reread', ())


def test_streamlit_batch_confirmation_is_bound_to_preview_targets(event):
    event.gate = replace(event.gate, blockers=())
    preview = replace(event.current.preview, gate=event.gate)
    event.current = replace(event.current, preview=preview)
    batch = workbench.BatchPreview(preview, recovery.InheritancePreview(event.gate, (('supp', None),)))
    app = _app(assemble(event, batch=batch))
    _widget(app.button, '生成批次预览').click().run()
    assert app.session_state['last_action'] == ('prepare', (preview,))
    assert _widget(app.button, '确认当前批次').disabled
    _widget(app.text_input, '确认依据').input('已核对逐条证据').run()
    _widget(app.checkbox, '我已核对当前结论与本次目标').check().run()
    _widget(app.button, '确认当前批次').click().run()
    action, (submitted, confirmation) = app.session_state['last_action']
    assert action == 'submit' and submitted is batch
    assert confirmation.confirmed and confirmation.conclusion == NORMAL
    assert confirmation.targets == ('supp',) and confirmation.reason == '已核对逐条证据'


def test_streamlit_withdrawal_requires_explicit_review_of_original_impact(event):
    app = _app(assemble(event, withdrawal=event.withdrawal))
    _widget(app.button, '按原批次预览撤回').click().run()
    assert app.session_state['last_action'] == ('withdraw_preview', ('original-batch',))
    assert any('counter · 保留直接复核' in item.value for item in app.text)
    assert _widget(app.button, '提交撤回').disabled
    _widget(app.text_input, '撤回原因').input('人工决定撤回').run()
    _widget(app.checkbox, '我已核对原批次的撤回影响').check().run()
    _widget(app.button, '提交撤回').click().run()
    action, args = app.session_state['last_action']
    assert action == 'withdraw' and args[0] == event.withdrawal and args[1]
    assert args[2:] == ('人工决定撤回', True)


def _event_section_fixture():
    import streamlit as st
    from types import SimpleNamespace
    from liushao.workbench import render_evidence_section, render_batch_section

    def open_record(rid):
        st.session_state['opened'] = rid
        st.session_state['details'] = True

    def back():
        st.session_state['details'] = False

    if st.session_state.get('details'):
        st.button('返回事件', on_click=back)
        return
    view = st.session_state['view']
    actions = SimpleNamespace(request_record=open_record, resolve_block=lambda entry: open_record(entry.record_id),
                              prepare_batch=lambda preview: None, reread=lambda: None,
                              preview_withdrawal=lambda batch: None)
    lines = {'supp': 4, 'outside': 5, 'rep': 2, 'counter': 6, 'pending': 7}
    records = {'supp': {'record_id': 'supp', 'source_file': 'capture<one>.csv', 'source_row': 312,
                        'rf_score': .42, 'lr_score': .7}}
    if st.session_state['section'] == 'batch':
        render_batch_section(view.confirmation, actions, 'event', False, False, line_numbers=lines)
    else:
        render_evidence_section(view, actions, 'event', False, current_id=st.session_state['member'],
                                select_member=lambda rid: st.session_state.update(member=rid),
                                line_numbers=lines, record_lookup=records.get,
                                thresholds={'rf_threshold': .5, 'lr_threshold': .6})


def _section_app(event, section):
    from streamlit.testing.v1 import AppTest

    app = AppTest.from_function(_event_section_fixture)
    app.session_state['view'] = assemble(event)
    app.session_state['section'] = section
    app.session_state['member'] = 'supp'
    app.run()
    assert not app.exception
    return app


def test_evidence_keeps_the_selected_candidate_and_scope_across_record_navigation(event):
    app = _section_app(event, 'evidence')
    assert app.button(key='event_open_candidate').label == '打开CSV 第 4 行核查'
    assert any('capture&lt;one&gt;.csv · 来源第 312 行' in item.proto.body for item in app.get('html'))
    _widget(app.selectbox, '核查范围').select('全部').run()
    _widget(app.selectbox, '选择候选记录').select('outside').run()
    assert app.session_state['member'] == 'supp'
    assert 'opened' not in app.session_state
    app.button(key='event_open_candidate').click().run()
    assert app.session_state['opened'] == 'outside'
    _widget(app.button, '返回事件').click().run()
    assert _widget(app.selectbox, '核查范围').value == '全部'
    assert _widget(app.selectbox, '选择候选记录').value == 'outside'
    _widget(app.selectbox, '选择候选记录').select('pending').run()
    assert app.session_state['member'] == 'pending'
    assert app.button(key='event_open_candidate').label == '打开CSV 第 7 行核查'
    assert not app.exception


def test_gate_locates_a_representative_without_a_physical_row_and_ignores_no_targets_as_a_task(event):
    missing = GateBlock('required_check', '代表缺少有效的直接确认', GateEntry('rep', 'representative', 'review_record'))
    empty = GateBlock('no_targets', '当前没有可授权目标', GateEntry(None, 'authorization_scope', 'review_record'))
    event.gate = replace(event.gate, blockers=(missing, empty), targets=())
    event.current = replace(event.current, preview=replace(event.current.preview, gate=event.gate))
    app = _section_app(event, 'batch')
    assert app.button(key='event_fix_block').label == '逐条核查 · CSV 第 2 行'
    assert any('1 项操作' in item.value for item in app.warning)
    assert not any(box.label == '待处理动作' for box in app.selectbox)
    app.button(key='event_fix_block').click().run()
    assert app.session_state['opened'] == 'rep'
