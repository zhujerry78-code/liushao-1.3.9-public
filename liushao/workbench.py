"""Streamlit workbench for one event in the active input.

The host supplies the event summary and the existing draft-aware navigation.
Search, Gate, Recovery and Dynamic own the decisions and their persistence.
"""

import json
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from html import escape
from uuid import uuid4

import streamlit as st

from app_ui_state import literal_markdown
from . import dynamic, recovery
from .gate import GateBlock, GateBlocked, GateConfirmation, GateDecision, GateEntry
from .reference import number
from .search import AnalysisContext, Candidate, Finding, HumanReview, SearchResult


@dataclass(frozen=True)
class EventDetails:
    event_id: int
    summary: str
    merge_basis: tuple[str, ...]
    representative_reasons: Mapping[str, str]


@dataclass(frozen=True)
class MemberView:
    record_id: str
    current: dynamic.CurrentRecord | None
    status: str
    representative_reason: str | None


@dataclass(frozen=True)
class EventOverview:
    event: EventDetails
    identity: AnalysisContext
    members: tuple[MemberView, ...]
    member_total: int
    status_counts: Mapping[str, int]
    representatives: tuple[str, ...]
    uncovered_members: tuple[str, ...]
    risk_diagnostics: tuple[Finding, ...]
    authorization_batches: tuple[str, ...]


@dataclass(frozen=True)
class CandidateView:
    candidate: Candidate
    position: int | None
    differences: tuple[Finding, ...]
    check_focus: tuple[str, ...]
    unavailable: tuple[Finding, ...]


@dataclass(frozen=True)
class EvidenceView:
    representative_support: tuple[HumanReview, ...]
    counterexamples: tuple[HumanReview, ...]
    candidates: tuple[CandidateView, ...]
    pending: tuple[CandidateView, ...]
    retained: tuple[CandidateView, ...]


@dataclass(frozen=True)
class BatchPreview:
    dynamic: dynamic.Preview
    inheritance: recovery.InheritancePreview | None = None


@dataclass(frozen=True)
class ConfirmationView:
    gate: GateDecision
    current_preview: dynamic.Preview | None
    batch: BatchPreview | None
    requires_reread: bool
    affected_basis: tuple[GateBlock, ...]
    history: tuple[str, ...]
    export: dynamic.Export | None
    withdrawal_batches: tuple[str, ...]
    withdrawal: recovery.WithdrawalPreview | None
    withdrawal_error: str | None


@dataclass(frozen=True)
class WorkbenchView:
    overview: EventOverview
    evidence: EvidenceView
    confirmation: ConfirmationView


def _member_status(row):
    if row is None:
        return '待核验'
    decision = row.decision
    if decision.state == recovery.RecoveryState.PENDING:
        return '待补证' if decision.action == 'supplement_evidence' else '待复核'
    return str(decision.state)


def _candidate_view(candidate, position, blockers):
    evidence = candidate.basis.evidence
    signals = (finding.reason for finding in evidence if finding.state == 'signal')
    focus = tuple(dict.fromkeys((*blockers, *signals))) or candidate.reasons
    return CandidateView(
        candidate, position,
        tuple(finding for finding in evidence if finding.source == 'representative_comparison'),
        focus, tuple(finding for finding in evidence if finding.state not in ('available', 'signal')),
    )


def _basis_to_reread(previous, current):
    """Name the displayed inputs that must be read again after a revision change."""
    if previous is None:
        return ('analysis_revision',)
    comparisons = {
        'analysis_identity': (previous.search.context, current.search.context),
        'analysis_revision': (previous.analysis_revision, current.analysis_revision),
        'evidence_revision': (previous.evidence_revision, current.evidence_revision),
        'membership_and_reviews': (previous.members, current.members),
        'search_evidence': (previous.search, current.search),
        'authorization_scope': ((previous.conclusion, previous.targets, previous.exclusions),
                                (current.conclusion, current.targets, current.exclusions)),
        'review_budget': ((previous.budget, previous.workload), (current.budget, current.workload)),
    }
    return tuple(source for source, (before, after) in comparisons.items() if before != after) or (
        '当前输入、逐条复核与授权依赖',)


def build_workbench(search: SearchResult, gate: GateDecision, current: dynamic.CurrentScope,
                    event: EventDetails, *, batch: BatchPreview | None = None,
                    export: dynamic.Export | None = None,
                    withdrawal: recovery.WithdrawalPreview | None = None,
                    batch_error: GateBlocked | recovery.RecoveryConflict | None = None,
                    withdrawal_error: str | None = None) -> WorkbenchView:
    """Assemble the three zones from detached results for the selected event.

    CurrentScope contains Recovery's current decisions as verified by Dynamic.
    Search order covers both event and release paths. Only Gate's roster counts
    toward the selected event. A held batch can belong to an older revision.
    """
    if gate.search != search or gate.release_id is not None or search.scope.event_id != event.event_id:
        raise ValueError('工作台需要同一选定事件的 Search 与 Gate 结果')
    preview = current.preview
    member_ids = frozenset(member.record_id for member in gate.members)
    if preview is not None and (preview.search != search or preview.gate != gate
                                or preview.revision.scope.record_ids != member_ids
                                or preview.revision.scope.model_version != search.context.model_version):
        raise ValueError('Dynamic 当前预览与所选事件不一致')
    if batch is not None and (batch.dynamic.search.scope.event_id != event.event_id
                              or batch.dynamic.search.context.input_sha256 != search.context.input_sha256
                              or batch.dynamic.search != batch.dynamic.gate.search
                              or batch.dynamic.gate.release_id is not None
                              or batch.inheritance is not None and batch.inheritance.decision != batch.dynamic.gate):
        raise ValueError('批次预览与当前输入、事件或 Recovery 依据不一致')

    current_by_id = {row.record_id: row for row in current.records}
    reps = search.scope.representative_ids
    members = tuple(
        MemberView(rid, current_by_id.get(rid), _member_status(current_by_id.get(rid)),
                   event.representative_reasons.get(rid, '选择依据待核验') if rid in reps else None)
        for rid in sorted(member_ids)
    )
    batch_ids = {
        batch_id for member in members if member.current is not None
        for batch_id in (member.current.inheritance_source, member.current.withdrawal_batch) if batch_id
    }
    verified_export = export if preview is not None and export is not None and (
        export.revision == preview.revision) else None
    history = verified_export.history if verified_export else ()
    history_batches = {entry['id'] for row in history if (entry := json.loads(row)).get('kind') == 'authorize'}
    withdrawal_batches = tuple(sorted(batch_ids | history_batches))
    overview = EventOverview(
        event, search.context, members, len(members), dict(Counter(m.status for m in members)),
        tuple(sorted(reps)), gate.uncovered_members, search.batch_diagnostics, tuple(sorted(batch_ids)),
    )

    blockers = {}
    for block in gate.blockers:
        blockers.setdefault(block.entry.record_id, []).append(block.reason)
    candidates = tuple(_candidate_view(item, position, blockers.get(item.record_id, ()))
                       for position, item in enumerate(search.ordered_candidates, 1))
    evidence = EvidenceView(
        tuple(review for review in search.supporting_reviews
              if review.record_id in reps and 'event' in review.supporting_paths),
        search.confirmed_counterexamples, candidates,
        tuple(_candidate_view(item, None, blockers.get(item.record_id, ()))
              for item in search.pending_verification),
        tuple(_candidate_view(item, None, blockers.get(item.record_id, ()))
              for item in search.retained_checks),
    )

    reread = preview is None or batch_error is not None or (
        batch is not None and preview.revision != batch.dynamic.revision)
    affected = batch_error.blockers if isinstance(batch_error, GateBlocked) else ()
    if reread and not affected:
        reason = str(batch_error) if batch_error else '研判修订待核验，需重读所列依据'
        sources = _basis_to_reread(batch.dynamic.gate if batch else None, gate)
        affected = tuple(GateBlock('revision', reason, GateEntry(None, source, 'refresh_preview'))
                         for source in sources)
    confirmation = ConfirmationView(
        gate, preview, batch, reread, affected, history, verified_export, withdrawal_batches,
        withdrawal if withdrawal and withdrawal.batch_id in withdrawal_batches else None, withdrawal_error,
    )
    return WorkbenchView(overview, evidence, confirmation)


def prepare_batch(db, preview: dynamic.Preview, *, load: dynamic.Loader) -> BatchPreview:
    """Generate a revision-bound preview, including the current Gate blockers."""
    verified = dynamic.revalidate(db, preview.revision, load)
    inheritance = (recovery.preview_inheritance(db, verified.search, verified.gate)
                   if verified.gate.passed else None)
    return BatchPreview(verified, inheritance)


def submit_batch(db, batch: BatchPreview, confirmation: GateConfirmation, *,
                 load: dynamic.Loader, load_gate_state, actor=None):
    """Authorize with Dynamic, Gate and Recovery checks in the write transaction."""
    if batch.inheritance is None or batch.inheritance.decision != batch.dynamic.gate:
        raise ValueError('请先生成可确认的批次预览')

    def load_current(connection):
        dynamic.revalidate(connection, batch.dynamic.revision, load)
        return load_gate_state(connection)

    return recovery.submit_inheritance(db, batch.inheritance, confirmation,
                                       load_current=load_current, actor=actor)


@dataclass(frozen=True)
class WorkbenchActions:
    """Host callbacks update state before rerendering.

    Switch callbacks enter the host request/save/discard/cancel flow. Record
    navigation opens the existing per-record evidence, source and history.
    The host passes back prepared previews, or caught submit conflicts, on
    the next render. Withdrawal callbacks use Recovery's original-batch API.
    """

    request_event: Callable[[int], None]
    request_record: Callable[[str], None]
    search_same_input: Callable[[], None]
    resolve_block: Callable[[GateEntry], None]
    reread: Callable[[], None]
    prepare_batch: Callable[[dynamic.Preview], None]
    submit_batch: Callable[[BatchPreview, GateConfirmation], None]
    preview_withdrawal: Callable[[str], None]
    submit_withdrawal: Callable[[recovery.WithdrawalPreview, str, str, bool], None]


def _render_overview(view, actions, events, key, locked):
    st.header('事件概览')
    event_id = view.event.event_id
    choices = tuple(dict.fromkeys((event_id, *events)))
    if len(choices) > 1:
        chosen = st.selectbox('疑似事件', choices, format_func=lambda value: f'事件 {value}',
                              key=f'{key}_event')
        st.button('切换事件', key=f'{key}_switch', disabled=locked or chosen == event_id,
                  on_click=actions.request_event, args=(chosen,))
    st.subheader(f'事件 {event_id}')
    st.text(view.event.summary)
    st.text('归并依据 · ' + ('、'.join(view.event.merge_basis) or '待核验'))
    st.write(f'成员 {view.member_total} 条 · 代表 {len(view.representatives)} 条 · '
             f'门控证据覆盖 {view.member_total - len(view.uncovered_members)}/{view.member_total}')
    st.write(' · '.join(f'{status} {count} 条' for status, count in view.status_counts.items()))
    for member in view.members:
        if member.representative_reason is not None:
            st.text(f'代表 {member.record_id} · {member.representative_reason}')
    st.text('当前成员关联的授权批次 · ' + ('、'.join(view.authorization_batches) or '无可核验的批次'))
    for finding in view.risk_diagnostics:
        st.text(finding.reason)
    with st.expander('分析身份与可核验范围'):
        st.json(asdict(view.identity))
        st.text('门控证据尚未覆盖 · ' + ('、'.join(view.uncovered_members) or '无'))
        for member in view.members:
            row = member.current
            if row is not None:
                st.text(f'{member.record_id} · {member.status} · '
                        + (row.decision.conclusion or '未形成当前结论'))
                if row.inheritance_source or row.withdrawal_batch:
                    st.text(f'原授权批次 {row.inheritance_source or row.withdrawal_batch}')


def _scope_text(paths):
    labels = {'event': '事件成员', 'release': '同批队列外'}
    return '、'.join(labels[path] for path in paths)


def _render_reviews(title, reviews, actions, key, locked, *, counterexamples=False):
    st.subheader(title)
    if not reviews:
        st.caption('暂无')
        return
    for index, review in enumerate(reviews):
        paths = review.counterexample_paths if counterexamples else review.supporting_paths
        st.text(f'{review.record_id} · {_scope_text(paths)} · {review.conclusion}')
        st.text(review.reason)
        st.button('查看逐条证据与历史', key=f'{key}_{index}', disabled=locked,
                  on_click=actions.request_record, args=(review.record_id,))


def _render_candidates(candidates, actions, key, locked):
    if not candidates:
        st.caption('暂无')
        return
    selected = st.selectbox(
        '记录', candidates, key=f'{key}_select',
        format_func=lambda item: (f'{item.position}. ' if item.position is not None else '')
        + f'{item.candidate.record_id} · {_scope_text(item.candidate.paths)}',
    )
    item = selected.candidate
    st.text('原生队列 · ' + (item.native_queue if item.native_queue else
                            '队列外' if item.native_queue == '' else '待核验'))
    if item.physical_row is not None:
        st.text(f'输入 CSV 第 {item.physical_row} 行')
    st.text('排位或处理依据 · ' + '；'.join(item.reasons))
    st.text('与代表的差异 · ' + ('；'.join(f.reason for f in selected.differences) or
                              '当前依据未列出差异'))
    st.text('人工核对重点 · ' + '；'.join(selected.check_focus))
    if not item.basis.evidence:
        st.text('依据完整性 · 待核验')
    elif selected.unavailable:
        st.text('依据完整性 · ' + '；'.join(f.reason for f in selected.unavailable))
    else:
        st.text('依据完整性 · 所列依据可核查')
    for finding in item.basis.evidence:
        if finding.category in ('event_heterogeneity', 'prediction_uncertainty', 'model_conflict'):
            st.text(finding.reason)
            if finding.value is not None:
                st.json({'观测值': finding.value})
    st.button('查看逐条证据、来源与历史', key=f'{key}_record', disabled=locked,
              on_click=actions.request_record, args=(item.record_id,))


def _render_evidence(view, members, actions, key, locked):
    st.header('证据与成员')
    support, counter, candidates = st.columns(3)
    with support:
        _render_reviews('代表支持', view.representative_support, actions, f'{key}_support', locked)
    with counter:
        _render_reviews('已确认人工反证', view.counterexamples, actions, f'{key}_counter',
                        locked, counterexamples=True)
    with candidates:
        st.subheader('未查候选')
        _render_candidates(view.candidates, actions, f'{key}_candidate', locked)
    for label, group, suffix in (('待核验候选', view.pending, 'pending'),
                                 ('保留的核查记录', view.retained, 'retained')):
        if group:
            with st.expander(label):
                _render_candidates(group, actions, f'{key}_{suffix}', locked)
    if members:
        member = st.selectbox('事件成员', members, format_func=lambda m: f'{m.record_id} · {m.status}',
                              key=f'{key}_member')
        st.button('打开成员逐条证据、来源与历史', key=f'{key}_open_member', disabled=locked,
                  on_click=actions.request_record, args=(member.record_id,))
    st.button('搜索同批队列外记录', key=f'{key}_outside', disabled=locked,
              on_click=actions.search_same_input)


def _render_gate(gate, line_numbers=None):
    st.text(f'拟确认结论 · {gate.conclusion if gate.passed else "待定"}')
    reviewed = sum(member.state.status == recovery.RecoveryState.DIRECT for member in gate.members)
    st.write(f'事件范围 {len(gate.members)} 条 · 本次目标 {len(gate.targets)} 条 · 已逐条确认 {reviewed} 条'
             f' · 排除 {len(gate.exclusions)} 条')
    if gate.targets or gate.exclusions:
        with st.expander('查看目标和排除依据'):
            for rid in gate.targets:
                st.text(f'本次目标 · {_record_location(rid, line_numbers)}')
            for rid, reason in gate.exclusions:
                st.text(f'排除 · {_record_location(rid, line_numbers)} · {reason}')
    work = gate.workload
    limit = '不限' if gate.budget is None else str(gate.budget)
    st.write(f'预算已用 {len(work.rounds)}/{limit} 轮 · 超额 {work.overflow} 轮 · '
             f'需追加核查 {work.additional_checks} 条')
    if gate.risk_requirements:
        st.text('风险前提 · ' + '；'.join(f'{item.source} 接受 {"、".join(item.accepted_values)}'
                                       for item in gate.risk_requirements))


def _render_batch(view, actions, key, locked, dirty, line_numbers=None):
    if view.requires_reread:
        st.warning('预览已过期或当前依据待核验。请重读依据后重新生成并确认。')
        for block in view.affected_basis:
            st.text(f'{block.entry.record_id or "当前批次"} · {block.entry.evidence_source} · {block.reason}')
        if view.batch:
            gate = view.batch.dynamic.gate
            with st.expander('重读原预览依据'):
                st.text(f'分析修订 {gate.analysis_revision} · 证据修订 {gate.evidence_revision}')
                _render_gate(gate, line_numbers)
                for block in gate.blockers:
                    st.text(f'{block.entry.evidence_source} · {block.reason}')
        st.button('重读依据', key=f'{key}_reread', disabled=locked, on_click=actions.reread)
        return
    st.button('生成批次预览', key=f'{key}_prepare', disabled=locked or dirty,
              on_click=actions.prepare_batch, args=(view.current_preview,))
    if view.batch is None:
        return
    gate = view.batch.dynamic.gate
    with st.expander('核对批次预览', expanded=True):
        _render_gate(gate, line_numbers)
        if view.batch.inheritance is None or not gate.requires_confirmation:
            st.info('请先处理门控原因，再生成可确认的预览。')
            return
        token = sha256(repr((view.batch.dynamic.revision, view.batch.inheritance.revisions)).encode()).hexdigest()
        reason = st.text_input('确认依据', key=f'{key}_reason_{token}', max_chars=4000)
        confirmed = st.checkbox('我已核对当前结论与本次目标', key=f'{key}_confirm_{token}')
        if st.button('确认当前批次', key=f'{key}_submit_{token}',
                     disabled=locked or dirty or not confirmed or not reason.strip()):
            confirmation = GateConfirmation(uuid4().hex, gate.conclusion, gate.targets, reason, confirmed)
            actions.submit_batch(view.batch, confirmation)
            st.rerun()


def _render_withdrawal(view, actions, key, locked, dirty):
    if not view.withdrawal_batches:
        return
    batch_id = st.selectbox('原授权批次', view.withdrawal_batches, key=f'{key}_original')
    st.button('按原批次预览撤回', key=f'{key}_withdraw_preview', disabled=locked or dirty,
              on_click=actions.preview_withdrawal, args=(batch_id,))
    preview = view.withdrawal
    if preview is None or preview.batch_id != batch_id:
        if view.withdrawal_error:
            st.warning(view.withdrawal_error)
        return
    with st.expander('核对撤回影响', expanded=True):
        labels = {'affected': '将撤回继承', 'direct_review': '保留直接复核',
                  'other_authorization': '保留其他授权', 'already_withdrawn': '已撤回',
                  'later_review': '保留后续复核'}
        affected = sum(member.disposition == 'affected' for member in preview.members)
        st.write(f'原批次 {len(preview.members)} 条 · 将撤回 {affected} 条 · 保留 {len(preview.members) - affected} 条')
        for member in preview.members:
            st.text(f'{member.record_id} · {labels[member.disposition]}')
        if view.withdrawal_error:
            st.warning(f'{view.withdrawal_error}。请重读原批次影响并重新预览。')
            return
        token = sha256(repr(preview).encode()).hexdigest()
        reason = st.text_input('撤回原因', key=f'{key}_withdraw_reason_{token}', max_chars=4000)
        confirmed = st.checkbox('我已核对原批次的撤回影响', key=f'{key}_withdraw_confirm_{token}')
        if st.button('提交撤回', key=f'{key}_withdraw_{token}',
                     disabled=locked or dirty or not confirmed or not reason.strip()):
            actions.submit_withdrawal(preview, uuid4().hex, reason, confirmed)
            st.rerun()


def _render_confirmation(view, actions, key, locked, dirty):
    st.header('确认与恢复')
    _render_gate(view.gate)
    for index, block in enumerate(view.gate.blockers):
        entry = block.entry
        st.text(f'{entry.record_id or "当前批次"} · {block.reason}')
        label = {'review_record': '逐条确认', 'supplement_evidence': '补证'}.get(entry.action)
        if label:
            st.button(label, key=f'{key}_block_{index}', disabled=locked,
                      on_click=actions.resolve_block, args=(entry,))
    if not view.requires_reread and any(block.entry.action == 'refresh_preview' for block in view.gate.blockers):
        st.button('重读门控依据', key=f'{key}_gate_reread', disabled=locked, on_click=actions.reread)
    if dirty:
        st.info('请先保存或放弃当前填写，再办理批次确认或撤回。')
    _render_batch(view, actions, key, locked, dirty)
    with st.expander('历史操作与审计记录'):
        if view.export is None:
            st.caption('请读取当前修订的审计投影。')
        elif not view.history:
            st.caption('当前范围暂无已保存的操作。')
        for entry in view.history:
            st.json(json.loads(entry))
    _render_withdrawal(view, actions, key, locked, dirty)


def render_workbench(view: WorkbenchView, actions: WorkbenchActions, *,
                     events: tuple[int, ...] = (), locked: bool = False,
                     dirty: bool = False, key: str = 'workbench') -> None:
    """Render one event. Pass the host's pending-switch lock and draft state."""
    identity = view.overview.identity
    scope_key = (identity.input_sha256, identity.model_sha256, view.overview.event.event_id)
    key = f'{key}_{sha256(repr(scope_key).encode()).hexdigest()}'
    _render_overview(view.overview, actions, events, key, locked)
    _render_evidence(view.evidence, view.overview.members, actions, key, locked)
    _render_confirmation(view.confirmation, actions, key, locked, dirty)


def render_history(view):
    if view.export is None:
        st.caption('当前审计修订无法读取。')
        return
    if not view.history:
        st.caption('该事件尚无逐条复核或批次操作。')
        return
    labels = {'authorize': '批次授权', 'invalidate': '继承失效', 'withdraw': '撤回授权'}
    entries = sorted((json.loads(row) for row in view.history),
                     key=lambda row: (row.get('created_at', ''), str(row.get('id', ''))), reverse=True)
    timeline = [{'保存时间（UTC）': row.get('created_at'), '操作': labels.get(row.get('kind'), '逐条复核'),
                 '结论或依据': row.get('reason') or ' · '.join(filter(None, (row.get('status'), row.get('note')))),
                 '操作编号': str(row.get('id', '')), '原批次或记录': row.get('batch_id') or row.get('record_id') or row.get('id')}
                for row in entries]
    st.dataframe(timeline, hide_index=True, width='stretch')
    with st.expander('技术证据'):
        index = st.selectbox('操作记录', range(len(entries)), format_func=lambda i: (
            f'{timeline[i]["保存时间（UTC）"]} · {timeline[i]["操作"]} · {timeline[i]["操作编号"]}'))
        st.json(entries[index], expanded=False)


def _record_location(record_id, line_numbers, physical=None):
    line = physical or (line_numbers or {}).get(record_id)
    if line is not None:
        return f'CSV 第 {line} 行'
    return f'记录 {record_id}' if record_id else '本事件'


def _render_human_evidence(evidence, actions, key, locked, line_numbers):
    counters = evidence.counterexamples
    supports = evidence.representative_support
    with st.container(key='event_human_evidence'):
        if not counters and not supports:
            st.caption('人工核查尚无可核验的代表支持或反证。')
            return
        if counters:
            st.warning(f'已有 {len(counters)} 条人工反证，请先核对其结论和适用范围。')
        for title, reviews, counter in (('人工反证', counters, True), ('代表支持', supports, False)):
            if not reviews:
                continue
            with st.expander(f'{title} · {len(reviews)} 条', expanded=counter):
                review = reviews[0]
                if len(reviews) > 1:
                    review = st.selectbox('已保存的人工复核', reviews, key=f'{key}_human_{counter}',
                        format_func=lambda item: f'{_record_location(item.record_id, line_numbers)} · {item.conclusion}',
                        disabled=locked)
                paths = review.counterexample_paths if counter else review.supporting_paths
                location = _record_location(review.record_id, line_numbers)
                st.markdown(literal_markdown(f'{location} · {_scope_text(paths)} · {review.conclusion} · 复核记录 #{review.review_id}'))
                st.text(review.reason)
                st.button(f'打开{location}查看记录', key=f'{key}_evidence_{counter}',
                          disabled=locked, on_click=actions.request_record, args=(review.record_id,))


def _observations(candidate, record, thresholds, observation):
    entries = []
    if record is not None:
        source, original_line = record.get('source_file'), number(record.get('source_row'))
        if isinstance(source, str) and source.strip() and original_line is not None:
            entries.append(('来源定位', f'{source} · 来源第 {int(original_line)} 行'))
    if observation:
        comparison = observation.get('直观对照')
        value = (f"{comparison}。原值 {observation['当前值']}" if comparison
                 else f"{observation['当前值']}，{observation['当前状态']}。{observation['训练正常参考']}")
        entries.append((observation['数据项'], value))
    by_source = {item.source: item for item in candidate.basis.evidence if item.category != 'authorization'}
    scores = []
    for model in ('RF', 'LR'):
        direction = by_source.get(model)
        if direction is None or direction.state != 'available':
            continue
        score = number(record.get(model.lower() + '_score')) if record is not None else None
        threshold = number((thresholds or {}).get(model.lower() + '_threshold'))
        if score is not None and threshold is not None:
            scores.append(f'{model} {score:.4f}（分界 {threshold:.4f}）')
        else:
            scores.append(direction.reason)
    if scores:
        entries.append(('模型评分', ' / '.join(scores)))
    signals = []
    for source in ('cp_set', 'ow_state', 'representative_comparison', 'member_comparison'):
        finding = by_source.get(source)
        if finding is None or finding.state != 'signal':
            continue
        if source == 'cp_set' and finding.value == ('不确定', 0):
            signals.append('共形候选 0 类，正常和攻击均未入选')
        elif source == 'cp_set' and finding.value == ('不确定', 2):
            signals.append('共形候选 2 类，正常和攻击均保留')
        elif source == 'ow_state':
            signals.append(f'引擎标记为{finding.value}')
        else:
            signals.append(finding.reason)
    if signals:
        entries.append(('系统线索', '。'.join(signals)))
    if not entries:
        entries.append(('当前依据', '、'.join(candidate.reasons) or '记录定位或模型观测仍需核实'))
    rows = ''.join(f'<div><dt>{escape(str(label))}</dt><dd>{escape(str(value))}</dd></div>'
                   for label, value in entries)
    st.html('<style>.event-observations{margin:0;font-size:.9rem;line-height:1.5}'
            '.event-observations>div{display:grid;grid-template-columns:minmax(5em,24%) minmax(0,1fr);'
            'gap:.65rem;padding:.42rem 0;border-bottom:1px solid var(--line,#dde2ea)}'
            '.event-observations dt{font-weight:600}.event-observations dd{margin:0;overflow-wrap:anywhere}</style>'
            f'<dl class="event-observations" aria-label="本条记录的核查依据">{rows}</dl>')


def _candidate_options(evidence, key, locked, current_id, select_member, line_numbers):
    all_rows = {}
    for label, group in (('待核查', evidence.candidates), ('待补证', evidence.pending), ('已核查留存', evidence.retained)):
        all_rows.update((row.candidate.record_id, (label, row)) for row in group)
    if not all_rows:
        return None
    saved_key = key + '_saved_evidence_choice'
    saved = st.session_state.get(saved_key, {})
    if saved.get('member') != current_id:
        saved = {}
    paths = {'事件成员': 'event', '同批队列外': 'release'}
    visible_paths = [label for label, path in paths.items()
                     if any(path in row.candidate.paths for _, row in all_rows.values())]
    if not visible_paths:
        return None
    path = visible_paths[0]
    if len(visible_paths) > 1:
        available = (*visible_paths, '全部')
        path_key = key + '_path'
        path = st.selectbox('核查范围', available, key=path_key, disabled=locked,
                            index=available.index(saved['path']) if saved.get('path') in available else 0)
    options = {rid: item for rid, item in all_rows.items() if path == '全部' or paths[path] in item[1].candidate.paths}
    ids = tuple(options)
    remembered = saved.get('candidate') if saved.get('path') == path else None
    preferred = remembered if remembered in options else current_id if current_id in options else ids[0]
    rid = preferred
    if len(ids) > 1:
        choice_key = f'{key}_candidate_{path}_{current_id}'

        def pick_candidate():
            chosen = st.session_state[choice_key]
            sync_member = chosen != current_id and 'event' in options[chosen][1].candidate.paths and select_member
            st.session_state[saved_key] = dict(path=path, candidate=chosen, member=chosen if sync_member else current_id)
            if sync_member:
                select_member(chosen)

        if choice_key in st.session_state and st.session_state[choice_key] != preferred:
            st.session_state[choice_key] = preferred
        rid = st.selectbox('选择候选记录', ids, index=ids.index(preferred), key=choice_key, disabled=locked,
                           on_change=pick_candidate, format_func=lambda item:
                           f'{_record_location(item, line_numbers, options[item][1].candidate.physical_row)} · {options[item][0]}')
    st.session_state[saved_key] = dict(path=path, candidate=rid, member=current_id)
    return options[rid]


def render_evidence_section(view, actions, key, locked, select_member=None, current_id=None, line_numbers=None,
                            record_lookup=None, thresholds=None, evidence_lookup=None):
    evidence = view.evidence
    _render_human_evidence(evidence, actions, key, locked, line_numbers)
    option = _candidate_options(evidence, key, locked, current_id, select_member, line_numbers)
    if option is None:
        st.info('当前范围没有待查或留存的候选。可在研判历史核对已保存的依据。')
        return
    disposition, row = option
    candidate = row.candidate
    rid = candidate.record_id
    location = _record_location(rid, line_numbers, candidate.physical_row)
    representative = rid in view.overview.representatives
    role = '事件代表' if representative else _scope_text(candidate.paths)
    record = record_lookup(rid) if record_lookup else None
    if record is not None and record.get('record_id') != rid:
        record = None
    payload = evidence_lookup(rid) if evidence_lookup else None
    observation = None
    if payload and payload.get('record_id') == rid:
        from app_review_guidance import review_focus_rows

        observations, _ = review_focus_rows(payload)
        observation = observations[0] if observations else None
    with st.container(key='event_record_evidence'):
        st.markdown('**' + literal_markdown(f'{location} · {role} · {disposition}') + '**')
        gaps = [block for block in view.confirmation.gate.blockers
                if block.entry.record_id == rid and block.code != 'no_targets']
        if gaps:
            st.caption('待解决 · ' + literal_markdown(gaps[0].reason))
        elif disposition == '待补证' or not candidate.basis.evidence:
            st.caption('待核实 · ' + literal_markdown('、'.join(candidate.reasons)))
        elif candidate.prior_check and candidate.prior_check.reason:
            st.caption('上次核查 · ' + literal_markdown(candidate.prior_check.reason))
        _observations(candidate, record, thresholds, observation)
        if observation:
            step = observation['具体核查']
        elif disposition == '待补证' or not candidate.basis.evidence:
            step = '核对原始行和采集来源，补齐缺失信息。'
        else:
            step = '按原始记录的时间与资产，核对服务日志和授权任务。'
        st.markdown('**下一步**　' + literal_markdown(step))
        st.caption('把查到的事实写入人工依据。资料不足时，选择“待补证”并记下缺什么。')
        st.button(f'打开{location}核查', key=key+'_open_candidate', on_click=actions.request_record,
                  args=(rid,), disabled=locked, type='primary')
        with st.expander('查看完整技术依据'):
            st.json(asdict(candidate), expanded=False)


def render_batch_section(view, actions, key, locked, dirty, line_numbers=None):
    gate = view.gate
    line_numbers = dict(line_numbers or {})
    for candidate in (*gate.search.ordered_candidates, *gate.search.pending_verification, *gate.search.retained_checks):
        if candidate.physical_row is not None:
            line_numbers.setdefault(candidate.record_id, candidate.physical_row)
    grouped = {}
    for block in gate.blockers:
        if block.code != 'no_targets':
            grouped.setdefault((block.entry.record_id, block.entry.action), []).append(block)
    labels = {'review_record': '逐条核查', 'supplement_evidence': '补齐证据', 'refresh_preview': '重读依据'}
    if grouped:
        with st.container(key='event_block_action'):
            st.warning(f'批次确认有 {len(gate.blockers)} 项条件未满足，需办理 {len(grouped)} 项操作。')
            choices = tuple(grouped)
            selected = choices[0]
            if len(choices) > 1:
                saved_key = key + '_saved_block'
                preferred = st.session_state.get(saved_key)
                selected = st.selectbox('待处理动作', choices, key=key+'_blocked_choice',
                    index=choices.index(preferred) if preferred in grouped else 0, disabled=locked,
                    format_func=lambda item: _record_location(item[0], line_numbers) + ' · ' + labels.get(item[1], '核对'))
                st.session_state[saved_key] = selected
            entries = grouped[selected]
            entry = entries[0].entry
            for reason in dict.fromkeys(block.reason for block in entries):
                st.text(reason)
            location = _record_location(entry.record_id, line_numbers, entry.physical_row)
            if entry.action == 'refresh_preview':
                st.caption('按当前成员范围和人工核查预算重新读取依据，再核对可确认的目标。')
                if not view.requires_reread:
                    st.button('重读当前依据', key=key+'_refresh', on_click=actions.reread, disabled=locked)
            elif entry.record_id:
                st.caption('打开这条记录，对照原始观测与业务日志，保存人工依据后回到本事件。')
                st.button(f'{labels.get(entry.action, "核查")} · {location}', key=key+'_fix_block',
                          on_click=actions.resolve_block, args=(entry,), disabled=locked, type='primary')
            else:
                st.caption('请核对本事件的引擎状态和适用范围后再生成预览。')
    elif not gate.targets:
        unavailable = f'{len(gate.blockers)} 项条件均涉及授权范围。' if gate.blockers else ''
        st.info(unavailable + '当前可批量确认的成员为 0 条。可展开范围明细核对排除原因。')
    if dirty:
        st.info('请先保存当前填写，再生成批次预览或撤回。')
    if view.batch is None or view.requires_reread:
        _render_gate(gate, line_numbers)
    _render_batch(view, actions, key, locked, dirty, line_numbers)
    _render_withdrawal(view, actions, key, locked, dirty)
