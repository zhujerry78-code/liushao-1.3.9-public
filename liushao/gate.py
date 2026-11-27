"""Evidence authorization previews and SQLite submit-time revalidation."""

import hashlib
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import asdict, dataclass

from app_types import EngineStatus

from .reviews import _select, review_with_evidence
from .search import ANOMALY, NORMAL, AnalysisContext, Candidate, HumanReview, SearchResult, _same_engine


@dataclass(frozen=True)
class MemberState:
    status: str
    eligible: bool


@dataclass(frozen=True)
class RiskRequirement:
    source: str
    accepted_values: tuple[str, ...]


@dataclass(frozen=True)
class ReviewWork:
    record_id: str
    round_id: str
    purposes: frozenset[str]


@dataclass(frozen=True)
class WorkflowActivity:
    action: str
    seconds: float | None = None


@dataclass(frozen=True)
class GateState:
    """Trusted server inputs for one event or queue-external release.

    Analysis identity and revisions come from current server state. Members
    are the full event roster, or the release record. Eligibility includes
    whether inheritance has been withdrawn. Work contains the actual human
    judgments in the shared budget. Use 'review:<id>' for direct judgments.
    """

    search: SearchResult
    members: Mapping[str, MemberState]
    analysis_identity: AnalysisContext
    analysis_revision: int | str
    evidence_revision: int | str
    engine_status: EngineStatus
    risk_requirements: tuple[RiskRequirement, ...] = ()
    budget: int | None = None
    work: tuple[ReviewWork, ...] = ()
    activities: tuple[WorkflowActivity, ...] = ()


@dataclass(frozen=True)
class GateEntry:
    record_id: str | None
    evidence_source: str
    action: str
    review_id: int | None = None
    physical_row: int | None = None


@dataclass(frozen=True)
class GateBlock:
    code: str
    reason: str
    entry: GateEntry


@dataclass(frozen=True)
class MemberRevision:
    record_id: str
    state: MemberState
    review_status: str | None
    review_id: int | None
    evidence_sha256: str | None


@dataclass(frozen=True)
class Workload:
    rounds: tuple[ReviewWork, ...]
    unique_records: int
    repeat_checks: int
    within_budget: int
    overflow: int
    still_blocked: int
    additional_checks: int
    activities: tuple[WorkflowActivity, ...]


@dataclass(frozen=True)
class GateDecision:
    search: SearchResult
    release_id: str | None
    conclusion: str | None
    targets: tuple[str, ...]
    exclusions: tuple[tuple[str, str], ...]
    members: tuple[MemberRevision, ...]
    analysis_revision: int | str
    evidence_revision: int | str
    engine_status: EngineStatus
    risk_requirements: tuple[RiskRequirement, ...]
    budget: int | None
    blockers: tuple[GateBlock, ...]
    supporting_reviews: tuple[HumanReview, ...]
    counterexamples: tuple[HumanReview, ...]
    required_checks: tuple[Candidate, ...]
    completed_checks: tuple[Candidate, ...]
    recommendations: tuple[Candidate, ...]
    uncovered_members: tuple[str, ...]
    workload: Workload

    @property
    def passed(self):
        """Whether the operator may explicitly authorize the preview."""
        return not self.blockers and bool(self.targets)

    @property
    def requires_confirmation(self):
        return self.passed


def _block(code, reason, record_id, source, *, action='review_record',
           review_id=None, physical_row=None):
    return GateBlock(code, reason, GateEntry(record_id, source, action, review_id, physical_row))


def _current_direct(human, review, context):
    # Search decides applicability and direction. Verify that its review is
    # still the current persisted judgment and that its snapshot is readable.
    if not review or human.context != context:
        return False
    current = review_with_evidence(review)
    snapshot = current['evidence_snapshot']
    if (current.get('id') != human.review_id or current.get('status') != human.conclusion
            or current.get('note') != human.reason or snapshot is None
            or snapshot.get('record_id') != human.record_id or 'event_review' in snapshot):
        return False
    if any(snapshot.get(field) != getattr(context, field)
           for field in ('input_sha256', 'model_version', 'model_sha256', 'descriptor_sha256')):
        return False
    engine = snapshot.get('engine')
    if not isinstance(engine, dict):
        return not context.engine_identity and engine is None
    if not context.engine_identity:
        return engine.get('status') in (None, 'not_used')
    return (engine.get('status') == 'verified'
            and _same_engine(engine.get('identity'), engine.get('open_world'), context))


def _workload(state, direct, representatives, blockers, additional):
    rounds = {}
    for item in state.work:
        rounds.setdefault((item.record_id, item.round_id), set()).update(item.purposes)
    for human in direct:
        purposes = rounds.setdefault((human.record_id, f'review:{human.review_id}'), set())
        purposes.add('direct_review')
        if human.record_id in representatives:
            purposes.add('representative_review')
    entries = tuple(ReviewWork(rid, round_id, frozenset(purposes))
                    for (rid, round_id), purposes in sorted(rounds.items()))
    count = len(entries)
    unique = len({item.record_id for item in entries})
    within = count if state.budget is None else min(count, state.budget)
    affected = {item.entry.record_id for item in blockers if item.entry.record_id is not None}
    return Workload(entries, unique, count - unique, within, count - within,
                    len(affected), len(additional), state.activities)


def evaluate_gate(state: GateState, current_reviews: Mapping[str, Mapping],
                  *, release_id: str | None = None) -> GateDecision:
    """Evaluate an event preview or a single queue-external release in memory."""
    search = state.search
    scope, context = search.scope, search.context
    path = 'event' if release_id is None else 'release'
    representatives = scope.representative_ids if path == 'event' else frozenset()
    proposed = scope.authorization_ids if path == 'event' else frozenset({release_id})
    conclusion = scope.event_conclusion if path == 'event' else NORMAL
    candidates = (*search.ordered_candidates, *search.pending_verification, *search.retained_checks)
    observed = {item.record_id for item in candidates if path in item.paths}
    support = tuple(item for item in search.supporting_reviews if path in item.supporting_paths)
    counter = tuple(item for item in search.confirmed_counterexamples if path in item.counterexample_paths)
    observed.update(item.record_id for item in (*support, *counter))
    members = tuple(MemberRevision(rid, member, current_reviews.get(rid, {}).get('status'),
                                   current_reviews.get(rid, {}).get('id'),
                                   current_reviews.get(rid, {}).get('evidence_sha256'))
                    for rid, member in sorted(state.members.items())
                    if path == 'event' or rid == release_id)
    member_ids = {item.record_id for item in members}
    blockers, exclusions = [], {}
    additional = set()

    if state.analysis_identity != context:
        for rid in sorted(proposed | representatives) or [None]:
            blockers.append(_block('version_identity', 'Search 与当前分析的版本身份不一致', rid,
                                   'analysis_identity', action='refresh_preview'))
    if path == 'event':
        identity_errors = (member_ids ^ observed) | ((proposed | representatives) - member_ids)
        if scope.event_id is None or conclusion not in (NORMAL, ANOMALY) or not representatives:
            for rid in sorted(proposed | member_ids) or [None]:
                blockers.append(_block('required_check', '事件或代表的直接确认待核对', rid, 'representative'))
    else:
        identity_errors = proposed - (member_ids & observed & scope.pending_release_ids)
    for rid in sorted(identity_errors):
        blockers.append(_block('member_state', '成员范围或队列外放行身份不可核验', rid, 'membership',
                               action='refresh_preview'))

    support = tuple(item for item in support if item.record_id in member_ids)
    counter = tuple(item for item in counter if item.record_id in member_ids)
    watched = proposed | representatives
    valid_support, valid_direct = set(), set()
    direct = []
    for human in (*support, *counter):
        rid = human.record_id
        if not _current_direct(human, current_reviews.get(rid), context):
            if rid in watched:
                additional.add(rid)
                blockers.append(_block('critical_evidence', '直接复核或其证据快照已变化或不可核验',
                                       rid, 'direct_review', action='supplement_evidence',
                                       review_id=human.review_id))
            continue
        direct.append(human)
        valid_direct.add(rid)
        if path in human.supporting_paths:
            valid_support.add(rid)
        elif rid in watched and human.counterexample_applies_to_authorization:
            reason = f'直接复核结论为{human.conclusion}，与拟结论{conclusion}相反。{human.reason}'
            blockers.append(_block('direct_conflict', reason, rid, 'direct_review',
                                   review_id=human.review_id))

    for rid in sorted(representatives - valid_direct):
        additional.add(rid)
        blockers.append(_block('required_check', '代表缺少有效的直接确认', rid, 'representative',
                               review_id=current_reviews.get(rid, {}).get('id')))
    for rid in sorted((representatives & valid_direct) - valid_support):
        if not any(item.code == 'direct_conflict' and item.entry.record_id == rid for item in blockers):
            blockers.append(_block('direct_conflict', '代表的直接确认与拟结论不一致', rid,
                                   'representative', review_id=current_reviews[rid]['id']))

    pending_ids = {item.record_id for item in search.pending_verification}
    retained_ids = {item.record_id for item in search.retained_checks}
    required, completed = [], []
    locations = {}
    for item in candidates:
        rid = item.record_id
        if path not in item.paths or rid not in member_ids:
            continue
        locations[rid] = item.physical_row
        evidence = item.basis.evidence
        review = current_reviews.get(rid, {})
        review_id = review.get('id')
        pending_evidence = review.get('status') == '待补证'
        if rid in watched and rid not in pending_ids and review.get('status') in (NORMAL, ANOMALY):
            matching_inheritance = any(finding.source == 'event_review' and finding.value == review_id
                                       for finding in evidence)
            snapshot = review_with_evidence(review)['evidence_snapshot'] if matching_inheritance else None
            if not snapshot or 'event_review' not in snapshot:
                additional.add(rid)
                blockers.append(_block('critical_evidence', '当前确认记录与 Search 的复核依据不一致',
                                       rid, 'direct_review', action='supplement_evidence',
                                       review_id=review_id, physical_row=item.physical_row))
        unavailable = tuple(finding for finding in evidence if finding.category == 'authorization'
                            and finding.state not in ('available', 'signal'))
        # These are Search's findings and priorities, including findings on
        # retained checks whose Search priority is no longer populated.
        opposing = tuple(
            finding for finding in evidence if finding.category == 'prediction_direction'
            and finding.state == 'available' and finding.value in (NORMAL, ANOMALY)
            and finding.value != conclusion)
        heterogeneous = tuple(finding for finding in evidence
                              if finding.source == 'member_comparison' and finding.state == 'signal')
        release_signals = tuple(finding for finding in evidence if path == 'release'
                                and finding.category in ('model_conflict', 'prediction_uncertainty')
                                and finding.state == 'signal')
        mandatory = (path == 'event' and (heterogeneous or item.uncovered_form)) or (
            rid in watched and (rid in pending_ids or pending_evidence or unavailable or opposing or release_signals
                                or item.priority == 1))
        if mandatory:
            required.append(item)
        prior = item.prior_check
        retained = (rid in retained_ids and not pending_evidence and prior is not None
                    and prior.disposition == 'checked' and bool(prior.reason.strip())
                    and prior.basis.recheck_key == item.basis.recheck_key)
        if retained:
            completed.append(item)
        if rid in watched:
            for finding in unavailable:
                detail = getattr(finding.value, 'reason', finding.reason)
                blockers.append(_block('critical_evidence', detail, rid, finding.source,
                                       action='supplement_evidence', physical_row=item.physical_row))
        if mandatory and (not retained or unavailable):
            additional.add(rid)
            if unavailable:
                continue
            if pending_evidence or (rid in pending_ids and
                                    (not evidence or review.get('status') in (NORMAL, ANOMALY))):
                blockers.append(_block('critical_evidence', '；'.join(item.reasons), rid, 'search_pending',
                                       action='supplement_evidence', review_id=review_id,
                                       physical_row=item.physical_row))
                continue
            findings = (*opposing, *heterogeneous, *release_signals)
            details = tuple((finding.source, finding.reason) for finding in findings)
            for source, reason in dict.fromkeys(details or (('search_candidate', '；'.join(item.reasons)),)):
                blockers.append(_block('required_check', reason, rid, source, review_id=review_id,
                                       physical_row=item.physical_row))

    diagnostics = {item.source: item for item in search.batch_diagnostics}
    mode = diagnostics.get('b2_mode')
    degraded = state.engine_status.code != 'ok' or not context.engine_identity
    batch_blocks = []
    if degraded:
        batch_blocks.append(('engine_degraded', 'engine_status',
                             state.engine_status.message or '当前分析没有可验证的引擎身份'))
    if mode and mode.value == 'fallback':
        fallback = diagnostics.get('b2_fallback_reason', mode)
        batch_blocks.append(('engine_degraded', fallback.source, fallback.reason))
    for requirement in state.risk_requirements:
        diagnostic = diagnostics.get(requirement.source)
        if (diagnostic is None or diagnostic.state != 'available'
                or diagnostic.value not in requirement.accepted_values):
            batch_blocks.append(('critical_evidence', requirement.source,
                                 diagnostic.reason if diagnostic else '授权依赖的风险前提未核验'))
    for code, source, reason in batch_blocks:
        for rid in sorted(watched | additional) or [None]:
            blockers.append(_block(code, reason, rid, source, action='supplement_evidence',
                                   physical_row=locations.get(rid), review_id=current_reviews.get(rid, {}).get('id')))

    for member in members:
        rid = member.record_id
        if rid in valid_direct:
            exclusions[rid] = '已有有效的直接判断'
        elif rid in additional:
            exclusions[rid] = '仍需逐条核查或补证'
        elif rid not in proposed:
            exclusions[rid] = '当前拟授权范围之外'
        elif not member.state.eligible:
            exclusions[rid] = member.state.status
            blockers.append(_block('member_state', '拟授权成员当前不可授权', rid, 'membership',
                                   action='refresh_preview', review_id=member.review_id))
    targets = tuple(sorted((proposed & member_ids) - exclusions.keys()))
    if not targets:
        for rid in sorted(proposed) or [None]:
            blockers.append(_block('no_targets', '当前没有可授权目标', rid, 'authorization_scope'))

    workload = _workload(state, direct, representatives, blockers, additional)
    if workload.overflow:
        for rid in sorted(proposed) or [None]:
            blockers.append(_block('required_check', '已完成的人工核查超出当前授权预算', rid, 'review_budget',
                                   action='refresh_preview'))
        workload = _workload(state, direct, representatives, blockers, additional)
    covered = valid_direct | {item.record_id for item in completed}
    required_ids = {item.record_id for item in required}
    recommendations = tuple(item for item in search.recommendations
                            if path in item.paths and item.record_id in member_ids
                            and item.record_id not in required_ids)
    if state.budget is not None:
        remaining = max(0, state.budget - workload.within_budget - workload.additional_checks)
        recommendations = recommendations[:remaining]
    return GateDecision(search, release_id, conclusion, targets, tuple(sorted(exclusions.items())),
                        members, state.analysis_revision, state.evidence_revision, state.engine_status,
                        state.risk_requirements, state.budget, tuple(blockers), support, counter,
                        tuple(required), tuple(completed), recommendations,
                        tuple(sorted(member_ids - covered)), workload)


@dataclass(frozen=True)
class GateConfirmation:
    operation_id: str
    conclusion: str
    targets: tuple[str, ...]
    reason: str
    confirmed: bool


@dataclass(frozen=True)
class GateSubmission:
    operation_id: str
    replayed: bool


class GateBlocked(RuntimeError):
    def __init__(self, blockers):
        self.blockers = tuple(blockers)
        super().__init__('；'.join(dict.fromkeys(item.reason for item in self.blockers)))


def _submission_blocks(preview, code, reason, source, record_ids=(), *, action='refresh_preview'):
    ids = sorted(record_ids or preview.targets)
    return tuple(_block(code, reason, rid, source, action=action) for rid in ids or [None])


def _read_current_reviews(db, model_version, record_ids):
    reviews = {}
    for start in range(0, len(record_ids), 400):
        ids = record_ids[start:start + 400]
        placeholders = ','.join('?' for _ in ids)
        cursor = db.execute(_select(True) + f' WHERE r.model_version=? AND r.record_id IN ({placeholders})'
                            ' AND r.id=(SELECT MAX(id) FROM reviews'
                            ' WHERE model_version=r.model_version AND record_id=r.record_id)',
                            (model_version, *ids))
        names = tuple(column[0] for column in cursor.description)
        for row in cursor:
            review = dict(zip(names, row))
            reviews[review['record_id']] = review
    return reviews


def _revalidate(preview, current):
    def issue(code, reason, source, ids=()):
        return _submission_blocks(preview, code, reason, source, ids)

    blockers = list(current.blockers)
    if preview.search.context != current.search.context:
        blockers.extend(issue('version_identity', '分析版本身份已变化', 'analysis_identity'))
    if (preview.analysis_revision, preview.evidence_revision) != (current.analysis_revision, current.evidence_revision):
        blockers.extend(issue('revision', '分析或证据修订已变化', 'analysis_revision'))
    before = {item.record_id: item for item in preview.members}
    after = {item.record_id: item for item in current.members}
    changed_members = {rid for rid in before.keys() & after.keys()
                       if (before[rid].state, before[rid].review_status) != (after[rid].state, after[rid].review_status)}
    changed_members.update(before.keys() ^ after.keys())
    if changed_members:
        blockers.extend(issue('member_state', '成员范围或当前状态已变化', 'membership', changed_members))
    changed_reviews = {rid for rid in before.keys() & after.keys()
                       if (before[rid].review_id, before[rid].evidence_sha256)
                       != (after[rid].review_id, after[rid].evidence_sha256)}
    if changed_reviews:
        blockers.extend(issue('revision', '逐条复核修订或证据引用已变化', 'review_revision', changed_reviews))
    if not blockers and current != preview:
        blockers.extend(issue('preview_changed', '授权范围、证据或预算已变化', 'authorization_scope'))
    if blockers:
        raise GateBlocked(blockers)


def _operation_key(preview, confirmation):
    scope = preview.search.scope
    binding = dict(context=asdict(preview.search.context), event=scope.event_id,
                   representatives=sorted(scope.representative_ids), release_id=preview.release_id,
                   proposed=sorted(scope.authorization_ids if preview.release_id is None
                                   else {preview.release_id}),
                   members=[asdict(item) for item in preview.members],
                   exclusions=preview.exclusions, budget=preview.budget,
                   risk_requirements=[asdict(item) for item in preview.risk_requirements],
                   analysis_revision=preview.analysis_revision, evidence_revision=preview.evidence_revision,
                   conclusion=confirmation.conclusion, targets=confirmation.targets, reason=confirmation.reason)
    return hashlib.sha256(json.dumps(binding, sort_keys=True).encode('utf-8')).hexdigest()


def submit_gate(db: sqlite3.Connection, preview: GateDecision, confirmation: GateConfirmation | None,
                *, load_current, find_operation, write_operation) -> GateSubmission:
    """Recheck and write on one idle SQLite connection under BEGIN IMMEDIATE.

    The server supplies the preview and confirmation. load_current(db) returns
    the trusted GateState after reading current persisted membership, analysis
    identity and revisions under the lock. It provides the SearchResult for
    those materials. Gate reads current review rows directly under the lock.
    find_operation(db, operation_id) returns a persisted binding or None.
    write_operation(db, decision, confirmation, binding) saves the operation,
    binding, derived rows, dependencies and snapshots in the same transaction.
    Callbacks use this connection and leave transaction control to Gate.
    """
    if not preview.passed:
        raise GateBlocked(preview.blockers)
    if (confirmation is None or confirmation.confirmed is not True
            or not confirmation.operation_id.strip() or not confirmation.reason.strip()
            or confirmation.conclusion != preview.conclusion or confirmation.targets != preview.targets):
        raise GateBlocked(_submission_blocks(preview, 'confirmation_required', '需明确确认当前结论与目标并填写依据',
                                             'human_confirmation', action='confirm_preview'))
    binding = _operation_key(preview, confirmation)
    operation_id = confirmation.operation_id
    owns_transaction = False
    try:
        if db.in_transaction:
            raise GateBlocked(_submission_blocks(preview, 'transaction', '提交需要独立的 SQLite 写事务', 'database'))
        db.execute('BEGIN IMMEDIATE')
        owns_transaction = True
        existing = find_operation(db, operation_id)
        if existing is not None:
            if existing != binding:
                raise GateBlocked(_submission_blocks(preview, 'operation_conflict', '操作标识已用于另一份确认',
                                                     'operation'))
        else:
            state = load_current(db)
            current_reviews = _read_current_reviews(db, state.analysis_identity.model_version, sorted(state.members))
            current = evaluate_gate(state, current_reviews, release_id=preview.release_id)
            _revalidate(preview, current)
            write_operation(db, current, confirmation, binding)
            if find_operation(db, operation_id) != binding:
                raise GateBlocked(_submission_blocks(preview, 'transaction', '提交缺少同事务的操作记录', 'operation'))
        db.execute('COMMIT')
    except BaseException as exc:
        if owns_transaction and db.in_transaction:
            db.execute('ROLLBACK')
        if isinstance(exc, (OSError, sqlite3.Error)):
            raise GateBlocked(_submission_blocks(preview, 'transaction', '当前输入或数据库无法核验，提交已阻断',
                                                 'database')) from exc
        raise
    return GateSubmission(operation_id, replayed=existing is not None)
