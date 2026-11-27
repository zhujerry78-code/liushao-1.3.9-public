"""Append-only authorization audit and recoverable inheritance.

Use an idle SQLite connection to the reviews v2 database with its dedicated,
caller-owned ledger attached as ``recovery``. Only the ledger is written here. Direct
reviews are appended through reviews.save_review and read under the same write
lock as the ledger. The attachment keeps the existing reviews schema compatible.
"""

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
from uuid import uuid4

from .gate import GateConfirmation, GateDecision, _read_current_reviews, submit_gate
from .reviews import ReviewConflictError, review_with_evidence, schema_kind
from .search import ANOMALY, NORMAL, SearchResult


class RecoveryState(StrEnum):
    PENDING = '待复核或待补证'
    INHERITED = '继承有效'
    INVALIDATED = '继承失效待重新确认'
    WITHDRAWN = '已撤回'
    DIRECT = '直接确认'


# Withdrawal is retained on the inheritance. Its current record then projects
# to PENDING, or to the later direct/new authorization that superseded it.
_AUTHORIZABLE = {RecoveryState.PENDING, RecoveryState.INVALIDATED, RecoveryState.WITHDRAWN}
_WITHDRAWABLE = {RecoveryState.INHERITED, RecoveryState.INVALIDATED}
_TRANSITIONS = {RecoveryState.INVALIDATED: (RecoveryState.INHERITED,),
                RecoveryState.WITHDRAWN: (RecoveryState.INHERITED, RecoveryState.INVALIDATED)}


class RecoveryConflict(ReviewConflictError):
    """The operator must read the new revision and confirm it again."""


@dataclass(frozen=True)
class EvidenceRef:
    """A dependency within one original authorization snapshot."""

    source: str
    record_id: str | None = None
    shared: bool = False


@dataclass(frozen=True)
class DecisionView:
    state: RecoveryState
    conclusion: str | None = None
    review_id: int | None = None
    inheritance_id: str | None = None
    action: str | None = None
    snapshot_status: str | None = None
    batch_id: str | None = None


@dataclass(frozen=True)
class InheritancePreview:
    decision: GateDecision
    revisions: tuple[tuple[str, str | None], ...]


@dataclass(frozen=True)
class WithdrawalMember:
    record_id: str
    inheritance_id: str
    validity_revision: str
    current_revision: str | None
    review_revision: tuple[int | None, str | None, str | None]
    disposition: str


@dataclass(frozen=True)
class WithdrawalPreview:
    batch_id: str
    members: tuple[WithdrawalMember, ...]


_OPERATIONS = '''CREATE TABLE recovery.operations (
    id TEXT PRIMARY KEY, kind TEXT NOT NULL, batch_id TEXT, reason TEXT NOT NULL,
    actor TEXT, created_at TEXT NOT NULL, binding TEXT NOT NULL, evidence TEXT NOT NULL)'''
_INHERITANCES = '''CREATE TABLE recovery.inheritances (
    seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, batch_id TEXT NOT NULL,
    model_version TEXT NOT NULL, record_id TEXT NOT NULL, conclusion TEXT NOT NULL,
    state TEXT NOT NULL, revision TEXT NOT NULL, baseline TEXT NOT NULL,
    dependencies TEXT NOT NULL, UNIQUE (batch_id, record_id))'''


@contextmanager
def _transaction(db, *, write=True):
    if db.in_transaction:
        raise RecoveryConflict('Recovery 需要独立事务')
    db.execute('BEGIN IMMEDIATE' if write else 'BEGIN')
    try:
        yield
        db.execute('COMMIT')
    except BaseException:
        if db.in_transaction:
            db.execute('ROLLBACK')
        raise


def initialize_recovery(db: sqlite3.Connection):
    """Initialize a new attachment or open the known ledger schema."""
    if schema_kind(db) != 'v2':
        raise ValueError('请通过 reviews 的现有兼容流程打开复核库')
    with _transaction(db):
        version = db.execute('PRAGMA recovery.user_version').fetchone()[0]
        schema = {row[0]: row[1] for row in db.execute(
            "SELECT name, sql FROM recovery.sqlite_master WHERE name NOT LIKE 'sqlite_%'")}
        expected = {'operations': _OPERATIONS.replace('recovery.', '', 1),
                    'inheritances': _INHERITANCES.replace('recovery.', '', 1),
                    'inheritance_current': 'CREATE INDEX inheritance_current ON inheritances '
                                           '(model_version, record_id, seq)'}
        if version == 1 and schema == expected:
            return
        if version != 0 or schema:
            raise ValueError('未知 Recovery 数据库结构；请保留原库并核对版本')
        db.execute(_OPERATIONS)
        db.execute(_INHERITANCES)
        db.execute('CREATE INDEX recovery.inheritance_current ON inheritances (model_version, record_id, seq)')
        db.execute('PRAGMA recovery.user_version=1')


def _encode(value):
    def convert(item):
        if is_dataclass(item):
            return asdict(item)
        if isinstance(item, (set, frozenset)):
            return sorted(item)
        raise TypeError(f'未知 Recovery 证据类型：{type(item).__name__}')

    return json.dumps(value, default=convert, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False)


def _rows(db, sql, params):
    cursor = db.execute(sql, params)
    names = tuple(column[0] for column in cursor.description)
    return [dict(zip(names, row)) for row in cursor]


def _operation(db, operation_id):
    return next(iter(_rows(db, 'SELECT * FROM recovery.operations WHERE id=?', (operation_id,))), None)


def _write_operation(db, operation_id, kind, batch_id, reason, actor, binding, evidence):
    db.execute('INSERT INTO recovery.operations VALUES (?,?,?,?,?,?,?,?)',
               (operation_id, kind, batch_id, reason, actor,
                datetime.now(timezone.utc).isoformat(), binding, _encode(evidence)))


def _sha256(serialized):
    return hashlib.sha256(serialized.encode('utf-8')).hexdigest()


def _reference(operation):
    return {'operation_id': operation['id'], 'evidence_sha256': _sha256(operation['evidence'])}


def operation_history(db: sqlite3.Connection, operation_id: str):
    """Return the persisted audit entry, including the original evidence and links."""
    operation = _operation(db, operation_id)
    if operation is not None:
        operation['actor'] = operation['actor'] or '未记录'
        operation['evidence'] = json.loads(operation['evidence'])
    return operation


def _latest(db, model_version, record_id):
    return next(iter(_rows(db, 'SELECT * FROM recovery.inheritances WHERE model_version=? '
                              'AND record_id=? ORDER BY seq DESC LIMIT 1', (model_version, record_id))), None)


def _review_revision(review):
    return review.get('id'), review.get('status'), review.get('evidence_sha256')


def _legacy_view(review):
    verified = review_with_evidence(review)
    snapshot = verified['evidence_snapshot']
    if snapshot is not None and snapshot.get('schema_version', 1) not in (1, 2):
        raise ValueError('未知复核证据快照版本；请保留原快照并核对版本')
    # The P0-hardened review_with_evidence reports an unknown version as
    # 'invalid' with snapshot None, so the check above never sees it. Read the
    # stored version column directly so the unknown version is still rejected
    # and the stored evidence is never rewritten.
    stored_version = review.get('evidence_schema_version')
    if stored_version is not None and stored_version not in (1, 2):
        raise ValueError('未知复核证据快照版本；请保留原快照并核对版本')
    action = 'supplement_evidence' if review.get('status') == '待补证' else 'review_record'
    if review.get('status') in (NORMAL, ANOMALY):
        if snapshot is not None:
            if 'event_review' in snapshot or snapshot['method_version'] == 'o1-event-review-v1':
                # The old event payload identifies representatives, but has no
                # durable batch operation and authorization dependency graph.
                return DecisionView(RecoveryState.INVALIDATED, review_id=review['id'],
                                    action='review_record', snapshot_status='saved')
            if type(review.get('id')) is int and review['id'] > 0 and review.get('note', '').strip():
                return DecisionView(RecoveryState.DIRECT, review['status'], review['id'], snapshot_status='saved')
        action = 'supplement_evidence'
    return DecisionView(RecoveryState.PENDING, review_id=review.get('id'), action=action,
                        snapshot_status=verified['evidence_snapshot_status'] if review else None)


def _view(review, latest):
    legacy = _legacy_view(review)
    if (legacy.state == RecoveryState.DIRECT or latest is None
            or legacy.snapshot_status == 'invalid'
            or _encode(_review_revision(review)) != latest['baseline']):
        return legacy
    state = RecoveryState(latest['state'])
    if state == RecoveryState.WITHDRAWN:
        return DecisionView(RecoveryState.PENDING, review_id=legacy.review_id, inheritance_id=latest['id'],
                            action=legacy.action, snapshot_status=legacy.snapshot_status)
    return DecisionView(state, latest['conclusion'] if state == RecoveryState.INHERITED else None,
                        legacy.review_id, latest['id'], 'review_record' if state == RecoveryState.INVALIDATED else None,
                         legacy.snapshot_status, batch_id=latest['batch_id'])


def _representatives_confirmed(db, model_version, view):
    original = _operation(db, view.batch_id)
    if not original or original['kind'] != 'authorize':
        return False
    try:
        gate = json.loads(original['evidence'])['gate']
        if gate['release_id'] is not None:
            return True
        ids = gate['search']['scope']['representative_ids']
    except (KeyError, TypeError, ValueError):
        return False
    if not isinstance(ids, list) or not ids or any(not isinstance(rid, str) or not rid for rid in ids):
        return False
    reviews = _read_current_reviews(db, model_version, ids)
    try:
        return all(reviews.get(rid, {}).get('status') == view.conclusion
                   and _legacy_view(reviews[rid]).state == RecoveryState.DIRECT for rid in ids)
    except ValueError:
        return False


def current_decision(db: sqlite3.Connection, model_version: str, record_id: str) -> DecisionView:
    """Project the current conclusion while keeping historical validity separate."""
    with _transaction(db, write=False):
        review = _read_current_reviews(db, model_version, [record_id]).get(record_id, {})
        view = _view(review, _latest(db, model_version, record_id))
        if view.state == RecoveryState.INHERITED and not _representatives_confirmed(db, model_version, view):
            return replace(view, state=RecoveryState.INVALIDATED, conclusion=None, action='review_record')
        return view


def _checked_reviews(db, decision):
    members = {member.record_id: member for member in decision.members}
    reviews = _read_current_reviews(db, decision.search.context.model_version, sorted(members))
    for rid, member in members.items():
        review = reviews.get(rid, {})
        if _review_revision(review) != (member.review_id, member.review_status, member.evidence_sha256):
            raise RecoveryConflict('逐条复核已变化，请重新预览')
        _legacy_view(review)
    return reviews


def _target_revisions(db, decision, reviews):
    revisions = []
    for rid in decision.targets:
        latest = _latest(db, decision.search.context.model_version, rid)
        if _view(reviews.get(rid, {}), latest).state not in _AUTHORIZABLE:
            raise RecoveryConflict('目标已有有效结论，请重新预览')
        revisions.append((rid, latest['revision'] if latest else None))
    return tuple(revisions)


def _check_decision(search, decision):
    if search != decision.search:
        raise ValueError('Search 结果与 Gate 决策不一致')


def preview_inheritance(db: sqlite3.Connection, search: SearchResult, decision: GateDecision) -> InheritancePreview:
    """Bind a passed Gate preview to the Recovery revisions to be confirmed."""
    _check_decision(search, decision)
    if not decision.passed:
        raise ValueError('Gate 未通过')
    with _transaction(db, write=False):
        return InheritancePreview(decision, _target_revisions(db, decision, _checked_reviews(db, decision)))


def _dependencies(decision, reviews):
    search = decision.search
    shared = {EvidenceRef(source, shared=True) for source in
              ('analysis_identity', 'engine_status', 'membership', 'b2_mode', 'b2_fallback_reason',
               *(item.source for item in decision.risk_requirements))}
    for human in (*decision.supporting_reviews, *decision.counterexamples):
        direct = _legacy_view(reviews.get(human.record_id, {}))
        if direct.state != RecoveryState.DIRECT or direct.review_id != human.review_id:
            continue
        shared.add(EvidenceRef('direct_review', human.record_id, True))
        if human.record_id in search.scope.representative_ids and decision.release_id is None:
            shared.add(EvidenceRef('representative', human.record_id, True))

    # Gate completed checks and coverage can support the entire authorization.
    # Target evidence remains independently addressable even in the same batch.
    for check in decision.completed_checks:
        for source in ('search_candidate', 'search_pending', *(item.source for item in check.basis.evidence)):
            shared.add(EvidenceRef(source, check.record_id, True))

    path = 'release' if decision.release_id is not None else 'event'
    local = {rid: {EvidenceRef(source, rid) for source in
                   ('direct_review', 'membership', 'search_candidate', 'search_pending')}
             for rid in decision.targets}
    for candidate in (*search.ordered_candidates, *search.pending_verification, *search.retained_checks):
        if candidate.record_id in local and path in candidate.paths:
            local[candidate.record_id].update(EvidenceRef(finding.source, candidate.record_id)
                                              for finding in candidate.basis.evidence)
    shared_sources = {(ref.source, ref.record_id) for ref in shared}
    return {rid: sorted(shared | {ref for ref in refs if (ref.source, ref.record_id) not in shared_sources},
                        key=lambda ref: (ref.shared, ref.record_id or '', ref.source))
            for rid, refs in local.items()}


def submit_inheritance(db: sqlite3.Connection, preview: InheritancePreview,
                       confirmation: GateConfirmation, *, load_current, actor: str | None = None):
    """Use Gate's submit-time revalidation and write its ledger in that transaction."""
    request = _sha256(_encode((preview, confirmation, actor)))

    def find_operation(connection, operation_id):
        existing = _operation(connection, operation_id)
        if existing is None:
            return None
        if existing['kind'] != 'authorize' or json.loads(existing['evidence'])['request_sha256'] != request:
            raise RecoveryConflict('操作标识已用于另一份确认')
        return existing['binding']

    def write_operation(connection, decision, confirmed, binding):
        reviews = _checked_reviews(connection, decision)
        if _target_revisions(connection, decision, reviews) != preview.revisions:
            raise RecoveryConflict('继承修订已变化，请重新预览')
        if decision.release_id is None:
            for rid in decision.search.scope.representative_ids:
                if _legacy_view(reviews.get(rid, {})).state != RecoveryState.DIRECT:
                    raise RecoveryConflict('代表必须有可核验的直接复核')
        _write_operation(connection, confirmed.operation_id, 'authorize', None, confirmed.reason, actor, binding,
                         {'request_sha256': request, 'gate': decision, 'reviews': reviews})
        for rid, refs in _dependencies(decision, reviews).items():
            connection.execute('INSERT INTO recovery.inheritances '
                               '(id,batch_id,model_version,record_id,conclusion,state,revision,baseline,dependencies) '
                               'VALUES (?,?,?,?,?,?,?,?,?)',
                               (uuid4().hex, confirmed.operation_id, decision.search.context.model_version,
                                rid, decision.conclusion, RecoveryState.INHERITED, confirmed.operation_id,
                                _encode(_review_revision(reviews.get(rid, {}))), _encode(refs)))

    return submit_gate(db, preview.decision, confirmation, load_current=load_current,
                       find_operation=find_operation, write_operation=write_operation)


def _batch(db, batch_id):
    operation = _operation(db, batch_id)
    if not operation or operation['kind'] != 'authorize':
        raise ValueError('原批次授权不可核验，请逐条复核')
    rows = _rows(db, 'SELECT * FROM recovery.inheritances WHERE batch_id=? ORDER BY record_id', (batch_id,))
    if not rows:
        raise ValueError('原批次缺少继承记录，请核对审计库')
    return operation, rows


def inheritance_history(db: sqlite3.Connection, batch_id: str):
    """Read each inheritance identity, validity, baseline and evidence dependencies."""
    _, rows = _batch(db, batch_id)
    for row in rows:
        row['baseline'] = tuple(json.loads(row['baseline']))
        row['dependencies'] = tuple(EvidenceRef(**ref) for ref in json.loads(row['dependencies']))
    return rows


def _inspect(db, batch_id, rows):
    model_version = rows[0]['model_version']
    reviews = _read_current_reviews(db, model_version, [row['record_id'] for row in rows])
    members = []
    for row in rows:
        rid = row['record_id']
        review = reviews.get(rid, {})
        latest = _latest(db, model_version, rid)
        view = _view(review, latest)
        if view.state == RecoveryState.DIRECT:
            disposition = 'direct_review'
        elif latest['id'] != row['id']:
            disposition = 'other_authorization'
        elif row['state'] == RecoveryState.WITHDRAWN:
            disposition = 'already_withdrawn'
        elif view.inheritance_id != row['id']:
            disposition = 'later_review'
        elif row['state'] in _WITHDRAWABLE:
            disposition = 'affected'
        else:
            raise ValueError('原批次继承状态不可核验')
        members.append(WithdrawalMember(rid, row['id'], row['revision'], latest['revision'],
                                        _review_revision(review), disposition))
    return WithdrawalPreview(batch_id, tuple(members)), reviews


def preview_withdrawal(db: sqlite3.Connection, batch_id: str) -> WithdrawalPreview:
    with _transaction(db, write=False):
        _, rows = _batch(db, batch_id)
        return _inspect(db, batch_id, rows)[0]


def _transition(db, inheritance_id, revision, state):
    allowed = _TRANSITIONS[state]
    cursor = db.execute('UPDATE recovery.inheritances SET state=?, revision=? '
                        'WHERE id=? AND state IN (' + ','.join('?' for _ in allowed) + ')',
                        (state, revision, inheritance_id, *allowed))
    if cursor.rowcount != 1:
        raise RecoveryConflict('继承状态已变化，请重新预览')


def _validate_operation(operation_id, reason):
    if not operation_id.strip() or not reason.strip():
        raise ValueError('请填写操作标识与原因')


def _replay(db, operation_id, kind, binding):
    existing = _operation(db, operation_id)
    if existing is None:
        return False
    if (existing['kind'], existing['binding']) != (kind, binding):
        raise RecoveryConflict('操作标识已用于另一份确认')
    return True


def withdraw(db: sqlite3.Connection, preview: WithdrawalPreview, *, operation_id: str,
             reason: str, confirmed: bool, actor: str | None = None):
    """Withdraw exactly the previewed dependencies and return the saved audit entry."""
    _validate_operation(operation_id, reason)
    if confirmed is not True:
        raise ValueError('撤回需要明确确认影响范围')
    binding = _sha256(_encode((preview, reason, actor)))
    with _transaction(db):
        if _replay(db, operation_id, 'withdraw', binding):
            return operation_history(db, operation_id)
        batch, rows = _batch(db, preview.batch_id)
        existing = db.execute('SELECT id FROM recovery.operations WHERE batch_id=? AND kind=?',
                              (preview.batch_id, 'withdraw')).fetchone()
        if existing is not None:
            return operation_history(db, existing[0])
        current, reviews = _inspect(db, preview.batch_id, rows)
        if current != preview:
            raise RecoveryConflict('撤回影响范围或复核修订已变化，请重新确认')
        evidence = {'original': _reference(batch), 'impact': current, 'reviews': reviews,
                    'validity': [_reference(_operation(db, revision)) for revision in sorted(
                        {member.validity_revision for member in current.members} - {batch['id']})]}
        _write_operation(db, operation_id, 'withdraw', batch['id'], reason, actor, binding, evidence)
        for member in current.members:
            if member.disposition == 'affected':
                _transition(db, member.inheritance_id, operation_id, RecoveryState.WITHDRAWN)
        return operation_history(db, operation_id)


def _verify_blocked_dependency(original, dependency, decision):
    source = dependency.source
    block_codes = {'analysis_identity': 'version_identity', 'engine_status': 'engine_degraded',
                   'b2_mode': 'engine_degraded', 'b2_fallback_reason': 'engine_degraded'}
    code = block_codes.get(source) if dependency.shared and dependency.record_id is None else None
    if not any(block.entry.evidence_source == source
               and (dependency.record_id is None or block.entry.record_id == dependency.record_id)
               and (code is None or block.code == code) for block in decision.blockers):
        raise ValueError('Gate 未判定该授权证据失效')
    if not dependency.shared or dependency.record_id is not None:
        return
    if source == 'membership':
        original_scope = original['search']['scope']
        current_scope = decision.search.scope
        if ({member['record_id'] for member in original['members']} ==
                {member.record_id for member in decision.members}
                and set(original_scope['representative_ids']) == current_scope.representative_ids):
            raise ValueError('批次成员范围未变化，请核对目标证据依赖')
    elif source in {requirement['source'] for requirement in original['risk_requirements']}:
        before = [finding for finding in original['search']['batch_diagnostics'] if finding['source'] == source]
        after = [finding for finding in decision.search.batch_diagnostics if finding.source == source]
        if _encode(before) == _encode(after):
            raise ValueError('批次风险前提未变化，请核对目标证据依赖')


def invalidate(db: sqlite3.Connection, batch_id: str, dependency: EvidenceRef, search: SearchResult,
               decision: GateDecision, *, operation_id: str, reason: str, actor: str | None = None):
    """Invalidate current dependents of a recorded premise blocked by the new Gate decision."""
    _validate_operation(operation_id, reason)
    _check_decision(search, decision)
    binding = _sha256(_encode((batch_id, dependency, decision, reason, actor)))
    with _transaction(db):
        if _replay(db, operation_id, 'invalidate', binding):
            return operation_history(db, operation_id)
        batch, rows = _batch(db, batch_id)
        original = json.loads(batch['evidence'])['gate']
        context = original['search']['context']
        if (search.context.input_sha256 != context['input_sha256']
                or search.context.model_version != context['model_version']
                or decision.release_id != original['release_id']
                or decision.conclusion != original['conclusion']
                or (decision.release_id is None and search.scope.event_id != original['search']['scope']['event_id'])):
            raise ValueError('Gate 决策与原授权的分析范围不一致')
        _verify_blocked_dependency(original, dependency, decision)
        _checked_reviews(db, decision)
        dependents = {row['id'] for row in rows if asdict(dependency) in json.loads(row['dependencies'])}
        if not dependents:
            raise ValueError('原授权未记录该证据依赖')
        impact, reviews = _inspect(db, batch_id, rows)
        inherited = {row['id'] for row in rows if row['state'] == RecoveryState.INHERITED}
        affected = tuple(member for member in impact.members if member.disposition == 'affected'
                         and member.inheritance_id in dependents & inherited)
        _write_operation(db, operation_id, 'invalidate', batch_id, reason, actor, binding,
                         {'original': _reference(batch), 'dependency': dependency, 'gate': decision,
                          'affected': affected, 'reviews': reviews})
        for member in affected:
            _transition(db, member.inheritance_id, operation_id, RecoveryState.INVALIDATED)
        return operation_history(db, operation_id)
