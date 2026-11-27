"""Scope refresh, atomic review invalidation and revision-bound export.

Use the same idle SQLite connection and caller-owned Recovery attachment as
Recovery. A trusted loader builds Search and Gate from the supplied snapshot.
The loader also rechecks each original authorization against current evidence
and native membership, using its original proposal and authorization policy.
Display filters and draft budgets belong only to the new preview.
"""

import json
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from uuid import uuid4

from . import recovery
from .gate import GateDecision, _read_current_reviews
from .reviews import STATUSES, _select, _snapshot_json, review_with_evidence
from .search import ANOMALY, NORMAL, SearchResult


class RefreshRequired(recovery.RecoveryConflict):
    """The current materials must be verified or the artifact regenerated."""


@dataclass(frozen=True)
class Scope:
    """Stable server key and full record roster, including evidence dependencies."""

    key: str
    model_version: str
    record_ids: frozenset[str]


@dataclass(frozen=True)
class Authorization:
    batch_id: str
    original: str  # Saved Recovery evidence JSON, including the original Gate decision
    targets: tuple[str, ...]


@dataclass(frozen=True)
class ScopeInputs:
    reviews: Mapping[str, Mapping]
    decisions: Mapping[str, recovery.DecisionView]
    authorizations: tuple[Authorization, ...]
    # Read current reviews for an original authorization that spans rosters.
    reviews_for: Callable[[str, tuple[str, ...]], dict]


@dataclass(frozen=True)
class Analysis:
    search: SearchResult
    gate: GateDecision
    # Missing rechecks mean the original authorizations await verification.
    rechecks: Mapping[str, GateDecision]


Loader = Callable[[Scope, ScopeInputs], Analysis | None]


@dataclass(frozen=True)
class ReviewChange:
    record_id: str
    model_version: str
    status: str
    note: str
    expected_review_id: int
    evidence_snapshot: dict | None = None


@dataclass(frozen=True)
class Revision:
    scope: Scope
    checkpoint: str
    fingerprint: str


@dataclass(frozen=True)
class Preview:
    search: SearchResult
    gate: GateDecision
    revision: Revision


@dataclass(frozen=True)
class CurrentRecord:
    record_id: str
    decision: recovery.DecisionView
    inheritance_source: str | None = None
    withdrawal_batch: str | None = None
    physical_row: int | None = None
    evidence_json: str | None = None


@dataclass(frozen=True)
class CurrentScope:
    records: tuple[CurrentRecord, ...]
    preview: Preview | None


@dataclass(frozen=True)
class Export:
    revision: Revision
    exported_at: str
    checks: tuple[CurrentRecord, ...]
    history: tuple[str, ...]

    @property
    def conclusions(self):
        return tuple(row for row in self.checks if row.decision.state in (
            recovery.RecoveryState.DIRECT, recovery.RecoveryState.INHERITED))

    @property
    def confirmed_count(self):
        return len(self.conclusions)


@dataclass
class _Snapshot:
    scope: Scope
    reviews: dict
    latest: dict
    decisions: dict
    batches: dict
    operations: tuple
    authorization_versions: tuple

    @property
    def authorizations(self):
        return tuple(Authorization(batch_id, original['evidence'], tuple(
            row['record_id'] for row in active))
            for batch_id, (original, active) in self.batches.items() if active)


def _digest(value):
    return recovery._sha256(recovery._encode(value))


def _checkpoint(db, key):
    row = db.execute("SELECT id, binding, evidence FROM recovery.operations "
                     "WHERE kind='dynamic_scope' AND batch_id=? ORDER BY rowid DESC LIMIT 1",
                     (key,)).fetchone()
    return (row[0], row[1], json.loads(row[2])) if row else (None, None, {})


def _write_checkpoint(db, scope, fingerprint, reason):
    revision = uuid4().hex
    recovery._write_operation(db, revision, 'dynamic_scope', scope.key, reason, None, fingerprint,
                              {'scope': scope})
    return revision


def _authorization_versions(db, batch_id):
    versions = {}
    for operation, state, evidence in db.execute(
            "SELECT id, binding, evidence FROM recovery.operations WHERE kind='dynamic_authorization' "
            'AND batch_id=? ORDER BY rowid', (batch_id,)):
        versions[json.loads(evidence)['scope_key']] = operation, state
    return versions


def _set_authorization_pending(db, batch_id, scope_key, pending, reason):
    recovery._write_operation(db, uuid4().hex, 'dynamic_authorization', batch_id, reason, None,
                              'pending' if pending else 'verified', {'scope_key': scope_key})


def _related_rows(db, scope, previous):
    """Find ledger rows for both rosters, so scope movement rechecks old grants."""
    groups = {scope.model_version: set(scope.record_ids)}
    if previous:
        groups.setdefault(previous['model_version'], set()).update(previous['record_ids'])
    rows = []
    for model, ids in groups.items():
        keys = sorted(ids)
        for start in range(0, len(keys), 400):
            chunk = keys[start:start + 400]
            placeholders = ','.join('?' for _ in chunk)
            rows.extend(recovery._rows(db, 'SELECT * FROM recovery.inheritances WHERE model_version=? '
                                       f'AND record_id IN ({placeholders})', (model, *chunk)))
    return sorted(rows, key=lambda row: row['seq'])


def _snapshot(db, scope, previous=None):
    rows = _related_rows(db, scope, previous)
    latest = {(row['model_version'], row['record_id']): row for row in rows}
    batches = {}
    operations = []
    for batch_id in dict.fromkeys(row['batch_id'] for row in rows):
        original, members = recovery._batch(db, batch_id)
        active = []
        for row in members:
            if row['state'] != recovery.RecoveryState.INHERITED:
                continue
            key = row['model_version'], row['record_id']
            newest = latest.get(key) or recovery._latest(db, *key)
            if newest['id'] == row['id']:
                active.append(row)
        batches[batch_id] = original, tuple(active)
        operations.append(original)
        operations.extend(recovery._rows(db, 'SELECT * FROM recovery.operations WHERE batch_id=? '
                                         "AND kind IN ('invalidate','withdraw') ORDER BY rowid", (batch_id,)))
    reviews = _read_current_reviews(db, scope.model_version, sorted(scope.record_ids))
    own_latest = {rid: latest.get((scope.model_version, rid)) for rid in scope.record_ids}
    # Recovery.current_decision uses this same projection in its own read
    # transaction. Share it here so all records use one SQLite snapshot.
    decisions = {rid: recovery._view(reviews.get(rid, {}), own_latest[rid])
                 for rid in sorted(scope.record_ids)}
    versions = tuple(version for batch_id, (_, active) in batches.items() if active
                     for version in _authorization_versions(db, batch_id).values())
    return _Snapshot(scope, reviews, own_latest, decisions, batches, tuple(operations), versions)


def _load(db, snapshot, load):
    # The loader receives detached review dictionaries and Recovery conclusions.
    result = load(snapshot.scope, ScopeInputs(
        {rid: dict(row) for rid, row in snapshot.reviews.items()}, dict(snapshot.decisions),
        snapshot.authorizations, lambda model, ids: _read_current_reviews(db, model, sorted(set(ids)))))
    if result is None:
        return None
    scope = snapshot.scope
    if (result.search != result.gate.search or result.search.context.model_version != scope.model_version
            or {member.record_id for member in result.gate.members} != scope.record_ids):
        raise RefreshRequired('Search、Gate 与当前研判范围不一致')
    recovery._checked_reviews(db, result.gate)
    for auth in snapshot.authorizations:
        if auth.batch_id in result.rechecks:
            recovery._checked_reviews(db, result.rechecks[auth.batch_id])
    return result


def _verified_fingerprint(snapshot, analysis):
    if analysis is None or any(state == 'pending' for _, state in snapshot.authorization_versions):
        return None
    rechecks = {auth.batch_id: analysis.rechecks.get(auth.batch_id) for auth in snapshot.authorizations}
    if any(recheck is None for recheck in rechecks.values()):
        return None
    gate = analysis.gate
    return _digest((snapshot.scope, snapshot.reviews, snapshot.latest, snapshot.operations, snapshot.authorization_versions,
                    gate.search.context, gate.analysis_revision, gate.evidence_revision,
                    gate.engine_status, gate.risk_requirements, rechecks))


def _lost_dependencies(original, row, current, view):
    if view.state != recovery.RecoveryState.INHERITED or view.inheritance_id != row['id']:
        return ('当前逐条复核已取代该继承',)
    before = original['search']
    after = current.search
    identity = ('input_sha256', 'model_version', 'model_sha256', 'descriptor_sha256', 'engine_identity')
    after_context = json.loads(recovery._encode(after.context))
    reasons = []
    if any(before['context'][key] != after_context[key] for key in identity):
        reasons.append('原授权的分析身份已变化')
    # Event numbers can change when alpha reorders the native queue.
    if ({member['record_id'] for member in original['members']} !=
            {member.record_id for member in current.members}
            or (current.release_id is None and
                frozenset(before['scope']['representative_ids']) != after.scope.representative_ids)):
        reasons.append('原授权成员或代表范围已变化')
    refs = tuple(recovery.EvidenceRef(**ref) for ref in json.loads(row['dependencies']))
    proposed = after.scope.authorization_ids if current.release_id is None else {current.release_id}
    conflicts = {human.record_id for human in current.counterexamples
                 if human.counterexample_applies_to_authorization}
    for block in current.blockers:
        entry = block.entry
        if block.code == 'no_targets':
            continue
        if (entry.record_id in (None, row['record_id'])
                or block.code == 'direct_conflict' and entry.record_id in conflicts
                or entry.record_id not in proposed
                or any(ref.shared and ref.record_id == entry.record_id
                       and ref.source == entry.evidence_source for ref in refs)):
            reasons.append(block.reason)
    return tuple(dict.fromkeys(reasons))


def _invalidate(db, snapshot, analysis, reason):
    """Apply Gate's loss findings with Recovery's ledger transition and audit."""
    for batch_id, (original, active) in snapshot.batches.items():
        if not active:
            continue
        saved = json.loads(original['evidence'])['gate']
        gate = analysis.rechecks.get(batch_id)
        if gate is None:
            raise RefreshRequired('原授权依赖待核验')
        members = {member.record_id for member in gate.members}
        event = saved['release_id'] is None
        original_targets = set(saved['search']['scope']['authorization_ids'])
        missing_event = event and not members and gate.search.scope.event_id is None
        if (gate.release_id != saved['release_id'] or gate.search.context != analysis.search.context
                or not missing_event and gate.conclusion != saved['conclusion']
                or event and gate.search.scope.authorization_ids != original_targets & members):
            raise RefreshRequired('原授权重验未绑定原拟结论、目标及当前分析')
        model = active[0]['model_version']
        targets = {row['record_id'] for row in active}
        reviews = _read_current_reviews(db, model, sorted(targets))
        affected = {}
        for row in active:
            view = recovery._view(reviews.get(row['record_id'], {}), row)
            losses = _lost_dependencies(saved, row, gate, view)
            if losses:
                affected[row['id']] = {'record_id': row['record_id'], 'reasons': losses}
        if affected:
            revision = uuid4().hex
            evidence = {'original': recovery._reference(original), 'gate': gate,
                        'affected': affected, 'reviews': reviews}
            recovery._write_operation(db, revision, 'invalidate', batch_id, reason, None,
                                      _digest(evidence), evidence)
            for inheritance_id in affected:
                recovery._transition(db, inheritance_id, revision, recovery.RecoveryState.INVALIDATED)
        version = _authorization_versions(db, batch_id).get(snapshot.scope.key)
        if version is not None and version[1] == 'pending':
            _set_authorization_pending(db, batch_id, snapshot.scope.key, False, reason)


def _append_review(db, change):
    # reviews.save_review owns a separate connection. Reuse its persisted
    # schema and snapshot validation under the enclosing Dynamic transaction.
    rid, model, status, note = change.record_id, change.model_version, change.status, change.note
    expected = change.expected_review_id
    if (not rid or not model or status not in STATUSES or type(expected) is not int or expected < 0
            or len(note) > 4000 or status in (NORMAL, ANOMALY) and not note.strip()):
        raise ValueError('逐条复核、确认依据或复核修订无效')
    previous = _read_current_reviews(db, model, [rid]).get(rid, {})
    if (previous.get('id') or 0) != expected:
        raise RefreshRequired('逐条复核已变化，请重新读取草稿依据')
    snapshot = change.evidence_snapshot
    payload = _snapshot_json(snapshot, rid, model) if snapshot is not None else None
    cursor = db.execute('INSERT INTO reviews (record_id,model_version,status,note,created_at) '
                        'VALUES (?,?,?,?,?)', (rid, model, status, note, datetime.now(timezone.utc).isoformat()))
    if payload is not None:
        db.execute('INSERT INTO review_evidence VALUES (?,?,?,?)',
                   (cursor.lastrowid, snapshot.get('schema_version', 1), payload, recovery._sha256(payload)))


def _ensure_review_coverage(db, changes, snapshots):
    covered = {auth.batch_id for snapshot in snapshots for auth in snapshot.authorizations}
    changed = {(item.model_version, item.record_id) for item in changes}
    if not changed:
        return
    watched = {(snapshot.scope.model_version, rid) for snapshot in snapshots for rid in snapshot.scope.record_ids}
    if not changed <= watched:
        raise ValueError('写入的逐条复核必须属于当前研判范围')
    models = {model for model, _ in changed}
    for model in models:
        for batch_id, evidence in db.execute(
                'SELECT DISTINCT o.id,o.evidence FROM recovery.operations o '
                'JOIN recovery.inheritances i ON i.batch_id=o.id '
                'WHERE i.model_version=? AND i.state=? AND i.seq=('
                'SELECT MAX(j.seq) FROM recovery.inheritances j '
                'WHERE j.model_version=i.model_version AND j.record_id=i.record_id)',
                (model, recovery.RecoveryState.INHERITED)):
            original = json.loads(evidence)['gate']
            if batch_id not in covered and any((model, member['record_id']) in changed
                                                for member in original['members']):
                raise RefreshRequired('本次复核还影响其他授权范围，请一并重验')


def rebuild(db: sqlite3.Connection, scopes: Scope | tuple[Scope, ...], load: Loader, *,
            reason: str, reviews: tuple[ReviewChange, ...] = (),
            expected: Revision | None = None) -> tuple[CurrentScope, ...]:
    """Refresh affected scopes, optionally appending direct reviews atomically.

    Callers provide every affected scope for a review write. For scope movement
    reuse its stable key. A failed load or incomplete recheck rolls back the
    write. Mark a scope pending before an asynchronous analysis change.
    """
    scopes = (scopes,) if isinstance(scopes, Scope) else scopes
    if (not reason.strip() or not scopes or any(not scope.key or not scope.model_version for scope in scopes)
            or len({scope.key for scope in scopes}) != len(scopes)):
        raise ValueError('需要有效的研判范围和更新原因')
    with recovery._transaction(db):
        if expected is not None:
            revalidate(db, expected, load)
        previous = [_checkpoint(db, scope.key)[2].get('scope') for scope in scopes]
        if reviews:
            _ensure_review_coverage(db, reviews, [
                _snapshot(db, scope, before) for scope, before in zip(scopes, previous)])
            for change in reviews:
                _append_review(db, change)
        for scope, before in zip(scopes, previous):
            snapshot = _snapshot(db, scope, before)
            analysis = _load(db, snapshot, load)
            if analysis is None:
                raise RefreshRequired('当前分析待核验')
            _invalidate(db, snapshot, analysis, reason)
        results = []
        for scope in scopes:
            snapshot = _snapshot(db, scope)
            analysis = _load(db, snapshot, load)
            fingerprint = _verified_fingerprint(snapshot, analysis)
            if fingerprint is None:
                raise RefreshRequired('当前研判依赖待核验')
            revision = _write_checkpoint(db, scope, fingerprint, reason)
            results.append(_project(snapshot, analysis, revision))
        return tuple(results)


def mark_pending(db: sqlite3.Connection, scope: Scope, *, reason: str):
    """Invalidate old tokens before a scope change whose recheck is deferred."""
    if not reason.strip():
        raise ValueError('需要待刷新原因')
    with recovery._transaction(db):
        # Preserve the last roster until a successful rebuild can recheck it.
        previous = _checkpoint(db, scope.key)[2].get('scope')
        if previous:
            scope = Scope(scope.key, previous['model_version'], frozenset(previous['record_ids']))
        for auth in _snapshot(db, scope).authorizations:
            _set_authorization_pending(db, auth.batch_id, scope.key, True, reason)
        _write_checkpoint(db, scope, '', reason)


def _project(snapshot, analysis, checkpoint):
    preview = None
    authorized_rows = {}
    if checkpoint is not None:
        revision = Revision(snapshot.scope, checkpoint, _digest((analysis.search, analysis.gate)))
        preview = Preview(analysis.search, analysis.gate, revision)
        for batch_id, (original, active) in snapshot.batches.items():
            if active:
                search = json.loads(original['evidence'])['gate']['search']
                authorized_rows[batch_id] = {item['record_id']: item['physical_row']
                                             for item in search.get('ordered_candidates', ())
                                             if item.get('physical_row') is not None}
    physical_rows = {}
    if analysis is not None:
        search = analysis.search
        for group in (search.ordered_candidates, search.pending_verification, search.retained_checks):
            for item in group:
                if item.physical_row is not None:
                    physical_rows[item.record_id] = item.physical_row
    rows = []
    for rid, view in snapshot.decisions.items():
        inheritance = snapshot.latest[rid]
        inherited = view.state == recovery.RecoveryState.INHERITED
        if inherited and preview is None:
            view = recovery.DecisionView(recovery.RecoveryState.PENDING, action='review_record')
        source = view.batch_id if view.state == recovery.RecoveryState.INHERITED else None
        withdrawal = (inheritance['batch_id'] if inheritance and view.inheritance_id == inheritance['id']
                      and view.state == recovery.RecoveryState.INVALIDATED else None)
        saved = review_with_evidence(snapshot.reviews.get(rid, {}))['evidence_snapshot']
        if view.state == recovery.RecoveryState.DIRECT and analysis is not None:
            context = analysis.search.context
            identity = ('input_sha256', 'model_sha256', 'descriptor_sha256')
            if any(saved[key] != getattr(context, key) for key in identity):
                view = recovery.DecisionView(recovery.RecoveryState.PENDING, review_id=view.review_id,
                                             action='review_record', snapshot_status=view.snapshot_status)
        if inherited and preview is None:
            saved = None
        physical_row = (saved or {}).get('input_csv_row', physical_rows.get(rid))
        if source is not None:
            physical_row = authorized_rows.get(source, {}).get(rid, physical_row)
        rows.append(CurrentRecord(rid, view, source, withdrawal, physical_row,
                                  recovery._encode(saved) if saved is not None else None))
    return CurrentScope(tuple(rows), preview)


def _read(db, scope, load):
    snapshot = _snapshot(db, scope)
    checkpoint, expected, _ = _checkpoint(db, scope.key)
    try:
        analysis = _load(db, snapshot, load)
        actual = _verified_fingerprint(snapshot, analysis)
    except (recovery.RecoveryConflict, OSError, sqlite3.Error):
        analysis, actual = None, None
    ready = checkpoint if actual is not None and actual == expected else None
    return snapshot, _project(snapshot, analysis, ready)


def read_current(db: sqlite3.Connection, scope: Scope, load: Loader) -> CurrentScope:
    """Read current decisions, suppressing inheritance until it is verified."""
    with recovery._transaction(db, write=False):
        return _read(db, scope, load)[1]


def revalidate(db: sqlite3.Connection, revision: Revision, load: Loader) -> Preview:
    """Recheck a token using server materials, including under an action's lock.

    Saving or withdrawing via an existing Gate or Recovery action must run
    this guard under the same write transaction as that action.
    """
    def check():
        _, current = _read(db, revision.scope, load)
        if current.preview is None or current.preview.revision != revision:
            raise RefreshRequired('研判修订已变化，请重新预览或生成导出')
        return current.preview

    if db.in_transaction:
        return check()
    with recovery._transaction(db, write=False):
        return check()


def prepare_export(db: sqlite3.Connection, scope: Scope, load: Loader) -> Export:
    """Freeze the checks, confirmed conclusions and audit at one revision."""
    with recovery._transaction(db, write=False):
        snapshot, current = _read(db, scope, load)
        if current.preview is None:
            raise RefreshRequired('当前研判或存储待核验，请重新生成导出')
        direct = []
        record_ids = sorted(scope.record_ids)
        for start in range(0, len(record_ids), 400):
            ids = record_ids[start:start + 400]
            placeholders = ','.join('?' for _ in ids)
            direct.extend(recovery._rows(db, _select(True) + ' WHERE r.model_version=? '
                                         f'AND r.record_id IN ({placeholders})', (scope.model_version, *ids)))
        direct.sort(key=lambda row: row['id'])
        checks = tuple(replace(row, decision=replace(row.decision, inheritance_id=None, batch_id=None),
                               withdrawal_batch=None)
                       if row.decision.state not in (recovery.RecoveryState.DIRECT, recovery.RecoveryState.INHERITED)
                       else row for row in current.records)
        history = [review_with_evidence(row) for row in direct]
        return Export(current.preview.revision, datetime.now(timezone.utc).isoformat(), checks,
                      tuple(map(recovery._encode, (*history, *snapshot.operations))))


def download_export(db: sqlite3.Connection, export: Export, load: Loader) -> Export:
    """Return the frozen export only if its scope revision still matches."""
    try:
        revalidate(db, export.revision, load)
    except sqlite3.Error as exc:
        raise RefreshRequired('存储无法核验，请重新生成导出') from exc
    return export
