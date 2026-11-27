"""P1-1 counterexample search over one trusted, unfiltered analysis batch."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from app_engine_evidence import IDENTITY_COLUMNS
from .reference import number
from .reviews import review_with_evidence


NORMAL, ANOMALY = '确认正常', '确认异常'
LABELS = {'正常': NORMAL, '攻击': ANOMALY}
DIRECTION_NAMES = {NORMAL: '正常', ANOMALY: '异常'}


@dataclass(frozen=True)
class AnalysisContext:
    input_sha256: str
    model_version: str
    model_sha256: str
    descriptor_sha256: str
    engine_identity: tuple[str, ...] = ()
    risk_alpha: float | None = None


@dataclass(frozen=True)
class SearchScope:
    event_id: int | None = None
    event_conclusion: str | None = None
    representative_ids: frozenset[str] = frozenset()
    authorization_ids: frozenset[str] = frozenset()
    pending_release_ids: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Finding:
    category: str
    source: str
    state: str
    reason: str
    value: object = None


@dataclass(frozen=True)
class CheckBasis:
    context: AnalysisContext
    paths: tuple[str, ...]
    event_question: tuple | None
    evidence: tuple[Finding, ...]

    @property
    def recheck_key(self):
        # A check can itself expand coverage. That bookkeeping alone cannot reopen it.
        relevant = tuple(item for item in self.evidence if item.source != 'member_comparison'
                         or item.category == 'authorization')
        return self.context, self.paths, self.event_question, relevant


@dataclass(frozen=True)
class PriorCheck:
    basis: CheckBasis
    disposition: str
    reason: str


@dataclass(frozen=True)
class Candidate:
    record_id: str
    paths: tuple[str, ...]
    native_queue: str | None
    physical_row: int | None
    engine_rank: int | None
    basis: CheckBasis
    reasons: tuple[str, ...]
    priority: int | None = None
    uncovered_form: bool = False
    prior_check: PriorCheck | None = None


@dataclass(frozen=True)
class HumanReview:
    record_id: str
    review_id: int
    conclusion: str
    reason: str
    context: AnalysisContext
    counterexample_paths: tuple[str, ...]
    supporting_paths: tuple[str, ...]
    counterexample_applies_to_authorization: bool


@dataclass(frozen=True)
class SearchResult:
    context: AnalysisContext
    scope: SearchScope
    ordered_candidates: tuple[Candidate, ...]
    budget: int | None
    confirmed_counterexamples: tuple[HumanReview, ...]
    supporting_reviews: tuple[HumanReview, ...]
    pending_verification: tuple[Candidate, ...]
    retained_checks: tuple[Candidate, ...]
    engine_rank_used: bool
    batch_diagnostics: tuple[Finding, ...]

    @property
    def recommendations(self):
        return self.ordered_candidates[:self.budget]


def _text(value):
    return value if isinstance(value, str) else None


def _integer(value, minimum):
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value) if value >= minimum else None
    if (isinstance(value, Real) and not isinstance(value, (bool, np.bool_))
            and np.isfinite(value) and value >= minimum and int(value) == value):
        return int(value)
    return None


def _bool(value):
    return bool(value) if isinstance(value, (bool, np.bool_)) else None


def _probability(value):
    value = number(value) if isinstance(value, Real) and _bool(value) is None else None
    return value if value is not None and 0 <= value <= 1 else None


def _unknown(category, source, engine_available=True):
    state = 'unknown' if engine_available else 'not_applicable'
    label = '未验证或缺失' if engine_available else '本次未使用引擎'
    return Finding(category, source, state, f'{source}：{label}')


def _engine_matches(row, context):
    identity = tuple(_text(row.get(name)) for name in IDENTITY_COLUMNS)
    if context.engine_identity:
        alpha_matches = ('risk_alpha' not in row if context.risk_alpha is None
                         else _probability(row.get('risk_alpha')) == context.risk_alpha)
        return identity == context.engine_identity and alpha_matches
    return not any(identity) and 'cp_set' not in row


def _snapshot_matches(snapshot, row, physical, context):
    if not isinstance(snapshot, dict) or physical is None:
        return False
    if any(snapshot.get(key) != getattr(context, key) for key in
           ('input_sha256', 'model_version', 'model_sha256', 'descriptor_sha256')):
        return False
    source = snapshot.get('identity')
    if (snapshot.get('record_id') != row['record_id'] or snapshot.get('input_csv_row') != physical
            or not isinstance(source, dict) or any(source.get(key) is None or source[key] != row.get(key)
                                                   for key in ('source_file', 'source_sha256', 'source_row'))):
        return False
    engine = snapshot.get('engine') or {}
    if not isinstance(engine, dict):
        return False
    if 'event_review' in snapshot:
        inheritance = snapshot['event_review']
        if not isinstance(inheritance, dict) or not context.engine_identity:
            return False
        identity, world = inheritance.get('engine_identity'), inheritance.get('open_world')
        return _same_engine(identity, world, context)
    if not context.engine_identity:
        return engine.get('status') in (None, 'not_used')
    return engine.get('status') == 'verified' and _same_engine(engine.get('identity'), engine.get('open_world'), context)


def _same_engine(identity, world, context):
    return (isinstance(identity, dict) and isinstance(world, dict) and world.get('alpha') == context.risk_alpha
            and tuple(identity.get(name) for name in IDENTITY_COLUMNS) == context.engine_identity)


def _queue(row, context):
    if not _engine_matches(row, context):
        return None
    if context.engine_identity:
        if 'ow_queue' in row:
            queue = _text(row['ow_queue'])
            return queue if queue in ('', '模糊', '新异', '风险控制') else None
        flag = _bool(row.get('engine_review'))
        return None if flag is None else ('需复核' if flag else '')
    # Same native subset as liushao.model.review_ranked.
    flags = (_bool(row.get('disagreement')), _bool(row.get('near_threshold')))
    if True in flags:
        return '需复核'
    return '' if flags == (False, False) else None


def _prediction_findings(row, context):
    conflict, uncertainty = 'model_conflict', 'prediction_uncertainty'
    directions = {name: ANOMALY if flag else NORMAL for name in ('RF', 'LR')
                  if (flag := _bool(row.get(name.lower() + '_anomaly'))) is not None}
    pair = tuple(directions.get(name) for name in ('RF', 'LR'))
    disagree = None not in pair and pair[0] != pair[1]
    conflict_reason = (f'RF 指向{DIRECTION_NAMES[pair[0]]}，LR 指向{DIRECTION_NAMES[pair[1]]}'
                       if disagree else 'RF/LR 阈值方向可核对')
    findings = [Finding(conflict, 'RF/LR', 'signal' if disagree else 'available',
                        conflict_reason, pair)
                if None not in pair else _unknown(conflict, 'RF/LR')]

    near = _bool(row.get('near_threshold'))
    findings.append(_unknown(uncertainty, 'near_threshold') if near is None else
                    Finding(uncertainty, 'near_threshold', 'signal' if near else 'available',
                            'RF 接近现有判定阈值' if near else 'RF 处于现有近阈值范围外', near))
    engine_available = bool(context.engine_identity)
    readouts = row if engine_available else {}
    k = _probability(readouts.get('ds_conflict'))
    gate = _bool(readouts.get('ow_conflict_gate'))
    origin = '（开放框架 RF/LR）' if 'ds_bel_unknown' in readouts else ''
    findings.extend((
        _unknown(conflict, 'ds_conflict', engine_available) if k is None else
        Finding(conflict, 'ds_conflict', 'available', f'D-S 冲突量 K{origin}={k:g}', k),
        _unknown(conflict, 'ow_conflict_gate', engine_available) if gate is None else
        Finding(conflict, 'ow_conflict_gate', 'signal' if gate else 'available',
                '引擎冲突门控触发' if gate else '引擎冲突门控未触发', gate),
    ))

    label, size = _text(readouts.get('cp_set')), _integer(readouts.get('cp_set_size'), 0)
    if size in (0, 2) and label == '不确定':
        reason = '共形空集，两类均未纳入' if size == 0 else '共形双标签，两类均保留'
        findings.append(Finding(uncertainty, 'cp_set', 'signal', reason, (label, size)))
    elif size == 1 and label in LABELS:
        directions['共形单标签'] = LABELS[label]
        findings.append(Finding(uncertainty, 'cp_set', 'available', f'共形单标签：{label}', (label, size)))
    else:
        findings.append(_unknown(uncertainty, 'cp_set', engine_available))

    state = _text(readouts.get('ow_state'))
    if state in ('正常', '攻击', '模糊', '新异'):
        findings.append(Finding(uncertainty, 'ow_state', 'signal' if state in ('模糊', '新异') else 'available',
                                f'引擎四态：{state}', state))
        if state in LABELS:
            directions['引擎四态'] = LABELS[state]
    else:
        findings.append(_unknown(uncertainty, 'ow_state', engine_available))
    low, high = (_probability(readouts.get(key)) for key in ('va_low', 'va_high'))
    findings.append(Finding(uncertainty, 'venn_abers', 'available', 'Venn–Abers 概率端点', (low, high))
                    if low is not None and high is not None and low <= high else
                    _unknown(uncertainty, 'venn_abers', engine_available))

    if engine_available:
        decision = (LABELS.get(state) if 'ow_state' in readouts else directions.get('共形单标签'))
    else:
        decision = pair[0] if pair[0] == pair[1] else None
    return tuple(findings), directions, decision


def _event_shapes(directions, payload):
    # Frozen event membership provides the shared signature. Compare directions
    # only within a source, and ECOD shapes only when every reported tail is usable.
    shapes = dict(directions)
    tails, reference, responses = [], [], []
    if payload and 'event_review' not in payload:
        engine = payload['engine']
        cards = {card['field']: card for card in payload.get('cards', ())}
        top = engine.get('ecod_decomposition', {}).get('top_fields', ())
        for tail in top:
            field, direction = tail.get('field'), tail.get('direction')
            if not isinstance(field, str) or direction not in ('left', 'right') or tail.get('input_missing') is not False:
                continue
            tails.append((field, direction))
            card = cards.get(field, {})
            position = card.get('reference_position') or {}
            if position.get('valid') is True and card.get('missing') is False and isinstance(position.get('status'), str):
                reference.append((field, position['status']))
                if card.get('role') in ('支持当前模型提示', '反向线索'):
                    responses.append((field, card['role']))
        if tails and len(tails) == len(top):
            shapes['ECOD 尾部特征与方向'] = tuple(sorted(tails))
        for field, status in reference:
            shapes[f'{field} 的训练正常参考位置'] = status
    return shapes, tuple(tails), tuple(reference), tuple(responses)


def _batch_diagnostics(row, context):
    if row is None or not context.engine_identity or not _engine_matches(row, context):
        return tuple(_unknown('applicability', name, bool(context.engine_identity))
                     for name in ('drift', 'guarantee_status'))
    drift = _bool(row.get('drift_flag'))
    strength = _probability(row.get('drift_delta'))
    missing_strength = row.get('drift_delta') is None or (
        isinstance(row.get('drift_delta'), Real) and np.isnan(row['drift_delta']))
    if drift is False and 'drift_delta' in row and missing_strength:
        out = [Finding('applicability', 'drift', 'not_tested', '本批未检测漂移')]
    elif drift is not None and strength is not None:
        deltas = tuple(_probability(row.get('drift_delta_' + name)) for name in ('rf', 'lr', 'typ'))
        out = [Finding('applicability', 'drift', 'available', '本批检测到漂移' if drift else '本批未检测到漂移',
                       (drift, strength, deltas))]
    else:
        out = [_unknown('applicability', 'drift')]
    guarantee = _text(row.get('guarantee_status'))
    states = {'有效': 'available', '加权有效': 'available', '失效': 'invalid', '不适用': 'not_applicable'}
    out.append(Finding('applicability', 'guarantee_status', states[guarantee], f'逐条风险保证：{guarantee}', guarantee)
               if guarantee in states else _unknown('applicability', 'guarantee_status'))
    for key in ('b2_mode', 'b2_fallback_reason', 'b3_calculation_status', 'b3_statistical_status'):
        if value := _text(row.get(key)):
            out.append(Finding('applicability', key, 'available', value, value))
    return tuple(out)


def search_counterexamples(predictions, physical_rows, *, context, scope, reviews=None,
                           evidence=None, required=None, prior_checks=None, reopen=None, budget=None):
    """Search app_service predictions with aligned physical rows and base_meta identity.

    Pass current persisted reviews with full evidence (e.g. reviews.current_reviews).
    Optional evidence is current, server-built record evidence with a verified engine
    block. Required source names are supplied per record by the authorization caller.
    Pass prior check receipts to retain dispositions. Reopening a completed check
    requires a changed basis and a caller-supplied reason the check is insufficient.
    A reopening reason can also explicitly resume an existing pending evidence task.
    """
    if budget is not None and (not isinstance(budget, Integral) or isinstance(budget, bool) or budget < 0):
        raise ValueError('Search budget must be a nonnegative integer')
    if (not all((context.input_sha256, context.model_version, context.model_sha256, context.descriptor_sha256))
            or context.engine_identity and (len(context.engine_identity) != len(IDENTITY_COLUMNS)
                                            or not all(context.engine_identity))
            or context.risk_alpha is not None and not context.engine_identity):
        raise ValueError('Search requires a complete analysis identity')
    if not predictions.index.is_unique or not predictions.index.equals(physical_rows.index):
        raise ValueError('Search requires aligned batch records and physical rows')
    keys = predictions['record_id'].tolist()
    if any(not isinstance(rid, str) or not rid for rid in keys) or len(set(keys)) != len(keys):
        raise ValueError('Search requires unique stable record identities')
    if not scope.pending_release_ids <= set(keys):
        raise ValueError('Release scope contains records outside the analysis batch')
    if scope.event_id is None:
        if scope.event_conclusion is not None or scope.representative_ids or scope.authorization_ids:
            raise ValueError('Event search requires an event identifier')
        members = set()
    else:
        if scope.event_conclusion not in (NORMAL, ANOMALY) or 'ow_event' not in predictions:
            raise ValueError('Event search requires the proposed conclusion and event membership')
        members = set(predictions.loc[predictions['ow_event'].eq(scope.event_id), 'record_id'])
        if not members or not (scope.representative_ids | scope.authorization_ids) <= members:
            raise ValueError('Event scope contains records outside the selected event')

    reviews, evidence, required = reviews or {}, evidence or {}, required or {}
    prior_checks, reopen = prior_checks or {}, reopen or {}
    pool = members | scope.pending_release_ids
    selected = predictions['record_id'].isin(pool)
    batch = predictions.loc[selected].to_dict('records')
    rows = zip(batch, physical_rows.loc[selected])
    diagnostics = _batch_diagnostics(predictions.iloc[0] if len(predictions) else None, context)
    evidence_by_source = {item.source: item for item in diagnostics}
    records, pending, retained, counterexamples, supporting = {}, [], [], [], []
    representative_shapes, covered_shapes = {}, {}
    event_question = (scope.event_id, scope.event_conclusion, scope.representative_ids, scope.authorization_ids)

    for row, physical_value in rows:
        rid = row['record_id']
        physical = _integer(physical_value, 2)
        queue = _queue(row, context)
        event = rid in members
        release = rid in scope.pending_release_ids and queue in ('', None)
        paths = tuple(path for path, included in (('event', event), ('release', release)) if included)
        if not paths:
            continue
        bound = _engine_matches(row, context) and _text(row.get('model_version')) == context.model_version
        prior = prior_checks.get(rid)
        question = event_question if event else None
        if not bound or physical is None or release and queue is None:
            basis = CheckBasis(context, paths, question, ())
            reason = '记录或分析身份待核对' if not bound or physical is None else '原生队列状态待核对'
            pending.append(Candidate(rid, paths, queue, physical, None, basis, (reason,), prior_check=prior))
            continue
        findings, directions, decision = _prediction_findings(row, context)
        findings += tuple(Finding('prediction_direction', name, 'available',
                                  f'{name} 指向{DIRECTION_NAMES[direction]}', direction)
                          for name, direction in directions.items())
        review = review_with_evidence(reviews[rid]) if rid in reviews else {}
        snapshot = review.get('evidence_snapshot')
        review_id, conclusion, note = review.get('id'), review.get('status'), review.get('note')
        payload = evidence.get(rid)
        if not context.engine_identity or not _snapshot_matches(payload, row, physical, context):
            payload = None
        shapes, tails, references, responses = _event_shapes(directions, payload) if event else ({}, (), (), ())
        if event:
            category = 'event_heterogeneity'
            for source, value in (('ecod_tails', tails), ('training_reference', references), ('reference_replacement', responses)):
                findings += (Finding(category, source, 'available', source, value) if value else
                             _unknown(category, source),)
        if rid in scope.representative_ids:
            for source, shape in shapes.items():
                representative_shapes.setdefault(source, set()).add(shape)
                covered_shapes.setdefault(source, set()).add(shape)
        rank = _integer(row.get('engine_rank'), 1) if context.engine_identity else None
        if conclusion in (NORMAL, ANOMALY):
            if (snapshot and 'event_review' not in snapshot and _snapshot_matches(snapshot, row, physical, context)
                    and _integer(review_id, 1) is not None and isinstance(note, str) and note.strip()):
                questions = {'event': scope.event_conclusion, 'release': NORMAL}
                against = tuple(path for path in paths if conclusion != questions[path])
                supports = tuple(path for path in paths if conclusion == questions[path])
                human = HumanReview(rid, review_id, conclusion, note, context, against, supports,
                                    'release' in against or ('event' in against and
                                    (rid in scope.authorization_ids or rid in scope.representative_ids)))
                if against:
                    counterexamples.append(human)
                if supports:
                    supporting.append(human)
                if event:
                    for source, shape in shapes.items():
                        covered_shapes.setdefault(source, set()).add(shape)
                continue
            if snapshot is None or 'event_review' not in snapshot:
                basis = CheckBasis(context, paths, question, findings)
                pending.append(Candidate(rid, paths, queue, physical, rank, basis,
                                         ('已有确认复核，快照或适用上下文待核对',), prior_check=prior))
                continue
            if not _snapshot_matches(snapshot, row, physical, context):
                basis = CheckBasis(context, paths, question, findings)
                pending.append(Candidate(rid, paths, queue, physical, rank, basis,
                                         ('已有继承结论，适用上下文待核对',), prior_check=prior))
                continue
            findings += (Finding('review_provenance', 'event_review', 'available', '已有代表复核继承记录',
                                 review_id),)
        records[rid] = (paths, queue, physical, rank, findings, directions, decision, shapes, conclusion, prior)
        if (event and prior and prior.disposition == 'checked' and prior.basis.context == context
                and prior.basis.evidence[:len(findings)] == findings):
            for source, shape in shapes.items():
                covered_shapes.setdefault(source, set()).add(shape)

    ranked = []
    for rid, (paths, queue, physical, rank, findings, directions, decision, shapes, conclusion, prior) in records.items():
        event, release = 'event' in paths, 'release' in paths
        differences = tuple(source for source, shape in shapes.items()
                            if source in covered_shapes and shape not in covered_shapes[source])
        representative_differences = tuple(source for source, shape in shapes.items()
                                           if source in representative_shapes and shape not in representative_shapes[source])
        if event:
            category = 'event_heterogeneity'
            if representative_differences:
                findings += (Finding(category, 'representative_comparison', 'signal',
                                     '与代表可核验的差异：' + '、'.join(representative_differences),
                                     representative_differences),)
            if differences:
                findings += (Finding(category, 'member_comparison', 'signal',
                                     '相同事件内未覆盖的形态：' + '、'.join(differences), differences),)
            else:
                comparable = any(source in covered_shapes for source in shapes)
                findings += (Finding(category, 'member_comparison', 'available', '有可对照的同类证据')
                             if comparable else _unknown(category, 'member_comparison'),)
        available = {**evidence_by_source, **{item.source: item for item in findings}}
        requirements = tuple(sorted(set(required.get(rid, ()))))
        missing = tuple(key for key in requirements
                        if key not in available or available[key].state not in ('available', 'signal'))
        if requirements:
            findings += tuple(Finding('authorization', key, 'unknown' if key in missing else 'available',
                                      f'授权所需证据：{key}', available.get(key)) for key in requirements)
        basis = CheckBasis(context, paths, event_question if event else None, findings)
        reasons = []
        if event and rid in scope.authorization_ids and decision is not None and decision != scope.event_conclusion:
            reasons.append(f'成员当前判定指向{DIRECTION_NAMES[decision]}，与拟继承结论{scope.event_conclusion}相反')
        if release:
            anomalous = [name for name, direction in directions.items() if direction == ANOMALY]
            if anomalous:
                reasons.append('待放行记录存在异常方向线索：' + '、'.join(anomalous))
        if missing:
            reasons.append('授权关键证据缺失、失效或不适用：' + '、'.join(missing))
        signals = [item.reason for item in findings if item.state == 'signal']
        priority = 1 if reasons else 2 if signals else 3
        reasons.extend(signals)
        if priority == 3:
            reasons.append('覆盖核查：当前未见反向、冲突、不确定或异质线索')
        if event:
            reasons.append('完整事件成员范围')
        if release:
            reasons.append('队列外待放行核查，进入本记录的复核或放行流程')
        changed = prior is not None and prior.basis.recheck_key != basis.recheck_key
        if prior and not (reopen.get(rid) and (changed or prior.disposition == 'pending_evidence')):
            destination = pending if prior.disposition == 'pending_evidence' else retained
            destination.append(Candidate(rid, paths, queue, physical, rank, basis, (prior.reason,), prior_check=prior))
        elif not prior and conclusion == '待补证' and not reopen.get(rid):
            pending.append(Candidate(rid, paths, queue, physical, rank, basis,
                                     ('续办已有待补证复核',)))
        else:
            if reopen.get(rid):
                reasons.append(reopen[rid])
            ranked.append(Candidate(rid, paths, queue, physical, rank, basis, tuple(reasons),
                                    priority, bool(differences), prior))

    ranks = {item.engine_rank for item in ranked}
    rank_used = bool(ranked) and None not in ranks and len(ranks) == len(ranked)
    ranked.sort(key=lambda item: (item.priority, not item.uncovered_form,
                                 item.engine_rank if rank_used else 0, item.physical_row, item.record_id))
    for group in (pending, retained):
        group.sort(key=lambda item: (item.physical_row is None, item.physical_row, item.record_id))
    for group in (counterexamples, supporting):
        group.sort(key=lambda item: (item.review_id, item.record_id))
    return SearchResult(context, scope, tuple(ranked), budget, tuple(counterexamples), tuple(supporting),
                        tuple(pending), tuple(retained), rank_used, diagnostics)
