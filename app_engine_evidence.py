"""Versioned, serializable engine evidence; never changes frozen model parameters.

The ECOD score chooses one *aggregate* tail branch.  Its additive explanation
must use that same branch, not a separate maximum for each feature.

Schema 2 (1.3.8) adds engine_format and an open_world block (four-way state,
Bel(U), drift delta, alpha, guarantee status, novelty cluster).  Every schema 1
field keeps its name and meaning; values the engine did not produce are None.
Saved schema 1 blocks are displayed as stored and never upgraded.
"""
import hashlib
import json
import math
from contextlib import closing
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from app_engine_inference import EFFECTIVE_BATCH_COLUMN

ENGINE_EVIDENCE_VERSION = 'engine-evidence-v1'
ENGINE_EVIDENCE_SCHEMA = 2
IDENTITY_COLUMNS = ('engine_sha256', 'engine_descriptor_sha256', 'engine_inference_version')


def engine_identity(record):
    return tuple(str(record.get(name, '')) for name in IDENTITY_COLUMNS)


def ecod_contributions(features_row, engine):
    from liushao_engine.ecod import tail_logs
    if len(features_row) != 1 or not features_row.columns.is_unique:
        raise ValueError('ECOD evidence requires one row with unique fields')
    left, right = tail_logs(engine['ecod'], features_row.to_numpy(dtype=float))
    left, right = left[0], right[0]
    use_left = np.asarray(engine['ecod']['skew_sign']) < 0
    branches = [left, right, np.where(use_left, left, right)]
    totals = np.array([values.sum() for values in branches])
    selected = int(np.argmax(totals))  # deterministic tie: left, right, auto
    values = branches[selected]
    directions = (np.ones(len(values), dtype=bool) if selected == 0 else
                  np.zeros(len(values), dtype=bool) if selected == 1 else use_left)
    fields = [{'field': str(name), 'contribution': float(value),
               'direction': 'left' if directions[i] else 'right',
               'input_missing': bool(not np.isfinite(features_row.iloc[0, i]))}
              for i, (name, value) in enumerate(zip(features_row.columns, values))]
    score = float(totals[selected])
    reconstruction = math.fsum(item['contribution'] for item in fields)
    return {'method': 'winning-aggregate-tail-additive-v1',
            'selected_branch': ('left', 'right', 'auto')[selected],
            'branch_totals': dict(zip(('left', 'right', 'auto'), map(float, totals))),
            'score': score, 'contribution_sum': reconstruction,
            'residual': score - reconstruction, 'fields': fields,
            'top_fields': sorted(fields, key=lambda item: -item['contribution'])[:6]}


def _plain(value):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return None if not math.isfinite(float(value)) else float(value)
    return str(value)


def open_world_fields(record):
    """Schema 2 open-world block, read only from columns the engine actually wrote."""
    from liushao_engine import OPEN_WORLD_FIELDS
    out = {}
    for field, column in OPEN_WORLD_FIELDS.items():
        value = record[column] if column in record else None
        try:  # pd.isna also covers pd.NA, which nullable integer columns (ow_event, novelty_cluster) hold
            missing = value is None or bool(np.ndim(value) == 0 and pd.isna(value))
        except Exception:
            missing = False
        out[field] = None if missing else _plain(value)
    from liushao_engine import B2_COLUMNS, B3_COLUMNS
    for field in B2_COLUMNS + B3_COLUMNS:
        if field in record:
            out[field] = None if pd.isna(record[field]) else _plain(record[field])
    return out


def batch_evidence_reference(frame, features, bundle, engine):
    """Verify a displayed batch at its active alpha using the same inference path as analysis."""
    if not frame.index.equals(features.index):
        raise ValueError('分析结果与特征行未对齐')
    alpha = None
    if 'risk_alpha' in frame:
        if frame.risk_alpha.nunique(dropna=False) != 1:
            raise ValueError('同一批次的 risk_alpha 必须一致')
        alpha = float(frame.risk_alpha.iloc[0])
    from app_engine_inference import analyze_current_engine
    reference, used = analyze_current_engine(frame, features, bundle, engine, alpha=alpha, return_used=True)
    if EFFECTIVE_BATCH_COLUMN in reference:
        reference['_engine_used'] = [used] * len(reference)
    return reference


def build_engine_evidence(record, features_row, bundle, engine, *, reference=None):
    """Bind saved output to verified engine bytes and verify all displayed values."""
    if 'cp_set' not in record:
        return {'schema_version': ENGINE_EVIDENCE_SCHEMA, 'method_version': ENGINE_EVIDENCE_VERSION,
                'status': 'not_used', 'reason': '本次分析未使用研判引擎，仅有 RF/LR 结果。'}
    import liushao_engine
    if engine is None:
        raise ValueError('本次分析使用过研判引擎，但当前参数无法校验；请恢复配套文件并重新分析。')
    identity = engine.get('_artifact_identity', {})
    if (not all(engine_identity(record)) or
            engine_identity(record) != tuple(identity.get(name) for name in IDENTITY_COLUMNS)):
        raise ValueError('当前引擎与这次分析的引擎身份不一致，请重新检查文件后保存。')
    if list(features_row.columns) != list(bundle['config']['features']):
        raise ValueError('Engine evidence feature order does not match frozen model')
    if len(features_row) != 1 or features_row.index[0] != record.name:
        raise ValueError('Engine evidence record and feature row are not aligned')
    if reference is None:
        reference = batch_evidence_reference(record.to_frame().T, features_row, bundle, engine).iloc[0]
    if reference.name != record.name or reference.get('record_id') != record.get('record_id'):
        raise ValueError('证据快照与批次记录未对齐')
    provenance = record.get(EFFECTIVE_BATCH_COLUMN)
    if provenance != reference.get(EFFECTIVE_BATCH_COLUMN):
        raise ValueError('显示结果与证据快照的有效校准或批次不一致。')
    used = engine
    if provenance is not None:
        used = reference.get('_engine_used')
        if used is None:
            raise ValueError('折扣后证据需要完整批次的有效引擎。')
        if any(record[name] != reference[name] for name in ('rf_score', 'lr_score', 'typ_score')):
            raise ValueError('显示结果与证据快照的专家分数不一致。')
    fields, expected_fields = open_world_fields(record), open_world_fields(reference)
    if fields != expected_fields:
        changed = [k for k in sorted(fields.keys() | expected_fields.keys())
                   if k not in fields or k not in expected_fields or fields[k] != expected_fields[k]]
        raise ValueError('Engine evidence result mismatch: ' + ', '.join(
            liushao_engine.OPEN_WORLD_FIELDS.get(k, k) for k in changed))
    expected = reference
    values = {}
    for name in liushao_engine.COLUMNS:
        value, check = record[name], expected[name]
        if name in ('cp_set', 'engine_version'):
            if str(value) != str(check):
                raise ValueError('Engine evidence result mismatch: ' + name)
            values[name] = str(value)
        elif name in ('cp_set_size', 'engine_review'):
            if value != check:
                raise ValueError('Engine evidence result mismatch: ' + name)
            values[name] = bool(value) if name == 'engine_review' else int(value)
        else:
            if not math.isclose(float(value), float(check), rel_tol=1e-10, abs_tol=1e-10):
                raise ValueError('Engine evidence result mismatch: ' + name)
            values[name] = float(value)
    decomposition = ecod_contributions(features_row, engine)
    if not math.isclose(decomposition['contribution_sum'], values['ecod_score'], rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError('ECOD contributions do not reconstruct recorded score')
    return {'schema_version': ENGINE_EVIDENCE_SCHEMA, 'method_version': ENGINE_EVIDENCE_VERSION,
            'status': 'verified', 'engine_format': int(engine.get('format', 1)),
            'identity': dict(identity), 'results': values, 'open_world': fields,
            'parameters': {'epsilon': float(used['conformal']['epsilon']),
                           'conflict_cut': float(used['conflict_cut']),
                           'va_method': 'exact-isotonic-two-label-refits',
                           'va_calibration_source': ('validation-B1-class-scores' if 'venn_abers_calibration' in engine
                                                     else 'frozen-conformal-B-class-scores'),
                           'evidence_frame': engine['fusion'].get('frame', 'AN'),
                           'conformal_scores': engine['conformal'].get('scores', 'betp'),
                           'calibrator_kind': engine.get('calibrator_kind', 'step'),
                           'benchmark_protocol_sha256': engine.get('benchmark_protocol_sha256')},
            'source_masses': liushao_engine.source_masses(record, used).to_dict(orient='records'),
            **({'effective_batch': json.loads(provenance)} if provenance is not None else {}),
            'ecod_decomposition': decomposition,
            'limitations': ['保存的是当时的模型输出及其计算依据，不是人工确认的真实标签。',
                            'ECOD 贡献解释原始异常分数，不是对最终攻击结论的因果解释。']}


def require_complete_engine_evidence(payload):
    if payload.get('engine', {}).get('status') == 'unavailable':
        raise ValueError(payload['engine']['reason'])
    return payload


def historical_engine_label(payload, current_identity=None):
    """Read historical identity only; never reconstruct a missing old snapshot."""
    block = payload.get('engine') if isinstance(payload, dict) else None
    if block is None or isinstance(block, dict) and not block:
        return '旧快照未包含研判引擎证据；不能据此还原当时的引擎结论。'
    if not isinstance(block, dict):
        return '保存时的研判引擎证据不完整，请查看原始快照说明。'
    if block.get('status') == 'not_used':
        return '保存时仅使用 RF/LR，未使用研判引擎。'
    if block.get('status') != 'verified':
        return '保存时的研判引擎证据不完整，请查看原始快照说明。'
    identity = block.get('identity')
    if (not isinstance(identity, dict) or any(not isinstance(identity.get(name), str) or not identity[name].strip()
                                               for name in ('engine_version', 'engine_sha256'))):
        return '保存时的研判引擎身份记录不完整；以下仅展示当时保存的证据。'
    message = '保存时引擎：' + identity['engine_version'] + ' · SHA256 ' + identity['engine_sha256'][:12]
    if current_identity is not None and tuple(identity.get(k) for k in IDENTITY_COLUMNS) != tuple(current_identity):
        message += '；与当前分析引擎不同，以下仅展示当时保存的证据。'
    else:
        message += '；以下仅展示当时保存的证据。'
    return message


def event_members_and_representatives(frame, event_id, reps):
    from liushao_engine import EVENT_SEED

    group = frame.loc[frame.ow_event.eq(event_id), ['record_id', 'engine_rank']]
    ids = group.sort_values('engine_rank', kind='stable').record_id.tolist()
    if not ids or reps < 1:
        return ids, []
    pick = np.random.default_rng([EVENT_SEED, event_id]).choice(len(ids), min(reps, len(ids)), replace=False)
    return ids, [ids[i] for i in pick]


@dataclass
class EventReviewState:
    status: str | None
    disagreement: bool
    targets: list[str]


def event_review_state(members, representatives, reviews):
    """An event can inherit only directly confirmed representative judgments."""
    choices = []
    for rid in representatives:
        review = reviews.get(rid, {})
        snapshot = review.get('evidence_snapshot')
        if (snapshot is not None and 'event_review' not in snapshot
                and review.get('status') in ('确认正常', '确认异常')):
            choices.append(review['status'])
    consensus = set(choices)
    disagreement = len(consensus) > 1
    status = choices[0] if len(consensus) == 1 and len(choices) == len(representatives) else None
    targets = [rid for rid in members if rid not in reviews]
    return EventReviewState(status, disagreement, targets)


def event_signature_fields(engine, features_row):
    # Match the frozen event key, including stable feature order on ties.
    from liushao_engine.ecod import tail_logs

    left, right = tail_logs(engine['ecod'], features_row.to_numpy(dtype=np.float64))
    top = np.argsort(-np.maximum(left, right)[0], kind='stable')[:engine['events']['k']]
    return features_row.columns[np.sort(top)].tolist()


def event_review_payload(record, physical_row, meta, event_id, representative_reviews, *,
                         event_size, signature_fields):
    """Save displayed row values and provenance for an inherited judgment."""
    from liushao_engine import COLUMNS
    provenance = record.get(EFFECTIVE_BATCH_COLUMN)
    return dict(schema_version=2, method_version='o1-event-review-v1',
                record_id=record.record_id, model_version=meta['model_version'],
                input_name=meta['input_name'], input_csv_row=int(physical_row),
                **{key: meta[key] for key in ('input_sha256', 'model_sha256', 'descriptor_sha256', 'ui_revision')},
                identity=dict(source_file=str(record.source_file), source_row=int(record.source_row),
                              source_sha256=str(record.source_sha256)),
                scores=dict(rf=float(record.rf_score), lr=float(record.lr_score), rf_anomaly=bool(record.rf_anomaly),
                            rf_threshold=meta['rf_threshold'], lr_threshold=meta['lr_threshold']),
                cards=[], selection={'status': 'event_review'}, decomposition={'status': 'event_review'},
                assessment={'status': '由代表复核结论一致后整簇确认；本条未单独复核。'},
                event_review=dict(mode='representative_consensus', event_id=event_id,
                                  event_size=event_size, signature_fields=signature_fields,
                                  representative_review_ids=[item['id'] for item in representative_reviews],
                                  engine_identity=dict(zip(IDENTITY_COLUMNS, engine_identity(record))),
                                  open_world=open_world_fields(record),
                                  engine_results={key: _plain(record[key]) for key in COLUMNS},
                                  **({'effective_batch': json.loads(provenance)} if provenance is not None else {})))


def save_event_reviews(path, model_version, status, note, expected_ids, payloads):
    """Append the inherited reviews and evidence in one existing-schema transaction."""
    from liushao.reviews import ReviewConflictError, _snapshot_json, connect

    if status not in ('确认异常', '确认正常') or not note.strip() or len(note) > 4000:
        raise ValueError('整簇确认需要明确的判断与备注。')
    rows = [(p['record_id'], p['schema_version'], _snapshot_json(p, p['record_id'], model_version)) for p in payloads]
    targets = {rid for rid, _, _ in rows}
    if not rows or len(targets) != len(rows) or any(expected_ids.get(rid) != 0 for rid in targets):
        raise ValueError('整簇复核没有新的有效记录。')
    stamp = datetime.now(timezone.utc).isoformat()
    with closing(connect(path)) as db, db:
        db.execute('BEGIN IMMEDIATE')
        for rid, expected in expected_ids.items():
            actual = db.execute('SELECT MAX(id) FROM reviews WHERE model_version=? AND record_id=?',
                                (model_version, rid)).fetchone()[0] or 0
            if actual != expected:
                raise ReviewConflictError('事件内记录的复核已经更新，请重新检查代表结论。')
        for rid, version, serialized in rows:
            review_id = db.execute('INSERT INTO reviews (record_id, model_version, status, note, created_at) VALUES (?, ?, ?, ?, ?)',
                                   (rid, model_version, status, note, stamp)).lastrowid
            digest = hashlib.sha256(serialized.encode('utf-8')).hexdigest()
            db.execute('INSERT INTO review_evidence (review_id, schema_version, payload_json, payload_sha256) VALUES (?, ?, ?, ?)',
                       (review_id, version, serialized, digest))
