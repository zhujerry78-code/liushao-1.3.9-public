"""Application-only review deliverables and descriptive statistics."""
from datetime import datetime, timezone
import math
import json
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from liushao.reviews import review_with_evidence

CONFIRMED_COLUMNS = ['输入文件', '输入CSV行号', '来源文件', '来源行号', '保存时系统提示', '我的判断', '备注',
    '复核记录ID', '复核保存时间UTC', '保存时证据摘要', '结论单生成时间UTC', '记录ID', '模型版本',
    '保存时输入SHA256', '保存时模型SHA256', '保存时描述SHA256', '证据快照状态', '证据快照SHA256', '解释方法版本', '证据快照JSON']

def format_review_time(value, *, beijing=False):
    """Display an aware ISO timestamp without pandas' nanosecond year limits.

    An extreme offset can put the converted year outside datetime's range.
    Keep that value explicitly labelled in its original timezone; never claim
    the unconverted text is UTC or overwrite the persisted timestamp.
    """
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None:
        raise ValueError('已保存操作时间缺少时区。')
    target = ZoneInfo('Asia/Shanghai') if beijing else timezone.utc
    try:
        converted = stamp.astimezone(target)
    except OverflowError:
        label = '北京时间' if beijing else ' UTC'
        return f'无法换算为{label}；原始时间 {value}'
    if beijing:
        return (f'{converted.year:04d}-{converted.month:02d}-{converted.day:02d} '
                f'{converted.hour:02d}:{converted.minute:02d}:{converted.second:02d}')
    return converted.isoformat()

def latest_reviews(audit, model_version):
    latest = {}
    for row in audit:
        previous = latest.get(row['record_id'])
        if row['model_version'] == model_version and (previous is None or row['id'] > previous['id']):
            latest[row['record_id']] = row
    return latest

def batch_positions(features, row_index):
    if not features.index.is_unique or row_index not in features.index:
        raise ValueError('本批记录索引不一致，请重新检查文件。')
    result = {}
    for field in features:
        values = features[field].to_numpy(dtype=float)
        valid = values[np.isfinite(values)]
        value = float(features.at[row_index, field])
        percentile = (100 * (np.count_nonzero(valid < value) + .5 * np.count_nonzero(valid == value)) / len(valid)) if np.isfinite(value) and len(valid) else None
        result[field] = (percentile, len(valid))
    return result

def evidence_summaries(predictions, features, bundle, field_name):
    """Same feature order/reference tests as evidence_for; at most three messages."""
    if not predictions.index.is_unique or not features.index.equals(predictions.index):
        raise ValueError('证据与记录索引不一致，请重新检查文件。')
    count = len(predictions)
    parts = [[] for _ in range(count)]
    engine = 'engine_review' in predictions
    for i, (disagreement, near) in enumerate(zip(predictions.disagreement, predictions.near_threshold)):
        if engine and predictions.cp_set.iat[i] == '不确定': parts[i].append('研判引擎无法给出确定结论，建议再检查。')
        elif engine and predictions.engine_review.iat[i]: parts[i].append('多源证据冲突较高，建议再检查。')
        elif disagreement: parts[i].append('两种判断不一致，建议再检查。')
        elif near and not engine: parts[i].append('接近判断分界，建议再检查。')
    for field in bundle['config']['features']:
        values = features[field].to_numpy(dtype=float)
        missing = np.isnan(values)
        reference = bundle['reference'][field]
        low, high = reference['q01'], reference['q99']
        outside = ((values < low) | (values > high)) if low is not None and high is not None else np.zeros(count, bool)
        for i in np.flatnonzero(missing | outside):
            if len(parts[i]) < 3:
                parts[i].append(field_name(field) + ('：数据缺失，已补齐后分析。' if missing[i] else '：与参考数据差异较大。'))
    return ['；'.join(items) if items else '两种判断一致，可按需查看详细数据。' for items in parts]


def valid_saved_event_review(event):
    """Whether saved event metadata can support a consensus presentation."""
    if not isinstance(event, dict):
        return False
    event_id, refs = event.get('event_id'), event.get('representative_review_ids')
    # Real event IDs are integers; preserve older textual IDs as literal labels.
    known_id = ((type(event_id) is int and event_id >= 0)
                or (isinstance(event_id, str) and bool(event_id.strip())))
    return (event.get('mode') == 'representative_consensus' and known_id
            and isinstance(refs, list) and bool(refs)
            and all(type(value) is int and value > 0 for value in refs)
            and len(set(refs)) == len(refs))


def saved_engine_summary(snapshot):
    """Describe saved batch context, including inherited event snapshots.

    Optional historical fields may be incomplete. Never borrow current scores,
    invent a queue/rejection reason, or turn a batch guarantee into a row label.
    """
    event = snapshot.get('event_review')
    if isinstance(event, dict) and valid_saved_event_review(event):
        if 'open_world' not in event:
            return ['旧快照未记录引擎四态与批次保证状态。']
        world = event.get('open_world')
    else:
        engine = snapshot.get('engine')
        if engine is None:
            return ['旧快照未记录引擎四态与批次保证状态。']
        if isinstance(engine, dict) and engine.get('status') == 'not_used':
            return ['保存时仅使用 RF/LR，未使用研判引擎。']
        if not isinstance(engine, dict) or engine.get('status') != 'verified':
            return ['保存时引擎信息不完整，原值见操作记录 JSON。']
        if 'open_world' not in engine:
            return ['旧快照未记录引擎四态与批次保证状态。']
        world = engine.get('open_world')
    if not isinstance(world, dict):
        return ['保存时引擎信息不完整，原值见操作记录 JSON。']

    def finite(value, low, high):
        try:
            return type(value) in (int, float) and math.isfinite(value) and low <= value <= high
        except OverflowError:
            return False

    lines, incomplete = [], False
    state = world.get('state')
    if isinstance(state, str) and state in ('正常', '攻击', '模糊', '新异'):
        lines.append('保存时引擎四态：' + state)
    else:
        incomplete = True
    status, alpha = world.get('guarantee_status'), world.get('alpha')
    if (isinstance(status, str) and status in ('有效', '加权有效', '失效', '不适用')
            and finite(alpha, 0., 1.) and 0 < alpha < 1):
        lines.append(f'保存时批次保证状态：{status}（α = {alpha:g}）')
        if status in ('有效', '加权有效'):
            lines.append('在相应校准假设成立时，已知攻击被放行的比例不超过 α；未见攻击不在保证范围，本条不具有单条安全保证。')
        elif status == '失效':
            lines.append('本批不在已知攻击放行风险的保证范围内。')
        else:
            lines.append('保存时批次的风险保证不适用。')
    else:
        incomplete = True
    delta = world.get('drift_delta')
    if delta is not None:
        if finite(delta, 0., 1.):
            lines.append(f'保存时批次漂移强度 δ = {delta:g}（不代表攻击概率）')
        else:
            incomplete = True
    # Only scalar strings enter the shared presentation helper; corrupt optional
    # containers remain in the raw JSON and cannot break a confirmation export.
    fields = ('b2_mode', 'b2_fallback_reason', 'b3_calculation_status', 'b3_statistical_status')
    diagnostics = {name: world[name] for name in fields
                   if name in world and (world[name] is None or isinstance(world[name], str))}
    if any(name in world and name not in diagnostics for name in fields):
        incomplete = True
    from app_ui_state import engine_diagnostic_lines
    lines.extend(engine_diagnostic_lines(diagnostics))
    if incomplete:
        lines.append('保存时引擎信息不完整，未记录或无效的字段不补写，原值见操作记录 JSON。')
    return lines


def saved_evidence_summary(snapshot):
    """Format only values frozen with the review; never rerun an explanation."""
    scores = snapshot['scores']
    def number(value):
        if value is None:
            return '未记录'
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            try:
                return f'{value:.6g}'
            except OverflowError:
                return '数值过大，原值见操作记录 JSON'
        return str(value)
    parts = []
    if 'event_review' in snapshot:
        event = snapshot['event_review']
        if valid_saved_event_review(event):
            refs = '、'.join(str(value) for value in event['representative_review_ids'])
            parts.append(f"疑似事件 {event['event_id']} 的代表一致确认，依据复核操作 {refs}；本条未单独复核。")
        else:
            parts.append('保存时事件复核信息不完整，原始内容见复核记录。')
    parts += [f"RF 分数 {number(scores.get('rf'))} / 分界 {number(scores.get('rf_threshold'))}",
             f"LR 对照 {number(scores.get('lr'))} / 分界 {number(scores.get('lr_threshold'))}"]
    parts.extend(saved_engine_summary(snapshot))
    for card in snapshot['cards']:
        reference, rf = card.get('reference', {}), card.get('rf', {})
        parts.append(f"{card.get('name', card.get('field', '数据项'))}={number(card.get('value'))} {card.get('unit', '')}；"
                     f"训练正常参考 P01—P99：{number(reference.get('p01'))}—{number(reference.get('p99'))}；"
                     f"单项替换参考中位数后 RF：{number(rf.get('before'))}→{number(rf.get('after'))}")
    assessment = snapshot.get('assessment')
    if isinstance(assessment, dict):
        status, reasons = assessment.get('status', ''), assessment.get('reasons', [])
        if isinstance(status, str):
            parts.append(status)
        if isinstance(reasons, list) and all(isinstance(reason, str) for reason in reasons):
            parts.extend(reasons)
        if not isinstance(status, str) or not isinstance(reasons, list) or any(not isinstance(reason, str) for reason in reasons):
            parts.append('保存时补充说明信息不完整，原始内容见复核记录。')
    parts.append('敏感性试算只解释当前模型响应，不能单独确认攻击。')
    return '；'.join(str(part) for part in parts)

def input_csv_row(value):
    """Return an exact, exportable physical line number, or no locator."""
    if isinstance(value, (bool, np.bool_)):
        return None
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value) or not value.is_integer():
            return None
        value = int(value)
    if isinstance(value, (int, np.integer)) and 2 <= value <= np.iinfo(np.int64).max:
        return int(value)
    return None


def export_physical_rows(pred, physical_rows):
    """Check identity before using index alignment as a row locator."""
    if (not pred.index.is_unique or 'record_id' not in pred
            or not pred.record_id.is_unique or pred.record_id.isna().any()):
        raise ValueError('本批记录标识或索引不一致，不能生成检查结果。')
    if not (isinstance(physical_rows, pd.Series) and physical_rows.index.is_unique
            and physical_rows.index.equals(pred.index)):
        return None
    values = [input_csv_row(value) for value in physical_rows]
    if any(value is None for value in values) or len(set(values)) != len(values):
        raise ValueError('输入 CSV 行号映射无效，不能生成检查结果。')
    return pd.Series(values, index=pred.index, dtype='int64')


def confirmed_frame(active, pred, audit, meta, features, physical_rows, bundle, field_name, generated_at=None):
    latest = latest_reviews(audit, meta['model_version'])
    selected = pred.loc[pred.record_id.map(lambda rid: latest.get(rid, {}).get('status') in ('确认正常', '确认异常'))]
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    rows, missing_rows = [], 0
    for index, row in selected.iterrows():
        review = review_with_evidence(latest[row.record_id])
        snapshot = review['evidence_snapshot']
        identity = snapshot.get('identity', {}) if snapshot else {}
        physical = snapshot.get('input_csv_row') if snapshot else None
        if isinstance(physical, (float, np.floating)) and np.isfinite(physical) and physical.is_integer():
            physical = int(physical)
        if not isinstance(physical, (int, np.integer)) or isinstance(physical, bool) or physical < 2:
            physical = '未记录'; missing_rows += 1
        try:
            stamp = format_review_time(review.get('created_at'))
        except (TypeError, ValueError):
            stamp = '未记录'
        scores = snapshot['scores'] if snapshot else {}
        rf_anomaly = scores['rf'] >= scores['rf_threshold'] if snapshot else None
        # Only verified frozen evidence may supply historical fields. The
        # current input and model must never fill gaps in an old snapshot.
        def recorded(value):
            return '未记录' if value is None or value == '' else value

        frozen_json = (json.dumps(snapshot, ensure_ascii=False, sort_keys=True,
                                  separators=(',', ':'), allow_nan=False) if snapshot else '未记录')
        rows.append([recorded(snapshot.get('input_name') if snapshot else None), physical,
            recorded(identity.get('source_file')), recorded(identity.get('source_row')),
            ('提示异常' if rf_anomaly else '未提示异常') if rf_anomaly is not None else '无当时快照',
            review['status'], recorded(review.get('note')), recorded(review.get('id')), stamp,
            saved_evidence_summary(snapshot) if snapshot else review['evidence_snapshot_label'],
            generated_at, row.record_id, review['model_version'],
            snapshot['input_sha256'] if snapshot else '未记录', snapshot['model_sha256'] if snapshot else '未记录',
            snapshot['descriptor_sha256'] if snapshot else '未记录', review['evidence_snapshot_label'],
            recorded(review['evidence_sha256']), snapshot['method_version'] if snapshot else '未记录', frozen_json])
    return pd.DataFrame(rows, columns=CONFIRMED_COLUMNS), missing_rows
