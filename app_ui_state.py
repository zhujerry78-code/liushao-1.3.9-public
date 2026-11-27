"""Presentation-only helpers for the local r8 UI revision."""
import hashlib
import json
import math
from numbers import Real
import re

import pandas as pd
from pandas.api.types import is_bool, is_scalar

from app_types import Filters, ExportSnapshot, line_number

from trial_build import BUILD_LABEL
UI_REVISION = BUILD_LABEL
ENGINE_STATES = ('正常', '攻击', '模糊', '新异')
ENGINE_QUEUES = ('模糊', '新异', '风险控制')
QUEUE_SCOPES = {name + '队列': name for name in ENGINE_QUEUES}


def engine_diagnostic_lines(record):
    """Translate saved/current B2 and B3 fields without inferring a guarantee."""
    def value(name):
        item = record.get(name)
        return None if item is None or pd.isna(item) else str(item)

    lines = []
    if 'b2_mode' in record:
        mode = {'weighted': '加权', 'unweighted': '非加权', 'fallback': '非加权回退'}.get(value('b2_mode'), '—')
        reasons = []
        reason_codes = value('b2_fallback_reason')
        checks = {'count': '校准样本数不足', 'ess': '有效样本量不足',
                  'resolution': '加权 p 值分辨率不足', 'weights': '权重无效',
                  'weight_sum': '权重总量无效'}
        for code in reason_codes.split(',') if reason_codes else []:
            code = code.strip()
            if code == 'weight_fit':
                reasons.append('密度比权重估计失败')
                continue
            kind, _, check = code.partition(':')
            label = {'normal': '正常类', 'attack': '攻击类'}.get(kind)
            reasons.append(label + checks[check] if label and check in checks else '其他筛查条件未通过')
        lines.append('B2 共形 · ' + mode + ('（' + '；'.join(reasons) + '）' if reasons else ''))
    if 'b3_calculation_status' in record:
        status = {'passes': '数值核验通过（不等于统计保证成立）', 'fails': '数值核验未通过',
                  'unavailable': '数值核验不可用'}.get(value('b3_calculation_status'), '—')
        lines.append('B3 数值状态：' + status)
    if 'b3_statistical_status' in record:
        status = {
            'premises_unverified': '加权风险控制的统计前提未验证。',
            'weight_invalid': '权重无效，不能据此声明加权保证。',
            'not_applicable': '本批未完成漂移检测，统计前提不适用。',
            'exchangeability_assumed': '非加权风险控制以交换性成立为前提。',
        }.get(value('b3_statistical_status'), '—')
        lines.append('B3 统计状态：' + status)
    return lines


def engine_batch_summary(pred):
    if pred.empty:
        return {}, {}, []
    state = pred.get('ow_state')
    queue = pred.get('ow_queue')
    first = pred.iloc[0]
    counts = {name: int(state.eq(name).sum()) for name in ENGINE_STATES} if state is not None else {}
    queues = {name: int(queue.eq(name).sum()) for name in ENGINE_QUEUES} if queue is not None else {}
    lines = []
    drift_delta, drift_flag = first.get('drift_delta'), first.get('drift_flag')
    delta_missing = is_scalar(drift_delta) and pd.isna(drift_delta)
    try:
        delta_finite = isinstance(drift_delta, Real) and not is_bool(drift_delta) and math.isfinite(drift_delta)
    except OverflowError:
        delta_finite = False
    flag_known = is_bool(drift_flag)
    # Existing engine encoding: both columns exist, false flag + missing delta =
    # too small to test. Missing columns/flags do not carry that meaning.
    drift_untested = ({'drift_delta', 'drift_flag'} <= set(pred)
                      and flag_known and not drift_flag and delta_missing)
    if {'drift_delta', 'drift_flag'} & set(pred):
        if drift_untested:
            lines.append('漂移检测：记录太少，未检测 · δ —')
        elif delta_finite and flag_known:
            result = '本批与校准数据的分布有差异' if drift_flag else '本批未报告与校准数据的分布差异'
            lines.append(f'漂移检测：{result} · δ {drift_delta:.3f}')
        else:
            delta_label = f'{drift_delta:.3f}' if delta_finite else '—'
            lines.append('漂移检测：诊断不完整，无法判断 · δ ' + delta_label)
    lines.extend(engine_diagnostic_lines(first))
    if {'guarantee_status', 'risk_alpha'} <= set(pred):
        status, alpha = first.guarantee_status, first.risk_alpha
        if pd.notna(status) and pd.notna(alpha):
            lines.append(f'保证状态：{status}（α = {alpha:g}）')
            if status in ('有效', '加权有效'):
                lines.append('在相应校准假设成立时，已知攻击被放行的比例不超过 α。未见攻击不在保证范围。')
            elif status == '失效':
                lines.append('本批不在已知攻击放行风险的保证范围内。')
            elif status == '不适用':
                reasons = []
                if drift_untested:
                    reasons.append('小批次未检测漂移')
                if counts:
                    if (counts['模糊'] + counts['新异']) / len(pred) >= .5:
                        reasons.append('模糊与新异合计达到 50%')
                    if not counts['攻击']:
                        reasons.append('未见引擎判为攻击的已知攻击证据')
                if reasons:
                    lines.append('不适用原因：' + '，'.join(reasons) + '。')
            ess = first.get('guarantee_ess')
            if pd.notna(ess):
                lines.append(f'加权共形有效样本量（ESS）：{ess:,.1f}')
    return counts, queues, lines

STATUS_NAMES = {'待复核': '未检查', '待补证': '需要补充信息（待补证）',
                '确认异常': '确认异常', '确认正常': '确认正常', '复核不可用': '暂不可用'}
FIELD_NAMES = {
    'Flow Duration': '连接持续时间', 'Flow Byts/s': '每秒传输字节数',
    'Flow Pkts/s': '每秒数据包数', 'Tot Fwd Pkts': '发出数据包数',
    'Tot Bwd Pkts': '返回数据包数', 'TotLen Fwd Pkts': '发出数据总量',
    'TotLen Bwd Pkts': '返回数据总量', 'Fwd Pkts/s': '每秒发出数据包数',
    'Bwd Pkts/s': '每秒返回数据包数', 'Pkt Size Avg': '平均数据包大小',
    'Down/Up Ratio': '收发数量比', 'Fwd Seg Size Avg': '平均发出数据段大小',
    'Bwd Seg Size Avg': '平均返回数据段大小', 'Fwd Byts/b Avg': '发出数据块平均大小',
    'Fwd Pkts/b Avg': '每块平均发出包数', 'Fwd Blk Rate Avg': '平均发出数据块速率',
    'Bwd Byts/b Avg': '返回数据块平均大小', 'Bwd Pkts/b Avg': '每块平均返回包数',
    'Bwd Blk Rate Avg': '平均返回数据块速率', 'Init Fwd Win Byts': '发出方向起始窗口大小',
    'Init Bwd Win Byts': '返回方向起始窗口大小', 'Fwd Act Data Pkts': '含数据的发出包数',
    'Fwd Seg Size Min': '发出数据段最小大小', 'Protocol': '通信协议',
    'Dst Port': '目标端口', 'Fwd Header Len': '发出包头总长度',
    'Bwd Header Len': '返回包头总长度',
}
for prefix, name in [('Fwd Pkt Len', '发出包长度'), ('Bwd Pkt Len', '返回包长度'),
                     ('Pkt Len', '数据包长度'), ('Flow IAT', '数据包间隔'),
                     ('Fwd IAT', '发出包间隔'), ('Bwd IAT', '返回包间隔'),
                     ('Active', '活跃时长'), ('Idle', '空闲时长')]:
    for suffix, meaning in [('Max', '最大'), ('Min', '最小'), ('Mean', '平均'),
                            ('Std', '波动'), ('Tot', '总计'), ('Var', '差异')]:
        FIELD_NAMES[f'{prefix} {suffix}'] = meaning + name
for direction, meaning in [('Fwd', '发出'), ('Bwd', '返回')]:
    for flag in ['PSH', 'URG']:
        FIELD_NAMES[f'{direction} {flag} Flags'] = f'{meaning}{flag}标记数'
    FIELD_NAMES[f'Subflow {direction} Pkts'] = f'分段{meaning}包数'
    FIELD_NAMES[f'Subflow {direction} Byts'] = f'分段{meaning}字节数'
for flag in ['FIN', 'SYN', 'RST', 'PSH', 'ACK', 'URG', 'CWE', 'ECE']:
    flag_name = {'FIN': '连接结束', 'SYN': '连接请求', 'RST': '连接重置', 'PSH': '数据推送',
                 'ACK': '确认接收', 'URG': '紧急数据', 'CWE': '拥塞窗口调整', 'ECE': '拥塞提示'}[flag]
    FIELD_NAMES[f'{flag} Flag Cnt'] = flag_name + '标记数'
FIELD_NAMES['CWE Flag Count'] = '拥塞窗口调整标记数'
for direction, meaning in [('Fwd', '发出'), ('Bwd', '返回')]:
    FIELD_NAMES[f'{direction} PSH Flags'] = meaning + '数据推送标记数'
    FIELD_NAMES[f'{direction} URG Flags'] = meaning + '紧急数据标记数'


def field_name(name):
    return FIELD_NAMES.get(name, '其他数据项')


def literal_markdown(value):
    """Display input names as text, without Markdown formatting or remote embeds."""
    return re.sub(r'([\\`*_{}\[\]()#+.!|<>-])', r'\\\1', str(value))


def safe_csv_text(value):
    if not isinstance(value, str): return value
    at = 0
    while at < len(value) and (value[at].isspace() or ord(value[at]) < 33 or value[at] == '\ufeff'):
        at += 1
    if value.startswith(('\t','\r','\n')) or value[at:].startswith(('=', '+', '-', '@')):
        return "'" + value
    return value


def csv_bytes(frame):
    safe = frame.copy()
    for col in safe.select_dtypes(include=['object', 'string', 'category']).columns:
        # Categories may contain literal filenames/notes, including formula text.
        safe[col] = safe[col].astype(object).map(safe_csv_text)
    return safe.to_csv(index=False).encode('utf-8-sig')


def default_filters():
    return Filters()


def review_candidates(frame):
    if 'ow_queue' in frame:
        return frame.ow_queue.isin(ENGINE_QUEUES)
    if 'engine_review' in frame:
        return frame.engine_review.fillna(False)
    return frame.rf_anomaly.ne(frame.lr_anomaly) | frame.near_threshold


def conclusion_source(record):
    states = {'直接确认': '人工逐条确认', '继承有效': '由事件继承',
              '继承失效待重新确认': '继承失效', '已撤回': '授权已撤回'}
    state = record.get('authorization_state')
    if state in states:
        return states[state]
    if record.get('review_status') == '复核不可用' or state is None:
        return '来源待核验'
    return '待补证' if record.get('review_status') == '待补证' else '尚无有效结论'


def queue_reason(record, config, *, include_lr_near=False):
    reasons = []
    if record.get('authorization_state') == '继承失效待重新确认':
        reasons.append('原授权失效，需重新核查')
    if record.get('review_status') == '待补证':
        reasons.append('上次逐条复核要求补证')
    queue = record.get('ow_queue')
    if queue in ENGINE_QUEUES:
        reasons.append(f'引擎分入{queue}队列')
    if record.get('rf_anomaly') != record.get('lr_anomaly'):
        reasons.append('RF 与 LR 提示不一致')
    if record.get('near_threshold'):
        reasons.append('RF 分数接近分界')
    if include_lr_near and abs(record.lr_score - config['lr_threshold']) <= config['near_margin']:
        reasons.append('LR 分数接近分界')
    if record.get('engine_review') and not reasons:
        reasons.append('引擎标记需复核')
    if not reasons and record.get('rf_anomaly'):
        reasons.append('RF 达到异常提示分界')
    return ' · '.join(reasons) if reasons else '按当前查看范围列出'


def filtered(pred, filters, bundle, *, physical_rows=None):
    """Derive the same ordered membership for the queue, next record and export."""
    from app_quick_search import matching_records
    from app_score_charts import in_score_bin
    table = pred
    if physical_rows is not None:
        table = table.assign(input_csv_row=physical_rows)
    scope = filters['scope']
    priority = review_candidates(table)
    if filters.get('include_lr_near') and scope in ('需要再检查', '待办记录'):
        cfg = bundle['config']
        priority = priority | (table.lr_score - cfg['lr_threshold']).abs().le(cfg['near_margin'])
    if scope in ('需要再检查', '待办记录'):
        if scope == '待办记录':
            open_status = table.review_status.isin(('待复核', '待补证'))
            invalid = table.get('authorization_state', pd.Series('', index=table.index)).eq('继承失效待重新确认')
            priority = (priority & open_status) | invalid | table.review_status.eq('待补证')
        table = table[priority]
    elif scope in QUEUE_SCOPES:
        table = table[table.ow_queue.eq(QUEUE_SCOPES[scope])] if 'ow_queue' in table else table.iloc[:0]
    elif scope in ('异常提示', '未提示异常'):
        table = table[table.rf_anomaly.eq(scope == '异常提示')]
    if filters['status'] != '全部':
        table = table[table.review_status.eq(filters['status'])]
    table = table[table.rf_score.ge(filters['minimum'])]
    if filters['disagreement']:
        table = table[table.rf_anomaly.ne(table.lr_anomaly)]
    if filters['near']:
        table = table[table.near_threshold]
    if filters.get('score_bin') is not None:
        table = table[in_score_bin(table.rf_score, filters['score_bin'])]
    if filters['line'].strip():
        number = line_number(filters['line'])
        field = 'input_csv_row' if filters.get('line_kind') == '输入 CSV 行号' else 'source_row'
        table = table[table[field].eq(number)] if number is not None and field in table else table.iloc[:0]
    if filters.get('keyword', '').strip():
        table = table[matching_records(table, filters['keyword'])]
    sort = filters['queue']
    if sort == '高异常得分':
        cols = ['rf_score', 'lr_score', 'record_id'] if filters['lr_tie_break'] else ['rf_score', 'record_id']
        return table.sort_values(cols, ascending=[c == 'record_id' for c in cols], kind='stable')
    if sort == '待核实':
        first = review_candidates(table)
        queued, rest = table[first], table[~first]
        if 'engine_rank' in table:
            queued = queued.sort_values(['engine_rank', 'record_id'], kind='stable')
        elif len(queued):
            queued = queued.assign(_distance=(queued.rf_score-bundle['config']['rf_threshold']).abs())
            queued = queued.sort_values(['_distance', 'record_id'], kind='stable').drop(columns='_distance')
        return pd.concat([queued, rest])
    return table


def dirty(editor):
    return bool(editor and (editor['status'], editor['note']) != tuple(editor['original']))


def base_meta(active, result, bundle, verified):
    pred, rejected = result.predictions, result.rejected
    from app_engine_evidence import IDENTITY_COLUMNS
    identities = ({name: str(pred.iloc[0][name]) for name in IDENTITY_COLUMNS}
                  if len(pred) and set(IDENTITY_COLUMNS).issubset(pred.columns) else None)
    return dict(model_version=bundle['model_version'], data_version=bundle['config']['data_version'],
        model_sha256=verified.model_sha256,
        descriptor_sha256=verified.descriptor_sha256,
        input_sha256=active['digest'], input_name=active['name'], input_source=active['source'],
        valid_rows=len(pred), unjudgeable_rows=len(rejected), rf_threshold=bundle['config']['rf_threshold'],
        lr_threshold=bundle['config']['lr_threshold'], near_margin=bundle['config']['near_margin'],
        ui_revision=UI_REVISION, engine_identity=identities)


def export_snapshot(active, pred, audit, meta, available, *, features=None, physical_rows=None, bundle=None):
    from liushao.reviews import review_with_evidence
    from app_record_tools import confirmed_frame, export_physical_rows
    physical_rows = export_physical_rows(pred, physical_rows)
    # Read only persisted explanations; an export must never recreate the past
    # using today's model, input, or explanation implementation.
    audit = [review_with_evidence(row) for row in audit]
    meta = dict(meta, review_storage_available=available, has_unsaved_review=False,
                saved_review_records=len({r['record_id'] for r in audit}) if available else None,
                review_history_entries=len(audit) if available else None,
                review_evidence_schema='review-evidence-2',
                review_evidence_saved=sum(row['evidence_snapshot_status'] == 'saved' for row in audit) if available else None,
                review_evidence_missing=sum(row['evidence_snapshot_status'] == 'missing' for row in audit) if available else None,
                review_evidence_invalid=sum(row['evidence_snapshot_status'] == 'invalid' for row in audit) if available else None)
    confirmed, missing = None, 0
    if available:
        confirmed, missing = confirmed_frame(active, pred, audit, meta, features, physical_rows, bundle, field_name)
    exported=pred.copy()
    if physical_rows is not None:
        exported['input_csv_row'] = physical_rows
    else:
        exported['input_csv_row']=pd.Series(pd.NA,index=pred.index,dtype='Int64')
    return ExportSnapshot(input_key=active['key'], meta=meta,
                predictions=csv_bytes(exported) if available else None,
                history=json.dumps(audit, ensure_ascii=False, indent=2).encode('utf-8') if available else None,
                manifest=json.dumps(meta, ensure_ascii=False, indent=2).encode('utf-8'),
                confirmed=csv_bytes(confirmed) if confirmed is not None else None,
                confirmed_count=len(confirmed) if confirmed is not None else 0, missing_input_rows=missing)
