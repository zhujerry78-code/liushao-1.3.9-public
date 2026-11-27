"""Compact review reasoning and a separate, complete evidence detail view."""
import math
from numbers import Real
from html import escape

import pandas as pd
from pandas.api.types import is_bool, is_scalar
import streamlit as st
from app_ui_state import literal_markdown, field_name, engine_diagnostic_lines
from app_review_guidance import review_guidance, review_check_items

try:
    from app_evidence_rules import classify_attribution_archetype, summarize_interval_bands
    _INTERVAL_RULES_AVAILABLE = True
except ImportError:
    _INTERVAL_RULES_AVAILABLE = False
    def summarize_interval_bands(all_fields):
        return {'high_risk': 0, 'outside_moderate': 0, 'outside_total': 0, 'near_boundary': 0,
                'core_normal': 0, 'insufficient_reference': 0, 'missing': 0, 'discrete_or_fixed': 0,
                'total': len(all_fields or [])}

    def classify_attribution_archetype(payload):
        return {'code': 'explanation_unavailable', 'label': '规则模块版本未对齐（待核查）',
                'headline': '请重新解压完整运行包，再查看评分线索。',
                'bands': summarize_interval_bands((payload or {}).get('all_fields', []))}

try:
    from app_review_guidance import (
        review_focus_rows,
        unified_interval_contribution_rows,
        build_audit_note_draft,
    )
    _CONTRIBUTION_GUIDANCE_AVAILABLE = True
except ImportError:
    _CONTRIBUTION_GUIDANCE_AVAILABLE = False
    def review_focus_rows(payload):
        return [], []

    def unified_interval_contribution_rows(payload, *, top_only=False, limit=12):
        return []

    def build_audit_note_draft(payload):
        return '【客观证据核对】请结合日志核查后填写结论：'


def _fmt(value):
    return '不可用' if value is None else f'{value:,.6g}'


def _short_names(names, limit=2):
    names = list(dict.fromkeys(names))
    return '、'.join(names[:limit]) + (f'等{len(names)}项' if len(names) > limit else '')


def _next_checks(cards, missing=False):
    """Merge the existing field-specific checks without repeating whole actions."""
    actions = list(dict.fromkeys(c.get('action', {}).get('next', '') for c in cards))
    checks, logs = [], []
    for action in actions:
        if '协议、连接建立/重置日志' in action:
            checks.append('协议与连接建立/重置记录'); logs.extend(['连接/服务日志', '维护工单'])
        elif '服务端超时和连接数日志' in action:
            checks.append('邻近连接与超时记录'); logs.extend(['服务日志', '压测/维护工单'])
        elif '代理/DLP/备份日志' in action:
            checks.append('传输方向、目的资产与业务时段'); logs.extend(['代理/DLP/备份日志'])
        elif '同目的端口的短连接与包速率' in action:
            checks.append('同端口连接与包速率'); logs.extend(['防火墙/服务日志', '压测工单'])
    checks = list(dict.fromkeys(checks)); logs = list(dict.fromkeys(logs))
    if '防火墙/服务日志' in logs and '服务日志' in logs:
        logs.remove('服务日志')
    if '压测/维护工单' in logs:
        logs = [item for item in logs if item not in ('压测工单', '维护工单')]
    start = '先核对缺失原值；' if missing else ''
    if checks:
        return start + '核对' + '、'.join(checks) + '，查' + '、'.join(logs) + '；对照业务与授权记录后判断，证据不足选“待补证”。'
    return start + '对齐采集时间与资产，查服务/防火墙/代理日志及授权任务；证据不足选“待补证”。'


def compact_evidence_summary(payload):
    """Build presentation-only prose; never change the persisted evidence payload."""
    selection = payload.get('selection', {})
    scores = payload.get('scores', {})
    rf_anomaly = scores.get('rf_anomaly', scores.get('rf', 0) >= scores.get('rf_threshold', 1))
    lr_anomaly = scores.get('lr_anomaly', scores.get('lr', 0) >= scores.get('lr_threshold', 1))
    cards = payload.get('cards', [])
    facts = []
    for card in cards[:3]:
        value = '原始缺失' if card.get('missing') else _fmt(card.get('value')) + ' ' + card.get('unit', '')
        status = card.get('reference_status', '')
        facts.append(card['name'] + ' ' + value + (('，' + status) if status and status != '原始缺失' else ''))
    alerts = []
    for key, label in (('missing_fields', '原始缺失'), ('insufficient_references', '参考不足'),
                       ('direction_differences', '替换方向不同'), ('counterevidence', '反向线索')):
        names = selection.get(key, [])
        if names:
            alerts.append(f'{label}{len(names)}项（{_short_names(names)}）')
    if rf_anomaly != lr_anomaly:
        alerts.insert(0, 'RF 与 LR 的阈值判断不同')
    if selection.get('status') == 'unavailable':
        return dict(facts=[], observed='当前证据暂不可用。',
                    reason='解释计算未完成。',
                    next='核对输入或在详情查看失败原因；查外部日志后再判断，证据不足选“待补证”。',
                    alerts=alerts, unavailable=selection.get('reason', '原因未知'))
    support = [c for c in cards if c.get('role') == '支持当前模型提示']
    if support:
        if rf_anomaly:
            reason = f'主要证据中{len(support)}项分别改回训练正常中位值后，RF 分数下降，支持当前提示；仍需日志核对原因。'
        else:
            reason = f'主要证据中{len(support)}项分别改为参考中位值后，RF 分数上升，说明当前输入使分数更低。'
    else:
        reason = ('当前高分尚缺可核查的主要支持项，请先补证。' if rf_anomaly
                  else '尚未找到明显的主要支持项；未提示异常仍需结合业务核对。')
    if selection.get('counterevidence'):
        reason += '反向线索也需一并核对。'
    observed = '；'.join(facts) + '。' if facts else '未筛出明显问题项。'
    return dict(facts=facts, observed=observed, reason=reason,
                next=_next_checks(cards, bool(selection.get('missing_fields'))), alerts=alerts,
                unavailable=None)


def render_record_evidence(payload):
    summary = review_guidance(payload)
    st.markdown('**建议核查**  \n' + literal_markdown(summary['next']))
    if summary['alerts']:
        st.caption('；'.join(summary['alerts']))
    if summary['unavailable']:
        st.caption('失败原因：' + summary['unavailable'])


def _signed(value):
    return '不可用' if value is None else f'{value:+.4f}'


def _brief_archetype(arch):
    labels = {
        'dual_model_disagreement': '两模型提示不同',
        'missing_imputation_sensitive': '先核对缺失数据',
        'outlier_feature_driven': '超出参考的项目推高评分',
        'single_feature_push': '单项推高评分',
        'multi_in_range_joint_push': '范围内多项推高评分',
        'unverified_multi_driver_push': '多项推高评分',
        'insufficient_driver_evidence': '需补充核查依据',
        'normal_background_consistent': '两模型均未提示异常',
        'score_unavailable': '评分不完整',
        'explanation_unavailable': '评分依据不可用',
    }
    return labels.get(arch.get('code'), arch.get('label', '评分线索').split('（', 1)[0])


def _review_styles():
    return """<style>
.st-key-record_checklist [data-testid="stExpanderDetails"] > [data-testid="stVerticalBlock"]{height:auto;flex:none;gap:.75rem;}
.st-key-record_checklist [data-testid="stExpanderDetails"] [data-testid="stElementContainer"]{flex-shrink:0;}
.st-key-record_checklist h4{margin:0!important;padding:0!important;font-size:1.08rem;}
.review-check-table{width:100%;margin:0!important;border-collapse:collapse;font-size:.9rem;line-height:1.55;table-layout:fixed;}
.review-check-table th,.review-check-table td{padding:10px 12px;text-align:left;vertical-align:top;border-bottom:1px solid #dce4ee;overflow-wrap:anywhere;}
.review-check-table th{background:#edf3f8;font-weight:600;color:#294760;}
.review-check-table th:nth-child(1){width:26%}.review-check-table th:nth-child(2){width:47%}.review-check-table th:nth-child(3){width:27%}
.review-check-table .check-state{display:block;color:#516e83;font-size:.8rem;margin-top:2px;}
.review-check-table .score-line{display:block;white-space:normal;font-variant-numeric:tabular-nums;}
.review-note-preview{padding:1rem 1.15rem;border:1px solid #dce4ee;border-radius:8px;background:#fff;line-height:1.65;overflow-wrap:anywhere;font-size:.9rem;}
.review-note-preview section+section{margin-top:1rem;}
.review-note-preview strong{display:block;margin-bottom:.3rem;}
.review-check-actions{margin:0;padding-left:1.3rem;line-height:1.65;font-size:.9rem;overflow-wrap:anywhere;}
.review-check-actions li+li{margin-top:.65rem;}
.review-check-actions strong{display:block;margin-bottom:.15rem;}
.review-field-summary{container-type:inline-size;min-width:0}
.review-first-action{display:block;background:#eff5fb;border-left:3px solid #356785;padding:.58rem .8rem;color:#203f55;line-height:1.5;overflow-wrap:anywhere}
.review-first-action strong{display:block;margin-bottom:.18rem;color:#234c68}
.review-observations{margin:.4rem 0 0;padding:0;list-style:none}
.review-observations li{display:grid;grid-template-columns:minmax(8rem,.9fr) minmax(0,1.4fr);gap:.25rem .9rem;padding:.5rem .15rem;border-bottom:1px solid #e1e7ed;line-height:1.4;overflow-wrap:anywhere}
.review-observations li:last-child{border:0}
.review-observations small{display:block;margin-top:.18rem;color:#4e6171;font-size:.83rem}
.review-observations .observed{font-variant-numeric:tabular-nums}
.review-score-wrap{container-type:inline-size;min-width:0}
@container(max-width:26rem){.review-observations li{grid-template-columns:1fr}.review-observations .field small{display:inline;margin-left:.45rem}}
@container(max-width:30rem){.review-check-table thead{display:none}.review-check-table,.review-check-table tbody,.review-check-table tr,.review-check-table td{display:block;width:100%;box-sizing:border-box}
.review-check-table tr{border-bottom:1px solid #dce4ee;padding:.4rem}.review-check-table td{border:0;padding:2px 0}.review-check-table td:before{content:attr(data-label) "　";color:#516e83;font-size:.82rem}.review-check-table td:first-child:before{content:none}}
@media(max-width:700px){.review-check-table thead{display:none}.review-check-table,.review-check-table tbody,.review-check-table tr,.review-check-table td{display:block;width:100%;box-sizing:border-box;}
.review-check-table tr{border:1px solid #dce4ee;border-radius:8px;margin-bottom:8px;padding:5px 8px;}
.review-check-table td{border:0;padding:3px 0}.review-check-table td:before{content:attr(data-label) "：";color:#516e83;font-size:.82rem;}
.review-check-table td:first-child:before{content:none}}
</style>"""


def _check_table(rows, contrib_lookup=None, *, show_checks=False):
    """Keep observations, references and the two score operations distinct."""
    headings = ('数据项', '当前值 / 训练参考', 'RF 评分影响')
    body = []
    contrib_lookup = contrib_lookup or {}
    for row in rows:
        reference = row['训练正常参考'].replace('训练正常记录参考范围：', '').replace('中间值（排序后居中的值）：', '中间值 ')
        c_info = row.get('_contribution', contrib_lookup.get(row['数据项'], {}))
        band_text = c_info.get('区间分级', '')
        state_display = band_text or row['当前状态']
        values = (
            '<strong>' + escape(str(row['数据项'])) + '</strong><span class="check-state" title="'
            + escape(state_display, quote=True) + '">' + escape(state_display.split('（', 1)[0]) + '</span>',
            '<strong>' + escape(str(row['当前值'])) + '</strong><span class="check-state">参考：' + escape(reference) + '</span>',
            '<span class="score-line">路径贡献 <strong>' + _signed(c_info.get('rf_path_contribution')) + '</strong></span>'
            + '<span class="score-line">替换后变化 <strong>' + _signed(c_info.get('rf_replace_delta')) + '</strong></span>'
            + ('<span class="check-state">含缺失状态变化</span>' if c_info.get('effect_note') else ''),
        )
        cells = ['<td data-label="' + label + '">' + value + '</td>' for label, value in zip(headings, values)]
        body.append('<tr>' + ''.join(cells) + '</tr>')
    st.markdown(_review_styles() + '<div class="review-score-wrap"><table class="review-check-table"><thead><tr>'
        + ''.join('<th>' + label + '</th>' for label in headings)
        + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table></div>', unsafe_allow_html=True)
    if show_checks:
        sentences = list(dict.fromkeys(part for row in rows for part in row['具体核查'].split('。') if part))
        st.caption('具体核查：' + '。'.join(sentences) + '。')


def _render_draft_preview(draft_text):
    sections = []
    for paragraph in draft_text.strip().split('\n\n'):
        title, _, body = paragraph.partition('\n')
        sections.append('<section><strong>' + escape(title.strip('【】')) + '</strong>'
                        + escape(body).replace('\n', '<br>') + '</section>')
    st.markdown('<div class="review-note-preview">' + ''.join(sections) + '</div>', unsafe_allow_html=True)


def _render_check_actions(primary, supplement):
    """Group identical follow-up actions once, keeping their affected fields."""
    grouped = {}
    for row in primary:
        for sentence in row['具体核查'].split('。'):
            if sentence:
                grouped.setdefault(sentence, []).append(row['数据项'])
    items = []
    for action, names in grouped.items():
        fields = '、'.join(dict.fromkeys(names))
        title = '共同检查' if len(set(names)) == len(primary) and len(primary) > 1 else fields
        items.append('<li><strong title="' + escape(fields, quote=True) + '">' + escape(title)
                     + '</strong>' + escape(action) + '。</li>')
    for row in supplement:
        value = row.get('当前值', '')
        items.append('<li><strong>' + escape(row['数据项']) + '</strong>'
                     + (escape(value) + '<br>' if value else '') + escape(row['具体核查']) + '</li>')
    st.markdown('<ul class="review-check-actions">' + ''.join(items) + '</ul>', unsafe_allow_html=True)


def _render_first_checks(rows):
    first = rows[0]
    items = []
    for row in rows[:3]:
        reference = row['训练正常参考'].removeprefix('训练正常记录参考范围：').replace('中间值（排序后居中的值）：', '中间值 ')
        comparison = row.get('直观对照')
        observed = comparison or row['当前值']
        detail = '原值　' + row['当前值'] if comparison else '训练参考　' + reference
        items.append('<li><div class="field"><b>' + escape(row['数据项']) + '</b><small>'
                     + escape(row['当前状态']) + '</small></div><div class="observed"><b>'
                     + escape(observed) + '</b><small>' + escape(detail)
                     + '</small></div></li>')
    st.html('<div class="review-field-summary"><div class="review-first-action"><strong>先核对'
            + escape(first['数据项']) + '</strong>' + escape(first['具体核查']) + '</div>'
            + '<ul class="review-observations" aria-label="重点字段、输入值和训练参考">'
            + ''.join(items) + '</ul></div>')


def _merged_note(existing, draft_text):
    """Append once; never discard an existing human note or truncate it."""
    if draft_text in existing:
        return existing
    merged = existing + ('\n\n' if existing else '') + draft_text
    return merged if len(merged) <= 4000 else None


def _current_editor_note(editor):
    """Use a just-submitted text value only when its widget owns this draft."""
    identity = (editor.input_key, editor.record_id, getattr(editor, 'baseline', None))
    if st.session_state.get('_editor_identity') == identity:
        return st.session_state.get('_note', editor.note)
    return editor.note


def _apply_draft_note(draft_text, input_key=None, record_id=None):
    """Append to the current unlocked editor without saving or setting a verdict."""
    ui = st.session_state.get('ui')
    editor = getattr(ui, 'editor', None)
    if editor is None or getattr(ui, 'exports', None) is not None or getattr(ui, 'pending', None) is not None:
        return
    if (input_key is not None and editor.input_key != input_key) or (record_id is not None and editor.record_id != record_id):
        return
    note = _merged_note(_current_editor_note(editor), draft_text)
    if note is not None:
        st.session_state['_note'] = editor.note = note


def _render_band_summary(bands):
    """Non-overlapping categories; distance is a reference comparison, not risk."""
    categories = [('high_risk', '显著偏离参考'), ('outside_moderate', '一般超出参考'),
                  ('near_boundary', '范围内靠近边界'), ('core_normal', '范围内其他'),
                  ('discrete_or_fixed', '编号 / 固定值一致'), ('insufficient_reference', '参考不足'), ('missing', '原始缺失')]
    chips = ''.join('<span class="interval-chip">' + label + ' <strong>' + str(bands.get(key, 0)) + '</strong></span>'
                    for key, label in categories if bands.get(key, 0))
    st.markdown('<style>.interval-summary{display:flex;flex-wrap:wrap;gap:6px;margin:5px 0 8px}'
                '.interval-chip{padding:4px 9px;border-radius:6px;background:#f0f4f8;color:#36516a;font-size:.82rem;}'
                '.interval-chip strong{font-variant-numeric:tabular-nums}</style>'
                '<div class="interval-summary">' + chips + '</div>', unsafe_allow_html=True)
    st.caption(f'共 {bands.get("total", 0)} 项')


def _render_deviation_chart(all_fields):
    """Top fields by deviation from the frozen training reference interval.

    Band = [P01, P99] of the training-normal reference; marker = actual value.
    Outside the band does not mean attack; it only prioritises human checks.
    """
    rows = []
    for r in all_fields or []:
        try:
            value = float(r.get('value'))
            ref = r.get('reference') or {}
            p01, p99 = float(ref.get('p01')), float(ref.get('p99'))
        except (TypeError, ValueError):
            continue
        if r.get('missing') or not (p99 > p01):
            continue
        width = p99 - p01
        if value > p99:
            dev = (value - p99) / width
        elif value < p01:
            dev = (p01 - value) / width
        else:
            dev = 0.0
        rows.append({'字段': r.get('name') or r.get('field'), '实际值': value,
                     '参考下界': p01, '参考上界': p99, '单位': r.get('unit') or '',
                     '位置': '超出参考' if dev > 0 else '参考内', '_dev': dev})
    if not rows:
        return
    rows.sort(key=lambda d: -d['_dev'])
    top = rows[:5]
    if not any(d['_dev'] > 0 for d in top):
        return
    import altair as alt
    st.markdown('###### 重点字段偏离（参考区间与实际值）')
    for d in top:
        width = d['参考上界'] - d['参考下界']
        lo = min(d['参考下界'], d['实际值']) - width * 0.08
        if min(d['参考下界'], d['实际值']) >= 0:
            lo = max(0.0, lo)
        hi = max(d['参考上界'], d['实际值']) + width * 0.08
        one = pd.DataFrame([d])
        band = alt.Chart(one).mark_bar(height=13, color='#e3e6ee', cornerRadius=3).encode(
            x=alt.X('参考下界:Q', title=f"实际值（{d['单位']}）" if d['单位'] else '实际值',
                    scale=alt.Scale(domain=[lo, hi])),
            x2='参考上界:Q',
        )
        point = alt.Chart(one).mark_point(filled=True, size=110).encode(
            x='实际值:Q',
            color=alt.Color('位置:N', legend=None, scale=alt.Scale(
                domain=['超出参考', '参考内'], range=['#c75c59', '#5e6ad2'])),
            tooltip=['字段', '实际值', '参考下界', '参考上界', '单位', '位置'],
        )
        st.markdown(f"**{d['字段']}**")
        st.altair_chart((band + point).properties(height=44), use_container_width=True)
    st.caption('浅色带为训练正常参考 P01–P99，圆点为本条实际值；每行独立刻度，超出参考只表示优先核查，不等于攻击。')


def render_review_checklist(payload):
    """Put concrete checks, interval summary, and contribution impact before the judgment form."""
    st.markdown(_review_styles(), unsafe_allow_html=True)
    with st.container(border=True, key='record_checklist'):
        st.markdown('#### 核查重点')
        if not (_INTERVAL_RULES_AVAILABLE and _CONTRIBUTION_GUIDANCE_AVAILABLE):
            st.warning('文件版本不一致，请重新解压运行包。')
            return
        primary, supplement = review_focus_rows(payload)
        if not primary and not supplement:
            st.info('当前记录缺少可用的字段解释。请按输入行号核对原始 CSV 和同一时段的服务日志。证据不足时可保存为“待补证”。')
            reason = payload.get('selection', {}).get('reason')
            if reason:
                with st.expander('查看计算原因'):
                    st.text(str(reason))
            return
        arch = classify_attribution_archetype(payload)
        bands = arch['bands']
        counts = [f"输入字段 {bands['total']} 项", f"超出参考 {bands['outside_total']} 项"]
        counts.extend(f'{label} {bands[key]}' for key, label in
                      [('near_boundary', '近边界'), ('missing', '缺失'), ('insufficient_reference', '参考不足')]
                      if bands.get(key, 0))
        st.caption(_brief_archetype(arch) + ' · ' + ' · '.join(counts))
        if primary:
            _render_first_checks(primary)
            _render_deviation_chart(payload.get('all_fields', []))
        else:
            st.info('本条记录未选出重点字段。可先按地址和时间核对本批邻近记录，再查同一业务的日志或工单。')
        more = f'各项核查方法与评分明细（{len(primary)} 项）' if primary else '查看邻近记录与日志核查方法'
        with st.expander(more):
            _render_check_actions(primary, supplement)
            if primary:
                _check_table(primary)
        if 'ui' in st.session_state and getattr(st.session_state['ui'], 'editor', None) is not None:
            draft_text = build_audit_note_draft(payload)
            with st.expander('系统核查线索'):
                _render_draft_preview(draft_text)
                st.caption('按这些线索核对后，请在人工依据栏记录实际观察。')


def field_frame(payload):
    return pd.DataFrame([{'数据项': r['name'], '字段': r['field'], '原值': r['value'], '单位': r['unit'],
        '缺失': r['missing'], 'RF填补值': r['imputed_rf'], 'LR填补值': r['imputed_lr'],
        'P01': r['reference']['p01'], 'P50': r['reference']['p50'], 'P99': r['reference']['p99'],
        '参考状态': r['reference_status'], '本批百分位': r['batch_percentile'], '有效观测数': r['batch_observed_count']}
        for r in sorted(payload.get('all_fields', []), key=lambda r: r['missing'])])


def rf_contribution_rows(payload):
    """Chart shown fields plus the exact sum of every omitted field."""
    result = payload['decomposition']
    fields = {row['field']: row for row in payload.get('all_fields', [])}
    visible = list(dict.fromkeys(result.get('default_fields', result['rf']['top_fields'])))[:12]
    contributions = result['rf']['fields']
    shown = sorted((row for row in contributions if row['field'] in visible),
                   key=lambda row: -abs(row['contribution']))
    rows = [{'field': row['field'], 'name': fields.get(row['field'], {}).get('name', row['field']),
             'contribution': row['contribution'], 'kind': '字段'} for row in shown]
    omitted = [row for row in contributions if row['field'] not in visible]
    if omitted:
        rows.append({'field': '__remaining__', 'name': f'其余{len(omitted)}项合计',
                     'contribution': math.fsum(row['contribution'] for row in omitted), 'kind': '其余合计'})
    return rows


def _contribution_spec(rows):
    names = [row['name'] for row in rows]
    values = [dict(row, label=(row['name'] + f'（{i+1}）') if names.count(row['name']) > 1 else row['name'],
                   direction='抬高分数' if row['contribution'] > 0 else
                   '降低分数' if row['contribution'] < 0 else '无加减') for i, row in enumerate(rows)]
    return {'height': max(90, len(rows) * 28) + 150, 'data': {'values': values},
            'layer': [
                {'mark': {'type': 'bar', 'cornerRadiusEnd': 2}, 'encoding': {
                    'x': {'field': 'contribution', 'type': 'quantitative', 'title': '对主要模型（RF）分数的加减',
                          'scale': {'zero': True}, 'axis': {'tickCount': 4, 'labelFlush': False}},
                    'y': {'field': 'label', 'type': 'nominal', 'sort': None, 'title': None,
                          'axis': {'labelLimit': 165}},
                    'color': {'field': 'direction', 'type': 'nominal', 'title': None,
                              'scale': {'domain': ['抬高分数', '降低分数', '无加减'],
                                        'range': ['#B55335', '#27758B', '#82909C']},
                              'legend': {'orient': 'top', 'direction': 'vertical', 'columns': 1}},
                    'tooltip': [{'field': 'name', 'title': '数据项'}, {'field': 'field', 'title': '字段'},
                                {'field': 'contribution', 'title': 'RF加减', 'format': '+.6f'}]}},
                {'mark': {'type': 'rule', 'color': '#718091'}, 'encoding': {'x': {'datum': 0}}}],
            'config': {'view': {'stroke': None}}}


def render_score_accounting(payload):
    if not (_INTERVAL_RULES_AVAILABLE and _CONTRIBUTION_GUIDANCE_AVAILABLE):
        st.warning('文件版本不一致，请重新解压运行包。')
        return
    result = payload.get('decomposition', {})
    if result.get('status') != 'available':
        st.info('评分分解暂不可用：' + result.get('reason', '当前无法还原分数。'))
        return
    fields = {r['field']: r for r in payload.get('all_fields', [])}
    rf, lr = result['rf'], result['lr']
    chart_rows = rf_contribution_rows(payload)
    arch = classify_attribution_archetype(payload)
    bands = arch['bands']

    st.markdown('**' + literal_markdown(_brief_archetype(arch)) + '**')
    _render_band_summary(bands)

    st.markdown('**RF 评分构成**')
    st.write(f'起始 {rf["base"]:.4f} {rf["sum"]:+.4f} ＝ {rf["reconstructed"]:.4f}')
    st.vega_lite_chart(_contribution_spec(chart_rows), width='stretch')

    unified_top = unified_interval_contribution_rows(payload, top_only=True, limit=12)
    if unified_top:
        st.markdown('**关键数据**')
        visible_rows = [dict(数据项=row['数据项'], 当前值=row['当前原值'],
                             训练正常参考=row['训练正常区间 (P01～P99)'], 当前状态=row['五级区间状态'],
                             具体核查='', _contribution=dict(区间分级=row['五级区间状态'],
                                 rf_path_contribution=row['RF路径贡献 (概率点)'],
                                 rf_replace_delta=row['常见值对照变化 (ΔRF)'])) for row in unified_top]
        _check_table(visible_rows[:6])
        if len(visible_rows) > 6:
            with st.expander(f'其余 {len(visible_rows)-6} 项重点数据'):
                _check_table(visible_rows[6:])

    unified_all_map = {r['字段']: r for r in unified_interval_contribution_rows(payload, top_only=False)}

    def row(item):
        field = fields.get(item['field'], {})
        u_row = unified_all_map.get(item['field'], {})
        return {
            '数据项': field.get('name', item['field']),
            '字段': item['field'],
            '原值': field.get('value'),
            '单位': field.get('unit', ''),
            '缺失': field.get('missing'),
            '训练正常区间 (P01～P99)': u_row.get('训练正常区间 (P01～P99)', ''),
            '参考区间状态': u_row.get('五级区间状态', field.get('reference_status', '')),
            '参考状态': field.get('reference_status', ''),
            '本批百分位': u_row.get('本批百分位', ''),
            'RF分数加减': item['contribution'],
            '常见值对照变化 (ΔRF)': u_row.get('常见值对照变化 (ΔRF)'),
            '路径影响说明': u_row.get('对异常指数的作用说明', ''),
        }

    with st.expander('RF 完整计算'):
        st.dataframe(pd.DataFrame([row(r) for r in rf['fields']]), hide_index=True, height=320, width='stretch')
        rest = next((r['contribution'] for r in chart_rows if r['kind'] == '其余合计'), 0.)
        st.write(f'基准 {rf["base"]:.12g}；其余合计 {rest:+.12g}；全部合计 {rf["sum"]:+.12g}；'
                 f'重构 {rf["reconstructed"]:.12g}；原分数 {rf["score"]:.12g}。')
        st.caption(f'实际 {rf["tree_count"]} 棵树；重构误差 {result["errors"]["rf_score"]:.3g}；'
                   f'未单列字段最大绝对贡献 {result.get("omitted_max_abs_path_contribution", 0):.9g}。')
        st.json(rf, expanded=False)
    st.markdown('**辅助模型（LR）**')
    st.write(f'当前 LR 分数：{lr["reconstructed"]:.4f}')
    with st.expander('LR 完整计算'):
        st.write(f'截距 {lr["intercept"]:.4f} → 内部线性值 {lr["logit"]:.4f} → LR 分数 {lr["reconstructed"]:.4f}')
        rf_terms = {r['field']: r['contribution'] for r in rf['fields']}
        st.dataframe(pd.DataFrame([{'数据项': fields.get(r['field'], {}).get('name', r['field']),
            '字段': r['field'], 'LR线性项（对数几率）': r['term'],
            '贡献方向说明': '各自基准下贡献方向不同' if rf_terms.get(r['field'], 0)*r['term'] < 0 else ''}
            for r in lr['fields']]), hide_index=True, height=320, width='stretch')
        st.caption(f'线性值误差 {result["errors"]["lr_logit"]:.3g}；分数误差 {result["errors"]["lr_score"]:.3g}。')
        st.json(lr, expanded=False)


def _render_card(card):
    plain = review_check_items({'cards': [card]})[0]
    st.markdown('**' + literal_markdown(card['name']) + '** · ' + card['role'])
    if card['missing']:
        st.write('原值：原始缺失')
        st.caption('替换试算包含缺失状态变化，不能解释为缺失项的原值贡献。')
    else:
        st.write('原值 ' + plain['当前值'] + '；' + plain['当前状态'] + '。')
    ref, pos = card['reference'], card['reference_position']
    st.write(plain['训练正常参考'])
    if pos.get('near'):
        side = '低端' if pos['near_side'] == '下界' else '高端'
        st.caption(f'距参考{side} {pos["boundary_fraction"]:.2%} 中位数分段跨度')
    if card['missing']:
        st.write(f'模型填补值：RF {_fmt(card["imputed_rf"])}；LR {_fmt(card["imputed_lr"])}')
    st.markdown('**参考值对照**')
    st.dataframe(pd.DataFrame([{'模型': '主要模型（RF）' if key == 'rf' else '辅助模型（LR）', '当前分数': card[key]['before'],
        '按常见值对照后': card[key]['after'], '分数变化': card[key]['delta'],
        '怎么看': '暂时无法比较' if card[key]['delta'] is None else '变化很小' if card[key]['weak'] else card[key]['direction']}
        for key in ('rf', 'lr')]), hide_index=True, width='stretch')
    if card['direction_difference']:
        st.caption('两模型变化方向不同' + ('，LR变化较小。' if card['lr']['weak'] else '。'))
    st.markdown('**核查方法**')
    st.write(plain['具体核查'])
    with st.expander('字段详情'):
        st.caption(card['field'] + ' · ' + '；'.join(card['selection_reasons']))
        st.caption(f'P01（低端）/ P50（中间值）/ P99（高端）：{_fmt(ref["p01"])} / {_fmt(ref["p50"])} / {_fmt(ref["p99"])} {card["unit"]}')
        percentile = '不可用' if card['batch_percentile'] is None else f'{card["batch_percentile"]:.1f}%'
        st.caption(f'本批位置 {percentile} · 有效观测 {card["batch_observed_count"]:,} 条')
        st.caption(card['meaning'])


def _render_evidence_cards(payload):
    cards = payload.get('cards', [])
    selection = payload.get('selection', {})
    if selection.get('status') == 'unavailable':
        st.warning('主要证据暂不可用：' + selection.get('reason', '原因未知'))
        return
    st.caption(f'主要核查数据 {len(cards)} 项')
    if cards:
        frame = pd.DataFrame([{'数据项': c['name'], '原值': '缺失' if c['missing'] else _fmt(c['value']), '单位': c['unit'],
            '参考状态': review_check_items({'cards':[c]})[0]['当前状态'], '为何列出': c['role']} for c in cards])
        st.dataframe(frame, hide_index=True, height=min(260, 38 + len(cards)*35), width='stretch')
        fields = [c['field'] for c in cards]
        names = {c['field']: c['name'] for c in cards}
        chosen = st.selectbox('查看一项证据', fields, format_func=names.get,
                              key='_detail_evidence_field_' + str(payload.get('record_id', '')))
        with st.container(border=True):
            _render_card(next(c for c in cards if c['field'] == chosen))
    else:
        st.info('未筛出明显问题项。' +
                ('当前高分仍缺可核查解释，建议待补证。' if payload.get('scores', {}).get('rf_anomaly') else ''))
    if selection.get('lr_only'):
        with st.expander(f'辅助模型另外提醒了 {len(selection["lr_only"])} 项'):
            st.caption('LR 对照变化较大、RF 对照变化较小的项目')
            st.dataframe(pd.DataFrame(selection['lr_only']), hide_index=True, width='stretch')


def _render_all_field_issues(payload):
    selection = payload.get('selection', {})
    st.markdown('**其他线索**')
    alerts = review_guidance(payload)['alerts']
    for alert in alerts:
        st.write(alert)
    if not alerts:
        st.caption('无其他线索。')
    rows = [{'数据项': field_name(r['field']), '怎么看': r['direction_label'], '与当前提示的关系': r['role'],
             '主要模型对照后的变化': r['rf']['delta'], '辅助模型对照后的变化': r['lr']['delta']}
            for r in payload.get('all_fields', []) if r['direction_difference'] or r['role'] == '反向线索']
    if rows:
        with st.expander(f'方向差异与反向线索（{len(rows)}项）'):
            st.dataframe(pd.DataFrame(rows), hide_index=True, height=260, width='stretch')
    if selection.get('omitted', 0):
        st.caption(f'其他候选 {selection["omitted"]} 项，见全部字段。')


def _render_raw_methods(payload):
    _render_all_field_issues(payload)
    full = field_frame(payload)
    if not full.empty:
        with st.expander('全部字段与填补值'):
            st.caption('训练正常参考：P01 低端 · P50 中间值 · P99 高端')
            st.dataframe(full, hide_index=True, height=320, width='stretch')
    if payload.get('input_quality_details'):
        with st.expander('原值与数据问题'):
            st.dataframe(pd.DataFrame(payload['input_quality_details']), hide_index=True, width='stretch')
    with st.expander('多项参考值对照'):
        joint, stability = payload.get('joint_test', {}), payload.get('stability', {})
        st.caption('将主要项目一起换为参考值，查看分数变化。')
        st.write(stability.get('status', '稳定性不可评估'))
        st.json({'joint': joint, 'stability': stability}, expanded=False)
    with st.expander('邻近记录与日志'):
        neighbor = payload.get('neighbors', {})
        st.caption(neighbor.get('scope', '未保留可比较的IP或时间上下文。'))
        if neighbor.get('available'):
            st.write(f'同一来源地址的其他记录 {_fmt(neighbor.get("same_source"))} 条；同一目标地址的其他记录 {_fmt(neighbor.get("same_destination"))} 条；'
                     f'该来源还涉及 {_fmt(neighbor.get("source_fanout"))} 个目标地址；地址、端口、协议均相同的其他记录 {_fmt(neighbor.get("repeat_five_tuple"))} 条。')
        if neighbor.get('time_window'):
            window = neighbor['time_window']
            st.write(f'前后各{neighbor.get("window_seconds", 60)}秒：{window["all"]} 条；'
                     f'可比较时间 {window["valid_timestamps"]} 条（含当前）；{window["timezone"]}。')
        if not neighbor.get('available') and not neighbor.get('time_window'):
            st.caption('请从原始流量或外部日志补证。')
        for note in neighbor.get('notes', []):
            st.caption(note)
    with st.expander('完整计算快照'):
        st.write(payload.get('assessment', {}).get('status', '当前解释不可用。'))
        for reason in payload.get('assessment', {}).get('reasons', []):
            st.caption(reason)
        st.json(payload, expanded=False)


def render_evidence_detail(payload):
    """Return tabs so the host can append existing tools to the raw-data tab."""
    tabs = st.tabs(['评分图解', '证据明细', '原始数据与方法'])
    with tabs[0]:
        render_score_accounting(payload)
    with tabs[1]:
        _render_evidence_cards(payload)
    with tabs[2]:
        _render_raw_methods(payload)
    return tabs


def render_historical_event_review(event):
    """Present a saved event reference without promoting malformed metadata."""
    from app_record_tools import valid_saved_event_review
    if not valid_saved_event_review(event):
        st.warning('保存时事件复核信息不完整，请查看原始快照；不能据此还原整簇确认依据。')
        return
    refs = event['representative_review_ids']
    labels = '、'.join(str(value) for value in refs)
    st.caption('整簇确认 · 疑似事件 ' + literal_markdown(event['event_id'])
               + f' · 代表复核操作 {labels}。本条未单独复核。')
    fields = event.get('signature_fields', [])
    if not isinstance(fields, list) or any(not isinstance(field, str) for field in fields):
        st.warning('保存时事件复核信息不完整，无法展示归并依据或代表覆盖。')
    elif fields:
        size = event.get('event_size')
        if type(size) is not int or size < len(refs):
            st.warning('保存时事件复核信息不完整，无法展示归并依据或代表覆盖。')
        else:
            st.caption('归并依据 · ' + '、'.join(map(field_name, fields))
                       + f' · 代表抽样覆盖 {len(refs)}/{size}')


def _engine_snapshot_ready(name, ready):
    if not ready:
        st.warning(name + '记录不完整，无法展示这一部分。')
    return ready


def _engine_render_numbers(record, names):
    for name in names:
        value = record.get(name)
        if not isinstance(value, Real) or is_bool(value):
            return False
        try:
            if not math.isfinite(value):
                return False
        except OverflowError:
            return False
    return True


def render_engine_panel(snapshot, *, historical=False):
    """Render the same immutable values used by save/export, including past rows."""
    if snapshot is None:
        return
    if not isinstance(snapshot, dict):
        st.warning('研判引擎证据记录不完整，无法展示。')
        return
    if not snapshot or snapshot.get('status') == 'not_used':
        return
    if snapshot.get('status') != 'verified':
        reason = snapshot.get('reason')
        st.warning('研判引擎证据暂不可用：' + (reason if isinstance(reason, str) and reason else '证据不完整'))
        return
    record = snapshot.get('results') if isinstance(snapshot.get('results'), dict) else {}
    params = snapshot.get('parameters') if isinstance(snapshot.get('parameters'), dict) else {}
    with st.container(border=True, key='historical_engine_panel' if historical else 'engine_panel'):
        st.markdown('**保存时的研判引擎证据**' if historical else '**研判引擎证据**')
        world = snapshot.get('open_world') if isinstance(snapshot.get('open_world'), dict) else {}
        state = world.get('state')
        state = state if isinstance(state, str) else None
        diagnostic_keys = ('b2_mode', 'b2_fallback_reason', 'b3_calculation_status', 'b3_statistical_status')
        if _engine_snapshot_ready('B2/B3 诊断', all(is_scalar(world.get(k)) for k in diagnostic_keys)):
            for line in engine_diagnostic_lines(world):
                st.caption(('保存时批次 · ' if historical else '本批 · ') + line)
        if state:
            st.markdown(f'引擎四态判断 **{state}**')
        heading = '共形预测集' if state else '研判结论'
        frame = params.get('evidence_frame')
        open_frame = isinstance(frame, str) and frame == 'ANU'
        if _engine_snapshot_ready('研判结论', isinstance(record.get('cp_set'), str) and
                                  _engine_render_numbers(record, ('ds_belief', 'ds_plausibility', 'ds_conflict'))):
            st.markdown(f"{heading} **{record['cp_set']}** · 对已知攻击的证据支持度 Bel **{record['ds_belief']:.3f}** · "
                        f"似然度 Pl **{record['ds_plausibility']:.3f}** · 冲突度 K **{record['ds_conflict']:.3f}**")
            if _engine_snapshot_ready('证据框架', isinstance(frame, str) and frame in ('AN', 'ANU')):
                st.caption('Mondrian 共形预测集：只含一个标签时确定，否则为不确定（集合为空表示既不像正常、'
                           '也不像已知攻击）；[Bel, Pl] 是融合后「已知攻击」的信任区间，K 只来自 RF 与 LR 的分歧，'
                           '典型性证据与它们在结构上不冲突。' if open_frame else
                           '结论来自 Mondrian 共形预测集：只含一个标签时确定，否则为不确定；'
                           '[Bel, Pl] 是 Dempster 融合后的攻击信任区间，K 是三个证据源相互矛盾的质量。')
        if _engine_snapshot_ready('共形与概率端点',
                                  _engine_render_numbers(record, ('cp_p_attack', 'cp_p_normal', 'va_low', 'va_high'))
                                  and _engine_render_numbers(params, ('epsilon',))):
            st.caption(f"共形 p 值：攻击 {record['cp_p_attack']:.3f} · 正常 {record['cp_p_normal']:.3f}"
                       f"（显著性 ε = {params['epsilon']:.2f}）；"
                       f"Venn–Abers 概率端点 [{record['va_low']:.3f}, {record['va_high']:.3f}]")
            method = params.get('va_method')
            if _engine_snapshot_ready('概率端点计算方式', isinstance(method, str) and bool(method)):
                st.caption('概率端点按冻结校准样本逐分数精确计算；这不是固定置信水平的置信区间。'
                           if method == 'exact-isotonic-two-label-refits' else
                           '此历史快照使用旧版网格近似概率端点，仅供参考。')
        if _engine_snapshot_ready('复核提示', is_bool(record.get('engine_review')) and
                                  _engine_render_numbers(params, ('conflict_cut',))):
            st.caption(f"复核提示：{'需要再检查' if record['engine_review'] else '未进入引擎复核队列'}；"
                       f"冲突阈值 {params['conflict_cut']:.4f}。结论仍需结合外部信息核实。")
        with st.expander('三个证据源的基本概率分配'):
            sources = snapshot.get('source_masses')
            if _engine_snapshot_ready('证据源质量分配', isinstance(sources, list) and all(isinstance(v, dict) for v in sources)):
                masses = pd.DataFrame(sources)
                st.dataframe(masses, hide_index=True, width='stretch',
                             column_config={k: st.column_config.NumberColumn(format='%.4f')
                                            for k in ['原始分数', '可靠度 r', 'm(攻击)', 'm(正常)', 'm(主张一)', 'm(主张二)',
                                                      'm(未知Θ)'] if k in masses})
            if open_frame:
                st.caption('RF、LR 只对「是不是已知攻击」表态；典型性证据只对「像不像见过的流量」表态。'
                           '下面的 ECOD 逐字段贡献解释的是求和形式的尾部分数，典型性证据取其中最极端的三项。')
            decomposition = snapshot.get('ecod_decomposition')
            decomposition = decomposition if isinstance(decomposition, dict) else {}
            top = decomposition.get('top_fields')
            branch = decomposition.get('selected_branch')
            ready = (isinstance(top, list) and bool(top) and all(isinstance(row, dict)
                     and isinstance(row.get('field'), str) and isinstance(row.get('direction'), str)
                     and row['direction'] in ('left', 'right') and is_bool(row.get('input_missing'))
                     and _engine_render_numbers(row, ('contribution',)) for row in top)
                     and isinstance(decomposition.get('fields'), list) and isinstance(branch, str)
                     and branch in ('left', 'right', 'auto')
                     and _engine_render_numbers(decomposition, ('contribution_sum',))
                     and _engine_render_numbers(record, ('ecod_score',)))
            if _engine_snapshot_ready('ECOD 解释', ready):
                table = pd.DataFrame(top)
                table.insert(0, '数据项', table['field'].map(field_name))
                table['direction'] = table['direction'].map({'left':'偏低（左尾）','right':'偏高（右尾）'})
                table['input_missing'] = table['input_missing'].map({True:'已使用训练中位数',False:'原始有效值'})
                table = table.rename(columns={'field':'字段','contribution':'尾部贡献','direction':'方向','input_missing':'取值状态'})
                branch = {'left':'左尾','right':'右尾','auto':'偏度自动方向'}[branch]
                st.caption(f"ECOD 采用{branch}分支；全部 {len(decomposition['fields'])} 项贡献合计 "
                           f"{decomposition['contribution_sum']:.3f}，还原原始分数 {record['ecod_score']:.3f}。"
                           '下表仅列前六项；完整贡献随操作记录保存。')
                st.dataframe(table, hide_index=True, width='stretch',
                             column_config={'尾部贡献': st.column_config.NumberColumn(format='%.3f')})
                st.caption('贡献用于解释 ECOD 原始分数，不代表该字段对最终攻击结论的因果影响。')
