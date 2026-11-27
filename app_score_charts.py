"""Bounded pure aggregation and local Vega-Lite score displays."""
from functools import lru_cache
import math

import numpy as np


BIN_EDGES = tuple(i / 20 for i in range(21))
BIN_COLORS = ['#b5741d', '#5863b3']


def in_score_bin(scores, index):
    if type(index) is not int or not 0 <= index < 20:
        raise ValueError('分数区间编号超出范围')
    start, end = BIN_EDGES[index:index + 2]
    return scores.ge(start) & (scores.le(end) if index == 19 else scores.lt(end))


def bin_label(index):
    return f'[{BIN_EDGES[index]:.2f}, {BIN_EDGES[index + 1]:.2f}' + (']' if index == 19 else ')')


@lru_cache(maxsize=2)
def _histogram(scores):
    if any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores):
        raise ValueError('分数必须是 0 到 1 之间的有限数值')
    values, edges = np.histogram(scores, bins=BIN_EDGES)
    return tuple((float(edges[i]), float(edges[i+1]), int(value)) for i, value in enumerate(values))


def histogram_rows(scores):
    return [{'start': start, 'end': end, 'count': count}
            for start, end, count in _histogram(tuple(float(value) for value in scores))]


def histogram_spec(scores, threshold, margin, *, count_scale='symlog'):
    if count_scale not in ('symlog', 'linear'):
        raise ValueError('不支持的数量刻度')
    rows = histogram_rows(scores)
    scale = {'type': count_scale, 'zero': True}
    axis = {'format': ',d', 'tickMinStep': 1}
    if count_scale == 'symlog':
        scale['constant'] = 1
        maximum = max(1, max(row['count'] for row in rows))
        scale.update(domain=[0, maximum], nice=False)
        ticks = [0]
        value = 1
        while value < maximum:
            ticks.append(value)
            value *= 10
        axis['values'] = ticks + [maximum]
    return {'height': 230, 'layer': [
        {'data': {'values': rows}, 'mark': {'type': 'rect', 'color': '#245C8D'},
         'encoding': {'x': {'field': 'start', 'type': 'quantitative', 'title': 'RF 排序分数', 'scale': {'domain': [0, 1], 'nice': False}},
                      'x2': {'field': 'end'}, 'y': {'field': 'count', 'type': 'quantitative',
                          'title': '记录数（对数刻度）' if count_scale == 'symlog' else '记录数',
                          'scale': scale, 'axis': axis}, 'y2': {'datum': 0},
                      'tooltip': [{'field': 'start', 'title': '区间起点', 'format': '.2f'},
                                  {'field': 'end', 'title': '区间终点', 'format': '.2f'},
                                  {'field': 'count', 'title': '记录数'}]}},
        {'data': {'values': [{'lo': max(0., threshold-margin), 'hi': min(1., threshold+margin)}]},
         'mark': {'type': 'rect', 'color': '#F4B860', 'opacity': .22},
         'encoding': {'x': {'field': 'lo', 'type': 'quantitative'}, 'x2': {'field': 'hi'},
                      'tooltip': [{'field': 'lo', 'title': '范围下限', 'format': '.2f'},
                                  {'field': 'hi', 'title': '范围上限', 'format': '.2f'}]}},
        {'data': {'values': [{'threshold': threshold}]}, 'mark': {'type': 'rule', 'color': '#B54C25', 'strokeWidth': 2},
         'encoding': {'x': {'field': 'threshold', 'type': 'quantitative'},
                      'tooltip': [{'field': 'threshold', 'title': 'RF 判断分界'}]}}
    ], 'config': {'view': {'stroke': None}}}


def ruler_spec(rf_score, lr_score, config):
    values = [{'model': name.upper(), 'score': float(score), 'threshold': config[name+'_threshold'], 'zero': 0., 'one': 1.}
              for name, score in (('rf', rf_score), ('lr', lr_score))]
    y = {'field': 'model', 'type': 'nominal', 'title': None, 'sort': ['RF', 'LR']}
    return {'height': 160, 'layer': [
        {'data': {'values': values}, 'mark': {'type': 'rule', 'strokeWidth': 8, 'color': '#DCE4EE'},
         'encoding': {'x': {'field': 'zero', 'type': 'quantitative', 'title': '排序分数', 'scale': {'domain': [0, 1], 'nice': False}},
                      'x2': {'field': 'one'}, 'y': y, 'tooltip': None}},
        {'data': {'values': [{'model': 'RF', 'lo': max(0., config['rf_threshold']-config['near_margin']),
                             'hi': min(1., config['rf_threshold']+config['near_margin'])}]},
         'mark': {'type': 'rule', 'strokeWidth': 10, 'color': '#F4B860'},
         'encoding': {'x': {'field': 'lo', 'type': 'quantitative'}, 'x2': {'field': 'hi'}, 'y': y,
                      'tooltip': [{'field': 'lo', 'title': '范围下限', 'format': '.2f'},
                                  {'field': 'hi', 'title': '范围上限', 'format': '.2f'}]}},
        {'data': {'values': values}, 'mark': {'type': 'tick', 'color': '#B54C25', 'thickness': 2, 'size': 24},
         'encoding': {'x': {'field': 'threshold', 'type': 'quantitative'}, 'y': y,
                      'tooltip': [{'field': 'model', 'title': '模型'}, {'field': 'threshold', 'title': '判断分界'}]}},
        {'data': {'values': values}, 'mark': {'type': 'point', 'filled': True, 'color': '#17314D', 'size': 110},
         'encoding': {'x': {'field': 'score', 'type': 'quantitative'}, 'y': y,
                      'tooltip': [{'field': 'model', 'title': '模型'}, {'field': 'score', 'title': '当前分数', 'format': '.6f'}]}}
    ], 'config': {'view': {'stroke': None}}}


def render_histogram(predictions, rejected_count, config, *, on_bin=None, chart_key='batch_scores'):
    import streamlit as st
    if predictions.empty:
        return
    st.subheader('本次记录的分数分布')
    largest = max(histogram_rows(predictions.rf_score), key=lambda row: row['count'])
    st.markdown(f"最多集中在 **{largest['start']:.2f}–{largest['end']:.2f}**，"
                f"共 **{largest['count']:,} 条**（{largest['count']/len(predictions):.1%}）。")
    st.caption(f"图中 {len(predictions):,} 条，另有 {rejected_count:,} 条无法判断。"
               f"RF 分界 {config['rf_threshold']:.2f}，近界范围 ±{config['near_margin']:.2f}。")
    if on_bin is not None:
        st.html(f'''<div class="score-legend" aria-label="柱形图例">
            <span><i style="background:{BIN_COLORS[0]}"></i>与近界范围相交</span>
            <span><i style="background:{BIN_COLORS[1]}"></i>其他区间</span>
            </div><style>
            .score-legend{{display:flex;align-items:center;gap:.35rem 1rem;flex-wrap:wrap;color:var(--sub);font-size:.85rem}}
            .score-legend span{{display:inline-flex;gap:.35rem;align-items:center}}
            .score-legend i{{width:.65rem;height:.65rem;border-radius:2px}}
            </style>''')
    else:
        st.caption('橙线为分界，浅橙区域为近界范围。')
    choice = st.radio('数量刻度', ['对数数量', '线性数量'], horizontal=True, key='_histogram_scale')
    spec = histogram_spec(predictions.rf_score, config['rf_threshold'], config['near_margin'],
                          count_scale='symlog' if choice == '对数数量' else 'linear')
    if on_bin is None:
        st.vega_lite_chart(spec, use_container_width=True)
    else:
        bars = spec['layer'][0]
        for index, row in enumerate(bars['data']['values']):
            row['score_bin'] = index
            lo, hi = config['rf_threshold'] - config['near_margin'], config['rf_threshold'] + config['near_margin']
            row['region'] = '接近分界' if row['end'] > lo and row['start'] <= hi else '其他区间'
        # Streamlit 1.50 supports selection only on a unit view. Encode the
        # threshold range on the bars themselves to keep all 20 bins clickable.
        bars['height'] = spec['height']
        bars['config'] = spec['config']
        bars['mark'] = {'type': 'rect', 'cursor': 'pointer'}
        bars['encoding']['color'] = {'field': 'region', 'type': 'nominal', 'legend': None,
                                     'scale': {'domain': ['接近分界', '其他区间'], 'range': BIN_COLORS}}
        bars['params'] = [{'name': 'bucket',
                           'select': {'type': 'point', 'fields': ['score_bin'], 'toggle': False}}]

        def follow_bar():
            state = st.session_state[chart_key].get('selection', {}).get('bucket', [])
            chosen = state[-1].get('score_bin') if isinstance(state, list) and state else None
            if isinstance(state, dict):
                values = state.get('score_bin', [])
                chosen = values[-1] if isinstance(values, list) and values else values
            if type(chosen) is int and 0 <= chosen < 20:
                on_bin(chosen)

        st.vega_lite_chart(bars, use_container_width=True, key=chart_key,
                           on_select=follow_bar, selection_mode='bucket')
        rows = bars['data']['values']
        selected = st.selectbox('按 RF 分数区间查看记录', range(20), index=None,
                                placeholder='选择区间，也可以直接点击图中柱形',
                                format_func=lambda i: f"{bin_label(i)} · {rows[i]['count']:,} 条",
                                key=chart_key + '_choice')
        st.button('在复核队列查看区间', key=chart_key + '_open', disabled=selected is None,
                  on_click=on_bin, args=(selected,))


def render_score_comparison(record, config):
    """Show the inference scores and the model's own thresholds on one scale."""
    import streamlit as st
    rows = []
    for model, score in (('rf', record.rf_score), ('lr', record.lr_score)):
        threshold = float(config[model + '_threshold'])
        score = float(score)
        if not math.isfinite(score) or not math.isfinite(threshold):
            rows.append(f'<div class="q-score-values"><b>{model.upper()}</b><span>分数或分界暂不可用</span></div>')
            continue
        band = ''
        if model == 'rf':
            low = max(0., threshold - config['near_margin'])
            high = min(1., threshold + config['near_margin'])
            band = f'<i class="q-score-band" style="left:{low:.3%};width:{high - low:.3%}"></i>'
        rows.append(f'<div class="q-score-row"><div class="q-score-values">'
                    f'<b>{model.upper()} <span>分数</span> {score:.4f}</b><span>分界 <b>{threshold:.2f}</b></span></div>'
                    f'<div class="q-score-track" aria-hidden="true">{band}'
                    f'<i class="q-score-cut" style="left:{threshold:.3%}"></i>'
                    f'<i class="q-score-dot" style="left:clamp(3px,{score:.3%},calc(100% - 3px))"></i></div></div>')
    st.html("""<style>
    .q-score-comparison{display:grid;gap:.5rem;padding:.45rem .65rem .55rem;border:1px solid var(--line);
        border-radius:5px;background:var(--paper);color:var(--ink);font-variant-numeric:tabular-nums}
    .q-score-values{display:flex;justify-content:space-between;align-items:baseline;gap:.6rem;font-size:.79rem;line-height:1.3}
    .q-score-values b{font-weight:600}
    .q-score-values span{color:var(--sub)}
    .q-score-track{position:relative;height:4px;margin:.32rem 3px 0;border-radius:3px;background:#e0e5eb}
    .q-score-track i{position:absolute;display:block}
    .q-score-band{height:4px;background:#e8c68c}
    .q-score-cut{top:-2px;height:8px;width:2px;background:#936522;transform:translateX(-50%)}
    .q-score-dot{top:-2px;width:8px;height:8px;border-radius:50%;background:var(--accent);transform:translateX(-50%)}
    </style><div class="q-score-comparison" role="group" aria-label="模型分数与判断分界">""" + ''.join(rows) + '</div>')


def render_ruler(record, config, *, compact=False):
    import streamlit as st
    spec = ruler_spec(record.rf_score, record.lr_score, config)
    if compact:
        spec['height'] = 160
        for layer in spec['layer']:
            data = layer.get('data', {}).get('values', [])
            for item in data:
                if 'score' in item:
                    item['label'] = f"{item['score']:.3f} / 分界 {item['threshold']:.2f}"
        spec['layer'].append({'data': spec['layer'][0]['data'],
            'mark': {'type': 'text', 'align': 'right', 'dy': -16, 'fontSize': 12, 'color': '#17314D'},
            'encoding': {'x': {'datum': 1}, 'y': spec['layer'][0]['encoding']['y'],
                         'text': {'field': 'label'}}})
    st.vega_lite_chart(spec, use_container_width=True)
    if compact:
        st.caption('实心点：当前分数 · 橙线：判断分界 · 浅橙段：接近分界')
        return
    st.caption(f"RF {record.rf_score:.6f} / 分界 {config['rf_threshold']:.2f} · "
               f"LR {record.lr_score:.6f} / 分界 {config['lr_threshold']:.2f}。实心点为分数，浅橙段为接近分界。")


def expert_hint_summary(frame, config):
    """Display-only aggregation of existing batch predictions.

    Hint semantics are the frozen engine rules: the rf_anomaly / lr_anomaly
    columns produced during inference are preferred; when a flag column is
    absent, score >= the frozen threshold from the model config is used.
    A record counts toward an expert's denominator only when its score is a
    finite number. Returns per-expert counts, the RF/LR cross-tab on jointly
    valid records, and engine ow_state counts (empty when unavailable).
    No inference is performed and no threshold is created here.
    """
    import pandas as pd

    def _expert(score_col, flag_col, threshold_key):
        if score_col not in frame:
            return None
        scores = pd.to_numeric(frame[score_col], errors='coerce')
        valid = scores.notna() & np.isfinite(scores)
        if flag_col in frame:
            flags = frame[flag_col]
            hints = valid & flags.eq(True)
        else:
            threshold = (config or {}).get(threshold_key)
            if threshold is None:
                return None
            hints = valid & scores.ge(float(threshold))
        return {'valid': int(valid.sum()), 'hints': int(hints.sum()),
                '_hint_mask': hints, '_valid_mask': valid}

    rf = _expert('rf_score', 'rf_anomaly', 'rf_threshold')
    lr = _expert('lr_score', 'lr_anomaly', 'lr_threshold')
    result = {'rf': rf, 'lr': lr, 'cross': None, 'ow_state': {}}
    if rf is not None and lr is not None:
        both = rf['_valid_mask'] & lr['_valid_mask']
        if int(both.sum()):
            rh, lh = rf['_hint_mask'], lr['_hint_mask']
            result['cross'] = {
                'both_valid': int(both.sum()),
                'both_hint': int((both & rh & lh).sum()),
                'rf_only': int((both & rh & ~lh).sum()),
                'lr_only': int((both & ~rh & lh).sum()),
                'neither': int((both & ~rh & ~lh).sum()),
            }
    if 'ow_state' in frame:
        states = frame['ow_state'].dropna()
        states = states[states.astype(str).str.strip() != '']
        if len(states):
            result['ow_state'] = {str(k): int(v) for k, v in states.value_counts().items()}
    for expert in (rf, lr):
        if expert is not None:
            expert.pop('_hint_mask', None)
            expert.pop('_valid_mask', None)
    return result
