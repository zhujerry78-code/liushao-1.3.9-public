"""Incident roster and selected member, built from this input's verified review projection."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from html import escape

import streamlit as st

from .recovery import RecoveryState

GROUPS = ('待复核', '已确认', '待处理', '状态待核验')
COLORS = {'待复核': '#ba8419', '已确认': '#359574', '待处理': '#848894', '状态待核验': '#848894',
          '待补证': '#cb782a', '人工逐条确认': '#277d63', '由事件继承': '#5865ba', '继承失效': '#bb545d'}
SECTIONS = ('属性', '相关证据', '研判历史', '批次处置')


@dataclass(frozen=True)
class Event:
    number: int
    name: str
    group: str
    members: tuple
    counts: dict
    mixed: bool = False

    @property
    def code(self):
        return f'EVT-{self.number:04d}'


def member_status(row):
    state = row.get('authorization_state')
    if state == RecoveryState.DIRECT:
        return '人工逐条确认'
    if state == RecoveryState.INHERITED:
        return '由事件继承'
    if state == RecoveryState.INVALIDATED:
        return '继承失效'
    if state is None:
        return '状态待核验'
    return '待补证' if row.get('review_status') == '待补证' else '待复核'


def event_roster(predictions, reviews, started=()):
    if not {'ow_event', 'ow_state', 'engine_rank'} <= set(predictions):
        return []
    roster = []
    columns = ['ow_event', 'ow_state', 'record_id', 'engine_rank']
    columns.extend(name for name in ('authorization_state', 'review_status') if name in predictions)
    frame = predictions.loc[predictions.ow_event.notna(), columns].sort_values('engine_rank', kind='stable')
    for number, rows in frame.groupby('ow_event', sort=True):
        members = tuple(rows.record_id)
        states = Counter(member_status(row) for row in rows.to_dict('records'))
        conclusions = set(rows['review_status']) & {'确认异常', '确认正常'} if 'review_status' in rows else set()
        mixed = len(conclusions) > 1
        if '状态待核验' in states:
            status = '状态待核验'
        elif mixed:
            status = '待复核'
        elif states['人工逐条确认'] + states['由事件继承'] == len(members):
            status = '已确认'
        elif number in started or states['待补证'] or states['继承失效'] or any(rid in reviews for rid in members):
            status = '待复核'
        else:
            status = '待处理'
        system_state = str(rows.iloc[0].ow_state)
        name = {'模糊': '证据模糊事件', '新异': '新异流量事件'}.get(system_state, f'{system_state}流量事件')
        roster.append(Event(int(number), name, status, members, dict(states), mixed))
    return roster


def matching_events(events, status, keyword):
    keyword = keyword.strip().casefold()
    return [event for event in events if (status == '全部' or event.group == status) and
            (not keyword or keyword in f'{event.code} {event.name} {event.group}'.casefold())]


def evidence_progress(event):
    total = len(event.members)
    direct = event.counts.get('人工逐条确认', 0)
    states = ' · '.join(f'{label} {count}' for label, count in event.counts.items())
    st.html(f'<div class="event-progress"><div><span>人工逐条确认</span><b>{direct} / {total} 条</b></div>'
            f'<progress aria-label="人工逐条确认" value="{direct}" max="{total}"></progress>'
            f'<small>{escape(states)}</small></div>')


def _narrow_navigation(prefix, returned_section=None):
    """Restore the row focus and scroll after the mobile list and detail swap."""
    from streamlit.components.v1 import html

    html('''<script>
    (() => {
      let host, doc;
      try { host = window.parent; doc = host.document; } catch (_) { return }
      const prefix = "PREFIX";
      const narrow = () => doc.querySelector('.st-key-event_grid')?.getBoundingClientRect().width <= 880;
      const state = host.__liushaoEventPane || (host.__liushaoEventPane = {});
      if (state.cleanup) state.cleanup();
      if (state.batch !== prefix) { state.batch = prefix; state.action = null; }
      if ("RETURN_SECTION") { state.section = "RETURN_SECTION"; state.action = 'section'; }
      function restore() {
        const mode = state.action;
        if (!mode) return;
        let settled = 0, attempts = 0;
        function position() {
          if (state.action !== mode || ++attempts > 400) return;
          const main = doc.querySelector('[data-testid="stMain"]');
          const app = doc.querySelector('[data-testid="stApp"]');
          const list = '.st-key-event_list_mode ';
          const button = mode === 'section' ? doc.querySelector('.event-section-title[data-section="'+state.section+'"]') :
              mode === 'wide-detail' ? doc.querySelector('.st-key-event_details .event-detail-head') :
              mode === 'detail' ? doc.querySelector('.st-key-event_detail_mode .st-key-event_back button') :
              (state.row && doc.querySelector(list+'.'+state.row+' button')) ||
              doc.querySelector(list+'.st-key-event_list button[kind=primary]');
          if (main && app && button && button.offsetWidth &&
              (mode !== 'wide-detail' || doc.querySelector('.'+state.row+' button[kind=primary]')) &&
              app.getAttribute('data-test-script-state') === 'notRunning') {
            if (++settled >= 3) {
              if (mode !== 'wide-detail') button.focus({preventScroll:true});
              if (mode === 'section') {
                const rect = button.getBoundingClientRect();
                const bottom = doc.querySelector('.statusbar')?.getBoundingClientRect().top || host.innerHeight;
                if (rect.top < 0 || rect.bottom + 70 > bottom) button.scrollIntoView({block:'center'});
              } else if (mode === 'wide-detail') {
                if (button.getBoundingClientRect().top < 0) button.scrollIntoView({block:'start'});
              } else { main.scrollTop = mode === 'detail' ? 0 : (state.top || 0); }
              state.action = null;
              return;
            }
          } else { settled = 0; }
          host.requestAnimationFrame(position);
        }
        host.requestAnimationFrame(position);
      }
      state.listener = event => {
        const button = event.target.closest('button');
        if (!button) return;
        if (button.closest('.st-key-event_sections,.st-key-event_batch')) {
          const labels = {'属性':'属性','查看相关证据':'相关证据','研判历史':'研判历史','批次确认与撤回':'批次处置'};
          state.section = labels[button.innerText.trim()];
          if (state.section) { state.action = 'section'; restore(); }
          return;
        }
        const main = doc.querySelector('[data-testid="stMain"]');
        const row = button.closest('[class*="_row_"]');
        if (main && row && button.closest('.st-key-event_list')) {
          const key = Array.from(row.classList).find(c => c.startsWith('st-key-'+prefix+'_row_'));
          if (!key) return;
          state.row = key; state.top = main.scrollTop; state.action = narrow() ? 'detail' : 'wide-detail';
        } else if (button.closest('.st-key-event_back')) { state.action = 'list'; }
        else { return; }
        restore();
      };
      doc.addEventListener('click', state.listener, true);
      const divider = '.event-divider';
      const root = doc.documentElement;
      function limits() {
        const width = doc.querySelector('.st-key-event_grid')?.getBoundingClientRect().width;
        if (!width || narrow()) return [50,72];
        return [Math.max(50, Math.ceil(48000/width)), Math.min(72, Math.floor(100-28500/width))];
      }
      function getWidth() {
        const [low,high] = limits();
        return Math.min(high, Math.max(low, state.width ?? 64));
      }
      function size(value) {
        if (value == null) {
          delete state.width;
        } else {
          const [low,high] = limits();
          state.width = Math.min(high, Math.max(low, Math.round(value)));
        }
        sync();
      }
      function pointerDown(e) {
        const handle = e.target.closest(divider);
        if (!handle || e.button !== 0) return;
        const grid = handle.closest('.st-key-event_grid').getBoundingClientRect();
        handle.focus({preventScroll:true});
        handle.setPointerCapture(e.pointerId);
        const move = ev => size((grid.right - ev.clientX) * 100 / grid.width);
        const finish = () => {
          handle.removeEventListener('pointermove',move);
          handle.removeEventListener('lostpointercapture',finish);
        };
        handle.addEventListener('pointermove',move);
        handle.addEventListener('lostpointercapture',finish);
        e.preventDefault();
      }
      function keyDown(e) {
        if (e.key === 'Escape' && e.target.matches('.event-section-title')) {
          const section = e.target.dataset.section;
          const button = section === '批次处置' ? doc.querySelector('.st-key-event_batch button') :
              doc.querySelector('.st-key-event_sections button[kind="primary"]');
          if (button) {
            button.focus({preventScroll:true});
            button.scrollIntoView({block:'nearest'});
            e.preventDefault();
          }
          return;
        }
        if (!e.target.closest(divider)) return;
        const step = e.shiftKey ? 5 : 2;
        if (e.key === 'ArrowLeft') size(getWidth()+step);
        else if (e.key === 'ArrowRight') size(getWidth()-step);
        else if (e.key === 'Home' || e.key === 'Enter') size(null);
        else return;
        e.preventDefault();
      }
      function reset(e) { if (e.target.closest(divider)) size(null); }
      doc.addEventListener('pointerdown',pointerDown,true);
      doc.addEventListener('keydown',keyDown,true);
      doc.addEventListener('dblclick',reset,true);
      const sync = () => {
        const handle = doc.querySelector(divider);
        if (!handle) return;
        const [low,high] = limits(), width = getWidth();
        if (state.width == null) root.style.removeProperty('--liushao-event-detail');
        else root.style.setProperty('--liushao-event-detail', width+'%');
        handle.setAttribute('aria-valuemin', String(low));
        handle.setAttribute('aria-valuemax', String(high));
        handle.setAttribute('aria-valuenow', String(width));
        doc.querySelectorAll('.st-key-event_rows button,.st-key-event_sections button').forEach(button =>
          button.setAttribute('aria-pressed', String(button.getAttribute('kind') === 'primary')));
        doc.querySelector('.st-key-event_batch button')?.setAttribute('aria-pressed',
          String(Boolean(doc.querySelector('.event-section-title[data-section="批次处置"]'))));
      };
      const observer = new host.MutationObserver(sync);
      observer.observe(doc.body,{childList:true,subtree:true});
      host.addEventListener('resize',sync);
      state.cleanup = () => {
        doc.removeEventListener('click',state.listener,true);
        doc.removeEventListener('pointerdown',pointerDown,true);
        doc.removeEventListener('keydown',keyDown,true);
        doc.removeEventListener('dblclick',reset,true);
        host.removeEventListener('resize',sync);
        observer.disconnect();
      };
      const cleanup = state.cleanup;
      window.addEventListener('pagehide', () => { if (state.cleanup === cleanup) cleanup(); }, {once:true});
      sync();
      restore();
    })();
    </script>'''.replace('PREFIX', prefix).replace('RETURN_SECTION', returned_section if returned_section in SECTIONS else ''),
         height=0, width=0, scrolling=False)

def render_page(events, *, active_key, target, section, section_callback, record_callback,
                open_queue, pred, physical, config, locked, epoch, detail_callback, event_basis=None):
    """All changes to the selected event/member go through the host's draft guard."""
    from app_score_charts import render_ruler

    token = sha256(active_key.encode()).hexdigest()[:18]
    prefix = f'ev_{token}'
    st.html('''<style>
    .st-key-event_grid{container:event-area / inline-size}
    .st-key-event_grid [data-testid="stVerticalBlock"]{gap:.42rem}
    .st-key-event_list,.st-key-event_details{container-type:inline-size}
    .st-key-event_list{padding:0 .65rem 0 .2rem}
    .st-key-event_details{position:relative;padding:0 .25rem 0 1.1rem;border-left:1px solid var(--line)}
    .event-list-title,.event-title{font-size:1.17rem!important;line-height:1.4!important;
        font-weight:650!important;padding:0!important;margin:0!important;color:var(--ink)}
    .event-title{font-size:1.08rem!important;overflow-wrap:anywhere}
    .event-summary,.event-meta,.event-code,.event-section-title small,.event-progress small{color:var(--sub)}
    .event-summary,.event-meta,.event-progress small{font-size:.76rem;line-height:1.45}
    .event-list-head,.event-detail-head{padding:.75rem 0 .3rem}
    .event-list-head{display:flex;gap:.4rem 1rem;align-items:baseline;justify-content:space-between;flex-wrap:wrap}
    .event-head-meta{display:flex;align-items:center;justify-content:space-between;gap:.4rem;margin-bottom:.3rem}
    .event-code{font:600 .76rem/1.4 Consolas,"Microsoft YaHei UI",monospace;letter-spacing:.035em}
    .event-status{font-size:.75rem;font-weight:550;white-space:nowrap}
    .event-status::before,.event-bucket i{display:inline-block;content:"";width:.48rem;height:.48rem;
        background:var(--event-dot);border-radius:50%;margin-right:.35rem;vertical-align:.055em}
    .event-bucket{display:flex;align-items:center;gap:.2rem;font-size:.78rem;font-weight:600;
       background:var(--panel);padding:.38rem .6rem;border-block:1px solid var(--line)}
    .event-bucket span{font-size:.73rem;color:var(--sub);font-weight:500;margin-left:.2rem}
    .st-key-event_rows,.st-key-event_rows [data-testid="stVerticalBlock"]{gap:0!important}
    .st-key-event_group_0{--event-dot:#b78823}.st-key-event_group_1{--event-dot:#39856e}
    .st-key-event_group_2,.st-key-event_group_3{--event-dot:#89909c}
    .st-key-event_list [class*="_row_"] button{border:0!important;border-bottom:1px solid var(--line)!important;
       min-height:2.72rem;border-radius:0;box-shadow:none;gap:.45rem;padding:.38rem .65rem;
       justify-content:flex-start;text-align:left;background:var(--paper);color:var(--ink)}
    .st-key-event_list [class*="_row_"] button:hover{background:var(--panel)}
    .st-key-event_list [class*="_row_"] button[kind=primary]{background:var(--pick);box-shadow:inset 3px 0 var(--accent)}
    .st-key-event_list [class*="_row_"] button::before{content:"";flex:0 0 .43rem;height:.43rem;
       border-radius:50%;background:var(--event-dot)}
    .st-key-event_list [class*="_row_"] button::after{content:"";width:.36rem;height:.36rem;
       border-top:1.5px solid var(--sub);border-right:1.5px solid var(--sub);transform:rotate(45deg);margin:0 .1rem 0 .2rem}
    .st-key-event_list [class*="_row_"] button>div{flex:1;min-width:0}
    .st-key-event_list [class*="_row_"] button p{display:grid;align-items:center;grid-template-columns:minmax(0,1fr) 6.8em 4.1em;
       gap:.35rem;font-size:.87rem;color:var(--ink)!important;line-height:1.4;white-space:normal}
    .st-key-event_list [class*="_row_"] button p strong{font-weight:550}
    .st-key-event_list [class*="_row_"] button p code{background:none;color:var(--sub);font:500 .72rem/1.4 Consolas,monospace;padding:0}
    .st-key-event_list [class*="_row_"] button p em{font-style:normal;font-size:.75rem;color:var(--sub);text-align:right}
    .st-key-event_list [class*="_row_"] button p>span{grid-column:1 / -1;font-size:.74rem;line-height:1.25;
        text-align:left;color:var(--sub)!important}
    .event-progress{padding:.48rem 0 .2rem}
    .event-progress>div,.event-gate{display:flex;align-items:baseline;justify-content:space-between;gap:.6rem;font-size:.8rem}
    .event-progress>div span,.event-gate>span{color:var(--sub)}
    .event-progress b,.event-gate b{font-weight:550;color:var(--ink)}
    .event-progress progress{width:100%;height:4px;display:block;margin:.36rem 0 .27rem;border:0;border-radius:4px;
       appearance:none;background:var(--line);overflow:hidden}
    .event-progress progress::-webkit-progress-bar{background:var(--line)}
    .event-progress progress::-webkit-progress-value{background:var(--good)}
    .event-gate{padding:0 0 .36rem}
    .event-section-title{font-size:.91rem!important;font-weight:650!important;line-height:1.4!important;
       margin:0!important;padding:.2rem 0!important;scroll-margin-block:1rem}
    .event-section-title:focus-visible{outline:2px solid var(--accent);outline-offset:2px;border-radius:3px}
    .st-key-event_filters [data-testid="stHorizontalBlock"],
    .st-key-event_actions [data-testid="stHorizontalBlock"],.st-key-event_sections [data-testid="stHorizontalBlock"],
    .st-key-event_pagination [data-testid="stHorizontalBlock"]{flex-flow:row nowrap!important;gap:.35rem}
    .st-key-event_filters [data-testid=stColumn],.st-key-event_actions [data-testid=stColumn],
    .st-key-event_sections [data-testid=stColumn],.st-key-event_pagination [data-testid=stColumn]{min-width:0!important}
    .st-key-event_filters [data-baseweb="select"]>div,.st-key-event_filters [data-baseweb="input"]{min-height:2.12rem}
    .st-key-event_filters [data-baseweb="select"] div{font-size:.82rem}
    .st-key-event_filters input{height:2.12rem;font-size:.81rem}
    .event-member-state{display:flex;flex-wrap:wrap;gap:.1rem .9rem}
    .event-preview{margin:.2rem 0;font-size:.83rem}
    .event-preview>div{display:flex;align-items:baseline;justify-content:space-between;gap:.6rem;
       padding:.38rem 0;border-bottom:1px solid var(--line)}
    .event-preview dt{color:var(--sub)}.event-preview dd{margin:0;text-align:right;color:var(--ink)}
    .st-key-event_sections{border-bottom:1px solid var(--line);margin:.2rem 0}
    .st-key-event_sections button{min-height:2.24rem;border:0!important;border-radius:0;background:transparent!important}
    .st-key-event_sections button p{font-size:.8rem;font-weight:500;white-space:nowrap}
    .st-key-event_sections button[kind=primary]{box-shadow:inset 0 -2px var(--accent)}
    .st-key-event_sections button[kind=primary] p{color:var(--accent)!important;font-weight:650}
    .st-key-event_actions button{padding:.33rem .25rem;min-height:2.25rem}
    .st-key-event_actions button p,.st-key-event_pagination button p{font-size:.82rem}
    .st-key-event_batch button[aria-pressed=true]{color:var(--accent);border-color:var(--accent)}
    .st-key-event_pagination button{min-height:2rem}
    .st-key-event_back button{min-height:2rem;font-size:.78rem}
    .event-divider{position:absolute;left:-6px;top:0;bottom:0;width:11px;z-index:3;cursor:col-resize;touch-action:none}
    .st-key-event_details [data-testid=stElementContainer]:has(.event-divider),
    .st-key-event_details [data-testid=stHtml]:has(.event-divider){display:contents!important}
    .event-divider::before{content:"";position:absolute;width:3px;height:25px;border-radius:4px;background:var(--line);
        left:4px;top:58px}
    .event-divider:hover::before,.event-divider:focus-visible::before{background:var(--accent)}
    .event-divider:focus-visible{outline:2px solid var(--accent);outline-offset:-3px}
    @container(max-width:390px){
      .st-key-event_list [class*="_row_"] button p{grid-template-columns:minmax(0,1fr) auto;row-gap:.06rem}
      .st-key-event_list [class*="_row_"] button p strong{grid-column:1;grid-row:1}
      .st-key-event_list [class*="_row_"] button p code{grid-column:1;grid-row:2}
      .st-key-event_list [class*="_row_"] button p em{grid-column:2;grid-row:1}
      .st-key-event_list [class*="_row_"] button p>span{grid-column:2;grid-row:2;text-align:right}
      .event-list-head{display:block}.event-list-head .event-summary{display:block;margin-top:.2rem}
    }
    @container event-area (width > 880px){
      .st-key-event_grid [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-event_details){flex-wrap:nowrap!important;gap:0!important}
      .st-key-event_grid [data-testid="stColumn"]:has(.st-key-event_details){min-width:0!important;flex:0 0 var(--liushao-event-detail,40%)!important}
      .st-key-event_grid [data-testid="stColumn"]:has(.st-key-event_list){min-width:0!important;flex:1 1 0!important;
        align-self:flex-start;position:sticky;top:.5rem;max-height:calc(100dvh - 3.5rem);overflow:auto;scrollbar-width:thin}
      .st-key-event_back{display:none}}
    @container event-area (max-width:880px){
      .st-key-event_list,.st-key-event_details{border:0;padding:0}.event-divider{display:none}
      .st-key-event_grid [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] .st-key-event_details){display:block}
      .st-key-event_grid [data-testid="stColumn"]:has(.st-key-event_details){display:none}
      .st-key-event_detail_mode .st-key-event_grid [data-testid="stColumn"]:has(.st-key-event_details){display:block;width:100%!important;flex:1 1 100%!important}
      .st-key-event_detail_mode .st-key-event_grid [data-testid="stColumn"]:has(.st-key-event_list){display:none}
      .st-key-event_list_mode .st-key-event_grid [data-testid="stColumn"]:has(.st-key-event_list){width:100%!important;flex:1 1 100%!important}
      .event-list-head,.event-detail-head{padding-top:.25rem}}
    </style>''')
    if not events:
        st.header('事件研判')
        st.info('当前批次没有可聚合的疑似事件。可以在复核队列中逐条查看系统提示与证据。')
        st.button('打开复核队列', key=f'{prefix}_queue', on_click=open_queue)
        return
    chosen = next((e for e in events if target and target[:2] == (active_key, e.number)), None)
    mobile_detail = bool(chosen and st.session_state.get('_ev_mobile_detail'))
    returned_section = st.session_state.pop('_focus_return_event', None)
    _narrow_navigation(prefix, returned_section)
    lines = dict(zip(pred.record_id, physical))
    with st.container(key='event_detail_mode' if mobile_detail else 'event_list_mode'):
        with st.container(key='event_grid'):
            left, right = st.columns([58, 42], gap='small')
            with left, st.container(key='event_list'):
                heading = st.empty()
                def reset_page():
                    st.session_state[f'{prefix}_page'] = 0
                    st.session_state['_ev_mobile_detail'] = False
                    st.session_state[f'{prefix}_saved_filters'] = (
                        st.session_state[f'{prefix}_status'], st.session_state[f'{prefix}_query'])

                saved = st.session_state.get(f'{prefix}_saved_filters', ('全部', ''))
                if returned_section and chosen and saved[0] not in ('全部', chosen.group):
                    saved = ('全部', saved[1])
                    st.session_state[f'{prefix}_saved_filters'] = saved
                    st.session_state[f'{prefix}_status'] = saved[0]
                    st.caption('事件的处理状态已变化，已切换为全部状态以显示该事件。')
                st.session_state.setdefault(f'{prefix}_status', saved[0])
                st.session_state.setdefault(f'{prefix}_query', saved[1])
                with st.container(key='event_filters'):
                    filters, search = st.columns([1, 1.05], gap='small')
                    status = filters.selectbox('事件状态', ('全部', *GROUPS), key=f'{prefix}_status',
                                               format_func=lambda value: '全部状态' if value == '全部' else value,
                                               on_change=reset_page, disabled=locked, label_visibility='collapsed')
                    term = search.text_input('搜索事件', placeholder='编号或名称', key=f'{prefix}_query',
                                             icon=':material/search:', on_change=reset_page, disabled=locked,
                                             label_visibility='collapsed')
                if status != '全部' or term:
                    def clear_filters():
                        st.session_state[f'{prefix}_status'] = '全部'
                        st.session_state[f'{prefix}_query'] = ''
                        reset_page()
                    st.button('清除事件筛选', key=f'{prefix}_clear', on_click=clear_filters, disabled=locked)
                visible = matching_events(events, status, term)
                visible.sort(key=lambda e: (GROUPS.index(e.group), e.number))
                heading.html(f'<div class="event-list-head"><h2 class="event-list-title">事件研判</h2>'
                             f'<span class="event-summary">{len(visible)} 个事件 · 共 '
                             f'{sum(len(e.members) for e in visible):,} 条成员记录</span></div>')
                if not visible:
                    st.info('没有符合条件的事件。请调整状态或清除搜索。')
                    return
                selected = chosen if chosen in visible else visible[0]
                explicit = chosen == selected
                page_key = f'{prefix}_page'
                page = min(st.session_state.get(page_key, 0), (len(visible) - 1) // 10)
                if st.session_state.pop('_ev_reveal', False) and explicit:
                    page = visible.index(chosen) // 10
                st.session_state[page_key] = page
                shown = visible[page*10:page*10+10]
                if selected not in shown:
                    selected, explicit = shown[0], False
                with st.container(key='event_rows'):
                    for group_index, group in enumerate(GROUPS):
                        matches = [event for event in shown if event.group == group]
                        if not matches:
                            continue
                        with st.container(key=f'event_group_{group_index}'):
                            st.html(f'<div class="event-bucket"><i aria-hidden="true"></i>{group}'
                                    f'<span>{sum(e.group == group for e in visible)}</span></div>')
                            for event in matches:
                                first = lines.get(event.members[0])
                                locator = f' :gray[排序首条 · CSV 第 {first} 行]' if first is not None else ''
                                st.button(f'**{event.name}** `{event.code}` *{len(event.members):,} 条成员*{locator}',
                                          key=f'{prefix}_row_{event.number}',
                                          type='primary' if event == selected else 'secondary',
                                          on_click=section_callback, args=(event.number, None, '属性'),
                                          disabled=locked, width='stretch')
                if len(visible) > 10:
                    def jump(offset):
                        st.session_state[page_key] = page + offset
                        st.session_state['_ev_mobile_detail'] = False
                    with st.container(key='event_pagination'):
                        a, indicator, b = st.columns([1,1,1], vertical_alignment='center')
                        a.button('上一页', key=f'{prefix}_prev', disabled=locked or page == 0,
                                 on_click=jump, args=(-1,), width='stretch')
                        indicator.html(f'<div class="event-meta" style="text-align:center">'
                                       f'{page+1} / {(len(visible)+9)//10} 页</div>')
                        b.button('下一页', key=f'{prefix}_next', disabled=locked or (page+1)*10 >= len(visible),
                                 on_click=jump, args=(1,), width='stretch')
            with right, st.container(key='event_details'):
                event = selected
                rid = target[2] if explicit and target[2] in event.members else event.members[0]
                scope = f'{prefix}_{event.number}'
                current_section = section if explicit else '属性'
                with st.container(key='event_back'):
                    st.button('‹ 返回事件列表', key=f'{scope}_back', disabled=locked,
                              on_click=lambda: st.session_state.update(_ev_mobile_detail=False))
                st.html('<div class="event-divider" role="separator" tabindex="0" aria-orientation="vertical" '
                        'aria-label="调整研判详情宽度" aria-valuemin="50" aria-valuemax="72" aria-valuenow="64" '
                        'title="拖动或使用左右方向键调整详情宽度，双击或按 Home 恢复默认"></div>')
                st.html(f'<div class="event-detail-head"><div class="event-head-meta"><span class="event-code">'
                        f'{event.code}</span><span class="event-status" style="--event-dot:{COLORS[event.group]}">'
                        f'{escape(event.group)}</span></div><h2 class="event-title">{escape(event.name)}</h2></div>')
                evidence_progress(event)
                gate_slot = st.empty()
                if event.mixed:
                    st.warning('成员的有效结论同时包含确认正常与确认异常，请核查分歧。')
                if event.counts.get('继承失效'):
                    st.warning(f'{event.counts["继承失效"]} 条继承已失效，请核对反证及原授权批次。')
                with st.container(key='event_actions'):
                    show_member_action = current_section in ('属性', '研判历史')
                    batch = st
                    if show_member_action:
                        enter, batch = st.columns([1, 1], gap='small')
                        enter.button('进入研判', type='primary', key=f'{scope}_enter', on_click=record_callback,
                                     args=(rid,), disabled=locked, width='stretch', help=f'核查输入 CSV 第 {lines[rid]} 行')
                    with batch.container(key='event_batch'):
                        st.button('批次确认与撤回', key=f'{scope}_批次处置', on_click=section_callback,
                                  args=(event.number, rid, '批次处置'), disabled=locked,
                                  width='stretch' if show_member_action else 'content')
                with st.container(key='event_sections'):
                    sections = (('属性', '属性'), ('查看相关证据', '相关证据'), ('研判历史', '研判历史'))
                    for col, (label, destination) in zip(st.columns([.8,1.4,1.1], gap='small'), sections):
                        col.button(label, key=f'{scope}_{destination}', disabled=locked, on_click=section_callback,
                                   type='primary' if destination == current_section else 'secondary',
                                   args=(event.number, rid, destination), width='stretch')
                title = '研判属性' if current_section == '属性' else current_section
                st.html(f'<h3 tabindex="-1" class="event-section-title" data-section="{escape(current_section)}" '
                        f'aria-keyshortcuts="Escape" title="按 Esc 返回当前分区入口">'
                        f'{escape(title)}</h3>')
                if current_section == '属性':
                    basis = event_basis(event) if event_basis else '引擎状态与尾部特征相同'
                    st.html(f'<div class="event-meta">归并依据　{escape(basis)}</div>')
                    if not explicit:
                        position, csv_line = next((i, line) for i, (member, line) in
                                                  enumerate(zip(pred.record_id, physical)) if member == rid)
                        preview_row = pred.iloc[position]
                        st.html(f'<dl class="event-preview"><div><dt>当前成员</dt>'
                                f'<dd>CSV 第 {escape(str(csv_line))} 行</dd></div><div><dt>处理状态</dt>'
                                f'<dd>{escape(member_status(preview_row))}</dd></div></dl>')
                if explicit and current_section == '属性':
                    data = pred.set_index('record_id', drop=False)
                    member_key = f'{scope}_member_{epoch}'
                    st.selectbox('当前事件成员', event.members, index=event.members.index(rid),
                                 format_func=lambda item: f'CSV 第 {lines[item]} 行',
                                 key=member_key, disabled=locked,
                                 on_change=lambda: section_callback(event.number, st.session_state[member_key], current_section))
                    record = data.loc[rid]
                    source = member_status(record)
                    judgment = str(record.get('review_status', '待核验'))
                    judgment_text = f'<span>当前判断　{escape(judgment)}</span>' if judgment != source else ''
                    st.html(f'<div class="event-meta event-member-state"><span>成员状态　{escape(source)}</span>'
                            f'{judgment_text}</div>')
                    st.markdown('**排序分数**')
                    render_ruler(record, config, compact=True)
                    st.caption(f'输入 CSV 第 {lines[rid]} 行 · RF {record.rf_score:.4f} / 分界 {config["rf_threshold"]:.2f}'
                               f' · LR {record.lr_score:.4f} / 分界 {config["lr_threshold"]:.2f}')
                    with st.expander('追溯标识'):
                        st.code(rid, language=None)
                detail_callback(event, rid, current_section, gate_slot)
