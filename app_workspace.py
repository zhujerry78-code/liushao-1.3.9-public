"""Presentation of the local Streamlit workspace."""
from html import escape

import streamlit as st
from streamlit.components.v1 import html as component_html

from app_navigation import PAGES


def styles():
    st.html("""<style>
    @font-face{font-family:"Inter";src:url("app/static/fonts/Inter-400.woff2") format("woff2");font-weight:400;font-display:swap}
    @font-face{font-family:"Inter";src:url("app/static/fonts/Inter-500.woff2") format("woff2");font-weight:500;font-display:swap}
    @font-face{font-family:"Inter";src:url("app/static/fonts/Inter-600.woff2") format("woff2");font-weight:600;font-display:swap}
    @font-face{font-family:"Noto Sans SC";src:url("app/static/fonts/NotoSansSC-VF.ttf") format("truetype");font-weight:100 900;font-display:swap}
    :root,.stApp {color-scheme:light;--paper:#ffffff;--panel:#f7f8fa;--ink:#24252c;
        --sub:#777982;--line:#e6e7eb;--accent:#5e6ad2;--pick:#edeef2;--good:#419779;
        --warn:#b28b38;--bad:#c75c59;}
    .stApp {background:var(--paper);color:var(--ink);--ui-font:"Inter","Noto Sans SC","Segoe UI","Microsoft YaHei UI","Microsoft YaHei",sans-serif;
       font-family:var(--ui-font);font-size:14px;line-height:1.55;
       --st-text-color:var(--ink);--st-background-color:var(--paper);--st-secondary-background-color:var(--panel);
       --st-border-color:var(--line);--st-primary-color:var(--accent);}
    [data-testid=stHeader],[data-testid=stToolbar],[data-testid=stDecoration]{display:none}
    .block-container{max-width:1800px;padding:0 1rem 3rem}
    [data-testid=stMainBlockContainer] > [data-testid=stVerticalBlock]{gap:.5rem}
    h1,h2,h3,p,label{color:var(--ink)} h1{font-size:1.25rem!important;font-weight:600!important} h2{font-size:1.05rem!important;font-weight:600!important}
    h3{font-size:.95rem!important;font-weight:600!important} [data-testid=stCaptionContainer], [data-testid=stCaptionContainer] p{color:var(--sub);opacity:1!important}
    .stApp :is(h1,h2,h3,h4,p,label,input,textarea,select,[data-testid=stMetricValue]),.vega-embed text{font-family:var(--ui-font)}
    [data-testid=stMetricValue],.vega-embed text,.statusbar,.st-key-event_list,.st-key-workbody table{font-variant-numeric:tabular-nums}
    button:focus-visible,input:focus-visible,summary:focus-visible{outline:2px solid var(--accent)!important;outline-offset:2px}
    .st-key-brandbar{background:var(--panel);border-bottom:1px solid var(--line);padding:.43rem .8rem}
    .brand{display:flex;align-items:center;gap:.5rem;min-height:35px;font-size:1.08rem;color:var(--ink)}
    .brand small{font-size:.72rem;color:var(--sub);letter-spacing:.065em}
    .brand img{width:22px;height:24px;color:var(--accent)}
    .st-key-brandbar [data-testid=stPopoverButton]{background:transparent;border:0;color:var(--sub);width:100%}
    .st-key-worknav{background:var(--panel);border-right:1px solid var(--line);padding:.4rem .46rem}
    .st-key-worknav,.st-key-worknav [data-testid=stVerticalBlock]{gap:.12rem!important}
    .st-key-worknav button{background:transparent;border:0;border-radius:4px;justify-content:flex-start;padding:.3rem .45rem;min-height:2.05rem}
    .st-key-worknav button:hover{background:var(--pick)}
    .st-key-worknav button[kind=primary]{background:var(--pick);color:var(--ink);box-shadow:inset 3px 0 var(--accent)}
    .st-key-worknav button p{color:var(--ink)}
    .st-key-nav_search button:after{content:"Ctrl K";margin-left:auto;font-size:.67rem;color:var(--sub);line-height:1.25;border:1px solid var(--line);border-radius:3px;padding:.05rem .22rem}
    .st-key-worknav [data-testid=stCaptionContainer]{padding:.45rem .45rem 0;font-size:.73rem}
    [data-testid=stBaseButton-primary]{background:var(--accent);border-color:var(--accent);border-radius:5px}
    [data-testid=stBaseButton-primary] p{color:var(--paper)}
    [data-testid=stBaseButton-secondary],[data-testid=stBaseButton-secondaryFormSubmit],
    [data-testid=stDownloadButton] button,[data-testid=stPopoverButton]{background:var(--paper);color:var(--ink);border-color:var(--line)}
    button:disabled{opacity:.75}
    .stApp button[kind=primary]:disabled,.stApp button[data-testid=stBaseButton-primary]:disabled{background:#eceef2;border-color:#d8dce3;opacity:1}
    .stApp button[kind=primary]:disabled p,.stApp button[data-testid=stBaseButton-primary]:disabled p{color:#525b69}
    [data-testid=stText], [data-testid=stText] pre,[data-testid=stText] div,
    [data-testid=stCode] pre,[data-testid=stMetricValue]{color:var(--ink)!important}
    [data-testid=stCaptionContainer] p,.st-key-worknav [data-testid=stCaptionContainer] p{color:var(--sub)!important;opacity:1}
    [data-testid=stExpander], [data-testid=stMetric]{border-color:var(--line)}
    [data-testid=stPopoverBody], [role=listbox], [role=option]{background:var(--panel);color:var(--ink)}
    [role=option]:hover,[role=option][aria-selected=true]{background:var(--pick)}
    .vega-embed .role-axis-label text,.vega-embed .role-axis-title text,.vega-embed .role-legend-label text{fill:var(--ink)}
    .vega-embed .role-axis-domain path,.vega-embed .role-axis-tick line{stroke:var(--sub)}
    [data-testid=stMetricValue]{font-size:1.7rem}
    [data-baseweb=input],[data-baseweb=textarea],[data-baseweb=select]>div{background:var(--panel);color:var(--ink)}
    input,textarea{color:var(--ink)!important;-webkit-text-fill-color:var(--ink)}
    .statusbar{position:fixed;z-index:8;bottom:0;left:0;right:0;padding:.33rem 1.4rem;background:var(--panel);
        border-top:1px solid var(--line);color:var(--sub);font-size:.74rem;display:flex;justify-content:space-between}
    .statusbar b{color:var(--good);font-weight:600}
    .st-key-workbody{min-width:0}
    .st-key-first_actions,.st-key-overview_actions{gap:.5rem!important}
    .st-key-overview_stats{border-block:1px solid var(--line);padding:.65rem 0}
    .st-key-overview_stats [data-testid=stMetric]{padding:0}
    .st-key-overview_stats [data-testid=stMetricLabel] p{color:var(--sub)}
    .local-data{margin:.2rem 0 .6rem;width:100%;border-collapse:collapse;font-size:.89rem}
    .local-data :is(th,td){text-align:left;padding:.52rem .35rem;border-bottom:1px solid var(--line);overflow-wrap:anywhere}
    .local-data th{font-weight:600}.local-data td{color:var(--sub)}
    .st-key-shell_shortcuts{height:0;min-height:0;overflow:hidden;margin:0;padding:0}
    @media(min-width:621px){
      .st-key-shell [data-testid=stHorizontalBlock]:has(> [data-testid=stColumn] .st-key-worknav){
        flex-flow:row nowrap!important;gap:.8rem}
      .st-key-shell [data-testid=stColumn]:has(.st-key-worknav){flex:0 0 208px!important;min-width:0!important;width:auto!important}
      .st-key-shell [data-testid=stColumn]:has(.st-key-workbody){flex:1 1 0%!important;min-width:0!important;width:auto!important}
      .st-key-worknav{min-height:calc(100vh - 105px)}
    }
    @media(max-width:800px){.block-container{padding:.05rem .5rem 2.8rem}}
    @media(max-width:820px){.st-key-nav_search button:after{display:none}}
    @media(min-width:621px) and (max-width:820px){
      .st-key-shell [data-testid=stColumn]:has(.st-key-worknav){flex-basis:130px!important}
      .st-key-worknav{padding-inline:.2rem}
      .st-key-worknav button{padding-inline:.22rem;gap:.24rem}
      .st-key-worknav button p{white-space:nowrap;font-size:.88rem}
    }
    .st-key-brandbar [data-testid=stHorizontalBlock]{flex-flow:row nowrap!important;gap:.3rem}
    .st-key-brandbar [data-testid=stColumn]{min-width:0!important}
    .st-key-brandbar [data-testid=stColumn]:first-child{flex:1!important}
    .st-key-brandbar [data-testid=stColumn]:last-child{flex:0 0 112px!important}
    @media(max-width:620px){.st-key-shell [data-testid=stColumn]:has(.st-key-worknav){display:none}
      .st-key-shell [data-testid=stColumn]:has(.st-key-workbody){width:100%!important;flex:1 1 100%!important}}
    </style>""")


def local_paths(items):
    rows = ''.join(f'<tr><th scope="row">{escape(label)}</th><td>{escape(path.name)}</td></tr>'
                   for label, path in items)
    st.html(f'<table class="local-data" aria-label="本机文件及用途"><tbody>{rows}</tbody></table>')
    with st.expander('查看和复制完整路径'):
        for label, path in items:
            st.caption(label)
            st.code(str(path), language=None, wrap_lines=True)


def search_shortcuts(on_search_page):
    # Drive the same Streamlit buttons as pointer navigation, including draft updates.
    with st.container(key='shell_shortcuts'):
        component_html("""<script>
        (() => {
          let win; try {win=window.parent; void win.document;} catch (_) {return;}
          const doc=win.document, field=()=>doc.querySelector('.st-key-global_search_query input');
          const back=()=>doc.querySelector('.st-key-search_back button');
          const nav=()=>doc.querySelector('.st-key-nav_search button');
          const state=win.__liushaoSearchKeys ||= {};
          if(state.listener) doc.removeEventListener('keydown',state.listener,true);
          function restore() {
            let target=state.originKey && doc.querySelector('.'+CSS.escape(state.originKey)+' :is(button,input,textarea,select)');
            if (!target || !target.getClientRects().length) target=nav();
            if (!target || !target.getClientRects().length) target=doc.querySelector('.st-key-brandbar [data-testid=stPopoverButton]');
            target?.focus({preventScroll:true});
          }
          state.listener=(e)=>{
            if(e.isComposing || e.keyCode===229 || e.repeat || e.defaultPrevented) return;
            if((e.ctrlKey||e.metaKey) && !e.shiftKey && !e.altKey && e.key.toLowerCase()==='k'){
              if (!nav()) return;
              e.preventDefault();
              if(field()) {field().focus();return;}
              const origin=e.target.closest?.('[class*="st-key-"]');
              state.originKey=origin && Array.from(origin.classList).find(x=>x.startsWith('st-key-'));
              nav().click();
            } else if(e.key==='Escape' && back() && !doc.querySelector('[role=listbox],[role=dialog]')){
              e.preventDefault();state.returnFocus=true;back().click();
            }
          };
          doc.addEventListener('keydown',state.listener,true);
          // This frame rerenders when navigation changes. Focus only on a transition.
          const searchOpen=SEARCH_OPEN, transition=state.wasOpen!==searchOpen;
          if(transition && (searchOpen || state.returnFocus)) state.pendingFocus=true;
          state.wasOpen=searchOpen;
          if (!state.pendingFocus) return;
          const started=Date.now();
          const focusWhenReady=()=>{
            if (!state.pendingFocus || state.wasOpen!==searchOpen) return;
            const ready=doc.querySelector('[data-testid=stApp]')?.getAttribute('data-test-script-state')==='notRunning';
            if(ready && (Boolean(back())===searchOpen)) {
              if(searchOpen) {
                const target=field()||back();
                if(target && (doc.activeElement===doc.body || transition || doc.activeElement===nav()))
                  target.focus({preventScroll:true});
              } else if(state.returnFocus) restore();
              if(Date.now()-started>450) {state.pendingFocus=false;state.returnFocus=false;return;}
            }
            if(Date.now()-started<3000) win.setTimeout(focusWhenReady,60);
          };
          win.setTimeout(focusWhenReady,20);
        })();
        </script>""".replace('SEARCH_OPEN', 'true' if on_search_page else 'false'), height=0, width=0)


def links(nav, current, prefix='', compact=False):
    def link(pid, label, icon):
        st.button(label, icon=':material/'+icon+':', width='stretch', key=prefix+'nav_'+pid,
                  type='primary' if active == pid else 'secondary', on_click=nav, args=(pid,))
    active = {'quality':'detection','download':'audit','evidence_detail':'review','help':'preferences',
              'about':'preferences','batch':'detection','auxiliary':'detection'}.get(current,current)
    link('search','搜索','search')
    for page in PAGES:
        if not compact and page.id in ('overview','preferences'):
            st.caption('工作台' if page.id == 'overview' else '系统')
        link(page.id,page.label,page.icon)


def header(nav, current):
    with st.container(key='brandbar'):
        left, right = st.columns([8, 1.4], vertical_alignment='center')
        left.html('<div class="brand"><img src="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgc3Ryb2tlPSIjNjg2YWQ1IiBzdHJva2Utd2lkdGg9IjEuOCIgYXJpYS1oaWRkZW49InRydWUiPjxwYXRoIGQ9Im0xMiAyIDggMy43djYuNWMwIDQuMy0zLjEgNy43LTggOS44LTQuOS0yLjEtOC01LjUtOC05LjhWNS43eiIvPjxwYXRoIGQ9Im04LjUgMTEuOCAyLjMgMi40IDQuNi01Ii8+PC9zdmc+" alt=""/><strong>流哨</strong><small>LIUSHAO</small></div>')
        with right.popover('工作空间', width='stretch'):
            links(nav,current,'menu_',True)
            st.button('操作说明', key='header_help', on_click=nav, args=('help',))


def footer(build, model_ready, engine):
    status = ('底座模型校验失败' if not model_ready else '等待分析 · 底座模型已核验' if engine is None
              else '本地研判引擎就绪' if not engine.degraded else '引擎降级 · 仅基础检测')
    st.html(f'<div class="statusbar"><span><b>○</b> &nbsp;{status}</span><span>{build}</span></div>')

