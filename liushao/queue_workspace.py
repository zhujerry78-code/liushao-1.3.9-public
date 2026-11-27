"""Presentation for the review roster and its narrow-screen detail switch."""
import streamlit as st


def styles():
    st.html("""<style>
    .st-key-workbody:has(.st-key-queue_filters,.st-key-event_record_review){container:queue-area / inline-size}
    .st-key-queue_filters,.st-key-queue_list,.st-key-queue_detail,.st-key-record_header,
    .st-key-review_workspace,.st-key-queue_rows{gap:.4rem}
    .st-key-queue_tools [data-testid=stHorizontalBlock],
    .st-key-queue_summary [data-testid=stHorizontalBlock],
    .st-key-queue_pager [data-testid=stHorizontalBlock],
    .st-key-queue_save [data-testid=stHorizontalBlock]{flex-wrap:nowrap!important;gap:.45rem}
    .st-key-queue_tools [data-testid=stColumn],.st-key-queue_summary [data-testid=stColumn],
    .st-key-queue_pager [data-testid=stColumn],.st-key-queue_save [data-testid=stColumn]{min-width:0!important}
    .st-key-queue_tools [data-testid=stColumn]:first-child{flex:1.1 1 0!important}
    .st-key-queue_tools [data-testid=stColumn]:nth-child(2){flex:2 1 0!important}
    .st-key-queue_tools [data-testid=stColumn]:last-child{flex:0 0 5.6rem!important}
    .st-key-queue_summary [data-testid=stColumn]:first-child{flex:1 1 0!important}
    .st-key-queue_summary [data-testid=stColumn]:last-child{flex:0 0 auto!important;width:auto!important}
    .st-key-clear_filters button,.st-key-queue_back button{background:transparent;border:0;padding:.2rem .35rem;color:var(--accent)}
    .st-key-clear_filters button p,.st-key-queue_back button p{font-size:.83rem;color:var(--accent)}
    .st-key-queue_rows{border-block:1px solid var(--line);scrollbar-width:thin;overscroll-behavior:contain}
    .st-key-queue_rows [data-testid=stVerticalBlock],.st-key-queue_rows{gap:0!important}
    .st-key-queue_rows button{justify-content:flex-start;text-align:start;border:0;border-bottom:1px solid var(--line);
        border-radius:0;background:var(--paper);padding:.48rem .65rem;box-shadow:none;min-height:3rem}
    .st-key-queue_rows button [data-testid=stMarkdownContainer]{width:100%}
    .st-key-queue_rows button p{font-size:.82rem;font-weight:400;line-height:1.4;color:var(--sub);overflow-wrap:anywhere}
    .st-key-queue_rows button strong{font-size:.90rem;color:var(--ink)}
    .st-key-queue_rows button:hover{background:var(--panel)}
    .st-key-queue_rows button[kind=primary]{background:var(--pick);box-shadow:inset 3px 0 var(--accent)}
    .st-key-queue_rows button[kind=primary] strong{color:var(--accent)}
    .st-key-queue_pager [data-testid=stColumn]{flex:1 1 0!important}
    .st-key-queue_pager [data-testid=stColumn]:nth-child(2){text-align:center}
    .st-key-queue_save [data-testid=stColumn]:first-child{flex:2 1 0!important}
    .st-key-queue_save [data-testid=stColumn]:last-child{flex:1 1 0!important}
    .st-key-queue_detail{border-left:1px solid var(--line);padding-left:1rem;min-width:0}
    .st-key-queue_back{display:none}
    .st-key-event_record_review .st-key-queue_detail{border:0;padding:0}
    .st-key-event_record_review .st-key-event_record_back button{min-height:2.25rem;color:var(--accent)}
    .record-jumps{display:flex;flex-wrap:wrap;gap:.25rem 1.2rem;border-bottom:1px solid var(--line)}
    .record-jumps a{display:inline-block;padding:.45rem .1rem;color:var(--accent);font-size:.87rem}
    .review-anchor{scroll-margin-top:1rem}
    .st-key-record_judgment{gap:.35rem;min-width:0}
    .st-key-record_evidence{min-width:0}
    @container queue-area (min-width:760px){
      .st-key-event_record_review .st-key-review_workspace{display:grid!important;
        grid-template-columns:minmax(0,1.35fr) minmax(0,1fr);align-items:start;gap:1.2rem}
      .st-key-event_record_review .st-key-record_judgment{position:sticky;top:.55rem}
      .st-key-event_record_review .record-jumps{display:none}
    }
    @container queue-area (width > 880px){
      .st-key-queue_workspace [data-testid=stHorizontalBlock]:has(> [data-testid=stColumn] .st-key-queue_detail){align-items:flex-start;flex-wrap:nowrap}
      .st-key-queue_workspace [data-testid=stColumn]:has(.st-key-queue_list){min-width:0!important;flex:0 0 32%!important;
        position:sticky;top:.5rem;max-height:calc(100dvh - 3.5rem);overflow:auto;scrollbar-width:thin}
      .st-key-queue_workspace [data-testid=stColumn]:has(.st-key-queue_detail){min-width:0!important;flex:1 1 0!important}
    }
    @container queue-area (max-width:880px){
      .st-key-queue_workspace [data-testid=stHorizontalBlock]:has(.st-key-queue_detail){display:block}
      .st-key-queue_workspace [data-testid=stColumn]:has(.st-key-queue_list),
      .st-key-queue_workspace [data-testid=stColumn]:has(.st-key-queue_detail){width:100%!important;min-width:0!important}
      .st-key-queue_mode_list .st-key-queue_workspace [data-testid=stColumn]:has(.st-key-queue_detail),
      .st-key-queue_mode_detail .st-key-queue_workspace [data-testid=stColumn]:has(.st-key-queue_list),
      .st-key-queue_mode_detail:has(.st-key-queue_workspace) .st-key-queue_filters{display:none}
      .st-key-queue_detail{border:0;padding:0}
      .st-key-queue_back{display:block}
    }
    @container queue-area (max-width:440px){
      .st-key-queue_tools [data-testid=stHorizontalBlock]{display:grid!important;grid-template-columns:minmax(0,1fr) 5.6rem}
      .st-key-queue_tools [data-testid=stColumn]{width:100%!important}
      .st-key-queue_tools [data-testid=stColumn]:first-child{grid-area:1 / 1}
      .st-key-queue_tools [data-testid=stColumn]:nth-child(2){grid-area:2 / 1 / 3 / -1}
      .st-key-queue_tools [data-testid=stColumn]:last-child{grid-area:1 / 2}
    }
    </style>""")


def focus_pane(detail):
    """Return to the selected row or reveal the new record when switching panes."""
    from streamlit.components.v1 import html
    target = '.st-key-queue_back button' if detail else '.st-key-queue_rows button[kind=primary]'
    html("""<script>
    (() => {
        let doc;
        try { doc = window.parent.document; } catch (_) { return; }
        let count = 0;
        function restore() {
            const app = doc.querySelector('[data-testid=stApp]');
            const back = doc.querySelector('.st-key-queue_back');
            const row = doc.querySelector('TARGET');
            if (app?.getAttribute('data-test-script-state') !== 'notRunning' || !back || !row) {
                if (++count < 240) requestAnimationFrame(restore);
                return;
            }
            if (getComputedStyle(back).display === 'none') {
                if (row.closest('.st-key-queue_back')) {
                    const head = doc.querySelector('.st-key-record_header');
                    if (head && head.getBoundingClientRect().top < 0) head.scrollIntoView({block:'start'});
                }
                return;
            }
            if (!row.getClientRects().length) return;
            row.focus({preventScroll:true});
            row.scrollIntoView({block:'nearest', inline:'nearest', behavior:'instant'});
        }
        requestAnimationFrame(restore);
    })();
    </script>""".replace('TARGET', target), height=0, width=0, scrolling=False)
