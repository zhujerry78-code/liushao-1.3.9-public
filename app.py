"""Local competition interface. Run through delivery/run_demo.py."""
import base64
import hashlib
import io
import json
import os
import sqlite3
import time
from contextlib import closing, contextmanager
from pathlib import Path
from uuid import uuid4

import pandas as pd
import streamlit as st

from app_score_charts import render_histogram, render_score_comparison
from app_evaluation import SUMMARY_PATH, render_evaluation
from app_navigation import PAGES, BY_ID, page_id, page_label
from app_navigation_scroll import queue_navigation_scroll, render_navigation_scroll
from app_model import load_model_bundle, verified_model
from app_record_tools import batch_positions, confirmed_frame, format_review_time
from app_evidence import build_record_evidence, METHOD_VERSION
from app_engine_evidence import (IDENTITY_COLUMNS, batch_evidence_reference, build_engine_evidence, engine_identity, historical_engine_label,
                                 require_complete_engine_evidence, event_members_and_representatives, event_signature_fields,
                                 ENGINE_EVIDENCE_SCHEMA, ENGINE_EVIDENCE_VERSION)
from app_evidence_ui import (render_record_evidence, render_evidence_detail, render_review_checklist,
                             render_engine_panel, render_historical_event_review)
from app_quick_search import promote_matches
from trial_build import BUILD_LABEL
import app_workspace as shell
import app_local_workspace as local
from app_input import preflight_input, problem_sample, present_error, UserError
from app_session import consent_generation, saved_upload, save_upload, clear_upload, load_upload
from app_service import analyze_input, load_trusted_engine, read_review_snapshot
from app_types import (SessionState, Filters, Draft, PendingAction, invalidate_model, advance_suspended_draft,
                       needs_review, line_number)
from app_quality import build_quality_view
from app_ui_state import (STATUS_NAMES, UI_REVISION, base_meta, csv_bytes, default_filters,
                          dirty, engine_batch_summary, export_snapshot, field_name, filtered, literal_markdown, QUEUE_SCOPES)
from liushao.data import read_csv_checked, sha256
from liushao.model import evidence_for, infer
from liushao import dynamic, gate, recovery, workbench
from liushao.reviews import STATUSES, ReviewConflictError, connect, save_review, evidence_for_review
from liushao.search import ANOMALY, NORMAL, AnalysisContext, SearchScope, search_counterexamples

ROOT = Path(__file__).resolve().parent
ARTIFACTS = Path(os.environ.get('LIUSHAO_ARTIFACTS_DIR', str(ROOT / 'artifacts')))
DB = Path(os.environ.get('LIUSHAO_REVIEW_DB', str(ROOT / 'data/reviews.sqlite')))
EVALUATION = Path(os.environ.get('LIUSHAO_EVALUATION_SUMMARY', str(ROOT / SUMMARY_PATH)))
VERSION = (ROOT / 'VERSION').read_text(encoding='utf-8').strip()
st.set_page_config(page_title='流哨 · 工作空间', page_icon=str(ROOT / 'liushao_icon.svg'), layout='wide')
shell.styles()

if 'ui' not in st.session_state:
    st.session_state.ui = SessionState()
u = st.session_state.ui
u.page = page_id(u.page)
if u.page == 'evaluation': u.page = 'help'
st.session_state['_sample_kind'] = st.session_state.get('_sample_kind', '缺少必需列')
st.session_state['_remember'] = st.session_state.get('_remember', False)
st.session_state['_source'] = st.session_state.get('_source', u.stage_source)


def nav(page):
    update_editor()
    destination = page_id(page)
    if destination != 'evidence_detail' and not (u.page == 'evidence_detail' and destination == 'review'):
        origin = st.session_state.pop('_event_record_origin', None)
        if origin and destination == 'review':
            set_queue_pane(False)
    if destination == 'search' and u.page != 'search':
        st.session_state['_search_origin'] = u.page
    if u.page == 'evidence_detail' and destination == 'review':
        queue_navigation_scroll(st.session_state, destination)
    u.page = destination


def open_evidence_detail():
    update_editor()
    st.session_state['_evidence_detail_identity'] = (u.active.key, u.selected, model_identity)
    u.page = 'evidence_detail'
    queue_navigation_scroll(st.session_state, u.page)


def update_editor(expected_identity=None):
    editor = u.editor
    if editor:
        identity = (editor.input_key, editor.record_id, editor.baseline)
        # A delayed callback may belong to the record that was visible before
        # navigation. Never copy its shared widget values into another draft.
        if (st.session_state.get('_editor_identity') != identity
                or (expected_identity is not None and expected_identity != identity)):
            return
        editor.status = st.session_state.get('_status', editor.status)
        editor.note = st.session_state.get('_note', editor.note)
        if dirty(editor):
            u.exports = None


def reload_saved_editor():
    # A callback completes before a navigation rerun can interrupt the discard.
    if u.pending is not None or not available:
        return
    u.editor = None
    for key in ('_editor_identity', '_status', '_note'):
        st.session_state.pop(key, None)


def show_local_file_error(error, details='本机文件操作详情'):
    message = error if isinstance(error, UserError) else present_error(error, '本机文件操作')
    st.warning(message.reason)
    st.caption(message.action)
    with st.expander(details):
        st.text(message.detail)


def request(kind, value=None):
    # Streamlit may deliver the text-area blur and navigation in one request;
    # callback order must not decide whether the unsaved-change guard runs.
    update_editor()
    if u.pending is not None:
        return
    if kind == 'record' and value == u.selected:
        return
    same_draft = u.editor and ((kind == 'review_record' and u.editor.record_id == value)
                               or (kind == 'event_record' and u.editor.record_id == value[-1]))
    if dirty(u.editor) and not same_draft and kind in ('record', 'filters', 'run', 'review_scope', 'event_selection', 'restore', 'resume_draft', 'review_record', 'event_record', 'queue_page'):
        u.pending = PendingAction(kind, value)
    else:
        apply_action(kind, value)
    if kind != 'record': u.table_epoch += 1


def filter_change(name, key):
    request('filters', {**u.filters, name: st.session_state[key]})


def row_selected(key, record_ids):
    if (key not in ('_records_main','_records_'+str(u.table_epoch))) or key not in st.session_state:
        return
    selection = st.session_state[key].selection
    rows = selection.get('rows', []) or ([selection.cells[0][0]] if selection.get('cells') else [])
    if rows and type(rows[0]) is int and 0 <= rows[0] < len(record_ids):
        open_queue_row(record_ids[rows[0]])


def set_queue_pane(detail, *, focus=True):
    st.session_state['_queue_detail_batch'] = u.active.key if detail else None
    if focus:
        st.session_state['_focus_queue_detail'] = detail


def open_queue_row(rid):
    if u.pending is None and rid == u.selected:
        set_queue_pane(True)
    else:
        request('record', rid)


def page_to(index):
    request('queue_page', index)


def open_score_bin(index):
    if type(index) is int and 0 <= index < 20:
        request('review_scope', dict(Filters(scope='全部记录', score_bin=index)))


def apply_action(kind, value):
    if kind == 'event_selection':
        apply_event_selection(value)
    elif kind == 'event_record':
        batch_key, number, member, section, record_id = value
        if (not valid_result or batch_key != u.active.key or 'ow_event' not in pred
                or not (pred.record_id.eq(member) & pred.ow_event.eq(number)).any()
                or not pred.record_id.eq(record_id).any()):
            return
        apply_action('review_record', record_id)
        st.session_state['_event_record_origin'] = value
        queue_navigation_scroll(st.session_state, 'review')
    elif kind == 'review_record':
        if not valid_result or value not in set(raw_pred.record_id):
            return
        st.session_state.pop('_event_record_origin', None)
        u.filters = Filters(scope='全部记录')
        u.search_keyword = ''
        for key in list(st.session_state):
            if key.startswith('_filter_'):
                del st.session_state[key]
        if u.editor is not None and u.editor.record_id != value:
            u.editor = None
        u.selected, u.page = value, 'review'
        set_queue_pane(True)
    elif kind == 'queue_page':
        u.page_index, u.selected, u.editor = max(0, int(value)), None, None
        set_queue_pane(False)
    elif kind == 'record':
        if u.selected != value:
            u.selected, u.editor = value, None
        set_queue_pane(True)
    elif kind in ('filters', 'review_scope'):
        u.filters = Filters(**dict(value))
        for key in list(st.session_state):
            if key.startswith('_filter_'):
                del st.session_state[key]
        u.page_index = 0
        u.search_keyword = u.filters.keyword
        u.selected, u.editor = None, None
        set_queue_pane(False, focus=False)
        if kind == 'review_scope':
            u.selected, u.editor, u.page = None, None, 'review'
    elif kind == 'run':
        run_input(value)
    elif kind == 'restore':
        try:
            payload = load_upload(DB, value, model_identity)
            u.preflight = u.preflight_key = u.preflight_error = None
            u.save_generation = u.remember_key = None
            st.session_state['_remember'] = False
            run_input(payload)
        except Exception as error:
            u.run_error = present_error(error, '恢复文件')
            u.show_import = True
    elif kind == 'resume_draft':
        if (value is u.suspended_draft and u.active is not None
                and value.input_key == u.active.key and u.result is not None
                and value.record_id in set(u.result.predictions.record_id)):
            u.editor, u.selected = value, value.record_id
            advance_suspended_draft(u)
            u.exports, u.page = None, 'review'
            set_queue_pane(True)
            st.session_state.pop('_editor_identity', None)
    u.pending = None


@st.cache_resource(max_entries=2)
def load_model(identity, _content, _descriptor):
    return load_model_bundle(_content, _descriptor)


bundle = None
model_error = None
verified = None
try:
    verified = verified_model(ARTIFACTS / 'model.joblib', ARTIFACTS / 'model.json')
    model_identity = verified.identity
    bundle = load_model(model_identity, verified.content, verified.descriptor)
    if u.active and u.active.model != model_identity:
        invalidate_model(u)
        st.session_state['_remember'] = False
except Exception as error:
    model_error = str(error)
    model_identity = None
    invalidate_model(u)
    st.session_state['_remember'] = False


def run_input(payload):
    if bundle is None:
        u.run_error = '分析程序尚未就绪，请在“偏好设置”打开“查看构建与模型”。'
        return
    try:
        permission = (u.remember_key, u.save_generation)
        input_key = (hashlib.sha256(payload[0]).hexdigest(), payload[1], payload[2])
        with st.spinner('格式核验完成，正在计算排序分数并整理事件…'):
            active, result, physical, quality, error = analyze_input(*payload, bundle, model_identity, ARTIFACTS,
                                                                  preflight=u.preflight)
        u.update(active=active, result=result, physical_rows=physical, quality=quality,
                 quality_error=error, quality_selected=None, quality_page=0,
                 editor=None, selected=None, filters=Filters(), page_index=0,
                 show_import=False, run_error=None, exports=None, diagnostics=None,
                 batch=None, clues=None, page='events', tool_downloads={})
        u.search_keyword = ''
        st.session_state.pop('_position_fingerprint', None)
        st.session_state.pop('_expected_batch',None)
        for key in list(st.session_state):
            if key.startswith(('_queue_', '_event_', '_histogram_', '_export_range', '_review_export', '_export_event')):
                del st.session_state[key]
        try:
            position = local.last_position(DB, active)
            if position:
                saved_filters = {k:v for k,v in position.get('filters',{}).items() if k in Filters()}
                u.filters = Filters(**saved_filters)
                u.selected = position.get('record') if position.get('record') in set(result.predictions.record_id) else None
                u.page_index = max(0,int(position.get('page_index',0)))
                if position.get('page') in ('events','review','detection','overview','audit'):
                    u.page = position['page']
                event = position.get('event')
                if isinstance(event,list) and len(event)==2 and event[1] in set(result.predictions.record_id):
                    st.session_state['_event_target'] = (active.key, *event)
                u.flash = '已恢复该批次上次的工作位置。'
        except (OSError, sqlite3.Error, ValueError, TypeError):
            u.flash = '分析完成。上次工作位置暂不可读取。'
        u.table_epoch += 1
        for key in list(st.session_state):
            if key.startswith(('_filter_', '_quality_')) or key in ('_status', '_note'):
                del st.session_state[key]
    except Exception as error:
        u.run_error = present_error(error)
        return
    u.cache_message = None
    if (payload[2] == '上传 CSV' and st.session_state.get('_remember', False)
            and permission[0] == input_key and permission == (u.remember_key, u.save_generation)
            and permission[1] is not None):
        try:
            saved = save_upload(DB, payload[0], payload[1], model_identity, permission[1])
            u.cache_message = '已在本机保留此文件，下次可继续检查。' if saved else '另一窗口已清除缓存，本次未再保留文件。'
        except Exception as error:
            u.cache_message = present_error(error, '分析已完成，保留文件未完成')


if u.suspended_draft:
    st.warning('模型曾改变或校验失败，未保存填写已暂存。请重新检查原文件后明确恢复。')
    draft = u.suspended_draft
    can_restore = bool(u.active and u.active.key == draft.input_key and u.result is not None
                       and draft.record_id in set(u.result.predictions.record_id))
    st.caption('暂存判断：' + STATUS_NAMES.get(draft.status, draft.status))
    st.code(draft.note or '（无备注）')
    if u.suspended_drafts:
        st.caption(f'还有 {len(u.suspended_drafts)} 份暂存填写；处理当前这一份后会依次显示。')
    if (st.button('恢复暂存填写', key='restore_suspended', disabled=not can_restore or u.pending is not None)
            and can_restore and u.pending is None):
        request('resume_draft', draft)
        st.rerun()
    if st.button('放弃暂存填写', key='discard_suspended', disabled=u.pending is not None) and u.pending is None:
        advance_suspended_draft(u)
        st.rerun()


valid_result = bool(bundle is not None and u.active and u.active.model == model_identity and u.result is not None)
available, reviews, audit = True, {}, []
if valid_result:
    result = u.result
    raw_pred, features, rejected = result.predictions, result.features, result.rejected
    elapsed, total = result.elapsed, result.total
    try:
        snapshot = read_review_snapshot(DB, bundle['model_version'], set(raw_pred.record_id))
        reviews, audit = snapshot.current, snapshot.history
    except (OSError, sqlite3.Error, ValueError, TypeError, KeyError) as error:
        available = False
        if u.exports and u.exports.predictions is not None:
            u.exports = None
        st.warning('暂时无法读取已保存的操作。分析结果仍可查看，保存和相关下载已暂停。')
        with st.expander('查看错误详情'):
            st.write(str(error))
        st.button('重试读取', key='retry_reviews')
    pred = raw_pred.copy()
    pred['review_status'] = pred.record_id.map(lambda rid: reviews.get(rid, {}).get('status', '待复核')) if available else '复核不可用'
    pred['review_note'] = pred.record_id.map(lambda rid: reviews.get(rid, {}).get('note', ''))
    meta = dict(base_meta(u.active, u.result, bundle, verified), build=BUILD_LABEL, ui_revision=BUILD_LABEL)
    if u.exports and u.exports['input_key'] != u.active.key:
        u.exports = None
else:
    pred = None


def save_editor(next_after=False):
    # Sync widget values when the editor helper is available. The export
    # test extracts save_editor alone, where the helper is absent and the
    # editor already carries the values under test.
    if 'update_editor' in globals():
        update_editor()
    editor = u.editor
    if not valid_result or not available or not editor:
        st.error('当前暂不能保存，请先恢复文件与保存记录。')
        return False
    if editor.input_key != u.active.key:
        st.error('文件已改变，请重新选择记录。')
        return False
    if editor.status in ('确认正常', '确认异常'):
        from app_review_guidance import human_basis
        if not human_basis(editor.note):
            st.error('请填写你实际核对的记录、日志或业务依据后再确认。')
            return False
    if getattr(editor, 'original', None) == (editor.status, editor.note):
        st.info('填写没有变化，未新增操作记录。')
        return False
    if next_after:
        order = filtered(pred, u.filters, bundle, physical_rows=u.physical_rows).record_id.tolist()
        following = order[order.index(editor.record_id) + 1:] if editor.record_id in order else []
    try:
        save_review(DB, editor.record_id, bundle['model_version'], editor.status, editor.note,
                    expected_review_id=editor.baseline,
                    evidence_snapshot=require_complete_engine_evidence(current_evidence(editor.record_id)))
    except ReviewConflictError:
        editor.conflict = True
        st.error('另一窗口已修改这条记录。你的填写已保留，请先查看操作历史，再读取最新记录。')
        return False
    except Exception as error:
        st.error(f'保存失败，填写内容已保留：{error}')
        return False
    if next_after:
        st.session_state['_queue_next'] = (u.active.key, following)
    u.selected = editor.record_id
    u.editor, u.exports = None, None
    u.flash = '逐条复核已保存。'
    if 'st' in globals():
        row = raw_pred.index[raw_pred.record_id.eq(editor.record_id)][0]
        number = int(u.physical_rows.at[row]) if u.physical_rows is not None else '未知'
        st.session_state['_queue_receipt'] = (u.active.key, f'已保存 {u.active.name} · 输入 CSV 第 {number} 行 · {editor.status}')
    return True


def render_pending_action():
    if not u.pending:
        return
    st.warning('有尚未保存的填写。请先保存，或明确放弃后再切换；取消会留在原记录。')
    a, b, c = st.columns(3)
    if a.button('保存后继续', key='pending_save', disabled=not available or not valid_result):
        pending = u.pending
        if save_editor():
            apply_action(pending.kind, pending.value)
            st.rerun()
    if b.button('放弃填写并继续', key='pending_discard'):
        pending = u.pending
        u.editor = None
        apply_action(pending.kind, pending.value)
        st.rerun()
    if c.button('取消切换', key='pending_cancel'):
        for key in list(st.session_state):
            if key.startswith('_filter_'):
                del st.session_state[key]
        u.pending = None
        u.table_epoch += 1
        st.rerun()


def empty_page():
    st.info('请先导入文件并完成检查。')
    st.button('前往导入数据', key='to_import', on_click=nav, args=('detection',))


def file_bar():
    left, right = st.columns([5, 1])
    label = '演示数据' if u.active.source == '内置演示数据' else u.active.name
    left.caption(f'{literal_markdown(label)} · 已检查 {total:,} 条')
    if right.button('更换文件', key='change_file', width='stretch'):
        u.show_import, u.page = True, 'detection'
        st.rerun()


def upload_changed():
    upload = st.session_state.get('_upload')
    u.stage_upload = None if upload is None else (upload.getvalue(), upload.name)
    u.preflight = u.preflight_key = u.preflight_error = None
    u.remember_key = u.save_generation = None
    st.session_state['_remember'] = False


def remember_changed():
    u.save_generation = None
    if st.session_state.get('_remember') and u.stage_upload:
        try:
            content, name = u.stage_upload
            u.remember_key = (hashlib.sha256(content).hexdigest(), name, '上传 CSV')
            u.save_generation = consent_generation(DB)
        except Exception as error:
            st.session_state['_remember'] = False
            u.cache_message = present_error(error, '启用文件保留')


def resume_card():
    try:
        saved = saved_upload(DB)
        if saved:
            if u.save_generation is not None and u.save_generation != saved['generation']:
                u.save_generation = u.remember_key = None
                st.session_state['_remember'] = False
            st.caption('上次保留：' + literal_markdown(saved['name']) + ' · ' + saved.get('saved_at', ''))
            if saved['model'] != model_identity:
                st.caption('当前模型身份不匹配，不能恢复；可重新上传原文件。')
            if st.button('继续上次检查', key='restore_last', disabled=bundle is None or saved['model'] != model_identity or u.pending is not None):
                request('restore', (saved['generation'], saved['id']))
                st.rerun()
        elif u.save_generation is not None:
            # A clear in another window must also revoke the old checkbox permission.
            from app_session import session_root
            if session_root(DB).exists() and consent_generation(DB) != u.save_generation:
                u.save_generation = u.remember_key = None
                st.session_state['_remember'] = False
    except Exception as error:
        show_local_file_error(error, '查看恢复文件详情')


def localize_csv_uploader():
    # Streamlit 1.50 does not expose these native uploader labels in Python.
    # Keep its original input, drop and keyboard handlers and translate only its text.
    from streamlit.components.v1 import html
    with st.container(key='upload_language'):
        html('''<script>
        (() => {
          const doc = parent.document;
          const host = doc.querySelector('.st-key-import_card');
          if (!host) return;
          function translate() {
            const zone = host.querySelector('[data-testid="stFileUploaderDropzone"]');
            if (!zone) return;
            const replacements = {
              'Drag and drop ': '拖放', 'file': ' CSV 文件', ' here': '到此处',
              'Browse files': '选择文件', 'File size is too small.': '文件过小。',
              'Only one file is allowed.': '每次只能选择一个文件。',
              'Unknown error': '文件上传失败，请重试。',
              'Unexpected error. Please try again.': '文件上传失败，请重试。'
            };
            const walker = doc.createTreeWalker(zone, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {
              const node = walker.currentNode;
              if (Object.hasOwn(replacements, node.data)) node.data = replacements[node.data];
              else if (node.data.startsWith('Limit ') && node.data.endsWith('MB per file'))
                node.data = '单个文件最多 ' + node.data.slice(6, -11) + ' MiB';
            }
            for (const row of host.querySelectorAll('[data-testid="stFileUploaderFile"]')) {
              const remove = row.querySelector('[data-testid="stFileUploaderDeleteBtn"] button');
              const name = row.querySelector('[data-testid="stFileUploaderFileName"]');
              const label = name && '移除 ' + name.textContent;
              if (remove && label && remove.getAttribute('aria-label') !== label) remove.setAttribute('aria-label', label);
              const error = row.querySelector('[data-testid="stFileUploaderFileErrorMessage"]')?.firstChild;
              if (error?.nodeType === 3) {
                if (Object.hasOwn(replacements, error.data)) error.data = replacements[error.data];
                else if (error.data.endsWith(' files are not allowed.')) error.data = '文件类型不支持，请选择 CSV 文件。';
                else if (error.data.startsWith('File must be ') && error.data.endsWith(' or smaller.'))
                  error.data = '文件超过大小限制，请按上传区所示容量缩小文件。';
              }
            }
          }
          translate();
          const observer = new MutationObserver(translate);
          observer.observe(host, {subtree:true, childList:true, characterData:true});
          window.addEventListener('pagehide', () => observer.disconnect(), {once:true});
        })();
        </script>''', height=0, width=0, scrolling=False)


def import_card():
    st.html('''<style>
      .st-key-upload_language{display:none}
      .st-key-import_card > [data-testid=stVerticalBlock]{gap:.5rem}
    </style>''')
    with st.container(border=True, key='import_card'):
        st.markdown('**导入数据**')
        resume_card()
        st.session_state.setdefault('_source', u.stage_source)
        source = st.radio('数据来源', ['内置演示数据', '上传 CSV'], horizontal=True, key='_source',
                          format_func=lambda x: {'上传 CSV': '上传文件', '内置演示数据': '使用演示数据', '格式练习': '格式练习'}[x],
                          on_change=lambda: u.update(stage_source=st.session_state['_source']))
        content, name = None, None
        if source == '上传 CSV':
            st.caption('UTF-8 CSV · 最多 50,000 条记录 · 77 个必需特征列')
            upload = st.file_uploader('选择 CSV 文件', type=['csv'], key='_upload', on_change=upload_changed)
            localize_csv_uploader()
            if upload is not None:
                u.stage_upload = (upload.getvalue(), upload.name)
            if u.stage_upload:
                content, name = u.stage_upload
                st.caption(f'待检查：{literal_markdown(name)}')
        else:
            try:
                content = (ARTIFACTS / 'demo.csv').read_bytes()
                name = 'demo.csv'
                st.caption('内置演示 · 按 demo.csv 的实际输入行定位。')
            except OSError:
                st.warning('演示文件暂不可用，请上传文件。')
        expected = st.session_state.get('_expected_batch') if source=='上传 CSV' else None
        if expected:
            st.info('正在恢复原批次 ' + expected[:12] + '，所选内容须与原文件一致。')
            if st.button('取消恢复，导入新批次'):
                st.session_state.pop('_expected_batch',None)
                st.rerun()
        blocked = False
        if content is not None and bundle is not None:
            key = (hashlib.sha256(content).hexdigest(), name, source, model_identity)
            if u.preflight_key != key:
                u.preflight, u.preflight_error, u.run_error = None, None, None
                u.preflight_key = key
                try:
                    u.preflight = preflight_input(content, bundle, model_identity)
                except Exception as error:
                    u.preflight_error = u.run_error = present_error(error)
            blocked = u.preflight_error is not None
            if expected and key[0] != expected:
                blocked = True
                st.warning('文件内容与所选批次不符。请选择原 CSV，或取消恢复以导入新批次。')
            if u.preflight:
                info = u.preflight.summary()
                st.caption(f'格式通过：{info["rows"]:,} 条记录 · {info["columns"]} 列（必需 {info["required"]}，额外 {info["extra"]}）')
                if u.preflight.time_counts is None:
                    st.caption('文件没有 Timestamp 列，分时统计与相邻时间查找不可用。')
                else:
                    parsed, empty, invalid = u.preflight.time_counts
                    st.caption(f'可解析时间 {parsed:,} 条 · 缺少时间 {empty:,} 条 · 格式无法解析 {invalid:,} 条。')
        if source == '上传 CSV' and u.stage_upload:
            st.checkbox('在本机保留此文件，便于下次继续', key='_remember', on_change=remember_changed,
                        help='取消勾选仅撤销本次保存。已保留的 CSV 可在“偏好设置”中清除。')
        a, b, c = st.columns([2, 2, 3])
        if a.button('开始检查', key='run', type='primary', width='stretch', disabled=blocked or content is None or bundle is None or u.pending is not None):
            request('run', (content, name, source))
            st.rerun()
        if valid_result and b.button('取消更换', key='cancel_import', width='stretch'):
            u.show_import, u.run_error, u.page = False, None, 'events'
            st.rerun()
        with c.popover('格式说明'):
            st.write('支持 UTF-8 CSV，最多 50 MiB、50,000 条记录。请保留全部 77 个必需特征列。可附加时间、标签等其他列，标签列不参与分析。')
            st.caption('重传相同内容可找回已保存判断。修改文件会建立新批次，原判断仍归属于此前的文件。')
            if bundle:
                st.download_button('下载空白 CSV 模板', csv_bytes(pd.DataFrame(columns=bundle['config']['features'])),
                                   'liushao_features_template.csv', 'text/csv', key='template_csv')
            demo = ARTIFACTS / 'demo.csv'
            if demo.is_file():
                st.download_button('下载演示文件', demo.read_bytes(), 'liushao_demo.csv', 'text/csv', key='demo_download')
            if bundle:
                with st.expander('查看需要的数据列'):
                    st.write(bundle['config']['features'])
    if u.run_error:
        st.error('文件检查未完成，请查看原因后重试。')
        if hasattr(u.run_error, 'reason'):
            st.write(u.run_error.reason)
            st.write(u.run_error.action)
            with st.expander('技术详情'):
                st.text(u.run_error.detail)
        else:
            st.write(u.run_error)
        if valid_result:
            st.caption('下方仍是上一次成功检查的文件结果：' + literal_markdown(u.active.name))


def verified_display_engine():
    engine = load_trusted_engine(ARTIFACTS, bundle)
    identity = engine.get('_artifact_identity', {}) if engine else {}
    if engine is None or tuple(identity.get(name) for name in IDENTITY_COLUMNS) != engine_identity(pred.iloc[0]):
        raise ValueError('引擎校验未通过。请恢复配套文件并重新检查。')
    return engine


def render_risk_controls():
    if pred.empty or not {'risk_alpha', 'guarantee_status', 'typ_score'} <= set(pred):
        return
    from liushao_engine import RISK_ALPHAS, alpha_for_budget, apply_alpha
    current = pred.risk_alpha.iloc[0]
    if current not in RISK_ALPHAS or pd.isna(pred.guarantee_status.iloc[0]):
        return
    with st.expander('调整 α 与查看复核预算'):
        locked = u.pending is not None or dirty(u.editor)
        if locked:
            st.caption('先保存或放弃当前记录的填写。')
        a, b = st.columns(2)
        selected = a.selectbox('风险水平 α', RISK_ALPHAS, index=RISK_ALPHAS.index(current),
                               key='risk_alpha_choice', disabled=locked)
        budget = b.number_input('可复核的记录数', min_value=0, max_value=len(pred), step=1, key='risk_budget',
                                disabled=locked)
        if (a.button('应用 α', key='apply_risk_alpha', disabled=locked or selected == current)
                and not locked and selected != current):
            try:
                u.result.predictions = apply_alpha(raw_pred, verified_display_engine(), float(selected))
                u.exports = u.diagnostics = u.batch = u.clues = None
                u.table_epoch += 1
                st.session_state.pop('_engine_evidence_batch', None)
                st.session_state.pop('_record_evidence_cache', None)
                st.session_state.pop('_risk_budget_response', None)
                u.flash = f'已按 α = {selected:g} 更新本批复核队列。'
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                st.error(f'未更新风险水平：{error}')
            else:
                st.rerun()
        if b.button('查看最小 α', key='check_risk_budget', disabled=locked) and not locked:
            try:
                minimum = alpha_for_budget(raw_pred, verified_display_engine(), int(budget))
                text = '当前预算内没有可用的 α 档位。' if minimum is None else f'当前预算内最小 α：{minimum:g}。'
                st.session_state['_risk_budget_response'] = (u.active.key, current, budget, text)
            except (ValueError, KeyError, TypeError, RuntimeError) as error:
                st.error(f'无法核对复核预算：{error}')
        response = st.session_state.get('_risk_budget_response')
        if response and response[:3] == (u.active.key, current, budget):
            st.caption(response[-1])


def editor_for(rid, next_id):
    editor = u.editor
    if editor is None or editor.record_id != rid or editor.input_key != u.active.key:
        saved = reviews.get(rid, {})
        status, note = saved.get('status', '待复核'), saved.get('note', '')
        u.editor = Draft(record_id=rid, input_key=u.active.key, status=status, note=note,
            original=(status, note), baseline=saved.get('id', 0), next_id=next_id, conflict=False)
    else:
        editor.next_id = next_id
    return u.editor


def current_evidence(rid):
    row_index=pred.index[pred.record_id.eq(rid)][0];record=pred.loc[row_index]
    engine=load_trusted_engine(ARTIFACTS,bundle)
    current_engine_identity=engine.get('_artifact_identity',{}) if engine else {}
    key=(u.active.key,rid,id(u.result.predictions),meta['model_sha256'],meta['descriptor_sha256'],METHOD_VERSION,
         ENGINE_EVIDENCE_VERSION,engine_identity(record),
         current_engine_identity.get('engine_sha256'),current_engine_identity.get('engine_descriptor_sha256'),
         current_engine_identity.get('engine_inference_version'), record.get('risk_alpha'))
    cache=st.session_state.setdefault('_record_evidence_cache',{})
    if key not in cache:
        physical=int(u.physical_rows.at[row_index]) if u.physical_rows is not None else None
        try:
            context_key=st.session_state.get('_raw_context_key')
            if context_key!=u.active.key:
                st.session_state['_raw_context']=read_csv_checked(io.BytesIO(u.active.content))
                st.session_state['_raw_context_key']=u.active.key
            from app_quality import aligned_context
            context=aligned_context(st.session_state['_raw_context'],u.physical_rows,features.index)
            context.attrs['record_ids'] = pred.record_id.to_dict()
            payload=build_record_evidence(record,features,bundle,context=context,meta=dict(meta,input_csv_row=physical))
            if u.quality is not None and rid in u.quality.records.index:
                details=u.quality.details(rid)
                payload['input_quality_details']=details.to_dict(orient='records')
            cache[key]=payload
        except Exception as error:
            cache[key]={'schema_version':2,'method_version':METHOD_VERSION,'record_id':rid,
                'model_version':bundle['model_version'],'input_name':u.active.name,'input_csv_row':physical,
                'input_sha256':meta['input_sha256'],'model_sha256':meta['model_sha256'],
                'descriptor_sha256':meta['descriptor_sha256'],
                'identity':dict(source_file=str(record.source_file),source_row=int(record.source_row),source_sha256=str(record.source_sha256)),
                'scores':dict(rf=float(record.rf_score),lr=float(record.lr_score),rf_threshold=bundle['config']['rf_threshold'],lr_threshold=bundle['config']['lr_threshold']),
                'cards':[],'all_fields':[],'selection':dict(status='unavailable',reason=str(error)),
                'decomposition':dict(status='unavailable',reason=str(error)),
                'assessment':dict(status='证据计算暂不可用，需外部补证',reasons=[str(error)])}
        try:
            reference = None
            if engine is not None and 'cp_set' in record and engine_identity(record) == tuple(
                    current_engine_identity.get(name) for name in IDENTITY_COLUMNS):
                batch_key = key[:1] + key[2:]
                cached = st.session_state.get('_engine_evidence_batch')
                if cached is None or cached[0] != batch_key:
                    cached = (batch_key, batch_evidence_reference(pred, features, bundle, engine))
                    st.session_state['_engine_evidence_batch'] = cached
                reference = cached[1].loc[row_index]
            cache[key]['engine']=build_engine_evidence(record,features.loc[[row_index]],bundle,engine,reference=reference)
        except Exception as error:
            cache[key]['engine']={'schema_version':ENGINE_EVIDENCE_SCHEMA,'method_version':ENGINE_EVIDENCE_VERSION,
                                  'status':'unavailable','reason':str(error)}
        while len(cache)>8: del cache[next(iter(cache))]
    return cache[key]


def record_panel(rid, order, event_origin=None):
    from app_ui_state import conclusion_source, queue_reason
    row_index = pred.index[pred.record_id.eq(rid)][0]
    record = pred.loc[row_index]
    pos = order.index(rid) if rid in order else -1
    next_id = order[pos + 1] if pos >= 0 and pos + 1 < len(order) else None
    editor = editor_for(rid, next_id)
    with st.container(key='record_header'):
        state = record.get('ow_state', record.get('cp_set'))
        engine_label = f' · 引擎研判 {state}' if pd.notna(state) else ''
        physical = int(u.physical_rows.at[row_index]) if u.physical_rows is not None else '未知'
        st.markdown(f'### 输入 CSV 第 {physical} 行 · {record.review_status}')
        st.caption(('系统提示异常' if record.rf_anomaly else '未提示异常') + engine_label + ' · ' + conclusion_source(record))
        st.write(queue_reason(record, bundle['config'], include_lr_near=u.filters.include_lr_near))
        render_score_comparison(record, bundle['config'])
        with st.expander('来源与追溯'):
            st.text(f'文件 {u.active.name}\n批次 {u.active.digest}\n来源 {record.source_file} 第 {record.source_row} 行\n记录 {rid}')
            if record.get('authorization_batch'):
                st.text('原授权批次 ' + record.authorization_batch)
        if rid not in order[u.page_index*50:(u.page_index+1)*50]:
            st.caption('当前编辑记录不在当前页，详情与填写仍保留。')
    payload = current_evidence(rid)
    st.html('<nav class="record-jumps" aria-label="本条记录内跳转"><a href="#review-checks">查看核查重点</a>'
            '<a href="#review-decision">填写人工判断</a></nav>')
    with st.container(key='review_workspace'):
        with st.container(key='record_evidence'):
            st.html('<span id="review-checks" tabindex="-1" class="review-anchor" aria-label="核查重点"></span>')
            render_review_checklist(payload)
            st.button('查看完整评分依据', key='open_evidence_detail', on_click=open_evidence_detail,
                      disabled=u.pending is not None, icon=':material/insights:')
            with st.expander('技术证据与模型解释'):
                render_record_evidence(payload)
                render_engine_panel(payload.get('engine'))
        with st.container(key='record_judgment'):
            st.html('<h3 id="review-decision" tabindex="-1" class="review-anchor">填写人工判断</h3>')
            render_editor(editor, event_origin)
    opened_history = st.session_state.get('_opened_history')
    with st.expander('操作历史', expanded=bool(opened_history and opened_history['record_id'] == rid)):
        entries=[item for item in audit if item['record_id']==rid]
        if not available: st.info('暂时无法读取已保存操作。')
        elif not entries: st.info('这条记录还没有已保存操作。')
        else:
            frame=pd.DataFrame(entries)[['id','status','note','created_at','evidence_snapshot_label']].copy()
            frame.status=frame.status.map(STATUS_NAMES)
            frame.created_at=frame.created_at.map(lambda value: format_review_time(value, beijing=True))
            st.caption('默认北京时间；无法换算的时间保留原始时区 · 点击查看保存时的证据')
            st.dataframe(frame.rename(columns={'id':'操作ID','status':'我的判断','note':'备注','created_at':'保存时间','evidence_snapshot_label':'当时证据'}),hide_index=True,width='stretch')
            selected_id=st.selectbox('查看哪次保存', [item['id'] for item in entries],key='_history_choice_'+rid)
            if st.button('查看当时证据',key='read_history_'+rid):
                try:
                    opened = evidence_for_review(DB, selected_id)
                except Exception as error:
                    st.warning('暂时无法读取这次保存的证据，当前填写已保留。请稍后重试。')
                    with st.expander('查看读取错误详情'):
                        st.text(str(error))
                else:
                    st.session_state['_opened_history'] = opened
                    st.rerun()
            opened=st.session_state.get('_opened_history')
            if opened and opened['record_id']==rid and opened['id']==selected_id:
                st.info(opened['evidence_snapshot_label'])
                if opened['evidence_snapshot'] is not None:
                    if 'event_review' in opened['evidence_snapshot']:
                        render_historical_event_review(opened['evidence_snapshot']['event_review'])
                    else:
                        st.caption(historical_engine_label(opened['evidence_snapshot'],engine_identity(record)))
                        render_engine_panel(opened['evidence_snapshot'].get('engine'),historical=True)
                    st.json(opened['evidence_snapshot'],expanded=False)


def render_editor(editor, event_origin=None):
    from app_review_guidance import human_basis
    rid = editor.record_id
    latest = reviews.get(rid, {}).get('id', 0)
    if editor.conflict or latest != editor.baseline:
        st.warning('另一窗口已修改这条记录。核对两个版本后可读取最新记录。')
        mine, theirs = st.columns(2)
        mine.caption('本窗口草稿')
        mine.text(editor.status + '\n' + editor.note)
        theirs.caption(f'已保存版本 #{latest}')
        theirs.text(reviews.get(rid, {}).get('status', '') + '\n' + reviews.get(rid, {}).get('note', ''))
    locked = u.pending is not None
    if u.pending is not None:
        st.info('请先处理上方的保存、放弃或取消切换。')
    widget_identity = (editor.input_key, editor.record_id, editor.baseline)
    if (st.session_state.get('_editor_identity') != widget_identity
            or '_status' not in st.session_state or '_note' not in st.session_state):
        st.session_state['_status'], st.session_state['_note'] = editor.status, editor.note
        st.session_state['_editor_identity'] = widget_identity
    source_row = pred.loc[pred.record_id.eq(rid), 'source_row'].iloc[0]
    with st.container(key='record_editor_context'):
        st.caption(f'当前填写：来源第 {source_row} 行')
    st.selectbox('人工逐条判断', STATUSES, key='_status', format_func=lambda s: STATUS_NAMES[s],
                 disabled=locked or not available, on_change=update_editor, args=(widget_identity,))
    st.text_area('人工依据（确认时必填）', key='_note', height=86, max_chars=4000,
                 placeholder='写下核对的日志、记录或业务事实，以及它如何支持你的判断。',
                 disabled=locked or not available, on_change=update_editor, args=(widget_identity,))
    missing = editor.status in ('确认正常', '确认异常') and not human_basis(editor.note)
    if missing:
        st.warning('确认时需要填写本次人工核查的依据。')
    blocked = locked or not available or not dirty(editor) or missing or latest != editor.baseline or editor.conflict
    a, b = st.container(key='queue_save').columns([2, 1])
    if event_origin:
        next_save = a.button('保存并返回事件', key='save_event_review', type='primary', width='stretch', disabled=blocked)
    else:
        next_save = a.button('保存并下一条', key='save_review_next', type='primary', width='stretch',
                             disabled=blocked or editor.next_id is None)
    save = b.button('保存', key='save_review', width='stretch', disabled=blocked)
    if not locked and available and (save or next_save):
        if save_editor(next_save and event_origin is None):
            if next_save and event_origin:
                apply_action('event_selection', event_origin[:4])
            st.rerun()
    if dirty(editor):
        st.caption('有未保存的填写；切栏目会保留，下载前请先保存。')
    if not event_origin and editor.next_id is None:
        st.caption('当前已是最后一条，仍可点击“保存”。')
    if dirty(editor) or editor.conflict or latest != editor.baseline:
        st.button('放弃填写并读取已保存记录', key='reload_review', disabled=locked or not available,
                  on_click=reload_saved_editor)


def evidence_detail_page():
    st.button('返回复核', key='back_to_review', on_click=nav, args=('review',), icon=':material/arrow_back:')
    identity = st.session_state.get('_evidence_detail_identity')
    if (not valid_result or u.selected is None
            or identity != (u.active.key, u.selected, model_identity)
            or u.selected not in set(pred.record_id)):
        st.warning('当前记录的评分依据已失效，请返回复核重新选择记录。')
        return
    row_index = pred.index[pred.record_id.eq(u.selected)][0]
    record = pred.loc[row_index]
    st.markdown(f'### 来源第 {record.source_row} 行 · ' + ('系统提示异常' if record.rf_anomaly else '未提示异常'))
    st.caption(f'输入 CSV 第 {int(u.physical_rows.at[row_index])} 行 · 来源文件：'+literal_markdown(str(record.source_file)))
    render_score_comparison(record, bundle['config'])
    if dirty(u.editor):
        st.caption('未保存的判断和备注已保留，返回复核后可继续填写。')
    tabs = render_evidence_detail(current_evidence(u.selected))
    if tabs is not None:
        with tabs[2]:
            neighbors = current_evidence(u.selected).get('neighbors', {})
            related = neighbors.get('related_records', {})
            options = {rid: [] for members in related.values() for rid in members}
            for why, members in related.items():
                for rid in members:
                    options[rid].append(why)
            if options:
                with st.expander('打开本批关联记录'):
                    locators = pred[pred.record_id.isin(options)].copy()
                    locators['csv_row'] = u.physical_rows
                    options = {r.record_id: f'输入 CSV 第 {r.csv_row} 行 · {"、".join(options[r.record_id])}'
                               for r in locators.itertuples()}
                    st.caption('每种关系最多列出前 20 条，打开后可以核对具体证据。')
                    related_id = st.selectbox('关联行', list(options), format_func=options.get,
                                              key=f'_related_{u.selected}')
                    st.button('进入关联行复核', disabled=u.pending is not None,
                              on_click=request, args=('review_record', related_id))
            with st.expander('当前输入 CSV 原始记录'):
                original = st.session_state.get('_raw_context')
                physical = int(u.physical_rows.at[row_index]) if u.physical_rows is not None else None
                locations = list(original.attrs.get('source_rows', ())) if original is not None else []
                if st.session_state.get('_raw_context_key') == u.active.key and physical in locations:
                    raw_row = original.iloc[[locations.index(physical)]]
                    st.dataframe(raw_row, hide_index=True, width='stretch')
                    st.download_button('下载当前输入行 CSV', csv_bytes(raw_row),
                                       f'{u.active.digest[:12]}-row-{physical}.csv', mime='text/csv')
                else:
                    st.info('无法核对本批原始行位置，请在原 CSV 中查看。')
            with st.expander('字段替换试算与试算 JSON'):
                from app_ui_tools import single_record
                single_record(u, record, row_index, features, bundle, meta, csv_bytes)


@contextmanager
def review_connection():
    with closing(connect(DB)) as db:
        ledger = DB.with_name(DB.name + '.recovery.sqlite')
        db.execute('ATTACH DATABASE ? AS recovery', (str(ledger),))
        recovery.initialize_recovery(db)
        yield db


def review_analysis(event_id=None, budget=None):
    """Build current Gate inputs from the unfiltered server analysis and SQLite snapshot."""
    ids = frozenset(raw_pred.record_id)
    model = bundle['model_version']
    scope_key = f'{u.active.digest}:{model}'
    selected_state = None

    def roster(event, engine):
        if event is None or not engine or 'ow_event' not in raw_pred:
            return frozenset(), frozenset()
        members, reps = event_members_and_representatives(raw_pred, event, engine.get('events', {}).get('reps') or 0)
        return frozenset(members), frozenset(reps)

    if event_id is not None:
        members, _ = roster(event_id, verified_display_engine())
        ids = members
        scope_key += f':event:{event_id}'
    scope = dynamic.Scope(scope_key, model, ids)

    def load(requested_scope, inputs):
        nonlocal selected_state
        if requested_scope != scope:
            return None
        row = raw_pred.iloc[0] if len(raw_pred) else None
        identity = engine_identity(row) if row is not None and set(IDENTITY_COLUMNS) <= set(raw_pred) else ()
        engine = verified_display_engine() if identity else None
        alpha = float(row.risk_alpha) if engine and 'risk_alpha' in row else None
        context = AnalysisContext(meta['input_sha256'], model, meta['model_sha256'],
                                  meta['descriptor_sha256'], identity, alpha)
        engine_status = u.result.engine_status

        def evaluate(search_scope, members, states, current_reviews, limit, requirements=(), release_id=None):
            evidence = {}
            required = {}
            if search_scope.event_id is not None or release_id is not None:
                for rid in members:
                    if rid not in review_evidence:
                        review_evidence[rid] = current_evidence(rid)
                    evidence[rid] = review_evidence[rid]
                sources = ('RF', 'LR', 'cp_set', 'ow_state')
                if release_id is None:
                    sources += ('ecod_tails', 'member_comparison')
                required = {rid: sources for rid in search_scope.authorization_ids | search_scope.pending_release_ids}
            result = search_counterexamples(raw_pred, u.physical_rows, context=context, scope=search_scope,
                                           reviews=current_reviews, evidence=evidence, required=required, budget=limit)
            state = gate.GateState(result, states, context, u.active.key, u.active.key,
                                   engine_status, risk_requirements=requirements, budget=limit)
            return state, gate.evaluate_gate(state, current_reviews, release_id=release_id)

        def member_state(rid):
            view = inputs.decisions[rid]
            return gate.MemberState(str(view.state), view.state != recovery.RecoveryState.INHERITED)

        if event_id is None:
            search_scope = SearchScope()
        else:
            members, reps = roster(event_id, engine)
            if members != ids:
                return None
            choices = {inputs.reviews.get(rid, {}).get('status') for rid in reps}
            conclusion = ANOMALY if choices == {ANOMALY} else NORMAL
            targets = frozenset(rid for rid in members - reps if member_state(rid).eligible)
            search_scope = SearchScope(event_id, conclusion, reps, targets)
        selected_state, decision = evaluate(search_scope, ids, {rid: member_state(rid) for rid in ids},
                                            inputs.reviews, budget)

        rechecks = {}
        for auth in inputs.authorizations:
            old = json.loads(auth.original)['gate']
            old_scope = old['search']['scope']
            release = old['release_id']
            if release is None:
                old_reps = old_scope['representative_ids']
                events = (raw_pred.loc[raw_pred.record_id.isin(old_reps), 'ow_event'].dropna().unique()
                          if engine and 'ow_event' in raw_pred else ())
                current_event = int(events[0]) if len(events) == 1 else None
                members, reps = roster(current_event, engine)
                targets = frozenset(old_scope['authorization_ids']) & members
                check_scope = (SearchScope(current_event, old['conclusion'], reps, targets)
                               if members else SearchScope())
            else:
                members = frozenset({release}) & frozenset(raw_pred.record_id)
                check_scope = SearchScope(pending_release_ids=members)
            current_reviews = inputs.reviews_for(model, tuple(members))
            states = {rid: gate.MemberState('原授权重验', True) for rid in members}
            requirements = tuple(gate.RiskRequirement(req['source'], tuple(req['accepted_values']))
                                 for req in old['risk_requirements'])
            _, rechecks[auth.batch_id] = evaluate(check_scope, members, states, current_reviews,
                                                  old['budget'], requirements, release)
        return dynamic.Analysis(decision.search, decision, rechecks)

    # Workbench calls the loader inside the submit transaction before asking
    # for GateState. Capture the state built from that locked snapshot.
    def load_gate_state(db):
        if not db.in_transaction or selected_state is None:
            raise dynamic.RefreshRequired('需要在提交事务内重验门控')
        return selected_state

    return scope, load, load_gate_state


def refresh_review_scope(db, scope, load):
    current = dynamic.read_current(db, scope, load)
    if current.preview is None:
        current, = dynamic.rebuild(db, scope, load, reason='当前复核或研判依据已更新')
    return current


def review_projection(frame, records, saved_reviews):
    current = {row.record_id: row for row in records}
    view = frame.copy()
    view['review_status'] = [
        current[rid].decision.conclusion or
        ('待补证' if current[rid].decision.action == 'supplement_evidence' else '待复核')
        for rid in frame.record_id
    ]
    view['review_note'] = [saved_reviews.get(rid, {}).get('note', '')
                           if current[rid].decision.review_id is not None else ''
                           for rid in frame.record_id]
    view['authorization_state'] = [str(current[rid].decision.state) for rid in frame.record_id]
    view['authorization_batch'] = [current[rid].inheritance_source or current[rid].withdrawal_batch or ''
                                   for rid in frame.record_id]
    return view


def apply_event_selection(value):
    from liushao import event_workspace

    batch_key, number, asked_member, *section_choice = value
    if not valid_result or batch_key != u.active.key or 'ow_event' not in pred:
        return
    member_rows = pred[pred.ow_event.eq(number)].sort_values('engine_rank', kind='stable')
    ids = tuple(member_rows.record_id)
    if not ids or (asked_member is not None and asked_member not in ids):
        return
    rid = asked_member or ids[0]
    section = section_choice[0] if section_choice else '属性'
    if section not in (*event_workspace.SECTIONS, '进入研判'):
        return
    previous_target = st.session_state.get('_event_target')
    previous_section = (st.session_state.get('_event_section', '属性') if previous_target and
                        previous_target[:2] == (batch_key, number) else '属性')
    if st.session_state.pop('_event_record_origin', None):
        st.session_state['_focus_return_event'] = section
    if u.selected != rid:
        u.editor = None
    u.selected = rid
    u.page = 'events'
    st.session_state['_event_target'] = (batch_key, int(number), rid)
    st.session_state['_event_section'] = '属性' if section == '进入研判' else section
    st.session_state['_ev_mobile_detail'] = True
    st.session_state['_ev_reveal'] = True
    if not section_choice:
        # A result chosen in global search may sit outside the previous event filters.
        prefix = 'ev_' + hashlib.sha256(batch_key.encode()).hexdigest()[:18]
        st.session_state[f'{prefix}_status'] = '全部'
        st.session_state[f'{prefix}_query'] = ''
        st.session_state[f'{prefix}_saved_filters'] = ('全部', '')
    if section == '进入研判':
        started = st.session_state.setdefault('_ev_started', {})
        started.setdefault(batch_key, set()).add(number)
        apply_action('event_record', (batch_key, number, rid, previous_section, rid))


def event_page():
    from liushao import event_workspace

    active_key = u.active.key
    events = event_workspace.event_roster(pred, reviews, st.session_state.get('_ev_started', {}).get(active_key, ()))

    def select(number, rid, section):
        request('event_selection', (active_key, number, rid, section))

    def enter(rid):
        event = next(e for e in events if rid in e.members)
        select(event.number, rid, '进入研判')

    def basis(event):
        try:
            engine = verified_display_engine()
            index = pred.index[pred.record_id.eq(event.members[0])][0]
            names = event_signature_fields(engine, features.loc[[index]])
            return '、'.join(field_name(name) if field_name(name) != '其他数据项' else name for name in names)
        except (OSError, TypeError, KeyError, ValueError):
            return '尾部特征依据暂不可用'

    event_workspace.render_page(
        events, active_key=active_key, target=st.session_state.get('_event_target'),
        section=st.session_state.get('_event_section', '属性'), section_callback=select, record_callback=enter,
        open_queue=lambda: nav('review'), pred=pred, physical=u.physical_rows, config=bundle['config'],
        locked=u.pending is not None, epoch=u.table_epoch, detail_callback=render_event_details, event_basis=basis,
    )


def render_event_details(event, rid, section, gate_slot):
    try:
        if not available:
            raise dynamic.RefreshRequired('复核存储暂不可用')
        engine = verified_display_engine()
        if engine is None:
            raise dynamic.RefreshRequired('事件引擎身份未通过校验')
        members, reps = event_members_and_representatives(pred, event.number, engine.get('events', {}).get('reps') or 0)
        if not members or not reps:
            raise dynamic.RefreshRequired('当前事件没有可核验的代表成员')
        locked = u.pending is not None
        key = f'event_{hashlib.sha256(u.active.key.encode()).hexdigest()[:16]}_{event.number}'
        budget_key = f'{key}_budget'
        saved_budget_key = f'{key}_saved_budget'
        budget = st.session_state.get(saved_budget_key, len(members))
        scope, load, load_gate_state = review_analysis(event.number, int(budget))
        with review_connection() as db:
            current = refresh_review_scope(db, scope, load)
            if current.preview is None:
                raise dynamic.RefreshRequired('当前事件修订待核验')
            export = dynamic.prepare_export(db, scope, load) if section in ('研判历史', '批次处置') else None
            if export is not None and export.revision != current.preview.revision:
                raise dynamic.RefreshRequired('另一窗口更新了研判，请重新读取')
    except (OSError, sqlite3.Error, ReviewConflictError, TypeError, KeyError, ValueError) as error:
        gate_slot.html('<div class="event-gate"><span>证据门控</span><b>暂不可核验</b></div>')
        if section != '属性':
            st.warning(f'本次事件依据暂不可用。{error}')
            st.button('重新读取事件', key='event_retry')
        return

    decision = current.preview.gate
    if decision.passed:
        gate_status = f'可确认 · {len(decision.targets)} 条待授权'
    elif decision.blockers:
        gate_status = f'{len(decision.blockers)} 项条件待处理'
    else:
        gate_status = '当前无可授权成员'
    gate_slot.html(f'<div class="event-gate"><span>证据门控</span><b>{gate_status}</b></div>')
    if section == '属性':
        return

    state = st.session_state.setdefault('_event_batch', {})
    if state.get('scope') != scope:
        state.clear()
        state['scope'] = scope

    def allow_write():
        update_editor()
        return available and u.pending is None and not dirty(u.editor)

    def prepare_batch(preview):
        if allow_write():
            try:
                with review_connection() as db:
                    state['batch'] = workbench.prepare_batch(db, preview, load=load)
                state.pop('error', None)
            except (OSError, sqlite3.Error, ReviewConflictError, TypeError, KeyError, ValueError) as error:
                state['error'] = recovery.RecoveryConflict(str(error))

    def submit_batch(batch, confirmation):
        if allow_write():
            try:
                with review_connection() as db:
                    workbench.submit_batch(db, batch, confirmation, load=load, load_gate_state=load_gate_state)
                state.pop('batch', None)
                state.pop('error', None)
                u.editor = None
                state['receipt'] = f'已授权 {len(confirmation.targets):,} 条 · 操作 {confirmation.operation_id}'
            except (OSError, sqlite3.Error, ReviewConflictError, gate.GateBlocked, TypeError, KeyError, ValueError) as error:
                state['error'] = (error if isinstance(error, (gate.GateBlocked, recovery.RecoveryConflict))
                                  else recovery.RecoveryConflict(str(error)))

    def preview_withdrawal(batch_id):
        if allow_write():
            try:
                with review_connection() as db:
                    state['withdrawal'] = recovery.preview_withdrawal(db, batch_id)
                state.pop('withdrawal_error', None)
            except (OSError, sqlite3.Error, ReviewConflictError, ValueError) as error:
                state['withdrawal_error'] = str(error)

    def submit_withdrawal(preview, operation_id, reason, confirmed):
        if allow_write():
            try:
                with review_connection() as db:
                    recovery.withdraw(db, preview, operation_id=operation_id, reason=reason, confirmed=confirmed)
                state.pop('withdrawal', None)
                state.pop('withdrawal_error', None)
                affected = sum(member.disposition == 'affected' for member in preview.members)
                state['receipt'] = f'已撤回 {affected} 条继承 · 原批次 {preview.batch_id} · 操作 {operation_id}'
            except (OSError, sqlite3.Error, ReviewConflictError, ValueError) as error:
                state['withdrawal_error'] = str(error)

    first = pred.index[pred.record_id.eq(members[0])][0]
    details = workbench.EventDetails(event.number, event.name,
                                     tuple(map(field_name, event_signature_fields(engine, features.loc[[first]]))),
                                     {member: '按事件内固定抽样选取' for member in reps})
    preview = current.preview
    view = workbench.build_workbench(preview.search, preview.gate, current, details, export=export,
                                    batch=state.get('batch'), batch_error=state.get('error'),
                                    withdrawal=state.get('withdrawal'), withdrawal_error=state.get('withdrawal_error'))
    def open_record(member):
        request('event_record', (u.active.key, event.number, rid, section, member))

    actions = workbench.WorkbenchActions(
        lambda number: request('event_selection', (u.active.key, number, None)), open_record,
        lambda: request('review_scope', dict(Filters(scope='全部记录'))),
        lambda entry: open_record(entry.record_id) if entry.record_id else None,
        lambda: (state.pop('batch', None), state.pop('error', None)),
        prepare_batch, submit_batch, preview_withdrawal, submit_withdrawal,
    )
    if state.get('receipt'):
        st.success(state['receipt'])
    if section == '相关证据':
        workbench.render_evidence_section(view, actions, key, locked,
            select_member=lambda member: request('event_selection', (u.active.key, event.number, member, section)),
            current_id=rid, line_numbers=dict(zip(pred.record_id, u.physical_rows)),
            record_lookup=lambda member: pred.loc[pred.record_id.eq(member)].iloc[0],
            thresholds=bundle['config'], evidence_lookup=review_evidence.get)
    elif section == '研判历史':
        workbench.render_history(view.confirmation)
    else:
        workbench.render_batch_section(view.confirmation, actions, key, locked, dirty(u.editor),
                                      line_numbers=dict(zip(pred.record_id, u.physical_rows)))
        with st.expander('调整本事件核查预算'):
            st.number_input('本事件人工核查预算', min_value=0, max_value=len(members), value=budget,
                            key=budget_key, disabled=locked,
                            on_change=lambda: st.session_state.update({saved_budget_key: st.session_state[budget_key]}))


def review_page():
    from app_score_charts import bin_label
    from app_ui_state import conclusion_source, queue_reason
    from liushao import queue_workspace

    if not len(pred):
        st.info('本批没有可复核记录。请查看文件问题。')
        st.button('查看文件问题', on_click=nav, args=('quality',))
        return
    f = u.filters
    locked = u.pending is not None
    queue_workspace.styles()
    origin = st.session_state.get('_event_record_origin')
    if origin and origin[0] == u.active.key and origin[-1] == u.selected:
        with st.container(key='event_record_review'), st.container(key='queue_detail'):
            st.button(f'返回事件 EVT-{origin[1]:04d}', key='event_record_back', icon=':material/arrow_back:',
                      help=f'回到此事件的{origin[3]}', on_click=request, args=('event_selection', origin[:4]), disabled=locked)
            record_panel(u.selected, [u.selected], origin)
        return
    detail_open = st.session_state.get('_queue_detail_batch') == u.active.key
    mode = st.container(key='queue_mode_detail' if detail_open else 'queue_mode_list')
    filters = mode.container(key='queue_filters')

    def control(place, method, name, label, *values, **options):
        key = '_filter_' + name
        st.session_state.setdefault(key, f[name])
        return getattr(place, method)(label, *values, key=key, on_change=filter_change,
                                     args=(name, key), disabled=locked, **options)

    scopes = ['待办记录', '全部记录', '异常提示', '未提示异常', '需要再检查']
    if 'ow_queue' in pred:
        scopes.extend(QUEUE_SCOPES)
    a, b, c = filters.container(key='queue_tools').columns([1.1, 2, .7], vertical_alignment='center')
    control(a, 'selectbox', 'scope', '查看范围', scopes, label_visibility='collapsed')
    control(b, 'text_input', 'keyword', '检索当前范围', label_visibility='collapsed',
            placeholder='搜行号或关键词', help='输入 CSV 行号、来源文件、记录 ID 或完整状态，按回车检索。')
    extra_columns = ['来源行号', '来源文件', '引擎研判', 'RF 排序分数', 'LR 排序分数']
    with c.popover('筛选', width='stretch'):
        control(st, 'selectbox', 'status', '有效处理状态', ['全部'] + STATUSES)
        control(st, 'selectbox', 'queue', '排列方式', ['高异常得分', '待核实', '原始顺序'],
                format_func=lambda key: {'高异常得分':'RF 分数从高到低', '待核实':'优先需要再检查', '原始顺序':'按输入顺序'}[key])
        x, y = st.columns(2)
        control(x, 'selectbox', 'line_kind', '行号含义', ['输入 CSV 行号', '来源行号'])
        control(y, 'text_input', 'line', '精确行号')
        control(st, 'slider', 'minimum', '最低 RF 排序分数', 0.0, 1.0, step=.01)
        for name, label in [('near', '只看 RF 接近分界'), ('disagreement', '只看两模型提示不一致'),
                            ('lr_tie_break', 'RF 同分时优先较高 LR 分数'),
                            ('include_lr_near', '在待办及需再检查范围纳入 LR 接近分界记录')]:
            control(st, 'checkbox', name, label)
        extras = st.multiselect('字段表格的附加列', extra_columns, default=['RF 排序分数'], key='_queue_columns')
        preset = st.session_state.get('_queue_view')
        p, q = st.columns(2)
        p.button('记住本窗口当前视图', key='_queue_remember_view', disabled=locked,
                 on_click=lambda: st.session_state.update(_queue_view=(dict(f), list(extras))))
        def restore_view():
            if preset is not None:
                request('filters', preset[0])
                if u.pending is None:
                    st.session_state['_queue_columns'] = list(preset[1])
        q.button('恢复已记住视图', key='_queue_restore_view', disabled=locked or preset is None,
                 on_click=restore_view)
    tags = [f.scope, f.status, {'高异常得分':'RF 降序', '待核实':'需核查优先', '原始顺序':'输入顺序'}[f.queue]]
    if f.keyword: tags.append('检索 ' + f.keyword)
    if f.line: tags.append(f'{f.line_kind} {f.line}')
    if f.score_bin is not None: tags.append('RF 区间 ' + bin_label(f.score_bin))
    if f.minimum: tags.append(f'RF ≥ {f.minimum:.2f}')
    for name, label in [('near', '接近分界'), ('disagreement', '模型不一致'), ('lr_tie_break', 'LR 同分排序'), ('include_lr_near', '纳入 LR 近界')]:
        if f[name]: tags.append(label)
    table = filtered(pred, f, bundle, physical_rows=u.physical_rows)
    order = table.record_id.tolist()
    positions = {rid: i for i, rid in enumerate(order)}
    a, b = filters.container(key='queue_summary').columns([4, 1], vertical_alignment='center')
    a.caption(' · '.join(literal_markdown(t) for t in tags) + f' ｜ 命中 {len(order):,} 条')
    b.button('清除筛选', key='clear_filters', on_click=request,
             args=('filters', Filters(scope='全部记录')), disabled=locked)
    receipt = st.session_state.get('_queue_receipt')
    if receipt and receipt[0] == u.active.key:
        mode.success(literal_markdown(receipt[1]))
    if not order:
        if dirty(u.editor):
            st.warning('原范围的草稿仍保留。清除筛选可返回该记录。')
        else:
            u.selected = u.editor = None
        st.info('当前条件没有匹配记录。')
        return
    follow = st.session_state.pop('_queue_next', None)
    if follow and follow[0] == u.active.key:
        u.selected = next((rid for rid in follow[1] if rid in positions), None)
    pages = (len(order) + 49) // 50
    u.page_index = min(max(0, u.page_index), pages - 1)
    if u.selected in positions:
        u.page_index = positions[u.selected] // 50
    elif dirty(u.editor):
        st.warning('当前草稿来自已移出筛选范围的记录，填写内容已保留。')
        st.button('打开原记录与草稿', on_click=request, args=('review_record', u.editor.record_id))
        return
    else:
        u.selected, u.editor = order[u.page_index * 50], None
    start = u.page_index * 50
    chunk = table.iloc[start:start + 50]
    with mode.container(key='queue_workspace'):
        listing, detail = st.columns([32, 68], gap='small')
        with listing.container(key='queue_list'):
            rows = pd.DataFrame(index=chunk.index)
            rows['当前'] = chunk.record_id.map(lambda rid: '▸' if rid == u.selected else '')
            rows['输入 CSV 行号'] = chunk.get('input_csv_row')
            rows['复核理由'] = chunk.apply(lambda r: queue_reason(r, bundle['config'], include_lr_near=f.include_lr_near), axis=1)
            rows['系统提示'] = chunk.rf_anomaly.map({True:'提示异常', False:'未提示异常'})
            rows['有效状态'] = chunk.review_status
            rows['结论来源'] = chunk.apply(conclusion_source, axis=1)
            for name, column in zip(extra_columns, ['source_row','source_file','ow_state','rf_score','lr_score']):
                if name in extras and column in chunk:
                    rows[name] = chunk[column]
            ids = list(chunk.record_id)
            labels = {rid: f'输入 CSV 第 {row} 行' for rid, row in zip(ids, rows['输入 CSV 行号'])}
            with st.container(key='queue_rows', height=430 if len(rows) > 7 else 'content', border=False):
                for rid, (line, status, reason) in zip(ids, rows[['输入 CSV 行号','有效状态','复核理由']].itertuples(index=False, name=None)):
                    label = f'**第 {line} 行**　{status}  \n{reason}'
                    st.button(label, key=f'queue_pick_{rid}', type='primary' if rid == u.selected else 'secondary',
                              width='stretch', disabled=locked, on_click=open_queue_row, args=(rid,))
            x, y, z = st.container(key='queue_pager').columns([1, 1, 1], vertical_alignment='center')
            x.button('上一页', key='prev_page', disabled=locked or u.page_index == 0, on_click=page_to, args=(u.page_index-1,))
            y.caption(f'第 {u.page_index+1} / {pages} 页')
            z.button('下一页', key='next_page', disabled=locked or u.page_index == pages-1, on_click=page_to, args=(u.page_index+1,))
            with st.expander('字段表格与键盘定位'):
                key = '_records_' + str(u.table_epoch)
                st.dataframe(rows, hide_index=True, width='stretch', height=min(430, 38 + 35 * len(rows)),
                             selection_mode='single-row', key=key,
                             on_select='ignore' if locked else lambda: row_selected(key, ids))
                choose = f'_queue_jump_{u.table_epoch}_{u.page_index}_{u.selected}'
                # Streamlit may resend the preceding select value while a different row is
                # rendering. Only an explicit form submission may navigate to that value.
                with st.form('queue_jump_form', border=False):
                    st.selectbox('键盘选择本页记录', ids, index=ids.index(u.selected),
                                 format_func=labels.get, key=choose, disabled=locked)
                    st.form_submit_button('打开所选记录', disabled=locked,
                                          on_click=lambda: open_queue_row(st.session_state[choose]))
        with detail.container(key='queue_detail'):
            st.container(key='queue_back').button('返回记录列表', icon=':material/arrow_back:',
                                                on_click=set_queue_pane, args=(False,), disabled=locked)
            record_panel(u.selected, order)
    focus_detail = st.session_state.pop('_focus_queue_detail', None)
    if focus_detail is not None:
        queue_workspace.focus_pane(focus_detail)


def quality_page():
    quality = u.quality
    if quality is None:
        st.warning('本次无法生成文件问题明细。分析结果和已保存的操作仍可查看。')
        with st.expander('查看详情'):
            st.write(u.quality_error)
        return
    if not quality.affected_rows:
        st.success('没有发现缺失数据或无法分析的记录。')
        return
    st.write(f'**{quality.affected_rows:,} 条记录存在数据问题** · {quality.accepted_problem_rows:,} 条填补后分析 · {quality.rejected_rows:,} 条无法分析')
    records = quality.records
    with st.container(horizontal=True, gap='small'):
        st.download_button('下载完整问题清单', csv_bytes(records), f'文件问题_{u.active.digest[:10]}.csv', 'text/csv')
        if len(rejected):
            st.download_button('下载全部无法分析的记录', csv_bytes(rejected), f'排除记录_{u.active.digest[:10]}.csv', 'text/csv')
    locate = st.text_input('查找文件行号', key='_quality_line', placeholder='上传文件中的行号')
    if locate.strip():
        number = line_number(locate)
        records = records[records.input_row.eq(number)] if number is not None else records.iloc[:0]
    if records.empty:
        st.info('没有符合行号的问题记录。')
        return
    pages = max(1, (len(records) + 49) // 50)
    u.quality_page = max(0, min(u.quality_page, pages-1))
    if pages > 1:
        with st.container(horizontal=True, gap='small', vertical_alignment='center'):
            if st.button('上一页问题', disabled=u.quality_page == 0):
                u.quality_page -= 1
                st.rerun()
            st.caption(f'问题记录第 {u.quality_page+1} / {pages} 页')
            if st.button('下一页问题', disabled=u.quality_page == pages-1):
                u.quality_page += 1
                st.rerun()
    records = records.iloc[u.quality_page*50:(u.quality_page+1)*50]
    ids = records.index.tolist()
    chosen = u.get('quality_selected')
    if chosen not in ids:
        chosen = ids[0]
    labels = {rid: f'第 {records.loc[rid, "input_row"]} 行' for rid in ids}
    if st.session_state.get('_quality_record') not in ids:
        st.session_state['_quality_record'] = chosen
    chosen = st.selectbox('选择问题记录', ids, format_func=labels.get, key='_quality_record',
                          on_change=lambda: setattr(u, 'quality_selected', st.session_state['_quality_record']))
    u.quality_selected = chosen
    with st.container(key='quality_record_detail'):
        item = records.loc[chosen]
        st.markdown(f'**文件第 {item.input_row} 行**')
        rejected_item = any(word in str(item.disposition) for word in ('拒绝', '无法判断'))
        st.write('这条记录无法分析，请补齐数据后重新上传。' if rejected_item else '部分数据缺失；系统已补齐后分析，建议核对原文件。')
        details = quality.details(chosen).copy()
        if len(details):
            details['字段'] = details['字段'].map(field_name)
            details = details.rename(columns={'字段': '数据项', 'CSV读取值': '文件中的内容', '适配结果': '发现的问题', '处理状态': '处理方式'})
            if '处理方式' in details: details['处理方式'] = '无法分析' if rejected_item else '补齐后分析'
            st.dataframe(details, hide_index=True, width='stretch')
        else:
            st.info('没有更具体的数据项原因，请查看原文件。')
        with st.expander('查看原始问题信息'):
            st.write(f'来源：{item.source_file}，第 {item.source_row} 行')
            st.write(item.reason)
            st.dataframe(quality.details(chosen), hide_index=True)
    with st.expander('整批问题字段汇总'):
        summary = quality.summary
        summary = summary[summary['问题单元格数'].gt(0) | summary['整行拒绝涉及数'].gt(0)].copy()
        summary['字段'] = summary['字段'].map(field_name)
        st.dataframe(summary.rename(columns={'字段': '数据项', '可判断记录缺失数': '缺失记录数', '整行拒绝涉及数': '无法分析记录数', '问题单元格数': '问题数量'}), hide_index=True, width='stretch')



DOWNLOAD_DESCRIPTIONS = {
    'predictions': '检查结果 CSV：全部可判断记录的分数、提示、来源行号及当前有效结论。',
    'history': '操作记录 JSON：已保存的逐条判断、证据和批次授权、失效及撤回记录。',
    'manifest': '运行信息 JSON：输入文件、模型版本、判断分界、记录数量和存储状态。',
}


def prepare_review_exports(choice='整个批次', event=None, scope_name='整个批次'):
    from trial_build import BUILD_LABEL
    with review_connection() as db:
        refresh_review_scope(db, review_scope, review_loader)
        export = dynamic.prepare_export(db, review_scope, review_loader)
    entries = [json.loads(row) for row in export.history]
    direct = [row for row in entries if 'record_id' in row]
    by_id = {row['id']: row for row in direct}
    latest = {row['record_id']: row for row in direct}
    projected = review_projection(raw_pred, export.checks, latest)
    if choice != '整个批次':
        projected = export_members(projected,choice,event)
    ids = set(projected.record_id)
    physical = u.physical_rows.reindex(projected.index)
    selected_direct = [r for r in direct if r['record_id'] in ids]
    projected = projected.assign(input_source=u.active.source, trial_build=BUILD_LABEL,
                                  export_revision=export.revision.checkpoint)
    snapshot = export_snapshot(u.active, projected, selected_direct, meta, True, physical_rows=physical)
    conclusions = [r for r in export.conclusions if r.record_id in ids]
    direct_conclusions = [by_id[row.decision.review_id] for row in conclusions if not row.inheritance_source]
    frame, missing = confirmed_frame(u.active, projected, direct_conclusions, meta, features, u.physical_rows,
                                     bundle, field_name, generated_at=export.exported_at)
    frame['结论来源'] = '直接确认'
    frame['原授权批次ID'] = ''
    inherited = []
    operations = {row['id']: row for row in entries if row.get('kind') == 'authorize'}
    for row in conclusions:
        if row.inheritance_source is None:
            continue
        entry = operations[row.inheritance_source]
        frozen = json.loads(entry['evidence'])['gate']['search']['context']
        inherited.append({
            '输入CSV行号': row.physical_row or '未记录', '我的判断': row.decision.conclusion,
            '备注': entry['reason'], '复核保存时间UTC': entry['created_at'],
            '保存时证据摘要': '按原批次授权 ID 在操作记录中查看门控和依赖证据',
            '结论单生成时间UTC': export.exported_at, '记录ID': row.record_id,
            '模型版本': frozen['model_version'], '保存时输入SHA256': frozen['input_sha256'],
            '保存时模型SHA256': frozen['model_sha256'], '保存时描述SHA256': frozen['descriptor_sha256'],
            '结论来源': '继承有效', '原授权批次ID': row.inheritance_source,
        })
        missing += row.physical_row is None
    if inherited:
        frame = pd.concat([frame, pd.DataFrame(inherited).reindex(columns=frame.columns, fill_value='未记录')], ignore_index=True)
    frame['结论来源说明'] = frame['结论来源'].map({'直接确认':'人工逐条确认','继承有效':'由事件继承'})
    frame['本次输入来源'] = u.active.source
    frame['批次SHA256'] = u.active.digest
    frame['发布构建'] = BUILD_LABEL
    frame['复核修订'] = export.revision.checkpoint
    snapshot.confirmed = csv_bytes(frame) if len(conclusions) else None
    snapshot.confirmed_count = len(conclusions)
    snapshot.missing_input_rows = missing
    snapshot.history = json.dumps(entries, ensure_ascii=False, indent=2).encode('utf-8')
    snapshot.meta.update(confirmed_count=len(conclusions), exported_at=export.exported_at,
                         revision=export.revision.checkpoint, export_scope=scope_name,
                         filters=dict(u.filters) if choice=='当前复核筛选' else None,
                         included_rows=len(ids), excluded_valid_rows=len(raw_pred)-len(ids),
                         excluded_unconfirmed=len(ids)-len(conclusions), history_scope='整个批次')
    snapshot.manifest = json.dumps(snapshot.meta, ensure_ascii=False, indent=2).encode('utf-8')
    return snapshot, export


def send_download(content, filename, mime):
    from streamlit.components.v1 import html

    # Streamlit 1.50 download_button publishes a reusable media URL before
    # its callback runs. Send bytes only in the response to a checked click.
    payload = base64.b64encode(content).decode('ascii')
    html(f"""<body><script>
    // {uuid4().hex}
    const encoded = "{payload}";
    const parts = [];
    for (let i = 0; i < encoded.length; i += 32768) {{
        parts.push(Uint8Array.from(atob(encoded.slice(i, i + 32768)), c => c.charCodeAt(0)));
    }}
    const url = URL.createObjectURL(new Blob(parts, {{type: {json.dumps(mime)}}}));
    const link = document.createElement('a');
    link.href = url;
    link.download = {json.dumps(filename)};
    document.body.appendChild(link);
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
    </script></body>""", height=0)


def checked_download(label, content, filename, mime, key, *, primary=False):
    if not st.button(label, key=key, type='primary' if primary else 'secondary',
                     disabled=u.pending is not None):
        return
    token = st.session_state.get('_review_export')
    try:
        if token is None or token.revision.scope != review_scope:
            raise dynamic.RefreshRequired('当前导出待重新生成')
        with review_connection() as db:
            dynamic.download_export(db, token, review_loader)
    except (OSError, sqlite3.Error, ReviewConflictError, TypeError, KeyError, ValueError) as error:
        st.warning(f'导出已过期或无法核验，请重新生成下载文件。{error}')
        return
    try:
        local.log_download(DB,u.active,filename,token.revision.checkpoint,u.exports.meta['export_scope'])
    except (OSError,ValueError,sqlite3.Error) as error:
        st.warning(f'下载交付记录未保存，请重试。{error}')
        return
    send_download(content, filename, mime)


def export_members(frame, choice, event):
    if choice == '当前复核筛选':
        return filtered(frame,u.filters,bundle,physical_rows=u.physical_rows)
    if choice == '选定事件':
        return frame[frame.ow_event.eq(event)]
    return frame


def download_page():
    if dirty(u.editor) or (u.editor and u.editor.conflict):
        st.warning('保存尚未成功确认：请先保存或放弃当前填写，再生成复核交付文件。')
        st.button('返回填写位置', key='return_editor', on_click=nav, args=('review',))
        st.button('生成下载文件',key='prepare_exports',disabled=True)
        return
    try:
        if not available or current_review is None or current_review.preview is None:
            st.warning('本批复核修订暂不可核验，仅可下载运行信息。')
            st.download_button('下载运行信息',json.dumps(meta,ensure_ascii=False,indent=2),
                               f'运行信息_{u.active.digest[:10]}.json','application/json')
            return
        choices = ['整个批次','当前复核筛选']
        event_options = sorted(int(e) for e in pred.get('ow_event',pd.Series(dtype=float)).dropna().unique() if int(e)>=0)
        if event_options:
            choices.append('选定事件')
        choice = st.radio('导出范围',choices,key='_export_range',horizontal=True)
        event = None
        if choice == '选定事件':
            target = st.session_state.get('_event_target')
            selected = target[1] if target and target[0]==u.active.key and target[1] in event_options else event_options[0]
            event = st.selectbox('来源事件',event_options,index=event_options.index(selected),
                                 format_func=lambda e:f'EVT-{e:04d}',key='_export_event')
        scope_name = f'事件 EVT-{event:04d}' if event is not None else choice
        subset = export_members(pred,choice,event)
        ids = set(subset.record_id)
        decisions = [r for r in current_review.records if r.record_id in ids]
        direct = sum(r.decision.state==recovery.RecoveryState.DIRECT for r in decisions)
        inherited = sum(r.decision.state==recovery.RecoveryState.INHERITED for r in decisions)
        st.write(f'{literal_markdown(u.active.name)} · 批次 {u.active.digest[:12]} · {scope_name}')
        st.markdown(f'**可导出有效结论 {direct+inherited:,} 条**')
        st.write(f'人工逐条确认 {direct:,} 条、有效继承 {inherited:,} 条。')
        st.caption(f'范围内 {len(ids):,} 条，其中 {len(ids)-direct-inherited:,} 条无有效结论。'
                   f'范围外可分析 {len(pred)-len(ids):,} 条 · 导入时排除 {len(rejected):,} 条。')
        st.caption('审计留档保留整批历史以支持追溯。')
        basis = (choice,event,dict(u.filters) if choice=='当前复核筛选' else None)
        fixed, token = u.exports, st.session_state.get('_review_export')
        current = (fixed is not None and token is not None and
                   token.revision==current_review.preview.revision and
                   basis==st.session_state.get('_export_basis'))
        if not current:
            if fixed is not None:
                st.info('修订或范围已改变。请重新生成当前文件。')
            if st.button('更新导出文件' if fixed else '生成下载文件',type='primary',key='prepare_exports',
                         disabled=u.pending is not None):
                u.exports, st.session_state['_review_export'] = prepare_review_exports(choice,event,scope_name)
                st.session_state['_export_basis'] = basis
                st.rerun()
            return
        suffix = ('演示_' if u.active.source=='内置演示数据' else '')+u.active.digest[:10]+'_'+token.revision.checkpoint[:8]
        if event is not None:
            suffix += f'_EVT-{event:04d}'
        elif choice == '当前复核筛选':
            suffix += '_筛选'+hashlib.sha256(json.dumps(basis,sort_keys=True,ensure_ascii=False).encode()).hexdigest()[:6]
        for title,content,ext,key in [('复核结论单',fixed.confirmed,'csv','confirmed'),
                                     ('分析明细',fixed.predictions,'csv','predictions'),
                                     ('整批审计记录',fixed.history,'json','history'),
                                     ('运行信息',fixed.manifest,'json','manifest')]:
            if content is not None:
                checked_download('下载'+title,content,title+'_'+suffix+'.'+ext,
                                 'text/csv' if ext=='csv' else 'application/json',
                                 'export_'+key,primary=key=='confirmed')
        if fixed.confirmed is None:
            st.caption('此范围还没有有效的确认结论。')
        st.button('返回继续复核',on_click=nav,args=('review',))
    except Exception as error:
        st.error(f'导出失败，未生成可下载文件。{error}')



def help_page():
    st.markdown("1. 在 **流量检测** 上传 CSV，或运行内置演示。查看格式预检与文件问题。\n"
                "2. 在 **事件研判** 选择事件，打开 **查看相关证据**。先看系统为何选中这条记录，再按按钮上的 CSV 行号打开它。\n"
                "3. 在 **核查重点** 对照字段原值和训练参考，再查建议的日志或工单。在人工依据栏记下观察并保存，点 **返回事件** 继续。材料不足时可选 **待补证**。\n"
                "4. 在 **审计记录** 回查判断、授权与撤回，按批次、事件或筛选范围导出。")
    with st.expander('CSV 格式练习'):
        if bundle and bundle['config'].get('features'):
            kind = st.selectbox('练习内容', ['缺少必需列','缺值与无效记录'])
            st.download_button('下载格式练习 CSV',problem_sample(bundle['config']['features'],kind),
                               '格式练习.csv','text/csv')
    with st.expander('文件和保存位置'):
        st.write('已保存记录在下方本机数据库中。换电脑或删除数据库后不会自动恢复；下载的操作记录用于留档，暂不支持导入。')
        st.code(str(DB))
    with st.expander('模型与评测'):
        render_evaluation(EVALUATION, verified, bundle)
    with st.expander('怎样理解提示'):
        st.write('双模型判断不一致或接近分界时，结合来源记录复核。“文件问题”列出缺失数据和格式问题。')


def first_batch():
    st.subheader('选择数据来源')
    st.caption('运行演示可直接查看事件和成员证据。检查自己的数据时，选择流量特征 CSV。')
    actions = st.container(horizontal=True, key='first_actions')
    if actions.button('运行内置演示', key='first_demo', type='primary', disabled=bundle is None):
        try:
            request('run', ((ARTIFACTS / 'demo.csv').read_bytes(), 'demo.csv', '内置演示数据'))
        except OSError as error:
            u.run_error = present_error(error, '读取演示文件')
        if u.run_error:
            u.page = 'detection'
        st.rerun()
    def to_upload():
        st.session_state['_source'] = u.stage_source = '上传 CSV'
        nav('detection')
    actions.button('导入自己的 CSV', on_click=to_upload)
    resume_card()
    recent_batches()


def recent_batches():
    try:
        history = local.recent(DB)
    except (OSError, ValueError, sqlite3.Error) as error:
        show_local_file_error(error)
        return
    if history:
        with st.expander('最近批次', expanded=not valid_result):
            by_key = {(r['digest'], r['model']):r for r in history}
            chosen = st.selectbox('选择最近批次',list(by_key),
                format_func=lambda k: f"{by_key[k]['name']} · {k[0][:10]} · 上次剩余 {by_key[k]['state'].get('pending','未知')} 条待办")
            item = by_key[chosen]
            if item['model'] != model_identity:
                st.caption('恢复这个批次需要此前的模型版本。')
            elif valid_result and chosen == (u.active.digest,u.active.model):
                st.button('打开当前批次复核',on_click=nav,args=('review',))
            elif item['source'] == '内置演示数据':
                if st.button('重新打开此演示批次'):
                    content = (ARTIFACTS/'demo.csv').read_bytes()
                    if hashlib.sha256(content).hexdigest() == item['digest']:
                        request('run',(content, 'demo.csv','内置演示数据'))
                        st.rerun()
                    else:
                        st.warning('演示内容已改变，无法匹配此前批次。')
            else:
                cached = saved_upload(DB)
                if cached and cached['sha256'] == item['digest'] and cached['model'] == item['model']:
                    st.button('恢复已保留的文件', on_click=request,args=('restore',(cached['generation'],cached['id'])))
                else:
                    def choose_file():
                        st.session_state['_expected_batch'] = item['digest']
                        st.session_state['_source'] = u.stage_source = '上传 CSV'
                        u.show_import = True
                        nav('detection')
                    st.caption('此批次未保留原 CSV。重新选择相同内容可恢复原来的判断和位置。')
                    st.button('重新选择此批次的 CSV',on_click=choose_file)


def workspace_overview():
    file_bar()
    pending = filtered(pred, dict(default_filters(),scope='待办记录'), bundle, physical_rows=u.physical_rows)
    import altair as alt
    from app_score_charts import expert_hint_summary
    try:
        from liushao import event_workspace as _ew
        _events = _ew.event_roster(pred, reviews, st.session_state.get('_ev_started', {}).get(u.active.key, ()))
        _members = sum(len(e.members) for e in _events)
    except Exception:
        _events, _members = [], 0
    rf_hit = int(pred.rf_anomaly.sum()) if len(pred) and 'rf_anomaly' in pred else 0
    with st.container(key='overview_stats'):
        metrics = st.columns(4)
        metrics[0].metric('可分析记录', f'{len(pred):,}')
        metrics[1].metric('RF 异常提示', f'{rf_hit:,}', help='提示不等于攻击')
        metrics[2].metric('待办记录', f'{len(pending):,}' if current_review is not None else '暂不可用')
        metrics[3].metric('研判事件', f'{len(_events):,}' if _events else '—')
        if _events:
            metrics[3].caption(f'{_members:,} 条成员记录')
    try:
        summary = expert_hint_summary(pred, bundle['config'] if bundle else {})
    except Exception:
        summary = {'rf': None, 'lr': None, 'cross': None, 'ow_state': {}}
    experts = st.columns(2, gap='large')
    with experts[0]:
        st.subheader('多专家提示对比')
        rows = []
        for key, label in (('rf', 'RF'), ('lr', 'LR')):
            e = summary.get(key)
            if e and e['valid']:
                rows.append({'专家': label, '提示数': e['hints'], '有效数': e['valid'],
                             '占比': e['hints'] / e['valid'],
                             '标注': f"{e['hints']:,} / {e['valid']:,} · {e['hints'] / e['valid']:.1%}"})
        if rows:
            frame = pd.DataFrame(rows)
            bars = alt.Chart(frame).mark_bar(height=16, cornerRadius=4, color='#5e6ad2').encode(
                x=alt.X('占比:Q', title='提示占比（分母为该专家有效记录）',
                        axis=alt.Axis(format='.0%'), scale=alt.Scale(domain=[0, 1])),
                y=alt.Y('专家:N', title=None, sort=['RF', 'LR']),
                tooltip=['专家', '提示数', '有效数'],
            )
            labels = alt.Chart(frame).mark_text(align='left', dx=4, fontSize=12, color='#3c3f4a').encode(
                x='占比:Q', y=alt.Y('专家:N', title=None, sort=['RF', 'LR']), text='标注:N')
            st.altair_chart((bars + labels).properties(height=52 * len(frame) + 18), use_container_width=True)
            st.caption('仅展示当前引擎已启用且有冻结提示规则的专家（RF / LR）。专家阈值提示 ≠ 攻击标签；缺失不计作未提示。')
        else:
            st.caption('暂无可统计的专家输出。')
    with experts[1]:
        st.subheader('RF / LR 提示一致性')
        cross = summary.get('cross')
        if cross:
            cells = pd.DataFrame([
                {'RF': 'RF 提示', 'LR': 'LR 提示', '记录数': cross['both_hint'], '类别': '同向'},
                {'RF': 'RF 提示', 'LR': 'LR 未提示', '记录数': cross['rf_only'], '类别': '分歧'},
                {'RF': 'RF 未提示', 'LR': 'LR 提示', '记录数': cross['lr_only'], '类别': '分歧'},
                {'RF': 'RF 未提示', 'LR': 'LR 未提示', '记录数': cross['neither'], '类别': '同向'},
            ])
            rect = alt.Chart(cells).mark_rect(cornerRadius=6, stroke='#ffffff', strokeWidth=3).encode(
                x=alt.X('LR:N', title=None, sort=['LR 提示', 'LR 未提示'],
                        axis=alt.Axis(orient='top', labelFontSize=12, labelFontWeight=600, labelAngle=0)),
                y=alt.Y('RF:N', title=None, sort=['RF 提示', 'RF 未提示'],
                        axis=alt.Axis(labelFontSize=12, labelFontWeight=600)),
                color=alt.Color('类别:N', legend=None, scale=alt.Scale(
                    domain=['同向', '分歧'], range=['#e8eaf3', '#f0dfba'])),
                tooltip=['RF', 'LR', '记录数'],
            )
            text = alt.Chart(cells).mark_text(fontSize=16, fontWeight=600, color='#2e3038').encode(
                x=alt.X('LR:N', sort=['LR 提示', 'LR 未提示']),
                y=alt.Y('RF:N', sort=['RF 提示', 'RF 未提示']),
                text=alt.Text('记录数:Q', format=','),
            )
            st.altair_chart((rect + text).properties(height=150), use_container_width=True)
            st.caption(f"共同有效 {cross['both_valid']:,} 条；分歧 {cross['rf_only'] + cross['lr_only']:,} 条。"
                       '同向与分歧均为提示层统计，不等于 D-S 冲突量，也不等于错误。')
        else:
            st.caption('RF 与 LR 尚无共同有效记录可供交叉统计。')
    ow = summary.get('ow_state') or {}
    if ow:
        st.subheader('引擎融合研判状态')
        order = ['攻击', '新异', '模糊', '正常']
        palette = {'攻击': '#c75c59', '新异': '#b28b38', '模糊': '#9a9da6', '正常': '#419779'}
        names = [n for n in order if n in ow] + [n for n in ow if n not in order]
        total = sum(ow.values())
        seg_rows, acc = [], 0
        for name in names:
            seg_rows.append({'状态': name, '起': acc, '止': acc + ow[name], '记录数': ow[name]})
            acc += ow[name]
        frame = pd.DataFrame(seg_rows)
        stacked = alt.Chart(frame).mark_bar(height=16, cornerRadius=4).encode(
            x=alt.X('起:Q', title=None, axis=None), x2='止:Q',
            color=alt.Color('状态:N', legend=None, scale=alt.Scale(
                domain=names, range=[palette.get(n, '#7a7e8a') for n in names])),
            tooltip=['状态', '记录数'],
        ).properties(height=34)
        st.altair_chart(stacked, use_container_width=True)
        legend = '　'.join(f'<span style="color:{palette.get(n, "#7a7e8a")}">●</span> {n} {ow[n]:,}' for n in names)
        st.markdown(legend, unsafe_allow_html=True)
        st.caption(f'分母为本批次有引擎研判状态的 {total:,} 条记录；该状态为 D-S 证据融合与共形预测的真实引擎输出，与人工结论分开统计。')
    bottom = st.columns(2, gap='large')
    with bottom[0]:
        st.subheader('事件级归并')
        if _events:
            merge_rows = pd.DataFrame([
                {'类别': '成员记录', '数量': _members},
                {'类别': '研判事件', '数量': len(_events)},
            ])
            merge_chart = alt.Chart(merge_rows).mark_bar(height=15, cornerRadius=4).encode(
                x=alt.X('数量:Q', title=None, axis=alt.Axis(tickCount=5),
                        scale=alt.Scale(domain=[0, max(_members, 1)])),
                y=alt.Y('类别:N', title=None, sort=['成员记录', '研判事件']),
                color=alt.Color('类别:N', legend=None, scale=alt.Scale(
                    domain=['成员记录', '研判事件'], range=['#c9ccd4', '#5e6ad2'])),
                tooltip=['类别', '数量'],
            ).properties(height=88)
            st.altair_chart(merge_chart, use_container_width=True)
            _cut = 1 - len(_events) / _members if _members else 0
            st.caption(f'当前事件成员记录 → 研判事件：数量减少 {_cut:.1%} · 非准确率提升。进入事件研判可查看每个事件的成员、证据与门控状态。')
        else:
            st.caption('当前批次没有可归并的疑似事件。')
    with bottom[1]:
        st.subheader('复核进度')
        if current_review:
            decisions = [r.decision for r in current_review.records]
            direct = sum(d.state == recovery.RecoveryState.DIRECT for d in decisions)
            inherited = sum(d.state == recovery.RecoveryState.INHERITED for d in decisions)
            seg = [('待办', len(pending), '#b28b38'), ('人工逐条确认', direct, '#419779'),
                   ('事件有效继承', inherited, '#5e6ad2'), ('无法分析', len(rejected), '#9a9da6')]
            total = sum(v for _, v, _ in seg)
            if total:
                rows, acc = [], 0
                for name, value, color in seg:
                    if value:
                        rows.append({'状态': name, '起': acc, '止': acc + value, '记录数': value})
                        acc += value
                frame = pd.DataFrame(rows)
                stacked = alt.Chart(frame).mark_bar(height=16, cornerRadius=4).encode(
                    x=alt.X('起:Q', title=None, axis=None),
                    x2='止:Q',
                    color=alt.Color('状态:N', legend=None, scale=alt.Scale(
                        domain=[s[0] for s in seg], range=[s[2] for s in seg])),
                    tooltip=['状态', '记录数'],
                ).properties(height=34)
                st.altair_chart(stacked, use_container_width=True)
            legend = '　'.join(f'<span style="color:{c}">●</span> {n} {v:,}' for n, v, c in seg)
            st.markdown(legend, unsafe_allow_html=True)
            if not len(pending):
                st.caption('本批次当前没有待办记录。')
        else:
            st.caption('尚无复核记录。')
    actions = st.container(horizontal=True, key='overview_actions')
    if current_review is not None:
        actions.button('继续上次复核' if len(pending) else '查看复核记录', key='overview_continue',
                       type='primary', on_click=nav, args=('review',))
    actions.button('查看事件', key='overview_events', on_click=nav, args=('events',))
    actions.button('导出复核结论单', key='overview_export', on_click=nav, args=('download',))
    recent_batches()


def detection_page():
    if not valid_result or u.show_import:
        import_card()
        if valid_result:
            st.divider()
    if not valid_result:
        return
    file_bar()
    if u.quality is not None:
        st.write(f'填补后可分析 {u.quality.accepted_problem_rows:,} 条 · 无法分析 {len(rejected):,} 条')
    st.button('查看文件问题与清单', key='detection_quality', on_click=nav, args=('quality',))
    if len(pred):
        render_histogram(raw_pred,len(rejected),bundle['config'],on_bin=open_score_bin,
                         chart_key='batch_scores_'+u.active.digest)
    with st.expander('时间分布'):
        timestamps = pd.concat([p for p in (pred, rejected) if 'timestamp' in p], ignore_index=True)
        covered = timestamps.dropna(subset=['timestamp'])
        st.caption(f'可解析 {len(covered):,} / {total:,} 条 · 排除缺失或格式不支持 {total-len(covered):,} 条。支持日/月/年 时:分:秒和 ISO 8601。')
        if len(covered):
            covered = covered.copy()
            covered['timestamp_basis'] = covered.get('timestamp_basis','文件本地时间（时区未提供）')
            for basis, group in covered.groupby('timestamp_basis'):
                st.caption(f'每小时记录数 · {basis}')
                group = group.assign(时段=group.timestamp.dt.floor('h'), 记录量=1,
                                     系统提示异常=group.rf_anomaly.eq(True).astype(int),
                                     人工逐条确认异常=(group.review_status.eq(ANOMALY) &
                                        group.authorization_state.eq(str(recovery.RecoveryState.DIRECT))).astype(int)
                                     if 'authorization_state' in group else 0)
                times = group.groupby('时段')[['记录量','系统提示异常','人工逐条确认异常']].sum()
                st.bar_chart(times[['记录量']],color='#777ac8')
                st.dataframe(times, width='stretch')
    render_risk_controls()
    with st.expander('其他分析工具'):
        st.button('辅助检查',on_click=nav,args=('auxiliary',))
        st.button('整批数据分析',on_click=nav,args=('batch',))


def preferences_page():
    st.subheader('本机数据')
    ledger = DB.with_name(DB.name+'.recovery.sqlite')
    from app_session import session_root
    shell.local_paths([('逐条复核与证据', DB), ('事件授权、失效与撤回', ledger),
                       ('原 CSV 缓存、最近位置与下载记录', session_root(DB))])
    st.subheader('数据清理')
    action = st.selectbox('清理范围',['仅清除已保留的原 CSV','清除复核与事件历史','清除上述全部数据'])
    clear_history = action != '仅清除已保留的原 CSV'
    clear_cache = action != '清除复核与事件历史'
    if clear_history:
        st.warning('将清除所有批次的逐条判断、依据、授权、撤回、下载记录与最近位置。清理后无法在应用内恢复。')
    else:
        st.caption('清理后，重新上传相同文件仍可读取已保存的复核记录。')
    confirm = st.checkbox('我已核对清理范围',key='clear_ack_'+action)
    if u.editor and dirty(u.editor) and clear_history:
        st.warning('清理复核历史也会丢弃当前未保存的填写。')
    if st.button('执行清理', key='clear_data',disabled=not confirm):
        messages = []
        try:
            if clear_history:
                with review_connection() as db:
                    counts = local.clear_reviews(db, DB)
                messages.append(f"已清除 {counts['main.reviews']} 条逐条操作、{counts['recovery.inheritances']} 条继承记录")
                u.result = u.active = u.selected = u.editor = u.pending = u.exports = None
                u.quality = u.physical_rows = u.batch = u.clues = u.diagnostics = u.percentiles = None
                u.tool_downloads.clear()
                u.stage_upload = u.preflight = u.preflight_key = u.preflight_error = None
                u.suspended_draft = None
                u.suspended_drafts.clear()
                for key in list(st.session_state):
                    if key.startswith(('_event_','_queue_','_review_export','_editor_identity','_position_fingerprint')):
                        st.session_state.pop(key,None)
            if clear_cache:
                clear_upload(DB)
                u.save_generation = u.remember_key = None
                st.session_state['_remember'] = False
                messages.append('已清除保留的 CSV 缓存')
        except (OSError, ValueError, sqlite3.Error, TimeoutError) as error:
            messages.append(f'清理未全部完成：{error}')
        u.cache_message = '。'.join(messages)
        st.rerun()
    st.button('查看构建与模型',on_click=nav,args=('about',))


def search_page():
    st.button('返回原页面', key='search_back', icon=':material/arrow_back:',
              on_click=nav, args=(st.session_state.get('_search_origin','events'),), help='Esc')
    if not valid_result:
        st.info('载入演示或 CSV 后，可以搜索当前批次的记录和事件。')
        first_batch()
        return
    query = st.text_input('搜索记录或事件', key='global_search_query',
                          placeholder='输入 CSV 行号、完整记录 ID 或 EVT-事件编号')
    value = query.strip()
    if not value:
        return
    if value.upper().startswith('EVT-') and 'ow_event' in pred:
        number = line_number(value[4:])
        if number is not None and pred.ow_event.eq(number).any():
            st.button(f'打开事件 EVT-{number:04d}',on_click=request,
                      args=('event_selection',(u.active.key, number, None)))
            return
    number = line_number(value)
    matches = pred[pred.record_id.eq(value) | u.physical_rows.eq(number)]
    if matches.empty:
        st.info('当前批次没有匹配的记录或事件。')
    else:
        for i,row in matches.iterrows():
            st.button(f'打开输入 CSV 第 {u.physical_rows.loc[i]} 行', on_click=request,
                      args=('review_record',row.record_id))


def audit_page():
    st.caption(f'{literal_markdown(u.active.name)} · 批次 {u.active.digest[:12]}')
    st.button('导出复核结论、分析明细或审计文件',type='primary',on_click=nav,args=('download',))
    if current_review is None:
        st.warning('当前复核存储不可核验。')
        return
    try:
        with review_connection() as db:
            token = dynamic.prepare_export(db, review_scope, review_loader)
        ops = [json.loads(line) for line in token.history]
        rows_by_rid = dict(zip(pred.record_id, u.physical_rows))
        event_by_batch = {}
        for op in ops:
            if op.get('kind')=='authorize':
                gate_data = json.loads(op['evidence']).get('gate',{})
                number = gate_data.get('search',{}).get('scope',{}).get('event_id')
                if number is not None:
                    event_by_batch[op['id']] = int(number)
        records = [{'时间UTC':v.get('created_at'), '操作':{'authorize':'事件授权','invalidate':'继承失效','withdraw':'撤回授权'}.get(v.get('kind'),v.get('status','逐条复核')),
                    '输入CSV行号':rows_by_rid.get(v.get('record_id')), '依据':v.get('note') or v.get('reason'),
                    '批次或记录ID':v.get('batch_id') or v.get('record_id') or v.get('id'),
                    '操作ID':str(v['id'])} for v in ops]
        if records:
            st.dataframe(pd.DataFrame(records).sort_values('时间UTC',ascending=False), hide_index=True,width='stretch')
            if event_by_batch:
                batch_id = st.selectbox('回查授权批次的来源事件',list(event_by_batch),
                    format_func=lambda k:f'EVT-{event_by_batch[k]:04d} · {k}')
                st.button('打开来源事件',on_click=request,
                          args=('event_selection',(u.active.key,event_by_batch[batch_id],None,'研判历史')))
            direct = [r for r in reversed(ops) if r.get('record_id') in rows_by_rid]
            if direct:
                rid = st.selectbox('回查逐条记录', list(dict.fromkeys(r['record_id'] for r in direct)),
                                   format_func=lambda rid:f'输入 CSV 第 {rows_by_rid[rid]} 行')
                st.button('打开记录与证据',on_click=request,args=('review_record',rid))
        else:
            st.info('本批次还没有逐条判断或事件授权记录。')
        downloads = local.download_history(DB, u.active)
        if downloads:
            st.subheader('下载交付记录')
            st.dataframe(pd.DataFrame(downloads),hide_index=True,width='stretch')
    except (sqlite3.Error,OSError,ValueError,ReviewConflictError) as error:
        st.warning(f'审计明细暂不可用：{error}')


def remember_position():
    if not valid_result or u.active is None or current_review is None:
        return
    if u.page not in ('overview','events','detection','review','audit'):
        return
    event = st.session_state.get('_event_target')
    event = list(event[1:]) if event and event[0]==u.active.key else None
    value = dict(filters=dict(u.filters),record=u.selected,page=u.page,page_index=u.page_index,
                 event=event, pending=len(filtered(pred,dict(default_filters(),scope='待办记录'),
                                                   bundle,physical_rows=u.physical_rows)))
    tag = (u.active.key,json.dumps(value,sort_keys=True))
    if st.session_state.get('_position_fingerprint') != tag:
        try:
            local.save_position(DB,u.active,value)
            st.session_state['_position_fingerprint'] = tag
        except (OSError,ValueError,sqlite3.Error) as error:
            st.caption(f'工作位置未保存：{error}')



review_scope = review_loader = current_review = None
review_evidence = {}
if valid_result and available:
    try:
        review_scope, review_loader, _ = review_analysis()
        with review_connection() as db:
            current_review = refresh_review_scope(db, review_scope, review_loader)
        pred = review_projection(raw_pred, current_review.records, reviews)
        if u.exports and not u.exports.meta['review_storage_available']:
            u.exports = None
    except (OSError, sqlite3.Error, ReviewConflictError, TypeError, KeyError, ValueError) as error:
        pred['review_status'] = '复核不可用'
        pred['review_note'] = ''
        st.warning(f'当前授权状态暂无法核验。复核结论及相关下载已暂停。{error}')


# Pending saves need current_evidence and the remaining page helpers to be
# defined before execution, including on a fresh Streamlit script rerun.
shell.header(nav, u.page)
with st.container(key='shell'):
    leftnav, main = st.columns([1,4], gap='small')
    with leftnav.container(key='worknav'):
        shell.links(nav,u.page)
    with main.container(key='workbody'):
        if u.page != 'events' or not valid_result:
            st.title(page_label(u.page))
        if u.flash:
            st.success(u.flash)
            u.flash = None
        if u.cache_message:
            if isinstance(u.cache_message, UserError):
                show_local_file_error(u.cache_message)
            else:
                st.info(u.cache_message)
            u.cache_message = None
        if model_error and u.page != 'about':
            st.error('分析程序尚未就绪。你仍可查看帮助，或在“偏好设置”查看构建与模型。')
        render_pending_action()
        if u.page == 'help':
            help_page()
        elif u.page == 'about':
            st.write('流哨 · 本机网络记录检查助手')
            st.caption(f'软件构建 {BUILD_LABEL}')
            st.write('检查流量、查看依据、保存复核记录。')
            for name, label in [('MODEL_CARD.md', '模型卡'), ('LOCAL_DATA.md', '本地文件与使用边界'), ('MIGRATION.md', '旧版复核迁移')]:
                document = ROOT / 'docs/usage' / name
                if document.is_file():
                    with st.expander(label):
                        st.markdown(document.read_text(encoding='utf-8'))
            if model_error:
                st.error('分析程序未就绪。')
                st.write(model_error)
            with st.expander('版本与技术信息'):
                st.write('当前模型', bundle['model_version'] if bundle else '暂不可用')
                if bundle:
                    st.json({key: bundle['config'][key] for key in ('rf_threshold', 'lr_threshold', 'near_margin', 'data_version')})
                    st.write(bundle['config']['features'])
                st.caption('RF：主要判断模型。LR：对照判断模型。阈值：模型使用的判断分界。')
        elif u.page == 'detection':
            detection_page()
        elif u.page == 'preferences':
            preferences_page()
        elif u.page == 'search':
            search_page()
        elif not valid_result:
            first_batch()
        elif u.page == 'events':
            event_page()
        elif u.page == 'overview':
            workspace_overview()
        elif u.page == 'audit':
            audit_page()
        elif u.page == 'evidence_detail':
            evidence_detail_page()
        elif u.page == 'review':
            review_page()
        elif u.page == 'quality':
            quality_page()
        elif u.page == 'download':
            download_page()
        else:
            from app_ui_tools import auxiliary, batch_page
            if u.page == 'auxiliary':
                auxiliary(u, pred, features, bundle, meta, csv_bytes)
            else:
                batch_page(u, pred, features, bundle, meta, csv_bytes)

remember_position()
shell.footer(BUILD_LABEL,bundle is not None,u.result.engine_status if valid_result else None)
shell.search_shortcuts(u.page == 'search')
from app_desktop_ready import render_desktop_ready
render_navigation_scroll(st.session_state, u.page)
render_desktop_ready(ROOT)
