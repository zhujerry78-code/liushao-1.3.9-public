"""Export frozen synthetic evidence without model or experiment artifacts."""
import ast
from contextlib import closing
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from app_record_tools import confirmed_frame
from app_ui_state import export_snapshot
from liushao import reviews


def payload():
    return dict(schema_version=2, method_version='synthetic-export', record_id='r',
                model_version='m', input_sha256='a'*64, model_sha256='b'*64,
                descriptor_sha256='c'*64, cards=[], selection={}, decomposition={},
                input_name='冻结输入.csv', input_csv_row=7,
                identity=dict(source_file='冻结来源.csv', source_row=9),
                scores=dict(rf=.2, lr=.3, rf_threshold=.5, lr_threshold=.5))


def export(audit):
    pred = pd.DataFrame([dict(record_id='r', source_file='当前来源.csv', source_row=99)])
    return export_snapshot(dict(key='input', name='当前输入.csv'), pred, audit,
                           dict(model_version='m'), True, physical_rows=pd.Series([88]))


def test_csv_row_links_to_exact_frozen_bytes_and_saved_conclusion(tmp_path):
    path = tmp_path/'reviews.sqlite'
    reviews.save_review(path, 'r', 'm', '确认异常', '旧结论', evidence_snapshot=payload())
    reviews.save_review(path, 'r', 'm', '确认正常', '已确认依据，含中文\n第二行', evidence_snapshot=payload())
    audit = reviews.history(path)
    fixed = export(audit)
    row = pd.read_csv(BytesIO(fixed.confirmed), keep_default_na=False).iloc[0]
    with closing(sqlite3.connect(path)) as db:
        raw, digest = db.execute('SELECT payload_json,payload_sha256 FROM review_evidence WHERE review_id=?',
                                 (int(row['复核记录ID']),)).fetchone()
    assert row['证据快照JSON'] == raw
    assert hashlib.sha256(row['证据快照JSON'].encode('utf-8')).hexdigest() == digest == row['证据快照SHA256']
    history = json.loads(fixed.history)
    linked = next(r for r in history if r['id'] == int(row['复核记录ID']))
    assert row['记录ID'] == linked['record_id'] == 'r'
    assert row['模型版本'] == linked['model_version'] == 'm'
    assert row['我的判断'] == linked['status'] == '确认正常'
    assert row['备注'] == linked['note']
    assert row['复核保存时间UTC'] == linked['created_at']
    assert row['输入文件'] == '冻结输入.csv' and row['输入CSV行号'] == 7
    assert row['来源文件'] == '冻结来源.csv' and row['来源行号'] == 9
    assert json.loads(raw) == linked['evidence_snapshot']
    assert '未提示异常' == row['保存时系统提示']
    assert reviews.history(path) == audit


@pytest.mark.parametrize('damage', ['missing', 'hash', 'payload', 'schema', 'partial'])
def test_missing_invalid_and_partial_evidence_never_borrows_current_input(tmp_path, damage):
    path = tmp_path/'reviews.sqlite'
    frozen = payload()
    if damage == 'partial':
        for field in ('input_name', 'input_csv_row', 'identity'):
            frozen.pop(field)
    reviews.save_review(path, 'r', 'm', '确认异常', '仍保留结论',
                        evidence_snapshot=None if damage == 'missing' else frozen)
    with closing(sqlite3.connect(path)) as db, db:
        if damage == 'hash':
            db.execute("UPDATE review_evidence SET payload_sha256=?", ('d'*64,))
        elif damage == 'payload':
            db.execute("UPDATE review_evidence SET payload_json='{}'")
        elif damage == 'schema':
            db.execute('UPDATE review_evidence SET schema_version=99')
    before = path.read_bytes()
    row = pd.read_csv(BytesIO(export(reviews.history(path)).confirmed), keep_default_na=False).iloc[0]
    assert row['我的判断'] == '确认异常' and row['备注'] == '仍保留结论'
    for field in ('输入文件', '输入CSV行号', '来源文件', '来源行号'):
        assert row[field] == '未记录'
    if damage == 'partial':
        assert row['证据快照状态'] == '已保存当时证据'
    else:
        assert row['证据快照状态'] == ('无当时快照' if damage == 'missing' else '当时快照校验失败')
        assert row['证据快照JSON'] == row['保存时输入SHA256'] == '未记录'
        assert 'RF 分数' not in row['保存时证据摘要']
    assert path.read_bytes() == before


def app_function(name, namespace):
    tree = ast.parse((Path(__file__).parents[1]/'app.py').read_text(encoding='utf-8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'app.py', 'exec'), namespace)
    return namespace[name]


def test_save_failure_retains_editor_and_blocks_download():
    editor = SimpleNamespace(input_key='input', record_id='r', baseline=0,
                             status='确认正常', note='未保存依据', conflict=False)
    state = SimpleNamespace(editor=editor, exports=None, active=SimpleNamespace(key='input'))
    st = Mock()
    ns = dict(u=state, st=st, valid_result=True, available=True, DB='unused',
              bundle=dict(model_version='m'), save_review=Mock(side_effect=OSError('disk full')),
              ReviewConflictError=reviews.ReviewConflictError,
              current_evidence=lambda rid: payload(), require_complete_engine_evidence=lambda p: p)
    assert app_function('save_editor', ns)() is False
    assert state.editor is editor and state.exports is None
    assert '保存失败' in st.error.call_args.args[0]
    ns.update(pred=pd.DataFrame([dict(record_id='r')]), dirty=lambda e: True,
              DOWNLOAD_DESCRIPTIONS={}, nav=Mock())
    app_function('download_page', ns)()
    assert '保存尚未成功确认' in st.warning.call_args.args[0]
    assert st.button.call_args.kwargs['disabled'] is True
    assert state.exports is None


def test_export_read_failure_is_visible_and_publishes_no_files():
    state = SimpleNamespace(editor=None, exports=None, pending=None)
    st = Mock()
    st.button.return_value = True
    ns = dict(u=state, st=st, pred=pd.DataFrame([dict(record_id='r')]),
              dirty=lambda e: False, available=True, audit=[], DOWNLOAD_DESCRIPTIONS={},
              DB='unused', bundle=dict(model_version='m'), sqlite3=sqlite3,
              history=Mock(side_effect=sqlite3.OperationalError('read failed')))
    app_function('download_page', ns)()
    assert '导出失败' in st.error.call_args.args[0]
    assert state.exports is None


def test_incomplete_review_timestamp_is_marked_not_invented():
    audit = [dict(id=1, record_id='r', model_version='m', status='确认正常', note=None)]
    row = pd.read_csv(BytesIO(export(audit).confirmed), keep_default_na=False).iloc[0]
    assert row['复核保存时间UTC'] == row['备注'] == '未记录'
