"""Exercise the event and download routes against a shared SQLite store."""

import base64
from io import BytesIO
from pathlib import Path
import re
from types import SimpleNamespace

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

import app_engine_evidence
import app_model

import app_service
from liushao import recovery
from liushao.reviews import save_review
from liushao.search import ANOMALY, NORMAL


ROOT = Path(__file__).resolve().parents[1]
INPUT_SHA = 'a' * 64
MODEL_SHA = 'b' * 64
DESCRIPTOR_SHA = 'c' * 64
MODEL = 'synthetic-model'
ENGINE = dict(zip(app_engine_evidence.IDENTITY_COLUMNS, ('d' * 64, 'e' * 64, 'v1')))
ALPHA = .05


def payload(row, physical):
    return dict(schema_version=2, method_version='synthetic', record_id=row.record_id,
                model_version=MODEL, input_name='fixture.csv', input_sha256=INPUT_SHA,
                model_sha256=MODEL_SHA, descriptor_sha256=DESCRIPTOR_SHA, input_csv_row=physical,
                identity=dict(source_file=row.source_file, source_row=int(row.source_row),
                              source_sha256=row.source_sha256),
                scores=dict(rf=.1, lr=.1, rf_threshold=.5, lr_threshold=.5),
                cards=[], selection={}, decomposition={})


@pytest.fixture
def windows(tmp_path, monkeypatch):
    import app_evaluation
    from app_types import ActiveInput, AnalysisResult, EngineStatus, SessionState


    path = tmp_path / 'reviews.sqlite'
    monkeypatch.setenv('LIUSHAO_REVIEW_DB', str(path))
    verified = SimpleNamespace(identity=MODEL, content=b'', descriptor='{}',
                               model_sha256=MODEL_SHA, descriptor_sha256=DESCRIPTOR_SHA)
    bundle = dict(model_version=MODEL, config=dict(rf_threshold=.5, lr_threshold=.5,
                                                  near_margin=.1, data_version='test', features=['test']))
    engine = {'_artifact_identity': ENGINE, 'events': {'reps': 1}}
    engine_evidence = dict(status='verified', identity=ENGINE, open_world=dict(alpha=ALPHA),
                           ecod_decomposition=dict(top_fields=[dict(field='test', direction='right',
                                                                   input_missing=False)]))
    monkeypatch.setattr(app_model, 'verified_model', lambda *_: verified)
    monkeypatch.setattr(app_model, 'load_model_bundle', lambda *_: bundle)
    monkeypatch.setattr(app_service, 'load_trusted_engine', lambda *_: engine)
    monkeypatch.setattr(app_evaluation, 'render_evaluation', lambda *_: None)
    monkeypatch.setattr(app_engine_evidence, 'event_signature_fields', lambda *_: ('test',))
    monkeypatch.setattr(app_engine_evidence, 'batch_evidence_reference',
                        lambda pred, *_: pd.Series(None, index=pred.index, dtype=object))
    monkeypatch.setattr(app_engine_evidence, 'build_engine_evidence', lambda *_, **__: engine_evidence)

    def record_evidence(row, _features, _bundle, *, context, meta):
        return payload(row, meta['input_csv_row'])

    import app_evidence
    monkeypatch.setattr(app_evidence, 'build_record_evidence', record_evidence)

    def open_windows(*, reverse=False):
        frame = pd.DataFrame([
            dict(record_id=rid, model_version=MODEL, source_file='source.csv', source_row=index + 100,
                 source_sha256='f' * 64, rf_score=.1, lr_score=.1, rf_anomaly=False, lr_anomaly=False,
                 near_threshold=False, risk_alpha=ALPHA, ow_event=0, ow_queue='模糊', ow_state='模糊',
                 engine_rank=index + 1, cp_set='不确定', cp_set_size=2, **ENGINE)
            for index, rid in enumerate(('first', 'second'))
        ])
        _, reps = app_engine_evidence.event_members_and_representatives(frame, 0, 1)
        rep = reps[0]
        member = next(rid for rid in frame.record_id if rid != rep)
        if reverse:
            frame.loc[frame.record_id.eq(member), 'lr_anomaly'] = True
        index = frame.index[frame.record_id.eq(rep)][0]
        saved = payload(frame.loc[index], int(index) + 2)
        saved['engine'] = engine_evidence
        save_review(path, rep, MODEL, NORMAL, '代表已人工核查', evidence_snapshot=saved)

        def open_window(page):
            active = ActiveInput(b'', 'fixture.csv', '上传', INPUT_SHA, 'fixture', MODEL)
            features = pd.DataFrame({'test': [1., 1.]})
            source = features.copy()
            source.attrs['source_rows'] = [2, 3]
            state = SessionState(page=page, active=active, physical_rows=pd.Series([2, 3]),
                                 result=AnalysisResult(frame.copy(), features, pd.DataFrame(), 0, 2,
                                                       EngineStatus('ok', '')))
            state.filters.scope = '模糊队列'
            script = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=15)
            script.session_state['ui'] = state
            script.session_state['_raw_context_key'] = active.key
            script.session_state['_raw_context'] = source
            script.session_state['test_rep'] = rep
            script.run()
            assert not script.exception
            return script

        return SimpleNamespace(open=open_window, rep=rep, member=member)

    return open_windows


def click(app, label):
    next(button for button in app.button if button.label == label).click().run()
    assert not app.exception, [item.message for item in app.exception]
    assert not app.error, [item.value for item in app.error]


def approve(app):
    click(app, '生成批次预览')
    next(item for item in app.text_input if item.label == '确认依据').set_value('核查本次目标').run()
    next(item for item in app.checkbox if item.label == '我已核对当前结论与本次目标').check().run()
    click(app, '确认当前批次')


def iframe_download(app):
    matches = [re.search(r'const encoded = "([^"]+)"', iframe.proto.srcdoc) for iframe in app.get('iframe')]
    return [base64.b64decode(match[1]) for match in matches if match]


def test_event_preview_blocks_unreviewed_opposing_evidence(windows):
    sample = windows(reverse=True)
    event = sample.open('events')
    click(event, '批次确认与撤回')
    click(event, '生成批次预览')
    batch = event.session_state['_event_batch']['batch']
    assert batch.inheritance is None
    assert any(block.entry.record_id == sample.member and block.entry.evidence_source == 'LR'
               for block in batch.dynamic.gate.blockers)
    assert all(button.label != '确认当前批次' for button in event.button)
    download = sample.open('download')
    click(download, '生成下载文件')
    assert tuple(row.record_id for row in download.session_state['_review_export'].conclusions) == (sample.rep,)


def test_event_record_roundtrip_protects_draft_and_restores_source_section(windows):
    sample = windows(reverse=True)
    app = sample.open('events')
    click(app, '查看相关证据')
    next(button for button in app.button if str(button.key).endswith('_open_candidate')).click().run()
    assert not app.exception
    origin = app.session_state['_event_record_origin']
    assert origin[1] == 0 and origin[3] == '相关证据' and app.session_state.ui.selected == origin[-1]
    assert not any(button.key == 'save_review_next' for button in app.button)
    app.text_area(key='_note').input('软件回归样例，等待核对本次记录。').run()
    click(app, '返回事件 EVT-0000')
    assert app.session_state.ui.pending.kind == 'event_selection'
    click(app, '取消切换')
    assert app.text_area(key='_note').value == '软件回归样例，等待核对本次记录。'
    click(app, '保存并返回事件')
    assert app.session_state.ui.page == 'events'
    assert app.session_state['_event_target'] == origin[:3]
    assert app.session_state['_event_section'] == origin[3]
    assert app.session_state.ui.editor is None


@pytest.mark.parametrize('change', ['改判', '撤回'])
def test_second_window_invalidates_prepared_download(windows, change):
    sample = windows()
    editor = sample.open('events')
    click(editor, '批次确认与撤回')
    approve(editor)
    download = sample.open('download')
    click(download, '生成下载文件')
    initial = download.session_state['_review_export']
    assert initial.confirmed_count == 2
    assert {row.decision.state for row in initial.conclusions} == {
        recovery.RecoveryState.DIRECT, recovery.RecoveryState.INHERITED,
    }
    assert not iframe_download(download)
    click(download, '下载复核结论单')
    first, = iframe_download(download)
    historical = pd.read_csv(BytesIO(first)).set_index('记录ID')
    assert historical.loc[sample.member, '结论来源'] == '继承有效'
    assert historical.loc[sample.member, '我的判断'] == NORMAL

    editor = sample.open('events')
    click(editor, '批次确认与撤回')
    if change == '改判':
        click(editor, '属性')
        next(item for item in editor.selectbox if item.label == '当前事件成员').select(sample.rep).run()
        click(editor, '进入研判')
        assert editor.session_state.ui.selected == sample.rep
        next(item for item in editor.selectbox if item.label == '人工逐条判断').select(ANOMALY).run()
        next(item for item in editor.text_area if item.label == '人工依据（确认时必填）').input('逐条核查后改判').run()
        click(editor, '保存')
    else:
        click(editor, '按原批次预览撤回')
        next(item for item in editor.text_input if item.label == '撤回原因').set_value('复查撤回授权').run()
        next(item for item in editor.checkbox if item.label == '我已核对原批次的撤回影响').check().run()
        click(editor, '提交撤回')

    click(download, '下载复核结论单')
    assert not iframe_download(download)
    assert any('修订或范围已改变' in item.value for item in download.info)
    assert not any(button.label.startswith('下载复核结论单') for button in download.button)
    click(download, '更新导出文件')
    current = download.session_state['_review_export']
    assert current.confirmed_count == 1
    assert tuple(row.record_id for row in current.conclusions) == (sample.rep,)
    assert current.conclusions[0].decision.conclusion == (ANOMALY if change == '改判' else NORMAL)
    assert any('人工逐条确认 1 条、有效继承 0 条' in item.value for item in download.markdown)
    click(download, '下载分析明细')
    checked, = iframe_download(download)
    checks = pd.read_csv(BytesIO(checked)).set_index('record_id')
    assert checks.loc[sample.member, 'review_status'] == '待复核'
    assert historical.loc[sample.member, '我的判断'] == NORMAL
