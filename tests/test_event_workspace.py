import pandas as pd
from streamlit.testing.v1 import AppTest

from liushao.event_workspace import event_roster, matching_events


def _rows():
    return pd.DataFrame([
        (3, '模糊', 4, 'a', '待复核或待补证', '待补证'),
        (3, '模糊', 1, 'b', '直接确认', '确认异常'),
        (4, '新异', 2, 'c', '继承有效', '确认正常'),
        (5, '新异', 3, 'd', '待复核或待补证', '待复核'),
        (None, '正常', 5, 'q', '直接确认', '确认正常'),
        (6, '新异', 7, 'e', '继承失效待重新确认', '待复核'),
    ], columns=['ow_event', 'ow_state', 'engine_rank', 'record_id', 'authorization_state', 'review_status'])


def test_roster_has_real_groups_unique_members_and_stable_ranking():
    events = event_roster(_rows(), {})
    assert [event.group for event in events] == ['待复核', '已确认', '待处理', '待复核']
    assert events[0].members == ('b', 'a')
    assert events[0].counts == {'人工逐条确认': 1, '待补证': 1}
    assert events[1].counts == {'由事件继承': 1}
    assert events[-1].counts == {'继承失效': 1}
    assert matching_events(events, '已确认', 'evt-0004') == [events[1]]
    assert matching_events(events, '待处理', '0004') == []
    assert event_roster(_rows(), {}, started={5})[2].group == '待复核'
    assert all(e.group == '状态待核验' for e in event_roster(_rows().drop(columns='authorization_state'), {}))


def _fixture():
    import streamlit as st
    from liushao.event_workspace import event_roster, render_page

    frame = st.session_state['frame']
    events = event_roster(frame, {})
    def choose(event, rid, section):
        chosen = next(e for e in events if e.number == event)
        st.session_state['target'] = ('file', event, rid or chosen.members[0])
        st.session_state['section'] = section
        st.session_state['_ev_mobile_detail'] = True
        st.session_state['epoch'] = st.session_state.get('epoch', 0) + 1

    render_page(events, active_key='file', target=st.session_state.get('target'),
                section=st.session_state.get('section', '\u5c5e\u6027'), section_callback=choose,
                record_callback=lambda rid: st.session_state.update(opened=rid), open_queue=lambda: None,
                pred=frame, physical=[2, 3, 4, 5, 6, 7],
                config={'rf_threshold': .65, 'lr_threshold': .55, 'near_margin': .05},
                locked=False, epoch=st.session_state.get('epoch', 0), detail_callback=lambda *_: None)


def test_page_opens_member_score_only_after_selection_and_filters_hide_old_detail():
    app = AppTest.from_function(_fixture)
    app.session_state['frame'] = _rows().assign(rf_score=.71, lr_score=.15)
    app.run()
    assert not app.exception
    assert not app.get('arrow_vega_lite_chart')
    next(button for button in app.button if 'EVT-0003' in button.label).click().run()
    assert not app.exception
    assert app.session_state['target'] == ('file', 3, 'b')
    assert any('RF 0.7100' in cap.value and '第 3 行' in cap.value for cap in app.caption)
    next(box for box in app.selectbox if box.label == '当前事件成员').select('a').run()
    assert app.session_state['target'] == ('file', 3, 'a')
    next(button for button in app.button if button.label == '查看相关证据').click().run()
    assert app.session_state['section'] == '相关证据'
    assert not any(button.label == '进入研判' for button in app.button)
    assert not app.get('arrow_vega_lite_chart')
    next(button for button in app.button if button.label == '批次确认与撤回').click().run()
    assert not any(box.label == '当前事件成员' for box in app.selectbox)
    assert not any(button.label == '进入研判' for button in app.button)
    next(button for button in app.button if button.label == '属性').click().run()
    assert next(box for box in app.selectbox if box.label == '当前事件成员').value == 'a'
    assert any('第 2 行' in cap.value and 'RF 0.7100' in cap.value for cap in app.caption)
    next(button for button in app.button if button.label == '进入研判').click().run()
    assert app.session_state['opened'] == 'a'
    next(box for box in app.text_input if box.label == '搜索事件').input('不存在').run()
    assert not any(button.label == '进入研判' for button in app.button)
    assert any('没有符合条件' in entry.value for entry in app.info)
