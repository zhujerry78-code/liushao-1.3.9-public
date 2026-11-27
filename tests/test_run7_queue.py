"""Regression coverage for the run7 candidate's queue and interactive bins."""
import numpy as np
import pandas as pd
import pytest

from app_evidence import neighbor_summary
from app_quick_search import matching_records
from app_review_guidance import human_basis, review_focus_rows
from app_score_charts import BIN_EDGES, histogram_rows, in_score_bin
from app_types import Filters
from app_ui_state import conclusion_source, filtered


CONFIG = {'config': {'lr_threshold': .5, 'rf_threshold': .5, 'near_margin': .05}}


def frame():
    return pd.DataFrame([
        dict(record_id=f'id-{i}', source_row=100+i, source_file='fixture.csv',
             rf_score=rf, lr_score=lr, rf_anomaly=rf>=.5, lr_anomaly=lr>=.5,
             near_threshold=abs(rf-.5) <= .05, ow_queue=queue, engine_rank=i,
             review_status=status, authorization_state=state)
        for i, (rf, lr, queue, status, state) in enumerate([
            (.1, .1, '', '待复核', '待复核或待补证'),
            (.9, .1, '新异', '待复核', '待复核或待补证'),
            (.1, .1, '', '待补证', '待复核或待补证'),
            (.9, .8, '', '确认异常', '继承有效'),
            (.88, .8, '', '确认异常', '直接确认'),
            (.3, .2, '', '待复核', '继承失效待重新确认')])])


def test_exact_statuses_and_input_source_row_identity():
    pred = frame()
    physical = pd.Series([2, 4, 7, 10, 15, 16])
    pred['input_csv_row'] = physical
    assert list(pred[matching_records(pred, '提示异常')].index) == [1, 3, 4]
    assert list(pred[matching_records(pred, '未提示异常')].index) == [0, 2, 5]
    for query in ('第 4 行', '来源第101行', '输入CSV第4行'):
        assert list(pred[matching_records(pred, query)].index) == [1]
    assert filtered(pred, Filters(scope='全部记录', keyword='零结果'), CONFIG).empty
    assert filtered(pred, Filters(scope='全部记录', line='7'), CONFIG, physical_rows=physical).index.tolist() == [2]


@pytest.mark.parametrize('sort', ['高异常得分', '待核实', '原始顺序'])
def test_sort_preserves_scope_and_pending_includes_rechecks(sort):
    pred = frame()
    full = filtered(pred, Filters(scope='全部记录', queue=sort), CONFIG)
    assert set(full.record_id) == set(pred.record_id)
    pending = filtered(pred, Filters(queue=sort), CONFIG)
    assert set(pending.index) == {1, 2, 5}
    assert [conclusion_source(pred.iloc[i]) for i in [3,4,5]] == ['由事件继承','人工逐条确认','继承失效']


def test_clickable_bins_partition_boundaries_and_include_one():
    scores = pd.Series(sorted([0.,1.,.15] + list(BIN_EDGES) +
                              [np.nextafter(x, 0.) for x in BIN_EDGES[1:]]))
    rows = histogram_rows(scores)
    members = [set(scores[in_score_bin(scores, i)].index) for i in range(20)]
    assert list(map(len, members)) == [item['count'] for item in rows]
    assert sum(map(len, members)) == len(scores) == len(set.union(*members))
    assert scores[in_score_bin(scores, 19)].max() == 1.
    assert 0.15 in scores[in_score_bin(scores, 3)].values


def test_legacy_system_draft_requires_independent_observation():
    prefix = '【评分线索】\n自动线索\n\n【复核结论】\n'
    assert not human_basis(prefix)
    assert human_basis(prefix + '核对服务器日志发现超时') == '核对服务器日志发现超时'
    assert human_basis('已核对资产工单\n' + prefix) == '已核对资产工单'


def test_neighbor_links_derive_from_same_batch_context():
    context = pd.DataFrame({'Timestamp': ['2026-01-01T10:00:00','2026-01-01T10:01:00','2026-01-01T10:02:00']})
    context.attrs['record_ids'] = {0:'first', 1:'second', 2:'third'}
    result = neighbor_summary(context, 0)
    assert result['related_records'] == {'前后各 60 秒':['second']}
    assert result['time_window']['all'] == 1


def test_first_check_matches_input_value_and_score_after_presentational_reordering():
    reference = dict(p01=1, p50=3, p99=5)
    cards = [
        dict(field='Flow IAT Mean', name='平均间隔', unit='微秒', value=3, reference=reference, rf=dict(delta=.01)),
        dict(field='Flow IAT Max', name='最大间隔', unit='微秒', value=30, reference=reference, rf=dict(delta=-.08)),
        dict(field='Flow IAT Min', name='最小间隔', unit='微秒', missing=True, reference=reference, rf=dict(delta=.12)),
    ]
    rows, context = review_focus_rows(dict(cards=cards))
    assert [row['数据项'] for row in rows] == ['最小间隔', '最大间隔', '平均间隔']
    assert rows[0]['当前值'] == '原始缺失' and '采集日志' in rows[0]['具体核查']
    assert rows[1]['当前值'] == '30 微秒' and rows[1]['_contribution']['rf_replace_delta'] == -.08
    assert [row['数据项'] for row in context] == ['本批邻近记录', '同端口连接数、服务状态和授权任务']
    assert review_focus_rows(dict(selection=dict(status='unavailable'))) == ([], [])


def test_duration_comparison_uses_same_scale_for_observation_bound_and_difference():
    card = dict(field='Flow IAT Min', name='最小数据包间隔', value=87_800_000, unit='微秒 (μs)',
                reference=dict(p01=0, p50=7, p99=86_100_000))
    rows, _ = review_focus_rows(dict(cards=[card]))
    assert rows[0]['直观对照'] == '87.8 秒，比训练参考上界 86.1 秒长 1.7 秒'
    assert rows[0]['当前值'] == '87,800,000 微秒 (μs)'
    card['value'] = 3
    assert review_focus_rows(dict(cards=[card]))[0][0]['直观对照'] == ''
