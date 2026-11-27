"""expert_hint_summary: display-only aggregation of frozen hint outputs."""
import numpy as np
import pandas as pd

from app_score_charts import expert_hint_summary

CFG = {'rf_threshold': 0.12, 'lr_threshold': 0.84}


def test_summary_counts_and_cross():
    frame = pd.DataFrame({
        'rf_score': [0.5, 0.01, 0.9, 0.02],
        'lr_score': [0.9, 0.1, 0.2, 0.95],
        'rf_anomaly': [True, False, True, False],
        'lr_anomaly': [True, False, False, True],
        'ow_state': ['攻击', '正常', '新异', '攻击'],
    })
    s = expert_hint_summary(frame, CFG)
    assert s['rf'] == {'valid': 4, 'hints': 2}
    assert s['lr'] == {'valid': 4, 'hints': 2}
    assert s['cross'] == {'both_valid': 4, 'both_hint': 1, 'rf_only': 1,
                          'lr_only': 1, 'neither': 1}
    assert s['ow_state'] == {'攻击': 2, '正常': 1, '新异': 1}


def test_summary_fallback_threshold_and_nan():
    frame = pd.DataFrame({'rf_score': [0.5, np.nan, 0.01],
                          'lr_score': [0.9, 0.99, np.nan]})
    s = expert_hint_summary(frame, CFG)
    assert s['rf'] == {'valid': 2, 'hints': 1}
    assert s['lr'] == {'valid': 2, 'hints': 2}
    assert s['cross']['both_valid'] == 1
    assert s['ow_state'] == {}


def test_summary_missing_expert():
    frame = pd.DataFrame({'rf_score': [0.5, 0.01]})
    s = expert_hint_summary(frame, CFG)
    assert s['rf'] == {'valid': 2, 'hints': 1}
    assert s['lr'] is None and s['cross'] is None
