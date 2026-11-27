"""Literal text and exact status/line matching in a single batch."""
import re
import unicodedata
import pandas as pd
from app_ui_state import STATUS_NAMES


def normalized(value):
    return ' '.join(unicodedata.normalize('NFKC',str(value)).casefold().split())


def promote_matches(table, keyword):
    """Compatibility helper for callers that explicitly request match promotion."""
    mask = matching_records(table, keyword)
    if not normalized(keyword):
        return table, mask
    return pd.concat([table.loc[mask], table.loc[~mask]]), mask


def matching_records(table, keyword):
    query=normalized(keyword)
    mask=pd.Series(False,index=table.index)
    if not query:
        return ~mask
    if query in ('提示异常', '系统提示异常', '异常提示'):
        return table.rf_anomaly.eq(True)
    if query == '未提示异常':
        return table.rf_anomaly.eq(False)
    for state, label in STATUS_NAMES.items():
        if query in (normalized(state), normalized(label)):
            return table.review_status.eq(state)
    match=re.fullmatch(r'(?:(来源|输入(?:\s*csv)?|csv)\s*)?(?:第\s*)?(\d{1,12})(?:\s*行)?',query)
    if match:
        column = 'source_row' if match.group(1) == '来源' else 'input_csv_row'
        return table[column].eq(int(match.group(2))) if column in table else mask
    for column in ('source_file', 'record_id', 'ow_state', 'ow_queue', 'authorization_state'):
        if column in table:
            mask |= table[column].fillna('').map(normalized).str.contains(query, regex=False)
    return mask
