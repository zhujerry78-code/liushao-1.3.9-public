"""Workspace routes and callbacks accepted from earlier releases."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Page:
    id: str
    label: str
    icon: str
    group: str = 'main'

    @property
    def key(self):
        return 'nav_' + self.id


PAGES = (
    Page('overview', '概览', 'dashboard'),
    Page('detection', '流量检测', 'analytics'),
    Page('events', '事件研判', 'inventory_2'),
    Page('review', '复核队列', 'inbox'),
    Page('audit', '审计记录', 'history'),
    Page('preferences', '偏好设置', 'settings', 'system'),
)
BY_ID = {p.id: p for p in (*PAGES,
    Page('search', '搜索', 'search'), Page('quality', '文件问题', 'description'),
    Page('download', '导出文件', 'download'), Page('auxiliary', '辅助检查', 'manage_search'),
    Page('batch', '整批数据分析', 'bar_chart'), Page('help', '操作说明', 'help'),
    Page('about', '构建与模型', 'info'), Page('evidence_detail', '评分依据', 'insights'))}
LEGACY = {**{p.label: p.id for p in BY_ID.values()}, '查看结果':'overview', '检查记录':'review',
          '下载结果':'download', '模型与评测':'help', 'evaluation':'help', '关于流哨':'about'}


def page_id(value):
    if not isinstance(value, str):
        return 'events'
    return value if value in BY_ID else LEGACY.get(value, 'events')


def page_label(value):
    return BY_ID[page_id(value)].label

