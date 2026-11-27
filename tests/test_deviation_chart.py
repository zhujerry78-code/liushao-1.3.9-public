"""Deviation charts must render valid Vega-Lite per field (mixed units, negatives)."""
import app_evidence_ui as ui


class _Stub:
    def __init__(self):
        self.charts = []

    def markdown(self, *a, **k):
        pass

    def caption(self, *a, **k):
        pass

    def altair_chart(self, chart, **k):
        self.charts.append(chart)


def test_deviation_charts_per_field_scales(monkeypatch):
    stub = _Stub()
    monkeypatch.setattr(ui, "st", stub)
    fields = [
        {"name": "最小数据包间隔", "value": 87.8,
         "reference": {"p01": 0.0, "p99": 86.1}, "unit": "秒"},
        {"name": "最大数据包长", "value": 4_800_000.0,
         "reference": {"p01": 0.0, "p99": 3_900_000.0}, "unit": "字节"},
        {"name": "初始窗口字节", "value": -1.0,
         "reference": {"p01": 0.0, "p99": 65535.0}, "unit": "字节"},
    ]
    ui._render_deviation_chart(fields)
    assert len(stub.charts) == 3
    domains = []
    for chart in stub.charts:
        spec = chart.to_dict()  # full Vega-Lite validation, as Streamlit render does
        band = spec["layer"][0]
        domains.append(band["encoding"]["x"]["scale"]["domain"])
    uppers = sorted(d[1] for d in domains)
    assert uppers[-1] > 4_800_000.0  # byte-scale field keeps its own scale
    assert any(d[0] == 0.0 and 86.1 < d[1] < 100.0 for d in domains)  # seconds field
    assert any(d[0] < -1.0 for d in domains)  # negative value stays inside its own scale
