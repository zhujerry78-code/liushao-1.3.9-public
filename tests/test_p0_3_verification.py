"""Run independent P0-3 probes against the checkout under test."""

import ast
import csv
import hashlib
import io
import json
import os
from contextlib import closing, contextmanager
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest


if __name__ == "__main__":
    import pandas as pd

    from app_engine_evidence import (
        IDENTITY_COLUMNS,
        engine_identity,
        event_members_and_representatives,
        require_complete_engine_evidence,
    )
    from app_record_tools import confirmed_frame
    from app_types import ActiveInput, Draft, EngineStatus
    from app_ui_state import csv_bytes, export_snapshot
    from liushao import dynamic, gate, recovery, reviews, search, workbench
    from liushao.search import ANOMALY, NORMAL, AnalysisContext, SearchScope, search_counterexamples


MODEL = "p0-3-fixture"
INPUT = "a" * 64
MODEL_SHA = "b" * 64
DESCRIPTOR = "c" * 64
ALPHA = 0.05
BATCH = "p0-3-authorization"
WITHDRAWAL = "p0-3-withdrawal"


def read_csv(data, key):
    return {row[key]: row for row in csv.DictReader(io.StringIO(data.decode("utf-8-sig")))}


def canonical(snapshot):
    raw = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return raw, hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Scenario:
    def __init__(self):
        self.db_path = Path.cwd() / "reviews.sqlite"
        engine_id = dict(zip(IDENTITY_COLUMNS, ("d" * 64, "e" * 64, "engine-v1")))
        self.frame = pd.DataFrame([
            dict(record_id=rid, model_version=MODEL, source_file="live-source.csv", source_row=100 + i,
                 source_sha256="f" * 64, rf_score=.1, lr_score=.1, rf_anomaly=False, lr_anomaly=False,
                 near_threshold=False, risk_alpha=ALPHA, ow_event=event, ow_queue="模糊", ow_state="模糊",
                 engine_rank=i + 1, cp_set="不确定", cp_set_size=2, **engine_id)
            for i, (rid, event) in enumerate((("first", 0), ("second", 0), ("prediction", 1)))
        ])
        self.physical = pd.Series([2, 3, 4])
        members, reps = event_members_and_representatives(self.frame, 0, 1)
        self.rep, = reps
        self.member, = set(members) - set(reps)
        self.engine = dict(status="verified", identity=engine_id, open_world=dict(alpha=ALPHA),
                           ecod_decomposition=dict(top_fields=[
                               dict(field="test", direction="right", input_missing=False)]))
        self.ui = SimpleNamespace(active=ActiveInput(b"", "live.csv", "上传", INPUT, "fixture", MODEL),
                                  physical_rows=self.physical, result=SimpleNamespace(engine_status=EngineStatus("ok", "")),
                                  editor=None, exports=None, selected=None, flash=None)
        # These functions are defined in the Streamlit entry point. Execute the
        # definitions with synthetic inputs, bypassing page startup.
        app_path = Path(sys.argv[2]) / "app.py"
        names = {"review_connection", "review_analysis", "refresh_review_scope",
                 "review_projection", "save_editor", "prepare_review_exports"}
        tree = ast.parse(app_path.read_text(encoding="utf-8"), filename=str(app_path))
        definitions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
        assert {node.name for node in definitions} == names
        self.app = dict(globals(), u=self.ui, DB=self.db_path, raw_pred=self.frame,
                        meta=dict(model_version=MODEL, input_sha256=INPUT, model_sha256=MODEL_SHA,
                                  descriptor_sha256=DESCRIPTOR),
                        bundle=dict(model_version=MODEL), features=pd.DataFrame(), field_name=None,
                        valid_result=True, available=True, connect=reviews.connect,
                        save_review=reviews.save_review, ReviewConflictError=reviews.ReviewConflictError,
                        verified_display_engine=lambda: {"events": {"reps": 1}},
                        current_evidence=self.evidence, review_evidence={})
        exec(compile(ast.Module(body=definitions, type_ignores=[]), str(app_path), "exec"), self.app)
        scope, load, _ = self.app["review_analysis"]()
        self.app.update(review_scope=scope, review_loader=load)

    def evidence(self, rid):
        index, = self.frame.index[self.frame.record_id.eq(rid)]
        row = self.frame.loc[index]
        return dict(schema_version=2, record_id=rid, model_version=MODEL, method_version="fixture-v2",
                    input_name="saved.csv", input_sha256=INPUT, model_sha256=MODEL_SHA,
                    descriptor_sha256=DESCRIPTOR, input_csv_row=int(self.physical.at[index]),
                    identity=dict(source_file=row.source_file, source_sha256=row.source_sha256,
                                  source_row=int(row.source_row)),
                    scores=dict(rf=.1, lr=.1, rf_threshold=.5, lr_threshold=.5),
                    cards=[], selection={}, decomposition={}, engine=self.engine)

    def connect(self):
        return self.app["review_connection"]()

    def save(self, status, note, baseline=0):
        self.ui.editor = Draft(input_key=self.ui.active.key, record_id=self.rep, status=status,
                               note=note, baseline=baseline, original=("待复核", ""), conflict=False)
        assert self.app["save_editor"]()

    def authorize(self):
        self.save(NORMAL, "代表直接核查")
        scope, load, load_gate_state = self.app["review_analysis"](event_id=0)
        with self.connect() as db:
            event = self.app["refresh_review_scope"](db, scope, load)
            batch = workbench.prepare_batch(db, event.preview, load=load)
            assert batch.dynamic.gate.passed, batch.dynamic.gate.blockers
            confirmation = gate.GateConfirmation(BATCH, NORMAL, batch.dynamic.gate.targets,
                                                 "授权已核查事件范围", True)
            workbench.submit_batch(db, batch, confirmation, load=load, load_gate_state=load_gate_state)

    def export(self):
        return self.app["prepare_review_exports"]()

    def current(self, db):
        return {row.record_id: row for row in dynamic.read_current(
            db, self.app["review_scope"], self.app["review_loader"]).records}

    def withdraw(self):
        with self.connect() as db:
            preview = recovery.preview_withdrawal(db, BATCH)
            return recovery.withdraw(db, preview, operation_id=WITHDRAWAL, reason="复查后撤回", confirmed=True)


def conclusion_sources():
    sample = Scenario()
    predicted, token = sample.export()
    assert token.confirmed_count == 0 and predicted.confirmed is None
    assert json.loads(predicted.history) == reviews.history(sample.db_path) == []
    assert {row["review_status"] for row in read_csv(predicted.predictions, "record_id").values()} == {"待复核"}
    with sample.connect() as db:
        assert db.execute("SELECT COUNT(*) FROM recovery.inheritances").fetchone()[0] == 0
        assert db.execute("SELECT COUNT(*) FROM recovery.operations WHERE kind='authorize'").fetchone()[0] == 0

    sample.authorize()
    exported, token = sample.export()
    direct, = reviews.history(sample.db_path)
    assert direct["record_id"] == sample.rep and direct["evidence_snapshot_status"] == "saved"
    with sample.connect() as db:
        inherited, = recovery.inheritance_history(db, BATCH)
        rep = recovery.current_decision(db, MODEL, sample.rep)
        member = recovery.current_decision(db, MODEL, sample.member)
        projected = sample.current(db)
    assert rep.state == recovery.RecoveryState.DIRECT and rep.review_id == direct["id"]
    assert member.state == recovery.RecoveryState.INHERITED and member.conclusion == NORMAL
    assert member.inheritance_id == inherited["id"]
    assert member.batch_id == inherited["batch_id"] == BATCH
    assert projected[sample.member].decision == member and projected[sample.member].inheritance_source == BATCH
    assert projected["prediction"].decision.state == recovery.RecoveryState.PENDING
    checks = read_csv(exported.predictions, "record_id")
    conclusions = read_csv(exported.confirmed, "记录ID")
    assert token.confirmed_count == len(conclusions) == 2
    assert checks["prediction"]["review_status"] == "待复核"
    assert checks[sample.rep]["authorization_state"] == conclusions[sample.rep]["结论来源"] == "直接确认"
    assert checks[sample.member]["authorization_state"] == conclusions[sample.member]["结论来源"] == "继承有效"
    assert checks[sample.member]["authorization_batch"] == conclusions[sample.member]["原授权批次ID"] == BATCH
    assert checks[sample.rep]["authorization_batch"] == conclusions[sample.rep]["原授权批次ID"] == ""
    authorization = next(row for row in json.loads(exported.history) if row.get("kind") == "authorize")
    candidates = json.loads(authorization["evidence"])["gate"]["search"]["ordered_candidates"]
    original_row = next(item["physical_row"] for item in candidates if item["record_id"] == sample.member)
    assert original_row == 2
    assert projected[sample.member].physical_row == original_row
    assert conclusions[sample.member]["输入CSV行号"] == str(original_row)
    return dict(ai_only_review_rows=0, review_rows=1, ledger_members=1, conclusions=2,
                batch=BATCH, recovery_batch=member.batch_id,
                frozen_member_row=original_row, conclusion_member_row=conclusions[sample.member]["输入CSV行号"])


def inactive_inheritance():
    sample = Scenario()
    sample.authorize()
    _, frozen = sample.export()
    with sample.connect() as db:
        baseline = recovery.current_decision(db, MODEL, sample.rep).review_id
    sample.save(ANOMALY, "代表复查后改判", baseline)
    # Read from a fresh connection before any refresh can invalidate the batch.
    with sample.connect() as db:
        immediate = recovery.current_decision(db, MODEL, sample.member)
        before_refresh = sample.current(db)[sample.member]
        operations_before_refresh = db.execute(
            "SELECT COUNT(*) FROM recovery.operations WHERE kind='invalidate' AND batch_id=?", (BATCH,)
        ).fetchone()[0]
        with pytest.raises(dynamic.RefreshRequired):
            dynamic.download_export(db, frozen, sample.app["review_loader"])
    assert operations_before_refresh == 0
    assert immediate.state == recovery.RecoveryState.INVALIDATED and immediate.conclusion is None
    assert immediate.batch_id == BATCH
    assert before_refresh.decision.state == recovery.RecoveryState.PENDING
    assert before_refresh.decision.conclusion is None
    invalid, _ = sample.export()
    with sample.connect() as db:
        invalidated = recovery.current_decision(db, MODEL, sample.member)
        assert invalidated.state == recovery.RecoveryState.INVALIDATED and invalidated.conclusion is None
    audit = sample.withdraw()
    withdrawn, _ = sample.export()
    with sample.connect() as db:
        pending = recovery.current_decision(db, MODEL, sample.member)
        assert pending.state == recovery.RecoveryState.PENDING and pending.conclusion is None
        row, = recovery.inheritance_history(db, BATCH)
        assert row["state"] == recovery.RecoveryState.WITHDRAWN
    assert audit["batch_id"] == BATCH
    for snapshot in (invalid, withdrawn):
        assert set(read_csv(snapshot.confirmed, "记录ID")) == {sample.rep}
        member = read_csv(snapshot.predictions, "record_id")[sample.member]
        assert member["review_status"] == "待复核" and member["authorization_batch"] == ""
    return dict(immediate_recovery=immediate.state, immediate_conclusion=immediate.conclusion,
                immediate_dynamic=before_refresh.decision.state, invalidations_before_refresh=operations_before_refresh,
                stale_download="rejected",
                after_refresh=invalidated.state, after_withdrawal=pending.state, history_state=row["state"],
                conclusions_after_each=[invalid.confirmed_count, withdrawn.confirmed_count])


def audit_history():
    sample = Scenario()
    sample.authorize()
    first, = reviews.history(sample.db_path)
    sample.save(ANOMALY, "代表复查后改判", first["id"])
    sample.export()
    sample.withdraw()
    exported, token = sample.export()
    entries = json.loads(exported.history)
    assert entries == [json.loads(row) for row in token.history]
    direct = [row for row in entries if "record_id" in row]
    operations = [row for row in entries if "kind" in row]
    assert len(entries) == 5 and len(direct) == 2
    author, invalidation, withdrawal = operations
    assert [row["kind"] for row in operations] == ["authorize", "invalidate", "withdraw"]
    with sample.connect() as db:
        for row in operations:
            saved = recovery.operation_history(db, row["id"])
            assert row["reason"] == saved["reason"] and row["reason"].strip()
            assert row["created_at"] and json.loads(row["evidence"]) == saved["evidence"]
        grant, loss, undo = [json.loads(row["evidence"]) for row in operations]
        inherited, = recovery.inheritance_history(db, BATCH)
    assert author["id"] == inherited["batch_id"] == BATCH
    assert grant["gate"]["targets"] == [sample.member]
    for row in direct:
        assert row["id"] and row["created_at"] and row["note"].strip() and row["model_version"] == MODEL
        evidence = row["evidence_snapshot"]
        assert row["record_id"] == evidence["record_id"] == sample.rep
        assert row["evidence_snapshot_status"] == "saved"
        assert row["evidence_sha256"] == canonical(evidence)[1]
    assert grant["reviews"][sample.rep]["id"] == direct[0]["id"]
    assert loss["gate"]["counterexamples"][0]["review_id"] == direct[1]["id"]
    for operation, evidence in ((invalidation, loss), (withdrawal, undo)):
        assert operation["batch_id"] == evidence["original"]["operation_id"] == BATCH
        assert evidence["original"]["evidence_sha256"] == hashlib.sha256(author["evidence"].encode()).hexdigest()
    assert loss["affected"][inherited["id"]]["record_id"] == sample.member
    assert undo["impact"]["members"][0]["inheritance_id"] == inherited["id"]
    assert undo["validity"][0]["operation_id"] == invalidation["id"]
    return dict(history_entries=5, direct_ids=[row["id"] for row in direct],
                recovery_kinds=[row["kind"] for row in operations],
                linked_batch=BATCH, hashes_verified=True)


def strict_snapshots():
    sample = Scenario()
    path = sample.db_path
    snapshot = sample.evidence(sample.rep)
    for version in (0, 3, True, 1.0, "2", None):
        with pytest.raises(ValueError):
            reviews.save_review(path, sample.rep, MODEL, NORMAL, "核查", evidence_snapshot=dict(
                snapshot, schema_version=version))
    assert not path.exists()
    missing_scores = {key: value for key, value in snapshot.items() if key != "scores"}
    with pytest.raises(ValueError):
        reviews.save_review(path, sample.rep, MODEL, NORMAL, "核查", evidence_snapshot=missing_scores)
    assert not path.exists()
    for version in (1, 2):
        reviews.save_review(path, sample.rep, MODEL, NORMAL, "核查", evidence_snapshot=dict(
            snapshot, schema_version=version))
    assert [(row["evidence_snapshot"]["schema_version"], row["evidence_snapshot_status"])
            for row in reviews.history(path)] == [(1, "saved"), (2, "saved")]

    cases = [
        ("tampered", 2, dict(snapshot, input_name="changed.csv"), canonical(snapshot)[1]),
        ("unknown_version", 3, dict(snapshot, schema_version=3), None),
        ("missing_scores", 2, missing_scores, None),
        ("version_mismatch", 1, snapshot, None),
    ]
    results = {}
    for label, version, payload, original_digest in cases:
        raw, digest = canonical(payload)
        with closing(sqlite3.connect(path)) as db, db:
            db.execute("UPDATE review_evidence SET schema_version=?, payload_json=?, payload_sha256=? WHERE review_id=2",
                       (version, raw, original_digest or digest))
        row = reviews.evidence_for_review(path, 2)
        assert row["evidence_snapshot"] is None and row["evidence_snapshot_status"] == "invalid"
        results[label] = row["evidence_snapshot_status"]
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("DELETE FROM review_evidence WHERE review_id=2")
    exported, token = sample.export()
    row = json.loads(exported.history)[-1]
    assert row["evidence_snapshot_status"] == "missing" and row["evidence_snapshot"] is None
    assert token.confirmed_count == 0 and exported.confirmed is None
    assert read_csv(exported.predictions, "record_id")[sample.rep]["review_status"] == "待补证"
    return dict(accepted_versions=[1, 2], rejected_writes=7, reads=results,
                absent_snapshot="missing", absent_snapshot_conclusions=0)


def legacy_compatibility():
    sample = Scenario()
    path = sample.db_path
    v1 = sample.evidence(sample.rep)
    for key in ("schema_version", "input_name", "input_csv_row", "identity", "engine"):
        del v1[key]
    v1["method_version"] = "fixture-v1"
    old_event = dict(v1, record_id="prediction", method_version="o1-event-review-v1",
                     event_review={"representative_review_ids": [7]})
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, "
                   "model_version TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL)")
        db.executemany("INSERT INTO reviews VALUES (?, ?, ?, ?, ?, '2020-01-01T00:00:00+00:00')", [
            (7, sample.rep, MODEL, NORMAL, "旧直接判断"),
            (8, sample.member, MODEL, ANOMALY, "旧判断缺快照"),
            (9, "prediction", MODEL, NORMAL, "旧继承缺原批次"),
        ])
        db.execute("CREATE TABLE review_evidence (review_id INTEGER PRIMARY KEY, evidence_json TEXT NOT NULL, "
                   "evidence_sha256 TEXT NOT NULL)")
        db.executemany("INSERT INTO review_evidence VALUES (?, ?, ?)", [(7, *canonical(v1)), (9, *canonical(old_event))])
    migrated = reviews.history(path)
    assert [(row["id"], row["evidence_snapshot_status"]) for row in migrated] == [
        (7, "saved"), (8, "missing"), (9, "saved")]
    assert migrated[0]["evidence_snapshot"] == v1 and migrated[0]["evidence_schema_version"] == 1
    assert migrated[2]["evidence_snapshot"] == old_event
    backup = path.with_name(path.name + ".before-evidence-v2.bak")
    assert backup.is_file()
    with closing(sqlite3.connect(backup)) as db:
        assert reviews.schema_kind(db) == "working-v1"
        assert db.execute("SELECT evidence_json,evidence_sha256 FROM review_evidence WHERE review_id=7").fetchone() == canonical(v1)
    exported, token = sample.export()
    with sample.connect() as db:
        assert reviews.schema_kind(db) == "v2"
        assert recovery.current_decision(db, MODEL, sample.member).state == recovery.RecoveryState.PENDING
        assert recovery.current_decision(db, MODEL, "prediction").state == recovery.RecoveryState.INVALIDATED
    conclusions = read_csv(exported.confirmed, "记录ID")
    assert token.confirmed_count == 1 and set(conclusions) == {sample.rep}
    direct = conclusions[sample.rep]
    historical_fields = ("输入文件", "输入CSV行号", "来源文件", "来源行号")
    assert all(direct[field] == "未记录" for field in historical_fields)
    checked = read_csv(exported.predictions, "record_id")
    assert checked[sample.member]["review_status"] == "待补证"
    assert checked["prediction"]["review_status"] == "待复核"
    assert json.loads(exported.history) == migrated
    return dict(migrated=[7, 8, 9], evidence_status=["saved", "missing", "saved"], historical_fields_unrecorded=4,
                current_conclusions=1, missing_review=checked[sample.member]["review_status"],
                old_inheritance=checked["prediction"]["review_status"], backup="working-v1")


PROBES = (conclusion_sources, inactive_inheritance, audit_history, strict_snapshots, legacy_compatibility)


@pytest.mark.parametrize("probe", [probe.__name__ for probe in PROBES])
def test_p0_3_reference_probe(probe):
    workspace = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="p0-3-") as name:
        scratch = Path(name).resolve()
        assert not scratch.is_relative_to(workspace)
        script = scratch / "probe.py"
        shutil.copyfile(__file__, script)
        env = {**os.environ, "PYTHONPATH": str(workspace), "PYTHONNOUSERSITE": "1",
               "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}
        child = subprocess.run([sys.executable, "-B", str(script), probe, str(workspace)], cwd=scratch, env=env,
                               capture_output=True, encoding="utf-8", timeout=60, check=False)
    assert not scratch.exists()
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout)
    assert result["probe"] == probe and result["evidence"]
    print(child.stdout.rstrip())


if __name__ == "__main__":
    workspace = Path(sys.argv[2])
    for module in (dynamic, gate, recovery, reviews, search, workbench):
        assert Path(module.__file__).resolve().is_relative_to(workspace), module.__file__
    for func in (csv_bytes, export_snapshot, confirmed_frame, event_members_and_representatives):
        assert Path(func.__code__.co_filename).resolve().is_relative_to(workspace)
    probe = {item.__name__: item for item in PROBES}[sys.argv[1]]
    print(json.dumps({"probe": probe.__name__, "evidence": probe()}, ensure_ascii=False, sort_keys=True))
