"""Concurrent startup and damaged legacy evidence keep review history usable."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
import hashlib
import json
import sqlite3
import threading

import pytest

from liushao import reviews


def legacy_database(path, raw=None):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE reviews (id INTEGER PRIMARY KEY AUTOINCREMENT, record_id TEXT NOT NULL, model_version TEXT NOT NULL, status TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL)')
        db.execute("INSERT INTO reviews VALUES(7,'record','model','待补证','保留原有备注','2020-01-01T00:00:00+00:00')")
        if raw is not None:
            db.execute('CREATE TABLE review_evidence (review_id INTEGER PRIMARY KEY, evidence_json TEXT NOT NULL, evidence_sha256 TEXT NOT NULL)')
            db.execute('INSERT INTO review_evidence VALUES(?,?,?)',
                       (7,raw,hashlib.sha256(raw.encode('utf-8')).hexdigest()))


def database_dump(path):
    with closing(sqlite3.connect(path)) as db:
        return list(db.iterdump())


def snapshot(version):
    return dict(schema_version=version,record_id='record',model_version='model',
                method_version='synthetic',input_sha256='a'*64,model_sha256='b'*64,
                descriptor_sha256='c'*64,cards=[],selection={},decomposition={},
                scores=dict(rf=.5,lr=.5,rf_threshold=.5,lr_threshold=.5))


@pytest.mark.parametrize('raw', [
    'not json', '[]', 'null', '{"schema_version":null}',
    '{"schema_version":[]}', '{"schema_version":9223372036854775808}',
] + [json.dumps(snapshot(version)) for version in (0,True,1.0,3,'2')])
def test_damaged_legacy_evidence_preserves_review_bytes_and_backup(tmp_path,raw):
    path=tmp_path/'legacy.sqlite'
    legacy_database(path,raw)
    original=database_dump(path)
    row=reviews.history(path)[0]
    assert row['id']==7 and row['note']=='保留原有备注' and row['status']=='待补证'
    assert row['evidence_snapshot_status']=='invalid' and row['evidence_snapshot'] is None
    with closing(sqlite3.connect(path)) as db:
        assert db.execute('SELECT payload_json,payload_sha256 FROM review_evidence').fetchone()==(
            raw,hashlib.sha256(raw.encode('utf-8')).hexdigest())
        assert db.execute('PRAGMA integrity_check').fetchone()==('ok',)
    backup=path.with_name(path.name+'.before-evidence-v2.bak')
    assert database_dump(backup)==original
    backup_bytes=backup.read_bytes()
    # The backup connection must be closed on Windows too.
    renamed=backup.with_suffix('.checked')
    backup.rename(renamed)
    renamed.rename(backup)
    reviews.save_review(path,'record','model','待补证','后续复核',expected_review_id=7)
    assert [entry['id'] for entry in reviews.history(path)]==[7,8]
    assert backup.read_bytes()==backup_bytes
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('version',[0,True,1.0,3,'2',None,[]])
def test_unsupported_snapshot_version_is_rejected_before_creating_database(tmp_path,version):
    path=tmp_path/'absent'/'reviews.sqlite'
    with pytest.raises(ValueError):
        reviews.save_review(path,'record','model','待补证','test',evidence_snapshot=snapshot(version))
    assert not path.parent.exists()


@pytest.mark.parametrize('legacy',[False,True])
def test_simultaneous_schema_upgrade_rechecks_under_write_lock(tmp_path,monkeypatch,legacy):
    path=tmp_path/'shared.sqlite'
    if legacy:
        legacy_database(path)
    ready=threading.Barrier(2)
    original=reviews.schema_kind
    def synchronized(db):
        kind=original(db)
        if not db.in_transaction:
            ready.wait(timeout=10)
        return kind
    monkeypatch.setattr(reviews,'schema_kind',synchronized)
    def upgrade(_):
        with closing(sqlite3.connect(path,timeout=10)) as db:
            reviews.upgrade_schema(db)
            return original(db)
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(upgrade,range(2)))==['v2','v2']
    monkeypatch.setattr(reviews,'schema_kind',original)
    rows=reviews.history(path)
    assert len(rows)==int(legacy)
    if legacy:
        assert rows[0]['note']=='保留原有备注'


def test_simultaneous_legacy_connections_publish_complete_original_backup(tmp_path,monkeypatch):
    path=tmp_path/'shared.sqlite'
    legacy_database(path,'not json')
    original_dump=database_dump(path)
    ready=threading.Barrier(2)
    original=reviews._backup_before_upgrade
    def synchronized(db,path):
        ready.wait(timeout=10)
        return original(db,path)
    monkeypatch.setattr(reviews,'_backup_before_upgrade',synchronized)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results=list(pool.map(lambda _:reviews.history(path),range(2)))
    assert results[0]==results[1]
    assert results[0][0]['evidence_snapshot_status']=='invalid'
    backup=path.with_name(path.name+'.before-evidence-v2.bak')
    assert database_dump(backup)==original_dump
    with closing(sqlite3.connect(backup)) as db:
        assert reviews.schema_kind(db)=='working-v1'
    assert not list(tmp_path.glob('*.tmp'))


def test_failed_backup_does_not_publish_partial_file_or_leave_handle(tmp_path):
    path=tmp_path/'legacy.sqlite'
    legacy_database(path)
    original=path.read_bytes()
    class FailedBackup:
        def backup(self,target):
            target.execute('CREATE TABLE incomplete(value)')
            raise sqlite3.OperationalError('injected backup failure')
    with pytest.raises(sqlite3.OperationalError,match='injected backup failure'):
        reviews._backup_before_upgrade(FailedBackup(),path)
    assert path.read_bytes()==original
    assert sorted(p.name for p in tmp_path.iterdir())==['legacy.sqlite']
