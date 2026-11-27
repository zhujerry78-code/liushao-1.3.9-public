"""Local workspace position and download receipts. Never stores uploaded CSV bytes."""
from contextlib import closing, contextmanager
from datetime import datetime, timezone
import json
import sqlite3

from app_session import session_root, _plain


@contextmanager
def workspace(database):
    root = _plain(session_root(database))
    root.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(_plain(root / 'workspace.sqlite'), timeout=4)) as db, db:
        db.row_factory = sqlite3.Row
        db.execute('CREATE TABLE IF NOT EXISTS positions (digest TEXT, model TEXT, name TEXT, '
                   'source TEXT, updated TEXT, state TEXT, PRIMARY KEY(digest, model))')
        db.execute('CREATE TABLE IF NOT EXISTS downloads (digest TEXT, model TEXT, created TEXT, '
                   'filename TEXT, revision TEXT, scope TEXT)')
        yield db


def now():
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def recent(database):
    with workspace(database) as db:
        return [dict(r, state=json.loads(r['state'])) for r in db.execute(
            'SELECT * FROM positions ORDER BY updated DESC, rowid DESC LIMIT 12')]


def last_position(database, active):
    with workspace(database) as db:
        row = db.execute('SELECT state FROM positions WHERE digest=? AND model=?',
                         (active.digest, active.model)).fetchone()
        return json.loads(row[0]) if row else None


def save_position(database, active, state):
    with workspace(database) as db:
        db.execute('INSERT OR REPLACE INTO positions VALUES (?,?,?,?,?,?)',
                   (active.digest, active.model, active.name, active.source, now(),
                    json.dumps(state, ensure_ascii=False)))


def log_download(database, active, filename, revision, scope):
    with workspace(database) as db:
        db.execute('INSERT INTO downloads VALUES (?,?,?,?,?,?)',
                   (active.digest, active.model, now(), filename, revision, scope))


def download_history(database, active):
    with workspace(database) as db:
        return [dict(r) for r in db.execute('SELECT created AS 时间UTC, filename AS 文件, '
            'revision AS 修订, scope AS 范围 FROM downloads WHERE digest=? AND model=? ORDER BY rowid DESC',
            (active.digest, active.model))]


def clear_reviews(db, database):
    """Keep the two SQLite ledgers and their navigation/audit index in one transaction."""
    with workspace(database):
        pass
    index = _plain(session_root(database) / 'workspace.sqlite')
    db.execute('ATTACH DATABASE ? AS workspace_index', (str(index),))
    db.execute('BEGIN IMMEDIATE')
    try:
        counts = {}
        for table in ('main.review_evidence', 'main.reviews', 'recovery.inheritances',
                      'recovery.operations', 'workspace_index.positions', 'workspace_index.downloads'):
            counts[table] = db.execute('DELETE FROM ' + table).rowcount
        db.commit()
    except BaseException:
        db.rollback()
        raise
    return counts

