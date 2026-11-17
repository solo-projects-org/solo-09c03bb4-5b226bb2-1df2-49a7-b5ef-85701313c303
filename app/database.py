"""SQLite connection management and schema definition.

FastAPI runs sync endpoints in a worker thread, while the lifespan runs in the
main thread, so connections are thread-local (all pointing at the same file).
WAL permits concurrent readers; writers serialize via ``BEGIN IMMEDIATE`` and
the busy timeout.
"""
from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

SCHEMA = """
PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- Every meter reading submission.  Revisions of one source are kept forever
-- for audit; only the highest revision of each source_id is authoritative.
CREATE TABLE IF NOT EXISTS readings (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   TEXT    NOT NULL,
    revision    INTEGER NOT NULL CHECK (revision >= 1),
    device_id   TEXT    NOT NULL,
    ts          TEXT    NOT NULL,           -- UTC, ISO-8601
    meter_no    TEXT    NOT NULL,
    cumulative  REAL    NOT NULL CHECK (cumulative >= 0),
    content_hash TEXT   NOT NULL,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(source_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_readings_device ON readings(device_id);

-- Meter-change handover statements, same revision rules as readings.
CREATE TABLE IF NOT EXISTS handovers (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id      TEXT    NOT NULL,
    revision       INTEGER NOT NULL CHECK (revision >= 1),
    device_id      TEXT    NOT NULL,
    ts             TEXT    NOT NULL,
    old_meter_no   TEXT    NOT NULL,
    old_final      REAL    NOT NULL CHECK (old_final >= 0),
    new_meter_no   TEXT    NOT NULL,
    new_initial    REAL    NOT NULL CHECK (new_initial >= 0),
    content_hash   TEXT    NOT NULL,
    created_at     TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    UNIQUE(source_id, revision)
);
CREATE INDEX IF NOT EXISTS idx_handover_device ON handovers(device_id);

-- Computed consumption intervals.
--   status = 'current' : the interval is eligible to supply data
--   status = 'conflict': the interval overlaps a cross-device meter
--                        registration conflict; energy_kwh is NULL and it must
--                        never be served as current energy
--   status = 'stale'   : superseded after a late submission/correction or an
--                        interrupted rebuild; kept for audit, never served as
--                        current data
-- An undeterminable interval is still a valid 'current' row: energy_kwh is
-- NULL and reason explains why.  We never write 0 to mean "cannot decide".
CREATE TABLE IF NOT EXISTS intervals (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id        TEXT    NOT NULL,
    start_ts         TEXT    NOT NULL,
    end_ts           TEXT    NOT NULL,
    start_meter_no   TEXT    NOT NULL,
    end_meter_no     TEXT    NOT NULL,
    energy_kwh       REAL,                  -- NULL when undeterminable
    status           TEXT    NOT NULL CHECK (status IN ('current','stale','conflict')),
    reason           TEXT,                  -- set exactly when energy_kwh IS NULL
    -- provenance: the source ids / revisions of the readings bounding the interval
    start_source_id  TEXT    NOT NULL,
    start_revision   INTEGER NOT NULL,
    end_source_id    TEXT    NOT NULL,
    end_revision     INTEGER NOT NULL,
    -- handover used to settle a cross-meter interval (if any)
    handover_source_id TEXT,
    handover_revision  INTEGER,
    -- Set when the interval overlaps a cross-device meter conflict.  Such a
    -- row is never eligible to supply current energy (see below).
    conflict         INTEGER NOT NULL DEFAULT 0 CHECK (conflict IN (0,1)),
    conflict_detail  TEXT,
    computed_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    superseded_at    TEXT
);
CREATE INDEX IF NOT EXISTS idx_intervals_device_status
    ON intervals(device_id, status);
CREATE INDEX IF NOT EXISTS idx_intervals_status ON intervals(status);

-- Rebuild run audit trail.  'failed'/'interrupted' runs never delete stale
-- rows, and no run is ever started on service boot.
CREATE TABLE IF NOT EXISTS rebuild_runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    device_id   TEXT,
    status      TEXT    NOT NULL CHECK (status IN ('running','success','failed','interrupted')),
    detail      TEXT,
    started_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
    finished_at TEXT
);
"""


def _open_one(db_path: str) -> sqlite3.Connection:
    if db_path != ":memory:":
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(
        db_path,
        timeout=30,
        check_same_thread=False,
        # Explicit BEGIN/COMMIT control everywhere: the default implicit
        # transactions leave a read statement (e.g. a SELECT in a query
        # endpoint) open on a pooled connection, which then fails the next
        # writer that reuses the same thread with
        # "cannot start a transaction within a transaction".
        isolation_level=None,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA busy_timeout = 30000")
    return conn


class ConnectionPool:
    """SQLite connections checked out for the lifetime of one request.

    FastAPI may run a sync dependency and the endpoint body on *different*
    threadpool threads, so a thread-local connection is not a request-local
    one: two requests could end up sharing a connection (and its transaction).
    Instead every request checks out a dedicated connection and returns it
    when the request ends.  Idle connections are reused, all against the same
    file.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._idle: list[sqlite3.Connection] = []
        self._all: list[sqlite3.Connection] = []
        self._lock = threading.Lock()
        # Fallback for code paths outside a request (startup, tests).
        self._local = threading.local()

    def _new(self) -> sqlite3.Connection:
        conn = _open_one(self.db_path)
        with self._lock:
            self._all.append(conn)
        return conn

    def acquire(self) -> sqlite3.Connection:
        with self._lock:
            conn = self._idle.pop() if self._idle else None
        if conn is None:
            return self._new()
        if conn.in_transaction:  # never hand out a dangling transaction
            conn.rollback()
        return conn

    def release(self, conn: sqlite3.Connection) -> None:
        if conn.in_transaction:
            conn.rollback()
        with self._lock:
            self._idle.append(conn)

    def get(self) -> sqlite3.Connection:
        """The connection for the current request, or a thread-local one."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = self.acquire()
            self._local.conn = conn
        elif conn.in_transaction:
            conn.rollback()
        return conn

    def init_schema(self) -> None:
        conn = self.acquire()
        try:
            init_db(conn)
        finally:
            self._local.conn = conn

    @contextmanager
    def read_snapshot(self):
        """A read transaction over the committed database state.

        The audit endpoint uses this so a concurrent submission either is
        fully visible or not visible at all: the audit can never observe half
        of a submission that is still in flight.  WAL lets this coexist with
        writers.
        """
        conn = self.get()
        conn.execute("BEGIN")
        try:
            yield conn
        finally:
            conn.rollback()

    def close(self) -> None:
        with self._lock:
            for conn in self._all:
                conn.close()
            self._all.clear()
            self._idle.clear()
        # The startup connection is closed too; drop the thread-local handle.
        if getattr(self._local, "conn", None) is not None:
            self._local.conn = None


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate_intervals(conn)
    # A process restart must not resurrect anything: rebuilds left in the
    # 'running' state by a crash are marked interrupted, stale rows stay stale.
    conn.execute(
        "UPDATE rebuild_runs SET status = 'interrupted', "
        "detail = COALESCE(detail,'') || ' [marked on service restart]', "
        "finished_at = strftime('%Y-%m-%dT%H:%M:%fZ','now') "
        "WHERE status = 'running'"
    )
    conn.commit()


def _migrate_intervals(conn: sqlite3.Connection) -> None:
    """Bring a pre-existing intervals table up to the conflict-aware schema.

    SQLite cannot alter a CHECK constraint in place, so a database created
    before the 'conflict' status existed is rebuilt once, preserving every
    row (including stale audit rows).  This is additive only: no historical
    result is deleted or turned back into current data.
    """
    info = conn.execute("PRAGMA table_info(intervals)").fetchall()
    columns = {row["name"] for row in info}
    if "conflict" in columns and "conflict_detail" in columns:
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='intervals'"
        ).fetchone()
        if row is not None and "'conflict'" in (row["sql"] or ""):
            return

    conn.execute("BEGIN IMMEDIATE")
    try:
        conn.execute("ALTER TABLE intervals RENAME TO intervals_old")
        conn.executescript(
            """
            CREATE TABLE intervals (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                device_id        TEXT    NOT NULL,
                start_ts         TEXT    NOT NULL,
                end_ts           TEXT    NOT NULL,
                start_meter_no   TEXT    NOT NULL,
                end_meter_no     TEXT    NOT NULL,
                energy_kwh       REAL,
                status           TEXT    NOT NULL CHECK (status IN ('current','stale','conflict')),
                reason           TEXT,
                start_source_id  TEXT    NOT NULL,
                start_revision   INTEGER NOT NULL,
                end_source_id    TEXT    NOT NULL,
                end_revision     INTEGER NOT NULL,
                handover_source_id TEXT,
                handover_revision  INTEGER,
                conflict         INTEGER NOT NULL DEFAULT 0 CHECK (conflict IN (0,1)),
                conflict_detail  TEXT,
                computed_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ','now')),
                superseded_at    TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_intervals_device_status
                ON intervals(device_id, status);
            CREATE INDEX IF NOT EXISTS idx_intervals_status ON intervals(status);
            """
        )
        old_cols = {r["name"] for r in conn.execute("PRAGMA table_info(intervals_old)")}
        conflict_expr = "conflict" if "conflict" in old_cols else "0"
        detail_expr = "conflict_detail" if "conflict_detail" in old_cols else "NULL"
        conn.execute(
            f"INSERT INTO intervals (id, device_id, start_ts, end_ts, start_meter_no, "
            f"end_meter_no, energy_kwh, status, reason, start_source_id, "
            f"start_revision, end_source_id, end_revision, handover_source_id, "
            f"handover_revision, conflict, conflict_detail, computed_at, superseded_at) "
            f"SELECT id, device_id, start_ts, end_ts, start_meter_no, end_meter_no, "
            f"energy_kwh, status, reason, start_source_id, start_revision, "
            f"end_source_id, end_revision, handover_source_id, handover_revision, "
            f"{conflict_expr}, {detail_expr}, computed_at, superseded_at "
            f"FROM intervals_old"
        )
        conn.execute("DROP TABLE intervals_old")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
