"""
backend/database.py
BSW Kota Bogor Multi-Camera Traffic Intelligence Dashboard.
SQLite WAL Persistence Layer (F-ARCH-01).

Implements:
- High-performance SQLite database in WAL (Write-Ahead Logging) mode.
- Thread-safe connection pooling and pragma optimizations.
- Schema DDL for cameras, topology_edges, vehicle_sightings and observation_windows.
- Automatic seeding of every BSW CCTV camera and topology edge from data/bogor_camera_topology.json.
- Core querying API: vehicle sightings insertion, recent sighting retrieval, multi-filter search,
  counting-line observation windows and rates, and traffic statistics.
- All-time dashboard aggregates are O(cameras), not O(sightings): AFTER INSERT triggers on
  ``vehicle_sightings`` maintain ``sighting_totals`` / ``camera_track_totals`` / ``sighting_tracks`` /
  ``sighting_counters`` (see ``TOTALS_DDL``). The counters count *inserts*, so
  deleting old sightings (retention) never lowers an all-time total.
- Storage upkeep: ``compact()`` (one-off VACUUM into incremental auto-vacuum), ``incremental_vacuum()``,
  and opt-in retention (``prune_sightings``).
"""

import json
import logging
import shutil
from contextlib import contextmanager
import os
import sqlite3
import threading
import weakref
import time
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from backend.camera_sources import load_extra_nodes

try:  # tunables; the module stays importable on its own
    from backend.config import DB_JOURNAL_SIZE_LIMIT_BYTES
except ImportError:  # pragma: no cover
    DB_JOURNAL_SIZE_LIMIT_BYTES = 32 * 1024 * 1024

logger = logging.getLogger("database")

# Base path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.environ.get("TRAFFIC_DB_PATH") or os.path.join(PROJECT_ROOT, "data", "traffic_db.sqlite")
DEFAULT_TOPOLOGY_PATH = os.path.join(PROJECT_ROOT, "data", "bogor_camera_topology.json")

SCHEMA_DDL = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;

CREATE TABLE IF NOT EXISTS cameras (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    corridor TEXT NOT NULL,
    latitude REAL NOT NULL,
    longitude REAL NOT NULL,
    camera_type TEXT DEFAULT 'FIXED',
    stream_url TEXT,
    status TEXT DEFAULT 'online'
);

CREATE TABLE IF NOT EXISTS topology_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id INTEGER NOT NULL,
    target_id INTEGER NOT NULL,
    distance_m REAL NOT NULL,
    typical_speed_kmh REAL NOT NULL,
    t_min_s REAL NOT NULL,
    t_max_s REAL NOT NULL,
    corridor TEXT,
    FOREIGN KEY (source_id) REFERENCES cameras(id),
    FOREIGN KEY (target_id) REFERENCES cameras(id)
);

CREATE TABLE IF NOT EXISTS vehicle_sightings (
    sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    local_track_id INTEGER NOT NULL,
    vehicle_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    heading_deg REAL NOT NULL,
    speed_px_s REAL NOT NULL,
    timestamp_iso TEXT NOT NULL,
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

CREATE INDEX IF NOT EXISTS idx_sightings_cam_time ON vehicle_sightings(camera_id, timestamp_iso);
CREATE INDEX IF NOT EXISTS idx_sightings_time ON vehicle_sightings(timestamp_iso);

-- (idx_sightings_agg / idx_sightings_local_track used to live here as covering indexes for the
-- all-time dashboard aggregates; those now come from the trigger-maintained totals tables and the
-- migration drops both indexes -- they cost ~60 bytes per sighting.)

CREATE TABLE IF NOT EXISTS maintenance_state (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
);

-- Traffic volume: one row per observation window of a camera (a rolling visit, or one minute of a
-- watched camera). Rates are crossings / observed_s, so quiet periods count toward the rate too.
CREATE TABLE IF NOT EXISTS observation_windows (
    window_id INTEGER PRIMARY KEY AUTOINCREMENT,
    camera_id INTEGER NOT NULL,
    start_ts REAL NOT NULL,
    end_ts REAL NOT NULL,
    start_iso TEXT NOT NULL,
    observed_s REAL NOT NULL,
    vehicles INTEGER NOT NULL DEFAULT 0,
    dir_pos INTEGER NOT NULL DEFAULT 0,
    dir_neg INTEGER NOT NULL DEFAULT 0,
    cars INTEGER NOT NULL DEFAULT 0,
    motorcycles INTEGER NOT NULL DEFAULT 0,
    trucks INTEGER NOT NULL DEFAULT 0,
    buses INTEGER NOT NULL DEFAULT 0,
    line_counts INTEGER NOT NULL DEFAULT 0,
    passage_counts INTEGER NOT NULL DEFAULT 0,
    kind TEXT NOT NULL DEFAULT 'visit',
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);
CREATE INDEX IF NOT EXISTS idx_obs_cam_time ON observation_windows(camera_id, end_ts);
CREATE INDEX IF NOT EXISTS idx_obs_time ON observation_windows(end_ts);
"""


# ── All-time aggregates maintained by triggers ──────────────────────────────────
# The dashboard's all-time numbers used to be COUNT / COUNT(DISTINCT) over every sighting (~0.6 s
# per refresh at a week of data, growing linearly, plus ~60 B/sighting of covering indexes).
# These tables are updated by AFTER INSERT triggers instead, so reading them is O(cameras).
#
# SEMANTICS: they count *inserts*. Deleting sightings (retention, ``prune_sightings``) does not
# lower them -- "all-time" means all time, not "what is still stored".
#   sighting_totals(camera_id, vehicle_class)  rows + last timestamp per camera and class
#   sighting_tracks(local_track_id, camera_id) set of distinct tracks seen (exact COUNT(DISTINCT))
#   camera_track_totals(camera_id)             distinct local_track_id per camera
#   sighting_counters(key)                     'tracks' (distinct local_track_id over all cameras)
TOTALS_MARKER = "totals_v1"
COMPACT_MARKER = "compacted_v1"
COMPACT_BUSY_TIMEOUT_MS = 10_000

TOTALS_TABLES_DDL = """
CREATE TABLE IF NOT EXISTS sighting_totals (
    camera_id INTEGER NOT NULL,
    vehicle_class TEXT NOT NULL,
    n INTEGER NOT NULL DEFAULT 0,
    last_seen_iso TEXT,
    PRIMARY KEY (camera_id, vehicle_class)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS sighting_tracks (
    local_track_id INTEGER NOT NULL,
    camera_id INTEGER NOT NULL,
    PRIMARY KEY (local_track_id, camera_id)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS camera_track_totals (
    camera_id INTEGER PRIMARY KEY,
    tracks INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS sighting_counters (
    key TEXT PRIMARY KEY,
    value INTEGER NOT NULL
) WITHOUT ROWID;
"""

TOTALS_TRIGGERS_DDL = """
CREATE TRIGGER IF NOT EXISTS trg_sightings_totals AFTER INSERT ON vehicle_sightings
BEGIN
    INSERT INTO sighting_totals (camera_id, vehicle_class, n, last_seen_iso)
    VALUES (NEW.camera_id, NEW.vehicle_class, 1, NEW.timestamp_iso)
    ON CONFLICT(camera_id, vehicle_class) DO UPDATE SET
        n = n + 1,
        last_seen_iso = max(last_seen_iso, excluded.last_seen_iso);

    -- distinct tracks: the global and per-camera counters move only when the pair is new
    UPDATE sighting_counters SET value = value + 1
    WHERE key = 'tracks'
      AND NOT EXISTS (SELECT 1 FROM sighting_tracks WHERE local_track_id = NEW.local_track_id);
    INSERT INTO camera_track_totals (camera_id, tracks)
    SELECT NEW.camera_id, 1
    WHERE NOT EXISTS (SELECT 1 FROM sighting_tracks
                      WHERE local_track_id = NEW.local_track_id AND camera_id = NEW.camera_id)
    ON CONFLICT(camera_id) DO UPDATE SET tracks = tracks + 1;
    INSERT OR IGNORE INTO sighting_tracks (local_track_id, camera_id)
    VALUES (NEW.local_track_id, NEW.camera_id);
END;
"""

# Rebuilds every totals table from the rows currently in vehicle_sightings.
TOTALS_BACKFILL_SQL = (
    "DELETE FROM sighting_totals",
    "DELETE FROM sighting_tracks",
    "DELETE FROM camera_track_totals",
    "DELETE FROM sighting_counters",
    """INSERT INTO sighting_totals (camera_id, vehicle_class, n, last_seen_iso)
       SELECT camera_id, vehicle_class, COUNT(*), MAX(timestamp_iso)
       FROM vehicle_sightings GROUP BY camera_id, vehicle_class""",
    # sorted so the WITHOUT ROWID b-tree is built by appending
    """INSERT INTO sighting_tracks (local_track_id, camera_id)
       SELECT local_track_id, camera_id FROM
           (SELECT DISTINCT local_track_id, camera_id FROM vehicle_sightings
            ORDER BY local_track_id, camera_id)""",
    """INSERT INTO camera_track_totals (camera_id, tracks)
       SELECT camera_id, COUNT(*) FROM sighting_tracks GROUP BY camera_id""",
    """INSERT INTO sighting_counters (key, value) VALUES
           ('tracks', (SELECT COUNT(DISTINCT local_track_id) FROM sighting_tracks))""",
)


@contextmanager
def _transaction(conn: sqlite3.Connection):
    """Commits on success and rolls back on any error. Without the rollback, a failed write
    (e.g. a FOREIGN KEY violation) leaves the implicit transaction open and its connection
    holds the database write lock indefinitely, blocking every other writer."""
    try:
        yield conn.cursor()
        conn.commit()
    except BaseException:
        conn.rollback()
        raise


def _configure_connection(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Applies high-throughput WAL pragmas and Row factory to connection."""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    # Truncate the WAL back to this size after a checkpoint instead of leaving it at its high-water mark.
    conn.execute(f"PRAGMA journal_size_limit = {int(DB_JOURNAL_SIZE_LIMIT_BYTES)};")
    return conn


def get_db_connection(db_path: Optional[str] = None) -> sqlite3.Connection:
    """
    Returns an optimized, WAL-configured SQLite connection.
    Defaults to data/traffic_db.sqlite.
    """
    path = db_path or DEFAULT_DB_PATH
    if path != ":memory:":
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    return _configure_connection(conn)


def _table_columns(conn: sqlite3.Connection, table: str) -> set:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _rebuild_totals(cur: sqlite3.Cursor) -> None:
    """Recomputes every totals table from the rows that exist now (call inside a transaction)."""
    for statement in TOTALS_BACKFILL_SQL:
        cur.execute(statement)


# Left behind in databases created while the dashboard had cross-camera re-identification.
_REID_TABLES = ("active_targets", "handoff_events", "global_vehicles")   # active_targets first: it references sightings
_REID_SIGHTING_COLUMNS = ("global_vehicle_id", "embedding", "color_hist", "crop_thumbnail")


def _drop_reid_schema(conn: sqlite3.Connection) -> bool:
    """Drops the re-identification tables, counters and sighting columns from an existing database.
    ``embedding`` / ``color_hist`` were NOT NULL, so sighting inserts fail until they are gone.
    The totals trigger read ``global_vehicle_id`` (SQLite refuses to drop a column a trigger uses),
    so it is dropped first and, if the totals migration already ran, recreated without it.
    Dropping a column rewrites the table once (needs SQLite >= 3.35). Returns True if it changed anything."""
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    stale_columns = [c for c in _REID_SIGHTING_COLUMNS if c in _table_columns(conn, "vehicle_sightings")]
    if not (tables & set(_REID_TABLES) or stale_columns):
        return False
    t0 = time.perf_counter()
    with _transaction(conn) as cur:
        cur.execute("BEGIN IMMEDIATE")
        totals_done = cur.execute("SELECT 1 FROM maintenance_state WHERE key = ?", (TOTALS_MARKER,)).fetchone()
        cur.execute("DROP TRIGGER IF EXISTS trg_sightings_totals")
        cur.execute("DROP TRIGGER IF EXISTS trg_handoffs_total")
        for table in _REID_TABLES:
            cur.execute(f"DROP TABLE IF EXISTS {table}")
        cur.execute("DROP INDEX IF EXISTS idx_sightings_global_id")
        cur.execute("DROP INDEX IF EXISTS idx_active_targets_cam")
        for column in stale_columns:
            cur.execute(f"ALTER TABLE vehicle_sightings DROP COLUMN {column}")
        if "sighting_counters" in tables:
            cur.execute("DELETE FROM sighting_counters WHERE key IN ('global_vehicles', 'handoffs')")
        cur.execute("DELETE FROM maintenance_state WHERE key = 'embeddings_expired_upto'")
        if totals_done:
            for part in _split_statements(TOTALS_TRIGGERS_DDL):
                cur.execute(part)
    logger.info(f"Removed re-identification tables/columns in {time.perf_counter() - t0:.1f}s "
                f"(dropped columns: {stale_columns or 'none'})")
    return True


def migrate_schema(conn: sqlite3.Connection) -> Dict[str, Any]:
    """
    Idempotent schema migration, run on every start (a no-op once applied):

    * Re-identification leftovers (tables, counters, sighting columns) are dropped
      (``_drop_reid_schema``).
    * The totals tables + AFTER INSERT triggers are created and backfilled *once*, in the same
      ``BEGIN IMMEDIATE`` transaction (so no insert can slip between the backfill and the triggers),
      then the two covering indexes they replace are dropped. Marker: ``maintenance_state.totals_v1``.
    """
    report: Dict[str, Any] = {"totals_backfilled": False, "backfill_s": 0.0, "dropped_indexes": [],
                              "reid_schema_dropped": _drop_reid_schema(conn)}

    def totals_done() -> bool:
        return conn.execute("SELECT 1 FROM maintenance_state WHERE key = ?", (TOTALS_MARKER,)).fetchone() is not None

    if totals_done():
        return report

    t0 = time.perf_counter()
    with _transaction(conn) as cur:
        cur.execute("BEGIN IMMEDIATE")
        if totals_done():          # another process migrated while we waited for the write lock
            return report
        for statement in (TOTALS_TABLES_DDL, TOTALS_TRIGGERS_DDL):
            for part in _split_statements(statement):
                cur.execute(part)
        _rebuild_totals(cur)
        for idx in ("idx_sightings_agg", "idx_sightings_local_track"):
            if cur.execute("SELECT 1 FROM sqlite_master WHERE type = 'index' AND name = ?", (idx,)).fetchone():
                cur.execute(f"DROP INDEX {idx}")
                report["dropped_indexes"].append(idx)
        cur.execute("INSERT INTO maintenance_state (key, value) VALUES (?, 1) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (TOTALS_MARKER,))
    report["totals_backfilled"] = True
    report["backfill_s"] = round(time.perf_counter() - t0, 3)
    logger.info(f"Totals tables created and backfilled in {report['backfill_s']}s; "
                f"dropped indexes: {report['dropped_indexes'] or 'none'}")
    return report


def _like_contains(value: str) -> str:
    """LIKE pattern (use with ESCAPE '\\') matching ``value`` literally anywhere in a string."""
    return "%" + value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"


def _split_statements(script: str) -> List[str]:
    """Splits a DDL script into single statements (``executescript`` would COMMIT the transaction we
    are in). ``sqlite3.complete_statement`` knows that a trigger body's ';' is not the end."""
    statements: List[str] = []
    buf = ""
    for line in script.splitlines():
        if not buf and (not line.strip() or line.strip().startswith("--")):
            continue
        buf += line + "\n"
        if sqlite3.complete_statement(buf):
            statements.append(buf.strip())
            buf = ""
    if buf.strip():
        statements.append(buf.strip())
    return statements


def seed_cameras_from_topology(
    conn: sqlite3.Connection, topology_path: Optional[str] = None
) -> int:
    """
    Seeds cameras and topology edges from data/bogor_camera_topology.json, adding any camera or
    edge the database does not have yet (new cameras appear when the BSW site adds them).
    Existing camera rows are left as they are. Returns the number of cameras in the database.
    """
    path = topology_path or DEFAULT_TOPOLOGY_PATH
    cur = conn.cursor()
    if not os.path.exists(path):
        cur.execute("SELECT count(*) FROM cameras;")
        return cur.fetchone()[0]

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes", []) + load_extra_nodes()
    edges = data.get("edges", [])

    cur.execute("SELECT id FROM cameras;")
    have_cams = {row[0] for row in cur.fetchall()}
    cur.execute("SELECT source_id, target_id FROM topology_edges;")
    have_edges = {(row[0], row[1]) for row in cur.fetchall()}

    new_nodes = [n for n in nodes if n["id"] not in have_cams]
    new_edges = [e for e in edges
                 if e.get("source_id") is not None and e.get("target_id") is not None
                 and (e["source_id"], e["target_id"]) not in have_edges]
    if new_nodes or new_edges:
        with _transaction(conn) as cur:
            for node in new_nodes:
                cur.execute(
                    """
                    INSERT OR IGNORE INTO cameras (
                        id, name, corridor, latitude, longitude, camera_type, stream_url, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        node["id"],
                        node.get("name", f"CCTV-{node['id']}"),
                        node.get("corridor", "BOGOR_CITY"),
                        node.get("latitude", node.get("lat", 0.0)),
                        node.get("longitude", node.get("lng", 0.0)),
                        node.get("camera_type", "FIXED"),
                        node.get("stream_url", ""),
                        node.get("status", "online"),
                    ),
                )
            for edge in new_edges:
                dist = float(edge.get("distance_m", edge.get("road_distance_m", 0.0)))
                speed = float(edge.get("typical_speed_kmh", 30.0))
                t_win = edge.get("travel_time_window_s", [edge.get("t_min_s", 10.0), edge.get("t_max_s", 60.0)])
                cur.execute(
                    """
                    INSERT INTO topology_edges (
                        source_id, target_id, distance_m, typical_speed_kmh, t_min_s, t_max_s, corridor
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (edge["source_id"], edge["target_id"], dist, speed,
                     float(t_win[0]), float(t_win[1]), edge.get("corridor", "")),
                )

    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM cameras;")
    return cur.fetchone()[0]


class _ThreadConnection:
    """One thread's connection. It is held only by that thread's ``threading.local``, so it is
    closed when the thread ends -- request-pool threads come and go, and keeping every
    connection they ever opened leaked ~2 file descriptors and ~0.2 MB per thread."""

    __slots__ = ("conn", "__weakref__")

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def close(self) -> None:
        conn, self.conn = self.conn, None
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass

    def __del__(self):
        self.close()


class Database:
    """
    Thread-safe SQLite Database manager with connection pooling and caching.
    """

    def __init__(self, db_path: Optional[str] = None, auto_seed: bool = True):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._all_conns = weakref.WeakSet()   # _ThreadConnection of every live thread
        self._all_conns_lock = threading.Lock()
        self.migration_report: Dict[str, Any] = {}

        # Initialize schema using a primary setup connection
        init_conn = get_db_connection(self.db_path)
        try:
            init_conn.executescript(SCHEMA_DDL)
            init_conn.commit()
            self.migration_report = migrate_schema(init_conn)
            if auto_seed:
                seed_cameras_from_topology(init_conn)
        finally:
            if self.db_path == ":memory:":
                # Keep reference to in-memory connection so it survives across calls
                self._shared_conn = init_conn
            else:
                self._shared_conn = None
                init_conn.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """Returns the thread-local connection for the current thread."""
        if self._shared_conn is not None:
            return self._shared_conn

        holder = getattr(self._thread_local, "holder", None)
        if holder is None or holder.conn is None:
            holder = _ThreadConnection(get_db_connection(self.db_path))
            self._thread_local.holder = holder
            with self._all_conns_lock:
                self._all_conns.add(holder)
        return holder.conn

    def close(self):
        """Closes all thread-local or shared connections across all threads."""
        with self._all_conns_lock:
            for holder in list(self._all_conns):
                holder.close()
            self._all_conns.clear()

        if self._shared_conn is not None:
            try:
                self._shared_conn.close()
            except Exception:
                pass
        self._shared_conn = None
        self._thread_local.holder = None

    def optimize(self) -> None:
        """Refreshes the query planner's statistics where they are stale (cheap; SQLite decides).
        Without statistics windowed dashboard queries scanned whole indexes. ``analysis_limit``
        makes any ANALYZE it triggers sample ~1000 index entries per index instead of reading the
        whole (multi-million row) index, so this stays milliseconds at any table size."""
        conn = self.conn
        conn.execute("PRAGMA analysis_limit = 1000;")
        conn.execute("PRAGMA optimize;")

    # ── Storage upkeep ──────────────────────────────────────────────────────────

    def _file_bytes(self) -> int:
        """Main database file + WAL, in bytes (0 for an in-memory database)."""
        if self.db_path == ":memory:":
            return 0
        total = 0
        for suffix in ("", "-wal"):
            try:
                total += os.path.getsize(self.db_path + suffix)
            except OSError:
                pass
        return total

    def compact(self, force: bool = False) -> Dict[str, Any]:
        """
        One-off storage conversion + defragmentation. Dropped columns (the old appearance blobs) and
        deleted rows leave pages half-empty and SQLite never reuses those; ``VACUUM`` rewrites the file
        compactly and, after ``PRAGMA auto_vacuum = INCREMENTAL``, converts it so later deletes
        (retention) can hand pages back through ``incremental_vacuum``.

        Runs at most once per database (marker row ``maintenance_state.compacted_v1``) unless
        ``force``. It is never called implicitly -- the server calls it at start-up, before the
        tracker starts writing. It needs free disk space of about the database size and an
        exclusive write lock for the duration (seconds per GB); it *fails soft*: if the database is
        busy or the disk is too full it logs, returns ``compacted=False`` with a ``reason`` and
        leaves the marker unset so the next start retries.

        Returns ``{"compacted", "skipped", "reason", "before_bytes", "after_bytes",
        "reclaimed_bytes", "seconds", "auto_vacuum"}`` (sizes are main file + WAL).
        """
        result: Dict[str, Any] = {"compacted": False, "skipped": None, "reason": None, "before_bytes": 0,
                                  "after_bytes": 0, "reclaimed_bytes": 0, "seconds": 0.0, "auto_vacuum": None}
        if self.db_path == ":memory:":
            result.update(skipped=True, reason="in-memory database")
            return result
        try:
            marked = self.conn.execute("SELECT 1 FROM maintenance_state WHERE key = ?", (COMPACT_MARKER,)).fetchone()
        except sqlite3.Error as exc:
            logger.warning(f"compact(): cannot read marker: {exc}")
            result.update(skipped=True, reason=f"database error: {exc}")
            return result
        result["before_bytes"] = result["after_bytes"] = self._file_bytes()
        if marked and not force:
            result.update(skipped=True, reason="already compacted")
            return result

        free = shutil.disk_usage(os.path.dirname(os.path.abspath(self.db_path))).free
        needed = int(1.2 * os.path.getsize(self.db_path)) + (64 << 20)
        if free < needed:
            logger.warning(f"compact(): not enough free disk ({free >> 20} MB free, ~{needed >> 20} MB needed); skipped")
            result.update(skipped=True, reason=f"insufficient disk space ({free >> 20} MB free, {needed >> 20} MB needed)")
            return result

        t0 = time.perf_counter()
        conn = None
        try:
            conn = get_db_connection(self.db_path)
            conn.execute(f"PRAGMA busy_timeout = {COMPACT_BUSY_TIMEOUT_MS};")
            conn.execute("PRAGMA temp_store = FILE;")           # VACUUM builds the copy in a temp db; not in RAM
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
            conn.execute("PRAGMA auto_vacuum = INCREMENTAL;")
            conn.execute("VACUUM;")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")    # the rewritten pages arrive via the WAL
            result["auto_vacuum"] = conn.execute("PRAGMA auto_vacuum;").fetchone()[0]
            with _transaction(conn) as cur:
                cur.execute("INSERT INTO maintenance_state (key, value) VALUES (?, ?) "
                            "ON CONFLICT(key) DO UPDATE SET value = excluded.value", (COMPACT_MARKER, int(time.time())))
        except sqlite3.OperationalError as exc:      # busy / locked / disk full: leave the database as it was
            logger.warning(f"compact(): database busy or unavailable, will retry next start: {exc}")
            result.update(reason=f"busy or unavailable: {exc}", seconds=round(time.perf_counter() - t0, 2))
            result["after_bytes"] = self._file_bytes()
            return result
        finally:
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        result["after_bytes"] = self._file_bytes()
        result["reclaimed_bytes"] = max(0, result["before_bytes"] - result["after_bytes"])
        result["compacted"] = True
        result["seconds"] = round(time.perf_counter() - t0, 2)
        logger.info(f"Database compacted: {result['before_bytes'] / 1e6:.0f} MB -> {result['after_bytes'] / 1e6:.0f} MB "
                    f"in {result['seconds']}s (auto_vacuum={result['auto_vacuum']})")
        return result

    def incremental_vacuum(self, max_pages: int = 1000, conn: Optional[sqlite3.Connection] = None) -> int:
        """
        Returns up to ``max_pages`` free pages to the filesystem (the file shrinks at the next WAL
        checkpoint). Only has an effect once ``compact()`` has switched the file to incremental
        auto-vacuum; otherwise it returns 0. Returns the number of pages freed. Takes the write
        lock for as long as the pages take to move, so call it with a modest ``max_pages``.
        """
        target_conn = conn or self.conn
        max_pages = max(0, int(max_pages))
        if max_pages == 0 or target_conn.execute("PRAGMA auto_vacuum;").fetchone()[0] != 2:
            return 0
        before = target_conn.execute("PRAGMA freelist_count;").fetchone()[0]
        if before == 0:
            return 0
        with _transaction(target_conn):
            # executescript steps the pragma to completion; cursor.execute() would step it only once,
            # i.e. free a single page (the pragma yields one result row per page and has no columns).
            # It commits first, so don't call this with a caller transaction still open.
            target_conn.executescript(f"PRAGMA incremental_vacuum({max_pages});")
        after = target_conn.execute("PRAGMA freelist_count;").fetchone()[0]
        return max(0, before - after)

    def freelist_pages(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """Number of free (reusable / reclaimable) pages in the database file."""
        return int((conn or self.conn).execute("PRAGMA freelist_count;").fetchone()[0])

    def insert_vehicle_sighting(
        self,
        camera_id: int,
        local_track_id: int,
        vehicle_class: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        heading_deg: float,
        speed_px_s: float,
        timestamp_iso: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """
        Inserts a vehicle sighting into the database.
        Returns the generated sighting_id.
        """
        target_conn = conn or self.conn
        with _transaction(target_conn) as cur:
            cur.execute(
                """
                INSERT INTO vehicle_sightings (
                    camera_id, local_track_id, vehicle_class,
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    heading_deg, speed_px_s, timestamp_iso
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    local_track_id,
                    vehicle_class,
                    float(confidence),
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    float(heading_deg),
                    float(speed_px_s),
                    timestamp_iso,
                ),
            )
        return cur.lastrowid

    def get_recent_sightings(
        self,
        camera_id: int,
        window_s: float = 60.0,
        limit: int = 100,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns recent vehicle sightings at camera_id.
        Determines temporal window relative to either the latest sighting in DB or current wall clock.
        """
        target_conn = conn or self.conn
        cur = target_conn.cursor()

        # Find latest timestamp for this camera
        cur.execute(
            "SELECT max(timestamp_iso) FROM vehicle_sightings WHERE camera_id = ?",
            (camera_id,),
        )
        row = cur.fetchone()
        latest_ts_str = row[0] if row and row[0] else None

        if not latest_ts_str:
            return []

        # Parse reference timestamp
        try:
            latest_dt = datetime.fromisoformat(latest_ts_str.replace("Z", "+00:00"))
            cutoff_dt = datetime.fromtimestamp(
                latest_dt.timestamp() - window_s, tz=timezone.utc
            )
            cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            cutoff_iso = latest_ts_str

        cur.execute(
            """
            SELECT s.*, c.name as camera_name, c.corridor
            FROM vehicle_sightings s
            JOIN cameras c ON s.camera_id = c.id
            WHERE s.camera_id = ? AND s.timestamp_iso >= ?
            ORDER BY s.timestamp_iso DESC
            LIMIT ?
            """,
            (camera_id, cutoff_iso, limit),
        )
        return [dict(r) for r in cur.fetchall()]

    def search_vehicles(
        self,
        vehicle_class: Optional[str] = None,
        camera_id: Optional[int] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 50,
        conn: Optional[sqlite3.Connection] = None,
    ) -> List[Dict[str, Any]]:
        """
        Searches vehicle sightings with composite filters.
        """
        target_conn = conn or self.conn
        query = ["SELECT s.*, c.name as camera_name FROM vehicle_sightings s JOIN cameras c ON s.camera_id = c.id WHERE 1=1"]
        params: List[Any] = []

        if vehicle_class:
            query.append("AND s.vehicle_class = ?")
            params.append(vehicle_class)
        if camera_id is not None:
            query.append("AND s.camera_id = ?")
            params.append(camera_id)
        if start_time:
            query.append("AND s.timestamp_iso >= ?")
            params.append(start_time)
        if end_time:
            query.append("AND s.timestamp_iso <= ?")
            params.append(end_time)

        query.append("ORDER BY s.timestamp_iso DESC LIMIT ?")
        params.append(limit)

        cur = target_conn.cursor()
        cur.execute(" ".join(query), params)
        return [dict(r) for r in cur.fetchall()]

    def get_latest_sightings(
        self, limit: int = 50, conn: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        """Most recent sightings across *all* cameras, newest first."""
        target_conn = conn or self.conn
        cur = target_conn.cursor()
        cur.execute(
            """
            SELECT s.*, c.name as camera_name, c.corridor
            FROM vehicle_sightings s
            CROSS JOIN cameras c ON s.camera_id = c.id
            ORDER BY s.sighting_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def search_sightings(self, query: str, limit: int = 50,
                         conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """
        Free-text sighting search for the dashboard, newest first, in the row shape of
        ``get_latest_sightings`` (``s.*`` + ``camera_name`` + ``corridor``).

        * numeric query ``"1234"`` / ``"#1234"``: sightings of camera 1234 (exact);
        * text equal (case-insensitively) to a vehicle class that exists (car, motorcycle, bus,
          truck, ...): sightings of that class;
        * any text: sightings of every camera whose name or corridor contains it (case-insensitive
          LIKE, ``%``/``_`` are literal; a space also matches the underscore of ``SSA_RING``); the
          camera ids are resolved first from the 100-odd-row ``cameras`` table.
        * empty query: the latest sightings.

        Matches from the different rules are merged (de-duplicated, ordered by
        ``timestamp_iso DESC, sighting_id DESC``). Each rule reads only the newest ``limit`` index
        entries of its camera / class, so cost is independent of table size for all
        common cases (a class so rare that it has no sighting among the newest rows is the one
        exception and degrades to a scan of ``idx_sightings_time``).
        """
        target_conn = conn or self.conn
        limit = min(max(0, int(limit)), 5000)
        text = (query or "").strip()
        if limit == 0:
            return []
        if not text:
            return self.get_latest_sightings(limit=limit, conn=target_conn)

        numeric = text.lstrip("#")
        candidates: Dict[int, Tuple[str, int]] = {}       # sighting_id -> (timestamp_iso, sighting_id)

        def collect(sql: str, params: Sequence[Any]) -> None:
            for ts, sid in target_conn.execute(sql, params).fetchall():
                candidates[sid] = (ts, sid)

        if numeric.isascii() and numeric.isdigit():
            n = int(numeric)
            if n > 2 ** 63 - 1:             # cannot be a camera id (and would overflow SQLite's INTEGER)
                return []
            collect("SELECT timestamp_iso, sighting_id FROM vehicle_sightings WHERE camera_id = ? "
                    "ORDER BY timestamp_iso DESC, sighting_id DESC LIMIT ?", (n, limit))
        else:
            lowered = text.lower()
            known = target_conn.execute(
                "SELECT vehicle_class FROM sighting_totals WHERE lower(vehicle_class) = ? GROUP BY vehicle_class",
                (lowered,)).fetchall()
            for (cls,) in known:
                collect("SELECT timestamp_iso, sighting_id FROM vehicle_sightings WHERE vehicle_class = ? "
                        "ORDER BY timestamp_iso DESC, sighting_id DESC LIMIT ?", (cls, limit))
            # corridors are spelled SSA_RING: a space in the query matches that underscore too
            cam_ids = [r[0] for r in target_conn.execute(
                "SELECT id FROM cameras WHERE name LIKE ? ESCAPE '\\' OR corridor LIKE ? ESCAPE '\\' "
                "OR replace(corridor, '_', ' ') LIKE ? ESCAPE '\\'",
                (_like_contains(text), _like_contains(text), _like_contains(text.replace("_", " ")))).fetchall()]
            for cid in cam_ids:
                collect("SELECT timestamp_iso, sighting_id FROM vehicle_sightings WHERE camera_id = ? "
                        "ORDER BY timestamp_iso DESC, sighting_id DESC LIMIT ?", (cid, limit))

        if not candidates:
            return []
        top = sorted(candidates.values(), reverse=True)[:limit]
        ids = [sid for _, sid in top]
        rows = target_conn.execute(
            "SELECT s.*, c.name AS camera_name, c.corridor "
            "FROM vehicle_sightings s JOIN cameras c ON s.camera_id = c.id "
            f"WHERE s.sighting_id IN ({','.join('?' * len(ids))})", ids).fetchall()
        by_id = {r["sighting_id"]: dict(r) for r in rows}
        return [by_id[sid] for sid in ids if sid in by_id]

    def prune_sightings(self, older_than_s: float, conn: Optional[sqlite3.Connection] = None,
                        batch: int = 2000, pause_s: float = 0.0,
                        should_stop: Optional[Any] = None) -> int:
        """
        Deletes sightings older than ``older_than_s`` seconds, oldest first, ``batch`` rows per
        transaction with ``pause_s`` between batches (the tracker keeps writing in between).
        Returns the number of sightings removed.

        * All-time totals (``sighting_totals``, ``sighting_counters``, ...) are unchanged: they count
          inserts. ``observation_windows`` (volume history) is not touched.
        * Free pages go back to the file via ``incremental_vacuum`` (see ``SightingMaintenance``).
        """
        target_conn = conn or self.conn
        cutoff = datetime.fromtimestamp(time.time() - older_than_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        batch = max(1, int(batch))
        removed = 0
        while not (should_stop is not None and should_stop()):
            with _transaction(target_conn) as cur:
                cur.execute(
                    "DELETE FROM vehicle_sightings WHERE sighting_id IN ("
                    " SELECT sighting_id FROM vehicle_sightings WHERE timestamp_iso < ?"
                    "  ORDER BY timestamp_iso LIMIT ?)", (cutoff, batch))
                n = max(0, cur.rowcount)
            removed += n
            if n < batch:
                break
            if pause_s > 0:
                time.sleep(pause_s)
        return removed

    def get_camera_vehicle_counts(
        self, window_s: Optional[float] = None, conn: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns real-time vehicle counts across all 94 CCTV cameras.
        Includes total sightings, unique tracked vehicles, class breakdowns,
        and last seen timestamps.

        All-time (``window_s`` None) counts come from the trigger-maintained totals tables (O(cameras),
        and they include sightings that retention has since deleted); a window reads the sightings
        of that window through ``idx_sightings_cam_time``. ``unique_vehicles`` is the number of
        distinct ``local_track_id`` seen at the camera in both cases.
        """
        target_conn = conn or self.conn
        cur = target_conn.cursor()

        params: List[Any] = []
        if window_s is not None and window_s > 0:
            from datetime import timedelta
            cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=window_s)
            cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            params.append(cutoff_iso)
            query = """
            SELECT
                c.id as camera_id,
                c.name as camera_name,
                c.corridor as corridor,
                c.latitude,
                c.longitude,
                c.status as camera_status,
                COALESCE(s.total_sightings, 0) as total_sightings,
                COALESCE(s.unique_vehicles, 0) as unique_vehicles,
                s.last_seen_iso,
                COALESCE(s.cars, 0) as cars,
                COALESCE(s.motorcycles, 0) as motorcycles,
                COALESCE(s.trucks, 0) as trucks,
                COALESCE(s.buses, 0) as buses
            FROM cameras c
            LEFT JOIN (
                SELECT
                    camera_id,
                    COUNT(*) as total_sightings,
                    COUNT(DISTINCT local_track_id) as unique_vehicles,
                    MAX(timestamp_iso) as last_seen_iso,
                    SUM(CASE WHEN vehicle_class = 'car' THEN 1 ELSE 0 END) as cars,
                    SUM(CASE WHEN vehicle_class = 'motorcycle' THEN 1 ELSE 0 END) as motorcycles,
                    SUM(CASE WHEN vehicle_class = 'truck' THEN 1 ELSE 0 END) as trucks,
                    SUM(CASE WHEN vehicle_class IN ('bus', 'angkot') THEN 1 ELSE 0 END) as buses
                FROM vehicle_sightings
                WHERE timestamp_iso >= ?
                GROUP BY camera_id
            ) s ON c.id = s.camera_id
            ORDER BY total_sightings DESC, c.id ASC
            """
        else:
            query = """
            SELECT
                c.id as camera_id,
                c.name as camera_name,
                c.corridor as corridor,
                c.latitude,
                c.longitude,
                c.status as camera_status,
                COALESCE(s.total_sightings, 0) as total_sightings,
                COALESCE(k.tracks, 0) as unique_vehicles,
                s.last_seen_iso,
                COALESCE(s.cars, 0) as cars,
                COALESCE(s.motorcycles, 0) as motorcycles,
                COALESCE(s.trucks, 0) as trucks,
                COALESCE(s.buses, 0) as buses
            FROM cameras c
            LEFT JOIN (
                SELECT
                    camera_id,
                    SUM(n) as total_sightings,
                    MAX(last_seen_iso) as last_seen_iso,
                    SUM(CASE WHEN vehicle_class = 'car' THEN n ELSE 0 END) as cars,
                    SUM(CASE WHEN vehicle_class = 'motorcycle' THEN n ELSE 0 END) as motorcycles,
                    SUM(CASE WHEN vehicle_class = 'truck' THEN n ELSE 0 END) as trucks,
                    SUM(CASE WHEN vehicle_class IN ('bus', 'angkot') THEN n ELSE 0 END) as buses
                FROM sighting_totals
                GROUP BY camera_id
            ) s ON c.id = s.camera_id
            LEFT JOIN camera_track_totals k ON k.camera_id = c.id
            ORDER BY total_sightings DESC, c.id ASC
            """
        cur.execute(query, params)
        rows = cur.fetchall()

        results = []
        for r in rows:
            results.append({
                "camera_id": r["camera_id"],
                "name": r["camera_name"],
                "corridor": r["corridor"],
                "latitude": r["latitude"],
                "longitude": r["longitude"],
                "status": r["camera_status"],
                "total_sightings": int(r["total_sightings"]),
                "unique_vehicles": int(r["unique_vehicles"]),
                "last_seen_iso": r["last_seen_iso"],
                "classes": {
                    "car": int(r["cars"]),
                    "motorcycle": int(r["motorcycles"]),
                    "truck": int(r["trucks"]),
                    "bus": int(r["buses"]),
                }
            })
        return results

    def insert_observation_window(self, window: Dict[str, Any],
                                  conn: Optional[sqlite3.Connection] = None) -> int:
        """Stores one observation window (see the observation_windows table)."""
        cols = ("camera_id", "start_ts", "end_ts", "start_iso", "observed_s", "vehicles", "dir_pos",
                "dir_neg", "cars", "motorcycles", "trucks", "buses", "line_counts", "passage_counts", "kind")
        with _transaction(conn or self.conn) as cur:
            cur.execute(
                f"INSERT INTO observation_windows ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
                tuple(window.get(c, 0) for c in cols),
            )
        return cur.lastrowid

    def get_camera_rates(self, window_s: float = 3600.0, now: Optional[float] = None,
                         conn: Optional[sqlite3.Connection] = None,
                         camera_ids: Optional[Iterable[int]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Vehicles per hour for every camera observed in the last ``window_s`` seconds:
        crossings / observed seconds * 3600, overall, per direction and per class, with the
        observed time and number of windows the estimate rests on.

        ``camera_ids`` restricts the computation to those cameras (an empty iterable gives ``{}``);
        asking for one camera reads only that camera's windows via ``idx_obs_cam_time``.
        """
        cutoff = (now if now is not None else time.time()) - window_s
        params: List[Any] = [cutoff]
        camera_filter = ""
        if camera_ids is not None:
            ids = sorted({int(c) for c in camera_ids})
            if not ids:
                return {}
            camera_filter = f"AND camera_id IN ({','.join('?' * len(ids))})"
            params.extend(ids)
        cur = (conn or self.conn).cursor()
        cur.execute(
            f"""
            SELECT camera_id, COUNT(*) AS windows, SUM(observed_s) AS observed_s,
                   SUM(vehicles) AS vehicles, SUM(dir_pos) AS dir_pos, SUM(dir_neg) AS dir_neg,
                   SUM(cars) AS cars, SUM(motorcycles) AS motorcycles, SUM(trucks) AS trucks,
                   SUM(buses) AS buses, SUM(line_counts) AS line_counts,
                   SUM(passage_counts) AS passage_counts, MAX(end_ts) AS last_observed_ts
            FROM observation_windows
            WHERE end_ts >= ? {camera_filter}
            GROUP BY camera_id
            """,
            params,
        )
        rates: Dict[int, Dict[str, Any]] = {}
        for r in cur.fetchall():
            obs = float(r["observed_s"] or 0.0)
            per_h = (3600.0 / obs) if obs > 0 else 0.0
            rates[r["camera_id"]] = {
                "observed_s": round(obs, 1),
                "windows": int(r["windows"]),
                "vehicles": int(r["vehicles"] or 0),
                "vph": round((r["vehicles"] or 0) * per_h, 1),
                "vph_dir_pos": round((r["dir_pos"] or 0) * per_h, 1),
                "vph_dir_neg": round((r["dir_neg"] or 0) * per_h, 1),
                "vph_classes": {
                    "car": round((r["cars"] or 0) * per_h, 1),
                    "motorcycle": round((r["motorcycles"] or 0) * per_h, 1),
                    "truck": round((r["trucks"] or 0) * per_h, 1),
                    "bus": round((r["buses"] or 0) * per_h, 1),
                },
                "counting_mode": "line" if (r["line_counts"] or 0) >= (r["passage_counts"] or 0) else "passage",
                # A rate from under a minute of observation is a rough estimate (a few vehicles).
                "low_sample": obs < 60.0,
                "last_observed_ts": r["last_observed_ts"],
            }
        return rates

    def get_camera_first_observed(self, camera_id: int,
                                  conn: Optional[sqlite3.Connection] = None) -> Optional[float]:
        """End time of the camera's oldest observation window (None if it was never observed)."""
        cur = (conn or self.conn).cursor()
        cur.execute("SELECT MIN(end_ts) FROM observation_windows WHERE camera_id = ?", (camera_id,))
        row = cur.fetchone()
        return None if row is None or row[0] is None else float(row[0])

    def get_traffic_stats(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Returns real-time aggregated traffic statistics across all 94 cameras.

        Every all-time figure is read from the trigger-maintained totals tables, O(cameras) rather
        than a scan of ``vehicle_sightings``. They count inserts, so ``total_sightings`` and friends
        include sightings that retention has deleted. ``total_unique_vehicles`` is the number of
        distinct ``local_track_id``.
        """
        target_conn = conn or self.conn
        cur = target_conn.cursor()

        cur.execute("SELECT count(*) FROM cameras WHERE status = 'online'")
        active_cameras = cur.fetchone()[0]

        cur.execute("SELECT COALESCE(SUM(n), 0) FROM sighting_totals")
        total_sightings = cur.fetchone()[0]
        counters = {row["key"]: row["value"] for row in cur.execute("SELECT key, value FROM sighting_counters")}
        unique_local_vehicles = counters.get("tracks", 0)

        # Sightings breakdown by class
        cur.execute("SELECT vehicle_class, SUM(n) AS count FROM sighting_totals GROUP BY vehicle_class")
        class_breakdown = {row["vehicle_class"]: row["count"] for row in cur.fetchall()}

        # Per-camera vehicle counts summary across all 94 cameras
        cam_counts_list = self.get_camera_vehicle_counts(conn=target_conn)
        camera_counts_map = {c["camera_id"]: c["total_sightings"] for c in cam_counts_list}
        cameras_with_detections = sum(1 for c in cam_counts_list if c["total_sightings"] > 0)
        top_cameras = [c for c in cam_counts_list if c["total_sightings"] > 0][:10]

        return {
            "active_cameras": active_cameras,
            "total_sightings": total_sightings,
            "total_vehicles_counted": total_sightings,
            "total_unique_vehicles": unique_local_vehicles,
            "class_breakdown": class_breakdown,
            "cameras_with_detections": cameras_with_detections,
            "camera_counts": camera_counts_map,
            "top_cameras": top_cameras,
        }

    def rebuild_totals(self) -> Dict[str, Any]:
        """
        Recomputes every trigger-maintained totals table from the sightings that are stored *right now*, in one transaction. Repair tool for counters that drifted (e.g. rows
        loaded with the triggers dropped). WARNING: after retention has deleted rows this resets the
        all-time totals to what is still stored; do not run it on a pruned database.
        """
        conn = self.conn
        t0 = time.perf_counter()
        with _transaction(conn) as cur:
            cur.execute("BEGIN IMMEDIATE")
            _rebuild_totals(cur)
        return {"seconds": round(time.perf_counter() - t0, 3)}
