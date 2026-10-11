"""
backend/database.py
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard.
SQLite WAL Persistence Layer (F-ARCH-01).

Implements:
- High-performance SQLite database in WAL (Write-Ahead Logging) mode.
- Thread-safe connection pooling and pragma optimizations.
- Schema DDL for cameras, topology_edges, vehicle_sightings, handoff_events, and active_targets.
- Automatic seeding of every BSW CCTV camera and topology edge from data/bogor_camera_topology.json.
- Core MTMC querying API: vehicle sightings insertion, chronological journey lookup,
  recent sighting retrieval, multi-filter search, active target tracking, and traffic statistics.
"""

import json
from contextlib import contextmanager
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple, Union

# Base path resolution
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DB_PATH = os.path.join(PROJECT_ROOT, "data", "traffic_db.sqlite")
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
    global_vehicle_id INTEGER,
    vehicle_class TEXT NOT NULL,
    confidence REAL NOT NULL,
    bbox_x1 INTEGER NOT NULL,
    bbox_y1 INTEGER NOT NULL,
    bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL,
    heading_deg REAL NOT NULL,
    speed_px_s REAL NOT NULL,
    embedding BLOB NOT NULL,
    color_hist BLOB NOT NULL,
    timestamp_iso TEXT NOT NULL,
    crop_thumbnail BLOB,
    FOREIGN KEY (camera_id) REFERENCES cameras(id)
);

CREATE TABLE IF NOT EXISTS handoff_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    global_vehicle_id INTEGER NOT NULL,
    from_camera_id INTEGER NOT NULL,
    to_camera_id INTEGER NOT NULL,
    confidence REAL NOT NULL,
    timestamp_iso TEXT NOT NULL,
    FOREIGN KEY (from_camera_id) REFERENCES cameras(id),
    FOREIGN KEY (to_camera_id) REFERENCES cameras(id)
);

CREATE TABLE IF NOT EXISTS active_targets (
    global_vehicle_id INTEGER PRIMARY KEY,
    current_camera_id INTEGER NOT NULL,
    last_sighting_id INTEGER NOT NULL,
    locked_at_iso TEXT NOT NULL,
    status TEXT DEFAULT 'tracking',
    FOREIGN KEY (current_camera_id) REFERENCES cameras(id),
    FOREIGN KEY (last_sighting_id) REFERENCES vehicle_sightings(sighting_id)
);

CREATE INDEX IF NOT EXISTS idx_sightings_cam_time ON vehicle_sightings(camera_id, timestamp_iso);
CREATE INDEX IF NOT EXISTS idx_sightings_global_id ON vehicle_sightings(global_vehicle_id);
CREATE INDEX IF NOT EXISTS idx_sightings_time ON vehicle_sightings(timestamp_iso);
CREATE INDEX IF NOT EXISTS idx_active_targets_cam ON active_targets(current_camera_id);

-- Narrow covering indexes for the dashboard aggregates (per-camera counts, class split, distinct
-- tracks). Sightings rows are wide, so aggregating from the table reads hundreds of MB; these
-- indexes hold only the columns the aggregates need.
CREATE INDEX IF NOT EXISTS idx_sightings_agg ON vehicle_sightings(camera_id, local_track_id, vehicle_class, timestamp_iso);
CREATE INDEX IF NOT EXISTS idx_sightings_local_track ON vehicle_sightings(local_track_id);

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
    nodes = data.get("nodes", [])
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


class Database:
    """
    Thread-safe SQLite Database manager with connection pooling and caching.
    """

    def __init__(self, db_path: Optional[str] = None, auto_seed: bool = True):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._thread_local = threading.local()
        self._lock = threading.Lock()
        self._all_conns = set()
        self._all_conns_lock = threading.Lock()

        # Initialize schema using a primary setup connection
        init_conn = get_db_connection(self.db_path)
        try:
            init_conn.executescript(SCHEMA_DDL)
            init_conn.commit()
            if auto_seed:
                seed_cameras_from_topology(init_conn)
        finally:
            if self.db_path == ":memory:":
                # Keep reference to in-memory connection so it survives across calls
                self._shared_conn = init_conn
                with self._all_conns_lock:
                    self._all_conns.add(init_conn)
            else:
                self._shared_conn = None
                init_conn.close()

    @property
    def conn(self) -> sqlite3.Connection:
        """Returns the thread-local connection for the current thread."""
        if self._shared_conn is not None:
            return self._shared_conn

        if not hasattr(self._thread_local, "conn") or self._thread_local.conn is None:
            c = get_db_connection(self.db_path)
            self._thread_local.conn = c
            with self._all_conns_lock:
                self._all_conns.add(c)
        return self._thread_local.conn

    def close(self):
        """Closes all thread-local or shared connections across all threads."""
        with self._all_conns_lock:
            for c in list(self._all_conns):
                try:
                    c.close()
                except Exception:
                    pass
            self._all_conns.clear()

        self._shared_conn = None
        if hasattr(self._thread_local, "conn"):
            self._thread_local.conn = None

    def insert_vehicle_sighting(
        self,
        camera_id: int,
        local_track_id: int,
        global_vehicle_id: Optional[int],
        vehicle_class: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        heading_deg: float,
        speed_px_s: float,
        embedding: bytes,
        color_hist: bytes,
        timestamp_iso: str,
        crop_thumbnail: Optional[bytes] = None,
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
                    camera_id, local_track_id, global_vehicle_id, vehicle_class,
                    confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    heading_deg, speed_px_s, embedding, color_hist, timestamp_iso, crop_thumbnail
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    camera_id,
                    local_track_id,
                    global_vehicle_id,
                    vehicle_class,
                    float(confidence),
                    int(bbox[0]),
                    int(bbox[1]),
                    int(bbox[2]),
                    int(bbox[3]),
                    float(heading_deg),
                    float(speed_px_s),
                    embedding,
                    color_hist,
                    timestamp_iso,
                    crop_thumbnail,
                ),
            )
        return cur.lastrowid

    def record_handoff_event(
        self,
        global_vehicle_id: int,
        from_camera_id: int,
        to_camera_id: int,
        confidence: float,
        timestamp_iso: str,
        conn: Optional[sqlite3.Connection] = None,
    ) -> int:
        """
        Records a multi-camera predictive handoff event.
        Returns event_id.
        """
        target_conn = conn or self.conn
        with _transaction(target_conn) as cur:
            cur.execute(
                """
                INSERT INTO handoff_events (
                    global_vehicle_id, from_camera_id, to_camera_id, confidence, timestamp_iso
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(global_vehicle_id),
                    int(from_camera_id),
                    int(to_camera_id),
                    float(confidence),
                    timestamp_iso,
                ),
            )
        return cur.lastrowid

    def get_vehicle_journey(
        self, global_vehicle_id: int, conn: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves all chronological sightings for global_vehicle_id joined with camera metadata.
        """
        target_conn = conn or self.conn
        cur = target_conn.cursor()
        cur.execute(
            """
            SELECT s.*, c.name as camera_name, c.latitude, c.longitude, c.corridor, c.stream_url
            FROM vehicle_sightings s
            JOIN cameras c ON s.camera_id = c.id
            WHERE s.global_vehicle_id = ?
            ORDER BY s.timestamp_iso ASC
            """,
            (global_vehicle_id,),
        )
        return [dict(row) for row in cur.fetchall()]

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
        global_vehicle_id: Optional[int] = None,
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
        if global_vehicle_id is not None:
            query.append("AND s.global_vehicle_id = ?")
            params.append(global_vehicle_id)
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

    def get_max_global_vehicle_id(self, conn: Optional[sqlite3.Connection] = None) -> int:
        """Highest global_vehicle_id ever assigned (0 if none); used to seed the id allocator."""
        target_conn = conn or self.conn
        row = target_conn.execute("SELECT COALESCE(MAX(global_vehicle_id), 0) FROM vehicle_sightings").fetchone()
        return int(row[0])

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
            JOIN cameras c ON s.camera_id = c.id
            ORDER BY s.sighting_id DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]

    def prune_sightings(self, older_than_s: float, conn: Optional[sqlite3.Connection] = None) -> int:
        """Deletes sightings older than ``older_than_s`` seconds. Returns rows removed."""
        target_conn = conn or self.conn
        cutoff = datetime.fromtimestamp(time.time() - older_than_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with _transaction(target_conn) as cur:
            cur.execute("DELETE FROM vehicle_sightings WHERE timestamp_iso < ?", (cutoff,))
        return cur.rowcount

    def expire_embeddings(self, older_than_s: float, batch: int = 2000, pause_s: float = 0.0,
                          should_stop: Optional[Any] = None,
                          conn: Optional[sqlite3.Connection] = None) -> int:
        """
        Blanks the appearance blobs (``embedding``, ``color_hist``; ~1.5 KB per row, 99% of the
        table) of sightings older than ``older_than_s``. Re-identification matches against
        in-memory state only, so the stored blobs are never read back; the rows (class, camera,
        time, journey) stay. Rows are processed in id order in small transactions, resuming from a
        persisted watermark, so the tracker's writes are never blocked for long and old rows are
        never rescanned. Returns the number of rows blanked.
        """
        target_conn = conn or self.conn
        cutoff = datetime.fromtimestamp(time.time() - older_than_s, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        row = target_conn.execute(
            "SELECT value FROM maintenance_state WHERE key = 'embeddings_expired_upto'").fetchone()
        low = int(row[0]) if row else 0
        # Stop at the first row that is still young: ids normally rise with time, but never assume
        # it (an old-timestamped row inserted late must not drag younger rows into the range).
        first_young = target_conn.execute(
            "SELECT min(sighting_id) FROM vehicle_sightings WHERE sighting_id > ? AND timestamp_iso >= ?",
            (low, cutoff)).fetchone()[0]
        if first_young is not None:
            upper = first_young - 1
        else:
            upper = target_conn.execute("SELECT max(sighting_id) FROM vehicle_sightings").fetchone()[0] or 0
        if upper <= low:
            return 0

        blanked = 0
        while low < upper and not (should_stop is not None and should_stop()):
            high = min(low + max(1, int(batch)), upper)
            with _transaction(target_conn) as cur:
                cur.execute(
                    "UPDATE vehicle_sightings SET embedding = X'', color_hist = X'' "
                    "WHERE sighting_id > ? AND sighting_id <= ? AND timestamp_iso < ? "
                    "AND (length(embedding) > 0 OR length(color_hist) > 0)",
                    (low, high, cutoff),
                )
                blanked += max(0, cur.rowcount)
                cur.execute(
                    "INSERT INTO maintenance_state (key, value) VALUES ('embeddings_expired_upto', ?) "
                    "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                    (high,),
                )
            low = high
            if pause_s > 0:
                time.sleep(pause_s)
        return blanked

    def update_active_target(
        self,
        global_vehicle_id: int,
        current_camera_id: int,
        last_sighting_id: int,
        status: str = "tracking",
        locked_at_iso: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ):
        """
        Inserts or updates the active target state in active_targets table.
        """
        target_conn = conn or self.conn
        locked_time = locked_at_iso or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        with _transaction(target_conn) as cur:
            cur.execute(
                """
                INSERT INTO active_targets (
                    global_vehicle_id, current_camera_id, last_sighting_id, locked_at_iso, status
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(global_vehicle_id) DO UPDATE SET
                    current_camera_id=excluded.current_camera_id,
                    last_sighting_id=excluded.last_sighting_id,
                    status=excluded.status
                """,
                (global_vehicle_id, current_camera_id, last_sighting_id, locked_time, status),
            )

    def get_active_target(
        self, global_vehicle_id: int, conn: Optional[sqlite3.Connection] = None
    ) -> Optional[Dict[str, Any]]:
        """Returns the current state of an active tracked target."""
        target_conn = conn or self.conn
        cur = target_conn.cursor()
        cur.execute(
            """
            SELECT a.*, c.name as camera_name, c.stream_url, s.bbox_x1, s.bbox_y1, s.bbox_x2, s.bbox_y2, s.heading_deg
            FROM active_targets a
            JOIN cameras c ON a.current_camera_id = c.id
            JOIN vehicle_sightings s ON a.last_sighting_id = s.sighting_id
            WHERE a.global_vehicle_id = ?
            """,
            (global_vehicle_id,),
        )
        row = cur.fetchone()
        return dict(row) if row else None

    def is_active_target(self, global_vehicle_id: int, conn: Optional[sqlite3.Connection] = None) -> bool:
        """True if the vehicle is currently being followed (status 'tracking')."""
        cur = (conn or self.conn).cursor()
        cur.execute(
            "SELECT 1 FROM active_targets WHERE global_vehicle_id = ? AND status = 'tracking' LIMIT 1",
            (global_vehicle_id,),
        )
        return cur.fetchone() is not None

    def list_active_targets(self, conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """Returns all currently active tracked targets."""
        target_conn = conn or self.conn
        cur = target_conn.cursor()
        cur.execute(
            """
            SELECT a.*, c.name as camera_name, c.stream_url
            FROM active_targets a
            JOIN cameras c ON a.current_camera_id = c.id
            WHERE a.status = 'tracking'
            ORDER BY a.locked_at_iso DESC
            """
        )
        return [dict(r) for r in cur.fetchall()]

    def get_camera_vehicle_counts(
        self, window_s: Optional[float] = None, conn: Optional[sqlite3.Connection] = None
    ) -> List[Dict[str, Any]]:
        """
        Returns real-time vehicle counts across all 94 CCTV cameras.
        Includes total sightings, unique tracked vehicles, class breakdowns,
        and last seen timestamps.
        """
        target_conn = conn or self.conn
        cur = target_conn.cursor()

        params: List[Any] = []
        where_clause = ""
        if window_s is not None and window_s > 0:
            from datetime import timedelta
            cutoff_dt = datetime.now(timezone.utc) - timedelta(seconds=window_s)
            cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            where_clause = "WHERE timestamp_iso >= ?"
            params.append(cutoff_iso)

        query = f"""
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
            {where_clause}
            GROUP BY camera_id
        ) s ON c.id = s.camera_id
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
                         conn: Optional[sqlite3.Connection] = None) -> Dict[int, Dict[str, Any]]:
        """
        Vehicles per hour for every camera observed in the last ``window_s`` seconds:
        crossings / observed seconds * 3600, overall, per direction and per class, with the
        observed time and number of windows the estimate rests on.
        """
        cutoff = (now if now is not None else time.time()) - window_s
        cur = (conn or self.conn).cursor()
        cur.execute(
            """
            SELECT camera_id, COUNT(*) AS windows, SUM(observed_s) AS observed_s,
                   SUM(vehicles) AS vehicles, SUM(dir_pos) AS dir_pos, SUM(dir_neg) AS dir_neg,
                   SUM(cars) AS cars, SUM(motorcycles) AS motorcycles, SUM(trucks) AS trucks,
                   SUM(buses) AS buses, SUM(line_counts) AS line_counts,
                   SUM(passage_counts) AS passage_counts, MAX(end_ts) AS last_observed_ts
            FROM observation_windows
            WHERE end_ts >= ?
            GROUP BY camera_id
            """,
            (cutoff,),
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

    def get_camera_flow_series(self, camera_id: int, minutes: float = 60.0, bucket_minutes: float = 5.0,
                               now: Optional[float] = None,
                               conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
        """
        Counting-line flow of one camera over the last ``minutes`` in epoch-aligned buckets of
        ``bucket_minutes``: crossings / observed seconds * 3600 per bucket (None when the camera
        was not observed in that bucket). Buckets without observation are omitted.
        """
        bucket_s = max(1.0, float(bucket_minutes) * 60.0)
        cutoff = (now if now is not None else time.time()) - float(minutes) * 60.0
        cur = (conn or self.conn).cursor()
        cur.execute(
            """
            SELECT CAST(end_ts / ? AS INTEGER) AS bucket, COUNT(*) AS windows, SUM(observed_s) AS observed_s,
                   SUM(vehicles) AS vehicles, SUM(cars) AS cars, SUM(motorcycles) AS motorcycles,
                   SUM(trucks) AS trucks, SUM(buses) AS buses
            FROM observation_windows
            WHERE camera_id = ? AND end_ts >= ?
            GROUP BY bucket ORDER BY bucket
            """,
            (bucket_s, camera_id, cutoff),
        )
        series = []
        for r in cur.fetchall():
            obs = float(r["observed_s"] or 0.0)
            per_h = (3600.0 / obs) if obs > 0 else 0.0
            series.append({
                "t": r["bucket"] * bucket_s,
                "t_end": (r["bucket"] + 1) * bucket_s,
                "windows": int(r["windows"]),
                "observed_s": round(obs, 1),
                "vehicles": int(r["vehicles"] or 0),
                "vph": round((r["vehicles"] or 0) * per_h, 1) if obs > 0 else None,
                "low_sample": obs < 60.0,
            })
        return series

    def get_traffic_stats(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Returns real-time aggregated traffic statistics across all 94 cameras.
        """
        target_conn = conn or self.conn
        cur = target_conn.cursor()

        cur.execute("SELECT count(*) FROM cameras WHERE status = 'online'")
        active_cameras = cur.fetchone()[0]

        # One DISTINCT per query: with two in one statement SQLite falls back to scanning the wide
        # table instead of each column's narrow index.
        cur.execute("SELECT count(*) FROM vehicle_sightings")
        total_sightings = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT global_vehicle_id) FROM vehicle_sightings")
        unique_global_vehicles = cur.fetchone()[0]
        cur.execute("SELECT count(DISTINCT local_track_id) FROM vehicle_sightings")
        unique_local_vehicles = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM active_targets WHERE status = 'tracking'")
        active_tracking_count = cur.fetchone()[0]

        cur.execute("SELECT count(*) FROM handoff_events")
        total_handoffs = cur.fetchone()[0]

        # Sightings breakdown by class
        cur.execute(
            """
            SELECT vehicle_class, count(*) as count
            FROM vehicle_sightings
            GROUP BY vehicle_class
            """
        )
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
            "unique_global_vehicles": unique_global_vehicles,
            "total_unique_vehicles": unique_local_vehicles or unique_global_vehicles,
            "active_targets_count": active_tracking_count,
            "total_handoffs": total_handoffs,
            "class_breakdown": class_breakdown,
            "cameras_with_detections": cameras_with_detections,
            "camera_counts": camera_counts_map,
            "top_cameras": top_cameras,
        }


# Singleton database instance for convenient global access
_GLOBAL_DB: Optional[Database] = None
_INIT_LOCK = threading.Lock()


def init_db(db_path: Optional[str] = None, auto_seed: bool = True) -> Database:
    """Initializes and returns the global Database instance."""
    global _GLOBAL_DB
    with _INIT_LOCK:
        if _GLOBAL_DB is None or (db_path and _GLOBAL_DB.db_path != db_path):
            _GLOBAL_DB = Database(db_path, auto_seed=auto_seed)
        return _GLOBAL_DB


# Module-level convenience functions delegating to default Database
def insert_vehicle_sighting(
    camera_id: int,
    local_track_id: int,
    global_vehicle_id: Optional[int],
    vehicle_class: str,
    confidence: float,
    bbox: Tuple[int, int, int, int],
    heading_deg: float,
    speed_px_s: float,
    embedding: bytes,
    color_hist: bytes,
    timestamp_iso: str,
    crop_thumbnail: Optional[bytes] = None,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    return init_db().insert_vehicle_sighting(
        camera_id,
        local_track_id,
        global_vehicle_id,
        vehicle_class,
        confidence,
        bbox,
        heading_deg,
        speed_px_s,
        embedding,
        color_hist,
        timestamp_iso,
        crop_thumbnail,
        conn=conn,
    )


def record_handoff_event(
    global_vehicle_id: int,
    from_camera_id: int,
    to_camera_id: int,
    confidence: float,
    timestamp_iso: str,
    conn: Optional[sqlite3.Connection] = None,
) -> int:
    return init_db().record_handoff_event(
        global_vehicle_id, from_camera_id, to_camera_id, confidence, timestamp_iso, conn=conn
    )


def get_vehicle_journey(
    global_vehicle_id: int, conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    return init_db().get_vehicle_journey(global_vehicle_id, conn=conn)


def get_recent_sightings(
    camera_id: int,
    window_s: float = 60.0,
    limit: int = 100,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    return init_db().get_recent_sightings(camera_id, window_s, limit, conn=conn)


def search_vehicles(
    vehicle_class: Optional[str] = None,
    camera_id: Optional[int] = None,
    global_vehicle_id: Optional[int] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 50,
    conn: Optional[sqlite3.Connection] = None,
) -> List[Dict[str, Any]]:
    return init_db().search_vehicles(
        vehicle_class, camera_id, global_vehicle_id, start_time, end_time, limit, conn=conn
    )


def update_active_target(
    global_vehicle_id: int,
    current_camera_id: int,
    last_sighting_id: int,
    status: str = "tracking",
    locked_at_iso: Optional[str] = None,
    conn: Optional[sqlite3.Connection] = None,
):
    return init_db().update_active_target(
        global_vehicle_id, current_camera_id, last_sighting_id, status, locked_at_iso, conn=conn
    )


def get_active_target(
    global_vehicle_id: int, conn: Optional[sqlite3.Connection] = None
) -> Optional[Dict[str, Any]]:
    return init_db().get_active_target(global_vehicle_id, conn=conn)


def list_active_targets(conn: Optional[sqlite3.Connection] = None) -> List[Dict[str, Any]]:
    return init_db().list_active_targets(conn=conn)


def get_traffic_stats(conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
    return init_db().get_traffic_stats(conn=conn)


def get_camera_vehicle_counts(
    window_s: Optional[float] = None, conn: Optional[sqlite3.Connection] = None
) -> List[Dict[str, Any]]:
    return init_db().get_camera_vehicle_counts(window_s=window_s, conn=conn)

