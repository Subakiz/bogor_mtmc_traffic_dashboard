"""
backend/database.py
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard.
SQLite WAL Persistence Layer (F-ARCH-01).

Implements:
- High-performance SQLite database in WAL (Write-Ahead Logging) mode.
- Thread-safe connection pooling and pragma optimizations.
- Schema DDL for cameras, topology_edges, vehicle_sightings, handoff_events, and active_targets.
- Automatic seeding of all 94 BSW CCTV cameras and 220 topology edges from data/bogor_camera_topology.json.
- Core MTMC querying API: vehicle sightings insertion, chronological journey lookup,
  recent sighting retrieval, multi-filter search, active target tracking, and traffic statistics.
"""

import json
import os
import sqlite3
import threading
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
"""


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
    Automatically seeds cameras and topology edges from data/bogor_camera_topology.json
    if not already seeded. Returns number of cameras seeded.
    """
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM cameras;")
    count = cur.fetchone()[0]
    if count >= 94:
        return count

    path = topology_path or DEFAULT_TOPOLOGY_PATH
    if not os.path.exists(path):
        return count

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = data.get("nodes", [])
    edges = data.get("edges", [])

    cur.execute("BEGIN TRANSACTION;")
    try:
        # Seed Cameras
        for node in nodes:
            cur.execute(
                """
                INSERT OR REPLACE INTO cameras (
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

        # Seed Topology Edges
        for edge in edges:
            source_id = edge.get("source_id")
            target_id = edge.get("target_id")
            if source_id is None or target_id is None:
                continue

            dist = float(edge.get("distance_m", edge.get("road_distance_m", 0.0)))
            speed = float(edge.get("typical_speed_kmh", 30.0))
            t_win = edge.get("travel_time_window_s", [edge.get("t_min_s", 10.0), edge.get("t_max_s", 60.0)])
            t_min = float(t_win[0])
            t_max = float(t_win[1])
            corridor = edge.get("corridor", "")

            cur.execute(
                """
                INSERT OR IGNORE INTO topology_edges (
                    source_id, target_id, distance_m, typical_speed_kmh, t_min_s, t_max_s, corridor
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (source_id, target_id, dist, speed, t_min, t_max, corridor),
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise

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
        cur = target_conn.cursor()
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
        target_conn.commit()
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
        cur = target_conn.cursor()
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
        target_conn.commit()
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
        cur = target_conn.cursor()
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
        target_conn.commit()

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

    def get_traffic_stats(self, conn: Optional[sqlite3.Connection] = None) -> Dict[str, Any]:
        """
        Returns real-time aggregated traffic statistics.
        """
        target_conn = conn or self.conn
        cur = target_conn.cursor()

        cur.execute("SELECT count(*) FROM cameras WHERE status = 'online'")
        active_cameras = cur.fetchone()[0]

        cur.execute("SELECT count(*), count(DISTINCT global_vehicle_id) FROM vehicle_sightings")
        sighting_row = cur.fetchone()
        total_sightings = sighting_row[0]
        unique_global_vehicles = sighting_row[1]

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

        return {
            "active_cameras": active_cameras,
            "total_sightings": total_sightings,
            "unique_global_vehicles": unique_global_vehicles,
            "active_targets_count": active_tracking_count,
            "total_handoffs": total_handoffs,
            "class_breakdown": class_breakdown,
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
