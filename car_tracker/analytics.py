"""
Per-Camera Traffic Volume & Time-Series Analytics Engine for CCTV YOLO ANE.

Provides:
- Real-time vehicle kinematics, displacement, speed proxy, and flow vector tracking.
- Per-camera vehicle volume and class breakdown (cars, motorcycles, trucks, buses).
- Traffic occupancy density (ratio of frame covered by vehicles) and congestion index.
- Window records of any length (rolling visits close a window of 2-62 s); ``query_timeseries``
  can aggregate them into fixed buckets weighted by window length.
- Continuous cyclic temporal features (sin_hour, cos_hour, sin_dow, cos_dow, is_weekend), computed in
  LOCAL time (Asia/Jakarta, WIB, UTC+7, no DST) because the cameras are in Kota Bogor.
- Persistent SQLite storage in WAL mode (data/traffic_timeseries.db).
- Legacy exporters (CSV, JSON, numpy/LSTM sequences). The analysis-ready, regular per-camera dataset
  (observed-time coverage, crossings prorated over buckets, wide pivots, raw exports) lives in
  ``backend/dataset.py``.

Time conventions of ``camera_traffic_timeseries``
-------------------------------------------------
``timestamp``            Unix epoch seconds of the window END.  The window is [timestamp - window_seconds, timestamp].
``timestamp_iso``        the same instant as ISO-8601 in UTC (``+00:00``); kept UTC so the column stays sortable.
``timestamp_local_iso``  the same instant in WIB with its ``+07:00`` offset.
``hour``/``minute``/``day_of_week``/``is_weekend``/``sin_*``/``cos_*``  LOCAL (WIB) wall-clock features of the
                         window end (day_of_week 0 = Monday).  Rows written before the WIB change held UTC
                         values; the migration in ``TrafficAnalyticsDatabase`` rewrites them once.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np

try:  # IANA zone when the system has tzdata, otherwise the same fixed offset (WIB has no DST)
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Jakarta")
except Exception:  # noqa: BLE001  (ZoneInfoNotFoundError, ImportError)
    LOCAL_TZ = timezone(timedelta(hours=7), "WIB")
LOCAL_TZ_NAME = "Asia/Jakarta"


def local_iso(ts: float) -> str:
    """ISO-8601 of an epoch instant in local (WIB) time, with its +07:00 offset."""
    return datetime.fromtimestamp(ts, LOCAL_TZ).isoformat()

# Default storage location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "traffic_timeseries.db")


# ── Traffic-state model ────────────────────────────────────────────────────────
# Congestion is judged only on vehicles that are part of the traffic stream: vehicles that move,
# and vehicles standing still where moving traffic drives (queued). A vehicle standing still
# anywhere else is parked and left out of density, stopped-share and speed; otherwise a lot full of
# parked cars reads as a permanent jam (camera 167). Where traffic drives is learned per camera
# (``_LaneMap``): judging "parked" by "has not moved yet" alone called most of a stop-and-go queue
# parked and rated a jammed junction "Smooth" (camera 168: 11 of 16 vehicles "parked").
LOOKBACK_S = 3.0          # window for a track's recent speed
MIN_SPEED_SPAN_S = 1.5    # shortest history a recent speed is computed from
GRID_W, GRID_H = 48, 27   # coverage grid: union of boxes, so overlapping boxes aren't double counted
DENSITY_FULL = 0.30       # grid coverage that counts as fully dense
STOPPED_FULL = 0.50       # share of traffic standing still that counts as a full queue
SLOW_ABOVE = 0.55         # median speed at or above this fraction of free flow is not slow at all...
SLOW_FULL_BELOW = 0.15    # ...and at or below this fraction it is fully slow
MIN_FREE_FLOW_BPS = 0.25  # sanity floor for a learned free-flow speed
MIN_TRAFFIC_FOR_FULL_CONFIDENCE = 6   # measurable vehicles for the index to count in full (quadratic ramp)
LIVE_SPAN_S = 20.0        # the live index is the mean of the per-frame index over this many seconds
LIVE_MAX_AGE_S = 15.0     # a snapshot older than this is no longer "live" (the camera is not being observed)
RECENT_SUMMARY_WINDOWS = 60   # latest windows behind get_camera_summary()'s "recent" averages
MIN_REF_SAMPLES = 30      # speed samples a window needs before it updates the free-flow reference
REF_PERCENTILE = 85       # free-flow speed = this percentile of a camera's observed speeds
# The reference rises quickly when a window shows faster traffic and sinks only slowly: following
# every window (0.9/0.1) let a camera that is congested most of the day learn its jam speed as
# "free flow", after which the jam rated "Smooth".
FREE_FLOW_RISE = 0.20     # weight of a faster window's percentile
FREE_FLOW_SINK = 0.01     # weight of a slower one

# Lane map: a decaying heat map (on the coverage grid) of the ground footprints (lower half of the
# box) of vehicles seen moving.
LANE_HALF_LIFE = 5000     # moving-footprint frames over which old lane evidence halves
LANE_MIN_OBS = 300.0      # (decayed) moving footprints before a camera's lane map is trusted
LANE_MIN_SCORE = 0.35     # mean normalised lane heat under a still vehicle's footprint for "queued"
LANE_CELL_MIN = 0.15      # normalised heat at which a grid cell counts as lane (road) area
LANE_DENSITY_FULL = 0.45  # share of the lane area covered by vehicle footprints that counts as fully dense
QUEUE_AFTER_S = 2.0       # a still vehicle in a lane is "queued" once seen this long (younger: undetermined)


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class TrackKinematics:
    """Stateful kinematic memory for an active vehicle track."""
    track_id: int
    class_id: int
    class_name: str
    first_seen: float
    last_seen: float
    last_cx: float
    last_cy: float
    last_w: float
    last_h: float
    speed_px_s: float = 0.0
    dx_px_s: float = 0.0
    dy_px_s: float = 0.0
    total_displacement: float = 0.0
    frame_count: int = 1
    is_stopped: bool = False
    # Congestion state. Distances are in "body sizes" (the vehicle's own box size), so a speed or
    # displacement means the same thing near and far from the camera and on every camera.
    first_cx: float = 0.0
    first_cy: float = 0.0
    size: float = 0.0                 # smoothed box size (max of width/height) in px
    max_extent: float = 0.0           # farthest the track has been from where it was first seen (px)
    is_static: bool = False           # never moved since first seen -> parked, not traffic
    recent_bps: Optional[float] = None  # body sizes / s over the last LOOKBACK_S (None: too short)
    history: deque = field(default_factory=lambda: deque(maxlen=64))   # (t, cx, cy)


@dataclass
class TimeWindowMetrics:
    """
    Standard aggregated time-bucket observation for a single camera.
    Directly ingestible into LSTM, GRU, ARIMA, or GNN traffic forecasting models.
    """
    camera_id: int
    camera_name: str
    timestamp: float                    # Unix epoch seconds (window END)
    timestamp_iso: str                  # ISO 8601 of the window end, UTC
    window_seconds: int                 # Observed length of the window in seconds (2-62 for rolling visits)

    # Volume (unique vehicle counts entered/passed within window)
    total_volume: int
    car_volume: int
    motorcycle_volume: int
    truck_volume: int
    bus_volume: int

    # Instantaneous density & occupancy
    avg_active_vehicles: float          # Mean active vehicle count per frame
    peak_active_vehicles: int           # Maximum simultaneous vehicles in a frame
    occupancy_ratio: float              # Frame area bounding-box coverage ratio [0.0 - 1.0]

    # Kinematics & congestion metrics
    avg_speed_px_s: float               # Mean track velocity proxy (pixels/second)
    stopped_vehicles: int               # Count of unique stopped / idling vehicles
    congestion_index: float             # Calibrated traffic congestion score [0.0 - 1.0]
    avg_flow_dx: float                  # Net directional flow velocity X
    avg_flow_dy: float                  # Net directional flow velocity Y

    # Normalized throughput rates
    flow_rate_vpm: float                # Equivalent volume rate (vehicles per minute)
    flow_rate_vph: float                # Equivalent hourly flow rate (vehicles per hour)

    # Temporal features for LSTM / time-series ML -- LOCAL (WIB, UTC+7) wall clock of the window end
    hour: int                           # Hour of day [0 - 23]
    minute: int                         # Minute of hour [0 - 59]
    day_of_week: int                    # Day of week [0=Monday, 6=Sunday]
    is_weekend: int                     # 1 if Saturday or Sunday, else 0
    sin_hour: float                     # sin(2*pi*(hour + minute/60)/24)
    cos_hour: float                     # cos(2*pi*(hour + minute/60)/24)
    sin_dow: float                      # sin(2*pi*dow/7)
    cos_dow: float                      # cos(2*pi*dow/7)

    # What the volume columns count: "crossings" = vehicles that crossed the camera's counting line
    # (the reliable measure), "tracks" = distinct track ids (older rows; inflated by track
    # fragmentation, so never mix the two in one series).
    volume_basis: str = "tracks"

    # Added later; all optional so older callers/rows stay valid (NULL = not measured).
    timestamp_local_iso: str = ""                   # the window end in WIB with its +07:00 offset
    median_speed_bps: Optional[float] = None        # median speed of moving-stream vehicles, body sizes/s
    mean_speed_bps: Optional[float] = None          # mean of the same samples
    queue_share: Optional[float] = None             # share of those samples below the "stopped" speed
    avg_traffic_vehicles: Optional[float] = None    # mean moving-stream vehicles measurable per frame
    lane_occupancy: Optional[float] = None          # mean share of the learned lane area covered by traffic

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── SQLite Persistence Layer ──────────────────────────────────────────────────

LOCAL_UTC_OFFSET_S = int(datetime(2026, 1, 1, tzinfo=LOCAL_TZ).utcoffset().total_seconds())   # 25200: WIB has no DST

_FEATURE_COLUMNS = ("hour", "minute", "day_of_week", "is_weekend", "sin_hour", "cos_hour", "sin_dow", "cos_dow")

# Columns added after the first release: (name, SQL type). NULL = not measured / not recorded.
_ADDED_COLUMNS = (
    ("volume_basis", "TEXT NOT NULL DEFAULT 'tracks'"),
    ("timestamp_local_iso", "TEXT"),
    ("median_speed_bps", "REAL"),
    ("mean_speed_bps", "REAL"),
    ("queue_share", "REAL"),
    ("avg_traffic_vehicles", "REAL"),
    ("lane_occupancy", "REAL"),
)

_INSERT_COLUMNS = (
    "camera_id", "camera_name", "timestamp", "timestamp_iso", "window_seconds",
    "total_volume", "car_volume", "motorcycle_volume", "truck_volume", "bus_volume",
    "avg_active_vehicles", "peak_active_vehicles", "occupancy_ratio",
    "avg_speed_px_s", "stopped_vehicles", "congestion_index",
    "avg_flow_dx", "avg_flow_dy", "flow_rate_vpm", "flow_rate_vph",
    "hour", "minute", "day_of_week", "is_weekend",
    "sin_hour", "cos_hour", "sin_dow", "cos_dow", "volume_basis",
    "timestamp_local_iso", "median_speed_bps", "mean_speed_bps", "queue_share", "avg_traffic_vehicles",
    "lane_occupancy",
)
_INSERT_SQL = (f"INSERT INTO camera_traffic_timeseries ({', '.join(_INSERT_COLUMNS)}) "
               f"VALUES ({', '.join('?' * len(_INSERT_COLUMNS))});")


def _wmean(col: str, where: str = "") -> str:
    """SQL: mean of ``col`` weighted by window length over the windows where it is known (NULL if none)."""
    cond = f"{col} IS NOT NULL" + (f" AND {where}" if where else "")
    return (f"SUM(CASE WHEN {cond} THEN {col} * window_seconds END) * 1.0 / "
            f"SUM(CASE WHEN {cond} THEN window_seconds END)")


class TrafficAnalyticsDatabase:
    """
    Thread-safe SQLite storage for per-camera traffic time-series data.

    One persistent WAL writer connection (every write goes through it under ``_lock``) and one
    persistent read-only connection (under ``_read_lock``). Opening a connection per insert cost
    ~0.5 ms and forced a WAL checkpoint every time the last connection closed. Connections are
    created lazily, so the object keeps working after :meth:`close`.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()          # the writer connection
        self._read_lock = threading.Lock()     # the reader connection
        self._wconn: Optional[sqlite3.Connection] = None
        self._rconn: Optional[sqlite3.Connection] = None
        self._init_db()

    # ── connections ──────────────────────────────────────────────────────────

    def _connect(self, readonly: bool = False) -> sqlite3.Connection:
        # check_same_thread=False: the connection is shared between threads but only ever used under
        # its lock, one thread at a time.
        conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 10000;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        if readonly:
            conn.execute("PRAGMA query_only = ON;")
        else:
            conn.execute("PRAGMA journal_mode = WAL;")
            conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _writer(self) -> sqlite3.Connection:
        """The writer connection; the caller holds ``_lock``."""
        if self._wconn is None:
            self._wconn = self._connect()
        return self._wconn

    def _drop_writer(self) -> None:
        conn, self._wconn = self._wconn, None
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass

    def _read(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._read_lock:
            if self._rconn is None:
                self._rconn = self._connect(readonly=True)
            try:
                return self._rconn.execute(sql, params).fetchall()
            except sqlite3.Error:
                conn, self._rconn = self._rconn, None      # reconnect on the next call
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
                raise

    def close(self) -> None:
        """Closes both connections (the last close checkpoints the WAL). Safe to call twice."""
        with self._lock:
            self._drop_writer()
        with self._read_lock:
            conn, self._rconn = self._rconn, None
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass

    # ── schema ───────────────────────────────────────────────────────────────

    def _init_db(self):
        with self._lock:
            conn = self._writer()
            with conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS camera_traffic_timeseries (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        camera_id INTEGER NOT NULL,
                        camera_name TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        timestamp_iso TEXT NOT NULL,
                        window_seconds INTEGER NOT NULL,
                        total_volume INTEGER NOT NULL,
                        car_volume INTEGER NOT NULL,
                        motorcycle_volume INTEGER NOT NULL,
                        truck_volume INTEGER NOT NULL,
                        bus_volume INTEGER NOT NULL,
                        avg_active_vehicles REAL NOT NULL,
                        peak_active_vehicles INTEGER NOT NULL,
                        occupancy_ratio REAL NOT NULL,
                        avg_speed_px_s REAL NOT NULL,
                        stopped_vehicles INTEGER NOT NULL,
                        congestion_index REAL NOT NULL,
                        avg_flow_dx REAL NOT NULL,
                        avg_flow_dy REAL NOT NULL,
                        flow_rate_vpm REAL NOT NULL,
                        flow_rate_vph REAL NOT NULL,
                        hour INTEGER NOT NULL,
                        minute INTEGER NOT NULL,
                        day_of_week INTEGER NOT NULL,
                        is_weekend INTEGER NOT NULL,
                        sin_hour REAL NOT NULL,
                        cos_hour REAL NOT NULL,
                        sin_dow REAL NOT NULL,
                        cos_dow REAL NOT NULL,
                        volume_basis TEXT NOT NULL DEFAULT 'tracks',
                        timestamp_local_iso TEXT,
                        median_speed_bps REAL,
                        mean_speed_bps REAL,
                        queue_share REAL,
                        avg_traffic_vehicles REAL,
                        lane_occupancy REAL
                    );
                """)
                cols = {r[1] for r in conn.execute("PRAGMA table_info(camera_traffic_timeseries)")}
                for name, decl in _ADDED_COLUMNS:
                    if name not in cols:
                        conn.execute(f"ALTER TABLE camera_traffic_timeseries ADD COLUMN {name} {decl}")
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cam_ts_win
                    ON camera_traffic_timeseries (camera_id, window_seconds, timestamp);
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_timestamp
                    ON camera_traffic_timeseries (timestamp);
                """)
                # One camera over a time range (dataset exports, congestion series): idx_cam_ts_win has
                # window_seconds between the two, so it can only narrow by camera.
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_cam_time
                    ON camera_traffic_timeseries (camera_id, timestamp);
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS camera_cumulative_volume (
                        camera_id INTEGER PRIMARY KEY,
                        camera_name TEXT NOT NULL,
                        cumulative_volume INTEGER NOT NULL DEFAULT 0,
                        cumulative_cars INTEGER NOT NULL DEFAULT 0,
                        cumulative_motorcycles INTEGER NOT NULL DEFAULT 0,
                        cumulative_trucks INTEGER NOT NULL DEFAULT 0,
                        cumulative_buses INTEGER NOT NULL DEFAULT 0,
                        last_updated REAL NOT NULL
                    );
                """)
                conn.execute("CREATE TABLE IF NOT EXISTS analytics_meta "
                             "(key TEXT PRIMARY KEY, value TEXT NOT NULL);")
            try:
                self._migrate_data(conn)
            except sqlite3.Error as exc:       # e.g. another process holds the write lock: retry next start
                print(f"[analytics] data migration postponed: {exc}")

    def _migrate_data(self, conn: sqlite3.Connection) -> None:
        """One-time, idempotent repairs of rows written by earlier versions (guarded by ``analytics_meta``)."""
        def done(key: str) -> bool:
            return conn.execute("SELECT 1 FROM analytics_meta WHERE key = ?", (key,)).fetchone() is not None

        def mark(key: str, value: str) -> None:
            conn.execute("INSERT OR REPLACE INTO analytics_meta (key, value) VALUES (?, ?)", (key, value))

        # 1. Time features were computed in UTC; the cameras are on WIB. Recompute them (and fill the
        #    local ISO column) from the stored epoch so one column never mixes two meanings.
        if not done("features_tz"):
            with conn:
                last = 0
                while True:
                    rows = conn.execute("SELECT id, timestamp FROM camera_traffic_timeseries "
                                        "WHERE id > ? ORDER BY id LIMIT 5000", (last,)).fetchall()
                    if not rows:
                        break
                    conn.executemany(
                        "UPDATE camera_traffic_timeseries SET hour=?, minute=?, day_of_week=?, is_weekend=?, "
                        "sin_hour=?, cos_hour=?, sin_dow=?, cos_dow=?, timestamp_local_iso=? WHERE id=?",
                        [(*TrafficAnalyticsEngine.compute_temporal_features(r["timestamp"]),
                          local_iso(r["timestamp"]), r["id"]) for r in rows])
                    last = rows[-1]["id"]
                mark("features_tz", LOCAL_TZ_NAME)

        # 2. Windows without detections never saw a counted_now flag and were stored as 'tracks' even
        #    when the caller reports crossings. From the first crossings-based row on, an empty
        #    track-based window is an empty crossings window.
        if not done("empty_window_basis"):
            with conn:
                conn.execute(
                    "UPDATE camera_traffic_timeseries SET volume_basis = 'crossings' "
                    "WHERE volume_basis = 'tracks' AND total_volume = 0 AND timestamp >= "
                    "(SELECT MIN(timestamp) FROM camera_traffic_timeseries WHERE volume_basis = 'crossings')")
                mark("empty_window_basis", "1")

    # ── writes ───────────────────────────────────────────────────────────────

    def insert_metrics(self, m: TimeWindowMetrics) -> int:
        """Persist a single completed time-window metric record."""
        values = {**asdict(m), "timestamp_local_iso": m.timestamp_local_iso or local_iso(m.timestamp)}
        row = tuple(values[c] for c in _INSERT_COLUMNS)
        with self._lock:
            conn = self._writer()
            try:
                with conn:      # commit on success, roll back on any error
                    rec_id = conn.execute(_INSERT_SQL, row).lastrowid

                    # Upsert cumulative counts
                    conn.execute("""
                        INSERT INTO camera_cumulative_volume (
                            camera_id, camera_name, cumulative_volume,
                            cumulative_cars, cumulative_motorcycles, cumulative_trucks, cumulative_buses,
                            last_updated
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ON CONFLICT(camera_id) DO UPDATE SET
                            camera_name = excluded.camera_name,
                            cumulative_volume = cumulative_volume + excluded.cumulative_volume,
                            cumulative_cars = cumulative_cars + excluded.cumulative_cars,
                            cumulative_motorcycles = cumulative_motorcycles + excluded.cumulative_motorcycles,
                            cumulative_trucks = cumulative_trucks + excluded.cumulative_trucks,
                            cumulative_buses = cumulative_buses + excluded.cumulative_buses,
                            last_updated = excluded.last_updated;
                    """, (
                        m.camera_id, m.camera_name, m.total_volume,
                        m.car_volume, m.motorcycle_volume, m.truck_volume, m.bus_volume,
                        m.timestamp
                    ))
                return rec_id
            except sqlite3.Error:
                self._drop_writer()     # never keep a possibly broken connection
                raise

    # ── reads ────────────────────────────────────────────────────────────────

    def congestion_series(self, camera_id: int, since_ts: float, bucket_s: float) -> List[Dict[str, Any]]:
        """Congestion index of one camera in epoch-aligned buckets: the mean over the windows that
        ended in the bucket, weighted by window length (windows can be any length: rolling visits
        close a window at the end of each visit)."""
        sql = """
            SELECT CAST(timestamp / ? AS INTEGER) AS bucket,
                   SUM(congestion_index * window_seconds) / SUM(window_seconds) AS congestion,
                   SUM(window_seconds) AS seconds
            FROM camera_traffic_timeseries
            WHERE camera_id = ? AND timestamp >= ? AND window_seconds > 0
            GROUP BY bucket ORDER BY bucket;
        """
        rows = self._read(sql, (bucket_s, camera_id, since_ts))
        return [{"t": r["bucket"] * bucket_s, "congestion": round(float(r["congestion"]), 3),
                 "seconds": int(r["seconds"])} for r in rows]

    def query_timeseries(
        self,
        camera_id: Optional[int] = None,
        window_seconds: Optional[int] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 500,
        order_asc: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query time-series records.

        ``window_seconds=None`` returns the stored windows as they are (any length: focus windows
        are ~60 s, rolling visits 2-62 s). Earlier versions filtered ``window_seconds = 60`` and so
        returned about 1% of the rows.

        ``window_seconds=N`` aggregates the windows into N-second buckets aligned to local (WIB)
        midnight, bucketed by the window END (``timestamp`` of the result row = bucket end): volumes
        are summed, means are weighted by window length, ``flow_rate_*`` = volume / observed seconds.
        Windows are not split across bucket edges here; ``backend/dataset.py`` prorates them and
        also reports observed-time coverage, which is what analysis should use.
        """
        clauses: List[str] = []
        params: List[Any] = []
        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if start_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("timestamp <= ?")
            params.append(end_ts)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        direction = "ASC" if order_asc else "DESC"

        if window_seconds is None:
            rows = self._read(f"SELECT * FROM camera_traffic_timeseries {where} "
                              f"ORDER BY timestamp {direction}, id {direction} LIMIT ?;", [*params, limit])
            return [dict(row) for row in rows]

        n = int(window_seconds)
        if n <= 0:
            raise ValueError("window_seconds must be positive (or None for the raw windows)")
        off = LOCAL_UTC_OFFSET_S
        sql = f"""
            SELECT CAST((timestamp + {off}) / {n} AS INTEGER) AS bucket, camera_id,
                   MAX(camera_name) AS camera_name, COUNT(*) AS n_windows,
                   SUM(window_seconds) AS seconds,
                   SUM(total_volume) AS total_volume, SUM(car_volume) AS car_volume,
                   SUM(motorcycle_volume) AS motorcycle_volume, SUM(truck_volume) AS truck_volume,
                   SUM(bus_volume) AS bus_volume, MAX(peak_active_vehicles) AS peak_active_vehicles,
                   SUM(stopped_vehicles) AS stopped_vehicles,
                   {_wmean('avg_active_vehicles')} AS avg_active_vehicles,
                   {_wmean('occupancy_ratio')} AS occupancy_ratio,
                   {_wmean('avg_speed_px_s', 'avg_speed_px_s > 0')} AS avg_speed_px_s,
                   {_wmean('congestion_index')} AS congestion_index,
                   {_wmean('avg_flow_dx')} AS avg_flow_dx, {_wmean('avg_flow_dy')} AS avg_flow_dy,
                   {_wmean('median_speed_bps')} AS median_speed_bps,
                   {_wmean('mean_speed_bps')} AS mean_speed_bps,
                   {_wmean('queue_share')} AS queue_share,
                   {_wmean('avg_traffic_vehicles')} AS avg_traffic_vehicles,
                   SUM(volume_basis = 'crossings') AS n_crossings, SUM(volume_basis = 'tracks') AS n_tracks
            FROM camera_traffic_timeseries {where}
            GROUP BY bucket, camera_id
            ORDER BY bucket {direction}, camera_id {direction}
            LIMIT ?;
        """
        out: List[Dict[str, Any]] = []
        for r in self._read(sql, [*params, limit]):
            start = r["bucket"] * n - off
            end = start + n
            seconds = float(r["seconds"] or 0.0)
            total = int(r["total_volume"] or 0)
            hour, minute, dow, wknd, sin_h, cos_h, sin_d, cos_d = TrafficAnalyticsEngine.compute_temporal_features(start)
            basis = ("crossings" if r["n_crossings"] == r["n_windows"] else
                     "tracks" if r["n_tracks"] == r["n_windows"] else "mixed")

            def num(key: str, nd: int) -> Optional[float]:
                return None if r[key] is None else round(float(r[key]), nd)

            out.append({
                "camera_id": r["camera_id"], "camera_name": r["camera_name"],
                "timestamp": float(end), "timestamp_iso": datetime.fromtimestamp(end, timezone.utc).isoformat(),
                "timestamp_local_iso": local_iso(end),
                "window_seconds": n, "observed_seconds": seconds, "n_windows": int(r["n_windows"]),
                "total_volume": total, "car_volume": int(r["car_volume"] or 0),
                "motorcycle_volume": int(r["motorcycle_volume"] or 0), "truck_volume": int(r["truck_volume"] or 0),
                "bus_volume": int(r["bus_volume"] or 0),
                "avg_active_vehicles": num("avg_active_vehicles", 2) or 0.0,
                "peak_active_vehicles": int(r["peak_active_vehicles"] or 0),
                "occupancy_ratio": num("occupancy_ratio", 4) or 0.0,
                "avg_speed_px_s": num("avg_speed_px_s", 2) or 0.0,
                "stopped_vehicles": int(r["stopped_vehicles"] or 0),
                "congestion_index": num("congestion_index", 3) or 0.0,
                "avg_flow_dx": num("avg_flow_dx", 2) or 0.0, "avg_flow_dy": num("avg_flow_dy", 2) or 0.0,
                "flow_rate_vpm": round(total / seconds * 60.0, 2) if seconds > 0 else 0.0,
                "flow_rate_vph": round(total / seconds * 3600.0, 1) if seconds > 0 else 0.0,
                # local wall clock of the bucket START (raw rows carry that of the window end)
                "hour": hour, "minute": minute, "day_of_week": dow, "is_weekend": wknd,
                "sin_hour": sin_h, "cos_hour": cos_h, "sin_dow": sin_d, "cos_dow": cos_d,
                "volume_basis": basis,
                "median_speed_bps": num("median_speed_bps", 3), "mean_speed_bps": num("mean_speed_bps", 3),
                "queue_share": num("queue_share", 4), "avg_traffic_vehicles": num("avg_traffic_vehicles", 2),
            })
        return out

    def get_cumulative_summary(self, camera_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get cumulative statistics across one or all cameras."""
        if camera_id is not None:
            rows = self._read("SELECT * FROM camera_cumulative_volume WHERE camera_id = ?;", (camera_id,))
        else:
            rows = self._read("SELECT * FROM camera_cumulative_volume ORDER BY cumulative_volume DESC;")
        return [dict(row) for row in rows]


class _LaneMap:
    """Where a camera's moving traffic drives: a decaying heat map of the ground footprints (lower
    half of the box) of vehicles seen moving, on the ``GRID_W`` x ``GRID_H`` coverage grid.

    A vehicle standing still on cells that moving traffic regularly uses is queued; one standing
    where traffic does not go (kerb, car park, shop front) is parked. The same map gives the road
    area, so occupancy is measured against the lanes rather than the whole picture (sky,
    buildings). Evidence decays per moving-footprint frame (``LANE_HALF_LIFE``), not per second,
    so the map survives the hours a rolling camera goes unobserved and a long jam with few movers.
    """

    __slots__ = ("heat", "obs", "_ref")
    _DECAY = 0.5 ** (1.0 / LANE_HALF_LIFE)

    def __init__(self, heat: Optional[np.ndarray] = None, obs: float = 0.0):
        self.heat = (np.asarray(heat, dtype=np.float32).copy() if heat is not None
                     else np.zeros((GRID_H, GRID_W), dtype=np.float32))
        self.obs = float(obs)
        self._ref: Optional[float] = None

    @staticmethod
    def cells(box: Tuple[float, float, float, float], fw: int, fh: int) -> Tuple[int, int, int, int]:
        """Grid rows/cols (r1, r2, c1, c2) of a box's ground footprint: its lower half."""
        x1, y1, x2, y2 = box
        c1 = min(max(int(x1 / fw * GRID_W), 0), GRID_W - 1)
        c2 = min(max(math.ceil(x2 / fw * GRID_W), c1 + 1), GRID_W)
        r1 = min(max(int((y1 + y2) / 2.0 / fh * GRID_H), 0), GRID_H - 1)
        r2 = min(max(math.ceil(y2 / fh * GRID_H), r1 + 1), GRID_H)
        return r1, r2, c1, c2

    @property
    def ready(self) -> bool:
        return self.obs >= LANE_MIN_OBS

    def add(self, boxes: Sequence[Tuple[float, float, float, float]], fw: int, fh: int) -> None:
        """One frame's moving vehicles as lane evidence."""
        if not boxes or fw <= 0 or fh <= 0:
            return
        self.heat *= self._DECAY
        for box in boxes:
            r1, r2, c1, c2 = self.cells(box, fw, fh)
            self.heat[r1:r2, c1:c2] += 1.0
        self.obs = self.obs * self._DECAY + len(boxes)
        self._ref = None

    def _reference(self) -> float:
        """Heat of a well-used lane cell: the 90th percentile of the non-zero cells."""
        if self._ref is None:
            used = self.heat[self.heat > 0]
            self._ref = float(np.percentile(used, 90)) if used.size else 0.0
        return self._ref

    def score(self, box: Tuple[float, float, float, float], fw: int, fh: int) -> float:
        """How much of a box's footprint lies on lanes (mean normalised heat, 0..1)."""
        ref = self._reference()
        if ref <= 0 or fw <= 0 or fh <= 0:
            return 0.0
        r1, r2, c1, c2 = self.cells(box, fw, fh)
        return float(np.minimum(self.heat[r1:r2, c1:c2] / ref, 1.0).mean())

    def occupancy(self, boxes: Sequence[Tuple[float, float, float, float]], fw: int, fh: int) -> Optional[float]:
        """Share of the lane area covered by the footprints of ``boxes`` (None if no lane area)."""
        ref = self._reference()
        if ref <= 0 or fw <= 0 or fh <= 0:
            return None
        lane = self.heat >= LANE_CELL_MIN * ref
        area = int(lane.sum())
        if area == 0:
            return None
        covered = np.zeros_like(lane)
        for box in boxes:
            r1, r2, c1, c2 = self.cells(box, fw, fh)
            covered[r1:r2, c1:c2] = True
        return float((covered & lane).sum()) / area


class _LiveMean:
    """Mean of (t, value) samples over the last ``span`` seconds in O(1) per update (running sum,
    re-summed exactly now and then so rounding error cannot build up)."""
    __slots__ = ("span", "_q", "_sum", "_ops")

    def __init__(self, span: float):
        self.span = span
        self._q: deque = deque()
        self._sum = 0.0
        self._ops = 0

    def add(self, t: float, value: float) -> float:
        q = self._q
        q.append((t, value))
        self._sum += value
        while len(q) > 1 and t - q[0][0] > self.span:
            self._sum -= q.popleft()[1]
        self._ops += 1
        if self._ops >= 4096 or len(q) == 1:
            self._sum = math.fsum(v for _, v in q)
            self._ops = 0
        return self._sum / len(q)


class _ClosedWindow(NamedTuple):
    """A window closed under the engine lock, waiting to be written to the database."""
    metrics: TimeWindowMetrics
    learned_free_flow: bool


# ── Analytics Engine ──────────────────────────────────────────────────────────

class TrafficAnalyticsEngine:
    """
    Per-Camera Traffic Volume & Time-Series Extraction Engine.

    Designed for Apple Silicon Neural Engine real-time inference pipelines.
    Computes kinematics, density, spatial occupancy, congestion score,
    and publishes structured rolling buckets for time-series ML models.
    """

    def __init__(
        self,
        db_path: str = DEFAULT_DB_PATH,
        primary_window_seconds: int = 60,
        micro_window_seconds: int = 10,
        macro_window_seconds: int = 300,
        stop_bodies_per_s: float = 0.10,        # below this (body sizes/s) a moving-stream vehicle is stopped
        static_after_s: float = 6.0,            # unmoved this long since first seen -> parked
        moved_extent: float = 0.6,              # body sizes travelled that prove a track is moving traffic
        free_flow_path: Optional[str] = None,   # JSON file persisting each camera's learned free-flow speed
        assume_crossings: Optional[bool] = None,  # does the caller report counting-line crossings (``counted_now``)?
        lanes_path: Optional[str] = None,       # .npz persisting each camera's learned lane map
    ):
        """``assume_crossings``: True/False states whether the caller reports crossings; None (default)
        infers it -- once any detection has carried ``counted_now`` the caller is known to report
        crossings, so a window with NO detections is a (zero-volume) crossings window too. Windows
        with detections but no ``counted_now`` are always track-based."""
        self.db = TrafficAnalyticsDatabase(db_path)
        self.primary_window_seconds = primary_window_seconds
        self.micro_window_seconds = micro_window_seconds
        self.macro_window_seconds = macro_window_seconds
        self.stop_bodies_per_s = stop_bodies_per_s
        self.static_after_s = static_after_s
        self.moved_extent = moved_extent

        # Per-camera free-flow speed (body sizes/s), learned from that camera's own traffic. Until a
        # camera has one, slowness is not judged (queueing still is): any fixed default would call
        # a narrow slow street permanently congested.
        self._free_flow_path = free_flow_path
        self._free_flow_saved = 0.0
        self._free_flow_bps: Dict[int, float] = self._load_free_flow(free_flow_path)
        self._ff_io_lock = threading.Lock()      # serialises writes of the free-flow JSON (outside ``_lock``)
        # Per-camera lane maps (where moving traffic drives), persisted like the free-flow speeds.
        self._lanes_path = lanes_path
        self._lanes_saved = 0.0
        self._lanes: Dict[int, _LaneMap] = self._load_lanes(lanes_path)
        # Per-camera running mean of the per-frame congestion values behind the live index.
        self._live_mean: Dict[int, _LiveMean] = {}
        self._assume_crossings = assume_crossings
        self._crossings_seen = False             # any detection so far carried ``counted_now``

        # Guards the in-memory state below. No file or database I/O happens while it is held: the
        # inference thread would otherwise stall every other caller (and the dashboard) on disk.
        self._lock = threading.Lock()

        # Per-camera active vehicle kinematics: {camera_id: {track_id: TrackKinematics}}
        self._camera_tracks: Dict[int, Dict[int, TrackKinematics]] = defaultdict(dict)

        # Per-camera window accumulators:
        # {camera_id: {
        #     "window_start": float,
        #     "seen_tracks": set(),
        #     "class_tracks": defaultdict(set),
        #     "frame_counts": [],
        #     "frame_occupancies": [],
        #     "frame_speeds": [],
        #     "stopped_tracks": set(),
        #     "flow_dxs": [],
        #     "flow_dys": [],
        # }}
        self._window_state: Dict[int, Dict[str, Any]] = defaultdict(self._new_window_state)

        # In-memory rolling history for instant dashboard charting (max 60 data points)
        self._history_60s: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))

        # Real-time instantaneous snapshot per camera
        self._live_snapshot: Dict[int, Dict[str, Any]] = {}

    def _new_window_state(self) -> Dict[str, Any]:
        return {
            "window_start": time.time(),
            "seen_tracks": set(),
            "class_tracks": defaultdict(set),
            "frame_counts": [],
            "frame_occupancies": [],
            "frame_lane_occ": [],
            "frame_speeds": [],
            "frame_cong": [],
            "frame_bps": [],
            "crossing_mode": False,          # a detection carried counted_now (see record_frame)
            "saw_detections": False,         # at least one detection arrived in this window
            "crossings": 0,
            "class_crossings": defaultdict(int),
            "stopped_tracks": set(),
            "flow_dxs": [],
            "flow_dys": [],
            "last_ts": 0.0,
            "camera_name": "",
        }

    # ── Temporal ML feature helpers ───────────────────────────────────────────

    @staticmethod
    def compute_temporal_features(ts: float) -> Tuple[int, int, int, int, float, float, float, float]:
        """
        Extract cyclic temporal encodings essential for LSTM convergence, in LOCAL (WIB, UTC+7) time.
        Returns: (hour, minute, day_of_week, is_weekend, sin_hour, cos_hour, sin_dow, cos_dow)
        with day_of_week 0 = Monday.
        """
        dt = datetime.fromtimestamp(ts, tz=LOCAL_TZ)
        hour = dt.hour
        minute = dt.minute
        dow = dt.weekday()
        is_weekend = 1 if dow >= 5 else 0

        # Continuous hour angle [0, 2pi)
        hour_cont = hour + minute / 60.0
        angle_hour = 2.0 * math.pi * (hour_cont / 24.0)
        sin_hour = round(math.sin(angle_hour), 5)
        cos_hour = round(math.cos(angle_hour), 5)

        # Day of week angle [0, 2pi)
        angle_dow = 2.0 * math.pi * (dow / 7.0)
        sin_dow = round(math.sin(angle_dow), 5)
        cos_dow = round(math.cos(angle_dow), 5)

        return hour, minute, dow, is_weekend, sin_hour, cos_hour, sin_dow, cos_dow

    # ── Kinematic analysis & Frame ingestion ──────────────────────────────────

    def record_frame(
        self,
        camera_id: int,
        camera_name: str,
        timestamp: float,
        active_counts: Dict[str, int],
        detections: Sequence[Dict[str, Any]],
        frame_shape: Tuple[int, int],
    ) -> Dict[str, Any]:
        """
        Ingest a processed frame from the detector/tracker.
        
        Args:
            camera_id: Unique integer ID of camera.
            camera_name: Human-readable location / stream name.
            timestamp: Precise frame timestamp (epoch seconds).
            active_counts: Count of vehicles currently in frame by class name.
            detections: List of detection dicts:
                [{'track_id': int, 'class_id': int, 'class_name': str,
                  'box': (x1, y1, x2, y2), 'conf': float}]
            frame_shape: (height, width) of the video frame.

        Returns:
            Dict containing instantaneous live metrics for this camera.
        """
        fh, fw = frame_shape

        total_in_frame = sum(active_counts.values())

        current_frame_speeds: List[float] = []
        current_frame_dxs: List[float] = []
        current_frame_dys: List[float] = []
        traffic_boxes: List[Tuple[float, float, float, float]] = []
        traffic_bps: List[float] = []        # recent body-sizes/s of traffic-stream vehicles
        moving_boxes: List[Tuple[float, float, float, float]] = []   # lane evidence
        current_frame_stopped = 0
        static_in_frame = 0
        queued_in_frame = 0
        pending: Optional[_ClosedWindow] = None

        with self._lock:
            cam_tracks = self._camera_tracks[camera_id]
            win = self._window_state[camera_id]
            lanes = self._lanes.get(camera_id)
            if lanes is None:
                lanes = self._lanes[camera_id] = _LaneMap()
            lanes_ready = lanes.ready

            if not win["frame_counts"]:
                win["window_start"] = timestamp
            if detections:
                win["saw_detections"] = True

            for det in detections:
                tid = det["track_id"]
                cid = det.get("class_id", 2)
                cname = det.get("class_name", "vehicle")
                x1, y1, x2, y2 = det["box"]

                # Volume from counting-line crossings when the caller reports them (a detection
                # carries ``counted_now`` = this track crossed the line in this frame).
                if "counted_now" in det:
                    win["crossing_mode"] = True
                    self._crossings_seen = True
                    if det["counted_now"]:
                        win["crossings"] += 1
                        win["class_crossings"][cname] += 1

                w = max(0, x2 - x1)
                h = max(0, y2 - y1)
                size = float(max(w, h, 1))
                cx = (x1 + x2) / 2.0
                cy = (y1 + y2) / 2.0

                prev = cam_tracks.get(tid)
                if prev is None:
                    # New track: too young to tell parked from moving, so it is left out of the
                    # congestion measures until it has either moved or stayed put long enough.
                    track = cam_tracks[tid] = TrackKinematics(
                        track_id=tid, class_id=cid, class_name=cname,
                        first_seen=timestamp, last_seen=timestamp,
                        last_cx=cx, last_cy=cy, last_w=w, last_h=h,
                        speed_px_s=0.0, frame_count=1,
                        first_cx=cx, first_cy=cy, size=size,
                    )
                    track.history.append((timestamp, cx, cy))
                    win["seen_tracks"].add(tid)
                    win["class_tracks"][cname].add(tid)
                    continue

                track = prev
                dt = max(timestamp - track.last_seen, 1e-4)
                dx = cx - track.last_cx
                dy = cy - track.last_cy
                disp = math.sqrt(dx * dx + dy * dy)

                # Pixel speed (kept as the dataset's "speed proxy"), exponentially smoothed.
                inst_speed = disp / dt
                smooth_speed = 0.7 * track.speed_px_s + 0.3 * inst_speed if track.frame_count > 1 else inst_speed

                track.size = 0.8 * track.size + 0.2 * size
                track.max_extent = max(track.max_extent, math.hypot(cx - track.first_cx, cy - track.first_cy))
                track.history.append((timestamp, cx, cy))
                while track.history and timestamp - track.history[0][0] > LOOKBACK_S:
                    track.history.popleft()
                span = track.history[-1][0] - track.history[0][0]
                if span >= MIN_SPEED_SPAN_S:
                    _, fx, fy = track.history[0]
                    track.recent_bps = math.hypot(cx - fx, cy - fy) / max(track.size, 1.0) / span
                else:
                    track.recent_bps = None

                moved = track.max_extent >= self.moved_extent * track.size
                age = timestamp - track.first_seen
                # A still vehicle where moving traffic drives is queued, not parked.
                queued = (not moved and lanes_ready and age >= QUEUE_AFTER_S
                          and lanes.score((x1, y1, x2, y2), fw, fh) >= LANE_MIN_SCORE)
                track.is_static = (not moved) and (not queued) and age >= self.static_after_s

                track.last_seen = timestamp
                track.last_cx, track.last_cy, track.last_w, track.last_h = cx, cy, w, h
                track.speed_px_s = smooth_speed
                track.dx_px_s, track.dy_px_s = dx / dt, dy / dt
                track.total_displacement += disp
                track.frame_count += 1

                if track.is_static:
                    # Parked: not traffic, so not volume either.
                    static_in_frame += 1
                    win["seen_tracks"].discard(tid)
                    win["class_tracks"][cname].discard(tid)
                    track.is_stopped = False
                    continue

                win["seen_tracks"].add(tid)
                win["class_tracks"][cname].add(tid)
                if not moved and not queued:
                    track.is_stopped = False    # undetermined (young and still): left out
                    continue

                # Part of the traffic stream: moving, or standing in a lane (queued).
                traffic_boxes.append((x1, y1, x2, y2))
                current_frame_speeds.append(smooth_speed)
                current_frame_dxs.append(track.dx_px_s)
                current_frame_dys.append(track.dy_px_s)
                if moved:
                    moving_boxes.append((x1, y1, x2, y2))
                else:
                    queued_in_frame += 1
                bps = track.recent_bps
                if bps is None and queued:
                    bps = 0.0                   # still in its lane for >= QUEUE_AFTER_S
                if bps is not None:
                    traffic_bps.append(bps)
                    track.is_stopped = bps < self.stop_bodies_per_s
                else:
                    track.is_stopped = False
                if track.is_stopped:
                    current_frame_stopped += 1
                    win["stopped_tracks"].add(tid)

            # Prune stale tracks that haven't been seen in > 10 seconds
            stale_threshold = timestamp - 10.0
            for t in [t for t, k in cam_tracks.items() if k.last_seen < stale_threshold]:
                del cam_tracks[t]

            # ── Congestion ───────────────────────────────────────────────────
            frame_occupancy = self._coverage(traffic_boxes, fw, fh)
            lane_occupancy = lanes.occupancy(traffic_boxes, fw, fh) if lanes_ready else None
            lanes.add(moving_boxes, fw, fh)
            avg_speed_frame = float(np.mean(current_frame_speeds)) if current_frame_speeds else 0.0
            free_flow = self._free_flow_bps.get(camera_id)
            median_bps = float(np.median(traffic_bps)) if traffic_bps else 0.0
            frame_cong = self._congestion_index(
                coverage=frame_occupancy, speeds=traffic_bps,
                stopped=sum(1 for v in traffic_bps if v < self.stop_bodies_per_s), free_flow=free_flow,
                lane_occupancy=lane_occupancy)
            live_mean = self._live_mean.get(camera_id)
            if live_mean is None:
                live_mean = self._live_mean[camera_id] = _LiveMean(LIVE_SPAN_S)
            congestion_index = round(max(0.0, min(1.0, live_mean.add(timestamp, frame_cong))), 3)

            # Accumulate in current window
            win["last_ts"] = timestamp
            win["camera_name"] = camera_name
            win["frame_counts"].append(total_in_frame)
            win["frame_occupancies"].append(frame_occupancy)
            if lane_occupancy is not None:
                win["frame_lane_occ"].append(lane_occupancy)
            win["frame_cong"].append(frame_cong)
            win["frame_bps"].extend(traffic_bps)
            if current_frame_speeds:
                win["frame_speeds"].extend(current_frame_speeds)
            if current_frame_dxs:
                win["flow_dxs"].extend(current_frame_dxs)
                win["flow_dys"].extend(current_frame_dys)

            # Check if primary aggregation window (e.g. 60 seconds) elapsed
            elapsed = timestamp - win["window_start"]
            if elapsed >= self.primary_window_seconds and win["frame_counts"]:
                pending = self._close_window_locked(camera_id, camera_name, timestamp, elapsed)

            # Store live instantaneous snapshot
            basis = self._window_basis(win)
            live_snapshot = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "timestamp": timestamp,
                "active_total": total_in_frame,
                "active_counts": dict(active_counts),
                "traffic_vehicles": len(traffic_boxes),      # moving + queued
                "queued_vehicles": queued_in_frame,          # still, but standing in a lane
                "parked_vehicles": static_in_frame,
                "occupancy_ratio": round(frame_occupancy, 4),   # share of the FRAME covered by traffic
                "occupancy_percent": round(frame_occupancy * 100.0, 1),
                # share of the learned lane (road) area covered by traffic; None until lanes are learned
                "lane_occupancy": round(lane_occupancy, 4) if lane_occupancy is not None else None,
                "lanes_learned": lanes_ready,
                "avg_speed_px_s": round(avg_speed_frame, 1),
                "median_speed_bodies_s": round(median_bps, 2),
                "free_flow_bodies_s": round(free_flow, 2) if free_flow else None,
                "stopped_vehicles": current_frame_stopped,
                "congestion_index": congestion_index,
                "congestion_label": self.get_congestion_label(congestion_index),
                "unique_window_volume": len(win["seen_tracks"]),
                "window_volume": win["crossings"] if basis == "crossings" else len(win["seen_tracks"]),
                "volume_basis": basis,
                "window_elapsed_s": round(elapsed, 1),
            }
            self._live_snapshot[camera_id] = live_snapshot

        if pending is not None:
            self._persist(pending)      # database and file I/O happen outside the lock
        return live_snapshot

    def _window_basis(self, win: Dict[str, Any]) -> str:
        """What a window's volume counts. It follows whether the CALLER provides crossings, not
        whether this window happened to contain detections (an empty window is zero crossings on a
        crossings-reporting caller)."""
        if win["crossing_mode"]:
            return "crossings"
        if win["saw_detections"]:
            return "tracks"         # detections without ``counted_now``: the caller gave no crossings
        if self._assume_crossings is not None:
            return "crossings" if self._assume_crossings else "tracks"
        return "crossings" if self._crossings_seen else "tracks"

    # ── Congestion model ──────────────────────────────────────────────────────

    @staticmethod
    def _coverage(boxes: Sequence[Tuple[float, float, float, float]], fw: int, fh: int) -> float:
        """Share of the frame covered by the union of ``boxes`` (overlaps counted once)."""
        if not boxes or fw <= 0 or fh <= 0:
            return 0.0
        grid = np.zeros((GRID_H, GRID_W), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            c1 = min(max(int(x1 / fw * GRID_W), 0), GRID_W)   # plain min/max: np.clip on scalars was 10x slower
            c2 = min(max(math.ceil(x2 / fw * GRID_W), 0), GRID_W)
            r1 = min(max(int(y1 / fh * GRID_H), 0), GRID_H)
            r2 = min(max(math.ceil(y2 / fh * GRID_H), 0), GRID_H)
            if c2 > c1 and r2 > r1:
                grid[r1:r2, c1:c2] = True
        return float(grid.mean())

    @staticmethod
    def _congestion_index(coverage: float, speeds: Sequence[float], stopped: int,
                          free_flow: Optional[float], lane_occupancy: Optional[float] = None) -> float:
        """
        Congestion in [0, 1] from the moving-stream vehicles only.

        - queue (0.40): share of them standing still (full at ``STOPPED_FULL``);
        - slowness (0.45): how far their median speed is below the camera's own free-flow speed
          (none at ``SLOW_ABOVE`` of it, full at ``SLOW_FULL_BELOW``); until the camera has a learned
          free-flow speed only a standstill counts as slow;
        - density (0.15): how full the lanes are (``lane_occupancy``; the whole frame until the
          camera's lanes are learned), but only counted when traffic is also slow, so a busy road
          whose traffic flows freely is not "congested".
        Scaled down (quadratically) while fewer than ``MIN_TRAFFIC_FOR_FULL_CONFIDENCE`` vehicles can
        be measured: two cars stopped at a light are not a jam.
        """
        if not speeds:
            return 0.0
        queue = min(1.0, (stopped / len(speeds)) / STOPPED_FULL)
        if free_flow:
            ratio = float(np.median(speeds)) / free_flow
            slow = min(1.0, max(0.0, (SLOW_ABOVE - ratio) / (SLOW_ABOVE - SLOW_FULL_BELOW)))
        else:
            slow = queue ** 2   # no reference yet: only a (near) full standstill is slow whatever "normal" is
        if lane_occupancy is not None:
            dense = min(1.0, lane_occupancy / LANE_DENSITY_FULL)
        else:
            dense = min(1.0, coverage / DENSITY_FULL)
        raw = 0.40 * queue + 0.45 * slow + 0.15 * dense * slow
        return float(raw * min(1.0, len(speeds) / MIN_TRAFFIC_FOR_FULL_CONFIDENCE) ** 2)

    # ── Learned free-flow speeds ──────────────────────────────────────────────

    @staticmethod
    def _load_free_flow(path: Optional[str]) -> Dict[int, float]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return {int(k): float(v) for k, v in json.load(f).items() if float(v) >= MIN_FREE_FLOW_BPS}
        except (OSError, ValueError, TypeError):
            return {}

    def _save_free_flow(self, min_interval_s: float = 30.0) -> None:
        """Writes the learned free-flow speeds (at most every ``min_interval_s``). Call it WITHOUT
        holding ``_lock``: only the in-memory snapshot is taken under the lock, the file is written
        outside it."""
        if not self._free_flow_path:
            return
        with self._lock:
            if time.time() - self._free_flow_saved < min_interval_s:
                return
            self._free_flow_saved = time.time()
            snapshot = {str(k): round(v, 4) for k, v in self._free_flow_bps.items()}
        with self._ff_io_lock:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self._free_flow_path)), exist_ok=True)
                tmp = self._free_flow_path + ".tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(snapshot, f)
                os.replace(tmp, self._free_flow_path)
            except OSError:
                pass   # best effort: it is relearned from traffic anyway

    def _persist(self, pending: "_ClosedWindow") -> None:
        """Stores a closed window and, if it taught a free-flow speed, the reference file. Never
        called with ``_lock`` held."""
        try:
            self.db.insert_metrics(pending.metrics)
        except Exception as exc:  # noqa: BLE001
            print(f"[analytics] DB write error: {exc}")
        if pending.learned_free_flow:
            self._save_free_flow()
        self._save_lanes()

    # ── Learned lane maps ─────────────────────────────────────────────────────

    @staticmethod
    def _load_lanes(path: Optional[str]) -> Dict[int, _LaneMap]:
        if not path or not os.path.exists(path):
            return {}
        try:
            with np.load(path) as data:
                obs = {int(c): float(o) for c, o in data["obs"]} if "obs" in data else {}
                lanes = {}
                for key in data.files:
                    if key.startswith("heat_") and data[key].shape == (GRID_H, GRID_W):
                        cid = int(key[5:])
                        lanes[cid] = _LaneMap(data[key], obs.get(cid, 0.0))
                return lanes
        except (OSError, ValueError, KeyError):
            return {}

    def _save_lanes(self, min_interval_s: float = 60.0) -> None:
        """Writes the lane maps (at most every ``min_interval_s``); like ``_save_free_flow`` only
        the snapshot is taken under ``_lock``."""
        if not self._lanes_path:
            return
        with self._lock:
            if time.time() - self._lanes_saved < min_interval_s:
                return
            self._lanes_saved = time.time()
            arrays = {f"heat_{cid}": lane.heat.copy() for cid, lane in self._lanes.items() if lane.obs > 0}
            obs = np.array([[cid, lane.obs] for cid, lane in self._lanes.items() if lane.obs > 0],
                           dtype=np.float64).reshape(-1, 2)
        with self._ff_io_lock:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(self._lanes_path)), exist_ok=True)
                tmp = self._lanes_path + ".tmp.npz"
                np.savez_compressed(tmp, obs=obs, **arrays)
                os.replace(tmp, self._lanes_path)
            except OSError:
                pass   # best effort: relearned from traffic

    def close(self) -> None:
        """Releases the database connections (call at shutdown). Open windows are not flushed;
        the learned lane maps are saved."""
        self._save_lanes(min_interval_s=0.0)
        self.db.close()

    def flush_camera(self, camera_id: int, min_observed_s: float = 2.0) -> Optional[TimeWindowMetrics]:
        """
        Close a camera's partially filled window at the end of an observation burst.

        Cameras that are sampled intermittently (rolling coverage) would otherwise keep a
        window open across the unobserved gap, and ``volume / wall-clock elapsed`` would
        wildly under-report flow. The window is closed using only the *observed* span
        (first to last frame); bursts shorter than ``min_observed_s`` are discarded.
        Per-track kinematics are cleared so stale tracks never bridge the gap.
        """
        with self._lock:
            win = self._window_state.get(camera_id)
            self._camera_tracks[camera_id].clear()
            self._live_mean.pop(camera_id, None)
            if not win or not win["frame_counts"]:
                self._window_state[camera_id] = self._new_window_state()
                return None
            observed = win["last_ts"] - win["window_start"]
            if observed < min_observed_s:
                self._window_state[camera_id] = self._new_window_state()
                return None
            pending = self._close_window_locked(
                camera_id, win.get("camera_name") or f"Camera {camera_id}", win["last_ts"], observed
            )
        if pending is None:
            return None
        self._persist(pending)
        return pending.metrics

    @staticmethod
    def get_congestion_label(index: float) -> str:
        """Classify congestion index into traffic engineering level of service."""
        if index < 0.25:
            return "Smooth"
        elif index < 0.55:
            return "Moderate"
        elif index < 0.80:
            return "Heavy"
        else:
            return "Congested"

    def _close_window_locked(
        self,
        camera_id: int,
        camera_name: str,
        now: float,
        elapsed: float
    ) -> Optional["_ClosedWindow"]:
        """Aggregate the current window and rotate it. In-memory work only (``_lock`` is held): the
        caller persists the returned window after releasing the lock."""
        win = self._window_state[camera_id]
        if not win["frame_counts"]:
            return None

        basis = self._window_basis(win)
        if basis == "crossings":
            cls = win["class_crossings"]
            car_vol, moto_vol, truck_vol, bus_vol = (cls["car"], cls["motorcycle"], cls["truck"], cls["bus"])
            total_volume = win["crossings"]
        else:
            car_vol = len(win["class_tracks"]["car"])
            moto_vol = len(win["class_tracks"]["motorcycle"])
            truck_vol = len(win["class_tracks"]["truck"])
            bus_vol = len(win["class_tracks"]["bus"])
            total_volume = len(win["seen_tracks"])

        avg_active = float(np.mean(win["frame_counts"]))
        peak_active = int(np.max(win["frame_counts"]))
        avg_occupancy = float(np.mean(win["frame_occupancies"]))
        avg_speed = float(np.mean(win["frame_speeds"])) if win["frame_speeds"] else 0.0
        avg_dx = float(np.mean(win["flow_dxs"])) if win["flow_dxs"] else 0.0
        avg_dy = float(np.mean(win["flow_dys"])) if win["flow_dys"] else 0.0
        stopped_count = len(win["stopped_tracks"])

        # Flow rates normalized
        window_duration = max(elapsed, 1.0)
        flow_rate_vpm = round((total_volume / window_duration) * 60.0, 2)
        flow_rate_vph = round((total_volume / window_duration) * 3600.0, 1)

        # Window congestion index: mean of the per-frame indices (quiet frames count as 0), so a
        # window is "heavy" only if traffic was slow for much of it, not at one instant.
        congestion_index = round(float(np.clip(np.mean(win["frame_cong"]), 0.0, 1.0)), 3) if win["frame_cong"] else 0.0

        # Speed of the moving stream in body sizes/s (comparable across cameras); NULL when the window
        # had no measurable moving-stream vehicle, i.e. when speed is undefined rather than zero.
        samples = win["frame_bps"]
        median_bps = mean_bps = queue_share = None
        if samples:
            arr = np.asarray(samples, dtype=float)
            median_bps = round(float(np.median(arr)), 3)
            mean_bps = round(float(arr.mean()), 3)
            queue_share = round(float((arr < self.stop_bodies_per_s).mean()), 4)
        avg_traffic = round(len(samples) / max(len(win["frame_counts"]), 1), 2)

        # Learn this camera's free-flow speed from the speeds its traffic actually shows.
        learned_free_flow = False
        if len(samples) >= MIN_REF_SAMPLES:
            p = max(float(np.percentile(samples, REF_PERCENTILE)), MIN_FREE_FLOW_BPS)
            learned = self._free_flow_bps.get(camera_id)
            if learned is None:
                self._free_flow_bps[camera_id] = p
            else:
                weight = FREE_FLOW_RISE if p > learned else FREE_FLOW_SINK
                self._free_flow_bps[camera_id] = learned + weight * (p - learned)
            learned_free_flow = True

        # Temporal features: local (WIB) wall clock of the window end
        hour, minute, dow, is_wknd, sin_h, cos_h, sin_d, cos_d = self.compute_temporal_features(now)

        metrics = TimeWindowMetrics(
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=now,
            timestamp_iso=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            window_seconds=int(round(window_duration)),
            total_volume=total_volume,
            car_volume=car_vol,
            motorcycle_volume=moto_vol,
            truck_volume=truck_vol,
            bus_volume=bus_vol,
            avg_active_vehicles=round(avg_active, 2),
            peak_active_vehicles=peak_active,
            occupancy_ratio=round(avg_occupancy, 4),
            avg_speed_px_s=round(avg_speed, 2),
            stopped_vehicles=stopped_count,
            congestion_index=congestion_index,
            avg_flow_dx=round(avg_dx, 2),
            avg_flow_dy=round(avg_dy, 2),
            flow_rate_vpm=flow_rate_vpm,
            flow_rate_vph=flow_rate_vph,
            hour=hour,
            minute=minute,
            day_of_week=dow,
            is_weekend=is_wknd,
            sin_hour=sin_h,
            cos_hour=cos_h,
            sin_dow=sin_d,
            cos_dow=cos_d,
            volume_basis=basis,
            timestamp_local_iso=local_iso(now),
            median_speed_bps=median_bps,
            mean_speed_bps=mean_bps,
            queue_share=queue_share,
            avg_traffic_vehicles=avg_traffic,
            lane_occupancy=(round(float(np.mean(win["frame_lane_occ"])), 4)
                            if win["frame_lane_occ"] else None),
        )

        # Add to in-memory trend history for dashboard
        self._history_60s[camera_id].append(metrics.to_dict())

        # Reset window accumulator
        self._window_state[camera_id] = self._new_window_state()
        self._window_state[camera_id]["window_start"] = now

        return _ClosedWindow(metrics, learned_free_flow)

    # ── Real-time query methods for API & Dashboard ───────────────────────────

    def get_live_metrics(self, camera_id: int, now: Optional[float] = None) -> Dict[str, Any]:
        """Get live instantaneous metrics and rolling window preview.

        ``live`` is the newest snapshot only while the camera is being observed (newer than
        ``LIVE_MAX_AGE_S``); rolling cameras are watched for a short visit every few minutes, and
        their last reading must not pass for the current one. ``last_measured`` is the newest
        snapshot whatever its age, with ``live_age_s`` (None if the camera was never observed)."""
        now = time.time() if now is None else now
        with self._lock:
            snapshot = self._live_snapshot.get(camera_id)
            age = round(max(0.0, now - snapshot["timestamp"]), 1) if snapshot else None
            live = snapshot if snapshot and age <= LIVE_MAX_AGE_S else {}
            win = self._window_state.get(camera_id)

            if win:
                crossing_mode = self._window_basis(win) == "crossings"
                vol = win["crossings"] if crossing_mode else len(win["seen_tracks"])
                elapsed = max(now - win["window_start"], 1.0)
                rolling_vpm = round((vol / elapsed) * 60.0, 1)
                class_breakdown = (dict(win["class_crossings"]) if crossing_mode
                                   else {k: len(v) for k, v in win["class_tracks"].items()})
            else:
                crossing_mode = False
                vol = 0
                rolling_vpm = 0.0
                class_breakdown = {}

            history = list(self._history_60s.get(camera_id, ()))

            return {
                "live": live,
                "last_measured": snapshot or {},
                "live_age_s": age,
                "current_window": {
                    "volume_basis": "crossings" if crossing_mode else "tracks",
                    "total_volume": vol,
                    "flow_rate_vpm": rolling_vpm,
                    "class_volumes": class_breakdown,
                },
                "recent_history": history[-30:],  # last 30 minutes / intervals
            }

    def get_camera_summary(self, camera_id: int) -> Dict[str, Any]:
        """Get cumulative totals and historical averages for a camera."""
        cumul_list = self.db.get_cumulative_summary(camera_id)
        cumul = cumul_list[0] if cumul_list else {}
        # The latest windows of any length, weighted by their length: the flow rate is volume per
        # OBSERVED second and speed/congestion are time-weighted (speed only over windows that had
        # moving traffic: a window without any has no speed, not a speed of zero).
        recent = self.db.query_timeseries(camera_id=camera_id, limit=RECENT_SUMMARY_WINDOWS, order_asc=False)

        def weighted(key: str, only_positive: bool = False) -> float:
            pairs = [(float(r[key]), float(r["window_seconds"])) for r in recent
                     if r.get(key) is not None and r["window_seconds"] > 0 and (r[key] > 0 or not only_positive)]
            weight = sum(w for _, w in pairs)
            return sum(v * w for v, w in pairs) / weight if weight > 0 else 0.0

        seconds = sum(float(r["window_seconds"]) for r in recent)
        avg_vpm = round(sum(r["total_volume"] for r in recent) / seconds * 60.0, 2) if seconds > 0 else 0.0
        avg_speed = round(weighted("avg_speed_px_s", only_positive=True), 2)
        avg_cong = round(weighted("congestion_index"), 3)

        return {
            "camera_id": camera_id,
            "camera_name": cumul.get("camera_name", f"Camera {camera_id}"),
            "cumulative_volume": cumul.get("cumulative_volume", 0),
            "cumulative_breakdown": {
                "cars": cumul.get("cumulative_cars", 0),
                "motorcycles": cumul.get("cumulative_motorcycles", 0),
                "trucks": cumul.get("cumulative_trucks", 0),
                "buses": cumul.get("cumulative_buses", 0),
            },
            "recent_avg_flow_rate_vpm": avg_vpm,
            "recent_avg_speed_px_s": avg_speed,
            "recent_avg_congestion_index": avg_cong,
            "recent_congestion_label": self.get_congestion_label(avg_cong),
            "last_updated": cumul.get("last_updated", 0),
        }

    def get_congestion_series(self, camera_id: int, minutes: float = 60.0, bucket_minutes: float = 5.0,
                              now: Optional[float] = None) -> List[Dict[str, Any]]:
        bucket_s = max(1.0, float(bucket_minutes) * 60.0)
        since = (now if now is not None else time.time()) - float(minutes) * 60.0
        return self.db.congestion_series(camera_id, since, bucket_s)

    def get_all_camera_summaries(self) -> List[Dict[str, Any]]:
        """List summaries across all cameras."""
        return self.db.get_cumulative_summary()

    # ── Time-Series & LSTM Data Extraction ────────────────────────────────────

    def export_timeseries_dataset(
        self,
        camera_id: Optional[int] = None,
        window_seconds: Optional[int] = None,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Time-series rows of the stored windows (``window_seconds=None``) or aggregated into
        ``window_seconds`` buckets (see :meth:`TrafficAnalyticsDatabase.query_timeseries`).

        For analysis and model training prefer ``backend/dataset.py``: it prorates windows across
        bucket edges, reports observed-time coverage, uses counting-line vph and local-time features.
        """
        return self.db.query_timeseries(
            camera_id=camera_id,
            window_seconds=window_seconds,
            start_ts=start_ts,
            end_ts=end_ts,
            limit=limit,
            order_asc=True
        )

    def export_csv_str(
        self,
        camera_id: Optional[int] = None,
        window_seconds: Optional[int] = None,
        limit: int = 5000,
    ) -> str:
        """Export time-series records formatted as standard CSV with headers."""
        records = self.export_timeseries_dataset(camera_id=camera_id, window_seconds=window_seconds, limit=limit)
        if not records:
            # Return empty template with standard column headers
            fieldnames = [
                "timestamp_iso", "timestamp", "camera_id", "camera_name", "window_seconds",
                "total_volume", "car_volume", "motorcycle_volume", "truck_volume", "bus_volume",
                "avg_active_vehicles", "peak_active_vehicles", "occupancy_ratio",
                "avg_speed_px_s", "stopped_vehicles", "congestion_index",
                "avg_flow_dx", "avg_flow_dy", "flow_rate_vpm", "flow_rate_vph",
                "hour", "minute", "day_of_week", "is_weekend",
                "sin_hour", "cos_hour", "sin_dow", "cos_dow", "volume_basis",
                "timestamp_local_iso", "median_speed_bps", "mean_speed_bps", "queue_share", "avg_traffic_vehicles",
            ]
            buf = io.StringIO()
            writer = csv.DictWriter(buf, fieldnames=fieldnames)
            writer.writeheader()
            return buf.getvalue()

        fieldnames = list(records[0].keys())
        if "id" in fieldnames:
            fieldnames.remove("id")

        buf = io.StringIO()
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            row = dict(r)
            if "id" in row:
                del row["id"]
            writer.writerow(row)
        return buf.getvalue()

    def create_lstm_sequences(
        self,
        records: List[Dict[str, Any]],
        seq_len: int = 12,
        horizon: int = 1,
        feature_cols: Optional[List[str]] = None,
        target_col: str = "total_volume",
    ) -> Dict[str, Any]:
        """
        Convert time-series records into 3D sliding-window tensor arrays (X, y)
        for recurrent neural networks (LSTM, GRU, RNN):
        
        X shape: (num_samples, seq_len, num_features)
        y shape: (num_samples, horizon)
        """
        if not records or len(records) < (seq_len + horizon):
            return {
                "X": np.empty((0, seq_len, len(feature_cols or [])), dtype=np.float32),
                "y": np.empty((0, horizon), dtype=np.float32),
                "feature_names": feature_cols or [],
                "timestamps": [],
                "sample_count": 0,
            }

        if feature_cols is None:
            feature_cols = [
                "total_volume", "car_volume", "motorcycle_volume", "truck_volume", "bus_volume",
                "avg_active_vehicles", "occupancy_ratio", "avg_speed_px_s",
                "congestion_index", "flow_rate_vpm",
                "sin_hour", "cos_hour", "sin_dow", "cos_dow", "is_weekend"
            ]

        # Extract numeric array
        matrix = []
        target_vals = []
        timestamps = []

        for r in records:
            row_features = [float(r.get(c) or 0.0) for c in feature_cols]
            matrix.append(row_features)
            target_vals.append(float(r.get(target_col) or 0.0))
            timestamps.append(r.get("timestamp_iso", ""))

        matrix_np = np.array(matrix, dtype=np.float32)
        target_np = np.array(target_vals, dtype=np.float32)

        X_samples = []
        y_samples = []
        target_timestamps = []

        num_possible = len(matrix_np) - seq_len - horizon + 1
        for i in range(num_possible):
            X_seq = matrix_np[i : i + seq_len]
            y_seq = target_np[i + seq_len : i + seq_len + horizon]
            X_samples.append(X_seq)
            y_samples.append(y_seq)
            target_timestamps.append(timestamps[i + seq_len + horizon - 1])

        return {
            "X": np.array(X_samples, dtype=np.float32),
            "y": np.array(y_samples, dtype=np.float32),
            "feature_names": feature_cols,
            "target_name": target_col,
            "timestamps": target_timestamps,
            "sample_count": len(X_samples),
        }
