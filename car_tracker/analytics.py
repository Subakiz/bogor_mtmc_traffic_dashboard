"""
Per-Camera Traffic Volume & Time-Series Analytics Engine for CCTV YOLO ANE.

Provides:
- Real-time vehicle kinematics, displacement, speed proxy, and flow vector tracking.
- Per-camera vehicle volume and class breakdown (cars, motorcycles, trucks, buses).
- Traffic occupancy density (ratio of frame covered by vehicles) and congestion index.
- Regular time-bucket aggregation (10s, 30s, 60s, 300s) optimized for time-series ML & LSTM.
- Continuous cyclic temporal features (sin_hour, cos_hour, sin_dow, cos_dow, is_weekend).
- Persistent SQLite storage in WAL mode (data/traffic_timeseries.db).
- Dataset exporter (CSV, JSON, numpy/LSTM sequences) for PyTorch / TensorFlow model training.
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
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

# Default storage location
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = str(PROJECT_ROOT / "data" / "traffic_timeseries.db")


# ── Traffic-state model ────────────────────────────────────────────────────────
# Congestion is judged only on vehicles that are actually part of the traffic stream. A track that
# has never moved since it was first seen is parked (or too new to tell) and is left out of density,
# stopped-share and speed; otherwise a lot full of parked cars reads as a permanent jam.
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
MIN_REF_SAMPLES = 30      # speed samples a window needs before it updates the free-flow reference
REF_PERCENTILE = 85       # free-flow speed = this percentile of a camera's observed speeds


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
    timestamp: float                    # Unix epoch seconds (window end)
    timestamp_iso: str                  # ISO 8601 UTC formatted string
    window_seconds: int                 # Bucket length in seconds (e.g. 60)

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

    # Temporal features for LSTM / time-series ML
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

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── SQLite Persistence Layer ──────────────────────────────────────────────────

class TrafficAnalyticsDatabase:
    """Thread-safe SQLite storage for per-camera traffic time-series data."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA temp_store = MEMORY;")
        return conn

    def _init_db(self):
        with self._lock:
            conn = self._get_connection()
            try:
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
                            volume_basis TEXT NOT NULL DEFAULT 'tracks'
                        );
                    """)
                    cols = {r[1] for r in conn.execute("PRAGMA table_info(camera_traffic_timeseries)")}
                    if "volume_basis" not in cols:
                        conn.execute("ALTER TABLE camera_traffic_timeseries "
                                     "ADD COLUMN volume_basis TEXT NOT NULL DEFAULT 'tracks'")
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_cam_ts_win
                        ON camera_traffic_timeseries (camera_id, window_seconds, timestamp);
                    """)
                    conn.execute("""
                        CREATE INDEX IF NOT EXISTS idx_timestamp
                        ON camera_traffic_timeseries (timestamp);
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
            finally:
                conn.close()

    def insert_metrics(self, m: TimeWindowMetrics) -> int:
        """Persist a single completed time-window metric record."""
        with self._lock:
            conn = self._get_connection()
            try:
                with conn:
                    cur = conn.execute("""
                        INSERT INTO camera_traffic_timeseries (
                            camera_id, camera_name, timestamp, timestamp_iso, window_seconds,
                            total_volume, car_volume, motorcycle_volume, truck_volume, bus_volume,
                            avg_active_vehicles, peak_active_vehicles, occupancy_ratio,
                            avg_speed_px_s, stopped_vehicles, congestion_index,
                            avg_flow_dx, avg_flow_dy, flow_rate_vpm, flow_rate_vph,
                            hour, minute, day_of_week, is_weekend,
                            sin_hour, cos_hour, sin_dow, cos_dow, volume_basis
                        ) VALUES (
                            ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?,
                            ?, ?, ?,
                            ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?,
                            ?, ?, ?, ?, ?
                        );
                    """, (
                        m.camera_id, m.camera_name, m.timestamp, m.timestamp_iso, m.window_seconds,
                        m.total_volume, m.car_volume, m.motorcycle_volume, m.truck_volume, m.bus_volume,
                        m.avg_active_vehicles, m.peak_active_vehicles, m.occupancy_ratio,
                        m.avg_speed_px_s, m.stopped_vehicles, m.congestion_index,
                        m.avg_flow_dx, m.avg_flow_dy, m.flow_rate_vpm, m.flow_rate_vph,
                        m.hour, m.minute, m.day_of_week, m.is_weekend,
                        m.sin_hour, m.cos_hour, m.sin_dow, m.cos_dow, m.volume_basis
                    ))
                    rec_id = cur.lastrowid

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
            finally:
                conn.close()

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
        with self._lock:
            conn = self._get_connection()
            try:
                rows = conn.execute(sql, (bucket_s, camera_id, since_ts)).fetchall()
                return [{"t": r["bucket"] * bucket_s, "congestion": round(float(r["congestion"]), 3),
                         "seconds": int(r["seconds"])} for r in rows]
            finally:
                conn.close()

    def query_timeseries(
        self,
        camera_id: Optional[int] = None,
        window_seconds: int = 60,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 500,
        order_asc: bool = True
    ) -> List[Dict[str, Any]]:
        """Query time-series records matching criteria."""
        clauses = ["window_seconds = ?"]
        params: List[Any] = [window_seconds]

        if camera_id is not None:
            clauses.append("camera_id = ?")
            params.append(camera_id)
        if start_ts is not None:
            clauses.append("timestamp >= ?")
            params.append(start_ts)
        if end_ts is not None:
            clauses.append("timestamp <= ?")
            params.append(end_ts)

        direction = "ASC" if order_asc else "DESC"
        sql = f"""
            SELECT * FROM camera_traffic_timeseries
            WHERE {' AND '.join(clauses)}
            ORDER BY timestamp {direction}
            LIMIT ?;
        """
        params.append(limit)

        with self._lock:
            conn = self._get_connection()
            try:
                cur = conn.execute(sql, params)
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()

    def get_cumulative_summary(self, camera_id: Optional[int] = None) -> List[Dict[str, Any]]:
        """Get cumulative statistics across one or all cameras."""
        with self._lock:
            conn = self._get_connection()
            try:
                if camera_id is not None:
                    cur = conn.execute(
                        "SELECT * FROM camera_cumulative_volume WHERE camera_id = ?;", (camera_id,)
                    )
                else:
                    cur = conn.execute(
                        "SELECT * FROM camera_cumulative_volume ORDER BY cumulative_volume DESC;"
                    )
                return [dict(row) for row in cur.fetchall()]
            finally:
                conn.close()


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
    ):
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
        # Per-camera recent per-frame congestion values (t, index) behind the live index.
        self._cong_recent: Dict[int, deque] = defaultdict(deque)

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
        self._history_10s: Dict[int, deque] = defaultdict(lambda: deque(maxlen=60))

        # Real-time instantaneous snapshot per camera
        self._live_snapshot: Dict[int, Dict[str, Any]] = {}

    def _new_window_state(self) -> Dict[str, Any]:
        return {
            "window_start": time.time(),
            "seen_tracks": set(),
            "class_tracks": defaultdict(set),
            "frame_counts": [],
            "frame_occupancies": [],
            "frame_speeds": [],
            "frame_cong": [],
            "frame_bps": [],
            "crossing_mode": False,          # the caller reports counting-line crossings (see record_frame)
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
        Extract cyclic temporal encodings essential for LSTM convergence.
        Returns: (hour, minute, day_of_week, is_weekend, sin_hour, cos_hour, sin_dow, cos_dow)
        """
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
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
        traffic_bps: List[float] = []        # recent body-sizes/s of moving-stream vehicles
        current_frame_stopped = 0
        static_in_frame = 0

        with self._lock:
            cam_tracks = self._camera_tracks[camera_id]
            win = self._window_state[camera_id]

            if not win["frame_counts"]:
                win["window_start"] = timestamp

            for det in detections:
                tid = det["track_id"]
                cid = det.get("class_id", 2)
                cname = det.get("class_name", "vehicle")
                x1, y1, x2, y2 = det["box"]

                # Volume from counting-line crossings when the caller reports them (a detection
                # carries ``counted_now`` = this track crossed the line in this frame).
                if "counted_now" in det:
                    win["crossing_mode"] = True
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
                track.is_static = (not moved) and (timestamp - track.first_seen) >= self.static_after_s

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
                if not moved:
                    track.is_stopped = False    # undetermined (young and still): left out
                    continue

                # Part of the moving stream.
                traffic_boxes.append((x1, y1, x2, y2))
                current_frame_speeds.append(smooth_speed)
                current_frame_dxs.append(track.dx_px_s)
                current_frame_dys.append(track.dy_px_s)
                if track.recent_bps is not None:
                    traffic_bps.append(track.recent_bps)
                    track.is_stopped = track.recent_bps < self.stop_bodies_per_s
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
            avg_speed_frame = float(np.mean(current_frame_speeds)) if current_frame_speeds else 0.0
            free_flow = self._free_flow_bps.get(camera_id)
            median_bps = float(np.median(traffic_bps)) if traffic_bps else 0.0
            frame_cong = self._congestion_index(
                coverage=frame_occupancy, speeds=traffic_bps,
                stopped=sum(1 for v in traffic_bps if v < self.stop_bodies_per_s), free_flow=free_flow)
            recent = self._cong_recent[camera_id]
            recent.append((timestamp, frame_cong))
            while recent and timestamp - recent[0][0] > LIVE_SPAN_S:
                recent.popleft()
            congestion_index = round(float(np.clip(np.mean([c for _, c in recent]), 0.0, 1.0)), 3)

            # Accumulate in current window
            win["last_ts"] = timestamp
            win["camera_name"] = camera_name
            win["frame_counts"].append(total_in_frame)
            win["frame_occupancies"].append(frame_occupancy)
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
                self._flush_window_locked(camera_id, camera_name, timestamp, elapsed)

            # Store live instantaneous snapshot
            live_snapshot = {
                "camera_id": camera_id,
                "camera_name": camera_name,
                "timestamp": timestamp,
                "active_total": total_in_frame,
                "active_counts": dict(active_counts),
                "traffic_vehicles": len(traffic_boxes),
                "parked_vehicles": static_in_frame,
                "occupancy_ratio": round(frame_occupancy, 4),
                "occupancy_percent": round(frame_occupancy * 100.0, 1),
                "avg_speed_px_s": round(avg_speed_frame, 1),
                "median_speed_bodies_s": round(median_bps, 2),
                "free_flow_bodies_s": round(free_flow, 2) if free_flow else None,
                "stopped_vehicles": current_frame_stopped,
                "congestion_index": congestion_index,
                "congestion_label": self.get_congestion_label(congestion_index),
                "unique_window_volume": len(win["seen_tracks"]),
                "window_volume": win["crossings"] if win["crossing_mode"] else len(win["seen_tracks"]),
                "volume_basis": "crossings" if win["crossing_mode"] else "tracks",
                "window_elapsed_s": round(elapsed, 1),
            }
            self._live_snapshot[camera_id] = live_snapshot
            return live_snapshot

    # ── Congestion model ──────────────────────────────────────────────────────

    @staticmethod
    def _coverage(boxes: Sequence[Tuple[float, float, float, float]], fw: int, fh: int) -> float:
        """Share of the frame covered by the union of ``boxes`` (overlaps counted once)."""
        if not boxes or fw <= 0 or fh <= 0:
            return 0.0
        grid = np.zeros((GRID_H, GRID_W), dtype=bool)
        for x1, y1, x2, y2 in boxes:
            c1 = int(np.clip(x1 / fw * GRID_W, 0, GRID_W))
            c2 = int(np.clip(math.ceil(x2 / fw * GRID_W), 0, GRID_W))
            r1 = int(np.clip(y1 / fh * GRID_H, 0, GRID_H))
            r2 = int(np.clip(math.ceil(y2 / fh * GRID_H), 0, GRID_H))
            if c2 > c1 and r2 > r1:
                grid[r1:r2, c1:c2] = True
        return float(grid.mean())

    @staticmethod
    def _congestion_index(coverage: float, speeds: Sequence[float], stopped: int,
                          free_flow: Optional[float]) -> float:
        """
        Congestion in [0, 1] from the moving-stream vehicles only.

        - queue (0.40): share of them standing still (full at ``STOPPED_FULL``);
        - slowness (0.45): how far their median speed is below the camera's own free-flow speed
          (none at ``SLOW_ABOVE`` of it, full at ``SLOW_FULL_BELOW``); until the camera has a learned
          free-flow speed only a standstill counts as slow;
        - density (0.15): how full the frame is, but only counted when traffic is also slow, so a
          busy road whose traffic flows freely is not "congested".
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
        if not self._free_flow_path or time.time() - self._free_flow_saved < min_interval_s:
            return
        self._free_flow_saved = time.time()
        try:
            os.makedirs(os.path.dirname(os.path.abspath(self._free_flow_path)), exist_ok=True)
            tmp = self._free_flow_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump({str(k): round(v, 4) for k, v in self._free_flow_bps.items()}, f)
            os.replace(tmp, self._free_flow_path)
        except OSError:
            pass   # best effort: it is relearned from traffic anyway

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
            self._cong_recent.pop(camera_id, None)
            if not win or not win["frame_counts"]:
                self._window_state[camera_id] = self._new_window_state()
                return None
            observed = win["last_ts"] - win["window_start"]
            if observed < min_observed_s:
                self._window_state[camera_id] = self._new_window_state()
                return None
            return self._flush_window_locked(
                camera_id, win.get("camera_name") or f"Camera {camera_id}", win["last_ts"], observed
            )

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

    def _flush_window_locked(
        self,
        camera_id: int,
        camera_name: str,
        now: float,
        elapsed: float
    ) -> Optional[TimeWindowMetrics]:
        """Aggregate current window, persist to DB, and rotate window."""
        win = self._window_state[camera_id]
        if not win["frame_counts"]:
            return None

        if win["crossing_mode"]:
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

        # Learn this camera's free-flow speed from the speeds its traffic actually shows.
        samples = win["frame_bps"]
        if len(samples) >= MIN_REF_SAMPLES:
            p = max(float(np.percentile(samples, REF_PERCENTILE)), MIN_FREE_FLOW_BPS)
            learned = self._free_flow_bps.get(camera_id)
            self._free_flow_bps[camera_id] = p if learned is None else 0.9 * learned + 0.1 * p
            self._save_free_flow()

        # Temporal features
        hour, minute, dow, is_wknd, sin_h, cos_h, sin_d, cos_d = self.compute_temporal_features(now)
        dt_iso = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()

        metrics = TimeWindowMetrics(
            camera_id=camera_id,
            camera_name=camera_name,
            timestamp=now,
            timestamp_iso=dt_iso,
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
            volume_basis="crossings" if win["crossing_mode"] else "tracks",
        )

        # Persist to SQLite
        try:
            self.db.insert_metrics(metrics)
        except Exception as exc:
            print(f"[analytics] DB write error: {exc}")

        # Add to in-memory trend history for dashboard
        self._history_60s[camera_id].append(metrics.to_dict())

        # Reset window accumulator
        self._window_state[camera_id] = self._new_window_state()
        self._window_state[camera_id]["window_start"] = now

        return metrics

    # ── Real-time query methods for API & Dashboard ───────────────────────────

    def get_live_metrics(self, camera_id: int) -> Dict[str, Any]:
        """Get live instantaneous metrics and rolling window preview."""
        with self._lock:
            live = self._live_snapshot.get(camera_id, {})
            win = self._window_state.get(camera_id)

            if win:
                crossing_mode = win["crossing_mode"]
                vol = win["crossings"] if crossing_mode else len(win["seen_tracks"])
                elapsed = max(time.time() - win["window_start"], 1.0)
                rolling_vpm = round((vol / elapsed) * 60.0, 1)
                class_breakdown = (dict(win["class_crossings"]) if crossing_mode
                                   else {k: len(v) for k, v in win["class_tracks"].items()})
            else:
                crossing_mode = False
                vol = 0
                rolling_vpm = 0.0
                class_breakdown = {}

            history = list(self._history_60s[camera_id])

            return {
                "live": live,
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
        recent = self.db.query_timeseries(camera_id=camera_id, window_seconds=60, limit=20, order_asc=False)

        avg_vpm = round(float(np.mean([r["flow_rate_vpm"] for r in recent])), 2) if recent else 0.0
        avg_speed = round(float(np.mean([r["avg_speed_px_s"] for r in recent])), 2) if recent else 0.0
        avg_cong = round(float(np.mean([r["congestion_index"] for r in recent])), 3) if recent else 0.0

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
        window_seconds: int = 60,
        start_ts: Optional[float] = None,
        end_ts: Optional[float] = None,
        limit: int = 5000,
    ) -> List[Dict[str, Any]]:
        """
        Extract clean time-series observations ready for analysis or model training.
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
        window_seconds: int = 60,
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
                "sin_hour", "cos_hour", "sin_dow", "cos_dow", "volume_basis"
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
            row_features = [float(r.get(c, 0.0)) for c in feature_cols]
            matrix.append(row_features)
            target_vals.append(float(r.get(target_col, 0.0)))
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
