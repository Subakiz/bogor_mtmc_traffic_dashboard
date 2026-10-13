#!/usr/bin/env python3
"""
FastAPI Server for BSW Kota Bogor MTMC Traffic Intelligence Dashboard
Integrates:
- All 94 CCTV cameras with real-time stream links
- Road network topology (downstream scheduling, junction clusters)
- SQLite WAL vehicle sighting registry & counting-line volume
- Background ANE indexing daemon control
- Interactive Leaflet map & HLS video player frontend
"""

import os
import sys
import json
import time
import asyncio
import functools
import threading
from typing import Optional, List, Dict, Any, Tuple
from pathlib import Path

# Maximize file descriptor limit to avoid Errno 24 with CoreML / OpenCV / HLS sockets
try:
    import resource
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target = min(10240, hard if hard > 0 else 10240)
    if soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))
except Exception:
    pass

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from starlette.middleware.gzip import DEFAULT_EXCLUDED_CONTENT_TYPES, GZipMiddleware
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import (
    PROJECT_ROOT, DATA_DIR, FRONTEND_DIR, ENGINE_ROOT, ANALYTICS_DB_PATH, DEFAULT_MODEL_PATH,
    COUNTING_LINES_PATH, FREE_FLOW_PATH, TRAFFIC_LANES_PATH,
    MAINTENANCE_INTERVAL_S, STATS_CACHE_TTL_S, ensure_engine_importable,
)
from backend.database import Database, DEFAULT_DB_PATH
from backend.topology_engine import get_default_engine
from backend.counting import CountingLines
from backend.maintenance import SightingMaintenance
from backend.rolling_tracker import RollingTracker
from backend.detector import ANEVehicleDetector, ALLOWED_VEHICLE_CLASSES

app = FastAPI(title="BSW Kota Bogor Traffic Intelligence Dashboard", version="1.0.0")

class _RevalidatePageAssets:
    """The page and its static assets are revalidated on every load (``no-cache``: the browser
    sends If-None-Match and gets a 304 when nothing changed) instead of being re-downloaded.
    Plain ASGI on purpose: ``@app.middleware("http")`` (BaseHTTPMiddleware) wraps every request,
    including each frame poll, and halved the server's request throughput."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "") if scope["type"] == "http" else ""
        if not (path == "/" or path.startswith("/static/")):
            return await self.app(scope, receive, send)

        async def send_revalidate(message):
            if message["type"] == "http.response.start":
                headers = [(k, v) for k, v in message.get("headers", []) if k.lower() != b"cache-control"]
                message["headers"] = headers + [(b"cache-control", b"no-cache")]
            await send(message)

        await self.app(scope, receive, send_revalidate)


app.add_middleware(_RevalidatePageAssets)
# JSON (topology, cameras, counts) compresses ~8x; JPEG frames and the MJPEG stream are left alone.
app.add_middleware(GZipMiddleware, minimum_size=1000, compresslevel=5,
                   exclude_content_types=DEFAULT_EXCLUDED_CONTENT_TYPES + ("multipart/x-mixed-replace",))

# Mount frontend static directory
FRONTEND_DIR.mkdir(exist_ok=True)

# Shared engines
db = Database(str(DEFAULT_DB_PATH))
topology = get_default_engine()
maintenance = SightingMaintenance(db, MAINTENANCE_INTERVAL_S)


class _TTLCache:
    """Computes each keyed value at most once per ``ttl`` seconds, however many callers ask.
    Concurrent callers for a stale key wait for one computation instead of each running the
    (full-table) aggregate queries themselves. Cached values are shared: never mutate them."""

    _MAX_KEYS = 32

    def __init__(self, ttl: float):
        self.ttl = ttl
        self._lock = threading.Lock()
        self._entries: Dict[Any, Tuple[float, Any]] = {}
        self._key_locks: Dict[Any, threading.Lock] = {}

    def get(self, key: Any, fn):
        if self.ttl <= 0:
            return fn()
        with self._lock:
            entry = self._entries.get(key)
            if entry and entry[0] > time.monotonic():
                return entry[1]
            key_lock = self._key_locks.setdefault(key, threading.Lock())
        with key_lock:
            with self._lock:
                entry = self._entries.get(key)
                if entry and entry[0] > time.monotonic():
                    return entry[1]
            value = fn()
            with self._lock:
                if len(self._entries) >= self._MAX_KEYS:   # arbitrary query params must not grow it forever
                    self._entries.clear()
                    self._key_locks.clear()
                self._entries[key] = (time.monotonic() + self.ttl, value)
            return value

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_stats_cache = _TTLCache(STATS_CACHE_TTL_S)

ensure_engine_importable()

# Per-camera traffic analytics (car_tracker engine)
try:
    from car_tracker import TrafficAnalyticsEngine
    _analytics_engine = TrafficAnalyticsEngine(
        db_path=ANALYTICS_DB_PATH, free_flow_path=FREE_FLOW_PATH, lanes_path=TRAFFIC_LANES_PATH,
        assume_crossings=True)   # the rolling tracker always reports counting-line crossings
except Exception as _e:
    print(f"Warning: TrafficAnalyticsEngine unavailable: {_e}")
    _analytics_engine = None

def _camera_or_none(camera_id: int) -> Optional[Dict[str, Any]]:
    """TopologyEngine.get_camera_info raises KeyError for unknown ids."""
    try:
        return topology.get_camera_info(camera_id)
    except KeyError:
        return None


def _camera_or_404(camera_id: int) -> Dict[str, Any]:
    cam = _camera_or_none(camera_id)
    if cam is None:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return cam

@app.on_event("startup")
def startup_event():
    from backend.database import seed_cameras_from_topology
    seed_cameras_from_topology(db.conn, str(DATA_DIR / "bogor_camera_topology.json"))

    # One-time: switch to incremental auto-vacuum and reclaim the pages left ~90% empty by the
    # appearance blobs older versions stored (a no-op once done). Runs before the tracker writes.
    result = db.compact()
    if result.get("compacted"):
        print(f"Database compacted: {result['before_bytes'] / 1e6:.1f} MB -> "
              f"{result['after_bytes'] / 1e6:.1f} MB in {result['seconds']:.1f}s.")
    elif result.get("reason") not in (None, "already compacted"):
        print(f"Database compaction skipped: {result.get('reason')}")

    # Rolling tracker: tracks the viewed camera and rolls through every other camera so
    # vehicles are counted citywide. Disable the rolling part with ROLLING_ENABLED=0.
    if rolling_tracker.rolling_enabled:
        rolling_tracker.start()
        print(f"Rolling tracker active across {len(rolling_tracker.cameras)} cameras.")
    maintenance.start()

@app.on_event("shutdown")
def shutdown_event():
    maintenance.stop()
    rolling_tracker.stop()
    if _analytics_engine is not None:
        _analytics_engine.close()

@app.api_route("/", methods=["GET", "HEAD"])
def get_dashboard():
    index_file = FRONTEND_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return HTMLResponse("<h1>Dashboard loading...</h1>")

@app.get("/api/cameras")
def list_cameras():
    """Returns all 94 BSW CCTV cameras with metadata, coordinates, and stream URLs."""
    cameras = topology.get_all_cameras()
    return {
        "total": len(cameras),
        "cameras": cameras
    }

def _citywide_vph(rates: Dict[int, Dict[str, Any]]) -> float:
    """Sum of vehicles/hour over junctions, counting each junction once (its busiest camera):
    cameras at one junction watch the same vehicles, so adding them would double count."""
    best: Dict[Tuple[int, ...], float] = {}
    for cid, r in rates.items():
        cluster = tuple(topology.get_intra_junction_cluster(cid)) if cid in topology.cameras else (cid,)
        best[cluster] = max(best.get(cluster, 0.0), r["vph"])
    return round(sum(best.values()), 1)


@app.get("/api/cameras/counts")
def get_camera_counts(window_minutes: Optional[float] = None, rate_window_minutes: float = 60.0):
    """
    Returns live vehicle counts across all BSW CCTV cameras in Kota Bogor.
    Supports optional temporal filtering via window_minutes (e.g. 5, 15, 60 min).
    Traffic volume comes as rates (``camera_rates``): vehicles/hour from counting-line crossings
    over the time each camera was actually observed in the last ``rate_window_minutes``.
    Cached for a few seconds (STATS_CACHE_TTL_S) and shared by all clients.
    """
    return _stats_cache.get(("counts", window_minutes, rate_window_minutes),
                            lambda: _compute_camera_counts(window_minutes, rate_window_minutes))


def _compute_camera_counts(window_minutes: Optional[float], rate_window_minutes: float) -> Dict[str, Any]:
    window_s = (window_minutes * 60.0) if window_minutes and window_minutes > 0 else None
    counts = db.get_camera_vehicle_counts(window_s=window_s)
    total_vehicles = sum(c["total_sightings"] for c in counts)
    total_unique = sum(c["unique_vehicles"] for c in counts)
    active_cameras_count = sum(1 for c in counts if c["total_sightings"] > 0)

    class_totals = {"car": 0, "motorcycle": 0, "truck": 0, "bus": 0}
    for c in counts:
        for k, v in c.get("classes", {}).items():
            if k in class_totals:
                class_totals[k] += v

    camera_counts_map = {c["camera_id"]: c["total_sightings"] for c in counts}
    top_cameras = [c for c in counts if c["total_sightings"] > 0][:10]

    rates = db.get_camera_rates(window_s=max(1.0, rate_window_minutes) * 60.0)
    for c in counts:
        c["rate"] = rates.get(c["camera_id"])

    return {
        "total_cameras": len(counts),
        "cameras_with_detections": active_cameras_count,
        "total_vehicles_counted": total_vehicles,
        "total_unique_vehicles": total_unique,
        "class_breakdown": class_totals,
        "top_cameras": top_cameras,
        "camera_counts": counts,
        "camera_counts_map": camera_counts_map,
        "camera_rates": rates,
        "camera_vph_map": {cid: r["vph"] for cid, r in rates.items()},
        "cameras_with_rates": len(rates),
        "citywide_vph": _citywide_vph(rates),
        "rate_window_minutes": rate_window_minutes,
        "timestamp": time.time(),
    }

@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: int):
    return _camera_or_404(camera_id)

@app.get("/api/topology")
def get_topology():
    """Returns all camera nodes and directed road transitions for map visualization."""
    return {
        "nodes": list(topology.cameras.values()),
        "edges": topology.edges,
        "corridors": topology.corridors,
        "total_nodes": len(topology.cameras),
        "total_edges": len(topology.edges)
    }

def _cached_stats() -> Dict[str, Any]:
    return _stats_cache.get("stats", db.get_traffic_stats)


@app.get("/api/stats")
def get_stats():
    """Returns live citywide traffic metrics and unique vehicle counts across all 94 cameras."""
    return {
        "stats": _cached_stats(),
        "timestamp": time.time()
    }

def _latest_sightings_cached(limit: int) -> List[Dict[str, Any]]:
    return _stats_cache.get(("latest", limit), lambda: db.get_latest_sightings(limit=limit))


@app.get("/api/recent")
def recent_sightings(limit: int = 50, camera_id: Optional[int] = None):
    """Returns recent vehicle sightings across all cameras or a specific camera."""
    if camera_id is None:
        sightings = _latest_sightings_cached(limit)
    else:
        sightings = db.get_recent_sightings(camera_id=camera_id, limit=limit)
    return {
        "count": len(sightings),
        "sightings": sightings
    }

@app.get("/api/daemon/status")
def daemon_status():
    """Rolling multi-camera tracker status (focus camera, current visits, coverage, counts)."""
    status = rolling_tracker.get_status()
    status.update({
        "mode": "Apple Neural Engine (ANE)",
        "model": os.path.basename(DEFAULT_MODEL_PATH),
        "device": "CPU_AND_NE",
    })
    return status

@app.post("/api/daemon/start")
def start_daemon():
    """Enables rolling coverage of all cameras (the viewed camera is always tracked)."""
    already = rolling_tracker.running and rolling_tracker.rolling_enabled
    rolling_tracker.set_rolling_enabled(True)
    if already:
        return {"status": "already_running", "message": "Rolling tracker is already running"}
    return {"status": "started", "message": "Rolling multi-camera tracking active"}

@app.post("/api/daemon/stop")
def stop_daemon():
    """Stops rolling coverage; the viewed camera keeps being tracked."""
    rolling_tracker.set_rolling_enabled(False)
    return {"status": "stopped", "message": "Rolling coverage stopped"}

def _heartbeat_payload() -> Dict[str, Any]:
    """What the dashboard refreshes on each tick: ``counts`` / ``recent``, which it would
    otherwise fetch with two more requests. (It used to also carry all-time stats, analytics
    summaries and rolling status that the page never read, at ~4x the size and DB work.)"""
    return {
        "type": "heartbeat",
        "timestamp": time.time(),
        "counts": get_camera_counts(),
        "recent": _latest_sightings_cached(12),
    }


HEARTBEAT_INTERVAL_S = 2.0
_heartbeat_cache = _TTLCache(HEARTBEAT_INTERVAL_S / 2)


def _heartbeat_text() -> str:
    """The heartbeat serialized once and shared by every open socket."""
    return _heartbeat_cache.get("heartbeat", lambda: json.dumps(_heartbeat_payload(), default=str))


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Periodic live stats; the DB work and serialization run off the event loop.
            await websocket.send_text(await asyncio.to_thread(_heartbeat_text))
            await asyncio.sleep(HEARTBEAT_INTERVAL_S)
    except (WebSocketDisconnect, RuntimeError):
        pass   # client went away (RuntimeError: send on a socket that already closed)


# ── Per-Camera Traffic Analytics & Time-Series Endpoints (LSTM / ML Ready) ────
@app.get("/api/stats/time-series")
def get_traffic_time_series(
    camera_id: Optional[int] = None,
    window_seconds: int = 60,
    limit: int = 100,
    start_time: Optional[float] = None,
    end_time: Optional[float] = None,
):
    if _analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    target_cam = None if camera_id in (-1, None) else camera_id
    return _analytics_engine.export_timeseries_dataset(
        camera_id=target_cam,
        window_seconds=window_seconds,
        start_ts=start_time,
        end_ts=end_time,
        limit=limit,
    )


@app.get("/api/stats/export")
def export_traffic_dataset(
    format: str = "csv",
    camera_id: Optional[int] = None,
    window_seconds: int = 60,
    limit: int = 5000,
):
    if _analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    target_cam = None if camera_id in (-1, None) else camera_id

    fmt = format.lower()
    if fmt == "csv":
        csv_text = _analytics_engine.export_csv_str(
            camera_id=target_cam,
            window_seconds=window_seconds,
            limit=limit,
        )
        cam_suffix = str(target_cam) if target_cam is not None else "all"
        filename = f"cctv_camera_{cam_suffix}_traffic_timeseries.csv"
        return Response(
            content=csv_text,
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"}
        )
    elif fmt == "json":
        return _analytics_engine.export_timeseries_dataset(
            camera_id=target_cam,
            window_seconds=window_seconds,
            limit=limit,
        )
    else:
        raise HTTPException(400, detail=f"Unsupported format '{format}'. Supported: 'csv', 'json'.")


FLOW_MAX_MINUTES = 30 * 24 * 60.0   # the Analytics chart's longest timeframe (30 days)
FLOW_MAX_BUCKETS = 720               # bucket size is raised so a series never has more points


def _flow_summary(camera_id: int, minutes: float, bucket_minutes: float) -> Dict[str, Any]:
    """Counting-line flow of one camera: the headline rate over ``minutes`` and a bucketed series
    (flow in veh/h per bucket, with the mean congestion index of the same bucket).
    ``history_start_ts`` is when the camera's stored data begins (None if it has none), so the
    dashboard can tell which timeframes the data actually covers."""
    minutes = min(max(1.0, minutes), FLOW_MAX_MINUTES)
    bucket_minutes = min(max(1.0, bucket_minutes, minutes / FLOW_MAX_BUCKETS), minutes)
    rate = db.get_camera_rates(window_s=minutes * 60.0, camera_ids=[camera_id]).get(camera_id)
    series = db.get_camera_flow_series(camera_id, minutes=minutes, bucket_minutes=bucket_minutes)
    congestion = {c["t"]: c["congestion"] for c in
                  (_analytics_engine.get_congestion_series(camera_id, minutes, bucket_minutes)
                   if _analytics_engine else [])}
    for point in series:
        point["congestion"] = congestion.pop(point["t"], None)
    # Buckets with a congestion reading but no counted observation still belong on the chart.
    series.extend({"t": t, "t_end": t + bucket_minutes * 60.0, "windows": 0, "observed_s": 0.0,
                   "vehicles": 0, "vph": None, "low_sample": True, "congestion": c} for t, c in congestion.items())
    series.sort(key=lambda p: p["t"])
    starts = [ts for ts in (db.get_camera_first_observed(camera_id),
                            _analytics_engine.get_first_timestamp(camera_id) if _analytics_engine else None)
              if ts is not None]
    return {
        "basis": "counting-line crossings",
        "window_minutes": minutes,
        "bucket_minutes": bucket_minutes,
        "history_start_ts": min(starts) if starts else None,
        "rate": rate,
        "windows": rate["windows"] if rate else 0,
        "series": series,
    }


@app.get("/api/cameras/{camera_id}/metrics")
def get_camera_traffic_metrics(camera_id: int, minutes: float = 60.0, bucket_minutes: float = 5.0):
    if _analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    cam = _camera_or_none(camera_id)
    cam_name = cam.get("name", f"Camera #{camera_id}") if cam else f"Camera #{camera_id}"
    metrics = _analytics_engine.get_live_metrics(camera_id)
    metrics["camera_name"] = cam_name
    metrics["camera_id"] = camera_id
    metrics["flow"] = _flow_summary(camera_id, minutes, bucket_minutes)
    metrics["rate"] = metrics["flow"]["rate"] if minutes == 60.0 else db.get_camera_rates(window_s=3600.0, camera_ids=[camera_id]).get(camera_id)
    line = rolling_tracker.counting.line(camera_id)
    metrics["counting_line"] = None if line is None else {
        "endpoints": line.endpoints(), "direction_labels": line.labels(),
    }
    return metrics


@app.get("/api/stats/summary")
def get_traffic_summary():
    if _analytics_engine is None:
        raise HTTPException(status_code=503, detail="Analytics engine not initialized")
    return _analytics_engine.get_all_camera_summaries()


# Shared ANE detector (independent ByteTrack per camera). Only the rolling tracker's worker
# thread runs inference; request handlers render the frames it produced.
_detector: Optional[ANEVehicleDetector] = None
_detector_lock = threading.Lock()

def get_detector() -> ANEVehicleDetector:
    global _detector
    with _detector_lock:
        if _detector is None:
            _detector = ANEVehicleDetector()
        return _detector

rolling_tracker = RollingTracker(
    db=db, topology=topology, analytics=_analytics_engine, detector_factory=get_detector,
    counting=CountingLines(COUNTING_LINES_PATH),
)

def _draw_counting_overlay(img: np.ndarray, latest: Dict[str, Any]) -> None:
    """Counting line (once learned) and this session's crossing counts, bottom-left."""
    h, w = img.shape[:2]
    line, labels = latest.get("line"), latest.get("line_labels")
    if line:
        (x1, y1), (x2, y2) = line
        cv2.line(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 230, 255), 2, cv2.LINE_AA)
    rows = [f"COUNTED {latest.get('counted_total', 0)}"]
    if labels:
        rows.append(f"{labels[0]}: {latest.get('counted_pos', 0)}   {labels[1]}: {latest.get('counted_neg', 0)}")
    else:
        rows.append("learning counting line...")
    bw, bh = 330, 18 * len(rows) + 12
    cv2.rectangle(img, (15, h - bh - 15), (15 + bw, h - 15), (15, 15, 18), -1)
    cv2.rectangle(img, (15, h - bh - 15), (15 + bw, h - 15), (0, 230, 255), 1)
    for i, text in enumerate(rows):
        cv2.putText(img, text, (25, h - bh - 15 + 22 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.46 if i == 0 else 0.4, (0, 240, 255) if i == 0 else (210, 210, 210), 1, cv2.LINE_AA)


MAX_FRAME_WIDTH = 1280      # annotated frames are downscaled to at most this width


def _render_annotated_frame(frame: Any, detections: List[Dict[str, Any]], ane_ms: float,
                            latest: Optional[Dict[str, Any]] = None) -> bytes:
    """Draws the rolling tracker's detections for one focus frame and encodes it as JPEG.
    Boxes are labelled by vehicle class and confidence only; no track or vehicle ids are drawn."""
    annotated = frame.copy()
    if latest is not None:
        _draw_counting_overlay(annotated, latest)

    for det in detections or []:
        cls_name = ALLOWED_VEHICLE_CLASSES.get(int(det["class_id"]), "vehicle")
        score = det["confidence"]
        x1, y1, x2, y2 = (int(v) for v in det["bbox"])

        color = (
            (0, 255, 0) if cls_name == "car"
            else ((255, 170, 0) if cls_name == "motorcycle"
            else ((0, 165, 255) if cls_name == "truck" else (255, 0, 230)))
        )
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 4 if det.get("counted") else 2)

        badge_text = f" {cls_name.upper()} {score:.2f} "
        (tw, th), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        by = max(y1 - 18, 0)
        cv2.rectangle(annotated, (x1, by), (x1 + tw + 6, by + 17), color, -1)
        cv2.putText(annotated, badge_text, (x1 + 3, by + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 1, cv2.LINE_AA)

    # ANE Performance HUD
    cv2.rectangle(annotated, (15, 15), (330, 48), (15, 15, 18), -1)
    cv2.rectangle(annotated, (15, 15), (330, 48), (0, 215, 255), 1)
    cv2.putText(annotated, f"ANE: {ane_ms:.1f}ms | YOLOv8 CoreML", (25, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 240, 255), 1)

    # 1080p cameras made ~440 KB frames (~4 MB/s per viewer at 10 fps); 1280 px wide is what the
    # player shows and halves both the size and the encode time.
    h, w = annotated.shape[:2]
    if w > MAX_FRAME_WIDTH:
        annotated = cv2.resize(annotated, (MAX_FRAME_WIDTH, round(h * MAX_FRAME_WIDTH / w)),
                               interpolation=cv2.INTER_AREA)
    ret_jpg, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes() if ret_jpg else b""

@functools.lru_cache(maxsize=256)
def _generate_placeholder_frame(camera_name: str, message: str) -> bytes:
    import numpy as np
    img = np.zeros((720, 1280, 3), dtype=np.uint8)
    img[:] = (20, 20, 24)
    for y in range(0, 720, 60):
        cv2.line(img, (0, y), (1280, y), (30, 30, 36), 1)
    for x in range(0, 1280, 80):
        cv2.line(img, (x, 0), (x, 720), (30, 30, 36), 1)

    cv2.rectangle(img, (380, 280), (900, 440), (35, 35, 45), -1)
    cv2.rectangle(img, (380, 280), (900, 440), (0, 180, 255), 2)
    cv2.putText(img, "BSW KOTA BOGOR CCTV", (410, 325),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 220, 255), 2)
    cv2.putText(img, str(camera_name)[:38], (410, 365),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1)
    cv2.putText(img, message, (410, 405),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 100), 1)
    ret_jpg, jpeg = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes() if ret_jpg else b""

# Rendered JPEG of the newest focus frame, shared by every viewer of that camera, keyed by the
# tracker's frame_no (a reopened capture restarts its own seq at 1, which served a stale frame).
_render_cache: Dict[int, Tuple[int, bytes]] = {}
_render_lock = threading.Lock()

FRAME_WAIT_S = 1.0         # a frame request with ?after= waits this long for a newer frame
FRAME_POLL_S = 0.02
VIEWER_LIMIT_MESSAGE = "VIEWER LIMIT: OTHER CAMERAS ARE BEING WATCHED"
_NO_STORE = {"Cache-Control": "no-store"}


def _render_focus_jpeg(camera_id: int, latest: Dict[str, Any]) -> bytes:
    with _render_lock:
        cached = _render_cache.get(camera_id)
        if cached is not None and cached[0] == latest["frame_no"]:
            return cached[1]
    jpeg_bytes = _render_annotated_frame(latest["frame"], latest["detections"], latest["ane_ms"], latest)
    if jpeg_bytes:
        with _render_lock:
            _render_cache[camera_id] = (latest["frame_no"], jpeg_bytes)
    return jpeg_bytes


@app.get("/api/stream/{camera_id}/frame")
async def get_single_annotated_frame(camera_id: int, after: Optional[int] = None):
    """Newest annotated JPEG of a watched camera; polling it keeps the camera in focus.

    Pass the previous response's ``X-Frame-No`` as ``after``: the request then waits up to
    FRAME_WAIT_S for a newer frame and answers 204 if none came, so a viewer never downloads
    (and the server never re-sends) the same frame twice -- plain polling re-sent ~70% of them.
    """
    cam = _camera_or_404(camera_id)
    cam_name = cam.get("name", f"Camera #{camera_id}")
    if not cam.get("stream_url"):
        return Response(content=_generate_placeholder_frame(cam_name, "NO STREAM URL CONFIGURED"),
                        media_type="image/jpeg", headers=_NO_STORE)

    loop = asyncio.get_running_loop()
    deadline = loop.time() + FRAME_WAIT_S
    while True:
        if not rolling_tracker.set_focus(camera_id):
            return Response(content=_generate_placeholder_frame(cam_name, VIEWER_LIMIT_MESSAGE),
                            media_type="image/jpeg", headers={**_NO_STORE, "X-Frame-State": "viewer-limit"})
        latest = rolling_tracker.get_focus_frame(camera_id)
        if latest is not None and latest["frame_no"] != after:
            jpeg_bytes = await asyncio.to_thread(_render_focus_jpeg, camera_id, latest)
            if jpeg_bytes:
                return Response(content=jpeg_bytes, media_type="image/jpeg",
                                headers={**_NO_STORE, "X-Frame-No": str(latest["frame_no"])})
        if loop.time() >= deadline:
            if latest is None:
                with _render_lock:
                    _render_cache.pop(camera_id, None)
                return Response(content=_generate_placeholder_frame(cam_name, "CONNECTING HLS STREAM..."),
                                media_type="image/jpeg", headers={**_NO_STORE, "X-Frame-State": "connecting"})
            return Response(status_code=204, headers=_NO_STORE)
        await asyncio.sleep(FRAME_POLL_S)

# ── Data export for further analysis (backend/dataset.py; CLI: scripts/export_dataset.py) ──────
EXPORT_MAX_DAYS = 31
_EXPORT_MEDIA = {"csv": "text/csv; charset=utf-8", "jsonl": "application/x-ndjson"}


def _export_range(start: Optional[str], end: Optional[str]) -> Tuple[float, float]:
    from backend import dataset
    try:
        end_ts = dataset.parse_time(end) if end else time.time()
        start_ts = dataset.parse_time(start) if start else end_ts - 86400.0
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    if end_ts <= start_ts:
        raise HTTPException(status_code=400, detail="end must be after start")
    if end_ts - start_ts > EXPORT_MAX_DAYS * 86400:
        raise HTTPException(status_code=400, detail=f"range is limited to {EXPORT_MAX_DAYS} days; "
                                                    "use scripts/export_dataset.py for longer exports")
    return start_ts, end_ts


def _export_cameras(camera_ids: Optional[str]) -> Optional[List[int]]:
    if not camera_ids:
        return None
    try:
        return [int(c) for c in camera_ids.split(",") if c.strip()]
    except ValueError:
        raise HTTPException(status_code=400, detail="camera_ids must be comma-separated integers")


def _export_response(chunks, fmt: str, name: str, start_ts: float, end_ts: float) -> StreamingResponse:
    from backend import dataset
    stamp = lambda ts: dataset.to_local_iso(ts)[:16].replace(":", "")   # noqa: E731
    filename = f"{name}_{stamp(start_ts)}_{stamp(end_ts)}.{fmt}"
    return StreamingResponse(chunks, media_type=_EXPORT_MEDIA[fmt],
                             headers={"Content-Disposition": f'attachment; filename="{filename}"',
                                      "Cache-Control": "no-store"})


@app.get("/api/export/dataset")
def export_dataset(start: Optional[str] = None, end: Optional[str] = None, bucket_minutes: float = 15,
                   camera_ids: Optional[str] = None, format: str = "csv", layout: str = "long",
                   fill_gaps: bool = False, value: str = "vph"):
    """Analysis-ready per-camera time series: fixed WIB-aligned buckets with counting-line volume
    (prorated by overlap), observed seconds/coverage and congestion metrics. ``start``/``end`` are
    ISO (no offset = WIB) or epoch seconds; default the last 24 h. Columns: /api/export/dictionary."""
    from backend import dataset
    if format not in _EXPORT_MEDIA:
        raise HTTPException(status_code=400, detail="format must be csv or jsonl (Parquet: scripts/export_dataset.py)")
    start_ts, end_ts = _export_range(start, end)
    try:
        stream = dataset.build_dataset(start_ts, end_ts, bucket_minutes, camera_ids=_export_cameras(camera_ids),
                                       fill_gaps=fill_gaps, main_db=db.db_path, analytics_db=ANALYTICS_DB_PATH,
                                       layout=layout, value=value)
        chunks = dataset.iter_text(stream, format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _export_response(chunks, format, f"traffic_{layout}_{int(bucket_minutes)}min", start_ts, end_ts)


@app.get("/api/export/raw")
def export_raw_table(table: str, start: Optional[str] = None, end: Optional[str] = None,
                     camera_ids: Optional[str] = None, format: str = "csv"):
    """Raw rows for the range: observation_windows, sightings, cameras or analytics windows, with a WIB ``*_local`` column next to each UTC time."""
    from backend import dataset
    if format not in _EXPORT_MEDIA:
        raise HTTPException(status_code=400, detail="format must be csv or jsonl")
    start_ts, end_ts = _export_range(start, end)
    try:
        stream = dataset.iter_raw(table, start_ts, end_ts, camera_ids=_export_cameras(camera_ids),
                                  main_db=db.db_path, analytics_db=ANALYTICS_DB_PATH)
        chunks = dataset.iter_text(stream, format)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    return _export_response(chunks, format, table, start_ts, end_ts)


@app.get("/api/export/dictionary")
def export_dictionary(name: str = "dataset"):
    """Column descriptions, units and types of an export (``dataset``, ``dataset_wide`` or a raw table)."""
    from backend import dataset
    try:
        return dataset.data_dictionary(name)
    except (ValueError, KeyError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))


# Mount static frontend assets
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    # HOST=0.0.0.0 (default) serves the dashboard to the local network; HOST=127.0.0.1 keeps it local.
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "8000"))
    print("\n" + "=" * 65)
    print(" Starting BSW Kota Bogor MTMC Dashboard Server")
    print(f" Listening on {host}:{port}")
    print("=" * 65 + "\n")
    uvicorn.run("backend.server:app", host=host, port=port, reload=False)
