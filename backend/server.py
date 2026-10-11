#!/usr/bin/env python3
"""
FastAPI Server for BSW Kota Bogor MTMC Traffic Intelligence Dashboard
Integrates:
- All 94 CCTV cameras with real-time stream links
- Road network topology & predictive camera handoff engine
- SQLite WAL vehicle registry & journey search
- Background ANE indexing daemon control
- Interactive Leaflet map & HLS video player frontend
"""

import os
import sys
import json
import time
import asyncio
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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from pydantic import BaseModel
import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.config import (
    PROJECT_ROOT, DATA_DIR, FRONTEND_DIR, ENGINE_ROOT, ANALYTICS_DB_PATH, DEFAULT_MODEL_PATH,
    EMBEDDING_TTL_S, MAINTENANCE_INTERVAL_S, STATS_CACHE_TTL_S, ensure_engine_importable,
)
from backend.database import Database, DEFAULT_DB_PATH
from backend.topology_engine import get_default_engine
from backend.counting import CountingLines
from backend.maintenance import SightingMaintenance
from backend.rolling_tracker import RollingTracker
from backend.detector import ANEVehicleDetector, ALLOWED_VEHICLE_CLASSES

app = FastAPI(title="BSW Kota Bogor Traffic Intelligence Dashboard", version="1.0.0")

# Anti-cache middleware for local development
@app.middleware("http")
async def add_no_cache_headers(request: Request, call_next):
    response = await call_next(request)
    if request.url.path.startswith("/static/") or request.url.path == "/":
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

# Mount frontend static directory
FRONTEND_DIR.mkdir(exist_ok=True)

# Shared engines
db = Database(str(DEFAULT_DB_PATH))
topology = get_default_engine()
maintenance = SightingMaintenance(db, EMBEDDING_TTL_S, MAINTENANCE_INTERVAL_S)


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
        db_path=ANALYTICS_DB_PATH, free_flow_path=str(DATA_DIR / "free_flow_speeds.json"))
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

# Active WebSocket connections
active_connections: List[WebSocket] = []

class HandoffRequest(BaseModel):
    camera_id: int
    exit_direction: str
    vehicle_id: int
    vehicle_type: Optional[str] = "car"

@app.on_event("startup")
def startup_event():
    from backend.database import seed_cameras_from_topology
    seed_cameras_from_topology(db.conn, str(DATA_DIR / "bogor_camera_topology.json"))

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

def _cached_stats() -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    return _stats_cache.get("stats", lambda: (db.get_traffic_stats(), db.list_active_targets()))


@app.get("/api/stats")
def get_stats():
    """Returns live citywide traffic metrics and unique vehicle counts across all 94 cameras."""
    stats, active_targets = _cached_stats()
    return {
        "stats": stats,
        "active_tracked_targets": active_targets,
        "timestamp": time.time()
    }

def _sanitize_sighting(s: Dict[str, Any]) -> Dict[str, Any]:
    item = dict(s)
    for k in ["embedding", "color_hist", "crop_thumbnail"]:
        if k in item and isinstance(item[k], bytes):
            del item[k]
    return item

def _latest_sightings_cached(limit: int) -> List[Dict[str, Any]]:
    return _stats_cache.get(("latest", limit), lambda: db.get_latest_sightings(limit=limit))


@app.get("/api/recent")
def recent_sightings(limit: int = 50, camera_id: Optional[int] = None):
    """Returns recent vehicle sightings across all cameras or a specific camera."""
    if camera_id is None:
        raw_sightings = _latest_sightings_cached(limit)
    else:
        raw_sightings = db.get_recent_sightings(camera_id=camera_id, limit=limit)
    sightings = [_sanitize_sighting(s) for s in raw_sightings]
    return {
        "count": len(sightings),
        "sightings": sightings
    }

@app.get("/api/vehicles/{vehicle_id}/timeline")
def vehicle_timeline(vehicle_id: int):
    """Returns complete chronological journey across cameras for a specific vehicle ID."""
    journey = db.get_vehicle_journey(vehicle_id)
    if not journey:
        raise HTTPException(status_code=404, detail=f"Vehicle #{vehicle_id} not found in database")
    return [_sanitize_sighting(s) for s in journey]

@app.get("/api/vehicles/search")
def search_vehicle_ids(query: str = ""):
    """Searches vehicles by ID or attributes."""
    results = db.search_vehicles(query)
    clean_results = [_sanitize_sighting(r) for r in results]
    return {"query": query, "results": clean_results}

def _is_removed(camera_id: int) -> bool:
    return (_camera_or_none(camera_id) or {}).get("status", "online") == "removed"


def _live_downstream_candidates(camera_id: int, exit_direction: str, max_hops: int = 3) -> List[Dict[str, Any]]:
    """
    Downstream candidates, routing *through* cameras the BSW site no longer lists: they stay in
    the road graph for connectivity, but a vehicle can only be picked up again on a live camera,
    so a removed candidate is replaced by the live cameras beyond it (distances and arrival
    windows accumulated along the way).
    """
    results: Dict[int, Dict[str, Any]] = {}
    frontier = [(c, 0) for c in topology.get_downstream_candidates(camera_id, exit_direction)]
    while frontier:
        cand, hops = frontier.pop(0)
        target = cand["target_camera_id"]
        if target == camera_id or target in results:
            continue
        if not _is_removed(target):
            results[target] = cand
            continue
        if hops >= max_hops:
            continue
        for nxt in topology.get_downstream_candidates(target, "*"):
            merged = dict(nxt)
            merged["distance_m"] = cand.get("distance_m", 0) + nxt.get("distance_m", 0)
            merged["t_min_s"] = cand.get("t_min_s", 0) + nxt.get("t_min_s", 0)
            merged["t_max_s"] = cand.get("t_max_s", 0) + nxt.get("t_max_s", 0)
            merged["exit_direction"] = cand.get("exit_direction", nxt.get("exit_direction"))
            merged["road_name"] = cand.get("road_name") or nxt.get("road_name", "")
            frontier.append((merged, hops + 1))
    return list(results.values())


@app.post("/api/handoff/predict")
def predict_handoff(req: HandoffRequest):
    """
    Predicts downstream candidate cameras when a vehicle exits a junction heading a specific direction.
    Calculates expected arrival window [t_min, t_max].
    """
    candidates = _live_downstream_candidates(req.camera_id, req.exit_direction)
    results = []

    for cand in candidates:
        target_cam_id = cand["target_camera_id"]
        target_cam = _camera_or_none(target_cam_id)
        results.append({
            "target_camera_id": target_cam_id,
            "target_camera_name": cand.get("target_name") or (target_cam["name"] if target_cam else f"Camera #{target_cam_id}"),
            "stream_url": target_cam.get("stream_url") if target_cam else None,
            "coordinates": [target_cam.get("latitude"), target_cam.get("longitude")] if target_cam else None,
            "travel_time_seconds": int((cand.get("t_min_s", 60) + cand.get("t_max_s", 120)) / 2),
            "arrival_window": {
                "t_min_seconds": int(cand.get("t_min_s", 40)),
                "t_max_seconds": int(cand.get("t_max_s", 120))
            },
            "road_name": cand.get("road_name", ""),
            "distance_m": cand.get("distance_m", 0)
        })

    # Record handoff intent only for a vehicle that has actually been sighted: the dashboard
    # calls this on every camera switch with a placeholder id, and active_targets must point at
    # a real sighting (FOREIGN KEY on last_sighting_id).
    best_candidate_id = candidates[0]["target_camera_id"] if candidates else None
    journey = db.get_vehicle_journey(req.vehicle_id) if best_candidate_id else []
    if journey:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        latest = journey[-1]
        try:
            db.record_handoff_event(
                global_vehicle_id=req.vehicle_id,
                from_camera_id=req.camera_id,
                to_camera_id=best_candidate_id,
                confidence=0.92,
                timestamp_iso=now_iso
            )
            db.update_active_target(
                global_vehicle_id=req.vehicle_id,
                current_camera_id=latest["camera_id"],
                last_sighting_id=latest["sighting_id"],
                status="tracking",
                locked_at_iso=now_iso
            )
        except Exception as e:
            print(f"Warning: could not record handoff intent for vehicle {req.vehicle_id}: {e}")

    return {
        "vehicle_id": req.vehicle_id,
        "from_camera": req.camera_id,
        "exit_direction": req.exit_direction,
        "candidate_count": len(results),
        "candidates": results
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
    """Everything the dashboard refreshes on each tick. Built on a worker thread from the shared
    cache, so N open tabs cost one computation per cache interval and never block the event loop.
    ``counts`` / ``recent`` are what the dashboard would otherwise fetch with two more requests."""
    stats, active_targets = _cached_stats()
    analytics_summary = _analytics_engine.get_all_camera_summaries() if _analytics_engine else []
    return {
        "type": "heartbeat",
        "timestamp": time.time(),
        "stats": stats,
        "camera_counts": stats.get("camera_counts", {}),
        "top_cameras": stats.get("top_cameras", []),
        "active_targets": active_targets,
        "analytics_summary": analytics_summary,
        "rolling": _rolling_summary(),
        "counts": get_camera_counts(),
        "recent": [_sanitize_sighting(r) for r in _latest_sightings_cached(12)],
    }


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Periodic live stats every 2 seconds; the DB work runs off the event loop.
            await websocket.send_json(await asyncio.to_thread(_heartbeat_payload))
            await asyncio.sleep(2.0)
    except (WebSocketDisconnect, RuntimeError):
        pass   # client went away (RuntimeError: send on a socket that already closed)
    finally:
        if websocket in active_connections:
            active_connections.remove(websocket)


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


def _flow_summary(camera_id: int, minutes: float, bucket_minutes: float) -> Dict[str, Any]:
    """Counting-line flow of one camera: the headline rate over ``minutes`` and a bucketed series
    (flow in veh/h per bucket, with the mean congestion index of the same bucket)."""
    minutes = min(max(1.0, minutes), 1440.0)
    bucket_minutes = min(max(1.0, bucket_minutes), minutes)
    rate = db.get_camera_rates(window_s=minutes * 60.0).get(camera_id)
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
    return {
        "basis": "counting-line crossings",
        "window_minutes": minutes,
        "bucket_minutes": bucket_minutes,
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
    metrics["rate"] = metrics["flow"]["rate"] if minutes == 60.0 else db.get_camera_rates(window_s=3600.0).get(camera_id)
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
    counting=CountingLines(str(DATA_DIR / "counting_lines.json")),
)

def _rolling_summary() -> Dict[str, Any]:
    st = rolling_tracker.get_status()
    return {k: st[k] for k in ("running", "rolling_enabled", "focus_camera_id", "visits",
                               "cameras_covered", "total_cameras", "vehicles_counted")}

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

    ret_jpg, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return jpeg.tobytes() if ret_jpg else b""

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

# Rendered JPEG of the newest focus frame, shared by every viewer of that camera.
_render_cache: Dict[int, Tuple[int, bytes]] = {}
_render_lock = threading.Lock()

def _latest_focus_jpeg(camera_id: int) -> bytes:
    """Marks the camera as viewed and returns its newest annotated frame (b"" while connecting)."""
    rolling_tracker.set_focus(camera_id)
    latest = rolling_tracker.get_focus_frame(camera_id)
    if latest is None:
        return b""
    with _render_lock:
        cached = _render_cache.get(camera_id)
        if cached is not None and cached[0] == latest["seq"]:
            return cached[1]
    jpeg_bytes = _render_annotated_frame(latest["frame"], latest["detections"], latest["ane_ms"], latest)
    if jpeg_bytes:
        with _render_lock:
            _render_cache[camera_id] = (latest["seq"], jpeg_bytes)
    return jpeg_bytes

@app.get("/api/stream/{camera_id}/frame")
def get_single_annotated_frame(camera_id: int):
    """Returns the latest single annotated JPEG frame for smooth fallback polling."""
    cam = _camera_or_404(camera_id)

    cam_name = cam.get("name", f"Camera #{camera_id}")
    stream_url = cam.get("stream_url")
    if not stream_url:
        placeholder = _generate_placeholder_frame(cam_name, "NO STREAM URL CONFIGURED")
        return Response(content=placeholder, media_type="image/jpeg")

    jpeg_bytes = _latest_focus_jpeg(camera_id)

    if not jpeg_bytes:
        placeholder = _generate_placeholder_frame(cam_name, "CONNECTING HLS STREAM...")
        return Response(
            content=placeholder,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
        )

    return Response(
        content=jpeg_bytes,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/api/stream/{camera_id}")
async def stream_ai_video(camera_id: int, request: Request):
    """
    Streams live video annotated on Apple Neural Engine (ANE)
    with YOLOv8 bounding boxes, vehicle classifications, and and per-class colours.
    """
    cam = _camera_or_404(camera_id)

    cam_name = cam.get("name", f"Camera #{camera_id}")
    stream_url = cam.get("stream_url")
    if not stream_url:
        raise HTTPException(status_code=404, detail="Camera stream not found")

    def part(jpeg: bytes) -> bytes:
        return (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " + str(len(jpeg)).encode("ascii")
                + b"\r\n\r\n" + jpeg + b"\r\n")

    async def frame_gen():
        # Stop as soon as the viewer goes away: every iteration keeps this camera in focus, so a
        # stream left running after a camera switch would hold a tracker focus slot.
        while not await request.is_disconnected():
            jpeg_bytes = await asyncio.to_thread(_latest_focus_jpeg, camera_id)
            if not jpeg_bytes:
                yield part(_generate_placeholder_frame(cam_name, "CONNECTING STREAM..."))
                await asyncio.sleep(0.3)
                continue
            yield part(jpeg_bytes)
            await asyncio.sleep(0.035)

    return StreamingResponse(
        frame_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "close"
        }
    )

# Mount static frontend assets
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 65)
    print(" Starting BSW Kota Bogor MTMC Dashboard Server")
    print(" URL: http://127.0.0.1:8000")
    print("=" * 65 + "\n")
    uvicorn.run("backend.server:app", host="127.0.0.1", port=8000, reload=False)
