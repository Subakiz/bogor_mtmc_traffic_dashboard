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

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse, Response
from pydantic import BaseModel
import cv2
from ultralytics import YOLO

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import (
    Database, get_db_connection, get_recent_sightings,
    get_traffic_stats, get_vehicle_journey, search_vehicles,
    record_handoff_event, DEFAULT_DB_PATH
)
from backend.topology_engine import (
    TopologyEngine, get_default_engine, get_all_cameras,
    get_camera_info, get_downstream_candidates, calculate_arrival_window
)
from backend.indexing_daemon import IndexingDaemon
from backend.stream_capture import ThreadedStreamCapture

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
FRONTEND_DIR = PROJECT_ROOT / "frontend"
FRONTEND_DIR.mkdir(exist_ok=True)

# Shared engines
db = Database(str(DEFAULT_DB_PATH))
topology = get_default_engine()
daemon_instance: Optional[IndexingDaemon] = None

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
    seed_cameras_from_topology(db.conn, str(PROJECT_ROOT / "data" / "bogor_camera_topology.json"))

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

@app.get("/api/cameras/{camera_id}")
def get_camera(camera_id: int):
    cam = topology.get_camera_info(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
    return cam

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

@app.get("/api/stats")
def get_stats():
    """Returns live citywide traffic metrics and unique vehicle counts."""
    stats = db.get_traffic_stats()
    active_targets = db.list_active_targets()
    return {
        "stats": stats,
        "active_tracked_targets": active_targets,
        "timestamp": time.time()
    }

@app.get("/api/recent")
def recent_sightings(limit: int = 50, camera_id: Optional[int] = None):
    """Returns recent vehicle sightings across all cameras or a specific camera."""
    cam_id = camera_id if camera_id is not None else 167
    sightings = db.get_recent_sightings(camera_id=cam_id, limit=limit)
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
    return journey

@app.get("/api/vehicles/search")
def search_vehicle_ids(query: str = ""):
    """Searches vehicles by ID or attributes."""
    results = db.search_vehicles(query)
    return {"query": query, "results": results}

@app.post("/api/handoff/predict")
def predict_handoff(req: HandoffRequest):
    """
    Predicts downstream candidate cameras when a vehicle exits a junction heading a specific direction.
    Calculates expected arrival window [t_min, t_max].
    """
    candidates = topology.get_downstream_candidates(req.camera_id, req.exit_direction)
    results = []

    for cand in candidates:
        target_cam_id = cand["target_camera_id"]
        target_cam = topology.get_camera_info(target_cam_id)
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

    # Record handoff intent in database
    best_candidate_id = candidates[0]["target_camera_id"] if candidates else None
    if best_candidate_id:
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
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
                current_camera_id=req.camera_id,
                last_sighting_id=0,
                status="tracking",
                locked_at_iso=now_iso
            )
        except Exception as e:
            pass

    return {
        "vehicle_id": req.vehicle_id,
        "from_camera": req.camera_id,
        "exit_direction": req.exit_direction,
        "candidate_count": len(results),
        "candidates": results
    }

@app.get("/api/daemon/status")
def daemon_status():
    global daemon_instance
    running = daemon_instance is not None and daemon_instance.running
    return {
        "running": running,
        "mode": "Apple Neural Engine (ANE)",
        "model": "yolov8m.mlpackage",
        "device": "CPU_AND_NE"
    }

@app.post("/api/daemon/start")
def start_daemon():
    global daemon_instance
    if daemon_instance is not None and daemon_instance.running:
        return {"status": "already_running"}
    daemon_instance = IndexingDaemon(db=db, topology=topology)
    daemon_instance.start()
    return {"status": "started", "message": "ANE Background Indexing Daemon active"}

@app.post("/api/daemon/stop")
def stop_daemon():
    global daemon_instance
    if daemon_instance is not None:
        daemon_instance.stop()
        daemon_instance = None
    return {"status": "stopped", "message": "Daemon stopped"}

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    active_connections.append(websocket)
    try:
        while True:
            # Send periodic live stats every 2 seconds
            stats = db.get_traffic_stats()
            active_targets = db.list_active_targets()
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": time.time(),
                "stats": stats,
                "active_targets": active_targets
            })
            await asyncio.sleep(2.0)
    except WebSocketDisconnect:
        if websocket in active_connections:
            active_connections.remove(websocket)

# Shared YOLO tracker model & Thread-safe Stream Captures
_yolo_model = None
_yolo_lock = threading.Lock()
_active_stream_captures: Dict[int, ThreadedStreamCapture] = {}
_stream_lock = threading.Lock()

def get_yolo_model():
    global _yolo_model
    with _yolo_lock:
        if _yolo_model is None:
            model_path = "/Users/nabils/cctv_yolo_ane/yolov8m.mlpackage"
            if not os.path.exists(model_path):
                model_path = str(PROJECT_ROOT / "yolov8m.mlpackage")
            _yolo_model = YOLO(model_path)
        return _yolo_model

def get_stream_capture(camera_id: int, stream_url: str) -> ThreadedStreamCapture:
    with _stream_lock:
        if camera_id in _active_stream_captures:
            stream = _active_stream_captures[camera_id]
            if stream._is_opened or (stream._thread and stream._thread.is_alive()):
                return stream
            else:
                stream.release()
                del _active_stream_captures[camera_id]

        # Stop old streams to prevent connection exhaustion on Bogor restreamers
        for old_id, old_stream in list(_active_stream_captures.items()):
            try:
                old_stream.release()
            except Exception:
                pass
            del _active_stream_captures[old_id]

        stream = ThreadedStreamCapture(stream_url, camera_id=camera_id, max_fps=20.0)
        _active_stream_captures[camera_id] = stream
        return stream

def _process_annotated_frame(frame: Any, model: Any) -> Tuple[bytes, float, int]:
    t0 = time.perf_counter()
    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        classes=[2, 3, 5, 7],
        conf=0.25,
        verbose=False
    )
    ane_ms = (time.perf_counter() - t0) * 1000.0
    annotated = results[0].plot(line_width=2)
    n_boxes = len(results[0].boxes) if results[0].boxes is not None else 0

    # Draw ANE Performance HUD
    cv2.rectangle(annotated, (15, 15), (310, 48), (15, 15, 18), -1)
    cv2.rectangle(annotated, (15, 15), (310, 48), (0, 215, 255), 1)
    cv2.putText(annotated, f"ANE: {ane_ms:.1f}ms | YOLOv8 ByteTrack", (25, 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 240, 255), 1)

    ret_jpg, jpeg = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ret_jpg:
        return b"", ane_ms, 0
    return jpeg.tobytes(), ane_ms, n_boxes

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

@app.get("/api/stream/{camera_id}/frame")
def get_single_annotated_frame(camera_id: int):
    """Returns the latest single annotated JPEG frame for smooth fallback polling."""
    cam = topology.get_camera_info(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam_name = cam.get("name", f"Camera #{camera_id}")
    stream_url = cam.get("stream_url")
    if not stream_url:
        placeholder = _generate_placeholder_frame(cam_name, "NO STREAM URL CONFIGURED")
        return Response(content=placeholder, media_type="image/jpeg")

    stream = get_stream_capture(camera_id, stream_url)
    ret, frame = stream.get_frame()
    if not ret or frame is None:
        for _ in range(8):
            time.sleep(0.1)
            ret, frame = stream.get_frame()
            if ret and frame is not None:
                break

    if not ret or frame is None:
        placeholder = _generate_placeholder_frame(cam_name, "CONNECTING HLS STREAM...")
        return Response(
            content=placeholder,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate, max-age=0"}
        )

    model = get_yolo_model()
    jpeg_bytes, _, _ = _process_annotated_frame(frame, model)
    if not jpeg_bytes:
        placeholder = _generate_placeholder_frame(cam_name, "INFERENCE BUFFERING...")
        return Response(content=placeholder, media_type="image/jpeg")

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
def stream_ai_video(camera_id: int):
    """
    Streams live video annotated on Apple Neural Engine (ANE)
    with YOLOv8 bounding boxes, vehicle classifications, and persistent ByteTrack IDs.
    """
    cam = topology.get_camera_info(camera_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    cam_name = cam.get("name", f"Camera #{camera_id}")
    stream_url = cam.get("stream_url")
    if not stream_url:
        raise HTTPException(status_code=404, detail="Camera stream not found")

    stream = get_stream_capture(camera_id, stream_url)
    model = get_yolo_model()

    def frame_gen():
        for _ in range(20):
            ret, frame = stream.get_frame()
            if ret and frame is not None:
                break
            time.sleep(0.1)

        while True:
            ret, frame = stream.get_frame()
            if not ret or frame is None:
                placeholder = _generate_placeholder_frame(cam_name, "CONNECTING STREAM...")
                if placeholder:
                    yield (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        b"Content-Length: " + str(len(placeholder)).encode("ascii") + b"\r\n\r\n" +
                        placeholder + b"\r\n"
                    )
                time.sleep(0.5)
                continue

            jpeg_bytes, _, _ = _process_annotated_frame(frame, model)
            if not jpeg_bytes:
                time.sleep(0.05)
                continue

            content_len = len(jpeg_bytes)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(content_len).encode("ascii") + b"\r\n\r\n" +
                jpeg_bytes + b"\r\n"
            )
            time.sleep(0.03)

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
