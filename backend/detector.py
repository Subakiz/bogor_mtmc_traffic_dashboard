"""
backend/detector.py
BSW Kota Bogor Multi-Camera Traffic Intelligence Dashboard.
CoreML Detection Service on Apple Neural Engine (FEAT-M1).

Implements:
- Shared CoreML YOLOv8 detector (ct.ComputeUnit.CPU_AND_NE) from the cctv_yolo_ane engine.
- Independent ByteTrack state per camera, so several cameras can be tracked through the
  same model without mixing their tracks (required for rolling multi-camera tracking).
- Strict vehicle class filtering (car, motorcycle, bus, truck; COCO IDs 2, 3, 5, 7).
- Motion vector, speed (px/s), and real-world road exit heading angle Theta in [0, 360) deg.
- Deterministic synthetic tracker for headless test environments (``test_mode=True`` only;
  without a model a production detector detects nothing rather than inventing vehicles).
"""

import collections
import math
import os
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.config import DEFAULT_MODEL_PATH, WIDE_MODEL_PATH, ensure_engine_importable
from car_tracker.config import CLASS_NAMES as ALLOWED_VEHICLE_CLASSES   # COCO ids 2, 3, 5, 7

ALLOWED_CLASS_NAMES = set(ALLOWED_VEHICLE_CLASSES.values())

_HISTORY_TTL_S = 60.0   # drop heading history of tracks not seen for this long

_LOAD_RETRY_S = 60.0     # a model that failed to load is retried after this long
_ERROR_LOG_EVERY_S = 30.0

_SHARED_DETECTOR = None
_SHARED_DETECTOR_LOCK = threading.Lock()
_last_load_attempt = 0.0


def get_shared_detector(model_path: str = DEFAULT_MODEL_PATH):
    """Returns the process-wide engine ``VehicleDetector`` (one CoreML model, per-camera trackers),
    or None while the model is missing or fails to load (retried every ``_LOAD_RETRY_S``)."""
    global _SHARED_DETECTOR, _last_load_attempt
    with _SHARED_DETECTOR_LOCK:
        now = time.monotonic()
        if _SHARED_DETECTOR is None and (not _last_load_attempt or now - _last_load_attempt >= _LOAD_RETRY_S):
            _last_load_attempt = now
            if not os.path.exists(model_path):
                print(f"[detector] Model not found: {model_path} -- no vehicles will be detected.")
                return None
            try:
                ensure_engine_importable()
                from car_tracker import VehicleDetector

                _SHARED_DETECTOR = VehicleDetector(model_path, wide_model_path=WIDE_MODEL_PATH)
            except Exception as exc:  # noqa: BLE001
                print(f"[detector] Could not load {model_path}: {exc}")
                _SHARED_DETECTOR = None
        return _SHARED_DETECTOR


class ANEVehicleDetector:
    """
    Apple Neural Engine (ANE) accelerated YOLOv8 vehicle detector with per-camera tracking
    and a motion heading tracker.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        conf_threshold: float = 0.25,
        compute_unit: str = "CPU_AND_NE",
        test_mode: bool = False,
    ):
        self.model_path = model_path
        self.conf_threshold = conf_threshold
        self.compute_unit = compute_unit
        self.test_mode = test_mode

        # Track history: track_id -> deque of (cx, cy, timestamp)
        self.track_history: Dict[int, collections.deque] = collections.defaultdict(
            lambda: collections.deque(maxlen=24)
        )
        self._next_track_id = 1
        self._active_tracks: Dict[int, Dict[str, Any]] = {}
        self._engine = None
        self._is_native_loaded = False
        self.last_inference_ms = 0.0
        self._last_prune = 0.0
        self._last_error_log = 0.0

        if not self.test_mode:
            self._load_coreml_model()

    @property
    def _model(self):
        """Backwards-compatible handle to the underlying CoreML model (None in test mode)."""
        return self._engine.model if self._engine is not None else None

    @property
    def model_loaded(self) -> bool:
        return self._is_native_loaded

    def _load_coreml_model(self):
        """Attaches to the shared engine detector singleton."""
        self._engine = get_shared_detector(self.model_path)
        self._is_native_loaded = self._engine is not None

    def _log_error(self, message: str) -> None:
        now = time.monotonic()
        if now - self._last_error_log >= _ERROR_LOG_EVERY_S:
            self._last_error_log = now
            print(f"[detector] {message}")

    def compute_heading_and_speed(
        self,
        track_id: int,
        current_centroid: Tuple[int, int],
        current_time: float,
        camera_azimuth_deg: float = 0.0,
    ) -> Tuple[float, float]:
        """
        Computes motion vector, pixel speed (px/s), and exit heading angle Theta in [0, 360) deg.
        """
        history = self.track_history[track_id]
        cx, cy = current_centroid
        history.append((cx, cy, current_time))

        if len(history) < 2:
            return float(camera_azimuth_deg % 360.0), 0.0

        old_cx, old_cy, old_time = history[0]
        dt = current_time - old_time
        if dt <= 1e-4:
            return float(camera_azimuth_deg % 360.0), 0.0

        dx = cx - old_cx
        dy = cy - old_cy
        dist_px = math.hypot(dx, dy)
        speed_px_s = dist_px / dt

        if dist_px < 2.0:
            # Negligible displacement: default to camera mounting azimuth
            heading_deg = camera_azimuth_deg % 360.0
        else:
            angle_rad = math.atan2(dy, dx)
            heading_deg = (math.degrees(angle_rad) + camera_azimuth_deg) % 360.0

        return float(heading_deg), float(speed_px_s)

    def _prune_history(self, now: float) -> None:
        """Track ids are never reused, so stale histories must be dropped to bound memory."""
        if now - self._last_prune < _HISTORY_TTL_S:
            return
        self._last_prune = now
        for tid in [t for t, h in self.track_history.items() if not h or now - h[-1][2] > _HISTORY_TTL_S]:
            del self.track_history[tid]

    def detect(
        self,
        frame: np.ndarray,
        camera_azimuth_deg: float = 0.0,
        timestamp: Optional[float] = None,
        synthetic_objects: Optional[List[Dict[str, Any]]] = None,
        camera_id: int = 0,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Runs vehicle detection and tracking for ``camera_id`` on a BGR frame.
        Each camera has its own tracker state; track ids are unique across cameras.
        Returns None when the model is not loaded or inference failed (the frame was not
        observed), so callers can tell that from an empty road.
        """
        curr_time = timestamp or time.time()
        h, w = frame.shape[:2]
        self._prune_history(curr_time)

        # 1. Native ANE CoreML YOLO execution
        if not self.test_mode:
            if self._engine is None:
                self._load_coreml_model()   # retried at most every _LOAD_RETRY_S
                if self._engine is None:
                    return None
            try:
                raw = self._engine.track(camera_id, frame)
                self.last_inference_ms = self._engine.last_inference_ms
            except Exception as exc:  # noqa: BLE001
                self._log_error(f"inference failed on camera {camera_id}: {exc}")
                return None

            if raw is None:
                return None   # never fall through to the synthetic extractor on a real frame
            detections = []
            for det in raw:
                if det["class_id"] not in ALLOWED_VEHICLE_CLASSES or det["conf"] < self.conf_threshold:
                    continue
                x1 = max(0, min(det["box"][0], w - 1))
                y1 = max(0, min(det["box"][1], h - 1))
                x2 = max(x1 + 1, min(det["box"][2], w))
                y2 = max(y1 + 1, min(det["box"][3], h))
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)
                heading, speed = self.compute_heading_and_speed(
                    det["track_id"], (cx, cy), curr_time, camera_azimuth_deg
                )
                detections.append({
                    "track_id": det["track_id"],
                    "class_name": det["class_name"],
                    "class_id": det["class_id"],
                    "confidence": float(det["conf"]),
                    "bbox": (x1, y1, x2, y2),
                    "centroid": (cx, cy),
                    "heading_deg": float(heading),
                    "speed_px_s": float(speed),
                })
            return detections

        # 2. Resilient test / synthetic tracker mode
        raw_objects = synthetic_objects or self._extract_synthetic_objects(frame)
        detections = []

        for obj in raw_objects:
            cls_name = obj.get("class", "car")
            if cls_name not in ALLOWED_CLASS_NAMES:
                continue

            conf = float(obj.get("conf", 0.90))
            if conf < self.conf_threshold:
                continue

            bbox = obj.get("bbox", (10, 10, 100, 80))
            x1 = max(0, min(int(bbox[0]), w - 1))
            y1 = max(0, min(int(bbox[1]), h - 1))
            x2 = max(x1 + 1, min(int(bbox[2]), w))
            y2 = max(y1 + 1, min(int(bbox[3]), h))
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            track_id = self._match_or_create_track(cx, cy, cls_name, curr_time, camera_id=camera_id)

            heading, speed = self.compute_heading_and_speed(
                track_id, (cx, cy), curr_time, camera_azimuth_deg
            )

            detections.append({
                "track_id": track_id,
                "class_name": cls_name,
                "class_id": 2 if cls_name == "car" else (3 if cls_name == "motorcycle" else (5 if cls_name == "bus" else 7)),
                "confidence": conf,
                "bbox": (x1, y1, x2, y2),
                "centroid": (cx, cy),
                "heading_deg": heading,
                "speed_px_s": speed,
            })

        return detections

    def _extract_synthetic_objects(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts foreground colored bounding boxes in synthetic test frames.
        """
        h, w = frame.shape[:2]
        sub = frame[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        if np.std(sub) > 10.0:
            return [{
                "class": "car",
                "conf": 0.92,
                "bbox": (int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)),
            }]
        return []

    def _match_or_create_track(
        self, cx: int, cy: int, cls_name: str, curr_time: float,
        max_dist: float = 120.0, camera_id: int = 0,
    ) -> int:
        """Associates centroid with an active track on the same camera or creates a new track ID."""
        best_id = None
        best_dist = float("inf")

        for tid, tinfo in list(self._active_tracks.items()):
            # Drop tracks not seen in 5 seconds
            if curr_time - tinfo["last_seen"] > 5.0:
                del self._active_tracks[tid]
                continue

            if tinfo["class"] == cls_name and tinfo.get("camera_id", 0) == camera_id:
                dist = math.hypot(cx - tinfo["cx"], cy - tinfo["cy"])
                if dist < max_dist and dist < best_dist:
                    best_dist = dist
                    best_id = tid

        if best_id is not None:
            self._active_tracks[best_id]["cx"] = cx
            self._active_tracks[best_id]["cy"] = cy
            self._active_tracks[best_id]["last_seen"] = curr_time
            return best_id

        new_id = self._next_track_id
        self._next_track_id += 1
        self._active_tracks[new_id] = {
            "cx": cx,
            "cy": cy,
            "class": cls_name,
            "camera_id": camera_id,
            "last_seen": curr_time,
        }
        return new_id

    def reset_tracks(self, camera_id: Optional[int] = None):
        """Clears tracking state. With ``camera_id`` only that camera's tracker is reset."""
        if camera_id is None:
            self.track_history.clear()
            self._active_tracks.clear()
            self._next_track_id = 1
            if self._engine is not None:
                for cid in self._engine.pool.active_cameras():
                    self._engine.reset(cid)
            return
        for tid in [t for t, i in self._active_tracks.items() if i.get("camera_id", 0) == camera_id]:
            del self._active_tracks[tid]
        if self._engine is not None:
            self._engine.reset(camera_id)
