"""
Multi-camera vehicle detection + tracking on the Apple Neural Engine.

``model.track(persist=True)`` keeps a single ByteTrack instance per model, so feeding
frames from several cameras through one model silently mixes their tracks. This module
runs the (CoreML) detector once per frame and keeps an *independent* ByteTrack instance
for every camera, which is what makes rolling multi-camera tracking possible.

Track IDs are drawn from one process-wide counter, so a ``track_id`` is unique across
cameras as well as within a camera.
"""

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import _GLOBAL_ANE_LOCK, CLASS_NAMES, DETECTION_CONF, VEHICLE_CLASS_IDS

Detection = Dict[str, Any]


def _load_tracker_cfg(tracker: str = "bytetrack.yaml"):
    from ultralytics.utils import YAML, IterableSimpleNamespace
    from ultralytics.utils.checks import check_yaml

    cfg = IterableSimpleNamespace(**YAML.load(check_yaml(tracker)))
    cfg.device = "cpu"
    return cfg


@dataclass
class _CameraState:
    tracker: Any
    last_used: float = field(default_factory=time.time)


class CameraTrackerPool:
    """One ByteTrack instance per camera id."""

    def __init__(self, tracker: str = "bytetrack.yaml"):
        self._cfg = _load_tracker_cfg(tracker)
        self._states: Dict[int, _CameraState] = {}
        self._lock = threading.Lock()

    def _new_tracker(self):
        from ultralytics.trackers.basetrack import BaseTrack
        from ultralytics.trackers.byte_tracker import BYTETracker

        # BYTETracker.__init__ resets the *global* ID counter to 0; restore it so IDs
        # stay unique across every camera's tracker.
        previous = BaseTrack._count
        tracker = BYTETracker(args=self._cfg)
        BaseTrack._count = previous
        return tracker

    def update(self, camera_id: int, boxes: Any, frame: np.ndarray) -> np.ndarray:
        """Feed one frame's detections to the camera's tracker; returns ByteTrack rows
        ``[x1, y1, x2, y2, track_id, score, cls, det_idx]``."""
        with self._lock:
            state = self._states.get(camera_id)
            if state is None:
                state = self._states[camera_id] = _CameraState(self._new_tracker())
            state.last_used = time.time()
            return state.tracker.update(boxes, frame)

    def reset(self, camera_id: int) -> None:
        """Forget a camera's tracker (e.g. after the camera was unwatched for a while)."""
        with self._lock:
            self._states.pop(camera_id, None)

    def active_cameras(self) -> List[int]:
        with self._lock:
            return list(self._states)


class VehicleDetector:
    """Shared CoreML YOLO detector with per-camera tracking."""

    def __init__(self, model_path: str = "yolov8m.mlpackage", conf: float = DETECTION_CONF,
                 imgsz: Optional[int] = None):
        from ultralytics import YOLO

        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.model = YOLO(model_path, task="detect")
        self.pool = CameraTrackerPool()
        self.last_inference_ms = 0.0

    def detect(self, frame: np.ndarray) -> Any:
        """Raw detections for one frame (no tracking), as an ultralytics ``Boxes``."""
        kwargs: Dict[str, Any] = dict(conf=self.conf, classes=VEHICLE_CLASS_IDS, verbose=False)
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
        t0 = time.perf_counter()
        with _GLOBAL_ANE_LOCK:
            result = self.model.predict(frame, **kwargs)[0]
        self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
        return result.boxes.cpu().numpy()

    def track(self, camera_id: int, frame: np.ndarray) -> List[Detection]:
        """Detect + track vehicles for ``camera_id``. Returns detection dicts with
        ``track_id``, ``class_id``, ``class_name``, ``box`` and ``conf``."""
        boxes = self.detect(frame)
        rows = self.pool.update(camera_id, boxes, frame)
        return rows_to_detections(rows)

    def reset(self, camera_id: int) -> None:
        self.pool.reset(camera_id)


def rows_to_detections(rows: np.ndarray) -> List[Detection]:
    """Convert ByteTrack output rows into detection dicts."""
    dets: List[Detection] = []
    if rows is None or len(rows) == 0:
        return dets
    for x1, y1, x2, y2, tid, score, cls, *_ in rows:
        cid = int(cls)
        dets.append({
            "track_id": int(tid),
            "class_id": cid,
            "class_name": CLASS_NAMES.get(cid, "vehicle"),
            "box": (int(x1), int(y1), int(x2), int(y2)),
            "conf": float(score),
        })
    return dets
