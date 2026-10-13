"""
Multi-camera vehicle detection + tracking on the Apple Neural Engine.

``model.track(persist=True)`` keeps a single ByteTrack instance per model, so feeding
frames from several cameras through one model silently mixes their tracks. This module
runs the (CoreML) detector once per frame and keeps an *independent* ByteTrack instance
for every camera, which is what makes rolling multi-camera tracking possible.

Track IDs are drawn from one process-wide counter, so a ``track_id`` is unique across
cameras as well as within a camera.
"""

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .config import _GLOBAL_ANE_LOCK, CLASS_NAMES, DETECTION_CONF, VEHICLE_CLASS_IDS

logger = logging.getLogger(__name__)

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


class CoreMLDetector:
    """A CoreML YOLO export with NMS baked in (``yolo export format=coreml nms=True``), called
    directly instead of through ultralytics' predictor.

    Ultralytics' CoreML backend turns every frame into a float tensor and back into a PIL image
    from a non-contiguous array, which costs 4-7 ms of CPU per frame (more than the ANE forward
    pass of yolov8m) and holds the GIL throughout. A cv2 letterbox straight into PIL costs ~0.3 ms
    and gives identical boxes.
    """

    _VEHICLE_CLASSES = np.array(VEHICLE_CLASS_IDS)

    def __init__(self, model_path: str):
        import coremltools as ct

        self.model_path = model_path
        self.model = ct.models.MLModel(model_path, compute_units=ct.ComputeUnit.CPU_AND_NE)
        spec = self.model.get_spec().description
        image = spec.input[0]
        if image.type.WhichOneof("Type") != "imageType":
            raise ValueError(f"{model_path}: expected an image input")
        if not {"confidence", "coordinates"} <= {o.name for o in spec.output}:
            raise ValueError(f"{model_path}: export with nms=True (needs confidence/coordinates outputs)")
        self.input_name = image.name
        self.width = int(image.type.imageType.width)
        self.height = int(image.type.imageType.height)

    def scale_for(self, width: int, height: int) -> float:
        """Letterbox scale factor this model applies to a ``width`` x ``height`` frame."""
        return min(self.width / width, self.height / height)

    def predict(self, frame: np.ndarray, conf: float) -> np.ndarray:
        """Vehicle detections for a BGR frame as rows ``[x1, y1, x2, y2, conf, cls]`` in frame pixels."""
        import cv2
        from PIL import Image

        h, w = frame.shape[:2]
        r = self.scale_for(w, h)
        nw, nh = int(round(w * r)), int(round(h * r))
        top, left = (self.height - nh) // 2, (self.width - nw) // 2
        im = frame if (nw, nh) == (w, h) else cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        im = cv2.copyMakeBorder(im, top, self.height - nh - top, left, self.width - nw - left,
                                cv2.BORDER_CONSTANT, value=(114, 114, 114))
        image = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        with _GLOBAL_ANE_LOCK:
            out = self.model.predict({self.input_name: image})

        scores = out["confidence"]
        if len(scores) == 0:
            return np.empty((0, 6), dtype=np.float32)
        score, cls = scores.max(1), scores.argmax(1)
        keep = (score >= conf) & np.isin(cls, self._VEHICLE_CLASSES)
        xywh = out["coordinates"][keep] * (self.width, self.height, self.width, self.height)
        rows = np.empty((len(xywh), 6), dtype=np.float32)
        rows[:, 0] = (xywh[:, 0] - xywh[:, 2] / 2 - left) / r
        rows[:, 1] = (xywh[:, 1] - xywh[:, 3] / 2 - top) / r
        rows[:, 2] = (xywh[:, 0] + xywh[:, 2] / 2 - left) / r
        rows[:, 3] = (xywh[:, 1] + xywh[:, 3] / 2 - top) / r
        rows[:, 4] = score[keep]
        rows[:, 5] = cls[keep]
        return rows


class VehicleDetector:
    """Shared CoreML YOLO detector with per-camera tracking.

    CoreML exports (``.mlpackage``/``.mlmodel``) run through :class:`CoreMLDetector`; anything
    else (``.pt``, or an explicit ``imgsz``) goes through ultralytics. ``wide_model_path`` is an
    optional second export at a 16:9 input size (e.g. 384x640), used for frames it covers at no
    loss of resolution: letterboxing 16:9 video into 640x640 spends 44% of the ANE pass on padding.
    """

    def __init__(self, model_path: str = "yolov8m.mlpackage", conf: float = DETECTION_CONF,
                 imgsz: Optional[int] = None, wide_model_path: Optional[str] = None):
        self.model_path = model_path
        self.conf = conf
        self.imgsz = imgsz
        self.pool = CameraTrackerPool()
        self.last_inference_ms = 0.0
        self._coreml: Optional[CoreMLDetector] = None
        self._wide: Optional[CoreMLDetector] = None

        if imgsz is None and model_path.endswith((".mlpackage", ".mlmodel")):
            try:
                self._coreml = CoreMLDetector(model_path)
            except Exception as exc:  # noqa: BLE001 - fall back to the ultralytics predictor
                logger.warning(f"Direct CoreML path unavailable for {model_path} ({exc}); using ultralytics.")
        if self._coreml is not None:
            self.model = self._coreml.model
            if wide_model_path and os.path.exists(wide_model_path):
                try:
                    self._wide = CoreMLDetector(wide_model_path)
                except Exception as exc:  # noqa: BLE001
                    logger.warning(f"Ignoring wide model {wide_model_path}: {exc}")
        else:
            from ultralytics import YOLO

            self.model = YOLO(model_path, task="detect")

    def _engine_for(self, width: int, height: int) -> CoreMLDetector:
        wide = self._wide
        if wide is not None and wide.scale_for(width, height) >= self._coreml.scale_for(width, height) - 1e-6:
            return wide
        return self._coreml

    def detect(self, frame: np.ndarray) -> Any:
        """Raw detections for one frame (no tracking), as an ultralytics ``Boxes``."""
        from ultralytics.engine.results import Boxes

        t0 = time.perf_counter()
        if self._coreml is not None:
            h, w = frame.shape[:2]
            rows = self._engine_for(w, h).predict(frame, self.conf)
            self.last_inference_ms = (time.perf_counter() - t0) * 1000.0
            return Boxes(rows, (h, w))

        kwargs: Dict[str, Any] = dict(conf=self.conf, classes=VEHICLE_CLASS_IDS, verbose=False)
        if self.imgsz:
            kwargs["imgsz"] = self.imgsz
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
