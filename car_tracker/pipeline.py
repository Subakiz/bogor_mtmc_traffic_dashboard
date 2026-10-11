from collections import defaultdict, deque
from typing import Any, Callable, Dict, List, Optional

import cv2
import numpy as np

from .config import CLASS_COLORS

LabelFn = Callable[[Dict[str, Any]], str]

_TRAIL_LEN = 24


def default_label(det: Dict[str, Any]) -> str:
    """Class-only label ('CAR'); track ids are deliberately not drawn."""
    return det['class_name'].upper()


def count_by_class(detections: List[Dict[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = defaultdict(int)
    for det in detections:
        counts[det["class_name"]] += 1
    return dict(counts)


def update_trails(trails: Dict[int, deque], detections: List[Dict[str, Any]]) -> None:
    """Append each detection's bottom-centre point to its trail and drop dead trails."""
    live = set()
    for det in detections:
        x1, _, x2, y2 = det["box"]
        tid = det["track_id"]
        live.add(tid)
        trails.setdefault(tid, deque(maxlen=_TRAIL_LEN)).append((int((x1 + x2) / 2), int(y2 - 4)))
    for tid in [t for t in trails if t not in live]:
        del trails[tid]


def draw_detections(
    annotated: np.ndarray,
    detections: List[Dict[str, Any]],
    trails: Optional[Dict[int, deque]] = None,
    label_fn: LabelFn = default_label,
) -> None:
    """Draw boxes, labels and motion trails onto ``annotated`` in place."""
    for det in sorted(detections, key=lambda d: _area(d["box"]), reverse=True):
        x1, y1, x2, y2 = det["box"]
        color = CLASS_COLORS.get(det["class_name"], (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        text = f" {label_fn(det)} {det['conf']:.2f} "
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        by = max(y1 - 20, 0)
        cv2.rectangle(annotated, (x1, by), (x1 + tw + 6, by + 19), color, -1)
        cv2.putText(annotated, text, (x1 + 3, by + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 0, 0), 1, cv2.LINE_AA)

        pts = trails.get(det["track_id"]) if trails is not None else None
        if pts and len(pts) > 1:
            cv2.polylines(annotated, [np.array(pts, dtype=np.int32).reshape(-1, 1, 2)], False, color, 2)
            cv2.circle(annotated, pts[-1], 4, color, -1)


def _area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])
