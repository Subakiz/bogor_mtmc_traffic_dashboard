from typing import Dict

import cv2
import numpy as np

from .config import CLASS_COLORS


def draw_hud(frame: np.ndarray, camera_name: str, active_counts: Dict[str, int],
             fps: float, ane_ms: float) -> None:
    """Draw a compact status panel (camera, latency, per-class counts) onto ``frame``.

    Only the panel's region of interest is blended, so the cost is independent of
    the frame resolution.
    """
    x, y, w, h = 15, 15, 300, 130
    fh, fw = frame.shape[:2]
    x2, y2 = min(x + w, fw), min(y + h, fh)
    if x2 <= x or y2 <= y:
        return

    roi = frame[y:y2, x:x2]
    cv2.addWeighted(np.full_like(roi, (23, 17, 14)), 0.82, roi, 0.18, 0, dst=roi)
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 212, 255), 1)

    name = (camera_name[:30] + "..") if len(camera_name) > 32 else camera_name
    cv2.putText(frame, name, (x + 10, y + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (230, 230, 230), 1, cv2.LINE_AA)
    cv2.putText(frame, f"ANE {ane_ms:.1f} ms | {fps:.1f} FPS | {sum(active_counts.values())} in view",
                (x + 10, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (170, 205, 215), 1, cv2.LINE_AA)

    row_y = y + 66
    for cls in ("car", "motorcycle", "truck", "bus"):
        cv2.putText(frame, f"{cls.capitalize():<11}{active_counts.get(cls, 0):>3}", (x + 10, row_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, CLASS_COLORS[cls], 1, cv2.LINE_AA)
        row_y += 16
