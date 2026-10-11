"""
backend/reid.py
Cross-camera vehicle identity (the "global vehicle id" behind Follow-That-Car).

Two parts:
- ``extract_appearance``: a compact colour/layout descriptor computed from the vehicle crop.
- ``IdentityResolver``: gives each confirmed single-camera track a global vehicle id, either
  by linking it to a vehicle that recently *departed* a topologically reachable camera
  (appearance similarity gated by the road-network travel-time window) or by minting a new id.

Honest limits: the descriptor is colour-based, so it separates a red hatchback from a white
van but not two white vans. Matching therefore requires both a high appearance similarity and a
physically plausible transit time between the cameras, and every departure can be consumed once.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple

import cv2
import numpy as np

EMBEDDING_DIM = 256   # stored as float32 -> 1024 bytes
COLOR_HIST_DIM = 128  # stored as float32 -> 512 bytes

_MAX_SIDE = 64
_MIN_CROP_PX = 8


def _hellinger_l1(hist: np.ndarray) -> np.ndarray:
    total = float(hist.sum())
    if total <= 0:
        return np.zeros_like(hist)
    return np.sqrt(hist / total)


def extract_appearance(frame: np.ndarray, box: Tuple[int, int, int, int]
                       ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Returns ``(embedding[256], color_hist[128])`` as float32, or ``None`` if the crop is unusable.
    ``embedding`` is L2-normalised so cosine similarity is a dot product; ``color_hist`` sums to 1.
    """
    fh, fw = frame.shape[:2]
    x1, y1, x2, y2 = (int(v) for v in box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(fw, x2), min(fh, y2)
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None

    crop = frame[y1:y2, x1:x2]
    scale = _MAX_SIDE / float(max(crop.shape[:2]))
    if scale < 1.0:
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    # Drop the outer 15% ring: bounding boxes include a lot of road / background.
    h, w = crop.shape[:2]
    mh, mw = int(h * 0.15), int(w * 0.15)
    if h - 2 * mh >= 4 and w - 2 * mw >= 4:
        crop = crop[mh:h - mh, mw:w - mw]
    if crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    whole = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
    half = hsv.shape[0] // 2
    top = bottom = np.zeros(64, dtype=np.float32)
    if half >= 2:
        top = cv2.calcHist([hsv[:half]], [0, 1, 2], None, [8, 2, 4], [0, 180, 0, 256, 0, 256]).flatten()
        bottom = cv2.calcHist([hsv[half:]], [0, 1, 2], None, [8, 2, 4], [0, 180, 0, 256, 0, 256]).flatten()

    color_hist = (whole / max(float(whole.sum()), 1e-8)).astype(np.float32)
    embedding = np.concatenate([_hellinger_l1(whole), _hellinger_l1(top), _hellinger_l1(bottom)]).astype(np.float32)
    norm = float(np.linalg.norm(embedding))
    if norm <= 1e-8:
        return None
    return embedding / norm, color_hist


@dataclass
class Departure:
    gid: int
    camera_id: int
    class_name: str
    embedding: np.ndarray
    t_depart: float


class IdentityResolver:
    """Thread-safe global-id allocator with topology-gated appearance matching."""

    def __init__(
        self,
        topology: Optional[Any] = None,
        start_id: int = 1,
        same_camera_threshold: float = 0.93,
        cross_camera_threshold: float = 0.88,
        ttl_s: float = 900.0,
        max_pending: int = 1500,
    ):
        self.topology = topology
        self.same_camera_threshold = same_camera_threshold
        self.cross_camera_threshold = cross_camera_threshold
        self.ttl_s = ttl_s
        self._next_id = max(1, int(start_id))
        self._pending: Deque[Departure] = deque(maxlen=max_pending)
        self._window_cache: Dict[Tuple[int, int], Optional[Tuple[float, float]]] = {}
        self._lock = threading.Lock()

    # ── id allocation ─────────────────────────────────────────────────────────
    def new_global_id(self) -> int:
        with self._lock:
            gid = self._next_id
            self._next_id += 1
            return gid

    def ensure_next_id_above(self, gid: int) -> None:
        with self._lock:
            self._next_id = max(self._next_id, int(gid) + 1)

    # ── departures / matching ─────────────────────────────────────────────────
    def register_departure(self, gid: int, camera_id: int, class_name: str,
                           embedding: np.ndarray, t_depart: float) -> None:
        with self._lock:
            self._pending.append(Departure(gid, camera_id, class_name, embedding, t_depart))

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    def _window(self, src: int, dst: int) -> Optional[Tuple[float, float]]:
        key = (src, dst)
        if key not in self._window_cache:
            window = None
            if self.topology is not None:
                try:
                    t_min, t_max = self.topology.calculate_arrival_window(src, dst)
                    if t_max != float("inf"):
                        window = (float(t_min), float(t_max))
                except KeyError:
                    window = None
            self._window_cache[key] = window
        return self._window_cache[key]

    def _transit_score(self, dep: Departure, camera_id: int, elapsed: float) -> float:
        """Plausibility in [0, 1] that a vehicle gone from ``dep.camera_id`` reappears here."""
        if dep.camera_id == camera_id:
            # Track fragmentation (occlusion / re-entry): only plausible shortly after departing.
            return 1.0 if 0.0 <= elapsed <= 30.0 else 0.0
        window = self._window(dep.camera_id, camera_id)
        if window is None:
            return 0.0
        valid, score = self.topology.score_transition_window(window[0], window[1], elapsed)
        return score if valid else 0.0

    def resolve(self, camera_id: int, class_name: str, embedding: np.ndarray, timestamp: float
                ) -> Tuple[int, float, Optional[int]]:
        """
        Returns ``(global_id, confidence, matched_from_camera)``. ``matched_from_camera`` is None
        when a brand-new id was minted.
        """
        with self._lock:
            while self._pending and timestamp - self._pending[0].t_depart > self.ttl_s:
                self._pending.popleft()

            best: Optional[Departure] = None
            best_score = 0.0
            for dep in self._pending:
                if dep.class_name != class_name:
                    continue
                elapsed = timestamp - dep.t_depart
                if elapsed < 0:
                    continue
                transit = self._transit_score(dep, camera_id, elapsed)
                if transit <= 0.0:
                    continue
                sim = float(np.dot(embedding, dep.embedding))
                threshold = (self.same_camera_threshold if dep.camera_id == camera_id
                             else self.cross_camera_threshold)
                if sim < threshold:
                    continue
                score = 0.7 * sim + 0.3 * transit
                if score > best_score:
                    best, best_score = dep, score

            if best is not None:
                self._pending.remove(best)
                return best.gid, round(best_score, 4), best.camera_id

            gid = self._next_id
            self._next_id += 1
            return gid, 0.0, None
