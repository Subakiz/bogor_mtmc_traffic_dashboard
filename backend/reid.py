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

Matching cost: the pending departures live in fixed-size numpy arrays (a ring with tombstones).
A query first narrows them with vectorised masks (class, elapsed time, a conservative bound on the
transit window of each source camera, then one matrix-vector product for appearance with a small
numeric margin) and only the survivors go through the exact per-departure scoring, so decisions
are the same as scoring every pending departure one by one.
"""

import threading
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.topology_engine import TopologyEngine

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


@dataclass(eq=False)
class Departure:
    """A vehicle that left a camera. ``eq=False``: identity comparison -- the generated ``__eq__``
    compared the numpy embeddings and raised ValueError whenever two departures shared
    gid / camera / class (``list.remove`` and ``deque.remove`` compare with ``==``)."""
    gid: int
    camera_id: int
    class_name: str
    embedding: np.ndarray
    t_depart: float


_SAME_CAMERA_MAX_GAP_S = 30.0   # track fragmentation (occlusion / re-entry): only plausible this soon
# Pre-filter margins. TopologyEngine.score_transition_window accepts an early arrival while
# exp(-0.5 (delta/sigma)^2) >= 0.10 (delta <= 2.14597 sigma) and a late one while >= 0.05
# (delta <= 2.44775 sigma); the factors below are slightly larger, the exact score is recomputed.
_EARLY_SIGMAS = 2.15
_LATE_SIGMAS = 2.45
_TIME_EPS_S = 1e-6
_SIM_REL_MARGIN = 1e-4          # float32 dot products of 256 values agree to ~1.5e-5 * |a||b|
_SIM_ABS_MARGIN = 1e-6
_NEVER = (float("inf"), float("-inf"))   # acceptance interval that matches no elapsed time


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
        self._window_cache: Dict[Tuple[int, int], Optional[Tuple[float, float]]] = {}
        self._lock = threading.Lock()

        # Pending departures, oldest first. Storage is a ring of 2 * max_pending slots: consumed
        # entries become tombstones (``_alive`` False) so the order of the rest is untouched, and
        # the live ones are packed to the front when the ring reaches its end. At most
        # ``max_pending`` are live; a further departure evicts the oldest.
        self._cap = max(0, int(max_pending))
        phys = 2 * self._cap
        self._deps: List[Optional[Departure]] = [None] * phys
        self._emb: Optional[np.ndarray] = None            # [phys, dim] float32, allocated on first use
        self._enorm = np.zeros(phys, dtype=np.float64)
        self._t = np.zeros(phys, dtype=np.float64)
        self._cam = np.zeros(phys, dtype=np.int32)         # compact camera index (see _cam_index)
        self._cls = np.zeros(phys, dtype=np.int32)         # compact class index
        self._alive = np.zeros(phys, dtype=bool)
        self._start = 0     # first slot that may be live
        self._end = 0       # next slot to write
        self._live = 0
        self._cam_index: Dict[int, int] = {}
        self._cam_ids: List[int] = []
        self._cls_index: Dict[str, int] = {}
        # destination camera -> [lo, hi, n_known]: per source camera (compact index), the range of
        # elapsed times that can score above zero.
        self._dst_bounds: Dict[int, list] = {}
        self._gaussian_scoring = (getattr(topology, "score_transition_window", None)
                                  is TopologyEngine.score_transition_window)

    # ── id allocation ─────────────────────────────────────────────────────────
    def new_global_id(self) -> int:
        with self._lock:
            gid = self._next_id
            self._next_id += 1
            return gid

    def ensure_next_id_above(self, gid: int) -> None:
        with self._lock:
            self._next_id = max(self._next_id, int(gid) + 1)

    # ── pending departures ────────────────────────────────────────────────────
    def register_departure(self, gid: int, camera_id: int, class_name: str,
                           embedding: np.ndarray, t_depart: float) -> None:
        with self._lock:
            if self._cap == 0:
                return
            vec = np.asarray(embedding, dtype=np.float32)
            if vec.ndim != 1:
                raise ValueError(f"embedding must be 1-D, got shape {vec.shape}")
            if self._emb is None:
                self._emb = np.zeros((2 * self._cap, vec.shape[0]), dtype=np.float32)
            elif vec.shape[0] != self._emb.shape[1]:
                raise ValueError(f"embedding has {vec.shape[0]} values, expected {self._emb.shape[1]}")
            if self._live >= self._cap:
                self._drop_oldest()
            if self._end >= len(self._deps):
                self._compact()
            i = self._end
            self._end += 1
            self._live += 1
            self._deps[i] = Departure(gid, camera_id, class_name, embedding, t_depart)
            self._emb[i] = vec
            self._enorm[i] = float(np.linalg.norm(vec))
            self._t[i] = float(t_depart)
            self._cam[i] = self._compact_camera(camera_id)
            self._cls[i] = self._cls_index.setdefault(class_name, len(self._cls_index))
            self._alive[i] = True

    def pending_count(self) -> int:
        with self._lock:
            return self._live

    def pending_snapshot(self) -> List[Departure]:
        """The pending departures, oldest first (for tests and diagnostics)."""
        with self._lock:
            return [d for d in self._deps[self._start:self._end] if d is not None]

    def _compact_camera(self, camera_id: int) -> int:
        idx = self._cam_index.get(camera_id)
        if idx is None:
            idx = self._cam_index[camera_id] = len(self._cam_ids)
            self._cam_ids.append(camera_id)
        return idx

    def _kill(self, i: int) -> None:
        self._alive[i] = False
        self._deps[i] = None
        self._live -= 1

    def _first_live(self) -> int:
        i = self._start
        while not self._alive[i]:
            i += 1
        self._start = i
        return i

    def _drop_oldest(self) -> None:
        if self._live:
            self._kill(self._first_live())

    def _compact(self) -> None:
        """Packs the live entries (in order) to the front of the ring."""
        s, e = self._start, self._end
        idx = np.flatnonzero(self._alive[s:e]) + s
        n = len(idx)
        for arr in (self._emb, self._enorm, self._t, self._cam, self._cls):
            arr[:n] = arr[idx]
        self._alive[:n] = True
        self._alive[n:e] = False
        deps = [self._deps[i] for i in idx]
        self._deps[:n] = deps
        self._deps[n:e] = [None] * (e - n)
        self._start, self._end = 0, n

    def _expire(self, timestamp: float) -> None:
        # Only the oldest entries are checked, in registration order (as a queue would): a stale
        # departure behind a fresh one stays until the fresh one is consumed or expires.
        while self._live:
            i = self._first_live()
            if timestamp - self._deps[i].t_depart > self.ttl_s:
                self._kill(i)
            else:
                break

    # ── matching ──────────────────────────────────────────────────────────────
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
            return 1.0 if 0.0 <= elapsed <= _SAME_CAMERA_MAX_GAP_S else 0.0
        window = self._window(dep.camera_id, camera_id)
        if window is None:
            return 0.0
        valid, score = self.topology.score_transition_window(window[0], window[1], elapsed)
        return score if valid else 0.0

    def _accept_interval(self, src: int, dst: int) -> Tuple[float, float]:
        """Range of elapsed seconds in which a departure from ``src`` can still match at ``dst``
        (a superset of where ``_transit_score`` is positive)."""
        if src == dst:
            return 0.0, _SAME_CAMERA_MAX_GAP_S
        window = self._window(src, dst)
        if window is None:
            return _NEVER
        if not self._gaussian_scoring:
            return float("-inf"), float("inf")      # unknown scoring function: leave it to the exact check
        t_min, t_max = window
        sigma_early = max(0.25 * t_min, 2.0)
        sigma_late = max(0.50 * (t_max - t_min), 30.0)
        return (t_min - _EARLY_SIGMAS * sigma_early - _TIME_EPS_S,
                t_max + _LATE_SIGMAS * sigma_late + _TIME_EPS_S)

    def _bounds_for(self, dst: int) -> Tuple[np.ndarray, np.ndarray]:
        entry = self._dst_bounds.get(dst)
        n = len(self._cam_ids)
        if entry is None:
            entry = self._dst_bounds[dst] = [np.empty(0), np.empty(0), 0]
        lo, hi, known = entry
        if known < n:
            lo2, hi2 = np.full(n, _NEVER[0]), np.full(n, _NEVER[1])
            lo2[:known], hi2[:known] = lo, hi
            for j in range(known, n):
                lo2[j], hi2[j] = self._accept_interval(self._cam_ids[j], dst)
            entry[:] = [lo2, hi2, n]
        return entry[0], entry[1]

    def _candidates(self, camera_id: int, class_name: str, embedding: Any, timestamp: float) -> List[int]:
        """Slots that could match, oldest first: a superset of the exact matches."""
        ci = self._cls_index.get(class_name)
        if ci is None or not self._live:
            return []
        s, e = self._start, self._end
        elapsed = timestamp - self._t[s:e]
        cams = self._cam[s:e]
        lo, hi = self._bounds_for(camera_id)
        mask = self._alive[s:e] & (self._cls[s:e] == ci) & (elapsed >= lo[cams]) & (elapsed <= hi[cams])
        mask &= elapsed >= -_TIME_EPS_S
        idx = np.flatnonzero(mask)
        if idx.size == 0:
            return []
        idx += s
        try:
            q = np.asarray(embedding, dtype=np.float32)
        except (TypeError, ValueError):
            return idx.tolist()
        if q.ndim != 1 or q.shape[0] != self._emb.shape[1]:
            return idx.tolist()          # odd query: the exact scoring below reports it as before
        if idx.size * 3 > e - s:
            sims = (self._emb[s:e] @ q)[idx - s]
        else:
            sims = self._emb[idx] @ q
        qcam = self._cam_index.get(camera_id, -1)
        threshold = np.where(self._cam[idx] == qcam, self.same_camera_threshold, self.cross_camera_threshold)
        margin = _SIM_REL_MARGIN * float(np.linalg.norm(q)) * self._enorm[idx] + _SIM_ABS_MARGIN
        return idx[sims >= threshold - margin].tolist()

    def resolve(self, camera_id: int, class_name: str, embedding: np.ndarray, timestamp: float
                ) -> Tuple[int, float, Optional[int]]:
        """
        Returns ``(global_id, confidence, matched_from_camera)``. ``matched_from_camera`` is None
        when a brand-new id was minted.
        """
        with self._lock:
            self._expire(timestamp)

            best_i: Optional[int] = None
            best_score = 0.0
            for i in self._candidates(camera_id, class_name, embedding, timestamp):
                dep = self._deps[i]
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
                    best_i, best_score = i, score

            if best_i is not None:
                best = self._deps[best_i]
                self._kill(best_i)
                return best.gid, round(best_score, 4), best.camera_id

            gid = self._next_id
            self._next_id += 1
            return gid, 0.0, None
