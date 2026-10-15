"""
backend/counting.py
Per-camera counting lines for traffic volume, learned from how vehicles actually move.

Counting vehicles "in view" over-counts queued and parked vehicles and double-counts tracks that
fragment behind occlusions. Volume counters instead count a vehicle when it crosses a line across
the road. Every camera here has a different view, so the line is learned automatically:

- ``MotionModel`` accumulates the trajectories of moving tracks (the last ``LINE_MEMORY`` or so) (start -> end, in frame
  coordinates normalised to 0..1). The dominant axis of travel is the length-weighted axial mean of
  their directions (angles doubled, so opposite lanes reinforce instead of cancelling), and the line
  is placed through the length-weighted middle of the traffic, perpendicular to that axis.
- Until ``LINE_MIN_TRACKS`` trajectories are known, a vehicle is counted once it has moved
  ``PASSAGE_FRAC`` of the frame instead ("passage" mode), so a new camera counts from its first visit.
- Directions are averaged in pixel space (x scaled by the frame aspect, ``set_aspect``), so the line
  is perpendicular to the traffic in the image, not just in 0..1 coordinates (on 16:9 frames the
  old lines were a median 26 deg off). Models learned before that keep their 0..1-space angles
  (``aspect`` None): re-angling them turned every line ~25 deg and broke each camera's rate history.
- The sign of the axis (which crossing direction is +1) is pinned to the previous fit, so
  ``dir_pos``/``dir_neg`` don't swap when the axis wobbles across 45 deg.
- Coherence (|resultant| / weight: 1 = all traffic on one axis, 0 = all directions) gates the
  learned line: below ``COHERENCE_MIN`` (a junction where flows cancel, e.g. camera 193 at ~0.1) the
  axis is noise and the camera counts in passage mode; below ``COHERENCE_LOW`` the line is kept but
  its rates are flagged low confidence (``quality``).
- A manual override file (``data/counting_line_overrides.json``) wins over the learned line: a
  hand-placed line or segment per camera, or ``exclude`` for cameras with no road (parks). Junctions
  and roundabouts re-learn the same bad geometry, so they need a hand-placed line.

Positions are a vehicle's ground point (bottom-centre of its box). A vehicle is counted when its
current position and its reference position (where it was first seen) lie on opposite sides of the
line *as it is now*, which needs no frame-to-frame continuity and ignores jitter within
``HYSTERESIS`` of the line. Both sides are judged against the current line every time: the line is
refit while tracks are still alive (every departing track teaches the model), and a side remembered
from the older line made a queued car look as if it had crossed when the line moved past it.
"""

import json
import logging
import math
import os
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

Point = Tuple[float, float]

LINE_MIN_TRACKS = 12      # moving trajectories needed before a learned line replaces passage mode
LINE_MEMORY = 500         # older trajectories fade beyond this many, so a re-aimed camera's line follows
MIN_MOVE = 0.08           # a trajectory shorter than this (fraction of the frame) doesn't teach the model
PASSAGE_FRAC = 0.15       # passage mode: count after moving this far
HYSTERESIS = 0.015        # points this close to the line have no side yet
MIN_CROSS_MOVE = 2 * HYSTERESIS   # a counted vehicle moved at least this far (implied by the two sides)
COHERENCE_MIN = 0.25      # below: the learned axis is noise (flows cancel), count in passage mode instead
COHERENCE_LOW = 0.5       # below: the learned line is kept but its rates are flagged low confidence


def ground_point(bbox: Tuple[int, int, int, int], width: int, height: int) -> Point:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0 / max(1, width), y2 / max(1, height))


@dataclass
class CountingLine:
    px: float
    py: float
    nx: float   # unit normal: the dominant direction of travel; crossing along +n is direction +1
    ny: float
    half: Optional[float] = None   # a segment: half its length along the line around (px, py); None = infinite

    @classmethod
    def through(cls, p1: Point, p2: Point, segment: bool) -> "CountingLine":
        """The line through p1 and p2; +1 is the right-hand side of p1 -> p2 as seen on screen."""
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        return cls((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, -dy / length, dx / length,
                   length / 2.0 if segment else None)

    def distance(self, pt: Point) -> float:
        return (pt[0] - self.px) * self.nx + (pt[1] - self.py) * self.ny

    def side(self, pt: Point) -> int:
        d = self.distance(pt)
        return 0 if abs(d) < HYSTERESIS else (1 if d > 0 else -1)

    def spans(self, a: Point, b: Point) -> bool:
        """Whether the move a -> b (on opposite sides) crosses within the segment."""
        if self.half is None:
            return True
        da, db = self.distance(a), self.distance(b)
        t = da / (da - db)
        qx, qy = a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])
        return abs((qx - self.px) * -self.ny + (qy - self.py) * self.nx) <= self.half + HYSTERESIS

    def labels(self) -> Tuple[str, str]:
        """Human labels for crossing directions (+1, -1) in image terms."""
        if abs(self.ny) >= abs(self.nx):
            pair = ("towards camera", "away from camera")
            return pair if self.ny > 0 else pair[::-1]
        pair = ("left to right", "right to left")
        return pair if self.nx > 0 else pair[::-1]

    def endpoints(self) -> Tuple[Point, Point]:
        """The line clipped to the frame (or the segment), for drawing (normalised coordinates)."""
        dx, dy = -self.ny, self.nx
        if self.half is not None:
            h = self.half
            return (self.px - h * dx, self.py - h * dy), (self.px + h * dx, self.py + h * dy)
        ts = []
        for edge, (p, d) in (("x", (self.px, dx)), ("y", (self.py, dy))):
            if abs(d) > 1e-9:
                ts += [(0.0 - p) / d, (1.0 - p) / d]
        pts = [(self.px + t * dx, self.py + t * dy) for t in ts]
        pts = [p for p in pts if -1e-6 <= p[0] <= 1 + 1e-6 and -1e-6 <= p[1] <= 1 + 1e-6]
        if len(pts) < 2:
            return (self.px - dx, self.py - dy), (self.px + dx, self.py + dy)
        pts.sort(key=lambda p: (p[0] - self.px) * dx + (p[1] - self.py) * dy)
        return pts[0], pts[-1]


def _canonical(n: Point) -> Point:
    """The sign rule used before signs were pinned: +y if mostly vertical, else +x."""
    if (abs(n[1]) >= abs(n[0]) and n[1] < 0) or (abs(n[0]) > abs(n[1]) and n[0] < 0):
        return -n[0], -n[1]
    return n


@dataclass
class MotionModel:
    c2: float = 0.0
    s2: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    weight: float = 0.0
    tracks: int = 0
    aspect: Optional[float] = None    # frame width / height the angles are measured in (None: legacy, 1.0)
    ref: Optional[Point] = None       # the previous fit's normal: pins the sign of the axis
    learned: Optional[CountingLine] = None
    manual: Optional[CountingLine] = None   # override line (wins over the learned one)
    excluded: bool = False                  # override: no road in view, never counts

    @property
    def coherence(self) -> float:
        return math.hypot(self.c2, self.s2) / self.weight if self.weight > 0 else 0.0

    @property
    def line(self) -> Optional[CountingLine]:
        """The line vehicles are counted against; None = passage mode (or excluded)."""
        if self.excluded:
            return None
        if self.manual is not None:
            return self.manual
        if self.learned is not None and self.coherence >= COHERENCE_MIN:
            return self.learned
        return None

    def quality(self) -> str:
        """excluded | manual | learned | low (weakly coherent line) | passage (no usable line)."""
        if self.excluded:
            return "excluded"
        if self.manual is not None:
            return "manual"
        if self.line is None:
            return "passage"
        return "low" if self.coherence < COHERENCE_LOW else "learned"

    def add(self, start: Point, end: Point) -> bool:
        """Adds one trajectory; returns True if it was long enough to teach the model."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < MIN_MOVE:
            return False
        theta = math.atan2(dy, dx * (self.aspect or 1.0))   # the angle in pixels
        self.c2 += length * math.cos(2 * theta)
        self.s2 += length * math.sin(2 * theta)
        self.sx += length * (start[0] + end[0]) / 2.0
        self.sy += length * (start[1] + end[1]) / 2.0
        self.weight += length
        self.tracks += 1
        if self.tracks > LINE_MEMORY:
            k = LINE_MEMORY / self.tracks
            self.c2, self.s2, self.sx, self.sy, self.weight = (v * k for v in (self.c2, self.s2, self.sx, self.sy, self.weight))
            self.tracks = LINE_MEMORY
        self.ref = self.axis() or self.ref
        if self.tracks >= LINE_MIN_TRACKS:
            self.learned = self._fit()
        return True

    def set_aspect(self, aspect: float) -> bool:
        """Measures angles in a frame of this width/height from now on; returns True if it changed.
        The stored axis is carried over (same direction in the image, same coherence). A model learned
        before pixel-space angles (aspect None, has data) is left as it is, so its line doesn't move."""
        if self.aspect is None and self.tracks:
            return False
        old = self.aspect or 1.0
        if self.aspect is not None and abs(aspect - old) < 1e-3:
            return False
        r = math.hypot(self.c2, self.s2)
        if r > 0:
            theta = 0.5 * math.atan2(self.s2, self.c2)
            theta = math.atan2(math.sin(theta), math.cos(theta) * aspect / old)
            self.c2, self.s2 = r * math.cos(2 * theta), r * math.sin(2 * theta)
        self.aspect = aspect
        if self.tracks >= LINE_MIN_TRACKS:
            self.learned = self._fit()
        return True

    def axis(self) -> Optional[Point]:
        """Normal of the counting line in 0..1 coordinates (unit vector): the dominant direction of
        travel, made perpendicular in pixels; its sign agrees with the previous fit. None without data."""
        if self.weight <= 0 or math.hypot(self.c2, self.s2) < 1e-9:
            return None
        theta = 0.5 * math.atan2(self.s2, self.c2)
        nx, ny = math.cos(theta) * (self.aspect or 1.0), math.sin(theta)
        norm = math.hypot(nx, ny)
        n = (nx / norm, ny / norm)
        if self.ref is None:
            return _canonical(n)
        return n if n[0] * self.ref[0] + n[1] * self.ref[1] >= 0 else (-n[0], -n[1])

    def _fit(self) -> Optional[CountingLine]:
        axis = self.axis()
        if axis is None:
            return None
        return CountingLine(self.sx / self.weight, self.sy / self.weight, axis[0], axis[1])

    def to_dict(self) -> Dict[str, Any]:
        return {"c2": self.c2, "s2": self.s2, "sx": self.sx, "sy": self.sy,
                "weight": self.weight, "tracks": self.tracks, "aspect": self.aspect,
                "ref": list(self.ref) if self.ref else None}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MotionModel":
        m = cls(**{k: d[k] for k in ("c2", "s2", "sx", "sy", "weight", "tracks", "aspect") if k in d})
        # Legacy models (no ref) keep the sign they had under the old canonical rule.
        m.ref = tuple(d["ref"]) if d.get("ref") else m.axis()
        if m.tracks >= LINE_MIN_TRACKS:
            m.learned = m._fit()
        return m


class TrackCounter:
    """Decides when one track is counted, and in which direction."""

    __slots__ = ("start", "last", "ref", "counted", "direction", "mode")

    def __init__(self, pt: Point):
        self.start = pt
        self.last = pt
        self.ref = pt          # where the track is measured from: first position with a clear side
        self.counted = False
        self.direction = 0
        self.mode = ""

    def update(self, pt: Point, model: MotionModel, can_count: bool) -> bool:
        """Feeds a new position; returns True the moment the track should be counted."""
        self.last = pt
        if model.excluded:
            return False
        line = model.line
        if line is not None:
            side = line.side(pt)
            ref_side = line.side(self.ref)
            if ref_side == 0:
                # Started on the line (or the line has just moved onto the reference point): measure
                # from the first position that has a clear side.
                if side != 0:
                    self.ref = pt
                return False
            if (side != 0 and side != ref_side
                    and math.hypot(pt[0] - self.ref[0], pt[1] - self.ref[1]) >= MIN_CROSS_MOVE):
                if not line.spans(self.ref, pt):
                    self.ref = pt      # went past the end of the segment: measure from here
                    return False
                if can_count and not self.counted:
                    self.counted, self.direction, self.mode = True, side, "line"
                    return True
            return False
        dx, dy = pt[0] - self.start[0], pt[1] - self.start[1]
        if can_count and not self.counted and math.hypot(dx, dy) >= PASSAGE_FRAC:
            axis = model.axis() if model.tracks >= 3 else None
            proj = (dx * axis[0] + dy * axis[1]) if axis else 0.0
            self.counted, self.direction, self.mode = True, (1 if proj > 0 else -1 if proj < 0 else 0), "passage"
            return True
        return False


def load_overrides(path: str) -> Dict[int, Dict[str, Any]]:
    """Reads the manual override file: ``{"<camera id>": {"line": [x1, y1, x2, y2], "segment": bool}
    | {"exclude": true}}`` (0..1 coordinates; keys starting with ``_`` and ``version`` are notes).
    A bad entry is logged and skipped on its own, so one typo doesn't drop every override."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        return {}
    except (OSError, ValueError) as exc:
        logger.error(f"Counting line overrides {path} unreadable, none applied: {exc}")
        return {}
    out: Dict[int, Dict[str, Any]] = {}
    for key, entry in raw.items():
        if key.startswith("_") or key == "version":
            continue
        try:
            cid = int(key)
            if entry.get("exclude"):
                if "line" in entry:
                    raise ValueError("both line and exclude")
                out[cid] = {"exclude": True}
                continue
            x1, y1, x2, y2 = (float(v) for v in entry["line"])
            if not all(0.0 <= v <= 1.0 for v in (x1, y1, x2, y2)) or math.hypot(x2 - x1, y2 - y1) < 0.02:
                raise ValueError("coordinates must be in 0..1 and the line at least 0.02 long")
            out[cid] = {"line": CountingLine.through((x1, y1), (x2, y2), bool(entry.get("segment", False)))}
        except (KeyError, TypeError, ValueError, AttributeError) as exc:
            logger.error(f"Counting line override for camera {key!r} ignored: {exc}")
    return out


class CountingLines:
    """Motion models for every camera, persisted to a JSON file so learned lines survive restarts,
    plus the manual overrides (read-only; never written back)."""

    def __init__(self, path: Optional[str] = None, overrides_path: Optional[str] = None):
        self.path = path
        self._models: Dict[int, MotionModel] = {}
        self._lock = threading.Lock()
        self._dirty = False
        self._overrides = load_overrides(overrides_path) if overrides_path else {}
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for cid, d in json.load(f).items():
                        self._models[int(cid)] = MotionModel.from_dict(d)
            except (OSError, ValueError, TypeError, AttributeError) as exc:
                # Move it aside: the next save() would overwrite it and lose every learned line.
                logger.error(f"Counting lines {path} unreadable ({exc}); moved to {path}.bad, starting empty")
                self._models = {}
                try:
                    os.replace(path, path + ".bad")
                except OSError:
                    pass
        for cid, o in self._overrides.items():
            m = self._models.setdefault(cid, MotionModel())
            m.manual, m.excluded = o.get("line"), bool(o.get("exclude"))

    def _get(self, camera_id: int) -> MotionModel:
        m = self._models.get(camera_id)
        if m is None:
            m = self._models[camera_id] = MotionModel()
        return m

    def model(self, camera_id: int) -> MotionModel:
        with self._lock:
            return self._get(camera_id)

    def learn(self, camera_id: int, start: Point, end: Point) -> None:
        with self._lock:
            if self._get(camera_id).add(start, end):
                self._dirty = True

    def set_aspect(self, camera_id: int, aspect: float) -> None:
        with self._lock:
            if self._get(camera_id).set_aspect(aspect):
                self._dirty = True

    def line(self, camera_id: int) -> Optional[CountingLine]:
        with self._lock:
            m = self._models.get(camera_id)
            return m.line if m else None

    def quality(self, camera_id: int) -> str:
        with self._lock:
            m = self._models.get(camera_id)
            return m.quality() if m else "passage"

    def coherence(self, camera_id: int) -> Optional[float]:
        with self._lock:
            m = self._models.get(camera_id)
            return round(m.coherence, 3) if m and m.weight > 0 else None

    def excluded(self, camera_id: int) -> bool:
        return camera_id in self._overrides and bool(self._overrides[camera_id].get("exclude"))

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            if not self._dirty:
                return
            data = {str(cid): m.to_dict() for cid, m in self._models.items() if m.tracks}
            self._dirty = False
        tmp = self.path + ".tmp"
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except BaseException:
            self._dirty = True   # retry on the next save
            raise
