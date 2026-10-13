"""
backend/counting.py
Per-camera counting lines for traffic volume, learned from how vehicles actually move.

Counting vehicles "in view" over-counts queued and parked vehicles and double-counts tracks that
fragment behind occlusions. Volume counters instead count a vehicle when it crosses a line across
the road. Every camera here has a different view, so the line is learned automatically:

- ``MotionModel`` accumulates the trajectories of moving tracks (start -> end, in frame
  coordinates normalised to 0..1). The dominant axis of travel is the length-weighted axial mean of
  their directions (angles doubled, so opposite lanes reinforce instead of cancelling), and the line
  is placed through the length-weighted middle of the traffic, perpendicular to that axis.
- Until ``LINE_MIN_TRACKS`` trajectories are known, a vehicle is counted once it has moved
  ``PASSAGE_FRAC`` of the frame instead ("passage" mode), so a new camera counts from its first visit.

Positions are a vehicle's ground point (bottom-centre of its box). A vehicle is counted when its
current position and its reference position (where it was first seen) lie on opposite sides of the
line *as it is now*, which needs no frame-to-frame continuity and ignores jitter within
``HYSTERESIS`` of the line. Both sides are judged against the current line every time: the line is
refit while tracks are still alive (every departing track teaches the model), and a side remembered
from the older line made a queued car look as if it had crossed when the line moved past it.
"""

import json
import math
import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

Point = Tuple[float, float]

LINE_MIN_TRACKS = 12      # moving trajectories needed before a learned line replaces passage mode
MIN_MOVE = 0.08           # a trajectory shorter than this (fraction of the frame) doesn't teach the model
PASSAGE_FRAC = 0.15       # passage mode: count after moving this far
HYSTERESIS = 0.015        # points this close to the line have no side yet
MIN_CROSS_MOVE = 2 * HYSTERESIS   # a counted vehicle moved at least this far (implied by the two sides)


def ground_point(bbox: Tuple[int, int, int, int], width: int, height: int) -> Point:
    x1, _, x2, y2 = bbox
    return ((x1 + x2) / 2.0 / max(1, width), y2 / max(1, height))


@dataclass
class CountingLine:
    px: float
    py: float
    nx: float   # unit normal: the dominant direction of travel; crossing along +n is direction +1
    ny: float

    def side(self, pt: Point) -> int:
        d = (pt[0] - self.px) * self.nx + (pt[1] - self.py) * self.ny
        return 0 if abs(d) < HYSTERESIS else (1 if d > 0 else -1)

    def labels(self) -> Tuple[str, str]:
        """Human labels for crossing directions (+1, -1) in image terms."""
        if abs(self.ny) >= abs(self.nx):
            return ("towards camera", "away from camera")
        return ("left to right", "right to left")

    def endpoints(self) -> Tuple[Point, Point]:
        """The line clipped to the frame, for drawing (normalised coordinates)."""
        dx, dy = -self.ny, self.nx
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


@dataclass
class MotionModel:
    c2: float = 0.0
    s2: float = 0.0
    sx: float = 0.0
    sy: float = 0.0
    weight: float = 0.0
    tracks: int = 0
    line: Optional[CountingLine] = field(default=None)

    def add(self, start: Point, end: Point) -> bool:
        """Adds one trajectory; returns True if it was long enough to teach the model."""
        dx, dy = end[0] - start[0], end[1] - start[1]
        length = math.hypot(dx, dy)
        if length < MIN_MOVE:
            return False
        theta = math.atan2(dy, dx)
        self.c2 += length * math.cos(2 * theta)
        self.s2 += length * math.sin(2 * theta)
        self.sx += length * (start[0] + end[0]) / 2.0
        self.sy += length * (start[1] + end[1]) / 2.0
        self.weight += length
        self.tracks += 1
        if self.tracks >= LINE_MIN_TRACKS:
            self.line = self._fit()
        return True

    def axis(self) -> Optional[Point]:
        """Dominant direction of travel (unit vector, canonical sign) or None without data."""
        if self.weight <= 0 or math.hypot(self.c2, self.s2) < 1e-9:
            return None
        theta = 0.5 * math.atan2(self.s2, self.c2)
        nx, ny = math.cos(theta), math.sin(theta)
        # Canonical sign so direction labels stay stable as the model refines.
        if (abs(ny) >= abs(nx) and ny < 0) or (abs(nx) > abs(ny) and nx < 0):
            nx, ny = -nx, -ny
        return nx, ny

    def _fit(self) -> Optional[CountingLine]:
        axis = self.axis()
        if axis is None:
            return None
        return CountingLine(self.sx / self.weight, self.sy / self.weight, axis[0], axis[1])

    def to_dict(self) -> Dict[str, float]:
        return {"c2": self.c2, "s2": self.s2, "sx": self.sx, "sy": self.sy,
                "weight": self.weight, "tracks": self.tracks}

    @classmethod
    def from_dict(cls, d: Dict[str, float]) -> "MotionModel":
        m = cls(**{k: d[k] for k in ("c2", "s2", "sx", "sy", "weight", "tracks") if k in d})
        if m.tracks >= LINE_MIN_TRACKS:
            m.line = m._fit()
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
            if (can_count and not self.counted and side != 0 and side != ref_side
                    and math.hypot(pt[0] - self.ref[0], pt[1] - self.ref[1]) >= MIN_CROSS_MOVE):
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


class CountingLines:
    """Motion models for every camera, persisted to a JSON file so learned lines survive restarts."""

    def __init__(self, path: Optional[str] = None):
        self.path = path
        self._models: Dict[int, MotionModel] = {}
        self._lock = threading.Lock()
        self._dirty = False
        if path and os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for cid, d in json.load(f).items():
                        self._models[int(cid)] = MotionModel.from_dict(d)
            except (OSError, ValueError, TypeError):
                self._models = {}

    def model(self, camera_id: int) -> MotionModel:
        with self._lock:
            return self._models.setdefault(camera_id, MotionModel())

    def learn(self, camera_id: int, start: Point, end: Point) -> None:
        with self._lock:
            if self._models.setdefault(camera_id, MotionModel()).add(start, end):
                self._dirty = True

    def line(self, camera_id: int) -> Optional[CountingLine]:
        with self._lock:
            m = self._models.get(camera_id)
            return m.line if m else None

    def save(self) -> None:
        if not self.path:
            return
        with self._lock:
            if not self._dirty:
                return
            data = {str(cid): m.to_dict() for cid, m in self._models.items()}
            self._dirty = False
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f)
        os.replace(tmp, self.path)
