"""
backend/rolling_tracker.py
Rolling multi-camera vehicle tracking and counting for the BSW Kota Bogor dashboard.

One inference worker drives the shared ANE detector for two kinds of work:

- Focus: every camera a viewer is watching (several tabs or viewers can watch different cameras,
  up to MAX_FOCUS_CAMERAS) is tracked continuously at FOCUS_FPS. Its newest frame and detections are
  exposed to the video endpoints. Each focus lapses after FOCUS_TTL_S without a viewer poll.
- Rolling visits: up to ``MAX_OPEN_STREAMS - 1`` other cameras are observed concurrently, each
  for a short dwell burst at ROLLING_FPS (long enough for ByteTrack to follow vehicles through
  the view), then released for the next camera. Priority:
    tier 1  follow candidates: downstream of the focus camera, or of a followed target's last
            handoff (until its arrival window closes). Revisited every TIER1_REVISIT_S.
    tier 2  arterial hubs, revisited every TIER2_REVISIT_S.
    tier 3  everything else, least recently visited first.

HLS connects take seconds and OpenCV opens are serialised, so streams are opened and released by
a separate pool thread; the inference worker never blocks on the network.

Sightings: once a track has been seen in MIN_TRACK_HITS frames it gets a global vehicle id from
Re-ID and a single ``vehicle_sightings`` row (journeys, Follow-That-Car). A stationary vehicle still
in place on the next visit inherits its earlier id instead of becoming a new sighting.

Volume: vehicles are counted when they cross the camera's counting line (backend/counting.py,
learned from traffic motion; "passage" mode until learned), with direction. Counts are stored per
observation window (``observation_windows``: one per visit, one per minute for watched cameras)
together with the seconds actually observed, so vehicles/hour = crossings / observed time.
Every processed frame also feeds the per-camera TrafficAnalyticsEngine; ``flush_camera`` closes
its window at the end of a visit.
"""

import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse

import numpy as np

from backend.config import (
    DWELL_TIER1_S, DWELL_TIER2_S, DWELL_TIER3_S, FOCUS_FPS, FOCUS_TTL_S, MAX_FOCUS_CAMERAS,
    MAX_OPEN_STREAMS, MAX_STREAMS_PER_HOST, ROLLING_ENABLED, ROLLING_FPS, TIER1_REVISIT_S,
    TIER2_REVISIT_S,
)
from backend.counting import CountingLines, TrackCounter, ground_point
from backend.indexing_daemon import DEFAULT_ARTERIAL_HUBS
from backend.reid import IdentityResolver, extract_appearance

logger = logging.getLogger("rolling_tracker")

MIN_TRACK_HITS = 3             # frames a track must be seen in before it is counted
TRACK_DEPARTURE_GRACE_S = 3.0  # unseen this long within an observation -> departed
PARKED_SPEED_PX_S = 6.0        # below this a track still in view at visit end counts as parked
PARKED_IOU = 0.5
PARKED_SIMILARITY = 0.90
PARKED_TTL_S = 1800.0
PARKED_MAX_PER_CAMERA = 64
FIRST_FRAME_TIMEOUT_S = 12.0   # stream open but no frame yet
REQUEST_TIMEOUT_S = 45.0       # stream requested but never opened (queued behind other opens)
STALL_TIMEOUT_S = 4.0          # visit ends early if its stream stops producing frames
FAILED_BACKOFF_S = 120.0       # skip a camera this long after it failed to deliver frames...
MAX_FAILED_BACKOFF_S = 1800.0  # ...doubling with each consecutive failure up to this (±20% jitter)
OPEN_RETRY_S = 5.0             # pause before re-opening a wanted stream whose open failed
SLOW_OPEN_S = 4.0              # a failed open slower than this means the restreamer host is down
HOST_BACKOFF_S = 60.0          # skip every camera on a host that timed out
OBS_WINDOW_S = 60.0            # a watched camera's counts are stored as one window per minute
MIN_WINDOW_S = 2.0             # shorter observations are too short to estimate a rate from
OPEN_WORKERS = 6               # HLS streams opened concurrently (opens are slow, not CPU-bound)

StreamFactory = Callable[[int, str, float], Any]


def _default_stream_factory(camera_id: int, url: str, fps: float) -> Any:
    from backend.stream_capture import ThreadedStreamCapture
    return ThreadedStreamCapture(url, camera_id=camera_id, max_fps=fps)


def _iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])  # noqa: E731
    return inter / float(area(a) + area(b) - inter)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── Stream pool ───────────────────────────────────────────────────────────────

class StreamPool:
    """
    Keeps open exactly the streams the tracker currently wants (at most ``max_open``, counting
    opens still in progress), opening and releasing them off the inference thread.
    ``set_wanted`` takes a priority-ordered list. With ``start()`` up to ``open_workers`` streams
    open concurrently (an HLS open takes 2-13 s); without it ``pump()`` opens one per call.
    """

    def __init__(self, factory: StreamFactory, max_open: int, open_workers: int = OPEN_WORKERS):
        self._factory = factory
        self._max_open = max(1, int(max_open))
        self._open_workers = max(1, int(open_workers))
        self._lock = threading.Lock()
        self._wanted: List[Tuple[int, str, float]] = []
        self._streams: Dict[int, Any] = {}
        self._opening: Set[int] = set()
        self._opened_at: Dict[int, float] = {}
        self._retry_at: Dict[int, float] = {}
        self._failures: List[Tuple[int, bool]] = []   # (camera_id, slow) since last take_failures()
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._executor: Optional[ThreadPoolExecutor] = None

    def set_wanted(self, wanted: List[Tuple[int, str, float]]) -> None:
        with self._lock:
            self._wanted = list(wanted)[: self._max_open]
            for cid, _, fps in self._wanted:
                cap = self._streams.get(cid)
                if cap is not None and hasattr(cap, "max_fps"):
                    cap.max_fps = fps
        self._wake.set()

    def get(self, camera_id: int) -> Optional[Any]:
        with self._lock:
            return self._streams.get(camera_id)

    def opened_at(self, camera_id: int) -> Optional[float]:
        with self._lock:
            return self._opened_at.get(camera_id)

    def open_ids(self) -> List[int]:
        with self._lock:
            return list(self._streams)

    def opening_ids(self) -> List[int]:
        with self._lock:
            return list(self._opening)

    def take_failures(self) -> List[Tuple[int, bool]]:
        """Streams that failed to open since the last call, as ``(camera_id, slow)``."""
        with self._lock:
            failures, self._failures = self._failures, []
            return failures

    def pump(self) -> bool:
        """Releases unwanted streams and starts opening missing ones in priority order.
        Returns True if a synchronous caller should pump again."""
        with self._lock:
            wanted_ids = {cid for cid, _, _ in self._wanted}
            to_close = [self._streams.pop(cid) for cid in list(self._streams) if cid not in wanted_ids]
            for cid in list(self._opened_at):
                if cid not in self._streams:
                    del self._opened_at[cid]
            now = time.time()
            missing = [w for w in self._wanted
                       if w[0] not in self._streams and w[0] not in self._opening
                       and self._retry_at.get(w[0], 0.0) <= now]
            room = self._max_open - len(self._streams) - len(self._opening)
            # Synchronous callers open one stream per pump(); the executor opens several at once.
            room = min(room, self._open_workers - len(self._opening)) if self._executor is not None else min(room, 1)
            batch = missing[:max(0, room)]
            self._opening.update(w[0] for w in batch)
        for cap in to_close:
            self._release(cap)
        if not batch:
            return False
        if self._executor is None:
            self._open(batch[0])
            return len(missing) > 1
        for w in batch:
            self._executor.submit(self._open, w)
        return False

    def _open(self, wanted: Tuple[int, str, float]) -> None:
        cid, url, fps = wanted
        t_open = time.time()
        try:
            cap = self._factory(cid, url, fps)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Opening stream for camera {cid} failed: {exc}")
            cap = None
        if cap is not None and hasattr(cap, "is_opened") and not cap.is_opened():
            # Release at once: a failed capture keeps reconnecting in the background otherwise.
            self._release(cap)
            cap = None
        with self._lock:
            self._opening.discard(cid)
            if cap is None:
                self._failures.append((cid, time.time() - t_open > SLOW_OPEN_S))
                self._retry_at[cid] = time.time() + OPEN_RETRY_S
            elif (not self._stop.is_set() and any(w[0] == cid for w in self._wanted)
                  and cid not in self._streams):
                self._streams[cid] = cap
                self._opened_at[cid] = time.time()
                self._retry_at.pop(cid, None)
                cap = None
        if cap is not None:
            self._release(cap)
        self._wake.set()

    @staticmethod
    def _release(cap: Any) -> None:
        try:
            cap.release()
        except Exception:  # noqa: BLE001
            pass

    def _run(self) -> None:
        while not self._stop.is_set():
            self.pump()
            self._wake.wait(0.5)
            self._wake.clear()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._executor = ThreadPoolExecutor(max_workers=self._open_workers, thread_name_prefix="StreamOpen")
        self._thread = threading.Thread(target=self._run, name="RollingStreamPool", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        if self._executor is not None:
            # Opens still in flight release their capture themselves once they finish.
            self._executor.shutdown(wait=False)
            self._executor = None
        with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()
            self._opened_at.clear()
            self._wanted = []
        for cap in streams:
            self._release(cap)


# ── Tracker state ─────────────────────────────────────────────────────────────

@dataclass
class _Track:
    track_id: int
    class_name: str
    first_ts: float
    last_ts: float
    bbox: Tuple[int, int, int, int]
    confidence: float
    heading_deg: float
    speed_px_s: float
    hits: int = 1
    gid: Optional[int] = None
    embedding: Optional[np.ndarray] = None
    counter: Optional[TrackCounter] = None


def _new_window() -> Dict[str, int]:
    return dict.fromkeys(("vehicles", "dir_pos", "dir_neg", "cars", "motorcycles", "trucks", "buses",
                          "line_counts", "passage_counts"), 0)


_CLASS_COLUMN = {"car": "cars", "motorcycle": "motorcycles", "truck": "trucks", "bus": "buses"}


@dataclass
class _Parked:
    gid: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    embedding: np.ndarray
    ts: float


@dataclass
class _Session:
    """One camera under observation, either the focus camera or a rolling visit."""
    camera_id: int
    name: str
    azimuth: float
    kind: str                    # "focus" | "visit"
    tier: int
    dwell_s: float
    requested_at: float
    started: float = 0.0         # time of the first processed frame
    last_frame_ts: float = 0.0
    last_process: float = 0.0
    last_seq: int = 0
    frames: int = 0
    tracks: Dict[int, _Track] = field(default_factory=dict)
    win_start: float = 0.0       # start of the current observation window
    win: Dict[str, int] = field(default_factory=_new_window)
    counted_total: int = 0       # crossings since the session started (for the video overlay)
    counted_pos: int = 0
    counted_neg: int = 0


class RollingTracker:
    """Focus-camera tracking plus rolling coverage of every other camera. See module docstring."""

    def __init__(
        self,
        db: Any,
        topology: Any,
        analytics: Optional[Any] = None,
        detector: Optional[Any] = None,
        detector_factory: Optional[Callable[[], Any]] = None,
        identity_resolver: Optional[IdentityResolver] = None,
        stream_factory: StreamFactory = _default_stream_factory,
        threaded_streams: bool = True,
        rolling_enabled: bool = ROLLING_ENABLED,
        focus_fps: float = FOCUS_FPS,
        rolling_fps: float = ROLLING_FPS,
        dwell_s: Optional[Dict[int, float]] = None,
        tier1_revisit_s: float = TIER1_REVISIT_S,
        tier2_revisit_s: float = TIER2_REVISIT_S,
        max_open_streams: int = MAX_OPEN_STREAMS,
        max_per_host: int = MAX_STREAMS_PER_HOST,
        max_focus_cameras: int = MAX_FOCUS_CAMERAS,
        focus_ttl_s: float = FOCUS_TTL_S,
        hubs: Optional[Set[int]] = None,
        counting: Optional[CountingLines] = None,
        autostart: bool = True,
    ):
        self.db = db
        self.topology = topology
        self.analytics = analytics
        self._detector = detector
        self._detector_factory = detector_factory
        self.identity_resolver = identity_resolver or IdentityResolver(topology=topology)
        self.counting = counting or CountingLines()   # in-memory unless given a persisted one
        self.pool = StreamPool(stream_factory, max_open_streams)
        self._threaded_streams = threaded_streams
        self._autostart = autostart

        self.rolling_enabled = rolling_enabled
        self.focus_interval = 1.0 / max(0.1, focus_fps)
        self.rolling_interval = 1.0 / max(0.1, rolling_fps)
        self.dwell_s = dwell_s or {1: DWELL_TIER1_S, 2: DWELL_TIER2_S, 3: DWELL_TIER3_S}
        self.revisit_s = {1: tier1_revisit_s, 2: tier2_revisit_s, 3: 0.0}
        self.max_open_streams = max(1, int(max_open_streams))
        self.max_per_host = max(1, int(max_per_host))
        self.max_focus_cameras = max(1, min(int(max_focus_cameras), self.max_open_streams))
        self.focus_ttl_s = focus_ttl_s
        self.focus_fps = focus_fps
        self.rolling_fps = rolling_fps

        # Only cameras the BSW site still lists (scrape_cctv_streams.py marks the rest "removed").
        self.cameras: Dict[int, Dict[str, Any]] = {
            int(cid): cam for cid, cam in topology.cameras.items()
            if cam.get("stream_url") and cam.get("status", "online") == "online"
        }
        self._host: Dict[int, str] = {
            cid: (urlparse(cam["stream_url"]).hostname or "") for cid, cam in self.cameras.items()
        }
        self.hubs = set(hubs if hubs is not None else DEFAULT_ARTERIAL_HUBS) & set(self.cameras)
        self._downstream: Dict[int, Set[int]] = {}
        for edge in getattr(topology, "edges", []):
            src, dst = int(edge["source_id"]), int(edge["target_id"])
            if src != dst and dst in self.cameras:
                self._downstream.setdefault(src, set()).add(dst)

        # Shared with request threads (set_focus / get_focus_frame / status)
        self._lock = threading.RLock()
        self._focus_touched: Dict[int, float] = {}          # camera_id -> last viewer poll
        self._focus_frames: Dict[int, Dict[str, Any]] = {}

        # Worker-thread state
        self._focus: Dict[int, _Session] = {}
        self._visits: Dict[int, _Session] = {}
        self._follow: Set[int] = set()
        self._boost_until: Dict[int, float] = {}
        self._last_visit_end: Dict[int, float] = {}
        self._failed_until: Dict[int, float] = {}
        self._fail_streak: Dict[int, int] = {}   # consecutive failures per camera
        self._parked: Dict[int, List[_Parked]] = {}

        # Telemetry
        self._observed_s: Dict[int, float] = {}
        self._visit_counts: Dict[int, int] = {}
        self._counted: Dict[int, int] = {}   # line/passage crossings per camera
        self._seen: Dict[int, int] = {}      # vehicle_sightings rows per camera
        self._frames_processed = 0
        self._parked_dedups = 0
        self._handoffs = 0
        self._started_at = 0.0

        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────────────

    @property
    def detector(self) -> Any:
        if self._detector is None and self._detector_factory is not None:
            self._detector = self._detector_factory()
        return self._detector

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.running:
                return
            self._stop.clear()
            self._started_at = time.time()
            if self._threaded_streams:
                self.pool.start()
            self._thread = threading.Thread(target=self._run, name="RollingTracker", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        now = time.time()
        for session in list(self._visits.values()) + list(self._focus.values()):
            self._end_session(session, now)
        self._visits.clear()
        self._focus.clear()
        with self._lock:
            self._focus_touched.clear()
            self._focus_frames.clear()
        self.pool.stop()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                if self.tick(time.time()) == 0:
                    time.sleep(0.01)
            except Exception as exc:  # noqa: BLE001 - keep tracking through transient faults
                logger.error(f"Rolling tracker tick failed: {exc}", exc_info=True)
                time.sleep(0.5)

    # ── API for request handlers ──────────────────────────────────────────────

    def set_focus(self, camera_id: int) -> None:
        """Marks ``camera_id`` as being viewed; call on every viewer poll to keep it alive.
        Beyond ``max_focus_cameras`` the least recently viewed camera stops being a focus."""
        with self._lock:
            self._focus_touched[int(camera_id)] = time.time()
            while len(self._focus_touched) > self.max_focus_cameras:
                del self._focus_touched[min(self._focus_touched, key=self._focus_touched.get)]
        if self._autostart and not self.running:
            self.start()

    def get_focus_frame(self, camera_id: int) -> Optional[Dict[str, Any]]:
        """Newest processed focus frame: ``{seq, frame, detections, ane_ms, timestamp}``."""
        with self._lock:
            return self._focus_frames.get(camera_id)

    def set_rolling_enabled(self, enabled: bool) -> None:
        self.rolling_enabled = bool(enabled)
        if enabled and self._autostart and not self.running:
            self.start()

    def get_status(self) -> Dict[str, Any]:
        now = time.time()
        with self._lock:
            focus_ids = sorted(self._focus_touched, key=self._focus_touched.get, reverse=True)
        visits = [
            {
                "camera_id": s.camera_id,
                "name": s.name,
                "tier": s.tier,
                "state": "observing" if s.started else "connecting",
                "elapsed_s": round(now - s.started, 1) if s.started else 0.0,
                "dwell_s": s.dwell_s,
                "tracks": len(s.tracks),
            }
            for s in list(self._visits.values())
        ]
        visited = set(self._visit_counts) | set(focus_ids)
        return {
            "running": self.running,
            "rolling_enabled": self.rolling_enabled,
            "uptime_s": round(now - self._started_at, 1) if self._started_at else 0.0,
            "focus_camera_id": focus_ids[0] if focus_ids else None,   # most recently viewed
            "focus_camera_ids": focus_ids,
            "follow_candidates": sorted(self._follow),
            "visits": visits,
            "open_streams": sorted(self.pool.open_ids()),
            "max_open_streams": self.max_open_streams,
            "max_streams_per_host": self.max_per_host,
            "focus_fps": self.focus_fps,
            "rolling_fps": self.rolling_fps,
            "total_cameras": len(self.cameras),
            "cameras_covered": len(visited),
            "frames_processed": self._frames_processed,
            "vehicles_counted": sum(self._counted.values()),
            "vehicles_seen": sum(self._seen.values()),
            "counting_lines_learned": sum(1 for c in self.cameras if self.counting.line(c) is not None),
            "parked_dedups": self._parked_dedups,
            "handoffs": self._handoffs,
            "reid_pending_departures": self.identity_resolver.pending_count(),
            "cameras_backing_off": sorted(c for c, t in self._failed_until.items() if t > now),
            "cameras_unreachable": {
                c: {"failures": n, "retry_in_s": round(max(0.0, self._failed_until.get(c, 0.0) - now))}
                for c, n in sorted(self._fail_streak.items())
            },
            "observed_s": {c: round(s, 1) for c, s in self._observed_s.items()},
        }

    # ── scheduling ────────────────────────────────────────────────────────────

    def tick(self, now: float) -> int:
        """One scheduling + processing step. Returns the number of frames processed."""
        self._sync_focus(now)
        self._apply_open_failures(now)
        self._finish_visits(now)
        if self.rolling_enabled:
            self._fill_slots(now)
        self._update_wanted()
        if not self._threaded_streams:
            while self.pool.pump():
                pass

        processed = 0
        for session in list(self._focus.values()):
            if session.win_start and session.last_frame_ts - session.win_start >= OBS_WINDOW_S:
                self._flush_window(session)
            if now - session.last_process >= self.focus_interval:
                processed += self._process(session, now)
        for session in list(self._visits.values()):
            if now - session.last_process >= self.rolling_interval:
                processed += self._process(session, now)
        return processed

    def _sync_focus(self, now: float) -> None:
        with self._lock:
            for cid in [c for c, t in self._focus_touched.items() if now - t > self.focus_ttl_s]:
                del self._focus_touched[cid]
            wanted = {c for c in self._focus_touched if c in self.cameras}

        for cid in [c for c in self._focus if c not in wanted]:
            self._end_session(self._focus.pop(cid), now)
            with self._lock:
                self._focus_frames.pop(cid, None)
        for cid in wanted - set(self._focus):
            session = self._visits.pop(cid, None)
            if session is not None:
                # The camera was mid-visit: keep its tracks and continue as a focus camera.
                session.kind = "focus"
                session.tier = 0
            else:
                session = self._new_session(cid, "focus", tier=0, now=now)
            self._focus[cid] = session

        self._follow = set().union(*(self._downstream.get(c, set()) for c in self._focus)) - set(self._focus)
        self._trim_visits(now)

    def _trim_visits(self, now: float) -> None:
        """Gives stream slots back to focus cameras by ending the most recently started visits."""
        excess = len(self._visits) - (self.max_open_streams - len(self._focus))
        for s in sorted(self._visits.values(), key=lambda s: s.requested_at, reverse=True)[:max(0, excess)]:
            del self._visits[s.camera_id]
            self._end_session(s, now)

    def _apply_open_failures(self, now: float) -> None:
        for cid, slow in self.pool.take_failures():
            down = [cid]
            if slow:
                host = self._host.get(cid)
                down = [c for c, h in self._host.items() if h == host]
                logger.info(f"Restreamer {host} timed out; skipping its {len(down)} cameras "
                            f"for {HOST_BACKOFF_S:.0f}s.")
            for c in down:
                if c == cid:
                    self._mark_failed(cid, now, "stream did not open")
                else:
                    self._failed_until[c] = max(self._failed_until.get(c, 0.0),
                                                now + HOST_BACKOFF_S * random.uniform(0.8, 1.2))
                session = self._visits.pop(c, None)
                if session is not None:
                    self._end_session(session, now)

    def _mark_failed(self, camera_id: int, now: float, reason: str) -> None:
        """Backs a camera off for longer after each consecutive failure, with jitter so cameras
        that failed together (e.g. a restreamer outage) don't all retry at the same moment."""
        streak = self._fail_streak[camera_id] = self._fail_streak.get(camera_id, 0) + 1
        backoff = min(FAILED_BACKOFF_S * 2 ** (streak - 1), MAX_FAILED_BACKOFF_S) * random.uniform(0.8, 1.2)
        self._failed_until[camera_id] = max(self._failed_until.get(camera_id, 0.0), now + backoff)
        if streak == 1:
            logger.warning(f"Camera {camera_id} unreachable ({reason}); backing off, retries get less frequent.")

    def _tier(self, camera_id: int, now: float) -> int:
        if camera_id in self._follow or self._boost_until.get(camera_id, 0.0) > now:
            return 1
        if camera_id in self.hubs:
            return 2
        return 3

    def _rank_candidates(self, now: float) -> List[Tuple[int, int]]:
        busy = set(self._visits) | set(self._focus)
        ranked = []
        for cid in self.cameras:
            if cid in busy or self._failed_until.get(cid, 0.0) > now:
                continue
            tier = self._tier(cid, now)
            since = now - self._last_visit_end.get(cid, 0.0)
            overdue = since - self.revisit_s[tier]
            if tier < 3 and overdue < 0:
                continue
            ranked.append(((tier, -overdue, cid), cid, tier))
        ranked.sort()
        return [(cid, tier) for _, cid, tier in ranked]

    def _fill_slots(self, now: float) -> None:
        slots = self.max_open_streams - len(self._focus)
        free = slots - len(self._visits)
        if free <= 0:
            return
        # Spread connections over the restreamer hosts: no host gets more than ``max_per_host``
        # concurrent streams, however many of its cameras rank highly (a host that is overloaded
        # starts refusing connections and takes all of its cameras out for a while).
        load: Dict[str, int] = {}
        for cid in list(self._focus) + list(self._visits):
            host = self._host.get(cid, "")
            load[host] = load.get(host, 0) + 1
        for cid, tier in self._rank_candidates(now):
            if free <= 0:
                break
            host = self._host.get(cid, "")
            if load.get(host, 0) >= self.max_per_host:
                continue
            load[host] = load.get(host, 0) + 1
            self._visits[cid] = self._new_session(cid, "visit", tier=tier, now=now)
            self._visit_counts[cid] = self._visit_counts.get(cid, 0) + 1
            free -= 1

    def _new_session(self, camera_id: int, kind: str, tier: int, now: float) -> _Session:
        cam = self.cameras[camera_id]
        return _Session(
            camera_id=camera_id,
            name=cam.get("name", f"Camera #{camera_id}"),
            azimuth=float(cam.get("default_azimuth") or 0.0),
            kind=kind,
            tier=tier,
            dwell_s=self.dwell_s.get(tier, DWELL_TIER3_S),
            requested_at=now,
        )

    def _update_wanted(self) -> None:
        wanted: List[Tuple[int, str, float]] = []
        with self._lock:
            order = {c: -t for c, t in self._focus_touched.items()}
        for cid in sorted(self._focus, key=lambda c: order.get(c, 0.0)):   # most recently viewed first
            wanted.append((cid, self.cameras[cid]["stream_url"], self.focus_fps))
        for s in sorted(self._visits.values(), key=lambda s: s.requested_at):
            wanted.append((s.camera_id, self.cameras[s.camera_id]["stream_url"], self.rolling_fps))
        self.pool.set_wanted(wanted)

    def _finish_visits(self, now: float) -> None:
        for cid, s in list(self._visits.items()):
            done = False
            if not s.started:
                opened = self.pool.opened_at(cid)
                waited = (now - opened) if opened is not None else (now - s.requested_at)
                limit = FIRST_FRAME_TIMEOUT_S if opened is not None else REQUEST_TIMEOUT_S
                if waited > limit:
                    self._mark_failed(cid, now, "no video within the timeout")
                    done = True
            elif now - s.started >= s.dwell_s or now - s.last_frame_ts > STALL_TIMEOUT_S:
                done = True
            if done:
                del self._visits[cid]
                self._end_session(s, now)

    # ── per-frame processing ──────────────────────────────────────────────────

    def _process(self, session: _Session, now: float) -> int:
        cap = self.pool.get(session.camera_id)
        if cap is None:
            return 0
        ok, seq, frame = cap.get_new_frame(session.last_seq, timeout=0.0)
        if not ok or frame is None:
            return 0
        session.last_seq = seq
        session.last_process = now
        session.last_frame_ts = now
        if not session.started:
            session.started = now
            session.win_start = now
            if self._fail_streak.pop(session.camera_id, 0):
                logger.warning(f"Camera {session.camera_id} reachable again.")
        session.frames += 1
        self._frames_processed += 1

        detector = self.detector
        detections = detector.detect(frame, camera_azimuth_deg=session.azimuth, timestamp=now,
                                     camera_id=session.camera_id)
        ane_ms = float(getattr(detector, "last_inference_ms", 0.0))
        self._update_tracks(session, frame, detections, now)

        if self.analytics is not None:
            records = [{
                "track_id": d["track_id"], "class_id": d["class_id"], "class_name": d["class_name"],
                "box": tuple(int(v) for v in d["bbox"]), "conf": float(d["confidence"]),
                "counted_now": bool(d.get("counted_now")),   # crossed the counting line in this frame
            } for d in detections]
            counts: Dict[str, int] = {}
            for r in records:
                counts[r["class_name"]] = counts.get(r["class_name"], 0) + 1
            try:
                self.analytics.record_frame(
                    camera_id=session.camera_id, camera_name=session.name, timestamp=now,
                    active_counts=counts, detections=records, frame_shape=frame.shape[:2],
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Analytics record failed on camera {session.camera_id}: {exc}")

        if session.kind == "focus":
            with self._lock:
                line = self.counting.line(session.camera_id)
                self._focus_frames[session.camera_id] = {
                    "camera_id": session.camera_id, "seq": seq, "frame": frame,
                    "detections": detections, "ane_ms": ane_ms, "timestamp": now,
                    "line": line.endpoints() if line else None,
                    "line_labels": line.labels() if line else None,
                    "counted_total": session.counted_total,
                    "counted_pos": session.counted_pos, "counted_neg": session.counted_neg,
                }
        return 1

    def _update_tracks(self, session: _Session, frame: np.ndarray,
                       detections: List[Dict[str, Any]], now: float) -> None:
        height, width = frame.shape[:2]
        model = self.counting.model(session.camera_id)
        for det in detections:
            tid = int(det["track_id"])
            bbox = tuple(int(v) for v in det["bbox"])
            tr = session.tracks.get(tid)
            if tr is None:
                tr = _Track(tid, det["class_name"], now, now, bbox, float(det["confidence"]),
                            float(det.get("heading_deg", 0.0)), float(det.get("speed_px_s", 0.0)))
                session.tracks[tid] = tr
            else:
                tr.last_ts = now
                tr.bbox = bbox
                tr.confidence = float(det["confidence"])
                tr.heading_deg = float(det.get("heading_deg", tr.heading_deg))
                tr.speed_px_s = float(det.get("speed_px_s", tr.speed_px_s))
                tr.hits += 1
            if tr.gid is None and tr.hits >= MIN_TRACK_HITS:
                self._confirm_track(session, tr, frame, now)
            if tr.gid is not None:
                det["global_vehicle_id"] = tr.gid

            pt = ground_point(bbox, width, height)
            if tr.counter is None:
                tr.counter = TrackCounter(pt)
            det["counted_now"] = False
            if tr.counter.update(pt, model, can_count=tr.hits >= MIN_TRACK_HITS):
                self._count(session, tr)
                det["counted_now"] = True
            det["counted"] = tr.counter.counted

        for tid, tr in list(session.tracks.items()):
            if now - tr.last_ts > TRACK_DEPARTURE_GRACE_S:
                del session.tracks[tid]
                self._learn_motion(session.camera_id, tr)
                self._register_departure(session.camera_id, tr)

    def _count(self, session: _Session, tr: _Track) -> None:
        """Records one vehicle crossing in the session's current observation window."""
        c = tr.counter
        w = session.win
        w["vehicles"] += 1
        session.counted_total += 1
        if c.direction > 0:
            w["dir_pos"] += 1
            session.counted_pos += 1
        elif c.direction < 0:
            w["dir_neg"] += 1
            session.counted_neg += 1
        column = _CLASS_COLUMN.get(tr.class_name)
        if column:
            w[column] += 1
        w["line_counts" if c.mode == "line" else "passage_counts"] += 1
        self._counted[session.camera_id] = self._counted.get(session.camera_id, 0) + 1

    def _learn_motion(self, camera_id: int, tr: _Track) -> None:
        if tr.counter is not None:
            self.counting.learn(camera_id, tr.counter.start, tr.counter.last)

    def _flush_window(self, session: _Session) -> None:
        """Stores the current observation window (if long enough) and starts the next one."""
        if not session.win_start:
            return
        observed = session.last_frame_ts - session.win_start
        if observed >= MIN_WINDOW_S:
            row = dict(session.win)
            row.update(camera_id=session.camera_id, start_ts=session.win_start, end_ts=session.last_frame_ts,
                       start_iso=_iso(session.win_start), observed_s=round(observed, 2), kind=session.kind)
            try:
                self.db.insert_observation_window(row)
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"Observation window insert failed on camera {session.camera_id}: {exc}")
        session.win = _new_window()
        session.win_start = session.last_frame_ts

    def _confirm_track(self, session: _Session, tr: _Track, frame: np.ndarray, now: float) -> None:
        appearance = extract_appearance(frame, tr.bbox)
        if appearance is None:
            # Crop too small to describe; wait for the vehicle to come closer, but don't let a
            # vehicle that stays tiny go uncounted.
            if tr.hits < MIN_TRACK_HITS * 3:
                return
            emb = np.zeros(256, dtype=np.float32)
            hist = np.zeros(128, dtype=np.float32)
        else:
            emb, hist = appearance
        cid = session.camera_id
        tr.embedding = emb if appearance is not None else None

        if appearance is not None:
            parked = self._match_parked(cid, tr, emb, now)
            if parked is not None:
                tr.gid = parked.gid
                self._parked_dedups += 1
                return

        if appearance is not None:
            gid, confidence, matched_from = self.identity_resolver.resolve(cid, tr.class_name, emb, now)
        else:
            gid, confidence, matched_from = self.identity_resolver.new_global_id(), 0.0, None
        tr.gid = gid

        iso = _iso(now)
        try:
            sid = self.db.insert_vehicle_sighting(
                camera_id=cid, local_track_id=tr.track_id, global_vehicle_id=gid,
                vehicle_class=tr.class_name, confidence=tr.confidence, bbox=tr.bbox,
                heading_deg=tr.heading_deg, speed_px_s=tr.speed_px_s,
                embedding=emb.astype(np.float32).tobytes(), color_hist=hist.astype(np.float32).tobytes(),
                timestamp_iso=iso,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Sighting insert failed on camera {cid}: {exc}")
            return
        self._seen[cid] = self._seen.get(cid, 0) + 1

        if matched_from is not None and matched_from != cid:
            self._on_handoff(gid, matched_from, cid, confidence, sid, iso, now)

    def _on_handoff(self, gid: int, from_cid: int, to_cid: int, confidence: float,
                    sid: int, iso: str, now: float) -> None:
        self._handoffs += 1
        try:
            self.db.record_handoff_event(global_vehicle_id=gid, from_camera_id=from_cid,
                                         to_camera_id=to_cid, confidence=confidence, timestamp_iso=iso)
            if not self.db.is_active_target(gid):
                return
            self.db.update_active_target(global_vehicle_id=gid, current_camera_id=to_cid,
                                         last_sighting_id=sid, status="tracking", locked_at_iso=iso)
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"Handoff bookkeeping failed for gid {gid}: {exc}")
            return
        # A followed vehicle reached a new junction: prioritise the cameras it can reach next
        # until its arrival window there has closed.
        for nxt in self._downstream.get(to_cid, set()):
            try:
                _, t_max = self.topology.calculate_arrival_window(to_cid, nxt)
            except KeyError:
                continue
            if t_max != float("inf"):
                self._boost_until[nxt] = max(self._boost_until.get(nxt, 0.0), now + float(t_max))

    def _match_parked(self, camera_id: int, tr: _Track, emb: np.ndarray, now: float) -> Optional[_Parked]:
        entries = self._parked.get(camera_id)
        if not entries:
            return None
        entries[:] = [p for p in entries if now - p.ts <= PARKED_TTL_S]
        for p in entries:
            if (p.class_name == tr.class_name and _iou(p.bbox, tr.bbox) >= PARKED_IOU
                    and float(np.dot(emb, p.embedding)) >= PARKED_SIMILARITY):
                entries.remove(p)
                return p
        return None

    def _register_departure(self, camera_id: int, tr: _Track) -> None:
        if tr.gid is not None and tr.embedding is not None:
            self.identity_resolver.register_departure(tr.gid, camera_id, tr.class_name,
                                                      tr.embedding, tr.last_ts)

    def _end_session(self, session: _Session, now: float) -> None:
        """Closes an observation: parks stationary vehicles, registers departures, closes the
        analytics window over the observed span and drops the camera's tracker state."""
        cid = session.camera_id
        self._flush_window(session)
        for tr in session.tracks.values():
            self._learn_motion(cid, tr)
        try:
            self.counting.save()
        except OSError as exc:
            logger.warning(f"Could not save counting lines: {exc}")
        parked = self._parked.setdefault(cid, [])
        for tr in session.tracks.values():
            if tr.gid is None or tr.embedding is None:
                continue
            if tr.speed_px_s < PARKED_SPEED_PX_S:
                parked.append(_Parked(tr.gid, tr.class_name, tr.bbox, tr.embedding, tr.last_ts))
            else:
                self._register_departure(cid, tr)
        del parked[:-PARKED_MAX_PER_CAMERA]
        session.tracks.clear()

        if session.started:
            self._observed_s[cid] = self._observed_s.get(cid, 0.0) + max(0.0, session.last_frame_ts - session.started)
            if self.analytics is not None:
                try:
                    self.analytics.flush_camera(cid)
                except Exception as exc:  # noqa: BLE001
                    logger.debug(f"Analytics flush failed on camera {cid}: {exc}")
        if self._detector is not None:
            try:
                self._detector.reset_tracks(cid)
            except Exception:  # noqa: BLE001
                pass
        self._last_visit_end[cid] = now
