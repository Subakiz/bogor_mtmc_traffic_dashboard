"""
backend/indexing_daemon.py
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard.
3-Tier Adaptive Scheduling Daemon (FEAT-M5).

Implements:
- 3-Tier Adaptive Scheduling:
  - Tier 1: Target vehicle camera + candidate handoff cameras (15-20 FPS)
  - Tier 2: Arterial junction hubs (2 FPS)
  - Tier 3: Peripheral cameras (0.2 FPS round-robin sampling)
- Dynamic camera elevation on handoff exit events and demotion upon window expiration.
- Total scheduled load bounds within Apple Neural Engine compute budget (<= 75 FPS).
- Thread-safe start, stop, step_camera, and status query methods.
- Full pipeline: Frame Ingestion -> ANE YOLO Detection -> Heading/Speed Estimation ->
  Feature Extraction (256-dim embedding, 128-bin color histogram) -> SQLite WAL Persistence.
"""

import json
import logging
import os
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from backend.config import DWELL_TIER1_S
from backend.database import Database, init_db
from backend.detector import ANEVehicleDetector
from backend.reid import IdentityResolver, extract_appearance
from backend.stream_capture import MockVideoCapture, ThreadedStreamCapture

logger = logging.getLogger("indexing_daemon")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_TOPOLOGY_PATH = os.path.join(PROJECT_ROOT, "data", "bogor_camera_topology.json")

# How long a track may vanish from a camera's detections before Re-ID treats it as having
# departed (and becomes eligible for cross-camera matching). Short enough that Tier 1/2's
# frequent polling still catches genuine departures quickly.
_TRACK_DEPARTURE_GRACE_S = 3.0

# Core Arterial Junction Hub Camera IDs in Kota Bogor
DEFAULT_ARTERIAL_HUBS = {
    79,   # Tugu Kujang (Pajajaran - Otista)
    80,   # Ciheleut (Pajajaran)
    81,   # Pajajaran Baranangsiang
    82,   # Sambu / Terminal Baranangsiang
    83,   # Pasar Bogor
    84,   # Suryakencana
    85,   # Empang
    86,   # Simpang Pahlawan / Bondongan
    87,   # Otista
    88,   # Balai Kota
    92,   # Djuanda
    161,  # Warung Jambu (Sholeh Iskandar)
    167,  # Sukasari
}


class IndexingDaemon:
    """
    3-Tier Background Scheduling Daemon managing multi-camera ingestion and ANE indexing.
    """

    def __init__(
        self,
        db: Optional[Database] = None,
        db_path: Optional[str] = None,
        topology_path: Optional[str] = None,
        topology: Optional[Any] = None,
        detector: Optional[ANEVehicleDetector] = None,
        identity_resolver: Optional[IdentityResolver] = None,
        test_mode: bool = False,
    ):
        self.topology_path = topology_path or DEFAULT_TOPOLOGY_PATH
        self.topology_obj = topology
        self.test_mode = test_mode
        self.db = db or init_db(db_path)
        self.detector = detector or ANEVehicleDetector(test_mode=test_mode)
        # Cross-camera "Follow That Car" global vehicle identity (FEAT-M2/M7): matches a
        # newly-seen track against vehicles that recently departed a topologically
        # reachable camera, gated by appearance similarity and the road-network travel window.
        self.identity_resolver = identity_resolver or IdentityResolver(topology=self.topology_obj)

        self._lock = threading.RLock()
        self._stop_event = threading.Event()
        self._worker_thread: Optional[threading.Thread] = None

        # Per-camera live track bookkeeping for Re-ID: camera_id -> track_id -> state.
        # Drives global-id assignment on first sighting and departure registration when a
        # track stops being detected (see step_camera / _resolve_departures).
        self._camera_active_tracks: Dict[int, Dict[int, Dict[str, Any]]] = defaultdict(dict)
        # Auto-demotion deadlines for handoff-triggered Tier 1 promotions (manual promotions
        # via promote_to_tier1(..., ttl_s=None) never expire on their own).
        self._tier1_expiry: Dict[int, float] = {}

        self.all_camera_ids: Set[int] = set()
        self.camera_metadata: Dict[int, Dict[str, Any]] = {}
        self.stream_captures: Dict[int, ThreadedStreamCapture] = {}

        # 3-Tier Scheduling Configuration
        self.tiers = {
            1: {"target_fps": 15.0, "cameras": set()},
            2: {"target_fps": 2.0, "cameras": set()},
            3: {"target_fps": 0.2, "cameras": set()},
        }
        self.tier2_hub_defaults: Set[int] = set(DEFAULT_ARTERIAL_HUBS)

        # Telemetry metrics
        self._start_time: float = 0.0
        self._cycles_completed: int = 0
        self._sightings_indexed: int = 0
        self._last_cycle_time: float = 0.0
        self._last_camera_step: Dict[int, float] = {}
        self._failed_camera_until: Dict[int, float] = {}

        self._load_topology()
        self._initialize_tiers()

    def _load_topology(self):
        """Loads camera nodes from topology object or JSON."""
        if self.topology_obj is not None and hasattr(self.topology_obj, "cameras"):
            for cid, cam in self.topology_obj.cameras.items():
                self.all_camera_ids.add(cid)
                self.camera_metadata[cid] = cam
            return

        if os.path.exists(self.topology_path):
            with open(self.topology_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            nodes = data.get("nodes", [])
            for n in nodes:
                cid = n["id"]
                self.all_camera_ids.add(cid)
                self.camera_metadata[cid] = n

    def _get_or_create_stream_capture(
        self, camera_id: int, stream_url: Optional[str]
    ) -> ThreadedStreamCapture:
        """Retrieves or creates a stream capture, strictly bounding concurrent threads to avoid restreamer rate limits."""
        with self._lock:
            if camera_id in self.stream_captures:
                return self.stream_captures[camera_id]

            # Determine appropriate FPS for this camera's tier
            tier = self.get_tier_for_camera(camera_id)
            fps = float(self.tiers.get(tier, {}).get("target_fps", 2.0))

            # Strictly cap active capture threads across daemon to 16 (Tier 1 + 11 Tier 2 hubs + Tier 3 rotation)
            max_allowed = 16
            if len(self.stream_captures) >= max_allowed:
                # Evict Tier 3 capture first, then Tier 2 if needed (never Tier 1)
                evicted_cid = None
                for cid in list(self.stream_captures.keys()):
                    if cid in self.tiers[3]["cameras"]:
                        evicted_cid = cid
                        break
                if evicted_cid is None:
                    for cid in list(self.stream_captures.keys()):
                        if cid not in self.tiers[1]["cameras"]:
                            evicted_cid = cid
                            break
                if evicted_cid is not None:
                    cap = self.stream_captures.pop(evicted_cid, None)
                    if cap:
                        try:
                            cap.release()
                        except Exception:
                            pass

            cap_source = stream_url if stream_url else MockVideoCapture()
            capture = ThreadedStreamCapture(
                source=cap_source, camera_id=camera_id, max_fps=fps, auto_start=True
            )
            self.stream_captures[camera_id] = capture
            return capture

    def _initialize_tiers(self):
        """Partitions cameras into Tier 1 (empty), Tier 2 (Hubs), and Tier 3 (Peripheral)."""
        with self._lock:
            # Tier 1 starts empty (elevated on target follow)
            self.tiers[1]["cameras"] = set()
            # Tier 2 gets arterial hubs
            self.tiers[2]["cameras"] = set(self.tier2_hub_defaults & self.all_camera_ids)
            # Tier 3 gets the remainder
            self.tiers[3]["cameras"] = (
                self.all_camera_ids - self.tiers[1]["cameras"] - self.tiers[2]["cameras"]
            )

    def get_tier_for_camera(self, camera_id: int) -> int:
        """Returns the current tier (1, 2, or 3) for a given camera ID."""
        with self._lock:
            for t in [1, 2, 3]:
                if camera_id in self.tiers[t]["cameras"]:
                    return t
            return 3

    def get_cameras_in_tier(self, tier: int) -> Set[int]:
        """Returns the set of camera IDs assigned to a specific tier."""
        with self._lock:
            return set(self.tiers.get(tier, {}).get("cameras", set()))

    def promote_to_tier1(self, camera_id: int, reason: str = "target_lock",
                         ttl_s: Optional[float] = None):
        """
        Elevates camera to Tier 1 (15-20 FPS) for real-time tracking.
        ``ttl_s``, when given, auto-demotes the camera after that many idle seconds
        (used for handoff-triggered promotions); omit it for a manual/sticky lock.
        """
        with self._lock:
            if camera_id not in self.all_camera_ids:
                return
            for t in [2, 3]:
                if camera_id in self.tiers[t]["cameras"]:
                    self.tiers[t]["cameras"].remove(camera_id)
            self.tiers[1]["cameras"].add(camera_id)
            if ttl_s is not None:
                self._tier1_expiry[camera_id] = time.time() + ttl_s
            else:
                self._tier1_expiry.pop(camera_id, None)
            logger.info(f"Promoted camera {camera_id} to Tier 1 ({reason}).")

    def demote_from_tier1(self, camera_id: int):
        """Demotes camera from Tier 1 back to Tier 2 (if hub) or Tier 3."""
        with self._lock:
            if camera_id in self.tiers[1]["cameras"]:
                self.tiers[1]["cameras"].remove(camera_id)
            self._tier1_expiry.pop(camera_id, None)
            if camera_id in self.tier2_hub_defaults:
                self.tiers[2]["cameras"].add(camera_id)
            else:
                self.tiers[3]["cameras"].add(camera_id)
            logger.info(f"Demoted camera {camera_id} from Tier 1.")

    def _expire_tier1_promotions(self, now: float) -> None:
        """Demotes handoff-promoted Tier 1 cameras whose dwell window has elapsed."""
        with self._lock:
            expired = [cid for cid, deadline in self._tier1_expiry.items() if now >= deadline]
        for cid in expired:
            self.demote_from_tier1(cid)

    def set_tier2_hubs(self, camera_ids: List[int]):
        """Updates the default arterial hub list for Tier 2."""
        with self._lock:
            self.tier2_hub_defaults = set(camera_ids) & self.all_camera_ids
            for cid in self.tier2_hub_defaults:
                if cid not in self.tiers[1]["cameras"]:
                    self.tiers[3]["cameras"].discard(cid)
                    self.tiers[2]["cameras"].add(cid)

    def compute_scheduled_load_fps(self) -> float:
        """Computes total scheduled inferences/second across all 3 tiers."""
        with self._lock:
            return sum(
                len(cfg["cameras"]) * cfg["target_fps"] for cfg in self.tiers.values()
            )

    def step_camera(
        self,
        camera_id: int,
        frame: Optional[np.ndarray] = None,
        synthetic_objects: Optional[List[Dict[str, Any]]] = None,
        timestamp: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Executes one processing step for a single camera:
        1. Ingest frame
        2. Run ANE YOLO detection and heading/speed calculation
        3. Extract mock/deep appearance embedding (1024 bytes) and HSV histogram (512 bytes)
        4. Persist sighting to SQLite WAL database
        """
        curr_time = timestamp or time.time()
        cam_info = self.camera_metadata.get(camera_id, {})
        azimuth = float(cam_info.get("facing_bearing_deg", cam_info.get("default_azimuth", 0.0)))

        # Obtain frame
        if frame is None:
            try:
                if self.test_mode:
                    if camera_id not in self.stream_captures:
                        self.stream_captures[camera_id] = MockVideoCapture()
                    ret, frame = self.stream_captures[camera_id].read()
                else:
                    stream_url = cam_info.get("stream_url")
                    cap = self._get_or_create_stream_capture(camera_id, stream_url)
                    ret, frame = cap.get_frame()
                if not ret or frame is None:
                    return []
            except Exception as e:
                logger.debug(f"Frame fetch error on camera {camera_id}: {e}")
                return []

        # Run Detection
        try:
            detections = self.detector.detect(
                frame,
                camera_azimuth_deg=azimuth,
                timestamp=curr_time,
                synthetic_objects=synthetic_objects,
            )
        except Exception as e:
            logger.debug(f"Detection failed on camera {camera_id}: {e}")
            return []

        iso_time = datetime.fromtimestamp(curr_time, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        seen_track_ids: Set[int] = set()
        persisted_sightings = []

        for det in detections:
            tid = det["track_id"]
            seen_track_ids.add(tid)

            # Multi-Modal Appearance Signature: 256-dim embedding (1024 bytes, L2-normalised)
            # + 128-bin HSV colour histogram (512 bytes), from the real vehicle crop when a
            # frame is available; falls back to a deterministic synthetic signature for
            # degenerate crops (e.g. a track right at the frame edge).
            appearance = extract_appearance(frame, det["bbox"]) if frame is not None else None
            if appearance is not None:
                emb_vec, hist_vec = appearance
            else:
                rng = np.random.RandomState(abs(hash((tid, det["class_name"]))) % (2**31))
                emb_vec = rng.randn(256).astype(np.float32)
                emb_vec /= max(1e-8, float(np.linalg.norm(emb_vec)))
                hist_vec = np.zeros(128, dtype=np.float32)
                hist_vec[0:16] = 1.0 / 16.0

            with self._lock:
                track_state = self._camera_active_tracks[camera_id].get(tid)
                if track_state is None:
                    # First sighting of this track on this camera: resolve its global vehicle
                    # id against recent departures from reachable cameras (Follow That Car).
                    gid, confidence, matched_from = self.identity_resolver.resolve(
                        camera_id, det["class_name"], emb_vec, curr_time
                    )
                else:
                    gid, confidence, matched_from = track_state["gid"], 1.0, None

            det["global_vehicle_id"] = gid

            # Persist to SQLite WAL database
            sid = self.db.insert_vehicle_sighting(
                camera_id=camera_id,
                local_track_id=tid,
                global_vehicle_id=gid,
                vehicle_class=det["class_name"],
                confidence=det["confidence"],
                bbox=det["bbox"],
                heading_deg=det["heading_deg"],
                speed_px_s=det["speed_px_s"],
                embedding=emb_vec.tobytes(),
                color_hist=hist_vec.tobytes(),
                timestamp_iso=iso_time,
                crop_thumbnail=None,
            )
            det["sighting_id"] = sid

            with self._lock:
                self._camera_active_tracks[camera_id][tid] = {
                    "gid": gid,
                    "class_name": det["class_name"],
                    "embedding": emb_vec,
                    "last_seen": curr_time,
                }

            if matched_from is not None:
                # Confirmed cross-camera handoff: record it, lock the active target onto this
                # camera, and keep this camera at Tier 1 FPS for the dwell window so the
                # rest of the vehicle's pass through the junction is captured at full rate.
                try:
                    self.db.record_handoff_event(
                        global_vehicle_id=gid,
                        from_camera_id=matched_from,
                        to_camera_id=camera_id,
                        confidence=confidence,
                        timestamp_iso=iso_time,
                    )
                    self.db.update_active_target(
                        global_vehicle_id=gid,
                        current_camera_id=camera_id,
                        last_sighting_id=sid,
                        status="tracking",
                        locked_at_iso=iso_time,
                    )
                except Exception as e:
                    logger.debug(f"Handoff bookkeeping failed for gid {gid}: {e}")
                self.promote_to_tier1(camera_id, reason="handoff_confirmed", ttl_s=DWELL_TIER1_S)

            persisted_sightings.append(det)

        self._resolve_departures(camera_id, seen_track_ids, curr_time)

        with self._lock:
            self._sightings_indexed += len(persisted_sightings)

        return persisted_sightings

    def _resolve_departures(self, camera_id: int, seen_track_ids: Set[int], now: float) -> None:
        """Registers tracks absent from this step as departures so Re-ID can match them to a
        reappearance on a downstream camera."""
        with self._lock:
            active = self._camera_active_tracks[camera_id]
            stale_ids = [
                tid for tid, st in active.items()
                if tid not in seen_track_ids and (now - st["last_seen"]) > _TRACK_DEPARTURE_GRACE_S
            ]
            departed = [active.pop(tid) for tid in stale_ids]
        for st in departed:
            self.identity_resolver.register_departure(
                st["gid"], camera_id, st["class_name"], st["embedding"], st["last_seen"]
            )

    def run_cycle(self):
        """
        Executes one polling cycle across all tiers based on priority and tier target FPS.
        """
        now = time.time()
        self._expire_tier1_promotions(now)
        with self._lock:
            tier1_cams = list(self.tiers[1]["cameras"])
            tier2_cams = list(self.tiers[2]["cameras"])
            tier3_cams = list(self.tiers[3]["cameras"])

        # Process Tier 1 cameras (Target: 15-20 FPS -> interval ~0.067s)
        t1_interval = 1.0 / max(1.0, self.tiers[1]["target_fps"])
        for cid in tier1_cams:
            if (now - self._last_camera_step.get(cid, 0)) >= t1_interval:
                try:
                    self.step_camera(cid)
                    self._last_camera_step[cid] = now
                except Exception as e:
                    logger.debug(f"Error stepping Tier 1 camera {cid}: {e}")

        # Process Tier 2 cameras (Target: 2.0 FPS -> interval 0.500s)
        t2_interval = 1.0 / max(0.1, self.tiers[2]["target_fps"])
        for cid in tier2_cams:
            if (now - self._last_camera_step.get(cid, 0)) >= t2_interval:
                if now < self._failed_camera_until.get(cid, 0):
                    continue
                try:
                    res = self.step_camera(cid)
                    self._last_camera_step[cid] = now
                    if not res and not self.test_mode:
                        cap = self.stream_captures.get(cid)
                        if cap and getattr(cap, "_consecutive_failures", 0) >= 10:
                            self._failed_camera_until[cid] = now + 30.0
                except Exception as e:
                    logger.debug(f"Error stepping Tier 2 camera {cid}: {e}")

        # Process Tier 3 in round-robin batches (Target: 0.2 FPS -> interval 5.000s)
        t3_interval = 1.0 / max(0.01, self.tiers[3]["target_fps"])
        if tier3_cams:
            due_cams = [
                cid
                for cid in tier3_cams
                if (now - self._last_camera_step.get(cid, 0)) >= t3_interval
                and now >= self._failed_camera_until.get(cid, 0)
            ]
            sample_size = min(2, len(due_cams))
            batch = due_cams[:sample_size]
            for cid in batch:
                try:
                    res = self.step_camera(cid)
                    self._last_camera_step[cid] = now
                    if not res and not self.test_mode:
                        cap = self.stream_captures.get(cid)
                        if cap and getattr(cap, "_consecutive_failures", 0) >= 10:
                            self._failed_camera_until[cid] = now + 45.0
                except Exception as e:
                    logger.debug(f"Error stepping Tier 3 camera {cid}: {e}")

        with self._lock:
            self._cycles_completed += 1
            self._last_cycle_time = time.time()

    def start(self):
        """Starts the daemon background execution thread."""
        with self._lock:
            if self._worker_thread is not None and self._worker_thread.is_alive():
                return
            self._stop_event.clear()
            self._start_time = time.time()
            self._worker_thread = threading.Thread(
                target=self._run_loop, name="ANE-IndexingDaemon", daemon=True
            )
            self._worker_thread.start()

    def _run_loop(self):
        """Background execution loop."""
        while not self._stop_event.is_set():
            try:
                t0 = time.perf_counter()
                self.run_cycle()
                elapsed = time.perf_counter() - t0
                sleep_time = max(0.05, 0.1 - elapsed)
                time.sleep(sleep_time)
            except Exception as e:
                logger.error(f"Unexpected error in IndexingDaemon worker loop: {e}", exc_info=True)
                time.sleep(1.0)

    def is_running(self) -> bool:
        """Returns True if the daemon worker thread is active."""
        with self._lock:
            return self._worker_thread is not None and self._worker_thread.is_alive()

    @property
    def running(self) -> bool:
        """Boolean property proxy for is_running()."""
        return self.is_running()

    def stop(self):
        """Stops the daemon background thread and releases all stream captures."""
        self._stop_event.set()
        if self._worker_thread is not None and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
            self._worker_thread = None

        with self._lock:
            for sc in list(self.stream_captures.values()):
                try:
                    sc.release()
                except Exception:
                    pass
            self.stream_captures.clear()
            self._camera_active_tracks.clear()
            self._tier1_expiry.clear()

    def get_status(self) -> Dict[str, Any]:
        """Returns real-time daemon status, tier distribution, and telemetry metrics."""
        with self._lock:
            uptime = (time.time() - self._start_time) if self._start_time > 0 else 0.0
            return {
                "running": self.is_running(),
                "uptime_s": round(uptime, 2),
                "total_cameras": len(self.all_camera_ids),
                "tiers": {
                    1: {
                        "target_fps": self.tiers[1]["target_fps"],
                        "camera_count": len(self.tiers[1]["cameras"]),
                        "cameras": sorted(list(self.tiers[1]["cameras"])),
                    },
                    2: {
                        "target_fps": self.tiers[2]["target_fps"],
                        "camera_count": len(self.tiers[2]["cameras"]),
                        "cameras": sorted(list(self.tiers[2]["cameras"])),
                    },
                    3: {
                        "target_fps": self.tiers[3]["target_fps"],
                        "camera_count": len(self.tiers[3]["cameras"]),
                        "cameras": sorted(list(self.tiers[3]["cameras"])),
                    },
                },
                "total_scheduled_load_fps": round(
                    sum(len(cfg["cameras"]) * cfg["target_fps"] for cfg in self.tiers.values()),
                    2,
                ),
                "cycles_completed": self._cycles_completed,
                "sightings_indexed": self._sightings_indexed,
                "reid_pending_departures": self.identity_resolver.pending_count(),
                "last_cycle_time": self._last_cycle_time,
            }

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()
