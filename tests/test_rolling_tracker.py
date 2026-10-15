"""
tests/test_rolling_tracker.py
Rolling multi-camera tracking: vehicles are counted on every camera, not only the viewed one.

The tracker is driven deterministically through ``tick(now)`` with fake HLS streams and a fake
detector that finds a red "vehicle" patch in each frame, so scheduling, counting, parked-vehicle
deduplication and focus handling are tested without CoreML or the network.
"""

import json
import os
import tempfile
import threading
import time
import unittest
from unittest import mock
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from backend.database import Database
from backend import counting, rolling_tracker
from backend.rolling_tracker import FIRST_FRAME_TIMEOUT_S, FOCUS_EVICT_IDLE_S, RollingTracker
from backend.topology_engine import TopologyEngine

FrameFn = Callable[[int], Optional[Tuple[int, int, int, int]]]  # frame index -> car bbox or None


def _frame(bbox: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    img = np.full((360, 640, 3), 60, dtype=np.uint8)
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        img[y1:y2, x1:x2] = (30, 30, 220)        # red car body
        img[y1:(y1 + y2) // 2, x1:x2] = (40, 40, 160)  # darker roof keeps the descriptor non-trivial
    return img


class FakeStream:
    def __init__(self, fn: Optional[FrameFn]):
        self.fn = fn
        self.k = 0
        self.released = False

    def get_new_frame(self, last_seq=0, timeout=0.5, copy=False):
        if self.fn is None:          # dead stream: connects but never delivers a frame
            return False, last_seq, None
        self.k += 1
        return True, self.k, _frame(self.fn(self.k))

    def release(self):
        self.released = True


class FakeDetector:
    """Finds the red patch; one track per camera until reset, ids unique across cameras."""

    def __init__(self):
        self.last_inference_ms = 1.0
        self._next_id = 1
        self._tracks: Dict[int, Tuple[int, float, float, float]] = {}  # cam -> (tid, cx, cy, ts)
        self.resets: List[int] = []

    def detect(self, frame, camera_azimuth_deg=0.0, timestamp=None, camera_id=0, **_):
        mask = (frame[:, :, 2] > 150) & (frame[:, :, 0] < 100)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            self._tracks.pop(camera_id, None)    # vehicle left the view: the next one is a new track
            return []
        x1, y1, x2, y2 = int(xs.min()), int(ys.min()) - 60, int(xs.max()) + 1, int(ys.max()) + 1
        y1 = max(0, y1)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        prev = self._tracks.get(camera_id)
        if prev is None:
            tid, speed = self._next_id, 0.0
            self._next_id += 1
        else:
            tid = prev[0]
            dt = max(1e-3, timestamp - prev[3])
            speed = float(np.hypot(cx - prev[1], cy - prev[2]) / dt)
        self._tracks[camera_id] = (tid, cx, cy, timestamp)
        return [{
            "track_id": tid, "class_id": 2, "class_name": "car", "confidence": 0.9,
            "bbox": (x1, y1, x2, y2), "centroid": (int(cx), int(cy)),
            "heading_deg": 0.0, "speed_px_s": speed,
        }]

    def reset_tracks(self, camera_id=None):
        self.resets.append(camera_id)
        self._tracks.pop(camera_id, None)


class FakeTopology:
    """Three cameras in a line: 1 -> 2 -> 3."""

    def __init__(self, ids=(1, 2, 3)):
        self.cameras = {
            cid: {"id": cid, "name": f"CAM {cid}", "stream_url": f"https://example.test/{cid}.m3u8",
                  "default_azimuth": 0.0}
            for cid in ids
        }
        self.edges = [{"source_id": a, "target_id": b} for a, b in zip(ids, ids[1:])]


class StubAnalytics:
    def __init__(self):
        self.frames: Dict[int, int] = {}
        self.flushes: List[int] = []
        self.crossings: Dict[int, int] = {}
        self.records_missing_counted_now = 0

    def record_frame(self, camera_id, camera_name, timestamp, active_counts, detections, frame_shape):
        self.frames[camera_id] = self.frames.get(camera_id, 0) + 1
        self.records_missing_counted_now += sum(1 for d in detections if "counted_now" not in d)
        self.crossings[camera_id] = self.crossings.get(camera_id, 0) + sum(1 for d in detections if d.get("counted_now"))

    def flush_camera(self, camera_id, min_observed_s=2.0):
        self.flushes.append(camera_id)


def static_car(_k):
    return (200, 160, 320, 260)


def moving_car(k):
    x = 40 + 6 * k
    return (x, 160, x + 120, 260)


def passing_traffic(k):
    """A car drives left to right across the frame every 80 frames, with a gap between cars."""
    step = k % 80
    if step >= 60:
        return None
    x = 10 + 8 * step
    return (x, 160, x + 120, 260)


class TrackerTestBase(unittest.TestCase):
    """Shared helpers: a tracker driven by tick(now) over fake streams and a fake detector."""

    def make(self, frame_fns: Dict[int, Optional[FrameFn]], topology=None, **kw):
        self.db = Database(":memory:", auto_seed=False)
        topology = topology or FakeTopology()
        for cid, cam in topology.cameras.items():
            self.db.conn.execute(
                "INSERT INTO cameras (id, name, corridor, latitude, longitude, stream_url) VALUES (?,?,?,?,?,?)",
                (cid, cam["name"], "TEST", -6.6, 106.8, cam["stream_url"]),
            )
        self.db.conn.commit()
        self.streams: Dict[int, FakeStream] = {}

        def factory(cid, url, fps):
            self.streams[cid] = FakeStream(frame_fns.get(cid, static_car))
            return self.streams[cid]

        self.detector = FakeDetector()
        self.analytics = StubAnalytics()
        params = dict(
            db=self.db, topology=topology, analytics=self.analytics, detector=self.detector,
            stream_factory=factory, threaded_streams=False, autostart=False, rolling_enabled=True,
            focus_fps=10.0, rolling_fps=10.0, dwell_s={1: 0.6, 2: 0.6, 3: 0.6},
            tier1_revisit_s=0.0, tier2_revisit_s=0.0, max_open_streams=2, focus_ttl_s=30.0, hubs=set(),
            standby_slots=0,     # the scheduling tests below assume every free slot is an active visit
        )
        params.update(kw)
        self.tracker = RollingTracker(**params)
        self.t0 = time.time()
        return self.tracker

    def run_for(self, seconds: float, start: float = 0.0, step: float = 0.05):
        t = start
        while t < start + seconds:
            self.tracker.tick(self.t0 + t)
            self.assertLessEqual(len(self.tracker.pool.open_ids()), self.tracker.max_open_streams)
            t += step
        return t

    def counts(self) -> Dict[int, int]:
        rows = self.db.conn.execute(
            "SELECT camera_id, COUNT(*) AS n FROM vehicle_sightings GROUP BY camera_id").fetchall()
        return {r["camera_id"]: r["n"] for r in rows}

    def tearDown(self):
        if hasattr(self, "tracker"):
            self.tracker.stop()
            self.db.close()


class RollingTrackerTest(TrackerTestBase):

    def test_counts_vehicles_on_every_camera_without_a_viewer(self):
        self.make({})
        self.run_for(6.0)
        self.assertEqual(set(self.counts()), {1, 2, 3}, "every camera should be visited and counted")
        status = self.tracker.get_status()
        self.assertIsNone(status["focus_camera_id"])
        self.assertEqual(status["cameras_covered"], 3)
        self.assertGreater(status["frames_processed"], 30)
        # Each visit closes its analytics window over the observed span.
        for cid in (1, 2, 3):
            self.assertIn(cid, self.analytics.flushes)
            self.assertGreater(self.analytics.frames[cid], 0)

    def test_parked_vehicle_counted_once_across_frames_and_revisits(self):
        self.make({})
        self.run_for(10.0)
        visits = self.tracker._visit_counts
        self.assertTrue(all(visits[c] >= 2 for c in (1, 2, 3)), visits)
        # One stationary car per camera: one sighting each despite many frames and revisits.
        self.assertEqual(self.counts(), {1: 1, 2: 1, 3: 1})
        self.assertGreaterEqual(self.tracker.get_status()["parked_dedups"], 3)

    def test_moving_traffic_is_counted_on_each_visit(self):
        self.make({1: moving_car, 2: moving_car, 3: moving_car})
        self.run_for(10.0)
        # A moving vehicle at a revisit is a new passage, so counts grow with visits.
        for cid in (1, 2, 3):
            self.assertGreaterEqual(self.counts()[cid], 2)

    def test_focus_camera_tracked_and_downstream_cameras_visited_first(self):
        self.make({}, max_open_streams=2, tier1_revisit_s=0.0)
        self.tracker.set_focus(1)
        self.tracker.tick(self.t0)
        self.assertEqual(self.tracker.get_status()["downstream_of_focus"], [2])
        # One slot is reserved for the focus camera, the other goes to its downstream neighbour.
        self.assertEqual(sorted(self.tracker._visits), [2])
        self.run_for(3.0, start=0.05)
        latest = self.tracker.get_focus_frame(1)
        self.assertIsNotNone(latest)
        self.assertEqual(latest["detections"][0]["class_name"], "car")
        self.assertIsNone(self.tracker.get_focus_frame(2))
        self.assertNotIn(1, self.tracker._visit_counts, "focus camera must not be rolled")
        self.assertEqual(self.counts().get(1), 1)

    def test_focus_lapses_without_viewer_polls(self):
        self.make({}, focus_ttl_s=1.0)
        self.tracker.set_focus(1)
        self.run_for(0.5)
        self.assertEqual(self.tracker.get_status()["focus_camera_id"], 1)
        self.run_for(1.5, start=0.5)
        self.assertIsNone(self.tracker.get_status()["focus_camera_id"])
        self.assertIsNone(self.tracker.get_focus_frame(1))
        self.assertIn(1, self.analytics.flushes)

    def test_switching_focus_to_a_camera_mid_visit_keeps_its_tracks(self):
        self.make({}, max_open_streams=3)
        self.run_for(0.3)
        self.assertIn(2, self.tracker._visits)
        self.tracker.set_focus(2)
        self.run_for(0.2, start=0.3)
        self.assertNotIn(2, self.tracker._visits)
        self.assertIn(2, self.tracker._focus)
        self.assertEqual(self.counts().get(2), 1)

    def test_two_viewers_on_different_cameras_both_get_tracked_frames(self):
        # Regression: a single focus slot made two viewers steal focus from each other on every
        # poll, restarting both sessions, so both feeds stayed on "CONNECTING" forever.
        self.make({}, max_open_streams=4, max_focus_cameras=3)
        t = 0.0
        while t < 3.0:
            self.tracker.set_focus(1)
            self.tracker.set_focus(3)
            self.tracker.tick(self.t0 + t)
            t += 0.05
        self.assertEqual(sorted(self.tracker._focus), [1, 3])
        self.assertIsNotNone(self.tracker.get_focus_frame(1))
        self.assertIsNotNone(self.tracker.get_focus_frame(3))
        self.assertNotIn(1, self.detector.resets, "focus session for camera 1 was restarted")
        self.assertNotIn(3, self.detector.resets, "focus session for camera 3 was restarted")
        self.assertEqual(self.counts().get(1), 1)
        self.assertEqual(self.counts().get(3), 1)

    def test_focus_cap_evicts_least_recently_viewed_idle_camera(self):
        self.make({}, max_open_streams=4, max_focus_cameras=2)
        for cid in (1, 2):
            self.assertTrue(self.tracker.set_focus(cid))
            time.sleep(0.01)
        # Camera 1's viewer went quiet: its slot can be handed to a new viewer.
        self.tracker._focus_touched[1] -= FOCUS_EVICT_IDLE_S + 1.0
        self.assertTrue(self.tracker.set_focus(3))
        self.tracker.tick(self.t0)
        self.assertEqual(sorted(self.tracker._focus), [2, 3])
        self.assertEqual(self.tracker.get_status()["focus_camera_ids"], [3, 2])

    def test_extra_viewer_is_refused_instead_of_evicting_active_ones(self):
        # Regression: with every focus slot actively polled, one more viewer evicted the least
        # recently polled camera on each poll, so all viewers kept restarting each other's
        # sessions and streams and none of them ever got a frame.
        self.make({}, max_open_streams=4, max_focus_cameras=2)
        refused = 0
        t = 0.0
        while t < 2.0:
            for cid in (1, 2, 3):
                refused += not self.tracker.set_focus(cid)
            self.tracker.tick(self.t0 + t)
            t += 0.05
        self.assertEqual(sorted(self.tracker._focus), [1, 2])
        self.assertGreater(refused, 0)
        self.assertIsNotNone(self.tracker.get_focus_frame(1))
        self.assertIsNotNone(self.tracker.get_focus_frame(2))
        self.assertNotIn(1, self.detector.resets)
        self.assertNotIn(2, self.detector.resets)

    def test_unknown_camera_does_not_take_a_focus_slot(self):
        self.make({}, max_focus_cameras=1)
        self.assertFalse(self.tracker.set_focus(999))
        self.assertTrue(self.tracker.set_focus(1))

    def test_new_focus_takes_a_slot_from_a_visit_without_marking_it_failed(self):
        self.make({}, max_open_streams=2)
        self.run_for(0.3)
        self.assertEqual(len(self.tracker._visits), 2)
        self.tracker.set_focus(3)
        self.run_for(0.3, start=0.3)
        self.assertIn(3, self.tracker._focus)
        self.assertLessEqual(len(self.tracker._visits) + len(self.tracker._focus), 2)
        self.assertEqual(self.tracker.get_status()["cameras_backing_off"], [])

    def test_rolling_disabled_tracks_only_the_focus_camera(self):
        self.make({}, rolling_enabled=False)
        self.tracker.set_focus(3)
        self.run_for(2.0)
        self.assertEqual(set(self.counts()), {3})
        self.assertEqual(self.tracker._visits, {})

    def test_dead_stream_backs_off_instead_of_blocking_a_slot(self):
        self.make({2: None}, max_open_streams=1)
        self.run_for(FIRST_FRAME_TIMEOUT_S + 6.0, step=0.1)
        self.assertIn(2, self.tracker.get_status()["cameras_backing_off"])
        self.assertIn(1, self.counts())
        self.assertIn(3, self.counts())
        self.assertNotIn(2, self.counts())

    def test_stream_that_fails_to_open_is_released_and_backed_off(self):
        self.make({}, max_open_streams=1)
        base_factory = self.tracker.pool._factory

        def factory(cid, url, fps):
            cap = base_factory(cid, url, fps)
            cap.is_opened = (lambda: False) if cid == 1 else (lambda: True)
            return cap

        self.tracker.pool._factory = factory
        self.run_for(3.0)
        self.assertTrue(self.streams[1].released, "failed capture must be released immediately")
        self.assertIn(1, self.tracker.get_status()["cameras_backing_off"])
        self.assertNotIn(2, self.tracker.get_status()["cameras_backing_off"])
        self.assertEqual(set(self.counts()), {2, 3})

    def _failing_factory(self, should_fail):
        base_factory = self.tracker.pool._factory
        self.opens = []

        def factory(cid, url, fps):
            cap = base_factory(cid, url, fps)
            failing = should_fail(cid)
            if cid == 1:
                self.opens.append(failing)
            cap.is_opened = (lambda: False) if failing else (lambda: True)
            return cap
        self.tracker.pool._factory = factory

    def test_repeated_failures_back_off_longer_with_jitter(self):
        self.make({}, max_open_streams=1)
        self._failing_factory(lambda cid: cid == 1)
        retry_gaps, last_fail = [], None
        with mock.patch.object(rolling_tracker, "FAILED_BACKOFF_S", 1.0), \
                mock.patch.object(rolling_tracker, "MAX_FAILED_BACKOFF_S", 8.0), \
                mock.patch.object(rolling_tracker, "OPEN_RETRY_S", 0.0):
            t, seen = 0.0, 0
            while t < 40.0:
                self.tracker.tick(self.t0 + t)
                if len(self.opens) > seen:
                    seen = len(self.opens)
                    if last_fail is not None:
                        retry_gaps.append(t - last_fail)
                    last_fail = t
                t += 0.05
        streak = self.tracker._fail_streak[1]
        self.assertGreaterEqual(streak, 4)
        self.assertTrue(all(b > a for a, b in zip(retry_gaps[:3], retry_gaps[1:4])), retry_gaps)
        self.assertLessEqual(max(retry_gaps), 8.0 * 1.2 + 15.0)   # capped (plus waiting for a free slot)
        status = self.tracker.get_status()["cameras_unreachable"]
        self.assertEqual(status[1]["failures"], streak)

    def test_recovered_camera_resets_its_failure_streak(self):
        self.make({}, max_open_streams=1)
        attempts = {"n": 0}

        def flaky(cid):
            if cid != 1:
                return False
            attempts["n"] += 1
            return attempts["n"] <= 2           # fails twice, then works
        self._failing_factory(flaky)
        with mock.patch.object(rolling_tracker, "FAILED_BACKOFF_S", 1.0), \
                mock.patch.object(rolling_tracker, "OPEN_RETRY_S", 0.0):
            self.run_for(30.0, step=0.05)
        self.assertGreaterEqual(attempts["n"], 3)
        self.assertNotIn(1, self.tracker._fail_streak)
        self.assertEqual(self.tracker.get_status()["cameras_unreachable"], {})
        self.assertIn(1, self.counts())

    def test_visits_are_spread_over_restreamer_hosts(self):
        topo = FakeTopology(ids=tuple(range(1, 9)))
        for cid in (5, 6, 7, 8):
            topo.cameras[cid]["stream_url"] = f"https://other-host.test/{cid}.m3u8"
        self.make({}, topology=topo, max_open_streams=6, max_per_host=2)
        self.tracker.tick(self.t0)
        hosts = [self.tracker._host[cid] for cid in self.tracker._visits]
        self.assertEqual(len(hosts), 4, "2 per host x 2 hosts, even though 6 slots are free")
        self.assertEqual(sorted(hosts.count(h) for h in set(hosts)), [2, 2])

    def test_cameras_failing_together_retry_at_different_times(self):
        topo = FakeTopology(ids=tuple(range(1, 9)))
        self.make({}, topology=topo, max_open_streams=8, max_per_host=8)
        self._failing_factory(lambda cid: True)
        self.tracker.tick(self.t0)          # visits requested, opens fail
        self.tracker.tick(self.t0 + 0.05)   # failures applied
        retry_at = sorted(self.tracker._failed_until.values())
        self.assertEqual(len(retry_at), 8)
        self.assertGreater(retry_at[-1] - retry_at[0], 1.0, "retries should be spread out, not in lockstep")

    def test_timed_out_restreamer_backs_off_all_its_cameras(self):
        topo = FakeTopology(ids=(1, 2, 3, 4))
        topo.cameras[4]["stream_url"] = "https://other-host.test/4.m3u8"
        self.make({}, topology=topo, max_open_streams=1)
        base_factory = self.tracker.pool._factory

        def factory(cid, url, fps):
            cap = base_factory(cid, url, fps)
            cap.is_opened = (lambda: False) if cid == 1 else (lambda: True)
            return cap

        self.tracker.pool._factory = factory
        with mock.patch.object(rolling_tracker, "SLOW_OPEN_S", -1.0):
            self.run_for(2.0)
        backing_off = self.tracker.get_status()["cameras_backing_off"]
        self.assertEqual(backing_off, [1, 2, 3], "every camera on the dead host is skipped")
        self.assertEqual(set(self.counts()), {4})

    def test_cameras_removed_from_the_site_are_not_visited(self):
        topo = FakeTopology(ids=(1, 2, 3))
        topo.cameras[2]["status"] = "removed"
        self.make({}, topology=topo)
        self.run_for(4.0)
        self.assertNotIn(2, self.tracker.cameras)
        self.assertEqual(set(self.counts()), {1, 3})


class DownDetector(FakeDetector):
    """The production detector with no loadable model: every frame is unobserved."""

    def detect(self, *args, **kwargs):
        return None


class DetectorOutageTest(TrackerTestBase):
    def test_frames_the_detector_could_not_process_are_not_observed_time(self):
        self.make({1: passing_traffic, 2: passing_traffic, 3: passing_traffic}, detector=DownDetector())
        self.tracker.set_focus(1)
        self.run_for(6.0)
        self.assertEqual(self.analytics.frames, {}, "no frame may reach analytics as an empty road")
        credited = self.db.conn.execute(
            "SELECT COUNT(*) FROM observation_windows WHERE observed_s > 0").fetchone()[0]
        self.assertEqual(credited, 0, "an outage must not be stored as observed time with 0 crossings")
        self.assertEqual(self.tracker.get_status()["frames_processed"], 0)
        self.assertIsNone(self.tracker.get_focus_frame(1))


class SwitchableDetector(FakeDetector):
    """The production detector whose model is down until ``down`` is cleared."""

    def __init__(self):
        super().__init__()
        self.down = True

    def detect(self, *args, **kwargs):
        return None if self.down else super().detect(*args, **kwargs)


class ModelRecoveryTest(TrackerTestBase):
    def test_model_outage_does_not_back_off_cameras_whose_streams_work(self):
        detector = SwitchableDetector()
        self.make({}, detector=detector)
        t = self.run_for(rolling_tracker.FIRST_FRAME_TIMEOUT_S + 6.0)    # visits time out while the model is down
        self.assertEqual(self.tracker._fail_streak, {}, "the streams were fine; the model was not")
        self.assertEqual(self.tracker.get_status()["cameras_unreachable"], {})
        self.assertEqual(self.analytics.frames, {})
        detector.down = False
        self.run_for(5.0, start=t)                                       # far less than FAILED_BACKOFF_S
        self.assertEqual(set(self.analytics.frames), {1, 2, 3}, "every camera visited again promptly")
        self.assertEqual(set(self.counts()), {1, 2, 3})
        self.assertEqual(self.tracker._fail_streak, {})


class BackoffOverflowTest(TrackerTestBase):
    def test_a_camera_that_fails_thousands_of_times_does_not_overflow_the_backoff(self):
        self.make({})
        self.tracker._fail_streak[1] = 5000      # a dead focus camera retried every ~5 s for days
        now = self.t0
        self.tracker._mark_failed(1, now, "stream did not open")   # raised OverflowError at streak ~1025
        self.assertLessEqual(self.tracker._failed_until[1], now + rolling_tracker.MAX_FAILED_BACKOFF_S * 1.2)


if __name__ == "__main__":
    unittest.main()


class VolumeCountingTest(TrackerTestBase):
    """Line/passage crossings stored per observation window, and rates derived from them."""

    def windows(self, cid=None):
        q = "SELECT * FROM observation_windows" + (" WHERE camera_id = ?" if cid else "") + " ORDER BY window_id"
        return [dict(r) for r in self.db.conn.execute(q, (cid,) if cid else ()).fetchall()]

    def test_windows_store_crossings_and_observed_time(self):
        self.make({1: passing_traffic, 2: static_car, 3: None}, max_open_streams=1,
                  dwell_s={1: 12.0, 2: 12.0, 3: 12.0})
        self.run_for(30.0, step=0.1)
        w1, w2 = self.windows(1), self.windows(2)
        self.assertTrue(w1 and w2)
        self.assertGreater(sum(w["vehicles"] for w in w1), 0, "passing cars must be counted")
        self.assertEqual(sum(w["vehicles"] for w in w2), 0, "a parked car is seen but never counted")
        self.assertGreater(sum(w["observed_s"] for w in w2), 5.0, "quiet time still counts as observed")
        self.assertTrue(all(w["kind"] == "visit" for w in w1 + w2))
        self.assertTrue(all(w["dir_pos"] + w["dir_neg"] <= w["vehicles"] for w in w1))
        self.assertEqual(sum(w["cars"] for w in w1), sum(w["vehicles"] for w in w1))

        rates = self.db.get_camera_rates(window_s=3600, now=self.t0 + 31)
        obs = sum(w["observed_s"] for w in w1)
        vehicles = sum(w["vehicles"] for w in w1)
        self.assertAlmostEqual(rates[1]["vph"], round(vehicles / obs * 3600, 1), places=1)
        self.assertEqual(rates[2]["vph"], 0.0)
        self.assertNotIn(3, rates)

    def test_excluded_camera_records_no_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "overrides.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"1": {"exclude": True}}, f)
            self.make({1: passing_traffic, 2: passing_traffic, 3: None}, max_open_streams=1,
                      dwell_s={1: 12.0, 2: 12.0, 3: 12.0}, counting=counting.CountingLines(None, path))
            self.run_for(30.0, step=0.1)
        self.assertEqual(self.windows(1), [], "no observation windows: not counted, rather than 0 vph")
        self.assertEqual(self.tracker._counted.get(1, 0), 0)
        self.assertGreater(len(self.windows(2)), 0)
        self.assertGreater(self.tracker._counted.get(2, 0), 0)
        self.assertEqual(self.analytics.crossings.get(1, 0), 0)

    def test_counting_learns_the_frame_aspect(self):
        self.make({1: passing_traffic}, rolling_enabled=False, focus_ttl_s=1e9)
        for t in range(20):
            self.tracker.set_focus(1)
            self.tracker.tick(self.t0 + t * 0.1)
        frame = self.streams[1].get_new_frame()[2]
        self.assertAlmostEqual(self.tracker.counting.model(1).aspect, frame.shape[1] / frame.shape[0])

    def test_analytics_is_told_about_every_counting_line_crossing(self):
        self.make({1: passing_traffic, 2: static_car, 3: None}, max_open_streams=1,
                  dwell_s={1: 12.0, 2: 12.0, 3: 12.0})
        self.run_for(30.0, step=0.1)
        self.assertEqual(self.analytics.records_missing_counted_now, 0)
        self.assertGreater(self.analytics.crossings[1], 0)
        self.assertEqual(self.analytics.crossings[1], self.tracker._counted[1],
                         "analytics volume must equal the tracker's counting-line count")
        self.assertEqual(self.analytics.crossings.get(2, 0), 0, "a parked car never crosses")

    def test_watched_camera_stores_a_window_per_minute(self):
        self.make({1: passing_traffic}, rolling_enabled=False, focus_ttl_s=1e9)
        with mock.patch.object(rolling_tracker, "OBS_WINDOW_S", 5.0):
            t = 0.0
            while t < 16.0:
                self.tracker.set_focus(1)
                self.tracker.tick(self.t0 + t)
                t += 0.1
        focus_windows = self.windows(1)
        self.assertGreaterEqual(len(focus_windows), 3)
        self.assertTrue(all(w["kind"] == "focus" for w in focus_windows))
        self.assertTrue(all(4.5 <= w["observed_s"] <= 5.5 for w in focus_windows))
        latest = self.tracker.get_focus_frame(1)
        self.assertEqual(latest["counted_total"],
                         sum(w["vehicles"] for w in focus_windows) + self.tracker._focus[1].win["vehicles"])

    def test_camera_switches_from_passage_to_learned_line(self):
        with mock.patch.object(counting, "LINE_MIN_TRACKS", 3):
            self.make({1: passing_traffic}, rolling_enabled=False, focus_ttl_s=1e9)
            t = 0.0
            while t < 90.0:      # a car every 8 s; the line is learned once 3 have left the view
                self.tracker.set_focus(1)
                self.tracker.tick(self.t0 + t)
                t += 0.1
            self.tracker.stop()
            line = self.tracker.counting.line(1)
            self.assertIsNotNone(line, "line should be learned from passing traffic")
            self.assertAlmostEqual(abs(line.nx), 1.0, places=2)     # cars drive left to right
            ws = self.windows(1)
            self.assertGreater(sum(w["passage_counts"] for w in ws), 0)
            self.assertGreater(sum(w["line_counts"] for w in ws), 0)
            self.assertGreater(sum(w["dir_pos"] for w in ws), 0)
            self.assertEqual(sum(w["dir_neg"] for w in ws), 0)


class StreamPoolConcurrencyTest(unittest.TestCase):
    def test_opens_run_concurrently_and_count_toward_the_connection_cap(self):
        from backend.rolling_tracker import StreamPool

        lock = threading.Lock()
        active, peak = [0], [0]

        def slow_factory(cid, url, fps):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.4)
            with lock:
                active[0] -= 1
            return FakeStream(static_car)

        pool = StreamPool(slow_factory, max_open=3, open_workers=4)
        pool.start()
        try:
            t0 = time.time()
            pool.set_wanted([(c, f"u{c}", 8.0) for c in (1, 2, 3, 4, 5)])   # truncated to max_open
            while len(pool.open_ids()) < 3 and time.time() - t0 < 5:
                time.sleep(0.02)
            elapsed = time.time() - t0
            self.assertEqual(sorted(pool.open_ids()), [1, 2, 3])
            self.assertLess(elapsed, 1.0, "three 0.4 s opens should overlap, not run back to back")
            self.assertLessEqual(peak[0], 3, "opens in flight must respect the connection cap")
        finally:
            pool.stop()
