"""
tests/test_tracker_robustness.py
Latent faults in the rolling tracker: parked entries removed by ``==`` instead of position, and
``stop()`` closing sessions underneath a worker that was still inside a tick.
"""

import threading
import time
import unittest
from unittest import mock


from backend import rolling_tracker
from backend.rolling_tracker import _Parked, _Track
from tests.test_rolling_tracker import FakeDetector, TrackerTestBase


class ParkedMatchingTest(TrackerTestBase):
    BOX = (200, 160, 320, 260)

    def track(self):
        return _Track(1, "car", 0.0, 0.0, self.BOX, 0.9, 0.0, 0.0)

    def test_only_the_matching_parked_entry_is_consumed(self):
        # Regression: _Parked was a dataclass with eq=True, so list.remove() compared entries with ==;
        # entries are removed by position now.
        self.make({})
        elsewhere = _Parked("car", (10, 10, 60, 50), ts=1.0)
        wanted = _Parked("car", self.BOX, ts=1.0)
        self.tracker._parked[1] = [elsewhere, wanted]
        match = self.tracker._match_parked(1, self.track(), now=2.0)
        self.assertIs(match, wanted)
        self.assertEqual(len(self.tracker._parked[1]), 1, "only the matched entry is consumed")
        self.assertIs(self.tracker._parked[1][0], elsewhere)

    def test_identical_entries_are_consumed_one_at_a_time(self):
        self.make({})
        a, b = (_Parked("car", self.BOX, ts=1.0) for _ in range(2))
        self.tracker._parked[1] = [a, b]
        self.assertIs(self.tracker._match_parked(1, self.track(), 2.0), a)
        self.assertIs(self.tracker._match_parked(1, self.track(), 2.0), b)
        self.assertIsNone(self.tracker._match_parked(1, self.track(), 2.0))

    def test_a_parked_entry_of_another_class_is_not_matched(self):
        self.make({})
        self.tracker._parked[1] = [_Parked("truck", self.BOX, ts=1.0)]
        self.assertIsNone(self.tracker._match_parked(1, self.track(), 2.0))

    def test_session_and_track_records_compare_by_identity(self):
        self.make({})
        a = self.track()
        b = self.track()
        self.assertNotEqual(a, b)
        self.assertEqual(a, a)


class BlockingDetector(FakeDetector):
    """Blocks inside ``detect`` once armed, like a worker caught in a long inference call."""

    def __init__(self):
        super().__init__()
        self.armed = False
        self.inside = threading.Event()
        self.release = threading.Event()
        self.busy = False

    def detect(self, *args, **kwargs):
        if self.armed:
            self.busy = True
            self.inside.set()
            self.release.wait(10.0)
            self.busy = False
        return super().detect(*args, **kwargs)


class ShutdownTest(TrackerTestBase):
    def start_worker(self, detector=None):
        self.make({}, max_open_streams=2, dwell_s={1: 100.0, 2: 100.0, 3: 100.0},
                  detector=detector or FakeDetector())
        self.detector = self.tracker._detector
        self.ended = []
        real_end = self.tracker._end_session

        def spy(session, now):
            self.ended.append((session.camera_id, threading.current_thread().name, getattr(self.detector, "busy", False)))
            return real_end(session, now)

        self.tracker._end_session = spy
        self.tracker.start()
        deadline = time.time() + 5.0
        while time.time() < deadline and len(self.analytics.frames) < 2:
            time.sleep(0.01)
        self.assertGreaterEqual(len(self.analytics.frames), 2, "worker should be tracking two cameras")

    def test_clean_stop_ends_every_session_once_and_flushes_analytics(self):
        self.start_worker()
        sessions = sorted(self.tracker._visits)
        self.tracker.stop()
        self.assertFalse(self.tracker.running)
        self.assertEqual(sorted(c for c, _, _ in self.ended), sessions)
        self.assertEqual({thread for _, thread, _ in self.ended}, {"RollingTracker"},
                         "the worker closes its own sessions as it exits")
        self.assertEqual(sorted(self.analytics.flushes), sessions)
        self.tracker.stop()                                  # a second stop is harmless
        self.assertEqual(len(self.ended), len(sessions))
        self.assertEqual(self.tracker._visits, {})

    def test_stop_with_a_stuck_worker_leaves_its_sessions_to_the_worker(self):
        # Regression: stop() joined for 3 s and then called _end_session for every session even if
        # the worker was still inside a tick, racing it on the same dictionaries and trackers.
        detector = BlockingDetector()
        self.start_worker(detector)
        worker = self.tracker._thread
        detector.armed = True
        self.assertTrue(detector.inside.wait(5.0), "worker should be inside detect()")
        with mock.patch.object(rolling_tracker, "STOP_JOIN_S", 0.2):
            t0 = time.time()
            self.tracker.stop()
            self.assertLess(time.time() - t0, 2.0)
        self.assertTrue(worker.is_alive(), "worker is still stuck in detect()")
        self.assertEqual(self.ended, [], "no session may be closed under a running worker")
        self.assertNotEqual(self.tracker._visits, {}, "its state was not touched either")

        detector.armed = False
        detector.release.set()
        worker.join(5.0)
        self.assertFalse(worker.is_alive())
        self.assertEqual(len(self.ended), 2)
        self.assertEqual({thread for _, thread, _ in self.ended}, {"RollingTracker"})
        self.assertFalse(any(busy for _, _, busy in self.ended), "closed between ticks, not inside detect()")
        self.assertEqual(self.tracker._visits, {})
        self.tracker.stop()                                  # and stopping again does not close anything twice
        self.assertEqual(len(self.ended), 2)

    def test_start_does_not_launch_a_second_worker_while_the_old_one_is_still_stopping(self):
        detector = BlockingDetector()
        self.start_worker(detector)
        worker = self.tracker._thread
        detector.armed = True
        self.assertTrue(detector.inside.wait(5.0))
        with mock.patch.object(rolling_tracker, "STOP_JOIN_S", 0.2):
            self.tracker.stop()
            self.tracker.start()
        alive = [t for t in threading.enumerate() if t.name == "RollingTracker" and t.is_alive()]
        self.assertEqual(alive, [worker], "exactly one worker may exist")
        detector.armed = False
        detector.release.set()
        worker.join(5.0)

    def test_tracker_can_be_restarted_after_a_clean_stop(self):
        self.start_worker()
        self.tracker.stop()
        before = self.analytics.frames.copy()
        self.tracker.start()
        deadline = time.time() + 5.0
        while time.time() < deadline and self.analytics.frames == before:
            time.sleep(0.01)
        self.assertNotEqual(self.analytics.frames, before, "processing resumed")
        self.tracker.stop()

    def test_stop_without_a_worker_still_closes_sessions_driven_by_tick(self):
        self.make({}, max_open_streams=2, dwell_s={1: 100.0, 2: 100.0, 3: 100.0})
        for k in range(10):
            self.tracker.tick(self.t0 + k * 0.05)
        self.assertEqual(self.analytics.flushes, [])
        self.tracker.stop()
        self.assertEqual(sorted(self.analytics.flushes), [1, 2])


if __name__ == "__main__":
    unittest.main()
