"""
tests/test_tracker_standby.py
Warm standby streams: the next-ranked cameras are opened and decoding but not inferred, and take over
a freed visit slot at once. Driven deterministically through ``tick(now)`` with fake streams whose
frames only start flowing a configurable time after they are opened.
"""

import os
import time
import types
import unittest
from unittest import mock
from typing import Dict, Optional, Set

from backend import rolling_tracker
from backend.rolling_tracker import FIRST_FRAME_TIMEOUT_S, STALL_TIMEOUT_S
from tests.test_rolling_tracker import FakeStream, FakeTopology, TrackerTestBase, static_car


class TimedStream(FakeStream):
    """A stream that starts delivering frames ``delay`` seconds after it was opened and can be
    told to stop. Time is the test's virtual clock. ``max_fps`` records what the pool asked for."""

    def __init__(self, fn, clock, delay: Optional[float], fps: float = 0.0):
        super().__init__(fn)
        self.clock = clock
        self.opened = clock()
        self.delay = delay            # None: never delivers
        self.stop_at = float("inf")
        self.max_fps = fps

    def get_new_frame(self, last_seq=0, timeout=0.5, copy=False):
        now = self.clock()
        if self.delay is None or now < self.opened + self.delay or now >= self.stop_at:
            return False, last_seq, None
        return super().get_new_frame(last_seq, timeout, copy)


def chain_topology(n: int, hosts: int = 1) -> FakeTopology:
    topo = FakeTopology(ids=tuple(range(1, n + 1)))
    for cid in topo.cameras:
        topo.cameras[cid]["stream_url"] = f"https://host{cid % hosts}.test/{cid}.m3u8"
    return topo


class StandbyTestBase(TrackerTestBase):
    def setUp(self):
        # The stream pool timestamps opens with time.time(); tests run on a virtual clock that is far
        # ahead of the real one, which would make a stream that takes a while to deliver look ancient.
        self.vnow = time.time()
        shim = types.SimpleNamespace(time=lambda: self.vnow, monotonic=time.monotonic, sleep=time.sleep)
        patcher = mock.patch.object(rolling_tracker, "time", shim)
        patcher.start()
        self.addCleanup(patcher.stop)

    def make_standby(self, n_cameras=8, delays: Optional[Dict[int, Optional[float]]] = None,
                     default_delay: float = 0.0, hosts: int = 1, **kw):
        """A tracker over ``n_cameras`` where camera ``c`` starts delivering ``delays[c]`` seconds
        after it is opened (``default_delay`` otherwise)."""
        params = dict(max_open_streams=5, max_per_host=99, standby_slots=2, standby_fps=2.0,
                      rolling_fps=8.0, dwell_s={1: 100.0, 2: 100.0, 3: 100.0})
        params.update(kw)
        tracker = self.make({}, topology=chain_topology(n_cameras, hosts), **params)
        self.vnow = self.t0
        delays = delays or {}

        def factory(cid, url, fps):
            self.streams[cid] = TimedStream(static_car, lambda: self.vnow, delays.get(cid, default_delay), fps)
            self.opens.append(cid)
            return self.streams[cid]

        self.opens = []
        tracker.pool._factory = factory
        return tracker

    def force_tier1(self, cameras: Set[int]) -> Set[int]:
        """Makes ``cameras`` (a live set the test may change) count as downstream of a focus camera
        without opening a focus stream, so they rank tier 1."""
        tracker = self.tracker
        sync = tracker._sync_focus

        def sync_and_force(now):
            sync(now)
            tracker._downstream_of_focus |= cameras

        tracker._sync_focus = sync_and_force
        return cameras

    def step(self, t: float):
        self.vnow = self.t0 + t
        self.tracker.tick(self.vnow)

    def run_until(self, until: float, start: float = 0.0, dt: float = 0.05) -> float:
        t = start
        while t < until:
            self.step(t)
            self.assertLessEqual(len(self.tracker.pool.open_ids()), self.tracker.max_open_streams)
            t += dt
        return t

    def end_visit_now(self, camera_id: int):
        """Makes the running visit of ``camera_id`` reach its dwell on the next tick."""
        session = self.tracker._visits[camera_id]
        session.started = self.vnow - session.dwell_s - 0.01
        session.last_frame_ts = self.vnow

    def states(self):
        return {v["camera_id"]: v["state"] for v in self.tracker.get_status()["visits"]}


class StandbySlotPlanTest(StandbyTestBase):
    def plan(self, max_open, standby, focus=0):
        self.make_standby(max_open_streams=max_open, standby_slots=standby, max_focus_cameras=3)
        for cid in range(1, focus + 1):
            self.tracker._focus[cid] = object()      # only len() matters for the plan
        try:
            return self.tracker._slot_plan()
        finally:
            self.tracker._focus.clear()

    def test_standby_never_leaves_fewer_than_two_active_visits(self):
        self.assertEqual(self.plan(12, 2), (12, 2))
        self.assertEqual(self.plan(12, 2, focus=3), (9, 2))
        self.assertEqual(self.plan(12, 5, focus=3), (9, 5))
        self.assertEqual(self.plan(4, 2), (4, 2))
        self.assertEqual(self.plan(3, 2), (3, 1))
        self.assertEqual(self.plan(2, 2), (2, 0))
        self.assertEqual(self.plan(1, 2), (1, 0))
        self.assertEqual(self.plan(12, 0), (12, 0))
        self.assertEqual(self.plan(5, 2, focus=3), (2, 0))

    def test_standby_is_off_by_default_and_zero_disables(self):
        # Off by default: in simulation standby connections cost more coverage than they save.
        if "ROLLING_STANDBY_SLOTS" not in os.environ:
            self.assertEqual(rolling_tracker.STANDBY_SLOTS, 0)
        self.make_standby(standby_slots=0)
        self.run_until(1.0)
        self.assertEqual(self.tracker._standby, {})
        self.assertEqual(len(self.tracker._visits), 5)


class StandbyOperationTest(StandbyTestBase):
    def test_standby_streams_are_open_at_a_low_rate_and_never_inferred(self):
        self.make_standby()
        self.run_until(2.0)
        t = self.tracker
        self.assertEqual(sorted(t._visits), [1, 2, 3])
        self.assertEqual(sorted(t._standby), [4, 5])
        self.assertEqual(sorted(t.pool.open_ids()), [1, 2, 3, 4, 5], "standby streams count against the cap")
        self.assertEqual({c: self.streams[c].max_fps for c in (1, 4)}, {1: 8.0, 4: 2.0})
        self.assertEqual(set(self.analytics.frames), {1, 2, 3}, "standby cameras are not processed")
        self.assertNotIn(4, self.tracker._visit_counts)
        self.assertEqual(self.counts().keys(), {1, 2, 3})

    def test_status_lists_standby_sessions(self):
        self.make_standby(delays={5: None})
        self.run_until(2.0)
        status = self.tracker.get_status()
        self.assertEqual(self.states(), {1: "observing", 2: "observing", 3: "observing", 4: "standby", 5: "standby"})
        by_id = {v["camera_id"]: v for v in status["visits"]}
        self.assertTrue(by_id[4]["ready"])
        self.assertFalse(by_id[5]["ready"], "its stream is open but has produced nothing")
        self.assertEqual(status["standby_camera_ids"], [4, 5])
        self.assertEqual(status["standby_slots"], 2)
        self.assertEqual(status["open_streams"], [1, 2, 3, 4, 5])

    def test_ready_standby_takes_a_freed_slot_instantly_and_dwell_counts_from_promotion(self):
        # Fresh streams need 0.5 s before the first frame, but standby 4 has been open for ages.
        self.make_standby(default_delay=0.5, dwell_s={1: 3.0, 2: 3.0, 3: 3.0})
        t = self.run_until(2.0)
        self.assertEqual(self.states()[4], "standby")
        # Run to the tick on which the first visit (camera 1) ends.
        while 1 in self.tracker._visits:
            self.step(t)
            t += 0.05
        self.assertEqual(self.tracker._standby_promotions, 2, "both standby streams took over on that tick")
        promoted = self.tracker._visits[4]
        self.assertGreater(promoted.promoted_at, 0.0)
        self.assertGreater(promoted.started, 0.0, "processed on the tick it was promoted: no connecting phase")
        self.assertLess(promoted.started - promoted.promoted_at, 0.05)
        self.assertEqual(promoted.frames, 1)
        self.assertEqual(self.states()[4], "observing")
        # Its dwell is counted from the promotion: still observing 2.5 s later, done after 3 s.
        promoted_at = promoted.started
        while self.vnow < promoted_at + 2.5:
            self.step(t)
            t += 0.05
        self.assertIn(4, self.tracker._visits)
        while 4 in self.tracker._visits:
            self.step(t)
            t += 0.05
        self.assertAlmostEqual(self.vnow - promoted_at, 3.0, delta=0.15)
        self.assertEqual(self.tracker._visit_counts[4], 1)

    def test_without_standby_a_fresh_stream_needs_its_connect_time(self):
        self.make_standby(default_delay=0.5, dwell_s={1: 1.0, 2: 1.0, 3: 1.0}, standby_slots=0)
        t = self.run_until(1.5)
        while 1 in self.tracker._visits and self.tracker._visits[1].frames > 0:
            self.step(t)
            t += 0.05
        waits = []
        for _ in range(40):
            self.step(t)
            t += 0.05
            for s in self.tracker._visits.values():
                if s.started and s.requested_at > self.t0 + 1.0:
                    waits.append(s.started - s.requested_at)
        self.assertTrue(waits)
        self.assertGreaterEqual(min(waits), 0.45)

    def test_a_standby_that_became_ready_is_preferred_over_one_still_connecting(self):
        self.make_standby(delays={4: 30.0})      # camera 4 won't deliver for 30 s, camera 5 is ready at once
        t = self.run_until(1.5)
        self.assertEqual(sorted(self.tracker._standby), [4, 5])
        self.end_visit_now(1)                                    # camera 1's visit ends now
        t = self.run_until(2.0, start=t)
        self.assertIn(5, self.tracker._visits, "the ready standby was promoted")
        self.assertIn(4, self.tracker._standby, "the one that is still connecting stays on standby")

    def test_unready_standby_is_promoted_when_nothing_is_ready_and_waits_for_its_first_frame(self):
        self.make_standby(delays={4: 1.0, 5: 1.0}, default_delay=0.0)
        t = self.run_until(0.4)
        self.assertEqual(sorted(self.tracker._standby), [4, 5])
        self.assertFalse(any(v["ready"] for v in self.tracker.get_status()["visits"] if v["state"] == "standby"))
        self.end_visit_now(1)
        self.step(t)
        t += 0.05
        self.assertIn(4, self.tracker._visits, "a freed slot is never left idle")
        self.assertEqual(self.states()[4], "connecting")
        t = self.run_until(2.0, start=t)
        self.assertEqual(self.states()[4], "observing")

    def test_standby_that_never_delivers_is_marked_failed_and_replaced(self):
        self.make_standby(delays={4: None})
        self.run_until(FIRST_FRAME_TIMEOUT_S + 2.0, dt=0.1)
        t = self.tracker
        self.assertIn(4, t.get_status()["cameras_backing_off"])
        self.assertEqual(t._fail_streak[4], 1)
        self.assertNotIn(4, t._standby)
        self.assertNotIn(4, t._visits)
        self.assertEqual(len(t._standby), 2)
        self.assertIn(6, t._standby, "the next-ranked camera took its place")
        self.assertTrue(self.streams[4].released, "its stream was released")

    def test_standby_that_fails_to_open_is_backed_off_and_replaced(self):
        self.make_standby()
        base = self.tracker.pool._factory

        def factory(cid, url, fps):
            cap = base(cid, url, fps)
            cap.is_opened = (lambda: False) if cid == 4 else (lambda: True)
            return cap
        self.tracker.pool._factory = factory
        self.run_until(2.0)
        t = self.tracker
        self.assertIn(4, t.get_status()["cameras_backing_off"])
        self.assertEqual(sorted(t._standby), [5, 6])
        self.assertTrue(self.streams[4].released)

    def test_a_timed_out_host_takes_its_standby_streams_down_too(self):
        self.make_standby(n_cameras=8, hosts=2, max_open_streams=6)      # host0: even ids, host1: odd ids
        base = self.tracker.pool._factory

        def factory(cid, url, fps):
            cap = base(cid, url, fps)
            cap.is_opened = (lambda: False) if cid == 4 else (lambda: True)
            return cap
        self.tracker.pool._factory = factory
        with mock.patch.object(rolling_tracker, "SLOW_OPEN_S", -1.0):
            self.run_until(0.5)
        t = self.tracker
        for cid in (2, 4, 6, 8):
            self.assertNotIn(cid, t._standby)
            self.assertNotIn(cid, t._visits)
            self.assertIn(cid, t.get_status()["cameras_backing_off"])

    def test_standby_streams_obey_the_per_host_cap(self):
        self.make_standby(n_cameras=8, hosts=2, max_open_streams=6, max_per_host=3)
        for step in range(60):
            self.step(step * 0.1)
            load = self.tracker._host_load()
            self.assertTrue(all(n <= 3 for n in load.values()), load)
        self.assertEqual(sum(self.tracker._host_load().values()), 6)
        self.assertEqual(len(self.tracker._standby), 2)

    def test_stalled_standby_is_dropped_without_a_failure_and_ranked_last(self):
        self.make_standby()
        self.run_until(1.0)
        self.streams[4].stop_at = self.vnow          # camera 4's stream dies while on standby
        self.run_until(1.0 + STALL_TIMEOUT_S + 1.0, start=1.0)
        t = self.tracker
        self.assertNotIn(4, t._standby)
        self.assertNotIn(4, t.get_status()["cameras_backing_off"], "it delivered earlier: not a failure")
        self.assertEqual(t._fail_streak, {})
        self.assertEqual(sorted(t._standby), [5, 6])

    def test_rolling_disabled_releases_the_standby_streams(self):
        self.make_standby()
        self.run_until(1.0)
        self.assertEqual(len(self.tracker._standby), 2)
        self.tracker.rolling_enabled = False
        self.run_until(1.5, start=1.0)
        self.assertEqual(self.tracker._standby, {})
        self.assertEqual(sorted(self.tracker.pool.open_ids()), [1, 2, 3], "only the running visits keep streams")


class StandbyTrimAndPriorityTest(StandbyTestBase):
    def test_focus_takes_standby_slots_before_active_visits(self):
        self.make_standby(max_open_streams=6, max_focus_cameras=3)
        self.run_until(1.0)
        t = self.tracker
        self.assertEqual((sorted(t._visits), sorted(t._standby)), ([1, 2, 3, 4], [5, 6]))
        active = dict(t._visits)
        for i, session in enumerate(active.values()):      # visits that started a moment apart
            session.requested_at = session.active_since = self.t0 + 0.001 * i

        t.set_focus(8)
        self.step(1.0)
        self.assertEqual(len(t._standby), 1, "first focus camera costs one standby slot")
        self.assertEqual(t._visits, active, "active visits untouched")

        t.set_focus(7)
        self.step(1.05)
        self.assertEqual(len(t._standby), 0)
        self.assertEqual(t._visits, active)
        self.assertLessEqual(len(t.pool.open_ids()), 6)

        t.set_focus(6)        # a third watched camera: now an active visit has to give way
        self.step(1.1)
        self.assertEqual(len(t._visits) + len(t._standby) + len(t._focus), 6)
        self.assertEqual(len(t._visits), 3, "only now does a visit give up its slot")
        self.assertEqual(sorted(t._visits), [1, 2, 3], "the most recently started visit goes first")

    def test_trimmed_standby_is_neither_failed_nor_counted_as_visited(self):
        self.make_standby(max_open_streams=6, max_focus_cameras=3)
        self.run_until(1.0)
        self.tracker.set_focus(8)
        self.step(1.0)
        t = self.tracker
        dropped = ({5, 6} - set(t._standby)).pop()
        self.assertNotIn(dropped, t._last_visit_end)
        self.assertNotIn(dropped, t._failed_until)
        self.assertNotIn(dropped, t._visit_counts)

    def test_tier1_camera_without_a_stream_preempts_a_standby_slot_quickly(self):
        topo = chain_topology(9)
        self.make_standby(max_open_streams=6, max_focus_cameras=3, tier1_revisit_s=0.0)
        self.tracker.topology = topo
        self.run_until(1.0)
        t = self.tracker
        self.assertEqual((sorted(t._visits), sorted(t._standby)), ([1, 2, 3, 4], [5, 6]))
        # Camera 8 becomes downstream of a focus camera, i.e. tier 1.
        self.force_tier1({8})
        self.step(1.0)
        self.assertEqual(t._tier(8, self.vnow), 1)
        self.assertIn(8, t._standby, "tier-1 camera 8 displaced a tier-3 standby on the very next tick")
        self.assertEqual(len(t._standby), 2)
        self.assertEqual(len(t._visits), 4, "no active visit was touched")
        evicted = {5, 6} - set(t._standby)
        self.assertEqual(len(evicted), 1)
        gone = evicted.pop()
        self.assertNotIn(gone, t._failed_until)
        self.assertNotIn(gone, t._last_visit_end)

        # The tier-1 standby is promoted before the other (lower-tier) standby when a slot frees up.
        self.end_visit_now(1)
        self.run_until(1.5, start=1.05)
        self.assertIn(8, t._visits)

    def test_tier1_displacement_respects_the_host_cap(self):
        self.make_standby(n_cameras=9, hosts=3, max_open_streams=6, max_per_host=2, max_focus_cameras=3)
        self.run_until(1.0)
        t = self.tracker
        before = dict(t._standby)
        victim_hosts = {t._host[c] for c in before}
        # A tier-1 camera on a host that is already at its cap, and not the host of any standby.
        full_hosts = [h for h, n in t._host_load().items() if n >= 2 and h not in victim_hosts]
        if not full_hosts:
            self.skipTest("layout did not produce a full host without standby streams")
        cand = next(c for c, h in t._host.items() if h == full_hosts[0] and c not in t._visits and c not in t._standby)
        self.force_tier1({cand})
        self.step(1.0)
        self.assertNotIn(cand, t._standby, "host is at its cap: the swap would exceed it")
        self.assertEqual(dict(t._standby), before)


class StandbyInvariantsTest(StandbyTestBase):
    def test_slot_accounting_holds_under_random_focus_failures_and_toggles(self):
        import random
        rng = random.Random(7)
        delays = {c: rng.choice([0.0, 0.3, 0.8, 2.0, None]) for c in range(1, 15)}
        self.make_standby(n_cameras=14, hosts=3, max_open_streams=7, max_per_host=3, max_focus_cameras=3,
                          delays=delays, standby_slots=2, dwell_s={1: 1.5, 2: 1.5, 3: 1.5}, focus_ttl_s=1.0,
                          tier1_revisit_s=2.0)
        tier1 = self.force_tier1(set())
        t = 0.0
        for step in range(2500):
            if rng.random() < 0.02:
                self.tracker.set_focus(rng.randint(1, 14))
            if rng.random() < 0.002:
                self.tracker.set_rolling_enabled(not self.tracker.rolling_enabled)
            if rng.random() < 0.01:
                self.tracker.rolling_enabled = True
            if rng.random() < 0.01:                       # downstream-of-focus set changes
                tier1.clear()
                tier1.add(rng.randint(1, 14))
            self.step(t)
            t += 0.05
            tr = self.tracker
            ids = list(tr._focus) + list(tr._visits) + list(tr._standby)
            self.assertEqual(len(ids), len(set(ids)), f"a camera holds two slots at step {step}")
            self.assertLessEqual(len(ids), tr.max_open_streams)
            self.assertLessEqual(len(tr.pool.open_ids()), tr.max_open_streams)
            visit_slots, standby_cap = tr._slot_plan()
            self.assertLessEqual(len(tr._visits) + len(tr._standby), visit_slots)
            self.assertLessEqual(len(tr._standby), standby_cap)
            # (a viewer may open a watched camera on a host that is already at its cap; only the
            # rolling visits and standby streams are held to it)
            rolling = {}
            for cid in list(tr._visits) + list(tr._standby):
                rolling[tr._host[cid]] = rolling.get(tr._host[cid], 0) + 1
            self.assertTrue(all(n <= tr.max_per_host for n in rolling.values()), rolling)
            for s in tr._standby.values():
                self.assertTrue(s.standby and not s.started and not s.tracks and s.frames == 0)
            for s in tr._visits.values():
                self.assertFalse(s.standby)
        self.assertGreater(tr._standby_promotions, 20)


class StandbyThroughputTest(StandbyTestBase):
    def test_slots_keep_getting_filled_when_visits_end_back_to_back(self):
        self.make_standby(default_delay=0.5, dwell_s={1: 2.0, 2: 2.0, 3: 2.0})
        self.run_until(20.0)
        t = self.tracker
        self.assertGreaterEqual(t._standby_promotions, 6)
        self.assertEqual(len(t._visits), 3)
        self.assertEqual(len(t._standby), 2)
        self.assertEqual(t.get_status()["cameras_unreachable"], {})


if __name__ == "__main__":
    unittest.main()
