"""
Queued vs parked: a vehicle standing still where moving traffic drives is part of a queue, one
standing elsewhere is parked. Regression for camera 168 (a jammed junction rated "Smooth" because
11 of 16 queued vehicles were classified parked) without undoing the camera 167 fix (a car park
next to the road must not read as a jam).
"""

import os
import tempfile
import unittest

from car_tracker.analytics import (
    FREE_FLOW_RISE, FREE_FLOW_SINK, LANE_MIN_OBS, TrafficAnalyticsEngine, _LaneMap,
)

FW, FH = 1280, 720
FPS = 10.0
LANE_Y = (410, 490)          # the road: a horizontal lane across the lower half
KERB_Y = (100, 180)          # a parking strip well away from it


class LaneScenario(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.engine = TrafficAnalyticsEngine(db_path=os.path.join(self.tmp.name, "ts.db"),
                                             lanes_path=os.path.join(self.tmp.name, "lanes.npz"))
        self.t = 1_791_700_000.0
        self.next_id = 1

    def tearDown(self):
        self.engine.close()
        self.tmp.cleanup()

    def new_id(self) -> int:
        self.next_id += 1
        return self.next_id

    def frame(self, dets, cam=1):
        self.t += 1.0 / FPS
        counts = {}
        for d in dets:
            counts[d["class_name"]] = counts.get(d["class_name"], 0) + 1
        return self.engine.record_frame(camera_id=cam, camera_name="test", timestamp=self.t,
                                        active_counts=counts, detections=dets, frame_shape=(FH, FW))

    @staticmethod
    def car(tid, x, y_band, w=120):
        return {"track_id": tid, "class_id": 2, "class_name": "car",
                "box": (x, y_band[0], x + w, y_band[1]), "conf": 0.9}

    def learn_lane(self, seconds=30.0, cam=1, speed_px=20.0, cars=4):
        """Moving cars along the lane, ids renewed each lap, until the lane map is trusted."""
        ids = [self.new_id() for _ in range(cars)]
        pos = [k * 1400.0 / cars for k in range(cars)]
        last = None
        for _ in range(int(seconds * FPS)):
            dets = []
            for k in range(cars):
                pos[k] += speed_px
                if pos[k] > 1400:
                    pos[k] -= 1400
                    ids[k] = self.new_id()
                if pos[k] < FW - 120:
                    dets.append(self.car(ids[k], pos[k], LANE_Y))
            last = self.frame(dets, cam)
        self.assertTrue(self.engine._lanes[cam].ready, "lane map should be learned")
        return last


class QueueVsParkedTest(LaneScenario):
    def test_stop_and_go_queue_in_a_lane_is_congested_not_parked(self):
        self.learn_lane()
        # A queue fills the lane: ten cars standing (1-px jitter), track ids renewed now and then
        # at staggered times like ByteTrack does in a dense scene.
        slots = [self.new_id() for _ in range(10)]
        live = None
        for i in range(int(25 * FPS)):
            dets = []
            for k in range(10):
                if (i + 7 * k) % 80 == 0:
                    slots[k] = self.new_id()
                dets.append(self.car(slots[k], 5 + k * 125 + (i % 2), LANE_Y))
            live = self.frame(dets)
        self.assertEqual(live["parked_vehicles"], 0)
        self.assertGreaterEqual(live["queued_vehicles"], 6)
        self.assertGreaterEqual(live["congestion_index"], 0.55, live)
        self.assertIn(live["congestion_label"], ("Heavy", "Congested"))
        self.assertIsNotNone(live["lane_occupancy"])
        self.assertGreater(live["lane_occupancy"], live["occupancy_ratio"])   # lanes, not the whole frame

    def test_parked_cars_off_the_lanes_stay_parked_while_traffic_flows(self):
        self.learn_lane()
        parked = [self.new_id() for _ in range(8)]
        movers = [self.new_id() for _ in range(3)]
        pos = [0.0, 400.0, 800.0]
        live = None
        for i in range(int(25 * FPS)):
            dets = [self.car(parked[k], 5 + k * 150, KERB_Y) for k in range(8)]
            for k in range(3):
                pos[k] += 20.0
                if pos[k] > FW - 120:
                    pos[k] = 0.0
                    movers[k] = self.new_id()
                dets.append(self.car(movers[k], pos[k], LANE_Y))
            live = self.frame(dets)
        self.assertEqual(live["parked_vehicles"], 8)
        self.assertEqual(live["queued_vehicles"], 0)
        self.assertLess(live["congestion_index"], 0.25, live)
        self.assertEqual(live["congestion_label"], "Smooth")

    def test_without_learned_lanes_still_vehicles_are_parked_as_before(self):
        cars = [self.new_id() for _ in range(10)]
        live = None
        for _ in range(int(10 * FPS)):
            live = self.frame([self.car(cars[k], 5 + k * 125, LANE_Y) for k in range(10)], cam=2)
        self.assertFalse(live["lanes_learned"])
        self.assertIsNone(live["lane_occupancy"])
        self.assertEqual(live["parked_vehicles"], 10)
        self.assertEqual(live["queued_vehicles"], 0)


class FreeFlowReferenceTest(LaneScenario):
    def run_window(self, speed_px, seconds=20.0):
        """Cars moving along the lane at ``speed_px`` per frame, then the window is closed."""
        ids = [self.new_id() for _ in range(5)]
        pos = [k * 250.0 for k in range(5)]
        for _ in range(int(seconds * FPS)):
            dets = []
            for k in range(5):
                pos[k] += speed_px
                if pos[k] > FW - 120:
                    pos[k] = 0.0
                    ids[k] = self.new_id()
                dets.append(self.car(ids[k], pos[k], LANE_Y))
            self.frame(dets)
        self.engine.flush_camera(1)
        return self.engine._free_flow_bps.get(1)

    def test_reference_rises_fast_and_sinks_slowly(self):
        fast = self.run_window(20.0)          # ~1.7 body sizes / s
        self.assertIsNotNone(fast)
        slow = self.run_window(2.0)           # a crawl: ~0.17 body sizes / s
        # One congested window moves the reference by at most FREE_FLOW_SINK of the gap...
        self.assertGreater(slow, fast * (1 - FREE_FLOW_SINK) - 1e-6)
        self.assertLess(slow, fast)
        self.engine._free_flow_bps[1] = fast / 2
        rose = self.run_window(20.0)
        # ...while a free-flowing one closes FREE_FLOW_RISE of it.
        self.assertGreater(rose, fast / 2 + FREE_FLOW_RISE * (fast / 2) * 0.5)


class LaneMapTest(unittest.TestCase):
    def test_footprint_score_and_occupancy(self):
        lanes = _LaneMap()
        for _ in range(int(LANE_MIN_OBS) + 10):
            lanes.add([(0, 410, 1280, 490)], FW, FH)
        self.assertTrue(lanes.ready)
        self.assertGreater(lanes.score((300, 410, 420, 490), FW, FH), 0.9)
        self.assertEqual(lanes.score((300, 100, 420, 180), FW, FH), 0.0)
        self.assertAlmostEqual(lanes.occupancy([(0, 410, 640, 490)], FW, FH), 0.5, delta=0.05)

    def test_lane_maps_persist_across_restarts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "lanes.npz")
            engine = TrafficAnalyticsEngine(db_path=os.path.join(tmp, "ts.db"), lanes_path=path)
            for _ in range(int(LANE_MIN_OBS) + 10):
                engine._lanes.setdefault(5, _LaneMap()).add([(0, 410, 1280, 490)], FW, FH)
            engine.close()
            again = TrafficAnalyticsEngine(db_path=os.path.join(tmp, "ts.db"), lanes_path=path)
            try:
                self.assertTrue(again._lanes[5].ready)
                self.assertGreater(again._lanes[5].score((300, 410, 420, 490), FW, FH), 0.9)
            finally:
                again.close()


class ParkedCarDoesNotLearnItsBayAsLaneTest(LaneScenario):
    def test_car_that_drove_in_and_parked_adds_no_lane_evidence_while_still(self):
        # ``moved`` stays latched after the drive-in; only movement now may heat the lane map.
        tid = self.new_id()
        x = 0.0
        for _ in range(int(2 * FPS)):              # drives in along the kerb strip at ~2 bodies/s
            x += 20.0
            self.frame([self.car(tid, x, KERB_Y)])
        for _ in range(int(8 * FPS)):              # parks; lane evidence stops static_after_s after it last advanced
            self.frame([self.car(tid, x, KERB_Y)])
        lanes = self.engine._lanes[1]
        r1, r2, c1, c2 = _LaneMap.cells((x, KERB_Y[0], x + 120, KERB_Y[1]), FW, FH)
        heat_after_parking = float(lanes.heat[r1:r2, c1:c2].sum())
        obs_after_parking = lanes.obs
        for _ in range(int(30 * FPS)):             # stays parked for half a minute
            self.frame([self.car(tid, x, KERB_Y)])
        self.assertAlmostEqual(float(lanes.heat[r1:r2, c1:c2].sum()), heat_after_parking, places=3)
        self.assertAlmostEqual(lanes.obs, obs_after_parking, places=3)
        self.assertFalse(lanes.ready, "half a minute parked must not teach the camera a lane")

    def test_crawling_stop_and_go_queue_still_teaches_the_lanes(self):
        # A chronic jam (camera 168): cars creep at 0.12 body/s with 1 s stops, so their recent speed
        # (~0.06-0.08 body/s) never reaches stop_bodies_per_s. They still advance, so they are lane
        # evidence; otherwise a jammed camera would never learn its lanes.
        cars = [self.new_id() for _ in range(10)]
        step = 0.12 * 120 / FPS                    # px per frame at 0.12 body sizes/s (120 px bodies)
        off, live = 0.0, None
        for i in range(int(45 * FPS)):
            if (i // int(FPS)) % 2 == 0:           # 1 s creeping, 1 s standing
                off += step
            live = self.frame([self.car(cars[k], 5 + k * 125 + off, LANE_Y) for k in range(10)])
        bps = [t.recent_bps for t in self.engine._camera_tracks[1].values()]
        self.assertTrue(bps and all(b is not None and b < self.engine.stop_bodies_per_s for b in bps), bps)
        self.assertTrue(self.engine._lanes[1].ready, "a crawling queue must still teach the lane map")
        self.assertTrue(live["lanes_learned"])


if __name__ == "__main__":
    unittest.main()
