"""
Tests for Per-Camera Traffic Volume & Time-Series Analytics Engine.

Validates:
- Vehicle kinematics, displacement, speed proxy, and flow vector tracking.
- Per-camera volume aggregation and class breakdowns.
- Spatial occupancy ratio and congestion index calculation.
- Continuous cyclic temporal encodings (sin_hour, cos_hour, sin_dow, cos_dow).
- SQLite persistence in WAL mode.
- LSTM 3D sliding-window tensor formatting (X, y).
"""

import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


from car_tracker.analytics import (
    TimeWindowMetrics,
    TrafficAnalyticsDatabase,
    TrafficAnalyticsEngine,
)


def insert_fixture_windows(engine, camera_id=167, camera_name="WARUNG JAMBU", count=120, window_seconds=60):
    """Writes ``count`` deterministic consecutive windows straight into the engine's DB (test data only)."""
    start = 1_700_000_000.0
    for i in range(count):
        ts = start + i * window_seconds
        hour, minute, dow, wknd, sin_h, cos_h, sin_d, cos_d = engine.compute_temporal_features(ts)
        volume = 10 + (i % 7)
        engine.db.insert_metrics(TimeWindowMetrics(
            camera_id=camera_id, camera_name=camera_name, timestamp=ts,
            timestamp_iso=datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
            window_seconds=window_seconds, total_volume=volume, car_volume=volume // 3,
            motorcycle_volume=volume // 2, truck_volume=1, bus_volume=volume - volume // 3 - volume // 2 - 1,
            avg_active_vehicles=2.0, peak_active_vehicles=4, occupancy_ratio=0.05,
            avg_speed_px_s=30.0, stopped_vehicles=0, congestion_index=0.2,
            avg_flow_dx=1.0, avg_flow_dy=2.0, flow_rate_vpm=float(volume),
            flow_rate_vph=volume * 60.0, hour=hour, minute=minute, day_of_week=dow,
            is_weekend=wknd, sin_hour=sin_h, cos_hour=cos_h, sin_dow=sin_d, cos_dow=cos_d,
        ))


class TestTrafficAnalyticsEngine(unittest.TestCase):
    """Test unit kinematics, window aggregation, and ML features."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.temp_dir.name) / "test_traffic.db")
        self.engine = TrafficAnalyticsEngine(
            db_path=self.db_path,
            primary_window_seconds=10,  # Short window for testing
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_cyclic_temporal_features(self):
        """Verify cyclic sine/cosine encodings cycle smoothly and stay bounded in [-1, 1]."""
        # Midnight UTC: hour=0, min=0 -> sin(0)=0, cos(0)=1
        ts_midnight = 1728518400.0  # An exact midnight epoch
        h, m, dow, is_wknd, sin_h, cos_h, sin_d, cos_d = self.engine.compute_temporal_features(ts_midnight)
        self.assertEqual(h, 0)
        self.assertEqual(m, 0)
        self.assertAlmostEqual(sin_h, 0.0, places=3)
        self.assertAlmostEqual(cos_h, 1.0, places=3)
        self.assertGreaterEqual(sin_h, -1.0)
        self.assertLessEqual(sin_h, 1.0)
        self.assertGreaterEqual(cos_h, -1.0)
        self.assertLessEqual(cos_h, 1.0)
        self.assertIn(is_wknd, [0, 1])

    def test_record_frame_and_kinematics(self):
        """Simulate moving vehicles and verify displacement and speed proxy."""
        cam_id = 167
        cam_name = "TEST_CAMERA"
        t0 = 1000.0

        # Frame 1: Vehicle 1 (car) at (100, 100), Vehicle 2 (motorcycle) at (200, 200)
        dets_f1 = [
            {"track_id": 1, "class_id": 2, "class_name": "car", "box": (80, 80, 120, 120), "conf": 0.9},
            {"track_id": 2, "class_id": 3, "class_name": "motorcycle", "box": (190, 190, 210, 210), "conf": 0.85},
        ]
        counts_f1 = {"car": 1, "motorcycle": 1}
        snap1 = self.engine.record_frame(cam_id, cam_name, t0, counts_f1, dets_f1, (720, 1280))

        self.assertEqual(snap1["active_total"], 2)
        # A brand-new track can't yet be told apart from a parked one, so it isn't in the
        # congestion measures (occupancy) until it has moved.
        self.assertEqual(snap1["occupancy_ratio"], 0.0)
        self.assertEqual(snap1["unique_window_volume"], 2)

        # Frame 2: 1.0 second later, Vehicle 1 moved 30px, Vehicle 2 moved 40px
        t1 = 1001.0
        dets_f2 = [
            {"track_id": 1, "class_id": 2, "class_name": "car", "box": (110, 80, 150, 120), "conf": 0.9},
            {"track_id": 2, "class_id": 3, "class_name": "motorcycle", "box": (230, 190, 250, 210), "conf": 0.85},
        ]
        counts_f2 = {"car": 1, "motorcycle": 1}
        snap2 = self.engine.record_frame(cam_id, cam_name, t1, counts_f2, dets_f2, (720, 1280))

        self.assertGreater(snap2["avg_speed_px_s"], 0.0)
        self.assertGreater(snap2["occupancy_ratio"], 0.0, "both vehicles have now moved: they are traffic")
        self.assertIn(snap2["congestion_label"], ["Smooth", "Moderate", "Heavy", "Congested"])

    def test_window_flush_and_db_persistence(self):
        """Simulate elapsed window duration and verify persistence in SQLite."""
        cam_id = 81
        cam_name = "DJUANDA"
        t = 2000.0

        # Ingest frames across 12 seconds (> primary_window_seconds=10)
        for i in range(12):
            dets = [
                {"track_id": 10 + i, "class_id": 2, "class_name": "car", "box": (50, 50, 100, 100), "conf": 0.9},
                {"track_id": 20 + i, "class_id": 3, "class_name": "motorcycle", "box": (20, 20, 40, 40), "conf": 0.8},
            ]
            self.engine.record_frame(cam_id, cam_name, t + i, {"car": 1, "motorcycle": 1}, dets, (720, 1280))

        # Query time-series from DB
        records = self.engine.export_timeseries_dataset(camera_id=cam_id, window_seconds=10)
        self.assertGreaterEqual(len(records), 1)

        rec = records[0]
        self.assertEqual(rec["camera_id"], cam_id)
        self.assertEqual(rec["camera_name"], cam_name)
        self.assertGreater(rec["total_volume"], 0)
        self.assertGreater(rec["flow_rate_vpm"], 0.0)
        self.assertIn("sin_hour", rec)
        self.assertIn("cos_hour", rec)

    def test_csv_export(self):
        """Verify CSV export contains correct headers and values."""
        insert_fixture_windows(self.engine)
        csv_str = self.engine.export_csv_str(camera_id=167, window_seconds=60)
        lines = csv_str.strip().splitlines()
        self.assertGreater(len(lines), 2)

        headers = lines[0].split(",")
        self.assertIn("timestamp_iso", headers)
        self.assertIn("total_volume", headers)
        self.assertIn("car_volume", headers)
        self.assertIn("motorcycle_volume", headers)
        self.assertIn("flow_rate_vpm", headers)
        self.assertIn("congestion_index", headers)
        self.assertIn("sin_hour", headers)

    def test_lstm_sequence_creation(self):
        """Verify transformation of time-series into 3D LSTM tensor arrays (X, y)."""
        insert_fixture_windows(self.engine)
        records = self.engine.export_timeseries_dataset(camera_id=167, window_seconds=60)
        seq_len = 10
        horizon = 1

        seq_dict = self.engine.create_lstm_sequences(records, seq_len=seq_len, horizon=horizon)
        X = seq_dict["X"]
        y = seq_dict["y"]

        self.assertEqual(len(X.shape), 3)  # (N, seq_len, features)
        self.assertEqual(X.shape[1], seq_len)
        self.assertEqual(y.shape[1], horizon)
        self.assertEqual(X.shape[0], len(records) - seq_len - horizon + 1)
        self.assertGreater(X.shape[2], 5)  # Multiple ML features


class TestCongestionModel(unittest.TestCase):
    """Congestion must reflect the moving stream: parked cars and busy-but-flowing roads are not jams."""

    CAM, SHAPE, FPS = 167, (720, 1280), 5.0

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = TrafficAnalyticsEngine(
            db_path=str(Path(self.temp_dir.name) / "t.db"), primary_window_seconds=60)

    def tearDown(self):
        self.temp_dir.cleanup()

    def run_scene(self, seconds, position_fn, ids, size=(80, 60), start=1000.0):
        """position_fn(track_id, t) -> (cx, cy) or None; returns the last live snapshot."""
        snap = None
        for i in range(int(seconds * self.FPS)):
            t = i / self.FPS
            dets = []
            for tid in ids:
                pos = position_fn(tid, t)
                if pos is None:
                    continue
                cx, cy = pos
                dets.append({"track_id": tid, "class_id": 2, "class_name": "car", "conf": 0.9,
                             "box": (cx - size[0] / 2, cy - size[1] / 2, cx + size[0] / 2, cy + size[1] / 2)})
            snap = self.engine.record_frame(self.CAM, "TEST", start + t, {"car": len(dets)}, dets, self.SHAPE)
        return snap

    def test_parked_cars_are_not_a_queue(self):
        parked = lambda tid, t: (100 + 120 * tid, 500)
        snap = self.run_scene(30, parked, range(1, 7))
        self.assertEqual(snap["congestion_index"], 0.0)
        self.assertEqual(snap["parked_vehicles"], 6)
        self.assertEqual(snap["traffic_vehicles"], 0)
        self.assertEqual(snap["occupancy_ratio"], 0.0)
        self.assertEqual(snap["stopped_vehicles"], 0)
        self.assertEqual(snap["unique_window_volume"], 0, "parked cars are not volume either")

    def test_busy_road_with_flowing_traffic_is_not_congested(self):
        # 12 vehicles crossing the frame at ~2 body sizes per second, new ones entering as others leave
        def flowing(tid, t):
            x = (tid * 110 + t * 160) % 1400 - 60
            return (x, 300 + (tid % 3) * 90)
        snap = self.run_scene(40, flowing, range(1, 13))
        self.assertLess(snap["congestion_index"], 0.15)
        self.assertEqual(snap["parked_vehicles"], 0)

    def test_parked_cars_do_not_hide_flowing_traffic_or_inflate_congestion(self):
        def scene(tid, t):
            if tid >= 100:                       # a row of parked cars at the kerb
                return (100 + 90 * (tid - 100), 600)
            return ((tid * 130 + t * 160) % 1400 - 60, 300)
        busy_only = self.run_scene(40, scene, range(1, 9))
        with_parked = self.run_scene(40, scene, list(range(1, 9)) + list(range(100, 106)), start=5000.0)
        self.assertLess(with_parked["congestion_index"], 0.15)
        self.assertEqual(with_parked["parked_vehicles"], 6)
        self.assertAlmostEqual(with_parked["traffic_vehicles"], busy_only["traffic_vehicles"], delta=2)

    def test_queue_of_vehicles_that_were_moving_is_congestion(self):
        # approach for 4 s, then everyone stops (a jam that formed in view)
        def queue(tid, t):
            return (80 + 130 * tid + min(t, 4.0) * 40, 400)
        snap = self.run_scene(20, queue, range(1, 9))     # no learned free-flow speed yet
        self.assertGreater(snap["congestion_index"], 0.55)
        self.assertIn(snap["congestion_label"], ("Heavy", "Congested"))
        self.assertEqual(snap["stopped_vehicles"], 8)

    def test_queue_is_congestion_with_a_learned_free_flow_speed_too(self):
        self.engine._free_flow_bps[self.CAM] = 1.2
        queue = lambda tid, t: (80 + 130 * tid + min(t, 4.0) * 40, 400)
        snap = self.run_scene(45, queue, range(1, 9))     # the live index averages the last 20 s
        self.assertGreater(snap["congestion_index"], 0.8)
        self.assertEqual(snap["congestion_label"], "Congested")

    def test_two_stopped_vehicles_are_not_a_jam(self):
        def queue(tid, t):
            return (80 + 200 * tid + min(t, 4.0) * 40, 400)
        snap = self.run_scene(20, queue, range(1, 3))
        self.assertLess(snap["congestion_index"], 0.25)
        self.assertEqual(snap["congestion_label"], "Smooth")

    def test_slowness_is_not_judged_before_the_free_flow_speed_is_learned(self):
        slow = lambda tid, t: ((tid * 110 + t * 24) % 1400 - 60, 300)      # everyone at ~0.3 body sizes/s
        snap = self.run_scene(30, slow, range(1, 13))
        self.assertIsNone(snap["free_flow_bodies_s"])
        self.assertEqual(snap["stopped_vehicles"], 0)
        self.assertLess(snap["congestion_index"], 0.05, "no reference yet: only a standstill can be judged")

    def test_crawl_far_below_the_cameras_free_flow_is_congestion(self):
        self.engine._free_flow_bps[self.CAM] = 1.2
        crawl = lambda tid, t: ((tid * 110 + t * 24) % 1400 - 60, 300)     # 0.3 body sizes/s = 25% of free flow
        snap = self.run_scene(30, crawl, range(1, 13))
        self.assertGreater(snap["congestion_index"], 0.25)
        self.assertLess(snap["congestion_index"], 0.55, "slow but moving: Moderate, not a standstill")
        self.assertEqual(snap["congestion_label"], "Moderate")

    def test_a_slow_street_is_not_congested_at_its_own_normal_speed(self):
        self.engine._free_flow_bps[self.CAM] = 0.5                          # this camera's traffic is normally slow
        normal = lambda tid, t: ((tid * 110 + t * 24) % 1400 - 60, 300)    # 0.3 body sizes/s = 60% of ITS free flow
        snap = self.run_scene(30, normal, range(1, 13))
        self.assertLess(snap["congestion_index"], 0.05)

    def test_standstill_among_moving_traffic_is_heavy(self):
        self.engine._free_flow_bps[self.CAM] = 1.2
        def scene(tid, t):
            if tid <= 5:                                                   # queue that formed in view
                return (80 + 130 * tid + min(t, 4.0) * 40, 400)
            return ((tid * 130 + t * 160) % 1400 - 60, 250)                # cross traffic still flowing
        snap = self.run_scene(25, scene, range(1, 11))
        self.assertGreater(snap["congestion_index"], 0.25)

    def test_free_flow_speed_is_persisted_and_reloaded(self):
        path = str(Path(self.temp_dir.name) / "ff.json")
        engine = TrafficAnalyticsEngine(db_path=str(Path(self.temp_dir.name) / "p.db"),
                                        primary_window_seconds=60, free_flow_path=path)
        engine._free_flow_bps[self.CAM] = 0.9
        engine._save_free_flow(min_interval_s=0)
        again = TrafficAnalyticsEngine(db_path=str(Path(self.temp_dir.name) / "p.db"), free_flow_path=path)
        self.assertAlmostEqual(again._free_flow_bps[self.CAM], 0.9, places=3)

    def test_window_flush_learns_the_cameras_free_flow_speed(self):
        flowing = lambda tid, t: ((tid * 110 + t * 160) % 1400 - 60, 300)
        self.run_scene(70, flowing, range(1, 11))
        self.assertIn(self.CAM, self.engine._free_flow_bps)
        self.assertGreater(self.engine._free_flow_bps[self.CAM], 0.5)
        rows = self.engine.export_timeseries_dataset(camera_id=self.CAM, window_seconds=60)
        self.assertTrue(rows)
        self.assertLess(rows[-1]["congestion_index"], 0.15)


class TestCrossingVolume(unittest.TestCase):
    """Window volume comes from counting-line crossings when the caller reports them."""

    CAM = 81

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.engine = TrafficAnalyticsEngine(db_path=str(Path(self.temp_dir.name) / "c.db"),
                                             primary_window_seconds=60)

    def tearDown(self):
        self.temp_dir.cleanup()

    def feed(self, crossing_plan, with_flag=True, frames=12):
        """12 one-second frames, a fresh track id every frame (fragmented tracks); crossing_plan
        maps frame index -> class that crossed in that frame."""
        for i in range(frames):
            det = {"track_id": 100 + i, "class_id": 2, "class_name": "car", "box": (50 + 5 * i, 50, 110 + 5 * i, 110), "conf": 0.9}
            if with_flag:
                det["counted_now"] = i in crossing_plan
                if i in crossing_plan:
                    det["class_name"] = crossing_plan[i]
            self.engine.record_frame(self.CAM, "TEST", 3000.0 + i, {"car": 1}, [det], (720, 1280))
        return self.engine.flush_camera(self.CAM)

    def rows(self):
        return self.engine.db.query_timeseries(camera_id=self.CAM, window_seconds=11)

    def test_volume_is_crossings_not_track_ids(self):
        metrics = self.feed({2: "car", 5: "motorcycle", 9: "car"})
        self.assertEqual(metrics.volume_basis, "crossings")
        self.assertEqual(metrics.total_volume, 3, "12 distinct track ids appeared but only 3 crossed the line")
        self.assertEqual((metrics.car_volume, metrics.motorcycle_volume, metrics.truck_volume, metrics.bus_volume), (2, 1, 0, 0))
        self.assertAlmostEqual(metrics.flow_rate_vpm, round(3 / 11 * 60, 2), places=2)
        row = self.rows()[0]
        self.assertEqual((row["total_volume"], row["volume_basis"]), (3, "crossings"))

    def test_no_crossings_means_zero_volume(self):
        self.assertEqual(self.feed({}).total_volume, 0)

    def test_callers_without_crossing_info_keep_track_based_volume(self):
        metrics = self.feed({}, with_flag=False)
        self.assertEqual(metrics.volume_basis, "tracks")
        self.assertEqual(metrics.total_volume, 12)

    def test_live_window_reports_its_basis(self):
        self.feed({2: "car"}, frames=5)
        # window is still open after 5 frames: check before flush by re-feeding without flushing
        engine = TrafficAnalyticsEngine(db_path=str(Path(self.temp_dir.name) / "live.db"))
        engine.record_frame(self.CAM, "T", 10.0, {"car": 1}, [{"track_id": 1, "class_id": 2, "class_name": "car",
                            "box": (0, 0, 50, 50), "conf": 0.9, "counted_now": True}], (720, 1280))
        win = engine.get_live_metrics(self.CAM)["current_window"]
        self.assertEqual((win["volume_basis"], win["total_volume"], win["class_volumes"]), ("crossings", 1, {"car": 1}))

    def test_csv_export_carries_the_volume_basis(self):
        self.feed({2: "car"})
        csv_text = self.engine.export_csv_str(camera_id=self.CAM, window_seconds=11)
        self.assertIn("volume_basis", csv_text.splitlines()[0])
        self.assertIn("crossings", csv_text)

    def test_existing_database_is_migrated_and_old_rows_are_marked_as_track_based(self):
        path = str(Path(self.temp_dir.name) / "old.db")
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE camera_traffic_timeseries (id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL, camera_name TEXT NOT NULL, timestamp REAL NOT NULL, timestamp_iso TEXT NOT NULL,
            window_seconds INTEGER NOT NULL, total_volume INTEGER NOT NULL, car_volume INTEGER NOT NULL,
            motorcycle_volume INTEGER NOT NULL, truck_volume INTEGER NOT NULL, bus_volume INTEGER NOT NULL,
            avg_active_vehicles REAL NOT NULL, peak_active_vehicles INTEGER NOT NULL, occupancy_ratio REAL NOT NULL,
            avg_speed_px_s REAL NOT NULL, stopped_vehicles INTEGER NOT NULL, congestion_index REAL NOT NULL,
            avg_flow_dx REAL NOT NULL, avg_flow_dy REAL NOT NULL, flow_rate_vpm REAL NOT NULL, flow_rate_vph REAL NOT NULL,
            hour INTEGER NOT NULL, minute INTEGER NOT NULL, day_of_week INTEGER NOT NULL, is_weekend INTEGER NOT NULL,
            sin_hour REAL NOT NULL, cos_hour REAL NOT NULL, sin_dow REAL NOT NULL, cos_dow REAL NOT NULL)""")
        conn.execute("INSERT INTO camera_traffic_timeseries VALUES (NULL,5,'OLD',1.0,'x',60,99,1,1,1,1,1.0,1,0.1,1.0,0,0.1,0,0,99,99,0,0,0,0,0,1,0,1)")
        conn.commit(); conn.close()
        db = TrafficAnalyticsEngine(db_path=path).db
        row = db.query_timeseries(camera_id=5)[0]
        self.assertEqual((row["total_volume"], row["volume_basis"]), (99, "tracks"))

    def test_congestion_series_buckets_weight_windows_by_length(self):
        for ts, win, cong in ((1000.0, 10, 0.2), (1100.0, 30, 0.6), (1400.0, 60, 0.1)):
            self.engine.db.insert_metrics(TimeWindowMetrics(
                camera_id=7, camera_name="C", timestamp=ts, timestamp_iso="x", window_seconds=win,
                total_volume=1, car_volume=1, motorcycle_volume=0, truck_volume=0, bus_volume=0,
                avg_active_vehicles=1.0, peak_active_vehicles=1, occupancy_ratio=0.1, avg_speed_px_s=1.0,
                stopped_vehicles=0, congestion_index=cong, avg_flow_dx=0.0, avg_flow_dy=0.0,
                flow_rate_vpm=1.0, flow_rate_vph=60.0, hour=0, minute=0, day_of_week=0, is_weekend=0,
                sin_hour=0.0, cos_hour=1.0, sin_dow=0.0, cos_dow=1.0, volume_basis="crossings"))
        series = self.engine.get_congestion_series(7, minutes=60, bucket_minutes=5, now=1500.0)
        self.assertEqual([p["t"] for p in series], [900.0, 1200.0])
        self.assertAlmostEqual(series[0]["congestion"], round((0.2 * 10 + 0.6 * 30) / 40, 3), places=3)
        self.assertAlmostEqual(series[1]["congestion"], 0.1, places=3)


if __name__ == "__main__":
    unittest.main()
