"""
Tests for Per-Camera Traffic Volume & Time-Series Analytics Engine.

Validates:
- Vehicle kinematics, displacement, speed proxy, and flow vector tracking.
- Per-camera volume aggregation and class breakdowns.
- Spatial occupancy ratio and congestion index calculation.
- Continuous cyclic temporal encodings (sin_hour, cos_hour, sin_dow, cos_dow), in local WIB time.
- SQLite persistence in WAL mode on one persistent connection; no I/O under the engine lock.
- Windows of any length are queryable (no window_seconds equality filter); bucketed queries are
  weighted by window length.
- Volume basis follows whether the caller reports counting-line crossings.
- LSTM 3D sliding-window tensor formatting (X, y).
"""

import math
import os
import random
import sqlite3
import statistics
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock


from car_tracker.analytics import (
    LOCAL_TZ,
    TimeWindowMetrics,
    TrafficAnalyticsDatabase,
    TrafficAnalyticsEngine,
    _LiveMean,
)

WIB = timezone(timedelta(hours=7))


def wib(text):
    """Epoch seconds of a WIB wall-clock time, computed without the code under test."""
    return datetime.fromisoformat(text).replace(tzinfo=WIB).timestamp()


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
        # Local (WIB) midnight: hour=0, min=0 -> sin(0)=0, cos(0)=1
        ts_midnight = wib("2024-10-10T00:00:00")
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

    def test_temporal_features_are_local_wib_time_not_utc(self):
        # 1728518400 is 2024-10-10 00:00 UTC = 07:00 WIB, a Thursday
        self.assertEqual(self.engine.compute_temporal_features(1728518400.0)[:4], (7, 0, 3, 0))
        # Friday 18 Oct 2024 20:30 UTC is already Saturday 03:30 WIB: weekend, and the day rolled over
        h, m, dow, wknd, *_ = self.engine.compute_temporal_features(wib("2024-10-19T03:30:00"))
        self.assertEqual((h, m, dow, wknd), (3, 30, 5, 1))
        self.assertEqual(datetime.fromtimestamp(wib("2024-10-19T03:30:00"), timezone.utc).weekday(), 4)
        # 06:00 local = a quarter of the day
        _, _, _, _, sin_h, cos_h, _, _ = self.engine.compute_temporal_features(wib("2024-10-19T06:00:00"))
        self.assertAlmostEqual(sin_h, 1.0, places=4)
        self.assertAlmostEqual(cos_h, 0.0, places=4)
        self.assertEqual(LOCAL_TZ.utcoffset(datetime(2024, 10, 10)), timedelta(hours=7))

    def test_flushed_windows_store_local_features_and_both_iso_forms(self):
        t0 = wib("2024-10-19T03:30:00")          # Saturday 03:30 WIB = Friday 20:30 UTC
        for i in range(12):
            self.engine.record_frame(7, "C", t0 + i, {"car": 1}, [
                {"track_id": 1, "class_id": 2, "class_name": "car", "box": (50 + 5 * i, 50, 110 + 5 * i, 110), "conf": 0.9}],
                (720, 1280))
        self.engine.flush_camera(7)
        (row,) = self.engine.db.query_timeseries(camera_id=7)
        self.assertEqual((row["hour"], row["minute"], row["day_of_week"], row["is_weekend"]), (3, 30, 5, 1))
        end = row["timestamp"]
        self.assertEqual(row["timestamp_iso"], datetime.fromtimestamp(end, timezone.utc).isoformat())
        self.assertEqual(row["timestamp_iso"][-6:], "+00:00", "the sortable UTC column keeps its meaning")
        self.assertEqual(row["timestamp_local_iso"], datetime.fromtimestamp(end, WIB).isoformat())
        self.assertEqual(row["timestamp_local_iso"][-6:], "+07:00")

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
        return self.engine.db.query_timeseries(camera_id=self.CAM)

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

    def test_live_snapshot_expires_when_the_camera_is_no_longer_observed(self):
        # A rolling camera's last reading must not be served as live once its visit is over.
        engine = TrafficAnalyticsEngine(db_path=str(Path(self.temp_dir.name) / "stale.db"))
        engine.record_frame(self.CAM, "T", 1000.0, {"car": 1}, [{"track_id": 1, "class_id": 2, "class_name": "car",
                            "box": (0, 0, 50, 50), "conf": 0.9}], (720, 1280))
        fresh = engine.get_live_metrics(self.CAM, now=1005.0)
        self.assertEqual(fresh["live"]["timestamp"], 1000.0)
        self.assertEqual(fresh["live_age_s"], 5.0)

        stale = engine.get_live_metrics(self.CAM, now=1000.0 + 600)
        self.assertEqual(stale["live"], {})
        self.assertEqual(stale["last_measured"]["timestamp"], 1000.0)
        self.assertEqual(stale["live_age_s"], 600.0)

        never = engine.get_live_metrics(4242, now=1000.0)
        self.assertEqual((never["live"], never["last_measured"], never["live_age_s"]), ({}, {}, None))
        self.assertNotIn(4242, engine._history_60s)   # an unknown camera leaves no state behind

    def test_csv_export_carries_the_volume_basis(self):
        self.feed({2: "car"})
        csv_text = self.engine.export_csv_str(camera_id=self.CAM)
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
        self.assertEqual(self.engine.get_first_timestamp(7), 1000.0)
        self.assertIsNone(self.engine.get_first_timestamp(8))


def window(camera_id, end, seconds, volume=0, congestion=0.1, basis="crossings", **kw):
    return TimeWindowMetrics(
        camera_id=camera_id, camera_name=f"CAM {camera_id}", timestamp=end,
        timestamp_iso=datetime.fromtimestamp(end, timezone.utc).isoformat(), window_seconds=seconds,
        total_volume=volume, car_volume=volume, motorcycle_volume=0, truck_volume=0, bus_volume=0,
        avg_active_vehicles=kw.get("active", 2.0), peak_active_vehicles=kw.get("peak", 3),
        occupancy_ratio=kw.get("occupancy", 0.1), avg_speed_px_s=kw.get("speed", 30.0), stopped_vehicles=0,
        congestion_index=congestion, avg_flow_dx=0.0, avg_flow_dy=0.0, flow_rate_vpm=0.0, flow_rate_vph=0.0,
        hour=0, minute=0, day_of_week=0, is_weekend=0, sin_hour=0.0, cos_hour=1.0, sin_dow=0.0, cos_dow=1.0,
        volume_basis=basis, median_speed_bps=kw.get("median"), queue_share=kw.get("queue"))


class TestPersistentConnection(unittest.TestCase):
    """One writer connection for the life of the engine instead of one per insert."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = TrafficAnalyticsDatabase(str(Path(self.tmp.name) / "p.db"))
        self.addCleanup(self.db.close)

    def test_inserts_do_not_open_connections(self):
        real = sqlite3.connect
        with mock.patch("car_tracker.analytics.sqlite3.connect", side_effect=real) as connect:
            for i in range(50):
                self.db.insert_metrics(window(1, 1000.0 + i, 10, volume=1))
            self.assertEqual(connect.call_count, 0)
            self.db.query_timeseries(camera_id=1)
            self.db.query_timeseries(camera_id=1)
            self.assertEqual(connect.call_count, 1, "one read connection, opened on first use and then reused")
        writer = self.db._wconn
        self.db.insert_metrics(window(1, 2000.0, 10))
        self.assertIs(self.db._wconn, writer)

    def test_wal_mode_and_busy_timeout(self):
        conn = self.db._writer()
        self.assertEqual(conn.execute("PRAGMA journal_mode").fetchone()[0].lower(), "wal")
        self.assertGreaterEqual(conn.execute("PRAGMA busy_timeout").fetchone()[0], 5000)

    def test_close_is_idempotent_and_the_database_reconnects_lazily(self):
        self.db.insert_metrics(window(1, 1000.0, 10, volume=2))
        self.db.close()
        self.db.close()
        self.assertIsNone(self.db._wconn)
        self.db.insert_metrics(window(1, 1010.0, 10, volume=3))
        self.assertEqual([r["total_volume"] for r in self.db.query_timeseries(camera_id=1)], [2, 3])

    def test_a_failed_insert_leaves_nothing_open_and_the_next_insert_works(self):
        bad = window(1, 1000.0, 10)
        bad.camera_name = None                       # NOT NULL violation
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.insert_metrics(bad)
        self.db.insert_metrics(window(1, 1001.0, 10, volume=4))
        self.assertEqual(len(self.db.query_timeseries()), 1)
        other = sqlite3.connect(self.db.db_path, timeout=1)       # no write lock is left behind
        other.execute("INSERT INTO analytics_meta VALUES ('probe', '1')")
        other.commit()
        other.close()

    def test_threads_can_write_and_read_concurrently(self):
        errors = []

        def writer(base):
            try:
                for i in range(60):
                    self.db.insert_metrics(window(base, 5000.0 + i, 10, volume=1))
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        def reader():
            try:
                for _ in range(60):
                    self.db.query_timeseries(limit=50)
                    self.db.congestion_series(1, 0.0, 60.0)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(c,)) for c in (1, 2, 3)] + [threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(20)
        self.assertEqual(errors, [])
        self.assertEqual(len(self.db.query_timeseries(limit=1000)), 180)
        self.assertEqual({r["camera_id"]: r["cumulative_volume"] for r in self.db.get_cumulative_summary()},
                         {1: 60, 2: 60, 3: 60})


class TestLockDiscipline(unittest.TestCase):
    """The engine lock guards in-memory state only: no database or file I/O while it is held."""

    CAM = 5

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ff = str(Path(self.tmp.name) / "ff.json")
        self.engine = TrafficAnalyticsEngine(db_path=str(Path(self.tmp.name) / "l.db"), primary_window_seconds=5,
                                             free_flow_path=self.ff)
        self.addCleanup(self.engine.close)

    def frames(self, seconds, start=1000.0, fps=5):
        for i in range(int(seconds * fps)):
            t = i / fps
            dets = [{"track_id": tid, "class_id": 2, "class_name": "car", "conf": 0.9,
                     "box": ((tid * 110 + t * 160) % 1400 - 60, 300, (tid * 110 + t * 160) % 1400 + 20, 360),
                     "counted_now": False} for tid in range(1, 11)]
            self.engine.record_frame(self.CAM, "C", start + t, {"car": len(dets)}, dets, (720, 1280))

    def test_database_insert_and_free_flow_file_are_written_outside_the_lock(self):
        held = []
        real_insert = self.engine.db.insert_metrics
        self.engine.db.insert_metrics = lambda m: (held.append(("insert", self.engine._lock.locked())), real_insert(m))[1]
        real_replace = os.replace
        with mock.patch("car_tracker.analytics.os.replace",
                        side_effect=lambda a, b: (held.append(("free_flow", self.engine._lock.locked())), real_replace(a, b))[1]):
            self.frames(12)
            self.engine.flush_camera(self.CAM)
        kinds = {k for k, _ in held}
        self.assertEqual(kinds, {"insert", "free_flow"}, held)
        self.assertFalse([h for h in held if h[1]], f"I/O happened under the engine lock: {held}")
        self.assertTrue(os.path.exists(self.ff))

    def test_a_slow_database_does_not_block_other_callers(self):
        entered, release = threading.Event(), threading.Event()
        real_insert = self.engine.db.insert_metrics

        def slow(m):
            entered.set()
            release.wait(10)
            return real_insert(m)

        self.engine.db.insert_metrics = slow
        worker = threading.Thread(target=self.frames, args=(8,))
        worker.start()
        try:
            self.assertTrue(entered.wait(10), "a window should have closed and reached the database")
            done = []
            probe = threading.Thread(target=lambda: done.append(self.engine.get_live_metrics(self.CAM)))
            probe.start()
            probe.join(2)
            self.assertEqual(len(done), 1, "get_live_metrics blocked on the database write (engine lock held)")
        finally:
            release.set()
            worker.join(20)

    def test_dead_code_is_gone(self):
        self.assertFalse(hasattr(self.engine, "_history_10s"))
        self.assertFalse(hasattr(self.engine, "_cong_recent"))


class TestLiveMean(unittest.TestCase):
    def test_matches_a_brute_force_windowed_mean(self):
        rng, t, kept = random.Random(7), 0.0, []
        mean = _LiveMean(20.0)
        for _ in range(12000):
            t += rng.uniform(0.0, 0.5)
            v = rng.random()
            got = mean.add(t, v)
            kept = [(a, b) for a, b in kept if t - a <= 20.0] + [(t, v)]
            self.assertAlmostEqual(got, statistics.fmean(b for _, b in kept), places=9)

    def test_window_resets_after_a_long_pause(self):
        mean = _LiveMean(20.0)
        for i in range(10):
            mean.add(float(i), 1.0)
        self.assertEqual(mean.add(1000.0, 0.25), 0.25)

    def test_engine_live_index_is_the_mean_over_the_last_20_seconds(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = TrafficAnalyticsEngine(db_path=os.path.join(tmp.name, "m.db"))
        self.addCleanup(engine.close)
        values = []
        with mock.patch.object(TrafficAnalyticsEngine, "_congestion_index", side_effect=lambda **kw: values[-1]):
            seen = []
            for i in range(60):
                values.append(0.0 if i < 30 else 1.0)
                snap = engine.record_frame(1, "C", 100.0 + i, {}, [], (720, 1280))
                seen.append(snap["congestion_index"])
        self.assertEqual(seen[29], 0.0)
        self.assertAlmostEqual(seen[39], round(10 / 21, 3), places=3)         # t=130..139 of the 20 s span 119..139
        self.assertEqual(seen[59], 1.0)


class TestWindowQueries(unittest.TestCase):
    """Rolling visits write windows of 2-62 s; queries must not assume 60."""

    CAM = 9

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = TrafficAnalyticsEngine(db_path=str(Path(self.tmp.name) / "w.db"))
        self.addCleanup(self.engine.close)
        self.t0 = wib("2026-10-11T08:00:00")

    def put(self, end_offset, seconds, volume, congestion, **kw):
        self.engine.db.insert_metrics(window(self.CAM, self.t0 + end_offset, seconds, volume, congestion, **kw))

    def test_raw_query_returns_windows_of_every_length(self):
        for i, secs in enumerate((2, 12, 37, 60, 62)):
            self.put(100 * (i + 1), secs, 1, 0.1)
        rows = self.engine.db.query_timeseries(camera_id=self.CAM)
        self.assertEqual([r["window_seconds"] for r in rows], [2, 12, 37, 60, 62])
        self.assertEqual(len(self.engine.export_timeseries_dataset(camera_id=self.CAM)), 5)
        self.assertEqual(len(self.engine.export_csv_str(camera_id=self.CAM).strip().splitlines()), 6)
        newest = self.engine.db.query_timeseries(camera_id=self.CAM, limit=2, order_asc=False)
        self.assertEqual([r["window_seconds"] for r in newest], [62, 60])

    def test_bucketed_query_weights_windows_by_length(self):
        self.put(10, 10, 2, 0.2, speed=20.0)
        self.put(40, 30, 4, 0.6, speed=40.0)
        self.put(100, 20, 1, 0.1, speed=30.0)
        rows = self.engine.export_timeseries_dataset(camera_id=self.CAM, window_seconds=60)
        self.assertEqual(len(rows), 2)
        a, b = rows
        self.assertEqual((a["timestamp"], a["window_seconds"], a["observed_seconds"], a["n_windows"]), (self.t0 + 60, 60, 40.0, 2))
        self.assertEqual(a["total_volume"], 6)
        self.assertAlmostEqual(a["congestion_index"], (0.2 * 10 + 0.6 * 30) / 40, places=3)
        self.assertAlmostEqual(a["avg_speed_px_s"], (20 * 10 + 40 * 30) / 40, places=2)
        self.assertAlmostEqual(a["flow_rate_vpm"], 6 / 40 * 60, places=2)        # per OBSERVED second, not per bucket second
        self.assertAlmostEqual(a["flow_rate_vph"], 6 / 40 * 3600, places=1)
        self.assertEqual((a["volume_basis"], a["hour"], a["minute"]), ("crossings", 8, 0))     # local time of the bucket start
        self.assertEqual(a["timestamp_local_iso"], datetime.fromtimestamp(self.t0 + 60, WIB).isoformat())
        self.assertEqual((b["timestamp"], b["total_volume"], b["observed_seconds"]), (self.t0 + 120, 1, 20.0))

    def test_bucketed_query_marks_mixed_volume_bases_and_ignores_zero_speed(self):
        self.put(10, 10, 2, 0.2, speed=0.0, occupancy=0.0)
        self.put(20, 10, 3, 0.2, speed=50.0, basis="tracks")
        (row,) = self.engine.export_timeseries_dataset(camera_id=self.CAM, window_seconds=60)
        self.assertEqual(row["volume_basis"], "mixed")
        self.assertAlmostEqual(row["avg_speed_px_s"], 50.0, places=2, msg="a window without moving traffic has no speed")

    def test_bucket_edges_are_local_midnight_aligned(self):
        self.put(30, 10, 1, 0.1)
        (row,) = self.engine.export_timeseries_dataset(camera_id=self.CAM, window_seconds=3600)
        self.assertEqual(row["timestamp"], wib("2026-10-11T09:00:00"))
        (row,) = self.engine.export_timeseries_dataset(camera_id=self.CAM, window_seconds=86400)
        self.assertEqual(row["timestamp"], wib("2026-10-12T00:00:00"))
        with self.assertRaises(ValueError):
            self.engine.db.query_timeseries(window_seconds=0)

    def test_time_filters_and_camera_filter(self):
        self.put(10, 10, 1, 0.1)
        self.put(500, 10, 1, 0.1)
        self.engine.db.insert_metrics(window(77, self.t0 + 20, 10, 1))
        self.assertEqual(len(self.engine.db.query_timeseries(start_ts=self.t0 + 100)), 1)
        self.assertEqual(len(self.engine.db.query_timeseries(end_ts=self.t0 + 100)), 2)
        self.assertEqual(len(self.engine.db.query_timeseries()), 3)

    def test_camera_summary_uses_windows_of_any_length(self):
        for i in range(10):
            self.put(20 * (i + 1), 12, 3, 0.4, speed=45.0)
        summary = self.engine.get_camera_summary(self.CAM)
        self.assertAlmostEqual(summary["recent_avg_flow_rate_vpm"], 30 / 120 * 60, places=2)
        self.assertAlmostEqual(summary["recent_avg_speed_px_s"], 45.0, places=2)
        self.assertAlmostEqual(summary["recent_avg_congestion_index"], 0.4, places=3)
        self.assertEqual(summary["recent_congestion_label"], "Moderate")
        self.assertEqual(self.engine.get_camera_summary(404)["recent_avg_flow_rate_vpm"], 0.0)


class TestVolumeBasisFollowsTheCaller(unittest.TestCase):
    """An empty window carries the basis of the caller that reports crossings, not of its contents."""

    SHAPE = (720, 1280)

    def engine_at(self, name, **kw):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        engine = TrafficAnalyticsEngine(db_path=str(Path(tmp.name) / f"{name}.db"), **kw)
        self.addCleanup(engine.close)
        return engine

    @staticmethod
    def det(tid, counted=None):
        d = {"track_id": tid, "class_id": 2, "class_name": "car", "box": (50 + 3 * tid, 50, 110 + 3 * tid, 110), "conf": 0.9}
        if counted is not None:
            d["counted_now"] = counted
        return d

    def visit(self, engine, cam, dets_fn, start=1000.0, frames=12):
        for i in range(frames):
            dets = dets_fn(i)
            engine.record_frame(cam, "C", start + i, {"car": len(dets)}, dets, self.SHAPE)
        return engine.flush_camera(cam)

    def test_an_empty_window_on_a_crossings_caller_is_a_crossings_window(self):
        engine = self.engine_at("a")
        self.visit(engine, 1, lambda i: [self.det(100 + i, counted=False)])          # the caller shows it reports crossings
        empty = self.visit(engine, 2, lambda i: [], start=2000.0)
        self.assertEqual((empty.volume_basis, empty.total_volume), ("crossings", 0))
        (row,) = engine.db.query_timeseries(camera_id=2)
        self.assertEqual(row["volume_basis"], "crossings")

    def test_without_any_crossing_report_an_empty_window_stays_track_based(self):
        engine = self.engine_at("b")
        empty = self.visit(engine, 2, lambda i: [])
        self.assertEqual((empty.volume_basis, empty.total_volume), ("tracks", 0))

    def test_explicit_assume_crossings_overrides_inference(self):
        yes = self.engine_at("c", assume_crossings=True)
        self.assertEqual(self.visit(yes, 2, lambda i: []).volume_basis, "crossings")
        no = self.engine_at("d", assume_crossings=False)
        self.visit(no, 1, lambda i: [self.det(100 + i, counted=False)])
        self.assertEqual(self.visit(no, 2, lambda i: [], start=2000.0).volume_basis, "tracks")

    def test_detections_without_counted_now_stay_track_based_even_if_others_report_crossings(self):
        engine = self.engine_at("e")
        self.visit(engine, 1, lambda i: [self.det(100 + i, counted=True)])
        moving = lambda i: [{"track_id": 1, "class_id": 2, "class_name": "car", "box": (50 + 25 * i, 50, 110 + 25 * i, 110), "conf": 0.9}]
        legacy = self.visit(engine, 3, moving, start=3000.0)
        self.assertEqual((legacy.volume_basis, legacy.total_volume), ("tracks", 1))

    def test_live_snapshot_and_live_metrics_report_the_same_basis(self):
        engine = self.engine_at("f")
        engine.record_frame(1, "C", 10.0, {"car": 1}, [self.det(1, counted=False)], self.SHAPE)
        snap = engine.record_frame(2, "C", 10.0, {}, [], self.SHAPE)
        self.assertEqual((snap["volume_basis"], snap["window_volume"]), ("crossings", 0))
        self.assertEqual(engine.get_live_metrics(2)["current_window"]["volume_basis"], "crossings")

    def test_primary_window_flush_uses_the_same_rule(self):
        engine = self.engine_at("g", primary_window_seconds=5)
        engine.record_frame(1, "C", 10.0, {"car": 1}, [self.det(1, counted=False)], self.SHAPE)
        for i in range(8):
            engine.record_frame(2, "C", 100.0 + i, {}, [], self.SHAPE)
        rows = engine.db.query_timeseries(camera_id=2)
        self.assertTrue(rows)
        self.assertEqual({r["volume_basis"] for r in rows}, {"crossings"})


class TestSpeedColumns(unittest.TestCase):
    CAM, SHAPE = 3, (720, 1280)

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = TrafficAnalyticsEngine(db_path=str(Path(self.tmp.name) / "s.db"), primary_window_seconds=600)
        self.addCleanup(self.engine.close)

    def scene(self, seconds, pos, ids, fps=5.0, start=1000.0):
        for i in range(int(seconds * fps)):
            t = i / fps
            dets = [{"track_id": tid, "class_id": 2, "class_name": "car", "conf": 0.9,
                     "box": (pos(tid, t)[0] - 40, pos(tid, t)[1] - 30, pos(tid, t)[0] + 40, pos(tid, t)[1] + 30)} for tid in ids]
            self.engine.record_frame(self.CAM, "T", start + t, {"car": len(dets)}, dets, self.SHAPE)
        return self.engine.flush_camera(self.CAM)

    def test_flowing_traffic_has_a_body_size_speed_and_no_queue(self):
        m = self.scene(30, lambda tid, t: ((tid * 110 + t * 160) % 1400 - 60, 300), range(1, 11))
        self.assertGreater(m.median_speed_bps, 1.5)
        self.assertLess(m.median_speed_bps, 2.5)
        self.assertGreater(m.mean_speed_bps, 1.0)
        self.assertEqual(m.queue_share, 0.0)
        self.assertGreater(m.avg_traffic_vehicles, 5)
        (row,) = self.engine.db.query_timeseries(camera_id=self.CAM)
        self.assertEqual(row["median_speed_bps"], m.median_speed_bps)
        self.assertEqual(row["queue_share"], 0.0)

    def test_a_standing_queue_is_all_queue_share(self):
        m = self.scene(30, lambda tid, t: (80 + 130 * tid + min(t, 4.0) * 40, 400), range(1, 9))
        self.assertGreater(m.queue_share, 0.7)
        self.assertLess(m.median_speed_bps, 0.1)

    def test_no_moving_vehicles_means_speed_is_null_not_zero(self):
        m = self.scene(15, lambda tid, t: (100 + 120 * tid, 500), range(1, 5))        # parked
        self.assertIsNone(m.median_speed_bps)
        self.assertIsNone(m.mean_speed_bps)
        self.assertIsNone(m.queue_share)
        self.assertEqual(m.avg_traffic_vehicles, 0.0)
        (row,) = self.engine.db.query_timeseries(camera_id=self.CAM)
        self.assertIsNone(row["median_speed_bps"])


class TestMigrations(unittest.TestCase):
    """Rows written by earlier versions are repaired once when the database is opened."""

    LEGACY = """CREATE TABLE camera_traffic_timeseries (id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id INTEGER NOT NULL, camera_name TEXT NOT NULL, timestamp REAL NOT NULL, timestamp_iso TEXT NOT NULL,
        window_seconds INTEGER NOT NULL, total_volume INTEGER NOT NULL, car_volume INTEGER NOT NULL,
        motorcycle_volume INTEGER NOT NULL, truck_volume INTEGER NOT NULL, bus_volume INTEGER NOT NULL,
        avg_active_vehicles REAL NOT NULL, peak_active_vehicles INTEGER NOT NULL, occupancy_ratio REAL NOT NULL,
        avg_speed_px_s REAL NOT NULL, stopped_vehicles INTEGER NOT NULL, congestion_index REAL NOT NULL,
        avg_flow_dx REAL NOT NULL, avg_flow_dy REAL NOT NULL, flow_rate_vpm REAL NOT NULL, flow_rate_vph REAL NOT NULL,
        hour INTEGER NOT NULL, minute INTEGER NOT NULL, day_of_week INTEGER NOT NULL, is_weekend INTEGER NOT NULL,
        sin_hour REAL NOT NULL, cos_hour REAL NOT NULL, sin_dow REAL NOT NULL, cos_dow REAL NOT NULL,
        volume_basis TEXT NOT NULL DEFAULT 'tracks')"""

    def make_legacy(self, path, rows):
        conn = sqlite3.connect(path)
        conn.execute(self.LEGACY)
        for ts, volume, basis in rows:
            utc = datetime.fromtimestamp(ts, timezone.utc)
            conn.execute("INSERT INTO camera_traffic_timeseries VALUES (NULL,5,'OLD',?,?,12,?,0,0,0,0,1.0,1,0.1,1.0,0,0.1,"
                         "0,0,0,0,?,?,?,?,0,1,0,1,?)",
                         (ts, utc.isoformat(), volume, utc.hour, utc.minute, utc.weekday(), 1 if utc.weekday() >= 5 else 0, basis))
        conn.commit()
        conn.close()

    def test_time_features_are_rewritten_in_local_time_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.db")
            ts = wib("2026-10-12T03:30:00")                    # Monday 03:30 WIB = Sunday 20:30 UTC
            self.make_legacy(path, [(ts, 4, "crossings")])
            db = TrafficAnalyticsDatabase(path)
            (row,) = db.query_timeseries()
            self.assertEqual((row["hour"], row["minute"], row["day_of_week"], row["is_weekend"]), (3, 30, 0, 0))
            self.assertEqual(row["timestamp_local_iso"], datetime.fromtimestamp(ts, WIB).isoformat())
            self.assertAlmostEqual(row["sin_hour"], math.sin(2 * math.pi * 3.5 / 24), places=4)
            # already migrated: a second open must not touch rows again
            db._writer().execute("UPDATE camera_traffic_timeseries SET hour = 99")
            db._writer().commit()
            db.close()
            again = TrafficAnalyticsDatabase(path)
            self.assertEqual(again.query_timeseries()[0]["hour"], 99)
            again.close()

    def test_empty_track_windows_after_the_first_crossings_window_become_crossings(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "old.db")
            self.make_legacy(path, [(1000.0, 5, "tracks"), (1010.0, 0, "tracks"),        # before the caller reported crossings
                                    (1020.0, 3, "crossings"), (1030.0, 0, "tracks"),     # empty windows of the crossings era
                                    (1040.0, 7, "tracks")])                              # a real track-based window
            db = TrafficAnalyticsDatabase(path)
            got = [(r["timestamp"], r["volume_basis"]) for r in db.query_timeseries()]
            db.close()
            self.assertEqual(got, [(1000.0, "tracks"), (1010.0, "tracks"), (1020.0, "crossings"),
                                   (1030.0, "crossings"), (1040.0, "tracks")])

    def test_a_new_database_needs_no_migration_and_has_all_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = TrafficAnalyticsDatabase(os.path.join(tmp, "new.db"))
            cols = {r[1] for r in db._read("PRAGMA table_info(camera_traffic_timeseries)")}
            db.close()
        self.assertLessEqual({"timestamp_local_iso", "median_speed_bps", "mean_speed_bps", "queue_share",
                              "avg_traffic_vehicles", "volume_basis"}, cols)


if __name__ == "__main__":
    unittest.main()
