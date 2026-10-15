"""
tests/test_volume_rates.py
Traffic volume as rates: observation windows -> vehicles/hour per camera, and a citywide figure
that counts each junction once.
"""

import os
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

import backend.server as server
from backend.database import Database
from backend.dataset import DatasetStream, flow_series


def window(cid, end, observed, vehicles, pos=0, neg=0, cars=0, motos=0, line=0, passage=0):
    return {"camera_id": cid, "start_ts": end - observed, "end_ts": end, "start_iso": "x",
            "observed_s": observed, "vehicles": vehicles, "dir_pos": pos, "dir_neg": neg,
            "cars": cars, "motorcycles": motos, "line_counts": line, "passage_counts": passage, "kind": "visit"}


class CameraRatesTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "rates.sqlite"), auto_seed=True)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_rate_is_crossings_over_observed_time(self):
        now = time.time()
        self.db.insert_observation_window(window(79, now - 100, 15.0, 5, pos=3, neg=2, cars=4, motos=1, line=5))
        self.db.insert_observation_window(window(79, now - 50, 15.0, 1, pos=1, cars=1, line=1))
        self.db.insert_observation_window(window(87, now - 10, 12.0, 0))            # quiet but observed
        self.db.insert_observation_window(window(83, now - 7200, 12.0, 9))          # outside the hour
        rates = self.db.get_camera_rates(window_s=3600, now=now)
        self.assertEqual(set(rates), {79, 87})
        r = rates[79]
        self.assertEqual((r["observed_s"], r["vehicles"], r["windows"]), (30.0, 6, 2))
        self.assertEqual(r["vph"], 720.0)                     # 6 vehicles / 30 s * 3600
        self.assertEqual((r["vph_dir_pos"], r["vph_dir_neg"]), (480.0, 240.0))
        self.assertEqual(r["vph_classes"]["car"], 600.0)
        self.assertEqual(r["counting_mode"], "line")
        self.assertEqual(rates[87]["vph"], 0.0)
        self.assertTrue(r["low_sample"])                      # 30 s observed: rough estimate

    def test_first_observed_is_the_oldest_window(self):
        self.assertIsNone(self.db.get_camera_first_observed(79))
        self.db.insert_observation_window(window(79, 5000.0, 15.0, 1))
        self.db.insert_observation_window(window(79, 2000.0, 15.0, 1))
        self.db.insert_observation_window(window(87, 1000.0, 15.0, 1))
        self.assertEqual(self.db.get_camera_first_observed(79), 2000.0)


class CitywideRateTest(unittest.TestCase):
    def test_cameras_at_one_junction_count_once(self):
        cluster = server.topology.get_intra_junction_cluster(79)
        self.assertIn(199, cluster)                            # Tugu Kujang hub
        rates = {79: {"vph": 900.0}, 199: {"vph": 600.0}, 167: {"vph": 300.0}}
        self.assertNotIn(167, cluster)
        self.assertEqual(server._citywide_vph(rates), 1200.0)  # max(900, 600) + 300

    def test_removed_cameras_are_not_citywide(self):
        self.assertNotIn(1001, server.topology.cameras)            # a dropped Tasikmalaya camera
        self.assertEqual(server._citywide_vph({167: {"vph": 300.0}, 1001: {"vph": 500.0}}), 300.0)

    def test_recent_limit_is_clamped(self):
        client = TestClient(server.app)
        for limit in (-1, 0, 10 ** 9):
            self.assertEqual(client.get(f"/api/recent?limit={limit}").status_code, 200)
        calls = []
        real = server.db.get_latest_sightings
        server.db.get_latest_sightings = lambda limit: calls.append(limit) or []
        try:
            server._stats_cache.clear()
            client.get("/api/recent?limit=-1")
            client.get("/api/recent?limit=999999")
        finally:
            server.db.get_latest_sightings = real
        self.assertEqual(calls, [1, server.RECENT_MAX_LIMIT])

    def test_counts_endpoint_exposes_rates(self):
        data = TestClient(server.app).get("/api/cameras/counts").json()
        for key in ("camera_rates", "camera_vph_map", "cameras_with_rates", "citywide_vph", "rate_window_minutes"):
            self.assertIn(key, data)
        self.assertTrue(all("rate" in c for c in data["camera_counts"]))


if __name__ == "__main__":
    unittest.main()


class CameraFlowSeriesTest(unittest.TestCase):
    """The Analytics tab's chart: counting-line flow per time bucket, gaps left empty. It uses the
    export's bucketing rule, so the two cannot disagree."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "series.sqlite"), auto_seed=True)
        self.no_analytics = os.path.join(self.tmp.name, "absent.db")
        self.now = 10_000.0     # buckets of 5 min = 300 s: [9600, 9900) and [9900, 10200)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def series(self, camera=79):
        return flow_series(camera, minutes=10, bucket_minutes=5, now=self.now,
                           main_db=self.db.db_path, analytics_db=self.no_analytics)

    def test_buckets_sum_crossings_over_observed_seconds(self):
        self.db.insert_observation_window(window(79, 9650, 15.0, 3))
        self.db.insert_observation_window(window(79, 9700, 15.0, 2))
        self.db.insert_observation_window(window(79, 9950, 40.0, 10))
        series = [p for p in self.series() if p["windows"]]
        self.assertEqual([p["t"] for p in series], [9600.0, 9900.0])
        first, second = series
        self.assertEqual((first["vehicles"], first["observed_s"], first["windows"]), (5, 30.0, 2))
        self.assertAlmostEqual(first["vph"], 5 / 30 * 3600, places=1)
        self.assertTrue(first["low_sample"])
        self.assertAlmostEqual(second["vph"], 900.0, places=1)
        self.assertTrue(second["low_sample"])                        # 40 s observed

    def test_a_window_on_a_bucket_edge_is_shared_out_by_overlap(self):
        self.db.insert_observation_window(window(79, 9950, 60.0, 6))     # 9890-9950: 10 s | 50 s
        before, after = [p for p in self.series() if p["windows"]]
        self.assertEqual((before["t"], after["t"]), (9600.0, 9900.0))
        self.assertAlmostEqual(before["vehicles"], 1.0)
        self.assertAlmostEqual(after["vehicles"], 5.0)
        self.assertAlmostEqual(before["observed_s"] + after["observed_s"], 60.0)

    def test_other_cameras_and_old_windows_are_excluded(self):
        self.db.insert_observation_window(window(80, 9950, 60.0, 10))
        self.db.insert_observation_window(window(79, 5000, 60.0, 10))
        self.assertEqual(self.series(), [])

    def test_quiet_but_observed_bucket_is_a_zero_rate_not_a_gap(self):
        self.db.insert_observation_window(window(79, 9950, 20.0, 0))
        (point,) = self.series()
        self.assertEqual(point["vph"], 0.0)

    def test_congestion_joins_by_bucket_and_a_bucket_without_observation_still_shows(self):
        from car_tracker.analytics import TrafficAnalyticsDatabase
        from tests.test_traffic_analytics import window as analytics_window
        an_path = os.path.join(self.tmp.name, "an.db")
        an = TrafficAnalyticsDatabase(an_path)
        an.insert_metrics(analytics_window(79, 9700.0, 60, congestion=0.2))       # 9640-9700
        an.insert_metrics(analytics_window(79, 9760.0, 60, congestion=0.6))       # 9700-9760
        an.insert_metrics(analytics_window(79, 10100.0, 60, congestion=0.9))      # bucket 9900: no observation
        self.db.insert_observation_window(window(79, 9650, 15.0, 3))
        pts = {p["t"]: p for p in flow_series(79, 10, 5, now=self.now, main_db=self.db.db_path, analytics_db=an_path)}
        self.assertAlmostEqual(pts[9600.0]["congestion"], 0.4, places=3)           # equal-length windows
        self.assertEqual((pts[9900.0]["vph"], pts[9900.0]["congestion"]), (None, 0.9))

    def test_chart_and_export_agree_for_the_same_camera_and_hour(self):
        for end, secs, vehicles in ((9650, 15.0, 3), (9700, 15.0, 2), (9950, 60.0, 6), (10100, 30.0, 4)):
            self.db.insert_observation_window(window(79, end, secs, vehicles))
        chart = {p["t"]: p for p in self.series()}
        export = DatasetStream(9300.0, 10200.0, 5, camera_ids=[79], main_db=self.db.db_path,
                               analytics_db=self.no_analytics)
        rows = {r["bucket_start_ts"]: r for r in export}
        self.assertEqual(set(chart), set(rows))
        for t, row in rows.items():
            self.assertAlmostEqual(chart[t]["vph"], row["vph"], places=1)
            self.assertAlmostEqual(chart[t]["observed_s"], row["observed_s"], places=1)
            self.assertAlmostEqual(chart[t]["vehicles"], row["crossings"], places=1)


class CameraMetricsEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_metrics_expose_counting_line_flow(self):
        res = self.client.get("/api/cameras/167/metrics?minutes=60&bucket_minutes=5")
        self.assertEqual(res.status_code, 200)
        flow = res.json()["flow"]
        self.assertEqual(flow["basis"], "counting-line crossings")
        self.assertEqual((flow["window_minutes"], flow["bucket_minutes"]), (60.0, 5.0))
        self.assertIsInstance(flow["series"], list)
        for point in flow["series"]:
            self.assertEqual(set(point) >= {"t", "t_end", "vph", "observed_s", "congestion"}, True)
        if flow["rate"] is not None:
            self.assertEqual(flow["windows"], flow["rate"]["windows"])

    def test_window_parameters_are_clamped(self):
        flow = self.client.get("/api/cameras/167/metrics?minutes=99999&bucket_minutes=0").json()["flow"]
        self.assertEqual(flow["window_minutes"], 43200.0)      # 30 days, the chart's longest range
        self.assertEqual(flow["bucket_minutes"], 60.0)         # raised to keep <= 720 buckets
        flow = self.client.get("/api/cameras/167/metrics?minutes=60&bucket_minutes=0").json()["flow"]
        self.assertEqual(flow["bucket_minutes"], 1.0)

    def test_metrics_report_where_the_stored_history_starts(self):
        flow = self.client.get("/api/cameras/167/metrics?minutes=10080&bucket_minutes=120").json()["flow"]
        self.assertEqual((flow["window_minutes"], flow["bucket_minutes"]), (10080.0, 120.0))
        self.assertIn("history_start_ts", flow)
        if flow["series"]:
            self.assertLessEqual(flow["history_start_ts"], flow["series"][0]["t_end"])
