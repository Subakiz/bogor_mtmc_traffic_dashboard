"""
backend/insights.py and /api/insights/citywide: the City tab's citywide patterns.
Synthetic databases in temp dirs; WIB = UTC+7; 2026-10-12 is a Monday.
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient

from backend import insights
from tests.test_dataset_fixtures import CAMERAS, DatasetTestCase, wib

NOW = wib("2026-10-14T12:00:00")          # "today" is Wednesday the 14th
JUNCTION = {167: [131, 167], 131: [131, 167]}


def obs(cam, start, seconds, vehicles, **kw):
    t = wib(start)
    return {"camera_id": cam, "start": t, "end": t + seconds, "vehicles": vehicles, **kw}


class TestCitywideInsights(DatasetTestCase):
    def run_insights(self, **kw):
        return insights.citywide_insights(7, now=NOW, main_db=self.main, analytics_db=self.analytics, **kw)

    def test_hour_profile_is_the_median_camera_and_keeps_today_apart(self):
        self.build(CAMERAS, windows=[
            obs(167, "2026-10-12T06:10:00", 600, 200),     # 1200 veh/h
            obs(131, "2026-10-12T06:20:00", 600, 300),     # 1800 veh/h
            obs(167, "2026-10-14T06:10:00", 600, 100),     # today: 600 veh/h
            obs(131, "2026-10-13T06:10:00", 30, 50),       # 30 s watched: too thin to count
        ])
        h6 = self.run_insights()["hours"][6]
        self.assertEqual((h6["vph"], h6["camera_hours"], h6["n_days"]), (1500.0, 2, 1))
        self.assertEqual((h6["today"]["vph"], h6["today"]["camera_hours"]), (600.0, 1))

    def test_hour_flow_weights_camera_hours_by_watched_time(self):
        self.build(CAMERAS, windows=[obs(167, "2026-10-12T06:10:00", 600, 200),     # 1200 veh/h, 10 min
                                     obs(131, "2026-10-12T06:20:00", 120, 60)])     # 1800 veh/h, 2 min
        self.assertEqual(self.run_insights()["hours"][6]["vph"], 1200.0)           # unweighted: 1500

    def test_corridor_peak_tie_takes_the_earliest_hour(self):
        self.build(CAMERAS, analytics=[{"camera_id": 167, "end": wib(f"2026-10-12T{h}:30:00"), "seconds": s,
                                        "congestion": .5} for h, s in (("09", 300), ("16", 200))])
        cor = {c["corridor"]: c for c in self.run_insights()["corridors"]}
        self.assertEqual((cor["BOGOR_SELATAN"]["peak_hour"], cor["BOGOR_SELATAN"]["peak_ci"]), (9, .5))

    def test_slow_share_counts_junction_hours_at_moderate_or_worse(self):
        self.build(CAMERAS, analytics=[
            {"camera_id": 167, "end": wib("2026-10-12T16:30:00"), "seconds": 300, "congestion": .6},
            {"camera_id": 131, "end": wib("2026-10-12T16:30:00"), "seconds": 300, "congestion": .1},
            {"camera_id": 131, "end": wib("2026-10-12T17:30:00"), "seconds": 20, "congestion": .9},   # too short
        ])
        out = self.run_insights()                            # no junction map: two one-camera junctions
        self.assertEqual((out["hours"][16]["slow_share"], out["hours"][16]["measured"]), (.5, 2))
        self.assertIsNone(out["hours"][17]["slow_share"])
        joined = self.run_insights(clusters=JUNCTION)       # one 2-camera junction, slow on one side: slow once
        self.assertEqual((joined["hours"][16]["slow_share"], joined["hours"][16]["measured"]), (1.0, 1))
        cor = {c["corridor"]: c for c in out["corridors"]}
        self.assertEqual(cor["BOGOR_SELATAN"]["cells"][16]["label"], "Heavy")      # 0.6: Heavy starts at 0.55
        self.assertEqual(cor["BOGOR_SELATAN"]["peak_hour"], 16)
        self.assertEqual(out["corridors"][0]["corridor"], "BOGOR_SELATAN")    # most congested corridor first

    def test_rankings_keep_one_camera_per_junction(self):
        self.build(CAMERAS, windows=[obs(167, "2026-10-12T09:00:00", 2000, 500), obs(131, "2026-10-12T09:00:00", 2000, 900)],
                   analytics=[{"camera_id": c, "end": wib("2026-10-12T09:30:00"), "seconds": 1000, "congestion": v}
                              for c, v in ((167, .7), (131, .3))])
        out = self.run_insights(clusters=JUNCTION)
        self.assertEqual([(r["camera_id"], r["junction_size"]) for r in out["congested"]], [(167, 2)])
        self.assertEqual(out["congested"][0]["congestion_label"], "Heavy")
        self.assertEqual(out["congested"][0]["worst_hour"], 9)
        self.assertEqual([r["camera_id"] for r in out["busiest"]], [131])
        alone = self.run_insights()                         # no junction map: each camera stands alone
        self.assertEqual([r["camera_id"] for r in alone["congested"]], [167, 131])

    def test_thinly_watched_cameras_are_not_ranked(self):
        self.build(CAMERAS, windows=[obs(167, "2026-10-12T09:00:00", 600, 500)],
                   analytics=[{"camera_id": 167, "end": wib("2026-10-12T09:30:00"), "seconds": 300, "congestion": .9}])
        out = self.run_insights()
        self.assertEqual((out["congested"], out["busiest"]), ([], []))

    def test_vehicle_mix_shares(self):
        self.build(CAMERAS, windows=[obs(167, "2026-10-12T06:10:00", 600, 100, cars=40, motos=50, trucks=8, buses=2)])
        out = self.run_insights()
        self.assertEqual(out["hours"][6]["mix"], {"car": .4, "motorcycle": .5, "truck": .08, "bus": .02})
        self.assertIsNone(out["hours"][7]["mix"]["car"])
        self.assertEqual(out["corridors"][0]["mix"]["motorcycle"], .5)

    def test_camera_filter_and_empty_data(self):
        self.build(CAMERAS, windows=[obs(131, "2026-10-12T06:10:00", 600, 100)])
        out = self.run_insights(camera_ids=[167])
        self.assertEqual(out["cameras"], 0)
        self.assertEqual(len(out["hours"]), 24)
        self.assertTrue(all(h["vph"] is None for h in out["hours"]))
        self.assertEqual((out["now_hour"], out["today"]), (12, "2026-10-14"))


class TestCitywideEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from backend import server
        cls.server = server
        cls.client = TestClient(server.app)

    def setUp(self):
        self.server._insights_cache.clear()

    def test_shape_and_now(self):
        res = self.client.get("/api/insights/citywide")
        self.assertEqual(res.status_code, 200)
        body = res.json()
        self.assertEqual(len(body["hours"]), 24)
        self.assertEqual(set(body["now"]), {"vph", "cameras", "typical_vph"})
        for key in ("corridors", "congested", "busiest", "baseline_days", "observed_hours"):
            self.assertIn(key, body)

    def test_days_clamped_and_cached(self):
        real = insights.citywide_insights
        with mock.patch.object(self.server.insights, "citywide_insights", side_effect=real) as spy:
            self.client.get("/api/insights/citywide?days=100")
            self.client.get("/api/insights/citywide?days=100")
        self.assertEqual(spy.call_count, 1)
        self.assertEqual(spy.call_args.args[0], 28)
        kw = spy.call_args.kwargs
        self.assertEqual((kw["main_db"], kw["analytics_db"]), (self.server.db.db_path, self.server.ANALYTICS_DB_PATH))


class TestBaselineKind(DatasetTestCase):
    def test_weekday_baseline_leaves_weekends_out_until_none_are_left(self):
        self.build(CAMERAS, windows=[obs(167, "2026-10-12T06:10:00", 600, 200),     # Monday: 1200 veh/h
                                     obs(167, "2026-10-11T06:10:00", 600, 20)])     # Sunday: 120 veh/h
        out = insights.citywide_insights(7, now=NOW, main_db=self.main, analytics_db=self.analytics)
        self.assertEqual((out["baseline_kind"], out["hours"][6]["vph"]), ("weekday", 1200.0))
        saturday = wib("2026-10-17T12:00:00")
        out = insights.citywide_insights(7, now=saturday, main_db=self.main, analytics_db=self.analytics)
        self.assertEqual((out["baseline_kind"], out["hours"][6]["vph"]), ("weekend", 120.0))

    def test_falls_back_to_all_days(self):
        self.build(CAMERAS, windows=[obs(167, "2026-10-11T06:10:00", 600, 20)])     # only a Sunday seen
        out = insights.citywide_insights(7, now=NOW, main_db=self.main, analytics_db=self.analytics)
        self.assertEqual((out["baseline_kind"], out["hours"][6]["vph"]), ("all", 120.0))
