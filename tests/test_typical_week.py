"""
backend/typical_week.py: the typical congestion per (weekday, hour), per camera and per junction cluster.
Synthetic databases in temp dirs; WIB = UTC+7.
"""

import sqlite3
import unittest

from backend import typical_week as tw
from tests.test_dataset_fixtures import CAMERAS, DatasetTestCase, wib


def row(cam, date, hour, congestion, seconds=300.0, dow=0):
    return {"camera_id": cam, "bucket_start_local": f"{date}T{hour:02d}:00:00+07:00", "hour": hour,
            "day_of_week": dow, "congestion_index": congestion, "analytics_s": seconds}


def slot(slots, scope, ref, dow, hour):
    hit = [s for s in slots if (s["scope"], s["ref_id"], s["day_of_week"], s["hour"]) == (scope, ref, dow, hour)]
    return hit[0] if hit else None


class TestTypicalSlots(unittest.TestCase):
    def test_leans_busy_across_days(self):
        # three Mondays at 8 o'clock: two quiet, one jammed. The mean (0.4) would hide the jam.
        rows = [row(1, d, 8, c) for d, c in (("2026-09-14", .1), ("2026-09-21", .1), ("2026-09-28", 1.0))]
        s = slot(tw.typical_slots(rows, [(1,)]), "camera", 1, 0, 8)
        self.assertEqual(s["n_days"], 3)
        self.assertEqual(s["value"], 0.55)     # 75th percentile, linear interpolation

    def test_hours_too_thinly_watched_do_not_count(self):
        rows = [row(1, "2026-09-14", 8, .9, seconds=10), row(1, "2026-09-21", 8, .2, seconds=300),
                row(1, "2026-09-28", 8, None, seconds=300)]
        s = slot(tw.typical_slots(rows, [(1,)]), "camera", 1, 0, 8)
        self.assertEqual((s["n_days"], s["value"]), (1, .2))

    def test_few_days_are_flagged_low_data(self):
        rows = [row(1, "2026-09-14", 8, .5), row(1, "2026-09-21", 9, .5), row(1, "2026-09-28", 9, .5)]
        slots = tw.typical_slots(rows, [(1,)])
        self.assertTrue(slot(slots, "camera", 1, 0, 8)["low_data"])
        self.assertFalse(slot(slots, "camera", 1, 0, 9)["low_data"])

    def test_cluster_takes_the_busiest_camera(self):
        rows = [row(1, d, 8, .1) for d in ("2026-09-14", "2026-09-21")] + \
               [row(2, d, 8, .9) for d in ("2026-09-14", "2026-09-21")]
        slots = tw.typical_slots(rows, [(1, 2)])
        self.assertEqual(slot(slots, "cluster", 1, 0, 8)["value"], .9)     # keyed by the lowest camera id
        self.assertEqual(slot(slots, "cluster", 1, 0, 8)["n_days"], 2)

    def test_cluster_thinness_follows_the_camera_that_set_the_value(self):
        rows = [row(1, d, 8, .2) for d in ("2026-09-14", "2026-09-21", "2026-09-28")] + [row(2, "2026-09-14", 8, .9)]
        s = slot(tw.typical_slots(rows, [(1, 2)]), "cluster", 1, 0, 8)
        self.assertEqual((s["value"], s["n_days"], s["low_data"]), (.9, 1, 1))

    def test_weekdays_are_kept_apart(self):
        rows = [row(1, "2026-09-14", 8, .1, dow=0), row(1, "2026-09-15", 8, .9, dow=1)]
        slots = tw.typical_slots(rows, [(1,)])
        self.assertEqual(slot(slots, "camera", 1, 0, 8)["value"], .1)
        self.assertEqual(slot(slots, "camera", 1, 1, 8)["value"], .9)

    def test_camera_in_no_cluster_still_gets_its_own_cluster(self):
        slots = tw.typical_slots([row(7, "2026-09-14", 8, .4)], [])
        self.assertEqual(slot(slots, "cluster", 7, 0, 8)["value"], .4)


class TestRefresh(DatasetTestCase):
    def test_refresh_stores_and_replaces(self):
        windows = [{"camera_id": 167, "end": wib(f"2026-10-{d:02d}T08:30:00"), "seconds": 120, "congestion": c}
                   for d, c in ((5, .2), (12, .8))]       # Mondays 5th and 12th (2026-10-05 is a Monday)
        self.build(CAMERAS, analytics=windows)
        now = wib("2026-10-14T12:00:00")
        n = tw.refresh(self.main, self.analytics, [(167, 131)], now=now, days=14)
        self.assertGreater(n, 0)
        conn = sqlite3.connect(self.main)
        got = conn.execute("SELECT value, n_days, low_data FROM typical_congestion WHERE scope='camera' "
                           "AND ref_id=167 AND day_of_week=0 AND hour=8").fetchone()
        self.assertEqual(got[1:], (2, 0))
        self.assertAlmostEqual(got[0], .2 + (.8 - .2) * .75, places=3)
        self.assertEqual(tw.refresh(self.main, self.analytics, [(167, 131)], now=now, days=14), n)   # idempotent
        self.assertEqual(conn.execute("SELECT COUNT(*) FROM typical_congestion").fetchone()[0], n)
        conn.close()

    def test_refresh_with_no_analytics_leaves_the_table_empty(self):
        self.build(CAMERAS)
        self.assertEqual(tw.refresh(self.main, self.analytics, [(167,)], now=wib("2026-10-14T12:00:00")), 0)


if __name__ == "__main__":
    unittest.main()
