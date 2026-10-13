"""
Database.search_sightings (index-backed free-text search) and the camera_ids filter of
get_camera_rates.
"""

import os
import tempfile
import time
import unittest

from backend.database import Database


def ts(i):
    return "2026-10-01T%02d:%02d:%02dZ" % (i // 3600 % 24, i // 60 % 60, i % 60)


class SearchSightingsTest(unittest.TestCase):
    """Camera 79 'CCTV - TUGU KUJANG' (SSA_RING), 199 'CCTV TUGU ARAH OTISTA' (SSA_RING),
    87 'CCTV OTISTA' (SSA_RING), 161 'CCTV SIMPANG A. YANI - DADALI 2 -F' (PAJAJARAN_CORRIDOR)."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Database(os.path.join(cls.tmp.name, "s.sqlite"), auto_seed=True)
        # (camera, track, class)
        plan = [(79, 1, "car"), (79, 2, "motorcycle"), (87, 3, "car"), (87, 4, "bus"),
                (199, 5, "truck"), (161, 6, "car"), (161, 7, "motorcycle"), (79, 8, "car"),
                (83, 9, "car"), (92, 10, "angkot")]
        cls.sid = {}
        for i, (cam, track, klass) in enumerate(plan):
            cls.sid[track] = cls.db.insert_vehicle_sighting(cam, track, klass, 0.9, (0, 0, 1, 1), 0.0, 0.0, ts(i))
        cls.db.optimize()

    @classmethod
    def tearDownClass(cls):
        cls.db.close()
        cls.tmp.cleanup()

    def tracks(self, query, **kw):
        return [r["local_track_id"] for r in self.db.search_sightings(query, **kw)]

    def test_row_shape_matches_get_latest_sightings(self):
        latest = self.db.get_latest_sightings(1)[0]
        found = self.db.search_sightings("79", limit=1)[0]
        self.assertEqual(set(found), set(latest))
        self.assertIn("camera_name", found)
        self.assertIn("corridor", found)

    def test_numeric_matches_the_camera_id_exactly(self):
        self.assertEqual(self.tracks("79"), [8, 2, 1])                        # camera 79, newest first
        self.assertEqual(self.tracks("#79"), [8, 2, 1])
        self.assertEqual(self.tracks("161"), [7, 6])
        self.assertEqual(self.tracks("10"), [], "exact match only: no camera 10")
        self.assertEqual(self.tracks("99999"), [])
        self.assertEqual(self.tracks("9" * 40), [], "beyond SQLite's integer range: no match, no OverflowError")
        self.assertEqual(self.tracks("\u00b2"), [], "non-ASCII digits are text, not an id")

    def test_vehicle_class_is_an_exact_case_insensitive_match(self):
        self.assertEqual(self.tracks("car"), [9, 8, 6, 3, 1])
        self.assertEqual(self.tracks("CAR"), [9, 8, 6, 3, 1])
        self.assertEqual(self.tracks("bus"), [4])
        self.assertEqual(self.tracks("angkot"), [10])
        self.assertEqual(self.tracks("ca"), [], "no partial class names")
        self.assertEqual(self.tracks("train"), [], "a class that never occurred is answered from the totals, not by a scan")

    def test_camera_name_and_corridor_are_case_insensitive_substrings(self):
        self.assertEqual(sorted(self.tracks("tugu")), [1, 2, 5, 8])           # cameras 79 and 199
        self.assertEqual(self.tracks("Tugu Kujang"), [8, 2, 1])
        self.assertEqual(sorted(self.tracks("otista")), [3, 4, 5])            # 87 and 199
        self.assertEqual(sorted(self.tracks("pajajaran")), [6, 7])            # corridor of 161
        self.assertEqual(sorted(self.tracks("ssa_ring")), [1, 2, 3, 4, 5, 8, 9, 10])
        self.assertEqual(sorted(self.tracks("ssa ring")), [1, 2, 3, 4, 5, 8, 9, 10], "a space matches the corridor underscore")

    def test_like_wildcards_in_the_query_are_literal(self):
        self.assertEqual(self.tracks("%"), [])
        self.assertEqual(self.tracks("tugu_"), [], "an unescaped _ would match the space in 'TUGU KUJANG'")
        self.assertEqual(self.tracks("tugu%"), [])
        self.assertEqual(self.tracks("'; DROP TABLE cameras; --"), [])
        self.assertGreater(self.db.conn.execute("SELECT COUNT(*) FROM cameras").fetchone()[0], 90)

    def test_sources_are_merged_deduplicated_and_ordered(self):
        # "otista" matches cameras 87 and 199; "car" would add class matches -- here only the name rule fires
        rows = self.db.search_sightings("otista")
        ids = [r["sighting_id"] for r in rows]
        self.assertEqual(len(ids), len(set(ids)))
        stamps = [(r["timestamp_iso"], r["sighting_id"]) for r in rows]
        self.assertEqual(stamps, sorted(stamps, reverse=True))

    def test_limit_applies_to_the_merged_result(self):
        self.assertEqual(self.tracks("car", limit=2), [9, 8])
        self.assertEqual(self.tracks("tugu", limit=3), [8, 5, 2])
        self.assertEqual(self.tracks("car", limit=0), [])

    def test_empty_query_returns_the_latest_sightings(self):
        self.assertEqual(self.tracks("", limit=3), [10, 9, 8])
        self.assertEqual(self.tracks("   ", limit=2), [10, 9])

    def test_common_queries_use_indexes_not_table_scans(self):
        statements = []
        self.db.conn.set_trace_callback(lambda sql: statements.append(sql)
                                         if sql.startswith("SELECT timestamp_iso, sighting_id") else None)
        try:
            for q in ("161", "79", "tugu", "ssa ring", "car"):
                self.db.search_sightings(q)
        finally:
            self.db.conn.set_trace_callback(None)
        # the candidate lookups; the final fetch by sighting_id is a primary-key lookup on a real
        # table (on this 10-row fixture the planner may prefer to scan it)
        self.assertGreaterEqual(len(statements), 5)
        for sql in statements:
            plan = " | ".join(r[3] for r in self.db.conn.execute("EXPLAIN QUERY PLAN " + sql))
            for step in plan.split(" | "):
                if step.startswith(("SCAN vehicle_sightings", "SCAN s")):
                    self.assertIn("USING", step, f"unindexed scan of vehicle_sightings in: {sql}\n{plan}")
            self.assertNotIn("idx_sightings_agg", plan)


class CameraRatesFilterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "rates.sqlite"), auto_seed=True)
        self.now = time.time()
        for cam, vehicles in ((79, 5), (87, 3), (83, 9)):
            self.db.insert_observation_window({
                "camera_id": cam, "start_ts": self.now - 40, "end_ts": self.now - 10, "start_iso": "x",
                "observed_s": 30.0, "vehicles": vehicles, "cars": vehicles, "line_counts": vehicles, "kind": "visit"})

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_default_still_returns_every_observed_camera(self):
        self.assertEqual(set(self.db.get_camera_rates(window_s=3600, now=self.now)), {79, 87, 83})

    def test_filter_returns_only_the_requested_cameras_with_identical_values(self):
        everything = self.db.get_camera_rates(window_s=3600, now=self.now)
        one = self.db.get_camera_rates(window_s=3600, now=self.now, camera_ids=[79])
        self.assertEqual(one, {79: everything[79]})
        two = self.db.get_camera_rates(3600, self.now, None, camera_ids=(87, 83))
        self.assertEqual(two, {87: everything[87], 83: everything[83]})

    def test_unobserved_empty_and_duplicate_ids(self):
        self.assertEqual(self.db.get_camera_rates(window_s=3600, now=self.now, camera_ids=[161]), {})
        self.assertEqual(self.db.get_camera_rates(window_s=3600, now=self.now, camera_ids=[]), {})
        self.assertEqual(set(self.db.get_camera_rates(window_s=3600, now=self.now, camera_ids=[79, 79, 87])), {79, 87})

    def test_single_camera_query_uses_the_per_camera_index(self):
        plan = " ".join(r[3] for r in self.db.conn.execute(
            "EXPLAIN QUERY PLAN SELECT camera_id, SUM(vehicles) FROM observation_windows "
            "WHERE end_ts >= ? AND camera_id IN (?) GROUP BY camera_id", (0, 79)))
        self.assertIn("idx_obs_cam_time", plan)


if __name__ == "__main__":
    unittest.main()
