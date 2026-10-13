"""
Trigger-maintained all-time totals (backend/database.py: TOTALS_*): they must equal the old
COUNT / COUNT(DISTINCT) scans over vehicle_sightings, survive the one-off migration of a database
that predates them, and keep counting after old sightings are deleted.
"""

import os
import random
import sqlite3
import tempfile
import threading
import unittest

from backend.database import Database, TOTALS_MARKER

CAMERAS = (79, 87, 83, 161, 75, 92)
CLASSES = ("car", "motorcycle", "truck", "bus", "angkot")


def reference_stats(conn):
    """The aggregates exactly as the pre-trigger implementation computed them (full scans)."""
    one = lambda sql: conn.execute(sql).fetchone()[0]
    per_cam = {r[0]: r[1:] for r in conn.execute(
        "SELECT camera_id, COUNT(*), COUNT(DISTINCT local_track_id), MAX(timestamp_iso), "
        "SUM(vehicle_class = 'car'), SUM(vehicle_class = 'motorcycle'), SUM(vehicle_class = 'truck'), "
        "SUM(vehicle_class IN ('bus', 'angkot')) FROM vehicle_sightings GROUP BY camera_id")}
    return {
        "total": one("SELECT COUNT(*) FROM vehicle_sightings"),
        "gid": one("SELECT COUNT(DISTINCT global_vehicle_id) FROM vehicle_sightings"),
        "tracks": one("SELECT COUNT(DISTINCT local_track_id) FROM vehicle_sightings"),
        "handoffs": one("SELECT COUNT(*) FROM handoff_events"),
        "classes": dict(conn.execute("SELECT vehicle_class, COUNT(*) FROM vehicle_sightings GROUP BY 1").fetchall()),
        "per_cam": per_cam,
    }


def random_rows(n, seed, with_null_gid=True):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        cam = rng.choice(CAMERAS)
        # small id ranges so tracks repeat on a camera and across cameras, like the live data
        track = rng.randint(1, max(2, n // 3))
        gid = None if (with_null_gid and rng.random() < 0.1) else rng.randint(1, max(2, n // 4))
        # timestamps deliberately not in insertion order
        ts = f"2026-10-0{rng.randint(1, 9)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
        rows.append((cam, track, gid, rng.choice(CLASSES), ts))
    return rows


def insert_rows(db, rows):
    for cam, track, gid, cls, ts in rows:
        db.insert_vehicle_sighting(cam, track, gid, cls, 0.9, (0, 0, 10, 10), 0.0, 0.0, b"", b"", ts)


def raw_insert_rows(conn, rows):
    conn.executemany(
        "INSERT INTO vehicle_sightings (camera_id, local_track_id, global_vehicle_id, vehicle_class, confidence, "
        "bbox_x1, bbox_y1, bbox_x2, bbox_y2, heading_deg, speed_px_s, embedding, color_hist, timestamp_iso) "
        "VALUES (?, ?, ?, ?, 0.9, 0, 0, 10, 10, 0, 0, X'', X'', ?)",
        [(c, t, g, k, ts) for c, t, g, k, ts in rows])


class TotalsMixin:
    def assert_totals_match_scans(self, db):
        ref = reference_stats(db.conn)
        stats = db.get_traffic_stats()
        self.assertEqual(stats["total_sightings"], ref["total"])
        self.assertEqual(stats["total_vehicles_counted"], ref["total"])
        self.assertEqual(stats["unique_global_vehicles"], ref["gid"])
        self.assertEqual(stats["total_unique_vehicles"], ref["tracks"] or ref["gid"])
        self.assertEqual(stats["total_handoffs"], ref["handoffs"])
        self.assertEqual(stats["class_breakdown"], ref["classes"])
        counts = {c["camera_id"]: c for c in db.get_camera_vehicle_counts()}
        self.assertEqual(set(ref["per_cam"]), {cid for cid, c in counts.items() if c["total_sightings"]})
        for cid, (n, tracks, last, cars, motos, trucks, buses) in ref["per_cam"].items():
            c = counts[cid]
            self.assertEqual((c["total_sightings"], c["unique_vehicles"], c["last_seen_iso"]), (n, tracks, last), cid)
            self.assertEqual(c["classes"], {"car": cars, "motorcycle": motos, "truck": trucks, "bus": buses}, cid)
        self.assertEqual([c["total_sightings"] for c in db.get_camera_vehicle_counts()],
                         sorted((c["total_sightings"] for c in counts.values()), reverse=True))

    def assert_global_vehicles_match(self, db):
        """first/last = first/last *inserted* sighting of each vehicle (insertion order)."""
        conn = db.conn
        expected = {}
        for sid, gid, cam, ts in conn.execute(
                "SELECT sighting_id, global_vehicle_id, camera_id, timestamp_iso FROM vehicle_sightings "
                "WHERE global_vehicle_id IS NOT NULL ORDER BY sighting_id"):
            e = expected.setdefault(gid, {"first": (ts, cam), "n": 0})
            e["last"] = (ts, cam)
            e["n"] += 1
        got = {r[0]: r[1:] for r in conn.execute(
            "SELECT global_vehicle_id, first_seen_iso, first_camera_id, last_seen_iso, last_camera_id, sightings "
            "FROM global_vehicles")}
        self.assertEqual(set(got), set(expected))
        for gid, e in expected.items():
            self.assertEqual(got[gid], (e["first"][0], e["first"][1], e["last"][0], e["last"][1], e["n"]), gid)


class TriggerTotalsTest(TotalsMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "t.sqlite"), auto_seed=True)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_empty_database_reads_zero(self):
        stats = self.db.get_traffic_stats()
        self.assertEqual((stats["total_sightings"], stats["unique_global_vehicles"], stats["total_handoffs"]), (0, 0, 0))
        self.assertEqual(stats["class_breakdown"], {})
        self.assertTrue(all(c["total_sightings"] == 0 for c in self.db.get_camera_vehicle_counts()))

    def test_totals_equal_full_scans_on_random_data(self):
        for seed in (1, 2, 3):
            insert_rows(self.db, random_rows(400, seed))
            self.assert_totals_match_scans(self.db)
            self.assert_global_vehicles_match(self.db)

    def test_handoffs_are_counted(self):
        for i in range(5):
            self.db.record_handoff_event(i + 1, 79, 87, 0.9, "2026-10-01T00:00:0%dZ" % i)
        self.assertEqual(self.db.get_traffic_stats()["total_handoffs"], 5)
        self.assert_totals_match_scans(self.db)

    def test_failed_insert_does_not_move_counters(self):
        insert_rows(self.db, random_rows(20, 9))
        before = self.db.get_traffic_stats()
        with self.assertRaises(sqlite3.IntegrityError):   # unknown camera: FK violation, trigger rolled back too
            self.db.insert_vehicle_sighting(999999, 1, 1, "car", 0.9, (0, 0, 1, 1), 0, 0, b"", b"", "2026-10-01T00:00:00Z")
        self.assertEqual(self.db.get_traffic_stats(), before)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM sighting_tracks").fetchone()[0],
                         self.db.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT local_track_id, camera_id FROM vehicle_sightings)").fetchone()[0])

    def test_null_global_id_is_not_a_vehicle(self):
        self.db.insert_vehicle_sighting(79, 1, None, "car", 0.9, (0, 0, 1, 1), 0, 0, b"", b"", "2026-10-01T00:00:00Z")
        stats = self.db.get_traffic_stats()
        self.assertEqual((stats["total_sightings"], stats["unique_global_vehicles"]), (1, 0))
        self.assertEqual(stats["total_unique_vehicles"], 1)             # falls back to distinct local tracks
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM global_vehicles").fetchone()[0], 0)

    def test_track_ids_repeat_across_cameras_like_the_live_data(self):
        for cam in (79, 87):
            for _ in range(2):
                self.db.insert_vehicle_sighting(cam, 7, 70, "car", 0.9, (0, 0, 1, 1), 0, 0, b"", b"", "2026-10-01T00:00:00Z")
        stats = self.db.get_traffic_stats()
        self.assertEqual(stats["total_sightings"], 4)
        self.assertEqual(stats["total_unique_vehicles"], 1)             # COUNT(DISTINCT local_track_id) over everything
        counts = {c["camera_id"]: c for c in self.db.get_camera_vehicle_counts()}
        self.assertEqual((counts[79]["unique_vehicles"], counts[87]["unique_vehicles"]), (1, 1))
        self.assert_totals_match_scans(self.db)

    def test_global_vehicles_tracks_first_and_last_sighting(self):
        self.db.insert_vehicle_sighting(79, 1, 42, "car", 0.9, (0, 0, 1, 1), 0, 0, b"", b"", "2026-10-01T10:00:00Z")
        self.db.insert_vehicle_sighting(87, 2, 42, "car", 0.9, (0, 0, 1, 1), 0, 0, b"", b"", "2026-10-01T10:01:00Z")
        self.db.insert_vehicle_sighting(83, 3, 42, "car", 0.9, (0, 0, 1, 1), 0, 0, b"", b"", "2026-10-01T10:02:00Z")
        row = self.db.conn.execute("SELECT * FROM global_vehicles WHERE global_vehicle_id = 42").fetchone()
        self.assertEqual((row["first_seen_iso"], row["first_camera_id"]), ("2026-10-01T10:00:00Z", 79))
        self.assertEqual((row["last_seen_iso"], row["last_camera_id"], row["sightings"]), ("2026-10-01T10:02:00Z", 83, 3))

    def test_get_max_global_vehicle_id_survives_deleting_the_sightings(self):
        insert_rows(self.db, [(79, 1, 5, "car", "2026-01-01T00:00:00Z"), (79, 2, 900, "car", "2026-01-01T00:00:01Z")])
        self.assertEqual(self.db.get_max_global_vehicle_id(), 900)
        self.db.prune_sightings(older_than_s=0)         # delete everything
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM vehicle_sightings").fetchone()[0], 0)
        self.assertEqual(self.db.get_max_global_vehicle_id(), 900, "the id allocator must never reuse an id")

    def test_concurrent_writers_keep_totals_exact(self):
        def work(offset):
            insert_rows(self.db, [(CAMERAS[i % len(CAMERAS)], offset + i, offset + i // 2, CLASSES[i % len(CLASSES)],
                                   "2026-10-01T00:00:%02dZ" % (i % 60)) for i in range(150)])
        threads = [threading.Thread(target=work, args=(k * 1000,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.db.get_traffic_stats()["total_sightings"], 600)
        self.assert_totals_match_scans(self.db)
        self.assert_global_vehicles_match(self.db)

    def test_rebuild_totals_repairs_drifted_counters(self):
        insert_rows(self.db, random_rows(300, 4))
        with self.db.conn:
            self.db.conn.execute("UPDATE sighting_counters SET value = value + 17")
            self.db.conn.execute("DELETE FROM sighting_totals WHERE camera_id = 79")
        self.db.rebuild_totals()
        self.assert_totals_match_scans(self.db)
        self.assert_global_vehicles_match(self.db)

    def test_windowed_counts_still_scan_the_requested_window(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        iso = lambda s: (now - timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        insert_rows(self.db, [(79, 1, 1, "car", iso(30)), (79, 2, 2, "truck", iso(60)), (79, 3, 3, "car", iso(7200))])
        recent = {c["camera_id"]: c for c in self.db.get_camera_vehicle_counts(window_s=300)}
        self.assertEqual(recent[79]["total_sightings"], 2)
        self.assertEqual(recent[79]["classes"]["truck"], 1)
        self.assertEqual({c["camera_id"]: c for c in self.db.get_camera_vehicle_counts()}[79]["total_sightings"], 3)


class MigrationTest(TotalsMixin, unittest.TestCase):
    """A database created before the totals existed is converted once, on open."""

    LEGACY_INDEXES = ("idx_sightings_agg", "idx_sightings_local_track")

    def make_legacy_db(self, path, rows, handoffs=3):
        Database(path, auto_seed=True).close()
        c = sqlite3.connect(path)
        for trg in ("trg_sightings_totals", "trg_handoffs_total"):
            c.execute(f"DROP TRIGGER {trg}")
        for tbl in ("sighting_totals", "sighting_tracks", "camera_track_totals", "global_vehicles", "sighting_counters"):
            c.execute(f"DROP TABLE {tbl}")
        c.execute("CREATE INDEX idx_sightings_agg ON vehicle_sightings(camera_id, local_track_id, vehicle_class, timestamp_iso)")
        c.execute("CREATE INDEX idx_sightings_local_track ON vehicle_sightings(local_track_id)")
        c.execute("DELETE FROM maintenance_state WHERE key = ?", (TOTALS_MARKER,))
        c.execute("ALTER TABLE active_targets DROP COLUMN updated_ts")
        raw_insert_rows(c, rows)
        for i in range(handoffs):
            c.execute("INSERT INTO handoff_events (global_vehicle_id, from_camera_id, to_camera_id, confidence, timestamp_iso)"
                      " VALUES (?, 79, 87, 0.9, '2026-10-01T00:00:00Z')", (i + 1,))
        c.commit()
        c.close()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "legacy.sqlite")

    def tearDown(self):
        self.tmp.cleanup()

    def indexes(self, db):
        return {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")}

    def test_migration_backfills_exactly_and_drops_the_legacy_indexes(self):
        self.make_legacy_db(self.path, random_rows(600, 11))
        db = Database(self.path, auto_seed=False)
        try:
            self.assertTrue(db.migration_report["totals_backfilled"])
            self.assertEqual(sorted(db.migration_report["dropped_indexes"]), sorted(self.LEGACY_INDEXES))
            self.assertFalse(self.indexes(db) & set(self.LEGACY_INDEXES))
            self.assertEqual(db.get_traffic_stats()["total_handoffs"], 3)
            self.assert_totals_match_scans(db)
            self.assert_global_vehicles_match(db)
            self.assertIn("updated_ts", {r[1] for r in db.conn.execute("PRAGMA table_info(active_targets)")})
            # the triggers are live after the migration
            insert_rows(db, random_rows(100, 12))
            self.assert_totals_match_scans(db)
        finally:
            db.close()

    def test_migration_runs_once_and_reopening_changes_nothing(self):
        self.make_legacy_db(self.path, random_rows(200, 13))
        Database(self.path, auto_seed=False).close()
        db = Database(self.path, auto_seed=False)
        try:
            self.assertFalse(db.migration_report["totals_backfilled"])
            self.assertEqual(db.conn.execute("SELECT value FROM maintenance_state WHERE key = ?", (TOTALS_MARKER,)).fetchone()[0], 1)
            self.assert_totals_match_scans(db)
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0], 2)
        finally:
            db.close()

    def test_migrating_an_empty_legacy_database(self):
        self.make_legacy_db(self.path, [], handoffs=0)
        db = Database(self.path, auto_seed=False)
        try:
            self.assertEqual(db.get_traffic_stats()["total_sightings"], 0)
            insert_rows(db, random_rows(10, 1))
            self.assert_totals_match_scans(db)
        finally:
            db.close()

    def test_existing_active_targets_survive_the_new_column(self):
        self.make_legacy_db(self.path, [(79, 1, 5, "car", "2026-10-01T00:00:00Z")], handoffs=0)
        c = sqlite3.connect(self.path)
        sid = c.execute("SELECT sighting_id FROM vehicle_sightings").fetchone()[0]
        c.execute("INSERT INTO active_targets (global_vehicle_id, current_camera_id, last_sighting_id, locked_at_iso, status) "
                  "VALUES (5, 79, ?, '2026-10-01T00:00:00Z', 'tracking')", (sid,))
        c.commit()
        c.close()
        db = Database(self.path, auto_seed=False)
        try:
            self.assertEqual(db.get_active_target(5)["status"], "tracking")
            self.assertFalse(db.is_active_target(5), "locked in 2026-10-01 and never updated: long stale")
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
