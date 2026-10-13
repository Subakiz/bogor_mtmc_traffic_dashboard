"""
Trigger-maintained all-time totals (backend/database.py: TOTALS_*): they must equal the old
COUNT / COUNT(DISTINCT) scans over vehicle_sightings, survive the one-off migration of a database
that predates them (or still carries the re-identification schema), and keep counting after old
sightings are deleted.
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
        "tracks": one("SELECT COUNT(DISTINCT local_track_id) FROM vehicle_sightings"),
        "classes": dict(conn.execute("SELECT vehicle_class, COUNT(*) FROM vehicle_sightings GROUP BY 1").fetchall()),
        "per_cam": per_cam,
    }


def random_rows(n, seed):
    rng = random.Random(seed)
    rows = []
    for i in range(n):
        cam = rng.choice(CAMERAS)
        # small id ranges so tracks repeat on a camera and across cameras, like the live data
        track = rng.randint(1, max(2, n // 3))
        # timestamps deliberately not in insertion order
        ts = f"2026-10-0{rng.randint(1, 9)}T{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}:{rng.randint(0, 59):02d}Z"
        rows.append((cam, track, rng.choice(CLASSES), ts))
    return rows


def insert_rows(db, rows):
    for cam, track, cls, ts in rows:
        db.insert_vehicle_sighting(cam, track, cls, 0.9, (0, 0, 10, 10), 0.0, 0.0, ts)


def raw_insert_rows(conn, rows):
    conn.executemany(
        "INSERT INTO vehicle_sightings (camera_id, local_track_id, vehicle_class, confidence, "
        "bbox_x1, bbox_y1, bbox_x2, bbox_y2, heading_deg, speed_px_s, timestamp_iso) "
        "VALUES (?, ?, ?, 0.9, 0, 0, 10, 10, 0, 0, ?)", rows)


class TotalsMixin:
    def assert_totals_match_scans(self, db):
        ref = reference_stats(db.conn)
        stats = db.get_traffic_stats()
        self.assertEqual(stats["total_sightings"], ref["total"])
        self.assertEqual(stats["total_vehicles_counted"], ref["total"])
        self.assertEqual(stats["total_unique_vehicles"], ref["tracks"])
        self.assertEqual(stats["class_breakdown"], ref["classes"])
        counts = {c["camera_id"]: c for c in db.get_camera_vehicle_counts()}
        self.assertEqual(set(ref["per_cam"]), {cid for cid, c in counts.items() if c["total_sightings"]})
        for cid, (n, tracks, last, cars, motos, trucks, buses) in ref["per_cam"].items():
            c = counts[cid]
            self.assertEqual((c["total_sightings"], c["unique_vehicles"], c["last_seen_iso"]), (n, tracks, last), cid)
            self.assertEqual(c["classes"], {"car": cars, "motorcycle": motos, "truck": trucks, "bus": buses}, cid)
        self.assertEqual([c["total_sightings"] for c in db.get_camera_vehicle_counts()],
                         sorted((c["total_sightings"] for c in counts.values()), reverse=True))

class TriggerTotalsTest(TotalsMixin, unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "t.sqlite"), auto_seed=True)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def test_empty_database_reads_zero(self):
        stats = self.db.get_traffic_stats()
        self.assertEqual((stats["total_sightings"], stats["total_unique_vehicles"]), (0, 0))
        self.assertEqual(stats["class_breakdown"], {})
        self.assertTrue(all(c["total_sightings"] == 0 for c in self.db.get_camera_vehicle_counts()))

    def test_totals_equal_full_scans_on_random_data(self):
        for seed in (1, 2, 3):
            insert_rows(self.db, random_rows(400, seed))
            self.assert_totals_match_scans(self.db)
    
    def test_failed_insert_does_not_move_counters(self):
        insert_rows(self.db, random_rows(20, 9))
        before = self.db.get_traffic_stats()
        with self.assertRaises(sqlite3.IntegrityError):   # unknown camera: FK violation, trigger rolled back too
            self.db.insert_vehicle_sighting(999999, 1, "car", 0.9, (0, 0, 1, 1), 0, 0, "2026-10-01T00:00:00Z")
        self.assertEqual(self.db.get_traffic_stats(), before)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM sighting_tracks").fetchone()[0],
                         self.db.conn.execute("SELECT COUNT(*) FROM (SELECT DISTINCT local_track_id, camera_id FROM vehicle_sightings)").fetchone()[0])

    def test_track_ids_repeat_across_cameras_like_the_live_data(self):
        for cam in (79, 87):
            for _ in range(2):
                self.db.insert_vehicle_sighting(cam, 7, "car", 0.9, (0, 0, 1, 1), 0, 0, "2026-10-01T00:00:00Z")
        stats = self.db.get_traffic_stats()
        self.assertEqual(stats["total_sightings"], 4)
        self.assertEqual(stats["total_unique_vehicles"], 1)             # COUNT(DISTINCT local_track_id) over everything
        counts = {c["camera_id"]: c for c in self.db.get_camera_vehicle_counts()}
        self.assertEqual((counts[79]["unique_vehicles"], counts[87]["unique_vehicles"]), (1, 1))
        self.assert_totals_match_scans(self.db)

    def test_concurrent_writers_keep_totals_exact(self):
        def work(offset):
            insert_rows(self.db, [(CAMERAS[i % len(CAMERAS)], offset + i, CLASSES[i % len(CLASSES)],
                                   "2026-10-01T00:00:%02dZ" % (i % 60)) for i in range(150)])
        threads = [threading.Thread(target=work, args=(k * 1000,)) for k in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(self.db.get_traffic_stats()["total_sightings"], 600)
        self.assert_totals_match_scans(self.db)

    def test_rebuild_totals_repairs_drifted_counters(self):
        insert_rows(self.db, random_rows(300, 4))
        with self.db.conn:
            self.db.conn.execute("UPDATE sighting_counters SET value = value + 17")
            self.db.conn.execute("DELETE FROM sighting_totals WHERE camera_id = 79")
        self.db.rebuild_totals()
        self.assert_totals_match_scans(self.db)

    def test_windowed_counts_still_scan_the_requested_window(self):
        from datetime import datetime, timedelta, timezone
        now = datetime.now(timezone.utc)
        iso = lambda s: (now - timedelta(seconds=s)).strftime("%Y-%m-%dT%H:%M:%SZ")
        insert_rows(self.db, [(79, 1, "car", iso(30)), (79, 2, "truck", iso(60)), (79, 3, "car", iso(7200))])
        recent = {c["camera_id"]: c for c in self.db.get_camera_vehicle_counts(window_s=300)}
        self.assertEqual(recent[79]["total_sightings"], 2)
        self.assertEqual(recent[79]["classes"]["truck"], 1)
        self.assertEqual({c["camera_id"]: c for c in self.db.get_camera_vehicle_counts()}[79]["total_sightings"], 3)


class MigrationTest(TotalsMixin, unittest.TestCase):
    """A database created before the totals existed is converted once, on open."""

    LEGACY_INDEXES = ("idx_sightings_agg", "idx_sightings_local_track")

    def make_legacy_db(self, path, rows):
        Database(path, auto_seed=True).close()
        c = sqlite3.connect(path)
        c.execute("DROP TRIGGER trg_sightings_totals")
        for tbl in ("sighting_totals", "sighting_tracks", "camera_track_totals", "sighting_counters"):
            c.execute(f"DROP TABLE {tbl}")
        c.execute("CREATE INDEX idx_sightings_agg ON vehicle_sightings(camera_id, local_track_id, vehicle_class, timestamp_iso)")
        c.execute("CREATE INDEX idx_sightings_local_track ON vehicle_sightings(local_track_id)")
        c.execute("DELETE FROM maintenance_state WHERE key = ?", (TOTALS_MARKER,))
        raw_insert_rows(c, rows)
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
            self.assert_totals_match_scans(db)
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
            self.assertEqual(db.conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type = 'trigger'").fetchone()[0], 1)
        finally:
            db.close()

    def test_migrating_an_empty_legacy_database(self):
        self.make_legacy_db(self.path, [])
        db = Database(self.path, auto_seed=False)
        try:
            self.assertEqual(db.get_traffic_stats()["total_sightings"], 0)
            insert_rows(db, random_rows(10, 1))
            self.assert_totals_match_scans(db)
        finally:
            db.close()


# The re-identification schema as databases written before its removal still carry it: extra
# sighting columns, three tables, and a totals trigger that reads global_vehicle_id.
REID_ERA_DDL = """
ALTER TABLE vehicle_sightings ADD COLUMN global_vehicle_id INTEGER;
ALTER TABLE vehicle_sightings ADD COLUMN embedding BLOB NOT NULL DEFAULT X'';
ALTER TABLE vehicle_sightings ADD COLUMN color_hist BLOB NOT NULL DEFAULT X'';
ALTER TABLE vehicle_sightings ADD COLUMN crop_thumbnail BLOB;
CREATE INDEX idx_sightings_global_id ON vehicle_sightings(global_vehicle_id);
CREATE TABLE handoff_events (event_id INTEGER PRIMARY KEY AUTOINCREMENT, global_vehicle_id INTEGER NOT NULL,
    from_camera_id INTEGER NOT NULL, to_camera_id INTEGER NOT NULL, confidence REAL NOT NULL,
    timestamp_iso TEXT NOT NULL);
CREATE TABLE active_targets (global_vehicle_id INTEGER PRIMARY KEY, current_camera_id INTEGER NOT NULL,
    last_sighting_id INTEGER NOT NULL, locked_at_iso TEXT NOT NULL, status TEXT DEFAULT 'tracking',
    updated_ts REAL, FOREIGN KEY (last_sighting_id) REFERENCES vehicle_sightings(sighting_id));
CREATE INDEX idx_active_targets_cam ON active_targets(current_camera_id);
CREATE TABLE global_vehicles (global_vehicle_id INTEGER PRIMARY KEY, first_seen_iso TEXT NOT NULL,
    first_camera_id INTEGER NOT NULL, last_seen_iso TEXT NOT NULL, last_camera_id INTEGER NOT NULL,
    sightings INTEGER NOT NULL);
INSERT INTO sighting_counters (key, value) VALUES ('global_vehicles', 0), ('handoffs', 0);
INSERT INTO maintenance_state (key, value) VALUES ('embeddings_expired_upto', 0);
DROP TRIGGER trg_sightings_totals;
CREATE TRIGGER trg_sightings_totals AFTER INSERT ON vehicle_sightings
BEGIN
    INSERT INTO sighting_totals (camera_id, vehicle_class, n, last_seen_iso)
    VALUES (NEW.camera_id, NEW.vehicle_class, 1, NEW.timestamp_iso)
    ON CONFLICT(camera_id, vehicle_class) DO UPDATE SET
        n = n + 1, last_seen_iso = max(last_seen_iso, excluded.last_seen_iso);
    UPDATE sighting_counters SET value = value + 1
    WHERE key = 'tracks'
      AND NOT EXISTS (SELECT 1 FROM sighting_tracks WHERE local_track_id = NEW.local_track_id);
    INSERT INTO camera_track_totals (camera_id, tracks)
    SELECT NEW.camera_id, 1
    WHERE NOT EXISTS (SELECT 1 FROM sighting_tracks
                      WHERE local_track_id = NEW.local_track_id AND camera_id = NEW.camera_id)
    ON CONFLICT(camera_id) DO UPDATE SET tracks = tracks + 1;
    INSERT OR IGNORE INTO sighting_tracks (local_track_id, camera_id) VALUES (NEW.local_track_id, NEW.camera_id);
    INSERT INTO global_vehicles (global_vehicle_id, first_seen_iso, first_camera_id, last_seen_iso,
                                 last_camera_id, sightings)
    SELECT NEW.global_vehicle_id, NEW.timestamp_iso, NEW.camera_id, NEW.timestamp_iso, NEW.camera_id, 1
    WHERE NEW.global_vehicle_id IS NOT NULL
    ON CONFLICT(global_vehicle_id) DO UPDATE SET sightings = sightings + 1;
END;
CREATE TRIGGER trg_handoffs_total AFTER INSERT ON handoff_events
BEGIN
    UPDATE sighting_counters SET value = value + 1 WHERE key = 'handoffs';
END;
"""


class ReidSchemaRemovalTest(TotalsMixin, unittest.TestCase):
    """A database that still has the re-identification schema (with the totals already migrated)
    loses it on open, keeps its sightings and totals, and keeps counting new sightings."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "reid.sqlite")
        db = Database(self.path, auto_seed=True)
        insert_rows(db, random_rows(300, 21))
        db.close()
        c = sqlite3.connect(self.path)
        c.executescript(REID_ERA_DDL)
        c.execute("INSERT INTO handoff_events (global_vehicle_id, from_camera_id, to_camera_id, confidence, "
                  "timestamp_iso) VALUES (1, 79, 87, 0.9, '2026-10-01T00:00:00Z')")
        c.commit()
        c.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_reid_tables_columns_and_counters_are_dropped(self):
        db = Database(self.path, auto_seed=False)
        try:
            self.assertTrue(db.migration_report["reid_schema_dropped"])
            self.assertFalse(db.migration_report["totals_backfilled"], "totals were already migrated")
            names = {r[0] for r in db.conn.execute("SELECT name FROM sqlite_master")}
            self.assertFalse(names & {"handoff_events", "active_targets", "global_vehicles", "trg_handoffs_total",
                                      "idx_sightings_global_id", "idx_active_targets_cam"})
            self.assertIn("trg_sightings_totals", names)
            columns = {r[1] for r in db.conn.execute("PRAGMA table_info(vehicle_sightings)")}
            self.assertFalse(columns & {"global_vehicle_id", "embedding", "color_hist", "crop_thumbnail"})
            self.assertEqual([r[0] for r in db.conn.execute("SELECT key FROM sighting_counters")], ["tracks"])
            self.assertIsNone(db.conn.execute(
                "SELECT 1 FROM maintenance_state WHERE key = 'embeddings_expired_upto'").fetchone())
            self.assertEqual(db.get_traffic_stats()["total_sightings"], 300)
            self.assert_totals_match_scans(db)
            insert_rows(db, random_rows(100, 22))      # the recreated trigger is live
            self.assertEqual(db.get_traffic_stats()["total_sightings"], 400)
            self.assert_totals_match_scans(db)
        finally:
            db.close()

    def test_reopening_after_the_removal_changes_nothing(self):
        Database(self.path, auto_seed=False).close()
        db = Database(self.path, auto_seed=False)
        try:
            self.assertFalse(db.migration_report["reid_schema_dropped"])
            self.assert_totals_match_scans(db)
        finally:
            db.close()


if __name__ == "__main__":
    unittest.main()
