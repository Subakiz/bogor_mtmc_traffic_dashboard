"""
tests/test_database.py
Comprehensive Unit Test Suite for SQLite WAL Persistence Layer (F-ARCH-01).
Tests:
- Schema creation (cameras, topology_edges, vehicle_sightings, handoff_events, active_targets)
- WAL journal mode pragma and synchronous pragma
- Automatic seeding of all 94 BSW CCTV cameras and 220 topology edges
- Vehicle sightings insertion with 1024-byte embedding & 512-byte color histogram BLOBs
- Chronological journey reconstruction with camera metadata join
- Recent sightings retrieval with temporal sliding window
- Multi-parameter vehicle search
- Predictive handoff event recording
- Active target state management
- Traffic statistics aggregation
- SQL injection immunity and foreign key integrity enforcement
"""

import os
import sqlite3
import tempfile
import time
import unittest
import numpy as np

from tests import TOTAL_CAMERAS
from backend.database import (
    Database,
    get_db_connection,
    seed_cameras_from_topology,
    init_db,
    insert_vehicle_sighting,
    get_vehicle_journey,
    get_recent_sightings,
    search_vehicles,
    update_active_target,
    get_active_target,
    list_active_targets,
    record_handoff_event,
    get_traffic_stats,
)


class TestDatabaseUnit(unittest.TestCase):
    def setUp(self):
        # Create a fresh temporary SQLite file for WAL mode testing (WAL requires a file)
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_traffic.sqlite")
        self.db = Database(self.db_path, auto_seed=True)

    def tearDown(self):
        self.db.close()
        self.temp_dir.cleanup()

    def test_schema_tables_and_indices_created(self):
        """Verify all 5 required tables and performance indices are created."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        required = {"cameras", "topology_edges", "vehicle_sightings", "handoff_events", "active_targets"}
        for t in required:
            self.assertIn(t, tables, f"Missing table: {t}")

        cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indices = {row[0] for row in cur.fetchall()}
        self.assertIn("idx_sightings_cam_time", indices)
        self.assertIn("idx_sightings_global_id", indices)
        self.assertIn("idx_sightings_time", indices)
        # the covering indexes of the old COUNT(DISTINCT) scans are gone: the all-time aggregates now
        # come from trigger-maintained tables (tests/test_db_totals.py)
        self.assertNotIn("idx_sightings_agg", indices)
        self.assertNotIn("idx_sightings_local_track", indices)
        for t in ("sighting_totals", "sighting_tracks", "camera_track_totals", "global_vehicles", "sighting_counters"):
            self.assertIn(t, tables, f"Missing totals table: {t}")
        cur.execute("SELECT name FROM sqlite_master WHERE type='trigger';")
        self.assertEqual({row[0] for row in cur.fetchall()}, {"trg_sightings_totals", "trg_handoffs_total"})

    def test_wal_journal_mode_applied(self):
        """Verify SQLite WAL pragma and synchronous NORMAL are active."""
        cur = self.db.conn.cursor()
        cur.execute("PRAGMA journal_mode;")
        mode = cur.fetchone()[0]
        self.assertEqual(mode.lower(), "wal")

        cur.execute("PRAGMA synchronous;")
        sync = cur.fetchone()[0]
        # 1 corresponds to NORMAL in SQLite
        self.assertIn(sync, [1, "NORMAL", "normal"])

    def test_automatic_seeding_94_cameras_and_topology_edges(self):
        """Verify all 94 BSW CCTV cameras and topology edges are automatically seeded."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT count(*) FROM cameras;")
        cam_count = cur.fetchone()[0]
        self.assertEqual(cam_count, TOTAL_CAMERAS, f"Expected {TOTAL_CAMERAS} cameras, found {cam_count}")

        cur.execute("SELECT count(*) FROM topology_edges;")
        edge_count = cur.fetchone()[0]
        self.assertGreaterEqual(edge_count, 200, f"Expected >=200 edges, found {edge_count}")

        # Check key arterial cameras exist
        for cid in [75, 79, 83, 87, 92, 161]:
            cur.execute("SELECT id, name, corridor, latitude, longitude FROM cameras WHERE id = ?", (cid,))
            row = cur.fetchone()
            self.assertIsNotNone(row, f"Camera {cid} missing")
            self.assertNotEqual(row["name"], "")
            self.assertNotEqual(row["latitude"], 0.0)

    def test_insert_sighting_with_vector_blobs(self):
        """Verify inserting sighting stores 1024-byte embedding, 512-byte histogram, and crop thumbnail."""
        emb_bytes = np.random.randn(256).astype(np.float32).tobytes()
        hist_bytes = np.random.randn(128).astype(np.float32).tobytes()
        thumb = b"\xff\xd8\xff\xe0mock_jpeg_stream_data"

        sighting_id = self.db.insert_vehicle_sighting(
            camera_id=79,
            local_track_id=42,
            global_vehicle_id=1001,
            vehicle_class="car",
            confidence=0.94,
            bbox=(100, 150, 220, 260),
            heading_deg=245.5,
            speed_px_s=72.4,
            embedding=emb_bytes,
            color_hist=hist_bytes,
            timestamp_iso="2026-09-23T04:20:00Z",
            crop_thumbnail=thumb,
        )
        self.assertGreater(sighting_id, 0)

        cur = self.db.conn.cursor()
        cur.execute("SELECT * FROM vehicle_sightings WHERE sighting_id = ?", (sighting_id,))
        row = cur.fetchone()
        self.assertEqual(row["camera_id"], 79)
        self.assertEqual(row["local_track_id"], 42)
        self.assertEqual(row["global_vehicle_id"], 1001)
        self.assertEqual(row["vehicle_class"], "car")
        self.assertAlmostEqual(row["confidence"], 0.94)
        self.assertEqual((row["bbox_x1"], row["bbox_y1"], row["bbox_x2"], row["bbox_y2"]), (100, 150, 220, 260))
        self.assertAlmostEqual(row["heading_deg"], 245.5)
        self.assertEqual(len(row["embedding"]), 1024)
        self.assertEqual(len(row["color_hist"]), 512)
        self.assertEqual(row["crop_thumbnail"], thumb)

    def test_chronological_journey_retrieval(self):
        """Verify get_vehicle_journey returns chronological sequence with joined camera details."""
        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        # Insert sightings at 3 separate timestamps across different cameras
        self.db.insert_vehicle_sighting(79, 1, 888, "car", 0.9, (10, 10, 50, 50), 240.0, 50.0, emb, hist, "2026-09-23T05:00:00Z")
        self.db.insert_vehicle_sighting(87, 2, 888, "car", 0.92, (20, 20, 60, 60), 235.0, 55.0, emb, hist, "2026-09-23T05:01:15Z")
        self.db.insert_vehicle_sighting(83, 3, 888, "car", 0.91, (30, 30, 70, 70), 180.0, 48.0, emb, hist, "2026-09-23T05:02:30Z")

        journey = self.db.get_vehicle_journey(888)
        self.assertEqual(len(journey), 3)
        self.assertEqual([j["camera_id"] for j in journey], [79, 87, 83])
        self.assertIn("TUGU KUJANG", journey[0]["camera_name"].upper())
        self.assertIn("OTISTA", journey[1]["camera_name"].upper())
        self.assertIn("PASAR BOGOR", journey[2]["camera_name"].upper())

    def test_recent_sightings_temporal_window(self):
        """Verify get_recent_sightings filters by window_s."""
        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        # Insert sightings at 0s, 30s, and 120s
        self.db.insert_vehicle_sighting(79, 1, None, "car", 0.9, (10, 10, 50, 50), 0, 0, emb, hist, "2026-09-23T05:00:00Z")
        self.db.insert_vehicle_sighting(79, 2, None, "car", 0.9, (10, 10, 50, 50), 0, 0, emb, hist, "2026-09-23T05:01:30Z")
        self.db.insert_vehicle_sighting(79, 3, None, "car", 0.9, (10, 10, 50, 50), 0, 0, emb, hist, "2026-09-23T05:02:00Z")

        # Window of 45 seconds from latest (05:02:00Z) should return sightings at 05:02:00 and 05:01:30, but not 05:00:00
        recent = self.db.get_recent_sightings(79, window_s=45.0)
        self.assertEqual(len(recent), 2)
        local_ids = {r["local_track_id"] for r in recent}
        self.assertEqual(local_ids, {2, 3})

    def test_search_vehicles_with_filters(self):
        """Verify search_vehicles by class, camera_id, time range."""
        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        self.db.insert_vehicle_sighting(79, 1, 101, "car", 0.9, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T01:00:00Z")
        self.db.insert_vehicle_sighting(79, 2, 102, "truck", 0.9, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T01:05:00Z")
        self.db.insert_vehicle_sighting(87, 3, 103, "bus", 0.9, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T01:10:00Z")

        trucks = self.db.search_vehicles(vehicle_class="truck")
        self.assertEqual(len(trucks), 1)
        self.assertEqual(trucks[0]["vehicle_class"], "truck")

        cam79_results = self.db.search_vehicles(camera_id=79)
        self.assertEqual(len(cam79_results), 2)

    def test_active_target_upsert_and_retrieve(self):
        """Verify active target locking, state query, and listing."""
        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()
        sid = self.db.insert_vehicle_sighting(79, 1, 999, "car", 0.9, (10, 10, 50, 50), 240.0, 30.0, emb, hist, "2026-09-23T02:00:00Z")

        self.db.update_active_target(global_vehicle_id=999, current_camera_id=79, last_sighting_id=sid, status="tracking")

        target = self.db.get_active_target(999)
        self.assertIsNotNone(target)
        self.assertEqual(target["current_camera_id"], 79)
        self.assertEqual(target["status"], "tracking")

        active_list = self.db.list_active_targets()
        self.assertEqual(len(active_list), 1)
        self.assertEqual(active_list[0]["global_vehicle_id"], 999)

    def test_handoff_event_and_referential_integrity(self):
        """Verify handoff events recording and foreign key enforcement."""
        eid = self.db.record_handoff_event(
            global_vehicle_id=999,
            from_camera_id=79,
            to_camera_id=87,
            confidence=0.91,
            timestamp_iso="2026-09-23T02:01:30Z",
        )
        self.assertGreater(eid, 0)

        # Non-existent camera violates foreign key constraint
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.record_handoff_event(
                global_vehicle_id=999,
                from_camera_id=999999,
                to_camera_id=87,
                confidence=0.9,
                timestamp_iso="2026-09-23T02:01:30Z",
            )

    def test_traffic_stats_aggregation(self):
        """Verify get_traffic_stats returns correct aggregated metrics."""
        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        self.db.insert_vehicle_sighting(79, 1, 501, "car", 0.9, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T03:00:00Z")
        self.db.insert_vehicle_sighting(79, 2, 502, "motorcycle", 0.85, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T03:00:05Z")
        self.db.insert_vehicle_sighting(87, 3, 501, "car", 0.92, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T03:01:20Z")

        stats = self.db.get_traffic_stats()
        self.assertEqual(stats["total_sightings"], 3)
        self.assertEqual(stats["unique_global_vehicles"], 2)
        self.assertEqual(stats["class_breakdown"]["car"], 2)
        self.assertEqual(stats["class_breakdown"]["motorcycle"], 1)

    def test_sql_injection_safety(self):
        """Verify parameterized queries handle SQL injection strings safely."""
        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()
        payload = "'; DROP TABLE cameras; --"
        sid = self.db.insert_vehicle_sighting(79, 1, None, payload, 0.9, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T00:00:00Z")
        self.assertGreater(sid, 0)

        # Confirm cameras table still exists and has 94 rows
        cur = self.db.conn.cursor()
        cur.execute("SELECT count(*) FROM cameras;")
        self.assertEqual(cur.fetchone()[0], TOTAL_CAMERAS)

    def test_get_camera_vehicle_counts_all_94_cameras(self):
        """Verify get_camera_vehicle_counts accounts for all 94 cameras with accurate metrics."""
        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        # Insert sightings for camera 79 and 161
        self.db.insert_vehicle_sighting(79, 1, 101, "car", 0.9, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T04:00:00Z")
        self.db.insert_vehicle_sighting(79, 2, 102, "motorcycle", 0.85, (0, 0, 10, 10), 0, 0, emb, hist, "2026-09-23T04:00:05Z")
        self.db.insert_vehicle_sighting(79, 1, 101, "car", 0.91, (5, 5, 15, 15), 0, 0, emb, hist, "2026-09-23T04:00:10Z")
        self.db.insert_vehicle_sighting(161, 3, 103, "truck", 0.88, (0, 0, 20, 20), 0, 0, emb, hist, "2026-09-23T04:01:00Z")

        counts = self.db.get_camera_vehicle_counts()
        self.assertEqual(len(counts), TOTAL_CAMERAS, "Must return count records for all 94 cameras")

        # Find camera 79
        cam79 = next(c for c in counts if c["camera_id"] == 79)
        self.assertEqual(cam79["total_sightings"], 3)
        self.assertEqual(cam79["unique_vehicles"], 2)  # track 1 and track 2
        self.assertEqual(cam79["classes"]["car"], 2)
        self.assertEqual(cam79["classes"]["motorcycle"], 1)
        self.assertEqual(cam79["classes"]["truck"], 0)

        # Find camera 161
        cam161 = next(c for c in counts if c["camera_id"] == 161)
        self.assertEqual(cam161["total_sightings"], 1)
        self.assertEqual(cam161["unique_vehicles"], 1)
        self.assertEqual(cam161["classes"]["truck"], 1)

        # Verify cameras without sightings have 0 counts
        cam80 = next(c for c in counts if c["camera_id"] == 80)
        self.assertEqual(cam80["total_sightings"], 0)
        self.assertEqual(cam80["unique_vehicles"], 0)
        self.assertIsNone(cam80["last_seen_iso"])

        # Check traffic stats integration
        stats = self.db.get_traffic_stats()
        self.assertEqual(stats["total_sightings"], 4)
        self.assertEqual(stats["cameras_with_detections"], 2)
        self.assertEqual(stats["camera_counts"][79], 3)
        self.assertEqual(stats["camera_counts"][161], 1)
        self.assertEqual(stats["camera_counts"][80], 0)


class TestThreadConnections(unittest.TestCase):
    def test_connection_of_a_finished_thread_is_closed(self):
        # Regression: every request-pool thread's connection was kept open forever (fd leak).
        import gc
        import threading
        from backend.database import Database

        with tempfile.TemporaryDirectory() as tmp:
            db = Database(os.path.join(tmp, "t.sqlite"), auto_seed=False)
            conns = []

            def work():
                conn = db.conn
                conn.execute("SELECT 1").fetchone()
                conns.append(conn)

            threads = [threading.Thread(target=work) for _ in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            gc.collect()
            self.assertEqual(len(db._all_conns), 0)
            for conn in conns:
                with self.assertRaises(sqlite3.ProgrammingError):   # closed
                    conn.execute("SELECT 1")
            db.close()


if __name__ == "__main__":
    unittest.main()
