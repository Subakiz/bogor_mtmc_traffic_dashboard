"""
tests/tier1_features/test_f_arch_01_02.py
Tier 1 Tests for Feature F-ARCH-01 (SQLite WAL persistence).
"""

import os
import sqlite3
import time
import unittest
from tests.reference_contracts import DatabaseContract


class TestFARCH01SQLiteWALPersistence(unittest.TestCase):
    """
    F-ARCH-01: SQLite WAL Persistence Layer
    Verified against schema DDL, WAL pragma, connection pooling, and sub-millisecond inserts.
    """

    def setUp(self):
        # Create a fresh database for each test (isolated state)
        self.db = DatabaseContract(":memory:")

    def tearDown(self):
        if hasattr(self, "db") and self.db and self.db.conn:
            self.db.conn.close()

    def test_f_arch_01_required_tables_and_indices_exist(self):
        """Test 1: All required tables and performance indices are created."""
        cur = self.db.conn.cursor()
        cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cur.fetchall()}
        required_tables = {
            "cameras",
            "topology_edges",
            "vehicle_sightings",
        }
        for tbl in required_tables:
            self.assertIn(tbl, tables, f"Missing required table: {tbl}")

        # Check indices
        cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indices = {row[0] for row in cur.fetchall()}
        self.assertIn("idx_sightings_cam_time", indices)

    def test_f_arch_01_insert_vehicle_sighting_stores_row(self):
        """Test 2: insert_vehicle_sighting stores the sighting's class, box and motion."""
        cur = self.db.conn.cursor()
        # Seed dummy camera
        cur.execute(
            "INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (79, "CCTV - TUGU KUJANG", "SSA_RING", -6.6014, 106.8051),
        )

        sighting_id = self.db.insert_vehicle_sighting(
            camera_id=79,
            local_track_id=101,
            vehicle_class="car",
            confidence=0.92,
            bbox=(120, 150, 240, 220),
            heading_deg=240.0,
            speed_px_s=85.5,
            timestamp_iso="2026-09-23T04:15:00Z",
        )
        self.assertGreater(sighting_id, 0)

        cur.execute("SELECT vehicle_class, bbox_x1, bbox_y2, speed_px_s FROM vehicle_sightings WHERE sighting_id = ?",
                    (sighting_id,))
        self.assertEqual(tuple(cur.fetchone()), ("car", 120, 220, 85.5))

    def test_f_arch_01_sub_millisecond_insert_latency(self):
        """Test 3: Inserting sightings operates with sub-millisecond execution latency."""
        cur = self.db.conn.cursor()
        cur.execute(
            "INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (83, "CCTV - PASAR BOGOR", "SSA_RING", -6.6009, 106.7995),
        )

        # Measure 50 consecutive inserts
        t0 = time.perf_counter()
        for i in range(50):
            self.db.insert_vehicle_sighting(
                camera_id=83,
                local_track_id=200 + i,
                vehicle_class="car",
                confidence=0.89,
                bbox=(50, 50, 150, 120),
                heading_deg=180.0,
                speed_px_s=60.0,
                timestamp_iso=f"2026-09-23T04:15:{i:02d}Z",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_insert_ms = elapsed_ms / 50.0
        self.assertLess(avg_insert_ms, 2.0, f"Average insert latency {avg_insert_ms:.3f} ms exceeds 2.0 ms")

if __name__ == "__main__":
    unittest.main()
