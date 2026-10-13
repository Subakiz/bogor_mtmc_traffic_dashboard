"""
tests/test_indexing_daemon.py
Unit Test Suite for 3-Tier Adaptive Scheduling Daemon (FEAT-M5).
Tests:
- 3-Tier camera partitioning across the complete 94-camera network
- Tier 1 (15-20 FPS), Tier 2 (2 FPS), and Tier 3 (0.2 FPS) configuration
- Dynamic promotion of candidate cameras to Tier 1
- Dynamic demotion back to lower tiers
- Compute budget verification ensuring total scheduled load <= 75 FPS
- Start, stop, and status query lifecycle
- End-to-end pipeline execution: camera stepping, detection, and database persistence
"""

import os
import tempfile
import time
import unittest
import numpy as np

from backend.database import Database
from backend.detector import ANEVehicleDetector
from backend.indexing_daemon import IndexingDaemon
from tests import TOTAL_CAMERAS


class TestIndexingDaemonUnit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "test_daemon.sqlite")
        self.db = Database(self.db_path, auto_seed=True)
        self.detector = ANEVehicleDetector(test_mode=True)
        self.daemon = IndexingDaemon(
            db=self.db, detector=self.detector, test_mode=True
        )

    def tearDown(self):
        self.daemon.stop()
        self.db.close()
        self.temp_dir.cleanup()

    def test_3tier_partitioning_counts(self):
        """Verify all 94 Kota Bogor cameras are partitioned across the 3 tiers."""
        self.assertEqual(len(self.daemon.all_camera_ids), TOTAL_CAMERAS)
        c1 = self.daemon.get_cameras_in_tier(1)
        c2 = self.daemon.get_cameras_in_tier(2)
        c3 = self.daemon.get_cameras_in_tier(3)

        total_partitioned = len(c1) + len(c2) + len(c3)
        self.assertEqual(total_partitioned, TOTAL_CAMERAS)
        # Disjoint sets
        self.assertEqual(len(c1 & c2), 0)
        self.assertEqual(len(c2 & c3), 0)
        self.assertEqual(len(c1 & c3), 0)

    def test_tier_target_fps_allocations(self):
        """Verify Tier 1 has 15-20 FPS, Tier 2 has 2 FPS, Tier 3 has 0.2 FPS."""
        self.assertGreaterEqual(self.daemon.tiers[1]["target_fps"], 15.0)
        self.assertLessEqual(self.daemon.tiers[1]["target_fps"], 20.0)
        self.assertEqual(self.daemon.tiers[2]["target_fps"], 2.0)
        self.assertAlmostEqual(self.daemon.tiers[3]["target_fps"], 0.2)

    def test_dynamic_promotion_to_tier1(self):
        """Verify promoting camera elevates it to Tier 1 and removes from previous tier."""
        cam_id = 87  # Otista
        prev_tier = self.daemon.get_tier_for_camera(cam_id)
        self.assertIn(prev_tier, [2, 3])

        self.daemon.promote_to_tier1(cam_id, reason="target_exit_event")
        self.assertEqual(self.daemon.get_tier_for_camera(cam_id), 1)
        self.assertIn(cam_id, self.daemon.get_cameras_in_tier(1))
        self.assertNotIn(cam_id, self.daemon.get_cameras_in_tier(prev_tier))

    def test_dynamic_demotion_from_tier1(self):
        """Verify demoting camera restores it to Tier 2 (if hub) or Tier 3."""
        cam_id = 87  # Otista is in default hubs -> should restore to Tier 2
        self.daemon.promote_to_tier1(cam_id)
        self.assertEqual(self.daemon.get_tier_for_camera(cam_id), 1)

        self.daemon.demote_from_tier1(cam_id)
        self.assertEqual(self.daemon.get_tier_for_camera(cam_id), 2)

        # Peripheral camera demotion
        periph_cam = 75  # Gang Aut (Tier 3)
        self.daemon.promote_to_tier1(periph_cam)
        self.daemon.demote_from_tier1(periph_cam)
        self.assertEqual(self.daemon.get_tier_for_camera(periph_cam), 3)

    def test_compute_budget_stays_within_ane_limits(self):
        """Verify scheduled compute load across all 94 cameras is <= 75 FPS."""
        # Baseline load with 1 target in Tier 1
        self.daemon.promote_to_tier1(79)
        load_fps = self.daemon.compute_scheduled_load_fps()
        self.assertLess(load_fps, 75.0, f"Scheduled load {load_fps} FPS exceeds 75 FPS")

    def test_daemon_thread_lifecycle_and_status(self):
        """Verify start(), stop(), and get_status() lifecycle."""
        self.assertFalse(self.daemon.is_running())
        self.daemon.start()
        self.assertTrue(self.daemon.is_running())

        time.sleep(0.1)
        status = self.daemon.get_status()
        self.assertTrue(status["running"])
        self.assertEqual(status["total_cameras"], TOTAL_CAMERAS)
        self.assertIn(1, status["tiers"])
        self.assertIn(2, status["tiers"])
        self.assertIn(3, status["tiers"])
        self.assertGreater(status["uptime_s"], 0.0)

        self.daemon.stop()
        self.assertFalse(self.daemon.is_running())

    def test_step_camera_end_to_end_persistence(self):
        """Verify stepping camera runs detection and inserts sighting into SQLite database."""
        synthetic_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        synthetic_objs = [
            {"class": "car", "conf": 0.94, "bbox": (100, 100, 200, 200)},
            {"class": "motorcycle", "conf": 0.88, "bbox": (250, 150, 300, 220)},
        ]

        sightings = self.daemon.step_camera(
            camera_id=79,
            frame=synthetic_frame,
            synthetic_objects=synthetic_objs,
            timestamp=1000.0,
        )

        self.assertEqual(len(sightings), 2)
        for s in sightings:
            self.assertIn("sighting_id", s)
            self.assertGreater(s["sighting_id"], 0)

        # Query database to confirm persisted record
        recent = self.db.get_recent_sightings(79, window_s=60.0)
        self.assertEqual(len(recent), 2)
        classes = {r["vehicle_class"] for r in recent}
        self.assertEqual(classes, {"car", "motorcycle"})


if __name__ == "__main__":
    unittest.main()
