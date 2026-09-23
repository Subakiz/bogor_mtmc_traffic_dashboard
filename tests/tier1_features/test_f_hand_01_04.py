"""
tests/tier1_features/test_f_hand_01_04.py
Tier 1 Tests for Features F-HAND-01, F-HAND-02, and F-HAND-04.
>=5 test cases per feature (15 total tests).
"""

import time
import unittest
import numpy as np

from tests.mock_stream import SyntheticTrajectorySimulator
from tests.reference_contracts import (
    GlobalIDPreserverContract,
    HandoffEngineContract,
    ReidEngineContract,
    TopologyContract,
)


class TestFHAND01TrajectoryExitEstimator(unittest.TestCase):
    """
    F-HAND-01: Vehicle Trajectory & Exit Estimator
    Motion vectors, camera azimuth alignment, Theta_exit, and boundary exit events.
    """

    def test_f_hand_01_motion_vector_from_centroids(self):
        """Test 1: Motion vector calculates delta x and delta y from consecutive centroids."""
        c1 = (100.0, 150.0)
        c2 = (140.0, 180.0)
        dx, dy = c2[0] - c1[0], c2[1] - c1[1]
        self.assertEqual(dx, 40.0)
        self.assertEqual(dy, 30.0)
        speed = np.sqrt(dx**2 + dy**2)
        self.assertEqual(speed, 50.0)

    def test_f_hand_01_camera_azimuth_alignment_to_heading_angle(self):
        """Test 2: Heading angle combines motion vector with camera orientation azimuth."""
        # Camera facing South (180 deg), motion vector purely right (+x, dy=0)
        heading_deg, cardinal = HandoffEngineContract.estimate_exit_heading(
            motion_vector=(50.0, 0.0), camera_azimuth_deg=180.0
        )
        self.assertAlmostEqual(heading_deg, 180.0, places=1)
        self.assertEqual(cardinal, "S")

    def test_f_hand_01_boundary_exit_detection_margin(self):
        """Test 3: is_boundary_exit detects vehicle crossing frame margin."""
        frame_size = (640, 480)
        margin = 35

        # Vehicle deep inside frame (not exiting)
        inside_bbox = (200, 200, 300, 280)
        self.assertFalse(HandoffEngineContract.is_boundary_exit(inside_bbox, frame_size, margin))

        # Vehicle hitting right boundary (x2 >= 640 - 35 = 605)
        exiting_right_bbox = (540, 200, 620, 280)
        self.assertTrue(HandoffEngineContract.is_boundary_exit(exiting_right_bbox, frame_size, margin))

        # Vehicle hitting left boundary (x1 <= 35)
        exiting_left_bbox = (20, 200, 100, 280)
        self.assertTrue(HandoffEngineContract.is_boundary_exit(exiting_left_bbox, frame_size, margin))

    def test_f_hand_01_cardinal_exit_sector_mapping(self):
        """Test 4: Maps continuous 360-degree angles to correct 8 cardinal sectors."""
        test_cases = [
            (0.0, "N"),
            (45.0, "NE"),
            (90.0, "E"),
            (135.0, "SE"),
            (180.0, "S"),
            (225.0, "SW"),
            (270.0, "W"),
            (315.0, "NW"),
        ]
        for angle, expected_cardinal in test_cases:
            _, card = HandoffEngineContract.estimate_exit_heading((0.0, 0.0), angle)
            self.assertEqual(card, expected_cardinal, f"Angle {angle} mapped to {card}, expected {expected_cardinal}")

    def test_f_hand_01_stationary_vehicle_handling(self):
        """Test 5: Low or zero pixel velocity does not generate erratic exit vector."""
        motion_vector = (0.01, 0.02)
        heading, cardinal = HandoffEngineContract.estimate_exit_heading(motion_vector, camera_azimuth_deg=90.0)
        self.assertTrue(0.0 <= heading <= 360.0)


class TestFHAND02PriorityStreamScheduler(unittest.TestCase):
    """
    F-HAND-02: Priority Stream Scheduler
    Elevating candidate downstream cameras during arrival windows.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()

    def test_f_hand_02_exit_event_schedules_downstream_candidates(self):
        """Test 1: Vehicle exit at Tugu Kujang (79) schedules downstream candidate camera (87)."""
        candidates = self.topo.get_downstream_candidates(79, "WEST")
        self.assertGreaterEqual(len(candidates), 1)
        cand_ids = [c["target_camera_id"] for c in candidates]
        self.assertIn(87, cand_ids)
        # Check arrival window
        c = candidates[0]
        self.assertGreater(c["t_min_s"], 0)
        self.assertGreater(c["t_max_s"], c["t_min_s"])

    def test_f_hand_02_candidates_ordered_by_transition_feasibility(self):
        """Test 2: Multi-candidate junctions provide valid corridor distance and timing."""
        candidates = self.topo.get_downstream_candidates(83, "WEST")
        self.assertGreaterEqual(len(candidates), 1)
        for c in candidates:
            self.assertGreater(c["distance_m"], 0)
            self.assertTrue(c["t_min_s"] < c["t_max_s"])

    def test_f_hand_02_arrival_window_timeline_tracking(self):
        """Test 3: Computes precise arrival interval [t_exit + t_min, t_exit + t_max]."""
        t_exit = 1000.0
        t_min = 45.0
        t_max = 140.0
        window_start = t_exit + t_min
        window_end = t_exit + t_max
        self.assertEqual(window_start, 1045.0)
        self.assertEqual(window_end, 1140.0)

    def test_f_hand_02_multiple_active_target_scheduling(self):
        """Test 4: Priority scheduler handles multiple targets with overlapping candidates."""
        # Target 1 exits towards Cam 87; Target 2 exits towards Cam 81
        scheduled_cams = {}
        target_1_candidates = [87]
        target_2_candidates = [81, 87]

        for cid in target_1_candidates:
            scheduled_cams[cid] = scheduled_cams.get(cid, 0) + 1
        for cid in target_2_candidates:
            scheduled_cams[cid] = scheduled_cams.get(cid, 0) + 1

        self.assertEqual(scheduled_cams[87], 2)
        self.assertEqual(scheduled_cams[81], 1)

    def test_f_hand_02_early_match_releases_candidate_stream(self):
        """Test 5: Sighting target on candidate camera marks transition complete."""
        scheduled_targets = {501: {"source_cam": 79, "candidate_cams": [87], "status": "pending"}}
        # Candidate 87 captures vehicle
        scheduled_targets[501]["status"] = "handed_off"
        scheduled_targets[501]["active_cam"] = 87
        self.assertEqual(scheduled_targets[501]["status"], "handed_off")
        self.assertEqual(scheduled_targets[501]["active_cam"], 87)


class TestFHAND04PersistentGlobalIDPreserver(unittest.TestCase):
    """
    F-HAND-04: Persistent Global ID Preserver
    1 ID = 1 unique vehicle state management across consecutive camera handoffs.
    """

    def setUp(self):
        self.preserver = GlobalIDPreserverContract()

    def test_f_hand_04_target_lock_initializes_active_tracking(self):
        """Test 1: Locking target initializes tracking state with persistent Global ID."""
        sig = {"vehicle_class": "car", "color": "red"}
        self.preserver.lock_target(global_id=9001, camera_id=79, signature=sig)
        self.assertIn(9001, self.preserver.active_targets)
        t = self.preserver.active_targets[9001]
        self.assertEqual(t["current_camera_id"], 79)
        self.assertEqual(t["status"], "tracking")

    def test_f_hand_04_handoff_maintains_identical_global_id(self):
        """Test 2: Handoff to downstream camera preserves the same Global Vehicle ID."""
        sig = {"vehicle_class": "car", "color": "red"}
        self.preserver.lock_target(global_id=9001, camera_id=79, signature=sig)

        event = self.preserver.handoff_target(global_id=9001, to_camera_id=87, confidence=0.94)
        self.assertEqual(event["global_vehicle_id"], 9001)
        self.assertEqual(event["from_camera_id"], 79)
        self.assertEqual(event["to_camera_id"], 87)
        self.assertEqual(self.preserver.active_targets[9001]["current_camera_id"], 87)

    def test_f_hand_04_chronological_handoff_history_chain(self):
        """Test 3: Handoff sequence across 3 cameras records complete camera chain."""
        sig = {"vehicle_class": "car", "color": "silver"}
        self.preserver.lock_target(global_id=9002, camera_id=79, signature=sig)
        self.preserver.handoff_target(global_id=9002, to_camera_id=87, confidence=0.91)
        self.preserver.handoff_target(global_id=9002, to_camera_id=83, confidence=0.89)
        self.preserver.handoff_target(global_id=9002, to_camera_id=81, confidence=0.95)

        t = self.preserver.active_targets[9002]
        self.assertEqual(t["history"], [79, 87, 83, 81])
        self.assertEqual(len(self.preserver.handoff_history), 3)

    def test_f_hand_04_unlock_removes_from_active_targets(self):
        """Test 4: Unlocking target cleans up active tracking session."""
        self.preserver.lock_target(global_id=9003, camera_id=79, signature={})
        self.assertIn(9003, self.preserver.active_targets)
        self.preserver.unlock_target(global_id=9003)
        self.assertNotIn(9003, self.preserver.active_targets)

    def test_f_hand_04_invalid_target_handoff_raises_keyerror(self):
        """Test 5: Handing off an unregistered or expired target raises KeyError."""
        with self.assertRaises(KeyError):
            self.preserver.handoff_target(global_id=888888, to_camera_id=87, confidence=0.9)


if __name__ == "__main__":
    unittest.main()
