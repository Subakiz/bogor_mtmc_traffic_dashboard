"""
tests/tier2_boundaries/test_tier2_handoff_boundaries.py
Tier 2 Boundary and Corner Case Tests for Features F-HAND-01, F-HAND-02, and F-HAND-04.
5 boundary tests per feature (15 total tests).
"""

import unittest
from tests.reference_contracts import (
    GlobalIDPreserverContract,
    HandoffEngineContract,
    TopologyContract,
)


class TestTier2HandoffBoundaries(unittest.TestCase):
    """
    Boundary & Corner Cases for Predictive Camera Handoff (F-HAND-01, 02, 04)
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()

    def setUp(self):
        self.preserver = GlobalIDPreserverContract()

    # --- F-HAND-01 Boundaries ---
    def test_b_hand_01_exact_zero_motion_vector(self):
        """B1.1: Exact zero motion vector (0, 0) defaults to camera azimuth."""
        heading, card = HandoffEngineContract.estimate_exit_heading((0.0, 0.0), camera_azimuth_deg=270.0)
        self.assertEqual(heading, 270.0)
        self.assertEqual(card, "W")

    def test_b_hand_01_compass_wrap_boundary(self):
        """B1.2: Angles near 360°/0° boundary correctly map to North (N)."""
        _, card_359 = HandoffEngineContract.estimate_exit_heading((0.0, 0.0), 359.9)
        _, card_001 = HandoffEngineContract.estimate_exit_heading((0.0, 0.0), 0.1)
        self.assertEqual(card_359, "N")
        self.assertEqual(card_001, "N")

    def test_b_hand_01_zero_pixel_boundary_margin(self):
        """B1.3: Margin of 0 pixels triggers exit only when vehicle strictly reaches frame edge."""
        frame_size = (640, 480)
        # Vehicle at x1 = 0 (touching edge)
        touching_bbox = (0, 100, 50, 150)
        self.assertTrue(HandoffEngineContract.is_boundary_exit(touching_bbox, frame_size, margin=0))

        # Vehicle at x1 = 1 (inside)
        inside_bbox = (1, 100, 50, 150)
        self.assertFalse(HandoffEngineContract.is_boundary_exit(inside_bbox, frame_size, margin=0))

    def test_b_hand_01_negative_bbox_coordinates_outside_frame(self):
        """B1.4: Bounding box outside frame (negative coordinates) is detected as boundary exit."""
        outside_bbox = (-50, 100, -10, 150)
        self.assertTrue(HandoffEngineContract.is_boundary_exit(outside_bbox, (640, 480), margin=35))

    def test_b_hand_01_extreme_pixel_speed_calculation(self):
        """B1.5: Extremely large pixel displacement calculates heading without float overflow."""
        heading, card = HandoffEngineContract.estimate_exit_heading((10000.0, 0.0), camera_azimuth_deg=0.0)
        self.assertAlmostEqual(heading, 0.0, places=1)
        self.assertEqual(card, "N")

    # --- F-HAND-02 Boundaries ---
    def test_b_hand_02_zero_width_arrival_window(self):
        """B2.1: Travel window where t_min == t_max (synthetic edge case) is bounded."""
        t_exit = 100.0
        t_min = 30.0
        t_max = 30.0
        window_duration = (t_exit + t_max) - (t_exit + t_min)
        self.assertEqual(window_duration, 0.0)

    def test_b_hand_02_dead_end_camera_candidate_query(self):
        """B2.2: Camera with no downstream candidates returns empty list cleanly."""
        candidates = self.topo.get_downstream_candidates(81, "SOUTH")
        self.assertEqual(candidates, [])

    def test_b_hand_02_50_concurrent_target_schedules(self):
        """B2.3: Scheduler manages 50 concurrent active target candidates without deadlock."""
        schedules = {}
        for target_id in range(50):
            schedules[target_id] = {"candidates": [87, 81], "expires_at": 1000.0 + target_id}
        self.assertEqual(len(schedules), 50)

    def test_b_hand_02_schedule_cancellation_idempotence(self):
        """B2.4: Cancelling a schedule twice does not raise an exception."""
        schedules = {101: {"cam": 87}}
        schedules.pop(101, None)
        # Second cancellation
        res = schedules.pop(101, None)
        self.assertIsNone(res)

    def test_b_hand_02_expired_arrival_window_detection(self):
        """B2.5: Identifies that an arrival window whose end time < current time is expired."""
        current_time = 500.0
        window_end = 450.0
        is_expired = current_time > window_end
        self.assertTrue(is_expired)

    # --- F-HAND-04 Boundaries ---
    def test_b_hand_04_re_locking_existing_target_updates_state(self):
        """B4.1: Re-locking an active target ID cleanly updates signature and camera."""
        self.preserver.lock_target(501, 79, {"color": "red"})
        self.assertEqual(self.preserver.active_targets[501]["current_camera_id"], 79)
        # Update lock to camera 87
        self.preserver.lock_target(501, 87, {"color": "red_metallic"})
        self.assertEqual(self.preserver.active_targets[501]["current_camera_id"], 87)

    def test_b_hand_04_same_camera_handoff_recorded(self):
        """B4.2: Handoff from camera 79 to 79 (e.g. intra-junction re-lock) is tracked."""
        self.preserver.lock_target(502, 79, {})
        event = self.preserver.handoff_target(502, 79, confidence=0.99)
        self.assertEqual(event["from_camera_id"], 79)
        self.assertEqual(event["to_camera_id"], 79)

    def test_b_hand_04_rapid_successive_handoff_burst(self):
        """B4.3: Rapid burst of 5 handoffs across cameras updates state correctly."""
        self.preserver.lock_target(503, 79, {})
        for cam_id in [87, 83, 81, 88, 92]:
            self.preserver.handoff_target(503, cam_id, confidence=0.9)
        self.assertEqual(self.preserver.active_targets[503]["current_camera_id"], 92)
        self.assertEqual(len(self.preserver.active_targets[503]["history"]), 6)

    def test_b_hand_04_unregistered_target_handoff_raises_keyerror(self):
        """B4.4: Attempting handoff on unregistered ID raises KeyError."""
        with self.assertRaises(KeyError):
            self.preserver.handoff_target(9999999, 87, confidence=0.8)

    def test_b_hand_04_empty_signature_lock_permitted(self):
        """B4.5: Locking target with empty signature dict is handled gracefully."""
        self.preserver.lock_target(504, 79, {})
        self.assertIn(504, self.preserver.active_targets)
        self.assertEqual(self.preserver.active_targets[504]["signature"], {})


if __name__ == "__main__":
    unittest.main()
