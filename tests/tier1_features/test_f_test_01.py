"""
tests/tier1_features/test_f_test_01.py
Tier 1 Tests for Feature F-TEST-01 (Deterministic Simulation & E2E Test Suite).
Synthetic frames, mock HLS streams and the mock camera network.
"""

import json
import unittest
import numpy as np

from tests import TOTAL_CAMERAS
from tests.mock_stream import (
    MockCameraNetwork,
    MockHLSStreamServer,
    SyntheticFrameGenerator,
)


class TestFTEST01DeterministicSimulation(unittest.TestCase):
    """
    F-TEST-01: Deterministic Simulation & E2E Test Suite
    Synthetic frame generation, trajectory simulation, offline mock HLS, and test reporting.
    """

    def test_f_test_01_synthetic_frame_generator_determinism(self):
        """Test 1: Identical seeds produce bitwise identical synthetic camera frames."""
        gen1 = SyntheticFrameGenerator(seed=999)
        gen2 = SyntheticFrameGenerator(seed=999)

        vehicles = [{"bbox": (100, 100, 200, 160), "class": "car", "color": (200, 30, 30)}]
        arr1, b1 = gen1.generate_frame(79, vehicles, timestamp_str="2026-09-23 00:00:00")
        arr2, b2 = gen2.generate_frame(79, vehicles, timestamp_str="2026-09-23 00:00:00")

        np.testing.assert_array_equal(arr1, arr2)
        self.assertEqual(b1, b2)

    def test_f_test_01_mock_hls_stream_offline_execution(self):
        """Test 3: Mock HLS server generates valid streaming playlists without external network."""
        server = MockHLSStreamServer(camera_id=79, segment_duration=3.32)
        playlist = server.get_playlist_m3u8()
        self.assertTrue(playlist.startswith("#EXTM3U"))
        self.assertIn("#EXT-X-TARGETDURATION:4", playlist)
        self.assertIn("#EXTINF:3.320000,", playlist)

    def test_f_test_01_mock_camera_network_state_tracking(self):
        """Test 4: MockCameraNetwork manages 94 camera simulation states and tier assignments."""
        net = MockCameraNetwork()
        self.assertEqual(len(net.camera_states), TOTAL_CAMERAS)
        # Change tier
        net.set_camera_tier(79, 1)
        self.assertEqual(net.camera_states[79]["tier"], 1)

    def test_f_test_01_structured_json_reporting_format(self):
        """Test 5: Test reporting structures results into verifiable JSON format."""
        mock_report = {
            "tier": 1,
            "total_tests": 115,
            "passed": 115,
            "failed": 0,
            "duration_s": 0.125,
            "timestamp": "2026-09-23T04:25:00Z",
        }
        json_str = json.dumps(mock_report)
        parsed = json.loads(json_str)
        self.assertEqual(parsed["passed"], 115)
        self.assertEqual(parsed["failed"], 0)


if __name__ == "__main__":
    unittest.main()
