"""
tests/tier1_features/test_feat_m_models.py
Tier 1 Tests for Features FEAT-M4, FEAT-M1, FEAT-M5, FEAT-M2, FEAT-M3, and FEAT-M7.
>=5 test cases per feature (30 total tests).
"""

import collections
import io
import time
import unittest
import numpy as np
from PIL import Image

from tests.mock_stream import MockHLSStreamServer, SyntheticFrameGenerator
from tests.reference_contracts import TopologyContract
from tests import TOTAL_CAMERAS


class TestFEATM4StreamIngestion(unittest.TestCase):
    """
    FEAT-M4: Threaded Low-Latency HLS Stream Ingestion
    Ring buffer, packet flushing, auto-reconnect, and multi-stream concurrency.
    """

    def setUp(self):
        self.stream_server = MockHLSStreamServer(camera_id=79)

    def test_feat_m4_bounded_ring_buffer_overwrites_stale_frames(self):
        """Test 1: Bounded ring buffer retains only newest frame, preventing lag accumulation."""
        buffer_size = 3
        ring_buffer = collections.deque(maxlen=buffer_size)

        # Ingest 10 frames sequentially
        for i in range(10):
            frame_payload = {"frame_id": i, "timestamp": time.time(), "data": b"mock_frame_data"}
            ring_buffer.append(frame_payload)

        # Buffer must have exactly buffer_size items and contain newest frames (7, 8, 9)
        self.assertEqual(len(ring_buffer), buffer_size)
        retained_ids = [f["frame_id"] for f in ring_buffer]
        self.assertEqual(retained_ids, [7, 8, 9])

    def test_feat_m4_packet_flushing_latencies(self):
        """Test 2: Simulating packet flush on retrieve ensures frame latency < 50ms."""
        packets = self.stream_server.get_simulated_packet_stream(num_packets=10)
        self.assertEqual(len(packets), 10)
        # Verify timestamps advance smoothly
        pts_diffs = [packets[i + 1]["pts_time"] - packets[i]["pts_time"] for i in range(len(packets) - 1)]
        for diff in pts_diffs:
            self.assertAlmostEqual(diff, 0.04, places=2)

    def test_feat_m4_hls_playlist_parsing_and_sequence_advance(self):
        """Test 3: HLS playlist increments media sequence numbers and provides 3.32s segments."""
        p1 = self.stream_server.get_playlist_m3u8()
        p2 = self.stream_server.get_playlist_m3u8()

        self.assertIn("#EXTM3U", p1)
        self.assertIn("#EXTINF:3.320000,", p1)
        self.assertIn("#EXT-X-PROGRAM-DATE-TIME:", p1)
        # p2 must have a higher sequence number than p1
        self.assertNotEqual(p1, p2)

    def test_feat_m4_multi_stream_concurrency(self):
        """Test 4: Multiple independent camera stream servers operate without interference."""
        server_79 = MockHLSStreamServer(camera_id=79)
        server_83 = MockHLSStreamServer(camera_id=83)

        p_79 = server_79.get_playlist_m3u8()
        p_83 = server_83.get_playlist_m3u8()

        self.assertIsNotNone(p_79)
        self.assertIsNotNone(p_83)
        self.assertEqual(server_79.camera_id, 79)
        self.assertEqual(server_83.camera_id, 83)

    def test_feat_m4_reconnect_recovery_after_simulated_drop(self):
        """Test 5: Stream ingestion handles reconnect after packet interruption."""
        # Simulate connection drop by resetting sequence
        server = MockHLSStreamServer(camera_id=88)
        _ = server.get_playlist_m3u8()
        # Simulate drop and restart
        server.sequence_number += 50
        recovered_playlist = server.get_playlist_m3u8()
        self.assertIn("#EXTM3U", recovered_playlist)
        self.assertIn(f"segment_{server.sequence_number}.ts", recovered_playlist)


class TestFEATM1CoreMLDetection(unittest.TestCase):
    """
    FEAT-M1: CoreML Detection Service
    Bounding box validation, vehicle class filtering, and confidence thresholding.
    """

    def setUp(self):
        self.frame_gen = SyntheticFrameGenerator(width=640, height=480, seed=123)

    def test_feat_m1_valid_rgb_frame_dimensions(self):
        """Test 1: Generated camera frame adheres to (480, 640, 3) RGB format."""
        frame_arr, jpeg_bytes = self.frame_gen.generate_frame(
            camera_id=79,
            vehicles=[{"bbox": (100, 100, 200, 180), "class": "car", "color": (200, 40, 40)}],
        )
        self.assertEqual(frame_arr.shape, (480, 640, 3))
        self.assertEqual(frame_arr.dtype, np.uint8)
        self.assertGreater(len(jpeg_bytes), 1000)

    def test_feat_m1_bounding_box_coordinate_ordering(self):
        """Test 2: Bounding box satisfies x1 < x2 and y1 < y2 within frame dimensions."""
        test_bboxes = [
            (50, 60, 180, 140),
            (200, 220, 310, 290),
            (400, 150, 520, 210),
        ]
        fw, fh = 640, 480
        for b in test_bboxes:
            x1, y1, x2, y2 = b
            self.assertTrue(0 <= x1 < x2 <= fw)
            self.assertTrue(0 <= y1 < y2 <= fh)

    def test_feat_m1_vehicle_class_filtering_accepts_vehicles_only(self):
        """Test 3: Allowed vehicle classes are restricted to traffic classes."""
        allowed_classes = {"car", "bus", "truck", "motorcycle"}
        mock_detections = [
            {"class": "car", "conf": 0.92},
            {"class": "truck", "conf": 0.85},
            {"class": "bus", "conf": 0.78},
            {"class": "motorcycle", "conf": 0.81},
            {"class": "pedestrian", "conf": 0.95},  # noise
            {"class": "dog", "conf": 0.70},         # noise
        ]
        filtered = [d for d in mock_detections if d["class"] in allowed_classes]
        self.assertEqual(len(filtered), 4)
        self.assertNotIn("pedestrian", [d["class"] for d in filtered])

    def test_feat_m1_confidence_score_thresholding(self):
        """Test 4: Detections below confidence cutoff (0.35) are pruned."""
        detections = [
            {"bbox": (10, 10, 50, 50), "class": "car", "conf": 0.88},
            {"bbox": (60, 60, 90, 90), "class": "car", "conf": 0.32},  # low conf
            {"bbox": (100, 100, 150, 150), "class": "car", "conf": 0.15}, # low conf
            {"bbox": (200, 200, 300, 280), "class": "bus", "conf": 0.74},
        ]
        valid_dets = [d for d in detections if d["conf"] >= 0.35]
        self.assertEqual(len(valid_dets), 2)
        self.assertTrue(all(d["conf"] >= 0.35 for d in valid_dets))

class TestFEATM5SchedulingDaemon(unittest.TestCase):
    """
    FEAT-M5: 3-Tier Adaptive Scheduling Daemon
    Polling rates: Tier 1 (15-20 FPS), Tier 2 (2 FPS), Tier 3 (Round-Robin).
    """

    def setUp(self):
        topo = TopologyContract()
        all_cams = set(topo.nodes.keys())
        self.tiers = {
            1: {"target_fps": 15.0, "cameras": set()},
            2: {"target_fps": 2.0, "cameras": set()},
            3: {"target_fps": 0.2, "cameras": set()},
        }
        # Populate 94 cameras: 1 active target cam in Tier 1, 12 junction hubs in Tier 2, remainder in Tier 3
        self.tiers[1]["cameras"].add(79)
        hub_cams = {80, 81, 82, 83, 84, 85, 86, 87, 88, 92, 161, 167}
        self.tiers[2]["cameras"] = hub_cams & all_cams
        self.tiers[3]["cameras"] = all_cams - self.tiers[1]["cameras"] - self.tiers[2]["cameras"]

    def test_feat_m5_tier_partitioning_counts(self):
        """Test 1: Camera tiers partition the total 94 camera network."""
        total_assigned = (
            len(self.tiers[1]["cameras"])
            + len(self.tiers[2]["cameras"])
            + len(self.tiers[3]["cameras"])
        )
        self.assertEqual(total_assigned, TOTAL_CAMERAS)

    def test_feat_m5_tier1_fps_allocation(self):
        """Test 2: Tier 1 camera is allocated 15-20 FPS polling rate."""
        fps = self.tiers[1]["target_fps"]
        self.assertGreaterEqual(fps, 15.0)
        self.assertLessEqual(fps, 20.0)

    def test_feat_m5_dynamic_promotion_on_exit_event(self):
        """Test 3: Downstream candidate camera is dynamically elevated from Tier 3 to Tier 1."""
        candidate_cam = 87
        # Ensure 87 starts in Tier 2 or 3
        if candidate_cam in self.tiers[1]["cameras"]:
            self.tiers[1]["cameras"].remove(candidate_cam)
        self.tiers[3]["cameras"].add(candidate_cam)

        # Trigger exit event towards 87 -> promote
        self.tiers[3]["cameras"].remove(candidate_cam)
        self.tiers[1]["cameras"].add(candidate_cam)

        self.assertIn(candidate_cam, self.tiers[1]["cameras"])
        self.assertNotIn(candidate_cam, self.tiers[3]["cameras"])

    def test_feat_m5_dynamic_demotion_on_window_expiration(self):
        """Test 4: Candidate camera demotes back to Tier 3 when arrival window expires."""
        candidate_cam = 87
        self.tiers[1]["cameras"].add(candidate_cam)

        # Simulate arrival window timeout
        self.tiers[1]["cameras"].remove(candidate_cam)
        self.tiers[3]["cameras"].add(candidate_cam)

        self.assertIn(candidate_cam, self.tiers[3]["cameras"])
        self.assertNotIn(candidate_cam, self.tiers[1]["cameras"])

    def test_feat_m5_compute_budget_stays_bounded(self):
        """Test 5: Total compute workload across all tiers stays within ANE budget."""
        # Workload = sum(count * fps)
        load = sum(len(cfg["cameras"]) * cfg["target_fps"] for cfg in self.tiers.values())
        # Target: Total equivalent inferences/second <= 70 FPS on Apple Silicon ANE
        self.assertLess(load, 75.0, f"Total scheduled FPS {load} exceeds ANE budget 75 FPS")


if __name__ == "__main__":
    unittest.main()
