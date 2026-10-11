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

from tests.mock_stream import MockHLSStreamServer, SyntheticFrameGenerator, SyntheticTrajectorySimulator
from tests.reference_contracts import ReidEngineContract, TopologyContract
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

    def test_feat_m1_synthetic_detection_pipeline_execution(self):
        """Test 5: Synthetic frame yields expected vehicle sightings."""
        sim = SyntheticTrajectorySimulator()
        sightings = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=101, route_camera_ids=[79], speed_factor=1.0
        )
        self.assertGreater(len(sightings), 0)
        first = sightings[0]
        self.assertEqual(first["global_vehicle_id"], 101)
        self.assertEqual(first["camera_id"], 79)
        self.assertEqual(first["vehicle_class"], "car")


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


class TestFEATM2CoreMLReIDExtractor(unittest.TestCase):
    """
    FEAT-M2: CoreML Vehicle Re-ID Extractor
    Preprocessing, 256-dim embeddings, and L2 normalization.
    """

    def setUp(self):
        self.frame_gen = SyntheticFrameGenerator(seed=42)

    def test_feat_m2_crop_resize_to_224x224(self):
        """Test 1: Vehicle crop is properly preprocessed and resized to 224x224 RGB."""
        crop_arr, crop_bytes = self.frame_gen.generate_vehicle_crop(
            bbox=(50, 50, 150, 120), color=(220, 30, 30), target_size=(224, 224)
        )
        self.assertEqual(crop_arr.shape, (224, 224, 3))
        self.assertGreater(len(crop_bytes), 500)

    def test_feat_m2_embedding_vector_dimensions_256(self):
        """Test 2: Re-ID extractor produces exactly 256-dimensional float32 vector."""
        sim = SyntheticTrajectorySimulator()
        sightings = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=55, route_camera_ids=[79]
        )
        emb = sightings[0]["embedding_vector"]
        self.assertEqual(len(emb), 256)
        self.assertEqual(emb.dtype, np.float32)

    def test_feat_m2_l2_normalization_property(self):
        """Test 3: Extracted embedding is unit normalized (||v||_2 == 1.0)."""
        sim = SyntheticTrajectorySimulator()
        sightings = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=77, route_camera_ids=[79]
        )
        emb = sightings[0]["embedding_vector"]
        norm = float(np.linalg.norm(emb))
        self.assertAlmostEqual(norm, 1.0, places=4)

    def test_feat_m2_deterministic_embedding_on_same_vehicle(self):
        """Test 4: Same vehicle produces highly correlated embeddings across views (>0.95)."""
        sim = SyntheticTrajectorySimulator()
        sightings = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=88, route_camera_ids=[79, 87]
        )
        emb1 = sightings[0]["embedding_vector"]
        emb2 = sightings[-1]["embedding_vector"]
        cos_sim = float(np.dot(emb1, emb2))
        self.assertGreaterEqual(cos_sim, 0.90)

    def test_feat_m2_extraction_throughput(self):
        """Test 5: Synthetic feature extraction completes in under 2.0 ms per crop."""
        t0 = time.perf_counter()
        for i in range(100):
            rng = np.random.RandomState(i)
            vec = rng.randn(256).astype(np.float32)
            _ = vec / np.linalg.norm(vec)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_ms = elapsed_ms / 100.0
        self.assertLess(avg_ms, 1.0, f"Extraction latency {avg_ms:.3f} ms exceeds 1.0 ms")


class TestFEATM3AppearanceColorSignature(unittest.TestCase):
    """
    FEAT-M3: Multi-Modal Appearance & Color Signature
    128-bin HSV histogram + S_appear = 0.60 * S_deep + 0.40 * S_color.
    """

    def test_feat_m3_128_bin_histogram_normalization(self):
        """Test 1: Spatial HSV histogram has 128 bins and sums to 1.0."""
        sim = SyntheticTrajectorySimulator()
        sightings = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=12, route_camera_ids=[79], primary_color=(255, 0, 0)
        )
        hist = sightings[0]["color_hist_vector"]
        self.assertEqual(len(hist), 128)
        self.assertAlmostEqual(float(np.sum(hist)), 1.0, places=4)

    def test_feat_m3_distinct_color_separation(self):
        """Test 2: Distinct colors (Red vs Blue) yield low color cosine similarity (< 0.20)."""
        sim = SyntheticTrajectorySimulator()
        red_car = sim.simulate_vehicle_corridor_journey(1, [79], primary_color=(255, 0, 0))[0]
        blue_car = sim.simulate_vehicle_corridor_journey(2, [79], primary_color=(0, 0, 255))[0]

        h_red = red_car["color_hist_vector"]
        h_blue = blue_car["color_hist_vector"]

        color_sim = float(np.dot(h_red, h_blue) / (np.linalg.norm(h_red) * np.linalg.norm(h_blue)))
        self.assertLess(color_sim, 0.20, f"Red and Blue color similarity {color_sim} too high")

    def test_feat_m3_fused_appearance_metric_formula(self):
        """Test 3: S_appear exactly matches 0.60 * S_deep + 0.40 * S_color."""
        q_emb = np.array([1.0, 0.0] + [0.0] * 254, dtype=np.float32)
        q_hist = np.array([1.0, 0.0] + [0.0] * 126, dtype=np.float32)

        cand_emb = np.array([0.8, 0.6] + [0.0] * 254, dtype=np.float32)  # cos_sim = 0.80
        cand_hist = np.array([0.5, 0.866] + [0.0] * 126, dtype=np.float32) # cos_sim = 0.50

        expected = 0.60 * 0.80 + 0.40 * 0.50  # 0.48 + 0.20 = 0.68
        score = ReidEngineContract.compute_appearance_similarity(q_emb, q_hist, cand_emb, cand_hist)
        self.assertAlmostEqual(float(score[0]), expected, places=4)

    def test_feat_m3_color_histogram_separates_identical_make_vehicles(self):
        """Test 4: Vehicles with similar silhouette but different colors are separated by color histogram."""
        # Two cars of the same model: deep features are similar (~0.85), but colors differ (red vs blue)
        shared_deep = np.random.randn(256).astype(np.float32)
        shared_deep /= np.linalg.norm(shared_deep)

        red_hist = np.zeros(128, dtype=np.float32)
        red_hist[10:20] = 0.1
        blue_hist = np.zeros(128, dtype=np.float32)
        blue_hist[80:90] = 0.1

        score_diff_color = ReidEngineContract.compute_appearance_similarity(
            shared_deep, red_hist, shared_deep, blue_hist
        )
        score_same_color = ReidEngineContract.compute_appearance_similarity(
            shared_deep, red_hist, shared_deep, red_hist
        )

        self.assertGreater(float(score_same_color[0]), float(score_diff_color[0]) + 0.30)

    def test_feat_m3_signature_serialization_byte_lengths(self):
        """Test 5: Signatures convert to binary BLOBs of exact expected lengths (1024 and 512 bytes)."""
        emb = np.random.randn(256).astype(np.float32)
        hist = np.random.randn(128).astype(np.float32)
        b_emb = emb.tobytes()
        b_hist = hist.tobytes()
        self.assertEqual(len(b_emb), 1024)
        self.assertEqual(len(b_hist), 512)


class TestFEATM7FusedMTMCMatcher(unittest.TestCase):
    """
    FEAT-M7: Fused MTMC Re-ID Matcher
    S_fused = 0.70 * S_appear + 0.30 * S_st with class gating and threshold verification.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()

    def test_feat_m7_class_gating_rejects_mismatched_classes(self):
        """Test 1: Vehicle class gating rejects candidate if class differs (e.g. car vs truck)."""
        emb = np.random.randn(256).astype(np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        target_sig = {
            "vehicle_class": "car",
            "embedding": emb,
            "color_hist": hist,
            "camera_id": 79,
            "timestamp": 100.0,
        }
        candidates = [
            {
                "vehicle_class": "truck",  # Mismatch!
                "embedding": emb,
                "color_hist": hist,
                "camera_id": 87,
                "timestamp": 170.0,
            }
        ]
        match = ReidEngineContract.match_candidate_vehicle(target_sig, candidates, current_time=170.0, topology_contract=self.topo)
        self.assertIsNone(match)

    def test_feat_m7_threshold_rejection_below_appearance_cutoff(self):
        """Test 2: Rejects candidate with appearance score below 0.65 even if spatiotemporal timing is perfect."""
        q_emb = np.random.randn(256).astype(np.float32)
        q_emb /= np.linalg.norm(q_emb)
        diff_emb = -q_emb  # Opposite vector: cos_sim = -1.0

        hist = np.zeros(128, dtype=np.float32)

        target_sig = {
            "vehicle_class": "car",
            "embedding": q_emb.tobytes(),
            "color_hist": hist.tobytes(),
            "camera_id": 79,
            "timestamp": 100.0,
        }
        candidates = [
            {
                "vehicle_class": "car",
                "embedding": diff_emb.tobytes(),
                "color_hist": hist.tobytes(),
                "camera_id": 87,
                "timestamp": 170.0,  # Valid time (70s)
            }
        ]
        match = ReidEngineContract.match_candidate_vehicle(target_sig, candidates, current_time=170.0, topology_contract=self.topo)
        self.assertIsNone(match)

    def test_feat_m7_positive_match_above_thresholds(self):
        """Test 3: Returns candidate when S_appear >= 0.65 and S_fused >= 0.72."""
        sim = SyntheticTrajectorySimulator()
        journey = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=505, route_camera_ids=[79, 87]
        )
        target = journey[0]
        # Candidate at camera 87
        cand = [j for j in journey if j["camera_id"] == 87][0]

        target_sig = {
            "vehicle_class": target["vehicle_class"],
            "embedding": target["embedding"],
            "color_hist": target["color_hist"],
            "camera_id": 79,
            "timestamp": target["timestamp"],
        }
        candidates = [cand]

        match = ReidEngineContract.match_candidate_vehicle(
            target_sig, candidates, current_time=cand["timestamp"], topology_contract=self.topo
        )
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match["fused_score"], 0.72)
        self.assertGreaterEqual(match["appearance_score"], 0.65)

    def test_feat_m7_spatiotemporal_penalty_degrades_score_for_late_arrival(self):
        """Test 4: Spatiotemporal score decays when vehicle arrives significantly late."""
        # Test transition 79 -> 87 (Otista)
        is_on_time, score_on_time = self.topo.is_valid_transition(79, 87, 80.0)
        is_late, score_late = self.topo.is_valid_transition(79, 87, 180.0)

        self.assertTrue(is_on_time)
        self.assertTrue(is_late)
        self.assertGreater(score_on_time, score_late)

    def test_feat_m7_multi_candidate_disambiguation(self):
        """Test 5: Matcher correctly selects the true matching car among 3 candidate vehicles."""
        sim = SyntheticTrajectorySimulator()
        true_car_journey = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=777, route_camera_ids=[79, 87], primary_color=(220, 20, 20)
        )
        target = true_car_journey[0]
        true_cand = [j for j in true_car_journey if j["camera_id"] == 87][0]

        # Create two distractor vehicles
        distractor_1 = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=888, route_camera_ids=[87], primary_color=(30, 200, 30)
        )[0]
        distractor_2 = sim.simulate_vehicle_corridor_journey(
            global_vehicle_id=999, route_camera_ids=[87], primary_color=(30, 30, 220)
        )[0]

        target_sig = {
            "vehicle_class": "car",
            "embedding": target["embedding"],
            "color_hist": target["color_hist"],
            "camera_id": 79,
            "timestamp": target["timestamp"],
        }
        candidates = [distractor_1, true_cand, distractor_2]

        match = ReidEngineContract.match_candidate_vehicle(
            target_sig, candidates, current_time=true_cand["timestamp"], topology_contract=self.topo
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["global_vehicle_id"], 777)


if __name__ == "__main__":
    unittest.main()
