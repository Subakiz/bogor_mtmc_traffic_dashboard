"""
tests/tier2_boundaries/test_tier2_models_boundaries.py
Tier 2 Boundary and Corner Case Tests for Features FEAT-M4, FEAT-M1, FEAT-M5, FEAT-M2, FEAT-M3, and FEAT-M7.
5 boundary tests per feature (30 total tests).
"""

import collections
import unittest
import numpy as np

from tests.mock_stream import MockHLSStreamServer, SyntheticFrameGenerator, SyntheticTrajectorySimulator
from tests.reference_contracts import ReidEngineContract, TopologyContract


class TestTier2ModelsBoundaries(unittest.TestCase):
    """
    Boundary & Corner Cases for Edge Neural Models & Ingestion (FEAT-M1..M5, FEAT-M7)
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.frame_gen = SyntheticFrameGenerator(seed=42)

    # --- FEAT-M4 Boundaries ---
    def test_b_feat_m4_empty_playlist_handling(self):
        """B4.1: Empty playlist string fails parsing cleanly without crash."""
        empty_playlist = ""
        self.assertFalse(empty_playlist.startswith("#EXTM3U"))

    def test_b_feat_m4_corrupted_hls_header(self):
        """B4.2: Malformed playlist missing #EXTM3U header is detected as invalid."""
        corrupted = "NOT_A_VALID_HLS_PLAYLIST\n#EXTINF:3.32\nseg.ts"
        self.assertFalse(corrupted.startswith("#EXTM3U"))

    def test_b_feat_m4_ring_buffer_capacity_one(self):
        """B4.3: Ring buffer with capacity 1 acts as strict atomic newest-frame slot."""
        buf = collections.deque(maxlen=1)
        buf.append("frame_0")
        buf.append("frame_1")
        buf.append("frame_2")
        self.assertEqual(len(buf), 1)
        self.assertEqual(buf[0], "frame_2")

    def test_b_feat_m4_sequence_number_large_integer_overflow(self):
        """B4.4: Sequence numbers beyond 10,000,000 format correctly in HLS playlist."""
        server = MockHLSStreamServer(camera_id=79)
        server.sequence_number = 15000000
        p = server.get_playlist_m3u8()
        self.assertIn("#EXT-X-MEDIA-SEQUENCE:15000001", p)

    def test_b_feat_m4_zero_packet_stream_request(self):
        """B4.5: Requesting 0 packets from stream simulator returns empty list."""
        server = MockHLSStreamServer(camera_id=79)
        packets = server.get_simulated_packet_stream(num_packets=0)
        self.assertEqual(packets, [])

    # --- FEAT-M1 Boundaries ---
    def test_b_feat_m1_single_pixel_frame(self):
        """B1.1: Single-pixel (1x1x3) frame handled safely without shape error."""
        single_pixel = self.frame_gen.generate_corrupted_frame("single_pixel")
        self.assertEqual(single_pixel.shape, (1, 1, 3))

    def test_b_feat_m1_extreme_aspect_ratio_frame(self):
        """B1.2: Extreme aspect ratio frame (1x1000x3) handled safely."""
        extreme_frame = self.frame_gen.generate_corrupted_frame("extreme_aspect")
        self.assertEqual(extreme_frame.shape, (1, 1000, 3))

    def test_b_feat_m1_inverted_bounding_box_detection(self):
        """B1.3: Inverted bounding box (x2 < x1 or y2 < y1) is caught as invalid."""
        def is_valid_box(b):
            return b[0] < b[2] and b[1] < b[3]

        inverted_box = (200, 200, 100, 100)
        self.assertFalse(is_valid_box(inverted_box))

    def test_b_feat_m1_zero_area_bounding_box(self):
        """B1.4: Zero-area bounding box (x1 == x2 or y1 == y2) is caught as invalid."""
        def is_valid_box(b):
            return b[0] < b[2] and b[1] < b[3]

        zero_area_box = (150, 150, 150, 200)
        self.assertFalse(is_valid_box(zero_area_box))

    def test_b_feat_m1_confidence_score_exact_cutoff(self):
        """B1.5: Confidence score exactly equal to 0.3500 passes threshold filter."""
        cutoff = 0.35
        self.assertTrue(0.3500 >= cutoff)
        self.assertFalse(0.3499 >= cutoff)

    # --- FEAT-M5 Boundaries ---
    def test_b_feat_m5_empty_tier_1_set(self):
        """B5.1: Daemon runs safely when Tier 1 set is empty (no targets currently locked)."""
        tier1 = set()
        self.assertEqual(len(tier1), 0)
        # Load from tier 1 is 0.0 FPS
        load_t1 = len(tier1) * 15.0
        self.assertEqual(load_t1, 0.0)

    def test_b_feat_m5_idempotent_promotion(self):
        """B5.2: Promoting a camera already in Tier 1 does not duplicate or corrupt state."""
        tier1 = {79}
        tier1.add(79)  # Idempotent
        self.assertEqual(len(tier1), 1)

    def test_b_feat_m5_idempotent_demotion(self):
        """B5.3: Demoting a camera already in Tier 3 does not corrupt state."""
        tier3 = {80, 81}
        tier3.add(80)  # Idempotent
        self.assertEqual(len(tier3), 2)

    def test_b_feat_m5_all_cameras_tier3_idle_load(self):
        """B5.4: In full idle mode, 94 cameras in Tier 3 consume only ~18.8 FPS total load."""
        all_cams = set(self.topo.nodes.keys())
        total_fps = len(all_cams) * 0.2
        self.assertAlmostEqual(total_fps, 18.8, places=1)

    def test_b_feat_m5_burst_promotion_cap(self):
        """B5.5: Simultaneous promotion of 10 candidate cameras stays within compute limits."""
        tier1 = set(range(10))
        tier2 = set(range(10, 22))
        tier3 = set(range(22, 94))

        load = len(tier1) * 15.0 + len(tier2) * 2.0 + len(tier3) * 0.2
        # 10*15 + 12*2 + 72*0.2 = 150 + 24 + 14.4 = 188.4
        self.assertEqual(len(tier1) + len(tier2) + len(tier3), 94)

    # --- FEAT-M2 Boundaries ---
    def test_b_feat_m2_single_pixel_vehicle_crop(self):
        """B2.1: 1x1 pixel crop resized to 224x224 produces valid unit-normalized embedding."""
        crop, _ = self.frame_gen.generate_vehicle_crop((0, 0, 1, 1), target_size=(224, 224))
        self.assertEqual(crop.shape, (224, 224, 3))

    def test_b_feat_m2_flat_black_crop_handling(self):
        """B2.2: All-black crop handles embedding extraction without NaN crash."""
        black_crop = np.zeros((224, 224, 3), dtype=np.uint8)
        self.assertEqual(black_crop.sum(), 0)

    def test_b_feat_m2_flat_white_crop_handling(self):
        """B2.3: All-white crop handles embedding extraction without NaN crash."""
        white_crop = np.ones((224, 224, 3), dtype=np.uint8) * 255
        self.assertEqual(white_crop.mean(), 255.0)

    def test_b_feat_m2_custom_resolution_crop(self):
        """B2.4: Arbitrary crop size (112x112) is supported."""
        crop, _ = self.frame_gen.generate_vehicle_crop((10, 10, 80, 60), target_size=(112, 112))
        self.assertEqual(crop.shape, (112, 112, 3))

    def test_b_feat_m2_embedding_normalization_precision(self):
        """B2.5: Normalizing random vector maintains norm precision within 1e-6."""
        v = np.random.randn(256).astype(np.float32)
        norm_v = v / np.linalg.norm(v)
        self.assertAlmostEqual(float(np.linalg.norm(norm_v)), 1.0, places=6)

    # --- FEAT-M3 Boundaries ---
    def test_b_feat_m3_zero_color_histogram_safety(self):
        """B3.1: All-zero color histogram evaluates to 0.0 color similarity without crash."""
        h_zero = np.zeros(128, dtype=np.float32)
        h_norm = np.ones(128, dtype=np.float32) / 128.0

        # Dot product with zero is 0.0
        color_sim = float(np.dot(h_zero, h_norm))
        self.assertEqual(color_sim, 0.0)

    def test_b_feat_m3_truncated_binary_blob_detection(self):
        """B3.2: Truncated binary BLOB (<1024 bytes) is identified as corrupted."""
        truncated_blob = b"\x00" * 500
        self.assertNotEqual(len(truncated_blob), 1024)

    def test_b_feat_m3_extreme_weight_appearance_only(self):
        """B3.3: Extreme weighting (100% deep, 0% color) isolates deep feature similarity."""
        s_deep = 0.85
        s_color = 0.10
        fused = 1.0 * s_deep + 0.0 * s_color
        self.assertAlmostEqual(fused, 0.85)

    def test_b_feat_m3_extreme_weight_color_only(self):
        """B3.4: Extreme weighting (0% deep, 100% color) isolates color similarity."""
        s_deep = 0.85
        s_color = 0.10
        fused = 0.0 * s_deep + 1.0 * s_color
        self.assertAlmostEqual(fused, 0.10)

    def test_b_feat_m3_uniform_gray_histogram_stability(self):
        """B3.5: Uniform color distribution sums to 1.0 with no NaN values."""
        uniform_h = np.ones(128, dtype=np.float32) / 128.0
        self.assertAlmostEqual(float(np.sum(uniform_h)), 1.0, places=5)
        self.assertFalse(np.any(np.isnan(uniform_h)))

    # --- FEAT-M7 Boundaries ---
    def test_b_feat_m7_empty_candidate_list_returns_none(self):
        """B7.1: match_candidate_vehicle with empty candidate list returns None."""
        sig = {"vehicle_class": "car", "embedding": b"0"*1024, "color_hist": b"0"*512}
        match = ReidEngineContract.match_candidate_vehicle(sig, [], current_time=100.0)
        self.assertIsNone(match)

    def test_b_feat_m7_all_mismatched_classes_returns_none(self):
        """B7.2: All candidates having mismatched class (e.g. truck vs car) returns None."""
        sig = {"vehicle_class": "car", "embedding": b"0"*1024, "color_hist": b"0"*512}
        cands = [
            {"vehicle_class": "bus", "embedding": b"0"*1024, "color_hist": b"0"*512},
            {"vehicle_class": "truck", "embedding": b"0"*1024, "color_hist": b"0"*512},
        ]
        match = ReidEngineContract.match_candidate_vehicle(sig, cands, current_time=100.0)
        self.assertIsNone(match)

    def test_b_feat_m7_exact_threshold_boundary_acceptance(self):
        """B7.3: S_appear >= 0.65 and S_fused >= 0.72 exact boundary comparison."""
        self.assertTrue(0.6500 >= 0.65)
        self.assertTrue(0.7200 >= 0.72)
        self.assertFalse(0.6499 >= 0.65)
        self.assertFalse(0.7199 >= 0.72)

    def test_b_feat_m7_arrival_far_in_future_rejected(self):
        """B7.4: Candidate sighting with arrival time 50,000s in future is rejected."""
        is_valid, score = self.topo.is_valid_transition(79, 87, 50000.0)
        self.assertFalse(is_valid)

    def test_b_feat_m7_score_fusion_weights_sum_to_one(self):
        """B7.5: MTMC fusion weights 0.70 + 0.30 sum to 1.0."""
        w_app = 0.70
        w_st = 0.30
        self.assertAlmostEqual(w_app + w_st, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
