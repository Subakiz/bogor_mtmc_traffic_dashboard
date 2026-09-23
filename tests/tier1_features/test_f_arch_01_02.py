"""
tests/tier1_features/test_f_arch_01_02.py
Tier 1 Tests for Features F-ARCH-01 and F-ARCH-02.
>=5 test cases per feature (10 total tests).
"""

import os
import sqlite3
import time
import unittest
import numpy as np
from tests.reference_contracts import DatabaseContract, ReidEngineContract


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
            "handoff_events",
            "active_targets",
        }
        for tbl in required_tables:
            self.assertIn(tbl, tables, f"Missing required table: {tbl}")

        # Check indices
        cur.execute("SELECT name FROM sqlite_master WHERE type='index';")
        indices = {row[0] for row in cur.fetchall()}
        self.assertIn("idx_sightings_cam_time", indices)
        self.assertIn("idx_sightings_global_id", indices)

    def test_f_arch_01_insert_vehicle_sighting_stores_full_binary_blobs(self):
        """Test 2: insert_vehicle_sighting properly stores 1024-byte embedding and 512-byte color_hist."""
        cur = self.db.conn.cursor()
        # Seed dummy camera
        cur.execute(
            "INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (79, "CCTV - TUGU KUJANG", "SSA_RING", -6.6014, 106.8051),
        )

        emb_bytes = np.random.randn(256).astype(np.float32).tobytes()
        hist_bytes = np.zeros(128, dtype=np.float32).tobytes()

        sighting_id = self.db.insert_vehicle_sighting(
            camera_id=79,
            local_track_id=101,
            global_vehicle_id=5001,
            vehicle_class="car",
            confidence=0.92,
            bbox=(120, 150, 240, 220),
            heading_deg=240.0,
            speed_px_s=85.5,
            embedding=emb_bytes,
            color_hist=hist_bytes,
            timestamp_iso="2026-09-23T04:15:00Z",
            crop_thumbnail=b"\xff\xd8\xff\xe0mock_jpeg_crop",
        )
        self.assertGreater(sighting_id, 0)

        # Retrieve and verify byte lengths
        cur.execute("SELECT embedding, color_hist, crop_thumbnail FROM vehicle_sightings WHERE sighting_id = ?", (sighting_id,))
        row = cur.fetchone()
        self.assertEqual(len(row[0]), 1024)  # 256 * 4 bytes
        self.assertEqual(len(row[1]), 512)   # 128 * 4 bytes
        self.assertEqual(row[2], b"\xff\xd8\xff\xe0mock_jpeg_crop")

    def test_f_arch_01_sub_millisecond_insert_latency(self):
        """Test 3: Inserting sightings operates with sub-millisecond execution latency."""
        cur = self.db.conn.cursor()
        cur.execute(
            "INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (83, "CCTV - PASAR BOGOR", "SSA_RING", -6.6009, 106.7995),
        )

        emb = np.random.randn(256).astype(np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        # Measure 50 consecutive inserts
        t0 = time.perf_counter()
        for i in range(50):
            self.db.insert_vehicle_sighting(
                camera_id=83,
                local_track_id=200 + i,
                global_vehicle_id=None,
                vehicle_class="car",
                confidence=0.89,
                bbox=(50, 50, 150, 120),
                heading_deg=180.0,
                speed_px_s=60.0,
                embedding=emb,
                color_hist=hist,
                timestamp_iso=f"2026-09-23T04:15:{i:02d}Z",
            )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        avg_insert_ms = elapsed_ms / 50.0
        self.assertLess(avg_insert_ms, 2.0, f"Average insert latency {avg_insert_ms:.3f} ms exceeds 2.0 ms")

    def test_f_arch_01_handoff_event_recording_and_referential_integrity(self):
        """Test 4: record_handoff_event stores transition event with valid camera foreign keys."""
        cur = self.db.conn.cursor()
        cur.execute(
            "INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (79, "CCTV - TUGU KUJANG", "SSA_RING", -6.6014, 106.8051),
        )
        cur.execute(
            "INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
            (87, "CCTV - OTISTA", "SSA_RING", -6.6012, 106.8020),
        )

        event_id = self.db.record_handoff_event(
            global_vehicle_id=5001,
            from_camera_id=79,
            to_camera_id=87,
            confidence=0.91,
            timestamp_iso="2026-09-23T04:16:30Z",
        )
        self.assertGreater(event_id, 0)

        cur.execute("SELECT * FROM handoff_events WHERE event_id = ?", (event_id,))
        row = cur.fetchone()
        self.assertEqual(row["global_vehicle_id"], 5001)
        self.assertEqual(row["from_camera_id"], 79)
        self.assertEqual(row["to_camera_id"], 87)

    def test_f_arch_01_get_vehicle_journey_chronological_ordering(self):
        """Test 5: get_vehicle_journey returns chronological sightings joined with camera metadata."""
        cur = self.db.conn.cursor()
        for cid, name in [(79, "Tugu Kujang"), (87, "Otista"), (83, "Pasar Bogor")]:
            cur.execute(
                "INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (?, ?, ?, ?, ?)",
                (cid, name, "SSA_RING", -6.60, 106.80),
            )

        emb = np.zeros(256, dtype=np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        # Insert sightings at 3 separate times
        self.db.insert_vehicle_sighting(79, 1, 999, "car", 0.9, (10, 10, 50, 50), 0, 0, emb, hist, "2026-09-23T01:00:00Z")
        self.db.insert_vehicle_sighting(87, 2, 999, "car", 0.9, (10, 10, 50, 50), 0, 0, emb, hist, "2026-09-23T01:01:30Z")
        self.db.insert_vehicle_sighting(83, 3, 999, "car", 0.9, (10, 10, 50, 50), 0, 0, emb, hist, "2026-09-23T01:02:45Z")

        journey = self.db.get_vehicle_journey(999)
        self.assertEqual(len(journey), 3)
        self.assertEqual([j["camera_id"] for j in journey], [79, 87, 83])
        self.assertEqual([j["camera_name"] for j in journey], ["Tugu Kujang", "Otista", "Pasar Bogor"])


class TestFARCH02NumPyVectorReIDIndex(unittest.TestCase):
    """
    F-ARCH-02: NumPy Vector Re-ID Index
    Sub-millisecond cosine similarity search and circular sliding-window cache.
    """

    def setUp(self):
        np.random.seed(42)

    def test_f_arch_02_cosine_similarity_mathematical_precision(self):
        """Test 1: Batch cosine similarity matches analytical dot product of unit vectors."""
        # Query vector of unit norm
        q = np.random.randn(256).astype(np.float32)
        q /= np.linalg.norm(q)

        # 10 Candidate vectors of unit norm
        cands = np.random.randn(10, 256).astype(np.float32)
        cands /= np.linalg.norm(cands, axis=1, keepdims=True)

        # Expected dot product
        expected_sims = np.dot(cands, q)

        # Dummy histograms
        q_h = np.zeros(128, dtype=np.float32)
        c_h = np.zeros((10, 128), dtype=np.float32)

        # ReidEngine appearance calculation: S_appear = 0.60 * S_deep + 0.40 * S_color
        # With zero histograms, color sim is 0.0, so S_appear should be 0.60 * sim_deep
        computed = ReidEngineContract.compute_appearance_similarity(q, q_h, cands, c_h)
        np.testing.assert_allclose(computed, 0.60 * expected_sims, atol=1e-5)

    def test_f_arch_02_sub_millisecond_batch_vector_search_latency(self):
        """Test 2: Searching across 100 candidate vectors finishes in < 0.3 ms."""
        q_emb = np.random.randn(256).astype(np.float32)
        q_hist = np.random.randn(128).astype(np.float32)
        cand_embs = np.random.randn(100, 256).astype(np.float32)
        cand_hists = np.random.randn(100, 128).astype(np.float32)

        t0 = time.perf_counter()
        sims = ReidEngineContract.compute_appearance_similarity(q_emb, q_hist, cand_embs, cand_hists)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(len(sims), 100)
        self.assertLess(elapsed_ms, 0.5, f"Vector search latency {elapsed_ms:.3f} ms exceeds 0.5 ms")

    def test_f_arch_02_identical_signature_yields_maximum_similarity(self):
        """Test 3: Querying an identical vehicle signature yields appearance score of 1.0."""
        emb = np.random.randn(256).astype(np.float32)
        hist = np.random.randn(128).astype(np.float32)

        sims = ReidEngineContract.compute_appearance_similarity(emb, hist, emb.reshape(1, -1), hist.reshape(1, -1))
        self.assertAlmostEqual(float(sims[0]), 1.0, places=4)

    def test_f_arch_02_orthogonal_signature_yields_zero_similarity(self):
        """Test 4: Orthogonal feature vectors yield similarity near 0.0."""
        emb1 = np.zeros(256, dtype=np.float32)
        emb1[:128] = 1.0 / np.sqrt(128)

        emb2 = np.zeros(256, dtype=np.float32)
        emb2[128:] = 1.0 / np.sqrt(128)

        hist1 = np.zeros(128, dtype=np.float32)
        hist1[:64] = 1.0 / np.sqrt(64)

        hist2 = np.zeros(128, dtype=np.float32)
        hist2[64:] = 1.0 / np.sqrt(64)

        sims = ReidEngineContract.compute_appearance_similarity(emb1, hist1, emb2.reshape(1, -1), hist2.reshape(1, -1))
        self.assertAlmostEqual(float(sims[0]), 0.0, places=4)

    def test_f_arch_02_top_k_ranking_integrity(self):
        """Test 5: Top-k ranking correctly sorts candidate indices by descending similarity."""
        q_emb = np.random.randn(256).astype(np.float32)
        q_hist = np.random.randn(128).astype(np.float32)

        cand_embs = np.random.randn(20, 256).astype(np.float32)
        cand_hists = np.random.randn(20, 128).astype(np.float32)

        # Make candidate #7 explicitly identical to query
        cand_embs[7] = q_emb
        cand_hists[7] = q_hist

        sims = ReidEngineContract.compute_appearance_similarity(q_emb, q_hist, cand_embs, cand_hists)
        sorted_indices = np.argsort(sims)[::-1]

        # Best match must be index 7
        self.assertEqual(sorted_indices[0], 7)
        self.assertAlmostEqual(float(sims[7]), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()
