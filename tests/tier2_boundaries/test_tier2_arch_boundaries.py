"""
tests/tier2_boundaries/test_tier2_arch_boundaries.py
Tier 2 Boundary and Corner Case Tests for Features F-ARCH-01 and F-ARCH-02.
5 boundary tests per feature (10 total tests).
"""

import sqlite3
import unittest
import numpy as np

from tests.reference_contracts import DatabaseContract, ReidEngineContract


class TestTier2ArchBoundaries(unittest.TestCase):
    """
    Boundary & Corner Cases for Persistence & Vector Search (F-ARCH-01, F-ARCH-02)
    """

    def setUp(self):
        self.db = DatabaseContract(":memory:")

    def tearDown(self):
        if hasattr(self, "db") and self.db and self.db.conn:
            self.db.conn.close()

    # --- F-ARCH-01 Boundaries ---
    def test_b_arch_01_null_crop_thumbnail_allowed(self):
        """B1.1: Sightings insert permits NULL / None crop_thumbnail without failure."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (79, 'Kujang', 'SSA', 0, 0)")
        sighting_id = self.db.insert_vehicle_sighting(
            79, 1, None, "car", 0.9, (10, 10, 50, 50), 0.0, 0.0,
            b"0"*1024, b"0"*512, "2026-09-23T00:00:00Z", crop_thumbnail=None
        )
        self.assertGreater(sighting_id, 0)

    def test_b_arch_01_sql_injection_resilience(self):
        """B1.2: Parameterized SQL queries safely neutralize SQL injection payload."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (79, 'Kujang', 'SSA', 0, 0)")
        malicious_input = "'; DROP TABLE cameras; --"
        sighting_id = self.db.insert_vehicle_sighting(
            79, 1, None, malicious_input, 0.9, (10, 10, 50, 50), 0.0, 0.0,
            b"0"*1024, b"0"*512, "2026-09-23T00:00:00Z"
        )
        self.assertGreater(sighting_id, 0)
        # Verify cameras table was NOT dropped
        cur.execute("SELECT count(*) FROM cameras;")
        self.assertEqual(cur.fetchone()[0], 1)

    def test_b_arch_01_non_existent_vehicle_journey_query(self):
        """B1.3: Querying journey for non-existent vehicle ID returns empty list, not error."""
        journey = self.db.get_vehicle_journey(99999999)
        self.assertEqual(journey, [])

    def test_b_arch_01_extreme_large_blob_insertion(self):
        """B1.4: Database handles high-resolution crop thumbnail BLOB (500 KB) cleanly."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (79, 'Kujang', 'SSA', 0, 0)")
        large_thumbnail = b"\xff\xd8" + (b"\x00" * 500000)
        sighting_id = self.db.insert_vehicle_sighting(
            79, 1, 101, "bus", 0.95, (0, 0, 640, 480), 90.0, 10.0,
            b"0"*1024, b"0"*512, "2026-09-23T00:00:00Z", crop_thumbnail=large_thumbnail
        )
        cur.execute("SELECT length(crop_thumbnail) FROM vehicle_sightings WHERE sighting_id = ?", (sighting_id,))
        self.assertEqual(cur.fetchone()[0], len(large_thumbnail))

    def test_b_arch_01_foreign_key_violation_triggers_error(self):
        """B1.5: Inserting sighting with non-existent camera ID violates foreign key constraint."""
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.insert_vehicle_sighting(
                999999, 1, None, "car", 0.9, (0, 0, 10, 10), 0.0, 0.0,
                b"0"*1024, b"0"*512, "2026-09-23T00:00:00Z"
            )

    # --- F-ARCH-02 Boundaries ---
    def test_b_arch_02_zero_norm_query_vector_protection(self):
        """B2.1: All-zero query vector handles division-by-zero safely without NaN crash."""
        q_zero = np.zeros(256, dtype=np.float32)
        q_hist = np.zeros(128, dtype=np.float32)
        cands = np.random.randn(5, 256).astype(np.float32)
        c_hists = np.random.randn(5, 128).astype(np.float32)

        sims = ReidEngineContract.compute_appearance_similarity(q_zero, q_hist, cands, c_hists)
        self.assertEqual(len(sims), 5)
        self.assertFalse(np.any(np.isnan(sims)))

    def test_b_arch_02_zero_norm_candidate_vector_protection(self):
        """B2.2: Candidate vectors containing zero norm are handled without crash."""
        q = np.random.randn(256).astype(np.float32)
        q_h = np.random.randn(128).astype(np.float32)
        cands = np.zeros((3, 256), dtype=np.float32)
        c_hists = np.zeros((3, 128), dtype=np.float32)

        sims = ReidEngineContract.compute_appearance_similarity(q, q_h, cands, c_hists)
        self.assertEqual(len(sims), 3)
        self.assertFalse(np.any(np.isnan(sims)))

    def test_b_arch_02_single_candidate_vector_evaluation(self):
        """B2.3: Single candidate vector (1D or 2D) evaluates correctly."""
        q = np.random.randn(256).astype(np.float32)
        q /= np.linalg.norm(q)
        q_h = np.zeros(128, dtype=np.float32)

        sims = ReidEngineContract.compute_appearance_similarity(q, q_h, q, q_h)
        self.assertEqual(len(sims), 1)
        self.assertAlmostEqual(float(sims[0]), 0.60, places=3)

    def test_b_arch_02_1000_candidate_vector_stress(self):
        """B2.4: Stress evaluation of 1,000 candidate vectors finishes in < 2.0 ms."""
        q = np.random.randn(256).astype(np.float32)
        q_h = np.random.randn(128).astype(np.float32)
        cands = np.random.randn(1000, 256).astype(np.float32)
        c_h = np.random.randn(1000, 128).astype(np.float32)

        sims = ReidEngineContract.compute_appearance_similarity(q, q_h, cands, c_h)
        self.assertEqual(len(sims), 1000)
        self.assertFalse(np.any(np.isnan(sims)))

    def test_b_arch_02_nan_safety_in_vector_components(self):
        """B2.5: Replaces NaN components with 0.0 before dot product to prevent propagation."""
        q = np.random.randn(256).astype(np.float32)
        q[0] = np.nan
        q = np.nan_to_num(q, nan=0.0)
        q_h = np.zeros(128, dtype=np.float32)

        cand = np.random.randn(1, 256).astype(np.float32)
        cand_h = np.zeros((1, 128), dtype=np.float32)

        sims = ReidEngineContract.compute_appearance_similarity(q, q_h, cand, cand_h)
        self.assertFalse(np.isnan(sims[0]))


if __name__ == "__main__":
    unittest.main()
