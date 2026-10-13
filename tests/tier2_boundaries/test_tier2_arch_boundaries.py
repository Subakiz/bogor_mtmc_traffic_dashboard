"""
tests/tier2_boundaries/test_tier2_arch_boundaries.py
Tier 2 Boundary and Corner Case Tests for Feature F-ARCH-01 (SQLite WAL persistence).
"""

import sqlite3
import unittest

from tests.reference_contracts import DatabaseContract


class TestTier2ArchBoundaries(unittest.TestCase):
    """
    Boundary & Corner Cases for Persistence (F-ARCH-01)
    """

    def setUp(self):
        self.db = DatabaseContract(":memory:")

    def tearDown(self):
        if hasattr(self, "db") and self.db and self.db.conn:
            self.db.conn.close()

    # --- F-ARCH-01 Boundaries ---
    def test_b_arch_01_sql_injection_resilience(self):
        """B1.2: Parameterized SQL queries safely neutralize SQL injection payload."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (79, 'Kujang', 'SSA', 0, 0)")
        malicious_input = "'; DROP TABLE cameras; --"
        sighting_id = self.db.insert_vehicle_sighting(
            79, 1, malicious_input, 0.9, (10, 10, 50, 50), 0.0, 0.0, "2026-09-23T00:00:00Z"
        )
        self.assertGreater(sighting_id, 0)
        # Verify cameras table was NOT dropped
        cur.execute("SELECT count(*) FROM cameras;")
        self.assertEqual(cur.fetchone()[0], 1)

    def test_b_arch_01_foreign_key_violation_triggers_error(self):
        """B1.5: Inserting sighting with non-existent camera ID violates foreign key constraint."""
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.insert_vehicle_sighting(
                999999, 1, "car", 0.9, (0, 0, 10, 10), 0.0, 0.0, "2026-09-23T00:00:00Z"
            )


if __name__ == "__main__":
    unittest.main()
