"""Housekeeping and caching: SightingMaintenance construction and the shared stats cache."""

import threading
import time
import unittest

from backend.database import Database
from backend.maintenance import SightingMaintenance


class MaintenanceConstructionTest(unittest.TestCase):
    def test_server_style_positional_construction_keeps_working(self):
        db = Database(":memory:")
        try:
            m = SightingMaintenance(db, 300.0)               # exactly how backend/server.py builds it
            self.assertEqual(m.interval_s, 300.0)
            self.assertEqual(m.retention_days, 0.0, "retention is opt-in (SIGHTING_RETENTION_DAYS)")
            self.assertEqual(m.run_once(), 0)
            self.assertEqual(set(m.last_run), {"pruned_sightings", "pages_freed"})
        finally:
            db.close()


class StatsCacheTest(unittest.TestCase):
    def setUp(self):
        from backend.server import _TTLCache
        self.cls = _TTLCache

    def test_value_is_computed_once_per_interval(self):
        cache, calls = self.cls(30.0), []
        for _ in range(3):
            cache.get("k", lambda: calls.append(1) or len(calls))
        self.assertEqual(len(calls), 1)

    def test_zero_ttl_disables_caching(self):
        cache, calls = self.cls(0), []
        for _ in range(3):
            cache.get("k", lambda: calls.append(1))
        self.assertEqual(len(calls), 3)

    def test_expired_value_is_recomputed(self):
        cache, calls = self.cls(0.05), []
        cache.get("k", lambda: calls.append(1))
        time.sleep(0.08)
        cache.get("k", lambda: calls.append(1))
        self.assertEqual(len(calls), 2)

    def test_concurrent_callers_share_one_computation(self):
        cache, calls, results = self.cls(30.0), [], []

        def slow():
            calls.append(1)
            time.sleep(0.1)
            return "v"

        threads = [threading.Thread(target=lambda: results.append(cache.get("k", slow))) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(calls), 1)
        self.assertEqual(results, ["v"] * 8)

    def test_distinct_keys_are_independent_and_bounded(self):
        cache = self.cls(30.0)
        for i in range(100):
            cache.get(("k", i), lambda i=i: i)
        self.assertLessEqual(len(cache._entries), cache._MAX_KEYS)


if __name__ == "__main__":
    unittest.main()
