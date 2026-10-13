"""Housekeeping and caching: appearance-blob expiry and the shared stats cache."""

import threading
import time
import unittest
from datetime import datetime, timedelta, timezone

from backend.database import Database
from backend.maintenance import SightingMaintenance

CAMERA = 79
EMB, HIST = b"\x01" * 1024, b"\x02" * 512


def iso(age_s: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(seconds=age_s)).strftime("%Y-%m-%dT%H:%M:%SZ")


class ExpireEmbeddingsTest(unittest.TestCase):
    def setUp(self):
        self.db = Database(":memory:")
        for i in range(5):
            self._add(i, age_s=7200)          # two hours old
        for i in range(5, 8):
            self._add(i, age_s=60)            # recent

    def tearDown(self):
        self.db.close()

    def _add(self, track, age_s):
        return self.db.insert_vehicle_sighting(
            CAMERA, track, track, "car", 0.9, (0, 0, 10, 10), 0.0, 0.0, EMB, HIST, iso(age_s))

    def _blob_lengths(self):
        rows = self.db.conn.execute(
            "SELECT local_track_id, length(embedding) AS e, length(color_hist) AS h "
            "FROM vehicle_sightings ORDER BY sighting_id").fetchall()
        return {r["local_track_id"]: (r["e"], r["h"]) for r in rows}

    def test_blanks_only_old_rows_and_keeps_the_rows(self):
        self.assertEqual(self.db.expire_embeddings(3600, batch=2), 5)
        lengths = self._blob_lengths()
        for track in range(5):
            self.assertEqual(lengths[track], (0, 0))
        for track in range(5, 8):
            self.assertEqual(lengths[track], (1024, 512))
        self.assertEqual(self.db.get_traffic_stats()["total_sightings"], 8)
        self.assertEqual(len(self.db.get_vehicle_journey(0)), 1, "journeys still work without blobs")

    def test_resumes_from_watermark_and_is_idempotent(self):
        self.assertEqual(self.db.expire_embeddings(3600), 5)
        self.assertEqual(self.db.expire_embeddings(3600), 0)
        value = self.db.conn.execute(
            "SELECT value FROM maintenance_state WHERE key = 'embeddings_expired_upto'").fetchone()[0]
        self.assertEqual(value, 5)
        self.assertEqual(self._blob_lengths()[5], (1024, 512))

    def test_out_of_order_old_row_never_blanks_younger_rows(self):
        self.db.expire_embeddings(3600)
        self._add(99, age_s=7200)             # old timestamp, inserted after younger rows
        self.assertEqual(self.db.expire_embeddings(3600), 0, "waits behind the younger rows")
        lengths = self._blob_lengths()
        for track in (5, 6, 7):
            self.assertEqual(lengths[track], (1024, 512))
        self.assertEqual(lengths[99], (1024, 512))

    def test_new_old_rows_are_picked_up_on_later_passes(self):
        self.db.expire_embeddings(3600)
        self.db.conn.execute("DELETE FROM vehicle_sightings WHERE local_track_id >= 5")   # no young rows left
        self.db.conn.commit()
        self._add(50, age_s=7200)
        self._add(51, age_s=7200)
        self.assertEqual(self.db.expire_embeddings(3600), 2)

    def test_stop_request_ends_the_pass_early(self):
        calls = []
        n = self.db.expire_embeddings(3600, batch=1, should_stop=lambda: len(calls.append(1) or calls) > 2)
        self.assertEqual(n, 2)

    def test_maintenance_with_zero_ttl_does_nothing(self):
        self.assertEqual(SightingMaintenance(self.db, 0).run_once(), 0)
        self.assertEqual(self._blob_lengths()[0], (1024, 512))
        self.assertEqual(SightingMaintenance(self.db, 3600, pause_s=0).run_once(), 5)


class MaintenanceConstructionTest(unittest.TestCase):
    def test_server_style_positional_construction_keeps_working(self):
        db = Database(":memory:")
        try:
            m = SightingMaintenance(db, 3600.0, 300.0)       # exactly how backend/server.py builds it
            self.assertEqual((m.embedding_ttl_s, m.interval_s), (3600.0, 300.0))
            self.assertEqual(m.retention_days, 0.0, "retention is opt-in (SIGHTING_RETENTION_DAYS)")
            self.assertEqual(m.run_once(), 0)
            self.assertEqual(set(m.last_run), {"blanked", "expired_targets", "pruned_sightings",
                                               "pruned_handoffs", "pages_freed"})
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
