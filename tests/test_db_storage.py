"""
Storage upkeep: compact() (one-off VACUUM + incremental auto-vacuum), incremental_vacuum(),
the WAL size cap and PRAGMA optimize.
"""

import os
import sqlite3
import tempfile
import unittest
from unittest import mock

import backend.database as database
from backend.database import Database, COMPACT_MARKER

BLOB = b"\x01" * 1024


def fill(db, n, blob=True):
    emb, hist = (BLOB, BLOB[:512]) if blob else (b"", b"")
    for i in range(n):
        db.insert_vehicle_sighting(79 if i % 2 else 87, i, i // 2, "car", 0.9, (0, 0, 10, 10), 0.0, 0.0,
                                   emb, hist, "2026-10-01T00:%02d:%02dZ" % (i // 60 % 60, i % 60))


class CompactTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "c.sqlite")
        self.db = Database(self.path, auto_seed=True)
        fill(self.db, 1500)                                           # ~2.3 MB with blobs
        self.db.expire_embeddings(0.0)                                # blank them all: pages are now ~90% empty
        self.db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def marker(self):
        return self.db.conn.execute("SELECT value FROM maintenance_state WHERE key = ?", (COMPACT_MARKER,)).fetchone()

    def test_not_run_implicitly(self):
        self.assertIsNone(self.marker())
        self.assertEqual(self.db.conn.execute("PRAGMA auto_vacuum").fetchone()[0], 0)

    def test_compact_shrinks_converts_and_sets_the_marker(self):
        before_rows = self.db.conn.execute("SELECT COUNT(*) FROM vehicle_sightings").fetchone()[0]
        r = self.db.compact()
        self.assertTrue(r["compacted"], r)
        self.assertEqual(r["auto_vacuum"], 2)
        self.assertLess(r["after_bytes"], r["before_bytes"] // 2, "blanked pages must be reclaimed")
        self.assertEqual(r["reclaimed_bytes"], r["before_bytes"] - r["after_bytes"])
        self.assertIsNotNone(self.marker())
        self.assertEqual(self.db.conn.execute("PRAGMA auto_vacuum").fetchone()[0], 2)
        self.assertEqual(self.db.conn.execute("PRAGMA quick_check").fetchone()[0], "ok")
        # data and the trigger-maintained totals are intact
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM vehicle_sightings").fetchone()[0], before_rows)
        self.assertEqual(self.db.get_traffic_stats()["total_sightings"], before_rows)
        # and writing still works afterwards (triggers survived the VACUUM)
        fill(self.db, 5, blob=False)
        self.assertEqual(self.db.get_traffic_stats()["total_sightings"], before_rows + 5)

    def test_second_call_is_a_noop_until_forced(self):
        self.assertTrue(self.db.compact()["compacted"])
        again = self.db.compact()
        self.assertFalse(again["compacted"])
        self.assertTrue(again["skipped"])
        self.assertEqual(again["reason"], "already compacted")
        forced = self.db.compact(force=True)
        self.assertTrue(forced["compacted"])

    def test_fails_soft_when_the_database_is_busy(self):
        blocker = sqlite3.connect(self.path, timeout=1.0, isolation_level=None)
        blocker.execute("BEGIN IMMEDIATE")                            # holds the write lock
        try:
            with mock.patch.object(database, "COMPACT_BUSY_TIMEOUT_MS", 150):
                r = self.db.compact()
            self.assertFalse(r["compacted"])
            self.assertFalse(r["skipped"])
            self.assertIn("busy", r["reason"])
            self.assertIsNone(self.marker(), "an unfinished compaction must be retried next start")
        finally:
            blocker.execute("ROLLBACK")
            blocker.close()
        self.assertTrue(self.db.compact()["compacted"], "works once the lock is gone")

    def test_skips_when_the_disk_is_too_full(self):
        usage = mock.Mock(free=1024)
        with mock.patch.object(database.shutil, "disk_usage", return_value=usage):
            r = self.db.compact()
        self.assertFalse(r["compacted"])
        self.assertIn("disk", r["reason"])
        self.assertIsNone(self.marker())

    def test_in_memory_database_is_skipped(self):
        mem = Database(":memory:")
        try:
            r = mem.compact()
            self.assertFalse(r["compacted"])
            self.assertTrue(r["skipped"])
        finally:
            mem.close()


class IncrementalVacuumTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "v.sqlite")
        self.db = Database(self.path, auto_seed=True)
        fill(self.db, 1500)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def delete_all(self):
        self.db.prune_sightings(older_than_s=0)

    def test_returns_pages_after_compact(self):
        self.db.compact(force=True)
        fill(self.db, 1500)
        self.db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        size = os.path.getsize(self.path)
        self.delete_all()
        free = self.db.freelist_pages()
        self.assertGreater(free, 10)
        freed = self.db.incremental_vacuum(max_pages=5)
        self.assertEqual(freed, 5)
        total = freed
        while True:
            n = self.db.incremental_vacuum(max_pages=10_000)
            if n == 0:
                break
            total += n
        self.assertEqual(total, free)
        self.assertEqual(self.db.freelist_pages(), 0)
        self.db.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self.assertLess(os.path.getsize(self.path), size)

    def test_without_compact_it_does_nothing(self):
        self.delete_all()
        self.assertEqual(self.db.incremental_vacuum(max_pages=1000), 0)

    def test_zero_pages_requested(self):
        self.db.compact(force=True)
        self.assertEqual(self.db.incremental_vacuum(max_pages=0), 0)


class ConnectionSetupTest(unittest.TestCase):
    def test_wal_size_cap_is_applied_to_every_connection(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(os.path.join(tmp, "j.sqlite"), auto_seed=False)
            try:
                limit = db.conn.execute("PRAGMA journal_size_limit").fetchone()[0]
                self.assertEqual(limit, database.DB_JOURNAL_SIZE_LIMIT_BYTES)
                self.assertGreater(limit, 0)
            finally:
                db.close()

    def test_optimize_is_cheap_and_repeatable(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = Database(os.path.join(tmp, "o.sqlite"), auto_seed=True)
            try:
                fill(db, 300)
                db.optimize()
                db.optimize()
                self.assertEqual(db.conn.execute("PRAGMA analysis_limit").fetchone()[0], 1000)
            finally:
                db.close()


if __name__ == "__main__":
    unittest.main()
