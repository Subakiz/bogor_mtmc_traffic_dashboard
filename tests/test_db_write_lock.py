"""
tests/test_db_write_lock.py
Regression: a failed write must not leave its transaction open. SQLite keeps the write lock for
an open transaction, so one swallowed FOREIGN KEY error used to block every later insert (the
rolling tracker froze on its first sighting after one failed write).
"""

import os
import sqlite3
import tempfile
import threading
import unittest

from backend.database import Database


class _LockTestBase(unittest.TestCase):
    __test__ = False

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "lock.sqlite")
        self.db = Database(self.path, auto_seed=True)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def _other_thread_can_write(self) -> bool:
        result = {}

        def write():
            conn = sqlite3.connect(self.path, timeout=1.0)
            try:
                conn.execute("INSERT INTO observation_windows (camera_id, start_ts, end_ts, start_iso, "
                             "observed_s) VALUES (79, 0, 10, '2026-01-01T00:00:00Z', 10)")
                conn.commit()
                result["ok"] = True
            except sqlite3.OperationalError as exc:
                result["error"] = str(exc)
            finally:
                conn.close()

        t = threading.Thread(target=write)
        t.start()
        t.join()
        return result.get("ok", False)


class TestFailedWriteReleasesLock(_LockTestBase):
    __test__ = True

    def test_foreign_key_failure_rolls_back(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.insert_observation_window({"camera_id": 999999, "start_ts": 0.0, "end_ts": 10.0,
                                               "start_iso": "2026-01-01T00:00:00Z", "observed_s": 10.0})
        self.assertFalse(self.db.conn.in_transaction)
        self.assertTrue(self._other_thread_can_write(), "write lock leaked after a failed write")

    def test_failed_sighting_insert_rolls_back(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.insert_vehicle_sighting(
                camera_id=999999, local_track_id=1, vehicle_class="car",
                confidence=0.9, bbox=(0, 0, 10, 10), heading_deg=0.0, speed_px_s=0.0,
                timestamp_iso="2026-01-01T00:00:00Z",
            )
        self.assertTrue(self._other_thread_can_write())


class TestStorageOperationsReleaseTheLock(_LockTestBase):
    __test__ = True

    def _add(self, track, age_days):
        from datetime import datetime, timedelta, timezone
        ts = (datetime.now(timezone.utc) - timedelta(days=age_days)).strftime("%Y-%m-%dT%H:%M:%SZ")
        return self.db.insert_vehicle_sighting(79, track, "car", 0.9, (0, 0, 10, 10), 0.0, 0.0, ts)

    def test_prune_leaves_no_transaction_open(self):
        self._add(1, age_days=30)
        self._add(2, age_days=30)
        self.assertEqual(self.db.prune_sightings(older_than_s=86400, batch=1), 2)
        self.assertFalse(self.db.conn.in_transaction)
        self.assertTrue(self._other_thread_can_write())

    def test_vacuum_leaves_no_transaction_open(self):
        self._add(1, age_days=0)
        self.db.compact(force=True)
        self.db.incremental_vacuum(10)
        self.assertFalse(self.db.conn.in_transaction)
        self.assertTrue(self._other_thread_can_write())


if __name__ == "__main__":
    unittest.main()
