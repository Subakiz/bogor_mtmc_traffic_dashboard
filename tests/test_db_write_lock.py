"""
tests/test_db_write_lock.py
Regression: a failed write must not leave its transaction open. SQLite keeps the write lock for
an open transaction, so one swallowed FOREIGN KEY error used to block every later insert (the
rolling tracker froze on its first sighting after the dashboard called /api/handoff/predict).
"""

import os
import sqlite3
import tempfile
import threading
import unittest

from fastapi.testclient import TestClient

from backend.database import Database


class TestFailedWriteReleasesLock(unittest.TestCase):
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
                conn.execute("INSERT INTO handoff_events (global_vehicle_id, from_camera_id, to_camera_id, "
                             "confidence, timestamp_iso) VALUES (1, 79, 87, 0.5, '2026-01-01T00:00:00Z')")
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

    def test_foreign_key_failure_rolls_back(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.update_active_target(global_vehicle_id=14, current_camera_id=79,
                                         last_sighting_id=0, status="tracking")
        self.assertFalse(self.db.conn.in_transaction)
        self.assertTrue(self._other_thread_can_write(), "write lock leaked after a failed write")

    def test_failed_sighting_insert_rolls_back(self):
        with self.assertRaises(sqlite3.IntegrityError):
            self.db.insert_vehicle_sighting(
                camera_id=999999, local_track_id=1, global_vehicle_id=1, vehicle_class="car",
                confidence=0.9, bbox=(0, 0, 10, 10), heading_deg=0.0, speed_px_s=0.0,
                embedding=b"\0" * 1024, color_hist=b"\0" * 512, timestamp_iso="2026-01-01T00:00:00Z",
            )
        self.assertTrue(self._other_thread_can_write())


class TestHandoffPredictDoesNotWriteForUnknownVehicle(unittest.TestCase):
    def test_placeholder_vehicle_records_nothing_and_holds_no_lock(self):
        import backend.server as server

        client = TestClient(server.app)
        before = server.db.get_traffic_stats()["total_handoffs"]
        res = client.post("/api/handoff/predict",
                          json={"camera_id": 79, "exit_direction": "South", "vehicle_id": 987654321})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(server.db.get_traffic_stats()["total_handoffs"], before)
        self.assertIsNone(server.db.get_active_target(987654321))


if __name__ == "__main__":
    unittest.main()
