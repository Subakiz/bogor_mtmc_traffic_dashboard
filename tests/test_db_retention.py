"""
Opt-in retention (prune_sightings) and the SightingMaintenance passes that drive it.
"""

import os
import sqlite3
import tempfile
import time
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import backend.database as database
from backend.database import Database
from backend.maintenance import SightingMaintenance

DAY = 86400.0


def iso(age_s):
    return (datetime.now(timezone.utc) - timedelta(seconds=age_s)).strftime("%Y-%m-%dT%H:%M:%SZ")


class DbCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = Database(os.path.join(self.tmp.name, "r.sqlite"), auto_seed=True)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def add(self, age_s, track, cam=79, cls="car"):
        return self.db.insert_vehicle_sighting(cam, track, cls, 0.9, (0, 0, 10, 10), 0.0, 0.0, iso(age_s))

    def n_sightings(self):
        return self.db.conn.execute("SELECT COUNT(*) FROM vehicle_sightings").fetchone()[0]

class RetentionTest(DbCase):
    def setUp(self):
        super().setUp()
        self.old_ids = [self.add(10 * DAY + i, track=i) for i in range(10)]
        self.new_ids = [self.add(3600 + i, track=100 + i) for i in range(4)]
        self.db.insert_observation_window({"camera_id": 79, "start_ts": time.time() - 10 * DAY - 15,
                                           "end_ts": time.time() - 10 * DAY, "start_iso": "x", "observed_s": 15.0,
                                           "vehicles": 3, "kind": "visit"})

    def test_deletes_old_rows_in_batches_and_keeps_young_ones(self):
        deletes = []
        self.db.conn.set_trace_callback(lambda sql: deletes.append(sql) if sql.startswith("DELETE FROM vehicle_sightings") else None)
        with mock.patch.object(database.time, "sleep") as sleep:
            removed = self.db.prune_sightings(7 * DAY, batch=3, pause_s=0.01)
        self.db.conn.set_trace_callback(None)
        self.assertEqual(removed, 10)
        self.assertEqual(len(deletes), 4, "3 + 3 + 3 + 1 rows -> four short transactions")
        self.assertEqual(sleep.call_count, 3, "a pause after every full batch")
        remaining = [r[0] for r in self.db.conn.execute("SELECT sighting_id FROM vehicle_sightings ORDER BY sighting_id")]
        self.assertEqual(remaining, self.new_ids)
        self.assertEqual(self.db.prune_sightings(7 * DAY, batch=3), 0, "idempotent")

    def test_exact_multiple_of_batch_size_terminates(self):
        self.assertEqual(self.db.prune_sightings(7 * DAY, batch=5), 10)
        self.assertEqual(self.n_sightings(), 4)

    def test_all_time_totals_and_volume_history_are_unchanged(self):
        before = self.db.get_traffic_stats()
        counts_before = self.db.get_camera_vehicle_counts()
        self.db.prune_sightings(7 * DAY, batch=4)
        self.assertEqual(self.n_sightings(), 4)
        self.assertEqual(self.db.get_traffic_stats(), before, "totals count inserts, not stored rows")
        self.assertEqual(self.db.get_camera_vehicle_counts(), counts_before)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM observation_windows").fetchone()[0], 1)

    def test_stop_request_ends_the_pass_early(self):
        calls = []
        removed = self.db.prune_sightings(7 * DAY, batch=2, should_stop=lambda: calls.append(1) or len(calls) > 2)
        self.assertEqual(removed, 4)

    def test_out_of_order_ids_are_still_pruned_by_timestamp(self):
        late_old = self.add(30 * DAY, track=500)                          # old timestamp, highest id
        self.assertEqual(self.db.prune_sightings(7 * DAY, batch=100), 11)
        self.assertNotIn(late_old, [r[0] for r in self.db.conn.execute("SELECT sighting_id FROM vehicle_sightings")])


class MaintenancePassTest(DbCase):
    def test_one_pass_prunes_and_reports(self):
        for i in range(6):
            self.add(10 * DAY + i, track=i)
        self.add(60, track=50)
        self.add(30, track=60)
        stats_before = self.db.get_traffic_stats()

        m = SightingMaintenance(self.db, retention_days=7, batch=2, pause_s=0.0)
        self.assertEqual(m.run_once(), 6)
        self.assertEqual(m.last_run["pruned_sightings"], 6)
        self.assertEqual(self.n_sightings(), 2)
        self.assertEqual(m.total_pruned, 6)
        self.assertEqual(self.db.get_traffic_stats()["total_sightings"], stats_before["total_sightings"])

    def test_retention_off_by_default_keeps_everything(self):
        self.add(400 * DAY, track=1)
        m = SightingMaintenance(self.db, retention_days=0, pause_s=0.0)
        m.run_once()
        self.assertEqual(self.n_sightings(), 1)
        self.assertEqual(m.last_run["pruned_sightings"], 0)

    def test_config_defaults_come_from_backend_config(self):
        import backend.config as config
        m = SightingMaintenance(self.db)
        self.assertEqual(m.retention_days, config.SIGHTING_RETENTION_DAYS)
        self.assertEqual(config.SIGHTING_RETENTION_DAYS, 0.0, "retention must be opt-in")

    def test_a_failing_step_does_not_block_the_others(self):
        self.add(10 * DAY, track=2)
        self.db.compact(force=True)
        m = SightingMaintenance(self.db, retention_days=7, pause_s=0.0)
        with mock.patch.object(self.db, "prune_sightings", side_effect=sqlite3.OperationalError("database is locked")), \
                mock.patch.object(self.db, "incremental_vacuum", wraps=self.db.incremental_vacuum) as vacuum:
            with self.assertLogs("maintenance", level="WARNING") as logs:
                m.run_once()
        self.assertIn("prune sightings", logs.output[0])
        self.assertEqual(m.last_run["pruned_sightings"], 0)
        self.assertTrue(vacuum.called, "the vacuum step still runs")

    def test_pass_returns_freed_pages_once_compacted(self):
        for i in range(1500):
            self.db.insert_vehicle_sighting(79, i, "car", 0.9, (0, 0, 1, 1), 0, 0, iso(10 * DAY) + "x" * 1500)
        self.db.compact(force=True)
        m = SightingMaintenance(self.db, retention_days=7, batch=500, pause_s=0.0,
                                vacuum_pages_per_run=100, vacuum_chunk_pages=40)
        m.run_once()
        self.assertEqual(self.n_sightings(), 0)
        self.assertEqual(m.last_run["pages_freed"], 100, "bounded by the per-run page budget")
        m.run_once()
        self.assertEqual(m.last_run["pages_freed"], 100)
        self.assertEqual(m.total_pages_freed, 200)


if __name__ == "__main__":
    unittest.main()
