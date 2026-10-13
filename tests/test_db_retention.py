"""
Opt-in retention (prune_sightings / prune_handoff_events), active-target expiry, and the
SightingMaintenance passes that drive them.
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
        self.db = Database(os.path.join(self.tmp.name, "r.sqlite"), auto_seed=True, active_target_ttl_s=1800)

    def tearDown(self):
        self.db.close()
        self.tmp.cleanup()

    def add(self, age_s, track, gid=None, cam=79, cls="car"):
        return self.db.insert_vehicle_sighting(cam, track, gid if gid is not None else track, cls, 0.9,
                                               (0, 0, 10, 10), 0.0, 0.0, b"", b"", iso(age_s))

    def n_sightings(self):
        return self.db.conn.execute("SELECT COUNT(*) FROM vehicle_sightings").fetchone()[0]

    def age_target(self, gid, age_s):
        with self.db.conn:
            self.db.conn.execute("UPDATE active_targets SET updated_ts = ? WHERE global_vehicle_id = ?",
                                 (time.time() - age_s, gid))


class RetentionTest(DbCase):
    def setUp(self):
        super().setUp()
        self.old_ids = [self.add(10 * DAY + i, track=i) for i in range(10)]
        self.new_ids = [self.add(3600 + i, track=100 + i) for i in range(4)]
        for age in (10 * DAY, 10 * DAY, 60):
            self.db.record_handoff_event(1, 79, 87, 0.9, iso(age))
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
        gv_before = self.db.conn.execute("SELECT COUNT(*) FROM global_vehicles").fetchone()[0]
        self.db.prune_sightings(7 * DAY, batch=4)
        self.db.prune_handoff_events(7 * DAY, batch=1)
        self.assertEqual(self.n_sightings(), 4)
        self.assertEqual(self.db.get_traffic_stats(), before, "totals count inserts, not stored rows")
        self.assertEqual(self.db.get_camera_vehicle_counts(), counts_before)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM global_vehicles").fetchone()[0], gv_before)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM observation_windows").fetchone()[0], 1)

    def test_handoff_events_are_pruned_in_batches(self):
        self.assertEqual(self.db.prune_handoff_events(7 * DAY, batch=1), 2)
        self.assertEqual(self.db.conn.execute("SELECT COUNT(*) FROM handoff_events").fetchone()[0], 1)
        self.assertEqual(self.db.get_traffic_stats()["total_handoffs"], 3)

    def test_a_live_target_pins_its_sighting_without_breaking_foreign_keys(self):
        pinned = self.old_ids[3]
        self.db.update_active_target(global_vehicle_id=3, current_camera_id=79, last_sighting_id=pinned)
        removed = self.db.prune_sightings(7 * DAY, batch=4)              # must not raise IntegrityError
        self.assertEqual(removed, 9)
        self.assertEqual(self.n_sightings(), 5)
        self.assertIsNotNone(self.db.get_active_target(3))
        self.assertTrue(self.db.is_active_target(3))

    def test_an_expired_target_releases_its_sighting(self):
        old = self.old_ids[3]
        self.db.update_active_target(global_vehicle_id=3, current_camera_id=79, last_sighting_id=old)
        self.age_target(3, 5000)
        self.assertEqual(self.db.expire_active_targets(), 1)
        self.assertEqual(self.db.prune_sightings(7 * DAY), 10)
        self.assertIsNone(self.db.get_active_target(3), "the stale target row goes with its sighting")

    def test_an_expired_target_on_a_young_sighting_is_kept(self):
        self.db.update_active_target(global_vehicle_id=100, current_camera_id=79, last_sighting_id=self.new_ids[0])
        self.age_target(100, 5000)
        self.db.expire_active_targets()
        self.db.prune_sightings(7 * DAY)
        self.assertEqual(self.db.get_active_target(100)["status"], "expired")

    def test_stop_request_ends_the_pass_early(self):
        calls = []
        removed = self.db.prune_sightings(7 * DAY, batch=2, should_stop=lambda: calls.append(1) or len(calls) > 2)
        self.assertEqual(removed, 4)

    def test_out_of_order_ids_are_still_pruned_by_timestamp(self):
        late_old = self.add(30 * DAY, track=500)                          # old timestamp, highest id
        self.assertEqual(self.db.prune_sightings(7 * DAY, batch=100), 11)
        self.assertNotIn(late_old, [r[0] for r in self.db.conn.execute("SELECT sighting_id FROM vehicle_sightings")])


class ActiveTargetExpiryTest(DbCase):
    def setUp(self):
        super().setUp()
        self.sid = self.add(10, track=1, gid=7)

    def follow(self, gid=7):
        self.db.update_active_target(global_vehicle_id=gid, current_camera_id=79, last_sighting_id=self.sid)

    def test_fresh_target_is_active(self):
        self.follow()
        self.assertTrue(self.db.is_active_target(7))
        self.assertEqual([t["global_vehicle_id"] for t in self.db.list_active_targets()], [7])
        self.assertEqual(self.db.get_traffic_stats()["active_targets_count"], 1)

    def test_stale_tracking_row_is_inactive_before_maintenance_even_runs(self):
        self.follow()
        self.age_target(7, 1801)
        self.assertEqual(self.db.get_active_target(7)["status"], "tracking")        # the row itself is untouched...
        self.assertFalse(self.db.is_active_target(7))                                # ...but no longer counts
        self.assertEqual(self.db.list_active_targets(), [])
        self.assertEqual(self.db.get_traffic_stats()["active_targets_count"], 0)

    def test_expire_marks_only_stale_tracking_rows(self):
        self.follow(7)
        sid2 = self.add(5, track=2, gid=8)
        self.db.update_active_target(global_vehicle_id=8, current_camera_id=79, last_sighting_id=sid2)
        self.age_target(7, 3600)
        self.assertEqual(self.db.expire_active_targets(), 1)
        self.assertEqual(self.db.get_active_target(7)["status"], "expired")
        self.assertEqual(self.db.get_active_target(8)["status"], "tracking")
        self.assertEqual(self.db.expire_active_targets(), 0, "idempotent")
        self.assertEqual([t["global_vehicle_id"] for t in self.db.list_active_targets()], [8])

    def test_every_update_refreshes_the_clock(self):
        self.follow()
        self.age_target(7, 1700)
        self.follow()                                                               # tracker saw it again
        self.assertEqual(self.db.expire_active_targets(), 0)
        self.assertTrue(self.db.is_active_target(7))

    def test_relocking_an_expired_target_reactivates_it(self):
        self.follow()
        self.age_target(7, 4000)
        self.db.expire_active_targets()
        self.assertFalse(self.db.is_active_target(7))
        self.db.update_active_target(global_vehicle_id=7, current_camera_id=79, last_sighting_id=self.sid,
                                     locked_at_iso="2026-12-01T00:00:00Z")
        target = self.db.get_active_target(7)
        self.assertEqual((target["status"], target["locked_at_iso"]), ("tracking", "2026-12-01T00:00:00Z"))
        self.assertTrue(self.db.is_active_target(7))

    def test_ttl_zero_means_targets_never_expire(self):
        db = Database(os.path.join(self.tmp.name, "forever.sqlite"), auto_seed=True, active_target_ttl_s=0)
        try:
            sid = db.insert_vehicle_sighting(79, 1, 7, "car", 0.9, (0, 0, 1, 1), 0, 0, b"", b"", iso(10))
            db.update_active_target(7, 79, sid, locked_at_iso="2001-01-01T00:00:00Z")
            with db.conn:
                db.conn.execute("UPDATE active_targets SET updated_ts = 1")
            self.assertEqual(db.expire_active_targets(), 0)
            self.assertTrue(db.is_active_target(7))
        finally:
            db.close()

    def test_rows_without_updated_ts_fall_back_to_the_lock_time(self):
        with self.db.conn:
            self.db.conn.execute(
                "INSERT INTO active_targets (global_vehicle_id, current_camera_id, last_sighting_id, locked_at_iso, status)"
                " VALUES (9, 79, ?, ?, 'tracking')", (self.sid, iso(60)))
            self.db.conn.execute(
                "INSERT INTO active_targets (global_vehicle_id, current_camera_id, last_sighting_id, locked_at_iso, status)"
                " VALUES (10, 79, ?, '2026-10-11T12:58:40.078715+00:00', 'tracking')", (self.sid,))
        self.assertTrue(self.db.is_active_target(9))
        self.assertFalse(self.db.is_active_target(10))
        self.assertEqual(self.db.expire_active_targets(), 1)


class MaintenancePassTest(DbCase):
    def test_one_pass_expires_prunes_and_reports(self):
        old = [self.add(10 * DAY + i, track=i) for i in range(6)]
        self.add(60, track=50, gid=50)
        sid = self.add(30, track=60, gid=60)
        self.db.update_active_target(global_vehicle_id=60, current_camera_id=79, last_sighting_id=sid)
        self.db.update_active_target(global_vehicle_id=2, current_camera_id=79, last_sighting_id=old[2])
        self.age_target(2, 7200)
        self.db.record_handoff_event(1, 79, 87, 0.9, iso(10 * DAY))
        stats_before = self.db.get_traffic_stats()

        m = SightingMaintenance(self.db, embedding_ttl_s=0, retention_days=7, active_target_ttl_s=1800,
                                batch=2, pause_s=0.0)
        self.assertEqual(m.run_once(), 0)
        self.assertEqual(m.last_run["expired_targets"], 1)
        self.assertEqual(m.last_run["pruned_sightings"], 6)
        self.assertEqual(m.last_run["pruned_handoffs"], 1)
        self.assertEqual(self.n_sightings(), 2)
        self.assertEqual((m.total_pruned, m.total_targets_expired), (6, 1))
        self.assertTrue(self.db.is_active_target(60))
        self.assertEqual(self.db.get_traffic_stats()["total_sightings"], stats_before["total_sightings"])

    def test_retention_off_by_default_keeps_everything(self):
        self.add(400 * DAY, track=1)
        m = SightingMaintenance(self.db, embedding_ttl_s=3600, retention_days=0, pause_s=0.0)
        m.run_once()
        self.assertEqual(self.n_sightings(), 1)
        self.assertEqual(m.last_run["pruned_sightings"], 0)

    def test_config_defaults_come_from_backend_config(self):
        import backend.config as config
        m = SightingMaintenance(self.db, embedding_ttl_s=3600)
        self.assertEqual(m.retention_days, config.SIGHTING_RETENTION_DAYS)
        self.assertEqual(m.active_target_ttl_s, config.ACTIVE_TARGET_TTL_S)
        self.assertEqual(config.SIGHTING_RETENTION_DAYS, 0.0, "retention must be opt-in")

    def test_a_failing_step_does_not_block_the_others(self):
        sid = self.add(10, track=1, gid=7)
        self.db.update_active_target(7, 79, sid)
        self.age_target(7, 7200)
        self.add(10 * DAY, track=2)
        m = SightingMaintenance(self.db, embedding_ttl_s=0, retention_days=7, pause_s=0.0)
        with mock.patch.object(self.db, "prune_sightings", side_effect=sqlite3.OperationalError("database is locked")):
            with self.assertLogs("maintenance", level="WARNING") as logs:
                m.run_once()
        self.assertIn("prune sightings", logs.output[0])
        self.assertEqual(m.last_run["expired_targets"], 1)
        self.assertEqual(m.last_run["pruned_sightings"], 0)

    def test_pass_returns_freed_pages_once_compacted(self):
        for i in range(1500):
            self.db.insert_vehicle_sighting(79, i, i, "car", 0.9, (0, 0, 1, 1), 0, 0, b"\x01" * 1024, b"\x02" * 512, iso(10 * DAY))
        self.db.compact(force=True)
        m = SightingMaintenance(self.db, embedding_ttl_s=0, retention_days=7, batch=500, pause_s=0.0,
                                vacuum_pages_per_run=100, vacuum_chunk_pages=40)
        m.run_once()
        self.assertEqual(self.n_sightings(), 0)
        self.assertEqual(m.last_run["pages_freed"], 100, "bounded by the per-run page budget")
        m.run_once()
        self.assertEqual(m.last_run["pages_freed"], 100)
        self.assertEqual(m.total_pages_freed, 200)


if __name__ == "__main__":
    unittest.main()
