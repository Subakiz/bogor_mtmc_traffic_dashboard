"""
backend/dataset.py: bucketing, proration, vph math, gap filling, local-time features.
Synthetic databases in temp dirs; WIB = UTC+7.
"""

import hashlib
import sqlite3
import unittest

from backend import dataset
from car_tracker.congestion import congestion_label
from tests.test_dataset_fixtures import CAMERAS, DatasetTestCase, make_analytics_db, make_main_db, wib


def by_ts(rows, camera_id=None):
    return {r["bucket_start_local"]: r for r in rows if camera_id is None or r["camera_id"] == camera_id}


class TestBucketAlignment(DatasetTestCase):
    def test_buckets_align_to_wib_wall_clock(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:07:00"), "end": wib("2026-10-11T08:09:00"),
                              "vehicles": 4}])
        rows, stream = self.dataset("2026-10-11T08:00", "2026-10-11T09:00", bucket_minutes=15, fill_gaps=True)
        self.assertEqual([r["bucket_start_local"] for r in rows],
                         [f"2026-10-11T08:{m:02d}:00+07:00" for m in (0, 15, 30, 45)])
        self.assertEqual([r["bucket_start_utc"] for r in rows],
                         [f"2026-10-11T01:{m:02d}:00Z" for m in (0, 15, 30, 45)])
        self.assertEqual(rows[0]["bucket_start_ts"], wib("2026-10-11T08:00:00"))
        self.assertEqual({r["camera_id"] for r in rows}, {167})
        self.assertEqual(stream.camera_ids, [167])

    def test_hourly_and_daily_buckets_start_at_local_boundaries(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:07:00"), "end": wib("2026-10-11T08:09:00"),
                              "vehicles": 1}])
        rows, _ = self.dataset("2026-10-11T00:00", "2026-10-12T00:00", bucket_minutes=60, fill_gaps=True)
        self.assertEqual(len(rows), 24)
        self.assertEqual(rows[0]["bucket_start_utc"], "2026-10-10T17:00:00Z")      # WIB midnight
        self.assertEqual([r["hour"] for r in rows], list(range(24)))
        rows, _ = self.dataset("2026-10-11", "2026-10-13", bucket_minutes=1440, fill_gaps=True)
        self.assertEqual([r["bucket_start_local"] for r in rows],
                         ["2026-10-11T00:00:00+07:00", "2026-10-12T00:00:00+07:00"])

    def test_time_features_are_local_not_utc(self):
        # 20:00 UTC on Sunday 11 Oct is 03:00 WIB on Monday 12 Oct
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-12T03:01:00"), "end": wib("2026-10-12T03:02:00"),
                              "vehicles": 1}])
        rows, _ = self.dataset("2026-10-12T03:00", "2026-10-12T03:15", bucket_minutes=15)
        r = rows[0]
        self.assertEqual(r["bucket_start_utc"], "2026-10-11T20:00:00Z")
        self.assertEqual((r["hour"], r["minute_of_day"], r["day_of_week"], r["is_weekend"]), (3, 180, 0, 0))
        self.assertAlmostEqual(r["sin_hour"], 0.70711, places=4)   # 03:00 = an eighth of the day (45 degrees)
        self.assertAlmostEqual(r["cos_hour"], 0.70711, places=4)
        self.assertAlmostEqual(r["sin_dow"], 0.0, places=4)        # Monday
        self.assertAlmostEqual(r["cos_dow"], 1.0, places=4)

    def test_weekend_flag_follows_the_local_date(self):
        # Saturday 17 Oct 02:00 WIB is still Friday 16 Oct 19:00 UTC
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-17T02:01:00"), "end": wib("2026-10-17T02:02:00"),
                              "vehicles": 1}])
        rows, _ = self.dataset("2026-10-17T02:00", "2026-10-17T02:15")
        self.assertEqual((rows[0]["day_of_week"], rows[0]["is_weekend"]), (5, 1))

    def test_only_whole_buckets_inside_the_range_are_produced(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:00:00"), "end": wib("2026-10-11T10:00:00"),
                              "vehicles": 10}])
        rows, stream = self.dataset("2026-10-11T08:05", "2026-10-11T09:50", bucket_minutes=15)
        self.assertEqual(rows[0]["bucket_start_local"], "2026-10-11T08:15:00+07:00")
        self.assertEqual(rows[-1]["bucket_start_local"], "2026-10-11T09:30:00+07:00")      # ends 09:45 <= 09:50
        self.assertEqual(len(rows), 6)
        self.assertEqual(stream.first_bucket_ts, wib("2026-10-11T08:15:00"))
        self.assertEqual(stream.end_bucket_ts, wib("2026-10-11T09:45:00"))

    def test_empty_or_inverted_ranges(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:00:00"), "end": wib("2026-10-11T08:10:00"),
                              "vehicles": 1}])
        self.assertEqual(self.dataset("2026-10-11T08:00", "2026-10-11T08:10", bucket_minutes=15)[0], [])
        self.assertEqual(self.dataset("2026-10-11T08:00", "2026-10-11T08:00")[0], [])
        with self.assertRaises(ValueError):
            self.dataset("2026-10-11T09:00", "2026-10-11T08:00")

    def test_bucket_minutes_must_divide_a_day(self):
        for bad in (0, -5, 7, 0.5, 45 * 60 + 1, "x"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                dataset.bucket_seconds(bad)
        self.assertEqual([dataset.bucket_seconds(m) for m in (1, 5, 15, 30, 60, 1440)],
                         [60, 300, 900, 1800, 3600, 86400])


class TestParseTime(unittest.TestCase):
    def test_accepts_iso_naive_offset_utc_and_epoch(self):
        expected = wib("2026-10-11T00:00:00")
        for value in ("2026-10-11", "2026-10-11T00:00", "2026-10-11T00:00:00+07:00", "2026-10-10T17:00:00Z",
                      "2026-10-10T17:00:00+00:00", expected, int(expected), str(int(expected)), f"{expected:.1f}"):
            self.assertEqual(dataset.parse_time(value), expected, msg=repr(value))

    def test_rejects_nonsense(self):
        for bad in ("yesterday", "", None, True, 1.8e12, "2026-13-45"):
            with self.assertRaises(ValueError, msg=repr(bad)):
                dataset.parse_time(bad)


class TestProrationAndRates(DatasetTestCase):
    def test_window_spanning_a_bucket_edge_is_prorated_by_overlap(self):
        # 08:14:00 - 08:16:00 (120 s observed), 12 vehicles: 1 minute falls in each of two buckets
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:14:00"), "end": wib("2026-10-11T08:16:00"),
                              "vehicles": 12, "cars": 8, "motos": 4, "pos": 9, "neg": 3, "line": 12}])
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:30", bucket_minutes=15)
        a, b = rows
        self.assertEqual((a["bucket_start_local"], b["bucket_start_local"]),
                         ("2026-10-11T08:00:00+07:00", "2026-10-11T08:15:00+07:00"))
        for r in (a, b):
            self.assertAlmostEqual(r["observed_s"], 60.0, places=3)
            self.assertAlmostEqual(r["crossings"], 6.0, places=3)
            self.assertEqual(r["n_windows"], 1)
            self.assertAlmostEqual(r["vph"], 360.0, places=2)
            self.assertAlmostEqual(r["vph_car"], 240.0, places=2)
            self.assertAlmostEqual(r["vph_motorcycle"], 120.0, places=2)
            self.assertAlmostEqual(r["vph_dir_pos"], 270.0, places=2)
            self.assertAlmostEqual(r["vph_dir_neg"], 90.0, places=2)
        self.assertAlmostEqual(a["crossings"] + b["crossings"], 12.0, places=3)         # nothing lost or duplicated

    def test_unequal_split_and_conservation_over_many_buckets(self):
        # 08:10 - 09:20 = 70 min over five 15-minute buckets; 140 vehicles
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:10:00"), "end": wib("2026-10-11T09:20:00"),
                              "vehicles": 140, "observed": 4200}])
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:30", bucket_minutes=15)
        self.assertEqual([round(r["crossings"], 3) for r in rows], [10.0, 30.0, 30.0, 30.0, 30.0, 10.0])
        self.assertAlmostEqual(sum(r["crossings"] for r in rows), 140.0, places=2)
        self.assertAlmostEqual(sum(r["observed_s"] for r in rows), 4200.0, places=2)
        self.assertTrue(all(abs(r["vph"] - 140 * 3600 / 4200) < 0.01 for r in rows))

    def test_vph_is_crossings_per_observed_second_not_per_bucket_second(self):
        # two visits in one bucket: 20 s with 5 vehicles and 40 s with 4 -> 9 crossings in 60 s watched
        self.build(CAMERAS, [
            {"camera_id": 167, "start": wib("2026-10-11T08:01:00"), "end": wib("2026-10-11T08:01:20"), "vehicles": 5},
            {"camera_id": 167, "start": wib("2026-10-11T08:05:00"), "end": wib("2026-10-11T08:05:40"), "vehicles": 4,
             "kind": "focus"}])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15", bucket_minutes=15)
        self.assertEqual((r["observed_s"], r["crossings"], r["n_windows"]), (60.0, 9.0, 2))
        self.assertAlmostEqual(r["vph"], 540.0, places=2)                       # 9 * 3600 / 60, not 9 * 4
        self.assertAlmostEqual(r["coverage_ratio"], 60 / 900, places=4)
        self.assertAlmostEqual(r["focus_share"], 40 / 60, places=4)
        self.assertFalse(r["low_sample"])                                         # 60 s is not below the threshold

    def test_low_sample_flag_and_threshold(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:01:00"), "end": wib("2026-10-11T08:01:30"),
                              "vehicles": 2}])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15")
        self.assertTrue(r["low_sample"])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15", low_sample_s=30)
        self.assertFalse(r["low_sample"])

    def test_quiet_observed_bucket_is_zero_flow_not_missing(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:01:00"), "end": wib("2026-10-11T08:01:30"),
                              "vehicles": 0}])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15")
        self.assertEqual((r["crossings"], r["vph"], r["vph_car"]), (0.0, 0.0, 0.0))
        self.assertIsNone(r["line_share"])
        self.assertIsNone(r["counting_mode"])

    def test_counting_mode_share(self):
        self.build(CAMERAS, [
            {"camera_id": 167, "start": wib("2026-10-11T08:01:00"), "end": wib("2026-10-11T08:02:00"),
             "vehicles": 3, "line": 3, "passage": 0},
            {"camera_id": 131, "start": wib("2026-10-11T08:01:00"), "end": wib("2026-10-11T08:02:00"),
             "vehicles": 4, "line": 1, "passage": 3}])
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15")
        a, b = {r["camera_id"]: r for r in rows}[167], {r["camera_id"]: r for r in rows}[131]
        self.assertEqual((a["line_share"], a["counting_mode"]), (1.0, "line"))
        self.assertEqual((b["line_share"], b["counting_mode"]), (0.25, "passage"))

    def test_volume_basis_is_always_crossings(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:01:00"), "end": wib("2026-10-11T08:02:00"),
                              "vehicles": 3}], analytics=[
            {"camera_id": 167, "end": wib("2026-10-11T08:02:00"), "seconds": 60, "basis": "tracks", "volume": 99}])
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:00", fill_gaps=True)
        self.assertEqual({r["volume_basis"] for r in rows}, {"crossings"})
        self.assertEqual(rows[0]["crossings"], 3.0, "analytics (track-based) volume must not leak into the dataset")

    def test_window_with_zero_length_lands_in_its_bucket(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:20:00"), "end": wib("2026-10-11T08:20:00"),
                              "vehicles": 2, "observed": 5.0}])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:00")
        self.assertEqual((r["bucket_start_local"], r["crossings"], r["observed_s"]), ("2026-10-11T08:15:00+07:00", 2.0, 5.0))

    def test_chunking_does_not_change_the_result(self):
        windows = [{"camera_id": cam, "start": wib("2026-10-11T08:00:00") + i * 137, "end": wib("2026-10-11T08:00:00") + i * 137 + 61,
                    "vehicles": (i * 7) % 11} for cam in (167, 131) for i in range(60)]
        self.build(CAMERAS, windows, analytics=[
            {"camera_id": 167, "end": wib("2026-10-11T08:00:00") + i * 90, "seconds": 60, "congestion": (i % 10) / 10}
            for i in range(80)])
        whole, _ = self.dataset("2026-10-11T08:00", "2026-10-11T11:00", bucket_minutes=5, fill_gaps=True)
        tiny, _ = self.dataset("2026-10-11T08:00", "2026-10-11T11:00", bucket_minutes=5, fill_gaps=True, chunk_buckets=1)
        by_time, _ = self.dataset("2026-10-11T08:00", "2026-10-11T11:00", bucket_minutes=5, fill_gaps=True,
                                  order="time", chunk_buckets=4)
        self.assertEqual(whole, tiny)
        self.assertEqual(sorted(whole, key=lambda r: (r["bucket_start_ts"], r["camera_id"])), by_time)
        self.assertEqual(len(whole), 2 * 36)
        total = sum(r["crossings"] or 0 for r in whole)
        self.assertAlmostEqual(total, sum(w["vehicles"] for w in windows), places=2)


class TestGapsAndCameras(DatasetTestCase):
    def setUp(self):
        super().setUp()
        self.build(CAMERAS, [
            {"camera_id": 167, "start": wib("2026-10-11T08:02:00"), "end": wib("2026-10-11T08:03:00"), "vehicles": 6},
            {"camera_id": 167, "start": wib("2026-10-11T08:47:00"), "end": wib("2026-10-11T08:48:00"), "vehicles": 3}],
            analytics=[{"camera_id": 167, "end": wib("2026-10-11T08:03:00"), "seconds": 60, "congestion": 0.4}])

    def test_default_omits_buckets_nobody_observed(self):
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:00")
        self.assertEqual([r["bucket_start_local"][11:16] for r in rows], ["08:00", "08:45"])

    def test_fill_gaps_gives_a_regular_sequence_with_null_metrics(self):
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:00", fill_gaps=True)
        self.assertEqual([r["bucket_start_local"][11:16] for r in rows], ["08:00", "08:15", "08:30", "08:45"])
        diffs = {b["bucket_start_ts"] - a["bucket_start_ts"] for a, b in zip(rows, rows[1:])}
        self.assertEqual(diffs, {900})
        gap = rows[2]
        self.assertEqual((gap["observed_s"], gap["coverage_ratio"], gap["n_windows"], gap["analytics_s"]), (0.0, 0.0, 0, 0.0))
        for col in ("crossings", "vph", "vph_car", "vph_dir_pos", "congestion_index", "congestion_label",
                    "median_speed_bps", "queue_share", "occupancy_ratio", "focus_share", "counting_mode"):
            self.assertIsNone(gap[col], col)
        self.assertTrue(gap["low_sample"])
        self.assertEqual((gap["camera_name"], gap["hour"], gap["minute_of_day"], gap["volume_basis"]),
                         ("SIMPANG PAHLAWAN", 8, 510, "crossings"))
        self.assertEqual(rows[0]["congestion_index"], 0.4)

    def test_camera_metadata_is_joined(self):
        (r, _), _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:00")
        self.assertEqual((r["camera_name"], r["corridor"], r["latitude"], r["longitude"]),
                         ("SIMPANG PAHLAWAN", "BOGOR_SELATAN", -6.6079, 106.7952))

    def test_explicit_cameras_are_kept_even_without_data_and_discovery_skips_idle_ones(self):
        rows, stream = self.dataset("2026-10-11T08:00", "2026-10-11T09:00", camera_ids=[131, 167], fill_gaps=True)
        self.assertEqual(stream.camera_ids, [131, 167])
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r["observed_s"] == 0 for r in rows if r["camera_id"] == 131))
        rows, stream = self.dataset("2026-10-11T08:00", "2026-10-11T09:00", fill_gaps=True)
        self.assertEqual(stream.camera_ids, [167], "camera 131 has no window in the range")
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:00", camera_ids=[999])
        self.assertEqual(rows, [])

    def test_unknown_camera_takes_its_name_from_analytics(self):
        make_analytics_db(self.analytics, [{"camera_id": 555, "name": "ANALYTICS ONLY", "end": wib("2026-10-11T08:03:00"),
                                                 "seconds": 30, "congestion": 0.2}])
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T09:00", camera_ids=[555])
        self.assertEqual((rows[0]["camera_name"], rows[0]["corridor"], rows[0]["observed_s"], rows[0]["vph"]),
                         ("ANALYTICS ONLY", None, 0.0, None))
        self.assertEqual(rows[0]["congestion_index"], 0.2)

    def test_camera_major_vs_time_major_order(self):
        make_main_db(str(self.dir / "m2.sqlite"), CAMERAS, [
            {"camera_id": c, "start": wib("2026-10-11T08:02:00"), "end": wib("2026-10-11T08:03:00"), "vehicles": 1}
            for c in (167, 131)] + [
            {"camera_id": c, "start": wib("2026-10-11T08:32:00"), "end": wib("2026-10-11T08:33:00"), "vehicles": 1}
            for c in (167, 131)])
        kw = dict(main_db=str(self.dir / "m2.sqlite"), analytics_db=self.analytics)
        cam_major = [(r["camera_id"], r["bucket_start_local"][11:16]) for r in dataset.build_dataset(
            "2026-10-11T08:00", "2026-10-11T09:00", **kw)]
        time_major = [(r["camera_id"], r["bucket_start_local"][11:16]) for r in dataset.build_dataset(
            "2026-10-11T08:00", "2026-10-11T09:00", order="time", **kw)]
        self.assertEqual(cam_major, [(131, "08:00"), (131, "08:30"), (167, "08:00"), (167, "08:30")])
        self.assertEqual(time_major, [(131, "08:00"), (167, "08:00"), (131, "08:30"), (167, "08:30")])


class TestAnalyticsMetrics(DatasetTestCase):
    def test_metrics_are_window_length_weighted_means(self):
        self.build(CAMERAS, [], analytics=[
            {"camera_id": 167, "end": wib("2026-10-11T08:01:20"), "seconds": 20, "congestion": 0.2, "occupancy": 0.1,
             "median_bps": 1.0, "queue": 0.0, "traffic": 8.0, "peak": 7},
            {"camera_id": 167, "end": wib("2026-10-11T08:02:30"), "seconds": 40, "congestion": 0.5, "occupancy": 0.4,
             "median_bps": 0.4, "queue": 0.3, "traffic": 4.0, "peak": 12}])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15")
        self.assertAlmostEqual(r["congestion_index"], (0.2 * 20 + 0.5 * 40) / 60, places=3)
        self.assertEqual(r["congestion_label"], "Moderate")
        self.assertAlmostEqual(r["occupancy_ratio"], (0.1 * 20 + 0.4 * 40) / 60, places=4)
        self.assertAlmostEqual(r["median_speed_bps"], (1.0 * 20 + 0.4 * 40) / 60, places=3)
        self.assertAlmostEqual(r["queue_share"], 0.2, places=4)
        self.assertAlmostEqual(r["avg_traffic_vehicles"], (8 * 20 + 4 * 40) / 60, places=2)
        self.assertEqual(r["peak_active_vehicles"], 12)
        self.assertEqual(r["analytics_s"], 60.0)
        self.assertEqual((r["observed_s"], r["vph"]), (0.0, None), "no observation window: no flow, only congestion")

    def test_analytics_windows_are_prorated_across_bucket_edges(self):
        # a 60 s window from 08:14:30 to 08:15:30 is half in each bucket
        self.build(CAMERAS, [], analytics=[
            {"camera_id": 167, "end": wib("2026-10-11T08:15:30"), "seconds": 60, "congestion": 0.9},
            {"camera_id": 167, "end": wib("2026-10-11T08:14:30"), "seconds": 30, "congestion": 0.1}])
        a, b = self.dataset("2026-10-11T08:00", "2026-10-11T08:30")[0]
        self.assertAlmostEqual(a["analytics_s"], 60.0, places=3)             # 30 s of the first + all 30 s of the second
        self.assertAlmostEqual(a["congestion_index"], (0.9 * 30 + 0.1 * 30) / 60, places=3)
        self.assertAlmostEqual(b["analytics_s"], 30.0, places=3)
        self.assertAlmostEqual(b["congestion_index"], 0.9, places=3)

    def test_a_metric_is_averaged_only_over_the_windows_that_report_it(self):
        self.build(CAMERAS, [], analytics=[
            {"camera_id": 167, "end": wib("2026-10-11T08:01:00"), "seconds": 30, "congestion": 0.1},                    # old row: no bps
            {"camera_id": 167, "end": wib("2026-10-11T08:02:00"), "seconds": 30, "congestion": 0.3, "median_bps": 0.8}])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15")
        self.assertAlmostEqual(r["median_speed_bps"], 0.8, places=3, msg="the NULL window must not drag the mean to 0.4")
        self.assertAlmostEqual(r["congestion_index"], 0.2, places=3)

    def test_speed_is_undefined_not_zero_when_nothing_moved(self):
        self.build(CAMERAS, [], analytics=[
            {"camera_id": 167, "end": wib("2026-10-11T08:01:00"), "seconds": 30, "speed_px": 0.0, "occupancy": 0.0},
            {"camera_id": 167, "end": wib("2026-10-11T08:02:00"), "seconds": 30, "speed_px": 60.0, "occupancy": 0.1}])
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15")
        self.assertAlmostEqual(r["avg_speed_px_s"], 60.0, places=2)

    def test_labels_follow_the_single_set_of_thresholds(self):
        expected = {0.0: "Smooth", 0.249: "Smooth", 0.25: "Moderate", 0.549: "Moderate", 0.55: "Heavy",
                    0.799: "Heavy", 0.8: "Congested", 1.0: "Congested"}
        for v, label in expected.items():
            self.assertEqual(congestion_label(v), label, v)
        self.assertIsNone(congestion_label(None))

    def test_works_without_an_analytics_database(self):
        make_main_db(self.main, CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:01:00"),
                                                "end": wib("2026-10-11T08:02:00"), "vehicles": 3}])
        stream = dataset.build_dataset("2026-10-11T08:00", "2026-10-11T08:15", main_db=self.main,
                                       analytics_db=str(self.dir / "missing.db"))
        (r,) = list(stream)
        self.assertEqual((r["vph"], r["congestion_index"], r["analytics_s"]), (180.0, None, 0.0))
        self.assertIsNone(stream.source_paths["analytics_db"])

    def test_works_with_an_older_analytics_schema_lacking_the_newer_columns(self):
        make_main_db(self.main, CAMERAS, [])
        conn = sqlite3.connect(self.analytics)
        conn.execute("""CREATE TABLE camera_traffic_timeseries (id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_id INTEGER NOT NULL, camera_name TEXT NOT NULL, timestamp REAL NOT NULL, timestamp_iso TEXT NOT NULL,
            window_seconds INTEGER NOT NULL, total_volume INTEGER NOT NULL, occupancy_ratio REAL NOT NULL,
            avg_speed_px_s REAL NOT NULL, congestion_index REAL NOT NULL, avg_active_vehicles REAL NOT NULL,
            peak_active_vehicles INTEGER NOT NULL)""")
        conn.execute("INSERT INTO camera_traffic_timeseries VALUES (NULL,167,'OLD',?,'x',30,5,0.2,50.0,0.3,4.0,6)",
                     (wib("2026-10-11T08:01:00"),))
        conn.commit()
        conn.close()
        (r,), _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:15")
        self.assertEqual((r["congestion_index"], r["avg_speed_px_s"], r["median_speed_bps"], r["queue_share"]),
                         (0.3, 50.0, None, None))

    def test_missing_main_database_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            dataset.build_dataset("2026-10-11", "2026-10-12", main_db=str(self.dir / "nope.sqlite"))


class TestReadOnly(DatasetTestCase):
    def sha(self, path):
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()

    def test_sources_are_never_modified(self):
        self.build(CAMERAS, [{"camera_id": 167, "start": wib("2026-10-11T08:01:00"), "end": wib("2026-10-11T08:02:00"),
                              "vehicles": 3}], analytics=[{"camera_id": 167, "end": wib("2026-10-11T08:02:00"), "seconds": 60}],
                   sightings=[{"camera_id": 167, "iso": "2026-10-11T01:01:30Z"}])
        before = (self.sha(self.main), self.sha(self.analytics))
        self.dataset("2026-10-11T08:00", "2026-10-11T09:00", fill_gaps=True)
        for table in dataset.RAW_TABLES:
            list(dataset.iter_raw(table, "2026-10-11", "2026-10-12", main_db=self.main, analytics_db=self.analytics))
        dataset.build_provenance(main_db=self.main, analytics_db=self.analytics)
        self.assertEqual(before, (self.sha(self.main), self.sha(self.analytics)))

    def test_connections_are_read_only(self):
        self.build(CAMERAS)
        conn = dataset._open_ro(self.main)
        with self.assertRaises(sqlite3.OperationalError):
            conn.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (1,'x','y',0,0)")
        conn.close()


if __name__ == "__main__":
    unittest.main()
