"""
backend/dataset.py: writers (CSV / JSONL / Parquet), wide pivot, raw exports, data dictionary,
provenance, the export bundle, the CLI and LSTM sequences. Synthetic databases in temp dirs.
"""

import csv
import io
import json
import unittest

from backend import dataset
from tests.test_dataset_fixtures import BLOB_MARKER, CAMERAS, DatasetTestCase, wib

START, END = "2026-10-11T08:00", "2026-10-11T09:00"


class ExportCase(DatasetTestCase):
    def setUp(self):
        super().setUp()
        t = wib("2026-10-11T08:00:00")
        self.build(
            CAMERAS,
            [{"camera_id": 167, "start": t + 120, "end": t + 180, "vehicles": 6, "cars": 4, "motos": 2},
             {"camera_id": 167, "start": t + 2820, "end": t + 2880, "vehicles": 3},
             {"camera_id": 131, "start": t + 200, "end": t + 260, "vehicles": 9, "kind": "focus"},
             # outside the 08:00-09:00 range
             {"camera_id": 131, "start": t + 7200, "end": t + 7260, "vehicles": 5}],
            analytics=[{"camera_id": 167, "end": t + 180, "seconds": 60, "congestion": 0.3, "median_bps": 0.7, "queue": 0.1},
                       {"camera_id": 131, "end": t + 260, "seconds": 60, "congestion": 0.6}],
            sightings=[{"camera_id": 167, "iso": "2026-10-11T01:02:30Z", "gid": 7},
                       {"camera_id": 131, "iso": "2026-10-11T01:30:00.250000+00:00", "gid": 8},
                       {"camera_id": 167, "iso": "2026-10-11T05:00:00Z", "gid": 9}],        # 12:00 WIB: out of range
            handoffs=[{"gid": 7, "from": 131, "to": 167, "iso": "2026-10-11T01:05:00Z"},
                      {"gid": 8, "from": 500, "to": 501, "iso": "2026-10-11T01:06:00Z"}])

    def rows(self, **kw):
        return self.dataset(START, END, **kw)


def read_csv(text):
    return list(csv.DictReader(io.StringIO(text)))


class TestWriters(ExportCase):
    def test_csv_round_trip(self):
        rows, stream = self.rows(fill_gaps=True)
        out = self.dir / "d.csv"
        info = dataset.write_dataset(iter(rows), "csv", out, columns=stream.columns)
        self.assertEqual(info["rows"], len(rows))
        parsed = read_csv(out.read_text())
        self.assertEqual(len(parsed), len(rows))
        self.assertEqual(list(parsed[0]), dataset.LONG_COLUMNS)
        for original, back in zip(rows, parsed):
            for col, value in original.items():
                text = back[col]
                if value is None:
                    self.assertEqual(text, "", col)
                elif isinstance(value, bool):
                    self.assertEqual(text, "true" if value else "false", col)
                elif isinstance(value, (int, float)):
                    self.assertEqual(float(text), float(value), col)
                else:
                    self.assertEqual(text, value, col)
        self.assertFalse(list(self.dir.glob("*.tmp")), "the temp file must be renamed away")

    def test_jsonl_round_trip(self):
        rows, _ = self.rows(fill_gaps=True)
        out = self.dir / "d.jsonl"
        dataset.write_dataset(iter(rows), "jsonl", out)
        back = [json.loads(line) for line in out.read_text().splitlines()]
        self.assertEqual(back, rows)

    def test_stream_columns_are_used_for_an_empty_dataset(self):
        stream = dataset.build_dataset("2026-11-01", "2026-11-02", main_db=self.main, analytics_db=self.analytics)
        buf = io.StringIO()
        info = dataset.write_dataset(stream, "csv", buf)
        self.assertEqual(info["rows"], 0)
        self.assertEqual(buf.getvalue().strip().split(","), dataset.LONG_COLUMNS)

    def test_writes_to_an_open_file_and_iter_text_chunks(self):
        rows, _ = self.rows(fill_gaps=True)
        buf = io.StringIO()
        dataset.write_dataset(iter(rows), "csv", buf)
        chunks = list(dataset.iter_text(iter(rows), "csv", dataset.LONG_COLUMNS, chunk_rows=3))
        self.assertGreater(len(chunks), 2)
        self.assertEqual("".join(chunks), buf.getvalue())
        lines = "".join(dataset.iter_text(iter(rows), "jsonl", chunk_rows=2)).splitlines()
        self.assertEqual(len(lines), len(rows))

    def test_unknown_format_and_missing_out(self):
        with self.assertRaises(ValueError):
            dataset.write_dataset(iter([]), "xlsx", self.dir / "x")
        with self.assertRaises(ValueError):
            dataset.write_dataset(iter([]), "csv", None)

    def test_parquet_needs_pyarrow_or_round_trips(self):
        rows, stream = self.rows(fill_gaps=True)
        out = self.dir / "d.parquet"
        if not dataset.parquet_available():
            with self.assertRaises(RuntimeError):
                dataset.write_dataset(iter(rows), "parquet", out, columns=stream.columns)
            self.assertFalse(out.exists())
            return
        import pyarrow.parquet as pq
        dataset.write_dataset(iter(rows), "parquet", out, columns=stream.columns)
        table = pq.read_table(out)
        self.assertEqual(table.num_rows, len(rows))
        self.assertEqual(table.column_names, dataset.LONG_COLUMNS)
        self.assertEqual(table.column("camera_id").to_pylist(), [r["camera_id"] for r in rows])
        self.assertEqual(table.column("vph").to_pylist(), [r["vph"] for r in rows])


class TestWideLayout(ExportCase):
    def test_wide_pivot_matches_the_long_values(self):
        long_rows, _ = self.rows(fill_gaps=True)
        wide, stream = self.rows(fill_gaps=True, layout="wide", value="vph")
        self.assertEqual(stream.columns, dataset.wide_columns([131, 167]))
        self.assertEqual(len(wide), 4)                                       # 4 buckets x 1 row each
        self.assertEqual([w["bucket_start_local"][11:16] for w in wide], ["08:00", "08:15", "08:30", "08:45"])
        lookup = {(r["bucket_start_ts"], r["camera_id"]): r["vph"] for r in long_rows}
        for w in wide:
            for cam in (131, 167):
                self.assertEqual(w[f"cam_{cam}"], lookup[(w["bucket_start_ts"], cam)])
        self.assertEqual(wide[0]["cam_167"], 360.0)                          # 6 crossings in 60 s
        self.assertEqual(wide[0]["cam_131"], 540.0)
        self.assertIsNone(wide[1]["cam_167"])                                # nobody watched it then
        for w in wide:
            self.assertEqual(list(w), stream.columns)

    def test_wide_carries_local_time_features(self):
        wide, _ = self.rows(layout="wide")
        self.assertEqual((wide[0]["hour"], wide[0]["minute_of_day"], wide[0]["day_of_week"]), (8, 480, 6))

    def test_wide_value_can_be_another_metric(self):
        wide, _ = self.rows(layout="wide", value="congestion_index")
        self.assertEqual((wide[0]["cam_167"], wide[0]["cam_131"]), (0.3, 0.6))
        with self.assertRaises(ValueError):
            self.rows(layout="wide", value="camera_name")
        with self.assertRaises(ValueError):
            self.rows(layout="sideways")

    def test_wide_without_fill_gaps_skips_buckets_nobody_observed(self):
        wide, _ = self.rows(layout="wide")
        self.assertEqual([w["bucket_start_local"][11:16] for w in wide], ["08:00", "08:45"])

    def test_wide_csv_round_trip(self):
        wide, stream = self.rows(fill_gaps=True, layout="wide")
        buf = io.StringIO()
        dataset.write_dataset(iter(wide), "csv", buf, columns=stream.columns)
        parsed = read_csv(buf.getvalue())
        self.assertEqual(list(parsed[0]), stream.columns)
        self.assertEqual(float(parsed[0]["cam_167"]), 360.0)
        self.assertEqual(parsed[1]["cam_167"], "")


class TestRawExports(ExportCase):
    def raw(self, table, **kw):
        stream = dataset.iter_raw(table, START, END, main_db=self.main, analytics_db=self.analytics, **kw)
        return list(stream), stream

    def test_observation_windows_overlapping_the_range(self):
        rows, stream = self.raw("observation_windows")
        self.assertEqual(sorted((r["camera_id"], r["vehicles"]) for r in rows), [(131, 9), (167, 3), (167, 6)])
        self.assertIn("start_local", stream.columns)
        self.assertEqual(rows[0]["start_local"][-6:], "+07:00")
        self.assertEqual([r["camera_id"] for r in self.raw("windows", camera_ids=[131])[0]], [131])

    def test_sightings_never_contain_blob_columns(self):
        rows, stream = self.raw("sightings")
        self.assertEqual(len(rows), 2, "the 12:00 WIB sighting is outside the range")
        for blob in ("embedding", "color_hist", "crop_thumbnail"):
            self.assertNotIn(blob, stream.columns)
            self.assertTrue(all(blob not in r for r in rows))
        out = self.dir / "s.csv"
        dataset.export_raw("vehicle_sightings", START, END, None, "csv", out, self.main, self.analytics)
        self.assertNotIn(BLOB_MARKER, out.read_bytes())
        self.assertEqual(rows[1]["timestamp_local"], "2026-10-11T08:30:00.250000+07:00")
        self.assertEqual(self.raw("sightings", camera_ids=[167])[0][0]["global_vehicle_id"], 7)

    def test_sighting_time_range_uses_utc_strings_of_either_format(self):
        rows = list(dataset.iter_raw("sightings", "2026-10-11T08:30:00", "2026-10-11T08:31:00", main_db=self.main))
        self.assertEqual([r["global_vehicle_id"] for r in rows], [8], "fractional '+00:00' timestamps are matched too")
        self.assertEqual(list(dataset.iter_raw("sightings", "2026-10-11T08:30:01", "2026-10-11T08:31:00", main_db=self.main)), [])

    def test_handoffs_match_either_camera(self):
        rows, _ = self.raw("handoffs", camera_ids=[167])
        self.assertEqual([r["global_vehicle_id"] for r in rows], [7])
        self.assertEqual(len(self.raw("handoff_events")[0]), 2)

    def test_cameras_metadata_hides_the_stream_url(self):
        rows, stream = self.raw("cameras")
        self.assertEqual({r["id"] for r in rows}, {131, 167})
        self.assertNotIn("stream_url", stream.columns)
        self.assertEqual([r["id"] for r in self.raw("cameras", camera_ids=[167])[0]], [167])

    def test_analytics_windows_are_exported_with_local_time(self):
        rows, stream = self.raw("analytics")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(r["timestamp_local_iso"].endswith("+07:00") for r in rows))
        self.assertEqual(stream.table, "analytics_windows")

    def test_unknown_table_and_open_ended_ranges(self):
        with self.assertRaises(ValueError):
            dataset.iter_raw("passwords", main_db=self.main)
        self.assertEqual(len(list(dataset.iter_raw("observation_windows", main_db=self.main))), 4)

    def test_raw_csv_and_jsonl_export(self):
        for fmt in ("csv", "jsonl"):
            info = dataset.export_raw("observation_windows", START, END, [167], fmt, self.dir / f"w.{fmt}",
                                      self.main, self.analytics)
            self.assertEqual((info["rows"], info["table"]), (2, "observation_windows"))
        self.assertEqual(len(read_csv((self.dir / "w.csv").read_text())), 2)
        self.assertEqual(len((self.dir / "w.jsonl").read_text().splitlines()), 2)


class TestDictionary(ExportCase):
    def check_entry(self, name, meta):
        self.assertEqual(set(meta), {"description", "unit", "dtype", "nullable", "group"}, name)
        self.assertTrue(meta["description"].strip(), name)
        self.assertIn(meta["dtype"], ("int", "float", "string", "bool"), name)

    def test_dictionary_covers_every_emitted_long_column(self):
        rows, stream = self.rows(fill_gaps=True)
        doc = dataset.data_dictionary("dataset")
        self.assertEqual(set(doc), set(stream.columns))
        for r in rows:
            self.assertEqual(set(r), set(doc))
        for name, meta in doc.items():
            self.check_entry(name, meta)

    def test_dictionary_dtypes_match_the_emitted_values(self):
        rows, _ = self.rows(fill_gaps=True)
        kinds = {"int": (int,), "float": (int, float), "string": (str,), "bool": (bool,)}
        for col, meta in dataset.data_dictionary("dataset").items():
            for r in rows:
                v = r[col]
                if v is None:
                    self.assertTrue(meta["nullable"], f"{col} is documented as never null")
                else:
                    self.assertIsInstance(v, kinds[meta["dtype"]], col)
        self.assertFalse(isinstance(rows[0]["n_windows"], bool))

    def test_dictionary_covers_the_wide_columns(self):
        _, stream = self.rows(layout="wide", value="vph")
        doc = dataset.data_dictionary("dataset_wide", stream.camera_ids, "vph")
        self.assertEqual(list(doc), stream.columns)
        self.assertIn("camera 167", doc["cam_167"]["description"])

    def test_dictionary_covers_every_raw_column(self):
        for table in dataset.RAW_TABLES:
            stream = dataset.iter_raw(table, START, END, main_db=self.main, analytics_db=self.analytics)
            doc = stream.dictionary()
            self.assertEqual(list(doc), stream.columns, table)
            for name, meta in doc.items():
                self.check_entry(f"{table}.{name}", meta)
            for r in stream:
                self.assertEqual(set(r), set(doc), table)

    def test_unknown_raw_columns_are_described_from_the_schema(self):
        import sqlite3
        conn = sqlite3.connect(self.main)
        conn.execute("ALTER TABLE observation_windows ADD COLUMN fps_hint REAL")
        conn.commit()
        conn.close()
        stream = dataset.iter_raw("observation_windows", main_db=self.main)
        doc = stream.dictionary()
        self.assertIn("fps_hint", doc)
        self.assertEqual(doc["fps_hint"]["dtype"], "float")
        self.assertIn("undocumented", doc["fps_hint"]["description"])

    def test_wide_value_must_be_numeric(self):
        for bad in ("camera_name", "low_sample", "bucket_start_utc", "nope"):
            with self.assertRaises(ValueError):
                dataset.data_dictionary("dataset_wide", [167], bad)
        self.assertIn("vph", dataset.NUMERIC_COLUMNS)
        self.assertNotIn("hour", dataset.NUMERIC_COLUMNS)


class TestProvenanceAndBundle(ExportCase):
    def test_provenance_documents_the_export(self):
        rows, stream = self.rows(fill_gaps=True, bucket_minutes=15)
        prov = dataset.build_provenance(stream, files=[{"name": "x.csv", "rows": len(rows)}])
        self.assertEqual(prov["rows"], len(rows))
        self.assertEqual((prov["bucket_minutes"], prov["timezone"], prov["utc_offset"], prov["volume_basis"]),
                         (15.0, "Asia/Jakarta", "+07:00", "crossings"))
        self.assertEqual(prov["cameras"], [131, 167])
        self.assertTrue(prov["fill_gaps"])
        self.assertEqual(prov["range_covered"]["start_local"], "2026-10-11T08:00:00+07:00")
        self.assertRegex(prov["generated_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ$")
        main = prov["sources"]["main_db"]
        self.assertTrue(main["path"].endswith("traffic_db.sqlite"))
        self.assertEqual(main["observation_windows"]["max_ts"], wib("2026-10-11T08:00:00") + 7260)
        self.assertEqual(main["vehicle_sightings"]["max_iso"], "2026-10-11T05:00:00Z")
        self.assertEqual(main["handoff_events"]["max_iso"], "2026-10-11T01:06:00Z")
        self.assertEqual(prov["sources"]["analytics_db"]["camera_traffic_timeseries"]["max_ts"], wib("2026-10-11T08:04:20"))
        joined = " ".join(prov["caveats"]).lower()
        self.assertIn("crossings", joined)
        self.assertIn("never be mixed", joined)
        self.assertIn("observed", joined)
        self.assertEqual(prov["stats"]["cameras_with_data"], 2)
        json.dumps(prov)

    def test_export_bundle_writes_data_dictionary_and_provenance(self):
        out = self.dir / "exports" / "nested"          # created on use
        prov = dataset.export_bundle(START, END, out, bucket_minutes=15, camera_ids=[167, 131], fill_gaps=True,
                                     fmt="csv", raw=["observation_windows", "sightings", "handoffs", "cameras"],
                                     main_db=self.main, analytics_db=self.analytics)
        names = sorted(p.name for p in out.iterdir())
        self.assertIn("dictionary.json", names)
        self.assertIn("provenance.json", names)
        self.assertIn("traffic_long_15min_20261011T0800_20261011T0900.csv", names)
        self.assertIn("raw_cameras.csv", names)
        self.assertIn("raw_vehicle_sightings_20261011T0800_20261011T0900.csv", names)
        provenance = json.loads((out / "provenance.json").read_text())
        dictionary = json.loads((out / "dictionary.json").read_text())
        self.assertEqual(provenance["rows"], 8)
        files = {f["name"]: f for f in provenance["files"]}
        main = files["traffic_long_15min_20261011T0800_20261011T0900.csv"]
        self.assertEqual(main["rows"], len(read_csv((out / main["name"]).read_text())))
        self.assertEqual(set(dictionary["tables"]),
                         {"traffic_long", "raw_observation_windows", "raw_vehicle_sightings", "raw_handoff_events", "raw_cameras"})
        header = (out / main["name"]).read_text().splitlines()[0].split(",")
        self.assertEqual(header, list(dictionary["tables"]["traffic_long"]))
        for name, f in files.items():
            if name.startswith("raw_"):
                table = dictionary["tables"][f["kind"]]
                self.assertEqual((out / name).read_text().splitlines()[0].split(","), list(table), name)
        self.assertEqual(prov["rows"], provenance["rows"])

    def test_export_bundle_wide_jsonl_and_dictionary(self):
        out = self.dir / "w"
        dataset.export_bundle(START, END, out, layout="wide", value="congestion_index", fmt="jsonl",
                              main_db=self.main, analytics_db=self.analytics)
        data = [json.loads(line) for line in (out / "traffic_wide_15min_20261011T0800_20261011T0900.jsonl").read_text().splitlines()]
        doc = json.loads((out / "dictionary.json").read_text())["tables"]["traffic_wide"]
        self.assertEqual(set(data[0]), set(doc))
        self.assertEqual((data[0]["cam_167"], data[0]["cam_131"]), (0.3, 0.6))

    def test_default_export_dir_is_under_data_exports(self):
        self.assertTrue(dataset.default_export_dir().replace("\\", "/").endswith("data/exports"))


class TestSequences(DatasetTestCase):
    def setUp(self):
        super().setUp()
        t = wib("2026-10-11T08:00:00")
        # camera 167 observed in every 5-minute bucket for 3 hours except one gap
        windows = [{"camera_id": 167, "start": t + i * 300 + 10, "end": t + i * 300 + 70, "vehicles": i}
                   for i in range(36) if i != 20]
        self.build(CAMERAS, windows, analytics=[
            {"camera_id": 167, "end": t + i * 300 + 70, "seconds": 60, "congestion": 0.01 * i} for i in range(36)])

    def test_windows_are_consecutive_and_never_span_a_gap(self):
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T11:00", bucket_minutes=5, fill_gaps=True)
        seq = dataset.build_sequences(rows, seq_len=4, horizon=1, features=["vph", "congestion_index", "sin_hour"], target="vph")
        self.assertEqual(seq["X"].shape[1:], (4, 3))
        self.assertEqual(seq["y"].shape[1:], (1,))
        # 36 buckets, bucket 20 is empty (null vph): windows touching it are dropped
        ts = list(seq["target_ts"])
        gap_ts = rows[20]["bucket_start_ts"]
        self.assertNotIn(gap_ts, ts)
        for target_ts in ts:
            self.assertFalse(gap_ts <= target_ts <= gap_ts + 4 * 300)       # the target or any of its 4 inputs is the gap
        self.assertEqual(seq["X"].shape[0], len(ts))
        i = ts.index(rows[10]["bucket_start_ts"])
        self.assertAlmostEqual(float(seq["y"][i][0]), rows[10]["vph"], places=2)
        self.assertAlmostEqual(float(seq["X"][i][-1][0]), rows[9]["vph"], places=2)

    def test_nan_policy_zero_keeps_windows_with_null_inputs(self):
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T11:00", bucket_minutes=5, fill_gaps=True)
        drop = dataset.build_sequences(rows, seq_len=4, features=["vph"], target="congestion_index")
        zero = dataset.build_sequences(rows, seq_len=4, features=["vph"], target="congestion_index", nan_policy="zero")
        self.assertGreater(zero["X"].shape[0], drop["X"].shape[0])
        self.assertFalse(float("nan") in zero["X"])

    def test_too_short_series_gives_empty_arrays(self):
        rows, _ = self.dataset("2026-10-11T08:00", "2026-10-11T08:20", bucket_minutes=5)
        seq = dataset.build_sequences(rows, seq_len=12)
        self.assertEqual(seq["X"].shape[0], 0)

    def test_npz_bundle(self):
        import numpy as np
        out = self.dir / "npz"
        dataset.export_bundle("2026-10-11T08:00", "2026-10-11T11:00", out, bucket_minutes=5, camera_ids=[167],
                              fill_gaps=True, fmt="npz", sequences={"seq_len": 6, "horizon": 2, "nan_policy": "zero",
                                                                   "features": ["vph", "sin_hour"], "target": "vph"},
                              main_db=self.main, analytics_db=self.analytics)
        data = np.load(next(out.glob("*.npz")))
        self.assertEqual(data["X"].shape[1:], (6, 2))
        self.assertEqual(data["y"].shape[1:], (2,))
        self.assertEqual(list(data["feature_names"]), ["vph", "sin_hour"])


class TestCommandLine(ExportCase):
    def run_cli(self, *args):
        import contextlib
        from scripts import export_dataset
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = export_dataset.main(["--main-db", self.main, "--analytics-db", self.analytics, *args])
        return code, out.getvalue(), err.getvalue()

    def test_cli_writes_the_bundle(self):
        out = self.dir / "cli"
        code, stdout, stderr = self.run_cli("--start", START, "--end", END, "--bucket", "15", "--cameras", "167,131",
                                            "--format", "jsonl", "--fill-gaps", "--raw", "observation_windows,sightings",
                                            "--out", str(out))
        self.assertEqual(code, 0, stderr)
        self.assertIn("dictionary.json", stdout)
        files = sorted(p.name for p in out.iterdir())
        self.assertIn("provenance.json", files)
        self.assertTrue(any(f.startswith("traffic_long_15min_") and f.endswith(".jsonl") for f in files), files)
        self.assertTrue(any(f.startswith("raw_vehicle_sightings_") and f.endswith(".jsonl") for f in files), files)
        self.assertEqual(json.loads((out / "provenance.json").read_text())["rows"], 8)

    def test_cli_wide_csv(self):
        out = self.dir / "cliw"
        code, _, stderr = self.run_cli("--start", START, "--end", END, "--layout", "wide", "--value", "vph", "--out", str(out))
        self.assertEqual(code, 0, stderr)
        csv_file = next(out.glob("traffic_wide_*.csv"))
        self.assertIn("cam_167", csv_file.read_text().splitlines()[0])

    def test_cli_reports_bad_input_with_exit_code_2(self):
        for args in (["--bucket", "7"], ["--start", "garbage"], ["--format", "npz", "--layout", "wide"],
                     ["--raw", "passwords"], ["--value", "camera_name", "--layout", "wide"]):
            code, _, stderr = self.run_cli("--start", START, "--end", END, "--out", str(self.dir / "bad"), *args)
            self.assertEqual(code, 2, args)
            self.assertIn("error:", stderr)

    def test_cli_default_output_is_data_exports_and_nothing_is_written_here(self):
        from scripts import export_dataset
        action = next(a for a in export_dataset.build_parser()._actions if a.dest == "out")
        self.assertTrue(action.default.replace("\\", "/").endswith("data/exports"))

    def test_cli_empty_range_warns(self):
        code, _, stderr = self.run_cli("--start", "2026-01-01", "--end", "2026-01-02", "--out", str(self.dir / "empty"))
        self.assertEqual(code, 0)
        self.assertIn("warning: no rows", stderr)


if __name__ == "__main__":
    unittest.main()
