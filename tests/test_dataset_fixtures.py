"""
Synthetic databases for the test_dataset_* tests (built in temp dirs; the live databases are never
touched). Also guards the schema contract: the columns backend/dataset.py reads must exist in the
schema that backend/database.py and car_tracker/analytics.py actually create.
"""

import os
import sqlite3
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

WIB = timezone(timedelta(hours=7))

# Same shape as backend/database.py (kept literal so these tests describe the layout they rely on;
# test_schema_contract below fails if database.py drifts from it).
MAIN_DDL = """
CREATE TABLE cameras (id INTEGER PRIMARY KEY, name TEXT NOT NULL, corridor TEXT NOT NULL,
    latitude REAL NOT NULL, longitude REAL NOT NULL, camera_type TEXT DEFAULT 'FIXED', stream_url TEXT,
    status TEXT DEFAULT 'online');
CREATE TABLE vehicle_sightings (sighting_id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id INTEGER NOT NULL,
    local_track_id INTEGER NOT NULL, vehicle_class TEXT NOT NULL,
    confidence REAL NOT NULL, bbox_x1 INTEGER NOT NULL, bbox_y1 INTEGER NOT NULL, bbox_x2 INTEGER NOT NULL,
    bbox_y2 INTEGER NOT NULL, heading_deg REAL NOT NULL, speed_px_s REAL NOT NULL, timestamp_iso TEXT NOT NULL);
CREATE TABLE observation_windows (window_id INTEGER PRIMARY KEY AUTOINCREMENT, camera_id INTEGER NOT NULL,
    start_ts REAL NOT NULL, end_ts REAL NOT NULL, start_iso TEXT NOT NULL, observed_s REAL NOT NULL,
    vehicles INTEGER NOT NULL DEFAULT 0, dir_pos INTEGER NOT NULL DEFAULT 0, dir_neg INTEGER NOT NULL DEFAULT 0,
    cars INTEGER NOT NULL DEFAULT 0, motorcycles INTEGER NOT NULL DEFAULT 0, trucks INTEGER NOT NULL DEFAULT 0,
    buses INTEGER NOT NULL DEFAULT 0, line_counts INTEGER NOT NULL DEFAULT 0,
    passage_counts INTEGER NOT NULL DEFAULT 0, kind TEXT NOT NULL DEFAULT 'visit');
CREATE INDEX idx_obs_cam_time ON observation_windows(camera_id, end_ts);
"""

BLOB_MARKER = b"SECRET-BLOB"


def wib(text: str) -> float:
    """Epoch seconds of a WIB wall-clock time, computed without the code under test."""
    return datetime.fromisoformat(text).replace(tzinfo=WIB).timestamp()


def iso_utc(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_main_db(path: str, cameras: Iterable[tuple] = (), windows: Iterable[Dict[str, Any]] = (),
                 sightings: Iterable[Dict[str, Any]] = ()) -> str:
    """cameras: (id, name, corridor, lat, lon); windows: dicts with camera_id, start, end (epoch) and counts."""
    conn = sqlite3.connect(path)
    conn.executescript(MAIN_DDL)
    for cid, name, corridor, lat, lon in cameras:
        conn.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude, stream_url) VALUES (?,?,?,?,?,?)",
                     (cid, name, corridor, lat, lon, f"https://example.invalid/{cid}.m3u8"))
    for w in windows:
        vehicles = w.get("vehicles", 0)
        conn.execute(
            "INSERT INTO observation_windows (camera_id, start_ts, end_ts, start_iso, observed_s, vehicles, dir_pos, "
            "dir_neg, cars, motorcycles, trucks, buses, line_counts, passage_counts, kind) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (w["camera_id"], w["start"], w["end"], iso_utc(w["start"]), w.get("observed", w["end"] - w["start"]),
             vehicles, w.get("pos", vehicles), w.get("neg", 0), w.get("cars", vehicles), w.get("motos", 0),
             w.get("trucks", 0), w.get("buses", 0), w.get("line", vehicles), w.get("passage", 0),
             w.get("kind", "visit")))
    for s in sightings:
        conn.execute(
            "INSERT INTO vehicle_sightings (camera_id, local_track_id, vehicle_class, confidence, "
            "bbox_x1, bbox_y1, bbox_x2, bbox_y2, heading_deg, speed_px_s, timestamp_iso) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (s["camera_id"], s.get("track", 1), s.get("cls", "car"), 0.9, 1, 2, 30, 40, 90.0, 12.5, s["iso"]))
    conn.commit()
    conn.close()
    return path


def make_analytics_db(path: str, windows: Iterable[Dict[str, Any]] = ()) -> str:
    """windows: dicts with camera_id, end (epoch), seconds, congestion and optional metrics. Written through
    the real TrafficAnalyticsDatabase so the schema is the engine's own."""
    from car_tracker.analytics import TimeWindowMetrics, TrafficAnalyticsDatabase

    db = TrafficAnalyticsDatabase(path)
    for w in windows:
        end = w["end"]
        db.insert_metrics(TimeWindowMetrics(
            camera_id=w["camera_id"], camera_name=w.get("name", f"CAM {w['camera_id']}"), timestamp=end,
            timestamp_iso=iso_utc(end), window_seconds=w["seconds"], total_volume=w.get("volume", 0),
            car_volume=0, motorcycle_volume=0, truck_volume=0, bus_volume=0,
            avg_active_vehicles=w.get("active", 3.0), peak_active_vehicles=w.get("peak", 5),
            occupancy_ratio=w.get("occupancy", 0.1), avg_speed_px_s=w.get("speed_px", 40.0), stopped_vehicles=0,
            congestion_index=w.get("congestion", 0.1), avg_flow_dx=0.0, avg_flow_dy=0.0,
            flow_rate_vpm=0.0, flow_rate_vph=0.0, hour=0, minute=0, day_of_week=0, is_weekend=0,
            sin_hour=0.0, cos_hour=1.0, sin_dow=0.0, cos_dow=1.0, volume_basis=w.get("basis", "crossings"),
            median_speed_bps=w.get("median_bps"), mean_speed_bps=w.get("mean_bps"),
            queue_share=w.get("queue"), avg_traffic_vehicles=w.get("traffic")))
    db.close()
    return path


class DatasetTestCase(unittest.TestCase):
    """Creates a temp dir per test with ``self.main`` / ``self.analytics`` paths."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self.main = str(self.dir / "traffic_db.sqlite")
        self.analytics = str(self.dir / "traffic_timeseries.db")
        self.addCleanup(self._tmp.cleanup)

    def build(self, cameras=(), windows=(), analytics=(), sightings=()):
        make_main_db(self.main, cameras, windows, sightings)
        make_analytics_db(self.analytics, analytics)

    def dataset(self, start, end, **kw):
        from backend import dataset
        stream = dataset.build_dataset(start, end, main_db=self.main, analytics_db=self.analytics, **kw)
        return list(stream), stream


CAMERAS = [(167, "SIMPANG PAHLAWAN", "BOGOR_SELATAN", -6.6079, 106.7952),
           (131, "TAMAN EKSPRESI", "SSA_RING", -6.5910, 106.7981)]


class TestSchemaContract(unittest.TestCase):
    def test_fixture_ddl_matches_the_backends_schema(self):
        """The columns dataset.py reads must exist in what database.py really creates."""
        from backend import database

        conn = sqlite3.connect(":memory:")
        conn.executescript(database.SCHEMA_DDL)
        real = {t: {r[1] for r in conn.execute(f"PRAGMA table_info({t})")} for t in
                ("cameras", "observation_windows", "vehicle_sightings")}
        fixture = sqlite3.connect(":memory:")
        fixture.executescript(MAIN_DDL)
        for table, cols in real.items():
            mine = {r[1] for r in fixture.execute(f"PRAGMA table_info({table})")}
            self.assertLessEqual(mine, cols, f"{table}: fixture has columns the backend does not")
        needed = {"camera_id", "start_ts", "end_ts", "observed_s", "vehicles", "dir_pos", "dir_neg", "cars",
                  "motorcycles", "trucks", "buses", "line_counts", "passage_counts", "kind"}
        self.assertLessEqual(needed, real["observation_windows"])
        self.assertLessEqual({"id", "name", "corridor", "latitude", "longitude"}, real["cameras"])

    def test_analytics_table_has_the_columns_the_dataset_reads(self):
        from backend import dataset
        from car_tracker.analytics import TrafficAnalyticsDatabase

        with tempfile.TemporaryDirectory() as tmp:
            db = TrafficAnalyticsDatabase(os.path.join(tmp, "a.db"))
            cols = {r[1] for r in db._read("PRAGMA table_info(camera_traffic_timeseries)")}
            db.close()
        wanted = {c for c, _ in dataset._AN_WANTED}
        self.assertLessEqual(wanted, cols)


if __name__ == "__main__":
    unittest.main()
