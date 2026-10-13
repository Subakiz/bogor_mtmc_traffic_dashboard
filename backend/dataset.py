"""
backend/dataset.py
Analysis-ready data extraction for the BSW Kota Bogor traffic dashboard.

Turns the two live SQLite databases into a tidy, regular per-camera time series (pandas / ML
friendly) and into raw table exports, always together with a data dictionary and provenance.

Sources (opened strictly read-only, ``mode=ro``; nothing is ever written to them)
    * main DB (``data/traffic_db.sqlite``): ``observation_windows`` = counting-line crossings per
      observation window with the OBSERVED seconds, classes, directions, kind (focus/visit) and
      counting mode; ``cameras`` (name, corridor, position); ``vehicle_sightings`` and
      ``handoff_events`` for raw exports.
    * analytics DB (``data/traffic_timeseries.db``): ``camera_traffic_timeseries`` = per-window
      congestion index, speeds, occupancy, queue share.

How a row is built (long layout: one row per camera x bucket)
    * Buckets are ``bucket_minutes`` long and aligned to LOCAL wall-clock boundaries (Asia/Jakarta,
      WIB, UTC+7, no DST). Only buckets lying completely inside ``[start, end)`` are produced.
    * Every observation window is spread over the buckets it overlaps, crossings and observed
      seconds in proportion to the overlap (crossings are assumed uniform in time inside a window).
    * Analytics windows are weighted by their overlap in seconds; every metric is the
      window-length-weighted mean over the windows that report it.
    * ``vph = crossings * 3600 / observed_s``: flow while the camera was WATCHED. Cameras are only
      sampled intermittently (rolling visits of a few seconds), so ``observed_s`` /
      ``coverage_ratio`` / ``low_sample`` say how much the number rests on. ``crossings`` is never
      the count for the whole bucket.
    * Volume is counting-line crossings only; track-based volume is never mixed in.

Public API
    build_dataset(...)  -> DatasetStream (iterator of row dicts, ``.columns``, ``.stats``)
    write_dataset(rows, fmt, out)    iter_text(rows, fmt)    build_sequences(rows, ...)
    export_raw(table, start, end, camera_ids, fmt, out)      iter_raw(...)
    data_dictionary(name)            build_provenance(...)   export_bundle(...)
Needs only the standard library (+ numpy for ``build_sequences``); Parquet needs ``pyarrow``.
"""

from __future__ import annotations

import csv
import io
import json
import math
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, TextIO, Tuple, Union

try:  # IANA zone when the system has tzdata, otherwise the same fixed offset (WIB has no DST)
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Jakarta")
except Exception:  # noqa: BLE001  (ZoneInfoNotFoundError, ImportError)
    LOCAL_TZ = timezone(timedelta(hours=7), "WIB")
LOCAL_TZ_NAME = "Asia/Jakarta"
LOCAL_UTC_OFFSET_S = int(datetime(2026, 1, 1, tzinfo=LOCAL_TZ).utcoffset().total_seconds())   # 25200

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
LOW_SAMPLE_S = 60.0              # a bucket with less observed time than this is flagged ``low_sample``
FORMATS = ("csv", "jsonl", "parquet")

TimeLike = Union[str, int, float, datetime]


def default_main_db() -> str:
    return os.environ.get("TRAFFIC_DB_PATH") or str(PROJECT_ROOT / "data" / "traffic_db.sqlite")


def default_analytics_db() -> str:
    return os.environ.get("ANALYTICS_DB_PATH") or str(PROJECT_ROOT / "data" / "traffic_timeseries.db")


def default_export_dir() -> str:
    return str(PROJECT_ROOT / "data" / "exports")


# ── Time helpers ────────────────────────────────────────────────────────────────

def parse_time(value: TimeLike) -> float:
    """
    Epoch seconds from epoch seconds (int/float, or a string of 9+ digits), a ``datetime`` or an ISO
    string (``2026-10-11``, ``2026-10-11T08:30``, ``...Z``, ``...+07:00``). A value with no offset
    is local (WIB) time.
    """
    if isinstance(value, bool):
        raise ValueError(f"not a time: {value!r}")
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1e11:
            raise ValueError(f"{value!r} looks like epoch milliseconds; pass epoch seconds")
        return ts
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        text = value.strip()
        head = text.split(".", 1)[0]
        if head.isdigit() and len(head) >= 9 and text.replace(".", "", 1).isdigit():
            return parse_time(float(text))
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            raise ValueError(f"cannot parse time {value!r}: use ISO-8601 (2026-10-11T08:00) or epoch seconds") from exc
    else:
        raise ValueError(f"not a time: {value!r}")
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=LOCAL_TZ)
    return dt.timestamp()


def to_utc_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def to_local_iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, LOCAL_TZ).isoformat()


def _iso_to_local(text: Any) -> Optional[str]:
    """Local ISO of a stored UTC ISO string (``...Z`` or ``...+00:00``, optional fractions)."""
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(str(text))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ).isoformat()


def bucket_seconds(bucket_minutes: Union[int, float]) -> int:
    """Bucket length in seconds. Must be whole minutes dividing a day (1, 5, 10, 15, 30, 60, 120, ... 1440)."""
    try:
        bs = int(round(float(bucket_minutes) * 60))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bucket_minutes must be a number, got {bucket_minutes!r}") from exc
    if bs < 60 or 86400 % bs != 0:
        raise ValueError("bucket_minutes must be a whole number of minutes that divides a day "
                         f"(1, 5, 15, 30, 60, 120, 1440 ...), got {bucket_minutes!r}")
    return bs


def _bucket_range(start_ts: float, end_ts: float, bs: int) -> Tuple[int, int]:
    """Indices ``[k0, k1)`` of the buckets lying completely inside ``[start_ts, end_ts)``; bucket
    ``k`` covers UTC ``[k*bs - off, (k+1)*bs - off)``, i.e. it is aligned to local wall-clock time."""
    k0 = math.ceil((start_ts + LOCAL_UTC_OFFSET_S) / bs)
    k1 = math.floor((end_ts + LOCAL_UTC_OFFSET_S) / bs)
    return k0, max(k0, k1)


def congestion_label(index: Optional[float]) -> Optional[str]:
    """Same thresholds as ``TrafficAnalyticsEngine.get_congestion_label`` (pinned by a test)."""
    if index is None:
        return None
    if index < 0.25:
        return "Smooth"
    if index < 0.55:
        return "Moderate"
    if index < 0.80:
        return "Heavy"
    return "Congested"


# ── Data dictionary ─────────────────────────────────────────────────────────────

def _col(description: str, dtype: str, unit: Optional[str] = None, nullable: bool = True,
         group: str = "") -> Dict[str, Any]:
    return {"description": description, "unit": unit, "dtype": dtype, "nullable": nullable, "group": group}


_DIRECTION_NOTE = ("Direction is relative to each camera's learned counting line (+ = along the line's "
                   "unit normal, the dominant direction of travel in image terms), so it is comparable "
                   "within one camera, not across cameras.")

# Column order of the long layout (one row per camera x bucket).
DATASET_COLUMNS: Dict[str, Dict[str, Any]] = {
    "bucket_start_ts": _col("Start of the bucket as Unix epoch seconds (UTC). Buckets are [start, start + bucket length).",
                            "int", "s", False, "time"),
    "bucket_start_utc": _col("Start of the bucket, ISO-8601 in UTC (suffix Z).", "string", None, False, "time"),
    "bucket_start_local": _col("Start of the bucket, ISO-8601 in local time (Asia/Jakarta, WIB) with its +07:00 offset. "
                               "Bucket edges are aligned to this clock.", "string", None, False, "time"),
    "camera_id": _col("Camera id (BSW CCTV id; joins to cameras.id).", "int", None, False, "camera"),
    "camera_name": _col("Camera name from the cameras table (analytics name if the camera is unknown there).",
                        "string", None, True, "camera"),
    "corridor": _col("Corridor the camera belongs to (topology).", "string", None, True, "camera"),
    "latitude": _col("Camera latitude (WGS84).", "float", "deg", True, "camera"),
    "longitude": _col("Camera longitude (WGS84).", "float", "deg", True, "camera"),
    "observed_s": _col("Seconds of this bucket in which the camera was actually watched: observation-window "
                       "observed seconds, prorated over the buckets each window overlaps. 0 for gap rows.",
                       "float", "s", False, "coverage"),
    "coverage_ratio": _col("observed_s divided by the bucket length, capped at 1. Cameras are sampled "
                           "intermittently, so this is usually well below 1.", "float", "ratio", False, "coverage"),
    "n_windows": _col("Number of observation windows overlapping the bucket (a window spanning two buckets "
                      "counts in both).", "int", "windows", False, "coverage"),
    "focus_share": _col("Share of observed_s that came from focus windows (a camera being viewed on the "
                        "dashboard) rather than rolling visits. Null when observed_s is 0.", "float", "ratio", True, "coverage"),
    "low_sample": _col("True when observed_s is below the low-sample threshold (default 60 s): vph rests on "
                       "only a few vehicles and is a rough estimate. Always true for gap rows.", "bool", None, False, "coverage"),
    "crossings": _col("Vehicles that crossed the camera's counting line during the OBSERVED part of the bucket "
                      "(prorated, hence fractional). Null when the camera was not observed. Not the bucket's "
                      "total traffic: divide by observed_s (vph) to get a rate.", "float", "vehicles", True, "volume"),
    "vph": _col("Flow rate while watched = crossings * 3600 / observed_s. Null when observed_s is 0.",
                "float", "vehicles/h", True, "volume"),
    "vph_car": _col("Cars: rate while watched (cars crossed * 3600 / observed_s).", "float", "vehicles/h", True, "volume"),
    "vph_motorcycle": _col("Motorcycles: rate while watched.", "float", "vehicles/h", True, "volume"),
    "vph_truck": _col("Trucks: rate while watched.", "float", "vehicles/h", True, "volume"),
    "vph_bus": _col("Buses: rate while watched.", "float", "vehicles/h", True, "volume"),
    "vph_dir_pos": _col("Rate of crossings in the + direction of the camera's counting line. " + _DIRECTION_NOTE,
                        "float", "vehicles/h", True, "volume"),
    "vph_dir_neg": _col("Rate of crossings in the - direction of the camera's counting line. " + _DIRECTION_NOTE,
                        "float", "vehicles/h", True, "volume"),
    "line_share": _col("Share of crossings counted against the learned counting line (rest were counted in "
                       "'passage' mode, used until a camera has enough trajectories). Null without crossings.",
                       "float", "ratio", True, "volume"),
    "counting_mode": _col("'line' when at least half of the crossings used the learned line, else 'passage'. "
                          "Null without crossings.", "string", None, True, "volume"),
    "analytics_s": _col("Seconds of analytics windows overlapping the bucket (the weight behind the congestion "
                        "and speed columns). Independent of observed_s: the two sources close windows separately.",
                        "float", "s", False, "congestion"),
    "congestion_index": _col("Congestion index in [0, 1] judged on the moving stream only (queue share, slowness "
                             "against the camera's own free-flow speed, density when slow). Window-length-weighted "
                             "mean. Null when no analytics window overlaps the bucket.", "float", "index", True, "congestion"),
    "congestion_label": _col("Smooth (<0.25), Moderate (<0.55), Heavy (<0.8) or Congested, from congestion_index.",
                             "string", None, True, "congestion"),
    "median_speed_bps": _col("Median speed of the moving-stream vehicles in body sizes per second (comparable across "
                             "cameras). Null when no vehicle was measurable, and for windows recorded before this "
                             "column existed.", "float", "bodies/s", True, "congestion"),
    "mean_speed_bps": _col("Mean of the same speed samples (body sizes per second). Null as median_speed_bps.",
                           "float", "bodies/s", True, "congestion"),
    "avg_speed_px_s": _col("Mean smoothed pixel speed of moving-stream vehicles (pixels/s; depends on camera "
                           "resolution and distance, use only within one camera). Windows without moving traffic "
                           "are left out, not counted as 0.", "float", "px/s", True, "congestion"),
    "occupancy_ratio": _col("Share of the frame covered by moving-stream vehicles (union of boxes), 0-1.",
                            "float", "ratio", True, "congestion"),
    "queue_share": _col("Share of moving-stream vehicle samples standing still (below the stopped speed), 0-1. "
                        "Null when not measured (older windows).", "float", "ratio", True, "congestion"),
    "lane_occupancy": _col("Share of the camera's learned lane (road) area covered by traffic-stream vehicles "
                           "(moving or queued), 0-1. Null until the camera's lane map is learned (rows before "
                           "2026-10-12 included).", "float", "ratio", True, "congestion"),
    "avg_active_vehicles": _col("Mean number of detected vehicles per frame (parked ones included).",
                                "float", "vehicles", True, "congestion"),
    "peak_active_vehicles": _col("Largest number of vehicles detected in one frame.", "int", "vehicles", True, "congestion"),
    "avg_traffic_vehicles": _col("Mean moving-stream vehicles measurable per frame. Null for older windows.",
                                 "float", "vehicles", True, "congestion"),
    "hour": _col("Local (WIB) hour of day of the bucket start, 0-23.", "int", "h", False, "time_features"),
    "minute_of_day": _col("Local minutes since midnight of the bucket start, 0-1439.", "int", "min", False, "time_features"),
    "day_of_week": _col("Local day of week of the bucket start, 0 = Monday ... 6 = Sunday.", "int", None, False, "time_features"),
    "is_weekend": _col("1 on local Saturday or Sunday, else 0.", "int", None, False, "time_features"),
    "sin_hour": _col("sin(2*pi*minute_of_day/1440): cyclic encoding of the time of day.", "float", None, False, "time_features"),
    "cos_hour": _col("cos(2*pi*minute_of_day/1440).", "float", None, False, "time_features"),
    "sin_dow": _col("sin(2*pi*day_of_week/7): cyclic encoding of the day of week.", "float", None, False, "time_features"),
    "cos_dow": _col("cos(2*pi*day_of_week/7).", "float", None, False, "time_features"),
    "volume_basis": _col("What the volume columns count. Always 'crossings' (counting-line crossings); "
                         "track-based volume is never part of this dataset.", "string", None, False, "volume"),
}
LONG_COLUMNS: List[str] = list(DATASET_COLUMNS)

# Long-layout columns that carry a number (usable as the value of a wide pivot).
NUMERIC_COLUMNS: List[str] = [c for c, m in DATASET_COLUMNS.items()
                              if m["dtype"] in ("int", "float") and m["group"] not in ("time", "camera", "time_features")]

_TIME_FEATURE_COLUMNS = ["hour", "minute_of_day", "day_of_week", "is_weekend", "sin_hour", "cos_hour", "sin_dow", "cos_dow"]
_WIDE_BASE_COLUMNS = ["bucket_start_ts", "bucket_start_utc", "bucket_start_local"] + _TIME_FEATURE_COLUMNS


def wide_column_name(camera_id: int) -> str:
    return f"cam_{camera_id}"


def wide_columns(camera_ids: Sequence[int]) -> List[str]:
    return _WIDE_BASE_COLUMNS + [wide_column_name(c) for c in camera_ids]


# Documentation of the raw tables. Columns not listed here (schema changes) are documented from the
# database's own declaration so that the dictionary always covers every exported column.
_RAW_DOCS: Dict[str, Dict[str, Dict[str, Any]]] = {
    "observation_windows": {
        "window_id": _col("Row id.", "int", None, False),
        "camera_id": _col("Camera id.", "int", None, False),
        "start_ts": _col("Window start, Unix epoch seconds (UTC).", "float", "s", False),
        "end_ts": _col("Window end, Unix epoch seconds (UTC).", "float", "s", False),
        "start_iso": _col("Window start, ISO-8601 UTC.", "string", None, False),
        "observed_s": _col("Seconds the camera was actually observed in this window.", "float", "s", False),
        "vehicles": _col("Counting-line crossings in the window.", "int", "vehicles", False),
        "dir_pos": _col("Crossings in the + direction of the counting line.", "int", "vehicles", False),
        "dir_neg": _col("Crossings in the - direction.", "int", "vehicles", False),
        "cars": _col("Cars crossed.", "int", "vehicles", False),
        "motorcycles": _col("Motorcycles crossed.", "int", "vehicles", False),
        "trucks": _col("Trucks crossed.", "int", "vehicles", False),
        "buses": _col("Buses crossed.", "int", "vehicles", False),
        "line_counts": _col("Crossings counted against the learned counting line.", "int", "vehicles", False),
        "passage_counts": _col("Crossings counted in 'passage' mode (no learned line yet).", "int", "vehicles", False),
        "kind": _col("'focus' (camera being viewed, ~1 min windows) or 'visit' (rolling coverage, 2-20 s).", "string", None, False),
        "start_local": _col("start_ts as local (WIB) ISO-8601 with offset.", "string", None, False),
        "end_local": _col("end_ts as local (WIB) ISO-8601 with offset.", "string", None, False),
    },
    "vehicle_sightings": {
        "sighting_id": _col("Row id.", "int", None, False),
        "camera_id": _col("Camera id.", "int", None, False),
        "local_track_id": _col("Tracker id, unique only within a camera visit.", "int", None, False),
        "global_vehicle_id": _col("Cross-camera identity assigned by re-identification (null if none).", "int", None, True),
        "vehicle_class": _col("car, motorcycle, bus or truck.", "string", None, False),
        "confidence": _col("Detector confidence 0-1.", "float", None, False),
        "bbox_x1": _col("Bounding box left (px).", "int", "px", False),
        "bbox_y1": _col("Bounding box top (px).", "int", "px", False),
        "bbox_x2": _col("Bounding box right (px).", "int", "px", False),
        "bbox_y2": _col("Bounding box bottom (px).", "int", "px", False),
        "heading_deg": _col("Heading of motion in image terms (degrees).", "float", "deg", False),
        "speed_px_s": _col("Pixel speed.", "float", "px/s", False),
        "timestamp_iso": _col("Sighting time, ISO-8601 UTC.", "string", None, False),
        "timestamp_local": _col("Sighting time in local (WIB) ISO-8601 with offset.", "string", None, True),
    },
    "handoff_events": {
        "event_id": _col("Row id.", "int", None, False),
        "global_vehicle_id": _col("Cross-camera identity that was handed off.", "int", None, False),
        "from_camera_id": _col("Camera the vehicle left.", "int", None, False),
        "to_camera_id": _col("Camera it was matched at.", "int", None, False),
        "confidence": _col("Match confidence 0-1.", "float", None, False),
        "timestamp_iso": _col("Handoff time, ISO-8601 UTC.", "string", None, False),
        "timestamp_local": _col("Handoff time in local (WIB) ISO-8601 with offset.", "string", None, True),
    },
    "cameras": {
        "id": _col("Camera id.", "int", None, False),
        "name": _col("Camera name.", "string", None, False),
        "corridor": _col("Corridor.", "string", None, False),
        "latitude": _col("Latitude (WGS84).", "float", "deg", False),
        "longitude": _col("Longitude (WGS84).", "float", "deg", False),
        "camera_type": _col("FIXED or PTZ.", "string", None, True),
        "status": _col("online, or removed when the camera vanished from the municipal site.", "string", None, True),
    },
    "analytics_windows": {
        "id": _col("Row id.", "int", None, False),
        "camera_id": _col("Camera id.", "int", None, False),
        "camera_name": _col("Camera name at the time.", "string", None, False),
        "timestamp": _col("Window END, Unix epoch seconds (UTC). The window is [timestamp - window_seconds, timestamp].", "float", "s", False),
        "timestamp_iso": _col("Window end, ISO-8601 UTC.", "string", None, False),
        "timestamp_local_iso": _col("Window end, ISO-8601 local (WIB).", "string", None, True),
        "window_seconds": _col("Observed length of the window (2-62 s for rolling visits).", "int", "s", False),
        "total_volume": _col("Volume in the window; what it counts is volume_basis.", "int", "vehicles", False),
        "volume_basis": _col("'crossings' (counting-line) or 'tracks' (distinct track ids, inflated): do not mix.", "string", None, False),
        "congestion_index": _col("Window congestion index 0-1.", "float", "index", False),
        "hour": _col("Local (WIB) hour of the window end.", "int", "h", False),
        "minute": _col("Local minute of the window end.", "int", "min", False),
        "day_of_week": _col("Local day of week of the window end, 0 = Monday.", "int", None, False),
        "is_weekend": _col("1 on local Saturday/Sunday.", "int", None, False),
        "median_speed_bps": _col("Median moving-stream speed, body sizes/s (null for older rows).", "float", "bodies/s", True),
        "mean_speed_bps": _col("Mean moving-stream speed, body sizes/s (null for older rows).", "float", "bodies/s", True),
        "queue_share": _col("Share of moving-stream samples standing still (null for older rows).", "float", "ratio", True),
        "avg_traffic_vehicles": _col("Mean measurable moving-stream vehicles per frame (null for older rows).", "float", "vehicles", True),
        "lane_occupancy": _col("Share of the learned lane area covered by traffic (null until lanes are learned).", "float", "ratio", True),
    },
}
_SQLITE_DTYPE = {"INTEGER": "int", "INT": "int", "REAL": "float", "FLOAT": "float", "DOUBLE": "float",
                 "NUMERIC": "float", "TEXT": "string", "VARCHAR": "string", "CHAR": "string"}


def _sqlite_dtype(decl: str) -> str:
    d = (decl or "").upper()
    for key, dtype in _SQLITE_DTYPE.items():
        if key in d:
            return dtype
    return "string"


def data_dictionary(name: str = "dataset", camera_ids: Optional[Sequence[int]] = None,
                    value: str = "vph") -> Dict[str, Dict[str, Any]]:
    """
    column -> ``{description, unit, dtype, nullable, group}``.

    ``name``: ``"dataset"`` (long layout, the default), ``"dataset_wide"`` (describes the ``cam_<id>``
    columns for ``camera_ids`` holding ``value``) or a raw table (``observation_windows``,
    ``vehicle_sightings``, ``handoff_events``, ``cameras``, ``analytics_windows``; the dictionary of an
    actual export, which also covers columns this version does not know, is ``RawStream.dictionary()``).
    """
    if name == "dataset":
        return {c: dict(m) for c, m in DATASET_COLUMNS.items()}
    if name == "dataset_wide":
        if value not in NUMERIC_COLUMNS:
            raise ValueError(f"value must be one of {NUMERIC_COLUMNS}, got {value!r}")
        out = {c: dict(DATASET_COLUMNS[c]) for c in _WIDE_BASE_COLUMNS}
        meta = DATASET_COLUMNS[value]
        for cid in camera_ids or []:
            out[wide_column_name(cid)] = _col(f"{value} of camera {cid}: {meta['description']}",
                                              meta["dtype"], meta["unit"], True, "cameras")
        return out
    table = _RAW_ALIASES.get(name, name)
    if table not in _RAW_DOCS:
        raise ValueError(f"unknown table {name!r}; choose from {sorted(_RAW_DOCS)}")
    return {c: dict(m) for c, m in _RAW_DOCS[table].items()}


# ── Read-only access ────────────────────────────────────────────────────────────

def _open_ro(path: Union[str, os.PathLike]) -> sqlite3.Connection:
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"database not found: {p}")
    conn = sqlite3.connect(f"{p.as_uri()}?mode=ro", uri=True, timeout=10.0, check_same_thread=False)
    conn.execute("PRAGMA query_only = ON;")
    conn.execute("PRAGMA busy_timeout = 10000;")
    return conn


def _table_columns(conn: Optional[sqlite3.Connection], table: str) -> Dict[str, str]:
    """column -> declared type; empty when the table does not exist."""
    if conn is None:
        return {}
    return {r[1]: (r[2] or "") for r in conn.execute(f"PRAGMA table_info({table})")}


def _select_list(have: Dict[str, str], wanted: Sequence[Tuple[str, str]]) -> str:
    """SELECT list that substitutes a literal for a column an older database does not have."""
    return ", ".join(c if c in have else f"{default} AS {c}" for c, default in wanted)


_OBS_WANTED = (("camera_id", "NULL"), ("start_ts", "NULL"), ("end_ts", "NULL"), ("observed_s", "NULL"),
               ("vehicles", "0"), ("dir_pos", "0"), ("dir_neg", "0"), ("cars", "0"), ("motorcycles", "0"),
               ("trucks", "0"), ("buses", "0"), ("line_counts", "0"), ("passage_counts", "0"),
               ("kind", "'visit'"))
_OBS_REQUIRED = ("camera_id", "start_ts", "end_ts", "observed_s")
_AN_WANTED = (("camera_id", "NULL"), ("camera_name", "NULL"), ("timestamp", "NULL"), ("window_seconds", "NULL"),
              ("congestion_index", "NULL"), ("occupancy_ratio", "NULL"), ("avg_speed_px_s", "NULL"),
              ("avg_active_vehicles", "NULL"), ("peak_active_vehicles", "NULL"), ("median_speed_bps", "NULL"),
              ("mean_speed_bps", "NULL"), ("queue_share", "NULL"), ("avg_traffic_vehicles", "NULL"),
              ("lane_occupancy", "NULL"))
_AN_REQUIRED = ("camera_id", "timestamp", "window_seconds")
# analytics metrics that are window-length-weighted means (the order of ``_An.sums``)
_AN_MEANS = ("congestion_index", "occupancy_ratio", "avg_speed_px_s", "avg_active_vehicles",
             "median_speed_bps", "mean_speed_bps", "queue_share", "avg_traffic_vehicles", "lane_occupancy")
_MAX_IN = 200


class _Sources:
    """Read-only connections to both databases plus what they contain."""

    def __init__(self, main_db: Optional[str], analytics_db: Optional[str]):
        self.main_path = str(Path(main_db or default_main_db()).expanduser())
        self.analytics_path = str(Path(analytics_db or default_analytics_db()).expanduser())
        self.main = _open_ro(self.main_path)
        self.analytics: Optional[sqlite3.Connection] = None
        try:
            obs = _table_columns(self.main, "observation_windows")
            missing = [c for c in _OBS_REQUIRED if c not in obs]
            if missing:
                raise ValueError(f"{self.main_path} has no usable observation_windows table (missing {missing or 'table'})")
            self.obs_select = _select_list(obs, _OBS_WANTED)
            if os.path.exists(self.analytics_path):
                self.analytics = _open_ro(self.analytics_path)
                an = _table_columns(self.analytics, "camera_traffic_timeseries")
                if all(c in an for c in _AN_REQUIRED):
                    self.an_select = _select_list(an, _AN_WANTED)
                else:
                    self.analytics.close()
                    self.analytics = None
        except BaseException:
            self.close()
            raise
        self.an_max_window = 0.0
        if self.analytics is not None:
            row = self.analytics.execute("SELECT MAX(window_seconds) FROM camera_traffic_timeseries").fetchone()
            self.an_max_window = float(row[0] or 0.0)

    def close(self) -> None:
        for conn in (getattr(self, "main", None), getattr(self, "analytics", None)):
            if conn is not None:
                try:
                    conn.close()
                except sqlite3.Error:
                    pass
        self.main = None            # type: ignore[assignment]
        self.analytics = None

    # -- metadata --
    def cameras(self) -> Dict[int, Dict[str, Any]]:
        cols = _table_columns(self.main, "cameras")
        if not {"id", "name"} <= set(cols):
            return {}
        wanted = [c for c in ("id", "name", "corridor", "latitude", "longitude") if c in cols]
        return {r[0]: dict(zip(wanted, r)) for r in self.main.execute(f"SELECT {', '.join(wanted)} FROM cameras")}

    def analytics_names(self, camera_ids: Sequence[int]) -> Dict[int, str]:
        if self.analytics is None or not camera_ids:
            return {}
        out = {}
        for r in self.analytics.execute("SELECT camera_id, MAX(camera_name) FROM camera_traffic_timeseries GROUP BY camera_id"):
            if r[0] in camera_ids and r[1]:
                out[r[0]] = r[1]
        return out

    def discover_cameras(self, start_ts: float, end_ts: float) -> List[int]:
        ids: Set[int] = {r[0] for r in self.main.execute(
            "SELECT DISTINCT camera_id FROM observation_windows WHERE end_ts > ? AND start_ts < ?", (start_ts, end_ts))}
        if self.analytics is not None:
            ids.update(r[0] for r in self.analytics.execute(
                "SELECT DISTINCT camera_id FROM camera_traffic_timeseries WHERE timestamp > ? AND timestamp < ? "
                "AND timestamp - window_seconds < ?", (start_ts, end_ts + self.an_max_window, end_ts)))
        return sorted(ids)


def _camera_clause(column: str, camera_ids: Optional[Sequence[int]]) -> Tuple[str, List[Any]]:
    if camera_ids is None or len(camera_ids) > _MAX_IN:
        return "", []                      # many cameras: filtered in Python
    if len(camera_ids) == 1:
        return f" AND {column} = ?", [camera_ids[0]]
    return f" AND {column} IN ({','.join('?' * len(camera_ids))})", list(camera_ids)


# ── Aggregation ─────────────────────────────────────────────────────────────────

class _Obs:
    """Observation-window totals of one (camera, bucket)."""
    __slots__ = ("observed", "n", "vehicles", "pos", "neg", "cars", "motos", "trucks", "buses", "line", "passage", "focus_s")

    def __init__(self) -> None:
        self.observed = self.vehicles = self.pos = self.neg = self.cars = 0.0
        self.motos = self.trucks = self.buses = self.line = self.passage = self.focus_s = 0.0
        self.n = 0


class _An:
    """Analytics-window totals of one (camera, bucket): per-metric weighted sums."""
    __slots__ = ("w", "sums", "wts", "peak", "name")

    def __init__(self) -> None:
        self.w = 0.0
        self.sums = [0.0] * len(_AN_MEANS)
        self.wts = [0.0] * len(_AN_MEANS)
        self.peak: Optional[float] = None
        self.name: Optional[str] = None


def _spread(s_ts: float, e_ts: float, k0: int, k1: int, bs: int) -> Iterator[Tuple[int, float, float]]:
    """(bucket, fraction of the interval, overlap seconds) for each bucket in ``[k0, k1)`` that the
    interval ``[s_ts, e_ts]`` overlaps. A zero-length interval falls entirely in its bucket."""
    off = LOCAL_UTC_OFFSET_S
    length = e_ts - s_ts
    if length <= 0:
        k = math.floor((e_ts + off) / bs)
        if k0 <= k < k1:
            yield k, 1.0, 0.0
        return
    ka = max(k0, math.floor((s_ts + off) / bs))
    kb = min(k1 - 1, math.ceil((e_ts + off) / bs) - 1)
    for k in range(ka, kb + 1):
        lo = max(s_ts, k * bs - off)
        hi = min(e_ts, (k + 1) * bs - off)
        if hi > lo:
            yield k, (hi - lo) / length, hi - lo


def _aggregate(src: _Sources, camera_ids: Optional[Sequence[int]], k0: int, k1: int, bs: int
               ) -> Tuple[Dict[int, Dict[int, _Obs]], Dict[int, Dict[int, _An]]]:
    """Prorates the observation and analytics windows of ``camera_ids`` (None: all) into the buckets
    ``[k0, k1)``. Returns ``{camera: {bucket: accumulator}}`` for both sources."""
    off = LOCAL_UTC_OFFSET_S
    s_ts, e_ts = k0 * bs - off, k1 * bs - off
    wanted: Optional[Set[int]] = set(camera_ids) if camera_ids is not None else None

    obs: Dict[int, Dict[int, _Obs]] = {}
    clause, params = _camera_clause("camera_id", camera_ids)
    sql = (f"SELECT {src.obs_select} FROM observation_windows WHERE end_ts > ? AND start_ts < ?{clause}")
    for cam, w_s, w_e, observed, vehicles, pos, neg, cars, motos, trucks, buses, line, passage, kind in \
            src.main.execute(sql, [s_ts, e_ts, *params]):
        if wanted is not None and cam not in wanted:
            continue
        by_k = obs.setdefault(cam, {})
        observed = float(observed or 0.0)
        for k, frac, _ in _spread(w_s, w_e, k0, k1, bs):
            a = by_k.get(k)
            if a is None:
                a = by_k[k] = _Obs()
            a.n += 1
            a.observed += observed * frac
            a.vehicles += (vehicles or 0) * frac
            a.pos += (pos or 0) * frac
            a.neg += (neg or 0) * frac
            a.cars += (cars or 0) * frac
            a.motos += (motos or 0) * frac
            a.trucks += (trucks or 0) * frac
            a.buses += (buses or 0) * frac
            a.line += (line or 0) * frac
            a.passage += (passage or 0) * frac
            if kind == "focus":
                a.focus_s += observed * frac

    an: Dict[int, Dict[int, _An]] = {}
    if src.analytics is not None:
        clause, params = _camera_clause("camera_id", camera_ids)
        sql = (f"SELECT {src.an_select} FROM camera_traffic_timeseries "
               f"WHERE timestamp > ? AND timestamp < ?{clause}")
        for row in src.analytics.execute(sql, [s_ts, e_ts + src.an_max_window, *params]):
            cam, name, ts, ws = row[0], row[1], row[2], row[3]
            if wanted is not None and cam not in wanted:
                continue
            congestion, occupancy, speed_px, avg_active, peak_active, med_bps, mean_bps, queue, traffic, lane_occ = row[4:14]
            if speed_px is not None and speed_px == 0 and not occupancy:
                speed_px = None            # no moving-stream vehicle in the window: speed undefined, not 0
            metrics = (congestion, occupancy, speed_px, avg_active, med_bps, mean_bps, queue, traffic, lane_occ)
            by_k = an.setdefault(cam, {})
            for k, _, overlap in _spread(ts - ws, ts, k0, k1, bs):
                if overlap <= 0:
                    continue
                a = by_k.get(k)
                if a is None:
                    a = by_k[k] = _An()
                a.w += overlap
                for i, v in enumerate(metrics):
                    if v is not None:
                        a.sums[i] += v * overlap
                        a.wts[i] += overlap
                if peak_active is not None:
                    a.peak = peak_active if a.peak is None else max(a.peak, peak_active)
                if name:
                    a.name = name
    return obs, an


def _clean(value: Any, nd: int) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return round(value, nd)


class _TimeFeatures:
    """Per-bucket local-time columns, cached (the same buckets recur for every camera)."""

    def __init__(self, bs: int):
        self.bs = bs
        self._cache: Dict[int, Tuple[Any, ...]] = {}

    def get(self, k: int) -> Tuple[Any, ...]:
        hit = self._cache.get(k)
        if hit is not None:
            return hit
        if len(self._cache) > 100_000:
            self._cache.clear()
        ts = k * self.bs - LOCAL_UTC_OFFSET_S
        local = datetime.fromtimestamp(ts, LOCAL_TZ)
        mod = local.hour * 60 + local.minute
        dow = local.weekday()
        ang_h = 2.0 * math.pi * mod / 1440.0
        ang_d = 2.0 * math.pi * dow / 7.0
        feats = (ts, to_utc_iso(ts), local.isoformat(), local.hour, mod, dow, 1 if dow >= 5 else 0,
                 round(math.sin(ang_h), 5), round(math.cos(ang_h), 5),
                 round(math.sin(ang_d), 5), round(math.cos(ang_d), 5))
        self._cache[k] = feats
        return feats


def _make_row(cam: int, k: int, obs: Optional[_Obs], an: Optional[_An], meta: Dict[str, Any],
              bs: int, tf: _TimeFeatures, low_sample_s: float) -> Dict[str, Any]:
    ts, utc, local, hour, mod, dow, wknd, sin_h, cos_h, sin_d, cos_d = tf.get(k)
    observed = obs.observed if obs is not None else 0.0
    has_obs = obs is not None and obs.n > 0
    per_h = 3600.0 / observed if observed > 0 else None

    def rate(count: float) -> Optional[float]:
        return round(count * per_h, 2) if per_h is not None else None

    line_share = counting_mode = None
    if has_obs and (obs.line + obs.passage) > 0:
        line_share = round(obs.line / (obs.line + obs.passage), 4)
        counting_mode = "line" if obs.line >= obs.passage else "passage"

    means: List[Optional[float]] = [None] * len(_AN_MEANS)
    if an is not None:
        means = [an.sums[i] / an.wts[i] if an.wts[i] > 0 else None for i in range(len(_AN_MEANS))]
    congestion, occupancy, speed_px, avg_active, med_bps, mean_bps, queue, traffic, lane_occ = means
    congestion = _clean(congestion, 3)

    return {
        "bucket_start_ts": ts, "bucket_start_utc": utc, "bucket_start_local": local,
        "camera_id": cam,
        "camera_name": meta.get("name") or (an.name if an is not None else None),
        "corridor": meta.get("corridor"), "latitude": meta.get("latitude"), "longitude": meta.get("longitude"),
        "observed_s": round(observed, 3),
        "coverage_ratio": round(min(1.0, observed / bs), 4),
        "n_windows": obs.n if obs is not None else 0,
        "focus_share": round(min(1.0, obs.focus_s / observed), 4) if observed > 0 else None,
        "low_sample": observed < low_sample_s,
        "crossings": round(obs.vehicles, 3) if has_obs else None,
        "vph": rate(obs.vehicles) if has_obs else None,
        "vph_car": rate(obs.cars) if has_obs else None,
        "vph_motorcycle": rate(obs.motos) if has_obs else None,
        "vph_truck": rate(obs.trucks) if has_obs else None,
        "vph_bus": rate(obs.buses) if has_obs else None,
        "vph_dir_pos": rate(obs.pos) if has_obs else None,
        "vph_dir_neg": rate(obs.neg) if has_obs else None,
        "line_share": line_share, "counting_mode": counting_mode,
        "analytics_s": round(an.w, 3) if an is not None else 0.0,
        "congestion_index": congestion, "congestion_label": congestion_label(congestion),
        "median_speed_bps": _clean(med_bps, 3), "mean_speed_bps": _clean(mean_bps, 3),
        "avg_speed_px_s": _clean(speed_px, 2), "occupancy_ratio": _clean(occupancy, 4),
        "queue_share": _clean(queue, 4), "lane_occupancy": _clean(lane_occ, 4),
        "avg_active_vehicles": _clean(avg_active, 2),
        "peak_active_vehicles": int(an.peak) if an is not None and an.peak is not None else None,
        "avg_traffic_vehicles": _clean(traffic, 2),
        "hour": hour, "minute_of_day": mod, "day_of_week": dow, "is_weekend": wknd,
        "sin_hour": sin_h, "cos_hour": cos_h, "sin_dow": sin_d, "cos_dow": cos_d,
        "volume_basis": "crossings",
    }


class DatasetStream:
    """
    Iterator over the rows of a dataset, returned by :func:`build_dataset`.

    Attributes: ``columns`` (column order of the rows), ``camera_ids``, ``layout``, ``value``,
    ``bucket_minutes``, ``start_ts``/``end_ts`` (as requested), ``first_bucket_ts``/``end_bucket_ts``
    (the whole buckets actually covered), ``stats`` (filled while iterating) and ``source_paths``.
    Database connections are opened on construction and closed when the iterator is exhausted or
    :meth:`close` is called.
    """

    def __init__(self, start: TimeLike, end: TimeLike, bucket_minutes: Union[int, float] = 15,
                 camera_ids: Optional[Iterable[int]] = None, fill_gaps: bool = False,
                 main_db: Optional[str] = None, analytics_db: Optional[str] = None, layout: str = "long",
                 value: str = "vph", low_sample_s: float = LOW_SAMPLE_S, order: Optional[str] = None,
                 chunk_buckets: Optional[int] = None):
        if layout not in ("long", "wide"):
            raise ValueError("layout must be 'long' or 'wide'")
        if layout == "wide" and value not in NUMERIC_COLUMNS:
            raise ValueError(f"value must be one of {NUMERIC_COLUMNS}, got {value!r}")
        self.bs = bucket_seconds(bucket_minutes)
        self.bucket_minutes = self.bs / 60
        self.start_ts, self.end_ts = parse_time(start), parse_time(end)
        if self.end_ts < self.start_ts:
            raise ValueError("end is before start")
        self.layout, self.value, self.fill_gaps, self.low_sample_s = layout, value, bool(fill_gaps), float(low_sample_s)
        self.order = order or ("time" if layout == "wide" else "camera")
        if self.order not in ("camera", "time"):
            raise ValueError("order must be 'camera' or 'time'")
        self.k0, self.k1 = _bucket_range(self.start_ts, self.end_ts, self.bs)
        self.first_bucket_ts = self.k0 * self.bs - LOCAL_UTC_OFFSET_S
        self.end_bucket_ts = self.k1 * self.bs - LOCAL_UTC_OFFSET_S
        self.stats = {"rows": 0, "rows_observed": 0, "observed_s": 0.0, "crossings": 0.0, "cameras_with_data": 0}
        self._cameras_with_data: Set[int] = set()
        self._src = _Sources(main_db, analytics_db)
        self.source_paths = {"main_db": self._src.main_path,
                             "analytics_db": self._src.analytics_path if self._src.analytics is not None else None}
        try:
            if camera_ids is None:
                self.camera_ids = self._src.discover_cameras(self.first_bucket_ts, self.end_bucket_ts) if self.k1 > self.k0 else []
            else:
                self.camera_ids = sorted({int(c) for c in camera_ids})
            self.columns = LONG_COLUMNS if layout == "long" else wide_columns(self.camera_ids)
            if chunk_buckets is None:
                if self.order == "camera":
                    chunk_buckets = max(1, 31 * 86400 // self.bs)
                else:
                    chunk_buckets = max(1, min(31 * 86400 // self.bs, 60_000 // max(1, len(self.camera_ids))))
            self._chunk = max(1, int(chunk_buckets))
            self._meta = self._src.cameras()
            missing = [c for c in self.camera_ids if not (self._meta.get(c) or {}).get("name")]
            if missing:
                for cid, name in self._src.analytics_names(missing).items():
                    self._meta.setdefault(cid, {})["name"] = name
        except BaseException:
            self._src.close()
            raise
        self._it = self._generate()

    # -- iterator protocol --
    def __iter__(self) -> "DatasetStream":
        return self

    def __next__(self) -> Dict[str, Any]:
        try:
            return next(self._it)
        except StopIteration:
            self.close()
            raise

    def close(self) -> None:
        self._src.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass

    # -- generation --
    def _long_rows(self) -> Iterator[Dict[str, Any]]:
        bs, tf = self.bs, _TimeFeatures(self.bs)
        chunks = [(a, min(a + self._chunk, self.k1)) for a in range(self.k0, self.k1, self._chunk)]
        cams = self.camera_ids

        def rows_for(cam_list: Sequence[int], a: int, b: int) -> Dict[int, List[Dict[str, Any]]]:
            obs, an = _aggregate(self._src, cam_list, a, b, bs)
            out: Dict[int, List[Dict[str, Any]]] = {}
            for cam in cam_list:
                o, n = obs.get(cam, {}), an.get(cam, {})
                ks = range(a, b) if self.fill_gaps else sorted(set(o) | set(n))
                meta = self._meta.get(cam) or {}
                out[cam] = [_make_row(cam, k, o.get(k), n.get(k), meta, bs, tf, self.low_sample_s) for k in ks]
            return out

        if self.order == "camera":
            for cam in cams:
                for a, b in chunks:
                    yield from rows_for([cam], a, b)[cam]
        else:
            for a, b in chunks:
                per_cam = rows_for(cams, a, b)
                merged = [r for rows in per_cam.values() for r in rows]
                merged.sort(key=lambda r: (r["bucket_start_ts"], r["camera_id"]))
                yield from merged

    def _generate(self) -> Iterator[Dict[str, Any]]:
        try:
            if self.k1 <= self.k0 or not self.camera_ids:
                return
            rows = self._long_rows()
            if self.layout == "wide":
                rows = self._pivot(self._counted(rows))
                for row in rows:
                    self.stats["rows"] += 1
                    yield row
            else:
                for row in self._counted(rows):
                    self.stats["rows"] += 1
                    yield row
        finally:
            self.stats["cameras_with_data"] = len(self._cameras_with_data)
            self.close()

    def _counted(self, rows: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Accumulates the long-row statistics (what the provenance reports) as rows pass."""
        for r in rows:
            if r["observed_s"] > 0:
                self.stats["rows_observed"] += 1
                self.stats["observed_s"] += r["observed_s"]
                self._cameras_with_data.add(r["camera_id"])
            if r["crossings"]:
                self.stats["crossings"] += r["crossings"]
            yield r

    def _pivot(self, rows: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Time-major long rows -> one row per bucket with a column per camera."""
        cam_cols = [wide_column_name(c) for c in self.camera_ids]
        current: Optional[Dict[str, Any]] = None
        for r in rows:
            if current is None or r["bucket_start_ts"] != current["bucket_start_ts"]:
                if current is not None:
                    yield current
                current = {c: r[c] for c in _WIDE_BASE_COLUMNS}
                current.update(dict.fromkeys(cam_cols))
            current[wide_column_name(r["camera_id"])] = r[self.value]
        if current is not None:
            yield current


def build_dataset(start: TimeLike, end: TimeLike, bucket_minutes: Union[int, float] = 15,
                  camera_ids: Optional[Iterable[int]] = None, fill_gaps: bool = False,
                  main_db: Optional[str] = None, analytics_db: Optional[str] = None, *,
                  layout: str = "long", value: str = "vph", low_sample_s: float = LOW_SAMPLE_S,
                  order: Optional[str] = None, chunk_buckets: Optional[int] = None) -> DatasetStream:
    """
    Regular per-camera time series as an iterator of row dicts (memory use is bounded by one chunk,
    so a month of data can be streamed).

    ``start``/``end``: ISO string (naive = WIB), epoch seconds or ``datetime``; ``[start, end)``,
    only buckets lying completely inside it are produced. ``bucket_minutes`` must divide a day.
    ``camera_ids=None``: every camera with an observation or analytics window in the range.
    ``fill_gaps=True`` also emits buckets nobody observed (observed_s = 0, null metrics) so each
    camera has a regular sequence. ``layout="wide"``: one row per bucket with a ``cam_<id>`` column
    holding ``value`` (any numeric long column, default ``vph``). ``order``: ``"camera"`` (default
    for long: each camera's rows contiguous) or ``"time"``. ``main_db``/``analytics_db`` default to
    the live databases, opened read-only. The returned :class:`DatasetStream` knows its ``columns``.
    """
    return DatasetStream(start, end, bucket_minutes, camera_ids, fill_gaps, main_db, analytics_db, layout, value,
                         low_sample_s, order, chunk_buckets)


# ── Writers ─────────────────────────────────────────────────────────────────────

def _fmt_csv(value: Any) -> Any:
    if value is None:
        return ""
    if value is True:
        return "true"
    if value is False:
        return "false"
    return value


def _resolve_columns(rows: Any, columns: Optional[Sequence[str]]) -> Tuple[List[str], Iterator[Dict[str, Any]]]:
    """Column list for ``rows`` (explicit, the stream's own, or peeked from the first row)."""
    cols = list(columns) if columns is not None else (list(rows.columns) if hasattr(rows, "columns") else None)
    it = iter(rows)
    if cols is None:
        try:
            first = next(it)
        except StopIteration:
            return list(LONG_COLUMNS), iter(())
        cols = list(first)
        it = _chain_first(first, it)
    return cols, it


def _chain_first(first: Dict[str, Any], rest: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
    yield first
    yield from rest


def iter_text(rows: Iterable[Dict[str, Any]], fmt: str = "csv", columns: Optional[Sequence[str]] = None,
              chunk_rows: int = 500) -> Iterator[str]:
    """
    The rows as chunks of text (CSV with header, or JSON lines): feed it to a streaming HTTP
    response or write the chunks to a file. Rows are produced and encoded lazily.
    """
    if fmt not in ("csv", "jsonl"):
        raise ValueError("iter_text supports 'csv' and 'jsonl'")
    cols, it = _resolve_columns(rows, columns)
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n") if fmt == "csv" else None
    if writer is not None:
        writer.writerow(cols)
    n = 0
    for row in it:
        if writer is not None:
            writer.writerow([_fmt_csv(row.get(c)) for c in cols])
        else:
            buf.write(json.dumps({c: row.get(c) for c in cols}, separators=(",", ":"), default=str, allow_nan=False))
            buf.write("\n")
        n += 1
        if n % chunk_rows == 0:
            yield buf.getvalue()
            buf.seek(0)
            buf.truncate()
    if buf.tell():
        yield buf.getvalue()


def parquet_available() -> bool:
    try:
        import pyarrow  # noqa: F401
        import pyarrow.parquet  # noqa: F401
        return True
    except ImportError:
        return False


def _write_parquet(rows: Iterator[Dict[str, Any]], cols: Sequence[str], path: str,
                   dtypes: Dict[str, str], batch: int = 50_000) -> int:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("Parquet output needs the 'pyarrow' package (pip install pyarrow); use csv or jsonl") from exc
    kinds = {"int": pa.int64(), "float": pa.float64(), "string": pa.string(), "bool": pa.bool_()}
    schema = pa.schema([(c, kinds[dtypes.get(c, "string")]) for c in cols])
    count = 0
    with pq.ParquetWriter(path, schema) as writer:
        buffer: List[Dict[str, Any]] = []

        def flush() -> None:
            if buffer:
                writer.write_table(pa.Table.from_pydict({c: [r.get(c) for r in buffer] for c in cols}, schema=schema))
                buffer.clear()

        for row in rows:
            buffer.append(row)
            count += 1
            if len(buffer) >= batch:
                flush()
        flush()
    return count


def write_dataset(rows: Iterable[Dict[str, Any]], fmt: str = "csv", out: Union[str, os.PathLike, TextIO, None] = None,
                  columns: Optional[Sequence[str]] = None, dtypes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Streams ``rows`` to ``out`` (a path, or an open text file for csv/jsonl) as ``fmt`` =
    ``csv`` | ``jsonl`` | ``parquet`` (pyarrow). A path is written atomically (temp file + rename).
    Returns ``{"format", "path", "rows", "bytes", "columns"}``. ``dtypes`` (column -> int|float|
    string|bool) types a Parquet file; it defaults to the dataset dictionary.
    """
    fmt = fmt.lower()
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {FORMATS}")
    if out is None:
        raise ValueError("out is required")
    cols, it = _resolve_columns(rows, columns)
    counter = {"rows": 0}

    def counted(source: Iterator[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        for r in source:
            counter["rows"] += 1
            yield r

    counted_rows = counted(it)
    if hasattr(out, "write"):
        if fmt == "parquet":
            raise ValueError("parquet needs a file path, not an open file")
        for chunk in iter_text(counted_rows, fmt, cols):
            out.write(chunk)                                   # type: ignore[union-attr]
        return {"format": fmt, "path": None, "rows": counter["rows"], "bytes": None, "columns": cols}

    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        if fmt == "parquet":
            doc = {c: m["dtype"] for c, m in DATASET_COLUMNS.items()}
            doc.update({c: "float" for c in cols if c.startswith("cam_")})      # wide layout
            doc.update(dtypes or {})
            _write_parquet(counted_rows, cols, str(tmp), doc)
        else:
            with open(tmp, "w", encoding="utf-8", newline="") as f:
                for chunk in iter_text(counted_rows, fmt, cols):
                    f.write(chunk)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return {"format": fmt, "path": str(path), "rows": counter["rows"], "bytes": path.stat().st_size, "columns": cols}


DEFAULT_SEQUENCE_FEATURES = ("vph", "congestion_index", "coverage_ratio", "sin_hour", "cos_hour",
                             "sin_dow", "cos_dow", "is_weekend")


def build_sequences(rows: Iterable[Dict[str, Any]], seq_len: int = 12, horizon: int = 1,
                    features: Sequence[str] = DEFAULT_SEQUENCE_FEATURES, target: str = "vph",
                    nan_policy: str = "drop") -> Dict[str, Any]:
    """
    Sliding windows for LSTM-style models from long-layout rows (``X`` = ``seq_len`` consecutive
    buckets of ``features``, ``y`` = the next ``horizon`` values of ``target``), per camera. Windows
    never span a missing bucket (use ``fill_gaps=True`` for regular sequences). ``nan_policy="drop"``
    skips windows with a null feature/target; ``"zero"`` replaces nulls in X by 0 (nulls in y still
    drop the window). Returns numpy arrays ``X`` (N, seq_len, F), ``y`` (N, horizon) plus
    ``camera_id``/``target_ts`` per sample and ``feature_names``.
    """
    import numpy as np

    if nan_policy not in ("drop", "zero"):
        raise ValueError("nan_policy must be 'drop' or 'zero'")
    feats, need = list(features), seq_len + horizon
    by_cam: Dict[int, List[Dict[str, Any]]] = {}
    step: Optional[int] = None
    for r in rows:
        by_cam.setdefault(r["camera_id"], []).append(r)
    X: List[Any] = []
    y: List[Any] = []
    cams: List[int] = []
    tss: List[int] = []
    for cam, series in by_cam.items():
        series.sort(key=lambda r: r["bucket_start_ts"])
        if len(series) < need:
            continue
        if step is None:
            diffs = {b["bucket_start_ts"] - a["bucket_start_ts"] for a, b in zip(series, series[1:])}
            step = min(diffs) if diffs else None
        mat = np.array([[np.nan if r.get(f) is None else float(r[f]) for f in feats] for r in series], dtype=np.float32)
        tgt = np.array([np.nan if r.get(target) is None else float(r[target]) for r in series], dtype=np.float32)
        ts = [r["bucket_start_ts"] for r in series]
        for i in range(len(series) - need + 1):
            if step is not None and ts[i + need - 1] - ts[i] != step * (need - 1):
                continue                                     # a bucket is missing inside the window
            x, t = mat[i:i + seq_len], tgt[i + seq_len:i + need]
            if np.isnan(t).any():
                continue
            if np.isnan(x).any():
                if nan_policy == "drop":
                    continue
                x = np.nan_to_num(x, nan=0.0)
            X.append(x)
            y.append(t)
            cams.append(cam)
            tss.append(ts[i + need - 1])
    return {
        "X": np.stack(X) if X else np.empty((0, seq_len, len(feats)), dtype=np.float32),
        "y": np.stack(y) if y else np.empty((0, horizon), dtype=np.float32),
        "camera_id": np.array(cams, dtype=np.int64), "target_ts": np.array(tss, dtype=np.int64),
        "feature_names": np.array(feats), "target_name": np.array(target),
    }


# ── Raw exports ─────────────────────────────────────────────────────────────────

_RAW_ALIASES = {"windows": "observation_windows", "obs": "observation_windows", "sightings": "vehicle_sightings",
                "handoffs": "handoff_events", "camera": "cameras", "metadata": "cameras",
                "analytics": "analytics_windows", "camera_traffic_timeseries": "analytics_windows"}
RAW_TABLES = tuple(_RAW_DOCS)
_BLOB_NOTE = "BLOB columns (appearance embeddings, color histograms, crop thumbnails) are never exported."

_RAW_SPECS: Dict[str, Dict[str, Any]] = {
    "observation_windows": {"db": "main", "table": "observation_windows", "cam": ["camera_id"],
                            "order": "end_ts, window_id", "local_ts": [("start_ts", "start_local"), ("end_ts", "end_local")]},
    "vehicle_sightings": {"db": "main", "table": "vehicle_sightings", "cam": ["camera_id"], "iso": "timestamp_iso",
                          "order": "timestamp_iso, sighting_id", "local_iso": [("timestamp_iso", "timestamp_local")]},
    "handoff_events": {"db": "main", "table": "handoff_events", "cam": ["from_camera_id", "to_camera_id"],
                       "iso": "timestamp_iso", "order": "timestamp_iso, event_id",
                       "local_iso": [("timestamp_iso", "timestamp_local")]},
    "cameras": {"db": "main", "table": "cameras", "cam": ["id"], "order": "id", "drop": ("stream_url",)},
    "analytics_windows": {"db": "analytics", "table": "camera_traffic_timeseries", "cam": ["camera_id"],
                          "order": "timestamp, id", "local_ts": [("timestamp", "timestamp_local_iso")]},
}


def _canonical_table(table: str) -> str:
    name = _RAW_ALIASES.get(table, table)
    if name not in _RAW_SPECS:
        raise ValueError(f"unknown table {table!r}; choose from {sorted(_RAW_SPECS)}")
    return name


class RawStream:
    """Iterator over the rows of a raw export (dicts); ``columns`` lists them in order."""

    def __init__(self, table: str, start: Optional[TimeLike] = None, end: Optional[TimeLike] = None,
                 camera_ids: Optional[Iterable[int]] = None, main_db: Optional[str] = None,
                 analytics_db: Optional[str] = None):
        self.table = _canonical_table(table)
        spec = _RAW_SPECS[self.table]
        self.start_ts = parse_time(start) if start is not None else None
        self.end_ts = parse_time(end) if end is not None else None
        cams = None if camera_ids is None else sorted({int(c) for c in camera_ids})
        path = (main_db or default_main_db()) if spec["db"] == "main" else (analytics_db or default_analytics_db())
        self.source_path = str(Path(path).expanduser())
        self._conn = _open_ro(self.source_path)
        try:
            have = _table_columns(self._conn, spec["table"])
            if not have:
                raise ValueError(f"{self.source_path} has no table {spec['table']}")
            drop = set(spec.get("drop", ()))
            self._db_cols = [c for c, t in have.items() if "BLOB" not in t.upper() and c not in drop]
            self._derived: List[Tuple[str, str, str]] = []   # (source column, output column, kind)
            for src_col, dst, kind in ([(a, b, "ts") for a, b in spec.get("local_ts", [])] +
                                       [(a, b, "iso") for a, b in spec.get("local_iso", [])]):
                if src_col in have:
                    self._derived.append((src_col, dst, kind))
            self.columns = self._db_cols + [dst for _, dst, _ in self._derived if dst not in self._db_cols]
            where, params = ["1=1"], []
            self._cam_filter = None
            cam_cols = [c for c in spec["cam"] if c in have]
            if cams is not None and cam_cols:
                if len(cams) <= _MAX_IN:
                    ors = " OR ".join(f"{c} IN ({','.join('?' * len(cams))})" for c in cam_cols)
                    where.append(f"({ors})")
                    params += cams * len(cam_cols)
                else:                                   # too many for an IN list: filtered in Python
                    self._cam_filter = (set(cams), cam_cols)
            if "iso" in spec:
                if self.start_ts is not None:
                    where.append(f"{spec['iso']} >= ?")
                    params.append(datetime.fromtimestamp(math.floor(self.start_ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
                if self.end_ts is not None:
                    where.append(f"{spec['iso']} < ?")
                    params.append(datetime.fromtimestamp(math.ceil(self.end_ts), timezone.utc).strftime("%Y-%m-%dT%H:%M:%S"))
            elif self.table == "observation_windows":
                if self.start_ts is not None:
                    where.append("end_ts > ?")
                    params.append(self.start_ts)
                if self.end_ts is not None:
                    where.append("start_ts < ?")
                    params.append(self.end_ts)
            elif self.table == "analytics_windows":
                if self.start_ts is not None:
                    where.append("timestamp > ?")
                    params.append(self.start_ts)
                if self.end_ts is not None:
                    where.append("timestamp - window_seconds < ?")
                    params.append(self.end_ts)
            self._declared = {c: have[c] for c in self._db_cols}
            order = spec["order"] if all(c.strip() in have for c in spec["order"].split(",")) else None
            sql = f"SELECT {', '.join(self._db_cols)} FROM {spec['table']} WHERE {' AND '.join(where)}"
            if order:
                sql += f" ORDER BY {order}"
            self._sql, self._params = sql, params
        except BaseException:
            self._conn.close()
            raise
        self._it = self._generate()

    def dictionary(self) -> Dict[str, Dict[str, Any]]:
        """Data dictionary of exactly the columns this stream emits (columns the static docs do not know
        are described from the database's own declaration)."""
        docs = data_dictionary(self.table)
        for c in self.columns:
            if c not in docs:
                decl = self._declared.get(c, "")
                docs[c] = _col(f"(undocumented column, declared {decl or 'untyped'})", _sqlite_dtype(decl), None, True)
        return {c: docs[c] for c in self.columns}

    def _generate(self) -> Iterator[Dict[str, Any]]:
        try:
            pos = {c: i for i, c in enumerate(self._db_cols)}
            for values in self._conn.execute(self._sql, self._params):
                if self._cam_filter is not None:
                    wanted, cols = self._cam_filter
                    if not any(values[pos[c]] in wanted for c in cols if c in pos):
                        continue
                row = dict(zip(self._db_cols, values))
                for src_col, dst, kind in self._derived:
                    v = row.get(src_col)
                    if kind == "ts":
                        row[dst] = to_local_iso(v) if v is not None else None
                    else:
                        row[dst] = _iso_to_local(v)
                yield row
        finally:
            self.close()

    def __iter__(self) -> "RawStream":
        return self

    def __next__(self) -> Dict[str, Any]:
        return next(self._it)

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def iter_raw(table: str, start: Optional[TimeLike] = None, end: Optional[TimeLike] = None,
             camera_ids: Optional[Iterable[int]] = None, main_db: Optional[str] = None,
             analytics_db: Optional[str] = None) -> RawStream:
    """
    Rows of a raw table for ``[start, end)`` (observation / analytics windows: those overlapping it;
    sightings and handoffs: by timestamp, at second resolution), optionally only for ``camera_ids``
    (handoffs: either end). Tables: ``observation_windows``, ``vehicle_sightings``, ``handoff_events``,
    ``cameras``, ``analytics_windows`` (aliases ``windows``, ``sightings``, ``handoffs``, ``analytics``).
    BLOB columns are never included and a ``*_local`` (WIB) column is added next to each UTC time.
    """
    return RawStream(table, start, end, camera_ids, main_db, analytics_db)


def export_raw(table: str, start: Optional[TimeLike], end: Optional[TimeLike],
               camera_ids: Optional[Iterable[int]] = None, fmt: str = "csv",
               out: Union[str, os.PathLike, TextIO, None] = None, main_db: Optional[str] = None,
               analytics_db: Optional[str] = None) -> Dict[str, Any]:
    """Writes :func:`iter_raw` to ``out`` as csv/jsonl (parquet needs pyarrow). Returns the
    :func:`write_dataset` summary plus ``table``, ``source`` and ``dictionary``."""
    stream = iter_raw(table, start, end, camera_ids, main_db, analytics_db)
    dictionary = stream.dictionary()
    info = write_dataset(stream, fmt, out, columns=stream.columns,
                         dtypes={c: m["dtype"] for c, m in dictionary.items()})
    info.update(table=stream.table, source=stream.source_path, dictionary=dictionary)
    return info


# ── Provenance & bundle ─────────────────────────────────────────────────────────

CAVEATS = [
    "Volume is counting-line crossings only (observation_windows). Track-based volume is not part of this "
    "dataset and must never be mixed with it: tracks are inflated ~10x by track fragmentation.",
    "Cameras are observed intermittently (rolling visits of 2-20 s, ~1 min windows for watched cameras). vph "
    "is crossings per OBSERVED second, i.e. the flow while the camera was watched; crossings is not the "
    "bucket's total traffic. Use observed_s / coverage_ratio / low_sample to weight or filter rows.",
    "A window that spans a bucket edge is split by time-overlap fraction (crossings assumed uniform in "
    "time inside the window), so crossings can be fractional.",
    "Congestion, speed and occupancy come from the analytics engine's own windows (a separate source from the "
    "observation windows) as window-length-weighted means; they are null where no analytics window overlaps "
    "the bucket or the metric was not recorded (speed in body sizes/s and queue share exist only for windows "
    "recorded after they were introduced).",
    "Counting direction (vph_dir_pos/neg) is relative to each camera's own counting line.",
    "Buckets are aligned to local wall-clock time (Asia/Jakarta, WIB, UTC+7, no DST); time-of-day features use "
    "that clock. Only buckets lying completely inside [start, end) are produced.",
    _BLOB_NOTE,
]


def _source_info(main_path: str, analytics_path: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {}
    for key, path, probes in (
        ("main_db", main_path, (("observation_windows", "MIN(end_ts)", "MAX(end_ts)", "epoch"),
                                ("vehicle_sightings", "MIN(timestamp_iso)", "MAX(timestamp_iso)", "iso"),
                                ("handoff_events", "MIN(timestamp_iso)", "MAX(timestamp_iso)", "iso"))),
        ("analytics_db", analytics_path, (("camera_traffic_timeseries", "MIN(timestamp)", "MAX(timestamp)", "epoch"),)),
    ):
        entry: Dict[str, Any] = {"path": str(Path(path).expanduser().resolve()), "exists": os.path.exists(path)}
        if entry["exists"]:
            entry["size_bytes"] = os.path.getsize(path)
            try:
                conn = _open_ro(path)
            except (sqlite3.Error, FileNotFoundError) as exc:
                entry["error"] = str(exc)
            else:
                try:
                    for table, lo, hi, kind in probes:
                        if not _table_columns(conn, table):
                            continue
                        try:
                            a, b = conn.execute(f"SELECT {lo}, {hi} FROM {table}").fetchone()
                        except sqlite3.Error:
                            continue
                        entry[table] = ({"min_ts": a, "max_ts": b, "min_utc": to_utc_iso(a) if a is not None else None,
                                         "max_utc": to_utc_iso(b) if b is not None else None} if kind == "epoch"
                                        else {"min_iso": a, "max_iso": b})
                finally:
                    conn.close()
        info[key] = entry
    return info


def build_provenance(stream: Optional[DatasetStream] = None, *, rows: Optional[int] = None,
                     main_db: Optional[str] = None, analytics_db: Optional[str] = None,
                     files: Optional[List[Dict[str, Any]]] = None, extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Provenance record of an export: generation time, source databases with their time ranges, bucket,
    timezone, layout, row count and the caveats a reader must know. Call it after the stream is
    exhausted so the row count and statistics are final.
    """
    main_path = (stream.source_paths["main_db"] if stream else None) or main_db or default_main_db()
    an_path = (stream.source_paths["analytics_db"] if stream else None) or analytics_db or default_analytics_db()
    doc: Dict[str, Any] = {
        "generated_at": to_utc_iso(time.time()),
        "generator": "backend/dataset.py",
        "schema_version": SCHEMA_VERSION,
        "timezone": LOCAL_TZ_NAME,
        "utc_offset": "+%02d:%02d" % divmod(LOCAL_UTC_OFFSET_S // 60, 60),
        "volume_basis": "crossings",
        "sources": _source_info(main_path, an_path),
        "caveats": CAVEATS,
    }
    if stream is not None:
        doc.update({
            "range_requested": {"start_utc": to_utc_iso(stream.start_ts), "end_utc": to_utc_iso(stream.end_ts),
                                "start_local": to_local_iso(stream.start_ts), "end_local": to_local_iso(stream.end_ts)},
            "range_covered": {"start_utc": to_utc_iso(stream.first_bucket_ts), "end_utc": to_utc_iso(stream.end_bucket_ts),
                              "start_local": to_local_iso(stream.first_bucket_ts), "end_local": to_local_iso(stream.end_bucket_ts)},
            "bucket_minutes": stream.bucket_minutes,
            "layout": stream.layout,
            "value": stream.value if stream.layout == "wide" else None,
            "fill_gaps": stream.fill_gaps,
            "low_sample_s": stream.low_sample_s,
            "cameras": stream.camera_ids,
            "n_cameras": len(stream.camera_ids),
            "rows": stream.stats["rows"] if rows is None else rows,
            "stats": dict(stream.stats),
        })
    elif rows is not None:
        doc["rows"] = rows
    if files:
        doc["files"] = files
    if extra:
        doc.update(extra)
    return doc


def _stamp(stream: DatasetStream) -> str:
    return (f"{datetime.fromtimestamp(stream.first_bucket_ts, LOCAL_TZ):%Y%m%dT%H%M}_"
            f"{datetime.fromtimestamp(stream.end_bucket_ts, LOCAL_TZ):%Y%m%dT%H%M}")


def export_bundle(start: TimeLike, end: TimeLike, out_dir: Union[str, os.PathLike, None] = None, *,
                  bucket_minutes: Union[int, float] = 15, camera_ids: Optional[Iterable[int]] = None,
                  fill_gaps: bool = False, fmt: str = "csv", layout: str = "long", value: str = "vph",
                  raw: Sequence[str] = (), raw_fmt: Optional[str] = None, main_db: Optional[str] = None,
                  analytics_db: Optional[str] = None, low_sample_s: float = LOW_SAMPLE_S,
                  sequences: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Writes a complete export to ``out_dir`` (default ``data/exports``): the dataset, the requested raw
    tables, ``dictionary.json`` and ``provenance.json``. ``fmt`` is ``csv``/``jsonl``/``parquet`` or
    ``npz`` (LSTM tensors via :func:`build_sequences`; ``sequences`` holds its keyword arguments).
    Returns the provenance document.
    """
    t0 = time.time()
    out = Path(out_dir or default_export_dir()).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    cams = None if camera_ids is None else sorted({int(c) for c in camera_ids})
    fmt = fmt.lower()
    stream = build_dataset(start, end, bucket_minutes, cams, fill_gaps, main_db, analytics_db, layout=layout,
                           value=value, low_sample_s=low_sample_s, order="camera" if fmt == "npz" else None)
    stem = f"traffic_{layout}_{int(stream.bucket_minutes)}min_{_stamp(stream)}"
    files: List[Dict[str, Any]] = []
    if fmt == "npz":
        import numpy as np
        seq = build_sequences(stream, **(sequences or {}))
        path = out / f"{stem}_sequences.npz"
        np.savez_compressed(path, **seq)
        files.append({"name": path.name, "format": "npz", "rows": int(seq["X"].shape[0]), "bytes": path.stat().st_size,
                      "shapes": {"X": list(seq["X"].shape), "y": list(seq["y"].shape)}})
        dictionary_name = "dataset"
    else:
        info = write_dataset(stream, fmt, out / f"{stem}.{fmt}")
        files.append({"name": Path(info["path"]).name, "format": fmt, "rows": info["rows"], "bytes": info["bytes"],
                      "kind": f"dataset_{layout}"})
        dictionary_name = "dataset" if layout == "long" else "dataset_wide"
    dictionaries = {f"traffic_{layout}": data_dictionary(dictionary_name, stream.camera_ids, value)}

    rfmt = raw_fmt or (fmt if fmt in ("csv", "jsonl") else "csv")
    for table in raw:
        canon = _canonical_table(table)
        suffix = "" if canon == "cameras" else f"_{_stamp(stream)}"
        info = export_raw(canon, stream.first_bucket_ts, stream.end_bucket_ts, cams, rfmt,
                          out / f"raw_{canon}{suffix}.{rfmt}", main_db, analytics_db)
        files.append({"name": Path(info["path"]).name, "format": rfmt, "rows": info["rows"], "bytes": info["bytes"],
                      "kind": f"raw_{canon}"})
        dictionaries[f"raw_{canon}"] = info["dictionary"]

    prov = build_provenance(stream, main_db=main_db, analytics_db=analytics_db, files=files,
                            extra={"elapsed_s": round(time.time() - t0, 3), "out_dir": str(out.resolve())})
    (out / "dictionary.json").write_text(json.dumps({
        "generated_at": prov["generated_at"], "timezone": LOCAL_TZ_NAME, "tables": dictionaries,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "provenance.json").write_text(json.dumps(prov, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return prov
