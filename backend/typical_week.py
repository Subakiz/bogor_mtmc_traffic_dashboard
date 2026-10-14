"""
backend/typical_week.py
The typical congestion of each (weekday, hour), per camera and per junction cluster, so the
dashboard can answer "when is this area usually busy?" (e.g. planning a meeting).

* Source: the hourly rows of ``dataset.build_dataset`` (same WIB bucketing as the chart and the
  export), congestion index only. An hour on a given day counts when the analytics watched it for
  at least ``MIN_SAMPLE_S`` seconds.
* Across days the typical value is the ``QUANTILE`` (75th percentile), not the mean: calling a busy
  hour quiet is the worse error, so one jammed Monday out of four still shows. A cluster takes its
  busiest camera for the same reason.
* A slot built from fewer than ``MIN_DAYS`` days is stored with ``low_data = 1``; readers decide how
  to show it. Weekday/hour is the whole grid, so a "typical week" is the 7 x 24 slots.
* Stored in ``typical_congestion`` (main DB), replaced wholesale by :func:`refresh`.
"""

from __future__ import annotations

import sqlite3
import time
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from backend import database, dataset

MIN_SAMPLE_S = 60.0      # analytics seconds an hour needs on a day to count
MIN_DAYS = 2             # fewer days than this is flagged low_data
QUANTILE = 0.75          # across days; above the median so busy hours are not averaged away
DEFAULT_DAYS = 28

DDL = """
CREATE TABLE IF NOT EXISTS typical_congestion (
    scope TEXT NOT NULL,                 -- 'camera' | 'cluster'
    ref_id INTEGER NOT NULL,             -- camera id, or the lowest camera id of the cluster
    day_of_week INTEGER NOT NULL,        -- 0 = Monday (WIB)
    hour INTEGER NOT NULL,               -- 0-23 (WIB)
    value REAL NOT NULL,                 -- congestion index 0-1
    n_days INTEGER NOT NULL,
    analytics_s REAL NOT NULL,
    low_data INTEGER NOT NULL,
    computed_at REAL NOT NULL,
    PRIMARY KEY (scope, ref_id, day_of_week, hour)
) WITHOUT ROWID;
"""


def _quantile(values: Sequence[float], q: float) -> float:
    v = sorted(values)
    pos = (len(v) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def typical_slots(rows: Iterable[Dict[str, Any]], clusters: Iterable[Sequence[int]]) -> List[Dict[str, Any]]:
    """Slots for every camera in ``rows`` and for every cluster containing one (a camera in no
    listed cluster is its own cluster). ``rows`` are hourly dataset rows."""
    days: Dict[Tuple[int, int, int], Dict[str, Tuple[float, float]]] = defaultdict(dict)
    for r in rows:
        c, s = r.get("congestion_index"), r.get("analytics_s") or 0.0
        if c is None or s < MIN_SAMPLE_S:
            continue
        days[(int(r["camera_id"]), int(r["day_of_week"]), int(r["hour"]))][r["bucket_start_local"][:10]] = (c, s)

    cameras: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for (cam, dow, hour), by_day in days.items():
        n = len(by_day)
        cameras[(cam, dow, hour)] = {"value": _quantile([c for c, _ in by_day.values()], QUANTILE), "n_days": n,
                                     "analytics_s": sum(s for _, s in by_day.values())}

    out = [{"scope": "camera", "ref_id": cam, "day_of_week": dow, "hour": hour, **v}
           for (cam, dow, hour), v in cameras.items()]
    member_of: Dict[int, Tuple[int, ...]] = {}
    for cl in clusters:
        for cam in cl:
            member_of[int(cam)] = tuple(sorted(int(x) for x in cl))
    groups: Dict[Tuple[Tuple[int, ...], int, int], List[Dict[str, Any]]] = defaultdict(list)
    for (cam, dow, hour), v in cameras.items():
        groups[(member_of.get(cam, (cam,)), dow, hour)].append(v)
    for (members, dow, hour), vs in groups.items():
        out.append({"scope": "cluster", "ref_id": members[0], "day_of_week": dow, "hour": hour,
                    "value": max(v["value"] for v in vs), "n_days": max(v["n_days"] for v in vs),
                    "analytics_s": sum(v["analytics_s"] for v in vs)})
    for s in out:
        s["value"] = round(s["value"], 4)
        s["low_data"] = int(s["n_days"] < MIN_DAYS)
    return out


def refresh(main_db: str, analytics_db: str, clusters: Iterable[Sequence[int]], now: Optional[float] = None,
            days: int = DEFAULT_DAYS) -> int:
    """Recompute the last ``days`` days of hourly congestion into ``typical_congestion`` (replacing it);
    returns the number of slots stored."""
    now = time.time() if now is None else now
    stream = dataset.build_dataset(now - days * 86400, now, bucket_minutes=60, main_db=main_db,
                                   analytics_db=analytics_db)
    slots = typical_slots(stream, clusters)
    conn = database.get_db_connection(main_db)
    try:
        conn.execute(DDL)
        with database._transaction(conn) as cur:
            cur.execute("DELETE FROM typical_congestion")
            cur.executemany(
                "INSERT INTO typical_congestion (scope, ref_id, day_of_week, hour, value, n_days, analytics_s, "
                "low_data, computed_at) VALUES (:scope, :ref_id, :day_of_week, :hour, :value, :n_days, "
                ":analytics_s, :low_data, :computed_at)", [{**s, "computed_at": now} for s in slots])
    finally:
        conn.close()
    return len(slots)


def load(conn: sqlite3.Connection, scope: str, ref_id: int) -> List[Dict[str, Any]]:
    """Stored slots of one camera or cluster, ordered by weekday then hour."""
    cur = conn.execute("SELECT scope, ref_id, day_of_week, hour, value, n_days, analytics_s, low_data "
                       "FROM typical_congestion WHERE scope = ? AND ref_id = ? ORDER BY day_of_week, hour",
                       (scope, int(ref_id)))
    cols = [d[0] for d in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]
