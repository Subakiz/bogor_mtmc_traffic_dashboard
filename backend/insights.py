"""
backend/insights.py
Citywide patterns for the dashboard's City tab, folded by WIB hour of day from the export's own
hourly buckets (:func:`backend.dataset.build_dataset`), so every figure matches the dataset export.

* Flow is per camera: each (camera, hour) bucket's veh/h from counting-line crossings over observed
  time, summarised by the median across camera-hours weighted by observed seconds (a few very busy
  cameras can't dominate, a short glimpse weighs less than a long visit, and which cameras the rolling
  tracker happened to visit in an hour matters less). Buckets under ``low_sample_s`` are left out.
* Congestion is the share of junction-hours (one per junction cluster and day) that were Moderate or
  worse, because the citywide mean index stays "Smooth" all day (quiet cameras dilute the hotspots).
  A junction counts as slow when any of its cameras was (the busy side of a junction is what you meet).
* "Today" (WIB calendar day of ``now``) is kept apart from the baseline (the ``days`` before it). The
  hour profile's baseline holds only days of today's kind (weekdays or weekend days) when it has any,
  because weekend days have no rush hours; otherwise all days.
"""

from __future__ import annotations

import math
import time
from datetime import datetime
from typing import Any, Dict, Hashable, Iterable, List, Optional, Sequence, Tuple

from car_tracker.analytics import LOCAL_TZ
from car_tracker.congestion import congestion_label
from backend.bucketing import LOCAL_UTC_OFFSET_S
from backend.dataset import LOW_SAMPLE_S, build_dataset

MIN_ANALYTICS_S = 60.0          # a camera-hour needs this much congestion measurement to count
RANK_MIN_ANALYTICS_S = 900.0    # a camera needs 15 min of daytime congestion readings to be ranked
RANK_MIN_OBSERVED_S = 1800.0    # ... and 30 min of observed flow to be ranked by flow
DAYTIME = range(6, 22)          # hours the congestion ranking averages over (night is empty roads)
RANK_SIZE = 8
CLASSES = ("car", "motorcycle", "truck", "bus")


def _wmedian(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """Median of (value, weight) pairs; equal weights give the plain median."""
    pairs = sorted(p for p in pairs if p[1] > 0)
    if not pairs:
        return None
    half, cum = sum(w for _, w in pairs) / 2.0, 0.0
    for i, (v, w) in enumerate(pairs):
        cum += w
        if math.isclose(cum, half) and i + 1 < len(pairs):
            return round((v + pairs[i + 1][0]) / 2.0, 1)
        if cum > half:
            return round(v, 1)
    return round(pairs[-1][0], 1)


def _wmean(s: float, w: float, nd: int = 3) -> Optional[float]:
    return round(s / w, nd) if w > 0 else None


class _Hour:
    __slots__ = ("vph", "junctions", "days")

    def __init__(self) -> None:
        self.vph: List[Tuple[float, float]] = []            # (veh/h, observed seconds) per camera-hour
        self.junctions: Dict[Tuple[str, Hashable], bool] = {}   # (day, junction) -> slow
        self.days: set = set()

    def add(self, r: Dict[str, Any], day: str, junction: Hashable, low_sample_s: float) -> None:
        if r["vph"] is not None and r["observed_s"] >= low_sample_s:
            self.vph.append((r["vph"], r["observed_s"]))
            self.days.add(day)
        if r["congestion_index"] is not None and r["analytics_s"] >= MIN_ANALYTICS_S:
            key = (day, junction)
            self.junctions[key] = self.junctions.get(key, False) or r["congestion_label"] != "Smooth"

    def out(self) -> Dict[str, Any]:
        measured = len(self.junctions)
        return {"vph": _wmedian(self.vph), "camera_hours": len(self.vph), "n_days": len(self.days),
                "slow_share": round(sum(self.junctions.values()) / measured, 3) if measured else None,
                "measured": measured}


def citywide_insights(days: int = 7, now: Optional[float] = None, camera_ids: Optional[Sequence[int]] = None,
                      clusters: Optional[Dict[int, Iterable[int]]] = None, main_db: Optional[str] = None,
                      analytics_db: Optional[str] = None, low_sample_s: float = LOW_SAMPLE_S) -> Dict[str, Any]:
    """Hour-of-day profile (baseline and today), corridor x hour congestion, class mix and junction
    rankings over the ``days`` WIB days before today plus today so far. ``clusters`` maps a camera to
    the cameras of its junction (rankings keep one camera per junction); ``camera_ids=None``: all."""
    now = time.time() if now is None else float(now)
    off = LOCAL_UTC_OFFSET_S
    today_start = math.floor((now + off) / 86400) * 86400 - off
    start = today_start - int(days) * 86400
    end = math.ceil((now + off) / 3600) * 3600 - off           # through the hour holding ``now``

    local_now = datetime.fromtimestamp(now, LOCAL_TZ)
    weekend = local_now.weekday() >= 5
    base_same = [_Hour() for _ in range(24)]        # baseline days of today's kind
    base_all = [_Hour() for _ in range(24)]
    today = [_Hour() for _ in range(24)]
    mix = [[0.0] * len(CLASSES) for _ in range(24)]             # crossings per class, all days
    corridors: Dict[str, Dict[str, Any]] = {}
    cams: Dict[int, Dict[str, Any]] = {}
    observed_total = 0.0
    days_seen: set = set()

    for r in build_dataset(start, end, 60, camera_ids=camera_ids, main_db=main_db, analytics_db=analytics_db,
                           low_sample_s=low_sample_s):
        h, ts, obs = r["hour"], r["bucket_start_ts"], r["observed_s"]
        day = r["bucket_start_local"][:10]
        junction = frozenset((clusters or {}).get(r["camera_id"], ())) or r["camera_id"]
        if ts >= today_start:
            today[h].add(r, day, junction, low_sample_s)
        else:
            base_all[h].add(r, day, junction, low_sample_s)
            if bool(r["is_weekend"]) == weekend:
                base_same[h].add(r, day, junction, low_sample_s)
        observed_total += obs
        if obs > 0:
            days_seen.add(day)
        crossings = r["crossings"] or 0.0
        if obs > 0:
            for i, c in enumerate(CLASSES):
                mix[h][i] += (r[f"vph_{c}"] or 0.0) * obs / 3600.0

        cor = corridors.setdefault(r["corridor"] or "OTHER", {
            "ci": [0.0] * 24, "w": [0.0] * 24, "obs": 0.0, "crossings": 0.0, "classes": [0.0] * len(CLASSES)})
        cam = cams.setdefault(r["camera_id"], {
            "name": r["camera_name"], "corridor": r["corridor"], "obs": 0.0, "crossings": 0.0,
            "ci": 0.0, "w": 0.0, "hour_ci": [0.0] * 24, "hour_w": [0.0] * 24})
        cor["obs"] += obs
        cor["crossings"] += crossings
        cam["obs"] += obs
        cam["crossings"] += crossings
        if obs > 0:
            for i, c in enumerate(CLASSES):
                cor["classes"][i] += (r[f"vph_{c}"] or 0.0) * obs / 3600.0
        ci, w = r["congestion_index"], r["analytics_s"]
        if ci is not None and w > 0:
            cor["ci"][h] += ci * w
            cor["w"][h] += w
            cam["hour_ci"][h] += ci * w
            cam["hour_w"][h] += w
            if h in DAYTIME:
                cam["ci"] += ci * w
                cam["w"] += w

    same = any(x.vph for x in base_same)
    base = base_same if same else base_all
    hours = []
    for h in range(24):
        total = sum(mix[h])
        hours.append({"hour": h, **base[h].out(), "today": today[h].out(),
                      "mix": {c: round(mix[h][i] / total, 3) if total else None for i, c in enumerate(CLASSES)}})

    corridor_rows = []
    for name, c in corridors.items():
        total = sum(c["classes"])
        cells = []
        for h in range(24):
            v = _wmean(c["ci"][h], c["w"][h])
            cells.append({"ci": v, "label": congestion_label(v), "analytics_s": round(c["w"][h], 1)})
        measured = [h for h in range(24) if cells[h]["ci"] is not None and cells[h]["analytics_s"] >= MIN_ANALYTICS_S]
        peak_h = max(measured, key=lambda h: cells[h]["ci"], default=None)      # ties: the earliest hour
        corridor_rows.append({
            "corridor": name, "cells": cells,
            "vph": round(c["crossings"] * 3600.0 / c["obs"], 1) if c["obs"] > 0 else None,
            "observed_s": round(c["obs"], 1),
            "mix": {k: round(c["classes"][i] / total, 3) if total else None for i, k in enumerate(CLASSES)},
            "peak_hour": peak_h, "peak_ci": None if peak_h is None else cells[peak_h]["ci"],
        })
    corridor_rows.sort(key=lambda c: -(c["peak_ci"] or 0))

    ranked = []
    for cid, c in cams.items():
        worst_h, worst = None, None
        for h in DAYTIME:
            if c["hour_w"][h] >= 2 * MIN_ANALYTICS_S:
                v = c["hour_ci"][h] / c["hour_w"][h]
                if worst is None or v > worst:
                    worst_h, worst = h, v
        ci = _wmean(c["ci"], c["w"])
        ranked.append({
            "camera_id": cid, "name": c["name"], "corridor": c["corridor"],
            "congestion": ci, "congestion_label": congestion_label(ci), "analytics_s": round(c["w"], 1),
            "worst_hour": worst_h, "worst_congestion": None if worst is None else round(worst, 3),
            "worst_label": congestion_label(worst),
            "vph": round(c["crossings"] * 3600.0 / c["obs"], 1) if c["obs"] > 0 else None,
            "observed_s": round(c["obs"], 1),
        })

    def top(key: str, ok) -> List[Dict[str, Any]]:
        out, seen = [], set()
        for row in sorted((r for r in ranked if ok(r)), key=lambda r: -r[key]):
            junction = frozenset((clusters or {}).get(row["camera_id"], ())) or frozenset([row["camera_id"]])
            if junction & seen:
                continue
            seen |= junction
            out.append({**row, "junction_size": len(junction)})
            if len(out) == RANK_SIZE:
                break
        return out

    return {
        "generated_at": now, "now_hour": local_now.hour, "today": local_now.date().isoformat(),
        "baseline_days": len(days_seen - {local_now.date().isoformat()}), "window_days": int(days),
        "baseline_kind": ("weekend" if weekend else "weekday") if same else "all",
        "observed_hours": round(observed_total / 3600.0, 1), "cameras": len(cams),
        "hours": hours, "corridors": corridor_rows,
        "congested": top("congestion", lambda r: r["congestion"] is not None and r["analytics_s"] >= RANK_MIN_ANALYTICS_S),
        "busiest": top("vph", lambda r: r["vph"] is not None and r["observed_s"] >= RANK_MIN_OBSERVED_S),
    }
