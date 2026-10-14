"""
backend/bucketing.py
The one rule for cutting time into buckets, shared by the Analytics chart and the dataset export.

* Buckets are whole minutes dividing a day, aligned to LOCAL wall-clock boundaries (Asia/Jakarta,
  WIB, UTC+7, no DST): bucket ``k`` of length ``bs`` covers UTC ``[k*bs - off, (k+1)*bs - off)``.
* An interval (an observation or analytics window) is spread over the buckets it overlaps in
  proportion to the overlap, so sums over buckets equal the sums over windows.
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Iterator, Tuple, Union

try:  # IANA zone when the system has tzdata, otherwise the same fixed offset (WIB has no DST)
    from zoneinfo import ZoneInfo
    LOCAL_TZ = ZoneInfo("Asia/Jakarta")
except Exception:  # noqa: BLE001  (ZoneInfoNotFoundError, ImportError)
    LOCAL_TZ = timezone(timedelta(hours=7), "WIB")
LOCAL_TZ_NAME = "Asia/Jakarta"
LOCAL_UTC_OFFSET_S = int(datetime(2026, 1, 1, tzinfo=LOCAL_TZ).utcoffset().total_seconds())   # 25200


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


def bucket_range(start_ts: float, end_ts: float, bs: int) -> Tuple[int, int]:
    """Indices ``[k0, k1)`` of the buckets lying completely inside ``[start_ts, end_ts)``; bucket
    ``k`` covers UTC ``[k*bs - off, (k+1)*bs - off)``, i.e. it is aligned to local wall-clock time."""
    k0 = math.ceil((start_ts + LOCAL_UTC_OFFSET_S) / bs)
    k1 = math.floor((end_ts + LOCAL_UTC_OFFSET_S) / bs)
    return k0, max(k0, k1)


def spread(s_ts: float, e_ts: float, k0: int, k1: int, bs: int) -> Iterator[Tuple[int, float, float]]:
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
