#!/usr/bin/env python3
"""
Export an analysis-ready traffic dataset (tidy per-camera time series) plus raw tables.

Reads the live databases strictly read-only (data/traffic_db.sqlite and data/traffic_timeseries.db),
so it is safe to run while the dashboard server is up. Writes the dataset, optional raw tables,
``dictionary.json`` (column -> description/unit/dtype) and ``provenance.json`` (sources, ranges, row
counts, caveats) into --out (default data/exports/, created on use).

Times are ISO-8601 or epoch seconds; a time without an offset is local WIB (UTC+7). [start, end) is
cut into whole buckets aligned to local wall-clock boundaries.

Examples
    # one day, 15-minute buckets, every observed camera, CSV (long layout)
    .venv/bin/python3.11 scripts/export_dataset.py --start 2026-10-11 --end 2026-10-12

    # two cameras, 5-minute buckets, regular sequences (gaps emitted), JSON lines, plus raw tables
    .venv/bin/python3.11 scripts/export_dataset.py --start 2026-10-11T06:00 --end 2026-10-11T22:00 \\
        --bucket 5 --cameras 167,131 --fill-gaps --format jsonl \\
        --raw observation_windows,sightings,handoffs --out /tmp/export

    # bucket x camera matrix of congestion for spatiotemporal models
    .venv/bin/python3.11 scripts/export_dataset.py --start 2026-10-11 --end 2026-10-12 \\
        --layout wide --value congestion_index

    # LSTM tensors (replaces extract_lstm_dataset.py --format npz)
    .venv/bin/python3.11 scripts/export_dataset.py --start 2026-10-11 --end 2026-10-12 --bucket 5 \\
        --cameras 167 --fill-gaps --format npz --seq-len 12 --horizon 1 --target vph
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend import dataset  # noqa: E402


def _int_list(text: str):
    try:
        return [int(t) for t in text.split(",") if t.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected comma-separated camera ids, got {text!r}") from exc


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start", help="range start, inclusive (default: 24 h before --end)")
    p.add_argument("--end", help="range end, exclusive (default: now)")
    p.add_argument("--bucket", type=float, default=15, metavar="MIN",
                   help="bucket length in minutes; must divide a day: 1, 5, 15, 30, 60, ... (default 15)")
    p.add_argument("--cameras", type=_int_list, metavar="ID,ID", help="camera ids (default: every observed camera)")
    p.add_argument("--format", choices=("csv", "jsonl", "parquet", "npz"), default="csv",
                   help="parquet needs pyarrow; npz writes LSTM tensors (default csv)")
    p.add_argument("--layout", choices=("long", "wide"), default="long",
                   help="long: one row per camera x bucket; wide: one row per bucket, a column per camera")
    p.add_argument("--value", default="vph", help="metric in the wide cells (any numeric dataset column; default vph)")
    p.add_argument("--fill-gaps", action="store_true",
                   help="also emit buckets nobody observed (observed_s = 0, null metrics): regular sequences")
    p.add_argument("--low-sample-s", type=float, default=dataset.LOW_SAMPLE_S, metavar="S",
                   help="buckets with less observed time are flagged low_sample (default 60)")
    p.add_argument("--raw", default="", metavar="TABLES",
                   help="raw tables for the same range, comma-separated or 'all': observation_windows, "
                        "sightings (never blobs), handoffs, cameras, analytics")
    p.add_argument("--raw-format", choices=("csv", "jsonl"), help="format of the raw files (default: --format, else csv)")
    p.add_argument("--out", default=dataset.default_export_dir(), metavar="DIR",
                   help="output directory (default: data/exports)")
    p.add_argument("--main-db", help="traffic database (default: data/traffic_db.sqlite)")
    p.add_argument("--analytics-db", help="analytics database (default: data/traffic_timeseries.db)")
    g = p.add_argument_group("npz (LSTM tensors)")
    g.add_argument("--seq-len", type=int, default=12, help="steps of history per sample (default 12)")
    g.add_argument("--horizon", type=int, default=1, help="steps ahead to predict (default 1)")
    g.add_argument("--target", default="vph", help="column to predict (default vph)")
    g.add_argument("--features", type=lambda t: [c for c in t.split(",") if c], metavar="COL,COL",
                   help=f"input columns (default {','.join(dataset.DEFAULT_SEQUENCE_FEATURES)})")
    g.add_argument("--nan-policy", choices=("drop", "zero"), default="drop",
                   help="windows with a null input: drop them or zero-fill (null targets always drop)")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        end = dataset.parse_time(args.end) if args.end else time.time()
        start = dataset.parse_time(args.start) if args.start else end - 86400
        raw = list(dataset.RAW_TABLES) if args.raw.strip() == "all" else [t for t in args.raw.split(",") if t.strip()]
        sequences = None
        if args.format == "npz":
            sequences = {"seq_len": args.seq_len, "horizon": args.horizon, "target": args.target,
                         "nan_policy": args.nan_policy}
            if args.features:
                sequences["features"] = args.features
            if args.layout != "long":
                raise ValueError("npz output needs --layout long")
            if not args.fill_gaps:
                print("note: sequences need consecutive buckets; consider --fill-gaps", file=sys.stderr)
        elif args.format == "parquet" and not dataset.parquet_available():
            raise RuntimeError("parquet output needs the 'pyarrow' package (pip install pyarrow); use csv or jsonl")
        prov = dataset.export_bundle(
            start, end, args.out, bucket_minutes=args.bucket, camera_ids=args.cameras, fill_gaps=args.fill_gaps,
            fmt=args.format, layout=args.layout, value=args.value, raw=raw, raw_fmt=args.raw_format,
            main_db=args.main_db, analytics_db=args.analytics_db, low_sample_s=args.low_sample_s,
            sequences=sequences)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    cov = prov["range_covered"]
    print(f"range    {cov['start_local']} -> {cov['end_local']}  ({prov['bucket_minutes']:g}-minute buckets, "
          f"{prov['layout']} layout, {prov['n_cameras']} cameras)")
    for f in prov["files"]:
        print(f"wrote    {prov['out_dir']}/{f['name']}  ({f['rows']} rows, {f['bytes']:,} bytes)")
    print(f"wrote    {prov['out_dir']}/dictionary.json, provenance.json")
    st = prov["stats"]
    if st["rows"] == 0:
        print("warning: no rows. Nothing was observed in this range (check --start/--end, camera ids and the "
              "sources in provenance.json).", file=sys.stderr)
    else:
        print(f"observed {st['observed_s']:.0f} s on {st['cameras_with_data']} cameras, "
              f"{st['crossings']:.0f} crossings counted; {prov['elapsed_s']} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
