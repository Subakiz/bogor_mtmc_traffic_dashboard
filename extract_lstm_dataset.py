#!/usr/bin/env python3
"""
Traffic Time-Series Dataset Extractor for LSTM & Deep Learning Models.

Usage:
    # 1. Export standard CSV with all traffic & cyclic time features:
    python3 extract_lstm_dataset.py --cam-id 167 --window 60 --format csv --output dataset.csv

    # 2. Export 3D NumPy sliding-window tensors (X, y) ready for PyTorch / TensorFlow LSTM:
    python3 extract_lstm_dataset.py --cam-id 167 --window 60 --format npz --seq-len 12 --horizon 1 --output lstm_tensors.npz

PyTorch Ingestion Example:
    import numpy as np
    import torch
    import torch.nn as nn

    data = np.load("lstm_tensors.npz")
    X = torch.tensor(data["X"], dtype=torch.float32)  # Shape: (N, seq_len=12, features=15)
    y = torch.tensor(data["y"], dtype=torch.float32)  # Shape: (N, horizon=1)

    lstm = nn.LSTM(input_size=X.shape[2], hidden_size=64, batch_first=True)
    out, (hn, cn) = lstm(X)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List

import numpy as np

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from car_tracker.analytics import (
    DEFAULT_DB_PATH,
    TrafficAnalyticsDatabase,
    TrafficAnalyticsEngine,
)


def main():
    parser = argparse.ArgumentParser(
        description="Extract and preprocess CCTV traffic volume time-series data for LSTM / ML models."
    )
    parser.add_argument(
        "--cam-id",
        type=int,
        default=None,
        help="CCTV camera ID to filter by (default: all monitored cameras).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=60,
        help="Time-bucket duration in seconds (default: 60s).",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["csv", "json", "npz"],
        default="csv",
        help="Output format: 'csv', 'json', or 'npz' (NumPy 3D tensors X, y for LSTM).",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=12,
        help="LSTM lookback sequence length in steps (default: 12 intervals, e.g. 12 minutes for 60s windows).",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=1,
        help="Prediction horizon in steps ahead (default: 1).",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="total_volume",
        help="Target column to predict (default: 'total_volume').",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10000,
        help="Maximum records to retrieve (default: 10000).",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=DEFAULT_DB_PATH,
        help=f"Path to SQLite database (default: {DEFAULT_DB_PATH}).",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Destination file path. If omitted, prints summary to stdout or defaults by format.",
    )

    args = parser.parse_args()

    engine = TrafficAnalyticsEngine(db_path=args.db_path, primary_window_seconds=args.window)

    records = engine.export_timeseries_dataset(
        camera_id=args.cam_id,
        window_seconds=args.window,
        limit=args.limit,
    )

    if not records:
        print("[!] No records found matching query criteria.")
        print(f"    Database: {args.db_path}")
        print("    Tip: run the dashboard so cameras are observed and windows are recorded.")
        sys.exit(1)

    print("=" * 65)
    print("  CCTV Traffic Intelligence — Time-Series Dataset Extractor")
    print("=" * 65)
    print(f"  Records retrieved : {len(records)}")
    print(f"  Camera filter     : {args.cam_id if args.cam_id is not None else 'All cameras'}")
    print(f"  Window bucket     : {args.window} seconds")
    print(f"  Time span         : {records[0]['timestamp_iso']}  →  {records[-1]['timestamp_iso']}")

    vols = [r["total_volume"] for r in records]
    speeds = [r["avg_speed_px_s"] for r in records]
    congs = [r["congestion_index"] for r in records]
    print(f"  Total volume stats: Min={min(vols)}, Max={max(vols)}, Mean={np.mean(vols):.1f}")
    print(f"  Speed proxy stats : Min={min(speeds):.1f}, Max={max(speeds):.1f}, Mean={np.mean(speeds):.1f} px/s")
    print(f"  Congestion stats  : Min={min(congs):.2f}, Max={max(congs):.2f}, Mean={np.mean(congs):.2f}")
    print("-" * 65)

    # Determine output path
    out_path = args.output
    if not out_path:
        suffix = f"cam_{args.cam_id}" if args.cam_id is not None else "all"
        out_path = f"traffic_timeseries_{suffix}_{args.window}s.{args.format}"

    out_file = Path(out_path)

    if args.format == "csv":
        csv_str = engine.export_csv_str(
            camera_id=args.cam_id,
            window_seconds=args.window,
            limit=args.limit,
        )
        out_file.write_text(csv_str)
        print(f"[✓] CSV exported to: {out_file.resolve()}")
        print(f"    Columns: {len(csv_str.splitlines()[0].split(','))} features (includes cyclic temporal encodings).")

    elif args.format == "json":
        out_file.write_text(json.dumps(records, indent=2))
        print(f"[✓] JSON exported to: {out_file.resolve()}")

    elif args.format == "npz":
        seq_data = engine.create_lstm_sequences(
            records=records,
            seq_len=args.seq_len,
            horizon=args.horizon,
            target_col=args.target,
        )
        X = seq_data["X"]
        y = seq_data["y"]
        np.savez_compressed(
            out_file,
            X=X,
            y=y,
            feature_names=seq_data["feature_names"],
            target_name=seq_data["target_name"],
            timestamps=seq_data["timestamps"],
        )
        print(f"[✓] NumPy LSTM tensors saved to: {out_file.resolve()}")
        print(f"    X tensor shape : {X.shape}  (Samples, Seq_Len={args.seq_len}, Features={X.shape[2]})")
        print(f"    y tensor shape : {y.shape}  (Samples, Horizon={args.horizon})")
        print(f"    Feature list   : {seq_data['feature_names']}")

    print("=" * 65)


if __name__ == "__main__":
    main()
