#!/usr/bin/env python3
"""
Refreshes data/sources/<name>.json from the public camera portals in backend/camera_sources.py,
probes each playlist, and upserts the cameras into data/traffic_db.sqlite.

  .venv/bin/python3.11 scrape_extra_sources.py [--source tasikmalaya] [--no-check] [--no-db]

Restart the dashboard afterwards (scripts/start_daemon.sh).
"""
import argparse
import json
import os
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from backend.camera_sources import ADAPTERS, SOURCES_DIR
from scrape_cctv_streams import DB_PATH, probe_stream


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=sorted(ADAPTERS), action="append", help="default: all")
    ap.add_argument("--no-check", action="store_true", help="skip probing each playlist")
    ap.add_argument("--no-db", action="store_true", help="only write the JSON file")
    args = ap.parse_args()

    os.makedirs(SOURCES_DIR, exist_ok=True)
    for name in args.source or sorted(ADAPTERS):
        nodes = ADAPTERS[name]()
        if not args.no_check:
            with ThreadPoolExecutor(max_workers=8) as pool:
                for n, ok in zip(nodes, pool.map(lambda n: probe_stream(n["stream_url"]), nodes)):
                    n["status"] = "online" if ok else "offline"
        path = os.path.join(SOURCES_DIR, f"{name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"source": name, "scraped_at": datetime.now(timezone.utc).isoformat(), "nodes": nodes},
                      f, indent=2, ensure_ascii=False)
        print(f"{name}: {len(nodes)} cameras, {sum(n['status'] == 'online' for n in nodes)} online -> {path}")
        if not args.no_db and os.path.exists(DB_PATH):
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            try:
                with conn:
                    for n in nodes:
                        conn.execute("INSERT OR IGNORE INTO cameras (id, name, corridor, latitude, longitude, camera_type, "
                                     "stream_url, status) VALUES (?,?,?,?,?,?,?,?)",
                                     (n["id"], n["name"], n["corridor"], n["latitude"], n["longitude"],
                                      n["camera_type"], n["stream_url"], n["status"]))
                        conn.execute("UPDATE cameras SET name=?, stream_url=?, status=? WHERE id=?",
                                     (n["name"], n["stream_url"], n["status"], n["id"]))
            finally:
                conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
