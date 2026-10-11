#!/usr/bin/env python3
"""
Scrapes every CCTV camera and its live HLS (.m3u8) stream from BSW Kota Bogor
(https://bsw.kotabogor.go.id/cctv) and keeps the project's copies in sync.

Writes (paths relative to this file, not the working directory):
  cctv_streams.json            engine stream list (car_tracker.load_cctv_streams)
  data/cctv_streams.json/.csv  same list for the dashboard data folder

and, unless --no-sync is given, updates the dashboard's view of each camera:
  data/bogor_camera_topology.json  stream_url / embed_url / detail_url / name / status of known nodes
  data/traffic_db.sqlite           cameras table (stream_url / name / status)

Cameras the site no longer lists are marked status "removed". New cameras have no coordinates on
the site, so they are only reported: add them to the topology once geocoded.

Usage:
  .venv/bin/python3.11 scrape_cctv_streams.py              # scrape, probe streams, sync
  .venv/bin/python3.11 scrape_cctv_streams.py --no-check   # skip the stream reachability probe
  .venv/bin/python3.11 scrape_cctv_streams.py --no-sync    # only write the stream lists
"""

import argparse
import csv
import json
import os
import re
import sqlite3
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional

BASE_LIST_URL = "https://bsw.kotabogor.go.id/cctv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
WORKERS = 8
RETRIES = 2

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ROOT_JSON = os.path.join(ROOT, "cctv_streams.json")
DATA_JSON = os.path.join(DATA_DIR, "cctv_streams.json")
DATA_CSV = os.path.join(DATA_DIR, "cctv_streams.csv")
TOPOLOGY_JSON = os.path.join(DATA_DIR, "bogor_camera_topology.json")
DB_PATH = os.path.join(DATA_DIR, "traffic_db.sqlite")
CSV_FIELDS = ["id", "name", "status", "reachable", "stream_url", "embed_url", "detail_url"]


def _get(url: str, timeout: float = 15.0) -> str:
    last_exc: Optional[Exception] = None
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            return urllib.request.urlopen(req, timeout=timeout).read().decode("utf-8", errors="ignore")
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(1.0 + attempt)
    raise last_exc  # type: ignore[misc]


def get_cctv_list() -> List[Dict]:
    html = _get(BASE_LIST_URL)
    pattern = r'<a\s+href=["\'](https://bsw\.kotabogor\.go\.id/cctv/(\d+)/detail)["\'][^>]*>(.*?)</a>'
    cams: Dict[int, Dict] = {}
    for url, cid, name in re.findall(pattern, html, re.DOTALL):
        cid = int(cid)
        cams.setdefault(cid, {"id": cid, "name": re.sub(r"<.*?>|\s+", " ", name).strip(), "detail_url": url})
    if not cams:
        raise RuntimeError("No cameras found on the list page; the site layout may have changed.")
    return sorted(cams.values(), key=lambda c: c["id"])


def fetch_stream_url(cam: Dict) -> Dict:
    result = {"id": cam["id"], "name": cam["name"], "detail_url": cam["detail_url"],
              "stream_url": None, "embed_url": None, "status": "offline", "reachable": None}
    try:
        html = _get(cam["detail_url"], timeout=12.0)
    except Exception as exc:  # noqa: BLE001
        result["status"] = f"error: {exc}"
        return result

    m = re.search(r"https?://[^\s\"'()<>]+\.m3u8", html)
    if m:
        result["stream_url"] = m.group(0)
        result["status"] = "online"
        m_embed = re.match(r"(https?://[^/]+)/memfs/([a-f0-9\-]+)\.m3u8", result["stream_url"])
        if m_embed:
            result["embed_url"] = f"{m_embed.group(1)}/{m_embed.group(2)}.html"
    m_title = re.search(r'<span class="sub-title[^"]*">\s*(.*?)\s*</span>', html, re.DOTALL)
    if m_title:
        result["name"] = re.sub(r"\s+", " ", m_title.group(1)).strip()
    return result


def probe_stream(url: str) -> bool:
    """True if the playlist answers with an HLS manifest within a few seconds."""
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=8.0) as resp:
            return resp.status == 200 and b"#EXTM3U" in resp.read(2048)
    except Exception:  # noqa: BLE001
        return False


def write_lists(results: List[Dict]) -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    for path in (ROOT_JSON, DATA_JSON):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
    with open(DATA_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def sync_topology(results: List[Dict]) -> Dict[str, List[int]]:
    with open(TOPOLOGY_JSON, "r", encoding="utf-8") as f:
        topo = json.load(f)
    scraped = {r["id"]: r for r in results}
    changed, removed = [], []
    for node in topo["nodes"]:
        r = scraped.get(node["id"])
        if r is None:
            if node.get("status") != "removed":
                node["status"] = "removed"
                removed.append(node["id"])
            continue
        before = (node.get("stream_url"), node.get("name"), node.get("status"))
        node.update({"stream_url": r["stream_url"], "embed_url": r["embed_url"],
                     "detail_url": r["detail_url"], "name": r["name"], "status": r["status"]})
        if (node["stream_url"], node["name"], node["status"]) != before:
            changed.append(node["id"])
    topo.setdefault("metadata", {})["streams_scraped_at"] = datetime.now(timezone.utc).isoformat()
    tmp = TOPOLOGY_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(topo, f, indent=2, ensure_ascii=False)
    os.replace(tmp, TOPOLOGY_JSON)
    known = {n["id"] for n in topo["nodes"]}
    return {"changed": changed, "removed": removed, "new": sorted(set(scraped) - known)}


def sync_database(results: List[Dict], removed: List[int]) -> int:
    if not os.path.exists(DB_PATH):
        return 0
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        with conn:  # commits, or rolls back on error
            n = 0
            for r in results:
                n += conn.execute("UPDATE cameras SET stream_url = ?, name = ?, status = ? WHERE id = ?",
                                  (r["stream_url"] or "", r["name"], r["status"], r["id"])).rowcount
            for cid in removed:
                conn.execute("UPDATE cameras SET status = 'removed' WHERE id = ?", (cid,))
        return n
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--no-check", action="store_true", help="skip probing each .m3u8 for reachability")
    parser.add_argument("--no-sync", action="store_true", help="don't update the topology JSON or the database")
    args = parser.parse_args()

    print(f"Fetching CCTV list from {BASE_LIST_URL}")
    cams = get_cctv_list()
    print(f"Found {len(cams)} cameras; fetching stream links...")
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = sorted(pool.map(fetch_stream_url, cams), key=lambda r: r["id"])

    if not args.no_check:
        print("Probing streams...")
        with_url = [r for r in results if r["stream_url"]]
        with ThreadPoolExecutor(max_workers=WORKERS) as pool:
            for r, ok in zip(with_url, pool.map(lambda r: probe_stream(r["stream_url"]), with_url)):
                r["reachable"] = ok

    for r in results:
        reach = {True: "ok", False: "DOWN", None: "-"}[r["reachable"]]
        print(f"  {r['id']:4d}  {r['name'][:38]:38s}  {r['status']:8s}  {reach:4s}  {r['stream_url'] or ''}")

    previous = {}
    if os.path.exists(ROOT_JSON):
        with open(ROOT_JSON, "r", encoding="utf-8") as f:
            previous = {c["id"]: c for c in json.load(f)}
    write_lists(results)

    online = sum(1 for r in results if r["stream_url"])
    url_changed = [r["id"] for r in results if r["id"] in previous and previous[r["id"]].get("stream_url") != r["stream_url"]]
    print(f"\n{online}/{len(results)} cameras have a stream URL"
          + (f"; {sum(1 for r in results if r['reachable'])} answered the probe" if not args.no_check else ""))
    print(f"Stream URL changed since last scrape: {url_changed or 'none'}")
    print(f"Wrote {os.path.relpath(ROOT_JSON, ROOT)}, {os.path.relpath(DATA_JSON, ROOT)}, {os.path.relpath(DATA_CSV, ROOT)}")

    if not args.no_sync:
        report = sync_topology(results)
        rows = sync_database(results, report["removed"])
        print(f"Topology: {len(report['changed'])} cameras updated {report['changed'] or ''}; "
              f"removed from site: {report['removed'] or 'none'}")
        print(f"Database: {rows} camera rows refreshed")
        if report["new"]:
            new = [f"{cid} {next(r['name'] for r in results if r['id'] == cid)}" for cid in report["new"]]
            print("New cameras not in the topology yet (need coordinates + road edges to appear on the map):")
            for line in new:
                print(f"  {line}")
        print("Restart the dashboard server to load the new stream URLs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
