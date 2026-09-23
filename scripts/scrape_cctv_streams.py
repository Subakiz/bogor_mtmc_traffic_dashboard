#!/usr/bin/env python3
"""
Scrapes all CCTV cameras and direct streaming (.m3u8) links from BSW Kota Bogor.
Outputs to: cctv_streams.json and cctv_streams.csv
"""

import json
import csv
import re
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_LIST_URL = "https://bsw.kotabogor.go.id/cctv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}

def get_cctv_list():
    req = urllib.request.Request(BASE_LIST_URL, headers=HEADERS)
    html = urllib.request.urlopen(req, timeout=15).read().decode("utf-8", errors="ignore")
    pattern = r'<a\s+href=["\'](https://bsw\.kotabogor\.go\.id/cctv/(\d+)/detail)["\'][^>]*>(.*?)</a>'
    matches = re.findall(pattern, html, re.DOTALL)
    
    unique_cams = {}
    for url, cid, name in matches:
        clean_name = re.sub(r'<.*?>|\s+', ' ', name).strip()
        cid = int(cid)
        if cid not in unique_cams:
            unique_cams[cid] = {"id": cid, "name": clean_name, "detail_url": url}
    return sorted(unique_cams.values(), key=lambda x: x["id"])

def fetch_stream_url(cam):
    url = cam["detail_url"]
    cam_id = cam["id"]
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        html = urllib.request.urlopen(req, timeout=12).read().decode("utf-8", errors="ignore")
        
        # Match .m3u8
        m = re.search(r"https?://[^\s\"'\(\)]+\.m3u8", html)
        stream_url = m.group(0) if m else None
        
        # Match restreamer embed if available
        embed_url = None
        if stream_url:
            # E.g. https://restreamer6.kotabogor.go.id/memfs/<uuid>.m3u8 -> https://restreamer6.kotabogor.go.id/<uuid>.html
            m_embed = re.match(r"(https?://[^/]+)/memfs/([a-f0-9\-]+)\.m3u8", stream_url)
            if m_embed:
                embed_url = f"{m_embed.group(1)}/{m_embed.group(2)}.html"
        
        # Also refine title if available inside detail page
        m_title = re.search(r'<span class="sub-title[^"]*">\s*(.*?)\s*</span>', html)
        if m_title:
            cam["name"] = m_title.group(1).strip()

        return {
            "id": cam_id,
            "name": cam["name"],
            "detail_url": url,
            "stream_url": stream_url,
            "embed_url": embed_url,
            "status": "online" if stream_url else "offline"
        }
    except Exception as e:
        return {
            "id": cam_id,
            "name": cam["name"],
            "detail_url": url,
            "stream_url": None,
            "embed_url": None,
            "status": f"error: {str(e)}"
        }

def main():
    print("Fetching CCTV list from:", BASE_LIST_URL)
    cams = get_cctv_list()
    print(f"Discovered {len(cams)} CCTV entries. Fetching individual stream links in parallel...")

    results = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_cam = {executor.submit(fetch_stream_url, cam): cam for cam in cams}
        for future in as_completed(future_to_cam):
            res = future.result()
            results.append(res)
            print(f"[{len(results):02d}/{len(cams):02d}] ID {res['id']:3d}: {res['name']:35s} -> {res['stream_url'] or 'OFFLINE'}")

    results.sort(key=lambda x: x["id"])

    # Save to JSON
    with open("cctv_streams.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save to CSV
    with open("cctv_streams.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "name", "status", "stream_url", "embed_url", "detail_url"])
        writer.writeheader()
        writer.writerows(results)

    online_count = sum(1 for r in results if r["stream_url"])
    print(f"\nFinished! Found {online_count}/{len(results)} active streams.")
    print("Saved to:")
    print(" - cctv_streams.json")
    print(" - cctv_streams.csv")

if __name__ == "__main__":
    main()
