"""
Extra camera sources beyond Kota Bogor's BSW portal.

Each source is a JSON file ``data/sources/<name>.json`` holding camera nodes in the topology node
shape (id, name, latitude, longitude, stream_url, corridor, ...). They are appended to the topology
and the cameras table with no road edges, so the rolling tracker just visits them like any
peripheral camera. Ids are namespaced (``ID_BASE`` per source) so they never collide with Bogor's.

``scrape_extra_sources.py`` refreshes the files from the public portals via the adapters below.
Set ``EXTRA_CAMERA_SOURCES=0`` to ignore the files (the test suite does).
"""

import glob
import json
import os
import re
import urllib.request
from typing import Dict, List, Optional

SOURCES_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "sources")
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}


def load_extra_nodes(sources_dir: Optional[str] = None) -> List[Dict]:
    """Nodes of every source file; [] when disabled (env) or the directory is missing."""
    if sources_dir is None:
        if os.environ.get("EXTRA_CAMERA_SOURCES", "1").lower() in ("0", "false", "no", "off"):
            return []
        sources_dir = SOURCES_DIR
    nodes: List[Dict] = []
    for path in sorted(glob.glob(os.path.join(sources_dir, "*.json"))):
        with open(path, "r", encoding="utf-8") as f:
            nodes.extend(json.load(f).get("nodes", []))
    return nodes


# --- Tasikmalaya city ATCS -------------------------------------------------------------------

TASIK_URL = "https://atcs.tasikmalayakota.go.id/"
TASIK_ID_BASE = 1000
TASIK_CORRIDOR = "TASIKMALAYA"


def _junction(name: str) -> str:
    """'Simpang Alun-alun Arah Otista' -> 'Simpang Alun-alun' (cameras of one junction share it)."""
    return re.sub(r"\s+(Arah\s.*|PTZ|FIX)$", "", name).strip()


def parse_tasikmalaya(html: str) -> List[Dict]:
    """Camera nodes from the portal's home page, which embeds ``var cctv={...};`` with name,
    coordinates and a plain HTTPS .m3u8 per camera."""
    m = re.search(r"var cctv=(\{.*?\});\s*\n", html)
    if not m:
        raise RuntimeError("Tasikmalaya page has no camera list; the site layout may have changed.")
    nodes = []
    for cam in json.loads(m.group(1)).values():
        lat, lng = float(cam["lokasi_lat"]), float(cam["lokasi_lng"])
        nodes.append({
            "id": TASIK_ID_BASE + int(cam["id"]),
            "name": cam["nama"],
            "junction_name": _junction(cam["nama"]),
            "corridor": TASIK_CORRIDOR,
            "latitude": lat, "longitude": lng, "lat": lat, "lng": lng,
            "camera_type": "PTZ" if cam["jenis"].upper() == "PTZ" else "FIXED",
            "stream_url": cam["link"],
            "detail_url": TASIK_URL,
            "status": "online",
            "source": "tasikmalaya",
            "source_id": int(cam["id"]),
        })
    return sorted(nodes, key=lambda n: n["id"])


def fetch_tasikmalaya() -> List[Dict]:
    req = urllib.request.Request(TASIK_URL, headers=HEADERS)
    return parse_tasikmalaya(urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore"))


ADAPTERS = {"tasikmalaya": fetch_tasikmalaya}
