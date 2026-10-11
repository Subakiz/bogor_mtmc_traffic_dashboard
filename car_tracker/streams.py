import json
import os
from typing import Dict

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_STREAMS_PATH = os.path.join(_PROJECT_DIR, "cctv_streams.json")


def load_cctv_streams(json_path: str = DEFAULT_STREAMS_PATH) -> Dict[int, dict]:
    """Return ``{camera_id: info}`` from the scraped CCTV stream list."""
    if not os.path.exists(json_path):
        return {}
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {item["id"]: item for item in data}
