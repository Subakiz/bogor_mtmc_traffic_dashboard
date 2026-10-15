"""
backend/config.py
Central place for filesystem locations and tunables shared by the backend modules.

The vehicle detector/tracker and analytics engine (package ``car_tracker``) and the CoreML
models live in the project root next to ``backend/``. Point ``CCTV_YOLO_ANE_ROOT`` elsewhere
to use a different engine checkout.
"""

import glob
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FRONTEND_DIR = PROJECT_ROOT / "frontend"

ENGINE_ROOT = Path(os.environ.get("CCTV_YOLO_ANE_ROOT", PROJECT_ROOT)).expanduser()
DEFAULT_MODEL_PATH = os.environ.get("DEFAULT_MODEL", str(ENGINE_ROOT / "yolov8m.mlpackage"))
# Optional 16:9 export of the same model (``yolo export model=yolov8m.pt format=coreml imgsz=384,640
# nms=True``), used for frames it covers without losing resolution: ~40% less ANE time per frame.
WIDE_MODEL_PATH = os.environ.get("WIDE_MODEL", str(ENGINE_ROOT / "yolov8m_384x640.mlpackage"))
# Writable state; each path can be redirected (the test suite points them at a temp dir so it never
# touches the live databases while the server runs).
ANALYTICS_DB_PATH = os.environ.get("ANALYTICS_DB_PATH") or str(DATA_DIR / "traffic_timeseries.db")
COUNTING_LINES_PATH = os.environ.get("COUNTING_LINES_PATH") or str(DATA_DIR / "counting_lines.json")
# Hand-placed counting lines / excluded cameras (git-tracked, read-only); win over learned lines.
COUNTING_LINE_OVERRIDES_PATH = (os.environ.get("COUNTING_LINE_OVERRIDES_PATH")
                                or str(DATA_DIR / "counting_line_overrides.json"))
FREE_FLOW_PATH = os.environ.get("FREE_FLOW_PATH") or str(DATA_DIR / "free_flow_speeds.json")
TRAFFIC_LANES_PATH = os.environ.get("TRAFFIC_LANES_PATH") or str(DATA_DIR / "traffic_lanes.npz")


def ensure_engine_importable() -> None:
    """Make ``car_tracker`` (and the engine's virtualenv packages) importable."""
    if str(ENGINE_ROOT) not in sys.path:
        sys.path.insert(0, str(ENGINE_ROOT))
    for site in glob.glob(str(ENGINE_ROOT / ".venv" / "lib" / "python*" / "site-packages")):
        if site not in sys.path:
            sys.path.append(site)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


# ── Rolling tracker tunables (all overridable through the environment) ──────────
FOCUS_FPS = _env_float("ROLLING_FOCUS_FPS", 15.0)        # inference rate for the camera being viewed
ROLLING_FPS = _env_float("ROLLING_FPS", 8.0)             # inference rate while rolling through other cameras
DWELL_TIER1_S = _env_float("ROLLING_DWELL_TIER1_S", 20.0)  # seconds per visit: cameras downstream of a focus camera
DWELL_TIER2_S = _env_float("ROLLING_DWELL_TIER2_S", 15.0)  # seconds per visit: arterial hubs
DWELL_TIER3_S = _env_float("ROLLING_DWELL_TIER3_S", 12.0)  # seconds per visit: peripheral cameras
TIER1_REVISIT_S = _env_float("ROLLING_TIER1_REVISIT_S", 8.0)
TIER2_REVISIT_S = _env_float("ROLLING_TIER2_REVISIT_S", 60.0)
MAX_OPEN_STREAMS = _env_int("ROLLING_MAX_STREAMS", 12)   # concurrent HLS connections across all restreamers
MAX_STREAMS_PER_HOST = _env_int("ROLLING_MAX_PER_HOST", 3)   # concurrent connections to any one restreamer host
MAX_FOCUS_CAMERAS = _env_int("ROLLING_MAX_FOCUS", 3)   # cameras watched at once (tabs/viewers), each at FOCUS_FPS
FOCUS_TTL_S = _env_float("ROLLING_FOCUS_TTL_S", 10.0)    # viewer considered gone after this many idle seconds
ROLLING_ENABLED = os.environ.get("ROLLING_ENABLED", "1").lower() not in ("0", "false", "no", "off")

# ── Housekeeping / response caching ─────────────────────────────────────────────
MAINTENANCE_INTERVAL_S = _env_float("MAINTENANCE_INTERVAL_S", 300.0)
# Dashboard aggregates are computed at most once per this many seconds, however many tabs poll.
STATS_CACHE_TTL_S = _env_float("STATS_CACHE_TTL_S", 3.0)

# ── Storage / retention ─────────────────────────────────────────────────────────
# Sightings older than this many days are deleted in small batches by the
# maintenance thread. 0 keeps everything. All-time totals (dashboard counters)
# are maintained by insert triggers and are NOT reduced by pruning; ``observation_windows`` (the
# traffic-volume history) is never pruned.
SIGHTING_RETENTION_DAYS = _env_float("SIGHTING_RETENTION_DAYS", 0.0)
# Cap on the WAL file kept after a checkpoint (PRAGMA journal_size_limit), so one large write
# burst or a long reader cannot leave a multi-hundred-MB WAL on disk.
DB_JOURNAL_SIZE_LIMIT_BYTES = _env_int("DB_JOURNAL_SIZE_LIMIT_BYTES", 32 * 1024 * 1024)

# ── Rolling-tracker scheduling ──────────────────────────────────────────────────
# Core Arterial Junction Hub Camera IDs in Kota Bogor
DEFAULT_ARTERIAL_HUBS = {
    79,   # Tugu Kujang (Pajajaran - Otista)
    80,   # Ciheleut (Pajajaran)
    81,   # Pajajaran Baranangsiang
    82,   # Sambu / Terminal Baranangsiang
    83,   # Pasar Bogor
    84,   # Suryakencana
    85,   # Empang
    86,   # Simpang Pahlawan / Bondongan
    87,   # Otista
    88,   # Balai Kota
    92,   # Djuanda
    161,  # Warung Jambu (Sholeh Iskandar)
    167,  # Sukasari
}
