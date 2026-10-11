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
ANALYTICS_DB_PATH = str(DATA_DIR / "traffic_timeseries.db")


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
DWELL_TIER1_S = _env_float("ROLLING_DWELL_TIER1_S", 20.0)  # seconds per visit: follow-candidate cameras
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
# Appearance blobs of sightings older than this are blanked (re-identification only needs recent
# ones, held in memory). 0 keeps them forever.
EMBEDDING_TTL_S = _env_float("EMBEDDING_TTL_S", 3600.0)
MAINTENANCE_INTERVAL_S = _env_float("MAINTENANCE_INTERVAL_S", 300.0)
# Dashboard aggregates are computed at most once per this many seconds, however many tabs poll.
STATS_CACHE_TTL_S = _env_float("STATS_CACHE_TTL_S", 3.0)
