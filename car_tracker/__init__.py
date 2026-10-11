"""
CCTV vehicle tracker on the Apple Neural Engine (ANE).

Provides the CoreML vehicle detector with independent per-camera tracking, shared
drawing helpers, and the per-camera traffic analytics engine. The multi-camera
dashboard in ``backend/`` builds its rolling tracker on top of this.
"""

from .config import (
    CLASS_NAMES,
    CLASS_COLORS,
    VEHICLE_CLASS_IDS,
    DETECTION_CONF,
    _GLOBAL_ANE_LOCK,
)
from .tracking import CameraTrackerPool, VehicleDetector, rows_to_detections
from .pipeline import count_by_class, default_label, draw_detections, update_trails
from .hud import draw_hud
from .streams import load_cctv_streams
from .analytics import (
    TrafficAnalyticsEngine,
    TrafficAnalyticsDatabase,
    TimeWindowMetrics,
    TrackKinematics,
    DEFAULT_DB_PATH,
)
from .cli import track_stream, main
