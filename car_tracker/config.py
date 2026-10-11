import threading

# COCO vehicle classes tracked by the system.
VEHICLE_CLASS_IDS = [2, 3, 5, 7]
CLASS_NAMES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
CLASS_COLORS = {
    "car":        (0, 255, 0),
    "motorcycle": (255, 170, 0),
    "truck":      (0, 165, 255),
    "bus":        (255, 0, 230),
}

# Minimum detector confidence for a vehicle to be tracked.
DETECTION_CONF = 0.25

# CoreML models are not safe for simultaneous forward passes on macOS:
# every inference call must hold this lock.
_GLOBAL_ANE_LOCK = threading.Lock()
