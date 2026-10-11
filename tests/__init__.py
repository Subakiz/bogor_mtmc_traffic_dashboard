"""
CCTV YOLO ANE + BSW Kota Bogor MTMC dashboard test suite (car_tracker engine and backend/).
"""

import json
import os

# Camera/edge totals come from the topology file so the suite follows the real camera list
# (scrape_cctv_streams.py keeps it in sync with bsw.kotabogor.go.id) instead of a fixed 94/220.
_TOPOLOGY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "data", "bogor_camera_topology.json")
with open(_TOPOLOGY, "r", encoding="utf-8") as _f:
    _topology = json.load(_f)
TOTAL_CAMERAS = len(_topology["nodes"])
TOTAL_EDGES = len(_topology["edges"])
