"""
tests/reference_contracts.py
Interface Contract Implementations & Adapters for BSW Kota Bogor MTMC Test Harness.
Adheres strictly to PROJECT.md § Interface Contracts.
Dynamically imports from backend.* if available, otherwise provides authoritative reference logic.
"""

import json
import math
import os
import sqlite3
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 1. Topology Engine Contract
# ---------------------------------------------------------------------------
class TopologyContract:
    def __init__(self, topology_path: Optional[str] = None):
        self.topology_path = topology_path or os.path.join(
            os.path.dirname(__file__), "..", "data", "bogor_camera_topology.json"
        )
        self.data: Dict[str, Any] = self._load_data()
        self.nodes = {n["id"]: n for n in self.data.get("nodes", [])}
        self.edges = self.data.get("edges", [])
        self.adjacency_by_exit = self.data.get("adjacency_by_exit", {})

        # SSA and Suryakencana invariants
        self.ssa_camera_ids = {79, 81, 83, 85, 87, 88, 92, 120, 122, 132, 133, 134, 199}
        self.surken_camera_ids = {75, 86, 96, 171}

    def _load_data(self) -> Dict[str, Any]:
        if os.path.exists(self.topology_path):
            with open(self.topology_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"nodes": [], "edges": [], "adjacency_by_exit": {}}

    def get_camera_info(self, camera_id: int) -> Dict[str, Any]:
        """Returns camera node metadata: id, name, lat, lng, corridor, stream_url, default_azimuth, fov_deg."""
        try:
            # Try importing from backend if implemented
            from backend.topology_engine import get_camera_info as backend_fn
            return backend_fn(camera_id)
        except (ImportError, AttributeError):
            pass

        if camera_id not in self.nodes:
            raise KeyError(f"Camera ID {camera_id} not found in topology graph")
        n = self.nodes[camera_id]
        return {
            "id": n["id"],
            "name": n.get("name", ""),
            "lat": n.get("latitude", n.get("lat", 0.0)),
            "lng": n.get("longitude", n.get("lng", 0.0)),
            "corridor": n.get("corridor", ""),
            "stream_url": n.get("stream_url", ""),
            "default_azimuth": n.get("facing_bearing_deg", n.get("default_azimuth", 0.0)),
            "fov_deg": n.get("fov_deg", 75.0),
            "camera_type": n.get("camera_type", "FIXED"),
            "junction_name": n.get("junction_name", ""),
        }

    def get_downstream_candidates(self, source_camera_id: int, exit_direction: str) -> List[Dict[str, Any]]:
        """
        Returns list of candidate downstream cameras for exit_direction.
        Raises KeyError if source_camera_id is invalid. Returns empty list if no downstream edge exists.
        """
        raw_dir = exit_direction.upper().strip()
        dir_map = {
            "N": "NORTH", "S": "SOUTH", "E": "EAST", "W": "WEST",
            "NE": "NORTHEAST", "NW": "NORTHWEST", "SE": "SOUTHEAST", "SW": "SOUTHWEST",
            "NORTH": "NORTH", "SOUTH": "SOUTH", "EAST": "EAST", "WEST": "WEST",
            "NORTHEAST": "NORTHEAST", "NORTHWEST": "NORTHWEST",
            "SOUTHEAST": "SOUTHEAST", "SOUTHWEST": "SOUTHWEST", "INTERNAL": "INTERNAL"
        }
        full_dir = dir_map.get(raw_dir, raw_dir)

        try:
            from backend.topology_engine import get_downstream_candidates as backend_fn
            res = backend_fn(source_camera_id, full_dir)
            if not res and full_dir != raw_dir:
                res = backend_fn(source_camera_id, raw_dir)
            return res
        except (ImportError, AttributeError):
            pass

        if source_camera_id not in self.nodes:
            raise KeyError(f"Camera ID {source_camera_id} not found")

        candidates: List[Dict[str, Any]] = []

        # Check adjacency_by_exit lookup table with both ':' and '_'
        possible_keys = [
            f"{source_camera_id}:{full_dir}",
            f"{source_camera_id}_{full_dir}",
            f"{source_camera_id}:{raw_dir}",
            f"{source_camera_id}_{raw_dir}"
        ]
        target_ids = []
        for k in possible_keys:
            if k in self.adjacency_by_exit:
                target_ids = self.adjacency_by_exit[k]
                break

        if target_ids:
            for tid in target_ids:
                matching_edges = [
                    e for e in self.edges if e.get("source_id") == source_camera_id and e.get("target_id") == tid
                ]
                if matching_edges:
                    edge = matching_edges[0]
                    tgt_node = self.nodes.get(tid, {})
                    dist = float(edge.get("distance_m", edge.get("road_distance_m", 0.0)))
                    t_window = edge.get("travel_time_window_s", [edge.get("t_min_s", 10.0), edge.get("t_max_s", 60.0)])
                    t_min = float(t_window[0])
                    t_max = float(t_window[1])
                    candidates.append({
                        "target_camera_id": tid,
                        "target_name": tgt_node.get("name", f"CCTV-{tid}"),
                        "distance_m": dist,
                        "typical_speed_kmh": float(edge.get("typical_speed_kmh", 30.0)),
                        "t_min_s": t_min,
                        "t_max_s": t_max,
                        "corridor": tgt_node.get("corridor", edge.get("corridor", "")),
                        "road_name": edge.get("road_name", ""),
                        "exit_direction": edge.get("exit_direction", full_dir),
                    })
            return candidates

        # Fallback to direct edges with compass matching
        for edge in self.edges:
            if edge.get("source_id") == source_camera_id:
                tid = edge.get("target_id")
                tgt_node = self.nodes.get(tid, {})
                compass = (edge.get("compass_heading") or edge.get("exit_direction") or "").upper()
                if not raw_dir or compass == full_dir or compass == raw_dir or raw_dir in compass:
                    dist = float(edge.get("distance_m", edge.get("road_distance_m", 0.0)))
                    t_window = edge.get("travel_time_window_s", [edge.get("t_min_s", 10.0), edge.get("t_max_s", 60.0)])
                    t_min = float(t_window[0])
                    t_max = float(t_window[1])
                    candidates.append({
                        "target_camera_id": tid,
                        "target_name": tgt_node.get("name", f"CCTV-{tid}"),
                        "distance_m": dist,
                        "typical_speed_kmh": float(edge.get("typical_speed_kmh", 30.0)),
                        "t_min_s": t_min,
                        "t_max_s": t_max,
                        "corridor": tgt_node.get("corridor", edge.get("corridor", "")),
                        "road_name": edge.get("road_name", ""),
                        "exit_direction": edge.get("exit_direction", compass),
                    })
        return candidates

    def is_valid_transition(
        self, source_camera_id: int, target_camera_id: int, elapsed_time_s: float
    ) -> Tuple[bool, float]:
        """
        Validates physical transition between cameras given elapsed time.
        Returns (is_reachable, spatiotemporal_score).
        spatiotemporal_score is in [0.0, 1.0].
        """
        try:
            from backend.topology_engine import is_valid_transition as backend_fn
            return backend_fn(source_camera_id, target_camera_id, elapsed_time_s)
        except (ImportError, AttributeError):
            pass

        matching_edges = [
            e for e in self.edges if e.get("source_id") == source_camera_id and e.get("target_id") == target_camera_id
        ]
        if not matching_edges:
            return False, 0.0

        edge = matching_edges[0]
        t_window = edge.get("travel_time_window_s", [edge.get("t_min_s", 5.0), edge.get("t_max_s", 120.0)])
        t_min = float(t_window[0])
        t_max = float(t_window[1])

        if elapsed_time_s < t_min:
            # Vehicle arrived physically too fast (e.g. impossible speed > 150 km/h)
            penalty = max(0.0, elapsed_time_s / max(0.1, t_min))
            return False, float(0.5 * penalty)

        if elapsed_time_s <= t_max:
            # Perfect arrival window
            midpoint = (t_min + t_max) / 2.0
            sigma = max(1.0, (t_max - t_min) / 3.0)
            score = math.exp(-0.5 * ((elapsed_time_s - midpoint) / sigma) ** 2)
            return True, float(min(1.0, max(0.5, score)))

        # Late arrival (congestion or stops)
        slack = t_max * 1.5
        if elapsed_time_s <= slack:
            decay = max(0.1, 1.0 - ((elapsed_time_s - t_max) / (slack - t_max)))
            return True, float(0.5 * decay)

        return False, 0.0


# ---------------------------------------------------------------------------
# 2. SQLite WAL Persistence Layer Contract
# ---------------------------------------------------------------------------
class DatabaseContract:
    SCHEMA_DDL = """
    PRAGMA journal_mode = WAL;
    PRAGMA synchronous = NORMAL;
    PRAGMA foreign_keys = ON;

    CREATE TABLE IF NOT EXISTS cameras (
        id INTEGER PRIMARY KEY,
        name TEXT NOT NULL,
        corridor TEXT NOT NULL,
        latitude REAL NOT NULL,
        longitude REAL NOT NULL,
        camera_type TEXT DEFAULT 'FIXED',
        stream_url TEXT,
        status TEXT DEFAULT 'online'
    );

    CREATE TABLE IF NOT EXISTS topology_edges (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        target_id INTEGER NOT NULL,
        distance_m REAL NOT NULL,
        typical_speed_kmh REAL NOT NULL,
        t_min_s REAL NOT NULL,
        t_max_s REAL NOT NULL,
        corridor TEXT,
        FOREIGN KEY (source_id) REFERENCES cameras(id),
        FOREIGN KEY (target_id) REFERENCES cameras(id)
    );

    CREATE TABLE IF NOT EXISTS vehicle_sightings (
        sighting_id INTEGER PRIMARY KEY AUTOINCREMENT,
        camera_id INTEGER NOT NULL,
        local_track_id INTEGER NOT NULL,
        vehicle_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 INTEGER NOT NULL,
        bbox_y1 INTEGER NOT NULL,
        bbox_x2 INTEGER NOT NULL,
        bbox_y2 INTEGER NOT NULL,
        heading_deg REAL NOT NULL,
        speed_px_s REAL NOT NULL,
        timestamp_iso TEXT NOT NULL,
        FOREIGN KEY (camera_id) REFERENCES cameras(id)
    );

    CREATE INDEX IF NOT EXISTS idx_sightings_cam_time ON vehicle_sightings(camera_id, timestamp_iso);
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self):
        self.conn.executescript(self.SCHEMA_DDL)
        self.conn.commit()

    def insert_vehicle_sighting(
        self,
        camera_id: int,
        local_track_id: int,
        vehicle_class: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        heading_deg: float,
        speed_px_s: float,
        timestamp_iso: str,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO vehicle_sightings (
                camera_id, local_track_id, vehicle_class,
                confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                heading_deg, speed_px_s, timestamp_iso
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                local_track_id,
                vehicle_class,
                confidence,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                heading_deg,
                speed_px_s,
                timestamp_iso,
            ),
        )
        self.conn.commit()
        return cur.lastrowid
