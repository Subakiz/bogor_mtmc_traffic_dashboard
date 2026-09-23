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
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


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
        global_vehicle_id INTEGER,
        vehicle_class TEXT NOT NULL,
        confidence REAL NOT NULL,
        bbox_x1 INTEGER NOT NULL,
        bbox_y1 INTEGER NOT NULL,
        bbox_x2 INTEGER NOT NULL,
        bbox_y2 INTEGER NOT NULL,
        heading_deg REAL NOT NULL,
        speed_px_s REAL NOT NULL,
        embedding BLOB NOT NULL,
        color_hist BLOB NOT NULL,
        timestamp_iso TEXT NOT NULL,
        crop_thumbnail BLOB,
        FOREIGN KEY (camera_id) REFERENCES cameras(id)
    );

    CREATE TABLE IF NOT EXISTS handoff_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        global_vehicle_id INTEGER NOT NULL,
        from_camera_id INTEGER NOT NULL,
        to_camera_id INTEGER NOT NULL,
        confidence REAL NOT NULL,
        timestamp_iso TEXT NOT NULL,
        FOREIGN KEY (from_camera_id) REFERENCES cameras(id),
        FOREIGN KEY (to_camera_id) REFERENCES cameras(id)
    );

    CREATE TABLE IF NOT EXISTS active_targets (
        global_vehicle_id INTEGER PRIMARY KEY,
        current_camera_id INTEGER NOT NULL,
        last_sighting_id INTEGER NOT NULL,
        locked_at_iso TEXT NOT NULL,
        status TEXT DEFAULT 'tracking',
        FOREIGN KEY (current_camera_id) REFERENCES cameras(id),
        FOREIGN KEY (last_sighting_id) REFERENCES vehicle_sightings(sighting_id)
    );

    CREATE INDEX IF NOT EXISTS idx_sightings_cam_time ON vehicle_sightings(camera_id, timestamp_iso);
    CREATE INDEX IF NOT EXISTS idx_sightings_global_id ON vehicle_sightings(global_vehicle_id);
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
        global_vehicle_id: Optional[int],
        vehicle_class: str,
        confidence: float,
        bbox: Tuple[int, int, int, int],
        heading_deg: float,
        speed_px_s: float,
        embedding: bytes,
        color_hist: bytes,
        timestamp_iso: str,
        crop_thumbnail: Optional[bytes] = None,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO vehicle_sightings (
                camera_id, local_track_id, global_vehicle_id, vehicle_class,
                confidence, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                heading_deg, speed_px_s, embedding, color_hist, timestamp_iso, crop_thumbnail
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                camera_id,
                local_track_id,
                global_vehicle_id,
                vehicle_class,
                confidence,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                heading_deg,
                speed_px_s,
                embedding,
                color_hist,
                timestamp_iso,
                crop_thumbnail,
            ),
        )
        self.conn.commit()
        return cur.lastrowid

    def record_handoff_event(
        self,
        global_vehicle_id: int,
        from_camera_id: int,
        to_camera_id: int,
        confidence: float,
        timestamp_iso: str,
    ) -> int:
        cur = self.conn.cursor()
        cur.execute(
            """
            INSERT INTO handoff_events (
                global_vehicle_id, from_camera_id, to_camera_id, confidence, timestamp_iso
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (global_vehicle_id, from_camera_id, to_camera_id, confidence, timestamp_iso),
        )
        self.conn.commit()
        return cur.lastrowid

    def get_vehicle_journey(self, global_vehicle_id: int) -> List[Dict[str, Any]]:
        cur = self.conn.cursor()
        cur.execute(
            """
            SELECT s.*, c.name as camera_name, c.latitude, c.longitude, c.corridor
            FROM vehicle_sightings s
            JOIN cameras c ON s.camera_id = c.id
            WHERE s.global_vehicle_id = ?
            ORDER BY s.timestamp_iso ASC
            """,
            (global_vehicle_id,),
        )
        return [dict(row) for row in cur.fetchall()]


# ---------------------------------------------------------------------------
# 3. Re-ID Engine & Cosine Similarity Contract
# ---------------------------------------------------------------------------
class ReidEngineContract:
    @staticmethod
    def compute_appearance_similarity(
        query_emb: np.ndarray,
        query_hist: np.ndarray,
        cand_embs: np.ndarray,
        cand_hists: np.ndarray,
    ) -> np.ndarray:
        """
        Computes fused appearance similarity scores:
        S_appear = 0.60 * cos_sim(query_emb, cand_embs) + 0.40 * cos_sim(query_hist, cand_hists)
        Returns 1D array of float32 in [-1.0, 1.0].
        """
        try:
            from backend.reid_engine import compute_appearance_similarity as backend_fn
            return backend_fn(query_emb, query_hist, cand_embs, cand_hists)
        except (ImportError, AttributeError):
            pass

        if cand_embs.ndim == 1:
            cand_embs = cand_embs.reshape(1, -1)
        if cand_hists.ndim == 1:
            cand_hists = cand_hists.reshape(1, -1)

        # Normalize query vectors
        q_emb_norm = query_emb / max(1e-8, float(np.linalg.norm(query_emb)))
        q_hist_norm = query_hist / max(1e-8, float(np.linalg.norm(query_hist)))

        # Normalize candidate vectors
        cand_embs_norms = np.linalg.norm(cand_embs, axis=1, keepdims=True)
        cand_embs_norms[cand_embs_norms < 1e-8] = 1.0
        norm_cand_embs = cand_embs / cand_embs_norms

        cand_hists_norms = np.linalg.norm(cand_hists, axis=1, keepdims=True)
        cand_hists_norms[cand_hists_norms < 1e-8] = 1.0
        norm_cand_hists = cand_hists / cand_hists_norms

        # Vector dot products
        sim_deep = np.dot(norm_cand_embs, q_emb_norm)
        sim_color = np.dot(norm_cand_hists, q_hist_norm)

        s_appear = 0.60 * sim_deep + 0.40 * sim_color
        return np.clip(s_appear, -1.0, 1.0).astype(np.float32)

    @staticmethod
    def match_candidate_vehicle(
        target_vehicle_signature: Dict[str, Any],
        candidate_sightings: List[Dict[str, Any]],
        current_time: float,
        topology_contract: Optional[TopologyContract] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Evaluates candidate sightings using fused score:
        S_fused = 0.70 * S_appear + 0.30 * S_st
        Requires vehicle_class match, S_appear >= 0.65, S_fused >= 0.72.
        Returns best matching sighting dict or None.
        """
        try:
            from backend.reid_engine import match_candidate_vehicle as backend_fn
            return backend_fn(target_vehicle_signature, candidate_sightings, current_time)
        except (ImportError, AttributeError):
            pass

        if not candidate_sightings:
            return None

        target_class = target_vehicle_signature.get("vehicle_class")
        target_emb = np.frombuffer(target_vehicle_signature["embedding"], dtype=np.float32)
        target_hist = np.frombuffer(target_vehicle_signature["color_hist"], dtype=np.float32)
        source_cam = target_vehicle_signature.get("camera_id")
        target_time = target_vehicle_signature.get("timestamp", current_time - 30.0)

        best_match = None
        best_score = -1.0

        for cand in candidate_sightings:
            # Class Gating
            if target_class and cand.get("vehicle_class") != target_class:
                continue

            cand_emb = np.frombuffer(cand["embedding"], dtype=np.float32)
            cand_hist = np.frombuffer(cand["color_hist"], dtype=np.float32)

            # Appearance Similarity
            s_appear_arr = ReidEngineContract.compute_appearance_similarity(
                target_emb, target_hist, cand_emb, cand_hist
            )
            s_appear = float(s_appear_arr[0])

            if s_appear < 0.65:
                continue

            # Spatiotemporal Feasibility
            cand_cam = cand.get("camera_id")
            cand_time = cand.get("timestamp", current_time)
            elapsed_time = cand_time - target_time

            s_st = 0.5
            if topology_contract and source_cam and cand_cam:
                is_valid, st_score = topology_contract.is_valid_transition(
                    source_cam, cand_cam, elapsed_time
                )
                if not is_valid:
                    continue
                s_st = st_score

            s_fused = 0.70 * s_appear + 0.30 * s_st
            if s_fused >= 0.72 and s_fused > best_score:
                best_score = s_fused
                best_match = cand.copy()
                best_match["fused_score"] = float(s_fused)
                best_match["appearance_score"] = float(s_appear)
                best_match["spatiotemporal_score"] = float(s_st)

        return best_match


# ---------------------------------------------------------------------------
# 4. Handoff Engine Contract
# ---------------------------------------------------------------------------
class HandoffEngineContract:
    @staticmethod
    def estimate_exit_heading(
        motion_vector: Tuple[float, float], camera_azimuth_deg: float
    ) -> Tuple[float, str]:
        """
        Estimates real-world road exit heading (Theta_exit) and cardinal direction.
        motion_vector is (delta_x, delta_y) in image space.
        """
        try:
            from backend.handoff_engine import estimate_exit_heading as backend_fn
            return backend_fn(motion_vector, camera_azimuth_deg)
        except (ImportError, AttributeError):
            pass

        dx, dy = motion_vector
        if abs(dx) < 1e-4 and abs(dy) < 1e-4:
            heading = camera_azimuth_deg % 360.0
        else:
            # Angle in standard screen coordinates (y down)
            angle_rad = math.atan2(dy, dx)
            # Offset by camera mounting azimuth
            heading = (math.degrees(angle_rad) + camera_azimuth_deg) % 360.0

        # Map heading to cardinal sector (8 directions)
        sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
        idx = int(((heading + 22.5) % 360.0) // 45.0)
        cardinal = sectors[idx]
        return float(heading), cardinal

    @staticmethod
    def is_boundary_exit(
        bbox: Tuple[int, int, int, int], frame_size: Tuple[int, int] = (640, 480), margin: int = 35
    ) -> bool:
        """
        Detects if vehicle bounding box crosses the camera frame exit margin.
        """
        x1, y1, x2, y2 = bbox
        fw, fh = frame_size
        return (x1 <= margin) or (y1 <= margin) or (x2 >= fw - margin) or (y2 >= fh - margin)


# ---------------------------------------------------------------------------
# 5. Global ID Preserver Contract
# ---------------------------------------------------------------------------
class GlobalIDPreserverContract:
    def __init__(self):
        self.active_targets: Dict[int, Dict[str, Any]] = {}
        self.handoff_history: List[Dict[str, Any]] = []

    def lock_target(self, global_id: int, camera_id: int, signature: Dict[str, Any]):
        self.active_targets[global_id] = {
            "global_vehicle_id": global_id,
            "current_camera_id": camera_id,
            "signature": signature,
            "locked_at": time.time(),
            "status": "tracking",
            "history": [camera_id],
        }

    def handoff_target(self, global_id: int, to_camera_id: int, confidence: float) -> Dict[str, Any]:
        if global_id not in self.active_targets:
            raise KeyError(f"Target {global_id} not currently locked")
        t = self.active_targets[global_id]
        from_cam = t["current_camera_id"]
        t["current_camera_id"] = to_camera_id
        t["history"].append(to_camera_id)

        event = {
            "type": "CAMERA_SWITCH",
            "global_vehicle_id": global_id,
            "from_camera_id": from_cam,
            "to_camera_id": to_camera_id,
            "confidence": confidence,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.handoff_history.append(event)
        return event

    def unlock_target(self, global_id: int):
        self.active_targets.pop(global_id, None)
