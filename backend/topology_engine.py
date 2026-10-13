#!/usr/bin/env python3
"""
BSW Kota Bogor Multi-Camera Traffic Intelligence Dashboard
Milestone 1: Camera Geospatial & Road Network Topology Engine

Implements:
- F-GEO-01: 94-Camera Geocoding Master Database
- F-GEO-02: Directed Topology Transition Graph
- F-GEO-03: Spatiotemporal Arrival Window Model
- F-GEO-04: One-Way Directional Enforcer
- F-GEO-05: Intra-Junction Multi-Camera Clustering
- F-GEO-06: Adjacency-by-Exit Index Service
"""

from typing import Optional, List, Dict, Tuple, Any, Set
import os
import json
import math
import heapq


class TopologyEngine:
    """
    Topology Engine for BSW Kota Bogor Multi-Camera Tracking.
    
    Provides spatial routing, corridor indexing, one-way enforcement,
    dynamic arrival window calculations, and spatiotemporal transition validation.
    """

    # Forbidden reverse edges on one-way arterial corridors
    SSA_FORBIDDEN_REVERSE_EDGES: List[Tuple[int, int]] = [
        (87, 199),  # Otista Tengah -> Tugu Kujang
        (87, 79),   # Otista Tengah -> Tugu Kujang
        (83, 87),   # Pasar Bogor -> Otista Tengah
        (81, 83),   # Simpang BTM -> Pasar Bogor
        (85, 81),   # Juanda Museum -> Simpang BTM
        (133, 85),  # Juanda Korem -> Juanda Museum
        (134, 133), # Balaikota -> Juanda Korem
        (92, 134),  # Denpom -> Balaikota
        (120, 92),  # Jalak Harupat -> Denpom
        (122, 120), # Taman Sempur -> Jalak Harupat Barat
        (132, 122), # Pangrango -> Taman Sempur
        (79, 132),  # Tugu Kujang -> Pangrango
    ]

    SURYAKENCANA_FORBIDDEN_REVERSE_EDGES: List[Tuple[int, int]] = [
        (75, 86),   # Gang Aut -> Lawang Seketeng (Northbound forbidden)
        (86, 96),   # Lawang Seketeng -> Surken Gate (Northbound forbidden)
        (75, 96),   # Gang Aut -> Surken Gate
    ]

    def __init__(self, topology_path: Optional[str] = None):
        """
        Initializes the TopologyEngine.
        
        Args:
            topology_path: Optional path to bogor_camera_topology.json.
                           If None, resolves relative to data/bogor_camera_topology.json.
        """
        if topology_path is None:
            # Look in standard data directory locations
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            candidate = os.path.join(base_dir, "data", "bogor_camera_topology.json")
            if os.path.isfile(candidate):
                topology_path = candidate
            else:
                # Fallback to local cwd
                topology_path = os.path.join("data", "bogor_camera_topology.json")

        self.topology_path = topology_path
        self._load_topology(topology_path)

    def _load_topology(self, path: str) -> None:
        """Loads and indexes nodes, edges, adjacency, and corridors."""
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Topology file not found at: {path}")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.metadata = data.get("metadata", {})
        # Totals the file declares; validate_invariants checks the loaded graph against them.
        self.expected_cameras = int(self.metadata.get("total_cameras", len(data.get("nodes", []))))
        self.expected_edges = int(self.metadata.get("total_directed_edges", len(data.get("edges", []))))
        
        # Load and index camera nodes
        self.cameras: Dict[int, dict] = {}
        self.corridors: Dict[str, List[int]] = {}
        
        for raw_node in data.get("nodes", []):
            cid = int(raw_node["id"])
            node = dict(raw_node)
            node["id"] = cid
            # Guarantee both alias and canonical keys exist per contract
            node["lat"] = float(raw_node.get("lat", raw_node.get("latitude", 0.0)))
            node["lng"] = float(raw_node.get("lng", raw_node.get("longitude", 0.0)))
            node["latitude"] = node["lat"]
            node["longitude"] = node["lng"]
            node["corridor"] = str(raw_node.get("corridor", "UNKNOWN"))
            node["stream_url"] = str(raw_node.get("stream_url", ""))
            node["default_azimuth"] = float(raw_node.get("default_azimuth", raw_node.get("facing_bearing_deg", 0.0)))
            node["facing_bearing_deg"] = node["default_azimuth"]
            
            cam_type = str(raw_node.get("camera_type", "FIXED")).upper()
            node["camera_type"] = cam_type
            if "fov_deg" not in node:
                node["fov_deg"] = 360.0 if cam_type == "PTZ" else 75.0
            else:
                node["fov_deg"] = float(node["fov_deg"])
                
            self.cameras[cid] = node
            
            # Index by corridor
            c_name = node["corridor"].upper()
            if c_name not in self.corridors:
                self.corridors[c_name] = []
            self.corridors[c_name].append(cid)

        # Load and index edges
        self.edges: List[dict] = []
        self.outgoing_edges: Dict[int, List[dict]] = {cid: [] for cid in self.cameras}
        self.incoming_edges: Dict[int, List[dict]] = {cid: [] for cid in self.cameras}
        self.edge_by_pair: Dict[Tuple[int, int], List[dict]] = {}

        for raw_edge in data.get("edges", []):
            u = int(raw_edge["source_id"])
            v = int(raw_edge["target_id"])
            edge = dict(raw_edge)
            edge["source_id"] = u
            edge["target_id"] = v
            edge["exit_direction"] = str(raw_edge.get("exit_direction", "")).upper()
            edge["road_distance_m"] = float(raw_edge.get("road_distance_m", 0.0))
            edge["typical_speed_kmh"] = float(raw_edge.get("typical_speed_kmh", 25.0))
            
            window = raw_edge.get("travel_time_window_s", [5.0, 60.0])
            edge["travel_time_window_s"] = [float(window[0]), float(window[1])]
            edge["t_min_s"] = float(window[0])
            edge["t_max_s"] = float(window[1])

            self.edges.append(edge)
            if u in self.outgoing_edges:
                self.outgoing_edges[u].append(edge)
            if v in self.incoming_edges:
                self.incoming_edges[v].append(edge)

            pair = (u, v)
            if pair not in self.edge_by_pair:
                self.edge_by_pair[pair] = []
            self.edge_by_pair[pair].append(edge)

        # Load or construct adjacency_by_exit table
        self.adjacency_by_exit: Dict[str, List[int]] = {}
        raw_adj = data.get("adjacency_by_exit", {})
        if raw_adj:
            for k, targets in raw_adj.items():
                self.adjacency_by_exit[k] = [int(t) for t in targets]
        else:
            for edge in self.edges:
                key = f"{edge['source_id']}:{edge['exit_direction']}"
                if key not in self.adjacency_by_exit:
                    self.adjacency_by_exit[key] = []
                if edge["target_id"] not in self.adjacency_by_exit[key]:
                    self.adjacency_by_exit[key].append(edge["target_id"])

        # Intra-junction clusters map: cid -> set of co-located cids
        self._build_intra_junction_clusters()

    def _build_intra_junction_clusters(self) -> None:
        """Identifies and indexes intra-junction multi-camera clusters (Fixed + PTZ)."""
        self.intra_junction_clusters: Dict[int, Set[int]] = {cid: {cid} for cid in self.cameras}
        for edge in self.edges:
            # Internal links or very close proximity (< 40m with low travel time <= 15s)
            is_internal = edge["exit_direction"] == "INTERNAL"
            is_fast_transition = edge["t_min_s"] <= 5.0 and edge["t_max_s"] <= 20.0 and edge["road_distance_m"] <= 40.0
            if is_internal or is_fast_transition:
                u, v = edge["source_id"], edge["target_id"]
                cluster = self.intra_junction_clusters[u] | self.intra_junction_clusters[v]
                for member in cluster:
                    self.intra_junction_clusters[member] = cluster

    # =========================================================================
    # Contract Implementation: Camera Metadata & Inventory
    # =========================================================================

    def get_camera_info(self, camera_id: int) -> dict:
        """
        Returns camera node metadata: id, name, lat, lng, corridor, stream_url,
        default_azimuth, fov_deg.
        
        Raises KeyError if camera_id is not found.
        """
        cid = int(camera_id)
        if cid not in self.cameras:
            raise KeyError(f"Camera ID {camera_id} not found in topology database.")
        return dict(self.cameras[cid])

    def get_all_cameras(self) -> List[dict]:
        """Returns all 94 cameras in the database."""
        return [dict(c) for c in self.cameras.values()]

    def get_corridor_cameras(self, corridor_name: str) -> List[int]:
        """
        Returns list of camera IDs belonging to the given corridor.
        Case-insensitive matching.
        """
        norm = corridor_name.strip().upper()
        # Direct lookup
        if norm in self.corridors:
            return list(self.corridors[norm])
        # Substring lookup
        matches: List[int] = []
        for cname, cids in self.corridors.items():
            if norm in cname or cname in norm:
                matches.extend(cids)
        return list(dict.fromkeys(matches))

    def get_all_edges(self) -> List[dict]:
        """Returns all directed topology transition edges."""
        return [dict(e) for e in self.edges]

    def get_corridors(self) -> List[str]:
        """Returns list of unique corridor names."""
        return list(self.corridors.keys())

    # =========================================================================
    # Contract Implementation: Downstream Candidates & Routing
    # =========================================================================

    def get_downstream_candidates(self, source_camera_id: int, exit_direction: str) -> List[dict]:
        """
        Returns candidate downstream cameras:
        [
            {
                "target_camera_id": int,
                "target_name": str,
                "distance_m": float,
                "typical_speed_kmh": float,
                "t_min_s": float,
                "t_max_s": float,
                "corridor": str
            }, ...
        ]
        Raises KeyError if source_camera_id is invalid.
        Returns empty list if no downstream edge exists.
        """
        src_id = int(source_camera_id)
        if src_id not in self.cameras:
            raise KeyError(f"Source camera ID {source_camera_id} not found.")

        exit_dir_norm = exit_direction.strip().upper() if exit_direction else ""
        candidates: List[dict] = []
        seen_targets: Set[int] = set()

        outgoing = self.outgoing_edges.get(src_id, [])

        for edge in outgoing:
            target_id = edge["target_id"]
            edge_dir = edge["exit_direction"].upper()

            # Direction matching: exact match, wildcard, or internal fallback
            direction_matches = (
                exit_dir_norm == "*"
                or exit_dir_norm == "ANY"
                or edge_dir == exit_dir_norm
                or (exit_dir_norm == "" and edge_dir == "INTERNAL")
            )

            if direction_matches and target_id not in seen_targets:
                seen_targets.add(target_id)
                target_cam = self.cameras.get(target_id, {})
                candidates.append({
                    "target_camera_id": target_id,
                    "target_name": target_cam.get("name", f"Camera {target_id}"),
                    "distance_m": float(edge["road_distance_m"]),
                    "typical_speed_kmh": float(edge["typical_speed_kmh"]),
                    "t_min_s": float(edge["t_min_s"]),
                    "t_max_s": float(edge["t_max_s"]),
                    "corridor": target_cam.get("corridor", "UNKNOWN"),
                    "exit_direction": edge["exit_direction"],
                    "road_name": edge.get("road_name", "")
                })

        return candidates

    # =========================================================================
    # Contract Implementation: Arrival Window & Spatiotemporal Feasibility
    # =========================================================================

    def calculate_arrival_window(self, source_camera_id: int, target_camera_id: int) -> Tuple[float, float]:
        """
        Calculates the spatiotemporal travel time arrival window (t_min_s, t_max_s)
        between two cameras.
        
        If a direct directed edge exists, returns the direct travel time window.
        If no direct edge exists, computes shortest path arrival window via Dijkstra.
        If completely unreachable, returns (inf, inf).
        """
        src_id = int(source_camera_id)
        tgt_id = int(target_camera_id)

        if src_id not in self.cameras or tgt_id not in self.cameras:
            raise KeyError(f"Invalid camera IDs: source={source_camera_id}, target={target_camera_id}")

        # Identical camera
        if src_id == tgt_id:
            return (0.0, 5.0)

        # Check direct edge(s)
        pair = (src_id, tgt_id)
        if pair in self.edge_by_pair and self.edge_by_pair[pair]:
            edges = self.edge_by_pair[pair]
            t_min = min(e["t_min_s"] for e in edges)
            t_max = max(e["t_max_s"] for e in edges)
            return (float(t_min), float(t_max))

        # Check multi-hop directed shortest path
        path_info = self.find_shortest_path(src_id, tgt_id)
        if path_info is not None:
            _, path_t_min, path_t_max = path_info
            return (float(path_t_min), float(path_t_max))

        return (float("inf"), float("inf"))

    def is_valid_transition(
        self,
        source_camera_id: int,
        target_camera_id: int,
        elapsed_time_s: float,
        direct_only: bool = False
    ) -> Tuple[bool, float]:
        """
        Validates physical transition between cameras given elapsed time in seconds.
        Returns (is_reachable, spatiotemporal_score).
        spatiotemporal_score is in [0.0, 1.0].
        
        Enforces one-way flow invariants and realistic traffic physics:
        - If elapsed_time_s < 0: returns (False, 0.0)
        - If disconnected / opposite one-way: returns (False, 0.0)
        - Within [t_min, t_max]: score = 1.0 (is_reachable = True)
        - Outside [t_min, t_max]: Gaussian decay score based on delta
        - If physically unfeasible (score < 0.10 for early, < 0.05 for late): returns (False, 0.0)
        """
        src_id = int(source_camera_id)
        tgt_id = int(target_camera_id)
        t = float(elapsed_time_s)

        if src_id not in self.cameras or tgt_id not in self.cameras:
            return (False, 0.0)

        if t < 0.0:
            return (False, 0.0)

        # Same camera re-detection
        if src_id == tgt_id:
            if t <= 30.0:
                return (True, 1.0)
            # Gradual decay for vehicle lingering or parking at same camera
            score = math.exp(-0.5 * ((t - 30.0) / 60.0) ** 2)
            is_valid = score >= 0.05
            return (is_valid, round(score if is_valid else 0.0, 4))

        # Check direct edge lookup
        pair = (src_id, tgt_id)
        direct_edges = self.edge_by_pair.get(pair, [])

        if direct_edges:
            t_min = min(e["t_min_s"] for e in direct_edges)
            t_max = max(e["t_max_s"] for e in direct_edges)
        elif direct_only:
            # Direct transition requested but no direct edge exists
            return (False, 0.0)
        else:
            # Multi-hop shortest path
            path_info = self.find_shortest_path(src_id, tgt_id)
            if path_info is None:
                # Disconnected in directed graph (e.g. against one-way flow)
                return (False, 0.0)
            _, t_min, t_max = path_info

        return self.score_transition_window(t_min, t_max, t)

    @staticmethod
    def score_transition_window(t_min: float, t_max: float, t: float) -> Tuple[bool, float]:
        """
        Scores elapsed time ``t`` against a physical arrival window [t_min, t_max].
        Returns (is_valid, score in [0, 1]); Gaussian decay outside the window.
        """
        # Physical arrival window evaluation
        if t_min <= t <= t_max:
            return (True, 1.0)

        if t < t_min:
            # Arrived earlier than physically feasible (speeding or teleportation)
            delta = t_min - t
            sigma = max(0.25 * t_min, 2.0)
            score = math.exp(-0.5 * (delta / sigma) ** 2)
            is_valid = score >= 0.10
            return (is_valid, round(score if is_valid else 0.0, 4))
        else:
            # Arrived later than expected (traffic delay, red light congestion)
            delta = t - t_max
            sigma = max(0.50 * (t_max - t_min), 30.0)
            score = math.exp(-0.5 * (delta / sigma) ** 2)
            is_valid = score >= 0.05
            return (is_valid, round(score if is_valid else 0.0, 4))

    # =========================================================================
    # Intra-Junction Clustering & Pathfinding
    # =========================================================================

    def is_intra_junction(self, camera_id_1: int, camera_id_2: int) -> bool:
        """Returns True if both cameras belong to the same junction cluster."""
        cid1, cid2 = int(camera_id_1), int(camera_id_2)
        if cid1 not in self.cameras or cid2 not in self.cameras:
            return False
        return cid2 in self.intra_junction_clusters.get(cid1, set())

    def get_intra_junction_cluster(self, camera_id: int) -> List[int]:
        """Returns all camera IDs co-located at the same junction as camera_id."""
        cid = int(camera_id)
        return sorted(list(self.intra_junction_clusters.get(cid, {cid})))

    def find_shortest_path(self, source_id: int, target_id: int) -> Optional[Tuple[List[int], float, float]]:
        """
        Dijkstra shortest path algorithm on directed graph.
        Returns: (path_node_ids, total_t_min, total_t_max) or None if unreachable.
        """
        src = int(source_id)
        tgt = int(target_id)
        if src not in self.cameras or tgt not in self.cameras:
            return None

        # Priority queue stores: (cumulative_t_min, current_node, cumulative_t_max, path)
        pq: List[Tuple[float, int, float, List[int]]] = [(0.0, src, 0.0, [src])]
        visited: Dict[int, float] = {}

        while pq:
            curr_t_min, u, curr_t_max, path = heapq.heappop(pq)

            if u in visited and visited[u] <= curr_t_min:
                continue
            visited[u] = curr_t_min

            if u == tgt:
                return (path, curr_t_min, curr_t_max)

            for edge in self.outgoing_edges.get(u, []):
                v = edge["target_id"]
                next_t_min = curr_t_min + edge["t_min_s"]
                next_t_max = curr_t_max + edge["t_max_s"]
                if v not in visited or next_t_min < visited[v]:
                    heapq.heappush(pq, (next_t_min, v, next_t_max, path + [v]))

        return None

    def haversine_distance_m(self, camera_id_1: int, camera_id_2: int) -> float:
        """Computes direct great-circle Haversine distance in meters between two cameras."""
        c1 = self.get_camera_info(camera_id_1)
        c2 = self.get_camera_info(camera_id_2)
        
        r = 6371000.0  # Earth radius in meters
        phi1, phi2 = math.radians(c1["lat"]), math.radians(c2["lat"])
        dphi = math.radians(c2["lat"] - c1["lat"])
        dlambda = math.radians(c2["lng"] - c1["lng"])
        
        a = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
        return 2.0 * r * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))

    # =========================================================================
    # Invariant Verification & Health Audit
    # =========================================================================

    def validate_invariants(self) -> Dict[str, Any]:
        """
        Performs exhaustive verification of all Milestone 1 topological invariants:
        1. Every camera node (count recorded in the topology metadata) with WGS84 coordinates and a stream URL.
        2. Every directed edge (count recorded in the metadata) with positive distances and travel time windows.
        3. Zero dead ends (every camera has outgoing transitions).
        4. Zero unreachable nodes (every camera has incoming transitions).
        5. 100% reachability from Tugu Kujang (ID 79).
        6. Zero one-way violations on SSA Ring Road (counter-clockwise) and Suryakencana (southbound).
        
        Returns a dictionary of audit metrics and status.
        """
        node_ids = set(self.cameras.keys())
        sources = set(self.outgoing_edges.keys())
        targets = set(self.incoming_edges.keys())

        # Check dead-ends and unreachable nodes
        dead_ends = [cid for cid, out_edges in self.outgoing_edges.items() if len(out_edges) == 0]
        unreachable = [cid for cid, in_edges in self.incoming_edges.items() if len(in_edges) == 0]

        # BFS Reachability from Tugu Kujang (79)
        visited: Set[int] = set()
        queue = [79] if 79 in self.cameras else []
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                for edge in self.outgoing_edges.get(curr, []):
                    nxt = edge["target_id"]
                    if nxt not in visited:
                        queue.append(nxt)

        # One-way invariant violation detection
        forbidden = set(self.SSA_FORBIDDEN_REVERSE_EDGES + self.SURYAKENCANA_FORBIDDEN_REVERSE_EDGES)
        violations = []
        for edge in self.edges:
            pair = (edge["source_id"], edge["target_id"])
            if pair in forbidden:
                violations.append(pair)

        # Edge validity checks
        invalid_edges = []
        for edge in self.edges:
            if edge["road_distance_m"] <= 0.0 or edge["t_min_s"] <= 0.0 or edge["t_max_s"] < edge["t_min_s"]:
                invalid_edges.append(edge)

        # Coordinate checks
        invalid_coords = []
        for cid, cam in self.cameras.items():
            if not (-6.75 <= cam["lat"] <= -6.45 and 106.65 <= cam["lng"] <= 106.90):
                invalid_coords.append(cid)
            if not cam["stream_url"] or not cam["stream_url"].startswith("http"):
                invalid_coords.append(cid)

        is_valid = (
            len(self.cameras) == self.expected_cameras
            and len(self.edges) == self.expected_edges
            and len(dead_ends) == 0
            and len(unreachable) == 0
            and len(visited) == len(self.cameras)
            and len(violations) == 0
            and len(invalid_edges) == 0
            and len(invalid_coords) == 0
        )

        return {
            "valid": is_valid,
            "total_cameras": len(self.cameras),
            "total_edges": len(self.edges),
            "dead_end_nodes": dead_ends,
            "unreachable_nodes": unreachable,
            "kujang_reachable_count": len(visited),
            "one_way_violations": violations,
            "invalid_edge_count": len(invalid_edges),
            "invalid_coord_count": len(invalid_coords)
        }


# =============================================================================
# Module-level convenience functions (PROJECT.md contract direct imports)
# =============================================================================

_default_engine: Optional[TopologyEngine] = None


def get_default_engine() -> TopologyEngine:
    """Returns singleton TopologyEngine instance."""
    global _default_engine
    if _default_engine is None:
        _default_engine = TopologyEngine()
    return _default_engine


def get_camera_info(camera_id: int) -> dict:
    """Returns camera node metadata: id, name, lat, lng, corridor, stream_url, default_azimuth, fov_deg."""
    return get_default_engine().get_camera_info(camera_id)


def get_all_cameras() -> List[dict]:
    """Returns all 94 cameras in the database."""
    return get_default_engine().get_all_cameras()


def get_corridor_cameras(corridor_name: str) -> List[int]:
    """Returns camera IDs belonging to the given corridor."""
    return get_default_engine().get_corridor_cameras(corridor_name)


def get_downstream_candidates(source_camera_id: int, exit_direction: str) -> List[dict]:
    """Returns candidate downstream cameras for exit_direction."""
    return get_default_engine().get_downstream_candidates(source_camera_id, exit_direction)


def is_valid_transition(source_camera_id: int, target_camera_id: int, elapsed_time_s: float) -> Tuple[bool, float]:
    """Validates physical transition between cameras given elapsed time."""
    return get_default_engine().is_valid_transition(source_camera_id, target_camera_id, elapsed_time_s)


def calculate_arrival_window(source_camera_id: int, target_camera_id: int) -> Tuple[float, float]:
    """Calculates arrival window (t_min_s, t_max_s) between two cameras."""
    return get_default_engine().calculate_arrival_window(source_camera_id, target_camera_id)

