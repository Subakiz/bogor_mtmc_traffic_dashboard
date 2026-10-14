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

from backend.camera_sources import load_extra_nodes


class TopologyEngine:
    """
    Topology Engine for BSW Kota Bogor Multi-Camera Tracking.
    
    Provides the camera registry, corridor indexing, one-way enforcement and
    intra-junction clustering.
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
        
        # Extra cities (data/sources/*.json): nodes only, no road edges; ids are namespaced.
        for raw_node in data.get("nodes", []) + load_extra_nodes():
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
        # Edgeless sources name their junctions: cameras sharing one watch the same vehicles.
        by_junction: Dict[str, Set[int]] = {}
        for cid, cam in self.cameras.items():
            if cam.get("source"):
                by_junction.setdefault(cam.get("junction_name") or str(cid), set()).add(cid)
        for cluster in by_junction.values():
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
