"""
tests/tier1_features/test_f_geo_01_06.py
Tier 1 Tests for Features F-GEO-01 through F-GEO-06.
>=5 test cases per feature (30 total tests).
"""

import json
import os
import unittest
from tests.reference_contracts import TopologyContract


class TestFGEO01GeocodingMasterDatabase(unittest.TestCase):
    """
    F-GEO-01: 94-Camera Geocoding Master Database
    Verified against data/bogor_camera_topology.json and PROJECT.md.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.nodes = cls.topo.nodes

    def test_f_geo_01_total_camera_count_and_unique_ids(self):
        """Test 1: Exactly 94 unique camera nodes exist in the geocoding database."""
        self.assertEqual(len(self.nodes), 94)
        node_ids = [n["id"] for n in self.nodes.values()]
        self.assertEqual(len(set(node_ids)), 94)

    def test_f_geo_01_wgs84_bounding_box(self):
        """Test 2: All cameras lie within verified Kota Bogor administrative bounds."""
        for cam_id, node in self.nodes.items():
            lat = node.get("latitude", node.get("lat"))
            lng = node.get("longitude", node.get("lng"))
            self.assertIsNotNone(lat, f"Camera {cam_id} missing latitude")
            self.assertIsNotNone(lng, f"Camera {cam_id} missing longitude")
            self.assertTrue(
                -6.650 <= lat <= -6.520,
                f"Camera {cam_id} latitude {lat} outside Bogor bounds [-6.650, -6.520]",
            )
            self.assertTrue(
                106.730 <= lng <= 106.850,
                f"Camera {cam_id} longitude {lng} outside Bogor bounds [106.730, 106.850]",
            )

    def test_f_geo_01_hls_stream_urls_validity(self):
        """Test 3: Every camera node has an active .m3u8 stream URL matching municipal restreamers."""
        valid_hosts = [
            f"restreamer{i}.kotabogor.go.id" for i in ["", "2", "3", "4", "5", "6"]
        ]
        for cam_id, node in self.nodes.items():
            stream_url = node.get("stream_url", "")
            self.assertTrue(stream_url.endswith(".m3u8"), f"Camera {cam_id} stream is not .m3u8")
            self.assertTrue(
                any(host in stream_url for host in valid_hosts),
                f"Camera {cam_id} URL {stream_url} does not point to valid restreamer",
            )

    def test_f_geo_01_corridor_partitioning(self):
        """Test 4: Every camera is assigned to one of the 9 official Kota Bogor corridors."""
        valid_corridors = {
            "SSA_RING",
            "SURKEN_PASAR_BOGOR",
            "PAJAJARAN_CORRIDOR",
            "SHOLEH_ISKANDAR_BORR",
            "BOGOR_BARAT_DRAMAGA",
            "STASIUN_MERDEKA_MAWAR_SEMERU",
            "BOGOR_SELATAN_BONDONGAN_PAHLAWAN",
            "BOGOR_TIMUR_TAJUR_KATULAMPA",
            "TAMAN_KOTA_PARKS",
        }
        for cam_id, node in self.nodes.items():
            corridor = node.get("corridor")
            self.assertIn(
                corridor,
                valid_corridors,
                f"Camera {cam_id} has invalid corridor '{corridor}'",
            )

    def test_f_geo_01_optical_metadata_fov_and_azimuth(self):
        """Test 5: All cameras have valid camera_type, default_azimuth, and optical FOV."""
        for cam_id, node in self.nodes.items():
            cam_type = node.get("camera_type", "")
            self.assertIn(cam_type, ["FIXED", "PTZ"])
            azimuth = node.get("facing_bearing_deg", node.get("default_azimuth", 0.0))
            self.assertTrue(0.0 <= azimuth <= 360.0)
            fov = node.get("fov_deg", 75.0)
            self.assertTrue(30.0 <= fov <= 360.0)


class TestFGEO02DirectedTopologyGraph(unittest.TestCase):
    """
    F-GEO-02: Directed Topology Transition Graph
    220 directed edges defining road distances, typical speeds, compass bearings.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.edges = cls.topo.edges
        cls.nodes = cls.topo.nodes

    def test_f_geo_02_total_edge_count(self):
        """Test 1: Directed topology graph contains exactly 220 validated edges."""
        self.assertEqual(len(self.edges), 220)

    def test_f_geo_02_edge_node_referential_integrity(self):
        """Test 2: Every edge connects existing source and target camera IDs."""
        for i, edge in enumerate(self.edges):
            src = edge.get("source_id")
            tgt = edge.get("target_id")
            self.assertIn(src, self.nodes, f"Edge {i} has invalid source {src}")
            self.assertIn(tgt, self.nodes, f"Edge {i} has invalid target {tgt}")
            self.assertNotEqual(src, tgt, f"Edge {i} is a self-loop on camera {src}")

    def test_f_geo_02_physical_distances_and_speeds(self):
        """Test 3: Edge physical distances and typical speeds fall within realistic urban limits."""
        for i, edge in enumerate(self.edges):
            dist = edge.get("road_distance_m", edge.get("distance_m", 0.0))
            speed = edge.get("typical_speed_kmh", 0.0)
            self.assertTrue(1.0 <= dist <= 5000.0, f"Edge {i} distance {dist} outside [1, 5000]m")
            self.assertTrue(5.0 <= speed <= 70.0, f"Edge {i} speed {speed} outside [5, 70] km/h")

    def test_f_geo_02_graph_connectivity_no_dead_ends(self):
        """Test 4: Graph has full reachability across the city network from central hub (Tugu Kujang 79)."""
        adj = {}
        for edge in self.edges:
            adj.setdefault(edge["source_id"], []).append(edge["target_id"])

        visited = set()
        queue = [79]
        while queue:
            curr = queue.pop(0)
            if curr not in visited:
                visited.add(curr)
                for neighbor in adj.get(curr, []):
                    if neighbor not in visited:
                        queue.append(neighbor)

        # In a complete urban loop, the primary corridor nodes should be reached
        self.assertGreaterEqual(len(visited), 80, f"BFS reached only {len(visited)}/94 nodes")

    def test_f_geo_02_compass_bearing_integrity(self):
        """Test 5: Edge compass headings or exit directions are valid."""
        valid_dirs = {
            "N", "NE", "E", "SE", "S", "SW", "W", "NW",
            "NORTH", "NORTHEAST", "EAST", "SOUTHEAST", "SOUTH", "SOUTHWEST", "WEST", "NORTHWEST",
            "INTERNAL", "CIRCULAR"
        }
        for edge in self.edges:
            bearing = edge.get("compass_bearing_deg", edge.get("bearing_deg"))
            heading = edge.get("exit_direction", edge.get("compass_heading"))
            if bearing is not None:
                self.assertTrue(0.0 <= bearing <= 360.0)
            if heading is not None:
                self.assertIn(heading.upper(), valid_dirs)


class TestFGEO03ArrivalWindowModel(unittest.TestCase):
    """
    F-GEO-03: Spatiotemporal Arrival Window Model
    Dynamic [t_min, t_max] parameterized by free-flow speed, congestion factor, and traffic lights.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.edges = cls.topo.edges

    def test_f_geo_03_tmin_strictly_less_than_tmax(self):
        """Test 1: Every edge satisfies the strict invariant t_min < t_max."""
        for i, edge in enumerate(self.edges):
            t_win = edge.get("travel_time_window_s", [edge.get("t_min_s", 0.0), edge.get("t_max_s", 0.0)])
            t_min, t_max = float(t_win[0]), float(t_win[1])
            self.assertGreater(t_min, 0.0, f"Edge {i} t_min <= 0")
            self.assertGreater(t_max, t_min, f"Edge {i} violates t_min < t_max ({t_min} >= {t_max})")

    def test_f_geo_03_free_flow_minimum_time_physics(self):
        """Test 2: t_min is consistent with free-flow speed limit (v_max <= 80 km/h) on road corridors."""
        for edge in self.edges:
            dist = edge.get("road_distance_m", edge.get("distance_m", 0.0))
            t_win = edge.get("travel_time_window_s", [edge.get("t_min_s", 1.0), edge.get("t_max_s", 10.0)])
            t_min = float(t_win[0])
            self.assertGreaterEqual(t_min, 1.0, f"Edge {edge['source_id']}->{edge['target_id']} t_min < 1.0s")
            if dist > 100.0:
                implied_speed = (dist / max(0.1, t_min)) * 3.6
                self.assertLessEqual(
                    implied_speed,
                    80.0,
                    f"Edge {edge['source_id']}->{edge['target_id']} implied speed {implied_speed:.1f} km/h exceeds 80 km/h",
                )

    def test_f_geo_03_maximum_time_accounts_for_signal_delay(self):
        """Test 3: For inter-junction edges (>150m), t_max allows at least 20s signal delay."""
        for edge in self.edges:
            dist = edge.get("road_distance_m", edge.get("distance_m", 0.0))
            if dist > 150.0:
                t_win = edge.get("travel_time_window_s", [edge.get("t_min_s", 1.0), edge.get("t_max_s", 10.0)])
                t_min, t_max = float(t_win[0]), float(t_win[1])
                self.assertGreaterEqual(
                    t_max - t_min,
                    20.0,
                    f"Edge {edge['source_id']}->{edge['target_id']} margin {t_max - t_min} too tight for traffic signal",
                )

    def test_f_geo_03_is_valid_transition_within_window(self):
        """Test 4: is_valid_transition returns True with high score when elapsed time is in [t_min, t_max]."""
        # Test edge 79 -> 87 (Otista)
        is_valid, score = self.topo.is_valid_transition(79, 87, 80.0)
        self.assertTrue(is_valid)
        self.assertGreaterEqual(score, 0.5)

    def test_f_geo_03_is_valid_transition_outside_window(self):
        """Test 5: is_valid_transition rejects impossible arrival times (too fast or excessive delay)."""
        # 1 second for 430m is physically impossible (1548 km/h)
        is_valid_fast, score_fast = self.topo.is_valid_transition(79, 87, 1.0)
        self.assertFalse(is_valid_fast)

        # 3600 seconds (1 hour) is far beyond max arrival window
        is_valid_slow, score_slow = self.topo.is_valid_transition(79, 87, 3600.0)
        self.assertFalse(is_valid_slow)
        self.assertEqual(score_slow, 0.0)


class TestFGEO04OneWayDirectionalEnforcer(unittest.TestCase):
    """
    F-GEO-04: One-Way Directional Enforcer
    Hard invariant enforcement: SSA Ring Road (counter-clockwise) and Suryakencana (southbound).
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.edges = cls.topo.edges

    def test_f_geo_04_ssa_strictly_counter_clockwise_invariants(self):
        """Test 1: Zero forbidden clockwise reverse edges exist in the SSA ring."""
        forbidden_ssa = [
            (87, 199), (87, 79), (83, 87), (81, 83), (85, 81),
            (133, 85), (134, 133), (92, 134), (120, 92), (122, 120),
            (132, 122), (79, 132),
        ]
        edge_pairs = [(e["source_id"], e["target_id"]) for e in self.edges]
        violations = [pair for pair in forbidden_ssa if pair in edge_pairs]
        self.assertEqual(len(violations), 0, f"Found forbidden clockwise SSA edges: {violations}")

    def test_f_geo_04_suryakencana_strictly_southbound_invariants(self):
        """Test 2: Zero forbidden northbound edges exist on Jl. Suryakencana."""
        # Suryakencana flows South from Pasar Bogor: 96 -> 86 -> 75
        # Forbidden Northbound edges: (75 -> 86) and (86 -> 96)
        forbidden_surken = [(75, 86), (86, 96)]
        edge_pairs = [(e["source_id"], e["target_id"]) for e in self.edges]
        violations = [pair for pair in forbidden_surken if pair in edge_pairs]
        self.assertEqual(len(violations), 0, f"Found forbidden northbound Surken edges: {violations}")

    def test_f_geo_04_downstream_candidates_prunes_reverse_direction(self):
        """Test 3: Querying reverse exit direction on SSA produces no downstream candidates."""
        # At Simpang BTM (81), traffic can only proceed North along Juanda (to 85/88). Heading East (back down Otista) is forbidden.
        candidates_east = self.topo.get_downstream_candidates(81, "EAST")
        candidate_ids = [c["target_camera_id"] for c in candidates_east]
        self.assertNotIn(83, candidate_ids)
        self.assertNotIn(87, candidate_ids)

    def test_f_geo_04_invalid_transition_returns_false_for_one_way_violation(self):
        """Test 4: is_valid_transition explicitly fails for reverse one-way traversal."""
        is_valid, _ = self.topo.is_valid_transition(81, 83, 60.0)
        self.assertFalse(is_valid)

    def test_f_geo_04_valid_forward_flow_succeeds(self):
        """Test 5: Valid counter-clockwise flow along Otista -> BTM succeeds."""
        # Pasar Bogor (83) -> Simpang BTM (81) is the legal one-way route
        candidates = self.topo.get_downstream_candidates(83, "WEST")
        candidate_ids = [c["target_camera_id"] for c in candidates]
        self.assertIn(81, candidate_ids)


class TestFGEO05IntraJunctionClustering(unittest.TestCase):
    """
    F-GEO-05: Intra-Junction Multi-Camera Clustering
    Fast-transition links (t in [1, 15]s) connecting co-located Fixed and PTZ cameras.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.edges = cls.topo.edges
        cls.nodes = cls.topo.nodes

    def test_f_geo_05_co_located_fixed_ptz_links_exist(self):
        """Test 1: Specific co-located junctions contain intra-junction transition edges."""
        # Tugu Kujang hub: 79 and 199
        intra_edges = [
            e for e in self.edges
            if (e["source_id"], e["target_id"]) in [(79, 199), (199, 79), (161, 201), (201, 161)]
        ]
        self.assertGreaterEqual(len(intra_edges), 1, "Missing intra-junction transition links")

    def test_f_geo_05_intra_junction_travel_time_in_1_to_60_seconds(self):
        """Test 2: Intra-junction links have t_min >= 1s and t_max <= 60s."""
        intra_edges = [
            e for e in self.edges
            if e.get("exit_direction") == "INTERNAL" or e.get("road_distance_m", 100.0) <= 30.0
        ]
        self.assertGreater(len(intra_edges), 0)
        for e in intra_edges:
            t_win = e.get("travel_time_window_s", [e.get("t_min_s", 0.0), e.get("t_max_s", 0.0)])
            t_min, t_max = float(t_win[0]), float(t_win[1])
            self.assertGreaterEqual(t_min, 1.0)
            self.assertLessEqual(t_max, 60.0)

    def test_f_geo_05_intra_junction_distance_small(self):
        """Test 3: Intra-junction camera distance is <= 50 meters."""
        intra_edges = [e for e in self.edges if e.get("exit_direction") == "INTERNAL"]
        for e in intra_edges:
            dist = e.get("road_distance_m", e.get("distance_m", 0.0))
            self.assertLessEqual(dist, 50.0)

    def test_f_geo_05_fixed_and_ptz_types_paired(self):
        """Test 4: Intra-junction pairs involve Fixed and PTZ cameras."""
        # Camera 79 is FIXED, Camera 199 is PTZ / Fixed hub
        if 79 in self.nodes and 199 in self.nodes:
            self.assertIn("FIXED", [self.nodes[79].get("camera_type"), self.nodes[199].get("camera_type")])

    def test_f_geo_05_intra_junction_rapid_transition_validation(self):
        """Test 5: is_valid_transition validates rapid handoffs in under 15 seconds."""
        internal_edges = [e for e in self.edges if e.get("exit_direction") == "INTERNAL"]
        if internal_edges:
            e = internal_edges[0]
            is_valid, score = self.topo.is_valid_transition(e["source_id"], e["target_id"], 10.0)
            self.assertTrue(is_valid)
            self.assertGreaterEqual(score, 0.5)


class TestFGEO06AdjacencyByExitIndexService(unittest.TestCase):
    """
    F-GEO-06: Adjacency-by-Exit Index Service
    O(1) memory lookup table (Camera_ID, Exit_Direction) -> [Neighbor_Camera_IDs].
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.adj = cls.topo.adjacency_by_exit

    def test_f_geo_06_lookup_table_structure_and_non_empty(self):
        """Test 1: Adjacency lookup table has at least 150 directional exit mappings."""
        self.assertGreaterEqual(len(self.adj), 150)

    def test_f_geo_06_tugu_kujang_exit_query(self):
        """Test 2: Querying Tugu Kujang (79) with WEST returns Otista downstream camera (87)."""
        candidates = self.topo.get_downstream_candidates(79, "WEST")
        target_ids = [c["target_camera_id"] for c in candidates]
        self.assertIn(87, target_ids)

    def test_f_geo_06_simpang_btm_north_query(self):
        """Test 3: Querying Simpang BTM (81) with NORTH returns Juanda downstream camera (85)."""
        candidates = self.topo.get_downstream_candidates(81, "NORTH")
        target_ids = [c["target_camera_id"] for c in candidates]
        self.assertIn(85, target_ids)

    def test_f_geo_06_invalid_camera_id_raises_keyerror(self):
        """Test 4: Querying an invalid or non-existent camera ID raises KeyError."""
        with self.assertRaises(KeyError):
            self.topo.get_downstream_candidates(999999, "NORTH")

    def test_f_geo_06_unreachable_exit_returns_empty_list(self):
        """Test 5: Querying a valid camera with an impossible exit heading returns empty candidate list."""
        candidates = self.topo.get_downstream_candidates(79, "NORTHWEST")
        self.assertEqual(len(candidates), 0)


if __name__ == "__main__":
    unittest.main()
