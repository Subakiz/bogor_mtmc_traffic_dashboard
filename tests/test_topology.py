#!/usr/bin/env python3
"""
Unit Test Suite for Milestone 1: Camera Geospatial & Road Network Topology Engine
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard

Verifies:
- F-GEO-01: 94-Camera Geocoding Master Database & WGS84 Coordinates
- F-GEO-02: 220-Edge Directed Road Network Topology Transition Graph
- F-GEO-03: Spatiotemporal Arrival Window Model
- F-GEO-04: One-Way Directional Enforcer (SSA Counter-Clockwise & Surken Southbound)
- F-GEO-05: Intra-Junction Multi-Camera Clustering & Fast Transitions
- F-GEO-06: Adjacency-by-Exit Index Service & Downstream Queries
"""

import unittest
import math
import os
import json
from backend.topology_engine import TopologyEngine


class TestCameraGeocodingDatabase(unittest.TestCase):
    """F-GEO-01: Master Geocoding Database tests for all 94 CCTV cameras."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_total_camera_count_is_94(self):
        """Verify exactly 94 cameras exist in the database."""
        cameras = self.engine.get_all_cameras()
        self.assertEqual(len(cameras), 94, f"Expected 94 cameras, got {len(cameras)}")
        self.assertEqual(len(self.engine.cameras), 94)

    def test_camera_wgs84_coordinates_validity(self):
        """Verify all 94 cameras possess realistic Bogor municipal coordinates."""
        # Kota Bogor bounds: latitude in [-6.70, -6.50], longitude in [106.70, 106.88]
        for cam in self.engine.get_all_cameras():
            cid = cam["id"]
            lat = cam["lat"]
            lng = cam["lng"]
            self.assertGreaterEqual(lat, -6.70, f"Camera {cid} latitude {lat} too far south")
            self.assertLessEqual(lat, -6.50, f"Camera {cid} latitude {lat} too far north")
            self.assertGreaterEqual(lng, 106.70, f"Camera {cid} longitude {lng} too far west")
            self.assertLessEqual(lng, 106.88, f"Camera {cid} longitude {lng} too far east")

    def test_camera_stream_urls_validity(self):
        """Verify every camera has a non-empty active HLS stream URL."""
        for cam in self.engine.get_all_cameras():
            cid = cam["id"]
            url = cam["stream_url"]
            self.assertTrue(url, f"Camera {cid} has empty stream_url")
            self.assertTrue(url.startswith("https://") or url.startswith("http://"),
                            f"Camera {cid} stream_url is not HTTP/HTTPS: {url}")
            self.assertTrue(url.endswith(".m3u8"), f"Camera {cid} stream_url is not HLS (.m3u8): {url}")

    def test_camera_info_contract_fields(self):
        """Verify camera info dictionary contains all required contract fields."""
        required_fields = [
            "id", "name", "lat", "lng", "corridor", "stream_url", "default_azimuth", "fov_deg"
        ]
        sample_ids = [79, 83, 81, 161, 201, 75, 123, 173]
        for cid in sample_ids:
            cam_info = self.engine.get_camera_info(cid)
            self.assertIsInstance(cam_info, dict)
            for field in required_fields:
                self.assertIn(field, cam_info, f"Missing field '{field}' in camera {cid}")
            self.assertIsInstance(cam_info["id"], int)
            self.assertIsInstance(cam_info["name"], str)
            self.assertIsInstance(cam_info["lat"], float)
            self.assertIsInstance(cam_info["lng"], float)
            self.assertIsInstance(cam_info["corridor"], str)
            self.assertIsInstance(cam_info["stream_url"], str)
            self.assertIsInstance(cam_info["default_azimuth"], (float, int))
            self.assertIsInstance(cam_info["fov_deg"], (float, int))

    def test_get_camera_info_invalid_id_raises_key_error(self):
        """Verify querying non-existent camera ID raises KeyError."""
        with self.assertRaises(KeyError):
            self.engine.get_camera_info(999999)
        with self.assertRaises(KeyError):
            self.engine.get_camera_info(-1)

    def test_corridor_partitioning(self):
        """Verify the 9 municipal corridors partition all 94 cameras."""
        expected_corridors = {
            "SSA_RING": 17,
            "SURKEN_PASAR_BOGOR": 7,
            "PAJAJARAN_CORRIDOR": 10,
            "SHOLEH_ISKANDAR_BORR": 15,
            "BOGOR_BARAT_DRAMAGA": 7,
            "STASIUN_MERDEKA_MAWAR_SEMERU": 12,
            "BOGOR_SELATAN_BONDONGAN_PAHLAWAN": 7,
            "BOGOR_TIMUR_TAJUR_KATULAMPA": 11,
            "TAMAN_KOTA_PARKS": 8,
        }
        total_corridor_cams = 0
        for corridor, expected_count in expected_corridors.items():
            cam_ids = self.engine.get_corridor_cameras(corridor)
            self.assertEqual(
                len(cam_ids), expected_count,
                f"Corridor {corridor} expected {expected_count} cameras, got {len(cam_ids)}"
            )
            total_corridor_cams += len(cam_ids)
        self.assertEqual(total_corridor_cams, 94, "Corridors must partition all 94 cameras")

    def test_get_corridor_cameras_case_insensitive(self):
        """Verify corridor camera lookup is case-insensitive."""
        cams_upper = self.engine.get_corridor_cameras("SSA_RING")
        cams_lower = self.engine.get_corridor_cameras("ssa_ring")
        cams_mixed = self.engine.get_corridor_cameras("Ssa_Ring")
        self.assertEqual(cams_upper, cams_lower)
        self.assertEqual(cams_upper, cams_mixed)
        self.assertEqual(len(cams_upper), 17)


class TestTopologyDirectedGraph(unittest.TestCase):
    """F-GEO-02: Directed Topology Transition Graph validation."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_total_edge_count_is_220(self):
        """Verify exactly 220 directed edges exist in the topology graph."""
        edges = self.engine.get_all_edges()
        self.assertEqual(len(edges), 220, f"Expected 220 edges, got {len(edges)}")

    def test_all_edges_positive_distance(self):
        """Verify all 220 edges have strictly positive road distance."""
        for edge in self.engine.get_all_edges():
            dist = edge["road_distance_m"]
            self.assertGreater(dist, 0.0, f"Edge {edge['source_id']} -> {edge['target_id']} non-positive distance: {dist}")

    def test_all_edges_valid_travel_time_window(self):
        """Verify every edge has t_min > 0 and t_max >= t_min."""
        for edge in self.engine.get_all_edges():
            t_min = edge["t_min_s"]
            t_max = edge["t_max_s"]
            u, v = edge["source_id"], edge["target_id"]
            self.assertGreater(t_min, 0.0, f"Edge {u} -> {v} t_min must be positive, got {t_min}")
            self.assertGreaterEqual(t_max, t_min, f"Edge {u} -> {v} t_max ({t_max}) < t_min ({t_min})")

    def test_zero_dead_ends(self):
        """Verify zero dead-end nodes: every camera has outgoing edges."""
        audit = self.engine.validate_invariants()
        dead_ends = audit["dead_end_nodes"]
        self.assertEqual(len(dead_ends), 0, f"Dead-end nodes found: {dead_ends}")

    def test_zero_unreachable_nodes(self):
        """Verify zero unreachable nodes: every camera has incoming edges."""
        audit = self.engine.validate_invariants()
        unreachable = audit["unreachable_nodes"]
        self.assertEqual(len(unreachable), 0, f"Unreachable nodes found: {unreachable}")

    def test_full_network_reachability_from_tugu_kujang(self):
        """Verify 100% reachability (94/94 cameras) originating from Tugu Kujang (ID 79)."""
        audit = self.engine.validate_invariants()
        reachable_count = audit["kujang_reachable_count"]
        self.assertEqual(reachable_count, 94, f"Only {reachable_count}/94 cameras reachable from Tugu Kujang")

    def test_validate_invariants_method(self):
        """Verify validate_invariants returns valid=True and 0 violations."""
        audit = self.engine.validate_invariants()
        self.assertTrue(audit["valid"])
        self.assertEqual(audit["total_cameras"], 94)
        self.assertEqual(audit["total_edges"], 220)
        self.assertEqual(audit["one_way_violations"], [])
        self.assertEqual(audit["invalid_edge_count"], 0)
        self.assertEqual(audit["invalid_coord_count"], 0)


class TestOneWayDirectionalEnforcer(unittest.TestCase):
    """F-GEO-04: Hard invariant enforcement for one-way corridors."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_ssa_ring_counter_clockwise_invariants(self):
        """
        Verify that NO forbidden reverse edges exist along the SSA Ring Road.
        The SSA operates strictly counter-clockwise around the Botanical Gardens:
        Otista (West) -> Juanda (North) -> Jalak Harupat (East) -> Pajajaran SSA (South) -> Tugu Kujang.
        """
        edges = self.engine.get_all_edges()
        forbidden_ssa = [
            (87, 199),  # Otista Tengah -> Tugu Kujang (Eastbound forbidden)
            (87, 79),   # Otista Tengah -> Tugu Kujang
            (83, 87),   # Pasar Bogor -> Otista Tengah
            (81, 83),   # Simpang BTM -> Pasar Bogor
            (85, 81),   # Juanda Museum -> Simpang BTM (Southbound forbidden)
            (133, 85),  # Juanda Korem -> Juanda Museum
            (134, 133), # Balaikota -> Juanda Korem
            (92, 134),  # Denpom -> Balaikota
            (120, 92),  # Jalak Harupat -> Denpom (Westbound forbidden)
            (122, 120), # Taman Sempur -> Jalak Harupat Barat
            (132, 122), # Pangrango -> Taman Sempur
            (79, 132),  # Tugu Kujang -> Pangrango (Northbound forbidden)
        ]
        edge_pairs = set((e["source_id"], e["target_id"]) for e in edges)
        violations = [pair for pair in forbidden_ssa if pair in edge_pairs]
        self.assertEqual(len(violations), 0, f"SSA one-way violations detected: {violations}")

    def test_suryakencana_southbound_invariants(self):
        """
        Verify that Suryakencana operates strictly southbound.
        Pasar Bogor (83) -> Surken Gate (96) -> Lawang Seketeng (86) -> Gang Aut (75).
        Northbound reverse transitions are strictly forbidden.
        """
        edges = self.engine.get_all_edges()
        forbidden_surken = [
            (75, 86),   # Gang Aut -> Lawang Seketeng (Northbound forbidden)
            (86, 96),   # Lawang Seketeng -> Surken Gate (Northbound forbidden)
            (75, 96),   # Gang Aut -> Surken Gate
        ]
        edge_pairs = set((e["source_id"], e["target_id"]) for e in edges)
        violations = [pair for pair in forbidden_surken if pair in edge_pairs]
        self.assertEqual(len(violations), 0, f"Suryakencana one-way violations detected: {violations}")

    def test_is_valid_transition_rejects_reverse_ssa(self):
        """
        Verify is_valid_transition rejects illegal reverse transitions on SSA.
        Simpang BTM (81) to Pasar Bogor (83) is ~480m apart, but going reverse is illegal.
        In legal circulation, traveling 81 -> 83 requires >380s via the full loop.
        An elapsed time of 40s must be strictly rejected (False, 0.0).
        """
        valid, score = self.engine.is_valid_transition(81, 83, 40.0)
        self.assertFalse(valid, "Reverse SSA transition 81 -> 83 in 40s must be invalid")
        self.assertEqual(score, 0.0, "Reverse SSA score must be 0.0")

        # Direct-only transition check must be completely rejected
        valid_dir, score_dir = self.engine.is_valid_transition(81, 83, 400.0, direct_only=True)
        self.assertFalse(valid_dir)
        self.assertEqual(score_dir, 0.0)

    def test_is_valid_transition_rejects_reverse_suryakencana(self):
        """
        Verify is_valid_transition rejects reverse northbound transitions on Suryakencana.
        Gang Aut (75) to Lawang Seketeng (86) in 30s must be rejected.
        """
        valid, score = self.engine.is_valid_transition(75, 86, 30.0)
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)


class TestDownstreamCandidatesAndRouting(unittest.TestCase):
    """F-GEO-06: Adjacency-by-Exit Index Service and Follow Mode downstream candidate queries."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_pasar_bogor_west_returns_simpang_btm(self):
        """
        Verify sample downstream routing query:
        Vehicle at Pasar Bogor (ID 83) heading WEST -> returns Simpang BTM (IDs 81 and 106).
        """
        candidates = self.engine.get_downstream_candidates(83, "WEST")
        self.assertIsInstance(candidates, list)
        self.assertGreaterEqual(len(candidates), 2, "Expected at least 2 candidates from 83 WEST")
        candidate_ids = [c["target_camera_id"] for c in candidates]
        self.assertIn(81, candidate_ids, "Missing Simpang BTM (81) in downstream candidates")
        self.assertIn(106, candidate_ids, "Missing Simpang BTM Surken (106) in downstream candidates")

    def test_tugu_kujang_west_returns_otista(self):
        """Verify vehicle at Tugu Kujang (79) heading WEST transitions toward Otista Tengah (87)."""
        candidates = self.engine.get_downstream_candidates(79, "WEST")
        candidate_ids = [c["target_camera_id"] for c in candidates]
        self.assertIn(87, candidate_ids, "Tugu Kujang WEST must lead to Otista Tengah (87)")

    def test_invalid_source_camera_raises_key_error(self):
        """Verify invalid source camera ID raises KeyError."""
        with self.assertRaises(KeyError):
            self.engine.get_downstream_candidates(9999, "WEST")

    def test_non_existent_direction_returns_empty_list(self):
        """
        Verify querying a non-existent exit direction returns an empty list.
        Pasar Bogor (83) has only WEST and SOUTH exits. Querying EAST returns [].
        """
        candidates = self.engine.get_downstream_candidates(83, "EAST")
        self.assertEqual(candidates, [])

    def test_downstream_candidates_contract_format(self):
        """Verify every candidate dictionary complies with the PROJECT.md interface contract."""
        required_keys = [
            "target_camera_id", "target_name", "distance_m", "typical_speed_kmh",
            "t_min_s", "t_max_s", "corridor"
        ]
        candidates = self.engine.get_downstream_candidates(83, "WEST")
        for cand in candidates:
            for k in required_keys:
                self.assertIn(k, cand, f"Candidate missing contract key '{k}'")
            self.assertIsInstance(cand["target_camera_id"], int)
            self.assertIsInstance(cand["target_name"], str)
            self.assertIsInstance(cand["distance_m"], float)
            self.assertIsInstance(cand["typical_speed_kmh"], float)
            self.assertIsInstance(cand["t_min_s"], float)
            self.assertIsInstance(cand["t_max_s"], float)
            self.assertIsInstance(cand["corridor"], str)
            self.assertGreater(cand["distance_m"], 0.0)
            self.assertGreater(cand["t_min_s"], 0.0)
            self.assertGreaterEqual(cand["t_max_s"], cand["t_min_s"])


class TestIntraJunctionClustering(unittest.TestCase):
    """F-GEO-05: Intra-Junction Multi-Camera Clustering and fast transitions."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_warung_jambu_fixed_ptz_fast_transitions(self):
        """
        Verify intra-junction fast transitions (t in [1, 15]s) between co-located
        Fixed (161) and PTZ (201) cameras at Simpang Dadali - A. Yani (Warung Jambu).
        """
        # 161 is Fixed, 201 is PTZ
        c161 = self.engine.get_camera_info(161)
        c201 = self.engine.get_camera_info(201)
        self.assertEqual(c161["camera_type"], "FIXED")
        self.assertEqual(c201["camera_type"], "PTZ")

        # Arrival window must be [1, 15]s
        w_f_to_p = self.engine.calculate_arrival_window(161, 201)
        self.assertLessEqual(w_f_to_p[0], 5.0)
        self.assertLessEqual(w_f_to_p[1], 15.0)

        w_p_to_f = self.engine.calculate_arrival_window(201, 161)
        self.assertLessEqual(w_p_to_f[0], 5.0)
        self.assertLessEqual(w_p_to_f[1], 15.0)

        # Transition at 3s should be valid with score 1.0
        valid, score = self.engine.is_valid_transition(161, 201, 3.0)
        self.assertTrue(valid)
        self.assertEqual(score, 1.0)

    def test_cibadak_fixed_ptz_transitions(self):
        """Verify fast transitions between PTZ 169 and Fixed 170/171 at Simpang Cibadak."""
        w1 = self.engine.calculate_arrival_window(169, 170)
        self.assertLessEqual(w1[0], 5.0)
        self.assertLessEqual(w1[1], 15.0)

        w2 = self.engine.calculate_arrival_window(171, 169)
        self.assertLessEqual(w2[0], 5.0)
        self.assertLessEqual(w2[1], 15.0)

    def test_pasar_mawar_fixed_ptz_transitions(self):
        """Verify fast transitions between Fixed 150 and PTZ 152 at Pasar Mawar."""
        w = self.engine.calculate_arrival_window(150, 152)
        self.assertLessEqual(w[0], 5.0)
        self.assertLessEqual(w[1], 15.0)

    def test_semeru_fixed_ptz_transitions(self):
        """Verify fast transitions between Fixed 157 and PTZ 158 at Jl. Dr. Sumeru."""
        w = self.engine.calculate_arrival_window(157, 158)
        self.assertLessEqual(w[0], 5.0)
        self.assertLessEqual(w[1], 15.0)

    def test_is_intra_junction_cluster_membership(self):
        """Verify intra-junction cluster detection logic."""
        # 161 and 201 are co-located at Warung Jambu
        self.assertTrue(self.engine.is_intra_junction(161, 201))
        self.assertTrue(self.engine.is_intra_junction(201, 161))

        # 161 (Warung Jambu) and 79 (Tugu Kujang) are kilometers apart
        self.assertFalse(self.engine.is_intra_junction(161, 79))
        self.assertFalse(self.engine.is_intra_junction(201, 83))

        cluster_161 = self.engine.get_intra_junction_cluster(161)
        self.assertIn(161, cluster_161)
        self.assertIn(201, cluster_161)


class TestSpatiotemporalArrivalWindow(unittest.TestCase):
    """F-GEO-03: Spatiotemporal arrival window modeling and transition feasibility scoring."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_calculate_arrival_window_direct(self):
        """Verify arrival window calculation for direct edges (83 -> 81)."""
        w = self.engine.calculate_arrival_window(83, 81)
        t_min, t_max = w
        self.assertGreater(t_min, 0.0)
        self.assertGreater(t_max, t_min)
        # Distance is ~480m, speed ~20 km/h: t_min ~64s, t_max ~177s
        self.assertGreaterEqual(t_min, 40.0)
        self.assertLessEqual(t_max, 250.0)

    def test_calculate_arrival_window_same_camera(self):
        """Verify arrival window for the same camera returns (0.0, 5.0)."""
        w = self.engine.calculate_arrival_window(79, 79)
        self.assertEqual(w, (0.0, 5.0))

    def test_calculate_arrival_window_multihop(self):
        """Verify arrival window calculation for multi-hop path (e.g. 79 -> 83 via 87)."""
        w = self.engine.calculate_arrival_window(79, 83)
        t_min, t_max = w
        self.assertGreater(t_min, 50.0)
        self.assertGreater(t_max, t_min)

    def test_is_valid_transition_within_window(self):
        """Verify transition within [t_min, t_max] yields score = 1.0 and valid = True."""
        t_min, t_max = self.engine.calculate_arrival_window(83, 81)
        mid_time = (t_min + t_max) / 2.0
        valid, score = self.engine.is_valid_transition(83, 81, mid_time)
        self.assertTrue(valid)
        self.assertEqual(score, 1.0)

    def test_is_valid_transition_negative_elapsed_time(self):
        """Verify negative elapsed time is rejected immediately."""
        valid, score = self.engine.is_valid_transition(83, 81, -5.0)
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)

    def test_is_valid_transition_early_arrival(self):
        """Verify early arrival (t << t_min) is rejected or heavily penalized."""
        t_min, _ = self.engine.calculate_arrival_window(83, 81)
        # Significantly earlier than t_min (e.g., 5 seconds when t_min > 50s)
        valid, score = self.engine.is_valid_transition(83, 81, 5.0)
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)

    def test_is_valid_transition_late_arrival(self):
        """Verify late arrival (t >> t_max) decays to 0.0."""
        _, t_max = self.engine.calculate_arrival_window(83, 81)
        # Extremely late (e.g., 2 hours later)
        valid, score = self.engine.is_valid_transition(83, 81, t_max + 7200.0)
        self.assertFalse(valid)
        self.assertEqual(score, 0.0)

    def test_haversine_distance_computation(self):
        """Verify physical haversine distance between cameras matches reality."""
        # Distance between Tugu Kujang (79) and Masjid Raya (103) is ~700-800m
        dist = self.engine.haversine_distance_m(79, 103)
        self.assertGreaterEqual(dist, 600.0)
        self.assertLessEqual(dist, 900.0)

        # Distance from camera to itself is 0
        dist_self = self.engine.haversine_distance_m(79, 79)
        self.assertAlmostEqual(dist_self, 0.0, places=2)


if __name__ == "__main__":
    unittest.main()
