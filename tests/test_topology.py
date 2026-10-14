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
from tests import TOTAL_CAMERAS, TOTAL_EDGES


class TestCameraGeocodingDatabase(unittest.TestCase):
    """F-GEO-01: Master Geocoding Database tests for all 94 CCTV cameras."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_total_camera_count_is_94(self):
        """Verify exactly 94 cameras exist in the database."""
        cameras = self.engine.get_all_cameras()
        self.assertEqual(len(cameras), TOTAL_CAMERAS, f"Expected {TOTAL_CAMERAS} cameras, got {len(cameras)}")
        self.assertEqual(len(self.engine.cameras), TOTAL_CAMERAS)

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
        """Verify the 9 municipal corridors partition every camera."""
        # 2026-10-11: +6 SSA_RING (203, 204, 205, 207, 208, 209), +1 SURKEN (206), +2 DRAMAGA (148, 155)
        expected_corridors = {
            "SSA_RING": 23,
            "SURKEN_PASAR_BOGOR": 8,
            "PAJAJARAN_CORRIDOR": 10,
            "SHOLEH_ISKANDAR_BORR": 15,
            "BOGOR_BARAT_DRAMAGA": 9,
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
        self.assertEqual(total_corridor_cams, TOTAL_CAMERAS, "Corridors must partition every camera")

    def test_get_corridor_cameras_case_insensitive(self):
        """Verify corridor camera lookup is case-insensitive."""
        cams_upper = self.engine.get_corridor_cameras("SSA_RING")
        cams_lower = self.engine.get_corridor_cameras("ssa_ring")
        cams_mixed = self.engine.get_corridor_cameras("Ssa_Ring")
        self.assertEqual(cams_upper, cams_lower)
        self.assertEqual(cams_upper, cams_mixed)
        self.assertEqual(len(cams_upper), 23)


class TestTopologyDirectedGraph(unittest.TestCase):
    """F-GEO-02: Directed Topology Transition Graph validation."""

    def setUp(self):
        self.engine = TopologyEngine()

    def test_total_edge_count_is_220(self):
        """Verify exactly 220 directed edges exist in the topology graph."""
        edges = self.engine.get_all_edges()
        self.assertEqual(len(edges), TOTAL_EDGES, f"Expected {TOTAL_EDGES} edges, got {len(edges)}")

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




class TestDownstreamCandidatesAndRouting(unittest.TestCase):
    """F-GEO-06: Adjacency-by-Exit Index Service and Follow Mode downstream candidate queries."""

    def setUp(self):
        self.engine = TopologyEngine()







class TestIntraJunctionClustering(unittest.TestCase):
    """F-GEO-05: Intra-Junction Multi-Camera Clustering and fast transitions."""

    def setUp(self):
        self.engine = TopologyEngine()





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










if __name__ == "__main__":
    unittest.main()
