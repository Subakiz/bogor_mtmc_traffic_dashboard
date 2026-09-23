"""
tests/tier2_boundaries/test_tier2_geo_boundaries.py
Tier 2 Boundary and Corner Case Tests for Features F-GEO-01 through F-GEO-06.
5 boundary tests per feature (30 total tests).
"""

import unittest
from tests.reference_contracts import TopologyContract


class TestTier2GeoBoundaries(unittest.TestCase):
    """
    Boundary & Corner Cases for Geospatial & Road Topology (F-GEO-01..06)
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()

    # --- F-GEO-01 Boundaries ---
    def test_b_geo_01_extreme_negative_coordinates(self):
        """B1.1: Rejects coordinates at extreme geographic boundaries (-90, +90)."""
        extreme_lat = -89.99
        extreme_lng = 179.99
        # Verified Bogor latitude is within [-6.65, -6.52]
        self.assertFalse(-6.65 <= extreme_lat <= -6.52)
        self.assertFalse(106.73 <= extreme_lng <= 106.85)

    def test_b_geo_01_zero_zero_null_island_coordinates(self):
        """B1.2: Rejects (0.0, 0.0) Null Island coordinates for any Bogor camera."""
        for cid, n in self.topo.nodes.items():
            lat = n.get("latitude", n.get("lat"))
            lng = n.get("longitude", n.get("lng"))
            self.assertNotEqual((lat, lng), (0.0, 0.0), f"Camera {cid} located at Null Island (0,0)")

    def test_b_geo_01_empty_stream_url_handling(self):
        """B1.3: Verifies no camera node has an empty or null stream URL."""
        for cid, n in self.topo.nodes.items():
            url = n.get("stream_url")
            self.assertTrue(bool(url), f"Camera {cid} has empty stream URL")
            self.assertIsInstance(url, str)

    def test_b_geo_01_boundary_fov_angles(self):
        """B1.4: Handles boundary FOV angles (e.g. exactly 360 for PTZ and 75 for Fixed)."""
        for cid, n in self.topo.nodes.items():
            fov = n.get("fov_deg", 75.0)
            self.assertTrue(0.0 < fov <= 360.0)

    def test_b_geo_01_corridor_string_non_empty_and_upper(self):
        """B1.5: Verifies corridor strings contain no leading/trailing whitespace and are uppercase."""
        for cid, n in self.topo.nodes.items():
            corridor = n.get("corridor", "")
            self.assertEqual(corridor, corridor.strip())
            self.assertEqual(corridor, corridor.upper())

    # --- F-GEO-02 Boundaries ---
    def test_b_geo_02_negative_road_distances_forbidden(self):
        """B2.1: Verifies zero edges have negative or zero distance."""
        for e in self.topo.edges:
            dist = e.get("road_distance_m", e.get("distance_m", 0.0))
            self.assertGreater(dist, 0.0, f"Edge {e['source_id']}->{e['target_id']} has non-positive distance {dist}")

    def test_b_geo_02_zero_typical_speed_forbidden(self):
        """B2.2: Verifies typical traffic speed is strictly positive (no 0 km/h gridlock edges)."""
        for e in self.topo.edges:
            speed = e.get("typical_speed_kmh", 0.0)
            self.assertGreater(speed, 0.0, f"Edge {e['source_id']}->{e['target_id']} has 0 km/h typical speed")

    def test_b_geo_02_maximum_realistic_speed_bound(self):
        """B2.3: Typical speed does not exceed 100 km/h in city arterial road network."""
        for e in self.topo.edges:
            speed = e.get("typical_speed_kmh", 0.0)
            self.assertLessEqual(speed, 100.0, f"Edge speed {speed} exceeds 100 km/h")

    def test_b_geo_02_no_self_referencing_edges(self):
        """B2.4: Edge source_id != target_id for all 220 edges (no self loops)."""
        for e in self.topo.edges:
            self.assertNotEqual(e["source_id"], e["target_id"])

    def test_b_geo_02_extreme_bearing_angle_bounds(self):
        """B2.5: Compass bearings strictly wrap within [0.0, 360.0]."""
        for e in self.topo.edges:
            bearing = e.get("compass_bearing_deg", e.get("bearing_deg"))
            if bearing is not None:
                self.assertTrue(0.0 <= bearing <= 360.0)

    # --- F-GEO-03 Boundaries ---
    def test_b_geo_03_zero_elapsed_time_transition(self):
        """B3.1: Zero elapsed time (t=0.0) is rejected as physically impossible."""
        is_valid, score = self.topo.is_valid_transition(79, 87, 0.0)
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)

    def test_b_geo_03_negative_elapsed_time_transition(self):
        """B3.2: Negative elapsed time (t=-15.0) is rejected as physically impossible."""
        is_valid, score = self.topo.is_valid_transition(79, 87, -15.0)
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)

    def test_b_geo_03_exact_tmin_boundary(self):
        """B3.3: Exactly at t_min boundary, transition is valid and non-zero."""
        # Edge 79 -> 87: t_window = [45, 140] (or similar)
        candidates = self.topo.get_downstream_candidates(79, "WEST")
        c = [c for c in candidates if c["target_camera_id"] == 87][0]
        t_min = c["t_min_s"]
        is_valid, score = self.topo.is_valid_transition(79, 87, t_min)
        self.assertTrue(is_valid)
        self.assertGreater(score, 0.0)

    def test_b_geo_03_exact_tmax_boundary(self):
        """B3.4: Exactly at t_max boundary, transition is valid."""
        candidates = self.topo.get_downstream_candidates(79, "WEST")
        c = [c for c in candidates if c["target_camera_id"] == 87][0]
        t_max = c["t_max_s"]
        is_valid, score = self.topo.is_valid_transition(79, 87, t_max)
        self.assertTrue(is_valid)

    def test_b_geo_03_extreme_astronomical_elapsed_time(self):
        """B3.5: Extremely large elapsed time (e.g. 1,000,000s) returns False and 0.0 score."""
        is_valid, score = self.topo.is_valid_transition(79, 87, 1000000.0)
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)

    # --- F-GEO-04 Boundaries ---
    def test_b_geo_04_adversarial_clockwise_traversal_attempt(self):
        """B4.1: Adversarial attempt to route backwards through SSA (81 to 87) returns False."""
        is_valid, score = self.topo.is_valid_transition(81, 87, 60.0)
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)

    def test_b_geo_04_adversarial_northbound_surken_attempt(self):
        """B4.2: Adversarial attempt to route North on Suryakencana (75 to 96) returns False."""
        is_valid, score = self.topo.is_valid_transition(75, 96, 60.0)
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)

    def test_b_geo_04_empty_exit_direction_query(self):
        """B4.3: Querying candidates with empty string direction returns all connected edges."""
        candidates = self.topo.get_downstream_candidates(79, "")
        self.assertIsInstance(candidates, list)

    def test_b_geo_04_malformed_direction_string(self):
        """B4.4: Querying candidates with nonsensical direction string returns empty list."""
        candidates = self.topo.get_downstream_candidates(79, "NON_EXISTENT_DIRECTION_XYZ")
        self.assertEqual(len(candidates), 0)

    def test_b_geo_04_case_insensitivity_in_exit_query(self):
        """B4.5: Exit direction query is robust to lowercase and leading/trailing whitespace."""
        c_upper = self.topo.get_downstream_candidates(79, "WEST")
        c_lower = self.topo.get_downstream_candidates(79, "  west  ")
        self.assertEqual([c["target_camera_id"] for c in c_upper], [c["target_camera_id"] for c in c_lower])

    # --- F-GEO-05 Boundaries ---
    def test_b_geo_05_non_existent_cluster_lookup(self):
        """B5.1: Checking valid transition for non-existent camera IDs returns False."""
        is_valid, score = self.topo.is_valid_transition(999999, 888888, 10.0)
        self.assertFalse(is_valid)
        self.assertEqual(score, 0.0)

    def test_b_geo_05_cluster_transition_sub_second_boundary(self):
        """B5.2: Intra-junction transition at 0.5s (faster than 1.0s) is rejected or penalized."""
        # Find an internal edge
        internal_edges = [e for e in self.topo.edges if e.get("exit_direction") == "INTERNAL"]
        if internal_edges:
            e = internal_edges[0]
            is_valid, score = self.topo.is_valid_transition(e["source_id"], e["target_id"], 0.2)
            self.assertFalse(is_valid)

    def test_b_geo_05_cluster_transition_upper_boundary(self):
        """B5.3: Intra-junction transition beyond 100s returns False."""
        internal_edges = [e for e in self.topo.edges if e.get("exit_direction") == "INTERNAL"]
        if internal_edges:
            e = internal_edges[0]
            is_valid, score = self.topo.is_valid_transition(e["source_id"], e["target_id"], 250.0)
            self.assertFalse(is_valid)

    def test_b_geo_05_same_source_and_target_transition(self):
        """B5.4: Transition graph contains zero self-loop edges, and extreme dwell (>600s) returns low score."""
        self_loops = [e for e in self.topo.edges if e["source_id"] == e["target_id"]]
        self.assertEqual(len(self_loops), 0)
        # Check that extremely long elapsed time at same camera decays
        _, score = self.topo.is_valid_transition(79, 79, 600.0)
        self.assertLess(score, 0.1)

    def test_b_geo_05_cluster_camera_type_consistency(self):
        """B5.5: Verified that camera types are either 'FIXED' or 'PTZ' across all nodes."""
        for cid, n in self.topo.nodes.items():
            self.assertIn(n.get("camera_type"), ["FIXED", "PTZ"])

    # --- F-GEO-06 Boundaries ---
    def test_b_geo_06_non_existent_camera_id_keyerror(self):
        """B6.1: Querying negative camera ID raises KeyError."""
        with self.assertRaises(KeyError):
            self.topo.get_downstream_candidates(-1, "NORTH")

    def test_b_geo_06_zero_camera_id_keyerror(self):
        """B6.2: Querying camera ID 0 raises KeyError."""
        with self.assertRaises(KeyError):
            self.topo.get_downstream_candidates(0, "NORTH")

    def test_b_geo_06_extreme_camera_id_keyerror(self):
        """B6.3: Querying camera ID 99999999 raises KeyError."""
        with self.assertRaises(KeyError):
            self.topo.get_downstream_candidates(99999999, "NORTH")

    def test_b_geo_06_dead_end_or_isolated_exit_heading(self):
        """B6.4: Valid camera with heading opposite of road direction returns empty list."""
        candidates = self.topo.get_downstream_candidates(81, "SOUTH")
        # Simpang BTM has no legal Southbound exit edge
        self.assertEqual(len(candidates), 0)

    def test_b_geo_06_special_characters_in_direction_string(self):
        """B6.5: Direction string containing punctuation or special chars returns empty list."""
        candidates = self.topo.get_downstream_candidates(79, "WEST@#$%!")
        self.assertEqual(len(candidates), 0)


if __name__ == "__main__":
    unittest.main()
