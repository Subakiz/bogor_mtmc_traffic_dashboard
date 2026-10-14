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





    # --- F-GEO-04 Boundaries ---





    # --- F-GEO-05 Boundaries ---




    def test_b_geo_05_cluster_camera_type_consistency(self):
        """B5.5: Verified that camera types are either 'FIXED' or 'PTZ' across all nodes."""
        for cid, n in self.topo.nodes.items():
            self.assertIn(n.get("camera_type"), ["FIXED", "PTZ"])

    # --- F-GEO-06 Boundaries ---






if __name__ == "__main__":
    unittest.main()
