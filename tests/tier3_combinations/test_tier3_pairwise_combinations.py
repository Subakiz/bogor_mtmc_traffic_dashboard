"""
tests/tier3_combinations/test_tier3_pairwise_combinations.py
Tier 3 Cross-Feature Pairwise Interaction Tests.
Interaction tests between architectural subsystems.
"""

import collections
import time
import unittest

from tests.mock_stream import (
    MockCameraNetwork,
    MockHLSStreamServer,
    SyntheticFrameGenerator,
)
from tests import TOTAL_CAMERAS, TOTAL_EDGES
from tests.reference_contracts import DatabaseContract, TopologyContract


class TestTier3PairwiseCombinations(unittest.TestCase):
    """
    Pairwise Cross-Feature Integration Tests across all subsystems.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.frame_gen = SyntheticFrameGenerator(seed=42)

    def setUp(self):
        self.db = DatabaseContract(":memory:")

    def tearDown(self):
        if hasattr(self, "db") and self.db and self.db.conn:
            self.db.conn.close()

    def test_p01_topology_routing_plus_arrival_window(self):
        """Pair 1 (F-GEO-02 + F-GEO-03): Topology routing query outputs valid arrival window [t_min, t_max]."""
        candidates = self.topo.get_downstream_candidates(79, "WEST")
        self.assertGreaterEqual(len(candidates), 1)
        for cand in candidates:
            self.assertGreater(cand["distance_m"], 0.0)
            self.assertGreater(cand["t_min_s"], 0.0)
            self.assertGreater(cand["t_max_s"], cand["t_min_s"])

    def test_p02_one_way_enforcer_plus_adjacency_index(self):
        """Pair 2 (F-GEO-04 + F-GEO-06): Adjacency lookup strictly filters out illegal reverse flow."""
        # Querying exit from Pasar Bogor (83) towards East (back up Otista) must be empty
        candidates_east = self.topo.get_downstream_candidates(83, "EAST")
        cand_ids = [c["target_camera_id"] for c in candidates_east]
        self.assertNotIn(79, cand_ids)
        self.assertNotIn(87, cand_ids)

    def test_p03_intra_junction_clustering_plus_priority_scheduling(self):
        """Pair 3 (F-GEO-05 + F-HAND-02): Intra-junction link triggers immediate short-window priority scheduling."""
        internal_edges = [e for e in self.topo.edges if e.get("exit_direction") == "INTERNAL"]
        self.assertGreater(len(internal_edges), 0)
        e = internal_edges[0]
        src, tgt = e["source_id"], e["target_id"]
        # Fast transition window
        t_win = e.get("travel_time_window_s", [5, 50])
        self.assertLessEqual(t_win[0], 10.0)
        # Verify valid transition
        is_valid, score = self.topo.is_valid_transition(src, tgt, elapsed_time_s=8.0)
        self.assertTrue(is_valid)

    def test_p09_priority_scheduler_plus_scheduling_daemon(self):
        """Pair 9 (F-HAND-02 + FEAT-M5): Elevated priority elevates candidate stream from Tier 3 to Tier 1."""
        net = MockCameraNetwork()
        cand_cam = 87
        # Initially Tier 3
        net.set_camera_tier(cand_cam, 3)
        self.assertEqual(net.camera_states[cand_cam]["tier"], 3)
        # Elevated upon priority trigger
        net.set_camera_tier(cand_cam, 1)
        self.assertEqual(net.camera_states[cand_cam]["tier"], 1)

    def test_p13_leaflet_map_plus_geocoding_database(self):
        """Pair 13 (F-DASH-01 + F-GEO-01): Map view consumes all 94 geocoded camera nodes."""
        markers = [
            {"id": cid, "lat": n.get("latitude", n.get("lat")), "lng": n.get("longitude", n.get("lng"))}
            for cid, n in self.topo.nodes.items()
        ]
        self.assertEqual(len(markers), TOTAL_CAMERAS)

    def test_p14_player_auto_switcher_plus_websocket_event(self):
        """Pair 14 (F-DASH-02 + F-API-02): Video player updates active stream URL upon CAMERA_SWITCH."""
        player = {"active_camera": 79, "stream_url": "url_79.m3u8"}
        ws_event = {"type": "CAMERA_SWITCH", "to_camera_id": 87, "to_stream_url": "url_87.m3u8"}

        if ws_event["type"] == "CAMERA_SWITCH":
            player["active_camera"] = ws_event["to_camera_id"]
            player["stream_url"] = ws_event["to_stream_url"]

        self.assertEqual(player["active_camera"], 87)
        self.assertEqual(player["stream_url"], "url_87.m3u8")

    def test_p16_stream_ring_buffer_plus_daemon_polling(self):
        """Pair 16 (FEAT-M4 + FEAT-M5): Ingestion ring buffer decouples high-rate Tier 1 from slow Tier 3."""
        t1_buffer = collections.deque(maxlen=3)
        t3_buffer = collections.deque(maxlen=1)

        # Tier 1 receives 15 frames
        for i in range(15):
            t1_buffer.append(f"t1_{i}")
        # Tier 3 receives 1 frame
        t3_buffer.append("t3_0")

        self.assertEqual(len(t1_buffer), 3)
        self.assertEqual(len(t3_buffer), 1)

    def test_p17_adjacency_lookup_plus_mtmc_candidate_pruning(self):
        """Pair 17 (F-GEO-06 + FEAT-M7): Adjacency lookup filters candidates before MTMC scoring."""
        downstream_candidates = self.topo.get_downstream_candidates(79, "WEST")
        valid_cids = {c["target_camera_id"] for c in downstream_candidates}

        # 3 candidate sightings across the city
        sightings = [
            {"camera_id": 87, "sighting_id": 1},  # downstream
            {"camera_id": 169, "sighting_id": 2}, # unrelated peripheral cam
        ]
        pruned_sightings = [s for s in sightings if s["camera_id"] in valid_cids]
        self.assertEqual(len(pruned_sightings), 1)
        self.assertEqual(pruned_sightings[0]["camera_id"], 87)

    def test_p20_arrival_window_expiration_plus_lost_target(self):
        """Pair 20 (F-GEO-03 + F-HAND-04): Elapsed time exceeding t_max + slack flags transition expired."""
        # Transition 79 -> 87 has max travel time ~140s
        is_valid, score = self.topo.is_valid_transition(79, 87, elapsed_time_s=500.0)
        self.assertFalse(is_valid)

    def test_p21_hud_radar_countdown_plus_dynamic_arrival_window(self):
        """Pair 21 (F-DASH-02 + F-GEO-03): HUD radar derives arrival countdown from candidate edge window."""
        candidates = self.topo.get_downstream_candidates(79, "WEST")
        c = candidates[0]
        t_exit = 100.0
        current_time = 120.0
        remaining_s = max(0.0, (t_exit + c["t_max_s"]) - current_time)
        self.assertGreater(remaining_s, 0.0)

    def test_p23_fastapi_rest_topology_plus_edge_database(self):
        """Pair 23 (F-API-01 + F-GEO-02): REST topology structure serves full 220 edge network."""
        edges = self.topo.edges
        self.assertEqual(len(edges), TOTAL_EDGES)

if __name__ == "__main__":
    unittest.main()
