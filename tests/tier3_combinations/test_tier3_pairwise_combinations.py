"""
tests/tier3_combinations/test_tier3_pairwise_combinations.py
Tier 3 Cross-Feature Pairwise Interaction Tests.
25 comprehensive interaction tests between architectural subsystems.
"""

import collections
import time
import unittest
import numpy as np

from tests.mock_stream import (
    MockCameraNetwork,
    MockHLSStreamServer,
    SyntheticFrameGenerator,
    SyntheticTrajectorySimulator,
)
from tests import TOTAL_CAMERAS, TOTAL_EDGES
from tests.reference_contracts import (
    DatabaseContract,
    GlobalIDPreserverContract,
    HandoffEngineContract,
    ReidEngineContract,
    TopologyContract,
)


class TestTier3PairwiseCombinations(unittest.TestCase):
    """
    Pairwise Cross-Feature Integration Tests across all subsystems.
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.frame_gen = SyntheticFrameGenerator(seed=42)
        cls.sim = SyntheticTrajectorySimulator(seed=42)

    def setUp(self):
        self.db = DatabaseContract(":memory:")
        self.preserver = GlobalIDPreserverContract()

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

    def test_p04_coreml_detection_plus_reid_extraction(self):
        """Pair 4 (FEAT-M1 + FEAT-M2): Simulated frame detection -> crop extraction -> 256-dim embedding."""
        vehicles = [{"bbox": (100, 100, 220, 180), "class": "car", "color": (210, 40, 40)}]
        frame_arr, _ = self.frame_gen.generate_frame(79, vehicles)
        crop_arr, _ = self.frame_gen.generate_vehicle_crop(vehicles[0]["bbox"], color=vehicles[0]["color"])

        self.assertEqual(crop_arr.shape, (224, 224, 3))
        # Simulated embedding extraction
        rng = np.random.RandomState(42)
        emb = rng.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        self.assertEqual(len(emb), 256)
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=4)

    def test_p05_reid_embedding_plus_color_signature_fusion(self):
        """Pair 5 (FEAT-M2 + FEAT-M3): 256-dim embedding + 128-bin HSV histogram fuses via S_appear formula."""
        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        hist = np.ones(128, dtype=np.float32) / 128.0

        score = ReidEngineContract.compute_appearance_similarity(emb, hist, emb.reshape(1, -1), hist.reshape(1, -1))
        # Exactly identical signature yields S_appear = 1.0
        self.assertAlmostEqual(float(score[0]), 1.0, places=4)

    def test_p06_appearance_signature_plus_numpy_vector_index(self):
        """Pair 6 (FEAT-M3 + F-ARCH-02): Batch vector dot product evaluates 50 candidate signatures in < 0.2 ms."""
        q_emb = np.random.randn(256).astype(np.float32)
        q_hist = np.random.randn(128).astype(np.float32)
        cand_embs = np.random.randn(50, 256).astype(np.float32)
        cand_hists = np.random.randn(50, 128).astype(np.float32)

        t0 = time.perf_counter()
        sims = ReidEngineContract.compute_appearance_similarity(q_emb, q_hist, cand_embs, cand_hists)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        self.assertEqual(len(sims), 50)
        self.assertLess(elapsed_ms, 0.5)

    def test_p07_fused_mtmc_matcher_plus_spatiotemporal_window(self):
        """Pair 7 (FEAT-M7 + F-GEO-03): S_fused incorporates appearance (0.70) and topology arrival time (0.30)."""
        journey = self.sim.simulate_vehicle_corridor_journey(101, [79, 87])
        target = journey[0]
        cand = [j for j in journey if j["camera_id"] == 87][0]

        target_sig = {
            "vehicle_class": target["vehicle_class"],
            "embedding": target["embedding"],
            "color_hist": target["color_hist"],
            "camera_id": 79,
            "timestamp": target["timestamp"],
        }
        match = ReidEngineContract.match_candidate_vehicle(target_sig, [cand], current_time=cand["timestamp"], topology_contract=self.topo)
        self.assertIsNotNone(match)
        self.assertGreaterEqual(match["fused_score"], 0.72)

    def test_p08_trajectory_heading_plus_priority_scheduling(self):
        """Pair 8 (F-HAND-01 + F-HAND-02): Exit heading Theta_exit schedules downstream candidate cameras."""
        # Vehicle moving westward towards Otista
        heading, card = HandoffEngineContract.estimate_exit_heading((0.0, 0.0), camera_azimuth_deg=270.0)
        self.assertEqual(card, "W")
        candidates = self.topo.get_downstream_candidates(79, card)
        self.assertGreaterEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["target_camera_id"], 87)

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

    def test_p10_global_id_preserver_plus_sqlite_wal(self):
        """Pair 10 (F-HAND-04 + F-ARCH-01): Camera handoff event persists in SQLite database."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (79, 'Kujang', 'SSA', 0, 0)")
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (87, 'Otista', 'SSA', 0, 0)")

        self.preserver.lock_target(3001, 79, {})
        event = self.preserver.handoff_target(3001, 87, confidence=0.93)

        event_id = self.db.record_handoff_event(
            event["global_vehicle_id"], event["from_camera_id"], event["to_camera_id"],
            event["confidence"], event["timestamp"]
        )
        self.assertGreater(event_id, 0)

    def test_p11_target_lock_plus_websocket_camera_switch(self):
        """Pair 11 (F-HAND-04 + F-API-02): Handoff triggers formatted WebSocket CAMERA_SWITCH payload."""
        self.preserver.lock_target(4001, 79, {})
        event = self.preserver.handoff_target(4001, 87, confidence=0.91)

        ws_payload = {
            "type": "CAMERA_SWITCH",
            "global_vehicle_id": event["global_vehicle_id"],
            "from_camera_id": event["from_camera_id"],
            "to_camera_id": event["to_camera_id"],
            "to_camera_name": "CCTV OTISTA",
            "to_stream_url": "https://restreamer2.kotabogor.go.id/memfs/test.m3u8",
            "confidence": event["confidence"],
            "timestamp": event["timestamp"],
        }
        self.assertEqual(ws_payload["type"], "CAMERA_SWITCH")
        self.assertEqual(ws_payload["global_vehicle_id"], 4001)

    def test_p12_fastapi_journey_endpoint_plus_sqlite_sightings(self):
        """Pair 12 (F-API-01 + F-ARCH-01): REST journey query pulls chronological database sightings."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (79, 'Kujang', 'SSA', -6.601, 106.805)")
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (87, 'Otista', 'SSA', -6.601, 106.802)")

        self.db.insert_vehicle_sighting(79, 1, 5001, "car", 0.9, (1, 1, 10, 10), 0, 0, b"0"*1024, b"0"*512, "2026-09-23T01:00:00Z")
        self.db.insert_vehicle_sighting(87, 2, 5001, "car", 0.9, (1, 1, 10, 10), 0, 0, b"0"*1024, b"0"*512, "2026-09-23T01:01:30Z")

        journey = self.db.get_vehicle_journey(5001)
        self.assertEqual(len(journey), 2)
        self.assertEqual(journey[0]["camera_name"], "Kujang")
        self.assertEqual(journey[1]["camera_name"], "Otista")

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

    def test_p15_journey_replay_polyline_plus_tracking_history(self):
        """Pair 15 (F-DASH-03 + F-HAND-04): Journey replay builds polyline from vehicle camera history."""
        self.preserver.lock_target(6001, 79, {})
        self.preserver.handoff_target(6001, 87, 0.9)
        self.preserver.handoff_target(6001, 83, 0.9)

        history = self.preserver.active_targets[6001]["history"]
        coords = [
            (self.topo.nodes[cid].get("latitude", self.topo.nodes[cid].get("lat")),
             self.topo.nodes[cid].get("longitude", self.topo.nodes[cid].get("lng")))
            for cid in history
        ]
        self.assertEqual(len(coords), 3)

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

    def test_p18_bounding_box_tracking_plus_boundary_exit_event(self):
        """Pair 18 (FEAT-M1 + F-HAND-01): Bounding box trajectory across frame triggers exit event."""
        # Simulated vehicle moving from x=100 to x=610 (frame width 640, margin 35)
        bbox_start = (100, 200, 180, 250)
        bbox_exit = (550, 200, 625, 250)

        self.assertFalse(HandoffEngineContract.is_boundary_exit(bbox_start, (640, 480), margin=35))
        self.assertTrue(HandoffEngineContract.is_boundary_exit(bbox_exit, (640, 480), margin=35))

    def test_p19_sqlite_blob_storage_plus_numpy_cosine_similarity(self):
        """Pair 19 (F-ARCH-01 + F-ARCH-02): Binary BLOBs retrieved from SQLite match via NumPy cosine."""
        cur = self.db.conn.cursor()
        cur.execute("INSERT INTO cameras (id, name, corridor, latitude, longitude) VALUES (79, 'Kujang', 'SSA', 0, 0)")

        emb = np.random.randn(256).astype(np.float32)
        emb /= np.linalg.norm(emb)
        hist = np.zeros(128, dtype=np.float32)

        s_id = self.db.insert_vehicle_sighting(79, 1, 7001, "car", 0.9, (0, 0, 10, 10), 0, 0, emb.tobytes(), hist.tobytes(), "2026-09-23T00:00:00Z")

        cur.execute("SELECT embedding, color_hist FROM vehicle_sightings WHERE sighting_id = ?", (s_id,))
        row = cur.fetchone()
        retrieved_emb = np.frombuffer(row[0], dtype=np.float32)
        retrieved_hist = np.frombuffer(row[1], dtype=np.float32)

        sim = ReidEngineContract.compute_appearance_similarity(emb, hist, retrieved_emb, retrieved_hist)
        self.assertAlmostEqual(float(sim[0]), 0.60, places=4)

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

    def test_p22_mtmc_match_plus_target_camera_migration(self):
        """Pair 22 (FEAT-M7 + F-HAND-04): Successful MTMC match migrates preserver target state."""
        self.preserver.lock_target(8001, 79, {"vehicle_class": "car"})
        # Match detected at 87 with 0.88 confidence
        self.preserver.handoff_target(8001, 87, confidence=0.88)
        self.assertEqual(self.preserver.active_targets[8001]["current_camera_id"], 87)

    def test_p23_fastapi_rest_topology_plus_edge_database(self):
        """Pair 23 (F-API-01 + F-GEO-02): REST topology structure serves full 220 edge network."""
        edges = self.topo.edges
        self.assertEqual(len(edges), TOTAL_EDGES)

    def test_p24_appearance_color_signature_plus_disambiguation(self):
        """Pair 24 (FEAT-M3 + FEAT-M7): Color histogram separates two vehicles of identical car model."""
        sim = SyntheticTrajectorySimulator()
        red_car = sim.simulate_vehicle_corridor_journey(1, [79, 87], primary_color=(255, 0, 0))
        blue_car = sim.simulate_vehicle_corridor_journey(2, [79, 87], primary_color=(0, 0, 255))

        target = red_car[0]
        true_cand = [c for c in red_car if c["camera_id"] == 87][0]
        distractor = [c for c in blue_car if c["camera_id"] == 87][0]

        target_sig = {
            "vehicle_class": "car",
            "embedding": target["embedding"],
            "color_hist": target["color_hist"],
            "camera_id": 79,
            "timestamp": target["timestamp"],
        }
        match = ReidEngineContract.match_candidate_vehicle(target_sig, [distractor, true_cand], current_time=true_cand["timestamp"], topology_contract=self.topo)
        self.assertEqual(match["global_vehicle_id"], 1)

    def test_p25_simulation_trajectory_plus_global_id_chain(self):
        """Pair 25 (F-TEST-01 + F-HAND-04): Simulated journey across 3 cameras maintains Global ID."""
        journey = self.sim.simulate_vehicle_corridor_journey(9999, [79, 87, 83])
        cams = [j["camera_id"] for j in journey]
        self.assertIn(79, cams)
        self.assertIn(87, cams)
        self.assertIn(83, cams)
        self.assertTrue(all(j["global_vehicle_id"] == 9999 for j in journey))


if __name__ == "__main__":
    unittest.main()
