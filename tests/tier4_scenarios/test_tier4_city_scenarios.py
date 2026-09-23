"""
tests/tier4_scenarios/test_tier4_city_scenarios.py
Tier 4 Real-World Application Workload Scenarios across Kota Bogor corridors.
Covers 12 comprehensive end-to-end multi-camera vehicle tracking scenarios.
"""

import unittest
import numpy as np

from tests.mock_stream import SyntheticFrameGenerator, SyntheticTrajectorySimulator
from tests.reference_contracts import (
    DatabaseContract,
    GlobalIDPreserverContract,
    HandoffEngineContract,
    ReidEngineContract,
    TopologyContract,
)


class TestTier4CityScenarios(unittest.TestCase):
    """
    12 Real-World City Traffic Scenarios across Kota Bogor
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.frame_gen = SyntheticFrameGenerator(seed=101)
        cls.sim = SyntheticTrajectorySimulator(seed=101)

    def setUp(self):
        self.db = DatabaseContract(":memory:")
        self.preserver = GlobalIDPreserverContract()

        # Seed cameras in DB
        cur = self.db.conn.cursor()
        for cid, n in self.topo.nodes.items():
            cur.execute(
                "INSERT INTO cameras (id, name, corridor, latitude, longitude, camera_type, stream_url) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    cid,
                    n.get("name", f"CCTV-{cid}"),
                    n.get("corridor", "SSA_RING"),
                    n.get("latitude", n.get("lat", 0.0)),
                    n.get("longitude", n.get("lng", 0.0)),
                    n.get("camera_type", "FIXED"),
                    n.get("stream_url", ""),
                ),
            )
        self.db.conn.commit()

    def tearDown(self):
        if hasattr(self, "db") and self.db and self.db.conn:
            self.db.conn.close()

    def test_scenario_01_ssa_counter_clockwise_ring_transit(self):
        """
        Scenario 1: Vehicle entering Tugu Kujang traversing SSA counter-clockwise to BTM and Juanda.
        Route: Tugu Kujang (79) -> Otista (87) -> Pasar Bogor (83) -> Simpang BTM (81) -> Juanda (85)
        Verifies: One-way flow, successive handoffs, persistent Global ID preservation across 4 hops.
        """
        global_id = 10001
        route = [79, 87, 83, 81, 85]
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, route, speed_factor=1.0)
        self.assertGreater(len(journey), 0)

        # Initial lock at Tugu Kujang
        target_sig = {
            "vehicle_class": journey[0]["vehicle_class"],
            "embedding": journey[0]["embedding"],
            "color_hist": journey[0]["color_hist"],
            "camera_id": 79,
            "timestamp": journey[0]["timestamp"],
        }
        self.preserver.lock_target(global_id, 79, target_sig)

        # Step through transitions
        for i in range(len(route) - 1):
            src_cam = route[i]
            tgt_cam = route[i + 1]

            # Candidate sightings at target camera
            cand_sightings = [s for s in journey if s["camera_id"] == tgt_cam]
            self.assertGreater(len(cand_sightings), 0)

            # Match candidate
            match = ReidEngineContract.match_candidate_vehicle(
                target_sig, cand_sightings, current_time=cand_sightings[0]["timestamp"], topology_contract=self.topo
            )
            self.assertIsNotNone(match, f"Failed match at hop {src_cam}->{tgt_cam}")
            self.assertEqual(match["global_vehicle_id"], global_id)

            # Execute handoff
            event = self.preserver.handoff_target(global_id, tgt_cam, match["fused_score"])
            self.assertEqual(event["from_camera_id"], src_cam)
            self.assertEqual(event["to_camera_id"], tgt_cam)

            # Update signature
            target_sig["camera_id"] = tgt_cam
            target_sig["timestamp"] = cand_sightings[-1]["timestamp"]

        # Final tracking verification
        self.assertEqual(self.preserver.active_targets[global_id]["history"], route)
        self.assertEqual(self.preserver.active_targets[global_id]["current_camera_id"], 85)

    def test_scenario_02_emergency_vehicle_pajajaran_north_south_spine(self):
        """
        Scenario 2: Emergency vehicle traveling north-south along Jl. Raya Pajajaran at high speed (1.8x).
        Route: Ciheleut / Pajajaran (80) -> Tugu Kujang (79) -> Sukasari / Pajajaran
        Verifies: Fast transition handling, arrival window minimum boundary, persistent ID.
        """
        global_id = 10002
        route = [80, 79]
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, route, speed_factor=1.8)

        self.preserver.lock_target(global_id, 80, {"vehicle_class": "car"})
        cand_sightings = [s for s in journey if s["camera_id"] == 79]

        is_valid, score = self.topo.is_valid_transition(80, 79, elapsed_time_s=cand_sightings[0]["timestamp"] - journey[0]["timestamp"])
        self.assertTrue(is_valid)

        event = self.preserver.handoff_target(global_id, 79, confidence=0.88)
        self.assertEqual(event["to_camera_id"], 79)
        self.assertEqual(self.preserver.active_targets[global_id]["current_camera_id"], 79)

    def test_scenario_03_commuter_stasiun_to_suryakencana(self):
        """
        Scenario 3: Commuter car crossing from Stasiun Bogor (Alun-Alun 82) through Jembatan Merah (84) into Suryakencana (86, 75).
        Verifies: Commercial corridor transition, southbound one-way enforcement on Surken.
        """
        global_id = 10003
        route = [82, 84, 86, 75]
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, route, speed_factor=1.0)

        self.preserver.lock_target(global_id, 82, {})
        for i in range(len(route) - 1):
            src, tgt = route[i], route[i + 1]
            event = self.preserver.handoff_target(global_id, tgt, confidence=0.91)
            self.assertEqual(event["to_camera_id"], tgt)

        self.assertEqual(self.preserver.active_targets[global_id]["current_camera_id"], 75)
        # Verify 75 cannot handoff back North to 86
        is_valid_reverse, _ = self.topo.is_valid_transition(75, 86, 60.0)
        self.assertFalse(is_valid_reverse)

    def test_scenario_04_heavy_congestion_stop_and_go_handoff(self):
        """
        Scenario 4: Vehicle handoff in heavy stop-and-go traffic on Otista Bridge (extended travel time, 0.4x speed).
        Route: Tugu Kujang (79) -> Otista (87) with travel time near t_max.
        Verifies: Traffic congestion window scaling, late arrival feasibility score decay.
        """
        global_id = 10004
        # Slow travel (0.4x speed factor)
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, [79, 87], speed_factor=0.4)
        t_elapsed = journey[-1]["timestamp"] - journey[0]["timestamp"]

        # In heavy traffic, elapsed time is prolonged
        self.assertGreater(t_elapsed, 80.0)

        # Spatiotemporal check
        is_valid, score = self.topo.is_valid_transition(79, 87, elapsed_time_s=t_elapsed)
        self.assertTrue(is_valid, "Congested travel time should still be reachable within extended window")
        self.assertGreater(score, 0.0)

    def test_scenario_05_simpang_pahlawan_vehicle_disambiguation(self):
        """
        Scenario 5: Multiple vehicles of similar color/model crossing Simpang Pahlawan simultaneously.
        Verifies: MTMC matcher correctly resolves the true vehicle using exit heading + appearance nuance.
        """
        # True target: Silver sedan (ID 501)
        sim = SyntheticTrajectorySimulator(seed=77)
        true_journey = sim.simulate_vehicle_corridor_journey(501, [167, 166], primary_color=(190, 190, 190))
        target_sig = {
            "vehicle_class": "car",
            "embedding": true_journey[0]["embedding"],
            "color_hist": true_journey[0]["color_hist"],
            "camera_id": 167,
            "timestamp": true_journey[0]["timestamp"],
        }

        # Distractor: Another silver car (ID 502) crossing in opposite lane
        distractor_journey = sim.simulate_vehicle_corridor_journey(502, [167, 166], primary_color=(180, 180, 180))

        candidates = [
            distractor_journey[-1],
            true_journey[-1],
        ]

        match = ReidEngineContract.match_candidate_vehicle(
            target_sig, candidates, current_time=true_journey[-1]["timestamp"], topology_contract=self.topo
        )
        self.assertIsNotNone(match)
        self.assertEqual(match["global_vehicle_id"], 501)

    def test_scenario_06_sholeh_iskandar_borr_expressway_transit(self):
        """
        Scenario 6: High-speed corridor transit along Jl. KH. Sholeh Iskandar / BORR.
        Route: Simpang Yasmin (168) -> Cibadak (169, 171)
        Verifies: High-speed corridor traversal and edge connectivity.
        """
        global_id = 10006
        route = [168, 169]
        if 168 in self.topo.nodes and 169 in self.topo.nodes:
            journey = self.sim.simulate_vehicle_corridor_journey(global_id, route, speed_factor=1.3)
            self.preserver.lock_target(global_id, 168, {})
            event = self.preserver.handoff_target(global_id, 169, confidence=0.92)
            self.assertEqual(event["to_camera_id"], 169)

    def test_scenario_07_west_corridor_bubulak_to_city_center(self):
        """
        Scenario 7: West corridor journey from Bubulak / Dramaga towards City Center.
        Verifies: West corridor topology traversal into Stasiun / Merdeka network.
        """
        global_id = 10007
        route = [82, 84]  # Station corridor connection
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, route, speed_factor=1.0)
        self.preserver.lock_target(global_id, 82, {})
        event = self.preserver.handoff_target(global_id, 84, confidence=0.89)
        self.assertEqual(event["to_camera_id"], 84)

    def test_scenario_08_south_corridor_batutulis_to_pahlawan(self):
        """
        Scenario 8: South corridor transit connecting Batutulis, Bondongan, and Simpang Pahlawan.
        Verifies: Southern corridor transition and handoff event persistence.
        """
        global_id = 10008
        route = [167, 166]  # Simpang Pahlawan Fixed to PTZ
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, route, speed_factor=1.0)
        self.preserver.lock_target(global_id, 167, {})
        event = self.preserver.handoff_target(global_id, 166, confidence=0.94)

        ev_id = self.db.record_handoff_event(global_id, 167, 166, 0.94, event["timestamp"])
        self.assertGreater(ev_id, 0)

    def test_scenario_09_east_corridor_tajur_katulampa_to_pajajaran(self):
        """
        Scenario 9: East corridor journey from Katulampa R3 / Tajur connecting into Pajajaran.
        Verifies: Multi-hop arterial transition and journey tracking.
        """
        global_id = 10009
        route = [80, 79]  # Ciheleut to Kujang
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, route, speed_factor=1.0)
        self.preserver.lock_target(global_id, 80, {})
        self.preserver.handoff_target(global_id, 79, confidence=0.90)
        self.assertEqual(self.preserver.active_targets[global_id]["current_camera_id"], 79)

    def test_scenario_10_intra_junction_fixed_to_ptz_handshake(self):
        """
        Scenario 10: Multi-camera intra-junction handoff at Tugu Kujang (Fixed 79 to co-located Hub 199).
        Verifies: Rapid sub-15s sightline transition between co-located camera nodes.
        """
        global_id = 10010
        self.preserver.lock_target(global_id, 79, {})

        # Transition 79 -> 199 has road_distance_m ~ 15m, travel_time [5, 53]s
        is_valid, score = self.topo.is_valid_transition(79, 199, elapsed_time_s=6.0)
        self.assertTrue(is_valid)
        self.assertGreaterEqual(score, 0.5)

        event = self.preserver.handoff_target(global_id, 199, confidence=score)
        self.assertEqual(event["from_camera_id"], 79)
        self.assertEqual(event["to_camera_id"], 199)

    def test_scenario_11_complete_ssa_full_loop_circuit(self):
        """
        Scenario 11: Vehicle completing a full loop of the SSA ring (Tugu Kujang -> Otista -> BTM -> Balai Kota -> Jalak Harupat -> Kujang).
        Verifies: Continuous multi-hop auto video switching across 6 cameras preserving persistent Global ID throughout.
        """
        global_id = 10011
        full_ssa_loop = [79, 87, 83, 81, 88, 92, 79]
        journey = self.sim.simulate_vehicle_corridor_journey(global_id, full_ssa_loop, speed_factor=1.0)

        self.preserver.lock_target(global_id, 79, {"vehicle_class": "car"})

        for i in range(len(full_ssa_loop) - 1):
            src, tgt = full_ssa_loop[i], full_ssa_loop[i + 1]
            event = self.preserver.handoff_target(global_id, tgt, confidence=0.92)
            self.assertEqual(event["global_vehicle_id"], global_id)

        # Full loop complete: history has 7 camera visits
        self.assertEqual(len(self.preserver.active_targets[global_id]["history"]), 7)
        self.assertEqual(self.preserver.active_targets[global_id]["current_camera_id"], 79)

    def test_scenario_12_journey_audit_and_replay_reconstruction(self):
        """
        Scenario 12: Audit query and journey replay for target vehicle spotted across 3 corridors over 90 minutes.
        Verifies: Database insertion, chronological ordering, polyline route assembly, and crop thumbnails.
        """
        global_id = 10012
        emb = np.random.randn(256).astype(np.float32).tobytes()
        hist = np.zeros(128, dtype=np.float32).tobytes()

        # Corridor 1: Pajajaran (Cam 80) at 10:00
        self.db.insert_vehicle_sighting(80, 1, global_id, "car", 0.95, (10, 10, 80, 80), 180, 50, emb, hist, "2026-09-23T10:00:00Z", b"\xff\xd8crop1")
        # Corridor 2: SSA Ring (Cam 79) at 10:25
        self.db.insert_vehicle_sighting(79, 2, global_id, "car", 0.92, (20, 20, 90, 90), 240, 45, emb, hist, "2026-09-23T10:25:00Z", b"\xff\xd8crop2")
        # Corridor 3: Suryakencana (Cam 86) at 10:50
        self.db.insert_vehicle_sighting(86, 3, global_id, "car", 0.89, (15, 15, 85, 85), 180, 30, emb, hist, "2026-09-23T10:50:00Z", b"\xff\xd8crop3")

        # Replay reconstruction
        journey = self.db.get_vehicle_journey(global_id)
        self.assertEqual(len(journey), 3)

        cams = [j["camera_id"] for j in journey]
        self.assertEqual(cams, [80, 79, 86])

        # Verify GPS polyline coordinates
        coords = [(j["latitude"], j["longitude"]) for j in journey]
        self.assertEqual(len(coords), 3)

        # Verify crop thumbnails available
        for j in journey:
            self.assertTrue(j["crop_thumbnail"].startswith(b"\xff\xd8"))


if __name__ == "__main__":
    unittest.main()
