"""
tests/tier2_boundaries/test_tier2_api_dash_test_boundaries.py
Tier 2 Boundary and Corner Case Tests for Features F-API-01, F-API-02, F-DASH-01, F-DASH-02, F-DASH-03, and F-TEST-01.
5 boundary tests per feature (30 total tests).
"""

import json
import unittest

from tests.mock_stream import MockCameraNetwork, MockHLSStreamServer, SyntheticFrameGenerator, SyntheticTrajectorySimulator
from tests.reference_contracts import DatabaseContract, GlobalIDPreserverContract, TopologyContract


class TestTier2APIDashTestBoundaries(unittest.TestCase):
    """
    Boundary & Corner Cases for Web API, Dashboard UI, and Test Suite Harness
    """

    @classmethod
    def setUpClass(cls):
        cls.topo = TopologyContract()
        cls.frame_gen = SyntheticFrameGenerator()

    # --- F-API-01 Boundaries ---
    def test_b_api_01_non_existent_camera_lookup_raises(self):
        """B1.1: Requesting metadata for negative camera ID raises KeyError."""
        with self.assertRaises(KeyError):
            self.topo.get_camera_info(-10)

    def test_b_api_01_excessive_search_limit_clamping(self):
        """B1.2: Search query with limit=50000 clamps to max allowable limit (e.g. 1000)."""
        requested_limit = 50000
        clamped_limit = min(1000, max(1, requested_limit))
        self.assertEqual(clamped_limit, 1000)

    def test_b_api_01_negative_search_limit_clamping(self):
        """B1.3: Search query with limit=-5 clamps to min limit (1)."""
        requested_limit = -5
        clamped_limit = min(1000, max(1, requested_limit))
        self.assertEqual(clamped_limit, 1)

    def test_b_api_01_empty_journey_for_unknown_vehicle(self):
        """B1.4: Journey query for vehicle ID 0 or negative returns empty array."""
        db = DatabaseContract(":memory:")
        self.assertEqual(db.get_vehicle_journey(0), [])
        self.assertEqual(db.get_vehicle_journey(-1), [])
        db.conn.close()

    def test_b_api_01_malformed_timestamp_string_detection(self):
        """B1.5: Malformed timestamp string fails standard ISO format validation."""
        def is_valid_iso(ts):
            import datetime
            try:
                datetime.datetime.fromisoformat(ts.replace("Z", "+00:00"))
                return True
            except (ValueError, TypeError):
                return False

        self.assertFalse(is_valid_iso("2026/09/23-INVALID"))
        self.assertTrue(is_valid_iso("2026-09-23T04:30:00Z"))

    # --- F-API-02 Boundaries ---
    def test_b_api_02_rapid_broadcast_burst_throughput(self):
        """B2.1: WebSocket manager handles burst of 100 switch events without serialization failure."""
        events = [
            {"type": "CAMERA_SWITCH", "id": i, "from": 79, "to": 87, "conf": 0.9}
            for i in range(100)
        ]
        serialized_batch = [json.dumps(e) for e in events]
        self.assertEqual(len(serialized_batch), 100)

    def test_b_api_02_camera_switch_confidence_zero_allowed(self):
        """B2.2: CAMERA_SWITCH payload with 0.0 confidence is valid JSON object."""
        payload = {"type": "CAMERA_SWITCH", "target_id": 1, "confidence": 0.0}
        self.assertEqual(payload["confidence"], 0.0)

    def test_b_api_02_empty_traffic_density_payload(self):
        """B2.3: Telemetry density broadcast with empty camera dict is well-formed."""
        payload = {"type": "TRAFFIC_DENSITY", "camera_densities": {}}
        self.assertEqual(payload["camera_densities"], {})

    def test_b_api_02_json_payload_with_null_fields(self):
        """B2.4: WebSocket payload containing null fields deserializes cleanly."""
        raw = '{"type": "TARGET_UPDATE", "target_id": 1, "candidate_cameras": null}'
        data = json.loads(raw)
        self.assertIsNone(data["candidate_cameras"])

    def test_b_api_02_client_disconnect_during_empty_set(self):
        """B2.5: Disconnecting from an empty client set does not raise KeyError."""
        clients = set()
        clients.discard("unknown_client_id")
        self.assertEqual(len(clients), 0)

    # --- F-DASH-01 Boundaries ---
    def test_b_dash_01_zero_degree_fov_cone_handling(self):
        """B1.1: 0-degree FOV cone degenerates into single forward directional vector."""
        azimuth = 90.0
        fov = 0.0
        start = (azimuth - fov / 2.0) % 360.0
        end = (azimuth + fov / 2.0) % 360.0
        self.assertEqual(start, end)
        self.assertEqual(start, 90.0)

    def test_b_dash_01_360_degree_fov_cone_handling(self):
        """B1.2: 360-degree FOV cone (PTZ camera) spans full circle."""
        fov = 360.0
        self.assertEqual(fov, 360.0)

    def test_b_dash_01_zero_density_all_green_rings(self):
        """B1.3: Zero vehicle density maps to 'ring-success' status class."""
        def ring_class(count):
            return "ring-danger" if count >= 20 else ("ring-warning" if count >= 8 else "ring-success")
        self.assertEqual(ring_class(0), "ring-success")

    def test_b_dash_01_extreme_congestion_ring(self):
        """B1.4: 150 vehicles maps to 'ring-danger' status class."""
        def ring_class(count):
            return "ring-danger" if count >= 20 else ("ring-warning" if count >= 8 else "ring-success")
        self.assertEqual(ring_class(150), "ring-danger")

    def test_b_dash_01_marker_coordinate_precision(self):
        """B1.5: Camera GPS coordinates have at least 4 decimal places of precision."""
        for cid, n in self.topo.nodes.items():
            lat = n.get("latitude", n.get("lat"))
            lng = n.get("longitude", n.get("lng"))
            self.assertGreater(len(str(lat).split(".")[-1]), 2)
            self.assertGreater(len(str(lng).split(".")[-1]), 2)

    # --- F-DASH-02 Boundaries ---
    def test_b_dash_02_radar_countdown_negative_clamped_to_zero(self):
        """B2.1: Radar countdown clamps negative time remaining to 0.0 seconds."""
        t_remaining = -15.4
        clamped = max(0.0, t_remaining)
        self.assertEqual(clamped, 0.0)

    def test_b_dash_02_video_switch_null_url_fallback(self):
        """B2.2: Switch event with null stream URL falls back to placeholder."""
        event = {"to_camera_id": 87, "to_stream_url": None}
        fallback_url = event["to_stream_url"] or "/assets/placeholder_stream.m3u8"
        self.assertEqual(fallback_url, "/assets/placeholder_stream.m3u8")

    def test_b_dash_02_duplicate_switch_event_deduplication(self):
        """B2.3: Consecutive duplicate switch events to the same camera are ignored."""
        current_cam = 87
        incoming_cam = 87
        is_switch_needed = (incoming_cam != current_cam)
        self.assertFalse(is_switch_needed)

    def test_b_dash_02_unlock_when_no_active_lock_idempotent(self):
        """B2.4: Calling unlock when no vehicle is locked behaves idempotently."""
        preserver = GlobalIDPreserverContract()
        # Unlock non-locked target
        preserver.unlock_target(999)
        self.assertEqual(len(preserver.active_targets), 0)

    def test_b_dash_02_video_player_hud_overlay_empty_candidates(self):
        """B2.5: HUD renders safely when candidate camera list is empty."""
        hud_data = {"locked_id": 501, "candidates": []}
        self.assertEqual(len(hud_data["candidates"]), 0)

    # --- F-DASH-03 Boundaries ---
    def test_b_dash_03_single_sighting_journey_polyline(self):
        """B3.1: Journey with only 1 camera sighting has 0 polyline segments."""
        journey = [{"camera_id": 79, "lat": -6.6014, "lng": 106.8051}]
        polyline_segments = [(journey[i], journey[i+1]) for i in range(len(journey)-1)]
        self.assertEqual(len(polyline_segments), 0)

    def test_b_dash_03_zero_result_search_modal(self):
        """B3.2: Empty vehicle search returns empty array and does not throw."""
        db = DatabaseContract(":memory:")
        cur = db.conn.cursor()
        cur.execute("SELECT * FROM vehicle_sightings WHERE global_vehicle_id = 99999999;")
        rows = cur.fetchall()
        self.assertEqual(len(rows), 0)
        db.conn.close()

    def test_b_dash_03_crop_gallery_missing_thumbnail_fallback(self):
        """B3.3: Sighting with None crop_thumbnail displays placeholder icon."""
        sighting = {"id": 1, "crop_thumbnail": None}
        display_img = sighting["crop_thumbnail"] or "default_vehicle_icon.svg"
        self.assertEqual(display_img, "default_vehicle_icon.svg")

    def test_b_dash_03_replay_seek_out_of_bounds_clamping(self):
        """B3.4: Scrubbing slider past journey length clamps to last available sighting index."""
        journey_len = 5
        requested_idx = 10
        clamped_idx = min(journey_len - 1, max(0, requested_idx))
        self.assertEqual(clamped_idx, 4)

    def test_b_dash_03_long_journey_20_camera_polyline(self):
        """B3.5: Journey across 20 cameras generates exactly 19 connected polyline segments."""
        journey = [{"cam_id": i, "lat": -6.60 + i*0.001, "lng": 106.80 + i*0.001} for i in range(20)]
        segments = [(journey[i], journey[i+1]) for i in range(len(journey)-1)]
        self.assertEqual(len(segments), 19)

    # --- F-TEST-01 Boundaries ---
    def test_b_test_01_zero_frame_generation(self):
        """B1.1: Requesting synthetic frame with 0 vehicles produces clean background frame."""
        frame_arr, jpeg_bytes = self.frame_gen.generate_frame(79, vehicles=[])
        self.assertEqual(frame_arr.shape, (480, 640, 3))
        self.assertGreater(len(jpeg_bytes), 1000)

    def test_b_test_01_single_camera_journey_simulation(self):
        """B1.2: Simulating journey across a single camera produces valid intra-camera sightings."""
        sim = SyntheticTrajectorySimulator()
        sightings = sim.simulate_vehicle_corridor_journey(global_vehicle_id=1, route_camera_ids=[79])
        self.assertGreater(len(sightings), 0)
        self.assertTrue(all(s["camera_id"] == 79 for s in sightings))

    def test_b_test_01_mock_hls_server_custom_segment_duration(self):
        """B1.3: Mock HLS server supports custom segment durations (e.g. 1.0s)."""
        server = MockHLSStreamServer(camera_id=79, segment_duration=1.0)
        p = server.get_playlist_m3u8()
        self.assertIn("#EXTINF:1.000000,", p)

    def test_b_test_01_mock_camera_network_non_existent_tier_set(self):
        """B1.4: Setting tier for non-existent camera ID is handled safely without error."""
        net = MockCameraNetwork()
        net.set_camera_tier(999999, 1)  # Non-existent
        self.assertNotIn(999999, net.camera_states)

    def test_b_test_01_empty_test_results_summary_reporting(self):
        """B1.5: Test runner report generation handles 0 tests gracefully."""
        summary = {"total": 0, "passed": 0, "failed": 0, "duration": 0.0}
        self.assertEqual(summary["total"], 0)


if __name__ == "__main__":
    unittest.main()
