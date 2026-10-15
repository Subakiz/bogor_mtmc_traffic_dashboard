"""
tests/test_server_counts.py
Tests for /api/cameras/counts endpoint and live traffic vehicle counting across all 94 cameras.
"""

import unittest
from unittest import mock

from fastapi.testclient import TestClient
from backend.server import app
from tests import TOTAL_CAMERAS


class TestServerCameraCounts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_api_cameras_counts_returns_all_94_cameras(self):
        """Verify /api/cameras/counts returns metrics for all 94 BSW CCTV cameras."""
        res = self.client.get("/api/cameras/counts")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertEqual(data["total_cameras"], TOTAL_CAMERAS)
        self.assertIn("total_vehicles_counted", data)
        self.assertIn("cameras_with_detections", data)
        self.assertIn("class_breakdown", data)
        self.assertIn("top_cameras", data)
        self.assertIn("camera_counts", data)
        self.assertIn("camera_counts_map", data)

        # Check all 94 cameras are represented in camera_counts list
        self.assertEqual(len(data["camera_counts"]), TOTAL_CAMERAS)
        for cam in data["camera_counts"]:
            self.assertIn("camera_id", cam)
            self.assertIn("name", cam)
            self.assertIn("corridor", cam)
            self.assertIn("total_sightings", cam)
            self.assertIn("classes", cam)

    def test_api_stats_includes_camera_counts(self):
        """Verify /api/stats includes camera_counts map and top_cameras."""
        res = self.client.get("/api/stats")
        self.assertEqual(res.status_code, 200)
        data = res.json()

        self.assertIn("stats", data)
        stats = data["stats"]
        self.assertIn("total_sightings", stats)
        self.assertIn("camera_counts", stats)
        self.assertIn("top_cameras", stats)
        self.assertIn("cameras_with_detections", stats)
        self.assertGreaterEqual(stats["active_cameras"], 0)


class TestCitywideVph(unittest.TestCase):
    """Each junction counts once, by its busiest well-sampled camera: one crossing in a 6 s look
    (600 vph, low_sample) must not stand in for a junction that has a properly sampled rate."""

    def setUp(self):
        from backend import server
        self.server = server
        self.clusters = {1: [1, 2], 2: [1, 2], 3: [3]}

    def vph(self, rates):
        cams = {cid: {} for cid in self.clusters}
        with mock.patch.object(self.server.topology, "cameras", cams), \
                mock.patch.object(self.server.topology, "get_intra_junction_cluster",
                                  side_effect=lambda cid: self.clusters[cid]):
            return self.server._citywide_vph(rates)

    def test_low_sample_rate_does_not_beat_a_well_sampled_camera_at_the_same_junction(self):
        rates = {1: {"vph": 300.0, "low_sample": False}, 2: {"vph": 600.0, "low_sample": True}}
        self.assertEqual(self.vph(rates), 300.0)

    def test_low_sample_rate_stands_in_for_a_junction_with_nothing_better(self):
        rates = {1: {"vph": 300.0, "low_sample": False}, 3: {"vph": 600.0, "low_sample": True}}
        self.assertEqual(self.vph(rates), 900.0)

    def test_busiest_well_sampled_camera_counts_once_per_junction(self):
        rates = {1: {"vph": 300.0, "low_sample": False}, 2: {"vph": 450.0, "low_sample": False}}
        self.assertEqual(self.vph(rates), 450.0)


class TestServerEntryPoint(unittest.TestCase):
    def test_main_runs_the_app_object_not_an_import_string(self):
        # start.sh runs backend/server.py as __main__; an import string re-imports the module and
        # builds a second database, analytics engine and tracker. (A source check: running it would
        # start the whole server.)
        import ast
        import inspect
        from backend import server
        calls = [n for n in ast.walk(ast.parse(inspect.getsource(server)))
                 if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "run"
                 and getattr(n.func.value, "id", None) == "uvicorn"]
        self.assertEqual(len(calls), 1)
        self.assertIsInstance(calls[0].args[0], ast.Name)
        self.assertEqual(calls[0].args[0].id, "app")


if __name__ == "__main__":
    unittest.main()
