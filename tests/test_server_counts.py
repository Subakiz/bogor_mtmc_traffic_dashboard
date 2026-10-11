"""
tests/test_server_counts.py
Tests for /api/cameras/counts endpoint and live traffic vehicle counting across all 94 cameras.
"""

import unittest
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
            self.assertIn("unique_vehicles", cam)
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


if __name__ == "__main__":
    unittest.main()
