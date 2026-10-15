"""
tests/test_server_hardening.py
Host/Origin checks, bounded query parameters and export limits of the dashboard server.
"""

import unittest

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend import server


class HostAndOriginTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_foreign_host_header_is_refused(self):
        self.assertEqual(self.client.get("/api/stats", headers={"Host": "evil.example"}).status_code, 400)
        self.assertEqual(self.client.get("/api/stats").status_code, 200)
        self.assertEqual(self.client.get("/api/stats", headers={"Host": "127.0.0.1:8000"}).status_code, 200)
        self.assertEqual(self.client.get("/api/stats", headers={"Host": "[::1]:8000"}).status_code, 200)

    def test_websocket_from_a_foreign_origin_is_refused(self):
        with self.assertRaises(WebSocketDisconnect):
            with self.client.websocket_connect("/ws/live", headers={"Origin": "http://evil.example"}):
                pass
        with self.client.websocket_connect("/ws/live", headers={"Origin": "http://localhost:8000"}) as ws:
            self.assertIn("counts", ws.receive_json())

    def test_counts_windows_are_bounded(self):
        self.assertEqual(self.client.get("/api/cameras/counts?window_minutes=1e300").status_code, 422)
        self.assertEqual(self.client.get("/api/cameras/counts?rate_window_minutes=nan").status_code, 422)
        self.assertEqual(self.client.get("/api/cameras/counts?window_minutes=1440").status_code, 200)


class ExportLimitsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)

    def test_oversized_dataset_export_is_refused(self):
        res = self.client.get("/api/export/dataset?start=2026-09-01&end=2026-10-01&bucket_minutes=1&fill_gaps=true")
        self.assertEqual(res.status_code, 413)

    def test_second_concurrent_export_gets_429_and_slots_come_back(self):
        taken = 0
        while server._export_slots.acquire(blocking=False):   # as if the allowed exports were streaming
            taken += 1
        try:
            self.assertEqual(self.client.get("/api/export/raw?table=cameras").status_code, 429)
        finally:
            for _ in range(taken):
                server._export_slots.release()
        for _ in range(taken + 1):   # every completed export gives its slot back
            self.assertEqual(self.client.get("/api/export/raw?table=cameras").status_code, 200)
        self.assertEqual(self.client.get("/api/export/raw?table=nope").status_code, 400)   # failed setup too
        for _ in range(taken):
            self.assertTrue(server._export_slots.acquire(blocking=False))
        for _ in range(taken):
            server._export_slots.release()
