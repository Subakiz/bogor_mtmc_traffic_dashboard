"""
tests/test_handoff_candidates.py
Handoff prediction never suggests a camera the BSW site no longer lists; it routes through it to
the next live cameras, so following a vehicle past a removed camera still works.
"""

import unittest

from fastapi.testclient import TestClient

import backend.server as server


class TestLiveHandoffCandidates(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)
        cls.removed = {cid for cid, cam in server.topology.cameras.items() if cam.get("status") == "removed"}

    def predict(self, camera_id, direction):
        res = self.client.post("/api/handoff/predict",
                               json={"camera_id": camera_id, "exit_direction": direction, "vehicle_id": 987654321})
        self.assertEqual(res.status_code, 200)
        return res.json()["candidates"]

    def test_no_removed_camera_is_ever_suggested(self):
        for cid in server.topology.cameras:
            if cid in self.removed:
                continue
            for cand in self.predict(cid, "*"):
                self.assertNotIn(cand["target_camera_id"], self.removed, f"camera {cid} suggested a removed camera")

    def test_routes_through_a_removed_camera_to_the_next_live_one(self):
        if not self.removed:
            self.skipTest("no removed cameras in the topology")
        removed = next(iter(sorted(self.removed)))
        upstream = next(e for e in server.topology.edges
                        if e["target_id"] == removed and e["source_id"] not in self.removed)
        direct = next(e for e in server.topology.edges
                      if e["source_id"] == upstream["source_id"] and e["target_id"] == removed)
        cands = self.predict(upstream["source_id"], direct["exit_direction"])
        beyond = [c for c in cands if c["target_camera_id"] != removed]
        self.assertTrue(beyond, "expected live cameras beyond the removed one")
        for c in beyond:
            self.assertGreaterEqual(c["distance_m"], direct["road_distance_m"])


if __name__ == "__main__":
    unittest.main()
