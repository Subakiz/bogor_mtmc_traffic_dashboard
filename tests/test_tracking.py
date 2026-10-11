"""Tests for per-camera tracking and drawing helpers (no CoreML model required)."""

import unittest
from collections import deque

import numpy as np
from ultralytics.engine.results import Boxes

from car_tracker.pipeline import count_by_class, default_label, draw_detections, update_trails
from car_tracker.tracking import CameraTrackerPool, rows_to_detections


def _boxes(*xyxy_conf_cls, shape=(720, 1280)) -> Boxes:
    arr = np.array(xyxy_conf_cls, dtype=np.float32).reshape(-1, 6)
    return Boxes(arr, shape)


def _car(x, y=300, w=120, h=80, conf=0.9, cls=2):
    return (x, y, x + w, y + h, conf, cls)


class TestCameraTrackerPool(unittest.TestCase):
    def setUp(self):
        self.pool = CameraTrackerPool()
        self.frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    def _step(self, cam, *cars):
        return rows_to_detections(self.pool.update(cam, _boxes(*cars), self.frame))

    def test_track_id_is_stable_for_a_moving_vehicle(self):
        ids = []
        for i in range(5):
            dets = self._step(1, _car(100 + i * 8))
            if dets:
                ids.append(dets[0]["track_id"])
        self.assertGreaterEqual(len(ids), 3)
        self.assertEqual(len(set(ids)), 1)

    def test_track_ids_are_unique_across_cameras(self):
        a = b = None
        for i in range(3):
            a = self._step(1, _car(100 + i * 5))
            b = self._step(2, _car(100 + i * 5))
        self.assertTrue(a and b)
        self.assertNotEqual(a[0]["track_id"], b[0]["track_id"])

    def test_creating_a_new_camera_does_not_reset_existing_ids(self):
        for i in range(3):
            first = self._step(1, _car(100 + i * 5))
        first_id = first[0]["track_id"]
        # Registering camera 2 must not rewind the global counter used by camera 1.
        for i in range(3):
            self._step(2, _car(400 + i * 5))
        for i in range(3):
            fresh = self._step(1, _car(100 + i * 5), _car(700 + i * 5))
        ids = {d["track_id"] for d in fresh}
        self.assertEqual(len(ids), 2)
        self.assertIn(first_id, ids)

    def test_cameras_do_not_share_tracks(self):
        for i in range(3):
            first = self._step(1, _car(100 + i * 5))
        # Same position on a different camera must start its own track, not inherit camera 1's.
        dets = self._step(2, _car(115))
        self.assertTrue(dets)
        self.assertNotEqual(dets[0]["track_id"], first[0]["track_id"])
        self.assertEqual(self.pool.active_cameras(), [1, 2])

    def test_empty_frames_age_out_tracks(self):
        for i in range(3):
            self._step(1, _car(100 + i * 5))
        for _ in range(40):
            dets = self._step(1)
        self.assertEqual(dets, [])

    def test_reset_forgets_camera(self):
        self._step(1, _car(100))
        self.pool.reset(1)
        self.assertEqual(self.pool.active_cameras(), [])

    def test_rows_to_detections_maps_classes(self):
        rows = np.array([[10, 20, 110, 120, 7, 0.8, 3, 0]], dtype=np.float32)
        det = rows_to_detections(rows)[0]
        self.assertEqual(det["class_name"], "motorcycle")
        self.assertEqual(det["track_id"], 7)
        self.assertEqual(det["box"], (10, 20, 110, 120))
        self.assertEqual(rows_to_detections(np.empty((0, 8))), [])


class TestDrawing(unittest.TestCase):
    DETS = [
        {"track_id": 3, "class_id": 2, "class_name": "car", "box": (100, 100, 220, 180), "conf": 0.91},
        {"track_id": 4, "class_id": 3, "class_name": "motorcycle", "box": (300, 120, 340, 190), "conf": 0.77},
    ]

    def test_label_is_class_only(self):
        self.assertEqual(default_label(self.DETS[0]), "CAR")

    def test_count_by_class(self):
        self.assertEqual(count_by_class(self.DETS), {"car": 1, "motorcycle": 1})

    def test_trails_follow_live_tracks_only(self):
        trails = {}
        update_trails(trails, self.DETS)
        self.assertEqual(set(trails), {3, 4})
        update_trails(trails, self.DETS[:1])
        self.assertEqual(set(trails), {3})
        self.assertIsInstance(trails[3], deque)

    def test_draw_detections_modifies_frame(self):
        frame = np.zeros((300, 500, 3), dtype=np.uint8)
        trails = {}
        update_trails(trails, self.DETS)
        update_trails(trails, [dict(self.DETS[0], box=(110, 100, 230, 180))])
        draw_detections(frame, self.DETS, trails)
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
