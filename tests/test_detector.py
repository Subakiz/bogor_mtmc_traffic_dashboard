"""
tests/test_detector.py
Unit Test Suite for CoreML Detection Service on Apple Neural Engine (FEAT-M1).
Tests:
- Strict vehicle class filtering (car, motorcycle, bus, truck accepted; non-vehicles discarded)
- Bounding box coordinate ordering and boundary validation
- Centroid motion vector, speed, and heading angle calculation in [0, 360) deg
- Persistent track ID continuity across sequential frames
- Confidence score thresholding and cutoff pruning
- Track reset functionality
"""

import math
import unittest
import numpy as np

from backend.detector import ANEVehicleDetector, ALLOWED_CLASS_NAMES


class TestDetectorUnit(unittest.TestCase):
    def setUp(self):
        self.detector = ANEVehicleDetector(test_mode=True, conf_threshold=0.35)

    def test_vehicle_class_filtering_discards_non_vehicles(self):
        """Verify only allowed vehicle classes (car, motorcycle, bus, truck) are accepted."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        synthetic_inputs = [
            {"class": "car", "conf": 0.90, "bbox": (50, 50, 150, 150)},
            {"class": "motorcycle", "conf": 0.85, "bbox": (160, 50, 200, 150)},
            {"class": "bus", "conf": 0.88, "bbox": (220, 50, 350, 180)},
            {"class": "truck", "conf": 0.82, "bbox": (360, 50, 480, 200)},
            {"class": "person", "conf": 0.95, "bbox": (10, 10, 30, 80)},    # Non-vehicle
            {"class": "dog", "conf": 0.75, "bbox": (10, 100, 30, 120)},     # Non-vehicle
        ]

        detections = self.detector.detect(frame, synthetic_objects=synthetic_inputs)
        detected_classes = [d["class_name"] for d in detections]

        self.assertEqual(len(detections), 4)
        for c in detected_classes:
            self.assertIn(c, ALLOWED_CLASS_NAMES)
        self.assertNotIn("person", detected_classes)
        self.assertNotIn("dog", detected_classes)

    def test_bounding_box_coordinate_constraints(self):
        """Verify bounding boxes satisfy 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        synthetic_inputs = [
            {"class": "car", "conf": 0.9, "bbox": (-20, -10, 700, 500)},  # Out of bounds
            {"class": "car", "conf": 0.9, "bbox": (100, 100, 100, 100)},  # Zero area
        ]

        detections = self.detector.detect(frame, synthetic_objects=synthetic_inputs)
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            self.assertGreaterEqual(x1, 0)
            self.assertGreaterEqual(y1, 0)
            self.assertLess(x1, x2)
            self.assertLess(y1, y2)
            self.assertLessEqual(x2, 640)
            self.assertLessEqual(y2, 480)

    def test_confidence_threshold_pruning(self):
        """Verify detections with confidence below threshold are pruned."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        synthetic_inputs = [
            {"class": "car", "conf": 0.90, "bbox": (10, 10, 50, 50)},
            {"class": "car", "conf": 0.40, "bbox": (60, 10, 100, 50)},
            {"class": "car", "conf": 0.30, "bbox": (110, 10, 150, 50)},  # Below 0.35
            {"class": "car", "conf": 0.15, "bbox": (160, 10, 200, 50)},  # Below 0.35
        ]

        detections = self.detector.detect(frame, synthetic_objects=synthetic_inputs)
        self.assertEqual(len(detections), 2)
        for d in detections:
            self.assertGreaterEqual(d["confidence"], 0.35)

    def test_motion_heading_and_speed_calculation(self):
        """Verify heading degrees and speed are correctly calculated from centroid motion."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        # Frame 1: Vehicle at (100, 100) at t=0.0
        det1 = self.detector.detect(
            frame,
            synthetic_objects=[{"class": "car", "conf": 0.9, "bbox": (80, 80, 120, 120)}],
            timestamp=100.0,
            camera_azimuth_deg=0.0,
        )[0]
        track_id = det1["track_id"]
        # Initial frame speed should be 0.0
        self.assertEqual(det1["speed_px_s"], 0.0)

        # Frame 2: Vehicle moves East (dx = +100 px, dy = 0) at t=101.0 (1.0s elapsed)
        det2 = self.detector.detect(
            frame,
            synthetic_objects=[{"class": "car", "conf": 0.9, "bbox": (180, 80, 220, 120)}],
            timestamp=101.0,
            camera_azimuth_deg=0.0,
        )[0]

        self.assertEqual(det2["track_id"], track_id)
        # Heading for pure East (dx > 0, dy = 0) is 0 deg
        self.assertAlmostEqual(det2["heading_deg"], 0.0, places=2)
        # Speed should be 100 px / 1.0 s = 100 px/s
        self.assertAlmostEqual(det2["speed_px_s"], 100.0, places=1)

    def test_heading_azimuth_offset(self):
        """Verify heading degree respects camera mounting azimuth angle."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Moving South (dx = 0, dy = +100 px) in screen space is 90 deg
        # With camera azimuth 180 deg, heading should be (90 + 180) % 360 = 270 deg
        _ = self.detector.detect(
            frame,
            synthetic_objects=[{"class": "car", "conf": 0.9, "bbox": (100, 50, 140, 90)}],
            timestamp=10.0,
            camera_azimuth_deg=180.0,
        )
        det2 = self.detector.detect(
            frame,
            synthetic_objects=[{"class": "car", "conf": 0.9, "bbox": (100, 150, 140, 190)}],
            timestamp=11.0,
            camera_azimuth_deg=180.0,
        )[0]

        self.assertAlmostEqual(det2["heading_deg"], 270.0, places=1)

    def test_track_reset(self):
        """Verify reset_tracks clears tracking history and resets track IDs."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        det1 = self.detector.detect(
            frame,
            synthetic_objects=[{"class": "car", "conf": 0.9, "bbox": (100, 100, 150, 150)}],
            timestamp=1.0,
        )[0]
        self.assertEqual(det1["track_id"], 1)

        self.detector.reset_tracks()
        self.assertEqual(len(self.detector.track_history), 0)

        # After reset, next detection gets track ID 1
        det2 = self.detector.detect(
            frame,
            synthetic_objects=[{"class": "car", "conf": 0.9, "bbox": (200, 200, 250, 250)}],
            timestamp=2.0,
        )[0]
        self.assertEqual(det2["track_id"], 1)


if __name__ == "__main__":
    unittest.main()
