"""
tests/test_stream_capture.py
Unit Test Suite for Threaded Low-Latency HLS Stream Ingestion (FEAT-M4).
Tests:
- ThreadedStreamCapture initialization and background grab thread
- get_frame() freshest frame retrieval without queue accumulation
- Continuous buffer flushing via cap.grab()
- Resilient auto-reconnection with exponential backoff on simulated drop
- Clean release and resource deallocation
- Telemetry reporting through get_status()
"""

import time
import unittest
import numpy as np

from backend.stream_capture import MockVideoCapture, ThreadedStreamCapture


class TestStreamCaptureUnit(unittest.TestCase):
    def test_mock_video_capture_direct_interface(self):
        """Verify MockVideoCapture adheres to OpenCV VideoCapture contract."""
        cap = MockVideoCapture(width=640, height=480, fps=30.0)
        self.assertTrue(cap.isOpened())

        grabbed = cap.grab()
        self.assertTrue(grabbed)

        ret, frame = cap.retrieve()
        self.assertTrue(ret)
        self.assertEqual(frame.shape, (480, 640, 3))
        self.assertEqual(frame.dtype, np.uint8)

        cap.release()
        self.assertFalse(cap.isOpened())

    def test_threaded_stream_capture_frame_retrieval(self):
        """Verify ThreadedStreamCapture retrieves valid frames."""
        mock_cap = MockVideoCapture()
        sc = ThreadedStreamCapture(source=mock_cap, camera_id=79, auto_start=True)

        try:
            # Allow background thread a moment to grab initial frames
            time.sleep(0.1)
            ret, frame = sc.get_frame()
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
        finally:
            sc.release()

    def test_packet_flushing_returns_newest_frame(self):
        """Verify get_frame() consumes freshest frame without queue delay."""
        mock_cap = MockVideoCapture()
        sc = ThreadedStreamCapture(source=mock_cap, camera_id=79, max_fps=100.0, auto_start=True)

        try:
            # Let background grabber run for 100ms
            time.sleep(0.1)
            ret1, frame1 = sc.get_frame()
            self.assertTrue(ret1)

            # Wait another 100ms so grab count advances
            time.sleep(0.1)
            ret2, frame2 = sc.get_frame()
            self.assertTrue(ret2)

            status = sc.get_status()
            self.assertGreater(status["frames_grabbed"], 5)
            self.assertGreaterEqual(status["frames_retrieved"], 2)
            # The background grabber has grabbed more frames than retrieved (flushing stale frames)
            self.assertGreaterEqual(status["dropped_frames"], 0)
        finally:
            sc.release()

    def test_clean_release_and_thread_termination(self):
        """Verify release() terminates background thread and frees capture."""
        mock_cap = MockVideoCapture()
        sc = ThreadedStreamCapture(source=mock_cap, camera_id=83, auto_start=True)

        self.assertTrue(sc._thread.is_alive())
        sc.release()

        # Thread must have terminated
        self.assertFalse(sc._thread is not None and sc._thread.is_alive())
        self.assertFalse(sc.is_opened())
        self.assertIsNone(sc._cap)

    def test_status_telemetry_reporting(self):
        """Verify get_status returns accurate camera_id, fps, and frame metrics."""
        mock_cap = MockVideoCapture()
        sc = ThreadedStreamCapture(source=mock_cap, camera_id=92, name="CCTV-92", auto_start=True)

        try:
            time.sleep(0.05)
            sc.get_frame()
            status = sc.get_status()

            self.assertEqual(status["camera_id"], 92)
            self.assertEqual(status["name"], "CCTV-92")
            self.assertTrue(status["is_opened"])
            self.assertIn("fps", status)
            self.assertIn("frames_grabbed", status)
            self.assertIn("uptime_s", status)
        finally:
            sc.release()

    def test_reconnection_exponential_backoff_on_drop(self):
        """Verify stream drop triggers exponential backoff."""
        class DroppingCapture(MockVideoCapture):
            def __init__(self):
                super().__init__()
                self.calls = 0

            def grab(self) -> bool:
                self.calls += 1
                if self.calls > 2:
                    return False  # Simulate drop
                return True

        drop_cap = DroppingCapture()
        sc = ThreadedStreamCapture(
            source=drop_cap,
            camera_id=101,
            initial_backoff=0.05,
            max_backoff=0.2,
            auto_start=True,
        )

        try:
            # Wait for failure threshold (5 consecutive failures)
            time.sleep(0.3)
            status = sc.get_status()
            # Backoff must have increased
            self.assertGreater(status["current_backoff_s"], 0.05)
        finally:
            sc.release()


    def test_realtime_playout_follows_the_stream_clock(self):
        """Live sources are played at stream speed (not raced through or slowed down), only
        frames due at max_fps are converted, and a max_fps change applies immediately."""
        class ClockCapture(MockVideoCapture):
            def __init__(self):
                super().__init__(fps=15.0)
                self.grabs = 0
                self.retrieves = 0

            def grab(self) -> bool:
                self.grabs += 1
                return super().grab()

            def retrieve(self):
                self.retrieves += 1
                return super().retrieve()

        cap = ClockCapture()
        sc = ThreadedStreamCapture(source=cap, camera_id=7, max_fps=5.0, realtime=True)
        try:
            time.sleep(1.0)
            grabs, retrieves = cap.grabs, cap.retrieves
            # ~15 grabs/s of a 15 fps stream (an unpaced loop would grab thousands) ...
            self.assertGreaterEqual(grabs, 10)
            self.assertLessEqual(grabs, 22)
            # ... of which only ~5/s are converted.
            self.assertGreaterEqual(retrieves, 3)
            self.assertLessEqual(retrieves, 8)

            sc.max_fps = 15.0
            time.sleep(1.0)
            self.assertGreaterEqual(cap.retrieves - retrieves, 10)
        finally:
            sc.release()


if __name__ == "__main__":
    unittest.main()
