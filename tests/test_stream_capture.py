"""
tests/test_stream_capture.py
Unit Test Suite for Threaded Low-Latency HLS Stream Ingestion (FEAT-M4).
Tests:
- ThreadedStreamCapture initialization and background grab thread
- get_new_frame() freshest frame retrieval without queue accumulation
- Resilient auto-reconnection with exponential backoff on simulated drop
- Clean release and resource deallocation
- Real-time playout against the stream clock
"""

import time
import unittest

import cv2
import numpy as np

from backend.stream_capture import ThreadedStreamCapture


class MockVideoCapture:
    """Synthetic OpenCV-style capture: a gradient frame per grab, a stream clock at ``fps``."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 25.0):
        self.width, self.height, self.fps = width, height, fps
        self._opened = True
        self._frame_count = 0

    def isOpened(self) -> bool:
        return self._opened

    def grab(self) -> bool:
        if not self._opened:
            return False
        self._frame_count += 1
        return True

    def retrieve(self):
        if not self._opened:
            return False, None
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[:, :] = ((self._frame_count * 5) % 255, 128, 255 - (self._frame_count * 5) % 255)
        return True, frame

    def get(self, prop_id: int) -> float:
        return self._frame_count * 1000.0 / self.fps if prop_id == cv2.CAP_PROP_POS_MSEC else 0.0

    def release(self):
        self._opened = False


class TestStreamCaptureUnit(unittest.TestCase):
    def test_threaded_stream_capture_frame_retrieval(self):
        """Verify ThreadedStreamCapture retrieves valid frames."""
        mock_cap = MockVideoCapture()
        sc = ThreadedStreamCapture(source=mock_cap, camera_id=79, auto_start=True)

        try:
            # Allow background thread a moment to grab initial frames
            time.sleep(0.1)
            ret, _, frame = sc.get_new_frame(0)
            self.assertTrue(ret)
            self.assertIsNotNone(frame)
            self.assertEqual(frame.shape, (480, 640, 3))
        finally:
            sc.release()

    def test_packet_flushing_returns_newest_frame(self):
        """Verify get_new_frame() hands out the newest frame, skipping the ones in between."""
        mock_cap = MockVideoCapture()
        sc = ThreadedStreamCapture(source=mock_cap, camera_id=79, max_fps=100.0, auto_start=True)

        try:
            # Let background grabber run for 100ms
            time.sleep(0.1)
            ret1, seq1, _ = sc.get_new_frame(0)
            self.assertTrue(ret1)

            time.sleep(0.1)
            ret2, seq2, _ = sc.get_new_frame(seq1)
            self.assertTrue(ret2)
            self.assertGreater(seq2, seq1 + 1)          # frames in between were skipped, not queued
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
            self.assertGreater(sc.current_backoff, 0.05)   # backoff must have increased
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
