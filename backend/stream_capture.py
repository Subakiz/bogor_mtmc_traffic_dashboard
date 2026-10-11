"""
backend/stream_capture.py
BSW Kota Bogor Multi-Camera Tracking ("Follow That Car") Traffic Intelligence Dashboard.
Threaded Low-Latency HLS Stream Ingestion (FEAT-M4).

Implements:
- ThreadedStreamCapture: Decoupled background capture thread running `cap.grab()` in a tight loop.
- Network packet flushing: Continuously flushes old packets from the OS/ffmpeg network buffer.
- `get_frame()`: Decodes and returns only the freshest retrieved frame (`cap.retrieve()`).
- Resilient auto-reconnection: Exponential backoff on stream/network interruptions.
- Clean resource management: Safe `release()` closing capture, terminating thread, and reclaiming memory.
"""

import collections
import glob
import logging
import os
import sys
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

# Keep OpenCV/FFmpeg quiet: joining a live HLS stream mid-GOP prints harmless h264 "reference
# picture missing" warnings for every capture, and dead cameras are reported by the rolling tracker.
os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")   # AV_LOG_FATAL

# Ensure cv2 from the engine virtualenv is accessible if not in the base environment
try:
    import cv2
except ImportError:
    from backend.config import ensure_engine_importable

    ensure_engine_importable()
    try:
        import cv2
    except ImportError:
        cv2 = None

if cv2 is not None:
    try:   # cv2 may already have been imported (and its log level fixed) by the caller
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
    except AttributeError:
        pass

logger = logging.getLogger("stream_capture")

# Low-latency FFmpeg options for HLS (without them OpenCV drifts 15-25 s behind live). Set once
# for the whole process so captures can be opened concurrently: an open takes 2-13 s, and
# serialising them behind a lock left the rolling tracker connecting most of the time.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", (
    "protocol_whitelist;file,http,https,tcp,tls,crypto,httpproxy,rtp,udp|"
    "fflags;nobuffer|flags;low_delay|"
    "live_start_index;-1|"
    "http_persistent;1|"
    "reorder_queue_size;0|"
    "probesize;32768|analyzeduration;0|"
    "max_delay;500000|"
    "timeout;5000000"
))


class MockVideoCapture:
    """Mock video capture object for testing environments without live RTSP/HLS stream."""

    def __init__(self, width: int = 640, height: int = 480, fps: float = 25.0):
        self.width = width
        self.height = height
        self.fps = fps
        self._opened = True
        self._frame_count = 0
        self._last_time = time.time()

    def isOpened(self) -> bool:
        return self._opened

    def grab(self) -> bool:
        if not self._opened:
            return False
        self._frame_count += 1
        return True

    def retrieve(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self._opened:
            return False, None
        # Generate synthetic frame with rolling gradient and timestamp marker
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        color_val = (self._frame_count * 5) % 255
        frame[:, :] = (color_val, 128, 255 - color_val)
        return True, frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.grab():
            return self.retrieve()
        return False, None

    def get(self, prop_id: int) -> float:
        if cv2 is not None and prop_id == cv2.CAP_PROP_POS_MSEC:
            return float(self._frame_count * (1000.0 / self.fps))
        return 0.0

    def release(self):
        self._opened = False


class ThreadedStreamCapture:
    """
    Decoupled background HLS/RTSP capture client.
    Continuously flushes stale network packets via cap.grab() so that get_frame()
    always retrieves the zero-latency live edge.
    """

    def __init__(
        self,
        source: Union[str, int, Any],
        camera_id: Optional[int] = None,
        name: str = "",
        max_fps: float = 30.0,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        buffer_size: int = 2,
        auto_start: bool = True,
    ):
        self.source = source
        self.camera_id = camera_id
        self.name = name or (f"Camera-{camera_id}" if camera_id else str(source))
        self.max_fps = max_fps
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.current_backoff = initial_backoff
        self.buffer_size = buffer_size

        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_available = threading.Condition(self._lock)

        self._cap: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None

        self._is_opened = False
        self._grab_count = 0
        self._retrieve_count = 0
        self._dropped_frames_count = 0
        self._consecutive_failures = 0
        self._last_frame_time = 0.0
        self._start_time = 0.0

        # Ring buffer retaining only the newest frames
        self._ring_buffer = collections.deque(maxlen=self.buffer_size)
        self._cached_frame: Optional[np.ndarray] = None
        self._frame_seq = 0
        self._has_new_grab = False

        if auto_start:
            self._init_initial_capture()
            self.start()

    def _init_initial_capture(self):
        cap = self._create_capture()
        with self._lock:
            if cap is not None and cap.isOpened():
                self._cap = cap
                self._is_opened = True
                self._consecutive_failures = 0
            else:
                self._is_opened = False
                self._cap = None

    def _create_capture(self) -> Optional[Any]:
        """Initializes OpenCV or Mock VideoCapture."""
        if hasattr(self.source, "grab") and hasattr(self.source, "retrieve"):
            # Custom or mock capture provided directly
            return self.source

        if cv2 is None:
            # OpenCV missing: fallback to mock capture
            logger.warning(f"OpenCV not found. Using MockVideoCapture for {self.name}.")
            return MockVideoCapture()

        try:
            # FFmpeg only for URLs: on failure OpenCV would otherwise fall back to its image-sequence
            # reader, which fails again and logs "Couldn't read video stream from file".
            if isinstance(self.source, str):
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(self.source)
            if cap.isOpened():
                return cap
            logger.debug(f"VideoCapture({self.source}) failed to open.")
            return None
        except Exception as e:
            logger.debug(f"Exception initializing capture for {self.source}: {e}")
            return None

    def start(self):
        """Starts the background grabber thread."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._start_time = time.time()
            self._thread = threading.Thread(
                target=self._capture_loop, name=f"Capture-{self.name}", daemon=True
            )
            self._thread.start()

    def is_opened(self) -> bool:
        """Returns True if the stream is currently active and open."""
        with self._lock:
            return self._is_opened and (self._cap is not None)

    def _capture_loop(self):
        """
        Background worker thread. Continuously flushes old frames using cap.grab()
        and manages reconnection with exponential backoff on stream drops.
        """
        min_sleep = 1.0 / max(1.0, self.max_fps * 1.5)
        start_wall = time.time()
        start_video_msec: Optional[float] = None
        max_tolerated_lag_ms = 3500.0

        while not self._stop_event.is_set():
            # Ensure capture is open
            if self._cap is None or not self._is_opened:
                self._reconnect()
                if self._stop_event.is_set():
                    break
                if not self._is_opened:
                    continue
                start_wall = time.time()
                start_video_msec = None

            if self._stop_event.is_set():
                break

            grabbed = False
            try:
                with self._cap_lock:
                    if not self._stop_event.is_set() and self._cap is not None:
                        grabbed = self._cap.grab()
            except Exception as e:
                logger.debug(f"Stream error during grab on {self.name}: {e}")
                grabbed = False

            if grabbed:
                # Fast backlog drainage for network streams to maintain live edge
                pos_msec = 0.0
                if hasattr(self._cap, "get") and cv2 is not None:
                    try:
                        pos_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
                    except Exception:
                        pos_msec = 0.0

                if start_video_msec is None and pos_msec > 0:
                    start_video_msec = pos_msec
                    start_wall = time.time()

                if start_video_msec is not None and pos_msec >= start_video_msec:
                    wall_ms = (time.time() - start_wall) * 1000.0
                    video_ms = pos_msec - start_video_msec
                    lag_ms = wall_ms - video_ms

                    if lag_ms > max_tolerated_lag_ms:
                        dropped = 0
                        while lag_ms > 2200.0 and dropped < 60 and not self._stop_event.is_set():
                            t_g0 = time.perf_counter()
                            with self._cap_lock:
                                if self._cap is None:
                                    break
                                ret_drain = self._cap.grab()
                            if not ret_drain or (time.perf_counter() - t_g0) > 0.015:
                                break
                            dropped += 1
                            if hasattr(self._cap, "get") and cv2 is not None:
                                try:
                                    pos_msec = float(self._cap.get(cv2.CAP_PROP_POS_MSEC))
                                except Exception:
                                    break
                            wall_ms = (time.time() - start_wall) * 1000.0
                            video_ms = pos_msec - start_video_msec
                            lag_ms = wall_ms - video_ms
                        self._dropped_frames_count += dropped

                # Decode only the freshest frame at the live edge
                ret, frame = False, None
                try:
                    with self._cap_lock:
                        if not self._stop_event.is_set() and self._cap is not None:
                            ret, frame = self._cap.retrieve()
                except Exception as e:
                    logger.debug(f"Stream error during retrieve on {self.name}: {e}")
                    ret, frame = False, None

                if ret and frame is not None:
                    with self._lock:
                        self._grab_count += 1
                        self._cached_frame = frame
                        self._frame_seq += 1
                        self._has_new_grab = True
                        self._consecutive_failures = 0
                        self.current_backoff = self.initial_backoff
                        self._last_frame_time = time.time()
                        self._frame_available.notify_all()
                else:
                    with self._lock:
                        self._consecutive_failures += 1
            else:
                with self._lock:
                    self._consecutive_failures += 1
                    # In tests / mock captures, failure threshold is 5; for real network streams tolerate brief packet jitter
                    fail_limit = 5 if not isinstance(self.source, str) else 25
                    now = time.time()
                    time_since_frame = (now - self._last_frame_time) if self._last_frame_time > 0 else 999.0
                    if self._consecutive_failures >= fail_limit and (not isinstance(self.source, str) or time_since_frame >= 1.5):
                        self._is_opened = False
                        if self._cap is not None:
                            try:
                                with self._cap_lock:
                                    if self._cap is not None:
                                        self._cap.release()
                            except Exception:
                                pass
                            self._cap = None
                        logger.info(f"Stream dropped on {self.name}. Reconnecting...")

            time.sleep(min_sleep)

    def _reconnect(self):
        """Attempts reconnection with exponential backoff."""
        if self._stop_event.is_set():
            return

        with self._lock:
            self._reconnect_attempts = getattr(self, "_reconnect_attempts", 0) + 1
            if not isinstance(self.source, str) and self._reconnect_attempts > 5:
                # Max reconnect attempts reached in tests / mock captures: stop cleanly
                self._is_opened = False
                self._stop_event.set()
                return

        time.sleep(self.current_backoff)
        self.current_backoff = min(self.current_backoff * 1.5, self.max_backoff)

        cap = self._create_capture()
        with self._lock:
            if cap is not None and cap.isOpened():
                self._cap = cap
                self._is_opened = True
                self._consecutive_failures = 0
                self._reconnect_attempts = 0
                self.current_backoff = self.initial_backoff
                self._last_frame_time = time.time()
                logger.debug(f"Connected to stream {self.name}.")
            else:
                self._is_opened = False
                self._cap = None

    def get_frame(self, timeout: float = 0.5) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Retrieves the freshest decoded frame safely from cached frame.
        Non-blocking and eliminates multithreaded VideoCapture race conditions.
        """
        with self._lock:
            if self._cached_frame is not None:
                self._retrieve_count += 1
                self._ring_buffer.append(
                    {"frame_id": self._retrieve_count, "timestamp": time.time()}
                )
                return True, self._cached_frame.copy()

            return False, None

    def get_new_frame(
        self, last_seq: int = 0, timeout: float = 0.5, copy: bool = False
    ) -> Tuple[bool, int, Optional[np.ndarray]]:
        """
        Blocks up to ``timeout`` seconds for a frame newer than ``last_seq``.
        Returns (ok, seq, frame). Frames are never mutated after caching, so the default
        ``copy=False`` is safe for read-only consumers and avoids a per-frame memcpy.
        """
        with self._frame_available:
            if self._frame_seq <= last_seq or self._cached_frame is None:
                self._frame_available.wait(timeout=timeout)
            if self._cached_frame is None or self._frame_seq <= last_seq:
                return False, last_seq, None
            self._retrieve_count += 1
            frame = self._cached_frame.copy() if copy else self._cached_frame
            return True, self._frame_seq, frame

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Standard OpenCV-compatible interface for get_frame()."""
        return self.get_frame()

    def get_status(self) -> Dict[str, Any]:
        """Returns real-time stream telemetry and health metrics."""
        with self._lock:
            uptime = (time.time() - self._start_time) if self._start_time > 0 else 0.0
            computed_fps = (self._retrieve_count / uptime) if uptime > 0 else 0.0
            return {
                "camera_id": self.camera_id,
                "name": self.name,
                "source": str(self.source),
                "is_opened": self._is_opened,
                "frames_grabbed": self._grab_count,
                "frames_retrieved": self._retrieve_count,
                "dropped_frames": max(0, self._grab_count - self._retrieve_count),
                "current_backoff_s": round(self.current_backoff, 2),
                "fps": round(computed_fps, 2),
                "last_frame_time": self._last_frame_time,
                "uptime_s": round(uptime, 2),
            }

    def release(self):
        """
        Stops background capture thread, closes VideoCapture, and releases memory.
        """
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
            self._thread = None

        with self._lock:
            if self._cap is not None:
                try:
                    with self._cap_lock:
                        self._cap.release()
                except Exception:
                    pass
                self._cap = None
            self._is_opened = False
            self._cached_frame = None
            self._ring_buffer.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
