"""
backend/stream_capture.py
BSW Kota Bogor Multi-Camera Traffic Intelligence Dashboard.
Threaded Low-Latency HLS Stream Ingestion (FEAT-M4).

Implements:
- ThreadedStreamCapture: background capture thread that grabs every packet and plays a live
  stream out in real time against its own timestamps (HLS delivers whole segments at once).
- Only frames due at ``max_fps`` are converted to BGR (``cap.retrieve()``), the rest are just
  grabbed; ``max_fps`` is re-read every frame, so a rolling visit promoted to focus speeds up.
- `get_new_frame()`: returns the newest converted frame.
- Resilient auto-reconnection: Exponential backoff on stream/network interruptions.
- Clean resource management: Safe `release()` closing capture and terminating the thread.
"""

import logging
import os
import threading
import time
from typing import Any, Optional, Tuple, Union

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

# FFmpeg options for HLS. ``live_start_index;-1`` starts at the newest segment (OpenCV's default
# starts several segments back and runs 15-25 s behind live). Set once for the whole process so
# captures can be opened concurrently: an open takes 2-13 s, and serialising them behind a lock
# left the rolling tracker connecting most of the time.
# No ``fflags;nobuffer``: it discards the packets read while probing -- including the keyframe
# that starts the segment -- so the first frame only arrived one GOP (median 3.2 s) after open.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", (
    "protocol_whitelist;file,http,https,tcp,tls,crypto,httpproxy,rtp,udp|"
    "flags;low_delay|"
    "live_start_index;-1|"
    "http_persistent;1|"
    "reorder_queue_size;0|"
    "probesize;32768|analyzeduration;0|"
    "max_delay;500000|"
    "timeout;5000000"
))


# A live stream more than this far off its real-time playout schedule is re-anchored (stream
# restart, timestamp jump, or a network stall longer than a segment).
RESYNC_S = 3.0


class ThreadedStreamCapture:
    """
    Decoupled background HLS/RTSP capture client.

    Live (URL) sources are played out in real time: each grabbed frame is held until its stream
    timestamp is due on the wall clock. HLS hands over a whole segment (1-5 s of video) at a time,
    so without this the reader either races through a segment and then stalls, or -- pacing grabs
    on the wall clock below the source frame rate -- plays the video in slow motion (~0.75x),
    which made rolling-visit rates and speeds read ~25% low. Playout runs about one segment
    behind live. ``realtime=False`` (the default for injected capture objects) consumes frames
    as fast as ``max_fps`` allows instead.
    """

    def __init__(
        self,
        source: Union[str, int, Any],
        camera_id: Optional[int] = None,
        name: str = "",
        max_fps: float = 30.0,
        initial_backoff: float = 0.5,
        max_backoff: float = 8.0,
        auto_start: bool = True,
        realtime: Optional[bool] = None,
    ):
        self.source = source
        self.realtime = isinstance(source, str) if realtime is None else bool(realtime)
        self.camera_id = camera_id
        self.name = name or (f"Camera-{camera_id}" if camera_id else str(source))
        self.max_fps = max_fps
        self.initial_backoff = initial_backoff
        self.max_backoff = max_backoff
        self.current_backoff = initial_backoff

        self._lock = threading.Lock()
        self._cap_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._frame_available = threading.Condition(self._lock)

        self._cap: Optional[Any] = None
        self._thread: Optional[threading.Thread] = None

        self._is_opened = False
        self._consecutive_failures = 0
        self._last_frame_time = 0.0

        self._cached_frame: Optional[np.ndarray] = None
        self._frame_seq = 0

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
        """Opens the OpenCV capture (or returns an injected capture object)."""
        if hasattr(self.source, "grab") and hasattr(self.source, "retrieve"):
            # Custom or mock capture provided directly
            return self.source

        if cv2 is None:
            logger.warning(f"OpenCV not found; cannot open {self.name}.")
            return None

        try:
            # FFmpeg only for URLs: on failure OpenCV would otherwise fall back to its image-sequence
            # reader, which fails again and logs "Couldn't read video stream from file".
            if isinstance(self.source, str):
                # One decoder thread: frame-threaded decoding (OpenCV defaults to one thread per
                # CPU) adds ~19 native threads and ~60 MB per stream plus a 10-frame holdback,
                # and costs more CPU in total; one core decodes 1080p20 comfortably.
                n_threads = getattr(cv2, "CAP_PROP_N_THREADS", None)
                params = [n_threads, 1] if n_threads is not None else []
                cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG, params)
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
            self._thread = threading.Thread(
                target=self._capture_loop, name=f"Capture-{self.name}", daemon=True
            )
            self._thread.start()

    def is_opened(self) -> bool:
        """Returns True if the stream is currently active and open."""
        with self._lock:
            return self._is_opened and (self._cap is not None)

    def _position_s(self) -> Optional[float]:
        """Stream timestamp of the last grabbed frame, or None if the source has no clock."""
        cap = self._cap
        if cv2 is None or cap is None or not hasattr(cap, "get"):
            return None
        try:
            pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
        except Exception:
            return None
        return pos_ms / 1000.0 if pos_ms > 0 else None

    def _capture_loop(self):
        """
        Background worker thread: grabs every packet (H.264 has to decode every frame anyway),
        holds live frames until they are due, converts the ones due at ``max_fps`` and
        reconnects with exponential backoff on stream drops.
        """
        anchor_wall: Optional[float] = None   # monotonic time at which ``anchor_pos`` plays
        anchor_pos = 0.0
        last_out_pos: Optional[float] = None  # stream time of the last converted frame

        while not self._stop_event.is_set():
            # Ensure capture is open
            if self._cap is None or not self._is_opened:
                self._reconnect()
                if self._stop_event.is_set():
                    break
                if not self._is_opened:
                    continue
                anchor_wall, last_out_pos = None, None

            grabbed = False
            try:
                with self._cap_lock:
                    if not self._stop_event.is_set() and self._cap is not None:
                        grabbed = self._cap.grab()
            except Exception as e:
                logger.debug(f"Stream error during grab on {self.name}: {e}")
                grabbed = False

            if not grabbed:
                self._on_grab_failure()
                self._stop_event.wait(0.02)
                continue

            interval = 1.0 / max(1.0, float(self.max_fps))
            pos = self._position_s() if self.realtime else None
            due = True
            if pos is not None:
                now = time.monotonic()
                if anchor_wall is None or abs(now - (anchor_wall + pos - anchor_pos)) > RESYNC_S:
                    anchor_wall, anchor_pos, last_out_pos = now, pos, None
                wait = anchor_wall + (pos - anchor_pos) - now
                if wait > 0 and self._stop_event.wait(wait):
                    break
                due = last_out_pos is None or not (0.0 <= pos - last_out_pos < interval * 0.9)

            if due:
                ret, frame = False, None
                try:
                    with self._cap_lock:
                        if not self._stop_event.is_set() and self._cap is not None:
                            ret, frame = self._cap.retrieve()
                except Exception as e:
                    logger.debug(f"Stream error during retrieve on {self.name}: {e}")
                    ret, frame = False, None

                if ret and frame is not None:
                    last_out_pos = pos
                    with self._lock:
                        self._cached_frame = frame
                        self._frame_seq += 1
                        self._consecutive_failures = 0
                        self.current_backoff = self.initial_backoff
                        self._last_frame_time = time.time()
                        self._frame_available.notify_all()
                else:
                    with self._lock:
                        self._consecutive_failures += 1

            if pos is None:
                # No stream clock to play against: pace on the wall clock instead.
                self._stop_event.wait(interval / 1.5)

    def _on_grab_failure(self) -> None:
        with self._lock:
            self._consecutive_failures += 1
            # In tests / mock captures, failure threshold is 5; for real network streams tolerate brief packet jitter
            fail_limit = 5 if not isinstance(self.source, str) else 25
            now = time.time()
            time_since_frame = (now - self._last_frame_time) if self._last_frame_time > 0 else 999.0
            dropped = (self._consecutive_failures >= fail_limit
                       and (not isinstance(self.source, str) or time_since_frame >= 1.5))
            if dropped:
                self._is_opened = False
        if not dropped:
            return
        with self._cap_lock:
            cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        logger.info(f"Stream dropped on {self.name}. Reconnecting...")

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

        # Released while backing off: don't start another (slow) open to a failing host.
        if self._stop_event.wait(self.current_backoff):
            return
        self.current_backoff = min(self.current_backoff * 1.5, self.max_backoff)

        cap = self._create_capture()
        if self._stop_event.is_set():
            if cap is not None and cap is not self.source:
                try:
                    cap.release()
                except Exception:
                    pass
            return
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
            frame = self._cached_frame.copy() if copy else self._cached_frame
            return True, self._frame_seq, frame

    def release(self):
        """
        Stops background capture thread and closes the VideoCapture.
        """
        self._stop_event.set()
        with self._frame_available:
            self._frame_available.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._thread = None

        # _cap_lock waits for an in-flight grab (bounded by the FFmpeg timeout); _lock is not
        # held meanwhile, so readers aren't blocked behind it.
        with self._cap_lock:
            cap, self._cap = self._cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        with self._lock:
            self._is_opened = False
            self._cached_frame = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
