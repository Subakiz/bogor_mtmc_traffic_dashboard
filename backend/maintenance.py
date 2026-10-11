"""
backend/maintenance.py
Background housekeeping for the sightings table.

Every vehicle sighting stores ~1.5 KB of appearance blobs that nothing reads back after the vehicle
has left re-identification range (matching uses in-memory state), so left alone the table grows by
roughly a gigabyte per day of tracking. ``SightingMaintenance`` blanks those blobs once they are
older than ``embedding_ttl_s``; the sighting rows themselves (class, camera, time) are kept.
"""

import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("maintenance")


class SightingMaintenance:
    def __init__(self, db: Any, embedding_ttl_s: float, interval_s: float = 300.0,
                 first_run_delay_s: float = 30.0, batch: int = 2000, pause_s: float = 0.05):
        self.db = db
        self.embedding_ttl_s = float(embedding_ttl_s)
        self.interval_s = max(1.0, float(interval_s))
        self.first_run_delay_s = max(0.0, float(first_run_delay_s))
        self.batch = batch
        self.pause_s = pause_s
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.total_blanked = 0

    def run_once(self) -> int:
        """One expiry pass; returns the number of rows whose blobs were blanked."""
        if self.embedding_ttl_s <= 0:
            return 0
        n = self.db.expire_embeddings(self.embedding_ttl_s, batch=self.batch, pause_s=self.pause_s,
                                      should_stop=self._stop.is_set)
        self.total_blanked += n
        if n:
            logger.info(f"Blanked appearance blobs of {n} sightings older than {self.embedding_ttl_s:.0f}s.")
        return n

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="SightingMaintenance", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _run(self) -> None:
        delay = self.first_run_delay_s
        while not self._stop.wait(delay):
            try:
                self.run_once()
            except Exception as exc:  # noqa: BLE001 - housekeeping must never take the server down
                logger.warning(f"Sighting maintenance failed: {exc}")
            delay = self.interval_s
