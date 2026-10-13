"""
backend/maintenance.py
Background housekeeping for the traffic database.

``SightingMaintenance`` runs one pass every ``interval_s``:

1. opt-in retention: deletes sightings older than ``retention_days`` in small batches (all-time
   totals and ``observation_windows`` are untouched);
2. hands freed pages back to the filesystem (``incremental_vacuum``; effective once
   ``Database.compact()`` has converted the file) and refreshes planner statistics.

Each step is independent: one failing does not stop the others.
"""

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("maintenance")

SECONDS_PER_DAY = 86400.0


class SightingMaintenance:
    def __init__(self, db: Any, interval_s: float = 300.0,
                 first_run_delay_s: float = 30.0, batch: int = 2000, pause_s: float = 0.05,
                 retention_days: Optional[float] = None,
                 vacuum_pages_per_run: int = 20000, vacuum_chunk_pages: int = 500):
        if retention_days is None:
            from backend.config import SIGHTING_RETENTION_DAYS
            retention_days = SIGHTING_RETENTION_DAYS
        self.db = db
        self.interval_s = max(1.0, float(interval_s))
        self.first_run_delay_s = max(0.0, float(first_run_delay_s))
        self.batch = batch
        self.pause_s = pause_s
        self.retention_days = max(0.0, float(retention_days))
        self.vacuum_pages_per_run = max(0, int(vacuum_pages_per_run))
        self.vacuum_chunk_pages = max(1, int(vacuum_chunk_pages))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.total_pruned = 0
        self.total_pages_freed = 0
        self.last_run: Dict[str, Any] = {}

    # ── one pass ─────────────────────────────────────────────────────────────────

    def run_once(self) -> int:
        """One housekeeping pass; returns the number of sightings deleted by retention
        (``last_run`` holds the full breakdown)."""
        stats: Dict[str, Any] = {"pruned_sightings": 0, "pages_freed": 0}

        if self.retention_days > 0 and not self._stop.is_set():
            age_s = self.retention_days * SECONDS_PER_DAY
            stats["pruned_sightings"] = self._step("prune sightings", lambda: self.db.prune_sightings(
                age_s, batch=self.batch, pause_s=self.pause_s, should_stop=self._stop.is_set))

        if not self._stop.is_set():
            stats["pages_freed"] = self._step("incremental vacuum", self._vacuum)

        try:
            self.db.optimize()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"PRAGMA optimize failed: {exc}")

        self.total_pruned += stats["pruned_sightings"]
        self.total_pages_freed += stats["pages_freed"]
        self.last_run = stats
        if stats["pruned_sightings"]:
            logger.info(f"Retention ({self.retention_days:g} d): deleted {stats['pruned_sightings']} sightings; "
                        f"returned {stats['pages_freed']} pages.")
        return stats["pruned_sightings"]

    def _step(self, name: str, fn) -> int:
        try:
            return int(fn() or 0)
        except Exception as exc:  # noqa: BLE001 - one failing step must not block the others
            logger.warning(f"Maintenance step '{name}' failed: {exc}")
            return 0

    def _vacuum(self) -> int:
        """Returns freed pages to the file in short write-locked chunks, up to a per-run budget."""
        freed = 0
        while freed < self.vacuum_pages_per_run and not self._stop.is_set():
            n = self.db.incremental_vacuum(min(self.vacuum_chunk_pages, self.vacuum_pages_per_run - freed))
            if n <= 0:
                break
            freed += n
            if self.pause_s > 0:
                self._stop.wait(self.pause_s)
        return freed

    # ── thread ───────────────────────────────────────────────────────────────────

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
