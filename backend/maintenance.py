"""
backend/maintenance.py
Background housekeeping for the traffic database.

Every vehicle sighting used to store ~1.5 KB of appearance blobs that nothing reads back after the
vehicle has left re-identification range (matching uses in-memory state). ``SightingMaintenance``
runs one pass every ``interval_s``:

1. blanks the blobs of sightings older than ``embedding_ttl_s`` (rows are kept);
2. marks followed vehicles ('tracking' active targets) that have not been updated for
   ``active_target_ttl_s`` as 'expired', so they stop boosting cameras;
3. opt-in retention: deletes sightings and handoff events older than ``retention_days`` in small
   batches (all-time totals and ``observation_windows`` are untouched);
4. hands freed pages back to the filesystem (``incremental_vacuum``; effective once
   ``Database.compact()`` has converted the file) and refreshes planner statistics.

Each step is independent: one failing does not stop the others.
"""

import logging
import threading
from typing import Any, Dict, Optional

logger = logging.getLogger("maintenance")

SECONDS_PER_DAY = 86400.0


class SightingMaintenance:
    def __init__(self, db: Any, embedding_ttl_s: float, interval_s: float = 300.0,
                 first_run_delay_s: float = 30.0, batch: int = 2000, pause_s: float = 0.05,
                 retention_days: Optional[float] = None, active_target_ttl_s: Optional[float] = None,
                 vacuum_pages_per_run: int = 20000, vacuum_chunk_pages: int = 500):
        if retention_days is None or active_target_ttl_s is None:
            from backend.config import ACTIVE_TARGET_TTL_S, SIGHTING_RETENTION_DAYS
            retention_days = SIGHTING_RETENTION_DAYS if retention_days is None else retention_days
            active_target_ttl_s = ACTIVE_TARGET_TTL_S if active_target_ttl_s is None else active_target_ttl_s
        self.db = db
        self.embedding_ttl_s = float(embedding_ttl_s)
        self.interval_s = max(1.0, float(interval_s))
        self.first_run_delay_s = max(0.0, float(first_run_delay_s))
        self.batch = batch
        self.pause_s = pause_s
        self.retention_days = max(0.0, float(retention_days))
        self.active_target_ttl_s = max(0.0, float(active_target_ttl_s))
        self.vacuum_pages_per_run = max(0, int(vacuum_pages_per_run))
        self.vacuum_chunk_pages = max(1, int(vacuum_chunk_pages))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.total_blanked = 0
        self.total_pruned = 0
        self.total_handoffs_pruned = 0
        self.total_targets_expired = 0
        self.total_pages_freed = 0
        self.last_run: Dict[str, Any] = {}

    # ── one pass ─────────────────────────────────────────────────────────────────

    def run_once(self) -> int:
        """One housekeeping pass; returns the number of rows whose blobs were blanked
        (``last_run`` holds the full breakdown)."""
        stats: Dict[str, Any] = {"blanked": 0, "expired_targets": 0, "pruned_sightings": 0,
                                 "pruned_handoffs": 0, "pages_freed": 0}

        if self.embedding_ttl_s > 0:
            stats["blanked"] = self._step("blank embeddings", lambda: self.db.expire_embeddings(
                self.embedding_ttl_s, batch=self.batch, pause_s=self.pause_s, should_stop=self._stop.is_set))

        if self.active_target_ttl_s > 0 and not self._stop.is_set():
            stats["expired_targets"] = self._step("expire active targets", lambda: self.db.expire_active_targets(
                self.active_target_ttl_s))

        if self.retention_days > 0 and not self._stop.is_set():
            age_s = self.retention_days * SECONDS_PER_DAY
            stats["pruned_sightings"] = self._step("prune sightings", lambda: self.db.prune_sightings(
                age_s, batch=self.batch, pause_s=self.pause_s, should_stop=self._stop.is_set))
            stats["pruned_handoffs"] = self._step("prune handoff events", lambda: self.db.prune_handoff_events(
                age_s, batch=self.batch, pause_s=self.pause_s, should_stop=self._stop.is_set))

        if not self._stop.is_set():
            stats["pages_freed"] = self._step("incremental vacuum", self._vacuum)

        try:
            self.db.optimize()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"PRAGMA optimize failed: {exc}")

        self.total_blanked += stats["blanked"]
        self.total_targets_expired += stats["expired_targets"]
        self.total_pruned += stats["pruned_sightings"]
        self.total_handoffs_pruned += stats["pruned_handoffs"]
        self.total_pages_freed += stats["pages_freed"]
        self.last_run = stats
        if stats["blanked"]:
            logger.info(f"Blanked appearance blobs of {stats['blanked']} sightings older than {self.embedding_ttl_s:.0f}s.")
        if stats["expired_targets"]:
            logger.info(f"Expired {stats['expired_targets']} stale active target(s).")
        if stats["pruned_sightings"] or stats["pruned_handoffs"]:
            logger.info(f"Retention ({self.retention_days:g} d): deleted {stats['pruned_sightings']} sightings and "
                        f"{stats['pruned_handoffs']} handoff events; returned {stats['pages_freed']} pages.")
        return stats["blanked"]

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
