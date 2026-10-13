# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BSW Kota Bogor multi-camera traffic intelligence dashboard: YOLOv8 (CoreML on the Apple Neural Engine) + ByteTrack over the city's municipal HLS CCTV feeds, citywide vehicle counting, spatiotemporal "Follow That Car" re-identification across cameras, FastAPI backend + Leaflet frontend. Engine (`car_tracker/`) and dashboard (`backend/`, `frontend/`) live in this one repo.

**Stale docs:** `TEST_INFRA.md`, `TEST_READY.md` (and `HANDOFF.md` / the gitignored `.agents/`, if present locally) are old notes written for a previous two-folder layout (`teamwork_projects/...`). Do not trust their paths, test counts, or the license-plate (LPR/TNKB) rules — **the plate-recognition feature was intentionally removed** (backend endpoints, overlays and frontend included); vehicles are labelled by class only, and no track/vehicle ids are drawn on video frames.

## Commands

Always use the project venv (Python 3.11, macOS/Apple Silicon required for CoreML):

```bash
./start.sh                                   # run server at http://127.0.0.1:8000 (kills whatever holds :8000)
.venv/bin/python3.11 -m pytest -q            # full suite (testpaths=tests, offline, no network)
.venv/bin/python3.11 -m pytest tests/test_rolling_tracker.py -q
.venv/bin/python3.11 -m pytest tests/test_rolling_tracker.py::test_name -q   # single test
.venv/bin/python3.11 scrape_cctv_streams.py  # re-scrape bsw.kotabogor.go.id; syncs topology JSON + DB
```

`scripts/start_daemon.sh` / `stop_daemon.sh` run uvicorn under `caffeinate`. `tests/tier1_features` … `tier5_adversarial` are collected by plain pytest too; `tests/test_runner.py` is a legacy tiered CLI runner for them, use pytest instead. `test_f_arch_02` (latency) is timing-flaky under load.

## Architecture

**Request/inference flow.** `backend/server.py` (FastAPI, ~770 lines) is a thin layer: REST + `/ws/live` + frame endpoints. All inference is done by `backend/rolling_tracker.py` (`RollingTracker`, a single inference worker thread) — the server only renders annotated frames (`_render_annotated_frame`) from what the tracker hands back. `backend/indexing_daemon.py` (`IndexingDaemon`) is **not used by the server any more** (class and tests kept); `/api/daemon/*` controls the rolling tracker.

**Rolling tracker model.**
- *Focus cameras*: cameras currently being viewed (`set_focus` is called on every frame-endpoint poll; lapses after `FOCUS_TTL_S`), up to `MAX_FOCUS_CAMERAS`, run at `FOCUS_FPS`.
- *Rolling visits*: remaining stream slots (`MAX_OPEN_STREAMS`, default 12, at most `MAX_STREAMS_PER_HOST`=3 per restreamer host — the 96 cameras sit on 6 hosts, and an overloaded host refuses connections and gets backed off wholesale) visit every other camera for a dwell time by tier (tier1 = downstream of focus / follow-target candidates, tier2 = arterial hubs, tier3 = LRU) at `ROLLING_FPS`, so vehicles are counted citywide. Tunables are env-overridable in `backend/config.py`; env names differ from the Python names (`ROLLING_MAX_STREAMS` → `MAX_OPEN_STREAMS`, `ROLLING_FOCUS_FPS` → `FOCUS_FPS`, `ROLLING_ENABLED=0` → focus only).
- `StreamPool` opens/releases HLS captures off the inference thread (`backend/stream_capture.py`, `ThreadedStreamCapture`). Never open streams on the inference thread. Dead cameras/hosts get escalating jittered backoff.
- A track becomes one `vehicle_sightings` row after `MIN_TRACK_HITS` frames; parked vehicles are re-matched by IoU+appearance on the next visit so they aren't recounted.

**Congestion index (`car_tracker/analytics.py`).** Judged only on the *moving stream*: each track is "undetermined" (new, hasn't moved), "static" (never moved since first seen after `static_after_s`=6 s → parked) or "traffic" (has moved ≥0.6 body sizes). Parked/undetermined vehicles are excluded from density, queue share, speed and window volume — counting them made a lot of parked cars read as a permanent jam (camera 167). All distances/speeds are in *body sizes* (box size) per second, so they are comparable near/far and across cameras; never reintroduce fixed px/s thresholds. Index = 0.40·queue (share of traffic standing still, full at 50%) + 0.45·slowness (median speed vs the camera's *own* learned free-flow speed = p85 of its traffic speeds, persisted in `data/free_flow_speeds.json`; until learned only a full standstill counts) + 0.15·density·slowness (union-of-boxes coverage, counted only when slow), damped quadratically below 6 measurable vehicles. Live value = mean of the last 20 s of per-frame values; window value = mean over the window. Labels (backend `get_congestion_label`, mirrored in `app.js`): <0.25 Smooth, <0.55 Moderate, <0.8 Heavy, else Congested. Known limits: a queue standing still for a whole 12–20 s rolling visit looks parked; red-light stops read as queueing; analytics window `total_volume` still counts track ids (inflated by track fragmentation) — counting-line rates are the reliable volume.

**Analytics volume = counting-line crossings.** The rolling tracker marks each detection record `counted_now` (that track crossed the camera's counting line in this frame); when the key is present, the engine builds window volume/class volume from those crossings, so `camera_traffic_timeseries.total_volume` matches `observation_windows` (the older track-id volume was inflated ~10× by track fragmentation). Rows carry `volume_basis` (`crossings` | `tracks`; the live DB's track-based rows were purged on 2026-10-11, so it only holds `crossings`) — never mix the two in one series. Callers that don't send `counted_now` (tests, standalone CLI) keep track-based volume. The Analytics tab reads `/api/cameras/{id}/metrics` → `flow`: hourly rate from `db.get_camera_rates`, per-bucket series from `db.get_camera_flow_series` (epoch-aligned buckets, unobserved buckets omitted → chart line breaks) joined with `analytics.get_congestion_series` (length-weighted). Params `minutes` (≤1440), `bucket_minutes`.

**Throughput ceiling.** The single worker spends ~86% of its time in the ANE `detect()` call (~20 ms under load, 13 ms idle), so total throughput is ~40 fps regardless of slot count; going from 6 to 12 slots raised observed camera-seconds per wall-second from ~4.2 to ~6.1, beyond that (tested 18–20) there is no gain and restreamers start failing. More coverage needs cheaper inference (smaller model for rolling visits, batched CoreML export), not more streams.

**Engine (`car_tracker/`)**: `ANEVehicleDetector.detect(frame, camera_id=...)` does per-camera ByteTrack state (`detector.reset_tracks(cid)` at visit end); `analytics.py` is `TrafficAnalyticsEngine` (time-series windows → `data/traffic_timeseries.db`). `backend/detector.py` is the shared ANE model loader. `backend/config.py` `ENGINE_ROOT` defaults to the project root (override `CCTV_YOLO_ANE_ROOT`) and `ensure_engine_importable()` puts it on `sys.path`.

**Volume vs sightings (two separate pipelines).** *Traffic volume* comes from `backend/counting.py`: a learned counting line per camera (`data/counting_lines.json`; "passage" mode until 12 trajectories), crossings recorded in `observation_windows` with observed seconds, rates via `db.get_camera_rates()` → `/api/cameras/counts` (`camera_rates`, `citywide_vph` — one camera per junction cluster). Rates from <60 s observation are flagged `low_sample`. *Sightings* drive journeys / Re-ID, not volume.

**Re-ID / handoff.** `backend/topology_engine.py` loads `data/bogor_camera_topology.json` (camera nodes + directed edges, one-way enforcement, arrival windows [t_min, t_max]) and predicts downstream candidates; `backend/reid.py` `IdentityResolver` assigns global IDs and is wired into `RollingTracker` (and `IndexingDaemon.step_camera`). Frontend calls `POST /api/handoff/predict` on camera load/switch.

**Persistence.** `backend/database.py`: SQLite WAL at `data/traffic_db.sqlite`, gitignored and created on startup (schema + cameras seeded from the topology file); `counting_lines.json` / `free_flow_speeds.json` are likewise runtime state rebuilt from traffic. **Every write must go through `database._transaction()`** (commit/rollback) and never be wrapped in a bare `except: pass` — a swallowed FK error previously left a transaction (write lock) open forever and froze the tracker. Debug hangs by dumping `sys._current_frames()` in-process (py-spy absent; macOS `sample` shows no Python frames).

**Dashboard stats path (keep it off the event loop and off wide tables).** `vehicle_sightings` rows are wide (appearance blobs), so aggregates read from narrow covering indexes (`idx_sightings_agg`, `idx_sightings_local_track`, `idx_sightings_global_id`) — keep each `DISTINCT` in its own query or SQLite falls back to scanning the table. Dashboard aggregates go through `server._stats_cache` (`_TTLCache`, `STATS_CACHE_TTL_S`=3 s, one computation shared by all tabs; cached values are shared, never mutate them). The `/ws/live` heartbeat builds its payload in `asyncio.to_thread` and carries `counts` + `recent`, so the frontend applies them directly (`applyStats`) instead of refetching. `backend/maintenance.py` blanks appearance blobs older than `EMBEDDING_TTL_S` (1 h; ReID matches in-memory state, nothing reads blobs back) via `db.expire_embeddings` (small batches, persisted watermark in `maintenance_state`); rows are kept. Without it the table grew ~1 GB/day.

**Frontend** (`frontend/index.html`, `app.js`, no build step): Leaflet map + HUD. Uses sequential single-frame polling of `/api/stream/{id}/frame` (one in-flight request per tab; `onload` → next) rather than MJPEG/WebSocket video, because MJPEG stalls in Safari/WebKit.

## Gotchas

- CoreML models are not thread-safe for concurrent forward passes; inference goes through the single detector path/lock — don't add new threads that call the model directly.
- `cctv_streams.json` at root and `data/cctv_streams.{json,csv}` are scrape outputs; the dashboard reads the topology JSON + DB, not `cctv_streams.json`. Camera/edge totals in tests are derived from the topology file (`tests/__init__.py`), not hardcoded.
- Model weights (`*.pt`, `*.mlpackage`) are gitignored; export with `yolo export model=yolov8m.pt format=coreml`.
- Streams come from municipal restreamers with connection limits and flaky cameras; keep FFmpeg/OpenCV quieted (`OPENCV_LOG_LEVEL=ERROR`, `OPENCV_FFMPEG_LOGLEVEL=8`, set in `start.sh`) and `ulimit -n` raised to avoid Errno 24.
