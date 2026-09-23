# 🚦 BSW Kota Bogor — Multi-Camera Traffic Intelligence & Vehicle Tracking (MTMC)

[![Apple Neural Engine](https://img.shields.io/badge/Hardware-Apple%20Neural%20Engine%20(ANE)-555555?logo=apple&logoColor=white)](https://apple.com)
[![YOLOv8](https://img.shields.io/badge/AI%20Model-YOLOv8%20%7C%20CoreML-00FFFF)](https://ultralytics.com)
[![ByteTrack](https://img.shields.io/badge/Tracker-ByteTrack%20(MOT)-00FF66)](https://github.com/ifzhang/ByteTrack)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%20%2B%20SQLite%20WAL-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Leaflet](https://img.shields.io/badge/Frontend-Leaflet%20%2B%20TailwindCSS-199900?logo=leaflet&logoColor=white)](https://leafletjs.com)
[![Tests](https://img.shields.io/badge/Tests-278%20Passed%20(5%20Tiers)-brightgreen)](tests/)

An enterprise-grade, end-to-end intelligent transportation surveillance platform that scrapes all **94 municipal CCTV cameras** from the Kota Bogor government (`bsw.kotabogor.go.id/cctv`), executes real-time vehicle detection and multi-object tracking accelerated locally on the **Apple Neural Engine (ANE)**, and establishes persistent spatiotemporal vehicle tracking across city camera junctions (**"Follow That Car"** MTMC engine).

---

## 📸 System Previews

<div align="center">
  <img src="assets/live_tracking_cctv167.jpg" width="85%" alt="Live Tracking with IDs and ANE Latency on Camera 167" />
  <p><i>Live detection and ByteTrack ID assignment running on ANE (~11.8ms inference) at Simpang Pahlawan during nighttime rain.</i></p>
</div>

<br/>

<div align="center">
  <img src="assets/tracked_cars_trajectories.jpg" width="48%" alt="Multi-Camera Vehicle Trajectories" />
  <img src="assets/cctv_snapshot_detected.jpg" width="48%" alt="Vehicle Counting and Detection" />
  <p><i>Left: Dynamic trajectory tracking & motion vectors. Right: Multi-class vehicle classification (cars, motorcycles, angkot, trucks).</i></p>
</div>

---

## ⚡ Key Architecture & Features

### 1. Apple Neural Engine (ANE) Acceleration
- **Zero-Cloud, Ultra-Low Power**: Inference runs directly on Apple Silicon Neural Engine (`ComputeUnit.CPU_AND_NE`) via CoreML.
- **Sub-12ms Latency**: `YOLOv8m` runs in **~11.7ms – 18.5ms** (~60–85 FPS), enabling real-time multi-camera processing with negligible CPU/GPU thermal impact.
- **Strict COCO Vehicle Filtering**: Detects and classifies `car` (2), `motorcycle` (3), `bus / angkot` (5), and `truck` (7).

### 2. 94-Camera Geocoded Road Network Topology Graph
- **Complete Citywide Master Database**: All 94 cameras geocoded with precise GPS coordinates, compass azimuths, and corridor groupings.
- **220 Directed Transitions**: Modeled road adjacency graph reflecting real-world road physical constraints.
- **Strict One-Way Directional Enforcer**: Prevents impossible upstream reverse matches along Bogor's famous **Sistem Satu Arah (SSA Kebun Raya)** and **Jl. Suryakencana**.
- **Dynamic Arrival Windows $[t_{min}, t_{max}]$**: Computes spatiotemporal transit feasibility windows based on road distances, speed limits, and traffic congestion factors.

### 3. Spatiotemporal Handoff Engine ("Follow That Car")
- When a vehicle exits camera $A$ at bearing $\theta$, the engine calculates candidate downstream cameras, validates directional exit cones, and primes downstream tracking.
- Interactive **"Follow Mode"** in the dashboard automatically predicts, highlights, and switches camera feeds as the target moves across the city.

### 4. Resilient Streaming Architecture (Engineered for Safari & WebKit)
- **Decoupled Threaded Capture**: Ingests remote HLS `.m3u8` feeds using a background reader (`ThreadedStreamCapture`) with packet ring-buffering and a 5-second FFmpeg network timeout (`timeout;5000000`), completely shielding the server from remote restreamer stalls.
- **WebKit-Compliant MJPEG Stream**: Injects proper `Content-Length` headers per multipart frame boundary, eliminating Safari image stalls and WebKit memory leaks.
- **Adaptive Sequential Frame Streaming**: Frontend chains frame requests sequentially (`onload` $\rightarrow$ next frame), ensuring strictly 1 in-flight frame per tab and zero connection piling.
- **Standby Placeholder HUD**: Automatically renders a modern standby graphic (`[CCTV OFFLINE / CONNECTING STREAM...]`) with `200 OK` for buffering feeds, preventing broken image icons or 503 error storms.

### 5. Persistent SQLite WAL Vehicle Registry
- Logs intra-camera sightings, track histories, speed estimates, and exit angles.
- Reconstructs complete multi-hop chronological vehicle journeys across Kota Bogor.

---

## 📂 Project Structure

```
bogor_mtmc_traffic_dashboard/
├── backend/
│   ├── server.py              # FastAPI server, MJPEG streaming, WebSocket & REST APIs
│   ├── detector.py            # CoreML YOLOv8 + ByteTrack ANE inference wrapper
│   ├── stream_capture.py      # Threaded low-latency HLS stream reader & packet flusher
│   ├── topology_engine.py     # 94-camera road graph, arrival windows, SSA one-way logic
│   ├── database.py            # SQLite WAL vehicle registry & journey query service
│   └── indexing_daemon.py     # Background multi-camera ingestion & tracking daemon
├── frontend/
│   ├── index.html             # Tactical dark-mode dashboard (Tailwind CSS + Leaflet)
│   └── app.js                 # Map controller, adaptive frame stream, and handoff UI
├── data/
│   ├── bogor_camera_topology.json  # 94 geocoded nodes, corridors, and directed edges
│   ├── cctv_streams.json      # Master stream URLs, embed URLs, and metadata
│   ├── cctv_streams.csv       # Tabular camera directory
│   └── traffic_db.sqlite      # SQLite WAL database for vehicle sightings & journeys
├── scripts/
│   ├── scrape_cctv_streams.py # Scraper for all 94 BSW CCTV stream endpoints
│   ├── cctv_ane_counter.py    # Standalone ANE vehicle counter & benchmark tool
│   └── car_tracker.py         # Standalone ByteTrack trajectory visualizer
├── assets/                    # Screenshots, trajectory visualizers, and diagrams
├── tests/                     # 278-test verification suite (Tiers 1 to 5)
├── requirements.txt           # Python dependencies
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
# Clone the repository
git clone https://github.com/Subakiz/bogor_mtmc_traffic_dashboard.git
cd bogor_mtmc_traffic_dashboard

# Create and activate virtual environment (Python 3.11+ recommended)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare CoreML Model (Apple Neural Engine)

Export the YOLOv8 model to CoreML format targeting the Apple Neural Engine:

```bash
# Export YOLOv8m to CoreML format (.mlpackage)
yolo export model=yolov8m.pt format=coreml
```

*Note: The system automatically looks for `yolov8m.mlpackage` in the project root or `/Users/nabils/cctv_yolo_ane/yolov8m.mlpackage`.*

### 3. Launch the Dashboard

To run the server in the background persistently:

```bash
# Start FastAPI server with Uvicorn
nohup uvicorn backend.server:app --host 127.0.0.1 --port 8000 > server.log 2>&1 &

# Or run interactively in the terminal
uvicorn backend.server:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser and navigate to:
👉 **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 📡 REST API Reference

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/` | `GET` | Serves the interactive MTMC Traffic Dashboard web application |
| `/api/cameras` | `GET` | Lists all 94 BSW CCTV cameras with GPS coordinates, corridor, and stream URLs |
| `/api/cameras/{id}` | `GET` | Retrieves detailed metadata and junction info for a specific camera |
| `/api/topology` | `GET` | Returns camera nodes, directed road graph edges, and corridor groupings |
| `/api/stream/{id}` | `GET` | Real-time MJPEG live stream with bounding boxes, labels, and persistent ByteTrack IDs |
| `/api/stream/{id}/frame`| `GET` | Returns latest single annotated JPEG frame (used for adaptive client streaming) |
| `/api/handoff/predict` | `POST` | Predicts downstream candidate cameras and arrival windows based on exit heading |
| `/api/vehicles/{id}/timeline`| `GET`| Reconstructs multi-camera journey timeline and sighting history for a vehicle |
| `/api/recent` | `GET` | Fetches recent verified vehicle sightings across Kota Bogor |
| `/api/stats` | `GET` | Returns citywide traffic stats (unique vehicle counts, classification breakdown) |
| `/ws/live` | `WebSocket` | Live telemetry feed broadcasting vehicle detections and handoff events |

---

## 🧪 Comprehensive Verification Suite

The repository includes a 5-tier test suite covering 278 test assertions:

```bash
# Run the complete test suite
python tests/test_runner.py
```

- **Tier 1 (Core Features)**: Camera geocoding, ANE model inference, ByteTrack ID assignment, SQLite WAL persistence, FastAPI endpoints.
- **Tier 2 (Boundary Conditions)**: Zero-detection frames, missing stream URLs, edge road topologies, arrival window bounds $[t_{min}, t_{max}]$.
- **Tier 3 (Pairwise Combinations)**: Multi-camera transitions across various vehicle classes and weather/lighting conditions.
- **Tier 4 (Citywide Scenarios)**: Simulated journeys across SSA Ring Road, Pajajaran corridor, and Batutulis junction.
- **Tier 5 (Adversarial Stress)**: Graph traversal resilience, spatiotemporal jitter, network stream dropouts, and rapid camera switching.

---

## 🛠️ Standalone CLI Utilities

Inside `scripts/`:

```bash
# 1. Re-scrape latest live CCTV stream URLs from BSW Kota Bogor
python scripts/scrape_cctv_streams.py

# 2. Run vehicle counting benchmark on Camera 167 with GUI display
python scripts/cctv_ane_counter.py --display --conf 0.25

# 3. Run vehicle tracking with persistent trajectory trails
python scripts/car_tracker.py --camera-id 167 --trail-length 30
```

---

## 📜 License & Credits

- **Data Source**: CCTV Feeds and infrastructure provided by **Bina Sarana Wilayah (BSW) Pemerintah Kota Bogor** (`bsw.kotabogor.go.id`).
- **AI Frameworks**: [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), [Apple CoreML Tools](https://github.com/apple/coremltools), [ByteTrack](https://github.com/ifzhang/ByteTrack).
- **License**: MIT License. Built for municipal traffic intelligence research and edge AI benchmarking.
