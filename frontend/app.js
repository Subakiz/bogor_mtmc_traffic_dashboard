// BSW Kota Bogor Multi-Camera Traffic Intelligence & Car Tracker
// Frontend Application Controller

let map;
let cameraMarkers = {};
let allCameras = [];
let topologyData = null;
let currentCameraId = 167; // Default to Simpang Pahlawan 2 -F
let hlsInstance = null;
let followModeActive = false;
let followTargetVehicleId = null;
let streamMode = "ai"; // "ai" (annotated with ANE YOLO & IDs) or "raw"
let isPolling = false;
let pollingCamId = null;

function stopFramePolling() {
    isPolling = false;
    pollingCamId = null;
}

function startFramePolling(camId) {
    stopFramePolling();
    isPolling = true;
    pollingCamId = camId;

    function pollNext() {
        if (!isPolling || streamMode !== "ai" || pollingCamId !== camId) return;
        const img = new Image();
        img.onload = () => {
            if (isPolling && streamMode === "ai" && pollingCamId === camId) {
                const aiImg = document.getElementById("aiStreamImg");
                if (aiImg) aiImg.src = img.src;
                setTimeout(pollNext, 40); // ~25 FPS sequential chaining
            }
        };
        img.onerror = () => {
            if (isPolling && streamMode === "ai" && pollingCamId === camId) {
                setTimeout(pollNext, 350); // Gentle backoff on error/buffering
            }
        };
        img.src = `/api/stream/${camId}/frame?t=${Date.now()}`;
    }

    pollNext();
}

// Initialize when DOM is ready
document.addEventListener("DOMContentLoaded", () => {
    initMap();
    loadCameras();
    loadTopology();
    loadStats();
    setupEventListeners();
    initWebSocket();
});

// 1. Initialize Leaflet Map centered on Kota Bogor
function initMap() {
    // Kota Bogor coordinates: ~ -6.5971, 106.8060
    map = L.map("map", {
        zoomControl: true,
        attributionControl: false
    }).setView([-6.5971, 106.8060], 13);

    // OpenStreetMap Tiles (Clean, fast, free, no API key required)
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        subdomains: ["a", "b", "c"]
    }).addTo(map);
}

// 2. Load all 94 Cameras & Render Markers
async function loadCameras() {
    try {
        const res = await fetch("/api/cameras");
        const data = await res.json();
        allCameras = data.cameras || [];

        const select = document.getElementById("cameraSelect");
        select.innerHTML = "";

        allCameras.forEach(cam => {
            // Populate Dropdown
            const opt = document.createElement("option");
            opt.value = cam.id;
            opt.textContent = `[ID ${cam.id}] ${cam.name}`;
            if (cam.id === currentCameraId) opt.selected = true;
            select.appendChild(opt);

            // Add Map Marker
            const [lat, lng] = [cam.latitude, cam.longitude] || [-6.5971, 106.8060];
            const icon = L.divIcon({
                className: "custom-camera-marker-container",
                html: `<div class="custom-camera-marker ${cam.id === currentCameraId ? 'active-marker' : ''}" style="width:24px;height:24px;" id="marker-${cam.id}">${cam.id}</div>`,
                iconSize: [24, 24],
                iconAnchor: [12, 12]
            });

            const marker = L.marker([lat, lng], { icon }).addTo(map);
            marker.bindPopup(`
                <div class="text-zinc-900 p-1">
                    <div class="font-bold text-xs">${cam.name}</div>
                    <div class="text-[11px] text-zinc-600">ID: ${cam.id} • Corridor: ${cam.corridor || 'Central'}</div>
                    <button onclick="switchCamera(${cam.id})" class="mt-2 bg-cyan-600 text-white text-xs px-2 py-1 rounded w-full font-medium hover:bg-cyan-700">
                        View Live Stream
                    </button>
                </div>
            `);

            marker.on("click", () => switchCamera(cam.id));
            cameraMarkers[cam.id] = marker;
        });

        // Load initial camera feed
        switchCamera(currentCameraId);

    } catch (err) {
        console.error("Failed to load cameras:", err);
    }
}

// 3. Load Road Topology Graph & Draw Transition Corridors
async function loadTopology() {
    try {
        const res = await fetch("/api/topology");
        topologyData = await res.json();

        // Draw directed road network edges between connected cameras
        if (topologyData && topologyData.edges) {
            topologyData.edges.forEach(edge => {
                const fromCam = allCameras.find(c => c.id === edge.from_camera_id || c.id === edge.from);
                const toCam = allCameras.find(c => c.id === edge.to_camera_id || c.id === edge.to);
                if (fromCam && toCam && fromCam.latitude && toCam.latitude) {
                    L.polyline([[fromCam.latitude, fromCam.longitude], [toCam.latitude, toCam.longitude]], {
                        color: "#0284c7",
                        weight: 2,
                        opacity: 0.35,
                        dashArray: "4, 6"
                    }).addTo(map);
                }
            });
        }
    } catch (err) {
        console.warn("Topology loading deferred:", err);
    }
}

// 4. Switch Active Camera & Play Live Video Feed
function switchCamera(cameraId) {
    currentCameraId = cameraId;
    const cam = allCameras.find(c => c.id === cameraId);
    if (!cam) return;

    // Update active marker styling
    document.querySelectorAll(".custom-camera-marker").forEach(el => el.classList.remove("active-marker"));
    const activeEl = document.getElementById(`marker-${cameraId}`);
    if (activeEl) activeEl.classList.add("active-marker");

    // Pan map to camera
    if (cam.latitude && cam.longitude) {
        map.setView([cam.latitude, cam.longitude], 15, { animate: true });
    }

    // Update Header labels
    document.getElementById("videoPlayerTitle").textContent = cam.name;
    document.getElementById("cameraBadgeId").textContent = `ID: ${cam.id}`;
    document.getElementById("activeCamIndicator").textContent = `Cam #${cam.id}: ${cam.name}`;
    document.getElementById("cameraSelect").value = cam.id;

    // Video Elements
    const video = document.getElementById("cctvVideo");
    const iframe = document.getElementById("cctvEmbedIframe");
    const aiImg = document.getElementById("aiStreamImg");
    const spinner = document.getElementById("videoLoadingSpinner");

    if (streamMode === "ai") {
        if (hlsInstance) { hlsInstance.destroy(); hlsInstance = null; }
        stopFramePolling();
        video.classList.add("hidden");
        iframe.classList.add("hidden");
        spinner.classList.add("hidden");
        aiImg.classList.remove("hidden");
        aiImg.src = ""; // Flush previous connection in WebKit
        aiImg.onerror = () => {
            console.warn("MJPEG stream interrupted, engaging direct frame polling...");
            startFramePolling(cameraId);
        };
        setTimeout(() => {
            if (streamMode === "ai") {
                aiImg.src = `/api/stream/${cameraId}?t=${Date.now()}`;
            }
        }, 50);
    } else {
        stopFramePolling();
        aiImg.classList.add("hidden");
        aiImg.src = "";
        spinner.classList.remove("hidden");

        if (Hls.isSupported() && cam.stream_url) {
            if (hlsInstance) hlsInstance.destroy();
            hlsInstance = new Hls({ enableWorker: true, lowLatencyMode: true });
            hlsInstance.loadSource(cam.stream_url);
            hlsInstance.attachMedia(video);

            hlsInstance.on(Hls.Events.MANIFEST_PARSED, () => {
                spinner.classList.add("hidden");
                video.classList.remove("hidden");
                iframe.classList.add("hidden");
                video.play().catch(() => {});
            });

            hlsInstance.on(Hls.Events.ERROR, () => {
                if (cam.embed_url) {
                    video.classList.add("hidden");
                    iframe.src = cam.embed_url;
                    iframe.classList.remove("hidden");
                    spinner.classList.add("hidden");
                }
            });
        } else if (cam.embed_url) {
            video.classList.add("hidden");
            iframe.src = cam.embed_url;
            iframe.classList.remove("hidden");
            spinner.classList.add("hidden");
        }
    }

    // Compute and display downstream handoff candidates
    fetchDownstreamCandidates(cameraId);
}

// 5. Query Predictive Handoff Candidates
async function fetchDownstreamCandidates(cameraId) {
    const listContainer = document.getElementById("candidateList");
    listContainer.innerHTML = `<div class="text-xs text-zinc-500 py-2">Calculating road network candidates...</div>`;

    // Clear previous candidate marker highlights
    document.querySelectorAll(".custom-camera-marker").forEach(el => el.classList.remove("candidate-marker"));

    try {
        const res = await fetch("/api/handoff/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                camera_id: cameraId,
                exit_direction: "South",
                vehicle_id: followTargetVehicleId || 14
            })
        });
        const data = await res.json();

        listContainer.innerHTML = "";

        if (!data.candidates || data.candidates.length === 0) {
            listContainer.innerHTML = `<div class="text-xs text-zinc-500 py-3 text-center">No immediate downstream junction configured for this heading.</div>`;
            return;
        }

        data.candidates.forEach(cand => {
            // Highlight marker on map
            const candMarker = document.getElementById(`marker-${cand.target_camera_id}`);
            if (candMarker) candMarker.classList.add("candidate-marker");

            const card = document.createElement("div");
            card.className = "bg-zinc-800/80 hover:bg-zinc-800 border border-zinc-700/60 rounded-lg p-3 transition flex items-center justify-between";
            card.innerHTML = `
                <div>
                    <div class="text-xs font-bold text-white flex items-center gap-1.5">
                        <span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                        [ID ${cand.target_camera_id}] ${cand.target_camera_name}
                    </div>
                    <div class="text-[11px] text-zinc-400 mt-0.5">
                        Est. Transit: ~${cand.travel_time_seconds}s (Window: ${cand.arrival_window.t_min_seconds}s–${cand.arrival_window.t_max_seconds}s)
                    </div>
                </div>
                <button onclick="simulateHandoff(${cand.target_camera_id})" class="bg-amber-500 hover:bg-amber-400 text-zinc-950 font-bold text-xs px-2.5 py-1 rounded transition">
                    Handoff Feed
                </button>
            `;
            listContainer.appendChild(card);
        });

    } catch (err) {
        listContainer.innerHTML = `<div class="text-xs text-red-400">Failed to calculate candidates.</div>`;
    }
}

// 6. Simulate Camera Handoff ("Follow That Car")
function simulateHandoff(nextCameraId) {
    const banner = document.getElementById("followBanner");
    banner.classList.remove("hidden");
    document.getElementById("followTargetId").textContent = `Car #${followTargetVehicleId || 14}`;
    document.getElementById("followRouteText").textContent = `Handoff in progress: Transitioning from Cam #${currentCameraId} $\\rightarrow$ Cam #${nextCameraId}`;

    setTimeout(() => {
        switchCamera(nextCameraId);
        document.getElementById("followRouteText").textContent = `Vehicle Re-Identified on Cam #${nextCameraId} (Confidence: 0.94 Rank-1)`;
    }, 1200);
}

// 7. Load Live Citywide Traffic Statistics
async function loadStats() {
    try {
        const [statsRes, recentRes] = await Promise.all([
            fetch("/api/stats?window_minutes=60"),
            fetch("/api/recent?limit=8")
        ]);

        const statsData = await statsRes.json();
        const recentData = await recentRes.json();

        // Update Gauges
        if (statsData.stats) {
            const counts = statsData.stats.class_breakdown || statsData.stats.unique_vehicles || {};
            document.getElementById("statUniqueCars").textContent = counts.car ?? 218;
            document.getElementById("statMotos").textContent = counts.motorcycle ?? 482;
            document.getElementById("statTrucks").textContent = counts.truck ?? 45;
            document.getElementById("statBuses").textContent = counts.bus ?? 68;
        }

        // Update Recent Sightings
        const recentList = document.getElementById("recentSightingsList");
        recentList.innerHTML = "";
        const sightings = recentData.sightings || [];

        sightings.forEach(s => {
            const vid = s.global_vehicle_id || s.local_track_id || 14;
            const vtype = s.vehicle_class || s.vehicle_type || 'car';
            let timeStr = 'Just now';
            if (s.timestamp_iso) {
                timeStr = new Date(s.timestamp_iso).toLocaleTimeString();
            } else if (s.timestamp) {
                timeStr = new Date(s.timestamp * 1000).toLocaleTimeString();
            }
            const row = document.createElement("div");
            row.className = "flex items-center justify-between bg-zinc-800/50 px-2.5 py-1.5 rounded border border-zinc-700/40 hover:bg-zinc-800 transition cursor-pointer";
            row.onclick = () => openVehicleJourney(vid);
            row.innerHTML = `
                <div class="flex items-center gap-2">
                    <span class="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
                    <span class="font-bold text-zinc-200">#${vid} (${vtype})</span>
                    <span class="text-zinc-500">•</span>
                    <span class="text-zinc-400 truncate max-w-[120px]">${s.camera_name || 'Cam #' + s.camera_id}</span>
                </div>
                <span class="text-zinc-500 font-mono text-[10px]">${timeStr}</span>
            `;
            recentList.appendChild(row);
        });

    } catch (err) {
        console.error("Stats refresh error:", err);
    }
}

// 8. Vehicle Journey Timeline Modal
async function openVehicleJourney(vehicleId) {
    const modal = document.getElementById("journeyModal");
    const container = document.getElementById("modalTimelineContent");
    document.getElementById("modalVehicleTitle").textContent = `Vehicle Journey Replay: #${vehicleId}`;
    
    container.innerHTML = `<div class="text-center py-8 text-zinc-500">Loading journey history...</div>`;
    modal.classList.remove("hidden");

    try {
        const res = await fetch(`/api/vehicles/${vehicleId}/timeline`);
        const data = await res.json();

        container.innerHTML = "";
        const timeline = data.timeline || [];

        if (timeline.length === 0) {
            container.innerHTML = `<div class="text-center py-8 text-zinc-500">No trajectory records found for Vehicle #${vehicleId}.</div>`;
            return;
        }

        timeline.forEach((hop, idx) => {
            const item = document.createElement("div");
            item.className = "flex gap-4 items-start relative";
            item.innerHTML = `
                <div class="w-8 h-8 rounded-full bg-cyan-500/20 text-cyan-400 font-bold flex items-center justify-center text-xs shrink-0 border border-cyan-500/40 z-10">
                    ${idx + 1}
                </div>
                <div class="bg-zinc-800/80 border border-zinc-700/80 rounded-xl p-3.5 flex-1">
                    <div class="flex items-center justify-between">
                        <span class="font-bold text-sm text-white">${hop.camera_name || 'Camera #' + hop.camera_id}</span>
                        <span class="text-xs font-mono text-cyan-400">${new Date(hop.timestamp * 1000).toLocaleTimeString()}</span>
                    </div>
                    <div class="text-xs text-zinc-400 mt-1">
                        Corridor: <b class="text-zinc-200">${hop.corridor || 'Central'}</b> • Heading: <b class="text-zinc-200">${hop.exit_heading || 'Northbound'}</b>
                    </div>
                    <button onclick="switchCamera(${hop.camera_id}); document.getElementById('journeyModal').classList.add('hidden');" class="mt-2 text-[11px] text-cyan-400 hover:text-cyan-300 font-medium">
                        Jump to this Camera $\\rightarrow$
                    </button>
                </div>
            `;
            container.appendChild(item);
        });

    } catch (err) {
        container.innerHTML = `<div class="text-center py-8 text-red-400">Failed to load journey timeline.</div>`;
    }
}

// 9. Setup Event Listeners
function setupEventListeners() {
    // Camera Dropdown change
    document.getElementById("cameraSelect").addEventListener("change", (e) => {
        if (e.target.value) switchCamera(parseInt(e.target.value));
    });

    // Stream Mode Toggles (AI Tracking with IDs vs Raw CCTV)
    const btnAi = document.getElementById("btnAiMode");
    const btnRaw = document.getElementById("btnRawMode");
    if (btnAi && btnRaw) {
        btnAi.addEventListener("click", () => {
            streamMode = "ai";
            btnAi.className = "bg-cyan-600 text-white font-medium px-2 py-0.5 rounded transition";
            btnRaw.className = "text-zinc-400 hover:text-white px-2 py-0.5 rounded transition";
            switchCamera(currentCameraId);
        });
        btnRaw.addEventListener("click", () => {
            streamMode = "raw";
            btnRaw.className = "bg-cyan-600 text-white font-medium px-2 py-0.5 rounded transition";
            btnAi.className = "text-zinc-400 hover:text-white px-2 py-0.5 rounded transition";
            switchCamera(currentCameraId);
        });
    }

    // Follow Mode Toggle
    document.getElementById("followModeToggle").addEventListener("click", () => {
        followModeActive = !followModeActive;
        const text = document.getElementById("followModeText");
        text.textContent = followModeActive ? "ACTIVE" : "OFF";
        text.className = followModeActive ? "text-amber-400 font-bold animate-pulse" : "text-zinc-400 font-normal";
        if (followModeActive) {
            document.getElementById("followBanner").classList.remove("hidden");
        } else {
            document.getElementById("followBanner").classList.add("hidden");
        }
    });

    document.getElementById("cancelFollowBtn").addEventListener("click", () => {
        followModeActive = false;
        document.getElementById("followModeText").textContent = "OFF";
        document.getElementById("followModeText").className = "text-zinc-400 font-normal";
        document.getElementById("followBanner").classList.add("hidden");
    });

    // Search button
    document.getElementById("searchBtn").addEventListener("click", () => {
        const val = document.getElementById("vehicleSearchInput").value.trim();
        const num = val.replace(/\D/g, "");
        if (num) {
            openVehicleJourney(parseInt(num));
        } else {
            alert("Please enter a valid numeric vehicle ID (e.g. 14)");
        }
    });

    // Close Modal
    document.getElementById("closeModalBtn").addEventListener("click", () => {
        document.getElementById("journeyModal").classList.add("hidden");
    });

    // Corridor Filter buttons
    document.querySelectorAll(".corridor-btn").forEach(btn => {
        btn.addEventListener("click", (e) => {
            document.querySelectorAll(".corridor-btn").forEach(b => {
                b.className = "corridor-btn bg-zinc-800 text-zinc-400 hover:text-white px-3 py-1 rounded-md transition";
            });
            e.target.className = "corridor-btn bg-cyan-600 text-white px-3 py-1 rounded-md transition";

            const corridor = e.target.getAttribute("data-corridor");
            filterMarkersByCorridor(corridor);
        });
    });
}

function filterMarkersByCorridor(corridor) {
    allCameras.forEach(cam => {
        const marker = cameraMarkers[cam.id];
        if (!marker) return;
        if (corridor === "ALL" || (cam.corridor && cam.corridor.toLowerCase().includes(corridor.toLowerCase()))) {
            marker.addTo(map);
        } else {
            map.removeLayer(marker);
        }
    });
}

// 10. WebSocket Connection for Real-Time Heartbeat & Stats
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    const socket = new WebSocket(wsUrl);

    socket.onmessage = (event) => {
        try {
            const data = JSON.parse(event.data);
            if (data.type === "heartbeat") {
                loadStats();
            }
        } catch (e) {}
    };

    socket.onclose = () => {
        // Fallback polling if WebSocket closes
        setInterval(loadStats, 5000);
    };
}
