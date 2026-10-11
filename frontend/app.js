// Kota Bogor Traffic Intelligence — frontend controller

let map;
let cameraMarkers = {};
let allCameras = [];
let topologyData = null;
let currentCameraId = 167; // Default to Simpang Pahlawan 2 -F
let hlsInstance = null;
let followModeActive = false;
let followTargetVehicleId = null;
let journeyVehicleId = null;
let streamMode = "ai"; // "ai" (annotated with YOLO + tracker IDs) or "raw"
let isPolling = false;
let pollingCamId = null;
let cameraCountsMap = {};
let cameraRates = {};   // camera_id -> {vph, vph_dir_pos, vph_dir_neg, vph_classes, observed_s, ...}
let cameraDetailsList = [];
let activeTab = "cameras";
let cameraSearchFilter = "";
let activeCorridor = "ALL";

const VEHICLE_CLASSES = [
    { key: "car", label: "Car" },
    { key: "motorcycle", label: "Moto" },
    { key: "truck", label: "Truck" },
    { key: "bus", label: "Bus" },
];

// Escape API-provided strings before they are interpolated into HTML.
function esc(value) {
    return String(value ?? "").replace(/[&<>"']/g, ch => (
        { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]
    ));
}

// Traffic volume is a rate: crossings of each camera's counting line per hour of observation.
function fmtRate(vph, lowSample) {
    if (vph === null || vph === undefined) return "–";
    const text = vph >= 1000 ? `${(vph / 1000).toFixed(1)}k` : `${Math.round(vph)}`;
    return lowSample ? `~${text}` : text;   // "~": under a minute observed in the last hour
}
function rateLevel(vph) {
    if (vph === null || vph === undefined || vph <= 0) return "level-zero";
    if (vph > 1500) return "level-high";
    if (vph > 600) return "level-med";
    return "level-low";
}

function classChips(classes) {
    return VEHICLE_CLASSES.map(c =>
        `<span class="cls cls--${c.key}"><i></i>${c.label} ${fmtRate((classes || {})[c.key])}</span>`
    ).join("");
}

function popupHtml(cam, rate) {
    let rows;
    if (!rate) {
        rows = `<div class="popup__row"><span>Not observed in the last hour yet</span></div>`;
    } else {
        rows = `
            <div class="popup__row"><span>Flow</span><b>${fmtRate(rate.vph, rate.low_sample)} veh/h</b></div>
            <div class="popup__row"><span>By direction</span><b>${fmtRate(rate.vph_dir_pos)} / ${fmtRate(rate.vph_dir_neg)}</b></div>
            <div class="popup__row"><span>Observed</span><b>${Math.round(rate.observed_s)} s · ${rate.counting_mode === "line" ? "counting line" : "learning line"}</b></div>
            <div class="popup__row">${classChips(rate.vph_classes)}</div>`;
    }
    return `
        <div class="popup">
            <div class="popup__name">${esc(cam.name)}</div>
            <div class="popup__meta">ID ${esc(cam.id)} · ${esc(cam.corridor || "Central")}</div>
            <div class="popup__rows">${rows}</div>
            <button class="btn btn--primary btn--sm" data-switch-camera="${Number(cam.id)}">View live stream</button>
        </div>`;
}

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
                setTimeout(pollNext, 70); // ~14 FPS sequential chaining (one in-flight frame per tab)
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

document.addEventListener("DOMContentLoaded", () => {
    initMap();
    loadCameras();
    loadTopology();
    loadStats();
    setInterval(() => {
        if (activeTab === "timeseries") {
            loadTimeSeriesMetrics(currentCameraId);
        }
    }, 2500);
    setupEventListeners();
    initWebSocket();
});

// 1. Leaflet map centered on Kota Bogor
function initMap() {
    map = L.map("map", { zoomControl: true, attributionControl: true })
        .setView([-6.5971, 106.8060], 13);
    map.attributionControl.setPrefix(false);

    // OpenStreetMap tiles; the dark look comes from a CSS filter on the tile pane (see styles.css).
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
        maxZoom: 19,
        subdomains: ["a", "b", "c"],
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
}

// 2. Cameras: dropdown + map markers
async function loadCameras() {
    try {
        const res = await fetch("/api/cameras");
        const data = await res.json();
        // Cameras the BSW site no longer lists stay in the road graph but aren't shown.
        allCameras = (data.cameras || []).filter(c => c.status !== "removed");
        document.querySelectorAll(".cam-total").forEach(el => { el.textContent = allCameras.length; });

        const select = document.getElementById("cameraSelect");
        select.innerHTML = "";

        allCameras.forEach(cam => {
            const opt = document.createElement("option");
            opt.value = cam.id;
            opt.textContent = `${cam.id} · ${cam.name}`;
            if (cam.id === currentCameraId) opt.selected = true;
            select.appendChild(opt);

            if (cam.latitude == null || cam.longitude == null) return;

            const rate = cameraRates[cam.id];
            const count = fmtRate(rate ? rate.vph : null, rate && rate.low_sample);
            const levelClass = rateLevel(rate ? rate.vph : null);

            const icon = L.divIcon({
                className: "cam-marker-container",
                html: `<div class="cam-marker ${levelClass} ${cam.id === currentCameraId ? "active-marker" : ""}" id="marker-${cam.id}"><span>${cam.id}</span><span class="cam-marker__val" id="count-${cam.id}">${count}</span></div>`,
                iconSize: [58, 22],
                iconAnchor: [29, 11]
            });

            const marker = L.marker([cam.latitude, cam.longitude], { icon, title: cam.name }).addTo(map);
            marker.bindPopup(popupHtml(cam, rate));
            marker.on("click", () => switchCamera(cam.id));
            cameraMarkers[cam.id] = marker;
        });

        switchCamera(currentCameraId);
        filterMarkersByCorridor(activeCorridor);
    } catch (err) {
        console.error("Failed to load cameras:", err);
    }
}

// 3. Road topology: draw directed edges between connected cameras
async function loadTopology() {
    try {
        const res = await fetch("/api/topology");
        topologyData = await res.json();

        if (topologyData && topologyData.edges) {
            topologyData.edges.forEach(edge => {
                const fromCam = allCameras.find(c => c.id === edge.from_camera_id || c.id === edge.from);
                const toCam = allCameras.find(c => c.id === edge.to_camera_id || c.id === edge.to);
                if (fromCam && toCam && fromCam.latitude && toCam.latitude) {
                    L.polyline([[fromCam.latitude, fromCam.longitude], [toCam.latitude, toCam.longitude]], {
                        color: "#4c8dff",
                        weight: 1.5,
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

// 4. Switch active camera and start its feed
function switchCamera(cameraId) {
    cameraId = Number(cameraId);
    currentCameraId = cameraId;
    const cam = allCameras.find(c => c.id === cameraId);
    if (!cam) return;

    document.querySelectorAll(".cam-marker").forEach(el => el.classList.remove("active-marker"));
    const activeEl = document.getElementById(`marker-${cameraId}`);
    if (activeEl) activeEl.classList.add("active-marker");

    if (cam.latitude && cam.longitude) {
        map.setView([cam.latitude, cam.longitude], 15, { animate: true });
    }

    document.getElementById("videoPlayerTitle").textContent = cam.name;
    document.getElementById("cameraBadgeId").textContent = `ID ${cam.id}`;
    document.getElementById("activeCamIndicator").innerHTML = `Selected: <b>${esc(cam.name)}</b>`;
    document.getElementById("cameraSelect").value = cam.id;

    const video = document.getElementById("cctvVideo");
    const iframe = document.getElementById("cctvEmbedIframe");
    const aiImg = document.getElementById("aiStreamImg");
    const spinner = document.getElementById("videoLoadingSpinner");

    if (streamMode === "ai") {
        if (hlsInstance) { hlsInstance.destroy(); hlsInstance = null; }
        video.classList.add("hidden");
        iframe.classList.add("hidden");
        spinner.classList.add("hidden");
        aiImg.classList.remove("hidden");
        // Poll single frames instead of an MJPEG stream: WebKit can keep an old multipart
        // connection open after the src changes, and every open stream keeps its camera in the
        // tracker's focus set. Polling stops for the old camera the moment the camera changes.
        aiImg.onerror = null;
        aiImg.removeAttribute("src");
        startFramePolling(cameraId);
    } else {
        stopFramePolling();
        aiImg.classList.add("hidden");
        aiImg.src = "";
        spinner.classList.remove("hidden");

        if (Hls.isSupported() && cam.stream_url) {
            if (hlsInstance) hlsInstance.destroy();
            hlsInstance = new Hls({
                enableWorker: true,
                lowLatencyMode: true,
                liveSyncDurationCount: 1,
                liveMaxLatencyDurationCount: 2
            });
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

    fetchDownstreamCandidates(cameraId);
    renderAllCamerasCountsList();

    if (activeTab === "timeseries") {
        loadTimeSeriesMetrics(cameraId);
    }
}

// 5. Predictive handoff candidates
async function fetchDownstreamCandidates(cameraId) {
    const listContainer = document.getElementById("candidateList");
    listContainer.innerHTML = `<div class="empty">Calculating downstream candidates…</div>`;

    document.querySelectorAll(".cam-marker").forEach(el => el.classList.remove("candidate-marker"));

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
            listContainer.innerHTML = `<div class="empty">No downstream camera is configured for this heading.</div>`;
            return;
        }

        data.candidates.forEach(cand => {
            const candMarker = document.getElementById(`marker-${cand.target_camera_id}`);
            if (candMarker) candMarker.classList.add("candidate-marker");

            const card = document.createElement("div");
            card.className = "row";
            card.innerHTML = `
                <div class="row__main">
                    <div class="row__title truncate">${esc(cand.target_camera_name)}</div>
                    <div class="row__sub">
                        <span>ID ${esc(cand.target_camera_id)}</span>
                        <span>Transit ~${esc(cand.travel_time_seconds)} s</span>
                        <span>Window ${esc(cand.arrival_window.t_min_seconds)}–${esc(cand.arrival_window.t_max_seconds)} s</span>
                    </div>
                </div>
                <button class="btn btn--sm" data-handoff-camera="${Number(cand.target_camera_id)}">Switch feed</button>
            `;
            listContainer.appendChild(card);
        });
    } catch (err) {
        listContainer.innerHTML = `<div class="empty empty--error">Failed to calculate candidates.</div>`;
    }
}

// 6. Camera handoff ("Follow that car")
function setFollowBanner(title, text) {
    document.getElementById("followTargetId").textContent = title;
    document.getElementById("followRouteText").textContent = text;
}

function followBannerIdleText() {
    return followTargetVehicleId
        ? "Waiting for the next handoff."
        : "Open a vehicle from the Recent tab and choose Follow vehicle.";
}

function setFollowMode(active) {
    followModeActive = active;
    const btn = document.getElementById("followModeToggle");
    btn.setAttribute("aria-pressed", String(active));
    document.getElementById("followModeText").textContent = active ? "On" : "Off";
    document.getElementById("followBanner").classList.toggle("hidden", !active);
    if (active) {
        setFollowBanner(
            followTargetVehicleId ? `Following vehicle #${followTargetVehicleId}` : "No target selected",
            followBannerIdleText()
        );
    }
}

function handoffTo(nextCameraId) {
    const from = currentCameraId;
    document.getElementById("followBanner").classList.remove("hidden");
    setFollowBanner(
        followTargetVehicleId ? `Following vehicle #${followTargetVehicleId}` : "Camera handoff",
        `Switching from camera ${from} to camera ${nextCameraId}…`
    );
    setTimeout(() => {
        switchCamera(nextCameraId);
        setFollowBanner(
            followTargetVehicleId ? `Following vehicle #${followTargetVehicleId}` : "Camera handoff",
            `Now viewing camera ${nextCameraId}.`
        );
        if (!followModeActive) {
            setTimeout(() => {
                if (!followModeActive) document.getElementById("followBanner").classList.add("hidden");
            }, 4000);
        }
    }, 800);
}

// 7. Citywide statistics and per-camera rates
function updateCameraMarkersCounts() {
    allCameras.forEach(cam => {
        const marker = cameraMarkers[cam.id];
        if (!marker) return;
        const rate = cameraRates[cam.id];

        const countEl = document.getElementById(`count-${cam.id}`);
        if (countEl) countEl.textContent = fmtRate(rate ? rate.vph : null, rate && rate.low_sample);

        const markerDiv = document.getElementById(`marker-${cam.id}`);
        if (markerDiv) {
            markerDiv.classList.remove("level-zero", "level-low", "level-med", "level-high");
            markerDiv.classList.add(rateLevel(rate ? rate.vph : null));
        }

        marker.setPopupContent(popupHtml(cam, rate));
    });
}

function renderAllCamerasCountsList() {
    const container = document.getElementById("allCamerasCountsList");
    if (!container) return;

    let items = cameraDetailsList.length > 0 ? [...cameraDetailsList] : allCameras.map(c => ({
        camera_id: c.id,
        name: c.name,
        corridor: c.corridor,
    }));

    // Busiest cameras first; cameras without a recent observation last.
    items.sort((a, b) => ((cameraRates[b.camera_id] || {}).vph ?? -1) - ((cameraRates[a.camera_id] || {}).vph ?? -1));

    if (cameraSearchFilter) {
        items = items.filter(c =>
            String(c.camera_id).includes(cameraSearchFilter) ||
            (c.name && c.name.toLowerCase().includes(cameraSearchFilter)) ||
            (c.corridor && c.corridor.toLowerCase().includes(cameraSearchFilter))
        );
    }

    if (items.length === 0) {
        container.innerHTML = `<div class="empty">${cameraSearchFilter ? `No cameras match “${esc(cameraSearchFilter)}”.` : "No cameras loaded."}</div>`;
        return;
    }

    container.innerHTML = items.map(cam => {
        const rate = cameraRates[cam.camera_id];
        const isCurrent = cam.camera_id === currentCameraId;
        const title = rate ? `${Math.round(rate.observed_s)} s observed in the last hour` : "Not observed in the last hour";

        return `
            <div class="row row--click ${isCurrent ? "row--current" : ""}" role="button" tabindex="0" data-switch-camera="${Number(cam.camera_id)}">
                <div class="row__main">
                    <div class="row__title">
                        <span class="truncate">${esc(cam.name)}</span>
                        ${isCurrent ? '<span class="tag">VIEWING</span>' : ""}
                    </div>
                    <div class="row__sub">
                        <span>ID ${esc(cam.camera_id)}</span>
                        <span class="truncate" style="max-width:150px">${esc(cam.corridor || "–")}</span>
                        ${classChips((rate && rate.vph_classes) || {})}
                    </div>
                </div>
                <div class="row__side">
                    <span class="rate ${rateLevel(rate ? rate.vph : null)}" title="${esc(title)}">
                        ${fmtRate(rate ? rate.vph : null, rate && rate.low_sample)} <small>veh/h</small>
                    </span>
                </div>
            </div>`;
    }).join("");
}

function setupTabs() {
    const tabs = document.querySelectorAll(".tab[data-tab]");
    const panels = {
        cameras: document.getElementById("tabContentCameras"),
        timeseries: document.getElementById("tabContentTimeSeries"),
        handoff: document.getElementById("tabContentHandoff"),
        sightings: document.getElementById("tabContentSightings"),
    };

    function setTab(tab) {
        activeTab = tab;
        tabs.forEach(btn => btn.setAttribute("aria-selected", String(btn.dataset.tab === tab)));
        Object.entries(panels).forEach(([name, el]) => { if (el) el.hidden = name !== tab; });
        if (tab === "timeseries") loadTimeSeriesMetrics(currentCameraId);
    }

    window.switchTab = setTab;
    tabs.forEach(btn => btn.addEventListener("click", () => setTab(btn.dataset.tab)));

    const filterInput = document.getElementById("cameraFilterInput");
    if (filterInput) {
        filterInput.addEventListener("input", (e) => {
            cameraSearchFilter = e.target.value.toLowerCase().trim();
            renderAllCamerasCountsList();
        });
    }
}

// Initial load (and fallback when a heartbeat doesn't carry the data): fetch the same payloads the
// server pushes on every WebSocket heartbeat.
async function loadStats() {
    try {
        const [countsRes, recentRes] = await Promise.all([
            fetch("/api/cameras/counts"),
            fetch("/api/recent?limit=12")
        ]);
        const countsData = await countsRes.json();
        const recentData = await recentRes.json();
        applyStats(countsData, recentData.sightings || []);
    } catch (err) {
        console.error("Stats refresh error:", err);
    }
}

function applyStats(countsData, sightings) {
    cameraCountsMap = countsData.camera_counts_map || {};
    cameraRates = countsData.camera_rates || {};
    cameraDetailsList = (countsData.camera_counts || []).filter(c => c.status !== "removed");

    const classes = countsData.class_breakdown || {};
    const setNum = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = (value || 0).toLocaleString();
    };
    setNum("statTotalSightings", countsData.total_vehicles_counted);
    setNum("statCars", classes.car);
    setNum("statMotos", classes.motorcycle);
    setNum("statTrucks", classes.truck);
    setNum("statBuses", classes.bus);

    const headerCount = document.getElementById("headerVehicleCountText");
    if (headerCount) headerCount.textContent = `≈${fmtRate(countsData.citywide_vph || 0)} veh/h citywide`;

    const activeCamsEl = document.getElementById("camsWithTrafficCount");
    if (activeCamsEl) activeCamsEl.textContent = countsData.cameras_with_rates || 0;

    updateCameraMarkersCounts();
    renderAllCamerasCountsList();
    renderRecentSightings(sightings);
}

function renderRecentSightings(sightings) {
    const recentList = document.getElementById("recentSightingsList");
    if (!recentList) return;

    if (sightings.length === 0) {
        recentList.innerHTML = `<div class="empty">No vehicles recorded yet.</div>`;
        return;
    }

    recentList.innerHTML = sightings.map(s => {
        const vid = s.global_vehicle_id || s.local_track_id;
        const vtype = s.vehicle_class || s.vehicle_type || "car";
        let timeStr = "";
        if (s.timestamp_iso) timeStr = new Date(s.timestamp_iso).toLocaleTimeString();
        else if (s.timestamp) timeStr = new Date(s.timestamp * 1000).toLocaleTimeString();

        return `
            <div class="row row--click" role="button" tabindex="0" data-journey-vehicle="${Number(vid)}">
                <div class="row__main">
                    <div class="row__title">Vehicle #${esc(vid)}</div>
                    <div class="row__sub">
                        <span class="cls cls--${esc(vtype)}"><i></i>${esc(vtype)}</span>
                        <span class="truncate" style="max-width:200px">${esc(s.camera_name || "Camera " + s.camera_id)}</span>
                    </div>
                </div>
                <span class="muted num">${esc(timeStr)}</span>
            </div>`;
    }).join("");
}

// 8. Vehicle journey modal
function closeJourneyModal() {
    document.getElementById("journeyModal").classList.add("hidden");
}

async function openVehicleJourney(vehicleId) {
    if (!Number.isFinite(vehicleId) || vehicleId <= 0) return;
    journeyVehicleId = vehicleId;
    const modal = document.getElementById("journeyModal");
    const container = document.getElementById("modalTimelineContent");
    document.getElementById("modalVehicleTitle").textContent = `Vehicle #${vehicleId} journey`;

    container.innerHTML = `<div class="empty">Loading journey history…</div>`;
    modal.classList.remove("hidden");

    try {
        const res = await fetch(`/api/vehicles/${vehicleId}/timeline`);
        const data = await res.json();
        const timeline = data.timeline || [];

        if (timeline.length === 0) {
            container.innerHTML = `<div class="empty">No trajectory records found for vehicle #${vehicleId}.</div>`;
            return;
        }

        container.innerHTML = timeline.map((hop, idx) => `
            <div class="timeline-item">
                <div class="timeline-item__idx">${idx + 1}</div>
                <div class="timeline-item__card">
                    <div class="timeline-item__head">
                        <span>${esc(hop.camera_name || "Camera " + hop.camera_id)}</span>
                        <span class="num">${new Date(hop.timestamp * 1000).toLocaleTimeString()}</span>
                    </div>
                    <div class="timeline-item__meta">
                        ${esc(hop.corridor || "Central")} · Heading ${esc(hop.exit_heading || "unknown")}
                    </div>
                    <button class="link-btn" data-switch-camera="${Number(hop.camera_id)}" data-close-modal="1">View this camera →</button>
                </div>
            </div>`).join("");
    } catch (err) {
        container.innerHTML = `<div class="empty empty--error">Failed to load journey timeline.</div>`;
    }
}

// 9. Event listeners
function setupEventListeners() {
    document.getElementById("cameraSelect").addEventListener("change", (e) => {
        if (e.target.value) switchCamera(parseInt(e.target.value));
    });

    // Delegated actions for dynamically rendered content (lists, popups, modal)
    document.addEventListener("click", (e) => {
        const camBtn = e.target.closest("[data-switch-camera]");
        if (camBtn) {
            switchCamera(camBtn.dataset.switchCamera);
            if (camBtn.dataset.closeModal) closeJourneyModal();
            return;
        }
        const handoffBtn = e.target.closest("[data-handoff-camera]");
        if (handoffBtn) {
            handoffTo(Number(handoffBtn.dataset.handoffCamera));
            return;
        }
        const journeyRow = e.target.closest("[data-journey-vehicle]");
        if (journeyRow) openVehicleJourney(Number(journeyRow.dataset.journeyVehicle));
    });
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeJourneyModal();
        if ((e.key === "Enter" || e.key === " ") && e.target.matches && e.target.matches('[role="button"][tabindex]')) {
            e.preventDefault();
            e.target.click();
        }
    });

    // Stream mode toggle
    const btnAi = document.getElementById("btnAiMode");
    const btnRaw = document.getElementById("btnRawMode");
    const setMode = (mode) => {
        streamMode = mode;
        btnAi.setAttribute("aria-pressed", String(mode === "ai"));
        btnRaw.setAttribute("aria-pressed", String(mode === "raw"));
        switchCamera(currentCameraId);
    };
    btnAi.addEventListener("click", () => setMode("ai"));
    btnRaw.addEventListener("click", () => setMode("raw"));

    // Follow mode
    document.getElementById("followModeToggle").addEventListener("click", () => setFollowMode(!followModeActive));
    document.getElementById("cancelFollowBtn").addEventListener("click", () => {
        followTargetVehicleId = null;
        setFollowMode(false);
    });

    // Vehicle lookup → journey
    const runSearch = () => {
        const num = parseInt(document.getElementById("vehicleSearchInput").value.replace(/\D/g, ""), 10);
        if (num) openVehicleJourney(num);
    };
    document.getElementById("searchBtn").addEventListener("click", runSearch);
    document.getElementById("vehicleSearchForm").addEventListener("submit", (e) => {
        e.preventDefault();
        runSearch();
    });

    // Journey modal
    document.getElementById("closeModalBtn").addEventListener("click", closeJourneyModal);
    document.getElementById("journeyModal").addEventListener("click", (e) => {
        if (e.target.id === "journeyModal") closeJourneyModal();
    });
    document.getElementById("followVehicleBtn").addEventListener("click", () => {
        if (!journeyVehicleId) return;
        followTargetVehicleId = journeyVehicleId;
        setFollowMode(true);
        fetchDownstreamCandidates(currentCameraId);
        closeJourneyModal();
        window.switchTab("handoff");
    });

    // Corridor filter
    document.querySelectorAll(".corridor-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".corridor-btn").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
            activeCorridor = btn.getAttribute("data-corridor");
            filterMarkersByCorridor(activeCorridor);
        });
    });

    // Dataset export
    const exportUrl = (format) => {
        const win = document.getElementById("exportWindowSelect")?.value || 60;
        window.location.href = `/api/stats/export?camera_id=${currentCameraId}&format=${format}&window_seconds=${win}`;
    };
    document.getElementById("btnExportCsv")?.addEventListener("click", () => exportUrl("csv"));
    document.getElementById("btnExportJson")?.addEventListener("click", () => exportUrl("json"));

    setupTabs();
}

function filterMarkersByCorridor(corridor) {
    allCameras.forEach(cam => {
        const marker = cameraMarkers[cam.id];
        if (!marker) return;
        const show = corridor === "ALL" || (cam.corridor && cam.corridor.toLowerCase().includes(corridor.toLowerCase()));
        if (show) marker.addTo(map);
        else map.removeLayer(marker);
    });
}

// 10. WebSocket heartbeat drives stats refresh
let liveWebSocket = null;
function initWebSocket() {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const wsUrl = `${protocol}//${window.location.host}/ws/live`;
    try {
        liveWebSocket = new WebSocket(wsUrl);

        liveWebSocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                if (data.type === "heartbeat") {
                    if (data.counts) applyStats(data.counts, data.recent || []);
                    else loadStats();
                }
            } catch (e) {}
        };
        liveWebSocket.onclose = () => { setTimeout(initWebSocket, 2500); };
        liveWebSocket.onerror = () => { try { liveWebSocket.close(); } catch (e) {} };
    } catch (e) {
        console.warn("WebSocket initialization deferred:", e);
    }
}

// 11. Per-camera analytics
// Same thresholds as the backend's get_congestion_label (Smooth < 0.25 < Moderate < 0.55 < Heavy < 0.8 < Congested).
const CONGESTION_LEVELS = [
    { min: 0.8, label: "Congested", cls: "sev--severe" },
    { min: 0.55, label: "Heavy", cls: "sev--heavy" },
    { min: 0.25, label: "Moderate", cls: "sev--moderate" },
    { min: 0.1, label: "Light", cls: "sev--light" },
    { min: -Infinity, label: "Free flow", cls: "sev--free" },
];

function fmtDuration(seconds) {
    if (!(seconds > 0)) return "none";
    return seconds < 90 ? `${Math.round(seconds)} s` : `${(seconds / 60).toFixed(seconds < 600 ? 1 : 0)} min`;
}

async function loadTimeSeriesMetrics(cameraId) {
    try {
        const res = await fetch(`/api/cameras/${cameraId}/metrics`);
        if (!res.ok) return;
        const data = await res.json();

        const live = data.live || {};
        const history = data.recent_history || [];
        const lastRec = history.length > 0 ? history[history.length - 1] : null;
        const flow = data.flow || {};
        const rate = flow.rate || null;      // counting-line rate over the window; null if never observed
        const setText = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };

        setText("tsCamName", `${data.camera_name || "Camera"} (ID ${cameraId})`);

        // Flow: crossings of the camera's counting line per hour of observation
        setText("tsFlowVph", rate ? fmtRate(rate.vph, rate.low_sample) : "–");
        setText("tsFlowPerMin", rate ? (rate.vph / 60 < 10 ? (rate.vph / 60).toFixed(1) : Math.round(rate.vph / 60)) : "–");
        setText("tsObserved", rate ? fmtDuration(rate.observed_s) : "none");

        // Congestion Index
        const cong = live.congestion_index !== undefined ? live.congestion_index : (lastRec ? lastRec.congestion_index : 0.0);
        setText("tsCongestionIndex", Number(cong).toFixed(2));

        const occ = live.occupancy_ratio !== undefined ? (live.occupancy_ratio * 100).toFixed(1) : (lastRec ? (lastRec.occupancy_ratio * 100).toFixed(1) : "0.0");
        setText("tsOccupancyPct", `${occ}% road coverage`);

        const barEl = document.getElementById("tsCongestionBar");
        if (barEl) {
            barEl.style.width = `${Math.min(Math.max(cong * 100, 5), 100)}%`;
            barEl.style.background = cong >= 0.55 ? "var(--bad)" : (cong >= 0.25 ? "var(--warn)" : "var(--ok)");
        }
        const badgeEl = document.getElementById("tsCongestionBadge");
        if (badgeEl) {
            const level = CONGESTION_LEVELS.find(l => cong > l.min) || CONGESTION_LEVELS[CONGESTION_LEVELS.length - 1];
            badgeEl.textContent = level.label;
            badgeEl.className = `sev ${level.cls}`;
        }

        // Classes: counting-line rate per class
        const classes = (rate && rate.vph_classes) || {};
        setText("tsCarCount", rate ? fmtRate(classes.car) : "–");
        setText("tsMotoCount", rate ? fmtRate(classes.motorcycle) : "–");
        setText("tsTruckCount", rate ? fmtRate(classes.truck) : "–");
        setText("tsBusCount", rate ? fmtRate(classes.bus) : "–");

        // Kinematics
        const speed = live.avg_speed_px_s !== undefined ? live.avg_speed_px_s : (lastRec ? lastRec.avg_speed_px_s : 0.0);
        setText("tsSpeedProxy", Number(speed).toFixed(1));

        const dx = live.flow_dx !== undefined ? live.flow_dx : (lastRec ? lastRec.avg_flow_dx : 0.0);
        const dy = live.flow_dy !== undefined ? live.flow_dy : (lastRec ? lastRec.avg_flow_dy : 0.0);
        setText("tsFlowVector", `${Number(dx).toFixed(1)}, ${Number(dy).toFixed(1)}`);

        // Cyclic temporal encodings
        const now = new Date();
        const hour = now.getHours() + now.getMinutes() / 60.0;
        const sinH = lastRec ? lastRec.sin_hour : Math.sin(2 * Math.PI * hour / 24.0);
        const cosH = lastRec ? lastRec.cos_hour : Math.cos(2 * Math.PI * hour / 24.0);
        setText("tsCyclicHour", `${Number(sinH).toFixed(2)} / ${Number(cosH).toFixed(2)}`);

        const isWknd = lastRec ? lastRec.is_weekend : ([0, 6].includes(now.getDay()) ? 1 : 0);
        setText("tsWeekendBadge", isWknd ? "Weekend" : "Weekday");

        const minutes = flow.window_minutes || 60;
        const series = flow.series || [];
        setText("tsHistoryPoints", `last ${minutes} min`);
        setText("tsTotalDbRecords", `${flow.windows || 0} windows`);
        renderTimeSeriesChart(series, minutes);
    } catch (err) {
        console.warn("Error loading time-series metrics:", err);
    }
}

// Flow (veh/h, left axis) and congestion (0-1, right axis) over the last `minutes`, on a real time
// axis. A bucket the camera wasn't observed in has no flow value, so the line breaks there instead of
// drawing a number that was never measured; hollow dots mark buckets with under a minute observed.
function renderTimeSeriesChart(series, minutes) {
    const canvas = document.getElementById("trafficMlCanvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = rect.width;
    const h = rect.height;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const FLOW = "#4c8dff", CONGESTION = "#e8a93a", MUTED = "#778293", GRID = "#232c38";
    ctx.font = "10px Inter, system-ui, sans-serif";

    const hasData = series.some(p => p.vph !== null && p.vph !== undefined || p.congestion !== null && p.congestion !== undefined);
    if (!hasData) {
        ctx.fillStyle = MUTED;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = "12px Inter, system-ui, sans-serif";
        ctx.fillText(`No observations in the last ${minutes} min yet`, w / 2, h / 2);
        return;
    }

    const pad = { top: 12, right: 30, bottom: 22, left: 36 };
    const chartW = Math.max(w - pad.left - pad.right, 20);
    const chartH = Math.max(h - pad.top - pad.bottom, 20);
    const tEnd = Date.now() / 1000;
    const tStart = tEnd - minutes * 60;
    const xAt = t => pad.left + Math.min(Math.max((t - tStart) / (tEnd - tStart), 0), 1) * chartW;
    const maxFlow = Math.max(...series.map(p => p.vph || 0), 100);

    // Grid and axes
    ctx.strokeStyle = GRID;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i <= 3; i++) {
        const y = pad.top + (chartH / 3) * i;
        ctx.moveTo(pad.left, y);
        ctx.lineTo(w - pad.right, y);
    }
    ctx.stroke();
    ctx.textBaseline = "middle";
    for (let i = 0; i <= 3; i++) {
        const y = pad.top + (chartH / 3) * i;
        ctx.fillStyle = MUTED;
        ctx.textAlign = "right";
        ctx.fillText(fmtRate(maxFlow * (1 - i / 3)), pad.left - 5, y);
        ctx.fillStyle = CONGESTION;
        ctx.textAlign = "left";
        ctx.fillText((1 - i / 3).toFixed(1), w - pad.right + 5, y);
    }

    // Bucket centre, clipped so the newest partial bucket doesn't extend past "now"
    const centre = p => Math.min((p.t + p.t_end) / 2, tEnd);
    const gap = (series.length > 1 ? Math.min(...series.slice(1).map((p, i) => p.t - series[i].t).filter(d => d > 0)) : 300) * 1.5;

    const drawSeries = (valueOf, yOf, color, width, dash, fillArea) => {
        const pts = series.map(p => ({ p, v: valueOf(p) })).filter(o => o.v !== null && o.v !== undefined);
        const segments = [];
        pts.forEach(o => {
            const last = segments[segments.length - 1];
            if (last && centre(o.p) - centre(last[last.length - 1].p) <= gap) last.push(o);
            else segments.push([o]);
        });
        segments.forEach(seg => {
            if (seg.length > 1) {
                if (fillArea) {
                    const grad = ctx.createLinearGradient(0, pad.top, 0, pad.top + chartH);
                    grad.addColorStop(0, "rgba(76, 141, 255, 0.25)");
                    grad.addColorStop(1, "rgba(76, 141, 255, 0)");
                    ctx.beginPath();
                    ctx.moveTo(xAt(centre(seg[0].p)), pad.top + chartH);
                    seg.forEach(o => ctx.lineTo(xAt(centre(o.p)), yOf(o.v)));
                    ctx.lineTo(xAt(centre(seg[seg.length - 1].p)), pad.top + chartH);
                    ctx.closePath();
                    ctx.fillStyle = grad;
                    ctx.fill();
                }
                ctx.beginPath();
                ctx.strokeStyle = color;
                ctx.lineWidth = width;
                ctx.lineJoin = "round";
                ctx.setLineDash(dash);
                seg.forEach((o, i) => (i === 0 ? ctx.moveTo(xAt(centre(o.p)), yOf(o.v)) : ctx.lineTo(xAt(centre(o.p)), yOf(o.v))));
                ctx.stroke();
                ctx.setLineDash([]);
            }
        });
        return pts;
    };

    const yFlow = v => pad.top + chartH - (v / maxFlow) * chartH;
    const yCong = v => pad.top + chartH - Math.min(Math.max(v, 0), 1) * chartH;
    const flowPts = drawSeries(p => p.vph, yFlow, FLOW, 2, [], true);
    drawSeries(p => p.congestion, yCong, CONGESTION, 1.5, [4, 3], false);

    // Points: every measured bucket gets a dot (hollow when under a minute was observed)
    flowPts.forEach(o => {
        ctx.beginPath();
        ctx.arc(xAt(centre(o.p)), yFlow(o.v), 3, 0, Math.PI * 2);
        if (o.p.low_sample) { ctx.strokeStyle = FLOW; ctx.lineWidth = 1.5; ctx.stroke(); }
        else { ctx.fillStyle = FLOW; ctx.fill(); }
    });
    series.filter(p => p.congestion !== null && p.congestion !== undefined).forEach(p => {
        ctx.beginPath();
        ctx.arc(xAt(centre(p)), yCong(p.congestion), 2.5, 0, Math.PI * 2);
        ctx.fillStyle = CONGESTION;
        ctx.fill();
    });

    // Time labels
    ctx.fillStyle = MUTED;
    ctx.textBaseline = "top";
    ctx.textAlign = "left";
    ctx.fillText(`${minutes} min ago`, pad.left, pad.top + chartH + 6);
    ctx.textAlign = "center";
    ctx.fillText(`${Math.round(minutes / 2)} min`, pad.left + chartW / 2, pad.top + chartH + 6);
    ctx.textAlign = "right";
    ctx.fillText("Now", pad.left + chartW, pad.top + chartH + 6);
}
