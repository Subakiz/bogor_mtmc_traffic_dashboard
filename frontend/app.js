// Kota Bogor Traffic Intelligence — frontend controller

let map;
let cameraMarkers = {};
let allCameras = [];
let topologyData = null;
let currentCameraId = 167; // Default to Simpang Pahlawan 2 -F
let hlsInstance = null;
let streamMode = "ai"; // "ai" (annotated with YOLO + tracker IDs) or "raw"
let isPolling = false;
let pollingCamId = null;
let cameraCountsMap = {};
let cameraRates = {};   // camera_id -> {vph, vph_dir_pos, vph_dir_neg, vph_classes, observed_s, ...}
let cameraDetailsList = [];
let activeTab = "cameras";
let cameraSearchFilter = "";
let activeCorridor = "ALL";
let lastSightings = [];            // newest "recent" payload; drawn when the Recent tab is visible
let openPopupCamId = null;         // camera whose map popup is open (Leaflet keeps at most one)
let shownPopupRows = "";           // rows markup currently inside that popup
const cameraRows = new Map();      // camera id -> row record in the Cameras tab (see syncList)
const sightingRows = new Map();    // sighting id -> row record in the Recent tab

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

function makeEl(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
}

// Writes only when the value differs, so an unchanged heartbeat causes no DOM mutation at all.
function setNodeText(node, text) {
    if (!node) return;
    const first = node.firstChild;
    if (first && first.nodeType === Node.TEXT_NODE && first === node.lastChild) {
        if (first.data !== text) first.data = text;   // edit the text node instead of replacing it
    } else if (node.textContent !== text) {
        node.textContent = text;
    }
}
function setTextData(textNode, text) {
    if (textNode.data !== text) textNode.data = text;
}

// Keeps `container` in step with `items` (already filtered and sorted) while reusing the rows that
// are still wanted: rows are created for new keys, removed for gone keys, moved only when their
// position changed, and `update` refreshes the ones that stayed. Nothing is rebuilt per refresh, so
// keyboard focus, text selection and hover state survive. `cache` maps key -> {el, ...}.
function syncList(container, cache, items, keyOf, create, update) {
    const focused = document.activeElement;
    const keep = new Set();
    let cursor = container.firstElementChild;
    for (const item of items) {
        const key = keyOf(item);
        if (keep.has(key)) continue;
        keep.add(key);
        let rec = cache.get(key);
        if (!rec) {
            rec = create(item);
            cache.set(key, rec);
        }
        if (update) update(rec, item);
        if (rec.el === cursor) cursor = cursor.nextElementSibling;
        else container.insertBefore(rec.el, cursor);
    }
    // Everything from the cursor on was not claimed by an item (stale rows, a placeholder).
    while (cursor) {
        const next = cursor.nextElementSibling;
        cursor.remove();
        cursor = next;
    }
    for (const key of cache.keys()) if (!keep.has(key)) cache.delete(key);
    // Moving a node in the DOM drops focus from it; hand it back.
    if (focused && focused !== document.activeElement && container.contains(focused)) {
        focused.focus({ preventScroll: true });
    }
}

function setListEmpty(container, cache, message) {
    cache.clear();
    let empty = container.firstElementChild;
    if (!empty || container.childElementCount !== 1 || !empty.classList.contains("empty")) {
        container.replaceChildren();
        empty = makeEl("div", "empty");
        container.append(empty);
    }
    setNodeText(empty, message);
}

// Colors live in styles.css as theme tokens (light and dark); canvas and Leaflet drawing read them here.
function cssVar(name, fallback) {
    const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
    return value || fallback;
}

function classChips(classes) {
    return VEHICLE_CLASSES.map(c =>
        `<span class="cls cls--${c.key}"><i></i>${c.label} ${fmtRate((classes || {})[c.key])}</span>`
    ).join("");
}

// The rows are separate from the rest of the popup so an open popup can refresh its numbers
// without replacing the "View live stream" button the user may be about to click.
function popupRowsHtml(rate) {
    if (!rate) {
        return `<div class="popup__row"><span>Not observed in the last hour yet</span></div>`;
    }
    return `
        <div class="popup__row"><span>Flow</span><b>${fmtRate(rate.vph, rate.low_sample)} veh/h</b></div>
        <div class="popup__row"><span>By direction</span><b>${fmtRate(rate.vph_dir_pos)} / ${fmtRate(rate.vph_dir_neg)}</b></div>
        <div class="popup__row"><span>Observed</span><b>${Math.round(rate.observed_s)} s · ${rate.counting_mode === "line" ? "counting line" : "learning line"}</b></div>
        <div class="popup__row">${classChips(rate.vph_classes)}</div>`;
}

function popupHtml(cam, rate) {
    return `
        <div class="popup">
            <div class="popup__name">${esc(cam.name)}</div>
            <div class="popup__meta">ID ${esc(cam.id)} · ${esc(cam.corridor || "Central")}</div>
            <div class="popup__rows">${popupRowsHtml(rate)}</div>
            <button class="btn btn--primary btn--sm" data-switch-camera="${Number(cam.id)}">View live stream</button>
        </div>`;
}

let frameObjectUrl = null;
let framePollGeneration = 0;

function stopFramePolling() {
    isPolling = false;
    pollingCamId = null;
    framePollGeneration++;
}

// One request in flight per tab. Each request names the last frame shown (?after=<X-Frame-No>);
// the server holds it until a newer frame exists (204 after ~1 s without one), so frames arrive
// as soon as they are processed and none is downloaded twice. Polling keeps the camera in the
// tracker's focus, so it stops while the tab is hidden (see visibilitychange below).
function startFramePolling(camId) {
    stopFramePolling();
    isPolling = true;
    pollingCamId = camId;
    const generation = framePollGeneration;
    let lastFrameNo = null;
    const active = () => isPolling && generation === framePollGeneration && streamMode === "ai" && !document.hidden;

    async function pollNext() {
        if (!active()) return;
        let delay = 0;
        try {
            const after = lastFrameNo === null ? "" : `?after=${lastFrameNo}`;
            const res = await fetch(`/api/stream/${camId}/frame${after}`, { cache: "no-store" });
            if (res.status === 200) {
                const blob = await res.blob();
                const frameNo = res.headers.get("X-Frame-No");
                // Placeholders (connecting, viewer limit) carry no frame number: re-check gently.
                if (frameNo === null) delay = 1000;
                else lastFrameNo = frameNo;
                if (active()) {
                    const url = URL.createObjectURL(blob);
                    const aiImg = document.getElementById("aiStreamImg");
                    if (aiImg) aiImg.src = url;
                    if (frameObjectUrl) URL.revokeObjectURL(frameObjectUrl);
                    frameObjectUrl = url;
                }
            } else if (res.status !== 204) {
                delay = 1000;
            }
        } catch (err) {
            delay = 1000;   // server restarting / network blip
        }
        if (active()) setTimeout(pollNext, delay);
    }

    pollNext();
}

document.addEventListener("visibilitychange", () => {
    // A hidden tab must not hold one of the tracker's few focus slots.
    if (document.hidden) {
        if (isPolling) {
            const camId = pollingCamId;
            stopFramePolling();
            pollingCamId = camId;   // remembered for resuming
        }
    } else {
        // Restart unconditionally: a page opened in a background tab started polling while hidden,
        // so its loop exited at once although isPolling stayed true, and the feed never appeared.
        if (pollingCamId !== null && streamMode === "ai") startFramePolling(pollingCamId);
        // Heartbeats keep arriving while hidden but are not drawn; catch up now.
        updateCameraMarkersCounts();
        renderActiveList();
    }
});

document.addEventListener("DOMContentLoaded", () => {
    if (window.lucide) lucide.createIcons();   // static icons only; lucide and leaflet are deferred scripts
    initMap();
    loadCameras();
    loadTopology();
    loadStats();
    setInterval(() => {
        // Skip a tick while the last fetch is still out (long ranges on a busy server)
        if (activeTab === "timeseries" && !chartRange.inFlight) {
            loadTimeSeriesMetrics(currentCameraId);
        }
    }, 2500);
    setupEventListeners();
    setupRangeSlider();
    initWebSocket();
});

// 1. Leaflet map centered on Kota Bogor
function initMap() {
    map = L.map("map", { zoomControl: true, attributionControl: true })
        .setView([-6.5971, 106.8060], 13);
    map.attributionControl.setPrefix(false);

    // Leaflet measures its box once at start-up and again only on window resize. When the layout settles
    // later (stylesheets, fonts, the phone layout), re-measure so tiles and markers fill the whole map.
    const mapEl = document.getElementById("map");
    if (window.ResizeObserver) new ResizeObserver(() => map.invalidateSize()).observe(mapEl);
    window.addEventListener("load", () => map.invalidateSize());

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
        drawTopology();
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
            // Content is built when the popup opens; while it stays open only its rows are refreshed
            // (refreshOpenPopup), never the whole popup.
            marker.bindPopup(() => popupHtml(cam, cameraRates[cam.id]));
            marker.on("popupopen", () => {
                openPopupCamId = cam.id;
                shownPopupRows = popupRowsHtml(cameraRates[cam.id]);
            });
            marker.on("popupclose", () => { if (openPopupCamId === cam.id) openPopupCamId = null; });
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

        drawTopology();
    } catch (err) {
        console.warn("Topology loading deferred:", err);
    }
}

// Directed road links between cameras, as one layer. Needs both the topology and the camera list,
// which load concurrently, so it is called after each of them.
let topologyLayer = null;
function drawTopology() {
    if (!map || !topologyData || !topologyData.edges || allCameras.length === 0 || topologyLayer) return;
    const byId = new Map(allCameras.map(c => [Number(c.id), c]));
    const segments = [];
    topologyData.edges.forEach(edge => {
        const a = byId.get(Number(edge.source_id));
        const b = byId.get(Number(edge.target_id));
        if (a && b && a.latitude && b.latitude) {
            segments.push([[a.latitude, a.longitude], [b.latitude, b.longitude]]);
        }
    });
    if (segments.length === 0) return;
    topologyLayer = L.polyline(segments, {
        color: cssVar("--accent", "#0071e3"), weight: 1.5, opacity: 0.4, dashArray: "4, 6", interactive: false,
    }).addTo(map);
}

// 4. Switch active camera and start its feed
function switchCamera(cameraId) {
    cameraId = Number(cameraId);
    currentCameraId = cameraId;
    const cam = allCameras.find(c => c.id === cameraId);
    if (!cam) return;

    document.querySelectorAll(".cam-marker").forEach(node => node.classList.remove("active-marker"));
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

    if (activeTab === "cameras") renderAllCamerasCountsList();
    updateExportScope();

    if (activeTab === "timeseries") {
        loadTimeSeriesMetrics(cameraId);
    }
}

// 5. Citywide statistics and per-camera rates

// Label and colour of one camera marker. Reads the DOM rather than a cache: a marker that was
// hidden by the corridor filter gets a fresh icon when it comes back.
function updateMarkerLabel(cam) {
    const rate = cameraRates[cam.id];
    setNodeText(document.getElementById(`count-${cam.id}`), fmtRate(rate ? rate.vph : null, rate && rate.low_sample));

    const markerDiv = document.getElementById(`marker-${cam.id}`);
    if (markerDiv) {
        const level = rateLevel(rate ? rate.vph : null);
        if (!markerDiv.classList.contains(level)) {
            markerDiv.classList.remove("level-zero", "level-low", "level-med", "level-high");
            markerDiv.classList.add(level);
        }
    }
}

// A map popup that is open keeps its node tree; only the numbers inside it are swapped, and only
// when they changed. (Closed popups build their content when they open.)
function refreshOpenPopup() {
    if (openPopupCamId === null) return;
    const marker = cameraMarkers[openPopupCamId];
    const popup = marker && marker.getPopup();
    const rowsEl = popup && popup.isOpen() && popup.getElement() && popup.getElement().querySelector(".popup__rows");
    if (!rowsEl) return;
    const html = popupRowsHtml(cameraRates[openPopupCamId]);
    if (html !== shownPopupRows) {
        rowsEl.innerHTML = html;
        shownPopupRows = html;
    }
}

function updateCameraMarkersCounts() {
    allCameras.forEach(cam => {
        if (cameraMarkers[cam.id]) updateMarkerLabel(cam);
    });
    refreshOpenPopup();
}

// Cameras tab: one row per camera, updated in place (see syncList). Busiest cameras first.
function createCameraRow(cam) {
    const row = makeEl("div", "row row--click");
    row.setAttribute("role", "button");
    row.tabIndex = 0;
    row.dataset.switchCamera = String(Number(cam.camera_id));

    const title = makeEl("div", "row__title");
    const name = makeEl("span", "truncate");
    title.append(name);

    const sub = makeEl("div", "row__sub");
    const idEl = makeEl("span");
    const corridor = makeEl("span", "truncate");
    corridor.style.maxWidth = "150px";
    sub.append(idEl, corridor);
    const chipTexts = VEHICLE_CLASSES.map(c => {
        const chip = makeEl("span", `cls cls--${c.key}`);
        const text = document.createTextNode("");
        chip.append(makeEl("i"), text);
        sub.append(chip);
        return text;
    });

    const main = makeEl("div", "row__main");
    main.append(title, sub);

    const rate = makeEl("span", "rate");
    const rateText = document.createTextNode("");
    rate.append(rateText, " ", makeEl("small", null, "veh/h"));
    const side = makeEl("div", "row__side");
    side.append(rate);

    row.append(main, side);
    return { el: row, title, name, idEl, corridor, chipTexts, rate, rateText, tag: null };
}

function updateCameraRow(rec, cam) {
    const rate = cameraRates[cam.camera_id];
    const isCurrent = cam.camera_id === currentCameraId;

    setNodeText(rec.name, cam.name || "");
    setNodeText(rec.idEl, `ID ${cam.camera_id}`);
    setNodeText(rec.corridor, cam.corridor || "–");
    VEHICLE_CLASSES.forEach((c, i) => {
        setTextData(rec.chipTexts[i], `${c.label} ${fmtRate(((rate && rate.vph_classes) || {})[c.key])}`);
    });

    const rateClass = `rate ${rateLevel(rate ? rate.vph : null)}`;
    if (rec.rate.className !== rateClass) rec.rate.className = rateClass;
    const title = rate ? `${Math.round(rate.observed_s)} s observed in the last hour` : "Not observed in the last hour";
    if (rec.rate.title !== title) rec.rate.title = title;
    setTextData(rec.rateText, fmtRate(rate ? rate.vph : null, rate && rate.low_sample));

    rec.el.classList.toggle("row--current", isCurrent);
    if (isCurrent) rec.el.setAttribute("aria-current", "true");
    else rec.el.removeAttribute("aria-current");
    if (isCurrent && !rec.tag) {
        rec.tag = makeEl("span", "tag", "VIEWING");
        rec.title.append(rec.tag);
    } else if (!isCurrent && rec.tag) {
        rec.tag.remove();
        rec.tag = null;
    }
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
        setListEmpty(container, cameraRows, cameraSearchFilter ? `No cameras match “${cameraSearchFilter}”.` : "No cameras loaded.");
        return;
    }

    syncList(container, cameraRows, items, cam => cam.camera_id, createCameraRow, updateCameraRow);
}

// Heartbeats update the data every 2 s, but only the visible list is drawn; the other one is drawn
// from the last data when its tab is opened (setTab).
function renderActiveList() {
    if (activeTab === "cameras") renderAllCamerasCountsList();
    else if (activeTab === "sightings") renderRecentSightings(lastSightings);
}

function setupTabs() {
    const tabs = document.querySelectorAll(".tab[data-tab]");
    const panels = {
        cameras: document.getElementById("tabContentCameras"),
        timeseries: document.getElementById("tabContentTimeSeries"),
        sightings: document.getElementById("tabContentSightings"),
    };

    function setTab(tab) {
        activeTab = tab;
        tabs.forEach(btn => btn.setAttribute("aria-selected", String(btn.dataset.tab === tab)));
        Object.entries(panels).forEach(([name, panel]) => { if (panel) panel.hidden = name !== tab; });
        if (tab === "timeseries") {
            refreshExportRange();
            loadTimeSeriesMetrics(currentCameraId);
        }
        renderActiveList();
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
    lastSightings = sightings;

    const classes = countsData.class_breakdown || {};
    const setNum = (id, value) => setNodeText(document.getElementById(id), (value || 0).toLocaleString());
    setNum("statTotalSightings", countsData.total_vehicles_counted);
    setNum("statCars", classes.car);
    setNum("statMotos", classes.motorcycle);
    setNum("statTrucks", classes.truck);
    setNum("statBuses", classes.bus);

    setNodeText(document.getElementById("headerVehicleCountText"), `≈${fmtRate(countsData.citywide_vph || 0)} veh/h citywide`);
    setNodeText(document.getElementById("camsWithTrafficCount"), String(countsData.cameras_with_rates || 0));

    // Nothing below is visible in a background tab; visibilitychange repaints it.
    if (document.hidden) return;
    updateCameraMarkersCounts();
    renderActiveList();
}

function createSightingRow(s) {
    const vtype = s.vehicle_class || s.vehicle_type || "car";
    let timeStr = "";
    if (s.timestamp_iso) timeStr = new Date(s.timestamp_iso).toLocaleTimeString();
    else if (s.timestamp) timeStr = new Date(s.timestamp * 1000).toLocaleTimeString();

    const row = makeEl("div", "row");

    const cls = makeEl("span", `cls cls--${vtype}`);
    cls.append(makeEl("i"), document.createTextNode(vtype));
    const sub = makeEl("div", "row__sub");
    sub.append(cls, makeEl("span", null, `ID ${s.camera_id}`));

    const main = makeEl("div", "row__main");
    main.append(makeEl("div", "row__title truncate", s.camera_name || `Camera ${s.camera_id}`), sub);
    row.append(main, makeEl("span", "muted num", timeStr));
    return { el: row };
}

// Recent tab: a sighting never changes once recorded, so rows are created for new sightings,
// dropped when they leave the newest-12 window, and otherwise left alone.
const sightingKey = s => s.sighting_id ?? `${s.camera_id}:${s.local_track_id}:${s.timestamp_iso || s.timestamp}`;

function renderRecentSightings(sightings) {
    const recentList = document.getElementById("recentSightingsList");
    if (!recentList) return;

    if (sightings.length === 0) {
        setListEmpty(recentList, sightingRows, "No vehicles recorded yet.");
        return;
    }
    syncList(recentList, sightingRows, sightings, sightingKey, createSightingRow, null);
}

// 6. Event listeners
function setupEventListeners() {
    document.getElementById("cameraSelect").addEventListener("change", (e) => {
        if (e.target.value) switchCamera(parseInt(e.target.value));
    });

    // Delegated actions for dynamically rendered content (lists, popups, modal)
    document.addEventListener("click", (e) => {
        const camBtn = e.target.closest("[data-switch-camera]");
        if (camBtn) {
            switchCamera(camBtn.dataset.switchCamera);
        }
    });
    document.addEventListener("keydown", (e) => {
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

    // Corridor filter
    document.querySelectorAll(".corridor-btn").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll(".corridor-btn").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
            activeCorridor = btn.getAttribute("data-corridor");
            filterMarkersByCorridor(activeCorridor);
        });
    });

    setupExport();
    setupTabs();
}

// 9b. Data export (Analytics tab). The Download button and the raw-table links are plain
// <a download> elements whose href is rebuilt from the form on every change, so they work with the
// keyboard, can be copied, and the server's Content-Disposition does the rest.
const exportState = { hours: 24 };   // active quick range; null once the dates were typed in

const pad2 = n => String(n).padStart(2, "0");
function toLocalInput(date) {
    return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())}T${pad2(date.getHours())}:${pad2(date.getMinutes())}`;
}

function setExportRange(hours) {
    const startInput = document.getElementById("exportStart");
    const endInput = document.getElementById("exportEnd");
    if (!startInput || !endInput) return;
    exportState.hours = hours;
    const end = new Date(Math.ceil(Date.now() / 60000) * 60000);   // next whole minute, so "now" is included
    startInput.value = toLocalInput(new Date(end.getTime() - hours * 3600 * 1000));
    endInput.value = toLocalInput(end);
    document.querySelectorAll("[data-export-hours]").forEach(btn => {
        btn.setAttribute("aria-pressed", String(Number(btn.dataset.exportHours) === hours));
    });
    updateExportLinks();
}

// A quick range is relative to "now", so it is recomputed whenever the tab is opened.
function refreshExportRange() {
    if (exportState.hours) setExportRange(exportState.hours);
}

function updateExportScope() {
    const option = document.querySelector('#exportScope option[value="current"]');
    if (option) {
        const cam = allCameras.find(c => c.id === currentCameraId);
        option.textContent = cam ? `Current camera (ID ${cam.id})` : "Current camera";
    }
    updateExportLinks();
}

// Times are sent as epoch seconds, so they mean what the date pickers show (the browser's time
// zone) whatever zone the server runs in.
function buildExportParams() {
    const startMs = new Date(document.getElementById("exportStart").value).getTime();
    const endMs = new Date(document.getElementById("exportEnd").value).getTime();
    if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return { error: "Choose a start and an end time." };
    if (endMs <= startMs) return { error: "The end time must be after the start time." };

    const params = new URLSearchParams({
        start: String(Math.floor(startMs / 1000)),
        end: String(Math.floor(endMs / 1000)),
    });
    if (document.getElementById("exportScope").value === "current") params.set("camera_ids", String(currentCameraId));
    return { params };
}

function updateExportLinks() {
    const form = document.getElementById("exportForm");
    if (!form) return;
    const msg = document.getElementById("exportMsg");
    const built = buildExportParams();
    const links = form.querySelectorAll("[data-export]");

    if (built.error) {
        links.forEach(a => { a.setAttribute("aria-disabled", "true"); a.setAttribute("href", "#"); });
        setNodeText(msg, built.error);
        return;
    }
    setNodeText(msg, "");

    const dataset = new URLSearchParams(built.params);
    dataset.set("bucket_minutes", document.getElementById("exportBucket").value);
    dataset.set("format", document.getElementById("exportFormat").value);
    dataset.set("layout", document.getElementById("exportLayout").value);
    dataset.set("fill_gaps", String(document.getElementById("exportFillGaps").checked));

    links.forEach(a => {
        a.removeAttribute("aria-disabled");
        if (a.dataset.export === "dataset") {
            a.href = `/api/export/dataset?${dataset}`;
        } else {
            const raw = new URLSearchParams(built.params);
            raw.set("table", a.dataset.table);
            raw.set("format", a.dataset.format);
            a.href = `/api/export/raw?${raw}`;
        }
    });
}

function setupExport() {
    const form = document.getElementById("exportForm");
    if (!form) return;

    const tz = Intl.DateTimeFormat().resolvedOptions().timeZone;
    setNodeText(document.getElementById("exportTz"), tz ? `Times are in your browser time zone (${tz}).` : "");

    form.addEventListener("input", (e) => {
        if (e.target.id === "exportStart" || e.target.id === "exportEnd") {
            exportState.hours = null;   // typed-in dates: no longer a quick range
            document.querySelectorAll("[data-export-hours]").forEach(btn => btn.setAttribute("aria-pressed", "false"));
        }
        updateExportLinks();
    });
    form.addEventListener("click", (e) => {
        const preset = e.target.closest("[data-export-hours]");
        if (preset) { setExportRange(Number(preset.dataset.exportHours)); return; }
        const link = e.target.closest("[data-export]");
        if (link && link.getAttribute("aria-disabled") === "true") e.preventDefault();   // invalid range
    });
    form.addEventListener("submit", (e) => {   // Enter inside a field starts the download
        e.preventDefault();
        const download = document.getElementById("exportDownload");
        if (download && download.getAttribute("aria-disabled") !== "true") download.click();
    });

    setExportRange(24);
    updateExportScope();
}

function filterMarkersByCorridor(corridor) {
    allCameras.forEach(cam => {
        const marker = cameraMarkers[cam.id];
        if (!marker) return;
        const show = corridor === "ALL" || (cam.corridor && cam.corridor.toLowerCase().includes(corridor.toLowerCase()));
        if (show && !map.hasLayer(marker)) {
            marker.addTo(map);   // builds a fresh icon from the markup it was created with
            updateMarkerLabel(cam);
            const div = document.getElementById(`marker-${cam.id}`);
            if (div) {
                div.classList.toggle("active-marker", cam.id === currentCameraId);
            }
        } else if (!show && map.hasLayer(marker)) {
            map.removeLayer(marker);
        }
    });
    // A single-corridor view pans to its cameras (Tasikmalaya is ~200 km from Bogor).
    if (corridor !== "ALL") {
        const pts = allCameras.filter(c => cameraMarkers[c.id] && map.hasLayer(cameraMarkers[c.id]))
            .map(c => [c.latitude, c.longitude]);
        if (pts.length) map.fitBounds(pts, { padding: [30, 30], maxZoom: 15 });
    }
}

// 7. WebSocket heartbeat drives stats refresh
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

// 8. Per-camera analytics
// The backend's get_congestion_label levels (Smooth < 0.25 <= Moderate < 0.55 <= Heavy < 0.8 <= Congested).
// The label itself comes from the API when it is there; these thresholds are only the fallback.
const CONGESTION_LEVELS = [
    { min: 0.8, label: "Congested", cls: "sev--severe", color: "var(--bad)" },
    { min: 0.55, label: "Heavy", cls: "sev--heavy", color: "var(--bad)" },
    { min: 0.25, label: "Moderate", cls: "sev--moderate", color: "var(--warn)" },
    { min: -Infinity, label: "Smooth", cls: "sev--free", color: "var(--ok)" },
];
function congestionLevel(index, label) {
    return CONGESTION_LEVELS.find(l => l.label === label)
        || CONGESTION_LEVELS.find(l => index >= l.min);
}

function fmtAgo(seconds) {
    if (!(seconds >= 0)) return "earlier";
    if (seconds < 90) return `${Math.round(seconds)} s ago`;
    if (seconds < 5400) return `${Math.round(seconds / 60)} min ago`;
    return `${(seconds / 3600).toFixed(seconds < 36000 ? 1 : 0)} h ago`;
}

function fmtDuration(seconds) {
    if (!(seconds > 0)) return "none";
    return seconds < 90 ? `${Math.round(seconds)} s` : `${(seconds / 60).toFixed(seconds < 600 ? 1 : 0)} min`;
}

// Timeframes of the Analytics chart's slider. Buckets grow with the window so the chart keeps a
// readable number of points (a camera is only observed during visits, so short buckets over long
// windows would be mostly gaps).
const CHART_RANGES = [
    { label: "1 h", long: "1 hour", minutes: 60, bucket: 5 },
    { label: "6 h", long: "6 hours", minutes: 360, bucket: 15 },
    { label: "24 h", long: "24 hours", minutes: 1440, bucket: 30 },
    { label: "7 d", long: "7 days", minutes: 10080, bucket: 120 },
    { label: "30 d", long: "30 days", minutes: 43200, bucket: 360 },
];
const chartRange = { preferred: 0, historyStart: {}, request: 0, inFlight: false };   // historyStart: camera id -> ts | null
try {
    const saved = Number(localStorage.getItem("chartRange"));
    if (Number.isInteger(saved) && saved >= 0 && saved < CHART_RANGES.length) chartRange.preferred = saved;
} catch (e) { /* storage unavailable */ }

// Longest timeframe worth offering for a camera: a range is only enabled while the next-shorter one
// doesn't already show all of its stored data. Unknown coverage (not fetched yet) allows everything.
function maxChartRange(cameraId) {
    if (!(cameraId in chartRange.historyStart)) return CHART_RANGES.length - 1;
    const start = chartRange.historyStart[cameraId];
    if (start === null || start === undefined) return 0;
    const spanMin = (Date.now() / 1000 - start) / 60;
    let max = 0;
    while (max + 1 < CHART_RANGES.length && CHART_RANGES[max].minutes < spanMin) max++;
    return max;
}

const activeChartRange = cameraId => Math.min(chartRange.preferred, maxChartRange(cameraId));

function syncRangeSlider(cameraId) {
    const input = document.getElementById("tsRangeInput");
    const ticks = document.getElementById("tsRangeTicks");
    if (!input || !ticks) return;
    const active = activeChartRange(cameraId), max = maxChartRange(cameraId);
    const last = CHART_RANGES.length - 1;
    if (!ticks.children.length) {
        CHART_RANGES.forEach((r, i) => {
            const b = document.createElement("button");
            b.type = "button";
            b.textContent = r.label;
            b.tabIndex = -1;
            b.style.left = `calc(var(--knob) / 2 + (100% - var(--knob)) * ${i / last})`;
            b.addEventListener("click", () => setChartRange(i));
            ticks.appendChild(b);
        });
    }
    [...ticks.children].forEach((b, i) => {
        b.setAttribute("aria-pressed", String(i === active));
        b.disabled = i > max;
        b.title = i > max ? "No data stored that far back for this camera" : "";
    });
    input.value = String(active);
    input.style.setProperty("--pct", `${(active / last) * 100}%`);
    input.setAttribute("aria-valuetext", CHART_RANGES[active].long);
}

function setChartRange(index) {
    const max = maxChartRange(currentCameraId);
    chartRange.preferred = Math.min(Math.max(index, 0), max);
    try { localStorage.setItem("chartRange", String(chartRange.preferred)); } catch (e) { /* storage unavailable */ }
    syncRangeSlider(currentCameraId);
    loadTimeSeriesMetrics(currentCameraId);
}

function setupRangeSlider() {
    const input = document.getElementById("tsRangeInput");
    if (!input) return;
    input.max = String(CHART_RANGES.length - 1);
    input.addEventListener("input", () => {
        const max = maxChartRange(currentCameraId);
        if (Number(input.value) > max) input.value = String(max);   // ranges beyond the stored data
        setChartRange(Number(input.value));
    });
    syncRangeSlider(currentCameraId);
}

const fmtSpan = minutes => minutes < 120 ? `${Math.round(minutes)} min`
    : minutes < 2880 ? `${Math.round(minutes / 60)} h` : `${+(minutes / 1440).toFixed(1)} d`;

async function loadTimeSeriesMetrics(cameraId) {
    try {
        const range = CHART_RANGES[activeChartRange(cameraId)];
        const request = ++chartRange.request;
        chartRange.inFlight = true;
        let res, data;
        try {
            res = await fetch(`/api/cameras/${cameraId}/metrics?minutes=${range.minutes}&bucket_minutes=${range.bucket}`);
            data = res.ok ? await res.json() : null;
        } finally {
            if (request === chartRange.request) chartRange.inFlight = false;
        }
        if (!data || request !== chartRange.request) return;   // a newer request (other range/camera) wins
        if (cameraId !== currentCameraId) return;
        const flow = data.flow || {};
        chartRange.historyStart[cameraId] = flow.history_start_ts ?? null;
        if (activeChartRange(cameraId) !== CHART_RANGES.indexOf(range)) {
            // The chosen range reaches past this camera's stored data: show the longest one it covers.
            syncRangeSlider(cameraId);
            loadTimeSeriesMetrics(cameraId);
            return;
        }
        syncRangeSlider(cameraId);

        const live = data.live || {};
        const history = data.recent_history || [];
        const lastRec = history.length > 0 ? history[history.length - 1] : null;
        const rate = data.rate || null;      // counting-line rate over the last hour; null if not observed
        const setText = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value; };

        setText("tsCamName", `${data.camera_name || "Camera"} (ID ${cameraId})`);

        // Flow: crossings of the camera's counting line per hour of observation
        setText("tsFlowVph", rate ? fmtRate(rate.vph, rate.low_sample) : "–");
        setText("tsFlowPerMin", rate ? (rate.vph / 60 < 10 ? (rate.vph / 60).toFixed(1) : Math.round(rate.vph / 60)) : "–");
        setText("tsObserved", rate ? fmtDuration(rate.observed_s) : "none");

        // Congestion: the live reading while the camera is being observed. Rolling cameras are only
        // watched for a short visit every few minutes; between visits the last reading is shown with
        // its age instead of passing for the current one.
        const isLive = live.timestamp !== undefined;
        const measured = isLive ? live : (data.last_measured || {});
        const hasReading = measured.congestion_index !== undefined;
        const cong = hasReading ? Number(measured.congestion_index) : 0;
        const ago = !isLive && hasReading ? ` · ${fmtAgo(data.live_age_s)}` : "";
        setText("tsCongestionIndex", hasReading ? cong.toFixed(2) : "–");

        // Lane occupancy (share of the camera's learned road area that traffic covers) once the camera's
        // lanes are learned; until then only the share of the whole picture is known.
        if (!hasReading) {
            setText("tsOccupancyPct", "not observed yet");
        } else if (measured.lane_occupancy !== undefined && measured.lane_occupancy !== null) {
            setText("tsOccupancyPct", `${(measured.lane_occupancy * 100).toFixed(0)}% of lanes occupied${ago}`);
        } else {
            setText("tsOccupancyPct", `${((measured.occupancy_ratio || 0) * 100).toFixed(1)}% of frame (learning lanes)${ago}`);
        }

        const level = congestionLevel(cong, measured.congestion_label);
        const barEl = document.getElementById("tsCongestionBar");
        if (barEl) {
            barEl.style.width = hasReading ? `${Math.min(Math.max(cong * 100, 5), 100)}%` : "0%";
            barEl.style.background = level.color;
            barEl.style.opacity = isLive ? "" : "0.45";
        }
        const badgeEl = document.getElementById("tsCongestionBadge");
        if (badgeEl) {
            badgeEl.textContent = hasReading ? level.label + ago : "No data";
            badgeEl.className = `sev ${hasReading ? level.cls : ""}`;
            badgeEl.title = isLive ? "Live" : (hasReading ? "Last measured on this camera's latest visit" : "");
        }

        // Classes: counting-line rate per class
        const classes = (rate && rate.vph_classes) || {};
        setText("tsCarCount", rate ? fmtRate(classes.car) : "–");
        setText("tsMotoCount", rate ? fmtRate(classes.motorcycle) : "–");
        setText("tsTruckCount", rate ? fmtRate(classes.truck) : "–");
        setText("tsBusCount", rate ? fmtRate(classes.bus) : "–");

        // Kinematics
        const speed = measured.avg_speed_px_s !== undefined ? measured.avg_speed_px_s : (lastRec ? lastRec.avg_speed_px_s : null);
        setText("tsSpeedProxy", speed === null ? "–" : Number(speed).toFixed(1));

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

        const minutes = flow.window_minutes || range.minutes;
        const series = flow.series || [];
        setText("tsHistoryPoints", `last ${range.long} · WIB`);
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

    const FLOW = cssVar("--chart-flow", "#0071e3"), CONGESTION = cssVar("--chart-congestion", "#ff9500");
    const MUTED = cssVar("--chart-text", "#8e8e93"), GRID = cssVar("--chart-grid", "rgba(60, 60, 67, 0.1)");
    const FLOW_FILL = cssVar("--chart-flow-fill", "rgba(0, 113, 227, 0.16)");
    const FONT = cssVar("--font", "system-ui, sans-serif");
    ctx.font = `500 10.5px ${FONT}`;

    const hasData = series.some(p => p.vph !== null && p.vph !== undefined || p.congestion !== null && p.congestion !== undefined);
    if (!hasData) {
        ctx.fillStyle = MUTED;
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = `12.5px ${FONT}`;
        ctx.fillText(`No observations in the last ${fmtSpan(minutes)} yet`, w / 2, h / 2);
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
                    grad.addColorStop(0, FLOW_FILL);
                    grad.addColorStop(1, "rgba(0, 0, 0, 0)");
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

    // Time axis: clock times in WIB, at round steps chosen so the labels don't collide
    const { step, label } = wibTicks(tEnd - tStart, chartW, ctx);
    ctx.textBaseline = "top";
    ctx.textAlign = "center";
    const WIB = 7 * 3600;
    for (let t = Math.ceil((tStart + WIB) / step) * step - WIB; t <= tEnd; t += step) {
        const x = xAt(t), text = label(t), half = ctx.measureText(text).width / 2;
        if (x - half < 0 || x + half > w) continue;      // would be cut off at the canvas edge
        ctx.strokeStyle = GRID;
        ctx.beginPath();
        ctx.moveTo(x, pad.top + chartH);
        ctx.lineTo(x, pad.top + chartH + 3);
        ctx.stroke();
        ctx.fillStyle = MUTED;
        ctx.fillText(text, x, pad.top + chartH + 6);
    }
}

// Tick step (seconds) and label format for a time axis spanning `spanS` seconds over `widthPx`.
// Times are WIB (UTC+7, no daylight saving) whatever the viewer's own time zone: the cameras are in
// Bogor. Steps under a day read "14:00"; day steps read "Mon 13" (a week) or "13 Oct" (longer).
const WIB_OFFSET_S = 7 * 3600;
const TICK_STEPS_S = [300, 600, 900, 1800, 3600, 7200, 10800, 14400, 21600, 43200, 86400, 172800, 432000, 604800];
const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function wibTicks(spanS, widthPx, ctx) {
    const wib = t => new Date((t + WIB_OFFSET_S) * 1000);   // read with getUTC* = WIB wall clock
    const clock = t => { const d = wib(t); return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}`; };
    const weekday = t => { const d = wib(t); return `${WEEKDAYS[d.getUTCDay()]} ${d.getUTCDate()}`; };
    const date = t => { const d = wib(t); return `${d.getUTCDate()} ${MONTHS[d.getUTCMonth()]}`; };
    const formatFor = step => step < 86400 ? clock : (spanS <= 8 * 86400 ? weekday : date);
    for (const step of TICK_STEPS_S) {
        const minGap = ctx.measureText(formatFor(step)(0) + "0").width + 16;
        if (step < 86400 && spanS > 3 * 86400) continue;     // multi-day ranges tick on whole days
        if ((step / spanS) * widthPx >= minGap) return { step, label: formatFor(step) };
    }
    const step = TICK_STEPS_S[TICK_STEPS_S.length - 1];
    return { step, label: formatFor(step) };
}
