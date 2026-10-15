// City tab: citywide patterns from /api/insights/citywide (hour of day, corridors, junctions, vehicle mix).
// Uses app.js globals: makeEl, setNodeText, syncList, congestionLevel, switchCamera.
// Charts are SVG/HTML styled by theme tokens in styles.css, so light/dark needs no redraw. Elements are
// kept between refreshes and only their geometry changes, so updates transition instead of flashing.

(() => {
    "use strict";

    const NS = "http://www.w3.org/2000/svg";
    const REFRESH_MS = 60000;              // the server caches the aggregate for 2 min
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const CORRIDOR_NAMES = {          // [full, short for phone-width rows]
        SSA_RING: ["SSA Kebun Raya", "SSA ring"],
        PAJAJARAN_CORRIDOR: ["Pajajaran", "Pajajaran"],
        SHOLEH_ISKANDAR_BORR: ["Jl. Baru (Sholeh Iskandar)", "Jl. Baru"],
        BOGOR_TIMUR_TAJUR_KATULAMPA: ["Tajur – Katulampa", "Tajur"],
        BOGOR_BARAT_DRAMAGA: ["Dramaga – Bubulak", "Dramaga"],
        BOGOR_SELATAN_BONDONGAN_PAHLAWAN: ["Pahlawan – Bondongan", "Pahlawan"],
        STASIUN_MERDEKA_MAWAR_SEMERU: ["Stasiun – Merdeka", "Stasiun"],
        SURKEN_PASAR_BOGOR: ["Suryakencana – Pasar", "Suryakencana"],
        TAMAN_KOTA_PARKS: ["City parks", "Parks"],
    };
    // bottom to top in the mix chart: motorcycles first, the class whose swing is the story
    const MIX = [
        { key: "motorcycle", label: "Motorcycles", color: "var(--c-moto)" },
        { key: "car", label: "Cars", color: "var(--c-car)" },
        { key: "truck", label: "Trucks", color: "var(--c-truck)" },
        { key: "bus", label: "Buses", color: "var(--c-bus)" },
    ];

    let data = null;
    let timer = null;
    let inflight = false;
    let visible = false;
    let rankMode = "congested";
    let lastWidth = 0;
    let readoutHour = null;
    const rankRows = new Map();
    const heatRows = new Map();
    const mixRows = new Map();
    const $ = id => document.getElementById(id);

    // ── helpers ──────────────────────────────────────────────────────────
    function svg(tag, attrs, parent) {
        const n = document.createElementNS(NS, tag);
        setA(n, attrs);
        if (parent) parent.append(n);
        return n;
    }
    function setA(n, attrs) {
        for (const [k, v] of Object.entries(attrs || {})) n.setAttribute(k, v);
        return n;
    }
    const r1 = v => Math.round(v * 10) / 10;
    const hh = h => `${String(h).padStart(2, "0")}:00`;
    const fmtInt = v => (v === null || v === undefined || !isFinite(v)) ? "–" : Math.round(v).toLocaleString("en-US");
    const pct = v => (v === null || v === undefined) ? "–" : `${Math.round(v * 100)}%`;
    let narrow = false;               // phone width: short corridor names in the row labels
    const corridorName = (c, short) => (CORRIDOR_NAMES[c] || [])[short ? 1 : 0] || String(c || "Other").replace(/_/g, " ").toLowerCase().replace(/\b\w/g, m => m.toUpperCase());
    function cameraName(name) {
        const s = String(name || "").replace(/^CCTV\s*-?\s*/i, "").trim().toLowerCase();
        return s.replace(/(^|[\s(\-])(\w)/g, (m, p, c) => p + c.toUpperCase()).replace(/\s+-[a-z]$/i, "");   // drop the "-F" / "-P" camera-type suffix
    }
    function argmax(items, value, from = 0, to = items.length - 1) {
        let best = null;
        for (let i = from; i <= to; i++) {
            const v = value(items[i]);
            if (v !== null && v !== undefined && (best === null || v > value(items[best]))) best = i;
        }
        return best;
    }
    function niceMax(v) {
        const p = Math.pow(10, Math.floor(Math.log10(v)));
        return [1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10].map(m => m * p).find(m => m >= v) || v;
    }
    const fmtAxis = v => v >= 1000 ? `${r1(v / 1000)}k` : String(Math.round(v));

    // Counts a number up/down to its new value (instant with reduced motion).
    function tween(el, to, fmt) {
        const from = el.dataset.v === undefined || el.dataset.v === "" ? 0 : Number(el.dataset.v);
        el.dataset.v = to ?? "";
        cancelAnimationFrame(el._raf);
        if (to === null || to === undefined || reduceMotion.matches || from === to) { setNodeText(el, fmt(to)); return; }
        const t0 = performance.now(), dur = 800;
        const step = now => {
            const p = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - p, 3);
            setNodeText(el, fmt(from + (to - from) * e));
            if (p < 1) el._raf = requestAnimationFrame(step);
        };
        el._raf = requestAnimationFrame(step);
    }

    // Monotone cubic (Fritsch–Carlson) through contiguous points: smooth, no overshoot below zero.
    function smooth(pts) {
        const n = pts.length;
        if (!n) return "";
        let d = `M${r1(pts[0][0])},${r1(pts[0][1])}`;
        if (n === 1) return d;
        const dx = [], m = [], t = [];
        for (let i = 0; i < n - 1; i++) { dx[i] = pts[i + 1][0] - pts[i][0]; m[i] = (pts[i + 1][1] - pts[i][1]) / dx[i]; }
        t[0] = m[0]; t[n - 1] = m[n - 2];
        for (let i = 1; i < n - 1; i++) t[i] = m[i - 1] * m[i] <= 0 ? 0 : (m[i - 1] + m[i]) / 2;
        for (let i = 0; i < n - 1; i++) {
            if (m[i] === 0) { t[i] = t[i + 1] = 0; continue; }
            const a = t[i] / m[i], b = t[i + 1] / m[i], s = a * a + b * b;
            if (s > 9) { const k = 3 / Math.sqrt(s); t[i] = k * a * m[i]; t[i + 1] = k * b * m[i]; }
        }
        for (let i = 0; i < n - 1; i++) {
            const h = dx[i] / 3;
            d += `C${r1(pts[i][0] + h)},${r1(pts[i][1] + t[i] * h)} ${r1(pts[i + 1][0] - h)},${r1(pts[i + 1][1] - t[i + 1] * h)} ${r1(pts[i + 1][0])},${r1(pts[i + 1][1])}`;
        }
        return d;
    }
    // Contiguous runs of hours that have a value (unobserved hours break the line).
    function runs(values) {
        const out = [];
        let cur = null;
        values.forEach((v, h) => {
            if (v === null || v === undefined) { cur = null; return; }
            if (!cur) out.push(cur = []);
            cur.push(h);
        });
        return out;
    }

    // The plot is revealed left to right (time passing) by widening its clip rectangle. Animated from
    // JS: CSS transforms on <clipPath> children don't repaint reliably.
    function setClip(box, p, x, width, height) {
        p.clipW = width;
        const waiting = box.closest(".city-card--wait");
        setA(p.clipRect, { x, y: 0, height, width: waiting ? 0 : width });
        if (!box._reveal) box._reveal = () => {
            const t0 = performance.now(), dur = 1200;
            const step = now => {
                const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
                box._parts.clipRect.setAttribute("width", box._parts.clipW * e);
                if (k < 1) requestAnimationFrame(step);
            };
            requestAnimationFrame(step);
        };
    }

    // ── hero ─────────────────────────────────────────────────────────────
    function renderHero() {
        const H = data.hours;
        const am = argmax(H, h => h.vph, 4, 11), pm = argmax(H, h => h.vph, 12, 20);
        const slow = argmax(H, h => h.slow_share);
        const title = $("cityTitle"), lede = $("cityLede");
        if (am === null || pm === null || slow === null) {
            setNodeText(title, "Still learning the city's rhythm");
            setNodeText(lede, "Not enough observed hours yet. Patterns appear once every hour of the day has been watched.");
        } else {
            const amSlow = H[am].slow_share || 0, pmSlow = H[pm].slow_share || 0;
            const other = slow === am ? pm : am;      // the rush hour to compare the slowest hour with
            const similar = Math.abs(H[am].vph - H[pm].vph) / Math.max(H[am].vph, H[pm].vph) < 0.15;
            setNodeText(title, pmSlow >= 1.4 * amSlow ? "Afternoons are the slowest"
                : amSlow >= 1.4 * pmSlow ? "Mornings are the slowest" : `Rush hours at ${hh(am)} and ${hh(pm)}`);
            setNodeText(lede, `${similar ? "Mornings and afternoons carry about the same traffic" : "Traffic peaks twice"}: ` +
                `${fmtInt(H[am].vph)} veh/h per camera at ${hh(am)} and ${fmtInt(H[pm].vph)} at ${hh(pm)}. ` +
                `But ${pct(H[slow].slow_share)} of junctions run slow at ${hh(slow)}, against ${pct(H[other].slow_share)} at ${hh(other)}.`);
        }
        const now = data.now || {};
        tween($("cityNowVph"), now.vph, fmtInt);
        const delta = $("cityNowDelta");
        if (now.vph && now.typical_vph) {
            const d = now.vph / now.typical_vph - 1;
            setNodeText(delta, Math.abs(d) < 0.02 ? `about usual for ${hh(data.now_hour)}`
                : `${d > 0 ? "▲" : "▼"} ${Math.abs(Math.round(d * 100))}% vs usual at ${hh(data.now_hour)}`);
            delta.className = `metric__foot city-delta ${Math.abs(d) < 0.1 ? "" : d > 0 ? "city-delta--up" : "city-delta--down"}`;
        } else {
            setNodeText(delta, now.vph ? `usual for ${hh(data.now_hour)} not known yet` : "no camera watched in the last hour");
            delta.className = "metric__foot city-delta";
        }
        for (const [id, h] of [["cityAm", am], ["cityPm", pm]]) {
            setNodeText($(`${id}Hour`), h === null ? "–" : hh(h));
            setNodeText($(`${id}Note`), h === null ? "" : `${fmtInt(H[h].vph)} veh/h · ${pct(H[h].slow_share)} slow`);
            // the slower of the two rush hours is the one to plan around
            $(`${id}Hour`).closest(".metric").classList.toggle("metric--warn", h !== null && h === (
                (H[am]?.slow_share || 0) >= (H[pm]?.slow_share || 0) ? am : pm));
        }
    }

    // ── a day on the roads: usual flow (area), today (line), share of slow junctions (strip) ──
    function renderDay(w) {
        const box = $("cityDay");
        const narrow = w < 480;
        const H = 214, L = 34, R = 8, T = 22, plotH = 118, gap = 14, stripH = 34;
        const yStrip = T + plotH + gap, x0 = L, x1 = w - R, bw = (x1 - x0) / 24;
        const X = h => x0 + (h + 0.5) * bw;
        const hours = data.hours;
        const maxV = niceMax(Math.max(100, ...hours.map(h => Math.max(h.vph || 0, h.today.vph || 0))));
        const Y = v => T + plotH * (1 - v / maxV);
        const maxShare = Math.max(0.2, ...hours.map(h => h.slow_share || 0));

        let p = box._parts;
        if (!p) {
            const s = svg("svg", { class: "cx", role: "img", "aria-label": "Usual vehicles per hour per camera, today's flow, and the share of slow junctions, by hour of day" });
            const defs = svg("defs", {}, s);
            const clip = svg("clipPath", { id: "cityDayClip" }, defs);
            p = {
                s, clipRect: svg("rect", { class: "cx-reveal" }, clip),
                deco: svg("g", {}, s),
                plot: svg("g", { "clip-path": "url(#cityDayClip)" }, s),
                bars: svg("g", {}, s), marks: svg("g", {}, s), hover: svg("g", {}, s),
            };
            p.area = svg("path", { class: "cx-area" }, p.plot);
            p.base = svg("path", { class: "cx-line cx-draw", pathLength: 1 }, p.plot);
            p.today = svg("path", { class: "cx-today cx-draw", pathLength: 1 }, p.plot);
            p.barEls = Array.from({ length: 24 }, (_, h) => svg("rect", { class: "cx-bar", style: `--i:${h}` }, p.bars));
            box.replaceChildren(s);
            box._parts = p;
        }
        setA(p.s, { viewBox: `0 0 ${w} ${H}`, width: w, height: H });
        setClip(box, p, x0, x1 - x0, H);

        // static decorations: night shade, grid, axes
        const deco = [];
        for (const [a, b] of [[0, 6], [18, 24]]) deco.push(svg("rect", { class: "cx-night", x: x0 + a * bw, y: T - 6, width: (b - a) * bw, height: yStrip + stripH - T + 6 }));
        for (const v of [0, maxV / 2, maxV]) {
            deco.push(svg("line", { class: "cx-grid", x1: x0, x2: x1, y1: r1(Y(v)), y2: r1(Y(v)) }));
            const t = svg("text", { class: "cx-axis", x: x0 - 6, y: Y(v) + 3.5, "text-anchor": "end" });
            t.textContent = fmtAxis(v);
            deco.push(t);
        }
        const sl = svg("text", { class: "cx-axis", x: x0 - 6, y: yStrip + stripH - 2, "text-anchor": "end" });
        sl.textContent = "slow";
        deco.push(sl, svg("line", { class: "cx-grid", x1: x0, x2: x1, y1: yStrip + stripH + 0.5, y2: yStrip + stripH + 0.5 }));
        for (let h = 0; h <= 24; h += narrow ? 6 : 3) {
            const t = svg("text", { class: "cx-axis", x: x0 + h * bw, y: H - 4, "text-anchor": h === 0 ? "start" : h === 24 ? "end" : "middle" });
            t.textContent = String(h % 24).padStart(2, "0");
            deco.push(t);
        }
        p.deco.replaceChildren(...deco);

        // usual flow: area + line
        const base = hours.map(h => h.vph);
        let area = "", line = "";
        for (const run of runs(base)) {
            const pts = run.map(h => [X(h), Y(base[h])]);
            const d = smooth(pts);
            line += d;
            area += `${d}L${r1(X(run[run.length - 1]))},${Y(0)}L${r1(X(run[0]))},${Y(0)}Z`;
        }
        p.area.setAttribute("d", area || "M0,0");
        p.base.setAttribute("d", line || "M0,0");
        const today = hours.map(h => h.today.vph);
        p.today.setAttribute("d", runs(today).map(run => smooth(run.map(h => [X(h), Y(today[h])]))).join("") || "M0,0");

        // slow junctions: one bar per hour, height via scaleY so changes transition everywhere
        const slowPeak = argmax(hours, h => h.slow_share);
        p.barEls.forEach((el, h) => {
            const share = hours[h].slow_share;
            setA(el, { x: r1(X(h) - bw * 0.34), width: r1(bw * 0.68), y: yStrip, height: stripH, rx: Math.min(3, bw * 0.2) });
            el.style.transform = `scaleY(${share ? Math.max(0.04, share / maxShare) : 0})`;
            el.classList.toggle("is-peak", h === slowPeak);
        });

        // marks: now line, peak labels
        const marks = [];
        const now = data.now_hour;
        marks.push(svg("line", { class: "cx-now", x1: r1(X(now)), x2: r1(X(now)), y1: T - 6, y2: yStrip + stripH }));
        const nowT = svg("text", { class: "cx-now-label", x: X(now), y: T - 10, "text-anchor": X(now) > x1 - 20 ? "end" : "middle" });
        nowT.textContent = "now";
        marks.push(nowT);
        const am = argmax(hours, h => h.vph, 4, 11), pm = argmax(hours, h => h.vph, 12, 20);
        for (const h of [am, pm]) {
            if (h === null) continue;
            marks.push(svg("circle", { class: "cx-dot", cx: r1(X(h)), cy: r1(Y(base[h])), r: 3.5 }));
            const t = svg("text", { class: "cx-peak", x: X(h), y: Y(base[h]) - 8,
                "text-anchor": X(h) < x0 + 40 ? "start" : X(h) > x1 - 40 ? "end" : "middle" });
            t.textContent = narrow ? hh(h) : `${hh(h)} · ${fmtInt(base[h])}`;
            marks.push(t);
        }
        if (slowPeak !== null) {
            const t = svg("text", { class: "cx-peak cx-peak--warn", x: X(slowPeak), y: yStrip - 3, "text-anchor": "middle" });
            t.textContent = pct(hours[slowPeak].slow_share);
            marks.push(t);
        }
        p.marks.replaceChildren(...marks);

        // hover columns: a readout under the chart (and a title for screen readers / touch)
        const cols = hours.map((hr, h) => {
            const c = svg("rect", { class: "cx-col", x: x0 + h * bw, y: T - 6, width: bw, height: yStrip + stripH - T + 6 });
            const tt = svg("title", {}, c);
            tt.textContent = readout(h);
            c.addEventListener("mouseenter", () => setReadout(h));
            return c;
        });
        p.hover.replaceChildren(...cols);
        p.hover.onmouseleave = () => setReadout(null);
        setReadout(readoutHour);
    }
    function readout(h) {
        const hr = data.hours[h];
        const today = hr.today.vph !== null ? ` · today ${fmtInt(hr.today.vph)}` : "";
        const slow = hr.slow_share !== null ? ` · ${pct(hr.slow_share)} of junctions slow` : "";
        return `${hh(h)}: usual ${fmtInt(hr.vph)} veh/h per camera${today}${slow} (${hr.n_days} day${hr.n_days === 1 ? "" : "s"})`;
    }
    function setReadout(h) {
        readoutHour = h;
        setNodeText($("cityDayReadout"), readout(h === null ? data.now_hour : h));
    }

    // ── corridor × hour heat map ─────────────────────────────────────────
    function heatColor(cell) {
        if (!cell || cell.ci === null || cell.analytics_s < 60) return null;
        return congestionLevel(cell.label).color;
    }
    function createHeatRow(c) {
        const el = makeEl("div", "heat__row");
        const name = makeEl("span", "heat__name truncate");
        const cells = Array.from({ length: 24 }, (_, h) => {
            const i = makeEl("i");
            i.style.setProperty("--i", h);
            return i;
        });
        el.append(name, ...cells);
        return { el, name, cells };
    }
    function updateHeatRow(rec, c) {
        setNodeText(rec.name, corridorName(c.corridor, narrow));
        rec.name.title = c.peak_hour === null ? corridorName(c.corridor)
            : `${corridorName(c.corridor)}: worst around ${hh(c.peak_hour)} (${c.cells[c.peak_hour].label})`;
        rec.cells.forEach((el, h) => {
            const cell = c.cells[h], color = heatColor(cell);
            // colour = the API's label; strength = the index, so a high "Smooth" reads warmer than a low one
            el.style.background = color || "var(--fill)";
            el.style.opacity = color ? (0.15 + 0.85 * Math.min(1, cell.ci / 0.6)) * (cell.analytics_s < 600 ? 0.6 : 1) : 1;
            el.title = color ? `${corridorName(c.corridor)}, ${hh(h)}: ${cell.label} (${cell.ci.toFixed(2)})` + (cell.analytics_s < 600 ? ", few readings" : "")
                : `${corridorName(c.corridor)}, ${hh(h)}: not measured`;
        });
    }
    function renderHeat(narrow) {
        const head = $("cityHeatHours");
        if (!head.childElementCount) {
            head.append(makeEl("span"));
            for (let h = 0; h < 24; h++) head.append(makeEl("span", "", h % 3 === 0 ? String(h).padStart(2, "0") : ""));
        }
        head.classList.toggle("heat__hours--narrow", narrow);
        const rows = data.corridors.filter(c => c.cells.some(x => x.ci !== null));
        syncList($("cityHeatRows"), heatRows, rows, c => c.corridor, createHeatRow, updateHeatRow);
        const worst = rows.find(c => c.peak_hour !== null);
        setNodeText($("cityHeatNote"), worst
            ? `${corridorName(worst.corridor)} is the slowest corridor, worst around ${hh(worst.peak_hour)}.` : "");
    }

    // ── junction ranking ─────────────────────────────────────────────────
    function createRankRow(r) {
        const el = makeEl("button", "rank__row");
        el.type = "button";
        const n = makeEl("span", "rank__n");
        const main = makeEl("span", "rank__main");
        const name = makeEl("span", "rank__name truncate");
        const sub = makeEl("span", "rank__sub truncate");
        main.append(name, sub);
        const bar = makeEl("span", "rank__bar");
        const fill = makeEl("i");
        const tick = makeEl("u");
        bar.append(fill, tick);
        const value = makeEl("b", "rank__v");
        const pill = makeEl("span", "sev rank__pill");
        el.append(n, main, bar, value, pill);
        el.addEventListener("click", () => {
            if (typeof switchCamera === "function") switchCamera(Number(el.dataset.cam));
        });
        return { el, n, name, sub, fill, tick, value, pill };
    }
    function updateRankRow(rec, r, i, max) {
        rec.el.dataset.cam = r.camera_id;
        rec.el.style.setProperty("--i", i);
        setNodeText(rec.n, String(i + 1));
        setNodeText(rec.name, cameraName(r.name));
        const bits = [corridorName(r.corridor, narrow)];
        if (rankMode === "congested") {
            if (r.worst_hour !== null) bits.push(`worst ${hh(r.worst_hour)}`);
            tween(rec.value, r.congestion, v => v === null ? "–" : v.toFixed(2));
            const lvl = congestionLevel(r.congestion_label);
            rec.pill.className = `sev rank__pill ${lvl.cls}`;
            setNodeText(rec.pill, lvl.label);
            rec.fill.style.background = lvl.color;
            rec.fill.style.transform = `scaleX(${r.congestion || 0})`;   // the index is 0..1: absolute scale
            rec.tick.hidden = r.worst_congestion === null;
            rec.tick.style.left = `${Math.min(1, r.worst_congestion || 0) * 100}%`;
            rec.tick.title = r.worst_hour === null ? "" : `worst hour ${hh(r.worst_hour)}: ${r.worst_congestion.toFixed(2)}`;
        } else {
            tween(rec.value, r.vph, fmtInt);
            rec.pill.className = "sev rank__pill";
            setNodeText(rec.pill, "veh/h");
            rec.fill.style.background = "var(--chart-flow)";
            rec.fill.style.transform = `scaleX(${max ? r.vph / max : 0})`;
            rec.tick.hidden = true;
        }
        setNodeText(rec.sub, bits.join(" · "));
        rec.el.title = `Show ${cameraName(r.name)} (camera ${r.camera_id}` +
            (r.junction_size > 1 ? `, busiest of a ${r.junction_size}-camera junction)` : ")");
    }
    function renderRank() {
        const list = data[rankMode] || [];
        const key = rankMode === "congested" ? "congestion" : "vph";
        const max = Math.max(0, ...list.map(r => r[key] || 0));
        if (!list.length) { setListEmpty($("cityRankList"), rankRows, "Not enough observed junctions yet."); return; }
        let i = 0;
        // rows are keyed by mode too, so switching modes re-runs the bars' grow-in
        syncList($("cityRankList"), rankRows, list, r => `${rankMode}-${r.camera_id}`, createRankRow,
            (rec, r) => updateRankRow(rec, r, i++, max));
    }

    // ── vehicle mix by hour + by corridor ────────────────────────────────
    function renderMix(w) {
        const box = $("cityMix");
        const narrow = w < 480;
        const H = 150, L = 34, R = 8, T = 22, plotH = 100;
        const x0 = L, x1 = w - R, bw = (x1 - x0) / 24;
        const X = h => x0 + (h + 0.5) * bw;
        const Y = v => T + plotH * (1 - v);
        let p = box._parts;
        if (!p) {
            const s = svg("svg", { class: "cx", role: "img", "aria-label": "Share of each vehicle class by hour of day" });
            const clip = svg("clipPath", { id: "cityMixClip" }, svg("defs", {}, s));
            p = { s, clipRect: svg("rect", { class: "cx-reveal" }, clip), deco: svg("g", {}, s) };
            const plot = svg("g", { "clip-path": "url(#cityMixClip)" }, s);
            p.layers = MIX.map(m => svg("path", { class: "cx-layer", style: `fill:${m.color}` }, plot));
            p.marks = svg("g", {}, s);
            box.replaceChildren(s);
            box._parts = p;
        }
        setA(p.s, { viewBox: `0 0 ${w} ${H}`, width: w, height: H });
        setClip(box, p, x0, x1 - x0, H);

        const deco = [];
        for (const [a, b] of [[0, 6], [18, 24]]) deco.push(svg("rect", { class: "cx-night", x: x0 + a * bw, y: T, width: (b - a) * bw, height: plotH }));
        for (const v of [0, 0.5, 1]) {
            const t = svg("text", { class: "cx-axis", x: x0 - 6, y: Y(v) + 3.5, "text-anchor": "end" });
            t.textContent = `${v * 100}%`;
            deco.push(t);
        }
        for (let h = 0; h <= 24; h += narrow ? 6 : 3) {
            const t = svg("text", { class: "cx-axis", x: x0 + h * bw, y: H - 4, "text-anchor": h === 0 ? "start" : h === 24 ? "end" : "middle" });
            t.textContent = String(h % 24).padStart(2, "0");
            deco.push(t);
        }
        p.deco.replaceChildren(...deco);

        const hours = data.hours;
        const has = hours.map(h => h.mix && h.mix.car !== null ? 1 : null);
        const lower = hours.map(() => 0);
        MIX.forEach((m, k) => {
            let d = "";
            for (const run of runs(has)) {
                const upper = run.map(h => lower[h] + (hours[h].mix[m.key] || 0));
                const top = run.map((h, j) => `${j ? "L" : "M"}${r1(X(h))},${r1(Y(upper[j]))}`).join("");
                const bottom = run.slice().reverse().map(h => `L${r1(X(h))},${r1(Y(lower[h]))}`).join("");
                d += `${top}${bottom}Z`;
                run.forEach((h, j) => { lower[h] = upper[j]; });
            }
            p.layers[k].setAttribute("d", d || "M0,0");
        });

        const marks = [];
        const peak = argmax(hours, h => h.mix ? h.mix.motorcycle : null);
        if (peak !== null) {
            const share = hours[peak].mix.motorcycle;
            marks.push(svg("circle", { class: "cx-dot", cx: r1(X(peak)), cy: r1(Y(share)), r: 3.5 }));
            const t = svg("text", { class: "cx-peak", x: X(peak), y: Y(share) - 8, "text-anchor": X(peak) < x0 + 60 ? "start" : "middle" });
            t.textContent = `${pct(share)} motorcycles at ${hh(peak)}`;
            marks.push(t);
            const low = argmax(hours, h => h.mix && h.mix.motorcycle !== null ? -h.mix.motorcycle : null, 9, 23);
            setNodeText($("cityMixNote"), `Motorcycles are ${pct(share)} of traffic at ${hh(peak)}` +
                (low !== null ? `, ${pct(hours[low].mix.motorcycle)} at ${hh(low)}.` : "."));
        }
        p.marks.replaceChildren(...marks);

        const rows = data.corridors.filter(c => c.mix && c.mix.car !== null)
            .sort((a, b) => b.mix.motorcycle - a.mix.motorcycle);
        syncList($("cityMixRows"), mixRows, rows, c => c.corridor, c => {
            const el = makeEl("div", "mixrow");
            const name = makeEl("span", "mixrow__name truncate");
            const bar = makeEl("span", "mixrow__bar");
            const segs = MIX.map(m => {
                const s = makeEl("i");
                s.style.background = m.color;
                bar.append(s);
                return s;
            });
            const v = makeEl("b", "mixrow__v");
            el.append(name, bar, v);
            return { el, name, segs, v };
        }, (rec, c) => {
            setNodeText(rec.name, corridorName(c.corridor, narrow));
            MIX.forEach((m, k) => { rec.segs[k].style.width = `${(c.mix[m.key] || 0) * 100}%`; });
            rec.el.title = MIX.map(m => `${m.label} ${pct(c.mix[m.key])}`).join(", ");
            setNodeText(rec.v, pct(c.mix.motorcycle));
        });
    }

    // ── orchestration ────────────────────────────────────────────────────
    function render() {
        if (!data || !visible) return;
        const w = Math.floor($("cityDay").clientWidth);
        if (!w) return;
        lastWidth = w;
        narrow = w < 480;
        renderHero();
        renderDay(w);
        renderHeat(w < 480);
        renderRank();
        renderMix(w);
        setNodeText($("cityFoot"),
            `Based on ${fmtInt(data.observed_hours)} camera-hours watched over the last ${data.baseline_days} day${data.baseline_days === 1 ? "" : "s"} ` +
            `and today (${data.cameras} cameras, WIB). "Usual" is the median camera on ${data.baseline_kind === "all" ? "past days" : `past ${data.baseline_kind}s`}; "slow" is Moderate or worse. Each camera is watched only ` +
            `during visits, so figures are rates over watched time. After dark (shaded), small vehicles are detected less reliably.`);
        $("tabContentCity").classList.remove("city--loading");
        if (!observer) armIntros();
    }

    async function load() {
        if (inflight) return;
        inflight = true;
        try {
            const res = await fetch("/api/insights/citywide", { signal: AbortSignal.timeout(10000) });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            data = await res.json();
            render();
        } catch (err) {
            console.error("City tab:", err);
            if (!data) setNodeText($("cityTitle"), "City patterns are unavailable right now");
        } finally {
            inflight = false;
        }
    }

    // Each card plays its intro the first time it scrolls into view.
    let observer = null;
    function armIntros() {
        const cards = document.querySelectorAll("#tabContentCity .city-card");
        if (reduceMotion.matches || !("IntersectionObserver" in window)) return;
        observer = new IntersectionObserver(entries => {
            for (const e of entries) {
                if (!e.isIntersecting) continue;
                // two frames: the first paints the start state, the second starts the transitions
                requestAnimationFrame(() => requestAnimationFrame(() => {
                    e.target.classList.remove("city-card--wait");
                    e.target.querySelectorAll(".city-chart").forEach(c => c._reveal && c._reveal());
                }));
                observer.unobserve(e.target);
            }
        }, { threshold: 0.2 });
        cards.forEach(c => {
            c.classList.add("city-card--wait");
            c.querySelectorAll(".city-chart").forEach(ch => ch._parts && ch._parts.clipRect.setAttribute("width", 0));
            observer.observe(c);
        });
    }

    function show() {
        visible = true;
        if (data && Math.floor($("cityDay").clientWidth) !== lastWidth) render();
        load();
        clearInterval(timer);
        timer = setInterval(() => { if (!document.hidden) load(); }, REFRESH_MS);
    }
    function hide() {
        visible = false;
        clearInterval(timer);
        timer = null;
    }

    document.addEventListener("DOMContentLoaded", () => {
        document.querySelectorAll("#cityRankMode button").forEach(btn => btn.addEventListener("click", () => {
            rankMode = btn.dataset.mode;
            document.querySelectorAll("#cityRankMode button").forEach(b => b.setAttribute("aria-pressed", String(b === btn)));
            setNodeText($("cityRankHint"), rankMode === "congested"
                ? "Daytime (06–22) congestion index, one camera per junction" : "Vehicles per hour while watched, one camera per junction");
            if (data) renderRank();
        }));
        if ("ResizeObserver" in window) {
            new ResizeObserver(() => {
                if (visible && data && Math.floor($("cityDay").clientWidth) !== lastWidth) render();
            }).observe($("cityDay"));
        }
    });

    window.CityTab = { show, hide };
})();
