// Crowd viewer — drives deck.gl layers on top of an OSM (Carto) basemap
// from a frames-bucketed JSON produced by preprocess.py. O(1) seeking,
// no video re-rendering required.

const PALETTE = [
  '#E63946', '#1D9BF0', '#2DC653', '#F77F00',
  '#9B5DE5', '#00BBF9', '#FEE440', '#F15BB5',
  '#00F5D4', '#8AC926', '#FF6B6B', '#4361EE'
];

function hexToRGB(h) {
  const n = parseInt(h.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function crowdColor(cid, alpha = 230) {
  if (cid == null) return [136, 136, 136, 160];
  return [...hexToRGB(PALETTE[(cid - 1) % PALETTE.length]), alpha];
}

// light gray → dark navy, clamped to [0, dMax]
function densityColor(d, dMax = 0.5, alpha = 230) {
  if (d == null) return [136, 136, 136, 160];
  const t = Math.max(0, Math.min(1, d / dMax));
  return [
    Math.round(204 * (1 - t) + 10 * t),
    Math.round(204 * (1 - t) + 36 * t),
    Math.round(204 * (1 - t) + 99 * t),
    alpha
  ];
}

// Simple 2-D convex hull (monotone chain) for crowd outlines.
function convexHull(points) {
  if (points.length < 3) return points.slice();
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  return lower.slice(0, -1).concat(upper.slice(0, -1));
}

const STYLE = {
  version: 8,
  sources: {
    carto: {
      type: 'raster',
      tiles: [
        'https://a.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        'https://b.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png',
        'https://c.basemaps.cartocdn.com/light_all/{z}/{x}/{y}.png'
      ],
      tileSize: 256,
      attribution: '© OpenStreetMap contributors © CARTO'
    }
  },
  layers: [{ id: 'carto', type: 'raster', source: 'carto' }]
};

async function loadFrames(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status} ${r.statusText}`);
  return r.json();
}

async function init() {
  const loader = document.getElementById('loader');
  loader.textContent = 'Loading frames.json…';

  const data = await loadFrames('frames.json');
  const { metadata, frames } = data;
  const total = frames.length;
  const fpsSource = metadata.fps || 24;
  const streetIndexMap = metadata.street_index_map || null;  // {"0": "__unassigned__", "1": "Wood Street", ...}
  const idStride = metadata.id_stride || null;

  function streetNameFor(si, cid) {
    if (streetIndexMap && si != null && streetIndexMap[String(si)]) return streetIndexMap[String(si)];
    if (streetIndexMap && idStride && cid != null) {
      const key = String(Math.floor(cid / idStride));
      if (streetIndexMap[key]) return streetIndexMap[key];
    }
    return null;
  }

  function localCrowdIdFor(cid) {
    if (cid == null) return null;
    if (!idStride) return cid;
    return cid % idStride;
  }

  // Determine initial map center: home if present, otherwise first UAV point
  let center = null;
  if (metadata.home && metadata.home.latitude != null) {
    center = [metadata.home.longitude, metadata.home.latitude];
  }
  if (!center) {
    for (const f of frames) {
      if (f.uav) { center = [f.uav.lon, f.uav.lat]; break; }
    }
  }
  if (!center) center = [0, 0];

  // Pre-compute the full UAV trail once (static across frames)
  const uavTrailFull = [];
  for (const f of frames) if (f.uav) uavTrailFull.push([f.uav.lon, f.uav.lat]);

  // Global density cap (P99 across all crowd_density values)
  const densities = [];
  for (const f of frames) for (const p of f.persons) if (p.cd != null) densities.push(p.cd);
  densities.sort((a, b) => a - b);
  const dMax = densities.length
    ? Math.max(0.1, densities[Math.floor(0.99 * (densities.length - 1))])
    : 0.5;

  // Initialise map + deck overlay
  const map = new maplibregl.Map({
    container: 'map',
    style: STYLE,
    center, zoom: 17, pitch: 0, bearing: 0
  });
  map.addControl(new maplibregl.NavigationControl({ showCompass: true }), 'top-left');
  map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-left');

  await new Promise(res => map.once('load', res));

  const deckOverlay = new deck.MapboxOverlay({ interleaved: false, layers: [] });
  map.addControl(deckOverlay);

  // UI
  const seek = document.getElementById('seek');
  seek.max = total - 1;
  const playBtn = document.getElementById('play');
  const fpsSelect = document.getElementById('fps-select');
  const modeDensity = document.getElementById('mode-density');
  const showArrows = document.getElementById('show-arrows');
  const showHulls = document.getElementById('show-hulls');
  const info = document.getElementById('frame-info');
  const legend = document.getElementById('legend');

  let current = 0;
  let playing = false;
  let lastTick = 0;

  // ── ETA overlay state ─────────────────────────────────────────────
  let etaState = null; // {track_id, eta_s, distance_m, path: [[lon,lat], ...],
                       //  start: [lat,lon], station: [lat,lon], colorRgb}

  function setLegend(activeCrowdIds) {
    if (modeDensity.checked) {
      let html = '<b>Crowd density (p/m²)</b>';
      const ticks = [0, dMax * 0.2, dMax * 0.4, dMax * 0.6, dMax * 0.8, dMax];
      for (const v of ticks) {
        const [r, g, b] = densityColor(v, dMax);
        html += `<div><span class="swatch" style="background:rgb(${r},${g},${b})"></span>${v.toFixed(2)}</div>`;
      }
      html += `<div style="margin-top:6px;color:#666">cap = P99 (${dMax.toFixed(2)})</div>`;
      legend.innerHTML = html;
    } else {
      const ids = [...activeCrowdIds].sort((a, b) => a - b).slice(0, 12);
      let html = `<b>Active crowds <span class="pill">${activeCrowdIds.size}</span></b>`;
      for (const cid of ids) {
        const [r, g, b] = crowdColor(cid, 255);
        html += `<div><span class="swatch" style="background:rgb(${r},${g},${b})"></span>Crowd ${cid}</div>`;
      }
      if (activeCrowdIds.size > 12) html += `<div style="color:#666">…and ${activeCrowdIds.size - 12} more</div>`;
      legend.innerHTML = html;
    }
  }

  function computeHulls(persons) {
    const byCid = new Map();
    for (const p of persons) {
      if (p.cid == null) continue;
      if (!byCid.has(p.cid)) byCid.set(p.cid, []);
      byCid.get(p.cid).push([p.lon, p.lat]);
    }
    const hulls = [];
    for (const [cid, pts] of byCid) {
      if (pts.length < 3) continue;
      const h = convexHull(pts);
      if (h.length < 3) continue;
      h.push(h[0]); // close polygon
      hulls.push({ cid, path: h });
    }
    return hulls;
  }

  function render() {
    const frame = frames[current];
    const persons = frame.persons || [];
    const uav = frame.uav || null;

    const activeCrowdIds = new Set();
    for (const p of persons) if (p.cid != null) activeCrowdIds.add(p.cid);

    const { ScatterplotLayer, PathLayer } = deck;

    const uavTrailSoFar = uavTrailFull.slice(0, current + 1);

    const layers = [
      new PathLayer({
        id: 'uav-full',
        data: [{ path: uavTrailFull }],
        getPath: d => d.path,
        getColor: [150, 150, 150, 80],
        getWidth: 1, widthUnits: 'pixels'
      }),
      new PathLayer({
        id: 'uav-so-far',
        data: [{ path: uavTrailSoFar }],
        getPath: d => d.path,
        getColor: [30, 144, 255, 220],
        getWidth: 2, widthUnits: 'pixels'
      })
    ];

    if (showHulls.checked) {
      const hulls = computeHulls(persons);
      layers.push(new PathLayer({
        id: 'hulls',
        data: hulls,
        getPath: d => d.path,
        getColor: d => modeDensity.checked ? [70, 70, 70, 120] : crowdColor(d.cid, 160),
        getWidth: 1.5, widthUnits: 'pixels',
        getDashArray: [3, 2], dashJustified: false,
        extensions: []
      }));
    }

    layers.push(new ScatterplotLayer({
      id: 'persons',
      data: persons,
      getPosition: d => [d.lon, d.lat],
      getRadius: 2.5, radiusUnits: 'meters',
      radiusMinPixels: 3, radiusMaxPixels: 12,
      getFillColor: d => modeDensity.checked ? densityColor(d.cd, dMax) : crowdColor(d.cid),
      getLineColor: [255, 255, 255, 220],
      stroked: true, lineWidthMinPixels: 1,
      pickable: true,
      onHover: info => {
        const el = document.getElementById('tooltip');
        if (!el) return;
        if (!info.object) { el.style.display = 'none'; return; }
        const p = info.object;
        const street = streetNameFor(p.si, p.cid);
        const local = localCrowdIdFor(p.cid);
        const lines = [
          `<b>track ${p.tid}</b>`,
          p.cid != null
            ? (street ? `Crowd ${local} • ${street}` : `Crowd ${p.cid}`)
            : `<span style="color:#888">noise / unclustered</span>`,
          p.cd != null ? `density: ${p.cd.toFixed(3)} p/m²` : null,
          (p.vn || p.ve) ? `v: ${Math.hypot(p.vn||0, p.ve||0).toFixed(2)} m/s` : null,
        ].filter(Boolean).join('<br>');
        el.innerHTML = lines;
        el.style.left = info.x + 'px';
        el.style.top = info.y + 'px';
        el.style.display = 'block';
      }
    }));

    if (showArrows.checked) {
      const arrows = [];
      const latCos = Math.cos((uav ? uav.lat : center[1]) * Math.PI / 180);
      for (const p of persons) {
        const vn = p.vn || 0, ve = p.ve || 0;
        const s = Math.hypot(vn, ve);
        if (s < 0.15) continue;
        const dN = vn * 1.0;  // 1-second lookahead
        const dE = ve * 1.0;
        const lat2 = p.lat + dN / 111111;
        const lon2 = p.lon + dE / (111111 * latCos);
        arrows.push({
          path: [[p.lon, p.lat], [lon2, lat2]],
          color: modeDensity.checked ? densityColor(p.cd, dMax) : crowdColor(p.cid)
        });
      }
      layers.push(new PathLayer({
        id: 'arrows',
        data: arrows,
        getPath: d => d.path,
        getColor: d => d.color,
        getWidth: 1.5, widthUnits: 'pixels'
      }));
    }

    if (uav) {
      // heading wedge as a short triangle
      const hdg = (uav.heading ?? 0) * Math.PI / 180;
      const r = 0.0004; // degrees (~40 m at this lat)
      const wedge = [
        [uav.lon, uav.lat],
        [uav.lon + r * Math.sin(hdg + 0.5), uav.lat + r * Math.cos(hdg + 0.5)],
        [uav.lon + r * Math.sin(hdg - 0.5), uav.lat + r * Math.cos(hdg - 0.5)],
        [uav.lon, uav.lat]
      ];
      layers.push(new PathLayer({
        id: 'uav-heading',
        data: [{ path: wedge }],
        getPath: d => d.path,
        getColor: [220, 60, 60, 180],
        getWidth: 1.5, widthUnits: 'pixels'
      }));
      layers.push(new ScatterplotLayer({
        id: 'uav',
        data: [uav],
        getPosition: d => [d.lon, d.lat],
        getRadius: 4, radiusUnits: 'meters',
        radiusMinPixels: 6,
        getFillColor: [0, 0, 0, 255],
        getLineColor: [255, 255, 255, 255],
        stroked: true, lineWidthMinPixels: 2
      }));
    }

    // ── ETA overlay (path + endpoints) ────────────────────────────
    if (etaState) {
      const { path, start, station, colorRgb } = etaState;
      layers.push(new PathLayer({
        id: 'eta-path',
        data: [{ path }],
        getPath: d => d.path,
        getColor: [...colorRgb, 235],
        getWidth: 4, widthUnits: 'pixels',
        capRounded: true, jointRounded: true,
        zorder: 20
      }));
      layers.push(new ScatterplotLayer({
        id: 'eta-start',
        data: [{ p: [start[1], start[0]] }],
        getPosition: d => d.p,
        getRadius: 6, radiusUnits: 'meters', radiusMinPixels: 7,
        getFillColor: [...colorRgb, 255],
        getLineColor: [255, 255, 255, 255], stroked: true, lineWidthMinPixels: 2
      }));
      layers.push(new ScatterplotLayer({
        id: 'eta-station',
        data: [{ p: [station[1], station[0]] }],
        getPosition: d => d.p,
        getRadius: 8, radiusUnits: 'meters', radiusMinPixels: 9,
        getFillColor: [34, 139, 230, 255],
        getLineColor: [255, 255, 255, 255], stroked: true, lineWidthMinPixels: 2
      }));
    }

    deckOverlay.setProps({ layers });

    const t = current / fpsSource;
    const altTxt = uav ? ` alt=${uav.alt.toFixed(0)}m hdg=${uav.heading.toFixed(0)}°` : '';
    info.textContent =
      `Frame ${current}/${total - 1}  t=${t.toFixed(2)}s  crowds=${activeCrowdIds.size}  pers=${persons.length}${altTxt}`;
    setLegend(activeCrowdIds);
  }

  function tick(ts) {
    if (!playing) return;
    const fps = parseInt(fpsSelect.value, 10);
    if (!lastTick) lastTick = ts;
    const dt = ts - lastTick;
    const step = 1000 / fps;
    if (dt >= step) {
      const advance = Math.max(1, Math.floor(dt / step));
      current = (current + advance) % total;
      seek.value = current;
      render();
      lastTick = ts;
    }
    requestAnimationFrame(tick);
  }

  playBtn.addEventListener('click', () => {
    playing = !playing;
    playBtn.textContent = playing ? '❚❚ Pause' : '▶ Play';
    lastTick = 0;
    if (playing) requestAnimationFrame(tick);
  });
  document.addEventListener('keydown', e => {
    if (e.code === 'Space') { e.preventDefault(); playBtn.click(); }
    else if (e.code === 'ArrowRight') { current = Math.min(total - 1, current + 1); seek.value = current; render(); }
    else if (e.code === 'ArrowLeft') { current = Math.max(0, current - 1); seek.value = current; render(); }
  });
  seek.addEventListener('input', () => { current = parseInt(seek.value, 10); render(); });
  modeDensity.addEventListener('change', render);
  showArrows.addEventListener('change', render);
  showHulls.addEventListener('change', render);

  // ── ETA panel ─────────────────────────────────────────────────────
  const etaTid = document.getElementById('eta-tid');
  const etaGo = document.getElementById('eta-go');
  const etaClear = document.getElementById('eta-clear');
  const etaResult = document.getElementById('eta-result');

  async function computeETA() {
    const tid = parseInt(etaTid.value, 10);
    if (!Number.isFinite(tid)) {
      etaResult.className = 'err';
      etaResult.textContent = 'Enter a track_id.';
      return;
    }
    etaGo.disabled = true;
    etaResult.className = '';
    etaResult.textContent = `Computing ETA for track ${tid} at frame ${current}…`;
    try {
      const r = await fetch('/api/eta', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ frame_id: current, track_id: tid })
      });
      if (!r.ok) {
        const detail = (await r.json().catch(() => ({}))).detail || r.statusText;
        throw new Error(detail);
      }
      const resp = await r.json();
      // Map crowd_id colour from the source track, falling back to station blue
      const frame = frames[current];
      const member = frame?.persons?.find(p => p.tid === tid);
      const baseColor = member && member.cid != null ? crowdColor(member.cid, 255).slice(0, 3) : [220, 80, 40];
      etaState = {
        track_id: tid,
        eta_s: resp.eta_s,
        distance_m: resp.distance_m,
        speed_mps: resp.speed_mps,
        heading_deg: resp.heading_deg,
        path: resp.path_geojson.coordinates,
        start: resp.start_lat_lon,
        station: resp.station_lat_lon,
        stationLabel: resp.station_label,
        colorRgb: baseColor
      };
      etaResult.innerHTML =
        `Track <b>${tid}</b> → ${resp.station_label}<br>` +
        `&nbsp;ETA&nbsp;<b>${resp.eta_s.toFixed(1)} s</b> · dist ${resp.distance_m.toFixed(0)} m · ` +
        `speed ${resp.speed_mps.toFixed(2)} m/s` +
        (resp.heading_deg != null ? ` · hdg ${resp.heading_deg.toFixed(0)}°` : '');
      render();
    } catch (err) {
      etaState = null;
      etaResult.className = 'err';
      etaResult.textContent = `Error: ${err.message}`;
      render();
    } finally {
      etaGo.disabled = false;
    }
  }

  etaGo.addEventListener('click', computeETA);
  etaTid.addEventListener('keydown', e => { if (e.key === 'Enter') computeETA(); });
  etaClear.addEventListener('click', () => {
    etaState = null;
    etaResult.className = '';
    etaResult.textContent = 'Pause the video, type a track_id, hit Compute.';
    render();
  });

  loader.remove();
  render();
}

init().catch(err => {
  const el = document.getElementById('loader') || document.body;
  el.innerHTML = `<pre style="padding:20px;color:#c33;white-space:pre-wrap">Error: ${err.message}

Did you run preprocess.py to create crowd_service/frames.json ?
Then serve the folder with:
  cd crowd_service && python -m http.server 8000
and open http://localhost:8000/
</pre>`;
});
