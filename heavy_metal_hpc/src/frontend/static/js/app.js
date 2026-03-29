const body = document.body;
const searchUrl = body.dataset.searchUrl;
const riskUrl = body.dataset.riskUrl;
const scenarioUrl = body.dataset.scenarioUrl;
const compareUrl = body.dataset.compareUrl;
const mapUrl = body.dataset.mapUrl;
const briefUrl = body.dataset.briefUrl;

const sliderDefaults = {
  industrial_reduction: 20,
  wastewater_improvement: 30,
  plastic_reduction: 15,
  household_filter_adoption: 40,
  microfiber_filter_adoption: 50,
};

const factorMeta = {
  industrial_score: { label: "Industrial", color: "#ef6a5b" },
  wastewater_score: { label: "Wastewater", color: "#2d8cb2" },
  plastic_waste_score: { label: "Plastic Waste", color: "#f3b847" },
  microfiber_score: { label: "Microfiber", color: "#6b7df2" },
};

const state = {
  sliders: { ...sliderDefaults },
  mapPayload: null,
  mapInstance: null,
  mapLayers: {},
  activeView: "map",
  selection: null,
  selectionScenario: null,
  selectionBaseline: null,
  compareEnabled: false,
  compareSelection: null,
  compareData: null,
  pendingComparePick: false,
  view: null,
};

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function activateView(view) {
  state.activeView = view;
  document.querySelectorAll(".view-tab").forEach((button) => {
    const isActive = button.dataset.view === view;
    button.classList.toggle("is-active", isActive);
    button.setAttribute("aria-selected", String(isActive));
  });

  document.querySelectorAll("[data-view-panel]").forEach((panel) => {
    panel.classList.toggle("is-active", panel.dataset.viewPanel === view);
  });

  if (view === "map" && state.mapInstance) {
    window.setTimeout(() => {
      state.mapInstance.invalidateSize();
      drawMap();
    }, 0);
  }
}

function buildScenarioQuery() {
  const params = new URLSearchParams();
  Object.entries(state.sliders).forEach(([key, value]) => params.set(key, String(value)));
  return params.toString();
}

function riskLabel(score) {
  if (score >= 70) return "High";
  if (score >= 40) return "Medium";
  return "Low";
}

function populateSliderLabels() {
  Object.keys(state.sliders).forEach((key) => {
    const label = document.getElementById(`${key}-value`);
    if (label) {
      label.textContent = `${state.sliders[key]}%`;
    }
  });
}

function updateDatalist(options) {
  const datalist = document.getElementById("location-options");
  datalist.innerHTML = options.map((item) => `<option value="${item.name}"></option>`).join("");
}

function updateLocationPicker(options) {
  const picker = document.getElementById("location-picker");
  if (!picker) return;

  const currentValue = picker.value;
  picker.innerHTML = options
    .map(
      (item) =>
        `<option value="${item.id}">${item.name}${item.subtitle ? ` - ${item.subtitle}` : ""}</option>`,
    )
    .join("");

  if (state.selection?.location?.name) {
    const selectedOption = options.find((item) => item.name === state.selection.location.name);
    if (selectedOption) {
      picker.value = selectedOption.id;
      return;
    }
  }

  if (currentValue && options.some((item) => item.id === currentValue)) {
    picker.value = currentValue;
  }
}

function colorForScore(score) {
  if (score >= 70) return [239, 106, 91];
  if (score >= 40) return [243, 184, 71];
  return [75, 163, 187];
}

function popupMarkup(title, subtitle) {
  return `
    <div class="map-popup">
      <div class="map-popup-title">${title}</div>
      <div class="map-popup-copy">${subtitle}</div>
    </div>
  `;
}

function ensureLeafletMap() {
  if (!window.L || state.mapInstance) return;

  state.mapInstance = window.L.map("map", {
    zoomControl: true,
    attributionControl: true,
    scrollWheelZoom: true,
  });

  window.L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
    attribution: "&copy; OpenStreetMap contributors &copy; CARTO",
    subdomains: "abcd",
    maxZoom: 18,
  }).addTo(state.mapInstance);

  state.mapLayers = {
    outlines: window.L.layerGroup().addTo(state.mapInstance),
    markers: window.L.layerGroup().addTo(state.mapInstance),
    selection: window.L.layerGroup().addTo(state.mapInstance),
    compare: window.L.layerGroup().addTo(state.mapInstance),
    heat: null,
  };

  state.mapInstance.on("click", (event) => {
    const comparePick = state.compareEnabled && state.pendingComparePick && state.selection;
    const candidate = {
      name: `Selected Point (${event.latlng.lat.toFixed(3)}, ${event.latlng.lng.toFixed(3)})`,
      latitude: event.latlng.lat,
      longitude: event.latlng.lng,
    };
    selectLocationFromCandidate(candidate, comparePick)
      .then(() => {
        state.pendingComparePick = false;
      })
      .catch((error) => {
        document.getElementById("status-text").textContent = error.message;
      });
  });
}

function getMapGeometry() {
  const canvas = document.getElementById("map");
  const rect = canvas.getBoundingClientRect();
  return { canvas, rect, width: rect.width, height: rect.height };
}

function ensureView() {
  if (!state.mapPayload) return;
  if (state.view) return;
  resetZoom();
}

function resetZoom() {
  if (!state.mapPayload) return;
  if (state.mapInstance) {
    state.mapInstance.fitBounds([
      [state.mapPayload.bounds.min_lat, state.mapPayload.bounds.min_lon],
      [state.mapPayload.bounds.max_lat, state.mapPayload.bounds.max_lon],
    ]);
    return;
  }
  const bounds = state.mapPayload.bounds;
  state.view = {
    minLon: bounds.min_lon,
    maxLon: bounds.max_lon,
    minLat: bounds.min_lat,
    maxLat: bounds.max_lat,
  };
  drawMap();
}

function zoomToPoint(latitude, longitude) {
  if (state.mapInstance) {
    state.mapInstance.setView([latitude, longitude], 9, { animate: false });
    return;
  }
  const latSpan = 8;
  const lonSpan = 14;
  state.view = {
    minLon: clamp(longitude - lonSpan, state.mapPayload.bounds.min_lon, state.mapPayload.bounds.max_lon),
    maxLon: clamp(longitude + lonSpan, state.mapPayload.bounds.min_lon, state.mapPayload.bounds.max_lon),
    minLat: clamp(latitude - latSpan, state.mapPayload.bounds.min_lat, state.mapPayload.bounds.max_lat),
    maxLat: clamp(latitude + latSpan, state.mapPayload.bounds.min_lat, state.mapPayload.bounds.max_lat),
  };
  if (state.view.maxLon - state.view.minLon < 6) {
    state.view.maxLon += 3;
    state.view.minLon -= 3;
  }
  if (state.view.maxLat - state.view.minLat < 4) {
    state.view.maxLat += 2;
    state.view.minLat -= 2;
  }
  drawMap();
}

function worldToScreen(latitude, longitude, width, height) {
  ensureView();
  const x = ((longitude - state.view.minLon) / (state.view.maxLon - state.view.minLon)) * width;
  const y = height - ((latitude - state.view.minLat) / (state.view.maxLat - state.view.minLat)) * height;
  return { x, y };
}

function screenToWorld(x, y, width, height) {
  ensureView();
  const longitude = state.view.minLon + (x / Math.max(width, 1)) * (state.view.maxLon - state.view.minLon);
  const latitude = state.view.minLat + ((height - y) / Math.max(height, 1)) * (state.view.maxLat - state.view.minLat);
  return { latitude, longitude };
}

function visibleMarker(marker) {
  ensureView();
  return (
    marker.longitude >= state.view.minLon &&
    marker.longitude <= state.view.maxLon &&
    marker.latitude >= state.view.minLat &&
    marker.latitude <= state.view.maxLat
  );
}

function drawMarker(ctx, marker, width, height, selected = false) {
  if (!visibleMarker(marker)) return;
  const { x, y } = worldToScreen(marker.latitude, marker.longitude, width, height);
  ctx.save();
  ctx.beginPath();
  ctx.fillStyle = selected ? "#102235" : "#ffffff";
  ctx.strokeStyle = selected ? "#ffffff" : "rgba(16, 34, 53, 0.55)";
  ctx.lineWidth = 3;
  ctx.arc(x, y, selected ? 8 : 5, 0, Math.PI * 2);
  ctx.fill();
  ctx.stroke();
  ctx.restore();
}

function drawMap() {
  if (!state.mapPayload) return;
  if (window.L) {
    renderLeafletMap();
    return;
  }

  const { canvas, width, height } = getMapGeometry();
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const lats = state.mapPayload.latitudes;
  const lons = state.mapPayload.longitudes;
  const grid = state.mapPayload.scenario_grid;
  const base = state.mapPayload.baseline_grid;
  const maxScore = Math.max(...base.flat(), 1);

  ctx.clearRect(0, 0, width, height);
  const water = ctx.createLinearGradient(0, 0, width, height);
  water.addColorStop(0, "#d7eef7");
  water.addColorStop(0.5, "#bddbe8");
  water.addColorStop(1, "#dbeaf1");
  ctx.fillStyle = water;
  ctx.fillRect(0, 0, width, height);

  for (let row = 0; row < lats.length - 1; row += 1) {
    for (let col = 0; col < lons.length - 1; col += 1) {
      const lat0 = lats[row];
      const lat1 = lats[row + 1];
      const lon0 = lons[col];
      const lon1 = lons[col + 1];
      if (
        lon1 < state.view.minLon ||
        lon0 > state.view.maxLon ||
        lat1 < state.view.minLat ||
        lat0 > state.view.maxLat
      ) {
        continue;
      }
      const score = grid[row][col];
      const ratio = clamp(score / maxScore, 0, 1);
      const [r, g, b] = colorForScore(score);
      const topLeft = worldToScreen(lat1, lon0, width, height);
      const bottomRight = worldToScreen(lat0, lon1, width, height);
      ctx.fillStyle = `rgba(${r}, ${g}, ${b}, ${0.18 + ratio * 0.65})`;
      ctx.fillRect(
        Math.min(topLeft.x, bottomRight.x),
        Math.min(topLeft.y, bottomRight.y),
        Math.abs(bottomRight.x - topLeft.x) + 1,
        Math.abs(bottomRight.y - topLeft.y) + 1,
      );
    }
  }

  state.mapPayload.markers.forEach((marker) => drawMarker(ctx, marker, width, height, false));
  drawStateOutlines(ctx, width, height);
  if (state.selection) {
    drawMarker(ctx, state.selection.location, width, height, true);
  }
  if (state.compareSelection) {
    drawMarker(ctx, state.compareSelection.location, width, height, false);
  }

  if (state.selection) {
    document.getElementById("selection-coords").textContent =
      `${state.selection.location.latitude.toFixed(3)}, ${state.selection.location.longitude.toFixed(3)}`;
  }
}

function renderLeafletMap() {
  ensureLeafletMap();
  if (!state.mapInstance) return;

  const map = state.mapInstance;
  const bounds = [
    [state.mapPayload.bounds.min_lat, state.mapPayload.bounds.min_lon],
    [state.mapPayload.bounds.max_lat, state.mapPayload.bounds.max_lon],
  ];

  if (!map.__initialBoundsApplied) {
    map.fitBounds(bounds);
    map.__initialBoundsApplied = true;
  }

  map.invalidateSize();
  document.getElementById("map-tooltip").hidden = true;

  state.mapLayers.outlines.clearLayers();
  state.mapLayers.markers.clearLayers();
  state.mapLayers.selection.clearLayers();
  state.mapLayers.compare.clearLayers();
  if (state.mapLayers.heat) {
    map.removeLayer(state.mapLayers.heat);
  }

  (state.mapPayload.state_outlines || []).forEach((path) => {
    window.L.polyline(
      path.map(([longitude, latitude]) => [latitude, longitude]),
      { color: "rgba(16, 34, 53, 0.28)", weight: 1.1, opacity: 0.65 },
    ).addTo(state.mapLayers.outlines);
  });

  const directHeatPoints = state.mapPayload.heat_points || [];
  let heatPoints = [];
  if (directHeatPoints.length) {
    const maxIntensity = Math.max(...directHeatPoints.map((point) => point.intensity || 0), 1);
    heatPoints = directHeatPoints
      .map((point) => {
        const normalized = clamp((point.intensity || 0) / maxIntensity, 0, 1);
        if (normalized < 0.18) return null;
        return [
          point.latitude,
          point.longitude,
          clamp(normalized ** 1.55, 0.28, 1),
        ];
      })
      .filter(Boolean);

    if (state.selection) {
      heatPoints.push([
        state.selection.location.latitude,
        state.selection.location.longitude,
        1,
      ]);
    }
  } else {
    const lats = state.mapPayload.latitudes || [];
    const lons = state.mapPayload.longitudes || [];
    const grid = state.mapPayload.scenario_grid || [];
    const maxScore = Math.max(...grid.flat(), 1);
    for (let row = 0; row < lats.length - 1; row += 1) {
      for (let col = 0; col < lons.length - 1; col += 1) {
        const score = grid[row]?.[col] ?? 0;
        if (score <= 0) continue;
        heatPoints.push([
          (lats[row] + lats[row + 1]) / 2,
          (lons[col] + lons[col + 1]) / 2,
          clamp(score / maxScore, 0.08, 1),
        ]);
      }
    }
  }

  const zoomLevel = map.getZoom();
  const zoomBoost = directHeatPoints.length ? Math.max(0, zoomLevel - 8) : 0;
  state.mapLayers.heat = window.L.heatLayer(heatPoints, {
    radius: directHeatPoints.length ? 22 + zoomBoost * 3 : 28,
    blur: directHeatPoints.length ? 18 + zoomBoost * 2 : 24,
    maxZoom: 18,
    minOpacity: directHeatPoints.length ? 0.48 : 0.3,
    gradient: {
      0.18: "#2d8cb2",
      0.42: "#76bfd2",
      0.68: "#ffd45a",
      1.0: "#ef6a5b",
    },
  }).addTo(map);

  (state.mapPayload.markers || []).forEach((marker) => {
    const markerColor = marker.type === "wastewater" ? "#2d8cb2" : "#ef6a5b";
    const markerLayer = window.L.circleMarker([marker.latitude, marker.longitude], {
      radius: 7,
      color: "#ffffff",
      weight: 2,
      fillColor: markerColor,
      fillOpacity: 0.9,
    });
    markerLayer.bindPopup(popupMarkup(marker.name, marker.label));
    markerLayer.on("click", (event) => {
      window.L.DomEvent.stopPropagation(event);
      const comparePick = state.compareEnabled && state.pendingComparePick && state.selection;
      selectLocationFromCandidate(
        {
          name: marker.name,
          latitude: marker.latitude,
          longitude: marker.longitude,
        },
        comparePick,
      )
        .then(() => {
          state.pendingComparePick = false;
        })
        .catch((error) => {
          document.getElementById("status-text").textContent = error.message;
        });
    });
    markerLayer.addTo(state.mapLayers.markers);
  });

  if (state.selection) {
    window.L.circleMarker([state.selection.location.latitude, state.selection.location.longitude], {
      radius: 11,
      color: "#102235",
      weight: 4,
      fillColor: "#102235",
      fillOpacity: 0.95,
    }).addTo(state.mapLayers.selection);
    document.getElementById("selection-coords").textContent =
      `${state.selection.location.latitude.toFixed(3)}, ${state.selection.location.longitude.toFixed(3)}`;
  }

  if (state.compareSelection) {
    window.L.circleMarker([state.compareSelection.location.latitude, state.compareSelection.location.longitude], {
      radius: 10,
      color: "#ffffff",
      weight: 3,
      fillColor: "#2d8cb2",
      fillOpacity: 0.95,
    }).addTo(state.mapLayers.compare);
  }
}

function drawStateOutlines(ctx, width, height) {
  const outlines = state.mapPayload?.state_outlines || [];
  ctx.save();
  ctx.strokeStyle = "rgba(16, 34, 53, 0.28)";
  ctx.lineWidth = 1.1;
  outlines.forEach((path) => {
    let started = false;
    ctx.beginPath();
    path.forEach(([longitude, latitude]) => {
      if (
        longitude < state.view.minLon ||
        longitude > state.view.maxLon ||
        latitude < state.view.minLat ||
        latitude > state.view.maxLat
      ) {
        return;
      }
      const point = worldToScreen(latitude, longitude, width, height);
      if (!started) {
        ctx.moveTo(point.x, point.y);
        started = true;
      } else {
        ctx.lineTo(point.x, point.y);
      }
    });
    if (started) {
      ctx.stroke();
    }
  });
  ctx.restore();
}

function renderMapMetrics() {
  if (!state.mapPayload) return;
  document.getElementById("peak-metric").textContent = `${state.mapPayload.stats.baseline_peak}/100`;
  document.getElementById("mean-metric").textContent = `${state.mapPayload.stats.scenario_peak}/100`;
  document.getElementById("snap-metric").textContent = String(state.mapPayload.stats.hotspot_count);
}

function renderBreakdown(baseline, scenario) {
  const container = document.getElementById("breakdown-list");
  if (!container) return;
  container.innerHTML = Object.entries(factorMeta)
    .map(([key, meta]) => {
      const baselineValue = Math.round(baseline[key]);
      const scenarioValue = Math.round(scenario[key]);
      return `
        <div class="breakdown-row">
          <span class="breakdown-name">${meta.label}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:${Math.max(6, scenarioValue)}%; background:linear-gradient(90deg, ${meta.color}, ${meta.color}bb);"></div>
          </div>
          <span class="breakdown-value">${baselineValue} → ${scenarioValue}</span>
        </div>
      `;
    })
    .join("");
}

function renderSummary() {
  if (!state.selection || !state.selectionScenario) return;
  const { location } = state.selection;
  const baseline = state.selectionBaseline;
  const scenario = state.selectionScenario;
  updateLocationPicker(state.mapPayload?.top_places || []);
  document.getElementById("selected-location-name").textContent = location.name;
  document.getElementById("selected-location-subtitle").textContent = state.selection.location.subtitle;
  document.getElementById("risk-score-value").textContent = baseline.total_score.toFixed(1);
  document.getElementById("risk-score-label").textContent = baseline.risk_level;
  document.getElementById("scenario-delta").textContent = `-${scenario.improvement_pct.toFixed(1)}%`;
  document.getElementById("contamination-impact").textContent =
    `${baseline.total_score.toFixed(1)} → ${scenario.total_score.toFixed(1)}`;
  document.getElementById("exposure-impact").textContent =
    `${baseline.total_score.toFixed(1)} → ${scenario.exposure_score.toFixed(1)}`;
  document.getElementById("improvement-value").textContent = `${scenario.improvement_pct.toFixed(1)}% improvement`;
  document.getElementById("best-intervention").textContent = `Best intervention: ${scenario.best_intervention}`;
  renderBreakdown(baseline, scenario);
}

function renderDetails() {
  if (!state.selection) return;
  const explanation = document.getElementById("explanation-list");
  explanation.innerHTML = state.selection.explanation.map((item) => `<li>${item}</li>`).join("");

  const recommendations = document.getElementById("recommendations-list");
  recommendations.innerHTML = state.selection.recommendations.map((item) => `<li>${item}</li>`).join("");

  const sources = document.getElementById("nearby-sources");
  sources.innerHTML = state.selection.nearby_sources
    .map(
      (source, index) => `
        <div class="source-item">
          <strong>${index + 1}. ${source.name} (${source.distance_km} km)</strong>
          <span class="source-meta">${source.summary}</span>
        </div>
      `,
    )
    .join("");
}

function renderComparison() {
  const panel = document.getElementById("comparison-panel");
  if (!state.compareEnabled) {
    panel.innerHTML = "<div class='compare-insight'>Comparison is off. Select <strong>Compare Location</strong> and then search or click a second place.</div>";
    return;
  }
  if (!state.compareSelection || !state.compareData) {
    panel.innerHTML = "<div class='compare-insight'>Compare mode is on. Search for another location or click the map to choose Location B.</div>";
    return;
  }

  const a = state.compareData.location_a;
  const b = state.compareData.location_b;
  panel.innerHTML = `
    <div class="compare-grid">
      <div class="compare-card">
        <strong>${a.location.name}</strong>
        <ul>
          <li>Risk: ${a.scenario.total_score.toFixed(1)}</li>
          <li>Industrial: ${riskLabel(a.baseline.industrial_score)}</li>
          <li>Wastewater: ${riskLabel(a.baseline.wastewater_score)}</li>
          <li>Best intervention: ${a.scenario.best_intervention}</li>
        </ul>
      </div>
      <div class="compare-vs">vs</div>
      <div class="compare-card">
        <strong>${b.location.name}</strong>
        <ul>
          <li>Risk: ${b.scenario.total_score.toFixed(1)}</li>
          <li>Industrial: ${riskLabel(b.baseline.industrial_score)}</li>
          <li>Wastewater: ${riskLabel(b.baseline.wastewater_score)}</li>
          <li>Best intervention: ${b.scenario.best_intervention}</li>
        </ul>
      </div>
    </div>
    <div class="compare-insight">
      <strong>Key Insight:</strong> ${state.compareData.comparison.insight}
    </div>
  `;
}

function renderAll() {
  populateSliderLabels();
  renderSummary();
  renderDetails();
  renderComparison();
  drawMap();
}

async function fetchJSON(url, options = {}) {
  const response = await fetch(url, options);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.error || "Request failed");
  }
  return data;
}

async function searchPlaces(query) {
  const data = await fetchJSON(`${searchUrl}?q=${encodeURIComponent(query)}`);
  return data.results || [];
}

async function selectLocationFromCandidate(candidate, comparePick = false) {
  const riskData = await fetchJSON(
    `${riskUrl}?lat=${encodeURIComponent(candidate.latitude)}&lon=${encodeURIComponent(candidate.longitude)}&name=${encodeURIComponent(candidate.name)}`,
  );

  if (comparePick) {
    state.compareSelection = riskData;
    state.compareData = await fetchJSON(compareUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        locationA: {
          name: state.selection.location.name,
          latitude: state.selection.location.latitude,
          longitude: state.selection.location.longitude,
        },
        locationB: {
          name: riskData.location.name,
          latitude: riskData.location.latitude,
          longitude: riskData.location.longitude,
        },
        scenario: state.sliders,
      }),
    });
  } else {
    state.selection = riskData;
    state.selectionBaseline = riskData.baseline;
    state.selectionScenario = riskData.scenario;
    state.compareSelection = null;
    state.compareData = null;
    document.getElementById("location-search").value = candidate.name;
    zoomToPoint(riskData.location.latitude, riskData.location.longitude);
  }
  renderAll();
}

async function refreshScenario() {
  if (!state.selection) return;
  state.selectionScenario = (await fetchJSON(scenarioUrl, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      latitude: state.selection.location.latitude,
      longitude: state.selection.location.longitude,
      name: state.selection.location.name,
      scenario: state.sliders,
    }),
  })).scenario;

  if (state.compareEnabled && state.compareSelection) {
    state.compareData = await fetchJSON(compareUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        locationA: {
          name: state.selection.location.name,
          latitude: state.selection.location.latitude,
          longitude: state.selection.location.longitude,
        },
        locationB: {
          name: state.compareSelection.location.name,
          latitude: state.compareSelection.location.latitude,
          longitude: state.compareSelection.location.longitude,
        },
        scenario: state.sliders,
      }),
    });
  }
  await loadMap();
  renderAll();
}

async function loadMap() {
  document.getElementById("time-pill").textContent = "Loading map...";
  document.getElementById("stat-pill").textContent = "Building baseline risk surface...";
  state.mapPayload = await fetchJSON(`${mapUrl}?${buildScenarioQuery()}`);
  ensureView();
  renderMapMetrics();
  document.getElementById("time-pill").textContent = "Continental U.S. risk surface";
  document.getElementById("stat-pill").textContent = "Industrial + wastewater + population proxies";
  updateDatalist(state.mapPayload.top_places || []);
  updateLocationPicker(state.mapPayload.top_places || []);
  drawMap();
}

async function handleSearch(event) {
  event.preventDefault();
  const value = document.getElementById("location-search").value.trim();
  if (!value) return;
  const results = await searchPlaces(value);
  if (!results.length) {
    document.getElementById("status-text").textContent = "No match found";
    return;
  }
  const comparePick = state.compareEnabled && state.pendingComparePick && state.selection;
  await selectLocationFromCandidate(results[0], comparePick);
  state.pendingComparePick = false;
  document.getElementById("status-text").textContent = comparePick
    ? `Comparing against ${results[0].name}`
    : `Selected ${results[0].name}`;
}

async function handleMapClick(event) {
  if (state.mapInstance) return;
  if (!state.mapPayload) return;
  const { rect, width, height } = getMapGeometry();
  const x = event.clientX - rect.left;
  const y = event.clientY - rect.top;
  const world = screenToWorld(x, y, width, height);
  const candidate = {
    name: `Selected Point (${world.latitude.toFixed(3)}, ${world.longitude.toFixed(3)})`,
    latitude: world.latitude,
    longitude: world.longitude,
  };
  const comparePick = state.compareEnabled && state.pendingComparePick && state.selection;
  await selectLocationFromCandidate(candidate, comparePick);
  state.pendingComparePick = false;
}

function markerAtPoint(point) {
  const { width, height } = getMapGeometry();
  return (state.mapPayload?.markers || []).find((marker) => {
    if (!visibleMarker(marker)) return false;
    const markerPoint = worldToScreen(marker.latitude, marker.longitude, width, height);
    const dx = markerPoint.x - point.x;
    const dy = markerPoint.y - point.y;
    return Math.sqrt(dx * dx + dy * dy) < 12;
  });
}

function handleMapMove(event) {
  if (state.mapInstance) {
    document.getElementById("map-tooltip").hidden = true;
    return;
  }
  if (!state.mapPayload) return;
  const tooltip = document.getElementById("map-tooltip");
  const { rect } = getMapGeometry();
  const point = {
    x: event.clientX - rect.left,
    y: event.clientY - rect.top,
  };
  const marker = markerAtPoint(point);
  if (!marker) {
    tooltip.hidden = true;
    return;
  }
  tooltip.hidden = false;
  tooltip.textContent = marker.label;
  tooltip.style.left = `${point.x}px`;
  tooltip.style.top = `${point.y}px`;
}

function wireSliders() {
  Object.keys(sliderDefaults).forEach((key) => {
    const input = document.getElementById(`${key}-slider`);
    if (!input) return;
    input.addEventListener("input", async () => {
      state.sliders[key] = Number(input.value);
      populateSliderLabels();
      await refreshScenario();
    });
  });
}

async function resetSliders() {
  state.sliders = { ...sliderDefaults };
  Object.keys(sliderDefaults).forEach((key) => {
    const input = document.getElementById(`${key}-slider`);
    if (input) {
      input.value = sliderDefaults[key];
    }
  });
  await refreshScenario();
}

async function useMyLocation() {
  if (!navigator.geolocation) return;
  navigator.geolocation.getCurrentPosition(async (position) => {
    const candidate = {
      name: `My Location (${position.coords.latitude.toFixed(3)}, ${position.coords.longitude.toFixed(3)})`,
      latitude: position.coords.latitude,
      longitude: position.coords.longitude,
    };
    await selectLocationFromCandidate(candidate, false);
  });
}

async function generateBrief() {
  if (!state.selection) return;
  const button = document.getElementById("generate");
  const briefText = document.getElementById("brief-text");
  button.disabled = true;
  briefText.textContent = "Generating local brief...";
  try {
    const data = await fetchJSON(briefUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        latitude: state.selection.location.latitude,
        longitude: state.selection.location.longitude,
        name: state.selection.location.name,
        scenario: state.sliders,
      }),
    });
    briefText.textContent = data.brief || "No brief returned.";
  } finally {
    button.disabled = false;
  }
}

async function initialize() {
  await loadMap();
  const initial = (state.mapPayload.top_places || [])[0];
  if (initial) {
    await selectLocationFromCandidate(initial, false);
  }
  renderAll();
}

document.getElementById("search-form").addEventListener("submit", (event) => {
  handleSearch(event).catch((error) => {
    document.getElementById("status-text").textContent = error.message;
  });
});

document.getElementById("use-location").addEventListener("click", useMyLocation);

document.getElementById("compare-toggle").addEventListener("click", () => {
  state.compareEnabled = !state.compareEnabled;
  state.pendingComparePick = state.compareEnabled;
  if (!state.compareEnabled) {
    state.compareSelection = null;
    state.compareData = null;
  }
  document.getElementById("compare-toggle").textContent = state.compareEnabled
    ? "Pick Comparison"
    : "Compare Location";
  renderComparison();
  drawMap();
});

document.getElementById("toggle-map").addEventListener("click", resetZoom);

document.getElementById("reload-map").addEventListener("click", () => {
  loadMap().catch((error) => {
    document.getElementById("status-text").textContent = error.message;
  });
});

const resetSlidersButton = document.getElementById("reset-sliders");
if (resetSlidersButton) {
  resetSlidersButton.addEventListener("click", () => {
    resetSliders().catch((error) => {
      document.getElementById("status-text").textContent = error.message;
    });
  });
}

document.getElementById("map").addEventListener("click", (event) => {
  handleMapClick(event).catch((error) => {
    document.getElementById("status-text").textContent = error.message;
  });
});

document.getElementById("map").addEventListener("mousemove", handleMapMove);
document.getElementById("map").addEventListener("mouseleave", () => {
  document.getElementById("map-tooltip").hidden = true;
});

document.getElementById("generate").addEventListener("click", () => {
  generateBrief().catch((error) => {
    document.getElementById("brief-text").textContent = error.message;
  });
});

document.getElementById("location-picker").addEventListener("change", (event) => {
  const nextId = event.target.value;
  const candidate = (state.mapPayload?.top_places || []).find((item) => item.id === nextId);
  if (!candidate) return;

  selectLocationFromCandidate(candidate, false).catch((error) => {
    document.getElementById("status-text").textContent = error.message;
  });
});

document.querySelectorAll(".view-tab").forEach((button) => {
  button.addEventListener("click", () => {
    activateView(button.dataset.view);
  });
});

window.addEventListener("resize", drawMap);
wireSliders();
populateSliderLabels();
activateView("map");
initialize().catch((error) => {
  document.getElementById("status-text").textContent = error.message;
});
