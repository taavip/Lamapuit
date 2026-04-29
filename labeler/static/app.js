// CDW Brush Labeler — queue-first UI, multi-layer canvas, full keyboard.

let CHIP = 256;
const UNKNOWN = 128;
const CWD = 255;
const BG = 0;
const UNDO_MAX = 20;
const AUTOSAVE_MS = 2500;
const CONTEXT_ROW_RADIUS = 1;
const CONTEXT_COL_RADIUS = 1;
const HEATMAP_CYCLE = ["off", "IntGrad", "HiResCAM", "GradCAM+", "RISE"];
const ZOOM_MIN = 1.0;
const ZOOM_MAX = 4.5;

const api = {
  config: () => fetch("/api/config").then((r) => r.json()),
  tiles: (params = {}) =>
    fetch(`/api/tiles?${new URLSearchParams(params)}`).then((r) => r.json()),
  queue: (params = {}) =>
    fetch(`/api/queue?${new URLSearchParams(params)}`).then((r) => r.json()),
  catalog: () => fetch("/api/catalog").then((r) => r.json()),
  rasters: () => fetch("/api/rasters").then((r) => r.json()),
  meta: (id) => fetch(`/api/tile/${id}/meta`).then((r) => r.json()),
  years: (id) => fetch(`/api/tile/${id}/years`).then((r) => r.json()),
  products: (id) => fetch(`/api/tile/${id}/products`).then((r) => r.json()),
  score: (id) => fetch(`/api/tile/${id}/score`).then((r) => r.json()),
  neighbors: (id, rowRadius = 1, colRadius = 1) =>
    fetch(`/api/tile/${id}/neighbors?row_radius=${rowRadius}&col_radius=${colRadius}`).then((r) =>
      r.json()
    ),
  saveMeta: (id, body) =>
    fetch(`/api/tile/${id}/meta`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then((r) => r.json()),
  saveMask: (id, blob) =>
    fetch(`/api/tile/${id}/mask`, {
      method: "POST",
      headers: { "Content-Type": "image/png" },
      body: blob,
    }).then((r) => r.json()),
  chmUrl: (id, smooth) =>
    `/api/tile/${id}/chm${smooth ? "?smooth=1" : ""}`,
  hillshadeUrl: (id, smooth) =>
    `/api/tile/${id}/hillshade${smooth ? "?smooth=1" : ""}`,
  maskUrl: (id) => `/api/tile/${id}/mask`,
  predictUrl: (id, mode = "IntGrad") =>
    `/api/tile/${id}/predict?mode=${encodeURIComponent(mode)}`,
  orthoUrl: (id) => `/api/tile/${id}/ortho`,
  orthoContextUrl: (id, rowRadius = 1, colRadius = 1) =>
    `/api/tile/${id}/ortho_context?row_radius=${rowRadius}&col_radius=${colRadius}`,
};

// ---- Status / notice ------------------------------------------------------

const statusEl = document.getElementById("status-indicator");
const noticeEl = document.getElementById("notice");
let noticeTimer = null;

function setStatus(msg, kind = "") {
  statusEl.textContent = msg;
  statusEl.style.color =
    kind === "ok" ? "var(--ok)" : kind === "err" ? "var(--bad)" : "var(--muted)";
}
function notice(msg, ttl = 1600) {
  noticeEl.textContent = msg;
  noticeEl.classList.add("show");
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => noticeEl.classList.remove("show"), ttl);
}

// ---- State ----------------------------------------------------------------

let queueData = [];
let currentIndex = -1; // position in queueData (when in queue mode)
let currentTileId = null;
let currentYears = [];
let currentProducts = [];
let currentMeta = null;
let currentNeighbors = {};
let catalog = { grids: [], years: [], products: [] };
let currentHeatmapMode = "off";
let orthoContextImage = null;
let orthoContextBlobUrl = null;
let sidebarResizerInitialized = false;

function clearOrthoContextCache() {
  orthoContextImage = null;
  if (orthoContextBlobUrl) {
    URL.revokeObjectURL(orthoContextBlobUrl);
    orthoContextBlobUrl = null;
  }
}

function loadSelectOptions(selectEl, values, allLabel) {
  const prev = selectEl.value;
  selectEl.innerHTML =
    `<option value="">${allLabel}</option>` +
    values.map((v) => `<option value="${v}">${v}</option>`).join("");
  if (prev && values.includes(prev)) {
    selectEl.value = prev;
  }
}

async function loadCatalog() {
  catalog = await api.catalog();
  loadSelectOptions(queueGridSel, catalog.grids || [], "all");
  loadSelectOptions(queueYearSel, catalog.years || [], "all");
  loadSelectOptions(queueProductSel, catalog.products || [], "all");

  loadSelectOptions(gridSelect, catalog.grids || [], "(all)");
  loadSelectOptions(yearSelect, catalog.years || [], "(all)");
  loadSelectOptions(productSelect, catalog.products || [], "(all)");
}

// ---- View switching -------------------------------------------------------

const views = {
  queue: document.getElementById("queue-view"),
  browser: document.getElementById("browser-view"),
  label: document.getElementById("label-view"),
};
const navBtns = {
  queue: document.getElementById("nav-queue"),
  browser: document.getElementById("nav-browser"),
  label: document.getElementById("nav-labeler"),
};
function showView(v) {
  for (const k of Object.keys(views)) {
    views[k].hidden = k !== v;
    navBtns[k].classList.toggle("active", k === v);
  }
  if (v === "label") fitStage();
}
navBtns.queue.onclick = () => showView("queue");
navBtns.browser.onclick = () => showView("browser");
navBtns.label.onclick = () => currentTileId && showView("label");

// ---- Queue view -----------------------------------------------------------

const queueGrid = document.getElementById("queue-grid");
const queueGridSel = document.getElementById("queue-grid-select");
const queueYearSel = document.getElementById("queue-year-select");
const queueProductSel = document.getElementById("queue-product-select");
const queueStatusSel = document.getElementById("queue-status-select");
const queueLimitSel = document.getElementById("queue-limit");
const queueCount = document.getElementById("queue-count");
const queueProgress = document.getElementById("queue-progress");
document.getElementById("queue-refresh").onclick = loadQueue;
queueGridSel.onchange = loadQueue;
queueYearSel.onchange = loadQueue;
queueProductSel.onchange = loadQueue;
queueStatusSel.onchange = loadQueue;
queueLimitSel.onchange = loadQueue;

async function loadQueue() {
  setStatus("loading queue…");
  const params = { limit: queueLimitSel.value };
  if (queueGridSel.value) params.grid = queueGridSel.value;
  if (queueYearSel.value) params.year = queueYearSel.value;
  if (queueProductSel.value) params.product = queueProductSel.value;
  if (queueStatusSel.value) params.status_filter = queueStatusSel.value;
  const data = await api.queue(params);
  queueData = data.items;
  renderQueue();
  queueCount.textContent = `${data.items.length} shown`;
  queueProgress.textContent =
    `queue: ${data.labeled}✓ ${data.needs_review}⚑ / ${data.total}`;
  setStatus("");
}

function renderQueue() {
  queueGrid.innerHTML = "";
  const frag = document.createDocumentFragment();
  queueData.forEach((t, i) => {
    const card = document.createElement("div");
    card.className = `tile-card status-${t.status}`;
    card.innerHTML = `
      <div class="rank">#${t.rank}</div>
      <img loading="lazy" src="${api.chmUrl(t.tile_id)}" alt="" />
      <div class="meta"><span>${t.grid || "?"} ${t.year || ""}</span><span class="badge">${t.status}</span></div>
      <div class="meta"><span>${t.product || "unknown"}</span><span>${t.source || ""}</span></div>
      <div class="meta"><span>score ${t.score.toFixed(3)}</span><span>r${t.row_off}c${t.col_off}</span></div>`;
    card.onclick = () => {
      currentIndex = i;
      openLabelerWithGuard(t.tile_id);
    };
    frag.appendChild(card);
  });
  queueGrid.appendChild(frag);
}

// ---- Raster browser view --------------------------------------------------

const tileGrid = document.getElementById("tile-grid");
const gridSelect = document.getElementById("grid-select");
const yearSelect = document.getElementById("year-select");
const productSelect = document.getElementById("product-select");
const rasterSelect = document.getElementById("raster-select");
const statusSelect = document.getElementById("status-select");
const browserCount = document.getElementById("browser-count");
document.getElementById("refresh-btn").onclick = loadTiles;
gridSelect.onchange = loadTiles;
yearSelect.onchange = loadTiles;
productSelect.onchange = loadTiles;
rasterSelect.onchange = loadTiles;
statusSelect.onchange = loadTiles;

async function loadRasters() {
  const rasters = await api.rasters();
  rasterSelect.innerHTML =
    '<option value="">(all rasters)</option>' +
    rasters
      .map(
        (r) =>
          `<option value="${r.stem}">${r.stem} — ${r.labeled_count}/${r.chip_count}</option>`
      )
      .join("");
}

async function loadTiles() {
  setStatus("loading tiles…");
  const params = { limit: 2000 };
  if (gridSelect.value) params.grid = gridSelect.value;
  if (yearSelect.value) params.year = yearSelect.value;
  if (productSelect.value) params.product = productSelect.value;
  if (rasterSelect.value) params.raster = rasterSelect.value;
  if (statusSelect.value) params.status_filter = statusSelect.value;
  const tiles = await api.tiles(params);
  tileGrid.innerHTML = "";
  const frag = document.createDocumentFragment();
  for (const t of tiles) {
    const card = document.createElement("div");
    card.className = `tile-card status-${t.status}`;
    card.innerHTML = `
      <img loading="lazy" src="${api.chmUrl(t.tile_id)}" alt="" />
      <div class="meta"><span>${t.grid || "?"} ${t.year || ""}</span><span class="badge">${t.status}</span></div>
      <div class="meta"><span>${t.product || "unknown"}</span><span>${t.source || ""}</span></div>`;
    card.onclick = () => {
      currentIndex = -1;
      openLabelerWithGuard(t.tile_id);
    };
    frag.appendChild(card);
  }
  tileGrid.appendChild(frag);
  browserCount.textContent = `${tiles.length} chips`;
  setStatus("");
}

// ---- Konva stage ----------------------------------------------------------

let stage, chmLayer, hillLayer, orthoLayer, heatLayer, maskLayer, brushLayer;
let chmNode, hillNode, orthoNode, heatNode, maskImageNode, brushPreviewNode;
let maskData = null;
let maskCanvas = null;
let undoStack = [];
let dirty = false;
let autoSaveTimer = null;
let painting = false;
let lastXY = null;
let viewZoom = 1.0;
let stageLayout = { side: CHIP, left: 0, top: 0 };

function initStage() {
  if (stage) return;
  stage = new Konva.Stage({ container: "stage", width: CHIP, height: CHIP });
  chmLayer = new Konva.Layer({ listening: false });
  hillLayer = new Konva.Layer({ listening: false, visible: false });
  orthoLayer = new Konva.Layer({ listening: false, visible: false });
  heatLayer = new Konva.Layer({ listening: false, visible: false });
  maskLayer = new Konva.Layer({ listening: false });
  brushLayer = new Konva.Layer({ listening: false });
  stage.add(chmLayer, orthoLayer, hillLayer, heatLayer, maskLayer, brushLayer);

  const c = stage.container();
  c.style.cursor = "none";
  c.addEventListener("pointerdown", onDown);
  c.addEventListener("pointermove", onMove);
  c.addEventListener("pointerup", onUp);
  c.addEventListener("pointerleave", onLeave);
  c.addEventListener("contextmenu", (e) => e.preventDefault());
  c.addEventListener("wheel", onWheelZoom, { passive: false });
  // Global pointerup/cancel — ensures we exit paint mode even if release lands off-canvas.
  window.addEventListener("pointerup", onUp);
  window.addEventListener("pointercancel", () => {
    painting = false;
    lastXY = null;
  });
  // Blur the stage when modifier keys/focus move away so browser doesn't keep capture.
  window.addEventListener("blur", () => {
    painting = false;
    lastXY = null;
    hideBrushPreview();
  });

  maskCanvas = document.createElement("canvas");
  maskCanvas.width = CHIP;
  maskCanvas.height = CHIP;

  window.addEventListener("resize", fitStage);
}

function fitStage() {
  if (!stage) return;
  const wrap = document.getElementById("stage-wrap");
  const compositeW = CHIP * (CONTEXT_COL_RADIUS * 2 + 1);
  const compositeH = CHIP * (CONTEXT_ROW_RADIUS * 2 + 1);
  const baseScale = Math.min(wrap.clientWidth / compositeW, wrap.clientHeight / compositeH);
  if (!isFinite(baseScale) || baseScale <= 0) return;

  const scale = baseScale * viewZoom;
  const side = Math.round(CHIP * scale);
  if (side < 48) return;
  const pixelScale = side / CHIP;

  stage.width(side);
  stage.height(side);
  for (const layer of [chmLayer, hillLayer, orthoLayer, heatLayer, maskLayer, brushLayer]) {
    layer.scale({ x: pixelScale, y: pixelScale });
  }

  const stageLeft = Math.round((wrap.clientWidth - side) / 2);
  const stageTop = Math.round((wrap.clientHeight - side) / 2);
  stageLayout = { side, left: stageLeft, top: stageTop };

  const c = stage.container();
  c.style.position = "absolute";
  c.style.left = `${stageLeft}px`;
  c.style.top = `${stageTop}px`;
  c.style.width = `${side}px`;
  c.style.height = `${side}px`;
  c.style.imageRendering = "pixelated";
  try {
    const content = c.querySelector(".konvajs-content");
    if (content) {
      content.style.position = "absolute";
      content.style.left = "0";
      content.style.top = "0";
      content.style.width = `${side}px`;
      content.style.height = `${side}px`;
      content.style.imageRendering = "pixelated";
    }
    const canvases = Array.from(c.querySelectorAll("canvas"));
    canvases.forEach((cv) => {
      cv.style.position = "absolute";
      cv.style.left = "0";
      cv.style.top = "0";
      cv.style.width = `${side}px`;
      cv.style.height = `${side}px`;
      cv.style.imageRendering = "pixelated";
      const ctx = cv.getContext("2d");
      if (ctx) {
        ctx.imageSmoothingEnabled = false;
      }
    });
  } catch (err) {
    console.warn("fitStage: failed to adjust Konva canvases", err);
  }

  const corners = document.getElementById("tile-corners");
  if (corners) {
    const cornerLen = Math.max(12, Math.round(side * 0.14));
    corners.style.display = "block";
    corners.style.left = `${stageLeft}px`;
    corners.style.top = `${stageTop}px`;
    corners.style.width = `${side}px`;
    corners.style.height = `${side}px`;
    corners.style.setProperty("--corner-len", `${cornerLen}px`);
  }

  renderContextMosaic();
  stage.draw();
}

function contextBaseUrl(tileId, smooth) {
  if (document.getElementById("toggle-hillshade").checked) {
    return api.hillshadeUrl(tileId, smooth);
  }
  return api.chmUrl(tileId, smooth);
}

function renderContextMosaic() {
  const grid = document.getElementById("context-grid");
  if (!grid || !currentTileId || !stage) return;

  const side = Math.round(stageLayout.side || stage.width());
  if (!side || side <= 0) return;

  const smooth = document.getElementById("toggle-smooth").checked;
  const cols = [];
  const rows = [];
  for (let dc = -CONTEXT_COL_RADIUS; dc <= CONTEXT_COL_RADIUS; dc++) cols.push(dc);
  for (let dr = -CONTEXT_ROW_RADIUS; dr <= CONTEXT_ROW_RADIUS; dr++) rows.push(dr);

  grid.style.width = `${side * cols.length}px`;
  grid.style.height = `${side * rows.length}px`;
  grid.style.left = `${Math.round(stageLayout.left - CONTEXT_COL_RADIUS * side)}px`;
  grid.style.top = `${Math.round(stageLayout.top - CONTEXT_ROW_RADIUS * side)}px`;
  grid.innerHTML = "";

  const orthoOn = document.getElementById("toggle-ortho").checked;
  if (orthoOn && orthoContextImage) {
    const cell = document.createElement("div");
    cell.className = "context-cell center";
    cell.style.left = "0px";
    cell.style.top = "0px";
    cell.style.width = `${side * cols.length}px`;
    cell.style.height = `${side * rows.length}px`;
    const img = new Image();
    img.loading = "lazy";
    img.alt = "";
    img.src = orthoContextBlobUrl || orthoContextImage.src;
    cell.appendChild(img);
    grid.appendChild(cell);
    return;
  }

  const frag = document.createDocumentFragment();
  for (const dr of rows) {
    for (const dc of cols) {
      const key = `${dr}_${dc}`;
      const tileId = dr === 0 && dc === 0 ? currentTileId : currentNeighbors[key] || null;
      const cell = document.createElement("div");
      cell.className = `context-cell${tileId ? "" : " missing"}${dr === 0 && dc === 0 ? " center" : ""}`;
      cell.style.left = `${(dc + CONTEXT_COL_RADIUS) * side}px`;
      cell.style.top = `${(dr + CONTEXT_ROW_RADIUS) * side}px`;
      cell.style.width = `${side}px`;
      cell.style.height = `${side}px`;
      if (tileId) {
        const img = new Image();
        img.loading = "lazy";
        img.alt = "";
        img.src = contextBaseUrl(tileId, smooth);
        cell.appendChild(img);
      }
      frag.appendChild(cell);
    }
  }
  grid.appendChild(frag);
}

function onWheelZoom(e) {
  if (!stage || views.label.hidden) return;
  e.preventDefault();

  const direction = e.deltaY < 0 ? 1 : -1;
  if (e.ctrlKey || e.metaKey) {
    const b = document.getElementById("brush-size");
    const step = e.shiftKey ? 3 : 1;
    b.value = Math.max(+b.min, Math.min(+b.max, +b.value + direction * step));
    b.dispatchEvent(new Event("input"));
    return;
  }

  const factor = e.deltaY < 0 ? 1.1 : 1.0 / 1.1;
  viewZoom = Math.max(ZOOM_MIN, Math.min(ZOOM_MAX, viewZoom * factor));
  fitStage();
}

function loadImage(node, url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => {
      try {
        const layer = node && typeof node.getLayer === "function" ? node.getLayer() : null;
        if (!node || !layer) {
          resolve(false);
          return;
        }
        node.image(img);
        node.width(CHIP);
        node.height(CHIP);
        if (typeof node.imageSmoothingEnabled === "function") {
          node.imageSmoothingEnabled(false);
        }
        layer.batchDraw();
        resolve(true);
      } catch (err) {
        resolve(false);
      }
    };
    img.onerror = () => resolve(false);
    img.src = url;
  });
}

function loadImageObject(url) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = url;
  });
}

// ---- Labeler open ---------------------------------------------------------

async function openLabelerWithGuard(tileId) {
  if (!tileId) return;
  if (dirty) await saveMask();
  await openLabeler(tileId);
}

async function openLabeler(tileId) {
  currentTileId = tileId;
  navBtns.label.disabled = false;
  showView("label");
  setStatus("loading…");
  initStage();

  // Reset stage images
  chmLayer.destroyChildren();
  hillLayer.destroyChildren();
  orthoLayer.destroyChildren();
  heatLayer.destroyChildren();
  maskLayer.destroyChildren();
  brushLayer.destroyChildren();
  chmNode = new Konva.Image({ width: CHIP, height: CHIP });
  hillNode = new Konva.Image({ width: CHIP, height: CHIP });
  orthoNode = new Konva.Image({ width: CHIP, height: CHIP, opacity: 0.85 });
  heatNode = new Konva.Image({ width: CHIP, height: CHIP });
  maskImageNode = new Konva.Image({ width: CHIP, height: CHIP });
  [chmNode, hillNode, orthoNode, heatNode, maskImageNode].forEach((node) => {
    if (typeof node.imageSmoothingEnabled === "function") {
      node.imageSmoothingEnabled(false);
    }
  });
  brushPreviewNode = new Konva.Circle({
    x: -100,
    y: -100,
    radius: brushRadius(),
    stroke: "rgba(255,255,255,0.95)",
    strokeWidth: 1,
    dash: [4, 3],
    shadowColor: "black",
    shadowBlur: 0,
    shadowOffsetX: 1,
    shadowOffsetY: 1,
    shadowOpacity: 0.7,
    listening: false,
    visible: false,
  });
  chmLayer.add(chmNode);
  hillLayer.add(hillNode);
  orthoLayer.add(orthoNode);
  heatLayer.add(heatNode);
  maskLayer.add(maskImageNode);
  brushLayer.add(brushPreviewNode);

  undoStack = [];
  dirty = false;
  maskData = new Uint8ClampedArray(CHIP * CHIP).fill(UNKNOWN);

  // Reset toggles that should default per tile
  currentHeatmapMode = "off";
  clearOrthoContextCache();
  document.getElementById("toggle-heatmap").checked = false;
  heatLayer.visible(false);
  document.getElementById("toggle-ortho").checked = false;
  orthoLayer.visible(false);

  const smooth = document.getElementById("toggle-smooth").checked;
  await Promise.all([
    loadImage(chmNode, api.chmUrl(tileId, smooth)),
    loadImage(hillNode, api.hillshadeUrl(tileId, smooth)),
    loadExistingMask(tileId),
  ]);

  currentMeta = await api.meta(tileId);
  document.getElementById("review-flag").checked = !!currentMeta.meta?.review_flag;
  updateInfoPanel(currentMeta);

  currentYears = await api.years(tileId);
  currentProducts = await api.products(tileId);
  loadClassifierScore(tileId);
  renderYearStrip();
  renderProductStrip();
  syncLayerControlVisibility();
  await loadNeighbors(tileId);

  renderMask();
  fitStage();
  setStatus("ready", "ok");
}

function updateInfoPanel(m) {
  const el = (id) => document.getElementById(id);
  el("info-tile").innerHTML = `<span class="k">tile</span>${m.tile_id}`;
  el("info-raster").innerHTML = `<span class="k">raster</span>${m.raster_stem}`;
  el("info-year").innerHTML = `<span class="k">year</span>${m.year || "?"} · ${m.source || ""}`;
  el("info-status").innerHTML = `<span class="k">status</span>${m.status} · ${m.product || "unknown"}`;
  el("info-rowcol").innerHTML = `<span class="k">row/col</span>${m.row_off} / ${m.col_off} · key ${m.label_key || m.tile_id}`;
  if (m.queue) {
    const hasModel = Number.isFinite(m.queue.model_score);
    const modelTxt = hasModel ? ` · model ${(m.queue.model_score * 100).toFixed(1)}%` : "";
    el("info-score").innerHTML =
      `<span class="k">score</span>${m.queue.score.toFixed(3)} ` +
      `(cwd ${(m.queue.cwd_frac * 100).toFixed(1)}% · edge ${(m.queue.edge_frac * 100).toFixed(1)}%${modelTxt})`;
  } else {
    el("info-score").innerHTML = `<span class="k">score</span>—`;
  }
  // Progress indicator
  if (currentIndex >= 0) {
    queueProgress.textContent = `tile ${currentIndex + 1} / ${queueData.length}`;
  }
}

async function loadClassifierScore(tileId) {
  const el = document.getElementById("info-cnn");
  if (!el) return;
  el.innerHTML = `<span class="k">cnn</span>…`;
  try {
    const data = await api.score(tileId);
    if (tileId !== currentTileId) return; // user navigated away
    if (!data.available) {
      el.innerHTML = `<span class="k">cnn</span><span style="color:var(--warn)">model missing</span>`;
      return;
    }
    const pct = (data.score * 100).toFixed(1);
    const color =
      data.score >= 0.5 ? "var(--bad)" : data.score >= 0.25 ? "var(--warn)" : "var(--ok)";
    el.innerHTML = `<span class="k">cnn</span><span style="color:${color}">${pct}% P(CWD)</span>`;
  } catch (err) {
    el.innerHTML = `<span class="k">cnn</span><span style="color:var(--muted)">error</span>`;
  }
}

function renderYearStrip() {
  const strip = document.getElementById("year-strip");
  strip.innerHTML = "";
  for (const v of currentYears) {
    const chip = document.createElement("span");
    chip.className = "year-chip" + (v.is_current ? " current" : "");
    chip.innerHTML = `${v.year}<span class="token">${v.token}</span>`;
    chip.onclick = async () => {
      if (!v.is_current) await openLabelerWithGuard(v.tile_id);
    };
    strip.appendChild(chip);
  }
}

function renderProductStrip() {
  const strip = document.getElementById("product-strip");
  strip.innerHTML = "";
  for (const v of currentProducts) {
    const chip = document.createElement("span");
    chip.className = "product-chip" + (v.is_current ? " current" : "");
    chip.textContent = `${v.product}`;
    chip.onclick = async () => {
      if (!v.is_current) await openLabelerWithGuard(v.tile_id);
    };
    strip.appendChild(chip);
  }
}

function syncLayerControlVisibility() {
  const hasProductSwitch = (currentProducts || []).length > 1;
  const chmRow = document.getElementById("layer-chm-row");
  const smoothRow = document.getElementById("layer-smooth-row");
  if (chmRow) chmRow.classList.toggle("is-hidden", hasProductSwitch);
  if (smoothRow) smoothRow.classList.toggle("is-hidden", hasProductSwitch);

  if (hasProductSwitch) {
    const smoothToggle = document.getElementById("toggle-smooth");
    if (smoothToggle.checked) {
      smoothToggle.checked = false;
      refreshChmLayers();
    }
  }
}

async function loadExistingMask(tileId) {
  const resp = await fetch(api.maskUrl(tileId));
  if (!resp.ok) return;
  const blob = await resp.blob();
  const img = await new Promise((res, rej) => {
    const i = new Image();
    i.onload = () => res(i);
    i.onerror = rej;
    i.src = URL.createObjectURL(blob);
  });
  const tmp = document.createElement("canvas");
  tmp.width = CHIP;
  tmp.height = CHIP;
  const ctx = tmp.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, CHIP, CHIP);
  const { data } = ctx.getImageData(0, 0, CHIP, CHIP);
  for (let i = 0, j = 0; i < data.length; i += 4, j++) {
    const a = data[i + 3];
    const r = data[i];
    if (a === 0) maskData[j] = UNKNOWN;
    else if (r >= 192) maskData[j] = CWD;
    else if (r <= 64) maskData[j] = BG;
    else maskData[j] = UNKNOWN;
  }
}

// ---- Mask render + brush --------------------------------------------------

function renderMask() {
  const ctx = maskCanvas.getContext("2d");
  const img = ctx.createImageData(CHIP, CHIP);
  const data = img.data;
  const alpha = +document.getElementById("mask-alpha").value * 2.55;
  for (let j = 0, i = 0; j < maskData.length; j++, i += 4) {
    const v = maskData[j];
    if (v === CWD) {
      data[i] = 255; data[i + 1] = 77; data[i + 2] = 94; data[i + 3] = alpha;
    } else if (v === BG) {
      data[i] = 46; data[i + 1] = 168; data[i + 2] = 255; data[i + 3] = alpha * 0.7;
    } else {
      data[i + 3] = 0;
    }
  }
  ctx.putImageData(img, 0, 0);
  maskImageNode.image(maskCanvas);
  maskLayer.batchDraw();
}

function brushValue() {
  const k = document.getElementById("brush-class").value;
  return k === "cwd" ? CWD : k === "bg" ? BG : UNKNOWN;
}
function brushRadius() { return +document.getElementById("brush-size").value; }

function setBrushPreview(x, y, visible = true) {
  if (!brushPreviewNode) return;
  brushPreviewNode.x(x);
  brushPreviewNode.y(y);
  brushPreviewNode.radius(brushRadius());
  brushPreviewNode.visible(visible);
  // Colour the preview by active class so the user knows what they'll paint.
  const kind = document.getElementById("brush-class").value;
  const stroke =
    kind === "cwd" ? "rgba(255,80,100,0.95)" :
    kind === "bg"  ? "rgba(70,180,255,0.95)" :
                     "rgba(255,255,255,0.95)";
  brushPreviewNode.stroke(stroke);
  if (brushLayer) brushLayer.batchDraw();
}

function hideBrushPreview() {
  if (!brushPreviewNode) return;
  brushPreviewNode.visible(false);
  if (brushLayer) brushLayer.batchDraw();
}

function paintCircle(cx, cy, r, v) {
  const r2 = r * r;
  const x0 = Math.max(0, Math.floor(cx - r));
  const x1 = Math.min(CHIP - 1, Math.ceil(cx + r));
  const y0 = Math.max(0, Math.floor(cy - r));
  const y1 = Math.min(CHIP - 1, Math.ceil(cy + r));
  for (let y = y0; y <= y1; y++) {
    const dy = y - cy;
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      if (dx * dx + dy * dy <= r2) maskData[y * CHIP + x] = v;
    }
  }
}
function paintLine(x0, y0, x1, y1, r, v) {
  const steps = Math.max(1, Math.ceil(Math.hypot(x1 - x0, y1 - y0)));
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    paintCircle(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, r, v);
  }
}
function pushUndo() {
  undoStack.push(new Uint8ClampedArray(maskData));
  if (undoStack.length > UNDO_MAX) undoStack.shift();
}
function pointerXY(e) {
  const rect = stage.container().getBoundingClientRect();
  return {
    x: ((e.clientX - rect.left) / rect.width) * CHIP,
    y: ((e.clientY - rect.top) / rect.height) * CHIP,
    button: e.button,
  };
}
function onDown(e) {
  if (e.button !== 0 && e.button !== 2) return;
  e.preventDefault();
  painting = true;
  pushUndo();
  const pt = pointerXY(e);
  setBrushPreview(pt.x, pt.y, true);
  const v = e.button === 2 ? BG : brushValue();
  paintCircle(pt.x, pt.y, brushRadius(), v);
  lastXY = { ...pt, v };
  renderMask();
}
function onMove(e) {
  const pt = pointerXY(e);
  setBrushPreview(pt.x, pt.y, true);
  if (!painting || !lastXY) return;
  paintLine(lastXY.x, lastXY.y, pt.x, pt.y, brushRadius(), lastXY.v);
  lastXY = { ...pt, v: lastXY.v };
  renderMask();
}
function onUp() {
  if (!painting) return;
  painting = false;
  lastXY = null;
  dirty = true;
  scheduleAutoSave();
}

function onLeave() {
  onUp();
  hideBrushPreview();
}

// ---- Save -----------------------------------------------------------------

function scheduleAutoSave() {
  if (!document.getElementById("auto-save").checked) return;
  clearTimeout(autoSaveTimer);
  autoSaveTimer = setTimeout(saveMask, AUTOSAVE_MS);
}
async function saveMask() {
  if (!currentTileId || !dirty) return;
  setStatus("saving…");
  const blob = await maskToPng();
  try {
    await api.saveMask(currentTileId, blob);
    await api.saveMeta(currentTileId, {
      review_flag: document.getElementById("review-flag").checked,
    });
    dirty = false;
    setStatus("saved " + new Date().toLocaleTimeString(), "ok");
  } catch (err) {
    setStatus("save failed: " + err, "err");
  }
}
function maskToPng() {
  const c = document.createElement("canvas");
  c.width = CHIP; c.height = CHIP;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(CHIP, CHIP);
  for (let j = 0, i = 0; j < maskData.length; j++, i += 4) {
    const v = maskData[j];
    img.data[i] = v; img.data[i + 1] = v; img.data[i + 2] = v;
    img.data[i + 3] = v === UNKNOWN ? 0 : 255;
  }
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => c.toBlob(res, "image/png"));
}

// ---- Layer toggles / actions ----------------------------------------------

async function refreshChmLayers() {
  if (!currentTileId) return;
  const smooth = document.getElementById("toggle-smooth").checked;
  await Promise.all([
    loadImage(chmNode, api.chmUrl(currentTileId, smooth)),
    loadImage(hillNode, api.hillshadeUrl(currentTileId, smooth)),
  ]);
  renderContextMosaic();
}

async function togglePrediction(force, mode = null) {
  if (!currentTileId) return;
  const cb = document.getElementById("toggle-heatmap");
  if (force !== undefined) cb.checked = force;

  if (mode) currentHeatmapMode = mode;
  if (!cb.checked) currentHeatmapMode = "off";
  if (cb.checked && currentHeatmapMode === "off") currentHeatmapMode = "IntGrad";

  if (cb.checked) {
    setStatus(`heatmap (${currentHeatmapMode})…`);
    const ok = await loadImage(heatNode, api.predictUrl(currentTileId, currentHeatmapMode));
    heatLayer.visible(ok);
    heatLayer.batchDraw();
    const fallback = !serverConfig.classifier_model_available;
    if (ok) {
      const suffix = fallback ? " · fallback (no CNN)" : "";
      setStatus(`heatmap ready (${currentHeatmapMode})${suffix}`, fallback ? "" : "ok");
      if (fallback) notice("classifier model missing — showing CHM-edge fallback");
    } else {
      setStatus("heatmap failed", "err");
      notice("heatmap request failed");
      cb.checked = false;
      currentHeatmapMode = "off";
    }
  } else {
    heatLayer.visible(false);
    heatLayer.batchDraw();
  }
}

async function toggleOrtho(force) {
  if (!currentTileId) return;
  const cb = document.getElementById("toggle-ortho");
  if (force !== undefined) cb.checked = force;
  if (cb.checked) {
    setStatus("fetching orthophoto context…");
    let fetchedBlobUrl = null;
    try {
      const resp = await fetch(
        api.orthoContextUrl(currentTileId, CONTEXT_ROW_RADIUS, CONTEXT_COL_RADIUS)
      );
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const blob = await resp.blob();
      fetchedBlobUrl = URL.createObjectURL(blob);
    } catch (_err) {
      fetchedBlobUrl = null;
    }

    const img = fetchedBlobUrl ? await loadImageObject(fetchedBlobUrl) : null;
    if (!img) {
      if (fetchedBlobUrl) URL.revokeObjectURL(fetchedBlobUrl);
      clearOrthoContextCache();
      orthoLayer.visible(false);
      orthoLayer.batchDraw();
      setStatus("ortho unavailable", "err");
      notice("orthophoto unavailable for this raster/year");
      cb.checked = false;
      renderContextMosaic();
      return;
    }

    clearOrthoContextCache();
    orthoContextBlobUrl = fetchedBlobUrl;
    orthoContextImage = img;
    orthoNode.image(img);
    orthoNode.crop({
      x: CONTEXT_COL_RADIUS * CHIP,
      y: CONTEXT_ROW_RADIUS * CHIP,
      width: CHIP,
      height: CHIP,
    });
    orthoNode.width(CHIP);
    orthoNode.height(CHIP);
    if (typeof orthoNode.imageSmoothingEnabled === "function") {
      orthoNode.imageSmoothingEnabled(false);
    }
    orthoLayer.visible(true);
    orthoLayer.batchDraw();
    setStatus("ortho loaded", "ok");
    renderContextMosaic();
  } else {
    clearOrthoContextCache();
    orthoNode.crop({ x: 0, y: 0, width: CHIP, height: CHIP });
    orthoLayer.visible(false);
    orthoLayer.batchDraw();
    renderContextMosaic();
  }
}

async function cycleHeatmapMode() {
  const cb = document.getElementById("toggle-heatmap");
  const cur = HEATMAP_CYCLE.indexOf(currentHeatmapMode);
  const nextMode = HEATMAP_CYCLE[(cur + 1 + HEATMAP_CYCLE.length) % HEATMAP_CYCLE.length];
  currentHeatmapMode = nextMode;
  if (nextMode === "off") {
    cb.checked = false;
    await togglePrediction(false);
    notice("heatmap: off");
    return;
  }
  cb.checked = true;
  await togglePrediction(true, nextMode);
  notice(`heatmap: ${nextMode}`);
}

async function loadNeighbors(tileId) {
  const { neighbors } = await api.neighbors(tileId, CONTEXT_ROW_RADIUS, CONTEXT_COL_RADIUS);
  currentNeighbors = neighbors || {};
  const smooth = document.getElementById("toggle-smooth").checked;
  const grid = document.getElementById("neighbor-grid");
  grid.innerHTML = "";
  const order = ["-1_-1", "-1_0", "-1_1", "0_-1", "0_0", "0_1", "1_-1", "1_0", "1_1"];
  for (const key of order) {
    const cell = document.createElement("div");
    if (key === "0_0") {
      cell.className = "center";
      cell.innerHTML = `<img src="${api.chmUrl(tileId, smooth)}" />`;
    } else if (currentNeighbors[key]) {
      cell.innerHTML = `<img src="${api.chmUrl(currentNeighbors[key], smooth)}" />`;
      cell.onclick = () => openLabelerWithGuard(currentNeighbors[key]);
    } else {
      cell.className = "empty";
    }
    grid.appendChild(cell);
  }
  renderContextMosaic();
}

// ---- Navigation -----------------------------------------------------------

async function goToQueueOffset(delta) {
  if (currentIndex < 0) {
    notice("Not in queue mode");
    return;
  }
  const next = currentIndex + delta;
  if (next < 0 || next >= queueData.length) {
    notice(next < 0 ? "Start of queue" : "End of queue");
    return;
  }
  currentIndex = next;
  await openLabelerWithGuard(queueData[next].tile_id);
}

async function cycleYear(step) {
  if (!currentYears.length) return;
  const ci = currentYears.findIndex((v) => v.is_current);
  const ni = (ci + step + currentYears.length) % currentYears.length;
  if (ni === ci) return;
  await openLabelerWithGuard(currentYears[ni].tile_id);
  notice(`year ${currentYears[ni].year} ${currentYears[ni].token}`);
}

async function cycleProduct(step) {
  if (!currentProducts.length) return;
  const ci = currentProducts.findIndex((v) => v.is_current);
  const ni = (ci + step + currentProducts.length) % currentProducts.length;
  if (ni === ci) return;
  await openLabelerWithGuard(currentProducts[ni].tile_id);
  notice(`product ${currentProducts[ni].product}`);
}

function initSidebarResizer() {
  if (sidebarResizerInitialized) return;
  const resizer = document.getElementById("sidebar-resizer");
  const labelView = document.getElementById("label-view");
  if (!resizer || !labelView) return;

  const MIN_WIDTH = 300;
  const MAX_WIDTH = 500;
  let dragging = false;

  const applyWidth = (clientX) => {
    const rect = labelView.getBoundingClientRect();
    const raw = Math.round(clientX - rect.left);
    const clamped = Math.max(MIN_WIDTH, Math.min(MAX_WIDTH, raw));
    document.documentElement.style.setProperty("--sidebar-width", `${clamped}px`);
    fitStage();
  };

  const stopDrag = (pointerId = null) => {
    if (!dragging) return;
    dragging = false;
    resizer.classList.remove("dragging");
    if (pointerId !== null) {
      try {
        resizer.releasePointerCapture(pointerId);
      } catch (_err) {
        // Ignore release errors for lost captures.
      }
    }
  };

  resizer.addEventListener("pointerdown", (e) => {
    dragging = true;
    resizer.classList.add("dragging");
    resizer.setPointerCapture(e.pointerId);
    applyWidth(e.clientX);
    e.preventDefault();
  });

  resizer.addEventListener("pointermove", (e) => {
    if (!dragging) return;
    applyWidth(e.clientX);
  });

  resizer.addEventListener("pointerup", (e) => stopDrag(e.pointerId));
  resizer.addEventListener("pointercancel", (e) => stopDrag(e.pointerId));

  sidebarResizerInitialized = true;
}

// ---- Toolbar wiring -------------------------------------------------------

document.getElementById("back-btn").onclick = async () => {
  if (dirty) await saveMask();
  showView(currentIndex >= 0 ? "queue" : "browser");
  if (currentIndex >= 0) loadQueue();
  else loadTiles();
};
document.getElementById("undo-btn").onclick = () => {
  const prev = undoStack.pop();
  if (!prev) return;
  maskData = prev;
  dirty = true;
  renderMask();
  scheduleAutoSave();
};
document.getElementById("clear-btn").onclick = () => {
  pushUndo();
  maskData.fill(UNKNOWN);
  dirty = true;
  renderMask();
  scheduleAutoSave();
};
document.getElementById("save-btn").onclick = saveMask;
document.getElementById("predict-btn").onclick = () => togglePrediction(true);
document.getElementById("prev-btn").onclick = () => goToQueueOffset(-1);
document.getElementById("next-btn").onclick = () => goToQueueOffset(1);
document.getElementById("skip-btn").onclick = () => goToQueueOffset(5);

// Bulk label: mark a set of tiles as background (no CWD)
function createBgMaskBlob() {
  const c = document.createElement("canvas");
  c.width = CHIP;
  c.height = CHIP;
  const ctx = c.getContext("2d");
  const img = ctx.createImageData(CHIP, CHIP);
  for (let j = 0, i = 0; j < CHIP * CHIP; j++, i += 4) {
    img.data[i] = BG;
    img.data[i + 1] = BG;
    img.data[i + 2] = BG;
    img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return new Promise((res) => c.toBlob(res, "image/png"));
}

async function labelAllAsBackground() {
  // Determine tile set: queue view uses `queueData`, browser uses filtered `api.tiles`
  let tiles = [];
  if (!views.queue.hidden) {
    tiles = queueData;
  } else if (!views.browser.hidden) {
    setStatus("fetching tiles…");
    const params = { limit: 5000 };
    if (gridSelect.value) params.grid = gridSelect.value;
    if (yearSelect.value) params.year = yearSelect.value;
    if (productSelect.value) params.product = productSelect.value;
    if (statusSelect.value) params.status_filter = statusSelect.value;
    try {
      tiles = await api.tiles(params);
    } catch (err) {
      setStatus("failed to fetch tiles", "err");
      return;
    }
  } else {
    notice("Open Queue or Browser view to bulk-label");
    return;
  }

  const ids = tiles.map((t) => t.tile_id).filter(Boolean);
  if (!ids.length) {
    notice("No tiles to label");
    return;
  }

  if (!confirm(`Label ${ids.length} tiles as background (no CWD)? This will overwrite existing masks.`)) return;

  setStatus(`creating BG mask…`);
  const bgBlob = await createBgMaskBlob();

  const concurrency = 8;
  let done = 0;
  for (let i = 0; i < ids.length; i += concurrency) {
    const batch = ids.slice(i, i + concurrency);
    await Promise.all(
      batch.map(async (id) => {
        try {
          await api.saveMask(id, bgBlob);
        } catch (err) {
          console.error("label-all-bg failed", id, err);
        }
        done += 1;
        setStatus(`labeled ${done}/${ids.length}`);
      })
    );
  }

  setStatus("bulk label done", "ok");
  notice(`Labeled ${done} tiles as background`);
  if (!views.queue.hidden) await loadQueue();
  else await loadTiles();
}

document.getElementById("label-all-bg-btn").onclick = labelAllAsBackground;

document.getElementById("brush-size").oninput = (e) => {
  document.getElementById("brush-size-val").textContent = e.target.value;
  if (brushPreviewNode && brushPreviewNode.visible()) {
    brushPreviewNode.radius(brushRadius());
    if (brushLayer) brushLayer.batchDraw();
  }
};
document.getElementById("brush-class").onchange = () => {
  if (brushPreviewNode && brushPreviewNode.visible()) {
    setBrushPreview(brushPreviewNode.x(), brushPreviewNode.y(), true);
  }
};
document.getElementById("mask-alpha").oninput = renderMask;

document.getElementById("toggle-chm").onchange = (e) => {
  chmLayer.visible(e.target.checked);
  chmLayer.batchDraw();
};
document.getElementById("toggle-smooth").onchange = refreshChmLayers;
document.getElementById("toggle-hillshade").onchange = (e) => {
  hillLayer.visible(e.target.checked);
  hillLayer.batchDraw();
  renderContextMosaic();
};
document.getElementById("toggle-ortho").onchange = (e) => toggleOrtho(e.target.checked);
document.getElementById("toggle-heatmap").onchange = (e) => {
  if (e.target.checked && currentHeatmapMode === "off") currentHeatmapMode = "IntGrad";
  togglePrediction(e.target.checked, currentHeatmapMode);
};
document.getElementById("toggle-mask").onchange = (e) => {
  maskLayer.visible(e.target.checked);
  maskLayer.batchDraw();
  // Brush preview is on a separate layer, so it stays visible regardless.
};

// ---- Keyboard -------------------------------------------------------------

document.addEventListener("keydown", async (e) => {
  if (views.label.hidden) return;
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;

  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") {
    e.preventDefault();
    document.getElementById("undo-btn").click();
    return;
  }
  if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") {
    e.preventDefault();
    saveMask();
    return;
  }

  const k = e.key;
  if (k === "[") {
    const b = document.getElementById("brush-size");
    b.value = Math.max(+b.min, +b.value - 2);
    b.dispatchEvent(new Event("input"));
  } else if (k === "]") {
    const b = document.getElementById("brush-size");
    b.value = Math.min(+b.max, +b.value + 2);
    b.dispatchEvent(new Event("input"));
  } else if (k === "1") {
    document.getElementById("brush-class").value = "cwd";
    notice("class: CWD");
  } else if (k === "2") {
    document.getElementById("brush-class").value = "bg";
    notice("class: background");
  } else if (k === "3") {
    document.getElementById("brush-class").value = "erase";
    notice("class: erase");
  } else if (k === "w") {
    await cycleYear(+1);
  } else if (k === "s") {
    await cycleYear(-1);
  } else if (k === "a") {
    await cycleProduct(-1);
  } else if (k === "d") {
    await cycleProduct(+1);
  } else if (k === "g") {
    const c = document.getElementById("toggle-smooth");
    c.checked = !c.checked;
    notice("Gaussian " + (c.checked ? "on" : "off"));
    refreshChmLayers();
  } else if (k === "o") {
    const c = document.getElementById("toggle-ortho");
    c.checked = !c.checked;
    toggleOrtho(c.checked);
  } else if (k === "h") {
    await cycleHeatmapMode();
  } else if (k === "n") {
    await goToQueueOffset(1);
  } else if (k === "p") {
    await goToQueueOffset(-1);
  } else if (k === "b") {
    // Bulk-label shown tiles as background (no CWD)
    await labelAllAsBackground();
  } else if (k === "ArrowDown") {
    e.preventDefault();
    await goToQueueOffset(5);
  } else if (k === "Escape") {
    document.getElementById("back-btn").click();
  }
});

// ---- Init -----------------------------------------------------------------

let serverConfig = {};
(async function init() {
  const cfg = await api.config();
  serverConfig = cfg;
  CHIP = cfg.chip_size;
  if (!cfg.classifier_model_available) {
    console.warn(
      "Classifier model not available — predict/score will fall back to CHM-edge heatmap.",
      cfg.classifier_model_path
    );
  }

  initSidebarResizer();
  const rawSidebarWidth = parseInt(
    getComputedStyle(document.documentElement).getPropertyValue("--sidebar-width"),
    10
  );
  const sidebarWidth = Number.isFinite(rawSidebarWidth) ? rawSidebarWidth : 320;
  document.documentElement.style.setProperty(
    "--sidebar-width",
    `${Math.max(300, Math.min(500, sidebarWidth))}px`
  );

  setStatus(`chip ${CHIP}px · ${cfg.queue_size}/${cfg.total_chips} in queue`);
  await Promise.all([loadCatalog(), loadRasters()]);
  await loadQueue();
})();
