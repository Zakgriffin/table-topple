import { Camera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { GRID_STEP } from '../constants.ts';
import { projectedUVScale } from '../pipeline/decodeGrid.ts';
import { circularFit, computePooledGaps, GnomonicPoint, GridPeriodPhaseResult, PeriodSearchSample } from '../pipeline/gridPeriodPhase.ts';
import { DecodeCellDebug } from '../types.ts';
import { gridPeriodPhasePlotSvg, gridPeriodPhaseProjectedCanvas, gridPeriodPhaseProjectedCtx } from '../ui/dom.ts';
import { svgEl, svgText } from './svgUtil.ts';

// ── Grid period/phase debug visualizations (pipeline/gridPeriodPhase.ts) ──

// Default view: a modest 1.1x padding around the search BRACKET's own
// width, centered on its midpoint -- deliberately NOT based on pooledGaps'
// own min/max: pooledGaps is every PAIRWISE difference between detected
// lines, so it routinely contains gaps spanning 2-3x the true period (e.g.
// row-1-to-row-3) even though the seed itself correctly ignores those (uses
// the MODE of the distribution, not an outlier-prone smallest-few average --
// see gridPeriodPhase.ts). Basing the default view on that raw extent let
// one outlier gap dominate the display. Interactive pan/zoom (wheel + drag,
// wired below) overrides this once the user touches the plot -- see
// camera.gridPeriodPhaseViewMin/Max's own comment in camera/model.ts.
function defaultViewRange(gpp: GridPeriodPhaseResult): [number, number] {
  const { bracket } = gpp.debug;
  const center = (bracket[0] + bracket[1]) / 2;
  const halfSpan = ((bracket[1] - bracket[0]) / 2) * 1.1;
  return [Math.max(0, center - halfSpan), center + halfSpan];
}
function getViewRange(camera: Camera, gpp: GridPeriodPhaseResult): [number, number] {
  if (camera.gridPeriodPhaseViewMin !== null && camera.gridPeriodPhaseViewMax !== null) {
    return [camera.gridPeriodPhaseViewMin, camera.gridPeriodPhaseViewMax];
  }
  return defaultViewRange(gpp);
}

// Steps 3/4/5 (seed, bracketed search, final phase) all plotted on one
// shared x-axis, since a pairwise gap and a candidate period are the exact
// same kind of quantity (spacing, not position) -- see gridPeriodPhase.ts's
// own header for why that's true and why it's NOT true of the raw
// row/column `value`s themselves. SVG rather than a raster canvas -- crisp
// at any zoom/DPR, and elements are individually inspectable in devtools,
// which matters more for a debug plot than raw draw-call throughput ever
// would here (a few dozen shapes, redrawn once per capture, not per frame).
export function drawGridPeriodPhasePlot(camera: Camera) {
  const svg = gridPeriodPhasePlotSvg;
  svg.style.display = 'block';
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  // Fills whatever width its container actually gives it (the SVG's own CSS
  // width is 100%, see sphere-lab.html) instead of staying a fixed 340 --
  // the viewBox is resynced to the real rendered width on every draw so the
  // internal coordinate math below stays a direct 1:1 pixel mapping (no
  // stretch distortion), only the HEIGHT stays fixed.
  const [, , , vbH] = svg.getAttribute('viewBox')!.split(' ').map(Number);
  const W = Math.max(200, svg.clientWidth), H = vbH;
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);

  const gpp = camera.lastGridPeriodPhase;
  if (!gpp) {
    svg.appendChild(svgText(8, H / 2, 'no data yet -- capture now', { fill: '#888', 'font-size': 11, 'font-family': 'sans-serif' }));
    return;
  }
  const { bracket, coarseSamples, chosenPeriod, candidates } = gpp.debug;
  // Pooled all-pairs gap histogram -- computed here on demand (O(n^2),
  // debug-only). KEPT as the "eyeball the periodicity humps" backdrop; the
  // old red/blue per-family neighbor-gap ticks + median lines + seed marker
  // are gone (they visualized the old median-seed search, which no longer
  // exists -- see gridPeriodPhase.ts).
  const pooledGaps = computePooledGaps(gpp.rowLines, gpp.colLines);
  const marginBottom = 16, marginTop = 14;
  const plotH = H - marginBottom - marginTop;

  const [xMin, xMax] = getViewRange(camera, gpp);
  const span = xMax - xMin || 1;
  const xToPx = (x: number) => ((x - xMin) / span) * W;

  // Bracket shading -- the integer-count search evaluates only within [Pmin, Pmax].
  svg.appendChild(svgEl('rect', {
    x: xToPx(bracket[0]), y: marginTop, width: Math.max(0, xToPx(bracket[1]) - xToPx(bracket[0])), height: plotH,
    fill: 'rgba(100,180,255,0.15)',
  }));

  // Gray pooled-pairwise-gap histogram -- unchanged. SKIP (not clamp) gaps
  // outside the plotted range (multi-period outliers routinely fall outside),
  // so they don't pile a false bar into the edge bin.
  const BINS = camera.settings.gridPeriodPhaseBinCount;
  const counts = new Array(BINS).fill(0);
  for (const g of pooledGaps) {
    if (g < xMin || g > xMax) continue;
    const bi = Math.min(BINS - 1, Math.max(0, Math.floor(((g - xMin) / span) * BINS)));
    counts[bi]++;
  }
  const maxCount = Math.max(1, ...counts);
  for (let i = 0; i < BINS; i++) {
    if (counts[i] === 0) continue;
    const barH = (counts[i] / maxCount) * plotH * 0.55;
    const x0 = (i / BINS) * W, x1 = ((i + 1) / BINS) * W;
    svg.appendChild(svgEl('rect', {
      x: x0 + 1, y: marginTop + plotH - barH, width: Math.max(1, x1 - x0 - 2), height: barH,
      fill: 'rgba(200,200,200,0.5)',
    }));
  }

  // R(P) resultant curve. The translucent extension recomputes the score
  // across the WHOLE current view (so it stays smooth when panned/zoomed past
  // the searched bracket); the solid curve is the actual integer-count search.
  const rowValues = gpp.rowLines.map((s) => s.value), rowWeights = gpp.rowLines.map((s) => s.weight);
  const colValues = gpp.colLines.map((s) => s.value), colWeights = gpp.colLines.map((s) => s.weight);
  const TRANSLUCENT_SAMPLES = 200;
  const translucentSamples: PeriodSearchSample[] = [];
  for (let i = 0; i < TRANSLUCENT_SAMPLES; i++) {
    const period = xMin + ((xMax - xMin) * i) / (TRANSLUCENT_SAMPLES - 1);
    if (period <= 1e-9) continue;
    translucentSamples.push({ period, score: circularFit(rowValues, rowWeights, period).resultant + circularFit(colValues, colWeights, period).resultant });
  }
  const maxScore = Math.max(1e-6, ...coarseSamples.map((s) => s.score), ...translucentSamples.map((s) => s.score));
  const toCurvePoints = (samples: PeriodSearchSample[]) =>
    samples.map((s) => `${xToPx(s.period)},${marginTop + plotH - (s.score / maxScore) * plotH}`).join(' ');
  if (translucentSamples.length > 0) {
    svg.appendChild(svgEl('polyline', { points: toCurvePoints(translucentSamples), fill: 'none', stroke: 'rgba(255,200,60,0.3)', 'stroke-width': 2 }));
  }
  svg.appendChild(svgEl('polyline', { points: toCurvePoints(coarseSamples), fill: 'none', stroke: 'rgb(255,200,60)', 'stroke-width': 2 }));

  // Image-tested candidate peaks: a circle per candidate whose HEIGHT encodes
  // its cell-centre distinctness (higher = more distinct), normalized across
  // the candidates. The chosen one (whose distinctness won) is green, the
  // rejected sub-multiples orange -- so a glance shows the true period sitting
  // ABOVE its sub-multiples in distinctness, which is exactly why it was
  // picked. Candidates with no distinctness (couldn't sample) sit on the axis.
  const maxDistinct = Math.max(1e-6, ...candidates.map((c) => c.distinctness ?? 0));
  for (const c of candidates) {
    const px = xToPx(c.period);
    if (px < 0 || px > W) continue;
    const isChosen = Math.abs(c.period - chosenPeriod) < 1e-9;
    const dNorm = c.distinctness !== null ? c.distinctness / maxDistinct : 0;
    const cy = marginTop + plotH - dNorm * plotH * 0.5;
    svg.appendChild(svgEl('line', { x1: px, y1: marginTop + plotH, x2: px, y2: cy, stroke: isChosen ? 'rgba(80,255,120,0.5)' : 'rgba(255,150,60,0.4)', 'stroke-width': 1 }));
    svg.appendChild(svgEl('circle', {
      cx: px, cy, r: isChosen ? 4 : 3,
      fill: isChosen ? 'rgb(80,255,120)' : 'none', stroke: isChosen ? 'rgb(80,255,120)' : 'rgb(255,150,60)', 'stroke-width': 1.5,
    }));
  }

  // Final chosen period (solid green), and -- for a simulated camera -- the
  // TRUE period (magenta dashed) as an accuracy reference.
  const winPx = xToPx(gpp.period);
  svg.appendChild(svgEl('line', { x1: winPx, y1: marginTop, x2: winPx, y2: marginTop + plotH, stroke: 'rgb(80,255,120)', 'stroke-width': 2 }));
  if (isSimulated(camera)) {
    const truePeriod = GRID_STEP / camera.settings.camY;
    const tpx = xToPx(truePeriod);
    if (tpx >= 0 && tpx <= W) {
      svg.appendChild(svgEl('line', { x1: tpx, y1: marginTop, x2: tpx, y2: marginTop + plotH, stroke: 'rgb(255,80,220)', 'stroke-width': 1.5, 'stroke-dasharray': '4,3' }));
    }
  }

  svg.appendChild(svgText(2, H - 4, xMin.toFixed(3), { fill: '#aaa', 'font-size': 9, 'font-family': 'sans-serif' }));
  svg.appendChild(svgText(W - 34, H - 4, xMax.toFixed(3), { fill: '#aaa', 'font-size': 9, 'font-family': 'sans-serif' }));

  const heightStr = gpp.height !== null ? gpp.height.toFixed(3) : '—';
  svg.appendChild(svgText(
    4, 10, `P=${gpp.period.toFixed(4)}  h=${heightStr}  rows=${gpp.rowLines.length}  cols=${gpp.colLines.length}  cand=${candidates.length}`,
    { fill: '#ddd', 'font-size': 10, 'font-family': 'sans-serif' },
  ));
}

// ── Interactive pan/zoom -- wheel to zoom (centered on the cursor), drag to
// pan. Registered once at module scope (not per-draw); reads/writes the
// ACTIVE camera's own gridPeriodPhaseViewMin/Max (see camera/model.ts's own
// comment on why that state lives there, not in settings) and redraws.
gridPeriodPhasePlotSvg.addEventListener('wheel', (e) => {
  const cam = activeCamera();
  if (!cam || !cam.lastGridPeriodPhase) return;
  e.preventDefault();
  const [min, max] = getViewRange(cam, cam.lastGridPeriodPhase);
  const rect = gridPeriodPhasePlotSvg.getBoundingClientRect();
  const cursorFrac = Math.min(1, Math.max(0, (e.clientX - rect.left) / rect.width));
  const cursorValue = min + cursorFrac * (max - min);
  const zoomFactor = e.deltaY > 0 ? 1.15 : 1 / 1.15; // scroll down/back = zoom out, up/forward = zoom in
  cam.gridPeriodPhaseViewMin = Math.max(0, cursorValue - (cursorValue - min) * zoomFactor);
  cam.gridPeriodPhaseViewMax = cursorValue + (max - cursorValue) * zoomFactor;
  drawGridPeriodPhasePlot(cam);
}, { passive: false });

let dragStartClientX: number | null = null;
let dragStartMin = 0, dragStartMax = 0;
gridPeriodPhasePlotSvg.addEventListener('mousedown', (e) => {
  const cam = activeCamera();
  if (!cam || !cam.lastGridPeriodPhase) return;
  [dragStartMin, dragStartMax] = getViewRange(cam, cam.lastGridPeriodPhase);
  dragStartClientX = e.clientX;
  gridPeriodPhasePlotSvg.style.cursor = 'grabbing';
});
addEventListener('mousemove', (e) => {
  if (dragStartClientX === null) return;
  const cam = activeCamera();
  if (!cam) return;
  const rect = gridPeriodPhasePlotSvg.getBoundingClientRect();
  const span = dragStartMax - dragStartMin;
  const shift = (-(e.clientX - dragStartClientX) / rect.width) * span; // dragging right pans the view left (content follows the cursor)
  cam.gridPeriodPhaseViewMin = Math.max(0, dragStartMin + shift);
  cam.gridPeriodPhaseViewMax = dragStartMax + shift;
  drawGridPeriodPhasePlot(cam);
});
addEventListener('mouseup', () => {
  dragStartClientX = null;
  gridPeriodPhasePlotSvg.style.cursor = 'grab';
});
gridPeriodPhasePlotSvg.addEventListener('dblclick', () => {
  const cam = activeCamera();
  if (!cam) return;
  cam.gridPeriodPhaseViewMin = null;
  cam.gridPeriodPhaseViewMax = null;
  drawGridPeriodPhasePlot(cam);
});
// Re-fit to the container's width if it changes (e.g. window resize) --
// see the W computation in drawGridPeriodPhasePlot's own comment.
addEventListener('resize', () => {
  const cam = activeCamera();
  if (cam) drawGridPeriodPhasePlot(cam);
});

// Bounding box of every detected line's own gnomonically-projected
// endpoints, in (xRow, xCol) tangent-plane space -- the shared coordinate
// frame both the rectified-lines overlay and the sample lattice below map
// onto whatever screen rect they're given.
export function hideGridPeriodPhaseProjected() {
  gridPeriodPhaseProjectedCanvas.style.display = 'none';
}

// Draws, on the Projected-Cam rect: (1) every detected composite line,
// gnomonically rectified (straight, per the whole point of the projection)
// -- shown whenever the debug pipeline is on, no separate toggle, since
// it's the direct visual evidence behind the period/phase numbers; (2) the
// sample lattice -- gated on its own toggle (showSampleLattice), drawing the
// real decode grid's own points (pipeline/decodeGrid.ts's
// buildDecodeSampleGrid) directly, so it always matches decode's actual
// corner-quad-bounded extent instead of an independently re-derived one.
//
// Both reuse camera.lastProjectedBins' own bounds/bin-size and the exact
// (bu,bv)->pixel convention drawSampleLattice used to (see
// projectedCamOverlays.ts) -- NOT a bounding box computed from this
// pipeline's own line endpoints -- so these lines land pixel-for-pixel on
// top of the actual bucketed image instead of drifting by whatever the
// (unrelated) extent of the detected lines happens to be. gpp's own
// {xRow,xCol} are converted into that same u/v space via projectedUVScale
// (pipeline/decodeGrid.ts), a single shared scalar.
// rotationSteps: multiples of 90 degrees (0-3) -- see renderProjectedViewport's
// matching param (scene/quadRenderers.ts), same "use true cardinal
// orientation" toggle. Applied as a canvas transform around the rect's own
// center, before any drawing, so every draw call below it (lines, lattice)
// lands rotated consistently without needing its own rotated coordinates.
export function drawGridPeriodPhaseProjected(camera: Camera, x: number, y: number, w: number, h: number, rotationSteps = 0) {
  const canvas = gridPeriodPhaseProjectedCanvas, ctx = gridPeriodPhaseProjectedCtx;
  const gpp = camera.lastGridPeriodPhase;
  const showLines = !!gpp;
  const showLattice = camera.settings.showSampleLattice && gpp;
  if (!showLines && !showLattice) { hideGridPeriodPhaseProjected(); return; }
  const bins = camera.lastProjectedBins;
  const uvScale = projectedUVScale(camera);
  if (!bins || uvScale === null) { hideGridPeriodPhaseProjected(); return; }

  canvas.style.display = 'block';
  canvas.style.left = x + 'px';
  canvas.style.top = y + 'px';
  canvas.width = Math.round(w);
  canvas.height = Math.round(h);
  canvas.style.width = w + 'px';
  canvas.style.height = h + 'px';
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (rotationSteps !== 0) {
    ctx.translate(canvas.width / 2, canvas.height / 2);
    ctx.rotate(rotationSteps * (Math.PI / 2));
    ctx.translate(-canvas.width / 2, -canvas.height / 2);
  }

  const toScreen = (p: GnomonicPoint) => {
    const u = uvScale * p.xRow, v = uvScale * p.xCol;
    const bu = (bins.maxU - u) / bins.binWidthU, bv = (v - bins.minV) / bins.binWidthV;
    return {
      px: (bu / bins.w) * canvas.width,
      py: (1 - bv / bins.h) * canvas.height,
    };
  };

  if (showLines && gpp) {
    for (const s of gpp.rowLines) {
      const a = toScreen(s.p1), b = toScreen(s.p2);
      ctx.strokeStyle = 'rgba(60,140,255,0.8)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
    }
    for (const s of gpp.colLines) {
      const a = toScreen(s.p1), b = toScreen(s.p2);
      ctx.strokeStyle = 'rgba(255,60,60,0.8)'; ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(a.px, a.py); ctx.lineTo(b.px, b.py); ctx.stroke();
    }
  }

  if (showLattice && gpp) {
    // Draws the REAL decode grid's own points (pipeline/decodeGrid.ts's
    // buildDecodeSampleGrid, corner-quad-bounded) rather than re-deriving an
    // independent loop range from the detected composite lines' own index
    // spread -- that used to draw a dot for every cell in a rectangle padded
    // around the line detections, regardless of whether decode considered it
    // inside the actual visible quad, which drifted from (and could draw
    // well outside) the true bounds. camera.lastDecodeRotated is preferred
    // (its indices line up with lastDecodeCorrectness), falling back to
    // lastDecodeGrid (pre-rotation -- same u/v values either way, since
    // rotation only permutes array indices, see decodeGrid.ts's readRotated)
    // so the lattice still shows something before a De Bruijn match is found.
    const grid = camera.lastDecodeRotated ?? camera.lastDecodeGrid;
    const correctness = camera.lastDecodeRotated ? camera.lastDecodeCorrectness : null;
    if (grid) {
      for (let i = 0; i < grid.rows; i++) {
        for (let j = 0; j < grid.cols; j++) {
          const pt = grid.points[i][j];
          if (!pt.valid) continue; // outside the quad (or failed grazing) -- zero, no dot
          const bu = (bins.maxU - pt.u) / bins.binWidthU, bv = (pt.v - bins.minV) / bins.binWidthV;
          const px = (bu / bins.w) * canvas.width, py = (1 - bv / bins.h) * canvas.height;
          const debug: DecodeCellDebug | null = correctness ? correctness[i][j] : null;
          ctx.beginPath(); ctx.arc(px, py, 2.5, 0, Math.PI * 2);
          ctx.fillStyle = debug ? (debug.bit ? '#000' : '#fff') : '#888';
          ctx.fill();
          ctx.strokeStyle = debug ? (debug.correct ? '#0f0' : '#f00') : 'rgba(0,0,0,0.6)';
          ctx.lineWidth = debug ? 1.5 : 1;
          ctx.stroke();
        }
      }
    }
  }
}
