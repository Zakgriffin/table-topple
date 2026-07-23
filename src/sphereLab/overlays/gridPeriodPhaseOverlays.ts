import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { projectedUVScale } from '../pipeline/decodeGrid.ts';
import { circularFit, GnomonicPoint, GridPeriodPhaseResult, PeriodSearchSample } from '../pipeline/gridPeriodPhase.ts';
import { DecodeCellDebug } from '../types.ts';
import { gridPeriodPhasePlotSvg, gridPeriodPhaseProjectedCanvas, gridPeriodPhaseProjectedCtx } from '../ui/dom.ts';

// ── Grid period/phase debug visualizations (pipeline/gridPeriodPhase.ts) ──

const SVG_NS = 'http://www.w3.org/2000/svg';
function svgEl<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>): SVGElementTagNameMap[K] {
  const el = document.createElementNS(SVG_NS, tag) as SVGElementTagNameMap[K];
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
  return el;
}
function svgText(x: number, y: number, content: string, attrs: Record<string, string | number>): SVGTextElement {
  const el = svgEl('text', { x, y, ...attrs });
  el.textContent = content;
  return el;
}

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
  if (!camera.settings.showGridPeriodPhaseDebug) { svg.style.display = 'none'; return; }
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
    svg.appendChild(svgText(8, H / 2, 'no data yet -- capture with the toggle on', { fill: '#888', 'font-size': 11, 'font-family': 'sans-serif' }));
    return;
  }
  const { pooledGaps, seedPeriod, bracket, coarseSamples } = gpp.debug;
  const marginBottom = 16, marginTop = 14;
  const plotH = H - marginBottom - marginTop;

  const [xMin, xMax] = getViewRange(camera, gpp);
  const span = xMax - xMin || 1;
  const xToPx = (x: number) => ((x - xMin) / span) * W;

  // Bracket shading -- the search never evaluates outside this range, see
  // gridPeriodPhase.ts's own comment on why it's kept deliberately narrow.
  svg.appendChild(svgEl('rect', {
    x: xToPx(bracket[0]), y: marginTop, width: Math.max(0, xToPx(bracket[1]) - xToPx(bracket[0])), height: plotH,
    fill: 'rgba(100,180,255,0.15)',
  }));

  // Step 3: histogram of pooled pairwise gaps -- SKIP (not clamp) anything
  // outside the plotted range. pooledGaps includes multi-period outliers
  // (see the xMin/xMax comment above), which now routinely fall outside this
  // bracket-relative range -- clamping them into the edge bin would pile up
  // a misleadingly tall bar there instead of just not showing them.
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

  // Step 4: a tick per coarse-sampled candidate period.
  for (const s of coarseSamples) {
    const px = xToPx(s.period);
    svg.appendChild(svgEl('line', { x1: px, y1: marginTop + plotH - 5, x2: px, y2: marginTop + plotH, stroke: 'rgba(255,255,255,0.3)', 'stroke-width': 1 }));
  }

  // Step 5 extension (VISUALIZATION ONLY -- not part of the real bracketed
  // search, see pipeline/gridPeriodPhase.ts's own comment on why the real
  // search stays deliberately narrow, and mergeAt-style "don't widen the
  // decision path" reasoning elsewhere in this session). If panning/zooming
  // reveals area outside the actual searched bracket, recompute the SAME
  // circular-resultant score out there too using the same row/col samples
  // -- otherwise the curve would just stop dead at the bracket edge. Drawn
  // BEFORE the real curve (so the real one renders on top at the boundary)
  // and in a lighter, more transparent orange to keep it visually distinct
  // from the actual decision-driving search.
  const rowValues = gpp.rowLines.map((s) => s.value), rowWeights = gpp.rowLines.map((s) => s.weight);
  const colValues = gpp.colLines.map((s) => s.value), colWeights = gpp.colLines.map((s) => s.weight);
  const bracketWidth = bracket[1] - bracket[0];
  const samplesPerUnit = bracketWidth > 1e-9 ? (coarseSamples.length - 1) / bracketWidth : 40;
  function extraSamples(lo: number, hi: number): PeriodSearchSample[] {
    if (hi <= lo) return [];
    const n = Math.max(4, Math.min(150, Math.round((hi - lo) * samplesPerUnit)));
    const out: PeriodSearchSample[] = [];
    for (let i = 0; i <= n; i++) {
      const period = lo + ((hi - lo) * i) / n;
      if (period <= 1e-9) continue;
      const rowFit = circularFit(rowValues, rowWeights, period);
      const colFit = circularFit(colValues, colWeights, period);
      out.push({ period, score: rowFit.resultant + colFit.resultant });
    }
    return out;
  }
  const leftExtra = xMin < bracket[0] ? extraSamples(xMin, bracket[0]) : [];
  const rightExtra = xMax > bracket[1] ? extraSamples(bracket[1], xMax) : [];

  // Shared height scale across the real search AND the extended (dimmer)
  // curve, so a stray taller peak outside the bracket doesn't get clipped.
  const maxScore = Math.max(1e-6, ...coarseSamples.map((s) => s.score), ...leftExtra.map((s) => s.score), ...rightExtra.map((s) => s.score));
  const toCurvePoints = (samples: PeriodSearchSample[]) =>
    samples.map((s) => `${xToPx(s.period)},${marginTop + plotH - (s.score / maxScore) * plotH}`).join(' ');
  if (leftExtra.length > 0) {
    svg.appendChild(svgEl('polyline', { points: toCurvePoints(leftExtra), fill: 'none', stroke: 'rgba(255,200,60,0.3)', 'stroke-width': 2 }));
  }
  if (rightExtra.length > 0) {
    svg.appendChild(svgEl('polyline', { points: toCurvePoints(rightExtra), fill: 'none', stroke: 'rgba(255,200,60,0.3)', 'stroke-width': 2 }));
  }

  // Step 5: the REAL R(P) search curve, scaled to the same height scale.
  svg.appendChild(svgEl('polyline', { points: toCurvePoints(coarseSamples), fill: 'none', stroke: 'rgb(255,200,60)', 'stroke-width': 2 }));

  // Seed estimate (dashed) and final winning period (solid green).
  const seedPx = xToPx(seedPeriod);
  svg.appendChild(svgEl('line', {
    x1: seedPx, y1: marginTop, x2: seedPx, y2: marginTop + plotH,
    stroke: 'rgba(160,160,160,0.7)', 'stroke-width': 1, 'stroke-dasharray': '3,3',
  }));
  const winPx = xToPx(gpp.period);
  svg.appendChild(svgEl('line', { x1: winPx, y1: marginTop, x2: winPx, y2: marginTop + plotH, stroke: 'rgb(80,255,120)', 'stroke-width': 2 }));

  svg.appendChild(svgText(2, H - 4, xMin.toFixed(3), { fill: '#aaa', 'font-size': 9, 'font-family': 'sans-serif' }));
  svg.appendChild(svgText(W - 34, H - 4, xMax.toFixed(3), { fill: '#aaa', 'font-size': 9, 'font-family': 'sans-serif' }));

  const heightStr = gpp.height !== null ? gpp.height.toFixed(3) : '—';
  svg.appendChild(svgText(
    4, 10, `P=${gpp.period.toFixed(4)}  h=${heightStr}  rows=${gpp.rowLines.length}  cols=${gpp.colLines.length}`,
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
// sample lattice, built purely from the recovered (period, phiRow, phiCol)
// -- gated on its own toggle (showSampleLattice). This replaced the
// original bins/autocorrelation-derived lattice (projectedCamOverlays.ts's
// now-unused drawSampleLattice) once this pipeline proved out.
//
// Both reuse camera.lastProjectedBins' own bounds/bin-size and the exact
// (bu,bv)->pixel convention drawSampleLattice used to (see
// projectedCamOverlays.ts) -- NOT a bounding box computed from this
// pipeline's own line endpoints -- so these lines land pixel-for-pixel on
// top of the actual bucketed image instead of drifting by whatever the
// (unrelated) extent of the detected lines happens to be. gpp's own
// {xRow,xCol} are converted into that same u/v space via projectedUVScale
// (pipeline/decodeGrid.ts), a single shared scalar.
export function drawGridPeriodPhaseProjected(camera: Camera, x: number, y: number, w: number, h: number) {
  const canvas = gridPeriodPhaseProjectedCanvas, ctx = gridPeriodPhaseProjectedCtx;
  const gpp = camera.lastGridPeriodPhase;
  const showLines = camera.settings.showGridPeriodPhaseDebug && gpp;
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
  ctx.clearRect(0, 0, canvas.width, canvas.height);

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
    const rowIdx = gpp.rowLines.map((s) => s.index), colIdx = gpp.colLines.map((s) => s.index);
    if (rowIdx.length > 0 && colIdx.length > 0) {
      const kMin = Math.min(...rowIdx) - 1, kMax = Math.max(...rowIdx) + 1;
      const mMin = Math.min(...colIdx) - 1, mMax = Math.max(...colIdx) + 1;
      // Same fill/stroke scheme drawSampleLattice (projectedCamOverlays.ts,
      // the OLD marginals/decode-driven lattice this one is meant to
      // replace) uses: fill = decoded bit (black=1, white=0, gray=no debug
      // data), stroke = correct (green) vs wrong (red) vs no data (dark).
      // This lattice's own (period, phiRow, phiCol) come from a totally
      // independent estimator (pipeline/gridPeriodPhase.ts) from decode's
      // own grid (pipeline/decodeGrid.ts's buildDecodeSampleGrid/
      // computeDecodeMarginals) -- rather than re-deriving a bit read here,
      // borrow whichever decode cell is physically nearest in (u,v) (same
      // units camera.lastProjectedBins/camera.lastDecodeRotated already
      // use, see projectedUVScale), since both lattices describe the same
      // real floor when both pipelines are working.
      const nearestDebug = (u: number, v: number): DecodeCellDebug | null => {
        const grid = camera.lastDecodeRotated, correctness = camera.lastDecodeCorrectness;
        if (!grid || !correctness) return null;
        let best: DecodeCellDebug | null = null, bestD2 = Infinity;
        for (let i = 0; i < grid.rows; i++) {
          for (let j = 0; j < grid.cols; j++) {
            const pt = grid.points[i][j];
            if (!pt.valid) continue;
            const d2 = (pt.u - u) * (pt.u - u) + (pt.v - v) * (pt.v - v);
            if (d2 < bestD2) { bestD2 = d2; best = correctness[i][j]; }
          }
        }
        return best;
      };
      for (let k = kMin; k <= kMax; k++) {
        for (let m = mMin; m <= mMax; m++) {
          // Row line k has constant xCol = k*period+phiRow; column line m
          // has constant xRow = m*period+phiCol -- their intersection is
          // this cell's own CORNER. Offset by half a period on each axis
          // to land on the cell's center instead, since that's what a
          // "sample point for this cell" should mean.
          const point: GnomonicPoint = {
            xRow: (m + 0.5) * gpp.period + gpp.phiCol,
            xCol: (k + 0.5) * gpp.period + gpp.phiRow,
          };
          const { px, py } = toScreen(point);
          const debug = nearestDebug(uvScale * point.xRow, uvScale * point.xCol);
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
