import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { projectedUVScale } from '../pipeline/decodeGrid.ts';
import { circularFit, computePooledGaps, GnomonicPoint, GridPeriodPhaseResult, median, PeriodSearchSample } from '../pipeline/gridPeriodPhase.ts';
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
  const { rowNeighborGaps, colNeighborGaps, seedPeriod, bracket, coarseSamples } = gpp.debug;
  // Computed here, on demand, rather than eagerly inside computeGridPeriodPhase
  // itself -- see that function's own comment (O(n^2), debug-only, used to
  // run on every capture regardless of whether this plot was ever drawn).
  const pooledGaps = computePooledGaps(gpp.rowLines, gpp.colLines);
  const marginBottom = 16, marginTop = 14;
  const plotH = H - marginBottom - marginTop;

  const [xMin, xMax] = getViewRange(camera, gpp);
  const span = xMax - xMin || 1;
  const xToPx = (x: number) => ((x - xMin) / span) * W;
  const gapLowerBound = camera.settings.gridPeriodPhaseGapLowerBound;

  // Bracket shading -- the search never evaluates outside this range, see
  // gridPeriodPhase.ts's own comment on why it's kept deliberately narrow.
  svg.appendChild(svgEl('rect', {
    x: xToPx(bracket[0]), y: marginTop, width: Math.max(0, xToPx(bracket[1]) - xToPx(bracket[0])), height: plotH,
    fill: 'rgba(100,180,255,0.15)',
  }));

  // Gray "excluded by gap lower bound" band, [0, gapLowerBound] -- the exact
  // same threshold the per-family median lines below prune by, so a glance
  // shows how much of the visible ticks/histogram sits inside the zone
  // being excluded from those medians.
  svg.appendChild(svgEl('rect', {
    x: xToPx(0), y: marginTop, width: Math.max(0, xToPx(gapLowerBound) - xToPx(0)), height: plotH,
    fill: 'rgba(128,128,128,0.25)',
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

  // rowNeighborGaps/colNeighborGaps: unlike pooledGaps' O(n^2) all-pairs mix
  // of both families, these are each family's own n-1 genuinely-adjacent
  // gaps (sorted values, consecutive differences) -- ticks at the TOP of the
  // plot, opposite the bottom-anchored histogram/candidate ticks below, so a
  // family's real adjacent spacings can be compared directly against where
  // the pooled histogram piles up. Drawn outside the [xMin,xMax] skip used
  // for pooledGaps too, for the same reason (see the comment above Step 3).
  // Colored by `rank` (each gap's two lines' averaged position in their
  // family's value-sorted order) using the EXACT same rank->channel mapping
  // drawVoteFamilyLines uses for the composite lines themselves
  // (hoverDebugOverlays.ts) -- rgb(0,0,rank*255) for rows, rgb(rank*255,0,0)
  // for columns -- so a tick's color visually traces back to the specific
  // pair of composite lines that produced it.
  for (const { gap, rank } of rowNeighborGaps) {
    if (gap < xMin || gap > xMax) continue;
    const px = xToPx(gap);
    svg.appendChild(svgEl('line', { x1: px, y1: marginTop, x2: px, y2: marginTop + 10, stroke: `rgb(0,0,${Math.round(rank * 255)})`, 'stroke-width': 1 }));
  }
  for (const { gap, rank } of colNeighborGaps) {
    if (gap < xMin || gap > xMax) continue;
    const px = xToPx(gap);
    svg.appendChild(svgEl('line', { x1: px, y1: marginTop, x2: px, y2: marginTop + 10, stroke: `rgb(${Math.round(rank * 255)},0,0)`, 'stroke-width': 1 }));
  }

  // Tall solid line at each family's own MEDIAN neighbor gap -- same
  // full-height treatment as the winning-period line further down, but one
  // per family (not pooled), so the two can be compared directly against
  // each other and against the pooled seed/winning period. Gaps below the
  // gap-lower-bound slider are pruned out FIRST (before median()'s own
  // sort) -- the same near-duplicate-line noise gaps the seed-mode search
  // in gridPeriodPhase.ts is built to shrug off via its histogram-mode
  // trick, but a plain median over a small per-family sample has no such
  // protection and would get dragged toward ~0 by even one of them.
  const rowGapMedian = median(rowNeighborGaps.map((s) => s.gap).filter((g) => g >= gapLowerBound));
  if (rowGapMedian !== null) {
    const px = xToPx(rowGapMedian);
    svg.appendChild(svgEl('line', { x1: px, y1: marginTop, x2: px, y2: marginTop + plotH, stroke: 'rgb(80,140,255)', 'stroke-width': 2 }));
  }
  const colGapMedian = median(colNeighborGaps.map((s) => s.gap).filter((g) => g >= gapLowerBound));
  if (colGapMedian !== null) {
    const px = xToPx(colGapMedian);
    svg.appendChild(svgEl('line', { x1: px, y1: marginTop, x2: px, y2: marginTop + plotH, stroke: 'rgb(255,90,90)', 'stroke-width': 2 }));
  }

  // Step 4: a tick per coarse-sampled candidate period.
  for (const s of coarseSamples) {
    const px = xToPx(s.period);
    svg.appendChild(svgEl('line', { x1: px, y1: marginTop + plotH - 5, x2: px, y2: marginTop + plotH, stroke: 'rgba(255,255,255,0.3)', 'stroke-width': 1 }));
  }

  // Step 5 extension (VISUALIZATION ONLY -- not part of the real bracketed
  // search, see pipeline/gridPeriodPhase.ts's own comment on why the real
  // search stays deliberately narrow, and mergeAt-style "don't widen the
  // decision path" reasoning elsewhere in this session). Recomputes the SAME
  // circular-resultant score across the plot's ENTIRE current view range
  // [xMin, xMax] -- not just the area outside the bracket -- at a fixed
  // TRANSLUCENT_SAMPLES evenly-spaced points, redone on every redraw
  // (including pan/zoom, since this whole function reruns on those) so
  // resolution always matches whatever span is on screen right now instead
  // of thinning out as you zoom in or clumping as you zoom out. Drawn
  // BEFORE the real curve (so the real one renders on top over the
  // bracket) and in a lighter, more transparent orange to keep it visually
  // distinct from the actual decision-driving search.
  const rowValues = gpp.rowLines.map((s) => s.value), rowWeights = gpp.rowLines.map((s) => s.weight);
  const colValues = gpp.colLines.map((s) => s.value), colWeights = gpp.colLines.map((s) => s.weight);
  const TRANSLUCENT_SAMPLES = 100;
  const translucentSamples: PeriodSearchSample[] = [];
  for (let i = 0; i < TRANSLUCENT_SAMPLES; i++) {
    const period = xMin + ((xMax - xMin) * i) / (TRANSLUCENT_SAMPLES - 1);
    if (period <= 1e-9) continue;
    const rowFit = circularFit(rowValues, rowWeights, period);
    const colFit = circularFit(colValues, colWeights, period);
    translucentSamples.push({ period, score: rowFit.resultant + colFit.resultant });
  }

  // Shared height scale across the real search AND the translucent curve,
  // so a stray taller peak outside the bracket doesn't get clipped.
  const maxScore = Math.max(1e-6, ...coarseSamples.map((s) => s.score), ...translucentSamples.map((s) => s.score));
  const toCurvePoints = (samples: PeriodSearchSample[]) =>
    samples.map((s) => `${xToPx(s.period)},${marginTop + plotH - (s.score / maxScore) * plotH}`).join(' ');
  if (translucentSamples.length > 0) {
    svg.appendChild(svgEl('polyline', { points: toCurvePoints(translucentSamples), fill: 'none', stroke: 'rgba(255,200,60,0.3)', 'stroke-width': 2 }));
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
