import { Camera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { GRID_STEP, MATH_QUAT } from '../constants.ts';
import { getAnalysisVFovRad } from '../pipeline/capture.ts';
import { projectedUVScale } from '../pipeline/decodeGrid.ts';
import { circularFit, computePooledGaps, GnomonicPoint, GridPeriodPhaseResult, makeCellCentreDistinctness, PeriodSearchSample } from '../pipeline/gridPeriodPhase.ts';
import { DecodeCellDebug } from '../types.ts';
import { gridPeriodPhasePlotSvg, gridPeriodPhaseProjectedCanvas, gridPeriodPhaseProjectedCtx, persistControl, toggleDistinctnessCurveBtn, toggleGapHistogramBtn, toggleProductCurveBtn, toggleValueHistogramBtn } from '../ui/dom.ts';
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

// Everything is plotted on one shared x-axis. For the gap histogram, the
// resultant curve and the candidate/chosen markers that's unambiguous: a
// pairwise gap and a candidate period are the exact same kind of quantity
// (spacing, not position), see gridPeriodPhase.ts's own header.
//
// The white VALUE histogram is deliberately drawn on that same axis anyway,
// and it is worth being explicit that this is a considered choice rather than
// an oversight, because the raw `value`s are POSITIONS, not spacings. What
// makes it defensible: values carry the same units and the same period (they
// are `phi + k*P`, so consecutive teeth sit exactly P apart), so the white
// comb's SPACING is directly comparable to the gray humps' spacing and to the
// green chosen-period line. What it costs: the white bars' absolute
// x-positions are anchored at the unknown phase phi and mean nothing on a
// period axis, so x means two different things depending on which series you
// read. Roughly a quarter of the comb's teeth land inside the default view
// (the view spans ~0.28*spread and teeth sit P = spread/n apart, so ~0.28*n of
// them), which is enough to read a spacing off. Both series are individually
// toggleable, so the ambiguity only exists when someone opts into it.
//
// SVG rather than a raster canvas -- crisp
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

  // ── The two optional histograms ────────────────────────────────────────
  //
  // Both are drawing-only: the period search itself never reads either one
  // (circularFit folds the O(n) line values directly). They share the
  // bucket-count slider and the plot's x-axis, but the axis means something
  // DIFFERENT for each, which is worth keeping straight when reading them
  // together:
  //   GRAY (pooled all-pairs gaps) is anchored at 0, so a bar's x-POSITION is
  //   a period -- the first hump sits at P, the next at 2P, and so on. That's
  //   what makes it directly comparable to the resultant curve and the green
  //   chosen-period line.
  //   WHITE (raw line values) is anchored at the grid's unknown phase, so a
  //   bar's x-position is just where some grid line happened to land and means
  //   nothing about periods. Only the SPACING between white bars carries P.
  // Both are combs of the same period regardless, which is the point of
  // showing them together -- white tooth spacing should match the gray hump
  // spacing, and both should match the green line's distance from x=0.
  //
  // Each is normalized to its OWN max (one counts gaps, the other counts
  // lines -- unrelated magnitudes), and each SKIPS rather than clamps
  // out-of-range samples so multi-period gap outliers can't pile a false bar
  // into the edge bin.
  const BINS = camera.settings.gridPeriodPhaseBinCount;
  const binOf = (v: number) => (v < xMin || v > xMax ? -1 : Math.min(BINS - 1, Math.max(0, Math.floor(((v - xMin) / span) * BINS))));
  const binCounts = (samples: readonly number[]) => {
    const counts = new Array<number>(BINS).fill(0);
    for (const v of samples) { const bi = binOf(v); if (bi >= 0) counts[bi]++; }
    return counts;
  };
  const showGaps = camera.settings.showGapHistogram, showValues = camera.settings.showValueHistogram;
  // When both are on, bars are drawn side-by-side WITHIN each bucket rather
  // than overlapping -- the white comb is sparse and the gray humps are broad,
  // so an overlay would mostly read as one series occluding the other exactly
  // where the comparison matters (does a white tooth line up with a gray
  // hump). Each takes the full bucket width when it's the only one shown.
  const drawHistogram = (samples: readonly number[], fill: string, slot: 0 | 1, slots: 1 | 2) => {
    const counts = binCounts(samples);
    const maxCount = Math.max(1, ...counts);
    for (let i = 0; i < BINS; i++) {
      if (counts[i] === 0) continue;
      const barH = (counts[i] / maxCount) * plotH * 0.55;
      const x0 = (i / BINS) * W, binW = W / BINS;
      const slotW = (binW - 2) / slots;
      svg.appendChild(svgEl('rect', {
        x: x0 + 1 + slot * slotW, y: marginTop + plotH - barH, width: Math.max(1, slotW), height: barH, fill,
      }));
    }
  };
  const slots: 1 | 2 = showGaps && showValues ? 2 : 1;
  if (showGaps) {
    // computePooledGaps is O(n^2) in the detected line count and exists purely
    // for this histogram, so it stays behind the toggle -- not computed at all
    // when the series is hidden. This is the whole reason the toggle gates
    // COMPUTATION and not just drawing.
    drawHistogram(computePooledGaps(gpp.rowLines, gpp.colLines), 'rgba(200,200,200,0.5)', 0, slots);
  }
  if (showValues) {
    const values = [...gpp.rowLines.map((s) => s.value), ...gpp.colLines.map((s) => s.value)];
    drawHistogram(values, 'rgba(255,255,255,0.75)', slots === 2 ? 1 : 0, slots);
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

  // ── Distinctness (blue) and resultant x distinctness (violet) ───────────
  //
  // Both are computed HERE, not in the pipeline, and both span the whole
  // current view the way the translucent resultant above does -- so they
  // re-evaluate on pan/zoom and stay meaningful outside the searched bracket.
  // That's the point: the most useful thing to see is whether the joint peak
  // sits somewhere the integer-count search never looked, which a
  // pipeline-precomputed curve (fixed to the bracket, fixed at capture time)
  // structurally cannot show.
  //
  // The product is the interesting one. The search decides in two stages
  // (threshold on resultant, then argmax on distinctness) precisely because
  // neither signal is sufficient alone: the resultant is blind between a
  // period and its sub-multiples (both fold perfectly), while distinctness is
  // blind to spurious LONG periods (they under-sample, so their neighbours
  // still differ). Their product is high only where BOTH are, which is the
  // conjunction the two-stage filter is approximating -- so a clean single
  // peak here says the staged decision landed where a continuous joint
  // criterion would have, and a peak somewhere else says it didn't.
  //
  // Each is normalized to its OWN max before multiplying: the resultant is
  // bounded (0..2, a sum of two unit resultants) but distinctness is a raw
  // mean grayscale difference with a data-dependent scale, so a raw product
  // would just be distinctness with a slight tilt.
  const wantDistinct = camera.settings.showDistinctnessCurve, wantProduct = camera.settings.showProductCurve;
  const axes = camera.lastRecoveredAxes;
  if ((wantDistinct || wantProduct) && axes && camera.lastNoisedPreviewGray) {
    // Deliberately coarser than TRANSLUCENT_SAMPLES: each sample reverse-
    // projects a 13x13 cell patch (~169 grayscale reads), and unlike the
    // resultant this redraws on every wheel/drag event. 200 samples of that
    // would make panning visibly chunky; 90 keeps the curve smooth enough to
    // read a peak off while staying responsive.
    const D_SAMPLES = 90;
    const distinctnessAt = makeCellCentreDistinctness({
      gray: camera.lastNoisedPreviewGray, w: camera.rtSize.w, h: camera.rtSize.h,
      quat: MATH_QUAT, vFovRad: getAnalysisVFovRad(camera), aspect: camera.aspect,
      Drow: axes.Drow, Dcol: axes.Dcol, Dnormal: axes.Dnormal,
      cellPitch: GRID_STEP, minGrazingCos: camera.settings.minGrazingCos,
      rowValues, colValues,
    });
    const pts: { period: number; score: number; d: number | null }[] = [];
    for (let i = 0; i < D_SAMPLES; i++) {
      const period = xMin + ((xMax - xMin) * i) / (D_SAMPLES - 1);
      if (period <= 1e-9) continue;
      // One fit per family, reused for both the resultant and the phase the
      // patch placement needs -- see makeCellCentreDistinctness' own comment.
      const fitRow = circularFit(rowValues, rowWeights, period), fitCol = circularFit(colValues, colWeights, period);
      pts.push({ period, score: fitRow.resultant + fitCol.resultant, d: distinctnessAt(period, fitRow.phase, fitCol.phase) });
    }
    const maxD = Math.max(1e-6, ...pts.map((s) => s.d ?? 0));
    // Drawn as SEGMENTS split on unsampled periods rather than one polyline
    // through them: a period whose patch fell off-image has no reading at all,
    // and bridging that with a straight line would invent one.
    const drawSplit = (valueOf: (s: typeof pts[number]) => number | null, stroke: string, width: number) => {
      let run: string[] = [];
      const flush = () => {
        if (run.length > 1) svg.appendChild(svgEl('polyline', { points: run.join(' '), fill: 'none', stroke, 'stroke-width': width }));
        run = [];
      };
      for (const s of pts) {
        const v = valueOf(s);
        if (v === null) { flush(); continue; }
        run.push(`${xToPx(s.period)},${marginTop + plotH - v * plotH}`);
      }
      flush();
    };
    if (wantDistinct) drawSplit((s) => (s.d === null ? null : s.d / maxD), 'rgba(120,220,255,0.6)', 1.5);
    if (wantProduct) drawSplit((s) => (s.d === null ? null : (s.score / maxScore) * (s.d / maxD)), 'rgb(190,120,255)', 2.5);
  }

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

// The two histogram toggles. Registered here rather than in ui/cameraPanel.ts
// because they affect NOTHING but this plot -- no recompute, no capture, no
// other overlay -- so a plain redraw is the entire reaction. Persistence key
// matches the BUTTON's own id, the same convention every other button-driven
// boolean in camera/settings.ts uses.
function bindHistogramToggle(btn: HTMLButtonElement, id: string, get: (c: Camera) => boolean, set: (c: Camera, v: boolean) => void) {
  btn.addEventListener('click', () => {
    const cam = activeCamera(); if (!cam) return;
    set(cam, !get(cam));
    btn.classList.toggle('active', get(cam));
    persistControl(id, get(cam) ? '1' : '0');
    drawGridPeriodPhasePlot(cam);
  });
}
bindHistogramToggle(toggleGapHistogramBtn, 'toggleGapHistogram',
  (c) => c.settings.showGapHistogram, (c, v) => { c.settings.showGapHistogram = v; });
bindHistogramToggle(toggleValueHistogramBtn, 'toggleValueHistogram',
  (c) => c.settings.showValueHistogram, (c, v) => { c.settings.showValueHistogram = v; });
bindHistogramToggle(toggleDistinctnessCurveBtn, 'toggleDistinctnessCurve',
  (c) => c.settings.showDistinctnessCurve, (c, v) => { c.settings.showDistinctnessCurve = v; });
bindHistogramToggle(toggleProductCurveBtn, 'toggleProductCurve',
  (c) => c.settings.showProductCurve, (c, v) => { c.settings.showProductCurve = v; });

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
