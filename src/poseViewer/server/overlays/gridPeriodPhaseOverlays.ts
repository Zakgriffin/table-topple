import * as THREE from 'three';
import { type Camera, type CompositeLine, type DecodeLattice } from '../../shared/camera/model.ts';
import { persistConfig } from '../../shared/config.ts';
import { activeCamera } from '../camera/store.ts';
import { type ProjectedBins } from '../../shared/types.ts';
import { gridPeriodPhasePlotSvg, projectedSvgOverlay, rectifiedLinesGroup, sampleLatticeGroup, toggleDistinctnessCurveBtn, toggleGapHistogramBtn, toggleProductCurveBtn, toggleValueHistogramBtn } from '../ui/dom.ts';
import { svgEl, svgText } from './svgUtil.ts';

// ── Grid period/phase debug visualizations — the PLOT is still empty ─────
//
// This file drew the period search's work: the [Pmin, Pmax] bracket, the
// integer-count resultant curve, each image-tested candidate with its
// distinctness, the pooled all-pairs gap histogram, the rectified line values,
// and the sample lattice on the Projected-Cam rect. Roughly 480 lines of it.
//
// THE SAMPLE LATTICE IS BACK (see the bottom of this file). The PLOT is not, and
// its data is the part that genuinely no longer exists: it all came from
// `camera.pose.gridPeriodPhase`, which the deleted the old pipeline produced. pose
// recovers the same period and phase on the DEVICE and without the search's
// working set -- no rowLines/colLines, no coarse sample curve, no candidate list
// on the host. Some of it is a readback away (`rowSamples`/`colSamples`,
// `scores`); the gap histogram is not, because pose enumerates `spread / n`
// directly and has no gap distribution to plot.
//
// So the plot is an EMPTY STATE rather than a deletion. The panel, its four
// toggle buttons and their persistence all still work -- the UI shape is intact
// and says why it is blank.

export function drawGridPeriodPhasePlot(_camera: Camera) {
  const svg = gridPeriodPhasePlotSvg;
  svg.style.display = 'block';
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const [, , vbW, vbH] = svg.getAttribute('viewBox')!.split(' ').map(Number);
  svg.appendChild(svgEl('rect', {
    x: 0, y: 0, width: vbW!, height: vbH!, fill: 'rgba(255,255,255,0.03)',
  }));
  svg.appendChild(svgText(vbW! / 2, vbH! / 2 - 6, 'no period/phase data', {
    fill: 'rgba(255,255,255,0.5)', 'font-size': 11, 'text-anchor': 'middle',
  }));
  svg.appendChild(svgText(vbW! / 2, vbH! / 2 + 10, 'awaiting pose intermediate readback', {
    fill: 'rgba(255,255,255,0.32)', 'font-size': 9, 'text-anchor': 'middle',
  }));
}

// The four histogram/curve toggles. They affect NOTHING but this plot -- no
// recompute, no capture, no other overlay -- so a redraw is the whole reaction.
// Kept live, and kept persisting, so the panel behaves as it always did.
function bindHistogramToggle(btn: HTMLButtonElement, get: (c: Camera) => boolean, set: (c: Camera, v: boolean) => void) {
  btn.addEventListener('click', () => {
    const cam = activeCamera(); if (!cam) return;
    set(cam, !get(cam));
    btn.classList.toggle('active', get(cam));
    persistConfig();
    drawGridPeriodPhasePlot(cam);
  });
}
bindHistogramToggle(toggleGapHistogramBtn,
  (c) => c.settings.showGapHistogram, (c, v) => { c.settings.showGapHistogram = v; });
bindHistogramToggle(toggleValueHistogramBtn,
  (c) => c.settings.showValueHistogram, (c, v) => { c.settings.showValueHistogram = v; });
bindHistogramToggle(toggleDistinctnessCurveBtn,
  (c) => c.settings.showDistinctnessCurve, (c, v) => { c.settings.showDistinctnessCurve = v; });
bindHistogramToggle(toggleProductCurveBtn,
  (c) => c.settings.showProductCurve, (c, v) => { c.settings.showProductCurve = v; });

// The pan/zoom handlers are gone with the plot they navigated: every one of
// them read `cam.pose.gridPeriodPhase` to get a view range and bailed when it
// was null, so against no data they were already no-ops. `gridPeriodPhaseViewMin`
// and `gridPeriodPhaseViewMax` stay on the Camera -- they are view state, not
// pipeline output, and the plot will want them back.

function clearGroups() {
  for (const g of [sampleLatticeGroup, rectifiedLinesGroup]) {
    while (g.firstChild) g.removeChild(g.firstChild);
  }
}

export function hideGridPeriodPhaseProjected() {
  clearGroups();
  projectedSvgOverlay.style.display = 'none';
  // The memo below has to be dropped WITH the elements it describes, or coming
  // back to Projected-Cam against an unchanged pose would find "nothing changed"
  // and leave the group empty.
  drawn = null;
}

// What the group currently holds, so an unchanged frame costs a few comparisons
// instead of a thousand DOM nodes.
//
// THIS MEMO IS NOT AN OPTIMIZATION, IT IS WHY SVG IS VIABLE HERE. main.ts calls
// the draw below from animate(), i.e. every frame while Projected-Cam is on
// screen. Clearing and rebuilding a few hundred <circle> elements at 60Hz is a
// different proposition from a few hundred canvas arcs, and it is also pure
// waste: the drawing changes only when the POSE changes (a new lattice object),
// the rect moves (a resize), or the rotation/toggle flips.
//
// Compared by IDENTITY on the two objects, which is exact rather than a
// heuristic: `pose.lattice` and `lastProjectedBins` are both rebuilt
// per capture, so a new one is a new object and an unchanged one is the same
// pointer.
let drawn: {
  lattice: DecodeLattice | undefined;
  // The COMPOSITES, because that is what drawRectifiedLines now projects --
  // memoizing on `rectified` would miss a frame where the segments moved but
  // the samples array happened to be reused.
  rectLines: { index: number; line: CompositeLine }[] | undefined;
  bins: ProjectedBins;
  x: number; y: number; w: number; h: number; rot: number;
} | null = null;

// ── BOTH PROJECTED-CAM OVERLAYS: the lattice, and the rectified lines ────
//
// The LATTICE is one dot per decode sample, filled by the bit it read and ringed
// by whether that bit matched the printed board -- off `pose.lattice`, see
// pipeline/decodeLattice.ts.
//
// The RECTIFIED LINES are each detected segment drawn as the straight line it
// becomes once the recovered axes flatten the floor: blue for the row family,
// red for the column family. This is the view that shows what the period search
// is actually looking at, because a line that rectifies badly -- crooked, or
// classified into the wrong family -- is exactly what makes a period wrong.
//
// SVG, LIKE THE THROUGH-CAM OVERLAYS, and no longer a canvas. Same shape as
// overlays/lsdOverlay.ts: a full-viewport <svg> that never moves, with children
// placed in SCREEN coordinates. Vectors rather than rasters, so they stay crisp
// at any zoom or DPR, and the element stops being resized and repositioned on
// every animation frame. What it costs is that a rebuild is DOM work, which the
// memo above keeps off the frame loop.
//
// EVERYTHING HERE LIVES IN FLOOR (u, v), the same space projectedBins.ts bins
// the image into, so placing anything is a bin lookup and no geometry.
// `rotationSteps` matches the true-cardinal display rotation applied to the
// texture underneath, and both groups share it.
export function drawGridPeriodPhaseProjected(
  camera: Camera, x: number, y: number, w: number, h: number, rotationSteps = 0,
) {
  const s = camera.settings;
  const lattice = s.showSampleLattice ? camera.pose?.lattice : undefined;
  const rectLines = s.showRectifiedLines ? camera.pose?.composites : undefined;
  const bins = camera.lastProjectedBins;
  if ((!lattice && !rectLines) || !bins) { hideGridPeriodPhaseProjected(); return; }
  if (drawn && drawn.lattice === lattice && drawn.rectLines === rectLines && drawn.bins === bins
    && drawn.x === x && drawn.y === y && drawn.w === w && drawn.h === h && drawn.rot === rotationSteps) return;

  clearGroups();
  projectedSvgOverlay.style.display = 'block';
  // ONE transform per group instead of the canvas's translate/rotate/translate.
  // SVG's rotate() takes the centre directly, and its positive direction is
  // clockwise in this y-down space, which is the same sense ctx.rotate had.
  const transform = rotationSteps === 0 ? null : `rotate(${rotationSteps * 90} ${x + w / 2} ${y + h / 2})`;
  for (const g of [sampleLatticeGroup, rectifiedLinesGroup]) {
    if (transform === null) g.removeAttribute('transform');
    else g.setAttribute('transform', transform);
  }

  // U is MIRRORED in this view (see projectedBins.ts's own bu), which is why
  // this reads `maxU - u` where v reads `v - minV`. Screen coordinates now, so
  // the rect's own origin is added -- the canvas was positioned AT the rect and
  // drew from zero.
  const toScreenU = (u: number) => x + ((bins.maxU - u) / bins.binWidthU / bins.w) * w;
  const toScreenV = (v: number) => y + (1 - (v - bins.minV) / bins.binWidthV / bins.h) * h;

  if (rectLines) drawRectifiedLines(camera, toScreenU, toScreenV);

  if (!lattice) { drawn = { lattice, rectLines, bins, x, y, w, h, rot: rotationSteps }; return; }
  const { rows, cols, uAt, vAt, packed, correct } = lattice;
  const frag = document.createDocumentFragment();
  for (let i = 0; i < rows; i++) {
    const cy = toScreenV(vAt[i]!);
    for (let j = 0; j < cols; j++) {
      const p = packed[i * cols + j]!;
      // Bit 0 clear means decode.build could not resolve the cell at all --
      // past the grazing cutoff, behind the camera, or off the image. No dot,
      // rather than a dot claiming a reading it never took.
      if ((p & 1) === 0) continue;
      const cx = toScreenU(uAt[j]!);

      // A SET bit is the DARK cell -- decode.build thresholds with `<`, so the
      // fill here is the cell as printed, not as sampled brightness.
      const bit = (p >> 1) & 1;
      // No ring where there is no verdict: an undecoded frame, or a
      // re-derivation that failed its own check against the device's counters.
      const verdict = correct ? correct[i * cols + j]! : -1;
      frag.appendChild(svgEl('circle', {
        cx, cy, r: 2.5,
        fill: bit ? '#000' : '#fff',
        stroke: verdict < 0 ? 'rgba(0,0,0,0.6)' : (verdict === 1 ? '#0f0' : '#f00'),
        'stroke-width': verdict < 0 ? 1 : 1.5,
      }));
    }
  }
  // One insertion rather than one per dot: appending into a live tree makes the
  // browser consider layout for each child.
  sampleLatticeGroup.appendChild(frag);

  drawn = { lattice, rectLines, bins, x, y, w, h, rot: rotationSteps };
}

// ── The detected segments, rectified -- PROJECTED, not parameterized ─────
//
// ── WHY THIS NO LONGER READS `samples` FOR THE GEOMETRY ──
//
// It used to draw each line from `samples`: `value` on one axis, the cross span
// on the other. That form CANNOT PRODUCE A TILTED LINE -- both endpoints were
// handed the same `value` -- so every line came out exactly horizontal or
// exactly vertical no matter how wrong the triad was. A 74-degree orientation
// error rendered as a tidy grid, and the one overlay whose job is to show
// whether the fit worked was structurally incapable of showing that it had not.
//
// The residual is not merely unrendered, it is DESTROYED UPSTREAM.
// GPP_CLASSIFY_WGSL projects both endpoints and then averages:
//
//     let xCol = (xCol1 + xCol2) * 0.5;
//
// and |xCol1 - xCol2| IS the misfit -- zero exactly when the line really does
// rectify to a constant coordinate, which is what a correct triad means. At a
// broken pose the median line's "constant" coordinate varied by 53% of the
// length the line ran, i.e. a ~28 degree tilt, all of it averaged away and then
// drawn flat.
//
// So this projects the composite's OWN TWO ENDPOINTS through the same gnomonic
// map, and draws the segment between them. With a good fit that is the same
// picture as before; with a bad one the lines fan and tilt, which is the whole
// point. `samples` is still what the PERIOD SEARCH consumes -- this is a display
// disagreeing with a summary statistic, not with the pipeline.
//
// GNOMONIC IN, FLOOR OUT: gnomonic coordinates times the camera's distance, the
// same `distance * x` decode.layout uses to place the lattice. That is why this
// needs `recoveredAxes` and draws nothing without it.
//
// `family` still comes from the pipeline rather than being re-derived here -- it
// is only the COLOUR, and the row/col predicate lives in GPP_CLASSIFY_WGSL.
function drawRectifiedLines(
  camera: Camera,
  toScreenU: (u: number) => number, toScreenV: (v: number) => number,
) {
  const axes = camera.pose?.recoveredAxes;
  const family = camera.pose?.lineFamily;
  const composites = camera.pose?.composites;
  if (!axes || !family || !composites) return;
  const { Drow, Dcol, Dnormal, distance, tanHalf, aspect } = axes;
  // A remote pose carries no projection -- see RecoveredAxes. Drawing nothing is
  // right: there is no ray to cast, and a guessed one would put every line in
  // the wrong place while looking entirely plausible.
  if (tanHalf === undefined || aspect === undefined) return;
  const { w, h } = camera.rtSize;

  // Byte-identical to GPP_CLASSIFY_WGSL's `rayDir`, which is itself byte-identical
  // to VOTES_WGSL's. Three copies of one projection is two too many, but the
  // alternative here is a fourth convention: the display MUST cast the ray the
  // device cast or it is drawing a different line.
  const ray = (px: number, py: number) => {
    const ndcU = (px / w) * 2 - 1;
    const ndcV = 1 - (py / h) * 2;
    return new THREE.Vector3(tanHalf * aspect * ndcU, tanHalf * ndcV, -1).normalize();
  };

  const frag = document.createDocumentFragment();
  const n = Math.min(composites.length, family.length);
  for (let i = 0; i < n; i++) {
    // -1 is a real third state: a degenerate line, or one whose endpoints had no
    // gnomonic image. It is not a column line, and drawing it as one would put a
    // segment at value 0 across the whole rect.
    const fam = family[i]!;
    if (fam < 0) continue;
    const { line } = composites[i]!;
    const r1 = ray(line.x1, line.y1), r2 = ray(line.x2, line.y2);
    const d1 = r1.dot(Dnormal), d2 = r2.dot(Dnormal);
    // The same guard the shader applies, and for the same reason: a ray nearly
    // parallel to the tangent plane has no gnomonic image, and dividing by it
    // puts an infinity into the path coordinates.
    if (Math.abs(d1) < 1e-9 || Math.abs(d2) < 1e-9) continue;
    const p1 = { u: (-r1.dot(Drow) / d1) * distance, v: (-r1.dot(Dcol) / d1) * distance };
    const p2 = { u: (-r2.dot(Drow) / d2) * distance, v: (-r2.dot(Dcol) / d2) * distance };
    frag.appendChild(svgEl('line', {
      x1: toScreenU(p1.u), y1: toScreenV(p1.v),
      x2: toScreenU(p2.u), y2: toScreenV(p2.v),
      stroke: fam === 1 ? 'rgba(60,140,255,0.8)' : 'rgba(255,60,60,0.8)',
      'stroke-width': 1.5,
    }));
  }
  rectifiedLinesGroup.appendChild(frag);
}
