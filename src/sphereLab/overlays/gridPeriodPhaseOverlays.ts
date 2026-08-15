import { type Camera } from '../camera/model.ts';
import { persistConfig } from '../config.ts';
import { activeCamera } from '../camera/store.ts';
import { gridPeriodPhasePlotSvg, gridPeriodPhaseProjectedCanvas, gridPeriodPhaseProjectedCtx, toggleDistinctnessCurveBtn, toggleGapHistogramBtn, toggleProductCurveBtn, toggleValueHistogramBtn } from '../ui/dom.ts';
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
// `camera.pose.gridPeriodPhase`, which the deleted `src/pose` produced. pose2
// recovers the same period and phase on the DEVICE and without the search's
// working set -- no rowLines/colLines, no coarse sample curve, no candidate list
// on the host. Some of it is a readback away (`rowSamples`/`colSamples`,
// `scores`); the gap histogram is not, because pose2 enumerates `spread / n`
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
  svg.appendChild(svgText(vbW! / 2, vbH! / 2 + 10, 'awaiting pose2 intermediate readback', {
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

export function hideGridPeriodPhaseProjected() {
  gridPeriodPhaseProjectedCanvas.style.display = 'none';
}

// ── THE SAMPLE LATTICE IS BACK; THE RECTIFIED LINES ARE NOT ──────────────
//
// This canvas drew two things over the Projected-Cam rect. The LATTICE -- one
// dot per decode sample, filled by the bit it read and ringed by whether that
// bit matched the printed board -- is below, off `intermediates.lattice` (see
// pipeline/decodeLattice.ts).
//
// The RECTIFIED LINES (blue row family, red column family) are still dark. They
// came from `gpp.rowLines`/`colLines`, which do not exist: pose2's `rowSamples`/
// `colSamples` carry (value, weight, crossMin, crossMax) per line, which is
// enough to reconstruct a segment, but that is a separate readback and a
// separate decision about what the segment MEANS in this view.
//
// The lattice sits in FLOOR (u, v), the same space projectedBins.ts bins the
// image into, so placing a dot is a bin lookup and no geometry -- see
// DecodeLattice. `rotationSteps` rotates the canvas to match the true-cardinal
// display toggle applied to the texture underneath it.
export function drawGridPeriodPhaseProjected(
  camera: Camera, x: number, y: number, w: number, h: number, rotationSteps = 0,
) {
  const lattice = camera.settings.showSampleLattice ? camera.pose?.intermediates.lattice : undefined;
  const bins = camera.lastProjectedBins;
  // Hidden rather than cleared-and-shown: an empty canvas over the rect would
  // still intercept nothing but would leave a stale frame visible for one tick
  // if the sizing below were ever skipped.
  if (!lattice || !bins) { hideGridPeriodPhaseProjected(); return; }

  const canvas = gridPeriodPhaseProjectedCanvas, ctx = gridPeriodPhaseProjectedCtx;
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

  // U is MIRRORED in this view (see projectedBins.ts's own bu), which is why
  // this reads `maxU - u` where v reads `v - minV`.
  const { rows, cols, uAt, vAt, packed, correct } = lattice;
  for (let i = 0; i < rows; i++) {
    const bv = (vAt[i]! - bins.minV) / bins.binWidthV;
    const py = (1 - bv / bins.h) * canvas.height;
    for (let j = 0; j < cols; j++) {
      const p = packed[i * cols + j]!;
      // Bit 0 clear means decode.build could not resolve the cell at all --
      // past the grazing cutoff, behind the camera, or off the image. No dot,
      // rather than a dot claiming a reading it never took.
      if ((p & 1) === 0) continue;
      const bu = (bins.maxU - uAt[j]!) / bins.binWidthU;
      const px = (bu / bins.w) * canvas.width;

      // A SET bit is the DARK cell -- decode.build thresholds with `<`, so the
      // fill here is the cell as printed, not as sampled brightness.
      const bit = (p >> 1) & 1;
      const verdict = correct ? correct[i * cols + j]! : -1;
      ctx.beginPath();
      ctx.arc(px, py, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = bit ? '#000' : '#fff';
      ctx.fill();
      // No ring where there is no verdict: an undecoded frame, or a
      // re-derivation that failed its own check against the device's counters.
      ctx.strokeStyle = verdict < 0 ? 'rgba(0,0,0,0.6)' : (verdict === 1 ? '#0f0' : '#f00');
      ctx.lineWidth = verdict < 0 ? 1 : 1.5;
      ctx.stroke();
    }
  }
}
