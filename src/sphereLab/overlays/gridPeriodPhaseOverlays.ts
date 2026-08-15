import { type Camera } from '../camera/model.ts';
import { persistConfig } from '../config.ts';
import { activeCamera } from '../camera/store.ts';
import { gridPeriodPhasePlotSvg, gridPeriodPhaseProjectedCanvas, toggleDistinctnessCurveBtn, toggleGapHistogramBtn, toggleProductCurveBtn, toggleValueHistogramBtn } from '../ui/dom.ts';
import { svgEl, svgText } from './svgUtil.ts';

// ── Grid period/phase debug visualizations — EMPTY, pending pose2 plumbing ──
//
// This drew the period search's work: the [Pmin, Pmax] bracket, the integer-
// count resultant curve, each image-tested candidate with its distinctness, the
// pooled all-pairs gap histogram, the rectified line values, and the sample
// lattice on the Projected-Cam rect. Roughly 480 lines of it.
//
// ALL OF THAT DATA CAME FROM `camera.pose.gridPeriodPhase`, which the deleted
// `src/pose` produced and nothing produces now. `src/pose2` recovers the same
// period and phase, but on the device and without the search's working set:
// there are no rowLines/colLines, no coarse sample curve and no candidate list
// on the host, because none of it crosses the bus. Reproducing this plot means
// giving pose2 an opt-in intermediate readback first (§22 of
// full_system_breakdown.md), which does not exist yet.
//
// So the plot is an EMPTY STATE rather than a deletion. The panel, its four
// toggle buttons and their persistence all still work -- the UI shape is intact
// and says why it is blank -- and when the readback lands this file is where the
// drawing goes back.

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

// Drew the rectified lines and the sample lattice onto the Projected-Cam rect.
// Both came from the same deleted result, so there is nothing to draw and the
// canvas stays hidden rather than showing a stale frame.
export function drawGridPeriodPhaseProjected(
  _camera: Camera, _x: number, _y: number, _w: number, _h: number, _rotationSteps = 0,
) {
  hideGridPeriodPhaseProjected();
}
