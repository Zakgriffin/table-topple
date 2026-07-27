import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import { computeLsdRectangles, LsdRectangle } from '../pipeline/lsdSegments.ts';
import { computeThroughRect } from '../ui/layout.ts';
import {
  lsdReadout, lsdRectanglesGroup, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn, toggleLsdSegmentsBtn,
} from '../ui/dom.ts';
import { svgEl } from './svgUtil.ts';

// Deterministic (seed pixel index -> hue), NOT the rectangle's own fitted
// angle -- so two distinct flood-fill blobs that happen to share a similar
// direction (e.g. two separate rows of the grid) still render as visibly
// different colors, which is exactly what these debug views are for
// (inspecting flood-fill/fragmentation behavior, not direction). rawMembers[0]
// is always the seed pixel -- growRegions (pipeline/lsdSegments.ts) pushes
// it as a region's very first member before any growth happens.
export function hashSeedIndexToHueDeg(seedIndex: number): number {
  let x = seedIndex | 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = x ^ (x >>> 16);
  return ((x >>> 0) / 0xffffffff) * 360;
}

// Per-pixel raster overlays (same "flat Uint8Array + DataTexture + quad"
// shape as the contamination overlays, see camera/model.ts's own comment on
// why) -- both fully repaint (fill(0) first) rather than incrementally
// updating, since the set of accepted/rejected rectangles can change
// completely between calls.
function paintRawRegionsRaster(rects: readonly LsdRectangle[], out: Uint8Array) {
  out.fill(0);
  for (const r of rects) {
    if (!r.accepted) continue;
    const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(r.rawMembers[0]), 1, 1);
    for (let mi = 0; mi < r.rawMembers.length; mi++) {
      const o = r.rawMembers[mi] * 4;
      out[o] = hr; out[o + 1] = hg; out[o + 2] = hb; out[o + 3] = 220;
    }
  }
}
function paintRejectedRaster(rects: readonly LsdRectangle[], out: Uint8Array) {
  out.fill(0);
  for (const r of rects) {
    if (r.accepted) continue;
    for (let mi = 0; mi < r.rawMembers.length; mi++) {
      const o = r.rawMembers[mi] * 4;
      out[o] = 255; out[o + 1] = 40; out[o + 2] = 40; out[o + 3] = 200;
    }
  }
}

// SVG wireframe rectangles (replaces the old gradientArrowCtx canvas draw --
// a real <svg> element, not a flat bitmap, per this session's request).
// Accepted (showLsdSegments) and rejected (showLsdRejected) are drawn
// independently -- neither requires the other. A small yellow dot marks a
// rectangle that only passed NFA after at least one retry.
function drawRectanglesSvg(camera: Camera, rects: readonly LsdRectangle[]) {
  while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
  const settings = camera.settings;
  if (!settings.showLsdSegments && !settings.showLsdRejected) return;

  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const toScreen = (fx: number, fy: number) => ({
    x: rect.x + (fx + 0.5) * (rect.w / fieldW),
    y: rect.y + rect.h - (fy + 0.5) * (rect.h / fieldH),
  });

  for (const r of rects) {
    const draw = r.accepted ? settings.showLsdSegments : settings.showLsdRejected;
    if (!draw) continue;

    const ax = Math.cos(r.theta), ay = Math.sin(r.theta);
    const px = -ay, py = ax;
    const hl = r.length / 2, hw = Math.max(r.width / 2, 0.5);
    const corners = [
      [r.cx + hl * ax + hw * px, r.cy + hl * ay + hw * py],
      [r.cx + hl * ax - hw * px, r.cy + hl * ay - hw * py],
      [r.cx - hl * ax - hw * px, r.cy - hl * ay - hw * py],
      [r.cx - hl * ax + hw * px, r.cy - hl * ay + hw * py],
    ].map(([x, y]) => toScreen(x, y));
    const points = corners.map((c) => `${c.x},${c.y}`).join(' ');

    let stroke: string, dash: string, width: number;
    if (r.accepted) {
      const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(r.rawMembers[0]), 1, 1);
      stroke = `rgb(${hr},${hg},${hb})`; dash = 'none'; width = 2;
    } else {
      stroke = 'rgba(255,40,40,0.85)'; dash = '4,3'; width = 1.5;
    }
    lsdRectanglesGroup.appendChild(svgEl('polygon', {
      points, fill: 'none', stroke, 'stroke-width': width, 'stroke-dasharray': dash,
    }));

    if (r.accepted && r.retries > 0) {
      const c = toScreen(r.cx, r.cy);
      lsdRectanglesGroup.appendChild(svgEl('circle', {
        cx: c.x, cy: c.y, r: 3.5, fill: 'rgb(255,225,0)', stroke: 'rgba(0,0,0,0.85)', 'stroke-width': 1,
      }));
    }
  }
}

// Recomputes the from-scratch traditional LSD pipeline (pipeline/
// lsdSegments.ts) and repaints its 3 independent debug views (accepted
// rectangles + rejected candidates, both SVG; raw region pixels, raster).
export function updateLsdOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showLsdSegments && !settings.showLsdRejected && !settings.showLsdRawRegions) {
    while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
    return;
  }
  if (settings.fieldView !== 'gradient2x2') return;
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const field = computeGradient2x2Field(camera.lastNoisedPreviewGray, w, h);
  const rects = computeLsdRectangles(field, {
    toleranceDeg: settings.lsdToleranceDeg,
    rhoNoiseThreshold: settings.lsdRhoNoiseThreshold,
    magnitudeBuckets: settings.lsdMagnitudeBuckets,
    nfaEpsilon: settings.lsdNfaEpsilon,
    nfaTestExponent: settings.lsdNfaTestExponent,
    maxRetries: settings.lsdMaxRetries,
    retryToleranceFactor: settings.lsdRetryToleranceFactor,
    retryShrinkFraction: settings.lsdRetryShrinkFraction,
  });
  camera.lastLsdRectangles = rects;

  if (settings.showLsdRawRegions) {
    paintRawRegionsRaster(rects, camera.lsdRawRegionsData);
    camera.lsdRawRegionsTex.needsUpdate = true;
  }
  if (settings.showLsdRejected) {
    paintRejectedRaster(rects, camera.lsdRejectedData);
    camera.lsdRejectedTex.needsUpdate = true;
  }
  drawRectanglesSvg(camera, rects);

  const accepted = rects.filter((r) => r.accepted);
  const retried = rects.filter((r) => r.retries > 0);
  const totalRetries = rects.reduce((sum, r) => sum + r.retries, 0);
  lsdReadout.textContent = `${accepted.length} accepted, ${rects.length - accepted.length} rejected -- `
    + `${retried.length} needed a retry (${totalRetries} retries total)`;
}

export function updateLsdAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.fieldView === 'gradient2x2';
  // All independently clickable -- none requires another to already be on.
  toggleLsdSegmentsBtn.disabled = !relevant;
  toggleLsdRejectedBtn.disabled = !relevant;
  toggleLsdRawRegionsBtn.disabled = !relevant;
  toggleLsdCompositeBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showLsdSegments = false;
    cam.settings.showLsdRejected = false;
    cam.settings.showLsdRawRegions = false;
    cam.settings.showLsdComposite = false;
    toggleLsdSegmentsBtn.classList.remove('active');
    toggleLsdRejectedBtn.classList.remove('active');
    toggleLsdRawRegionsBtn.classList.remove('active');
    toggleLsdCompositeBtn.classList.remove('active');
    cam.lastLsdRectangles = null;
    lsdReadout.textContent = '';
    while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
  }
}
