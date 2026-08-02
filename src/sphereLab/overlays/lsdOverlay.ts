import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import {
  computeEdgeNeighbors, computeLsdRectanglesAuto, computeMagTheta, growRegionsCCL, GrownRegion, LsdRectangle,
} from '../pipeline/lsdSegments.ts';
import { computeThroughRect } from '../ui/layout.ts';
import {
  growthCandidateGroup, lsdReadout, lsdRectanglesGroup, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn,
  toggleLsdSegmentsBtn,
} from '../ui/dom.ts';
import { svgEl } from './svgUtil.ts';

// Deterministic (seed pixel index -> hue), NOT the rectangle's own fitted
// angle -- so two distinct flood-fill blobs that happen to share a similar
// direction (e.g. two separate rows of the grid) still render as visibly
// different colors, which is exactly what these debug views are for
// (inspecting flood-fill/fragmentation behavior, not direction). rawMembers[0]
// is just some deterministic, stable-given-the-same-settings member pixel to
// hash off of -- growRegionsCCL (pipeline/lsdSegments.ts) collects a
// region's members in increasing pixel-index order, so this is actually its
// LOWEST-index member, not a meaningful "seed" (dense JFA seeding has no
// single privileged seed pixel the way the old serial BFS's seed-first
// growth order did).
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
// updating, since the set of accepted/rejected/grown regions can change
// completely between calls.
//
// Paints the TRUE growRegionsCCL state directly (every region, including
// degenerate 1-member ones) rather than filtering through fitRegionWithRetries
// the way the old (pre-fit-rectangle) accepted-only view did -- a region
// with fewer than 2 members never gets a rect at all (see that function's
// own comment) and so never became an LsdRectangle, accepted OR rejected,
// making it invisible to every debug view. That's the wrong lens for
// inspecting the growing algorithm ITSELF (e.g. watching "grow steps" scrub
// from 0 upward) -- at low round counts, dense seeding means MOST regions
// are still tiny/degenerate, and silently hiding all of them showed a sparse,
// biased sample of "whatever happened to grow fastest" instead of the real
// per-pixel label state. See this session's chat.
function paintRawGrownRegions(regions: readonly GrownRegion[], out: Uint8Array) {
  out.fill(0);
  for (const region of regions) {
    const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(region.members[0]), 1, 1);
    for (let mi = 0; mi < region.members.length; mi++) {
      const o = region.members[mi] * 4;
      out[o] = hr; out[o + 1] = hg; out[o + 2] = hb; out[o + 3] = 220;
    }
  }
}

// Edge-connectivity preview: for the pixel under the cursor, draws a short
// stub toward each of its 8 neighbors that it shares a GRAPH EDGE with --
// i.e. exactly the neighbors growRegionsCCL's hook pass would consider
// merging with. Calls computeEdgeNeighbors (pipeline/lsdSegments.ts -- the
// SAME predicate the real algorithm's inner loop uses, not an independently
// reimplemented copy that could drift out of sync).
//
// This replaces the JFA grower's growth-candidate arrows ("where would this
// pixel LOOK next", up to hundreds of pixels away, which needed viewport
// clipping to stay on screen). There is nothing left to clip: every candidate
// is one pixel away now, which is the whole point of the redesign. The
// question this view answers changed with it -- from "where is this pixel
// searching" to "why did/didn't this pixel connect to the ridge next to it",
// which is the actual question when a segment fragments unexpectedly.
//
// Connected neighbors draw white; neighbors that exist and are above rho-low
// but FAILED the angle test draw dim red, since "the predicate rejected this"
// and "there was nothing there at all" are the two explanations worth
// telling apart when a component ends where you didn't expect.
function drawEdgeConnectivityPreview(camera: Camera) {
  while (growthCandidateGroup.firstChild) growthCandidateGroup.removeChild(growthCandidateGroup.firstChild);
  if (!camera.settings.showLsdRawRegions) return;
  const cached = camera.lastLsdGrownRegions;
  const hoverIndex = camera.lastHoverFieldIndex;
  if (!cached || hoverIndex === null) return;
  const { mag, theta } = cached;
  const settings = camera.settings;
  if (mag[hoverIndex] <= settings.lsdRhoNoiseThreshold) return; // below the edge floor -- never participates, so no edges to show

  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const connected = new Set(computeEdgeNeighbors(
    mag, theta, fieldW, fieldH, hoverIndex, settings.lsdToleranceDeg, settings.lsdRhoNoiseThreshold,
  ));

  const rect = computeThroughRect(camera);
  const fieldCol = hoverIndex % fieldW, fieldRow = (hoverIndex / fieldW) | 0;
  const cellW = rect.w / fieldW, cellH = rect.h / fieldH;
  const toScreen = (fx: number, fy: number) => ({
    x: rect.x + (fx + 0.5) * cellW,
    y: rect.y + rect.h - (fy + 0.5) * cellH,
  });
  const origin = toScreen(fieldCol, fieldRow);

  for (let dy = -1; dy <= 1; dy++) {
    for (let dx = -1; dx <= 1; dx++) {
      if (dx === 0 && dy === 0) continue;
      const nx = fieldCol + dx, ny = fieldRow + dy;
      if (nx < 0 || nx >= fieldW || ny < 0 || ny >= fieldH) continue;
      const j = ny * fieldW + nx;
      const isConnected = connected.has(j);
      // A neighbor below the edge floor gets nothing drawn at all -- it isn't
      // a rejected candidate, it's not a node in the graph in the first place.
      if (!isConnected && mag[j] <= settings.lsdRhoNoiseThreshold) continue;
      const target = toScreen(nx, ny);
      growthCandidateGroup.appendChild(svgEl('line', {
        x1: String(origin.x), y1: String(origin.y),
        x2: String(target.x), y2: String(target.y),
        stroke: isConnected ? 'rgb(255,255,255)' : 'rgba(200,60,60,0.55)',
        'stroke-width': isConnected ? '2' : '1',
      }));
    }
  }
}


// Repaints the raw-regions raster to isolate just the ONE region under the
// cursor (camera.lastHoverFieldIndex, tracked unconditionally by overlays/
// hoverDebugOverlays.ts's updateHoverOverlays) instead of showing every
// region at once -- reads the cached growRegionsCCL output updateLsdOverlay
// already left on the camera rather than recomputing anything, so this is
// cheap enough to call on every pointermove (a full out.fill(0) plus
// iterating just ONE region's member list, not every region's). Falls back
// to the normal all-regions view when the cursor isn't over the field, or
// is sitting on a sub-rho pixel that never got a region (regionId -1) --
// see this session's chat. Also drives the growth-candidate-preview arrows
// above -- same hover event, same cached region data.
export function repaintLsdRawRegionsHighlight(camera: Camera) {
  drawEdgeConnectivityPreview(camera);
  if (!camera.settings.showLsdRawRegions) return;
  const cached = camera.lastLsdGrownRegions;
  if (!cached) return;
  const { regionId, regions } = cached;

  const hoverIndex = camera.lastHoverFieldIndex;
  const hoveredRegion = hoverIndex !== null ? regionId[hoverIndex] : -1;

  if (hoveredRegion === -1) {
    paintRawGrownRegions(regions, camera.lsdRawRegionsData);
  } else {
    const out = camera.lsdRawRegionsData;
    out.fill(0);
    const region = regions[hoveredRegion];
    const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(region.members[0]), 1, 1);
    for (let mi = 0; mi < region.members.length; mi++) {
      const o = region.members[mi] * 4;
      out[o] = hr; out[o + 1] = hg; out[o + 2] = hb; out[o + 3] = 220;
    }
  }
  camera.lsdRawRegionsTex.needsUpdate = true;
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
//
// Async now that computeLsdRectanglesAuto can go through a GPU round trip
// (see lsdSegments.ts/lsdFit.ts) -- every caller here fires this off without
// awaiting it (a live "redraw when ready" refresh, same pattern
// runAxesReconstruction's own RAF callback already uses), which on its own
// would let a slow call's stale result clobber a faster, NEWER call's result
// if two land out of start order (e.g. a fast slider drag). lsdOverlaySeq
// guards against that: only the most-recently-STARTED call is allowed to
// actually apply its result.
let lsdOverlaySeq = 0;
export async function updateLsdOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showLsdSegments && !settings.showLsdRejected && !settings.showLsdRawRegions) {
    while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
    return;
  }
  // Not gated on fieldView -- this overlay computes its own field from
  // lastNoisedPreviewGray just below, so gradient2x2 was never a real
  // dependency, only a cosmetic assumption about what you'd want it drawn
  // over. See pipeline/preview.ts's overlaysNeedGray, which is what actually
  // guarantees that gray is fresh on every view now.
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const field = computeGradient2x2Field(camera.lastNoisedPreviewGray, w, h);

  // Pure CPU and synchronous (unlike the fit/NFA pass below, which can go
  // through a GPU round trip) -- painted immediately, before the await, so
  // this view never waits on a GPU dispatch and can never land out of order
  // the way that result can (see lsdOverlaySeq's own comment): plain
  // synchronous JS code always runs in call order.
  if (settings.showLsdRawRegions) {
    const { mag, theta } = computeMagTheta(field);
    const { regionId, regions, roundsRun, converged } = growRegionsCCL(
      mag, theta, w, h, settings.lsdToleranceDeg, settings.lsdRhoNoiseThreshold, settings.lsdRhoHighThreshold,
      settings.lsdCclSteps, settings.lsdMinRegionSize,
    );
    camera.lastLsdGrownRegions = { regionId, regions, mag, theta, roundsRun, converged };
    repaintLsdRawRegionsHighlight(camera); // respects whatever camera.lastHoverFieldIndex already is, not just a fresh "every region" paint
  }

  const seq = ++lsdOverlaySeq;
  const rects = await computeLsdRectanglesAuto(field, {
    toleranceDeg: settings.lsdToleranceDeg,
    rhoNoiseThreshold: settings.lsdRhoNoiseThreshold,
    rhoHighThreshold: settings.lsdRhoHighThreshold,
    cclSteps: settings.lsdCclSteps,
    minRegionSize: settings.lsdMinRegionSize,
    nfaEpsilon: settings.lsdNfaEpsilon,
    nfaTestExponent: settings.lsdNfaTestExponent,
    maxRetries: settings.lsdMaxRetries,
    retryToleranceFactor: settings.lsdRetryToleranceFactor,
    retryShrinkFraction: settings.lsdRetryShrinkFraction,
  });
  if (seq !== lsdOverlaySeq) return; // a newer call started while this one was in flight -- its result wins instead
  camera.lastLsdRectangles = rects;

  if (settings.showLsdRejected) {
    paintRejectedRaster(rects, camera.lsdRejectedData);
    camera.lsdRejectedTex.needsUpdate = true;
  }
  drawRectanglesSvg(camera, rects);

  const accepted = rects.filter((r) => r.accepted);
  const retried = rects.filter((r) => r.retries > 0);
  const totalRetries = rects.reduce((sum, r) => sum + r.retries, 0);
  // The growth half of the readout is only populated when the raw-regions
  // view is on, since that's the only branch above that runs growRegionsCCL
  // separately (the rectangle path runs its own copy internally and doesn't
  // report rounds back). "capped" is the case worth seeing: it means the
  // CCL-steps scrubber stopped the loop short of the fixpoint, so the regions
  // on screen are mid-growth and NOT what production would produce.
  const grown = camera.lastLsdGrownRegions;
  const growthPart = settings.showLsdRawRegions && grown
    ? `${grown.regions.length} regions in ${grown.roundsRun} round${grown.roundsRun === 1 ? '' : 's'}`
      + `${grown.converged ? ' (converged)' : ' (CAPPED -- mid-growth, not the real result)'} -- `
    : '';
  lsdReadout.textContent = growthPart
    + `${accepted.length} accepted, ${rects.length - accepted.length} rejected -- `
    + `${retried.length} needed a retry (${totalRetries} retries total)`;
}

// No-op hook -- see contaminationOverlays.ts's updateContaminationAvailability
// for why this stays a named function now that these overlays render over
// every field view and there's nothing left to gate on. It used to force all
// four toggles OFF whenever the field view wasn't gradient2x2, which meant
// switching views silently discarded the user's own overlay selection.
export function updateLsdAvailability() {
  const cam = activeCamera(); if (!cam) return;
  // All independently clickable -- none requires another to already be on.
  toggleLsdSegmentsBtn.disabled = false;
  toggleLsdRejectedBtn.disabled = false;
  toggleLsdRawRegionsBtn.disabled = false;
  toggleLsdCompositeBtn.disabled = false;
}
