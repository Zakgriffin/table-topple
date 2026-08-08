import { type Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { pipelineField } from './pipelineField.ts';
import { computeEdgeNeighbors } from '../../pose/stages/lsd/levelLine.ts';
import type { GrownRegion, LsdRectangle } from '../../pose/stages/lsd/types.ts';
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
// hash off of -- growRegionsCCL (pose/stages/lsd/) collects a
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
function paintRawGrownRegions(regions: readonly GrownRegion[], out: Uint8Array, w: number, h: number) {
  out.fill(0);
  for (const region of regions) {
    const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(region.members[0]), 1, 1);
    for (let mi = 0; mi < region.members.length; mi++) {
      paintMember(out, region.members[mi], w, h, hr, hg, hb, 220);
    }
  }
}

// Member indices are the PIPELINE's, i.e. top-down; `out` is a flipY=false
// preview texture and is bottom-up. One place for that reversal so the three
// member-rasterizing loops in this file cannot disagree about it. Same rule as
// everywhere else: the flip goes on the way OUT to display.
function paintMember(
  out: Uint8Array, member: number, w: number, h: number,
  r: number, g: number, b: number, a: number,
) {
  const y = (member / w) | 0, x = member % w;
  const o = ((h - 1 - y) * w + x) * 4;
  out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = a;
}

// Edge-connectivity preview: for the pixel under the cursor, draws a short
// stub toward each of its 8 neighbors that it shares a GRAPH EDGE with --
// i.e. exactly the neighbors growRegionsCCL's hook pass would consider
// merging with. Calls computeEdgeNeighbors (pose/stages/lsd/ -- the
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
  const field = pipelineField(camera);
  const hoverIndex = camera.lastHoverFieldIndex;
  if (!field || hoverIndex === null) return;
  const { fx, fy } = field;
  const settings = camera.settings;
  // Same eligibility test the grower applies, just spelled with the gradient
  // directly now that there is no magnitude array to look it up in.
  if (Math.hypot(fx[hoverIndex], fy[hoverIndex]) <= settings.lsdRhoNoiseThreshold) return; // below the edge floor -- never participates, so no edges to show

  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  const connected = new Set(computeEdgeNeighbors(
    fx, fy, fieldW, fieldH, hoverIndex, settings.lsdToleranceDeg, settings.lsdRhoNoiseThreshold,
  ));

  const rect = computeThroughRect(camera);
  const fieldCol = hoverIndex % fieldW, fieldRow = (hoverIndex / fieldW) | 0;
  const cellW = rect.w / fieldW, cellH = rect.h / fieldH;
  // col/row, not fx/fy -- these are FIELD COORDINATES, and naming them fx/fy
  // would shadow the gradient arrays this function now reads.
  // Top-down rows (the pipeline's, and camera.lastHoverFieldIndex's) map
  // straight down the screen -- see drawRectanglesSvg's own toScreen.
  const toScreen = (col: number, row: number) => ({
    x: rect.x + (col + 0.5) * cellW,
    y: rect.y + (row + 0.5) * cellH,
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
      if (!isConnected && Math.hypot(fx[j], fy[j]) <= settings.lsdRhoNoiseThreshold) continue;
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
  const regions = camera.pose?.intermediates.regions;
  const regionId = camera.pose?.intermediates.regionId;
  if (!regions || !regionId) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;

  const hoverIndex = camera.lastHoverFieldIndex;
  const hoveredRegion = hoverIndex !== null ? regionId[hoverIndex] : -1;

  if (hoveredRegion === -1) {
    paintRawGrownRegions(regions, camera.lsdRawRegionsData, w, h);
  } else {
    const out = camera.lsdRawRegionsData;
    out.fill(0);
    const region = regions[hoveredRegion];
    const [hr, hg, hb] = hsvToRgb(hashSeedIndexToHueDeg(region.members[0]), 1, 1);
    for (let mi = 0; mi < region.members.length; mi++) {
      paintMember(out, region.members[mi], w, h, hr, hg, hb, 220);
    }
  }
  camera.lsdRawRegionsTex.needsUpdate = true;
}
function paintRejectedRaster(rects: readonly LsdRectangle[], out: Uint8Array, w: number, h: number) {
  out.fill(0);
  for (const r of rects) {
    if (r.accepted) continue;
    for (let mi = 0; mi < r.rawMembers.length; mi++) {
      paintMember(out, r.rawMembers[mi], w, h, 255, 40, 40, 200);
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
  // Rectangle centres are in the pipeline's TOP-DOWN field coordinates, so the
  // row maps straight down the screen -- the `rect.h -` this replaces was the
  // bottom-up flip the old, separately-computed field needed.
  const toScreen = (col: number, row: number) => ({
    x: rect.x + (col + 0.5) * (rect.w / fieldW),
    y: rect.y + (row + 0.5) * (rect.h / fieldH),
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
  }
}

// Repaints the three independent LSD debug views -- accepted rectangles and
// rejected candidates (SVG), raw region pixels (raster) -- from the POSE RUN's
// own stage 3b/4 output.
//
// It used to run a SECOND COMPLETE LSD CHAIN here. It rebuilt the gradient
// field, ran growRegionsCCL for the raw-regions view, and then called
// computeLsdRectanglesFromField -- grow, collect and fit all over again, on the
// row-flipped display gray, under a backend it read off globalState. The
// rectangles on screen were therefore never the rectangles the pose was
// computed from; they were a mirrored near-duplicate that happened to look the
// same. All of that is gone: axesReconstruction.ts asks for
// 'regions'/'regionId'/'rects' when one of these toggles is on, and this
// function draws what came back.
//
// It is synchronous again as a result. The async/lsdOverlaySeq machinery
// existed because computeLsdRectanglesFromField could go through a GPU round
// trip, so a slow call could land after a newer, faster one and clobber it.
// There is no call to race any more -- the data is already on the camera.
//
// Nothing here recomputes when the intermediates are absent, for the reason
// spelled out in overlays/pipelineField.ts: a fallback would put the duplicated
// chain straight back, on the path where it is hardest to notice.
export function updateLsdOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showLsdSegments && !settings.showLsdRejected && !settings.showLsdRawRegions) {
    while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
    return;
  }
  const rects = camera.pose?.intermediates.rects;
  if (!rects) return;

  if (settings.showLsdRawRegions) {
    // Respects whatever camera.lastHoverFieldIndex already is, rather than
    // forcing a fresh "every region" paint.
    repaintLsdRawRegionsHighlight(camera);
  }
  if (settings.showLsdRejected) {
    paintRejectedRaster(rects, camera.lsdRejectedData, camera.rtSize.w, camera.rtSize.h);
    camera.lsdRejectedTex.needsUpdate = true;
  }
  drawRectanglesSvg(camera, rects);

  const accepted = rects.filter((r) => r.accepted);
  // The growth half of the readout no longer reports a ROUND COUNT. roundsRun
  // and converged are growRegionsCCL return values that the chain discards, so
  // reporting them would mean publishing a new intermediate for a debug string.
  // The part that MATTERED is still here and comes straight off the setting:
  // lsdCclSteps != 0 means the scrubber stopped the loop short of the fixpoint,
  // so the regions on screen are mid-growth and NOT what production produces.
  const regions = camera.pose?.intermediates.regions;
  const growthPart = settings.showLsdRawRegions && regions
    ? `${regions.length} regions`
      + `${settings.lsdCclSteps === 0 ? '' : ` (CAPPED at ${settings.lsdCclSteps} steps -- mid-growth, not the real result)`} -- `
    : '';
  lsdReadout.textContent = growthPart
    + `${accepted.length} accepted, ${rects.length - accepted.length} rejected`;
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
