import { type Camera, type RegionCsr } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import {
  growthCandidateGroup, lsdReadout, lsdRectanglesGroup, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn,
  toggleLsdSegmentsBtn,
} from '../ui/dom.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { computeThroughRect } from '../ui/layout.ts';
import { svgEl } from './svgUtil.ts';

// Deterministic (seed pixel index -> hue), NOT the rectangle's own fitted
// angle -- so two distinct flood-fill blobs that happen to share a similar
// direction (e.g. two separate rows of the grid) still render as visibly
// different colors, which is exactly what these debug views are for
// (inspecting flood-fill/fragmentation behavior, not direction).
//
// THE KEY IS THE REGION INDEX. It used to be the region's lowest-index member
// pixel, which was never meaningful either -- just something deterministic and
// stable to hash. What the key must actually satisfy is that every view of the
// same region agrees, so a segment, its fitted outline and its member pixels all
// come out one colour; the region index does that and costs no extra buffer.
// The name is kept for its callers rather than for accuracy about what it hashes.
function hashSeedIndexToHueDeg(seedIndex: number): number {
  let x = seedIndex | 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = x ^ (x >>> 16);
  return ((x >>> 0) / 0xffffffff) * 360;
}

// ── THREE VIEWS ARE BACK; ONE IS STILL DARK ──────────────────────────────
//
// BACK: the fitted-rectangle outlines and the accept/reject readout off
// `pose.rects`, and both per-pixel rasters off `pose.regions`
// -- the region CSR (see pipeline/axesReconstruction.ts's inspectFor, which
// requests each only when its own toggle is on).
//
// STILL DARK: drawEdgeConnectivityPreview, which drew the hovered pixel's growth
// candidates. Re-deriving it means a second copy of GROW_WGSL's accept predicate
// on this side, and signed-vs-unsigned dot is precisely the mutation no test in
// this project caught -- so a host copy carrying the same error would show the
// bug and the bug agreeing. Deliberately not restored without a decision.
//
// WHAT DOES NOT HAPPEN WHEN THE DATA IS ABSENT: a recomputation. That has been
// this file's rule throughout and it survives every rewrite -- a fallback would
// put a second detector in an overlay, on the path where it is hardest to notice.
// Absent means blank, and blank says why.

/** 10 f32 per region -- see BUFFERS.rects. */
const RECT_STRIDE = 10;

// The colour every view of region `r` agrees on. One function rather than a
// convention, because the agreement IS the property these views exist for: a
// segment, its fitted outline and its member pixels being one colour is what
// makes "this blob fragmented into three" readable at a glance. Exported for
// drawCompositeLines, which is the fourth view of the same region.
export function regionRgb(r: number): [number, number, number] {
  return hsvToRgb(hashSeedIndexToHueDeg(r), 0.85, 1);
}

// Member indices are the PIPELINE's, i.e. top-down; `out` is a flipY=false
// preview buffer and is bottom-up. One place for that reversal, so the raster
// painters below cannot disagree about it. Same rule as everywhere else in this
// codebase: the flip goes on the way OUT to display.
function paintMember(
  out: Uint8Array, member: number, w: number, h: number,
  r: number, g: number, b: number, a: number,
) {
  const y = (member / w) | 0, x = member % w;
  const o = ((h - 1 - y) * w + x) * 4;
  out[o] = r; out[o + 1] = g; out[o + 2] = b; out[o + 3] = a;
}

function paintRegion(csr: RegionCsr, r: number, out: Uint8Array, w: number, h: number,
  rr: number, gg: number, bb: number, a: number) {
  const off = csr.offsets[r]!, size = csr.sizes[r]!;
  for (let m = 0; m < size; m++) paintMember(out, csr.members[off + m]!, w, h, rr, gg, bb, a);
}

// ── HOVER TO REGION, which is the one thing `regionId` bought ────────────
//
// The pipeline computes no per-pixel region id -- the CSR carries the same
// information the other way round, and re-adding a per-pixel buffer to the hot
// path to save a host loop is the wrong trade. So this inverts it here.
//
// Built ONCE per capture and cached against the `members` array itself: this is
// called on every pointermove, and inverting ~40k members per move to read one
// entry would be the same mistake the hover arrows used to make when they
// rebuilt the whole gradient field to sample a single pixel. A WeakMap keyed on
// the array means a new capture's CSR is a cache miss by construction and the
// old map is collectable -- there is no invalidation to get wrong.
const inverseMaps = new WeakMap<Uint32Array, Int32Array>();
function regionOfPixel(csr: RegionCsr, regionCount: number, n: number, pixel: number): number {
  let map = inverseMaps.get(csr.members);
  if (!map) {
    map = new Int32Array(n).fill(-1);
    for (let r = 0; r < regionCount; r++) {
      const off = csr.offsets[r]!, size = csr.sizes[r]!;
      for (let m = 0; m < size; m++) map[csr.members[off + m]!] = r;
    }
    inverseMaps.set(csr.members, map);
  }
  return map[pixel] ?? -1;
}

// What the two rasters share: the toggle is on, so SOMETHING must be written
// this frame -- a stale raster from the previous capture is worse than a blank
// one, because scene/throughCam2D.ts paints these buffers every frame in
// Through-Cam regardless of whether anything refreshed them.
function rasterInputs(camera: Camera): { csr: RegionCsr; regionCount: number } | null {
  const csr = camera.pose?.regions, regionCount = camera.pose?.regionCount;
  if (!csr || regionCount === undefined) return null;
  return { csr, regionCount };
}

// The grown regions' member pixels, hued per region.
//
// Paints the TRUE collect output -- every region that survived `minRegionSize`,
// including ones no rectangle was ever accepted for. That is the right lens for
// inspecting the GROWER: filtering through the fit would hide exactly the
// fragmentation these views are opened to look at.
//
// Isolates the ONE region under the cursor when there is one, and shows all of
// them otherwise (cursor off the field, or on a pixel below the edge floor that
// never joined a region).
export function repaintLsdRawRegionsHighlight(camera: Camera) {
  while (growthCandidateGroup.firstChild) growthCandidateGroup.removeChild(growthCandidateGroup.firstChild);
  if (!camera.settings.showLsdRawRegions) return;

  const out = camera.lsdRawRegionsData;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  out.fill(0);
  const inputs = rasterInputs(camera);
  if (inputs) {
    const { csr, regionCount } = inputs;
    const hover = camera.lastHoverFieldIndex;
    const hovered = hover === null ? -1 : regionOfPixel(csr, regionCount, w * h, hover);
    if (hovered >= 0) {
      const [rr, gg, bb] = regionRgb(hovered);
      paintRegion(csr, hovered, out, w, h, rr, gg, bb, 220);
    } else {
      for (let r = 0; r < regionCount; r++) {
        const [rr, gg, bb] = regionRgb(r);
        paintRegion(csr, r, out, w, h, rr, gg, bb, 220);
      }
    }
  }
  camera.lsdRawRegionsTex.needsUpdate = true;
}

// The member pixels of the regions whose rectangle the NFA test REJECTED. Flat
// red rather than the region hue: the question here is "what did the detector
// throw away", one set, not which blob was which.
//
// Joined by INDEX -- `rects[i]` is the fit of region `i`, on the device as much
// as here.
function repaintLsdRejectedRaster(camera: Camera) {
  if (!camera.settings.showLsdRejected) return;
  const out = camera.lsdRejectedData;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  out.fill(0);
  const inputs = rasterInputs(camera);
  const rects = camera.pose?.rects;
  if (inputs && rects) {
    const { csr, regionCount } = inputs;
    for (let r = 0; r < regionCount; r++) {
      if (rects[r * RECT_STRIDE + 6]! !== 0) continue; // accepted -- the outline view's job
      paintRegion(csr, r, out, w, h, 255, 40, 40, 200);
    }
  }
  camera.lsdRejectedTex.needsUpdate = true;
}

// Repaints all four LSD debug views from the pose run's own output: the two
// rasters (raw regions, rejected regions) and the two SVG outline sets
// (accepted, rejected). Each is gated on its own toggle and none requires
// another to be on.
//
// The rasters go first and unconditionally-if-toggled, including when the data
// is missing: repaintLsdRawRegionsHighlight clears before it checks, so a frame
// that was not asked for the CSR blanks the raster rather than leaving the last
// capture's pixels on screen under a live toggle.
export function updateLsdOverlay(camera: Camera) {
  while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);

  // Respects whatever camera.lastHoverFieldIndex already is, rather than forcing
  // a fresh "every region" paint over an isolated one. It owns growthCandidateGroup
  // (it is the hover-driven view), so clearing that here as well would be a second
  // claim on the same group.
  repaintLsdRawRegionsHighlight(camera);
  repaintLsdRejectedRaster(camera);

  // `regionCount` rides the pose block, so it is here on every local run whether
  // or not anything was requested -- which is why "how many regions" and "how
  // many rectangles were accepted" are now two independent questions rather than
  // one nested check on a `counts` readback that no longer happens.
  const rects = camera.pose?.rects;
  const regionCount = camera.pose?.regionCount;
  if (regionCount === undefined) {
    lsdReadout.textContent = 'no local detector run';
    return;
  }
  if (!rects) {
    lsdReadout.textContent = camera.settings.showLsdSegments || camera.settings.showLsdRejected
      ? `${regionCount} regions -- the last run was not asked for rectangles`
      : `${regionCount} regions (segment overlays off)`;
    return;
  }

  const rect = computeThroughRect(camera);
  const fieldW = camera.rtSize.w, fieldH = camera.rtSize.h;
  // TOP-DOWN in, screen out -- the same composition drawCompositeLines uses, and
  // for the same reason: these coordinates are the pipeline's own.
  const toScreen = (px: number, py: number) => ({
    x: rect.x + (px + 0.5) * (rect.w / fieldW),
    y: rect.y + rect.h - ((fieldH - 1 - py) + 0.5) * (rect.h / fieldH),
  });

  let accepted = 0, rejected = 0;
  for (let r = 0; r < regionCount; r++) {
    const o = r * RECT_STRIDE;
    const cx = rects[o]!, cy = rects[o + 1]!, theta = rects[o + 2]!;
    const halfL = rects[o + 3]! * 0.5, halfW = Math.max(rects[o + 4]!, 1) * 0.5;
    const isAccepted = rects[o + 6]! !== 0;
    if (isAccepted) accepted++; else rejected++;
    // Counted before it is gated: the readout describes what the DETECTOR did,
    // not what is currently drawn, so hiding rejects must not change the tally.
    if (isAccepted ? !camera.settings.showLsdSegments : !camera.settings.showLsdRejected) continue;

    // The fitted rectangle's four corners: half-length along its own axis, half
    // width across it. Drawn as the OUTLINE rather than as a centre-line, because
    // the width is what says whether the fit is a crisp edge or a smear.
    const ax = Math.cos(theta), ay = Math.sin(theta);
    const corners = [
      toScreen(cx + halfL * ax - halfW * ay, cy + halfL * ay + halfW * ax),
      toScreen(cx + halfL * ax + halfW * ay, cy + halfL * ay - halfW * ax),
      toScreen(cx - halfL * ax + halfW * ay, cy - halfL * ay - halfW * ax),
      toScreen(cx - halfL * ax - halfW * ay, cy - halfL * ay + halfW * ax),
    ];
    // Rejected rectangles are flat red, accepted ones take the region's own
    // colour -- through regionRgb, the same function the raw-regions raster
    // paints its member pixels with and the same key drawCompositeLines hashes,
    // so a segment, its outline and its pixels are one colour.
    const stroke = isAccepted ? `rgb(${regionRgb(r).join(',')})` : 'rgba(235,60,60,0.9)';
    lsdRectanglesGroup.appendChild(svgEl('polygon', {
      points: corners.map((c) => `${c.x},${c.y}`).join(' '),
      fill: 'none', stroke, 'stroke-width': isAccepted ? 1.5 : 1,
    }));
  }

  lsdReadout.textContent = `${regionCount} regions — ${accepted} accepted, ${rejected} rejected`;
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
