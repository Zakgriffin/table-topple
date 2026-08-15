import { type Camera } from '../camera/model.ts';
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
export function hashSeedIndexToHueDeg(seedIndex: number): number {
  let x = seedIndex | 0;
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b);
  x = x ^ (x >>> 16);
  return ((x >>> 0) / 0xffffffff) * 360;
}

// ── ONE OF THE THREE VIEWS IS BACK; TWO ARE STILL DARK ───────────────────
//
// BACK: the fitted-rectangle outlines and the accept/reject readout, off
// `intermediates.rects` (see pipeline/axesReconstruction.ts's inspectFor -- it is
// requested only when showLsdSegments is on).
//
// STILL DARK, and each for its own reason:
//
//   - the RAW-REGIONS and REJECTED rasters paint a pixel per MEMBER PIXEL of a
//     region, so they need the region CSR (`members`/`regionOffsets`/
//     `regionSizes`, 1.17 MiB) rather than the 0.6 MiB of rectangles. Their own
//     step;
//   - drawEdgeConnectivityPreview drew the hovered pixel's growth candidates.
//     Re-deriving it means a second copy of GROW_WGSL's accept predicate on this
//     side, and signed-vs-unsigned dot is precisely the mutation no test in this
//     project caught -- so a host copy with the same error would show the bug and
//     the bug agreeing. Deliberately not restored without a decision.
//
// WHAT DOES NOT HAPPEN WHEN THE DATA IS ABSENT: a recomputation. That has been
// this file's rule throughout and it survives every rewrite -- a fallback would
// put a second detector in an overlay, on the path where it is hardest to notice.
// Absent means blank, and blank says why.

export function repaintLsdRawRegionsHighlight(_camera: Camera) {
  while (growthCandidateGroup.firstChild) growthCandidateGroup.removeChild(growthCandidateGroup.firstChild);
}

/** 10 f32 per region -- see BUFFERS.rects. */
const RECT_STRIDE = 10;

export function updateLsdOverlay(camera: Camera) {
  while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
  while (growthCandidateGroup.firstChild) growthCandidateGroup.removeChild(growthCandidateGroup.firstChild);

  const rects = camera.pose?.intermediates.rects;
  const regionCount = camera.pose?.intermediates.regionCount;
  const wanted = camera.settings.showLsdSegments || camera.settings.showLsdRejected;
  if (!rects || regionCount === undefined) {
    lsdReadout.textContent = wanted
      ? 'no LSD data -- capture, or the last run was not asked for rectangles'
      : 'segment overlays off';
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
    // Rejected rectangles are flat red, accepted ones take the region's own hue
    // -- the SAME key drawCompositeLines hashes, so a segment and its outline
    // come out the same colour.
    const stroke = isAccepted ? hueColor(hashSeedIndexToHueDeg(r)) : 'rgba(235,60,60,0.9)';
    lsdRectanglesGroup.appendChild(svgEl('polygon', {
      points: corners.map((c) => `${c.x},${c.y}`).join(' '),
      fill: 'none', stroke, 'stroke-width': isAccepted ? 1.5 : 1,
    }));
  }

  lsdReadout.textContent = `${regionCount} regions — ${accepted} accepted, ${rejected} rejected`;
}

function hueColor(hueDeg: number): string {
  const [r, g, b] = hsvToRgb(hueDeg, 0.85, 1);
  return `rgb(${r},${g},${b})`;
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
