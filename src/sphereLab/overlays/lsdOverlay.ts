import { type Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import {
  growthCandidateGroup, lsdReadout, lsdRectanglesGroup, toggleLsdCompositeBtn, toggleLsdRawRegionsBtn, toggleLsdRejectedBtn,
  toggleLsdSegmentsBtn,
} from '../ui/dom.ts';

// Deterministic (seed pixel index -> hue), NOT the rectangle's own fitted
// angle -- so two distinct flood-fill blobs that happen to share a similar
// direction (e.g. two separate rows of the grid) still render as visibly
// different colors, which is exactly what these debug views are for
// (inspecting flood-fill/fragmentation behavior, not direction).
// `region.members[0]` is just some deterministic, stable-given-the-same-settings
// member pixel to hash off of -- growRegionsCCL (pose/stages/lsd/) collects a
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

// ── EVERY DRAWN VIEW IN THIS FILE IS DARK, pending pose2 plumbing ─────────
//
// What used to live between here and updateLsdAvailability, and what each one
// read:
//
//   - drawEdgeConnectivityPreview  -- the hovered pixel's growth candidates,
//     from the gradient field plus `computeEdgeNeighbors` (deleted with
//     src/pose/stages/lsd/levelLine.ts);
//   - repaintLsdRawRegionsHighlight -- every grown region painted by a hash of
//     its lowest-index member, or just the hovered one, from
//     `intermediates.regions` + `intermediates.regionId`;
//   - the rejected raster and the rectangle SVG, from `intermediates.rects`
//     joined to `intermediates.regions` by index;
//   - the readout line counting accepted vs rejected rectangles.
//
// All of them read pipeline intermediates. `Intermediates` no longer carries
// regions or rects at all -- pose2 keeps them on the device and reads back 128
// bytes of pose -- so there is nothing to draw and no data to fall back to.
//
// It matters that this is EMPTY rather than RECOMPUTED. The old note here said
// nothing recomputes when the intermediates are absent, because a fallback would
// put the duplicated LSD chain straight back on the path where it is hardest to
// notice. That reasoning survives the deletion exactly: the fix is an opt-in
// readback from pose2 (§22), not a second detector living in an overlay.
//
// The four toggle buttons stay live and stay persistent, and the groups are
// cleared rather than left holding the last capture's shapes.

export function repaintLsdRawRegionsHighlight(_camera: Camera) {
  while (growthCandidateGroup.firstChild) growthCandidateGroup.removeChild(growthCandidateGroup.firstChild);
}

export function updateLsdOverlay(_camera: Camera) {
  while (lsdRectanglesGroup.firstChild) lsdRectanglesGroup.removeChild(lsdRectanglesGroup.firstChild);
  while (growthCandidateGroup.firstChild) growthCandidateGroup.removeChild(growthCandidateGroup.firstChild);
  lsdReadout.textContent = 'no LSD data -- awaiting pose2 intermediate readback';
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
