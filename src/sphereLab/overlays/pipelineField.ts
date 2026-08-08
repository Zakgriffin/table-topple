import type { Camera } from '../camera/model.ts';
import type { GradientField } from '../../pose/results.ts';

// ── Reading the POSE RUN's own gradient field from display code ───────────
//
// Three overlays used to call computeGradient2x2Field themselves, on
// lastNoisedPreviewGray, once each -- three independent full-image
// recomputations of a stage the pipeline had just finished, with no sharing
// between them. hoverDebugOverlays did it on every pointermove, rebuilding
// ~307k pixels of gradient to read ONE of them.
//
// They did that because they had no choice: the residency was destroyed in
// computePoseFromCapture's finally, so by the time display ran, fx/fy did not
// exist. Now display asks for them (see pose/intermediates.ts) and this is
// where it collects the answer.
//
// ── TWO CONVENTIONS, and this module is the seam ──────────────────────────
//
// The pipeline's field is TOP-DOWN. Every consumer here paints into a
// bottom-up buffer, because the preview textures are flipY=false (see
// axesReconstruction.ts: the one remaining flip in the whole pipeline is
// applied on the way OUT to lastNoisedPreviewGray). So nothing may index the
// pipeline's field with a display row, and nothing may take its fy at face
// value as a screen-space dy.
//
// Both conversions live here, and both are needed:
//   displayRowIndex  -- a display (row, col) to a top-down index
//   flipDy           -- the sign flip on the vertical component
//
// ── This is NOT the same array the overlays used to compute ───────────────
//
// It is close, and it is the more correct one, but it is not identical, and the
// difference is worth knowing before comparing a screenshot to an old one. A
// vertical flip does not commute with the 2x2 stencil: computeGradient2x2Field
// reads the block (y, y+1), so flipping the image first pairs different rows
// than flipping the result would. `computeField(flipRows(gray))` and
// `flipRows(computeField(gray))` therefore differ by a one-row stencil shift as
// well as by fy's sign. The visible consequence is a half-pixel shift in these
// rasters. What you get in exchange is that the overlay is now showing the
// field the pose was actually computed from, rather than a mirrored
// near-duplicate of it.

// The pose run's field, or null when nobody asked for it (or no reconstruction
// has completed). Callers must handle null by drawing NOTHING rather than by
// falling back to a recomputation -- a silent fallback would put the duplicated
// stage straight back, and would do it on the path where it is hardest to
// notice: the one where the request was misconfigured.
export function pipelineField(camera: Camera): GradientField | null {
  const fx = camera.pose?.intermediates.fx;
  const fy = camera.pose?.intermediates.fy;
  if (!fx || !fy) return null;
  // r: 1 matches computeGradient2x2Field's own output -- the 2x2 stencil reads
  // one pixel forward, so the last row and column are never written.
  return { fx, fy, w: camera.rtSize.w, h: camera.rtSize.h, r: 1 };
}

// A display-space (row, col) to an index into the pipeline's top-down field.
export function displayRowIndex(displayRow: number, col: number, w: number, h: number): number {
  return (h - 1 - displayRow) * w + col;
}

// The pipeline's fy re-expressed in the display's BOTTOM-UP convention -- not
// a screen-space dy, which is a further negation the arrow drawing does for
// itself. Trivial, and named anyway: an un-negated fy read straight out of the
// pipeline is a bug that looks like a working arrow pointing the wrong way,
// which is the kind that survives review.
export function flipDy(fy: number): number {
  return -fy;
}
