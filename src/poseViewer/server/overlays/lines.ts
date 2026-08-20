import * as THREE from 'three';
import { type Camera, type CompositeLine } from '../../shared/camera/model.ts';
import { hsvToRgb } from '../pipeline/distortion.ts';
import { regionRgb } from './lsdOverlay.ts';
import { svgEl } from './svgUtil.ts';

// ── ONE LINE MODEL, ONE COLOUR, THREE PROJECTIONS ─────────────────────────
//
// The detected lines are ONE set of objects, and this file is the one place
// that decides what they are and what colour they are. Three views then draw
// that same list through three different maps:
//
//   Through Cam    image space, SVG            (drawLinesSvg + the rect map)
//   Projected Cam  rectified floor space, SVG  (drawLinesSvg + the gnomonic map)
//   World/Inside   great-circle arcs on the sphere (sphereOverlays.ts)
//
// THIS REPLACED THREE UNRELATED ENCODINGS OF THE SAME THING. Through-Cam hashed
// the composite index and shaded by rank at 0.7px/0.5 opacity; Projected-Cam
// drew flat family blue/red at 1.5px/0.8; World drew one full circle per VOTE
// coloured red->blue by vote WEIGHT, a quantity nothing else on screen used.
// Three pictures of one fact, none of which could be compared with another.
//
// ── WHY A SEGMENT KNOWS ITS COMPOSITE ────────────────────────────────────
//
// Everything here keys off the COMPOSITE index, including for raw segments.
// That is not a convenience: `family` is written by gpp.classify, which is bound
// to `compLines`, so it is indexed by composite and reading it with a segment
// index pairs a segment with an unrelated line's classification -- a bug this
// codebase has already had. `segments[j].composite` is join.reduce's own gather
// rerun on the host (see unpackSegments), so a segment and the composite that
// swallowed it come out the same colour, and several segments sharing a colour
// IS the join, made visible.

export type LineKind = 'segment' | 'composite';

export interface DrawableLine {
  /** Endpoints in the pipeline's own top-down pixel space. */
  line: CompositeLine;
  /** Index into `lineFamily` and into the colour space. -1 when unresolved. */
  composite: number;
  kind: LineKind;
}

// ── The one style table, read by ALL THREE views ─────────────────────────
//
// The two SVG views take both numbers directly. The sphere takes the OPACITY
// through its per-kind material (camera/factory.ts) and the WIDTH as a
// multiplier on the arc ribbon's half-width, so the ratio between the two kinds
// is the same on the sphere as on a rect even though the units are not.
//
// SEGMENTS ARE THE HEAVIER STROKE, which is deliberate and worth stating because
// the obvious guess is the other way round: the segments are the detector's
// actual output and the thing being read, and the composite over them is the
// summary. Drawn the other way the summary hides its own evidence.
export const LINE_STYLE: Record<LineKind, { width: number; opacity: number }> = {
  segment: { width: 1.2, opacity: 0.8 },
  composite: { width: 0.8, opacity: 0.6 },
};

/**
 * Each classified line's position within its own family, ordered by the
 * rectified coordinate -- which is the order the period search assigns integer
 * indices in, so this is literally the sequence the fit registers them as. A
 * segment whose shade breaks the gradient along the floor is one the search has
 * ordered wrongly, which is what makes a period wrong.
 *
 * Keyed by COMPOSITE index, and sorted here rather than upstream: `rectified`
 * arrives in line order because that is the order `samples` is written in, and
 * the two families interleave. The pipeline's own compaction does not sort
 * either, since gpp only ever needs the spread and the resultant. So the
 * ranking is a display question and this is where it belongs.
 */
function familyRanks(camera: Camera): Map<number, { isRow: boolean; t: number }> | null {
  const rectified = camera.pose?.rectified, family = camera.pose?.lineFamily;
  const lineCount = camera.pose?.lineCount;
  if (!rectified || !family || lineCount === undefined) return null;

  const n = Math.min(lineCount, family.length, rectified.length / 4);
  const byFamily: [{ i: number; value: number }[], { i: number; value: number }[]] = [[], []];
  for (let i = 0; i < n; i++) {
    const fam = family[i]!;
    if (fam < 0) continue; // neither family -- see CameraPose.lineFamily
    byFamily[fam]!.push({ i, value: rectified[i * 4]! });
  }
  const out = new Map<number, { isRow: boolean; t: number }>();
  for (const fam of [0, 1] as const) {
    const list = byFamily[fam]!;
    list.sort((p, q) => p.value - q.value);
    // A single-member family has no gradient to shade along, so it takes the
    // full colour rather than a 0/0.
    const last = Math.max(list.length - 1, 1);
    list.forEach((e, k) => out.set(e.i, { isRow: fam === 1, t: k / last }));
  }
  return out;
}

/**
 * The colour of every line in every view, as [r, g, b] 0-255.
 *
 * TWO COLOURINGS, answering different questions. By default a line takes
 * `regionRgb` hashed from its COMPOSITE index -- "which detection is this", and
 * the key that makes a composite and its member segments one object across all
 * three views. `colorLinesByFamily` answers "what will the period search make of
 * this" instead: blue for the row family, red for the column, each shaded by the
 * line's rank within its family. Grey for a line gpp classified into neither
 * (degenerate, or no gnomonic image).
 */
export function lineColorer(camera: Camera): (composite: number) => [number, number, number] {
  const ranks = camera.settings.colorLinesByFamily ? familyRanks(camera) : null;
  return (composite: number) => {
    if (composite < 0) return [130, 130, 130];
    if (!ranks) return regionRgb(composite);
    const ranked = ranks.get(composite);
    if (!ranked) return [130, 130, 130];
    // Shade from near-black at rank 0 to full colour at the last -- the
    // sequence, not just the membership. VALUE, not hue: the hue is the family
    // and has to stay readable as one of two things.
    return hsvToRgb(ranked.isRow ? 210 : 0, 0.85, 0.35 + 0.65 * ranked.t);
  };
}

/**
 * What both line toggles currently ask to be drawn, in one list.
 *
 * Composites last so they land ON TOP of their own members. Absent buffers mean
 * the run was never asked for them -- the standing rule in this directory is to
 * draw nothing rather than recompute, so a toggle listed here and not in
 * `inspectFor` fails as a blank overlay, which is visible.
 */
export function drawableLines(camera: Camera): DrawableLine[] {
  const s = camera.settings;
  const out: DrawableLine[] = [];
  if (s.showLineSegments) {
    for (const seg of camera.pose?.segments ?? []) {
      out.push({ line: seg.line, composite: seg.composite, kind: 'segment' });
    }
  }
  if (s.showCompositeLines) {
    for (const c of camera.pose?.composites ?? []) {
      out.push({ line: c.line, composite: c.index, kind: 'composite' });
    }
  }
  return out;
}

/**
 * Draw the model into an SVG group through an arbitrary projection.
 *
 * `project` returns null for a line this view cannot place -- the gnomonic map
 * has no image for a ray nearly parallel to the tangent plane -- and the line is
 * skipped rather than drawn at an invented position.
 *
 * One insertion, not one per line: appending into a live tree makes the browser
 * consider layout for each child.
 */
export function drawLinesSvg(
  group: SVGGElement, camera: Camera, lines: readonly DrawableLine[],
  project: (x: number, y: number) => { x: number; y: number } | null,
): void {
  while (group.firstChild) group.removeChild(group.firstChild);
  const colorOf = lineColorer(camera);
  const frag = document.createDocumentFragment();
  for (const l of lines) {
    const a = project(l.line.x1, l.line.y1);
    const b = project(l.line.x2, l.line.y2);
    if (!a || !b) continue;
    const [r, g, bl] = colorOf(l.composite);
    const style = LINE_STYLE[l.kind];
    frag.appendChild(svgEl('line', {
      x1: a.x, y1: a.y, x2: b.x, y2: b.y,
      stroke: `rgb(${r},${g},${bl})`,
      'stroke-width': style.width,
      opacity: style.opacity,
    }));
  }
  group.appendChild(frag);
}

/**
 * The camera-space ray through a pixel, byte-identical to VOTES_WGSL's `rayDir`
 * and to GPP_CLASSIFY_WGSL's copy of it.
 *
 * Three copies of one projection is two too many, but the alternative is a
 * fourth convention: a display MUST cast the ray the device cast or it is
 * drawing a different line. This one is the host's, and both the sphere arcs and
 * the rectified map go through it.
 */
export function cameraRay(
  tanHalf: number, aspect: number, w: number, h: number,
): (px: number, py: number) => THREE.Vector3 {
  return (px, py) => {
    const ndcU = (px / w) * 2 - 1;
    const ndcV = 1 - (py / h) * 2;
    return new THREE.Vector3(tanHalf * aspect * ndcU, tanHalf * ndcV, -1).normalize();
  };
}
