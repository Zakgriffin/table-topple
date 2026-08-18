import * as THREE from 'three';
import type { Resource } from './inventory.ts';

// A shared shape for "a thing on the board that a click can pick out and
// light up" -- denizens, terrain structures, trees, and (in kind, if not yet
// wired below) roads are all currently their own bespoke representation
// (Denizen, PlacedLandmark, an untracked InstancedMesh instance, Road), each
// with its own idea of "how far is this from a point" and "how do I glow".
// act.ts is the first thing that needs to ask that question of more than one
// of them at once, which is what makes a shared answer worth having instead
// of a fourth bespoke hit-test.
//
// Deliberately NOT a circle-vs-point unification: `distanceTo` is a method,
// not a stored radius, so a kind whose footprint isn't a circle (a road is a
// segment -- see roads.ts's own closestOnSegment) can still answer honestly
// instead of being squashed into one shared formula. Every kind built so far
// answers "distance to my own boundary, 0 once a point is inside/on it" --
// see denizens.ts, landmarks.ts, forest.ts.
//
// Pure: no scene.ts, same reasoning terrain.ts and motion.ts give for
// staying that way.

export type InteractableKind = 'denizen' | 'landmark' | 'tree' | 'road';

export interface Interactable {
  kind: InteractableKind;
  id: number;
  /** Where to aim a "walk here" command, or to center a selection ring --
   *  the canonical point this thing sits at, regardless of how `distanceTo`
   *  measures reach around it. */
  pos: THREE.Vector2;
  /** Footprint radius around `pos`, for a kind whose footprint IS a circle --
   *  denizen, landmark, tree all have one (their `distanceTo` is built from
   *  it: `max(0, dist(pt, pos) - radius)`). Optional, not required, because
   *  this is still not the circle-vs-point unification the header above
   *  rules out -- a road's footprint is a segment, has no single radius, and
   *  would leave this undefined rather than fake one. act.ts uses it for two
   *  things a bare distanceTo can't answer: drawing the hitbox itself
   *  on hover, and picking a walk destination on its EDGE rather than its
   *  center. */
  radius?: number;
  /** What acting on this thing pays out, one unit per act -- a tree's always
   *  wood, a landmark's whatever terrain.ts's patch grew (landmarks.ts).
   *  Undefined for a kind acting doesn't apply to (a denizen, a road), which
   *  is what act.ts's own commit checks before building a harvest action. */
  resource?: Resource;
  /** Closest distance from `pt` to this thing's own footprint. Never
   *  negative -- 0 once `pt` is already inside/on it, same convention
   *  roads.ts's own closestOnLandmark uses. */
  distanceTo(pt: THREE.Vector2): number;
  /** Turns this thing's highlight on/off, however its own visual setup makes
   *  that cheap: an emissive boost on materials that are never shared with
   *  anyone else (a denizen), a shared-material swap (a landmark, which
   *  reuses blocks.ts's sceneryMaterial/sceneryHighlightMaterial the same
   *  way roads.ts's own paintTarget does), or a no-op for a kind that has no
   *  cheap per-instance way to do it yet (a tree -- see forest.ts). */
  setHighlighted(on: boolean): void;
}

/**
 * The closest `item` to `pt` whose own `distanceTo(pt)` is within `maxDist`,
 * or null if nothing qualifies. Kind-agnostic -- callers filter to whichever
 * kinds are relevant first (see act.ts).
 */
export function nearestWithin(
  pt: THREE.Vector2,
  items: Iterable<Interactable>,
  maxDist: number,
): { item: Interactable; dist: number } | null {
  let best: { item: Interactable; dist: number } | null = null;
  for (const item of items) {
    const dist = item.distanceTo(pt);
    if (dist > maxDist) continue;
    if (!best || dist < best.dist) best = { item, dist };
  }
  return best;
}
