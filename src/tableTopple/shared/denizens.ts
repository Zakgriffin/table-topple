import * as THREE from 'three';
import { scene } from './scene.ts';
import { addCrown, buildCharacter, type Character } from './character.ts';
import { buildCastle, CASTLE_HALF_EXTENT } from './castle.ts';
import { forwardFromYaw, rightFromYaw } from './frame.ts';
import { RANKS, type Rank } from './ranks.ts';
import { createVitals, type Vitals } from './health.ts';
import type { WeaponKey } from './weapons.ts';
import {
  CASTLE_GAP, COLOR_TEAM_BLUE, COLOR_TEAM_GREEN, COLOR_TEAM_RED,
  COLOR_TEAM_YELLOW, FORMATION, RANK_SPACING, START_RADIUS,
} from './constants.ts';

// A denizen is one figure on the board: a blocky humanoid (character.ts) plus
// the game-side state that isn't the scene graph's business. Position is kept
// here as board-plane coordinates rather than read back off the transform, so
// rules (occupancy, range, collision) can reason in 2D without touching THREE.
//
// Everything a denizen can do is rank-independent -- a soldier and a king walk,
// turn, and hold weapons through exactly the same code. Rank only sets how big
// it is, how deep its color runs, and whether it wears a crown (ranks.ts).
export interface Denizen {
  /** Stable across this denizen's whole life, unlike its index in `denizens`:
   *  that array is spliced on death (see sim.ts's retire step), so an index
   *  would silently start naming someone else mid-battle. This is what a
   *  network message names a denizen by. */
  id: number;
  character: Character;
  rank: Rank;
  /** Which court it belongs to. Team color, before the rank's shading. */
  team: number;
  /** Size multiplier from its rank, cached here because nearly everything
   *  physical about a denizen scales with it: its footprint on the board, its
   *  stride length, how high its body bobs when walking. */
  scale: number;
  /** Ground position: x, z. The character's origin is between its feet, so
   *  this maps straight onto the group's transform, no height offset. */
  pos: THREE.Vector2;
  /** Facing, radians about +Y. 0 faces +Z, matching buildCharacter(). */
  facing: number;
  /** Position within the walk cycle, radians. Advanced by DISTANCE walked,
   *  not by time -- see animation.ts. */
  walkPhase: number;
  /** 0 = standing still, 1 = full stride. Eased, so starting and stopping
   *  blends in and out of the pose instead of snapping. */
  walkBlend: number;
  /** Current ground velocity, world units per second. Movement is integrated
   *  through this rather than applied straight to pos, so stopping coasts. */
  velocity: THREE.Vector2;
  /** Where this denizen is walking itself to, or null if it isn't. Set by
   *  drawing a region (regionDraw.ts); cleared on arrival, or the moment the
   *  human takes manual control with WASD. */
  moveTarget: THREE.Vector2 | null;
  /** Which weapon is in hand, or null for empty-handed. */
  weapon: WeaponKey | null;
  /** The scene object for that weapon, parented under the right arm. Kept as a
   *  handle so equipping can dispose the old one. */
  heldObject: THREE.Object3D | null;
  /** Where the weapon arm is pointing. pitch swings it forward/back about the
   *  shoulder, yaw swings it across the body. Only meaningful while armed --
   *  an empty arm belongs to the walk cycle. */
  aim: { pitch: number; yaw: number };

  // ── Fighting ────────────────────────────────────────────────────────────
  // Set by whoever is driving this denizen: the human through aim.ts, or a
  // brain through ai.ts. Everything downstream reads these without caring
  // which, so the human and an AI archer run the exact same firing code.

  /** The world point this denizen is aiming at, or null to stand at guard. */
  aimTarget: THREE.Vector3 | null;
  /** 0..1 draw or charge on the held weapon. */
  charge: number;
  /** Seconds into a melee swing, or null when not swinging. The blade only
   *  cuts while this is running. */
  swingFor: number | null;
  /** Position at the start of this frame, so the walk cycle can be driven by
   *  distance actually covered without every mover having to report it. */
  lastPos: THREE.Vector2;
}
/** Hit points, the floating bar, and the death sequence all live in health.ts;
 *  a Denizen simply carries its fields. Declared by extension rather than
 *  inline so health.ts stays the only file that knows what they mean. */
export interface Denizen extends Vitals {}

/** Assigned in build order and never reused, so an id always names the same
 *  denizen for the life of the page even once others have died and been
 *  spliced out of `denizens`. */
let nextId = 0;

function makeDenizen(team: number, rank: Rank, x: number, z: number, facing: number): Denizen {
  const def = RANKS[rank];
  // Rank deepens the team color rather than replacing it: same hue, richer.
  const color = new THREE.Color(team).multiplyScalar(def.shade);
  const character = buildCharacter(color.getHex());
  // One scale on the whole figure, so every part -- and anything it later
  // picks up -- is sized consistently without a single dimension being
  // duplicated per rank. A king's sword is a king-sized sword for free.
  character.group.scale.setScalar(def.scale);
  if (def.crown) addCrown(character);

  const denizen: Denizen = {
    id: nextId++,
    character, rank, team, scale: def.scale,
    pos: new THREE.Vector2(x, z), facing,
    walkPhase: 0, walkBlend: 0, velocity: new THREE.Vector2(), moveTarget: null,
    weapon: null, heldObject: null, aim: { pitch: 0, yaw: 0 },
    aimTarget: null, charge: 0, swingFor: null,
    lastPos: new THREE.Vector2(x, z),
    ...createVitals(),
  };
  syncMesh(denizen);
  scene.add(character.group);
  return denizen;
}

/** Push a denizen's logical position + facing into its scene transform.
 *  Deliberately does NOT touch position.y: the walk cycle owns the vertical
 *  (it bobs the body each step), and having two writers fight over one channel
 *  would make the bob flicker depending on call order. */
export function syncMesh(p: Denizen) {
  p.character.group.position.x = p.pos.x;
  p.character.group.position.z = p.pos.y;
  p.character.group.rotation.y = p.facing;
}

// ── The four courts ───────────────────────────────────────────────────────
// One court per edge, each facing the middle of the board, so red faces blue
// and green faces yellow across the center. Positions are derived from the
// facing rather than written out per court: a court is entirely described by
// "which way is it looking", and the rest follows.

const forward = new THREE.Vector2();
const right = new THREE.Vector2();

function court(team: number, facing: number): Denizen[] {
  // Standing at the edge looking in: the court's center is BACK along its own
  // facing, and the line runs across that facing.
  forwardFromYaw(facing, forward);
  rightFromYaw(facing, right);
  const cx = -forward.x * START_RADIUS, cz = -forward.y * START_RADIUS;

  const line = FORMATION.map((rank, i) => {
    const offset = (i - (FORMATION.length - 1) / 2) * RANK_SPACING;
    return makeDenizen(team, rank, cx + right.x * offset, cz + right.y * offset, facing);
  });

  // One castle per court, behind the line. The setback is derived from the
  // castle's own footprint rather than hardcoded, so changing the castle plan
  // can't quietly leave the front rank standing inside their own wall.
  const castle = buildCastle(team);
  const setback = CASTLE_HALF_EXTENT + CASTLE_GAP;
  castle.group.position.set(cx - forward.x * setback, 0, cz - forward.y * setback);
  castle.group.rotation.y = facing;
  scene.add(castle.group);

  return line;
}

// Facings: 0 looks +Z, and increasing yaw turns toward +X (frame.ts).
const redCourt = court(COLOR_TEAM_RED, 0); // stands at -Z, looks across to blue
const blueCourt = court(COLOR_TEAM_BLUE, Math.PI); // stands at +Z
const greenCourt = court(COLOR_TEAM_GREEN, Math.PI / 2); // stands at -X
const yellowCourt = court(COLOR_TEAM_YELLOW, -Math.PI / 2); // stands at +X

export const denizens: Denizen[] = [...redCourt, ...blueCourt, ...greenCourt, ...yellowCourt];

/** The one under human control: the red king. Everyone else holds their post
 *  until they have a mind. */
export const you = redCourt.find((d) => d.rank === 'king')!;
