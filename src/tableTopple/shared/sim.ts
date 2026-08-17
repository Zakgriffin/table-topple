import * as THREE from 'three';
import {
  BOARD_SIZE, WALK_ACCEL_RATE, WALK_DECEL_RATE, WALK_SPEED,
} from './constants.ts';
import { advance, arrivalSpeed } from './motion.ts';
import { CHARACTER_HALF_WIDTH } from './character.ts';
import { denizens, syncMesh, you, type Denizen } from './denizens.ts';
import { turnToward, updateWalk } from './animation.ts';
import { cameraYaw, moveAxes } from './input.ts';
import { basisToWorld, signedAlongYaw, yawFromDirection } from './frame.ts';
import { clearRegion } from './regionDraw.ts';
import { applyAimPose, isEngaged, updateAim } from './aim.ts';
import { advanceVitals, renderVitals, retire } from './health.ts';
import { equipWeapon } from './weapons.ts';
import { updateAI } from './ai.ts';
import { updateCombat } from './combat.ts';
import type { DenizenStateEntry } from './protocol.ts';
import './board.ts'; // side effect: adds the floor to the scene

// One frame of the world, and nothing about drawing it.
//
// This is the half of the old main.ts that both hosts share: the standalone
// page calls it from its own rAF loop before rendering. Deliberately no
// renderer, no controls.update(), no crosshair -- those are decisions about a
// viewport, and step() has none.
//
// Note that importing this module is also what BUILDS the world: the floor
// (board.ts) and the four courts (denizens.ts) add themselves to the scene as
// an import side effect, so a host that wants the board without simulating it
// still starts here and simply never calls step().
//
// ── SIMULATE vs RENDER, THE SEAM MULTIPLAYER RUNS ON ──
//
// step() decides things (AI, physics, combat, who's dying) and is desktop-
// only -- the ONE source of truth. Everything that turns a denizen's current
// fields into scene transforms lives instead in renderDenizen/renderWorld,
// below, which step() also calls but which know nothing about how those
// fields got their values. The AR overlay on the capture page is a pure
// receiver: it never calls step(), only renderWorld/applyDenizenSnapshot, off
// whatever a `denizenState` broadcast last wrote into `denizens` (see
// client/main.ts and shared/net.ts). Same render code either way, so the two
// hosts can't paint the same state two different ways.

const axes = new THREE.Vector2();
const dir = new THREE.Vector2();
/** Velocity the character is currently trying to have. The gap between this
 *  and its actual velocity is what acceleration and coasting live in. */
const desired = new THREE.Vector2();

/** How far the controlled denizen's center can get from the board's center
 *  before its edge would overhang. Scales with rank -- a king is twice as wide
 *  as a soldier -- so it can't be a constant. */
const limit = () => BOARD_SIZE / 2 - CHARACTER_HALF_WIDTH * you.scale;

/** Distance at which a walk-to is called done. The approach already tapers to
 *  a crawl by this point (arrivalSpeed), so closing the last sliver instantly
 *  isn't visible -- it just stops the target being chased forever. */
const ARRIVE_EPSILON = 0.03;

export function step(dt: number) {
  moveAxes(axes);

  // Snapshot every position up front. The walk cycle is driven by distance
  // actually covered, and there are now two things that move denizens -- the
  // human's input below and the brains in ai.ts -- so measuring it centrally
  // beats making each mover report its own displacement.
  for (const p of denizens) p.lastPos.copy(p.pos);

  // Holding the button with the sword out is what puts the character in a
  // fighting stance: the body stops steering and holds its heading so the
  // blade stays trained downrange, and WASD becomes footwork in the
  // character's own frame (W/S advance and back up, A/D sidestep). Let go and
  // it's an ordinary walk again -- camera-relative, turning to follow. Sword
  // merely DRAWN is not enough; it takes a held button.
  const stance = you.weapon !== null && isEngaged();

  // Nothing here writes position directly any more; it all goes through a
  // desired velocity, and advance() below does the integrating. That's what
  // makes releasing the keys coast to a halt rather than cutting out.
  desired.set(0, 0);
  // No input means coasting, which is the slow rate. Anything driving the
  // character uses the brisk one.
  let rate = WALK_DECEL_RATE;

  if (axes.lengthSq() > 0) {
    // The whole difference between the two movement styles: which yaw the
    // input axes are read against.
    basisToWorld(axes, stance ? you.facing : cameraYaw(), dir);

    // Grabbing the keys cancels a walk-to. Manual input should always win
    // outright -- fighting the auto-walk for control would feel broken. The
    // region goes with it: he's no longer heading there.
    if (you.moveTarget) { you.moveTarget = null; clearRegion(); }
    desired.copy(dir).multiplyScalar(WALK_SPEED);
    rate = WALK_ACCEL_RATE;
    if (!stance) turnToward(you, yawFromDirection(dir.x, dir.y), dt);
  } else if (you.moveTarget) {
    const dx = you.moveTarget.x - you.pos.x, dz = you.moveTarget.y - you.pos.y;
    const remaining = Math.hypot(dx, dz);
    if (remaining <= ARRIVE_EPSILON) {
      you.pos.copy(you.moveTarget);
      you.velocity.set(0, 0);
      you.moveTarget = null;
      // Arrived -- this is what finally wipes the region off the floor.
      clearRegion();
    } else {
      dir.set(dx / remaining, dz / remaining);
      // Tapered near the destination so the character eases into it instead of
      // running at full tilt until it snaps.
      desired.copy(dir).multiplyScalar(arrivalSpeed(remaining, WALK_SPEED, WALK_DECEL_RATE));
      rate = WALK_ACCEL_RATE;
      if (!stance) turnToward(you, yawFromDirection(dir.x, dir.y), dt);
    }
  }

  advance(you.pos, you.velocity, desired, rate, dt, limit());

  // The brains move everyone else, through the same motion and animation
  // helpers the human goes through.
  updateAI(dt);

  // Every denizen runs the same per-frame work -- this loop never names a
  // specific one, which is why the AI needed no seam cut for it.
  //
  // Backwards, because advanceVitals can retire a denizen mid-pass and
  // the loop below splices it out of this very array. Iterating forwards would
  // skip whoever moved into the gap.
  for (let i = denizens.length - 1; i >= 0; i--) {
    const p = denizens[i];
    // A dying figure is done aiming; health.ts owns its transform from here
    // (the shake moves it off its logical position, which syncMesh would
    // immediately undo). updateAim is the DECIDING half -- it turns aimTarget
    // into aim.pitch/yaw -- kept here, on the authoritative host only; the
    // PAINTING half (applyAimPose) moves to renderDenizen below, where a
    // receiving client can reach it too.
    if (p.dyingFor === null) updateAim(p, dt);
    if (advanceVitals(p, dt)) {
      // Retiring is split: health.ts frees the scene objects, and the roster
      // is spliced here, where we already hold the index. That split is what
      // keeps health.ts free of a runtime import back into denizens.ts.
      retire(p);
      denizens.splice(i, 1);
      continue;
    }
    renderDenizen(p, dt);
  }

  // After the poses, so the blade's hitbox is tested where it is actually
  // being drawn this frame rather than where it was last frame.
  updateCombat(dt);
}

// ── Presentation, shared with a receiving client ────────────────────────────
//
// Everything below turns a denizen's CURRENT fields into scene transforms and
// nothing else -- no AI, no physics, no combat decisions. step() above calls
// it once simulation has decided this tick's values; a client with no sim of
// its own (see net.ts/protocol.ts) calls it with values that just arrived
// over the wire instead. Same function either way, so the two can't render
// the same state two different ways.

/**
 * One denizen's presentation for this frame: the walk cycle (from distance
 * actually covered since the last call), the weapon arm, and the health
 * bar/death fade. Always updates `lastPos` at the end, which is what lets
 * BOTH step()'s own per-tick call and a client's per-snapshot call measure
 * "distance moved" the same way, regardless of how often either is called.
 */
export function renderDenizen(p: Denizen, dt: number) {
  if (p.dyingFor === null) {
    // Signed, not just a magnitude: a denizen can walk backwards or
    // sideways -- the human in stance, an archer backing off a rusher -- and
    // feeding the walk cycle a negative distance runs the legs in reverse,
    // so a backstep reads as a backstep instead of a forward march sliding
    // the wrong way.
    const dx = p.pos.x - p.lastPos.x, dz = p.pos.y - p.lastPos.y;
    let moved = Math.hypot(dx, dz);
    if (signedAlongYaw(dx, dz, p.facing) < 0) moved = -moved;

    syncMesh(p);
    updateWalk(p, moved, dt);
    // Strictly after updateWalk: an armed denizen's weapon arm is aimed, not
    // swung, so the aim pose overwrites what the walk cycle just wrote to
    // that one limb. Everything else keeps walking normally. Reversing these
    // two lines silently hands the arm back to the walk cycle.
    applyAimPose(p);
  }
  p.lastPos.copy(p.pos);
  renderVitals(p);
}

/** Renders every denizen with no simulation at all -- what a receiving client
 *  calls instead of step(), EVERY display frame regardless of whether a new
 *  network snapshot has arrived (see writeDenizenSnapshot), off whatever is
 *  currently in `denizens`. That "every frame, not just on new data" is
 *  deliberate: the render clock and the network clock are different rates,
 *  and a camera-facing health bar still needs to track a moving AR camera on
 *  a frame with no fresh state. */
export function renderWorld(dt: number) {
  for (const p of denizens) renderDenizen(p, dt);
}

/** Every denizen, projected for the wire (DenizenStateEntry, protocol.ts).
 *  The desktop calls this once per tick, after step(), to build the
 *  `denizenState` broadcast. */
export function snapshotDenizens(): DenizenStateEntry[] {
  return denizens.map((p) => ({
    id: p.id,
    pos: { x: p.pos.x, y: p.pos.y },
    facing: p.facing,
    weapon: p.weapon,
    aim: { pitch: p.aim.pitch, yaw: p.aim.yaw },
    charge: p.charge,
    swingFor: p.swingFor,
    hp: p.hp,
    barTimer: p.barTimer,
    dyingFor: p.dyingFor,
  }));
}

/**
 * Writes a `denizenState` snapshot into the LOCAL roster -- what a receiving
 * client calls the instant a message arrives (client/main.ts's NetHost).
 * Deliberately does NOT render: a network message and a display frame are
 * different clocks (see renderWorld's own comment), and painting here would
 * mean a health bar's camera-facing billboard only refreshing when a
 * message happens to land rather than every frame the AR camera moves.
 *
 * A denizen whose id is missing from `entries` has been retired by the
 * authoritative host, and is retired here too: there is no separate "denizen
 * removed" message, because absence from the next snapshot already says it.
 * Never decides anything itself -- hp, dyingFor, aim, all of it is copied in
 * verbatim from a host that already decided.
 */
export function writeDenizenSnapshot(entries: readonly DenizenStateEntry[]) {
  const byId = new Map(entries.map((e) => [e.id, e]));
  for (let i = denizens.length - 1; i >= 0; i--) {
    const p = denizens[i];
    const e = byId.get(p.id);
    if (!e) {
      retire(p);
      denizens.splice(i, 1);
      continue;
    }
    p.pos.set(e.pos.x, e.pos.y);
    p.facing = e.facing;
    if (p.weapon !== e.weapon) equipWeapon(p, e.weapon);
    p.aim.pitch = e.aim.pitch;
    p.aim.yaw = e.aim.yaw;
    p.charge = e.charge;
    p.swingFor = e.swingFor;
    p.hp = e.hp;
    p.barTimer = e.barTimer;
    p.dyingFor = e.dyingFor;
  }
}
