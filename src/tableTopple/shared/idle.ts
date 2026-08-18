import * as THREE from 'three';
import { denizens } from './denizens.ts';

// A little life for a standing denizen: the head picks a point somewhere in
// its own field of view, eases to it, holds, then picks a new one -- and the
// eyes blink on an independent, randomized clock. Fades out with walkBlend
// (animation.ts's own "0 standing, 1 full stride") so it never fights an
// actual turn or stride, and switches off entirely while dying -- health.ts
// owns the transform outright from there (its shake would otherwise be
// fighting this for the same rotation).
//
// Deliberately its own small file rather than folded into animation.ts:
// that module is specifically the walk cycle and turn easing, driven by
// distance moved; this is driven by nothing the sim decided, purely time
// passing, which is a different enough job to keep apart.
//
// Nothing else touches head.rotation or an eye's own scale.y, so both are
// written directly here with no seam to go through.

// ── Blinking ─────────────────────────────────────────────────────────────

/** How long a blink stays closed. */
const BLINK_DURATION = 0.12;
/** Range between blinks -- randomized per denizen, per blink, rather than a
 *  fixed period, so it never reads as a metronome. */
const BLINK_MIN_INTERVAL = 2.2;
const BLINK_MAX_INTERVAL = 6;
/** How far shut a blink pinches the eye -- not quite 0, which would collapse
 *  the geometry to a degenerate matrix. */
const BLINK_CLOSED_SCALE = 0.08;

// ── Looking around ───────────────────────────────────────────────────────
// A saccade, not a wander: pick a random point within a 90 degree cone dead
// ahead, ease-in-ease-out to it over MOVE_DURATION, hold there for
// HOLD_DURATION, then pick again. Both durations are re-rolled every time, so
// neither the turn itself nor the wait between turns reads as a metronome.

/** Half the field of view -- 90 degrees total, so a target's yaw is drawn
 *  from [-45deg, 45deg]. */
const FOV_HALF_YAW = Math.PI / 4;
/** Vertical glances stay subtler than side to side -- nodding reads as
 *  agreeing with someone, which isn't the effect wanted here. */
const FOV_HALF_PITCH = Math.PI / 10;

const MOVE_DURATION_MIN = 0.35, MOVE_DURATION_MAX = 0.8;
const HOLD_DURATION_MIN = 1.2, HOLD_DURATION_MAX = 3.5;

/** idle.ts's own per-denizen look state -- see denizens.ts's own `look`
 *  field, which just carries this and knows nothing about what it means.
 *  Nullable there rather than given a meaningful default, so this module
 *  stays the only one that ever constructs one (lazily, on first update,
 *  below) -- the same idiom actions.ts's PendingAction uses. */
export interface LookState {
  fromYaw: number;
  fromPitch: number;
  toYaw: number;
  toPitch: number;
  /** Seconds since `to` was chosen -- covers both the ease (while less than
   *  moveDuration) and the hold that follows it. */
  elapsed: number;
  moveDuration: number;
  holdDuration: number;
}

function randRange(lo: number, hi: number): number {
  return lo + Math.random() * (hi - lo);
}

/** A fresh look target, within the field of view. */
function randomTarget(): { yaw: number; pitch: number } {
  return { yaw: randRange(-FOV_HALF_YAW, FOV_HALF_YAW), pitch: randRange(-FOV_HALF_PITCH, FOV_HALF_PITCH) };
}

function initialLookState(): LookState {
  const target = randomTarget();
  return {
    fromYaw: 0, fromPitch: 0, toYaw: target.yaw, toPitch: target.pitch,
    elapsed: 0,
    moveDuration: randRange(MOVE_DURATION_MIN, MOVE_DURATION_MAX),
    holdDuration: randRange(HOLD_DURATION_MIN, HOLD_DURATION_MAX),
  };
}

/** Smoothstep -- the standard ease-in-ease-out: flat tangents at both ends,
 *  so a turn starts and settles gently instead of snapping into motion or
 *  stopping dead. */
function easeInOut(t: number): number {
  const c = THREE.MathUtils.clamp(t, 0, 1);
  return c * c * (3 - 2 * c);
}

/** Advances one denizen's look target/ease, picking a fresh target once the
 *  current move-and-hold cycle has run its course. Returns the eased
 *  (yaw, pitch) for this frame. */
function updateLook(state: LookState, dt: number): { yaw: number; pitch: number } {
  state.elapsed += dt;

  if (state.elapsed >= state.moveDuration + state.holdDuration) {
    const target = randomTarget();
    state.fromYaw = state.toYaw;
    state.fromPitch = state.toPitch;
    state.toYaw = target.yaw;
    state.toPitch = target.pitch;
    state.elapsed = 0;
    state.moveDuration = randRange(MOVE_DURATION_MIN, MOVE_DURATION_MAX);
    state.holdDuration = randRange(HOLD_DURATION_MIN, HOLD_DURATION_MAX);
  }

  // Clamped past 1 by easeInOut itself once past moveDuration, which is what
  // holds the pose steady through the rest of the cycle without a separate
  // branch for "moving" vs "holding".
  const t = easeInOut(state.elapsed / state.moveDuration);
  return {
    yaw: THREE.MathUtils.lerp(state.fromYaw, state.toYaw, t),
    pitch: THREE.MathUtils.lerp(state.fromPitch, state.toPitch, t),
  };
}

/**
 * Advances every denizen's idle look and blink by one frame. Called once a
 * frame from sim.ts's step(), same as ai.ts's updateAI -- a per-denizen
 * decision loop, not input, so it belongs in the simulation rather than a
 * page's own render loop.
 */
export function updateIdle(dt: number) {
  for (const d of denizens) {
    if (d.dyingFor !== null) continue;

    // ── Looking around ──────────────────────────────────────────────────
    d.look ??= initialLookState();
    const { yaw, pitch } = updateLook(d.look, dt);
    // Eases toward 0 as a walk picks up, same value updateWalk itself reads
    // -- the head settles back to looking forward well before full stride.
    const settle = 1 - d.walkBlend;
    d.character.head.rotation.y = yaw * settle;
    d.character.head.rotation.x = pitch * settle;

    // ── Blinking ─────────────────────────────────────────────────────────
    // Positive: waiting. Crosses zero: the blink itself, for BLINK_DURATION
    // more (negative) seconds. Past that: reroll a fresh wait.
    d.blinkTimer -= dt;
    const blinking = d.blinkTimer <= 0 && d.blinkTimer > -BLINK_DURATION;
    const eyeScale = blinking ? BLINK_CLOSED_SCALE : 1;
    d.character.eyes[0].scale.y = eyeScale;
    d.character.eyes[1].scale.y = eyeScale;
    if (d.blinkTimer <= -BLINK_DURATION) {
      d.blinkTimer = BLINK_MIN_INTERVAL + Math.random() * (BLINK_MAX_INTERVAL - BLINK_MIN_INTERVAL);
    }
  }
}
