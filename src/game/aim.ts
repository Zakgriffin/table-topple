import * as THREE from 'three';
import { camera, canvas } from './scene.ts';
import { equipped, mode, onModeChange, onWeaponChange, setMode } from './mode.ts';
import { equipWeapon, holdingArm, WEAPONS } from './weapons.ts';
import { you } from './denizens.ts';
import type { Denizen } from './denizens.ts';
import { yawFromDirection } from './frame.ts';

// Where the human is aiming, and the arm pose that follows from it.
//
// The single source of truth is a RETICLE: a point on screen, moved by raw
// mouse deltas, from which a ray is cast onto the ground. Everything
// downstream -- which way the body turns, where the arm points, where an arrow
// flies, which way the staff's cone opens -- derives from that one world
// point. An earlier version accumulated arm pitch/yaw straight from the mouse,
// which was fine for a sword but would let an arrow and the arm that loosed it
// disagree: the shoulder clamps at its limits and the arrow doesn't.
//
// Drawing a weapon is not the same as using it. With one out but no button
// held, the denizen stands ON GUARD -- the arm returns to the weapon's ready
// pose and the feet steer normally. Holding the button takes manual control
// and switches the feet to footwork (see main.ts). Releasing settles back.
//
// Holding a button, rather than tracking the pointer continuously, is also
// what makes this portable: press-drag-release means the same thing on a
// trackpad, a mouse, and a touchscreen, where there is no hovering with the
// finger up.
//
// The cursor is taken away with the Pointer Lock API rather than just
// `cursor: none`: locking is what turns the mouse into an unbounded stream of
// deltas. With a plain hidden cursor the pointer stops at the edge of the
// screen and the aim dies partway through a swing.

/** Screen pixels of reticle travel per pixel of mouse movement. */
const SENSITIVITY = 1;

/** How fast the arm settles back to guard once the button is released.
 *  Slower than the tracking rate on purpose -- a blade settling back has
 *  weight, and an instant snap looks like a glitch rather than a recovery. */
const RECOVER_RATE = 9;
/** How fast the arm chases the aim point while under manual control. */
const TRACK_RATE = 22;
/** How fast the body comes round to face where you're aiming. */
const BODY_TURN_RATE = 9;

// Limits, radians. Pitch runs from the arm hanging down (0) up past horizontal;
// yaw swings it across the body. Both are clamped so the arm can't invert
// through the torso -- there are no joint constraints in a stack of boxes, so
// the clamp IS the anatomy. The body turning to follow the reticle is what
// stops the yaw clamp ever being the thing that limits your aim.
const PITCH_MIN = -2.7, PITCH_MAX = 0.5;
const YAW_MIN = -1.1, YAW_MAX = 1.1;

const locked = () => document.pointerLockElement === canvas;

const crosshair = document.getElementById('crosshair') as HTMLDivElement;

/** Reticle position in CSS pixels. Starts centered. */
const reticle = { x: innerWidth / 2, y: innerHeight / 2 };

let engaged = false;

/** True while the button is held, i.e. the human is swinging, drawing, or
 *  charging. main.ts reads it to decide whether the feet steer or do footwork;
 *  combat.ts reads it to drive attacks. */
export function isEngaged(): boolean {
  return engaged;
}

// ── Reticle -> world ──────────────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const groundHit = new THREE.Vector3();

/**
 * The point on the ground the reticle is over.
 *
 * When the reticle is above the horizon the ray never meets the ground, so it
 * falls back to a point far along the ray -- aiming at the sky has to mean
 * something, and "a long way off in that direction" is the sane reading.
 */
export function aimPoint(out: THREE.Vector3): THREE.Vector3 {
  ndc.x = (reticle.x / innerWidth) * 2 - 1;
  ndc.y = -(reticle.y / innerHeight) * 2 + 1;
  raycaster.setFromCamera(ndc, camera);
  if (raycaster.ray.intersectPlane(groundPlane, groundHit)) return out.copy(groundHit);
  return out.copy(raycaster.ray.origin).addScaledVector(raycaster.ray.direction, 200);
}

/** World position of a denizen's weapon hand -- where a shot originates. */
export function handPosition(d: Denizen, out: THREE.Vector3): THREE.Vector3 {
  const arm = holdingArm(d);
  arm.updateWorldMatrix(true, false);
  out.setFromMatrixPosition(arm.matrixWorld);
  // The arm's origin is its shoulder; the fist is a limb-length down it.
  // Close enough to take the shot from, and it moves with the aim.
  return out;
}

// ── Input ─────────────────────────────────────────────────────────────────

addEventListener('mousemove', (e) => {
  if (!locked() || mode !== 'fight') return;
  // movementX/Y, not clientX/Y: under pointer lock the cursor has no position,
  // only deltas. The reticle is our own cursor, so it also has to be held
  // inside the window by hand.
  reticle.x = THREE.MathUtils.clamp(reticle.x + e.movementX * SENSITIVITY, 0, innerWidth);
  reticle.y = THREE.MathUtils.clamp(reticle.y + e.movementY * SENSITIVITY, 0, innerHeight);
});

// A resize can strand the reticle outside the new viewport.
addEventListener('resize', () => {
  reticle.x = THREE.MathUtils.clamp(reticle.x, 0, innerWidth);
  reticle.y = THREE.MathUtils.clamp(reticle.y, 0, innerHeight);
});

// pointerdown/up rather than mousedown/up: pointer events already cover touch,
// so the same press-drag-release path works unchanged on a phone later.
addEventListener('pointerdown', (e) => {
  if (mode !== 'fight' || e.button !== 0) return;
  engaged = true;
});
const disengage = () => { engaged = false; };
addEventListener('pointerup', disengage);
addEventListener('pointercancel', disengage);
// A window that loses focus never delivers the release, which would otherwise
// leave the weapon stuck under manual control with nothing driving it.
addEventListener('blur', disengage);

onModeChange((next) => {
  engaged = false;
  if (next === 'fight') {
    equipWeapon(you, equipped());
    // Re-center, so a fight always opens aiming straight ahead rather than
    // wherever the reticle happened to be left.
    reticle.x = innerWidth / 2;
    reticle.y = innerHeight / 2;
    // Can reject (no user activation, or a lock released moments ago). Not
    // fatal -- the weapon is still drawn and aimable the instant a lock does
    // land -- so swallow it rather than letting it surface as an unhandled
    // rejection.
    void canvas.requestPointerLock?.()?.catch?.(() => {});
  } else {
    equipWeapon(you, null);
    if (locked()) document.exitPointerLock();
  }
  crosshair.style.display = next === 'fight' ? 'block' : 'none';
});

// Swapping weapons WHILE already fighting. Without this, only the first 1/2/3
// did anything: entering fight mode equips via the mode change above, but a
// later press stays in fight mode, so nothing was listening and the old weapon
// simply stayed in hand.
onWeaponChange((key) => {
  if (mode === 'fight') equipWeapon(you, key);
});

// Escape never reaches our keydown handler while the pointer is locked -- the
// browser consumes it to release the lock. Catching the release here is what
// makes Escape still mean "back to camera", and it also covers the other ways
// a lock can end (tab switch, window manager).
document.addEventListener('pointerlockchange', () => {
  if (!locked() && mode === 'fight') setMode('camera');
});

// ── Pose ──────────────────────────────────────────────────────────────────

const target = new THREE.Vector3();
const hand = new THREE.Vector3();

/**
 * Turns the body toward the aim point and points the weapon arm at it.
 *
 * Must run AFTER updateWalk each frame, and that ordering is the whole design:
 * both want the same arm, and rather than teaching the walk cycle about
 * weapons, the armed pose simply overwrites it. Walking with a weapon out
 * still swings the legs and the free arm, so the character reads as walking --
 * only the weapon arm is held on target.
 */
export function updateAim(d: Denizen, dt: number) {
  if (!d.weapon) return;
  const def = WEAPONS[d.weapon];
  const arm = holdingArm(d);

  // Driven entirely by aimTarget, which the human sets from the reticle and a
  // brain sets from whoever it's fighting. There is deliberately no `d === you`
  // branch here: an AI archer poses through exactly the same code the human
  // does, so the two can't drift apart.
  if (d.aimTarget) {
    target.copy(d.aimTarget);
    handPosition(d, hand);

    // Bring the body round to face where you're aiming. This is what makes the
    // shoulder's yaw clamp a non-issue: aim wide and the whole figure turns,
    // rather than the arm hitting its limit while the shot goes somewhere the
    // arm isn't pointing.
    const wantYaw = yawFromDirection(target.x - d.pos.x, target.z - d.pos.y);
    // Short way round: without the wrap, aiming across the +/-PI seam sends
    // the body the long way about.
    const delta = Math.atan2(Math.sin(wantYaw - d.facing), Math.cos(wantYaw - d.facing));
    d.facing += delta * (1 - Math.exp(-BODY_TURN_RATE * dt));

    // Arm pitch from the elevation of the target as seen from the hand. The
    // arm hangs at 0 and points level at -PI/2, so level plus elevation is
    // -(PI/2 + elevation).
    const flat = Math.hypot(target.x - hand.x, target.z - hand.z);
    const elevation = Math.atan2(target.y - hand.y, Math.max(flat, 1e-4));
    const wantPitch = THREE.MathUtils.clamp(-(Math.PI / 2 + elevation), PITCH_MIN, PITCH_MAX);

    // Whatever the body hasn't turned through yet is taken up by the shoulder,
    // so the arm leads the turn instead of lagging behind it.
    const residual = Math.atan2(Math.sin(wantYaw - d.facing), Math.cos(wantYaw - d.facing));
    const wantArmYaw = THREE.MathUtils.clamp(residual, YAW_MIN, YAW_MAX);

    d.aim.pitch = THREE.MathUtils.damp(d.aim.pitch, wantPitch, TRACK_RATE, dt);
    d.aim.yaw = THREE.MathUtils.damp(d.aim.yaw, wantArmYaw, TRACK_RATE, dt);
  } else if (def.readyAim) {
    d.aim.pitch = THREE.MathUtils.damp(d.aim.pitch, def.readyAim[0], RECOVER_RATE, dt);
    d.aim.yaw = THREE.MathUtils.damp(d.aim.yaw, def.readyAim[1], RECOVER_RATE, dt);
  }

  arm.rotation.x = d.aim.pitch;
  arm.rotation.z = d.aim.yaw;

  // A swing overrides the aim pose outright, and is applied last. Aiming holds
  // the blade ON a target; a swing is the blade travelling THROUGH it, which
  // is the motion the hitbox is tested against (combat.ts). Blending the two
  // would give a blade that neither points anywhere nor sweeps anything.
  if (d.swingFor !== null) arm.rotation.x = d.aim.pitch + swingOffset(d.swingFor);
}

/** How far a swing has carried the arm past its aimed pose, radians.
 *  A fast chop forward and a slower recovery -- symmetric would read as a
 *  metronome rather than a strike. */
export const SWING_TIME = 0.36;
function swingOffset(t: number): number {
  const u = THREE.MathUtils.clamp(t / SWING_TIME, 0, 1);
  // Up on the first third, down hard through the rest.
  return u < 0.33
    ? -0.9 * (u / 0.33)
    : -0.9 + 2.1 * ((u - 0.33) / 0.67) ** 0.7;
}

const chargeRing = document.getElementById('charge') as HTMLDivElement;

/** Moves the on-screen crosshair to the reticle and fills its charge ring.
 *  Driven per frame rather than from the mousemove handler so it can't run
 *  ahead of the rendered frame. */
export function updateCrosshair(charge: number) {
  if (mode !== 'fight') return;
  crosshair.style.transform =
    `translate(${reticle.x}px, ${reticle.y}px) translate(-50%, -50%)`;
  chargeRing.style.setProperty('--charge', String(charge));
  // Hidden entirely at rest, so a sword (which never charges) shows no ring.
  chargeRing.style.opacity = charge > 0.001 ? '1' : '0';
}
