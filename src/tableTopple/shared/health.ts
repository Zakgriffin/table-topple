import * as THREE from 'three';
import { camera, scene } from './scene.ts';
import {
  DEATH_FADE_TIME, DEATH_SHAKE_TIME, HEALTH_BAR_FADE, HEALTH_BAR_LINGER,
  MAX_HP, SOLDIER_HEIGHT,
} from './constants.ts';
// Type-only, deliberately: denizens.ts calls createVitals() at module scope
// while building the courts, so a VALUE import back from here would be a real
// runtime cycle whose correctness depended on which module happened to
// evaluate first. A type import is erased, so there's no cycle at all --
// which is why taking a denizen out of the roster is main.ts's job, not this
// file's (see retire() below).
import type { Denizen } from './denizens.ts';

// Hit points, the bar that shows them, and dying.
//
// The bar is deliberately transient: it appears when a denizen is hit and
// fades out a few seconds later. A board with 36 figures on it would be a wall
// of permanent bars otherwise, and the information only matters just after
// something happens.

const BAR_W = 0.85, BAR_H = 0.1;
/** How far above the head the bar floats, in that denizen's own scaled units. */
const BAR_CLEARANCE = 0.35;

// Full green through amber to red. Read at a glance from across the board,
// which is the only thing this colour has to do.
const HEALTHY = new THREE.Color(0x5ecf6a);
const HURT = new THREE.Color(0xe8c547);
const CRITICAL = new THREE.Color(0xe5484d);

const barGeo = new THREE.PlaneGeometry(BAR_W, BAR_H);
// Shift the fill's vertices so its origin is its LEFT edge. Scaling x then
// eats it from the right like a real bar, instead of shrinking it toward its
// own middle from both ends.
const fillGeo = new THREE.PlaneGeometry(BAR_W, BAR_H).translate(BAR_W / 2, 0, 0);

export interface HealthBar {
  group: THREE.Group;
  fill: THREE.Mesh;
  backMat: THREE.MeshBasicMaterial;
  fillMat: THREE.MeshBasicMaterial;
}

/**
 * Builds one floating bar: dark backing, coloured fill that grows or shrinks
 * from its LEFT edge (fillGeo's own shifted origin, above) so a caller only
 * ever has to touch fill.scale.x. Exported and generic on purpose --
 * actions.ts's own harvest progress bar is built by this exact function,
 * just with a different starting fill colour, rather than a second copy of
 * this mesh setup existing anywhere. A caller that wants the colour to
 * change live (health's own healthy/hurt/critical, below) writes
 * bar.fillMat.color itself.
 */
export function createBar(fillColor: number): HealthBar {
  const group = new THREE.Group();

  // depthTest off: a bar half-buried in a castle wall is worse than one that
  // reads through it. renderOrder keeps it above the rest of the transparent
  // pass rather than fighting with it.
  const backMat = new THREE.MeshBasicMaterial({
    color: 0x11131a, transparent: true, opacity: 0.75, depthTest: false, depthWrite: false,
  });
  const fillMat = new THREE.MeshBasicMaterial({
    color: fillColor, transparent: true, depthTest: false, depthWrite: false,
  });

  const back = new THREE.Mesh(barGeo, backMat);
  const fill = new THREE.Mesh(fillGeo, fillMat);
  fill.position.x = -BAR_W / 2;
  fill.position.z = 0.001; // in front of the backing, in the bar's own plane
  group.add(back, fill);
  group.renderOrder = 20;
  group.visible = false;

  // Parented to the SCENE, not to the denizen: the bar has to billboard toward
  // the camera and stay one constant size, and living under a rotating,
  // rank-scaled, walk-bobbing character group would mean undoing all three
  // transforms every frame.
  scene.add(group);
  return { group, fill, backMat, fillMat };
}

function createHealthBar(): HealthBar {
  return createBar(HEALTHY.getHex());
}

/** Fields health.ts owns on a denizen. Created here so denizens.ts doesn't
 *  need to know the details of any of it. */
export interface Vitals {
  hp: number;
  /** Seconds remaining on the health bar's visibility. */
  barTimer: number;
  /** Seconds elapsed since death began, or null while alive. */
  dyingFor: number | null;
  bar: HealthBar;
}

export function createVitals(): Vitals {
  return { hp: MAX_HP, barTimer: 0, dyingFor: null, bar: createHealthBar() };
}

/** Alive and able to be hit. A denizen mid-death is neither -- otherwise a
 *  wide swing would keep 'killing' a corpse and restarting its fade. */
export function isTargetable(d: Denizen): boolean {
  return d.dyingFor === null && d.hp > 0;
}

/** Applies damage, shows the bar, and starts the death sequence at zero.
 *  Returns true if this blow was the killing one. */
export function damage(d: Denizen, amount: number): boolean {
  if (!isTargetable(d)) return false;

  d.hp = Math.max(0, d.hp - amount);
  d.barTimer = HEALTH_BAR_LINGER + HEALTH_BAR_FADE;

  const frac = d.hp / MAX_HP;
  d.bar.fill.scale.x = Math.max(frac, 0.0001); // never exactly 0: a zero scale is a degenerate matrix
  d.bar.fillMat.color.copy(
    frac > 0.5
      ? HEALTHY.clone().lerp(HURT, (1 - frac) * 2)
      : HURT.clone().lerp(CRITICAL, (0.5 - frac) * 2),
  );

  if (d.hp === 0) {
    d.dyingFor = 0;
    // Stop dead where it stands rather than coasting on through the death
    // animation (motion.ts would otherwise keep gliding the body).
    d.velocity.set(0, 0);
    d.moveTarget = null;
    return true;
  }
  return false;
}

const barPos = new THREE.Vector3();

/**
 * Advances the bar and death TIMERS by dt and decides whether this denizen is
 * finished dying. Pure bookkeeping -- SIMULATION, not painting -- so only the
 * authoritative host calls this (sim.ts's step()). Returns true once the
 * denizen should be taken off the board.
 */
export function advanceVitals(d: Denizen, dt: number): boolean {
  if (d.barTimer > 0) d.barTimer -= dt;
  if (d.dyingFor === null) return false;
  d.dyingFor += dt;
  return d.dyingFor - DEATH_SHAKE_TIME >= DEATH_FADE_TIME;
}

/**
 * Paints the floating bar and the death shake/fade from a denizen's CURRENT
 * hp/barTimer/dyingFor -- no dt, no decisions, purely a function of those
 * fields as they stand right now. Shared by the authoritative step() loop
 * (right after advanceVitals updates those fields) and a receiving client
 * applying network state, same split as aim.ts's applyAimPose.
 */
export function renderVitals(d: Denizen): void {
  const height = SOLDIER_HEIGHT * d.scale;

  // ── The bar ───────────────────────────────────────────────────────────
  if (d.barTimer > 0) {
    const bar = d.bar;
    barPos.set(d.pos.x, height + BAR_CLEARANCE * d.scale, d.pos.y);
    bar.group.position.copy(barPos);
    // Billboard: copy the camera's orientation outright, so the bar faces the
    // viewer from any orbit angle without any per-axis maths.
    bar.group.quaternion.copy(camera.quaternion);

    // Fade over the last stretch rather than vanishing between frames.
    const fade = d.barTimer < HEALTH_BAR_FADE ? Math.max(0, d.barTimer / HEALTH_BAR_FADE) : 1;
    // Dying denizens lose the bar early -- the corpse fade is the feedback at
    // that point, and a bar hanging over a fading body reads as still alive.
    const alive = d.dyingFor === null ? 1 : 0;
    bar.backMat.opacity = 0.75 * fade * alive;
    bar.fillMat.opacity = fade * alive;
    bar.group.visible = d.barTimer > 0 && alive === 1;
  } else if (d.bar.group.visible) {
    d.bar.group.visible = false;
  }

  // ── Dying ─────────────────────────────────────────────────────────────
  if (d.dyingFor === null) return;
  const t = d.dyingFor;

  if (t < DEATH_SHAKE_TIME) {
    const shake = deathShakeOffset(t, DEATH_SHAKE_TIME, d.scale);
    d.character.group.position.x = d.pos.x + shake.x;
    d.character.group.position.z = d.pos.y + shake.z;
    return;
  }

  // Settle back onto the exact spot, then fade out from there.
  d.character.group.position.x = d.pos.x;
  d.character.group.position.z = d.pos.y;

  const fadeT = Math.min(1, (t - DEATH_SHAKE_TIME) / DEATH_FADE_TIME);
  setOpacity(d.character.group, 1 - fadeT);
  // Sink very slightly as it goes, so it reads as collapsing rather than as a
  // texture dissolving in place.
  d.character.group.position.y = -0.1 * fadeT * d.scale;
}

/** A quick decaying shudder -- SAME shake a denizen's own death gives (just
 *  above), factored out so forest.ts's tree-fall sequence reuses this exact
 *  code rather than a look-alike copy: "something got destroyed" should read
 *  as one consistent effect, not two similar ones. Frequency is fixed;
 *  amplitude scales with `scale` (a king staggers further than a soldier, in
 *  proportion -- a big tree more than a sapling) and decays linearly to zero
 *  over `duration`. Returns an (x, z) OFFSET, not an absolute position -- the
 *  caller adds it to whatever "at rest" means for them, since a denizen
 *  shakes around its board position and a tree shakes around its planted
 *  one. */
export function deathShakeOffset(t: number, duration: number, scale: number): { x: number; z: number } {
  const decay = 1 - t / duration;
  const amp = 0.05 * scale * decay;
  return { x: Math.sin(t * 70) * amp, z: Math.sin(t * 53 + 1.7) * amp * 0.6 };
}

const applied = new Set<THREE.Material>();

/** Fades a whole figure, weapon included. Materials are per-character (see
 *  buildCharacter), so this can't bleed into anyone else -- but a Set still
 *  guards against touching a shared material twice in one pass. Exported so
 *  forest.ts's tree-fall sequence fades a felled tree with this exact
 *  function too, rather than a second copy of the same traversal. */
export function setOpacity(root: THREE.Object3D, opacity: number) {
  applied.clear();
  root.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) {
      if (applied.has(m)) continue;
      applied.add(m);
      m.transparent = true;
      m.opacity = opacity;
      m.depthWrite = opacity > 0.99;
    }
  });
}

/** Takes a denizen off the board for good: out of the scene, out of the update
 *  loop, and its own GPU resources released.
 *
 *  MATERIALS ONLY, not geometry. Every character shares one head box, one body
 *  box and one limb box (character.ts builds them once at module scope and
 *  hands the same instances to everybody) -- disposing those here would blank
 *  out all 35 other denizens the first time one died. Materials are the part
 *  that genuinely is per-character. The held weapon is the exception: its
 *  geometry IS built per equip, so that subtree can be freed outright. */
export function retire(d: Denizen) {
  scene.remove(d.character.group);
  scene.remove(d.bar.group);

  applied.clear();
  d.character.group.traverse((o) => {
    const mesh = o as THREE.Mesh;
    if (!mesh.isMesh) return;
    const mats = Array.isArray(mesh.material) ? mesh.material : [mesh.material];
    for (const m of mats) {
      if (applied.has(m)) continue;
      applied.add(m);
      m.dispose();
    }
  });
  if (d.heldObject) {
    d.heldObject.traverse((o) => {
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) mesh.geometry.dispose();
    });
  }
  d.bar.backMat.dispose();
  d.bar.fillMat.dispose();
}
