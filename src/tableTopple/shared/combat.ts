import * as THREE from 'three';
import { scene } from './scene.ts';
import { denizens, you, type Denizen } from './denizens.ts';
import { damage, isTargetable } from './health.ts';
import { aimPoint, handPosition, isEngaged } from './aim.ts';
import { mode, onWeaponChange } from './mode.ts';
import {
  BOWSTRING_NAME, STAFF_CRYSTAL_NAME, WEAPONS, holdingArm, type WeaponDef,
} from './weapons.ts';
import { CHARACTER_HALF_WIDTH } from './character.ts';
import { SOLDIER_HEIGHT } from './constants.ts';
import { yawFromDirection } from './frame.ts';
import { groundWedgeGeometry } from './wedge.ts';
import { send } from './net.ts';

// What attacks actually do. Three styles, one per weapon shape:
//
//   melee  the blade is a moving segment; anything it passes through is cut
//   bow    hold to draw, release to loose an arrow at the aim point
//   staff  hold to charge, release to blast a cone on the ground
//
// All three funnel into the same damage() call, and all three read their
// direction from the same reticle (aim.ts), so they can't disagree about where
// the human is pointing.
//
// ── WHY THIS FILE TALKS TO net.ts DIRECTLY ──
//
// loose()/blast() are the only two places an "attack" actually happens --
// meaning the only two places an `attackFx` message (protocol.ts) has
// anything to say. They only ever run on the authoritative host: this module
// is imported by both hosts (via sim.ts), but a receiving client never calls
// step()/updateCombat/chargeAndFire, so these functions are simply never
// invoked there (see sim.ts's header). send() is a no-op with nothing
// connected, so importing it here costs nothing on a host that never calls
// it. The alternative -- routing every attack through a callback into
// server/main.ts -- would be more indirection for a boundary this narrow.
//
// ── VISUAL vs DAMAGING, THE SAME SPLIT sim.ts USES ──
//
// A receiving client needs an arrow to fly and a blast to bloom -- exactly
// the pixels loose()/blast() already produce -- but must NEVER decide a hit
// itself (see spawnArrowFx/spawnBlastFx below): only the authoritative host's
// own damage() calls, driven by its own denizens, are real. An Arrow's
// `damages` flag is what lets one `arrows` array and one updateArrows() serve
// both a real shot and a client's replica of one without two copies of the
// flight physics.

/** Only other courts can be hurt -- a court never damages its own. */
function isEnemy(attacker: Denizen, other: Denizen): boolean {
  return other.team !== attacker.team && isTargetable(other);
}

/**
 * A denizen's body as a vertical capsule, for hit tests. Boxes are what get
 * drawn, but testing against the real box soup would mean 8 tests per figure
 * for no visible gain -- a capsule around the torso is what a player perceives
 * as "the body".
 */
function bodyRadius(d: Denizen): number {
  return CHARACTER_HALF_WIDTH * d.scale;
}
function bodyTop(d: Denizen): number {
  return SOLDIER_HEIGHT * d.scale;
}

// ── Charge ────────────────────────────────────────────────────────────────
// How long the button has been held, normalised against the weapon's own
// chargeTime. Sword ignores it; bow and staff are gated on it.

let wasEngaged = false;

/** 0..1, how far the human's weapon is drawn or charged. The UI reads it. */
export function chargeLevel(): number { return you.charge; }

// Swapping weapons mid-charge throws the charge away rather than carrying a
// bow's draw over into a staff's blast.
onWeaponChange(() => { you.charge = 0; });

// ── Melee ─────────────────────────────────────────────────────────────────
// The blade is a segment in world space, taken straight from the weapon's own
// geometry (WeaponDef.edge). Contact is tested as segment-to-capsule, and a
// per-victim cooldown stops a blade resting inside someone from dealing damage
// on all sixty frames of a second.

const lastHit = new Map<Denizen, number>();
const edgeA = new THREE.Vector3();
const edgeB = new THREE.Vector3();

/** Shortest distance from point p to segment ab, in the ground plane only.
 *  Height is handled separately, since a denizen is a vertical capsule. */
function segmentDistance2D(ax: number, az: number, bx: number, bz: number, px: number, pz: number): number {
  const dx = bx - ax, dz = bz - az;
  const lenSq = dx * dx + dz * dz;
  // A degenerate segment is just a point.
  const t = lenSq < 1e-9 ? 0 : Math.max(0, Math.min(1, ((px - ax) * dx + (pz - az) * dz) / lenSq));
  return Math.hypot(px - (ax + dx * t), pz - (az + dz * t));
}

export function meleeSweep(attacker: Denizen, def: WeaponDef, now: number) {
  if (!def.edge || !attacker.heldObject) return;

  // The blade's span, from its own local frame into the world. Reading it off
  // the object means the hitbox follows the arm's swing, the body's turn, and
  // the figure's rank scale automatically -- a king's longer sword really does
  // have longer reach.
  attacker.heldObject.updateWorldMatrix(true, false);
  edgeA.set(0, def.edge.fromY, 0).applyMatrix4(attacker.heldObject.matrixWorld);
  edgeB.set(0, def.edge.toY, 0).applyMatrix4(attacker.heldObject.matrixWorld);

  const bladeLow = Math.min(edgeA.y, edgeB.y);
  const bladeHigh = Math.max(edgeA.y, edgeB.y);

  for (const victim of denizens) {
    if (!isEnemy(attacker, victim)) continue;

    // Vertical overlap first: it's one comparison and rejects most of the
    // board when the blade is held high or low.
    if (bladeHigh < 0 || bladeLow > bodyTop(victim)) continue;

    const d = segmentDistance2D(edgeA.x, edgeA.z, edgeB.x, edgeB.z, victim.pos.x, victim.pos.y);
    if (d > bodyRadius(victim)) continue;

    const last = lastHit.get(victim) ?? -Infinity;
    if (now - last < (def.hitCooldown ?? 0.5)) continue;
    lastHit.set(victim, now);
    damage(victim, def.damage);
  }
}

// ── Bow ───────────────────────────────────────────────────────────────────

interface Arrow {
  mesh: THREE.Mesh;
  velocity: THREE.Vector3;
  /** Seconds left before it's given up on and removed. */
  life: number;
  team: number;
  damage: number;
  /** False for a client's visual-only replica (spawnArrowFx): flight and
   *  lifetime still run, but it can never call damage() -- only the
   *  authoritative host's own real arrows (loose()) may. */
  damages: boolean;
}

const arrows: Arrow[] = [];
const ARROW_LIFE = 4;
const arrowGeo = new THREE.BoxGeometry(0.04, 0.04, 0.55);
const arrowMat = new THREE.MeshStandardMaterial({ color: 0xd9cbb0, roughness: 0.8 });
const arrowHeadGeo = new THREE.BoxGeometry(0.07, 0.07, 0.12);
const arrowHeadMat = new THREE.MeshStandardMaterial({ color: 0x9aa3b2, roughness: 0.4, metalness: 0.6 });

const shotFrom = new THREE.Vector3();
const shotTo = new THREE.Vector3();

/** Builds the mesh + straight-line velocity both a real shot and a visual
 *  replica need, and pushes it into the one shared `arrows` array. Straight
 *  line, no drop: an arrow that arcs would need its launch angle solved to
 *  still land on the target point, and "it goes where it was aimed" is the
 *  promise that matters more here than ballistics. */
function spawnArrow(from: THREE.Vector3, to: THREE.Vector3, speed: number, team: number, dmg: number, damages: boolean) {
  const mesh = new THREE.Mesh(arrowGeo, arrowMat);
  const head = new THREE.Mesh(arrowHeadGeo, arrowHeadMat);
  head.position.z = 0.33;
  mesh.add(head);
  mesh.position.copy(from);
  // Point it down its own flight path: the shaft is modelled along +Z, and
  // lookAt orients +Z at the target.
  mesh.lookAt(to);
  scene.add(mesh);

  const velocity = to.clone().sub(from).normalize().multiplyScalar(speed);
  arrows.push({ mesh, velocity, life: ARROW_LIFE, team, damage: dmg, damages });
}

/** Looses a REAL arrow from `archer` at a world point, and tells every other
 *  connected host it happened. The target is passed in rather than read from
 *  the reticle, so a brain can shoot with the same call the human uses. */
export function loose(archer: Denizen, def: WeaponDef, power: number, at: THREE.Vector3) {
  handPosition(archer, shotFrom);
  shotTo.copy(at);
  const speed = (def.arrowSpeed ?? 30) * (0.45 + 0.55 * power);
  spawnArrow(shotFrom, shotTo, speed, archer.team, Math.round(def.damage * (0.4 + 0.6 * power)), true);

  send({
    type: 'attackFx', kind: 'arrow', attackerId: archer.id, team: archer.team,
    origin: { x: shotFrom.x, y: shotFrom.y, z: shotFrom.z },
    target: { x: shotTo.x, y: shotTo.y, z: shotTo.z },
    speed,
  });
}

/** A receiving client's counterpart to loose(): the same flight, none of the
 *  damage -- see this file's own header on why that split is load-bearing,
 *  not a shortcut. Called from a client's attackFx handler (client/main.ts),
 *  never from step(). */
export function spawnArrowFx(originX: number, originY: number, originZ: number, targetX: number, targetY: number, targetZ: number, speed: number, team: number) {
  spawnArrow(
    new THREE.Vector3(originX, originY, originZ),
    new THREE.Vector3(targetX, targetY, targetZ),
    speed, team, 0, false,
  );
}

const step = new THREE.Vector3();

/** Advances every arrow's flight and lifetime -- both the authoritative
 *  host's real ones AND a receiving client's visual replicas run through
 *  this same loop. Only a `damages: true` arrow ever reaches the damage()
 *  call; a replica still stops and disappears on "hit" (it just never hurts
 *  anyone), so the two look the same without a client ever touching health. */
function updateArrows(dt: number) {
  for (let i = arrows.length - 1; i >= 0; i--) {
    const a = arrows[i];
    const from = a.mesh.position.clone();
    step.copy(a.velocity).multiplyScalar(dt);
    a.mesh.position.add(step);
    a.life -= dt;

    let done = a.life <= 0 || a.mesh.position.y < 0;

    // Swept test against the segment travelled this frame, not just the end
    // point. At 34 units a second an arrow covers half a metre per frame and
    // would tunnel straight through a soldier otherwise.
    if (!done && a.damages) {
      for (const victim of denizens) {
        if (victim.team === a.team || !isTargetable(victim)) continue;
        if (a.mesh.position.y > bodyTop(victim) + 0.2) continue;
        const d = segmentDistance2D(
          from.x, from.z, a.mesh.position.x, a.mesh.position.z, victim.pos.x, victim.pos.y,
        );
        if (d > bodyRadius(victim)) continue;
        damage(victim, a.damage);
        done = true;
        break;
      }
    }

    if (done) {
      scene.remove(a.mesh);
      arrows.splice(i, 1);
    }
  }
}

/** Pulls the bowstring back and swings the free arm with it, proportional to
 *  the draw. This is the motion that reads as archery -- a bow held still is
 *  just a stick. */
function poseBow(archer: Denizen, drawn: number) {
  const bow = archer.heldObject;
  if (!bow) return;
  const string = bow.getObjectByName(BOWSTRING_NAME);
  if (string) {
    // The string bows toward the archer as it's drawn.
    string.position.z = 0.22 * drawn;
    string.scale.y = 1 - 0.12 * drawn;
  }
  // The off hand draws back past the shoulder. holdingArm() is the bow hand,
  // so the other one is the drawing hand.
  const drawArm = holdingArm(archer) === archer.character.leftArm
    ? archer.character.rightArm
    : archer.character.leftArm;
  drawArm.rotation.x = archer.aim.pitch + 0.25 + 0.5 * drawn;
  drawArm.rotation.z = -0.35 * drawn;
}

// ── Staff ─────────────────────────────────────────────────────────────────
// A ground wedge rather than a volumetric cone: the effect is judged on the
// board, where the denizens are, and a flat sector on the floor says exactly
// who is inside it. It also matches the board-game read of the whole page.

interface Blast {
  mesh: THREE.Mesh;
  age: number;
}
const blasts: Blast[] = [];
const BLAST_FADE = 0.45;

const blastAim = new THREE.Vector3();
const humanTarget = new THREE.Vector3();
/** The human's last aim point while the button was down. The shot happens on
 *  RELEASE, by which time aimTarget has already been cleared. */
const lastHumanTarget = new THREE.Vector3();

/** The wedge mesh itself, a sector already lying flat and centered on +Z (see
 *  wedge.ts), so aiming it is one rotation.y in the shared yaw convention.
 *  Split out of blast() so a receiving client can draw the same shape without
 *  ever running the damage loop above it (see this file's own header). */
function spawnBlastVisual(originX: number, originZ: number, ux: number, uz: number, range: number, halfAngle: number) {
  const geo = groundWedgeGeometry(range, halfAngle);
  const mat = new THREE.MeshBasicMaterial({
    color: 0x8fd8ff, transparent: true, opacity: 0.5, side: THREE.DoubleSide, depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  mesh.rotation.y = yawFromDirection(ux, uz);
  mesh.position.set(originX, 0.03, originZ);
  mesh.renderOrder = 2;
  scene.add(mesh);
  blasts.push({ mesh, age: 0 });
}

/** Opens a REAL cone from `caster` toward a world point, and tells every
 *  other connected host it happened. Same reasoning as loose(): the
 *  direction is an argument, not a global. */
export function blast(caster: Denizen, def: WeaponDef, power: number, at: THREE.Vector3) {
  const range = (def.coneRange ?? 4) * (0.5 + 0.5 * power);
  const halfAngle = def.coneHalfAngle ?? 0.6;

  blastAim.copy(at);
  const dirX = blastAim.x - caster.pos.x, dirZ = blastAim.z - caster.pos.y;
  const len = Math.hypot(dirX, dirZ) || 1;
  const ux = dirX / len, uz = dirZ / len;

  for (const victim of denizens) {
    if (!isEnemy(caster, victim)) continue;
    const vx = victim.pos.x - caster.pos.x, vz = victim.pos.y - caster.pos.y;
    const dist = Math.hypot(vx, vz);
    // Generous by the victim's own radius, so a soldier half inside the wedge
    // still takes it -- the alternative is a cone that visibly clips people
    // without hurting them.
    if (dist > range + bodyRadius(victim)) continue;
    if (dist > 1e-4) {
      const cos = (vx * ux + vz * uz) / dist;
      if (cos < Math.cos(halfAngle)) continue;
    }
    damage(victim, Math.max(1, Math.round(def.damage * (0.4 + 0.6 * power))));
  }

  spawnBlastVisual(caster.pos.x, caster.pos.y, ux, uz, range, halfAngle);

  send({
    type: 'attackFx', kind: 'blast', attackerId: caster.id, team: caster.team,
    origin: { x: caster.pos.x, y: caster.pos.y },
    target: { x: blastAim.x, y: blastAim.z },
    range, halfAngle,
  });
}

/** A receiving client's counterpart to blast(): the same wedge, none of the
 *  damage. Recomputes the aim direction from origin/target exactly as
 *  blast() does, rather than sending ux/uz over the wire, so the wire shape
 *  stays "what happened" (a world point) rather than a derived value. */
export function spawnBlastFx(originX: number, originZ: number, targetX: number, targetZ: number, range: number, halfAngle: number) {
  const dirX = targetX - originX, dirZ = targetZ - originZ;
  const len = Math.hypot(dirX, dirZ) || 1;
  spawnBlastVisual(originX, originZ, dirX / len, dirZ / len, range, halfAngle);
}

function updateBlasts(dt: number) {
  for (let i = blasts.length - 1; i >= 0; i--) {
    const b = blasts[i];
    b.age += dt;
    const t = b.age / BLAST_FADE;
    if (t >= 1) {
      scene.remove(b.mesh);
      b.mesh.geometry.dispose();
      (b.mesh.material as THREE.Material).dispose();
      blasts.splice(i, 1);
      continue;
    }
    (b.mesh.material as THREE.MeshBasicMaterial).opacity = 0.5 * (1 - t);
    // Swells slightly as it fades, which reads as a shockwave leaving.
    b.mesh.scale.setScalar(1 + 0.12 * t);
  }
}

/**
 * Advances every arrow and blast -- flight, fade, lifetime, removal -- for
 * BOTH a real attack and a receiving client's spawnArrowFx/spawnBlastFx
 * replica of one. Every host that can hold an arrow or a blast has to call
 * this every frame, not just the one that spawned it: a wedge left in the
 * `blasts` array with nothing ever advancing its age stays at full opacity
 * forever, which is exactly the "blue cone never clears on the client" bug
 * this function exists to make impossible to reintroduce -- see updateCombat
 * (desktop, via step()) and client/overlay.ts's renderOverlay (phone).
 */
export function updateAttackVisuals(dt: number) {
  updateArrows(dt);
  updateBlasts(dt);
}

/** Brightens and swells the staff's crystal with the charge. */
function poseStaff(caster: Denizen, charged: number) {
  const crystal = caster.heldObject?.getObjectByName(STAFF_CRYSTAL_NAME) as THREE.Mesh | undefined;
  if (!crystal) return;
  crystal.scale.setScalar(1 + 0.7 * charged);
  const mat = crystal.material as THREE.MeshStandardMaterial;
  mat.emissiveIntensity = 1 + 4 * charged;
}

// ── Per-frame ─────────────────────────────────────────────────────────────

let elapsed = 0;

/**
 * Drives the human's attacks. Edge-detects the button rather than taking
 * events, so the release and the aim point are read in the same frame and
 * can't drift apart.
 */
export function updateCombat(dt: number) {
  elapsed += dt;
  updateAttackVisuals(dt);

  const weapon = you.weapon;
  if (!weapon || mode !== 'fight') {
    you.charge = 0;
    you.aimTarget = null;
    wasEngaged = false;
    return;
  }
  const def: WeaponDef = WEAPONS[weapon];
  const engaged = isEngaged();

  // The reticle is the human's aim target; a brain sets its own (ai.ts). Both
  // then flow through the same posing and firing code below.
  you.aimTarget = engaged ? aimPoint(humanTarget) : null;

  if (def.style === 'melee') {
    // Continuous: the blade cuts whatever it sweeps through while held. The
    // human's swing is their own arm movement, so there's no swing timer --
    // unlike a brain, which has to be given one (ai.ts).
    if (engaged) meleeSweep(you, def, elapsed);
  } else {
    const chargeTime = def.chargeTime ?? 0.6;
    if (engaged) {
      you.charge = Math.min(1, you.charge + dt / chargeTime);
    } else if (wasEngaged) {
      // Released: this is the shot. Fired at the LAST aim point, captured
      // before aimTarget was cleared on release.
      if (def.style === 'bow') loose(you, def, you.charge, lastHumanTarget);
      else blast(you, def, you.charge, lastHumanTarget);
      you.charge = 0;
    }
  }
  if (you.aimTarget) lastHumanTarget.copy(you.aimTarget);

  if (def.style === 'bow') poseBow(you, engaged ? you.charge : 0);
  if (def.style === 'staff') poseStaff(you, engaged ? you.charge : 0);

  wasEngaged = engaged;
}

/** Runs a weapon's charge/fire cycle for a non-human denizen. Returns true on
 *  the frame it actually fires, so a brain can start its cooldown. */
export function chargeAndFire(d: Denizen, def: WeaponDef, dt: number, at: THREE.Vector3): boolean {
  const chargeTime = def.chargeTime ?? 0.6;
  d.charge = Math.min(1, d.charge + dt / chargeTime);
  if (def.style === 'bow') poseBow(d, d.charge);
  if (def.style === 'staff') poseStaff(d, d.charge);
  if (d.charge < 1) return false;

  if (def.style === 'bow') loose(d, def, d.charge, at);
  else blast(d, def, d.charge, at);
  d.charge = 0;
  if (def.style === 'bow') poseBow(d, 0);
  if (def.style === 'staff') poseStaff(d, 0);
  return true;
}
