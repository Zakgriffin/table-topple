import * as THREE from 'three';
import { denizens, type Denizen } from './denizens.ts';
import { isTargetable } from './health.ts';
import { equipWeapon, weaponDef, type WeaponKey } from './weapons.ts';
import { chargeAndFire, meleeSweep } from './combat.ts';
import { advance, arrivalSpeed } from './motion.ts';
import { turnToward } from './animation.ts';
import { SWING_TIME } from './aim.ts';
import { yawFromDirection } from './frame.ts';
import { CHARACTER_HALF_WIDTH } from './character.ts';
import { BOARD_SIZE, SOLDIER_HEIGHT, WALK_ACCEL_RATE, WALK_DECEL_RATE, WALK_SPEED } from './constants.ts';
import type { Rank } from './ranks.ts';

// Simple fighter behaviour, switched on from the button in the corner.
//
// Every brain does the same three things -- pick the nearest enemy, get to the
// range its weapon wants, attack when it's there -- and the only per-weapon
// part is what "the range it wants" means and what firing looks like:
//
//   melee  close right in, then swing; the blade's own hitbox does the rest
//   bow    hold at a distance, draw, loose; back off if something closes
//   staff  get just inside cone range, charge, blast
//
// Deliberately dumb: no formations, no target priority, no retreating, no
// avoidance of friendly fire (there isn't any -- a court can't hurt its own).
// Enough to watch a battle happen and see whether the weapons feel right
// against each other.

let battleOn = false;
export function isBattleOn(): boolean { return battleOn; }

/** Which weapon each rank carries into a fight. A placeholder assignment, but
 *  a deliberate one: it puts a mix on every side (7 blades, 2 bows, 2 staves
 *  per court) so all three behaviours are on show at once. */
const RANK_WEAPON: Record<Rank, WeaponKey> = {
  king: 'sword',
  constable: 'staff',
  captain: 'bow',
  soldier: 'sword',
};

/** Per-brain scratch state. Kept beside the denizen rather than on it, because
 *  it's meaningless outside a battle and shouldn't outlive one. */
interface Brain {
  target: Denizen | null;
  /** Seconds until this denizen may start another attack. */
  cooldown: number;
  /** Seconds until it reconsiders who to fight. */
  retargetIn: number;
}
const brains = new Map<Denizen, Brain>();

/** How long between attacks, per style. Melee is quickest; the ranged weapons
 *  pay for their reach with a slower cycle. */
const COOLDOWN = { melee: 0.85, bow: 1.5, staff: 2.6 };

const aimAt = new THREE.Vector3();
const desired = new THREE.Vector2();
const toTarget = new THREE.Vector2();

/** The point on a denizen a shot should be aimed at: mid-torso, not the feet,
 *  so arrows fly flat rather than into the ground. */
function chestOf(d: Denizen, out: THREE.Vector3): THREE.Vector3 {
  return out.set(d.pos.x, SOLDIER_HEIGHT * d.scale * 0.6, d.pos.y);
}

/** How close this denizen wants to be to whatever it's fighting. */
function preferredRange(d: Denizen): number {
  const def = weaponDef(d.weapon!);
  if (def.style === 'bow') return 9;
  // The staff wants to be just inside its cone, not at the very edge, so a
  // target that shuffles slightly doesn't fall out of it mid-charge.
  if (def.style === 'staff') return (def.coneRange ?? 4) * 0.7;
  // Melee: close enough that the blade's own span reaches, allowing for both
  // bodies. Derived from the weapon and the figure rather than guessed, so a
  // king with a king-sized sword correctly stops further out.
  const reach = Math.abs(def.edge!.toY - def.edge!.fromY) * d.scale;
  return reach * 0.8 + CHARACTER_HALF_WIDTH * d.scale;
}

function nearestEnemy(d: Denizen): Denizen | null {
  let best: Denizen | null = null;
  let bestDist = Infinity;
  for (const other of denizens) {
    if (other.team === d.team || !isTargetable(other)) continue;
    // The red king used to be left out of the AI's target list on purpose,
    // back when he was the one denizen under manual WASD control and dying
    // would have stranded that control on a removed body. WASD is gone
    // (sim.ts), so there's nothing left to strand -- he's targetable like
    // anyone else now.
    const dist = d.pos.distanceToSquared(other.pos);
    if (dist < bestDist) { bestDist = dist; best = other; }
  }
  return best;
}

/** Arms or disarms every brain, `you` included -- he's an ordinary denizen
 *  now, not a human keeping a hand-picked weapon (see denizens.ts's own
 *  comment on `you`). Toggling off puts the board back to rest. */
export function setBattle(on: boolean) {
  battleOn = on;
  for (const d of denizens) {
    if (on) {
      equipWeapon(d, RANK_WEAPON[d.rank]);
      brains.set(d, { target: null, cooldown: Math.random() * 0.8, retargetIn: 0 });
    } else {
      equipWeapon(d, null);
      brains.delete(d);
      d.aimTarget = null;
      d.charge = 0;
      d.swingFor = null;
      d.moveTarget = null;
      d.velocity.set(0, 0);
    }
  }
}

export function updateAI(dt: number) {
  for (const d of denizens) {
    // A swing runs to completion whatever else is happening, and the blade
    // cuts for its whole travel -- that's what makes an AI strike use the same
    // hitbox the human's does rather than a special "deal damage now" call.
    if (d.swingFor !== null) {
      d.swingFor += dt;
      meleeSweep(d, weaponDef(d.weapon!), performance.now() / 1000);
      if (d.swingFor > SWING_TIME) d.swingFor = null;
    }

    const brain = brains.get(d);
    if (!brain || !battleOn || !d.weapon || d.dyingFor !== null) continue;

    brain.cooldown -= dt;
    brain.retargetIn -= dt;

    // Re-pick only every so often, and whenever the current target is gone.
    // Choosing afresh every frame makes a denizen dither between two enemies
    // at nearly equal distance instead of committing to one.
    if (brain.retargetIn <= 0 || !brain.target || !isTargetable(brain.target)) {
      brain.target = nearestEnemy(d);
      brain.retargetIn = 0.4 + Math.random() * 0.3;
    }
    const target = brain.target;
    if (!target) { d.aimTarget = null; d.velocity.set(0, 0); continue; }

    const def = weaponDef(d.weapon);
    toTarget.set(target.pos.x - d.pos.x, target.pos.y - d.pos.y);
    const dist = toTarget.length();
    const want = preferredRange(d);

    // Always face and track what it's fighting, moving or not.
    chestOf(target, aimAt);
    d.aimTarget = aimAt.clone();

    // ── Feet ────────────────────────────────────────────────────────────
    desired.set(0, 0);
    let rate = WALK_DECEL_RATE;
    // A dead band around the preferred range, so a denizen that has arrived
    // stands and fights instead of jittering back and forth across the line.
    const band = 0.6;
    if (dist > want + band) {
      // Close the gap, easing in so it doesn't overshoot into the target.
      const speed = arrivalSpeed(dist - want, WALK_SPEED, WALK_DECEL_RATE);
      desired.set(toTarget.x / dist * speed, toTarget.y / dist * speed);
      rate = WALK_ACCEL_RATE;
    } else if (dist < want - band) {
      // Too close: an archer with someone in its face backs away rather than
      // shooting past them.
      desired.set(-toTarget.x / dist * WALK_SPEED * 0.7, -toTarget.y / dist * WALK_SPEED * 0.7);
      rate = WALK_ACCEL_RATE;
    }
    advance(d.pos, d.velocity, desired, rate, dt, BOARD_SIZE / 2 - CHARACTER_HALF_WIDTH * d.scale);
    // Turn to face the target, not the direction of travel -- backing away
    // should be a retreat, keeping the weapon up, not a turn and a run.
    turnToward(d, yawFromDirection(toTarget.x, toTarget.y), dt);

    // ── Attack ──────────────────────────────────────────────────────────
    const inRange = dist <= want + band;
    if (!inRange || brain.cooldown > 0) {
      // Let a part-built charge decay rather than holding it indefinitely,
      // so an archer that gets rushed mid-draw doesn't fire the instant it
      // gets clear again.
      if (def.style !== 'melee') d.charge = Math.max(0, d.charge - dt);
      continue;
    }

    if (def.style === 'melee') {
      if (d.swingFor === null) {
        d.swingFor = 0;
        brain.cooldown = COOLDOWN.melee;
      }
    } else if (chargeAndFire(d, def, dt, aimAt)) {
      brain.cooldown = def.style === 'bow' ? COOLDOWN.bow : COOLDOWN.staff;
    }
  }
}

// ── UI ────────────────────────────────────────────────────────────────────

/** Adopts the battle switch, if the host page has one. A host without it can
 *  still start a fight -- setBattle() above is the actual control, and the
 *  button is only one way to reach it. */
export function wireBattleButton() {
  const button = document.getElementById('battleToggle') as HTMLButtonElement | null;
  if (!button) return;
  button.addEventListener('click', () => {
    setBattle(!battleOn);
    button.classList.toggle('active', battleOn);
    button.textContent = battleOn ? 'battle: on' : 'battle: off';
    button.blur();
  });
}
