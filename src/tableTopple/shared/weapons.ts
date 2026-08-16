import * as THREE from 'three';
import { CHARACTER_UNIT as U, HAND_LOCAL_Y } from './character.ts';
import type { Denizen } from './denizens.ts';

// Weapons a denizen can hold. Only a sword exists today, but everything here is
// keyed off a registry rather than hardcoding one: adding a bow or an axe is a
// new entry in WEAPONS and nothing else changes -- the equip path, the mode
// switch, and the aiming code never name a specific weapon.

/** How a weapon is used, which is what decides the whole attack flow. Adding
 *  a fourth style means a new branch in combat.ts; adding a fourth weapon of
 *  an EXISTING style means only a new entry in WEAPONS below. */
export type WeaponStyle = 'melee' | 'bow' | 'staff';

export interface WeaponDef {
  /** Shown in the UI. */
  label: string;
  style: WeaponStyle;
  /** Hotkey digit that selects it. */
  key: string;
  /** Builds a fresh instance. Called per equip, so two denizens holding the
   *  same weapon get independent objects (a shared one can only be in a single
   *  place in the scene graph). */
  build(): THREE.Object3D;
  /** Which fist it goes in. A bow is held out in the off hand and drawn with
   *  the other, so it needs to say so. Defaults to the right. */
  hand?: 'left' | 'right';
  /** Rotation of the grip in the hand, radians XYZ. Lets a weapon sit in the
   *  fist at its own natural angle without baking that into its geometry. */
  gripRotation?: [number, number, number];
  /** Resting aim when this weapon is first drawn: [pitch, yaw] radians for the
   *  holding arm. A bow wants a different ready pose than a sword. */
  readyAim?: [number, number];

  /** Hit points taken off per connection. At MAX_HP 10 these are roughly
   *  three or four blows to a kill. */
  damage: number;

  /** melee: the damaging span of the blade, as local Y offsets in the held
   *  object's own frame (the grip sits at 0, the blade runs down -Y). */
  edge?: { fromY: number; toY: number };
  /** melee: seconds before the same victim can be cut by the same blade
   *  again. Without it a blade resting inside someone deals damage every
   *  frame, which is 60 hits a second. */
  hitCooldown?: number;

  /** bow/staff: seconds of holding the button to reach a full charge. */
  chargeTime?: number;
  /** bow: how fast the loosed arrow travels, world units per second. */
  arrowSpeed?: number;
  /** staff: how far the cone reaches, in cells (== world units). */
  coneRange?: number;
  /** staff: half-width of the cone, radians. */
  coneHalfAngle?: number;
}

// ── Sword ─────────────────────────────────────────────────────────────────
// Modelled in the character's own unit U, in the same blocky idiom as the
// body: no bevels, no taper, just stacked boxes. Built pointing down -Y, i.e.
// continuing the line of the arm it hangs from, so raising the arm sweeps the
// blade forward like a thrust.

const STEEL = 0xc9cedd;
const BRONZE = 0x9a7f45;
const LEATHER = 0x40312a;

const BLADE_L = 16 * U, BLADE_W = 2 * U, BLADE_T = 0.8 * U;
const GUARD_W = 7 * U, GUARD_H = 1.2 * U, GUARD_T = 1.4 * U;
const GRIP_L = 4 * U, GRIP_W = 1.4 * U;
const POMMEL = 2 * U;

function buildSword(): THREE.Object3D {
  const sword = new THREE.Group();
  const mat = (color: number, roughness: number, metalness: number) =>
    new THREE.MeshStandardMaterial({ color, roughness, metalness });

  // The grip sits at the group's origin -- that's the point that goes in the
  // fist -- with the blade below it and the pommel above.
  const grip = new THREE.Mesh(new THREE.BoxGeometry(GRIP_W, GRIP_L, GRIP_W), mat(LEATHER, 0.9, 0));
  grip.position.y = 0;

  const pommel = new THREE.Mesh(new THREE.BoxGeometry(POMMEL, POMMEL, POMMEL), mat(BRONZE, 0.5, 0.6));
  pommel.position.y = GRIP_L / 2 + POMMEL / 2;

  const guard = new THREE.Mesh(new THREE.BoxGeometry(GUARD_W, GUARD_H, GUARD_T), mat(BRONZE, 0.5, 0.6));
  guard.position.y = -(GRIP_L / 2 + GUARD_H / 2);

  const blade = new THREE.Mesh(new THREE.BoxGeometry(BLADE_W, BLADE_L, BLADE_T), mat(STEEL, 0.25, 0.85));
  blade.position.y = -(GRIP_L / 2 + GUARD_H + BLADE_L / 2);

  sword.add(grip, pommel, guard, blade);
  return sword;
}

/** Where the sword's cutting span sits, in its own local frame. Derived from
 *  the same constants that build it, so a longer blade can't silently keep a
 *  short hitbox. */
const SWORD_EDGE = {
  fromY: -(GRIP_L / 2 + GUARD_H),
  toY: -(GRIP_L / 2 + GUARD_H + BLADE_L),
};

// ── Bow ───────────────────────────────────────────────────────────────────
// The limbs are a stepped approximation of an arc -- four boxes, each rotated
// a little further than the last. Curves aren't in this project's vocabulary,
// and a stepped arc reads as a bow while staying honest to the idiom.

const WOOD = 0x6b4a2a;
const STRING_COLOR = 0xd8d4c8;
const BOW_LIMB_SEGMENTS = 4;
const BOW_HALF_H = 9 * U;
const BOW_T = 0.9 * U;

/** Name the string carries so the draw animation can find it without the
 *  build function having to return a bespoke handle for every weapon. */
export const BOWSTRING_NAME = 'bowstring';

function buildBow(): THREE.Object3D {
  const bow = new THREE.Group();
  const woodMat = new THREE.MeshStandardMaterial({ color: WOOD, roughness: 0.85 });
  const stringMat = new THREE.MeshStandardMaterial({ color: STRING_COLOR, roughness: 0.6 });

  // Limbs sweep away from the archer (-Z is toward the target when held), so
  // the bow's belly faces the string.
  const seg = BOW_HALF_H / BOW_LIMB_SEGMENTS;
  for (const side of [-1, 1]) {
    for (let i = 0; i < BOW_LIMB_SEGMENTS; i++) {
      const t = (i + 0.5) / BOW_LIMB_SEGMENTS;
      const piece = new THREE.Mesh(new THREE.BoxGeometry(BOW_T, seg * 1.08, BOW_T), woodMat);
      // Each segment steps a little further along -Z, tracing the arc.
      piece.position.set(0, side * (i + 0.5) * seg, -t * t * 2.6 * U);
      // NEGATIVE of the side: the arc curves toward -Z as it climbs, and
      // Rx(+angle) tilts a segment's +Y axis toward +Z -- the wrong way. With
      // the sign flipped the segments lean along their own curve instead of
      // splaying against it.
      piece.rotation.x = -side * t * 0.62;
      bow.add(piece);
    }
  }

  // One straight string from tip to tip. Scaled and shifted at draw time.
  const string = new THREE.Mesh(
    new THREE.BoxGeometry(0.25 * U, BOW_HALF_H * 2, 0.25 * U), stringMat,
  );
  string.name = BOWSTRING_NAME;
  bow.add(string);

  return bow;
}

// ── Staff ─────────────────────────────────────────────────────────────────
// A shaft with a crystal at its head. The crystal is emissive, so it glows on
// its own rather than relying on the scene's lights -- charging brightens and
// swells it, which is the whole read of "something is building up".

const STAFF_WOOD = 0x4b3b2c;
export const STAFF_CRYSTAL_NAME = 'crystal';
const STAFF_L = 26 * U, STAFF_T = 1.1 * U;
const CRYSTAL = 3.2 * U;

function buildStaff(): THREE.Object3D {
  const staff = new THREE.Group();
  const shaftMat = new THREE.MeshStandardMaterial({ color: STAFF_WOOD, roughness: 0.9 });
  const crystalMat = new THREE.MeshStandardMaterial({
    color: 0x8fd8ff, emissive: 0x2b6fa8, emissiveIntensity: 1, roughness: 0.2, metalness: 0.1,
  });

  // Built pointing down -Y, the same convention as the sword: the weapon
  // continues the line of the arm, so the crystal LEADS toward whatever is
  // being aimed at. Building it the other way up (crystal above the fist,
  // walking-staff style) puts the business end behind the caster the moment
  // the arm comes up to aim, which is exactly backwards.
  const BUTT_Y = 3 * U; // the stub that pokes up out of the fist
  const shaft = new THREE.Mesh(new THREE.BoxGeometry(STAFF_T, STAFF_L, STAFF_T), shaftMat);
  shaft.position.y = BUTT_Y - STAFF_L / 2;
  staff.add(shaft);

  const crystal = new THREE.Mesh(new THREE.BoxGeometry(CRYSTAL, CRYSTAL * 1.4, CRYSTAL), crystalMat);
  crystal.name = STAFF_CRYSTAL_NAME;
  crystal.position.y = BUTT_Y - STAFF_L - CRYSTAL * 0.5;
  crystal.rotation.y = Math.PI / 4; // corner-on, so it reads as a gem not a crate
  staff.add(crystal);

  return staff;
}

export const WEAPONS = {
  sword: {
    label: 'sword',
    style: 'melee',
    key: '1',
    build: buildSword,
    damage: 3,
    edge: SWORD_EDGE,
    hitCooldown: 0.55,
    // Tipped slightly UP out of the line of the arm, rather than dead
    // collinear with it (which reads as a spear), and rolled a quarter turn
    // about the blade's own axis.
    //
    // The grip's rotation is in the ARM's local frame, where the blade runs
    // down -Y:
    //   X  tips the point up/down -- the plane the arm itself swings in.
    //      (Z would cock it sideways across the body instead.)
    //   Y  rolls the sword about its own length. The guard is modelled long
    //      across local X, so unrolled it lies flat left-to-right; a quarter
    //      turn stands it up so the crossguard runs vertically and the blade
    //      presents its edge rather than its flat.
    // Euler XYZ composes as Rx*Ry*Rz, so the roll is applied first, about the
    // blade's own axis, and the tip-up happens after -- which is the order
    // that keeps the roll from skewing the tip angle.
    gripRotation: [-0.25, Math.PI / 2, 0],
    // Arm forward and a little raised: the "on guard" pose you start from.
    readyAim: [-1.35, 0.15],
  },

  bow: {
    label: 'bow',
    style: 'bow',
    key: '2',
    build: buildBow,
    // Held out in the LEFT fist so the right arm is free to draw the string --
    // which is the motion that says "archer" rather than "person holding a
    // bow". combat.ts animates the off hand.
    hand: 'left',
    // Rolled upright and turned so the limbs stand vertical and the string
    // faces the archer, with the arrow running along the line of the arm.
    gripRotation: [-Math.PI / 2, 0, 0],
    readyAim: [-1.5, -0.1],
    damage: 4,
    chargeTime: 0.7,
    arrowSpeed: 34,
  },

  staff: {
    label: 'staff',
    style: 'staff',
    key: '3',
    build: buildStaff,
    gripRotation: [-0.35, 0, 0],
    readyAim: [-1.1, 0.2],
    // Lower per-hit than the others because it lands on everything in the
    // cone at once -- a well-placed blast can take four soldiers down a third
    // of their health each.
    damage: 3,
    chargeTime: 0.9,
    // 4 cells, as asked. A cell is one world unit (constants.ts).
    coneRange: 4,
    coneHalfAngle: 0.62, // ~35 degrees to each side
  },
} satisfies Record<string, WeaponDef>;

/** Weapon keys in the order they're selected with 1/2/3. */
export const WEAPON_ORDER = ['sword', 'bow', 'staff'] as const;

export type WeaponKey = keyof typeof WEAPONS;

/** WEAPONS is declared with `satisfies` so the keys stay literal ('sword' |
 *  'bow' | 'staff') rather than widening to string. The cost is that each
 *  entry keeps its own narrow shape, so optional fields another weapon uses
 *  aren't visible on it. This reads one back as the full interface. */
export function weaponDef(key: WeaponKey): WeaponDef {
  return WEAPONS[key];
}

/** The arm a weapon is (or would be) held in. Everything that poses a weapon
 *  arm goes through this rather than assuming the right, since a bow lives in
 *  the off hand. */
export function holdingArm(denizen: Denizen): THREE.Mesh {
  const hand = denizen.weapon ? weaponDef(denizen.weapon).hand ?? 'right' : 'right';
  return hand === 'left' ? denizen.character.leftArm : denizen.character.rightArm;
}

/**
 * Puts `key` in the denizen's hand, or empties both when passed null.
 *
 * The weapon is parented to the arm MESH, not the character group, so it
 * inherits the arm's swing automatically -- there is no per-frame code
 * anywhere that moves a weapon.
 */
export function equipWeapon(denizen: Denizen, key: WeaponKey | null) {
  // The arm the OLD weapon was in, which is not necessarily where the new one
  // goes -- swapping a bow for a sword moves it across the body.
  const previousArm = holdingArm(denizen);

  if (denizen.heldObject) {
    previousArm.remove(denizen.heldObject);
    // Built per equip, so this instance is ours to release. Skipping it leaks
    // a little GPU memory on every draw/sheathe.
    denizen.heldObject.traverse((o) => {
      const m = o as THREE.Mesh;
      if (!m.isMesh) return;
      m.geometry.dispose();
      (Array.isArray(m.material) ? m.material : [m.material]).forEach((mm) => mm.dispose());
    });
    denizen.heldObject = null;
  }

  denizen.weapon = key;

  // Hand BOTH arms back to the walk cycle in a clean state. The cycle only
  // ever writes rotation.x, so an aimed-and-then-sheathed arm would keep
  // whatever sideways rotation.z aiming left on it -- and the bow poses the
  // off arm too, so clearing just the holding one isn't enough.
  denizen.character.leftArm.rotation.z = 0;
  denizen.character.rightArm.rotation.z = 0;

  if (!key) return;

  const def: WeaponDef = WEAPONS[key];
  const held = def.build();
  held.position.y = HAND_LOCAL_Y;
  if (def.gripRotation) held.rotation.set(...def.gripRotation);
  // Read AFTER denizen.weapon is set, so it lands in the new weapon's hand.
  holdingArm(denizen).add(held);
  denizen.heldObject = held;

  if (def.readyAim) {
    denizen.aim.pitch = def.readyAim[0];
    denizen.aim.yaw = def.readyAim[1];
  }
}
