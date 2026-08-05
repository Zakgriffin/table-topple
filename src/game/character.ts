import * as THREE from 'three';
import { SOLDIER_HEIGHT } from './constants.ts';

// A blocky humanoid: head, body, two arms, two legs. Six boxes plus two eye
// chips, assembled into a Group whose origin is between the feet -- so the
// caller positions it by dropping the origin on the ground, no half-height
// offset to remember (which is what the cylinder needed).
//
// Proportions are the classic Minecraft-skin ratios, in a unit U where the
// figure is 32U tall: head 8, body 8x12x4, each limb 4x12x4. Those numbers
// are load-bearing -- the reason a blocky figure reads as a PERSON and not a
// stack of crates is the 1:4 head-to-height ratio and limbs exactly half the
// body's width. U is derived from SOLDIER_HEIGHT so the whole character scales
// with that one constant.
const U = SOLDIER_HEIGHT / 32;

const HEAD = 8 * U;
const BODY_W = 8 * U, BODY_H = 12 * U, BODY_D = 4 * U;
const LIMB_W = 4 * U, LIMB_H = 12 * U, LIMB_D = 4 * U;

// Widest point: body plus an arm on each side. Used for board-edge clamping.
export const CHARACTER_HALF_WIDTH = (BODY_W + 2 * LIMB_W) / 2;
// Shoulder-to-hand / hip-to-foot distance. animation.ts derives the walk
// cycle's stride length from it, so the feet don't skate along the ground.
export const CHARACTER_LIMB_LENGTH = LIMB_H;
// The character's base unit, exported so weapons can be modelled in the same
// grid as the body instead of in magic decimals that silently stop matching
// if SOLDIER_HEIGHT ever changes.
export const CHARACTER_UNIT = U;
/** Where the hand is, in an arm's LOCAL space. The arm's origin is its
 *  shoulder and it hangs down -Y, so the fist is one limb-length below. Held
 *  items parent here and inherit the arm's swing for free. */
export const HAND_LOCAL_Y = -LIMB_H;

// Stacked from the ground up: legs (12U) then body (12U) then head (8U).
const BODY_Y = LIMB_H + BODY_H / 2;
const HEAD_Y = LIMB_H + BODY_H + HEAD / 2;
// Limbs hang FROM these heights rather than being centered on them -- see the
// geometry translate below. Shoulders at the top of the torso, hips at the top
// of the legs (which is the torso's bottom).
const SHOULDER_Y = LIMB_H + BODY_H;
const HIP_Y = LIMB_H;

// One geometry per part shape, shared across every character ever built --
// the parts differ only in material color and transform.
const headGeo = new THREE.BoxGeometry(HEAD, HEAD, HEAD);
const bodyGeo = new THREE.BoxGeometry(BODY_W, BODY_H, BODY_D);
const limbGeo = new THREE.BoxGeometry(LIMB_W, LIMB_H, LIMB_D);
// Shift the limb's vertices so its origin sits at the TOP face's center
// instead of the box's middle. That makes rotation.x on a limb a swing about
// the shoulder/hip, which is the joint an arm or leg actually has -- rotating
// the default center-origin box instead scissors the limb around its own
// midpoint, feet and all. Done to the geometry once rather than by nesting
// every limb in a pivot Group: same result, half the scene-graph nodes.
limbGeo.translate(0, -LIMB_H / 2, 0);
const eyeGeo = new THREE.BoxGeometry(1.5 * U, 1.5 * U, 0.5 * U);

// ── Crown ─────────────────────────────────────────────────────────────────
// A band around the head plus points rising off it. Sized to the head rather
// than to any particular rank's scale, so it rides correctly on a figure of
// any size (a king's whole body is scaled as one group).
const CROWN_BAND_H = 1.6 * U;
const CROWN_OVERHANG = 0.5 * U; // how far it sits proud of the skull
const crownBandGeo = new THREE.BoxGeometry(HEAD + CROWN_OVERHANG, CROWN_BAND_H, HEAD + CROWN_OVERHANG);
const crownPointGeo = new THREE.BoxGeometry(1.3 * U, 1.7 * U, 1.3 * U);
const crownCornerGeo = new THREE.BoxGeometry(1.5 * U, 2.4 * U, 1.5 * U);
// Gold, and deliberately NOT tinted toward the team color: every king wears
// the same crown, so the crown reads as "this one is the king" while the
// body's color keeps saying which team.
const CROWN_GOLD = 0xd9ac33;

/** Named handles on the parts, so animation later doesn't re-find them by
 *  digging through Group.children. */
export interface Character {
  group: THREE.Group;
  head: THREE.Mesh;
  body: THREE.Mesh;
  leftArm: THREE.Mesh;
  rightArm: THREE.Mesh;
  leftLeg: THREE.Mesh;
  rightLeg: THREE.Mesh;
}

/**
 * Builds a character in the given color. It faces +Z, matching the convention
 * that a Group's rotation.y of 0 looks down +Z; a caller that wants it facing
 * elsewhere sets group.rotation.y.
 */
export function buildCharacter(color: number): Character {
  // Three shades off the one team color: the head reads as skin/helmet, the
  // limbs sit a step darker so arms stay visible against the body from any
  // angle. A single flat color makes the silhouette collapse into a blob under
  // the scene's soft lighting -- there are no shadows here to separate parts.
  const base = new THREE.Color(color);
  const bodyMat = new THREE.MeshStandardMaterial({ color: base, roughness: 0.6, metalness: 0.05 });
  const headMat = new THREE.MeshStandardMaterial({
    color: base.clone().lerp(new THREE.Color(0xffffff), 0.25), roughness: 0.6, metalness: 0.05,
  });
  const limbMat = new THREE.MeshStandardMaterial({
    color: base.clone().multiplyScalar(0.62), roughness: 0.6, metalness: 0.05,
  });
  const eyeMat = new THREE.MeshStandardMaterial({ color: 0x111118, roughness: 0.4 });

  const group = new THREE.Group();

  const head = new THREE.Mesh(headGeo, headMat);
  head.position.y = HEAD_Y;

  // Eyes: the only thing that says which way the character is facing. Sunk a
  // hair into the head's +Z face (0.5U deep box, placed at half the head plus
  // a sliver) rather than sitting proud of it, so they can't z-fight.
  for (const sx of [-1, 1]) {
    const eye = new THREE.Mesh(eyeGeo, eyeMat);
    eye.position.set(sx * 2 * U, 1 * U, HEAD / 2 - 0.1 * U);
    head.add(eye);
  }

  const body = new THREE.Mesh(bodyGeo, bodyMat);
  body.position.y = BODY_Y;

  // Both hang from their joint (the geometry's origin is its top face), so
  // these positions are the shoulder and hip, not the limb's center.
  const arm = (sx: number) => {
    const m = new THREE.Mesh(limbGeo, limbMat);
    // Flush against the body's side: half the body plus half the arm.
    m.position.set(sx * (BODY_W + LIMB_W) / 2, SHOULDER_Y, 0);
    return m;
  };
  const leg = (sx: number) => {
    const m = new THREE.Mesh(limbGeo, limbMat);
    m.position.set(sx * LIMB_W / 2, HIP_Y, 0);
    return m;
  };

  // Left/right are the CHARACTER's, not the viewer's. Facing +Z with +Y up in
  // a right-handed frame, its right hand points down -X.
  const leftArm = arm(1), rightArm = arm(-1);
  const leftLeg = leg(1), rightLeg = leg(-1);

  group.add(head, body, leftArm, rightArm, leftLeg, rightLeg);
  return { group, head, body, leftArm, rightArm, leftLeg, rightLeg };
}

/**
 * Crowns a character. Parented to the HEAD, not the group, so it stays put if
 * the head is ever animated (a nod, a death topple) instead of hovering where
 * the head used to be.
 */
export function addCrown(character: Character) {
  const gold = new THREE.MeshStandardMaterial({ color: CROWN_GOLD, roughness: 0.35, metalness: 0.75 });
  const crown = new THREE.Group();

  const band = new THREE.Mesh(crownBandGeo, gold);
  crown.add(band);

  // Points around the band: taller ones at the four corners, shorter ones at
  // the middle of each side. The alternation is what makes a ring of boxes
  // read as a crown rather than as a chunky hat.
  const r = (HEAD + CROWN_OVERHANG) / 2;
  const inset = 0.75 * U; // pull the points in so they sit ON the band, not off its lip
  for (const [dx, dz] of [[-1, -1], [1, -1], [1, 1], [-1, 1]] as const) {
    const p = new THREE.Mesh(crownCornerGeo, gold);
    p.position.set(dx * (r - inset), CROWN_BAND_H / 2 + 2.4 * U / 2, dz * (r - inset));
    crown.add(p);
  }
  for (const [dx, dz] of [[0, -1], [1, 0], [0, 1], [-1, 0]] as const) {
    const p = new THREE.Mesh(crownPointGeo, gold);
    p.position.set(dx * (r - inset), CROWN_BAND_H / 2 + 1.7 * U / 2, dz * (r - inset));
    crown.add(p);
  }

  // Sits on the crown of the skull, overlapping it slightly so there's no
  // floating gap visible from a low camera angle.
  crown.position.y = HEAD / 2 - CROWN_BAND_H / 2 + 0.3 * U;
  character.head.add(crown);
}
