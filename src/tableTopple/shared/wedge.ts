import * as THREE from 'three';

/**
 * A circular sector lying flat in the ground plane, centered on +Z, with its
 * apex at the origin -- the footprint of a cone attack.
 *
 * Centered on +Z specifically so that aiming it is a plain `rotation.y = yaw`
 * in the project's own heading convention (frame.ts: yaw 0 faces +Z), which
 * `yawFromDirection` already inverts correctly. Nothing else about the mesh
 * needs rotating.
 *
 * This used to be done as rotation.x to lay it flat plus rotation.z to aim it,
 * on the same object. Euler 'XYZ' composes as Rx*Ry*Rz, so the aim was applied
 * BEFORE the flattening, inside a plane whose handedness inverts once flat:
 * the wedge came out mirrored, and turning the caster clockwise swept it
 * counter-clockwise. Flattening the GEOMETRY here, once, removes the
 * composition question entirely.
 *
 * CircleGeometry lays its sector in the XY plane measuring from +X, and
 * rotateX(-90) sends local +Y to world -Z -- so centering the sector on local
 * -Y is what lands it on +Z once flattened.
 */
export function groundWedgeGeometry(range: number, halfAngle: number, segments = 24): THREE.BufferGeometry {
  const geo = new THREE.CircleGeometry(range, segments, -Math.PI / 2 - halfAngle, halfAngle * 2);
  geo.rotateX(-Math.PI / 2);
  return geo;
}
