import * as THREE from 'three';
import { angleBetweenDegV, cornerDir } from '../math/geometry.ts';
import { GradientField } from '../types.ts';

// ── Contamination overlay math (pure) ────────────────────────────────────

export function computeContaminationAlpha(
  field: GradientField, agreement: Float64Array,
  dirA: THREE.Vector3, dirB: THREE.Vector3,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Float64Array {
  const { fx, fy, w, h, r } = field;
  const alpha = new Float64Array(w * h);
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, (py / h) * 2 - 1];
  for (let y = r; y < h - r; y++) {
    for (let x = r; x < w - r; x++) {
      const i = y * w + x;
      const mag = Math.hypot(fx[i], fy[i]);
      if (mag === 0) continue;
      let theta = Math.atan2(fy[i], fx[i]);
      if (theta < 0) theta += Math.PI;
      if (theta >= Math.PI) theta -= Math.PI;
      const tdx = -Math.sin(theta), tdy = Math.cos(theta);
      const [u1, v1] = toNDC(x, y);
      const [u2, v2] = toNDC(x + tdx, y + tdy);
      const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
      const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
      const n = ray1.clone().cross(ray2);
      if (n.lengthSq() < 1e-12) continue;
      n.normalize();
      const badnessA = 90 - angleBetweenDegV(n, dirA);
      const badnessB = 90 - angleBetweenDegV(n, dirB);
      const badnessAlpha = THREE.MathUtils.clamp(Math.min(badnessA, badnessB) / 45, 0, 1);
      alpha[i] = badnessAlpha * agreement[i];
    }
  }
  return alpha;
}

export function paintContaminationOverlay(alpha: Float64Array, color: readonly [number, number, number], out: Uint8Array) {
  for (let i = 0; i < alpha.length; i++) {
    const o = i * 4;
    out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2];
    out[o + 3] = Math.min(255, Math.round(alpha[i] * 255));
  }
}

export const TRUE_CONTAM_COLOR = [230, 40, 40] as const;
export const RECON_CONTAM_COLOR = [235, 150, 20] as const;

