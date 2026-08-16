import * as THREE from 'three';
import { angleBetweenDegV, cornerDir } from '../../shared/math/geometry.ts';
import { type GradientField } from '../../shared/types.ts';

// ── Contamination overlay math (pure) ────────────────────────────────────
//
// `weight` is a per-pixel [0,1] multiplier on the geometric badness, i.e. "how
// much does this pixel count". Its only supplier is computeGradientMagnitudeField
// (frame-normalized raw gradient magnitude) -- it stays a parameter rather than
// being computed inline because the badness math below is genuinely independent
// of the weighting, and this is where a spatially-aggregated "agreement" field
// used to be swapped in.
export function computeContaminationAlpha(
  field: GradientField, weight: Float64Array,
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
      alpha[i] = badnessAlpha * weight[i];
    }
  }
  return alpha;
}

// `alpha` is in the pipeline's TOP-DOWN order (it is computed over the
// pipeline's own field); `out` is a flipY=false preview texture and is
// bottom-up. Same rule as paintTopGradientOverlay: the flip goes on the way OUT
// to display. See overlays/pipelineField.ts.
export function paintContaminationOverlay(
  alpha: Float64Array, color: readonly [number, number, number], out: Uint8Array, w: number, h: number,
) {
  for (let y = 0; y < h; y++) {
    const src = y * w, dst = (h - 1 - y) * w;
    for (let x = 0; x < w; x++) {
      const o = (dst + x) * 4;
      out[o] = color[0]; out[o + 1] = color[1]; out[o + 2] = color[2];
      out[o + 3] = Math.min(255, Math.round(alpha[src + x] * 255));
    }
  }
}

export const TRUE_CONTAM_COLOR = [230, 40, 40] as const;
export const RECON_CONTAM_COLOR = [235, 150, 20] as const;

