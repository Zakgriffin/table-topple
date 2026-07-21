// WGSL source for the GPU port of pipeline/decodeGrid.ts's stage 1 (the
// ray-cast+project half of castAndBucketProjectedSamples) -- see
// projectSamples.ts's header for why stage 2 (bucket accumulation) stays on
// CPU for now.
//
// One thread per SCREEN pixel, dense output (valid=0 for pixels that miss
// the grazing-angle cutoff) -- every thread owns its own unique output
// slot, so (like every "map" stage ported so far) no atomics are needed
// here at all.

export const PROJECT_SAMPLES_WGSL = /* wgsl */ `
struct Uniforms {
  w: u32, h: u32, pad0: u32, pad1: u32,
  minGrazingCos: f32, distance: f32, vFovRad: f32, aspect: f32,
  quat: vec4<f32>,
  drowX: f32, drowY: f32, drowZ: f32, pad2: f32,
  dcolX: f32, dcolY: f32, dcolZ: f32, pad3: f32,
  normalX: f32, normalY: f32, normalZ: f32, pad4: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> fx: array<f32>;
@group(0) @binding(2) var<storage, read> fy: array<f32>;
@group(0) @binding(3) var<storage, read_write> sampleOut: array<vec4<f32>>; // (u, v, cx, cy)
@group(0) @binding(4) var<storage, read_write> validOut: array<u32>;

fn rotateByQuat(v: vec3<f32>, q: vec4<f32>) -> vec3<f32> {
  let t = 2.0 * cross(q.xyz, v);
  return v + q.w * t + cross(q.xyz, t);
}
fn cornerDir(ndcU: f32, ndcV: f32, quat: vec4<f32>, vFovRad: f32, aspect: f32) -> vec3<f32> {
  let halfV = vFovRad * 0.5;
  let yc = tan(halfV) * ndcV;
  let xc = tan(halfV) * aspect * ndcU;
  let local = normalize(vec3<f32>(xc, yc, -1.0));
  return rotateByQuat(local, quat);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= u.w || y >= u.h) { return; }
  let i = y * u.w + x;

  let drow = vec3<f32>(u.drowX, u.drowY, u.drowZ);
  let dcol = vec3<f32>(u.dcolX, u.dcolY, u.dcolZ);
  let normal = vec3<f32>(u.normalX, u.normalY, u.normalZ);

  let fw = f32(u.w); let fh = f32(u.h);
  let ndcU = (f32(x) / fw) * 2.0 - 1.0;
  let ndcV = (f32(y) / fh) * 2.0 - 1.0;
  let rayDir = cornerDir(ndcU, ndcV, u.quat, u.vFovRad, u.aspect);
  let denom = dot(rayDir, normal);
  if (denom >= -u.minGrazingCos) { validOut[i] = 0u; sampleOut[i] = vec4<f32>(0.0); return; }
  let t = -u.distance / denom;
  let hit = rayDir * t;
  let uu = dot(hit, drow); let vv = dot(hit, dcol);
  validOut[i] = 1u;

  var cx = 0.0; var cy = 0.0;
  let fxi = fx[i]; let fyi = fy[i];
  let mag = length(vec2<f32>(fxi, fyi));
  if (mag > 0.0) {
    let theta = atan2(fyi, fxi);
    let tdx = -sin(theta); let tdy = cos(theta);
    let ndcU2 = ((f32(x) + tdx) / fw) * 2.0 - 1.0;
    let ndcV2 = ((f32(y) + tdy) / fh) * 2.0 - 1.0;
    let rayDir2 = cornerDir(ndcU2, ndcV2, u.quat, u.vFovRad, u.aspect);
    let denom2 = dot(rayDir2, normal);
    if (denom2 < -u.minGrazingCos) {
      let t2 = -u.distance / denom2;
      let hit2 = rayDir2 * t2;
      let u2 = dot(hit2, drow); let v2 = dot(hit2, dcol);
      let du = u2 - uu; let dv = v2 - vv;
      if (length(vec2<f32>(du, dv)) > 1e-9) {
        let phiUV = atan2(dv, du);
        cx = -mag * cos(2.0 * phiUV);
        cy = -mag * sin(2.0 * phiUV);
      }
    }
  }
  sampleOut[i] = vec4<f32>(uu, vv, cx, cy);
}
`;
