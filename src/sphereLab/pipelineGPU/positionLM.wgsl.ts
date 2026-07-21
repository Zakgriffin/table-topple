// WGSL source for the GPU half of Phase 3 (photometric position LM) -- the
// GPU-resident counterpart to pipeline/positionLM.ts's refineOrientationAndPositionLM.
// Unlike vote generation, the LM *loop* itself (accept/reject, lambda
// damping, the tiny 5x5 linear solve) stays on CPU -- it's inherently
// sequential across iterations (iteration N+1's candidate pose depends on
// N's outcome) and operates on ~20 numbers, nothing GPU parallelism helps
// with. What this shader does per iteration is the expensive, embarrassingly
// parallel part: for every photometric sample, compute the residual under
// the CURRENT pose AND all 5 finite-difference-perturbed poses (3 rotation
// axes + 2 position axes) in one dispatch, so the CPU side only needs ONE
// readback per iteration instead of 6.
//
// One approximation worth flagging: the CPU version COMPACTS residualsFor's
// output (skips samples that fail the grazing-angle check, so the returned
// array length varies per pose and per-column Jacobian differences use
// whichever array is shorter, positionally). This shader instead always
// writes one result per sample INDEX, with a validity flag for samples that
// fail the grazing check at that specific pose -- the CPU-side reduction
// (see positionLM.ts) only sums a Jacobian column at sample i if BOTH the
// baseline AND that column's perturbed evaluation are valid at index i, i.e.
// per-index alignment rather than the CPU's post-compaction positional
// alignment. Given the perturbations are tiny (EPS_ROT=1e-5, EPS_POS=1e-3),
// a sample's grazing-check validity essentially never flips between the
// baseline and a perturbed pose, so this should be numerically
// indistinguishable from the CPU version's approximation in practice --
// verified on the saved-capture.json fixture, see this session's chat.
export const PHOTOMETRIC_RESIDUALS_WGSL = /* wgsl */ `
struct P3Uniforms {
  w: f32, h: f32, sampleCount: u32, torusR: i32,
  torusC: i32, distance: f32, vFovRad: f32, aspect: f32,
  minGrazingCos: f32, epsRot: f32, epsPos: f32, pad0: f32,
  wx0: f32, wz0: f32, pad1: f32, pad2: f32,
  q: vec4<f32>,
  camQuat: vec4<f32>,
  drow0: vec4<f32>,
  dcol0: vec4<f32>,
  dnormal0: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: P3Uniforms;
@group(0) @binding(1) var<storage, read> samplesPx: array<f32>;
@group(0) @binding(2) var<storage, read> samplesPy: array<f32>;
@group(0) @binding(3) var<storage, read> samplesObs: array<f32>;
@group(0) @binding(4) var<storage, read> torusBuf: array<f32>;
@group(0) @binding(5) var<storage, read_write> outResiduals: array<vec2<f32>>; // (residual, validFlag) x 6 per sample

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
fn quatMultiply(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
  return vec4<f32>(
    a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
    a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
    a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
  );
}
fn axisAngleQuat(axis: vec3<f32>, angle: f32) -> vec4<f32> {
  let half = angle * 0.5;
  let s = sin(half);
  return vec4<f32>(axis.x * s, axis.y * s, axis.z * s, cos(half));
}

fn torusBrightness(row: i32, col: i32) -> f32 {
  let r = ((row % u.torusR) + u.torusR) % u.torusR;
  let c = ((col % u.torusC) + u.torusC) % u.torusC;
  return torusBuf[u32(r) * u32(u.torusC) + u32(c)];
}
fn predictedBilinear(worldX: f32, worldZ: f32) -> f32 {
  let xf = worldX + f32(u.torusC) * 0.5 - 0.5;
  let zf = worldZ + f32(u.torusR) * 0.5 - 0.5;
  let c0 = i32(floor(xf)); let r0 = i32(floor(zf));
  let fx = xf - f32(c0); let fz = zf - f32(r0);
  let b00 = torusBrightness(r0, c0); let b10 = torusBrightness(r0, c0 + 1);
  let b01 = torusBrightness(r0 + 1, c0); let b11 = torusBrightness(r0 + 1, c0 + 1);
  return b00 * (1.0 - fx) * (1.0 - fz) + b10 * fx * (1.0 - fz) + b01 * (1.0 - fx) * fz + b11 * fx * fz;
}

fn computeResidual(qq: vec4<f32>, wx0: f32, wz0: f32, ndcU: f32, ndcV: f32, observed: f32) -> vec2<f32> {
  let drow = rotateByQuat(u.drow0.xyz, qq);
  let dcol = rotateByQuat(u.dcol0.xyz, qq);
  var normal = rotateByQuat(u.dnormal0.xyz, qq);
  let checkDir = cornerDir(0.0, 0.0, u.camQuat, u.vFovRad, u.aspect);
  if (dot(checkDir, normal) > 0.0) { normal = -normal; }

  let rayDir = cornerDir(ndcU, ndcV, u.camQuat, u.vFovRad, u.aspect);
  let denom = dot(rayDir, normal);
  if (denom >= -u.minGrazingCos) { return vec2<f32>(0.0, 0.0); }
  let hit = rayDir * (-u.distance / denom);
  let uu = dot(hit, drow);
  let vv = dot(hit, dcol);
  let predicted = predictedBilinear(wx0 + uu, wz0 + vv);
  return vec2<f32>(predicted - observed, 1.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.sampleCount) { return; }
  let px = samplesPx[i]; let py = samplesPy[i]; let observed = samplesObs[i];
  // Phase 3's OWN toNDC (positionLM.ts), deliberately NOT the same formula
  // votes.ts's computeWorldVotes uses -- no "1 - " flip on V here.
  let ndcU = (px / u.w) * 2.0 - 1.0;
  let ndcV = (py / u.h) * 2.0 - 1.0;

  let o = i * 6u;
  outResiduals[o + 0u] = computeResidual(u.q, u.wx0, u.wz0, ndcU, ndcV, observed);

  let qx = axisAngleQuat(vec3<f32>(1.0, 0.0, 0.0), u.epsRot);
  outResiduals[o + 1u] = computeResidual(quatMultiply(qx, u.q), u.wx0, u.wz0, ndcU, ndcV, observed);
  let qy = axisAngleQuat(vec3<f32>(0.0, 1.0, 0.0), u.epsRot);
  outResiduals[o + 2u] = computeResidual(quatMultiply(qy, u.q), u.wx0, u.wz0, ndcU, ndcV, observed);
  let qz = axisAngleQuat(vec3<f32>(0.0, 0.0, 1.0), u.epsRot);
  outResiduals[o + 3u] = computeResidual(quatMultiply(qz, u.q), u.wx0, u.wz0, ndcU, ndcV, observed);

  outResiduals[o + 4u] = computeResidual(u.q, u.wx0 + u.epsPos, u.wz0, ndcU, ndcV, observed);
  outResiduals[o + 5u] = computeResidual(u.q, u.wx0, u.wz0 + u.epsPos, ndcU, ndcV, observed);
}
`;
