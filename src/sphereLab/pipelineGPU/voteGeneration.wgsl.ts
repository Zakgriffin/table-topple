// WGSL source for the GPU vote-generation pipeline -- the GPU-resident
// counterpart to pipeline/gradientField.ts + pipeline/tangentWalk.ts +
// pipeline/votes.ts's computeWorldVotes. See voteGeneration.ts for the
// dispatch sequence and voteGeneration.wgsl.md-style notes below each
// shader for exactly which CPU function it mirrors and any deliberate
// deviation.
//
// One deliberate numeric deviation, worth flagging up front: the CPU
// agreement field normalizes by the frame's max raw gradient magnitude
// (maxRawMag in computeGradientAgreementField) -- originally so the (since
// removed) 'agreement' debug field-view displayed as a sane [0,1] grayscale
// value. That normalization is a uniform positive rescaling of every
// pixel's agreement value, and every downstream consumer (the walk's
// magnitude-ratio thresholds, fitPairOfPlanes' weight/maxWeight sharpening,
// the percentile-rank vote filtering) is scale-invariant to a uniform
// rescaling -- so this GPU path skips computing that normalization constant
// entirely (it would need its own reduction pass) and uses the raw,
// unnormalized agreement magnitude instead. Verified empirically (see
// pre-Stage-A history) to produce numerically identical votes/fit/LM/decode
// output on the saved-capture.json fixture.

// ── Stage 1: gradient field (mirrors computeGradientField) ────────────────
export const GRADIENT_WGSL = /* wgsl */ `
struct Dims { w: u32, h: u32, r: u32, pad: u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> fxOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> fyOut: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dims.w || y >= dims.h) { return; }
  let r = dims.r;
  let i = y * dims.w + x;
  if (x < r || x >= dims.w - r || y < r || y >= dims.h - r) {
    fxOut[i] = 0.0; fyOut[i] = 0.0;
    return;
  }
  fxOut[i] = gray[i + r] - gray[i - r];
  fyOut[i] = gray[i + r * dims.w] - gray[i - r * dims.w];
}
`;

// ── Stage 2a: double-angle fold (fx,fy -> cx,cy, the pre-blur step inside
// computeGradientAgreementField) ──────────────────────────────────────────
export const DOUBLE_ANGLE_WGSL = /* wgsl */ `
struct Dims { w: u32, h: u32, r: u32, pad: u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> fx: array<f32>;
@group(0) @binding(2) var<storage, read> fy: array<f32>;
@group(0) @binding(3) var<storage, read_write> cx: array<f32>;
@group(0) @binding(4) var<storage, read_write> cy: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dims.w || y >= dims.h) { return; }
  let i = y * dims.w + x;
  let fxi = fx[i]; let fyi = fy[i];
  let mag = length(vec2<f32>(fxi, fyi));
  if (mag == 0.0) { cx[i] = 0.0; cy[i] = 0.0; return; }
  // mag*cos(2*atan2(fy,fx)) == (fx^2-fy^2)/mag, mag*sin(2*atan2(fy,fx)) == 2*fx*fy/mag --
  // standard double-angle identity, avoids an atan2+cos+sin round trip.
  cx[i] = (fxi * fxi - fyi * fyi) / mag;
  cy[i] = (2.0 * fxi * fyi) / mag;
}
`;

// ── Stage 2b/2c: separable clamped-window box blur (mirrors
// distortion.ts's separableBoxBlur exactly, including its edge behavior --
// window shrinks near the border rather than zero-padding, normalized by
// the ACTUAL window size each pixel used, not a fixed 2r+1) ───────────────
export const BOX_BLUR_H_WGSL = /* wgsl */ `
struct BlurDims { w: u32, h: u32, radius: u32, pad: u32 }
@group(0) @binding(0) var<uniform> dims: BlurDims;
@group(0) @binding(1) var<storage, read> srcX: array<f32>;
@group(0) @binding(2) var<storage, read> srcY: array<f32>;
@group(0) @binding(3) var<storage, read_write> dstX: array<f32>;
@group(0) @binding(4) var<storage, read_write> dstY: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dims.w || y >= dims.h) { return; }
  let r = i32(dims.radius);
  let lo = max(0, i32(x) - r);
  let hi = min(i32(dims.w) - 1, i32(x) + r);
  var sumX = 0.0; var sumY = 0.0;
  let row = y * dims.w;
  for (var xi = lo; xi <= hi; xi = xi + 1) {
    sumX = sumX + srcX[row + u32(xi)];
    sumY = sumY + srcY[row + u32(xi)];
  }
  let count = f32(hi - lo + 1);
  let i = row + x;
  dstX[i] = sumX / count;
  dstY[i] = sumY / count;
}
`;
export const BOX_BLUR_V_WGSL = /* wgsl */ `
struct BlurDims { w: u32, h: u32, radius: u32, pad: u32 }
@group(0) @binding(0) var<uniform> dims: BlurDims;
@group(0) @binding(1) var<storage, read> srcX: array<f32>;
@group(0) @binding(2) var<storage, read> srcY: array<f32>;
@group(0) @binding(3) var<storage, read_write> dstX: array<f32>;
@group(0) @binding(4) var<storage, read_write> dstY: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dims.w || y >= dims.h) { return; }
  let r = i32(dims.radius);
  let lo = max(0, i32(y) - r);
  let hi = min(i32(dims.h) - 1, i32(y) + r);
  var sumX = 0.0; var sumY = 0.0;
  for (var yi = lo; yi <= hi; yi = yi + 1) {
    let idx = u32(yi) * dims.w + x;
    sumX = sumX + srcX[idx];
    sumY = sumY + srcY[idx];
  }
  let count = f32(hi - lo + 1);
  let i = y * dims.w + x;
  dstX[i] = sumX / count;
  dstY[i] = sumY / count;
}
`;

// ── Stage 3: effective field (mirrors computeEffectiveGradientField, fused
// with the "agreement = hypot(sx,sy)" step from computeGradientAgreementField
// -- see this file's header comment re: the skipped maxRawMag normalization) ─
export const EFFECTIVE_WGSL = /* wgsl */ `
struct Dims { w: u32, h: u32, r: u32, pad: u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> fx: array<f32>;
@group(0) @binding(2) var<storage, read> fy: array<f32>;
@group(0) @binding(3) var<storage, read> sx: array<f32>;
@group(0) @binding(4) var<storage, read> sy: array<f32>;
@group(0) @binding(5) var<storage, read_write> effFx: array<f32>;
@group(0) @binding(6) var<storage, read_write> effFy: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dims.w || y >= dims.h) { return; }
  let i = y * dims.w + x;
  let agreement = length(vec2<f32>(sx[i], sy[i]));
  effFx[i] = fx[i] * agreement;
  effFy[i] = fy[i] * agreement;
}
`;

// ── Stage 4: guided tangent walk + vote cast (mirrors
// guidedTangentDirection + computeWorldVotes' per-pixel loop body). Output
// is vec4<f32>(nx,ny,nz,weight) per pixel; weight==0 means "no vote here"
// (margin pixel, zero seed, or degenerate cross product) -- filtered out
// client-side after readback, see voteGeneration.ts. ───────────────────────
export const WALK_AND_VOTE_WGSL = /* wgsl */ `
struct WalkUniforms {
  w: u32, h: u32, r: u32, maxSteps: u32,
  devCos: f32, magFraction: f32, graceSamples: u32, aspect: f32,
  vFovRad: f32, pad0: f32, pad1: f32, pad2: f32,
  quat: vec4<f32>,
}
@group(0) @binding(0) var<uniform> u: WalkUniforms;
@group(0) @binding(1) var<storage, read> effFx: array<f32>;
@group(0) @binding(2) var<storage, read> effFy: array<f32>;
@group(0) @binding(3) var<storage, read_write> voteOut: array<vec4<f32>>;

const PI: f32 = 3.14159265358979;

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
  let r = u.r;
  let i = y * u.w + x;
  if (x < r || x >= u.w - r || y < r || y >= u.h - r) { voteOut[i] = vec4<f32>(0.0); return; }

  let seedFx = effFx[i]; let seedFy = effFy[i];
  if (seedFx == 0.0 && seedFy == 0.0) { voteOut[i] = vec4<f32>(0.0); return; }

  let seedMag = length(vec2<f32>(seedFx, seedFy));
  let seedTheta = atan2(seedFy, seedFx);
  let tdx = -sin(seedTheta); let tdy = cos(seedTheta);

  var sumCos = cos(2.0 * seedTheta) * seedMag;
  var sumSin = sin(2.0 * seedTheta) * seedMag;
  var runningMag = seedMag;
  var sampleCount = 1.0;

  // sign = +1 pass then sign = -1 pass (matches the CPU walk's own two
  // passes), sharing the running consensus across both halves of the line.
  for (var signIdx = 0u; signIdx < 2u; signIdx = signIdx + 1u) {
    let sign = select(-1.0, 1.0, signIdx == 0u);
    var violations = 0u;
    for (var k = 1u; k <= u.maxSteps; k = k + 1u) {
      let fk = f32(k);
      let sx = i32(round(f32(x) + sign * fk * tdx));
      let sy = i32(round(f32(y) + sign * fk * tdy));
      if (sx < 0 || sx >= i32(u.w) || sy < 0 || sy >= i32(u.h)) { break; }
      let si = u32(sy) * u.w + u32(sx);
      let sfx = effFx[si]; let sfy = effFy[si];
      let mag = length(vec2<f32>(sfx, sfy));
      if (mag == 0.0 || mag < runningMag * u.magFraction) {
        violations = violations + 1u;
        if (violations >= u.graceSamples) { break; }
        continue;
      }
      let theta = atan2(sfy, sfx);
      let c2 = cos(2.0 * theta); let s2 = sin(2.0 * theta);
      let avgLen = length(vec2<f32>(sumCos, sumSin));
      var cosDeviation = 1.0;
      if (avgLen > 0.0) { cosDeviation = (c2 * sumCos + s2 * sumSin) / avgLen; }
      if (cosDeviation < u.devCos) {
        violations = violations + 1u;
        if (violations >= u.graceSamples) { break; }
        continue;
      }
      violations = 0u;
      sumCos = sumCos + c2 * mag;
      sumSin = sumSin + s2 * mag;
      runningMag = (runningMag * sampleCount + mag) / (sampleCount + 1.0);
      sampleCount = sampleCount + 1.0;
    }
  }

  var theta2 = atan2(sumSin, sumCos) * 0.5;
  if (theta2 < 0.0) { theta2 = theta2 + PI; }
  if (theta2 >= PI) { theta2 = theta2 - PI; }
  let ttx = -sin(theta2); let tty = cos(theta2);

  let fw = f32(u.w); let fh = f32(u.h);
  let ndcU1 = (f32(x) / fw) * 2.0 - 1.0;
  let ndcV1 = 1.0 - (f32(y) / fh) * 2.0;
  let ndcU2 = ((f32(x) + ttx) / fw) * 2.0 - 1.0;
  let ndcV2 = 1.0 - ((f32(y) + tty) / fh) * 2.0;

  let ray1 = cornerDir(ndcU1, ndcV1, u.quat, u.vFovRad, u.aspect);
  let ray2 = cornerDir(ndcU2, ndcV2, u.quat, u.vFovRad, u.aspect);
  let n = cross(ray1, ray2);
  let nLenSq = dot(n, n);
  if (nLenSq < 1e-12) { voteOut[i] = vec4<f32>(0.0); return; }
  voteOut[i] = vec4<f32>(n / sqrt(nLenSq), seedMag);
}
`;
