// WGSL source for the GPU port of pipeline/lsdSegments.ts's stage 4
// (magnitude-weighted PCA rectangle fit) + stage 5's FIRST-PASS NFA test
// (countRectanglePixels + logBinomialTail, at the ORIGINAL toleranceDeg,
// no retry). One thread per region -- regions are already independent by
// construction (stage 3's serial flood-fill fixed the partition before this
// ever runs), and a region's own pixel count is typically small enough
// (tens-hundreds) that a single thread just running the same serial
// reductions the CPU version does, once per region, is simpler and safe
// -- no workgroup-shared-memory cooperative reduction needed, unlike
// fitPlanes.wgsl.ts's single GLOBAL reduction over every vote (a
// fundamentally different shape: many small independent per-region
// reductions here vs. one big reduction there).
//
// The retry loop (tighten-then-shrink on NFA rejection) is deliberately
// NOT ported -- retry 2+ needs a per-region partial sort/selection (drop
// the farthest-from-center fraction), a genuinely harder GPU problem than
// anything else in this stage. lsdFit.ts's caller instead falls back to
// the existing CPU retry loop (pipeline/lsdSegments.ts's
// fitRegionWithRetries) for any region this first pass rejects -- see this
// file's own header in lsdSegments.ts for why re-running that region's
// attempt 0 on CPU too (rather than resuming from this kernel's partial
// state) is the safer, simpler handoff.
//
// logBinomialTail's log-sum-exp is done as an ONLINE/streaming update
// (rescale-on-new-max) rather than the CPU version's two-pass
// (collect-all-terms-then-reduce) approach -- WGSL has no dynamic-length
// local arrays, and n-k here is unbounded (a rectangle's own bbox pixel
// count), so terms can't be buffered. Numerically equivalent, standard
// technique.
export const LSD_FIT_WGSL = /* wgsl */ `
struct Uniforms {
  w: u32, h: u32, regionCount: u32, pad0: u32,
  rho: f32, toleranceRad: f32, logNTests: f32, logEpsilon: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> mag: array<f32>;
@group(0) @binding(2) var<storage, read> theta: array<f32>;
@group(0) @binding(3) var<storage, read> memberOffsets: array<u32>; // [regionCount + 1]
@group(0) @binding(4) var<storage, read> memberIndices: array<u32>; // flat, CSR
@group(0) @binding(5) var<storage, read> meanAngles: array<f32>; // [regionCount]
@group(0) @binding(6) var<storage, read_write> outBuf: array<f32>; // [regionCount * 8]: cx,cy,theta,length,width,nfaLog10,accepted(0/1),pad

const PI: f32 = 3.14159265358979;
const LN10: f32 = 2.302585092994046;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let ri = gid.x;
  if (ri >= u.regionCount) { return; }
  let start = memberOffsets[ri];
  let end = memberOffsets[ri + 1u];
  let o = ri * 8u;
  if (end - start < 2u) {
    outBuf[o + 6u] = 0.0; // too few members -- leave rejected, same as CPU's "degenerate -- no meaningful axis"
    return;
  }

  // ── Stage 4: magnitude-weighted PCA rectangle fit ──────────────────────
  var sumW: f32 = 0.0; var sumX: f32 = 0.0; var sumY: f32 = 0.0;
  for (var mi = start; mi < end; mi = mi + 1u) {
    let i = memberIndices[mi];
    let m = mag[i];
    sumW = sumW + m;
    sumX = sumX + m * f32(i % u.w);
    sumY = sumY + m * f32(i / u.w);
  }
  let wcx = sumX / sumW;
  let wcy = sumY / sumW;

  var Ixx: f32 = 0.0; var Iyy: f32 = 0.0; var Ixy: f32 = 0.0;
  for (var mi = start; mi < end; mi = mi + 1u) {
    let i = memberIndices[mi];
    let m = mag[i];
    let x = f32(i % u.w) - wcx;
    let y = f32(i / u.w) - wcy;
    Ixx = Ixx + m * x * x; Iyy = Iyy + m * y * y; Ixy = Ixy + m * x * y;
  }
  var th = 0.5 * atan2(2.0 * Ixy, Ixx - Iyy);
  let meanAngle = meanAngles[ri];
  if (cos(th) * cos(meanAngle) + sin(th) * sin(meanAngle) < 0.0) { th = th + PI; }

  let ax = cos(th); let ay = sin(th);
  let px = -ay; let py = ax;
  var minProj: f32 = 1e30; var maxProj: f32 = -1e30;
  var minPerp: f32 = 1e30; var maxPerp: f32 = -1e30;
  for (var mi = start; mi < end; mi = mi + 1u) {
    let i = memberIndices[mi];
    let x = f32(i % u.w) - wcx;
    let y = f32(i / u.w) - wcy;
    let proj = x * ax + y * ay; let perp = x * px + y * py;
    minProj = min(minProj, proj); maxProj = max(maxProj, proj);
    minPerp = min(minPerp, perp); maxPerp = max(maxPerp, perp);
  }
  let midProj = (minProj + maxProj) * 0.5;
  let midPerp = (minPerp + maxPerp) * 0.5;
  let cx = wcx + midProj * ax + midPerp * px;
  let cy = wcy + midProj * ay + midPerp * py;
  let length = maxProj - minProj;
  let width = maxPerp - minPerp;

  // ── Stage 5 (first pass only): NFA validation ──────────────────────────
  let hl = length * 0.5;
  let hw = max(width * 0.5, 0.5);
  var minX: f32 = 1e30; var maxX: f32 = -1e30;
  var minY: f32 = 1e30; var maxY: f32 = -1e30;
  for (var c = 0u; c < 4u; c = c + 1u) {
    var a: f32; var b: f32;
    if (c == 0u) { a = hl; b = hw; }
    else if (c == 1u) { a = hl; b = -hw; }
    else if (c == 2u) { a = -hl; b = -hw; }
    else { a = -hl; b = hw; }
    let x = cx + a * ax + b * px; let y = cy + a * ay + b * py;
    minX = min(minX, x); maxX = max(maxX, x); minY = min(minY, y); maxY = max(maxY, y);
  }
  let x0 = u32(max(0.0, floor(minX)));
  let x1 = min(u.w - 1u, u32(max(0.0, ceil(maxX))));
  let y0 = u32(max(0.0, floor(minY)));
  let y1 = min(u.h - 1u, u32(max(0.0, ceil(maxY))));

  var n: u32 = 0u; var k: u32 = 0u;
  for (var y = y0; y <= y1; y = y + 1u) {
    for (var x = x0; x <= x1; x = x + 1u) {
      let dx = f32(x) - cx; let dy = f32(y) - cy;
      let proj = dx * ax + dy * ay; let perp = dx * px + dy * py;
      if (abs(proj) > hl || abs(perp) > hw) { continue; }
      n = n + 1u;
      let i = y * u.w + x;
      if (mag[i] <= u.rho) { continue; }
      var d = abs(theta[i] - th) % (2.0 * PI);
      if (d > PI) { d = 2.0 * PI - d; }
      if (d <= u.toleranceRad) { k = k + 1u; }
    }
  }

  let p = u.toleranceRad / PI;
  var logBinTail: f32 = 0.0; // log(1) -- k==0 trivially has P(X>=0)=1, matches CPU's early return
  if (k > 0u) {
    let logP = log(p); let log1mP = log(1.0 - p);
    var logChoose: f32 = 0.0;
    for (var i = 1u; i <= k; i = i + 1u) { logChoose = logChoose + log(f32(n - k + i) / f32(i)); }
    var runningMax = logChoose + f32(k) * logP + f32(n - k) * log1mP;
    var runningSum: f32 = 1.0;
    for (var j = k + 1u; j <= n; j = j + 1u) {
      logChoose = logChoose + log(f32(n - j + 1u) / f32(j));
      let logTerm = logChoose + f32(j) * logP + f32(n - j) * log1mP;
      if (logTerm > runningMax) {
        runningSum = runningSum * exp(runningMax - logTerm) + 1.0;
        runningMax = logTerm;
      } else {
        runningSum = runningSum + exp(logTerm - runningMax);
      }
    }
    logBinTail = runningMax + log(runningSum);
  }
  let logNfa = u.logNTests + logBinTail;
  let accepted = logNfa < u.logEpsilon;

  outBuf[o + 0u] = cx; outBuf[o + 1u] = cy; outBuf[o + 2u] = th;
  outBuf[o + 3u] = length; outBuf[o + 4u] = width;
  outBuf[o + 5u] = logNfa / LN10;
  outBuf[o + 6u] = select(0.0, 1.0, accepted);
  outBuf[o + 7u] = 0.0;
}
`;
