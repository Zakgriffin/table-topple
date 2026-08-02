// WGSL source for the GPU port of pipeline/gridPeriodPhase.ts's COARSE PERIOD
// SWEEP -- the `for (n = nMax; n >= nMin; n--)` loop that evaluates
// resultantAt(spread/n) for every integer cell-count candidate.
//
// One WORKGROUP per candidate period, one thread per line within it, then a
// shared-memory tree reduction down to that candidate's single score. The
// candidates are completely independent of each other (each folds the same
// values at a different period), which is what makes this an island worth
// porting despite the tiny data: ~256-512 candidates x 2 families x every
// detected line, off a few KB of input.
//
// ── The f32 range-reduction problem, and why the host pre-scales ──────────
//
// The CPU computes `theta = 2*pi*value/period` directly. Ported naively that is
// a precision trap: period = spread/n gets SMALL as n grows, so value/period
// reaches ~n (up to ~512), and f32's ~7 significant digits leave only ~3-4
// digits after the point at that magnitude. Feeding a large argument to sin/cos
// then compounds it with the hardware's own range reduction.
//
// So the host uploads `scaled[i] = (value[i] - minValue) / spread`, computed in
// f64 and landing in [0,1] where f32 is at full precision, and this shader
// forms `fract(scaled * n)` -- exact same fold, argument never leaves [0,1),
// and sin/cos only ever see [0, 2pi).
//
// Subtracting minValue is free for this kernel's purpose: shifting every value
// by a constant rotates every unit vector by the SAME angle, which leaves the
// resultant's magnitude untouched. It would move the PHASE, which is exactly
// why phase is not computed here -- gridPeriodPhase.ts keeps calling circularFit
// on CPU for the handful of candidates it actually selects.
export const PERIOD_SWEEP_WGSL = /* wgsl */ `
struct Uniforms {
  rowCount: u32, colCount: u32, nMin: u32, nMax: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
// Row family then col family, concatenated -- one buffer rather than two so the
// per-family loop below is a bounds change, not a second binding.
@group(0) @binding(1) var<storage, read> scaled: array<f32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
// One score per candidate, indexed by (nMax - n) so it comes back in ASCENDING
// PERIOD order -- the order gridPeriodPhase.ts's coarseSamples is built in, and
// which its interior-local-maximum scan depends on.
@group(0) @binding(3) var<storage, read_write> scores: array<f32>;

const WG = 64u;
const TAU = 6.283185307179586;

// Six accumulators, not three: the two families are reduced independently in
// one pass (their resultants are summed only at the very end, as resultantAt
// does), so a workgroup never needs a second dispatch or a second traversal.
var<workgroup> pCos: array<f32, WG>;
var<workgroup> pSin: array<f32, WG>;
var<workgroup> pW: array<f32, WG>;
var<workgroup> qCos: array<f32, WG>;
var<workgroup> qSin: array<f32, WG>;
var<workgroup> qW: array<f32, WG>;

@compute @workgroup_size(64)
fn sweep(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let slot = wg.x;
  let n = u.nMax - slot; // ascending period == descending n
  if (n < u.nMin) { return; }
  let fn32 = f32(n);
  let t = lid.x;

  var rc = 0.0; var rs = 0.0; var rw = 0.0;
  var cc = 0.0; var cs = 0.0; var cw = 0.0;

  // Grid-stride over each family. Lines are only a few hundred, so this loop
  // usually runs a handful of times per thread.
  for (var i = t; i < u.rowCount; i = i + WG) {
    let th = TAU * fract(scaled[i] * fn32);
    let wt = weights[i];
    rc = rc + wt * cos(th); rs = rs + wt * sin(th); rw = rw + wt;
  }
  for (var i = t; i < u.colCount; i = i + WG) {
    let j = u.rowCount + i;
    let th = TAU * fract(scaled[j] * fn32);
    let wt = weights[j];
    cc = cc + wt * cos(th); cs = cs + wt * sin(th); cw = cw + wt;
  }

  pCos[t] = rc; pSin[t] = rs; pW[t] = rw;
  qCos[t] = cc; qSin[t] = cs; qW[t] = cw;
  workgroupBarrier();

  // Tree reduction. The halving loop is uniform across the workgroup, so the
  // barrier inside it is uniformly reached -- required by WGSL.
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (t < s) {
      pCos[t] = pCos[t] + pCos[t + s]; pSin[t] = pSin[t] + pSin[t + s]; pW[t] = pW[t] + pW[t + s];
      qCos[t] = qCos[t] + qCos[t + s]; qSin[t] = qSin[t] + qSin[t + s]; qW[t] = qW[t] + qW[t + s];
    }
    workgroupBarrier();
  }

  if (t == 0u) {
    // Same guard and the same sum-of-two-resultants shape as resultantAt: a
    // family with no weight contributes 0 rather than a NaN.
    var total = 0.0;
    if (pW[0] >= 1e-9) { total = total + length(vec2<f32>(pCos[0], pSin[0])) / pW[0]; }
    if (qW[0] >= 1e-9) { total = total + length(vec2<f32>(qCos[0], qSin[0])) / qW[0]; }
    scores[slot] = total;
  }
}
`;
