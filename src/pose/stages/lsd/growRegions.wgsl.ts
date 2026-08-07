// WGSL source for the GPU port of pose/stages/lsd/lsdSegments.ts's stage 2+3
// (growRegionsCCL -- directed connected-component region growing).
//
// Three entry points, all pure per-pixel maps over the field:
//
//   init     seeds label[i] from the eligibility test and normalizes the
//            level-line vector once, since the hook pass re-reads it every
//            round and it never changes across rounds.
//   hook     next[i] = min(label[i], min over edge-connected neighbours j of
//            label[j]) -- reads ONLY the frozen `label`, writes ONLY `next`.
//   compress label[i] = next[next[i]] -- pointer jumping. Reads all of `next`,
//            reads and writes only its OWN label[i].
//
// Neither pass has a same-round cross-pixel write dependency, which is the
// property that makes each one a single trivially-parallel dispatch. hook
// writes `next` and never reads it; compress writes `label[i]` and never reads
// any OTHER thread's label. They run as SEPARATE compute passes rather than
// two dispatches in one pass, because the barrier between them is real:
// compress reads next[l] for an arbitrary l written by some other thread in
// hook. WebGPU guarantees a memory barrier between passes in a command
// encoder, not between dispatches inside one pass.
//
// See growRegionsCCL's own header in lsdSegments.ts for why the doubling
// happens in LABEL space (pointer jumping) rather than image space, and why
// that makes long-range shortcutting structurally unable to jump onto a
// different parallel ridge.
export const GROW_REGIONS_WGSL = /* wgsl */ `
struct Uniforms {
  w: u32, h: u32, pad0: u32, pad1: u32,
  rhoLowSq: f32, cosTol: f32, pad2: f32, pad3: f32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> fx: array<f32>;
@group(0) @binding(2) var<storage, read> fy: array<f32>;
@group(0) @binding(3) var<storage, read_write> ux: array<f32>;
@group(0) @binding(4) var<storage, read_write> uy: array<f32>;
@group(0) @binding(5) var<storage, read_write> label: array<i32>;
@group(0) @binding(6) var<storage, read_write> next: array<i32>;
@group(0) @binding(7) var<storage, read_write> changed: atomic<u32>;

// Matches NEIGHBOR_DX/NEIGHBOR_DY in lsdSegments.ts exactly -- the full
// 8-neighbourhood, including the perpendicular ones. At stride 1 a
// perpendicular neighbour is one pixel away, testing it is free, and it is
// what lets a genuinely 2px-thick ridge grow across its own width instead of
// splitting into two parallel components.
const NDX = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
const NDY = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.w || gid.y >= u.h) { return; }
  let i = gid.y * u.w + gid.x;
  // The level line runs perpendicular to the gradient: (ux, uy) = (-fy, fx)
  // normalized. This used to read a theta array and take its cos/sin -- an
  // angle that a previous pass had spent an atan2 building out of exactly
  // these two numbers. COMPONENT ORDER IS EASY TO GET WRONG HERE and was, from
  // faf55f6 until 2026-08-04 -- see the level-line vector block in
  // lsdSegments.ts for why growing stayed bit-identical while the fit did not.
  //
  // Eligibility is tested SQUARED, so an ineligible pixel costs no sqrt, and
  // only an eligible one is normalized. Ineligible entries keep (0,0), which
  // no round ever reads -- label -1 is checked first everywhere.
  let gx = fx[i];
  let gy = fy[i];
  let m2 = gx * gx + gy * gy;
  // Dense seeding: every eligible pixel starts as its own singleton. There is
  // no "which pixels get to seed" decision to make -- with a symmetric edge
  // predicate the seed set cannot influence the result.
  if (m2 > u.rhoLowSq) {
    let inv = inverseSqrt(m2);
    ux[i] = -gy * inv;
    uy[i] = gx * inv;
    label[i] = i32(i);
  } else {
    ux[i] = 0.0;
    uy[i] = 0.0;
    label[i] = -1;
  }
  next[i] = label[i];
}

@compute @workgroup_size(8, 8)
fn hook(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.w || gid.y >= u.h) { return; }
  let i = gid.y * u.w + gid.x;
  let own = label[i];
  if (own < 0) { next[i] = -1; return; } // below rhoLow -- never participates
  var best = own;
  let ci = ux[i];
  let si = uy[i];
  for (var k = 0; k < 8; k = k + 1) {
    let nx = i32(gid.x) + NDX[k];
    let ny = i32(gid.y) + NDY[k];
    if (nx < 0 || nx >= i32(u.w) || ny < 0 || ny >= i32(u.h)) { continue; }
    let j = u32(ny) * u.w + u32(nx);
    let nlab = label[j];
    // -1 is ineligible; >= best cannot lower our running min. Same two early
    // outs the CPU loop takes, in the same order.
    if (nlab < 0 || nlab >= best) { continue; }
    // THE predicate: one signed dot against cos(tau). Directed -- no abs() --
    // which is what keeps the two antiparallel edges of a thin stripe from
    // fusing. See levelLinesCompatible in lsdSegments.ts.
    if (ci * ux[j] + si * uy[j] < u.cosTol) { continue; }
    best = nlab;
  }
  next[i] = best;
}

@compute @workgroup_size(8, 8)
fn compress(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.w || gid.y >= u.h) { return; }
  let i = gid.y * u.w + gid.x;
  let l = next[i];
  // next[l] is always defined and non-negative for l >= 0: labels only ever
  // originate from eligible pixels, so no second guard is needed past this.
  var jumped: i32 = -1;
  if (l >= 0) { jumped = next[l]; }
  if (jumped != label[i]) { atomicStore(&changed, 1u); }
  label[i] = jumped;
}
`;
