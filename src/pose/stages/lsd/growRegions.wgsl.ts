// WGSL source for the GPU port of pose/stages/lsd/regions.cpu.ts's stage 2+3
// (growRegionsCCL -- directed connected-component region growing).
//
// Four entry points. The first three are pure per-pixel maps over the field:
//
//   init     seeds label[i] from the eligibility test and normalizes the
//            level-line vector once, since the hook pass re-reads it every
//            round and it never changes across rounds.
//   hook     next[i] = min(label[i], min over edge-connected neighbours j of
//            label[j]) -- reads ONLY the frozen `label`, writes ONLY `next`.
//   compress label[i] = next[next[i]] -- pointer jumping. Reads all of `next`,
//            reads and writes only its OWN label[i].
//   gate     ONE thread, and it is the whole convergence mechanism -- see below.
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
// See growRegionsCCL's own header in the LSD stage for why the doubling
// happens in LABEL space (pointer jumping) rather than image space, and why
// that makes long-range shortcutting structurally unable to jump onto a
// different parallel ridge.
//
// ── CONVERGENCE IS DECIDED ON DEVICE ──
//
// hook and compress dispatch INDIRECTLY, off `args`, and `gate` zeroes `args`
// on the first round that changes nothing. Every round encoded after that is a
// dispatch of zero workgroups -- valid, ordered, and no threads. So the host no
// longer has to know the round count to stop at: it encodes a batch, and the
// rounds past the fixpoint cost nothing but their encoding.
//
// The flag the host reads is `args.x` itself, not a separate word. Zero
// workgroups IS the converged state, so there is nothing to keep in sync -- and
// a real image can never make args.x legitimately zero, since it is
// ceil(w/8) >= 1 for any w >= 1.
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
// x, y, z as consumed by dispatchWorkgroupsIndirect, plus a fourth word the
// dispatch never reads: the count of rounds that actually changed something.
// It is here rather than in its own buffer because the host reads it in the
// same four words it already reads to test convergence.
//
// GROUP 1, AND THAT IS NOT COSMETIC. Only \`gate\` binds it. hook and compress
// dispatch INDIRECTLY off this same range, and a buffer cannot be both a
// writable storage binding and the indirect source of the same dispatch -- that
// is a usage-scope conflict, which WebGPU reports asynchronously, so the symptom
// would be the silent no-op encoder this file's host header describes rather
// than an exception. Keeping it out of group 0 keeps it out of hook's and
// compress's usage scope entirely.
struct Args { x: u32, y: u32, z: u32, activeRounds: u32 }
@group(1) @binding(0) var<storage, read_write> args: Args;

// Matches NEIGHBOR_DX/NEIGHBOR_DY in the LSD stage exactly -- the full
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
  // the LSD stage for why growing stayed bit-identical while the fit did not.
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
    // fusing. See levelLinesCompatible in the LSD stage.
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

// Runs after compress, once per round, on a single thread. It reads the flag
// THIS round just wrote, so a quiet round is detected on the round it happens
// -- the round operator is deterministic in \`label\` alone, so a round that
// changes no label is a fixpoint and every later round would recompute the
// identical result.
//
// Why the flag is cleared HERE rather than by a host clearBuffer: the host used
// to clear it on the last round of a batch, which made it mean "did the final
// round of this batch change anything". Per-round clearing is what lets the
// early-out fire mid-batch, and it removes the one place the host had to know
// where a batch boundary fell.
//
// Note the two branches are exclusive on purpose. Once converged the flag is
// left at 0 and never cleared again, so re-running gate on the no-op rounds
// that follow is idempotent -- it re-zeroes an already-zero \`args\`.
@compute @workgroup_size(1)
fn gate() {
  if (atomicLoad(&changed) == 0u) {
    args.x = 0u;
    args.y = 0u;
    args.z = 0u;
  } else {
    atomicStore(&changed, 0u);
    args.activeRounds = args.activeRounds + 1u;
  }
}
`;
