// ── The shaders, in pipeline order ────────────────────────────────────────
//
// One export per stage, ordered the way the data flows. See
// full_system_breakdown.md for what each one computes and why.
//
// Binding indices match the `binds` array in pipeline.ts positionally: entry 0
// of a stage's bind list is @binding(0) of its group. That correspondence is
// what lets the encode side build a bind group straight from the declaration,
// and assertBinds() is what keeps the two from drifting.

// ── S1 gradient ───────────────────────────────────────────────────────────
//
// The 2x2 block gradient. Deliberately NOT a symmetric centered difference:
// this leaves only the LAST row and column zero, where a symmetric kernel would
// zero an r-pixel margin on all four sides and silently discard the real data
// along the top and left.
export const GRADIENT_WGSL = /* wgsl */ `
struct U { w: u32, h: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> fx: array<f32>;
@group(0) @binding(3) var<storage, read_write> fy: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= u.w || y >= u.h) { return; }
  let i = y * u.w + x;
  if (x + 1u >= u.w || y + 1u >= u.h) { fx[i] = 0.0; fy[i] = 0.0; return; }
  let g00 = gray[i];
  let g10 = gray[i + 1u];
  let g01 = gray[i + u.w];
  let g11 = gray[i + 1u + u.w];
  // 1/(2*255), so hypot(fx, fy) has a true ceiling of 1 rather than 255.
  let norm = 1.0 / 510.0;
  fx[i] = ((g10 + g11) - (g00 + g01)) * norm;
  fy[i] = ((g01 + g11) - (g00 + g10)) * norm;
}
`;

// ── S2 grow: directed connected components by pointer jumping ─────────────
//
// Four entry points over ONE shared bind-group layout. The sharing is not
// stylistic: `layout: 'auto'` derives each pipeline's layout from the bindings
// that entry point actually references, and none of these four references all
// eight -- init never touches `changed`, hook never touches `changed`, compress
// never touches fx/fy/ux/uy, gate touches only `changed` and `args`. Binding one
// buffer set across four auto layouts fails validation on every
// createBindGroup, and WebGPU reports that ASYNCHRONOUSLY -- so the symptom is
// not an exception but a silently no-op encoder, i.e. an all-zero labelling that
// looks like one image-sized region.
//
//   init      seed every eligible pixel as its own singleton, and normalize the
//             level-line vector once since hook re-reads it every round.
//   hook      next[i] = min(label[i], min over compatible neighbours of label[j])
//   compress  label[i] = next[next[i]] -- pointer jumping, so a component of any
//             size collapses in O(log n) rounds rather than O(diameter).
//   gate      one thread; the entire convergence mechanism.
//
// Each is its OWN compute pass, not several dispatches in one: compress reads
// next[l] for an arbitrary l that some other thread wrote during hook, and
// WebGPU guarantees a storage-buffer barrier BETWEEN passes, not between
// dispatches inside one.
export const GROW_WGSL = /* wgsl */ `
struct U { w: u32, h: u32, pad0: u32, pad1: u32, rhoLowSq: f32, cosTol: f32, pad2: f32, pad3: f32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> fx: array<f32>;
@group(0) @binding(2) var<storage, read> fy: array<f32>;
@group(0) @binding(3) var<storage, read_write> ux: array<f32>;
@group(0) @binding(4) var<storage, read_write> uy: array<f32>;
@group(0) @binding(5) var<storage, read_write> label: array<i32>;
@group(0) @binding(6) var<storage, read_write> next: array<i32>;
@group(0) @binding(7) var<storage, read_write> changed: atomic<u32>;

// GROUP 1, AND THAT IS LOAD-BEARING. Only \`gate\` binds it. hook and compress
// dispatch INDIRECTLY off this same range, and a buffer cannot be both a
// writable storage binding and the indirect source of the same dispatch -- a
// usage-scope conflict, reported asynchronously, so the symptom is again the
// silent no-op rather than an error. Keeping it out of group 0 keeps it out of
// hook's and compress's usage scope entirely.
struct Args { x: u32, y: u32, z: u32, activeRounds: u32 }
@group(1) @binding(0) var<storage, read_write> args: Args;

// The full 8-neighbourhood, perpendicular neighbours included. At stride 1 a
// perpendicular neighbour is one pixel away and testing it is free, and it is
// what lets a genuinely 2px-thick ridge grow across its own width instead of
// splitting into two parallel components.
const NDX = array<i32, 8>(1, 1, 0, -1, -1, -1, 0, 1);
const NDY = array<i32, 8>(0, 1, 1, 1, 0, -1, -1, -1);

@compute @workgroup_size(8, 8)
fn init(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.w || gid.y >= u.h) { return; }
  let i = gid.y * u.w + gid.x;
  let gx = fx[i];
  let gy = fy[i];
  let m2 = gx * gx + gy * gy;
  // Dense seeding -- every eligible pixel starts as its own singleton. With a
  // symmetric edge predicate the seed set cannot influence the result, so there
  // is no "which pixels get to seed" decision to make.
  //
  // Eligibility is tested SQUARED, so an ineligible pixel costs no sqrt and
  // only an eligible one is normalized.
  if (m2 > u.rhoLowSq) {
    let inv = inverseSqrt(m2);
    // The level line runs PERPENDICULAR to the gradient: (ux, uy) = (-fy, fx).
    // Component order is easy to get wrong here and has been wrong in this
    // codebase before -- growing stayed bit-identical while the rectangle fit
    // did not, so nothing downstream of grow caught it.
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
  if (own < 0) { next[i] = -1; return; }
  var best = own;
  let ci = ux[i];
  let si = uy[i];
  for (var k = 0; k < 8; k = k + 1) {
    let nx = i32(gid.x) + NDX[k];
    let ny = i32(gid.y) + NDY[k];
    if (nx < 0 || nx >= i32(u.w) || ny < 0 || ny >= i32(u.h)) { continue; }
    let j = u32(ny) * u.w + u32(nx);
    let nlab = label[j];
    if (nlab < 0 || nlab >= best) { continue; }
    // THE predicate: one SIGNED dot against cos(tau). No abs(), and that is
    // what keeps the two antiparallel edges of a thin stripe from fusing into
    // one region.
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
  // next[l] is defined and non-negative for any l >= 0: labels only ever
  // originate from eligible pixels, so no second guard is needed.
  var jumped: i32 = -1;
  if (l >= 0) { jumped = next[l]; }
  if (jumped != label[i]) { atomicStore(&changed, 1u); }
  label[i] = jumped;
}

// Runs after compress, once per round, on one thread. It reads the flag THIS
// round wrote, so a quiet round is detected on the round it happens -- the round
// operator is deterministic in \`label\` alone, so a round that changes no label
// is a fixpoint and every later round would recompute the identical result.
//
// Clearing per round (rather than the host clearing on the last round of a
// batch) is what lets the early-out fire mid-batch, and removes the one place
// the host had to know where a batch boundary fell.
//
// The two branches are exclusive on purpose: once converged the flag is left at
// 0 and never cleared again, so re-running gate on the no-op rounds that follow
// is idempotent -- it re-zeroes an already-zero \`args\`.
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

// ── The prefix scan, used three times ─────────────────────────────────────
//
// An exclusive prefix sum over `vec2<u32>`, in three passes: scan each block,
// scan the block sums, add the block offsets back. Nothing recurses, because
// the block sums fit in one workgroup for any image this pipeline will see --
// see pipeline.ts, and planPool asserts the bound.
//
// ── WHY vec2 RATHER THAN u32 ──
//
// Every place this pipeline scans, it was about to scan the same array twice
// with two different accumulators:
//
//   collect  (keptFlag, keptCount) -> (dense region id, CSR member offset)
//   gpp      (isRow, 1 - isRow)    -> (row index, column index)
//   lines    (accepted, unused)
//
// One vec2 scan does both lanes in a single traversal, at identical memory
// cost, and its GRAND TOTAL is exactly the pair the next stage needs --
// [regionCount, memberCount], [rowCount, colCount]. That is why nothing here
// copies two scalars into a pair afterwards: the spine writes the pair itself.
//
// The lines scan wastes its .y lane. That costs maxRegions*4 bytes and buys one
// scan implementation instead of two.
export const SCAN_WGSL = /* wgsl */ `
struct U {
  count: u32,      // elements to scan
  perThread: u32,  // contiguous elements each thread owns
  pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> src: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read_write> dst: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read_write> sums: array<vec2<u32>>;

const TH = 256u;
var<workgroup> partials: array<vec2<u32>, 256>;

// One workgroup scans TH * perThread contiguous elements.
//
// Each thread owns a contiguous RUN rather than a strided set. Strided is
// friendlier to coalescing, but contiguous is what lets the same code serve the
// spine, where perThread is whatever it takes to cover the block sums in one
// workgroup. One implementation beats a marginally faster pair of them.
@compute @workgroup_size(256)
fn blocks(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
  let t = lid.x;
  let per = u.perThread;
  let base = wid.x * TH * per + t * per;

  var total = vec2<u32>(0u, 0u);
  for (var k = 0u; k < per; k = k + 1u) {
    let i = base + k;
    if (i < u.count) { total = total + src[i]; }
  }
  partials[t] = total;

  // Hillis-Steele inclusive scan over the 256 thread totals. The barriers sit in
  // uniform control flow -- the loop bound is a constant and the guards contain
  // no barrier -- which WGSL requires and which is easy to break by hoisting the
  // guard outward.
  for (var off = 1u; off < TH; off = off * 2u) {
    workgroupBarrier();
    var add = vec2<u32>(0u, 0u);
    if (t >= off) { add = partials[t - off]; }
    workgroupBarrier();
    if (t >= off) { partials[t] = partials[t] + add; }
  }
  workgroupBarrier();

  // Exclusive: this thread's run starts after everything to its left.
  var acc = vec2<u32>(0u, 0u);
  if (t > 0u) { acc = partials[t - 1u]; }
  for (var k = 0u; k < per; k = k + 1u) {
    let i = base + k;
    if (i < u.count) {
      dst[i] = acc;
      acc = acc + src[i];
    }
  }

  // partials[TH-1] is this workgroup's total. Written and read by the same
  // thread, so it needs no barrier of its own.
  if (t == TH - 1u) { sums[wid.x] = partials[TH - 1u]; }
}

@compute @workgroup_size(256)
fn add(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.count) { return; }
  // src is the block OFFSETS here, not the original input.
  dst[i] = dst[i] + src[i / (TH * u.perThread)];
}
`;


// ── S3 collect: a labelling -> an ordered list of regions ─────────────────
//
// Three jobs: hysteresis (drop regions whose pixels are all weak), a size
// filter, and a CSR structure -- a flat `members` array plus per-region offsets
// and sizes -- so later stages walk one region's pixels contiguously.
//
// Labels ARE pixel indices, so every per-label array has n slots even though
// only a few thousand ever become regions. That is the price of never hashing
// the label space, and it buys something real: scanning labels in ascending
// order makes the region ordering DETERMINISTIC. An atomic append would be one
// pass instead of five, but region ids would depend on scheduling, back-to-back
// runs would stop agreeing bit-for-bit, and the CPU twin would have to
// canonicalize before it could compare. Determinism is worth four passes.
//
// ── ONE SHADER PER ENTRY POINT, unlike grow ──
//
// A WGSL module's bindings are module-scope, shared by every entry point in it.
// Grow's four entry points all bind the same eight buffers, so one module with
// one shared layout is natural there. Collect's five each bind a DIFFERENT set,
// so a single module would force every pass to bind the union -- which would
// hold `members` and `meanDirs` live from the first pass and defeat the
// pooling. Five small modules keep each stage's bindings equal to what
// pipeline.ts declares for it, exactly and positionally.

/** The collect uniform, identical in all five. */
const COLLECT_U = /* wgsl */ `
struct U {
  w: u32, h: u32,
  rhoHighSq: f32,
  minRegionSize: u32,
  maxRegions: u32,
  pad0: u32, pad1: u32, pad2: u32,
}
@group(0) @binding(0) var<uniform> u: U;
`;

// Hysteresis and the histogram in ONE pass. They walk the same pixels and read
// the same label; the split existed only so the histogram could skip labels that
// had already failed hysteresis, and markKept ignores the count of a failed
// label anyway. So the split bought nothing and cost a pass.
//
// `survives` is a plain store of 1 and never a store of 0 -- a weak pixel
// writing 0 would race a strong pixel writing 1 for the same label. It is
// therefore CLEARED each frame (pipeline.ts marks it `zero`), without which a
// label with no strong pixel inherits last frame's 1 and a pure-noise region
// survives hysteresis.
export const COLLECT_TALLY_WGSL = COLLECT_U + /* wgsl */ `
@group(0) @binding(1) var<storage, read> label: array<i32>;
@group(0) @binding(2) var<storage, read> fx: array<f32>;
@group(0) @binding(3) var<storage, read> fy: array<f32>;
@group(0) @binding(4) var<storage, read_write> survives: array<u32>;
@group(0) @binding(5) var<storage, read_write> labelCounts: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.w * u.h) { return; }
  let l = label[i];
  if (l < 0) { return; }
  atomicAdd(&labelCounts[u32(l)], 1u);
  let gx = fx[i];
  let gy = fy[i];
  if (gx * gx + gy * gy > u.rhoHighSq) { survives[u32(l)] = 1u; }
}
`;

// The scan input, and the reason the scan is vec2: .x is 1 for a label that
// becomes a region, .y is that region's pixel count. One exclusive prefix sum
// over the pair yields the dense region id AND the CSR member offset.
export const COLLECT_MARKKEPT_WGSL = COLLECT_U + /* wgsl */ `
@group(0) @binding(1) var<storage, read> survives: array<u32>;
@group(0) @binding(2) var<storage, read> labelCounts: array<u32>;
@group(0) @binding(3) var<storage, read_write> kept: array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let l = gid.x;
  if (l >= u.w * u.h) { return; }
  let c = labelCounts[l];
  // Written unconditionally in both branches, so \`kept\` never needs clearing.
  if (survives[l] == 1u && c >= u.minRegionSize) {
    kept[l] = vec2<u32>(1u, c);
  } else {
    kept[l] = vec2<u32>(0u, 0u);
  }
}
`;

// Re-index the per-label arrays into per-region ones, and write the indirect
// dispatch args every later per-region pass runs off.
//
// Thread 0 writes the args BEFORE the guards. Putting it after would skip it
// whenever label 0 happens not to be a kept region -- which is almost always,
// since label 0 is the top-left pixel.
export const COLLECT_REGIONMETA_WGSL = COLLECT_U + /* wgsl */ `
@group(0) @binding(1) var<storage, read> kept: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> keptScan: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> counts: vec2<u32>;
@group(0) @binding(4) var<storage, read_write> regionOffsets: array<u32>;
@group(0) @binding(5) var<storage, read_write> regionSizes: array<u32>;
@group(0) @binding(6) var<storage, read_write> args: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let l = gid.x;
  if (l == 0u) {
    // Clamped, so an overflowing frame dispatches over the regions that fit
    // rather than over a count no buffer is sized for. \`finish\` compares the
    // unclamped counts.x against maxRegions and reports the overflow.
    let rc = min(counts.x, u.maxRegions);
    args[0] = (rc + 63u) / 64u;
    args[1] = 1u;
    args[2] = 1u;
  }
  if (l >= u.w * u.h) { return; }
  if (kept[l].x == 0u) { return; }
  let r = keptScan[l].x;
  if (r >= u.maxRegions) { return; }
  regionOffsets[r] = keptScan[l].y;
  regionSizes[r] = kept[l].y;
}
`;

// Place each pixel into its region's CSR slice. The cursor is indexed by REGION
// rather than by label: same job, and it turns an n-sized buffer (1.17 MiB at
// 480x640) into a maxRegions-sized one (16 KiB).
export const COLLECT_SCATTER_WGSL = COLLECT_U + /* wgsl */ `
@group(0) @binding(1) var<storage, read> label: array<i32>;
@group(0) @binding(2) var<storage, read> kept: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> keptScan: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> members: array<u32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.w * u.h) { return; }
  let l = label[i];
  if (l < 0) { return; }
  if (kept[u32(l)].x == 0u) { return; }
  let r = keptScan[u32(l)].x;
  if (r >= u.maxRegions) { return; }
  let slot = atomicAdd(&cursor[r], 1u);
  members[keptScan[u32(l)].y + slot] = i;
}
`;

// One thread per region: sort its member slice, then average its members' level
// line directions.
//
// The sort exists because the atomic cursor hands out slots in arrival order,
// which is nondeterministic -- it is the one place this stage's determinism has
// to be restored rather than arranged. Regions run to about a hundred members,
// so an insertion sort by a single thread is the right shape.
//
// The mean is a plain normalized sum of unit vectors with no sign resolution
// against a reference member. With DIRECTED growth every member is within
// tolerance of every member it connects through, so there is no polarity flip
// for the sum to cancel against.
export const COLLECT_FINALIZE_WGSL = COLLECT_U + /* wgsl */ `
@group(0) @binding(1) var<storage, read> counts: vec2<u32>;
@group(0) @binding(2) var<storage, read> fx: array<f32>;
@group(0) @binding(3) var<storage, read> fy: array<f32>;
@group(0) @binding(4) var<storage, read> regionOffsets: array<u32>;
@group(0) @binding(5) var<storage, read> regionSizes: array<u32>;
@group(0) @binding(6) var<storage, read_write> members: array<u32>;
@group(0) @binding(7) var<storage, read_write> meanDirs: array<vec2<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= min(counts.x, u.maxRegions)) { return; }
  let off = regionOffsets[r];
  let sz = regionSizes[r];

  for (var a = 1u; a < sz; a = a + 1u) {
    let v = members[off + a];
    var b = a;
    loop {
      if (b == 0u) { break; }
      let prev = members[off + b - 1u];
      if (prev <= v) { break; }
      members[off + b] = prev;
      b = b - 1u;
    }
    members[off + b] = v;
  }

  var sx = 0.0;
  var sy = 0.0;
  for (var a = 0u; a < sz; a = a + 1u) {
    let p = members[off + a];
    let gx = fx[p];
    let gy = fy[p];
    // Recomputed from the gradient rather than read from ux/uy. Identical
    // value, and it lets ux/uy die at the end of grow instead of staying live
    // across the whole of collect -- two n-sized buffers of pool pressure for
    // one inverseSqrt per member.
    let inv = inverseSqrt(gx * gx + gy * gy);
    sx = sx + -gy * inv;
    sy = sy + gx * inv;
  }
  let m = sqrt(sx * sx + sy * sy);
  if (m > 0.0) { meanDirs[r] = vec2<f32>(sx / m, sy / m); }
  else { meanDirs[r] = vec2<f32>(0.0, 0.0); }
}
`;

// ── S4 lsdFit: a region -> a line segment, and is it real? ────────────────
//
// Two jobs in one pass. The FIT is magnitude-weighted PCA: the region's own
// axis of elongation, its extent along that axis and across it. The TEST is
// NFA -- "if this image were pure noise, how many rectangles this good would I
// expect by chance?" -- which is what makes LSD principled rather than tuned.
// Nothing here has a threshold to pick: the acceptance criterion falls out of a
// binomial tail.
//
// ── ONE THREAD PER REGION, and it stays that way ──
//
// Regions are a disjoint partition, so they are independent by construction and
// this is a plain parallel map. Everything inside is serial: three passes over
// the members (each needs the previous one's answer -- centroid, then central
// moments, then extents along the axis those moments define) and one scan of
// the footprint. A region runs to tens or hundreds of members and its footprint
// to a few thousand pixels, so a thread doing the whole reduction itself is the
// right shape. Cooperating across a workgroup would need a per-region extent
// nobody knows at encode time.
//
// ── THREE THINGS THAT ARE NOT IN THE OLD KERNEL ──
//
// 1. The guard is min(counts.x, maxRegions), not counts.x. \`rects\` is capped
//    now rather than allocated at the provable bound, and regionMeta's clamp is
//    not enough on its own: it clamps the WORKGROUP count, so with maxRegions
//    not a multiple of 64 the last workgroup still runs threads past the cap.
//    Belt and braces rather than load-bearing -- WGSL discards an out-of-bounds
//    storage write and clamps an out-of-bounds read, so the uncapped version
//    would waste threads rather than corrupt anything. Which is exactly why it
//    is written down: nothing would ever have made it visible.
// 2. A zero total weight is degenerate. sumW is a sum of magnitudes, so it is
//    zero only if every member has none -- impossible while rhoLow > 0, and a
//    silent NaN centroid the moment it is not. The old kernel divided anyway.
// 3. Both degenerate paths zero all TEN slots. \`rects\` is reused across frames,
//    so an early return that writes only the accepted flag leaves the previous
//    frame's rectangle in the other nine -- and that rectangle is still a live
//    input to the line compaction. THE GREPPABLE SHAPE: an early return that
//    writes one field.
//
// ── THE CONSTANTS THAT MUST NOT DRIFT ──
//
// BOUNDARY_EPS exists because a rectangle's extent is DEFINED by its extreme
// members: length = maxProj - minProj puts those members at |proj| exactly hl,
// and on a thin region every member sits at |perp| exactly hw. Exact-boundary
// pixels are guaranteed in every region, not a rare case, and a bare
// \`abs(proj) > hl\` decides them on the last ulp -- measured as f32-vs-f64
// disagreement on n for 180 of 2931 regions, moving nfaLog10 by up to 4.7
// decades and flipping 6 accept/reject decisions.
//
// The alignment test is a SIGNED dot, and p = tau/pi is only correct because of
// it: p is the chance a uniformly-random DIRECTED level line lands within +/-tau
// of a fixed direction, measure 2tau out of 2pi. An abs() admits two such
// windows and is twice as permissive as the null model scoring it, so every
// region looks more significant than it is. If mod-pi counting ever comes back,
// p MUST double with it.
//
// ── THE FUSION THAT WAS OPEN HERE, AND WHY IT WAS REJECTED ──
//
// \`lines.flag\` (§9) computes accepted && len >= minLengthPx over exactly the
// rectangle this pass just built, so it looked like one more store at the bottom
// of this kernel and one fewer dispatch. Settled when §9 was built: it stays its
// own pass, and the reason is mechanical rather than aesthetic.
//
// This kernel is a map over [0, regionCount), dispatched off \`regionArgs\`.
// \`lineFlag\` is SCANNED, over a count fixed at encode time, which can only be
// maxRegions -- so it must be a TOTAL function over [0, maxRegions). A store at
// the bottom of this kernel reaches the first regionCount slots and no more,
// leaving the tail holding the previous frame's flags: a frame following a
// busier one emits lines from stale rectangles. Fusing would therefore also
// require \`lineFlag\` to be cleared every frame AND both zeroRect paths above to
// write it, since an early return that skips the store is the same trap again.
//
// THE GENERAL FORM, which is the counterexample worth keeping: a fusion that
// changes a pass's ITERATION DOMAIN is not free, even when both passes are on
// the same side of the bus.
export const LSD_FIT_WGSL = /* wgsl */ `
struct U {
  w: u32, h: u32, maxRegions: u32, pad0: u32,
  rhoSq: f32, toleranceRad: f32, logNTests: f32, logEpsilon: f32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> fx: array<f32>;
@group(0) @binding(2) var<storage, read> fy: array<f32>;
@group(0) @binding(3) var<storage, read> regionOffsets: array<u32>;
@group(0) @binding(4) var<storage, read> regionSizes: array<u32>;
@group(0) @binding(5) var<storage, read> members: array<u32>;
@group(0) @binding(6) var<storage, read> meanDirs: array<vec2<f32>>;
@group(0) @binding(7) var<storage, read> counts: vec2<u32>;
// 10 f32 per region: cx, cy, theta, len, width, nfaLog10, accepted, pad, n, k.
// n and k are diagnostic -- they are what tells a disagreement in the COUNTS
// (which pixels each side thought were inside, and aligned) apart from a
// disagreement in the tail ARITHMETIC. They cost two stores.
@group(0) @binding(8) var<storage, read_write> rects: array<f32>;

const PI: f32 = 3.14159265358979;
const LN10: f32 = 2.302585092994046;
const BOUNDARY_EPS: f32 = 1e-3;

fn zeroRect(o: u32) {
  for (var f = 0u; f < 10u; f = f + 1u) { rects[o + f] = 0.0; }
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= min(counts.x, u.maxRegions)) { return; }
  let o = r * 10u;
  let off = regionOffsets[r];
  let sz = regionSizes[r];
  // Fewer than two members is no axis to fit. minRegionSize normally excludes
  // it, but that is a slider and this is the definition.
  if (sz < 2u) { zeroRect(o); return; }

  // ── The magnitude-weighted centroid ──
  var sumW = 0.0; var sumX = 0.0; var sumY = 0.0;
  for (var a = 0u; a < sz; a = a + 1u) {
    let i = members[off + a];
    let m = sqrt(fx[i] * fx[i] + fy[i] * fy[i]);
    sumW = sumW + m;
    sumX = sumX + m * f32(i % u.w);
    sumY = sumY + m * f32(i / u.w);
  }
  if (sumW <= 0.0) { zeroRect(o); return; }
  let wcx = sumX / sumW;
  let wcy = sumY / sumW;

  // ── The principal axis ──
  //
  // Central moments, then the half-angle form of the major eigenvector. The
  // moments are accumulated ABOUT the centroid rather than about the origin and
  // shifted afterwards: the raw-moment identity is one loop cheaper and loses
  // it in f32, since Ixx would be ~1e5 while the centred value it is trying to
  // recover is ~10.
  var Ixx = 0.0; var Iyy = 0.0; var Ixy = 0.0;
  for (var a = 0u; a < sz; a = a + 1u) {
    let i = members[off + a];
    let m = sqrt(fx[i] * fx[i] + fy[i] * fy[i]);
    let x = f32(i % u.w) - wcx;
    let y = f32(i / u.w) - wcy;
    Ixx = Ixx + m * x * x;
    Iyy = Iyy + m * y * y;
    Ixy = Ixy + m * x * y;
  }
  var th = 0.5 * atan2(2.0 * Ixy, Ixx - Iyy);
  // PCA fixes the axis only up to 180 degrees -- an inertia matrix has no
  // notion of which way along its own axis. The region's own mean level-line
  // DIRECTION picks. Not cosmetic: the NFA test below compares pixel directions
  // against this angle, so a flipped theta makes every aligned pixel read as
  // anti-aligned and the region can never be accepted.
  let md = meanDirs[r];
  if (cos(th) * md.x + sin(th) * md.y < 0.0) { th = th + PI; }

  // ── The extent, and the rectangle it implies ──
  let ax = cos(th); let ay = sin(th);
  let px = -ay; let py = ax;
  var minProj = 1e30; var maxProj = -1e30;
  var minPerp = 1e30; var maxPerp = -1e30;
  for (var a = 0u; a < sz; a = a + 1u) {
    let i = members[off + a];
    let x = f32(i % u.w) - wcx;
    let y = f32(i / u.w) - wcy;
    let proj = x * ax + y * ay;
    let perp = x * px + y * py;
    minProj = min(minProj, proj); maxProj = max(maxProj, proj);
    minPerp = min(minPerp, perp); maxPerp = max(maxPerp, perp);
  }
  // The centre is the MIDPOINT OF THE EXTENT, not the weighted centroid. The
  // centroid is pulled toward whichever end carries more gradient weight, so
  // centring there clips real members off one end while overshooting empty
  // space at the other.
  let midProj = (minProj + maxProj) * 0.5;
  let midPerp = (minPerp + maxPerp) * 0.5;
  let cx = wcx + midProj * ax + midPerp * px;
  let cy = wcy + midProj * ay + midPerp * py;
  let len = maxProj - minProj;
  let wid = maxPerp - minPerp;

  // ── NFA: how many rectangles this good would noise produce? ──
  //
  // n counts pixels whose centre falls in the rotated footprint -- scanned over
  // its bounding box, and NOT the region's own members: the fitted rectangle is
  // a different set from the grown blob, and the whole test is whether the blob
  // fills its own rectangle better than noise would.
  let cosTol = cos(u.toleranceRad);
  let hl = len * 0.5;
  // Floored, so a one-pixel-wide ridge -- the common case on a clean edge --
  // still tests a real strip instead of a zero-area rectangle nothing lands in.
  let hw = max(wid * 0.5, 0.5);

  var minX = 1e30; var maxX = -1e30; var minY = 1e30; var maxY = -1e30;
  for (var c = 0u; c < 4u; c = c + 1u) {
    let a = select(-hl, hl, c < 2u);
    let b = select(hw, -hw, c == 1u || c == 2u);
    let x = cx + a * ax + b * px;
    let y = cy + a * ay + b * py;
    minX = min(minX, x); maxX = max(maxX, x);
    minY = min(minY, y); maxY = max(maxY, y);
  }
  let x0 = u32(max(0.0, floor(minX)));
  let x1 = min(u.w - 1u, u32(max(0.0, ceil(maxX))));
  let y0 = u32(max(0.0, floor(minY)));
  let y1 = min(u.h - 1u, u32(max(0.0, ceil(maxY))));

  var n = 0u; var k = 0u;
  for (var y = y0; y <= y1; y = y + 1u) {
    for (var x = x0; x <= x1; x = x + 1u) {
      let dx = f32(x) - cx; let dy = f32(y) - cy;
      let proj = dx * ax + dy * ay;
      let perp = dx * px + dy * py;
      if (abs(proj) > hl + BOUNDARY_EPS || abs(perp) > hw + BOUNDARY_EPS) { continue; }
      n = n + 1u;
      let i = y * u.w + x;
      // Squared first, so a sub-rho pixel costs no sqrt and only a survivor is
      // normalized. A weak pixel counts toward n -- it was geometrically inside
      // -- but never toward k, because its direction is not worth trusting.
      let gx = fx[i]; let gy = fy[i];
      let m2 = gx * gx + gy * gy;
      if (m2 <= u.rhoSq) { continue; }
      // The level-line unit vector (-gy, gx)/|g| dotted with the rectangle axis.
      let alignDot = (-gy * ax + gx * ay) * inverseSqrt(m2);
      if (alignDot >= cosTol) { k = k + 1u; }
    }
  }

  // ── The binomial tail, as an ONLINE log-sum-exp ──
  //
  // log(sum_{j=k}^{n} C(n,j) p^j (1-p)^(n-j)). The binomial coefficient is
  // carried incrementally (C(n,j) = C(n,j-1)*(n-j+1)/j) rather than rebuilt per
  // term, and the sum is rescaled whenever a new maximum arrives. The two-pass
  // form -- buffer every term, then reduce against their max -- is not available
  // here: WGSL has no dynamic-length local array and n - k is a footprint area,
  // not a bounded quantity.
  //
  // ── THE RESCALE IS OVERFLOW PROTECTION AND NOTHING ELSE, AND NO TEST
  //    REACHES IT. MEASURED, NOT ASSUMED. ──
  //
  // Deleting the whole "if (logTerm > runningMax)" branch changes not one
  // number in any test, including the rendered-frame comparison against the
  // twin. That is not a gap in the tests: log-sum-exp is exact against ANY
  // reference point, so the running maximum only ever keeps exp() from
  // overflowing f32 at 88 nats.
  //
  // Reaching that needs the max term e^88 above the k-th, i.e. k roughly 13
  // sigma below the mean np. At n = 10,000 footprint pixels that means k < 231;
  // below n ~ 1,000 it cannot happen at all. So it takes a large blob region
  // with almost nothing aligned -- possible at 480x640, absent from every
  // fixture here. The branch stays because that frame is reachable and the
  // failure would be an inf, but nothing in the suite proves it right.
  let p = u.toleranceRad / PI;
  var logBinTail = 0.0; // log(1): P(X >= 0) is 1, so k == 0 needs no work
  if (k > 0u) {
    let logP = log(p); let log1mP = log(1.0 - p);
    var logChoose = 0.0;
    for (var i = 1u; i <= k; i = i + 1u) { logChoose = logChoose + log(f32(n - k + i) / f32(i)); }
    var runningMax = logChoose + f32(k) * logP + f32(n - k) * log1mP;
    var runningSum = 1.0;
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

  rects[o + 0u] = cx;
  rects[o + 1u] = cy;
  rects[o + 2u] = th;
  rects[o + 3u] = len;
  rects[o + 4u] = wid;
  rects[o + 5u] = logNfa / LN10;
  rects[o + 6u] = select(0.0, 1.0, logNfa < u.logEpsilon);
  rects[o + 7u] = 0.0;
  rects[o + 8u] = f32(n);
  rects[o + 9u] = f32(k);
}
`;


// ── S5 lines: accepted rectangles -> usable line segments ─────────────────
//
// A rectangle becomes a line when it is BOTH real (the fitter's NFA verdict)
// and long enough to be worth voting with. The two criteria belong to different
// stages -- one is the fitter's own conclusion about noise, the other is this
// stage's usability floor -- and the compaction that follows is what turns a
// sparse maxRegions-wide flag array into a dense array of segments.
//
// ── WHY THIS IS NOT FUSED INTO lsdFit, AND WHY IT DISPATCHES DIRECTLY ──
//
// Both halves of that are the same fact. `lineFlag` is the input to a prefix
// scan, and a scan's element count is fixed at ENCODE time -- the host cannot
// know the region count, so the scan runs over maxRegions. That makes `lineFlag`
// a TOTAL function over [0, maxRegions): every slot must be written every frame,
// or the tail past regionCount still holds the last frame's flags and the
// compaction emits lines from stale rectangles.
//
// So `lines.flag` dispatches over maxRegions DIRECTLY rather than off
// `regionArgs`, and writes 0 for every r past the region count. That is what
// makes it a total function, and it is why the buffer needs no clear.
//
// lsdFit cannot do this: it is a map over [0, regionCount) by construction, so a
// store at the bottom of that kernel can never reach the tail. See LSD_FIT_WGSL's
// header -- a fusion that changes a pass's iteration domain is not free.
//
// Failure still propagates without an indirect dispatch. Zero regions means
// every flag is 0, so the scan total is 0, `lines.emit` writes zero workgroups
// into `lineArgs`, and everything downstream dispatches nothing.
export const LINES_FLAG_WGSL = /* wgsl */ `
struct U { maxRegions: u32, maxLines: u32, minLengthPx: f32, pad0: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> rects: array<f32>;
@group(0) @binding(2) var<storage, read> counts: vec2<u32>;
@group(0) @binding(3) var<storage, read_write> lineFlag: array<vec2<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r >= u.maxRegions) { return; }
  // The .y lane of the scan is unused here. That wastes maxRegions*4 bytes and
  // buys one scan implementation in the pipeline instead of two.
  var keep = 0u;
  // NOT an early return. Every slot is written, including the tail past the
  // region count -- see the header. min() rather than counts.x, because an
  // overflowing frame only ever fitted maxRegions rectangles.
  if (r < min(counts.x, u.maxRegions)) {
    let o = r * 10u;
    // rects[o+6] is the accepted flag as 0.0/1.0, rects[o+3] the length.
    if (rects[o + 6u] > 0.5 && rects[o + 3u] >= u.minLengthPx) { keep = 1u; }
  }
  lineFlag[r] = vec2<u32>(keep, 0u);
}
`;

// The compaction, plus the indirect args every per-line pass downstream runs off.
//
// A rectangle's two endpoints are its centre plus and minus half its length
// along its own axis -- the same two points compositesFromLsdRectangles emits,
// and the reason the old `root` counter disappears here. That counter existed
// only so a second consumer indexing a DIFFERENT set could join back against
// this one; in a flat pipeline both consumers read this same compacted array, so
// there is nothing to join and nothing to number.
//
// Thread 0 writes the args BEFORE the guards, for the reason collect.regionMeta
// records: putting it after skips it whenever region 0 is not a line, which is
// the ordinary case.
export const LINES_EMIT_WGSL = /* wgsl */ `
struct U { maxRegions: u32, maxLines: u32, minLengthPx: f32, pad0: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> rects: array<f32>;
@group(0) @binding(2) var<storage, read> lineFlag: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> lineScan: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> lines: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> lineCount: vec2<u32>;
@group(0) @binding(6) var<storage, read_write> args: array<u32>;
@group(0) @binding(7) var<storage, read_write> status: atomic<u32>;

const LINE_OVERFLOW: u32 = 8u; // bit 3 of the status word

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  if (r == 0u) {
    // Clamped, so an overflowing frame votes with the lines that fit rather than
    // dispatching over a count no buffer is sized for. The bit is set HERE
    // rather than derived at finish because this is the pass that truncates --
    // and it is the only one holding maxLines.
    let lc = min(lineCount.x, u.maxLines);
    args[0] = (lc + 63u) / 64u;
    args[1] = 1u;
    args[2] = 1u;
    if (lineCount.x > u.maxLines) { atomicOr(&status, LINE_OVERFLOW); }
  }
  if (r >= u.maxRegions) { return; }
  if (lineFlag[r].x == 0u) { return; }
  let i = lineScan[r].x;
  if (i >= u.maxLines) { return; }

  let o = r * 10u;
  let cx = rects[o + 0u];
  let cy = rects[o + 1u];
  let hl = rects[o + 3u] * 0.5;
  let ax = cos(rects[o + 2u]);
  let ay = sin(rects[o + 2u]);
  lines[i] = vec4<f32>(cx + hl * ax, cy + hl * ay, cx - hl * ax, cy - hl * ay);
}
`;

// ── S5b votes: a line segment -> one statement about the floor ────────────
//
// Back-project the segment's two endpoints into rays. The plane they span passes
// through the camera centre and contains the whole segment, so if the segment
// really lies on a straight line on the floor, that plane's NORMAL is
// perpendicular to the floor direction the line runs along. One vote per line:
// "the floor contains a direction perpendicular to me."
//
// ── THERE IS NO ORIENTATION INPUT, AND THAT IS THE POINT ──
//
// The rays are cast in CAMERA SPACE and never rotated. The host version takes a
// quaternion and is always handed MATH_QUAT, which is the identity -- so the
// argument was carrying no information, and reproducing it here would import a
// frame convention this pipeline does not need. The camera's orientation is
// exactly what the NEXT stage computes from these votes; a vote that already
// knew it would be circular.
//
// ── THE WEIGHT IS THE CONFIDENCE, AND ALSO THE VALIDITY FLAG ──
//
// |r1 x r2| is sin of the angle between the endpoint rays: the segment's
// projected arc length on the unit sphere. A long, well-separated pair gives a
// large, well-conditioned normal; a short one gives a small normal easily
// dominated by endpoint noise. So the weight IS a confidence, explicitly, rather
// than something implied by a pixel count.
//
// The host version drops a degenerate vote with `continue`, which renumbers
// everything after it. Compaction is not available here without a second scan,
// so a degenerate line writes weight 0 instead -- equivalent for the fit, whose
// scatter matrix skips zero weights, and it keeps vote i joined to line i.
// EVERY CONSUMER MUST GATE ON weight > 0: gpp classifies lines by their vote
// normal, and a zero normal classifies into a family arbitrarily.
//
// `maxWeight` is an atomicMax over the FLOAT BITS. That is only monotonic
// because an arc length is non-negative -- the IEEE-754 bit pattern of a
// non-negative float orders the same way the float does, and does not for
// negatives. This is why zero is the right initial value here while `uvBounds`
// and `growArgs`, which look like the same pattern, need a written one.
//
// ── WHAT THE MUTATION RUN COULD NOT REACH, RECORDED RATHER THAN IMPLIED ──
//
// 1. THE DEGENERATE GUARD IS UNREACHABLE, AND THAT IS A TRUE PROPERTY. Deleting
//    it changes no number in any test. It cannot: a pinhole projection is
//    injective on image points, and lines.flag has already dropped anything
//    shorter than minLengthPx, so the two endpoint rays are separated by a
//    definite angle and arc has a hard lower bound. Length is zero only for a
//    region whose members all coincide, which is not a region. The branch stays
//    because minLengthPx is a user-facing slider that can be set to 0 and the
//    failure would be a NaN normal silently poisoning the fit -- but nothing
//    here proves it right, exactly like lsdFit's log-sum-exp rescale.
// 2. `maxWeight` HAS NO CONSUMER YET. fit.ata (§10) is the only reader and is
//    not built, so nothing in the suite observes this atomic at all. It is
//    written and untested; the first test of it arrives with the fit.
// 3. THE OVERFLOW PATH IS UNTESTED. lineOverflow needs more accepted lines than
//    maxLines, which no fixture here approaches.
export const VOTES_WGSL = /* wgsl */ `
struct U {
  w: u32, h: u32, maxLines: u32, pad0: u32,
  tanHalf: f32, aspect: f32, pad1: f32, pad2: f32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> lines: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lineCount: vec2<u32>;
@group(0) @binding(3) var<storage, read_write> votes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read_write> maxWeight: atomic<u32>;

// cornerDir with the identity quaternion, in camera space. The NDC mapping is
// decodeGridBuild's inverse exactly, which is also what the simulator casts
// with -- so a renderer/pipeline disagreement cannot hide in this conversion.
fn rayDir(px: f32, py: f32) -> vec3<f32> {
  let ndcU = (px / f32(u.w)) * 2.0 - 1.0;
  let ndcV = 1.0 - (py / f32(u.h)) * 2.0;
  return normalize(vec3<f32>(u.tanHalf * u.aspect * ndcU, u.tanHalf * ndcV, -1.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  // Indirect off lineArgs, which is already clamped -- but the last workgroup
  // still runs 64 threads whatever the line count is.
  if (i >= min(lineCount.x, u.maxLines)) { return; }
  let seg = lines[i];
  let n = cross(rayDir(seg.x, seg.y), rayDir(seg.z, seg.w));
  let arc = length(n);
  // Written in BOTH branches, so \`votes\` never needs a clear.
  if (arc < 1e-12) { votes[i] = vec4<f32>(0.0, 0.0, 0.0, 0.0); return; }
  // Normalized by the length already computed rather than by a second hypot.
  votes[i] = vec4<f32>(n / arc, arc);
  atomicMax(&maxWeight, bitcast<u32>(arc));
}
`;

// ── S6a fit.ata: every vote -> one 6x6 scatter matrix ─────────────────────
//
// The votes fall into two families -- normals from row lines and normals from
// column lines -- and nothing has labelled which is which. Rather than cluster
// first, fit a DEGENERATE QUADRIC: the algebraic surface expressing "every vote
// lies on one plane OR the other". A pair of planes through the origin is the
// zero set of a quadratic form, so this is least squares in the 6 unique
// coefficients of a symmetric 3x3 form, i.e. the null space of the 6-column
// design matrix whose row for a vote n is
//
//     (nx^2, ny^2, nz^2, nx*ny, nx*nz, ny*nz)
//
// This pass builds A^T W A over those rows. Only the upper triangle is unique
// (6*7/2 = 21), packed a-outer, b-inner-from-a; fit.eigen unpacks in exactly
// that order.
//
// ── ONE WORKGROUP, AND THE PARTIALS BUFFER IS DELETED ──
//
// src/pose splits this in two: one workgroup per 64 votes tree-reduces into its
// own 21-float row, and the ~256 rows are summed BACK ON THE HOST. The
// declaration inherited that split as `fit.ata` + `fit.reduce` over an
// `ataPartials` buffer. Re-deriving it, neither the split nor the buffer
// survives, for two independent reasons:
//
// 1. THE SECOND PASS ONLY EXISTED BECAUSE THE FIRST ONE'S CONSUMER WAS A HOST.
//    With no readback there is nothing to hand rows to. A single workgroup
//    striding over the votes accumulates the whole thing in registers and
//    tree-reduces once -- so the intermediate never exists rather than being
//    summed by a second kernel.
//
// 2. THE INDIRECT FORM IS §9's DEFECT, LIVE IN THE DECLARATION. `fit.ata` was
//    declared `indirectFrom: 'lineArgs'`, so it dispatched ceil(lineCount/64)
//    workgroups and wrote a PREFIX of `ataPartials`. `fit.reduce`'s extent is
//    fixed at ENCODE time -- the host does not know the line count -- so it
//    would have summed the tail too, and the tail is the previous frame's
//    partial sums. A busy frame followed by a quiet one would fit the union of
//    the two. Producer and consumer iterating different domains, exactly as
//    `lines.flag` and `gpp.classify`; the third instance found by re-derivation
//    rather than by a test.
//
// WHAT IT COSTS, said plainly: at the `maxLines` cap this is 16,384 votes on 64
// lanes, i.e. 256 votes and ~5,400 fused multiply-adds serially per lane, on ONE
// compute unit. That is a few microseconds and the observed vote count is ~70 to
// ~400, but it IS a real serialization and the two-pass form is the answer if it
// ever shows up in a profile -- with the domain fix, which the deleted version
// did not have.
//
// `maxWeight` normalization is a NO-OP for the fit: scaling every weight scales
// ATA by that constant and leaves its eigenvectors untouched. It is here because
// the accumulation is f32 and wants its summands in [0, 1].
//
// `ata` needs no clear. Thread 0 writes all 21 entries unconditionally, so it is
// a total function over its own range -- the property §16 says the scan totals
// have and the scan INPUTS do not.
export const FIT_ATA_WGSL = /* wgsl */ `
struct U { maxLines: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> votes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lineCount: vec2<u32>;
// The same buffer votes.cast atomicMax'd, read here as plain float bits.
@group(0) @binding(3) var<storage, read> maxWeightBits: u32;
@group(0) @binding(4) var<storage, read_write> ata: array<f32, 21>;

const LANES: u32 = 64u;
var<workgroup> partial: array<array<f32, 21>, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let li = lid.x;
  var acc: array<f32, 21>;
  for (var k = 0u; k < 21u; k = k + 1u) { acc[k] = 0.0; }

  // The count is DEVICE-side; the cap is the only thing the host contributes.
  let count = min(lineCount.x, u.maxLines);
  let maxW = bitcast<f32>(maxWeightBits);
  let inv = select(0.0, 1.0 / maxW, maxW > 0.0);

  for (var i = li; i < count; i = i + LANES) {
    let v = votes[i];
    let w = v.w * inv;
    // A degenerate vote is written with weight 0 rather than dropped (see
    // VOTES_WGSL). Gating here is what makes that equivalent to dropping it.
    if (w <= 0.0) { continue; }
    var row: array<f32, 6>;
    row[0] = v.x * v.x; row[1] = v.y * v.y; row[2] = v.z * v.z;
    row[3] = v.x * v.y; row[4] = v.x * v.z; row[5] = v.y * v.z;
    var idx = 0u;
    for (var a = 0u; a < 6u; a = a + 1u) {
      let wra = w * row[a];
      for (var b = a; b < 6u; b = b + 1u) {
        acc[idx] = acc[idx] + wra * row[b];
        idx = idx + 1u;
      }
    }
  }

  for (var k = 0u; k < 21u; k = k + 1u) { partial[li][k] = acc[k]; }
  workgroupBarrier();

  var stride = LANES / 2u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) {
      for (var k = 0u; k < 21u; k = k + 1u) {
        partial[li][k] = partial[li][k] + partial[li + stride][k];
      }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (li == 0u) {
    for (var k = 0u; k < 21u; k = k + 1u) { ata[k] = partial[0][k]; }
  }
}
`;

// ── S6b fit.eigen: the scatter matrix -> the floor's triad ────────────────
//
// Smallest eigenvector of the 6x6 gives the quadratic form's coefficients; that
// form reshaped as a symmetric 3x3 has one near-zero eigenvalue whose
// eigenvector is the FLOOR NORMAL, and the other two eigenvectors b1, b2 are the
// bisectors of the plane pair -- so the two plane normals are b1 +/- b2, which
// are the floor's own in-plane axes.
//
// Fixed sizes throughout: 6x6 then 3x3, whatever the vote count. One thread.
//
// ── ONE JACOBI, PARAMETERIZED BY n, ON A SHARED 6x6 ARENA ──
//
// The host has two call sites at two sizes and one generic implementation
// (linalg.ts). WGSL has no generics, so the tempting port is two copies -- which
// is exactly the duplication §10 records this code growing once before, when the
// GPU path carried a verbatim 20-line copy of the eigen tail. It is avoidable
// without generics: the routine takes `n` and indexes at STRIDE n inside one
// array<f32,36>, so the 3x3 run reuses the first 9 slots. Two calls, one body.
//
// ── THREE THINGS THAT ARE NOT TRANSCRIPTIONS, AND WHY ──
//
// 1. THE CONVERGENCE TEST IS RELATIVE, NOT ABSOLUTE. linalg.ts breaks when the
//    off-diagonal sum of squares falls below 1e-30, which is an f64 threshold.
//    In f32 the off-diagonals bottom out around eps*scale ~ 1e-7*scale, so a sum
//    of squares of 1e-30 is unreachable for any matrix of ordinary magnitude and
//    the loop would simply run its full sweep budget every frame. The threshold
//    here is eps^2 against the matrix's own Frobenius norm, which is the same
//    intent expressed in the precision this actually runs in.
//
// 2. THE ROTATION GUARDS AN OVERFLOW THE f64 VERSION CANNOT HIT. `theta*theta`
//    overflows f32 at |theta| ~ 1.8e19, and theta is a ratio whose denominator
//    is a nearly-zero off-diagonal, so it grows without bound as the sweep
//    converges. Past 1e10 the exact asymptote t -> 1/(2*theta) is used instead.
//    In f64 the same expression has 300 more decades of headroom.
//
// 3. THE ORIENTATION TEST IS A SIGN, NOT A RAY CAST. Which way the normal points
//    is not determined by the fit -- an eigenvector's sign is arbitrary -- so it
//    has to be chosen. src/pose chooses it TWICE, independently, from the same
//    arbitrary vector: poseCompute casts cornerDir(0,0) to orient a local copy
//    for the handedness test, and gridPeriodPhase casts the same ray again to
//    orient its own. Deciding it once, here, at the only place the sign is
//    created, deletes both -- and the test collapses to one comparison, because
//    the camera looks down -z, so "the normal is on the camera's side" IS
//    `Dnormal.z > 0`. That is also STRICTLY MORE ROBUST than the ray it
//    replaces: cornerDir(0,0) is the image corner, which points at SKY before
//    the view axis does, so the corner test inverts the normal at grazing tilts
//    where the centre test is still correct. The centre ray is the last one to
//    cross the horizon.
//
// ── THE DECLARED DEGENERACY GUARD CANNOT FIRE, AND THE REAL ONE IS ELSEWHERE ──
//
// §10 and votes.ts guard `|b1+b2|^2 < 1e-9` before normalizing. That test is
// unreachable, and not in the "no fixture reaches it" sense: Jacobi accumulates
// its rotations into a matrix that starts as the identity, so V is ORTHOGONAL by
// construction and b1, b2 are orthonormal -- |b1 +/- b2|^2 is identically 2. It
// is dead in exact arithmetic, so it is not ported.
//
// What IS reachable, and what src/pose does not catch, is a scatter matrix of
// all zeros -- no lines, or every vote degenerate. Then every eigenvalue is 0,
// the smallest eigenvector is whichever identity column comes first, and the
// pipeline gets a confident triad built from nothing. That is the condition
// `fitDegenerate` reports here.
//
// (The triad is orthonormal BY CONSTRUCTION for the same reason the dead guard
// was dead: b1.b2 = 0 makes (b1+b2).(b1-b2) = |b1|^2 - |b2|^2 = 0, and both are
// perpendicular to the third eigenvector. So this fit cannot report its own
// misfit -- the residual is invisible in the output. Known, and out of scope
// for a port.)
//
// ── THE MUTATION RUN, AND WHAT IT FOUND ──
//
// Ten deliberate bugs across both passes. Eight caught, two not, and the two are
// true properties rather than gaps.
//
//   transposed unpack of the 21 entries      caught by both fixtures
//   3x3 cross terms not halved               caught by both fixtures
//   3x3 zero eigenvalue picked by SIGN       caught by both fixtures
//   one Jacobi sweep instead of 32           caught by both fixtures
//   tree reduction skips half the lanes      caught by the rendered frame only
//   no all-zero-scatter guard                caught by the empty-vote fixture
//   NO NORMAL ORIENTATION                    caught ONLY after a fixture was built
//   NO HANDEDNESS FLIP                       caught ONLY after a fixture was built
//   6x6 smallest by MAGNITUDE not signed     NOTHING -- and correctly so
//   no maxWeight normalization               NOTHING -- and correctly so
//
// THE TWO SIGN CONVENTIONS WERE BOTH INVISIBLE, and the reason generalizes:
// Jacobi's eigenvector signs are deterministic but arbitrary, and for an
// ordinary oblique view they come out already-correct -- so deleting either flip
// changed no number anywhere, including on a whole rendered frame, while the
// assertions written to check them passed on evidence they never saw. Finding
// inputs where the RAW output is wrong took disabling the flips and scanning 132
// poses. Across tilts 0-40 with no roll, 0 of 77 poses reach either branch;
// tilt 50 yaw 35 is left-handed, and the same pose with roll 245 puts the normal
// at z = -0.643. Both fixtures are now in the test, named.
//
// The corollary is the same one §8 and §9 produced: a fixture on the symmetry a
// mutation acts on cannot see it, and an assertion is not a test until something
// has made it fail.
//
// THE TWO SURVIVORS ARE TRUE PROPERTIES. ATA is a Gram matrix, so its
// eigenvalues are non-negative and the smallest by magnitude IS the smallest --
// the signed form is kept only because the two can differ through round-off at
// exactly the entry being selected, which is where matching linalg.ts is free.
// And normalizing by maxWeight scales ATA by a constant, which leaves its
// eigenvectors untouched; the doc has always said so, and this is the first
// thing to verify it. It also means `maxWeight` is STILL not observed by
// anything, even by its only consumer -- VOTES_WGSL's note that "the first test
// of it arrives with the fit" is wrong, and the fit cannot be the test.
export const FIT_EIGEN_WGSL = /* wgsl */ `
@group(0) @binding(0) var<storage, read> ata: array<f32, 21>;
@group(0) @binding(1) var<storage, read_write> triad: array<vec3<f32>, 3>;
@group(0) @binding(2) var<storage, read_write> status: atomic<u32>;

const FIT_DEGENERATE: u32 = 32u; // bit 5 of the status word

// One n x n symmetric matrix and its accumulated rotations, both at stride n
// inside a fixed 6x6 arena. workgroup_size is 1, so private storage is the whole
// state and no pointers have to be threaded through.
var<private> A: array<f32, 36>;
var<private> V: array<f32, 36>;

// Cyclic Jacobi. On return A's diagonal holds the eigenvalues and V's COLUMNS
// hold the eigenvectors: eigenvector k is (V[0*n+k], V[1*n+k], ...).
fn jacobi(n: u32) {
  for (var i = 0u; i < n * n; i = i + 1u) { V[i] = 0.0; }
  for (var i = 0u; i < n; i = i + 1u) { V[i * n + i] = 1.0; }

  var norm2 = 0.0;
  for (var i = 0u; i < n * n; i = i + 1u) { norm2 = norm2 + A[i] * A[i]; }
  if (norm2 <= 0.0) { return; }
  // f32 eps^2, relative to the matrix's own scale -- see this stage's header.
  let tol = 1e-14 * norm2;

  for (var sweep = 0u; sweep < 32u; sweep = sweep + 1u) {
    var off = 0.0;
    for (var p = 0u; p < n; p = p + 1u) {
      for (var q = p + 1u; q < n; q = q + 1u) { off = off + A[p * n + q] * A[p * n + q]; }
    }
    if (off <= tol) { break; }

    for (var p = 0u; p < n; p = p + 1u) {
      for (var q = p + 1u; q < n; q = q + 1u) {
        let apq = A[p * n + q];
        if (apq == 0.0) { continue; }
        let app = A[p * n + p];
        let aqq = A[q * n + q];
        let theta = (aqq - app) / (2.0 * apq);
        var t: f32;
        if (abs(theta) > 1e10) {
          // The exact asymptote. theta*theta would overflow f32 near 1.8e19.
          t = 1.0 / (2.0 * theta);
        } else {
          let sgn = select(1.0, sign(theta), theta != 0.0);
          t = sgn / (abs(theta) + sqrt(theta * theta + 1.0));
        }
        let c = 1.0 / sqrt(t * t + 1.0);
        let s = t * c;

        A[p * n + p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        A[q * n + q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        A[p * n + q] = 0.0;
        A[q * n + p] = 0.0;
        for (var k = 0u; k < n; k = k + 1u) {
          if (k != p && k != q) {
            let akp = A[k * n + p];
            let akq = A[k * n + q];
            A[k * n + p] = c * akp - s * akq; A[p * n + k] = A[k * n + p];
            A[k * n + q] = s * akp + c * akq; A[q * n + k] = A[k * n + q];
          }
        }
        for (var k = 0u; k < n; k = k + 1u) {
          let vkp = V[k * n + p];
          let vkq = V[k * n + q];
          V[k * n + p] = c * vkp - s * vkq;
          V[k * n + q] = s * vkp + c * vkq;
        }
      }
    }
  }
}

@compute @workgroup_size(1)
fn main() {
  // Unpack the upper triangle into a full symmetric 6x6, in the same
  // a-outer/b-inner-from-a order fit.ata packed it.
  var idx = 0u;
  var norm2 = 0.0;
  for (var a = 0u; a < 6u; a = a + 1u) {
    for (var b = a; b < 6u; b = b + 1u) {
      let e = ata[idx];
      A[a * 6u + b] = e;
      A[b * 6u + a] = e;
      norm2 = norm2 + e * e;
      idx = idx + 1u;
    }
  }

  // NO EVIDENCE AT ALL: no lines, or every vote degenerate. Every eigenvalue is
  // then 0 and the "smallest eigenvector" is an identity column, which yields a
  // confident-looking triad built from nothing. Written in this branch too, so
  // triad never carries the previous frame's axes.
  if (!(norm2 > 0.0)) {
    atomicOr(&status, FIT_DEGENERATE);
    triad[0] = vec3<f32>(1.0, 0.0, 0.0);
    triad[1] = vec3<f32>(0.0, 1.0, 0.0);
    triad[2] = vec3<f32>(0.0, 0.0, 1.0);
    return;
  }

  jacobi(6u);
  // The least-squares null space: SIGNED smallest, matching linalg.ts. For a
  // Gram matrix the eigenvalues are non-negative anyway, so signed and absolute
  // agree except through round-off near zero -- which is precisely the entry
  // being selected, so the tie-break is worth keeping identical.
  var mi = 0u;
  for (var i = 1u; i < 6u; i = i + 1u) {
    if (A[i * 6u + i] < A[mi * 6u + mi]) { mi = i; }
  }
  var m: array<f32, 6>;
  for (var i = 0u; i < 6u; i = i + 1u) { m[i] = V[i * 6u + mi]; }

  // Those 6 coefficients ARE a symmetric 3x3 quadratic form. The cross terms
  // halve because m[3] is the coefficient of nx*ny, which the matrix form
  // counts twice.
  A[0] = m[0];       A[1] = m[3] * 0.5; A[2] = m[4] * 0.5;
  A[3] = m[3] * 0.5; A[4] = m[1];       A[5] = m[5] * 0.5;
  A[6] = m[4] * 0.5; A[7] = m[5] * 0.5; A[8] = m[2];
  jacobi(3u);

  // Here the near-zero eigenvalue is the one wanted, and its SIGN is arbitrary
  // -- so this one is by magnitude.
  var zi = 0u;
  for (var i = 1u; i < 3u; i = i + 1u) {
    if (abs(A[i * 3u + i]) < abs(A[zi * 3u + zi])) { zi = i; }
  }

  var b: array<vec3<f32>, 2>;
  var bi = 0u;
  var nrm = vec3<f32>(0.0, 0.0, 0.0);
  for (var i = 0u; i < 3u; i = i + 1u) {
    let col = vec3<f32>(V[0u * 3u + i], V[1u * 3u + i], V[2u * 3u + i]);
    if (i == zi) { nrm = col; } else { b[bi] = col; bi = bi + 1u; }
  }

  var Dn = normalize(nrm);
  // The camera looks down -z, so the normal is on the camera's side exactly
  // when its z is positive. This is the ONLY place the normal's sign is
  // decided; see the header for the two ray casts it replaces.
  if (Dn.z < 0.0) { Dn = -Dn; }

  // b1 and b2 are orthonormal, so these have norm sqrt(2) always -- which is
  // why the host's degeneracy guard here is unreachable and is not ported.
  var Drow = normalize(b[0] + b[1]);
  var Dcol = normalize(b[0] - b[1]);
  // A CONSISTENTLY RIGHT-HANDED TRIAD, and it is load-bearing much later: §14's
  // closed-form camera quaternion rotates (Dcol, Drow) through the winning
  // decode orientation and relies on the third axis needing no independent
  // correction.
  if (dot(cross(Drow, Dcol), Dn) > 0.0) { Dcol = -Dcol; }

  triad[0] = Drow;
  triad[1] = Dcol;
  triad[2] = Dn;
}
`;

// ── S7a gpp.classify: a line -> one point on a periodic 1D lattice ────────
//
// This opens Act II. Orientation is known; what is not known is the grid's
// SPACING (which gives camera height) or its OFFSET (which gives where cell
// boundaries fall). Both are properties of a 1D point set, and this pass is what
// turns each detected line into one point in it.
//
// ── THE RECTIFICATION, AND WHY IT IS THE POINT-FORM AND NOT THE NORMAL-FORM ──
//
// Gnomonic projection of a ray onto the plane tangent to the unit sphere at
// -Dnormal:
//
//     gnomonic(r) = ( -dot(r,Drow)/dot(r,Dnormal), -dot(r,Dcol)/dot(r,Dnormal) )
//
// {Drow, Dcol, Dnormal} is orthonormal, so "perpendicular to Dnormal" already IS
// span{Drow, Dcol} and the triad doubles as the tangent plane's own basis for
// free. Under it, perspective distortion is undone: a floor point at camera-space
// offset (u along Drow, v along Dcol) at height H projects to exactly (u/H, v/H).
// So a line running ALONG Drow has the same xCol at every point on it, and that
// shared scalar IS its position in the periodic sequence.
//
// There is a second, algebraically tempting form -- rectify from the line's own
// vote NORMAL rather than from a point on it. Both are affine-linear in the
// line's true offset, but with DIFFERENT slope and intercept, so they are not
// interchangeable and mixing them silently misaligns period against phase. This
// pass uses the point form for `value` and the normal form ONLY to classify,
// which is legitimate because classification reads orientation alone.
//
// ── THE PERIOD FALLS OUT OF THIS DIRECTLY ──
//
// Grid lines are one cellPitch apart in the world, and the projection divides by
// H, so consecutive values are cellPitch/H apart. Recovering that spacing IS
// recovering the height. Nothing later in this stage does anything but find the
// period and phase of the points this pass emits.
//
// ── THERE IS NO QUATERNION HERE, AND src/pose's VERSION HAS TWO ──
//
// `computeGridPeriodPhase` takes the camera quaternion, casts world-space rays
// with it, and `makeCellCentreDistinctness` carries its inverse to get back to
// pixels. Neither is needed. votes.cast casts in CAMERA space, the triad is
// derived from those votes, so the triad is camera-space too -- and a gnomonic
// projection of camera-space rays onto a camera-space triad never leaves camera
// space. gpp.distinct's reprojection is then a bare pinhole with no rotation at
// all. Two quaternion round-trips deleted by the frame being consistent rather
// than by an optimization.
//
// ── DO NOT RE-ORIENT THE NORMAL. §10 ALREADY DID IT ──
//
// `computeGridPeriodPhase` opens by casting cornerDir(0,0) and negating a local
// copy of Dnormal if the dot is positive, because it is handed the RAW
// eigenvector whose sign the fit does not determine. Here fit.eigen decided that
// sign at the point it was created, so triad[2] arrives already pointing at the
// camera and re-orienting it would be a SECOND flip.
//
// One consequence to know before a number looks wrong: src/pose passes the RAW
// normal to gnomonic() and uses the oriented one only for its grazing gate, so
// `value` here may come out globally NEGATED relative to src/pose's. That is
// self-consistent -- a period is a spacing, and phase and the decode lattice are
// derived from these same coordinates -- but it means a value-for-value
// comparison against the old implementation is the wrong check. Ground truth is.
//
// ── DIRECT OVER maxLines, FOR THE REASON lines.flag IS ──
//
// `family` is a SCAN INPUT, so it must be a total function over the scanned
// range, and the scan's element count is fixed at encode time at maxLines. The
// declaration had this pass `indirectFrom: 'lineArgs'` -- writing [0, lineCount)
// and leaving the tail holding the previous frame's families. Third instance of
// that defect (after lineFlag and ataPartials) and, like both of those, found by
// re-deriving rather than by a test.
//
// The tail must be written vec2(0,0) and NOT vec2(0,1). `family` is
// (isRow, 1-isRow), so the obvious "not a row line, therefore a column line"
// encoding would count every one of the ~16,000 dead slots into the column
// family and put colCount at maxLines. Non-membership and column-membership are
// different facts and the pair has to be able to say so.
//
// ── THIS PASS ALSO EMITS THE DECODE LATTICE'S BOUNDS (§12) ──
//
// The decode lattice is bounded by the detected LINES rather than by the view
// quadrilateral -- measured 2026-08-14, and the argument is that a grid line is
// below the horizon by construction (you cannot detect one in sky) while an
// image corner is not, so the 49x49 ray grid and its grazing gate dissolve
// rather than being worked around.
//
// The quantity that needs is min/max over every classified endpoint's gnomonic
// (xRow, xCol), and this pass is where all four numbers exist: it computes both
// endpoints' full coordinates and, before this, threw three quarters of them
// away. So `samples` carries a third and fourth lane -- the span of the OTHER
// coordinate across this line's two endpoints -- and gpp.extent reduces them.
//
// Why the other coordinate only: a row line's own value IS its xCol, constant
// along the line by construction, so the value lane already bounds that axis for
// the row family. The cross lane is the axis the line RUNS ALONG, which is the
// half a per-family extent structurally cannot see. Using the averaged `value`
// rather than the two endpoints' own xCol loses the sub-pixel disagreement
// between them -- immaterial against a lattice quantized to whole cells and
// clamped to 144 of them, and stated because it is a real (tiny) difference from
// the measured harness, which bounded p1/p2 in both axes.
//
// There is deliberately NO grazing gate on the endpoints, matching what was
// measured. A line detected near the horizon projects arbitrarily far and can
// push the hull past the truly visible extent -- observed at h=10 tilt 45-55.
// decode.layout's 144 clamp covers it; gating the endpoints on grazing is the
// principled alternative and is unmeasured.
export const GPP_CLASSIFY_WGSL = /* wgsl */ `
struct U {
  w: u32, h: u32, maxLines: u32, nMin: u32,
  tanHalf: f32, aspect: f32, cellPitch: f32, boardCells: u32,
  nCap: u32, heightFactor: f32, candCount: u32, significance: f32,
  topK: u32, patchM: u32, minGrazingCos: f32, pad0: u32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> lines: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> lineCount: vec2<u32>;
@group(0) @binding(3) var<storage, read> votes: array<vec4<f32>>;
@group(0) @binding(4) var<storage, read> triad: array<vec3<f32>, 3>;
@group(0) @binding(5) var<storage, read_write> samples: array<vec4<f32>>;
@group(0) @binding(6) var<storage, read_write> family: array<vec2<u32>>;

// Byte-identical to VOTES_WGSL's, deliberately: these two passes must agree on
// which ray an endpoint is, or the normal a line is classified by and the point
// it is rectified from describe different lines.
fn rayDir(px: f32, py: f32) -> vec3<f32> {
  let ndcU = (px / f32(u.w)) * 2.0 - 1.0;
  let ndcV = 1.0 - (py / f32(u.h)) * 2.0;
  return normalize(vec3<f32>(u.tanHalf * u.aspect * ndcU, u.tanHalf * ndcV, -1.0));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.maxLines) { return; }

  // Every one of the three outputs is written on every path. \`family\` needs it
  // because it is a scan input; the other two ride along for free and it means
  // no consumer can read a stale value at an index it should not have reached.
  var value = 0.0;
  var weight = 0.0;
  var fam = vec2<u32>(0u, 0u);
  // Neutral under the min/max gpp.extent reduces these with, so a slot no family
  // ever compacts cannot widen the hull. f32::MAX and not an infinity for the
  // reason GPP_EXTENT_WGSL gives: WGSL rejects inf outright.
  var cross = vec2<f32>(3.4028234e38, -3.4028234e38);

  if (i < min(lineCount.x, u.maxLines)) {
    let v = votes[i];
    // A degenerate line was written with weight 0 rather than dropped, so its
    // normal is the zero vector -- which would classify into a family by
    // whichever side of an equality the hardware lands on. VOTES_WGSL's header
    // names this pass specifically.
    if (v.w > 0.0) {
      let Drow = triad[0];
      let Dcol = triad[1];
      let Dn = triad[2];
      let seg = lines[i];
      let r1 = rayDir(seg.x, seg.y);
      let r2 = rayDir(seg.z, seg.w);
      let d1 = dot(r1, Dn);
      let d2 = dot(r2, Dn);
      // A ray nearly parallel to the tangent plane has no gnomonic image. It
      // should not happen for a ray that hit the floor -- but the guard is what
      // stops a division by ~0 from writing an inf into a scan-adjacent array,
      // and unlike votes.cast's arc guard this one is genuinely reachable at
      // grazing incidence.
      if (abs(d1) > 1e-9 && abs(d2) > 1e-9) {
        // Both endpoints, both coordinates. The averages below are what the
        // period search wants; the SPANS are what the decode hull wants.
        let xRow1 = -dot(r1, Drow) / d1;
        let xRow2 = -dot(r2, Drow) / d2;
        let xCol1 = -dot(r1, Dcol) / d1;
        let xCol2 = -dot(r2, Dcol) / d2;
        let xRow = (xRow1 + xRow2) * 0.5;
        let xCol = (xCol1 + xCol2) * 0.5;
        // Orientation only, and the SAME test the plane-pair fit relies on: a
        // line running along Drow sweeps a plane whose normal is perpendicular
        // to Drow. Averaging the two endpoints is a free noise bonus -- in exact
        // arithmetic they are already identical, which is what "projects to a
        // constant-xCol line" means.
        let isRow = abs(dot(v.xyz, Drow)) < abs(dot(v.xyz, Dcol));
        value = select(xRow, xCol, isRow);
        weight = v.w;
        fam = select(vec2<u32>(0u, 1u), vec2<u32>(1u, 0u), isRow);
        // The coordinate the line RUNS ALONG, which is the one its own value
        // holds constant and therefore says nothing about. A row line runs along
        // Drow, so its cross coordinate is xRow.
        let along = select(vec2<f32>(xCol1, xCol2), vec2<f32>(xRow1, xRow2), isRow);
        cross = vec2<f32>(min(along.x, along.y), max(along.x, along.y));
      }
    }
  }

  samples[i] = vec4<f32>(value, weight, cross.x, cross.y);
  family[i] = fam;
}
`;

// ── S7b gpp.compact: two families, one scan ──────────────────────────────
//
// The exclusive prefix of (isRow, 1-isRow) carries BOTH compaction indices at
// once: .x counts row lines before i, .y counts column lines before i. So one
// vec2 scan replaces the two u32 scans this stage was declared with, and its
// grand total is exactly [rowCount, colCount] -- the third and last use of the
// shared primitive (§7).
//
// Direct over maxLines rather than indirect off lineArgs, matching classify. It
// would be correct either way -- past the line count every family is (0,0) and
// nothing writes -- but the two passes iterating the same domain is worth more
// than ~16,000 no-op threads cost.
export const GPP_COMPACT_WGSL = /* wgsl */ `
struct U {
  w: u32, h: u32, maxLines: u32, nMin: u32,
  tanHalf: f32, aspect: f32, cellPitch: f32, boardCells: u32,
  nCap: u32, heightFactor: f32, candCount: u32, significance: f32,
  topK: u32, patchM: u32, minGrazingCos: f32, pad0: u32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> samples: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> family: array<vec2<u32>>;
@group(0) @binding(3) var<storage, read> familyScan: array<vec2<u32>>;
@group(0) @binding(4) var<storage, read_write> rowSamples: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read_write> colSamples: array<vec4<f32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.maxLines) { return; }
  let f = family[i];
  let s = familyScan[i];
  // Both destinations are maxLines long and each family is a subset of the
  // lines, so neither index can escape. No clamp, and no tail to write: the
  // consumers below iterate familyCounts, never the array length.
  if (f.x == 1u) {
    rowSamples[s.x] = samples[i];
  } else if (f.y == 1u) {
    colSamples[s.y] = samples[i];
  }
}
`;

// ── S7c gpp.extent: how far the detected lines reach, PER FAMILY ─────────
//
// The period search needs a numerator. A detected line IS a grid line, so the
// two extreme lines of a family sit an INTEGER number of periods apart -- which
// is what makes the candidate set P_n = spread/n physical rather than a sampled
// guess, and what deletes the seed, the bracket width and the sample count that
// a seeded search would need to be told.
//
// ── PER FAMILY, AND THIS IS A BUG src/pose ALREADY FIXED ONCE ──
//
// A row line's value is its xCol; a column line's is its xRow. Two DIFFERENT
// axes. Pooling them and taking a global min/max measures the extent of the
// union of two unrelated coordinate sets, which exceeds either one whenever the
// nadir is off-centre asymmetrically between the axes -- generic for anything
// oblique. Since the candidate bound below is a per-family cell count, the
// inflated numerator pushed the true period out of the bracket entirely and the
// search never evaluated it.
//
// ── A REDUCTION, NOT AN ATOMIC, AND THAT DELETES A PASS ──
//
// The declaration had `gpp.extentInit` writing +/-inf into an atomic min/max
// target, the same shape as decode's `uvBounds`. It does not need to be one.
// `value` is SIGNED, so the atomicMax-over-float-bits trick `maxWeight` uses is
// not available here at all -- an ordered u32 encoding would be needed first
// (flip the sign bit for non-negatives, flip every bit for negatives), which is
// three lines of bit twiddling plus its inverse plus an init pass plus a
// declaration entry marked `written` rather than `zero`.
//
// One workgroup striding over the compacted arrays and tree-reducing gets the
// same answer with none of it: lane 0 writes all of `extent` unconditionally, so
// it is total by construction and needs neither clear nor init. Same argument
// fit.ata used to delete `ataPartials` -- and the same shape, which is why it is
// worth naming: an initialization pass is usually a symptom of choosing an
// atomic where a reduction would do.
//
// The +/-inf sentinels stay, they just live in registers now. A family with no
// lines leaves them untouched, which is why nothing downstream may read
// rowMin/rowMax without first checking the count -- `spread` and the hull below
// are the only values that are safe unconditionally.
//
// ── AND IT ALSO REDUCES THE DECODE HULL (§12) ──
//
// Four more lanes, no extra binding and no extra buffer: `samples` carries each
// line's cross-axis span (GPP_CLASSIFY_WGSL), gpp.compact already moves it into
// the two family arrays, and this reduction already walks both of them.
//
// The per-family extent and the hull are NOT the same measurement and neither
// implies the other, which is worth being explicit about since they now come out
// of one traversal. The extent bounds each family along ITS OWN coordinate --
// how far apart the outermost parallel lines of one direction are, the period
// search's numerator. The hull bounds each AXIS over every classified endpoint,
// including the direction each line runs along, which is the half no per-family
// value can see: a row line's value pins its xCol and says nothing at all about
// how far along xRow it extends.
//
// So xRow is bounded by (column values, row crosses) and xCol by (row values,
// column crosses). One line of either family already bounds both axes.
export const GPP_EXTENT_WGSL = /* wgsl */ `
struct U {
  w: u32, h: u32, maxLines: u32, nMin: u32,
  tanHalf: f32, aspect: f32, cellPitch: f32, boardCells: u32,
  nCap: u32, heightFactor: f32, candCount: u32, significance: f32,
  topK: u32, patchM: u32, minGrazingCos: f32, pad0: u32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> rowSamples: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> colSamples: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> familyCounts: vec2<u32>;
@group(0) @binding(4) var<storage, read_write> extent: array<f32, 12>;
@group(0) @binding(5) var<storage, read_write> status: atomic<u32>;

const LANES: u32 = 256u;
const GPP_NO_SAMPLES: u32 = 64u; // bit 6 of the status word

// Four quantities reduced in one traversal, and the lanes are:
//   .x  row family VALUE   -- an xCol, since a row line's constant coordinate is
//   .y  col family VALUE   -- an xRow
//   .z  row family CROSS   -- an xRow, the axis a row line runs along
//   .w  col family CROSS   -- an xCol
// The first two are the period search's per-family extent. All four together are
// the decode hull, because each AXIS is bounded by one family's value and the
// other family's cross: xRow by (.y, .z), xCol by (.x, .w).
var<workgroup> wlo: array<vec4<f32>, 256>;
var<workgroup> whi: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let li = lid.x;
  // f32::MAX and not an infinity: WGSL rejects inf as an f32 value outright,
  // including through a const-evaluated bitcast of its bit pattern. A finite
  // sentinel is enough anyway -- the classify guard bounds |value| by 1e9, since
  // a gnomonic coordinate is a unit-vector dot over a denominator held above
  // 1e-9 -- and it has the side benefit that no arithmetic on an untouched
  // family can produce a NaN.
  let big = 3.4028234e38;
  var lo = vec4<f32>(big, big, big, big);
  var hi = vec4<f32>(-big, -big, -big, -big);

  // Clamped for the same reason every per-line loop is: an overflowing frame
  // compacted at most maxLines values whatever the count says.
  let rc = min(familyCounts.x, u.maxLines);
  let cc = min(familyCounts.y, u.maxLines);
  for (var i = li; i < rc; i = i + LANES) {
    let s = rowSamples[i];
    lo.x = min(lo.x, s.x);
    hi.x = max(hi.x, s.x);
    lo.z = min(lo.z, s.z);
    hi.z = max(hi.z, s.w);
  }
  for (var i = li; i < cc; i = i + LANES) {
    let s = colSamples[i];
    lo.y = min(lo.y, s.x);
    hi.y = max(hi.y, s.x);
    lo.w = min(lo.w, s.z);
    hi.w = max(hi.w, s.w);
  }

  wlo[li] = lo;
  whi[li] = hi;
  workgroupBarrier();

  var stride = LANES / 2u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) {
      wlo[li] = min(wlo[li], wlo[li + stride]);
      whi[li] = max(whi[li], whi[li + stride]);
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (li == 0u) {
    let L = wlo[0]; let Hh = whi[0];
    let rlo = L.x; let rhi = Hh.x;
    let clo = L.y; let chi = Hh.y;
    // A single line has a position but no extent, so it contributes no
    // numerator -- matching extentOf's own "fewer than two values is zero".
    // This is also what keeps the inf sentinels from leaking into spread.
    let rs = select(0.0, rhi - rlo, rc >= 2u);
    let cs = select(0.0, chi - clo, cc >= 2u);
    let spread = max(rs, cs);
    extent[0] = rlo; extent[1] = rhi;
    extent[2] = clo; extent[3] = chi;
    extent[4] = spread;

    // THE DECODE HULL. Each axis takes one family's value and the other
    // family's cross, so ONE classified line of either family already bounds
    // both axes -- a row line's own xCol plus the xRow it spans. With no lines
    // at all every lane is still a sentinel, and writing those out would hand
    // decode.layout a lattice 1e38 cells wide; zeros make its max > min guard
    // fire instead, which is the condition it already has to test for.
    let any = (rc + cc) >= 1u;
    extent[5] = select(0.0, min(L.y, L.z), any);
    extent[6] = select(0.0, max(Hh.y, Hh.z), any);
    extent[7] = select(0.0, min(L.x, L.w), any);
    extent[8] = select(0.0, max(Hh.x, Hh.w), any);
    extent[9] = 0.0; extent[10] = 0.0; extent[11] = 0.0;
    // No extent means no periodicity to find: every line coincident, or no
    // lines at all. Ordinary rather than a defect -- an undecodable frame --
    // but everything downstream would divide by it.
    if (!(spread > 1e-9)) { atomicOr(&status, GPP_NO_SAMPLES); }
  }
}
`;

// ── The gpp uniform, shared by all six of its entry points ───────────────
//
// One struct rather than six, because every field is a frame constant and the
// alternative is six near-identical blocks that drift. `patchM` is the only one
// a shader clamps rather than trusting -- gpp.distinct's row buffers are fixed
// size, so it takes min(patchM, 6).
const GPP_U = /* wgsl */ `
struct U {
  w: u32, h: u32, maxLines: u32, nMin: u32,
  tanHalf: f32, aspect: f32, cellPitch: f32, boardCells: u32,
  nCap: u32, heightFactor: f32, candCount: u32, significance: f32,
  topK: u32, patchM: u32, minGrazingCos: f32, pad0: u32,
}
const TAU: f32 = 6.283185307179586;
`;

// ── S7d gpp.sweep: score every physically possible period ────────────────
//
// The candidate set is not sampled, it is enumerated. A detected line IS a grid
// line, so the two extreme lines of a family sit an INTEGER number of periods
// apart and the only possible periods are spread/n. No seed, no bracket width,
// no sample count -- three tuning parameters that a seeded search has to be given
// and that this construction does not have.
//
// The score is the weighted CIRCULAR RESULTANT: fold every value onto the unit
// circle at theta = 2*pi*value/P and measure how tightly the result clusters.
// 1 is a perfect lattice at that spacing, 0 is none. Row and column families are
// scored separately and SUMMED -- square cells force the same physical period on
// both axes, so pooling the evidence sharpens the peak, while pooling the VALUES
// would not (they are coordinates on two different axes).
//
// ── THE FOLD IS n*(value - familyMin)/spread, AND EVERY PART OF THAT MATTERS ──
//
// P = spread/n, so value/P is value*n/spread and the division by a candidate
// period never happens. What is left is a multiply by an integer.
//
// The shift by the family's own minimum is a precision move, and the mutation
// run measured how much it is worth: NOTHING, on every pose tried. Deleting it
// changes the recovered period in not one digit at four poses across two frame
// sizes. Stated plainly rather than left as an implied benefit, because an
// earlier version of this comment claimed it was load-bearing.
//
// The argument for it is still sound and the reason it does not bite is
// specific: the fold argument reaches ~n, so at the HARD cap n = 288 an
// unshifted value/P lands near 300 where f32 has three or four digits left --
// but the candidate that WINS has n between 14 and 30 on every measured frame,
// where the argument is ~100 and f32 is comfortable. The shift protects
// candidates that are never chosen.
//
// It cannot change the answer either way: a constant shift rotates every one of
// that family's unit vectors equally, so the resultant is invariant and only the
// PHASE moves -- which is why the phase is unshifted again before being written
// out. Kept because it is free and the cap is what it is.
//
// ── nMax IS DEVICE-SIDE, SO THE DISPATCH COVERS THE HARD CAP ──
//
// The frame's real upper bound is the looser of a board bound and a
// camera-height bound, and the height one is a function of `spread`, which only
// exists on the device. So the host cannot size this dispatch. Rather than write
// a fourth indirect-args triple and the hazard that comes with it, every
// candidate past the frame's own nMax scores itself ZERO -- a few hundred
// workgroups that each read nothing and write one vec4.
export const GPP_SWEEP_WGSL = GPP_U + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> rowSamples: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> colSamples: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> familyCounts: vec2<u32>;
@group(0) @binding(4) var<storage, read> extent: array<f32, 12>;
@group(0) @binding(5) var<storage, read_write> scores: array<vec4<f32>>;

const LANES: u32 = 256u;
var<workgroup> wacc: array<vec4<f32>, 256>; // rowCos, rowSin, colCos, colSin
var<workgroup> wsw: array<vec2<f32>, 256>;  // rowWeight, colWeight

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let j = wid.x;
  let li = lid.x;
  let spread = extent[4];
  let n = u.nMin + j;

  // The two independent physical bounds, and the looser one wins. Both track the
  // board size, and \`heightFactor\` already carries MAX_CAMERA_HEIGHT_IN_BOARDS
  // times the cell count -- see encodeGppSearch for the derivation.
  let nFromHeight = u32(max(0.0, floor(u.heightFactor * spread)));
  let nMax = min(u.nCap, max(u.boardCells, nFromHeight));
  let inRange = (spread > 1e-9) && (n <= nMax) && (j < u.candCount);

  var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  var sw = vec2<f32>(0.0, 0.0);
  let rc = min(familyCounts.x, u.maxLines);
  let cc = min(familyCounts.y, u.maxLines);

  if (inRange) {
    let k = TAU * f32(n) / spread;
    let rlo = extent[0];
    let clo = extent[2];
    for (var i = li; i < rc; i = i + LANES) {
      let sv = rowSamples[i];
      let t = k * (sv.x - rlo);
      let w = sv.y;
      acc.x = acc.x + w * cos(t);
      acc.y = acc.y + w * sin(t);
      sw.x = sw.x + w;
    }
    for (var i = li; i < cc; i = i + LANES) {
      let sv = colSamples[i];
      let t = k * (sv.x - clo);
      let w = sv.y;
      acc.z = acc.z + w * cos(t);
      acc.w = acc.w + w * sin(t);
      sw.y = sw.y + w;
    }
  }

  wacc[li] = acc;
  wsw[li] = sw;
  workgroupBarrier();

  var stride = LANES / 2u;
  loop {
    if (stride == 0u) { break; }
    if (li < stride) {
      wacc[li] = wacc[li] + wacc[li + stride];
      wsw[li] = wsw[li] + wsw[li + stride];
    }
    workgroupBarrier();
    stride = stride / 2u;
  }

  if (li == 0u && j < u.candCount) {
    let a = wacc[0];
    let s = wsw[0];
    let P = select(0.0, spread / f32(n), spread > 1e-9);
    var score = 0.0;
    var phiRow = 0.0;
    var phiCol = 0.0;
    if (inRange) {
      // Each family contributes its own resultant, and an EMPTY family
      // contributes nothing rather than poisoning the sum -- which is what makes
      // a one-family frame score half as high everywhere rather than not at all.
      if (s.x > 1e-9) {
        score = score + length(a.xy) / s.x;
        phiRow = extent[0] + atan2(a.y, a.x) * P / TAU;
      }
      if (s.y > 1e-9) {
        score = score + length(a.zw) / s.y;
        phiCol = extent[2] + atan2(a.w, a.z) * P / TAU;
      }
    }
    // Written on every path, in range or not, so the array is a total function
    // over [0, candCount) and needs no clear.
    scores[j] = vec4<f32>(score, phiRow, phiCol, P);
  }
}
`;

// ── S7e gpp.peaks: the real periodicity peaks, best first ────────────────
//
// ── WHY TWO SIGNALS ARE NEEDED AND NEITHER SUFFICES ALONE ──
//
// The resultant is high at the true period P0 and at EVERY sub-multiple P0/2,
// P0/3 -- a lattice at spacing P0 trivially lies on any finer sub-lattice, so
// the resultant cannot tell them apart. The next pass (distinctness) can, by
// looking at image content: a sub-multiple oversamples, so its "cells" repeat.
//
// But distinctness only demotes SUB-multiples. A spurious LONG period
// under-samples, so its neighbours still differ and it looks distinct too, and
// a pure noise bump looks distinct as well. Those are exactly what the resultant
// does demote -- neither folds coherently.
//
// So this pass applies the resultant half: keep interior local maxima clearing
// SIGNIFICANCE of the best score. That is a real-vs-noise cut with about a 10x
// margin (true and sub-multiple peaks sit at 0.7-1.0 of the best, noise and
// spurious-long below 0.2), not a fiddly decision boundary -- distinctness makes
// the actual decision among the survivors.
//
// ── ONE THREAD, AND THE FALLBACK MOVES HERE ──
//
// A few hundred serial iterations with an insertion into a six-entry list. That
// is nothing, and a parallel argmax with a cross-block tie-break argument would
// be more code than the whole pass.
//
// src/pose has a three-level fallback chain spread across two places: best
// distinctness, else the largest-period significant peak, else the plain global
// best. The last of those needs the SCORE array, which only this pass binds --
// so it is done here, and gpp.polish is left with the two that need
// distinctness. That is why this pass always emits at least one candidate
// whenever any score is positive: the alternative is polish holding a binding it
// needs for one branch.
export const GPP_PEAKS_WGSL = GPP_U + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> scores: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> topK: array<vec4<f32>, 8>;
@group(0) @binding(3) var<storage, read_write> status: atomic<u32>;

const GPP_NO_CANDIDATES: u32 = 128u; // bit 7 of the status word

@compute @workgroup_size(1)
fn main() {
  let cnt = u.candCount;
  let cap = min(u.topK, 6u);

  var gMax = 0.0;
  var gBest = 0u;
  for (var i = 0u; i < cnt; i = i + 1u) {
    if (scores[i].x > gMax) { gMax = scores[i].x; gBest = i; }
  }

  // Slot 0 is the count. Written first and on every path, so a frame that finds
  // nothing cannot leave the previous frame's candidate list for polish to read.
  topK[0] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  for (var k = 1u; k < 8u; k = k + 1u) { topK[k] = vec4<f32>(0.0, 0.0, 0.0, 0.0); }

  if (!(gMax > 0.0)) {
    atomicOr(&status, GPP_NO_CANDIDATES);
    return;
  }

  // Descending by score, capped. Insertion rather than a sort, because the list
  // is six long and the input is a few hundred.
  var best: array<vec4<f32>, 6>;
  var kept = 0u;
  for (var i = 1u; i + 1u < cnt; i = i + 1u) {
    let s = scores[i].x;
    // Strict on the left and non-strict on the right, so a flat pair reports
    // one peak rather than none. Candidates run in ASCENDING n, i.e. DESCENDING
    // period -- the neighbours are still the adjacent integer counts either way,
    // which is all the interior test needs.
    if (!(s > scores[i - 1u].x && s >= scores[i + 1u].x && s >= u.significance * gMax)) { continue; }
    let e = vec4<f32>(scores[i].w, s, scores[i].y, scores[i].z);
    var pos = kept;
    loop {
      if (pos == 0u) { break; }
      if (best[pos - 1u].y >= e.y) { break; }
      if (pos < cap) { best[pos] = best[pos - 1u]; }
      pos = pos - 1u;
    }
    if (pos < cap) {
      best[pos] = e;
      kept = min(kept + 1u, cap);
    }
  }

  if (kept == 0u) {
    // No interior maximum cleared the bar -- the curve is monotone, or one
    // sample wide, or every peak is at an end. The global argmax is still a real
    // answer and polish must not be handed an empty list.
    best[0] = vec4<f32>(scores[gBest].w, gMax, scores[gBest].y, scores[gBest].z);
    kept = 1u;
  }

  topK[0].x = f32(kept);
  for (var k = 0u; k < kept; k = k + 1u) { topK[1u + k] = best[k]; }
}
`;

// ── S7f gpp.distinct: which peak is the FUNDAMENTAL ──────────────────────
//
// The resultant cannot separate P0 from P0/2 or P0/3, because a lattice at
// spacing P0 lies on every finer sub-lattice. Image content can. At a
// sub-multiple each real cell becomes a kxk block of IDENTICAL samples, and the
// De Bruijn pattern is locally unique -- so at the TRUE period adjacent cell
// centres differ, and at a sub-multiple they repeat.
//
// So: reproject a small patch of CELL CENTRES at each candidate period and
// return the mean absolute adjacent grayscale difference. Highest wins.
//
// CENTRES, not boundaries: the phase sits ON the cell edges, where the grey is
// the average of two cells and says nothing. Half a period past it is the solid
// interior, which is also exactly the offset decode samples at.
//
// This earns compute back downstream as well as being correct -- it keeps decode
// fed with the coarsest correct period rather than a sub-multiple that would
// balloon its lattice by 4-9x.
//
// ── THE COORDINATE BRIDGE, RE-DERIVED RATHER THAN COPIED ──
//
// A floor point at camera-space offset (uu along Drow, vv along Dcol) sits at
// P = uu*Drow + vv*Dcol - distance*Dnormal, and its gnomonic image is
// (uu/distance, vv/distance) -- so uu = distance * xRow and vv = distance * xCol,
// with distance = cellPitch / period. That single scalar is the whole bridge
// between this stage's coordinates and world units.
//
// src/pose carries a `flipped` sign through all of this, because it is handed a
// normal whose direction is arbitrary. Here fit.eigen already oriented it, so
// there is no flip -- and no inverse quaternion either, because the triad is
// camera-space and P is therefore ALREADY in camera space. src/pose's
// makeCellCentreDistinctness applies invQuat at this point; that whole round
// trip is an artifact of its frame, not of the maths.
//
// ── TWO ROWS, NOT A PATCH ──
//
// The differences wanted are right-neighbour and down-neighbour, so only the
// previous row has to be alive. A 13x13 f32 patch is 676 bytes of per-thread
// private storage; two rows is 104. Same answer.
//
// ── ONE GUARD src/pose DOES NOT HAVE ──
//
// A cell BEHIND the camera projects through a negated depth to a mirrored image
// point that can land on-screen, and the grazing test cannot catch it: p is on
// the floor plane, so dot(p, Dnormal) is exactly -distance whatever uu and vv
// are, and the test is therefore blind to where the point is laterally. The
// grazing cutoff does bound |p| (to distance/minGrazingCos, i.e. 10x at the
// default), but a tilted camera reaches p.z >= 0 well inside that. Latent in
// src/pose rather than hypothetical.
export const GPP_DISTINCT_WGSL = GPP_U + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read> topK: array<vec4<f32>, 8>;
@group(0) @binding(3) var<storage, read> extent: array<f32, 12>;
@group(0) @binding(4) var<storage, read> triad: array<vec3<f32>, 3>;
@group(0) @binding(5) var<storage, read_write> distinctness: array<f32, 8>;

const MAX_M: u32 = 6u;
const MAX_W: u32 = 13u; // 2 * MAX_M + 1, and the row buffers' fixed size

// A cell centre at floor offset (uu, vv) -> its grayscale, or -1 if it is not
// resolvable. Negative is a safe "no sample" marker because a grayscale is not.
fn sampleCell(uu: f32, vv: f32, distance: f32) -> f32 {
  let p = triad[0] * uu + triad[1] * vv - triad[2] * distance;
  let len = length(p);
  if (len < 1e-9) { return -1.0; }
  // dot(p, Dnormal) is exactly -distance, so this is distance/|p| > cutoff:
  // a cell far enough out that the floor is seen at a glancing angle.
  if (!(distance / len > u.minGrazingCos)) { return -1.0; }
  // See the header. The camera looks down -z.
  if (p.z >= 0.0) { return -1.0; }
  let ndcU = -p.x / (p.z * u.tanHalf * u.aspect);
  let ndcV = -p.y / (p.z * u.tanHalf);
  let xx = i32(round(((ndcU + 1.0) * 0.5) * f32(u.w)));
  let yy = i32(round(((1.0 - ndcV) * 0.5) * f32(u.h)));
  if (xx < 0 || yy < 0 || xx >= i32(u.w) || yy >= i32(u.h)) { return -1.0; }
  return gray[u32(yy) * u.w + u32(xx)];
}

@compute @workgroup_size(8)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let k = lid.x;
  if (k >= 8u) { return; }
  // Every slot written every frame, before any early return: polish iterates the
  // candidate count, and a stale distinctness would decide the period.
  distinctness[k] = -1.0;

  let cnt = u32(topK[0].x);
  if (k >= cnt || k >= 6u) { return; }
  let e = topK[1u + k];
  let P = e.x;
  if (!(P > 0.0)) { return; }

  // e.z is the ROW family's phase, which is a phase in xCol; e.w is the COLUMN
  // family's, in xRow. The families are named for the lines, the coordinates for
  // the axes, and the two do not line up -- see GPP_CLASSIFY_WGSL.
  let phiXCol = e.z;
  let phiXRow = e.w;
  let distance = u.cellPitch / P;
  let cp = u.cellPitch;

  // Cell BOUNDARIES sit at the phase; centres are half a cell past it. The
  // round() folds the boundary into one cell so the patch indices stay small.
  let uB = distance * phiXRow;
  let vB = distance * phiXCol;
  let uPhase = (uB - round(uB / cp) * cp) + cp * 0.5;
  let vPhase = (vB - round(vB / cp) * cp) + cp * 0.5;

  // The patch sits at the middle of what was actually detected. src/pose uses
  // the per-family MEDIAN value; the extent midpoint is the same intent and is
  // already computed. A family's own coordinate is the other family's axis:
  // xRow is what the COLUMN family measures.
  let centreXRow = (extent[2] + extent[3]) * 0.5;
  let centreXCol = (extent[0] + extent[1]) * 0.5;
  let jC = round((distance * centreXRow - uPhase) / cp);
  let iC = round((distance * centreXCol - vPhase) / cp);

  let m = min(u.patchM, MAX_M);
  let width = 2u * m + 1u;
  var prev: array<f32, 13>;
  var cur: array<f32, 13>;
  var sum = 0.0;
  var count = 0u;

  for (var di = 0u; di < width; di = di + 1u) {
    let vv = vPhase + (iC + f32(di) - f32(m)) * cp;
    for (var dj = 0u; dj < width; dj = dj + 1u) {
      cur[dj] = sampleCell(uPhase + (jC + f32(dj) - f32(m)) * cp, vv, distance);
    }
    for (var dj = 0u; dj < width; dj = dj + 1u) {
      let a = cur[dj];
      if (a < 0.0) { continue; }
      if (dj + 1u < width) {
        let b = cur[dj + 1u];
        if (b >= 0.0) { sum = sum + abs(a - b); count = count + 1u; }
      }
      if (di > 0u) {
        let b = prev[dj];
        if (b >= 0.0) { sum = sum + abs(a - b); count = count + 1u; }
      }
    }
    for (var dj = 0u; dj < width; dj = dj + 1u) { prev[dj] = cur[dj]; }
  }

  // Too little of the patch landed on-image to judge. Stays -1, and polish falls
  // back rather than trusting a mean over three pixels.
  if (count < 8u) { return; }
  distinctness[k] = sum / f32(count);
}
`;

// ── S7g gpp.polish: pick the winner, refine it, report ────────────────────
//
// Two jobs. The decision among candidates, which is one comparison now that
// distinctness has been measured; and a golden-section refinement of the winner
// within its own integer-count neighbourhood, because the true period is
// generally NOT exactly spread/n -- the extreme detected lines are grid lines,
// but their fitted endpoints carry the detector's error.
//
// Golden section rather than a parabola fit: no assumption about the peak's
// shape, and the resultant near its maximum is not quadratic.
//
// ── ONE THREAD, AND THE COST IS STATED RATHER THAN ASSUMED ──
//
// Golden section is inherently serial -- each probe depends on the previous
// comparison -- so it cannot be parallelized across iterations. 16 evaluations
// over up to ~500 values is ~16,000 sin/cos on a single lane, tens of
// microseconds. "Naive and possibly slow" turns out not to be slow here.
//
// The refactor if a profile ever disagrees is a single workgroup with a tree
// reduction per evaluation: the control flow is uniform across lanes, so the
// barriers are legal exactly as they stand. Not done, because it is 60 more
// lines for a stage that is not on any measured critical path.
//
// ── THE BRACKET NEEDS NO SCORES ARRAY ──
//
// The winner's integer count is round(spread / P), so its neighbourhood is
// [spread/(n+1), spread/(n-1)] and this pass never has to index back into the
// candidate curve. src/pose declines to refine at all when the winner sits at
// either END of its sampled array; here the bracket is physical rather than an
// array index, so the end cases refine like any other.
export const GPP_POLISH_WGSL = GPP_U + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> rowSamples: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> colSamples: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read> familyCounts: vec2<u32>;
@group(0) @binding(4) var<storage, read> extent: array<f32, 12>;
@group(0) @binding(5) var<storage, read> topK: array<vec4<f32>, 8>;
@group(0) @binding(6) var<storage, read> distinctness: array<f32, 8>;
@group(0) @binding(7) var<storage, read_write> gppResult: vec4<f32>;

// This pass does NOT bind status, and that is deliberate rather than an
// omission. Its two failure conditions are both already reported: an empty
// candidate list means gpp.peaks found no positive score and set
// gppNoCandidates itself, and a non-positive spread means gpp.extent set
// gppNoSamples AND forced every score to zero, so peaks set the other bit too.
// A third atomicOr here would say nothing new -- and it would put this pass at
// nine storage buffers, one past the WebGPU baseline.

// (resultant, phase) for one family at period P. The shift by lo is the same
// precision move gpp.sweep makes, undone on the phase before returning.
fn circFit(isRow: bool, count: u32, lo: f32, P: f32) -> vec2<f32> {
  var sc = 0.0;
  var ss = 0.0;
  var sw = 0.0;
  let k = TAU / P;
  for (var i = 0u; i < count; i = i + 1u) {
    var v = 0.0;
    var w = 0.0;
    let sv = select(colSamples[i], rowSamples[i], isRow);
    v = sv.x;
    w = sv.y;
    let t = k * (v - lo);
    sc = sc + w * cos(t);
    ss = ss + w * sin(t);
    sw = sw + w;
  }
  if (sw < 1e-9) { return vec2<f32>(0.0, 0.0); }
  return vec2<f32>(length(vec2<f32>(sc, ss)) / sw, lo + atan2(ss, sc) * P / TAU);
}

fn scoreAt(rc: u32, cc: u32, P: f32) -> f32 {
  return circFit(true, rc, extent[0], P).x + circFit(false, cc, extent[2], P).x;
}

@compute @workgroup_size(1)
fn main() {
  let spread = extent[4];
  let rc = min(familyCounts.x, u.maxLines);
  let cc = min(familyCounts.y, u.maxLines);
  let cnt = u32(topK[0].x);

  // Written on every path. A frame that recovers nothing must report a period of
  // zero rather than the previous frame's, which decode.layout reads as failure.
  gppResult = vec4<f32>(0.0, 0.0, 0.0, 0.0);
  if (cnt == 0u || !(spread > 1e-9)) { return; }

  // The fundamental is the most DISTINCT of the real periodicity peaks; its
  // sub-multiples oversample and score lower. When nothing could be sampled --
  // too little of the patch on-image -- fall back to the LARGEST period, because
  // the failure mode this whole pass exists to prevent is choosing a
  // sub-multiple.
  var win = 0u;
  var bestD = -1.0;
  var largestP = -1.0;
  var largestAt = 0u;
  var haveD = false;
  for (var k = 0u; k < cnt; k = k + 1u) {
    let d = distinctness[k];
    if (d >= 0.0 && d > bestD) { bestD = d; win = k; haveD = true; }
    if (topK[1u + k].x > largestP) { largestP = topK[1u + k].x; largestAt = k; }
  }
  if (!haveD) { win = largestAt; }

  let p0 = topK[1u + win].x;
  var bestP = p0;
  let n = round(spread / p0);
  if (n >= 2.0 && p0 > 0.0) {
    // Golden section over [P(n+1), P(n-1)], maximizing the resultant.
    let gr = 0.6180339887498949;
    var a = spread / (n + 1.0);
    var b = spread / (n - 1.0);
    var c = b - gr * (b - a);
    var d = a + gr * (b - a);
    var fc = scoreAt(rc, cc, c);
    var fd = scoreAt(rc, cc, d);
    for (var it = 0u; it < 14u; it = it + 1u) {
      if (fc > fd) {
        b = d; d = c; fd = fc; c = b - gr * (b - a); fc = scoreAt(rc, cc, c);
      } else {
        a = c; c = d; fc = fd; d = a + gr * (b - a); fd = scoreAt(rc, cc, d);
      }
    }
    bestP = (a + b) * 0.5;
  }

  let rf = circFit(true, rc, extent[0], bestP);
  let cf = circFit(false, cc, extent[2], bestP);
  // height = cellPitch / period, which is the whole of Act II's output: the
  // period IS the camera's distance from the floor, in cells.
  gppResult = vec4<f32>(bestP, rf.y, cf.y, u.cellPitch / bestP);
}
`;

// ── S8 decode.layout: where the sampling lattice IS ───────────────────────
//
// This opens Act III. Orientation, period and phase are known, so the floor's
// cell grid is fully determined in camera space -- what is not yet known is
// WHICH cells to sample, how many there are, and how bright a pixel has to be to
// read as white. That is this stage, and it produces about 150 bytes.
//
// ── IT IS BOUNDED BY THE DETECTED LINES, NOT BY THE VIEW QUADRILATERAL ──
//
// src/pose casts a 49x49 grid of rays through the image, intersects each with
// the floor and takes the min/max of the results. That grid exists to work
// around an all-or-nothing four-corner test, which returned nothing the moment
// one corner pointed past the horizon and silently killed decode on any oblique
// view. Bounding by the detected LINES dissolves the problem instead: a grid
// line is below the horizon by construction, because you cannot detect one in
// sky. So decode.bounds, decode.boundsInit, `uvBounds`, 2,401 ray casts and
// their minGrazingCos gate are all gone, and what replaces them is four lanes
// gpp.extent already reduces.
//
// Measured before being built (scripts/hull-measure.ts, 2026-08-14, 204 poses of
// src/pose at 480x640): identical recovered position at every pose in the
// operating range, 0.02% of correct decode votes lost, consistency never worse.
// At grazing it clips 35% of all complete windows for 0.18% of the correct
// votes and RAISES consistency by up to 0.17 -- which is as direct a statement
// as that measurement can make that the clipped band is noise.
//
// ── THE COORDINATE BRIDGE IS ONE SCALAR, AND IT HAS NO SIGN ──
//
// gnomonic gives xRow = -dot(r,Drow)/dot(r,Dn); a floor point is p = t*r with
// t = -distance/dot(r,Dn), so u = dot(p,Drow) = distance * xRow. Exactly:
//
//     u = distance * xRow        v = distance * xCol
//
// src/pose carries a SIGN on that scalar (projectedUVScale returns +/-distance)
// because gnomonic() is handed a raw eigenvector while projectedUVBounds flips
// it toward the floor. Here fit.eigen oriented the normal once, at the point the
// sign was created, so the two agree and the scalar is just `distance`. That is
// the second consequence of §10's decision, after gpp's two deleted quaternion
// round trips.
//
// ── THE 144 CLAMP IS NOT A FAILURE, AND THAT IS MEASURED ──
//
// MAX_CELLS is torusR * torusC = 20,736 rather than a chosen constant, which is
// rule 3 applied to the last buffer still sized by guess. The clamp is what
// makes it a bound: the floor TILES and the anchor arithmetic is mod R/C, so a
// window from a second board period votes for the SAME anchor -- clipping loses
// redundant votes, not evidence. Clipping every edge to 144 changed the
// recovered position at 0 of 204 measured poses.
//
// But the raw hull genuinely reaches 169 cells on an edge at h=24 tilt 45-55,
// where the camera really does see 200+ cells and the recovered period is right
// to ~1.4%. So `gridOverflow` is a DIAGNOSTIC and the lattice keeps decoding.
// Treating it as a failure would throw away frames that decode correctly.
//
// The kept window is CENTRED on the hull. Any contiguous 144 works, since the
// arithmetic is mod R/C; centring is the least arbitrary choice rather than a
// measured best one -- keep-from-min is untested.
const LAYOUT_U = /* wgsl */ `
struct U {
  w: u32, h: u32, torusR: u32, torusC: u32,
  maxCells: u32, pad0: u32, pad1: u32, pad2: u32,
  cellPitch: f32, tanHalf: f32, aspect: f32, minGrazingCos: f32,
}
`;

// The 128-byte block every later decode pass reads instead of a uniform, because
// nothing in it is known to the host any more. `invQuat` is NOT in it: src/pose
// carries the inverse camera quaternion here to get from its analysis frame back
// to camera space, and in src/pose2 the triad is already camera-space, so the
// rotation is the identity. Same deletion gpp.distinct made for the same reason.
//
// THE THREE AXES ARE vec4 AND ONLY xyz IS USED, which costs 12 bytes and buys
// an unsurprising memory layout. A vec3 member is 12 bytes with align 16, so the
// NEXT scalar member packs into its trailing gap -- `distance` would sit at
// offset 44, not 48, and every offset after it would be 4 low. That is invisible
// in WGSL, which computes it for you, and wrong in any host-side reader written
// from the field order. It was wrong in this file's first test reader.
const LAYOUT_STRUCT = /* wgsl */ `
struct Layout {
  Drow: vec4<f32>,
  Dcol: vec4<f32>,
  normal: vec4<f32>,
  distance: f32,
  tanHalf: f32,
  aspect: f32,
  minGrazingCos: f32,
  uPhase: f32,
  vPhase: f32,
  cellPitch: f32,
  binThreshold: f32,
  rows: u32,
  cols: u32,
  imageW: u32,
  imageH: u32,
  kMinU: i32,
  kMinV: i32,
  zeroI: u32,
  zeroJ: u32,
  valid: u32,
}
`;

// ── S8a decode.binThreshPartials / binThreshReduce ────────────────────────
//
// The binarization threshold is the image's mean grey level. On the host that is
// a loop over `gray`; here it is a block sum and a reduce.
//
// Materializing a binarized image would be the obvious port and is the wrong
// one, as the current code's own comment says: the lattice reads on the order of
// a thousand of ~200,000 pixels, so all that is needed is the mean, compared at
// sample time.
//
// THE HIERARCHY IS WHAT MAKES f32 SAFE HERE. 307,200 pixels at up to 255 sum to
// ~7.8e7, well past f32's 1.7e7 exactly-representable integer range -- a naive
// serial accumulation would start dropping whole units near the end. Summing 256
// per workgroup (max 6.5e4), then ~5 blocks per lane, then a 256-lane tree keeps
// every partial small relative to its own addends. No compensated summation
// needed, and none of it is a choice about accuracy -- it falls out of the
// reduction shape.
export const DECODE_BINTHRESH_PARTIALS_WGSL = LAYOUT_U + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> partials: array<f32>;

const LANES: u32 = 256u;
var<workgroup> acc: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = u.w * u.h;
  let i = wid.x * LANES + lid.x;
  acc[lid.x] = select(0.0, gray[min(i, n - 1u)], i < n);
  workgroupBarrier();
  var stride = LANES / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { acc[lid.x] = acc[lid.x] + acc[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  // One write per dispatched workgroup, and the dispatch is exactly the buffer's
  // length -- total by construction, so no clear.
  if (lid.x == 0u) { partials[wid.x] = acc[0]; }
}
`;

export const DECODE_BINTHRESH_REDUCE_WGSL = LAYOUT_U + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> partials: array<f32>;
@group(0) @binding(2) var<storage, read_write> binThreshold: f32;

const LANES: u32 = 256u;
var<workgroup> acc: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let n = u.w * u.h;
  let blocks = (n + LANES - 1u) / LANES;
  var s = 0.0;
  for (var i = lid.x; i < blocks; i = i + LANES) { s = s + partials[i]; }
  acc[lid.x] = s;
  workgroupBarrier();
  var stride = LANES / 2u;
  loop {
    if (stride == 0u) { break; }
    if (lid.x < stride) { acc[lid.x] = acc[lid.x] + acc[lid.x + stride]; }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (lid.x == 0u) { binThreshold = acc[0] / f32(n); }
}
`;

// ── S8b decode.layout ─────────────────────────────────────────────────────
//
// One thread. Writes the layout block and the FIVE indirect-args triples whose
// extents are now device-side quantities: the build's, and the tally's four.
// (The correctness pass's sixth is decode.argmax's, because it depends on the
// winning orientation -- the second host dependency decode used to have, and the
// one that is easy to miss.)
//
// EIGHT storage buffers exactly, which is the WebGPU baseline. The declaration
// had nine and would not have built on a baseline device; deleting `uvBounds`
// and handing `correctnessArgs` to its real owner is what fits it.
//
// AND `layout` IS A WGSL RESERVED KEYWORD, so the binding is `lay`. The second
// one this rewrite has hit, after `active` in §11 -- but unlike that one it
// fails LOUDLY: a reserved word in a binding name is a parse error the error
// scope reports by name, while `active` as a local was accepted and then
// silently did nothing.
export const DECODE_LAYOUT_WGSL = LAYOUT_U + LAYOUT_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> triad: array<vec3<f32>, 3>;
@group(0) @binding(2) var<storage, read> gppResult: vec4<f32>;
@group(0) @binding(3) var<storage, read> extent: array<f32, 12>;
@group(0) @binding(4) var<storage, read> binThreshold: f32;
@group(0) @binding(5) var<storage, read_write> lay: Layout;
@group(0) @binding(6) var<storage, read_write> buildArgs: array<u32, 3>;
@group(0) @binding(7) var<storage, read_write> tallyArgs: array<u32, 12>;
@group(0) @binding(8) var<storage, read_write> status: atomic<u32>;

const LAYOUT_INVALID: u32 = 256u;  // bit 8
const GRID_OVERFLOW: u32 = 512u;   // bit 9
const WG2D: u32 = 8u;              // decode.build and decode.tally are 8x8

@compute @workgroup_size(1)
fn main() {
  let cp = u.cellPitch;
  let period = gppResult.x;
  // phiRow is the ROW family's phase and a row line's value is its xCol, so
  // phiRow lives in xCol and therefore in v. phiCol is the other way round. The
  // families are named for the lines and the coordinates for the axes; this
  // crossover is the one thing to get right here, and src/pose has it too
  // (uBoundaryRaw = uvScale * gpp.phiCol).
  let phiRow = gppResult.y;
  let phiCol = gppResult.z;
  let distance = gppResult.w;

  // The geometry is written on EVERY path, so decode.build and finish read this
  // frame's triad whatever happens. Only the lattice fields are conditional --
  // and valid, rows, cols and both args triples are the failure
  // propagation, so they are set to nothing first and filled in on success.
  lay.Drow = vec4<f32>(triad[0], 0.0);
  lay.Dcol = vec4<f32>(triad[1], 0.0);
  lay.normal = vec4<f32>(triad[2], 0.0);
  lay.distance = distance;
  lay.tanHalf = u.tanHalf;
  lay.aspect = u.aspect;
  lay.minGrazingCos = u.minGrazingCos;
  lay.cellPitch = cp;
  lay.binThreshold = binThreshold;
  lay.imageW = u.w;
  lay.imageH = u.h;
  lay.uPhase = 0.0;
  lay.vPhase = 0.0;
  lay.rows = 0u;
  lay.cols = 0u;
  lay.kMinU = 0;
  lay.kMinV = 0;
  lay.zeroI = 0u;
  lay.zeroJ = 0u;
  lay.valid = 0u;
  for (var k = 0u; k < 3u; k = k + 1u) { buildArgs[k] = 0u; }
  for (var k = 0u; k < 12u; k = k + 1u) { tallyArgs[k] = 0u; }

  // The hull, still in gnomonic units: [5],[6] bound xRow (which becomes u) and
  // [7],[8] bound xCol (which becomes v). gpp.extent writes zeros rather than
  // its sentinels when no line was classified, so an empty frame fails the
  // strict comparison rather than producing a lattice 1e38 cells wide.
  let ok = (period > 0.0) && (distance > 0.0)
    && (extent[6] > extent[5]) && (extent[8] > extent[7]);
  if (!ok) { atomicOr(&status, LAYOUT_INVALID); return; }

  // Cell BOUNDARIES sit at the phase; the lattice samples cell CENTRES, half a
  // cell past it. The round() folds the boundary into the cell containing the
  // origin so the integer indices below stay small. (WGSL's round() is
  // ties-to-even where JS's Math.round is ties-away -- reachable only for a phase
  // landing exactly on a half cell, and either choice is a valid cell centre.)
  let uB = distance * phiCol;
  let vB = distance * phiRow;
  let uPhase = (uB - round(uB / cp) * cp) + cp * 0.5;
  let vPhase = (vB - round(vB / cp) * cp) + cp * 0.5;

  var kMinU = i32(floor((distance * extent[5] - uPhase) / cp));
  var kMaxU = i32(ceil((distance * extent[6] - uPhase) / cp));
  var kMinV = i32(floor((distance * extent[7] - vPhase) / cp));
  var kMaxV = i32(ceil((distance * extent[8] - vPhase) / cp));

  // ONE BOARD PERIOD, CENTRED. A second period is redundancy rather than
  // evidence -- see the header. This FIRES in the grazing band on frames that
  // decode correctly, so it reports and continues.
  var clipped = false;
  let capU = i32(u.torusC);
  let capV = i32(u.torusR);
  if (kMaxU - kMinU + 1 > capU) {
    let drop = (kMaxU - kMinU + 1) - capU;
    kMinU = kMinU + drop / 2;
    kMaxU = kMinU + capU - 1;
    clipped = true;
  }
  if (kMaxV - kMinV + 1 > capV) {
    let drop = (kMaxV - kMinV + 1) - capV;
    kMinV = kMinV + drop / 2;
    kMaxV = kMinV + capV - 1;
    clipped = true;
  }
  if (clipped) { atomicOr(&status, GRID_OVERFLOW); }

  let cols = u32(kMaxU - kMinU + 1);
  let rows = u32(kMaxV - kMinV + 1);
  // Unreachable while maxCells >= torusR * torusC, which is what the clamp above
  // guarantees and what buffers.ts sizes 'packed' to. Kept because the
  // alternative to a guard here is decode.build writing past the end of a
  // buffer, which WGSL discards silently -- cells would simply vanish.
  if (rows * cols > u.maxCells) { atomicOr(&status, LAYOUT_INVALID); return; }

  // The lattice cell nearest the point directly beneath the camera. §14 needs it
  // as the reference whose world position the decoded anchor names.
  let zi = clamp(i32(round(-vPhase / cp)) - kMinV, 0, i32(rows) - 1);
  let zj = clamp(i32(round(-uPhase / cp)) - kMinU, 0, i32(cols) - 1);

  lay.uPhase = uPhase;
  lay.vPhase = vPhase;
  lay.rows = rows;
  lay.cols = cols;
  lay.kMinU = kMinU;
  lay.kMinV = kMinV;
  lay.zeroI = u32(zi);
  lay.zeroJ = u32(zj);
  lay.valid = 1u;

  // decode.build is one thread per cell over (rows, cols) with gid.x = i = row.
  buildArgs[0] = (rows + WG2D - 1u) / WG2D;
  buildArgs[1] = (cols + WG2D - 1u) / WG2D;
  buildArgs[2] = 1u;

  // The tally runs the grid in each of the four cardinal rotations, and rows and
  // cols SWAP at orientations 1 and 3. Its interior guard drops windows that run
  // off the end, so an edge shorter than ORDER costs a dispatch and no votes.
  for (var o = 0u; o < 4u; o = o + 1u) {
    let swap = (o == 1u || o == 3u);
    let rr = select(rows, cols, swap);
    let cc = select(cols, rows, swap);
    tallyArgs[o * 3u + 0u] = (rr + WG2D - 1u) / WG2D;
    tallyArgs[o * 3u + 1u] = (cc + WG2D - 1u) / WG2D;
    tallyArgs[o * 3u + 2u] = 1u;
  }
}
`;

// ── S9 decode.build: one lattice cell -> one bit ──────────────────────────
//
// One thread per lattice cell. Take the cell's (u, v) on the floor, form its
// position relative to the camera, project it to a pixel, sample `gray`, and
// threshold. Two bits out: whether the cell is resolvable at all, and what it
// says.
//
// ── THE PER-CELL GRAZING TEST IS NOT OPTIONAL, AND THE HULL RAISES THE STAKES ──
//
// A cell beyond the horizon projects through a NEGATED depth to a mirrored image
// point that lands on screen and passes every bounds test. `dot(rayDir, normal)
// < -minGrazingCos` is what rejects it. Measured under the hull lattice at
// grazing (§12): 9,811 such cells across 24 poses, 2,642 in the worst one, every
// one of which the screen-bounds test would have ACCEPTED. Deleting this test
// does not lose cells -- it silently samples the wrong pixel for each of them.
//
// §12 deletes the minGrazingCos gate from the BOUNDS computation, which is a
// different thing entirely, and the hull makes this one matter more rather than
// less: a line detected near the horizon has no grazing gate of its own, so the
// hull can reach further out than the old view quadrilateral did.
//
// ── ONE GUARD src/pose DOES NOT HAVE ──
//
// `p.z >= 0` -- a cell BEHIND the camera. The grazing test cannot catch it,
// because p is on the floor plane so dot(p, normal) is exactly -distance
// whatever the lateral offset, and the test is therefore blind to where the
// point is. The cutoff does bound |p| (10x distance at minGrazingCos 0.1) but a
// tilted camera reaches p.z >= 0 well inside that. Same guard gpp.distinct
// added, for the same reason, and unreachable at every pose tried there.
//
// ── TWO SUBTLETIES WORTH PRESERVING ──
//
// The pixel is bounds-tested UNROUNDED and then CLAMPED: px = w - 0.3 passes
// `px < w` and rounds to w. Measured 46 such cells in a 270x276 grid. And the
// divisor is guarded directly rather than testing the quotient for finiteness
// afterwards -- WGSL has no isFinite, and a compiler may assume its operands are
// finite, so a self-comparison NaN test is not reliable.
//
// ── NO invQuat, AND NO `geom` ──
//
// src/pose rotates p by the inverse camera quaternion to reach camera space.
// Here the triad IS camera-space, so that rotation is the identity -- the third
// quaternion round trip this rewrite has deleted by the frame being consistent.
// And `geom` (u, v, px, py per cell, 4 MiB at MAX_CELLS) is gone: nothing on the
// pose path reads it, it exists for the projected-cam overlay, and a pose-only
// pipeline should not allocate it.
export const DECODE_BUILD_WGSL = LAYOUT_STRUCT + /* wgsl */ `
@group(0) @binding(0) var<storage, read> lay: Layout;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> packed: array<u32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  let j = gid.y;
  // The dispatch is a CEILING over the 8x8 workgroup, so the last row and column
  // of workgroups run threads past the lattice.
  if (i >= lay.rows || j >= lay.cols) { return; }
  let idx = i * lay.cols + j;

  let v = lay.vPhase + f32(lay.kMinV + i32(i)) * lay.cellPitch;
  let uu = lay.uPhase + f32(lay.kMinU + i32(j)) * lay.cellPitch;
  // This floor point's position relative to the camera, in camera space.
  let p = lay.Drow.xyz * uu + lay.Dcol.xyz * v - lay.normal.xyz * lay.distance;

  let lenP = length(p);
  if (!(lenP > 0.0)) { packed[idx] = 0u; return; }
  // dot(p, normal) is exactly -distance, so this is distance / |p| > cutoff.
  if (!(lay.distance / lenP > lay.minGrazingCos)) { packed[idx] = 0u; return; }
  // Behind the camera, which looks down -z. See the header.
  if (p.z >= 0.0) { packed[idx] = 0u; return; }

  let denom = p.z * lay.tanHalf;
  if (denom == 0.0) { packed[idx] = 0u; return; }
  let ndcU = -p.x / (denom * lay.aspect);
  let ndcV = -p.y / denom;
  let fw = f32(lay.imageW);
  let fh = f32(lay.imageH);
  let px = ((ndcU + 1.0) * 0.5) * fw;
  let py = ((1.0 - ndcV) * 0.5) * fh;
  if (!(px >= 0.0 && px < fw && py >= 0.0 && py < fh)) { packed[idx] = 0u; return; }

  let xx = u32(clamp(round(px), 0.0, fw - 1.0));
  let yy = u32(clamp(round(py), 0.0, fh - 1.0));
  // A SET bit is the DARK cell: the pattern's 1 cells are printed black, so the
  // threshold comparison has to be < and not >.
  let bit = select(0u, 1u, gray[yy * lay.imageW + xx] < lay.binThreshold);
  packed[idx] = 1u | (bit << 1u);
}
`;

// ── S10 decode.tally: every window votes for where the board is ───────────
//
// One thread per ORDER x ORDER window, in each of four rotations. Pack the 25
// bits into a key, look it up in the De Bruijn table, and if it is a real
// window, atomicAdd a vote for the board anchor it implies.
//
// ── WHY FOUR ROTATIONS ──
//
// The camera's yaw relative to the board is unknown and the pattern is not
// rotationally symmetric, so rather than solve for the rotation, try all four
// cardinal ones and let the vote count decide. The three that are wrong produce
// window keys that are not in the table at all, so they cost lookups and cast
// almost no votes.
//
// ── WHY A HASH TABLE ──
//
// The lookup is from a 25-bit key to a torus position, with ~20,736 real entries
// in a 2^25 key space. Flattened into open addressing at load factor 0.5, built
// once per device (src/pose2/board.ts). `hashU32` must stay byte-identical
// between there and here -- JS Math.imul and a WGSL u32 multiply both wrap mod
// 2^32 -- or every lookup silently misses and the frame reports an ordinary
// undecodable outcome.
export const DECODE_TALLY_WGSL = LAYOUT_STRUCT + /* wgsl */ `
struct U {
  order: u32, torusR: u32, torusC: u32, tableSize: u32,
}
@group(0) @binding(0) var<uniform> u: U;
// THE LATTICE DIMENSIONS COME FROM HERE, NOT FROM THE UNIFORM, and that is the
// whole of what §12 called structurally disruptive. \`rows\` and \`cols\` are
// computed on the device by decode.layout, so a host-written uniform carrying
// them would be precisely the readback this pipeline exists to delete.
@group(0) @binding(1) var<storage, read> lay: Layout;
@group(0) @binding(2) var<storage, read> packed: array<u32>;
@group(0) @binding(3) var<storage, read> hashKeys: array<u32>;
@group(0) @binding(4) var<storage, read> hashValues: array<u32>;
@group(0) @binding(5) var<storage, read_write> hist: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> totalWindows: atomic<u32>;

const NOT_FOUND: u32 = 0xFFFFFFFFu;

fn hashU32(xIn: u32) -> u32 {
  var x = xIn;
  x = x ^ (x >> 16u);
  x = x * 0x85ebca6bu;
  x = x ^ (x >> 13u);
  x = x * 0xc2b2ae35u;
  x = x ^ (x >> 16u);
  return x;
}

// Bounded linear probe -- never more than tableSize steps, so a construction or
// hash mismatch cannot hang the shader, it just misses.
fn lookupTorus(key: u32) -> u32 {
  var slot = hashU32(key) % u.tableSize;
  for (var probe = 0u; probe < u.tableSize; probe = probe + 1u) {
    let val = hashValues[slot];
    if (val == NOT_FOUND) { return NOT_FOUND; }
    if (hashKeys[slot] == key) { return val; }
    slot = (slot + 1u) % u.tableSize;
  }
  return NOT_FOUND;
}

// Where rotated index (a, b) reads from in the UNROTATED grid. Four index maps,
// and decode.correctness must use the identical one.
fn origIndex(o: u32, rows: u32, cols: u32, a: u32, b: u32) -> vec2<u32> {
  if (o == 1u) { return vec2<u32>(rows - 1u - b, a); }
  if (o == 2u) { return vec2<u32>(rows - 1u - a, cols - 1u - b); }
  if (o == 3u) { return vec2<u32>(b, cols - 1u - a); }
  return vec2<u32>(a, b);
}

// FOUR ENTRY POINTS over one body, rather than four uniform blocks and a dynamic
// offset. The orientation is the only thing that differs between the four
// dispatches, and it is a compile-time constant at each -- so this is the same
// move grow's four entry points make, and it keeps \`pass()\` free of binding
// offsets it would otherwise need to understand.
fn tally(orient: u32, gid: vec3<u32>) {
  let rows = lay.rows;
  let cols = lay.cols;
  let swap = (orient == 1u || orient == 3u);
  let rr = select(rows, cols, swap);
  let cc = select(cols, rows, swap);
  let i0 = gid.x;
  let j0 = gid.y;
  if (i0 + u.order > rr || j0 + u.order > cc) { return; }

  var key = 0u;
  for (var di = 0u; di < u.order; di = di + 1u) {
    for (var dj = 0u; dj < u.order; dj = dj + 1u) {
      let oi = origIndex(orient, rows, cols, i0 + di, j0 + dj);
      let p = packed[oi.x * cols + oi.y];
      // One unresolvable cell and the whole window is out: a missing bit is not
      // a zero bit, and treating it as one would vote for a real anchor on
      // evidence that does not exist.
      if ((p & 1u) == 0u) { return; }
      key = (key << 1u) | ((p >> 1u) & 1u);
    }
  }
  atomicAdd(&totalWindows, 1u);

  // \`match\` is a WGSL reserved keyword. Third one this rewrite has hit, after
  // \`active\` (§11) and \`layout\` (decode.layout). All three are ordinary English
  // words that read as the obvious name.
  let found = lookupTorus(key);
  if (found == NOT_FOUND) { return; }
  let matchRow = found / u.torusC;
  let matchCol = found % u.torusC;
  // (matchRow - i0) mod R without underflow: matchRow is already < R, while i0
  // can exceed it -- the lattice may span more than one board period.
  let anchorRow = (matchRow + u.torusR - (i0 % u.torusR)) % u.torusR;
  let anchorCol = (matchCol + u.torusC - (j0 % u.torusC)) % u.torusC;
  atomicAdd(&hist[orient * u.torusR * u.torusC + anchorRow * u.torusC + anchorCol], 1u);
}

@compute @workgroup_size(8, 8) fn o0(@builtin(global_invocation_id) g: vec3<u32>) { tally(0u, g); }
@compute @workgroup_size(8, 8) fn o1(@builtin(global_invocation_id) g: vec3<u32>) { tally(1u, g); }
@compute @workgroup_size(8, 8) fn o2(@builtin(global_invocation_id) g: vec3<u32>) { tally(2u, g); }
@compute @workgroup_size(8, 8) fn o3(@builtin(global_invocation_id) g: vec3<u32>) { tally(3u, g); }
`;

// ── S11 decode.argmax: the histogram's winner ─────────────────────────────
//
// One workgroup of 256 threads reduces the whole histogram to one winner, and
// writes the correctness pass's dispatch extent while it is there.
//
// ── THE TIE-BREAK HAS TO MATCH EXACTLY ──
//
// Highest votes, then LOWEST index, and a winner needs at least one vote. Each
// thread strides upward taking `>` only, and the tree reduction prefers the
// smaller index on a tie, so both halves agree.
//
// ── ONE WORKGROUP, DELIBERATELY ──
//
// 4*R*C is 82,944 at a 144 board -- 324 strided iterations for 256 threads. A
// two-level reduction would use the machine better and would also need a second
// pass, a second buffer, and its own cross-block tie-break argument.
//
// ── AND IT OWNS correctnessArgs, WHICH IS WHY decode.layout DOES NOT ──
//
// The correctness pass's extent depends on the WINNING ORIENTATION, because rows
// and cols swap at 1 and 3. That is decode's second host dependency and the one
// easy to miss -- knowing the winner is not enough to size the dispatch, you
// have to know it BEFORE encoding. Writing the args here is what makes decode
// one submit.
export const DECODE_ARGMAX_WGSL = LAYOUT_STRUCT + /* wgsl */ `
struct U {
  n: u32, torusR: u32, torusC: u32, wg: u32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> lay: Layout;
@group(0) @binding(2) var<storage, read> hist: array<u32>;
@group(0) @binding(3) var<storage, read> totalWindows: u32;
@group(0) @binding(4) var<storage, read_write> result: array<u32>;
@group(0) @binding(5) var<storage, read_write> correctnessArgs: array<u32, 3>;

const NONE: u32 = 0xFFFFFFFFu;
const LANES: u32 = 256u;

var<workgroup> wv: array<u32, 256>;
var<workgroup> wi: array<u32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var bv = 0u;
  var bi = NONE;
  for (var i = t; i < u.n; i = i + LANES) {
    let v = hist[i];
    if (v > bv) { bv = v; bi = i; }
  }
  wv[t] = bv;
  wi[t] = bi;
  workgroupBarrier();

  var stride = LANES / 2u;
  loop {
    if (stride == 0u) { break; }
    if (t < stride) {
      let ov = wv[t + stride];
      let oi = wi[t + stride];
      if (ov > wv[t] || (ov == wv[t] && oi < wi[t])) { wv[t] = ov; wi[t] = oi; }
    }
    workgroupBarrier();
    stride = stride / 2u;
  }
  if (t != 0u) { return; }

  let votes = wv[0];
  let idx = wi[0];
  let found = select(0u, 1u, idx != NONE);
  let rc = u.torusR * u.torusC;
  // Guarded rather than left to divide a NONE index: an undecodable frame is an
  // ORDINARY outcome here, so it has to produce zeros.
  let orient = select(0u, idx / rc, found == 1u);
  let rem = select(0u, idx % rc, found == 1u);
  // Slots 0..5. Slots 6/7 are the correctness pass's atomicAdd targets and are
  // NOT touched here -- they are cleared before this stage's first bind.
  result[0] = found;
  result[1] = orient;
  result[2] = select(0u, rem / u.torusC, found == 1u);
  result[3] = select(0u, rem % u.torusC, found == 1u);
  result[4] = votes;
  result[5] = totalWindows;

  let swap = (orient == 1u || orient == 3u);
  let rr = select(lay.rows, lay.cols, swap);
  let cc = select(lay.cols, lay.rows, swap);
  correctnessArgs[0] = select(0u, (rr + u.wg - 1u) / u.wg, found == 1u);
  correctnessArgs[1] = select(0u, (cc + u.wg - 1u) / u.wg, found == 1u);
  correctnessArgs[2] = select(0u, 1u, found == 1u);
}
`;

// ── S12 decode.correctness: how much of the grid agrees ───────────────────
//
// Reduces the whole rotated grid to two integers: how many sampled bits match
// what the pattern says should be there and how many do not. Their ratio is
// `consistency`, the one decode number always on screen.
//
// It dispatches INDIRECTLY, off args decode.argmax wrote, because its extent
// depends on the winning orientation.
//
// AND CONSISTENCY IS NOT A CHECK ON THE POSE (§14). A frame can decode the board
// perfectly -- right anchor, right bits, consistency 1.000 -- while the final
// rotation from board frame to world frame is wrong. That failure was live in
// this codebase once and reported perfect consistency throughout.
export const DECODE_CORRECTNESS_WGSL = LAYOUT_STRUCT + /* wgsl */ `
struct U {
  torusR: u32, torusC: u32, pad0: u32, pad1: u32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> lay: Layout;
@group(0) @binding(2) var<storage, read> packed: array<u32>;
@group(0) @binding(3) var<storage, read> torus: array<u32>;
// The shared result block: 1..3 are the winner (settled by the pass before this
// one), 6..7 are this pass's counters. Bound as atomics throughout because one
// binding does both jobs; the winner slots are read with a plain atomicLoad.
@group(0) @binding(4) var<storage, read_write> result: array<atomic<u32>>;

fn origIndex(o: u32, rows: u32, cols: u32, a: u32, b: u32) -> vec2<u32> {
  if (o == 1u) { return vec2<u32>(rows - 1u - b, a); }
  if (o == 2u) { return vec2<u32>(rows - 1u - a, cols - 1u - b); }
  if (o == 3u) { return vec2<u32>(b, cols - 1u - a); }
  return vec2<u32>(a, b);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rows = lay.rows;
  let cols = lay.cols;
  let orient = atomicLoad(&result[1]);
  let anchorRow = atomicLoad(&result[2]);
  let anchorCol = atomicLoad(&result[3]);
  let swap = (orient == 1u || orient == 3u);
  let rr = select(rows, cols, swap);
  let cc = select(cols, rows, swap);
  let i = gid.x;
  let j = gid.y;
  // Still bounds-checked even though the extent is device-sized: the indirect
  // args are a CEILING over the 8x8 workgroup.
  if (i >= rr || j >= cc) { return; }

  let oi = origIndex(orient, rows, cols, i, j);
  let p = packed[oi.x * cols + oi.y];
  if ((p & 1u) == 0u) { return; } // unresolvable -- counted neither way
  let bit = (p >> 1u) & 1u;

  let torusRow = (anchorRow + i) % u.torusR;
  let torusCol = (anchorCol + j) % u.torusC;
  if (bit == torus[torusRow * u.torusC + torusCol]) {
    atomicAdd(&result[6], 1u);
  } else {
    atomicAdd(&result[7], 1u);
  }
}
`;

// ── S13 finish: the anchor -> a world pose ────────────────────────────────
//
// One thread, and the end of the pipeline. Turns the winning anchor into a
// world position and a camera quaternion, and assembles the 128 bytes that are
// the only thing crossing the bus.
//
// ── THE MATH ──
//
// The reference lattice cell is registered against the torus to get its
// absolute board row and column, which gives its WORLD position outright -- the
// board is a known printed thing at a known place. The same cell's position
// RELATIVE to the camera is pure arithmetic in (u, v). Rotate that into world
// space and subtract, and what is left is where the camera is.
//
// ── THE TRAP, AND IT IS A LIVE ONE ──
//
// solveRecoveredCamQuat rotates (Dcol, Drow) through the winning orientation
// with a SIGN FLIP at each step -- nextCol = -rowMath. An earlier version of
// this codebase got the axis swap right and missed the flips entirely. That is
// silently correct at orientation 0, which is most captures, and wrong at 1, 2
// and 3 -- while STILL REPORTING PERFECT BIT CONSISTENCY, because period, phase
// and anchor recovery are each independently correct and only this last rotation
// is wrong. It was caught live at yaw +2 degrees, where a 4-degree swing flipped
// which orientation won.
//
// So: consistency is NOT a check on this stage. Nothing downstream of here is.
//
// ── WHY ONE ROTATION FORMULA WORKS FOR ALL FOUR ORIENTATIONS ──
//
// (x, y) -> (y, -x) is a proper rotation: determinant +1, and it cycles back to
// the identity after four applications rather than after two. A per-case
// reflection fix would be the sign of a bug rather than a fix. What makes one
// formula sufficient is fit.eigen's HANDEDNESS enforcement back in §10 -- it
// guarantees the triad is consistently right-handed, so the third axis never
// needs an independent correction. That guarantee was established eight stages
// ago and this is the only place that spends it.
export const FINISH_WGSL = LAYOUT_STRUCT + /* wgsl */ `
struct U {
  torusR: u32, torusC: u32, maxRegions: u32, maxLines: u32,
}
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> lay: Layout;
@group(0) @binding(2) var<storage, read> result: array<u32, 8>;
@group(0) @binding(3) var<storage, read> counts: vec2<u32>;
@group(0) @binding(4) var<storage, read> lineCount: vec2<u32>;
@group(0) @binding(5) var<storage, read> growArgs: vec4<u32>;
@group(0) @binding(6) var<storage, read_write> status: atomic<u32>;
@group(0) @binding(7) var<storage, read_write> pose: array<u32, 32>;

const GROW_NOT_CONVERGED: u32 = 1u;
const REGION_OVERFLOW: u32 = 2u;
const NO_REGIONS: u32 = 4u;
const NO_VOTES: u32 = 16u;
const DECODE_NO_ANCHOR: u32 = 1024u;

// THREE's Quaternion.setFromRotationMatrix, transcribed. A correctness constant
// rather than a structure: the four-branch form exists because each branch
// divides by a term that the others cannot guarantee is non-zero, and picking
// the largest diagonal element is what bounds the division.
fn quatFromMat(m: mat3x3<f32>) -> vec4<f32> {
  // m[col][row], so m11 is m[0][0] and m12 is m[1][0].
  let m11 = m[0][0]; let m12 = m[1][0]; let m13 = m[2][0];
  let m21 = m[0][1]; let m22 = m[1][1]; let m23 = m[2][1];
  let m31 = m[0][2]; let m32 = m[1][2]; let m33 = m[2][2];
  let trace = m11 + m22 + m33;
  if (trace > 0.0) {
    let s = 0.5 / sqrt(trace + 1.0);
    return vec4<f32>((m32 - m23) * s, (m13 - m31) * s, (m21 - m12) * s, 0.25 / s);
  }
  if (m11 > m22 && m11 > m33) {
    let s = 2.0 * sqrt(1.0 + m11 - m22 - m33);
    return vec4<f32>(0.25 * s, (m12 + m21) / s, (m13 + m31) / s, (m32 - m23) / s);
  }
  if (m22 > m33) {
    let s = 2.0 * sqrt(1.0 + m22 - m11 - m33);
    return vec4<f32>((m12 + m21) / s, 0.25 * s, (m23 + m32) / s, (m13 - m31) / s);
  }
  let s = 2.0 * sqrt(1.0 + m33 - m11 - m22);
  return vec4<f32>((m13 + m31) / s, (m23 + m32) / s, 0.25 * s, (m21 - m12) / s);
}

fn rotateByQuat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = cross(q.xyz, v) + q.w * v;
  return v + 2.0 * cross(q.xyz, t);
}

fn setF32(slot: u32, x: f32) { pose[slot] = bitcast<u32>(x); }

@compute @workgroup_size(1)
fn main() {
  // Written in full on every path. This is THE readback, so a field that keeps
  // its previous value here is a previous frame's answer handed to the caller as
  // this frame's.
  for (var k = 0u; k < 32u; k = k + 1u) { pose[k] = 0u; }

  // ── The reporting half of the status word ──
  //
  // Derived here rather than set where each condition occurs, because this pass
  // already binds the buffer carrying the evidence. The PROPAGATION half is
  // separate and does not go through status at all -- a stage that detects a
  // failure zeroes the indirect args it owns.
  if (growArgs.x != 0u) { atomicOr(&status, GROW_NOT_CONVERGED); }
  if (counts.x == 0u) { atomicOr(&status, NO_REGIONS); }
  if (counts.x > u.maxRegions) { atomicOr(&status, REGION_OVERFLOW); }
  if (lineCount.x == 0u) { atomicOr(&status, NO_VOTES); }
  if (result[0] == 0u) { atomicOr(&status, DECODE_NO_ANCHOR); }

  let correct = result[6];
  let wrong = result[7];
  let total = correct + wrong;
  let consistency = select(0.0, f32(correct) / f32(total), total > 0u);

  // Diagnostics are reported whatever happened -- an undecodable frame is the
  // case someone most wants numbers from.
  pose[10] = result[1];
  pose[13] = result[4];
  pose[14] = result[5];
  pose[15] = correct;
  pose[16] = wrong;
  pose[17] = counts.x;
  pose[18] = counts.y;
  pose[19] = lineCount.x;
  pose[20] = lay.rows;
  pose[21] = lay.cols;
  pose[22] = growArgs.w;
  setF32(9, consistency);
  setF32(23, select(0.0, lay.cellPitch / lay.distance, lay.distance > 0.0));
  setF32(24, lay.distance);
  pose[0] = atomicLoad(&status);
  if (lay.valid == 0u || result[0] == 0u) { return; }

  let orient = result[1];
  let rows = lay.rows;
  let cols = lay.cols;

  // Where the reference cell ENDS UP after the winning rotation. Its INDEX moves;
  // the cell itself does not, so its (u, v) below still use the unrotated pair.
  var rzI = lay.zeroI;
  var rzJ = lay.zeroJ;
  if (orient == 1u) { rzI = lay.zeroJ; rzJ = rows - 1u - lay.zeroI; }
  else if (orient == 2u) { rzI = rows - 1u - lay.zeroI; rzJ = cols - 1u - lay.zeroJ; }
  else if (orient == 3u) { rzI = cols - 1u - lay.zeroJ; rzJ = lay.zeroI; }

  let refRow = (result[2] + rzI) % u.torusR;
  let refCol = (result[3] + rzJ) % u.torusC;

  // ── The camera's true world orientation, in closed form ──
  var rowMath = lay.Dcol.xyz;
  var colMath = lay.Drow.xyz;
  for (var step = 0u; step < orient; step = step + 1u) {
    let nextRow = colMath;
    let nextCol = -rowMath;   // THE SIGN FLIP. See the header.
    rowMath = nextRow;
    colMath = nextCol;
  }
  let thirdMath = normalize(cross(rowMath, colMath));
  let mathBasis = mat3x3<f32>(rowMath, colMath, thirdMath);
  // The board's own axes in world space: a board ROW runs along +Z and a board
  // COLUMN along +X. Their cross is +Y, which is up -- so this basis is
  // right-handed and matches the one the triad was forced into.
  let worldBasis = mat3x3<f32>(
    vec3<f32>(0.0, 0.0, 1.0),
    vec3<f32>(1.0, 0.0, 0.0),
    vec3<f32>(0.0, 1.0, 0.0));
  // mathBasis is orthonormal by construction, so its inverse is its transpose.
  let q = quatFromMat(worldBasis * transpose(mathBasis));

  // The reference cell's position relative to the camera, then in world space.
  let refU = lay.uPhase + f32(lay.kMinU + i32(lay.zeroJ)) * lay.cellPitch;
  let refV = lay.vPhase + f32(lay.kMinV + i32(lay.zeroI)) * lay.cellPitch;
  let hitRel = rotateByQuat(q,
    lay.Drow.xyz * refU + lay.Dcol.xyz * refV - lay.normal.xyz * lay.distance);

  // And its position in world space, which is known exactly: it is a printed
  // cell of a board at a known place. Cell centres sit at the half-integers.
  let cp = lay.cellPitch;
  let refWorld = vec3<f32>(
    (f32(refCol) + 0.5 - f32(u.torusC) * 0.5) * cp,
    0.0,
    (f32(refRow) + 0.5 - f32(u.torusR) * 0.5) * cp);
  let camPos = refWorld - hitRel;

  pose[1] = 1u;
  setF32(2, camPos.x); setF32(3, camPos.y); setF32(4, camPos.z);
  setF32(5, q.x); setF32(6, q.y); setF32(7, q.z); setF32(8, q.w);
  pose[11] = refRow;
  pose[12] = refCol;
  pose[0] = atomicLoad(&status);
}
`;
