// WGSL source for a general EXCLUSIVE PREFIX SUM (scan) over u32.
//
// Written as a reusable primitive rather than folded into its first caller
// (collectRegionsFromLabels' CSR build), because a scan is the one piece of
// machinery this whole GPU effort was missing, and the next thing that needs a
// compaction or a variable-length grouping will need it too.
//
// Three-level structure, the standard one:
//
//   scanBlock   each workgroup scans its own BLOCK elements in shared memory
//               and writes that block's TOTAL to blockSums[]
//   (recurse)   the host scans blockSums with this same routine -- it is just
//               another array -- until it fits in a single block
//   addOffsets  every element adds its block's exclusive offset
//
// The recursion is what makes this size-independent: 196k elements is 768
// blocks, which is 3 blocks, which is 1. No fixed "max two levels" assumption
// baked into the shader.
//
// The in-block scan is Hillis-Steele (O(n log n) work) rather than Blelloch
// (O(n) work, two sweeps). At BLOCK = 256 that is 8 iterations of a single
// shared-memory add against a more complex two-phase traversal, and this scan is
// nowhere near the bottleneck of anything calling it -- simplicity wins until a
// measurement says otherwise.
// TWO modules, not one with two entry points: WGSL forbids two module-scope
// variables sharing a @group/@binding pair, and these two passes genuinely want
// different things at binding 3 (a read_write blockSums to produce, versus a
// read-only blockOffsets to consume).
export const PREFIX_SUM_SCAN_WGSL = /* wgsl */ `
struct Uniforms { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> inBuf: array<u32>;
@group(0) @binding(2) var<storage, read_write> outBuf: array<u32>;
@group(0) @binding(3) var<storage, read_write> blockSums: array<u32>;

const BLOCK = 256u;
var<workgroup> tmp: array<u32, 256>;

@compute @workgroup_size(256)
fn scanBlock(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
  let t = lid.x;
  // Threads past the end contribute 0 rather than returning early: every thread
  // has to reach the barriers below, and a partial block must still produce the
  // correct block total for the level above.
  var v = 0u;
  if (i < u.n) { v = inBuf[i]; }
  tmp[t] = v;
  workgroupBarrier();

  for (var offset = 1u; offset < BLOCK; offset = offset << 1u) {
    // Read, barrier, write, barrier -- the read of tmp[t - offset] must not race
    // the write of tmp[t] from the thread that many slots along. The loop bound
    // is uniform across the workgroup, so both barriers are uniformly reached.
    var add = 0u;
    if (t >= offset) { add = tmp[t - offset]; }
    workgroupBarrier();
    tmp[t] = tmp[t] + add;
    workgroupBarrier();
  }

  // tmp[t] is now the INCLUSIVE scan within this block; subtracting the thread's
  // own value makes it exclusive without a second pass.
  let inclusive = tmp[t];
  if (i < u.n) { outBuf[i] = inclusive - v; }
  if (t == BLOCK - 1u) { blockSums[wid.x] = inclusive; }
}
`;

export const PREFIX_SUM_ADD_WGSL = /* wgsl */ `
struct Uniforms { n: u32, pad0: u32, pad1: u32, pad2: u32 }
@group(0) @binding(0) var<uniform> u2: Uniforms;
@group(0) @binding(1) var<storage, read> inBuf2: array<u32>;
@group(0) @binding(2) var<storage, read_write> outBuf2: array<u32>;
@group(0) @binding(3) var<storage, read> blockOffsets: array<u32>;
@group(0) @binding(4) var<storage, read_write> total: array<u32>;

@compute @workgroup_size(256)
fn addOffsets(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(workgroup_id) wid: vec3<u32>,
) {
  let i = gid.x;
  if (i >= u2.n) { return; }
  let scanned = outBuf2[i] + blockOffsets[wid.x];
  outBuf2[i] = scanned;
  // The last element knows the grand total without anyone reducing again:
  // exclusive[n-1] + in[n-1]. Written here so a caller that needs the total
  // (a CSR's member count, a compaction's output length) gets it for free
  // rather than reading two elements back and adding them on the host.
  if (i == u2.n - 1u) { total[0] = scanned + inBuf2[i]; }
}
`;
