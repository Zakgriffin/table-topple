// WGSL source for the GPU port of pose/stages/decode/decodeGrid.ts's tallyPositionVotes
// -- the "every ORDER x ORDER window, in every rotation, votes for a torus
// anchor" step. Per-window logic is identical across every (orientation,
// window) pair, so this is a one-thread-per-window parallel map -- the only
// real design question is how multiple windows voting for the SAME anchor
// get combined without colliding.
//
// Unlike fitPlanes.wgsl.ts's float reduction, this collapses to a plain
// integer histogram: count occurrences of (orientation, anchorRow,
// anchorCol) triples, over a small BOUNDED key space (4 * R * C = 262144
// triples for the default 256x256 torus). WGSL's atomics are natively
// integer-only (atomic<u32>/atomic<i32>, no atomic<f32>) -- which is exactly
// what this needs, so real atomicAdd on a dense u32 counter array handles
// the collisions directly, no workaround required.
//
// The one piece with no direct WGSL equivalent is debruijnLookup itself (a
// JS Map from packed window-key -> torus position, ~65536 real entries out
// of a 2^25 possible key space -- see debruijn.ts's buildLookupTableSparse).
// decodeTally.ts builds a flat open-addressing hash table from it once
// (cached per-device, like the torus-brightness buffer in positionLM.ts),
// uploaded as parallel keys[]/values[] arrays; lookupTorus below mirrors
// that construction exactly (same hash function, same linear probing).

const HASH_HELPERS_WGSL = /* wgsl */ `
// 32-bit finisher (murmur3 fmix32) -- integer-only, so this produces the
// IDENTICAL result in JS (via Math.imul) and WGSL (u32 multiply already
// wraps mod 2^32 by spec), which is required for the table decodeTally.ts
// builds on CPU to be probed correctly from here.
fn hashU32(xIn: u32) -> u32 {
  var x = xIn;
  x = x ^ (x >> 16u);
  x = x * 0x85ebca6bu;
  x = x ^ (x >> 13u);
  x = x * 0xc2b2ae35u;
  x = x ^ (x >> 16u);
  return x;
}

// Returns 0xFFFFFFFF ("not found", matches debruijnLookup.get(key) ===
// undefined) or the packed row*C+col torus position. Bounded-loop linear
// probe (never more than tableSize steps) so a construction/hash mismatch
// bug can't hang the shader. References the module-scope hashKeys/
// hashValues bindings directly (declared alongside wherever this template
// is spliced in) rather than taking them as parameters -- WGSL helper
// functions can see module-scope resource bindings without threading them
// through explicitly.
fn lookupTorus(key: u32, tableSize: u32) -> u32 {
  var slot = hashU32(key) % tableSize;
  for (var probe = 0u; probe < tableSize; probe = probe + 1u) {
    let val = hashValues[slot];
    if (val == 0xFFFFFFFFu) { return 0xFFFFFFFFu; }
    if (hashKeys[slot] == key) { return val; }
    slot = (slot + 1u) % tableSize;
  }
  return 0xFFFFFFFFu;
}
`;

// One dispatch per orientation (4 total, from decodeTally.ts) -- simpler
// than folding all 4 into one dispatch's bounds-checking, and correctness
// doesn't depend on dispatch ordering since accumulation is atomic.
export const DECODE_TALLY_WGSL = /* wgsl */ `
struct TallyUniforms {
  gr: u32, gc: u32, orient: u32, order: u32,
  torusR: u32, torusC: u32, tableSize: u32, pad: u32,
}
@group(0) @binding(0) var<uniform> u: TallyUniforms;
@group(0) @binding(1) var<storage, read> gridBuf: array<u32>; // packed bit0=valid, bit1=bit, row-major [gr x gc]
@group(0) @binding(2) var<storage, read> hashKeys: array<u32>;
@group(0) @binding(3) var<storage, read> hashValues: array<u32>;
@group(0) @binding(4) var<storage, read_write> tally: array<atomic<u32>>; // [4 x torusR x torusC]
@group(0) @binding(5) var<storage, read_write> totalWindows: array<atomic<u32>>; // [1]

${HASH_HELPERS_WGSL}

// Mirrors decodeGrid.ts's readRotated -- same 4 index transforms, just
// returning the ORIGINAL (row,col) to index gridBuf with instead of the
// point itself.
fn origIndex(o: u32, gr: u32, gc: u32, a: u32, b: u32) -> vec2<u32> {
  if (o == 1u) { return vec2<u32>(gr - 1u - b, a); }
  if (o == 2u) { return vec2<u32>(gr - 1u - a, gc - 1u - b); }
  if (o == 3u) { return vec2<u32>(b, gc - 1u - a); }
  return vec2<u32>(a, b);
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  // rotatedDims(gr, gc, o): rows/cols swap for o == 1 or 3.
  let swap = (u.orient == 1u || u.orient == 3u);
  let rr = select(u.gr, u.gc, swap);
  let cc = select(u.gc, u.gr, swap);
  let i0 = gid.x; let j0 = gid.y;
  if (i0 + u.order > rr || j0 + u.order > cc) { return; }

  var key = 0u;
  for (var di = 0u; di < u.order; di = di + 1u) {
    for (var dj = 0u; dj < u.order; dj = dj + 1u) {
      let oi = origIndex(u.orient, u.gr, u.gc, i0 + di, j0 + dj);
      let packed = gridBuf[oi.x * u.gc + oi.y];
      if ((packed & 1u) == 0u) { return; } // incomplete window (matches CPU's 'complete' early-break)
      key = (key << 1u) | ((packed >> 1u) & 1u);
    }
  }
  atomicAdd(&totalWindows[0], 1u);

  let matchPacked = lookupTorus(key, u.tableSize);
  if (matchPacked == 0xFFFFFFFFu) { return; }
  let matchRow = matchPacked / u.torusC;
  let matchCol = matchPacked % u.torusC;
  // (matchRow - i0) mod R, done in unsigned arithmetic without underflow --
  // matchRow < R already, i0 can exceed R (grid can span >1 torus period).
  let anchorRow = (matchRow + u.torusR - (i0 % u.torusR)) % u.torusR;
  let anchorCol = (matchCol + u.torusC - (j0 % u.torusC)) % u.torusC;
  let idx = u.orient * u.torusR * u.torusC + anchorRow * u.torusC + anchorCol;
  atomicAdd(&tally[idx], 1u);
}
`;

// ── The histogram's winner, on device ────────────────────────────────────
//
// This is the stage that used to run on the HOST on both backends: the GPU path
// read the entire `hist[4*R*C]` back (331KB at a 144x144 board, the largest
// readback in decode) purely to run an argmax over it, because the correctness
// pass needs the winner in a binding and only the host could put it there.
//
// That single host dependency is what made "fused" decode THREE submits. With
// the argmax here, build -> tally -> argmax -> correctness is one encoder, one
// submit, and one 32-byte readback.
//
// ── Tie-breaking, which has to match exactly ──
//
// The host loop it replaces was `if (v > best) { best = v; bestIdx = i; }` over
// ascending i -- so: strictly greater, therefore the LOWEST index among equal
// maxima, and a winner requires at least one vote. Both halves below preserve
// that: each thread strides upward and takes `>` only, and the tree reduction
// prefers the smaller index on a tie. A thread that never saw a positive count
// keeps idx = NONE, which loses every comparison against a real one.
//
// (The CPU reference in decodeGrid.ts breaks ties by Map INSERTION order
// instead, which is neither index order nor meaningful. That difference predates
// this and is only reachable on a frame where two anchors tie for first -- an
// ambiguous decode either way. Not introduced here, and not fixed here.)
//
// ── One workgroup, deliberately ──
//
// 4*R*C is 82944 u32 at the default board and 262144 at the largest; one
// workgroup of 256 threads reads that in 324 or 1024 strided iterations, which
// is well under the cost of the readback it removes. A two-level reduction would
// use the machine better and would also need a second pass, a second buffer, and
// its own tie-break argument across blocks. Revisit if a measurement ever puts
// this on the critical path -- it currently replaces a 331KB stall.
export const DECODE_ARGMAX_WGSL = /* wgsl */ `
struct ArgmaxUniforms {
  n: u32,        // 4 * torusR * torusC
  torusR: u32, torusC: u32,
  gr: u32, gc: u32,   // the UNROTATED grid dims, for the correctness dispatch below
  wgSize: u32, pad: u32,
}
@group(0) @binding(0) var<uniform> u: ArgmaxUniforms;
@group(0) @binding(1) var<storage, read> tally: array<u32>;
@group(0) @binding(2) var<storage, read> totalWindows: array<u32>;
// The decode result block -- see DECODE_RESULT_* in decodeTally.gpu.ts for the
// slot names. This pass writes 0..5; the correctness pass adds into 6..7.
@group(0) @binding(3) var<storage, read_write> result: array<u32>;
// Workgroup counts for the correctness pass, so its extent (which depends on the
// winning ORIENTATION) never has to come back to the host either.
@group(0) @binding(4) var<storage, read_write> dispatchArgs: array<u32>;

const NONE: u32 = 0xFFFFFFFFu;
const WG: u32 = 256u;

var<workgroup> partialVotes: array<u32, 256>;
var<workgroup> partialIdx: array<u32, 256>;

@compute @workgroup_size(256)
fn argmax(@builtin(local_invocation_id) lid: vec3<u32>) {
  let t = lid.x;
  var bv = 0u;
  var bi = NONE;
  for (var i = t; i < u.n; i = i + WG) {
    let v = tally[i];
    if (v > bv) { bv = v; bi = i; }
  }
  partialVotes[t] = bv;
  partialIdx[t] = bi;
  workgroupBarrier();

  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (t < s) {
      let ov = partialVotes[t + s];
      let oi = partialIdx[t + s];
      if (ov > partialVotes[t] || (ov == partialVotes[t] && oi < partialIdx[t])) {
        partialVotes[t] = ov;
        partialIdx[t] = oi;
      }
    }
    workgroupBarrier();
  }
  if (t != 0u) { return; }

  let votes = partialVotes[0];
  let idx = partialIdx[0];
  let found = select(0u, 1u, idx != NONE);
  let rc = u.torusR * u.torusC;
  // Guarded by found rather than left to read out of a NONE index: an
  // undecodable frame is an ordinary outcome here (the host falls back to the
  // CPU decode), so it has to produce zeros rather than 0xFFFFFFFF/rc.
  let orient = select(0u, idx / rc, found == 1u);
  let rem = select(0u, idx % rc, found == 1u);
  result[0] = found;
  result[1] = orient;
  result[2] = select(0u, rem / u.torusC, found == 1u);
  result[3] = select(0u, rem % u.torusC, found == 1u);
  result[4] = votes;
  result[5] = totalWindows[0];

  // rotatedDims(gr, gc, orient), then ceil to the correctness kernel's 8x8
  // workgroup. Zero on a miss, so the pass encoded after this one runs no
  // threads at all rather than counting against a garbage anchor.
  let swap = (orient == 1u || orient == 3u);
  let rr = select(u.gr, u.gc, swap);
  let cc = select(u.gc, u.gr, swap);
  dispatchArgs[0] = select(0u, (rr + u.wgSize - 1u) / u.wgSize, found == 1u);
  dispatchArgs[1] = select(0u, (cc + u.wgSize - 1u) / u.wgSize, found == 1u);
  dispatchArgs[2] = select(0u, 1u, found == 1u);
}
`;
