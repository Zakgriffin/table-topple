// WGSL source for the GPU port of pipeline/votes.ts's votesInMagnitudeBand --
// finding the [minPercent, maxPercent) rank band by weight and returning
// just those votes. The CPU version does this via a full comparator sort of
// every vote (O(n log n)) purely to locate two percentile cutoffs -- but
// nothing downstream needs the votes in ORDER, only "is this vote's weight
// inside the band," so a full sort is solving a harder problem than the one
// that actually exists. This does it in two O(n) parallel passes instead:
//
//   1. HISTOGRAM_WGSL: bucket every vote's weight into a fixed number of
//      buckets, atomicAdd the bucket counter (real atomic<u32>, no
//      float-atomics workaround needed -- same reasoning as decodeTally).
//   2. (CPU, voteBandSelect.ts) walk the (tiny, NUM_BUCKETS-sized) histogram
//      from the top bucket down to translate the two rank cutoffs into two
//      weight thresholds. This is O(buckets), not O(votes).
//   3. COMPACT_FILTER_WGSL: re-scan every vote, test against the two weight
//      thresholds, and if in-band, atomicAdd a global output-index counter
//      and write the vote into that compacted slot -- real GPU stream
//      compaction, using the same "atomics are integer-only" fact, this
//      time for an index counter rather than a count.
//
// Trade-off, stated plainly: bucketed threshold selection is approximate
// (bucket-resolution precision) versus the CPU's exact-rank cutoff. Nothing
// downstream needs an exact rank -- fitPairOfPlanes only uses band
// membership as a weight-sharpening heuristic -- so this is verified
// against the CPU version by comparing resulting vote counts/fit output,
// not by requiring byte-identical membership.

export const HISTOGRAM_WGSL = /* wgsl */ `
struct HistUniforms { voteCount: u32, numBuckets: u32, maxWeight: f32, pad: f32 }
@group(0) @binding(0) var<uniform> u: HistUniforms;
@group(0) @binding(1) var<storage, read> votes: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> hist: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.voteCount) { return; }
  let weight = votes[gid.x].w;
  var bucket = 0u;
  if (u.maxWeight > 0.0) {
    bucket = min(u.numBuckets - 1u, u32((weight / u.maxWeight) * f32(u.numBuckets)));
  }
  atomicAdd(&hist[bucket], 1u);
}
`;

export const COMPACT_FILTER_WGSL = /* wgsl */ `
struct FilterUniforms { voteCount: u32, pad0: u32, loThresh: f32, hiThresh: f32 }
@group(0) @binding(0) var<uniform> u: FilterUniforms;
@group(0) @binding(1) var<storage, read> votesIn: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> votesOut: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> outCount: array<atomic<u32>>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= u.voteCount) { return; }
  let v = votesIn[gid.x];
  if (v.w >= u.loThresh && v.w < u.hiThresh) {
    let idx = atomicAdd(&outCount[0], 1u);
    votesOut[idx] = v;
  }
}
`;
