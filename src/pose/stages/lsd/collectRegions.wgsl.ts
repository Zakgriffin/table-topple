// WGSL source for the GPU port of pose/stages/lsd/lsdSegments.ts's
// collectRegionsFromLabels -- turning a finished CCL labeling into hysteresis-
// filtered, size-filtered, deterministically-ordered regions in CSR form.
//
// An earlier comment in lsdSegments.ts called this step "inherently serial".
// That was simply wrong, and this file is the refutation: every stage below is a
// standard parallel pattern, and the only reason it hadn't moved is that the
// CSR build needs a prefix sum, which the pipeline didn't have until
// prefixSum.wgsl.ts.
//
// Six kernels around two scans. Labels ARE pixel indices, so every per-label
// array is size n and indexed directly -- no hashing, no compaction of the label
// space, and (the useful part) the scan visits labels in ASCENDING order, which
// is exactly the deterministic region ordering the CPU path gets from sorting
// its Map keys. Region numbering is preserved EXACTLY, not merely equivalently.
//
//   survive     labelSurvives[label] = 1 where a member clears rhoHigh (tested
//               squared, no sqrt -- see lsdSegments.ts). A
//               scatter of a constant -- every writer writes 1, so no atomic.
//   histogram   counts[label]++ over surviving labels. atomicAdd.
//   markKept    per label: keptFlag = survives && count >= minRegionSize, and
//               keptCount = keptFlag ? count : 0. These are the two scan inputs.
//   -- scan keptFlag  -> regionIndex[label], total = region count
//   -- scan keptCount -> memberOffset[label], total = member count
//   regionMeta  per label: publish offset/size/label for its region index.
//   scatter     per pixel: regionId[i], and place i in the CSR via a cursor.
//   finalize    one thread per region: sort its own slice, then sum its
//               members' level-line unit vectors into a mean direction.
//
// The sort in `finalize` is what makes member order deterministic. atomicAdd
// hands out slots in arrival order, which is nondeterministic; the CPU path
// emits members in ascending pixel index because it scans the image in order.
// Regions max out around 100 members (measured), so a per-region insertion sort
// by one thread is cheap -- the same one-thread-per-region shape lsdFit already
// validated as safe here.
export const COLLECT_SURVIVE_WGSL = /* wgsl */ `
struct U { n: u32, rhoHighSq: f32, minRegionSize: u32, pad: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> label: array<i32>;
@group(0) @binding(2) var<storage, read> fx: array<f32>;
@group(0) @binding(3) var<storage, read> fy: array<f32>;
@group(0) @binding(4) var<storage, read_write> labelSurvives: array<u32>;

@compute @workgroup_size(256)
fn survive(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.n) { return; }
  let lab = label[i];
  // Benign race by construction: every thread that writes here writes the same
  // value, so the result is the OR of all of them regardless of ordering.
  // Squared, so the hysteresis test costs no sqrt. See lsdSegments.ts.
  if (lab >= 0 && fx[i] * fx[i] + fy[i] * fy[i] > u.rhoHighSq) { labelSurvives[u32(lab)] = 1u; }
}
`;

export const COLLECT_HISTOGRAM_WGSL = /* wgsl */ `
struct U { n: u32, rhoHighSq: f32, minRegionSize: u32, pad: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> label: array<i32>;
@group(0) @binding(2) var<storage, read> labelSurvives: array<u32>;
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>;

@compute @workgroup_size(256)
fn histogram(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.n) { return; }
  let lab = label[i];
  if (lab < 0) { return; }
  if (labelSurvives[u32(lab)] == 0u) { return; }
  atomicAdd(&counts[u32(lab)], 1u);
}
`;

export const COLLECT_MARK_KEPT_WGSL = /* wgsl */ `
struct U { n: u32, rhoHighSq: f32, minRegionSize: u32, pad: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> labelSurvives: array<u32>;
@group(0) @binding(2) var<storage, read> counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> keptFlag: array<u32>;
@group(0) @binding(4) var<storage, read_write> keptCount: array<u32>;

@compute @workgroup_size(256)
fn markKept(@builtin(global_invocation_id) gid: vec3<u32>) {
  let l = gid.x;
  if (l >= u.n) { return; }
  // The minRegionSize prefilter, applied here rather than after the fact, so an
  // undersized component never gets a region index, a CSR row, or a thread --
  // matching the CPU path, where it skips the label before building a region.
  let keep = select(0u, 1u, labelSurvives[l] == 1u && counts[l] >= u.minRegionSize);
  keptFlag[l] = keep;
  keptCount[l] = keep * counts[l];
}
`;

export const COLLECT_REGION_META_WGSL = /* wgsl */ `
struct U { n: u32, rhoHighSq: f32, minRegionSize: u32, pad: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> keptFlag: array<u32>;
@group(0) @binding(2) var<storage, read> regionIndex: array<u32>;
@group(0) @binding(3) var<storage, read> memberOffset: array<u32>;
@group(0) @binding(4) var<storage, read> counts: array<u32>;
@group(0) @binding(5) var<storage, read_write> regionOffsets: array<u32>;
@group(0) @binding(6) var<storage, read_write> regionSizes: array<u32>;
@group(0) @binding(7) var<storage, read> totalRegions: array<u32>;
@group(0) @binding(8) var<storage, read_write> dispatchArgs: array<u32>;

// Re-indexes the per-LABEL arrays into per-REGION ones. Everything upstream is
// keyed by label (a pixel index, sparse); everything downstream wants dense
// region ids, and this is the one place that translation happens.
//
// It ALSO writes the workgroup count for every later per-region dispatch, and
// that is why the host no longer has to read the region count back. This pass is
// the first one that runs after both prefix scans, so totalRegions is final by
// the time any thread here observes it -- and since the pass already exists, the
// dispatch args cost ZERO extra passes, which matters because plan item 12
// measured time tracking pass count.
//
// Written by thread 0 BEFORE the two early-outs below, or it would be skipped
// whenever label 0 happens to be undersized or ineligible.
//
// 64 is not arbitrary: it is the workgroup size of BOTH consumers (finalize
// here, and lsdFit's fitRegion), so one triple serves both.
@compute @workgroup_size(256)
fn regionMeta(@builtin(global_invocation_id) gid: vec3<u32>) {
  let l = gid.x;
  if (l == 0u) {
    let rc = totalRegions[0];
    dispatchArgs[0] = (rc + 63u) / 64u; // 0 regions -> 0 workgroups, which is legal and skips the dispatch
    dispatchArgs[1] = 1u;
    dispatchArgs[2] = 1u;
  }
  if (l >= u.n) { return; }
  if (keptFlag[l] == 0u) { return; }
  let r = regionIndex[l];
  regionOffsets[r] = memberOffset[l];
  regionSizes[r] = counts[l];
}
`;

export const COLLECT_SCATTER_WGSL = /* wgsl */ `
struct U { n: u32, rhoHighSq: f32, minRegionSize: u32, pad: u32 }
@group(0) @binding(0) var<uniform> u: U;
@group(0) @binding(1) var<storage, read> label: array<i32>;
@group(0) @binding(2) var<storage, read> keptFlag: array<u32>;
@group(0) @binding(3) var<storage, read> regionIndex: array<u32>;
@group(0) @binding(4) var<storage, read> memberOffset: array<u32>;
@group(0) @binding(5) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(6) var<storage, read_write> members: array<u32>;
@group(0) @binding(7) var<storage, read_write> regionId: array<i32>;

@compute @workgroup_size(256)
fn scatter(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= u.n) { return; }
  let lab = label[i];
  if (lab < 0 || keptFlag[u32(lab)] == 0u) { regionId[i] = -1; return; }
  let l = u32(lab);
  regionId[i] = i32(regionIndex[l]);
  // Slot within this label's CSR row. Arrival order is arbitrary, which is why
  // finalize sorts the row afterwards.
  let slot = atomicAdd(&cursor[l], 1u);
  members[memberOffset[l] + slot] = i;
}
`;

export const COLLECT_FINALIZE_WGSL = /* wgsl */ `
// binding 0 is the COUNTS BUFFER, not a uniform. It used to be a uniform the
// host filled in after reading the region count back -- which was the readback
// this stage existed to force. Reading it from storage is what lets the whole
// collect run in one submit with nothing crossing the bus.
// counts[0] = regionCount, counts[1] = memberCount.
@group(0) @binding(0) var<storage, read> counts: array<u32>;
@group(0) @binding(1) var<storage, read> fx: array<f32>;
@group(0) @binding(2) var<storage, read> regionOffsets: array<u32>;
@group(0) @binding(3) var<storage, read> regionSizes: array<u32>;
@group(0) @binding(4) var<storage, read_write> members: array<u32>;
@group(0) @binding(5) var<storage, read_write> meanDirs: array<vec2<f32>>;
@group(0) @binding(6) var<storage, read> fy: array<f32>;

@compute @workgroup_size(64)
fn finalize(@builtin(global_invocation_id) gid: vec3<u32>) {
  let r = gid.x;
  // Still needed even though the dispatch is now exactly sized: the workgroup
  // count is a CEILING, so the last workgroup runs threads past the end.
  if (r >= counts[0]) { return; }
  let off = regionOffsets[r];
  let size = regionSizes[r];

  // Insertion sort, ascending pixel index -- restores the CPU path's member
  // order, which its in-order image scan produces for free and the atomicAdd
  // scatter above does not. Insertion sort rather than anything cleverer
  // because regions are small (measured max ~107) and it is in-place with no
  // scratch, which matters when one thread owns the whole row.
  for (var a = 1u; a < size; a = a + 1u) {
    let key = members[off + a];
    var b = a;
    for (; b > 0u && members[off + b - 1u] > key; b = b - 1u) {
      members[off + b] = members[off + b - 1u];
    }
    members[off + b] = key;
  }

  // Plain raw sum of unit vectors -- no sign resolution against a reference
  // member. With directed growth every member is within tau of every other
  // member it is connected THROUGH, so there is no polarity flip for the sum to
  // cancel against. See collectRegionsFromLabels' own comment.
  var sc = 0.0; var ss = 0.0;
  for (var k = 0u; k < size; k = k + 1u) {
    let p = members[off + k];
    let gx = fx[p];
    let gy = fy[p];
    let inv = inverseSqrt(gx * gx + gy * gy); // every member cleared rhoLow, so m2 > 0
    sc = sc + -gy * inv;
    ss = ss + gx * inv;
  }
  // Normalized rather than left as a raw sum: only the DIRECTION is load-
  // bearing (fitRectangle uses it to pick which way along the PCA axis), but
  // normalizing keeps this directly comparable to the CPU path. The degenerate
  // all-cancel case cannot arise from directed growth; pinned, not left NaN.
  let sLen = sqrt(sc * sc + ss * ss);
  if (sLen > 0.0) { meanDirs[r] = vec2<f32>(sc / sLen, ss / sLen); }
  else { meanDirs[r] = vec2<f32>(1.0, 0.0); }
}
`;
