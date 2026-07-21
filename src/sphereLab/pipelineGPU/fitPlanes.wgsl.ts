// WGSL source for the GPU port of pipeline/votes.ts's fitPairOfPlanes --
// specifically just the many-votes-to-21-numbers reduction (building the
// symmetric 6x6 ATA scatter matrix). The eigendecomposition that follows in
// fitPlanes.ts is a fixed-size 6x6/3x3 problem regardless of vote count, so
// it stays on CPU unchanged (see linalg.ts) -- porting a ~6-iteration Jacobi
// sweep to GPU would be pure overhead.
//
// No atomics anywhere here, deliberately: WGSL only has native atomic<u32>/
// atomic<i32>, not atomic<f32>, so a naive atomicAdd-per-vote scatter isn't
// available for this float reduction. Instead each workgroup tree-reduces
// its own 64 votes in workgroup-shared memory down to one partial-sum row of
// 21 floats (the unique upper-triangle entries of the symmetric 6x6, since
// ATA[a][b] == ATA[b][a]), then writes that row to its OWN unique slot in
// the output buffer -- workgroup N's slot, no other workgroup ever touches
// it, so there's no collision to resolve in the first place. The tiny
// final sum-of-partials (one add per workgroup, typically a few hundred)
// happens back on CPU in fitPlanes.ts.
export const FIT_PLANES_WGSL = /* wgsl */ `
struct Uniforms { voteCount: u32, maxWeight: f32, power: f32, pad: f32 }
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> votes: array<vec4<f32>>; // (nx,ny,nz,weight) per vote
@group(0) @binding(2) var<storage, read_write> outPartials: array<f32>; // numWorkgroups x 21

var<workgroup> partial: array<array<f32, 21>, 64>;

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
  @builtin(workgroup_id) wgid: vec3<u32>,
) {
  let li = lid.x;
  var row: array<f32, 6>;
  var sharpened = 0.0;
  if (gid.x < u.voteCount) {
    let v = votes[gid.x];
    if (u.maxWeight > 0.0) {
      sharpened = pow(v.w / u.maxWeight, u.power);
    }
    row[0] = v.x * v.x; row[1] = v.y * v.y; row[2] = v.z * v.z;
    row[3] = v.x * v.y; row[4] = v.x * v.z; row[5] = v.y * v.z;
  } else {
    for (var k = 0u; k < 6u; k = k + 1u) { row[k] = 0.0; }
  }

  // Pack the 21 unique a<=b products of row[a]*row[b] -- fixed iteration
  // order (a outer, b inner starting at a), matched exactly by fitPlanes.ts
  // when it unpacks the summed result back into a 6x6 matrix.
  var idx = 0u;
  for (var a = 0u; a < 6u; a = a + 1u) {
    for (var b = a; b < 6u; b = b + 1u) {
      partial[li][idx] = sharpened * row[a] * row[b];
      idx = idx + 1u;
    }
  }
  workgroupBarrier();

  var stride = 32u;
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
    let base = wgid.x * 21u;
    for (var k = 0u; k < 21u; k = k + 1u) {
      outPartials[base + k] = partial[0][k];
    }
  }
}
`;
