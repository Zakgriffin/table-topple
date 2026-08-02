// WGSL source for the GPU port of pipeline/decodeGrid.ts's
// buildDecodeSampleGrid -- one thread per lattice cell, reverse-projecting a
// floor point to a pixel and sampling its bit.
//
// The point of porting this is NOT the arithmetic (it's ~3ms of ray-casts at a
// zoomed-out 270x276 grid, ~0.17ms at a close-up 60x59 one). It is that the
// packed grid it produces is exactly what decodeTally.wgsl.ts consumes, and
// today that grid is built on CPU and UPLOADED -- 298KB per call at the large
// size. Building it here means the tally reads it in place and it never crosses
// the bus at all.
//
// TWO output buffers, deliberately split by who needs them:
//
//   gridPacked  bit0 = valid, bit1 = sampled bit. The tally's entire input.
//               Stays device-resident; never read back on the pose path.
//   gridGeom    (u, v, px, py) per cell. Nothing on the pose critical path
//               reads this -- it exists only for the debug overlays and the
//               phone's AR readout, so it is read back lazily, on demand.
//
// The pose path needs neither: the winner comes back from the tally, and the
// reference point's own u/v are pure arithmetic in (i, j) that the host redoes
// in f64 rather than reading back one vec4.
export const DECODE_GRID_BUILD_WGSL = /* wgsl */ `
struct Uniforms {
  drow: vec4<f32>,     // xyz used
  dcol: vec4<f32>,
  normal: vec4<f32>,   // already sign-corrected by the host, same as the CPU path
  invQuat: vec4<f32>,  // (x, y, z, w)
  // distance, tan(vFov/2), aspect, minGrazingCos
  geom: vec4<f32>,
  // uPhase, vPhase, GRID_STEP, binThreshold
  lattice: vec4<f32>,
  dims: vec4<u32>,     // rows, cols, imageW, imageH
  kmin: vec4<i32>,     // kMinU, kMinV, unused, unused
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> gridPacked: array<u32>;
@group(0) @binding(3) var<storage, read_write> gridGeom: array<vec4<f32>>;

// v + 2*cross(q.xyz, cross(q.xyz, v) + q.w*v) -- the standard quaternion
// sandwich without building a matrix. Matches THREE's applyQuaternion.
fn rotateByQuat(q: vec4<f32>, v: vec3<f32>) -> vec3<f32> {
  let t = cross(q.xyz, v) + q.w * v;
  return v + 2.0 * cross(q.xyz, t);
}

@compute @workgroup_size(8, 8)
fn build(@builtin(global_invocation_id) gid: vec3<u32>) {
  let rows = u.dims.x; let cols = u.dims.y;
  let i = gid.x; let j = gid.y;
  if (i >= rows || j >= cols) { return; }

  let gridStep = u.lattice.z;
  let v = u.lattice.y + f32(u.kmin.y + i32(i)) * gridStep;
  let uu = u.lattice.x + f32(u.kmin.x + i32(j)) * gridStep;

  let distance = u.geom.x;
  let normal = u.normal.xyz;
  // p is this floor point's position RELATIVE TO THE CAMERA at the analysis
  // frame's origin -- same construction as the CPU path.
  let p = u.drow.xyz * uu + u.dcol.xyz * v - normal * distance;

  let idx = i * cols + j;
  gridGeom[idx] = vec4<f32>(uu, v, 0.0, 0.0); // px/py filled in below

  let lenP = length(p);
  if (!(lenP > 0.0)) { gridPacked[idx] = 0u; return; }
  let rayDir = p / lenP;
  // Same grazing cutoff projectSamples applies going the other direction, so a
  // lattice point only decodes if the ray to it is one the projected image
  // actually respects.
  let grazingOk = dot(rayDir, normal) < -u.geom.w;

  let local = rotateByQuat(u.invQuat, p);
  let tanHalf = u.geom.y;
  let denom = local.z * tanHalf;
  // Guarding the divisor rather than testing the quotient for finiteness after
  // the fact: WGSL gives no isFinite, and a shader compiler is entitled to
  // assume operands are finite, so a self-comparison NaN test is not reliable.
  // This is the one place the port is structurally different from the CPU's
  // Number.isFinite check, and it is stricter, not looser.
  if (denom == 0.0 || local.z == 0.0) { gridPacked[idx] = 0u; return; }
  let ndcU = -local.x / (denom * u.geom.z);
  let ndcV = -local.y / denom;
  let fw = f32(u.dims.z); let fh = f32(u.dims.w);
  let px = ((ndcU + 1.0) * 0.5) * fw;
  let py = ((1.0 - ndcV) * 0.5) * fh;
  gridGeom[idx] = vec4<f32>(uu, v, px, py);

  let inBounds = px >= 0.0 && px < fw && py >= 0.0 && py < fh;
  if (!(grazingOk && inBounds)) { gridPacked[idx] = 0u; return; }

  // CLAMPED, exactly as the CPU path is: the bounds test above uses the
  // UNROUNDED value, so px = w - 0.3 passes it and then rounds to w. Measured
  // 46 such cells in a 270x276 grid. Out-of-range indexing is defined-but-
  // wrong in WGSL rather than undefined, which would have hidden the bug
  // instead of surfacing it the way the CPU's Uint8Array did.
  let xx = u32(clamp(round(px), 0.0, fw - 1.0));
  let yy = u32(clamp(round(py), 0.0, fh - 1.0));
  let bit = select(0u, 1u, gray[yy * u.dims.z + xx] < u.lattice.w);
  gridPacked[idx] = 1u | (bit << 1u);
}
`;

// Correctness counting, run AFTER the tally has picked a winner.
//
// This exists so `consistency` -- the one decode number that is always on
// screen -- can be produced without reading the grid back. It reduces the whole
// rotated grid to TWO integers. The per-cell correctness ARRAY that the
// projected-cam overlay draws is not built here; that is overlay-only, and is
// materialized from a lazy readback when something actually asks for it.
//
// Mirrors runPositionDecode's own loop exactly, including that it walks the
// ROTATED grid and skips invalid cells.
export const DECODE_CORRECTNESS_WGSL = /* wgsl */ `
struct Uniforms {
  gr: u32, gc: u32, orient: u32, pad0: u32,
  torusR: u32, torusC: u32, anchorRow: u32, anchorCol: u32,
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> gridPacked: array<u32>;
@group(0) @binding(2) var<storage, read> torus: array<u32>; // [torusR * torusC], 0/1
@group(0) @binding(3) var<storage, read_write> counts: array<atomic<u32>>; // [correct, wrong]

// Identical to decodeTally.wgsl.ts's origIndex -- readRotated's index map.
fn origIndex(o: u32, gr: u32, gc: u32, a: u32, b: u32) -> vec2<u32> {
  if (o == 1u) { return vec2<u32>(gr - 1u - b, a); }
  if (o == 2u) { return vec2<u32>(gr - 1u - a, gc - 1u - b); }
  if (o == 3u) { return vec2<u32>(b, gc - 1u - a); }
  return vec2<u32>(a, b);
}

@compute @workgroup_size(8, 8)
fn correctness(@builtin(global_invocation_id) gid: vec3<u32>) {
  let swap = (u.orient == 1u || u.orient == 3u);
  let rr = select(u.gr, u.gc, swap);
  let cc = select(u.gc, u.gr, swap);
  let i = gid.x; let j = gid.y;
  if (i >= rr || j >= cc) { return; }

  let oi = origIndex(u.orient, u.gr, u.gc, i, j);
  let packed = gridPacked[oi.x * u.gc + oi.y];
  if ((packed & 1u) == 0u) { return; } // invalid cell -- not counted either way
  let bit = (packed >> 1u) & 1u;

  let torusRow = (u.anchorRow + i) % u.torusR;
  let torusCol = (u.anchorCol + j) % u.torusC;
  if (bit == torus[torusRow * u.torusC + torusCol]) {
    atomicAdd(&counts[0], 1u);
  } else {
    atomicAdd(&counts[1], 1u);
  }
}
`;
