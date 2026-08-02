// WGSL source for the GPU 2x2 gradient field -- the GPU-resident
// counterpart to pipeline/gradientField.ts's computeGradient2x2Field.
//
// Deliberately NOT voteGeneration.wgsl.ts's GRADIENT_WGSL (the centered-
// difference field, mirroring computeGradientField) reused with r=1 --
// that kernel zeros a SYMMETRIC r-pixel margin on all 4 sides, whereas
// computeGradient2x2Field's 2x2-block gradient only leaves the LAST row
// and column zero (the first row/column are valid data, same as every
// other pixel). Copy-pasting the symmetric version would silently zero out
// real data along the top/left edge.
export const GRADIENT_2X2_WGSL = /* wgsl */ `
struct Dims { w: u32, h: u32, pad0: u32, pad1: u32 }
@group(0) @binding(0) var<uniform> dims: Dims;
@group(0) @binding(1) var<storage, read> gray: array<f32>;
@group(0) @binding(2) var<storage, read_write> fxOut: array<f32>;
@group(0) @binding(3) var<storage, read_write> fyOut: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y;
  if (x >= dims.w || y >= dims.h) { return; }
  let i = y * dims.w + x;
  if (x >= dims.w - 1u || y >= dims.h - 1u) {
    fxOut[i] = 0.0; fyOut[i] = 0.0;
    return;
  }
  let g00 = gray[i]; let g10 = gray[i + 1u]; let g01 = gray[i + dims.w]; let g11 = gray[i + 1u + dims.w];
  // norm = 1/(2*255) -- matches gradientField.ts's computeGradient2x2Field
  // normalization so hypot(fx,fy)'s true ceiling is 1, not 255.
  let norm = 1.0 / 510.0;
  fxOut[i] = ((g10 + g11) - (g00 + g01)) * norm;
  fyOut[i] = ((g01 + g11) - (g00 + g10)) * norm;
}
`;
