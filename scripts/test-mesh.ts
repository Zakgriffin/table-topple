// Validates src/mesh.ts's buildMesh against a realistic multi-junction
// synthetic patch (a random cell grid, not a single isolated junction like
// scripts/test-junctions.ts) with exact ground truth by construction: checks
// that mesh-relative (row,col) coordinates correctly track TRUE lattice
// index differences, that positions are accurate, and that coverage of the
// reachable (non-flat/straight) lattice is high.
//
// The whole canvas is filled with random pattern, not just a central patch
// on a white background — an earlier version used a white margin "for
// derivative-window safety" and that margin's own hard edge was detected as
// spurious corner-like structure, contaminating the results. Ground truth is
// only checked in an interior region, away from the real canvas edge.
//
// Usage: node scripts/test-mesh.ts

import { computeJunctionField, detectJunctions, refineJunctionSubPixel } from '../src/cornerdetect.ts';
import { buildMesh } from '../src/mesh.ts';
import type { RawJunction } from '../src/mesh.ts';

const CELL = 20;
const N = 24; // NxN cells, whole canvas
const BORDER = 5; // cells excluded from ground-truth checking near the canvas edge
const W = N * CELL, H = W;

function preBlur(gray: Float64Array, w: number, h: number, radius: number): Float64Array {
  const out = new Float64Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let sum = 0, count = 0;
      for (let dy = -radius; dy <= radius; dy++) {
        const yy = y + dy; if (yy < 0 || yy >= h) continue;
        for (let dx = -radius; dx <= radius; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= w) continue;
          sum += gray[yy * w + xx]; count++;
        }
      }
      out[y * w + x] = sum / count;
    }
  }
  return out;
}

function rotatePoint(x: number, y: number, cx: number, cy: number, theta: number): [number, number] {
  const dx = x - cx, dy = y - cy, c = Math.cos(theta), s = Math.sin(theta);
  return [dx * c - dy * s + cx, dy * c + dx * s + cy];
}

function runTrial(thetaDeg: number) {
  const cells: number[][] = Array.from({ length: N }, () => Array.from({ length: N }, () => (Math.random() < 0.5 ? 1 : 0)));

  const rawGray = new Float64Array(W * H);
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const ci = Math.floor(y / CELL), cj = Math.floor(x / CELL);
      rawGray[y * W + x] = cells[ci][cj] ? 0 : 255;
    }
  }
  const theta = thetaDeg * Math.PI / 180;
  const cx0 = W / 2, cy0 = H / 2;
  let gray: Float64Array<ArrayBufferLike> = rawGray;
  if (thetaDeg !== 0) {
    const rotated = new Float64Array(W * H);
    const c = Math.cos(theta), s = Math.sin(theta);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const dx = x - cx0, dy = y - cy0;
        const sx = Math.round(dx * c - dy * s + cx0), sy = Math.round(dx * s + dy * c + cy0);
        rotated[y * W + x] = (sx >= 0 && sx < W && sy >= 0 && sy < H) ? rawGray[sy * W + sx] : 255;
      }
    }
    gray = rotated;
  }
  gray = preBlur(gray, W, H, 2);

  const field = computeJunctionField(gray, W, H);
  const coarse = detectJunctions(field);
  const junctions: RawJunction[] = coarse.map(j => {
    const r = refineJunctionSubPixel(gray, W, H, j.x, j.y);
    return { x: r.x, y: r.y, type: j.type };
  });

  function trueType(i: number, j: number): 'none' | 'lcorner' | 'saddle' {
    const tl = cells[i - 1][j - 1], tr = cells[i - 1][j], bl = cells[i][j - 1], br = cells[i][j];
    const allSame = tl === tr && tr === bl && bl === br;
    if (allSame) return 'none';
    if (tl === br && tr === bl) return 'saddle';
    if ((tl === tr && bl === br) || (tl === bl && tr === br)) return 'none';
    return 'lcorner';
  }
  // Ground truth covers the WHOLE canvas, not just the interior — a
  // junction right at the interior boundary is real and correctly detected,
  // and needs its own true match available for nearest-point lookup rather
  // than being compared against whatever interior point happens to be
  // nearest (border-adjacent junctions were showing up as false mismatches
  // against a much farther interior point before this fix). Only the
  // interior is actually SCORED, below.
  const truePoints: { i: number; j: number; x: number; y: number; type: 'lcorner' | 'saddle' }[] = [];
  for (let i = 1; i < N; i++) {
    for (let j = 1; j < N; j++) {
      const t = trueType(i, j);
      if (t === 'none') continue;
      // Raster rotation above uses inverse-sampling (each destination pixel
      // looks up source content via +theta), which means the image CONTENT
      // is rotated by -theta relative to the original — ground-truth points
      // must be rotated the same way, not by +theta.
      const [x, y] = rotatePoint(j * CELL, i * CELL, cx0, cy0, -theta);
      truePoints.push({ i, j, x, y, type: t });
    }
  }
  const interior = (x: number, y: number) => x > BORDER * CELL && x < (N - BORDER) * CELL && y > BORDER * CELL && y < (N - BORDER) * CELL;

  const seedX = W / 2, seedY = H / 2;
  // Same -theta correction as the ground-truth points above — the actual
  // image content is rotated by -theta relative to the original (see the
  // raster rotation's inverse-sampling comment), so that's the direction
  // the mesh's seed basis vectors need to point in too.
  const mesh = buildMesh(junctions, seedX, seedY, CELL, CELL, -theta);
  const interiorNodes = mesh.nodes.filter(n => interior(n.x, n.y));

  const seedNode = mesh.nodes.find(n => n.row === 0 && n.col === 0)!;
  let seedTrue = truePoints[0], bestD = Infinity;
  for (const tp of truePoints) {
    const d = (tp.x - seedNode.x) ** 2 + (tp.y - seedNode.y) ** 2;
    if (d < bestD) { bestD = d; seedTrue = tp; }
  }

  const posErrors: number[] = [];
  let coordMismatches = 0, typeMismatches = 0;
  for (const node of interiorNodes) {
    let nearestTrue = truePoints[0], d0 = Infinity;
    for (const tp of truePoints) {
      const d = (tp.x - node.x) ** 2 + (tp.y - node.y) ** 2;
      if (d < d0) { d0 = d; nearestTrue = tp; }
    }
    posErrors.push(Math.sqrt(d0));
    const expectedRow = nearestTrue.i - seedTrue.i, expectedCol = nearestTrue.j - seedTrue.j;
    if (node.row !== expectedRow || node.col !== expectedCol) coordMismatches++;
    if (node.type !== nearestTrue.type) typeMismatches++;
  }

  const meanErr = posErrors.reduce((a, b) => a + b, 0) / posErrors.length;
  const maxErr = Math.max(...posErrors);
  const interiorTruePoints = truePoints.filter(tp => interior(tp.x, tp.y));
  const coverage = interiorNodes.length / interiorTruePoints.length;
  console.log(`theta=${thetaDeg}deg: mesh nodes=${interiorNodes.length}/${truePoints.length} (coverage=${(coverage * 100).toFixed(0)}%) posErr mean=${meanErr.toFixed(2)}px max=${maxErr.toFixed(2)}px coordMismatches=${coordMismatches} typeMismatches=${typeMismatches}`);
  return { coordMismatches, typeMismatches, maxErr, coverage };
}

let failures = 0;
for (const thetaDeg of [0, 10, 25, 40]) {
  const r = runTrial(thetaDeg);
  if (r.coordMismatches > 0 || r.typeMismatches > 0 || r.maxErr > 3 || r.coverage < 0.7) failures++;
}
console.log(`\n${failures === 0 ? 'PASS' : `FAIL (${failures}/4 trials outside tolerance)`}`);
if (failures > 0) process.exit(1);
