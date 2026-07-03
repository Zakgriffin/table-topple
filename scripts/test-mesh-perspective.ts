// The actual payoff test: validates the corner-mesh pipeline (cornerdetect +
// mesh) against REAL perspective distortion from scripts/lib/synth-camera.ts
// (not just rotation), at the same tilt angles scripts/test-perspective.ts
// showed the old rotation+uniform-scale pipeline collapsing at (0% correct
// by 20deg tilt). Ground truth comes directly from the known camera pose:
// each mesh node's presumed true lattice point (its own detected image
// position, inverse-projected to the world plane and rounded to the nearest
// real lattice point via the seed's own local difference) is projected back
// FORWARD through the same pose, and compared against where the node was
// actually detected.
//
// This does NOT check that the mesh's (row,col) axes line up with any FIXED
// world axis convention (e.g. "col always increases with world X") — under
// rotation, which screen direction "col" points in depends on the estimated
// rotation, exactly like the existing pipeline's 4-way reading-orientation
// ambiguity that pickBestCandidate already resolves via correlation, not by
// assuming a fixed orientation. What must hold, and what this validates, is
// that positions are self-consistent with SOME real rigid mapping from mesh
// coordinates to the pattern.
//
// Usage: node scripts/test-mesh-perspective.ts

import { PNG } from 'pngjs';
import { readFileSync } from 'node:fs';
import { toGrayscale, binarize, detectGrid, estimateRotationRad } from '../src/decode.ts';
import { computeJunctionField, detectJunctions, refineJunctionSubPixel } from '../src/cornerdetect.ts';
import { buildMesh } from '../src/mesh.ts';
import { captureHomography, makeHomographySampler, projectToImage } from './lib/synth-camera.ts';
import type { CameraPose } from './lib/synth-camera.ts';

const png = PNG.sync.read(readFileSync('samples/order4.png'));
const cellPx = png.width / 257; // matches generate-debruijn-torus.ts's order-4 torus dimensions

const RAW = 300;
const DIST = 300, FOCAL = 300;
const TRIALS = 5;

// detectGrid measures pitch via horizontal/vertical adjacent-pixel
// differences, implicitly assuming the grid lines are axis-aligned — under
// roll (in-plane rotation) the true lines are diagonal, so this overestimates
// pitch by roughly 1/cos(roll mod 90deg), up to ~41% at the worst case
// (45deg). That's enough to push the very first mesh search step outside its
// +-35% tolerance before any real vectors are established. Fix: derotate a
// small central patch by the already-computed rotation estimate before
// measuring pitch on it — same derotation this pipeline already does
// elsewhere (see src/main.ts's decodeFrame), just applied here only to get
// an accurate SEED rather than to the whole buffer.
const PATCH = 120;
function derotatePatch(gray: Float64Array, w: number, h: number, theta: number): Float64Array {
  const out = new Float64Array(PATCH * PATCH);
  const cosT = Math.cos(theta), sinT = Math.sin(theta), cx = w / 2, cy = h / 2;
  for (let ay = 0; ay < PATCH; ay++) {
    const relY = ay - PATCH / 2;
    for (let ax = 0; ax < PATCH; ax++) {
      const relX = ax - PATCH / 2;
      const sx = Math.round(relX * cosT - relY * sinT + cx);
      const sy = Math.round(relX * sinT + relY * cosT + cy);
      out[ay * PATCH + ax] = (sx >= 0 && sx < w && sy >= 0 && sy < h) ? gray[sy * w + sx] : 255;
    }
  }
  return out;
}

const TILT_DEGREES = [0, 10, 20, 30, 40, 50, 60];

for (const tiltDeg of TILT_DEGREES) {
  let trialsRun = 0, trialsFailed = 0, totalNodes = 0;
  const posErrors: number[] = [];

  for (let t = 0; t < TRIALS; t++) {
    const pose: CameraPose = {
      targetX: Math.random() * png.width, targetY: Math.random() * png.height,
      dist: DIST, focal: FOCAL, tilt: tiltDeg * Math.PI / 180,
      azimuth: Math.random() * 2 * Math.PI, roll: Math.random() * 2 * Math.PI,
    };
    const rgba = captureHomography(png, pose, RAW, RAW);
    const gray = toGrayscale(rgba, RAW, RAW);

    // Coarse rotation estimate — robust under any roll by construction (see
    // estimateRotationRad's mod-90 folding), used only as a search SEED for
    // buildMesh; no full derotation is needed for corner detection itself,
    // already validated as rotation-robust (scripts/test-junctions.ts).
    const theta0 = estimateRotationRad(gray, RAW, RAW);

    // Pitch, unlike rotation, DOES need derotation first to be accurate
    // under roll (see the derotatePatch comment above) — measured on a
    // small central patch, not the whole buffer, since it's only a seed.
    const patchGray = derotatePatch(gray, RAW, RAW, theta0);
    const patchBin = binarize(patchGray);
    const coarseGrid = detectGrid(patchBin, PATCH, PATCH);

    // tensorRadius/minDistance must scale with the actual apparent pitch —
    // the module's hardcoded defaults were only validated against earlier
    // synthetic tests' 20-40px cells; this pattern's real cell pitch is
    // 8px, an order of magnitude smaller.
    const apparentPitch = (coarseGrid.pitchX + coarseGrid.pitchY) / 2;
    const tensorRadius = Math.max(1, Math.round(apparentPitch / 10));
    const minDistance = Math.max(2, Math.round(apparentPitch / 4));

    const field = computeJunctionField(gray, RAW, RAW, 1, tensorRadius);
    const coarseJ = detectJunctions(field, 0.15, minDistance);
    if (coarseJ.length < 5) { trialsFailed++; continue; }
    const junctions = coarseJ.map(j => {
      const r = refineJunctionSubPixel(gray, RAW, RAW, j.x, j.y);
      return { x: r.x, y: r.y, type: j.type };
    });

    const seedX = RAW / 2, seedY = RAW / 2;
    const mesh = buildMesh(junctions, seedX, seedY, coarseGrid.pitchX, coarseGrid.pitchY, -theta0);
    if (mesh.nodes.length < 3) { trialsFailed++; continue; }

    const sampler = makeHomographySampler(pose, RAW, RAW);
    const seedNode = mesh.nodes.find(n => n.row === 0 && n.col === 0)!;
    const seedWorld = sampler(seedNode.x, seedNode.y);
    if (!seedWorld) { trialsFailed++; continue; }

    for (const node of mesh.nodes) {
      const w = sampler(node.x, node.y);
      if (!w) continue;
      // Round the DIFFERENCE from the seed, not each point's absolute world
      // position independently — at cellPx=8px, the few-px positional noise
      // from inverting through the homography is a large enough fraction of
      // one cell that independently rounding each endpoint first (then
      // subtracting) frequently rounds them to DIFFERENT nearby integers
      // even when they're really consistent. Rounding the raw difference
      // first lets correlated error (e.g. from the shared pose) cancel out.
      const dCol = Math.round((w[0] - seedWorld[0]) / cellPx);
      const dRow = Math.round((w[1] - seedWorld[1]) / cellPx);
      const trueX = seedWorld[0] + dCol * cellPx, trueY = seedWorld[1] + dRow * cellPx;
      const proj = projectToImage(pose, RAW, RAW, trueX, trueY);
      if (proj) posErrors.push(Math.hypot(proj[0] - node.x, proj[1] - node.y));
    }
    totalNodes += mesh.nodes.length;
    trialsRun++;
  }

  const meanErr = posErrors.length ? posErrors.reduce((a, b) => a + b, 0) / posErrors.length : NaN;
  const maxErr = posErrors.length ? Math.max(...posErrors) : NaN;
  console.log(`tilt ${String(tiltDeg).padStart(2)}deg: ${trialsRun}/${TRIALS} trials produced a mesh (${trialsFailed} no-signal), avg ${(totalNodes / Math.max(1, trialsRun)).toFixed(1)} nodes/trial, posErr mean=${meanErr.toFixed(2)}px max=${maxErr.toFixed(2)}px n=${posErrors.length}`);
}
