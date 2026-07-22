import * as THREE from 'three';
import { DEBUG_LAYER, GRID_STEP, VIS_HALF_EXTENT } from '../constants.ts';
import { ORDER5_CANDIDATE, buildLookupTableSparse, buildTorusFromCandidate, generateTorus } from '../../debruijn.ts';
import { globalState } from '../state.ts';
import { scene } from './renderer.ts';

// -- Floor: the actual De Bruijn torus, tiled seamlessly (it IS a torus, so
// repeat-wrapping the texture reproduces the true infinite pattern with no
// seam — the same fact the real tracker relies on to work from any crop).
export const ORDER = parseInt(new URLSearchParams(location.search).get('order') ?? '5', 10);
// Order 5's full R x C torus (~33.5M cells) has no known efficient
// construction free of D4 rotation/reflection collisions, so it isn't used
// directly -- ORDER5_CANDIDATE is a searched 256x256 sub-region with a low
// (1.027%) residual collision rate instead (see buildTorusFromCandidate's
// header comment in debruijn.ts). cropSize itself is now just the STARTING
// board size -- see globalState.boardSize / rebuildFloorPattern below, which
// is what actually drives it from here on (the "board size" slider in the
// global settings section). Only cropSize is runtime-adjustable; the taps/
// r0/c0 search that found this particular low-collision crop was NOT
// re-verified at every possible size, so collision rate at sizes other than
// 256 is unmeasured (not expected to be dramatically worse, just untested).
export let debruijn = ORDER === 5 ? buildTorusFromCandidate(5, ORDER5_CANDIDATE) : generateTorus(ORDER);
export let R = debruijn.R, C = debruijn.C, torus = debruijn.torus;
// For decoding an ORDER x ORDER sampled bit window back into an absolute
// torus (row,col) position -- see runPositionDecode.
export let debruijnLookup = buildLookupTableSparse(debruijn);
// One instance of the torus, sized in world units at GRID_STEP per cell —
// NOT tiled. Half-extents, since grid lines/great circles below are indexed
// out from the origin at the pattern's center.
export let HALF_C = (C * GRID_STEP) / 2;
export let HALF_R = (R * GRID_STEP) / 2;
// Floor below this many cells starts hitting degenerate cases (an empty
// canvas for rebuildFloorTexture, a zero-size GPU tally buffer, etc) --
// clamps the board-size slider's low end without needing to special-case
// every consumer for a board nobody would actually want to decode against.
const MIN_BOARD_SIZE = 8;

export const patternCanvas = document.createElement('canvas');
export const pctx = patternCanvas.getContext('2d')!;

// Cell subdivision, directly driven by globalState.floorCellOutlineSubdiv (0:
// off, exactly today's 1-texture-pixel-per-cell flat color) -- BORDER is
// the outermost ring's thickness in subdivided pixels, always the OPPOSITE
// of the cell's own color. At subdiv 1-2, BORDER(1) alone already covers the
// whole cell (no room left for an inner square), so the cell renders as
// solid opposite-color -- a real, continuous endpoint of the same formula.
export const FLOOR_OUTLINE_BORDER = 1;

export function rebuildFloorTexture() {
  const subdiv = globalState.floorCellOutlineSubdiv;
  const s = subdiv > 0 ? subdiv : 1;
  const width = C * s, height = R * s;
  patternCanvas.width = width; patternCanvas.height = height;
  const img = pctx.createImageData(width, height);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      // 1 -> dark, 0 -> light -- matches scripts/generate-debruijn-torus.ts's
      // canonical convention and binarize's "dark -> 1" intent (src/decode.ts).
      const inner = torus[r][c] ? 20 : 235;
      const outer = torus[r][c] ? 235 : 20;
      for (let sy = 0; sy < s; sy++) {
        const py = r * s + sy;
        const borderY = subdiv > 0 && (sy < FLOOR_OUTLINE_BORDER || sy >= s - FLOOR_OUTLINE_BORDER);
        for (let sx = 0; sx < s; sx++) {
          const px = c * s + sx;
          const borderX = subdiv > 0 && (sx < FLOOR_OUTLINE_BORDER || sx >= s - FLOOR_OUTLINE_BORDER);
          const v = (borderX || borderY) ? outer : inner;
          const i = (py * width + px) * 4;
          img.data[i] = img.data[i + 1] = img.data[i + 2] = v;
          img.data[i + 3] = 255;
        }
      }
    }
  }
  pctx.putImageData(img, 0, 0);
  floorTex.needsUpdate = true;
}

export const floorTex = new THREE.CanvasTexture(patternCanvas);
floorTex.wrapS = THREE.RepeatWrapping;
floorTex.wrapT = THREE.RepeatWrapping;
floorTex.magFilter = THREE.NearestFilter;
floorTex.colorSpace = THREE.SRGBColorSpace;
floorTex.repeat.set(1, 1); // exactly one instance of the torus, not tiled
rebuildFloorTexture(); // paint the initial pattern now that floorTex/patternCanvas both exist

export const floorMat = new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.95 });
export const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(C * GRID_STEP, R * GRID_STEP), floorMat);
floorMesh.rotation.x = -Math.PI / 2;
scene.add(floorMesh);

// Colored reference lines at the same integer cell boundaries the great
// circles below are computed from — row family (world +X direction, red)
// and column family (world +Z direction, blue), matching the sphere colors.
function computeGridLinePoints(axis: 'row' | 'col'): number[] {
  const half = Math.min(VIS_HALF_EXTENT, axis === 'row' ? HALF_R : HALF_C);
  const cross = axis === 'row' ? HALF_C : HALF_R;
  const pts: number[] = [];
  for (let k = -half; k <= half; k += GRID_STEP) {
    if (axis === 'row') pts.push(-cross, 0.01, k, cross, 0.01, k);
    else pts.push(k, 0.01, -cross, k, 0.01, cross);
  }
  return pts;
}
function buildGridLines(axis: 'row' | 'col', color: number): THREE.LineSegments {
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(computeGridLinePoints(axis), 3));
  return new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.35 }));
}
export const rowGridLines = buildGridLines('row', 0xff5555);
export const colGridLines = buildGridLines('col', 0x5599ff);
scene.add(rowGridLines, colGridLines);
for (const o of [rowGridLines, colGridLines]) o.layers.set(DEBUG_LAYER);

// Rebuilds the whole board at a new size (the "De Bruijn board size" global
// slider, see ui/cameraPanel.ts) -- re-crops the torus, rebuilds the decode
// lookup table, and resizes/repaints everything derived from R/C/HALF_R/
// HALF_C (floor mesh geometry, floor texture, the two reference-line
// meshes). Only meaningful for ORDER === 5 (the searched-crop path) -- the
// other orders' generateTorus has no "size" concept, it's always the one
// full R x C torus for that order, so this is a no-op there.
//
// Does NOT touch math/geometry.ts's rowLineKs/colLineKs (those read
// HALF_R/HALF_C live but need their own rebuild call) or the two per-device
// GPU caches keyed on the old board (pipelineGPU/decodeTally.ts's hash
// table, pipelineGPU/positionLM.ts's torus-brightness buffer) -- the
// caller (bindSlider('boardSize', ...)) is responsible for calling those
// too, in that order, since floor.ts can't import from either without a
// circular-import cycle back into itself.
export function rebuildFloorPattern(size: number) {
  if (ORDER !== 5) return;
  const cropSize = Math.max(MIN_BOARD_SIZE, Math.round(size));
  debruijn = buildTorusFromCandidate(5, { ...ORDER5_CANDIDATE, cropSize });
  R = debruijn.R; C = debruijn.C; torus = debruijn.torus;
  debruijnLookup = buildLookupTableSparse(debruijn);
  HALF_C = (C * GRID_STEP) / 2;
  HALF_R = (R * GRID_STEP) / 2;

  rebuildFloorTexture();

  floorMesh.geometry.dispose();
  floorMesh.geometry = new THREE.PlaneGeometry(C * GRID_STEP, R * GRID_STEP);

  for (const [lines, axis] of [[rowGridLines, 'row'], [colGridLines, 'col']] as const) {
    lines.geometry.dispose();
    lines.geometry = new THREE.BufferGeometry();
    lines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(computeGridLinePoints(axis), 3));
  }
}
