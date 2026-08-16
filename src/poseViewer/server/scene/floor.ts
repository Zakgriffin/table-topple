import * as THREE from 'three';
import { DEBUG_LAYER, GRID_STEP, VIS_HALF_EXTENT } from '../../shared/constants.ts';
import { HALF_C, HALF_R, ORDER, board, rebuildFloorPatternData } from '../../shared/floorPattern.ts';
import { globalState } from '../../shared/state.ts';
import { scene } from './renderer.ts';

// -- Floor: the actual De Bruijn torus, tiled seamlessly (it IS a torus, so
// repeat-wrapping the texture reproduces the true infinite pattern with no
// seam — the same fact the real tracker relies on to work from any crop).
// The pure DATA half (the `board` itself, ORDER, HALF_C/HALF_R,
// rebuildFloorPatternData) lives in ../floorPattern.ts (no THREE/DOM imports
// there) so it can be imported on a page with no #gl canvas. This file owns
// only the THREE-side construction (texture/mesh/grid-line geometry) on top,
// and imports the data like anyone else -- it used to re-export the whole set
// for "every existing consumer", of which there were none left.

const patternCanvas = document.createElement('canvas');
const pctx = patternCanvas.getContext('2d')!;

// Cell subdivision, directly driven by globalState.floorCellOutlineSubdiv (0:
// off, exactly today's 1-texture-pixel-per-cell flat color) -- BORDER is
// the outermost ring's thickness in subdivided pixels, always the OPPOSITE
// of the cell's own color. At subdiv 1-2, BORDER(1) alone already covers the
// whole cell (no room left for an inner square), so the cell renders as
// solid opposite-color -- a real, continuous endpoint of the same formula.
const FLOOR_OUTLINE_BORDER = 1;

export function rebuildFloorTexture() {
  const subdiv = globalState.floorCellOutlineSubdiv;
  const s = subdiv > 0 ? subdiv : 1;
  const { R, C, torus } = board;
  const width = C * s, height = R * s;
  patternCanvas.width = width; patternCanvas.height = height;
  const img = pctx.createImageData(width, height);
  for (let r = 0; r < R; r++) {
    for (let c = 0; c < C; c++) {
      // 1 -> dark, 0 -> light -- matches scripts/generate-debruijn-torus.ts's
      // canonical convention and binarize's "dark -> 1" intent (src/decode.ts).
      const inner = torus[r]![c] ? 20 : 235;
      const outer = torus[r]![c] ? 235 : 20;
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

const floorTex = new THREE.CanvasTexture(patternCanvas);
floorTex.wrapS = THREE.RepeatWrapping;
floorTex.wrapT = THREE.RepeatWrapping;
floorTex.magFilter = THREE.NearestFilter;
floorTex.colorSpace = THREE.SRGBColorSpace;
floorTex.repeat.set(1, 1); // exactly one instance of the torus, not tiled
rebuildFloorTexture(); // paint the initial pattern now that floorTex/patternCanvas both exist

const floorMat = new THREE.MeshStandardMaterial({ map: floorTex, roughness: 0.95 });
export const floorMesh = new THREE.Mesh(new THREE.PlaneGeometry(board.C * GRID_STEP, board.R * GRID_STEP), floorMat);
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
const rowGridLines = buildGridLines('row', 0xff5555);
const colGridLines = buildGridLines('col', 0x5599ff);
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
// HALF_R/HALF_C live but need their own rebuild call) or the per-device GPU
// cache keyed on the old board (pose/stages/decode/decodeTally.gpu.ts's hash table) --
// the caller (bindSlider('boardSize', ...)) is responsible for calling those
// too, in that order, since floor.ts can't import from either without a
// circular-import cycle back into itself.
export function rebuildFloorPattern(size: number) {
  if (ORDER !== 5) return;
  rebuildFloorPatternData(size);

  rebuildFloorTexture();

  floorMesh.geometry.dispose();
  floorMesh.geometry = new THREE.PlaneGeometry(board.C * GRID_STEP, board.R * GRID_STEP);

  for (const [lines, axis] of [[rowGridLines, 'row'], [colGridLines, 'col']] as const) {
    lines.geometry.dispose();
    lines.geometry = new THREE.BufferGeometry();
    lines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(computeGridLinePoints(axis), 3));
  }
}
