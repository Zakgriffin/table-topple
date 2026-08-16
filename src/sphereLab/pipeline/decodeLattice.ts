import type { DecodeLattice } from '../camera/model.ts';
import { board } from '../floorPattern.ts';
import type { Pose2Layout, Pose2Result } from '../../pose2/pose.ts';

// ── The decode sample lattice, rebuilt on this side ───────────────────────
//
// `src/pose2` reduces the whole grid to two integers -- how many sampled cells
// agree with the printed board and how many do not -- and reads back 128 bytes.
// The Projected-Cam lattice wants the grid itself, so this reassembles it from
// the two buffers that DO come back.
//
// Nothing here re-derives geometry. The cell POSITIONS are two lines of
// arithmetic off `layout`, and the per-cell verdict comes verbatim out of
// `packed`, which already carries decode.build's grazing / behind-camera /
// on-screen guards in bit 0. The projection stays in exactly one place.
//
// ── ONE THING IS GENUINELY DUPLICATED, AND IT HAS AN ORACLE ───────────────
//
// `origIndex` and the torus lookup below are a host copy of
// DECODE_CORRECTNESS_WGSL. There is no way around that -- the shader keeps only
// the counts -- and a silently-wrong copy is exactly the failure this project
// keeps finding: a rotation mapping that is right at orientation 0, which is
// most captures, and wrong at 1, 2 and 3.
//
// So it is CHECKED, every frame, against the device's own counters. The two
// implementations walk the same cells and must agree on both totals; when they
// do not, this returns a lattice with `correct: null` and the overlay draws the
// dots without rings. The check costs a comparison and is the only reason this
// duplication is acceptable. It also happens to cover the pattern itself: the
// `torus` read here is the same module `src/pose2/board.ts` uploads from, so a
// disagreement means the mapping, never two different boards.
export function buildDecodeLattice(
  layout: Pose2Layout, packedBytes: ArrayBuffer, resultBytes: ArrayBuffer, pose: Pose2Result,
): DecodeLattice | null {
  if (layout.valid !== 1) return null;
  const { rows, cols, cellPitch } = layout;
  if (rows === 0 || cols === 0) return null;

  // Separable: decode.build reads `uPhase + (kMinU + j) * cellPitch` for the
  // column and the same shape for the row, so the lattice is a product of two
  // 1-D sequences and there is nothing per-cell to store.
  const uAt = new Float64Array(cols);
  for (let j = 0; j < cols; j++) uAt[j] = layout.uPhase + (layout.kMinU + j) * cellPitch;
  const vAt = new Float64Array(rows);
  for (let i = 0; i < rows; i++) vAt[i] = layout.vPhase + (layout.kMinV + i) * cellPitch;

  // Clipped to the lattice: `packed` is sized for maxCells and only rows*cols of
  // it belongs to this frame.
  const packed = new Uint32Array(packedBytes).slice(0, rows * cols);

  return { rows, cols, uAt, vAt, packed, correct: correctnessOf(layout, packed, resultBytes, pose) };
}

// The rotated grid, cell by cell, scored against the printed pattern -- and then
// scored against the device, which is the point.
//
// ── THE ANCHOR COMES FROM `result`, NOT FROM THE POSE ─────────────────────
//
// `pose.boardRow` / `boardCol` are NOT the decode anchor. `finish` writes the
// REFERENCE CELL's torus position there -- the anchor offset by where the
// reference cell lands after the winning rotation (`refRow = result[2] + rzI`).
// Using them here is wrong at every orientation including 0, and it produced
// 113/136 against the device's 249/0: the same TOTAL, because the same cells are
// visited, with the pattern read at the wrong offset. That near-miss is why this
// function is checked rather than reviewed.
//
// So the anchor is READ, out of the 32-byte `result` block decode.argmax wrote
// it into. Re-deriving it would mean a host copy of finish's rzI/rzJ rotation on
// top of the origIndex copy below -- two duplications where a 32-byte readback
// removes one of them outright.
function correctnessOf(
  layout: Pose2Layout, packed: Uint32Array, resultBytes: ArrayBuffer, pose: Pose2Result,
): Int8Array | null {
  if (!pose.ok) return null;
  const result = new Uint32Array(resultBytes);
  // Slots 0..3: found, orientation, anchor row, anchor col. See DECODE_ARGMAX_WGSL.
  if (result[0] !== 1) return null;
  const orientation = result[1]!, anchorRow = result[2]!, anchorCol = result[3]!;

  const { rows, cols } = layout;
  const swap = orientation === 1 || orientation === 3;
  // The winning orientation's extent, which is the original one transposed at
  // orientations 1 and 3.
  const rr = swap ? cols : rows, cc = swap ? rows : cols;

  // Indexed in ORIGINAL grid order, because that is the order the dots are drawn
  // in. -1 is "not comparable", which covers both an unresolvable cell and the
  // fact that every original cell is visited exactly once by this loop.
  const out = new Int8Array(rows * cols).fill(-1);
  let correct = 0, wrong = 0;
  for (let i = 0; i < rr; i++) {
    for (let j = 0; j < cc; j++) {
      const [oi, oj] = origIndex(orientation, rows, cols, i, j);
      const p = packed[oi * cols + oj]!;
      if ((p & 1) === 0) continue; // unresolvable -- counted neither way, exactly as the shader does
      const bit = (p >> 1) & 1;
      const expected = board.torus[(anchorRow + i) % board.R]![(anchorCol + j) % board.C]!;
      const agrees = bit === expected;
      out[oi * cols + oj] = agrees ? 1 : 0;
      if (agrees) correct++; else wrong++;
    }
  }

  // THE ORACLE. Two independent walks of the same grid; the device's is the one
  // the consistency readout is computed from, so a disagreement means this copy
  // is wrong and its rings would be a confident lie.
  if (correct !== pose.correct || wrong !== pose.wrong) {
    console.warn(
      `[lattice] host correctness re-derivation disagrees with the device: ` +
      `${correct}/${wrong} here vs ${pose.correct}/${pose.wrong} reported. Drawing no rings.`);
    return null;
  }
  return out;
}

// A host copy of DECODE_CORRECTNESS_WGSL's function of the same name: where cell
// (a, b) of the ROTATED grid sits in the original one. Checked by the caller
// rather than trusted -- see this file's header.
function origIndex(o: number, rows: number, cols: number, a: number, b: number): [number, number] {
  if (o === 1) return [rows - 1 - b, a];
  if (o === 2) return [rows - 1 - a, cols - 1 - b];
  if (o === 3) return [b, cols - 1 - a];
  return [a, b];
}
