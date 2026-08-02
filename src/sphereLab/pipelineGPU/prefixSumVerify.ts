import { exclusiveScanU32, SCAN_BLOCK } from './prefixSum.ts';

// ── Dev harness: is the GPU exclusive scan correct? ──────────────────────
//
// Run from the devtools console:
//
//   await verifyPrefixSum()
//
// A scan is a primitive, so this tests SIZES rather than one workload -- the
// interesting failures all live at level boundaries, where a block is partial
// or the host recursion adds a level:
//
//   1, 2                    degenerate
//   255, 256, 257           one block, exactly one block, spills to two
//   65535, 65536, 65537     one full level of blocks (256*256), and its spill
//   196608                  the real target -- a 512x384 label array, 768
//                           blocks, i.e. 3 blocks, i.e. 1: three levels deep
//
// Values are deliberately not all-ones: a scan bug that drops or double-counts
// a block is invisible against a constant input if the block totals happen to
// line up, so each case uses a varying pattern and a couple use large values to
// exercise u32 accumulation.
const SIZES = [1, 2, 255, 256, 257, 511, 512, 513, 65535, 65536, 65537, 196608];

export interface PrefixSumCase {
  n: number;
  ok: boolean;
  firstBadIndex: number; // -1 when the scan matched
  expectedAt: number;
  actualAt: number;
  totalOk: boolean;
  expectedTotal: number;
  actualTotal: number;
  levels: number; // how deep the host recursion went for this n
  ms: number;
}

export interface PrefixSumVerifyReport {
  allOk: boolean;
  failures: number;
  cases: PrefixSumCase[];
}

function levelsFor(n: number): number {
  let levels = 0;
  for (let m = n; m > 1; m = Math.ceil(m / SCAN_BLOCK)) levels++;
  return Math.max(1, levels);
}

export async function verifyPrefixSum(): Promise<PrefixSumVerifyReport | string> {
  const cases: PrefixSumCase[] = [];
  for (const n of SIZES) {
    const values = new Uint32Array(n);
    // Varying, non-constant, and occasionally large -- see the header.
    for (let i = 0; i < n; i++) values[i] = (i % 7) + ((i % 1000 === 0) ? 100000 : 0);

    const t0 = performance.now();
    const got = await exclusiveScanU32(values);
    const ms = performance.now() - t0;
    if (!got) return 'exclusiveScanU32 returned null (WebGPU unavailable, or a validation error -- check the console)';

    let running = 0, firstBad = -1, expectedAt = 0, actualAt = 0;
    for (let i = 0; i < n; i++) {
      if (got.scan[i] !== running && firstBad === -1) {
        firstBad = i; expectedAt = running; actualAt = got.scan[i];
      }
      running += values[i];
    }
    cases.push({
      n, ok: firstBad === -1, firstBadIndex: firstBad, expectedAt, actualAt,
      totalOk: got.total === running, expectedTotal: running, actualTotal: got.total,
      levels: levelsFor(n), ms: +ms.toFixed(2),
    });
  }
  const failures = cases.filter((c) => !c.ok || !c.totalOk).length;
  return { allOk: failures === 0, failures, cases };
}
