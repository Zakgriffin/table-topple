// ── The pipeline, as DATA ─────────────────────────────────────────────────
//
// Every buffer the flat pose pipeline uses, and every pass that binds one, in
// execution order. See full_system_breakdown.md for what each stage computes.
//
// This file is a declaration, not an implementation. It exists because three
// things are derived from it rather than written by hand, and a derived fact
// cannot disagree with the code the way a maintained table can:
//
//   1. BUFFER LIVENESS -- first bind to last bind. buffers.ts colours the
//      resulting intervals to share allocations (§18).
//   2. THE CLEAR SCHEDULE -- a `zero` buffer is cleared immediately before the
//      first stage that binds it, wherever that lands.
//   3. TWO VALIDATION RULES that WebGPU only reports asynchronously, i.e. as a
//      silently no-op encoder rather than an exception:
//        - no pass may bind one GPUBuffer twice (usage scope), and
//        - a pass's indirect-args source may not appear in its own bind list.
//
// ── THE ONE RULE THAT IS EASY TO GET WRONG ──
//
// `binds` is what the pass BINDS, not what its shader READS. Grow's four entry
// points share one explicit bind-group layout, so `compress` binds fx/fy without
// touching them -- and that binding is what keeps fx/fy's live range open across
// the round loop. Trimming `binds` down to what a shader actually reads would
// free them early and hand `compress` an aliased buffer inside its own bind
// group, which is precisely the failure the colouring exists to make impossible.

export interface Dims {
  w: number;
  h: number;
  /** Cap on detected line-support regions. Overflow sets a status bit. */
  maxRegions: number;
  /** Cap on accepted line segments. Overflow sets a status bit. */
  maxLines: number;
  /** Cap on decode lattice cells. Overflow sets a status bit. */
  maxCells: number;
  /** De Bruijn torus dimensions -- the printed board. */
  torusR: number;
  torusC: number;
  /** Open-addressing table slots for the De Bruijn lookup (a power of two). */
  hashSlots: number;
}

export type Kind =
  /** A storage buffer. The only kind that participates in pooling. */
  | 'storage'
  /** A uniform block. Dedicated -- tiny, and a different usage flag. */
  | 'uniform'
  /** Consumed by dispatchWorkgroupsIndirect. Dedicated, and never pooled. */
  | 'indirect'
  /** Built once per device and kept across frames (torus, hash table). */
  | 'persistent';

export interface BufferSpec {
  kind: Kind;
  bytes: (d: Dims) => number;
  /**
   * Must be cleared to zero before its first bind. These are the atomicAdd
   * accumulators, plus every indirect-args buffer (the case that covers is the
   * pass that WRITES the args not running: a failed bind group is a silent
   * no-op, while the indirect dispatch reading them is a separate valid command
   * that would launch over the previous frame's extent).
   *
   * Buffers whose correct initial state is NOT zero are marked `written`
   * instead -- see below. Those are the dangerous ones, because a grep for
   * `zero` misses them.
   */
  zero?: boolean;
  /**
   * Correct initial state is not zero, so it needs a WRITE, not a clear. Value
   * documented per buffer; the writer is the stage named here.
   */
  written?: string;
  note?: string;
}

export interface Stage {
  id: string;
  /** Ordered bind-group entries. See the header: BINDS, not reads. */
  binds: readonly string[];
  /**
   * The buffer this pass takes its workgroup count from, if it dispatches
   * indirectly. Kept out of `binds` deliberately -- a buffer cannot be both a
   * binding and the indirect source of the same dispatch. Still counted as live.
   */
  indirectFrom?: string;
}

const n = (d: Dims) => d.w * d.h;

// ── The prefix scan, and why it is two levels rather than a recursion ─────
//
// One workgroup is 256 threads (WebGPU's cap) and each thread scans SCAN_PER
// contiguous elements, so one workgroup covers SCAN_BLOCK = 1024 elements.
// 307,200 pixels is 300 blocks, and 300 block sums fit inside a SINGLE
// workgroup. So the whole scan is exactly three passes -- block, spine, add --
// with no level tree to walk and no recursion depth to discover.
//
// The bound is `blocks <= 256 * perThread`, and `perThread` is chosen at encode
// time, so this holds for any image this pipeline will ever see: even at
// 4096x4096 the spine needs 16 elements per thread. planPool asserts it.
export const SCAN_THREADS = 256;
export const SCAN_PER = 4;
export const SCAN_BLOCK = SCAN_THREADS * SCAN_PER;
export const scanBlocks = (count: number): number => Math.max(1, Math.ceil(count / SCAN_BLOCK));

// ── The period search's candidate range ──────────────────────────────────
//
// A detected line IS a grid line, so the two extreme lines of a family sit an
// INTEGER number of periods apart and the only physically possible periods are
// `spread / n`. Those two constants are what bound n, and they live here rather
// than in pose.ts because `scores` is sized from them -- a buffer sized by a
// number chosen somewhere else is how the first draft of this file ended up
// with 256 slots for 286 candidates.
//
// nMin = 3: fewer spacings is not credibly periodic, and decode needs ORDER = 5
// cells anyway. The upper end is HEADROOM_CAP times the board's own cell count,
// which is the HARD cap -- the frame's actual nMax is the looser of a board
// bound and a camera-height bound, is a function of `spread`, and therefore only
// exists on the device. See encodeGppSearch.
export const GPP_N_MIN = 3;
export const GPP_HEADROOM_CAP = 2;
export const gppCandidateCount = (d: Dims): number =>
  GPP_HEADROOM_CAP * Math.max(d.torusR, d.torusC) - GPP_N_MIN + 1;

export const BUFFERS: Record<string, BufferSpec> = {
  // ── Input ──────────────────────────────────────────────────────────────
  gray: { kind: 'storage', bytes: (d) => n(d) * 4, note: 'f32 0..255, row-major. Live to decode.build' },

  // ── S1 gradient ────────────────────────────────────────────────────────
  gradientUni: { kind: 'uniform', bytes: () => 16 },
  fx: { kind: 'storage', bytes: (d) => n(d) * 4 },
  fy: { kind: 'storage', bytes: (d) => n(d) * 4 },

  // ── S2 grow ────────────────────────────────────────────────────────────
  growUni: { kind: 'uniform', bytes: () => 32 },
  ux: { kind: 'storage', bytes: (d) => n(d) * 4, note: 'level-line direction x = -fy normalized' },
  uy: { kind: 'storage', bytes: (d) => n(d) * 4, note: 'level-line direction y = +fx normalized' },
  label: { kind: 'storage', bytes: (d) => n(d) * 4, note: 'i32; -1 = ineligible' },
  next: { kind: 'storage', bytes: (d) => n(d) * 4, note: 'hook writes, compress reads' },
  changed: { kind: 'storage', bytes: () => 4, zero: true, note: 'atomic; nothing writes it before the first read' },
  growArgs: {
    kind: 'indirect', bytes: () => 16, written: 'grow.init',
    note: '[gx, gy, 1, activeRounds]. A CONVERGED RUN LEAVES [0,0,0] -- inheriting that dispatches zero rounds and emits init\'s singleton labelling, silently, at full speed',
  },

  // ── S3 collect ─────────────────────────────────────────────────────────
  collectUni: { kind: 'uniform', bytes: () => 32 },
  labelSurvives: {
    kind: 'storage', bytes: (d) => n(d) * 4, zero: true,
    note: 'per label. MUST BE CLEARED: only strong pixels ever write it, so a label with no strong pixel silently inherits last frame\'s 1',
  },
  labelCounts: { kind: 'storage', bytes: (d) => n(d) * 4, zero: true, note: 'per label, atomicAdd' },
  // ONE vec2 scan replaces the two parallel u32 scans this stage used to need.
  // .x is the kept flag, .y is that label's pixel count, so the exclusive prefix
  // gives the dense region id and the CSR member offset in a single traversal --
  // and the grand total IS [regionCount, memberCount], which is why `counts`
  // needs no copy from two separate scalars.
  kept: { kind: 'storage', bytes: (d) => n(d) * 8, note: 'vec2<u32> (keptFlag, keptFlag ? pixelCount : 0)' },
  keptScan: { kind: 'storage', bytes: (d) => n(d) * 8, note: 'vec2<u32> exclusive prefix of kept: (regionId, memberOffset)' },
  keptSums: { kind: 'storage', bytes: (d) => scanBlocks(n(d)) * 8 },
  keptOffs: { kind: 'storage', bytes: (d) => scanBlocks(n(d)) * 8 },
  keptScanUni: { kind: 'uniform', bytes: () => 16 },
  keptSpineUni: { kind: 'uniform', bytes: () => 16 },
  counts: { kind: 'storage', bytes: () => 8, note: '[regionCount, memberCount]; written directly by the scan spine' },
  cursor: {
    kind: 'storage', bytes: (d) => d.maxRegions * 4, zero: true,
    note: 'per REGION scatter cursor, atomicAdd. Indexed by region rather than by label: same job, 1.17MiB -> 16KiB',
  },
  members: { kind: 'storage', bytes: (d) => n(d) * 4, note: 'CSR payload; bounded by n exactly -- regions are a disjoint partition' },
  regionOffsets: { kind: 'storage', bytes: (d) => d.maxRegions * 4 },
  regionSizes: { kind: 'storage', bytes: (d) => d.maxRegions * 4 },
  meanDirs: { kind: 'storage', bytes: (d) => d.maxRegions * 8 },
  regionArgs: { kind: 'indirect', bytes: () => 12, written: 'collect.regionMeta', note: '(ceil(rc/64),1,1); serves finalize AND lsdFit' },

  // ── S4 lsdFit ──────────────────────────────────────────────────────────
  lsdFitUni: { kind: 'uniform', bytes: () => 32 },
  rects: { kind: 'storage', bytes: (d) => d.maxRegions * 40, note: '10 f32: cx,cy,theta,length,width,nfaLog10,accepted,pad,n,k' },

  // ── S5 lines + votes ───────────────────────────────────────────────────
  linesUni: { kind: 'uniform', bytes: () => 16 },
  // Same vec2 scan primitive. Only .x carries a flag here; .y rides along
  // unused, which costs maxRegions*4 bytes and buys one scan implementation
  // instead of two.
  // Written for EVERY r in [0, maxRegions) by lines.flag, including the tail past
  // the region count -- which is why it needs no clear despite being a scan input
  // whose producer only has regionCount regions to talk about.
  lineFlag: { kind: 'storage', bytes: (d) => d.maxRegions * 8, note: 'vec2<u32> (accepted && long enough, 0)' },
  lineScan: { kind: 'storage', bytes: (d) => d.maxRegions * 8 },
  lineSums: { kind: 'storage', bytes: (d) => scanBlocks(d.maxRegions) * 8 },
  lineOffs: { kind: 'storage', bytes: (d) => scanBlocks(d.maxRegions) * 8 },
  lineScanUni: { kind: 'uniform', bytes: () => 16 },
  lineSpineUni: { kind: 'uniform', bytes: () => 16 },
  lineCount: { kind: 'storage', bytes: () => 8, note: 'vec2<u32>; .x is the line count, written directly by the scan spine' },
  lines: { kind: 'storage', bytes: (d) => d.maxLines * 16, note: '(x1,y1,x2,y2) pixel space' },
  lineArgs: { kind: 'indirect', bytes: () => 12, written: 'lines.emit' },
  votesUni: { kind: 'uniform', bytes: () => 32 },
  votes: { kind: 'storage', bytes: (d) => d.maxLines * 16, note: '(nx,ny,nz,weight)' },
  maxWeight: { kind: 'storage', bytes: () => 4, zero: true, note: 'atomic max over float bits; zero is correct ONLY because weights are non-negative' },

  // ── S6 fit ─────────────────────────────────────────────────────────────
  fitUni: { kind: 'uniform', bytes: () => 16 },
  // `ataPartials` is GONE, and with it the second pass that summed it. See
  // FIT_ATA_WGSL's header: the partials existed so a HOST could add up the
  // per-workgroup rows, and with the host gone the reduction is one workgroup
  // striding over the votes. That also removes a live instance of §9's defect --
  // fit.ata was declared indirect off lineArgs, so it wrote a PREFIX of the
  // partials while its consumer's extent was fixed at encode time.
  ata: { kind: 'storage', bytes: () => 21 * 4, note: 'upper triangle of the symmetric 6x6, a<=b order. Written in full by one thread, so no clear' },
  triad: { kind: 'storage', bytes: () => 48, note: 'Drow, Dcol, Dnormal as three vec3<f32>, stride 16' },

  // ── S7 grid period + phase ─────────────────────────────────────────────
  gppUni: { kind: 'uniform', bytes: () => 64 },
  // (value, weight, crossMin, crossMax) per line, and the packing is not
  // cosmetic: WebGPU guarantees only EIGHT storage buffers per compute stage, and
  // gpp.polish needs the families, their counts, the extent, the candidates,
  // their distinctness and a result. As parallel f32 arrays that is ten bindings
  // and the pipeline does not build on a baseline device.
  //
  // `cross` is the DECODE HULL's per-line contribution (§12) and it rides here
  // for the same reason: it is the one place the two endpoints' second gnomonic
  // coordinate is still in registers, and adding it as its own maxLines buffer
  // would cost gpp.extent a ninth binding. A row line's `value` is its xCol and
  // its `cross` is the xRow span of its two endpoints; a column line's are the
  // other way round. See GPP_CLASSIFY_WGSL.
  samples: { kind: 'storage', bytes: (d) => d.maxLines * 16, note: 'vec4<f32> (value, weight, crossMin, crossMax) per line' },
  // (isRow, 1 - isRow) -- so ONE vec2 scan yields both families' compaction
  // indices at once, and its grand total IS [rowCount, colCount]. The two
  // separate scans this replaced differed only in which lane they counted.
  // SAME TOTAL-FUNCTION OBLIGATION AS lineFlag, and this entry USED to get it
  // wrong: gpp.classify was declared `indirectFrom: 'lineArgs'`, so it wrote
  // [0, lineCount) while the scan runs over maxLines. Fixed the way lines.flag
  // fixed it -- classify dispatches DIRECTLY with an interior guard -- so this
  // needs no clear. Note the tail is (0,0) and not (0,1): see GPP_CLASSIFY_WGSL.
  family: { kind: 'storage', bytes: (d) => d.maxLines * 8, note: 'vec2<u32> (isRow, 1-isRow); (0,0) = neither' },
  familyScan: { kind: 'storage', bytes: (d) => d.maxLines * 8, note: 'vec2<u32> (rowIdx, colIdx)' },
  familySums: { kind: 'storage', bytes: (d) => scanBlocks(d.maxLines) * 8 },
  familyOffs: { kind: 'storage', bytes: (d) => scanBlocks(d.maxLines) * 8 },
  familyScanUni: { kind: 'uniform', bytes: () => 16 },
  familySpineUni: { kind: 'uniform', bytes: () => 16 },
  rowSamples: { kind: 'storage', bytes: (d) => d.maxLines * 16, note: 'vec4<f32> (value, weight, crossMin, crossMax), compacted' },
  colSamples: { kind: 'storage', bytes: (d) => d.maxLines * 16 },
  familyCounts: { kind: 'storage', bytes: () => 8, note: '[rowCount, colCount]; written directly by the scan spine' },
  // [rowMin, rowMax, colMin, colMax, spread, hullRowMin, hullRowMax, hullColMin,
  // hullColMax, 0, 0, 0]. PER-FAMILY: pooling the first four measures the extent
  // of two unrelated coordinate sets -- a real bug this code already fixed once.
  // `gpp.extentInit` is GONE with the atomics it was initializing; one workgroup
  // reduces and lane 0 writes all 12 slots, so this is total by construction.
  // rowMin..colMax are only meaningful when the matching familyCounts lane is
  // >= 1; `spread` and the hull are safe unconditionally.
  //
  // Lanes 5-8 are THE DECODE LATTICE'S BOUNDS (§12, measured and adopted
  // 2026-08-14), named by AXIS rather than by family: hullRow* bounds xRow over
  // every classified endpoint, hullCol* bounds xCol. decode.layout scales both by
  // `distance` to get (u, v). This is what deleted decode.bounds, its init pass,
  // `uvBounds` and 2,401 ray casts.
  extent: { kind: 'storage', bytes: () => 48 },
  // vec4 per candidate: (score, phiRow, phiCol, period). The two phases ride
  // along because the sweep already formed the complex sum they come out of --
  // src/pose recomputes a circularFit per candidate to get them, and its own
  // comment calls that out as doubling the trig work of any sweep.
  //
  // A first draft sized this 256 entries, which is TOO SMALL: the hard candidate
  // bound is HEADROOM_CAP * max(torusR, torusC) = 288 at a 144 board, so 286
  // candidates. Derived here rather than chosen, which is why the mismatch could
  // not survive being written down.
  scores: { kind: 'storage', bytes: (d) => gppCandidateCount(d) * 16 },
  // vec4 x 8. Slot 0 is (count, 0, 0, 0); slots 1..GPP_TOP_K are
  // (period, score, phiRow, phiCol), descending by score.
  topK: { kind: 'storage', bytes: () => 128 },
  distinctness: { kind: 'storage', bytes: () => 32, note: 'one f32 per topK entry; -1 = could not be sampled' },
  gppResult: { kind: 'storage', bytes: () => 16, note: 'period, phiRow, phiCol, height' },

  // ── S8 decode layout ───────────────────────────────────────────────────
  layoutUni: { kind: 'uniform', bytes: () => 48 },
  // `uvBounds` IS GONE, along with decode.bounds and decode.boundsInit. The
  // lattice is bounded by the projected VOTE LINES instead of by a 49x49 grid of
  // rays through the view quadrilateral (§12, measured 2026-08-14), and that hull
  // is four lanes of gpp's `extent` -- so there is no min/max target here to
  // initialize to +/-inf, and no ray cast to gate on minGrazingCos. The PER-CELL
  // grazing test in decode.build is untouched and is more load-bearing than
  // before, not less.
  binThreshPartials: { kind: 'storage', bytes: (d) => Math.ceil(n(d) / 256) * 4 },
  binThreshold: { kind: 'storage', bytes: () => 4 },
  layout: { kind: 'storage', bytes: () => 128, note: 'the lattice description; written BY the device now' },
  buildArgs: { kind: 'indirect', bytes: () => 12, zero: true },
  tallyArgs: { kind: 'indirect', bytes: () => 4 * 12, zero: true, note: 'one triple per orientation' },
  // Written by decode.argmax, not by decode.layout -- its extent depends on the
  // WINNING orientation, which is the second host dependency decode used to have.
  // A failed layout still propagates: buildArgs and tallyArgs go to zero, so the
  // histogram stays cleared, argmax finds no winner and zeroes this itself.
  correctnessArgs: { kind: 'indirect', bytes: () => 12, zero: true },

  // ── S9-S12 decode ──────────────────────────────────────────────────────
  packed: { kind: 'storage', bytes: (d) => d.maxCells * 4, note: 'bit0 = valid, bit1 = sampled bit' },
  tallyUni: { kind: 'uniform', bytes: () => 16, note: 'order, torusR, torusC, tableSize -- all host constants; the orientation is an ENTRY POINT and rows/cols come off `layout`' },
  hist: { kind: 'storage', bytes: (d) => 4 * d.torusR * d.torusC * 4, zero: true, note: 'atomicAdd' },
  totalWindows: { kind: 'storage', bytes: () => 4, zero: true, note: 'atomicAdd' },
  argmaxUni: { kind: 'uniform', bytes: () => 16 },
  result: { kind: 'storage', bytes: () => 32, zero: true, note: '8 u32; slots 6/7 are atomicAdd targets' },
  correctUni: { kind: 'uniform', bytes: () => 16 },
  torus: { kind: 'persistent', bytes: (d) => d.torusR * d.torusC * 4 },
  hashKeys: { kind: 'persistent', bytes: (d) => d.hashSlots * 4 },
  hashValues: { kind: 'persistent', bytes: (d) => d.hashSlots * 4 },

  // ── S13 finish ─────────────────────────────────────────────────────────
  finishUni: { kind: 'uniform', bytes: () => 16, note: 'torusR, torusC, maxRegions, maxLines -- the caps the overflow bits are against' },
  status: { kind: 'storage', bytes: () => 4, zero: true, note: 'atomicOr bitfield; see the status table' },
  pose: { kind: 'storage', bytes: () => 128, note: 'THE ONE READBACK' },
};

export const STAGES: readonly Stage[] = [
  { id: 'upload', binds: ['gray'] },

  { id: 'gradient', binds: ['gradientUni', 'gray', 'fx', 'fy'] },

  // One shared explicit bind-group layout across all four entry points -- see
  // this file's header, and growRegions.gpu.ts's getPipelines for what
  // `layout: 'auto'` cost when the four entry points' binding usage differs.
  { id: 'grow.init', binds: ['growUni', 'fx', 'fy', 'ux', 'uy', 'label', 'next', 'changed'] },
  { id: 'grow.hook', binds: ['growUni', 'fx', 'fy', 'ux', 'uy', 'label', 'next', 'changed'], indirectFrom: 'growArgs' },
  { id: 'grow.compress', binds: ['growUni', 'fx', 'fy', 'ux', 'uy', 'label', 'next', 'changed'], indirectFrom: 'growArgs' },
  // `growArgs` IS bound here (group 1) because gate writes it -- and that is
  // exactly why hook and compress, which dispatch off it, must not bind it.
  { id: 'grow.gate', binds: ['growUni', 'fx', 'fy', 'ux', 'uy', 'label', 'next', 'changed', 'growArgs'] },

  // `survive` and `histogram` were two passes over the same pixels reading the
  // same label, split only so the histogram could skip doomed labels. That saved
  // nothing -- markKept ignores the count of a label that failed hysteresis --
  // and cost a whole pass plus a false dependency between them.
  { id: 'collect.tally', binds: ['collectUni', 'label', 'fx', 'fy', 'labelSurvives', 'labelCounts'] },
  { id: 'collect.markKept', binds: ['collectUni', 'labelSurvives', 'labelCounts', 'kept'] },
  { id: 'kept.scan', binds: ['keptScanUni', 'kept', 'keptScan', 'keptSums'] },
  { id: 'kept.spine', binds: ['keptSpineUni', 'keptSums', 'keptOffs', 'counts'] },
  { id: 'kept.add', binds: ['keptScanUni', 'keptOffs', 'keptScan'] },
  {
    id: 'collect.regionMeta',
    binds: ['collectUni', 'kept', 'keptScan', 'counts', 'regionOffsets', 'regionSizes', 'regionArgs'],
  },
  { id: 'collect.scatter', binds: ['collectUni', 'label', 'kept', 'keptScan', 'cursor', 'members'] },
  { id: 'collect.finalize', binds: ['collectUni', 'counts', 'fx', 'fy', 'regionOffsets', 'regionSizes', 'members', 'meanDirs'], indirectFrom: 'regionArgs' },

  {
    id: 'lsdFit',
    binds: ['lsdFitUni', 'fx', 'fy', 'regionOffsets', 'regionSizes', 'members', 'meanDirs', 'counts', 'rects'],
    indirectFrom: 'regionArgs',
  },

  // DIRECT, over maxRegions, and not indirect off regionArgs. `lineFlag` is a
  // scan input, so it must be TOTAL over the scanned range or its tail keeps the
  // previous frame's flags -- which an indirect dispatch over the region count
  // structurally cannot write. See LINES_FLAG_WGSL's header.
  { id: 'lines.flag', binds: ['linesUni', 'rects', 'counts', 'lineFlag'] },
  { id: 'line.scan', binds: ['lineScanUni', 'lineFlag', 'lineScan', 'lineSums'] },
  { id: 'line.spine', binds: ['lineSpineUni', 'lineSums', 'lineOffs', 'lineCount'] },
  { id: 'line.add', binds: ['lineScanUni', 'lineOffs', 'lineScan'] },
  { id: 'lines.emit', binds: ['linesUni', 'rects', 'lineFlag', 'lineScan', 'lines', 'lineCount', 'lineArgs', 'status'] },
  { id: 'votes.cast', binds: ['votesUni', 'lines', 'lineCount', 'votes', 'maxWeight'], indirectFrom: 'lineArgs' },

  // ONE workgroup, dispatched DIRECTLY, reading the vote count off the device --
  // not one workgroup per 64 votes indirect off lineArgs, which is what this
  // entry used to say and what src/pose does. Two reasons, in FIT_ATA_WGSL's
  // header: the partial rows only ever existed for a host to sum, and the
  // indirect form is §9's iteration-domain defect again (a producer writing a
  // prefix, a consumer whose extent is fixed at encode time).
  { id: 'fit.ata', binds: ['fitUni', 'votes', 'lineCount', 'maxWeight', 'ata'] },
  { id: 'fit.eigen', binds: ['ata', 'triad', 'status'] },

  // DIRECT over maxLines, not indirect off lineArgs, for exactly the reason
  // lines.flag is: `family` is a scan input and has to be TOTAL over the scanned
  // range. See GPP_CLASSIFY_WGSL.
  { id: 'gpp.classify', binds: ['gppUni', 'lines', 'lineCount', 'votes', 'triad', 'samples', 'family'] },
  { id: 'family.scan', binds: ['familyScanUni', 'family', 'familyScan', 'familySums'] },
  { id: 'family.spine', binds: ['familySpineUni', 'familySums', 'familyOffs', 'familyCounts'] },
  { id: 'family.add', binds: ['familyScanUni', 'familyOffs', 'familyScan'] },
  {
    id: 'gpp.compact',
    binds: ['gppUni', 'samples', 'family', 'familyScan', 'rowSamples', 'colSamples'],
  },
  { id: 'gpp.extent', binds: ['gppUni', 'rowSamples', 'colSamples', 'familyCounts', 'extent', 'status'] },
  { id: 'gpp.sweep', binds: ['gppUni', 'rowSamples', 'colSamples', 'familyCounts', 'extent', 'scores'] },
  { id: 'gpp.peaks', binds: ['gppUni', 'scores', 'topK', 'status'] },
  // `gppResult` was in this list and is not an input: distinct reads a candidate
  // period off topK and writes only `distinctness`. gpp.polish is the sole
  // writer of gppResult and it runs after.
  { id: 'gpp.distinct', binds: ['gppUni', 'gray', 'topK', 'extent', 'triad', 'distinctness'] },
  {
    id: 'gpp.polish',
    // EXACTLY EIGHT storage buffers, which is the WebGPU baseline per-stage
    // limit and the reason `samples` is a vec2 array rather than two f32 ones.
    binds: ['gppUni', 'rowSamples', 'colSamples', 'familyCounts', 'extent', 'topK', 'distinctness', 'gppResult'],
  },

  { id: 'decode.binThreshPartials', binds: ['layoutUni', 'gray', 'binThreshPartials'] },
  { id: 'decode.binThreshReduce', binds: ['layoutUni', 'binThreshPartials', 'binThreshold'] },
  {
    id: 'decode.layout',
    // EXACTLY EIGHT storage buffers again, and this is the entry the §12
    // measurement bought the headroom for: as declared it bound nine, which does
    // not build on a baseline device. Deleting `uvBounds` and handing
    // `correctnessArgs` to decode.argmax -- its real owner -- is what fits it,
    // rather than pairing fields the way gpp.polish had to.
    binds: ['layoutUni', 'triad', 'gppResult', 'extent', 'binThreshold', 'layout', 'buildArgs', 'tallyArgs', 'status'],
  },

  { id: 'decode.build', binds: ['layout', 'gray', 'packed'], indirectFrom: 'buildArgs' },

  // EVERY ONE OF THESE BINDS `layout`, and that is §12's disruption showing up
  // where it lands rather than where it was caused: `rows` and `cols` are
  // device-side quantities now, so a uniform cannot carry them. The uniforms
  // here hold only host constants -- the board size, ORDER, the hash table size.
  //
  // The four tally entries differ ONLY by orientation, which is a compile-time
  // constant in each, so they are four entry points over one module rather than
  // four uniform blocks and a dynamic offset `pass()` would have to learn.
  { id: 'decode.tally.o0', binds: ['tallyUni', 'layout', 'packed', 'hashKeys', 'hashValues', 'hist', 'totalWindows'], indirectFrom: 'tallyArgs' },
  { id: 'decode.tally.o1', binds: ['tallyUni', 'layout', 'packed', 'hashKeys', 'hashValues', 'hist', 'totalWindows'], indirectFrom: 'tallyArgs' },
  { id: 'decode.tally.o2', binds: ['tallyUni', 'layout', 'packed', 'hashKeys', 'hashValues', 'hist', 'totalWindows'], indirectFrom: 'tallyArgs' },
  { id: 'decode.tally.o3', binds: ['tallyUni', 'layout', 'packed', 'hashKeys', 'hashValues', 'hist', 'totalWindows'], indirectFrom: 'tallyArgs' },
  { id: 'decode.argmax', binds: ['argmaxUni', 'layout', 'hist', 'totalWindows', 'result', 'correctnessArgs'] },
  { id: 'decode.correctness', binds: ['correctUni', 'layout', 'packed', 'torus', 'result'], indirectFrom: 'correctnessArgs' },

  // ── Why so few stages bind `status` ──
  //
  // Most bits are DERIVED here rather than set where they occur, because
  // `finish` already binds the buffer that carries the evidence:
  //   growNotConverged <- growArgs.x != 0 (gate zeroes it on convergence)
  //   regionOverflow / noRegions <- counts[0]
  //   noVotes                    <- lineCount
  //   decodeNoAnchor             <- result[0]
  //
  // (`lineOverflow` was on that list and is not derivable there: finish binds no
  // uniform, so it holds no maxLines to compare against. It is set by lines.emit,
  // which is the pass that clamps -- see §15's table, which had it right. The
  // same objection applies to regionOverflow and is still open, since collect
  // does not bind status today.)
  //
  // That is only the REPORTING half. The PROPAGATION half is separate and does
  // not go through status at all: a stage that detects a failure zeroes the
  // indirect args it owns, so everything downstream dispatches no workgroups.
  // collect.regionMeta owns regionArgs, lines.emit owns lineArgs,
  // decode.layout owns buildArgs/tallyArgs/correctnessArgs. Reporting at the
  // end and propagating at the point of failure are two different jobs, and
  // conflating them is what would put a status read in every kernel.
  //
  // The bits that genuinely cannot be derived from a buffer read here --
  // fitDegenerate, gppNoSamples, gppNoCandidates, layoutInvalid, gridOverflow
  // -- are set by the stage that finds them, which is why those five bind it.
  // SEVEN storage buffers, and it was declared with nine. `triad` and `gppResult`
  // both came off: `layout` already carries the triad (decode.build needs it
  // there), and it carries `distance`, which IS the height -- so the period is
  // cellPitch / distance rather than a third buffer. Reading a value from the
  // one place that already had to have it is not a saving here, it is the
  // difference between building on a baseline device and not.
  { id: 'finish', binds: ['finishUni', 'layout', 'result', 'counts', 'lineCount', 'growArgs', 'status', 'pose'] },
];
