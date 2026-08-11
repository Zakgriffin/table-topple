// ── The arena: one allocation, handed out in slices ───────────────────────
//
// Step 1 of the flat-architecture restructure. This
// replaces three competing ownership patterns with one:
//
//   - FieldResidency's `owned[]`, destroyed in its own destroy();
//   - per-stage scratch, destroyed by hand on every path including the error
//     unwinds (miss one and it leaks -- collectRegionsGPU's `counts` and
//     `dispatchArgs` leaked ~7KB per reconstruction that way, invisible to
//     every check except the allocation probe's created-vs-destroyed pair);
//   - the deferral handles (PendingIntermediates, buildAndTallyDecodeGPU's
//     `release`), which exist to keep buffers alive past the call that made
//     them.
//
// All three go away if the CALLER owns the memory and stages only borrow. That
// is the whole idea: a stage takes slices in, writes slices out, and allocates
// nothing it has to remember to free.
//
// ── Why a bump allocator and not a pre-planned static partition ──
//
// The end state described in the plan is "static allocation of partitioned
// contiguous memory upfront, sized by image size and by each step's provable
// bound". A bump allocator over one contiguous buffer IS that, with the plan
// expressed as the ORDER of alloc() calls rather than as a separate table that
// can silently disagree with the code. Every property that motivated the static
// partition still holds -- one createBuffer, contiguous, no per-frame churn --
// and there is no second artifact to keep in sync.
//
// ── Overflow SPILLS; it never moves a live slice ──
//
// The capacity has to be known when the backing buffer is created, and on the
// first frame nobody knows it. Growing mid-frame is not an option: it would
// invalidate every slice already handed out, which is the aliasing failure this
// design exists to prevent. So an allocation that does not fit gets its OWN
// standalone buffer (correct, just not arena-backed) and the high-water mark is
// recorded; the next reset() resizes the arena to fit. The pipeline therefore
// self-tunes over one or two frames and degrades, at worst, to exactly the
// per-buffer allocation it replaces.
//
// ── The generation stamp is the safety property ──
//
// Bump allocation cannot produce overlapping LIVE slices by construction. The
// real hazard is different and it is the one FieldResidency's single-assignment
// invariant used to cover: using a slice AFTER the arena has been reset, when
// those same bytes now belong to someone else. That is not hypothetical here --
// the desktop parks intermediates and drains them a frame later (see
// runVisualTail), which is precisely a read across a reset.
//
// Every slice carries the generation it was cut in, and every use goes through
// bind()/sliceRange(), which compare it against the arena's current generation
// and throw on a mismatch. One check catches both the stale-read hazard and the
// aliasing it would cause, and it is a number compare, so it stays on in
// production rather than being a debug mode nobody enables.

// 256 satisfies both minStorageBufferOffsetAlignment and
// minUniformBufferOffsetAlignment at their default limits, so one alignment
// serves storage bindings, uniform bindings and indirect dispatch args without
// the allocator needing to know which a slice will be used as.
export const ARENA_ALIGN = 256;

export function alignUp(n: number, align: number = ARENA_ALIGN): number {
  return Math.ceil(n / align) * align;
}

// ── The pure half ─────────────────────────────────────────────────────────
//
// Offset arithmetic with no GPUBuffer anywhere, so the partitioning is testable
// under `node --test` where WebGPU does not exist. Every property worth
// asserting -- alignment, non-overlap, the high-water mark, spill behaviour --
// is a property of this class alone.
export interface Placement {
  offset: number;
  bytes: number;
  // True when this did not fit the capacity and needs a standalone buffer. The
  // offset is 0 and means nothing in that case.
  spilled: boolean;
}

export class BumpPlanner {
  private cursor = 0;
  // The largest total this planner has been asked for since it was constructed,
  // INCLUDING spills -- i.e. the capacity that would have made this frame fit.
  // Not reset by reset(), which is the point: it is what the next resize reads.
  private high = 0;

  capacity: number;
  readonly align: number;

  // Fields declared and assigned longhand rather than as PARAMETER PROPERTIES.
  // Node's strip-only TypeScript mode rejects `constructor(public capacity)`
  // outright (ERR_UNSUPPORTED_TYPESCRIPT_SYNTAX), so the shorthand would make
  // this module unimportable by `npm test` while compiling perfectly under tsc
  // -- the same shape of trap as verbatimModuleSyntax and module-scope browser
  // access (see tests/README.md).
  constructor(capacity: number, align: number = ARENA_ALIGN) {
    this.capacity = capacity;
    this.align = align;
  }

  get used(): number { return this.cursor; }
  get highWater(): number { return this.high; }

  take(bytes: number): Placement {
    // A zero-byte binding is not legal in WebGPU and an empty capture is an
    // ordinary input (a dark frame, everything below rhoHigh), so the floor is
    // one aligned block rather than a branch at every call site. Same reasoning
    // as the pad() FieldResidency applied to its CSR uploads.
    const size = alignUp(Math.max(bytes, 1), this.align);
    const end = this.cursor + size;
    // Tracked whether or not it fits: the high-water mark has to describe what
    // was ASKED for, not what was served, or a spilling frame would keep
    // reporting a capacity that is already too small and never grow out of it.
    if (end > this.high) this.high = end;
    if (end > this.capacity) return { offset: 0, bytes: size, spilled: true };
    const offset = this.cursor;
    this.cursor = end;
    return { offset, bytes: size, spilled: false };
  }

  reset(): void { this.cursor = 0; }
}

// ── The GPU half ──────────────────────────────────────────────────────────

export interface Slice {
  readonly buffer: GPUBuffer;
  readonly offset: number;
  readonly size: number;
  // Which arena generation cut this. Compared on every use; see the header.
  // A standalone spill buffer carries the generation too, so a stale spill is
  // caught by the same check as a stale arena slice.
  readonly gen: number;
  readonly label: string;
}

// What a stage receives. It can allocate, and it cannot free -- freeing is the
// arena's, i.e. the caller's, and that asymmetry is the ownership rule stated
// as a type.
export type Alloc = (bytes: number, label?: string) => Slice;

export interface ArenaStats {
  capacity: number;
  used: number;
  highWater: number;
  generation: number;
  // Allocations served from the arena vs. those that needed their own buffer.
  // A steady-state pipeline should show zero spills after the first frame or
  // two; a nonzero count here after warmup means a bound is wrong.
  served: number;
  spills: number;
  spilledBytes: number;
}

export interface Arena {
  readonly alloc: Alloc;
  readonly device: GPUDevice;
  // Frees every slice cut since the last reset and bumps the generation, so any
  // slice still held by anyone becomes an error rather than a silent alias.
  // Resizes the backing buffer if the last frame spilled.
  reset(): void;
  destroy(): void;
  stats(): ArenaStats;
}

// A FUNCTION, not a module-scope constant, and that is load-bearing:
// `GPUBufferUsage` is a browser global, so evaluating this at module scope makes
// arena.ts unimportable in node -- which would take the BumpPlanner tests down
// with it, the exact thing splitting the pure half was for. Same trap as
// floorPattern.ts's `location.search` read (see tests/README.md), one module
// over.
//
// One usage set for every slice, because the arena cannot know what a slice will
// be bound as: STORAGE and UNIFORM for bindings, COPY_SRC/COPY_DST for readbacks
// and clears, INDIRECT for dispatch args. WebGPU forbids only MAP_READ/MAP_WRITE
// from combining freely, and staging buffers are separate (see device.ts), so
// this combination is legal.
function arenaUsage(): number {
  return GPUBufferUsage.STORAGE | GPUBufferUsage.UNIFORM
    | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST | GPUBufferUsage.INDIRECT;
}

export interface ArenaOptions {
  // Where to start. Anything reasonable works -- the arena resizes to the true
  // high-water mark after the first frame -- so this only decides whether frame
  // one spills.
  initialBytes?: number;
  // Fills the whole arena with 0xDEADBEEF on reset. Turns a stale read from
  // plausible-looking previous-frame data into an obvious sentinel, which is the
  // difference between a bug that shows up as a slightly wrong pose and one that
  // shows up at all. Costs a full-arena clear per frame, so it is off by
  // default and belongs in the harness rather than in production.
  poisonOnReset?: boolean;
}

export function createArena(device: GPUDevice, opts: ArenaOptions = {}): Arena {
  const planner = new BumpPlanner(alignUp(opts.initialBytes ?? 1 << 22)); // 4MB
  let backing = device.createBuffer({
    size: planner.capacity, usage: arenaUsage(), label: 'pose.arena',
  });
  let generation = 1;
  let spillBuffers: GPUBuffer[] = [];
  let served = 0, spills = 0, spilledBytes = 0;

  const alloc: Alloc = (bytes, label = 'unlabelled') => {
    const p = planner.take(bytes);
    if (!p.spilled) {
      served++;
      return { buffer: backing, offset: p.offset, size: p.bytes, gen: generation, label };
    }
    // The arena is too small this frame. Serve it standalone so the frame is
    // CORRECT, and let reset() below resize for the next one.
    spills++;
    spilledBytes += p.bytes;
    const own = device.createBuffer({ size: p.bytes, usage: arenaUsage(), label: `pose.spill:${label}` });
    spillBuffers.push(own);
    return { buffer: own, offset: 0, size: p.bytes, gen: generation, label };
  };

  return {
    alloc,
    device,
    reset() {
      for (const b of spillBuffers) b.destroy();
      spillBuffers = [];
      // Resize BEFORE the generation bump so a slice from the old buffer can
      // never compare equal to the new generation.
      if (planner.highWater > planner.capacity) {
        backing.destroy();
        planner.capacity = alignUp(planner.highWater);
        backing = device.createBuffer({
          size: planner.capacity, usage: arenaUsage(), label: 'pose.arena',
        });
      }
      planner.reset();
      generation++;
      served = 0; spills = 0; spilledBytes = 0;
      if (opts.poisonOnReset) {
        const enc = device.createCommandEncoder();
        // clearBuffer writes zeroes, not a sentinel -- WebGPU has no fill value.
        // Zero is still a far better stale-read signal than last frame's data,
        // because every consumer here treats an all-zero field as an empty
        // capture rather than as a plausible one.
        enc.clearBuffer(backing);
        device.queue.submit([enc.finish()]);
      }
    },
    destroy() {
      for (const b of spillBuffers) b.destroy();
      spillBuffers = [];
      backing.destroy();
      generation++;
    },
    stats() {
      return {
        capacity: planner.capacity, used: planner.used, highWater: planner.highWater,
        generation, served, spills, spilledBytes,
      };
    },
  };
}

// ── Using a slice ─────────────────────────────────────────────────────────
//
// Both helpers exist to be the ONLY way a slice reaches WebGPU, so the
// generation check cannot be bypassed by a call site that binds `slice.buffer`
// directly. Reviewing for that is the one thing tsc cannot check here.

export class StaleSliceError extends Error {
  constructor(slice: Slice, current: number) {
    super(
      `pose arena: slice '${slice.label}' is from generation ${slice.gen} but the arena is at ${current}. `
      + 'Its bytes belong to a later allocation now -- something is holding a slice across a reset() '
      + 'across a reset().',
    );
    this.name = 'StaleSliceError';
  }
}

function check(slice: Slice, arena: Arena): void {
  const current = arena.stats().generation;
  if (slice.gen !== current) throw new StaleSliceError(slice, current);
}

// A bind group entry's resource. Every binding in every stage goes through this.
export function bind(slice: Slice, arena: Arena): GPUBufferBinding {
  check(slice, arena);
  return { buffer: slice.buffer, offset: slice.offset, size: slice.size };
}

// For the encoder-level operations that take a buffer plus an offset directly --
// copyBufferToBuffer, clearBuffer, dispatchWorkgroupsIndirect, and the readback
// helpers. Same check, different shape.
export function sliceRange(slice: Slice, arena: Arena): { buffer: GPUBuffer; offset: number; size: number } {
  check(slice, arena);
  return { buffer: slice.buffer, offset: slice.offset, size: slice.size };
}
