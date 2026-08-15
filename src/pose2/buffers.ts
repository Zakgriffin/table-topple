import { BUFFERS, SCAN_THREADS, STAGES, type BufferSpec, type Dims, type Stage, scanBlocks } from './pipeline.ts';

/**
 * How many elements one spine thread may serially scan. Not a hardware limit --
 * it is the point past which "one workgroup is enough" stops being obviously
 * true and deserves a second look rather than a bigger constant.
 */
const MAX_SPINE_PER_THREAD = 16;

// ── Buffer allocation, planned from the pipeline declaration ──────────────
//
// Split in two on purpose, the same way src/pose/gpu/arena.ts split BumpPlanner
// out: everything worth asserting here is a property of the ARITHMETIC over
// pipeline.ts, and none of it needs a GPUDevice -- so it runs under
// `node --test` rather than only in a browser.
//
//   planPool()      pure. Liveness, colouring, the clear schedule, validation.
//   createBuffers() the GPU half. createBuffer per slot, nothing else.
//
// ── WHY POOLING IS SAFE WITHOUT THE STAGES KNOWING ──
//
// The pipeline is a straight line, so each buffer's live range is a genuine
// interval and the conflict graph is an INTERVAL GRAPH. Greedy left-to-right
// colouring is optimal on those, and the colour count equals the peak
// concurrent liveness.
//
// The WebGPU usage-scope hazard -- one GPUBuffer bound twice in a dispatch,
// which is a validation error reported ASYNCHRONOUSLY as a silently no-op
// encoder rather than an exception -- is subsumed rather than being a second
// rule to remember:
//
//   If a pass binds two buffers, both are live during that pass by definition.
//   A correct colouring never gives two simultaneously-live buffers the same
//   slot. Therefore no pass can bind one buffer twice.
//
// validate() asserts that anyway, because "therefore" is doing real work in
// that sentence and the failure mode is silent.

export interface Interval {
  /** Index into STAGES of the first stage that binds this buffer. */
  first: number;
  /** Index into STAGES of the last stage that binds it. */
  last: number;
}

export interface Slot {
  index: number;
  bytes: number;
  /** Every buffer sharing this allocation, in birth order. */
  occupants: string[];
}

export interface PoolPlan {
  dims: Dims;
  /** The stage list this plan was built from. Carried so validate() and the
   * clear schedule refer to the same order the liveness came from. */
  stages: readonly Stage[];
  /** Pooled storage slots. */
  slots: Slot[];
  /** Storage buffer name -> slot index. */
  assignment: Map<string, number>;
  /** uniform / indirect / persistent buffers, one allocation each. */
  dedicated: string[];
  liveness: Map<string, Interval>;
  /**
   * The buffers this plan promises are readable after the last stage. See
   * PlanOptions.inspect; carried so run.ts can size one staging buffer and
   * reject a per-frame request for anything not declared here.
   */
  inspect: readonly string[];
  /**
   * The derived clear schedule: for each stage index, the `zero` buffers to
   * clear immediately before it. Derived rather than declared, because with
   * pooling a frame-start clear is wrong -- the memory is still someone else's
   * until that buffer is born.
   */
  clears: Map<number, string[]>;
  /** Total device bytes: pooled slots + dedicated + persistent. */
  totalBytes: number;
  /** Pooled bytes only, for reporting the saving against `alias: false`. */
  pooledBytes: number;
}

export interface PlanOptions {
  /**
   * Share allocations between buffers whose live ranges do not overlap.
   *
   * `false` is NOT a second code path -- it is the same colouring with a
   * degenerate liveness table where every buffer is live for the whole frame,
   * so every one gets its own slot. Ship v1 with it off; running both ways and
   * comparing poses is a free self-check for aliasing and clear-schedule bugs.
   */
  alias?: boolean;
  /**
   * Buffers the caller intends to read back once the frame is done -- the
   * display path of §22, which wants fx/fy or the region CSR and which the pose
   * path itself has no use for.
   *
   * DECLARED here rather than discovered per frame, for two reasons that are
   * both about not deciding anything on the hot path:
   *
   *   - under `alias` it is an INPUT TO THE COLOURING (see planPool), so the
   *     slot stays exclusive by construction rather than by an exception the
   *     colouring has to remember;
   *   - it sizes the single staging buffer, so no frame allocates.
   *
   * STORAGE buffers only. They are the only kind `createBuffers` gives
   * COPY_SRC, and a uniform/indirect/persistent name here would fail at copy
   * time -- which WebGPU reports asynchronously, i.e. as the silent no-op this
   * file exists to convert into a throw.
   */
  inspect?: readonly string[];
  /** Defaults to STAGES. Overridable so a test can plan a deliberately broken
   * pipeline and assert that validate() catches it. */
  stages?: readonly Stage[];
}

function specOf(name: string): BufferSpec {
  const s = BUFFERS[name];
  if (!s) throw new Error(`pipeline declares a bind on '${name}', which is not in BUFFERS`);
  return s;
}

/**
 * First-bind to last-bind per buffer, over the declared stage order.
 *
 * `indirectFrom` counts: a pass that dispatches off a buffer is using it, even
 * though it deliberately does not bind it.
 */
export function computeLiveness(stages: readonly Stage[] = STAGES): Map<string, Interval> {
  const live = new Map<string, Interval>();
  stages.forEach((stage, i) => {
    const touched = stage.indirectFrom ? [...stage.binds, stage.indirectFrom] : stage.binds;
    for (const name of touched) {
      specOf(name);
      const prev = live.get(name);
      if (prev) prev.last = i;
      else live.set(name, { first: i, last: i });
    }
  });
  return live;
}

/**
 * Greedy interval colouring, restricted to buffers of IDENTICAL byte size.
 *
 * Same-size-only is a simplification with no cost here -- every per-pixel array
 * is exactly n*4 -- and it removes the question of what a slot shared by a big
 * and a small buffer should be sized at. A mixed-size pool would have to take
 * the max and waste the difference on every frame.
 */
function colour(
  names: string[], liveness: Map<string, Interval>, bytesOf: (n: string) => number,
): { slots: Slot[]; assignment: Map<string, number> } {
  const slots: Slot[] = [];
  const assignment = new Map<string, number>();
  // Birth order. Ties broken by name so the assignment is deterministic across
  // runs -- a plan that shuffles between runs would make the aliased-vs-
  // unaliased comparison test useless.
  const ordered = [...names].sort((a, b) => {
    const d = liveness.get(a)!.first - liveness.get(b)!.first;
    return d !== 0 ? d : a.localeCompare(b);
  });

  for (const name of ordered) {
    const iv = liveness.get(name)!;
    const bytes = bytesOf(name);
    // The lowest-numbered same-size slot whose last occupant is already dead.
    // STRICTLY dead: a buffer dying at stage k and one born at stage k are both
    // bound during k, so they conflict.
    const fit = slots.find((s) => s.bytes === bytes && liveness.get(s.occupants[s.occupants.length - 1]!)!.last < iv.first);
    if (fit) {
      fit.occupants.push(name);
      assignment.set(name, fit.index);
    } else {
      const slot: Slot = { index: slots.length, bytes, occupants: [name] };
      slots.push(slot);
      assignment.set(name, slot.index);
    }
  }
  return { slots, assignment };
}

export function planPool(dims: Dims, opts: PlanOptions = {}): PoolPlan {
  const alias = opts.alias ?? false;
  const stages = opts.stages ?? STAGES;
  const inspect = opts.inspect ?? [];
  const liveness = computeLiveness(stages);

  // Checked before anything is derived from it, so the message names the
  // caller's mistake rather than surfacing three steps later as a missing slot.
  for (const name of inspect) {
    const spec = BUFFERS[name];
    if (!spec) throw new Error(`inspect names '${name}', which is not in BUFFERS`);
    if (spec.kind !== 'storage') {
      throw new Error(
        `inspect names '${name}', which is a ${spec.kind} buffer. Only storage buffers are created ` +
        `with COPY_SRC, and copying from one that is not fails asynchronously -- as a silent no-op, not an error.`);
    }
  }

  // The scan is three passes because its block sums fit in ONE workgroup. If
  // that ever stopped holding, the spine would scan a prefix of them and every
  // later block would carry a wrong offset -- regions would come out with
  // overlapping CSR slices, which reads downstream as a detector problem rather
  // than as an arithmetic one. Checked here, once, against the largest count any
  // scan in the pipeline runs over.
  const spineMax = SCAN_THREADS * MAX_SPINE_PER_THREAD;
  for (const [what, count] of [['pixels', dims.w * dims.h], ['regions', dims.maxRegions], ['lines', dims.maxLines]] as const) {
    const blocks = scanBlocks(count);
    if (blocks > spineMax) {
      throw new Error(
        `scan over ${count} ${what} needs ${blocks} blocks, and one spine workgroup covers at most ${spineMax}`);
    }
  }

  // MAX_CELLS IS DERIVED, NOT CHOSEN (§12). decode.layout clamps each lattice
  // edge to one board period, so rows * cols cannot exceed torusR * torusC --
  // and that clamp is the whole argument for the cap being a bound rather than a
  // hope. A caller passing less turns it back into a guess: the layout pass's
  // own overrun guard would start firing on ordinary oblique frames and report
  // them as layoutInvalid, which reads as a detection failure three stages from
  // the actual cause.
  const needCells = dims.torusR * dims.torusC;
  if (dims.maxCells < needCells) {
    throw new Error(
      `maxCells is ${dims.maxCells}, but the decode lattice is clamped to one board period ` +
      `(${dims.torusR}x${dims.torusC} = ${needCells} cells) and must be able to hold it`);
  }

  // A declared buffer nobody binds is dead weight that tsc cannot see -- the
  // same class of gap as an exported dispatcher with no importer.
  for (const name of Object.keys(BUFFERS)) {
    if (!liveness.has(name)) throw new Error(`BUFFERS declares '${name}', but no stage binds it`);
  }

  // ── INSPECTION IS A LIVE RANGE, NOT AN EXCEPTION ──
  //
  // A buffer somebody reads after the last stage is live to the END OF THE
  // FRAME. Stating it that way and feeding it to the same colouring is what
  // makes the slot exclusive; the alternative -- a set of names the colouring
  // has to skip -- is a second rule, and a rule that only fires under `alias`
  // is a rule that gets tested last and rots first.
  //
  // Applied to `effective` and NOT to `liveness`: the derived table is a fact
  // about the pipeline declaration, and both validate() and the clear schedule
  // read it. A caller's display request is not a fact about the pipeline.
  const lastStage = stages.length - 1;
  const effective = alias
    ? new Map([...liveness].map(([k, v]) =>
      [k, inspect.includes(k) ? { first: v.first, last: lastStage } : v] as const))
    // The degenerate table: everything live for the whole frame, so the
    // colouring below hands out one slot each -- and inspection is already
    // satisfied, which is why Sphere Lab can inspect freely with `alias` off.
    : new Map([...liveness].map(([k]) => [k, { first: 0, last: lastStage }] as const));

  const bytesOf = (name: string) => specOf(name).bytes(dims);
  const storage = Object.keys(BUFFERS).filter((n) => specOf(n).kind === 'storage');
  const dedicated = Object.keys(BUFFERS).filter((n) => specOf(n).kind !== 'storage');

  const { slots, assignment } = colour(storage, effective, bytesOf);

  const clears = new Map<number, string[]>();
  for (const name of Object.keys(BUFFERS)) {
    if (!specOf(name).zero) continue;
    const at = liveness.get(name)!.first;
    const list = clears.get(at);
    if (list) list.push(name);
    else clears.set(at, [name]);
  }

  const pooledBytes = slots.reduce((a, s) => a + s.bytes, 0);
  const totalBytes = pooledBytes + dedicated.reduce((a, n) => a + bytesOf(n), 0);

  const plan: PoolPlan = { dims, stages, slots, assignment, dedicated, liveness, inspect, clears, totalBytes, pooledBytes };
  validate(plan);
  return plan;
}

/**
 * The two rules WebGPU only enforces asynchronously, plus the declaration
 * hygiene that makes them meaningful. Throws with the offending stage named,
 * because the alternative is an encoder that silently does nothing.
 */
export function validate(plan: PoolPlan): void {
  for (const stage of plan.stages) {
    const seenNames = new Set<string>();
    const slotOwner = new Map<number, string>();
    for (const name of stage.binds) {
      // A name twice in one bind list is a declaration bug, and it would also
      // trivially defeat the slot check below.
      if (seenNames.has(name)) throw new Error(`stage '${stage.id}' binds '${name}' twice`);
      seenNames.add(name);

      const slot = plan.assignment.get(name);
      if (slot === undefined) continue; // dedicated -- one buffer each, cannot alias
      const other = slotOwner.get(slot);
      if (other !== undefined) {
        throw new Error(
          `usage-scope conflict in stage '${stage.id}': '${other}' and '${name}' share pool slot ${slot}. ` +
          `Their live ranges must overlap and do not -- check that pipeline.ts lists what each pass BINDS, not what its shader reads.`,
        );
      }
      slotOwner.set(slot, name);
    }
    // A buffer cannot be both a binding and the indirect source of the same
    // dispatch. This is what grow's group-1 split and decode's separate
    // correctnessArgs both exist to avoid.
    if (stage.indirectFrom !== undefined && stage.binds.includes(stage.indirectFrom)) {
      throw new Error(`stage '${stage.id}' dispatches indirectly off '${stage.indirectFrom}' while also binding it`);
    }
  }

  // A `written` buffer names the stage that establishes its non-zero initial
  // state. If that stage does not exist the note is a lie, and this is the only
  // family a grep for `zero` misses.
  for (const [name, spec] of Object.entries(BUFFERS)) {
    if (spec.written === undefined) continue;
    if (!plan.stages.some((s) => s.id === spec.written)) {
      throw new Error(`BUFFERS['${name}'].written names stage '${spec.written}', which does not exist`);
    }
  }
}

/**
 * The check that actually catches drift, called by each encode function with
 * the names it is ABOUT TO BIND.
 *
 * ── Why validate() above is not enough, and saying so plainly ──
 *
 * Liveness is DERIVED from `binds`, so validate() can never find a slot
 * conflict in a plan built from the same stage list -- two buffers listed in
 * one stage are both live in it by construction. That half of validate() is a
 * tautology, and a tautology is not a check.
 *
 * The real hazard is DRIFT: an encode function that binds a buffer pipeline.ts
 * does not list for it. Then the plan frees that buffer early, some later
 * buffer inherits its slot, and the pass binds one GPUBuffer twice -- a
 * validation error WebGPU reports asynchronously, so the symptom is an encoder
 * that silently does nothing, not an exception.
 *
 * This function closes that gap by comparing the two, so the declaration and
 * the code cannot disagree without something throwing at encode time.
 */
export function assertBinds(plan: PoolPlan, stageId: string, names: readonly string[]): void {
  const stage = plan.stages.find((s) => s.id === stageId);
  if (!stage) throw new Error(`assertBinds: no stage '${stageId}' in the pipeline declaration`);

  const declared = new Set(stage.binds);
  const actual = new Set(names);
  const missing = [...declared].filter((n) => !actual.has(n));
  const extra = [...actual].filter((n) => !declared.has(n));
  if (missing.length || extra.length) {
    throw new Error(
      `stage '${stageId}' binds a different set than pipeline.ts declares` +
      (extra.length ? `; binds but does not declare: ${extra.join(', ')}` : '') +
      (missing.length ? `; declares but does not bind: ${missing.join(', ')}` : '') +
      '. Liveness comes from the declaration, so the two must agree exactly.',
    );
  }

  // NO SLOT CHECK HERE, deliberately. An earlier version re-checked that these
  // names do not collide in the pool, which looked like defence in depth and was
  // dead code: validate() has already proved no stage's DECLARED binds collide,
  // and the comparison above has just proved these names ARE the declared binds.
  // The two compose into the guarantee; a third check cannot fire.
  //
  // It was worse than redundant. The set comparison throws first, so the only
  // input that could have reached the slot check was one this function rejects
  // earlier -- and the test covering it was passing on the set-comparison error
  // while claiming to cover the collision.
}

/**
 * Byte size of one declared buffer under these dims. Exported so run.ts can lay
 * out the staging buffer without a second copy of the size arithmetic -- the
 * same reason liveness and the clear schedule are derived rather than written.
 */
export function bufferBytes(dims: Dims, name: string): number {
  return specOf(name).bytes(dims);
}

/** Human-readable slot map, for the buffer labels and for eyeballing a plan. */
export function describePlan(plan: PoolPlan): string {
  const mib = (b: number) => (b / 1048576).toFixed(3);
  const lines = [
    `pool: ${plan.slots.length} slots, ${mib(plan.pooledBytes)} MiB`,
    `dedicated: ${plan.dedicated.length} buffers`,
    `total: ${mib(plan.totalBytes)} MiB`,
    '',
  ];
  for (const s of plan.slots) {
    lines.push(`  slot ${String(s.index).padStart(2)} ${mib(s.bytes).padStart(8)} MiB  ${s.occupants.join(' | ')}`);
  }
  return lines.join('\n');
}

export type Buffers = Record<string, GPUBuffer>;

/**
 * The GPU half. Every name resolves to a GPUBuffer; pooled names resolve to a
 * SHARED one, which is invisible at every use site by design.
 *
 * Buffers are labelled with their full alias list, because a dump where `next`
 * and `labelCounts` are the same memory is otherwise baffling.
 */
export function createBuffers(device: GPUDevice, plan: PoolPlan): Buffers {
  const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
  const out: Buffers = {};

  const slotBuffers = plan.slots.map((s) =>
    device.createBuffer({ size: s.bytes, usage: STORAGE, label: `pool[${s.index}]: ${s.occupants.join(' | ')}` }));
  for (const [name, slot] of plan.assignment) out[name] = slotBuffers[slot]!;

  for (const name of plan.dedicated) {
    const spec = BUFFERS[name]!;
    const usage = spec.kind === 'uniform'
      ? GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
      : spec.kind === 'indirect'
        ? STORAGE | GPUBufferUsage.INDIRECT
        : STORAGE;
    out[name] = device.createBuffer({ size: spec.bytes(plan.dims), usage, label: name });
  }
  return out;
}
