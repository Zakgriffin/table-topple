import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import { BUFFERS, SCAN_BLOCK, SCAN_THREADS, STAGES, type Dims, type Stage, scanBlocks } from '../src/pose/pipeline.ts';
import { assertBinds, computeLiveness, describePlan, planPool, validate } from '../src/pose/buffers.ts';

// ── The flat pipeline's buffer plan ───────────────────────────────────────
//
// buffers.ts's planner is the pure half of the allocation, split out for the
// same reason the old pipeline's gpu/arena.ts split out BumpPlanner: every property worth
// asserting is a property of the arithmetic over pipeline.ts, so it runs here
// under `node --test` rather than only in a browser.
//
// What is NOT covered here and needs a device: that GPUBufferUsage flags suit
// the real adapter, that createBuffer honours the sizes, and -- the big one --
// that each encode function actually calls assertBinds with the names it binds.
// The last of those is why assertBinds exists at all; see its header.

const DIMS: Dims = {
  w: 480, h: 640,
  maxRegions: 16384, maxLines: 16384, maxCells: 262144,
  torusR: 144, torusC: 144, hashSlots: 65536,
};

const N4 = DIMS.w * DIMS.h * 4; // one full-image array: 1.172 MiB

/** Peak simultaneous liveness among buffers of one byte size -- the lower bound
 *  a colouring can achieve, and (interval graphs being perfect) also the upper. */
function peakConcurrent(bytes: number, stages: readonly Stage[] = STAGES): number {
  const live = computeLiveness(stages);
  const names = [...live.keys()].filter((n) => BUFFERS[n]!.kind === 'storage' && BUFFERS[n]!.bytes(DIMS) === bytes);
  let peak = 0;
  for (let i = 0; i < stages.length; i++) {
    const c = names.filter((n) => live.get(n)!.first <= i && i <= live.get(n)!.last).length;
    if (c > peak) peak = c;
  }
  return peak;
}

test('the declared pipeline plans cleanly in both modes', () => {
  // planPool calls validate() internally, so this also covers the declaration
  // hygiene rules: no stage binds a name twice, no stage dispatches indirectly
  // off a buffer it also binds, every `written` note names a real stage.
  assert.doesNotThrow(() => planPool(DIMS, { alias: false }));
  assert.doesNotThrow(() => planPool(DIMS, { alias: true }));
});

test('every declared buffer is bound by some stage', () => {
  const live = computeLiveness();
  for (const name of Object.keys(BUFFERS)) {
    assert.ok(live.has(name), `BUFFERS declares '${name}' but no stage binds it`);
  }
});

test('liveness counts indirectFrom, which is bound by no bind group', () => {
  const live = computeLiveness();
  // grow.gate binds growArgs; hook and compress only dispatch off it. If
  // indirectFrom were ignored, growArgs' range would still look right here --
  // so assert on the one where it is load-bearing: regionArgs is written by
  // collect.regionMeta and then dispatched off by collect.finalize and lsdFit,
  // neither of which BINDS it.
  //
  // (This named lines.flag until §9 was built. That stage now dispatches over
  // maxRegions directly, because a scan input has to be written past the region
  // count -- so lsdFit is the last indirect consumer. The test caught the change,
  // which is what it is for.)
  const iv = live.get('regionArgs')!;
  const idx = (id: string) => STAGES.findIndex((s) => s.id === id);
  assert.equal(iv.first, idx('collect.regionMeta'));
  assert.equal(iv.last, idx('lsdFit'), 'regionArgs must stay live through its last indirect consumer');
});

test('alias:false gives every storage buffer its own slot', () => {
  const plan = planPool(DIMS, { alias: false });
  const storage = Object.keys(BUFFERS).filter((n) => BUFFERS[n]!.kind === 'storage');
  assert.equal(plan.slots.length, storage.length);
  assert.equal(new Set(plan.assignment.values()).size, storage.length);
});

test('alias:true reaches the peak-concurrency lower bound for the per-pixel class', () => {
  const plan = planPool(DIMS, { alias: true });
  const slotsAtN4 = plan.slots.filter((s) => s.bytes === N4).length;
  const bound = peakConcurrent(N4);
  // Interval graphs are perfect, so greedy left-to-right colouring is optimal:
  // the slot count must EQUAL the peak, not merely be near it. A regression
  // here means the colouring stopped being greedy-by-birth-order.
  assert.equal(slotsAtN4, bound, `n*4 slots ${slotsAtN4} should equal peak concurrency ${bound}`);

  // The headline number this design was justified on. If it moves, the memory
  // figures in full_system_breakdown.md §18 need updating with it -- which is
  // the point, and it has already fired once: redrawing collect around one vec2
  // scan took the full-image array count from 15 to 10 and the slots from 9 to 7.
  assert.equal(slotsAtN4, 7);

  const unaliased = Object.keys(BUFFERS).filter((n) => BUFFERS[n]!.kind === 'storage' && BUFFERS[n]!.bytes(DIMS) === N4).length;
  assert.equal(unaliased, 10, 'ten full-image n*4 arrays');
  // The vec2 scan's two n*8 arrays cannot alias each other -- a scan reads its
  // source while writing its destination -- so they are two slots, and they cost
  // exactly what the four n*4 arrays they replaced did.
  const n8 = Object.keys(BUFFERS).filter((n) => BUFFERS[n]!.kind === 'storage' && BUFFERS[n]!.bytes(DIMS) === N4 * 2).length;
  assert.equal(n8, 2);
  assert.equal(plan.slots.filter((s) => s.bytes === N4 * 2).length, 2);
});

test('pooling saves what the plan claims and nothing is lost', () => {
  const off = planPool(DIMS, { alias: false });
  const on = planPool(DIMS, { alias: true });
  assert.ok(on.totalBytes < off.totalBytes);
  // Every buffer still resolves under both plans -- pooling reassigns, it never
  // drops.
  for (const name of Object.keys(BUFFERS)) {
    const dedicated = BUFFERS[name]!.kind !== 'storage';
    assert.equal(on.assignment.has(name) || dedicated, true, `'${name}' vanished from the aliased plan`);
  }
  const savedMiB = (off.totalBytes - on.totalBytes) / 1048576;
  assert.ok(savedMiB > 5.0 && savedMiB < 5.6, `expected ~5.3 MiB saved, got ${savedMiB.toFixed(2)}`);
});

// ── Inspection: a live range, not an exception ────────────────────────────

test('WITHOUT the declaration, an inspected buffer would be read back CLOBBERED', () => {
  // THE DECISIVENESS STEP, and it comes first on purpose. The test below is
  // worth nothing unless `alias: true` genuinely parks a later buffer in
  // `lines`' slot -- if it did not, that test would stay green with the whole
  // mechanism deleted, which is the shape six earlier mutations in this rewrite
  // hid behind.
  //
  // Measured, not chosen: at 480x640 `lines` (maxLines*16) dies at join.reduce
  // -- gpp.classify reads `compLines` now, so `lines` dies one stage earlier
  // than it used to -- and `samples` is born at gpp.classify, so the colouring
  // hands the second the first's slot. Reading `lines` after the last stage
  // would return samples' bytes: a wrong answer that looks like a detector
  // fault, since both arrays are plausible-looking f32.
  //
  // The successor changed when the join landed (it was `colSamples`). That is
  // this fixture working as intended -- it is pinned to a MEASUREMENT, so a
  // liveness change has to be re-measured rather than silently absorbed.
  const plain = planPool(DIMS, { alias: true });
  const slot = plain.slots[plain.assignment.get('lines')!]!;
  const after = slot.occupants.slice(slot.occupants.indexOf('lines') + 1);
  assert.deepEqual(after, ['samples'],
    'lines no longer aliases anything -- this fixture can no longer see the mechanism, pick another buffer');
});

test('a declared-inspectable buffer holds its slot to the end of the frame', () => {
  const declared = planPool(DIMS, { alias: true, inspect: ['lines'] });
  const slot = declared.slots[declared.assignment.get('lines')!]!;
  assert.deepEqual(slot.occupants, ['lines'], 'an inspected buffer shares with nobody');

  // And the cost is REAL and visible, which is the honest half: colSamples has
  // to go somewhere, and there is nowhere free at that size.
  const plain = planPool(DIMS, { alias: true });
  assert.equal(declared.slots.length, plain.slots.length + 1);
  assert.ok(declared.totalBytes > plain.totalBytes);
  // Carried on the plan so run.ts sizes one staging buffer from it, and so a
  // per-frame request for anything else is a throw rather than a wrong answer.
  assert.deepEqual([...declared.inspect], ['lines']);
});

test('inspecting a buffer that is already last in its slot costs nothing', () => {
  // Stated because it is what makes a generous `inspect` catalogue affordable:
  // the price is paid per buffer that actually gets displaced, not per name.
  // `layout` shares slot with three earlier occupants and is the last of them,
  // so it is already readable at end of frame and declaring it changes nothing.
  const plain = planPool(DIMS, { alias: true });
  const declared = planPool(DIMS, { alias: true, inspect: ['layout'] });
  assert.equal(declared.slots.length, plain.slots.length);
  assert.equal(declared.totalBytes, plain.totalBytes);
});

test('inspection is free under alias: false, because everything already holds its own slot', () => {
  const plain = planPool(DIMS, { alias: false });
  const declared = planPool(DIMS, { alias: false, inspect: ['lines', 'votes', 'fx', 'fy'] });
  assert.equal(declared.slots.length, plain.slots.length);
  assert.equal(declared.totalBytes, plain.totalBytes);
  // Which is the whole reason Pose Viewer can inspect freely: the degenerate
  // liveness table already says every buffer is live for the whole frame.
});

test('inspect rejects a name that cannot be copied from, rather than failing at copy time', () => {
  // Uniform/indirect/persistent buffers are not created with COPY_SRC (see
  // createBuffers), and copying from one is a validation error WebGPU reports
  // ASYNCHRONOUSLY -- a silently no-op encoder, not an exception. So the name is
  // rejected where the mistake is, not three steps downstream.
  assert.throws(() => planPool(DIMS, { inspect: ['gradientUni'] }), /is a uniform buffer/);
  assert.throws(() => planPool(DIMS, { inspect: ['lineArgs'] }), /is an? indirect buffer/);
  assert.throws(() => planPool(DIMS, { inspect: ['torus'] }), /is a persistent buffer/);
  assert.throws(() => planPool(DIMS, { inspect: ['nosuchbuffer'] }), /not in BUFFERS/);
});

test('no stage binds two buffers that share a slot', () => {
  // Tautological on a plan derived from the same stage list -- two buffers
  // listed in one stage are both live in it by construction. Asserted anyway,
  // because "by construction" is a claim about the colouring, and this is the
  // property the whole design rests on. The check that catches real drift is
  // assertBinds; see below.
  for (const alias of [false, true]) {
    const plan = planPool(DIMS, { alias });
    for (const stage of STAGES) {
      const slots = new Set<number>();
      for (const name of stage.binds) {
        const s = plan.assignment.get(name);
        if (s === undefined) continue;
        assert.equal(slots.has(s), false, `stage '${stage.id}' binds slot ${s} twice (alias=${alias})`);
        slots.add(s);
      }
    }
  }
});

test('assertBinds catches an encode function that binds an undeclared buffer', () => {
  // THE MUTATION TEST. This is the failure the whole scheme exists to prevent,
  // and it is the one validate() structurally cannot see: code that binds a
  // buffer pipeline.ts does not list for that stage. Without this, the plan
  // frees the buffer early, a later one inherits its slot, and the pass binds
  // one GPUBuffer twice -- reported asynchronously, so the symptom is a
  // silently no-op encoder.
  const plan = planPool(DIMS, { alias: true });
  const stage = STAGES.find((s) => s.id === 'collect.scatter')!;

  assert.doesNotThrow(() => assertBinds(plan, 'collect.scatter', stage.binds));

  // Binds one extra buffer that the declaration does not list.
  assert.throws(
    () => assertBinds(plan, 'collect.scatter', [...stage.binds, 'labelSurvives']),
    /binds but does not declare: labelSurvives/,
  );
  // Declares one the code forgot -- drift in the other direction, which frees
  // nothing but means the plan is describing a pipeline that does not exist.
  assert.throws(
    () => assertBinds(plan, 'collect.scatter', stage.binds.filter((n) => n !== 'cursor')),
    /declares but does not bind: cursor/,
  );
});

test('the collision guarantee is a COMPOSITION, not a third check', () => {
  // No pass may bind one GPUBuffer twice -- WebGPU reports that asynchronously,
  // so the symptom is a silently no-op encoder rather than an error. Two checks
  // compose into that guarantee:
  //
  //   validate()    no stage's DECLARED binds share a pool slot
  //   assertBinds() what the code binds IS what the stage declares
  //
  // assertBinds used to re-check slots itself. That was unreachable -- its set
  // comparison throws first -- and the test for it passed on the wrong error.
  // Deleted; this pins the composition so the deletion stays honest.
  const plan = planPool(DIMS, { alias: true });

  // ── Half one, and note HOW it has to be reached ──
  //
  // A colliding plan cannot be built through planPool at all: liveness is
  // derived from `binds`, so adding a buffer to a stage extends its live range,
  // and the colouring simply stops sharing that slot. The check is only
  // reachable by corrupting an assignment directly, which is what a hand-edited
  // plan would amount to. So it is real, but it guards an import path rather
  // than a mistake anyone makes in pipeline.ts.
  const stage = STAGES.find((s) => s.binds.filter((b) => plan.assignment.has(b)).length >= 2)!;
  const [first, second] = stage.binds.filter((b) => plan.assignment.has(b));
  const corrupted = { ...plan, assignment: new Map(plan.assignment) };
  corrupted.assignment.set(second!, plan.assignment.get(first!)!);
  assert.throws(() => validate(corrupted), /usage-scope conflict in stage '/);

  // ── Half two: code that binds anything other than the declaration ──
  // This is the one that guards a mistake people actually make, and it is what
  // keeps half one meaningful.
  const other = plan.slots.find((sl) => sl.occupants.length > 1)!.occupants[1]!;
  assert.throws(() => assertBinds(plan, stage.id, [...stage.binds, other]),
    /binds but does not declare/);
});

test('the clear schedule puts each zeroed buffer at its first bind', () => {
  const plan = planPool(DIMS, { alias: true });
  const zeroed = Object.keys(BUFFERS).filter((n) => BUFFERS[n]!.zero);
  assert.ok(zeroed.length > 0);
  const scheduled = new Set([...plan.clears.values()].flat());
  for (const name of zeroed) {
    assert.ok(scheduled.has(name), `'${name}' is zero:true but never scheduled for a clear`);
    const at = [...plan.clears].find(([, list]) => list.includes(name))![0];
    assert.equal(at, plan.liveness.get(name)!.first,
      `'${name}' must be cleared at its first bind -- with pooling, a frame-start clear writes someone else's memory`);
  }
  // And nothing that is `written` rather than zeroed sneaks into the schedule:
  // those need a value, not a clear, and a clear would be actively wrong.
  for (const [name, spec] of Object.entries(BUFFERS)) {
    if (spec.written) assert.equal(scheduled.has(name), false, `'${name}' needs a WRITE, not a clear`);
  }
});

test('grow.args is written, not zeroed', () => {
  // The one buffer left whose correct initial state is not zero. Recorded in
  // full_system_breakdown.md §16 as the family a grep for `zero` misses; this is
  // the guard that keeps someone from "tidying" it into the clear list. A zeroed
  // growArgs makes the next reconstruction dispatch no rounds at all and emit
  // init's singleton labelling -- silently, at full speed.
  assert.equal(BUFFERS.growArgs!.zero, undefined);
  assert.equal(typeof BUFFERS.growArgs!.written, 'string');

  // `uvBounds` was the second member of that family and is DELETED, not fixed:
  // §12's line hull replaced the atomic min/max it was the init target for, the
  // same move that deleted gpp.extentInit. An initialization pass is usually a
  // symptom of choosing an atomic where a reduction would do -- and here the
  // reduction was already being computed one stage earlier.
  assert.equal(BUFFERS.uvBounds, undefined,
    'uvBounds is back -- the decode lattice is bounded by the line hull now');
  // The whole `written` set, pinned. Two different things wear this mark and
  // only one of them is §16's dangerous family: an indirect-args triple whose
  // producing stage always runs (regionArgs, lineArgs) is ordinary, while
  // growArgs is the one whose correct initial VALUE is not zero. A new entry in
  // either group is a §16 table update, so the test is on the set rather than on
  // a property no predicate can tell apart.
  assert.deepEqual(
    Object.entries(BUFFERS).filter(([, s]) => s.written).map(([n]) => n).sort(),
    ['growArgs', 'lineArgs', 'regionArgs']);
});

test('every indirect-args buffer is cleared', () => {
  // Not because something fails to write them, but because the case it covers
  // is the pass that writes them NOT RUNNING: a failed bind group is a silent
  // no-op, while the indirect dispatch reading them is a separate valid command
  // that would launch over the previous frame's extent.
  for (const [name, spec] of Object.entries(BUFFERS)) {
    if (spec.kind !== 'indirect') continue;
    assert.ok(spec.zero || spec.written, `indirect args '${name}' must be cleared or explicitly written`);
  }
});

test('planning is deterministic', () => {
  // A plan that shuffles between runs would make the aliased-vs-unaliased
  // comparison test worthless, since a difference could be the shuffle.
  const a = describePlan(planPool(DIMS, { alias: true }));
  const b = describePlan(planPool(DIMS, { alias: true }));
  assert.equal(a, b);
});

test('validate rejects a stage that dispatches off a buffer it also binds', () => {
  // A buffer cannot be both a writable binding and the indirect source of the
  // same dispatch. WebGPU reports that asynchronously, so the symptom is the
  // silently no-op encoder again.
  const broken: Stage[] = STAGES.map((s) =>
    s.id === 'decode.build' ? { ...s, binds: [...s.binds, 'buildArgs'] } : s);
  assert.throws(() => planPool(DIMS, { alias: true, stages: broken }), /dispatches indirectly off 'buildArgs' while also binding it/);
});

test('validate rejects a duplicated bind', () => {
  const broken: Stage[] = STAGES.map((s) =>
    s.id === 'gradient' ? { ...s, binds: [...s.binds, 'fx'] } : s);
  assert.throws(() => planPool(DIMS, { alias: true, stages: broken }), /binds 'fx' twice/);
});

test('the scan spine fits in ONE workgroup, which is what makes it two levels', () => {
  // THE assumption the whole scan design rests on. If the block sums ever
  // outgrew what a single workgroup can scan, three passes would silently stop
  // being enough and the scan would return partial sums -- a wrong answer that
  // looks like a plausible one. planPool asserts it; this pins the arithmetic.
  assert.equal(scanBlocks(307200), 300);
  assert.ok(300 <= SCAN_THREADS * 4, 'a 480x640 frame must not need more than 4 elements per spine thread');

  // Single-block base case still needs a slot, or the add pass reads nothing.
  assert.equal(scanBlocks(1), 1);
  assert.equal(scanBlocks(SCAN_BLOCK), 1);
  assert.equal(scanBlocks(SCAN_BLOCK + 1), 2);

  // THE ACTUAL CEILING, and it is lower than a first guess suggests. One spine
  // workgroup covers 256 * 16 = 4096 blocks, so 4096 * SCAN_BLOCK = 4.2M pixels,
  // i.e. 2048x2048. Past that the scan needs a third level and this design stops
  // being correct -- planPool throws rather than silently returning partial sums.
  assert.ok(scanBlocks(2048 * 2048) <= SCAN_THREADS * 16);
  assert.ok(scanBlocks(4096 * 4096) > SCAN_THREADS * 16, 'the ceiling must be a real one');
});
