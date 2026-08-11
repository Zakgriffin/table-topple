import test from 'node:test';
import assert from 'node:assert/strict';
import {
  type CriticalPath, type PathNode, type StageRecord, type StageTable,
  criticalPath, joinRecords, profilerBeginSession, spanEnd, spanStart,
} from '../src/sphereLab/profiling/profiler.ts';
import { recordTransfer, transfersFrom } from '../src/pose/gpu/device.ts';
import { POSE_STAGE_TABLE } from '../src/pose/timing/stages.ts';
import { ALL_STAGES } from '../src/sphereLab/profiling/stages.ts';
import { runPoseOn } from '../src/sphereLab/harness/runPose.ts';
import { loadInput } from './helpers/fixtures.ts';

// ── The join, tested on synthetic intervals ───────────────────────────────
//
// Structure used to come from a module-level span stack, which had two failure
// modes it could only REPORT: a concurrent operation's spans got reparented,
// and genuinely concurrent siblings recorded as parent/child produced negative
// self times. Both were fixed by construction rather than by check, and a fix
// by construction needs a test that would fail if the construction changed --
// otherwise "it cannot happen" is just an assertion in a comment.
//
// Synthetic records rather than a real pipeline run on purpose: the cases that
// matter (a foreign operation interleaved at exactly the wrong moment, two
// awaits overlapping by a specific amount) cannot be provoked reliably by
// running the real thing, which is precisely why the old design's failures were
// found by reading output rather than by testing.

let seq = 0;
function rec(id: string, start: number, end: number): StageRecord {
  seq++;
  return { id, start, end, attrs: null };
}

const TABLE: StageTable = {
  root: { label: 'root', within: null },
  a: { label: 'a', within: 'root' },
  b: { label: 'b', within: 'root' },
  leaf: { label: 'leaf', within: 'a' },
  // A second root, standing in for the display drain: a KNOWN stage that can
  // legitimately enclose a whole `root` subtree in wall-clock time without
  // being anybody's declared parent.
  drain: { label: 'drain', within: null },
};

test('a child attaches to the containing occurrence of its DECLARED parent', () => {
  const j = joinRecords([rec('root', 0, 100), rec('a', 10, 40), rec('leaf', 15, 20)], TABLE);
  assert.equal(j.roots.length, 1);
  const root = j.roots[0];
  assert.equal(root.children.length, 1);
  assert.equal(root.children[0].rec.id, 'a');
  assert.equal(root.children[0].children[0].rec.id, 'leaf');
  assert.deepEqual(j.orphans, []);
  assert.deepEqual(j.unknown, []);
});

test('two occurrences of a parent take the child that falls inside each', () => {
  const j = joinRecords([
    rec('root', 0, 100),
    rec('a', 10, 20), rec('a', 50, 60),
    rec('leaf', 12, 15), rec('leaf', 52, 55),
  ], TABLE);
  const as = j.roots[0].children;
  assert.equal(as.length, 2);
  assert.equal(as[0].children[0].rec.start, 12);
  assert.equal(as[1].children[0].rec.start, 52);
});

// ── THE NEGATIVE CONTROL for reparenting ──
//
// This is the case the old stack got wrong, and the reason the join needs a
// DECLARED parent rather than "whichever interval contains this one". The
// foreign record's interval strictly contains the whole pose subtree -- exactly
// what a display drain suspended across a capture looks like -- so containment
// alone would adopt it.
test('a foreign record enclosing everything adopts nothing, and changes no self time', () => {
  const pose = [rec('root', 10, 90), rec('a', 20, 40), rec('leaf', 25, 30)];
  const clean = joinRecords(pose, TABLE);
  const contaminated = joinRecords([rec('other.op', 0, 1000), ...pose], TABLE);

  assert.equal(contaminated.roots.length, 1, 'the foreign record must not become a root of this table');
  assert.equal(contaminated.unknown.length, 1);
  assert.equal(contaminated.unknown[0].id, 'other.op');
  // The measurement itself is untouched -- that is the property that lets the
  // harness stop VOIDING a breakdown when something else was running.
  assert.equal(contaminated.roots[0].selfMs, clean.roots[0].selfMs);
  assert.equal(contaminated.roots[0].children.length, clean.roots[0].children.length);
});

// The same control with a KNOWN id, which is the one that actually
// discriminates the design. The test above passes under a
// containment-only join too, because an unknown id would be filtered out
// either way. `drain` is in the table, and its interval strictly contains
// `root`'s -- so a join that inferred structure from interval nesting would
// make the whole reconstruction a child of the drain, which is the exact
// reparenting the old span stack did and the reason `within` is declared.
test('a KNOWN stage enclosing the tree does not adopt it, because it is not declared its parent', () => {
  const j = joinRecords([
    rec('drain', 0, 1000),
    rec('root', 10, 90), rec('a', 20, 40), rec('leaf', 25, 30),
  ], TABLE);
  const drain = j.roots.find((n) => n.rec.id === 'drain')!;
  const root = j.roots.find((n) => n.rec.id === 'root')!;
  assert.equal(j.roots.length, 2, 'both must be roots; neither declares the other');
  assert.deepEqual(drain.children, [], 'the enclosing stage must adopt nothing');
  assert.equal(drain.selfMs, 1000, 'and so its self time must not have the tree subtracted from it');
  assert.equal(root.children[0].rec.id, 'a');
  assert.deepEqual(j.orphans, []);
});

// ── THE NEGATIVE CONTROL for negative self time ──
//
// `a` and `b` overlap: 20-40 and 30-60, union 20-60 = 40ms of a 100ms parent.
// Summing their durations gives 20 + 30 = 50, so the sum and the union differ
// and the test can tell them apart. That is the whole point -- a test where
// they agree would pass under the old arithmetic too.
test('overlapping children are subtracted as a union, not as a sum', () => {
  const j = joinRecords([rec('root', 0, 100), rec('a', 20, 40), rec('b', 30, 60)], TABLE);
  assert.equal(j.roots[0].selfMs, 60, 'expected 100 - |[20,60]| = 60, not 100 - (20+30) = 50');
});

test('self time cannot go negative even when children cover the parent entirely', () => {
  const j = joinRecords([rec('root', 0, 50), rec('a', 0, 50), rec('b', 0, 50)], TABLE);
  assert.equal(j.roots[0].selfMs, 0);
});

// ── Root vs. orphan ──

test("a stage whose declared parent never ran is a ROOT, not an orphan", () => {
  // What the headless harness sees: `pose.run` with no `app.reconstruct` above
  // it. A false orphan here would put a warning on every harness run.
  const j = joinRecords([rec('a', 10, 20), rec('leaf', 12, 15)], TABLE);
  assert.deepEqual(j.orphans, []);
  assert.equal(j.roots.length, 1);
  assert.equal(j.roots[0].rec.id, 'a');
});

test('a stage whose declared parent ran DISJOINTLY is a root, not an orphan', () => {
  // `project.bins` runs inside the tail's `app.project` and also straight off a
  // mode switch with no tail at all. The second must stay quiet.
  const j = joinRecords([rec('root', 0, 10), rec('a', 50, 60)], TABLE);
  assert.deepEqual(j.orphans, []);
  assert.equal(j.roots.length, 2);
});

test('a stage that STRADDLES its declared parent is an orphan, and only that case is', () => {
  const j = joinRecords([rec('root', 0, 50), rec('a', 40, 80)], TABLE);
  assert.equal(j.orphans.length, 1);
  assert.equal(j.orphans[0].id, 'a');
});

// ── The declaration itself ──

// Both tables, because ALL_STAGES is where the CROSS-table edges live -- the
// app declares `app.project <- pose.drain` and overrides `pose.run` to sit
// inside `app.reconstruct`, and neither reference is visible from the library
// side. A dangling one there does not fail to compile: the fields are plain
// strings, and the only symptom is a stage that silently never joins.
for (const [name, table] of [['pose', POSE_STAGE_TABLE], ['app (ALL_STAGES)', ALL_STAGES]] as const) {
  test(`every \`within\` and \`inputs\` in the ${name} table names a stage that exists`, () => {
    const ids = new Set(Object.keys(table));
    const bad: string[] = [];
    for (const [id, n] of Object.entries(table)) {
      if (n.within !== null && !ids.has(n.within)) bad.push(`${id}.within -> ${n.within}`);
      for (const i of n.inputs ?? []) if (!ids.has(i)) bad.push(`${id}.inputs -> ${i}`);
    }
    assert.deepEqual(bad, [], `dangling stage references: ${bad.join(', ')}`);
  });

  test(`dependency edges in the ${name} table are acyclic`, () => {
    // Containment has its own check below; `inputs` needs one too, and for a
    // sharper reason: the backward walk follows these edges, and only the
    // monotonically-decreasing end times stop a cycle in the DECLARATION from
    // becoming a cycle in the walk. Keep the declaration acyclic and that
    // argument does not have to hold on its own.
    const visiting = new Set<string>(), done = new Set<string>();
    const visit = (id: string, trail: string[]): void => {
      if (done.has(id)) return;
      assert.ok(!visiting.has(id), `dependency cycle: ${[...trail, id].join(' -> ')}`);
      visiting.add(id);
      for (const i of table[id].inputs ?? []) visit(i, [...trail, id]);
      visiting.delete(id);
      done.add(id);
    };
    for (const id of Object.keys(table)) visit(id, []);
  });

  test(`containment in the ${name} table is acyclic`, () => {
    for (const id of Object.keys(table)) {
      const seen = new Set<string>([id]);
      let cur: string | null = table[id].within;
      while (cur !== null) {
        assert.ok(!seen.has(cur), `containment cycle through ${cur}`);
        seen.add(cur);
        cur = table[cur].within;
      }
    }
  });

  // A `within` that points at a stage the parent does not otherwise relate to
  // is fine; a stage declaring its own CONTAINER as a dependency is not -- that
  // is the pose.votes/pose.residency defect, in declaration form. The join
  // would leave it permanently unsatisfied, since a parent cannot finish before
  // the child it contains starts.
  test(`no stage in the ${name} table declares its own container as an input`, () => {
    const bad: string[] = [];
    for (const [id, n] of Object.entries(table)) {
      const ancestors = new Set<string>();
      for (let cur = n.within; cur !== null; cur = table[cur]?.within ?? null) ancestors.add(cur);
      for (const i of n.inputs ?? []) if (ancestors.has(i)) bad.push(`${id}.inputs -> ${i} (its own ancestor)`);
    }
    assert.deepEqual(bad, []);
  });
}

// ── The declarations against the REAL call graph ──
//
// Everything above tests the join on intervals someone made up. This runs an
// actual CPU reconstruction and joins its actual records, which is the only
// thing that can catch a `within` that is merely WRONG -- declaring a stage
// inside one that never encloses it. The synthetic tests cannot see that, and
// neither can tsc: a bad `within` is a perfectly well-typed string.
//
// CPU backend, so it runs headless. `fit.ata`/`fit.eigen` DO record here now
// (they used to be GPU-only, named after where the awaits fell); what is still
// uncovered is the GPU-only set -- lsd.fitDispatch and decode.fused -- which is
// owed a device session, stated rather than silently implied, because a green
// run here is not a green run for the whole table.
test('a real CPU reconstruction joins into ONE tree with no orphans', async () => {
  const input = loadInput();
  const session = profilerBeginSession();
  let records;
  try {
    // runPoseOn drains unconditionally -- there is nothing to request -- so
    // `pose.drain` records here. It is declared a root in the library table and
    // this is what proves it does not accidentally land inside the pose.
    await runPoseOn(input, 'cpu');
    records = session.takeRepRecords();
  } finally {
    session.end();
  }

  const j = joinRecords(records, POSE_STAGE_TABLE);
  assert.deepEqual(j.unknown.map((u) => u.id), [], 'a pose run recorded an id the library does not declare');
  assert.deepEqual(
    j.orphans.map((o) => o.id), [],
    'a stage straddled its declared parent -- the `within` for it is wrong in pose/timing/stages.ts',
  );
  // pose.run and pose.drain: the drain happens after the pose returns, so two
  // roots is the correct answer here and one would mean the drain got adopted.
  assert.deepEqual(j.roots.map((r) => r.rec.id).sort(), ['pose.drain', 'pose.run']);

  const run = j.roots.find((r) => r.rec.id === 'pose.run')!;
  // Every self time non-negative, on real intervals rather than made-up ones.
  const walk = (n: typeof run): void => {
    assert.ok(n.selfMs >= 0, `${n.rec.id} has negative self time ${n.selfMs}`);
    assert.ok(
      n.selfMs <= n.durationMs + 1e-9,
      `${n.rec.id} self ${n.selfMs} exceeds its own duration ${n.durationMs}`,
    );
    n.children.forEach(walk);
  };
  walk(run);

  // The CPU path's stages, present and nested where the table says. Named
  // explicitly so that a stage silently ceasing to record shows up as a
  // failure rather than as a quietly shorter table.
  const ids = new Set<string>();
  const collect = (n: typeof run): void => { ids.add(n.rec.id); n.children.forEach(collect); };
  collect(run);
  for (const id of ['pose.votes', 'pose.composites', 'lsd.gradient', 'lsd.grow',
    'votes.filter', 'votes.segments', 'pose.fit', 'pose.assembly', 'pose.distance',
    'gpp.classify', 'gpp.search', 'pose.decode', 'decode.build', 'decode.tally']) {
    assert.ok(ids.has(id), `${id} did not record inside pose.run on a CPU reconstruction`);
  }
});

// ── THE PER-OCCURRENCE PARENT (StageRecord.within) ────────────────────────
//
// Transfers are one call site serving a dozen stages, so their owner is passed
// per call rather than declared in the table. These check that the override
// wins in BOTH directions -- a one-sided test would pass under an
// implementation that only ever ADDED a parent, and the `null` direction is the
// one that keeps an unattributed transfer out of somebody else's decomposition.

test('a record\'s own `within` overrides the table', () => {
  // `b` is declared within `root`; this occurrence says `a`.
  const j = joinRecords([
    rec('root', 0, 100), rec('a', 10, 60), { ...rec('b', 20, 30), within: 'a' },
  ], TABLE);
  const a = j.roots[0].children.find((n) => n.rec.id === 'a')!;
  assert.deepEqual(a.children.map((c) => c.rec.id), ['b']);
  assert.equal(j.roots[0].children.length, 1, 'b must not ALSO be a child of root');
});

test('an explicit `within: null` makes a record a root even when the table names a parent', () => {
  const j = joinRecords([rec('root', 0, 100), { ...rec('a', 10, 60), within: null }], TABLE);
  assert.equal(j.roots.length, 2);
  assert.equal(j.roots[0].selfMs, 100, 'and its time is NOT subtracted from the parent it opted out of');
});

test('an override naming a stage that does not contain it leaves it a root', () => {
  // The misattribution case: a transfer names an owner whose span had already
  // closed. It has to stay OUT of the tree rather than being adopted by
  // whatever else was open -- that is the whole reparenting argument, applied
  // to a per-call parent. The harness reports these; see CrossingReport.
  const j = joinRecords([
    rec('root', 0, 100), rec('a', 10, 20), { ...rec('b', 50, 60), within: 'a' },
  ], TABLE);
  const b = j.roots.find((n) => n.rec.id === 'b')!;
  assert.ok(b, 'b did not join, so it must be a root');
  assert.deepEqual(j.orphans, [], 'disjoint, not straddling, so not an orphan either');
});

// ── The ledger is a VIEW of the record store, not a second store ──
//
// recordTransfer needs no device, so this runs headless and covers the whole
// round trip: a crossing recorded with an owner, joined under that owner, and
// projected back into the TransferSample shape every consumer reads.
test('a recorded transfer is one record, attributed and projected back', () => {
  const session = profilerBeginSession();
  let recs;
  try {
    const outer = spanStart('pose.decode');
    const t0 = performance.now();
    recordTransfer({
      what: 'test:payload', kind: 'readback', dir: 'down', bytes: 4096,
      ms: 2, startMs: t0, bareFenceMs: null, queueDrainMs: null,
    }, 'pose.decode');
    // A real spin, so the owning span really does CONTAIN the crossing. The
    // join is containment-checked, so a test that recorded a 2ms transfer and
    // closed the parent immediately would fail for the right reason -- and
    // faking it by hand-building records would stop testing recordTransfer.
    while (performance.now() - t0 < 3) { /* hold the stage open */ }
    spanEnd(outer);
    recs = session.takeRepRecords();
  } finally {
    session.end();
  }

  const led = transfersFrom(recs);
  assert.equal(led.length, 1, 'exactly one crossing, from one record');
  assert.equal(led[0].what, 'test:payload');
  assert.equal(led[0].kind, 'readback');
  assert.equal(led[0].bytes, 4096);
  assert.ok(Math.abs(led[0].ms - 2) < 1e-6, `expected the caller's own 2ms, got ${led[0].ms}`);

  // And it joined INSIDE the stage that claimed it, which is the half that did
  // not exist while the ledger was a separate array.
  const j = joinRecords(recs, POSE_STAGE_TABLE);
  const decode = j.roots.find((n) => n.rec.id === 'pose.decode')!;
  assert.deepEqual(decode.children.map((c) => c.rec.id), ['xfer.readback']);
  assert.ok(decode.selfMs < decode.durationMs, 'its cost must come OUT of the stage self time');
});

test('an unattributed transfer is a root, not adopted by whatever was open', () => {
  const session = profilerBeginSession();
  let recs;
  try {
    const outer = spanStart('pose.decode');
    recordTransfer({
      what: 'test:orphan', kind: 'upload', dir: 'up', bytes: 8,
      ms: 0.1, startMs: performance.now(), bareFenceMs: null, queueDrainMs: null,
    });
    spanEnd(outer);
    recs = session.takeRepRecords();
  } finally {
    session.end();
  }
  const j = joinRecords(recs, POSE_STAGE_TABLE);
  const decode = j.roots.find((n) => n.rec.id === 'pose.decode')!;
  assert.deepEqual(decode.children, [], 'containment alone must not adopt it');
  assert.equal(transfersFrom(recs).length, 1, 'but it is still IN the crossings table');
});

// ── THE CRITICAL PATH: joined on `inputs`, not `within` ───────────────────
//
// A separate table, because the containment fixture above has no dependency
// edges to speak of and the cases that matter here are about WHICH edge binds.
//
//   head -> mid -> tail            the spine
//   side                           runs early, feeds tail, and must LOSE to mid
//   deep                           produced inside `mid`, consumed by `tail`
//
const DEP: StageTable = {
  run: { label: 'run', within: null },
  head: { label: 'head', within: 'run' },
  mid: { label: 'mid', within: 'run', inputs: ['head'] },
  deep: { label: 'deep', within: 'mid' },
  side: { label: 'side', within: 'run', inputs: ['head'] },
  tail: { label: 'tail', within: 'run', inputs: ['side', 'deep'] },
};

function chainIds(cp: CriticalPath): string[] {
  const out: string[] = [];
  const walk = (ps: readonly PathNode[]): void => {
    for (const p of ps) { out.push(p.node.rec.id); walk(p.inner); }
  };
  walk(cp.chain);
  return out;
}

// ── THE CONTROL FOR THE LIFT ──
//
// `tail` declares `deep`, which runs two levels down inside `mid`. A resolver
// that only looked at SIBLINGS would find no producer for that edge, fall back
// to `side` (the other input), and report a chain that skips `mid` entirely --
// which is precisely the class of bug this found in the real table, where the
// whole 41ms LSD chain fell off the CPU path. So the assertion is not "the
// chain is right", it is "the deep edge beat the shallow one".
test('a producer nested inside a sibling still binds, lifted to that sibling', () => {
  const j = joinRecords([
    rec('run', 0, 100),
    rec('head', 0, 10),
    rec('side', 10, 12),       // ready at 12 -- early, so NOT binding
    rec('mid', 10, 40), rec('deep', 20, 30),
    rec('tail', 40, 50),
  ], DEP);
  const cp = criticalPath(j.roots[0]);
  assert.deepEqual(chainIds(cp), ['head', 'mid', 'deep', 'tail']);
  const tail = cp.chain[cp.chain.length - 1];
  assert.equal(tail.boundBy, 'deep', 'the binding input is the deep producer, named where it RAN');
  assert.equal(tail.readyAt, 30, 'readyAt is when the data existed, not when its container finished');
  assert.equal(tail.waitMs, 10, 'so the 10ms of `mid` after `deep` finished reads as wait');
  assert.deepEqual(cp.unsatisfied, []);
});

// The same fixture with the deep producer made EARLY, so `side` binds instead.
// Without this, a resolver that always preferred the deepest or the last-listed
// input would pass the test above for the wrong reason.
test('the binding input is the LATEST-finishing one, whichever that is', () => {
  const j = joinRecords([
    rec('run', 0, 100),
    rec('head', 0, 10),
    rec('mid', 10, 40), rec('deep', 11, 12),  // ready at 12
    rec('side', 30, 35),                      // ready at 35 -- now the later one
    rec('tail', 40, 50),
  ], DEP);
  const cp = criticalPath(j.roots[0]);
  assert.equal(cp.chain[cp.chain.length - 1].boundBy, 'side');
  assert.deepEqual(chainIds(cp), ['head', 'side', 'tail']);
});

test('the walk starts at the LAST-FINISHING stage, not the last recorded one', () => {
  // `side` is recorded after `tail` but finishes first. The floor is set by
  // whatever ends last, so the walk must begin at `tail` regardless of order.
  const j = joinRecords([
    rec('run', 0, 100),
    rec('head', 0, 10), rec('mid', 10, 20), rec('deep', 12, 15),
    rec('tail', 20, 60), rec('side', 10, 11),
  ], DEP);
  const cp = criticalPath(j.roots[0]);
  assert.equal(cp.chain[cp.chain.length - 1].node.rec.id, 'tail');
  assert.equal(cp.spanMs, 60, 'head start 0 to tail end 60');
});

test('a wait is the gap on the EDGE, not time inside either stage', () => {
  const j = joinRecords([rec('run', 0, 100), rec('head', 0, 10), rec('mid', 30, 40)], DEP);
  const cp = criticalPath(j.roots[0]);
  const mid = cp.chain[1];
  assert.equal(mid.readyAt, 10);
  assert.equal(mid.waitMs, 20, 'the 20ms between head ending and mid starting belongs to neither span');
  assert.equal(cp.chain[0].readyAt, null, 'the head declares no input that ran here');
  assert.equal(cp.chain[0].waitMs, 0);
});

// ── THE PAIR THAT MAKES `unsatisfied` MEAN SOMETHING ──
//
// Only the second of these two discriminates. A version that reported every
// unresolved input would pass the first and would then warn on EVERY run --
// `lsd.gradient` declares `pose.residency`, which by design runs a level above
// it, and the CPU backend records no `lsd.fitDispatch` at all. A warning that
// fires constantly is a warning nobody reads.
test('an input that ran but had not FINISHED is reported', () => {
  const j = joinRecords([rec('run', 0, 100), rec('head', 0, 50), rec('mid', 10, 40)], DEP);
  const cp = criticalPath(j.roots[0]);
  assert.deepEqual(cp.unsatisfied, ['mid <- head']);
});

test('an input that never recorded is SILENT, and the stage becomes a head', () => {
  const j = joinRecords([rec('run', 0, 100), rec('mid', 10, 40)], DEP);
  const cp = criticalPath(j.roots[0]);
  assert.deepEqual(cp.unsatisfied, [], 'a stage that did not run on this backend is not an anomaly');
  assert.equal(cp.chain.length, 1);
  assert.equal(cp.chain[0].readyAt, null);
});

test('a leaf has no inner path, and a childless root has no chain', () => {
  const j = joinRecords([rec('run', 0, 100)], DEP);
  const cp = criticalPath(j.roots[0]);
  assert.deepEqual(cp.chain, []);
  assert.equal(cp.spanMs, 0);
});

// ── THE CRITICAL PATH AGAINST THE REAL CALL GRAPH ──
//
// The synthetic tests above check the walk. This checks the DECLARATION, and it
// is the only thing that can: an `inputs` entry naming a stage that never
// records on this backend is a perfectly well-typed string, and its only
// symptom is a chain that quietly stops early.
//
// That is not hypothetical -- it is the defect this test was written after.
// `lsd.fit` was opened inside fitRegionsGPU, so a CPU run recorded none, and
// `votes.filter`'s declared input vanished. The walk terminated at
// `votes.filter` and reported 0.25ms of rectangle filtering as the whole of the
// LSD chain, with the 42ms of `lsd.grow` feeding it nowhere on the path. Every
// other check in this file was green throughout: the tree was well-formed, the
// self times were right, and the ranking was correct. Only the chain was wrong.
//
// CPU backend, so it runs headless. The GPU-only stages are still owed a device
// session -- and `lsd.fit`'s inner chain (fitDispatch -> fitUnpack -> wrap) is
// entirely among them, so the three edges declared there are UNVERIFIED here.
test('the critical path through a real CPU reconstruction reaches the whole chain', async () => {
  const input = loadInput();
  const session = profilerBeginSession();
  let records;
  try {
    await runPoseOn(input, 'cpu');
    records = session.takeRepRecords();
  } finally {
    session.end();
  }

  const j = joinRecords(records, POSE_STAGE_TABLE);
  const run = j.roots.find((r) => r.rec.id === 'pose.run')!;
  const cp = criticalPath(run);
  const ids = chainIds(cp);

  assert.deepEqual(
    cp.unsatisfied, [],
    'a declared input had not finished when its consumer started -- either `inputs` is wrong in '
    + 'pose/timing/stages.ts, or the two spans overlap and the edge is not a dependency',
  );
  // Named explicitly rather than counted: a stage silently dropping off the
  // chain is exactly the failure above, and a count would have gone from 13 to
  // 12 without saying which one left.
  // `pose.residency` is gone from this list along with the residency itself: it
  // measured a map insertion, and the gray upload it was named for happens
  // inside `lsd.gradient`. `lsd.collect` is NOT here either -- it is declared,
  // but on the CPU path `growRegionsCCL` runs the collect inside its own call,
  // so nothing records it and the walk treats an input that never recorded as
  // silent. That is the documented rule, not an omission; it becomes a real
  // stage on both backends at Step 3.
  for (const id of ['pose.votes', 'pose.composites', 'lsd.gradient',
    'lsd.grow', 'lsd.fit', 'votes.filter', 'votes.segments', 'pose.fit', 'pose.assembly',
    'pose.distance', 'gpp.classify', 'gpp.search', 'pose.decode']) {
    assert.ok(ids.includes(id), `${id} is not on the critical path of a CPU reconstruction`);
  }
  // `pose.votes` is the head of the reconstruction now, where it used to be
  // `pose.residency`: with the residency gone there is no sibling of the votes
  // stage left for it to wait on, so the chain starts at the first real stage.
  // The gradient is still the head of the chain INSIDE it -- asserted by its
  // presence above plus `unsatisfied` being empty.
  assert.equal(ids[0], 'pose.votes', 'the reconstruction no longer starts at its first stage');

  // The pipeline is fully serial on CPU, so the chain should account for very
  // nearly the whole reconstruction. A real number rather than a token one: if
  // this ever drops it means either genuine concurrency appeared (interesting)
  // or an edge stopped resolving (a bug), and both deserve a look.
  const share = cp.spanMs / run.durationMs;
  assert.ok(share > 0.9, `critical path covers only ${(share * 100).toFixed(0)}% of the reconstruction`);

  // Waits are gaps between stages, so they cannot exceed the window they sit in.
  const walkWaits = (ps: readonly PathNode[]): void => {
    for (const p of ps) {
      assert.ok(p.waitMs >= 0, `${p.node.rec.id} has a negative wait ${p.waitMs}`);
      assert.ok(p.waitMs <= cp.spanMs, `${p.node.rec.id} waits longer than the whole chain`);
      walkWaits(p.inner);
    }
  };
  walkWaits(cp.chain);
});

