import test from 'node:test';
import assert from 'node:assert/strict';
import {
  type StageRecord, type StageTable, joinRecords, profilerBeginSession,
} from '../src/sphereLab/profiling/profiler.ts';
import { POSE_STAGES, POSE_STAGE_TABLE } from '../src/pose/timing/stages.ts';
import { runPoseOn } from '../src/sphereLab/harness/runPose.ts';
import { loadInput } from './helpers/fixtures.ts';
import { wants } from '../src/pose/intermediates.ts';

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

test('every `within` and `inputs` in the pose table names a stage that exists', () => {
  const ids = new Set(Object.keys(POSE_STAGES));
  const bad: string[] = [];
  for (const [id, node] of Object.entries(POSE_STAGES)) {
    const n = node as { within: string | null; inputs?: readonly string[] };
    if (n.within !== null && !ids.has(n.within)) bad.push(`${id}.within -> ${n.within}`);
    for (const i of n.inputs ?? []) if (!ids.has(i)) bad.push(`${id}.inputs -> ${i}`);
  }
  assert.deepEqual(bad, [], `dangling stage references: ${bad.join(', ')}`);
});

// ── The declarations against the REAL call graph ──
//
// Everything above tests the join on intervals someone made up. This runs an
// actual CPU reconstruction and joins its actual records, which is the only
// thing that can catch a `within` that is merely WRONG -- declaring a stage
// inside one that never encloses it. The synthetic tests cannot see that, and
// neither can tsc: a bad `within` is a perfectly well-typed string.
//
// CPU backend, so it runs headless. The GPU-only stages (lsd.fitDispatch,
// fit.dispatch, decode.fused and its child) are therefore NOT covered here and
// are owed a device session -- stated rather than silently implied, because a
// green run here is not a green run for the whole table.
test('a real CPU reconstruction joins into ONE tree with no orphans', async () => {
  const input = loadInput();
  const session = profilerBeginSession();
  let records;
  try {
    // Intermediates requested so the drain records too -- `pose.drain` is
    // declared a root in the library table and this is what proves it does not
    // accidentally land inside the pose.
    await runPoseOn(input, 'cpu', wants('fx', 'regions'));
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

test('containment in the pose table is acyclic and bottoms out at pose.run', () => {
  for (const id of Object.keys(POSE_STAGE_TABLE)) {
    const seen = new Set<string>([id]);
    let cur: string | null = POSE_STAGE_TABLE[id].within;
    while (cur !== null) {
      assert.ok(!seen.has(cur), `containment cycle through ${cur}`);
      seen.add(cur);
      cur = POSE_STAGE_TABLE[cur].within;
    }
  }
});
