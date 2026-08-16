import { strict as assert } from 'node:assert';
import { test } from 'node:test';
import {
  type Span, formatSpanTree, getRecords, joinRecords, profilerReset,
  profilerSetDevToolsMirror, spanEnd, spanIngest, spanStart,
} from '../src/sphereLab/profiling/profiler.ts';
import { ALL_STAGES, appSpan, joinAll } from '../src/sphereLab/profiling/stages.ts';
import { GPU_UNATTRIBUTED_ID, ingestGpuFrame, ingestLinkSpans } from '../src/sphereLab/profiling/clocks.ts';
import type { GpuFrameTiming } from '../src/pose/pose.ts';

// ── The join, and the clock boundary ──────────────────────────────────────
//
// No device needed: everything here is arithmetic over records, which is the
// point -- the translation is the only part of the profiler that can be wrong in
// a way the pipeline cannot show you.

const spin = (ms: number) => { const t = performance.now(); while (performance.now() - t < ms); };

function findNode(roots: readonly { rec: Span; children: unknown[] }[], id: string): any {
  for (const r of roots as any[]) {
    if (r.rec.id === id) return r;
    const hit = findNode(r.children, id);
    if (hit) return hit;
  }
  return null;
}

// ── The table-less path ───────────────────────────────────────────────────
//
// This is the regression test for the defect that blocked Phase 4: joinRecords
// filtered on `table[r.id]` before it ever read `within`, so a record whose id
// no table declares could not join whatever parent it named.

test('a table-less id joins on its per-call parent, and keeps its own label', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(2);
  const child = spanIngest({
    id: 'gpu:lsd.gradient', start: parent.start + 0.5, end: parent.start + 1.0,
    clock: 'gpu', attrs: null, within: 'app.pose',
  });
  spanEnd(parent);

  const join = joinAll();
  assert.equal(join.unknown.length, 0, 'a record declaring a live parent must not be "unknown"');
  const node = findNode(join.roots, 'gpu:lsd.gradient');
  assert.ok(node, 'the GPU span did not join at all');
  assert.equal(node.node.label, 'gpu:lsd.gradient', 'the id is the label when no table names it');
  assert.equal(node.rec, child);
});

test('a table-less id with NO declared parent still stays out of the tree', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(1);
  // No `within`. This is a foreign subsystem recording into the same store, and
  // the whole value of `unknown` is that such records cannot land inside a stage
  // that is being decomposed.
  spanEnd(spanStart('some.other.subsystem'));
  spanEnd(parent);

  const join = joinAll();
  assert.deepStrictEqual(join.unknown.map((u) => u.id), ['some.other.subsystem']);
  assert.equal(findNode(join.roots, 'some.other.subsystem'), null);
});

// ── The translation ───────────────────────────────────────────────────────

/** A frame of three passes at known device offsets, inside a known host window. */
function fakeFrame(submittedAt: number, resolvedAt: number): GpuFrameTiming {
  return {
    submittedAt,
    resolvedAt,
    passes: [
      // 0.5ms long, starting at the submit.
      { stage: 'a', ns: 500_000, startNs: 0, index: 0 },
      // 0.25ms long, starting 0.5ms in -- back to back with the first.
      { stage: 'b', ns: 250_000, startNs: 500_000, index: 1 },
      // 0.25ms long, starting 1.0ms in, after a 0.25ms device-side gap.
      { stage: 'b', ns: 250_000, startNs: 1_000_000, index: 2 },
    ],
  };
}

test('GPU passes are placed on the host clock, anchored at the submit', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(4);
  const submittedAt = parent.start + 1;
  const resolvedAt = parent.start + 3;
  ingestGpuFrame(fakeFrame(submittedAt, resolvedAt), 'app.pose');
  spanEnd(parent);

  const recs = getRecords().filter((r) => r.id.startsWith('gpu:'));
  assert.equal(recs.length, 3);
  // Anchored: the first pass starts exactly at the submit, which is the lower
  // bound the design commits to -- not a measurement of when the GPU began.
  assert.equal(recs[0]!.start, submittedAt);
  // DURATIONS are the part that is exact. ns -> ms, unscaled.
  assert.equal(recs[0]!.end - recs[0]!.start, 0.5);
  assert.equal(recs[1]!.end - recs[1]!.start, 0.25);
  // RELATIVE SPACING is preserved: the 0.25ms device-side gap between pass 1
  // ending and pass 2 starting survives the translation.
  assert.equal(recs[2]!.start - recs[1]!.end, 0.25);
  for (const r of recs) assert.equal(r.clock, 'gpu');
});

test('the unattributed row closes the submit window exactly', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(4);
  const submittedAt = parent.start + 1;
  const resolvedAt = parent.start + 3;
  ingestGpuFrame(fakeFrame(submittedAt, resolvedAt), 'app.pose');
  spanEnd(parent);

  const un = getRecords().find((r) => r.id === GPU_UNATTRIBUTED_ID)!;
  assert.ok(un, 'the leftover must get a row rather than vanishing into self time');
  // The GPU block measured 1.25ms of the 2ms window, so 0.75ms is queue latency
  // plus fence plus map -- real cost, and for a one-fence pipeline it IS the
  // cost of crossing the bus.
  assert.equal(un.start, submittedAt + 1.25);
  assert.equal(un.end, resolvedAt);
  // Host-measured at both ends, whatever its start was derived from.
  assert.equal(un.clock, 'host');
});

test('a GPU block longer than its host window reports zero unattributed, never negative', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(4);
  // A window NARROWER than the GPU block: the anchor is a lower bound, so this
  // is possible in principle and must not produce a backwards span.
  const submittedAt = parent.start + 1;
  const resolvedAt = parent.start + 1.5;
  ingestGpuFrame(fakeFrame(submittedAt, resolvedAt), 'app.pose');
  spanEnd(parent);

  const un = getRecords().find((r) => r.id === GPU_UNATTRIBUTED_ID)!;
  assert.ok(un.end >= un.start, `unattributed ran backwards: ${un.start} -> ${un.end}`);
  assert.equal(un.end - un.start, 0);
});

test('the whole submit decomposes: GPU rows plus unattributed, nothing in self time', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(4);
  const submittedAt = parent.start + 1;
  const resolvedAt = parent.start + 3;
  ingestGpuFrame(fakeFrame(submittedAt, resolvedAt), 'app.pose');
  spanEnd(parent);

  const join = joinRecords(getRecords(), ALL_STAGES);
  const pose = findNode(join.roots, 'app.pose');
  assert.ok(pose);
  assert.equal(join.orphans.length, 0);
  assert.equal(join.unknown.length, 0);
  // Three passes plus the unattributed row.
  assert.equal(pose.children.length, 4);
  assert.ok(pose.selfMs >= 0, `negative self time: ${pose.selfMs}`);

  // What the children cover is the window MINUS the device-side gap: this frame
  // has a deliberate 0.25ms hole between pass 1 ending and pass 2 starting, and
  // no row claims it because nothing ran in it.
  //
  // So it stays in the parent's self time, along with upload and encode. That is
  // the correct behaviour of a UNION-based decomposition and not a leak: a gap
  // between children always belongs to the parent, and the alternative -- a row
  // asserting the GPU was busy when it was not -- would be a worse lie than an
  // unlabelled 0.25ms.
  const gapMs = 0.25;
  const covered = (resolvedAt - submittedAt) - gapMs;
  assert.ok(
    Math.abs(pose.selfMs - (pose.durationMs - covered)) < 1e-9,
    `self ${pose.selfMs} should be duration ${pose.durationMs} minus the covered ${covered}`,
  );
});

// ── §6: the DevTools mirror ───────────────────────────────────────────────
//
// The one part of the design that had no test, and it was wrong when one was
// finally written: GPU ids already carry a `gpu:` prefix, so naming the measure
// `~<clock>:<id>` emitted `~gpu:gpu:<stage>`.
//
// `performance.measure` is real per-record main-thread work INSIDE the window
// being measured, which is why it is the one thing the checkbox gates and why it
// is off on every load.

const measureNames = (): string[] => performance.getEntriesByType('measure').map((e) => e.name);

test('the mirror is off by default, and records nothing to User Timing', () => {
  profilerReset();
  profilerSetDevToolsMirror(false);
  spanEnd(appSpan('app.pose'));
  ingestLinkSpans({ sentAt: performance.timeOrigin + 1000, pulledAt: performance.timeOrigin + 1004, encodedAt: performance.timeOrigin + 1011 });
  assert.deepStrictEqual(measureNames(), [], 'nothing may reach User Timing with the mirror off');
});

test('mirrored names carry the anchored marker exactly once', () => {
  profilerReset();
  profilerSetDevToolsMirror(true);
  try {
    const parent = appSpan('app.pose');
    spin(2);
    ingestGpuFrame(fakeFrame(parent.start + 0.2, parent.start + 1.5), 'app.pose');
    spanEnd(parent);
    ingestLinkSpans({
      sentAt: performance.timeOrigin + 1000,
      pulledAt: performance.timeOrigin + 1004,
      encodedAt: performance.timeOrigin + 1011,
    });

    const names = measureNames();
    // A host span mirrors under its own id, unmarked.
    assert.ok(names.includes('app.pose'), `no app.pose measure in ${names.join(', ')}`);
    // A GPU pass mirrors as `~gpu:<stage>` -- ONE tilde, ONE `gpu:`, which is
    // what §6 specifies.
    assert.ok(names.includes('~gpu:a'), `expected ~gpu:a, got ${names.join(', ')}`);
    assert.ok(!names.some((n) => n.includes('gpu:gpu:')), 'the clock prefix was applied twice');
    // Peer spans the same way.
    assert.ok(names.includes('~link.pull'), `expected ~link.pull, got ${names.join(', ')}`);
    // Exactly one leading tilde on anything anchored, and none on host rows.
    for (const n of names) assert.doesNotMatch(n, /^~~/);
    assert.equal(names.filter((n) => n === 'app.pose').length, 1);
  } finally {
    profilerSetDevToolsMirror(false);
  }
});

test('profilerReset clears the User Timing buffer it filled', () => {
  profilerSetDevToolsMirror(true);
  try {
    spanEnd(appSpan('app.pose'));
    assert.ok(measureNames().length > 0);
    // The buffer is finite and would otherwise accumulate every record of every
    // run for the life of the tab.
    profilerReset();
    assert.deepStrictEqual(measureNames(), []);
  } finally {
    profilerSetDevToolsMirror(false);
  }
});

// ── Phase 6: the phone link ───────────────────────────────────────────────

test('phone stamps become peer spans on the host timeline', () => {
  profilerReset();
  // Epoch milliseconds, as the phone sends them: `performance.timeOrigin +
  // performance.now()` on ITS machine. Untranslated these are ~1.7e12 against a
  // store holding ~1e5, so a missing translation is not a subtle error.
  const sentAt = performance.timeOrigin + 1000;
  ingestLinkSpans({ sentAt, pulledAt: sentAt + 4, encodedAt: sentAt + 11 });

  const pull = getRecords().find((r) => r.id === 'link.pull')!;
  const enc = getRecords().find((r) => r.id === 'link.encode')!;
  assert.ok(pull && enc);
  assert.equal(pull.clock, 'peer');
  // Durations survive exactly: both endpoints are on the phone's clock, so the
  // cross-device skew cancels and these numbers are true whatever it is.
  assert.equal(pull.end - pull.start, 4);
  assert.equal(enc.end - enc.start, 7);
  // Translated onto THIS page's timeline -- the same scale performance.now()
  // produces, not epoch milliseconds.
  assert.equal(pull.start, 1000);
  assert.ok(pull.start < 1e9, 'an epoch stamp reached the store untranslated');
});

test('there is no transit span, because transit is not a duration', () => {
  profilerReset();
  const sentAt = performance.timeOrigin + 1000;
  ingestLinkSpans({ sentAt, pulledAt: sentAt + 4, encodedAt: sentAt + 11 });

  // The would-be third span is `receivedAt (DESKTOP) - encodedAt (PHONE)`, which
  // measures about -38ms on the real pair of machines. A negative span would
  // break containment and let a union of children exceed its parent, which is
  // the defect class the flat store exists to prevent. An omitted row beats a
  // confidently negative one.
  assert.equal(getRecords().filter((r) => r.id.startsWith('link.')).length, 2);
  assert.equal(getRecords().find((r) => r.id === 'link.transit'), undefined);
  for (const r of getRecords()) assert.ok(r.end >= r.start, `${r.id} runs backwards`);
});

test('peer spans join as roots and render with the anchored marker', () => {
  profilerReset();
  const sentAt = performance.timeOrigin + 1000;
  ingestLinkSpans({ sentAt, pulledAt: sentAt + 4, encodedAt: sentAt + 11 });

  const join = joinRecords(getRecords(), ALL_STAGES);
  // Declared in the table, so they are not "unknown" -- and roots, because they
  // happened on another machine before any desktop span existed to contain them.
  assert.equal(join.unknown.length, 0);
  assert.equal(join.orphans.length, 0);
  const text = formatSpanTree(join, ALL_STAGES);
  const row = text.split('\n').find((l) => l.includes('phone: pull video frame'))!;
  assert.ok(row, `no link.pull row in:\n${text}`);
  assert.match(row.trim(), /^~/, 'a peer row is anchored and must be marked');
});

// ── Phase 5: the renderer ─────────────────────────────────────────────────

test('repeats aggregate into one row, and anchored rows are marked', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(4);
  ingestGpuFrame(fakeFrame(parent.start + 1, parent.start + 3), 'app.pose');
  spanEnd(parent);

  const text = formatSpanTree(joinRecords(getRecords(), ALL_STAGES), ALL_STAGES);
  const lines = text.split('\n');

  // 'b' ran twice and both are leaves, so it collapses to one row carrying the
  // count -- otherwise a real frame prints 136 of these.
  const b = lines.find((l) => l.includes('gpu:b'));
  assert.ok(b, `no gpu:b row in:\n${text}`);
  assert.match(b, /n=2/);
  assert.match(b, /total=0\.50ms/);
  assert.match(b, /median=0\.250ms/);
  assert.equal(lines.filter((l) => l.includes('gpu:b')).length, 1, 'gpu:b must appear once');

  // 'a' ran once and keeps an ordinary row.
  assert.ok(lines.some((l) => l.includes('gpu:a') && !l.includes('n=')));

  // THE MARKER. Every gpu row is anchored, so every gpu row carries `~`; the
  // host rows must not, or the mark stops meaning anything.
  for (const l of lines.filter((x) => x.includes('gpu:'))) {
    assert.match(l.trim(), /^~/, `an anchored row must be marked: ${l}`);
  }
  const poseRow = lines.find((l) => l.includes('pose (submit'))!;
  assert.doesNotMatch(poseRow.trim(), /^~/, 'a host-measured row must NOT be marked');
  // The unattributed row is host-measured at both ends, so it is unmarked even
  // though it sits among the GPU rows and its start is derived from them.
  const unRow = lines.find((l) => l.includes(GPU_UNATTRIBUTED_ID))!;
  assert.doesNotMatch(unRow.trim(), /^~/);
});

test('repeated stage ids stay separate records and keep their pass index', () => {
  profilerReset();
  const parent = appSpan('app.pose');
  spin(4);
  ingestGpuFrame(fakeFrame(parent.start + 1, parent.start + 3), 'app.pose');
  spanEnd(parent);

  // 'b' ran twice. Both occurrences must survive as records -- aggregation is a
  // rendering decision, and folding them here would destroy the information.
  const bs = getRecords().filter((r) => r.id === 'gpu:b');
  assert.equal(bs.length, 2);
  assert.deepStrictEqual(bs.map((r) => r.attrs?.index), [1, 2]);
});
