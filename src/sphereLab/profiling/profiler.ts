// ── Nested-span CPU profiler ("poor-man's flamechart") ───────────────────
//
// Spans record ALWAYS. That is a deliberate change from the old design, where
// everything here sat behind an `enabled` flag, and it is what lets
// pipeline/poseCompute.ts report its per-stage timings straight off its own
// span objects instead of keeping a parallel set of performance.now() marks in
// step with them by hand.
//
// The cost of always recording is one object and two performance.now() calls
// per span, ~14 per reconstruction against a ~15ms budget. The expensive half
// was never the spans -- it was the User Timing mirror and (until it was
// deleted) a set of per-module GPU timestamp queries that cost a mapAsync each.
// Only the mirror is optional now, and `profilerSetDevToolsMirror` is the flag
// that gates it.
//
// ── The single shared stack, and what it costs ──
//
// The span stack is one module-level array, not per-call-isolated, so this
// assumes one profiled operation at a time. A second concurrent operation does
// not corrupt anything (spanEnd splices by identity) but its spans get
// REPARENTED under whatever the first has open, which files one subsystem's
// cost under another.
//
// That is not hypothetical and not rare: animate() runs a preview projectBins
// continuously, it suspends on a readback, and a reconstruction starting during
// that suspension lands inside it. So the TREE STRUCTURE is unreliable during
// ordinary app use, and formatFlamechart calls that out (see checkNesting).
//
// What survives it: an individual span's OWN duration. `end - start` is correct
// no matter where the span landed in the tree. That is why poseCompute reads
// its stage times from the span objects it holds by reference rather than by
// searching the tree, and why those numbers are trustworthy while a self-time
// breakdown computed from the same tree may not be.
//
// The deferred display tail (pipeline/axesReconstruction.ts's drainVisuals) is
// async and outlives its own caller, so it is held to the same exclusion as a
// capture -- captures and drains refuse to start underneath each other.

export interface ProfileSpan {
  name: string;
  start: number;
  end: number;
  children: ProfileSpan[];
  // True when the span provably encloses no `await`.
  //
  // This matters because a span around an await measures WALL TIME, so it
  // absorbs whatever the event loop did while suspended -- the fence being
  // waited on (wanted), but also a rAF render that landed mid-fence (not
  // wanted). No span can separate those two after the fact. A span with no
  // await inside cannot be interleaved into at all on a single-threaded event
  // loop, so its duration is real executing JS (or a GC pause), and only those
  // rows support a claim about host CPU cost.
  //
  // Declared HERE, at the span, rather than by a list of names kept in the
  // harness that reads them. That list lived in reconstructionTiming.ts, named
  // spans defined in four other modules, and had no way to be right about
  // `votes (segments)` -- which contains an await on one branch and not the
  // other. A call site knows; a name does not.
  sync: boolean;
}

let stack: ProfileSpan[] = [];
let roots: ProfileSpan[] = [];

// ── Bounding the tree, now that recording never stops ──
//
// With an `enabled` flag the tree was implicitly bounded: it only grew while
// someone was watching. Recording always means `roots` would otherwise grow for
// the life of the tab, holding every child span of every capture alive with it.
//
// Trimming from the FRONT at root-start time is safe: a root is only dropped
// when a new one begins, and the only root that can still be open is the last.
// 256 is far more than any reader wants (formatFlamechart is read one operation
// at a time) and small enough that the retained set is a few thousand objects.
const MAX_ROOTS = 256;

// ── The one flag: whether spans are mirrored into DevTools ───────────────
//
// The span tree above is only visible through this module's own
// formatFlamechart/getFlamechartJSON. Mirroring each span as a
// performance.measure ALSO puts it in Chrome DevTools' "Timings" track, which
// is worth more than it costs for one specific reason: the pose pipeline is an
// async function that suspends at roughly a dozen GPU readbacks, so in the
// main-thread flame chart it appears as a dozen unrelated task slices with the
// render loop interleaved between them, and reading it as one operation is
// essentially impossible. A measure spans the whole thing INCLUDING its
// suspensions, so the pipeline reads as one labeled bar in Timings while the
// interleaved work stays visible underneath in the main track.
//
// It is off by default because performance.measure is a real per-span cost and
// the User Timing buffer is finite -- neither is worth paying when nobody has
// DevTools open.
let mirrorOn = false;

export function profilerSetDevToolsMirror(v: boolean): void { mirrorOn = v; }
export function profilerDevToolsMirror(): boolean { return mirrorOn; }

export function profilerReset(): void {
  stack = [];
  roots = [];
  // The User Timing buffer is finite and would otherwise accumulate every span
  // of every run for the life of the tab. Clearing ALL measures rather than
  // ours by name is safe here specifically because this profiler is the only
  // producer of them in the app -- worth re-checking if that ever stops being
  // true.
  try { performance.clearMeasures(); } catch { /* diagnostics must not throw */ }
}

// Never returns null. Callers hold the returned span and read its duration
// directly (see spanDurationMs) -- that is the contract poseCompute's stage
// timings rest on, and it is why this cannot go back behind a flag without
// giving those callers a second source of numbers again.
// `sync` defaults to false, i.e. "assume this may await". That is the safe
// default: a span wrongly marked sync would put fence time into a total labelled
// host CPU, which is the direction that manufactures a finding. Wrongly marked
// async only leaves a real number out of that total.
export function spanStart(name: string, sync = false): ProfileSpan {
  const span: ProfileSpan = { name, start: performance.now(), end: 0, children: [], sync };
  const parent = stack[stack.length - 1];
  if (parent) {
    parent.children.push(span);
  } else {
    if (roots.length >= MAX_ROOTS) roots.splice(0, roots.length - MAX_ROOTS + 1);
    roots.push(span);
  }
  stack.push(span);
  return span;
}

export function spanEnd(span: ProfileSpan | null): void {
  if (!span) return;
  span.end = performance.now();
  const idx = stack.lastIndexOf(span);
  if (idx >= 0) stack.splice(idx, 1);
  if (!mirrorOn) return;
  try {
    // The object form (explicit start/end) rather than mark pairs: the
    // timestamps are already recorded, marks would double the entries for
    // nothing, and concurrent same-named marks would need disambiguating.
    performance.measure(span.name, { start: span.start, end: span.end });
  } catch {
    // Never let instrumentation break the thing it is instrumenting: the
    // object form is not universally supported, and a UA may reject timestamps
    // it considers out of range.
  }
}

// The duration of a span the caller holds. Tolerates null and an unclosed span
// so a caller on an early-return path gets 0 rather than a negative number
// derived from `end === 0`.
export function spanDurationMs(span: ProfileSpan | null): number {
  return span && span.end > 0 ? span.end - span.start : 0;
}

export function getRoots(): ProfileSpan[] { return roots; }

// ── Swapping the tree aside, for a caller that records many trees in a row ──
//
// profilerReset() is the wrong tool for that: it calls
// performance.clearMeasures(), so a caller resetting per iteration deletes
// earlier captures out of a DevTools recording in progress, and would also
// destroy a tree the user had already recorded and not yet printed.
//
// pipelineGPU/reconstructionTiming.ts needs one tree per rep across ~25 reps,
// so it takes the tree after each one and puts the user's own back when it is
// done. Nothing is cleared, so both the User Timing track and whatever the user
// had recorded survive a timing run.
export function profilerTakeRoots(): ProfileSpan[] {
  const taken = roots;
  roots = [];
  // The stack goes with it. A tree is only taken between operations, so a
  // non-empty stack here means a span was left open by an early return -- and
  // carrying that stale frame into the next rep would reparent the next rep's
  // spans underneath it, silently filing one rep's cost inside another.
  stack = [];
  return taken;
}

export function profilerPutRoots(rs: ProfileSpan[]): void { roots = rs; stack = []; }

// ── Structural validation, because the tree lies in ordinary use ─────────
//
// A child cannot take longer than its parent, and a child cannot start before
// its parent does. Both happen here anyway, via the reparenting described in
// the header, and the resulting flamechart looks entirely plausible until you
// notice a percentage over 100 -- a 102ms axesReconstruction nested inside a
// 54ms readback was the case that prompted this.
//
// Reporting it is the point. An instrument whose output can be structurally
// wrong has to say so itself rather than leaving a reader to spot it, which is
// the same reason the harness asserts on negative self times instead of
// rendering them.
interface NestingViolation { parent: string; child: string; parentMs: number; childMs: number }

export function checkNesting(spans: readonly ProfileSpan[] = roots): NestingViolation[] {
  const out: NestingViolation[] = [];
  const walk = (s: ProfileSpan) => {
    for (const c of s.children) {
      // 0.5ms of slack: spanEnd stamps the parent after the child, so exact
      // equality is never expected, and sub-tick jitter should not be reported
      // as corruption. A real reparenting is off by whole milliseconds.
      const pm = spanDurationMs(s), cm = spanDurationMs(c);
      if (cm > pm + 0.5 || c.start < s.start - 0.5) {
        out.push({ parent: s.name, child: c.name, parentMs: pm, childMs: cm });
      }
      walk(c);
    }
  };
  for (const r of spans) walk(r);
  return out;
}

// Nested indented text, duration + percent-of-parent per line -- sorted by
// start time within each level so the output reads top-to-bottom the way
// the work actually happened.
export function formatFlamechart(): string {
  const lines: string[] = [];
  const violations = checkNesting();
  if (violations.length) {
    lines.push(`!! TREE IS STRUCTURALLY INVALID -- ${violations.length} child span(s) longer than their parent.`);
    lines.push(`   Another operation recorded into the same span stack and its cost was filed under this one.`);
    lines.push(`   Percentages over 100% below are that, not a measurement. See profiler.ts's header.`);
    for (const v of violations.slice(0, 5)) {
      lines.push(`   ${v.child} (${v.childMs.toFixed(2)}ms) inside ${v.parent} (${v.parentMs.toFixed(2)}ms)`);
    }
  }
  const walk = (span: ProfileSpan, depth: number, parentDur: number) => {
    const dur = spanDurationMs(span);
    const pct = parentDur > 0 ? ((dur / parentDur) * 100).toFixed(1) : '100.0';
    lines.push(`${'  '.repeat(depth)}${span.name} -- ${dur.toFixed(2)}ms (${pct}%)`);
    const sortedChildren = [...span.children].sort((a, b) => a.start - b.start);
    for (const c of sortedChildren) walk(c, depth + 1, dur);
  };
  for (const r of [...roots].sort((a, b) => a.start - b.start)) walk(r, 0, spanDurationMs(r));
  return lines.join('\n');
}

// Same tree, as plain JSON -- for the dev-bridge test script to pull structured
// data back over eval instead of scraping the text format.
export function getFlamechartJSON(): ProfileSpan[] {
  return roots;
}
