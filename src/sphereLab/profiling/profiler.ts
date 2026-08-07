// ── Flat interval recorder + a declared-structure join ───────────────────
//
// Records ALWAYS. That is a deliberate change from the old design, where
// everything here sat behind an `enabled` flag, and it is what lets
// pose/poseCompute.ts report its per-stage timings straight off its own
// record objects instead of keeping a parallel set of performance.now() marks
// in step with them by hand.
//
// The cost of always recording is one object and two performance.now() calls
// per span, ~14 per reconstruction against a ~15ms budget. The expensive half
// was never the spans -- it was the User Timing mirror and (until it was
// deleted) a set of per-module GPU timestamp queries that cost a mapAsync each.
// Only the mirror is optional now, and `profilerSetDevToolsMirror` is the flag
// that gates it.
//
// ── RECORDING AND STRUCTURE ARE SEPARATE, and that is the whole design ──
//
// This module used to keep ONE module-level span STACK: `spanStart` looked at
// the top of it to decide a span's parent. That made call nesting the source of
// structure, and it had two failure modes that were reported rather than fixed:
//
//   1. A second concurrent operation got REPARENTED. The pose path suspends at
//      every readback; animate() runs a preview projectBins continuously and
//      suspends on its own. A reconstruction starting during that suspension
//      landed INSIDE it, filing one subsystem's cost under another. A
//      `checkNesting` walk and a "TREE IS STRUCTURALLY INVALID" banner existed
//      to tell the reader the output was lying.
//   2. Genuinely CONCURRENT work was recorded as parent/child. Two readbacks
//      awaited in one Promise.all overlap in time; a stack has no way to say
//      so, and the subtraction `parent - sum(children)` then produced NEGATIVE
//      self times that the harness had to detect and suppress.
//
// Both are gone, because there is no stack. `spanStart` appends a flat record
// and nothing else. Structure is a JOIN performed afterwards, against a table
// the stages DECLARE:
//
//   - declared: each stage's `within` (its containment parent) and `sync`.
//   - measured: the intervals.
//   - join rule: a record for stage S attaches to the record of its DECLARED
//     parent whose interval CONTAINS it.
//
// THE `within` HALF IS NOT OPTIONAL, and it is worth saying why, because
// "derive containment from interval nesting alone" looks equivalent and is not:
// a display drain's interval can physically enclose a pose's `pose.votes`, and
// nesting-by-measurement would happily adopt it -- reparenting again, in a new
// costume. Declared parent AND measured containment cannot do that, because
// `pose.votes` does not name the drain as its parent.
//
// What survives from the old design: an individual record's OWN duration.
// `end - start` was always correct no matter where the span landed, which is
// why poseCompute reads its stage timings from the records it holds by
// reference. That is unchanged -- it just no longer needs the caveat.

export interface SpanAttrs { readonly [k: string]: string | number | boolean }

// One measured interval. No parent pointer and no children: this is data, and
// structure is imposed on it later by joinRecords.
export interface StageRecord {
  readonly id: string;
  readonly start: number;
  end: number; // 0 while still open
  // The parts of a span that VARY per occurrence, kept off the id so they
  // cannot fragment it. `fitPairOfPlanes (CPU)` and `(GPU)` used to be two
  // different labels that no report could aggregate, and every readback minted
  // a fresh label per byte count. The id is the stage; the backend and the
  // byte count are attributes of one run of it.
  readonly attrs: SpanAttrs | null;
}

// What a stage DECLARES about itself, once, in a table next to its siblings.
export interface StageNode {
  readonly label: string;
  // The stage this one runs inside, or null for a root. A consumer's table may
  // OVERRIDE this for a stage it composes -- the library declares `pose.run` a
  // root because within the library it is one, and the app re-declares it as
  // running inside `app.reconstruct`. Composition is the consumer's knowledge,
  // not the library's.
  readonly within: string | null;
  // True when the stage provably encloses no `await`.
  //
  // This matters because a span around an await measures WALL TIME, so it
  // absorbs whatever the event loop did while suspended -- the fence being
  // waited on (wanted), but also a rAF render that landed mid-fence (not
  // wanted). No interval can separate those two after the fact. A stage with no
  // await inside cannot be interleaved into at all on a single-threaded event
  // loop, so its duration is real executing JS (or a GC pause), and only those
  // rows support a claim about host CPU cost.
  //
  // Declared here rather than passed at the call site, which is where it used
  // to live. The argument for the call site was that a NAME cannot know whether
  // a branch awaits -- true when names were free-form strings written inline.
  // With a closed id union the id IS the stage, one id is one call site, and a
  // stage whose two branches genuinely differ in this should be two ids.
  readonly sync?: boolean;
  // Data-dependency edges: the stages whose output this one consumes. Distinct
  // from `within`, which is containment. Consumed by the critical-path
  // analysis; unset means "no declared inputs", not "no inputs".
  readonly inputs?: readonly string[];
}

export type StageTable = Readonly<Record<string, StageNode>>;

let records: StageRecord[] = [];

// ── Bounding the list, now that recording never stops ──
//
// With an `enabled` flag the record set was implicitly bounded: it only grew
// while someone was watching. Recording always means this would otherwise grow
// for the life of the tab.
//
// Only CLOSED records are dropped, and only from the front. Dropping an open
// one would not corrupt anything (spanEnd stamps the object through the
// caller's own reference, not through this list) but it would silently omit a
// long-running operation from its own report, which is the one it was most
// likely opened to measure.
//
// A trimmed-away parent leaves its children ORPHANED rather than misfiled --
// joinRecords reports that rather than guessing -- and 4096 is large enough
// against the ~30 records one reconstruction emits that it only ever reaches
// records nobody is still reading.
const MAX_RECORDS = 4096;

// ── The one flag: whether records are mirrored into DevTools ─────────────
//
// The records above are only visible through this module's own formatting.
// Mirroring each as a performance.measure ALSO puts it in Chrome DevTools'
// "Timings" track, which is worth more than it costs for one specific reason:
// the pose pipeline is an async function that suspends at roughly a dozen GPU
// readbacks, so in the main-thread flame chart it appears as a dozen unrelated
// task slices with the render loop interleaved between them, and reading it as
// one operation is essentially impossible. A measure spans the whole thing
// INCLUDING its suspensions, so the pipeline reads as one labeled bar in
// Timings while the interleaved work stays visible underneath in the main
// track.
//
// It is off by default because performance.measure is a real per-record cost
// and the User Timing buffer is finite -- neither is worth paying when nobody
// has DevTools open.
let mirrorOn = false;

export function profilerSetDevToolsMirror(v: boolean): void { mirrorOn = v; }
export function profilerDevToolsMirror(): boolean { return mirrorOn; }

export function profilerReset(): void {
  records = [];
  // The User Timing buffer is finite and would otherwise accumulate every
  // record of every run for the life of the tab. Clearing ALL measures rather
  // than ours by name is safe here specifically because this profiler is the
  // only producer of them in the app -- worth re-checking if that ever stops
  // being true.
  try { performance.clearMeasures(); } catch { /* diagnostics must not throw */ }
}

// Never returns null. Callers hold the returned record and read its duration
// directly (see spanDurationMs) -- that is the contract poseCompute's stage
// timings rest on, and it is why this cannot go back behind a flag without
// giving those callers a second source of numbers again.
export function spanStart(id: string, attrs: SpanAttrs | null = null): StageRecord {
  if (records.length >= MAX_RECORDS) trimClosed();
  const rec: StageRecord = { id, start: performance.now(), end: 0, attrs };
  records.push(rec);
  return rec;
}

export function spanEnd(rec: StageRecord | null): void {
  if (!rec) return;
  rec.end = performance.now();
  if (!mirrorOn) return;
  try {
    // The object form (explicit start/end) rather than mark pairs: the
    // timestamps are already recorded, marks would double the entries for
    // nothing, and concurrent same-named marks would need disambiguating.
    performance.measure(rec.id, { start: rec.start, end: rec.end });
  } catch {
    // Never let instrumentation break the thing it is instrumenting: the
    // object form is not universally supported, and a UA may reject timestamps
    // it considers out of range.
  }
}

// The duration of a record the caller holds. Tolerates null and an unclosed
// record so a caller on an early-return path gets 0 rather than a negative
// number derived from `end === 0`.
export function spanDurationMs(rec: StageRecord | null): number {
  return rec && rec.end > 0 ? rec.end - rec.start : 0;
}

// The raw records. Serves both readers -- this module's own formatting and the
// dev-bridge scripts, which pull them back over eval as plain JSON rather than
// scraping a text format. There used to be two functions for those two callers
// (`getRoots` and `getFlamechartJSON`) returning the same tree; flat records are
// already JSON-shaped, so one suffices, and a consumer can run its own join
// against whatever table it cares about.
export function getRecords(): readonly StageRecord[] { return records; }

function trimClosed(): void {
  const keep = records.filter((r) => r.end === 0);
  const closed = records.filter((r) => r.end > 0);
  // Half the cap, so trimming is amortized rather than happening on every
  // subsequent spanStart once the cap is first reached.
  records = [...closed.slice(Math.max(0, closed.length - MAX_RECORDS / 2)), ...keep]
    .sort((a, b) => a.start - b.start);
}

// ── THE JOIN: declared structure, measured intervals ─────────────────────

export interface TreeNode {
  readonly rec: StageRecord;
  readonly node: StageNode;
  readonly children: TreeNode[];
  readonly durationMs: number;
  // This record's own duration minus the time covered by its children -- where
  // "covered" is the UNION of their intervals, not the SUM of their durations.
  //
  // The union is the entire correctness argument. Two children awaited
  // concurrently in one Promise.all OVERLAP; summing them double-counts the
  // overlap and can exceed the parent, which is how the old tree produced
  // negative self times that a reader had to be warned about. A union of
  // intervals contained in the parent cannot exceed the parent, so this is
  // non-negative by construction rather than by check.
  readonly selfMs: number;
}

export interface JoinResult {
  roots: TreeNode[];
  // Records whose declared parent DID run but whose interval no containing
  // occurrence of it covers. This is a real anomaly -- a stage ran outside the
  // stage that declares it -- and it is reported rather than resolved by
  // guessing, which is the behaviour the old stack could not offer.
  orphans: StageRecord[];
  // Records whose id is absent from the table entirely: another subsystem
  // recording into the same store. NOT corruption any more -- they simply do
  // not join, so nothing of theirs lands inside a stage that is being
  // decomposed. Worth reporting as context, not as a reason to void a report.
  unknown: StageRecord[];
}

// Containment treats an OPEN parent as extending to +Infinity (it has not
// finished yet, so anything started inside it is inside it) and an open CHILD
// as a point at its start (it began inside, and where it ends is not yet
// known).
function containsRec(parent: StageRecord, child: StageRecord): boolean {
  const parentEnd = parent.end > 0 ? parent.end : Infinity;
  const childEnd = child.end > 0 ? child.end : child.start;
  return parent.start <= child.start && childEnd <= parentEnd;
}

function overlaps(a: StageRecord, b: StageRecord): boolean {
  const aEnd = a.end > 0 ? a.end : Infinity;
  const bEnd = b.end > 0 ? b.end : b.start;
  return a.start < bEnd && b.start < aEnd;
}

// Total time covered by a set of intervals, merging overlaps. See TreeNode.selfMs.
function unionMs(kids: readonly TreeNode[]): number {
  const spans = kids
    .map((k) => [k.rec.start, k.rec.end > 0 ? k.rec.end : k.rec.start] as const)
    .sort((a, b) => a[0] - b[0]);
  let total = 0, curStart = 0, curEnd = -Infinity;
  for (const [s, e] of spans) {
    if (s > curEnd) {
      if (curEnd > curStart) total += curEnd - curStart;
      curStart = s; curEnd = e;
    } else if (e > curEnd) curEnd = e;
  }
  if (curEnd > curStart) total += curEnd - curStart;
  return total;
}

export function joinRecords(recs: readonly StageRecord[], table: StageTable): JoinResult {
  const unknown: StageRecord[] = [];
  const orphans: StageRecord[] = [];
  const known: StageRecord[] = [];
  const byId = new Map<string, StageRecord[]>();
  for (const r of recs) {
    if (!table[r.id]) { unknown.push(r); continue; }
    known.push(r);
    const list = byId.get(r.id);
    if (list) list.push(r); else byId.set(r.id, [r]);
  }

  const nodes = new Map<StageRecord, TreeNode>();
  for (const r of known) {
    nodes.set(r, { rec: r, node: table[r.id], children: [], durationMs: spanDurationMs(r), selfMs: 0 });
  }

  const roots: TreeNode[] = [];
  for (const r of known) {
    const within = table[r.id].within;
    const candidates = within === null ? undefined : byId.get(within);
    if (within === null || !candidates || candidates.length === 0) {
      roots.push(nodes.get(r)!);
      continue;
    }
    // The INNERMOST containing occurrence, i.e. the latest-starting one, so a
    // stage that legitimately runs twice inside two occurrences of its parent
    // lands in the right one.
    let best: StageRecord | null = null;
    let straddled = false;
    for (const c of candidates) {
      if (c === r) continue;
      if (containsRec(c, r)) { if (!best || c.start > best.start) best = c; }
      else if (overlaps(c, r)) straddled = true;
    }
    if (best) { nodes.get(best)!.children.push(nodes.get(r)!); continue; }
    // ── Root vs. orphan, and the distinction is what keeps orphans MEANING
    // something ──
    //
    // A declared parent that ran but never overlapped this record at all is
    // not an anomaly: the stage was simply called from somewhere else this
    // time. `project.bins` runs inside the display tail's `app.project` and
    // ALSO straight off a mode switch with no tail above it; the harness runs
    // `pose.run` with no `app.reconstruct` above it at all. Both are roots.
    //
    // A parent that OVERLAPS without containing is the real defect -- a stage
    // straddling its own parent's boundary means the declaration is wrong or
    // the spans are unbalanced. That is the only case worth a warning, and
    // keeping it narrow is what stops the warning from being ignored.
    if (straddled) orphans.push(r); else roots.push(nodes.get(r)!);
  }

  for (const n of nodes.values()) {
    n.children.sort((a, b) => a.rec.start - b.rec.start);
    // Written through the readonly view deliberately: selfMs cannot be computed
    // until the children are attached, and a second pass building fresh objects
    // would have to rebuild the whole tree to keep the references consistent.
    (n as { selfMs: number }).selfMs = n.durationMs - unionMs(n.children);
  }
  roots.sort((a, b) => a.rec.start - b.rec.start);
  return { roots, orphans, unknown };
}

// ── A TIMING SESSION: one record set per rep, and the caller's own preserved ──
//
// A harness that measures N reps needs a fresh record set per rep and must
// leave the page exactly as it found it. It used to do that by hand, with four
// exported primitives and a try/finally of its own -- which meant the harness
// had to know that the store is a module-level variable and that the DevTools
// mirror is a separate flag with its own save/restore. That is the module's
// internals, spelled out in a caller.
//
// This is that whole protocol as one object. `records` and `mirrorOn` are
// private to this file.
//
// profilerReset() is the wrong tool for per-rep clearing and always was: it
// calls performance.clearMeasures(), so a caller resetting per iteration
// deletes earlier captures out of a DevTools recording in progress, and would
// also destroy a record set the user had already taken and not yet printed.
// Nothing here clears anything -- the caller's records are swapped aside and
// put back, so both the User Timing track and whatever they had recorded
// survive.
export interface ProfilerSession {
  // Whether the DevTools mirror was on when the session began. Worth reporting
  // in a result: a caller's PREVIOUS numbers were taken with a
  // performance.measure per record and are not comparable to the ones from
  // inside a session, which forces it off.
  readonly mirrorWasOn: boolean;
  // This rep's records, and a fresh set started. Call it after each rep.
  takeRepRecords(): StageRecord[];
  // Restore the caller's records and their mirror setting. Idempotent, so it is
  // safe in a finally that may also be reached after an early return.
  end(): void;
}

export function profilerBeginSession(): ProfilerSession {
  const mirrorWasOn = mirrorOn;
  // Swapped aside, not cleared: whoever ran a profiled capture before calling
  // this still has their records to print afterwards.
  const callerRecords = takeRecords();
  // A performance.measure per record across ~25 reps is main-thread work inside
  // the window being timed, and nobody is watching a DevTools track during a
  // harness run. Recording itself is not optional and is NOT turned off -- it
  // is where the stage timings come from.
  mirrorOn = false;
  let ended = false;
  return {
    mirrorWasOn,
    takeRepRecords: () => takeRecords(),
    end() {
      if (ended) return;
      ended = true;
      mirrorOn = mirrorWasOn;
      records = callerRecords;
    },
  };
}

function takeRecords(): StageRecord[] {
  const taken = records;
  records = [];
  return taken;
}

// ── Rendering ────────────────────────────────────────────────────────────
//
// Nested indented text, duration + self time + percent-of-parent per line.
// There is no "TREE IS STRUCTURALLY INVALID" banner any more and nothing to
// put in one: percentages cannot exceed 100 because a child is only attached
// when its interval is contained in its parent's, and self times cannot go
// negative because children are subtracted as a union. What CAN still be
// reported is a record that did not join, and those get their own sections.
export function formatSpanTree(join: JoinResult, table: StageTable): string {
  const lines: string[] = [];
  const walk = (n: TreeNode, depth: number, parentMs: number) => {
    const pct = parentMs > 0 ? ((n.durationMs / parentMs) * 100).toFixed(1) : '100.0';
    const attrs = n.rec.attrs
      ? ' ' + Object.entries(n.rec.attrs).map(([k, v]) => `${k}=${v}`).join(' ')
      : '';
    lines.push(`${'  '.repeat(depth)}${n.node.label}${attrs} -- ${n.durationMs.toFixed(2)}ms`
      + ` (${pct}%, self ${n.selfMs.toFixed(2)}ms)${n.node.sync ? ' SYNC' : ''}`);
    for (const c of n.children) walk(c, depth + 1, n.durationMs);
  };
  for (const r of join.roots) walk(r, 0, r.durationMs);
  if (join.orphans.length) {
    lines.push(`!! ${join.orphans.length} record(s) ran OUTSIDE the stage that declares them:`);
    for (const o of join.orphans.slice(0, 8)) {
      lines.push(`   ${o.id} [${o.start.toFixed(1)}, ${o.end.toFixed(1)}] declares within=${table[o.id]?.within}`);
    }
  }
  if (join.unknown.length) {
    const ids = [...new Set(join.unknown.map((u) => u.id))];
    lines.push(`note: ${join.unknown.length} record(s) from outside this table (${ids.join(', ')}).`);
    lines.push(`      They did not join, so nothing of theirs is inside the rows above.`);
  }
  return lines.join('\n');
}
