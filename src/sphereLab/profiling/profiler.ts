// ── Flat interval recorder + a declared-structure join ───────────────────
//
// Records ALWAYS. That is deliberate: everything here used to sit behind an
// `enabled` flag, which meant any caller that wanted a duration it could rely
// on kept a parallel set of performance.now() marks in step with the profiler
// by hand. One store, always on, is what removes the second set -- and
// removing second sets is this module's whole job.
//
// The cost of always recording is one object and two clock reads per span,
// ~14 per reconstruction against a ~15ms budget. The expensive half was never
// the spans -- it was the User Timing mirror and (until it was deleted) a set
// of per-module GPU timestamp queries that cost a mapAsync each. Only the
// mirror is optional now, and `profilerSetDevToolsMirror` is the flag that
// gates it.
//
// ── ONE RECORD TYPE, AND THE CLOCK IS A FIELD ON IT ──
//
// A duration measured on the GPU, or on the phone across the network, is the
// same kind of fact as one measured here -- so it is the same record with a
// different `clock`, not a different system.
//
// What differs between clocks is NOT the units. By the time a Span is in this
// store its start/end are host-clock milliseconds no matter where they came
// from, because a foreign timestamp is translated ONCE, at the boundary where
// it enters. What differs is how much a reader may trust its ABSOLUTE
// POSITION: a GPU counter and performance.now() have no defined relationship,
// so a GPU span's length is exact and its placement is anchored to a lower
// bound. `clock` is what carries that caveat, and it is the only thing a
// consumer needs in order to know which of the two it is holding.
//
// Everything downstream of the store -- the join, the renderer, the DevTools
// mirror, a dev-bridge script -- sees one timeline and needs no special case.
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
// a display drain's interval can physically enclose a pose's span, and
// nesting-by-measurement would happily adopt it -- reparenting again, in a new
// costume. Declared parent AND measured containment cannot do that, because the
// pose does not name the drain as its parent.
//
// What survives from the old design: an individual record's OWN duration.
// `end - start` was always correct no matter where the span landed, which is
// why a caller holding a record can read its duration off the object directly.

export interface SpanAttrs { readonly [k: string]: string | number | boolean }

// Where a span's numbers CAME FROM, which is the only thing that varies once
// they are in the store. See the header: this is provenance, not units.
//
//   host -- performance.now() on this machine. Zero offset by definition.
//   gpu  -- a timestamp-query counter, translated at ingest. Length measured,
//           position anchored to the submit that carried it.
//   peer -- another device's clock (the phone's), on the shared epoch that
//           clock.ts establishes. Length measured, position good to NTP skew.
export type Clock = 'host' | 'gpu' | 'peer';

// One measured interval. No parent pointer and no children: this is data, and
// structure is imposed on it later by joinRecords.
export interface Span {
  readonly id: string;
  readonly start: number; // ms, ALWAYS host-clock -- translated before it lands here
  end: number; // 0 while still open
  readonly clock: Clock;
  // The parts of a span that VARY per occurrence, kept off the id so they
  // cannot fragment it. `fitPairOfPlanes (CPU)` and `(GPU)` used to be two
  // different labels that no report could aggregate, and every readback minted
  // a fresh label per byte count. The id is the stage; the backend and the
  // byte count are attributes of one run of it.
  readonly attrs: SpanAttrs | null;
  // The declared parent for THIS OCCURRENCE, overriding the table's `within`.
  // `undefined` means "use the table"; an explicit `null` means "a root".
  //
  // This exists for spans whose owner is a fact about the CALL rather than
  // about the stage, and it is not a loophole in "structure is declared": what
  // the caller passes is still a DECLARATION, made at the only place that knows
  // the answer, and the join treats it identically -- declared parent plus
  // measured containment. Inference is what stays banned, and it is unavailable
  // anyway now that there is no stack.
  //
  // The shape it is FOR: a GPU span belongs to the host span that submitted the
  // work, which is knowable at the injection point and nowhere else -- there is
  // no one parent to write in a static table. It currently has no callers,
  // because the GPU half is not built yet.
  //
  // Do NOT reach for this to avoid declaring a stage properly. A stage called
  // from two places with two different parents is two stages.
  readonly within?: string | null;
}

// What a stage DECLARES about itself, once, in a table next to its siblings.
export interface StageNode {
  readonly label: string;
  // The stage this one runs inside, or null for a root. A consumer's table may
  // OVERRIDE this for a stage it composes -- a library declares its entry point
  // a root because within the library it is one, and the app re-declares it as
  // running inside the app's own stage. Composition is the consumer's
  // knowledge, not the library's.
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
}

export type StageTable = Readonly<Record<string, StageNode>>;

let records: Span[] = [];

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
// a capture is an async operation that suspends at its GPU readback, so in the
// main-thread flame chart it appears as unrelated task slices with the render
// loop interleaved between them, and reading it as one operation is essentially
// impossible. A measure spans the whole thing INCLUDING its suspensions, so it
// reads as one labeled bar in Timings while the interleaved work stays visible
// underneath in the main track.
//
// It is off by default because performance.measure is a real per-record cost
// and the User Timing buffer is finite -- neither is worth paying when nobody
// has DevTools open.
let mirrorOn = false;

export function profilerSetDevToolsMirror(v: boolean): void { mirrorOn = v; }

export function profilerReset(): void {
  records = [];
  // The User Timing buffer is finite and would otherwise accumulate every
  // record of every run for the life of the tab. Clearing ALL measures rather
  // than ours by name is safe here specifically because this profiler is the
  // only producer of them in the app -- worth re-checking if that ever stops
  // being true.
  try { performance.clearMeasures(); } catch { /* diagnostics must not throw */ }
}

// Never returns null. Callers hold the returned record and read `end - start`
// off it directly -- that contract is why this cannot go back behind a flag
// without giving those callers a second source of numbers again.
//
// Everything opened here is `clock: 'host'` by construction, because
// performance.now() is what it reads. A span on another clock is not opened; it
// is INGESTED whole, already translated, by whoever owns the boundary.
export function spanStart(
  id: string, attrs: SpanAttrs | null = null, within?: string | null,
): Span {
  if (records.length >= MAX_RECORDS) trimClosed();
  const start = performance.now();
  const rec: Span = within === undefined
    ? { id, start, end: 0, clock: 'host', attrs }
    : { id, start, end: 0, clock: 'host', attrs, within };
  records.push(rec);
  return rec;
}

// The parent this record declares: its own override when it has one, the
// table's otherwise. One function so nothing can disagree about which wins.
function declaredParent(rec: Span, table: StageTable): string | null {
  return rec.within !== undefined ? rec.within : table[rec.id].within;
}

// A record's duration, tolerating one still open so a caller on an early-return
// path gets 0 rather than a negative number derived from `end === 0`.
function durationOf(rec: Span): number {
  return rec.end > 0 ? rec.end - rec.start : 0;
}

export function spanEnd(rec: Span | null): void {
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

/**
 * Append an already-finished span measured on ANOTHER CLOCK.
 *
 * `spanStart`/`spanEnd` cannot serve this: they read `performance.now()`, which
 * is the one thing a foreign span's endpoints are not. So a caller that owns a
 * clock boundary translates the endpoints to host-clock milliseconds itself and
 * hands over the finished record.
 *
 * **Translating is the caller's job and doing it BEFORE this call is the whole
 * contract.** Everything downstream of the store -- the join, the renderer, the
 * DevTools mirror -- assumes one timeline, and this function is what makes that
 * assumption safe by refusing to be the place where a second one could enter.
 */
export function spanIngest(span: Span): Span {
  if (records.length >= MAX_RECORDS) trimClosed();
  records.push(span);
  if (mirrorOn && span.end > 0) {
    try {
      // ── The caveat rides on the NAME, because a bar has nowhere else to put
      // it ──
      //
      // A mirrored GPU or peer span appears in DevTools' Timings track as an
      // ordinary bar, indistinguishable from a host-measured one. Its LENGTH is
      // measured and its POSITION is anchored (§3), and a reader who does not
      // know that will read a lower bound as a fact.
      //
      // So: a leading `~` and nothing else. The id already says what the span
      // is -- a GPU pass is `gpu:<stage>` -- so prefixing the clock name again
      // produced `~gpu:gpu:<stage>`, which §6 does not ask for and which reads
      // like a bug in the instrument rather than a caveat about it.
      performance.measure(mark(span.clock) + span.id, { start: span.start, end: span.end });
    } catch { /* diagnostics must not throw */ }
  }
  return span;
}

// The raw records. Serves both readers -- this module's own formatting and the
// dev-bridge scripts, which pull them back over eval as plain JSON rather than
// scraping a text format. There used to be two functions for those two callers
// (`getRoots` and `getFlamechartJSON`) returning the same tree; flat records are
// already JSON-shaped, so one suffices, and a consumer can run its own join
// against whatever table it cares about.
export function getRecords(): readonly Span[] { return records; }

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
  readonly rec: Span;
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
  orphans: Span[];
  // Records whose id is absent from the table entirely: another subsystem
  // recording into the same store. NOT corruption any more -- they simply do
  // not join, so nothing of theirs lands inside a stage that is being
  // decomposed. Worth reporting as context, not as a reason to void a report.
  unknown: Span[];
}

// Containment treats an OPEN parent as extending to +Infinity (it has not
// finished yet, so anything started inside it is inside it) and an open CHILD
// as a point at its start (it began inside, and where it ends is not yet
// known).
function containsRec(parent: Span, child: Span): boolean {
  const parentEnd = parent.end > 0 ? parent.end : Infinity;
  const childEnd = child.end > 0 ? child.end : child.start;
  return parent.start <= child.start && childEnd <= parentEnd;
}

function overlaps(a: Span, b: Span): boolean {
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

// ── THE TABLE-LESS PATH, and why it is not a hole in "structure is declared" ──
//
// A stage table names the ids one subsystem knows about ahead of time. GPU pass
// ids do not fit that: `src/pose2` owns them, there are ~40, they repeat, and
// mirroring its stage list in the app's table is precisely the drift this
// project keeps finding as a defect. The library is the authority on what its
// passes are called, and the app must not restate it.
//
// So an id absent from the table still joins IF it declares its parent for that
// occurrence. The declaration is per-call rather than per-stage, which is right
// here for the reason `Span.within` exists at all: a GPU span belongs to the
// host span that SUBMITTED it, which is a fact about the call.
//
// This was measured before it was written. The first version filtered on
// `table[r.id]` in this loop and pushed to `unknown` before ever reading
// `within`, so a GPU span could not join whatever it declared -- and its time
// stayed inside the submitter's SELF time, which is exactly the decomposition
// error the design forbids.
//
// What must NOT change: an id with no table entry AND no override is still a
// foreign subsystem's record and still stays out of the tree entirely.
function nodeFor(rec: Span, table: StageTable): StageNode | null {
  const declared = table[rec.id];
  if (declared) return declared;
  if (rec.within === undefined) return null;
  // The id is the label. There is nothing better and nothing to invent: a
  // synthesized label that dressed it up would be this file guessing at another
  // subsystem's naming.
  return { label: rec.id, within: rec.within };
}

export function joinRecords(recs: readonly Span[], table: StageTable): JoinResult {
  const unknown: Span[] = [];
  const orphans: Span[] = [];
  const known: Span[] = [];
  const byId = new Map<string, Span[]>();
  const declaredNode = new Map<Span, StageNode>();
  for (const r of recs) {
    const node = nodeFor(r, table);
    if (!node) { unknown.push(r); continue; }
    declaredNode.set(r, node);
    known.push(r);
    const list = byId.get(r.id);
    if (list) list.push(r); else byId.set(r.id, [r]);
  }

  const nodes = new Map<Span, TreeNode>();
  for (const r of known) {
    nodes.set(r, { rec: r, node: declaredNode.get(r)!, children: [], durationMs: durationOf(r), selfMs: 0 });
  }

  const roots: TreeNode[] = [];
  for (const r of known) {
    const within = declaredParent(r, table);
    const candidates = within === null ? undefined : byId.get(within);
    if (within === null || !candidates || candidates.length === 0) {
      roots.push(nodes.get(r)!);
      continue;
    }
    // The INNERMOST containing occurrence, i.e. the latest-starting one, so a
    // stage that legitimately runs twice inside two occurrences of its parent
    // lands in the right one.
    let best: Span | null = null;
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
    // ALSO straight off a mode switch with no tail above it. Both are roots.
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

// ── Rendering ────────────────────────────────────────────────────────────
//
// Nested indented text, duration + self time + percent-of-parent per line.
// There is no "TREE IS STRUCTURALLY INVALID" banner any more and nothing to
// put in one: percentages cannot exceed 100 because a child is only attached
// when its interval is contained in its parent's, and self times cannot go
// negative because children are subtracted as a union. What CAN still be
// reported is a record that did not join, and those get their own sections.
// A row's clock marker. `~` means the LENGTH is measured and the POSITION is
// anchored -- see the design's §3. A reader has to be told, and the alternative
// (a legend, or nothing) is how a lower bound gets read as a measurement.
function mark(clock: Clock): string {
  return clock === 'host' ? '' : '~';
}

function median(xs: readonly number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const mid = s.length >> 1;
  return s.length % 2 ? s[mid]! : (s[mid - 1]! + s[mid]!) / 2;
}

export function formatSpanTree(join: JoinResult, table: StageTable): string {
  const lines: string[] = [];
  const walk = (n: TreeNode, depth: number, parentMs: number) => {
    const pct = parentMs > 0 ? ((n.durationMs / parentMs) * 100).toFixed(1) : '100.0';
    const attrs = n.rec.attrs
      ? ' ' + Object.entries(n.rec.attrs).map(([k, v]) => `${k}=${v}`).join(' ')
      : '';
    lines.push(`${'  '.repeat(depth)}${mark(n.rec.clock)}${n.node.label}${attrs}`
      + ` -- ${n.durationMs.toFixed(2)}ms`
      + ` (${pct}%, self ${n.selfMs.toFixed(2)}ms)${n.node.sync ? ' SYNC' : ''}`);
    walkChildren(n.children, depth + 1, n.durationMs);
  };

  // ── Repeats aggregate HERE, not in the store ──
  //
  // grow's hook/compress/gate run once per convergence round -- up to 32 each,
  // so a frame carries ~136 GPU rows. Per-occurrence rows are unreadable, and
  // the answer is a rendering decision rather than a recording one: the raw
  // records stay in the store for anything that wants them.
  //
  // Only LEAVES are aggregated. A repeated stage that has children of its own
  // would have its subtree hidden by the collapse, and a shorter report is not
  // worth losing a level of decomposition.
  const walkChildren = (kids: readonly TreeNode[], depth: number, parentMs: number) => {
    const groups = new Map<string, TreeNode[]>();
    for (const k of kids) {
      const g = groups.get(k.rec.id);
      if (g) g.push(k); else groups.set(k.rec.id, [k]);
    }
    // First occurrence order, so the report still reads in execution order.
    for (const g of groups.values()) {
      if (g.length === 1 || g.some((k) => k.children.length > 0)) {
        for (const k of g) walk(k, depth, parentMs);
        continue;
      }
      const total = g.reduce((a, k) => a + k.durationMs, 0);
      const pct = parentMs > 0 ? ((total / parentMs) * 100).toFixed(1) : '100.0';
      lines.push(`${'  '.repeat(depth)}${mark(g[0]!.rec.clock)}${g[0]!.node.label}`
        + ` -- n=${g.length} total=${total.toFixed(2)}ms (${pct}%)`
        + ` median=${median(g.map((k) => k.durationMs)).toFixed(3)}ms`);
    }
  };

  for (const r of join.roots) walk(r, 0, r.durationMs);
  if (join.orphans.length) {
    lines.push(`!! ${join.orphans.length} record(s) ran OUTSIDE the stage that declares them:`);
    for (const o of join.orphans.slice(0, 8)) {
      const decl = o.within !== undefined ? `${o.within} (per-call)` : `${table[o.id]?.within}`;
      lines.push(`   ${o.id} [${o.start.toFixed(1)}, ${o.end.toFixed(1)}] declares within=${decl}`);
    }
  }
  if (join.unknown.length) {
    const ids = [...new Set(join.unknown.map((u) => u.id))];
    lines.push(`note: ${join.unknown.length} record(s) from outside this table (${ids.join(', ')}).`);
    lines.push(`      They did not join, so nothing of theirs is inside the rows above.`);
  }
  return lines.join('\n');
}
