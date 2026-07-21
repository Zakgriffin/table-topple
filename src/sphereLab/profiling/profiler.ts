// ── Nested-span CPU/GPU profiler ("poor-man's flamechart") ───────────────
//
// Diagnostic-only, gated behind profilerSetEnabled(true) so ordinary runs
// pay zero overhead. Assumes exactly one profiled call is in flight at a
// time -- the span stack is a single shared module-level array, not
// per-call-isolated -- which matches runAxesReconstruction's own
// camera.axesCapturing guard against overlapping captures.

export interface ProfileSpan {
  name: string;
  start: number;
  end: number;
  kind: 'cpu' | 'gpu';
  children: ProfileSpan[];
}

let stack: ProfileSpan[] = [];
let roots: ProfileSpan[] = [];
let enabled = false;

export function profilerSetEnabled(v: boolean): void { enabled = v; }
export function profilerEnabled(): boolean { return enabled; }

export function profilerReset(): void { stack = []; roots = []; }

export function spanStart(name: string, kind: 'cpu' | 'gpu' = 'cpu'): ProfileSpan | null {
  if (!enabled) return null;
  const span: ProfileSpan = { name, start: performance.now(), end: 0, kind, children: [] };
  const parent = stack[stack.length - 1];
  (parent ? parent.children : roots).push(span);
  stack.push(span);
  return span;
}

export function spanEnd(span: ProfileSpan | null): void {
  if (!span) return;
  span.end = performance.now();
  const idx = stack.lastIndexOf(span);
  if (idx >= 0) stack.splice(idx, 1);
}

// GPU kernel durations (from WebGPU timestamp queries) are only known once
// the async readback resolves, well after the kernel actually ran -- so
// there's no meaningful "current wall-clock position" to stamp them at.
// This lays the given stages out sequentially inside whatever span is
// currently open (starting at that span's own start time), purely so the
// flamechart's parent-relative percentages and ordering stay sane. The
// durations themselves are real, nanosecond-precision GPU-clock deltas;
// only their placement on the shared CPU timeline is synthetic.
export function attachGPUKernelBreakdown(stages: { name: string; durationMs: number }[]): void {
  if (!enabled || stages.length === 0) return;
  const parent = stack[stack.length - 1];
  const anchor = parent ? parent.start : performance.now() - stages.reduce((s, x) => s + x.durationMs, 0);
  let cursor = anchor;
  for (const { name, durationMs } of stages) {
    const span: ProfileSpan = { name, start: cursor, end: cursor + durationMs, kind: 'gpu', children: [] };
    (parent ? parent.children : roots).push(span);
    cursor = span.end;
  }
}

export function getRoots(): ProfileSpan[] { return roots; }

// Nested indented text, duration + percent-of-parent per line -- sorted by
// start time within each level so the output reads top-to-bottom the way
// the work actually happened.
export function formatFlamechart(): string {
  const lines: string[] = [];
  const walk = (span: ProfileSpan, depth: number, parentDur: number) => {
    const dur = span.end - span.start;
    const pct = parentDur > 0 ? ((dur / parentDur) * 100).toFixed(1) : '100.0';
    const tag = span.kind === 'gpu' ? ' [GPU kernel]' : '';
    lines.push(`${'  '.repeat(depth)}${span.name}${tag} -- ${dur.toFixed(2)}ms (${pct}%)`);
    const sortedChildren = [...span.children].sort((a, b) => a.start - b.start);
    for (const c of sortedChildren) walk(c, depth + 1, dur);
  };
  for (const r of [...roots].sort((a, b) => a.start - b.start)) walk(r, 0, r.end - r.start);
  return lines.join('\n');
}

// Same tree, as plain JSON -- for the dev-bridge test script to pull structured
// data back over eval instead of scraping the text format.
export function getFlamechartJSON(): ProfileSpan[] {
  return roots;
}
