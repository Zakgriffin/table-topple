# Profiling Rewrite — one timing system, project wide

Every duration this project measures, in one store, under one record type, with
one join and one renderer. Written 2026-08-15, before any code.

This document is the plan and the argument. `full_system_breakdown.md` is the
pipeline's; this one is the instrument's.

---

## START HERE

**The goal in one sentence:** there is exactly one way to record a duration, and
whether it was measured on the host CPU, on the GPU, or on a phone across the
network is a **field on the record**, not a different system.

**The one hard constraint, and everything below is shaped by it:** GPU
timestamps and `performance.now()` are different clocks with no defined
relationship. See [The three clocks](#the-three-clocks). It is resolved by
translating at the boundary, once, so that everything downstream sees one
timeline.

**What is being deleted:** roughly half of `profiling/profiler.ts`, all four
per-camera latency history arrays, and one duplicated stopwatch. See
[Demolition](#demolition).

**What is being added:** GPU pass timing, which this project has never had for
`src/pose2`, at a measured resolution of 41.7 ns.

---

## 1. What exists today, and why it is seven things

| # | system | where | state |
|---|---|---|---|
| 1 | span recorder + declared join | `profiling/profiler.ts` (636 lines), `stages.ts` (96) | the real one; 16 call sites, 14 declared stages, **all app-side** |
| 2 | `nowMs()` | `clock.ts` | a clock, not a profiler. **Keep as is.** |
| 3 | phone-link latencies | `Camera.lastPullMs` + 4 rolling arrays + `payloadBytesHistory` | hand-rolled ring buffers, one reader |
| 4 | `lastFrameStats` | `Camera`, self-reported by the phone every ~2s | counters, not durations. **Keep as is.** |
| 5 | phone `PoseRecord` ring | `mobileCapture.ts` | the IMU track's own recording |
| 6 | sweep stopwatch | `scripts/sweep.ts` | one `performance.now()` pair around `runPose2` |
| 7 | `projectMs` | `axesReconstruction.ts` | **a duplicate** — re-times what the `app.project` span already measures |

And an eighth entry that is the point of the whole exercise:

| 8 | `src/pose2` | — | **nothing at all.** Zero timestamps. The pipeline is 44 encoded passes and not one of them is measured. |

**Already dead**, by external-reference count: `profilerBeginSession` +
`ProfilerSession` (~55 lines, its harness went with `src/pose`),
`spanDurationMs`, `profilerDevToolsMirror`. `criticalPath` /
`formatCriticalPath` are reachable only from `formatSpanTree`.

---

## 2. The model

### One record

```ts
interface Span {
  id: string;
  start: number;      // ms, ALWAYS host-clock after translation
  end: number;        // 0 while open
  clock: Clock;       // where the number CAME from -- see below
  attrs?: SpanAttrs;
  within?: string | null;  // per-occurrence parent override
}

type Clock = 'host' | 'gpu' | 'peer';
```

`clock` is **provenance, not units**. By the time a record is in the store its
`start`/`end` are host-clock milliseconds. The field says how it got there, which
is what a reader needs to know how much to trust its absolute position.

### One store, flat, always recording

Unchanged from today and non-negotiable: `spanStart` appends a record and does
nothing else. There is no span stack, so nothing can be reparented by the
accident of what was suspended when. That design fixed two real bugs (a
reconstruction filed inside a display drain; negative self times from
concurrently-awaited children) and it is the part of the current profiler that is
worth keeping wholesale.

### Structure is declared, joined afterwards

Also unchanged: a stage declares its containment parent (`within`), and a record
attaches to the occurrence of that parent whose interval contains it. Declared
parent **and** measured containment — neither alone.

---

## 3. The three clocks

This is the section to read before writing any code.

| clock | source | offset to host clock | rate error |
|---|---|---|---|
| `host` | `performance.now()` | zero, by definition | — |
| `gpu` | `timestamp-query`, ns | **unknown, no API to obtain it** | negligible (both count real time) |
| `peer` | the phone's `nowMs()` | NTP skew between two machines | negligible |

**Durations are trustworthy on all three. Absolute positions are trustworthy
only on `host`.** That is the whole of the problem, and it is the same problem
twice: a GPU span and a phone span both know exactly how long they took and only
approximately when.

### The resolution: translate once, at the boundary

A GPU span is converted to host-clock milliseconds at the moment it enters the
store, and never again afterwards. Every consumer — the join, the renderer, the
DevTools mirror, a dev-bridge script — sees one timeline and needs no special
case.

The translation, for one submit:

```
hostSpan     = [tSubmit, tResolved]     measured on the host, around the submit
gpuSpan      = [gpuFirst, gpuLast]      the extremes of that submit's timestamps
offset       = tSubmit - gpuFirst       nanoseconds, converted to ms
span.start   = gpuStart * 1e-6 + offset
```

**No scaling, only an offset.** Both counters tick real time; even 100 ppm of
relative drift is 1.3 µs across a 13 ms reconstruction, well under the noise the
numbers already carry. Fitting a scale would be modelling precision we do not
have.

### What that guarantees, and what it does not

**Guaranteed:** every GPU span's *duration* is exact to the counter's tick, and
the *order and relative spacing* of GPU spans within one submit are exact. The
whole GPU block lies inside `[tSubmit, tResolved]`, because a submit precedes
execution and the map cannot resolve before the work completes.

**Not guaranteed:** the absolute placement. Anchoring `gpuFirst` at `tSubmit`
asserts the GPU began the instant the queue received the work, which is a **lower
bound**, not a measurement. In reality some of the window is queue latency before
the first pass and some is fence and map overhead after the last, and **nothing
available to us can say how it splits.**

So the leftover is not hidden — it becomes a span of its own:

```
pose.gpu.unattributed = [tSubmit + gpuTotal, tResolved]
```

with `clock: 'host'`, because both of its endpoints are host-measured. It is a
real quantity worth watching (it is queue latency + fence + map, and for pose2 it
is the entire cost of crossing the bus), and giving it a row keeps the
decomposition honest: the GPU rows plus this row account for the whole submit,
with nothing swept into a parent's self time.

**The one thing a reader must be told, and the renderer will say it:** GPU rows
are drawn with a `~` prefix on their start column. Length is measured; position
is anchored.

### The same rule for `peer`

A phone-link span (`link.transit`) has its start on the phone's clock and its end
on the desktop's. Both are epoch-based `nowMs()`, so they are directly
comparable **up to NTP skew** — the same shape of uncertainty, with the same
resolution: record it, mark the clock, trust the duration more than the position.
`clock.ts` already made this possible by putting both devices on one epoch; this
is the first thing to actually use it.

---

## 4. Structure: flat records, declared tree, no dependency graph

The current profiler declares two graphs. `within` gives containment and produces
the decomposition tree. `inputs` gives data dependency and produces a critical
path with a `waitMs` per edge.

**`inputs` and the critical path are being deleted**, and the argument is
specific rather than a preference for less code.

`waitMs` answered "which dependent chain sets the floor" for a pipeline with a
dozen interleaved GPU readbacks, where two stages could be genuinely concurrent
and the wait between them was invisible inside either span. **That pipeline is
deleted.** `src/pose2` is one submit and one fence. What is left at the app level
is `capture → pose → project → overlays`, strictly serial, plus the display tail
as an independent root. A graph walker for three edges is machinery without a
question to answer.

The concurrency that *does* remain is CPU-versus-GPU, and that is not a
dependency graph either — it is one number, `pose.gpu.unattributed`, computed
directly.

**If a future pipeline reintroduces overlapping awaits, this comes back.** The
records are flat and the ids are stable, so it can be rebuilt against records already captured.
Nothing about deleting it now is irreversible.

### Repeats aggregate at render time

`grow.hook` / `grow.compress` / `grow.gate` run once per convergence round — up
to 32, so ~136 encoded passes on a bad frame. Per-occurrence rows would be
unreadable. The renderer groups by id within a parent and prints
`n=32  total=4.21ms  median=0.12ms`; the raw records stay in the store for
anything that wants them.

---

## 5. The library boundary

**`src/pose2` gets no profiler, no spans, and no timing module.** It forwards raw
numbers and takes no position on what they mean.

```ts
// Pose2Frame gains one optional field.
gpu?: readonly { stage: string; ns: number; index: number }[];
```

`index` is the pass's position in the submit, so a consumer can order them
without re-deriving the encode order.

### Why the library reports GPU time and nothing else

Everything on the host clock is measurable from **outside** the library. The app
already wraps `runPose2` in the `app.pose` span; that span is upload + encode +
submit + fence, end to end. Subtracting the GPU total from it gives the host-side
remainder with no cooperation from the library at all.

What is *not* visible from outside is which pass inside the submit took the time,
and that is exactly what pose2 alone can say — it owns `pass()`, the single
`beginComputePass` in the codebase, which already takes a `stageId`.

**So the split is: the library reports what only it can see; the app derives
everything else by subtraction.** Upload and download are deliberately *not*
reported by the library, even though they happen inside it, because they are
host-clock quantities the app can bracket itself.

### Cost, and why it does not need a per-frame toggle

`resolveQuerySet` into a buffer copied into **the same staging buffer, in the
same encoder**, before the same submit. One submit, one fence, one map — exactly
the argument that made `inspect` cheap. Timestamps cost bytes, not a round trip,
and unlike `performance.measure` none of it runs on the main thread.

So GPU timing is **on whenever the device supports it**. The one thing that must
be decided up front is `requiredFeatures: ['timestamp-query']` at device
creation, which belongs to `src/gpu/device.ts` and cannot be changed per frame.
Request it when the adapter offers it; the pipeline must still run without it.

### Measured: the resolution is 41.7 ns

Probed 2026-08-15 against the same Dawn build the tests and sweep use.

- **`timestamp-query` is available** on the adapter.
- **The tick is 41.6667 ns — a 24 MHz counter.** No coarsening at any scale.
- Reported duration tracks work monotonically across 16 load levels from 2.4 µs
  to 1.14 ms, all 16 medians distinct.
- **The floor in practice is ~2.4 µs, and it is an EMPTY pass** — dispatch launch
  overhead, a real cost rather than measurement error. A one-workgroup stage
  reading 2.4 µs is reporting something true.

**A trap worth keeping:** the GCD of raw timestamp differences comes out **1 ns**,
which reads as a 1 ns timer and is not. The grid is 125/3 ns, which lands off the
nanosecond grid two ticks in three, so a GCD in ns is 1 by construction. Candidate
periods have to be **tested**, not derived. *A GCD of 1 means "not an integer
grid", not "no grid".*

Chrome does not coarsen either, per the user's own prior investigation, so one
API shape serves the browser and the harness. The library reports raw ns
regardless and lets the consumer decide what is meaningful — baking a
significance threshold into the library would be a policy that has to be right
for both.

---

## 6. The DevTools mirror

**Semantics unchanged.** The `profilerEnabled` checkbox gates one thing: whether
each closed record is also emitted as a `performance.measure`, putting it in
Chrome DevTools' Timings track. It is off on every load and never persisted,
because a `performance.measure` per span is real main-thread work inside whatever
is being measured and a checkbox that survived a reload would tax every
measurement taken afterwards.

**GPU spans mirror too**, and this is the part the translation buys. Because they
were converted to host-clock milliseconds at ingest, `performance.measure(id,
{start, end})` just works, and a reconstruction appears in the Timings track as
the host span with the GPU passes nested underneath it in the right order and at
the right relative spacing.

The caveat travels with the label rather than being lost: mirrored GPU measures
are named `~gpu:<stage>`. The tilde is the reminder that the bar's **length is
measured and its position is anchored**.

**Why mirror them at all, given the caveat:** the alternative is a second viewer
for the GPU half, and the whole point of translating at the boundary is that
there is no second anything. A bar in the right order, with the right length,
anchored to a lower bound is strictly more informative than no bar — as long as
it says so, which is what the tilde is for.

---

## 7. Demolition

Delete outright:

- `profilerBeginSession`, `ProfilerSession` — zero callers.
- `criticalPath`, `formatCriticalPath`, `PathNode`, `CriticalPath`, and every
  `inputs:` declaration in `stages.ts`. See §4.
- `spanDurationMs`, `profilerDevToolsMirror` — dead exports.
- `Camera.pullMsHistory` / `encodeMsHistory` / `transitMsHistory` /
  `payloadBytesHistory`, their caps, their `shift()` calls, and their manual
  reset in `profile-video-gap.mjs`. **They become spans** (`link.pull`,
  `link.encode`, `link.transit`, with `bytes` as an attr), after which the record
  store *is* the history and the script reads records like every other consumer.
- `projectMs` in `axesReconstruction.ts` — the readout line reads the
  `app.project` span it already has.

Keep, unchanged:

- `clock.ts`. It is the reason `peer` spans are expressible at all.
- `Camera.lastFrameStats`. Counters, not durations; a different instrument.
- `Camera.lastPullMs` / `lastEncodeMs` / `lastTransitMs` — the *latest* sample,
  read by the live UI. Cheap, and not a history.

Deliberately out of scope for the first pass:

- The phone's `PoseRecord` ring. It belongs to a paused track
  (motion-blur/IMU), it is dumped rather than joined, and folding it in now
  would change a recording format nobody is currently reading.

---

## 8. Phases

Each phase leaves the tree green and the app working.

**Phase 1 — demolition.** Everything in §7 that is a deletion, plus the
`Span.clock` field defaulting to `'host'`. No behaviour change; ~636 → ~300
lines. Verifiable by `tsc`, the suite, and `formatFlamechart()` still printing.

**Phase 2 — the device feature.** `src/gpu/device.ts` requests
`timestamp-query` when the adapter offers it. Nothing consumes it yet. The
pipeline must still run when it is absent, and that path needs a deliberate test
rather than an assumption.

**Phase 3 — the library forwards.** A query set sized from `plan.stages`,
`timestampWrites` in `pass()`, `resolveQuerySet` + copy into the existing staging
buffer, `Pose2Frame.gpu` populated. **The gate: one submit, one fence, one map,
byte-identical pose.** A stage test asserts the pose block is unchanged with
timing on and off, and that the reported pass count matches the encoded pass
count.

**Phase 4 — translation and ingest.** `app.pose` records `tSubmit`/`tResolved`,
translates the GPU spans, and appends them with `clock: 'gpu'` and a per-call
`within` pointing at the host span that submitted them. `pose.gpu.unattributed`
is computed here. **This is the only place in the project that knows there is
more than one clock.**

**Phase 5 — the renderer.** Aggregation by id, the `~` marker on anchored rows,
and the `~gpu:` prefix on mirrored measures.

**Phase 6 — the link spans.** Phone-link latencies become `peer` spans;
`profile-video-gap.mjs` re-points at the record store; the four arrays are
deleted.

---

## 9. Open questions

- **Does the per-call `within` override scale to 40 GPU ids?** The rule is "a GPU
  span belongs to the host span that submitted it", declared at the injection
  point — which is the existing per-occurrence override, used exactly as
  intended. But it means 40 ids that are in no table. They should join under
  their submitter and render fine; if `formatSpanTree` reports them as
  `unknown` instead, the join needs a table-less path rather than 40 hand-mirrored
  declarations. **Hand-mirroring the library's stage list in the app is the one
  outcome to refuse** — that drift is a defect this project has found repeatedly.
- **What is the actual CPU/GPU split of a 13.5 ms reconstruction?** Nobody has
  measured it. The old pipeline's "66% blocked in readback stalls" was seven
  fences and does not transfer. This is the first number the rewrite should
  produce, and it is the reason to build it.
- **Does the phone need any of this?** `mobileCapture.ts` has its own device and
  will have its own context. It is out of scope until it has a pipeline at all.
