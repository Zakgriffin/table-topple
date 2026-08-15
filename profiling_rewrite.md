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
per-camera latency history arrays, and every other stopwatch in the project —
the sweep's, the phone's, and the duplicated one in the display tail. §7 audits
every timestamp in the repo and gives each a verdict; the acceptance test for
"no old systems left" is a grep, stated at the end of §8.

**"One timing system" means one PROFILER, not one use of the clock.** A span
names an operation and answers *how long did that take*. Reading the clock to
schedule a repaint, to stamp a sensor event, or to poll a deadline is a different
question and stays exactly as it is. Drawing that line is §7's whole job.

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

### Measured: the resolution is 41.7 ns — ON THE FIRST FEW FRAMES ONLY

> **QUALIFIED 2026-08-15 by Phase 3.** The 41.667 ns tick is real, but it is
> only observable with Dawn's `timestamp_quantization` toggle DISABLED —
> otherwise every timestamp is rounded to a 65536 ns grid and most per-pass
> durations read zero. This probe ran on a quantizing instance and its "no
> coarsening at any scale" is therefore a statement about the probe, not the
> hardware. See Phase 3 in §8, and `DAWN_NODE_FLAGS` in `src/gpu/device.ts`.

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

**BUILT AND TESTED 2026-08-15 — and it was wrong until it was tested.** This was
the only part of the design with no test, and the first one written caught a
defect: GPU span ids already carry a `gpu:` prefix, so naming the measure
`~<clock>:<id>` emitted **`~gpu:gpu:<stage>`**. The rule is now simply `~` +
the span's id, using the same `mark()` the text renderer uses — so the marker
cannot diverge between the two views. Peer spans mirror as `~link.pull`.

Three tests cover it: nothing reaches User Timing with the mirror off, the names
carry exactly one marker (host rows unmarked), and `profilerReset` clears the
buffer it filled. Verified by mutation — restoring the old naming fails the
middle one.

**Why mirror them at all, given the caveat:** the alternative is a second viewer
for the GPU half, and the whole point of translating at the boundary is that
there is no second anything. A bar in the right order, with the right length,
anchored to a lower bound is strictly more informative than no bar — as long as
it says so, which is what the tilde is for.

---

## 7. Every timestamp in the project, with a verdict

**"One timing system" means one PROFILER, not one use of the clock.** There is a
real line here and it has to be drawn explicitly, or the rewrite either leaves
old systems standing or eats things that were never profiling.

A **span** names an operation with a beginning and an end, and the question it
answers is *how long did that take*. Everything else that reads a clock is
answering a different question — *what time is it*, *has enough time passed*,
*when did this sensor event happen* — and converting those to spans would be a
category error, not a unification.

Every `performance.now()` / `nowMs()` / `Date.now()` in the project, audited
2026-08-15:

### Becomes a span (the whole of the profiler's scope)

| site | today | after |
|---|---|---|
| `profiling/profiler.ts` + 16 call sites | the system | **the system**, halved |
| `axesReconstruction.ts` `projectMs` | its own stopwatch | deleted — `app.project` already measures it |
| `scripts/sweep.ts` per-pose `ms` | its own stopwatch | a span. **This is the payoff**: the sweep is where timing analysis actually happens (180 poses, medians, p90), so it gets the GPU pass breakdown for free |
| `mobileCapture.ts` `t0`/`totalMs` | its own stopwatch | a span on the phone |
| `capture.ts` `lastPullMs`/`lastEncodeMs`/`lastTransitMs` | three subtractions + three rolling arrays | `peer` spans; the latest sample is the newest record |
| `PoseRecord.computeMs` | a field on the IMU ring | gone; the ring keeps its non-timing fields and stops being a second profiler |

### Stays a raw clock read, and must not change

| site | why it is not profiling |
|---|---|
| `clock.ts` `nowMs()` | **the clock every span is built on.** Not a system; the foundation of the one system |
| `main.ts` rAF interval check | a SCHEDULER — "has `PREVIEW_UPDATE_INTERVAL_MS` elapsed" |
| `mobileCapture.ts` `sentAt`/`pulledAt`/`drawnAt`/`encodedAt` | the STAMPS the `peer` spans are built from, and they cross the wire. They stop being a parallel accounting system; they do not stop existing |
| IMU sample `t`, `observedAt`, `predictAt`, `settingsSyncedAt` | EVENT timestamps for sensor fusion. The filter integrates against them; they are not durations |
| `Camera.lastFrameStats` | COUNTERS (`loopTicks`, `backpressureBlockedTicks`) plus a rate summarized on the phone. A mean over a population that never crosses as individuals is not a span; if the individuals are ever wanted, they become spans then |
| `scripts/search-order5-torus*.ts` | a progress ETA on an offline search, unrelated to the pipeline |
| `profile-video-gap.mjs` `Date.now()` deadlines | polling loops. Scheduling |
| `src/game/ai.ts` | game simulation time, a separate track, and main must never edit `src/game` |

### Neither — deleted outright

`profilerBeginSession` / `ProfilerSession` (zero callers), `criticalPath` /
`formatCriticalPath` / `PathNode` / `CriticalPath` and every `inputs:`
declaration (§4), `spanDurationMs`, `profilerDevToolsMirror`, and the four
`Camera` history arrays with their caps, their `shift()` calls and their manual
reset in `profile-video-gap.mjs`.

**After this there is no second way to measure a duration anywhere in the
project.** The sweep, the app, the phone and the dev-bridge scripts all record
into one store and read it back the same way.

## 8. Phases

Each phase leaves the tree green and the app working.

**Phase 1 — demolition. DONE 2026-08-15.** Everything in §7 that is a deletion,
plus the `Span.clock` field defaulting to `'host'`. No behaviour change; 636 →
418 lines (the estimate was ~300; the difference is comment, not code).
`StageRecord` was renamed to `Span` to match §2, 5 call sites. Verified by `tsc`,
the suite (101/101, unchanged from baseline — no test ever covered the deleted
machinery), and `formatFlamechart()` still printing.

**Two things Phase 1 deliberately did NOT delete, though §7 lists them.** §7 is a
taxonomy — it says what each timestamp *is* — and §8 is the schedule. Where they
disagree, §8 wins. The four `Camera` history arrays are Phase **6**, because
deleting them before `profile-video-gap.mjs` re-points at the record store breaks
that script with nothing to replace it, which is a behaviour change; and the
sweep's and the phone's stopwatches are Phase **7** for the same reason.

**Phase 2 — the device feature. DONE 2026-08-15.** `src/gpu/device.ts` requests
`timestamp-query` when the adapter offers it. Nothing consumes it yet.

**It went to all THREE device-creation sites, not just the one this section
names** — the app, `scripts/sweep.ts`, and `tests/helpers/gpu.ts` — through one
exported `requestDeviceWithOptionalTimestamps(adapter)`. The sweep and the tests
are precisely where GPU timing gets read (Phase 7 and Phase 3 respectively), so a
site left on a bare `requestDevice()` would present as "timing is unavailable on
this machine" rather than as the omission it is. `canTimestamp(device)` is the
companion: read the DEVICE, never the adapter, since a device only has the
features it was created with.

**Requested when offered, never required.** A `requiredFeatures` entry the
adapter lacks makes `requestDevice` *reject*, so requiring it unconditionally
would trade the whole app for an instrument.

**The absent path is tested, not assumed** (`tests/pose2Timing.test.ts`, 3
tests). Since this adapter *does* offer the feature, the absence is manufactured:
a second device asking for no optional features at all. It is the only honest
way — the absence is a property of a device, fixed at creation, and no flag on
the shared device can imitate it. The middle test asserts the two devices
genuinely DIFFER, so the absent-path test cannot pass by silently running on a
capable device. Verified by mutation: making the helper skip the request fails
both feature tests and correctly leaves the untimed-pipeline test green.

The file also carries Phase 3's gate ahead of time — the pose from a timed and an
untimed device must be `deepStrictEqual`. It is **trivially green today**, and
that is stated in the test, so it starts failing the moment Phase 3's
`timestampWrites` perturbs anything.

**Phase 3 — the library forwards. DONE 2026-08-15.** `timestampWrites` in
`pass()`, `resolveQuerySet` + copy into the existing staging buffer,
`Pose2Frame.gpu` populated. **The gate holds and is measured, not asserted: one
submit, one fence, one map, and a pose `deepStrictEqual` between a timed and an
untimed device.** The counts come from `tests/helpers/countingDevice.ts`, which
counts the WebGPU calls themselves — asking the library how many passes it
encoded would have been one variable checked against itself.

**"A query set sized from `plan.stages`" was WRONG, and it is worth saying why.**
The encoded pass count is not the declared stage count: grow's
hook/compress/gate are re-encoded once per convergence round, all 32 every frame.
**Measured: 136 encoded passes at 96x128, against 14 declared stages.** The query
set is therefore sized by a capacity constant (`MAX_TIMED_PASSES = 512`) with a
throw in `pass()` on overflow and a test pinning the real count — rather than by
a function that predicts its own pass count, which would be the hand-mirrored
second declaration this project keeps finding as a defect.

### Dawn QUANTIZES timestamps by default, and it must be switched off

**The most important thing Phase 3 produced, and the one setting the whole
instrument depends on.**

Dawn ships a `timestamp_quantization` toggle, **ON by default**, a side-channel
mitigation that rounds every timestamp to a coarse grid. Measured here that grid
is **65536 ns**, against passes taking single-digit microseconds — so a pass's
begin and end land in the same tick and its duration reads exactly **zero**. A
typical frame: 114 of 136 passes at zero, five distinct values in the whole
frame.

The fix is one flag at INSTANCE creation, now `DAWN_NODE_FLAGS` in
`src/gpu/device.ts`, shared by the test helper and the sweep so they cannot
drift:

```
create(['disable-dawn-features=timestamp_quantization'])
```

With it, the same late frame reports **91 distinct durations and zero zeros**.

**It does not fail loudly.** It returns plausible numbers that happen to be
mostly zero, which reads as "the GPU is idle", or — as it did here for an
embarrassingly long stretch — as "the counter degrades after a couple of
frames". The misdiagnosis was reached honestly: the degradation really did track
frame count in the first experiments, and reuse and `destroy` were both ruled out
before the real cause turned up. **A quantized counter and a degrading one are
indistinguishable from the durations alone; only the raw u64s and the toggle
name tell them apart.** That is the transferable lesson, and it belongs with
§7's other "the instrument's own output can be structurally wrong while looking
plausible" entries.

**§5's "no coarsening at any scale" was measured on a quantizing instance and is
therefore not a statement about the hardware.** The 41.667 ns tick it reports is
real and now reproducible on every frame.

**THE OPEN QUESTION, and it is the one that matters for Phases 4-7: Chrome
applies the same mitigation.** An app-side capture needs Chrome launched with
`--disable-dawn-features=timestamp_quantization` (or
`--enable-webgpu-developer-features`), and **that is UNVERIFIED as of
2026-08-15**. Without it, every GPU number the app reports will be on the 65536
ns grid and will look like a pipeline that costs nothing. Verify before trusting
an app-side breakdown.

The tests keep the distinction visible rather than assuming it: they assert
structure (pass count, order, ids, non-negativity), which is sound whatever the
grid, and the reporting test prints a `COARSE COUNTER` marker whenever a frame
carries fewer than ten distinct durations.

**Phase 4 — translation and ingest. DONE 2026-08-15.** `src/sphereLab/profiling/
gpuSpans.ts` is **the only place in the project that knows there is more than one
clock**, and it is its own file to keep that true. GPU spans are appended with
`clock: 'gpu'` and a per-call `within` of `app.pose`; `pose.gpu.unattributed`
closes the window.

Two things the plan did not anticipate:

- **`tSubmit`/`tResolved` cannot be taken by `app.pose`.** The submit and the map
  both happen *inside* `runPose2`, so an outside bracket also contains upload and
  encode and would anchor the GPU block earlier than it could possibly have run.
  The library now reports `submittedAt`/`resolvedAt` alongside the passes. This
  does not breach §5: they are **stamps**, exactly like `mobileCapture`'s
  `sentAt`/`pulledAt`, and the library still builds no span and names no
  difference.
- **A pass needs its POSITION, not just its duration.** `PassTiming.startNs` is
  nanoseconds from the frame's first timestamp — relative, because the raw
  counter runs to ~1.4e15 and nothing outside the frame can use an absolute GPU
  time anyway.

`joinRecords` gained the table-less path §9 demanded (see that section). The
unattributed row is **clamped at the window's end**, since the anchor is a lower
bound and an over-long GPU block must report zero leftover rather than a
backwards span.

**Phase 5 — the renderer. DONE 2026-08-15.** Leaf repeats aggregate by id within
a parent (`n=32 total=0.43ms median=0.013ms`), the `~` marker goes on any row
whose clock is not `host`, and `spanIngest` mirrors non-host spans as
`~<clock>:<id>`. Only LEAVES aggregate — collapsing a repeated stage that has
children would hide a level of decomposition, which is not worth a shorter
report. A 101-pass frame renders as eight rows:

```
axesReconstruction -- 8.34ms (100.0%, self 0.24ms)
  capture+preprocess -- 2.08ms (25.0%, self 2.08ms)
  pose2 (submit + readback) -- 6.02ms (72.1%, self 2.04ms)
    ~gpu:gradient index=0 -- 0.01ms (0.1%, self 0.01ms)
    ~gpu:grow.hook -- n=32 total=0.43ms (7.1%) median=0.013ms
    ~gpu:fit.eigen index=99 -- 1.22ms (20.3%, self 1.22ms)
    pose.gpu.unattributed -- 1.04ms (17.3%, self 1.04ms)
```

**Phase 6 — the link spans. DONE 2026-08-15.** Phone-link latencies are `peer`
spans; the four history arrays and `lastPullMs`/`lastEncodeMs`/`lastTransitMs`
are deleted. **`profile-video-gap.mjs` was DELETED rather than re-pointed** — the
user's call. It was the only reader of those arrays, was referenced nowhere else
in the repo, and needed the dev bridge plus a focused tab plus a freshly loaded
phone to answer a question nobody is currently asking. If it comes back it is a
short script over the record store, and against better data than the arrays were.

**Only TWO spans came out of the three latencies, and this is the interesting
part.** `link.pull` (sentAt→pulledAt) and `link.encode` (pulledAt→encodedAt) have
both endpoints on the PHONE's clock, so the cross-device skew cancels and the
durations are true whatever it is.

**`link.transit` does not exist.** It would be `receivedAt` (DESKTOP clock) minus
`encodedAt` (PHONE clock), which capture.ts had already measured at **about
-38 ms** on this pair of machines — a negative network time. It is not a duration;
it is a measurement of the skew. A span cannot hold it: `end < start` breaks
containment and lets a union of children exceed its parent, the exact defect
class the flat store was built to eliminate, and unlike the GPU anchor there is
no lower bound to retreat to because the SIGN is wrong. Recovering real transit
means solving for the offset first. Until something does, an omitted row beats a
confidently negative one. The raw stamps all survive on `lastCaptureTiming`,
where the IMU work reads them.

`gpuSpans.ts` became **`clocks.ts`** when the peer boundary landed in it, so
"the only place in the project that knows there is more than one clock" stays
literally true of one file. The byte count that `payloadBytesHistory` carried is
now an attribute on `ingest.run`.

**Phase 7 — the other two stopwatches. DONE 2026-08-15.**

**The sweep** records a `sweep.pose` span instead of its own `performance.now()`
pair, ingests `frame.gpu` under it, and prints a per-stage GPU breakdown after
the accuracy summary — median, p90 and share, sorted. This is the payoff and it
works: a first reading at 240x320 puts **3.66 ms on device against 11.2 ms wall
on a warm pose** (43.7 ms cold, which is shader compilation).

One thing the plan did not foresee: **the store's 4096-record cap.** 180 poses x
~140 passes is ~25k records, so letting them accumulate would trim the early
poses straight out of the medians. Each pose is folded into the summary and the
store is reset — the store is the transport, the accumulator is the report, the
same relationship the renderer has to it.

**The phone's stopwatch was DELETED, not converted, and the reason is worth
keeping.** §7 says `t0`/`totalMs` becomes a span. It cannot: the pose computation
went with `src/pose`, so what sat between those two clock reads was
`void grayTopDown;` and an object literal. The stopwatch reported ~0 ms every
frame, and a span would have put that same 0 ms on a flamechart as a row — a more
confident way of saying something false. The on-page readout printed
`pose <totalMs>ms (<fps>fps)` off it and now says `no pose pipeline on device
yet` instead.

`PoseRecord.computeMs` is gone with it, so the ring keeps only non-timing fields
and has stopped being a second profiler, exactly as §7 wanted — just by
subtraction rather than by conversion. **A `TODO(phone-on-pose2)` marks the
capture site**: when that page runs `src/pose2`, the span opens around
`runPose2`, and `frame.gpu` gives the phone the same per-pass breakdown the
desktop has through `clocks.ts`.

### The acceptance grep passes

`grep -rn 'performance.now' src/ scripts/` now returns only: the recorder
(`profiler.ts`), the scheduler (`main.ts`), the game's simulation clock, `nowMs()`
itself, and three STAMPS — `devBridge/client.ts`'s `receivedAt` and
`run.ts`'s `submittedAt`/`resolvedAt`. Every one is a clock read rather than a
measurement. **There is no second way to measure a duration left in the
project.**

**After phase 7, `grep -rn 'performance.now' src/ scripts/` returns only the
recorder, the scheduler in `main.ts`, the game's simulation clock and the offline
search's ETA — every one of them a clock read rather than a measurement.** That
grep is the acceptance test for "no old systems left".

---

## 9. Open questions

- ~~**Does the per-call `within` override scale to 40 GPU ids?**~~ **ANSWERED
  2026-08-15, and the answer is the bad one: they report as `unknown` and do not
  join.** Measured in Phase 1 by opening a span with an id absent from the table
  and an explicit `within` pointing at a live parent. The cause is structural, not
  a tuning problem: `joinRecords` filters on `table[r.id]` in its first loop and
  pushes to `unknown` before it ever calls `declaredParent`, so **an id with no
  table entry can never join no matter what parent it declares.**

  The consequence is the one §3 says must not happen: the GPU span's time is not
  subtracted from anything, so it stays in the submitter's **self time**, and the
  decomposition silently stops adding up.

  **FIXED in Phase 4.** `nodeFor()` in `joinRecords` gives an unknown id with an
  explicit `within` a synthesized node whose label is the id; an id with no table
  entry *and* no override still lands in `unknown` and stays out of the tree.
  Both halves are regression-tested in `tests/profilerJoin.test.ts`, and the
  library's stage list is nowhere mirrored in the app.
- **What is the actual CPU/GPU split of a 13.5 ms reconstruction?** **First
  reading, 2026-08-15: about a third on device.** The sweep at 240x320 reports
  3.66 ms of GPU against 11.2 ms wall on a warm pose. **Indicative only** — two
  poses at reduced resolution, and the 13.5 ms figure is a 480x640 number, so the
  proper measurement is a full sweep at that resolution. The instrument to do it
  now exists and prints the breakdown by default. The old pipeline's "66% blocked
  in readback stalls" was seven fences and still does not transfer.
- **NEW: where does the host-side two thirds actually go?** `app.pose`'s self
  time is upload + encode + fence + map, and `pose.gpu.unattributed` separates
  the last two. Nobody has read those rows yet.
- **Does the phone need any of this?** `mobileCapture.ts` has its own device and
  will have its own context. It is out of scope until it has a pipeline at all.
