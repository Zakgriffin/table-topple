# Full System Breakdown — the flat, all-GPU pose pipeline

A from-scratch rewrite of `src/pose/` as a flat function that takes a grayscale
image and returns a camera pose, doing all work on the GPU with one readback.
This document is the single place to answer "what is this buffer for", "why is
it that size", and "what was decided and why".

---

## START HERE — state as of 2026-08-14

**Status: BUILT, END TO END.** Every declared stage runs on device. An image goes
in and a camera pose comes out: one upload, one submit, one 128-byte readback, no
host in the middle. `npm test` is 144 green; `npx tsc --noEmit` is clean.

Branch `consolidate-and-purge`; **every file of this work product is UNTRACKED —
`src/pose2/`, all `tests/pose2*`/`tests/helpers/gpu.ts` files, `scripts/sweep.ts`,
`scripts/hull-measure.ts`, and THIS DOCUMENT.** Only `package.json`,
`package-lock.json` and `tsconfig.json` are modified-tracked. So `git diff` and
`git stash` are blind to all of it and `git checkout .` would not restore one line
— copy files before experimenting on them.

| built | stage entries |
|---|---|
| §5 gradient | `gradient` |
| §6 grow | `grow.init/hook/compress/gate` |
| §7 collect | `collect.tally/markKept/regionMeta/scatter/finalize` |
| §8 lsdFit | `lsdFit` |
| §9 lines + votes | `lines.flag/emit`, `line.scan/spine/add`, `votes.cast` |
| §10 fit | `fit.ata`, `fit.eigen` — **and `fit.reduce` is DELETED** |
| §11 gpp | `gpp.classify/compact/extent/sweep/peaks/distinct/polish`, `family.scan/spine/add` — **and `gpp.extentInit` is DELETED** |
| §12 decode layout | `decode.binThreshPartials/binThreshReduce/layout` — **and `decode.bounds`/`decode.boundsInit` are DELETED, never written** |
| §13 decode | `decode.build`, `decode.tally.o0..o3`, `decode.argmax`, `decode.correctness` |
| §14 finish | `finish` |
| the shared scan | **all three uses are live** — collect's, lines' and gpp's |

**The entry point is `src/pose2/run.ts`**: `createPose2Context(device, dims)` once,
then `runPose2(ctx, gray, settings)` per frame. That file is deliberately the only
place the upload, the submit and the readback appear, so the count rule 1 exists to
protect is checkable by reading one screen.

### What to do next

1. **Read the sweep numbers** — `npm run sweep -- --pipeline both` scores
   `src/pose` and `src/pose2` over the same 180 poses and the same renders. This
   is §19's acceptance criterion and it is the thing that decides whether the
   rewrite succeeded. The baseline to beat is in §19.
2. **Then the sweep says which stage tests were worth having** (§19's calibration
   note), and the priority inverts: run it first, and let it localize.
3. **Phase 3 — replace.** Swap the app onto this pipeline, then delete
   `src/pose/`. Note `scripts/hull-measure.ts` imports `src/pose` and is a
   measurement harness rather than a shipping path.

### What is NOT done

- **The sweep has not been used to retire any stage test.** §19's calibration note
  says that is what should happen now that it can run. Whether it SHOULD is
  unsettled — the user's position as of 2026-08-14 is that the stage tests may be
  worth keeping regardless, and the mutation run below is an argument for that:
  a whole-pipeline pass rate cannot tell you which claim it is testing.
- **Phase 3, replace.** Not started. See §19.
- Two smaller ones: the hull was measured on `src/pose`'s detected lines and the
  grazing band (tilts 45–55) has not been checked against pose2's own detector;
  and open decision 9's `worst < 0.05` gate still has only ~1.4x headroom.

### The files

| file | lines | what it is |
|---|---|---|
| `src/pose2/pipeline.ts` | 456 | **The pipeline as DATA.** Every buffer, every stage. Liveness, the clear schedule and two validation rules are all *derived* from it |
| `src/pose2/buffers.ts` | 364 | Pure planner (`planPool`) + `createBuffers`. Runs under `node --test` with no device |
| `src/pose2/pose.ts` | 1041 | One encode function per stage. Never allocates, submits, awaits or reads back |
| `src/pose2/pose.wgsl.ts` | 3187 | The shaders, in pipeline order |
| `src/pose2/run.ts` | 117 | **The entry point.** The device lifecycle, THE upload, THE submit, THE readback — kept apart so the count is checkable |
| `src/pose2/board.ts` | 124 | The printed board as device buffers. The only place `src/pose2` reaches outside itself (open decision 5b) |
| `src/pose2/cpu.ts` | 467 | **The CPU twin** — an oracle, never imported by the pipeline |
| `src/pose2/sim.ts` | 289 | The simulator: renders a known pose to a grayscale frame |
| `src/pose2/sweep.ts` | 212 | The pose sweep. Deliberately pipeline-agnostic |

And the tests, which are where each stage's oracle actually lives:

| file | lines | what it covers |
|---|---|---|
| `tests/pose2Stages.test.ts` | 2135 | Every GPU stage, against hand-derived ground truth and rendered frames. **Where a new stage's tests go**, and it ends with the whole-pipeline test |
| `tests/pose2Cpu.test.ts` | 424 | The CPU twin, and the GPU-vs-twin comparisons for grow/collect/lsdFit |
| `tests/pose2Buffers.test.ts` | 296 | `planPool` arithmetic, with no device. §19 records this one as over-produced |
| `tests/pose2Sim.test.ts` | 205 | The simulator, incl. the bit-for-bit `rayDirInto` vs `cornerDir` check |
| `tests/pose2Sweep.test.ts` | 116 | The sweep harness itself, not a pipeline |
| `tests/helpers/gpu.ts` | 115 | `withDevice` (the error scope), `readF32/readU32`, the retained GPU instance |

### Running things

```
npm test                                         # everything, headless
node --test --test-force-exit tests/X.test.ts    # ONE file
node --test --test-force-exit --test-name-pattern '^gpp:' tests/pose2Stages.test.ts
npx tsc --noEmit
npm run sweep [--quick] [--pipeline pose|pose2|both]   # §19's acceptance criterion
npm run hull [--quick|--grazing] [--inset N]     # the §12 line-hull measurement
```

**`--test-force-exit` is not optional and not a style choice**: creating a device
segfaults at process teardown without it. A runner that omits it — `npx tsx
--test`, for instance — HANGS with no output, which reads as a broken test rather
than a missing flag.

**The mutation loop, since every stage here ends with one.** `src/pose2/` is
untracked, so git cannot restore it:

```
cp src/pose2/pose.wgsl.ts $SCRATCH/          # 1. copy OUT of the repo
                                              # 2. edit in the deliberate bug
node --test --test-force-exit tests/pose2Stages.test.ts
cp $SCRATCH/pose.wgsl.ts src/pose2/          # 3. restore
diff $SCRATCH/pose.wgsl.ts src/pose2/pose.wgsl.ts && echo CLEAN   # 4. PROVE it
```

Step 4 matters: a half-restored mutation is a silent wrong answer in every later
measurement.

**Drive it from a table, not by hand.** §10's run was a ~40-line script holding
`(name, oldText, newText)` triples, and three details in it are worth reusing:

- **Assert the pattern matched EXACTLY ONCE** before writing the file. A typo'd
  pattern otherwise mutates nothing and reports "no test caught it", which is the
  one wrong answer a mutation run must never produce.
- **Always patch from the pristine backup**, never from the working file, so a
  failed restore cannot compound across mutations.
- **Filter the run with `--test-name-pattern '^stage:'`** so each mutation costs
  a few seconds and the output is one line per mutation. A full-suite run per
  mutation makes a ten-mutation table too slow to bother finishing.

**And for any correction of an arbitrary-but-deterministic quantity, search for
the fixture rather than inventing one.** §10's two sign flips were both invisible
to assertions written specifically for them. What found the inputs: disable the
correction, sweep the input space, and print the cases where the RAW output is
wrong. That also yields the honest reachability statement, which "no test caught
it" cannot distinguish from a true property on its own.

**Where the settings constants live.** Every threshold this pipeline takes is a
real user-facing default in **`sphere-lab.config.json`**, not a number invented
here, and the tests use those values so a fixture cannot drift from the shipping
configuration. As of 2026-08-12: `lsdToleranceDeg 9.5`, `lsdRhoNoiseThreshold
0.132`, `lsdRhoHighThreshold 0`, `lsdNfaEpsilon 1`, `lsdNfaTestExponent 5`,
`lsdMinRegionSize 2`, `lsdMinLengthPx 3`. `src/pose2` deliberately does not
*import* the config — it takes a settings object per stage — but the values it is
tested at should keep matching it.

### What will bite you first

1. **A WebGPU validation error does not throw.** It makes every command using the
   offending resource a no-op, so the symptom is plausible zeros. Every test body
   runs inside an error scope (`withDevice`); a test asserting only on numbers
   would pass against a pipeline that never ran.
2. **`binds` in pipeline.ts is what a pass BINDS, not what its shader reads.**
   Liveness comes from it. Trimming it to actual reads frees a buffer early and
   hands the pass an aliased one inside its own bind group (§18).
3. **An arena slice is last frame's bytes.** Anything whose correct initial state
   is not zero needs a *write*, not a clear — and a grep for `zero` misses those.
   `growArgs` is the sharp one; `labelSurvives` was found by re-deriving (§7).
4. **Determinism is load-bearing.** Regions are ordered by ascending label, which
   is why collect scans instead of atomically appending, and why the twin can
   compare region-for-region with no canonicalization.
5. **A SCAN INPUT must be written over the whole scanned range, every frame.** A
   scan's element count is fixed at encode time, so it is always a cap
   (`maxRegions`, `maxLines`) and never the device-side count. A producer that
   dispatches indirectly over that count writes a prefix and leaves the tail
   holding the previous frame's data. Either dispatch it directly over the cap
   with an interior guard — what `lines.flag` does (§9) — or mark the buffer
   `zero`. **The declaration gets this wrong by default**: it was wrong for
   `lineFlag` and was wrong again for `family` (§11), because writing
   `indirectFrom: 'lineArgs'` on the producer looks obviously right. Both are
   fixed by dispatching directly with an interior guard. `family` carries a
   second trap on top: its dead tail must be written `(0,0)` and **not** `(0,1)`,
   because the pair is (isRow, 1−isRow) and "not a row line" is not the same
   fact as "a column line".

   **It is not only scans.** §10 found the third instance on `ataPartials`,
   which is a plain reduction: `fit.ata` was declared indirect off `lineArgs`
   and `fit.reduce`'s extent was fixed at encode time, so a quiet frame would
   have fitted the union of itself and a busy predecessor. The general form is
   **any producer and consumer that iterate different domains** — a scan input
   is just the most common way to get there. There it was fixed by deleting the
   consumer rather than by widening the producer, which is the better move when
   the intermediate only existed for a host that is gone.

6. **EIGHT storage buffers per compute stage is the WebGPU baseline**, and
   exceeding it is a validation error rather than a slow path. The adapter's own
   error message offers to raise the limit; taking that offer trades a portable
   pipeline for one that runs on this machine. Three stages have hit it and each
   was fixed differently, which is the useful part:
   - `gpp.polish` — **pair fields never read apart into a `vec2`** (value,
     weight). Done three times in §11.
   - `decode.layout` — **delete an input, and give another away.** `uvBounds`
     went with the hull, and `correctnessArgs` moved to `decode.argmax`, which is
     its real owner because the extent depends on the winning orientation.
   - `finish` — **read the value from the one place that already had to have it.**
     `triad` and `gppResult` both came off: `layout` already carries the triad,
     and `layout.distance` IS the height, so the period is `cellPitch / distance`.

7. **WGSL RESERVED KEYWORDS ARE ORDINARY ENGLISH WORDS.** Three so far, and the
   list is not the point — the point is that they read as the obvious name every
   time: `active` (§11), `layout` (decode.layout's own binding, now `lay`) and
   `match` (decode.tally's lookup result, now `found`). Two of the three fail
   LOUDLY, as a parse error the error scope reports by name. `active` did not: as
   a local it was accepted and then silently did nothing, and the symptom was a
   probe printing plausible zeros for every pose.

8. **A `vec3` STRUCT MEMBER IS 12 BYTES WITH ALIGN 16, so the next scalar packs
   into its trailing gap.** WGSL computes that for you and it is invisible there;
   a host-side reader written from the field order is four bytes out from the
   first scalar onward, and every value it reads is some other field. Hit on
   `Layout`'s first test reader. The fix in the struct rather than in the reader:
   the three axes are `vec4` with only `xyz` used, which costs 12 bytes and makes
   every offset the obvious one.

### The discipline this rewrite runs on

**Re-derive each stage from what the math needs; consult the old implementation
only for correctness constants** (tolerances, the NFA formula, tie-break rules),
never for structure. The declaration in `pipeline.ts` was written by reading
`src/pose`, so every stage still to be built has inherited decomposition in it
until someone checks. §7 is the worked example: six passes and two scans became
five passes and one, and the re-derivation is what found a zero-init bug the
declaration had.

**Mutation-test an oracle before believing it.** Six times now a green test was
green for the wrong reason — a bar too wide for its own predicate test, an
`assertBinds` check that was unreachable, an unsigned alignment dot that a whole
rendered frame could not see (§8), a stale-flag test that survived the exact
mutation it was written for (§9), and **both of §10's sign conventions, each
with an assertion written for it that passed while the flip was deleted**. Each
is written up where it happened.

**The §9 one is the sharpest, because writing a test FOR a bug does not make it
able to see the bug.** The fixture ran 6 regions then 1; the workgroup is 64
threads wide, so both runs wrote the same 64 slots and the stale tail beyond them
was never touched by either. Five of these six are the same shape — **a fixture
sitting exactly on the symmetry the mutation acts on** — so the question to ask
of any new fixture is not "does it exercise the code" but "could this input
possibly come out differently under the bug".

**§10 adds the recipe for answering that question instead of guessing it.** Both
its flips are corrections to an arbitrary-but-deterministic sign, so the way to
find a fixture is mechanical: *disable the correction and search the input space
for a case where the raw output is wrong.* 132 poses, of which 0 of the 77 inside
the normal operating range reach either branch. That also gives the honest
reachability statement for free, which is the thing "no test caught it" cannot
distinguish on its own.

**A mutation nothing catches is either a missing fixture or a true property, and
you have to say which.** §8 produced one of each in a single run: the unsigned
dot needed a fixture built to ask the question, while deleting the log-sum-exp
rescale correctly changes nothing. Stopping at "no test caught it" would have
left a real hole and manufactured a fake one.

**A decisiveness gate must be measured, not chosen.** Before asserting that two
implementations agree exactly on an integer, have the twin report how close the
closest decision came to flipping, and compare that against the disagreement the
two sides actually exhibit. A tolerance tuned until the test passes makes the
tolerance the thing under test. `cpuGrow`'s `marginalPairs` and `cpuLsdFit`'s
three `closest*` margins are the pattern.

---

---

## 1. What the pipeline actually computes

A camera looks at a flat floor covered in a printed black-and-white pattern. The
pattern is a **De Bruijn torus**: a grid of binary cells (currently 144×144)
with the property that every 5×5 window of cells is unique across the whole
board. Look at any 5×5 patch and you know exactly where on the board you are.

From one grayscale photo of that floor, recover **where the camera is** (a 3D
position) and **which way it is pointing** (a quaternion).

It happens in three conceptual acts:

**Act I — orientation, from lines.** The printed grid produces long straight
edges in the image. Straight lines on a plane, seen by a perspective camera,
converge to vanishing points. If you find enough line segments and ask "what
pair of 3D plane orientations best explains all of these," you recover the
floor's two in-plane axes and its normal — the camera's orientation relative to
the floor, but **not** its height and **not** which cell it is over.

**Act II — scale, from periodicity.** The grid lines are evenly spaced in the
real world. Project them onto the recovered floor plane and they become a set of
points on a line with a regular spacing. Find that spacing (the *period*) and
you know how far away the floor is, because you know the real cell size. Find
the offset of the lattice (the *phase*) and you know where the cell boundaries
fall. Now you have height.

**Act III — position, from the pattern.** With orientation, height and phase you
can compute, for any cell of the floor lattice, exactly which image pixel it
lands on. Sample all of them, threshold to bits, and you have a grid of the
pattern as the camera sees it. Slide a 5×5 window over it, look each window up
in the De Bruijn table, and every window votes for "the board is anchored here."
The anchor with the most votes wins. That tells you which cell you are over, and
from there the world position falls out.

Each act depends entirely on the previous one. Orientation is needed to project
lines for the period search; period is needed to build the sample lattice; the
lattice is needed to decode. **This is why the pipeline is a straight line with
no branching and no reordering** — and why a failure at any point makes
everything downstream meaningless rather than degraded.

---

## 2. The rules this design commits to

These are constraints, not preferences. Each one deletes a category of code.

1. **One upload, one readback.** A grayscale image goes up. 128 bytes come back.
   Nothing crosses the PCIe bus in between. This is the rule that forces the
   host math onto the device and dissolves the timing DAG (with no stalls, there
   is nothing to attribute).
2. **No CPU fallback.** A failure sets a bit and the function returns a pose
   marked invalid. There is no second implementation to fall back *to*, which is
   what deletes ~2,255 lines of CPU reference math.
3. **No worst-case buffer sizes.** Where a count is provable and tight (member
   count ≤ pixel count) use it. Where the provable bound is 30× pessimistic, use
   a **fixed cap plus an overflow bit** instead.
4. **No arena.** Every buffer is a distinct `GPUBuffer`, allocated once per
   `(width, height)` and reused across frames. No offsets, no generation stamps,
   no bump pointer, no spill logic.
5. **The math leads.** Stage boundaries are drawn where the algorithm changes,
   not where a verification strategy or a memory-ownership scheme wanted a seam.
6. **Flat is one FILE, not one FUNCTION.** The pipeline is ~14 named
   `encodeX(device, bufs, enc)` functions living in one file, all writing into
   one encoder. Flatness means deleting the directory tree, the residency, the
   arena, the backend dispatch and the abstraction layers — *not* deleting
   function boundaries. This costs nothing at runtime and is what makes the file
   both readable and testable (§19). Build it in from line one; retrofitting it
   is a rewrite of the rewrite.
7. **A stage clears its own accumulators**, as the first commands it encodes —
   never from a frame-start list somewhere else. A global clear list is
   knowledge maintained apart from the stages that depend on it, so it rots the
   first time someone adds an accumulator. This also happens to be what makes
   buffer aliasing free (§18).

### What rule 4 buys, and what it costs

Buys: [arena.ts](src/pose/gpu/arena.ts)'s 327 lines collapse to a
`makeBuffers(device, w, h)` function returning a struct of named buffers. It
also **dissolves the WebGPU usage-scope question** entirely — the open worry
about whether one contiguous buffer may legally back a uniform binding, a
read-only-storage binding and a writable-storage binding within a single
dispatch. With separate buffers the question never arises.

Costs: **buffer reuse means every buffer starts each frame holding the previous
frame's bytes.** This is the same hazard the arena introduced, and it does not
go away — see §13. It is arguably *better* here, because with per-frame reuse
the hazard is obvious from the design rather than being an emergent property of
an allocator.

### The mechanism that replaces host early-returns

With no host in the middle of the pipeline, there is nothing to `return` from.
The replacement:

> **Every dynamically-extended dispatch is indirect. Any pass that detects a
> failure zeroes the downstream args triple it owns.**

A failure then propagates as zero-workgroup dispatches all the way to the end —
legal, ordered, and free — and the final one-thread pass reads the status word
and writes `ok = false`. No branching in the hot kernels, no status check at the
top of every shader.

This is not new. It is exactly what
[growRegions.wgsl.ts](src/pose/stages/lsd/growRegions.wgsl.ts)'s `gate` pass
already does for convergence, generalized to every failure point.

---

## 3. Notation

| symbol | meaning | value at 480×640 |
|---|---|---|
| `n` | pixel count | 307,200 |
| `n·4B` | one full-image f32 or u32 array | 1.172 MiB |
| `MAX_REGIONS` | cap on detected line-support regions | 16,384 |
| `MAX_LINES` | cap on accepted line segments (= `MAX_REGIONS`) | 16,384 |
| `MAX_CELLS` | cap on decode lattice cells | 262,144 |
| `R`, `C` | De Bruijn torus dimensions | 144 × 144 |
| `ORDER` | De Bruijn window size | 5 |

Sizes are MiB (1,048,576 bytes). Observed values (~5,200 regions, a 270×276
decode lattice) come from the existing implementation's own measurements.

---

## 4. Stage 0 — Upload

**What it does.** Moves the grayscale image to the device. The only host→device
transfer in the pipeline.

| buffer | size | MiB | contents |
|---|---|---|---|
| `gray` | `n × 4` | 1.172 | One f32 per pixel, 0–255, row-major |

**Why f32 and not u8.** The image is 8-bit, so this is 4× larger than necessary.
Three consumers read it: the gradient kernel, the decode build (one sample per
lattice cell), and the binarization-threshold reduction. Packing 4 pixels per
u32 would save 0.879 MiB at the cost of shift-and-mask in all three. Not worth
it at this footprint; revisit only if memory becomes binding.

**Note on the existing path.** Today `gray` arrives as a `Float64Array` and is
narrowed to `Float32Array` on the host before upload. That narrowing loop is
measured as part of the ~2.1 MiB of byte-proportional cost per reconstruction.
In the rewrite the entry point should take a `Float32Array` or a `Uint8Array`
directly and skip it.

---

## 5. Stage 1 — Gradient

**What it does.** Computes the image gradient — how fast brightness changes, and
in which direction — at every pixel. This is the raw material for everything in
Act I: an edge in the image is a place where the gradient is large.

**The math.** A 2×2 block gradient. For each pixel, look at it and its right,
below, and below-right neighbours, and take the difference of the column sums
(horizontal derivative) and the row sums (vertical derivative), normalized by
`1/510` so that gradient magnitude tops out at 1.0 rather than 255.

Existing implementation:
[gradient2x2.wgsl.ts](src/pose/stages/gradient/gradient2x2.wgsl.ts).

| buffer | size | MiB | contents |
|---|---|---|---|
| `fx` | `n × 4` | 1.172 | Horizontal gradient component, f32 |
| `fy` | `n × 4` | 1.172 | Vertical gradient component, f32 |

**Subtotal: 2.344 MiB.**

**The one subtlety.** The last row and column are zeroed (a 2×2 block needs a
right and below neighbour), but the *first* row and column are valid data. A
symmetric-margin gradient kernel would zero all four edges and silently discard
real data along the top and left. The existing file's header calls this out
specifically as a copy-paste trap.

**Possible saving.** f16 halves this to 1.172 MiB. Worth noting that the
grow-stage predicate is already known to disagree between f32 (GPU) and f64
(CPU) near the tolerance boundary, so precision here is a live question, not a
settled one. Do not do this until the pipeline is trusted.

---

## 6. Stage 2 — Grow (connected-component region growing)

**What it does.** Groups pixels into *regions* that plausibly belong to the same
straight edge. This is the first half of LSD (Line Segment Detection).

**The math, and why it is unusual.** The standard approach is a flood fill from
seed pixels, which is inherently serial. This uses **directed connected
components via pointer jumping** instead, which is fully parallel:

- Every pixel with gradient magnitude above `rhoLow` starts as its own region,
  labelled with its own pixel index. (Dense seeding — with a symmetric edge
  predicate the choice of seeds cannot affect the outcome, so there is no
  "which pixels get to seed" decision to make.)
- Each round, every pixel looks at its 8 neighbours and adopts the smallest
  label among those it is *compatible* with. Compatible means their **level-line
  directions** — the direction perpendicular to the gradient, i.e. along the
  edge — agree to within a tolerance.
- Then every pixel follows its label's own label one step (`label[i] =
  next[next[i]]`). This is pointer jumping: it halves chain depth per round, so
  a component of any size collapses in O(log n) rounds rather than O(diameter).

The compatibility test is a **signed** dot product with no `abs()`. That is
load-bearing: the two sides of a thin dark stripe have *antiparallel* level-line
directions, and an unsigned test would fuse them into one region. The directed
test keeps them apart.

Existing implementation:
[growRegions.wgsl.ts](src/pose/stages/lsd/growRegions.wgsl.ts).

| buffer | size | MiB | contents |
|---|---|---|---|
| `ux` | `n × 4` | 1.172 | Level-line direction, x component. `(-fy, fx)` normalized |
| `uy` | `n × 4` | 1.172 | Level-line direction, y component |
| `label` | `n × 4` | 1.172 | Current region label per pixel (i32). `-1` = ineligible |
| `next` | `n × 4` | 1.172 | Next round's label, written by `hook`, read by `compress` |
| `changed` | 4 B | — | Did any label change this round? Atomic flag |
| `args` | 16 B | — | `[workgroupsX, workgroupsY, 1, activeRounds]` |

**Subtotal: 4.688 MiB.**

### Buffer-by-buffer

**`ux`/`uy`** exist so the direction is computed once in `init` rather than
re-derived every round from `fx`/`fy`. The component order `(-fy, fx)` is easy
to get wrong and *was* wrong in this codebase for a period — the header records
that growing stayed bit-identical while the rectangle fit did not, which is a
good example of a bug that one consumer is blind to.

**`label` and `next` must be separate.** `hook` reads only `label` and writes
only `next`; `compress` reads all of `next` and writes only its own `label[i]`.
Neither has a same-round cross-pixel write dependency, which is what makes each
one a single trivially-parallel dispatch. Collapsing them into one buffer would
introduce exactly that dependency.

**They must also be separate compute passes**, not two dispatches in one pass.
`compress` reads `next[l]` for an arbitrary `l` that some *other* thread wrote
during `hook`. WebGPU guarantees a storage-buffer memory barrier between
*passes* in an encoder, not between dispatches inside one pass.

**`args` is the convergence mechanism.** `hook` and `compress` dispatch
*indirectly* off it. A one-thread `gate` pass runs after `compress`, and if
nothing changed it writes `[0,0,0]` — so every subsequent round is a
zero-workgroup dispatch. That is legal, ordered, and costs nothing but its own
encoding.

**`args` must live in its own bind group.** A buffer cannot be both a writable
storage binding and the indirect source of the same dispatch — that is a
usage-scope conflict, reported asynchronously, whose symptom is a silently
no-op encoder rather than an exception. `gate` binds it in group 1; `hook` and
`compress` never bind it at all.

### The one thing the zero-crossing rule costs here

Today the host reads `args.x` once per 16-round batch to decide whether to
encode another batch. **With no readback, the round count must be fixed at
encode time.**

- Measured: **9 rounds** at 480×640 on the default fixture.
- Theoretical: O(log n) ≈ 19.
- Hard cap (worst-case pointer-jumping depth): 1,184 rounds = 3,552 passes.
  Almost certainly too many to encode per frame.

**Decision: encode 32 rounds, and set a `growNotConverged` status bit if
`args.x != 0` after the last one.** Rounds past the fixpoint are
zero-workgroup and nearly free to run (though real to *encode* — 32 rounds is 96
passes). Non-convergence becomes a reported frame failure rather than a silently
wrong labelling. This is the honest version of "early return or throw."

### Possible saving

`ux`/`uy` are pure functions of `fx`/`fy`. Dropping them and normalizing inline
costs one `inverseSqrt` per neighbour test per round and saves 2.344 MiB. A real
trade, not obviously worth taking; noted, not recommended.

---

## 7. Stage 3 — Collect

**What it does.** Turns a finished labelling into a clean, ordered list of
regions. Three jobs: apply **hysteresis** (drop regions whose pixels are all
weak), apply a **size filter** (drop regions with too few pixels), and build a
**CSR structure** (compressed sparse row — a flat `members` array plus per-region
offsets and sizes) so downstream stages can iterate one region's pixels
contiguously.

**Why hysteresis.** Grow uses a *low* threshold `rhoLow` so that faint parts of
a real edge still join the region. That also admits regions made entirely of
noise. The fix is Canny's: keep a region only if at least one of its pixels
clears a *high* threshold `rhoHigh`. Strong evidence anywhere vouches for the
whole connected region.

**The critical structural fact: labels ARE pixel indices.** A region's label is
the pixel index of its CCL root. So every per-label array must have `n` slots,
even though only ~5,200 of them ever become regions. This is deliberate — it is
the price of never hashing or compacting the label space — and it is why this
stage dominates the memory footprint. It also has a *benefit*: a prefix scan
over labels visits them in ascending pixel-index order, which reproduces the CPU
reference's region ordering exactly rather than merely equivalently.

Existing implementation:
[collectRegions.wgsl.ts](src/pose/stages/lsd/collectRegions.wgsl.ts) — described
in its own header as "the model implementation."

### REDESIGNED WHEN IT WAS BUILT, 2026-08-12

The table below originally described six passes and two prefix scans, taken
from reading the existing implementation. Implementing it meant re-deriving it,
and most of that structure turned out to be inherited rather than required.
**What was built is five passes and one scan.** The reasoning is worth keeping,
because the same three moves apply to the stages still to come.

**1. One vec2 scan instead of two u32 scans.** The stage needed an exclusive
prefix sum of `keptFlag` (giving each region its dense id) and another of
`keptCount` (giving each region its CSR offset) — over the same array, in the
same order, differing only in the accumulator. Scanning `vec2<u32>` does both
lanes in one traversal at *identical* memory cost, and its grand total is
exactly `[regionCount, memberCount]`. So `totalRegions`, `totalMembers` and the
`collect.counts` copy all disappear: the scan spine writes the pair itself.

The same collapse applies twice more — `gpp` scanned `isRow` and then `1-isRow`,
whose totals are `[rowCount, colCount]`. **Five declared scans became three
uses of one primitive.**

**2. The scan is two levels, not a recursion.** One workgroup is 256 threads and
each thread scans 4 contiguous elements, so a workgroup covers 1024 elements and
307,200 pixels is 300 blocks — which fit inside a *single* workgroup. Three
passes: block, spine, add. No level tree, no recursion depth to discover, and
none of the "a caller cannot enumerate these temporaries without reimplementing
the recursion" problem the original note made much of.

The ceiling is real and lower than it first looks: one spine workgroup covers
256 × 16 = 4096 blocks, so 4.2M pixels, i.e. 2048×2048. `planPool` throws past
that rather than silently returning partial sums. **A first draft of this
document claimed the bound held at 4096×4096. It does not — that needs 16,384
blocks.** The test caught it.

**3. `survive` and `histogram` were one pass.** They walked the same pixels and
read the same label; the split let the histogram skip labels that had already
failed hysteresis, which `markKept` ignores anyway. It bought nothing and cost a
pass plus a false dependency.

**A zero-initialization bug was found in the declaration itself.**
`labelSurvives` is only ever written with `1`, by strong pixels — a weak pixel
writing `0` would race a strong one. So a label with no strong pixel *inherits
last frame's 1* and a pure-noise region survives hysteresis. It was not marked
`zero` in the original table. This is the third member of that family (after
`growArgs` and the atomic accumulators) and the first one found by re-deriving
rather than by a test.

**`cursor` is indexed by region, not by label.** Same job, and it turns an
n-sized buffer (1.17 MiB) into a maxRegions-sized one (16 KiB).

### The five passes and one scan, as built

| # | pass | what it does |
|---|---|---|
| 1 | `tally` | Per pixel: `atomicAdd` its label's count, and mark the label surviving if the pixel clears `rhoHigh` |
| 2 | `markKept` | Per label: `kept = vec2(survives && count >= min, that ? count : 0)` — the scan input |
| — | scan ×3 | One exclusive `vec2<u32>` prefix sum → `(regionId, memberOffset)` per label; grand total → `counts` |
| 3 | `regionMeta` | Re-index per-label arrays into per-region ones. Thread 0 writes the indirect args **before the guards**, or it is skipped whenever label 0 is not a region — which is nearly always |
| 4 | `scatter` | Per pixel: place its index into the CSR via an atomic per-region cursor |
| 5 | `finalize` | One thread per region: insertion-sort its member slice, then average its members' level-line directions |

| buffer | size | MiB | contents |
|---|---|---|---|
| `labelSurvives` | `n × 4` | 1.172 | Per label: does any member clear `rhoHigh`? **Cleared each frame** |
| `labelCounts` | `n × 4` | 1.172 | Per label: pixel count. **atomicAdd target**, cleared each frame |
| `kept` | `n × 8` | 2.344 | `vec2<u32>` (keptFlag, keptFlag ? count : 0) — the scan input |
| `keptScan` | `n × 8` | 2.344 | `vec2<u32>` exclusive prefix — (regionId, memberOffset) |
| `keptSums` / `keptOffs` | `300 × 8` each | 0.005 | Block sums and block offsets. A scan reads its source while writing its destination, so these cannot be one buffer |
| `members` | `n × 4` | 1.172 | The CSR payload: pixel indices, grouped by region |
| `cursor` | `MAX_REGIONS × 4` | 0.063 | Per **region** write cursor. **atomicAdd target**, cleared each frame |
| `regionOffsets` | `MAX_REGIONS × 4` | 0.063 | Per region: start index into `members` |
| `regionSizes` | `MAX_REGIONS × 4` | 0.063 | Per region: member count |
| `meanDirs` | `MAX_REGIONS × 8` | 0.125 | Per region: mean level-line direction |
| `counts` | 8 B | — | `[regionCount, memberCount]`, written by the scan spine |
| `regionArgs` | 12 B | — | `(ceil(regionCount/64), 1, 1)` for per-region dispatches |

**Subtotal: 8.52 MiB**, down from 9.66 — still the largest stage in the
pipeline. The vec2 pair costs exactly what the four `n × 4` arrays it replaced
did; the saving is `cursor` moving to a per-region index, plus the two scalar
totals and the second scan's temporaries going away entirely.

`finalize` recomputes the level-line direction from `fx`/`fy` rather than reading
`ux`/`uy`. Identical value, one `inverseSqrt` per member, and it lets `ux`/`uy`
die at the end of grow instead of staying live across all of collect — two
full-image buffers of pool pressure.

**The sort is not optional.** The atomic cursor hands out slots in arrival
order, which is nondeterministic; the sort is where this stage's determinism is
restored rather than arranged.

**Why not skip the scan entirely?** An atomic append (`regionId =
atomicAdd(counter, 1)`) would replace `markKept`, the scan and `regionMeta` with
a single pass. It was rejected: region ids would then depend on GPU scheduling,
back-to-back runs would stop agreeing bit-for-bit, and the CPU twin would have to
canonicalize before it could compare anything. **Determinism is worth four
passes**, and it is what lets the twin test assert region-for-region equality.

### Buffer-by-buffer notes

**`regionId` is dropped, and it is free.** Verified 2026-08-12: nothing on the
pose path reads it. Its only consumers are the hover overlay
([lsdOverlay.ts:159](src/sphereLab/overlays/lsdOverlay.ts#L159)) and the verify
harnesses. Dropping it saves 1.172 MiB *and* removes a write from the scatter
pass. If an overlay ever needs it back, it is a one-line addition to `scatter`
plus a pinned pool slot (§18).

**`regionOffsets` and `regionSizes` are `MAX_REGIONS`, not `n`.** In the current
code they are allocated at `n × 4` and indexed by *region* — a straightforward
oversizing worth 2.34 MiB → 0.125 MiB. Same bound `meanDirs` already uses.

**`members` needs no cap.** Regions are a disjoint partition of the pixels, so
the member total is bounded by `n` exactly. This is the one device-side count
that is free.

**`meanDirs` is a direction, never an angle.** It is a plain normalized sum of
member unit vectors, with no sign resolution against a reference member — with
*directed* growth every member is within tolerance of every member it is
connected through, so there is no polarity flip for the sum to cancel against.

---

## 8. Stage 4 — Fit rectangles + NFA validation

**What it does.** Turns each region (a blob of pixels) into a **line segment** (a
centre, an angle, a length, a width), then decides whether it is a real line or
an accident of noise.

**The math, part 1 — the rectangle fit.** Magnitude-weighted PCA. Compute the
weighted centroid of the region's pixels, then the weighted second-moment matrix,
then the principal axis via `0.5 * atan2(2·Ixy, Ixx − Iyy)`. The PCA axis is only
defined up to 180°, so the region's own `meanDirs` entry picks which way it
points. Project every member onto the axis and its perpendicular to get the
extent, and you have an oriented rectangle.

**The math, part 2 — NFA (Number of False Alarms).** This is the part that makes
LSD principled rather than heuristic. Ask: *if this image were pure noise, how
many rectangles this good would I expect to see by chance?* Count `n` = pixels
inside the rectangle footprint and `k` = how many of those have a level-line
direction aligned with the rectangle's axis. Under a null hypothesis of uniformly
random gradient directions, the probability of a pixel being aligned is
`p = tolerance / π`, so `k` aligned out of `n` follows a binomial. The tail
probability `P(X ≥ k)` times the number of tests gives the expected false-alarm
count. Accept if it is below `epsilon`. **No threshold tuning** — the acceptance
criterion falls out of the statistics.

Existing implementation:
[lsdFit.wgsl.ts](src/pose/stages/lsd/lsdFit.wgsl.ts). One thread per region;
regions are independent by construction.

| buffer | size | MiB | contents |
|---|---|---|---|
| `rects` | `MAX_REGIONS × 40` | 0.625 | 10 f32 per region (see below) |

**Subtotal: 0.625 MiB.** Currently 6.14 MiB — allocated at the provable bound
`n / minRegionSize` = 153,600 regions.

**The 10-float stride:** `cx, cy, theta, length, width, nfaLog10, accepted,
pad, n, k`. The `n` and `k` are diagnostic — they exist so a disagreement in the
*counts* (which pixels each side thought were inside/aligned) can be told apart
from a disagreement in the tail *arithmetic*.

### Notes

**This is the single biggest memory saving in the rewrite, and it arrives for
free.** The current code allocates at the provable bound because bounding it to
a cap would mean the shader (which dispatches over `regionCount` and writes
`out[r]`) could write out of bounds — silently discarded in WGSL, so rectangles
would vanish with no signal. That needs a device-side guard and an overflow flag
the host reads. In the current architecture that is new machinery. **In this
design every dynamic buffer already has a cap and there is already one status
word**, so it costs nothing extra.

**The binomial tail is computed as an online log-sum-exp**, rescaling on each new
maximum, rather than the two-pass collect-then-reduce the CPU version uses. WGSL
has no dynamic-length local arrays and `n − k` is unbounded, so terms cannot be
buffered. Numerically equivalent, standard technique.

**The degenerate-region path must zero all ten slots.** A region with fewer than
2 members has no meaningful axis. The current code's early return writes only
`out[o+6] = 0`, which under buffer reuse leaves the previous frame's rectangle in
the other nine slots — and that rectangle is still returned and still drawn.
**The shape to watch for generally: an early return that writes one field.**

**The retry loop is not implemented and should not be.** Classic LSD, on NFA
rejection, tightens the tolerance and re-tests, then shrinks the region by
dropping the pixels farthest from its centre. Retry 2+ needs a per-region partial
sort — a genuinely harder GPU problem than anything else here. It is retired on
the CPU side too, so this kernel is the whole fitter, not a partial port.

### BUILT 2026-08-12, and the decomposition survived re-derivation

Unlike §7, **this stage came out of re-derivation exactly as declared: one pass,
one thread per region.** Worth saying why, because "the declaration was right"
is a result and not a non-event. Everything inside is serial and each step needs
the previous one's answer — centroid, then central moments, then the extents
along the axis those moments define — and regions are independent by
construction, so there is nothing to split and nothing to fuse. The only fusion
available is *forward*, into §9's `lines.flag`, and that is §9's call to make.

Two guards are new, and neither is in the old kernel:

- **The thread guard is `min(counts.x, maxRegions)`.** `rects` is capped now
  rather than allocated at the provable bound, and `regionMeta`'s clamp only
  bounds the *workgroup* count — with `maxRegions` not a multiple of 64 the last
  workgroup still runs threads past the cap. Belt and braces rather than
  load-bearing: WGSL discards an out-of-bounds store and clamps an out-of-bounds
  load, so the uncapped version wastes threads rather than corrupting anything.
  Which is exactly why it is written down — nothing would have made it visible.
- **A zero total weight is degenerate.** `sumW` is a sum of magnitudes, so it is
  zero only if every member has none — impossible while `rhoLow > 0`, and a
  silent NaN centroid the moment it is not. The old kernel divided anyway.

**The twin re-derives two of the three steps and cannot re-derive the third.**
The axis comes from solving the 2×2 eigenproblem and taking `atan2` of the major
eigenvector, not from the half-angle identity — so a wrong factor of two or a
swapped `atan2` argument has nowhere to hide. The tail is the textbook two-pass
log-sum-exp against the shader's online rescale. But *"pixels whose centre lies
inside this rectangle"* admits one formulation, and inventing a second would test
a rasterizer rather than the fit — so the footprint count is written from the
definition and carries the same two constants. `logBinomialTail` is itself
checked against an **exact** binomial sum at small `n`, which is the only place
in this stage where an oracle is checked by arithmetic rather than by a second
opinion.

### THE MUTATION RUN, and the two things it found

Seven deliberate shader bugs. Five were caught; the two that were not are the
findings.

| mutation | caught by |
|---|---|
| swapped `atan2` arguments | everything |
| no 180° disambiguation | the twin, and the polarity fixture |
| degenerate writes only the accepted flag | the degenerate fixture |
| dropping `BOUNDARY_EPS` | **the twin frame only** |
| centre at the centroid, not the extent midpoint | **the twin frame only** |
| unsigned alignment dot | **NOTHING — a test was built** |
| deleting the log-sum-exp rescale | **NOTHING — and correctly so** |

**The unsigned-dot bug was invisible, and the reason is structural.** `hw` is
half the region's own width, so a footprint is essentially the region's own
pixels — and directed growth already made those one polarity. Nothing
anti-aligned is ever *inside* a footprint on real data, so `abs()` changes no
count anywhere, including across a whole rendered frame. This is the same shape
as §19's four-pixel bar: a fixture that cannot ask the question it claims to.
The fix is a hand-built 5×3 block with an anti-aligned pixel and two sub-`rho`
pixels placed *inside* the footprint, which pins `n = 15` and `k = 12` and
separates three behaviours — `n` is geometric, `k` excludes anti-aligned, `k`
excludes weak. Unsigned gives 13; no `rho` gate gives 14.

**The log-sum-exp rescale is overflow protection and nothing else.** Deleting it
changes not one number, and that is not a test gap: log-sum-exp is exact against
*any* reference point, so the running maximum only ever stops `exp` overflowing
f32 at 88 nats. Reaching that needs the max term e⁸⁸ above the k-th, i.e. `k`
about 13σ below the mean `np` — at `n = 10,000` footprint pixels that means
`k < 231`, and below `n ≈ 1,000` it cannot happen at all. So it takes a large
blob region with almost nothing aligned: reachable at 480×640, absent from every
fixture here, and the wrong answer would be an `inf` that rejects a region the
correct arithmetic also rejects at `nfaEpsilon = 1`. The branch stays. **Nothing
in the suite proves it right, and that is recorded rather than papered over.**

**The two implementations report `theta` 2π apart, and both are correct.** The
half-angle form lands in `[-π/2, 3π/2]`; `atan2` of an eigenvector whose sign is
arbitrary lands in `[-π, 2π]`. Everything downstream uses `theta` through `cos`
and `sin`, so it is an angle and the comparison wraps. A genuine disambiguation
failure is π away and survives the wrap at full size.

**The decisiveness gates are measured, not chosen.** Asserting that two
implementations agree exactly on the integers `n` and `k` is worthless unless no
pixel sat where they could legitimately differ — so the twin reports the *closest
call* on each decision, and the test compares it against the disagreement the two
sides actually exhibit (`2 × dGeom` for inclusion, `dTheta` for alignment) rather
than against a tolerance tuned until it passed. Observed on a 96×128 frame:
inclusion margin 6.9e-4, **alignment margin 2.0e-6**, accept margin 0.35, minimum
anisotropy 0.47. The alignment one is the tight one and always will be — nothing
protects that boundary the way `BOUNDARY_EPS` protects inclusion, because
level-line direction varies continuously and pixels land wherever they land.

---

## 9. Stage 5 — Lines and votes  *(BUILT 2026-08-12)*

**What it does.** Filters the accepted rectangles down to usable line segments,
then converts each one into a **vote** about the floor plane's orientation.

**The math.** A line segment in the image, back-projected through the camera,
sweeps out a plane through the camera centre. That plane's normal is the cross
product of the rays to the segment's two endpoints. If the segment is really a
straight line on the floor, that normal is perpendicular to the floor's
direction along the line. So each segment casts one vote: *"the floor contains a
direction perpendicular to me."*

The vote's **weight** is the magnitude of that cross product, which equals the
sine of the angle between the two endpoint rays — the segment's projected arc
length on the unit sphere. Long, well-separated endpoints give a large,
well-conditioned normal; a short segment gives a small one easily corrupted by
noise. So the weight is exactly a confidence measure, and it is an explicit
per-line quantity rather than something buried in a per-pixel step.

Currently on the host:
[votes.ts](src/pose/stages/votes/votes.ts) — `compositesFromLsdRectangles` and
`computeSegmentVotes`.

| buffer | size | MiB | contents |
|---|---|---|---|
| `lineFlag` | `MAX_REGIONS × 8` | 0.125 | `vec2<u32>` (accepted && length >= lsdMinLengthPx, 0) — the scan input |
| `lineScan` | `MAX_REGIONS × 8` | 0.125 | `vec2<u32>` exclusive prefix; `.x` is the compaction index |
| `lineSums`, `lineOffs` | `2 × blocks × 8` | — | Scan block temporaries |
| `lineCount` | 8 B | — | `vec2<u32>`; `.x` is the line count, written by the scan spine |
| `lines` | `MAX_LINES × 16` | 0.250 | `(x1, y1, x2, y2)` in pixel space |
| `votes` | `MAX_LINES × 16` | 0.250 | `(nx, ny, nz, weight)` — the plane normal and its confidence |
| `maxWeight` | 4 B | — | Frame maximum, for normalization. Atomic max |
| `lineArgs` | 12 B | — | Indirect args for per-line dispatches |

**Subtotal: 0.75 MiB.**

**This stage uses the shared vec2 scan** (§7), already built. It is the one use
that wastes its `.y` lane — `MAX_REGIONS × 4` bytes, in exchange for having one
scan implementation rather than two.

### THE `lines.flag` DECISION — SETTLED 2026-08-12: IT STAYS, AND IT GOES DIRECT

The question was whether `lines.flag` needs to exist at all, given it computes
`accepted && len >= lsdMinLengthPx` from two fields `lsdFit` just built and still
has in registers. **For:** every fusion in this pipeline is a bus-crossing
fusion, so the pass buys nothing but a name. **Against:** rule 5 — NFA acceptance
is the fitter's verdict, the length floor is this stage's usability filter.

**Rule 5 wins, but it is not what decided it.** Re-deriving the stage produced a
mechanical argument that settles it without appeal to taste, and it found a live
bug in the declaration on the way:

> `lineFlag` is a SCAN INPUT, and a scan's element count is fixed at encode time
> — the host cannot know the region count, so the scan runs over `MAX_REGIONS`.
> That makes `lineFlag` a **total function** over `[0, MAX_REGIONS)`: every slot
> must be written every frame. `lsdFit` is a map over `[0, regionCount)`,
> dispatched off `regionArgs`. A store at the bottom of that kernel can never
> reach the tail.

So the tail past the region count would hold **last frame's flags**, and a quiet
frame after a busy one would emit lines from stale rectangles. `lineFlag` was not
marked `zero` in the declaration — the same family as §7's `labelSurvives`, found
the same way, and the second one re-derivation has caught that a test had not.

The fix is better than a clear: **`lines.flag` dispatches DIRECTLY over
`MAX_REGIONS`** and writes 0 past the region count, which makes it total by
construction and needs no clear at all. Fusing cannot do this at any price, and
would additionally need both `zeroRect` early-return paths to write the flag.

**The general form, which is the counterexample worth keeping:** §20 records that
every fusion here is a bus-crossing fusion and therefore free. That is true and
its converse is not — **a fusion that changes a pass's ITERATION DOMAIN is not
free, even when both passes are already on the same side of the bus.**

Failure still propagates with no indirect dispatch: zero regions means every flag
is 0, so the scan total is 0, `lines.emit` writes zero workgroups into `lineArgs`,
and everything downstream dispatches nothing. Note the scan is over `MAX_REGIONS`
either way — fusing would have removed a dispatch, not a scan.

**The same defect is still live in §11.** `family` has the identical obligation
and `gpp.classify` as declared does not meet it: it dispatches off `lineArgs`, so
it writes `[0, lineCount)` while the scan runs over `MAX_LINES`. Whoever builds
gpp must either dispatch it directly with an interior guard or mark `family`
`zero`. Noted in `pipeline.ts` at the buffer.

### BUILT 2026-08-12

Five passes: `lines.flag`, the shared vec2 scan ×3, `lines.emit`, then
`votes.cast`. Two things worth not re-deriving:

**`votes.cast` takes no orientation, and that is the clarifying framing.** The
host version's signature takes a quaternion and every call site passes
`MATH_QUAT`, which is the identity — so the argument carried no information. The
rays are cast in camera space and the camera's orientation is precisely what §10
derives *from* these votes; a vote that already knew it would be circular.

**A degenerate vote writes weight 0 rather than being dropped.** The host
`continue`s, which renumbers everything after it; compaction is not available
here without a second scan. Zero weight is equivalent for the fit (the scatter
matrix skips it) and keeps vote `i` joined to line `i`. **Every consumer must gate
on `weight > 0`** — gpp classifies lines by their vote normal, and a zero normal
classifies into a family arbitrarily.

### THE MUTATION RUN, and what it found

Eight deliberate bugs. Six caught, and the two that were not are written up in
`VOTES_WGSL`'s header rather than left implied.

| mutation | caught by |
|---|---|
| `lines.flag` dispatches indirectly off `regionArgs` | the TOTAL test — **on the second attempt** |
| `lines.flag` ignores the region count | everything |
| `lines.emit` puts both endpoints at `+hl` | the endpoint test, both vote tests |
| `lines.emit` writes its args after the guards | three tests |
| `lines.flag` uses `>` not `>=` on the length floor | the length-floor test |
| drop the aspect ratio from the NDC map | **the rendered frame only** |
| do not flip `ndcV` | **the rendered frame only** |
| delete the degenerate arc guard | **NOTHING — and correctly so** |

**The first version of the TOTAL test was green under its own mutation**, and the
reason is structural in the same way §8's unsigned dot was. It ran 6 stripes then
1, and the workgroup is 64 threads wide — so *both* runs dispatched one workgroup
and wrote slots 0..63. The stale tail starts at 64 and neither run reached it: a
fixture that cannot ask the question it claims to, for the third time in this
rewrite. The question needs run 1 to flag a slot run 2's dispatch **cannot**
reach, which is now two sub-cases — 576 regions then 1, and regions then none.

**The hand-derived vote fixture is blind to the aspect and to the `ndcV` sense**,
and the comment claiming otherwise has been corrected in place. It sits exactly
on the symmetry each acts on: dropping `aspect` multiplies `ndcU`, which is 0 for
a centred line (and `DIMS` is square anyway), while flipping `ndcV` mirrors the
endpoints and merely negates the normal, which an `|nx|` assertion cannot see.
Both are caught by ground truth, so this is a correction rather than a hole — the
hand fixture pins the *algebra*, the rendered frame pins the *projection*.

**The degenerate arc guard is unreachable, and that is a true property.** A
pinhole projection is injective on image points and `lines.flag` has already
dropped anything shorter than `lsdMinLengthPx`, so the two endpoint rays are
separated by a definite angle and `arc` has a hard lower bound. It stays for the
same reason §8's log-sum-exp rescale does — `lsdMinLengthPx` is a slider that can
be set to 0, and the failure would be a NaN normal silently poisoning the fit —
and nothing here proves it right.

**Two outputs are written and unobserved:** `maxWeight` (its only reader is
`fit.ata`, §10) and the `lineOverflow` bit (no fixture approaches `MAX_LINES`).

### THE FIRST GROUND-TRUTH SCORE IN THE PIPELINE

§19's table says votes are scored by ground truth rather than by the twin, and
this is where that starts. The claim under test: a line detected on the printed
floor lies along one of the two grid families, so the plane it sweeps through the
camera centre **contains** that family's floor direction — and the vote normal is
therefore perpendicular to `truth.DrowMath` or `truth.DcolMath`.

Measured on a rendered 96×128 frame at height 10, tilt 20°, yaw 15°:

```
68 votes   median residual 8.1e-3 (0.47 deg)   worst 3.5e-2 (2.0 deg)
           families 36/32                      degenerate 0
```

The gate is decisive rather than tuned: a normal perpendicular to *neither* axis
— what a dropped aspect or a flipped `ndcV` produces — lands in the tenths,
because the two families are 90° apart and there is nowhere small for it to go.
The thresholds sit an order of magnitude inside that gap. The honest caveat is
that `worst < 0.05` has only ~1.4× headroom and will go flaky first; `median` is
the load-bearing assertion.

**This also discharges the geometric half of `lsdFit`'s oracle** (open decision
8). A rectangle that is self-consistently wrong — right shape, wrong place —
passes every twin comparison and fails here.

### Notes

**This needs a third prefix scan** — over `MAX_REGIONS` rather than `n`, so it is
cheap (64 blocks → 1, two levels).

**One abstraction disappears here, and it is exactly the kind this exercise is
meant to find.** The current `compositesFromLsdRectangles` keeps a `root`
counter that increments for every *accepted* rectangle but only emits those
passing the length filter — so `root` is not the output index. It exists purely
so `gridPeriodPhase` can join back against a differently-indexed set. In a flat
pipeline both consumers read the same compacted array, so **`root` disappears
entirely.** It is an artifact of two consumers indexing differently, not
anything the math needs.

**`maxWeight` normalization is a no-op for the fit** — scaling every weight by a
constant scales the scatter matrix by that constant and leaves its eigenvectors
untouched. It is kept because the accumulation is in f32 and wants summands in
`[0, 1]`.

**The name "composite" is historical.** It once meant several raw segments
merged into one line by a join walk. That walk is retired, so today a composite
is exactly one accepted rectangle's own two endpoints and nothing is composed.
**The rewrite should rename this to `lines`** — the name is only threaded
through the app's wire format, which is not in scope here.

---

## 10. Stage 6 — Fit pair of planes  *(BUILT 2026-08-13)*

**What it does.** Takes all the votes and recovers the floor's two in-plane axes
and its normal. This is the end of Act I.

**The math.** The votes fall into two families — normals from row lines and
normals from column lines — but nothing has labelled which is which. Rather than
cluster them first, fit a **degenerate quadric**: the algebraic surface that best
expresses "every vote lies on one plane *or* the other." A pair of planes through
the origin is the zero set of a quadratic form, so this is a least-squares
problem in the 6 unique coefficients of a symmetric 3×3 form.

Build the weighted 6×6 scatter matrix `ATA` over the votes, take its smallest
eigenvector (the coefficient vector with least residual), reshape to a 3×3, and
its eigendecomposition gives the two plane normals. Their cross product is the
floor normal.

Existing implementations:
[fitPlanes.wgsl.ts](src/pose/stages/votes/fitPlanes.wgsl.ts) for the reduction,
`planesFromScatter` in [votes.ts](src/pose/stages/votes/votes.ts) for the tail.

| buffer | size | MiB | contents |
|---|---|---|---|
| ~~`ataPartials`~~ | ~~`ceil(MAX_LINES/64) × 21 × 4`~~ | ~~0.021~~ | **DELETED** — see below |
| `ata` | 84 B | — | The 21 upper-triangle entries |
| `triad` | 48 B | — | `Drow`, `Dcol`, `Dnormal` as three `vec3<f32>`, stride 16 |

**Subtotal: 132 bytes.**

### Notes

**Why 21 and not 36.** `ATA` is symmetric, so only the upper triangle is unique:
`6·7/2 = 21`. The packing order is `a` outer, `b` inner starting at `a`, and the
unpack must match exactly.

**There is a known duplication history here worth not repeating.** These two
halves were split once before (for a since-deleted IRLS refinement), merged when
it went, and re-split on a different argument — the GPU path had been carrying a
verbatim 20-line copy of the eigen tail. Say so before merging them again. The
port hits the same pressure from a new direction, because WGSL has no generics
and the host's one Jacobi serves two sizes; the answer is below.

### BUILT 2026-08-13, and the two-pass decomposition did NOT survive

**`fit.reduce` and `ataPartials` are both gone.** The declaration inherited a
split whose only justification expired with the host:

- **The partials existed so a HOST could sum them.** src/pose tree-reduces 64
  votes per workgroup into that workgroup's own 21-float row and reads ~256 rows
  back. With no readback there is nothing to hand rows to, so the intermediate
  stops having a job — the same finding as §18's, that intermediates are a
  symptom of an owner outside the pipeline.
- **And the indirect form carried a live bug.** `fit.ata` was declared
  `indirectFrom: 'lineArgs'`, so it wrote `ceil(lineCount/64)` rows while
  `fit.reduce`'s extent was fixed at encode time. The tail is last frame's
  partial sums, so a busy frame followed by a quiet one fits the union of the
  two. That is §9's iteration-domain defect for the third time, and the first
  outside a scan.

What replaced it is **one workgroup of 64 lanes striding over the votes**,
accumulating 21 floats in registers and tree-reducing once. `ata` is written in
full by lane 0, so it is total by construction and needs no clear.

**What that costs, stated rather than waved past.** At the `maxLines` cap this is
16,384 votes on 64 lanes — 256 votes and ~5,400 FMAs serially per lane, on ONE
compute unit, a few microseconds. Observed counts are 68–485. It is a real
serialization and the two-pass form is the answer if it ever appears in a
profile, but with the domain fix the deleted version did not have.

**The eigen tail is one Jacobi, parameterized by `n`, on a shared 6×6 arena.**
The tempting port of a generic host routine into a language without generics is
two copies at two sizes — which is exactly the duplication this section already
records happening once. It is avoidable: the routine takes `n` and indexes at
stride `n` inside one `array<f32,36>`, so the 3×3 call reuses the first 9 slots.

**Three things in it are not transcriptions**, and each is a place f64 and f32
differ rather than a preference:

1. **The convergence test is relative.** `linalg.ts` breaks at an off-diagonal
   sum of squares below `1e-30`, which f32 cannot reach for a matrix of ordinary
   magnitude — the loop would run its full sweep budget every frame. The port
   compares against eps² times the matrix's own Frobenius norm.
2. **`theta*theta` overflows f32** at |θ| ≈ 1.8e19, and θ is a ratio whose
   denominator is a converging-to-zero off-diagonal. Past 1e10 the exact
   asymptote `t → 1/(2θ)` is used. f64 has 300 more decades of headroom.
3. **The orientation test is a sign, not a ray cast** — the next note.

### THE NORMAL'S SIGN IS DECIDED ONCE, HERE

An eigenvector's sign is arbitrary, so which way the floor normal points is not
determined by the fit and has to be chosen. **src/pose chooses it twice**, from
the same arbitrary vector: `poseCompute` casts `cornerDir(0,0)` to orient a local
copy for the handedness test, and `gridPeriodPhase` casts the same ray again to
orient its own. Neither writes the choice down, so `triad` in this pipeline would
have carried the raw sign and every consumer would have re-derived it.

Deciding it at the point the sign is created deletes both, and collapses the test
to one comparison: **the camera looks down −z, so "the normal is on the camera's
side" IS `Dnormal.z > 0`.**

That is also strictly more robust than the ray it replaces. `cornerDir(0,0)` is
an image *corner*, which points above the horizon before the view axis does — so
the corner test inverts the normal at grazing tilts where the centre test is
still right. The centre ray is the last one to cross the horizon. (Reachable
around tilt 55° at the fixture FOV; outside the measured operating range, and a
latent bug in `src/pose` rather than a hypothetical.)

It also makes gnomonic projection's valid hemisphere deterministic instead of
arbitrary, which §11 inherits.

### THE DECLARED DEGENERACY GUARD CANNOT FIRE, AND THE REAL ONE IS ELSEWHERE

`planesFromScatter` guards `|b1 ± b2|² < 1e-9` before normalizing. **That test is
unreachable in exact arithmetic**, not merely unreached: Jacobi accumulates its
rotations into a matrix that starts as the identity, so `V` is orthogonal by
construction, `b1` and `b2` are orthonormal, and `|b1 ± b2|² ≡ 2`. It is not
ported.

What is reachable, and what `src/pose` does not catch, is **an all-zero scatter
matrix** — no lines, or every vote degenerate. Every eigenvalue is then 0, the
"smallest eigenvector" is whichever identity column comes first, and the pipeline
receives a perfectly orthonormal triad computed from nothing. `fitDegenerate`
reports that condition instead.

The same argument says the triad is **orthonormal by construction**:
`(b1+b2)·(b1−b2) = |b1|² − |b2|² = 0`, and both are perpendicular to the third
eigenvector. So this fit structurally cannot report its own misfit — the residual
is invisible in the output. Known, consistent with the BoofCV comparison's
"geometric versus algebraic cost" finding, and out of scope for a port.

**Handedness** stays exactly as `src/pose` has it: negate `Dcol` if
`Drow × Dcol · Dnormal > 0`, using the now-oriented normal. This matters much
later, in Stage 13 — see the note there.

### THE MUTATION RUN, and the two sign conventions that were invisible

Ten deliberate bugs. Eight caught, two not, and both survivors are true
properties.

| mutation | caught by |
|---|---|
| transposed unpack of the 21 entries | both fixtures |
| 3×3 cross terms not halved | both fixtures |
| 3×3 zero eigenvalue picked by sign, not magnitude | both fixtures |
| one Jacobi sweep instead of 32 | both fixtures |
| tree reduction skips half the lanes | the rendered frame only |
| no all-zero-scatter guard | the empty-vote fixture |
| **no normal orientation** | **nothing, until a fixture was built** |
| **no handedness flip** | **nothing, until a fixture was built** |
| 6×6 smallest by magnitude, not signed | **NOTHING — and correctly so** |
| no `maxWeight` normalization | **NOTHING — and correctly so** |

**Both sign conventions were invisible, and each had an assertion written
specifically for it that passed while the flip was deleted.** Jacobi's
eigenvector signs are deterministic but arbitrary, and for an ordinary oblique
view they come out already-correct — so deleting either flip changed no number
anywhere, including across a whole rendered frame. This is the fifth and sixth
instance of a fixture sitting on the symmetry a mutation acts on.

The fixtures came from a mechanical search rather than from thought: disable the
correction, scan the input space for a case where the RAW output is wrong.
Measured over 132 poses — **tilt 50 / yaw 35 is left-handed, and the same pose
with roll 245 puts the raw normal at z = −0.643**. Both are now named in the
test. The search also gives the reachability statement for free: **across tilts
0–40 with no roll, 0 of 77 poses reach either branch**, so both flips are cheap
insurance against a silently mirrored floor rather than routine corrections.

**The two survivors.** `ATA` is a Gram matrix, so its eigenvalues are
non-negative and smallest-by-magnitude *is* smallest; the signed form is kept
only because the two can differ through round-off at exactly the entry being
selected, where matching `linalg.ts` is free. And normalizing by `maxWeight`
scales `ATA` by a constant, which leaves its eigenvectors untouched — this
document has always claimed that, and the mutation is the first thing to verify
it. **It also means `maxWeight` is still not observed by anything, including by
its only consumer.** §9's note that "the first test of it arrives with the fit"
is wrong: the fit structurally cannot be that test.

### ACCURACY, AND WHY THE NUMBER IS NOT sqrt(N) BETTER

Scored against ground truth. On synthetic votes generated from a known axis pair
the answer is exact by construction and comes back at **6.1e-9** — f32 round-off,
so the assertion is a defect gate rather than a tolerance.

On a rendered board the interesting number is how the fit handles the detector's
real error. At 96×128, tilt 20, yaw 15: **normal 0.420°, axes 0.457°, from 68
votes whose own median residual is 0.47°.** The fit lands *at* the per-vote
error, not a factor of √68 below it — the detector's endpoint error is correlated
within a grid family rather than independent noise. That is a property of the
input, and the check that it is not a bias here is that it falls with resolution:

| pose | 96×128 | 192×256 | 360×480 |
|---|---|---|---|
| tilt 20 yaw 15 | 0.420° | 0.260° | 0.214° |
| nadir | 0.357° | 0.020° | **0.000°** |
| tilt 35 yaw 40 | 1.125° | 0.472° | 0.291° |

Nadir reaching exactly zero at 360×480 is the clearest statement available that
the arithmetic is exact and everything left is discretization. It also sets the
test's gate honestly: 1.0° has ~2.4× headroom on the fixture pose and would
*fail* at tilt 35 on the same frame size, so raising the frame is the right
response if it goes flaky, not raising the threshold.

---

## 11. Stage 7 — Grid period and phase  *(BUILT 2026-08-13)*

**This is Act II, the hardest stage to port, and the one to do last.**

**What it does.** Recovers the spacing of the floor grid (which gives camera
height) and its offset (which gives where cell boundaries fall).

**The math, in four steps.**

*Step 1 — classify.* Each line is either a "row line" or a "column line". Its own
vote normal decides: compare `|n·Drow|` against `|n·Dcol|`. This is an
orientation-only test, the same one the plane-pair fit relies on.

*Step 2 — rectify.* Project each line's two endpoints **gnomonically** onto the
recovered floor plane. This maps the perspective-distorted image into a space
where the grid is genuinely periodic. A row line's every point shares the same
`xCol` coordinate, so that scalar *is* the line's position in the periodic
sequence. (Averaging both endpoints is a cheap noise bonus — in theory they are
identical.)

> **DO NOT PORT `gridPeriodPhase`'s NORMAL ORIENTATION. §10 ALREADY DID IT.**
>
> `computeGridPeriodPhase` opens by casting `cornerDir(0,0)` and negating a local
> copy of `Dnormal` if the dot is positive, because it is handed the RAW
> eigenvector whose sign the fit does not determine. In `src/pose2` the sign is
> decided once, in `fit.eigen`, so **`triad[2]` arrives already oriented toward
> the camera (`z > 0`)** and re-orienting it here would be a second flip.
>
> Two consequences worth knowing before the first number looks wrong:
>
> - `src/pose` passes the **raw** normal to `gnomonic()` while using the oriented
>   one only for its grazing gate. Here they are the same vector, so **gnomonic's
>   `value` coordinates may come out globally negated relative to `src/pose`'s.**
>   That is self-consistent — period is a spacing, phase and the decode lattice
>   are derived from the same coordinates — but it means a value-for-value
>   comparison against the old implementation is not the right check. Ground
>   truth is.
> - It makes gnomonic's valid hemisphere (`r · Dnormal < 0`) deterministic
>   instead of depending on an arbitrary eigenvector sign, which is a
>   simplification rather than a compatibility cost.

*Step 3 — find the period.* This is not a closed-form step. The approach:

1. **Seed-free integer-count candidates.** A detected line *is* a grid line, so
   the two extreme detected lines sit an integer number of periods apart. The
   only physically possible periods are `P_n = spread/n`. `n` runs from 3 (fewer
   is not credibly periodic; decode needs `ORDER=5` cells anyway) up to a bound
   derived from the board's own cell count and a maximum assumed camera height.
   No seed, no tunable width, no arbitrary sample count.
2. **Score each candidate** by the weighted **circular resultant** — fold every
   value onto the unit circle at `θ = 2π·value/P` and measure how tightly they
   cluster. 1.0 = perfect periodicity, 0 = none.
3. **Harmonic disambiguation.** The resultant is high at the true period `P₀` and
   at *every* sub-multiple `P₀/2`, `P₀/3` — a lattice at spacing `P₀` trivially
   lies on any finer sub-lattice. The resultant alone cannot separate them. But a
   sub-multiple **oversamples**: each real cell becomes a `k×k` block of identical
   samples. The De Bruijn pattern is locally unique, so at the *true* period
   adjacent cell centres differ, while at a sub-multiple they repeat. So for the
   top few peaks, sample the grayscale at cell centres (at phase + half a period,
   to hit solid interiors rather than the gray edges the phase sits on) and pick
   the period whose neighbours are most **distinct**.
4. **Golden-section polish** within the winner's one-count neighbourhood — no
   parabola-shape assumption.

*Step 4 — height.* `height = cellPitch / period`. With `GRID_STEP = 1` this is
just `1/period`.

Existing implementation:
[gridPeriodPhase.ts](src/pose/stages/period/gridPeriodPhase.ts) (535 lines, host)
and [periodSweep.wgsl.ts](src/pose/stages/period/periodSweep.wgsl.ts) (the sweep
only, already on device).

| buffer | size | MiB | contents |
|---|---|---|---|
| `samples` | `MAX_LINES × 8` | 0.125 | `vec2<f32>` (value, weight) per line |
| `family` | `MAX_LINES × 8` | 0.125 | `vec2<u32>` (isRow, 1−isRow) — the scan input |
| `familyScan` | `MAX_LINES × 8` | 0.125 | `vec2<u32>` exclusive prefix — (rowIdx, colIdx) |
| `familySums`, `familyOffs` | `2 × blocks × 8` | — | Scan block temporaries |
| `rowSamples` | `MAX_LINES × 8` | 0.125 | Compacted row family, `vec2<f32>` |
| `colSamples` | `MAX_LINES × 8` | 0.125 | Compacted column family |
| `familyCounts` | 8 B | — | `[rowCount, colCount]` — the scan spine writes it directly, no copy |
| `extent` | 32 B | — | `[rowMin, rowMax, colMin, colMax, spread, …]`. Per-family |
| `scores` | `candCount × 16` | 0.005 | `vec4` per candidate: (score, phiRow, phiCol, period) |
| `topK` | 128 B | — | Slot 0 is the count; slots 1..6 are (period, score, phiRow, phiCol) |
| `distinctness` | 32 B | — | Image-content score per top-K candidate; −1 = unsampled |
| `gppResult` | 16 B | — | `period`, `phiRow`, `phiCol`, `height` |

**Subtotal: 0.563 MiB.** `MAX_CANDIDATES ≈ 256`, bounded by the board cell count.

**This stage uses the shared vec2 scan** (§7), already built and tested. The two
compaction scans it was going to need — one over `isRow`, one over `1−isRow` —
are one scan whose grand total *is* `[rowCount, colCount]`.

**THIS STAGE ALSO OWNS THE DECODE LATTICE BOUNDS — SETTLED 2026-08-14.** §12's
proposal replaces decode's 49x49 ray grid with a min/max over the
gnomonically-projected line endpoints, which `gpp.classify` computes anyway, and
the measurement adopted it. So **`extent` grows two lanes** (it has three free
slots and needs four) and `decode.bounds`/`decode.boundsInit` disappear.
`gpp.extent` as BUILT computes only the view-quad version, so this is the one
piece of §11 that decode has to come back and extend.

### Notes

**Memory here is trivial. The difficulty is entirely control flow.**

**The golden-section search is serial and that is fine.** Each probe depends on
the previous comparison, so it cannot be parallelized across iterations. As a
one-thread kernel: ~40 iterations × 2 probes × ~400 samples ≈ 32,000 serial
floating-point operations on a single GPU lane. That is tens of microseconds.
**"Naive and possibly slow" turns out not to be slow here** — worth knowing
before dreading this stage.

**The per-family extent is a real bug this code already fixed once**, and the
port must not reintroduce it. A row line's `value` is its `xCol` coordinate; a
column line's is its `xRow`. These are two *different axes*. Pooling them and
taking a global min/max measures the extent of the union of two unrelated
coordinate sets, which exceeds either one whenever the camera's nadir is
off-centre asymmetrically between the axes — generic for anything oblique. The
inflated numerator pushed the true period outside the candidate bracket entirely,
so the search never evaluated it. **`extent` must be per-family.**

**`extent` must be initialized to ±∞, not zero.** Atomic min/max over
zero-initialized slots silently clamps the range to include the origin. This is a
second instance of the same pattern as `grow.args` — a buffer whose correct
initial state is not zero, so it needs a *write*, not a *clear*.

**Known limit, documented and deferred.** All of this runs on the *sparse*
detected lines. Under heavy line dropout at extreme grazing (measured ~70 lines
vs ~400, at 40–80 cells up and −18° to −25°), the sparse set can grow a genuine
longer-period spurious structure — which also has high distinctness, because it
*under*-samples, so step 3.3 will not catch it. That regime needs a dense
per-pixel gnomonic histogram plus autocorrelation. Measured to beat this only at
those extremes; deferred.


### BUILT 2026-08-13 — seven passes, and the hard part was not the control flow

§11 called this the hardest stage in the plan and said the difficulty is entirely
control flow. That turned out to be wrong in an instructive way. The golden
section, the harmonic tie-break and the device-side candidate bound all went in
as declared and worked first try. **What actually bit was a resource limit and a
reserved word**, neither of which appears anywhere in this document.

- **WebGPU guarantees only EIGHT storage buffers per compute stage.** `gpp.polish`
  as declared binds ten. That is not tunable — the adapter here supports 10 and
  says so in the error, but requesting a raised limit makes the pipeline refuse to
  build on a baseline device. The fix is the better layout anyway: `sampleValue` +
  `sampleWeight` became one `vec2<f32>` array, and so did each compacted family.
  Value and weight are never read apart. **`decode.layout` binds nine and will hit
  this next** — see §12.
- **`active` is a WGSL reserved keyword.** The symptom was the documented one: no
  exception, every pass a silent no-op, and a probe printing plausible zeros for
  every pose. `withDevice`'s error scope is the only reason it took a minute
  rather than an afternoon.
- **WGSL rejects `inf`**, including through a const-evaluated `bitcast` of its bit
  pattern. `gpp.extent`'s sentinels are `f32::MAX`.

**`gpp.extentInit` is DELETED, and the argument generalizes.** The declaration had
an init pass writing ±∞ into an atomic min/max target, the same shape as decode's
`uvBounds`. It does not need to be one. `value` is SIGNED, so the
atomicMax-over-float-bits trick `maxWeight` uses is unavailable here — it would
need an order-preserving u32 encoding, its inverse, the init pass, and a
declaration entry marked `written` rather than `zero`. One workgroup striding and
tree-reducing gets the same answer with none of it, and lane 0 writing all eight
slots makes the buffer total by construction. Same move that deleted
`ataPartials` in §10. **An initialization pass is usually a symptom of choosing an
atomic where a reduction would do.**

**`gpp.classify` dispatches DIRECTLY over `maxLines`**, closing the third
iteration-domain defect this rewrite has found by re-derivation. The trap inside
it is worth naming separately: the dead tail must be written `vec2(0,0)` and
**not** `vec2(0,1)`. `family` is (isRow, 1−isRow), so the obvious "not a row line,
therefore a column line" encoding counts all ~16,000 dead slots into the column
family. Non-membership and column-membership are different facts.

**Two quaternion round-trips disappear.** `computeGridPeriodPhase` takes the
camera quaternion, casts world-space rays with it, and `makeCellCentreDistinctness`
carries its inverse to get back to pixels. Neither is needed: votes are cast in
camera space, the triad is derived from those votes, so gnomonic projection of
camera-space rays onto a camera-space triad never leaves camera space, and
`gpp.distinct`'s reprojection is a bare pinhole. That is a consequence of the
frame being consistent, not an optimization.

**One guard `src/pose` does not have.** A cell BEHIND the camera projects through a
negated depth to a mirrored image point that can land on-screen, and the grazing
test cannot catch it — `p` is on the floor plane, so `dot(p, Dnormal)` is exactly
−distance whatever the lateral offset, and the test is blind to it. The cutoff
does bound |p| (10× distance at `minGrazingCos` 0.1) but a tilted camera reaches
`p.z ≥ 0` well inside that. Latent in `src/pose` rather than hypothetical.

### ACCURACY, AND WHERE THE ERROR ACTUALLY COMES FROM

Period recovered against ground truth, 2026-08-13, supersample 4:

| pose | 96×128 | 192×256 |
|---|---|---|
| nadir | 0.29% | 0.14% |
| tilt 20 yaw 15 | 0.49% | 0.70% |
| tilt 30 yaw 25 | **4.42%** | 1.55% |
| tilt 35 yaw 40 | 0.21% | 0.95% |
| tilt 20 yaw 15, h=16 | 1.37% | 0.99% |
| tilt 10 yaw 60, h=6 | 0.69% | 0.13% |

**Every one of these equals the error of the PEAK in the sample values, measured
independently, to the digit.** So the search is finding the true maximum and what
remains is entirely §10's axis error rescaling the gnomonic coordinates upstream.
tilt 30 yaw 25 is the outlier in both tables for one reason: 0.88° of axis error
at 96×128, falling to 0.25° at 192×256, and the period error falls with it.

**This changes how the stage must be SCORED, and cost a test rewrite.** The
obvious oracle — the circular resultant evaluated AT the true period — is a JOINT
statement about this stage and the triad it was handed, and it is hypersensitive
to the joint part. At tilt 30 yaw 25 it reads 0.411 while the peak is a perfectly
respectable 0.846: the values span ~26 cells, so a 4.4% period error accumulates
more than a full period of drift across the data and the phases spread right round
the circle. The test now scores at the peak within a ±20% window (this stage's own
claim) and checks the peak's location separately (the joint claim). Recorded
because the first version of that test looked obviously right and was measuring
two things at once.

### THE MUTATION RUN — 14 bugs, 9 caught, and 5 measured rather than guessed

| mutation | caught by |
|---|---|
| classify dispatches one workgroup (the §9 stale-tail shape) | 2 tests |
| the dead tail says "column line" instead of "neither" | 2 tests |
| a family is measured along its OWN axis | 3 tests |
| the family test is inverted | 3 tests |
| extent pooled across families | the per-family test |
| polish takes the LEAST distinct candidate | 2 tests |
| no golden-section refinement | **the structural assertion, added for it** |
| refinement ignores its own comparisons | 2 tests |
| distinctness samples cell BOUNDARIES, not centres | **the margin gate, after it was tightened** |
| classify averages one endpoint instead of two | **NOTHING — and near enough correctly** |
| sweep drops the shift by the family minimum | **NOTHING — and the claim was overstated** |
| sweep does not score the column family | **NOTHING — the families are redundant** |
| peaks drops the significance cut | **NOTHING — a documented regime is missing** |
| no behind-the-camera guard in distinct | **NOTHING — unreachable here** |

**Two of the nine were only caught after being measured**, and both are the same
lesson in different clothes:

- **The distinctness margin was a near miss, not a gate.** Sampling on cell
  boundaries does *not* destroy distinctness — a boundary pixel is the average of
  two cells, and adjacent boundaries still differ when the cells do. It COMPRESSES
  the margin, from 2.03 (correct) to 1.53 (mutated), and a threshold picked by eye
  at 1.5 slipped underneath by 0.03. It is now 1.8, set from both measurements.
- **Golden section is worth about half the error and no accuracy gate can see
  it.** Unrefined against refined: 0.61→0.29%, 0.76→0.49%, −0.32→0.21%,
  1.59→1.37%. Every one of those is inside any threshold wide enough to pass the
  frame at all, because the upstream axis error dominates. So the assertion is
  structural instead: the answer moved OFF the candidate grid and stayed inside
  the winner's own one-count neighbourhood.

**The five survivors, each with the measurement that classifies it:**

1. **Averaging the two endpoints is worth ~0.02 percentage points.** Identical to
   four significant figures at three of four poses, 0.29→0.31% at the fourth. In
   exact arithmetic the two endpoints of a row line have the same `xCol` — that is
   what "projects to a constant-xCol line" means — so this is a noise bonus and
   very nearly a true property. Not worth a manufactured fixture.
2. **The shift by the family minimum protects candidates that are never chosen.**
   Deleting it changes the recovered period in not one digit at four poses across
   two frame sizes. The argument is sound at the hard cap n = 288, where an
   unshifted fold argument reaches ~300 and f32 has three digits left — but the
   winning candidate has n between 14 and 30 on every measured frame. **The
   shader comment claiming this was load-bearing is corrected in place.**
3. **The two families are redundant for the PERIOD.** Square cells force the same
   physical period on both axes, so either family alone determines it; pooling
   sharpens the peak rather than enabling it. Would matter on a frame where one
   family is nearly empty, which no fixture here produces.
4. **The significance cut has nothing to reject on a good frame.** This confirms
   rather than undermines the design: the cut is a real-vs-noise filter and
   *distinctness makes the actual decision*, exactly as §11 claims. What would
   need it is a spurious LONG period with high distinctness — which is precisely
   the known limit §11 already documents and defers (heavy line dropout at
   extreme grazing, ~70 lines against ~400). Missing fixture, in a deferred
   regime.
5. **The behind-the-camera guard is unreachable at every pose tried.** Same class
   as §8's log-sum-exp rescale and §9's degenerate arc guard: kept because the
   failure would be a silently mirrored sample rather than a loud one, and
   nothing here proves it right.

---

## 12. Stage 8 — Decode layout  *(BUILT 2026-08-14)*

**What it does.** Works out the sampling lattice: which floor cells are visible,
where their boundaries fall, and how bright a pixel has to be to count as white.

This is a small stage that produces ~150 bytes, and it is **the most structurally
disruptive change in the whole plan.**

**The math.** Cast rays through a 49×49 grid of image points, intersect each with
the recovered floor plane, and take the min/max of the resulting `(u,v)`
coordinates. That is the bounding rectangle of the resolvable visible floor.
Divide by `GRID_STEP` and offset by the recovered phase to get integer lattice
bounds `kMinU..kMaxU`, `kMinV..kMaxV`, and hence `rows` and `cols`.

**Why 49×49 sampling and not the 4 image corners.** An all-or-nothing corner test
returns nothing the instant one corner points past the horizon — which silently
killed decode at any oblique view, discarding an 80–94% usable frame along with
two grazing corners. Sampling and bounding only the points that clear the
grazing cutoff is both correct and strictly more permissive. For a non-grazing
view every sample clears and it reduces to the same min/max the corners gave.

Currently on the host: `projectedUVBounds` and `decodeGridLayout` in
[decodeGrid.ts](src/pose/stages/decode/decodeGrid.ts).

| buffer | size | MiB | contents |
|---|---|---|---|
| `uvBounds` | 16 B | — | `minU, maxU, minV, maxV`. **Init to ±∞.** Atomic min/max |
| `binThresholdPartials` | `ceil(n/256) × 4` | 0.005 | Per-block sums for the image mean |
| `binThreshold` | 4 B | — | Global mean gray level |
| `layout` | 128 B | — | The full lattice description (see below) |
| `buildArgs` | 12 B | — | Indirect args for the build pass |
| `tallyArgs` | `4 × 12 B` | — | Indirect args, one per orientation |
| `correctnessArgs` | 12 B | — | Indirect args for the correctness pass |

**Subtotal: 0.005 MiB.**

**The `layout` struct** holds: `Drow`, `Dcol`, `normal`, `invQuat`, `distance`,
`tan(vFov/2)`, `aspect`, `minGrazingCos`, `uPhase`, `vPhase`, `GRID_STEP`,
`binThreshold`, `rows`, `cols`, `imageW`, `imageH`, `kMinU`, `kMinV`. It is the
same 128-byte block the current host `buildUniforms` writes — **the only change
is that the device writes it instead.**

### PROPOSED (user, 2026-08-12): BOUND THE LATTICE BY THE VOTE LINES, NOT THE VIEW QUAD

**MEASURED, ADOPTED AND BUILT 2026-08-14.** The reasoning is kept in full because
it is worth more than the conclusion. What was built, and where each piece went:

- **`gpp.classify` emits the hull's per-line contribution.** It already computed
  both endpoints' full gnomonic coordinates and threw three quarters of them
  away; `samples` grew from `vec2` to `vec4` to carry `(crossMin, crossMax)` --
  the span of the coordinate the line RUNS ALONG, which is the half a per-family
  extent structurally cannot see. A row line's own value pins its xCol and says
  nothing about how far along xRow it reaches.
- **`gpp.extent` reduces four more lanes** in the same traversal, with no extra
  binding and no extra buffer, and `extent` grew from 8 lanes to 12. Each AXIS is
  bounded by one family's value and the OTHER family's cross: xRow by (column
  values, row crosses), xCol by (row values, column crosses). One line of either
  family already bounds both axes.
- **`decode.bounds` and `decode.boundsInit` were never written**, and `uvBounds`
  is gone from the declaration entirely.
- **Measured on the built version**, not only on the harness: at the §19 fixture
  pose the cross lanes widen the hull by 1.78 cells in xRow and 2.63 in xCol over
  the value box, and the hull covers 93% / 91% of the truly visible extent. Both
  are in `tests/pose2Stages.test.ts`, and the first is a GATE rather than a
  containment check -- a version that forgot the cross lanes produces the value
  box, which contains nothing suspicious on inspection.

Replace the 49x49 ray grid with a min/max over the **detected line endpoints,
projected onto the floor**. The lattice only needs to cover where decodable
pattern actually is, and that is inside the detected grid lines.

**The quantity already exists, per endpoint.** `GridLineSample` in
[gridPeriodPhase.ts](src/pose/stages/period/gridPeriodPhase.ts) carries `p1`/`p2`
-- each line's own two endpoints, gnomonically projected, *"in the SAME
coordinate space `value`/period/phase are expressed in."* So the hull is an
atomic min/max over data `gpp.classify` already holds in registers, not new
maths. In `src/pose2` it becomes two more lanes on gpp's existing `extent`
reduction.

**The two coordinate systems differ by one scalar.** `gnomonic()` gives
`xRow = -r.Drow / r.Dnormal`; `projectedUVBounds` gives `u = r.Drow * t` with
`t = -distance/denom`. So **`u = distance * xRow`**, and `distance` is gpp's own
output. Converting the hull to lattice units is one multiply in `decode.layout`.

**This DELETES a stage rather than shrinking a buffer.** The 49x49 grid exists
solely because an all-or-nothing corner test returns nothing the moment one
corner points past the horizon (see above). A detected line is below the horizon
*by construction* -- you cannot detect a grid line in sky -- so the problem
dissolves instead of being worked around. `decode.bounds`, `decode.boundsInit`,
2,401 ray casts and the `minGrazingCos` gate all go, and `uvBounds` stops being
decode's business.

**No hidden bulge.** Gnomonic projection maps straight lines to straight lines --
that is *why* a row line has constant `xCol` -- so a segment projects to a
segment and its extremes are its endpoints. Bounding endpoints genuinely bounds
the segments.

#### What this does to MAX_CELLS

**`MAX_CELLS` stops being a chosen constant and becomes `torusR * torusC`.** That
is rule 3 ("where a count is provable and tight, use it") applied to the last
buffer still sized by guess, and it closes open decision 2 *by construction*
rather than by measurement.

> **MEASURED — the constant is right, "by construction" is not.** The raw hull
> reaches a 169-cell edge at h=24 tilt 45-55, so `decode.layout` must clamp each
> edge to 144. Clamping is free (no pose changes anywhere) and the tiling
> argument below is why — but it is a clamp that FIRES in the grazing band, on
> frames whose period is correct to ~1.4%, so it must not be a failure. Details
> in the measurement at the end of this section.

Clipping to one board period is **information-free**: the floor tiles and
decode's anchor arithmetic is mod R/C, so windows from a second tile vote for the
*same* anchor. Capping loses redundant votes, not evidence.

The user's framing was "longer than 144 on either edge means something went
wrong", and that needs one correction: **exceeding 144 is legitimate, not
impossible** -- sim.ts is explicit that a lattice spanning more than one period is
an ordinary case. What makes it a *smell* is the line-hull version specifically:
resolving 144+ grid lines means ~3 px per cell at 480 px, at the edge of
detectability. Treat `gridOverflow` here as a strong diagnostic, not a proof.

And the reason this only works with the hull, which is the user's own point and
the load-bearing one: **with the view quadrilateral you cannot apply a 144 cap at
all, because you do not know WHICH 144x144 window to keep.** At a low angle the
quad sprawls without limit and any cap is an arbitrary truncation. The hull has a
principled origin -- the evidence itself.

**The memory win is small and should not be the argument: ~0.92 MiB** (1.0 ->
0.079). Against 18.4 MiB that is 5%, and §17 already says memory is not the
reason to do any of this. The wins are the deleted stage and an overflow
condition that means something.

#### THE ONE THING TO MEASURE FIRST, AND IT IS MEASURABLE TODAY

The premise -- valid sample cells always lie within the vote lines -- is right
about *decodability* and not automatically right about *sampling*. Sampling a
cell centre needs one pixel on the right side of a threshold; detecting the grid
line bounding it needs a whole NFA-significant region. **There is a band where
sampling still works and detection has already failed**, and §11's own known
limit says dropout gets severe at grazing (~70 lines vs ~400).

Expected loss is near zero, because cells in that band are sub-pixel and sample
to noise -- so clipping them should *improve* decode rather than cost it. That is
a prediction, not a measurement.

**It also breaks the circularity in open decision 2.** `src/pose` computes both
quantities already -- `projectedUVBounds` for the quad, gpp's `p1`/`p2` for the
hull -- so the EXISTING sweep can report, per pose:

1. quad-derived lattice dims (settles MAX_CELLS for the current design)
2. hull-derived lattice dims (settles it for this one)
3. **how many CORRECT decode votes come from cells outside the hull**

(3) is the decisive number, needs nothing built in `src/pose2`, and can be had
before §12 starts. **This is the next concrete action on decode.**

#### MEASURED 2026-08-14 — THE PROPOSAL IS ADOPTED, WITH ONE CORRECTION

`scripts/hull-measure.ts` (`--quick`, `--grazing`, `--inset N`) runs `src/pose`
over the §19 baseline's own 180 poses at 480x640, builds BOTH lattices from the
same recovered axes, decodes both, and scores both against ground truth.

**The 180-pose operating range (tilts 0-40), hull against quad:**

```
same recovered position          180/180        worst position change  0.0000 cells
correct votes lost to clipping    31/173,646    (0.02%)   worst single pose 0.6%
consistency                       never worse, better at 6 poses
MAX_CELLS needed          quad 25,384      hull 19,019     (torusR*torusC = 20,736)
worst lattice edge        quad    167      hull     144
hull/quad cells           median 0.947, and LARGER than the quad at 2 of 180 poses
```

**At grazing (tilts 45-55, heights 10/16/24 — outside the baseline range, where
§11's dropout limit says the premise is most at risk), it gets BETTER, not
worse:** 24/24 same position, 0.18% of correct votes lost — while **35% of all
complete windows are clipped.** Consistency rises on 14 of 24 poses, by up to
**+0.17** (0.60 -> 0.78 at h=24 tilt 55). Clipping a third of the windows costs
two parts in a thousand of the correct votes, which is as direct a statement as
this measurement can make that **the clipped band is noise.** §12's prediction
was right, including its sign.

**`MAX_CELLS = torusR * torusC` = 20,736 is CONFIRMED, and the clip that makes it
a bound is measured rather than argued.** Clipping every edge to 144 changes the
recovered position at **0 of 24 grazing poses and 0 of 180 baseline poses**;
correct votes lost goes 0.18% -> 0.46% (worst pose 3.3%) and consistency still
improves. With the clip the worst case across the grazing set is exactly 20,736
cells. So the tiling argument holds in fact and not only in principle: a second
board period is redundancy, not evidence.

**What the measurement corrects is the WORD "by construction".** The raw hull
reaches **169 cells on an edge** at h=24 tilt 45-55, so the layout must clamp —
and the clamp must **keep decoding**, because those poses are not failures:

| pose | raw hull | TRUE visible extent | period error |
|---|---|---|---|
| h=24 tilt 45 yaw 35 | 130x**169** | **223x200** | 1.47% |
| h=24 tilt 50 yaw 0 | 144x**154** | **222x199** | 1.37% |
| h=24 tilt 55 yaw 0 | 132x121 | 189x221 | 1.54% |

The true extent is computed from ground truth, not from a recovered pose. **The
camera really does see 200+ cells there and the period is right to ~1.4%** — so
"more than 144 means the period is wrong" is false, and `gridOverflow` must stay
a diagnostic rather than becoming a failure bit. Those poses recover position to
0.06-0.09 cells.

**A SECOND, INDEPENDENT INFLATION, and it is the one to know before building
`gpp.extent`.** At h=10 tilt 45-55 yaw 0 the hull exceeds the TRUE visible extent
(122x102 against 98x82). `projectedUVBounds` gates every ray on `minGrazingCos`;
**a gnomonically projected line endpoint has no such gate**, so a line detected
near the horizon projects arbitrarily far. The clip covers it, and gating the
endpoints on grazing is the principled alternative — unmeasured.

**Which means the PER-CELL grazing test in `decode.build` is load-bearing, and
the hull makes it more so.** "The `minGrazingCos` gate goes" above is about the
BOUNDS computation only. Measured over the 24 grazing poses: **9,811 hull cells
are rejected by the grazing test while the screen-bounds test would have ACCEPTED
them** (worst pose 2,642). Those are §11's mirrored cells — a cell past the
horizon projects through a negated depth onto a valid-looking pixel. Delete the
per-cell test and every one of them samples the wrong pixel silently.

**One choice, measured only one way.** The kept 144-window is CENTRED on the
hull. Any contiguous 144 works for the anchor, since the arithmetic is mod R/C;
keep-from-min is untested.

**Why the hull can exceed the quad at all**, which is the thing to know before
building `gpp.extent`'s two extra lanes: `projectedUVBounds` gates every ray on
`minGrazingCos`, and **a gnomonically projected line endpoint has no such gate.**
A line detected near the horizon projects arbitrarily far out. The per-cell
grazing test inside the build pass still invalidates those cells, so the cost is
wasted lattice rather than wrong bits — which is exactly what the clip bounds.

**Two things about the harness, because the numbers are only worth what it is
worth.** It duplicates two functions `src/pose` only exposes bound to the quad
lattice (building a grid from a supplied layout, and `finishPositionDecode`), so
it `selfCheck`s both against the real ones on the unmodified layout — cell for
cell, and `camPos` to 1e-9 — and throws on any disagreement. And **"0 correct
votes lost" is a claim about a counter that has to be shown able to move**: it
reports complete-windows-clipped alongside, and `--inset 3` shrinks the hull
deliberately, which takes the loss to 34/1,112 (10.8% worst pose). The zero is a
measurement, not a blind instrument.

### Why this stage is disruptive

**`rows` and `cols` become device-side quantities**, and this is the prediction
that held up best. It shows up three stages later, in code that has nothing to do
with the layout: `decode.tally`, `decode.argmax` and `decode.correctness` all had
to bind `layout` and read rows/cols off it, because a host-written uniform
carrying them would be exactly the readback this pipeline exists to delete. Their
uniforms carry only board size, ORDER and the hash table size.

Today they are host-known before anything is allocated, so `packed` and `geom`
are sized to the exact cell count. Once the layout is on device:

1. **The decode lattice cannot be sized.** It needs `MAX_CELLS` + an overflow
   bit. And the extent is *not* bounded by the board — it is the projected view
   quadrilateral, which grows without limit approaching grazing incidence and is
   only softly gated by `minGrazingCos`.
2. **Six dispatches become indirect.** `decode.build` dispatches over
   `(rows, cols)`; the tally dispatches four times over `rotatedDims(gr, gc, o)`;
   correctness is already indirect. All six extents are now device-side.

   **As built the layout pass writes FIVE of them, not six.** The sixth is
   `correctnessArgs`, and it belongs to `decode.argmax`: its extent depends on
   the WINNING ORIENTATION, so knowing the winner is not enough -- you have to
   know it before encoding. Giving it to its real owner is also what fits
   `decode.layout` inside the eight-storage-buffer baseline, which it exceeded as
   declared. Failure still propagates: a failed layout zeroes `buildArgs` and
   `tallyArgs`, the histogram stays cleared, argmax finds no winner and zeroes
   `correctnessArgs` itself.

   The four tally triples live in ONE buffer at 12-byte offsets, which is why
   `pass()` grew an `offset` on its indirect dispatch. The four dispatches differ
   only by orientation, which is a compile-time constant in each, so they are
   four ENTRY POINTS over one module rather than four uniform blocks -- the same
   shape as grow's four.

**`binThreshold` is a new full-image reduction.** Today it is a host loop over
`gray`. The current code already notes that materializing a full binarized image
is wasteful — the lattice only reads ~1,000 of ~200,000 pixels — so all it needs
is the mean, compared at sample time. On device that is one block-sum pass plus
one finalize.

---

## 13. Stages 9–12 — Decode build, tally, argmax, correctness  *(BUILT 2026-08-14)*

**This is Act III.** Four passes in sequence, currently already one encoder and
one submit ([decodeGridBuild.gpu.ts](src/pose/stages/decode/decodeGridBuild.gpu.ts)).

### 9 — Build

One thread per lattice cell. Take the cell's `(u,v)` on the floor, compute its
position relative to the camera, rotate into camera space by `invQuat`, project
to NDC, convert to a pixel, sample `gray`, threshold against `binThreshold`.
Write two bits: *valid* and *the bit value*.

Existing:
[decodeGridBuild.wgsl.ts](src/pose/stages/decode/decodeGridBuild.wgsl.ts).

| buffer | size | MiB | contents |
|---|---|---|---|
| `packed` | `MAX_CELLS × 4` | 1.000 | bit0 = valid, bit1 = sampled bit |
| ~~`geom`~~ | ~~`MAX_CELLS × 16`~~ | ~~4.000~~ | **DROP** — `(u,v,px,py)`, display only |

**Dropping `geom` saves 4.0 MiB.** Nothing on the pose path reads it; it exists
solely for the projected-cam overlay and the phone AR readout. A pose-only
pipeline should not allocate it.

**The per-cell grazing test is NOT optional, and the hull raises the stakes.**
`rayDir · normal < -minGrazingCos`, tested per cell, is what rejects a cell
beyond the horizon — which projects through a negated depth to a MIRRORED point
that lands on screen and passes every bounds test. Measured under the hull
lattice at grazing: 9,811 such cells across 24 poses, 2,642 in the worst one
(§12). The screen-bounds test cannot stand in for it.

**Two subtleties worth preserving.** The pixel coordinate is bounds-tested
*unrounded* and then *clamped* before indexing — `px = w − 0.3` passes the test
and rounds to `w`. Measured 46 such cells in a 270×276 grid. And the divisor is
guarded directly rather than testing the quotient for finiteness afterwards:
WGSL has no `isFinite`, and a compiler may assume operands are finite, so a
self-comparison NaN test is not reliable.

### 10 — Tally

One thread per `ORDER × ORDER` window, in each of 4 rotations. Pack the 25 bits
into a key, look it up in the De Bruijn hash table, and if found, `atomicAdd` a
vote for the implied board anchor.

**Why four rotations.** The camera's yaw relative to the board is unknown, and
the pattern is not rotationally symmetric. Rather than solve for the rotation,
try all four cardinal orientations and let the vote count decide.

**Why a hash table.** `debruijnLookup` is a sparse JS `Map` from a 25-bit key to
a torus position — ~20,736 real entries out of a 2²⁵ key space. It is flattened
into an open-addressing table with load factor 0.5, cached per device. The
`hashU32` finisher must stay byte-identical between JS (`Math.imul`) and WGSL
(u32 multiply wraps mod 2³² by spec), or lookups silently miss.

Existing: [decodeTally.wgsl.ts](src/pose/stages/decode/decodeTally.wgsl.ts).

| buffer | size | MiB | contents |
|---|---|---|---|
| `hist` | `4·R·C × 4` | 0.316 | Vote count per `(orientation, anchorRow, anchorCol)`. **atomicAdd** |
| `totalWindows` | 4 B | — | How many complete windows were examined. **atomicAdd** |
| `torus` | `R·C × 4` | 0.079 | The pattern itself, 0/1. **Cached, persistent** |
| `hashKeys` | `65536 × 4` | 0.250 | **Cached, persistent** |
| `hashValues` | `65536 × 4` | 0.250 | **Cached, persistent** |

**`hist` scales with board size squared** — 0.316 MiB at 144, 1.05 MiB at 256.
It is a host constant, so no cap is needed, but it is the one buffer a UI slider
resizes.

### 11 — Argmax

One workgroup of 256 threads reduces the whole histogram to a single winner.
Writes the result block *and* the correctness pass's dispatch args.

**The tie-break must match exactly:** highest votes, then **lowest index**, and a
winner requires at least one vote. Threads stride upward taking `>` only; the
tree reduction prefers the smaller index on a tie.

**One workgroup deliberately.** `4·R·C` is 82,944 at board 144 — 324 strided
iterations for 256 threads, far cheaper than the 331 KB readback it replaced. A
two-level reduction would use the machine better but needs a second pass, a
second buffer, and its own cross-block tie-break argument.

| buffer | size | MiB | contents |
|---|---|---|---|
| `result` | 32 B | — | 8 u32 (see below). **Clear — slots 6/7 are atomicAdd targets** |

**The result block:** `found, orientation, anchorRow, anchorCol, votes,
totalWindows, correct, wrong`. `found = 0` is an ordinary outcome (an undecodable
frame), not an error.

### 12 — Correctness

Reduces the whole rotated grid to two integers: how many sampled bits match what
the pattern says should be there, and how many do not. Their ratio is
`consistency`, the one decode number always on screen.

**It dispatches indirectly** because its extent depends on the winning
orientation — rows and cols swap at orientations 1 and 3. Even with the winner in
a buffer, the host would have had to know it to size this dispatch. That is the
*second* host dependency in decode, and it is the one easy to miss.

---

### THE MUTATION RUN FOR §12-14 — 26 bugs, and the fixtures were the finding

Run 2026-08-14 against `decode.layout`, `decode.build`, the tally, the argmax,
the correctness pass and `finish`. **Final: 25 caught, 1 survivor.** But the run
started at 16 caught, 8 surviving and **2 that never ran at all**, and the eight
are worth more than the number.

| mutation | caught by |
|---|---|
| phase omits the half-cell offset | 4 tests |
| phiRow / phiCol swapped | **NOTHING until the fixture moved off a cell centre** |
| the bounds read the wrong hull lanes | 2 tests |
| no 144 clamp | the clamp test |
| the kept window starts at min, not centred | the clamp test |
| buildArgs rows/cols swapped | **NOTHING until the lattice was elongated** |
| tallyArgs does not swap at o=1,3 | **NOTHING until the lattice was elongated** |
| binThreshold divides by the block count | 5 tests |
| `valid` set before the guard | the blank-frame test |
| build: threshold sense flipped | 4 tests |
| build: no per-cell grazing test | **NOTHING until a synthetic layout was built** |
| build: no behind-the-camera guard | **NOTHING until a synthetic layout was built** |
| build: lattice indexed transposed | 4 tests |
| tally: an unresolvable cell reads as 0 | the anchor test |
| tally: origIndex wrong at o=2 | the pipeline test |
| argmax: tie-break prefers the larger index | **NOTHING — and correctly so** |
| correctnessArgs does not swap at o=1,3 | **NOTHING until BOTH an exact count AND an elongated lattice** |
| **the §14 sign flip deleted** | the pipeline test |
| the §14 axis swap deleted | the pipeline test |
| the reference index not rotated | the pipeline test |
| the reference u/v use rotated indices | the pipeline test |
| camPos adds instead of subtracts | the pipeline test |
| cell centres treated as corners | the pipeline test |
| consistency divides by `correct` | **NOTHING until the assertion became an equality** |

**TWO PATTERNS NEVER MATCHED, which is the one wrong answer a mutation run must
never produce.** Both would have been reported as "no test caught it" by a script
without the match-exactly-once assertion. The causes are worth knowing because
neither is a typo: one pattern was a line that appears IDENTICALLY in two shaders
(`gpp.distinct` and `decode.layout` both fold a phase the same way), and the
other anchored on two functions being adjacent when a comment block sits between
them. Both were caught by the assertion, fixed, and then both mutations were
caught by tests.

### THE CEILING COINCIDENCE, AND A DECISIVENESS CHECK THAT MEASURED THE WRONG THING

Three of the eight survivors are the same bug in three passes: **an indirect-args
pair is a CEILING over the 8x8 workgroup, so what has to differ between rows and
cols is `ceil(n/8)`, not `n`.** The fixture lattice was 19x22 — genuinely
non-square, and `ceil(19/8) == ceil(22/8) == 3`, so a swapped pair is bit-identical
to a correct one.

**The args test had a decisiveness assertion written for exactly this, and it was
the wrong predicate.** It asserted `rows != cols`, which is true at 19x22 and
says nothing about the quantity under test. This is a new species: not a fixture
sitting on a symmetry with no check, but a check that measures a DIFFERENT
quantity from the one the mutation acts on. The seven earlier instances in this
rewrite were all the first kind.

**`correctnessArgs` needed two fixes, and the second is the interesting one.** An
exact assertion — `correct + wrong` equals the number of valid cells, which is an
equality because every valid cell appears exactly once in the rotated grid —
still did not catch it, because the ceiling ROUNDS UP far enough to cover: 19x22
becomes 24x24 threads either way round. It takes an ELONGATED lattice (tilt 40's
23x37, rounding to 24 and 40) for a swap to truncate anything. So the fixture
needs both an exact instrument and a shape that instrument can see.

### EVERY FIXTURE IN THE SUITE SAT ON THE SAME SYMMETRY

`phiRow` is the row family's phase and lives in xCol, so it becomes `v`; `phiCol`
becomes `u`. Swapping them was invisible, and the reason is that **every pose in
this suite — and every offset in the sweep — sits at `overRow`/`overCol` = *.5,
which is a cell centre in BOTH axes, where the two phases are equal.** Measured
on the reference cell's distance from a true cell centre:

```
over(40.5, 40.5)    0.127 -> 0.042    the swap IMPROVES it: pure symmetry
over(40.25, 40.75)  0.147 -> 0.172    still not separated
over(40.5, 40.2)    0.156 -> 0.229
over(40.1, 40.6)    0.123 -> 0.335    <- the fixture now
over(40, 40.5)      0.126 -> 0.491    a boundary-sitting pose
```

The layout fixture moved to (40.1, 40.6). **The sweep's three offsets are still
all .5**, which is a live gap rather than a fixed one — it cannot see this class
at any pose.

### THE TWO GUARDS NEEDED A SYNTHESIZED LATTICE, NOT A RENDERED ONE

`decode.build`'s grazing test and behind-the-camera guard both survived, and §13
already recorded WHY: counted directly at h=24 tilt 45 and tilt 55, zero cells
of either kind, because the line hull has already clipped that band at 96x128.

The fixture that reaches them does not render anything. `layout` is a plain
buffer and `decode.build` is its only reader, so the test writes a 128-byte block
by hand — a HORIZONTAL camera, normal +y, `Dcol` = −z — which puts floor cells on
both sides of it. At distance 10 that gives all three regimes at once: `v <= 0`
behind the camera (88 cells, every one of which projects ON SCREEN through a
negated depth), `13 < v < 99` valid (688 cells), and `v > 100` past the grazing
cutoff (136 cells). The reachability of each band is asserted BEFORE the
behaviour, and doing that caught the first version, whose 64-row lattice reached
v = 39 and had an empty grazing band.

### THE ONE SURVIVOR, MEASURED RATHER THAN ARGUED

**The argmax tie-break is unreachable on a decodable frame.** Counting entries
holding the maximum vote across four poses: 1 every time, with the runner-up at
0–6 against maxima of 115–501. A tie needs two anchors with EQUAL maximum votes,
which is an ambiguous decode — and `src/pose`'s CPU reference breaks that case by
Map insertion order anyway, so the two implementations are already known to
diverge exactly there. Kept for the same reason §8's log-sum-exp rescale and §9's
degenerate arc guard are, and nothing here proves it right.

---

## 14. Stage 13 — Finish  *(BUILT 2026-08-14)*

**What it does.** Converts the winning anchor into a world position and a camera
quaternion. One thread.

**The math.** Register the reference lattice cell against the torus to get its
absolute board row and column. Solve the camera's true world orientation in
closed form from `Drow`, `Dcol` and the winning orientation. Rotate the
camera-relative floor hit into world space and subtract it from the reference
cell's known world position.

Currently: `finishPositionDecode` and `solveRecoveredCamQuat` in
[decodeGrid.ts](src/pose/stages/decode/decodeGrid.ts).

| buffer | size | contents |
|---|---|---|
| `pose` | 128 B | **The one readback.** Status word, position, quaternion, diagnostics |

**Suggested layout (32 u32):** `status` bitfield, `ok`, `position` (3 f32),
`quaternion` (4 f32), `consistency` (f32), then diagnostics — `orientation`,
`boardRow`, `boardCol`, `votes`, `totalWindows`, `correct`, `wrong`,
`regionCount`, `memberCount`, `lineCount`, `gridRows`, `gridCols`, `growRounds`,
`period` (f32), `height` (f32).

### The trap in this stage

`solveRecoveredCamQuat` rotates `(Dcol, Drow)` through the winning orientation
with **sign flips** at each step (`nextCol = rowMath.negate()`). An earlier
version got the axis swap right and missed the sign flips entirely — silently
correct at orientation 0 (most captures) while **still reporting perfect bit
consistency at orientations 1, 2, 3**, because period, phase and anchor recovery
are all independently correct and only this final rotation was wrong. It was
caught live at yaw +2° where a ±2° swing flipped which orientation won.

### BUILT 2026-08-14, and the trap decided the fixture

Both lessons below survived the port intact, and the second one is what shaped
the test rather than a caveat attached to it.

**The whole-pipeline test runs FOUR poses chosen so the winning orientation is
not always 0**, and it asserts that at least three distinct orientations were
exercised BEFORE it asserts anything about the poses. Without that, the fixture
could pass with both sign flips deleted — which is the exact shape six earlier
mutation survivors in this rewrite had, a fixture sitting on the symmetry the bug
acts on. Measured: yaw 15 / 105 / 195 / 285 win orientations 3 / 2 / 1 / 0.

**`finish` was declared with NINE storage buffers** and does not build on a
baseline device. `triad` and `gppResult` both came off, and neither is a
substitute: `layout` already carries the triad because decode.build needs it
there, and `layout.distance` IS the height, so the period is
`cellPitch / distance`. Reading a value from the one place that already had to
have it is the third distinct fix for this limit in the pipeline — after pairing
fields into a `vec2` (§11) and giving a buffer to its real owner (§12).

**It also does the REPORTING half of the status word**, deriving five bits from
buffers it already binds rather than making every kernel test a status it does
not otherwise care about. That includes `regionOverflow`, from
`counts.x > maxRegions` — which closes the objection pipeline.ts carried, that
collect does not bind `status`. It does not need to.

**Two lessons for the port.** The handedness guarantee established back in
Stage 6 is what makes the closed form work — it guarantees the triad is
consistently right-handed so the third axis needs no independent correction.
And **consistency is not a check on this stage**: a pose can be completely wrong
while consistency reads 100%.

---

## 15. Cross-cutting: the status word

One buffer, cleared each frame, `atomicOr`'d by whichever pass detects a
condition, read back in the same 128 bytes as the pose.

| bit | name | set by | kind |
|---|---|---|---|
| 0 | `growNotConverged` | grow `gate`, after the last encoded round | budget |
| 1 | `regionOverflow` | collect `regionMeta` | **cap** |
| 2 | `noRegions` | collect `regionMeta` | ordinary |
| 3 | `lineOverflow` | line compaction | **cap** |
| 4 | `noVotes` | line compaction | ordinary |
| 5 | `fitDegenerate` | `fit.eigen`, an all-zero scatter matrix (**not** the 1e-9 guard, which cannot fire — §10) | ordinary |
| 6 | `gppNoSamples` | gpp classify | ordinary |
| 7 | `gppNoCandidates` | gpp candidate derivation | ordinary |
| 8 | `layoutInvalid` | decode layout, <4 valid rays | ordinary |
| 9 | `gridOverflow` | decode layout | **diagnostic** |
| 10 | `decodeNoAnchor` | decode argmax, `found == 0` | ordinary |

**Four kinds, and they must stay distinguishable.** *Ordinary* means "this frame
does not contain a decodable board" — expected, not a bug. ***Cap*** means "raise
a constant." *Budget* means "the round count was too small." Collapsing these
into one "failed" bit would make a real capacity problem look like an ordinary
empty frame.

***Diagnostic*** is the fourth and `gridOverflow` is the only one, added when §12
was measured. It means "the raw line hull exceeded one board period and the
lattice was clamped" — **and the frame decoded correctly anyway**. It is not a
cap bit: raising a constant is exactly the wrong response, because a second board
period is redundancy rather than evidence and the clamp changed no recovered pose
at any of 204 measured poses. It is not ordinary either, because it is a real
statement about the frame (the camera is seeing 144+ cells, i.e. ~3 px per cell
at 480 px, which is at the edge of detectability). Treat it as a strong smell,
never as a failure.

**Bits 1 and 3 are the cap bits, and they are the ones that mean "raise a
constant".** `regionOverflow` is derived in `finish` from `counts.x > maxRegions`
— which closes the objection §15's own table used to carry, that collect does not
bind `status`. It does not need to: `finish` binds the buffer holding the
evidence.

---

## 16. Cross-cutting: initialization

**Buffer reuse means every buffer starts the frame holding the previous frame's
bytes.** A missed clear does not fail loudly — it produces a count that is this
frame's plus some earlier frame's, larger every reconstruction, plausible for a
while, and wrong in a way that looks like a detection-threshold problem three
stages downstream.

**Clear to zero:** `changed`, `labelSurvives`, `labelCounts`, `cursor`,
`maxWeight`, `hist`, `totalWindows`, `result`, `status`, and **every
indirect-args triple**. Not at frame start — immediately before the buffer's
first bind, which `planPool` derives (§18).

### The four this list used to contain, and why they came off

A first draft also named `lineCount`, `familyCounts` and *"every scan's
`blockOffsets` at every level"*. **None of them needs a clear, and the reason is
the same for all four**: the scan's `blocks` entry point writes `dst[i]`
unconditionally for every `i < count`, and both the block offsets and the grand
total are written by a pass whose count is a host constant. They are fully
overwritten every frame by construction.

`counts`, `lineCount` and `familyCounts` are therefore *not* marked `zero` in
`pipeline.ts`, and that is correct rather than an oversight — **do not "fix" it.**
Clearing them would be harmless but would hide the real rule, which is the one
above it:

**What DOES need care is a scan's INPUT** — `kept`, `lineFlag`, `family` — because
its producer may iterate a smaller domain than the scan does. `kept` is safe
(written for every label in both branches of `markKept`), `lineFlag` is safe
because §9 made it so deliberately, and `family` is **not safe as declared**. See
bite-you-first #5.

**Two that are NOT zero**, and so will not be caught by grepping for the clear (it was three; see the middle row):

| buffer | correct initial state | what goes wrong otherwise |
|---|---|---|
| `grow.args` | `[gx, gy, 1, 0]` | A converged run leaves it `[0,0,0]`. The next frame dispatches zero rounds and emits `init`'s singleton labelling — **one region per eligible pixel, at full speed, with no error** |
| ~~`uvBounds`~~ | ~~`[+∞, −∞, +∞, −∞]`~~ | **DELETED**, not fixed. §12's line hull replaced the atomic min/max this was the init target for, and the reduction that replaces it writes all its lanes from one lane — total by construction, needing neither clear nor init. Third time that move has deleted a buffer, after `ataPartials` and `gpp.extentInit`: **an initialization pass is usually a symptom of choosing an atomic where a reduction would do** |
| `maxWeight` | `0` | Correct as a clear, but *only* because it is an atomicMax over FLOAT BITS and an arc length is non-negative — the IEEE-754 pattern of a non-negative float orders the way the float does, and does not for negatives. Said in `VOTES_WGSL`, because it looks like the same pattern as the two above and is not |

**Any buffer consumed as indirect args wants the clear regardless of what writes
it.** The case it covers is the pass that writes it *not running* — a bind group
that fails validation makes every command using it a silent no-op, while the
downstream indirect dispatch is a separate, valid command that will happily
launch over the previous frame's extent.

**The greppable shape:** an early return that writes one field. Known instances:
`lsdFit`'s degenerate-region path, and (to watch for) the eigen degeneracy guard
and the decode layout's invalid-bounds path.

**The cheapest test for this entire class: run a reconstruction twice and compare
the results.** Every bug in it is "run 2 inherits run 1's bytes," so a single run
— which is what a harness does by default — cannot see any of them.

---

## 17. Memory summary

**Measured from the declaration, not estimated** — `planPool` at 480×640 with
`maxRegions` 16,384, after the collect redesign in §7:

**CORRECTED 2026-08-13.** The previous version of this table claimed a transient
total of 18.44 MiB and a 89% full-image share. Both were wrong: the "decode +
hist" row read 0.128 MiB and omitted `packed`, which is 1.0 MiB on its own (§13
had it right; the summary row did not). The numbers below are printed from
`planPool` rather than transcribed, which is what the table always claimed to be.

| group | MiB | share |
|---|---|---|
| **Full-image arrays** (10 × `n·4B`, 2 × `n·8B`) | **16.41** | **82%** |
| gray | 1.172 | |
| fx, fy | 2.344 | |
| grow: ux, uy, label, next | 4.688 | |
| collect: labelSurvives, labelCounts, members | 3.516 | |
| collect: kept, keptScan (`n·8B`) | 4.688 | |
| **Everything else, transient** | **3.64** | 18% |
| collect: per-region + scan temporaries | 0.319 | |
| lsdFit rects | 0.625 | |
| lines + votes | 0.750 | |
| fit (`ata` + `triad`) | 0.000 | |
| gpp | 0.626 | |
| decode (`packed` 1.000, `hist` 0.316, rest 0.005) | 1.321 | |
| **Transient total** | **20.05** | |
| **With pool aliasing** (§18) — MEASURED on the complete pipeline | **15.11** | |
| Persistent (torus, hash table) | 0.58 | |

**Against ~30 MiB measured in the current implementation.** The collect redesign
took ~2.6 MiB off the unaliased total on its own. (The pre-redesign figure this
used to quote came from the same undercount, so only the delta survives the
correction — the before-state is no longer measurable.)

### Two honest readings

**Memory is not the reason to do this.** ~30 → ~20.0 MiB unaliased, ~16.0
aliased. Most of the delta is buffers that were simply oversized in the current
code — `lsdFit.out` (−5.5) and `regionOffsets`/`regionSizes` (−2.2) — and **both
of those savings are available today without a rewrite.** The reason to do this
is the crossings and the code volume, not the footprint.

**82% of the footprint is full-image data that no cap touches**, and "no
pessimistic worst-case sizes" only addresses the other 18% — most of which is
`packed`, whose cap §12 proposes to derive rather than choose. The levers against
the 82%, in order of value:

| lever | saving | cost |
|---|---|---|
| **pool aliasing** (§18) | **4.08 MiB** | a computed slot assignment; clears must be stage-local |
| drop `ux`/`uy`, normalize inline | 2.344 MiB | one `inverseSqrt` per neighbour test per round |
| f16 gradients | 1.172 MiB | precision, near the grow tolerance boundary |
| u8 gray, packed 4/word | 0.879 MiB | unpack in 3 consumers |

Aliasing saves less than the 7.03 MiB an earlier draft claimed, and the reason is
worth noting: **the collect redesign removed most of what aliasing was saving.**
Four `n·4B` arrays became two `n·8B` ones that cannot share a slot with each
other, and `cursor` stopped being full-image at all. Deleting a buffer beats
sharing it.

The label-space-is-pixel-space property is inherent and should not be attacked.
The *number of arrays over that space*, and how many exist **at once**, are both
negotiable — and the second is much cheaper than the first.

---

## 18. Buffer pooling by liveness

The full-image arrays are 82% of the footprint, but **they are not all alive at
once.** The pipeline is a straight line, so each buffer's live range — first bind
to last bind — is a genuine interval, and buffers whose intervals do not overlap
can share one allocation.

### The live ranges

Measured from the declaration (`computeLiveness`), not written by hand:

| array | born | dies |
|---|---|---|
| `gray` | upload | decode.build |
| `fx`, `fy` | gradient | lsdFit |
| `ux`, `uy` | grow.init | grow.gate |
| `next` | grow.init | grow.gate |
| `label` | grow.init | collect.scatter |
| `labelSurvives`, `labelCounts` | collect.tally | collect.markKept |
| `kept` (`n·8B`) | collect.markKept | collect.scatter |
| `keptScan` (`n·8B`) | kept.scan | collect.scatter |
| `members` | collect.scatter | lsdFit |

**Peak concurrent liveness at `n·4B` is 7**, and at `n·8B` it is 2 — the scan's
source and destination, which can never share since a scan reads one while
writing the other. So ten `n·4B` arrays fit in **seven slots**, taking the
pipeline total from 20.05 MiB to **15.97 MiB**. Printed by `planPool`, the
sharing is exactly `next | labelCounts | members` and `ux | labelSurvives`;
`gray`, `fx`, `fy`, `label` and `uy` each hold a slot alone.

That saving is smaller than the 7.03 MiB an earlier draft of this section
claimed, and the reason is the point: the collect redesign *deleted* most of what
aliasing had been saving. `labelCounts` used to live to `regionMeta`, four
separate `n·4B` scan arrays used to overlap at `scatter`, and `cursor` used to be
full-image. Aliasing is worth doing, but it is the second-best lever — deleting a
buffer beats sharing it, and it is the only one of the two that also removes
code.

### Why the stages do not need to know

Live ranges on a straight-line pipeline are intervals, so the conflict graph is
an **interval graph**. Greedy left-to-right colouring is provably optimal on
those, and the colour count equals the peak concurrent liveness. **The
assignment is computed, never hand-written.**

The WebGPU usage-scope hazard — one buffer bound twice in a dispatch, which is a
validation error reported asynchronously as a silently no-op encoder — is
**subsumed, not a second thing to remember**:

> If a pass binds two buffers, both are live during that pass by definition. A
> correct interval colouring never gives two simultaneously-live buffers the
> same slot. Therefore no pass can bind one buffer twice.

**Derive the colouring from the declared bind lists, not from a hand-written
table.** If the pipeline is a list of `(stageName, bindList)` pairs — which it
already is, since every stage builds a bind group from an ordered array — then
liveness is first-appearance to last-appearance and the colouring is ~20 lines.
A table derived from the code cannot disagree with the code.

Then add the assertion: walk every bind group at creation and throw if two
entries resolve to the same `GPUBuffer`. ~15 lines, runs once, converts a silent
async validation failure into a loud startup throw.

### THE ONE TRAP: liveness is defined by BINDING, not by USE

Grow's four entry points share one explicit bind-group layout, so `compress`
*binds* `fx`/`fy` without reading them. Computing liveness from what a shader
actually touches would free `fx` too early and hand `compress` an aliased buffer
inside its own bind group — the exact failure the colouring is supposed to make
impossible. **Take the interval from the bind list.**

### What this costs

**The clear schedule must be stage-local**, which is rule 7 in §2 and is worth
doing regardless. `labelCounts` is an `atomicAdd` target that must start zeroed;
if it shares a slot with `next`, a frame-start clear is useless because that
memory is still `next` at frame start. The clear belongs immediately before
`histogram`.

The failure mode also changes shape, and arguably improves: a missed clear gives
you *another array's data* (labels where counts should be) rather than last
frame's plausible-looking counts. Weird is easier to notice than subtly wrong.

**Reading an intermediate after the pipeline finishes pins its slot.** On a
pose-only pipeline there are no post-hoc reads, so this cannot arise. When
overlays come back, the rule is "a buffer someone reads later is live to end of
frame", which the colouring handles by construction — it just uses more slots.

**Debugging gets slightly worse.** A buffer dump where `next` and `labelCounts`
are the same memory is confusing. Label the pooled buffers with their full alias
list: `createBuffer({ label: 'pool[3]: next | labelCounts | cursor' })`.

### The flag

`alias: false` is **not a second code path** — it is the same colouring with a
degenerate liveness table where every buffer is live to end of frame. One line.

**Ship v1 with it OFF.** A few MiB on a budget nothing is hitting, against a
validation-failure risk. Turn it on when resolution goes up: at 1080p `n·4B` is
7.91 MiB, so 15 arrays is 119 MiB against 71 MiB for 9 — that is the difference
between comfortable and not.

### RUN 2026-08-14, AND IT IS CORRECT

Measured at 480x640 with the pipeline complete:

```
alias=false   53 slots   19.50 MiB
alias=true    36 slots   15.11 MiB      -4.39 MiB, -23%
```

Eleven slots hold more than one buffer. The two that matter are the full-image
ones — `next | labelCounts | members` and `ux | labelSurvives`, 1.17 MiB each —
and the rest are the small per-region and per-line arrays, where `meanDirs |
lineFlag | family` and `lines | colSamples` are the notable pairs.

**The self-check ran, both ways, and agrees exactly.** `npm run sweep --
--pipeline pose2 --alias` over the same 180 poses reports the SAME accuracy at
every figure and every tilt as the unpooled run: 180/180 recovered, 180/180
anchor exact, sub-cell median 0.051 / max 0.155, height and period median 0.12%.
Time is unchanged within noise (13.9 ms median against 13.5).

**And the check is decisive, which was verified rather than assumed.** A test
runs one frame both ways and compares all 128 bytes — the pose, the quaternion
and every diagnostic — after first asserting that pooling actually SHARED
something (36 slots against 53). Mutating `computeLiveness` so a live range ends
one stage early fails it immediately, and that mutation is §18's own named trap
in miniature: liveness defined by anything narrower than "last bind" hands a pass
an aliased buffer inside its own bind group.

The diagnostics are in the comparison deliberately, not just the pose: a stage
that read the wrong slot shows up in a region count or a line count long before
it moves the recovered position.

---

## 19. Verification, and the trajectory to replacement

**THE END GOAL IS REPLACEMENT.** This pipeline replaces `src/pose/` entirely.
It is not a spike, not a parallel experiment, and not a proof of concept. That
matters because it sets the bar: the new pipeline has to be shown at least as
accurate as the old one across the operating range, not merely shown to run.

### THE ACCEPTANCE CRITERION IS A POSE SWEEP

The deliverable that decides whether this rewrite succeeded:

> Sweep a wide set of camera positions and orientations in virtual space.
> For each pose, render what the camera would see, run the pipeline, and compare
> BOTH the final pose and every intermediate against ground truth. Report
> accuracy and timing, and show where the algorithm breaks down internally.

**Design this FIRST, not last.** Every other verification question -- which tests
are worth writing, whether the pipeline needs a CPU twin, what counts as "done"
for a stage -- is downstream of it. Building stages first and asking how to
verify them afterwards is how verification debt accumulates, and it is the
mistake this section exists to prevent.

The sweep gives something no unit test can: **ground truth by construction.**
The image was rendered FROM a known pose, so the true axes, height, period,
phase, anchor and camera pose are all known exactly -- not approximated by a
second implementation.

### The simulator does NOT need three.js

The existing simulated camera renders through the WebGL scene, which would tie
the sweep to a browser. It does not have to.

A flat textured plane under a pinhole camera is closed-form: per pixel, cast a
ray, intersect the floor, look up the torus cell, write the bit. Roughly 50
lines, runs headless, and it is at PARITY with what exists -- the current
simulator has no lens model either (see the BoofCV work list, where adding one
is item 1). Supersampling for antialiasing and optional noise/blur are additions
on top, not prerequisites.

This is the single enabler that makes a headless sweep possible at all.

### WHICH ORACLE EACH STAGE NEEDS

Ground truth covers most of the pipeline but genuinely does not cover all of it,
and the split is principled rather than a matter of convenience:

| stage | truth derivable from the pose? | oracle |
|---|---|---|
| gradient | yes, analytically from the image | synthetic |
| **grow** | **NO** -- a labelling is an artifact of the algorithm, not a property of the scene | **CPU reference** |
| **collect** | **NO** -- same reason | **CPU reference** |
| **lsdFit** | partial -- a detected line must LIE ON a true grid line, but WHICH lines get detected is the detector's business | CPU reference + geometric check |
| votes | yes -- each normal is perpendicular to a true floor direction | ground truth |
| fit | yes -- the true Drow / Dcol / Dnormal | ground truth |
| gpp | yes -- true period and phase | ground truth |
| decode layout | yes -- true lattice extent and phase | ground truth |
| decode tally | yes -- the true anchor | ground truth |
| finish | yes -- the true position and quaternion | ground truth |

**So the LSD front half needs a CPU twin and the back half does not.** In
practice that is `regions.cpu.ts` + `rectangles.cpu.ts` -- a fraction of the
2,255 lines, kept for a specific reason rather than out of caution.

### The CPU-reference tension is not real

Two different things, and only one of them is what this rewrite deletes:

- **A CPU implementation INSIDE the pipeline** -- a backend flag, a fallback
  path, stages drawn to correspond 1:1 across two implementations. That is what
  cost the line count AND shaped the decomposition, and it is what "no fallback"
  removes.
- **A CPU implementation OUTSIDE, as a test oracle.** Costs nothing
  structurally. The new pipeline never imports it and cannot know it exists.

Keeping the second while deleting the first is not a compromise between them.

### THE PHASES

**Phase 0 -- unblock. DONE 2026-08-12.**
The symptom was that the SECOND pipeline run in a process died -- segfault or
hang -- which read as "Dawn cannot do more than one run per process" and would
have been fatal, since a sweep is hundreds of runs in one process.

**It was our bug, not Dawn's, and the shape of it is worth keeping.** The test
helper called `create([]).requestAdapter()`, dropping the only reference to the
GPU instance that owns the adapter and device. Once V8 collected it, the native
side was freed underneath a still-live device and the next `mapAsync`
segfaulted. Retaining the instance at module scope fixes it: **200 consecutive
grow runs, clean exit.**

Two lessons, both general:
- **It presented as flaky infrastructure, not as a bug.** The crash landed
  wherever GC happened to run, so one test passed, two passed sometimes, and a
  longer run died. That pattern invited "the tool is unreliable, work around
  it" -- and three separate workarounds (dropping buffer destroys, memoizing
  bind groups, sharing one buffer set) all failed to help, which should have
  been the tell that the diagnosis was wrong.
- **A false claim reached this document and a source comment** before it was
  caught. Both are corrected in place rather than deleted.

**Phase 1 -- the sweep skeleton, BEFORE the pipeline is finished.**
- `renderPose(pose, dims) -> gray` (closed-form, above)
- `truthFor(pose) -> { Drow, Dcol, Dnormal, height, period, phiRow, phiCol, anchorRow, anchorCol, camPos, quat }`
- a pose generator sweeping height / tilt / yaw / lateral offset
- a report: per-pose, per-stage error and timing

Built early so every subsequent stage lands verified, rather than accumulating
debt against a harness that does not exist yet.

**Phase 1 -- BUILT 2026-08-12** (`src/pose2/sim.ts`, `tests/pose2Sim.test.ts`).
`renderPose` casts one ray per pixel through `cornerDir` -- the SAME function
the pipeline projects with, so the two agree by construction rather than by
inspection -- intersects the y=0 floor, and looks the cell up in the tiling
torus. Supersampled, because a hard-aliased edge produces a gradient field no
line detector could work with. `truthFor` returns camPos, quaternion, height,
period, anchor and the floor axes in both world and math frames.

**The simulator was validated against the EXISTING CPU pipeline before being
trusted to judge anything.** That ordering matters: a renderer with a sign or
axis error would report pipeline "error" that is really its own, and from
inside a sweep the two are indistinguishable. At nadir the mature pipeline
recovers the generating pose to **0.038 cells** with **consistency 1.000** and
**period error -0.01%** -- which no incorrect projection produces.

**A FINDING THAT WAS RETRACTED, kept because the retraction is the lesson.**
The first sweep reported the existing pipeline making whole-cell anchor errors
at tilts 5-20 -- dz moving in integer steps, correlated with consistency drops,
while dx stayed under 0.09. It was written up here as a real accuracy result
about `src/pose`.

**It was not real.** It was the renderer under-sampling diagonal edges; see the
near-miss below. At the corrected sampling rate the anchor is exact at every one
of 180 poses. The original numbers are gone rather than preserved, because a
wrong measurement kept "for reference" is worse than none.

**THE BASELINE, measured 2026-08-12** -- `src/pose`, cpu backend, 480x640,
supersample 4, 180 poses (heights 6/10/16/24 x tilts 0/10/20/30/40 x yaws
0/35/90 x 3 board neighbourhoods):

```
recovered      180/180
anchor exact   180/180

                 median      p90       max
sub-cell err      0.051     0.097     0.155   cells
height err        0.12%     1.25%     1.85%
period err        0.12%     1.26%     1.89%
time               31.7      59.9     111.4   ms

by tilt:   0   0.058 cells   0.01% height   cons 1.000    29.8 ms
          10   0.043         0.08%          cons 1.000    28.2
          20   0.050         0.13%          cons 1.000    26.9
          30   0.042         0.41%          cons 1.000    31.6
          40   0.065         1.25%          cons 0.999    63.8
```

**This is the number `src/pose2` has to match or beat**, and it is the first
absolute-accuracy measurement this project has had -- the existing goldens say
"unchanged", not "correct", and the true pose behind the fixture was never
known. Degradation with tilt is visible and gentle: height error grows 100x from
nadir to 40deg while position stays flat, which says the period search loses
precision at grazing before anything else does.

**THE NEAR-MISS, and it is the most important thing Phase 1 produced.** The
sweep's first full run reported mass line-detection failure at yawed poses --
2 votes where there should be ~170, a period 3.26x wrong, consistency at the
chance floor. That reads unambiguously as a detector problem. **It was the
renderer.** Supersampling, at 480x640 / height 10 / nadir / yaw 35:

| supersample | votes | period ratio | consistency |
|---|---|---|---|
| 1 | 0 | -- | no pose |
| 2 | 2 | 3.26 | 0.488 |
| **4** | **171** | **1.000** | **1.000** |
| 8 | 171 | 1.000 | 1.000 |

A DIAGONAL edge at 2x2 is a staircase, and the level-line directions it yields
are quantized to that staircase rather than to the true edge -- so directed
growth splits one line into many at a 9.5deg tolerance. **Axis-aligned edges
survive it**, which is exactly why the nadir yaw-0 validation passed cleanly and
hid the problem. The default is now 4.

**The general lesson, which outlives this instance:** a validated simulator is
validated *at the poses you validated it at*. Nadir agreeing to 0.038 cells said
nothing about diagonal edges. Before believing any sweep result that looks like
a pipeline failure, re-run the pose at ss=8.

**Position error decomposes, and keeping the halves apart is what makes the
sweep readable:** CONTINUOUS (period, height, sub-cell position -- a renderer
bug lives here) versus DISCRETE (whole-cell anchor jumps -- a pipeline property).
Score them separately.

`src/pose2/sweep.ts` + `scripts/sweep.ts` (`npm run sweep [--quick]`) are the
generator and report. The sweep is deliberately PIPELINE-AGNOSTIC -- a `Runner`
takes a grayscale image and returns what it recovered -- so the same harness
scores `src/pose` and `src/pose2` and the two sets of numbers are directly
comparable. That is the only way "is the rewrite at least as accurate" gets an
answer instead of an opinion.

Two things the sweep needed that are worth not re-deriving:
- **`cornerDir` allocates a Vector3 per ray.** A 480x640 frame at 4x4 is 4.9M
  rays; the allocation alone exhausted a 4GB heap. `rayDirInto` is that function
  written out into scalars, held to the original by a BIT-FOR-BIT test rather
  than by sharing the call -- and writing the quaternion sandwich in the
  familiar form instead of THREE's exact operation order made that test fail.
- **`runPoseOn` drains fx, fy, regionId, the region CSR and the decode grid**
  into host arrays. A sweep discards all of it; call `computePoseFromCapture`.

**Phase 2 -- stages in pipeline order, each landing with the oracle above.**
collect and lsdFit against the CPU twin; then votes, fit, gpp, decode and finish
against ground truth. **COMPLETE as of 2026-08-14.**

### THE ACCEPTANCE CRITERION IS MET — 180 poses, both pipelines, 2026-08-14

`npm run sweep -- --pipeline both` renders each pose once and runs BOTH pipelines
over it. 480x640, supersample 4, the same fixture settings for both:

```
                   src/pose (cpu)              src/pose2 (gpu)
recovered            180/180                     180/180
anchor exact         180/180                     180/180
                 median   p90    max         median   p90    max
sub-cell err      0.051  0.097  0.155         0.051  0.097  0.155   cells
height err        0.12%  1.25%  1.85%         0.12%  1.25%  1.85%
period err        0.12%  1.26%  1.89%         0.12%  1.26%  1.89%
time               27.9   62.2  111.9          13.5   16.7   51.3   ms
```

**Every accuracy figure is identical to the printed precision, and the per-tilt
breakdown matches digit for digit at all five tilts.** The rewrite is not "at
least as accurate" -- it is the same answer, roughly 2x faster at the median and
3.7x at p90, where the old pipeline's timing spread is the readback stalls.

**THAT RESULT IS TOO CLEAN TO ACCEPT FROM A ROUNDED SUMMARY, and it was checked
rather than believed.** Identical medians across 180 poses is exactly what a
harness silently running one pipeline twice would print. Two independent facts
say it is not:

- the timing column differs by 2x, so a different code path certainly ran; and
- **the recovered positions were diffed at full precision**, which is the check
  that actually settles it. At tilt 20 yaw 35 h=10 the two pipelines disagree by
  **4.7e-6 cells**; at tilt 40 yaw 90 h=24, by **1.5e-4 cells** — while both
  differ from TRUTH by 0.03 and 0.30 cells respectively.

So the f32-versus-f64 divergence is three to four orders of magnitude below the
recovery error, and the summary rounds it away. The reason it is that small is
not luck: the LSD front half is deterministic and both implementations find the
SAME lines (which is what `pose2Cpu.test.ts` asserts region-for-region and
rectangle-for-rectangle), so the votes agree to f32 rounding and everything
downstream inherits that.

**What this does NOT establish.** The sweep is synthetic: a pinhole camera with
no lens model, no noise and no motion blur (open decision 6, and the BoofCV work
list's item 1). Agreement here says the port is faithful, not that either
pipeline is right about a real capture.

**The twin exists -- `src/pose2/cpu.ts`, gradient + grow, 2026-08-12.** It is a
BFS from each unvisited seed, not a transcription of the shader's hook-and-
compress. That is the point: a twin copied from the implementation shares its
mistakes and tests the port rather than the algorithm. Because hook takes the
minimum neighbouring label and BFS labels by component minimum, the two produce
identical label *arrays*, so nothing has to canonicalize a partition. It also
reports `marginalPairs` -- neighbour pairs whose dot sits within 1e-4 of
cos(tolerance) -- so a test can show the f32-vs-f64 comparison was decisive
before asserting exact equality. On a rendered board frame that count is zero and
every label matches.

**The twin was mutation-tested before being believed, and that changed a test.**
Four deliberate shader bugs, and what caught them:

| mutation | diagonal-bar test | rendered-frame test |
|---|---|---|
| `abs()` in the predicate (unsigned dot) | **missed**, now caught | caught |
| 4-connectivity instead of 8 | missed | caught |
| hook never adopts a neighbour's label | caught | caught |
| swap the level-line components | **cannot be caught** (below) | cannot be caught |

The bar test missed the unsigned-dot bug because the bar was 4px wide, which puts
its two opposite-polarity edges two columns apart -- they never become
8-neighbours, so the predicate is never asked the question the test claims to
ask. **A one-pixel bar makes the two edges adjacent and the test catches it.**
The comment now says what it actually tests.

**Grow is invariant to any global rotation of the level-line field**, because its
only use of `(ux, uy)` is a dot between two of them and a dot is rotation-
invariant. Swapping the components, negating both, or using the gradient itself
instead of its perpendicular all leave the labelling bit-identical -- verified,
not assumed. This is a true property of the stage, not a hole in the twin; the
convention is observable only downstream in `lsdFit`, which uses the direction
itself. Do not add a grow test claiming to check it. (It is also the bug the
shader's own comment records from when it was real.)

**Collect is BUILT and green against the twin, 2026-08-12** — five passes plus
the shared vec2 scan, redesigned rather than transcribed (§7). `cpuCollect`
bucket-sorts by label in one pass where the GPU tallies, scans, atomically
scatters and sorts; the two agree region-for-region on a rendered frame with no
canonicalization, members included, and the region/member counts come back from
the device with no host involvement. Mutation-checked: dropping the block-offset
add pass, deleting `finalize`'s sort, and weakening hysteresis to `rhoLow` are
each caught.

Two things the buffer tests caught that the implementation did not:
- **A claim in this document was wrong.** The scan's one-workgroup spine was
  said to hold at 4096×4096; it needs 16,384 blocks against a 4,096 ceiling. The
  real limit is 2048×2048, and `planPool` now throws past it.
- **`assertBinds` had an unreachable slot check**, and the test covering it was
  passing on a *different* error. Deleted, with the composition it was
  duplicating written down in its place. A colliding plan turns out to be
  unconstructible through `planPool` at all — adding a buffer to a stage extends
  its live range, so the colouring just stops sharing the slot.

**lsdFit is BUILT and green against the twin, 2026-08-12** — one pass, the
decomposition unchanged by re-derivation, with the full mutation run and its two
findings written up in §8. It is the last stage a twin is the right oracle for;
everything after it is scored by ground truth.

**gpp is BUILT and scored by ground truth, 2026-08-13** (§11) — seven passes plus
the shared scan's third use, recovering the period to 0.13-1.6% at every pose
tried outside one whose upstream axis error is double the rest. Its mutation run
is where a NEAR MISS was found rather than a hole: a distinctness margin gate set
by eye at 1.5 against a mutated value of 1.53. The general form — **measure what
the mutation actually does to the number before deciding the gate can see it** —
is the thing to carry forward, and it also reclassified four survivors from
"untested" to "true property, here is the measurement".

**fit is BUILT and scored by ground truth, 2026-08-13** (§10) — two passes rather
than three, because `fit.reduce` and `ataPartials` both turn out to exist only
for a host that is gone, and the indirect form of `fit.ata` was carrying the
third instance of §9's iteration-domain defect. Its mutation run is where the
"disable the correction and search for an input where the raw answer is wrong"
recipe came from, and it found both sign conventions invisible to assertions
written specifically for them.

**The geometric half of lsdFit's oracle is still owed.** §19's table calls for
"CPU reference + geometric check", and only the first half exists. The check the
twin structurally cannot make is that a detected line **lies on a true grid
line** — which the twin has no access to and ground truth does. That belongs in
the sweep, against `truthFor`, and it is the one place a rectangle that is
self-consistently wrong would show up.

One process note: `src/pose2/` is still untracked, so `git diff` is blind to it.
The mutation runs were restored from a file copy and verified with `diff`.

**Phase 3 -- replace.** Swap the app onto this pipeline, then delete
`src/pose/`. NOT STARTED. Two things to know before it is:

- The entry point is `src/pose2/run.ts`, which returns a `Pose2Result` -- a plain
  struct, no THREE types, no chain, no timing DAG. The app boundary (the payload
  mailbox, `camera.pose`) is §22's territory and is not covered here.
- **`scripts/hull-measure.ts` imports `src/pose`** and would break. It is a
  measurement harness whose result is already recorded in §12, not a shipping
  path, so deleting it with `src/pose` is a legitimate option -- but it is the
  only way to re-run that measurement.

### Calibration note

Sixteen unit tests for the buffer planner was over-production for the simplest
component in the system. **The sweep is the primary oracle. Stage-level tests
exist only where the sweep cannot localize a failure** -- which, per the table
above, means the LSD front half and anything with a hand-checkable closed form.
Left in place because they are written and green, recorded here because the
density was wrong.

**THE PRECONDITION IS NOW MET.** That paragraph used to say the sweep could not
score `src/pose2` at all until `finish` (§14) landed. It has, so the priority
inverts as promised: **run the sweep first, and let it say which stage tests were
worth having.** That re-reading has not been done -- the stage tests were all
written before the sweep could run, and nothing has yet been retired on its
evidence.

### Practical obstacles, both real

**Headless WebGPU works, with caveats.** Verified: 112 tests green, including
GPU stage tests, GPU-vs-twin tests and back-to-back-runs checks. `@webgpu/dawn-node` is NOT installable
(it lives in the Dawn source tree and was never published). The `webgpu` package
-- github.com/dawn-gpu/node-webgpu, maintained by the Dawn and spec authors -- is
that binding, packaged, with the binary bundled. `@kmamal/gpu` was tried first
and rejected: a process holding one of its instances never exits.

Three settings changes it forced, each worth knowing:
- `skipLibCheck: true` in tsconfig -- `@webgpu/types` conflicts with the DOM
  WebGPU interfaces TypeScript already ships.
- `--test-force-exit` on the test script -- creating a device, with no other work
  at all, segfaults at process teardown.
- `npm approve-scripts webgpu` after a fresh clone, or the binary stays
  quarantined on macOS.

**Validation failures are silent.** A WebGPU validation error does not throw --
it makes every command using the offending resource a no-op, so the symptom is
plausible zeros. Confirmed live on the first probe: a `layout: 'auto'` mismatch
across two entry points returned zeros with nothing raised anywhere. Every test
body therefore runs inside an error scope; a test that only asserted on numbers
would PASS against a pipeline that never ran, and for most of this pipeline zero
is a plausible answer.

## 20. Code size

| | current `src/pose` | proposed | **built, complete** |
|---|---|---|---|
| WGSL | 1,316 | ~1,700 | 3,187 |
| CPU reference math | 2,255 | 0 in the pipeline | 467, test-only |
| Infrastructure (arena, device, timeline, timing, chain, poseCompute) | 2,141 | ~150 | 364 (`buffers.ts`) + 456 (`pipeline.ts`) + 117 (`run.ts`) + 124 (`board.ts`) |
| `.gpu.ts` encode plumbing | 1,546 | ~350 | 1,041 |
| **Total** | **7,518** | **~2,200** | **5,289 pipeline + 3,275 tests + 501 harness** |

**The ~2,200 estimate was wrong by a factor of 2.4 and should be retired rather
than defended.** It was passed with five of thirteen stages built and the final
figure is 5,289. Two separate things went into that miss and only one of them is
an overrun:

- **The infrastructure line was projected at ~150 and is 1,061.** That is a real
  miss, and most of it is `pipeline.ts` + `buffers.ts` -- the declaration and the
  planner. They are load-bearing rather than boilerplate that collapses later:
  liveness, the clear schedule and two silent-failure validation rules are all
  derived from them, and three of this rewrite's bugs were found BY re-deriving
  against the declaration rather than by a test.
- **The WGSL number is the FILE, comments included, and by now the comments are
  most of the growth.** Read it as "the file you have to open", not as
  instruction count.

What the estimate was a proxy for did happen. There is no arena, no residency, no
backend flag, no timing DAG, no chain, and no CPU implementation inside the
pipeline. The 2,255 lines of CPU reference math are 467 lines of test oracle that
the pipeline cannot import and does not know exists. Judge it on §19's sweep,
which is now measured, and on that list -- not on the line count.

## 21. Open decisions

1. ~~**Phase 0's outcome.**~~ **CLOSED** — headless works, the crash was ours,
   see §19. Kept as a numbered entry so the list below keeps its numbering.
2. ~~**`MAX_CELLS` value.**~~ **CLOSED 2026-08-14 by measurement** (§12, and
   `scripts/hull-measure.ts`). The lattice is bounded by the projected VOTE
   LINES, not the view quad: over the 180-pose baseline range the recovered
   position is identical at every pose, 0.02% of correct votes are lost, and
   consistency is never worse — while at grazing, clipping 35% of the complete
   windows costs 0.18% of the correct votes and RAISES consistency by up to
   0.17. `decode.bounds`/`decode.boundsInit` go, and `MAX_CELLS` becomes
   `torusR * torusC` = 20,736 — **measured sufficient**, not merely argued:
   clipping each edge to 144 changes no recovered pose at any of 204 poses. The
   one correction to the proposal is the phrase "by construction": the raw hull
   reaches a 169-cell edge at h=24 tilt 45-55, where the camera genuinely sees
   200+ cells and the period is right to 1.4%, so `decode.layout` must CLAMP and
   keep decoding. `gridOverflow` stays a diagnostic, never a failure.
3. **Grow round budget.** 32 encoded against 9 measured and an O(log n) ~ 19
   bound. 32 rounds is 96 passes. **The sweep now answers the affordability half:
   180 poses at 480x640 run in 13.5 ms median, against 27.9 for the CPU
   pipeline** -- so 96 encoded passes are not what dominates. The CONVERGENCE
   half is still open and is what `growNotConverged` reports; it did not fire at
   any of the 180 poses, which is evidence that 32 is enough for this pose range
   and not that it is enough.
4. ~~**How much CPU reference to keep.**~~ **CLOSED 2026-08-12** —
   `src/pose2/cpu.ts`, written fresh from the definition rather than copied from
   `src/pose`, and each function an independent algorithm rather than a
   transcription. It covers gradient, grow, collect and lsdFit, which is the
   whole of what a twin is the right oracle for (§19). Nothing in the pipeline
   imports it, and **nothing after lsdFit should be added to it** — from §9 on,
   ground truth knows the answer and a second implementation is only a second
   opinion.
5. ~~**One file or two?**~~ **SETTLED** — `pose.ts` + `pose.wgsl.ts`, and the
   split is earning itself at 485 lines of WGSL.
5b. ~~**Where the board pattern comes from.**~~ **CLOSED 2026-08-14 —
   `sphereLab/floorPattern` IS a shared leaf**, imported by `src/pose2/board.ts`,
   which is the only file in `src/pose2` that reaches outside itself. The
   argument is that the torus is DATA about the world, not an implementation:
   rule 2 deletes a second implementation of the ALGORITHM, and there is no
   CPU-versus-GPU version of a printed pattern to choose between. The decisive
   half is the failure mode — **a duplicated copy could disagree with the printed
   floor and nothing in this pipeline could detect it**: every consistency number
   would still read 1.000 while decode reported a confident, wrong position.
   That is strictly worse than the coupling it avoids.

   What is NOT imported is `src/pose`'s hash-table builder, which exists and
   works: that module is deleted in Phase 3, so importing it would make pose2
   un-shippable without the thing it replaces. The table is re-derived; the HASH
   FUNCTION is copied verbatim, being a correctness constant rather than a
   structure. **A drift between it and the WGSL probe fails silently and totally**
   -- every lookup misses, and the frame reports `decodeNoAnchor`, which is an
   ordinary outcome indistinguishable from an empty frame. The only check on that
   agreement is the end-to-end decode, and it is a total one.
6. **Simulator fidelity.** The closed-form renderer starts as a hard-edged
   pinhole projection. Supersampling, blur and noise are additions; whether the
   sweep needs them to be predictive of real captures is unknown until the first
   sweep runs against a real fixture for comparison.
7. ~~**Does `lines.flag` survive as its own pass?**~~ **CLOSED 2026-08-12 — it
   stays, and it dispatches DIRECTLY over maxRegions.** The deciding argument is
   mechanical rather than aesthetic: a scan input must be a total function over
   the scanned range, and a kernel dispatched over the region count structurally
   cannot write the tail. Full write-up in §9, including the live declaration bug
   it found (`lineFlag` was not marked `zero`) and the same defect still open on
   §11's `family`.
8. ~~**The geometric half of lsdFit's oracle.**~~ **CLOSED 2026-08-12** — it
   lands at the VOTE, not at the rectangle, and that is stronger than the sweep
   placement originally proposed. A vote normal must be perpendicular to a true
   floor axis, which is a per-line assertion against `truthFor` rather than an
   aggregate over a whole pose. See §9's ground-truth score; a rectangle that is
   self-consistently wrong fails it.
9. **The `worst < 0.05` gate in the vote ground-truth test.** ~1.4x headroom over
   the observed 3.5e-2, so it is the first assertion in §9 that will go flaky if
   endpoint precision moves. Either widen it with a measured justification or
   replace it with a quantile once the sweep says what the spread looks like
   across the pose range.

## 22. What this document does not cover

- The display and overlay paths. They read `geom`, region members and per-cell
  correctness arrays, none of which the pose path needs. If the rewrite keeps
  them, they are separate opt-in buffers and separate readbacks, and they should
  not influence any decision above.
- The app boundary — the payload mailbox, `camera.pose`, the dev bridge.
- IMU fusion.
- The lens model. Every projection here assumes a pinhole camera with one
  parameter (vertical FOV). Radial distortion breaks the "straight lines
  converge" premise in a spatially-correlated way that more lines cannot average
  out. That is a real accuracy limit and it is orthogonal to this restructure.
