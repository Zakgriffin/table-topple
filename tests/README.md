# Tests

    npm test

`node --test`, no dependencies, no build step. Node strips the TypeScript
itself.

## The two tiers

**The CPU tier runs here, headless.** The reference implementations
(`computeGradient2x2Field`, `growRegionsCCL`, `fitRegionsCPU`,
`computeSegmentVotes`, `fitPairOfPlanes`, `computeGridPeriodPhase`) are pure
functions of typed arrays and settings, and a whole `runPoseOn(input, 'cpu')`
reconstruction completes in ~60ms in node. That matters more than it sounds:
production runs none of those implementations any more, so they can rot
silently, and if they drift then the no-WebGPU fallback ships broken *and* the
reference every future GPU port is verified against is wrong. Until this suite
existed the only thing standing between them and that was a differential run in
a live browser, by hand.

**The GPU tier does not run here.** It needs a real `navigator.gpu`. Those are
the `verify*` harnesses in `src/poseViewer/harness/`, driven from the browser
console over the dev bridge:

    await verifyLsdChain(cameraInput())
    await verifyLsdChain(await fixtureInput('default'))

They take the same `HarnessInput` these tests do, so when the GPU tier does get
a runner it inherits one input format and one set of test bodies rather than a
second suite.

## Everything runs on a fixture

`fixtures/default.json` — pixels plus the config they are to be processed
under, in one validated file. See `src/poseViewer/fixture.ts`. Nothing here
reaches for a live camera, a global, or an ambient setting, which is what makes
a failure re-derivable a month later.

## Golden values say UNCHANGED, not CORRECT

Some assertions are invariants (orthonormal axes, positive distance, decode
consistency above the ~50% chance floor) and would fail on a wrong answer even
if it had always been wrong. The golden numbers are not: they were recorded
from this same code, so a bug already present is baked into them. They catch
drift, which is what they are for. Both kinds are labelled in the test names.

No wall-clock assertion appears anywhere, deliberately: a flaky perf threshold
reproduces the same "numbers you cannot interpret later" failure in a new place.
Use `timeReconstruction()` for that, and record what it says.

## What made this possible

Two constraints the codebase now holds, both of which will break this suite
first if they lapse:

- **`verbatimModuleSyntax`** is on. Node's type-stripping cannot tell a type
  import from a value import, so a type imported without the `type` keyword
  becomes a runtime import of an export that does not exist. `tsc` catches
  every one (TS1484).
- **No module-scope browser access on the pipeline's import path.**
  `floorPattern.ts` read `location.search` at module scope, which alone made
  `math/geometry.ts`, `votes.ts` and `gridPeriodPhase.ts` unimportable outside a
  browser. It is guarded now.
