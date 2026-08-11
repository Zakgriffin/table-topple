import {
  type SpanAttrs, type StageNode, type StageRecord, type StageTable, spanStart,
} from '../../sphereLab/profiling/profiler.ts';

// ── The pose pipeline's stages, DECLARED ─────────────────────────────────
//
// One entry per instrumented stage: what to call it, what it runs inside, and
// whether it provably encloses no `await`. This table is what replaced the
// profiler's module-level span stack -- see profiler.ts's header for why
// declared structure beats inferred structure, and why the `within` half
// cannot be replaced by "whichever interval happens to contain this one".
//
// ── Ids are STABLE; what varies goes in attrs ──
//
// `fitPairOfPlanes (CPU)` and `fitPairOfPlanes (GPU)` used to be two different
// span names, which meant no report could put a CPU run and a GPU run of the
// same stage in one row -- exactly the comparison the differential harness
// exists to make. The backend is an attribute of an occurrence, not a
// different stage, so it is `attrs: { backend }` on a single `pose.fit`.
//
// ── `within` is CONTAINMENT; `inputs` is DEPENDENCY ──
//
// They are different graphs and conflating them is what a call-stack tree does.
// `pose.decode` runs INSIDE `pose.run` and DEPENDS ON `pose.distance`. The
// first decides how self time decomposes; the second is what the critical-path
// analysis walks, and it is the DAG the whole instrument is named for.
//
// `pose.run` declares itself a root because within the library it is one. The
// app's own table overrides that entry to put it inside `app.reconstruct` --
// composition is the consumer's knowledge, and the library naming an app stage
// would be the boundary violation this whole restructure exists to remove.
export const POSE_STAGES = {
  'pose.run': { label: 'pose (whole reconstruction)', within: null },

  // ── the votes stage and the LSD chain underneath it ──
  //
  // `pose.residency` used to head this list, for the gray upload.
  // `createLsdChainResidency` only recorded a CPU array into a map, and the
  // 1.19MB crossing actually happened later, at the first `res.gpu('gray')` --
  // which this table's own comment had noticed and worked around by relabelling
  // rather than by moving. The upload now happens where it is written, at the
  // top of runLsdChainGPU, and is charged to `lsd.gradient`.
  'pose.votes': { label: 'votes stage', within: 'pose.run' },
  'pose.composites': { label: 'composites (2x2 gradient field)', within: 'pose.votes' },
  'lsd.gradient': { label: 'gradient2x2 (incl. gray up)', within: 'pose.composites' },
  'lsd.grow': { label: 'grow regions (round loop)', within: 'pose.composites', inputs: ['lsd.gradient'] },
  // Its OWN stage now. It used to run inside `growRegionsCCLGPU` behind a
  // `collectOnGPU` boolean, so `label` was not a stage output on either backend
  // and this seam could not be addressed without threading that flag down to the
  // kernel. Splitting it cost nothing: both stages encode into the same encoder,
  // so the labeling still never crosses the bus.
  'lsd.collect': { label: 'collect regions (CSR)', within: 'pose.composites', inputs: ['lsd.grow'] },
  // ONE input, and getting back to that is why the CPU-side grow/collect split
  // had to land first. `lsd.fit`'s real input is "the regions", which BOTH
  // backends now get from `lsd.collect`. This read `['lsd.collect', 'lsd.grow']`
  // for as long as `growRegionsCCL` ran the collect inside its own call and
  // recorded no separate span: declaring only `lsd.collect` then truncated the
  // CPU critical path at `lsd.fit`, because an input that never recorded is
  // SILENT by design, so the walk terminated there and reported rectangle
  // fitting as the whole chain with the grow nowhere on it. That is the
  // identical failure 6b was built to catch, and the same test caught it. The
  // extra edge is only safe to drop because `lsd.collect` records on a CPU run
  // now -- check that before narrowing any other stage's inputs.
  'lsd.fit': { label: 'fit regions', within: 'pose.composites', inputs: ['lsd.collect'] },
  'votes.filter': { label: 'compositesFromLsdRectangles', within: 'pose.composites', inputs: ['lsd.fit'], sync: true },
  'votes.segments': { label: 'votes (segments)', within: 'pose.votes', inputs: ['pose.composites'], sync: true },

  // ── the plane fit ──
  'pose.fit': { label: 'fitPairOfPlanes', within: 'pose.run', inputs: ['pose.votes'] },
  // TWO STAGES, recording on BOTH backends -- they used to be `fit.dispatch` and
  // `fit.finish`, which named where the GPU path's awaits fell rather than what
  // was computed, and had no CPU counterpart at all. `fit.ata` is the scatter
  // accumulation (the only vote-count-dependent part, and the only part with a
  // kernel); `fit.eigen` is the fixed-size decomposition that runs on the host
  // either way.
  'fit.ata': { label: 'ATA scatter accumulation', within: 'pose.fit' },
  // The edge that pays for the whole `inputs` half of this table. On the GPU
  // backend `fit.ata` closes at SUBMIT and `fit.eigen` opens after the partials
  // readback has resolved, so the fence between them is inside NEITHER span --
  // it would sit in `pose.fit`'s self time, where nothing says what it was
  // waiting on. As a declared edge it becomes `fit.eigen`'s waitMs, attributed
  // to the stall it actually is. On the CPU backend the two are adjacent and the
  // edge costs nothing. Same shape at decode.tally and votes.filter.
  'fit.eigen': { label: 'eigen solve (6x6 -> 3x3 -> triad)', within: 'pose.fit', inputs: ['fit.ata'] },

  'pose.assembly': { label: 'poseAssembly', within: 'pose.run', inputs: ['pose.fit'], sync: true },

  // ── distance ──
  'pose.distance': { label: 'gridPeriodPhase (distance source)', within: 'pose.run', inputs: ['pose.assembly', 'pose.composites'] },
  'gpp.classify': { label: 'classify lines (cornerDir+gnomonic x2 per line)', within: 'pose.distance' },
  'gpp.search': { label: 'period search (integer-count + distinctness + golden)', within: 'pose.distance', inputs: ['gpp.classify'] },
  'gpp.sweep': { label: 'period sweep', within: 'gpp.search' },

  // ── decode ──
  'pose.decode': { label: 'positionDecode', within: 'pose.run', inputs: ['pose.distance'] },
  // The whole GPU decode, and it is now ONE SUBMIT and one 32-byte readback --
  // build, tally, argmax and correctness encoded in sequence. There is no
  // sub-span inside it any more: `decode.tallyDispatch` used to close at a
  // submit that the tally made on its own, and with all four stages sharing an
  // encoder there is no submit there to close at. What is left inside this span
  // is microseconds of encoding plus the one stall, and the stall is already
  // attributed here by the readback's owner.
  'decode.fused': { label: 'decode (GPU, one submit)', within: 'pose.decode' },
  // The CPU fallback's two halves. Both can coexist with `decode.fused` in
  // one run: the GPU route is ATTEMPTED first and falls through when the
  // layout or the build comes back null.
  'decode.build': { label: 'buildDecodeSampleGrid', within: 'pose.decode', sync: true },
  'decode.tally': { label: 'tallyPositionVotes', within: 'pose.decode', inputs: ['decode.build'] },

  // Runs in the CALLER's drain, after computePoseFromCapture has returned, so
  // it is a root here rather than part of the pose. The app puts it inside its
  // display tail.
  'pose.drain': { label: 'intermediates drain', within: null },

  // ── bus crossings ──
  //
  // Three ids for every transfer in the pipeline, because the WHAT and the byte
  // count are attributes of an occurrence, not different stages -- the old span
  // form minted `GPU→CPU readback (1191680B)` as a fresh label per byte size,
  // which no report could aggregate.
  //
  // `within: null` is the DEFAULT, not the answer: a transfer helper is one
  // call site serving a dozen stages, so its owner is passed per call and
  // overrides this (see StageRecord.within). A transfer whose caller did not
  // say lands as a root -- visible in the crossings table, absent from any
  // stage's decomposition, which is the honest reading of "nobody claimed it".
  'xfer.readback': { label: 'GPU->CPU readback', within: null },
  // Uploads never fence: mappedAtCreation is synchronous, so this is allocation
  // plus memcpy and nothing else can be interleaved into it.
  'xfer.upload': { label: 'CPU->GPU upload', within: null, sync: true },
  // f64<->f32 narrowing/widening around a crossing. Pure CPU, and per
  // fieldResidency's own note the single largest per-crossing cost.
  'xfer.convert': { label: 'f64<->f32 convert', within: null, sync: true },
} as const satisfies Readonly<Record<string, StageNode>>;

export type PoseStageId = keyof typeof POSE_STAGES;

// Widened for the join, which takes any table. The `satisfies` above is what
// keeps the literal type intact for PoseStageId while still type-checking every
// entry against StageNode.
export const POSE_STAGE_TABLE: StageTable = POSE_STAGES;

// The library's only way to open a span, and the reason it exists is the id:
// `spanStart` takes a bare string, so a typo there is a record that silently
// fails to join and shows up as `unknown`. Through here a wrong id does not
// compile. Nothing else is added -- it is the same recorder.
export function poseSpan(id: PoseStageId, attrs: SpanAttrs | null = null): StageRecord {
  return spanStart(id, attrs);
}
