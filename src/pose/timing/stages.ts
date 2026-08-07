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
  'pose.residency': { label: 'residency create (gray up)', within: 'pose.run' },

  // ── the votes stage and the LSD chain underneath it ──
  'pose.votes': { label: 'votes stage', within: 'pose.run', inputs: ['pose.residency'] },
  'pose.composites': { label: 'composites (2x2 gradient field)', within: 'pose.votes', inputs: ['pose.residency'] },
  'lsd.gradient': { label: 'gradient2x2', within: 'pose.composites', inputs: ['pose.residency'] },
  'lsd.grow': { label: 'grow+collect regions', within: 'pose.composites', inputs: ['lsd.gradient'] },
  'lsd.fit': { label: 'fit regions', within: 'pose.composites', inputs: ['lsd.grow'] },
  // These two are the halves of fitAndTestRegionsGPU, and `lsd.wrap` is the
  // re-wrap that follows it in the same function. All three sit under
  // `lsd.fit`, which exists so the Promise.all inside it has somewhere to be:
  // `lsd.fitDispatch` and the region-CSR readback run CONCURRENTLY there, so
  // they overlap, which is precisely the case a stack had to lie about.
  'lsd.fitDispatch': { label: 'lsdFit dispatch+fence', within: 'lsd.fit' },
  'lsd.fitUnpack': { label: 'lsdFit unpack (objects)', within: 'lsd.fit', sync: true },
  'lsd.wrap': { label: 'rectangle wrap (objects)', within: 'lsd.fit', sync: true },
  'votes.filter': { label: 'compositesFromLsdRectangles', within: 'pose.composites', inputs: ['lsd.fit'], sync: true },
  'votes.segments': { label: 'votes (segments)', within: 'pose.votes', inputs: ['pose.composites'], sync: true },

  // ── the plane fit ──
  'pose.fit': { label: 'fitPairOfPlanes', within: 'pose.run', inputs: ['pose.votes'] },
  'fit.dispatch': { label: 'GPU dispatch (ATA reduction)', within: 'pose.fit' },
  'fit.finish': { label: 'CPU finish (sum partials + eigen)', within: 'pose.fit' },

  'pose.assembly': { label: 'poseAssembly', within: 'pose.run', inputs: ['pose.fit'], sync: true },

  // ── distance ──
  'pose.distance': { label: 'gridPeriodPhase (distance source)', within: 'pose.run', inputs: ['pose.assembly', 'pose.composites'] },
  'gpp.classify': { label: 'classify lines (cornerDir+gnomonic x2 per line)', within: 'pose.distance' },
  'gpp.search': { label: 'period search (integer-count + distinctness + golden)', within: 'pose.distance', inputs: ['gpp.classify'] },
  'gpp.sweep': { label: 'period sweep', within: 'gpp.search' },

  // ── decode ──
  'pose.decode': { label: 'positionDecode', within: 'pose.run', inputs: ['pose.distance'] },
  'decode.fused': { label: 'decode (fused GPU build+tally)', within: 'pose.decode' },
  'decode.tallyDispatch': { label: 'GPU dispatch (4 orientations)', within: 'decode.fused' },
  // The unfused fallback's two halves. Both can coexist with `decode.fused` in
  // one run: the fused route is ATTEMPTED first and falls through when the
  // layout or the build comes back null.
  'decode.build': { label: 'buildDecodeSampleGrid', within: 'pose.decode', sync: true },
  'decode.tally': { label: 'tallyPositionVotes', within: 'pose.decode', inputs: ['decode.build'] },

  // Runs in the CALLER's drain, after computePoseFromCapture has returned, so
  // it is a root here rather than part of the pose. The app puts it inside its
  // display tail.
  'pose.drain': { label: 'intermediates drain', within: null },
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
