import {
  type JoinResult, type Span, type SpanAttrs, type StageNode, type StageTable,
  formatSpanTree, getRecords, joinRecords, spanStart,
} from './profiler.ts';

// ── Pose Viewer's own stages ──────────────────────────────────────────────
//
// The app's own instrumented stages: the capture path, the deferred display
// tail, and the projection that only exists to feed display.
//
// ── THE LIBRARY HALF OF THIS FILE IS GONE ──
//
// This used to spread in `POSE_STAGES` from the old pipeline's `timing/stages.ts` and then
// apply two COMPOSITION OVERRIDES: the library declared `pose.run` and
// `pose.drain` as roots because within the library they were, and the app
// restated them as living inside `app.reconstruct` and `app.tail`. That seam was
// the interesting part -- the library hands back facts, the consumer says where
// they go.
//
// the old pipeline is deleted, so both the table and the overrides went with it.
// `src/pose` does not declare profiler stages at all: it is one submit and one
// readback, timed as a whole on the host and by GPU timestamps on the device, so
// there is no host-side span tree left for the app to compose with. If per-stage
// pose timing comes back, it comes back through that mechanism, and this file is
// not where it lands.
// ── THE `inputs` EDGES AND THE CRITICAL PATH ARE GONE ──
//
// Every stage here used to also declare the stages whose output it consumes, a
// second graph distinct from `within`, walked to produce a critical path with a
// `waitMs` on each edge.
//
// That answered "which dependent chain sets the floor" for a pipeline with a
// dozen interleaved GPU readbacks, where two stages could be genuinely
// concurrent and the wait between them was invisible inside either span. THAT
// PIPELINE IS DELETED. `src/pose` is one submit and one fence, so what is left
// at this level is capture -> pose -> project -> overlays, strictly serial,
// plus the display tail as an independent root. A graph walker for three edges
// is machinery with no question to answer.
//
// The concurrency that DOES remain is CPU-versus-GPU, and that is not a
// dependency graph either -- it is one number, computed directly from the GPU
// spans against the host span that submitted them.
//
// If a future pipeline reintroduces overlapping awaits this comes back, and
// nothing about removing it now is irreversible: the records are flat and the
// ids are stable, so it can be rebuilt against records already captured.
//
// One thing the edges carried that prose now has to: applyPoseVisualizations
// MUST stay downstream of the projection, because the floor quad is sized from
// that frame's bins. That constraint lives in its call site's own comment in
// axesReconstruction.ts, which is where it was before it was ever an edge.
const APP_STAGES = {
  // ── the capture path ──
  'app.reconstruct': { label: 'axesReconstruction', within: null },
  'app.capture': { label: 'capture+preprocess', within: 'app.reconstruct' },
  // The whole pipeline as ONE span, which is all the host can see: pose is one
  // submit and one fence, so this is upload -> submit -> the map resolving, with
  // no interior the host could time. Per-stage numbers, if they come back, are
  // GPU timestamps and do not land here.
  'app.pose': { label: 'pose (submit + readback)', within: 'app.reconstruct' },
  // A real stage, not a gap: it is opened when a physical camera finishes and
  // closed when the next frame arrives, so its duration is the shutter-to-
  // shutter idle the auto-capture interval is spending.
  'app.idle': { label: 'idle (waiting for next frame)', within: null },

  // ── the phone link, measured ON THE PHONE ──
  //
  // Roots, not children of `ingest.run`: they happened BEFORE the frame
  // arrived, on another machine, so no desktop span contains them. Both are
  // `peer` spans whose endpoints are on the phone's clock -- see
  // profiling/clocks.ts for why there is no `link.transit` to sit beside them.
  'link.pull': { label: 'phone: pull video frame', within: null },
  'link.encode': { label: 'phone: JPEG encode', within: null },

  'ingest.run': { label: 'ingest (decode+preprocess)', within: null },
  'ingest.decode': { label: 'image decode', within: 'ingest.run' },
  'ingest.readback': { label: 'pixel readback + grayscale', within: 'ingest.run' },

  // ── the deferred display tail ──
  'app.tail': { label: 'visual tail (deferred)', within: null },
  'app.project': { label: 'projectBins (display + decode-marginals bins)', within: 'app.tail' },
  'app.overlays': { label: 'poleMarkers+overlays', within: 'app.tail' },
  // Runs inside the tail's projection AND straight off a mode switch, with no
  // tail above it at all. Both are legitimate, which is exactly the case the
  // join's root-vs-orphan rule exists to keep quiet about.
  'project.bins': { label: 'projectBins', within: 'app.project' },
  'project.upload': { label: 'CPU→GPU upload phase (gray + uniforms)', within: 'project.bins' },
  'project.dispatch': { label: 'GPU dispatch (gradient + project)', within: 'project.bins' },
  // The dispatch span above closes at submit and this one opens after the
  // readback resolves, so the fence between them is inside NEITHER span. It is
  // the gap between two sibling rows, and reading it means subtracting their
  // ends and starts by hand -- which is what the deleted `waitMs` column used
  // to do for free, and the one thing genuinely lost with it.
  'project.finish': { label: 'CPU finish (unpack + min/max)', within: 'project.bins', sync: true },
} as const satisfies Readonly<Record<string, StageNode>>;

export type AppStageId = keyof typeof APP_STAGES;

// Every stage this app can record. Just the app's own now -- see the header for
// what used to be spread in here, and why it is not.
export const ALL_STAGES: StageTable = { ...APP_STAGES };

// The app's typed span opener: a mistyped id joins nowhere and reports as
// `unknown` rather than failing.
export function appSpan(id: AppStageId, attrs: SpanAttrs | null = null): Span {
  return spanStart(id, attrs);
}

export function joinAll(): JoinResult {
  return joinRecords(getRecords(), ALL_STAGES);
}

// The console entry point, reached by bare name through main.ts's
// import.meta.glob. Zero-argument on purpose: it is typed at a devtools prompt.
export function formatFlamechart(): string {
  return formatSpanTree(joinAll(), ALL_STAGES);
}
