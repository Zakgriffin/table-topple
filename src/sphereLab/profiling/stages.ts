import {
  type JoinResult, type SpanAttrs, type StageNode, type StageRecord, type StageTable,
  formatSpanTree, getRecords, joinRecords, spanStart,
} from './profiler.ts';

// ── Sphere Lab's own stages ──────────────────────────────────────────────
//
// The app's own instrumented stages: the capture path, the deferred display
// tail, and the projection that only exists to feed display.
//
// ── THE LIBRARY HALF OF THIS FILE IS GONE ──
//
// This used to spread in `POSE_STAGES` from `src/pose/timing/stages.ts` and then
// apply two COMPOSITION OVERRIDES: the library declared `pose.run` and
// `pose.drain` as roots because within the library they were, and the app
// restated them as living inside `app.reconstruct` and `app.tail`. That seam was
// the interesting part -- the library hands back facts, the consumer says where
// they go.
//
// src/pose is deleted, so both the table and the overrides went with it.
// `src/pose2` does not declare profiler stages at all: it is one submit and one
// readback, timed as a whole on the host and by GPU timestamps on the device, so
// there is no host-side span tree left for the app to compose with. If per-stage
// pose timing comes back, it comes back through that mechanism, and this file is
// not where it lands.
// ── `inputs` here are what make the CHAIN reach the pose ──
//
// The critical path is walked per level and recurses into the stages it
// selects, so it only descends into `pose.run` if something at the app level
// names it. Without these edges `formatFlamechart` shows a decomposition and no
// chain at all: `app.reconstruct` would have two children and no declared order
// between them, so the walk would stop at whichever finished last.
//
// All of them are strict sequencing that already exists in the code, not
// aspiration. `app.overlays <- app.project` is the one worth pointing at: its
// call site's own comment says applyPoseVisualizations MUST stay downstream of
// the projection, because the floor quad is sized from that frame's bins. That
// was a paragraph of prose keeping a hoist from happening; it is an edge now.
const APP_STAGES = {
  // ── the capture path ──
  'app.reconstruct': { label: 'axesReconstruction', within: null },
  'app.capture': { label: 'capture+preprocess', within: 'app.reconstruct' },
  // The whole pipeline as ONE span, which is all the host can see: pose2 is one
  // submit and one fence, so this is upload -> submit -> the map resolving, with
  // no interior the host could time. Per-stage numbers, if they come back, are
  // GPU timestamps and do not land here.
  'app.pose': { label: 'pose2 (submit + readback)', within: 'app.reconstruct', inputs: ['app.capture'] },
  // A real stage, not a gap: it is opened when a physical camera finishes and
  // closed when the next frame arrives, so its duration is the shutter-to-
  // shutter idle the auto-capture interval is spending.
  'app.idle': { label: 'idle (waiting for next frame)', within: null },

  'ingest.run': { label: 'ingest (decode+preprocess)', within: null },
  'ingest.decode': { label: 'image decode', within: 'ingest.run' },
  'ingest.readback': { label: 'pixel readback + grayscale', within: 'ingest.run', inputs: ['ingest.decode'] },

  // ── the deferred display tail ──
  'app.tail': { label: 'visual tail (deferred)', within: null },
  // NO `inputs` any more. It named `pose.drain`, which was the tail's readback of
  // the display intermediates -- deleted along with the drain itself, since pose2
  // hands them back with the pose. Nothing in the tail precedes the projection now.
  'app.project': { label: 'projectBins (display + decode-marginals bins)', within: 'app.tail' },
  'app.overlays': { label: 'poleMarkers+overlays', within: 'app.tail', inputs: ['app.project'] },
  // Runs inside the tail's projection AND straight off a mode switch, with no
  // tail above it at all. Both are legitimate, which is exactly the case the
  // join's root-vs-orphan rule exists to keep quiet about.
  'project.bins': { label: 'projectBins', within: 'app.project' },
  'project.upload': { label: 'CPU→GPU upload phase (gray + uniforms)', within: 'project.bins' },
  'project.dispatch': { label: 'GPU dispatch (gradient + project)', within: 'project.bins', inputs: ['project.upload'] },
  // Same shape as fit.eigen: the dispatch span closes at submit and this one
  // opens after the readback resolves, so the fence lands on this EDGE rather
  // than inside either span.
  'project.finish': { label: 'CPU finish (unpack + min/max)', within: 'project.bins', sync: true, inputs: ['project.dispatch'] },
} as const satisfies Readonly<Record<string, StageNode>>;

export type AppStageId = keyof typeof APP_STAGES;

// Every stage this app can record. Just the app's own now -- see the header for
// what used to be spread in here, and why it is not.
export const ALL_STAGES: StageTable = { ...APP_STAGES };

// The app's typed span opener: a mistyped id joins nowhere and reports as
// `unknown` rather than failing.
export function appSpan(id: AppStageId, attrs: SpanAttrs | null = null): StageRecord {
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
