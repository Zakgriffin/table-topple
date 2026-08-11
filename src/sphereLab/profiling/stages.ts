import { POSE_STAGES } from '../../pose/timing/stages.ts';
import {
  type JoinResult, type SpanAttrs, type StageNode, type StageRecord, type StageTable,
  formatSpanTree, getRecords, joinRecords, spanStart,
} from './profiler.ts';

// ── Sphere Lab's own stages, and how it COMPOSES the library's ───────────
//
// Two things live here, and the second is the interesting one.
//
// First, the app's own instrumented stages: the capture path, the deferred
// display tail, and the projection that only exists to feed display.
//
// Second, the two OVERRIDES at the bottom. pose/timing/stages.ts declares
// `pose.run` and `pose.drain` as roots, because within the library they are --
// `computePoseFromCapture` is the top of its own world and the drain happens
// after it returns. In THIS app they are not roots: a pose runs inside
// `app.reconstruct` and a drain inside `app.tail`. That knowledge belongs to
// the consumer, so the consumer states it, and the library never has to name an
// app stage. It is the same seam `applyPoseResult` is for the pose result
// itself -- the library hands back facts, the app says where they go.
//
// A different consumer composes differently and says so in its own table; the
// phone (mobileCapture.ts) runs `pose.run` with nothing above it, and the
// headless harness runs it with nothing above it either. Both get a root,
// because the join treats "declared parent produced no records" as a root
// rather than an anomaly. See joinRecords.
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
  // A real stage, not a gap: it is opened when a physical camera finishes and
  // closed when the next frame arrives, so its duration is the shutter-to-
  // shutter idle the auto-capture interval is spending.
  'app.idle': { label: 'idle (waiting for next frame)', within: null },

  'ingest.run': { label: 'ingest (decode+preprocess)', within: null },
  'ingest.decode': { label: 'image decode', within: 'ingest.run' },
  'ingest.readback': { label: 'pixel readback + grayscale', within: 'ingest.run', inputs: ['ingest.decode'] },

  // ── the deferred display tail ──
  'app.tail': { label: 'visual tail (deferred)', within: null },
  'app.project': { label: 'projectBins (display + decode-marginals bins)', within: 'app.tail', inputs: ['pose.drain'] },
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

// Every stage this app can record, library included, with the composition
// overrides applied. Spread order matters: the overrides must come last.
//
// The overrides add an `inputs` edge as well as a `within`, for the same
// reason: the library cannot know that a pose consumes an app capture, and the
// app cannot state it anywhere else. On a `recompute` there is no `app.capture`
// record at all and the edge simply does not resolve, which the walk treats as
// "this stage is a head" rather than as an anomaly -- see joinRecords and
// CriticalPath.unsatisfied.
export const ALL_STAGES: StageTable = {
  ...POSE_STAGES,
  ...APP_STAGES,
  'pose.run': { ...POSE_STAGES['pose.run'], within: 'app.reconstruct', inputs: ['app.capture'] },
  'pose.drain': { ...POSE_STAGES['pose.drain'], within: 'app.tail' },
};

// The app's typed span opener, mirroring pose/timing/stages.ts's poseSpan and
// there for the same reason: a mistyped id joins nowhere and reports as
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
