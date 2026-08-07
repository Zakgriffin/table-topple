import { FieldResidency } from '../../gpu/fieldResidency.ts';
import { computeGradient2x2FieldGPU } from '../gradient/gradient2x2.gpu.ts';
import { fitAndTestRegionsGPU } from './lsdFit.gpu.ts';
import { growRegionsCCLGPU } from './growRegions.gpu.ts';
import { spanEnd, spanStart } from '../../../sphereLab/profiling/profiler.ts';
import { type GradientField } from '../../../sphereLab/types.ts';
import { type Backend } from '../../backend.ts';
import { computeGradient2x2Field } from '../gradient/gradientField.ts';
import { growRegionsCCL } from './regions.cpu.ts';
import { fitRegionsCPU, nfaLog10ToLineScore, nfaLogTerms } from './rectangles.cpu.ts';
import { NO_MEMBERS, type LsdRectangle, type LsdSettings } from './types.ts';

// ── LSD (Line Segment Detector, von Gioi/Jakubowicz/Morel/Randall 2010) ───
// ── from scratch ────────────────────────────────────────────────────────
//
// A genuinely traditional reimplementation, and the PRODUCTION composite-line
// source: pose/stages/votes/votes.ts's computeGradient2x2Composites turns each accepted
// rectangle straight into one line.
//
// Pipeline: hysteresis-gated, directed-angle CONNECTED-COMPONENT region
// growing (stage 2+3, see growRegionsCCL below -- replaces the JFA-strided
// competitive relabeling that replaced the original magnitude-sorted serial
// BFS) -> magnitude-weighted PCA rectangle fit per region (stage 4) -> NFA
// statistical validation (stage 5).
//
// This file forms segments ONLY, and NOTHING bridges them afterwards.
// Bridging genuinely disjoint segments (across a gradient dropout, or across a
// level-line POLARITY flip -- see below) is therefore currently UNSOLVED rather
// than solved elsewhere; it was a no-op at the settings in use. The previous design
// tried to do both at once via long strided jumps, which is where all of its
// complexity and all of its failure modes lived: a jump of tens or hundreds
// of pixels only checks ANGLE agreement, so it could silently land on a
// completely different parallel ridge (an adjacent De Bruijn grid row) that
// merely shares a direction, and a bad merge like that compounds. Every pixel
// comparison here is now strictly between 8-NEIGHBORS, so that class of error
// is not merely guarded against, it is unreachable.
//
// GPU-friendliness note: stage 4+5's FIRST NFA pass HAS a GPU port
// (pose/stages/lsd/lsdFit.gpu.ts), and it is VERIFIED identical to this file's CPU
// path -- zero disagreements on n, k or accept/reject across a 2931-region
// capture, max nfaLog10 delta 7.7e-6 (see harness/lsdFitVerify.ts, which
// is how to re-check it after any change to either side). It defaults OFF
// anyway, for a PERFORMANCE reason: fitAndTestRegionsGPU still uploads
// mag/theta and the region CSR from CPU on every call, so it currently ADDS a
// round trip and measures slower (3.5ms CPU vs 8ms GPU). That inverts once
// stage 2+3 is GPU-resident and the labeling never lands on CPU.
// Stage 2+3 ALSO has a GPU port now (pose/stages/lsd/growRegions.gpu.ts, selected by the
// `backend` argument, alongside the fit stage in computeLsdRectanglesAuto). Like the JFA version it replaces, it was
// architecturally GPU-ready by construction -- each round is two
// frozen-buffer-in/fresh-buffer-out passes with no same-round cross-pixel
// dependency -- and it needed strictly LESS GPU machinery than that version
// would have: the per-round segmented reduction over region sums is gone
// entirely, leaving only the propagate step itself. It is the one stage that is
// NOT bit-identical to its CPU counterpart and cannot be made so; see its own
// header, and harness/growRegionsVerify.ts for how to measure the exposure
// on a given capture.
//
// Stage 5 had a tighten-then-shrink retry loop on NFA rejection. It is gone,
// code and settings both -- fitRegionOnce is the only fitter. It was never
// ported to GPU (retry 2+ needs a per-region partial sort, the hardest GPU
// problem in this file), and keeping it CPU-side meant every GPU-rejected
// region fell back to CPU and dragged mag/theta along with it, which was the
// last dependency preventing stages 1-4 from becoming one GPU-resident run. The
// CPU and GPU fitters now have identical scope by construction.
//
// Regions below settings.minRegionSize are dropped in
// collectRegionsFromLabels, before either fitter sees them. At the default
// floor of 2 that is free (a 1-member region has no axis to fit) and removes
// ~40% of the fit stage's input on a real capture.

// Stage 4 + stage 5 on GPU (pose/stages/lsd/lsdFit.gpu.ts). Null only if WebGPU itself
// is unavailable.
//
// REJECTED candidates are taken straight from the GPU's own output rather than
// re-fitted on CPU. That fallback existed only to run the retry loop on regions
// the first pass rejected; with the retry loop gone and the two paths
// verified to agree on n/k/accept, re-running a rejection on CPU would spend
// real time reproducing the identical numbers. The shader already returns full
// geometry for rejected regions, which is all the "show rejected candidates"
// overlay wants. Dropping it is also what frees stages 1-4 from needing
// the gradient field on CPU at all.
//
// ── THE MEMBERS READBACK IS NOW OPT-IN (2026-08-05) ──
//
// This used to `await res.regionsCPU()` alongside the fit, unconditionally, to
// fill in each rectangle's `rawMembers`: four readbacks (offsets/sizes/members/
// meanDirs) and the largest byte cost left in the chain. Now it happens only
// when `wantMembers` is set, which the pose path never sets.
//
// WHO ACTUALLY READS rawMembers, since a first audit got this wrong in a way
// worth recording. The tempting conclusion is that NOBODY does, because
// overlays/lsdOverlay.ts recomputes its own rectangles from
// lastNoisedPreviewGray rather than reading the camera field the pose path
// fills. That is true and it is not the point: its recomputation calls
// computeLsdRectanglesFromField, which lands in THIS function whenever
// the GPU fitter is in use -- which is the shipping setting. Deleting the readback
// outright would have blanked the rejected-region raster and thrown on
// `rawMembers[0]` in the accepted-rectangle hue. Same function, two callers,
// only one of which needs the data.
//
// So the split is by CALLER, not by field:
//   - pose path (votes.ts's computeGradient2x2Composites -> runLsdChain): the
//     rectangles feed compositesFromLsdRectangles, which reads only
//     accepted/length/theta/cx/cy, and are then dropped. Nothing stores them.
//     wantMembers stays false and the CSR never crosses.
//   - display (computeLsdRectanglesFromField, i.e. the LSD debug overlay and the
//     phone's sendDebugInfo pass): opts in, and pays for what it draws.
//   - harness/lsdChainVerify.ts wants the member TOTALS but not per
//     rectangle, so it leaves this false and asks the residency directly --
//     after taking the transfer ledger, so its own readback is not charged to
//     the configuration it is measuring.
async function fitRegionsGPU(
  res: FieldResidency, w: number, h: number, settings: LsdSettings, wantMembers: boolean,
): Promise<LsdRectangle[] | null> {
  const { logNTests, logEpsilon } = nfaLogTerms(w, h, settings);
  // Concurrent when both are wanted, not sequential: fitAndTestRegionsGPU binds
  // the region CSR synchronously before its first await, so starting it first
  // and letting the members readback run alongside puts that transfer under the
  // kernel instead of after it. Promise.all rather than two bare awaits so a
  // failure in either one can't leave the other rejecting unobserved.
  const [gpuResults, regions] = await Promise.all([
    fitAndTestRegionsGPU(res, w, h, settings.rhoNoiseThreshold, settings.toleranceDeg, logNTests, logEpsilon),
    wantMembers ? res.regionsCPU() : Promise.resolve(null),
  ]);
  if (!gpuResults) return null;

  // Counted off the fitter rather than off `regions`, which may not be here at
  // all now. fitAndTestRegionsGPU allocates `new Array(regionCount)` -- the same
  // count regionsCPU() returns -- so rectangle numbering downstream is
  // unchanged, and `regions[i]` still lines up index for index when present.
  // The SECOND object array of the same length in as many statements, and that
  // is the point of measuring it separately: fitAndTestRegionsGPU has just built
  // one LsdFitResult per region straight off the readback, and this immediately
  // re-wraps every one of them into an LsdRectangle. ~5200 regions is ~10400
  // short-lived objects per frame, of which compositesFromLsdRectangles keeps
  // 893. Whether that costs anything worth fixing is what this span and
  // 'lsdFit unpack (objects)' answer -- separately, so the two loops can be
  // judged apart rather than as one lump.
  const wrapSpan = spanStart('rectangle wrap (objects)', true);
  const results: LsdRectangle[] = [];
  for (let i = 0; i < gpuResults.length; i++) {
    const g = gpuResults[i];
    results.push({
      cx: g.cx, cy: g.cy, theta: g.theta, length: g.length, width: g.width,
      accepted: g.accepted, nfaLog10: g.nfaLog10,
      lineScore: nfaLog10ToLineScore(g.nfaLog10, logEpsilon),
      rawMembers: regions ? regions[i].members : NO_MEMBERS,
    });
  }
  spanEnd(wrapSpan);
  return results;
}

// Fully-CPU path, and the source of truth every GPU stage is verified against.
export function computeLsdRectangles(field: GradientField, settings: LsdSettings): LsdRectangle[] {
  const { w, h, fx, fy } = field;
  const { regions } = growRegionsCCL(
    fx, fy, w, h, settings.toleranceDeg, settings.rhoNoiseThreshold, settings.rhoHighThreshold, settings.cclSteps, settings.minRegionSize,
  );
  return fitRegionsCPU(regions, fx, fy, w, h, settings);
}


// Single dispatch point every caller uses -- centralizes the backend check once
// instead of duplicating it at each call site.
//
// Both stages follow the one `backend` argument, and either still falls back to
// CPU on
// its own without disturbing the other. That independence is the point -- the
// grower is the one stage whose GPU output is NOT bit-identical to its CPU
// output (see pose/stages/lsd/growRegions.gpu.ts's header), so it has to stay
// separately switchable from a stage that is.
//
// Takes the residency (pose/gpu/fieldResidency.ts) rather than a
// GradientField, and does NOT own it. That is deliberate: this function used to
// create one per call, which capped the chain at stage 3 and forced stage 1's
// output down to CPU and straight back up again no matter what either toggle
// said. The caller owning it is what lets the gradient join the chain -- see
// pose/poseCompute.ts, and computeLsdRectanglesFromField below for callers
// that genuinely do start from a CPU field.
//
// Stages transfer nothing themselves; they publish on the side they produced on
// and ask for the side they want, and the residency moves data only when a
// producer and a consumer are actually on opposite sides of the bus. That is
// what makes these toggles cost what the configuration implies rather than what
// each module hardcoded: with the whole chain on GPU, fx/fy are never uploaded
// at all and the labeling never crosses either.
//
// `wantMembers` decides whether the GPU fitter pays for LsdRectangle.rawMembers
// -- four readbacks of the region CSR, and the largest byte cost left in the
// chain. It defaults to FALSE because the pose path never looks at them; the two
// callers that render them (see computeLsdRectanglesFromField) opt in. The CPU
// fitter ignores the flag entirely: it is holding the regions already, so its
// members are free either way, and that is what keeps the two fitters
// comparable under the verify sweep.
async function computeLsdRectanglesAuto(
  res: FieldResidency, w: number, h: number, settings: LsdSettings, backend: Backend, wantMembers = false,
): Promise<LsdRectangle[]> {
  const growArgs = [
    w, h, settings.toleranceDeg, settings.rhoNoiseThreshold, settings.rhoHighThreshold, settings.cclSteps, settings.minRegionSize,
  ] as const;
  const growSpan = spanStart('grow+collect regions');
  const grown = backend === 'gpu' ? await growRegionsCCLGPU(res, ...growArgs, backend === 'gpu') : null;
  if (!grown) {
    // Either the toggle is off or the GPU grower bailed; on both paths it
    // published nothing, so the CPU grower owns these slots. Asking the
    // residency for the CPU side of fx/fy is what pulls stage 1's output back
    // down if -- and only if -- it was produced on the device.
    const cpu = growRegionsCCL(await res.cpuF64('fx'), await res.cpuF64('fy'), ...growArgs);
    res.provideCPU('regionId', cpu.regionId);
    res.provideRegionsCPU(cpu.regions);
  }
  // Closed AFTER the fallback, not after the GPU call, so the span means "grow
  // finished, whoever did it" -- otherwise a forceCPU run would report a grow of
  // ~0 and silently move the CPU grower's cost into its parent's self time,
  // which is the one number this instrumentation exists to read.
  //
  // Named grow+collect because collect lives INSIDE growRegionsCCLGPU (via its
  // collectOnGPU parameter), so there is no seam here to split them at. The two
  // are separable in gpuTimeline, which already reports grow:hook/grow:compress
  // and collect:finalize/regionMeta as distinct kernels -- this span is the host
  // time wrapped around all of them, and dividing it further would need a
  // boundary that does not exist in the call graph.
  spanEnd(growSpan);

  if (backend === 'gpu') {
    const gpu = await fitRegionsGPU(res, w, h, settings, wantMembers);
    if (gpu) return gpu;
  }
  return fitRegionsCPU(await res.regionsCPU(), await res.cpuF64('fx'), await res.cpuF64('fy'), w, h, settings);
}

// ── Stage 1: the 2x2 forward-difference gradient ──
//
// Publishes into the residency instead of returning a field, which is what
// makes stage 1 part of the chain rather than a thing that happens before it.
// Neither branch transfers anything: the GPU one leaves fx/fy on the device and
// the CPU one leaves them in JS, and whether either ever crosses is decided
// later, by whoever asks for the other side. Before this, the GPU branch
// uploaded gray and read fx/fy straight back, and then growRegionsCCLGPU
// uploaded those same two arrays again a moment later.
async function runGradient2x2Stage(res: FieldResidency, w: number, h: number, backend: Backend): Promise<void> {
  const useGPU = backend === 'gpu' && res.device !== null;
  const s = spanStart(useGPU ? 'gradient2x2 (GPU)' : 'gradient2x2 (CPU)');
  if (!(useGPU && await computeGradient2x2FieldGPU(res, w, h))) {
    // gray is CPU-resident by construction (createLsdChainResidency put it
    // there), so this is a lookup and never a readback.
    const field = computeGradient2x2Field(await res.cpuF64('gray'), w, h);
    res.provideCPU('fx', field.fx);
    res.provideCPU('fy', field.fy);
  }
  spanEnd(s);
}

// The chain's two entry points, always used as a pair. Split rather than fused
// because the caller has to be able to reach into the residency AFTER the
// rectangles come out, and because the caller owns the destroy. (The reader
// that motivated the split -- poseCompute's per-pixel "world votes" branch,
// which wanted fx/fy back on the CPU -- is deleted; the residency now outlives
// the rectangles for `gray`, so the fused decode can reuse it.) Anything that runs the chain should go through these two and nothing
// else, so that a residency-plumbing mistake is visible to the dev harness
// (harness/lsdChainVerify.ts) rather than only to production.
// `backend` decides whether a device is requested at all: an all-CPU chain never
// touches navigator.gpu. Stage 1 counts toward that, because it can want a device
// when nothing downstream does.
export async function createLsdChainResidency(
  gray: Float64Array, w: number, h: number, backend: Backend,
): Promise<FieldResidency> {
  const res = await FieldResidency.create(w * h, backend === 'gpu');
  res.provideCPU('gray', gray);
  return res;
}

// Stages 1 through 4: gray in, rectangles out, every intermediate left wherever
// its producer put it.
//
// No `wantMembers` parameter, deliberately: this is the POSE path's entry point
// (and the verify harness's), and neither renders per-rectangle membership.
// Leaving it off the signature means a future caller that does need members has
// to notice it is asking for four extra readbacks, rather than passing `true`
// through a chain of defaults without seeing the bill.
// `wantMembers` fills each rectangle's rawMembers by bringing the region CSR
// down. It is a HAND-BACK decision, not an algorithm one -- the members exist on
// whichever side the collect ran regardless, and nothing on the pose path reads
// them -- so passing it cannot change a pose. The production path leaves it
// false and pays nothing; a caller that asked for 'rects' (see
// pose/intermediates.ts) turns it on, because a rectangle with no members
// cannot be hued by its seed pixel or rasterized.
export async function runLsdChain(
  res: FieldResidency, w: number, h: number, settings: LsdSettings, backend: Backend,
  wantMembers = false,
): Promise<LsdRectangle[]> {
  await runGradient2x2Stage(res, w, h, backend);
  return await computeLsdRectanglesAuto(res, w, h, settings, backend, wantMembers);
}

// computeLsdRectanglesAuto for callers that already hold a CPU gradient field
// and have no interest in where anything lives: it wraps the field in a
// residency of its own for the duration of the call. The production pose path
// does NOT come through here -- it owns a residency spanning stage 1 as well,
// which is the whole point (see pose/poseCompute.ts) -- but the debug
// overlay and the phone both compute their gradient on CPU and would gain
// nothing from threading one through.
//
// Asks for rawMembers. ONE caller left: mobileCapture's sendDebugInfo pass.
// overlays/lsdOverlay.ts used to be the other, and that is the whole of what
// this function was for on the desktop -- a second complete LSD chain run
// purely so display could see rectangles the pose path had just computed and
// thrown away. It asks for them now (see pose/intermediates.ts), and
// runLsdChain takes wantMembers itself, so the desktop never comes through
// here. Still off the pose path either way.
export async function computeLsdRectanglesFromField(
  field: GradientField, settings: LsdSettings, backend: Backend,
): Promise<LsdRectangle[]> {
  const { w, h, fx, fy } = field;
  // Stage 1 has already happened on CPU here, so `backend` says nothing about
  // the gradient -- only about stages 2-4, which is why this creates its
  // residency from the same flag but publishes fx/fy on the CPU side regardless.
  const res = await FieldResidency.create(w * h, backend === 'gpu');
  try {
    res.provideCPU('fx', fx);
    res.provideCPU('fy', fy);
    return await computeLsdRectanglesAuto(res, w, h, settings, backend, true);
  } finally {
    res.destroy();
  }
}
