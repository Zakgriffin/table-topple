import * as THREE from 'three';
import { jacobiEigenSymmetric, smallestEigenvector } from '../../../linalg.ts';
import { cornerDir } from '../../../sphereLab/math/geometry.ts';
import { FieldResidency } from '../../gpu/fieldResidency.ts';
import { spanEnd, spanStart } from '../../../sphereLab/profiling/profiler.ts';
import { type CompositeLine, type Vote } from '../../results.ts';
import { type Backend } from '../../backend.ts';
import { computeLsdRectangles, runLsdChain } from '../lsd/chain.ts';
import type { LsdRectangle } from '../lsd/types.ts';
// Just the LSD tuning knobs computeGradient2x2Composites/
// compositesFromLsdRectangles actually read off `settings` -- narrowed off
// the full CameraSettingsCommon (which also carries dozens of display-only
// toggles a real Camera has but a bare phone-side PoseComputeState never
// will) so both functions stay callable with either a real camera's
// settings OR pose/poseCompute.ts's PoseComputeState.settings, which
// only carries this subset. A real CameraSettingsCommon still satisfies
// this structurally (extra fields are fine), so every existing desktop call
// site is unaffected.
export interface LsdCompositeSettings {
  lsdToleranceDeg: number; lsdRhoNoiseThreshold: number; lsdRhoHighThreshold: number; lsdCclSteps: number;
  lsdMinRegionSize: number;
  lsdNfaEpsilon: number;
  lsdNfaTestExponent: number;
  lsdMinLengthPx: number;
}

// One LINE per accepted LSD rectangle, tagged with its root -- the shared
// first stage behind computeSegmentVotes below AND pose/stages/period/gridPeriodPhase.ts.
// Runs the full traditional LSD pipeline (region growing -> rectangle fit ->
// NFA validation, see pose/stages/lsd/chain.ts's own header) over the gray `res`
// was created around, then turns each accepted rectangle into a line.
// It used to run a JOIN WALK between those two steps, merging collinear
// rectangles into composites; that is retired, see compositesFromLsdRectangles.
// Deliberately NOT the radius-driven gradient field x local-agreement
// "effective" field this used to run through -- see this session's chat for
// why. That whole construction is now DELETED rather than merely bypassed:
// nothing computes an agreement or effective field anywhere in the project.
// Pulled out as its own function (and called exactly ONCE per
// reconstruction pass, see pipeline/axesReconstruction.ts) so every
// downstream consumer -- vote casting here, row/col family classification
// in gridPeriodPhase.ts, and the "color composite lines by row/col family"
// debug overlay -- sees the exact same lines under the exact same root
// numbers. It used to take the gradient field pre-computed, so that the caller
// could pick CPU or GPU without this function -- or any of its downstream
// consumers -- needing to know or care which one produced it. The residency
// now carries that same ignorance one stage earlier and better: runLsdChain
// computes the gradient too, on whichever side the toggle says, and holds
// fx/fy there. Nothing between here and the fitter has to name a side, and the
// gradient no longer has to land on the CPU just to be passed along.
// Returns the RECTANGLES alongside the composite lines. Not because this
// function needs them -- it filters them down to lines and is done -- but
// because they are stage 4's output and a caller can ask for them (see
// pose/intermediates.ts). They were dropped on the floor here before, which
// is one of the reasons overlays/lsdOverlay.ts ran a second complete LSD chain:
// the first one's answer existed and then stopped existing.
export async function computeGradient2x2Composites(
  settings: LsdCompositeSettings,
  res: FieldResidency, w: number, h: number, backend: Backend, wantMembers = false,
): Promise<{ composites: { root: number; line: CompositeLine }[]; rects: LsdRectangle[] }> {
  const rects = await runLsdChain(res, w, h, {
    toleranceDeg: settings.lsdToleranceDeg,
    rhoNoiseThreshold: settings.lsdRhoNoiseThreshold,
    rhoHighThreshold: settings.lsdRhoHighThreshold,
    cclSteps: settings.lsdCclSteps,
    minRegionSize: settings.lsdMinRegionSize,
    nfaEpsilon: settings.lsdNfaEpsilon,
    nfaTestExponent: settings.lsdNfaTestExponent,
  }, backend, wantMembers);
  // Its own span inside this function's: it walks all ~5200 rectangles and keeps
  // ~893, and until it was split out that walk was indistinguishable from the
  // chain that produced them. Contains no await, so its duration is host CPU.
  const filterSpan = spanStart('compositesFromLsdRectangles', true);
  const out = compositesFromLsdRectangles(rects, w, h, settings);
  spanEnd(filterSpan);
  return { composites: out, rects };
}

// One line per ACCEPTED LSD rectangle -- its own two endpoints, no merging.
// Split out from computeGradient2x2Composites so overlays/lsdOverlay.ts's live
// debug preview (which already has its own freshly-computed LsdRectangle[] for
// the rectangle/rejected/raw-region views) can produce lines from that SAME
// array instead of re-running the chain a second time just to get here.
//
// ── The join walk was removed here, and it was a NO-OP when it went ──
//
// This used to merge rectangles into composite lines and emit one line per
// merge group. It ran at joinSteps 0, where the step loop never executes: every
// segment was already its own root and the length filter compared the same
// endpoint pair `r.length` compares below. So every line, root number and
// filter decision is unchanged by its removal.
//
// The consequence worth remembering is what that means for the OPEN question:
// "does voting per raw segment rather than per merged composite hurt the fit?"
// was never actually being tested, because at joinSteps 0 there was no merged
// composite to lose. See levelLine.ts for the one place
// this genuinely bites -- a polarity flip mid-line yields two antiparallel
// halves that nothing reassembles.
export function compositesFromLsdRectangles(
  rects: ReturnType<typeof computeLsdRectangles>, w: number, h: number, settings: LsdCompositeSettings,
): { root: number; line: CompositeLine }[] {
  const result: { root: number; line: CompositeLine }[] = [];
  let root = 0;
  for (const r of rects) {
    if (!r.accepted) continue;
    const id = root++; // consumed even when the length filter drops this line, to keep numbering stable
    if (r.length < settings.lsdMinLengthPx) continue;
    const ax = Math.cos(r.theta), ay = Math.sin(r.theta);
    const hl = r.length / 2;
    result.push({
      root: id,
      line: { x1: r.cx + hl * ax, y1: r.cy + hl * ay, x2: r.cx - hl * ax, y2: r.cy - hl * ay },
    });
  }
  return result;
}

// One vote per composite LINE (computeGradient2x2Composites above) -- casts
// a ray to each GROUP's composite endpoints, not each raw segment's own
// endpoints, so a chain of several short segments joined into one long line
// votes as ONE long, well-conditioned ray instead of several separate short,
// noisier ones.
export function computeSegmentVotes(
  composites: { root: number; line: CompositeLine }[],
  w: number, h: number,
  quat: THREE.Quaternion, vFovRad: number, aspect: number,
): Vote[] {
  const toNDC = (px: number, py: number): [number, number] => [(px / w) * 2 - 1, 1 - (py / h) * 2];
  const votes: Vote[] = [];
  for (const { line } of composites) {
    const [u1, v1] = toNDC(line.x1, line.y1);
    const [u2, v2] = toNDC(line.x2, line.y2);
    const ray1 = cornerDir(u1, v1, quat, vFovRad, aspect);
    const ray2 = cornerDir(u2, v2, quat, vFovRad, aspect);
    const n = ray1.clone().cross(ray2);
    const arcLen = n.length(); // sin(angle between the two endpoint rays) -- the composite line's own PROJECTED ARC LENGTH on the unit sphere
    if (arcLen < 1e-12) continue;
    n.divideScalar(arcLen); // normalize using the length already computed, instead of a second hypot
    // Weighted by that same projected arc length, not pixel count or average
    // magnitude -- a short/close-together line gives a small, easily-
    // corrupted cross product; a long one gives a large, well-conditioned
    // one, which is exactly the property we want a fit-confidence weight to
    // track -- an explicit per-composite-line quantity rather than something
    // buried in a 1-pixel step, which is how the retired per-pixel vote
    // sources relied on it implicitly.
    votes.push({ n, weight: arcLen });
  }
  return votes;
}


// Fits the degenerate quadric ("pair of planes through the origin") that best
// explains "every vote lies on one plane or the other" -- see pre-Stage-A
// history for the full derivation.
//
// Each vote enters the scatter matrix at its own weight over the frame's max
// weight. There used to be a `weightSharpenPower` exponent on that ratio, to
// let strong votes dominate weak ones superlinearly; it shipped at 1 (i.e. no
// sharpening) and is deleted, so this is exactly what it always computed.
// Note the normalization itself is a no-op for the FIT -- scaling every weight
// by a constant scales ATA by that constant and leaves its eigenvectors
// untouched -- it is kept because the GPU counterpart accumulates in f32 and
// wants the summands in [0,1].
//
// The accumulation and the eigen-solve used to be two named helpers, split so
// an IRLS refinement pass could reuse them with a residual-based reweight in
// place of the magnitude sharpen below. That pass is deleted, this is the only
// caller either half ever had, and the split's own comments described a
// sharing that no longer existed -- so they are one function again.
export function fitPairOfPlanes(votes: Vote[]): { Drow: THREE.Vector3; Dcol: THREE.Vector3; Dnormal: THREE.Vector3 } | null {
  let maxW = 0;
  for (const { weight } of votes) if (weight > maxW) maxW = weight;

  // Weighted 6x6 scatter matrix over the votes -- the only vote-count-
  // dependent part; everything below it is fixed-size.
  const ATA: number[][] = Array.from({ length: 6 }, () => new Array(6).fill(0));
  for (const v of votes) {
    const w = maxW > 0 ? v.weight / maxW : 0;
    if (w <= 0) continue;
    const { n } = v;
    const row = [n.x * n.x, n.y * n.y, n.z * n.z, n.x * n.y, n.x * n.z, n.y * n.z];
    for (let a = 0; a < 6; a++) {
      const wra = w * row[a];
      for (let b = 0; b < 6; b++) ATA[a][b] += wra * row[b];
    }
  }

  const m = smallestEigenvector(ATA);
  const M = [
    [m[0], m[3] / 2, m[4] / 2],
    [m[3] / 2, m[1], m[5] / 2],
    [m[4] / 2, m[5] / 2, m[2]],
  ];
  const { values, vectors } = jacobiEigenSymmetric(M);
  let zeroIdx = 0;
  for (let i = 1; i < 3; i++) if (Math.abs(values[i]) < Math.abs(values[zeroIdx])) zeroIdx = i;
  const others = [0, 1, 2].filter((i) => i !== zeroIdx);
  const b1 = new THREE.Vector3(vectors[others[0]][0], vectors[others[0]][1], vectors[others[0]][2]);
  const b2 = new THREE.Vector3(vectors[others[1]][0], vectors[others[1]][1], vectors[others[1]][2]);
  const Dnormal = new THREE.Vector3(vectors[zeroIdx][0], vectors[zeroIdx][1], vectors[zeroIdx][2]).normalize();
  const Drow = b1.clone().add(b2);
  const Dcol = b1.clone().sub(b2);
  if (Drow.lengthSq() < 1e-9 || Dcol.lengthSq() < 1e-9) return null;
  return { Drow: Drow.normalize(), Dcol: Dcol.normalize(), Dnormal };
}
