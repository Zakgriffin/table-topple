import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { computeAdjacencyMerges } from '../pipeline/bucketFillAdjacencyMerge.ts';
import { compositeLineLength, computeCompositeLines, computeMergeGroups, groupDisplayColors } from '../pipeline/bucketFillJoin.ts';
import { computeBucketFillRegions, paintBucketFillOverlay, segmentColors } from '../pipeline/bucketFillSegments.ts';
import { computeGradient2x2Field } from '../pipeline/gradientField.ts';
import { computeLabelPropagationRegions } from '../pipeline/labelPropagationSegments.ts';
import { FieldView } from '../types.ts';
import { toggleBucketFillBtn } from '../ui/dom.ts';

// Both the raw color-by-direction view and its local-maxima-only derivative
// share the same underlying computeGradient2x2Field, so bucket fill (and,
// via label propagation, the local maxima themselves as seeds) is equally
// meaningful under either one -- see updateBucketFillAvailability below.
function bucketFillRelevant(fieldView: FieldView): boolean {
  return fieldView === 'gradient2x2' || fieldView === 'gradient2x2LocalMax';
}

export function updateBucketFillOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showBucketFillSegments) return;
  if (!bucketFillRelevant(settings.fieldView)) return;
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const lum = camera.lastNoisedPreviewGray;

  const field = computeGradient2x2Field(lum, w, h);
  const { regionId, segments } = settings.bucketFillUseLabelPropagation
    ? computeLabelPropagationRegions(field, settings.bucketFillToleranceDeg, settings.bucketFillMagnitudeThreshold, settings.bucketFillMaxSteps)
    : computeBucketFillRegions(field, settings.bucketFillToleranceDeg, settings.bucketFillMagnitudeThreshold, settings.bucketFillMaxSteps);
  // baseColors (keyed by seedIndex, see segmentColors) is what's stored back
  // onto the camera below -- downstream consumers (the join walk's own
  // group-blending in overlays/bucketFillJoinOverlay.ts) expect the RAW
  // per-segment color, not whatever this function's own post-merge display
  // recolors segments to, or a join-walk merge group would blend an
  // already-blended color instead of each member's true own color.
  const baseColors = segmentColors(segments);
  let displayColors: readonly [number, number, number][] = baseColors;
  let eligibleOverride: boolean[] | undefined;
  if (settings.bucketFillShowMerged) {
    const merges = computeAdjacencyMerges(regionId, segments, w, h, settings.bucketFillMergeMinSimilarity);
    const groupOf = computeMergeGroups(segments.length, merges);
    displayColors = groupDisplayColors(groupOf, baseColors);
    // Eligibility measured by each MERGE GROUP's combined composite length,
    // not any one raw segment's own -- a fragment too short to pass on its
    // own is exactly the case this merge pass exists to rescue.
    const compositeByRoot = computeCompositeLines(segments, groupOf);
    eligibleOverride = segments.map((_, i) => {
      const line = compositeByRoot.get(groupOf[i]);
      return !!line && compositeLineLength(line) >= settings.bucketFillMinLengthPx;
    });
  }
  paintBucketFillOverlay(regionId, segments, displayColors, settings.bucketFillMinLengthPx, camera.bucketFillData, eligibleOverride);
  camera.bucketFillTex.needsUpdate = true;
  camera.lastBucketFillSegments = segments;
  camera.lastBucketFillColors = baseColors;
  camera.lastBucketFillRegionId = regionId;
}

export function updateBucketFillAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = bucketFillRelevant(cam.settings.fieldView);
  toggleBucketFillBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showBucketFillSegments = false;
    toggleBucketFillBtn.classList.remove('active');
    cam.bucketFillData.fill(0);
    cam.bucketFillTex.needsUpdate = true;
    cam.lastBucketFillSegments = null;
    cam.lastBucketFillColors = null;
    cam.lastBucketFillRegionId = null;
  }
}
