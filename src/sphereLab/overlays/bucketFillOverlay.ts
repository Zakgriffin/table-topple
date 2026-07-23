import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { computeBucketFillRegions, paintBucketFillOverlay, segmentColors } from '../pipeline/bucketFillSegments.ts';
import { computeGradient2x2Field, computeGradientField } from '../pipeline/gradientField.ts';
import { computeTopGradientAlpha } from '../pipeline/gradientHighlight.ts';
import { FieldView } from '../types.ts';
import { toggleBucketFillBtn } from '../ui/dom.ts';

// Every FieldView that's an actual GradientField -- same set
// overlays/gradientHighlightOverlays.ts uses.
const VECTOR_FIELD_VIEWS: readonly FieldView[] = ['gradient', 'gradient2x2'];

export function updateBucketFillOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showBucketFillSegments) return;
  if (!VECTOR_FIELD_VIEWS.includes(settings.fieldView)) return;
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const lum = camera.lastNoisedPreviewGray;

  const field = settings.fieldView === 'gradient'
    ? computeGradientField(lum, w, h, Math.round(settings.simGradRadius))
    : computeGradient2x2Field(lum, w, h);

  // No percentile cutoff anymore (see this session's chat) -- every pixel
  // is eligible to FOUND a region (see computeBucketFillRegions's own
  // comment for why absorption was already unrestricted).
  const seedEligible = computeTopGradientAlpha(field, 0, 100);
  const { regionId, segments } = computeBucketFillRegions(field, settings.bucketFillToleranceDeg, seedEligible, settings.bucketFillMagnitudeThreshold, settings.bucketFillMaxSteps);
  const colors = segmentColors(segments.length);
  paintBucketFillOverlay(regionId, segments, colors, settings.bucketFillMinLengthPx, camera.bucketFillData);
  camera.bucketFillTex.needsUpdate = true;
  camera.lastBucketFillSegments = segments;
  camera.lastBucketFillColors = colors;
  camera.lastBucketFillRegionId = regionId;
}

export function updateBucketFillAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = VECTOR_FIELD_VIEWS.includes(cam.settings.fieldView);
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
