import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { computeBucketFillRegions, paintBucketFillOverlay, randomSegmentColors } from '../pipeline/bucketFillSegments.ts';
import { computeEffectiveGradientField, computeGradient2x2Field, computeGradientAgreementField, computeGradientField } from '../pipeline/gradientField.ts';
import { computeTopGradientAlpha } from '../pipeline/gradientHighlight.ts';
import { computeWalkedGradientField } from '../pipeline/tangentWalk.ts';
import { FieldView } from '../types.ts';
import { toggleBucketFillBtn } from '../ui/dom.ts';

// Every FieldView that's an actual GradientField -- same set
// overlays/gradientHighlightOverlays.ts uses, for the same reason
// ("agreement" is scalar-only).
const VECTOR_FIELD_VIEWS: readonly FieldView[] = ['gradient', 'gradient2x2', 'effective', 'walked'];

export function updateBucketFillOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showBucketFillSegments) return;
  if (!VECTOR_FIELD_VIEWS.includes(settings.fieldView)) return;
  if (!camera.lastNoisedPreviewGray) return;
  const w = camera.rtSize.w, h = camera.rtSize.h;
  const lum = camera.lastNoisedPreviewGray;

  let field;
  if (settings.fieldView === 'gradient') {
    field = computeGradientField(lum, w, h, Math.round(settings.simGradRadius));
  } else if (settings.fieldView === 'gradient2x2') {
    field = computeGradient2x2Field(lum, w, h);
  } else {
    const rawField = computeGradientField(lum, w, h, Math.round(settings.simGradRadius));
    const agreement = computeGradientAgreementField(rawField, Math.round(settings.coherenceRadius));
    const effectiveField = computeEffectiveGradientField(rawField, agreement);
    field = settings.fieldView === 'effective' ? effectiveField : computeWalkedGradientField(settings, effectiveField);
  }

  // Same top-N% band the circles/top-gradient overlay already use -- only
  // pixels in that band are eligible to FOUND a region (see
  // computeBucketFillRegions's own comment for why absorption stays
  // unrestricted).
  const seedEligible = computeTopGradientAlpha(field, settings.circleSamplePercentMin, settings.circleSamplePercentMax);
  const { regionId, segments } = computeBucketFillRegions(field, settings.bucketFillToleranceDeg, seedEligible, settings.bucketFillMagnitudeThreshold);
  const colors = randomSegmentColors(segments.length);
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
