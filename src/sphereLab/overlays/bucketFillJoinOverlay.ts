import { Camera } from '../camera/model.ts';
import { activeCamera } from '../camera/store.ts';
import { compositeLineLength, CompositeLineDisplay, computeCompositeLines, computeJoinWalk, computeMergeGroups, groupDisplayColors, paintJoinOverlay } from '../pipeline/bucketFillJoin.ts';
import { toggleBucketFillCompositeBtn, toggleBucketFillJoinBtn, toggleBucketFillMergeMarkersBtn } from '../ui/dom.ts';

// Depends on camera.lastBucketFillSegments already being populated by
// overlays/bucketFillOverlay.ts's updateBucketFillOverlay -- call this AFTER
// that one (see this function's callers).
export function updateBucketFillJoinOverlay(camera: Camera) {
  const settings = camera.settings;
  if (!settings.showBucketFillJoin) return;
  if (!camera.lastBucketFillSegments || !camera.lastBucketFillColors || !camera.lastBucketFillRegionId) return;
  const { joinBuffer, merges, blueMergePoints, orangeMergePoints, redMergePoints } = computeJoinWalk(
    camera.lastBucketFillSegments, camera.lastBucketFillRegionId, camera.rtSize.w, camera.rtSize.h,
    settings.bucketFillMergeMinSimilarity, settings.bucketFillJoinSteps, settings.bucketFillMinLengthPx,
  );
  camera.lastBucketFillBlueMerges = blueMergePoints;
  camera.lastBucketFillOrangeMerges = orangeMergePoints;
  camera.lastBucketFillRedMerges = redMergePoints;
  // Colored by MERGE GROUP, not raw per-segment color -- unlike the base
  // fill (which stays per-segment so individual blobs stay distinguishable),
  // the whole point of this layer is showing which segments have been
  // judged the same line, so segments sharing a group show the same color
  // here even though they're different colors in the base overlay.
  const groupOf = computeMergeGroups(camera.lastBucketFillSegments.length, merges);
  const displayColors = groupDisplayColors(groupOf, camera.lastBucketFillColors);
  paintJoinOverlay(joinBuffer, displayColors, camera.bucketFillJoinData);
  camera.bucketFillJoinTex.needsUpdate = true;
  camera.lastBucketFillMerges = merges;

  if (settings.showBucketFillComposite) {
    const compositeByRoot = computeCompositeLines(camera.lastBucketFillSegments, groupOf);
    const composites: CompositeLineDisplay[] = [];
    for (const [root, line] of compositeByRoot) {
      // A segment too short to ever get join-walk fronts (see the
      // computeJoinWalk length filter above) can never appear in `merges`,
      // so it stays its own singleton group forever -- computeCompositeLines
      // still emits a "composite" for it (just its own raw endpoints), the
      // same tiny stray line the min-length filter is supposed to make
      // disappear everywhere else (base raster, markers, join-walk fronts,
      // votes.ts's own vote generation). Filtered out here the same way.
      if (compositeLineLength(line) < settings.bucketFillMinLengthPx) continue;
      composites.push({ ...line, color: displayColors[root] });
    }
    camera.lastBucketFillComposite = composites;
  } else {
    camera.lastBucketFillComposite = null;
  }
}

// Availability tracks the PARENT bucket-fill toggle, not fieldView directly
// -- there's nothing to join without segments to join in the first place.
export function updateBucketFillJoinAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.showBucketFillSegments;
  toggleBucketFillJoinBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showBucketFillJoin = false;
    toggleBucketFillJoinBtn.classList.remove('active');
    cam.bucketFillJoinData.fill(0);
    cam.bucketFillJoinTex.needsUpdate = true;
    cam.lastBucketFillMerges = null;
    cam.lastBucketFillComposite = null;
    cam.lastBucketFillBlueMerges = null;
    cam.lastBucketFillOrangeMerges = null;
    cam.lastBucketFillRedMerges = null;
  }
}

// Availability tracks the join toggle (its own parent), one level further
// down the chain -- there's nothing to compose without a join walk to
// derive merge groups from.
export function updateBucketFillCompositeAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.showBucketFillJoin;
  toggleBucketFillCompositeBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showBucketFillComposite = false;
    toggleBucketFillCompositeBtn.classList.remove('active');
    cam.lastBucketFillComposite = null;
  }
}

// Availability tracks the join toggle (its own parent), same level as
// composite -- the merge-direction X markers are computed unconditionally as
// part of the join walk itself (they're cheap, unlike composite lines), so
// there's nothing to additionally gate on except "did a join walk run".
export function updateBucketFillMergeMarkersAvailability() {
  const cam = activeCamera(); if (!cam) return;
  const relevant = cam.settings.showBucketFillJoin;
  toggleBucketFillMergeMarkersBtn.disabled = !relevant;
  if (!relevant) {
    cam.settings.showBucketFillMergeMarkers = false;
    toggleBucketFillMergeMarkersBtn.classList.remove('active');
  }
}
