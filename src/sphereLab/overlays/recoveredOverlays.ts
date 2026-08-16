import * as THREE from 'three';
import { type Camera } from '../camera/model.ts';
import { MATH_QUAT } from '../constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../math/geometry.ts';
import { globalState } from '../state.ts';

// The World-view recovered-floor quad's OUTLINE ONLY -- a 4-point closed
// THREE.LineLoop, positioned from pose+FOV alone (projectImageCornersToPlane,
// already narrowed off a bare data shape in Step 1b -- needs no pixels at
// all). Guards on the pose's recoveredAxes/positionDecode only, both present in
// EITHER compute mode (see this session's on-device-pose-recovery plan), so
// this is what actually satisfies "always draw the outline" -- unlike
// applyRecoveredFloorOverlay below (the projected-image FILL), which stays
// gated on lastProjectedBins (real pixel data, only ever populated by a
// desktop-compute capture).
//
// Reuses the exact same Drow/Dcol/normal-in-world-space + camPos math
// applyRecoveredFloorOverlay does, just applied to each of the 4 corner
// (u,v) points individually (a general quad, not assumed to reduce to a
// center + orientation the way an axis-aligned PlaneGeometry could) instead
// of collapsing to a single center/size/orientation.
// ── THE OUTLINE IS DARK, pending pose plumbing ──
//
// Its four corners came from `projectImageCornersToPlane`, which died with
// the old pipeline's decode stage: image corners cast through the pose onto the floor
// plane. That is a pure projection and it is genuinely re-derivable from what is
// already on this camera (pose, FOV, aspect) -- but re-deriving it is part of
// swapping the app onto pose, not part of deleting the old pipeline, and doing
// it here would be guessing at the new seam before it exists.
//
// The overlay object itself stays on the camera, so the mode/visibility wiring
// in main.ts's animate loop is untouched and this comes back by filling the
// geometry rather than by rebuilding the scene.
export function updateRecoveredFloorOutline(camera: Camera) {
  camera.recoveredFloorOutline.visible = false;
}

// Rebuilds the recovered-floor overlay's geometry/position/orientation --
// called once per fresh decode, not per frame.
export function applyRecoveredFloorOverlay(camera: Camera) {
  const axes = camera.pose?.recoveredAxes;
  const decode = camera.pose?.positionDecode;
  if (!decode || !axes || !camera.lastProjectedBins) {
    // No real pixel data to fill the quad with -- e.g. device-compute mode,
    // where the phone never sends an image at all (see this session's
    // on-device-pose-recovery plan). Explicit, rather than just returning
    // and leaving whatever a PRIOR desktop-compute capture on this same
    // camera last painted here visible.
    camera.recoveredFloorOverlay.visible = false;
    return;
  }
  const { Drow: DrowMath, Dcol: DcolMath, Dnormal, distance } = axes;
  const normalMath = Dnormal.clone();
  const vFovRad = getAnalysisVFovRad(camera);
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalMath) > 0) normalMath.negate();
  const { recoveredCamQuat } = decode;
  const Drow = DrowMath.clone().applyQuaternion(recoveredCamQuat);
  const Dcol = DcolMath.clone().applyQuaternion(recoveredCamQuat);
  const normal = normalMath.clone().applyQuaternion(recoveredCamQuat);
  const { minU, maxU, minV, maxV } = camera.lastProjectedBins;
  const width = maxU - minU, height = maxV - minV;
  if (!(width > 0) || !(height > 0)) return;

  camera.recoveredFloorOverlay.visible = true;
  camera.recoveredFloorOverlay.geometry.dispose();
  camera.recoveredFloorOverlay.geometry = new THREE.PlaneGeometry(width, height);
  camera.recoveredFloorOverlayMat.opacity = camera.settings.recoveredFloorOpacity;

  const centerU = (minU + maxU) / 2, centerV = (minV + maxV) / 2;
  camera.recoveredFloorOverlay.position.copy(decode.camPos)
    .addScaledVector(Drow, centerU)
    .addScaledVector(Dcol, centerV)
    .addScaledVector(normal, -distance);
  // Sits exactly on the true floor plane (y=0) when the decode is accurate,
  // which z-fights with floorMesh itself -- nudge up along world +Y (the
  // true floor's own up axis, not the recovered `normal`, so this stays a
  // fixed visual offset regardless of any residual orientation error).
  camera.recoveredFloorOverlay.position.y += 0.02;

  const drowDisplay = Drow.clone().negate();
  const zAxis = new THREE.Vector3().crossVectors(drowDisplay, Dcol).normalize();
  const basis = new THREE.Matrix4().makeBasis(drowDisplay, Dcol, zAxis);
  camera.recoveredFloorOverlay.quaternion.setFromRotationMatrix(basis);
}

// Same shape/size as the ground-truth gizmoBody, in green, at the DECODED
// position AND orientation from runPositionDecode.
export function updateRecoveredCamGizmo(camera: Camera) {
  const decode = camera.pose?.positionDecode;
  if (decode) {
    camera.recoveredCamGizmo.position.copy(decode.camPos);
    camera.recoveredCamGizmo.quaternion.copy(decode.recoveredCamQuat);
  }
  camera.recoveredCamGizmo.visible = globalState.mode === 'world' && camera.settings.showGizmoBody && !!decode;
}

