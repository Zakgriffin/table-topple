import * as THREE from 'three';
import { type Camera } from '../camera/model.ts';
import { MATH_QUAT } from '../constants.ts';
import { cornerDir, getAnalysisVFovRad } from '../math/geometry.ts';
import { projectImageCornersToPlane } from '../pipeline/decodeGrid.ts';
import { globalState } from '../state.ts';

// The World-view recovered-floor quad's OUTLINE ONLY -- a 4-point closed
// THREE.LineLoop, positioned from pose+FOV alone (projectImageCornersToPlane,
// already narrowed off a bare data shape in Step 1b -- needs no pixels at
// all). Guards on lastRecoveredAxes/lastPositionDecode only, both present in
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
export function updateRecoveredFloorOutline(camera: Camera) {
  if (!camera.lastRecoveredAxes || !camera.lastPositionDecode) {
    camera.recoveredFloorOutline.visible = false;
    return;
  }
  const corners = projectImageCornersToPlane(camera);
  if (!corners) {
    camera.recoveredFloorOutline.visible = false;
    return;
  }
  const { Drow: DrowMath, Dcol: DcolMath, Dnormal, distance } = camera.lastRecoveredAxes;
  const normalMath = Dnormal.clone();
  const vFovRad = getAnalysisVFovRad(camera);
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalMath) > 0) normalMath.negate();
  const { recoveredCamQuat, camPos } = camera.lastPositionDecode;
  const Drow = DrowMath.clone().applyQuaternion(recoveredCamQuat);
  const Dcol = DcolMath.clone().applyQuaternion(recoveredCamQuat);
  const normal = normalMath.clone().applyQuaternion(recoveredCamQuat);

  const pos = camera.recoveredFloorOutline.geometry.attributes.position as THREE.BufferAttribute;
  const p = new THREE.Vector3();
  corners.forEach((c, i) => {
    p.copy(camPos).addScaledVector(Drow, c.u).addScaledVector(Dcol, c.v).addScaledVector(normal, -distance);
    pos.setXYZ(i, p.x, p.y, p.z);
  });
  pos.needsUpdate = true;
  // Actual on-screen visibility (mode/showRecoveredFloor-gated) is set every
  // frame in main.ts's animate loop, same as recoveredFloorOverlay's own --
  // this just guarantees it's never left visible with stale geometry from a
  // camera that no longer has pose data at all.
  camera.recoveredFloorOutline.visible = true;
}

// Rebuilds the recovered-floor overlay's geometry/position/orientation --
// called once per fresh decode, not per frame.
export function applyRecoveredFloorOverlay(camera: Camera) {
  if (!camera.lastPositionDecode || !camera.lastRecoveredAxes || !camera.lastProjectedBins) {
    // No real pixel data to fill the quad with -- e.g. device-compute mode,
    // where the phone never sends an image at all (see this session's
    // on-device-pose-recovery plan). Explicit, rather than just returning
    // and leaving whatever a PRIOR desktop-compute capture on this same
    // camera last painted here visible.
    camera.recoveredFloorOverlay.visible = false;
    return;
  }
  const { Drow: DrowMath, Dcol: DcolMath, Dnormal, distance } = camera.lastRecoveredAxes;
  const normalMath = Dnormal.clone();
  const vFovRad = getAnalysisVFovRad(camera);
  if (cornerDir(0, 0, MATH_QUAT, vFovRad, camera.aspect).dot(normalMath) > 0) normalMath.negate();
  const { recoveredCamQuat } = camera.lastPositionDecode;
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
  camera.recoveredFloorOverlay.position.copy(camera.lastPositionDecode.camPos)
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
  if (camera.lastPositionDecode) {
    camera.recoveredCamGizmo.position.copy(camera.lastPositionDecode.camPos);
    camera.recoveredCamGizmo.quaternion.copy(camera.lastPositionDecode.recoveredCamQuat);
  }
  camera.recoveredCamGizmo.visible = globalState.mode === 'world' && camera.settings.showGizmoBody && !!camera.lastPositionDecode;
}

