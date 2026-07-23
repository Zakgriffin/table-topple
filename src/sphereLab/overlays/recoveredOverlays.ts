import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { MATH_QUAT } from '../constants.ts';
import { cornerDir } from '../math/geometry.ts';
import { getAnalysisVFovRad } from '../pipeline/capture.ts';
import { globalState } from '../state.ts';

// Rebuilds the recovered-floor overlay's geometry/position/orientation --
// called once per fresh decode, not per frame.
export function applyRecoveredFloorOverlay(camera: Camera) {
  if (!camera.lastPositionDecode || !camera.lastRecoveredAxes || !camera.lastProjectedBins) return;
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

