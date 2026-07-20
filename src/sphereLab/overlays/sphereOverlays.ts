import * as THREE from 'three';
import { Camera, SimulatedCamera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { COL_DIR, PATCH_RES, ROW_DIR, SPHERE_RADIUS, euler } from '../constants.ts';
import { colLineKs, cornerDir, greatCircleNormal, rowLineKs, slerpUnit, writeCirclePoints } from '../math/geometry.ts';
import { votesInMagnitudeBand } from '../pipeline/votes.ts';
import { readout } from '../ui/dom.ts';

export const DEBUG_CIRCLE_SEGMENTS = 48;
export const AXIS_VECTOR_LENGTH = 0.7;

export function updateGradientCirclesDebug(camera: Camera) {
  const chosen = votesInMagnitudeBand(camera.lastVotes, camera.settings.circleSamplePercentMin, camera.settings.circleSamplePercentMax);
  // vote.n lives in MATH_QUAT's fixed math frame (see PositionDecodeResult's
  // comment) -- rotate into true world space by the same anchorQuat
  // updateSphereOverlays uses (true camQuat for simulated cameras, so this
  // stays anchored to the *true* pose per the debug-visibility decision;
  // recoveredCamQuat for physical, which have no ground truth) before these
  // land as positions on sphereAnchor's children.
  const anchorQuat = isSimulated(camera) ? camera.camQuat : (camera.lastPositionDecode?.recoveredCamQuat ?? null);
  if (chosen.length === 0 || !anchorQuat) {
    camera.gradientCirclesGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    camera.axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    return;
  }

  let minW = Infinity, maxW = -Infinity;
  for (const vote of chosen) {
    if (vote.weight < minW) minW = vote.weight;
    if (vote.weight > maxW) maxW = vote.weight;
  }
  const wRange = maxW - minW;

  const positions = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 2 * 3);
  const colors = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 2 * 3);
  const axisPositions = new Float32Array(chosen.length * 2 * 3);
  const axisColors = new Float32Array(chosen.length * 2 * 3);
  let p = 0, pc = 0, ap = 0, apc = 0;
  const u = new THREE.Vector3(), v = new THREE.Vector3(), helper = new THREE.Vector3();
  for (const vote of chosen) {
    const normal = vote.n.clone().applyQuaternion(anchorQuat);
    const t = wRange > 0 ? (vote.weight - minW) / wRange : 0;
    const r = 1 - t, b = t;
    helper.set(0, 1, 0);
    if (Math.abs(normal.y) >= 0.9) helper.set(1, 0, 0);
    u.crossVectors(helper, normal).normalize();
    v.crossVectors(normal, u);
    for (let s = 0; s < DEBUG_CIRCLE_SEGMENTS; s++) {
      const a0 = (s / DEBUG_CIRCLE_SEGMENTS) * Math.PI * 2;
      const a1 = ((s + 1) / DEBUG_CIRCLE_SEGMENTS) * Math.PI * 2;
      const c0 = Math.cos(a0) * SPHERE_RADIUS, sn0 = Math.sin(a0) * SPHERE_RADIUS;
      const c1 = Math.cos(a1) * SPHERE_RADIUS, sn1 = Math.sin(a1) * SPHERE_RADIUS;
      positions[p++] = u.x * c0 + v.x * sn0; positions[p++] = u.y * c0 + v.y * sn0; positions[p++] = u.z * c0 + v.z * sn0;
      positions[p++] = u.x * c1 + v.x * sn1; positions[p++] = u.y * c1 + v.y * sn1; positions[p++] = u.z * c1 + v.z * sn1;
      colors[pc++] = r; colors[pc++] = 0; colors[pc++] = b;
      colors[pc++] = r; colors[pc++] = 0; colors[pc++] = b;
    }
    const len = maxW > 0 ? AXIS_VECTOR_LENGTH * Math.pow(vote.weight / maxW, camera.settings.weightSharpenPower) : 0;
    axisPositions[ap++] = 0; axisPositions[ap++] = 0; axisPositions[ap++] = 0;
    axisPositions[ap++] = normal.x * len;
    axisPositions[ap++] = normal.y * len;
    axisPositions[ap++] = normal.z * len;
    axisColors[apc++] = r; axisColors[apc++] = 0; axisColors[apc++] = b;
    axisColors[apc++] = r; axisColors[apc++] = 0; axisColors[apc++] = b;
  }
  camera.gradientCirclesGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  camera.gradientCirclesGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  camera.axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(axisPositions, 3));
  camera.axisVectorsGeo.setAttribute('color', new THREE.Float32BufferAttribute(axisColors, 3));
  camera.axisVectorsGeo.computeBoundingSphere();
  camera.gradientCirclesGeo.computeBoundingSphere();
}

export function updateGizmo(camera: SimulatedCamera): { hFovRad: number; vFovRad: number } {
  camera.camPos.set(camera.settings.camX, camera.settings.camY, camera.settings.camZ);
  euler.set(THREE.MathUtils.degToRad(camera.settings.camPitchDeg), THREE.MathUtils.degToRad(camera.settings.camYawDeg), 0);
  camera.camQuat.setFromEuler(euler);

  camera.gizmoCam.position.copy(camera.camPos);
  camera.gizmoCam.quaternion.copy(camera.camQuat);
  const hFovRad = THREE.MathUtils.degToRad(camera.settings.horizFovDeg);
  const vFovRad = 2 * Math.atan(Math.tan(hFovRad / 2) / camera.aspect);
  camera.gizmoCam.fov = THREE.MathUtils.radToDeg(vFovRad);
  camera.gizmoCam.aspect = camera.aspect;
  camera.gizmoCam.updateProjectionMatrix();

  camera.gizmoBody.position.copy(camera.camPos);
  camera.gizmoBody.quaternion.copy(camera.camQuat);
  camera.camHelper.update();

  if (camera === activeCamera()) {
    readout.innerHTML =
      `h-fov: ${THREE.MathUtils.radToDeg(hFovRad).toFixed(1)}&deg; &nbsp; v-fov: ${camera.gizmoCam.fov.toFixed(1)}&deg;<br>` +
      `pole separation: ${THREE.MathUtils.radToDeg(Math.acos(THREE.MathUtils.clamp(ROW_DIR.dot(COL_DIR), -1, 1))).toFixed(2)}&deg; (always 90&deg; — the orthogonal constraint)`;
  }

  return { hFovRad, vFovRad };
}

// Great-sphere overlays (poles/circles/frustum/patch/recovered markers) --
// repositioned (not rotated) to the camera's own origin each frame. A
// simulated camera anchors at its ground-truth camPos/camQuat, exactly as
// before. A physical camera has no ground truth, so it anchors at its own
// RECOVERED position/orientation once a decode exists (and shows nothing
// pose-dependent before that) -- a deliberate, plan-approved Stage A
// deviation from the pre-Stage-A app, which (having no per-camera-type
// concept at all) silently kept showing whatever the simulated sliders'
// last values happened to be even while real-capture mode was active. See
// this file's header comment / the Stage A report for the full rationale.
export function updateSphereOverlays(camera: Camera, vFovRad: number) {
  const settings = camera.settings;
  camera.circlesGroup.visible = settings.showCircles;
  camera.sphereShell.visible = settings.showSphere;

  // axesComputed only reflects orientation-fit success; the poles' actual
  // position is only ever written on a successful position decode (see
  // runPositionDecode's caller) -- gating on axesComputed alone leaves them
  // visible at a stale/default (0,0,0) whenever decode fails independently,
  // same failure updateRecoveredCamGizmo/applyRecoveredFloorOverlay already
  // guard against via lastPositionDecode.
  const recoveredPolesVisible = settings.showRecoveredPoles && camera.axesComputed && !!camera.lastPositionDecode;
  camera.recoveredRowPoleA.visible = recoveredPolesVisible;
  camera.recoveredRowPoleB.visible = recoveredPolesVisible;
  camera.recoveredColPoleA.visible = recoveredPolesVisible;
  camera.recoveredColPoleB.visible = recoveredPolesVisible;
  camera.axisVectorsLines.visible = settings.showAxisVectors;
  camera.gradientCirclesLines.visible = settings.showTopCircles;

  let anchorPos: THREE.Vector3;
  let anchorQuat: THREE.Quaternion | null;
  if (isSimulated(camera)) {
    anchorPos = camera.camPos;
    anchorQuat = camera.camQuat;
    camera.polesGroup.visible = settings.showPoles;
    if (settings.showPoles) {
      camera.rowPoleA.position.copy(ROW_DIR).multiplyScalar(SPHERE_RADIUS);
      camera.rowPoleB.position.copy(ROW_DIR).multiplyScalar(-SPHERE_RADIUS);
      camera.colPoleA.position.copy(COL_DIR).multiplyScalar(SPHERE_RADIUS);
      camera.colPoleB.position.copy(COL_DIR).multiplyScalar(-SPHERE_RADIUS);
    }
  } else {
    anchorPos = camera.lastPositionDecode?.camPos ?? new THREE.Vector3();
    anchorQuat = camera.lastPositionDecode?.recoveredCamQuat ?? null;
  }
  camera.sphereAnchor.position.copy(anchorPos);

  if (settings.showCircles) {
    const updateFamily = (ks: number[], pool: THREE.Line[], axis: 'row' | 'col', dir: THREE.Vector3) => {
      for (let i = 0; i < ks.length; i++) {
        const k = ks[i];
        const pointOnLine = axis === 'row' ? new THREE.Vector3(0, 0, k) : new THREE.Vector3(k, 0, 0);
        const n = greatCircleNormal(pointOnLine, dir, anchorPos);
        pool[i].visible = !!n;
        if (n) writeCirclePoints(pool[i], n, SPHERE_RADIUS);
      }
    };
    updateFamily(rowLineKs, camera.rowCirclePool, 'row', ROW_DIR);
    updateFamily(colLineKs, camera.colCirclePool, 'col', COL_DIR);
  }

  camera.frustumLine.visible = settings.showFrustum && !!anchorQuat;
  if (settings.showFrustum && anchorQuat) {
    const corners = [
      cornerDir(-1, -1, anchorQuat, vFovRad, camera.aspect),
      cornerDir(1, -1, anchorQuat, vFovRad, camera.aspect),
      cornerDir(1, 1, anchorQuat, vFovRad, camera.aspect),
      cornerDir(-1, 1, anchorQuat, vFovRad, camera.aspect),
    ];
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i < 4; i++) {
      const a = corners[i], b = corners[(i + 1) % 4];
      for (let t = 0; t < 16; t++) pts.push(slerpUnit(a, b, t / 16).multiplyScalar(SPHERE_RADIUS));
    }
    camera.frustumLine.geometry.dispose();
    camera.frustumLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
  }

  camera.patchMesh.visible = settings.showPatch && !!anchorQuat;
  if (settings.showPatch && anchorQuat) {
    const pos = camera.patchGeo.attributes.position as THREE.BufferAttribute;
    for (let j = 0; j <= PATCH_RES; j++) {
      const v = (j / PATCH_RES) * 2 - 1;
      for (let i = 0; i <= PATCH_RES; i++) {
        const u = (i / PATCH_RES) * 2 - 1;
        const d = cornerDir(u, v, anchorQuat, vFovRad, camera.aspect).multiplyScalar(SPHERE_RADIUS);
        const idx = j * (PATCH_RES + 1) + i;
        pos.setXYZ(idx, d.x, d.y, d.z);
      }
    }
    pos.needsUpdate = true;
    camera.patchGeo.computeVertexNormals();
  }
}

