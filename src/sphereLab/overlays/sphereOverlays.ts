import * as THREE from 'three';
import { Camera, SimulatedCamera } from '../camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { COL_DIR, PATCH_RES, ROW_DIR, SPHERE_RADIUS, euler } from '../constants.ts';
import { colLineKs, cornerDir, greatCircleNormal, rowLineKs, slerpUnit, writeCirclePoints } from '../math/geometry.ts';
import { readout } from '../ui/dom.ts';

export const DEBUG_CIRCLE_SEGMENTS = 48;
export const AXIS_VECTOR_LENGTH = 0.7;

export function updateGradientCirclesDebug(camera: Camera) {
  // Builds circle-segment geometry for every vote (no percentile cutoff
  // anymore, see this session's chat) -- skip entirely when neither toggle
  // that would actually show it is on. Callers that flip one of those
  // toggles ON call this directly to refresh (see ui/cameraPanel.ts), same
  // as changing the sharpen slider already does. NOTE: this can be
  // hundreds of thousands of votes on a real capture -- showTopCircles
  // defaults to off specifically because of this.
  if (!camera.settings.showTopCircles && !camera.settings.showAxisVectors) return;
  const chosen = camera.lastVotes;
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

  // Each circle segment becomes a flat quad (2 triangles, 6 verts) instead
  // of a single line -- extruded to +-halfWidth along `normal`, PERPENDICULAR
  // to the vote's own great-circle plane, not radially within it. Radial
  // in-plane extrusion (the first version of this) put both edges of the
  // ribbon on the exact same ray from the origin at every point around the
  // circle -- invisible (zero apparent width) from the Inside-Sphere view,
  // which sits AT the origin, no matter how large halfWidth was. Since
  // `normal` is perpendicular to every point's own radial direction (it's
  // perpendicular to both u and v, which span the circle's plane), extruding
  // along it produces genuine angular separation from ANY viewpoint,
  // including the origin. Cheap and reliable (see camera/model.ts's own
  // comment on gradientCirclesGeo for why this replaced a fat-line shader
  // approach) at the cost of the ring's apparent thickness scaling slightly
  // with zoom/distance, which is fine for a debug overlay.
  const halfWidth = camera.settings.topCirclesLineWidth * 0.006;
  const positions = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 6 * 3);
  const colors = new Float32Array(chosen.length * DEBUG_CIRCLE_SEGMENTS * 6 * 3);
  const axisPositions = new Float32Array(chosen.length * 2 * 3);
  const axisColors = new Float32Array(chosen.length * 2 * 3);
  let p = 0, pc = 0, ap = 0, apc = 0;
  const u = new THREE.Vector3(), v = new THREE.Vector3(), helper = new THREE.Vector3();
  const pushVert = (x: number, y: number, z: number, r: number, b: number) => {
    positions[p++] = x; positions[p++] = y; positions[p++] = z;
    colors[pc++] = r; colors[pc++] = 0; colors[pc++] = b;
  };
  for (const vote of chosen) {
    const normal = vote.n.clone().applyQuaternion(anchorQuat);
    const t = wRange > 0 ? (vote.weight - minW) / wRange : 0;
    const r = 1 - t, b = t;
    helper.set(0, 1, 0);
    if (Math.abs(normal.y) >= 0.9) helper.set(1, 0, 0);
    u.crossVectors(helper, normal).normalize();
    v.crossVectors(normal, u);
    const nx = normal.x * halfWidth, ny = normal.y * halfWidth, nz = normal.z * halfWidth;
    for (let s = 0; s < DEBUG_CIRCLE_SEGMENTS; s++) {
      const a0 = (s / DEBUG_CIRCLE_SEGMENTS) * Math.PI * 2;
      const a1 = ((s + 1) / DEBUG_CIRCLE_SEGMENTS) * Math.PI * 2;
      const dx0 = Math.cos(a0) * SPHERE_RADIUS, dy0 = Math.sin(a0) * SPHERE_RADIUS;
      const dx1 = Math.cos(a1) * SPHERE_RADIUS, dy1 = Math.sin(a1) * SPHERE_RADIUS;
      const b0x = u.x * dx0 + v.x * dy0, b0y = u.y * dx0 + v.y * dy0, b0z = u.z * dx0 + v.z * dy0;
      const b1x = u.x * dx1 + v.x * dy1, b1y = u.y * dx1 + v.y * dy1, b1z = u.z * dx1 + v.z * dy1;
      const i0x = b0x - nx, i0y = b0y - ny, i0z = b0z - nz;
      const o0x = b0x + nx, o0y = b0y + ny, o0z = b0z + nz;
      const i1x = b1x - nx, i1y = b1y - ny, i1z = b1z - nz;
      const o1x = b1x + nx, o1y = b1y + ny, o1z = b1z + nz;
      pushVert(i0x, i0y, i0z, r, b); pushVert(o0x, o0y, o0z, r, b); pushVert(o1x, o1y, o1z, r, b);
      pushVert(i0x, i0y, i0z, r, b); pushVert(o1x, o1y, o1z, r, b); pushVert(i1x, i1y, i1z, r, b);
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

