import * as THREE from 'three';
import { type Camera, type SimulatedCamera } from '../../shared/camera/model.ts';
import { activeCamera, isSimulated } from '../camera/store.ts';
import { COL_DIR, PATCH_RES, ROW_DIR, SPHERE_RADIUS, euler } from '../../shared/constants.ts';
import { colLineKs, cornerDir, greatCircleNormal, rowLineKs, slerpUnit, writeCirclePoints } from '../../shared/math/geometry.ts';
import { readout } from '../ui/dom.ts';
import { LINE_STYLE, cameraRay, drawableLines, lineColorer } from './lines.ts';
import { getAnalysisVFovRad } from '../pipeline/capture.ts';

const AXIS_VECTOR_LENGTH = 0.7;

// ── THE LINES, ON THE SPHERE ─────────────────────────────────────────────
//
// The third projection of overlays/lines.ts's one model. A line's two endpoints
// become two unit rays; the plane they span cuts the sphere in a great circle,
// and the segment BETWEEN them is the arc. `showLineRings` sweeps the same
// parameterisation all the way round instead of stopping at the second
// endpoint, so a ring is literally its arc continued -- same plane, same colour,
// one flag apart.
//
// ── THIS USED TO BE PER-VOTE, AND WEIGHT-COLOURED ────────────────────────
//
// It drew one FULL circle for every vote, shaded red->blue by vote weight: a
// quantity nothing else on screen used, on a set nothing else on screen drew,
// at a count its own comment called "hundreds of thousands on a real capture"
// -- which is why the toggle defaulted off. Now it draws exactly the lines the
// other two views draw, in the same colours, at ~300.
//
// `votes` is still read by the axis-vector overlay below, which genuinely IS
// per-vote and weight-scaled.
//
// ARC SUBDIVISION is per-radian rather than fixed, because these are segments
// of wildly different angular length -- a fixed count makes a short arc a dense
// blob and a long one a polygon.
const ARC_SEGMENTS_PER_RADIAN = 24;
const RING_SEGMENTS = 64;

// ── THE CROSS-SECTION IS A SQUARE, AND IT STRADDLES THE SPHERE ───────────
//
// An arc is a square TUBE swept along the great circle: at each step the four
// corners are `(R +- halfWidth) * radial +- halfWidth * normal`, so the section
// is halfWidth*2 on a side, centred exactly on SPHERE_RADIUS. Two things follow,
// and each fixes a defect a flat ribbon had:
//
// 1. IT POKES OUT BOTH SIDES. `patchMesh` -- the live viewport image patch -- has
//    its vertices at exactly `cornerDir(...) * SPHERE_RADIUS`, is OPAQUE and
//    writes depth. A flat ribbon AT that radius is coplanar with it, so depth
//    values agree to float precision and which surface survives varies per pixel
//    -- an arc clipped in patches by the rounded edge of the image. Pushing the
//    ribbon OUT to a larger radius fixed that from the outside and broke it from
//    the INSIDE, where the patch is then nearer than the arc and hides it
//    completely. Any radial offset is one-sided. A section that spans R-halfWidth
//    to R+halfWidth is visible from both, and needs no offset at all.
//
// 2. IT HAS WIDTH FROM EVERY DIRECTION. The flat ribbon was extruded along the
//    circle's plane normal, which is TANGENT to the sphere -- so viewed along
//    that normal it was edge-on and vanished, which is exactly the orbit angle
//    where you are looking at a great circle face-on. The tube's two SIDE faces
//    are what covers that view. (The original extrusion was still right about
//    the case it argued: from the origin, `normal` is perpendicular to the view
//    ray, so a flat ribbon never degenerates THERE. It just is not the only
//    viewpoint.)
//
// Costs 4 quads per step where the ribbon cost 1. At ~300 lines that is ~24k
// triangles for typical arcs and ~150k with both overlays on and rings swept --
// fine for a debug overlay, and still far below the per-vote circles this
// replaced.

export function updateLineArcs(camera: Camera) {
  const s = camera.settings;
  // vote.n and these rays share a frame: both are CAMERA-space (MATH_QUAT is
  // identity, see PositionDecodeResult), so the same anchorQuat updateSphereOverlays
  // uses puts them in world -- the TRUE camQuat for a simulated camera, so this
  // stays anchored to the true pose per the debug-visibility decision, and the
  // recovered one for a physical camera, which has no ground truth.
  const anchorQuat = isSimulated(camera) ? camera.camQuat : (camera.pose?.positionDecode?.recoveredCamQuat ?? null);
  const lines = anchorQuat ? drawableLines(camera) : [];
  const ray = cameraRay(Math.tan(getAnalysisVFovRad(camera) / 2), camera.aspect, camera.rtSize.w, camera.rtSize.h);
  const colorOf = lineColorer(camera);

  // ONE PASS PER KIND, into that kind's own mesh. The kinds differ in opacity,
  // and a material carries one opacity for everything drawn through it.
  for (const kind of ['segment', 'composite'] as const) {
    const geo = kind === 'segment' ? camera.segmentArcsGeo : camera.compositeArcsGeo;
    const of = lines.filter((l) => l.kind === kind);
    if (of.length === 0 || !anchorQuat) {
      geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
      continue;
    }

    // Two passes: the vertex count is not known until every arc's own angle is,
    // and growing a Float32Array per line is worse than measuring twice.
    type Arc = { u: THREE.Vector3; v: THREE.Vector3; n: THREE.Vector3; span: number; steps: number; rgb: [number, number, number] };
    const arcs: Arc[] = [];
    let quads = 0;
    for (const l of of) {
      const r1 = ray(l.line.x1, l.line.y1).applyQuaternion(anchorQuat);
      const r2 = ray(l.line.x2, l.line.y2).applyQuaternion(anchorQuat);
      const n = new THREE.Vector3().crossVectors(r1, r2);
      // A degenerate segment spans no plane -- both endpoints on one ray. There
      // is no circle to draw and normalizing would divide by zero.
      if (n.lengthSq() < 1e-18) continue;
      n.normalize();
      // `u` is the first endpoint and `v` completes the right-handed basis IN
      // the plane, so sweeping cos(t)u + sin(t)v from t=0 starts exactly at r1
      // and reaches r2 at t = the angle between them. That shared
      // parameterisation is what makes the ring an EXTENSION of the arc rather
      // than a second shape.
      const u = r1.clone();
      const v = new THREE.Vector3().crossVectors(n, u);
      const span = s.showLineRings ? Math.PI * 2 : Math.acos(THREE.MathUtils.clamp(r1.dot(r2), -1, 1));
      const steps = s.showLineRings ? RING_SEGMENTS : Math.max(2, Math.ceil(span * ARC_SEGMENTS_PER_RADIAN));
      arcs.push({ u, v, n, span, steps, rgb: colorOf(l.composite) });
      quads += steps;
    }

    // FOUR faces per step, 2 triangles each: a square tube, not a flat ribbon.
    const positions = new Float32Array(quads * 4 * 6 * 3);
    const colors = new Float32Array(quads * 4 * 6 * 3);
    let p = 0, pc = 0;
    // The kind's own stroke width is folded in here, so the two kinds stand in
    // the same RATIO on the sphere as they do on a rect -- the units differ
    // (world units against SVG pixels) but the relationship does not.
    // `lineRingWidth` scales both.
    //
    // ONE number, because the section is SQUARE: the radial and tangential
    // half-extents are equal by construction, so there is no second knob to keep
    // in step and no aspect that can drift.
    const halfWidth = s.lineRingWidth * LINE_STYLE[kind].width * 0.006;
    const pushVert = (x: number, y: number, z: number, rgb: [number, number, number]) => {
      positions[p++] = x; positions[p++] = y; positions[p++] = z;
      colors[pc++] = rgb[0] / 255; colors[pc++] = rgb[1] / 255; colors[pc++] = rgb[2] / 255;
    };
    // The section's four corners at parameter t, in CYCLIC order, so consecutive
    // pairs are the four faces and none of them is a diagonal:
    //
    //     0: +radial +normal   (outward, one side)
    //     1: +radial -normal   (outward, other side)
    //     2: -radial -normal   (inward,  other side)
    //     3: -radial +normal   (inward,  one side)
    //
    // `radial` is the unit direction to the point, so scaling it by R+-halfWidth
    // is what CENTRES the section on the sphere instead of hanging it outside.
    // `normal` is the circle's plane normal: constant along one arc, and tangent
    // to the sphere at every point of it.
    const corner = (
      out: number[], u: THREE.Vector3, v: THREE.Vector3, n: THREE.Vector3, t: number,
    ) => {
      const c = Math.cos(t), si = Math.sin(t);
      const rx = u.x * c + v.x * si, ry = u.y * c + v.y * si, rz = u.z * c + v.z * si;
      const outR = SPHERE_RADIUS + halfWidth, inR = SPHERE_RADIUS - halfWidth;
      const nx = n.x * halfWidth, ny = n.y * halfWidth, nz = n.z * halfWidth;
      out[0] = rx * outR + nx; out[1] = ry * outR + ny; out[2] = rz * outR + nz;
      out[3] = rx * outR - nx; out[4] = ry * outR - ny; out[5] = rz * outR - nz;
      out[6] = rx * inR - nx; out[7] = ry * inR - ny; out[8] = rz * inR - nz;
      out[9] = rx * inR + nx; out[10] = ry * inR + ny; out[11] = rz * inR + nz;
    };
    const ca = new Array<number>(12), cb = new Array<number>(12);
    for (const arc of arcs) {
      const { u, v, n, span, steps, rgb } = arc;
      for (let k = 0; k < steps; k++) {
        corner(ca, u, v, n, (k / steps) * span);
        corner(cb, u, v, n, ((k + 1) / steps) * span);
        // Face i joins corner i to corner i+1 across the step. Winding is not
        // load-bearing: the material is DoubleSide, which is also what lets the
        // Inside-Sphere view see the faces that point away from it.
        for (let i = 0; i < 4; i++) {
          const q = i * 3, r = ((i + 1) % 4) * 3;
          pushVert(ca[q]!, ca[q + 1]!, ca[q + 2]!, rgb);
          pushVert(ca[r]!, ca[r + 1]!, ca[r + 2]!, rgb);
          pushVert(cb[r]!, cb[r + 1]!, cb[r + 2]!, rgb);
          pushVert(ca[q]!, ca[q + 1]!, ca[q + 2]!, rgb);
          pushVert(cb[r]!, cb[r + 1]!, cb[r + 2]!, rgb);
          pushVert(cb[q]!, cb[q + 1]!, cb[q + 2]!, rgb);
        }
      }
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geo.computeBoundingSphere();
  }
}

// The vote normals as rays from the origin, scaled by weight. The one overlay
// here that is genuinely PER-VOTE, and the only remaining consumer of the
// `votes` readback.
export function updateAxisVectors(camera: Camera) {
  if (!camera.settings.showAxisVectors) return;
  const chosen = camera.pose?.votes ?? [];
  const anchorQuat = isSimulated(camera) ? camera.camQuat : (camera.pose?.positionDecode?.recoveredCamQuat ?? null);
  if (chosen.length === 0 || !anchorQuat) {
    camera.axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(0), 3));
    return;
  }
  let minW = Infinity, maxW = -Infinity;
  for (const vote of chosen) {
    if (vote.weight < minW) minW = vote.weight;
    if (vote.weight > maxW) maxW = vote.weight;
  }
  const wRange = maxW - minW;
  const positions = new Float32Array(chosen.length * 2 * 3);
  const colors = new Float32Array(chosen.length * 2 * 3);
  let ap = 0, apc = 0;
  for (const vote of chosen) {
    const normal = vote.n.clone().applyQuaternion(anchorQuat);
    const t = wRange > 0 ? (vote.weight - minW) / wRange : 0;
    const r = 1 - t, b = t;
    const len = maxW > 0 ? AXIS_VECTOR_LENGTH * (vote.weight / maxW) : 0;
    positions[ap++] = 0; positions[ap++] = 0; positions[ap++] = 0;
    positions[ap++] = normal.x * len; positions[ap++] = normal.y * len; positions[ap++] = normal.z * len;
    colors[apc++] = r; colors[apc++] = 0; colors[apc++] = b;
    colors[apc++] = r; colors[apc++] = 0; colors[apc++] = b;
  }
  camera.axisVectorsGeo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  camera.axisVectorsGeo.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
  camera.axisVectorsGeo.computeBoundingSphere();
}

export function updateGizmo(camera: SimulatedCamera): { hFovRad: number; vFovRad: number } {
  camera.camPos.set(camera.settings.camX, camera.settings.camY, camera.settings.camZ);
  // THE ONE PLACE tiltDeg BECOMES THREE'S PITCH. An unrotated THREE camera
  // looks down -Z, so -90 points it at the floor and tilt lifts it back toward
  // the horizon: pitch = tilt - 90. `euler` is order 'YXZ' (shared/constants.ts),
  // which is what makes this agree with the sweep's camQuatOf -- yaw about the
  // world vertical whatever the tilt is. Verified equal to 0 degrees at
  // tilt 58 / yaw -43; an 'XYZ' euler here would differ whenever BOTH are
  // nonzero, which is most of the sweep.
  euler.set(THREE.MathUtils.degToRad(camera.settings.tiltDeg - 90), THREE.MathUtils.degToRad(camera.settings.camYawDeg), 0);
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

// Vanishing-point sphere overlays (poles/circles/frustum/patch/recovered markers) --
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

  // BOTH, and the AND is the load-bearing half: the poles' actual position is
  // only ever written on a successful position decode, so gating on the axes
  // alone would leave them visible at a stale/default (0,0,0) whenever decode
  // failed -- the same failure updateRecoveredCamGizmo and
  // applyRecoveredFloorOverlay already guard against via positionDecode.
  //
  // This read `quadricPair` before: the same three vectors gated one stage
  // earlier. Nothing changes, because `positionDecode` cannot exist without the
  // lattice `recoveredAxes` waits for -- the pair was only ever as available as
  // the later of the two.
  const recoveredPolesVisible = settings.showRecoveredPoles && !!camera.pose?.recoveredAxes && !!camera.pose.positionDecode;
  camera.recoveredRowPoleA.visible = recoveredPolesVisible;
  camera.recoveredRowPoleB.visible = recoveredPolesVisible;
  camera.recoveredColPoleA.visible = recoveredPolesVisible;
  camera.recoveredColPoleB.visible = recoveredPolesVisible;
  camera.axisVectorsLines.visible = settings.showAxisVectors;
  // One mesh per kind, each gated on its own toggle. `showLineRings` does NOT
  // gate either -- it only decides how far each arc sweeps (see updateLineArcs),
  // so gating on it would hide the arcs it is meant to extend.
  camera.segmentArcsMesh.visible = settings.showLineSegments;
  camera.compositeArcsMesh.visible = settings.showCompositeLines;

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
    anchorPos = camera.pose?.positionDecode?.camPos ?? new THREE.Vector3();
    anchorQuat = camera.pose?.positionDecode?.recoveredCamQuat ?? null;
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

