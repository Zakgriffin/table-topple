import * as THREE from 'three';
import { CIRCLE_SEGMENTS, DEBUG_LAYER, PATCH_RES, SPHERE_RADIUS } from '../constants.ts';
import { colLineKs, rowLineKs } from '../math/geometry.ts';
import { scene } from '../scene/renderer.ts';
import { Camera, CameraBase, PhysicalCamera, SimulatedCamera } from './model.ts';
import { createDefaultPhysicalSettings, createDefaultSimulatedSettings } from './settings.ts';
import { bumpCameraSerial, cameras, isSimulated, nextCameraSerial } from './store.ts';

// ── Camera factories ─────────────────────────────────────────────────────
//
// Build every per-camera THREE object/buffer this file used to allocate
// once at module scope, add them to the shared `scene`, and return a fully
// populated Camera. Real allocate-N-of-them functions -- addSimulatedCamera
// (the tab bar's "+" button) and initDevBridge's realCapture handler (a
// phone connecting) both call these.

export function makeCameraBaseParts(rtSize: { w: number; h: number }, color: THREE.Color) {
  const aspect = rtSize.w / rtSize.h;

  // Recovered/decoded pose gizmo: TRANSLUCENT in the camera's own color --
  // translucent consistently means "recovered" (a decode, not a certainty)
  // across every camera, including physical ones (which have no
  // ground-truth gizmo at all). Ground-truth (only simulated cameras have
  // one, see createSimulatedCamera) is the solid/opaque one instead.
  const recoveredCamGizmo = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.25, 0.4),
    new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.4 }),
  );
  recoveredCamGizmo.visible = false;
  scene.add(recoveredCamGizmo);
  const recoveredCamAxes = new THREE.AxesHelper(0.6);
  recoveredCamGizmo.add(recoveredCamAxes);
  recoveredCamGizmo.traverse((o) => o.layers.set(DEBUG_LAYER));

  const sphereAnchor = new THREE.Object3D();
  scene.add(sphereAnchor);

  const sphereShell = new THREE.Mesh(
    new THREE.SphereGeometry(SPHERE_RADIUS, 48, 32),
    new THREE.MeshBasicMaterial({ color: 0x88aaff, transparent: true, opacity: 0.08, side: THREE.DoubleSide, depthWrite: false }),
  );
  sphereAnchor.add(sphereShell);
  sphereShell.layers.set(DEBUG_LAYER);

  const circlesGroup = new THREE.Group();
  sphereAnchor.add(circlesGroup);

  function buildCirclePool(count: number, color: number): THREE.Line[] {
    const lines: THREE.Line[] = [];
    for (let i = 0; i < count; i++) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array((CIRCLE_SEGMENTS + 1) * 3), 3));
      const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.5 }));
      line.layers.set(DEBUG_LAYER);
      circlesGroup.add(line);
      lines.push(line);
    }
    return lines;
  }
  const rowCirclePool = buildCirclePool(rowLineKs.length, 0xff5555);
  const colCirclePool = buildCirclePool(colLineKs.length, 0x5599ff);

  const frustumLine = new THREE.LineLoop(
    new THREE.BufferGeometry(),
    new THREE.LineBasicMaterial({ color: 0xffffff, linewidth: 2 }),
  );
  sphereAnchor.add(frustumLine);
  frustumLine.layers.set(DEBUG_LAYER);

  // Viewport-image patch: literally the camera's rendered pixels, wrapped
  // onto the sphere over exactly the solid angle the camera's frustum
  // subtends.
  const patchGeo = new THREE.BufferGeometry();
  {
    const verts: number[] = [], uvs: number[] = [], idx: number[] = [];
    for (let j = 0; j <= PATCH_RES; j++) for (let i = 0; i <= PATCH_RES; i++) { verts.push(0, 0, 0); uvs.push(i / PATCH_RES, j / PATCH_RES); }
    for (let j = 0; j < PATCH_RES; j++) {
      for (let i = 0; i < PATCH_RES; i++) {
        const a = j * (PATCH_RES + 1) + i, b = a + 1, c = a + PATCH_RES + 1, d = c + 1;
        idx.push(a, c, b, b, c, d);
      }
    }
    patchGeo.setAttribute('position', new THREE.Float32BufferAttribute(verts, 3));
    patchGeo.setAttribute('uv', new THREE.Float32BufferAttribute(uvs, 2));
    patchGeo.setIndex(idx);
  }

  const distortedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const distortedPreviewTex = new THREE.DataTexture(distortedPreviewData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  distortedPreviewTex.flipY = false;
  distortedPreviewTex.colorSpace = THREE.SRGBColorSpace;

  const patchMat = new THREE.MeshBasicMaterial({ map: distortedPreviewTex, side: THREE.DoubleSide });
  const patchMesh = new THREE.Mesh(patchGeo, patchMat);
  sphereAnchor.add(patchMesh);
  patchMesh.layers.set(DEBUG_LAYER);

  const projectedPreviewData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const projectedPreviewTex = new THREE.DataTexture(projectedPreviewData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  projectedPreviewTex.flipY = false;
  projectedPreviewTex.colorSpace = THREE.SRGBColorSpace;

  const trueContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const trueContamTex = new THREE.DataTexture(trueContamData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  trueContamTex.flipY = false;

  const reconContamData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const reconContamTex = new THREE.DataTexture(reconContamData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  reconContamTex.flipY = false;

  const topGradientData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const topGradientTex = new THREE.DataTexture(topGradientData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  topGradientTex.flipY = false;

  const tangentWalkPathData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const tangentWalkPathTex = new THREE.DataTexture(tangentWalkPathData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  tangentWalkPathTex.flipY = false;
  // These 3 overlays (unlike trueContam/reconContam/topGradient, which paint
  // one FIXED color everywhere and only vary alpha) paint a DIFFERENT color
  // per claimed pixel, next to unclaimed pixels left at RGB=0,alpha=0 --
  // under the default LinearFilter, GPU bilinear sampling blends RGB across
  // texel boundaries independent of alpha, so every claimed pixel bleeds
  // toward black at its edges. Sparse claims (bucket-fill-join especially,
  // at low step counts) make this read as an overall gray wash rather than
  // a subtle edge fringe. NearestFilter (already used by scene/floor.ts's
  // pattern texture for the same "don't blur discrete per-cell data"
  // reason) shows each field pixel as one flat block instead.
  tangentWalkPathTex.magFilter = THREE.NearestFilter;

  const bucketFillData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const bucketFillTex = new THREE.DataTexture(bucketFillData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  bucketFillTex.flipY = false;
  bucketFillTex.magFilter = THREE.NearestFilter;

  const bucketFillJoinData = new Uint8Array(rtSize.w * rtSize.h * 4);
  const bucketFillJoinTex = new THREE.DataTexture(bucketFillJoinData, rtSize.w, rtSize.h, THREE.RGBAFormat);
  bucketFillJoinTex.flipY = false;
  bucketFillJoinTex.magFilter = THREE.NearestFilter;

  // Reuses the SAME projectedPreviewTex "Projected Cam" mode already builds
  // (not a separate computation) as a decal on a plane placed at the
  // DECODED pose in the actual 3D world.
  const recoveredFloorOverlayMat = new THREE.MeshBasicMaterial({ map: projectedPreviewTex, side: THREE.DoubleSide, transparent: true, opacity: 0.92 });
  const recoveredFloorOverlay = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), recoveredFloorOverlayMat);
  recoveredFloorOverlay.visible = false;
  scene.add(recoveredFloorOverlay);
  recoveredFloorOverlay.layers.set(DEBUG_LAYER);

  function makeRecoveredPoleMarker(color: number): THREE.Mesh {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.09, 12, 8), new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.6 }));
    m.layers.set(DEBUG_LAYER);
    m.visible = false;
    sphereAnchor.add(m);
    return m;
  }
  const recoveredRowPoleA = makeRecoveredPoleMarker(0xff0000);
  const recoveredRowPoleB = makeRecoveredPoleMarker(0xff0000);
  const recoveredColPoleA = makeRecoveredPoleMarker(0x0000ff);
  const recoveredColPoleB = makeRecoveredPoleMarker(0x0000ff);

  // Rendered as a flat triangle ribbon, not a native GL line -- see
  // model.ts's own comment on gradientCirclesGeo for why. DoubleSide since
  // this is a debug overlay meant to stay visible from any orbit angle,
  // including nearly edge-on to a given circle's own plane.
  const gradientCirclesGeo = new THREE.BufferGeometry();
  const gradientCirclesMat = new THREE.MeshBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.35, side: THREE.DoubleSide });
  const gradientCirclesLines = new THREE.Mesh(gradientCirclesGeo, gradientCirclesMat);
  gradientCirclesLines.layers.set(DEBUG_LAYER);
  sphereAnchor.add(gradientCirclesLines);

  const axisVectorsGeo = new THREE.BufferGeometry();
  const axisVectorsMat = new THREE.LineBasicMaterial({ vertexColors: true, transparent: true, opacity: 0.6 });
  const axisVectorsLines = new THREE.LineSegments(axisVectorsGeo, axisVectorsMat);
  axisVectorsLines.layers.set(DEBUG_LAYER);
  axisVectorsLines.visible = false;
  sphereAnchor.add(axisVectorsLines);

  const base: Omit<CameraBase, 'id' | 'name' | 'color'> = {
    lastRecoveredAxes: null, lastPositionDecode: null, lastDecodeGrid: null, lastDecodeRotated: null,
    lastDecodeCorrectness: null, lastProjectedBins: null, lastMarginals: null, lastVotes: [],
    axesComputed: false, axesCapturing: false, lastAxesCapture: 0,
    rtSize: { ...rtSize }, aspect, pipRect: { x: 0, y: 0, w: 0, h: 0 }, captureDirty: true, lastPreviewUpdate: 0,
    lastNoisedPreviewGray: null, lastDisplayedVectorField: null,
    lastBucketFillSegments: null, lastBucketFillColors: null, lastBucketFillRegionId: null, lastBucketFillMerges: null, lastBucketFillComposite: null,
    lastBucketFillBlueMerges: null, lastBucketFillOrangeMerges: null, lastBucketFillRedMerges: null,
    lastGridPeriodPhase: null,
    gridPeriodPhaseViewMin: null, gridPeriodPhaseViewMax: null,
    distortedPreviewData, distortedPreviewTex, projectedPreviewData, projectedPreviewTex,
    trueContamData, trueContamTex, reconContamData, reconContamTex, topGradientData, topGradientTex,
    tangentWalkPathData, tangentWalkPathTex, bucketFillData, bucketFillTex, bucketFillJoinData, bucketFillJoinTex,
    recoveredCamGizmo, recoveredCamAxes,
    recoveredRowPoleA, recoveredRowPoleB, recoveredColPoleA, recoveredColPoleB,
    recoveredFloorOverlayMat, recoveredFloorOverlay,
    sphereAnchor, sphereShell, circlesGroup, rowCirclePool, colCirclePool, frustumLine,
    patchGeo, patchMat, patchMesh, gradientCirclesGeo, gradientCirclesMat, gradientCirclesLines,
    axisVectorsGeo, axisVectorsMat, axisVectorsLines,
  };
  return base;
}

export function createSimulatedCamera(color: THREE.Color): SimulatedCamera {
  const settings = createDefaultSimulatedSettings();
  const rtSize = { w: Math.round(settings.viewportW), h: Math.round(settings.viewportH) };
  const base = makeCameraBaseParts(rtSize, color);
  const aspect = rtSize.w / rtSize.h;

  const gizmoCam = new THREE.PerspectiveCamera(50, aspect, 0.05, 500);
  scene.add(gizmoCam);

  // Ground-truth (assumed) pose gizmo: SOLID/opaque in the same color the
  // recovered gizmo above uses translucent -- see makeCameraBaseParts' comment.
  const gizmoBody = new THREE.Mesh(
    new THREE.BoxGeometry(0.3, 0.25, 0.4),
    new THREE.MeshStandardMaterial({ color }),
  );
  scene.add(gizmoBody);
  const gizmoAxes = new THREE.AxesHelper(0.6);
  gizmoBody.add(gizmoAxes);
  gizmoBody.traverse((o) => o.layers.set(DEBUG_LAYER));

  const camHelper = new THREE.CameraHelper(gizmoCam);
  scene.add(camHelper);
  camHelper.layers.set(DEBUG_LAYER);

  const captureRTSize = { w: rtSize.w * settings.captureSupersample, h: rtSize.h * settings.captureSupersample };
  const camRT = new THREE.WebGLRenderTarget(captureRTSize.w, captureRTSize.h, { colorSpace: THREE.SRGBColorSpace });

  const polesGroup = new THREE.Group();
  base.sphereAnchor.add(polesGroup);
  function makePoleMarker(color: number): THREE.Mesh {
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.06, 12, 8), new THREE.MeshBasicMaterial({ color }));
    polesGroup.add(m);
    return m;
  }
  const rowPoleA = makePoleMarker(0xff5555);
  const rowPoleB = makePoleMarker(0xff5555);
  const colPoleA = makePoleMarker(0x5599ff);
  const colPoleB = makePoleMarker(0x5599ff);
  // traverse (not a single .set on the empty group) -- must run AFTER every
  // marker is added, since layers are per-object and not inherited by
  // children added later. Missing this left the ground-truth pole markers
  // on the default layer, which gizmoCam DOES render -- they'd leak into
  // the simulated capture instead of staying debug-only.
  polesGroup.traverse((o) => o.layers.set(DEBUG_LAYER));

  const camera: SimulatedCamera = {
    ...base,
    id: `sim-${nextCameraSerial}`, name: `Simulated ${nextCameraSerial}`, color,
    type: 'simulated', settings,
    camPos: new THREE.Vector3(), camQuat: new THREE.Quaternion(),
    gizmoCam, gizmoBody, gizmoAxes, camHelper, camRT, captureRTSize,
    polesGroup, rowPoleA, rowPoleB, colPoleA, colPoleB,
  };
  bumpCameraSerial();
  return camera;
}

export function createPhysicalCamera(color: THREE.Color, connectionId: string): PhysicalCamera {
  const settings = createDefaultPhysicalSettings();
  const rtSize = { w: Math.round(settings.viewportW), h: Math.round(settings.viewportH) };
  const base = makeCameraBaseParts(rtSize, color);
  const camera: PhysicalCamera = {
    ...base,
    id: `phys-${nextCameraSerial}`, name: `Physical ${nextCameraSerial}`, color,
    type: 'physical', settings,
    lastRealCaptureGray: null, lastRealCaptureW: 0, lastRealCaptureH: 0,
    connectionId, captureMode: 'single', lastReportedReady: true,
  };
  bumpCameraSerial();
  return camera;
}

// Disposes every THREE object/geometry/material/texture/render-target a
// camera owns and removes them from `scene` -- the mirror image of the
// factories above. Called from removeCameraTab (a "x"/kick click) and, for a
// physical camera specifically, from initDevBridge's captureDisconnected
// handler.
export function destroyCamera(camera: Camera) {
  const disposeObj = (o: THREE.Object3D) => {
    scene.remove(o);
    o.traverse((child) => {
      const mesh = child as THREE.Mesh;
      if (mesh.geometry) mesh.geometry.dispose();
      const mat = (child as any).material;
      if (Array.isArray(mat)) mat.forEach((m) => m.dispose());
      else if (mat) mat.dispose();
    });
  };
  disposeObj(camera.recoveredCamGizmo);
  disposeObj(camera.sphereAnchor); // takes sphereShell/circlesGroup/pools/frustumLine/patchMesh/pole markers/gradientCircles/axisVectors with it
  disposeObj(camera.recoveredFloorOverlay);
  camera.distortedPreviewTex.dispose();
  camera.projectedPreviewTex.dispose();
  camera.trueContamTex.dispose();
  camera.reconContamTex.dispose();
  camera.topGradientTex.dispose();
  camera.tangentWalkPathTex.dispose();
  camera.bucketFillTex.dispose();
  camera.bucketFillJoinTex.dispose();
  if (isSimulated(camera)) {
    disposeObj(camera.gizmoBody);
    scene.remove(camera.gizmoCam);
    scene.remove(camera.camHelper);
    camera.camHelper.dispose();
    camera.camRT.dispose();
  }
  cameras.delete(camera.id);
}
