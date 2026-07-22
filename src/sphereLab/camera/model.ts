import * as THREE from 'three';
import { CompositeLineDisplay, SegmentMerge } from '../pipeline/bucketFillJoin.ts';
import { BucketFillSegment } from '../pipeline/bucketFillSegments.ts';
import { GridPeriodPhaseResult } from '../pipeline/gridPeriodPhase.ts';
import { DecodeCellDebug, DecodeSampleGrid, GradientField, Marginals, PositionDecodeResult, ProjectedBins, RecoveredAxes, Vote } from '../types.ts';
import { PhysicalCameraSettings, SimulatedCameraSettings } from './settings.ts';

// ── Camera model ─────────────────────────────────────────────────────────
//
// Any number of these can exist at once (zero included -- see this file's
// header note); every per-camera THREE object/buffer lives on this object
// rather than as a module-level singleton.

export interface CameraBase {
  id: string;
  name: string;
  color: THREE.Color;

  // -- recovered/decoded state -- already fully source-agnostic (MATH_QUAT-
  // frame recovery + solveRecoveredCamQuat), unchanged by this stage.
  lastRecoveredAxes: RecoveredAxes | null;
  lastPositionDecode: PositionDecodeResult | null;
  lastDecodeGrid: DecodeSampleGrid | null;
  lastDecodeRotated: DecodeSampleGrid | null;
  lastDecodeCorrectness: (DecodeCellDebug | null)[][] | null;
  lastProjectedBins: ProjectedBins | null;
  lastMarginals: Marginals | null;
  lastVotes: Vote[];
  axesComputed: boolean;
  axesCapturing: boolean;
  lastAxesCapture: number;

  // -- capture/analysis buffers, shared shape for both camera types --
  rtSize: { w: number; h: number };
  aspect: number; // rtSize.w / rtSize.h -- replaces the old module-level RT_ASPECT
  pipRect: { x: number; y: number; w: number; h: number };
  captureDirty: boolean;
  lastPreviewUpdate: number;
  lastNoisedPreviewGray: Float64Array | null;
  lastDisplayedVectorField: GradientField | null;
  lastEffectiveField: GradientField | null;
  lastBucketFillSegments: BucketFillSegment[] | null;
  lastBucketFillColors: [number, number, number][] | null;
  // Per-pixel segment ownership from the flood fill itself (computeBucketFillRegions),
  // -1 = no segment -- kept around so the join walk can seed its own buffer
  // from it (see pipeline/bucketFillJoin.ts's computeJoinWalk).
  lastBucketFillRegionId: Int32Array | null;
  lastBucketFillMerges: SegmentMerge[] | null;
  lastBucketFillComposite: CompositeLineDisplay[] | null;
  // Merge-point markers, classified by how each merge's winning pair of
  // points was chosen -- see pipeline/bucketFillJoin.ts's computeJoinWalk
  // (mergeAt) for how blue/red/purple are decided.
  lastBucketFillBlueMerges: { x: number; y: number }[] | null;
  lastBucketFillOrangeMerges: { x: number; y: number }[] | null;
  lastBucketFillRedMerges: { x: number; y: number }[] | null;
  lastGridPeriodPhase: GridPeriodPhaseResult | null;
  // Interactive pan/zoom state for the period/phase debug plot (overlays/
  // gridPeriodPhaseOverlays.ts) -- null means "no interaction yet, use the
  // default bracket-relative view". Deliberately NOT in settings (not
  // persisted) -- this is scroll position, not a configuration choice, and
  // a saved zoom level from a totally different capture wouldn't mean
  // anything as a default for the next session.
  gridPeriodPhaseViewMin: number | null;
  gridPeriodPhaseViewMax: number | null;

  distortedPreviewData: Uint8Array; distortedPreviewTex: THREE.DataTexture;
  projectedPreviewData: Uint8Array; projectedPreviewTex: THREE.DataTexture;
  trueContamData: Uint8Array; trueContamTex: THREE.DataTexture;
  reconContamData: Uint8Array; reconContamTex: THREE.DataTexture;
  topGradientData: Uint8Array; topGradientTex: THREE.DataTexture;
  tangentWalkPathData: Uint8Array; tangentWalkPathTex: THREE.DataTexture;
  bucketFillData: Uint8Array; bucketFillTex: THREE.DataTexture;
  bucketFillJoinData: Uint8Array; bucketFillJoinTex: THREE.DataTexture;

  // -- THREE objects: recovered side (both camera types have these) --
  recoveredCamGizmo: THREE.Mesh; recoveredCamAxes: THREE.AxesHelper;
  recoveredRowPoleA: THREE.Mesh; recoveredRowPoleB: THREE.Mesh;
  recoveredColPoleA: THREE.Mesh; recoveredColPoleB: THREE.Mesh;
  recoveredFloorOverlayMat: THREE.MeshBasicMaterial; recoveredFloorOverlay: THREE.Mesh;

  // -- Great-sphere group: repositioned (not rotated) to the camera's own
  // origin each frame, since every direction it draws is expressed in WORLD
  // axes. --
  sphereAnchor: THREE.Object3D;
  sphereShell: THREE.Mesh;
  circlesGroup: THREE.Group;
  rowCirclePool: THREE.Line[]; colCirclePool: THREE.Line[];
  frustumLine: THREE.LineLoop;
  patchGeo: THREE.BufferGeometry; patchMat: THREE.MeshBasicMaterial; patchMesh: THREE.Mesh;
  // Rendered as a thin flat TRIANGLE RIBBON (2 triangles per circle segment,
  // extruded in-plane to +-halfWidth around the true SPHERE_RADIUS), not a
  // native GL line -- a real "fat line" addon (three/addons/lines,
  // LineSegments2/LineMaterial) was tried first for an adjustable stroke
  // weight, but rendered nothing at all in this environment (confirmed via
  // live dev-bridge inspection: draw calls happened, geometry data was
  // correct, no GL errors, yet zero pixels) for reasons not worth chasing
  // further -- ordinary triangles + MeshBasicMaterial has no such failure
  // mode. See updateGradientCirclesDebug for the ribbon construction.
  gradientCirclesGeo: THREE.BufferGeometry; gradientCirclesMat: THREE.MeshBasicMaterial; gradientCirclesLines: THREE.Mesh;
  axisVectorsGeo: THREE.BufferGeometry; axisVectorsMat: THREE.LineBasicMaterial; axisVectorsLines: THREE.LineSegments;
}
export interface SimulatedCamera extends CameraBase {
  type: 'simulated';
  settings: SimulatedCameraSettings;
  // Ground-truth pose, driven by settings.camX/Y/Z/camYawDeg/camPitchDeg --
  // see updateGizmo.
  camPos: THREE.Vector3; camQuat: THREE.Quaternion;
  gizmoCam: THREE.PerspectiveCamera; gizmoBody: THREE.Mesh; gizmoAxes: THREE.AxesHelper;
  camHelper: THREE.CameraHelper;
  camRT: THREE.WebGLRenderTarget;
  captureRTSize: { w: number; h: number };
  // Ground-truth pole markers (ROW_DIR/COL_DIR comparison) -- no equivalent
  // for a physical camera, since there's no ground truth to compare against.
  polesGroup: THREE.Group;
  rowPoleA: THREE.Mesh; rowPoleB: THREE.Mesh; colPoleA: THREE.Mesh; colPoleB: THREE.Mesh;
}
export interface PhysicalCamera extends CameraBase {
  type: 'physical';
  settings: PhysicalCameraSettings;
  lastRealCaptureGray: Float64Array | null;
  lastRealCaptureW: number; lastRealCaptureH: number;
  // The dev-bridge server's own id for the phone connection this camera was
  // auto-created for (see initDevBridge's realCapture handler) -- every
  // PhysicalCamera has one now, and always will: there's no manual/UI path
  // to create one anymore (see this file's header), only a real phone
  // connecting. That's what lets the tab bar's close button unconditionally
  // KICK any physical camera (see renderCameraTabs) instead of needing to
  // ask whether there's really a connection behind it to kick.
  connectionId: string;
  // Purely a reflection of whatever mode.html's mode toggle last reported
  // (see devBridge/client.ts's captureMode handler) -- Sphere Lab never
  // sets this itself, only displays it (see ui/cameraPanel.ts).
  captureMode: 'single' | 'video';
  // Mirrors axesCapturing, but tracks what was last actually SENT to the
  // phone as a captureReady signal (see main.ts's animate loop), so that
  // signal only goes out on a genuine true/false transition instead of
  // every frame.
  lastReportedReady: boolean;
}
export type Camera = SimulatedCamera | PhysicalCamera;
