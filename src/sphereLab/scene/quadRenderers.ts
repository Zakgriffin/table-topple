import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { renderer } from './renderer.ts';

// ── Reusable full-screen quad renderers (shared infra, NOT per-camera) ───
//
// A plain full-screen textured quad, rendered instead of a live gizmoCam
// scene pass for the PIP box / Through-Cam / Projected-Cam / contamination
// overlays -- each camera owns its OWN texture (distortedPreviewTex etc,
// see CameraBase above), but the Scene/Material/Mesh doing the actual blit
// is shared, reusable machinery: swap `.map` to whichever camera's texture
// needs drawing right before each render call.
export const quadCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
export function makeQuadRenderer(matOpts: THREE.MeshBasicMaterialParameters) {
  const mat = new THREE.MeshBasicMaterial(matOpts);
  const scene = new THREE.Scene();
  const geometry = new THREE.PlaneGeometry(2, 2);
  scene.add(new THREE.Mesh(geometry, mat));
  return { mat, scene, geometry };
}
// PlaneGeometry(2,2)'s 4 vertices, in order, are TL/TR/BL/BR with default UVs
// (0,1)/(1,1)/(0,0)/(1,0) -- rotating the sampled image 90 degrees clockwise
// means each corner now shows what used to be one step counter-clockwise
// from it (TL<-BL, TR<-TL, BR<-TR, BL<-BR); applying that permutation N
// times gives every multiple of 90.
const BASE_QUAD_UVS: [number, number][] = [[0, 1], [1, 1], [0, 0], [1, 0]];
function rotatedQuadUVs(steps: number): [number, number][] {
  let uv = BASE_QUAD_UVS;
  const CW_FROM = [2, 0, 3, 1]; // new[i] = old[CW_FROM[i]], one 90-degree step
  for (let s = 0; s < ((steps % 4) + 4) % 4; s++) uv = CW_FROM.map((i) => uv[i]) as [number, number][];
  return uv;
}
export const previewQuad = makeQuadRenderer({});
export const projectedQuad = makeQuadRenderer({});
export const trueContamQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export const reconContamQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export const topGradientQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export const tangentWalkPathQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export const bucketFillQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export const bucketFillJoinQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export function renderQuad(q: { mat: THREE.MeshBasicMaterial; scene: THREE.Scene }, tex: THREE.Texture, x: number, y: number, w: number, h: number) {
  q.mat.map = tex;
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(q.scene, quadCam);
}
export function renderPreviewViewport(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(previewQuad, camera.distortedPreviewTex, x, y, w, h); }
// rotationSteps: multiples of 90 degrees (0-3), purely a display-time
// rotation for the "use true cardinal orientation" toggle -- see settings.ts's
// useTrueCardinalOrientation doc comment. Applied by permuting projectedQuad's
// OWN geometry UVs (never camera.projectedPreviewTex's rotation/center --
// that texture object is also the World-view "recovered floor" overlay's
// decal map, camera/factory.ts's recoveredFloorOverlayMat, so mutating ITS
// state would leak the rotation into that unrelated overlay across mode
// switches). Doesn't touch the texture's actual pixel data either way.
export function renderProjectedViewport(camera: Camera, x: number, y: number, w: number, h: number, rotationSteps = 0) {
  const uvAttr = projectedQuad.geometry.attributes.uv as THREE.BufferAttribute;
  uvAttr.copyArray(rotatedQuadUVs(rotationSteps).flat());
  uvAttr.needsUpdate = true;
  renderQuad(projectedQuad, camera.projectedPreviewTex, x, y, w, h);
}
export function renderTrueContamOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(trueContamQuad, camera.trueContamTex, x, y, w, h); }
export function renderReconContamOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(reconContamQuad, camera.reconContamTex, x, y, w, h); }
export function renderTopGradientOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(topGradientQuad, camera.topGradientTex, x, y, w, h); }
export function renderTangentWalkPathOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(tangentWalkPathQuad, camera.tangentWalkPathTex, x, y, w, h); }
export function renderBucketFillOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(bucketFillQuad, camera.bucketFillTex, x, y, w, h); }
export function renderBucketFillJoinOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(bucketFillJoinQuad, camera.bucketFillJoinTex, x, y, w, h); }
