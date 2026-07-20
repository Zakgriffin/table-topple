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
  scene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), mat));
  return { mat, scene };
}
export const previewQuad = makeQuadRenderer({});
export const projectedQuad = makeQuadRenderer({});
export const trueContamQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export const reconContamQuad = makeQuadRenderer({ transparent: true, depthTest: false, depthWrite: false });
export function renderQuad(q: { mat: THREE.MeshBasicMaterial; scene: THREE.Scene }, tex: THREE.Texture, x: number, y: number, w: number, h: number) {
  q.mat.map = tex;
  renderer.setViewport(x, y, w, h);
  renderer.setScissor(x, y, w, h);
  renderer.setScissorTest(true);
  renderer.render(q.scene, quadCam);
}
export function renderPreviewViewport(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(previewQuad, camera.distortedPreviewTex, x, y, w, h); }
export function renderProjectedViewport(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(projectedQuad, camera.projectedPreviewTex, x, y, w, h); }
export function renderTrueContamOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(trueContamQuad, camera.trueContamTex, x, y, w, h); }
export function renderReconContamOverlay(camera: Camera, x: number, y: number, w: number, h: number) { renderQuad(reconContamQuad, camera.reconContamTex, x, y, w, h); }
