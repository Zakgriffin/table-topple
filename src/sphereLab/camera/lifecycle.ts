import { markCaptureDirty, renderCamRT } from '../pipeline/capture.ts';
import { buildProjectedTexture } from '../pipeline/decodeGrid.ts';
import { updateDistortedPreview } from '../pipeline/preview.ts';
import { globalState } from '../state.ts';
import { refreshCameraPanel, renderCameraTabs } from '../ui/cameraPanel.ts';
import { layoutPip } from '../ui/layout.ts';
import { setMode } from '../ui/mode.ts';
import { createSimulatedCamera, destroyCamera } from './factory.ts';
import { Camera, PhysicalCamera } from './model.ts';
import { activeCameraId, cameras, isSimulated, nextCameraColor, setActiveCameraId } from './store.ts';

export function findPhysicalCameraByConnection(connectionId: string): PhysicalCamera | undefined {
  for (const camera of cameras.values()) {
    if (camera.type === 'physical' && camera.connectionId === connectionId) return camera;
  }
  return undefined;
}

// Brings a freshly-created-or-reactivated camera's capture pipeline up to
// date -- the same handful of calls every camera-creation path needs.
export function primeCameraForDisplay(camera: Camera) {
  if (isSimulated(camera)) renderCamRT(camera); // populate camRT before reading it back below, so the preview isn't blank for the first frame or two
  updateDistortedPreview(camera);
  if (globalState.mode === 'projected') buildProjectedTexture(camera);
  markCaptureDirty(camera);
  layoutPip(camera);
}

// Adds a new simulated camera ALONGSIDE whatever already exists and makes it
// active. Offsets its default X position a few units per already-existing
// camera so a fresh gizmo doesn't spawn exactly on top of another one.
export function addSimulatedCamera() {
  const camera = createSimulatedCamera(nextCameraColor());
  camera.settings.camX += (cameras.size % 6) * 3;
  cameras.set(camera.id, camera);
  setActiveCameraId(camera.id);
  primeCameraForDisplay(camera);
  renderCameraTabs();
  refreshCameraPanel();
}

// Selects the Global tab -- no camera active, only globalSettingsSection
// shown (see refreshCameraPanel). Every other mode (Through-Cam/Inside-
// Sphere/Projected-Cam) needs an active camera to render anything, so this
// forces back to World rather than leaving one of them showing a blank
// screen with nothing selected to show.
export function selectGlobalTab() {
  if (activeCameraId === '') return;
  setActiveCameraId('');
  if (globalState.mode !== 'world') setMode('world');
  renderCameraTabs();
  refreshCameraPanel();
}

// Tears down one camera. If it was the active one, falls back to whichever
// camera is next in the map, or -- if that was the last camera left -- to
// the Global tab. Zero cameras is a normal, supported state (see this
// file's header): there's no more auto-created replacement papering over it.
export function removeCameraTab(id: string) {
  const camera = cameras.get(id);
  if (!camera) return;
  const wasActive = id === activeCameraId;
  destroyCamera(camera);
  if (wasActive) {
    const next = cameras.values().next().value;
    if (next) {
      setActiveCameraId(next.id);
      primeCameraForDisplay(next);
    } else {
      setActiveCameraId('');
      if (globalState.mode !== 'world') setMode('world');
    }
  }
  renderCameraTabs();
  refreshCameraPanel();
}
