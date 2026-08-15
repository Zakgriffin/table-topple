import * as THREE from 'three';
import { type Camera } from '../camera/model.ts';
import { isPhysical } from '../camera/store.ts';
import { positionReadout, sampleLatticeCanvas } from '../ui/dom.ts';

// ── Projected-Cam sample lattice ─────────────────────────────────────────

export function updatePositionReadoutText(camera: Camera) {
  if (!positionReadout) return;
  let decodeLines: string;
  const decode = camera.pose?.positionDecode;
  if (decode) {
    const rec = decode.camPos;
    if (isPhysical(camera)) {
      decodeLines =
        `torus cell: row ${decode.row}  col ${decode.col}\n` +
        `consistency: ${(decode.consistency * 100).toFixed(1)}%\n` +
        `recovered camPos: (${rec.x.toFixed(2)}, ${rec.y.toFixed(2)}, ${rec.z.toFixed(2)})`;
    } else {
      const errPos = rec.distanceTo(camera.camPos);
      const errOrientationDeg = THREE.MathUtils.radToDeg(camera.camQuat.angleTo(decode.recoveredCamQuat));
      decodeLines =
        `torus cell: row ${decode.row}  col ${decode.col}\n` +
        `consistency: ${(decode.consistency * 100).toFixed(1)}%\n` +
        `recovered camPos: (${rec.x.toFixed(2)}, ${rec.y.toFixed(2)}, ${rec.z.toFixed(2)})\n` +
        `true camPos: (${camera.camPos.x.toFixed(2)}, ${camera.camPos.y.toFixed(2)}, ${camera.camPos.z.toFixed(2)})\n` +
        `error: ${errPos.toFixed(3)} world units\n` +
        `orientation error: ${errOrientationDeg.toFixed(2)}° (recoveredCamQuat vs true camQuat -- ground-truth diagnostic, lab-only)`;
    }
  } else {
    decodeLines = 'position decode: no match (need periodicity + a successful orientation/distance fit)';
  }
  positionReadout.textContent = decodeLines;
}

function hideSampleLattice() {
  sampleLatticeCanvas.style.display = 'none';
}

// Unreferenced by the real app (replaced by gridPeriodPhaseOverlays.ts's
// sample lattice, see this session's chat) -- left defined for reference/
// dev-bridge use. NOTE: camera.settings.showSampleLattice was reassigned to
// gate the NEW lattice when the old dedicated toggle was removed, so if
// this is ever called manually again, it'll piggyback on that toggle's
// current value rather than anything meant for this function specifically.
// ── THE LATTICE IS DARK, pending pose2 plumbing ──
//
// It drew every decode sample point on the Projected-Cam rect, filled by its
// sampled bit and ringed green/red by whether that bit matched the printed
// board. Both inputs -- `decodeRotated` and `decodeCorrectness` -- were pipeline
// intermediates, and they are gone from Intermediates entirely: pose2 keeps the
// decode grid on the device and reads back 128 bytes of pose.
//
// The canvas, its sizing and its toggle are untouched, so this comes back by
// filling it rather than by rebuilding the panel.
export function drawSampleLattice(_camera: Camera, _x: number, _y: number, _w: number, _h: number) {
  hideSampleLattice();
}
