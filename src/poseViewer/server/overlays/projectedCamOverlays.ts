import * as THREE from 'three';
import { type Camera } from '../../shared/camera/model.ts';
import { isPhysical } from '../camera/store.ts';
import { positionReadout } from '../ui/dom.ts';

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

// ── THE SECOND drawSampleLattice IS GONE, and it was the dead one ────────
//
// Two functions drew a "sample lattice": this one, and the one inside
// gridPeriodPhaseOverlays.ts's drawGridPeriodPhaseProjected. The second
// superseded the first -- main.ts calls only that -- and this one survived as a
// stub kept "for reference/dev-bridge use", reachable by bare name and doing
// nothing but hiding its own canvas.
//
// Deleted rather than reimplemented alongside the real one. Two overlays with
// one name, one of them hollow, is what a reader has to disambiguate before they
// can start, and the reference value it was kept for is now served by an
// implementation that actually runs. Its `#sampleLattice` canvas, that canvas's
// CSS and both of its ui/dom.ts handles went with it -- a private surface with
// no writer left is the shape the dead-code guard cannot see, since
// `noUnusedLocals` says nothing about an unreferenced export.
