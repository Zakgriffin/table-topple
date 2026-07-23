import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { isPhysical } from '../camera/store.ts';
import { positionReadout, sampleLatticeCanvas, sampleLatticeCtx } from '../ui/dom.ts';

// ── Projected-Cam sample lattice ─────────────────────────────────────────

export function updatePositionReadoutText(camera: Camera) {
  if (!positionReadout) return;
  let decodeLines: string;
  if (camera.lastPositionDecode) {
    const rec = camera.lastPositionDecode.camPos;
    if (isPhysical(camera)) {
      decodeLines =
        `torus cell: row ${camera.lastPositionDecode.row}  col ${camera.lastPositionDecode.col}\n` +
        `consistency: ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%\n` +
        `recovered camPos: (${rec.x.toFixed(2)}, ${rec.y.toFixed(2)}, ${rec.z.toFixed(2)})`;
    } else {
      const errPos = rec.distanceTo(camera.camPos);
      const errOrientationDeg = THREE.MathUtils.radToDeg(camera.camQuat.angleTo(camera.lastPositionDecode.recoveredCamQuat));
      decodeLines =
        `torus cell: row ${camera.lastPositionDecode.row}  col ${camera.lastPositionDecode.col}\n` +
        `consistency: ${(camera.lastPositionDecode.consistency * 100).toFixed(1)}%\n` +
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

export function hideSampleLattice() {
  sampleLatticeCanvas.style.display = 'none';
}

// Unreferenced by the real app (replaced by gridPeriodPhaseOverlays.ts's
// sample lattice, see this session's chat) -- left defined for reference/
// dev-bridge use. NOTE: camera.settings.showSampleLattice was reassigned to
// gate the NEW lattice when the old dedicated toggle was removed, so if
// this is ever called manually again, it'll piggyback on that toggle's
// current value rather than anything meant for this function specifically.
export function drawSampleLattice(camera: Camera, x: number, y: number, w: number, h: number) {
  if (!camera.settings.showSampleLattice) { hideSampleLattice(); return; }
  const grid = camera.lastDecodeRotated;
  if (!grid || !camera.lastProjectedBins) { hideSampleLattice(); return; }
  const { maxU, binWidthU, minV, binWidthV, w: bw, h: bh } = camera.lastProjectedBins;

  sampleLatticeCanvas.style.display = 'block';
  sampleLatticeCanvas.style.left = x + 'px';
  sampleLatticeCanvas.style.top = y + 'px';
  sampleLatticeCanvas.width = Math.round(w);
  sampleLatticeCanvas.height = Math.round(h);
  sampleLatticeCanvas.style.width = w + 'px';
  sampleLatticeCanvas.style.height = h + 'px';
  const ctx = sampleLatticeCtx;
  ctx.clearRect(0, 0, sampleLatticeCanvas.width, sampleLatticeCanvas.height);

  const radius = 3;
  for (let i = 0; i < grid.rows; i++) {
    for (let j = 0; j < grid.cols; j++) {
      const pt = grid.points[i][j];
      if (!pt.valid) continue;
      const bu = (maxU - pt.u) / binWidthU;
      const bv = (pt.v - minV) / binWidthV;
      if (bu < 0 || bu >= bw || bv < 0 || bv >= bh) continue;
      const cx = (bu / bw) * sampleLatticeCanvas.width;
      const cy = (1 - bv / bh) * sampleLatticeCanvas.height;
      const debug = camera.lastDecodeCorrectness ? camera.lastDecodeCorrectness[i][j] : null;
      ctx.beginPath();
      ctx.arc(cx, cy, radius, 0, Math.PI * 2);
      ctx.fillStyle = debug ? (debug.bit ? '#000' : '#fff') : '#888';
      ctx.fill();
      ctx.strokeStyle = debug ? (debug.correct ? '#0f0' : '#f00') : 'rgba(0,0,0,0.6)';
      ctx.lineWidth = debug ? 2 : 1;
      ctx.stroke();
    }
  }
}
