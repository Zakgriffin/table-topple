import * as THREE from 'three';
import { Camera } from '../camera/model.ts';
import { isPhysical } from '../camera/store.ts';
import { GRID_STEP } from '../constants.ts';
import { marginalHueColor } from '../pipeline/positionLM.ts';
import { marginalBottomCanvas, marginalBottomCtx, marginalRightCanvas, marginalRightCtx, positionReadout, sampleLatticeCanvas, sampleLatticeCtx } from '../ui/dom.ts';

// ── Projected-Cam marginal graphs / sample lattice ───────────────────────

export const MARGINAL_THICKNESS = 90;

export function drawMarginalLines(camera: Camera, x: number, y: number, w: number, h: number) {
  if (!camera.lastMarginals) { hideMarginalLines(); return; }
  const m = camera.lastMarginals;

  marginalRightCanvas.style.display = 'block';
  marginalRightCanvas.style.left = (x + w) + 'px';
  marginalRightCanvas.style.top = y + 'px';
  marginalRightCanvas.width = MARGINAL_THICKNESS;
  marginalRightCanvas.height = Math.round(h);
  marginalRightCanvas.style.width = MARGINAL_THICKNESS + 'px';
  marginalRightCanvas.style.height = h + 'px';
  const rc = marginalRightCtx;
  rc.clearRect(0, 0, marginalRightCanvas.width, marginalRightCanvas.height);
  {
    const n = m.rowMag.length;
    let maxMag = 0;
    for (let i = 0; i < n; i++) if (m.rowMag[i] > maxMag) maxMag = m.rowMag[i];
    rc.lineWidth = 1;
    let prevPx = 0, prevPy = 0;
    for (let i = 0; i < n; i++) {
      const py = (1 - i / n) * marginalRightCanvas.height;
      const px = maxMag > 0 ? (m.rowMag[i] / maxMag) * (MARGINAL_THICKNESS - 4) : 0;
      if (i > 0) {
        rc.strokeStyle = marginalHueColor(m.rowHueCx[i], m.rowSumCy[i]);
        rc.beginPath(); rc.moveTo(prevPx, prevPy); rc.lineTo(px, py); rc.stroke();
      }
      prevPx = px; prevPy = py;
    }
    if (m.rowPeriod) {
      rc.strokeStyle = 'rgba(255,80,80,0.6)';
      for (let py = m.rowPhase; py < n; py += m.rowPeriod) {
        const yy = (1 - py / n) * marginalRightCanvas.height;
        rc.beginPath(); rc.moveTo(0, yy); rc.lineTo(MARGINAL_THICKNESS, yy); rc.stroke();
      }
    }
  }

  marginalBottomCanvas.style.display = 'block';
  marginalBottomCanvas.style.left = x + 'px';
  marginalBottomCanvas.style.top = (y + h) + 'px';
  marginalBottomCanvas.width = Math.round(w);
  marginalBottomCanvas.height = MARGINAL_THICKNESS;
  marginalBottomCanvas.style.width = w + 'px';
  marginalBottomCanvas.style.height = MARGINAL_THICKNESS + 'px';
  const bc = marginalBottomCtx;
  bc.clearRect(0, 0, marginalBottomCanvas.width, marginalBottomCanvas.height);
  {
    const n = m.colMag.length;
    let maxMag = 0;
    for (let i = 0; i < n; i++) if (m.colMag[i] > maxMag) maxMag = m.colMag[i];
    bc.lineWidth = 1;
    let prevPx = 0, prevPy = 0;
    for (let i = 0; i < n; i++) {
      const px = (i / n) * marginalBottomCanvas.width;
      const py = maxMag > 0 ? (m.colMag[i] / maxMag) * (MARGINAL_THICKNESS - 4) : 0;
      if (i > 0) {
        bc.strokeStyle = marginalHueColor(m.colSum[i], m.colSumCy[i]);
        bc.beginPath(); bc.moveTo(prevPx, prevPy); bc.lineTo(px, py); bc.stroke();
      }
      prevPx = px; prevPy = py;
    }
    if (m.colPeriod) {
      bc.strokeStyle = 'rgba(255,80,80,0.6)';
      for (let px = m.colPhase; px < n; px += m.colPeriod) {
        const xx = (px / n) * marginalBottomCanvas.width;
        bc.beginPath(); bc.moveTo(xx, 0); bc.lineTo(xx, MARGINAL_THICKNESS); bc.stroke();
      }
    }
  }

  updatePositionReadoutText(camera);
}

export function updatePositionReadoutText(camera: Camera) {
  if (!positionReadout) return;
  if (!camera.lastMarginals) { positionReadout.textContent = 'not yet computed (switch to Projected Cam or capture now)'; return; }
  const m = camera.lastMarginals;
  const uStep = m.colPeriod && camera.lastProjectedBins ? m.colPeriod * camera.lastProjectedBins.binWidthU : null;
  const vStep = m.rowPeriod && camera.lastProjectedBins ? m.rowPeriod * camera.lastProjectedBins.binWidthV : null;
  const periodicityLines =
    `col period: ${m.colPeriod ?? '—'} bins (phase ${m.colPhase.toFixed(1)})\n` +
    `row period: ${m.rowPeriod ?? '—'} bins (phase ${m.rowPhase.toFixed(1)})\n` +
    `implied grid step: U=${uStep?.toFixed(3) ?? '—'}  V=${vStep?.toFixed(3) ?? '—'}\n` +
    `(expect both ≈ ${GRID_STEP})`;
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
  positionReadout.textContent = `${periodicityLines}\n\n${decodeLines}`;
}

export function hideMarginalLines() {
  marginalRightCanvas.style.display = 'none';
  marginalBottomCanvas.style.display = 'none';
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
