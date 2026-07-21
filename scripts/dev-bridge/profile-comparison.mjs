// Timed CPU-vs-GPU pipeline comparison, using saved-capture.json as a fixed
// input so both runs analyze the exact same image. Restores that capture
// onto the active physical camera, then runs runAxesReconstruction once per
// pipeline configuration with the nested-span profiler
// (src/sphereLab/profiling/profiler.ts) enabled, and prints the resulting
// flamechart plus a CPU<->GPU data-transfer breakdown for each.
//
// IMPORTANT: keep the sphere-lab.html browser tab focused/visible for the
// whole run. requestAnimationFrame is throttled hard in a backgrounded tab
// (see cli.js's header comment for a real example) -- a run that should
// take ~1-2s can instead sit for 30+ seconds, and every wall-clock number
// this script prints stops meaning anything as a CPU/GPU comparison. Each
// configuration below prints a heads-up and waits START_DELAY_MS before
// actually triggering, specifically so you have a moment to refocus the tab
// between configs.
//
// Requires the active camera to already be a physical camera (toggle "use
// real capture" in the page first) -- same precondition restore-capture.mjs
// has, and for the same reason: this reuses ingestRealCapture's own
// capture-buffer fields on that camera, which only exist on the physical
// camera type.
//
// Phase 1 (orientationLM) and Phase 3 (positionLM) are both OFF by default --
// pass --with-phase1/--with-phase3 to opt into either. Default-off isolates
// the "simplest" pipeline (votes -> plane fit -> orientation) from the two
// iterative LM refinement passes, which dominate total time (Phase 3 alone
// was ~30-35% of the full run) and would otherwise swamp everything else in
// the comparison.
//
// Usage:
//   node scripts/dev-bridge/profile-comparison.mjs [--repeat N] [--with-phase1] [--with-phase3]

import { WebSocket } from 'ws';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const PORT = 8787;
const CAPTURE_PATH = join(dirname(fileURLToPath(import.meta.url)), 'saved-capture.json');
const START_DELAY_MS = 2000;

const args = process.argv.slice(2);
const repeatIdx = args.indexOf('--repeat');
const REPEAT = repeatIdx >= 0 ? parseInt(args[repeatIdx + 1], 10) : 1;
const WITH_PHASE1 = args.includes('--with-phase1');
const WITH_PHASE3 = args.includes('--with-phase3');

const gpuLabel = 'GPU (votes+fit+decode' + (WITH_PHASE3 ? '+Phase3' : '') + ')';
const CONFIGS = [
  { name: 'CPU only', useGPUVotes: false, useGPUFit: false, useGPUDecode: false, useGPUPositionLM: false },
  { name: gpuLabel, useGPUVotes: true, useGPUFit: true, useGPUDecode: true, useGPUPositionLM: WITH_PHASE3 },
];

function sleep(ms) { return new Promise((r) => setTimeout(r, ms)); }

let ws;
function connect() {
  return new Promise((resolve, reject) => {
    ws = new WebSocket(`ws://localhost:${PORT}`);
    const timeout = setTimeout(() => reject(new Error('timeout connecting to dev-bridge -- is server.js running?')), 8000);
    ws.on('open', () => { clearTimeout(timeout); ws.send(JSON.stringify({ role: 'controller' })); resolve(); });
    ws.on('error', (e) => { clearTimeout(timeout); reject(e); });
  });
}

function evalCode(code, timeoutMs = 15000) {
  return new Promise((resolve, reject) => {
    const id = Math.random().toString(36).slice(2);
    const timeout = setTimeout(() => reject(new Error(`eval timed out: ${code.slice(0, 80)}...`)), timeoutMs);
    const onMessage = (raw) => {
      const msg = JSON.parse(raw.toString());
      if (msg.id !== id) return;
      clearTimeout(timeout);
      ws.off('message', onMessage);
      if (!msg.ok) reject(new Error(msg.error));
      else resolve(msg.value);
    };
    ws.on('message', onMessage);
    ws.send(JSON.stringify({ type: 'eval', id, code }));
  });
}

// See cli.js's header for why this polling pattern (not a fixed sleep) is
// the only reliable way to know a capture has actually finished.
async function waitUntilIdle(pollMs = 250, maxWaitMs = 60000) {
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    const capturing = await evalCode('activeCamera() ? activeCamera().axesCapturing : false');
    if (!capturing) return;
    await sleep(pollMs);
  }
  throw new Error('timed out waiting for axesCapturing to clear -- is the browser tab focused? (backgrounded tabs throttle rAF hard, see cli.js header)');
}

function restoreCaptureCode() {
  const data = JSON.parse(readFileSync(CAPTURE_PATH, 'utf8'));
  return `(function() {
    // A physical camera only ever exists once some phone (real or synthetic)
    // has connected -- this test doesn't have a real phone handy, so it
    // creates one itself the same way findOrCreatePhysicalCamera does for a
    // genuine phone connection (see devBridge/client.ts), just with a fixed
    // synthetic connectionId instead of one assigned by server.js. Reuses an
    // existing 'profile-comparison-synthetic' camera across repeated runs
    // rather than piling up a fresh tab every time this script is invoked.
    let cam = Array.from(cameras.values()).find((c) => c.type === 'physical' && c.connectionId === 'profile-comparison-synthetic');
    if (!cam) {
      cam = createPhysicalCamera(nextCameraColor(), 'profile-comparison-synthetic');
      cameras.set(cam.id, cam);
      renderCameraTabs();
    }
    setActiveCameraId(cam.id);
    // Both LM refinement toggles default to off on a freshly-created camera
    // (see camera/settings.ts) already -- these two lines just make this
    // script's own --with-phase1/--with-phase3 opt-ins explicit rather than
    // relying on that default staying off.
    cam.settings.orientationLM = ${WITH_PHASE1};
    cam.settings.positionLM = ${WITH_PHASE3};
    refreshCameraPanel();
    const w = ${data.w}, h = ${data.h};
    const binary = atob(${JSON.stringify(data.b64)});
    const bytes = new Uint8Array(binary.length);
    for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
    const gray = new Float64Array(bytes.length);
    for (let i = 0; i < bytes.length; i++) gray[i] = bytes[i];
    if (w !== cam.rtSize.w || h !== cam.rtSize.h) resizeCaptureBuffers(cam, { w, h });
    cam.lastRealCaptureGray = gray;
    cam.lastRealCaptureW = w; cam.lastRealCaptureH = h;
    updateDistortedPreview(cam);
    return { ok: true, w, h };
  })()`;
}

function triggerCode(config) {
  return `(function() {
    const cam = activeCamera();
    globalState.useGPUVotes = ${config.useGPUVotes};
    globalState.useGPUFit = ${config.useGPUFit};
    globalState.useGPUDecode = ${config.useGPUDecode};
    globalState.useGPUPositionLM = ${config.useGPUPositionLM};
    profilerReset();
    profilerSetEnabled(true);
    runAxesReconstruction(cam);
    return { ok: true };
  })()`;
}

const RESULTS_CODE = `(function() {
  const flame = formatFlamechart();
  const tree = getFlamechartJSON();
  profilerSetEnabled(false);
  return { flame, tree, readout: axesReadout.textContent };
})()`;

// ── Tree analysis ───────────────────────────────────────────────────────
//
// Walks the returned span tree to split total wall-clock time into three
// buckets: CPU<->GPU data transfer (spans named by device.ts's upload/
// readback helpers), actual GPU kernel execution (timestamp-query spans,
// only present if the device/browser supports 'timestamp-query'), and
// everything else (CPU compute, JS glue, driver/queue submission overhead).

function walk(spans, visit) {
  for (const s of spans) { visit(s); walk(s.children, visit); }
}

function summarize(tree) {
  let transferMs = 0, kernelMs = 0;
  const root = tree[0];
  const totalMs = root ? root.end - root.start : NaN;
  walk(tree, (s) => {
    if (/CPU→GPU|GPU→CPU/.test(s.name)) transferMs += (s.end - s.start);
    if (s.kind === 'gpu') kernelMs += (s.end - s.start);
  });
  return { totalMs, transferMs, kernelMs, otherMs: totalMs - transferMs - kernelMs };
}

function pct(part, whole) { return whole > 0 ? `${((100 * part) / whole).toFixed(1)}%` : 'n/a'; }

// ── Main ──────────────────────────────────────────────────────────────────

async function main() {
  await connect();
  console.log(`[profile-comparison] Phase 1 (orientationLM): ${WITH_PHASE1 ? 'ON' : 'off'}   Phase 3 (positionLM): ${WITH_PHASE3 ? 'ON' : 'off'}`);
  console.log('[profile-comparison] restoring saved-capture.json...');
  const restore = await evalCode(restoreCaptureCode());
  if (!restore.ok) { console.error(`restore failed: ${restore.error}`); process.exit(1); }
  console.log(`[profile-comparison] restored ${restore.w}x${restore.h} capture`);

  const allResults = [];
  for (const config of CONFIGS) {
    for (let rep = 1; rep <= REPEAT; rep++) {
      const label = REPEAT > 1 ? `${config.name} (run ${rep}/${REPEAT})` : config.name;
      console.log(`\n=== ${label} -- make sure the sphere-lab.html tab is focused, starting in ${START_DELAY_MS / 1000}s ===`);
      await sleep(START_DELAY_MS);
      await evalCode(triggerCode(config));
      await waitUntilIdle();
      const result = await evalCode(RESULTS_CODE);
      console.log(`--- ${label}: flamechart ---`);
      console.log(result.flame);
      console.log(`--- ${label}: readout ---`);
      console.log(result.readout);
      const summary = summarize(result.tree);
      console.log(
        `--- ${label}: summary --- total ${summary.totalMs.toFixed(1)}ms  |  `
        + `CPU<->GPU transfer ${summary.transferMs.toFixed(1)}ms (${pct(summary.transferMs, summary.totalMs)})  |  `
        + `GPU kernels ${summary.kernelMs.toFixed(1)}ms (${pct(summary.kernelMs, summary.totalMs)})  |  `
        + `other (CPU/JS/driver) ${summary.otherMs.toFixed(1)}ms (${pct(summary.otherMs, summary.totalMs)})`,
      );
      allResults.push({ label, ...summary });
    }
  }

  console.log('\n=== Comparison ===');
  for (const r of allResults) {
    console.log(
      `${r.label.padEnd(30)} total ${r.totalMs.toFixed(1).padStart(8)}ms   `
      + `transfer ${r.transferMs.toFixed(1).padStart(7)}ms   kernels ${r.kernelMs.toFixed(1).padStart(7)}ms`,
    );
  }
  const cpu = allResults.find((r) => r.label.startsWith('CPU only'));
  const gpu = allResults.find((r) => r.label.startsWith('GPU'));
  if (cpu && gpu) {
    console.log(`\nspeedup: ${(cpu.totalMs / gpu.totalMs).toFixed(2)}x`);
    console.log(`GPU run's CPU<->GPU transfer overhead: ${pct(gpu.transferMs, gpu.totalMs)} of its total time`);
  }

  ws.close();
  process.exit(0);
}

main().catch((e) => { console.error(e.message); process.exit(1); });
