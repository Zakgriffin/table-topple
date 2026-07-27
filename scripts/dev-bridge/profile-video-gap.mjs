// Measures video mode's per-frame cycle breakdown live: how much time goes
// to the actual reconstruction (axesReconstruction), how much to decoding/
// preprocessing an incoming frame (ingest), and how much is the idle gap
// between the two (phone capture+encode, network there and back, mailbox
// pump delay -- see main.ts's animate loop and pipeline/capture.ts's
// ingestRealCapture for what opens/closes that span). Also pulls the
// phone's own self-reported camera-hardware frame-delivery stats, to tell
// "the round trip is slow" apart from "the phone's camera isn't producing
// new frames any faster than this in the first place" -- both are real,
// distinct explanations for a gap showing up on a flamechart.
//
// Needs: sphere-lab.html open in a focused browser tab (see below), and a
// phone already connected via mobile-capture.html WITH A FRESH PAGE LOAD --
// if mobileCapture.ts changed since the phone last loaded the page (check
// this session's edits), it's running stale JS with none of the current
// instrumentation (or possibly not even the current pipelining behavior),
// which silently produces nonsense numbers instead of an error. When in
// doubt, reload the phone page before running this. This script does NOT
// tell the phone to start streaming -- that's on you, when prompted.
//
// IMPORTANT: keep the browser tab focused/visible for the whole run, same
// as every other profile-*.mjs script here -- requestAnimationFrame is
// throttled hard in a backgrounded tab, which would make every number this
// prints meaningless as a measurement of the real pipeline.
//
// Usage:
//   node scripts/dev-bridge/profile-video-gap.mjs [--seconds N]

import { WebSocket } from 'ws';

const PORT = 8787;
const args = process.argv.slice(2);
const secIdx = args.indexOf('--seconds');
const SECONDS = secIdx >= 0 ? parseFloat(args[secIdx + 1]) : 10;

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

// eval() runs synchronously server-side and does NOT await promises (see
// this repo's other profile-*.mjs scripts for the same note) -- dynamic
// import() always returns one even for an already-loaded module, so this
// stashes the loaded modules on `window` once, then every later eval call
// can use them synchronously.
async function loadModules() {
  await evalCode(`
    (async () => {
      window.__gapMod = {
        prof: await import('/src/sphereLab/profiling/profiler.ts'),
        store: await import('/src/sphereLab/camera/store.ts'),
      };
      window.__gapModReady = true;
    })();
    'started'
  `);
  const deadline = Date.now() + 8000;
  while (Date.now() < deadline) {
    if (await evalCode('!!window.__gapModReady')) return;
    await sleep(100);
  }
  throw new Error('timed out loading profiler/store modules in the browser tab');
}

async function waitForPhysicalCamera(maxWaitMs = 15000) {
  const deadline = Date.now() + maxWaitMs;
  while (Date.now() < deadline) {
    const found = await evalCode(`
      (() => {
        const c = Array.from(window.__gapMod.store.cameras.values()).find((c) => c.type === 'physical');
        return c ? { id: c.id, captureMode: c.captureMode } : null;
      })()
    `);
    if (found) return found;
    await sleep(300);
  }
  return null;
}

function stats(values) {
  if (values.length === 0) return null;
  const sum = values.reduce((a, b) => a + b, 0);
  return { n: values.length, mean: sum / values.length, min: Math.min(...values), max: Math.max(...values) };
}
function fmtStats(s, unit = 'ms') {
  if (!s) return 'no samples';
  return `n=${s.n}  mean=${s.mean.toFixed(1)}${unit}  min=${s.min.toFixed(1)}${unit}  max=${s.max.toFixed(1)}${unit}`;
}

// Groups root-level spans (and, for ingest roots, their direct children) by
// name across the whole captured window -- not matched into strict
// alternating (idle, ingest, axesReconstruction) triples, since a
// profilerReset() landing mid-cycle or an ingest failure can leave the
// counts slightly uneven. Aggregate stats per name are robust to that.
function summarizeRoots(roots) {
  const byName = new Map();
  const childByName = new Map();
  for (const r of roots) {
    const dur = r.end - r.start;
    if (!(dur > 0)) continue; // an idleSpan orphaned by a mid-cycle reset never got an .end
    (byName.get(r.name) ?? byName.set(r.name, []).get(r.name)).push(dur);
    for (const c of r.children ?? []) {
      const cdur = c.end - c.start;
      if (!(cdur > 0)) continue;
      const key = `${r.name} > ${c.name}`;
      (childByName.get(key) ?? childByName.set(key, []).get(key)).push(cdur);
    }
  }
  return { byName, childByName };
}

async function main() {
  await connect();
  console.log(`Connected to dev-bridge on port ${PORT}.`);
  await loadModules();

  console.log('Looking for a connected physical camera...');
  const cam = await waitForPhysicalCamera();
  if (!cam) {
    throw new Error(
      "no physical camera found -- open mobile-capture.html on your phone and connect it first (a single tap/photo is enough to register it)."
    );
  }
  console.log(`Found camera ${cam.id} (captureMode: ${cam.captureMode}).`);

  await evalCode(`
    window.__gapMod.prof.profilerSetEnabled(true);
    window.__gapMod.prof.profilerReset();
    // pullMsHistory/encodeMsHistory/transitMsHistory/payloadBytesHistory
    // aren't part of the profiler's span tree, so profilerReset() alone
    // won't clear them -- without this, stale samples from earlier in the
    // session would silently contaminate this run's stats. lastFrameStats
    // is left alone -- it's a periodic snapshot (overwritten every ~2s
    // regardless), not an accumulating history, so there's nothing to reset.
    for (const c of window.__gapMod.store.cameras.values()) {
      if (c.type === 'physical') { c.pullMsHistory = []; c.encodeMsHistory = []; c.transitMsHistory = []; c.payloadBytesHistory = []; }
    }
    'ok'
  `);

  console.log(`\n>>> Switch the phone to VIDEO mode and start streaming now. Capturing for ${SECONDS}s... <<<\n`);
  await sleep(SECONDS * 1000);

  const { roots, cams } = await evalCode(`
    (() => {
      const roots = window.__gapMod.prof.getFlamechartJSON();
      const cams = Array.from(window.__gapMod.store.cameras.values())
        .filter((c) => c.type === 'physical')
        .map((c) => ({
          id: c.id, connectionId: c.connectionId, captureMode: c.captureMode,
          lastFrameStats: c.lastFrameStats,
          pullMsHistory: c.pullMsHistory,
          encodeMsHistory: c.encodeMsHistory,
          transitMsHistory: c.transitMsHistory,
          payloadBytesHistory: c.payloadBytesHistory,
        }));
      return { roots, cams };
    })()
  `);

  const { byName, childByName } = summarizeRoots(roots);
  const idle = stats(byName.get('idle (waiting for next frame)') ?? []);
  const ingest = stats(byName.get('ingest (decode+preprocess)') ?? []);
  const recon = stats(byName.get('axesReconstruction') ?? []);
  const decode = stats(childByName.get('ingest (decode+preprocess) > image decode') ?? []);
  const readback = stats(childByName.get('ingest (decode+preprocess) > pixel readback + grayscale') ?? []);

  console.log('── Per-cycle breakdown (root-level spans, whole capture window) ──');
  console.log(`idle (waiting for next frame):      ${fmtStats(idle)}`);
  console.log(`ingest (decode+preprocess), total:  ${fmtStats(ingest)}`);
  console.log(`  > image decode:                   ${fmtStats(decode)}`);
  console.log(`  > pixel readback + grayscale:      ${fmtStats(readback)}`);
  console.log(`axesReconstruction:                  ${fmtStats(recon)}`);
  if (idle && recon) {
    const cycleMean = idle.mean + (ingest?.mean ?? 0) + recon.mean;
    console.log(`\nEstimated mean full cycle: ${cycleMean.toFixed(1)}ms  (idle is ${((idle.mean / cycleMean) * 100).toFixed(1)}% of it)`);
  }

  for (const c of cams) {
    console.log(`\n── Camera ${c.id} (${c.connectionId}, mode: ${c.captureMode}) ──`);
    const pull = stats(c.pullMsHistory ?? []);
    const enc = stats(c.encodeMsHistory ?? []);
    const trans = stats(c.transitMsHistory ?? []);
    console.log(`Phone frame pull (canvas draw only):               ${fmtStats(pull)}`);
    console.log(`Phone JPEG encode (toDataURL, phone-side CPU):     ${fmtStats(enc)}`);
    console.log(`Network transit (ws.send -> desktop receipt):      ${fmtStats(trans)}`);
    const payloads = c.payloadBytesHistory ?? [];
    if (payloads.length > 0 && trans) {
      const kbStats = stats(payloads.map((b) => b / 1024));
      console.log(`Payload size: ${fmtStats(kbStats, 'KB')}`);
      // Real JPEG bytes are ~0.75x this (dataUrl.length counts the base64
      // string, ~1.33x inflation) -- close enough for an order-of-magnitude
      // throughput check, not meant to be exact. Uses transit time only
      // (not encode+transit combined) so a slow encode can't masquerade as
      // low bandwidth.
      const n = Math.min(payloads.length, c.transitMsHistory.length);
      const throughputsKBps = [];
      for (let i = 0; i < n; i++) {
        const ms = c.transitMsHistory[i];
        if (ms > 0) throughputsKBps.push((payloads[i] / 1024) / (ms / 1000));
      }
      const tp = stats(throughputsKBps);
      if (tp) console.log(`Implied throughput (transit only): ${fmtStats(tp, 'KB/s')}  -- if this looks capped/flat rather than varying with payload size, the transfer is likely bandwidth-bound, not encode-bound.`);
    }
    if (enc && trans && enc.mean > trans.mean) {
      console.log('  ! encode time exceeds transit time on average -- the phone-side JPEG encode (toDataURL), not the network, may be the bigger contributor.');
    }
    if (c.lastFrameStats) {
      const fs = c.lastFrameStats;
      // Last ~2s window only (this snapshot gets overwritten every flush,
      // not accumulated) -- a coarser, independent cross-check against the
      // per-frame histories above, not meant to line up sample-for-sample.
      console.log(`\nPhone loop accounting (most recent ~2s window, self-reported):`);
      const total = fs.loopTicks || 1;
      const pct = (n) => `${n} (${((n / total) * 100).toFixed(0)}%)`;
      console.log(`  requestAnimationFrame ticks: ${fs.loopTicks}  (~${(fs.loopTicks / 2).toFixed(0)}Hz over the window; 60Hz would be ~120)`);
      console.log(`  blocked by backpressure (bufferedAmount > 0): ${pct(fs.backpressureBlockedTicks)}`);
      console.log(`  blocked by Sphere Lab not-ready:              ${pct(fs.readinessBlockedTicks)}`);
      console.log(`  sends attempted:                              ${pct(fs.sendsAttempted)}`);
      if (fs.loopTicks < 90) {
        console.log(`  ! well under the ~120 ticks/2s a 60Hz display would give -- requestAnimationFrame itself is being starved on the phone (backgrounded tab, thermal throttling), independent of network.`);
      } else if (fs.backpressureBlockedTicks > fs.sendsAttempted * 2) {
        console.log(`  -> most skipped ticks are backpressure (network hasn't drained the last frame) -- consistent with a bandwidth/transit-bound bottleneck, not a phone-side scheduling problem.`);
      }
      if (fs.sampleCount) {
        console.log(
          `\nPhone-reported camera hardware: nominal=${fs.nominalFrameRate ?? '?'}fps` +
          `  actual avg interval=${fs.avgIntervalMs.toFixed(1)}ms  max interval=${fs.maxIntervalMs.toFixed(1)}ms  (n=${fs.sampleCount})`
        );
        const nominalIntervalMs = fs.nominalFrameRate ? 1000 / fs.nominalFrameRate : null;
        if (nominalIntervalMs && fs.avgIntervalMs > nominalIntervalMs * 1.5) {
          console.log(
            `  ! actual frame interval is notably above the nominal ${nominalIntervalMs.toFixed(1)}ms -- ` +
            `the camera hardware itself (likely auto-exposure under low light) may be a real contributor, not just the round trip.`
          );
        }
      } else {
        console.log('\nPhone-reported camera hardware: no requestVideoFrameCallback samples (unsupported on this browser, or camera not delivering frames).');
      }
    } else {
      console.log('Phone loop accounting: no frameStats received at all (old phone client -- reload the page).');
    }
  }

  if (idle && idle.mean < 5 && ingest) {
    console.log('\nIdle gap is near zero -- if a gap is still visible in Chrome DevTools, look at ingest/reconstruction sub-spans above, or the camera hardware stats.');
  }

  ws.close();
}

main().catch((e) => { console.error(e.message ?? e); process.exit(1); });
