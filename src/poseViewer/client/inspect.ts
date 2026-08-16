// ── The eval scope, declared rather than inherited ───────────────────────
//
// This module exists to BE the scope a dev-bridge expression runs in. That is
// its whole job, and the reason it is a file rather than an `eval` call sitting
// wherever the websocket happens to live.
//
// ── WHY ──
//
// `eval(code)` in its direct form sees the scope of the module it is
// physically written in -- so whatever file holds the call silently decides
// what the phone can be asked about, and the answer is "whatever that file
// happened to need to import". That is not a stable surface. It already
// degraded once without a word from the compiler: the handler's own comment
// promised `currentStream, cameraSettings, imuLastSample`, and `imuLastSample`
// stopped resolving the moment the IMU moved into its own module, because
// main.ts no longer had a reason to import it. The desktop side has drifted
// the same way -- server/devBridge/client.ts holds the eval, and feval.sh's
// cold-start recipe names `addSimulatedCamera`, `setActiveCameraId` and
// `runAxesReconstruction`, none of which that file imports.
//
// Nothing catches this. `noUnusedLocals` cannot see it (there is no unused
// local -- there is a MISSING one), no test loads this page, and the failure
// arrives as a ReferenceError in a terminal weeks later, mid-measurement.
//
// So the scope is declared here as one namespace per sibling module, and
// `inspectScope` below is a real, exported reference to each -- which both
// documents the surface and is what keeps the imports alive under
// `noUnusedLocals`. Adding a module to the page means adding it here; leaving
// it out is then a visible omission in one file rather than an invisible
// property of another file's import list.
//
// ── WHAT AN EXPRESSION LOOKS LIKE ──
//
//   scripts/dev-bridge/feval.sh --phone "camera.currentStream"
//   scripts/dev-bridge/feval.sh --phone "camera.latestFrameMeta"
//   scripts/dev-bridge/feval.sh --phone "imu.imuLastSample"
//   scripts/dev-bridge/feval.sh --phone "imu.imuStats"
//   scripts/dev-bridge/feval.sh --phone "poses.dumpRecording()"
//   scripts/dev-bridge/feval.sh --phone "relay.sendGateStatus()"
//   scripts/dev-bridge/feval.sh --phone "Object.keys(inspectScope)"
//
// The qualified form is the point: `camera.currentStream` cannot quietly start
// resolving to something else, and it says where to go read what it means.
// Unqualified globals still work for anything genuinely on globalThis --
// `dumpRecording()` among them, since poseRecords.ts keeps that one registered
// for the recipes already written down.
//
// ── A CONSEQUENCE FOR DEAD-CODE HUNTS ──
//
// `import * as camera` makes EVERY export of camera.ts reachable from an eval
// expression. So in client/, "this export has no importer" no longer means
// "this export is dead" -- `camera.probeLog`, `camera.currentStream`,
// `relay.ws` and `poseRecords.poseRing` are all exported for exactly this
// surface and nothing else. The usual rule (grep for importers, not mentions)
// still applies everywhere else; here, check whether the thing is plausibly
// something you would ASK THE PHONE before cutting it.
//
// It imports its siblings and NOTHING imports it back except main.ts, which
// hands `evalInScope` to relay.ts as a host field. So it sits at the top of
// the graph and can never be the module that is still initializing when
// someone reads it -- see main.ts's header on the temporal-dead-zone hazard.

import * as camera from './camera.ts';
import * as imu from './imu.ts';
import * as poses from './poseRecords.ts';
import * as relay from './relay.ts';

/**
 * Everything an eval expression can reach by name, and the only reason the
 * namespace imports above survive `noUnusedLocals`. Exported so
 * `Object.keys(inspectScope)` is a live answer to "what can I ask this page",
 * rather than a list in a comment that can rot.
 */
export const inspectScope = { camera, imu, poses, relay };

/**
 * Runs one dev-bridge expression. Deliberately synchronous and deliberately
 * unguarded: relay.ts catches, serializes and reports the throw, and the
 * caller's own race warning (cli.js's header) about un-awaited promises
 * applies here unchanged.
 */
export function evalInScope(code: string): unknown {
  return eval(code);
}
