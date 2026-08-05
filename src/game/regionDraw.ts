import * as THREE from 'three';
import { camera, canvas, scene } from './scene.ts';
import { mode } from './mode.ts';
import { BOARD_SIZE, COLOR_TEAM_RED } from './constants.ts';
import { CHARACTER_HALF_WIDTH } from './character.ts';
import { polygonCentroid } from './polygon.ts';
import { createRibbon } from './ribbon.ts';
import { you } from './denizens.ts';

// Draw a closed region on the floor with the mouse; your denizen walks to its
// center, and the region stays painted on the floor until he arrives. Only
// active while mode === 'path'.

/** Cap on stored path points. A fast drag fires a pointermove per frame, so
 *  even a long scribble lands well under this -- it exists so the preallocated
 *  buffer can never overflow, not as a real limit on drawing. */
const MAX_POINTS = 4096;
/** Minimum spacing between stored points, world units. Raw pointermove events
 *  bunch up when the cursor slows, and duplicate/near-duplicate vertices add
 *  nothing to the shape while skewing anything that averages vertices. */
const MIN_SPACING = 0.06;
/** Path height above the floor -- enough to clear the grid lines (0.002). */
const DRAW_Y = 0.02;
/** Outline thickness in WORLD units (half of it, to each side of the path).
 *  0.035 makes a stripe 7% of a board cell wide -- reads as drawn-on paint
 *  rather than a hairline, and holds up when the camera pulls back. */
const REGION_HALF_WIDTH = 0.035;

const raycaster = new THREE.Raycaster();
const pointer = new THREE.Vector2();
const groundPlane = new THREE.Plane(new THREE.Vector3(0, 1, 0), 0);
const hit = new THREE.Vector3();

// The floor is raycast as an infinite mathematical plane rather than against
// the floor Mesh: a drag that strays past the board's edge still yields a
// point instead of a gap in the path, and it costs no triangle tests.
function pointerToGround(e: PointerEvent, out: THREE.Vector2): boolean {
  pointer.x = (e.clientX / innerWidth) * 2 - 1;
  pointer.y = -(e.clientY / innerHeight) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  if (!raycaster.ray.intersectPlane(groundPlane, hit)) return false; // ray parallel to the floor
  out.set(hit.x, hit.z);
  return true;
}

// ── The region's visual ────────────────────────────────────────────────────

const ribbon = createRibbon(MAX_POINTS, COLOR_TEAM_RED, DRAW_Y, REGION_HALF_WIDTH);
scene.add(ribbon.mesh);

const path: THREE.Vector2[] = [];
let drawing = false;
const scratch = new THREE.Vector2();

function pushPoint(p: THREE.Vector2) {
  const last = path[path.length - 1];
  if (last && last.distanceTo(p) < MIN_SPACING) return;
  if (path.length >= MAX_POINTS) return;
  path.push(p.clone());
  // Open while the stroke is in progress: the segment that closes the loop
  // stays hidden until release, so the outline follows the cursor like a pen
  // stroke instead of a rubber band snapping back to the start.
  ribbon.update(path, false);
}

/** Wipe the region entirely -- both the stroke being drawn and any committed
 *  one still sitting on the floor. */
export function clearRegion() {
  path.length = 0;
  ribbon.hide();
  drawing = false;
}

// ── Pointer wiring ─────────────────────────────────────────────────────────

/** How far the controlled denizen's center can get from the board's center
 *  before its edge would overhang. Depends on its rank -- a king is twice as
 *  wide as a soldier -- so it's read from `you` rather than being a constant. */
const limit = () => BOARD_SIZE / 2 - CHARACTER_HALF_WIDTH * you.scale;

function commit() {
  drawing = false;
  const target = new THREE.Vector2();
  const centroid = polygonCentroid(path, target);

  if (centroid) {
    // Now that the stroke is finished, draw the closing segment: the region
    // becomes a closed shape and stays on the floor until he reaches it
    // (main.ts calls clearRegion() on arrival).
    ribbon.update(path, true);
  } else {
    // Degenerate: a click, or a scribble with no enclosed area. Treat the last
    // point as the destination -- "walk here" is the obvious reading of a
    // click, and it beats silently ignoring the input. There's no meaningful
    // region to leave behind, so drop the visual now.
    if (path.length === 0) { clearRegion(); return; }
    target.copy(path[path.length - 1]);
    clearRegion();
  }

  // Clamp into the walkable area. An unreachable destination (drawn off the
  // board) would otherwise leave the denizen grinding against the edge
  // forever, since it can never get close enough to call it arrived.
  const l = limit();
  target.x = THREE.MathUtils.clamp(target.x, -l, l);
  target.y = THREE.MathUtils.clamp(target.y, -l, l);
  you.moveTarget = target;
}

canvas.addEventListener('pointerdown', (e) => {
  if (mode !== 'path' || e.button !== 0) return;
  if (!pointerToGround(e, scratch)) return;
  // Starting a new stroke abandons the previous region, drawn or committed --
  // the newest instruction wins.
  clearRegion();
  drawing = true;
  // Capture, so a drag that leaves the window still delivers move/up here and
  // can't strand the tool mid-stroke.
  canvas.setPointerCapture(e.pointerId);
  pushPoint(scratch);
});

canvas.addEventListener('pointermove', (e) => {
  if (!drawing) return;
  // Switching mode mid-stroke hands the mouse elsewhere; the stroke it was in
  // the middle of is abandoned, not committed.
  if (mode !== 'path') { clearRegion(); return; }
  if (pointerToGround(e, scratch)) pushPoint(scratch);
});

canvas.addEventListener('pointerup', (e) => {
  if (!drawing) return;
  if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
  commit();
});

// A cancelled pointer (OS gesture, focus loss) is an abandoned stroke, not a
// destination -- drop it rather than committing a half-drawn shape.
canvas.addEventListener('pointercancel', clearRegion);
