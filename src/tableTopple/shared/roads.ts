import * as THREE from 'three';
import { scene } from './scene.ts';
import { mode, onModeChange } from './mode.ts';
import { createRibbon, type Ribbon } from './ribbon.ts';
import { pointerToGround } from './groundRay.ts';
import { buildJunctionPatch, buildRoadMesh, ROAD_HALF_WIDTH } from './roadMesh.ts';
import { sceneryHighlightMaterial, sceneryMaterial } from './blocks.ts';
import { placedLandmarks } from './landmarks.ts';
import type { RoadStateEntry } from './protocol.ts';

// ── Roads: drawn as blueprints, confirmed into permanent structures ────────
//
// Same drag-on-the-floor idiom as regionDraw.ts's `path` mode -- pointerdown
// starts one, pointermove follows the cursor, pointerup commits it -- but a
// road is a straight SEGMENT, not a freehand stroke: one drag makes one Road,
// full stop, and a network is built by chaining more drags off it rather than
// by drawing one continuous shape.
//
// Two representations share one `Road` type, distinguished by `confirmed`:
// unconfirmed is a translucent BLUEPRINT (a small ribbon, one per road, like
// path's own outline but far more of them alive at once) that can still be
// dragged onto, T-junctioned, or long-press deleted; confirmed is the
// permanent paved slab (roadMesh.ts) that a "confirm road" press turns EVERY
// current blueprint into, all at once -- there is no partial confirm. Ids are
// stable across that transition, since a connection references a roadId and
// confirming must never renumber anything one points at.
//
// ── DESKTOP-ONLY CONTROLS, NETWORKED STATE ──
//
// wireRoadBuild/confirmRoads are called only by server/main.ts, same as
// wireAim/wireRegionDraw -- a phone never calls them, so road-building is
// simply unreachable there today. snapshotRoads (desktop, after a confirm)
// and writeRoadSnapshot (phone, off the roadState broadcast) are the other
// half: both read/write the SAME `built` array and both call buildRoadMesh,
// so a road looks identical wherever it was actually placed. That symmetry is
// deliberate groundwork for later, when a phone drives its own confirms
// instead of only ever receiving them (see protocol.ts's own comment on why
// roadState is a full periodic snapshot rather than a one-shot event).
//
// ── SNAPPING ONTO A LANDMARK, NOT JUST ANOTHER ROAD ──
//
// landmarks.ts's PlacedLandmark is the other thing a road can snap onto: a
// circle (center + world-space radius) rather than a segment, so its "closest
// point" (closestOnLandmark below) is always somewhere on its OWN boundary in
// the direction of the approach, never inside it. Unlike a road-to-road
// connection this is one-directional and never symmetric -- landmarks are
// static, deterministically-seeded scenery already identical on every host,
// so nothing ever needs to look up "which roads touch landmark 3" FROM the
// landmark's own side the way removeConnectionsTo does for roads.
//
// ── HIGHLIGHTING THE CURRENT SNAP TARGET ──
//
// A blueprint's ribbon has its own private material (ribbon.ts builds one per
// call), so highlighting one is just brightening that instance directly. A
// BUILT road or a landmark is different: both share ONE sceneryMaterial
// instance (blocks.ts) across every piece of scenery in the game, so
// highlighting one has to mean swapping THAT SPECIFIC mesh's own `.material`
// to sceneryHighlightMaterial (also blocks.ts) and back -- mutating the
// shared material itself would brighten every road and landmark at once.

export interface RoadConnection {
  roadId: number;
  /** Where the OTHER road attaches, in world (x, z) -- expressed along THIS
   *  road's own line. Not only the two endpoints: a T-junction lands
   *  mid-span, and this is a real 2D point rather than a start/end flag so
   *  that case needs no special representation. */
  joinPoint: { x: number; y: number };
}

export interface LandmarkLink {
  landmarkId: number;
  /** Where this road touches the landmark's boundary, in world (x, z) -- see
   *  closestOnLandmark. Never inside the landmark's footprint. */
  joinPoint: { x: number; y: number };
}

export interface Road {
  id: number;
  start: THREE.Vector2;
  end: THREE.Vector2;
  connections: RoadConnection[];
  landmarkLinks: LandmarkLink[];
  confirmed: boolean;
}

// ── Sizing and interaction radii, all derived from ROAD_HALF_WIDTH ─────────
// (roadMesh.ts) rather than tuned separately -- widening the paved road later
// widens its snap zones to match for free instead of leaving them stale.

/** Perpendicular reach of "hovering near this road's centerline". Cut to
 *  about 1/4 of its first value (was 2.2x) -- that read as too eager to
 *  snap, especially once ROAD_HALF_WIDTH itself doubled and dragged this
 *  along with it. */
const AXIS_SNAP_RADIUS = ROAD_HALF_WIDTH * 0.55;
/** How far along the segment, from either end, still reads as "the endpoint"
 *  rather than a mid-span T-junction. Scaled down by the same ~1/4 as
 *  AXIS_SNAP_RADIUS, keeping the same ratio between the two. */
const ENDPOINT_CAPTURE_LENGTH = ROAD_HALF_WIDTH * 0.75;
/** A drag shorter than this is a click, not a road -- discarded on release
 *  rather than built as a degenerate zero-length segment. */
const MIN_ROAD_LENGTH = ROAD_HALF_WIDTH;
const LONG_PRESS_MS = 500;
/** World-unit drift allowed during a long-press before it reads as the start
 *  of a drag instead of a hold. */
const PRESS_JITTER = 0.08;

/** Height above the floor the blueprint ribbons float at -- distinct from
 *  regionDraw.ts's own DRAW_Y (0.02) so a path region and a road blueprint
 *  crossing the same point can't z-fight; both can legitimately be on screen
 *  at once (path doesn't clear itself on a mode switch, only on arrival). */
const DRAW_Y = 0.025;
/** Tall enough to clear the tallest thing roadMesh.ts ever builds (a junction
 *  plaza, its own two layers included) with room to spare -- a flat disc at a
 *  fixed height was invisible whenever it landed under a BUILT road's opaque
 *  bed/cobbles/plaza, since the marker floated below their surface and got
 *  fully occluded. Real vertical extent instead of a fixed Y is what makes it
 *  poke out and stay visible regardless of what it's hovering over. */
const HOVER_MARKER_HEIGHT = 0.24;
/** Translucent -- "a ghost of what would be built here", not paint already
 *  committed the way path's ribbon (opacity 0.9, ribbon.ts's default) is. */
const BLUEPRINT_OPACITY = 0.35;
const BLUEPRINT_COLOR = 0xc9b98a;

let nextId = 0;
const blueprints: Road[] = [];
const built: Road[] = [];
const ribbons = new Map<number, Ribbon>();
/** A confirmed road's own mesh, kept around solely so the highlight system
 *  can find and swap it -- confirmRoads/writeRoadSnapshot both populate this
 *  the instant they build one; nothing else needs to look a built road's
 *  mesh back up. */
const roadMeshes = new Map<number, THREE.Mesh>();
/** Landmarks by id, for the highlight system -- placedLandmarks itself stays
 *  the array form everywhere else (iteration order, findSnap's loop). */
const landmarksById = new Map(placedLandmarks.map((l) => [l.id, l]));

function findRoad(id: number): Road | undefined {
  return blueprints.find((r) => r.id === id) ?? built.find((r) => r.id === id);
}

// ── Hit-testing ──────────────────────────────────────────────────────────

interface SnapCandidate { kind: 'road' | 'landmark'; id: number; point: THREE.Vector2 }

const closest = new THREE.Vector2();

/** Closest point on a road's FINITE segment to `pt`, plus the perpendicular
 *  distance and how far along the segment it landed (0..length). */
function closestOnSegment(road: Road, pt: THREE.Vector2): { dist: number; t: number; length: number } {
  const dx = road.end.x - road.start.x, dz = road.end.y - road.start.y;
  const length = Math.hypot(dx, dz);
  if (length < 1e-6) {
    closest.copy(road.start);
    return { dist: pt.distanceTo(road.start), t: 0, length: 0 };
  }
  const ux = dx / length, uz = dz / length;
  const raw = (pt.x - road.start.x) * ux + (pt.y - road.start.y) * uz;
  const t = THREE.MathUtils.clamp(raw, 0, length);
  closest.set(road.start.x + ux * t, road.start.y + uz * t);
  return { dist: pt.distanceTo(closest), t, length };
}

/** Closest point on a landmark's own BOUNDARY to `pt` (never inside it),
 *  plus how far `pt` is from that boundary -- 0 once `pt` is already inside
 *  the footprint, same as a road segment reading 0 once you're right on top
 *  of its centerline. */
function closestOnLandmark(landmark: { x: number; z: number; radius: number }, pt: THREE.Vector2): { dist: number; point: THREE.Vector2 } {
  const dx = pt.x - landmark.x, dz = pt.y - landmark.z;
  const centerDist = Math.hypot(dx, dz);
  if (centerDist < 1e-6) {
    // Degenerate: pt sits exactly on the landmark's own center. Any boundary
    // point is as good as another.
    return { dist: 0, point: new THREE.Vector2(landmark.x, landmark.z + landmark.radius) };
  }
  const ux = dx / centerDist, uz = dz / centerDist;
  return {
    dist: Math.max(0, centerDist - landmark.radius),
    point: new THREE.Vector2(landmark.x + ux * landmark.radius, landmark.z + uz * landmark.radius),
  };
}

/**
 * The best thing to connect to from `pt` -- a road (blueprint or already
 * built) or a landmark -- or null if nothing is close enough. Checked against
 * ALL of them together, picking whichever is closest regardless of kind:
 * extending a paved network, or snapping onto a mine's mouth, have to feel
 * like the same gesture.
 */
function findSnap(pt: THREE.Vector2, excludeId?: number): SnapCandidate | null {
  let best: SnapCandidate | null = null;
  let bestDist = AXIS_SNAP_RADIUS;
  for (const road of blueprints.length ? blueprints.concat(built) : built) {
    if (road.id === excludeId) continue;
    const { dist, t, length } = closestOnSegment(road, pt);
    if (dist >= bestDist) continue;
    bestDist = dist;
    // Within the capture zone of an endpoint, snap exactly onto it even if
    // the raw projection landed a hair inside the segment -- an end-on
    // approach should never read as "just barely" a T-junction.
    const point = t <= ENDPOINT_CAPTURE_LENGTH ? road.start.clone()
      : t >= length - ENDPOINT_CAPTURE_LENGTH ? road.end.clone()
      : closest.clone();
    best = { kind: 'road', id: road.id, point };
  }
  for (const landmark of placedLandmarks) {
    const { dist, point } = closestOnLandmark(landmark, pt);
    if (dist >= bestDist) continue;
    bestDist = dist;
    best = { kind: 'landmark', id: landmark.id, point };
  }
  return best;
}

// ── Removing a road (long-press delete, or an abandoned in-progress drag) ──

function removeConnectionsTo(deletedId: number) {
  for (const road of blueprints.concat(built)) {
    const i = road.connections.findIndex((c) => c.roadId === deletedId);
    if (i !== -1) road.connections.splice(i, 1);
  }
}

/** Removes a road entirely: itself, any OTHER road's connection entry
 *  pointing at it, and its blueprint ribbon. Only ever called on an
 *  unconfirmed road -- see this file's header on why a built one can't be
 *  deleted at all, and there is deliberately no cascade: whatever was
 *  connected to the deleted road keeps its own geometry, just minus that one
 *  connection entry. */
function removeRoad(road: Road) {
  const i = blueprints.indexOf(road);
  if (i !== -1) blueprints.splice(i, 1);
  removeConnectionsTo(road.id);
  const ribbon = ribbons.get(road.id);
  if (ribbon) {
    scene.remove(ribbon.mesh);
    ribbon.mesh.geometry.dispose();
    (ribbon.mesh.material as THREE.Material).dispose();
    ribbons.delete(road.id);
  }
}

// ── Creating and dragging ────────────────────────────────────────────────

/**
 * Records a connection from `road` to whatever `target` was hit -- either end
 * of a drag can call this (createRoad for the start, commitDrag for the far
 * end), since a road doesn't know or care which of its two points is which
 * once it's built. `target.point` is used as the joinPoint on BOTH sides
 * rather than reading it back off `road`'s own start/end: the caller has
 * already snapped that point onto `target.point` before calling this (see
 * both call sites), so the two are numerically identical anyway, and reading
 * straight off `target` is what lets this function not need to know which
 * end it was.
 */
function attachConnection(road: Road, target: SnapCandidate) {
  const joinPoint = { x: target.point.x, y: target.point.y };
  if (target.kind === 'road') {
    const other = findRoad(target.id)!;
    // Written symmetrically, both sides, the instant the connection is made --
    // not just on this new road. The network is undirected: either road has
    // to be walkable from the other, and a later long-press delete needs
    // both sides to already agree who is connected to whom (removeRoad above
    // just scrubs whichever side references the deleted id).
    road.connections.push({ roadId: other.id, joinPoint });
    other.connections.push({ roadId: road.id, joinPoint });
  } else {
    // One-directional -- see this file's header on why a landmark never
    // needs a back-reference the way another road does.
    road.landmarkLinks.push({ landmarkId: target.id, joinPoint });
  }
}

function createRoad(start: THREE.Vector2, connectedTo: SnapCandidate | null): Road {
  const road: Road = {
    id: nextId++, start: start.clone(), end: start.clone(),
    connections: [], landmarkLinks: [], confirmed: false,
  };
  if (connectedTo) attachConnection(road, connectedTo);
  blueprints.push(road);
  const ribbon = createRibbon(2, BLUEPRINT_COLOR, DRAW_Y, ROAD_HALF_WIDTH, BLUEPRINT_OPACITY);
  ribbon.mesh.visible = mode === 'road';
  scene.add(ribbon.mesh);
  ribbons.set(road.id, ribbon);
  updateRibbon(road);
  return road;
}

function updateRibbon(road: Road) {
  ribbons.get(road.id)?.update([road.start, road.end], false);
}

let dragRoad: Road | null = null;
let pressTimer: number | undefined;
const pressStart = new THREE.Vector2();
const scratch = new THREE.Vector2();

function clearPressTimer() {
  clearTimeout(pressTimer);
  pressTimer = undefined;
}

/** Abandons whatever drag is in progress, if any -- it was never released, so
 *  it never became a real blueprint (see commitDrag). Used both when the
 *  mode changes mid-drag and when a long-press delete fires on top of what
 *  was ALSO the start of a new drag. */
function abandonDrag() {
  if (!dragRoad) return;
  removeRoad(dragRoad);
  dragRoad = null;
}

function commitDrag() {
  const road = dragRoad!;
  dragRoad = null;
  if (road.start.distanceTo(road.end) < MIN_ROAD_LENGTH) { removeRoad(road); return; }

  // The far end can ALSO land on something, not just wherever the drag
  // started -- excluded from matching itself (`road` is already sitting in
  // `blueprints` by this point), so dragging back near your own start can't
  // read as "connected to itself".
  const hit = findSnap(road.end, road.id);
  if (hit) {
    road.end.copy(hit.point);
    attachConnection(road, hit);
  }
  updateRibbon(road);
}

// ── Hover indicator: the snap-point circle ──────────────────────────────

const hoverIndicator = new THREE.Mesh(
  new THREE.CylinderGeometry(ROAD_HALF_WIDTH * 0.4, ROAD_HALF_WIDTH * 0.4, HOVER_MARKER_HEIGHT, 20),
  new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.85, depthWrite: false, side: THREE.DoubleSide,
  }),
);
hoverIndicator.renderOrder = 2;
hoverIndicator.visible = false;
scene.add(hoverIndicator);

function updateHover(pt: THREE.Vector2) {
  const hit = findSnap(pt);
  setHighlight(hit);
  if (!hit) { hoverIndicator.visible = false; return; }
  // Base at the floor, not centered on it -- CylinderGeometry is centered on
  // its own origin, so this is what plants its foot at y=0 and lets its
  // height carry the top up past a built road's surface.
  hoverIndicator.position.set(hit.point.x, HOVER_MARKER_HEIGHT / 2, hit.point.y);
  hoverIndicator.visible = true;
}

// ── Highlighting the current snap target ────────────────────────────────
// See this file's header for why a built road/landmark needs a material
// SWAP rather than an in-place edit, and a blueprint doesn't.

let highlighted: { kind: 'road' | 'landmark'; id: number } | null = null;

function paintTarget(kind: 'road' | 'landmark', id: number, on: boolean) {
  if (kind === 'landmark') {
    const landmark = landmarksById.get(id);
    if (landmark) landmark.mesh.material = on ? sceneryHighlightMaterial : sceneryMaterial;
    return;
  }
  const ribbon = ribbons.get(id);
  if (ribbon) {
    // A private material per ribbon (ribbon.ts), so brightening this one
    // can't bleed into any other blueprint the way sceneryMaterial would.
    (ribbon.mesh.material as THREE.MeshBasicMaterial).opacity = on ? Math.min(1, BLUEPRINT_OPACITY * 2.4) : BLUEPRINT_OPACITY;
    return;
  }
  const mesh = roadMeshes.get(id);
  if (mesh) mesh.material = on ? sceneryHighlightMaterial : sceneryMaterial;
}

/** Moves the highlight to whatever findSnap just returned, un-highlighting
 *  whatever it was on before. Called on every hover update, AND once when a
 *  drag starts (see wireRoadBuild's pointerdown) -- a drag doesn't call
 *  updateHover again until it ends, so without that second call site the
 *  target you're actively connected to would go dark the instant you started
 *  dragging away from it. */
function setHighlight(next: SnapCandidate | null) {
  if (highlighted && (!next || next.kind !== highlighted.kind || next.id !== highlighted.id)) {
    paintTarget(highlighted.kind, highlighted.id, false);
  }
  if (next && (!highlighted || next.kind !== highlighted.kind || next.id !== highlighted.id)) {
    paintTarget(next.kind, next.id, true);
  }
  highlighted = next ? { kind: next.kind, id: next.id } : null;
}

// ── Confirming ────────────────────────────────────────────────────────────

/**
 * Junction plazas already built, keyed by a rounded (x, y) so the SAME
 * physical point read off two different roads' connection entries (see
 * createRoad's own comment on why both sides get an identical joinPoint)
 * collapses to one lookup instead of one plaza per road that touches it --
 * every road at a 3-way junction would otherwise stack three coincident,
 * z-fighting copies on top of each other.
 */
const patchedJunctions = new Set<string>();
const JUNCTION_KEY_EPS = 1e-3;

function junctionKey(p: { x: number; y: number }): string {
  return `${Math.round(p.x / JUNCTION_KEY_EPS)},${Math.round(p.y / JUNCTION_KEY_EPS)}`;
}

/** Builds a junction plaza at every connection a just-processed road carries
 *  that hasn't already got one -- called once per road, the instant it goes
 *  permanent (confirmRoads below) or arrives over the network
 *  (writeRoadSnapshot), so a plaza appears exactly once regardless of which
 *  of a junction's roads happened to be processed first. */
function patchJunctionsFor(road: Road) {
  for (const c of road.connections) {
    const key = junctionKey(c.joinPoint);
    if (patchedJunctions.has(key)) continue;
    patchedJunctions.add(key);
    scene.add(buildJunctionPatch(c.joinPoint));
  }
}

/** Turns EVERY current blueprint into a permanent paved road, connected or
 *  not -- one press, no partial confirm (see this file's header). Ids and
 *  connections carry straight across; only the visual changes, from a ribbon
 *  to roadMesh.ts's slab. */
function confirmRoads() {
  if (blueprints.length === 0) return;
  for (const road of blueprints) {
    road.confirmed = true;
    const ribbon = ribbons.get(road.id);
    if (ribbon) {
      scene.remove(ribbon.mesh);
      ribbon.mesh.geometry.dispose();
      (ribbon.mesh.material as THREE.Material).dispose();
      ribbons.delete(road.id);
    }
    const mesh = buildRoadMesh(road.start, road.end, road.id);
    scene.add(mesh);
    roadMeshes.set(road.id, mesh);
    patchJunctionsFor(road);
    built.push(road);
  }
  blueprints.length = 0;
}

// ── Wiring ───────────────────────────────────────────────────────────────

/**
 * Starts listening for road drags/hovers/long-presses on the game's own
 * canvas, and wires the "confirm road" button. Desktop-only -- see this
 * file's header. Never wired means `road` mode is simply unreachable, the
 * same consequence wireRegionDraw's own comment describes for `path`.
 */
export function wireRoadBuild(canvas: HTMLElement) {
  // Resolved here rather than at module scope, same guarded-optional idiom
  // ai.ts's wireBattleButton uses -- a host that never calls wireRoadBuild
  // (a phone) never even looks for #confirmRoadBtn, rather than every host
  // paying for a DOM query at import time for a button only the desktop has.
  const confirmBtn = document.getElementById('confirmRoadBtn') as HTMLButtonElement | null;

  canvas.addEventListener('pointerdown', (e) => {
    if (mode !== 'road' || e.button !== 0) return;
    if (!pointerToGround(e, scratch)) return;

    const hit = findSnap(scratch);
    // A press ON an existing BLUEPRINT (never a built road, and never a
    // landmark -- neither can be deleted) starts a long-press timer instead
    // of committing to a drag; movement past PRESS_JITTER before it fires
    // cancels the timer and falls through to an ordinary connected drag.
    const pressed = hit?.kind === 'road' && blueprints.some((r) => r.id === hit.id) ? findRoad(hit.id)! : null;
    if (pressed) {
      pressStart.copy(scratch);
      pressTimer = window.setTimeout(() => {
        // The provisional road createRoad() is about to start below would
        // otherwise be left as a stray zero-length blueprint once the delete
        // also fires -- abandon it first.
        abandonDrag();
        removeRoad(pressed);
        pressTimer = undefined;
      }, LONG_PRESS_MS);
    }

    // Kept lit for the whole drag -- see setHighlight's own comment on why
    // this can't just wait for the next hover update.
    setHighlight(hit);
    dragRoad = createRoad(hit ? hit.point : scratch, hit);
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!pointerToGround(e, scratch)) return;

    if (pressTimer !== undefined && scratch.distanceTo(pressStart) > PRESS_JITTER) clearPressTimer();

    if (dragRoad) {
      if (mode !== 'road') { abandonDrag(); return; }
      dragRoad.end.copy(scratch);
      updateRibbon(dragRoad);
      // Live preview of where the far end would land if released right now --
      // the circle only, not the highlight. The highlight stays reserved for
      // the drag's fixed start anchor (set once in pointerdown) so the two
      // ends aren't fighting over the one "currently highlighted" slot.
      const endHit = findSnap(scratch, dragRoad.id);
      if (endHit) hoverIndicator.position.set(endHit.point.x, HOVER_MARKER_HEIGHT / 2, endHit.point.y);
      hoverIndicator.visible = !!endHit;
      return;
    }

    if (mode === 'road') updateHover(scratch);
    else hoverIndicator.visible = false;
  });

  canvas.addEventListener('pointerup', (e) => {
    if (canvas.hasPointerCapture(e.pointerId)) canvas.releasePointerCapture(e.pointerId);
    clearPressTimer();
    if (dragRoad) commitDrag();
  });

  // A cancelled pointer (OS gesture, focus loss) abandons the drag, same as
  // regionDraw.ts treats one -- not a road, since it was never released.
  canvas.addEventListener('pointercancel', () => { clearPressTimer(); abandonDrag(); });

  onModeChange((next, prev) => {
    const show = next === 'road';
    for (const ribbon of ribbons.values()) ribbon.mesh.visible = show;
    if (!show) { hoverIndicator.visible = false; setHighlight(null); }
    if (prev === 'road') abandonDrag(); // an in-progress drag doesn't survive leaving the mode
    if (confirmBtn) confirmBtn.style.display = show ? 'block' : 'none';
  });

  confirmBtn?.addEventListener('click', () => { confirmRoads(); confirmBtn.blur(); });
}

// ── Network: the desktop's half ─────────────────────────────────────────

/** Every CONFIRMED road, projected for the wire -- blueprints never leave
 *  the desktop (see this file's header). Called every tick from
 *  server/main.ts's animate(), same as sim.ts's snapshotDenizens. */
export function snapshotRoads(): RoadStateEntry[] {
  return built.map((r) => ({
    id: r.id,
    start: { x: r.start.x, y: r.start.y },
    end: { x: r.end.x, y: r.end.y },
    connections: r.connections.map((c) => ({ roadId: c.roadId, joinPoint: c.joinPoint })),
    landmarkLinks: r.landmarkLinks.map((l) => ({ landmarkId: l.landmarkId, joinPoint: l.joinPoint })),
  }));
}

// ── Network: a receiving client's half ──────────────────────────────────

const knownBuiltIds = new Set<number>();

/**
 * Applies a `roadState` snapshot -- what client/main.ts's NetHost calls on
 * every message. Append-only: a road already known never changes (confirmed
 * roads are permanent), so this only ever builds a mesh for an id it hasn't
 * seen before, unlike writeDenizenSnapshot's every-field-every-time copy.
 */
export function writeRoadSnapshot(entries: readonly RoadStateEntry[]) {
  for (const e of entries) {
    if (knownBuiltIds.has(e.id)) continue;
    knownBuiltIds.add(e.id);
    const start = new THREE.Vector2(e.start.x, e.start.y);
    const end = new THREE.Vector2(e.end.x, e.end.y);
    const road: Road = {
      id: e.id, start, end, confirmed: true,
      connections: e.connections.map((c) => ({ roadId: c.roadId, joinPoint: c.joinPoint })),
      landmarkLinks: e.landmarkLinks.map((l) => ({ landmarkId: l.landmarkId, joinPoint: l.joinPoint })),
    };
    built.push(road);
    const mesh = buildRoadMesh(start, end, e.id);
    scene.add(mesh);
    roadMeshes.set(e.id, mesh);
    patchJunctionsFor(road);
  }
}
