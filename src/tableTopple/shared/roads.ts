import * as THREE from 'three';
import { scene } from './scene.ts';
import { mode, onModeChange } from './mode.ts';
import { pointerToGround } from './groundRay.ts';
import {
  buildJunctionPatch, buildRoadGhostMesh, buildRoadMesh, ROAD_HALF_WIDTH,
  roadGhostHighlightMaterial, roadGhostMaterial,
} from './roadMesh.ts';
import { sceneryHighlightMaterial, sceneryMaterial } from './blocks.ts';
import { placedLandmarks } from './landmarks.ts';
import {
  buildTileHighlightGeometry, clampToTileWalk, tileAt, tilesWithinReach, tileWalk, type GameTile,
} from './gameTile.ts';
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
// unconfirmed is a translucent GHOST (roadMesh.ts's buildRoadGhostMesh, one
// per road) built from the SAME dirt-bed-and-cobblestones look a confirmed
// road gets, just see-through and rebuilt live while it's still being
// dragged -- can still be dragged onto, T-junctioned, or long-press deleted;
// confirmed is the permanent, opaque paved slab (buildRoadMesh) that a
// "confirm road" press turns EVERY current ghost into, all at once -- there
// is no partial confirm. Ids are stable across that transition, since a
// connection references a roadId and confirming must never renumber
// anything one points at.
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
// ── GAME TILES: A ROAD'S MAX SPAN ──
//
// gameTile.ts's grid (act.ts's own reach and board.ts's checkerboard are
// built on it too) also caps how far one road can run: MAX_ROAD_TILES,
// clamped LIVE while dragging (see the pointermove handler's dragRoad
// branch) rather than only checked once the drag is released -- a road
// hitting this wall should feel like a reach limit, not a rejected commit.
// The cap is walked on the GRID (gameTile.ts's tileWalk: ORTHOGONAL steps
// only, true Manhattan distance, never a corner-touching diagonal jump)
// rather than measured by distance along the raw drag segment -- a
// straight-line distance clamp let a shallow, patient drag sneak a 7th tile
// in past the limit, since a segment can graze a tile's corner without
// "using up" a full tile-width of distance the way a straight crossing does.
// tileHighlight shows exactly the walked path (clampToTileWalk's own
// `tiles`), so the highlight and the clamp can never disagree about which
// tiles are actually in reach.
//
// ── HIGHLIGHTING THE CURRENT SNAP TARGET ──
//
// A ghost, a BUILT road, and a landmark all highlight the same way: swapping
// that specific mesh's own `.material` between a shared pair (a plain one
// and a brightened one) and back, never mutating either material in place --
// roadGhostMaterial/roadGhostHighlightMaterial (roadMesh.ts) for a ghost,
// sceneryMaterial/sceneryHighlightMaterial (blocks.ts) for a built road or a
// landmark. Each pair is shared by every mesh of its own kind, so mutating
// one directly would brighten all of them at once instead of just whatever
// is actually the current snap target.

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

/** How many distinct game tiles (gameTile.ts) one road may span -- see this
 *  file's own header on why it's clamped live rather than checked on
 *  release. */
const MAX_ROAD_TILES = 6;
/** Clear of board.ts's own tileGrid lines (0.006) below, and low enough to
 *  sit INSIDE a ghost road's own translucent volume without fighting it --
 *  unlike a built road's opaque bed/cobbles (HOVER_MARKER_HEIGHT's own
 *  comment below), a ghost is see-through, so a flat decal buried inside it
 *  still reads, just blended rather than occluded. */
const TILE_HIGHLIGHT_Y = 0.02;
/** Tall enough to clear the tallest thing roadMesh.ts ever builds (a junction
 *  plaza, its own two layers included) with room to spare -- a flat disc at a
 *  fixed height was invisible whenever it landed under a BUILT road's opaque
 *  bed/cobbles/plaza, since the marker floated below their surface and got
 *  fully occluded. Real vertical extent instead of a fixed Y is what makes it
 *  poke out and stay visible regardless of what it's hovering over. */
const HOVER_MARKER_HEIGHT = 0.24;

let nextId = 0;
const blueprints: Road[] = [];
const built: Road[] = [];
/** A blueprint's own ghost mesh (roadMesh.ts's buildRoadGhostMesh), kept
 *  around so a drag can rebuild it in place and the highlight system can
 *  find and swap it -- createRoad populates this the instant a blueprint
 *  exists, mirroring roadMeshes below for a confirmed one. */
const ghosts = new Map<number, THREE.Mesh>();
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

/** findSnap, but discarding a hit that would pull the road's own far end
 *  past MAX_ROAD_TILES on the guided walk from `startTile` -- a snap target
 *  sitting just outside the last reachable tile (AXIS_SNAP_RADIUS reaches a
 *  little past a tile's own edge) shouldn't be able to drag the endpoint
 *  along with it. Shared by the pointermove preview and commitDrag so the
 *  hover circle never promises a connection the release won't actually
 *  make. */
function findReachableSnap(pt: THREE.Vector2, excludeId: number, startTile: GameTile): SnapCandidate | null {
  const hit = findSnap(pt, excludeId);
  if (!hit) return null;
  return tileWalk(startTile, tileAt(hit.point.x, hit.point.y)).length <= MAX_ROAD_TILES ? hit : null;
}

// ── Removing a road (long-press delete, or an abandoned in-progress drag) ──

function removeConnectionsTo(deletedId: number) {
  for (const road of blueprints.concat(built)) {
    const i = road.connections.findIndex((c) => c.roadId === deletedId);
    if (i !== -1) road.connections.splice(i, 1);
  }
}

/** Removes a road entirely: itself, any OTHER road's connection entry
 *  pointing at it, and its ghost mesh. Only ever called on an unconfirmed
 *  road -- see this file's header on why a built one can't be deleted at
 *  all, and there is deliberately no cascade: whatever was connected to the
 *  deleted road keeps its own geometry, just minus that one connection
 *  entry. */
function removeRoad(road: Road) {
  const i = blueprints.indexOf(road);
  if (i !== -1) blueprints.splice(i, 1);
  removeConnectionsTo(road.id);
  const ghost = ghosts.get(road.id);
  if (ghost) {
    // No material to dispose -- roadGhostMaterial/roadGhostHighlightMaterial
    // are shared across every ghost, same as sceneryMaterial is for built
    // scenery, not a private instance the way ribbon.ts's own material was.
    scene.remove(ghost);
    ghost.geometry.dispose();
    ghosts.delete(road.id);
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
  // start === end at this point (a fresh drag hasn't moved yet), so this is
  // a degenerate sliver -- immediately resized by the pointermove handler's
  // own updateGhost call as soon as the drag actually moves.
  const ghost = buildRoadGhostMesh(road.start, road.end, road.id);
  ghost.visible = mode === 'road';
  scene.add(ghost);
  ghosts.set(road.id, ghost);
  return road;
}

/** Rebuilds a blueprint's ghost mesh in place for its CURRENT start/end --
 *  called on every pointermove of an active drag (roadMesh.ts's own header
 *  on buildRoadGhostMesh explains why it's safe to rebuild that often
 *  without the result visibly jittering). */
function updateGhost(road: Road) {
  const ghost = ghosts.get(road.id);
  if (!ghost) return;
  const rebuilt = buildRoadGhostMesh(road.start, road.end, road.id);
  ghost.geometry.dispose();
  ghost.geometry = rebuilt.geometry;
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
  tileHighlight.visible = false;
  reachIndicator.visible = false;
}

function commitDrag() {
  const road = dragRoad!;
  dragRoad = null;
  tileHighlight.visible = false;
  reachIndicator.visible = false;
  if (road.start.distanceTo(road.end) < MIN_ROAD_LENGTH) { removeRoad(road); return; }

  // The far end can ALSO land on something, not just wherever the drag
  // started -- excluded from matching itself (`road` is already sitting in
  // `blueprints` by this point), so dragging back near your own start can't
  // read as "connected to itself". findReachableSnap (not findSnap) so a
  // target just past the span limit can't pull the endpoint past it either.
  const hit = findReachableSnap(road.end, road.id, tileAt(road.start.x, road.start.y));
  if (hit) {
    road.end.copy(hit.point);
    attachConnection(road, hit);
  }
  updateGhost(road);
}

// ── Tile highlight: every tile the current drag's guided walk crosses ────
// Geometry rebuilt straight in WORLD space (gameTile.ts's own
// buildTileHighlightGeometry bakes absolute tile coordinates in), so the
// mesh itself sits at a fixed origin and only its geometry ever changes --
// same idiom act.ts's own occupancy highlight uses for a hovered tree or
// structure. Rebuilt on every pointermove of an active drag rather than only
// when the tile set changes: unlike a hover staying still over one target,
// a drag's end moves on nearly every one of those events anyway.
//
// Coloured to match that same act.ts hover (occupancyHighlight's 0xbfe8ff at
// 0.22 opacity), a deliberate blue rather than any of the ghost road's own
// earth tones underneath -- the ghost already reads as "the road"; this is
// the separate "you're pointing at these tiles" feedback, and act mode
// already established what that looks like.
const TILE_HIGHLIGHT_COLOR = 0xbfe8ff;

const tileHighlight = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({
    color: TILE_HIGHLIGHT_COLOR, transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide,
  }),
);
tileHighlight.rotation.x = -Math.PI / 2;
tileHighlight.position.y = TILE_HIGHLIGHT_Y;
tileHighlight.renderOrder = 1;
tileHighlight.visible = false;
scene.add(tileHighlight);

/** Rebuilds tileHighlight for the exact tiles a guided walk (gameTile.ts's
 *  tileWalk, already computed by clampToTileWalk) passed through, and shows
 *  it -- called from the pointermove handler's dragRoad branch, once per
 *  move. Takes the tiles directly rather than re-deriving them from the
 *  road's start/end, so the highlight can never disagree with the clamp
 *  that just ran against those same tiles. */
function updateTileHighlight(tiles: readonly GameTile[]) {
  tileHighlight.geometry.dispose();
  tileHighlight.geometry = buildTileHighlightGeometry(tiles);
  tileHighlight.visible = true;
}

// ── Reach indicator: every tile the drag's START could possibly reach ────
// The same question act.ts's own radiusIndicator answers for a selected
// denizen -- tilesWithinReach, a Manhattan diamond (see that function's own
// header) -- reused here for a road's own MAX_ROAD_TILES instead of a
// denizen's ACT_REACH_SECTORS, and reach - 1 rather than reach: tileWalk
// already counts the START tile as the walk's own first tile, so a road
// with MAX_ROAD_TILES=6 can travel 5 tiles OUT from it, the same "5 away,
// 6 total" the rest of this file already carries.
//
// A plain white floor UNDER tileHighlight (ROAD_REACH_Y sits below
// TILE_HIGHLIGHT_Y) -- tileHighlight is "exactly these tiles, right now",
// this is "anywhere in here, eventually", so it has to read as the
// background the other one sits on top of rather than compete with its
// blue. Built once per drag, at its start (wireRoadBuild's own pointerdown),
// not every pointermove -- unlike tileHighlight, this depends only on the
// drag's fixed start tile, which never changes for the life of one drag.
const ROAD_REACH_Y = 0.014;
const reachIndicator = new THREE.Mesh(
  new THREE.BufferGeometry(),
  new THREE.MeshBasicMaterial({
    color: 0xffffff, transparent: true, opacity: 0.14, depthWrite: false, side: THREE.DoubleSide,
  }),
);
reachIndicator.rotation.x = -Math.PI / 2;
reachIndicator.position.y = ROAD_REACH_Y;
reachIndicator.renderOrder = 1;
reachIndicator.visible = false;
scene.add(reachIndicator);

function updateReachIndicator(startTile: GameTile) {
  reachIndicator.geometry.dispose();
  reachIndicator.geometry = buildTileHighlightGeometry(tilesWithinReach(startTile, MAX_ROAD_TILES - 1));
  reachIndicator.visible = true;
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
  const ghost = ghosts.get(id);
  if (ghost) {
    ghost.material = on ? roadGhostHighlightMaterial : roadGhostMaterial;
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
 *  connections carry straight across; only the visual changes, from a
 *  translucent ghost to roadMesh.ts's own opaque slab. */
function confirmRoads() {
  if (blueprints.length === 0) return;
  for (const road of blueprints) {
    road.confirmed = true;
    const ghost = ghosts.get(road.id);
    if (ghost) {
      scene.remove(ghost);
      ghost.geometry.dispose();
      ghosts.delete(road.id);
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
    updateReachIndicator(tileAt(dragRoad.start.x, dragRoad.start.y));
    canvas.setPointerCapture(e.pointerId);
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!pointerToGround(e, scratch)) return;

    if (pressTimer !== undefined && scratch.distanceTo(pressStart) > PRESS_JITTER) clearPressTimer();

    if (dragRoad) {
      if (mode !== 'road') { abandonDrag(); return; }
      // Clamped to a guided walk of MAX_ROAD_TILES tiles from the start tile
      // (gameTile.ts's tileWalk, orthogonal steps only) BEFORE anything
      // downstream sees it -- findReachableSnap below has to test against
      // the point the road would actually end at, not wherever the cursor
      // currently is, or a snap target just past the span limit would light
      // up as reachable when it isn't.
      const startTile = tileAt(dragRoad.start.x, dragRoad.start.y);
      const { point: clampedEnd, tiles } = clampToTileWalk(dragRoad.start, scratch, MAX_ROAD_TILES);
      dragRoad.end.copy(clampedEnd);
      updateGhost(dragRoad);
      updateTileHighlight(tiles);
      // Live preview of where the far end would land if released right now --
      // the circle only, not the highlight. The highlight stays reserved for
      // the drag's fixed start anchor (set once in pointerdown) so the two
      // ends aren't fighting over the one "currently highlighted" slot.
      const endHit = findReachableSnap(clampedEnd, dragRoad.id, startTile);
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
    for (const ghost of ghosts.values()) ghost.visible = show;
    if (!show) {
      hoverIndicator.visible = false; tileHighlight.visible = false; reachIndicator.visible = false;
      setHighlight(null);
    }
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
