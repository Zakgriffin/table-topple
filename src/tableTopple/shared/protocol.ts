import { Type } from '@sinclair/typebox';
import type { Static } from '@sinclair/typebox';

// ── Table Topple's wire protocol: what a "game message" is ──────────────────
//
// Three message types, matching the shapes of state that actually need to
// leave the desktop: CONTINUOUS, frequently-changing state sent as a
// periodic snapshot, fine to drop or coalesce because a stale one is
// superseded by the next (denizenState; also roadState, even though a road
// changes rarely -- see its own comment on why it still gets this shape
// rather than attackFx's); and ONE-SHOT events for something that happened
// and does not exist as a field anywhere a snapshot could recover it from
// (attackFx -- an arrow loosed, a staff blast fired, see combat.ts's
// arrows/blasts arrays). That split, not the names, is the actual boundary
// of what belongs in this union: a new message type only ever earns its
// place by being one or the other.
//
// Deliberately NOT here: connecting, declaring a role, reconnecting. Those are
// transport, not the game, and net.ts never turns them into a GameMessage --
// see its own header. Keeping that boundary is what lets this union grow
// (a denizen ability, a chat line, a match-over event) without ever touching
// net.ts or the relay.
//
// typebox, not a hand-written interface: it is already this codebase's one
// established "define the shape once, validate strictly, derive the TS type"
// tool (shared/configSchema.ts), and a message that crosses a process
// boundary deserves the same strictness a config file gets.

const strict = { additionalProperties: false } as const;

const Vec2Schema = Type.Object({ x: Type.Number(), y: Type.Number() }, strict);
const Vec3Schema = Type.Object({ x: Type.Number(), y: Type.Number(), z: Type.Number() }, strict);
const WeaponKeySchema = Type.Union([Type.Literal('sword'), Type.Literal('bow'), Type.Literal('staff')]);

// ── denizenState ─────────────────────────────────────────────────────────
//
// One entry per denizen currently alive on the authoritative host. A
// receiver treats absence from this array as "retired" -- there is no
// separate removal message, see sim.ts's applyDenizenSnapshot.
export const DenizenStateEntrySchema = Type.Object({
  id: Type.Number(),
  pos: Vec2Schema,
  facing: Type.Number(),
  weapon: Type.Union([WeaponKeySchema, Type.Null()]),
  aim: Type.Object({ pitch: Type.Number(), yaw: Type.Number() }, strict),
  charge: Type.Number(),
  swingFor: Type.Union([Type.Number(), Type.Null()]),
  hp: Type.Number(),
  barTimer: Type.Number(),
  dyingFor: Type.Union([Type.Number(), Type.Null()]),
}, strict);

export const DenizenStateMessageSchema = Type.Object({
  type: Type.Literal('denizenState'),
  denizens: Type.Array(DenizenStateEntrySchema),
}, strict);

// ── attackFx ─────────────────────────────────────────────────────────────
//
// One-shot, fired at the instant combat.ts's loose()/blast() actually let go
// -- never coalesced with the next one, unlike denizenState. Carries already-
// resolved numbers (a speed, a range, a half-angle) rather than a weapon key
// and a charge fraction, so a receiver can spawn the same visual without
// importing weapons.ts's damage/timing tables at all -- it only ever needs to
// draw what already happened, never decide what a weapon does.
const AttackFxMessageSchema = Type.Union([
  Type.Object({
    type: Type.Literal('attackFx'),
    kind: Type.Literal('arrow'),
    attackerId: Type.Number(),
    team: Type.Number(),
    origin: Vec3Schema,
    target: Vec3Schema,
    speed: Type.Number(),
  }, strict),
  Type.Object({
    type: Type.Literal('attackFx'),
    kind: Type.Literal('blast'),
    attackerId: Type.Number(),
    team: Type.Number(),
    origin: Vec2Schema,
    target: Vec2Schema,
    range: Type.Number(),
    halfAngle: Type.Number(),
  }, strict),
]);

// ── roadState ────────────────────────────────────────────────────────────
//
// CONFIRMED roads only -- a blueprint is desktop-local planning state, same
// as `path` mode's in-progress stroke never leaving the desktop either (see
// roads.ts's own header). Modelled as a periodic full snapshot rather than a
// one-shot "road built" event, even though roads only ever get ADDED and
// never change once confirmed: a one-shot-only design would mean a phone
// that connects mid-game silently never learns about anything built before
// it joined. Sent every tick alongside denizenState, same "lean for now"
// reasoning already used there -- at any board's realistic road count this
// is a non-issue, and it's what makes a late joiner catch up for free with
// no separate resync mechanism to design.
const RoadConnectionSchema = Type.Object({
  roadId: Type.Number(),
  joinPoint: Vec2Schema,
}, strict);

// A road can also snap onto a landmark (shared/landmarks.ts's PlacedLandmark)
// rather than another road -- these never need a symmetric back-reference the
// way RoadConnection does, since landmarks are static, deterministically
// seeded scenery that already exists identically on every host: nothing ever
// has to look "which roads touch landmark 3" up FROM the landmark's own side.
const LandmarkLinkSchema = Type.Object({
  landmarkId: Type.Number(),
  joinPoint: Vec2Schema,
}, strict);

export const RoadStateEntrySchema = Type.Object({
  id: Type.Number(),
  start: Vec2Schema,
  end: Vec2Schema,
  connections: Type.Array(RoadConnectionSchema),
  landmarkLinks: Type.Array(LandmarkLinkSchema),
}, strict);

export const RoadStateMessageSchema = Type.Object({
  type: Type.Literal('roadState'),
  roads: Type.Array(RoadStateEntrySchema),
}, strict);

export const GameMessageSchema = Type.Union([DenizenStateMessageSchema, AttackFxMessageSchema, RoadStateMessageSchema]);
export type GameMessage = Static<typeof GameMessageSchema>;
export type DenizenStateMessage = Static<typeof DenizenStateMessageSchema>;
export type AttackFxMessage = Static<typeof AttackFxMessageSchema>;
export type RoadStateMessage = Static<typeof RoadStateMessageSchema>;
/** One denizen's wire-safe presentation fields -- everything renderDenizen
 *  needs, nothing more. Defined here, off the schema, so sim.ts's
 *  DenizenStateEntry and the validated wire shape can never drift apart. */
export type DenizenStateEntry = Static<typeof DenizenStateEntrySchema>;
/** One road's wire-safe shape -- everything roadMesh.ts's buildRoadMesh and
 *  roads.ts's connection bookkeeping need, nothing more. */
export type RoadStateEntry = Static<typeof RoadStateEntrySchema>;
