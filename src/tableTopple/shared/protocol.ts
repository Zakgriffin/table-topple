import { Type } from '@sinclair/typebox';
import type { Static } from '@sinclair/typebox';

// ── Table Topple's wire protocol: what a "game message" is ──────────────────
//
// Exactly two message types, matching the two kinds of thing that actually
// need to leave the desktop: CONTINUOUS state (every denizen's pose, sent as
// a periodic snapshot, fine to drop or coalesce -- a stale one is superseded
// by the next) and ONE-SHOT events (an arrow loosed, a staff blast fired,
// which do not exist as a field on any Denizen and so cannot be recovered
// from a snapshot -- see combat.ts's arrows/blasts arrays). That split, not
// the two names, is the actual boundary of what belongs in this union: a
// third message type only ever earns its place by being one or the other.
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

export const GameMessageSchema = Type.Union([DenizenStateMessageSchema, AttackFxMessageSchema]);
export type GameMessage = Static<typeof GameMessageSchema>;
export type DenizenStateMessage = Static<typeof DenizenStateMessageSchema>;
export type AttackFxMessage = Static<typeof AttackFxMessageSchema>;
/** One denizen's wire-safe presentation fields -- everything renderDenizen
 *  needs, nothing more. Defined here, off the schema, so sim.ts's
 *  DenizenStateEntry and the validated wire shape can never drift apart. */
export type DenizenStateEntry = Static<typeof DenizenStateEntrySchema>;
