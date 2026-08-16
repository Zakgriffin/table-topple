// The denizen hierarchy. Rank is the single knob that says how big a figure
// is, how deep its color runs, and whether it wears a crown -- so adding a
// rank is one entry here, not a new branch anywhere else.
//
// Size carries the hierarchy on its own: a king is twice a soldier, and the
// two ranks between them step evenly through that range (4/3, 5/3), so a line
// of them reads as a ladder rather than as two big ones and two small ones.
export type Rank = 'king' | 'constable' | 'captain' | 'soldier';

export interface RankDef {
  label: string;
  /** Multiplier on a soldier's build. A soldier is 1 by definition. */
  scale: number;
  /** Multiplier on the team color. Higher ranks are deeper: the same red, but
   *  richer, so rank reads even in silhouette against a bright floor. Kept
   *  shallow (0.78 at the extreme) because pushing it further turns the four
   *  team colors into four muddy browns that no longer tell teams apart. */
  shade: number;
  crown?: boolean;
}

export const RANKS: Record<Rank, RankDef> = {
  king: { label: 'King', scale: 2, shade: 0.78, crown: true },
  constable: { label: 'Constable', scale: 5 / 3, shade: 0.85 },
  captain: { label: 'Captain', scale: 4 / 3, shade: 0.92 },
  soldier: { label: 'Soldier', scale: 1, shade: 1 },
};

/** Highest to lowest. */
export const RANK_ORDER: Rank[] = ['king', 'constable', 'captain', 'soldier'];
