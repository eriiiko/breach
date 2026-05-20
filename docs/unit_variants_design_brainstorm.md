# Unit Variants — Design Brainstorm

> Status: **brainstorm only — not a finalized design.** Captured 2026-05-20
> after a tired late-Wednesday discussion. Revisit fresh before implementing.

## Setting principle

Anyone in this universe can become a zombie. Zombies are crewmen who turned.
The underlying body matters — a turned ogryn-class human is bigger and tougher
than a turned ordinary crewman.

**Implication:** "zombie" is **not** a separate unit subtype. It remains a
**state** flag (`is_zombie`) on the underlying unit. We do **not** label units
as "zombie" or "strong_zombie" anywhere player-visible — players should
discover the infection by playing, not by reading "Zomb1" in a tooltip.

## Unit-subtype axis (player-visible category)

Provisional set:

- **Human** — the default. Covers crew, marines, scientists, kill-team
  operators. One unit subtype handles them all; the body stats vary per
  individual (see Stats below).
- **Ogryn** *(tentative — Warhammer-style big-strong-dumb humans).* Might be
  a separate subtype if their behavior or anatomy differs enough; might just
  be a tall end of the Human stat distribution. Decide later.

Open question: do we keep subtype as a coarse string discriminator (`"human"`,
`"ogryn"`), or does subtype disappear entirely once size + stats are
expressive enough?

## Stats (per-unit, not per-subtype)

Classic D&D-style attributes plus one Breach-original:

| Stat       | Notes                                                          |
|------------|----------------------------------------------------------------|
| strength   | Affects melee damage, knockback impulse, carry capacity        |
| dexterity  | Affects fire accuracy, dodge, fine motor (door hacking?)       |
| intellect  | Affects... AI/order complexity? Hacking? Squad-leader buffs?   |
| endurance  | Stamina under sustained action, resistance to environmental    |
| vitality   | Drives HP (max HP = f(vitality, size))                         |
| **size**   | **The original axis.** Float, average human = 1.0              |

Actual derived values (`actual_strength`, `actual_hp`) are computed from base
stats **attenuated/amplified by size**.

### Size as a float

- `1.0` = average human
- `2.0` = twice as big (linear or in some chosen norm — TBD)
- `0.8` = a small individual
- `3.0` = ogryn-class

Likely scaling rules to flesh out later:
- `actual_hp ∝ vitality * size^k` (`k` ≈ 2 or 3 if we want big things to be
  meaningfully tankier; `k=1` is conservative)
- `actual_strength ∝ strength * size` (linear is probably fine)
- speed-vs-size: bigger = slower? Or independent? Open.

### Size → tile footprint (threshold rule, brainstorm)

Current units occupy a 3-tile footprint (implemented as `unit.footprint = 3`,
the default). The global `CFG.display.coarse` has been removed (coord system
cleanup 2026-05-20). All code already uses `unit.footprint` for footprint
checks — the groundwork is laid for variable-size units.
Idea:
- If `size > size_large_threshold` (e.g. `≥ 1.5`), the unit occupies 4×4
  tiles instead (`footprint = 4`).
- If `size < size_small_threshold` (e.g. `≤ 0.7`), maybe `footprint = 2`.

Concrete consequences worth listing before committing:

- Pathfinding passes `footprint` to `is_passable_block` already — just
  needs non-3 values tested.
- The renderer needs to know the unit's tile footprint to draw the right
  sprite scale.
- Hit area for bullets/grenades scales with footprint.
- Movement speed semantics: a 4×4 unit moving one tile step covers
  less of its own body than a 3×3 unit does — does that matter for
  movement feel?

## Zombification

When a unit becomes a zombie:
- `is_zombie = True` is set; subtype + body stats stay as they are.
- Behavior switches to zombie AI (already implemented).
- Stats may be amplified (zombies historically don't feel pain, push through
  damage). Sketch: `hp_multiplier`, `strength_multiplier`, `pain_immunity`
  flag — TBD.
- **Size does not change at zombification** in this model — the body is
  still the body. (An ogryn-zombie is a big zombie because the ogryn was
  big. We don't grow them after turning.)

## What about the "strong_zombie" at (36, 76)?

In the final system: a size-3 human (e.g. an ogryn crew member) standing
at that tile, who happens to be in the `is_zombie = True` state. Same Unit
class, same code path, just heftier stats and bigger footprint.

Until the variant system lands: the tile is occupied by a plain zombie.
A `# TODO` comment in `level.toml` notes the intent.

## Decisions deferred (revisit when fresh)

1. Should subtype exist at all, or is size+stats sufficient?
2. Float size + threshold-based footprint, or quantize size to small/med/large?
3. Which derived formulas (HP from vitality+size, speed from size, etc)?
4. What does each of the six stats *actually do* mechanically (esp. intellect)?
5. UI: do we surface stats to the player, or keep them implicit?
6. How does subtype/size interact with the sprite atlas pipeline? (Likely
   one base sprite per subtype, scaled by size on render.)
7. Does the level.toml spawn entry name a *preset* (e.g. "ogryn_grunt") or
   list raw stats? Presets compose better with mission design.

## Adjacent ideas worth keeping

- A pre-built **stat-preset library** keyed by role (`"crew_engineer"`,
  `"crew_security"`, `"kill_team_operator"`, `"ogryn_porter"`). Levels
  reference presets; presets resolve to base stats; size lives on the
  individual spawn entry.
- A **squad-leader buff** mechanic using intellect — explains why intellect
  matters mechanically.
- Treating dexterity as the gate for non-combat verbs (hacking, lockpicking,
  defusing) ties it to the medium-term mission design rather than just
  combat tuning.

---

When this is picked up again: review the seven deferred decisions, pick the
two or three that unlock the rest, and write a proper design doc with
locked decisions before any code lands.
