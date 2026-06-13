# Mobility — movement as a coefficient

**Status:** reviewed (wave 1: 5 reviewers + red-team; wave 2: 2 adjudicators; synthesised with Erik
2026-06-12). Corrections folded in. NOT yet canon — lands into the chapters (§9) on implementation.
The half-built `move_cost` change has been reverted to green.

---

## 1. Why

Two forces meet on one decision:

1. **The material model wants it.** Chapter 03's thesis is "each transport system reads its own
   per-material *coefficient*, never one is-it-solid boolean" — light↔`light_atten`, gas↔`permeability`,
   heat↔`conductivity`, wave↔`wave_absorb`. The chapter names movement as the **single deliberate
   exception**: a hard boolean `passable`, "the wall hard-stop." That exception is now the only
   boolean left in an all-coefficient model.

2. **Gameplay needs the middle.** Erik's repainted ship has ~986 furniture tiles. Furniture should
   be *climbable at a penalty* (you clamber over a crate slowly), not a wall and not free. A boolean
   cannot say "passable but slow." 3×3 units stay 3×3 (the whole game is tuned on footprint 3); the
   terrain yields instead.

3. **Single source of truth.** A boolean and a speed penalty stored side by side are one fact written
   twice — they can drift, and the drift is a bug. `mobility` is the single representation; `passable`
   becomes the derived view `mobility > 0`. (It also costs nothing resident — see the honest
   accounting in §7; the *memory/training-farm* angle is real but small here and is **not** the
   headline reason, despite an earlier draft overstating it.)

4. **A richer NN channel.** A continuous mobility field is a more informative observation for the
   policy network than a walkability bit — for free, since it's the same field gameplay reads.

The resolution: **promote the last boolean into the coefficient it always wanted to be.** `passable`
becomes `mobility`, and movement joins every other system as a per-material coefficient. The chapter's
own thesis stops having an exception. The motivating win is the **feature** (point 2 — climbable
furniture, which a bool cannot express); correctness (3) and the NN channel (4) follow; the farm
economics are a footnote, not the case.

## 2. The coefficient

`mobility` — a per-material scalar, the **ease of movement through a tile** (the physics sense:
mobility = drift per unit force, the conductance-analog of friction/resistance).

```
mobility = 0.0      impassable          (the old passable=false; a wall)
mobility = 1.0      normal walking speed (air, an open door)
mobility = 0.4      40% speed           (furniture: 1/0.4 = 2.5x the step time)
mobility > 1.0      faster than walking (reserved: ice, gratings, conveyor — not used yet)
```

**Why mobility, not move_cost.** `move_cost` makes zero an *overloaded sentinel* — 0 cost should
read "free/instant," but we'd force it to mean "impossible," backwards and confusing. `mobility`'s
zero is *monotonic and honest*: less mobility = slower, 0 = stuck. It also matches the conductance
framing of `permeability`/`conductivity` (high = easy) and leaves a natural open top end (>1) for
speed boosts. Resistance is `1/mobility`, computed only where pathfinding needs a cost.

**`is_passable` becomes a derived accessor over the field:**

```python
def is_passable(tile):            # the float subsumes the boolean: passable ⟺ mobility > 0
    return mobility[tile] > 0.0
```

This is the single-source-of-truth crux: `passable ⟺ mobility > 0`, so we store the field and derive
the bool. Note this is a real behavioural change, not a transparent rename — today `is_passable`
hardcodes `material in (MAT_AIR, MAT_DOOR)` (`gamemap.py:595`) and never reads the `passable` column
at all (the column is authoring-only dead state, and a docstring at `gamemap.py:307` already *wrongly*
claims it's column-derived). The work is wiring the predicate to the field for the first time (§8).

**Occupancy is separate, and lives where it already lives — not in a new grid.** "Is a unit standing
here *right now*" is binary and dynamic, distinct from terrain. Unit-vs-unit blocking today is the
footprint re-check path (`is_passable_block` over live unit positions), **not** an `occupied` array
(none exists). `is_passable` above is terrain-only; the caller composes it with the existing occupancy
check exactly as today. (A materialised `occupied` grid is possible future work, not part of this.)

**Decoupled from flow, exactly as `passable` was.** A *closed door* is `permeability = 0` (stops
smoke now) yet `mobility = 1` (a unit walks through; it opens). Mobility replaces the boolean without
touching that decoupling — the door chapter's invariant is preserved verbatim.

## 3. Determinism — integer arithmetic, fixed-point milli-units

**Represent `mobility` as fixed-point integer milli-units** (air `1000`, furniture `400`,
impassable `0`). The editor and NN see a 0–1+ float via `/1000`; the *runtime* never touches the
float. The cadence cost is one integer expression — `base_ticks × n × 1000 / sum(mobility_milli)`
over the footprint (`n` = footprint tile count) — computed with **pure integer arithmetic** and a
**single documented rounding rule**: half-up as `(num + den//2) // den`, which never calls `round()`
on a float. Integer floor-division is bit-identical across architectures by language semantics, so
this is cross-machine-deterministic with no float-ULP edge at all.

> **Note on the precedent (corrected after review).** The temperature solver also bakes from a
> continuous material value, but it constrains to **power-of-two shifts** and *rejects* non-power-of-two
> `thermal_mass` at load (`materials.py:124-132`); its runtime is `x >> shift`, not a division. So
> mobility's *general* integer division is **not** licensed by analogy to the thermal shift table — it
> stands on its own (integer floor-division is deterministic regardless). Also: the existing thermal
> bake uses `int(round(...))`, which in Python 3 is **banker's** rounding — the very rule to avoid
> here; the mobility cost must use the explicit integer half-up above, never `round()`.

The footprint reduction is computed **live, once per executed tile-step** (A* is speed-blind, §5, so
it is never in a search loop) — trivially cheap; no pre-baked per-tile cost field is needed. A* itself
is untouched and keeps its existing float `g`/heuristic — that float is **pre-existing**, shared with
the whole engine's "Level-1, same-machine deterministic" tier (atmosphere/smoke/water are float too),
and is deferred to the engine-wide fixed-point sweep. **This PR adds no new float to any path.**

`mobility > 1` (speed boosts) works but quantises coarsely (the `max(1, …)` floor caps fast tiles at
1 tick/tile, and any `mobility ≥ base_ticks×1000` collapses to 1 tick — so the `>1` range is a near
no-op until a finer fixed-point cost unit exists). Not used by any current material; reserved.

## 4. Footprint reduction — two axes, do not conflate them

A footprint covers many tiles of mixed mobility. **Enterability** and **speed** are separate
questions and reduce differently.

- **Enterable** iff *every* footprint tile has `mobility > 0` — this is geometry, not a design knob:
  a tile at `mobility 0` is a wall, and a unit cannot overlap a wall, so any single 0 blocks the
  move. (Today's `is_passable_block`, unchanged.) The "best tile wins" intuition must **not** reach
  enterability or units clip their corners into walls.

- **Speed** = the **area-weighted average** mobility over the footprint (given it fits). A single
  obstacle is then a *fraction* of the body, so a larger unit is **not penalised** for clipping one
  crate-corner, while a unit genuinely buried in clutter is still slowed. Computed integer-safe (§3):
  `avg_milli = sum(mobility_milli over footprint) / n` (`n` = the footprint tile count, a constant),
  then the tick-cost from `avg_milli`.

  *Rejected alternatives:* **min-mobility / max-cost** ("as slow as the worst tile") inverts the size
  incentive — a bigger footprint clips more tiles, so it would move *slower* in mixed terrain purely
  for being bigger; backwards for large units (Erik, 2026-06-12). **max-mobility** ("fastest tile it
  touches") is too generous — a unit deep in furniture would sprint on one open corner, erasing the
  penalty. The area-average sits between them and is monotone **in the fraction of the body on bad
  terrain** (the honest property — it is *not* globally monotone: a faraway terrain edit that opens a
  new enterable placement can change which tiles a unit straddles, by design).

  *Known and accepted:* the average **dilutes** a single obstacle by body area — an 8-air + 1-furniture
  footprint moves near full speed. That is the intended "size is not a movement liability" feature, not
  a bug. A creature that *should* crawl through clutter does not change this default — it ships a
  different speed function (§4.1): **chokepoint-crawl is a per-race policy slot** (a min or
  worst-tile-floored function), never a property of the default.

- **Not this system: active bulldozing.** "A big/heavy unit moves through furniture *faster* than a
  small one, or crushes it" is a per-**unit** trait (mass / trample), not terrain — terrain
  `mobility` stays unit-agnostic (one value per material). If wanted, it lands in the units chapter as
  a modifier on the cost, or as furniture-destruction on entry. Flagged, deferred.

### 4.1 The footprint→speed reduction is a swappable per-race policy

The §4 reduction (footprint mobilities → effective speed) is **not hardcoded** — it is a single named
function with a fixed contract.

**It is consumed by movement CADENCE only — A\* never calls it (decided 2026-06-12).** Pathfinding
plans the **geometric shortest path** and is deliberately *speed-blind*: terrain speed must not bend a
route. The rationale is player agency — if a unit detours around slow terrain on its own, then the one
time you send it straight at an alien it avoids a puddle and gets eaten, and that reads as the unit
disobeying. With a speed-blind planner, *if* a unit crosses slow ground it is because **you** sent it
there; the consequence is yours, which is interesting instead of infuriating. (It also gives the
barricade tactic for free: dumb units take the direct path and are *slowed climbing* a crate wall
rather than cleverly routing around it.) This is the clean two-axis split — **A\* reads the binary
enterability (`mobility > 0`); the cadence reads the continuous speed.** They read different things, so
there is nothing to keep in sync.

**Evolvability + per-race variation (Erik, 2026-06-12).** Different creatures read the same terrain
differently. The contract — note it takes a baked integer policy id, **never the unit object** (a
float field on the unit, e.g. `base_speed`, must not leak in and break lockstep):

```
speed_fn(footprint_mobility_milli: int[], speed_class: int) -> tick_cost: int   # pure, integer, deterministic
```

- **default** (marines, generic) — the §4 area-weighted average;
- **heavy / ogryn** — weight toward the best tiles (bulldoze), or pair with furniture-destruction on
  entry (the §4 active-bulldoze trait);
- **chokepoint-crawler** — a min / worst-tile-floored function (the §4 accepted-dilution escape hatch);
- **small footprint** — trivially the single tile.

This lives in the **game / unit layer**: the engine owns the `mobility` *field* (a per-material terrain
coefficient — universal); how a creature *reduces its footprint to a speed* is unit policy — the same
engine/game split as the swappable ControlModes direction. **v1 ships exactly one function** (the
default average) behind the seam; the seam is load-bearing, the catalogue grows.

**Determinism contract:** every speed function is pure and integer-in / integer-out (fixed-point
milli-units, §3) — enforced by the signature (no unit object, a baked `int speed_class` quantised once
at spawn); the race→function mapping is part of the deterministic unit definition. A determinism test
asserts every registered function returns `int` and is run-to-run identical. Because A* does not call
it, the reduction runs **once per executed tile-step**, not per search node — trivial cost, no
pre-baked field needed.

**Forward-compatible inputs (Erik, 2026-06-12).** The function takes a `footprint_samples` struct, not
a bare mobility array, so it can later depend on *dynamic* fields without a breaking signature change.
The model: `mobility` is the **static per-material terrain floor**; the speed function *composes* it
with dynamic field factors —
`effective = mobility_factor × water_factor(depth) × pressure_factor(p) × …`, each a pure
deterministic piece. This unifies several mechanics into one seam instead of scattered special-cases:
the **water movement-penalty already in canon (ch.07 §5.5, "deeper water slows units")** is the first
planned extension — its home is this function. v1's struct carries `mobility` only. **Determinism
caveat:** a factor that samples a *float* field (water_depth, atmosphere are float32 today) is
same-machine-deterministic now and inherits that field's fixed-point tier for cross-machine lockstep —
the function is only as lockstep-safe as the fields it reads. Architecture permits arbitrary
complexity; player-predictability and tuning are the practical governors, per race.

**Symmetric future (flagged, not v1):** *enterability* could likewise become a per-race policy — a
fish that *requires* water, a flyer that ignores furniture — and `environment.py`'s existing
`REQUIRES_WATER` / `can_breathe_water` creature traits already hint at it. v1 keeps enterability as
universal geometry (`mobility > 0` for all) plus those existing trait checks; a per-race passability
policy is the same seam applied to the "does it fit" axis, later.

## 5. Pathfinding — speed-blind, enterability only

A* stays the existing **uniform-cost, geometric shortest-path** search. The *only* change: its
enterability gate switches from the hardcoded `material in (MAT_AIR, MAT_DOOR)` whitelist
(`gamemap.py:595,602`) to **`mobility > 0`** — so furniture (and anything else with partial mobility)
becomes pathable, while terrain *speed* is ignored by the route (§4.1: agency). No cost-aware A*, no
per-tile cost hook, no change to the float `g`/heuristic the existing search already uses (that float
is pre-existing and deferred to the engine-wide fixed-point sweep — this PR adds no new float to it).
The existing deterministic tie-break (`_DIRECTIONS` fixed order + the monotonic-counter heap key) is
unchanged and sufficient because costs are unchanged. Reference: chapter 10.

*Deferred nicety (not v1):* A* could break ties **among equal-length paths** by preferring higher
mobility — "don't trudge through mud when a same-length clear route exists." It can never lengthen a
path, so it cannot cause the avoid-the-puddle-and-die failure §4.1 guards against. Optional, later.

## 6. Material table values (chapter 03)

`passable` (bool column) → `mobility` (fixed-point integer milli-units). Initial values:

| material | mobility (milli) | note |
|----------|------------------|------|
| air | 1000 | normal |
| door | 1000 | walk-through (closed door opens; `permeability` stays 0 closed) |
| furniture | 400 | climb penalty, 2.5× step time |
| wood / hull / steel / glass | 0 | walls — impassable |

`mobility <= 0` is the impassable sentinel (defensive `<=`, mirroring `solid = permeability <= 0`).
This is a dtype change in `_SCALAR_COLUMNS` (bool→int) and a value change in every material row.

## 7. State-economy principle (lands in chapter 02)

A standing rule, stated **conditionally** so it does not contradict the engine's own correct practice
(it already projects `permeability`, `light_atten`, `face_shift`, … into materialised `(h,w)` caches
for the C++ kernels — `gamemap.py:320-391` — which is right, not a violation):

> **Store the single richest representation of each fact; choose the *view strategy* by the consumer.**
> Where a consumer already holds the source in a register, **derive** the cheap view in-register (a
> boolean from a coefficient is a free comparison — `is_passable = mobility > 0`). Where a native/CUDA
> kernel needs a contiguous per-tile buffer, **materialise** a cache *derived from* the single source
> (never a second source of truth). Never store two independent fields that encode one fact — they
> drift. And the representation is **per independent axis**: movement (`mobility`) and flow
> (`permeability`) are separate coefficients that legitimately disagree (a closed door is walkable yet
> gas-sealed) and stay separate — "one richest representation" is per-axis, not one field for all.
>
> *(The parallel-sim farm makes resident bytes matter — every field × N_sims — which is why the
> default leans "derive, don't duplicate." But for `mobility` specifically the byte saving is ~0: the
> boolean was never materialised. The case here rests on the **feature** and **single-source-of-truth**,
> not memory. Don't oversell the farm economics — they're a tiebreaker, not the argument.)*

## 8. Blast radius (enumerated in review)

- **The `passable` column has ZERO runtime readers** — it is authoring-only dead state. The real
  walkability gate is the hardcoded whitelist in `is_passable` (`gamemap.py:595`) and
  `is_passable_block` (`gamemap.py:602`): `material in (MAT_AIR, MAT_DOOR)`. **The core change is
  redirecting those two predicates to `mobility > 0`** (and fixing the stale docstring at
  `gamemap.py:307` that already claims they read the column). Callers of `is_passable_block`:
  `ai_zombie.py:135,148`, `simulation.py:311,420` — unaffected (they call through the predicate).
- **Cadence seam** gains the §4.1 `speed_fn` over the footprint samples: marines `_ticks_per_tile`
  (`simulation.py:116`), zombies `speed_ticks_per_tile` (`unit.py:188`). Today both are
  terrain-independent; `speed_fn` composes a terrain **multiplier** onto the existing order/species
  base cadence (it does not replace it).
- **A\* unchanged** except its enterability gate (above). No per-tile cost, no float touched.
- **Migration GATE (hard, pre-merge):** `unhcr_vessel_2` (the default level) has ~986 furniture tiles
  that are **walls today** (whitelist excludes id 6); at `mobility 400` they become walkable, changing
  pathing, blocking, and **spawn/reachability**. Re-verify every spawn (the `is_passable_block` spawn
  validator, `simulation.py:310`) and connectivity after the flip.
- **Tests:** the 4 `tbl.passable[...]` asserts (`test_materials.py:70-72,175`) → `tbl.mobility`;
  furniture-enterable; the cadence penalty; the `speed_fn` int/determinism contract. **No pathfinding
  tests exist** — cost is unchanged so routes don't change, but add a furniture-enterable + penalty
  test. (The earlier "3 red tests" note referred to the now-reverted `move_cost` work; baseline is
  green.)
- NN observation tensor (when it exists) feeds `mobility` — richer, single channel.
- Editor: `mobility` rides the material id, so painting already sets it — no editor work; a mobility
  tint overlay + a sealed-by-0-mobility reachability check are deferred niceties.

## 9. Where it lands (canon, on implementation)

- **03 material** — `passable` column → `mobility` (int milli); rewrite the "movement is the one
  boolean exception" framing — movement is now a coefficient like the rest; no exception remains.
- **02 state & ownership** — the §7 conditional state-economy principle (derive vs materialise by
  consumer; per independent axis).
- **01 grid** — the `is_passable` section: a terrain accessor `mobility > 0`, composed by the caller
  with the existing (non-grid) occupancy check; not a stored boolean grid.
- **10 pathfinding** — A* is **speed-blind** (geometric shortest path); only its enterability gate
  reads `mobility > 0`. Equal-length-tiebreak-by-mobility is a documented deferred option.
- **mechanics/01 units** — owns the footprint→speed function seam (§4.1) and the default (area-average)
  function; **cadence** calls it (A* does not). The engine stays terrain-only.

## 10. Review outcomes (wave 1 + 2, resolved)

1. **Determinism** — fixed-point milli-units + integer half-up `(num+den//2)//den` (no `round()`);
   integer floor-division is platform-stable, so no ULP edge. A* float is pre-existing, deferred to
   the fixed-point sweep — this PR adds no new float. (§3 corrected: the thermal precedent is *shifts*,
   not general division, and the worked example used the wrong tick base.)
2. **Footprint reduction = area-weighted average**, kept as the default (Erik); dilution is the
   intended "size ≠ liability" feature; chokepoint-crawl is a per-race `speed_fn` slot (§4, §4.1).
3. **Name** = `mobility` (final).
4. **No current consumer needs a materialised boolean** — `is_passable` is already a predicate, not a
   cache (the column is dead). Derived accessor is safe; §7 still allows materialised derived caches
   where a native kernel needs one.
5. **A* is speed-blind** (Erik) — routes are unchanged, so no route-regression; furniture only becomes
   *enterable*. No pathfinding tests exist; add enterable + penalty tests.
6. **Per-race functions run at cadence only** (not A*), once per tile-step — the "no shared cost field"
   perf concern is moot; no pre-baked field needed.

**Build is materially smaller than the first draft:** no cost-aware A*, no shared A*/cadence seam, no
pre-baked cost field, no A* de-floating. Net change = the material column + two predicate rewires + the
cadence `speed_fn` + the migration re-verify + tests.
