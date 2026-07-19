# Units

**Depends on:** [State & Ownership](../engine/02_state_and_ownership.md), [Material System](../engine/03_material_system.md).

> **Naming note (2026-07-19):** units are deliberately **NOT entities** in the
> [entity-system](../engine/16_entity_system.md) sense — no registry class, no
> signals, no `[[entity]]` block; they spawn via the permanent `[[spawn]]`
> syntax. The two systems converge at stack-2 (one shared id space, unit
> signals `hp`/`alive`/`faction`, digest merge) per the convergence contract
> in the entity design §3e. This chapter was renamed from "Units & Entities"
> when the entity system claimed the word.

A **unit** is a mobile actor in the world — a marine, a zombie, and in time any
creature, robot, or worm the game grows to need. Units are the only things in
Breach that act with intent. Everything else (walls, gas, fire, light) is field
data on the grid; units are objects that read those fields, move through them,
and stamp themselves back onto them.

This chapter defines the unit data model: what a unit *is*, how its statistics
are generated, how it occupies space, how it relates to its environment and to
other factions, and how zombification fits in. It does **not** define combat
resolution, AI behaviour, or the turn system — those are separate chapters. The
goal here is the *shape* of the entity, designed so the deferred systems attach
to it later without a rewrite.

---

## 1. One class, data-driven variation

There is exactly one `Unit` class (`src/simulation/unit.py`). Marines, zombies,
and every future creature are instances of it. Variation between kinds of unit
is **data**, not inheritance:

- A **`SpeciesDef`** is static design data — a record describing humans (and,
  later, xenos, Grays, robots, animals). A designer adds a species by adding a
  record, never by writing a subclass.
- A **`Unit`** is a live instance: one body, with a sampled stat vector, a
  position, a facing, and mutable runtime state.

This mirrors the world-state philosophy of the rest of the engine. Just as the
grid is arrays plus a material table rather than a grid of tile-objects, the
roster is one entity type plus a species table rather than a hierarchy of
hand-written subclasses. A subclass is added only if a species needs genuinely
divergent *code* — and even then a component is preferred. The default answer is
always data.

Units live in a flat list on the deterministic `Simulation` facade
(`sim.units`), not on the grid. They carry properties that do not map to a
spatial array — stats, action points, allegiance, AI state, inventory. Their
bridge to the grid is `stamp_units()` (§5), which each tick paints their
footprints into the `obstacles` array so physics, light, and pathfinding see
them as solid.

### Why one class and not a hierarchy

Marine-only fields (`orders`, `ap`) and zombie-only fields (`zombie_path`,
`last_melee_tick`) coexist on the same object. This is deliberate. Zombie-ness
is a **state** (`is_zombie`), not a type, because in this setting anyone can
turn — a zombie is a crewman who was converted, and the underlying body still
matters. Keeping marine and zombie state on one object means conversion is a
flag flip, not an object swap, and a converted unit keeps everything it was
carrying. The chokepoint that decides "should AI drive this unit" is the
`is_zombie` check inside action application, not the class.

---

## 2. Base vs. effective statistics

Every unit has a `BaseStats` vector, rolled once at creation and thereafter
**immutable**. The values gameplay actually uses are *effective* stats —
base plus the sum of all active modifiers (wounds, encumbrance, zombification,
fear, buffs). Effective stats are always read **through a function**, never
stored as the source of truth.

```python
def compute_effective_stats(unit) -> EffectiveStats: ...
def effective_vitality(unit) -> float: ...
def effective_strength(unit) -> float: ...
# ... one accessor per stat
```

This single rule pays for three things at once:

- **Zombification is a modifier, not a mutation.** A strong crewman who turns is
  still strong, automatically, because the base is never touched. The zombie
  modifier drives down *effective* will, intelligence, and so on without
  rewriting the roll.
- **Equipment is the same machinery.** A vac-suit is a modifier on the effective
  *environment* profile — same pattern as a stat buff.
- **One code path** for wounds, buffs, fear, and the future metaphysics.

Gameplay code must call the accessors, never read `BaseStats` directly, so the
modifier engine can be slotted in transparently later. **The modifier engine
itself is deferred** — for now `compute_effective_stats` returns base unchanged
and the modifier list is empty. The discipline of routing through the accessor
is what matters; it is what makes the deferred system a drop-in.

### The stat set

Seven primary stats, plus two physical attributes (`mass`, `base_speed`)
generated alongside them. The physical pair live directly on the `Unit`, not
inside `BaseStats`, because they participate in physics and movement rather than
the buff/wound machinery.

| Stat               | Player-visible | Meaning |
|--------------------|----------------|---------|
| `strength`         | yes            | Raw physical force. Absolute and comparable (9 > 8 always). Feeds derived melee / knockback / carry. |
| `agility`          | yes            | Fine motor control, dodge, reaction. |
| `endurance`        | yes            | Stamina pool — sprinting, sustained melee, held actions. |
| `vitality`         | yes            | HP pool — the maximum health value. |
| `intelligence`     | yes            | Cognition. Also selects the unit's AI tier. |
| `will_strength`    | yes (as WILL)  | Spell-interrupt resistance, morale, fear *susceptibility*. |
| `imagination`      | **hidden**     | Image-forming faculty. Always stored; rendered only if the unit is `awakened`. |
| `will_orientation` | **hidden**     | Float in `[-1, +1]`: separative ↔ universal. Never shown to the player (current design). |
| `mass`             | (see §4)       | Kilograms. Generated jointly with the stat vector so it can correlate with strength. |
| `base_speed`       | (see §6)       | Locomotion rate before modifiers. |

`endurance` (stamina) and `vitality` (health) are deliberately distinct and must
not be merged. `will_strength` and `will_orientation` are two halves of one
concept — the player sees only the magnitude; orientation is the hidden vector
half. `imagination` is hidden by *UI policy*, not by absence: the field is always
present and populated. The visibility rule lives in one helper,
`is_stat_player_visible(stat, unit)`, so the UI never hard-codes it: orientation
is never visible; imagination is visible only when `unit.awakened`; everything
else always.

The hidden stats — `imagination`, `will_orientation`, `awakened` — exist to
support a future metaphysics layer (a will-orientation model of fear and
"magic", with the Grays as a reality-distorting enemy faction). Only the **data
fields** are present today; none of that behaviour is built. They are carried now
so the unit does not need reshaping when the behaviour lands.

---

## 3. Statistic generation

Units are not hand-statted by default. Each species defines a **distribution**;
the generator samples it at spawn. Named or important units may override
specific stats with hand-set values while sampling the rest.

### Correlated sampling

Stats are **not** drawn independently — that would discard correlations like
"heavier units tend to be stronger." The whole 10-dimensional vector is drawn
from a **multivariate normal**, then each component is clamped to a hard
`[clamp_min, clamp_max]` band.

A `StatDistribution` is specified as a per-stat `mean`, a per-stat `stddev`, and
a `correlation` matrix (symmetric, unit diagonal). Covariance is composed from
them — `cov[i,j] = stddev[i] · stddev[j] · correlation[i,j]` — because a
correlation matrix plus standard deviations is far more intuitive for a designer
to author than a raw covariance matrix.

```python
def sample_unit_attributes(species, rng=None, overrides=None
                           ) -> tuple[BaseStats, float, float]:
    vec = rng.multivariate_normal(dist.mean, dist.covariance())
    vec = np.clip(vec, dist.clamp_min, dist.clamp_max)
    # overrides applied after clamping (named characters)
    # first 8 -> BaseStats; last 2 -> (mass, base_speed)
```

### Tuning methodology

Distributions are tuned in three steps, with `vitality` as the worked example:

1. **Mean encodes intent.** Set the mean from gameplay — how many typical hits a
   typical unit of this species should survive. Damage and mean vitality are
   tuned together.
2. **Variance encodes spread.** Choose `stddev` so a unit at the unusual-but-
   plausible end differs from the mean by a *meaningful but bounded* amount —
   e.g. survives exactly one more or one fewer hit. Not a wild swing.
3. **Clamp is the hard extreme.** `clamp_min` / `clamp_max` are the "this is as
   far as it goes" bounds that stop a freak draw from producing an absurd outlier.

The shipped human distribution sets `vitality` mean = 100 (reproducing the
legacy marine HP baseline) with `stddev` = 15; player-visible stats on a 1–10
scale; `mass` mean = 80 kg; `base_speed` mean = 1.0 (nominal). Correlations:
`mass`↔`strength` +0.6, `mass`↔`agility` −0.3, `mass`↔`base_speed` −0.2 —
heavier humans tend to be stronger but slightly slower and less nimble.

### Determinism

The unit constructor samples with a fresh default RNG so a `Unit` can be built
standalone in tests without booting a `Simulation`. When a unit is registered via
`Simulation.add_unit`, its stats are **re-sampled with the simulation's seeded
RNG**, overwriting the constructor draw. This is what makes spawns reproducible
when the sim is constructed with a seed — a requirement for deterministic
rollouts and the planned ML training pipeline.

---

## 4. Mass, occupancy, size, density

Four related but distinct physical quantities. All are kept.

- **`mass`** (kilograms) — a real per-unit number, generated with the stat vector
  so it can correlate with strength. Intended to drive knockback resistance and
  how hard the decompression wind can shove a unit.
- **occupancy** — the set of tiles a unit currently covers. The world scale is
  **1 tile = 1/3 metre**, so a baseline human occupying 3×3 tiles is 1 m². The
  *only* contract is the method interface (§5); how a unit stores the data behind
  it is its own business.
- **`size`** — a *derived* integer = `len(occupied_tiles())`. Computed, not
  stored. A convenient single scalar (useful as an ML feature, or for any
  mechanic that wants one number).
- **`density`** — derived = `mass / (size · tile_area_m²)`. Cheap to compute, and
  it gives physics a principled way to rank two units that share a footprint but
  differ in mass: same 3×3 tiles, different mass, different behaviour in a
  decompression wind — exactly the emergent texture Breach is built around.

**Strength stays absolute and comparable** — never scaled into the stored stat.
Physical *output* is a derived value that combines strength and mass (the design
suggestion is `strength · √(mass / 80)`), so a big strong unit out-hits a small
strong one while raw `strength` still reads cleanly on the sheet. Knockback
*resistance* scales directly with mass.

---

## 5. Spatial state and the occupancy interface

A unit's logical position is tile-discrete; movement is tick-quantised. The
authoritative position fields are float `x` / `y` in physics-tile units, so the
renderer can interpolate sub-tile motion. Integer matrix indices come from the
`tile_x` / `tile_y` properties (which floor the floats). There is a single
position source of truth — the older split `fx/fy` (int) and `fxf/fyf` (float)
fields are gone.

Which tiles a unit covers is queried **through methods on `Unit`, never by
reading its internal shape storage**:

```python
def occupied_tiles(self) -> list[tuple[int, int]]: ...
def occupies(self, tile) -> bool: ...
```

Every consumer — collision, line-of-sight, hit-detection, `stamp_units`, and the
physics/decompression projection — calls this interface and never touches the
backing data. This is what lets the storage representation differ per unit kind
without disturbing any consumer:

- **Rigid units** (the 3×3 human, any non-deforming body) hold a list of tile
  offsets relative to their anchor. `occupied_tiles()` returns
  `anchor + each offset`. The default offsets come from `SpeciesDef`; a unit may
  carry its own override.
- **Articulated units** (snakes, worms, deforming blobs) would hold per-instance
  state — a segment-chain deque whose footprint reshapes every tick — and
  override the method to walk it. *Not built; forward-looking note only.* The
  interface is designed to absorb them with zero change to consumers.

The bridge to the grid is `GameMap.stamp_units(units)`, called each tick: it
rebuilds the `obstacles` array as static walls plus living unit footprints by
iterating `unit.occupied_tiles()` for every living unit. Once stamped, units act
as walls for all physics — waves reflect off them, gas and smoke flow around
them, light is blocked — with no unit-aware code anywhere in the solvers.

Facing is a **float in radians**, math convention: `0` = East, increasing
counter-clockwise, so `π/2` = North (the default spawn facing), `π` = West,
`3π/2` = South. This matches `atan2` and makes angular arithmetic natural, and it
is the right representation for normal-map lighting and melee arcs. For sprite
selection, `facing_compass()` snaps the angle to the nearest 45° sector and
returns `"N"`, `"NE"`, … . The rigid offset list does not yet rotate with
facing; the symmetric 3×3 footprint does not need it. Rotation for asymmetric
rigid shapes is a documented future step.

---

## 6. Environment profile

Breach's premise is hull breaches and decompression, so a unit's relationship to
its surroundings is **foundational**, not a bolt-on. Each species carries an
`EnvironmentProfile` composed of orthogonal flags, so any combination is
expressible — a vacuum-proof robot, a water-only fish, an amphibian, a
vacuum-proof-but-water-fragile construct.

| Group       | Fields |
|-------------|--------|
| Respiration | `breathes`, `can_breathe_air`, `can_breathe_water`, `o2_reserve_max` |
| Pressure    | `pressure_min`, `pressure_max` |
| O₂ level    | `o2_level_min`, `o2_level_max` |
| Temperature | `temperature_min`, `temperature_max` |
| Submersion  | `submersion` ∈ {`DROWNS`, `UNAFFECTED`, `REQUIRES_WATER`} |
| Damage      | `environmental_damage_rate` (per tick outside any tolerance) |

The intended behaviour: a unit replenishes its O₂ reserve while in a medium it
can breathe whose O₂ level is in band, otherwise the reserve depletes; an empty
reserve causes asphyxiation damage; ambient pressure or temperature outside the
tolerance band causes environmental damage (the decompression-trauma and
heat-radiation channels). A non-breathing unit skips the O₂ logic entirely.
Suits and equipment extend survivability by modifying the *effective*
environment profile — the same base/effective pattern as stats.

**This is data only today.** No tick handler reads the profile; no environmental
damage, asphyxiation, or submersion logic is applied. The human profile (which is
just the dataclass defaults) is attached to every unit and waits for the
behaviour pass.

`base_speed` is generated with the stat vector (tight variance). Effective speed
is intended to derive from base speed, agility, encumbrance, wounds,
zombification, and terrain, mapping to a movement **cadence** in ticks-per-tile.
The unit-level derivation (agility/wounds/encumbrance) is not built — today the
order/species base cadence comes from config (zombies use `CFG.zombie.ticks_per_tile`;
marines override per order), and `base_speed` is stored but not yet consumed. The
**terrain** half, however, has shipped (commit a440d05) as the footprint→speed seam
described next.

### The footprint→speed function seam (SHIPPED, commit a440d05)

A unit covers a block of tiles of mixed `mobility` (the per-material terrain
coefficient, engine/03). How it reduces those tiles to an effective step time is a
**single swappable function behind a fixed contract** — the engine owns the
`mobility` *field*; how a creature reads its footprint is *unit policy*, so the
catalogue can grow (heavy/bulldoze, chokepoint-crawler) without touching the engine.
This is the same engine/game split as the swappable ControlModes direction.

```
speed_fn(footprint_samples, speed_class) -> tick_cost: int   # pure, integer, deterministic
```

- **Consumed by the movement CADENCE only — A\* never calls it.** Pathfinding is
  deliberately speed-blind (engine/10): terrain speed must not bend a route, so the
  reduction runs once per *executed* tile-step, not per search node.
- **Integer / deterministic.** Pure integer-in / integer-out, fixed-point milli-units,
  a single documented rounding rule (half-up `(num + den//2) // den`, never `round()`
  — Python 3 `round` is banker's). It takes a baked integer `speed_class` (the unit's
  order/species base cadence), **never the unit object** — a float field like
  `base_speed` must not leak in and break lockstep.
- **`mobility` is the static terrain floor the function composes.** The **v1 default**
  (`default_speed` in `src/simulation/movement.py`) is the **area-weighted average**
  mobility over the footprint: `tick_cost = half_up(base_ticks · n · 1000, Σ mobility)`.
  A single obstacle is then a *fraction* of the body — a large unit is not penalised
  for clipping one crate-corner, while a unit buried in clutter is genuinely slowed.
  (Enterability is the separate geometry axis: *every* footprint tile must have
  `mobility > 0`, engine/10 / `is_passable_block` — the "best tile wins" intuition must
  not reach it or units clip into walls.)
- **Forward-compatible inputs.** The function takes a `FootprintSamples` struct, not a
  bare array, so it can later *compose* dynamic field factors —
  `effective = mobility_factor × water_factor(depth) × …` — without a breaking
  signature change. The water movement-penalty already in canon (engine/07 §5.5,
  "deeper water slows units") is the first planned extension; its home is this
  function. v1's struct carries `mobility` only.

v1 ships exactly one function (the default average) behind the seam; the seam is
load-bearing, the catalogue grows. A determinism test asserts it returns `int` and is
run-to-run identical.

---

## 7. Inventory and carry

Inventory is a **base field on every unit**, always present. A converted zombie
keeps its grenades — and a carried grenade can still cook off when the unit walks
through fire, even though the zombie cannot issue a "use grenade" order. No
special-case code: the temperature and explosion systems handle it naturally.

The runtime `Inventory` holds `equipped` and `carried` lists and a
`current_load()`. The species-level `InventoryProfile` holds `has_inventory` and
`carry_capacity_base`. Effective carry capacity is meant to derive from effective
strength and the species baseline, with overload feeding the encumbrance term in
effective speed.

**The real item system is not built.** `Inventory` is a stub: the lists are empty
and `current_load()` returns 0. The actual ammo lives in two booleans on the
unit, `has_grenade` and `has_explosive`, which have not yet migrated into the
inventory container.

---

## 8. Faction and zombification

A unit stores only a `faction_id`. **Relationships between factions are not
stored on the unit** — they belong to a mission-level table, because they are
dynamic and per-mission: a faction may be friendly in one mission and hostile in
another, and one map may host three mutually hostile teams. The design names a
`FactionRelationshipTable` returning a `Stance`
(`ALLIED`/`FRIENDLY`/`NEUTRAL`/`HOSTILE`) between any two factions, owned by the
mission, not the unit.

**This table is not built.** `FactionId` is a plain `int` alias, and `faction_id`
is currently an alias for the legacy `team` (0 = marine/player, 1 = zombie/enemy).
Combat still asks `if a.team != b.team`. The `Stance` enum is defined but
consulted nowhere.

Zombification is a runtime `bool`, `is_zombie`. Its stat effects are *designed* to
route through the modifier layer as a zombie modifier — strength, mass, and
occupancy preserved (a big heavy unit stays big and heavy); will, orientation,
and intelligence driven down; the lowered effective intelligence rerouting the
unit to a lower AI tier. `SpeciesDef.can_become_zombie` gates eligibility. Zombies
are meant to belong to a zombie faction, hostile to all non-zombies and friendly
to itself — "zombies don't kill zombies."

**As shipped, none of the modifier-based zombie stat machinery exists.** Zombie
HP is set directly from `CFG.zombie.hp` in the AI code, which overwrites the
sampled vitality; zombie speed comes from config. Conversion (a marine killed by
a zombie turns at round end) flips `is_zombie` and `team`. The richer
size-aware variant idea — an ogryn-class crewman who turns into a big, tough
zombie because the body was big, all one `Unit` with heftier stats and a 4×4
footprint — is design intent, not code.

---

## 9. The assembled unit

| Group | Fields |
|-------|--------|
| Identity (immutable) | `id`, `species_id` |
| Static, rolled once | `base_stats`, `mass`, `base_speed`, `offsets` |
| Position / facing | `x`, `y` (float tiles), `facing` (radians) |
| Life | `life_state` (`ALIVE`/`DOWNED`/`DEAD`), `alive` (derived property) |
| Health | `current_hp` |
| Allegiance | `faction_id`, `team`, `is_zombie` |
| Environment | `environment` (species profile pointer) |
| Inventory | `inventory`, `has_grenade`, `has_explosive` |
| Hidden / metaphysics | `awakened` (gates `imagination` visibility) |
| Planning (marine) | `orders`, `ap`, `current_order_type` |
| AI (zombie) | `zombie_activated`, `zombie_path`, `last_melee_tick`, `killed_by_zombie`, … |
| Movement | `speed_ticks_per_tile`, `move_path` |

`life_state` is the authoritative life status; `alive` is a derived property
(with a setter that maps `False` → `DEAD`) so legacy `unit.alive` reads and
writes keep working. `current_hp` replaced the old `hp`/`max_hp` pair: there is no
stored max — the cap is `effective_vitality(unit)`, which the UI HP readout uses.

---

## Implementation status

Audited against the code in `src/simulation/` (`unit.py`, `stats.py`,
`species.py`, `environment.py`, `inventory.py`, `factions.py`, `generation.py`)
and its consumers.

**Built and in use:**

- **One `Unit` class**, marines and zombies as instances; `is_zombie` as state.
- **`BaseStats` + multivariate-normal generator.** `sample_unit_attributes`
  draws the correlated 10-vector, clamps it, peels off `mass`/`base_speed`, and
  supports overrides. The human `SpeciesDef` with tuned means/stddev/correlation
  ships. `add_unit` re-samples with the seeded RNG for deterministic spawns
  (note: the `unit.py` docstring claims no re-sampling — the code does re-sample;
  the code is authoritative).
- **Effective-stat indirection.** `compute_effective_stats` and per-stat
  accessors exist; `is_stat_player_visible` enforces the hidden-stat rule.
  `compute_effective_stats` returns base unchanged (no modifiers).
- **Occupancy interface.** `occupied_tiles()` / `occupies()` are implemented and
  `GameMap.stamp_units` / `is_passable_block` consume them. The 3×3 assumption is
  out of the physics path.
- **Footprint→speed seam (SHIPPED, commit a440d05).** `speed_fn(footprint_samples,
  speed_class) -> int` in `src/simulation/movement.py`, with `FootprintSamples` (carries
  `mobility`; forward-compatible struct), the `half_up` integer rounding rule, and the
  v1 `default_speed` area-weighted-average reduction. Pure / integer / deterministic;
  consumed by movement cadence, never A\* (engine/10 is speed-blind). `GameMap.
  footprint_mobility` supplies the per-tile terrain samples.
- **Facing as radians** + `facing_compass()`; the renderer selects sprites via it.
- **`current_hp`** is the sole sampled stat consumed by gameplay (combat and
  zombie melee deduct from it; the panel shows it against `effective_vitality`).
- **`LifeState`** enum with derived `alive` property.
- **Data-only fields present on every unit:** `EnvironmentProfile`, `Inventory`
  stub, `faction_id`, `awakened`, `mass`, `base_speed`, `offsets`, and the hidden
  stats.

**Designed but not built (data present, behaviour absent):**

- **Modifier engine** — the whole point of base/effective. No modifiers exist;
  effective == base.
- **Environment behaviour** — no O₂ reserve drain, no pressure/temperature/
  submersion damage. The profile is inert data.
- **Faction relationship table** — `FactionId` is an `int` alias; `Stance` is
  unused; combat uses `team` equality.
- **Zombie stat modifiers / variants** — no modifier-driven zombie stats; zombie
  HP is overwritten from `CFG.zombie.hp`, bypassing sampled vitality. The
  size-aware variant system (ogryn, strong/weak zombies, float `size` →
  footprint thresholds) is brainstorm only.
- **Metaphysics behaviour** — fear, the Gray image-seeding hook, will-orientation
  drift, manifestation, the awakening trigger. Only the fields exist.
- **Derived movement from `base_speed`** — the *unit-level* terms (agility,
  encumbrance, wounds, zombification) are unbuilt; movement's base cadence comes from
  config `speed_ticks_per_tile`, and `base_speed` is stored but unconsumed. The
  *terrain* half **has** shipped — see the next item.
- **Real item system** — `Inventory` is a stub; ammo lives in `has_grenade` /
  `has_explosive` booleans, not migrated into it.

**Gaps and inconsistencies to be aware of:**

- `mass`, `density`, and `size` are sampled/derivable but **no system reads
  them** yet — knockback and decompression-shove do not consult mass.
- Sampled `vitality` is effectively discarded for zombies, whose HP is forced
  from config; only marines' HP reflects their roll.
- Non-3 footprints are constructible (the constructor builds a square offset
  grid) but no spawn path or content uses anything other than 3×3, and offset
  rotation with facing is not implemented.
- The unit→state-vector encoding for the planned ML/AI pipeline does not exist;
  the data-driven shape is the groundwork for it, nothing more.
- `team` and `faction_id` are kept as redundant aliases of the same value
  pending the faction table.
