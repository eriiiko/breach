# Breach — Unit Class Design Specification

**Status:** Foundation spec. Implementation-ready.
**Target:** Implementation in Python first (the current Breach simulation layer),
with a planned port to C++ via pybind11 once the class shape stabilises. C++
syntax is used throughout the spec as design notation; translate to idiomatic
Python (dataclasses, type hints, methods) at implementation time.
**Scope of this document:** Defines the data model for units — stats, occupancy, mass,
environment tolerance, faction, inventory, runtime state, and the species-driven unit
generator. It deliberately does **not** specify combat resolution, the modifier/buff
system internals, AI behaviour, or the Hartmann "magic" systems. Those are deferred —
but the class must be *shaped* so they attach later without a rewrite. Deferred items
are listed explicitly in section 13.

---

## 1. Architecture principles

Three principles drive every decision below. Implement them first; everything else
hangs off them.

**1.1 — Definition vs. Instance.**
A `SpeciesDef` is static, data-driven design data (humans, xenos, Grays, animals,
robots). A `Unit` is a live instance. A designer adds a new species by editing data,
**not** by writing a subclass. There is exactly one `Unit` class; species variation is
data, not inheritance.

**1.2 — Base vs. Effective stats.**
Every unit has `BaseStats`, rolled **once** at creation and thereafter **immutable**.
The values used in gameplay are `EffectiveStats`, *computed* as base + the sum of all
active modifiers (wounds, encumbrance, zombification, fear, buffs). Effective stats are
always *queried through a function*, never stored as the source of truth.

This single rule resolves three otherwise-messy problems:
- Zombification becomes a *modifier*, not a mutation — so "if you were strong, you are
  still strong" is automatic, because the base is never touched.
- Equipment that extends survivability (e.g. a vac-suit) is a modifier on the effective
  environment profile — same machinery.
- Buffs, wounds, and the future fear/Hartmann effects all reuse one code path.

The modifier *system* is deferred (section 13). For the foundation pass, the class
must (a) keep base stats immutable after creation and (b) expose effective stats
through a function that, for now, simply returns base. The modifier list can be empty.

**1.3 — Build a rich foundation, not all systems.**
Fields and structure for future systems (magic, fear, AI nets) go in now. Their *logic*
does not. Where a system is deferred, the class carries the data it will need and
nothing more.

---

## 2. Core types and identifiers

```cpp
// Stable, unique, persistent per unit. Assigned by the unit registry
// (Simulation) from a monotonically-incrementing counter — IDs are NEVER
// reused, even after a unit dies. Needed for save/load, cross-system
// references, and the future NN training-data pipeline.
using UnitId    = uint64_t;          // 0 reserved as "invalid"
using SpeciesId = uint32_t;
using FactionId = uint32_t;

// Compile-time enum identifying which stat — used by helpers like
// is_stat_player_visible(StatId, Unit). NOT a runtime counter.
enum class StatId {
    Strength, Agility, Endurance, Vitality, Intelligence,
    WillStrength, Imagination, WillOrientation,
    Mass, BaseSpeed,
};

// Integer tile coordinate. The world grid is in tiles.
struct TileCoord { int32_t x; int32_t y; };

// World scale reference: 1 tile = 1/3 metre. A baseline human occupies
// 3x3 tiles = 1 m^2.
```

---

## 3. Stats

### 3.1 The stat set

Seven primary stats plus two physical attributes generated alongside them:

| Stat            | Player-visible | Meaning |
|-----------------|----------------|---------|
| `strength`      | yes            | Raw physical force. Absolute & comparable (9 > 8 always). Feeds derived melee/knockback/carry. |
| `agility`       | yes            | Fine motor control, dodge, reaction. |
| `endurance`     | yes            | **Stamina pool** — sprinting, sustained melee, held actions. |
| `vitality`      | yes            | **HP pool** — the maximum health value. |
| `intelligence`  | yes            | Cognition. Also selects the unit's NN tier (see 13). |
| `will_strength` | yes            | Shown to the player simply as **WILL**. RPG-familiar: spell-interrupt resistance, morale, fear *susceptibility*. |
| `imagination`   | **no (hidden)**| Image-forming faculty. Always exists in the class; never rendered unless the unit is `awakened`. |
| `will_orientation` | **no (hidden)** | Float in `[-1, +1]`: separative ↔ universal. Never rendered to the player at all (for now). Companion to `will_strength`. |
| `mass`          | (see §5)       | Kilograms. Generated with the stat vector so it can correlate with strength. |
| `base_speed`    | (see §8)       | Locomotion rate before modifiers. |

Notes:
- `endurance` and `vitality` are deliberately distinct: stamina vs. health. Do not merge.
- `will_strength` and `will_orientation` are two components of one concept. The player
  sees only the magnitude. Orientation is the hidden "vector" half.
- `imagination` is hidden by *UI policy*, not by absence — the field is always present
  and always populated. Visibility rule in §3.3.

### 3.2 Base vs. effective

```cpp
struct BaseStats {
    float strength;
    float agility;
    float endurance;       // max stamina
    float vitality;        // max HP
    float intelligence;
    float will_strength;
    float imagination;     // hidden stat — always stored
    float will_orientation;// hidden, [-1, +1]
    // mass and base_speed are stored on the Unit (see §5, §8) but are
    // generated jointly with these — see §11.
};
```

`EffectiveStats` has the same shape. It is produced by a function:

```cpp
EffectiveStats compute_effective_stats(const Unit& u);
// For now: returns BaseStats unchanged.
// Later: base + sum of active modifiers (wounds, zombie, encumbrance, ...).
```

Gameplay code must call `compute_effective_stats` (or per-stat accessors), never read
`BaseStats` directly, so the modifier system can be slotted in transparently.

### 3.3 Hidden-stat visibility rule

Visibility is a **UI concern**, not stored as part of the data:
- `imagination`: rendered on the character sheet **only if** `unit.awakened == true`.
- `will_orientation`: **never** rendered to the player in the current design.
- All other stats: always visible.

Provide one helper so the UI never hard-codes the rule:

```cpp
bool is_stat_player_visible(StatId stat, const Unit& unit);
```

---

## 4. Species definition (`SpeciesDef`)

Data-driven. One record per species. Loaded from data files; referenced by `Unit` via
`SpeciesId`.

```cpp
struct SpeciesDef {
    SpeciesId   id;
    std::string name;                 // "Human", "Xeno", "Gray", "Maintenance Robot"...

    StatDistribution stat_dist;          // §11 — multivariate generator parameters
    std::vector<TileCoord> default_offsets; // §5/§6 — rigid-body occupancy offsets
                                            //         from anchor. Articulated species
                                            //         add their own fields when built.
    EnvironmentProfile environment;      // §7 — default survivability profile
    InventoryProfile inventory_def;   // §9 — carry rules

    bool  can_become_zombie;          // §10 — some species are immune
    int   nn_intelligence_tier;       // §13 — which NN drives this species' AI

    // Future hook: ability/trait list. Empty for now.
    // std::vector<AbilityId> abilities;
};
```

Add a subclass of `Unit` **only** if a species needs genuinely divergent *code* — and
even then prefer a component. Default answer: data, not subclass.

---

## 5. Mass, occupancy, size, density

These four are related but distinct. Keep all of them.

- **`mass`** — kilograms. A real number per unit, generated with the stat vector so it
  can correlate with strength (§11). Drives knockback (resistance ∝ mass) and how
  strongly the decompression/wind sim can shove the unit.
- **occupancy** — the set of tiles a unit currently occupies on the world grid.
  The **only** contract is the method-based interface on `Unit` (§6):

```cpp
std::vector<TileCoord> Unit::occupied_tiles() const;
bool                   Unit::occupies(TileCoord) const;
```

  How a unit stores the data that backs `occupied_tiles()` is an
  **implementation detail** of that unit kind:
  - **Rigid units** (3×3 human, 4×4 large biped, any non-deforming body) typically
    hold a list of tile offsets relative to anchor and apply rotation/reflection
    to `facing` on read. The species default offsets live on `SpeciesDef`; a unit
    may carry its own override if it has been modified individually.
  - **Articulated units** (snakes/worms, deforming blobs) hold per-instance state
    such as a segment-chain deque. The footprint changes shape every tick. See
    §15 open-questions.

  The foundation pass implements only the rigid case. The interface contract
  must hold for both — consumers (collision, LOS, hit-detection, `stamp_units`,
  physics/decompression) call `occupied_tiles()` and never read the underlying
  storage.

- **`size`** — a *derived* integer = `occupied_tiles().size()`. Not stored as
  source of truth; computed. Exposed as a convenient scalar (useful as an ML
  feature, and for any mechanic that wants a single number). Cache per-tick if
  it ever shows up in a profile; otherwise just compute.
- **`density`** — derived = `mass / (size * tile_area_m2)`, where
  `tile_area_m2 = (1/3 m)^2`. Cheap to compute, and it gives the physics sim a
  principled way to rank units that share an occupancy footprint. Two units on
  the same 3×3 tiles but different mass behave differently in a decompression
  wind — exactly the kind of emergent behaviour Breach is built around.

### 5.1 Strength × mass interaction

`strength` stays **absolute and comparable** — never scaled into the stored stat.
Physical *output* is a derived value combining strength and mass:

```cpp
// Designer-tunable. One concrete suggestion:
float effective_physical_force(const Unit& u) {
    const float ref_mass = 80.0f;     // baseline human
    return effective_strength(u) * std::sqrt(u.mass / ref_mass);
}
```

So a big, strong unit out-hits a small, strong unit; raw `strength` still reads cleanly
on the sheet. Knockback *resistance* scales directly with `mass`.

---

## 6. Spatial state

```cpp
struct SpatialState {
    TileCoord anchor;        // the footprint's reference tile in the world grid
    float     facing;        // radians. Matters for normal-map lighting, LOS, melee arcs.
    // Rendering may interpolate sub-tile position separately; the logical
    // position is tile-discrete (movement is tick-quantised — see §8).
};
```

**Occupancy is queried through methods on `Unit`, never by reading the underlying
storage directly.** This mirrors the base/effective stat pattern (§1.2): access
through a function so the storage representation can change per unit kind.

```cpp
// Required interface for any consumer that needs to know which tiles a unit
// occupies (collision, line-of-sight, hit-detection, stamp_units for the
// physics/decompression sim, etc.). No consumer reads the unit's internal
// shape storage.
std::vector<TileCoord> Unit::occupied_tiles() const;
bool                   Unit::occupies(TileCoord) const;
```

For rigid units, `occupied_tiles()` returns `anchor + each tile offset` from the
unit's stored offset list (default from `SpeciesDef.default_offsets`), rotated /
reflected to `facing` if the shape is non-symmetric. Articulated units override
the method to walk their per-instance chain state (see §15). Consumers see a
uniform interface and never need to know which kind of body they are looking at.

---

## 7. Environment profile — atmosphere, pressure, water

Breach's core premise is hull breaches and decompression, so a unit's relationship to
its environment is **foundational**, not a later system. The profile is composed from
orthogonal flags so *any* combination is creatable (vacuum-proof robot, water-only fish,
amphibian, vacuum-proof-but-water-fragile robot, etc.).

```cpp
struct EnvironmentProfile {
    // --- Respiration ---
    bool breathes;            // false  -> needs no O2 at all (robots, constructs)
    bool can_breathe_air;     // extracts O2 from a gaseous atmosphere
    bool can_breathe_water;   // extracts O2 from water
    float o2_reserve_max;     // max held reserve (ticks of survival without a source)

    // --- Pressure ---
    float pressure_min;       // survivable ambient pressure range
    float pressure_max;

    // --- O2 concentration / partial pressure (of a breathable medium) ---
    float o2_level_min;
    float o2_level_max;

    // --- Temperature ---
    // Survivable ambient-temperature range. Breach does not necessarily
    // simulate a temperature field on every tile, but it does apply heat
    // *radiation* from fires and energy-weapon impacts; a unit inside that
    // radiation outside its tolerance band takes environmental damage.
    float temperature_min;
    float temperature_max;

    // --- Water / submersion ---
    enum class SubmersionRule {
        Drowns,        // submersion is lethal over time (most land units)
        Unaffected,    // sealed / indifferent (sealed robots, suited units)
        RequiresWater  // must stay submerged or asphyxiate (fish)
    };
    SubmersionRule submersion;

    // --- Damage taken per tick while outside any tolerance ---
    float environmental_damage_rate;
};
```

Behaviour:
- A unit replenishes its O2 reserve while in a medium it *can* breathe and whose O2
  level is within tolerance; otherwise the reserve depletes.
- Reserve hits zero -> asphyxiation damage at `environmental_damage_rate`.
- Ambient pressure outside `[pressure_min, pressure_max]` -> environmental damage
  (this is the decompression-trauma channel).
- Ambient temperature outside `[temperature_min, temperature_max]` -> environmental
  damage. The temperature a unit experiences comes from heat radiation (fires,
  energy-weapon impacts), not necessarily a per-tile temperature field.
- `breathes == false` -> the O2 reserve / asphyxiation logic is skipped entirely.

**Suits and equipment** extend survivability by modifying the *effective* environment
profile — same base/effective pattern as stats (§1.2). A baseline human has narrow
tolerances; an equipped vac-suit modifies the effective profile to survive vacuum and
supplies "canned" O2. The `EnvironmentProfile` on `SpeciesDef` is the **base**; the
effective profile = base + equipment modifiers. (Modifier system itself: deferred.)

Examples expressible with this struct:
- **Human:** breathes air; narrow pressure band; `Drowns`; modest O2 reserve.
- **Fish:** breathes water only; `RequiresWater`.
- **Vacuum robot:** `breathes = false`; full pressure range; `submersion = Drowns`
  (water shorts it out) — *or* `Unaffected` for an all-environment robot.

---

## 8. Speed and movement cadence

`base_speed` is a generated attribute (tighter variance than the combat stats — in the
generator this is just a small variance entry; no special-casing). Effective speed is
derived from base speed, agility, encumbrance (§9), wounds, zombification, and terrain.

Movement is tick-quantised. Effective speed maps to a **movement cadence** — ticks per
tile — feeding the existing tick model (baseline marine ≈ 1 tile / 3 ticks; xeno ≈
1 tile / 1 tick for horror pacing). The cadence is derived; the resolution system that
consumes it is deferred. The class carries `base_speed`; the rest is computed.

---

## 9. Inventory and carry capacity

```cpp
struct InventoryProfile {              // on SpeciesDef
    bool  has_inventory;               // some animals: false
    float carry_capacity_base;         // baseline; robots set high
};

struct Inventory {                     // runtime, on Unit
    std::vector<ItemId> equipped;      // weapon/armour/suit slots
    std::vector<ItemId> carried;       // backpack
    float current_load() const;        // sum of item masses
};
```

- Effective carry capacity is derived from `effective_strength` and the species
  `carry_capacity_base`. Aim for realism — capacities stay modest; robots are the
  exception and carry heavier loads.
- Load above capacity feeds the encumbrance term in effective speed (§8). Encumbrance
  is a modifier, so it routes through the deferred modifier system; for now just store
  load and capacity.

---

## 10. Faction, allegiance, and zombification

### 10.1 Faction

The `Unit` stores only a `faction_id`. **Relationships between factions are not stored
on the unit** — they are a mission/match-level table, because they are dynamic and
per-mission (a faction may be friendly in one mission, hostile in another; a map may
host three mutually hostile teams).

```cpp
using FactionId = uint32_t;

enum class Stance { Allied, Friendly, Neutral, Hostile };

// Owned by the mission/match, NOT by the unit.
struct FactionRelationshipTable {
    Stance stance_between(FactionId a, FactionId b) const;
};
```

This supports three-way chaos on one map, per-mission friend/foe assignment, and the
Grays' fight-or-flight behaviour (which is *AI behaviour* layered on top of stance —
deferred — not a faction property).

### 10.2 Zombification

`is_zombie` is a runtime `bool` on the `Unit`. Its stat effects go through the modifier
layer (§1.2) as a **zombie modifier** — they do **not** mutate `BaseStats`. Therefore:
- Strength, mass, occupancy — preserved (a big, heavy unit stays big and heavy).
- `will_strength`, `will_orientation`, `intelligence` — driven down by the modifier.
- The lowered effective intelligence reroutes the unit to a lower NN tier (§13).

`SpeciesDef.can_become_zombie` gates eligibility — some species are immune.
Faction: zombies belong to a zombie faction whose stance is `Hostile` to all
non-zombie factions and `Neutral`/`Friendly` to itself — "zombies don't kill zombies."

(`is_zombie` as a bool is fine for now. If more transformation states appear later it
should generalise to a small condition set — but not tonight.)

---

## 11. Unit generation

Units are not hand-statted by default. Each species defines a **distribution**; the
generator samples it at spawn. Named/important units can override with hand-set values.
A unit generator should live in the level editor.

### 11.1 Correlated sampling (multivariate normal)

Stats are **not** drawn independently — that would lose correlations like "bigger units
tend to be stronger." Draw the whole stat vector from a **multivariate normal**.

```cpp
// N = 10: strength, agility, endurance, vitality, intelligence,
//         will_strength, imagination, will_orientation, mass, base_speed.
constexpr int N_GENERATED_STATS = 10;

struct StatDistribution {                              // on SpeciesDef
    std::array<float, N_GENERATED_STATS> mean;         // per-stat mean
    std::array<float, N_GENERATED_STATS> stddev;       // per-stat standard deviation
    CorrelationMatrix                    correlation;  // 10x10, symmetric,
                                                       // positive semi-definite,
                                                       // unit diagonal
    std::array<float, N_GENERATED_STATS> clamp_min;    // hard lower bound per stat
    std::array<float, N_GENERATED_STATS> clamp_max;    // hard upper bound per stat
};
```

- Specifying a **correlation matrix** + per-stat `stddev` is more intuitive for a
  designer than a raw covariance matrix; the code composes covariance from them.
- Sampling: draw from the multivariate normal, then **clamp** each stat to
  `[clamp_min, clamp_max]`. Clamping slightly distorts the tails; if exactness matters,
  use rejection sampling within the bounds instead. Clamping is fine to start.
- Example correlation entries for a humanoid species: `mass`↔`strength` strongly
  positive; `mass`↔`agility` mildly negative; `mass`↔`base_speed` mildly negative.
- "Tighter variance on marine speed" needs no special case — it is simply a small
  `stddev` entry for `base_speed` in the marine species.
- `will_orientation`: most species centre near `0` with small `stddev`. Mirror People
  centre near `-0.9`. The generator treats it like any other vector component.

### 11.2 Overrides

A spawn request may carry explicit stat overrides; the generator samples the rest and
applies the overrides on top. Named characters use this.

---

## 12. The `Unit` class — assembled

```cpp
class Unit {
public:
    // --- Identity (immutable) ---
    UnitId    id;
    SpeciesId species;

    // --- Static attributes, rolled once at creation (immutable) ---
    BaseStats         base_stats;
    float             mass;            // kg
    float             base_speed;

    // --- Runtime state (mutable) ---
    SpatialState      spatial;         // anchor tile + facing
    float             current_hp;      // <= effective vitality
    float             current_stamina; // <= effective endurance
    float             o2_reserve;      // current reserve; meaningless if !breathes

    enum class LifeState { Alive, Downed, Dead };
    LifeState         life_state;

    FactionId         faction;
    bool              is_zombie;       // stat effects via modifier, not mutation
    bool              awakened;        // gates visibility of `imagination`

    Inventory         inventory;

    // --- Occupancy (storage is implementation-defined per unit kind; §5/§6) ---
    // Rigid units typically hold a list of tile offsets here, derived from the
    // species default and optionally per-instance overridden. Articulated units
    // hold per-instance chain state. The CONTRACT is the methods below, not
    // any particular field. No consumer reads the storage directly.
    std::vector<TileCoord> occupied_tiles() const;   // rigid + articulated
    bool                   occupies(TileCoord) const;

    // --- Future-system hooks (data only; logic deferred — see §13) ---
    // std::vector<Modifier> modifiers;        // wounds, buffs, zombie, fear...
    // ActionState           action_state;     // current tick-based action

    // --- Derived accessors ---
    int   size() const     { return (int)occupied_tiles().size(); }
    float area_m2() const  { return size() * (1.0f / 3.0f) * (1.0f / 3.0f); }
    float density() const  { return mass / area_m2(); }
};
```

`EnvironmentProfile`, `StatDistribution`, `InventoryProfile`, and the rigid-body
default occupancy offsets live on `SpeciesDef` and are reached via `species`. The
unit's *effective* environment profile = species base + equipment modifiers
(modifier system deferred). The `size()`, `area_m2()`, and `density()` accessors
above are computed on demand; cache them per-tick if a profile ever shows them
as hot — for the foundation pass, naive recompute is fine.

---

## 13. Explicitly deferred — hooks now, logic later

The class above carries the data these systems need. Their logic is **out of scope for
this foundation pass**:

- **Modifier system** — the engine that turns base stats + active modifiers into
  effective stats. For now `compute_effective_stats` returns base unchanged.
- **Combat / action resolution** — tick-based actions, cooldowns, the `ActionState`
  field, damage application.
- **The Hartmann "magic" systems** — fear, the Gray image-seeding hook, will-orientation
  *drift*, manifestation, the awakening *trigger* logic. Fields (`imagination`,
  `will_orientation`, `awakened`) exist now; behaviour does not.
  See the companion document, `breach_metaphysics_design_notes.md`.
- **AI / neural nets** — `intelligence` already selects an `nn_intelligence_tier`; the
  nets themselves and the unit→state-vector encoding are future work. Designing the
  class data-driven makes that encoding straightforward later.

---

## 14. Balance methodology (for tuning the generator)

When choosing a distribution for any stat — `vitality` is the worked example:

1. **Balance the mean first.** Set the mean from gameplay intent — how many hits a
   typical unit of this species should survive against typical damage. Damage and mean
   vitality are tuned together.
2. **Set variance from the extremes.** Decide where a roll stops being "normal" and
   counts as unusual. Choose `stddev` so that a unit at that unusual-but-plausible end
   differs from the mean by a *meaningful but bounded* amount — e.g. survives exactly
   one fewer / one more hit. Not a wild swing.
3. **The clamp is the extreme.** `clamp_min` / `clamp_max` (§11.1) are exactly the
   "this is as far as it goes" bounds — they prevent a freak multivariate draw from
   producing an absurd outlier.

Mean encodes the intended experience; variance encodes how far individuals stray;
clamp guarantees no individual breaks the design.

---

## 15. Open questions

1. **Footprint scope** — the foundation pass uses the 3×3 human footprint only.
   One larger zombie variant is a candidate for a 4×4 footprint (corners
   optionally removed); to be decided during implementation. Articulated /
   deforming bodies are a separate case — see item 6.
2. **Facing granularity** — continuous `float` radians (better for normal-map lighting)
   vs. snapped 8-direction. Spec assumes continuous with an optional snap helper.
3. **Rigid-footprint rotation** — for asymmetric rigid shapes (e.g. an L-shaped
   biped, a unit with a long held weapon), confirm whether the offset list
   rotates with `facing` or stays axis-aligned. Spec assumes it rotates. Not
   relevant while only the symmetric 3×3 footprint is in use.
4. **Sub-tile position** — logical position is tile-discrete here. Confirm rendering
   interpolation is handled entirely in the render layer (recommended) and never feeds
   logic.
5. **`will_orientation` at creation** — confirmed hidden from the player permanently for
   now; revisit if a reveal mechanic is ever wanted.
6. **Articulated and deforming footprints** — snakes/worms (segment-chain bodies)
   and deforming blobs (octopuses squeezing through gaps) are a known future case.
   The `occupied_tiles()` / `occupies()` interface on `Unit` (§6) is deliberately
   designed to absorb them with no change to consumers — collision, LOS,
   hit-detection, and `stamp_units` already query through methods, not fields.

   The intended segment-chain model: a deque of occupied tiles, head at one end.
   Each tick the head advances to a new tile; every following segment moves into
   the tile vacated by the segment ahead of it; the tail pops. The footprint
   changes shape every tick. Such a unit has no single `facing` and its occupancy
   cannot be derived from anchor + facing — it carries its own mutable
   per-instance footprint state. Continuous wriggle is render-layer interpolation
   only; game state stays tile-discrete.

   One downstream consequence to revisit when articulated bodies are actually
   built: mid-body damage could sever a unit into two units, which raises
   per-segment HP as its own design question.

   Not implemented in the foundation pass — forward-looking note only.
