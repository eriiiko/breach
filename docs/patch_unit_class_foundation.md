# Patch Plan — Unit Class Foundation Pass

> **Status:** Plan locked, implementation pending.
> **Drafted:** 2026-05-21.
> **Author intent:** Erik. Execution: implementation agent (sonnet, background).
> **Authoritative spec:** `docs/breach_unit_class_design.md` — read it first.
> **Companion (do NOT use for implementation, design context only):**
> `docs/breach_metaphysics_design_notes.md` — exists; the agent reads it ONLY
> if it needs context for the `awakened` / `imagination` / `will_orientation`
> fields. Their *behaviour* is deferred; only data fields land in this pass.

## 1. Motivation

The current `src/simulation/unit.py` is a single class with ad-hoc fields:
`hp`, `team`, `footprint` (int), `alive`, `facing` (string), etc. It works
but it has no architectural shape — every future system (stats, modifiers,
species variation, environment damage, faction tables, the Hartmann
metaphysics) would bolt on as more ad-hoc fields.

The Unit Class Design Spec lays a foundation that absorbs all of those
systems through three principles (Definition vs Instance; Base vs Effective
stats; rich data, no extra logic). This patch implements **only** the
foundation — fields, types, the species-driven generator, the occupancy
interface. Behaviour for modifiers, combat resolution, environment damage,
fear, and AI nets is deferred per spec §13.

## 2. Locked decisions (Erik, 2026-05-21)

These are NOT open for re-deliberation. The agent follows them.

1. **Language: Python.** Spec C++ syntax is design notation — translate
   to idiomatic Python (dataclasses, type hints, methods). Port to C++
   via pybind11 later, when the class is stable.
2. **Extend, don't replace.** Keep all current Unit runtime / AI fields
   (`orders`, `ap`, `move_path`, `zombie_path`, `last_fire_tick`,
   `has_grenade`, `has_explosive`, `killed_by_zombie`, etc.). Add the new
   spec fields alongside. Nothing in the current Unit gets deleted by
   this patch.
3. **Convenience constructor preserved.** The legacy
   `Unit(name, x, y, team, footprint)` signature MUST continue to work.
   Stat sampling happens internally — use `Simulation.rng` if reachable,
   else a default-seeded RNG so unit tests can construct units without
   booting a Simulation.
4. **One species only: "human".** Covers marines and zombies (via the
   `is_zombie` state). No "ogryn" or "gray" species defs in this pass.
5. **Faction is a thin alias.** `FactionId = int` type alias; `unit.faction_id`
   replaces (or aliases) the current `unit.team`. NO `FactionRelationshipTable`
   yet — combat code keeps its current `if u.team != target.team` checks
   in this pass.
6. **`hp` migration: minimal.** Rename `unit.hp` → `unit.current_hp`.
   `unit.max_hp` reads route through `effective_vitality(unit)` (which
   for now returns `unit.base_stats.vitality` unchanged).
7. **`facing` becomes float radians.** Add `unit.facing_compass()` → str
   returning "N"/"NE"/"E"/"SE"/"S"/"SW"/"W"/"NW" for the sprite lookup.
   Existing default of "N" becomes whatever radian value maps to "N".
   The sprite system in `renderer/sprites.py` reads
   `unit.facing_compass()` instead of `unit.facing`.
8. **Inventory: stub class + keep booleans.** Add a minimal `Inventory`
   class on Unit (empty `equipped`, `carried`, `current_load() == 0`).
   Keep `unit.has_grenade` and `unit.has_explosive` exactly as they are.
   Wiring the booleans INTO the inventory is a future task.
9. **All deferred per spec §13: data fields only, no behaviour.**
   - Modifier system not built; `compute_effective_stats(u)` returns base.
   - Environment damage not applied; `EnvironmentProfile` is data only.
   - Fear / Gray hook / awakening trigger not built.
   - `nn_intelligence_tier` is just an int field on `SpeciesDef`.
10. **`level.toml` schema UNCHANGED.** Spawn entries keep
    `name, team, x, y, footprint`. No `species` field, no stat overrides
    yet. The `team` value becomes the faction_id.

## 3. Naming & type map

| Spec name           | Python implementation                                |
|---------------------|------------------------------------------------------|
| `UnitId`            | `int` (already assigned by `Simulation._next_unit_id`) |
| `SpeciesId`         | `str` for now (one value: `"human"`)                 |
| `FactionId`         | `int` (alias for `team` in this pass)                |
| `TileCoord`         | `tuple[int, int]` — `(x, y)`                         |
| `StatId`            | `enum.Enum` with 10 members                          |
| `BaseStats`         | `@dataclass(frozen=True)`                            |
| `EffectiveStats`    | `@dataclass(frozen=True)` — same shape as BaseStats  |
| `EnvironmentProfile`| `@dataclass`                                         |
| `Inventory`         | regular class (mutable runtime state)                |
| `SpeciesDef`        | `@dataclass(frozen=True)`                            |
| `StatDistribution`  | `@dataclass(frozen=True)` — uses `np.ndarray` for vectors |
| `Stance`            | `enum.Enum` — `Allied / Friendly / Neutral / Hostile` (defined, unused) |
| `LifeState`         | `enum.Enum` — `Alive / Downed / Dead`                |
| `SpatialState`      | folded into Unit's existing `x, y, facing` fields — no separate struct |
| C++ `std::vector<TileCoord>` | Python `list[tuple[int, int]]`              |
| C++ `float`         | Python `float`                                       |

`SpatialState` isn't broken out because the current Unit already has `x`,
`y`, and `facing` directly; no need for a wrapper that adds nothing.

## 4. New modules (six)

### `src/simulation/stats.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class StatId(Enum):
    STRENGTH = "strength"
    AGILITY = "agility"
    ENDURANCE = "endurance"
    VITALITY = "vitality"
    INTELLIGENCE = "intelligence"
    WILL_STRENGTH = "will_strength"
    IMAGINATION = "imagination"
    WILL_ORIENTATION = "will_orientation"
    MASS = "mass"
    BASE_SPEED = "base_speed"

@dataclass(frozen=True)
class BaseStats:
    strength: float
    agility: float
    endurance: float
    vitality: float
    intelligence: float
    will_strength: float
    imagination: float       # hidden until unit.awakened
    will_orientation: float  # hidden permanently for now; [-1, +1]

# EffectiveStats is the same shape; alias for clarity.
EffectiveStats = BaseStats

def compute_effective_stats(unit) -> EffectiveStats:
    """Foundation pass: returns base unchanged. Modifier system slots
    in here later (spec §1.2, §13)."""
    return unit.base_stats

# Per-stat accessors so callers don't reach into BaseStats directly.
def effective_vitality(unit) -> float:    return compute_effective_stats(unit).vitality
def effective_strength(unit) -> float:    return compute_effective_stats(unit).strength
def effective_agility(unit) -> float:     return compute_effective_stats(unit).agility
def effective_endurance(unit) -> float:   return compute_effective_stats(unit).endurance
def effective_intelligence(unit) -> float: return compute_effective_stats(unit).intelligence
def effective_will_strength(unit) -> float: return compute_effective_stats(unit).will_strength
def effective_imagination(unit) -> float: return compute_effective_stats(unit).imagination
def effective_will_orientation(unit) -> float: return compute_effective_stats(unit).will_orientation

def is_stat_player_visible(stat: StatId, unit) -> bool:
    """Per spec §3.3."""
    if stat is StatId.WILL_ORIENTATION:
        return False
    if stat is StatId.IMAGINATION:
        return getattr(unit, "awakened", False)
    return True
```

### `src/simulation/environment.py`

```python
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum

class SubmersionRule(Enum):
    DROWNS = "drowns"
    UNAFFECTED = "unaffected"
    REQUIRES_WATER = "requires_water"

@dataclass(frozen=True)
class EnvironmentProfile:
    # Respiration
    breathes: bool = True
    can_breathe_air: bool = True
    can_breathe_water: bool = False
    o2_reserve_max: float = 60.0      # ticks of survival without a source

    # Pressure (1.0 = standard atmosphere)
    pressure_min: float = 0.4
    pressure_max: float = 2.5

    # O2 partial pressure / concentration of the breathable medium
    o2_level_min: float = 0.15
    o2_level_max: float = 1.0

    # Temperature tolerance band (units: arbitrary scalar for now)
    temperature_min: float = -20.0
    temperature_max: float = 60.0

    submersion: SubmersionRule = SubmersionRule.DROWNS

    # Damage per tick while outside any tolerance — data only;
    # not yet applied by any tick handler.
    environmental_damage_rate: float = 1.0

HUMAN_ENVIRONMENT = EnvironmentProfile()  # the default values ARE human
```

### `src/simulation/inventory.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field

# Item system not yet built — ItemId is a placeholder.
ItemId = int

@dataclass
class Inventory:
    """Stub. Real item system is a future task. For now this holds empty
    lists so the field exists on every Unit; the actual ammo / weapons
    live on Unit.has_grenade / has_explosive until they migrate here."""
    equipped: list = field(default_factory=list)
    carried: list = field(default_factory=list)

    def current_load(self) -> float:
        return 0.0

@dataclass(frozen=True)
class InventoryProfile:
    """On SpeciesDef. Carry rules per species (humans: modest; robots: high)."""
    has_inventory: bool = True
    carry_capacity_base: float = 30.0   # kg
```

### `src/simulation/factions.py`

```python
from __future__ import annotations
from enum import Enum

# Foundation pass: just a type alias. The full relationship table comes
# later (spec §10.1). For now combat code reads unit.team directly.
FactionId = int

class Stance(Enum):
    """Defined for completeness — not yet consulted by any code."""
    ALLIED = "allied"
    FRIENDLY = "friendly"
    NEUTRAL = "neutral"
    HOSTILE = "hostile"
```

### `src/simulation/species.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np

from simulation.environment import EnvironmentProfile, HUMAN_ENVIRONMENT
from simulation.inventory import InventoryProfile

SpeciesId = str   # "human", "gray", "ogryn", ...
N_GENERATED_STATS = 10  # see spec §11.1

# Order MUST match the BaseStats field order + (mass, base_speed) appended.
GENERATED_STAT_NAMES = (
    "strength", "agility", "endurance", "vitality", "intelligence",
    "will_strength", "imagination", "will_orientation",
    "mass", "base_speed",
)

@dataclass(frozen=True)
class StatDistribution:
    mean:   np.ndarray   # shape (N_GENERATED_STATS,)
    stddev: np.ndarray   # shape (N_GENERATED_STATS,)
    correlation: np.ndarray  # shape (N, N), symmetric, unit diagonal
    clamp_min: np.ndarray
    clamp_max: np.ndarray

    def covariance(self) -> np.ndarray:
        # cov[i,j] = stddev[i] * stddev[j] * correlation[i,j]
        s = self.stddev
        return np.outer(s, s) * self.correlation

@dataclass(frozen=True)
class SpeciesDef:
    id: SpeciesId
    name: str
    stat_dist: StatDistribution
    default_offsets: tuple    # tuple[tuple[int, int], ...] — rigid-body footprint
    environment: EnvironmentProfile = HUMAN_ENVIRONMENT
    inventory_profile: InventoryProfile = field(default_factory=InventoryProfile)
    can_become_zombie: bool = True
    nn_intelligence_tier: int = 0   # data-only for now

# --- Human species def -----------------------------------------------------

def _default_3x3_offsets() -> tuple:
    return tuple((dx, dy) for dy in range(3) for dx in range(3))

def _human_stat_distribution() -> StatDistribution:
    # Means tuned so a sampled "average" human reproduces the current
    # Unit's HP and movement values. Pick values that READ AS SENSIBLE
    # on a 1-10 scale for human-facing stats; mass and base_speed in real
    # units. (Agent: tune these against CFG.marine / CFG.zombie defaults
    # to keep current behavior approximately stable.)
    mean = np.array([
        5.0,   # strength
        5.0,   # agility
        5.0,   # endurance
        100.0, # vitality (= current CFG.marine.hp)
        5.0,   # intelligence
        5.0,   # will_strength
        5.0,   # imagination
        0.0,   # will_orientation (centred)
        80.0,  # mass kg
        1.0,   # base_speed (1.0 = nominal; cadence derived elsewhere)
    ], dtype=np.float32)
    stddev = np.array([
        1.0, 1.0, 1.0, 15.0, 1.5,
        1.0, 1.5, 0.2,
        10.0, 0.1,
    ], dtype=np.float32)
    # Correlation: mass<->strength strong positive; mass<->agility mild
    # negative; mass<->base_speed mild negative. Everything else 0.
    n = N_GENERATED_STATS
    corr = np.eye(n, dtype=np.float32)
    idx = {name: i for i, name in enumerate(GENERATED_STAT_NAMES)}
    def link(a, b, r):
        i, j = idx[a], idx[b]
        corr[i, j] = corr[j, i] = r
    link("mass", "strength", 0.6)
    link("mass", "agility", -0.3)
    link("mass", "base_speed", -0.2)
    clamp_min = np.array([1.0, 1.0, 1.0, 20.0, 1.0,
                          1.0, 1.0, -1.0,
                          30.0, 0.3], dtype=np.float32)
    clamp_max = np.array([10.0, 10.0, 10.0, 300.0, 10.0,
                          10.0, 10.0, 1.0,
                          200.0, 2.0], dtype=np.float32)
    return StatDistribution(mean=mean, stddev=stddev, correlation=corr,
                            clamp_min=clamp_min, clamp_max=clamp_max)

HUMAN = SpeciesDef(
    id="human",
    name="Human",
    stat_dist=_human_stat_distribution(),
    default_offsets=_default_3x3_offsets(),
    environment=HUMAN_ENVIRONMENT,
    can_become_zombie=True,
    nn_intelligence_tier=2,
)

SPECIES_REGISTRY: dict = {HUMAN.id: HUMAN}

def get_species(species_id: SpeciesId) -> SpeciesDef:
    return SPECIES_REGISTRY[species_id]
```

### `src/simulation/generation.py`

```python
from __future__ import annotations
import numpy as np

from simulation.species import SpeciesDef, GENERATED_STAT_NAMES
from simulation.stats import BaseStats

def sample_unit_attributes(species: SpeciesDef,
                           rng: np.random.Generator | None = None,
                           overrides: dict | None = None
                           ) -> tuple[BaseStats, float, float]:
    """Returns (base_stats, mass, base_speed).

    Samples the 10-dim stat vector from the species' multivariate normal,
    clamps each entry to its `[clamp_min, clamp_max]` bounds, then peels
    off mass and base_speed into separate returns (they live on Unit, not
    in BaseStats — see spec §3.2).

    overrides: optional dict {stat_name: value} applied AFTER sampling.
    Used for named characters per spec §11.2 — not used by spawn helpers
    in the foundation pass.
    """
    if rng is None:
        rng = np.random.default_rng()
    sd = species.stat_dist
    vec = rng.multivariate_normal(sd.mean, sd.covariance())
    vec = np.clip(vec, sd.clamp_min, sd.clamp_max)
    if overrides:
        for name, value in overrides.items():
            i = GENERATED_STAT_NAMES.index(name)
            vec[i] = float(value)
    # Peel apart: first 8 → BaseStats, last 2 → mass + base_speed.
    base = BaseStats(
        strength=float(vec[0]), agility=float(vec[1]),
        endurance=float(vec[2]), vitality=float(vec[3]),
        intelligence=float(vec[4]),
        will_strength=float(vec[5]), imagination=float(vec[6]),
        will_orientation=float(vec[7]),
    )
    mass = float(vec[8])
    base_speed = float(vec[9])
    return base, mass, base_speed
```

## 5. `src/simulation/unit.py` — target shape

Keep the entire existing `Unit` class. ADD:

- `self.species_id: str = "human"` — set in `__init__`
- `self.base_stats: BaseStats` — sampled at construction (see §6 below)
- `self.mass: float` — sampled
- `self.base_speed: float` — sampled
- `self.awakened: bool = False`
- `self.life_state: LifeState = LifeState.ALIVE` (new enum)
- `self.faction_id: int` — initialized from `team` (alias; see spec §10.1)
- `self.environment: EnvironmentProfile` — pointer to species default
- `self.inventory: Inventory` — empty stub
- `self.offsets: list[tuple[int, int]]` — per-unit copy of species default,
  trimmed/adjusted if the convenience `footprint` arg is non-default

RENAME:
- `self.hp` → `self.current_hp` (UPDATE every reader across the codebase)
- `self.facing` stays the same name BUT changes from `str` → `float radians`
- `self.max_hp` is **removed as a field**; readers call `effective_vitality(unit)` instead

KEEP unchanged (no spec coverage, runtime/AI state):
- `name`, `team` (kept alongside new `faction_id` for now; same value), `id`,
  `x`, `y`, `alive` (kept, derived from life_state), `is_zombie`,
  `current_order_type`, `orders`, `ap`, `has_grenade`, `has_explosive`,
  `last_fire_tick`, `fire_target`, `zombie_activated`, `zombie_path`,
  `zombie_path_idx`, `zombie_move_accumulator`, `last_melee_tick`,
  `killed_by_zombie`, `move_path`, `path_tick_offset`,
  `speed_ticks_per_tile`, `footprint` (the int, for now — see below)

ADD methods:
- `occupied_tiles(self) -> list[tuple[int, int]]` — returns
  `[(tile_x + dx, tile_y + dy) for (dx, dy) in self.offsets]`. No rotation
  applied yet — the 3×3 symmetric default doesn't need it. Add a TODO
  for non-symmetric rigid rotation per spec §15 item 3.
- `occupies(self, tile: tuple[int, int]) -> bool` — `tile in occupied_tiles()`
  or membership check.
- `facing_compass(self) -> str` — convert `self.facing` radians to "N"/"NE"/etc.
  Use the convention: 0 = East, π/2 = North, π = West, 3π/2 = South
  (standard math convention). Snap to nearest 45° sector. Default
  spawn facing of `π/2` (= North) preserves current behavior.

### Convenience constructor

```python
def __init__(self, name, x, y, team=0, footprint=3, species_id="human"):
    # ... all current init code stays ...
    
    # New: species + stat sampling. Use a fresh default RNG so tests
    # can construct units standalone; Simulation.add_unit may re-sample
    # with its seeded RNG if it wants deterministic spawns.
    self.species_id = species_id
    species = get_species(species_id)
    rng = np.random.default_rng()
    self.base_stats, self.mass, self.base_speed = \
        sample_unit_attributes(species, rng)
    self.environment = species.environment
    self.inventory = Inventory()
    self.life_state = LifeState.ALIVE
    self.awakened = False
    self.faction_id = int(team)   # alias for now
    
    # Per-unit offset list, copied from the species default.
    if footprint == 3:
        self.offsets = list(species.default_offsets)
    else:
        # Convenience override: build a square footprint of side `footprint`.
        self.offsets = [(dx, dy) for dy in range(footprint) for dx in range(footprint)]
    
    # Convert legacy string facing default ("N") to radians (π/2).
    self.facing = 1.5707963267948966   # math.pi / 2 = North
    
    # current_hp from sampled vitality.
    self.current_hp = float(self.base_stats.vitality)
```

The integer `self.footprint` field stays for backward compat with any
caller that reads it. Optionally: make it a `@property` returning the
side length of the offset list bounding box. Either works.

### LifeState enum

Place in `unit.py` (or `src/simulation/lifestate.py` if Erik prefers):

```python
from enum import Enum
class LifeState(Enum):
    ALIVE = "alive"
    DOWNED = "downed"
    DEAD = "dead"
```

Keep `unit.alive` as a `@property`:
```python
@property
def alive(self) -> bool:
    return self.life_state is LifeState.ALIVE
```

## 6. Consumer updates

The agent updates ONLY where the existing code reads the renamed fields
or could benefit from `occupied_tiles()`.

- **`src/simulation/gamemap.py`** — `stamp_units()` and `is_passable_block()`
  currently iterate `range(unit.footprint) × range(unit.footprint)`.
  Replace with iteration over `unit.occupied_tiles()`. This makes the
  3×3 assumption disappear from the physics path (spec §6 contract).
- **`src/simulation/simulation.py`** — every reader of `unit.hp` →
  `unit.current_hp`. Every reader of `unit.max_hp` → `effective_vitality(unit)`.
  `Simulation.add_unit`: no signature change needed, but the agent should
  consider re-sampling the unit's stats with `self.rng` so spawns are
  deterministic when `seed=` is passed. (Optional: if it complicates the
  diff, leave it for a follow-up.)
- **`src/simulation/combat.py`** — `unit.hp` → `unit.current_hp` (damage
  application). No team logic changes.
- **`src/simulation/ai_zombie.py`** — `unit.hp` → `unit.current_hp` (melee
  damage, kill check). No team logic changes.
- **`src/input_handler.py`** — unit-selection AABB uses `unit.tile_x +
  unit.footprint`; if `footprint` is still an int field (preferred for
  backward compat), no change. If it became a property, still works.
- **`renderer/sprites.py`** — `unit.facing` is now radians. Change
  `get_marine_sprite(unit.facing)` call sites to
  `get_marine_sprite(unit.facing_compass())`. (Find these in
  `renderer/game_renderer.py:_draw_units_world` or wherever the sprite
  pick happens.)
- **`renderer/game_renderer.py`** — find any read of `unit.hp` /
  `unit.max_hp` (likely in the right-side panel for selected unit) and
  switch to `unit.current_hp` / `effective_vitality(unit)`.

## 7. Tests

Add `tests/test_unit_class.py` covering:
- BaseStats fields exist and are sampleable from human species
- `sample_unit_attributes` returns values within clamp bounds for 100 seeded draws
- `Unit("test", 5, 5)` constructs cleanly with all new fields populated
- `unit.occupied_tiles()` returns 9 tiles for a default human
- `unit.occupied_tiles()` returns 16 for `Unit("test", 5, 5, footprint=4)`
- `unit.facing_compass()` returns expected string for each cardinal radian
- `effective_vitality(unit)` returns `unit.base_stats.vitality` (modifier-stub identity)
- `is_stat_player_visible(StatId.IMAGINATION, unit)` is False unless `unit.awakened`
- `is_stat_player_visible(StatId.WILL_ORIENTATION, unit)` is always False

Existing 8 tests must all still pass.

## 8. Step-by-step execution order

1. **Snapshot baseline.** `git status` clean; `pytest tests/` → 8 passing.
2. **Add `src/simulation/stats.py`.** No callers yet. Tests pass.
3. **Add `src/simulation/environment.py`.** No callers yet. Tests pass.
4. **Add `src/simulation/inventory.py`.** No callers yet. Tests pass.
5. **Add `src/simulation/factions.py`.** No callers yet. Tests pass.
6. **Add `src/simulation/species.py`.** Imports stats, env, inventory. Tests pass.
7. **Add `src/simulation/generation.py`.** Imports species, stats. Tests pass.
8. **Extend `src/simulation/unit.py`.** New fields + methods. Constructor
   updated. `hp` → `current_hp` rename here; `facing` becomes radians.
   `LifeState` enum added. **Tests WILL break here** — that's expected
   because consumers still read `unit.hp`. Move quickly to the next step.
9. **Update `gamemap.py`** — `stamp_units`, `is_passable_block` use
   `unit.occupied_tiles()`.
10. **Update `simulation.py`** — `hp` → `current_hp` everywhere.
    `max_hp` → `effective_vitality(unit)` everywhere.
11. **Update `combat.py`, `ai_zombie.py`** — `hp` → `current_hp`.
12. **Update `renderer/sprites.py`** + caller in `game_renderer.py` —
    `unit.facing` is now radians; sprite lookup uses
    `unit.facing_compass()`.
13. **Update `renderer/game_renderer.py` panel HP readout** — `hp` →
    `current_hp`, `max_hp` → `effective_vitality(unit)`.
14. **Update / add tests** — `tests/test_unit_class.py` new; fix any
    existing tests that read the renamed fields.
15. **Verify gates** (§9). All green.
16. **Commit in logical chunks** (one per new module, plus one for the
    Unit rewrite, plus one for consumers, plus one for tests, plus one
    for docs note). Push after each verified commit.

## 9. Verification gates (must all pass)

1. `pytest tests/` — original 8 + new ones, all passing.
2. `C:/Users/steen/anaconda3/python.exe tests/test_main_smoke.py --auto`
   → "OK — main_smoke rendered 600 frames; sim.tick=60, phase=1, paused=True".
3. `git status` clean except for intentional changes.
4. `git grep -n 'self\.hp\b\|\.max_hp\b' src/ main.py` returns no production
   hits (only test files allowed; comments OK).
5. The new modules import cleanly from `src/simulation/__init__.py` if
   they're meant to be public — check the existing `__init__.py` for
   convention.
6. Interactive verification by Erik:
   - Game launches, 7 units spawn at scouted coords
   - Marine sprites face north (default)
   - Selecting a unit, panel shows HP (still works)
   - Issuing move order, pressing Space — units walk
   - No new visible regressions vs the pre-patch behaviour

## 10. Final report format

```
## Unit class foundation — complete

### Commits landed (chronological)
- <hash> <subject>
- ...

### Pushed: yes / no

### Decisions made by the agent during work
- Any tactical detail not specified in this plan + how it was resolved.

### Test results
- pytest: N/N passing (specify which new tests landed)
- smoke test: <output line>

### Deviations from plan
<any place I diverged from this doc and why>

### Anything weird
<surprises, things flagged but not fixed, residual concerns>

### Pending for Erik
- Interactive launch verification
- Anything else worth flagging
```

## 11. Memory updates after merge

- Update `project_breach_architecture_2026_05.md` memory note: add
  "Unit class foundation completed 2026-05-21. BaseStats / SpeciesDef /
  EnvironmentProfile / Inventory in place; modifier system, environment
  damage, faction relationship table, Hartmann metaphysics still
  deferred per spec §13."

---

When this plan is picked up (fresh, by an agent or by Erik):

- The spec (`breach_unit_class_design.md`) is the canon. This patch plan
  is the implementation contract for the foundation pass only.
- The metaphysics doc is design context for the hidden fields; their
  behaviour is NOT in scope for this pass.
- Stash `stash@{0}` holds the planning UX bundles (A+B+C) — different
  feature, will be revisited after this lands.
