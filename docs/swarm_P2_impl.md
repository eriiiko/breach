# Swarm units P2 — implementation doc: the swarm store scaffold

> **v2 — 2026-09-24, post-critique; resolutions R1–R19 of
> `docs/swarm_P2_critique_2026-09-24.md` applied.** v1 = `c13b964`
> (pre-critique). Arc #63, branch `63-swarm-units` (patch branch
> `63-p2-swarm-store` per the plan). Every seam re-verified against HEAD
> `cb3742f` (P1 has landed: `src/simulation/philox32.py`, `cpp/src/philox32.h`,
> and the CLAUDE.md "Deterministic RNG" row). For every file P2 touches,
> HEAD equals the fire-12 merge-base `fdc986a`, so the line numbers below are
> valid against both.
>
> Executes `docs/architecture/engine/17_swarm_units.md` (BLESSED 2026-09-10)
> §2, §6, §7 and §9b's "shared foundation" row, under plan row P2
> (`docs/swarm_units_patch_plan_2026-09-10.md`). The chapter is executed, not
> re-designed; where its body disagrees with the Review log, the log wins
> (§12 lists every conflict found).
>
> **P2 in one sentence:** per-species SoA arrays on GameMap, born
> `(N_env, max_units)` through the residency path, with a lifecycle facade, a
> strict species table re-bound through one explicit reload seam, and a
> presence-gated `SWARM_SECT_V1` section that the digest hashes and the
> recorder mirrors raw — and no behaviour, no tick line, no save format and
> no C++.

## v2 changes (R# → what changed → where)

| R# | What changed | § |
|---|---|---|
| R1 | `diff_trajectories` handles `__swarm__` BEFORE its one-sided-key check; a missing key reads as an absent carrier. New test T9. `test_thermostat_books.py` joins the must-stay-green list | §6, §11 |
| R2 | Every integer in a swarm hash payload is `int()`-converted; the column header is `f"({int(hw)},)"`. New test T10 | §5.2, §5.3, §11 |
| R3 | ONE explicit reload seam, `Simulation.on_config_reload()`, called from `debug_keys.py:55`. The lazy identity check, the table's `.source` and the reloading `species` property are gone; every other reader is pure. Tests T23–T26 | §3, §4.3, §11 |
| R4 | §8's build is cut (export/import, `SwarmSave`, the decoder, v1's T21). Save is handed to chapter 02. New: picklability (T19) and `check_invariants(gmap)` (§2.6, T20) | §2.6, §8, §11 |
| R5 | Species fields are `entities/schema.py::Field`, plus one new kind `KIND_REAL_Q16`. The reload class is a separate mapping. Per-value checks go through `schema.field_value_error`. The hash uses `registry_content_hash`'s recipe over values | §4 |
| R6 | Section, carrier and recorder encode env 0 only, behind an explicit `N_ENV == 1` check (a raise, not a bare `assert`, which `python -O` strips). The bytes carry no `n_env`/`env` tokens. New test T11 | §5, §7 |
| R7 | Id guard `next_unit_id + k <= INT32_MAX` in int64. `faction` must fit int32; heading must fit int32 and is wrapped in int64. bool/float/uint64 inputs raise `TypeError`. Every check runs before any write | §2.2 |
| R8 | 411774 (wrap period) and 411775 (Philox draw range) are distinct on purpose. Every kit-derived heading passes the wrap. The endpoint double weight is accepted. T17's inputs grow | §2.4, §9, §11 |
| R9 | A per-species `gmap.swarm_host_dirty[sp]` flag is set by every facade write. The `_RESIDENT_SYNCED` comment carries the defaulted-`to_host()` caveat | §1.3, §2, §9 |
| R10 | The `free_count()` recommendation is removed, and the query with it: no synced capacity clamp | §2.2, §3 |
| R11 | Species regex `^[a-z][a-z0-9]*$`. Reserved column names. An import-time `check_store_names` asserts every attribute / `.npz` name is unique. New test T21 | §1.1, §11 |
| R12 | `SimState` is untouched, and the doc says so. New free reader `swarm_column(gmap, species, col)` | §3 |
| R13 | The recorder gets a per-tick species-hash ring (int64) and prints once on the first truncation | §7 |
| R14 | The harness locator compares via `swarm_section_bytes`, never `==` on ndarray dicts | §6 |
| R15 | The shared-`next_unit_id` cost is documented against §9b's claim | §1.1, §5.3 |
| R16 | `FACTION_NONE` lives in `factions.py`. `SpeciesDef` is distinguished. The precedent cited is `PROP_MATERIAL_CHOICES`. §9 states P4's `swarm_brood` hand-off | §1.1, §1.2, §9 |
| R17 | Draft rules tightened; `N_ENV` flagged as the first batch-axis constant | §15 |
| R18 | `SPECIES_RULES` → P3. `Simulation.spawn_swarm`/`reclaim_swarm` cut. xarch additions → P3. `swarm_fixed` re-exports `gas_fixed`'s rounding helpers | §3, §4, §6, §13 |
| R19 | §10 adds CLAUDE.md: rows appended after the Deterministic RNG row, the stale "F5 reload" Config row untouched. The heading law is its own appended row, keeping the CLAUDE.md diff append-only. fire-12's `rad_*_sweep` planes are cited as the precedent | §1.3, §10, §13, §15 |

## v2 deviations (resolution → nearest faithful version → evidence)

1. **R7, "heading pre-wrap to int64".** The heading is first range-checked
   to int32, then converted to int64 and wrapped. Evidence: an int64
   `PI_Q16 − h` itself overflows (silently, in numpy) for `h` near
   `INT64_MIN`, the same hazard L1-m5 names for int32. An int32 input bounds
   `|PI_Q16 − h| < 2³²`, so the int64 wrap cannot overflow. Every
   legitimate Q16 heading already fits int32, the store dtype.
2. **R8, "the double weight on direction 0".** Which direction gets the
   double weight depends on the mapping. Probe: the raw draw endpoints 0
   and 411774 both wrap to 0, while `draw − PI_Q16` sends them to −205887
   and +205887, which both wrap to +205887 (π). v2 accepts the bias for
   whichever direction a mapping doubles (2 of 411775 buckets) and states
   both cases (§2.4).
3. **R19, the `cool_shift` → `rad_*_sweep` citation.** The code comment on
   the added `_RESIDENT_SYNCED` line is self-contained and cites neither
   block. Evidence: HEAD's `cool_shift` caveat (`gamemap.py:163-179`) is
   deleted on fire-12, and fire-12's `rad_*_sweep` block (fire-12
   `gamemap.py:133-146`) is not in HEAD's tree when P2 builds. This doc
   cites fire-12's planes as the precedent (§1.3).

---

## 0. Scope

**In:**

- the store (§1);
- spawn/reclaim and the invariant checker (§2);
- the `SwarmUnits` facade, the reload seam and the Simulation hooks (§3);
- the `[swarm.<species>]` table (§4);
- `SWARM_SECT_V1` + carrier + digest fold (§5);
- the A/B harness (§6);
- the recorder (§7);
- picklability, P2's only save duty (§8).

**Out**, each a later patch; only the hand-off constraints P2 imposes
appear, in §9:

- kernel, CPU twin, `SwarmSolver`, sensing, death events, `SPECIES_RULES`,
  the xarch dumper lines (P3);
- `swarm_brood` and level spawning (P4);
- rendering (P5);
- food (P6);
- the save file format (chapter 02's contract).

P2 draws no randomness: spawn takes explicit values. P1 has landed, but P2
makes no Philox call.

---

## 1. The store

### 1.1 Layout: one store per species, one id space per env

Chapter §1 defines a species as "a species table row, **a set of
fixed-capacity SoA per-unit arrays**, and a behaviour kernel". §9b adds
"extensions are per-species columns in the same section, so adding a species
never bumps another species' digest". So **each species owns its own store**,
and all stores share one section. v1 of the roster has one species
(`larva`), but the structure is species-keyed from day one. That costs
nothing now and makes a second species purely additive.

- **The species set is closed in code and append-only:**
  `SWARM_SPECIES = ("larva",)` in `swarm.py`. A species cannot exist without
  a kernel, which is code, so the *set* lives in code and the *numbers* live
  in TOML (§4).
  - The precedent is `PROP_MATERIAL_CHOICES` (`entities/prop.py:32-35`,
    "APPEND-ONLY — enum digests hash the index"). It is not
    `WEAPON_ARCHETYPES`, which is a membership-only frozenset
    (`weapons.py:45-51`).
  - Tuple order = section block order, so the tuple only ever grows at the
    end.
  - Species names match `^[a-z][a-z0-9]*$` (no underscore; R11).
- **Not `simulation/species.py`.** Its `SpeciesDef`/`get_species` are the
  CPU-unit stat-generator species (`"human"`). Swarm species are a separate
  closed set that is never looked up through `get_species`.
- **GameMap attribute names** are `gmap.swarm_<species>_<col>` (e.g.
  `gmap.swarm_larva_pos_x`).
  - They are formed in ONE place, `swarm.store_attr(species, col)`, beside
    `high_water_attr(species)` and the fixed names `NEXT_UNIT_ID_ATTR =
    "swarm_next_unit_id"`, `SPECIES_ATTR = "swarm_species"`,
    `HOST_DIRTY_ATTR = "swarm_host_dirty"` and `SPECIES_HASH_NPZ_KEY =
    "swarm_species_hash"`.
  - The recorder's `.npz` keys *are* these attribute names (§7), so one
    naming function serves both.
  - Attribute names are internal. The wire contract is the section (§5),
    keyed by species name and short column name, so a later rename is
    digest-neutral.
- **Name hygiene (R11).** `check_store_names(species, roster)` runs at
  import on the real tuples. It asserts:
  - every species name matches the regex above;
  - every column name matches `^[a-z][a-z0-9_]*$`;
  - no column uses a reserved name (`RESERVED_COLUMNS = ("high_water",)`,
    which would alias `swarm_<sp>_high_water`);
  - every generated attribute name and `.npz` key is pairwise distinct.

  The last check catches, e.g., a future species `next`, whose
  `store_attr("next", "unit_id")` would equal `swarm_next_unit_id`.
- **Per-env scalars:**
  - `gmap.swarm_<species>_high_water` has shape `(N_ENV,)`, int32, one per
    species (slot spaces are per store).
  - `gmap.swarm_next_unit_id` has shape `(N_ENV,)`, int32, and is **shared
    by all species in an env**.

  Decision: a shared id space makes `unit_id` a true per-env identity across
  species, so the Philox counter `(tick, draw_index, salt, unit_id)`
  (CLAUDE.md "Deterministic RNG" row) never collides between two species
  that reuse a purpose salt.

  **Cost (R15):** once a second species spawns in a run, the shared counter
  interleaves ids, so species A's `unit_id` column and the section's
  `next_unit_id` differ from a run without species B. §9b's "never bumps
  another species' digest" therefore holds for every run in which the new
  species never spawns (its block is omitted, §5.3). That is accepted as
  the price of collision-free RNG streams.
- **`N_ENV = 1`**, a module constant in `swarm.py`: the §A rule-1 batch
  axis, and **the codebase's first batch-axis constant** (no other `N`
  exists in the code; checked). It is the one N until the RL arc's B1 turns
  it into a construction parameter.

### 1.2 The v1 roster (`swarm.ROSTER`, declaration order = section order)

`T_prev` is **absent**: the Review log (2026-09-10) ruled sensing to be a
spatial head-sweep, so the chapter's §2/§4/§4.1/§6 mentions of it are stale.

| col | dtype | meaning | empty-slot value | spawn writes |
|---|---|---|---|---|
| `pos_x` | int32 Q16.16 | x, tile units of the runtime grid | 0 | caller (raw Q16) |
| `pos_y` | int32 Q16.16 | y, tile units | 0 | caller |
| `heading` | int32 Q16.16 | radians, canonical (−π, π] (§2.4) | 0 | `wrap_heading_q16(caller)` |
| `hp` | int32 Q16.16 | hit points | 0 | `gmap.swarm_species.row(sp).hp_max_q` |
| `ai_state` | int32 | `AI_DEAD=0, AI_RUN=1, AI_EAT=2, AI_COMA=3` | 0 (`AI_DEAD`) | `AI_RUN` |
| `dist_walked` | uint32 Q16.16 | path length, wraps mod 2³² | 0 | 0 |
| `unit_id` | int32 | persistent identity (§2.3) | 0 ("no unit") | `next_unit_id` + k |
| `faction` | int32 | present from v1, unread in v1 | 0 | caller; default `factions.FACTION_NONE = -1` |
| `valid` | uint8 | slot occupied | 0 | 1 |

`FACTION_NONE` is defined in `src/simulation/factions.py` beside
`FactionId` (R16). No other faction sentinel exists, and the team ints in
use (0, 1) are non-negative.

**The empty-slot form is the all-zero slot.** `AI_DEAD = 0` plus ids that
start at 1 make every column of an unoccupied slot zero, whether the slot
was never spawned or was reclaimed. Consequences:

- A live unit is never `AI_DEAD` (chapter §7: death sets DEAD and
  `valid = 0` in the same step).
- `np.zeros` *is* the initial state.
- A freed slot physically carries nothing a render husk could be tempted
  to read (critique M2).

The enum values are hashed state, and P3 twins them (§9).

### 1.3 Allocation (the one GameMap hook) and the residency path

`swarm.allocate_swarm_store(gmap, cfg=CFG)` is called once, as the last
statement of `GameMap.__init__`, after `self.refresh_gas_energy()`
(`gamemap.py:825`):

1. `gmap.swarm_species = SwarmSpeciesTable.from_config(cfg)`. This is the
   load-time validation, and it raises on error (§4.2). It plays the same
   role as `self.materials = MaterialTable.from_config(CFG)`
   (`gamemap.py:254`) and `self.gases` (`:261`).
2. `gmap.swarm_next_unit_id = np.full((N_ENV,), FIRST_UNIT_ID, np.int32)`,
   with `FIRST_UNIT_ID = 1`.
3. For each species, `swarm_<sp>_high_water = np.zeros((N_ENV,), np.int32)`.
   For each ROSTER column, `np.zeros((N_ENV, row.max_units), col.dtype)`
   (C-contiguous, native little-endian).
4. `gmap.swarm_host_dirty = {sp: False for sp in SWARM_SPECIES}` (R9). This
   is a plain dict, not an array: not resident, not digested, not recorded.

This runs on every level and every reset (`Simulation._reset_internal`
builds a fresh GameMap, `simulation.py:262`). The store is ~264 KiB at 8,192
larvae, and allocation draws no RNG. Dormancy comes from the presence gate,
not from skipping allocation, so no code ever branches on "does the store
exist".

**Residency (§A rule 3, "born through the residency path from day one").**
`swarm.SWARM_RESIDENT_NAMES` is computed at import from
`SWARM_SPECIES × ROSTER`. It holds every store array: the columns, the
high-waters and `swarm_next_unit_id`. GameMap adds it with one line placed
after the `_RESIDENT_MASKS` literal and before `_RESIDENT_FIELD_NAMES =
...` (`gamemap.py:223`). The comment is self-contained (deviation 3):

```python
    # arc #63 P2 (engine/17 §2): the swarm store — host-written resident arrays.
    # No device kernel reads them in P2, so no per-tick transfer list names them
    # and device copies go stale after any host store write (spawn/reclaim set
    # gmap.swarm_host_dirty[sp]; P3's upload consumes it). Being in
    # _RESIDENT_SYNCED puts them in the DEFAULT to_host() set: a defaulted
    # to_host() would overwrite host writes with stale device data (the resident
    # tick already forbids a defaulted to_host(), physics_runner._step_resident).
    _RESIDENT_SYNCED = _RESIDENT_SYNCED + SWARM_RESIDENT_NAMES
```

The precedent is fire-12's radiation-sweep planes (`rad_net_sweep`,
`rad_flux_sweep`, `rad_amb_sweep`, `rad_fluence`; fire-12
`gamemap.py:133-146`). They are host-written, born resident, and named in
no per-tick list.

The effects:

- `enable_residency` (`gamemap.py:1694-1716`) gives each array a CuPy
  buffer. `cp.asarray` is shape-agnostic.
- `device_ptrs()` (`:1722-1727`) exposes the pointers.
- The `__setattr__` stale-pointer guard (`:225-238`) now forbids
  reassigning any store array while resident. Every P2 write is in place.
- `_RESIDENT_SYNCED` is the default set of `to_host()` (`:1737-1743`),
  hence the caveat in the comment. No defaulted caller exists today.

**No per-tick transfer is added.** The resident tick's lists are explicit
(`physics_runner.py:1187, 1197, 1242, 1299`). No device kernel reads the
store in P2.

### 1.4 What "born `(N, max_units)`" means for code that assumes `(h, w)`

| `(h, w)`-shaped machinery | P2's answer |
|---|---|
| residency alloc/transfer (`gamemap.py:1694-1743`) | Shape-agnostic already. It just works. |
| targeted per-tick transfer lists (`physics_runner.py:1187, 1242, 1299`) | Untouched in P2. P3 names the store arrays in its slot-7 D2H. |
| `DIGEST_FIELDS` + `_field_bytes` (`field_digest.py:79-115`) | The store **never** joins `DIGEST_FIELDS`, because a shape header would hash capacity. It rides the section instead (§5). |
| A/B `SIM_FIELDS` / `_snapshot` / per-cell locator (`field_ab_harness.py:83-96, 142, 378-423`) | Not in `SIM_FIELDS`. It rides the `__swarm__` carrier with its own slot locator (§6). |
| recorder grid rings `(capacity, fh, fw)` (`recorder.py:113-121`) | Never in `fields`. It gets a separate per-unit ring family (§7). |
| `_update_caches` / `on_tile_changed` / `reload_material_table` / `--res` | These operate on grids and never touch the store. |

---

## 2. Lifecycle (chapter §7 made exact)

### 2.1 `high_water`

`high_water[e]` is 1 + the highest slot index ever occupied in env `e`: an
exclusive bound, where 0 means nothing was ever spawned.

- It is monotone non-decreasing within a run. Spawn may raise it, reclaim
  never lowers it (no compaction), and only a fresh GameMap (reset / level
  load) returns it to 0.
- The digested and recorded extent is `[0, high_water)`, so capacity moves
  no byte.

### 2.2 Spawn: the k-th request takes the k-th lowest free slot

`SwarmUnits.spawn(species, pos_x_q, pos_y_q, heading_q, *,
faction=FACTION_NONE, env=0)` returns the slot indices (int64 array, in
request order).

**Slot choice.** A batch of k requests, in request order, occupies the k
lowest slots with `valid == 0`, ascending.

- On a fresh store that is sequential fill `0..k-1`.
- After reclaims, holes are reused lowest-first before `high_water`
  extends.
- This is exactly chapter §7's in-sim-birth rule ("k-th request pairs with
  k-th free slot via prefix-sum"), so the host spawn and a later device
  birth agree by construction.
- It is not an atomic ticket (Safari 2022), not LIFO, and not
  append-at-high-water.

**The write.** Spawn writes the "spawn writes" column of §1.2.

- Unit ids are `next_unit_id, next_unit_id+1, …` in request order.
- Then `next_unit_id += k` and `high_water = max(high_water, max(slots)+1)`.
- Then `gmap.swarm_host_dirty[species] = True` (R9).

**When full: atomic refusal.** If k exceeds the free slots, spawn raises
`swarm.SwarmFull` (a `ValueError` subclass; the `SealBlocked` precedent,
`gamemap.py:96-105`) and writes **nothing**. Decision: partial fill would
make "how many larvae exist" depend on `max_units`, a knob that must move no
behaviour.

**Synced callers never clamp by capacity (R10).** An overflow is
`SwarmFull`, which is an authoring error. If P4 ever needs a clamp, it is
an authored species-row cap (a `SPECIES_FIELDS` row), never a read of the
capacity. The facade offers no `free_count()`, because such a query would
exist only to enable that clamp.

**Validation.** All of it runs before any write, which is what makes the
refusal atomic.

1. `species ∈ SWARM_SPECIES`, else `ValueError`. `env` must be an int in
   `[0, N_ENV)`, else `ValueError`.
2. Each of `pos_x_q`, `pos_y_q`, `heading_q` and `faction` goes through
   `np.asarray`.
   - Its dtype must be a signed int (any width) or an unsigned int up to
     uint32.
   - `bool`, float, uint64, object and str inputs are a `TypeError` (R7).
     Raw Q16 only: the caller converts through `swarm_fixed` (door 2).
     An `OverflowError` from converting an over-large Python int is
     re-raised as `TypeError`.
   - Each input is converted to int64 and broadcast to one 1-D length k. An
     input with more than one dimension is a `ValueError`. k = 0 is a no-op
     that returns an empty int64 array and writes nothing, not even the
     dirty flag.
3. Range checks, in int64 (R7):
   - `0 <= pos_x < gmap._w << FP_SHIFT` and `0 <= pos_y < gmap._h <<
     FP_SHIFT` (`gamemap.py:248`);
   - `heading ∈ [INT32_MIN, INT32_MAX]` (deviation 1);
   - `faction ∈ [INT32_MIN, INT32_MAX]`;
   - `int(next_unit_id[env]) + k <= INT32_MAX`, so the counter never
     exceeds `INT32_MAX` and the last assignable id is `INT32_MAX − 1`.
4. `k <= (valid[env] == 0).sum()`, else `SwarmFull`.

Terrain is **not** checked. Placement is the caller's job (P4 scatters on
walkable tiles), and the store stays a data structure.

### 2.3 `unit_id` / `next_unit_id`

- Ids start at `FIRST_UNIT_ID = 1`; 0 is "no unit", the empty-slot value.
- They are strictly increasing in assignment order and never reused. A
  reused slot gets a new id. Reclaim never decrements the counter.
- `next_unit_id` is in the section, because reassigning ids after a load
  would fork every RNG stream (chapter §6).

### 2.4 Heading is canonical at every write

`swarm_fixed.wrap_heading_q16(h) = PI_Q16 − ((PI_Q16 − h) mod
HEADING_PERIOD_Q16)`, with floor-mod computed in **int64**.

- `PI_Q16 = 205887 == quantize(π)`, the kit's narrowed π
  (`fixed_point.h:900-903`).
- **`HEADING_PERIOD_Q16 = 2·PI_Q16 = 411774`**, not `quantize(2π) =
  411775`. Only a period of exactly twice the half-range makes (−P, P] a
  complete residue system, i.e. one representation per direction. At Q16
  the π constants are inconsistent (`fixed_point.h:890-894`), so this
  choice is forced, not free.

Probe-verified (NumPy 2.4.6, Python 3.11.7):

- `wrap(±P) = P` and `wrap(±3P) = P`;
- `wrap(411774) = 0` and `wrap(411775) = 1`;
- `wrap(INT32_MAX) = 82237` and `wrap(INT32_MIN) = −82238`;
- `wrap` is idempotent.

The int64 is load-bearing: an int32 `PI_Q16 − h` wraps silently in numpy
(probe: `P − INT32_MIN` gives `−2147277761` in int32 and `2147689535` in
int64). Spawn stores `wrap_heading_q16(int64(heading))` cast to int32.

**Kit angle sources (R8).**

- `philox32.TWO_PI_Q16 = 411775` is the **draw range**:
  `philox32_angle_q16` returns 0..411774 (`philox32.py:52, 103-109`).
  `HEADING_PERIOD_Q16 = 411774` is the **wrap period**. They are distinct
  on purpose, and they combine as draw-then-wrap. Nobody "unifies" them.
- Every kit-derived heading passes `wrap_heading_q16` before any store
  write. That covers an `atan2_q16` output (which can be −P, e.g.
  `atan2_q16(−1, −6553600) = −205887`), a Philox draw, and `draw − PI_Q16`.
- The two endpoint draws 0 and 411774 are congruent mod 411774, so one
  direction owns 2 of the 411775 buckets. That is direction 0 for a raw
  draw and direction π for `draw − PI_Q16` (deviation 2). This is an
  **accepted bias** (relative excess ≈ 2.4·10⁻⁶).

P3's C++ twin must use a floor-mod in int64, because C++ `%` truncates
(§9).

### 2.5 Reclaim

`reclaim(species, slots, *, env=0)` writes the empty-slot form (every column
zero) at each slot, then sets `gmap.swarm_host_dirty[species] = True`.
`high_water` and `next_unit_id` are unchanged.

**Validation** (atomic, before any write):

- `slots` is an integer array-like; bool or float is a `TypeError`.
- Every slot is in `[0, high_water)`, the slots are unique, and each has
  `valid == 1`.

Reclaiming an empty slot is a caller bug and raises `ValueError`. An empty
`slots` is a no-op.

The death **tick event** (chapter §7) is P3's, because it is behaviour.
Reclaim is the slot-freeing primitive and **defines the freed-slot form
that P3's death branch must reproduce** (§9).

### 2.6 The invariant checker (R4)

`swarm.check_invariants(gmap) -> list[str]` is a pure reader. It returns
human-readable violations, and an empty list means every invariant holds.
Per env and species:

1. `0 <= high_water <= max_units` (the array width).
2. `FIRST_UNIT_ID <= next_unit_id <= INT32_MAX`, and
   `next_unit_id == FIRST_UNIT_ID` ⇔ every species' `high_water == 0` in
   that env.
3. `valid ∈ {0, 1}` everywhere.
4. Every slot with `valid == 0`, freed or never spawned, at any index, is
   all-zero in every column.
5. Every slot with `valid == 1` lies in `[0, high_water)`, has
   `FIRST_UNIT_ID <= unit_id < next_unit_id`, has
   `ai_state ∈ {AI_RUN, AI_EAT, AI_COMA}`, and has heading ∈ (−PI_Q16,
   PI_Q16].
6. Live `unit_id`s are unique across **all** species in the env (the shared
   id space).

P2's lifecycle tests call it after every operation (T14). P3's gates reuse
it, and a future chapter-02 loader should pass it after restoring a store.

---

## 3. The facade, the reload seam, and how Simulation drives them

`swarm.SwarmUnits(gmap)` is the one small class chapter §9b allows ("the
only 'class' in the design"). It holds no arrays: GameMap stores them
(fork 2a), and the facade reads and writes `gmap` in place.

| member | role |
|---|---|
| `spawn(species, pos_x_q, pos_y_q, heading_q, *, faction=FACTION_NONE, env=0)` | §2.2 |
| `reclaim(species, slots, *, env=0)` | §2.5 |
| `live_count(sp, env=0)`, `high_water(sp, env=0)`, `present()` | Queries. |
| `column(sp, col)` | Delegates to the free `swarm_column(gmap, sp, col)`. |
| `species` (read-only property) | Returns `gmap.swarm_species`. It **never** reloads. |
| `carrier()` | Returns `swarm_carrier(self.gmap)` (§5.2). Pure. |

**Free functions** (module-level, taking `gmap`). P3's PhysicsRunner, P4
and P5 hold `gmap`, not `sim.swarm` (R12):

- `swarm_column(gmap, species, col)` returns the live gmap array. It is the
  read accessor, so no caller formats attribute names.
- `swarm_present(gmap)`
- `swarm_carrier(gmap)`
- `check_invariants(gmap)`
- `reload_species(gmap, cfg=CFG)` (§4.3)

**Simulation hooks (`simulation.py`, all thin; the tick conductor gets NO
new line).**

1. **Import** `SwarmUnits` and `reload_species` beside the system imports,
   after `from simulation.vent_system import build_vents, sweep_vents`
   (`:100`).
2. **Reset.** In `_reset_internal`, right after `self._seed = seed`
   (`:264`), add `self.swarm = SwarmUnits(self.gmap)`. A reset rebuilds the
   facade with the fresh GameMap, so "reset" is structural and needs no
   method.
3. **The reload seam (R3).** Add a public method, placed after `reset()`
   (`:249-258`) and before `_reset_internal`:

   ```python
   def on_config_reload(self) -> None:
       """THE config-reload seam: the Ctrl+R path calls this right after
       CFG.reload() (src/debug_keys.py). Re-binds every live table from the
       fresh CFG; applies from the next tick. Today: the swarm species table
       (arc #63 P2). Chapter 12's material/physics re-binds may join here."""
       reload_species(self.gmap)
   ```

   In `src/debug_keys.py`, inside the Ctrl+R `if` block (`:54-55`), add one
   line after `CFG.reload()`:

   ```python
       if ctrl_held and rl.is_key_pressed(K.KEY_R):
           CFG.reload()
           sim.on_config_reload()   # arc #63 P2: the one reload seam
   ```

   `debug_keys.py:55` is the only `CFG.reload()` call site in `src/`,
   `tools/` and `main.py` (grep). Its callers (`control_onephase.py:136`,
   `input_handler.py:121`) always pass a `Simulation`.
4. **Recorder kwargs.** At recorder construction (`:405-409`) add
   `swarm_units_cap=CFG.recorder.swarm_units_cap`. On the existing
   `record(...)` call (`:1582-1584`) add `swarm=self.swarm.carrier()`.

**Cut (R18): `Simulation.spawn_swarm` / `reclaim_swarm`.** They would have
no non-test caller. `sim.swarm` is the handle, and P4's load-time brood
calls `self.swarm.spawn` inside `_reset_internal`. Delegating methods are
added when a gameplay or tool caller first exists.

**No tick line.** P2 has no behaviour, so nothing happens per tick. The only
per-tick touch is a kwarg on an existing recorder line (host debug tooling;
§A rule 2 holds).

**`SimState` is untouched (R12).** `get_state()` is not a swarm capture path
in P2. The P5 renderer reads the arrays through
`swarm_column(state.gmap, …)`, which works because `SimState` already
references `gmap`. When a state consumer needs swarm state as data, the
additive route is a defaulted `swarm_state` field (the `entity_state`
precedent, `simulation.py:174`).

---

## 4. The species table (`[swarm.<species>]`)

### 4.1 Schema in code (`entities/schema.py::Field`), numbers in TOML

**R5: the descriptor is reused, not re-invented.** `swarm.SPECIES_FIELDS`
is a tuple of `simulation.entities.schema.Field` (`schema.py:104-120`, a
stdlib frozen dataclass). Every species field has `default=None`, which is
Field's documented meaning of REQUIRED, so a missing key is an error.

- **`KIND_INT`** (existing): a TOML int. `field_value_error` rejects bool
  and float. It is stored as-is under attribute `<name>`.
- **`KIND_REAL_Q16`** (NEW in `schema.py`, R5): "authored real, quantized
  to Q16.16 ONCE at load by its consumer", the `KIND_LENGTH_M` precedent.
  - Its `field_value_error` branch accepts an int or float, not bool, and
    rejects a non-finite float (`nan`/`inf`, which TOML can spell).
  - The swarm table stores it under `<name>_q` =
    `swarm_fixed.quantize_scalar(value)` (door 2).
  - It is unused by entities today. It joins `ALL_KINDS` only: not
    `NUMERIC_KINDS`, not `serialize.SYNCED_FIELD_KINDS`, not the editor's
    kind mirror (`tools/entity_editor_ui.py:72-89`). Adopting it for
    entities is a later decision that updates those three.
- **`minimum`/`maximum`** are inclusive bounds on the **authored** value
  (`field_value_error`'s rule, `schema.py:286-289`). For a `KIND_REAL_Q16`
  field they are chosen so the stored value lands in its intended raw range.
- **The reload class** is a separate mapping, `swarm.SPECIES_RELOAD:
  dict[str, str]` with values in `{"hot", "load"}`. An import-time
  assertion checks that its keys equal the `SPECIES_FIELDS` names.
- **`SwarmSpeciesRow`** is a **module-level `@dataclass(frozen=True)`**
  with fields `species: str`, `max_units: int`, `hp_max_q: int`. It is
  picklable because it is module-level and statically declared (R4). An
  import-time assertion checks that its field names equal `{"species"}` ∪
  the stored attributes of `SPECIES_FIELDS`, so a new `SPECIES_FIELDS` row
  without a matching dataclass field, or the reverse, fails at import.
- **`SPECIES_RULES` is deferred to P3 (R18).** P3 creates it together with
  its first rule, `speed·Δt < 1 tile` (chapter §2).

**P2's fields (only what P2 consumes):**

| key | `Field` | reload | stored as | P2 consumer |
|---|---|---|---|---|
| `max_units` | `Field("max_units", KIND_INT, minimum=1, maximum=1 << 20)` | **load** | `max_units` | allocation |
| `hp_max` | `Field("hp_max", KIND_REAL_Q16, minimum=2**-16, maximum=32767.0)` | hot | `hp_max_q` ∈ [1, 2147418112] | spawn writes `hp` |

`2**-16` and `32767.0` are exact binary floats. Their quantized values are
1 and `32767·65536 = 2147418112 < INT32_MAX`.

`max_units` sits in the species row because the store is per species
(§1.1). A moth swarm and a fish school will want different capacities.

**Deferred to P3**, which adds them as `SPECIES_FIELDS` rows plus TOML
values:

- `speed` (+ the `speed·Δt` rule);
- `radius`;
- the kill/coma thresholds, authored in Kelvin per the plan: `T_ctmin`,
  `T_coma_hyst`, `T_lethal`, `T_lethal_hot`;
- the head-sweep sensing and tumble dials.

**Deferred to P6:** `food_threshold`, `eat_start_prob`,
`eat_continue_prob`, `eat_rate`.

The chapter's `tumble_thermal_bias` and `heat_damage_coeff` are superseded
by the head-sweep and threshold rulings. P3 names what replaces them.

`config.toml` gets this block, appended at EOF after `[recorder]`:

```toml
[swarm.larva]
# Swarm species row (engine/17 §2, arc #63). Schema lives in code
# (simulation/swarm.py SPECIES_FIELDS + SPECIES_RELOAD: kind, reload class,
# bounds); an unknown key or species here is a load error. Hot on Ctrl+R
# (applied by Simulation.on_config_reload from the next tick) unless marked.
max_units = 8192    # LOAD-TIME: slots per env (fork 2b). A Ctrl+R edit applies at the next reset/level load.
hp_max    = 1.0     # hit points at spawn. Placeholder: v1 kills by threshold (fork 7b), hp is roster state for future targeting.
```

### 4.2 Validation (`SwarmSpeciesTable.from_config(cfg)`)

It raises `ValueError` naming `[swarm.<sp>].<key>` when:

- `[swarm]` is missing;
- a species in `SWARM_SPECIES` has no row;
- a Namespace under `[swarm]` names no code species (typo guard);
- a non-table key sits directly under `[swarm]` (P2 has no global swarm
  keys);
- a row has an unknown key or is missing a key;
- `schema.field_value_error(field, value)` (`schema.py:227`) returns an
  error, i.e. wrong type or out of bounds.

The structure checks mirror `apply_tuning_overlay`'s strictness
(`entities/registry.py:53-111`): unknown class or field is a hard error, and
there is no schema in TOML. The per-value check is the same function the
entity loader uses. This closes, for this table, the "no schema, typos
surface as AttributeError or silent `getattr` defaults" gap chapter 12
records. The validator is the candidate shared strict-config-table helper
(§15b); it is not extracted in P2.

**The table** is a module-level `@dataclass(frozen=True) class
SwarmSpeciesTable` with:

- `rows: tuple[SwarmSpeciesRow, ...]` in `SWARM_SPECIES` order;
- `hash: str`;
- `from_rows(rows)` (computes the hash);
- `from_config(cfg)` (validates, quantizes, then calls `from_rows`);
- `row(name)`.

There is no `.source` attribute: its only use was v1's lazy identity check,
which R3 removed.

**`.hash`** is SHA-256 hex over

```python
json.dumps({"format": "SWARM_SPECIES_V1",
            "species": {sp: {attr: int(value) for attr, value in row_values}}},
           sort_keys=True, separators=(",", ":"), ensure_ascii=True)
```

This is `registry_content_hash`'s recipe (`registry.py:171-181`) over a
different payload. The function itself does not fit: it walks Entity
classes. The payloads differ for a reason:

- The registry hash covers schema + effective defaults, because the
  registry IS the schema, shipped as the editor's JSON fallback.
- The species hash covers only the **numbers in force** (post-quantization,
  post-load-field retention, §4.3), because the swarm schema is code, and
  "comparable at equal `species_hash`" is a statement about tuning. Two TOML
  spellings of the same numbers (`1.0` vs `1`) hash equal.

### 4.3 Hot reload (fork 2c) through the one seam (R3)

In this codebase a reload is Ctrl+R → `CFG.reload()` (`debug_keys.py:53-55`).
F5 is the renderer's normal-map toggle (`renderer/game_renderer.py:1502`),
despite the chapter's and CLAUDE.md's "F5" wording (§12 item 2).

`CFG._load` re-wraps every section present in the file in a new Namespace
(`config.py:57-66`). The material, gas and weapon tables are
construction-bound and do not follow it (`weapons.py:31-36`; chapter 12
§5).

The swarm table follows through the explicit seam:
`debug_keys` → `Simulation.on_config_reload()` → `swarm.reload_species(gmap,
cfg=CFG) -> bool`:

1. **Build.** Build the candidate with `SwarmSpeciesTable.from_config(cfg)`,
   with full validation. On `ValueError`, print ONE line, `[swarm] RELOAD
   REJECTED: <error>`, keep the table in force, and return `False`. The
   rejection is loud per reload event and never a silent fallback: the old
   table was valid, and the message names the key.
2. **Load-class fields keep their in-force value.** For each species and
   `load` key whose TOML value differs from the in-force one, print exactly
   one line per reload event, e.g. `[swarm] larva.max_units 8192 -> 4096
   applies at the next reset/level load`. Then rebuild the candidate with
   the in-force values (`from_rows`), so its hash describes what is
   actually in force. Arrays are never reallocated mid-run, and the
   residency guard would forbid it anyway.
3. **Install.** `gmap.swarm_species = candidate`. **This line and allocation
   (§1.3) are the only writers of `gmap.swarm_species`.**
4. **Log a hash change.** If `.hash` changed, print `[swarm] species table
   reloaded: <old12> -> <new12> (replay comparability ends here)`. This is
   chapter §2c's "reload invalidates the replay from that tick". No replay
   machinery exists yet, so the log line is the record.
5. Return `True`.

**It applies from the next tick.** The seam runs in input handling, between
ticks.

**Everyone else is a pure reader** of `gmap.swarm_species`: spawn,
`carrier()`, `swarm_carrier`, the recorder, and P3's H2D (§9). None of them
reloads. So recorder-on and recorder-off runs, harness captures and RL
rollouts see the same table at every tick (T24).

Decision (unchanged from v1): a reload-time failure is rejected, while a
load-time failure raises. Fork 2c chose hot reload *for tuning*, and a
crashed play session on a Ctrl+R typo is the most tuning-hostile outcome.

**Known limit (pre-existing CFG behaviour).** `_load` only sets sections
present in the file, so deleting `[swarm]` mid-run leaves the stale
Namespace on `CFG` and the seam rebuilds the old numbers. Load-time
validation catches the deletion at the next reset.

**Which settings apply when:**

| setting | applies |
|---|---|
| `[swarm.*].hp_max` | Hot: from the tick after the Ctrl+R. The next spawn uses it; live units' `hp` is untouched. |
| `[swarm.*].max_units` | Load-time (next reset/level load). |
| `[recorder].swarm_units_cap` | Reset-bound: the recorder is built in `_reset_internal`. |
| `N_ENV`, `SWARM_SPECIES`, `ROSTER` | Code. |

A reload writes no store array, so **the species table never touches
digested state**. `species_hash` is provenance in the carrier and is never
hashed into the section (§5.3), the same as the entity `registry_hash`.

### 4.4 `swarm_fixed.py` (the Q16 boundary module, ruled name)

R18: no seventh hand-copy of the rounding helpers. Survey §E flag 6
(`docs/canonical_systems_survey_2026-08-22.md`) counts six byte-identical
copies. `swarm_fixed.py` re-exports them from `gas_fixed`, the existing
cross-module source (`temperature_scale.py:33` already imports
`gas_fixed.quantize_scalar`):

```python
from simulation.gas_fixed import (  # noqa: F401  re-export — survey §E flag 6: no 7th copy
    FP_SHIFT, FP_ONE, FP_ONE_F, quantize, quantize_scalar, dequantize, dequantize_f32,
)
```

It adds only what no module has:

- `PI_Q16 = 205887`, a checked-in constant. No Python `PI_Q16` exists: the
  only Python copy is the test-local `PI_Q` at `tests/test_fixed_trig.py:53`.
- `HEADING_PERIOD_Q16 = 2 * PI_Q16`.
- `wrap_heading_q16(h)`: a scalar int in gives an int out; an ndarray in
  gives an int64 ndarray out; the arithmetic is always int64.

It has no `breach_physics` import and no transcendental (π is a checked-in
constant, which is door 2). Nothing else hardcodes 65536.

---

## 5. `SWARM_SECT_V1`, the carrier, and the digest fold

### 5.1 "Present"

`present ⇔ ∃ species s : gmap.swarm_<s>_high_water[0] > 0` (env 0; §5.2).
That means at least one spawn ever succeeded on this GameMap (this run,
since the last reset/level load). This is chapter §6's "any slot ever
spawned this run".

Because `high_water` is monotone, **presence is monotone within a run**. A
run whose larvae have all died still hashes its section (with `valid = 0`
rows) and never falls back to the swarm-free hash.

### 5.2 The carrier (the `__entity__` idiom, `serialize.py:69, 200-241`)

The key is `SWARM_DIGEST_KEY = "__swarm__"`. `swarm_carrier(gmap)` returns
copies, never views, of **env 0 only**:

```python
{"present": bool,
 "next_unit_id": int,                                   # env 0, Python int
 "blocks": {sp: {"high_water": int,                     # Python int
                 "columns": {col: ndarray (hw,) roster dtype}}   # env 0, [0, hw)
            for sp in SWARM_SPECIES if high_water > 0},             # SWARM_SPECIES order
 "species_hash": str}                                   # provenance, NEVER hashed
```

- **R6.** `swarm_carrier` begins with `if N_ENV != 1: raise
  NotImplementedError("multi-env swarm section is the RL arc's B1
  decision")`. This is an explicit raise, not a bare `assert`. The store
  stays `(N_ENV, max_units)`; only the determinism artifacts are env-0-only.
  Keeping multi-env structure out of the bytes keeps "env k with seed s ==
  an N=1 run with seed s" reachable for B1's golden.
- **R2.** Every integer in the carrier is `int()`-converted at build.
- **Stand-in rule.** A gmap without `swarm_species` (a test stand-in)
  yields `{"present": False, "next_unit_id": FIRST_UNIT_ID, "blocks": {},
  "species_hash": ""}`.
- **Strict capture.** Capture paths **always** write the carrier (§6).
  `require_swarm_carrier(gmap, snapshot)` raises `KeyError` when a
  swarm-present gmap meets a snapshot without `__swarm__` (the A4 strict
  rule). Snapshots from before P2 are swarm-free by construction, and §6's
  diff reads their missing key as an absent carrier (R1).

### 5.3 Byte encoding — `swarm_section_bytes(carrier)`, the ONE encoder

```
section := "SWARM_SECT_V1\n" "next_unit_id|" nid "\n" block*
           -- blocks in SWARM_SPECIES order; a species with hw == 0 is OMITTED
block   := "species|" name "|high_water|" hw "\n" column{ROSTER}
           -- columns in ROSTER declaration order
column  := col "|" dtype.str "|(" hw ",)\n" raw
           -- raw = np.ascontiguousarray(arr[:hw]).tobytes(order="C"); big-endian dtype refused
```

- **R2.** Every integer token (`nid`, `hw`) is written as `str(int(x))`
  inside the encoder, even though the carrier already holds ints. A
  hand-built carrier with numpy scalars must encode identically. `str()` of
  a tuple holding numpy scalars is NumPy-version-dependent
  (`str((np.int32(5),))` is `'(np.int32(5),)'` on NumPy 2.4.6 and `'(5,)'`
  on 1.26), so the header never goes through tuple formatting.
- The header pins name, dtype and length (the `_field_bytes` idiom,
  `field_digest.py:108-115`), so parsing is unambiguous. No terminator is
  needed: the section is hashed alone, and there is no save format to
  delimit (§8).
- **Never in the bytes:** capacity, the species table, `species_hash`,
  slots at or beyond `high_water`, and any env other than 0.

Omitting hw-0 species blocks is what makes "adding a species never bumps
another species' digest" (§9b) true for every run in which the added
species never spawns. §1.1 states the shared-counter cost in runs where it
does (R15).

**The encoder rule (R4):** the digest hashes the one encoder's bytes; the
recorder stores the same columns raw; a test (T27) gates them equal by
re-encoding recorded rows. That is A4's one-serializer principle
(`serialize.py:1-9, 164`), realized with numpy in the swarm's own module.

The roster's own version rule (chapter §2: "additions bump the section
version") bumps the preamble to `SWARM_SECT_V2`, which is section-local.

### 5.4 The fold (`tests/field_digest.py::tick_digest`, `:137-176`)

Append after the entity fold (`:166-175`):

```python
    from simulation.swarm import SWARM_DIGEST_KEY, swarm_section_bytes
    sw = snapshot.get(SWARM_DIGEST_KEY)
    if sw is not None and sw["present"]:
        wh = hashlib.blake2b(swarm_section_bytes(sw), digest_size=32).hexdigest()
        h.update(b"|__swarm__|")
        h.update(wh.encode("ascii"))
```

The hashed stream becomes
`fd | unit_hash [|__entity__| eh |__signals__| sh] [|__swarm__| wh]`. The
digests are fixed-length hex and the markers are distinct non-hex tokens, so
no swarm-present stream collides with an entity-present one.
`field_digest()` and `trajectory_digest()` are unchanged.

### 5.5 Why no `DIGEST_SPEC_VERSION` bump — and why every golden stays byte-identical

`DIGEST_SPEC_VERSION` salts the field-plane stream: the preambles
`FIELD_DIGEST_V{n}` (`field_digest.py:121`), `PERFIELD_V{n}` and
`TRAJ_DIGEST_V{n}`. Its change procedure governs **`DIGEST_FIELDS`
membership, shape, dtype and order** (`field_digest.py:40-42`;
`field_digest_spec.toml:6-9`).

The swarm section is not a `DIGEST_FIELDS` row. It is folded only when
present, and it carries its own hashed version (`SWARM_SECT_V1`). This is
the A4 precedent: "the ENTITY/SIGNAL sections do NOT drive
DIGEST_SPEC_VERSION" (`field_digest.py:150-153`).

For a swarm-free snapshot the fold contributes **zero bytes**, so the hashed
stream is literally the pre-P2 stream. Every existing golden is swarm-free:

- `GOLDEN_AGGREGATE` (`_xarch_perfield_digest.py:225`, pinned by
  `test_w6_armory.py:566-588`);
- the B6 logic golden;
- the vent/B1/B2 digests.

So all of them stay byte-identical with no regeneration.

Honest limits of that claim:

- A swarm-present trajectory is comparable across builds only at equal
  section version, and for behaviour patches only at equal `species_hash`
  (recorded, like the entity registry hash).
- The food field (P6) **is** a `DIGEST_FIELDS` row and does take a spec
  bump (§12 item 3).

The spec toml gets an additive, documentary `[swarm_section]` table appended
at EOF, mirroring `[entity_section]` (`field_digest_spec.toml:77-92`). Its
`version` stays unchanged. The table records:

- `format = "SWARM_SECT_V1"`;
- the encoder, `simulation.swarm.swarm_section_bytes`;
- the extent: env 0, `[0, high_water)` per species, with hw-0 species
  blocks omitted, and the multi-env fold left to the RL arc's B1;
- that every integer is written as ASCII decimal via `int()`;
- that capacity and the species table are never hashed;
- provenance = `species_hash`;
- the rule "per-unit swarm state never joins `fields`".

---

## 6. A/B harness (`tests/field_ab_harness.py`)

**Capture** (`capture_trajectory`, `:268-304`). After
`require_entity_carrier(ents, snap)` (`:302`), add:

```python
        snap[SWARM_DIGEST_KEY] = swarm_carrier(sim.gmap)
        require_swarm_carrier(sim.gmap, snap)
```

The carrier is always written; on the canonical scenario it has
`present=False`.

**Diff (R1).** In `diff_trajectories` (`:378-423`), the swarm case is the
FIRST statement in the per-key loop, BEFORE the one-sided-key check at
`:391-393`:

```python
        for k in sorted(set(sa) | set(sb)):
            if k == SWARM_DIGEST_KEY:   # before the one-sided check: a pre-P2 snapshot has no key
                diffs.extend(_diff_swarm_state(t, sa.get(k), sb.get(k)))
                continue
            if k not in sa or k not in sb:
                ...
```

Without this, a pickled pre-P2 trajectory goes red against any fresh capture
("field '__swarm__' present in only one run"). The case that exists today is
`tests/_pg5_base_trajectory_45050f3.pkl`, loaded by
`test_thermostat_books.py:153-167`. The same applies to every
`save_trajectory`/`load_trajectory` A/B across the P2 boundary. Probe: the
pickle's snapshot keys are the 25 fields + `__entity__` + `__unit_state__`,
with no `__swarm__`.

**Locator (R14).** Add `_diff_swarm_state(t, ca, cb)` beside
`_diff_entity_state` (`:346-375`):

- `None` (a pre-P2 snapshot) reads as an absent carrier.
- If neither side is present (absent or `present=False`), return `[]`.
- If exactly one side is present, return one line: `tick t: __swarm__
  present in only one run`.
- If both are present, the fast path is
  `swarm_section_bytes(ca) == swarm_section_bytes(cb)` and equal
  `species_hash`, which returns `[]`. **Never `ca == cb`**: the carriers
  hold ndarrays, so `==` raises "truth value … is ambiguous". The entity
  fast path at `:353` must not be copied.
- Otherwise report:
  - a `next_unit_id` mismatch;
  - per species, a `high_water` mismatch or a block present on one side
    only;
  - per column, the first differing slot over the common prefix
    (`np.flatnonzero(a != b)`), with both values and both `unit_id`s;
  - a `species_hash` mismatch (provenance, but still reported, since A/B
    runs must share tuning).

  If nothing is located, report `… serialization drift?` (the
  `_diff_entity_state` pattern).

**Scenario.** Add `swarm_scenario_sim()` after `default_scenario_sim`
(`:109-139`). It takes the default scenario and, before the first step,
uses explicit raw-Q16 values from `swarm_fixed` (no RNG) to:

1. `sim.swarm.spawn` 12 larvae at interior tile centres
   (`(tx << FP_SHIFT) + (FP_ONE >> 1)`, interior = tiles 1..14 of
   `_scenario_level`'s 16×16 map), with fixed headings including `±PI_Q16`;
2. reclaim 3 of them, making holes;
3. spawn 2 more, which fill the holes.

The result has `high_water` > live count, non-contiguous ids and
`check_invariants(sim.gmap) == []`. It is the swarm-present fixture for P2's
tests and a ready scenario for P3's lockstep.

**Unchanged and cut:**

- `SIM_FIELDS` is unchanged. In `__main__`, add `SWARM_DIGEST_KEY` to the
  special-key tuple at `:451`.
- **The xarch dumpers are deferred to P3 (R18):** no edit to
  `_xarch_perfield_digest.py` or `xarch_digest.py`. Their runners capture
  only the swarm-free canonical scenario (`xarch_digest.py:50`), so a swarm
  line could not appear before P3's lockstep scenario exists.
  `GOLDEN_AGGREGATE` is untouched.

---

## 7. Recorder (`src/simulation/recorder.py`)

**Which arrays:** every `ROSTER` column of every present species, plus the
per-tick `high_water`, `next_unit_id` and species hash. Values are **raw, at
the roster dtype — no dequantize, no cast** (chapter §6). This is a new
ring family, separate from the grid `fields`. Like the carrier it feeds on,
it records **env 0 only** (R6).

**API:**

- `__init__(fh, fw, capacity=1200, fields=None, swarm_units_cap=1024)`.
- `record(..., swarm=None)`, where `swarm` is the §5.2 carrier. The digest
  and the recorder consume the same carrier structure. `None` is treated as
  absent.
- The `_FakeGmap` recorder tests are unaffected, because the recorder never
  reads `gmap.swarm_*`.

**Rings.** They are allocated **lazily at the first present tick**, with
`W = swarm_units_cap`:

- `ring[sp][col] = zeros((capacity, W), col.dtype)` per present species;
- `hw_ring[sp] = zeros((capacity,), int32)`;
- one `nid_ring = full((capacity,), FIRST_UNIT_ID, int32)` (earlier rows are
  truthfully "nothing spawned");
- one `shash_ring = zeros((capacity,), int64)` (R13), holding
  `int.from_bytes(bytes.fromhex(species_hash)[:8], "little", signed=True)`.
  int64 is an existing ring dtype class (`_INT64_FIELDS`, `recorder.py:94`).
  0 = "no swarm this tick".

Allocation prints its size.

**Writes.** Once the rings exist, **every** tick writes (at the same ring
index `i` as the grid rings):

- For each present species: `w = min(hw, W)`, `ring[i, :w] = column[:w]`,
  `ring[i, w:] = 0`, `hw_ring[i] = hw`.
- A species absent this tick writes zeros and `hw 0`.
- `nid_ring[i]` and `shash_ring[i]` are always written.

**Truncation.** The first tick with `hw > W` prints exactly one line per
recorder instance: `[recorder] swarm <sp> high_water <hw> >
swarm_units_cap <W>: recording truncated to the first <W> slots`.

**Dump.** It is presence-gated: if no ring exists, there are no `swarm_*`
keys and the key set is identical to today (the entity precedent,
`recorder.py:281-294`). Keys are the §1.1 attribute names:

- `swarm_<sp>_<col>`: `(T, wmax)` at the roster dtype, with `wmax = min(W,
  max recorded hw)`. That is the `[0, high_water)` bounding.
- `swarm_<sp>_high_water`: `(T,)` int32.
- `swarm_next_unit_id`: `(T,)` int32.
- `swarm_species_hash`: `(T,)` int64, per tick, so a reload inside the
  window is visible at its tick.

Truncation is also visible offline when `high_water > wmax`. Update the
module docstring's frozen-schema list with these additive keys.

**Dtype classes (the CLAUDE.md recorder rule).** Today the rings are
**grid** rings of bool, int64 or float32; int32 grids are dequantized into
float32 (`recorder.py:113-121`; the dequantize at `:190-194`). In the `.npz`:

- **uint8 (`valid`) and uint32 (`dist_walked`) are NEW dtype classes.** No
  key carries them today.
- **int32 is new as a ring dtype.** It exists in the file only as dump-time
  `unit_fx`/`tick_ids`.

The rule ("a contract extension with its own ring branch, never a cast") is
met **by construction**: the swarm family's one branch takes each ring's
dtype verbatim from `ROSTER`, so no column can be cast. T27 asserts
`recorded.dtype == roster dtype` for every column.

**The cap (chapter §2b "capped by config").** `[recorder] swarm_units_cap =
1024` is added to `config.toml:2435-2438` and is reset-bound.

- With `tps = 24` the ring is `round(100 s · 24) = 2400` slots
  (`config.py:143`; the TOML comment "1200 @ 12 tps" is stale and is not
  touched).
- At 33 B per unit per tick that is **≈ 77 MiB** when a swarm is present,
  comparable to the grid rings. It covers the "infestation" density
  (~1,000). A cap of 2048 would be ≈ 155 MiB.

---

## 8. Save / load — handed to chapter 02 (R4)

**The §8 build is cut.** `export_state`, `import_state`, `SwarmSave`,
`decode_swarm_section` and v1's T21 are gone, and nothing replaces them in
P2.

**Hand-off.** Save rides chapter 02's field-set contract. The swarm arrays
live on gmap (`SWARM_RESIDENT_NAMES`) and are saved with the field set;
`gmap.swarm_species.hash` rides as provenance. The hash-mismatch policy and
replay validity belong to that contract. A future loader restores a store
into a fresh run (reset), restores `high_water` and `next_unit_id` with it,
sets `gmap.swarm_host_dirty`, and should pass `check_invariants`.

**P2's only save duty: picklability.** The store arrays and
`gmap.swarm_species` (a module-level frozen `SwarmSpeciesTable` of
module-level frozen `SwarmSpeciesRow`s) pickle and round-trip equal (T19).
The test pickles those objects, not all of GameMap.

---

## 9. C++: none in P2 — and the contract P3 (and P4) inherit

**P2 writes no C++**: no `swarm.h`, no CMakeLists line. Nothing consumes a
header before P3's kernel, and a header with no consumer and no gate is
pure drift risk. P2's Python constants are the source that P3 twins.

P3's `cpp/src/swarm.h` twins these and identity-tests them against Python
(the philox32 cross-language precedent):

- the `ROSTER` dtypes;
- the `AI_*` enum values;
- `PI_Q16` / `HEADING_PERIOD_Q16`;
- `wrap_heading_q16`, with a **floor**-mod in int64;
- `FIRST_UNIT_ID`;
- the all-zero empty-slot form.

**Other constraints P2 hands P3:**

- The kernel/twin loop runs over `[0, high_water)` and skips `valid == 0`.
- The death branch writes the empty-slot form (all columns 0) and emits the
  event from the pre-zero values.
- **Every kit-derived heading passes `wrap_heading_q16` before any store
  write** (R8): an `atan2_q16` output, a Philox angle draw, `draw −
  PI_Q16`. Draw with `philox32.TWO_PI_Q16 = 411775`, wrap with
  `HEADING_PERIOD_Q16 = 411774`, and never "unify" the two. The endpoint
  double weight (§2.4) is accepted.
- **Residency (R9).** P3's transfer names the store arrays in slot 7's D2H.
  It uploads after **any** host store write: it consumes
  `gmap.swarm_host_dirty[sp]` (set by spawn, reclaim and any future host
  writer, e.g. a chapter-02 loader) and clears it after the upload. A stale
  device `valid=1` must never resurrect a reclaimed unit at the D2H.
- **Species-table H2D.** P3 keeps the hash it last uploaded and re-uploads
  when `gmap.swarm_species.hash` differs. That is a pure read with no flag:
  the reload seam (§4.3) is the only writer.
- **New tunables** are a `SPECIES_FIELDS` row + a `SPECIES_RELOAD` entry +
  a `SwarmSpeciesRow` field. P3 creates `SPECIES_RULES` with its first rule
  (`speed·Δt < 1 tile`).
- P3's gates reuse `check_invariants(gmap)`.
- P3 adds the presence-gated xarch lines (`_xarch_perfield_digest.py`
  per-column lines, `xarch_digest.py` suffix) together with its lockstep
  scenario (R18).

**The P4 hand-off (R16), `swarm_brood`.**

- The entity schema package is stdlib-only, so the brood class cannot
  import `swarm.py` (numpy). It declares `species` as `KIND_ENUM` with its
  own append-only choices tuple duplicating `SWARM_SPECIES`, and its
  load-time system module asserts the two agree. This is the prop route:
  `PROP_MATERIAL_CHOICES` (`entities/prop.py:32-35`) is validated against
  the canon table in `prop_system.py:34-41`.
- P4 spawns through `self.swarm.spawn` inside `_reset_internal`.
- An overflow is `SwarmFull`, an authoring error: no capacity clamp (R10).

---

## 10. Merge-friendliness vs `fire-12`

Checked with read-only `git diff -U0 $(git merge-base HEAD fire-12)
fire-12`. HEAD equals the merge-base for every file below, so these line
numbers are valid on HEAD.

| shared file | fire-12's hunks (merge-base lines) | P2 footprint | overlaps? |
|---|---|---|---|
| `src/simulation/gamemap.py` | 11, 132-137, 162-189, 254, 406-514, 1298 … 2352 | 1 import after the gases block (`:56`); 1 line + comment before `:223`; the allocate call after `:825` | No (gaps 190-253, 515-1297) |
| `src/simulation/simulation.py` | 1616 (+9) | import (`:100`); `on_config_reload` after `:258`; 1 line (`:264`); 2 kwargs (`:405-409`, `:1582-1584`) | No |
| `src/debug_keys.py` | 24, 30, 95-98, 112 | 1 line after `:55` | No |
| `src/simulation/factions.py` | none | `FACTION_NONE` + `__all__` | No |
| `src/simulation/entities/schema.py` | none | `KIND_REAL_Q16`: constant, docstring bullet, `ALL_KINDS`, one `field_value_error` branch, `import math` | No |
| `src/simulation/recorder.py` | none | §7 | No |
| `tests/field_digest.py` | 71 (+16), 99 | fold in `tick_digest` + a docstring sentence (`:24-28`) | No |
| `tests/field_digest_spec.toml` | 11, 59 | `[swarm_section]` at EOF (`:92`) | No |
| `tests/field_ab_harness.py` | 87 (+5), 511 | imports (~`:50`), capture (+2 at `:302`), `_diff_swarm_state`, the diff-loop case (`:390`), `swarm_scenario_sim`, `__main__` tuple (`:451`) | No |
| `config.toml` | … last at 2350 | `swarm_units_cap` in `[recorder]` (`:2435-2438`) + `[swarm.larva]` at EOF | No |
| `CLAUDE.md` | 3-4, 27-31, 69-70, 76, 90-91 (+rows), 115 | new rows appended **after the Deterministic RNG row** (HEAD `:93`, added by P1 after the merge-base). No existing row edited. | No (the unchanged RL-batch row, `:92`, separates it from fire-12's 90-91 insertion) |
| `physics_runner.py`, `CMakeLists.txt`, `_xarch_perfield_digest.py`, `xarch_digest.py`, `entities/serialize.py`, `entities/registry.py` | — | **untouched** | — |

**Do NOT touch** the stale "F5 reload" wording in CLAUDE.md's Config row
(HEAD `:75`, adjacent to fire-12's hunk at 76). Fix it after fire-12
merges, together with the same stale wording in `config.py:8, 50, 146` and
`temperature_scale.py:14` (R19).

The body lives in the new files `swarm.py` and `swarm_fixed.py`, plus the
tests.

---

## 11. Verification plan (tests are written from this list)

**The standard.** Each test names the property it protects and the change
that must break it. None pins the roster's size or order, and none pins a
set, count or order of anything designed to grow. Loops run over `ROSTER`,
`SPECIES_FIELDS`, `SWARM_SPECIES` or `SWARM_RESIDENT_NAMES` and assert the
rule each member obeys.

**Fixtures.**

- Store/lifecycle tests use `GameMap(_scenario_level())` + `SwarmUnits`, or
  `Simulation(_scenario_level(), breach_physics=None, …)` (fast, no build
  needed).
- Harness/golden tests need the CPU build.
- Config fixtures replace `CFG.swarm` with a fresh `config.Namespace`
  (`monkeypatch.setattr(CFG, "swarm", Namespace({...}))`, the shape
  `CFG.reload()` produces). Then they either construct a GameMap (the
  load-time path) or call `sim.on_config_reload()` (the reload path).
  Reload tests always go through the seam; they never call a species
  accessor expecting it to reload.

**`tests/test_swarm_digest.py`:**

| # | property | must break if… |
|---|---|---|
| T1 | Every existing golden is byte-identical. Oracle: the full suite green with **zero** golden edits (`test_w6_armory` canonical golden + RNG canary, B6, vent/B1/B2), plus a `git diff` of P2 touching no golden value and not `tests/_pg5_base_trajectory_45050f3.pkl`. | the fold is unconditional, capacity is hashed, or store arrays join `DIGEST_FIELDS`/`SIM_FIELDS` |
| T2 | Absence transparency: `tick_digest` of a snapshot with no key, with a `present=False` carrier, and a frozen verbatim pre-P2 `tick_digest` copy (the `_pre_a4_tick_digest` idiom, `test_entity_digest.py:116`) are all equal, on a synthetic snapshot with and without an entity carrier. | the marker or bytes are hashed when absent |
| T3 | The section is not vacuous: for **every** ROSTER column, perturbing one slot inside `[0, hw)` (including a reclaimed `valid=0` slot) changes `tick_digest`, and restoring it restores the digest; the same holds for `next_unit_id` and each species' `high_water`; `field_digest` never changes. | a column is dropped from the encoder, only live slots are hashed, or scalars are omitted |
| T4 | Capacity moves nothing: identical spawns with `max_units` 8192 vs 64 give identical section bytes and `tick_digest`; a value written at a slot ≥ `high_water` leaves the digest unchanged. | full capacity is serialized |
| T5 | Presence is monotone: spawn 1 and reclaim it → `live_count == 0` yet `present`, and the digest ≠ the swarm-free digest. | presence is keyed on live count |
| T6 | Strict carrier: a present gmap plus a snapshot without `__swarm__` raises `KeyError`; a swarm-free gmap accepts; `capture_trajectory` writes a carrier on every tick. | a capture path skips the carrier |
| T7 | Harness: `capture_trajectory(make_sim=swarm_scenario_sim)` twice → `assert_trajectories_match` at tol 0. A one-slot perturbation is reported with species/column/slot/both `unit_id`s. Comparing two present carriers never raises. | the locator treats the carrier as opaque, ignores it, or copies the entity `==` fast path |
| T8 | P2's swarm is inert on fields: per tick, `field_digest(swarm_scenario) == field_digest(default_scenario)`, while the tick digests differ. | larvae write a field — **expected to break at P6 (food depletion); retire it there** |
| T9 | (R1) Pre-P2 compatibility: a fresh default-scenario capture with `__swarm__` deleted from every snapshot (exactly the shape a pre-P2 build pickled) diffs clean against a fresh capture, in both argument orders. The same stripped trajectory against a `swarm_scenario_sim` capture reports "present in only one run" on every tick. Self-contained: it does not load the 45050f3 pickle, which fire-12 stops comparing (see below). | the one-sided-key check runs before the swarm case, or a missing key is compared as a mismatch against an absent carrier |
| T10 | (R2) Integer rendering: two carriers equal except that `next_unit_id`/`high_water` are `np.int32`/`np.int64` scalars in one and Python ints in the other encode to identical bytes; `b"np."` never appears in section bytes; header integers are ASCII decimal. | a header or scalar token is built through `str()`/`repr` of numpy scalars |
| T11 | (R6) Env-0 only: with `swarm.N_ENV` monkeypatched to 2, `swarm_carrier` raises `NotImplementedError` instead of encoding. | the carrier or encoder silently loops or folds envs (B1's decision) |

**`tests/test_swarm_store.py`:**

| # | property | must break if… |
|---|---|---|
| T12 | Every ROSTER column and each per-env scalar exists on GameMap with shape `(N_ENV, max_units)` / `(N_ENV,)`, the roster dtype, C-contiguous and little-endian, starts in the empty form, and **is in `GameMap._RESIDENT_FIELD_NAMES`** (born through residency). `gmap.swarm_host_dirty` has a key for each species, all False on a fresh map, and is not resident. | a column is mirror-only, has the wrong dtype, or is not resident; the dirty flag becomes a synced array |
| T13 | Spawn takes the lowest free slots in request order: fresh store → `0..k-1`; after reclaims, a batch of m gets `sorted(free_before)[:m]`. | append-at-hw, LIFO, compaction |
| T14 | Model-based: after a seeded random interleaving of spawn/reclaim batches (seeded `np.random.default_rng` **in the test**), the store equals a ~20-line reference model, and `check_invariants(gmap) == []` after **every** operation. Also: `high_water` never decreases and == 1 + max slot ever occupied; every assigned `unit_id` is unique and strictly increasing; `next_unit_id == FIRST_UNIT_ID + total spawned`; a reused slot gets a new id; every non-empty spawn/reclaim sets `host_dirty[sp]`. | id = slot, id reuse, hw recomputed from live slots, a lifecycle op that breaks an invariant, a write that skips the dirty flag |
| T15 | Reclaim writes the empty-slot form: every column of a reclaimed slot is byte-equal to a never-spawned slot's (so `valid=0`, `ai_state=AI_DEAD`). | reclaim that only clears `valid` |
| T16 | Atomic refusal: overfilling a small store raises `SwarmFull`; bad inputs raise (out-of-grid pos, bad env, unknown species, reclaim of an empty/duplicate/≥hw slot). R7 cases: with `next_unit_id = INT32_MAX − 1`, spawning 1 succeeds (id `INT32_MAX − 1`, counter → `INT32_MAX`) and the next spawn of 1 raises; `faction` of `2**31` or `−2**31 − 1` raises; a heading outside int32 raises; bool, float and uint64 arrays raise `TypeError`. In every case all arrays, scalars and `host_dirty` are byte-identical to before. | partial fill / partial write, a counter that can reach `INT32_MAX + 1`, silent int32 truncation of `faction` |
| T17 | Heading is canonical: the stored heading ∈ (−PI_Q16, PI_Q16] and ≡ input mod `HEADING_PERIOD_Q16`. Inputs: ±PI_Q16, ±3·PI_Q16, **411774, 411775**, 0, ±1, INT32_MIN, INT32_MAX, and the Philox endpoints (raw draws 0 and 411774, and each minus PI_Q16). `wrap` is idempotent, and its scalar and array forms agree. | a raw heading is stored, the period is `quantize(2π)`, the wrap runs in int32, or the scalar and array forms diverge |
| T18 | The lifecycle draws no randomness: spawn/reclaim leave `sim.rng.bit_generator.state` unchanged (the `test_w6_armory.py:548-563` canary idiom). | spawn draws from `sim.rng` |
| T19 | (R4) Picklable: a pickle round-trip of every array in `SWARM_RESIDENT_NAMES` (array_equal + dtype) and of `gmap.swarm_species` (dataclass `==`, including every `SwarmSpeciesRow`) returns equal objects. | a runtime-generated row class, a lambda or an unpicklable member on the table |
| T20 | (R4) `check_invariants` is not vacuous: on the swarm scenario it returns `[]`; each single deliberate corruption is reported: `next_unit_id ≠ FIRST` with every hw 0; `next_unit_id == FIRST` with hw > 0; a duplicate live id; a live id ≥ `next_unit_id`; a non-zero freed slot; a non-zero slot ≥ hw; `valid == 2`; a live heading of −PI_Q16; a live `AI_DEAD`; hw > capacity. | any listed check is dropped |
| T21 | (R11) Name hygiene: every `SWARM_SPECIES` name matches `^[a-z][a-z0-9]*$`; all store attribute names (= `.npz` keys) and the fixed names are pairwise distinct. `check_store_names` rejects a hypothetical species `next` (collides with `swarm_next_unit_id`), a species name with `_`, and a column named `high_water`. | the regex or the uniqueness check is removed |

**`tests/test_swarm_species.py`:**

| # | property | must break if… |
|---|---|---|
| T22 | Strict schema: each of these raises a `ValueError` naming the key at load (`GameMap(...)` construction): a missing `[swarm]`, a missing species row, an unknown species, an unknown key, and, **for every `SPECIES_FIELDS` entry**, a wrong type and an out-of-bounds value. Also: NaN and ±inf for every `KIND_REAL_Q16` entry and a bool for every `KIND_INT` entry. `SPECIES_RELOAD`'s keys and `SwarmSpeciesRow`'s fields match `SPECIES_FIELDS` (the import-time assertions hold). | Namespace pass-through without schema, a lax `KIND_REAL_Q16` branch, a drifted row class |
| T23 | (R3) Hot reload through the seam, with no species accessor touched: spawn A; replace `CFG.swarm` with `hp_max` changed; spawn B **without calling the seam** → B gets the OLD `hp`. Take digest D1 and call `sim.on_config_reload()` → digest still D1, every store array unchanged, `carrier()["species_hash"]` changed; spawn C → `hp == ` the new `hp_max_q`. A reload that changes only spelling (`1.0` → `1`) keeps `.hash` and prints no "replay comparability ends" line. | any reader reloads lazily, a reload rewrites live `hp`, the table is hashed into the section, or the hash is taken over authored text |
| T24 | (R3) Recorder-on vs recorder-off: two sims (`enable_recorder` True / False), same scripted spawns. `CFG.swarm` is swapped at tick 3 without calling the seam; spawn at tick 5; the seam is called on both at tick 6; spawn at tick 7 → identical section bytes on every tick. | any read path (`carrier()`, the recorder, spawn) reloads the table |
| T25 | Load-time field: each seam call whose TOML `max_units` differs from the in-force value prints exactly one warning line for that species and leaves capacity and the in-force value unchanged; a reload with an equal value prints none; `sim.reset()` then picks up the new capacity. | mid-run reallocation, a warning that spams or never appears |
| T26 | Rejected reload: an invalid value through the seam keeps the prior table (same object, same `.hash`), prints exactly one `RELOAD REJECTED` line naming the key per seam call, and touches no array. | crash, or silent partial apply |

**`tests/test_swarm_recorder.py`:**

| # | property | must break if… |
|---|---|---|
| T27 | A swarm-free dump has no `swarm_*` key (key set identical to pre-P2). Swarm-present: **every** ROSTER column is recorded at exactly its roster dtype, with width `min(max hw, cap)`. Rebuilding a carrier from recorded row t and encoding it through `swarm_section_bytes` equals the section bytes the digest hashed at tick t. With `cap < hw`: the width is `cap`, `swarm_<sp>_high_water` shows the true hw, and exactly one truncation line is printed. `swarm_species_hash` is per tick and changes exactly at the tick a seam reload changed `.hash`. | a float32 ring or cast, full-capacity recording, recorder/digest drift, a single 0-d hash, silent truncation |

**Existing gates that must stay green untouched:**

- `test_ingress_lint.py` (it scans `swarm.py`/`swarm_fixed.py`);
- `test_entity_digest.py`, `test_entities_core.py`, `test_entity_editor_ui.py`
  (schema.py gains a kind);
- `test_recorder_dump.py`, `test_vent_dormancy.py`;
- `test_onephase_control.py` (the debug-keys path);
- the residency-membership tests (`test_cool_shift_axis.py:257`,
  `test_fuel_fraction_axis.py:244`, `test_thermal_mass_axis.py:429`);
- `test_w6_armory.py`, `test_b6_logic_golden.py`, `test_field_ab_harness.py`;
- **`test_thermostat_books.py`** (R1: its
  `test_thermostat_books_byte_identical_to_base` diffs a fresh capture
  against the pre-P2 45050f3 pickle and passes today).

Note for the merge: fire-12 turns that thermostat case into an
unconditional skip (its G12 note, fire-12 `test_thermostat_books.py:166-203`).
After fire-12 merges, T9 alone carries the pre-P2-pickle guarantee.

**The plan gate:** full suite, no new reds; zero golden edits;
`tests/_pg5_base_trajectory_45050f3.pkl` untouched.

---

## 12. Conflicts found (chapter/task/plan vs code) and their resolutions

1. **`T_prev`** appears in chapter §2's table, §4, §4.1 and §6's gate text
   ("fuzz … incl. `T_prev`"), but the Review log ruled it out. It is
   excluded from ROSTER.
2. **"F5 hot reload"** (§2c). The code's reload is **Ctrl+R**
   (`debug_keys.py:53-55`), and F5 toggles normal maps
   (`renderer/game_renderer.py:1502`). Resolved as Ctrl+R → `CFG.reload()`
   → the explicit seam `Simulation.on_config_reload()` (§4.3, R3). The same
   stale "F5" wording sits in CLAUDE.md's Config row, `config.py:8, 50, 146`
   and `temperature_scale.py:14`. It is queued for after fire-12 merges and
   not touched in P2 (R19).
3. **"Food = spec v6"** (§6/§11). fire-12 already moves
   `DIGEST_SPEC_VERSION` 5 → **6** (T5b, `+dyn_heat_atten_q`). P6's food
   bump will be the next free number after fire-12 merges (v7). P2 has no
   spec bump.
4. **Save** (§6, and plan row P2's "save hooks"). The "(designed, unbuilt)
   02 save contract" is confirmed unbuilt (`simulation.py:53-57`: "not implemented yet"; engine/02
   "Implementation status"). P2 ships no save code: save rides 02's
   field-set contract, and P2's duty is picklability (§8, R4).
5. **Task: "save via the one serializer (`entities/serialize.py`)"**: not
   applicable. Save is 02's array contract, and that module is stdlib-only
   by rule while swarm units are not entities. The swarm digest's one
   encoder lives in `swarm.py`, with the digest/recorder principle of §5.3.
6. **Recorder "rings `(capacity, max_units)`"** (§2b) at the real
   `capacity = 2400` would be ≈ 620 MB at 8192. Resolved: env-0 rings
   `(capacity, swarm_units_cap)`, capped at 1024 by config, with
   `[0, high_water)` trimming at dump (§7).
7. **§1/§9b per-species SoA stores vs §2's singular `gmap.swarm_*`**:
   resolved as `gmap.swarm_<species>_<col>`, per-species `high_water`,
   per-env shared `next_unit_id`, and species-keyed section blocks (§1.1).
8. **Heading "(−π, π]" at Q16**: the canonical period must be
   `2·PI_Q16 = 411774`, not `quantize(2π) = 411775` (§2.4).
9. **§7 "Death … plus a tick event"**: the event is behaviour (P3). P2
   defines the freed-slot form P3 must reproduce (§2.5, §9).
10. **The `ai_state` enum order** is unspecified in the chapter. P2 fixes
    `DEAD=0, RUN=1, EAT=2, COMA=3` for the empty-slot invariant (§1.2).
11. **§2 "(N_env, max_units)" vs the section.** The store is born
    `(N_ENV, max_units)`. The section, carrier and recorder are env-0-only
    behind an explicit `N_ENV == 1` check, and the multi-env fold is the RL
    arc's B1 decision (§5.2, R6).
12. **§9b "one small class Simulation drives for spawn / reset /
    serialize"**: `SwarmUnits` via `sim.swarm`. Reset is structural (§3),
    serialize is the digest section (§5), and delegating `spawn_swarm` /
    `reclaim_swarm` methods wait for a first non-test caller (R18).

---

## 13. Build brief (ordered; a Sonnet implementer executes this without re-deriving)

Python: `C:/Users/steen/anaconda3/python.exe` (conda env `data`); never
bare `python`.

1. **`src/simulation/factions.py`**: add after `FactionId = int`:

   ```python
   # No faction (arc #63 P2): the swarm-unit default; v1 never targets swarm
   # units. Team ints in use are non-negative, so this never aliases a team.
   FACTION_NONE: FactionId = -1
   ```

   Then add `"FACTION_NONE"` to `__all__`.
2. **`src/simulation/entities/schema.py`** (R5):
   - `import math` after `import abc`.
   - A module-docstring bullet after `KIND_LENGTH_M`'s (`:20-22`):
     "`KIND_REAL_Q16` values are authored reals (TOML int or float),
     quantized to Q16.16 ONCE at load by their consumer (first consumer: the
     swarm species table, arc #63). Unused by entities today."
   - `KIND_REAL_Q16 = "real_q16"  # authored real; quantized to Q16.16 at
     load` after `KIND_LENGTH_M` (`:62`).
   - Append `KIND_REAL_Q16` to the end of `ALL_KINDS` (`:78-80`). Do
     **not** add it to `NUMERIC_KINDS`, `serialize.SYNCED_FIELD_KINDS` or
     the editor mirror.
   - In `field_value_error` (`:227`), after the
     `KIND_LENGTH_M`/`KIND_FLOAT_RENDER` branch (`:241`):

     ```python
     elif f.kind == KIND_REAL_Q16:
         if not _is_number(value) or (isinstance(value, float)
                                      and not math.isfinite(value)):
             err = ("must be a finite number (authored real, quantized to "
                    "Q16.16 at load)")
     ```

     The existing bounds block (`:286-289`) then applies unchanged.
3. **`src/simulation/swarm_fixed.py`** (new): §4.4. Write the module
   docstring in the `gas_fixed` style. Cite `fixed_point.h:890-903` for
   `PI_Q16` and survey §E flag 6 for the re-export. The body is the
   `gas_fixed` re-export line, then `PI_Q16`, `HEADING_PERIOD_Q16`, and
   `wrap_heading_q16`:
   - scalar path:
     `int(PI_Q16 - ((PI_Q16 - int(h)) % HEADING_PERIOD_Q16))` (Python ints
     are unbounded and `%` floors);
   - ndarray path: `h64 = np.asarray(h, np.int64)`, then
     `PI_Q16 - np.mod(PI_Q16 - h64, HEADING_PERIOD_Q16)` (`np.mod` floors);
     returns int64.
4. **`src/simulation/swarm.py`** (new), in this order:
   - Module docstring (engine/17 §2/§6/§7/§9b; this doc).
   - Imports: `hashlib`, `json`, `re`, `dataclasses`, `numpy`,
     `from config import CFG`,
     `from simulation.entities.schema import Field, KIND_INT, KIND_REAL_Q16, field_value_error`,
     `from simulation.factions import FACTION_NONE`,
     `from simulation import swarm_fixed`.
     **It must not import `simulation.gamemap` or `simulation.simulation`**
     (the import direction is gamemap → swarm).
   - Constants: `N_ENV = 1` (comment: "the first batch-axis constant — the
     one N until the RL arc's B1"), `SWARM_SPECIES`,
     `Column(name, dtype)` NamedTuple + `ROSTER` (§1.2 order),
     `AI_DEAD, AI_RUN, AI_EAT, AI_COMA = 0, 1, 2, 3`, `FIRST_UNIT_ID = 1`,
     `INT32_MIN`, `INT32_MAX`.
   - Names: `store_attr`, `high_water_attr`, `NEXT_UNIT_ID_ATTR`,
     `SPECIES_ATTR`, `HOST_DIRTY_ATTR`, `SPECIES_HASH_NPZ_KEY`,
     `RESERVED_COLUMNS`, `check_store_names(species, roster)` (§1.1; raises
     `ValueError`). Call it at import on `(SWARM_SPECIES, ROSTER)`. Then
     `SWARM_RESIDENT_NAMES` (a tuple: columns per species in ROSTER order,
     then each `high_water`, then `NEXT_UNIT_ID_ATTR`).
   - Schema: `SPECIES_FIELDS` (§4.1 table), `SPECIES_RELOAD`, the stored
     attribute rule (`<name>` for `KIND_INT`, `<name>_q` for
     `KIND_REAL_Q16`), `SwarmSpeciesRow` (module-level frozen dataclass),
     the two import-time assertions, and `SwarmSpeciesTable`
     (`rows`, `hash`, `from_rows`, `from_config`, `row`; §4.2).
   - `allocate_swarm_store(gmap, cfg=CFG)` (§1.3, steps 1-4).
   - `class SwarmFull(ValueError)`.
   - `swarm_column(gmap, species, col)`.
   - `check_invariants(gmap) -> list[str]` (§2.6).
   - Section: `SWARM_DIGEST_KEY = "__swarm__"`,
     `SWARM_SECT_PREAMBLE = b"SWARM_SECT_V1\n"`, `swarm_present`,
     `swarm_carrier` (§5.2, with the `N_ENV` raise first),
     `swarm_section_bytes` (§5.3), `require_swarm_carrier`.
   - `reload_species(gmap, cfg=CFG) -> bool` (§4.3, steps 1-5, with the
     exact print lines given there).
   - `class SwarmUnits` (§3 table): `spawn` (§2.2: validation order 1-4,
     then writes, then `host_dirty`), `reclaim` (§2.5), queries, `column`,
     `species` (read-only property), `carrier`.
5. **`config.toml`**:
   - in `[recorder]` (`:2435-2438`):

     ```toml
     # Swarm recorder ring width (arc #63 P2, engine/17 §6): slots recorded per
     # species per tick (the [0, high_water) extent, capped here). Memory while a
     # swarm is present: 33 B x cap x capacity -> ~77 MiB at 1024 x 2400 (24 tps).
     # Reset-bound (the recorder is built in Simulation._reset_internal).
     swarm_units_cap = 1024
     ```

   - the `[swarm.larva]` block of §4.1 at EOF.
6. **`src/simulation/gamemap.py`**:
   - after the gases import block (`:46-56`):
     `from simulation.swarm import SWARM_RESIDENT_NAMES, allocate_swarm_store  # arc #63 P2`;
   - the §1.3 comment + `_RESIDENT_SYNCED` line immediately before
     `_RESIDENT_FIELD_NAMES = _RESIDENT_SYNCED + _RESIDENT_MASKS` (`:223`);
   - after `self.refresh_gas_energy()` (`:825`), a comment line plus
     `allocate_swarm_store(self)` as the last statement of `__init__`.
7. **`src/simulation/simulation.py`**: the §3 hooks 1-4.
8. **`src/debug_keys.py`**: `sim.on_config_reload()` right after
   `CFG.reload()` (`:55`), in the same `if` block (§3 hook 3).
9. **`src/simulation/recorder.py`**: §7 (constructor kwarg, `record(swarm=)`,
   lazy env-0 rings incl. `shash_ring`, the truncation print, dump keys,
   docstring schema).
10. **`tests/field_digest.py`**: the §5.4 fold, plus one docstring sentence
    beside the A4 note (`:24-28`): "the `__swarm__` section folds the same
    way (arc #63 P2, `SWARM_SECT_V1`, `simulation.swarm.swarm_section_bytes`)".
11. **`tests/field_digest_spec.toml`**: `[swarm_section]` at EOF (§5.5).
12. **`tests/field_ab_harness.py`**: §6 (imports of `SWARM_DIGEST_KEY`,
    `swarm_carrier`, `require_swarm_carrier`, `swarm_section_bytes` from
    `simulation.swarm`, plus `swarm_fixed`; capture +2 lines;
    `_diff_swarm_state`; the diff-loop case placed before the one-sided
    check; `swarm_scenario_sim`; the `__main__` tuple).
13. **Tests**: the four files of §11 (T1-T27).
14. **Verify** with `C:/Users/steen/anaconda3/python.exe -m pytest tests -q`:
    - no new reds;
    - `git diff` shows no golden value changed and the 45050f3 pickle
      untouched;
    - `tests/field_ab_harness.py` `__main__` still self-matches;
    - `_xarch_perfield_digest.py` reports `matches golden = True`.
15. **`CLAUDE.md`**: write the §15(b) rows, pointing at the real code, as
    new rows **appended after the "Deterministic RNG" row** (Sim core
    table). Edit no existing row.

**Do NOT touch:**

- CLAUDE.md's Config row ("F5 reload" is stale, but fixing it conflicts
  with fire-12; R19), and the stale F5 docstrings in `config.py`,
  `temperature_scale.py` and the renderer;
- any fire-12 hunk region listed in §10;
- `physics_runner.py`, `CMakeLists.txt`;
- `_xarch_perfield_digest.py`, `xarch_digest.py` (P3);
- `entities/serialize.py`, `entities/registry.py`,
  `tools/entity_editor_ui.py`;
- `SimState` / `get_state`;
- `tests/_pg5_base_trajectory_45050f3.pkl` (never regenerate it);
- every golden value;
- `config.toml`'s stale "1200 @ 12 tps" comment.

Stage explicit paths only; never `git add -A`.

---

## 14. For Erik

None blocking. Every call above implements the blessed chapter or a
critique resolution, with its reason stated. Three things touch your tuning
loop and are worth a glance:

- Ctrl+R now calls one explicit seam, `sim.on_config_reload()`, right after
  `CFG.reload()`. The swarm species table is its first client. Chapter 12's
  material/physics re-binds could join it later; that is your call, and not
  P2's.
- A Ctrl+R with a bad `[swarm.larva]` value is **rejected loudly and the old
  table kept**, rather than crashing the session (§4.3).
- An unknown key in `[swarm.*]` is a load error (§4.2).

---

## 15. Systems

**(a) Existing canonical systems P2 uses (CLAUDE.md table) and the seam:**

- **GameMap** — stores the arrays, `swarm_species` and `swarm_host_dirty`
  (fork 2a). One hook at the end of `__init__`. No topology or field
  writes.
- **Simulation facade** — owns `sim.swarm` and gains one public method,
  `on_config_reload()` (the reload seam). No `get_state`/`SimState` change,
  and no `spawn_swarm`/`reclaim_swarm` (R18).
- **Tick conductor** — **no new slot** in P2. It only gains a kwarg on the
  existing recorder line.
- **Config** — `[swarm.larva]` and `[recorder] swarm_units_cap` via `CFG`.
  Ctrl+R is honoured through the explicit seam, and the stale "F5" row is
  left for after fire-12.
- **Q16 boundary modules** — the new `swarm_fixed.py` re-exports
  `gas_fixed`'s helpers (no seventh copy) and adds only the heading
  constants and law. No inline 65536.
- **Fixed-point kits** — no call in P2. `PI_Q16` is pinned to the kit's
  narrowed π (`fixed_point.h:900-903`), and P3 twins the heading law.
- **Deterministic RNG** — P2 draws nothing. `unit_id` is the Philox counter
  lane `(tick, draw_index, salt, unit_id)`, per env, shared across species.
  `philox32.TWO_PI_Q16 = 411775` (draw range) and `HEADING_PERIOD_Q16 =
  411774` (wrap period) combine as draw-then-wrap (§2.4).
- **Field digest** — the `tick_digest` fold only. `DIGEST_FIELDS` and the
  spec version are untouched.
- **GOLDEN_AGGREGATE** — unchanged (the gate).
- **A/B lockstep harness** — carrier capture, the strict rule, the
  pre-P2-safe diff (R1), the section-bytes locator (R14), the swarm
  scenario.
- **Recorder** — an additive, presence-gated, env-0 per-unit ring family.
  It adds the new dtype classes uint8 and uint32 at roster dtype with no
  cast, plus an int64 species-hash ring.
- **Entity system** — **`schema.Field` and `schema.field_value_error` are
  reused** for the species table, with one new kind, `KIND_REAL_Q16`.
  `registry_content_hash`'s recipe is reused for `.hash`. The carrier /
  strict-presence / section-local-version **pattern is copied**, but
  `serialize.py` is not reused (swarm units are not entities, and
  `serialize.py` is numpy-free by rule).
- **Factions** (`factions.py`, not a table row) — hosts `FACTION_NONE`.
- **Ingress lint** — scans the new modules.
- **RL-batch habits (§A)** — `(N_ENV, …)` birth, residency from day one, no
  host tick logic.
- **Material/gas table pattern** — `from_config` at GameMap construction.
- **Test conventions** — property tests plus a model-based oracle, with
  `check_invariants` as a reusable property gate.
- **Not used in P2:** FieldEdit, PhysicsRunner/PhysicsEngine, the coupling
  table, tick events, the CUDA harness (P3's 3-part gate), the xarch dumpers
  (P3), `simulation/species.py` (a different "species": CPU-unit stat
  generation).

**(b) New systems P2 creates — DRAFT one-line CLAUDE.md rules.** They are
written at implementation, pointing at the real code, as rows appended after
the Deterministic RNG row (R19):

- **Swarm store** — `src/simulation/swarm.py` (`ROSTER`,
  `allocate_swarm_store`, `check_invariants`) + `gmap.swarm_<species>_<col>`.
  THE per-unit state of swarm units: SoA `(N_ENV, max_units)` per species,
  born in `GameMap._RESIDENT_SYNCED`.
  - A species is a `SWARM_SPECIES` entry + a table row + a kernel — never a
    class, never a per-species if-chain.
  - Stable slots, no compaction, no atomic-ticket spawns: spawn takes the
    lowest free slots in request order. A freed slot is the all-zero form.
  - `unit_id` (per env, shared across species, from 1, never reused) is
    identity; the slot is not.
  - A new column is a `ROSTER` row + a `SWARM_SECT` version bump, never a
    stray gmap array or a `DIGEST_FIELDS` row.
  - `N_ENV` is the codebase's first batch-axis constant — the one N until
    the RL arc's B1 makes it a construction parameter; never a second N.
- **Swarm digest section** — `swarm.py::swarm_carrier` /
  `swarm_section_bytes` + `field_digest.tick_digest`. Per-unit swarm state
  enters determinism artifacts ONLY through the presence-gated `__swarm__`
  carrier: `SWARM_SECT_V1`, env 0 behind an explicit `N_ENV == 1` check,
  extent `[0, high_water)`, hw-0 species blocks omitted, every integer
  `int()`-rendered.
  - Capture paths always write the carrier; a snapshot without the key
    (pre-P2) reads as absent.
  - The digest hashes the one encoder's bytes; the recorder stores the same
    columns raw; T27 gates them equal.
  - Capacity and the species table never enter hashed bytes.
- **Swarm facade** — `swarm.SwarmUnits` (`sim.swarm`) + the free readers
  `swarm_column` / `swarm_carrier`. The only Python-side writer of the store
  (P3's kernel and its CPU twin are the sim-side writers).
  - Never index `gmap.swarm_*` to write from gameplay, tools or tests.
  - Spawn takes explicit Q16 values and draws no RNG; callers own
    randomness.
  - Overflow is `SwarmFull`, never clamped by capacity.
  - Every host write sets `gmap.swarm_host_dirty[sp]` for the device
    upload.
- **Swarm species table** — `swarm.SPECIES_FIELDS` (`entities/schema.py::Field`,
  incl. `KIND_REAL_Q16`) + `SPECIES_RELOAD` + `[swarm.<species>]`. A swarm
  tunable is a `Field` row + a reload class + a TOML value.
  - It is validated strictly at load and on every reload (unknown keys or
    species are errors) and quantized at ingress via `swarm_fixed`.
  - It is installed only at GameMap construction and by the config-reload
    seam; every consumer reads `gmap.swarm_species` and never reloads.
  - `hot` fields apply from the next tick, `load` fields at the next reset,
    and an invalid reload is rejected loudly.
  - The table hash is provenance, never digested.
  - **Candidate shared helper:** its strict validator (the structure checks
    + `field_value_error`) is the candidate shared strict-config-table
    helper for chapter 12's "no schema validation" gap. Extract it before a
    second TOML table copies it.
- **Config reload seam** — `Simulation.on_config_reload()`, called by
  `src/debug_keys.py` right after `CFG.reload()`. THE place a live table
  re-binds after Ctrl+R, applying from the next tick. A new hot-reloadable
  table adds one call here — never a lazy identity check inside a reader
  (a debug tool must not decide which table is in force).
- **Swarm heading law** — `swarm_fixed.wrap_heading_q16`. Every heading
  write is canonical (−PI_Q16, PI_Q16] with period `2·PI_Q16 = 411774`,
  deliberately ≠ `philox32.TWO_PI_Q16 = 411775` (the draw range: draw, then
  wrap). Every kit-derived heading passes it. The C++ twin (P3) floor-mods
  in int64 and is identity-tested against it. This is its own appended row,
  not an edit of the Q16 boundary modules row, so P2's CLAUDE.md diff stays
  append-only (R19).

`FACTION_NONE` (`factions.py`) and `KIND_REAL_Q16` (`schema.py`) are
constants inside existing systems, not rows of their own.
