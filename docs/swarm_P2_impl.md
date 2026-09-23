# Swarm units P2 — implementation doc: the swarm store scaffold

> Arc #63, branch `63-swarm-units` (patch branch `63-p2-swarm-store` per the
> plan). Written 2026-09-24, just-in-time against HEAD `e183fc4`. Executes
> `docs/architecture/engine/17_swarm_units.md` (BLESSED 2026-09-10) §2, §6, §7
> and §9b's "shared foundation" row, under plan row P2
> (`docs/swarm_units_patch_plan_2026-09-10.md`). Design-gated: this doc + one
> critic before the build. The chapter is executed, not re-designed; where
> its body disagrees with the Review log, the log wins (§12 lists every
> conflict found).
>
> **P2 in one sentence:** per-species SoA arrays on GameMap, born
> `(N_env, max_units)` through the residency path, with a lifecycle facade,
> a strict species table, and a presence-gated `SWARM_SECT_V1` section
> that digest, recorder and save all read — and no behaviour, no tick line
> and no C++.

---

## 0. Scope

**In:** the store (§1), spawn/reclaim lifecycle (§2), the `SwarmUnits`
facade + Simulation hooks (§3), the `[swarm.<species>]` table mechanism (§4),
`SWARM_SECT_V1` + carrier + digest fold (§5), A/B harness + xarch dumpers
(§6), recorder (§7), save/load primitives (§8).

**Out (later patches — only the hand-off constraints P2 imposes appear, in
§9):** kernel / CPU twin / `SwarmSolver` / sensing / death events (P3),
`swarm_brood` and level spawning (P4), rendering (P5), food (P6). P2 draws
no randomness; spawn takes explicit values. P2 does not depend on P1 in code
(no Philox call). If P1 has not merged when P2 builds, the plan order still
applies, but nothing in P2 breaks without it.

---

## 1. The store

### 1.1 Layout: one store per species, one id space per env

Chapter §1 defines a species as "a species table row, **a set of
fixed-capacity SoA per-unit arrays**, and a behaviour kernel". §9b adds
"extensions are per-species columns in the same section, so adding a species
never bumps another species' digest". So **each species owns its own store**,
and all stores share one section. v1 has one species (`larva`), but the
structure is species-keyed from day one. That costs nothing now, and it
means adding a second species later is purely additive.

- **Species set: closed in code, append-only:**
  `SWARM_SPECIES = ("larva",)` in `swarm.py`. A species cannot exist without
  a kernel, which is code, so the *set* lives in code and the *numbers* live
  in TOML (§4). The existing closed-set idiom is `WEAPON_ARCHETYPES`
  (`weapons.py:49-51`). Order = section block order, so the tuple only ever
  grows at the end.
- **GameMap attribute names:** `gmap.swarm_<species>_<col>` (e.g.
  `gmap.swarm_larva_pos_x`). They are formed in ONE place,
  `swarm.store_attr(species, col)`; no other code formats these names by
  hand. Attribute names are internal. The wire contract is the section
  (§5), which is keyed by species name and short column name, so a later
  rename is digest-neutral.
- **Per-env scalars:** `gmap.swarm_<species>_high_water` has shape `(N_ENV,)`
  int32 and there is one per species (slot spaces are per store).
  `gmap.swarm_next_unit_id` has shape `(N_ENV,)` int32 and is **shared by
  all species in an env**. Decision: a shared id space makes `unit_id` a true
  per-env identity across species, so the Philox counter
  `(tick, draw, salt, unit_id)` (chapter §3) can never collide between two
  species that reuse a purpose salt. With one species this behaves exactly
  like a per-species counter.
- **`N_ENV = 1`**, a module constant in `swarm.py`. This is the §A rule-1
  batch axis. No `N` exists anywhere else in the code today (checked); the RL
  arc (B1) turns it into a construction parameter.

### 1.2 The v1 roster (`swarm.ROSTER`, declaration order = section order)

`T_prev` is **absent**: the Review log (2026-09-10) ruled sensing to be a
spatial head-sweep, so the §2/§4/§4.1/§6 mentions of it are stale.

| col | dtype | meaning | empty-slot value | spawn writes |
|---|---|---|---|---|
| `pos_x` | int32 Q16.16 | x, tile units of the runtime grid | 0 | caller (raw Q16) |
| `pos_y` | int32 Q16.16 | y, tile units | 0 | caller |
| `heading` | int32 Q16.16 | radians, canonical (−π, π] (§2.4) | 0 | `wrap_heading_q16(caller)` |
| `hp` | int32 Q16.16 | hit points | 0 | species `hp_max_q` |
| `ai_state` | int32 | `AI_DEAD=0, AI_RUN=1, AI_EAT=2, AI_COMA=3` | 0 (`AI_DEAD`) | `AI_RUN` |
| `dist_walked` | uint32 Q16.16 | path length, wraps mod 2³² | 0 | 0 |
| `unit_id` | int32 | persistent identity (§2.3) | 0 ("no unit") | `next_unit_id` + k |
| `faction` | int32 | present from v1, unread in v1 | 0 | caller, default `FACTION_NONE = -1` |
| `valid` | uint8 | slot occupied | 0 | 1 |

**The empty-slot form is the all-zero slot.** Setting `AI_DEAD = 0` and
starting unit ids at 1 makes every column of an unoccupied slot zero,
whether the slot was never spawned or was reclaimed. So a live unit is
never `AI_DEAD` (chapter §7: death sets DEAD and `valid=0` in the same
step), `np.zeros` *is* the initial state, and a freed slot physically
carries nothing a render husk could be tempted to read (critique M2). The
enum values are hashed state, and P3 twins them (§9).

### 1.3 Allocation (the one GameMap hook) and the residency path

`swarm.allocate_swarm_store(gmap, cfg=CFG)` is called once, as the last
statement of `GameMap.__init__` (after `self.refresh_gas_energy()`,
`gamemap.py:824`):

1. `gmap.swarm_species = SwarmSpeciesTable.from_config(cfg)`. This is the
   load-time validation hook (§4), in the same role as
   `self.materials = MaterialTable.from_config(CFG)` (`gamemap.py:254`) and
   `self.gases` (`:261`).
2. `gmap.swarm_next_unit_id = np.full((N_ENV,), FIRST_UNIT_ID=1, np.int32)`.
3. For each species: `swarm_<sp>_high_water = np.zeros((N_ENV,), int32)`,
   and for each ROSTER column
   `np.zeros((N_ENV, row.max_units), col.dtype)` (C-contiguous, native
   little-endian).

This runs on every level and every reset (`Simulation._reset_internal`
builds a fresh GameMap, `simulation.py:262`). That is ~264 KB at 8,192
larvae and draws no RNG. Dormancy comes from the presence gate, not from
skipping allocation, so no code ever branches on "does the store exist".

**Residency (§A rule 3, "born through the residency path from day one").**
`swarm.SWARM_RESIDENT_NAMES` (every store array: the columns, the
high-waters, `swarm_next_unit_id`) is computed at import from
`SWARM_SPECIES × ROSTER`. GameMap adds it with **one line placed after the
`_RESIDENT_MASKS` literal and before** `_RESIDENT_FIELD_NAMES = ...`
(`gamemap.py:223`):

```python
    _RESIDENT_SYNCED = _RESIDENT_SYNCED + SWARM_RESIDENT_NAMES  # arc #63 P2: swarm store (engine/17 §2)
```

The effects:

- `enable_residency` (`gamemap.py:1694-1716`) gives each array a CuPy
  buffer. `cp.asarray` is shape-agnostic.
- `device_ptrs()` exposes the pointers (`:1722-1727`).
- The `__setattr__` stale-pointer guard (`:225-238`) now forbids
  reassigning any store array while resident. Every P2 write is in place.

**No per-tick transfer is added.** The resident tick's lists are explicit
(`physics_runner.py:1187-1188, 1242-1244, 1299-1300`, and a defaulted
`to_host()` is forbidden, `:1295-1298`), and no device kernel reads the
store in P2. Device copies therefore go stale after a host spawn while
resident, which is harmless until P3 adds a consumer (the `cool_shift`
caveat idiom, `gamemap.py:170-179`). Write that caveat as the comment on the
added line.

### 1.4 What "born `(N, max_units)`" means for code that assumes `(h, w)`

| `(h, w)`-shaped machinery | P2's answer |
|---|---|
| residency alloc/transfer (`gamemap.py:1694-1743`) | Shape-agnostic already. It just works. |
| targeted per-tick transfer lists (`physics_runner.py:1187,1242,1299`) | Untouched in P2; P3 names the store arrays in its slot-7 D2H. |
| `DIGEST_FIELDS` + `_field_bytes` (`field_digest.py:79-115`) | The store **never** joins `DIGEST_FIELDS`. A shape header would hash capacity. It rides the section instead (§5). |
| A/B `SIM_FIELDS` / `_snapshot` / per-cell locator (`field_ab_harness.py:83-96,142-144,400-413`) | Not in `SIM_FIELDS`. Rides the `__swarm__` carrier with its own slot locator (§6). |
| recorder grid rings `(capacity, fh, fw)` (`recorder.py:113-121`) | Never in `fields`. Gets a separate per-unit ring family (§7). |
| `_update_caches` / `on_tile_changed` / `reload_material_table` / `--res` | Operate on grids and never touch the store. |

---

## 2. Lifecycle (chapter §7 made exact)

### 2.1 `high_water`

`high_water[e]` = 1 + the highest slot index ever occupied in env `e` (an
exclusive bound; 0 = nothing ever spawned). It is monotone non-decreasing
within a run: spawn may raise it, reclaim never lowers it (no compaction),
and only a fresh GameMap (reset / level load) returns it to 0. The digested
and recorded extent is `[0, high_water)`, so capacity moves no byte.

### 2.2 Spawn: the k-th request takes the k-th lowest free slot

A batch of k requests, in request order, occupies the k lowest slots with
`valid == 0`, ascending. On a fresh store that is sequential fill
`0..k-1`. After reclaims, holes are reused lowest-first before `high_water`
extends. This is exactly chapter §7's in-sim-birth rule ("k-th request
pairs with k-th free slot via prefix-sum"), so the host spawn and a later
device birth agree by construction. It is not an atomic ticket (Safari
2022), not LIFO, and not append-at-high-water.

The spawn writes the "spawn writes" column of §1.2. Unit ids are
`next_unit_id, next_unit_id+1, …` in request order. Then
`next_unit_id += k` and `high_water = max(high_water, max(slots)+1)`.

**When full: atomic refusal.** If k exceeds the free slots,
`swarm.SwarmFull` (a `ValueError` subclass; the `SealBlocked` precedent,
`gamemap.py:96-105`) is raised and **nothing** is written. Decision:
partial fill would make "how many larvae exist" depend on `max_units`, a
knob that must move no behaviour. A loud refusal keeps that invariant for
every run that does not exceed capacity. A best-effort caller (P4, or the
future zombie brood) checks `free_count()` first. That query is
deterministic.

**Validation (all before any write → atomic):**

- species ∈ `SWARM_SPECIES`, and `0 <= env < N_ENV`.
- pos/heading/faction are integer array-likes. Floats are a `TypeError`:
  raw Q16 only, converted by the caller through `swarm_fixed` (door 2).
  They broadcast to one 1-D length k (k = 0 is a no-op returning an empty
  array).
- `0 <= pos_x < gmap._w << FP_SHIFT` and `0 <= pos_y < gmap._h << FP_SHIFT`.
- `next_unit_id + k - 1 <= INT32_MAX`.

Terrain is **not** checked. Placement is the caller's job (P4 scatters on
walkable tiles), and the store stays a data structure. Returns the slot
indices (int64 array, request order).

### 2.3 `unit_id` / `next_unit_id`

Ids start at `FIRST_UNIT_ID = 1` (0 = "no unit", the empty-slot value). They
are strictly increasing in assignment order and never reused. A reused slot
gets a new id. Reclaim never decrements the counter. `next_unit_id` is in
the section: reassigning ids after a load would fork every RNG stream
(chapter §6).

### 2.4 Heading is canonical at every write

`swarm_fixed.wrap_heading_q16(h) = PI_Q16 - ((PI_Q16 - h) mod HEADING_PERIOD_Q16)`,
with floor-mod computed in int64. The constants are
`PI_Q16 = 205887 == quantize(π)` (the kit's narrowed π,
`fixed_point.h:900-903`) and **`HEADING_PERIOD_Q16 = 2·PI_Q16 = 411774`**,
not `quantize(2π) = 411775`. Only a period of exactly twice the half-range
makes (−P, P] a complete residue system, i.e. one representation per
direction. At Q16 the π constants are inconsistent (`fixed_point.h:890-894`),
so this choice is forced, not free.

Probe-verified: `wrap(±P) = P`, `wrap(3P) = P`, `wrap(INT32_MIN) = -82238`,
and it is idempotent. Spawn stores `wrap_heading_q16(heading)`. P3's C++
twin must use a floor-mod (C++ `%` truncates) — see §9.

### 2.5 Reclaim

`reclaim(species, slots, *, env=0)` writes the empty-slot form (every column
zero) at each slot. `high_water` and `next_unit_id` are unchanged.

Validation (atomic): every slot is in `[0, high_water)`, the slots are
unique, and each has `valid == 1`. Reclaiming an empty slot is a caller bug
and raises `ValueError`.

The death **tick event** (chapter §7) is P3's, because it is behaviour.
Reclaim is the slot-freeing primitive and **defines the freed-slot form
that P3's death branch must reproduce** (§9).

---

## 3. The facade and how Simulation drives it

`swarm.SwarmUnits(gmap)` is the one small class chapter §9b allows ("the
only 'class' in the design"). It holds no arrays of its own: GameMap
stores them (fork 2a), and the facade reads and writes `gmap` in place.

| member | role |
|---|---|
| `species` (property) | The species table in force, with a lazy reload check (§4.3). |
| `reload_species()` | Explicit reload (used by the property and by tests). |
| `spawn(species, pos_x_q, pos_y_q, heading_q, *, faction=FACTION_NONE, env=0)` | §2.2 |
| `reclaim(species, slots, *, env=0)` | §2.5 |
| `live_count(sp, env=0)`, `free_count(sp, env=0)`, `high_water(sp, env=0)`, `present()` | Queries. |
| `column(sp, col)` | Returns the gmap array. The read accessor for P3/P5, so no caller formats names. |
| `carrier()` | `species` check, then `swarm_carrier(self.gmap)` (§5.2). |
| `export_state()` / `import_state(save)` | §8 |

**Simulation hooks (`simulation.py`, all thin; the tick conductor gets NO
new line).**

1. Import `SwarmUnits` beside the other system imports (after `:100`).
2. In `_reset_internal`, right after `self._seed = seed` (`:264`), add
   `self.swarm = SwarmUnits(self.gmap)`. A reset rebuilds the facade with
   the fresh GameMap, so "reset" is structural and needs no method.
3. Add two public mutation methods beside `add_unit`/`get_unit`
   (`:419-456`): `spawn_swarm(...)` and `reclaim_swarm(...)`, delegating to
   `self.swarm`. The CLAUDE.md facade row says Simulation is the only
   mutation seam, and chapter §1 says it "drives swarm-unit lifecycle
   exactly as it owns the unit list" (`add_unit` is the precedent). Internal
   callers (P4's load-time brood inside `_reset_internal`) may call
   `self.swarm.spawn` directly.
4. Add a recorder kwarg on the existing call (`:1582-1584`):
   `swarm=self.swarm.carrier()`. Pass
   `swarm_units_cap=CFG.recorder.swarm_units_cap` at recorder construction
   (`:405-409`).

**No tick line.** P2 has no behaviour, so nothing happens per tick. The only
per-tick touch is the recorder argument on an existing line (host debug
tooling, not sim logic; §A rule 2 holds). `get_state()`/`SimState` are
unchanged: the renderer (P5) reads the arrays through `gmap`, which
`SimState` already references.

---

## 4. The species table mechanism (`[swarm.<species>]`)

### 4.1 Schema in code, numbers in TOML

`swarm.SPECIES_FIELDS` is a tuple of
`SpeciesField(name, kind, reload, lo, hi, doc)`:

- `kind="count"` means a TOML int (a bool or float is rejected). It is
  stored as-is under attribute `<name>`.
- `kind="q16"` means a TOML number, quantized ONCE at build through
  `swarm_fixed.quantize_scalar` (ingress door 2). It is stored under
  attribute `<name>_q`.
- `lo`/`hi` are inclusive bounds on the **stored** value.
- `reload` ∈ {`"hot"`, `"load"`}.

`swarm.SPECIES_RULES` holds cross-field validators
`(species, row, cfg) -> None` and is empty in P2 (P3 appends
`speed·Δt < 1 tile`, chapter §2).

**P2's fields (only what P2 consumes):**

| key | kind | reload | bounds (stored) | P2 consumer |
|---|---|---|---|---|
| `max_units` | count | **load** | `[1, 1<<20]` | allocation |
| `hp_max` | q16 → `hp_max_q` | hot | `[1, 32767·FP_ONE]` | spawn writes `hp` |

`max_units` sits in the species row because the store is per species
(§1.1). A moth swarm and a fish school will want different capacities.

**Deferred to P3 (they belong to P3, which adds them as `SPECIES_FIELDS`
rows plus TOML values):**

- `speed` (+ the `speed·Δt` rule)
- `radius`
- the kill/coma thresholds, authored in Kelvin per the plan: `T_ctmin`,
  `T_coma_hyst`, `T_lethal`, `T_lethal_hot`
- the head-sweep sensing and tumble dials

**Deferred to P6:** `food_threshold`, `eat_start_prob`,
`eat_continue_prob`, `eat_rate`.

The chapter's `tumble_thermal_bias` and `heat_damage_coeff` are superseded
by the head-sweep and threshold rulings. P3 names what replaces them.

`config.toml` gets this, appended at EOF after `[recorder]`:

```toml
[swarm.larva]
# Swarm species row (engine/17 §2, arc #63). Schema lives in code
# (simulation/swarm.py SPECIES_FIELDS: kind, reload class, bounds); an
# unknown key or species here is a load error. Hot on Ctrl+R unless marked.
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
- a row has an unknown key or a missing key;
- a value has the wrong type or falls outside bounds;
- any `SPECIES_RULES` entry fails.

This closes the "no schema, typos surface as AttributeError or silent
`getattr` defaults" gap chapter 12 records, for this table.

The table also carries:

- `.rows` in `SWARM_SPECIES` order;
- `.row(name)`;
- `.hash`: blake2b-256 hex over `"SWARM_SPECIES_V1\n"` + `f"{sp}|{attr}|{value}\n"`
  over the stored ints, in species then schema order. Hashing quantized
  values means two TOML spellings of the same numbers hash equal.
- `.source`: the `CFG.swarm` Namespace object it was built from. It is
  never hashed or serialized.

### 4.3 Hot reload (fork 2c) — the first genuinely live table

In this codebase a reload is Ctrl+R → `CFG.reload()` (`debug_keys.py:53-55`;
chapter 12 §5; **F5 is the normal-map toggle**, `game_renderer.py:1502`,
despite the chapter's "F5" wording). `_load` re-wraps every section in a new
Namespace (`config.py:64`). The material, gas and weapon tables are
construction-bound and do *not* follow it (`weapons.py:31-36`; chapter 12
§5).

The swarm table does, lazily and for free:

1. `SwarmUnits.species` compares `getattr(CFG, "swarm", None) is
   gmap.swarm_species.source`, an O(1) identity check. On mismatch it calls
   `reload_species()`.
2. `reload_species()` rebuilds from CFG with full validation.
3. **Load-time fields keep their in-force value.** If a changed `max_units`
   is found, the new table keeps the allocated capacity and a single warning
   is printed: `[swarm] larva.max_units 8192 -> 4096 applies at the next
   reset/level load`. Arrays are never reallocated mid-run, and the residency
   guard would forbid it anyway.
4. The new table replaces `gmap.swarm_species`.
5. If `.hash` changed, it prints `[swarm] species table reloaded: <old12> ->
   <new12> (replay comparability ends here)`. This is chapter §2c's "reload
   invalidates the replay from that tick". No replay machinery exists yet,
   so the log line is the record.

**Reload-time validation failure** is a loud rejection. It prints
`[swarm] RELOAD REJECTED: <error>` and keeps the table in force, so no digest
state is touched and the new numbers simply do not apply. **Load-time
failure raises.** Decision: 2c chose hot reload *for tuning*, and a crashed
play session on a Ctrl+R typo is the most tuning-hostile outcome. The
rejection is not a silent fallback: the old table was valid, and the message
names the key.

**Which settings apply when:**

| setting | applies |
|---|---|
| `[swarm.*].hp_max` | Hot. The next spawn uses it; live units' `hp` is untouched. |
| `[swarm.*].max_units` | Load-time (next reset/level load). |
| `[recorder].swarm_units_cap` | Reset-bound: the recorder is built in `_reset_internal`. |
| `N_ENV`, `SWARM_SPECIES`, `ROSTER` | Code. |

A reload writes no gmap array, so **the species table never touches digested
state**. `species_hash` is provenance in the carrier, never hashed (§5.3),
the same as the entity `registry_hash`.

### 4.4 `swarm_fixed.py` (the Q16 boundary module, ruled name)

It follows the documented-twin pattern (`gas_fixed.py:37-66`,
`unit_fixed.py:43-72`):

- `FP_SHIFT`, `FP_ONE`;
- `quantize_scalar`, `quantize` (array), `dequantize`, `dequantize_f32`
  (round-half-away-from-zero, the `fixedpoint::quantize` twin);
- `PI_Q16`, `HEADING_PERIOD_Q16`, `wrap_heading_q16` (scalar and array).

It needs no `breach_physics` import. It uses no transcendental: π is a
checked-in constant, which is door 2. Never hardcode 65536 elsewhere.

---

## 5. `SWARM_SECT_V1`, the carrier, and the digest fold

### 5.1 "Present"

`present ⇔ ∃ species s, env e : gmap.swarm_<s>_high_water[e] > 0`, which
means at least one spawn ever succeeded on this GameMap (this run, since the
last reset/level load). This is chapter §6's "any slot ever spawned this
run". Because `high_water` is monotone, **presence is monotone within a
run**: a run whose larvae have all died still hashes its section (with
`valid=0` rows) and never falls back to the swarm-free hash.

### 5.2 The carrier (the `__entity__` idiom, `serialize.py:44-50, 200-241`)

Key `SWARM_DIGEST_KEY = "__swarm__"`. `swarm_carrier(gmap)` returns this
(copies, never views):

```python
{"present": bool,
 "n_env": int,
 "next_unit_id": ndarray (n_env,) int32,
 "blocks": {sp: {"high_water": ndarray (n_env,) int32,
                 "columns": {col: ndarray (n_env, max_e high_water[e])}}
            for sp in SWARM_SPECIES if any high_water > 0},   # SWARM_SPECIES order
 "species_hash": str}                                         # provenance, NEVER hashed
```

A gmap without swarm attributes (a test stand-in) yields
`present=False`, `n_env=0` and empty fields.

Capture paths **always** write the carrier. `require_swarm_carrier(gmap,
snapshot)` raises `KeyError` when a swarm-present gmap meets a snapshot
without `__swarm__` (the A4 strict rule). Snapshots from before P2 are
swarm-free by construction.

### 5.3 Byte encoding — `swarm_section_bytes(carrier)`, the ONE encoder

```
section := "SWARM_SECT_V1\n" "n_env|" N "\n" env{N} "end\n"
env     := "env|" e "|next_unit_id|" nid "\n" block*
           -- blocks in SWARM_SPECIES order; a species with high_water[e]==0 is OMITTED
block   := "species|" name "|high_water|" hw "\n" column{ROSTER}
           -- columns in ROSTER declaration order
column  := col "|" dtype.str "|" str((hw,)) "\n" raw
           -- raw = arr[e, :hw].tobytes(order="C"); big-endian dtype refused
```

Integers are ASCII decimal. The header pins name, dtype and length (the
`_field_bytes` idiom, `field_digest.py:108-115`), so parsing is unambiguous.
The `end\n` terminator makes the blob self-delimiting for save files.
**Capacity, the species table and slots at or beyond `high_water` are never
in the bytes.**

Omitting hw=0 species blocks is what makes "adding a species never bumps
another species' digest" (§9b) true. `decode_swarm_section(blob)` is the
strict inverse: it checks the preamble, `n_env`, known species in order,
dtype == roster, lengths and the terminator, and raises `ValueError` on any
deviation. It returns the carrier structure with `species_hash=""`.

Consumers of this encoder:

- the digest fold, which hashes it;
- the recorder-consistency gate (T22), which re-encodes recorded rows;
- save, which stores it.

That is A4's one-serializer principle (`serialize.py:164-171`). The
roster's own version rule (chapter §2: "additions bump the section version")
bumps the preamble to `SWARM_SECT_V2`, which is section-local.

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
  `test_w6_armory.py:566-588`)
- the B6 logic golden
- the vent/B1/B2 digests

So all of them are byte-identical with no regeneration.

Honest limits of that claim:

- A swarm-present trajectory is comparable across builds only at equal
  section version, and for behaviour patches at equal `species_hash`
  (recorded, like the entity registry hash).
- The food field (P6) **is** a `DIGEST_FIELDS` row and does take a spec
  bump (§12, item 3).

The spec toml gets an additive, documentary `[swarm_section]` table appended
at EOF, mirroring `[entity_section]` (`field_digest_spec.toml:77-92`). Its
`version` stays unchanged. The table records:

- `format = "SWARM_SECT_V1"`;
- the encoder `simulation.swarm.swarm_section_bytes`;
- the extent `[0, high_water)` per env and species, with hw-0 species blocks
  omitted;
- that capacity and the species table are never hashed;
- provenance = `species_hash`;
- the rule "per-unit swarm state never joins `fields`".

---

## 6. A/B harness and the xarch dumpers

**`tests/field_ab_harness.py`:**

- **Capture** (`capture_trajectory`, `:268-304`): after the entity carrier
  lines, add `snap[SWARM_DIGEST_KEY] = swarm_carrier(sim.gmap)` then
  `require_swarm_carrier(sim.gmap, snap)`. The carrier is always written;
  it has `present=False` on the canonical scenario.
- **Locator:** add `_diff_swarm_state(t, ca, cb)` beside `_diff_entity_state`
  (`:346-375`). It reports, as human-readable lines:
  - `present`, `n_env` and `next_unit_id` mismatches;
  - per-species `high_water` mismatches;
  - per column, the first differing `(env, slot)` with both values and both
    `unit_id`s;
  - a `species_hash` mismatch. That one is provenance, but still reported,
    since A/B runs must share tuning.
  `diff_trajectories` special-cases the key (after `:397-399`). The
  per-cell path never sees it.
- **Scenario:** add `swarm_scenario_sim()` after `default_scenario_sim`
  (`:109-139`). It takes the default scenario and, before the first step,
  uses explicit raw-Q16 values from `swarm_fixed` (no RNG) to:
  1. `sim.spawn_swarm` ~12 larvae at interior tile centres with fixed
     headings (including ±π);
  2. reclaim 3 of them, making holes;
  3. spawn 2 more, which fill the holes.
  The result has `high_water` greater than the live count and non-contiguous
  ids. It is the swarm-present fixture for P2's tests and a ready scenario
  for P3's lockstep.
- `SIM_FIELDS` is unchanged. In `__main__`, add `SWARM_DIGEST_KEY` to the
  special-key tuple at `:450-451`.

**`tests/_xarch_perfield_digest.py`:** in `build_perfield_lines`
(`:280-303`), after the unit lines of each tick, **only when the carrier is
present**, emit:

- one line per `(species, column)`, labelled `__swarm__.<sp>.<col>` (blake2b
  over `PERSWARM_V1\n` plus that column's env-ordered header+raw bytes);
- one `__swarm__.scalars` line (n_env, next_unit_id, high_waters).

Swarm-free files are byte-identical. A cross-machine diff then names the
diverging column. `GOLDEN_AGGREGATE` is untouched.

**`tests/xarch_digest.py`** (`:58-61`): a presence-gated suffix
`\tswarm=ssect_v1,sp=<species_hash[:12]>`. Swarm-free lines are unchanged.

---

## 7. Recorder (`src/simulation/recorder.py`)

**Which arrays:** every `ROSTER` column of every present species, plus the
per-tick `high_water` and `next_unit_id`. Values are **raw, at the roster
dtype — no dequantize, no cast** (chapter §6). This is a new ring family,
separate from the grid `fields`.

**API:**

- `__init__(..., swarm_units_cap=1024)`.
- `record(..., swarm=None)`, where `swarm` is the §5.2 carrier. The digest
  and the recorder consume the same carrier object.
- `_FakeGmap` recorder tests are unaffected, because the recorder never
  reads `gmap.swarm_*`.

**Rings.** They are allocated **lazily at the first present tick**, per
species:

- `ring[sp][col] = zeros((capacity, N, W), col.dtype)` with
  `W = swarm_units_cap`;
- `hw_ring[sp] = zeros((capacity, N), int32)`;
- one `nid_ring = full((capacity, N), FIRST_UNIT_ID, int32)`. Earlier rows
  are truthfully "nothing spawned".

Allocation prints its size. Each present tick writes
`w = min(hw_e, W)`, `ring[i, e, :w] = column[e, :w]` and `ring[i, e, w:] = 0`.
Species absent this tick write zeros and `hw 0`.

**Dump** (presence-gated: no swarm → no `swarm_*` keys and a key set
identical to today, the entity precedent at `recorder.py:282-302`) writes:

- `swarm_<sp>_<col>` as `(T, N, wmax)`, where `wmax = min(W, max recorded
  hw)`. That is the `[0, high_water)` bounding.
- `swarm_<sp>_high_water` as `(T, N)`.
- `swarm_next_unit_id` as `(T, N)`.
- `swarm_species_hash` as a 0-d str.

Truncation is visible offline when `high_water > wmax`. Update the module
docstring's frozen-schema list with these additive keys.

**Dtype classes (the CLAUDE.md recorder rule).** Today the rings are
**grid** rings of bool, int64 or float32. int32 grids are dequantized into
float32 (`recorder.py:113-121, 190-195`). In the `.npz`:

- **uint8 (`valid`) and uint32 (`dist_walked`) are NEW dtype classes.** No
  key carries them.
- **int32 is new as a ring dtype.** It exists in the file only as dump-time
  `unit_fx`/`tick_ids`.

The rule ("a contract extension with its own ring branch, never a cast") is
met **by construction**: the swarm family's one branch takes each ring's
dtype verbatim from `ROSTER`, so no column can be cast. T22 asserts
`recorded.dtype == roster dtype` for every column.

**The cap (chapter §2b "capped by config"):** `[recorder] swarm_units_cap =
1024`, added to `config.toml:2435-2438` and restart-bound. With `tps = 24`,
the ring is `round(100 s · 24) = 2400` slots (`config.py:143`; the TOML
comment "1200 @ 12 tps" is stale). At 33 B per unit per tick that is
**≈ 77 MB** when a swarm is present, comparable to the grid rings. It covers
the "infestation" density (~1,000). A cap of 2048 would be ≈ 155 MB.

---

## 8. Save / load

No save/load code exists (`simulation.py:53-57`; engine/02 "Implementation
status"). P2 therefore delivers the primitives the future 02 save contract
calls, gated by an exact round trip, and **no file format, no keys and no
UI**.

- `export_state() -> SwarmSave(section: bytes, species_hash: str)`
  (a NamedTuple). `section` is the §5.3 encoding of the full carrier. The
  encoding is total, so an absent store encodes with no blocks.
- `import_state(save)`:
  1. decode and validate fully first (atomic): `n_env == N_ENV` and every
     species' `high_water ≤` its capacity;
  2. zero the **entire** store in place (residency-safe);
  3. write each saved `[0, hw)`, set `high_water`, set `next_unit_id`;
  4. return the saved `species_hash`.

  The hash-mismatch policy belongs to the 02 contract, which also owns replay
  validity.

**Round-trip semantics.** Beyond `high_water` the store is all-zero by
invariant (never spawned). Import restores the whole store **byte-identical
over full capacity**, not just `[0, hw)`. Continuation is identical too: the
next spawn takes the same slot and the same id in both.

Why this lives in `swarm.py` and not `entities/serialize.py`: that module is
stdlib-only, with no numpy by rule (`serialize.py:5-9`), and swarm units are
deliberately not entities (chapter §1). The swarm keeps A4's
*principle* (one encoder for digest, recorder and save) in its own module
(§12, item 5).

---

## 9. C++: none in P2 — and the contract P3 inherits

**P2 writes no C++** (no `swarm.h`, no CMakeLists line). Nothing consumes a
header before P3's kernel. A header with no consumer and no gate is pure
drift risk. P2's Python constants are the source that P3 twins.

P3's `cpp/src/swarm.h` twins these and identity-tests them against Python
(the philox32 cross-language precedent):

- the `ROSTER` dtypes;
- the `AI_*` enum values;
- `PI_Q16` / `HEADING_PERIOD_Q16`;
- `wrap_heading_q16`, with a **floor**-mod in int64;
- `FIRST_UNIT_ID`;
- the all-zero empty-slot form.

Other constraints P2 hands P3, and nothing more:

- The kernel/twin loop runs over `[0, high_water)` and skips `valid == 0`.
- The death branch writes the empty-slot form (all columns 0) and emits the
  event from the pre-zero values.
- P3's residency transfer names the store arrays in slot 7's D2H, and
  uploads after host spawns while resident (the explicit-list rule).
- The species-table H2D happens when `species.hash` changes.
- New tunables are `SPECIES_FIELDS` rows plus `SPECIES_RULES`.

---

## 10. Merge-friendliness vs `fire-12`

Checked with read-only `git diff $(merge-base) fire-12`. fire-12's hunks:

- `gamemap.py`: old lines 8-14, 130-192, 251-257, 404-517, 1295+
- `simulation.py`: 1614-1628
- `field_digest.py`: 68-115 (version → **6**, `+dyn_heat_atten_q`)
- `field_ab_harness.py`: 85-90 and 508-520
- `_xarch_perfield_digest.py`: 222-225 (the golden)
- `field_digest_spec.toml`: 8-11 and 57-67
- `config.toml`: up to old 2347
- `recorder.py` and `entities/serialize.py`: untouched

| shared file | P2 footprint | overlaps fire-12? |
|---|---|---|
| `gamemap.py` | 1 import (after `:56`); 1 line before `:223`; 3-line block at end of `__init__` (`:824`) | No (gaps 193-250, 518-1294) |
| `simulation.py` | import (`:100`), 1 line (`:264`), 2 delegating methods (~`:456`), 2 kwargs (`:405-409`, `:1582-1584`) | No |
| `field_digest.py` | fold in `tick_digest` body + docstring line | No |
| `field_digest_spec.toml` | `[swarm_section]` at EOF | No |
| `field_ab_harness.py` | imports (~`:50`), capture (+2), locator fn, scenario fn, diff special-case, `__main__` tuple | No |
| `_xarch_perfield_digest.py` | import + presence-gated lines in `build_perfield_lines` | No |
| `config.toml` | `swarm_units_cap` in `[recorder]` + `[swarm.larva]` at EOF | No |
| `physics_runner.py`, `CMakeLists.txt` | **untouched** | — |

The body lives in the new files `swarm.py` and `swarm_fixed.py` plus the
tests.

---

## 11. Verification plan (tests are written from this list)

Each test names its property and the change that must break it. None pins
the roster's size or order. Loops run over `ROSTER`, `SPECIES_FIELDS` or
`SWARM_SPECIES` and assert the rule each member obeys.

- Store/lifecycle tests use `GameMap(_scenario_level())` + `SwarmUnits`, or
  `Simulation(..., breach_physics=None)` (fast, no build needed).
- Harness/golden tests need the CPU build.
- Capacity and reload fixtures replace `CFG.swarm` with a fresh
  `config.Namespace` (that is what `CFG.reload()` does). They do not
  monkeypatch attributes in place, which would not change identity.

**`tests/test_swarm_digest.py`:**

| # | property | must break if… |
|---|---|---|
| T1 | Every existing golden is byte-identical. Oracle: the full suite green with **zero** golden edits (`test_w6_armory` canonical golden + RNG canary, B6, vent/B1/B2), plus the `git diff` of P2 touching no golden value. | the fold is unconditional, capacity is hashed, or store arrays join `DIGEST_FIELDS`/`SIM_FIELDS` |
| T2 | Absence transparency: `tick_digest` of a snapshot with no key, with a `present=False` carrier, and a frozen verbatim pre-P2 `tick_digest` copy (the `_pre_a4_tick_digest` idiom, `test_entity_digest.py:115-128`) are all equal, on a synthetic snapshot with and without an entity carrier. | the marker or bytes are hashed when absent |
| T3 | The section is not vacuous: for **every** ROSTER column, perturbing one slot inside `[0, hw)` (including a reclaimed `valid=0` slot) changes `tick_digest`, and restoring it restores the digest; the same holds for `next_unit_id` and `high_water`; `field_digest` never changes. | a column is dropped from the encoder, only live slots are hashed, or scalars are omitted |
| T4 | Capacity moves nothing: identical spawns with `max_units` 8192 vs 64 give identical section bytes and `tick_digest`; writing a value at a slot ≥ `high_water` leaves the digest unchanged. | full capacity is serialized |
| T5 | Presence is monotone: spawn 1, reclaim it → `live_count == 0` yet `present`, and the digest ≠ the swarm-free digest. | presence is keyed on live count |
| T6 | Strict carrier: a present gmap plus a snapshot without `__swarm__` raises `KeyError`; a swarm-free gmap accepts; `capture_trajectory` writes a carrier on every tick. | a capture path skips the carrier |
| T7 | Harness: `capture_trajectory(make_sim=swarm_scenario_sim)` twice → `assert_trajectories_match` at tol 0; a one-slot perturbation is reported with species/column/env/slot/unit_id. | the locator treats the carrier as opaque or ignores it |
| T8 | P2 swarm is inert on fields: per tick, `field_digest(swarm_scenario) == field_digest(default_scenario)` and the tick digests differ. | larvae write a field — **expected to break at P6 (food depletion); retire it there** |
| T9 | xarch presence gate: `build_perfield_lines` on a swarm-free trajectory has no `__swarm__` label and one line per (species, column) + scalars per tick when present. | swarm lines appear unconditionally |

**`tests/test_swarm_store.py`:**

| # | property | must break if… |
|---|---|---|
| T10 | Every ROSTER column (and each per-env scalar) exists on GameMap with shape `(N_ENV, max_units)` / `(N_ENV,)`, the roster dtype, C-contiguous, little-endian, starting in the empty form, and **is in `GameMap._RESIDENT_FIELD_NAMES`** (born through residency). | a column is mirror-only, has the wrong dtype, or is not resident |
| T11 | Spawn takes the lowest free slots in request order: fresh store → `0..k-1`; after reclaims, a batch of m gets `sorted(free_before)[:m]`. | append-at-hw, LIFO, compaction |
| T12 | Model-based: after a seeded random interleaving of spawn/reclaim batches (seeded `np.random.default_rng` **in the test**), the store equals a ~20-line reference model. `high_water` never decreases and == 1 + max slot ever occupied; every assigned `unit_id` is unique and strictly increasing; `next_unit_id == FIRST_UNIT_ID + total spawned`; a reused slot gets a new id. | id = slot, id reuse, hw recomputed from live slots |
| T13 | Reclaim writes the empty-slot form: every column of a reclaimed slot is byte-equal to a never-spawned slot's (so `valid=0`, `ai_state=AI_DEAD`). | reclaim that only clears `valid` |
| T14 | Atomic refusal: overfilling a small store raises `SwarmFull`; bad inputs (out-of-grid pos, float pos, bad env, unknown species, reclaim of an empty/duplicate/≥hw slot) raise. In every case all arrays and scalars are byte-identical to before. | partial fill / partial write |
| T15 | Heading is canonical: stored heading ∈ (−PI_Q16, PI_Q16] and ≡ input mod `HEADING_PERIOD_Q16` for inputs including ±PI_Q16, ±3·PI_Q16 and INT32 extremes; `wrap` is idempotent. | raw heading stored, or a `quantize(2π)` period |
| T16 | The lifecycle draws no randomness: spawn/reclaim leave `sim.rng.bit_generator.state` unchanged (the `test_w6_armory.py:548-563` canary idiom). | spawn draws from `sim.rng` |

**`tests/test_swarm_species.py`:**

| # | property | must break if… |
|---|---|---|
| T17 | Strict schema: a missing `[swarm]`, a missing species row, an unknown species, an unknown key, and, **for every `SPECIES_FIELDS` entry**, a wrong type and an out-of-bounds value each raise a `ValueError` naming the key at load (`GameMap(...)` construction). | Namespace pass-through without schema |
| T18 | Hot reload touches no digested state: spawn → digest D0 → replace `CFG.swarm` with `hp_max` changed → access `sim.swarm.species` → digest still D0 and every store array unchanged; `carrier()["species_hash"]` changed; the next spawn's `hp == new hp_max_q`. | reload rewrites live `hp`, or the table is hashed into the section |
| T19 | Load-time field: a reload with `max_units` changed leaves capacity and the in-force value unchanged and prints the warning once; `sim.reset()` picks up the new capacity. | mid-run reallocation |
| T20 | Rejected reload: an invalid reload value keeps the prior table (same `.hash`), prints `RELOAD REJECTED`, and touches no array. | crash or silent partial apply |

**`tests/test_swarm_save_recorder.py`:**

| # | property | must break if… |
|---|---|---|
| T21 | Save round trip exact: export from `swarm_scenario_sim` → import into a fresh sim of the same level/config → every store array byte-identical over **full capacity**, scalars equal, `export_state()` bytes equal; a following spawn in both takes the same slot and id. `decode(encode(c))` reproduces `c`; truncated, bit-flipped-header and wrong-dtype blobs raise; import leaves the store untouched on a decode failure. | `next_unit_id` is dropped, the store is not zeroed, only live slots are serialized, or the decode is lax |
| T22 | Recorder: a swarm-free dump has no `swarm_*` key. Swarm-present, **every** ROSTER column is recorded at exactly its roster dtype, with width `min(max hw, cap)`. Re-encoding recorded row t through `swarm_section_bytes` equals the section bytes the digest hashed at tick t (digest and recorder consistency). With `cap < hw` the width is `cap` and `swarm_<sp>_high_water` shows the true hw. | a float32 ring or cast, full-capacity recording, or recorder/digest drift |

Existing gates that must stay green untouched: `test_ingress_lint.py` (it
scans `swarm.py`/`swarm_fixed.py`), `test_entity_digest.py`,
`test_recorder_dump.py`, `test_vent_dormancy.py`, and the residency-membership
tests. The plan gate is "full suite, no new reds; zero golden edits".

---

## 12. Conflicts found (chapter/task vs code) and their resolutions

1. **`T_prev`** appears in chapter §2's table, §4, §4.1 and §6's gate text
   ("fuzz … incl. `T_prev`"), but the Review log ruled it out. It is
   excluded from ROSTER.
2. **"F5 hot reload"** (§2c): the code's reload is **Ctrl+R**
   (`debug_keys.py:53-55`), and F5 toggles normal maps
   (`game_renderer.py:1502`). Resolved as Ctrl+R/`CFG.reload()` with lazy
   identity detection (§4.3). This also makes the swarm table the first
   table in breach that is really live on reload, where materials, gases and
   weapons are construction-bound.
3. **"Food = spec v6"** (§6/§11): fire-12 already moves
   `DIGEST_SPEC_VERSION` 5 → **6** (T5b, `+dyn_heat_atten_q`,
   `fire-12:tests/field_digest.py`). P6's food bump will be the next free
   number after fire-12 merges (v7). P2 is unaffected: it has no spec bump.
4. **Save** (§6): the "(designed, unbuilt) 02 save contract" is confirmed
   unbuilt. P2 ships round-trip primitives only (§8).
5. **Task: "save via the one serializer (`entities/serialize.py`)"**: that
   module is stdlib-only by rule and swarm units are not entities. Resolved:
   the swarm gets its own one encoder with the same digest/recorder/save
   principle (§5.3, §8).
6. **Recorder "rings `(capacity, max_units)`"** (§2b) at the real
   `capacity = 2400` would be ≈ 620 MB at 8192. Resolved: the ring is
   `(capacity, N, swarm_units_cap)`, capped at 1024 by config, with
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

---

## 13. Build brief (ordered; a Sonnet implementer executes this without re-deriving)

1. **`src/simulation/swarm_fixed.py`** (new): §4.4. Module docstring in the
   `gas_fixed` style, citing `fixed_point.h:890-903` for `PI_Q16`.
2. **`src/simulation/swarm.py`** (new), in this order:
   - module docstring (engine/17 §2/§6/§7/§9b, this doc);
   - `N_ENV`, `SWARM_SPECIES`, `Column` + `ROSTER`, `AI_*`, `FACTION_NONE`,
     `FIRST_UNIT_ID`;
   - `store_attr`, `high_water_attr`, `NEXT_UNIT_ID_ATTR`,
     `SWARM_RESIDENT_NAMES`;
   - `SpeciesField`, `SPECIES_FIELDS`, `SPECIES_RULES`, `SwarmSpeciesRow`
     (a schema-built immutable record with attribute access), and
     `SwarmSpeciesTable` (`from_config`, `.rows`, `.row`, `.hash`, `.source`);
   - `allocate_swarm_store`;
   - `SwarmFull`;
   - `SWARM_DIGEST_KEY`, `SWARM_SECT_PREAMBLE`, `swarm_present`,
     `swarm_carrier`, `swarm_section_bytes`, `decode_swarm_section`,
     `require_swarm_carrier`, `SwarmSave`;
   - `SwarmUnits` (§3 table).

   It must not import `gamemap` or `simulation.simulation` (the import
   direction is gamemap → swarm).
3. **`config.toml`**: `swarm_units_cap = 1024` (with a memory comment) in
   `[recorder]`; `[swarm.larva]` block at EOF (§4.1).
4. **`src/simulation/gamemap.py`**:
   - import `SWARM_RESIDENT_NAMES, allocate_swarm_store` after the gases
     import (`:56`);
   - the `_RESIDENT_SYNCED` extension line + caveat comment immediately
     before `_RESIDENT_FIELD_NAMES` (`:223`);
   - `allocate_swarm_store(self)` block after `self.refresh_gas_energy()`
     (`:824`).
5. **`src/simulation/simulation.py`**: the §3 hooks 1-4.
6. **`src/simulation/recorder.py`**: §7 (constructor kwarg, `record(swarm=)`,
   lazy rings, dump keys, docstring schema).
7. **`tests/field_digest.py`**: the §5.4 fold + one docstring sentence beside
   the A4 note (`:24-28`).
8. **`tests/field_digest_spec.toml`**: `[swarm_section]` at EOF (§5.5).
9. **`tests/field_ab_harness.py`**: §6 (imports, capture +2 lines,
   `_diff_swarm_state`, the `diff_trajectories` special-case,
   `swarm_scenario_sim`, the `__main__` tuple).
10. **`tests/_xarch_perfield_digest.py`** and **`tests/xarch_digest.py`**: §6.
11. **Tests**: the four files of §11.
12. **Verify**, running `C:/Users/steen/anaconda3/python.exe -m pytest tests -q`
    (conda env `data`):
    - no new reds;
    - `git diff` shows no golden value changed;
    - `tests/field_ab_harness.py` `__main__` still self-matches;
    - `_xarch_perfield_digest.py` reports `matches golden = True`.
13. **`CLAUDE.md`**: write the §15(b) rows pointing at the real code (the
    rules lifecycle).

Stage explicit paths only; never `git add -A`.

---

## 14. For Erik

None blocking. Every call above is inside the blessed chapter or is
Claude-fixable detail with its reason stated. Two things are worth a glance
because they touch your tuning loop:

- A Ctrl+R with a bad `[swarm.larva]` value is **rejected loudly and the old
  table kept**, rather than crashing the session (§4.3).
- An unknown key in `[swarm.*]` is a load error (§4.2).

---

## 15. Systems

**(a) Existing canonical systems P2 uses (CLAUDE.md table) and the seam:**

- **GameMap** — stores the arrays and `swarm_species` (fork 2a). One hook at
  the end of `__init__`. No topology or field writes.
- **Simulation facade** — owns `sim.swarm`. Public mutation via
  `spawn_swarm`/`reclaim_swarm`. No `get_state` change.
- **Tick conductor** — **no new slot** in P2. It only gains a kwarg on the
  existing recorder line.
- **Config** — `[swarm.larva]` and `[recorder] swarm_units_cap` via `CFG`.
  Ctrl+R reload is honoured through CFG-identity.
- **Q16 boundary modules** — the new `swarm_fixed.py`; no inline 65536.
- **Fixed-point kits** — no call in P2. `PI_Q16` is pinned to the kit's
  narrowed π (`fixed_point.h:900-903`); P3 twins the heading law.
- **Field digest** — the `tick_digest` fold only. `DIGEST_FIELDS` and the
  spec version are untouched.
- **GOLDEN_AGGREGATE** — unchanged (the gate).
- **A/B lockstep harness** — carrier capture, strict rule, slot locator,
  swarm scenario.
- **Recorder** — an additive, presence-gated per-unit ring family. It adds
  the new dtype classes uint8 and uint32 at roster dtype, with no cast.
- **Entity system** — its carrier / strict-presence / section-local-version
  **pattern is copied**, but the module is not reused (swarm units are not
  entities; `serialize.py` is numpy-free by rule).
- **Ingress lint** — scans the new modules.
- **RL-batch habits (§A)** — `(N_ENV, …)` birth, residency from day one, no
  host tick logic.
- **Material/gas table pattern** — `from_config` at GameMap construction.
- **Test conventions** — property tests plus a model-based oracle.
- **Not used in P2:** FieldEdit, PhysicsRunner/PhysicsEngine, the coupling
  table, tick events, the CUDA harness (P3's 3-part gate).

**(b) New systems P2 creates — DRAFT one-line CLAUDE.md rules (written at
implementation, pointing at the real code):**

- **Swarm store** — `src/simulation/swarm.py` (`ROSTER`,
  `allocate_swarm_store`) + `gmap.swarm_<species>_<col>`: THE per-unit
  state of swarm units. It is SoA `(N_env, max_units)` per species, born in
  `GameMap._RESIDENT_SYNCED`. Slots are stable and never compacted. Spawn
  takes the lowest free slots in request order. A freed slot is the all-zero
  form. `unit_id` (per env, from 1, never reused) is identity; the slot is
  not. A new column is a `ROSTER` row plus a `SWARM_SECT` version bump,
  never a stray gmap array or a `DIGEST_FIELDS` row.
- **Swarm digest section** — `swarm.py::swarm_carrier` /
  `swarm_section_bytes` + `field_digest.tick_digest`: per-unit swarm state
  enters determinism artifacts ONLY through the presence-gated `__swarm__`
  carrier (`SWARM_SECT_V1`, extent `[0, high_water)`, hw-0 species blocks
  omitted). Capture paths always write the carrier. Digest, recorder and
  save consume the one encoder. Capacity and the species table never enter
  hashed bytes.
- **Swarm facade** — `SwarmUnits` (`sim.swarm`; public writes via
  `Simulation.spawn_swarm`/`reclaim_swarm`): the only host-side lifecycle
  writer. Never index `gmap.swarm_*` to write from gameplay or tools. Spawn
  takes explicit Q16 values and draws no RNG; callers own randomness.
- **Swarm species table** — `swarm.SPECIES_FIELDS` / `SPECIES_RULES` +
  `[swarm.<species>]`: a swarm tunable is a schema row (kind, reload class,
  bounds) plus a TOML value. It is validated strictly at load and on every
  reload (unknown keys or species are errors) and quantized at ingress via
  `swarm_fixed`. `hot` fields apply on Ctrl+R (lazy CFG-identity; an
  invalid reload is rejected loudly), and `load` fields apply at the next
  reset. The table hash is provenance, never digested. The species *set* is
  closed in code (`SWARM_SPECIES`, append-only).
- **Swarm heading law** (folds into the existing "Q16 boundary modules" row)
  — `swarm_fixed.wrap_heading_q16`: every heading write is canonical
  (−PI_Q16, PI_Q16] with period `2·PI_Q16`. The C++ twin (P3) is
  identity-tested against it.
