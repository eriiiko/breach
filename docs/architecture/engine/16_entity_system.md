# 16 — Entity System (foundation)

**Depends on:** 02 (state & ownership — GameMap is the field truth), 03 (material system —
`MAT_DOOR_CLOSED` and the table-projection caches), 04 (atmosphere & pressure — the §2.2
occupancy-transition rule the seal/unseal primitives implement), 13 (FieldEdit — whose
structural carve-out door flips live in), 14 (determinism — digest/attestation the entity
sections join), 15 (level authoring — the format `[[entity]]` extends).

**Status:** Arc A foundation ✅ (A1–A9 merged 2026-07-19, Erik blessed) · signals/sensors/logic
✅ (Arc B, B1–B7 merged 2026-07-22, Erik blessed — §8) · editor UX ✅ (Arc C, C0–C9 merged
2026-07-22, merge `54cd6cd`).

> Canonized 2026-07-19 at Arc A close. The design record is
> `docs/entity_system_design_2026-07-18.md` (LOCKED model, erratad as-built) +
> `docs/level_editor_v3_design_2026-07-18.md` (LOCKED view). Build history:
> `docs/archive/arc_a_patch_plan_2026-07-18.md` (rulings 1–5),
> `docs/archive/a4_digest_impl_note_2026-07-18.md`,
> `docs/archive/a5_evacuation_impl_2026-07-18.md`,
> `docs/archive/a6_doors_v0_impl_2026-07-19.md`,
> `docs/archive/a7_rebaseline_rationale_2026-07-19.md`.

---

## 0. What an entity is

One model for everything placeable that isn't paintable matter: doors, sensors, logic
nodes, pumps, lights, the data chip — instances of registry classes, connected by signals
(Arc B). Logic is dataflow over integer signals, never a scripting language. **Units are
NOT entities in v1** — they keep `[[spawn]]` and their own data model
([mechanics/01 Units](../mechanics/01_units.md)); the convergence contract (one shared id
space, unit signals, digest merge) executes at stack-2, and until then the loader rejects
any wire/tag/sensor referencing a unit.

Three layers, pinned so "where does programming live" never blurs:

| Layer | Lives in | Example |
|---|---|---|
| **L0 — behavior** | Python code, `src/simulation/entities/` | what "door closed" means physically |
| **L1 — vocabulary** | the class declarations themselves + `entities.toml` tuning overlay | the `door` class exists, has `initial_state`, emits `is_open` |
| **L2 — composition** | `level.toml` `[[entity]]` blocks (editor or `level_lib`) | THIS door at THIS wall run |

No L3 (embedded scripting): anything dataflow can't express becomes a new L0 node class —
testable, deterministic, reusable.

## 1. Registry-in-code (A1)

Each entity kind is a subclass of `Entity` (`src/simulation/entities/schema.py`) declaring
fields/signals/inputs as class-level schema beside its L0 behavior; a registration
decorator adds it to the registry. Consumers **import the module directly** — no parallel
JSON schema to go stale. Constraints, both CI-tested:

- **Import-light rule:** `import simulation.entities` succeeds with no compiled
  `breach_physics` and without pulling `simulation.simulation` — the editor may import it
  cold.
- **Editor failure mode:** on import failure the editor falls back to the last-good
  `entity_registry.json` (rewritten on every successful launch) + a red banner; the
  palette stays usable minus the broken class.
- `entities.toml` holds tuning-number overrides only, **dev-only**: in any lockstep
  session or ML rollout the `registry_content_hash()` (which folds the overlay's effective
  defaults) is match-setup material, like the seed; changing a behavior-relevant number is
  a deliberate golden re-baseline event.

Shipped classes: `light` (the A1 exemplar), `door` (A6), zone classes (A8).

## 2. `[[entity]]` format, ids, migration (A3, A7)

- Every instance carries a **mandatory unique `id`** (slug like `door_3`); all references
  — wires, entity_refs, zone bindings — address ids, never array positions. Duplicate ids
  hard-error; dangling refs warn at load (authoring error) but a runtime-destroyed target
  is legal. Ids assign in file order; **every runtime sweep iterates in id (ordinal)
  order** — the single ordering rule that makes all entity iteration deterministic.
- Legacy `[[spawn]]`/`[[light]]` remain as aliases (`[[spawn]]` permanently — units are
  not entities); **mixed legacy+new forms in one file hard-error**. Migration is explicit,
  never a save side effect: `tools/migrate_level_entities.py` converts a level in place
  (`.bak`), grouping painted `MAT_DOOR` runs into door entities.
- Instances carry `tags = [...]`; wire targets are `id` or `tag:name`, resolved at
  runtime in member-id order (Arc B consumes this).
- A7 migrated **`test_level` only** (ruling 2): 5 doors + 8 lights. Shipped/showcase
  levels (`unhcr_vessel`, `playground`, …) stay legacy until Erik chooses, likely Arc C.
  The arc's one sanctioned re-baseline turned out **empty of committed artifacts** —
  every committed golden derives from synthetic in-memory scenarios, so only
  `test_level`'s (unpinned) digest identity flipped; evidence of record in
  `docs/archive/a7_rebaseline_rationale_2026-07-19.md`.

## 3. `level_lib` — THE data layer (A2)

`src/level_lib.py` is the **single read+write layer for level files, period**: the loader's
read side, the editor's managed-block writeback, the migration tool, and ML variant
generation are all clients. One writer implementation, ever. Writes are atomic
(temp + rename); mtime+hash recorded at load/save so the editor can prompt
reload-or-overwrite on a two-writers conflict; byte-stable round-trip is tested over every
repo level. Known gap: the baker's `[art]`/`[bake]` writeback is not yet a client
(accepted A2 gap; Arc C rider).

## 4. Digest sections (A4)

Entity state joins the ch. 14 bit-identity gate as two appended sections — designed so the
**dormancy guarantee** holds: an entity-free level hashes byte-identical to pre-entity
code, and no pre-existing golden was ever regenerated.

- **Absence-transparent fold:** `tick_digest` appends `|__entity__|…|__signals__|…` iff
  `n_entities > 0`; `DIGEST_SPEC_VERSION` stays 1. Format changes bump the section-local
  preambles (`ENTITY_SECT_V1\n` / `SIGNAL_SECT_V1\n`), never the global version.
- **One serializer** (`entities/serialize.py`) feeds digest, recorder, and `get_state` —
  per entity in ordinal order: header, declared synced-kind fields as signed little-endian
  int64 (KIND_LENGTH_M excluded — authoring-bound; its synced consequence is the tile
  state already hashed), the free `alive` row, then a per-class **runtime-row block**
  (door `state`/`want_open`/`hp_*` live here — runtime rows, not schema fields).
- **Strict carrier:** snapshots always carry `__entity__ = {"n_entities": N, …}`; a
  sim-with-entities whose snapshot lacks the key **raises** — no capture path can silently
  compute an entity-free digest for an entity-present run. Pre-A4 recordings (no key) are
  entity-free by construction.
- Entity-present digests are only comparable at equal `registry_content_hash()`; the
  x-arch artifact line appends `ents=N,esect_v1,reg=<hash[:12]>` so cross-machine
  mismatches are attributable in one diff.

## 5. Seal/unseal primitives (A5)

`GameMap.seal_tiles(span, material_id)` / `unseal_tiles(span)` / `can_seal_tiles(span)`
(`src/simulation/gamemap.py`, beside `destroy_wall`) implement the ch. 04 §2.2
occupancy-transition rule for door flips — pure integer, atomic (validate-then-mutate,
raise-free mutation pass), pinned iteration orders (row-major span; N,S,E,W receivers).

- **Seal** evacuates each tile's gas to open neighbors (equal split, remainder to the
  first receivers in N,S,E,W order) — exact to the LSB, all 7 slices — then writes the
  solver-owned fields to their solid steady state. **Close-T** (ruling 4 amendment): the
  sealed tile's temperature becomes the integer mean of its PRE-call solid neighbors
  (the panel takes the wall assembly's temperature; local air-T fallback) — no instant
  hot door from post-grenade air.
- **Unseal** seeds the opened tile **conservatively**: the seed is withdrawn from the
  donors, equalizing over donors plus the opened tile (the **k+1 divisor**) — opening a
  door does not create air. Cycling a door is exactly N-conserving, forever (tested ×100
  cycles, per-slice). Vacuum-adjacent opens join vacuum and seed nothing.
- **`destroy_wall` stays minting** (untouched canon): destruction events (breach,
  burn-through, burst, bullet chew) keep the neighbor-mean seed. The asymmetry is
  deliberate and bounded — destruction is one-shot per tile against a finite wall stock,
  so the mint is not an agent-cyclable pump; door cycling, which IS agent-cyclable, is
  exact. (Erik's rubble model: debris displaces volume, pressure constant.)
- **Refusals:** sealing over `water_depth > 0` → `SealBlocked` (a bit-conserved field the
  solver would silently delete); gas-holding tile with zero receivers (sealed pocket) →
  `SealBlocked`, never delete; caller bugs (OOB, duplicates, already-solid, non-solid
  material) → `ValueError`; receiver int32 overflow → loud `OverflowError`, pre-mutation.
  `can_seal_tiles` predicts all of it exactly. Unit occupancy is caller policy — the
  primitive never sees units.

## 6. Doors v0 (A6)

The first real entity: class `door` (`entities/door.py` schema + span math,
`simulation/door_system.py` runtime + sweep).

- **`MAT_DOOR_CLOSED` (material id 7):** a closed entity door is FULLY solid — flow and
  movement — which one material id cannot share with legacy painted `MAT_DOOR`
  (`mobility = 1000`, the walkable-but-flow-solid hybrid of the door-stamp-leak fix). The
  fork keeps ruling 2 (shipped levels unchanged) and rides every existing table-projection
  seam — movement, LOS, flow, burst, burn, pathing — with zero query-code changes. Open
  door tiles are plain `MAT_AIR`. The blast gate (`physics.py:104` hardcoded tuple) was
  extended to include it — flagged wart, see status below.
- **Span** = anchor tile + `orientation` + `length_m`, quantized ONCE at base resolution
  (exact `Fraction`, round-half-up), then replicated by `--res` factor (`res_factor` +
  `tile_size_m_base` carried on `LevelData` so upscaling can't corrupt the quantization).
  Default 1.0 m → 3 tiles. Load order: door tile state stamps **before** field seeding, so
  authored-open ≡ authored-air is field-bit-identical (the digest legitimately differs —
  it hashes the entity records; "loads bit-identical" means FIELD state).
- **Slot 9e sweep** (after 9d ignition, before the recorder snapshot), per door in
  ordinal order: reconcile external destruction, then apply the latch. Open =
  `unseal_tiles` (unconditional); close = `occupancy_clear(span) and can_seal_tiles(span)`
  → `seal_tiles(span, MAT_DOOR_CLOSED)`. Effects reach the solvers next tick via the
  step-6 restamp — the one-tick-delay story.
- **`want_open` latch (ruling 5):** synced entity state, hashed as a runtime row, with
  retry-until-clear semantics — a blocked close is not consumed, it retries every tick
  (this IS Arc B's while-held `close` input with the latch standing in for the wire). The
  dev-only **O key** flips the latch under the cursor (render/dev plumbing, same
  citizenship as I-ignite; the water-optics toggle moved to V). No key press →
  bit-identical trajectory.
- **HP is a per-tile runtime vector** (`hp_0…hp_{k-1}`, row-major span order): folded from
  `wall_hp` on open, restamped after close (undoing `on_tile_changed`'s table
  re-quantize) — no free heal by cycling, no damage smear; exact round-trip.
- **Whole-door-dies:** any span tile externally destroyed (burst, burn-through, blast) →
  the entity destroys its remaining tiles via `destroy_wall`, `alive = 0`,
  `state = DESTROYED`, latch dead, never re-seals. No stub panels. A door slammed across
  a >`burst_threshold` differential pops next tick — deliberate relief-valve physics.
  One accepted **boundary-tick lag**: phase-boundary explosive volleys run after slot 9e,
  so exactly one recorded tick may show CLOSED rows over minted-air tiles — deterministic,
  reconciled next tick, pinned by test.
- **Path-hold:** a unit whose next path tile turned impassable holds position and burns
  the tick (the can_move-suppression pattern); resumes when the door reopens. No re-path
  in v1 (WEGO: the plan is the plan). Zombies need nothing — their per-step
  `is_passable_block` re-check already stops them.

## 7. Zones, air_init, boundary (A8, A9)

- **Zones:** `zones.npy` integer paint-id grid + one `[[entity]]` instance per painted id
  (`zone_id`, breach-site `roster = [[unit_type, count], …]` in unit-system vocabulary).
  Validators: painted id ↔ exactly one instance; zero-tile zones warn; duplicate
  `zone_id` errors. `_upscale_level` replicates the grid like water.
- **`air_init.npy`** — authored initial-atmosphere seeding at load.
- **`boundary`** level field (`"space" | "ambient"`) — format + loader only; the
  semantics belong to the boundary-conditions physics project (ledger stack #1).

## 8. Signals, sensors & logic — Arc B (as-built)

The logic layer: dataflow over integer signals (never a scripting language). Design record
archived at `docs/archive/arc_b_impl_2026-07-21.md` (v2, 3-lens critique folded — 14
blockers; the resolution ledger is §0b there). B1–B7 merged 2026-07-22, Erik blessed
(airlock HUMAN-TEST). All state here is synced Q16.16 integer, digest-gated, recorder-joined.

- **SignalBus** (`src/simulation/signal_bus.py`): dense `pub`/`stg` int64 buffers over a
  frozen `(ordinal, name)` slot table. Built **only** when `sensors ∪ nodes ∪ wires ≠ ∅`,
  and enumerating **only** wire-referenced / sensor- / node-emitted signals — so an unwired
  door contributes no slot and a logic-free level has **no bus** (the dormancy split, D1/D2).
- **`[[wire]]` format** (`level_loader._parse_wires`, `level_lib.format_wire_lines`): a
  first-class level object, dotted `from = "id.signal"` / `to = "id.input" | "tag:name.input"`
  (Erik's call). Source-signal / target-input existence hard-errors; a dangling id warns +
  drops; a unit reference hard-errors (§3e). Tag targets pre-expand per member in ordinal
  order. Byte-stable round-trip (A2 client).
- **Slot 9e (split-gated):** the door structural sweep stays `if self._doors`; the logic
  block runs iff the bus exists. Order per tick — (a) sensors sample + `alive`/`is_open`
  emit, (b) node sweep (ordinal order, prev-read `pub` / next-write `stg`), (c) actuator
  input resolve, (d) pump + door sweeps, (e) node-signal swap. **Latency: 1 tick per node
  hop; sensor → world effect = 2 ticks** (golden-locked, B6).
- **Node set** (`entities/nodes.py` schema + `logic_nodes.py` runtime): `decider` (compare
  + `require_alive`), `gate_and/or/not`, `filter` (integer EMA, `k` snapped once at load by
  EXACT `Fraction`/`bit_length` — no float in the synced path). Input aggregation modes
  `HELD(OR)`/`EDGE`/`AND`/`SINGLE`.
- **Sensors** (`entities/sensors.py` + `sensor_system.py`, read through the §5a
  `EntityFieldAccessor`): `pressure`/`smoke`/`water_depth`/`o2` (air-family) +
  `temperature`/`fire` (solid-body family) + `clock` + `sensor_motion` (LOS, faction,
  corner anchor). `o2` = `gas[O2]` density (no `p_O2` field exists). Dead sensor → 0
  (fail-deadly). **§5a accessor is stubbed to the host mirror** — the `(n_sites ×
  n_channels)` int32 gather-buffer interface is frozen (incl. a `solid` channel), but the
  resident kernel is a **deferred follow-up** (was gated on S8c; now buildable — TODO).
- **Pump** (`pump_system.py` + a new integer, per-slice, zero-clamped `GameMap.inject_gas_n`
  / `extract_gas_n` — `FieldEdit`'s float+RNG gas path was unusable): N-feed, `ΔN` quantized
  once at load with a `ΔN < 2·band` assert; inject at the standard mix, extract ∝
  composition clamped ≥0 (exact N-conservation).
- **Automatic `airlock_controller`** (`entities/actuators.py` + runtime): the acceptance
  showcase — IDLE→CLOSING→EQUALIZE→OPEN_FAR→RESEAL→REPRESSURIZE + FAULT (breach-abort on a
  door reading `alive==0`, D12). **Bidirectional (Erik's Option 2):** both legs gate on a
  chamber **pressure sensor + two deciders** (`at_far`/`at_near`), the pump runs open-loop —
  so REPRESSURIZE genuinely refills. Fixture `levels/airlock_demo` (5-tall corridor so a
  3×3 marine has room — the fixed HUMAN-TEST bug). Manual open buttons = future "airlock v2",
  gated on the control-scheme decision (`button`/terminal still inert).
- **Digest/dormancy:** `__signals__` (SIGNAL_SECT_V1) populated from the bus; node/pump/
  airlock runtime state joins `__entity__` via `runtime_digest_rows` (the door's rows
  UNCHANGED — `is_open` rides `__signals__` only). A logic-free level is **byte-identical**
  to Arc A; **zero existing goldens re-baselined**. Cross-machine logic golden (B6): a
  sensor→filter→decider→door loop through the atmosphere solver, trajectory-hashed.

## 9. Forward pointers (what builds on this)

- **Resident sensor-gather kernel:** the §5a `(n_sites × n_channels)` int32 gather kernel on
  the GPU-resident path (interface frozen in §8; was deferred behind S8c, now buildable).
- **Arc C — editor UX: delivered.** C0–C9 merged 2026-07-22 (`54cd6cd`) — registry-driven
  palette/inspector, transaction-log undo, the DOOR/wire/zone tools, play-from-editor,
  icons, plus riders (baker writeback onto level_lib, a `MAT_DOOR_CLOSED`-outside-a-span
  validator warning). See Implementation status below; build record archived
  `docs/archive/arc_c_impl_2026-07-22.md`. Legacy-level migration stayed a separate,
  non-Arc-C call (Erik: keep `unhcr_vessel` as the legacy fixture in `levels/`, retire
  `unhcr_vessel_2` to `prototypes/`, migrate playground/planetside_demo).
- **Stack-2 — units convergence:** one id space, unit signals (`hp`, `alive`,
  `faction`), `SYNCED_UNIT_FIELDS` + entity digests merge as a planned single
  re-baseline.

---

## Implementation status (honest, updated as arcs land)

- A1–A9 all merged to main 2026-07-19 (tranche 1 auto-merged on green per ruling 3;
  A6/A7 human-tested and blessed by Erik). Suite green throughout; zero pre-existing
  goldens regenerated (the A7 event's committed-artifact footprint was empty).
- **B1–B7 merged to main 2026-07-22** (Erik blessed at the airlock HUMAN-TEST). §8 as-built;
  design record `docs/archive/arc_b_impl_2026-07-21.md`. Full suite green, zero goldens
  re-baselined, dormancy byte-identical throughout. Doors now driven by real wires (the dev
  O-key latch survives on unwired doors); `button`/terminal classes remain
  format-reserved and inert pending the control-scheme decision (interaction/cost-policy
  split — never bake AP/phase assumptions into entity code).
- **Arc C — editor UX (C0–C9) merged to main 2026-07-22** (`54cd6cd`). Registry-driven
  palette/inspector (C1); transaction-log undo replacing the four per-domain rings
  (C2, design-gated — `docs/archive/arc_c_c2_undo_design_2026-07-22.md`); DOOR
  span-placement + sensor placement + generic entity place-one (C3); multi-select + tags +
  clump copy/paste (C4); wand fill + zone/air/vacuum paint with hull-leak/zone-binding
  validators + a level-properties pane (C5); two-click wire tool + LOGIC overlay + tag
  badges (C6); play-from-editor (F5, `_editor_scratch/` loader form) (C7); an SVG->PNG
  icons pipeline for the palette (C8); baker `[art]`/`[bake]` writeback ported onto
  `level_lib` + a `MAT_DOOR_CLOSED`-outside-a-span validator warning (C9). **SPAWN stays a
  bespoke, non-registry placement tool** — units are not entities (§0), so there is no
  parity path to port it onto `[[entity]]`; the deeper unit-class/spawn-location question
  is deferred to its own design pass. Full suite green throughout (1547 passed at C9's
  gate), zero goldens/digests touched at any patch. Build record archived
  `docs/archive/arc_c_impl_2026-07-22.md` (+ kickoff + the C2 undo design doc, same
  folder).
- Open dials & warts: `physics.py:104` blast-tuple wart (steel/glass/furniture excluded
  from blast wall damage — pre-existing; decide at physics close-out) · staged
  tile-per-tick door opening if the instant whole-span transient ever reads badly (no
  change requested at the A6 human-test blessing; option recorded in a5/a6 docs) ·
  `bake_demo`
  stays legacy until its art rebakes · `burst_threshold` re-tune dial (ledger).
