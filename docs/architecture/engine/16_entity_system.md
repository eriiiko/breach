# 16 — Entity System (foundation)

**Depends on:** 02 (state & ownership — GameMap is the field truth), 03 (material system —
`MAT_DOOR_CLOSED` and the table-projection caches), 04 (atmosphere & pressure — the §2.2
occupancy-transition rule the seal/unseal primitives implement), 13 (FieldEdit — whose
structural carve-out door flips live in), 14 (determinism — digest/attestation the entity
sections join), 15 (level authoring — the format `[[entity]]` extends).

**Status:** Arc A foundation ✅ (A1–A9 merged 2026-07-19, Erik blessed) · signals/sensors/logic
📝 (Arc B, gated on the S8a spec rewrite) · editor UX 📝 (Arc C).

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
NOT entities in v1** — they keep `[[spawn]]`; the convergence contract (one shared id
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

## 8. Forward pointers (what builds on this)

- **Physics close-out first** (priority ledger #1): the S8a residency spec rewrite MUST
  include (a) the **sensor-gather contract** — sensor sample sites are static per level;
  one compact gather-kernel D2H per tick, never per-tick full-field streaming (design doc
  §7) — and (b) the **structural dirty-set rider** — once fields are GPU-resident,
  `destroy_wall`/`seal_tiles`/`unseal_tiles` must push their touched-tile set
  (`on_tile_changed` caches + gas ×7, atmosphere, wave_p, wind, flow, ripple, is_vacuum)
  to the device before the next kernel reads (a5 doc §9).
- **Arc B — logic:** SignalBus + the rest of slot 9e (sensors sample → logic sweep in id
  order → inputs resolve → door sweep), the two-tick latency contract, the v1 sensor
  catalog, the pump N-feed, the automatic airlock_controller, the cross-machine logic
  golden. `__signals__` is already defined (empty); `is_open`/`open`/`close` are already
  declared on the door class.
- **Arc C — editor UX:** registry-driven palette/inspector, transaction-log undo, the
  DOOR/wire/zone tools, play-from-editor, icons; plus riders — baker writeback onto
  level_lib, `MAT_DOOR_CLOSED`-outside-a-span validator warning, legacy-level migration
  (Erik's call per level).
- **Stack-2 — units convergence:** one id space, unit signals (`hp`, `alive`,
  `faction`), `SYNCED_UNIT_FIELDS` + entity digests merge as a planned single
  re-baseline.

---

## Implementation status (honest, updated as arcs land)

- A1–A9 all merged to main 2026-07-19 (tranche 1 auto-merged on green per ruling 3;
  A6/A7 human-tested and blessed by Erik). Suite green throughout; zero pre-existing
  goldens regenerated (the A7 event's committed-artifact footprint was empty).
- Doors have no drivers but the dev latch until Arc B; `button`/terminal classes are
  format-reserved and inert pending the control-scheme decision (interaction/cost-policy
  split — never bake AP/phase assumptions into entity code).
- Open dials & warts: `physics.py:104` blast-tuple wart (steel/glass/furniture excluded
  from blast wall damage — pre-existing; decide at physics close-out) · staged
  tile-per-tick door opening if the instant whole-span transient ever reads badly (no
  change requested at the A6 human-test blessing; option recorded in a5/a6 docs) ·
  `bake_demo`
  stays legacy until its art rebakes · `burst_threshold` re-tune dial (ledger).
