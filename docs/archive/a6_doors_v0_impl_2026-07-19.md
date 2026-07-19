# A6 impl design — Doors v0 (v2, critique folded, 2026-07-19)

Arc A patch A6 (plan: `arc_a_patch_plan_2026-07-18.md`, A6 row). Design-gated:
v1 went through independent adversarial critique and **survived with fixes**
— 1 blocker (B1, the blast-gate tuple), 5 should-fixes (S1–S5), 10 notes
(N1–N10), all folded below. The `MAT_DOOR_CLOSED` fork (§1) is ACCEPTED with
B1. Gate: digest green + **HUMAN-TEST (Erik)**; lands in tranche 2 with A7
(ruling 3 — no auto-merge).

v2 fold summary: B1 blast tuple + §13.13 regression (§1); S1 `--res`
base-resolution recovery (§3); S2 boundary-tick lag ACCEPTED + timing test
(§8, §13.14); S3 event contract restated slots-9/9b/9e-only (§8); S4 one
pinned hp_i↔tile order (§7); S5 = Erik's ruling 5 (§10, §16); N1 sweep-order
claim corrected (§5.2); N2 zombie citation fixed (§9); N3 observables-only
burst test (§13.12); N4 render scope honest-minimal (§14); N5 editor-palette
gap stated (§1, §15.13); N6 recorder-byte-identity scoped (§13.9); N7 vacuum
cycle test + gap line (§13.15, §15.14); N8 = the three added tests; N9
`mouse_to_tile()` (x,y) flip pinned (§10); N10 `Fraction(str(length_m))` +
explicit `length_m > 0` check (§3). One v2-discovered fix beyond the
critique: the **O key is already bound** — `renderer/game_renderer.py:871`
toggles the water-optics pass on KEY_O. Ruling 5 names the O-key for doors,
so the water-optics toggle moves to **V** (render-layer dev toggle; §10).

Scope: the `door` entity class; load-time tile derivation; the slot-9e
structural sweep (door half ONLY — no SignalBus, no sensors, no logic until
Arc B); occupancy/water close rules with retry; door HP as entity runtime
state; external-destruction reconciliation; path-hold; the debug toggle
hotkey (ruling 1); digest rows via the A4 runtime-row mechanism. Binding
inputs: entity doc §7/§3d/§5 (`entity_system_design_2026-07-18.md`), editor
doc §4/§6 (`level_editor_v3_design_2026-07-18.md`), the A5 primitives + §11
riders (`a5_evacuation_impl_2026-07-18.md`), rulings 1 and 4
(`arc_a_patch_plan_2026-07-18.md:13-38`).

---

## 0. Code-reality map (the anchors everything below builds on)

- **`step()` slots** (`src/simulation/simulation.py:635-858`): slot 9
  burn-through destroys walls at :740-748 (emits `DoorDestroyedEvent` when
  `mat == MAT_DOOR`, :745-746); slot 9b burst scan at :750-764 (same event
  split, :761-762); 9c/9c2/9c3 exchange couplings :766-813; 9d ignition
  :815-835; **recorder snapshot :837-841**. Slot 9e inserts between :835
  and :837.
- **A5 primitives as landed** (`src/simulation/gamemap.py`):
  `can_seal_tiles` :1185, `seal_tiles` :1202, `unseal_tiles` :1292,
  `_normalize_span` :1074-1084, `_seal_receivers` :1086-1101,
  `_seal_blockers` :1103+. `destroy_wall` :1006-1056 (minting; untouched
  canon). `on_tile_changed` :548+ re-quantizes `wall_hp` from the material
  table on every material change (the S3 heal gap A6 must close).
- **`stamp_units`** :626-643 (C++/Python dispatch); the unit-footprint
  contract is `u.occupied_tiles()` + the `u.alive` filter (:673-680).
  `obstacles` rebuilds at slot 6 (`simulation.py:721-722`) — the step-6
  restamp that carries 9e flips to the solvers next tick.
- **Passability is table-driven**: `is_passable` reads
  `materials.mobility[material] > 0` (`gamemap.py:824-835`);
  `is_passable_block` :837-849; `footprint_mobility` :851-866. Player-path
  A* consumes `is_passable_block` via the `is_blocked` closure
  (`simulation.py:484-485` inside `_compute_player_paths` :459-538); paths
  are per-tick float positions, possibly fractional mid-step (:521-525).
  Path consumption + the can_move-suppression pattern:
  `_update_player_movement` (`simulation.py:936-959`; the hold-and-burn
  branch is :951-953).
- **`u.occupied_tiles()`** returns `(tx, ty)` **(col, row)** tuples anchored
  at `(int(u.x), int(u.y))` (`src/simulation/unit.py:293-303`, tile_x/y
  :372-379). Span tiles everywhere in gamemap are `(fy, fx)` (row, col) —
  the conversion is pinned wherever the two meet (§6.3).
- **THE surprise (load-bearing)**: `MAT_DOOR` has **`mobility = 1000`** —
  fully passable to movement (`config.toml:646`; the "air, open door"
  comment `src/simulation/materials.py:54-58`). Today's painted door is the
  hybrid the entity doc §7 describes: walkable, flow-solid. Entity doors'
  CLOSED = fully solid therefore CANNOT be expressed by stamping `MAT_DOOR`
  — see §1, the one real design fork in this patch.
- **Entity machinery (A1-A4)**: `Entity` ABC + `runtime_digest_rows` hook
  (`src/simulation/entities/schema.py:139-176`); registration :354-372; the
  one serializer consumes runtime rows per class with zero mechanism surgery
  (`src/simulation/entities/serialize.py:151-158`), duck-typed on
  `ordinal/id/class_name/fields` + optional `alive` (:121-132, :148).
  `light` exemplar: `src/simulation/entities/light.py`.
  `EntityInstance` (parsed, fields as-authored, length_m NOT quantized):
  `level_loader.py:223-242`; `_parse_entities` :263-365.
- **Load order today**: `GameMap.__init__` fills `material`/`is_vacuum`
  from the tilemap at `gamemap.py:352-356`, then `_update_caches()` at :358
  seeds atmosphere (:461-463) and bulk gas (:476-479) from the solid mask;
  the water seed follows at :376-390 masked to `~solid & ~is_vacuum`.
- **Capture paths**: `get_state` builds `entity_carrier(self.level.entities)`
  (`simulation.py:555-563`); the recorder snapshot passes
  `entities=self.level.entities` (:839-841) into `record()`
  (`src/simulation/recorder.py:107-152`, entity bytes at :148-152); the
  digest folds the carrier presence-gated (`tests/field_digest.py:108-137`).
  Both call sites must switch to the sim's runtime entity list (§6.1).
- **Sim lifecycle**: `_reset_internal` (`simulation.py:217-283`) builds a
  fresh `GameMap(self.level)` at :219 on every construct/reset — door
  runtime state rebuilds there from authored initial state.
- **Debug-key seam**: `src/input_handler.py` — I ignite / J gas / U pour
  water on the cursor tile (:30-33, :113, :127, :134-135), each via
  `renderer.mouse_to_tile()` (`renderer/game_renderer.py:923`) and each
  **writing synced fields directly** (`_debug_ignite` writes `gmap.fire` and
  even `gmap.flammable`, :228-246). The door toggle is the same species.
- **`find_burst_walls`** is table-driven per material
  (`gamemap.py:921-1004`, threshold read :964-970); `[materials.door]`
  `burst_threshold = 2.0` (`config.toml:655`).
- **Blast wall damage is NOT table-driven (B1)**: `apply_explosion` gates
  its structural HP decrement on the hardcoded tuple
  `(MAT_HULL, MAT_WOOD, MAT_DOOR)` (`src/simulation/physics.py:104`) —
  without a fix, `MAT_DOOR_CLOSED` tiles are blast-proof and the breach
  charge no-ops on a closed entity door (§1).
- **Zombie door-awareness is the per-step re-check (N2)**: before each
  committed zombie step, `update_zombies_tick` re-checks
  `is_passable_block` on the next path tile and drops the stale path when
  blocked (`src/simulation/ai_zombie.py:181-192`) — a door closing in a
  zombie's face stops it the same tick's step, independent of the repath
  cadence.
- **`mouse_to_tile()` returns `(x, y)` (col, row)**
  (`renderer/game_renderer.py:923-929`) — every debug key unpacks
  `fx, fy = tile` (`input_handler.py:238`, `_debug_ignite`'s flip). The O
  key must apply the same flip before calling `door_at(fy, fx)` (N9, §10).
- **KEY_O is TAKEN** (v2 finding): `renderer/game_renderer.py:871` binds O
  to the water-optics pass toggle (main.py's help text agrees). §10 moves
  that render toggle to V so ruling 5's O-key is clean.
- `_upscale_level` (`main.py:122-196`) scales tilemap/spawns/lights/water/
  zones/air, and **divides `tile_size_m` by the factor BEFORE `GameMap`
  ever sees the level** (`main.py:147`) — the S1 trap §3 must survive. It
  does not yet touch `[[entity]]` data.

---

## 1. The design fork: CLOSED doors get their own material, `MAT_DOOR_CLOSED`

**Problem.** A closed entity door must be *fully* solid — flow AND movement
(entity doc §7 :247-252). Flow-solid comes free (`MAT_DOOR` permeability 0),
but movement-solid does not: walkability is the derived view
`mobility > 0` over the material table (`gamemap.py:827-835`), and
`MAT_DOOR.mobility = 1000` (`config.toml:646`). One material id cannot be
walkable for legacy painted doors and impassable for entity doors — and
ruling 2 pins that shipped levels keep legacy painted-door behavior until
Erik chooses otherwise. Flipping the table column would change
unhcr_vessel's gameplay from under him.

**Decision: add material id 7, `door_closed`** (`MAT_DOOR_CLOSED = 7`,
config key `[materials.door_closed]`), the entity door's CLOSED stamp:

- Column values: copy `[materials.door]` (`config.toml:639-655`) except
  **`mobility = 0`**. Same `hp = 40`, `flammable = false`,
  `burst_threshold = 2.0`, attenuations, conductivity, thermal_mass,
  ignition_temp, wave params. Ids stay contiguous
  (`materials.py:29-47`; `MaterialTable` validates contiguity, :38).
- Open door tiles are plain `MAT_AIR`. `MAT_DOOR` itself is untouched —
  it remains the *legacy painted* door material with today's hybrid
  semantics, until A7 migrates a level.

**Why this beats the alternatives:**

1. *Flip `MAT_DOOR.mobility` to 0*: breaks ruling 2 (legacy levels change
   behavior with no migration event) — rejected.
2. *A dynamic `door_blocked` mask composed into passability*: requires
   touching every passability consumer (`is_passable`, `is_passable_block`,
   `footprint_mobility`, the A* closure `simulation.py:484-485`, zombie AI)
   and breaks the project's own cache invariant — "every cache is a
   projection of the material-property table … no hardcoded material lists"
   (`gamemap.py:395-400`). Rejected.
3. The chosen material makes closed doors block movement, LOS, flow, burst,
   burn, and pathing through the exact seams walls already use, with ZERO
   query-code changes. The only new movement code in A6 is the path-hold
   check (§9) — because precomputed WEGO paths are the one consumer that
   does not re-query the table. Zombie A* re-paths per tick and player-order
   A* re-plans per order; both see the closed door automatically.

**Consequences, all pinned:**

- `seal_tiles(span, MAT_DOOR_CLOSED)` is legal — the primitive requires
  permeability ≤ 0 (`a5_evacuation_impl_2026-07-18.md` §2), which holds.
  The A5 doc's line "For doors v0 this will be `MAT_DOOR`" (§2) is
  superseded by this decision — A5 doc is append-only; this doc is the
  amendment of record.
- The event split in slots 9/9b keys on `mat == MAT_DOOR`
  (`simulation.py:745, :761`) — both extend to
  `mat in (MAT_DOOR, MAT_DOOR_CLOSED)` so an entity door tile destroyed by
  burst/burn-through still emits `DoorDestroyedEvent`
  (`src/simulation/events.py:109`).
- **B1 (BLOCKER, folded): the blast gate joins the tuple.**
  `apply_explosion`'s structural wall damage is gated on the hardcoded
  tuple `(MAT_HULL, MAT_WOOD, MAT_DOOR)` (`physics.py:104`) — NOT
  table-driven like burst/burn-through. `MAT_DOOR_CLOSED` is added to that
  tuple, or closed entity doors are blast-proof and the breach charge —
  the one door-opening tool the HUMAN-TEST plan leans on — no-ops.
  Regression test §13.13: a blast destroys a closed entity door.
  **Flagged for arc close, do NOT change now:** steel/glass/furniture
  being excluded from blast wall damage is a pre-existing wart of that
  tuple; widening it is a feel-adjacent behavior change outside A6's
  scope.
- `find_burst_walls` needs nothing: threshold comes from the table
  (`gamemap.py:964-970`). A closed entity door across > 2.0 atm bursts next
  tick exactly like a painted door (the A5 §8/S2 row).
- CSV vocabulary: v2 codes are read literally as material ids
  (`level_loader.py:726-734`), so `7` in a CSV "works"; but
  `MAT_DOOR_CLOSED` outside a door-entity span is an authoring accident
  (an unopenable door-looking wall). v1: allowed-but-meaningless under a
  span (the entity overrides it, §5.2); a validator warning elsewhere is
  Arc C's business (accepted gap §15.9).
- **`door_closed` auto-appears in every editor palette (N5, stated):**
  the map/level editors and the greybox tileset derive their palettes from
  `MATERIAL_NAMES` at call time (`tools/level_edit_common.py:66-73`,
  `tools/make_tileset.py:344`, with the no-recipe fallback color), so the
  new id shows up as a paintable material immediately. Benign, accepted
  gap (§15.13): painting it by hand is exactly the §15.9
  legal-but-inert pseudo-door; the DOOR tool teaching the editors about
  door *entities* is Arc C.
- Renderer: honest-minimal scope per N4 — the F6 debug HUD's material name
  comes from `MATERIAL_NAMES` (replacing the stale hardcoded 3-entry dict
  at `renderer/game_renderer.py:960`), and door spans get a simple
  per-tile fill/outline drawn in `compose_world` (§14). No new overlay
  system. Render-layer, exempt from determinism.
- Editor-doc §6's "its `MAT_DOOR` tiles are written to the grid" was written
  before this split; at canon fold it reads "the door material stamp —
  `MAT_DOOR_CLOSED` as of A6" (errata list, §16). The editor itself is
  Arc C; A6 owes it nothing.

---

## 2. Door entity class schema (`src/simulation/entities/door.py`)

Registered class `door`, following the `light` exemplar pattern
(`src/simulation/entities/light.py:21-48`). Stdlib-only module — the span
quantization uses `fractions.Fraction`, so the import-light CI test
(entity doc §3b) stays green.

### 2a. Declared FIELDS (authoring surface)

| field | kind | default | constraints | synced-kind hashed? |
|---|---|---|---|---|
| `x` | `KIND_INT` | *required* (None) | ≥ 0 | yes (int) |
| `y` | `KIND_INT` | *required* (None) | ≥ 0 | yes (int) |
| `orientation` | `KIND_ENUM` | `"h"` | choices `("h", "v")` | yes (choice index) |
| `length_m` | `KIND_LENGTH_M` | `1.0` | > 0 (explicit check, N10) | **no** (authoring-bound) |
| `initial_state` | `KIND_ENUM` | `"closed"` | choices `("closed", "open")` | yes (choice index) |

- **`x`, `y` are the anchor TILE (base resolution), integer** — not
  meters-float like lights. Justification: the editor's DOOR tool
  wall-run-snaps placement (editor doc §6 :98-103), so the anchor is
  tile-quantized by construction; storing it as the tile kills any
  float→tile conversion ambiguity, and `--res` replication scales tile ints
  by the integer factor exactly (editor doc §4 :62-67, matching
  `_upscale_level`'s spawn/footprint semantics `main.py:148-153`).
  Convention: `(x, y)` = (col, row) of the span's first tile (leftmost for
  `"h"`, topmost for `"v"`) — matches the authoring-facing x/y convention
  of `[[spawn]]`/`[[light]]`.
- **`length_m` + `orientation` + anchor, NOT an explicit tile list.**
  The DOOR tool is "snap, default 1.0 m, drag to resize" — a straight run
  parameterized by length is the authoring model; a free tile list would
  permit non-contiguous/bent spans the physics story never defined, and
  would bypass the meters-first quantization rule (editor doc §4). The
  entity is authoritative; tiles are always derived (§3).
- `length_m` is deliberately NOT hashed (the A4 critique-blocker-1
  partition, `serialize.py:22-26`): its synced consequence is the material
  grid, already hashed. `x/y/orientation/initial_state` ARE hashed as
  declared synced kinds — they are integer/enum authoring constants.
- **`length_m > 0` is an explicit derivation-time check (N10):** schema
  `minimum` bounds are inclusive (`schema.py:257`), so a declared minimum
  cannot express "strictly positive". The schema declares `minimum = 0.0`
  (rejects negatives at load) and the §3 quantizer hard-errors on
  `length_m <= 0` with the entity id in the message.
- Per the A3 loader (`level_loader.py:318-335`) all five validate through
  `field_value_error` (`schema.py:191-241`); `x/y` required-ness comes from
  `default=None` exactly like the instance `id`.

### 2b. Signals / inputs — format-reserved, inert (v1)

Declared so Arc B adds drivers without touching levels (entity doc §3d):

- `SIGNALS = (Signal("is_open"),)` — the §6 catalog's `door.is_open`;
  nothing emits it until the SignalBus exists.
- `INPUTS = (InputDecl("open", INPUT_HELD), InputDecl("close", INPUT_HELD))`
  with `INPUT_PRIORITY = (("close", "open"),)` — close beats open, the safe
  state (entity doc §4). Inert in v1; the debug latch (§7) is the only
  driver and deliberately mirrors a while-held `open` input.
- `INTERACTIONS = ()` in v1 (no unit-initiated path at all, §3d).
- `INTANGIBLE = False` (physical by default, entity doc §5).

### 2c. Runtime state (NOT schema fields — the A4 runtime-row block)

Runtime state lives on the sim-side runtime object (§6.1) and enters the
digest via `door.runtime_digest_rows(e)` (`schema.py:165-176` hook;
consumed at `serialize.py:151-158`):

| row name | value | meaning |
|---|---|---|
| `state` | int 0/1/2 | 0 = CLOSED, 1 = OPEN, 2 = DESTROYED |
| `want_open` | int 0/1 | the desired-state latch (§6.2) — synced |
| `hp_0` … `hp_{k-1}` | int Q16.16 | per-runtime-span-tile door HP, row-major span order (§7) |

Row names satisfy the `[A-Za-z0-9_]+` token charset (`serialize.py:84`,
:152-155). The row COUNT is span-length-dependent but constant for an
instance's lifetime (rows stay present, zeroed, after destruction — §8).
The free `alive` row is the mechanism's own (`serialize.py:148-150`); the
runtime object's `alive` attr drives it. `runtime_digest_rows` reads the
runtime attrs plainly; called on a bare `EntityInstance` (no runtime attrs)
it raises `AttributeError` — loud, a bug, never a fallback: digests are
only captured from constructed sims (§6.1).

ENTITY_SECT stays **V1**: rows are per-class by design; zero mechanism
surgery (A4 impl note critique 6; `serialize.py:28-30`).

---

## 3. Tile derivation — the canonical quantization, once

One pure function in `entities/door.py` (stdlib `Fraction` only), THE span
rule for loader, sweep, editor (Arc C), and migration tool (A7):

- `length_m` ingresses as **`Fraction(str(length_m))`** (N10, pinned): the
  decimal the author typed, never the binary float's full expansion —
  `Fraction(str(0.5)) = 1/2` exactly, and the round-half-up below then has
  no float in it at all. `length_m <= 0` hard-errors here (the §2b
  explicit check).
- `n_base = floor(length_m * tiles_per_m + 1/2)` in exact `Fraction`
  arithmetic — round-half-up, never banker's `round` (editor doc §4
  :59-61); clamped to ≥ 1.
- Base span = `n_base` tiles from the anchor along `orientation`:
  `"h"` → `(y, x), (y, x+1), …`; `"v"` → `(y, x), (y+1, x), …`
  (span tiles in gamemap `(fy, fx)` order).
- `tiles_per_m` comes from the level exactly: the shipped
  `tile_size_m = 0.333` maps to **exactly 3** (the editor-doc §4 migration
  rule :53-58, matched as `Fraction(str(tile_size_m)) == 333/1000`); any
  other value converts through
  `Fraction(1) / Fraction(str(tile_size_m))` and must be integral or the
  level hard-errors for door purposes (a non-integer tiles-per-meter level
  has no defined door quantization; none exists in the tree). The
  full integer-`tiles_per_m` format migration remains Arc C's; A6 ships
  only this read-side helper.
- Default `length_m = 1.0` at 3 tiles/m → a **3-tile** door — the 1 m
  doorway a 1 m marine footprint fits. A 1-tile door is authored
  `length_m = 0.333` (quantizes to 1).

**`--res N` replication (quantize once, then replicate — editor doc §4
:62-67):** the runtime tile set is the base span REPLICATED, exactly as
`np.repeat` replicates the painted grid (`main.py:145-146`): each base tile
`(fy, fx)` becomes the `N×N` block at `(N·fy … N·fy+N-1, N·fx … N·fx+N-1)`.
A base 1×3 door at `--res 2` is a 2×6 tile rectangle — identical to what
the same door painted in the CSV would become. Never re-derived from meters
at the scaled resolution.

**S1 (folded): base-resolution recovery is pinned, because `--res` breaks
the naive read.** `_upscale_level` divides `tile_size_m` by the factor
BEFORE `GameMap` exists (`main.py:147`), so a door consumer reading the
live `tile_size_m` at `--res 2` would see `0.1665` —
`Fraction(str(0.1665))` gives a non-integral `tiles_per_m` and the level
hard-errors. Chosen fix (of the two the critique offered — recompute
`tile_size_m * res_factor`, or carry the base — the CARRY, because a
float recompute is only IEEE-exact for power-of-two factors):
`LevelData` gains TWO defaulted fields —

- `res_factor: int = 1` — the accumulated integer replication factor;
- `tile_size_m_base: float | None = None` — the PRE-scale tile size;
  `None` means "unscaled: use `tile_size_m`".

`_upscale_level` sets `tile_size_m_base = tile_size_m` (the pre-division
value, captured BEFORE the mutation) and multiplies `res_factor` by the
factor, then divides `tile_size_m` as today. Every door consumer
quantizes against `tile_size_m_base or tile_size_m` and replicates the
base span by `res_factor`. Authored entity fields are NOT mutated (they
stay the authored record). The whole replicated rectangle is ONE
`seal_tiles` span (A5 §3.3: a multi-tile door closes as one call).

---

## 4. Load order — entity tile state BEFORE field seeding

### 4.1 Insertion point

`GameMap.__init__`, between the tilemap fill (`gamemap.py:354-356`) and
`_update_caches()` (:358):

1. For each `door` instance in `level_data.entities`, **ordinal order**
   (§3a single ordering rule): derive the runtime span (§3), validate
   (§4.2), then stamp `self.material[span]` = `MAT_DOOR_CLOSED` if
   `initial_state == "closed"`, else `MAT_AIR`.
2. Then `_update_caches()` runs as today and seeds atmosphere/gas against
   the POST-stamp solidity (:461-479): closed spans seed 0 (solid), open
   spans seed ambient (air).

**No `seal_tiles` at load** — there is no gas to evacuate yet; the stamp
precedes all field seeding, so conservation at t=0 is trivially exact.
This IS the entity-doc §7 load-order rule (:257-259). The water seed
(:376-390) runs after and is masked to `~solid`, so authored water on a
closed span is dropped-with-warning by the existing backstop — no new rule.

### 4.2 Load-time validation (hard `ValueError`s, path-named)

- Every span tile in bounds.
- Span tiles' CSV material ∈ {`MAT_AIR`, `MAT_DOOR`, `MAT_DOOR_CLOSED`} —
  the editor bakes a door stamp into the grid (editor doc §6 :100-103) and
  hand-authored files may leave air; a door through hull/steel is an
  authoring bug.
- No span tile `is_vacuum` (a door on the hull ring is an authoring error —
  the A5 §5.2/§10.9 stance).
- No two door spans overlap (runtime tile sets disjoint) — makes sweep
  order effects between doors impossible at the tile level and the hotkey
  hit test unique.

### 4.3 The authored-open ≡ authored-air round-trip (the A6 load gate)

Two levels identical except: (a) has a door entity `initial_state="open"`,
(b) has no entity and plain air tiles. Test: `GameMap(a)` and `GameMap(b)`
are **bit-identical across every synced array** (`material`, `is_vacuum`,
`gas` all slices, `atmosphere`, `temperature`, `water_depth`, `wall_hp`,
all structural caches). The digest of course differs — (a) hashes entity
records — the identity claim is FIELD state (entity doc §7 :257-259 says
"loads bit-identical"; this is its precise reading, pinned here). A second
variant: authored-closed ≡ the same span painted `MAT_DOOR_CLOSED` in the
CSV with no entity — field-identical too (same stamp, same seeding).

---

## 5. Slot 9e (door half) — the structural sweep

### 5.1 Placement and shape

Insert in `step()` after the 9d ignition block (`simulation.py:815-835`),
BEFORE the recorder snapshot (:837-841) — the entity-doc §7 slot contract
(:227-231): recorder/digest see entity state consistent with the flips
that happened, same tick. Comment it `# 9e. Entities — structural door
sweep (v1: doors only; sensors/logic arrive with Arc B's SignalBus).`

Runs whenever the sim has doors (`if self._doors:` — one attribute check;
entity-free and door-free levels do zero work: dormancy is structural).
Not gated on `physics_runner` — flips are pure gamemap edits and tests run
sim-without-physics.

### 5.2 The sweep, per door in ordinal order

The entity-doc sentence "the sweep collects flip intents and applies them
in entity-id order" is realized as ONE ordinal-order pass with inline
application: the intents ARE the per-door latches (read-only collection),
spans are disjoint (§4.2), and application order is the iteration order.
**N1 (folded): no "equivalent to collect-then-apply" claim** — flips are
applied sequentially in ordinal order; overlapping spans are
load-rejected; adjacent spans still interact *through gas* (an earlier
door's evacuation changes what a later door's seal/unseal sees) — the
behavior is simply deterministic BY ordinal order, which is all the
entity doc requires. Per door:

1. **Reconcile external destruction first** (§8). If it fires, the door is
   dead; skip step 2 forever.
2. **Apply the latch** (§6.2):
   - `state == CLOSED and want_open` → **open**: fold `wall_hp` span values
     into the entity `hp_*` rows (§7), then `gmap.unseal_tiles(span)`,
     `state = OPEN`. Opening is unconditional — the primitive's only
     refusals on solid tiles are caller bugs (A5 §7 validation), and per
     rider 3 any raise from a primitive here is a bug, never "door stays
     shut".
   - `state == OPEN and not want_open` → **close attempt**:
     `occupancy_clear(span) and gmap.can_seal_tiles(span)` (rider 3's exact
     composition). On pass: `gmap.seal_tiles(span, MAT_DOOR_CLOSED)`, then
     restamp `gmap.wall_hp[t] = hp_i` for every span tile (§7 — undoing
     `on_tile_changed`'s table re-quantize inside the seal), `state =
     CLOSED`. On fail: **do nothing** — the latch is NOT consumed (§6.2).
   - Otherwise (state matches latch): no-op.

Effects reach the solvers next tick via the step-6 restamp
(`simulation.py:721-722`; A5 §5.1) — the one-tick-delay story. Within-tick,
only the recorder and heat-clear run after 9e; both are indifferent. The
NEXT tick's 9b burst scan may pop a door just closed across a
super-threshold differential (`config.toml:655`; A5 §8/S2) — deliberate
relief-valve physics; §8 handles the entity side.

`occupancy_clear(span)`: for any living unit (`u.alive`, the
`stamp_units` filter `gamemap.py:673-675`), no `(tx, ty)` of
`u.occupied_tiles()` (`unit.py:293-303`) satisfies `(ty, tx) ∈ span_set` —
**note the (col,row)→(row,col) flip**, pinned here because it is the one
trap in the check. Dead units never block (a corpse can be sealed over —
accepted gap §15.2). Water/pocket/overflow refusals ride inside
`can_seal_tiles` (A5 §2, §11.3) — the primitive's own span-only water
guard is the "same blocked semantics as units" the entity doc requires.

---

## 6. Runtime object, the latch, and the two capture call sites

### 6.1 `DoorRuntime` (sim-side, `src/simulation/door_system.py`, new)

The entities package stays import-light, so runtime+sweep code lives
sim-side and imports `entities.door` for schema + span math. `DoorRuntime`
wraps the parsed `EntityInstance`, exposing the serializer duck-type
(`ordinal`, `id`, `class_name`, `fields` — delegated; plus `alive`) and
the runtime attrs: `state`, `want_open`, `hp` (list, §7), `span`
(runtime tile list, §3), all initialized from `initial_state` + the
material table.

Built in `_reset_internal` right after `GameMap` (`simulation.py:219`) —
so reset rebuilds doors fresh, and the shared `LevelData` is never
mutated with runtime state. The sim keeps `self.entities`: the level's
instances with door entries replaced by their `DoorRuntime` wrappers
(ordinal order preserved), and `self._doors`: the doors sublist.
**Both capture call sites switch to it**: `get_state`'s
`entity_carrier(self.level.entities)` (`simulation.py:563`) and the
recorder's `entities=self.level.entities` (:841) become `self.entities`.
With that one change, recorder dumps and `get_state` carry door rows
automatically — `record()` serializes whatever list it is handed through
THE one serializer (`recorder.py:148-152`), and the digest hashes the same
bytes (`tests/field_digest.py:121-137`). Verified by test §13.8.

Initial runtime values: `state` from `initial_state`; `want_open =
(initial_state == "open")` (latch agrees with state → first sweep is a
no-op); `hp_i = wall_fixed.quantize(materials.hp[MAT_DOOR_CLOSED])` per
runtime span tile (the same quantization `_update_caches` applies,
`gamemap.py:429-432`) — for closed doors this equals the freshly stamped
`wall_hp`, for open doors it is the full panel HP the first close will
stamp; `alive = True`.

### 6.2 Retry-latch semantics (pinned, v1)

With no SignalBus, the latch **is** the door's only driver: `want_open` is
a synced boolean on the runtime object, hashed as a runtime row (§2c). The
sweep works toward the latch every tick until achieved or re-toggled:

- A close blocked by occupancy or `can_seal_tiles` **does not consume or
  clear the latch** — the sweep simply fails this tick and retries next
  tick. This mirrors Arc B's while-held `close` input exactly (entity doc
  §7 occupancy rule :245-247: "the close input (while-held) naturally
  retries"), so the v1 behavior IS the v2 behavior with the latch standing
  in for the wire.
- Toggling back before the close lands (unit still in the doorway) simply
  leaves the door open: no queued intent, no edge memory — desired-state,
  not commands. Deterministic by construction: the latch is synced state,
  read at one pinned slot in ordinal order.
- Destroyed doors: latch is dead — the sweep skips dead doors; the toggle
  helper (§10) refuses to target them ("inputs go dead", entity doc §4).

Blocked-ness is not extra state: "close desired but blocked" is exactly
`state == OPEN and not want_open` after a sweep — no `pending` row needed.

---

## 7. Door HP — per-tile vector, fold on open, restamp on close (rider 2)

**Rule:** the entity carries **one Q16.16 HP value per runtime span tile**
(`hp_0 … hp_{k-1}`, row-major span order), NOT a scalar.

**S4 (folded): ONE pinned hp_i↔tile order, everywhere.** The door's
runtime span list IS the replicated tile set (§3) sorted row-major —
`sorted()` on `(fy, fx)` tuples — and `hp_i` indexes THAT list in every
consumer: the fold on open, the restamp on close, the zeroing on
destruction, and the `hp_*` digest rows. No consumer ever re-derives or
re-sorts its own tile order.

- **On close** (after `seal_tiles`): `gmap.wall_hp[t_i] = hp_i` for every
  span tile — overwriting the fresh table value `on_tile_changed` wrote
  inside the seal (`gamemap.py:548+`, the re-quantize the A5 S3 gap
  documents in its §5.1). No free heal by cycling — rider 2 closed.
- **On open** (before `unseal_tiles`): `hp_i = int(gmap.wall_hp[t_i])` —
  the panel remembers its damage, tile by tile.
- **While open**: HP is frozen; the retracted panel is not targetable in
  v1 (accepted gap §15.4).
- **While closed**: tiles take damage independently through the normal
  wall paths (bullet chew, fire, blast) writing `gmap.wall_hp` — the grid
  is authoritative while the door is a wall; the entity's rows are the
  ledger only across open periods. The rows therefore refresh at every
  fold (open) — between folds the digest's `hp_*` rows may lag the live
  `wall_hp` (which is itself hashed as a field), and that is fine: no
  state is lost, and the rows are exact at every boundary where they are
  load-bearing (the restamp).

**Why per-tile, not min/sum/mean** (the pinned decision the task requires):

- *min*: a single damaged tile smears its damage across the whole span on
  the next cycle — cycling becomes a door-weakening exploit (shoot one
  tile, cycle, whole door at min) and information is destroyed.
- *sum*: cannot be restamped — closing must invent a per-tile distribution,
  and any rule (even split, first-tile-heavy) either heals hot spots or
  concentrates damage that was never there. `wall_hp` is per-tile state;
  a scalar cannot round-trip it.
- *mean*: same restamp invention plus outright healing of the weakest tile
  — the exact exploit class rider 2 exists to kill.
- *per-tile vector*: exact round-trip, zero invention, kills both the heal
  and the smear. Cost: a span-length-dependent row count — which the A4
  row mechanism supports natively (`serialize.py:151-158`; rows are
  whatever the class emits). Burst/burn-through target single tiles
  (`simulation.py:740-764`), so per-tile fidelity is also what external
  destruction (§8) needs to be observable per tile.

---

## 8. External destruction — the whole-door rule (rider 1)

Slots 8/9/9b (and W2 bullet chew, any `destroy_wall` caller) can destroy a
CLOSED door's tile with the minting destruction path, outside the entity's
control (`simulation.py:740-764`; A5 §8 burst-after-seal row, §11.1). The
entity **observes the grid**, it does not consume events: at sweep step 1,
a door with `state == CLOSED` checks its span; any tile whose material is
no longer `MAT_DOOR_CLOSED` was externally destroyed (`destroy_wall` set it
`MAT_AIR`, `gamemap.py:1027-1028`; nothing else can touch a solid tile's
material — FieldEdits never write material, and overlapping door spans are
load-rejected §4.2). Observation beats event-consumption: it is immune to
event-ordering/citizenship questions.

**S2 (folded): the one-boundary-tick lag is ACCEPTED, and the v1 "zero
recorded desync ticks" claim was FALSE for slot 8.** Within-tick
destruction (slot 9 burn-through, slot 9b burst, the tick-0 start-P1
explosive volley at `simulation.py:665-670`) all run before 9e, so those
reconcile the same tick, before the recorder snapshot. But the
BETWEEN-PHASES and END-OF-ROUND door-explosive volleys run at the tick
BOTTOM (`simulation.py:868-886`) — after 9e AND after the recorder
snapshot — and destroy walls synchronously. A closed door destroyed there
is reconciled by 9e on the NEXT tick: exactly ONE recorded boundary tick
exists whose entity rows still say CLOSED over minted-air tiles. Accepted,
not fixed: the lag is deterministic (same tick, same order, every
machine — no cross-machine divergence, digests agree on the lagged state
itself), and re-running reconciliation after the boundary volleys would
mean a second sweep site with its own ordering contract for one cosmetic
tick. Pinned by the §13.14 timing test.

**S3 (folded): the event contract, restated.** `DoorDestroyedEvent` is
GUARANTEED only from slots 9, 9b, and 9e (each `destroy_wall` call at
those sites emits per tile, with the §1 material-check extension). Other
`destroy_wall` callers — `apply_explosion`'s blast damage, W2 bullet
chew — emit nothing for door tiles today and A6 does NOT touch that
combat code; a blast-destroyed door tile surfaces as the 9e
assembly-completion events next sweep. The event stays render/telemetry
only; no sim logic consumes it (the entity observes the grid). Arc B may
add an entity-level `door_destroyed` signal; not v1.

**Decision: partial destruction destroys the whole door.** On detecting ≥1
destroyed span tile:

1. `alive = False`, `state = DESTROYED`, all `hp_*` rows = 0. The latch is
   dead; the entity never re-seals (rider 1's exact demand — a destroyed
   door may not keep CLOSED state or later re-seal a minted-air breach).
2. Every REMAINING intact span tile (still `MAT_DOOR_CLOSED`) is destroyed
   via `self.gmap.destroy_wall(t)` in row-major span order, each appending
   `DoorDestroyedEvent(pos=t)` to `sim.tick_events` — matching the
   existing per-tile emission (`simulation.py:745-746, :761-762`).

**Why whole-door, not a stub:** a door is one panel and one mechanism — a
panel with a tile-sized hole neither seals nor operates; a surviving
"1-tile stub door" would be a new, never-designed object (what is its
length_m? its digest rows?) and the RL-visible physics of a
half-functional seal is exactly the kind of emergent weirdness the design
doc's determinism-first stance exists to avoid. **Why `destroy_wall` and
not `unseal_tiles` for the remainder:** the remaining tiles are being
*destroyed* (the assembly failed), not operated — and destruction MINTS,
by canon (A5 §1: `destroy_wall` untouched, the asymmetry deliberate and
bounded). The mint is bounded here too: a door dies once, the entity is
dead, nothing is agent-cyclable. Note: the entity-driven completion is NOT
subject to the 9b `burst_max_per_tick` cap (:757) — the cap protects
against mistuned thresholds nuking the ship; assembly-completion of an
already-dying door is not that.

**Event granularity:** once per destroyed door TILE — slots 9/9b for the
directly-hit tile, the 9e sweep for assembly-completion tiles (the S3
contract above).

Degenerate case: 1-tile door destroyed → step 2 is empty; just the state
flip. If the door was OPEN, its tiles are air — `destroy_wall` never
touches air (`gamemap.py:1027` gate), so open doors cannot be externally
destroyed in v1 (the retracted-panel invulnerability, gap §15.4).

---

## 9. Path-hold — one check, the existing suppression pattern

**Seam:** `_update_player_movement` (`simulation.py:936-959`). After the
can_move gate (:951-953) and the path-index computation (:954-955), before
committing `u.x/u.y` (:956-959): if the unit's next path position is no
longer enterable, **hold position and burn the tick** exactly like the
status gate — `u.path_tick_offset += 1; continue` (:952-953's pattern; the
path pauses, no catch-up teleport, the round may end before the tail is
walked — being door-blocked costs the distance, same as being knocked
down).

**The check:** `not gmap.is_passable_block(int(py), int(px), u.footprint)`
where `(px, py)` is `u.move_path[path_idx]` — the same block predicate the
plan-time A* used (`simulation.py:484-485`), anchored the same way the
unit's own footprint is (`int()` truncation = `tile_x/tile_y`,
`unit.py:372-379`). Fractional mid-step positions (:521-525) truncate to
the same tile block their endpoints use, so a close mid-interpolation
holds the unit at its current position, on-grid. When the door reopens
(or is destroyed), the check passes and the unit resumes at the next
un-walked index — no re-path in v1 (WEGO: the plan is the plan; gap
§15.3).

Zombies need nothing (N2, citation fixed): the load-bearing mechanism is
the **per-step `is_passable_block` re-check** — before committing each
zombie step, `update_zombies_tick` re-checks the next path tile against
the live grid and drops the stale path when blocked
(`ai_zombie.py:181-192`), so a door closing in a zombie's face stops it
at that step regardless of the every-5-steps repath cadence; the closed
material blocks the subsequent re-path's A* through the same table (§1).
Test §13.7 covers the marine case end-to-end.

---

## 10. Debug toggle hotkey (ruling 1, refined)

**Key: `O`** (mnemonic "operate" — the key ruling 5 names). Dev-only,
living beside the other DEBUG keys in `src/input_handler.py` (the I/J/U
block :108-135), NEVER in any shipped control scheme — same citizenship as
"I = ignite". **v2 fix: KEY_O was NOT free** — v1 checked only
`input_handler.py`, but the renderer binds O to the water-optics pass
toggle (`game_renderer.py:871`; main.py's help text). A dual binding would
flip the water rendering on every door toggle — directly poisoning the
§14.6 water-pool judgment — so the water-optics toggle moves to **`V`**
(free in both layers; render-layer dev toggle, zero sim surface;
main.py's help line updates to match). Precedent: config-reload moved to
Ctrl+R for exactly this class of collision (`input_handler.py:35-38`).

Flow, mirroring `_debug_ignite` (:228-246): `tile =
renderer.mouse_to_tile()` (`renderer/game_renderer.py:923`) — **N9
(folded): `mouse_to_tile()` returns `(x, y)` (col, row)**, so the handler
unpacks `fx, fy = tile` and calls `sim.door_at(fy, fx)` with the SAME
flip every other debug key applies — then `door_at` (a new sim helper)
returns the door whose runtime span contains the `(fy, fx)` tile (unique
by §4.2, ordinal-order scan; matches OPEN doors' spans too — the span is
geometry, not material); if found and alive:
`door.want_open = not door.want_open`, plus a `[debug]` console print.
Destroyed door or no door: print and do nothing.

**Ruling 5 (S5 resolved by Erik, 2026-07-19 — the sanctioned reading of
ruling 1):** the O-key stays dev-only, but the latch it flips IS synced
entity state (`want_open`, hashed as a runtime row §2c) with
retry-until-clear close semantics mirroring Arc B's while-held input.
Ruling 1's "NOT synced state" bound the KEY and its plumbing — binding,
cursor hit test, injection point, all render/dev-layer — never the entity
latch (`arc_a_patch_plan_2026-07-18.md:39-43`). This is the existing
debug-key contract: I/J/U already write `fire`/`gas`/`water_depth`
(synced, hashed fields) directly (`input_handler.py:228-303`); a dev
input is an external state injection between ticks, and the trajectory
that follows is deterministic given the injection. No key press →
bit-identical trajectory; lockstep/replay sessions have no such key. The
latch rides the digest so any injected toggle is VISIBLE in it — which is
what makes the human-test trajectory attestable at all.

---

## 11. Digest, dormancy, recorder — verification statements

- **Mechanism**: door rows enter `ENTITY_SECT_V1` via
  `runtime_digest_rows` with zero serializer changes
  (`serialize.py:151-158`); section version stays V1 (§2c). `__signals__`
  stays the empty-defined section (`serialize.py:174-197`) — the latch is
  an entity row, not a signal (the free-`alive` precedent, critique 7).
- **Dormancy (structural)**: entity-free levels build no `DoorRuntime`, the
  sweep is `if self._doors:`-gated, the loader stamp iterates zero doors,
  `MAT_DOOR_CLOSED` appears in no existing CSV, and the two capture call
  sites hand the serializer the same (entity-free) list as today → every
  existing level's trajectory, digest, and recorder .npz are byte-identical.
  Light-only levels: unchanged records (lights emit no runtime rows).
  Asserted by the full suite (§13.9).
- **No re-baseline**: no digest-suite level contains a door entity; the A6
  human-test level is NEW (§14). Existing goldens untouched — the arc's one
  re-baseline remains A7's (ruling 2/3). A6 adds no golden of its own; the
  logic golden with a sensor→door loop is Arc B's (entity doc §7 :266).
- **Recorder/get_state**: carry door rows automatically once both call
  sites use `self.entities` (§6.1) — the recorder persists the same bytes
  the digest hashes (one-serializer rule, `recorder.py:148-152`), and
  `get_state`'s carrier feeds `require_entity_carrier` strictness
  (`serialize.py:231-242`). Test §13.8 pins both.
- **GPU/CUDA lockstep**: flips are CPU structural edits at a pinned slot,
  exactly `destroy_wall`'s citizenship — pre-S8a both backends consume
  identical post-edit arrays every tick (A5 §9). The S8a dirty-set rider
  (A5 §9) already lists everything the sweep touches; `wall_hp` restamps
  ride the same `on_tile_changed` set. Nothing new to add.

---

## 12. Determinism summary

All new arithmetic is integer (span math in `Fraction` at load; HP
fold/restamp are int copies); all iteration orders pinned (ordinal door
order; row-major spans; the primitives' own N,S,E,W); the latch is synced
state read at one slot; no RNG anywhere in the door path; no dict-order
dependence (`self._doors` is a list in ordinal order). Two machines
running the same injected-toggle script produce bit-identical trajectories
— test §13.10.

---

## 13. Test plan (`tests/test_a6_doors.py` + a loader case; gate `pytest tests -q`)

Programmatic `LevelData` fixtures (the A5/`test_eos_p1_species_transport`
idiom) with `[[entity]]` door instances; sim built with
`breach_physics=None` where physics is irrelevant, real stepping where not.

1. **Span derivation**: 1.0 m @ 3 t/m → 3 tiles; 0.5 m → 2 (round-half-up
   of 1.5 — the banker's-round tripwire); 0.34 m → 1; h/v orientations;
   `res_factor=2` replication → the 2×2-per-base-tile rectangle; the
   0.333 → exactly-3 mapping.
2. **Load stamp + validation**: closed → `MAT_DOOR_CLOSED` pre-seed
   (atmosphere/gas 0 on span, wall_hp = quantized table HP); open →
   `MAT_AIR` (ambient seed); hard errors: OOB span, span over hull, span
   over vacuum, overlapping spans.
3. **Round-trip identity (§4.3)**: authored-open ≡ authored-air across
   every synced array; authored-closed ≡ CSV-painted `MAT_DOOR_CLOSED`.
4. **Flip cycle under the sweep**: toggle latch → open applies at next
   step's 9e (state row flips, tiles air, N conserved per slice — the A5
   invariant now exercised through the sim); close → seal + HP restamp;
   100 cycles → totals exact.
5. **Occupancy retry**: marine footprint overlapping span → close blocked,
   latch retained, retried each tick (state stays OPEN, `want_open=0`);
   move the marine off → closes on the next tick, marine-in-doorway never
   crushed. Corpse on span → closes over it.
6. **Water retry**: `water_depth > 0` on one span tile → blocked (via
   `can_seal_tiles`); drain → closes.
7. **Path-hold**: precomputed path through a doorway; close mid-walk →
   unit holds, `path_tick_offset` grows, no position change; reopen →
   resumes at next un-walked index. (The door-closes-across-path test the
   entity doc §7 demands, :253-256.)
8. **Capture carriage**: a stepped door sim's recorder
   `entity_snapshots[i]` bytes contain the door's `state`/`want_open`/
   `hp_*` rows and change across a flip; `get_state().entity_state` carrier
   matches the digest's section bytes (one serializer).
9. **Dormancy**: full existing suite green, zero golden edits; a light-only
   level's **per-tick record bytes** byte-identical to pre-A6. N6 scope
   pin: the byte-identity claim covers the recorder's per-tick
   entity/record bytes and every entity-free artifact —
   `registry_content_hash()` itself CHANGES by design when the `door`
   class registers (it folds the registry payload), which surfaces only
   in entity-PRESENT artifacts; that is match-setup provenance, not a
   dormancy break.
10. **Determinism**: identical toggle-script on two fresh sims →
    bit-identical synced arrays + digests every tick.
11. **HP ledger**: damage one tile of a closed 2-tile door
    (`wall_hp` write), cycle open-close → damaged tile restamped damaged,
    other tile full (no heal, no smear — the §7 rule).
12. **External destruction / whole-door — observables only (N3)**: two
    rooms > 2.0 atm apart, close the 2-tile door across the differential
    → within a tick the door dies. Asserted: all span tiles air, 2×
    `DoorDestroyedEvent` in that tick's events, `alive=0`, `state=2`,
    `hp_*=0`; a later toggle does nothing; digest rows stable thereafter.
    NOT asserted: which tile 9b popped vs which 9e completed — for a
    symmetric membrane the 9b victim choice is an implementation detail
    (sort order among equal differentials), and pinning the mechanism
    split would make the test brittle against find_burst_walls internals.
13. **B1 regression — blast destroys a closed door**: `apply_explosion`
    with door-killing `wall_damage` centered on a closed entity door →
    the door tile(s) destroyed (material air), and the next sweep
    completes the whole-door rule with 9e events. Guards the
    `physics.py:104` tuple forever.
14. **S2 timing — the boundary-tick lag, pinned**: a closed entity door
    destroyed by an end-of-phase-boundary explosive volley (slot 8 at the
    tick bottom, `simulation.py:868-886`) leaves ONE recorded tick whose
    entity rows still read CLOSED over minted-air tiles; the NEXT tick's
    9e reconciles (`alive=0`, `state=2`, assembly-completion events).
    Asserts both halves: the lag exists (documented behavior, not a bug)
    and it is exactly one tick.
15. **N7 — vacuum-adjacent door cycle**: a door span adjacent to exposed
    vacuum; open → the span joins vacuum permanently (`is_vacuum` set, no
    seed — A5 §5.2); re-close → `seal_tiles` succeeds gas-free and the
    tiles become the sealed-hull state (`solid ∧ is_vacuum` — a hull
    patch); re-open → joins vacuum again. Grid totals exact throughout;
    states/latch behave; nothing mints.

---

## 14. HUMAN-TEST plan (Erik plays; ruling 3 tranche gate)

**Level**: `levels/door_test/` — NEW, small (~40×30), never a migrated or
golden level. Content: two pressurizable rooms joined by a 3-tile door
(default 1.0 m); a 1-tile door (0.333 m) on an alcove; a corridor with a
door mid-path for the path-hold walk; a marine squad spawn; a shallow
water pool by one door (U-key tops it up); flat art (dev-quality).

Render obligations (N4, rescoped honest-minimal — dev-quality,
render-layer; the v1 wording overreached: `game_renderer.py:960` is the F6
*text HUD*'s hardcoded name dict, and there is NO per-tile material draw at
runtime — the ship is baked art). A6 ships exactly two small things, no
overlay system:

1. **F6 debug HUD name**: the hardcoded `{0: "air", 1: "hull", 3: "door"}`
   dict is replaced by a `MATERIAL_NAMES` lookup, so the cursor readout
   says `door_closed` (and every other material) correctly.
2. **A simple per-tile door draw in `compose_world`**: the sim hands the
   renderer its door list (span, state, alive — render-read only); closed
   door tiles get a flat amber fill, open door spans a thin outline (so
   there is something to aim the O key at), destroyed doors nothing.
   Drawn in world-pixel space beside the existing unit/grid draws.

The v1 "hovered-door highlight" is DROPPED (scope, N4) — the outline plus
the HUD readout locate the target well enough for a dev key.

Checklist (each is a feel judgment, not a pass/fail):

1. **Toggle feel**: O over a door opens/closes it; closed door blocks
   movement, sight-adjacent behaviors, and gas (pressure overlay).
2. **Ruling-4 rider (REQUIRED JUDGMENT)**: watch the pressure overlay as a
   3-tile door opens between unequal rooms — does the instant whole-span
   k+1 rarefaction (one-tick dip) read as a weird transient? If yes, the
   recorded options are staged tile-per-tick opening (entity-level;
   primitives unchanged) or the documented (d) delete+mint fallback
   (arc plan ruling 4; A5 §1) — Erik picks at the gate.
3. **Close-on-occupied retry feel**: close onto a standing marine — door
   refuses silently and slams the tick after he steps off. Does the
   silent retry read as "the door is trying", or does it need render
   feedback (a strobe/hint) before Arc B?
4. **Path-hold feel**: order a marine through a doorway, close it mid-walk
   — he stops at the door and resumes when it reopens. Does burning ticks
   (lost distance) feel fair in WEGO?
5. **Burst-after-seal**: pressurize one room (J/grenades), slam the door
   across the differential — it pops next tick with the relief-valve
   event. Correct-feeling physics or trap?
6. **Evacuation feel**: close a door in a small sealed room — the neighbor
   pressure bump (A5 close-T + evacuation) on the overlay.
7. **Destroyed stays dead**: after 5, the O key does nothing on the wreck.

---

## 15. Accepted gaps v1 (workflow rule; revisit at arc close)

1. **Instant whole-span open/close** — staged tile-per-tick opening is the
   noted option, pending the §14.2 judgment (ruling 4).
2. **No crush mechanics** — living footprints block closes; corpses and
   dropped items (none exist) are sealed over silently.
3. **No re-path on door change** — held units resume the stale plan; only
   plan-time A* sees door state. WEGO planning-loop integration (and any
   direct-control scheme) owns this later.
4. **Open doors are invulnerable** — the retracted panel has no tiles;
   HP frozen while open; `destroy_wall` can't touch air (§8).
5. **Doors don't burn/conduct specially** — `flammable = false` copied
   from `[materials.door]`; the A5 fire-carryover gap (§10.4) applies to
   `MAT_DOOR_CLOSED` unchanged.
6. **Door-over-water = refuse only** — displacement is future work (the
   locked entity-doc §7 water rule).
7. **No signals** — `is_open` declared but silent; the latch is the only
   driver until Arc B (entity doc §3d).
8. **Legacy painted doors unchanged** — still walkable-but-flow-solid;
   convergence is A7's migration + the arc's one re-baseline (ruling 2).
9. **`MAT_DOOR_CLOSED` in a CSV outside any span** is legal-but-inert
   authoring (an unopenable pseudo-door wall); Arc C validator warning.
10. **No door art/animation** — dev rendering only; real art is Arc C+.
11. **Latch injection is dev-only** — no order/replay citizenship for the
    O key; a lockstep-legal door driver arrives with Arc B signals (or the
    control-scheme arc).
12. **`hp_*` rows lag live `wall_hp` between folds** (§7) — exact at every
    boundary where they are load-bearing; both are hashed.
13. **`door_closed` is hand-paintable in every editor palette** (N5, §1) —
    `MATERIAL_NAMES`-derived palettes pick it up automatically; painting
    it outside a door span is the §15.9 pseudo-door. Arc C's DOOR tool +
    validator own the cleanup.
14. **Vacuum-adjacent doors convert to hull patches** (N7, §13.15) — a
    door opened next to exposed vacuum joins vacuum permanently; closing
    it again yields the sealed-hull state, not a working doorway.
    Physically coherent (the doorway vented to space; the panel now seals
    the hull); a level authoring a door beside a planned breach site gets
    this behavior by design.
15. **One boundary-tick entity/grid lag on phase-boundary explosive
    destruction** (S2, §8, §13.14) — deterministic, reconciled next tick.

---

## 16. Canon errata queued for arc close (do NOT edit locked docs now)

- Entity doc §7 :241 + eos §2.2 join sentence — already queued by A5 §1
  (conservative withdraw-seed vs minting destruction).
- Editor doc §6 :100 "its `MAT_DOOR` tiles" → "its door-material stamp
  (`MAT_DOOR_CLOSED` when closed, as of A6)" (§1).
- Entity doc §7 :257-259 "loads bit-identical" → precise reading pinned:
  field-state identity; the digest legitimately differs by the entity
  section (§4.3).
- A5 doc §2 "For doors v0 this will be `MAT_DOOR`" → superseded by
  `MAT_DOOR_CLOSED` (§1; A5 doc is append-only, amendment recorded here).
- Ruling 1 wording — RESOLVED by Erik as **ruling 5**
  (`arc_a_patch_plan_2026-07-18.md:39-43`, 2026-07-19): the O-key stays
  dev-only; the `want_open` latch it flips is synced entity state. §10
  cites the ruling; nothing left to errata beyond noting it at fold.

## 17. A5 §11 rider resolution (required checklist)

| Rider | Resolved by |
|---|---|
| 1. External destruction / entity-tile desync | §8: grid observation at 9e same tick; whole-door rule; dead latch; never re-seals. |
| 2. Door HP as entity runtime state | §7: per-tile `hp_*` rows; restamp on close, fold on open; no heal, no smear. |
| 3. `can_seal_tiles` composition | §5.2: `occupancy_clear(span) and can_seal_tiles(span)`; primitive raises are bugs, never "door stays open". |

---

*A6 design v2 — critique folded (B1 blast tuple; S1 `--res` base
recovery; S2 boundary-tick lag accepted; S3 event contract; S4 hp order
pin; S5 = ruling 5; N1-N10 corrections; the v2-discovered O/V key
collision fix). Author: Claude (Arc A design + implementation agent),
2026-07-19. Load-bearing sections: §1 (MAT_DOOR_CLOSED + B1), §3 (S1
recovery), §5-§8 (sweep, latch, HP, destruction), §4 (load order), §10
(ruling 5). Implementation follows this doc exactly.*
