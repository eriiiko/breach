# Entity system — design v2 (LOCKED)

**Status: LOCKED 2026-07-18** — Erik's final read approved ("everything is
perfect"); critique folded (48 findings, 3 critics); the units-design-now
question resolved as NO (§3e note). Companion:
`level_editor_v3_design_2026-07-18.md` (the editor VIEW; locked same day).
**Build per the arc plan (§10); Arc A is cleared to start.** Changes from
here require reopening the design — note it in §9 if so.

---

## 1. What this is

One model for everything placeable that isn't paintable matter: doors,
sensors, logic nodes, pumps, lights, the data chip — placed as instances of
registry classes, connected by signals. The editor renders and edits this
model; the game runs it. Logic is dataflow over signals, never a scripting
language. **Units are NOT entities in v1** (§3e).

## 2. The three layers — where programming lives

| Layer | Lives in | Authored by | Example |
|---|---|---|---|
| **L0 — behavior** | Python code (`src/simulation/entities/`) | Erik + Claude, per class | what "door closed" means physically; how a sensor samples; the airlock_controller state machine |
| **L1 — vocabulary** | the class declarations themselves (registry = code, §3b) + `entities.toml` tuning overlay | code for schema, TOML for dial numbers | door class exists, has `state`, emits `is_open`, accepts `toggle` |
| **L2 — composition** | `level.toml` (editor or `level_lib` scripts) | level design | THIS plate wires through THIS decider to THIS door |

No L3 (embedded scripting) — anything dataflow can't express becomes a new
L0 node class in code: testable, deterministic, reusable.

**Hot-reload constraint (critique):** the `entities.toml` tuning overlay is a
**dev-only** affordance. In any lockstep session or ML rollout the registry
content-hash is part of match setup (like the seed); mid-run reload is
disabled. Changing a registry number that alters behavior is a legitimate,
deliberate golden re-baseline event — documented as such so digest gates are
respected, not "fixed around".

## 3. Classes & instances

### 3a. Instances have mandatory ids (critique blocker)

Every `[[entity]]` instance carries a unique `id` (editor auto-generates
`door_3`-style slugs; user-renamable). **All references — wires, entity_refs,
zone bindings — address ids, never array positions.** Loader hard-errors on
duplicate ids, warns on dangling refs (a dangling ref at load is an authoring
error; a destroyed entity at runtime is not). Ids are assigned in file order
at load, never reused after destruction; **every runtime sweep (logic, tag
expansion, wire fan-out, structural door sweep) iterates in id order** — the
single ordering rule that makes all iteration deterministic.

### 3b. The schema lives in code — DECIDED: direct import

Each entity kind is a Python subclass of `Entity` declaring its schema
(fields/signals/inputs, with types, defaults, constraints) as class-level
declarations beside its L0 behavior; a registration decorator adds it to the
registry. The editor **imports the entity module directly** (it already
imports `simulation.materials`); the JSON-export alternative is rejected for
staleness. Two hard constraints:

- **Import-light rule (tested):** `import simulation.entities` must succeed
  with no compiled `breach_physics` present and without pulling in
  `simulation.simulation`. A CI test enforces this.
- **Editor failure mode:** the editor wraps the import; on failure (Erik's
  half-written class has a syntax error) it falls back to the last-good
  `entity_registry.json` (rewritten on every successful game/editor launch)
  and shows a persistent red status-bar banner with the exception. The
  palette stays usable; only the broken class is absent.

`entities.toml` holds tuning-number overrides only ("No schema in TOML").
The `[factions]` table's future home is stack-2's faction patch, not
entities.toml.

### 3c. Wiring at scale — tags + the authoring API

Unchanged from round 3 (accepted): instances carry `tags = [...]`; wire
targets are `id` or `tag:name`, tags resolved at runtime in member-id order.
`level_lib` is the scripted authoring path — **and (critique) it is THE
single data layer for level files, period**: the map editor, the baker's
TOML writeback, the migration tool, and ML variant generation are all its
clients. One writer implementation, ever. Two-writers session conflicts are
handled by the editor recording level.toml mtime+hash at load/save and
prompting reload-or-overwrite on mismatch; `level_lib` writes atomically
(temp + rename); ML variant generation always writes to new folders.

### 3d. Interactions & cost policy — REVISED: v1 is signal-only (Erik)

Erik postpones the control scheme (WEGO may become direct gamepad control),
and chose **signal-only for v1**: this arc ships NO unit-initiated
interaction path at all. No ORDER_USE, no AP table, no "who operates" rule —
the whole intent-delivery + cost-policy layer is deferred to the
control-scheme decision. Consequences:

- `button` / terminal classes are **format-reserved but inert** in v1
  (placeable, wired, never pressable — they wake up with the control scheme).
- Doors are driven by authored initial state, sensor logic, and destruction.
- Pen release v1 = damage, or sensor tricks (presence plate inside the pen's
  service corridor, pressure drop, a timer latch).
- The airlock showcase becomes **fully automatic**: a presence plate
  (`sensor_motion`) in the chamber triggers the cycle — see §6b.
- Entity schemas still *declare* interactions (format-reserved) so the
  control-scheme arc adds a policy layer without touching entities or levels.

### 3e. Units are OUT of the entity system in v1 (Erik + critique)

No `[entity.marine]`. Units spawn via the permanent `[[spawn]]` syntax or
breach-site rosters (editor doc §5), constructed as `Unit` exactly as today.
**Units design pass now? NO (Claude's call, delegated by Erik 2026-07-18,
at lock time).** Substantial unit design already exists
(`breach_unit_class_design.md` + the landed unit-class-foundation patch);
its open items (modifier system, environment damage, faction matrix,
variants) belong to stack-2 — which is also when convergence happens. This
seam contract is the only interface Arcs A–C need; a units pass now would go
stale before it builds. Design just-in-time, at stack-2.

**Convergence contract (written now, executed at stack-2):** entity ids and
unit ids will share one id space (one counter, one namespace) so wires and
sensors can later address units; unit signals (`hp`, `alive`, `faction`)
arrive with the stack-2 faction work; `SYNCED_UNIT_FIELDS` and entity-state
digests merge at that point as a planned, single re-baseline. Until then no
wire, tag, or `hp_below` sensor may reference a unit — the loader rejects it.

## 4. Signals & logic — the dataflow model (hardened)

Core unchanged: integer signals (Q16.16 where physical), nodes read
previous-tick inputs / write next-tick outputs, cycles legal, latches emerge.
Critique hardening:

- **Two-stage latency contract (pinned):** sensors write their *current-tick*
  samples before the logic sweep (same-tick visible); logic nodes read
  prev-tick / write next-tick. Total sensor→actuation latency is **2 ticks**,
  stated and golden-tested. (A literal "everything double-buffered" reading
  would shift latencies by a tick — this contract is the canonical one.)
- **Input aggregation (decision):** *while-held* inputs OR across all driving
  wires; *edge* inputs fire **once** per tick regardless of how many wires
  pulse them. Complementary-input conflicts (open AND close same tick)
  resolve by a fixed per-class priority declared in the registry — for doors,
  **close beats open** (the safe state).
- **The filter node (τ pinned):** integer EMA with shift alpha. `k =
  round(log2(τ · tps))` snapped **once at load** (door-2 quantization);
  filter state carries k guard bits and rounds-to-nearest before the shift
  (kills the truncation offset that could park a value permanently below
  threshold). The editor displays the *snapped effective τ*, meters-style.
- **`alive`, fail-deadly AND fail-safe — both, per wire (Erik confirmed):**
  every entity emits a free `alive` signal (1 while functional). A destroyed
  entity's signals read 0 and inputs go dead — so a bare `p < 0.8 → close`
  wire is **fail-deadly** (sensor dies → doors slam), and gating on alive
  makes the same wire **fail-safe** (sensor dies → condition can never
  fire). Both exist side by side in one level; the author picks per wire.
  Ergonomics: deciders carry an optional `require_alive = true` flag —
  sugar for AND-ing the source's `alive` — so fail-safe is a checkbox in
  the editor, not a hand-built gate. Sensor sabotage stays a real tactic,
  deliberately, exactly where the author allows it.
- **L0 node classes touch the world through the SignalBus ONLY.** The base
  class hands them prev-tick signal reads and next-tick emits — never entity
  references — so a controller's list position can never become observable
  (the order-independence rule survives arbitrary L0 code).

## 5. Physical logic (hardened)

Physical by default, `intangible = true` per node — unchanged. Hardening:

- **Sample tile ≠ body tile (critique blocker).** A physical sensor's body
  tile may be solid (wall-mounted) — sampling it would read 0 forever and
  slam every blast door at level start. The registry declares a `sample_tile`
  (the faced air tile, an offset field set at placement); behavior when the
  sample tile turns solid (buried) or vacuum (breached) is just the honest
  reading of that tile — no special case, but now it's the *right* tile.
- Destroyed node: signals 0, `alive` 0, inputs dead (§4).

## 6. Sensors — v1 catalog (split per critique)

**v1 (this design):** `pressure`, `temperature`, `o2`, `smoke`,
`water_depth`, `fire` (field sensors, each with optional `area`), `clock`,
`door.is_open`, and `sensor_motion` with `faction_filter = any | team-int`.

**Stack-2 riders (format-reserved, NOT built here):** `hostiles-of`
filtering (needs the hostility matrix), `chip.carried_by` (needs carry
rules), the `win(...)` sink (needs objective rules), `hp_below` on units
(needs §3e convergence).

Pinned geometry (critique):

- **`sensor_motion`:** a unit counts iff integer `dist²(sensor_tile,
  unit_anchor_tile) ≤ r_tiles²` AND (if `needs_los`) `has_los(sensor_tile →
  unit_anchor_tile)` with the **sensor as Bresenham origin** (has_los is
  direction-asymmetric — origin pinned). `min_footprint` compares the unit's
  *declared* footprint field, nothing geometric. `r_tiles` quantized from
  `radius` (length_m) at load by the canonical rule (editor doc §4).
- **Area-mean:** the disc is `dist < radius` (FieldEdit's strict rule),
  masked to **currently non-solid** tiles, integer sum with the live
  non-solid count as divisor, floor division. The set legitimately changes
  when a wall inside the disc is destroyed — that's physics being sensed,
  not drift.

## 6b. Actuators — pump respecced as an N-feed (critique)

Post-EOS, nothing injects "pressure": the pump is a **gas-mass (N) feed**.

- Per-tick quantum computed **once at load** (door-2): from `rate` (atm/s
  authored) via `ΔN = P/(C·T_std)` at standard temperature, integer Q16.16.
- **Inject** at the fixed standard O₂/N₂ mix; **extract** proportional to
  the tile's current composition, **clamped at zero** (the atmosphere edit
  path gets an explicit clamp — an unclamped remove could drive N negative
  and hand the Helmholtz solver garbage).
- `at_target` has a declared hysteresis band (default ±0.05 atm) — no
  bang-bang chatter pulsing edge-wired consumers at tick rate.
- The pump has a **port tile** distinct from its (solid) body — otherwise
  the skip-solid mask vetoes the pump's own edit and it silently no-ops.
- Intent note: a pump feeding a breached room makes venting non-terminating
  (steady source vs vacuum sink → steady wind). That is correct physics and
  occasionally a level mechanic; not a bug.

**The airlock (COMMITTED deliverable, now automatic per §3d):** chamber
presence plate + `airlock_controller` (L0 state machine: detects occupancy,
closes both doors, pumps to the far side's target, opens the far door;
`busy` signal; SignalBus-only I/O per §4). Terminals join when the control
scheme lands. Prefabs stay v2 — but the editor arc gains **clump
copy/paste** (multi-select copy preserving internal wires, re-id on paste,
external wires dropped), which covers most prefab value for the acceptance
level's several airlocks.

## 7. Runtime — pinned against the real conductor (critique blockers)

`simulation.py step()` numbered slots are the contract; entities add ONE new
slot. Pinned:

- **Slot (new, "9e entities"): after the exchange couplings (9c…), BEFORE
  the recorder snapshot.** Order within the slot: sensors sample this tick's
  post-physics fields → logic sweep (id order) → inputs resolve (§4
  aggregation) → **door structural sweep**. Recorder/digest therefore see
  entity state consistent with the signals that caused it, same tick.
- **Door flips are STRUCTURAL, not FieldEdits** (engine/13's own carve-out:
  topology stays structural). The sweep collects flips and applies them in
  entity-id order through the `destroy_wall`-style `on_tile_changed` cache
  rebuild. Effects reach the solvers next tick via the step-6 restamp —
  consistent with the one-tick-delay story. FieldEdit's flush-once,
  stable-sort contract is never touched.
- **Mass conservation prerequisite:** the EOS evacuation rule (design §2.2,
  the door-close half that was never built) ships BEFORE doors v0, with a
  test cycling a door in a sealed room asserting exact N conservation.
  Closing a door *evacuates* its tiles' gas to neighbors; ~~opening uses the
  existing joins-open-air rule~~ *[ERRATA 2026-07-19, ruling 4 — see a5 doc
  §1 (`archive/a5_evacuation_impl_2026-07-18.md`): as built, door flips use
  the conservative `seal_tiles`/`unseal_tiles` pair — the open direction
  withdraws its seed from the donor neighbors, equalizing over donors plus
  the opened tile (the k+1 divisor), so cycling is exactly N-conserving.
  Only DESTRUCTION (`destroy_wall`) keeps the minting joins-open-air rule —
  bounded and un-cyclable]*. **Water rule v1:** a door refuses to close
  over `water_depth > 0` (same blocked semantics as units — simple,
  deterministic; displacement is future work).
- **Occupancy rule (decision):** a door close is **blocked while any living
  footprint overlaps the span** — the close input (while-held) naturally
  retries; no crush mechanics in v1. Note: today's painted `MAT_DOOR` is a
  hybrid (passable to movement, solid to flow — the door-stamp-leak fix);
  entity doors' CLOSED = *fully* solid is a real behavior change. The
  migration of painted doors to entity doors is a **deliberate,
  once-per-arc golden re-baseline event with written rationale**, and the
  door-stamp-leak regression guard is ported to the new semantics.
- **Path invalidation (critique blocker):** movement gains the one check the
  WEGO contract needs: a unit whose next path tile is now impassable
  **holds position and burns the tick** (existing can_move-suppression
  pattern). Test: a door closes across a moving unit's precomputed path.
- **Load order:** entity-derived tile state (open door = air) is applied
  BEFORE field seeding; round-trip test asserts an authored-open door level
  loads bit-identical to the same level authored as air.
- **Digest & save (critique blocker):** SignalBus buffers + per-entity
  synced state (door state, EMA accumulators, controller phase, edge
  detectors' prev values) are integer synced state — hashed as
  `__entity__`/`__signals__` digest sections from Arc A patch one, included
  in `get_state`/recorder dumps (the f601455 recorder lesson), and covered
  by a **logic golden**: a small level with a sensor→filter→door feedback
  loop, trajectory-hashed cross-machine. **Dormancy guarantee:** an
  entity-free level is bit-identical to today; existing goldens are never
  re-baselined by this arc (sole exception: the deliberate MAT_DOOR
  migration event above).
- **S8a contract (critique blocker, cross-arc):** sensors must not
  reintroduce per-tick full-field GPU→CPU streaming. The sample-site list
  is static per level → a small gather kernel packs all sensor samples
  (incl. area sums/counts) into one compact D2H per tick, defined against
  that tick's materialized P. **This contract is written into the S8a spec
  rewrite (priority ledger item 1) before Arc B builds sensors.**

## 8. What stays out

Embedded scripting (indefinitely) · arithmetic combinators, multi-channel
wires, displays (v2+) · ORDER_USE + cost policy (control-scheme decision) ·
hostiles-of / carried_by / win / unit signals (stack 2) · full prefabs
(editor v2; clump copy/paste ships instead) · creature AI (beastiary track).

## 9. Decisions log

2026-07-18, rounds 1–5: joint model+view arc · dataflow over scripting ·
three layers · physical-by-default + intangible · meters-first · schema in
code · tags + level_lib · sensor catalog closed · pump + airlock committed ·
interaction/cost split · SVG icons.
2026-07-18, critique round (Erik's rulings): **Arc split A→B→C accepted,
physics close-out before Arc B** · **dead sensors fail-deadly with the
`alive` idiom** · **v1 signal-only, ORDER_USE deferred to the control-scheme
decision** · **units out of the registry, convergence contract written**.
2026-07-18, LOCK: Erik approved both docs ("everything is perfect") ·
fail-safe AND fail-deadly coexist per wire, `require_alive` decider flag
added (§4) · units design pass deferred to stack-2 (Claude's delegated
call, §3e) · wall-burst-differential blessed and merged same session.
Plus all critic-resolution decisions recorded inline above (close-beats-open,
occupied-blocks-close, structural door sweep, slot 9e, sample tiles, pump
N-feed, τ snap, id scheme, S8a gather, dormancy guarantee).
2026-07-19, Arc A close-out (as built; A1–A9 merged, Erik blessed):
**ruling 4** — door flips use the conservative `seal_tiles`/`unseal_tiles`
pair (exact N; open seed equalizes over donors + tile, k+1 divisor);
destruction keeps the minting rule (Erik's rubble model canon: debris
displaces volume, pressure constant); close-T = solid-neighbor mean, air-T
fallback · **ruling 5** — the door toggle is a synced `want_open` latch
(retry-until-clear); the O key stays dev-only · **`MAT_DOOR_CLOSED` (id 7)
forked** — closed entity doors are fully solid; painted `MAT_DOOR` stays
the legacy walkable hybrid · **whole-door-dies** — partial external
destruction kills the assembly, no stub panels · **door HP is a per-tile
runtime vector** (fold on open, restamp on close — no heal, no smear) ·
**A7 finding: the sanctioned re-baseline was EMPTY of committed artifacts**
(every golden derives from synthetic content; only `test_level`'s
uncommitted digest identity flipped — `archive/
a7_rebaseline_rationale_2026-07-19.md`) · **units still out** — the §3e
stack-2 convergence contract is unchanged. Canon fold:
`architecture/engine/16_entity_system.md`; arc docs archived under
`docs/archive/`.

## 10. The arc plan (replaces editor doc §9)

**Arc A — entity foundation** (may start on Erik's word):
Entity ABC + registration + import-light test + registry export/fallback ·
`[[entity]]` format with ids + legacy `[[spawn]]`/`[[light]]` aliases
(mixed-form hard error; migration is explicit via a one-shot tool that also
groups painted MAT_DOOR tiles into door entities) · `level_lib` as THE data
layer · EOS evacuation rule (prerequisite patch) · doors v0 (structural
sweep, occupancy/water rules, path-hold, load order, re-baseline event) ·
zones (id-grid + instance binding + validators + `--res` replication) ·
`air_init.npy` seeding · `boundary` field · `__entity__`/`__signals__`
digests + dormancy tests. Gates: digest-mechanical + the doors HUMAN-TEST.

**— physics close-out runs here (ledger #1: S8a spec rewrite WITH the
sensor-gather contract, residency patch, boundary conditions) —**

**Arc B — logic** (gated on the S8a contract): SignalBus + slot 9e ·
node set (decider/gates/filter) · v1 sensor catalog · pump · the automatic
airlock_controller · logic golden + cross-machine attestation. Gates:
digest + logic golden + HUMAN-TEST (feel of doors/sensors in play).

**Arc C — editor UX** (any time after A; benefits from B for wire UX):
panes shell + registry-driven palette/inspector (SPAWN/LIGHT modes honestly
ported onto the entity model, bespoke paths deleted) · transaction-log undo
(compound grid+entity ops; wires/tags/zones/field-edits all covered) ·
multi-select (box + shift-click) + tag assign + clump copy/paste · wand +
zone/air paint · two-click wire tool + selection-scoped overlay + tag badges
· play-from-editor (levels/_editor_scratch/, sys.executable, bake-reuse) ·
icons (SVG sources + committed PNGs + tools/rasterize_icons.py + generated
chip fallback). Gates: Erik-drives-it HUMAN-TEST per patch.

The Contested Chip acceptance level is authorable at the end of C and fully
runnable at the end of B (order between B and C is Erik's call at the time).
