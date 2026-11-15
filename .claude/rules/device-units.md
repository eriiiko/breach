---
paths:
  - "cpp/src/cuda_unit_*"
  - "cpp/src/unit_*_layout.h"
  - "src/simulation/unit_store.py"
  - "src/simulation/unit_tables.py"
  - "src/simulation/unit_events.py"
  - "src/simulation/unit_requests.py"
  - "src/simulation/unit_kernels.py"
  - "src/simulation/unit_draws.py"
  - "tests/cuda_unit_*.py"
  - "tools/gen_unit_layout.py"
  - "src/simulation/applier.py"
  - "cpp/src/applier_layout.h"
  - "cpp/src/cuda_applier.*"
  - "cpp/src/cuda_topology*"
  - "src/simulation/object_tables.py"
  - "cpp/src/object_tables_layout.h"
  - "cpp/src/cuda_object_table.*"
  - "tests/cuda_object_table_check.py"
---

# Device units: the store, its tables, its library, what kernels emit

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say. Every
system here is Class T: NOT part of the played tick until the `device` backend
(P8b). Which slots are ported is the twin manifest's to say.

## Unit store
- A unit is a ROW: the schema's group-S columns in schema order, then `valid`. Row r is the r-th unit of `sim.units` (id order, the laws' order); `(team, slot)` are columns, never the row order; a batch pads to the largest U with all-zero rows. Built only through `pack` / `unpack` over the schema's own `encode_unit` / `decode_into`.
- C++ reads offsets ONLY from the generated header. A new, removed or re-widened schema column, a config row that moves a block's width included (a hand-grenade row, a status kind, a damage type, a hook), means `tools/gen_unit_layout.py --write` and commit in the same patch, then a rebuild of `cpp/build_cuda`.
- NOT PART OF THE PLAYED TICK: no slot, `ResidentBatch` or `run_resident_batch` touches it, and the device never drives the game.
- `UnitStoreDevice` is the buffer, its copies, the per-env counts and the input staging: NEVER a method per slot.
- A new slot kernel is all of: its table values as Device tables rows; a free function in its family's TU `cpp/src/cuda_unit_<family>.cu` (declared in `cpp/src/cuda_unit_kernels.h`, on CMakeLists' CUDA list) over the Device unit library; a binding returning the three counts (coverage, faults, refused); a function in `src/simulation/unit_kernels.py`; a Class-T twin row; its shadow lockstep. A slot that emits goes through the Event ring, a slot that does what is not a row write through the Request list.
- A unit kernel is born `(N, U)` (one thread per row, padding skipped) on `tstream`, reads no law constant from its arguments, copies through pinned staging, and is integer for integer the Python law: an arithmetic change lands in the law first.
- A ported slot is gated by the SHADOW LOCKSTEP (`tests/cuda_unit_lockstep.py`), never a hand-rolled compare: rows at tol 0, env i of N == env i alone, the coverage counter == the rows the host changed, the hooked match equal to an unhooked one, scenes with non-neutral hooks for every hook the slot reads. Adjacent ported slots are also gated by the chained form.
- A slot that writes the map declares its FIELD OUTPUTS (`DeviceSlot(..., outputs=("field:<attr>", "book:<name>"), host_only=((attr, reason), ...))`): every declared plane is compared bit for bit, every book against the host's delta. Never a hand-rolled plane compare.
- The WRITE-SET gate fails on any GameMap attribute the host slot changed that is neither an output nor host-only with its reason: a plane the device forgot cannot pass silently. Arrays that are the same object (`obstacles` is `solid` until the first stamp) count as one; an object attribute is traced through an `OBJECT_PROBES` row.
- An input the conductor hands the slot as an argument (slot 7's burn-through list) is an `ARG_INPUTS` row, never a sim attribute invented for the test.
- A host LIST a slot reads or writes (the attack slot's effects; P7's tables and logs) is declared like a field (`inputs=("list:<name>",)` / `outputs=("list:<name>",)`) through ONE record function per list (a `LISTS` row), never a hand-rolled compare. A label whose callables also run outside the slot narrows itself with `DeviceSlot(select=...)` (slot 4c's resolve is the conductor's, not slot 2''s landing in `_update_rounds`).
- The ring is compared on SYNCED events only (`unit_events.is_synced`): a slot's render-only events have no device form and are dropped before the compare.

## Device tables
- EVERY table value a device unit kernel reads is ONE ROW of `unit_tables.ROWS`: a name, a width, the function reading it from the SAME validated object the reference law reads (never re-derived from config), and the range the kernels' arithmetic was proved for (refused by name outside it).
- A new table value is: one row, `tools/gen_unit_layout.py --write`, commit, rebuild `cpp/build_cuda`. Never a hand-edited device struct, never a law constant passed as a kernel argument.
- C++ reads offsets, hook indices and status ids only from the generated header, all in the fingerprint `DeviceStore` checks against the compiled module.
- A group-D unit value (a stability, a class hp) is reached through the group-S column that selects its row.

## Device unit library
- A function two slot kernels need lands HERE, never a copy in a slot TU; a slot's own law stays in its TU. Integer for integer the Python it names, held equal on synthetic rows over the whole range through TEST-ONLY probes (`cpp/src/cuda_unit_lib_probe.cu`, `cpp/src/cuda_unit_movement_probe.cu`, `cpp/src/cuda_unit_verbs_probe.cu`).
- A row whose operands leave the proved range FAULTS and is left unchanged; `put` marks a row changed only when it changes.
- The verb gates' device forms are `src/simulation/verb_gates.py`'s; P8b's mask kernel calls them too, never its own copy.
- THE DEVICE DAMAGE PIPELINE `cpp/src/cuda_unit_damage.cuh`: `apply_packet` is `damage.apply_packet` integer for integer, on any row it is handed, its events through the sink it is given, all or nothing. Every slot that hurts calls it: never a second packet law in a slot TU. Its `felt_temperature_q` is the heat row's and, from P8b, the observation's one read.
- The exposure landing is a MODE of that one `apply_packet` (an `ExposureLanding` handed in, landed by `land_exposure` at the walk's end), chosen by the CALLER per (stage, kind): a HIT in a stage that lands (`applier.STAGES`), never a packet's own choice and never a second packet law. Every other caller keeps the immediate form; a blast packet inside a resolve is immediate (the reference's `exchange.apply_blast_damage`).
- `impulse_add` / `angular_q` are THE device `unit_impulse.add` (the shove and the applier's PUSH both call them); `can_melee`, `melee_adjacent`, `melee_arc_target` are the melee verb's, also P8b's mask kernel's.
- `mag_settle` / `reload_begin` are THE device magazine law (`combat.mag_settle` / `reload_begin`): slot 1b's reload and slot 4's trigger both call them; a change to the Python magazine law lands here in the same patch.
- The falloffs (`falloff_floor_q` / `falloff_round_q`) are `FP_HD` kit twins in `cpp/src/fixed_point.h` (the Fixed-point kits' rule): a deposit weight, a blast's damage or its wall damage on the device calls them, never a re-derived root; a -1 is a FAULT. The kit's probes are `cpp/src/cuda_unit_kit_probe.cu` (draws, sight, falloffs), gated by `tests/cuda_unit_kit_check.py`.

## Device draws
- EVERY random number a device kernel draws is `cpp/src/cuda_unit_draws.cuh` over the vendored `philox32.h`: `draw_at` is `MatchDraws.at`, `draw_next` on a `DrawSite` is `.next`, `cone_offset_q16` / `roll_q16` the maps. Never a second block layout, never a curand stream, never a salt typed by hand: a salt is the generated `unit_draws::SALT_*`, so a new `Salt` member means `tools/gen_unit_layout.py --write`, commit, rebuild.
- The key and the tick reach a kernel as ONE per-env row (`unit_draws.draw_key`: `sim.draws.key` and `sim.draws.tick` wrapped to 32 bits, the WEGO base included), staged as a kernel input per env: never a kernel argument per draw, never `sim.tick` read alone.
- `.next`'s count is LOCAL TO THE ROW'S THREAD, exact only while every `.next` user of one salt sits in ONE kernel per tick: `unit_draws.NEXT_SITES` records the one `src/` site of each such salt and `tests/test_unit_draws.py` fails on a second. A second site needs the count moved off the thread (a design change), never a second counter.
- An operand outside the Python's domain (an id outside [-1, 2^32 - 1), an index outside [0, 2^32), a cone past `CONE_MAX`) sets `bad`: the caller FAULTS.

## Device sight
- EVERY line a device kernel asks is `unit_sight::has_los` (`cpp/src/cuda_unit_sight.cuh`): the sight form over `solid` and `sight_screen`, `physical` (a null screen) over `solid` alone. It is `GameMap.has_los` step for step: an arithmetic change lands in `has_los` first (the Sight rule), then here and in `sight_lines.cpp`, in one patch.
- `sight_screen` is a slot INPUT plane until the device tick makes it resident (`applier.non_resident_planes()`); the bulk form (`rays_clear`, the observation's) stays `sight_lines.cpp`'s until P8b, never a second device walk.

## The march (device)
- `unit_march::march` (`cpp/src/cuda_unit_march.cuh`) is `combat.BulletInFlight.advance` on the continuous path integer for integer, on ONE round-table row: slot 4's shooting (the rounds marched in their firing tick) and slot 2''s rounds both call it; a third caller (the beam, a turret) calls it too, never a copy. A change to `advance`, `push_on`, `chew_wall`'s record, `attack_resolver`'s exposure / crit rolls or `draws.round_index` lands here IN THE SAME PATCH: the corpus fires few rounds, so the twin ledger traces little of it, and `tests/cuda_unit_shooting_check.py` (the march on synthetic rounds) is the guard.
- The first hit is the first VALID ROW in row order (= `sim.units` order) whose footprint holds the tile, the shooter excluded BY ITS ID (the reference excludes by identity; a converted shooter keeps its id), the dead skipped. Its draws are `draw_at` with the reference's `round_index(tick - spawn_tick, ordinal)`, lazily (cover below ONE, a crit row): never a draw the reference does not make.
- Its records go to the sink it is handed, in the reference's order (CHEW, or HIT then PUSH, then a payload round's PAYLOAD); it writes no unit row. Physical cover and the facing defense (OnePhaseWEGO) are not ported (the round table refuses them).
- ACCEPTED GAP: the beam (`combat.fire_beam`, the shipped `lance_3` / `lance_5`) has no device form; slot 4's kernel refuses a hitscan trigger by name (`unit_kernels.ShotRefused`).
- ACCEPTED GAP: `combat` reads the PROCESS-WIDE weapon tables (`weapons.get_tables()`, rebuilt by every `Simulation` at its tile size) while the device tables read each sim's own `weapons_tables`: a batch must share one tile size (a finding for Erik, not changed).

## Noise list (device)
- `unit_schema.NOISE_TABLE` on the object tables is THE device `GameMap.noise_log`: a device producer of a noise (slot 4's shots today; payloads at P7c.1, the bash at P7d) appends `(fy, fx, radius)` in the reference's emission order (a per-row producer through the staging and the one gather, a serial one through `emit_record`), and a radius of 0 records nothing (`noise.emit`). Its record function is `unit_kernels.noise_of`.
- It is NOT a per-tick list: it persists across the tick boundary, is drained at slot 5's head (P7d's stage) and cleared only at env reset (design 4.4c decision 4). Its order is load-bearing (hearing ties to the earlier noise).

## Event ring
- What a device kernel emits is a RECORD of exactly `field_ab_harness._event_record`'s fields plus the render-only direction (`event_of(record_of(ev))` is the host event bit for bit), in a per-env ring per tick, in the reference's EMISSION ORDER: a per-row kernel stages per row and ONE gather per env appends in row (= id) order, never the thread order; a serial per-env producer appends directly (`env_sink`).
- Fields, kinds, causes, damage types and claim channel codes are ONE Python table with generated offsets: a new kind, field or cause is a row there, `tools/gen_unit_layout.py --write`, commit, rebuild.
- CAPACITY: past `E_max` or a row's staging nothing is written, the env's dropped count rises and the kernel's Python face refuses the run naming kernel and env (`UnitEventOverflow`).
- A slot that emits is shadowed with `DeviceSlot(..., events=True)`; in the chained form the ring's records join `sim.tick_events`, so the digest stays on the reference's path. ExplosionEvent joins with its producer (P7c).

## Request list
- What a device kernel does that is NOT a row write (a grenade leaving the hand, a charge laid, a door's `want_open`) is a REQUEST RECORD carrying every value the reference computes at that point, per env per tick, until the tables that will hold them reach the device (P7b, P6b).
- It is the Event ring's class generalised (`UnitRecordList`): ONE staging, ONE gather per env in row order, never a second gather, so the reference's order holds. An order-dependent decision over the list (which plant a full charge table refuses) is ONE thread per env walking it in list order after the gather.
- Fields and kinds are one Python table with a generated, fingerprint-checked header: a new kind or field is a row, regenerate, commit, rebuild. Capacity as the ring's (`UnitRequestOverflow`).
- Shadowed with `DeviceSlot(..., requests=True)`; the chained form applies the list through `apply_to_host` (each plant still offered to `charges.place`, which must agree).

## Object tables
- A host list of world objects a device slot reads or writes (the planted charges; the grenades and rounds in flight, `gmap.expansions`, each entity class as their patches land) is ONE table on `src/simulation/object_tables.py`: an `ObjectTable` schema in `unit_schema.OBJECT_TABLES` (columns with encode AND decode, capacity, live column, constructor). Every table fact is in the schema, none in the module: a new table is a schema row, `tools/gen_unit_layout.py --write`, commit, rebuild `cpp/build_cuda`. Never a second table generator, buffer class or compaction.
- Row i is the i-th object of the host list (its order is every reference loop's order); rows past the count are padding and never read. An object reference becomes an id or a row index, a variable-length member a span into a side table.
- The device table IS a record list (`ObjectTable : UnitRecordList`): a serial per-env producer appends through `emit_record`, a per-row kernel through the list's staging and its ONE gather (row order), both after what the table holds. The only shrink is the stable compaction (rows whose live column is 0 dropped, the rest in order), which keeps the reference's "survivors, then this tick's appends".
- CAPACITY: past it nothing is written, the env's overflow count rises and the Python face refuses the run naming the table and the env (`ObjectTableOverflow`). A table's capacity is storage; a law's own bound (`charges.place`'s live-charge refusal) stays the law's.
- A table is held by its IDENTITY gate (`tests/cuda_object_table_check.py` for the charges: `unpack(pack(list))`, the round trip, the append and the compaction against the host's own list on live scenes, env i of N == env i alone).
- ACCEPTED GAP: the charge table's slot (9e(d'), `charges.tick`) and the plant request's append are P7b's; until then the table has no producer in a slot.
- THE ROUND TABLE (`unit_schema.ROUND_TABLE`, `Simulation.bullets`): its record function is `unit_kernels.rounds_of` (behind `shooting_door`) and its way back `unit_kernels.rounds_to_host`, which binds each round's shooter BY ITS ID; never a second encoding of a round. Every host round flies, so its live column reads 1; the device marks a finished round 0 and compacts. A round carrying physical cover or the facing defense (OnePhaseWEGO) is refused by its encoder by name: the device rounds are the continuous path's.
- A new attribute of `BulletInFlight` that crosses a tick is a column of the round table in the same patch, or the twin forgets it in flight (rounds are in no digest: the table's gates are the guard).

## Per-env serial applier
- THE device form of every order-dependent law that writes the map or another unit's row (§4.3, design 4.4c decision 1): ONE block per env, thread 0 walking that env's effect records in list order, each effect a device function (the Device topology seam for the planes; the Device unit library and damage pipeline for the rows; the env's ring through `env_sink`) writing in place and booking its per-env channels. A new order-dependent writer is an effect KIND here, never a second walker, a second list or a per-producer kernel.
- ONE kernel, several STAGES (`applier.STAGES`: today slot 9's topology walk and the RESOLVE stage, which is slot 4c's resolve AND slot 2''s landing: one law, one landing): a later slot that walks the list or its own tables is a stage of the same kernel, never a copy of it.
- THE EFFECT LIST is a record list (the Event ring's `UnitRecordList`): a per-row producer stages per row and the ONE gather appends per env in row order AFTER whatever the list holds (several producers continue one list before it is walked: 4, 4a, 4b in the reference's order); a serial producer appends directly; the host staging (`DeviceApplier.stage`) sets the list. Never a second gather.
- The record, kinds, kind fields, stages, books and planes are ONE Python table (`src/simulation/applier.py`) with a generated, fingerprint-checked header: a new kind, field, stage, book or plane is a row there, `tools/gen_unit_layout.py --write`, commit, rebuild `cpp/build_cuda`. C++ reads a kind's field by its `APPLIER_<KIND>_<FIELD>` column, never a bare slot.
- A host `SimultaneousEffects` effect becomes a record ONLY through `applier.record_of` (its world calls recognised by their function: `chew_wall` is CHEW, `execute_payload` is PAYLOAD). A new effect kind or world call in `simultaneous.py` / its producers needs its device kind and its `record_of` branch IN THE SAME PATCH, or the gates refuse it by name. FACE has no device kind (WEGO-only producers).
- ACCEPTED GAP: PAYLOAD is a record only; the applier refuses to apply it (`refused` book, `EffectRefused`) until P7c.1 gives it its function.
- Each book names the host quantity it twins (`gmap.<attr>` or `<dict>:<key>`); the shadow harness compares it with the host's delta over the slot and the chained harness lands it on the host (the solid book through `_solid_energy_book`, so the solver's sink moves too).
- The planes reach it as device addresses of `(N, ...)` arrays in the GameMap field's own dtype, the resident fields as `ResidentBatch.buffers` holds them; `applier.non_resident_planes()` names those P8b must bring onto the device.
- CAPACITY: past the list's `cap` (or a row's staging, `applier.ROW_CAP`) the first records are kept, the rest counted, and the Python face refuses the run by name (`EffectOverflow`); an operand past its proved range writes nothing for that effect and is refused (`ApplierFault`).
- The fan-out of an effect's tile set over the block's threads (a barrier between effects) arrives with its first user (P7c's deposits): the launch keeps the block shape for it, so it lands without moving the kernel.

## Device topology seam
- `cpp/src/cuda_topology.cuh` is `GameMap.on_tile_changed` and `destroy_wall` integer for integer. A change to either Python function (or to the gas energy seam rows they call) lands here IN THE SAME PATCH: the twin ledger does not enforce it (the corpus burns no wall, so the `burn_through` twin's traced files are `simulation.py` alone) and the topology gate (`tests/cuda_unit_topology_check.py`) is the guard.
- Every operand is read before the first write: an effect is all or nothing.
- Float planes are written as their float32 bit patterns from table rows; every per-material value is a Device tables row from the env's own `GameMap.materials`, never an argument.
- Host-only, no device form: `render_dirty`, `_sky_mask_dirty`, `_tile_serial` itself (its count is the `tiles_changed` book), the solid books' sink, the render-only Wall / DoorDestroyedEvent. A chained caller does them in its host side.
- `seal_tiles` / `unseal_tiles` are not device functions yet: they arrive with their only callers, the doors (P6b).
