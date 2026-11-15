---
paths:
  - "cpp/src/cuda_unit_*"
  - "cpp/src/unit_*_layout.h"
  - "src/simulation/unit_store.py"
  - "src/simulation/unit_tables.py"
  - "src/simulation/unit_events.py"
  - "src/simulation/unit_requests.py"
  - "src/simulation/unit_kernels.py"
  - "tests/cuda_unit_*.py"
  - "tools/gen_unit_layout.py"
  - "src/simulation/applier.py"
  - "cpp/src/applier_layout.h"
  - "cpp/src/cuda_applier.*"
  - "cpp/src/cuda_topology*"
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

## Per-env serial applier
- THE device form of every order-dependent law that writes the map (§4.3): ONE block per env, thread 0 walking that env's effect records in list order, each effect a device function of the Device topology seam writing the env's planes in place and its per-env book channels. A new order-dependent map writer is an effect KIND here, never a second walker or a per-producer kernel.
- The record, kinds, books and planes are ONE Python table (`src/simulation/applier.py`) with a generated, fingerprint-checked header: a new kind, book or plane is a row there, `tools/gen_unit_layout.py --write`, commit, rebuild `cpp/build_cuda`.
- Each book names the host quantity it twins (`gmap.<attr>` or `<dict>:<key>`); the shadow harness compares it with the host's delta over the slot and the chained harness lands it on the host (the solid book through `_solid_energy_book`, so the solver's sink moves too).
- The planes reach it as device addresses of `(N, ...)` arrays in the GameMap field's own dtype, the resident fields as `ResidentBatch.buffers` holds them; `applier.non_resident_planes()` names those P8b must bring onto the device.
- CAPACITY: past the list's `cap` the first `cap` effects are applied, the rest counted, and the Python face refuses the run by name (`EffectOverflow`); an operand past its proved range writes nothing for that effect and is refused (`ApplierFault`).
- The fan-out of an effect's tile set over the block's threads (a barrier between effects) arrives with its first user (P7c's deposits): the launch keeps the block shape for it, so it lands without moving the kernel.

## Device topology seam
- `cpp/src/cuda_topology.cuh` is `GameMap.on_tile_changed` and `destroy_wall` integer for integer. A change to either Python function (or to the gas energy seam rows they call) lands here IN THE SAME PATCH: the twin ledger does not enforce it (the corpus burns no wall, so the `burn_through` twin's traced files are `simulation.py` alone) and the topology gate (`tests/cuda_unit_topology_check.py`) is the guard.
- Every operand is read before the first write: an effect is all or nothing.
- Float planes are written as their float32 bit patterns from table rows; every per-material value is a Device tables row from the env's own `GameMap.materials`, never an argument.
- Host-only, no device form: `render_dirty`, `_sky_mask_dirty`, `_tile_serial` itself (its count is the `tiles_changed` book), the solid books' sink, the render-only Wall / DoorDestroyedEvent. A chained caller does them in its host side.
- `seal_tiles` / `unseal_tiles` are not device functions yet: they arrive with their only callers, the doors (P6b).
