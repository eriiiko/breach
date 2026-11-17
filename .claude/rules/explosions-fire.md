---
paths:
  - "src/simulation/payloads.py"
  - "src/simulation/physics.py"
  - "src/simulation/blast_expansion.py"
  - "src/simulation/blast_air.py"
  - "src/simulation/timed_charge_system.py"
  - "src/simulation/heat_source_system.py"
  - "src/simulation/entities/charge.py"
  - "src/simulation/entities/heat_source.py"
  - "src/simulation/combat.py"
  - "src/simulation/weapons.py"
  - "src/debug_keys.py"
  - "cpp/src/fire_simulation.cpp"
  - "cpp/src/combustion.cpp"
  - "cpp/src/cuda_fire.cu"
---

# Explosions, payloads and fire

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Payload executor
- `execute_payload` has a device twin for EVERY row, `cpp/src/cuda_unit_payload.cuh` (the Payload function, `.claude/rules/device-units.md`) over `cpp/src/cuda_unit_blast.cuh` (the walk, the air read, the source): a change to it, `physics.apply_explosion` / `add_explosion_smoke`, `exchange.apply_blast_damage`, `flash_blind`, `emit_gas` / `ignite_ring` / `deposit_heat` / `deposit_over_tiles` / `linear_weights` / `blast_smoke_peak_q` / `inject_bulk_gas` or their claims, `blast_expansion` (the walk, the seeds, `start_source`, `start_expansion`, `add_source`, `take_source_plane`) or `blast_air` lands there in the SAME PATCH. The twin ledger does not catch it (the corpus detonates one frag in 500 ticks); `tests/cuda_unit_payloads_check.py` does.
- Every float such a sub-step converts per call is a door function of the law (`blast_expansion.source_total_q` / `expansion_r2`, `payloads.heat_base_q` / `soot_units_q`, `blast_air.disc_reach`), read ONCE by its Device tables row: a new per-call float is a new door function and row in the same patch, never an inline conversion.
- BULK gas (O2, inert N2) enters ONLY through the `bulk_gas` / `bulk_amount` / `bulk_radius` / `bulk_temp` columns (#133 E.2): `inject_bulk_gas` calls the pump primitive `inject_gas_n_vec` per open disc tile, booked in the seam channels `payload_bulk_gas` / `payload_bulk_gas_rail`. `emit_gas` stays the TRACE path and its door refuses a conservative gas.

## Generic explosion
- A `[payloads.*]` row with `strength > 0` is THE explosion (#31, `docs/explosion_design_2026-11-06.md`): expansion, heat and smoke derive from `[blast]`; the row keeps its explosive name and its direct damage (the disc's). A new explosive is a row, never a new blast shape.
- The row door (`weapons._strength_columns`) refuses `pressure`, `heat_*`, `explosive_kg`, `expansion_*`, `ignite_radius`, `ignite_intensity` by KEY PRESENCE and requires `explosive`, `emit_blast_smoke`, `radius > 0`.
- A blast PUSHES units only through its own wind (Wind load on a body): never a second blast impulse, so nothing is counted twice.
- The flash columns and the `blind_*` columns pass this door on any row; a row with a flash and nothing else is one render-only event. Rows without `strength` behave as they did.

## Divergence source
- `div(u) = S` entered by subtracting S from `div_u` before the multigrid build (Feldman, O'Brien & Arikan 2003), written only by the blast expansion module. It conserves N and E exactly: pair it with heat or the gas springs back.
- A dense int32 Q16 1/s plane, zeroed on consumption at the head of the physics step (a slot-9e blast leaves it for the next tick beside its queued edits); hashed while non-zero (a `DIGEST_SECTIONS` row, absence-transparent).
- A source accumulates in Python ints and saturates at `S_MAX_Q`, counted (`gmap.div_source_sat_hits`, `div_source_cut_q`), because the walk, which is level geometry, sets it. The subtraction is `sat_add_q16_i64` in all three twins (`cpp/src/eos_solver.cpp`, `cpp/src/cuda_eos_step.cu`, the resident launch).

## Blast reach
- 4-neighbour along the solve's own mask (never solid, vacuum or the ambient ring; passable fuel stays in it), squared distance under an integer `r2` fixed at the door, row-major; at detonation only, after the wall damage, so a tile the blast opened is open. A wall stops it, a gap does not, and the fixed totals land on it, so a corridor concentrates a blast.
- Seed rule: a centre still solid seeds from the approach side (`execute_payload(approach=)`), else from every open 4-neighbour. CLAUDE'S CALL; Erik may veto at a HUMAN-TEST.

## Fixed-total tile deposit
- One FieldEdit on one tile-list region, explicit integer weights over their sum (`weight_den`) and a total already in the field's currency (`total_q`), row-major; the flush rounds once per tile, so the total holds to n/2 LSB. Never a twin of `deposit_heat`, `_claim_disc` or `blast_smoke_peak` for a walked set; the claim is `_claim_tiles` on the same list.

## Starting a fire
- A blast has NO fire-starting rule (Erik, 2026-11-06, a core principle: situations fall out of the physics). Fuel catches only when the engine's ordinary heat takes it to its own `ignition_temp`; a strength row's heat is its fixed total, never raised, biased or special-cased on fuel.
- A fire is started by DELIVERING HEAT, never by writing `fire` alone: a flame on an ambient tile draws no O2, burns no fuel and does not go out (the O2 demand and fuel destruction ride the tile's temperature). Any new igniter (payload row, weapon, editor tool, test fixture, bench) seeds the tile AT OR ABOVE its own `ignition_temp` as well as lighting it. Prefer heat only and let `apply_temperature_ignition` light it where the one-tick delay is acceptable.
- Incidents, 2026-09: both incendiary ammo rows inert (#65), the dev key dead on arrival, five test fixtures silently measuring nothing.

## Timed charge
- A scripted or periodic detonation is a `timed_charge` entity row executing a `[payloads.*]` row through `execute_payload` on the sim clock: never a scheduler in `main.py`, a test-only hook, or a second payload path.
- Its synced state is the `ticks_to_fire` COUNTDOWN (steps, not `sim.tick`, which TwoPhaseWEGO rewinds each round), digested only where a charge exists.
- A stored hazard that goes off when its tile gets hot is `trigger = "heat"` (#133 E.2): `trigger_temp` quantized ONCE at load, an integer compare on its own tile's temperature MIRROR in the slot-9e sweep, a synced `fired` latch (its digest row written only for heat charges, so timer charges hash as before); it fires ONCE (`period_s` refused). Never a per-hazard check, never a reuse of `apply_temperature_ignition`. `tools/gen_ship.py` realises a room's `hazard` tag as one.
- `sweep_timed_charges` has a device twin, the applier's TIMED_CHARGES stage (the Timed charges and heat sources section of `.claude/rules/device-units.md`): a change to the sweep or a `TimedChargeRuntime` attribute that crosses a tick lands there in the SAME patch.

## Heat source
- `heat_per_s` quantized ONCE at build and landed through `payloads.deposit_heat`'s counted path (never a `temperature[...] =`); `max_temp` stops the deposit while its tile's MIRROR reads at or above it: never a thermostat.
- It dies ONLY from a blast: `execute_payload` appends every `radius > 0` detonation to `GameMap.blast_log` (a plain per-tick list, never digested, emptied by the 9e sweep); at hp 0 it runs `death_payload` once. `blast_log` is THE seam a slot-9e entity reads to react to blasts: never a second explosion hook (#133 E.2).
- `sweep_heat_sources` (the chain, `_blast_damage_q16`, `_die`, the deposit) has a device twin, the applier's HEAT_SOURCES stage, whose head refuses a chain past its proved bound by name: a change to the sweep, or one that lets a source append more than one disc, lands there in the SAME patch.
