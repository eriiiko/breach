---
paths:
  - "src/simulation/gamemap.py"
  - "src/simulation/field_edit.py"
  - "src/simulation/pump_system.py"
  - "src/simulation/vent_system.py"
  - "src/simulation/entities/vents.py"
  - "src/simulation/materials.py"
  - "src/simulation/gases.py"
  - "src/simulation/prop_system.py"
  - "src/simulation/*_fixed.py"
  - "cpp/src/bulk_transport.cpp"
  - "cpp/src/bulk_transport.h"
  - "cpp/src/cuda_bulk_transport.cu"
  - "cpp/src/cuda_bulk_transport.h"
  - "cpp/src/trace_decay.h"
  - "cpp/src/o2_pressure_factor.h"
  - "cpp/src/fire_simulation.cpp"
  - "cpp/src/combustion.cpp"
  - "cpp/src/eos_solver.cpp"
  - "cpp/src/cuda_kick_compression.cu"
  - "cpp/src/cuda_edit_queue.cu"
  - "cpp/src/fixed_point.h"
  - "cpp/src/cuda_fixedpoint_device.cuh"
---

# Fields and gas: the store, the writers, the currencies

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## GameMap
- `GameMap.destroy_wall` and `GameMap.on_tile_changed` have a device twin, `cpp/src/cuda_topology.cuh` (the Device topology seam, `.claude/rules/device-units.md`): a change to either, or to the gas energy primitives they call, lands there in the SAME patch. The twin ledger does not catch it (the corpus burns no wall); `tests/cuda_unit_topology_check.py` does.
- So do `seal_tiles` / `unseal_tiles` / `can_seal_tiles` (with `_seal_blockers`' refusals), `inject_gas_n`, `extract_gas_n(_vec)`, `transfer_gas_n` (with `_air_withdrawal` and `gas_proportional_split`), `inject_gas_n_vec` (#29 P6b.2) and `vent_system._aperture_blocked` (#29 P6b.1): a change to one lands in the seam in the SAME patch. The corpus seals nothing; `tests/cuda_unit_doors_check.py` (part PROBE) and `tests/cuda_unit_vents_check.py` (part PROBE) are the guard.

## FieldEdit
- The only way to write a continuous field: never `gmap.<field>[...] +=` inline.
- The `gas_energy` row is a DUAL WRITE (store + mirror, through the seam), vetoed by the accountable set. The `gas` policy REFUSES a conservative slice: bulk N is what `gas_energy` is denominated against (arc #54).
- INTEGER (#29 P3e): an edit's amount is quantized ONCE into its field's currency (`field_edit._CURRENCY`, or handed in as `amount_q`), every weight is an integer, the contribution is rounded ONCE; never a float between the door and the field. A policy dtype with no currency is refused.
- On the resident path the device edit queue (`cpp/src/cuda_edit_queue.cu`) IS the flush, never a second flush; the CPU flush stays THE oracle and the resident fallback (`edit_fallback_reason`).
- A new policy or mode lands in `FIELD_POLICY` AND in the device's ordinal and combine in the same patch (`EDIT_FIELD_ORDINALS` is refused unless equal to the lexicographic ranks).

## Gas pump primitives
- THE per-tick gas mass flux path for entity actuators (slot 9e sweep, integer-native): never a parallel flux path, never per-tick FieldEdits. Incident: vent design v1 nearly rebuilt it, 2026-08-23.
- Trace is priced by air (#88 P0): the extraction split runs over the bulk planes only and each trace plane leaves at the cell's `S/N_bulk`. The pump destroys what it takes, unbooked (ACCEPTED GAP, #88 G6, #98).
- Energy-aware (arc #54): the primitives carry `ΔN·T_abs` through the gas energy seam (extract at the cell's own `T_abs`, inject at the deposit's). The vent plenum ledger stays RELATIVE and converts here (`E = e + n·T_AMB`).
- A pump's `feed` is `"external"` (creates / destroys at its port) or `"transfer"` (#133: moves air through `GameMap.transfer_gas_n` and the shared `_air_withdrawal` rule, smoke riding along, energy at the donor's T; nothing minted or lost). An interior airlock's pumps are transfer pumps.

## Vent / duct system
- The plenum ledger is R3-counted (bulk pair + int64 raw-e), measured-delta booked, ENTITY_SECT-digested. Plenum bulk is circulating credit ONLY; a future reserve is a separate row pair.
- The sweep has a device twin, the applier's VENTS stage over `cpp/src/cuda_unit_vents.cuh` with the duct and vent tables (`.claude/rules/device-units.md`): a change to `_duct_sweep`, `VentRuntime.accrue`, a runtime attribute that crosses a tick, or a ledger's per-tick step lands there, in `unit_schema.DUCT_TABLE` / `VENT_TABLE` and in the device's proved ledger range in the SAME patch. `tests/cuda_unit_vents_check.py` is the guard.

## Trace transport
- A trace plane's TRANSPORT is ONLY the stage-3b ride on the bulk face flux (`price_face(dq, S, N)` at the donor) plus the once-per-tick Jacobi, magnitude-truncated face diffusion: never a semi-Lagrangian step, never a bespoke advection loop (#12, `docs/smoke_transport_design_2026-10-04.md`).
- No trace plane is clamped anywhere outside a reader, bar the counted int32 saturation (`TRACE_CELL_MAX`, `trace_add_sat`, #88) at every site that adds to a cell. Trace ADD edits are additive up to that saturation and never cut the tile.
- Every loss is booked (the vent's trace slots of `boundary_flux_`, wipe, sink, decay, clip) and every gain (`trace_mint_sum`, `trace_deposit_books`); other movers conserve or book through those. Pump extraction, FieldEdit REMOVE and the explosion's smoke clear destroy trace UNBOOKED (ACCEPTED GAP, #88 G6, #98).
- Decay rounds to nearest with a floor of one count (`trace_decay.h::lost`). A new gas is a `[gases.*]` row.

## Trace currency
- Trace planes are int32 in the TRACE currency, `2^(16+TRACE_FINE_BITS)` counts per real unit (#88); the bulk planes in the same `gas` array stay Q16, and Q16 fractions and temperatures have their own `q16_*` names.
- Every trace read or write converts ONCE through the kit (`trace_from_bulk`, `trace_density_shr`, `quantize_trace`, `trace_add_sat`) or `gas_fixed.trace_*`: never an inline shift, never a bulk name. Sums over a trace plane accumulate in int64; coefficients stay Q16 per real unit; no sum mixes the two currencies.
- The four twins (C++, `gas_fixed`, `sweep_ref_q`, the digest spec's `trace_fine_bits`) are held equal by `tests/test_trace_currency.py`. A new trace reader is registered in `tests/test_trace_reader_invariance.py`.

## O2 pressure factor
- One FP_HD `factor` read by BOTH law sites (the fire ODE's `o2f`, combustion's claim gate `o2f_j`) on both backends, multiplying the mole-fraction law (#7).
- Its input is the materialized `gmap.atmosphere` over the same air the mole fraction reads: never density (hot air is thin, not low-pressure: the July density-trap fix), never a re-derived p*. A new O2 read goes through it.

## Interior drag
- Momentum drag lives only in the staged drag block: extend its stages, never add a parallel damping site. The `dyn_wave_absorb` chain and the B3c space-sponge band are separately scoped neighbours, not under this rule.
- Drag L/Q deposits its removed KE as heat through the DERIVED `k_ke` (arc #54 §2.1); the absorb, sponge and cap stages EXPORT or DESTROY theirs, counted (D6). Never a heat dial.

## Fixed-point kits
- The only sim arithmetic, CPU and device: never re-derive a shift, a round or a reciprocal.
- The match path's Python integer helpers live in `src/simulation/unit_fixed.py`, each the exact real result rounded once. Their device twins are `FP_HD` in `cpp/src/fixed_point.h`, bound as `bp.fp_*`, held equal to Python by `tests/test_unit_impulse_twins.py` (the falloffs by `tests/test_unit_kit_twins.py`). A twin refuses outside the range it was proved for (the falloffs return -1) and its device caller FAULTS; a change to a helper lands in Python first, then its twin, in one patch.
- Kit code is a rounding or a root, never a law (a law lives in the Device unit library or its slot's TU).

## Material / gas tables
- New material or gas = a table row, never a hardcoded id or a per-material if.
- Material ids are unbounded rows up to int8's 127; SPACE is `SPACE_CODE = -1`, outside the id range (level format v3; Erik, 2026-10-29; `tools/migrate_tilemap_v3.py`).
- Row VALUES are placeholders (Erik, 2026-09-22). The thin-rows rulings ("no massive object ever burns"; the 6 mm lumped door, `THIN_LIMIT_M` / `LUMPED_CRITERION_EXEMPT`) are PROVISIONAL until his material design session (`docs/thin_material_rows_design_2026-09-20.md`).
