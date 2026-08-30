# Gas-energy conservation arc #54 — P-G5: the solid-side / thermostat ledger

**Erik's ruling (2026-08-30)**: walls decaying to ambient (`cool_shift`,
`TemperatureSolver::step` Pass 3, solids only) is a deliberate modelling
boundary — "the ship's heating system, not simulated further" — and it is a
**two-way thermostat**: a wall pinned near ambient also warms a sub-ambient
wall back up through the same relax law. The sealed-box bench's remaining
+5-ish game-deg over the old ±2 tolerance (design
`gas_energy_conservation_design_2026-08-29.md` — the seal event's cold
boundary shell, warmed back up by ambient-held walls) is the thermostat doing
its job, **to be booked, not fixed**.

This patch (P-G5) is counters ONLY — no physics changed, every field
trajectory stays byte-identical to the pre-patch base commit (verified via
`field_ab_harness.capture_trajectory` at tol 0; see
`tests/test_thermostat_books.py::test_thermostat_books_byte_identical_to_base`
and its fixture `tests/_pg5_base_trajectory_45050f3.pkl`, captured on
`45050f3`, the commit immediately before this patch).

## What P-G5 adds

Arc #54's existing identity covers **gas books only**
(`Σ_accountable gas_energy`); solids sat outside it entirely. P-G5 adds the
solid side's own books and the counters that close them:

- `TemperatureSolver.solid_energy_books_sum` — a **snapshot** (not an
  accumulator), Σ over `thermal_solid` cells of `thermal_mass_raw · T_raw`
  (ambient-relative — `temperature` already is), refreshed at the end of
  every `step()`. The solid-side twin of `GameMap.gas_energy`.
- `TemperatureSolver.e_solid_deposit_sum` — Pass 1's landing on thermal
  solids (the radiation fold + the heat→T bit-shift deposit), priced as the
  cell's ACTUAL applied ΔT (post every rail that pass applies) × its real
  capacity. Booking the applied ΔT rather than a pre-rail quantity means a
  T_MAX_PHYS/low-rail engagement is folded in automatically — no separate
  solid-rail counter is needed.
- `TemperatureSolver.e_solid_cond_sum` — Pass 2's landing on thermal solids:
  the same per-cell `dT · cap_real` already feeding `e_cond_trunc_sum` /
  `e_cond_cap_sum`, read a third way. Whether the energy crossed from another
  solid (nets to ~0 over the whole solid set, up to those already-counted
  residuals) or from an accountable gas cell (the real cross-book transfer)
  is not distinguished — the total is exactly this tick's change to
  `solid_energy_books_sum` from conduction either way.
- `TemperatureSolver.e_thermostat_sum` — Pass 3's relax-to-ambient on thermal
  solids, SIGNED, **positive = energy entering the sim from the thermostat**
  (a sub-ambient wall being warmed back up). Bit-for-bit the same quantity
  the pre-existing `e_cool_sum` already accumulated (same formula, same
  site) — both are kept: `e_cool_sum` is the P-E2a "open channel" name,
  `e_thermostat_sum` is the canonical name this closure identity and the
  benches use.
- `CombustionSolver.e_comb_solid_heat_sum` — discovered closing the identity,
  not part of the original ask: `combustion.cpp`'s `object_site` branch
  (a burning crate/furniture fuel cell) writes `temperature[s]` **directly**,
  bypassing `heat[]` and `TemperatureSolver::step` entirely. Without a
  counter for this write, `tests/_sealedbox_bisect_bench.py`'s TOTAL ledger
  was BROKEN on every tick with live fire (verified empirically — see
  Results below). Priced the same way as `e_solid_deposit_sum`.

The **solid-side closure identity**, exact in int64 every tick:

    Δ solid_energy_books_sum ==
        e_solid_deposit_sum + e_solid_cond_sum + e_thermostat_sum
        + CombustionSolver.e_comb_solid_heat_sum

and therefore the **TOTAL ledger**, combining arc #54's existing gas
identity with this one:

    Δ(gas books + solid books) ==
        EOS + thermal-solver-gas-side + combustion + seams + water-evac
        + solid-deposit + solid-cond + thermostat + comb-solid-heat

closes exactly — every external channel named once, in one of the two
groups.

## CUDA twin

`cuda_temperature.cu` grows `TEMPERATURE_ENERGY_SLOTS` 10 → 13 (slots 10-12:
`e_solid_deposit_sum`, `e_solid_cond_sum`, `e_thermostat_sum`), booked at the
same three sites as the CPU (`temp_convert_unified`, `temp_conduct`,
`temp_cool`). `solid_energy_books_sum` is computed **on the host**, after the
final D2H `temperature` copy, from the same `conduction::cell_capacity_q`
kit the device capacity build uses — no new device reduction needed.
`cuda_combustion.cu`'s `object_site` branch does **not** get the
`e_comb_solid_heat_sum` twin: that kernel predates the gas-energy seam
entirely (no `gas_energy` parameter at all) and is not wired to any live
backend flag (`set_combustion_backend` is not in `run_on_cuda.py`'s
`_BACKEND_SETTERS`), so it is out of scope here — a pre-existing gap, not a
regression.

Verified bit-identical at tol 0 via `tests/_pg2_ab_probe.py` on `playground`,
`planetside_demo`, `bench_two_room` (42 counters compared per tick,
auto-discovered by name from `runner.eos` / `runner.engine.temperature` —
the new fields joined automatically, no probe edit needed).

## Results

- `tests/test_thermostat_books.py` (new): a small sealed hull room (no
  breach, no fire), gas seeded to +300 game-deg, 200 ticks. Total ledger
  closes exactly every tick; `e_thermostat_sum` is negative (the room can
  only shed heat to the thermostat here) and the room's total (gas+solid)
  energy decays monotonically toward ambient.
- `tests/_quiet_books_bench.py` (playground, 60 s, no forcing): TOTAL ledger
  EXACT.
- `tests/_fire_bench.py` (crate fire, 10 s): TOTAL ledger EXACT.
- `tests/_sealedbox_bisect_bench.py` (crate fire + sealed glass box, 18 s):
  TOTAL ledger EXACT on `baseline`, `drag`, `no_vrail`, `wall2`, `wall3`,
  `nofire`, `nofire_nocond`. The restated gate (ii) — `ΔT_box` minus the
  box's OWN sealed-wall-ring thermostat/conduction contribution (computed
  directly from the box's wall mask, since `e_thermostat_sum` is a
  whole-map total dominated by the crate fire's own nearby walls and is not
  a valid subtraction term for a box-local headline) — narrows the residual
  but does not fully close it under ±2 for `baseline` (adjusted ≈ +4.6 vs
  raw +5.6); `wall3` (thicker glass) does close under tolerance. See
  "Open question" below. `comp_work` (`adiabatic_index = 1.0`) errors with a
  pre-existing, unrelated `k_flux_q out of range` guard; `flat_gs`
  (`use_multigrid = False`, already documented as a pathological control
  that "blows up") shows the TOTAL ledger BROKEN on 2/432 ticks at u_max ≈
  980 m/s — not chased further here; every other variant tested is exact.

## Open question for Erik

The exact "thermostat contribution to the box" figure used in gate (ii)'s
restatement is a judgment call, flagged in-line in
`tests/_sealedbox_bisect_bench.py`: it is computed as the box's own sealed
glass ring's `Σ thermal_mass_raw · T_raw` delta over the run (a direct
Python computation over the box's wall mask, not a new C++ counter), which
conflates the ring's OWN thermostat relaxation with conduction it receives
from the arena gas outside the box. A per-region C++ counter would let the
two be split cleanly, if the residual is worth chasing further.
