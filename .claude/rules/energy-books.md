---
paths:
  - "cpp/src/eos_solver.cpp"
  - "cpp/src/eos_solver.h"
  - "cpp/src/temperature_solver.cpp"
  - "cpp/src/temperature_solver.h"
  - "cpp/src/combustion.cpp"
  - "cpp/src/combustion.h"
  - "cpp/src/gas_energy.h"
  - "cpp/src/cuda_temperature.cu"
  - "cpp/src/cuda_combustion.cu"
  - "cpp/src/cuda_eos_step.cu"
  - "cpp/src/cuda_eos_resident.cu"
  - "src/simulation/gamemap.py"
---

# Energy books: gas energy, solid energy, the closure identity (arc #54)

Fires on contact with these files. CLAUDE.md's index says what each system is
and who must use it; this file holds the constraints the code cannot say.

## Gas energy field
- THE conserved truth for gas thermal energy: int64, the exact unshifted `N_raw·T_abs_raw` on the one canonical accountable set, and the cross-tick truth.
- `refresh_gas_energy` is the LEVEL-LOAD INITIALISER only (`reseed_gas_energy(sel)` for a scenario builder that just wrote bulk `gas` directly). Re-deriving it mid-run DESTROYS state: `N·floordiv(E,N) ≤ E` drains up to N−1 raw counts per cell, plus any sub-count energy a seam wrote.

## Gas energy seam
- EVERY change of gas N or gas heat goes through the seam: MOVED mass carries its source's `T_abs`, MINTED mass (no gas donor) is born at ambient.
- The seam's device structural rows (the saturating mirror, retire, born at ambient) live in `cpp/src/gas_energy.h`, shared by every device caller (the device edit queue, the Device topology seam): never a private copy in a TU.
- Each call BOOKS its net effect into a named `gas_energy_books` channel (`diag_*` = energy crossing the books' boundary, excluded from `gas_energy_seam_net()`). A writer that skips the seam is invisible to the closure identity, which is the whole failure mode arc #54 exists to close.

## Gas temperature is a mirror
- NOTHING writes a gas cell's `temperature` directly, tests, tools and benches included. The once-per-tick recovery derives it with the rails; a seam write refreshes the MIRROR ONLY (no rails, no write-back into E).
- A bare `temperature[...] =` moves no books, so it does not fail loudly: it goes VACUOUS. Incident: three test fixtures at P-G1b, one of which had stopped driving any wind.

## Face-flux energy step
- Energy exchange between gas cells is a per-face flux evaluated ONCE in canonical orientation (`i` = lower linear index) and applied with opposite signs to its two cells, never a per-cell source term: that is what makes `Σ_region ΔE ≡ 0` exact in int64.
- Wall AND thermal_solid faces carry no face; vacuum and ambient-ring faces export to a counter.

## Gas energy recovery
- The ONE place `temperature` is derived from `gas_energy` with RAILS, once per tick over the whole accountable set. Rails write `gas_energy` back ONLY when one binds (`e_rail_sum`): an unconditional write-back drains N−1 raw per cell per tick.

## Energy closure identity
- Across whole ticks in int64: `Δ Σ_accountable gas_energy == EOS + thermal-solver gas side + combustion + seams`. A new writer adds a counter AND a term in ONE of those four groups. There is no fifth, bar the one boundary export every gate already carries, `−PhysicsEngine.e_water_evac_export_sum` (water evaporation, arc #54 §2.7).
- Radiation into gas is the thermal-solver group's `e_gas_deposit_sum`. Radiative energy that does not land is COUNTED on the boundary and is in no group, because it never touched `gas_energy` and a term would break the identity by its own amount: `e_rad_clamp_drop_sum`, `e_rad_boundary_export_sum`, `e_rad_floor_drop_sum`, `e_dep_floor_drop_sum`.
- The ledger spans solids too: `TemperatureSolver.solid_energy_books_sum` closes against `e_solid_deposit_sum + e_solid_cond_sum + CombustionSolver.e_comb_solid_heat_sum + e_solid_low_rail_sum + e_solid_seam_sum` (#73). A new writer of thermal-solid membership books through `GameMap._solid_energy_book` (`TemperatureSolver.book_solid_seam` moves the snapshot too, so the identity closes on the tick a slot-9 seam fires).
- Nothing relaxes to ambient: the thermostat (Pass 3, `cool_shift`) is DELETED and its term REMOVED, not zeroed (gated by `tests/test_thermostat_books.py`).
- Gates: `tests/test_e1_hot_rail.py::test_no_transport_mint`, `tests/test_gas_energy_tile_flip.py`, and the §6 benches (`tests/_sealedbox_bisect_bench.py`, `tests/_quiet_books_bench.py`, and the fire / vent / blast benches).

## Temperature solver
- Heat deposits are Q16.16 saturating adds. SOLIDS' temperature is derived here alone (bar combustion's booked object-site deposit); GAS temperature is the energy field's mirror. `gas_energy == nullptr` keeps the pre-#54 T-form law bit-identical for direct-binding callers.
- The Pass-1 maximum-principle clamp lives HERE, between the saturating add and the rails, on CPU and CUDA, fed the sweep's `rad_fluence` and `engine.emissive`'s table (`nullptr` for either = no clamp). Its ceiling is `e_ceiling_q` at all four sites, one bucket of headroom (Erik, 2026-09-24): never `e_inv_q`.
- The radiation fold's GAS branch (`sweep_ref_q.fold_pass1_gas`): an accountable gas cell's `rad_net` goes through the staged `deposit_dT_wide_i64` chain and lands as `dE = N·(T_target − T_before)` through `deposit_railed`, booked in `e_gas_deposit_sum`. Never a `temperature[i] =`, never a scaled dE, never the design's letter `N·(T_target + t_amb) − E` (it drains the residual `E mod N`); no `min(N, N_AMB)/N_AMB` factor (the density law is in `a_gas`); a cell with `rad_net == 0` is not touched.
- Nothing leaves the radiative boundary uncounted, so `Σ rad_net − Σ landed − the three boundary counters` is the conversions' declared rounding, gated per tick by `tests/test_temperature_gas_radiation.py`.
- `n_floor_heat ≥ 0.01` is LOAD-BEARING for gas radiative stability (at ≤ 0.005 even the Fleck-damped gas step undershoots below ambient): never lower it without re-running the reference's gates 14 and 15.
