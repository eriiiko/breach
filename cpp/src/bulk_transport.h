#pragma once
// Donor-cell conservative flux transport for the BULK gas species
// (EOS refactor P1 — docs/eos_refactor_design.md §2.2, decisions log #11).
//
// The two CONSERVATIVE bulk species (O2 / inert_N2, simulation/gases.py) move
// by first-order upwind donor-cell flux on the solver's wind field, using the
// SAME pattern (gather-once wide flux + per-cell outflow limiter) as
// WaterSolver::step's donor-cell block (water_solver.cpp) — every subtraction
// has a matching addition, so mass is conserved to the integer LSB and a
// sealed room's O2+N2 total never drifts (the P1 gate: exact conservation
// over 1000 ticks). This is P1: purely ADDITIVE, riding TODAY's wind
// (gmap.wind_x/wind_y, already computed by AtmosphereSolver::diffuse_solve
// earlier this tick) — no solver change, nothing yet consumes N_O2/N_N2.
//
// Non-conservative TRACE gas planes (gas_conservative[gi] == false) are left
// completely untouched here — they stay on the existing per-gas
// semi-Lagrangian loop (PhysicsEngine::run_substeps), which in turn SKIPS the
// conservative planes so the two transport schemes never both touch the same
// plane (see the run_substeps body).
#include <cstdint>

// gas               : (n_gases, h, w) contiguous Q16.16 density planes, mutated in place
// gas_conservative  : (n_gases,) — true for the bulk pair (O2 / inert_N2); every
//                     other (trace) plane is skipped entirely
// wind_x, wind_y    : (h, w) Q16.16 — the solver's cell-centred velocity-like
//                     field (AtmosphereSolver's wind, this tick's fresh value)
// solid             : (h, w) — the physics solid mask (permeability <= 0); a
//                     solid cell always holds N == 0 (enforced by the final clamp)
// is_vacuum         : (h, w) — true vacuum; N is zeroed there every tick (mass
//                     legitimately leaves the system at a breach — a deliberate
//                     sink, not a conservation bug)
// dyn_permeability  : (h, w) — the live per-tick face-permeability field; gates
//                     flux exactly like the smoke/atmosphere stencils
//                     (face = min(perm_self, perm_neighbor))
// h, w              : grid dimensions
// dt                : the FULL tick length (seconds, sim_time); dx == 1 tile
void bulk_flux_transport(
    int32_t* gas,
    const bool* gas_conservative,
    int n_gases,
    const int32_t* wind_x,
    const int32_t* wind_y,
    const bool* solid,
    const bool* is_vacuum,
    const float* dyn_permeability,
    int h, int w,
    float dt);
