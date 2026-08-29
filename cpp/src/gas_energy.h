#pragma once
// ===========================================================================
// THE GAS ENERGY SEAM — arc #54 (gas-energy conservation), design §2.7.
//
//   docs/gas_energy_conservation_design_2026-08-29.md
//
// `gas_energy` (GameMap.gas_energy, int64 per cell) is the CONSERVED TRUTH for
// gas thermal energy: the exact unshifted product `N_raw * T_abs_raw` (Q32, no
// >>16), defined on the ACCOUNTABLE SET and zero elsewhere. `temperature` is
// its MIRROR on gas cells (and stays the truth on thermal solids).
//
// Under a STORED energy field every writer of gas N is a writer of gas T, so
// every such writer goes through the two rules this header transcribes ONCE
// for every C++ caller (the Python-visible twin lives on GameMap —
// `gas_energy_move` / `gas_energy_deposit`):
//
//   * MOVED  — mass leaving cell i carries `dN * T_abs,i` out with it, and
//              arriving mass carries its DONOR's `T_abs` in.
//   * MINTED — mass with no gas donor (destroy_wall's seed, ambient / ring /
//              vacuum / thermal-solid inflow) is born at ambient, credited
//              `dN * T_AMB_K_raw` ABSOLUTE.
//
// CADENCE (design §2.6, F7). The FULL recovery — mirror refresh WITH the
// T_MIN / T_MAX_PHYS rails — runs exactly ONCE per tick, at the end of the EOS
// energy pass (eos_solver.cpp step 7). A SEAM write refreshes the MIRROR ONLY:
// no rails, no write-back. `deposit_railed` is the ONE documented exception —
// combustion runs AFTER that recovery (physics_runner slot order), so its
// deposit site has to carry the ceiling itself or the once-per-tick bound on
// stored E (design §2.2: E <= 2^60) and `t_max_phys_hits` both stop meaning
// anything (tests/test_air_boundary.py:820's `== 0` STOP).
//
// WHY THE MIRROR IS NEVER WRITTEN BACK INTO E. `N * floordiv(E, N) <= E`, so a
// refresh that also stored `N * T_mirror` would drain up to `N - 1` raw counts
// per cell per write — exactly the drip class this arc exists to kill. The
// mirror is a LOSSY READ of E, never a round trip.
// ===========================================================================

#include <cstdint>

#include "fixed_point.h"   // FP_HD, floordiv_q — the ONE arithmetic kit

namespace gas_energy {

// The one bulk floor, shared verbatim with eos_solver.cpp's recovery and
// bulk_transport.cpp's divide policy (design §2.6: ONE value, every file).
// Below it a cell has no thermodynamic capacity to divide by, so the mirror
// reads ambient rather than dividing by ~0.
constexpr int64_t N_EPS_RAW = 1;

// The ACCOUNTABLE SET (design §2.2) — the one canonical skip-set complement,
// mirroring `e_participates()` (bulk_transport.cpp) and
// `GameMap._gas_energy_accountable` (gamemap.py). `is_ambient` and
// `thermal_solid` are nullable in the same dormancy-by-branch idiom the rest
// of the engine uses (a null thermal mask falls back to `solid`, which is
// elementwise equal on a furniture-free map).
FP_HD inline bool accountable(const bool* solid, const bool* thermal_solid,
                              const bool* is_vacuum, const bool* is_ambient,
                              int i) {
    if (solid[i] || is_vacuum[i]) return false;
    if (thermal_solid != nullptr && thermal_solid[i]) return false;
    if (is_ambient != nullptr && is_ambient[i]) return false;
    return true;
}

// The mirror READ: T_rel = floordiv(E, N) - T_AMB_raw. A sub-N_EPS cell has no
// meaningful temperature (that is exactly why the recovery WIPES it), so it
// reads ambient rather than dividing.
FP_HD inline int32_t mirror_q(int64_t e, int64_t n_raw, int32_t t_amb_q) {
    if (n_raw < N_EPS_RAW) return 0;
    return (int32_t)(fixedpoint::floordiv_q(e, n_raw) - (int64_t)t_amb_q);
}

// A MOVED parcel's energy: `dN` counts of mass carrying the source's absolute
// temperature. Named rather than inlined so every call site reads as the rule.
FP_HD inline int64_t parcel(int64_t dn_raw, int64_t t_abs_src_raw) {
    return dn_raw * t_abs_src_raw;
}

// A MINTED parcel's energy: born at ambient, ABSOLUTE (the P-E1 "born carrying
// zero RELATIVE energy" rule in the absolute currency).
FP_HD inline int64_t minted(int64_t dn_raw, int32_t t_amb_q) {
    return dn_raw * (int64_t)t_amb_q;
}

// THE DEPOSIT (seam form): add `de` to cell i's stored energy and refresh its
// mirror. No rails — the once-per-tick recovery owns those.
FP_HD inline void deposit(int64_t* gas_energy, int32_t* temperature, int i,
                          int64_t de, int64_t n_raw, int32_t t_amb_q) {
    gas_energy[i] += de;
    temperature[i] = mirror_q(gas_energy[i], n_raw, t_amb_q);
}

// THE DEPOSIT, RAILED (design §2.7 combustion row): the same, plus the
// T_MAX_PHYS ceiling applied AT THE DEPOSIT SITE because this caller runs
// after the once-per-tick recovery. The rail writes E back — that is the ONE
// sanctioned write-back, and its delta is booked so the closure identity
// stays exact. `e_rail` / `hits` are nullable.
FP_HD inline void deposit_railed(int64_t* gas_energy, int32_t* temperature,
                                 int i, int64_t de, int64_t n_raw,
                                 int32_t t_amb_q, int32_t t_max_phys_q,
                                 int64_t* e_rail, int64_t* hits) {
    gas_energy[i] += de;
    int32_t t = mirror_q(gas_energy[i], n_raw, t_amb_q);
    if (t > t_max_phys_q && n_raw >= N_EPS_RAW) {
        t = t_max_phys_q;
        const int64_t e_new = n_raw * ((int64_t)t + (int64_t)t_amb_q);
        if (e_rail) *e_rail += e_new - gas_energy[i];
        gas_energy[i] = e_new;
        if (hits) ++(*hits);
    }
    temperature[i] = t;
}

}  // namespace gas_energy
