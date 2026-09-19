#pragma once
// ============================================================================
// RadiationSweep — THE radiative transport path (ray-engine-v2 P1).
// docs/ray_engine_v2_design_v3_2026-09-15.md §2.3–2.8; the executable spec is
// docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py, which this class
// reproduces BIT FOR BIT (tests/test_radiation_sweep_reference.py, gate 0).
// ============================================================================
//
// Radiation stops being a per-emitter ray cast and becomes a field solve: ONE
// directional sweep visits every cell once per ordinate (S16, half-offset —
// S12 exists for the non-dyadic gate) and carries the heat channel in exact
// int64 arithmetic. The transport step is a PARAMETER (shear for heat, step
// for light at P6), never a fork; a new payload is a channel on this sweep,
// never a second traversal. Written in GATHER form: each cell reads its two
// upwind neighbours' stored outflow and recomputes the split from the same
// integers, so the CUDA twin (P4) needs no atomics inside an ordinate and is
// the same formulation by construction.
//
// Credits (the two published techniques this file implements):
//   * J.A. Fleck Jr., J.D. Cummings, "An implicit Monte Carlo scheme for
//     calculating time and frequency dependent nonlinear radiation transport",
//     J. Comput. Phys. 8 (1971) 313–342 — the emission damping factor
//     f = 1/(1 + α·g), here in its exact-integer excess form (§2.8), with
//     α = max(0, 1 − 1/g) collapsed into ONE kit floor-division:
//     f = T_abs / max(T_abs, 4L)  (Q24). The FLOOR IS 0, not Fleck's own ½
//     (design v3 row 39, RULED by Erik 2026-09-16, §2.8's RULED paragraph):
//     Fleck's α ≥ ½ is the bound for IMC's linearised equations with
//     effective scattering, while for OUR material update the monotone
//     condition is that x = g(1 + αg/4)/(1 + αg)² stays in (0, 1], which
//     α = max(0, 1 − 1/g) satisfies for every g. So nothing is damped where
//     the explicit update is provably stable (g ≤ 1) and a burning tile held
//     at its plateau by combustion radiates full black body there.
//   * A.D. Davis et al., "Discrete ordinates transport on structured grids",
//     2012 — the shear-vs-step transport trade (design §2.4). The discrete
//     ordinates (S_N) method itself: Chandrasekhar, "Radiative Transfer", 1950.
//   Papers under docs/papers/ (README_ray_engine_v2_2026-09-13.md).
//
// AT P1 THIS IS A SHADOW COMPUTATION: PhysicsEngine::step_tail runs it every
// tick (step 2b) into the four int64 shadow planes rad_net_sweep /
// rad_flux_sweep / rad_amb_sweep / rad_fluence, which nothing consumes yet;
// the old ray cast keeps feeding the fold until P3 flips it.

#include <cstdint>
#include <vector>

#include "fixed_point.h"
#include "emissive_table.h"

// One ordinate's checked-in constants (design §2.3, critique 3 §5b): the
// sign of its x/y direction cosines, whether |mu| >= |eta| (which of the two
// upwind reads is "straight"), and s_m — the MAJOR share of the downwind
// split in Q16 (shear: ONE − |minor/major|; step: |mu|/(|mu|+|eta|)). Integer
// LITERALS in radiation_sweep.cpp, never a libm cos/sin at load;
// tests/test_radiation_sweep_constants.py recomputes them within one count.
struct OrdinateConst {
    int8_t  sx;        // +1 / -1
    int8_t  sy;        // +1 / -1
    uint8_t x_major;   // 1: |mu| >= |eta| (shear only; step always reads x-side + y-side)
    int32_t s_m;       // the major share, Q16
};

// ---- the Fleck pre-pass primitives (FP_HD: P4's twin shares them) ---------
// L_q for a THERMAL SOLID: the cell's free excess-emission loss this tick in
// Q16.16 temperature, shr_round0((a·(E°[T] − E°[0])) >> 16, heat_inv_shift).
// `ex` is the int64 excess E°[T] − E°[0] (>= 0), `a_q` the material extinction.
FP_HD inline int64_t fleck_L_solid_q(int64_t ex, int32_t a_q, int his) {
    return fixedpoint::shr_round0_i64(((int64_t)a_q * ex) >> fixedpoint::FP_SHIFT, his);
}

// f_q24 = floordiv_q(T_abs_q << 24, max(T_abs_q, 4·L_q)) — design §2.8 +
// rows 32 and 39. α = max(0, 1 − 1/g) is never computed: 1 + α·g ==
// max(1, g). T_abs_q > 0 and D >= T_abs_q, so 0 < f <= 2^24, and f == 2^24
// EXACTLY when 4·L_q <= T_abs_q — the whole explicitly-stable range g <= 1
// is undamped (row 39; the superseded floor of ½ put T_abs + 2L in the
// denominator and damped at every L > 0). ONE exact kit division per cell
// per tick, door 1, identical on every backend.
FP_HD inline int32_t fleck_f_q24(int64_t T_abs_q, int64_t L_q) {
    const int64_t d2 = L_q << 2;                      // 4·L_q
    const int64_t D  = (T_abs_q > d2) ? T_abs_q : d2;  // floor 0 (design row 39)
    return (int32_t)fixedpoint::floordiv_q(T_abs_q << 24, D);
}

class RadiationSweep {
public:
    static constexpr int TRANSPORT_STEP  = 0;   // design §2.4: θ = 0
    static constexpr int TRANSPORT_SHEAR = 1;   //              θ = 1 (heat)
    static constexpr int F_SHIFT = 24;          // the Fleck factor's fixed point (row 32)
    static constexpr int32_t F_ONE = 1 << F_SHIFT;

    // The checked-in table for (n_ordinates, transport); nullptr if the pair
    // is not one of {12, 16} x {step, shear}.
    static const OrdinateConst* ordinate_table(int n_ordinates, int transport);

    // One tick of the sweep over all ordinates. run() OVERWRITES the four
    // output planes: it zeroes them itself before the first ordinate (the
    // per-ordinate books are `+=`), so they hold the LAST run's values until
    // the next run — the tile inspector reads them at render time, after the
    // tick has ended (design v3 row 38; the conductor's four `fill(0)` lines
    // died at P2a). A caller needs no wipe, and an ingress-rejected scene
    // leaves the planes untouched.
    //   temperature      : int32 Q16.16 (h, w)
    //   heat_atten_q     : int32 Q16 (h, w) — a_i, the material extinction
    //   dyn_heat_atten_q : int32 Q16 (h, w) — d_i >= a_i, material + stamped bodies
    //   heat_inv_shift   : int32 (h, w) — log2(thermal_mass), the solid capacity
    //   thermal_solid    : bool (h, w) — which cells take the solid Fleck branch
    //   e_table          : the E° table (E_TABLE_SIZE int64 entries)
    //   amb_level        : int64 (h, w) — THE PER-CELL AMBIENT LEVEL (thermal
    //                      model v2 R3), an EMISSIVE level in e_table's own
    //                      units, invariant 0 <= amb_level[i] <= e_table[0].
    //                      This is what the sweep RADIATES at, and the four
    //                      ambient-derived terms at cell i (its emission floor,
    //                      its ceiling's return `ret`, its body's re-emission,
    //                      and the virtual ring it reads) all take THIS cell's
    //                      value. Derived, never authored: derive_ambient()
    //                      below builds it from vacuum/interior state.
    //                      NOT a temperature — e_bucket_of floors at bucket 0,
    //                      so no temperature can express an ambient below
    //                      E°[0], and "0 K outside the hull" is exactly that.
    //   t_amb_q          : the absolute-temperature offset, Q16.16 (293 game)
    //                      — the SCALE's zero point, feeding only the Fleck
    //                      denominator. A global scalar, shared with the arc
    //                      #54 seam, and NOT the ambient the sweep radiates at
    //                      (thermal v2 §3.3 / critique L2-B1).
    //   k_leak_q         : the uniform out-of-plane leak coefficient, Q16 in [0, ONE]
    //   transport        : TRANSPORT_STEP or TRANSPORT_SHEAR
    //   n_ordinates      : 16 (S16) or 12 (S12)
    //   rad_net / rad_flux / rad_amb / rad_fluence : int64 (h, w), OVERWRITTEN
    //                      (zeroed here, then accumulated over the ordinates)
    //   fleck_enabled    : true on the live path (the engine ALWAYS damps);
    //                      false sets f = 2^24 on every cell — the reference's
    //                      `f_plane=None` configuration, which its isotropy and
    //                      float-agreement gates are measured in. A per-call
    //                      knob for the gates, never hidden state.
    // Throws std::invalid_argument on an unsupported (n_ordinates, transport),
    // a k_leak_q outside [0, ONE], a null amb_level or one outside
    // [0, e_table[0]], or a cell violating 0 <= a <= d <= ONE (the ingress
    // invariants the materials door enforces; re-checked here so a direct
    // caller cannot measure an illegal scene).
    void run(const int32_t* temperature,
             const int32_t* heat_atten_q, const int32_t* dyn_heat_atten_q,
             const int32_t* heat_inv_shift, const bool* thermal_solid,
             const int64_t* e_table, const int64_t* amb_level,
             int32_t t_amb_q, int32_t k_leak_q,
             int transport, int n_ordinates, int h, int w,
             int64_t* rad_net, int64_t* rad_flux, int64_t* rad_amb,
             int64_t* rad_fluence, bool fleck_enabled = true) const;

    // THE DERIVATION (thermal model v2 R3): the per-cell ambient level from
    // state the engine already has. A vacuum cell radiates against
    // `vac_level`; every other cell — interior air, solids, and the ambient
    // ring, which IS room-temperature air by definition — against e_table[0].
    // Nothing is authored: no material column, no level-data field. Fills and
    // returns the sweep's own scratch, valid until the next call.
    //   vac_level < 0  -> e_table[0], i.e. space at room temperature. That is
    //                     thermal v2 R4, what v1 ships, and it makes the plane
    //                     UNIFORM and the whole sweep bit-identical to the
    //                     scalar era.
    //   vac_level > e_table[0] -> std::invalid_argument (the ambient invariant;
    //                     a hotter-than-room ambient would make a cell's
    //                     excess negative, which the excess-form Fleck factor
    //                     and positivity both rest on being >= 0).
    // P4's CUDA twin does the same select per cell from the resident
    // `is_vacuum` plane and two scalars — no plane is uploaded.
    const int64_t* derive_ambient(const bool* is_vacuum, const int64_t* e_table,
                                  int64_t vac_level, int n) const;

    // The Fleck plane (Q24, (h, w)) computed by the last run() — observable so
    // gate 0 can compare the pre-pass, and the tile inspector can show it.
    const std::vector<int32_t>& fleck_plane() const { return f_q24_; }
    int last_h() const { return h_; }
    int last_w() const { return w_; }

    // Telemetry of the last run(): the smallest and largest per-ordinate
    // stream seen at any cell (gate 3 positivity is `min_stream >= 0`).
    mutable int64_t min_stream = 0;
    mutable int64_t max_stream = 0;

private:
    // Scratch, keyed by (n_ordinates, h, w), reallocated on change. The stored
    // outflow carries one plane per ordinate — the shape the concurrent CUDA
    // twin needs (design §8.2, row 29) — even though the CPU runs them one
    // after another.
    mutable std::vector<int64_t> outflow_;   // (n_ordinates, h, w)
    mutable std::vector<int64_t> ex_cell_;   // (h, w): E°[T_i] − amb_level[i]
    mutable std::vector<int64_t> amb_m_;     // (h, w): (amb_level[i] · w_m) >> 16
    mutable std::vector<int64_t> amb_derived_;  // (h, w): derive_ambient()'s output
    mutable std::vector<int32_t> f_q24_;     // (h, w): the Fleck factor, Q24
    mutable int h_ = 0, w_ = 0, n_ord_ = 0;
};
