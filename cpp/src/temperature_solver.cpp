// Temperature solver implementation — engine/06 §1 (heat -> temperature) +
// §2 (conduction relaxation). See temperature_solver.h for the determinism
// contract.

#include "temperature_solver.h"
#include "raycaster.h"     // HEAT_SCALE, heat_saturating_add (shared Q16.16 domain)
#include "fixed_point.h"   // S3c: quantize() for the o2_vacuum_thresh integer compare
#include "gas_energy.h"    // arc #54 P-G1b: THE gas energy seam (design §2.7)
#include "emissive_table.h" // ray-engine-v2 P1 / P5d: e_ceiling_q, the Pass-1 clamp's ceiling
#include <algorithm>        // P-E2b: std::clamp on the wide deposit-divide result

// Direction order for the per-tile face_shift cache (MUST match the Python
// bake in GameMap: index 0=N, 1=S, 2=E, 3=W).
namespace {
    constexpr int DIR_N = 0;
    constexpr int DIR_S = 1;
    constexpr int DIR_E = 2;
    constexpr int DIR_W = 3;
    // Row/col offset per direction, same order.
    constexpr int DY[4] = { -1, +1,  0,  0 };
    constexpr int DX[4] = {  0,  0, +1, -1 };

    // ------------------------------------------------------------------
    // P2 gas-T semi-Lagrangian advection helper (`gas_wall_at` +
    // `gas_backtrace_sample_q`, ~110 lines) — DELETED at P-E1 (energy-books
    // arc, design §2.1.1; round-1 finding L3-5). It served ONE caller: the
    // Pass-0b gas-T advection retired in step() below. That pass was the
    // engine's second semi-Lagrangian T-COPIER, i.e. the same free-energy
    // channel the EOS one was, dormant only because `step_tail` happens to
    // pass null winds. Gas temperature is now transported once, and
    // conservatively, by the EOS energy books. The MEDIUM-TEST SITE 2/6, 3/6
    // and 4/6 marks lived here and retire with it (sites 1/6, 5/6 and 6/6 —
    // the vacuum wipe, the Pass-1 medium branch and the Pass-2 conduction
    // medium — are untouched and still marked below). Deleted rather than
    // left dead so no future plumbing change can quietly re-adopt it; the
    // CUDA twin (`cuda_temperature.cu` temp_advect) is deleted identically.
    // ------------------------------------------------------------------
}

void TemperatureSolver::step(
    int32_t* temperature,
    const int32_t* heat,
    const int32_t* heat_inv_shift,
    const int32_t* face_shift,
    const bool* solid,
    const bool* is_vacuum,
    const int32_t* atmosphere,   // S3c: Q16.16 int32 (was float — the last float input)
    const int32_t* n_bulk,       // EOS P3: real bulk N_total (O2+N2 sum, Q16.16);
                                  // nullable — falls back to the P2 atmosphere
                                  // density-proxy (the direct-binding back-compat
                                  // path; the engine always passes the real sum)
    const int32_t* wind_x,       // P2: Q16.16 int32, may be null (Pass 0 skipped)
    const int32_t* wind_y,       // P2: Q16.16 int32, may be null (Pass 0 skipped)
    int h, int w,
    float dt,                    // P2: tick's elapsed seconds; <= 0 skips Pass 0
    const bool* is_ambient,      // BC: ambient ring mask (nullptr = space map)
    const bool* thermal_solid,   // thermal-mass axis: medium mask (nullptr -> solid)
    // T5b step 7 / R1: `cool_shift_grid` stood here. Pass 3 is deleted and the
    // parameter with it -- see the Pass-3 tombstone below.
    const int64_t* rad_net,         // P-R4: SIGNED radiation accumulator (int64 since P3a-1)
    int64_t* gas_energy,            // arc #54 P-G1b: the conserved gas energy
    int32_t t_amb_q,                // T_AMB_K raw (only read with gas_energy)
    const int64_t* rad_fluence,     // ray-engine-v2 P1: the sweep's Φ (clamp)
    const int64_t* e_table          // ray-engine-v2 P1: E° (clamp); both null = no clamp
) const {
    const int n = h * w;
    const bool ambient_mode = (is_ambient != nullptr);   // BC: dormancy by branch

    // --- THERMAL-MEDIUM mask (docs/thermal_mass_axis_design_2026-07-25.md) ---
    // `solid` (permeability <= 0) is a FLOW property; the per-medium thermal
    // branches must key on the THERMAL axis `thermal_solid` (thermal_mass > 0)
    // instead. Six sites below — and ONLY those six — use `ts`; every other
    // `solid`/`is_vacuum` meaning in this TU (vacuum structure, LoS, N==0) is
    // untouched. nullptr == "the caller has no thermal mask" (the documented
    // back-compat idiom this TU already uses for wind/n_bulk/is_ambient), and
    // falls back to `solid` — which is EXACTLY today's behaviour, and is also
    // elementwise equal to thermal_solid on any furniture-free map (addendum
    // D4), so the fallback is not a second code path in practice.
    const bool* ts = (thermal_solid != nullptr) ? thermal_solid : solid;

    // ---- arc #54 P-G1b: the gas-energy seam's two pass-entry facts --------
    // `e_on` is the ONE branch that selects the energy-form gas side; with it
    // false every line below is the pre-#54 T-form law, bit for bit (the
    // direct-binding / unit-test path). `acct` is the canonical accountable
    // set (design §2.2), read through the SAME predicate the EOS, the bulk
    // transport and GameMap._gas_energy_accountable all use — never re-derived
    // here. NOTE it keys on `ts`, i.e. it inherits this TU's documented
    // thermal-mask fallback, so a caller with no thermal mask sees the same
    // set the rest of its passes do.
    const bool e_on = (gas_energy != nullptr);
    auto acct = [&](int i) -> bool {
        return !solid[i] && !ts[i] && !is_vacuum[i]
            && !(ambient_mode && is_ambient[i]);
    };
    // The books' N: the SAME bulk sum the capacity build reads (and the same
    // O2+N2 Dalton total `gas_energy` was built from) — one source, no drift.
    auto n_books = [&](int i) -> int64_t {
        const int32_t nr = (n_bulk != nullptr) ? n_bulk[i] : atmosphere[i];
        return (nr > 0) ? (int64_t)nr : (int64_t)0;
    };

    // ---- P-E2a: the per-cell CAPACITY planes (design §2.3) -----------------
    // Built ONCE per step, ahead of every pass, because they depend only on
    // FROZEN inputs (the medium mask, N, and the two dials) — never on T. Three
    // passes read them: Pass 0a prices its wipes, Pass 2 IS the energy law, and
    // Pass 3 prices its cooling. `cap_used_` is the divisor of record;
    // `cap_real_` is the unfloored truth the counters are denominated in.
    //
    // `n_floor_q` is hoisted out of the Pass-1 block it used to live in so both
    // the deposit divide and the conduction capacity read ONE quantization of
    // the SAME dial (design §2.3: "use the same n_floor_heat dial the deposits
    // use"). The value is P-E2b's to move, not this patch's.
    // `c_v_q` is new: Pass 1 needs 1/c_v (a make_recip reciprocal), the capacity
    // needs c_v itself as a Q16.16 MULTIPLIER. Both are the same load-time dial.
    const int32_t n_floor_q = fixedpoint::quantize((double)n_floor_heat);
    const int32_t c_v_q     = fixedpoint::quantize(
        (c_v > 0.0f) ? (double)c_v : 1.0);
    cap_used_.resize(n);
    cap_real_.resize(n);
    no_cond_.resize(n);
    for (int i = 0; i < n; ++i) {
        // The SAME N source Pass 1's deposit divides by: the real bulk sum when
        // the engine supplies it, else the P2 atmosphere density proxy.
        const int32_t n_raw = (n_bulk != nullptr) ? n_bulk[i] : atmosphere[i];
        conduction::cell_capacity_q(ts[i], heat_inv_shift[i], n_raw, n_floor_q,
                                    c_v_q, &cap_used_[i], &cap_real_[i]);
        // T5b / ledger 7b: the VACUUM CONDUCTION MASK. A cell with no gas in
        // it is not a thermal medium and owns no conduction face — full
        // rationale (and why it is a mask, not a density law for kappa) in the
        // header at `no_cond_`. Built here because it reads `cap_real_`, which
        // this loop has just produced; all three terms are frozen inputs, so
        // the mask is as pass-invariant as the capacities are.
        no_cond_[i] = (uint8_t)(!ts[i] && (is_vacuum[i] || cap_real_[i] == 0));
    }

    // ---- Pass 0: gas-T zero-at-vacuum + semi-Lagrangian advection (P2, §4) ----
    // Structural invariant FIRST, unconditional: an OPEN (non-solid) vacuum
    // cell — a true breach — holds no gas, so it holds no gas-T either —
    // "energy leaves with the gas" (design §4). This also defends a cell that
    // just transitioned into an open breach carrying stale T from before the
    // mask flipped. GUARD: the `!ts[i]` test is load-bearing here — a THERMAL
    // SOLID cell that is ALSO flagged vacuum (the intact hull's own
    // space-exposure flag; gamemap.py: "an intact hull is vacuum AND solid") is
    // NOT a breach, it is a wall radiating to space through the SWEEP (T5b
    // step 7 deleted Pass 3's cool_shift_vacuum, which used to say this) —
    // its T is real solid-thermal-mass state and must survive across
    // ticks. Without this guard every space-facing hull tile would be wiped to
    // 0 before Pass 1 could deposit onto it, which is wrong (and was caught by
    // the sealed-room energy E2E's vacuum-exposed-hull scenario). MEDIUM-TEST
    // SITE 1/6: the guard is now the THERMAL medium, so a space-exposed crate
    // keeps its object temperature for the same reason a hull tile does; the
    // hull case is unchanged (hull is both solid and thermal_solid).
    for (int i = 0; i < n; ++i) {
        // BC (audit (b)): the ambient ring is an open (non-solid) boundary that
        // radiates to the T_amb sky — wiped to ΔT=0 exactly like a vacuum
        // breach. Branch-gated -> space maps byte-identical.
        if ((is_vacuum[i] || (ambient_mode && is_ambient[i])) && !ts[i]) {
            // P-E2a (L3-6): the LAW is unchanged — this is instrumentation.
            // Both wipes are SIGNED channels: they destroy the energy a breach
            // vents / the ring exports, and they CREATE energy whenever they
            // pin a sub-ambient (T < 0) cell up to 0. A vacuum cell that is
            // also flagged ambient is counted as vacuum (the test order below
            // matches the condition's own order — pinned so the two backends
            // and the ledger attribute it identically).
            const int32_t t_old = temperature[i];
            if (t_old != 0) {
                const int64_t de = -(int64_t)t_old * cap_real_[i];
                if (is_vacuum[i]) e_vac_wipe_sum += de;
                else              e_ring_pin_sum += de;
            }
            temperature[i] = 0;
            // arc #54 P-G1b (design §2.2): the wipe zeroes the stored energy
            // too. HYGIENE, NOT A BOOK: every cell this branch can reach is
            // vacuum or ring, i.e. OUTSIDE the accountable set, so its stored E
            // is already excluded from `Σ_accountable gas_energy` and zeroing
            // it moves no book. What CAN carry stale E here is a cell that only
            // just became a breach — and retiring THAT is the structural seam's
            // job (gamemap.on_tile_changed / destroy_wall, §2.7 row 8), where
            // the mask flip and its `e_retire_sum` booking happen together.
            // Zeroing here as well means no stale energy can ever survive a
            // mask flip that some future caller performs without the seam.
            if (e_on) gas_energy[i] = 0;
        }
    }

    // ---- Pass 0b: gas-T semi-Lagrangian advection — RETIRED (P-E1) ---------
    // Energy-books arc, design §2.1.1 (round-1 finding L3-5). This was the
    // SECOND semi-Lagrangian T-copier in the engine, and the same mint as the
    // EOS one: a T *copy* moves temperature onto mass it never paid for. It was
    // dormant in the live engine only by plumbing accident — `step_tail` passes
    // null winds (`physics_engine.cpp`), so wind was never supplied — and the
    // design's ruling is that "one plumbing change must not silently re-open
    // the mint". Gas temperature is now transported ONCE, conservatively, by
    // the EOS energy books (`bulk_flux_energy_transport_cached`).
    //
    // Retired by DELETION rather than by an assert: the CUDA twin
    // (`cuda_temperature.cu`) is retired identically, so both backends agree,
    // and a caller that still passes wind (the pybind back-compat surface keeps
    // the optional wind_x/wind_y args) now simply gets no advection instead of
    // an exception. `gas_advection_rate` / `gas_scratch_` are kept as inert
    // config/ABI surface (the P-T0 `inert_n2_idx` idiom); the shipped
    // solid-only unit tests exercised the wind-free branch and are unchanged.
    (void)wind_x;
    (void)wind_y;
    (void)dt;

    // ---- Pass 1: heat -> temperature conversion (proposal §1.2; P2 §4.3) ----
    // Solid tiles: UNCHANGED bit-shift path (bit-identical to pre-P2 — the
    // `if (deposit <= 0) continue` now runs before the solid branch instead of
    // after, but for a solid cell the net computation is identical: the OLD
    // code's `if (!solid[i]) continue` never touched a solid cell's arithmetic
    // either way). Open-air (non-vacuum) tiles: NEW radiation deposit
    // ΔT = ΔE / (N_total · c_v) — a per-tile dynamic-N reciprocal
    // (`reciprocal_q16`, the spike0b/S2c GS-Dinv class) composed with the
    // load-time-constant c_v reciprocal (`make_recip`/`recip_mul`, the
    // water_solver.cpp idiom), both precomputed/amortized ONCE per step (the
    // c_v recip is loop-invariant; the N recip is genuinely per-tile-per-tick,
    // so it CANNOT be a make_recip constant — that is exactly why the two
    // idioms are composed rather than either alone). `heat` is non-negative
    // (saturating accumulator), `atmosphere`/N is non-negative, c_v > 0 by
    // config contract -> every intermediate here is non-negative, so the
    // truncating narrows below cost no sign asymmetry (matches the solid
    // branch's own `>> shift` convention). Vacuum tiles: skipped (heat is
    // irrelevant there — Pass 0 already zeroed their T).
    {
        using namespace fixedpoint;
        // T5b / report_t2.md §10.2: `c_v` has exactly ONE integer representation
        // in this engine -- `c_v_q`, the Q16.16 MULTIPLIER the conduction
        // capacity is built from (hoisted to the top of step(), where the
        // `c_v <= 0` guard already lives). This deposit's divisor is its EXACT
        // inverse, NOT a second quantization of the same config scalar:
        // `make_recip(c_v)` and `1/quantize(c_v)` agree only at c_v == 1 and
        // differ by 0.0724 % at the shipped 0.0076849, which would make a gas
        // cell's deposit and its conduction face disagree about what one heat
        // count is worth. Bit-identical at c_v == 1 (65536/65536 is the very
        // same value), so the change was free at the old dial.
        // The local `c_v_safe` this line used to read is DELETED with it -- the
        // guard is `c_v_q`'s now, and one guard is the point.
        const int64_t recip_cv = make_recip((double)c_v_q / 65536.0);  // 1/c_v_q
        // (`n_floor_q` — the §4.3 deposit floor — is hoisted to the top of
        //  step() at P-E2a and shared with the conduction capacity build.)
        // v2.4 T_MAX_PHYS rail (temperature_solver.h; full rationale in
        // eos_solver.h): Pass 1 is a DEPOSIT path — clamp at the physical
        // ceiling (counted) so an N-starved reciprocal or a stacked
        // firestorm can never write past it. Both branches (solid shift,
        // gas reciprocal). Passes 2/3 need no rail: conduction is a convex
        // combination (discrete maximum principle — it cannot create a new
        // max) and cooling only shrinks |T|.
        const int32_t t_max_phys_q = quantize((double)T_MAX_PHYS);

        for (int i = 0; i < n; ++i) {
            // ---- P-R4: the SIGNED radiation fold (ruling A1.5/A1.7) -------
            // Runs FIRST and INDEPENDENTLY of the `deposit <= 0` skip below —
            // that skip is the painter-era gate and it would silently swallow
            // every radiative LOSS (a fire that cannot cool by radiating). The
            // conversion is the SAME per-material lumped absorption the heat
            // deposit uses (`>> heat_inv_shift`), but with `shr_round0` so it
            // is SYMMETRIC about zero: +x and −x lose the same magnitude, no
            // sign-dependent DC drift across a long burn.
            //
            // ORDER IS PINNED: radiation fold, THEN the heat deposit — both
            // clamp at the T_MAX_PHYS rail, so the order is observable there
            // and the CUDA twin pins the identical order.
            //
            // THE SOLID BRANCH: a thermal solid converts its rad_net through its
            // own `>> heat_inv_shift`. Since P5c a GAS cell has a branch of its
            // own below (its smoke term makes it absorb and emit, design v3
            // §6.3); the mask keeps the two from ever crossing — a gas cell
            // never takes the bit-shift path, a solid never the seam.
            if (rad_net != nullptr && ts[i]) {
                const int64_t rn = rad_net[i];
                if (rn != 0) {
                    // arc #54 P-G5: booked as the cell's ACTUAL applied ΔT
                    // (post every clamp below) × its real capacity — see the
                    // header block's rationale for why this needs no separate
                    // rail counter.
                    const int32_t t_before_rad = temperature[i];
                    // P3a-1: the int64 twins of the SAME two functions. On
                    // every int32-range `rn` they return exactly what
                    // shr_round0/sat_add_q16 returned (gated by
                    // tests/test_fixed_point_i64_twins.py), which is what makes
                    // the widening byte-identical; beyond it they are correct
                    // where the narrow forms could not be.
                    // M1: SIGNED in the exponent. `heat_inv_shift` may be
                    // NEGATIVE (a row lighter than one thermal_mass unit), and
                    // a negative `>>` is UB. On every non-negative exponent
                    // this returns exactly what shr_round0_i64 returned, so no
                    // shipped row moves; the negative branch is EXACT (a left
                    // shift loses nothing where the right shift truncates).
                    const int64_t dTr =
                        shr_round0_signed_i64(rn, heat_inv_shift[i]);
                    // SYMMETRIC saturating add: raycaster.h's
                    // heat_saturating_add early-returns on delta <= 0 (its
                    // accumulator is contractually non-negative), which would
                    // drop exactly the radiative losses this fold exists to
                    // deliver. sat_add_q16 is the kit's signed twin, built for
                    // temperature for precisely this reason (fixed_point.h).
                    temperature[i] = sat_add_q16_i64(temperature[i], dTr);
                    // ---- THE MAXIMUM-PRINCIPLE CLAMP (ray-engine-v2 P1,
                    // design v3 §2.8, in its CORRECT form, row 21) ------------
                    //   T_new = min(T_after, max(T_before, e_ceiling_q(Φ)))
                    // It binds only when radiation would RAISE the cell past
                    // the top of the first E° bucket that out-emits the fluence
                    // it absorbed (P5d, Erik's ruling of 2026-09-24: one bucket
                    // of headroom over the black body in equilibrium with it —
                    // the resolution the emission itself works at; E°⁻¹(Φ), a
                    // bucket short, pinned every undamped cell below its
                    // balance and withheld its whole surplus); a cell already
                    // above that (a burning crate, heated by combustion) keeps
                    // T_before as its ceiling and may only cool; a cooling step
                    // is never clipped. Its home
                    // is HERE — between the saturating add and the rails and
                    // BEFORE the applied-ΔT booking — so the P-G5 solid ledger
                    // books the clipped landing with no extra counter. Dormant
                    // unless both planes are supplied (see the header); the
                    // engagement count is `rad_clamp_hits`. This is the
                    // transcription of sweep_ref_q.py::fold_pass1_solid.
                    if (rad_fluence != nullptr && e_table != nullptr) {
                        const int32_t t_cap = e_ceiling_q(e_table, rad_fluence[i]);
                        const int32_t ceiling =
                            (t_cap > t_before_rad) ? t_cap : t_before_rad;
                        if (temperature[i] > ceiling) {
                            // P5c (design v3 §2.8 / §8.4): the energy the clamp
                            // WITHHOLDS, priced at the cell's real capacity —
                            // the same capacity the landing below is booked at,
                            // so landing + drop is the unclamped step exactly.
                            e_rad_clamp_drop_sum +=
                                ((int64_t)temperature[i] - ceiling) * cap_real_[i];
                            temperature[i] = ceiling; ++rad_clamp_hits;
                        }
                    }
                    if (temperature[i] > t_max_phys_q) {
                        temperature[i] = t_max_phys_q; ++t_max_phys_hits;
                    }
                    // ---- THE LOW RAIL (P-F1a; v7.2) -----------------------
                    //
                    // P-R4's "no LOW rail is needed" comment is VOID. It argued
                    // from antisymmetry that a tile can only be dragged TOWARD
                    // another tile's temperature — but that reasoned about ONE
                    // pair, and a tile's rad_net is the AGGREGATE of every pair
                    // it is in plus every direction it casts, INCLUDING rule
                    // 4's sky charge against a T = 0 ambient. Nothing in
                    // antisymmetry bounds that aggregate below zero.
                    //
                    // THE BUDGET ARGUMENT (what actually holds, v7.2): every
                    // term is clamped to a |ΔT|/2^RAD_LIM_SHIFT share of the
                    // gap through this end's own thermal mass — 1/16 per ray,
                    // and the mutual branch halves that again. With
                    // fire_ray_count == 8 the worst-case aggregate a tile can
                    // shed in one tick is bounded well inside its own gap to
                    // the coldest participant, so 0 is not approached in any
                    // operating scenario. The rail is therefore a COUNTED
                    // DIAGNOSTIC, not a load-bearing clamp: it must be INERT in
                    // every gate scenario (a hit inside a gate run is a RED),
                    // and it exists so that if the argument is ever wrong we
                    // learn it from a counter instead of from a tile going
                    // negative and inverting the next tick's exchange sign.
                    //
                    // 0 is the ambient floor every solid starts at (temperature
                    // is ΔT above the 20 °C reference, not an absolute).
                    if (temperature[i] < 0) {
                        temperature[i] = 0; ++t_low_rail_hits;
                    }
                    // arc #54 P-G5: the radiation fold's ACTUAL landing on
                    // this thermal solid, post both rails.
                    e_solid_deposit_sum +=
                        ((int64_t)temperature[i] - t_before_rad) * cap_real_[i];
                }
            }
            // ---- P5c: THE GAS BRANCH OF THE RADIATION FOLD ----------------
            // Design v3 §2.8 / §6.3; the transcription of sweep_ref_q.py::
            // fold_pass1_gas (gate G15), held to it bit for bit by
            // tests/test_temperature_gas_radiation.py. An ACCOUNTABLE gas cell
            // whose smoke term made the sweep book a nonzero rad_net takes it
            // into the conserved field, in order:
            //   dT       = sign(rn) · deposit_dT_wide_i64(|rn|, recip_N, recip_cv)
            //   T_before = mirror_q(E, N)            (the STORED energy's mirror)
            //   T_target = min(sat(T_before + dT), max(T_before, e_ceiling_q(Φ)))
            //              (P5d's ceiling — the solid branch above has the why)
            //   dE       = N · (T_target − T_before)
            //   deposit_railed(dE), e_gas_deposit_sum += dE
            // * THE CONVERSION is §2.8's STAGED chain — magnitude then sign, so
            //   +x and −x lose equal magnitude (the shr_round0 symmetry idiom) —
            //   in this pass's OWN gas currency: N floored at n_floor_q exactly
            //   as the heat deposit below floors it, the per-cell Newton
            //   reciprocal, and `recip_cv`, c_v's one exact inverse. It is the
            //   SAME chain the sweep's Fleck gas arm prices this cell's loss in
            //   (radiation_sweep.h fleck_L_gas_q, PhysicsEngine::gas_capacity_q),
            //   so the damping and the landing are one arithmetic.
            // * NO min(N, N_AMB)/N_AMB factor (design §6.3): the density law
            //   already lives in a_gas, so rad_net IS the absorbed amount —
            //   applying the heat deposit's factor as well would debit the
            //   stream in full and destroy the un-absorbed remainder.
            // * THE CLAMP IN ITS ENERGY FORM (§2.8): never `temperature[i] =`
            //   (CLAUDE.md "Gas temperature is a mirror": a bare write moves no
            //   books and goes VACUOUS). N·(T_target − T_before) lands the
            //   mirror EXACTLY on T_target, because the mirror is a floor-division
            //   of E by N. The design's letter, N·(T_target + t_amb) − E, lands on
            //   the same mirror but also DRAINS the cell's sub-LSB residual E mod
            //   N (up to N − 1 raw counts) on every clamped tick — the drip class
            //   arc #54 exists to kill — so this form keeps it: the clamp only
            //   ever WITHHOLDS part of a radiative gain, dE is never negative on
            //   a clamped cell. G15 (a) measures both forms.
            // * THE WITHHELD STEP IS COUNTED (§8.4) in `e_rad_clamp_drop_sum`,
            //   priced at cap_real_ like the solid branch above: a whole number
            //   of temperature LSBs times the real capacity, so it is exact.
            // * Cooling below ambient is not railed here: the once-per-tick
            //   recovery rails (eos_solver.cpp step 7) own T_MIN, and radiation
            //   alone cannot drive a cell below ambient (§6.3; gate 14 pins it
            //   for the Fleck-damped step).
            // * A cell with rn == 0 is untouched — not even its mirror — so a
            //   smoke-free scene (a_gas == 0, hence rn == 0, on every gas cell)
            //   folds bit-identically to the pre-P5c pass.
            // * NOTHING LEAVES THE BOUNDARY UNCOUNTED (P5c follow-up, design
            //   §8.4 "bounded AND counted"; sweep_ref_q's module docstring and
            //   gate G15 (g)). Two exits are neither a landing nor a truncation,
            //   so §8.4's one-LSB bound does not cover them, and both are counted
            //   in e_rad_clamp_drop_sum's heat currency (rn in heat counts, ×
            //   FP_ONE), SIGNED, as COUNTERS ONLY — no landing moves:
            //   - `e_rad_boundary_export_sum`: rad_net on a gas cell OUTSIDE the
            //     accountable set (the ambient ring, an open breach). It is never
            //     landed — the cell's energy is not in the books, and Pass 0 pins
            //     it to ambient every tick — so it leaves with the boundary: the
            //     radiative twin of the face flux's vacuum/ring OUTFLOW export.
            //   - `e_rad_floor_drop_sum`: on an accountable cell whose bulk N is
            //     below n_floor_q the chain divides by the floor and lands only
            //     ~N/n_floor of rn; the conversion's whole unlanded remainder,
            //     rn·FP_ONE − (T_after − T_before)·cap_real, is booked there — the
            //     radiative twin of Pass 2's e_cond_cap_sum.
            //   Neither is a term of the #54 identity: neither touches gas_energy,
            //   so a term there would break it by exactly its own amount.
            // Energy form only: with gas_energy == nullptr (the pre-#54 direct
            // binding path) a gas cell's rad_net is not folded, as before P5c.
            if (rad_net != nullptr && e_on && !ts[i] && !acct(i)) {
                const int64_t rn = rad_net[i];
                if (rn != 0) e_rad_boundary_export_sum += rn * (int64_t)FP_ONE;
            }
            if (rad_net != nullptr && e_on && !ts[i] && acct(i)) {
                const int64_t rn = rad_net[i];
                if (rn != 0) {
                    const int64_t nb = n_books(i);                 // the books' N
                    const int32_t N_raw = (n_bulk != nullptr) ? n_bulk[i] : atmosphere[i];
                    const bool floored = (N_raw < n_floor_q);      // the chain's floor
                    const int32_t N_q = floored ? n_floor_q : N_raw;
                    const int32_t recip_N_q = reciprocal_q16(N_q);
                    const int64_t mag = deposit_dT_wide_i64(
                        (rn < 0) ? -rn : rn, recip_N_q, recip_cv);
                    const int64_t dT = (rn < 0) ? -mag : mag;
                    const int32_t t_before =
                        gas_energy::mirror_q(gas_energy[i], nb, t_amb_q);
                    int32_t t_target = sat_add_q16_i64(t_before, dT);
                    if (floored) {
                        e_rad_floor_drop_sum += rn * (int64_t)FP_ONE
                            - ((int64_t)t_target - (int64_t)t_before) * cap_real_[i];
                    }
                    if (rad_fluence != nullptr && e_table != nullptr) {
                        const int32_t t_cap = e_ceiling_q(e_table, rad_fluence[i]);
                        const int32_t ceiling =
                            (t_cap > t_before) ? t_cap : t_before;
                        if (t_target > ceiling) {
                            e_rad_clamp_drop_sum +=
                                ((int64_t)t_target - ceiling) * cap_real_[i];
                            t_target = ceiling; ++rad_clamp_hits;
                        }
                    }
                    const int64_t dE = nb * ((int64_t)t_target - (int64_t)t_before);
                    gas_energy::deposit_railed(
                        gas_energy, temperature, i, dE, nb,
                        t_amb_q, t_max_phys_q, &e_gas_rail_sum,
                        &t_max_phys_hits);
                    e_gas_deposit_sum += dE;
                }
            }
            int32_t deposit = heat[i];
            if (deposit <= 0) continue;       // nothing to convert this tick
            // MEDIUM-TEST SITE 5/6: the heat->T convert branch. A thermal
            // solid takes the free per-tile bit-shift (heat >> log2(
            // thermal_mass)); gas takes the N-divided radiative deposit below.
            if (ts[i]) {
                const int32_t t_before_dep = temperature[i];  // arc #54 P-G5
                // M1: WIDE and SIGNED. `heat_inv_shift` may be NEGATIVE now
                // (a row lighter than one thermal_mass unit), and that turns
                // this divide into a MULTIPLY -- which is the one real
                // overflow risk of the whole change: `deposit << 4` leaves
                // int32 above 2^27, and `deposit` is a Q16.16 heat
                // accumulator that routinely carries far more than that.
                // Route through int64 and let the kit's saturating adder be
                // the only narrowing, exactly as the radiation fold above
                // does. `deposit > 0` here, so the round-toward-zero branch is
                // a plain `>>` and every non-negative exponent is bit-identical
                // to the shipped `deposit >> shift`.
                const int shift = heat_inv_shift[i];   // log2(thermal_mass)
                const int64_t gain =
                    shr_round0_signed_i64((int64_t)deposit, shift);
                temperature[i] = sat_add_q16_i64(temperature[i], (gain > 0) ? gain : 0);
                if (temperature[i] > t_max_phys_q) {
                    temperature[i] = t_max_phys_q; ++t_max_phys_hits;
                }
                // arc #54 P-G5: the heat-deposit's ACTUAL landing, post rail.
                e_solid_deposit_sum +=
                    ((int64_t)temperature[i] - t_before_dep) * cap_real_[i];
            } else if (!is_vacuum[i]) {
                // EOS P3 (TODO closed): the divisor is the REAL bulk-species
                // N_total (O2+N2, passed by the engine) — the P2 atmosphere
                // density-proxy remains only as the nullable back-compat
                // fallback for the direct Python binding.
                //
                // v2.4 ABSORPTION-PROPORTIONAL radiant deposit (optically-thin
                // form; PROVISIONAL, Erik review at P5 — design doc §4 v2.4).
                // The old formula deposited the FULL ray energy into the cell
                // no matter how thin its gas — as a hot zone's own pressure
                // evacuated its N, the same deposit divided by an ever-smaller
                // N and the reciprocal ran away (the measured decisions.md #16
                // driver). Physically, a gas ABSORBS radiation in proportion
                // to its density (this project's own engine/05 optics model,
                // applied to the heat channel):
                //     E_abs = deposit · min(N, N_AMB)/N_AMB
                //     ΔT    = E_abs / (max(N, N_FLOOR_HEAT) · c_v)
                // Consequences: for N_FLOOR_HEAT ≤ N ≤ N_AMB this collapses to
                // ΔT = deposit/(N_AMB·c_v) — BOUNDED regardless of N-collapse;
                // below the floor it decays linearly to 0 (a near-vacuum cell
                // absorbs almost nothing — the physical truth the old formula
                // violated); at/above ambient it reduces EXACTLY to the old
                // chain (e_abs == deposit, bit-identical — zero feel change in
                // normal air). N_AMB == FP_ONE by construction (§2.1 P1
                // calibration: ambient N_total quantizes to exactly 1.0), so
                // min(N, N_AMB)/N_AMB is just min(N, FP_ONE) — no new divide,
                // no new dial.
                int32_t N_raw = n_bulk ? n_bulk[i] : atmosphere[i];
                if (N_raw < 0) N_raw = 0;                 // no negative density
                const int32_t e_abs = (N_raw >= FP_ONE)
                    ? deposit                              // ambient+: exact old path
                    : mul_q16(deposit, (q16)N_raw);        // thin gas: ∝ density
                if (N_raw < FP_ONE) {
                    // P-E2b / L3-7: the (1-N)*deposit attenuation drop below
                    // ambient density is PHYSICAL and stays — it never had a
                    // counter. Same currency as `deposit`/`heat`. N_raw<FP_ONE
                    // here so deposit >= e_abs (one-way destruction).
                    e_deposit_drop_sum += (int64_t)deposit - (int64_t)e_abs;
                }
                int32_t N_q = N_raw;
                if (N_q < n_floor_q) N_q = n_floor_q;    // floor independent of anything else (N_FLOOR_HEAT)
                const int32_t recip_N_q = reciprocal_q16(N_q);        // 1/N, per-tile Newton recip
                // P-E2b: the WIDE E_abs/(N*c_v) chain (int64, no premature
                // q16 narrow — fixed_point.h's deposit_dT_wide_q16 header
                // comment: at n_floor_heat as low as 0.01-0.001, E_abs/floor
                // alone can exceed q16's ~32768 ceiling for a routine
                // deposit). Clamp to a safe non-negative int32 range BEFORE
                // narrowing for heat_saturating_add; an honestly-huge
                // deposit still hits the T_MAX_PHYS rail right below,
                // through a value that was never corrupted on the way there.
                const int64_t dT_wide =
                    deposit_dT_wide_q16(e_abs, recip_N_q, recip_cv);
                const int32_t dT =
                    (int32_t)std::clamp<int64_t>(dT_wide, 0, INT32_MAX);
                // arc #54 P-G1b (design §2.7 row 3): on an ACCOUNTABLE gas cell
                // the deposit lands in the conserved field, not in the mirror.
                // The QUANTITY is unchanged on purpose — `N·dT` is exactly the
                // books-energy the old `temperature += dT` implied, so this is
                // a booking change, not a retune: the absorption law, the
                // n_floor divide and their counters above are all still the
                // ones that decide `dT`. The T_MAX_PHYS rail rides the seam's
                // railed form so the ceiling is applied to the STORED energy
                // (a mirror-only clamp would be undone by the next refresh).
                if (e_on && acct(i)) {
                    const int64_t nb = n_books(i);
                    gas_energy::deposit_railed(
                        gas_energy, temperature, i, nb * (int64_t)dT, nb,
                        t_amb_q, t_max_phys_q, &e_gas_rail_sum,
                        &t_max_phys_hits);
                    e_gas_deposit_sum += nb * (int64_t)dT;
                } else {
                    heat_saturating_add(&temperature[i], dT);
                    if (temperature[i] > t_max_phys_q) {
                        temperature[i] = t_max_phys_q; ++t_max_phys_hits;
                    }
                }
            }
        }
    }

    // DEBUG probe (temporary): T after Pass 1 (heat -> temperature convert).
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_heat = temperature[dbg_probe_idx];

    // ---- Pass 2: conduction relaxation — ENERGY FORM (P-E2a, design §2.3) ----
    // Gather stencil, double-buffered so the whole pass reads the FROZEN
    // pre-conduction field and writes a fresh one (order-independent, no
    // scatter, no atomics). For tile i with 4 neighbours j ∈ {N,S,E,W}:
    //
    //     ΔE_i = Σ_faces  face_energy_q(T_i, T_j, C_i, C_j, s)
    //     T_new[i] = T_i + floordiv_q(ΔE_i, C_i)
    //
    // The four constraints and the full rationale live in the header block;
    // the two properties to keep in mind while reading this loop are:
    //   (1) `face_energy_q` is EXACTLY antisymmetric under swapping the
    //       endpoints, so the same face evaluated from j returns −ΔE. Nothing
    //       here may make an endpoint-asymmetric choice: `s` is the MAX of the
    //       two facing shifts (symmetric even if the bake were not), C_min is a
    //       min (symmetric), the limiter acts on the magnitude (symmetric).
    //   (2) with s ≥ SHIFT_MIN == 2 and C_min ≤ C_i, each face moves at most
    //       ΔT = g/4, so the update remains a convex combination over the four
    //       neighbours — the discrete maximum principle survives the change of
    //       currency, up to the ≤1-LSB `floordiv_q` overshoot on a LOSING cell.
    //
    // A NO_FACE face (grid edge or κ==0 either side) is skipped from BOTH
    // sides, so air-with-conductivity-0 stays a structural no-op and an air
    // tile at 0 stays bit-exactly 0. The skip set is otherwise UNCHANGED from
    // the old law: vacuum / ring / solid cells all still conduct exactly where
    // they did — this patch changes the currency, not who participates.
    scratch_.resize(n);
    int32_t* temp_new = scratch_.data();
    const int NO_FACE = no_face;
    // arc #54 P-G1b: the gas side's per-cell face sum, parked here and applied
    // AFTER the swap. It cannot be applied inside the gather loop: the seam
    // refreshes the mirror, and the pass's whole determinism argument is that
    // every cell reads the FROZEN pre-conduction `temperature`.
    if (e_on) {
        de_gas_.assign(n, 0);
    }

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            const int i = y * w + x;
            // T5b / ledger 7b: a vacuum cell owns no face at all (header:
            // `no_cond_`). Applied per CELL rather than per face, from both
            // ends, so it is symmetric by construction and can never produce a
            // one-sided face.
            if (no_cond_[i]) {
                temp_new[i] = temperature[i];
                continue;
            }
            const int32_t* fs = &face_shift[i * 4];  // [N,S,E,W] for this tile
            const int64_t ti = (int64_t)temperature[i];
            const int64_t cap_i = cap_used_[i];
            int64_t de = 0;
            for (int d = 0; d < 4; ++d) {
                const int s_i = fs[d];
                if (s_i == NO_FACE) continue;        // grid edge or κ==0 -> no face
                const int ny = y + DY[d];
                const int nx = x + DX[d];
                // NO_FACE already marks grid edges, so neighbours are in-bounds;
                // guard anyway for robustness against a mis-baked cache.
                if (ny < 0 || ny >= h || nx < 0 || nx >= w) continue;
                const int j = ny * w + nx;
                if (no_cond_[j]) continue;           // the other end of the mask
                // The NEIGHBOUR's facing entry, read so the face is skipped and
                // rated identically from both ends BY CONSTRUCTION. The shipped
                // harmonic-mean table is symmetric, so `max` picks the common
                // value and this is bit-identical to reading only our own side.
                const int s_j = face_shift[j * 4 + conduction::opposite_dir(d)];
                if (s_j == NO_FACE) continue;
                const int s = (s_i > s_j) ? s_i : s_j;
                de += conduction::face_energy_q(ti, (int64_t)temperature[j],
                                                cap_i, cap_used_[j], s,
                                                &cond_limit_hits);
            }
            if (de == 0) {                           // exact rest: equal neighbours
                temp_new[i] = (int32_t)ti;
                continue;
            }
            // ---- arc #54 P-G1b (design §2.7 row 3): THE ENDPOINT DIVIDE IS
            // DELETED FOR GAS. An accountable gas cell's four-face sum goes to
            // the seam with no Q16 truncation of ΔT, which is what the deleted
            // divide cost. This also retires this cell's `e_cond_trunc_sum`
            // contribution: with no divide there is no truncation residual to
            // count. The SOLIDS side (and any non-accountable gas cell — ring,
            // vacuum) keeps the T-form law below, unchanged (D2).
            //
            // ---- T5a: `de` IS NOT ALREADY IN THE BOOKS' CURRENCY -----------
            // What stood here until 2026-09-19 claimed it was: *"`de` is
            // already in the books' currency (a face quantum is |ΔT|·C with
            // C = N·c_v, and the books' capacity IS N)"*. Read it twice — it
            // names the two capacities in one breath and then equates them.
            // A face quantum is priced at `C = N·c_v`; the books' capacity is
            // `N`. They differ by exactly `c_v`, so booking `de` unconverted
            // gave the gas `c_v ×` the energy the faces moved.
            //
            // At the shipped `c_v = 1` the two coincide BY ACCIDENT, which is
            // why this survived arc #54, its closure identity and every golden.
            // T2 §3.1 measured the ratio on a live sim as `c_v` to eight
            // figures: at the derived physical `c_v = 0.0076849`, **99.23 % of
            // every joule conducted into air is destroyed at the face**. The
            // #54 closure identity closes 0/40 ticks bad in BOTH cases (T2
            // §5.1) — it books what was deposited, not what should have been,
            // so it is structurally blind to this and must never be cited as
            // evidence that the currency is sound.
            //
            // Two consequences worth keeping in the file, because they decide
            // the shape of the flip: (1) turning `c_v` alone would NOT fix the
            // conduction path — its gas ΔT was `de/N`, which has no `c_v` in
            // it — so a config-only flip ships an engine half-corrected in the
            // worst way, deposit path at the real capacity and conduction path
            // still at the convention; (2) the conversion below is provably
            // bit-identical at `c_v = 1`, which is why it lands as its own
            // patch, gated by unmoved goldens, BEFORE the dial moves.
            //
            // THE CAPACITY FLOOR STILL APPLIES, and it is load-bearing.
            // `cap_used` is `max(N, n_floor_heat)·c_v`; where that floor binds
            // (a near-vacuum cell), the T-form moved `ΔT = de/cap_used`, i.e.
            // the cell absorbed only a `cap_real/cap_used` FRACTION of the
            // face energy. That fraction is exactly what keeps the pass a
            // convex combination — the discrete maximum principle the header
            // leans on when it says Pass 2 needs no rail. Depositing the raw
            // `de` instead would overshoot at precisely those cells, and it
            // does: measured on the VENT bench (a room decompressing to
            // vacuum) the raw form drove `gas_energy` NEGATIVE on 8
            // cell-ticks, worst −4.5e8 — a negative absolute temperature, fed
            // straight into next tick's `p* = C·E` before the recovery could
            // rail it. So the floor's shrink is applied here, in the ENERGY
            // currency, and the difference keeps its existing name
            // (`e_cond_cap_sum`: "the cell moved ΔT through cap_used while
            // really owning cap_real").
            //
            // The product is int64-safe because `muldiv_floor_q` divides
            // FIRST — see its header. The shipped form's `de·cap_real` was
            // safe only because it ran ONLY when the floor bound; the
            // conversion below runs on every accountable gas cell, and the
            // naive `de·N` leaves int64 at N = 2 atm.
            if (e_on && acct(i)) {
                // T5a (T2 §4.5, §10.2): the conversion from HEAT COUNTS to the
                // books' N·T currency, written as the EXACT INVERSE OF THE
                // CAPACITY THE FACE QUANTUM WAS BUILT FROM. `de` is `ΔT·cap`,
                // so `N·ΔT` is "divide by the SAME cap, multiply by N" —
                // self-inverting to the LSB, and `c_v` never appears in it.
                // That last part is the point: `c_v` has TWO integer
                // representations in this TU (`c_v_q`, Q16.16, which built the
                // capacity; `recip_cv`, Q.32, which Pass 1 divides by) and they
                // agree ONLY at `c_v == 1`. Re-dividing here by the second one
                // would be permanently off by their disagreement; inverting
                // `cap` cannot be.
                const int64_t nb = n_books(i);           // real N, unfloored
                const int64_t de_books =
                    conduction::muldiv_floor_q(de, nb, cap_i);        // cap_USED
                // What this cell WOULD have booked at its real (unfloored,
                // unclamped) capacity. Their difference is exactly the capacity
                // floor's — or the ceiling's — contribution, which is what
                // `e_cond_cap_sum` has always named; it is now denominated in
                // the BOOKS' currency rather than in heat counts, so it keeps
                // meaning the same thing once `c_v` stops being 1.
                //
                // THE `cap_real == 0` CORNER is `N == 0`: the real books delta
                // is 0 and the WHOLE of `de` is the floor's fiction. Convert it
                // through the only capacity left — `c_v` itself, in the SAME
                // Q16.16 representation the `cap` planes were built from, never
                // through `recip_cv` (§10.2 again; `recip_mul` would also
                // narrow `de` to int32 on the way in).
                const int64_t de_full = (cap_real_[i] > 0)
                    ? conduction::muldiv_floor_q(de, nb, cap_real_[i])
                    : conduction::muldiv_floor_q(
                          de, (int64_t)fixedpoint::FP_ONE,
                          (c_v_q > 0) ? (int64_t)c_v_q
                                      : (int64_t)fixedpoint::FP_ONE);
                e_cond_cap_sum += de_full - de_books;
                de_gas_[i] = de_books;
                e_gas_cond_sum += de_books;
                temp_new[i] = (int32_t)ti;           // mirror refreshed post-swap
                continue;
            }
            // Endpoint-local conversion (R2), through the SHARED floor-division
            // helper so the residual is one-way DESTROYING in both signs — a
            // truncation toward zero would MINT on cells losing energy, and
            // both backends' `/` agree on that mint so only the ledger could
            // see it (fixed_point.h floordiv_q's own header says this).
            const int64_t dT = fixedpoint::floordiv_q(de, cap_i);
            // R3: every residual counted, in ENERGY.
            //  * trunc — what the endpoint divide dropped (≤ 0, always).
            //  * cap   — what the capacity FLOOR (thin gas) or the overflow
            //            CEILING implied: the cell moved ΔT through cap_used
            //            while really owning cap_real, so the books gained
            //            ΔT·(cap_real − cap_used) that the faces never moved.
            e_cond_trunc_sum += dT * cap_i - de;
            e_cond_cap_sum   += dT * (cap_real_[i] - cap_i);
            // arc #54 P-G5: this cell's exact conduction landing, priced at
            // its REAL capacity — dT*cap_real_[i] == de + (this cell's own
            // trunc+cap contributions above), i.e. the true ΔT·thermal_mass
            // this tick, whatever endpoint the energy crossed from.
            if (ts[i]) e_solid_cond_sum += dT * cap_real_[i];
            temp_new[i] = (int32_t)(ti + dT);
        }
    }

    // Swap temp_new -> temperature (write the new field back in place; the
    // caller's buffer is the persistent one, scratch_ is reused next tick).
    for (int i = 0; i < n; ++i) temperature[i] = temp_new[i];
    // arc #54 P-G1b: NOW apply the gas side's parked face sums, once the whole
    // frozen-field gather is finished. `deposit` (not `deposit_railed`) — the
    // rails belong to the once-per-tick recovery (design §2.6) and conduction
    // is a convex combination that cannot create a new maximum anyway.
    if (e_on) {
        for (int i = 0; i < n; ++i) {
            if (de_gas_[i] == 0) continue;
            gas_energy::deposit(gas_energy, temperature, i, de_gas_[i],
                                n_books(i), t_amb_q);
        }
    }
    // DEBUG probe (temporary): T after Pass 2 (conduction).
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_conduction = temperature[dbg_probe_idx];

    // ---- Pass 3 is DELETED (thermal model v2 R1, T5b step 7) ------------
    // What stood here was `T -= T >> cool_shift`, a Newtonian relaxation of
    // every thermal solid toward ambient, with a 4x-faster variant for a
    // vacuum-facing tile. Its own comment called it what it was: a hand-rolled
    // stand-in for radiative loss. The sweep computes the real one now
    // (`rad_net_sweep`, folded in Pass 1, with `k_leak` carrying the
    // out-of-plane share), so keeping this would count the same physics
    // TWICE -- and the hand-rolled copy is the one with no ledger, no
    // geometry and no temperature dependence.
    //
    // Deleted with it: the `cool_shift` / `cool_shift_vacuum` /
    // `cool_shift_floor` dials, the per-material `cool_shift` column and its
    // `cool_shift_grid` plane, and BOTH counters this pass fed --
    // `e_cool_sum` and `e_thermostat_sum`. The #54 closure identity now
    // closes with the thermostat TERM REMOVED, not zeroed: there is no
    // channel left to book. (design v2 R1 / section 7.3; the ambient
    // thermostat that Erik's 2026-08-30 ruling made a deliberate modelling
    // boundary was exactly this pass's solid side, and it goes with it.)
    //
    // NOTHING RELAXES TO AMBIENT ANY MORE. A solid's temperature changes only
    // through a booked channel: the Pass-1 radiation fold, the Pass-1 heat
    // deposit, Pass-2 conduction, and combustion's own object-site write.
    // (design v2 section 6 item 7, gated by
    // tests/test_no_relax_to_ambient.py.)

    // DEBUG probe (temporary): T after Pass 3 (ambient cooling).
    if (dbg_probe_idx >= 0 && dbg_probe_idx < n) dbg_T_post_cooling = temperature[dbg_probe_idx];

    // ---- arc #54 P-G5: solid_energy_books_sum SNAPSHOT ---------------------
    // Σ over thermal_solid cells of cap_real_[i]*temperature[i], as of the end
    // of this tick's Pass 3 — the solid-side twin of GameMap.gas_energy's own
    // Σ_accountable read. NOT accumulated: recomputed fresh every call, since
    // it prices current STATE, not a flow (see temperature_solver.h).
    {
        int64_t solid_sum = 0;
        for (int i = 0; i < n; ++i) {
            if (!ts[i]) continue;
            solid_sum += cap_real_[i] * (int64_t)temperature[i];
        }
        solid_energy_books_sum = solid_sum;
    }

    // STEP D (unit damage, §4) will add a further pass here, reading the
    // post-cool temperature field.
}
