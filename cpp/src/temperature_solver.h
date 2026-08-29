#pragma once
// Temperature solver — turns the per-tick `heat` deposit into a persistent
// `temperature` field on solids (engine/06 §1), then spreads it by CONDUCTION
// (engine/06 §2; temperature_design_proposal §2), then sheds it by AMBIENT
// COOLING (engine/06 §3; proposal §3).
//
// STEP A scope: the heat -> temperature CONVERSION pass (shipped).
// STEP B scope: the CONDUCTION RELAXATION pass, run AFTER the conversion, per
// the proposal §6 order.
// STEP C scope (this file): the AMBIENT COOLING pass, run AFTER conduction (it
// is the LAST thermal pass, §3.5). Unit damage (§4) is a LATER step and will be
// added to step() as a further pass.
//
// THERMAL-MASS AXIS AMENDMENT (2026-07-30 —
// docs/thermal_mass_axis_design_2026-07-25.md + its build addendum): every
// per-medium branch below now keys on `thermal_solid` (`thermal_mass > 0`),
// NOT on `solid` (`permeability <= 0`). Read the P2 text below with that
// substitution: wherever it says the gas rules run on `!solid && !is_vacuum`,
// the mask is `!thermal_solid && !is_vacuum`. WHY: `solid` is a FLOW property,
// and keying the thermal medium on it silently put furniture (permeability
// 0.5 — the deliberate "shield but not seal" soft body) into the GAS regime,
// so a burning crate's temperature was advected away by the fire's own plume.
// `permeability`/`solid`/mobility/LoS are UNTOUCHED — only the thermal medium
// moved. Exactly SIX sites in the .cpp changed (marked "MEDIUM-TEST SITE n/6");
// conduction's κ-keyed face bake is deliberately NOT one of them. furniture is
// the only material that is permeable AND thermally solid, so on any
// furniture-free map `thermal_solid == solid` elementwise and every path is
// byte-identical (the patch's zero-tolerance gate).
//
// EOS refactor P2 (docs/eos_refactor_design.md §4 + §8 patch P2) — UNIFIED
// TEMPERATURE, additive: `temperature` now ALSO carries gas-T on the open-air
// mask (`!solid[i] && !is_vacuum[i]`), the same array, the SAME Q16.16 scale.
//
//   *** REPRESENTATION — locked, do NOT change here ***
//   `temperature` stays Kelvin-RELATIVE-to-ambient (0 == ambient ~290 K), for
//   BOTH solid and gas cells. This is what keeps the solid math bit-untouched
//   (docs/eos_refactor_decisions.md item 7). Rebasing to absolute Kelvin
//   (T_abs = T_rel + T_AMB) is P3's job, at the EOS pressure derivation —
//   NOT here.
//
// Gas rules (open-air mask only, run as NEW passes around the existing
// solid-only ones):
//   * Pass 0 (NEW, "pre-pass"): zero T at every `is_vacuum` cell (energy
//     leaves with the venting gas — a structural invariant, unconditional);
//     then semi-Lagrangian advection of T on `wind_x/wind_y`, reusing the
//     integer DDA-wall-clip-march + bilinear-sample PATTERN of
//     smoke_dynamics.cpp's `backtrace_sample_q` (S2b SLint scheme), adapted to
//     this solver's own `solid`/`is_vacuum` masks (no separate obstacles/
//     permeability arrays needed — `solid` already IS the physics obstacle
//     set, gamemap.py: obstacles == solid == permeability<=0). Skipped
//     entirely when `dt <= 0` or `wind_x`/`wind_y` are null (the Python
//     direct-binding back-compat path — see bindings.cpp).
//   * Pass 1 (EXTENDED): the existing heat -> temperature convert pass gains
//     an `else if (!is_vacuum[i])` branch: an open-air cell with a nonzero
//     `heat` deposit receives `ΔT = ΔE / (N_total · c_v)` (§4.3), a per-tile
//     dynamic-divisor reciprocal (`reciprocal_q16`, the spike0b/S2c GS-Dinv
//     class) composed with the load-time-constant `c_v` reciprocal
//     (`make_recip`/`recip_mul`, the water_solver.cpp idiom). `N_total` is
//     read from `atmosphere` as the P2 DENSITY PROXY (1.0 == ambient) —
//     `// P3:` marks the read that P3 swaps for the real bulk-species
//     `N_total`. The solid branch (bit-shift convert) is UNCHANGED.
//   * Pass 2 (conduction): UNCHANGED CODE — air simply gets a small nonzero
//     `conductivity` in config/materials, so the existing whole-grid
//     face_shift-keyed pass (no solid/air branch) does air<->air AND the
//     solid<->air interface exchange for free (the primary sealed-room energy
//     sink; decisions.md item 7).
//   * Pass 3 (cooling): UNCHANGED — solid-only already (gas cells are
//     structurally excluded: no decay-to-ambient for gas, per design §4).
//
// NON-GOALS here (P3+): no compression-work term (needs the new solver's
// div u), no P/pressure changes, no O2/N2 species (P1, parallel worktree).
//
// Determinism (engine/06 §3, proposal §1.2 / §2.7): both `heat` and
// `temperature` are Q16.16 int32 sharing one scale (TEMP_SCALE == HEAT_SCALE).
//
//   Conversion (§1.2, solids only):
//       temperature[i] = sat_add( temperature[i], heat[i] >> heat_inv_shift[i] )
//   `heat` is a saturating accumulator of NON-NEGATIVE deposits, so the
//   arithmetic right shift is on a non-negative value -> portable and
//   bit-identical across machines/compilers (no float, no division). Air tiles
//   (not solid) are skipped, so an air tile that starts at 0 stays 0.
//
//   Conduction — THE PRE-P-E2a LAW, transcribed here because P-E2a replaced it
//   (design energy_transport_design_2026-08-16.md §2.3 requires the old law be
//   written down before the rewrite; the as-built doc
//   docs/e1_p_e2a_asbuilt_2026-08-17.md §1 carries the full transcription):
//       acc = Σ_{dir∈N,S,E,W}  (temp[n] - temp[i]) >> face_shift[i][dir]
//       temp_new[i] = temp[i] + acc            (then swap temp_new -> temp)
//   The DIFFERENCE was shifted (not the neighbour), so equal neighbours produced
//   EXACTLY 0 change (no drift). A face was skipped when face_shift == NO_FACE
//   (grid edge, or κ==0 on either side). With SHIFT_MIN==2 (max face rate ¼) and
//   4 neighbours, Σr ≤ 1, so the update was a convex combination of {T_i, T_n} —
//   the discrete maximum principle held for free (proposal §2.6).
//   WHY IT HAD TO GO: it relaxed TEMPERATURES, not energies. Across a
//   solid<->air face the wall and the gas have wildly different heat capacities
//   (hull thermal_mass 32 vs gas N·c_v ≈ 1), so "cell i loses ΔT, cell j gains
//   the same ΔT" moved 32× more energy into the wall than it took out of the
//   gas (or destroyed 32× more, in the other direction). That was the sealed
//   room's largest silent energy channel. It was not antisymmetric even in ΔT:
//   the arithmetic shift rounds toward −∞, so the hot side lost ceil(g/2^s)
//   while the cold side gained floor(g/2^s) — a 1-LSB-per-face-per-tick
//   uncounted destruction.
//
//   Conduction (P-E2a, design §2.3 — ENERGY form, gather + double-buffer):
//   Every cell carries a CAPACITY C (gas: N·c_v, floored by the shared
//   `n_floor_heat` dial; object: thermal_mass == 2^heat_inv_shift), held in
//   Q16.16 as `cap_q = C·65536`, so raw energy E = C·T. Then, per face:
//       g     = |T_j − T_i|                 (magnitude FIRST — see below)
//       C_min = min(C_i, C_j)               (symmetric)
//       ΔE    = ±((g·C_min) >> s),  clamped to ±((g·C_min) >> LIM_SHIFT)
//       ΔT_i  = floordiv_q(Σ_faces ΔE, C_i)     (endpoint-local, R2)
//   The FOUR design constraints, and where each lives:
//     1. FACE-ANTISYMMETRIC ΔE. `conduction::face_energy_q` computes the
//        magnitude from |ΔT| and re-applies the sign, and every other input
//        (C_min, s) is symmetric in the endpoint pair — so evaluating the same
//        face from the other cell returns EXACTLY the negation. What leaves i
//        enters j, bit for bit, in int64. (A plain `(T_j−T_i) >> s` is NOT
//        antisymmetric — that is the old law's silent leak, above.)
//     2. ENDPOINT-LOCAL CONVERSION (R2): each endpoint divides the energy it
//        received by ITS OWN capacity. A hull tile taking gas energy warms 32×
//        less than the gas cooled, because it is 32× heavier — which is the
//        physics the ΔT form was papering over.
//     3. ONE-WAY COUNTED GUARDS: the endpoint divide uses `floordiv_q`
//        (toward −∞, the shared §2.1.5/§2.7 helper) so the residual can only
//        DESTROY, never create, in both signs — counted in ENERGY by
//        `e_cond_trunc_sum`. The capacity floor/ceiling's energy is counted by
//        `e_cond_cap_sum`. Nothing else in this pass writes T.
//     4. PER-FACE LIMITER (LIM_SHIFT == 1): |ΔE| ≤ ½ of the energy that would
//        close the whole gap through the SMALLER endpoint capacity. Since
//        moving E across the face changes the gap by E/C_i + E/C_j ≥ E·2/C_min,
//        capping at (g·C_min)/2 guarantees the gap never inverts — neither
//        endpoint can pass the donor. This is the P-R4 `LIM_SHIFT`/A1.6 shift
//        idiom (cuda_raycaster.cu:263-264 precedent) and it restores per face
//        what the convex bound used to give for free. The AGGREGATE bound is
//        still SHIFT_MIN's: with s ≥ 2 and C_min ≤ C_i, each face moves
//        ΔT ≤ g/4, so Σ over 4 faces is still a convex combination and the
//        discrete maximum principle survives — now up to the ≤1-LSB overshoot
//        `floordiv_q`'s toward−∞ rounding can add on the losing side.
//   Equal neighbours still produce EXACTLY 0 (g = 0 ⇒ ΔE = 0). A face is
//   skipped when face_shift == NO_FACE on EITHER side (the neighbour's facing
//   entry is read too, and the slower shift `s = max(s_ij, s_ji)` is used, so
//   the pass is symmetric BY CONSTRUCTION rather than by trusting the bake to
//   be symmetric — the harmonic-mean table is symmetric, so this is
//   bit-identical there). Still order-independent (gather over a frozen
//   buffer), still float-free at runtime, still bit-identical cross-machine —
//   the one new operation is an int64 divide per active cell per tick.
//
//   Ambient cooling (§3, gather over the geometric 4-neighbours) — since the
//   COOL-SHIFT AXIS (2026-07-30) the base shift is PER TILE, not one global:
//       base  = cool_shift_grid ? cool_shift_grid[i] : cool_shift
//       shift = exposed ? max(cool_shift_floor, base - (cool_shift - cool_shift_vacuum))
//                       : base
//       T    -= (T < 0) ? -((-T) >> shift) : (T >> shift)
//   With cool_shift_grid null (or uniformly == cool_shift, the seeded config)
//   this is bit-exactly the old `exposed ? cool_shift_vacuum : cool_shift`.
//   Temperature stores ΔT above ambient, so T_ambient == 0 and cooling relaxes
//   toward 0 with NO subtraction (`T -= T >> shift`). `exposed` is true when ANY
//   in-bounds 4-neighbour is vacuum (is_vacuum) OR has atmosphere < a quantized
//   threshold — read from the SAME atmosphere/vacuum fields the rest of the
//   physics uses (no new field/buffer), so a freshly-breached, now-space-facing
//   wall sheds 4× faster through the existing seam. S3c: `atmosphere` is now the
//   int32 Q16.16 field (the LAST float input to this TU is gone — it is fully
//   integer, matching its already-integer heat/temperature fields). The exposure
//   test `atmosphere[n] < o2_vacuum_thresh` is a Q16.16 integer compare against
//   `quantize(o2_vacuum_thresh)` (computed ONCE per step, the load/boundary cast).
//   Runs on SOLID tiles only
//   (air is already 0, so it is skipped and stays bit-exactly 0). The signed
//   arithmetic right shift is pinned to round toward 0 symmetrically
//   (`x<0 ? -((-x)>>s) : x>>s`) — deterministic, identical cross-machine. The
//   residual DEAD-BAND is intentional: the last `(1<<shift)-1` counts above
//   ambient shift to 0 and never decay, giving an exact, jitter-free resting
//   state at ambient (NO "+1 if nonzero" nudge — that would break the fixed
//   point). The cooled magnitude is always ≤ |T|, so a single isolated tile
//   relaxes toward 0 and never crosses below ambient.

#include <cstdint>
#include <vector>

#include "fixed_point.h"   // P-E2a: FP_HD, FP_SHIFT, floordiv_q (shared helper)

// ===========================================================================
// P-E2a — the conduction ENERGY kit (design §2.3).
//
// These three helpers are the ONE transcription of the new law: the CPU pass
// (temperature_solver.cpp Pass 2) and the CUDA twin (cuda_temperature.cu
// temp_cap_build / temp_conduct) both call them, so the two backends cannot
// drift on an edit. FP_HD makes them callable from a __device__ kernel; the
// header is otherwise plain C++ and is included by the .cu for exactly this.
// ===========================================================================
namespace conduction {

// Capacity ceiling, as a shift: C ≤ 2^CAP_SHIFT_MAX. Load-bearing ONLY as the
// int64-overflow guard on the face product `g·C_min` (|g| ≤ 2^31 raw counts,
// so C ≤ 2^12 keeps g·C_min·(Q16.16 scale) ≤ 2^59 with room for the 4-face
// sum and for the `ΔT·C` counter products). The shipped material table's
// largest thermal_mass is 32 (shift 5) — SEVEN doublings below this ceiling —
// and gas N·c_v never approaches 4096 either, so it is inert in practice. It
// is a clamp rather than an assert because a clamp is deterministic on both
// backends; when it DOES bind, the energy it implies is counted (the cell
// converts through a smaller capacity than it really has, and the difference
// lands in `e_cond_cap_sum` exactly like the n_floor_heat floor's does).
constexpr int CAP_SHIFT_MAX = 12;

// Constraint 4's fraction, as a shift. 1 == "at most HALF the gap closed
// through the smaller endpoint capacity" — the design's pinned ≤ ½, the safe
// side of the f = 2 line.
constexpr int LIM_SHIFT = 1;

// One cell's heat capacity, Q16.16 (raw energy E = cap_q·T >> 16; equivalently
// C = cap_q / 65536 and E_raw = C · T_raw).
//   * object (`thermal_solid`): C = thermal_mass = 2^heat_inv_shift — the SAME
//     divisor Pass 1's `heat >> heat_inv_shift` deposit uses, so a deposit and
//     a conduction gain of equal energy raise T by equal amounts.
//   * gas: C = N·c_v, with N floored by `n_floor_heat` — the SAME dial Pass 1's
//     ΔT = E_abs/(max(N,floor)·c_v) deposit uses (P-E2b owns its VALUE; this
//     patch only shares it).
// `cap_used` is what the law divides by; `cap_real` is the unfloored,
// unclamped truth the ledger's Σ N·T sees. Their difference is the counted
// floor/ceiling term — the R3 "every floor counted in ENERGY units" rule.
FP_HD inline void cell_capacity_q(bool is_ts, int32_t heat_inv_shift_i,
                                  int32_t n_raw, int32_t n_floor_q,
                                  int32_t c_v_q,
                                  int64_t* cap_used, int64_t* cap_real) {
    if (is_ts) {
        int s = (int)heat_inv_shift_i;
        if (s < 0) s = 0;
        int s_used = (s > CAP_SHIFT_MAX) ? CAP_SHIFT_MAX : s;
        int s_real = (s > 30) ? 30 : s;          // int64 hygiene on cap_real too
        *cap_used = (int64_t)1 << (s_used + fixedpoint::FP_SHIFT);
        *cap_real = (int64_t)1 << (s_real + fixedpoint::FP_SHIFT);
    } else {
        const int64_t ceiling = (int64_t)1 << (CAP_SHIFT_MAX + fixedpoint::FP_SHIFT);
        int64_t nr = (int64_t)n_raw;
        if (nr < 0) nr = 0;                       // no negative density
        int64_t nu = (nr < (int64_t)n_floor_q) ? (int64_t)n_floor_q : nr;
        int64_t cr = (nr * (int64_t)c_v_q) >> fixedpoint::FP_SHIFT;
        int64_t cu = (nu * (int64_t)c_v_q) >> fixedpoint::FP_SHIFT;
        if (cu > ceiling) cu = ceiling;
        if (cr > ceiling) cr = ceiling;
        // Divide-by-zero guard: `n_floor_heat` and `c_v` are both positive by
        // config contract, but a direct-binding caller may set either to 0 and
        // the endpoint divide must not fault. 1 raw count of capacity is 2^-16
        // of a unit — far below any real cell, so this never binds in the sim.
        if (cu < 1) cu = 1;
        if (cr < 0) cr = 0;
        *cap_used = cu;
        *cap_real = cr;
    }
}

// ONE face's energy quantum, seen from cell i (positive == energy flows INTO
// i). Constraint 1 lives here: the magnitude is computed from |ΔT| and the
// sign re-applied, and C_min / s are symmetric in the pair — so calling this
// with (t_j, t_i, cap_j, cap_i, s) returns EXACTLY the negation. No rounding
// mode, no shift, and no clamp in this function can break that, because every
// one of them acts on the MAGNITUDE.
FP_HD inline int64_t face_energy_q(int64_t t_i, int64_t t_j,
                                   int64_t cap_i, int64_t cap_j, int s,
                                   int64_t* limit_hits) {
    const int64_t d = t_j - t_i;
    const int64_t g = (d < 0) ? -d : d;
    const int64_t cmin = (cap_i < cap_j) ? cap_i : cap_j;
    const int64_t full = g * cmin;          // energy to close the gap through C_min
    int64_t q = full >> s;                  // the face's baked rate
    const int64_t lim = full >> LIM_SHIFT;  // constraint 4
    if (q > lim) {
        q = lim;
        if (limit_hits) ++(*limit_hits);
    }
    return (d < 0) ? -q : q;
}

// The facing direction index for the shared-face lookup (N<->S, E<->W), in the
// fixed N,S,E,W order this whole TU family uses.
FP_HD inline int opposite_dir(int d) {
    return (d == 0) ? 1 : (d == 1) ? 0 : (d == 2) ? 3 : 2;
}

}  // namespace conduction

class TemperatureSolver {
public:
    // Sentinel face shift: grid edge or κ==0 on either side -> no conduction.
    // MUST match config [physics.thermal].NO_FACE (bound via set_no_face).
    int no_face = 63;

    void set_no_face(int v) { no_face = v; }
    int  get_no_face() const { return no_face; }

    // Ambient cooling shifts (§3.3), bound from config [physics.thermal].
    //   cool_shift        — interior Newtonian decay (T -= T >> cool_shift).
    //   cool_shift_vacuum — space-exposed decay (smaller shift -> faster).
    //
    // COOL-SHIFT AXIS (2026-07-30): these two scalars are no longer the whole
    // story — the decay shift is a PER-MATERIAL column projected to the
    // per-tile `cool_shift_grid` argument of step() below. They keep TWO live
    // jobs, both of which make the axis additive rather than a replacement:
    //   1. FALLBACK: `cool_shift` is the shift used when `cool_shift_grid` is
    //      null (the documented back-compat path, like `solid` is for
    //      `thermal_solid`) — so an un-plumbed caller behaves exactly as before.
    //   2. VACUUM OFFSET: the PAIR defines the space-exposure discount as a
    //      DIFFERENCE, not an absolute: a vacuum-exposed tile cools at
    //          max(cool_shift_floor, cool_shift_grid[i] - (cool_shift - cool_shift_vacuum))
    //      With the shipped 5/3 that is "two shifts == 4x faster", applied
    //      uniformly to every material, so each material still carries exactly
    //      ONE dial. At the seeded config (every material 5) it reproduces the
    //      old interior-5 / exposed-3 pair bit-exactly.
    // cool_shift_floor — the low clamp on that subtraction (bound from config
    //   [physics.thermal] SHIFT_MIN, the same "rate floor" the conduction
    //   buckets use, and the same floor materials.py validates the column
    //   against). Load-bearing: without it a material legally sitting AT the
    //   floor would derive a vacuum shift of 0 == `T -= T`, an instant total
    //   wipe. It never binds at the seeded values (5 - 2 == 3 > 2).
    // o2_vacuum_thresh — atmosphere value below which a neighbour counts as
    //   vacuum for the exposure test (in the same REAL units as gmap.atmosphere,
    //   i.e. the pre-quantize pressure). It is a config dial (bound from Python as
    //   a real value); S3c quantizes it ONCE per step to a Q16.16 count and the
    //   exposure test is then a pure integer compare on the int32 atmosphere field.
    //   Kept as a float member because it is a config/boundary value, not synced
    //   per-cell state (the documented boundary exception, like fire's `dt`).
    int   cool_shift = 5;
    int   cool_shift_vacuum = 3;
    int   cool_shift_floor = 2;      // == config SHIFT_MIN
    float o2_vacuum_thresh = 0.3f;

    void  set_cool_shift(int v) { cool_shift = v; }
    int   get_cool_shift() const { return cool_shift; }
    void  set_cool_shift_vacuum(int v) { cool_shift_vacuum = v; }
    int   get_cool_shift_vacuum() const { return cool_shift_vacuum; }
    void  set_cool_shift_floor(int v) { cool_shift_floor = v; }
    int   get_cool_shift_floor() const { return cool_shift_floor; }
    void  set_o2_vacuum_thresh(float v) { o2_vacuum_thresh = v; }
    float get_o2_vacuum_thresh() const { return o2_vacuum_thresh; }

    // --- P2 gas-T dials (docs/eos_refactor_design.md §4.3, §9) -------------
    // gas_advection_rate — the wind->displacement scale for the gas-T
    //   semi-Lagrangian pre-pass, the SAME config idiom as SmokeDynamics::
    //   advection_rate (dt_adv = gas_advection_rate * dt): "hot air rides the
    //   wind at the same visual rate as smoke" until P3 makes `wind` a real
    //   velocity. TUNING DIAL (§9), default mirrors smoke's shipped 900.0.
    // c_v — gas heat capacity constant for the radiation-deposit divide
    //   ΔT = ΔE/(N·c_v) (§4.3). A load-time-constant divisor -> reciprocal via
    //   make_recip/recip_mul (the water_solver.cpp idiom), computed ONCE per
    //   step (not per cell). TUNING DIAL.
    // n_floor_heat — floor on the per-tile N divisor (the real bulk N_total
    //   since P3; `atmosphere` only on the nullable back-compat path) for the
    //   SAME deposit divide. SHARED by BOTH deposit sites (this Pass-1 gas
    //   branch AND combustion.cpp's aggregate deposit, combustion.cpp:799-803)
    //   — the ONE floor in the system, wired from the same config key
    //   ([physics.thermal].n_floor_heat) to both.
    //
    //   RULING (energy-books arc, design v2.2 §2.2, Erik 2026-08-17): now a
    //   LOW, tunable dial — default 0.01 (was 0.05), swept DOWNWARD during
    //   tuning ("we can see how low we can go"). Its STABILITY job is GONE:
    //   P-E1 closed the EOS transport books (deposit spikes at thin cells now
    //   dilute honestly on contact instead of minting via the old SL T-copy)
    //   and T_MAX_PHYS is the real value backstop for a single-tick spike —
    //   see that field's rail, which is a COUNTED clamp, not this floor. So
    //   n_floor_heat is VALUE HYGIENE only now: it keeps the divide's
    //   reciprocal from blowing up arithmetically at N -> 0, nothing more.
    //   The v1 "0.25 shared with the trust gate" ruling is RETIRED — see
    //   n_work_ref (§2.4) for the (unrelated, unimplemented-this-patch) trust
    //   gate dial.
    //
    //   The v1 eos-p3fix-thermal-ceiling single-tick criterion above this
    //   ruling — N_floor >= heat_tick_max/(T_MAX_PHYS*c_v) — is STALE in the
    //   direction that mattered (it argued for RAISING the floor to protect
    //   T_MAX_PHYS); it no longer gates this dial's value, because T_MAX_PHYS
    //   is now the counted backstop regardless of what the floor lets through
    //   on the way there. The OLD 0.2-trial warning ("measurably perturbed
    //   marginal ignition timings suite-wide") is MOOT IN THIS DIRECTION: a
    //   LOWER floor means LESS dilution of a thin cell's deposit divide — the
    //   L1-4 objection inverts, and marginal ignition trends FASTER, not
    //   slower, as the floor drops (P-E2b measures this; see the as-built).
    //
    //   PRECISION: the divide is a per-cell Newton reciprocal
    //   (`fixedpoint::reciprocal_q16`), int64 internally already (the seed +
    //   4 Newton refinements all run in int64 before narrowing the RESULT to
    //   Q16.16); its self-guard floors any denom < 3 raw counts (< 0.0000458
    //   real) to 3, well below where this dial's sweep (down to 0.001, per
    //   design §2.2) ever lands. Verified by probe
    //   (`tools/e2b_floor_reciprocal_probe.py`): a 0.001 floor keeps the
    //   deposit divide's arithmetic sane (no overflow, no sign flip, no
    //   collapse) across the whole per-cell chain
    //   (reciprocal_q16 -> mul_q16 -> recip_mul).
    float gas_advection_rate = 900.0f;
    float c_v = 1.0f;
    float n_floor_heat = 0.01f;
    // T_MAX_PHYS (v2.4 as-built amendment, PROVISIONAL pending Erik's P5
    // review): the counted physical-maximum T rail — Pass 1's deposit clamps
    // at this ceiling (own counter below). One constant shared across
    // EOSSolver/TemperatureSolver/CombustionSolver, wired from
    // [physics.thermal] by physics_runner. Full rationale: eos_solver.h.
    float T_MAX_PHYS = 16000.0f;
    mutable int64_t t_max_phys_hits = 0;   // Pass-1 rail engagements
    // P-F1a (v7.2): the Pass-1 LOW rail's counter. The radiation fold is the
    // only SIGNED path into `temperature`, so it is the only one that can drive
    // a tile below the ambient floor at 0. The rail is a COUNTED DIAGNOSTIC,
    // justified inert by the per-term budget argument (every exchange term is
    // clamped to a |ΔT|/2^RAD_LIM_SHIFT share; the mutual branch halves it
    // again) — a hit inside a gate scenario is a RED, not a shrug. See the
    // block in temperature_solver.cpp Pass 1 for the full argument and for why
    // P-R4's "no low rail is needed" antisymmetry reasoning is void.
    mutable int64_t t_low_rail_hits = 0;   // Pass-1 LOW rail engagements

    // --- P-E2a ENERGY BOOKS (design §2.3, §5, §7) --------------------------
    // Every counter here is an int64 sum in RAW ENERGY counts (Q16.16 capacity
    // × Q16.16 temperature, i.e. the same unit the EOS books use), NOT a hit
    // count — R3's "all counted, in ENERGY units". They ACCUMULATE across
    // step() calls (the `t_max_phys_hits` idiom of this class; the ledger
    // diffs them per tick) and the CUDA twin folds its own totals into these
    // same fields, so telemetry is identical whichever backend ran.
    //
    // The conduction pass's exact ledger identity, asserted by
    // tests/test_temperature_conduction.py::test_conduction_energy_books_close
    // and by cuda_conduction_check PART 1:
    //
    //     Σ_cells ΔT_i · C_real_i  ==  e_cond_trunc_sum + e_cond_cap_sum
    //
    // because Σ_cells ΔE_i == 0 EXACTLY (constraint 1). Conduction's global
    // energy drift is therefore the counted floor terms and nothing else.
    mutable int64_t e_cond_trunc_sum = 0;  // endpoint floordiv residual (≤ 0: one-way)
    mutable int64_t e_cond_cap_sum   = 0;  // the capacity floor/ceiling term (signed)
    mutable int64_t cond_limit_hits  = 0;  // constraint-4 per-face limiter engagements

    // --- arc #54 P-G1b: THE GAS SIDE IS ENERGY NOW (design §2.7 row 3) -----
    //
    // When `gas_energy` is supplied, an ACCOUNTABLE gas cell no longer takes
    // the endpoint divide at all: Pass 1's deposit and Pass 2's conduction sum
    // are handed to the gas-energy seam (`gas_energy::deposit`) and the mirror
    // is re-read from the stored E. The SOLIDS side is untouched — thermal
    // solids are their own truth (D2) and keep the T-form law, its capacity
    // build, and `e_cond_trunc_sum` / `e_cond_cap_sum` exactly as before.
    //
    // `gas_energy == nullptr` (the direct-binding / unit-test path) keeps the
    // whole pre-#54 T-form law bit-identical, including the two counters above
    // — which is why tests/test_temperature_conduction.py's
    // `Σ ΔT_i·C_real_i == e_cond_trunc_sum + e_cond_cap_sum` identity survives
    // unchanged.
    //
    // THE STEP'S OWN CLOSURE IDENTITY (gated across ticks by
    // tests/test_e1_hot_rail.py::test_no_transport_mint):
    //
    //     Δ Σ_accountable gas_energy  ==  e_gas_deposit_sum
    //                                   + e_gas_cond_sum
    //                                   + e_gas_rail_sum
    //
    // exact in int64. Every one of these is a NET signed sum over accountable
    // cells, so a face that leaks energy into a ring / vacuum / solid cell
    // shows up as the export it is (that cell is not in the books) rather than
    // as an unexplained drift.
    mutable int64_t e_gas_deposit_sum = 0;  // Pass 1 heat->E on gas (net)
    mutable int64_t e_gas_cond_sum    = 0;  // Pass 2 conduction into gas E (net)
    mutable int64_t e_gas_rail_sum    = 0;  // Pass 1's T_MAX_PHYS rail (signed)
    // The three OPEN-BY-DESIGN channels, named as SIGNED per round-1 finding
    // L3-6. None of their LAWS changed at P-E2a — they are instrumented so
    // §7's "every creator named and counted" can actually be checked:
    //   * Pass 3 relaxes T toward 0 from BOTH sides, so it destroys energy
    //     above ambient and CREATES it below — a creator, not just a sink.
    //   * the Pass-0a vacuum wipe destroys the energy a breach vents (and
    //     creates, if it pins a sub-ambient cell up to 0).
    //   * the ambient-ring pin is the §5 boundary channel, bidirectional.
    // All three are priced at the cell's REAL capacity (unfloored), i.e. in
    // the same currency as the ledger's Σ N·T_abs estimator.
    mutable int64_t e_cool_sum      = 0;   // Pass 3 ambient cooling / sky (signed)
    mutable int64_t e_vac_wipe_sum  = 0;   // Pass 0a open-vacuum wipe (signed)
    mutable int64_t e_ring_pin_sum  = 0;   // Pass 0a ambient-ring pin (signed)
    // P-E2b (energy-books arc, design §2.2/§2.5, round-1 finding L3-7): the
    // Pass-1 v2.4 absorption-proportional deposit's (1-N)*deposit attenuation
    // drop below ambient density — PHYSICAL (absorptivity proportional to
    // density), stays, but never had a counter until now. Same currency as
    // `deposit`/`heat` (Q16.16, single power). One-way DESTRUCTION (N < FP_ONE
    // here, so >= 0 by construction) — accumulates across step() calls, the
    // t_max_phys_hits/t_low_rail_hits idiom of this class (never reset).
    mutable int64_t e_deposit_drop_sum = 0;

    void  set_gas_advection_rate(float v) { gas_advection_rate = v; }
    float get_gas_advection_rate() const { return gas_advection_rate; }
    void  set_c_v(float v) { c_v = v; }
    float get_c_v() const { return c_v; }
    void  set_n_floor_heat(float v) { n_floor_heat = v; }
    float get_n_floor_heat() const { return n_floor_heat; }

    // One tick of thermal work.
    //   Pass 0 — gas-T zero-at-vacuum + semi-Lagrangian advection on the
    //            open-air mask (NEW, P2 §4). Skipped (a clean no-op) when
    //            dt <= 0 or wind_x/wind_y are null.
    //   Pass 1 — heat -> temperature conversion (§1.2): solids via the
    //            UNCHANGED bit-shift; open-air (non-vacuum) cells via the NEW
    //            ΔT = ΔE/(N·c_v) reciprocal deposit (P2 §4.3).
    //   Pass 2 — conduction relaxation, gather + double-buffered. P-E2a: now
    //            in ENERGY form (see the header block above) — air<->air AND
    //            solid<->air both ride the same face-antisymmetric ΔE with the
    //            per-face limiter; each endpoint converts through its own
    //            capacity.
    //   Pass 3 — ambient cooling (§3), solids only, vacuum-exposure 1-bit.
    //            UNCHANGED: gas cells are structurally excluded (no decay).
    //
    //   temperature : Q16.16 int32, (h, w). Persistent field (ΔT above ambient;
    //                 T_ambient == 0, proposal §3.1). AMBIENT-RELATIVE for BOTH
    //                 solid and gas cells (P2 — do not rebase; see file header).
    //                 Mutated in place.
    //   heat        : Q16.16 int32, (h, w). Per-tick deposit from the ray pass.
    //                 Read NON-DESTRUCTIVELY (the caller clears it at end of tick,
    //                 after this and every other heat consumer).
    //   heat_inv_shift : int32, (h, w). Precomputed per-tile log2(thermal_mass)
    //                 cache (0..30). `heat >> heat_inv_shift` == heat /
    //                 thermal_mass, still Q16.16.
    //   face_shift  : int32, (h, w, 4). Per-tile face shift cache, dirs in fixed
    //                 order N,S,E,W. NO_FACE == grid edge or κ==0 either side ->
    //                 that face does not conduct. Baked at load from the
    //                 harmonic-mean face table, patched in on_tile_changed.
    //   solid       : bool, (h, w). The physics FLOW/obstacle mask
    //                 (permeability <= 0). Since the thermal-mass axis
    //                 (2026-07-30) it is read ONLY as the fallback when
    //                 `thermal_solid` is null — every per-medium thermal branch
    //                 keys on `thermal_solid` instead.
    //   thermal_solid : bool, (h, w). The per-medium THERMAL mask
    //                 (`thermal_mass > 0`; GameMap.thermal_solid). Conversion
    //                 and cooling run on THERMAL solids only; the P2 gas rules
    //                 run on the complementary open-air mask
    //                 (!thermal_solid && !is_vacuum). Nullable -> `solid`.
    //   is_vacuum   : bool, (h, w). The physics vacuum mask. A solid tile cools
    //                 at cool_shift_vacuum if ANY in-bounds 4-neighbour is vacuum
    //                 (§3.3). Same field the atmosphere/smoke solvers read. P2:
    //                 also zeroes gas-T at vacuum cells (Pass 0) and excludes
    //                 them from the open-air mask everywhere else.
    //   atmosphere  : int32 Q16.16, (h, w). The atmosphere field (S2c). A neighbour
    //                 with atmosphere < quantize(o2_vacuum_thresh) also counts as
    //                 vacuum-exposed — a pure integer compare (S3c: no float). P2:
    //                 ALSO read as the density proxy N for the gas radiation
    //                 deposit (Pass 1) — // P3: swap this read for the real
    //                 bulk-species N_total once P1/P3 land.
    //   wind_x/wind_y : Q16.16 int32, (h, w). The existing wind field (S2c) —
    //                 same field smoke/fire read. May be null (Pass 0 skipped)
    //                 for the Python direct-binding back-compat path.
    //   dt          : the tick's real elapsed seconds (== sim_time elsewhere).
    //                 <= 0 skips Pass 0 entirely (back-compat no-op).
    void step(
        int32_t* temperature,
        const int32_t* heat,
        const int32_t* heat_inv_shift,
        const int32_t* face_shift,
        const bool* solid,
        const bool* is_vacuum,
        const int32_t* atmosphere,
        const int32_t* n_bulk,   // EOS P3: real bulk N_total (nullable ->
                                  // atmosphere density-proxy fallback)
        const int32_t* wind_x,
        const int32_t* wind_y,
        int h, int w,
        float dt,
        // BC (boundary_conditions_spec_2026-07-19 §1, audit (b)): the ambient
        // ring is wiped to ΔT=0 in the Pass-0 pre-pass, the vacuum-breach idiom
        // verbatim (heat radiates to the T_amb sky). Default nullptr keeps the
        // space path AND the direct-binding test path byte-identical.
        const bool* is_ambient = nullptr,
        // THERMAL-MASS AXIS (docs/thermal_mass_axis_design_2026-07-25.md,
        // build addendum 2026-07-30): the per-medium THERMAL mask
        // (`thermal_mass > 0`, GameMap.thermal_solid) that replaces `solid` at
        // the six medium tests listed above. Default nullptr -> fall back to
        // `solid`, the pre-patch behaviour (the documented back-compat idiom
        // this signature already uses for wind/n_bulk/is_ambient).
        const bool* thermal_solid = nullptr,
        // COOL-SHIFT AXIS (2026-07-30): int32, (h, w). The per-tile AMBIENT
        // DECAY shift (`GameMap.cool_shift`, the per-material `cool_shift`
        // column projected by the material grid) — the LOSS-side twin of
        // `heat_inv_shift`. Pass 3 does `T -= T >> cool_shift_grid[i]`.
        // Default nullptr -> the scalar `cool_shift` member for every tile,
        // i.e. exactly the pre-axis single-global behaviour (the same
        // back-compat idiom as `thermal_solid` above). The vacuum-exposed
        // shift is derived from this same per-tile value by the global offset
        // documented at `cool_shift` — there is deliberately no second grid.
        const int32_t* cool_shift_grid = nullptr,
        // ---- P-R4 RADIATION (docs/radiation_raycaster_extinction_ruling_
        // 2026-07-31.md A1.7): the SIGNED per-tick radiation accumulator the
        // raycaster's net-T⁴ exchange writes. int32 Q16.16 heat counts, its own
        // plane (NOT `heat[]`) for one structural reason: `heat[]`'s adds are
        // POSITIVE-SATURATING, which is order-free only because positives are
        // monotone under a clamp; a SIGNED net under saturation is order-
        // DEPENDENT. `rad_net[]` therefore takes plain (wrapping) signed adds,
        // which ARE order-free — and it folds through a SIGNED conversion in
        // Pass 1 that the `deposit <= 0` skip must not gate, or an emitter's
        // radiative LOSS would silently never convert and fire could never cool
        // by radiating. Default nullptr -> no fold (every legacy caller and
        // every direct-binding test path stays byte-identical).
        const int32_t* rad_net = nullptr,
        // ---- arc #54 P-G1b (design §2.7 row 3): THE GAS SIDE IS ENERGY -----
        // `gas_energy` — the CONSERVED gas energy field, (h, w) int64, MUTATED
        // on ACCOUNTABLE gas cells (Pass 1's deposit and Pass 2's conduction
        // sum both go through the gas-energy seam instead of the endpoint
        // divide). nullptr keeps the entire pre-#54 T-form law bit-identical,
        // which is what leaves the direct-binding unit tests — and their
        // `Σ ΔT_i·C_real_i == e_cond_trunc_sum + e_cond_cap_sum` identity —
        // untouched. The SOLIDS side never changes either way (D2).
        // `t_amb_q` — T_AMB_K in raw Q16.16 counts, the absolute-temperature
        // offset the seam converts through; read only when gas_energy != null.
        // The engine folds it from the SAME temperature_scale accessor
        // EOSSolver's own fold reads, so the two cannot drift.
        int64_t* gas_energy = nullptr,
        int32_t t_amb_q = 0
    ) const;

    // --- DEBUG probe (temporary instrumentation, eos-p3fix-thermal-ceiling
    // investigation, decisions.md #16): T at ONE traced cell after Pass 2
    // (conduction) and after Pass 3 (ambient cooling). dbg_probe_idx = -1
    // disables (one branch/pass, no other cost). Raw Q16.16 counts.
    int dbg_probe_idx = -1;
    mutable int32_t dbg_T_post_heat       = 0;   // after Pass 1 (heat->T convert)
    mutable int32_t dbg_T_post_conduction = 0;
    mutable int32_t dbg_T_post_cooling    = 0;

private:
    // Double-buffer scratch for the conduction gather (temp -> temp_new). Owned
    // by the solver, resized on demand; reused across ticks (no per-tick alloc).
    // `mutable` so the const step() can use it as pure scratch.
    mutable std::vector<int32_t> scratch_;
    // P2: separate pre-advection snapshot scratch for the gas-T semi-Lagrangian
    // pass (Pass 0) — kept distinct from `scratch_` (the conduction double
    // buffer) so the two passes never alias each other's live data.
    mutable std::vector<int32_t> gas_scratch_;
    // P-E2a: the per-cell capacity planes, built ONCE per step() before any
    // pass runs (they depend only on frozen inputs — the medium mask, N and
    // the dials — never on T). `cap_used_` is what every conversion divides
    // by; `cap_real_` is the honest capacity the energy counters price with.
    // Transient scratch, never synced, never digested (R4).
    mutable std::vector<int64_t> cap_used_;
    mutable std::vector<int64_t> cap_real_;
    // arc #54 P-G1b: Pass 2's parked per-cell face sum for ACCOUNTABLE gas
    // cells. It exists because the seam refreshes the mirror, and the pass's
    // determinism rests on every cell reading the frozen pre-conduction field
    // — so the deposits are applied only after the double-buffer swap. Sized
    // (and zeroed) once per step, and only when `gas_energy` is supplied.
    mutable std::vector<int64_t> de_gas_;
};
