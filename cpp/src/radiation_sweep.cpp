// RadiationSweep — the directional sweep, CPU (ray-engine-v2 P1).
//
// Compiled /fp:strict (cpp/CMakeLists.txt) and pinned at 0/0/0 on the ratchet
// in tests/test_no_float_in_sim_tu.py: every quantity in this TU is an integer,
// and the two real-number type names that ratchet counts do not appear here —
// comments included. See radiation_sweep.h for the credits and the contract.
//
// THE SCHEME (design v3 §2.3, gather form). Per ordinate m, in wavefront
// order, for cell i (int64 throughout; ONE = 65536):
//
//   amb_m   = (amb_level[i] * w_m) >> 16    // THIS CELL'S ambient stream, NOT
//                                           // a hoisted global (thermal v2 R3)
//   fa      = (io_a * s_m) >> 16            // the straight upwind share
//   fb      = io_b - ((io_b * s_m) >> 16)   // the REMAINDER, never a second shift
//   i_in    = fa + fb                       // an out-of-grid upwind read is THIS
//                                           // cell's amb_m (the virtual ring)
//   leaked  = (i_in * k) >> 16 ;  ret = (amb_m * k) >> 16      // its own ceiling
//   stream  = i_in - leaked + ret
//   abs_mat = (stream * a) >> 16 ;  abs_body = (stream * b) >> 16      (b = d - a)
//   ex_m    = (ex_cell * w_m) >> 16      (>= 0; the excess over THIS cell's own
//                                         ambient, so amb_m + ex sums back to
//                                         E°[T_i] * w_m)
//   src     = amb_m + mul128_shr(ex_m, f_q24, 24)   // Fleck damps the EXCESS only
//             f_q24 = T_abs / max(T_abs, 4L)       // ALPHA FLOOR 0 (row 39):
//                                                  // f == 2^24 where g <= 1
//   emitted = (src * a) >> 16 ;   emit_body = (amb_m * b) >> 16   // its own
//   i_out   = stream - abs_mat - abs_body + emitted + emit_body
//   rad_net[i]     += abs_mat - emitted          // the material ledger, signed
//   rad_flux[i]    += abs_body - emit_body       // the body sensor, net above ambient
//   rad_amb[i]     += leaked - ret               // the ceiling channel (+ ring books)
//   rad_fluence[i] += stream                     // Φ, for the Pass-1 clamp
//
// where a, d are the EFFECTIVE planes (P5a, design §6.3): on a GAS cell the
// smoke term joins the material extinction, a = max(a, a_gas) with
// a_gas = min(ONE, Σ_g heat_absorb_q16[g]·max(0, N_g) >> 16) (0 below the
// N_EPS bulk floor), and d = max(d, a); a thermal solid keeps its own.
//
// A boundary cell additionally books into its own rad_amb every share of its
// i_out that leaves the grid (+) and every fa/fb it gathered from the virtual
// ring (−). CONSERVATION IS STRUCTURAL: every integer leaving the stream is
// added to a ledger as the same integer, the split carries its remainder, and
// the ring's two books close the boundary, so
//     sum(rad_net) + sum(rad_flux) + sum(rad_amb) == 0   exactly, in int64
// (the eos_solver face_flux idiom: one integer, applied twice with opposite
// signs). Gate 1 in tests/test_radiation_sweep_gates.py. A PER-CELL ambient
// does not touch that argument: every ambient-derived integer is still booked
// once with each sign, and whatever the ring hands a boundary cell is booked
// as it arrives — which is why WHICH cell's ambient the ring returns is a PIN
// (the reading cell's own; see sweep_ref_q.py::sweep_q and report_t1.md §2.1)
// rather than a correctness question.
//
// HEADROOM (critique 3 §2e, P0b G11): E°[3999] ~ 2^41.7; a stream never
// exceeds the largest upstream source by more than one count per cell, so
// every product here is below 2^58 and the widest one, ex_m * f_q24, is below
// 2^63 (measured 2^62.13 at S12) — and it goes through the kit's 128-bit
// mul128_shr anyway, so no headroom argument is load-bearing on it.

#include "radiation_sweep.h"

#include <algorithm>
#include <stdexcept>

namespace {

// ---- the checked-in per-ordinate constants (design §2.3, critique 3 §5b) --
// Half-offset ordinates, angle (m + 1/2)·2π/N, mu = cos, eta = sin. Computed
// ONCE by the integer reference's own ordinate_constants() (sweep_ref_q.py)
// and checked in as literals: never std::cos in a sim TU — a libm last-ULP
// difference across CRTs is engine/14's case 3, not door 2. The recompute
// test allows one count on s_m. NOTE the S12 diagonal ordinates (45°, 135°,
// ...): |mu| and |eta| differ only in the last ULP there, so x_major is the
// reference's own libm choice on the build machine — harmless, because s_m
// is 0 on those ordinates and with s_m == 0 the straight read contributes
// nothing and both traversal orders are topological for the pure-diagonal
// dependency, so the integers do not depend on x_major at all.
//   { sx, sy, x_major, s_m }
const OrdinateConst S16_SHEAR[16] = {
    {  1,  1, 1, 52500 },   // m= 0 mu=+0.980785 eta=+0.195090
    {  1,  1, 1, 21746 },   // m= 1 mu=+0.831470 eta=+0.555570
    {  1,  1, 0, 21746 },   // m= 2 mu=+0.555570 eta=+0.831470
    {  1,  1, 0, 52500 },   // m= 3 mu=+0.195090 eta=+0.980785
    { -1,  1, 0, 52500 },   // m= 4 mu=-0.195090 eta=+0.980785
    { -1,  1, 0, 21746 },   // m= 5 mu=-0.555570 eta=+0.831470
    { -1,  1, 1, 21746 },   // m= 6 mu=-0.831470 eta=+0.555570
    { -1,  1, 1, 52500 },   // m= 7 mu=-0.980785 eta=+0.195090
    { -1, -1, 1, 52500 },   // m= 8 mu=-0.980785 eta=-0.195090
    { -1, -1, 1, 21746 },   // m= 9 mu=-0.831470 eta=-0.555570
    { -1, -1, 0, 21746 },   // m=10 mu=-0.555570 eta=-0.831470
    { -1, -1, 0, 52500 },   // m=11 mu=-0.195090 eta=-0.980785
    {  1, -1, 0, 52500 },   // m=12 mu=+0.195090 eta=-0.980785
    {  1, -1, 0, 21746 },   // m=13 mu=+0.555570 eta=-0.831470
    {  1, -1, 1, 21746 },   // m=14 mu=+0.831470 eta=-0.555570
    {  1, -1, 1, 52500 },   // m=15 mu=+0.980785 eta=-0.195090
};
const OrdinateConst S16_STEP[16] = {
    {  1,  1, 1, 54663 },   // m= 0
    {  1,  1, 1, 39286 },   // m= 1
    {  1,  1, 1, 26250 },   // m= 2
    {  1,  1, 1, 10873 },   // m= 3
    { -1,  1, 1, 10873 },   // m= 4
    { -1,  1, 1, 26250 },   // m= 5
    { -1,  1, 1, 39286 },   // m= 6
    { -1,  1, 1, 54663 },   // m= 7
    { -1, -1, 1, 54663 },   // m= 8
    { -1, -1, 1, 39286 },   // m= 9
    { -1, -1, 1, 26250 },   // m=10
    { -1, -1, 1, 10873 },   // m=11
    {  1, -1, 1, 10873 },   // m=12
    {  1, -1, 1, 26250 },   // m=13
    {  1, -1, 1, 39286 },   // m=14
    {  1, -1, 1, 54663 },   // m=15
};
const OrdinateConst S12_SHEAR[12] = {
    {  1,  1, 1, 47976 },   // m= 0 mu=+0.965926 eta=+0.258819
    {  1,  1, 1,     0 },   // m= 1 the 45° diagonal (see the note above)
    {  1,  1, 0, 47976 },   // m= 2 mu=+0.258819 eta=+0.965926
    { -1,  1, 0, 47976 },   // m= 3
    { -1,  1, 0,     0 },   // m= 4 the 135° diagonal
    { -1,  1, 1, 47976 },   // m= 5
    { -1, -1, 1, 47976 },   // m= 6
    { -1, -1, 1,     0 },   // m= 7 the 225° diagonal
    { -1, -1, 0, 47976 },   // m= 8
    {  1, -1, 0, 47976 },   // m= 9
    {  1, -1, 0,     0 },   // m=10 the 315° diagonal
    {  1, -1, 1, 47976 },   // m=11
};
const OrdinateConst S12_STEP[12] = {
    {  1,  1, 1, 51687 },   // m= 0
    {  1,  1, 1, 32768 },   // m= 1
    {  1,  1, 1, 13849 },   // m= 2
    { -1,  1, 1, 13849 },   // m= 3
    { -1,  1, 1, 32768 },   // m= 4
    { -1,  1, 1, 51687 },   // m= 5
    { -1, -1, 1, 51687 },   // m= 6
    { -1, -1, 1, 32768 },   // m= 7
    { -1, -1, 1, 13849 },   // m= 8
    {  1, -1, 1, 13849 },   // m= 9
    {  1, -1, 1, 32768 },   // m=10
    {  1, -1, 1, 51687 },   // m=11
};

}  // namespace

const OrdinateConst* RadiationSweep::ordinate_table(int n_ordinates, int transport) {
    if (n_ordinates == 16) {
        if (transport == TRANSPORT_SHEAR) return S16_SHEAR;
        if (transport == TRANSPORT_STEP)  return S16_STEP;
        return nullptr;
    }
    if (n_ordinates == 12) {
        if (transport == TRANSPORT_SHEAR) return S12_SHEAR;
        if (transport == TRANSPORT_STEP)  return S12_STEP;
        return nullptr;
    }
    return nullptr;
}

void RadiationSweep::size_scratch_(int h, int w, int n_ordinates) const {
    const int n = h * w;
    if (h_ != h || w_ != w || n_ord_ != n_ordinates) {
        outflow_.assign((size_t)n_ordinates * (size_t)n, 0);
        ex_cell_.assign((size_t)n, 0);
        amb_m_.assign((size_t)n, 0);
        f_q24_.assign((size_t)n, F_ONE);
        a_eff_.assign((size_t)n, 0);
        d_eff_.assign((size_t)n, 0);
        h_ = h; w_ = w; n_ord_ = n_ordinates;
    }
}

int32_t* RadiationSweep::fleck_plane_for_twin(int h, int w, int n_ordinates) const {
    size_scratch_(h, w, n_ordinates);
    return f_q24_.data();
}

const int64_t* RadiationSweep::derive_ambient(const bool* is_vacuum,
                                             const int64_t* e_table,
                                             int64_t vac_level, int n) const {
    const int64_t e0 = e_table[0];
    if (vac_level > e0) {
        throw std::invalid_argument(
            "RadiationSweep::derive_ambient: the vacuum ambient level exceeds "
            "E°[0]; the per-cell ambient invariant is 0 <= amb <= E°[0], which "
            "is what keeps every cell's emission excess non-negative");
    }
    const int64_t vac = (vac_level < 0) ? e0 : vac_level;   // R4: space is room temp
    amb_derived_.resize((size_t)n);
    for (int i = 0; i < n; ++i) amb_derived_[i] = is_vacuum[i] ? vac : e0;
    return amb_derived_.data();
}

void RadiationSweep::run(const int32_t* temperature,
                         const int32_t* heat_atten_q, const int32_t* dyn_heat_atten_q,
                         const int32_t* heat_inv_shift, const bool* thermal_solid,
                         const int64_t* e_table, const int64_t* amb_level,
                         int32_t t_amb_q, int32_t k_leak_q,
                         int transport, int n_ordinates, int h, int w,
                         int64_t* rad_net, int64_t* rad_flux, int64_t* rad_amb,
                         int64_t* rad_fluence, bool fleck_enabled,
                         const int32_t* gas, int n_gases,
                         const int32_t* heat_absorb_q16,
                         const int32_t* n_bulk) const {
    using fixedpoint::FP_ONE;
    using fixedpoint::FP_SHIFT;

    const OrdinateConst* tbl = ordinate_table(n_ordinates, transport);
    if (tbl == nullptr) {
        throw std::invalid_argument(
            "RadiationSweep::run: unsupported (n_ordinates, transport) — "
            "n_ordinates must be 16 or 12, transport TRANSPORT_STEP (0) or "
            "TRANSPORT_SHEAR (1)");
    }
    if (k_leak_q < 0 || k_leak_q > FP_ONE) {
        throw std::invalid_argument(
            "RadiationSweep::run: k_leak_q outside [0, ONE] (design §2.3 invariant)");
    }
    if (amb_level == nullptr) {
        throw std::invalid_argument(
            "RadiationSweep::run: amb_level is null — every caller states the "
            "ambient it radiates against (thermal v2 R3); derive_ambient() "
            "builds the uniform R4 plane for a caller that has no opinion");
    }
    // ---- THE GAS EXTINCTION's ingress (P5a) ---------------------------------
    // The group is all-or-nothing, and the two bounds are the gas door's own
    // (gases.py): with n_gases <= 16 and every heat_absorb_q16 in [0, 2^28],
    // each density term is < 2^59 and the per-cell sum is exact in int64 for
    // ANY int32 density — the one argument the plain int64 sum below rests on.
    const bool gas_given = (gas != nullptr);
    if (gas_given != (heat_absorb_q16 != nullptr) || gas_given != (n_bulk != nullptr)) {
        throw std::invalid_argument(
            "RadiationSweep::run: gas, heat_absorb_q16 and n_bulk are given "
            "together or not at all (the smoke term, design v3 §6.3)");
    }
    if (n_gases < 0 || n_gases > N_GAS_PLANES_MAX || (!gas_given && n_gases != 0)) {
        throw std::invalid_argument(
            "RadiationSweep::run: n_gases outside [0, N_GAS_PLANES_MAX] — the "
            "gas density sum's int64 headroom is argued for at most 16 planes");
    }
    // The ACTIVE gases: a zero heat_absorb contributes exactly 0 to the sum, so
    // its plane is never read (every shipped row is 0.0 until P5b — the live
    // sweep reads no gas plane at all, and its cost does not move).
    int act_g[N_GAS_PLANES_MAX];
    int32_t act_hq[N_GAS_PLANES_MAX];
    int n_act = 0;
    for (int g = 0; g < n_gases; ++g) {
        const int32_t hq = heat_absorb_q16[g];
        if (hq < 0 || hq > HEAT_ABSORB_Q_MAX) {
            throw std::invalid_argument(
                "RadiationSweep::run: a heat_absorb_q16 outside [0, 2^28] (the "
                "gas door's [0, 4096]; negative would be a source, above it the "
                "density sum is no longer provably exact in int64)");
        }
        if (hq != 0) { act_g[n_act] = g; act_hq[n_act] = hq; ++n_act; }
    }

    const int n = h * w;
    size_scratch_(h, w, n_ordinates);

    // ---- the per-ordinate constants (door 1) ------------------------------
    // `amb_m` and `ret` were hoisted here while the ambient was one global
    // number. They are PER-CELL now (thermal v2 R3), computed in the pre-pass
    // below into amb_m_ and read inside the cell loop: a hoist is exactly the
    // bug thermal v2 §6 item 3 exists to make impossible.
    const int64_t e0    = e_table[0];
    const int64_t w_m   = (int64_t)FP_ONE / n_ordinates;   // 4096 at S16, 5461 at S12
    const int64_t k     = k_leak_q;
    const int64_t t_amb = t_amb_q;

    // ---- the pre-pass: the excess emission and the Fleck factor (§2.8) -----
    // Also the ingress re-check 0 <= a <= d <= ONE, the invariant positivity
    // rests on (abs_mat + abs_body <= stream because truncation only decreases
    // and a + b <= ONE).
    for (int i = 0; i < n; ++i) {
        const int64_t a = heat_atten_q[i];
        const int64_t d = dyn_heat_atten_q[i];
        if (a < 0 || a > d || d > FP_ONE) {
            throw std::invalid_argument(
                "RadiationSweep::run: extinction planes violate 0 <= a <= d <= ONE "
                "(heat_atten_q / dyn_heat_atten_q ingress invariant, design §2.3)");
        }
        // The ambient invariant (thermal v2 R3). The upper bound is what makes
        // every cell's excess non-negative: e_bucket_of floors at bucket 0, so
        // E°[T] >= E°[0] >= amb_level[i] for EVERY temperature. The excess-form
        // Fleck factor and the sweep's positivity both rest on that.
        const int64_t amb_i = amb_level[i];
        if (amb_i < 0 || amb_i > e0) {
            throw std::invalid_argument(
                "RadiationSweep::run: amb_level outside [0, E°[0]] (the per-cell "
                "ambient invariant, thermal model v2 R3)");
        }
        amb_m_[i] = (amb_i * w_m) >> FP_SHIFT;
        int64_t ex = e_table[e_bucket_of(temperature[i])] - amb_i;
        if (ex < 0) ex = 0;                 // dead under the invariant above; belt and braces
        ex_cell_[i] = ex;
        // THE EFFECTIVE EXTINCTION (P5a, design §6.3): a GAS cell takes the
        // smoke term — the density law over the active gases, the N_EPS floor
        // on the BULK count — as a MAX with its material extinction; the
        // stamped total follows as a MAX. A thermal solid keeps its own.
        // Transcribes sweep_ref_q.py::gas_extinction_plane / effective_extinction.
        int32_t a_eff = heat_atten_q[i];
        int32_t d_eff = dyn_heat_atten_q[i];
        if (n_act > 0 && !thermal_solid[i]) {
            int64_t sum = 0;
            for (int j = 0; j < n_act; ++j) {
                sum += gas_density_term(act_hq[j], gas[(size_t)act_g[j] * (size_t)n + (size_t)i]);
            }
            a_eff = gas_effective_a(a_eff, gas_extinction_finish(sum, n_bulk[i]), false);
            d_eff = gas_effective_d(d_eff, a_eff);
        }
        a_eff_[i] = a_eff;
        d_eff_[i] = d_eff;
        int64_t L = 0;
        if (thermal_solid[i]) {
            L = fleck_L_solid_q(ex, heat_atten_q[i], heat_inv_shift[i]);
        } else {
            // GAS: the arm is still L = 0, so f is exactly 2^24 — also on a
            // gas cell that ABSORBS through the smoke term since P5a. P5b fills
            // it (design §2.8 / §6.3), reference first: the same excess through
            // the STAGED wide chain, fixedpoint::deposit_dT_wide_i64(
            // (a_eff·ex) >> 16, recip_n, recip_cv) — two narrows, declared to
            // differ from the heat deposit's one-narrow chain by at most one
            // LSB (sweep_ref_q.py::fleck_L_gas_q is that chain, transcribed and
            // measured by p5a_gas_stiffness_study.py, and called by no pre-pass).
            L = 0;
        }
        // T_abs > 0 always in the engine (T_MIN = -292 game keeps T_abs >= 1);
        // the floor at 1 is the runner's own A7 floor, so a direct caller with
        // an absurd offset cannot divide by zero here.
        int64_t T_abs = (int64_t)temperature[i] + t_amb;
        if (T_abs < 1) T_abs = 1;
        // fleck_enabled == false is the reference's `f_plane=None` (undamped)
        // configuration, for the gates measured in it; the engine passes true.
        f_q24_[i] = fleck_enabled ? fleck_f_q24(T_abs, L) : F_ONE;
    }

    // ---- OVERWRITE, not accumulate: run() clears its own four outputs -----
    // The per-ordinate books below are `+=`, so the first ordinate needs clean
    // planes. P1 got them from the conductor's end-of-tick wipe, which also
    // BLINDED the tile inspector: `rad_fluence` was zero again by the time the
    // renderer read it (design v3 row 38). The wipe lives here from P2a and the
    // four `fill(0)` lines are gone from Simulation.step, so the planes hold
    // THIS run's values until the next run overwrites them — which is what the
    // render-time readout needs and what the digest does not care about (none
    // of the four is in DIGEST_FIELDS or SIM_FIELDS). Cleared AFTER the
    // pre-pass's ingress check, so a REJECTED scene leaves the caller's planes
    // exactly as it found them.
    for (int i = 0; i < n; ++i) {
        rad_net[i] = 0;
        rad_flux[i] = 0;
        rad_amb[i] = 0;
        rad_fluence[i] = 0;
    }

    // ---- the sweep, one ordinate after another (CPU) ----------------------
    int64_t min_s = 0, max_s = 0;
    bool first = true;
    const bool shear = (transport == TRANSPORT_SHEAR);
    for (int m = 0; m < n_ordinates; ++m) {
        const OrdinateConst oc = tbl[m];
        int64_t* store = outflow_.data() + (size_t)m * (size_t)n;
        const int sx = oc.sx;
        const int sy = oc.sy;
        const bool xm = (oc.x_major != 0);
        const int64_t s_m = oc.s_m;
        // The two UPWIND reads as (dy, dx) offsets — a: the straight one
        // (which fa is recomputed from), b: the other (which carries the
        // remainder fb). The DOWNWIND targets are the same offsets negated.
        //   shear, x-major:  a = (y, x-sx)   b = (y-sy, x-sx)
        //   shear, y-major:  a = (y-sy, x)   b = (y-sy, x-sx)
        //   step:            a = (y, x-sx)   b = (y-sy, x)
        int ady, adx, bdy, bdx;
        if (shear) {
            if (xm) { ady = 0;   adx = -sx; bdy = -sy; bdx = -sx; }
            else    { ady = -sy; adx = 0;   bdy = -sy; bdx = -sx; }
        } else {
            ady = 0; adx = -sx; bdy = -sy; bdx = 0;
        }

        auto visit = [&](int y, int x) {
            const int i = y * w + x;
            const int uay = y + ady, uax = x + adx;
            const int uby = y + bdy, ubx = x + bdx;
            const bool in_a = (uay >= 0 && uay < h && uax >= 0 && uax < w);
            const bool in_b = (uby >= 0 && uby < h && ubx >= 0 && ubx < w);
            // THIS cell's ambient stream. The virtual ring returns the READING
            // cell's own value: the ring cell is outside the grid and has none
            // of its own, and this is the only choice that keeps a boundary
            // cell at its own ambient an exact fixed point (report_t1.md §2.1).
            const int64_t amb_m = amb_m_[i];
            const int64_t io_a = in_a ? store[uay * w + uax] : amb_m;   // the virtual ring
            const int64_t io_b = in_b ? store[uby * w + ubx] : amb_m;
            // The split, recomputed from the stored outflow with the SAME shift
            // the pusher would have used; fb is the remainder (critique 1 fix 3).
            const int64_t fa = (io_a * s_m) >> FP_SHIFT;
            const int64_t fb = io_b - ((io_b * s_m) >> FP_SHIFT);
            const int64_t i_in = fa + fb;
            if (!in_a) rad_amb[i] -= fa;          // taken from the ring: booked
            if (!in_b) rad_amb[i] -= fb;

            const int64_t leaked = (i_in * k) >> FP_SHIFT;
            const int64_t ret    = (amb_m * k) >> FP_SHIFT;   // ITS OWN ceiling
            const int64_t stream = i_in - leaked + ret;
            const int64_t a = a_eff_[i];                        // material + smoke (P5a)
            const int64_t b = (int64_t)d_eff_[i] - a;           // the body share
            const int64_t abs_mat  = (stream * a) >> FP_SHIFT;
            const int64_t abs_body = (stream * b) >> FP_SHIFT;
            const int64_t ex_m = (ex_cell_[i] * w_m) >> FP_SHIFT;
            const int64_t src  = amb_m + fixedpoint::mul128_shr(ex_m, (int64_t)f_q24_[i], F_SHIFT);
            const int64_t emitted   = (src * a) >> FP_SHIFT;        // the SAME arithmetic as abs_mat
            const int64_t emit_body = (amb_m * b) >> FP_SHIFT;      // a body re-emits AMBIENT (row 25)
            const int64_t i_out = stream - abs_mat - abs_body + emitted + emit_body;

            rad_net[i]     += abs_mat - emitted;
            rad_flux[i]    += abs_body - emit_body;
            rad_amb[i]     += leaked - ret;
            rad_fluence[i] += stream;
            store[i] = i_out;

            // What this cell's own outflow sends into the virtual ring.
            const int64_t fa_o = (i_out * s_m) >> FP_SHIFT;
            const int64_t fb_o = i_out - fa_o;
            const int tay = y - ady, tax = x - adx;
            const int tby = y - bdy, tbx = x - bdx;
            if (!(tay >= 0 && tay < h && tax >= 0 && tax < w)) rad_amb[i] += fa_o;
            if (!(tby >= 0 && tby < h && tbx >= 0 && tbx < w)) rad_amb[i] += fb_o;

            if (first) { min_s = stream; max_s = stream; first = false; }
            else {
                if (stream < min_s) min_s = stream;
                if (stream > max_s) max_s = stream;
            }
        };

        // Wavefront order: a topological order of the per-ordinate DAG. For
        // shear the dependency is a whole column (x-major) or row (y-major);
        // for step an anti-diagonal — and any row-major walk in the ordinate's
        // direction is topological for it (critique 3 §2a). The gather form
        // makes the integers independent of WHICH valid order is used.
        const int x_begin = (sx > 0) ? 0 : w - 1;
        const int x_end   = (sx > 0) ? w : -1;
        const int y_begin = (sy > 0) ? 0 : h - 1;
        const int y_end   = (sy > 0) ? h : -1;
        if (shear && xm) {
            for (int x = x_begin; x != x_end; x += sx)
                for (int y = y_begin; y != y_end; y += sy) visit(y, x);
        } else {
            for (int y = y_begin; y != y_end; y += sy)
                for (int x = x_begin; x != x_end; x += sx) visit(y, x);
        }
    }
    min_stream = min_s;
    max_stream = max_s;
}
