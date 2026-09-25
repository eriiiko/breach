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
// N_EPS bulk floor), and d = max(d, a); a thermal solid keeps its own. The
// Fleck factor's L has TWO ARMS: a thermal solid's material emission through
// its heat_inv_shift, and (P5b) a gas cell's smoke-term emission through the
// temperature fold's own gas currency (fleck_L_gas_q, radiation_sweep.h).
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
//
// THE CURRENCY (#78). Every integer above is in the E° table's currency,
// `e_fine_bits` (emissive_table.h): the live table is baked 2^E_FINE_BITS finer
// than one heat count, so near ambient the per-ordinate terms carry thousands
// of counts instead of single digits and the floors stop deciding the exchange.
// The sweep's own arithmetic is unit-agnostic and unchanged; the pre-pass is
// the one place here that CONVERTS (the Fleck L is a temperature), through the
// kit's fine_heat_shr / the gas chain's final narrow. At the fine live scale the
// bounds above hold with room (sweep_ref_q_gates G11: E°[3999] = 2^41.1, plain
// products <= 2^57.1 (the pre-pass's a * ex), per-cell sums <= 2^41.1, the
// 128-bit Fleck product 2^61.5).

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

// P6a: the per-ordinate DIRECTION COSINES in Q16, for the light flux vector.
// Computed ONCE by the integer reference's own ordinate_dirs() (quant of the
// half-offset cos / sin) and checked in; tests/test_radiation_sweep_constants.py
// recomputes them within one count and holds their signs to the transport
// tables' sx / sy, ordinate by ordinate.
//   { mu_q, eta_q }
const OrdinateDir S16_DIRS[16] = {
    {  64277,  12785 },   // m= 0 mu=+0.980785 eta=+0.195090
    {  54491,  36410 },   // m= 1 mu=+0.831470 eta=+0.555570
    {  36410,  54491 },   // m= 2 mu=+0.555570 eta=+0.831470
    {  12785,  64277 },   // m= 3 mu=+0.195090 eta=+0.980785
    { -12785,  64277 },   // m= 4 mu=-0.195090 eta=+0.980785
    { -36410,  54491 },   // m= 5 mu=-0.555570 eta=+0.831470
    { -54491,  36410 },   // m= 6 mu=-0.831470 eta=+0.555570
    { -64277,  12785 },   // m= 7 mu=-0.980785 eta=+0.195090
    { -64277, -12785 },   // m= 8 mu=-0.980785 eta=-0.195090
    { -54491, -36410 },   // m= 9 mu=-0.831470 eta=-0.555570
    { -36410, -54491 },   // m=10 mu=-0.555570 eta=-0.831470
    { -12785, -64277 },   // m=11 mu=-0.195090 eta=-0.980785
    {  12785, -64277 },   // m=12 mu=+0.195090 eta=-0.980785
    {  36410, -54491 },   // m=13 mu=+0.555570 eta=-0.831470
    {  54491, -36410 },   // m=14 mu=+0.831470 eta=-0.555570
    {  64277, -12785 },   // m=15 mu=+0.980785 eta=-0.195090
};
const OrdinateDir S12_DIRS[12] = {
    {  63303,  16962 },   // m= 0 mu=+0.965926 eta=+0.258819
    {  46341,  46341 },   // m= 1 the 45° diagonal
    {  16962,  63303 },   // m= 2 mu=+0.258819 eta=+0.965926
    { -16962,  63303 },   // m= 3
    { -46341,  46341 },   // m= 4 the 135° diagonal
    { -63303,  16962 },   // m= 5
    { -63303, -16962 },   // m= 6
    { -46341, -46341 },   // m= 7 the 225° diagonal
    { -16962, -63303 },   // m= 8
    {  16962, -63303 },   // m= 9
    {  46341, -46341 },   // m=10 the 315° diagonal
    {  63303, -16962 },   // m=11
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

const OrdinateDir* RadiationSweep::ordinate_dirs(int n_ordinates) {
    if (n_ordinates == 16) return S16_DIRS;
    if (n_ordinates == 12) return S12_DIRS;
    return nullptr;
}

void RadiationSweep::size_light_scratch_(int h, int w, int n_ordinates) const {
    const size_t n3 = (size_t)h * (size_t)w * 3;
    if (lh_ != h || lw_ != w || ln_ord_ != n_ordinates) {
        l_outflow_.assign((size_t)n_ordinates * n3, 0);
        l_emit_.assign(n3, 0);
        l_a_.assign(n3, 0);
        l_d_.assign(n3, 0);
        l_g_.assign(n3, 0);
        lh_ = h; lw_ = w; ln_ord_ = n_ordinates;
    }
}

void RadiationSweep::set_light_telemetry_from_twin(const int64_t* books12,
                                                   int64_t min_s, int64_t max_s) const {
    for (int c = 0; c < 3; ++c) {
        light_emit_sum[c]     = books12[c];
        light_absorb_sum[c]   = books12[3 + c];
        light_ring_in_sum[c]  = books12[6 + c];
        light_ring_out_sum[c] = books12[9 + c];
    }
    light_min_stream = min_s;
    light_max_stream = max_s;
}

void RadiationSweep::validate_light_group(const LightChannels& light, int n_ordinates,
                                          const int32_t* gas, const int32_t* n_bulk,
                                          int n_gases) {
    if (light.light_atten_q == nullptr || light.dyn_light_atten_q == nullptr ||
        light.l_table == nullptr || light.light_q == nullptr ||
        light.light_flux_q == nullptr || light.light_glow == nullptr) {
        throw std::invalid_argument(
            "RadiationSweep: the light group needs its two extinction planes, the "
            "L° table and its three output planes (P6a)");
    }
    if (ordinate_table(n_ordinates, light.transport) == nullptr) {
        throw std::invalid_argument(
            "RadiationSweep: unsupported light transport -- TRANSPORT_STEP (0) or "
            "TRANSPORT_SHEAR (1)");
    }
    // THE DARK AMBIENT (decision 3): bucket 0 is where every cell at or below
    // ambient reads, and the body share re-emits that level -- a table that is
    // not dark there would make a room of ambient cells, and every marine, glow.
    for (int c = 0; c < 3; ++c) {
        if (light.l_table[(size_t)c * E_TABLE_SIZE] != 0) {
            throw std::invalid_argument(
                "RadiationSweep: the light table's bucket 0 must be 0 on every "
                "channel (L°[0]: a room-temperature body emits no visible light)");
        }
    }
    const bool lab = (light.light_absorb_q16 != nullptr);
    if (lab != (light.light_glow_q16 != nullptr)) {
        throw std::invalid_argument(
            "RadiationSweep: light_absorb_q16 and light_glow_q16 are given "
            "together or not at all (the light smoke term)");
    }
    if (lab) {
        if (gas == nullptr || n_bulk == nullptr) {
            throw std::invalid_argument(
                "RadiationSweep: the light smoke term reads the gas group (gas, "
                "n_bulk) -- give it");
        }
        for (int i = 0; i < n_gases * 3; ++i) {
            const int32_t a = light.light_absorb_q16[i];
            const int32_t g = light.light_glow_q16[i];
            if (a < 0 || a > HEAT_ABSORB_Q_MAX || g < 0 || g > HEAT_ABSORB_Q_MAX) {
                throw std::invalid_argument(
                    "RadiationSweep: a light_absorb_q16 / light_glow_q16 outside "
                    "[0, 2^28] (the gas door's bound: the density sum's int64 "
                    "headroom; negative would be a source)");
            }
        }
    }
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
                         const int64_t* e_table, int e_fine_bits,
                         const int64_t* amb_level,
                         int32_t t_amb_q, int32_t k_leak_q,
                         int transport, int n_ordinates, int h, int w,
                         int64_t* rad_net, int64_t* rad_flux, int64_t* rad_amb,
                         int64_t* rad_fluence, bool fleck_enabled,
                         const int32_t* gas, int n_gases,
                         const int32_t* heat_absorb_q16,
                         const int32_t* n_bulk,
                         int32_t n_floor_q, int64_t recip_cv,
                         const LightChannels* light) const {
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
    // #78: the table's CURRENCY. The sweep books everything in it; the pre-pass
    // converts out of it (both Fleck arms). [0, E_FINE_BITS] is what the int64
    // headroom argument covers (emissive_table.h) — the bake refuses the same.
    if (e_fine_bits < 0 || e_fine_bits > E_FINE_BITS) {
        throw std::invalid_argument(
            "RadiationSweep::run: e_fine_bits outside [0, E_FINE_BITS] -- the "
            "table's currency (#78, emissive_table.h)");
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
    // P5b: the gas arm's CURRENCY rides with the group. Both are the fold's
    // own positive integers (n_floor_heat > 0, c_v > 0); a zero here is a
    // caller that forgot them, and it would price every gas cell's L at 0 —
    // the arm silently undamped — so it is refused, as the reference's
    // validate_gas_capacity refuses it.
    if (gas_given && (n_floor_q <= 0 || recip_cv <= 0)) {
        throw std::invalid_argument(
            "RadiationSweep::run: the gas group needs the temperature fold's gas "
            "currency — n_floor_q (quantize(n_floor_heat)) and recip_cv "
            "(make_recip(c_v_q / 65536)) must both be > 0 (design v3 §2.8's gas "
            "arm; PhysicsEngine::gas_capacity_q() on the live path)");
    }
    // The ACTIVE gases: a zero heat_absorb contributes exactly 0 to the sum, so
    // its plane is never read (since P5c the live sweep reads exactly one, the
    // smoke plane: every other shipped row is 0.0).
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

    // ---- THE LIGHT GROUP's ingress (P6a) --------------------------------------
    // Everything but the per-cell extinction check, before any work: the group
    // is complete, the transport known, the table dark at bucket 0, the smoke
    // columns whole and inside the gas door. Its ACTIVE gases are its own (a gas
    // with any non-zero light coefficient) -- not the heat term's: soot is the
    // one shipped heat absorber, but every trace gas tints or glows.
    const bool light_on = (light != nullptr);
    int l_act_g[N_GAS_PLANES_MAX];
    int n_lact = 0;
    const OrdinateConst* ltbl = nullptr;
    const OrdinateDir* dirs = nullptr;
    if (light_on) {
        validate_light_group(*light, n_ordinates, gas, n_bulk, n_gases);
        ltbl = ordinate_table(n_ordinates, light->transport);
        dirs = ordinate_dirs(n_ordinates);
        if (light->light_absorb_q16 != nullptr) {
            for (int g = 0; g < n_gases; ++g) {
                bool any = false;
                for (int c = 0; c < 3; ++c) {
                    any = any || light->light_absorb_q16[g * 3 + c] != 0 ||
                          light->light_glow_q16[g * 3 + c] != 0;
                }
                if (any) l_act_g[n_lact++] = g;
            }
        }
    }

    const int n = h * w;
    size_scratch_(h, w, n_ordinates);
    if (light_on) size_light_scratch_(h, w, n_ordinates);

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
        int32_t a_gas = 0;                  // the smoke term's own extinction
        if (n_act > 0 && !thermal_solid[i]) {
            int64_t sum = 0;
            for (int j = 0; j < n_act; ++j) {
                sum += gas_density_term(act_hq[j], gas[(size_t)act_g[j] * (size_t)n + (size_t)i]);
            }
            a_gas = gas_extinction_finish(sum, n_bulk[i]);
            a_eff = gas_effective_a(a_eff, a_gas, false);
            d_eff = gas_effective_d(d_eff, a_eff);
        }
        a_eff_[i] = a_eff;
        d_eff_[i] = d_eff;
        int64_t L = 0;
        if (thermal_solid[i]) {
            // #78: ex is in the table's currency; the arm converts once
            // (fixedpoint::fine_heat_shr at his, inside fleck_L_solid_q).
            L = fleck_L_solid_q(ex, heat_atten_q[i], heat_inv_shift[i], e_fine_bits);
        } else if (a_gas > 0) {
            // THE GAS ARM (P5b, design §2.8 / §6.3; sweep_ref_q.py's
            // fleck_prepass): a gas cell whose smoke term absorbs is damped on
            // that term's excess, priced in the temperature fold's own gas
            // currency — N floored at n_floor_q, reciprocal_q16 per cell,
            // recip_cv — through §2.8's staged chain. a_gas > 0 implies an
            // active gas and a bulk count >= N_EPS_RAW, so n_bulk is live here.
            // A gas cell that absorbs nothing emits no excess and keeps
            // L = 0, f == 2^24 — which is every smoke-free gas cell of the
            // shipped game (only smoke absorbs, since P5c).
            L = fleck_L_gas_q(ex, a_gas, n_bulk[i], n_floor_q, recip_cv, e_fine_bits);
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

    // ---- THE LIGHT PRE-PASS (P6a): per cell, per channel -------------------------
    // The ingress check 0 <= a_c <= d_c <= ONE on the INPUT planes; the smoke
    // term on a gas cell, through the heat term's own FP_HD pieces (the density
    // law, the N_EPS floor, the two MAXes); the glow coefficient (the same law
    // over the albedo column, gas cells only); and the cell's per-ordinate
    // emission, formed ONCE -- w_m is uniform, so it is the same integer in every
    // ordinate (sweep_ref_q.py forms it per ordinate; the value is one). The
    // emission book is booked here, n_ordinates times that integer.
    int64_t l_emit_tot[3] = {0, 0, 0};
    if (light_on) {
        const int32_t* la_in = light->light_atten_q;
        const int32_t* ld_in = light->dyn_light_atten_q;
        const int64_t* ltab = light->l_table;
        for (int i = 0; i < n; ++i) {
            const int b = e_bucket_of(temperature[i]);
            const bool gas_cell = (n_lact > 0 && !thermal_solid[i]);
            for (int c = 0; c < 3; ++c) {
                const size_t ic = (size_t)i * 3 + (size_t)c;
                int32_t a = la_in[ic];
                int32_t d = ld_in[ic];
                if (a < 0 || a > d || d > FP_ONE) {
                    throw std::invalid_argument(
                        "RadiationSweep::run: the light extinction planes violate "
                        "0 <= a <= d <= ONE on some channel (light_atten_q / "
                        "dyn_light_atten_q ingress invariant, P6a)");
                }
                int32_t g_c = 0;
                if (gas_cell) {
                    int64_t sa = 0, sg = 0;
                    for (int j = 0; j < n_lact; ++j) {
                        const int g = l_act_g[j];
                        const int32_t dens = gas[(size_t)g * (size_t)n + (size_t)i];
                        sa += gas_density_term(light->light_absorb_q16[g * 3 + c], dens);
                        sg += gas_density_term(light->light_glow_q16[g * 3 + c], dens);
                    }
                    const int32_t a_gas = gas_extinction_finish(sa, n_bulk[i]);
                    a = gas_effective_a(a, a_gas, false);
                    d = gas_effective_d(d, a);
                    g_c = gas_extinction_finish(sg, n_bulk[i]);
                }
                l_a_[ic] = a;
                l_d_[ic] = d;
                l_g_[ic] = g_c;
                const int64_t src = (ltab[(size_t)c * E_TABLE_SIZE + (size_t)b] * w_m) >> FP_SHIFT;
                const int64_t em = (src * (int64_t)a) >> FP_SHIFT;
                l_emit_[ic] = em;
                l_emit_tot[c] += em;
            }
        }
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
    // P6a: the light group's three outputs join that zeroing (design row 38:
    // a new output plane on this sweep is zeroed HERE, never by a caller).
    if (light_on) {
        for (size_t i = 0; i < (size_t)n * 3; ++i) {
            light->light_q[i] = 0;
            light->light_glow[i] = 0;
        }
        for (size_t i = 0; i < (size_t)n * 2; ++i) light->light_flux_q[i] = 0;
    }

    // ---- the sweep, one ordinate after another (CPU) ----------------------
    int64_t min_s = 0, max_s = 0;
    bool first = true;
    // P6a: the light books and stream telemetry, accumulated locally and
    // published at the end (a rejected scene never reaches here).
    int64_t l_absorb[3] = {0, 0, 0}, l_ring_in[3] = {0, 0, 0}, l_ring_out[3] = {0, 0, 0};
    int64_t lmin_s = 0, lmax_s = 0;
    bool lfirst = true;
    const bool shear = (transport == TRANSPORT_SHEAR);
    const bool lshear = light_on && (light->transport == TRANSPORT_SHEAR);
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
        // P6a: LIGHT's constants for THIS ordinate -- the same direction (the
        // tables share sx, sy at every m), its own transport's pair and split,
        // and the direction cosines for the flux vector. Every walk below is
        // topological for either transport (the note at the walk).
        int lady = 0, ladx = 0, lbdy = 0, lbdx = 0;
        int64_t ls_m = 0, mu_q = 0, eta_q = 0;
        int64_t* lstore = nullptr;
        if (light_on) {
            const OrdinateConst loc = ltbl[m];
            ls_m = loc.s_m;
            if (lshear) {
                if (loc.x_major != 0) { lady = 0;   ladx = -sx; lbdy = -sy; lbdx = -sx; }
                else                  { lady = -sy; ladx = 0;   lbdy = -sy; lbdx = -sx; }
            } else {
                lady = 0; ladx = -sx; lbdy = -sy; lbdx = 0;
            }
            mu_q = dirs[m].mu_q;
            eta_q = dirs[m].eta_q;
            lstore = l_outflow_.data() + (size_t)m * (size_t)n * 3;
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

            if (!light_on) return;
            // ---- THE LIGHT CHANNELS at this cell, this ordinate (P6a) ----------
            // sweep_ref_q.py's LightGroup, line for line: LIGHT's upwind pair from
            // its own stores (an off-grid read is 0, the DARK ring -- P6b's sky
            // replaces that 0), the split with its remainder, the whole stamped
            // share absorbed, the per-ordinate emission added. Reads nothing the
            // heat code above writes and writes nothing it reads.
            const int luay = y + lady, luax = x + ladx;
            const int luby = y + lbdy, lubx = x + lbdx;
            const bool in_la = (luay >= 0 && luay < h && luax >= 0 && luax < w);
            const bool in_lb = (luby >= 0 && luby < h && lubx >= 0 && lubx < w);
            const int ltay = y - lady, ltax = x - ladx;
            const int ltby = y - lbdy, ltbx = x - lbdx;
            const bool out_la = !(ltay >= 0 && ltay < h && ltax >= 0 && ltax < w);
            const bool out_lb = !(ltby >= 0 && ltby < h && ltbx >= 0 && ltbx < w);
            const size_t i3 = (size_t)i * 3;
            const int64_t* la_src = in_la ? lstore + ((size_t)luay * w + luax) * 3 : nullptr;
            const int64_t* lb_src = in_lb ? lstore + ((size_t)luby * w + lubx) * 3 : nullptr;
            int64_t I_m = 0;
            for (int c = 0; c < 3; ++c) {
                const int64_t io_la = in_la ? la_src[c] : 0;      // the dark ring
                const int64_t io_lb = in_lb ? lb_src[c] : 0;
                const int64_t lfa = (io_la * ls_m) >> FP_SHIFT;
                const int64_t lfb = io_lb - ((io_lb * ls_m) >> FP_SHIFT);   // the REMAINDER
                const int64_t lstream = lfa + lfb;
                if (!in_la) l_ring_in[c] += lfa;
                if (!in_lb) l_ring_in[c] += lfb;
                const int64_t absorbed = (lstream * (int64_t)l_d_[i3 + c]) >> FP_SHIFT;
                const int64_t lout = lstream - absorbed + l_emit_[i3 + c];
                light->light_q[i3 + c] += lstream;
                l_absorb[c] += absorbed;
                lstore[i3 + c] = lout;
                if (out_la || out_lb) {
                    const int64_t lfa_o = (lout * ls_m) >> FP_SHIFT;
                    if (out_la) l_ring_out[c] += lfa_o;
                    if (out_lb) l_ring_out[c] += lout - lfa_o;
                }
                I_m += lstream;
                if (lfirst) { lmin_s = lstream; lmax_s = lstream; lfirst = false; }
                else {
                    if (lstream < lmin_s) lmin_s = lstream;
                    if (lstream > lmax_s) lmax_s = lstream;
                }
            }
            // the net flux vector Σ_m I_m ŝ_m: two multiply-adds, the SYMMETRIC
            // shift, so a mirrored scene books the exactly mirrored vector
            light->light_flux_q[(size_t)i * 2 + 0] += fixedpoint::shr_round0_i64(I_m * mu_q, FP_SHIFT);
            light->light_flux_q[(size_t)i * 2 + 1] += fixedpoint::shr_round0_i64(I_m * eta_q, FP_SHIFT);
        };

        // Wavefront order: a topological order of the per-ordinate DAG. For
        // shear the dependency is a whole column (x-major) or row (y-major);
        // for step an anti-diagonal — and any row-major walk in the ordinate's
        // direction is topological for it (critique 3 §2a). The gather form
        // makes the integers independent of WHICH valid order is used.
        // P6a: BOTH walks below are topological for BOTH transports, which is
        // what lets light ride this traversal whatever heat's transport is: in
        // the column walk (shear, x-major) step's upwind (y, x-sx) is in the
        // previous column and (y-sy, x) earlier in the same one; in the row walk
        // every upwind cell of either transport is earlier in the row or in the
        // previous row.
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

    if (!light_on) return;
    // ---- THE GLOW (P6a): render-only, never in the books -------------------
    // The in-scattered light, light_q x the cell's albedo-times-density, only
    // where the smoke term put a coefficient (0 on every other cell). The
    // product is formed in 128 bits (light_q reaches 2^40 over-driven), so no
    // headroom argument rests on it.
    for (size_t i = 0; i < (size_t)n * 3; ++i) {
        const int32_t g = l_g_[i];
        light->light_glow[i] = (g == 0) ? 0
            : fixedpoint::mul128_shr(light->light_q[i], (int64_t)g, FP_SHIFT);
    }
    for (int c = 0; c < 3; ++c) {
        light_emit_sum[c]     = l_emit_tot[c] * (int64_t)n_ordinates;
        light_absorb_sum[c]   = l_absorb[c];
        light_ring_in_sum[c]  = l_ring_in[c];
        light_ring_out_sum[c] = l_ring_out[c];
    }
    light_min_stream = lmin_s;
    light_max_stream = lmax_s;
}
