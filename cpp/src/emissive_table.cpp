// The E° bake — OUT OF LINE, on the /fp:strict list (cpp/CMakeLists.txt).
// Ray-engine-v2 P1 (design v3 §2.6, critique 3 §5a): the body below is the
// bake that lived in raycaster.cpp (P-R4 ruling A1.3; P-F1a L2-B3 widened it
// to int64), moved here VERBATIM so there is exactly one bake in the tree.
// Raycaster::bake_emissive_table() and EmissiveTable::bake() both call it;
// tests/test_emissive_table.py asserts the two tables are identical entry for
// entry and equal to the integer reference's bake_e_table().
//
// Credit: J.R. Howell, M.P. Mengüç, R. Siegel, "Thermal Radiation Heat
// Transfer" (E°(T) = σT⁴); the integer-K⁴ / one-boundary-multiply form is
// this tree's determinism idiom (raycaster.h's original E° block).
//
// WHY A STRICT TU AND NOT A HEADER: the chain `v = (double)k4 * scale;
// (int64)(v + 0.5)` is a multiply feeding an add. Under the global fast-math
// flag MSVC may contract it into a fused multiply-add (one rounding instead of
// two); a one-ULP change on `v` flips the `+0.5` rounding of any bucket that
// sits near a half — a different table on one build, i.e. a moved digest.
// bindings.cpp deliberately keeps the global fast-math flag, and it is a
// binding that triggers the bake.
//
// RATCHET NOTE (tests/test_no_float_in_sim_tu.py): this TU is on SIM_TUS with
// a baseline that admits exactly the real-number lines below — the bake is
// the ONE place the E° map touches a real number (the load-time boundary
// multiply, door 2), and that is what the baseline documents.

#include "emissive_table.h"

#include <cmath>       // std::floor, std::llround (the dial-integrality check)
#include <stdexcept>
#include <string>

void bake_emissive_table_exact(int64_t* out, double rad_scale,
                               double kelvin_ambient, double k_temp_to_kelvin,
                               int fine_bits) {
    // #78: the table's CURRENCY. A table 2^fine_bits finer than one heat count
    // is the SAME bake at rad_scale · 2^fine_bits — a multiply by a power of two,
    // which is EXACT (only the exponent moves), so the one rounding
    // boundary below is unchanged and the reference's bake_e_table(.., k) is
    // this table integer for integer. [0, E_FINE_BITS] is what the engine's int64
    // headroom argument covers (emissive_table.h).
    if (fine_bits < 0 || fine_bits > E_FINE_BITS) {
        throw std::invalid_argument(
            "bake_emissive_table_exact: fine_bits = " + std::to_string(fine_bits) +
            " outside [0, E_FINE_BITS = " + std::to_string(E_FINE_BITS) + "] -- the "
            "currencies the sweep's int64 headroom argument covers (#78)");
    }
    const double scale = rad_scale * (double)((int64_t)1 << fine_bits);
    const double amb   = kelvin_ambient;
    const double slope = k_temp_to_kelvin;
    // Integer-bake precondition (temperature_scale design §3a): the exact-int64
    // chain below is only exact while both Kelvin-map dials are whole numbers —
    // a HARD invariant (a config that moves them off-integer without also
    // updating this bake to a quantize path would silently desync CPU/CUDA),
    // so it throws instead of a debug-only assert.
    if (amb != std::floor(amb) || slope != std::floor(slope)) {
        throw std::runtime_error(
            "bake_emissive_table_exact: kelvin_ambient/k_temp_to_kelvin "
            "must be integer-valued for the exact int64 E-table bake "
            "(kelvin_ambient=" + std::to_string(amb) +
            ", k_temp_to_kelvin=" + std::to_string(slope) + ")");
    }
    const int64_t amb_i   = std::llround(amb);
    const int64_t slope_i = std::llround(slope);
    for (int t = 0; t < E_TABLE_SIZE; ++t) {
        const int64_t T_mid = 4LL * (int64_t)t + 2LL;
        const int64_t K   = amb_i + slope_i * T_mid;    // 295 + 4t at the shipped dials
        const int64_t k2  = K * K;                       // repeated multiplication
        const int64_t k4  = k2 * k2;                     // exact in int64 at every shipping dial
        // The ONE rounding boundary: int64->double conversion (inexact only at
        // the very top of the table, benign) followed by the rad_scale boundary
        // multiply. Round half up; a non-positive product is 0.
        const double  v   = (double)k4 * scale;
        out[t] = (v >= 9.2233720368547748e18)
            ? INT64_MAX
            : (int64_t)((v > 0.0) ? (v + 0.5) : 0.0);   // rad_scale > 0 by contract
    }
    // #78: the headroom door (emissive_table.h). A table the sweep's int64
    // argument does not cover is refused here, loudly, not swept into a wrap.
    if (out[E_TABLE_SIZE - 1] >= E_TABLE_TOP_MAX) {
        throw std::invalid_argument(
            "bake_emissive_table_exact: the table's top entry reaches 2^44 "
            "(E_TABLE_TOP_MAX) -- outside the radiation sweep's int64 headroom "
            "(#78). Bake a large scale in a coarser currency (fine_bits), or a "
            "smaller scale.");
    }
}

void EmissiveTable::bake() const {
    // #78: bake into a FRESH buffer and swap it in only on success, so a bake
    // the doors refuse leaves the cache exactly as it was (never a refused
    // table's entries under the previous dials' cache keys).
    std::vector<int64_t> fresh((size_t)E_TABLE_SIZE);
    bake_emissive_table_exact(fresh.data(), rad_scale, kelvin_ambient,
                              k_temp_to_kelvin, fine_bits);
    table_.swap(fresh);
    baked_scale_ = rad_scale;
    baked_amb_   = kelvin_ambient;
    baked_slope_ = k_temp_to_kelvin;
    baked_bits_  = fine_bits;
}

const int64_t* EmissiveTable::table() const {
    if (table_.size() != (size_t)E_TABLE_SIZE || baked_scale_ != rad_scale ||
        baked_amb_ != kelvin_ambient || baked_slope_ != k_temp_to_kelvin ||
        baked_bits_ != fine_bits) {
        bake();
    }
    return table_.data();
}

// ============================================================================
// THE LIGHT EMISSION TABLE L° (ray-engine-v2 P6a) -- CHECKED IN, never computed.
// light_emission_table.inc is written offline by tools/gen_light_table.py from
// renderer/blackbody.py (emissive_table.h has the argument); this TU is its
// one definition. Pure integer data: nothing here rounds, and the ratchet's
// baseline for this file does not move. Three compile-time doors on the data
// itself, so a table that breaks one does not build:
//   * it was generated at THIS header's currency (L_FINE_BITS),
//   * bucket 0 is dark on every channel (the light ambient is L°[0] = 0: a
//     room-temperature body emits no visible light, decision 3),
//   * every entry lies in [0, L_TABLE_TOP_MAX) (the light loop's headroom).
// ============================================================================
namespace {
#include "light_emission_table.inc"   // L_TABLE_GEN_FINE_BITS, L_TABLE_DATA[3 * 4000]

constexpr bool light_table_ambient_is_dark() {
    return L_TABLE_DATA[0] == 0 && L_TABLE_DATA[E_TABLE_SIZE] == 0 &&
           L_TABLE_DATA[2 * E_TABLE_SIZE] == 0;
}

constexpr bool light_table_inside_its_door() {
    for (int i = 0; i < L_CHANNELS * E_TABLE_SIZE; ++i) {
        if (L_TABLE_DATA[i] < 0 || L_TABLE_DATA[i] >= L_TABLE_TOP_MAX) return false;
    }
    return true;
}
}  // namespace

static_assert(L_TABLE_GEN_FINE_BITS == L_FINE_BITS,
              "light_emission_table.inc was generated at another currency than "
              "emissive_table.h's L_FINE_BITS: run tools/gen_light_table.py --write");
static_assert(sizeof(L_TABLE_DATA) == sizeof(int64_t) * L_CHANNELS * E_TABLE_SIZE,
              "the light table must be L_CHANNELS x E_TABLE_SIZE entries");
static_assert(light_table_ambient_is_dark(),
              "L°[0] must be 0 on every channel (the light ambient is dark)");
static_assert(light_table_inside_its_door(),
              "a light table entry is negative or at/above L_TABLE_TOP_MAX");

const int64_t* light_emission_table() { return L_TABLE_DATA; }
