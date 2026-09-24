#pragma once
// ============================================================================
// The emissive table E°(T) — THE black-body map, its inverse, and its bake.
// Ray-engine-v2 (docs/ray_engine_v2_design_v3_2026-09-15.md §2.6; critique
// 3 §5a). ONE instance is owned by PhysicsEngine (`emissive`) and read by the
// radiation sweep (LIVE since T5b) and the temperature solver's Pass-1 clamp.
// The old Raycaster keeps a vestigial copy of the SAME bake — only
// tests/test_emissive_table.py still reads it, since T6 retired its march —
// whose own bake CALLS the bake below (one implementation, two owners; never
// two bakes).
// ============================================================================
//
// Credit: J.R. Howell, M.P. Mengüç, R. Siegel, "Thermal Radiation Heat
// Transfer" — E°(T) = σT⁴ is the black-body emissive power this table holds;
// the emissivity ε is NOT in the bake (it is the per-material extinction
// coefficient, Kirchhoff: absorptivity == emissivity).
//
// THIS HEADER HOLDS ONLY THE INTEGER LOOKUPS (and the owner class's
// declaration). The bake BODY is out of line in emissive_table.cpp, which is on
// the /fp:strict list: its one real-number chain `(double)k4 * scale + 0.5` is
// a multiply feeding an add that MSVC may contract into a fused multiply-add
// under /fp:fast, and a header-inline bake would be compiled by whichever TU
// instantiated it — bindings.cpp is deliberately /fp:fast (cpp/CMakeLists.txt).
// A one-ULP change on `v` flips the `+0.5` rounding for any bucket near a half:
// a different table on one build, i.e. a moved digest.
//
// 4000 int64 entries over T_game in [0, 16000) in 4-game-unit buckets, ~32 KB.
// Bucket t covers [4t, 4t+4); its MIDPOINT is T_mid = 4t+2, so the absolute
// temperature at the midpoint is K(t) = kelvin_ambient + k_temp_to_kelvin·T_mid
// — an EXACT INTEGER for every bucket while both dials are integer-valued
// (the bake asserts this). K⁴ is built by REPEATED MULTIPLICATION in int64
// (k2 = K*K; k4 = k2*k2) — NEVER pow()/libm, whose last ULP varies across CRT
// versions and would desync a synced field across machines. In int64 the chain
// is exact (max K at the shipped dials keeps K⁴ inside INT64_MAX), so the ONLY
// rounding in the whole bake is the single `rad_scale · k4` boundary multiply.
// NO INTERPOLATION, deliberately: the 4-unit staircase means near-equal pairs
// land in the SAME bucket and net exactly 0.
//
// The lookups below are FP_HD so the CUDA twins (the P3 clamp in
// cuda_temperature.cu, the P4 sweep) share ONE definition with the host.

#include <cstdint>
#include <vector>

#include "fixed_point.h"   // FP_HD

static constexpr int E_TABLE_SIZE   = 4000;   // T_game in [0, 16000)
static constexpr int E_BUCKET_SHIFT = 2;      // 4 game units per bucket
// Total right shift from a Q16.16 temperature to a bucket index: 16 + 2.
static constexpr int E_INDEX_SHIFT  = 16 + E_BUCKET_SHIFT;
// Where E°⁻¹ saturates: the LAST bucket's low edge, 4·3999 = 15996 game — BELOW
// T_MAX_PHYS = 16000.
static constexpr int32_t E_INV_TOP_GAME = 4 * (E_TABLE_SIZE - 1);
// P5d: where the clamp's CEILING (e_ceiling_q, below) saturates — the LAST
// bucket's top, one Q16 LSB below 16000 game. Still BELOW T_MAX_PHYS, so on the
// radiative sub-step the clamp binds first and the T_MAX_PHYS rail stays
// reachable only through the `heat` deposit branch.
static constexpr int32_t E_CEILING_TOP_Q = ((4 * E_TABLE_SIZE) << 16) - 1;

// Q16.16 temperature -> E° bucket index. NEGATIVE T indexes bucket 0 (a tile
// below ambient does not emit less than the ambient floor in this model); T at
// or above the table top saturates on the last bucket. Pure integer.
FP_HD inline int e_bucket_of(int32_t T_q) {
    if (T_q <= 0) return 0;
    const int b = (int)(T_q >> E_INDEX_SHIFT);
    return (b >= E_TABLE_SIZE) ? (E_TABLE_SIZE - 1) : b;
}

// E°⁻¹(Φ): the largest bucket b with E°[b] <= Φ, mapped to the bucket's LOW
// edge and returned as a Q16.16 game temperature, (4·b) << 16. A FIXED 12-TRIP
// binary lifting (2048, 1024, ..., 1), branch-uniform in the sqrt_q16_dev
// idiom: one function for host and device, no data-dependent trip count
// (2^12 = 4096 >= E_TABLE_SIZE). Both edge cases of design v3 row 31:
//   * Φ < E°[0]      -> 0   (no radiative warming above the ambient floor;
//                            such a cell's rad_net is <= 0 anyway)
//   * Φ >= E°[3999]  -> 15996 game (E_INV_TOP_GAME), below T_MAX_PHYS
// Idempotent by construction — e_bucket_of((4b) << 16) == b, so
// E°[E°⁻¹(Φ)] <= Φ. This is the transcription of sweep_ref_q.py::e_inv_q,
// which is the spec. P5d: THE inverse of the table and the "radiation
// temperature" (the tile inspector's, the heat-law tests'), unchanged — but no
// longer the clamp's ceiling; the clamp reads e_ceiling_q below.
FP_HD inline int32_t e_inv_q(const int64_t* table, int64_t phi) {
    if (phi < table[0]) return 0;
    int lo = 0;
    for (int span = 1 << 11; span > 0; span >>= 1) {
        const int nxt = lo + span;
        const bool up = (nxt < E_TABLE_SIZE) && (table[nxt] <= phi);
        lo = up ? nxt : lo;
    }
    return (int32_t)((4 * lo) << 16);
}

// THE MAXIMUM-PRINCIPLE CLAMP'S CEILING (P5d, Erik's ruling of 2026-09-24;
// docs/ray_engine_v2_p5d_clamp_headroom_brief_2026-09-24.md): the TOP of the
// first bucket whose E° exceeds Φ, as a Q16.16 game temperature. With b =
// e_inv_q's bucket (the largest with E°[b] <= Φ), that bucket is b + 1, and its
// top is its last Q16.16 value:
//     e_ceiling_q(Φ) = (min(4·(b+2), 4·E_TABLE_SIZE) << 16) − 1
// WHY: the forward emission is a staircase with no temperature at which it
// equals Φ. An undamped cell with no clamp balances by flickering across the
// edge 4(b+1), where its bucket's E° crosses Φ; a ceiling at e_inv_q's 4b
// stopped it a bucket short, where it still emits E°[b] < Φ, so it net-absorbed
// every tick and the clamp withheld the whole surplus (P5c's counted shave).
// Radiation may now carry a cell into the first bucket in which it out-emits
// what it absorbs, and no further: "a clamped cell never emits more than it
// absorbs" became "... by more than one bucket", the resolution the emission
// itself works at. The edge cases:
//   * Φ < E°[0]      -> 0, UNCHANGED (design row 31: no radiative warming above
//                        the ambient floor — the one deliberate exception, so
//                        a shadowed cell cannot creep)
//   * Φ >= E°[3999]  -> E_CEILING_TOP_Q, one LSB below 16000 game, below
//                        T_MAX_PHYS (the clamp still binds before the rail)
// Built ON e_inv_q — the one fixed 12-trip lifting, not a second search — so
// the two cannot disagree about b. FP_HD: one definition, host and device (the
// Pass-1 clamp in temperature_solver.cpp and its CUDA twin). The transcription
// of sweep_ref_q.py::e_ceiling_q, which is the spec (gate G16).
FP_HD inline int32_t e_ceiling_q(const int64_t* table, int64_t phi) {
    if (phi < table[0]) return 0;
    const int b = (int)(e_inv_q(table, phi) >> E_INDEX_SHIFT);
    const int top_game = (b + 2 < E_TABLE_SIZE) ? 4 * (b + 2) : 4 * E_TABLE_SIZE;
    return (int32_t)((top_game << 16) - 1);
}

// The exact int64 bake (the body moved verbatim from raycaster.cpp, P-R4 /
// P-F1a lineage), OUT OF LINE in the strict TU emissive_table.cpp. Writes
// E_TABLE_SIZE entries into `out`. Throws std::runtime_error if either Kelvin
// dial is not integer-valued (the integer-bake precondition — a HARD
// invariant, not a debug assert: a config that moved them off-integer would
// silently desync CPU/CUDA through this table).
void bake_emissive_table_exact(int64_t* out, double rad_scale,
                               double kelvin_ambient, double k_temp_to_kelvin);

// The owner: the three dials (the SAME [physics] config homes the old
// Raycaster reads for kelvin_ambient/k_temp_to_kelvin; `rad_scale` itself is
// vestigial on the Raycaster since T6 — see raycaster.h) and the lazily
// baked table. `table()` re-bakes when a dial has moved since the last bake
// (the Raycaster::emissive_table() contract, kept). Every method is const and
// the table is `mutable` because it is a pure function of the dials: a cache,
// not hidden state.
class EmissiveTable {
public:
    double rad_scale        = 1.0e-5;   // heat counts per K⁴ ([physics.radiation] rad_scale_derived)
    double kelvin_ambient   = 293.0;    // [physics.temperature_scale] kelvin_ambient
    double k_temp_to_kelvin = 1.0;      // [physics.temperature_scale] k_temp_to_kelvin

    // Bake (or re-bake) from the CURRENT dials. Idempotent.
    void bake() const;
    // The baked table (E_TABLE_SIZE int64 entries). Bakes on first use and
    // whenever a dial has moved since the last bake.
    const int64_t* table() const;

private:
    mutable std::vector<int64_t> table_;
    mutable double baked_scale_ = 0.0;   // 0 == never baked
    mutable double baked_amb_   = 0.0;
    mutable double baked_slope_ = 0.0;
};
