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
//
// THE SWEEP'S HEAT CURRENCY (#78, 2026-09-25; Erik asked for the fix on
// 2026-09-24; docs/sweep_fine_heat_currency_brief_78_2026-09-25.md). At the
// live scale a room-temperature black body is only E°[0] = 125 heat counts per
// tick, so the sweep's per-ordinate terms were single-digit integers and every
// floor lost up to one count — ~10 K of emission-equivalent near ambient: cells
// a few kelvin warm read as net ABSORBERS and could not shed a small excess
// (the literature's STAGNATION; Croci & Giles 2023, Kloewer et al. 2020 — whose
// remedy, rescaling, is the ruled fix). So the table is baked 2^E_FINE_BITS
// finer than one heat count: rad_scale · 2^k, a power of two, exact in double,
// so the bake still rounds ONCE. Φ, the ambient level, and the sweep's four
// planes are then all in that fine currency — the sweep itself is
// unit-agnostic — and EVERY reader that turns one back into heat counts
// converts ONCE, through fixedpoint::fine_heat_shr (fixed_point.h; the gas
// chain at its final narrow, deposit_dT_wide_i64's fine_bits), never an inline
// shift. e_inv_q / e_ceiling_q need nothing: they compare Φ and E° in one unit.
// A table RECORDS its currency (EmissiveTable::fine_bits, below): the engine's
// is always E_FINE_BITS; a test instrument may bake a COARSE table (0 — the
// pre-#78 currency; the integer reference's resolving table, whose physics does
// not fit the fine currency inside int64), and every reader converts by the
// currency of the table it was handed.

#include <cstdint>
#include <vector>

#include "fixed_point.h"   // FP_HD

// THE ONE CONSTANT (#78): the sweep's heat channel works in a currency
// 2^E_FINE_BITS finer than one heat count. 11 is the largest k whose fine live
// table stays inside design v3's int64 bounds (per-cell sums < 2^46, plain
// products < 2^58; the integer reference's gate 11) — it puts the live table at
// ~0.66x the magnitude the reference's resolving table already exercises. Its
// Python twin is sweep_ref_q.FINE_BITS (tests/test_ray_engine_v2_integer_
// reference.py holds the two equal); Python reads this value through the
// binding (breach_physics.E_FINE_BITS / EmissiveTable.fine_bits), never a copy.
static constexpr int E_FINE_BITS = 11;

// THE TABLE'S HEADROOM DOOR (#78). The sweep's int64 arithmetic is argued for a
// table whose top entry is below 2^44: every plain int64 product it forms is at
// most 2^16 x the top (stream x a, ex_cell x w_m, the pre-pass's a x ex), i.e.
// below 2^60 -- >= 2^3 of int64 margin, the brief's criterion -- and the per-cell
// sums stay below 2^48 (sweep_ref_q_gates G11 measures the shipped tables: the
// fine live table's top is 2^41.1, the coarse resolving table's 2^41.7). A
// currency makes a table 2^k larger, so a scale x currency pair outside it --
// the resolving scale baked FINE (2^52.7) -- is REFUSED at the bake, never
// swept into a silent int64 wrap.
static constexpr int64_t E_TABLE_TOP_MAX = (int64_t)1 << 44;

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

// ============================================================================
// THE LIGHT EMISSION TABLE L°[T] (ray-engine-v2 P6a; design v3 §2.6 / §4.1 /
// §7.3 / §8.1; docs/ray_engine_v2_p6a_light_channels_brief_2026-09-25.md
// decisions 3-5). E°'s visible-band twin, in this module and this pattern:
// three int64 channels (R, G, B) over E°'s OWN 4000 buckets, indexed by the
// same e_bucket_of. A cell whose light extinction is a_c emits a_c · L°_c[T]
// (Kirchhoff per channel), split over the ordinates exactly as E° is.
//
// CHECKED IN, NEVER COMPUTED AT LOAD. The values are renderer/blackbody.py's
// Helland/Bartlett chroma x Macklin intensity (the ONE colour map) at the
// shipped [render.blackbody] dials, evaluated OFFLINE by tools/gen_light_table.py
// and written to light_emission_table.inc, which emissive_table.cpp compiles in:
// the two curves are libm (log, pow), a determinism hole the day P7 digests
// light_q (§8.1). tests/test_light_table.py re-runs the generator and requires
// every entry within one count. L°_c[0] == 0 in every channel: a room-
// temperature body emits no visible light, so the light ambient is DARK and a
// body (the stamped share d − a) re-emits nothing (decision 5).
//
// THE LIGHT CURRENCY (decision 4, #78's lesson built in): the table's integers
// are 2^L_FINE_BITS per LIGHT UNIT, the ramp's intensity 1.0 in the peak channel
// (Macklin's reference level at kelvin_ref). 37 is chosen so that
//   * the faintest glow the ramp SHOWS (its LUT's first non-zero step, intensity
//     7.23e-8 at 836 K) lands >= 2^8 counts per ordinate on an opaque cell:
//     2^37 · 7.23e-8 / 16 = 621 = 2^9.28 (at k = 36 it would be 311, 2^8.28 --
//     no margin); and a burning crate (1263 game, the shipped furniture row) is
//     ~2^21.8 counts per ordinate SIXTEEN tiles away on the dimmest bearing
//     (sweep_ref_q_gates G18 measures both);
//   * the table top (intensity_max 8, chroma 1) is 8 · 2^37 = 2^40, so a stream
//     is <= 2^36.4, every plain int64 product of the light loop <= 2^54 (the
//     flux term, 3 channels x 2^16), the per-cell light_q <= 2^40 -- 2^4 below
//     the design's 2^58 product line; G18 measures it over-driven.
// A table carries its currency (LightEmissionTable::fine_bits, the
// EmissiveTable::fine_bits pattern) and the generated data records the currency
// it was generated at (L_TABLE_GEN_FINE_BITS, static_asserted equal to this):
// moving this constant without regenerating the table does not build.
static constexpr int L_FINE_BITS = 37;
static constexpr int L_CHANNELS  = 3;
// THE LIGHT TABLE'S HEADROOM DOOR (the E_TABLE_TOP_MAX idiom): every entry of
// the checked-in table is below it -- static_asserted in emissive_table.cpp. A
// top below 2^44 keeps the light loop's widest plain product (the flux term,
// 3 streams x a Q16 cosine) below 2^58 and its per-cell sums below 2^46.
static constexpr int64_t L_TABLE_TOP_MAX = (int64_t)1 << 44;

// The checked-in L° table: L_CHANNELS x E_TABLE_SIZE int64, CHANNEL-MAJOR --
// L°_c[t] = light_emission_table()[c * E_TABLE_SIZE + t]. One definition
// (emissive_table.cpp, from light_emission_table.inc); host memory (a CUDA
// caller uploads it, as it uploads E°).
const int64_t* light_emission_table();

// The owner pattern (EmissiveTable's): the table and its CURRENCY travel
// together, so a reader converts by the currency of the table it was handed.
// Unlike E° there is no bake and no dial -- one checked-in table, one currency.
class LightEmissionTable {
public:
    static constexpr int fine_bits = L_FINE_BITS;
    const int64_t* table() const { return light_emission_table(); }
};

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
// #78: `fine_bits` bakes the table in a currency 2^fine_bits finer than one
// heat count — the scale becomes rad_scale · 2^fine_bits, a power-of-two
// multiply, EXACT in double, so the one rounding of the bake is unchanged.
// Throws std::invalid_argument outside [0, E_FINE_BITS] (the currencies the
// engine's int64 headroom argument covers: the engine's own, and coarse), or
// when the baked top entry reaches E_TABLE_TOP_MAX (the headroom door above).
void bake_emissive_table_exact(int64_t* out, double rad_scale,
                               double kelvin_ambient, double k_temp_to_kelvin,
                               int fine_bits);

// The owner: the three dials (the SAME [physics] config homes the old
// Raycaster reads for kelvin_ambient/k_temp_to_kelvin; `rad_scale` itself is
// vestigial on the Raycaster since T6 — see raycaster.h), the table's CURRENCY
// (#78) and the lazily baked table. `table()` re-bakes when a dial has moved
// since the last bake (the Raycaster::emissive_table() contract, kept). Every
// method is const and the table is `mutable` because it is a pure function of
// the dials: a cache, not hidden state.
class EmissiveTable {
public:
    double rad_scale        = 1.0e-5;   // heat counts per K⁴ ([physics.radiation] rad_scale_derived)
    double kelvin_ambient   = 293.0;    // [physics.temperature_scale] kelvin_ambient
    double k_temp_to_kelvin = 1.0;      // [physics.temperature_scale] k_temp_to_kelvin
    // #78: the CURRENCY the baked integers are in — 2^fine_bits of them per
    // heat count. The engine's table (PhysicsEngine::emissive) is always
    // E_FINE_BITS; 0 bakes the pre-#78 coarse table, which a test instrument
    // uses for the integer reference's RESOLVING table (its physics does not fit
    // the fine currency inside int64). Everything booked from this table — Φ,
    // the ambient level, the sweep's four planes — is in this currency, and
    // every reader converts by THIS value (fixedpoint::fine_heat_shr), so a
    // table and its planes can never be read in two currencies.
    int    fine_bits        = E_FINE_BITS;

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
    mutable int    baked_bits_  = -1;    // #78: the currency the cache holds
};
