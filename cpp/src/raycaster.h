#pragma once
// 2D raycaster — DDA ray marching for light and heat.
// Casts rays from light sources, deposits intensity into a light map.
//
// ============================================================================
// CREDIT THE SOURCE (project iron rule) — the published techniques this file
// implements as of P-R4 (docs/radiation_raycaster_extinction_ruling_2026-07-31
// .md A1):
//
//   * J.R. Howell, M.P. Mengüç, R. Siegel, "Thermal Radiation Heat Transfer"
//     (6th ed., CRC Press 2016) — the NET-EXCHANGE formulation between two
//     grey surfaces, Q_net = a_1·a_2·F_12·A·(E°(T_1) − E°(T_2)) with
//     E°(T) = σT⁴, and the view factor F as the fraction of emitted rays that
//     connect the pair. `march_ray_directional`'s radiation block below IS
//     that expression, with the 8-ray fan as a discrete view-factor sampler
//     (ruling A1.9) and Kirchhoff's law (ε == a == `heat_atten`) making the
//     pair coefficient symmetric — which is what makes the exchange
//     antisymmetric to the integer (ruling A1.1/A1.4).
//   * C.D. Levermore, G.C. Pomraning, "A Flux-Limited Diffusion Theory",
//     Astrophysical Journal 248:321 (1981) — the FLUX LIMITER: a radiative
//     transfer whose linearised coefficient steepens as T³ is capped at a
//     fraction of what would equalise the pair, so the explicit update stays
//     monotone. `RAD_LIM_SHIFT` below is that cap, as a power-of-two shift
//     (ruling A1.6).
//
// Both are listed for archival under docs/papers/ in
// docs/papers/README_radiation_2026-08.md (no PDF could be fetched from this
// machine — the README is the honest placeholder, not a fabricated archive).
// ============================================================================

#include <vector>
#include <cmath>
#include <cstdint>

// Falloff types for light sources
enum class Falloff : int { UNIFORM = 0, COSINE = 1, SHARP = 2 };

// CUDA-S2 gate: the host-precomputed ray POD lives in cuda_raycaster.h (a plain
// header, no CUDA symbols). Forward-declare it so build_ray_list can return a
// vector of them without dragging the CUDA header into every CPU TU that
// includes raycaster.h.
namespace breach_cuda { struct RayHD; }

// ---- Fixed-point heat format (ch.04 §Fixed-point format) ----
//
// `heat` is the only sim-affecting ray output. It is stored as a Q16.16
// fixed-point int32: 16 integer bits, 16 fractional bits. One "unit" of heat
// energy == HEAT_SCALE raw int counts. The deposit is QUANTIZED into this
// domain (round-to-nearest) and added with a SATURATING add (clamp at int32
// max, never wrap) so a firestorm depositing many sources into few cells can
// never overflow past the ignition threshold (ch.04 review #6). Integer +=
// is order-independent -> deterministic / cross-machine-safe (the property
// that lets `heat` become an atomicAdd on CUDA later, ch.03 §CUDA contract).
//
// Nothing READS heat yet (this slice only DEPOSITS); the temperature pass
// (ch.04) will consume it non-destructively.
static constexpr int32_t HEAT_SCALE = 65536;   // 2^16 (Q16.16)

// Saturating quantize: float energy -> Q16.16 int32, rounded, clamped.
inline int32_t heat_quantize(float energy) {
    if (energy <= 0.0f) return 0;
    double scaled = static_cast<double>(energy) * static_cast<double>(HEAT_SCALE);
    double max_i32 = static_cast<double>(INT32_MAX);
    if (scaled >= max_i32) return INT32_MAX;
    return static_cast<int32_t>(scaled + 0.5);
}

// Saturating add into a Q16.16 accumulator: clamp at INT32_MAX, never wrap.
inline void heat_saturating_add(int32_t* cell, int32_t delta) {
    if (delta <= 0) return;
    // Overflow-safe: if adding delta would exceed INT32_MAX, clamp.
    if (*cell > INT32_MAX - delta) {
        *cell = INT32_MAX;
    } else {
        *cell += delta;
    }
}

// ============================================================================
// P-F1a — THE VERIFIED RADIATION BOOKS (design v6.1 rules 1/3/4 as amended by
// v7 + the v7.1 closure edits; docs/fire_realism_design_2026-08-01.md). Shared
// by the CPU march (raycaster.cpp), the CUDA march (cuda_raycaster.cu) and the
// temperature solver's signed fold, so the three read ONE definition of every
// boundary.
//
// SUPERSEDES P-R4's exchange. There is NO sink, NO credit and NO refund of any
// kind: five compensating terms telescoped into four rules (round-3.5), rule 2
// was then re-derived as a HALF-WEIGHT copy of rule 1 (round-3.6/v7), and
// round-3.7 certified conservation, the ledger identity and second-law safety.
// ============================================================================
//
// EMITTERS (v7.1 item 13): `burning ∪ (thermal_solid && T >= quantize(
// T_emit_gate))`, materialised ONCE PER TICK as an integer-threshold MASK PLANE
// from the same temperature snapshot both backends read — so "is r an emitter?"
// is a plane lookup, never a re-derivation that could drift between CPU and GPU.
//
// Per absorbing marched cell r on a ray from emitter s (SELF-CELL WHOLLY
// EXCLUDED — the distance-0 cell deposits nothing and is charged nothing; the
// exclusion is coded explicitly rather than relying on diff == 0):
//
//  RULE 1 — r is NOT an emitter:
//     x = a_s · a_r · τ · w · ( E°[T_s] − E°[T_r] )              // SIGNED
//     rad_net[r] += x;   rad_net[s] −= x                          // the SAME int
//
//  RULE 2 (v7, replacing P-R4/v6's one-way potential term) — r IS an emitter:
//     the SAME gap-signed antisymmetric pair at HALF WEIGHT. The ×0.5f goes
//     INSIDE the pinned float fold BEFORE the single quantize (v7.1 item 7 —
//     one rounding boundary, sign-symmetric; NOT mul_q16(HALF_Q, ·)), and the
//     branch clamps at HALF the shared budget (RAD_LIM_SHIFT + 1,
//     clamp-AFTER-halve; v7.1 item 2). The two casts of a mutual pair then sum
//     to exactly 1× the antisymmetric exchange — no double count, two equal-T
//     emitters exchange exactly 0 structurally, and the s↔r exchange rate is
//     CONTINUOUS across r's own gate crossing (what changes at the gate is only
//     r beginning to pay its OWN other directions and sky).
//
//  RULE 3 — CONTACT FACES ARE RADIATION-INERT (and that is the stated
//     semantics, not a leak): a ray stepping from a solid marched cell into a
//     face-adjacent solid cell TERMINATES with NO deposit and NO charge —
//     conduction owns contact (Erik ruling 3). The DDA steps ONE axis per
//     iteration, so consecutive marched cells always share the crossed face;
//     "face-adjacent" therefore needs no extra geometry. The source's own first
//     step into a touching solid is exactly this case, which is why a
//     wall-adjacent crate is EXPECTED to lose less radiatively than an
//     open-field one (v7.1 item 12) — its contact face conducts instead.
//
//  RULE 4 — THE SKY TERM, the ONLY ledger entry: a ray that LEAVES THE GRID
//     charges the emitter the escaping residual
//         sky = a_s · τ_end · w · ( E°[T_s] − E°[0] )
//     clamped by rad_pair_budget(|T_s|, his_s) — the ambient counterparty is
//     the T = 0 partner (v7.1 item 8) — booked as
//         rad_net[s] −= sky;   rad_amb[s] += sky                  // the SAME int
//     Leaving the grid is the ONLY escape, because RADIATION_RANGE is a global
//     constant >= the grid diagonal (v7 rule 4): reach-termination can never
//     precede the world edge, so "genuinely escapes" == "left the world",
//     map-independent, and the corridor leak is structurally impossible.
//
// with a_x = heat_atten[x] (absorptivity == emissivity, Kirchhoff), τ the
// running material transmittance (`heat_survival`, already Π(1−a_k) over the
// tiles crossed), w = 1/ray_count, and E° the PURE black-body table below.
// Applying the SAME truncated integer + to one end and − to the other is the
// fixed-point kit's S1 conservation idiom: the pair conserves exactly, two
// equal-T tiles net EXACTLY 0 (same bucket ⇒ diff == 0 ⇒ net == 0), and the
// divergence hazard is impossible BY CONSTRUCTION rather than by tuning.
//
// THE LEDGER IDENTITY (gate ii): Σ rad_net + Σ rad_amb == 0 EXACTLY, evaluated
// PRE-FOLD (v7.1 item 9 — the fold discards the sub-2^his remainder, a bias
// toward zero in magnitude, i.e. a systematic slight UNDER-transfer, never a
// mint). Every pair term is one integer applied ±; the sky term is the lone
// ± pair that crosses into the ambient ledger. Nothing else moves energy.
//
// THE NAMED, ACCEPTED LEAK (v7.1 item 11): a ray culled at `heat_cull` still
// carries <= 1% of its direction share, and that residual is charged to NOBODY.
// It is an UNDER-cooling (safe direction), and gate (v)'s open-field grey-body
// tolerance is set BELOW it.

// ---- the E° table (ruling A1.3; WIDENED TO int64 at P-F1a) ----------------
//
// *** L2-B3: THE TABLE IS int64 NOW. ***  Under P-R4 it was int32, and at the
// shipped `rad_scale` every entry above T_game ≈ 1768 saturated at INT32_MAX —
// a SILENT CEILING on the whole law: two tiles above it read the same saturated
// E°, so `diff` collapsed to 0 and a 3000-game tile radiated to a 2000-game
// tile exactly nothing. That ceiling, not the flux limiter, was the real
// high-T behaviour. int64 removes it: the largest entry is
// rad_scale · K⁴(t=3999) = 1e-5 · 1.088e18 ≈ 1.088e13, ~6 orders below
// INT64_MAX, so no shipping `rad_scale` can saturate the bake. The per-pair
// PRODUCT stays int32-representable because the flux limiter caps it — see
// rad_quantize_signed64 / the NO-OVERFLOW BOUND on rad_signed_add below.
//
// 4000 int64 entries over T_game ∈ [0, 16000) (== T_MAX_PHYS) in 4-game-unit
// buckets, ~32 KB. Bucket t covers [4t, 4t+4); its MIDPOINT is T_mid = 4t+2,
// so the absolute temperature at the midpoint is
//     K(t) = 293 + 2·T_mid = 293 + 8t + 4 = 297 + 8t
// — an EXACT INTEGER for every bucket. That is what lets the bake be exact:
//
//   *** CRITICAL DETERMINISM RULE ***  K⁴ is built by REPEATED MULTIPLICATION
//   (k2 = K*K; k4 = k2*k2) in int64 — NEVER pow()/libm, whose last ULP varies
//   across CRT versions and would desync machines through a synced int32
//   field. In int64 the chain is EXACT (max K = 297+8·3999 = 32289,
//   K⁴ ≈ 1.09e18 < 9.22e18), so the ONLY rounding in the whole bake is the
//   single `rad_scale · k4` boundary multiply — the locked load-time
//   double->quantize idiom. (The ruling says "repeated multiplication in
//   double"; int64 is that same chain in a type that cannot round at all,
//   which is strictly stronger and needs no /fp: discipline.)
//
// NO INTERPOLATION, deliberately: the 4-unit staircase means near-equal pairs
// land in the SAME bucket and net exactly 0, which reinforces the antisymmetry
// gate; the step error at 1000 K is a few percent of E — below the limiter's
// granularity (ruling A1.3).
// RC_HD marks the exchange helpers callable from BOTH the CPU march (.cpp) and
// the CUDA march (.cu device code) — the fixed_point.h FP_HD idiom, so the two
// backends share ONE definition of every boundary instead of a hand-copied twin
// that can drift. Under a plain host compiler it expands to nothing.
#if defined(__CUDACC__)
  #define RC_HD __host__ __device__
#else
  #define RC_HD
#endif

static constexpr int E_TABLE_SIZE   = 4000;   // T_game ∈ [0, 16000)
static constexpr int E_BUCKET_SHIFT = 2;      // 4 game units per bucket
// Total right shift from a Q16.16 temperature to a bucket index: 16 + 2.
static constexpr int E_INDEX_SHIFT  = 16 + E_BUCKET_SHIFT;

// Q16.16 temperature -> E° bucket index. NEGATIVE T indexes bucket 0 (a tile
// below ambient does not emit less than the ambient floor in this model); T at
// or above the table top saturates on the last bucket. Pure integer.
RC_HD inline int e_bucket_of(int32_t T_q) {
    if (T_q <= 0) return 0;
    const int b = (int)(T_q >> E_INDEX_SHIFT);
    return (b >= E_TABLE_SIZE) ? (E_TABLE_SIZE - 1) : b;
}

// ---- the flux limiter (ruling A1.6; Levermore & Pomraning 1981) ------------
// Per pair, per ray, per tick, |net| may not exceed the heat that would close
// 1/2^RAD_LIM_SHIFT of the pair's temperature GAP through either end's own
// thermal mass. At 4 that is 1/16 of the gap per ray; with 8 rays the
// worst-case aggregate is half the gap per tick — 2x inside conduction's own
// monotone line (4 faces x 1/4 = 1) and 4x from divergence. It is a STABILITY
// constant, not a feel dial, and in normal operation it is INERT (the T⁴ net
// sits far below the budget) — it is a rail against the T³ steepening at
// T_MAX_PHYS-scale gaps.
static constexpr int RAD_LIM_SHIFT = 4;

// The pair budget for ONE end, from the Q16.16 gap |T_s − T_r| and that end's
// heat_inv_shift: (|ΔT| << his) >> shift, in HEAT counts. int64 because
// |ΔT| can reach T_MAX_PHYS·65536 ≈ 1.05e9 and his can reach 5 (steel).
//
// `shift` is RAD_LIM_SHIFT for the rule-1 (single-caster) branch and
// RAD_LIM_SHIFT + 1 — i.e. HALF the shared budget — for the rule-2 mutual
// half-weight branch (v7.1 item 2, M2). Halving the CAP alongside the term is
// what keeps the rail TRUE BY CONSTRUCTION when BOTH ends cast: the pair's two
// half-casts can together move at most 2 × (gap/2^(LIM+1)) == gap/2^LIM, i.e.
// exactly the single-caster rail, never more.
RC_HD inline int64_t rad_pair_budget_s(int64_t abs_dT_q, int his, int shift) {
    return (abs_dT_q << his) >> shift;
}
// The shared symmetric budget (rule 1 / the sky term).
RC_HD inline int64_t rad_pair_budget(int64_t abs_dT_q, int his) {
    return rad_pair_budget_s(abs_dT_q, his, RAD_LIM_SHIFT);
}

// ---- the ONE deposit boundary (mirrors heat_quantize's contract) -----------
// The radiation deposit is quantized ONCE per marched cell, exactly like the
// retired painter's `heat_quantize(heat_dep)`: the float march coefficient is
// promoted to double, multiplied by the (integer, full-precision) E° difference
// and rounded HALF-AWAY-FROM-ZERO so +x and −x behave identically (no sign DC
// bias — quantize()'s convention). E° is ALREADY in Q16.16 heat counts, so
// there is no second ×65536 here: this is a rounding, not a scale change.
RC_HD inline int32_t rad_quantize_signed(double v) {
    if (v >=  2147483647.0) return INT32_MAX;
    if (v <= -2147483648.0) return INT32_MIN;
    return (int32_t)((v >= 0.0) ? (v + 0.5) : (v - 0.5));
}

// The int64 twin, for the EXCHANGE terms now that E° is int64 (L2-B3). SAME
// rounding rule (half away from zero, one boundary), a wider saturation window
// so the clamp below — not the quantize — is what bounds the result.
//
// WHY THE WIDTH MATTERS: at T_MAX_PHYS the E° difference reaches ~1.09e13 and
// w·a_s·a_r·τ can be as large as 1/ray_count, so the RAW product can exceed
// INT32_MAX by orders of magnitude. Quantizing to int32 first would saturate to
// INT32_MAX and only THEN meet the limiter — which happens to give the same
// answer (INT32_MAX > any cap), but only by accident of the cap being smaller.
// Quantizing to int64 and clamping in int64 makes the bound structural instead
// of incidental, and it is the form the tol-0 CPU/CUDA contract is written on.
//
// THE NARROWING IS SAFE (re-derived at P-F1a): after the clamp,
//   |x| <= (|ΔT| << his) >> RAD_LIM_SHIFT
//       <= (T_MAX_PHYS·65536 << 5) >> 4          (his <= 5 for steel)
//        = 16000·65536·2 = 2.097e9 < 2^31 − 1 = 2.147e9.
// So every clamped term — and the halved rule-2 term, which is smaller still —
// fits int32 exactly. The narrow is therefore total, not a truncation.
RC_HD inline int64_t rad_quantize_signed64(double v) {
    if (v >=  9.2233720368547748e18) return INT64_MAX;
    if (v <= -9.2233720368547748e18) return INT64_MIN;
    return (int64_t)((v >= 0.0) ? (v + 0.5) : (v - 0.5));
}

// Signed accumulation into `rad_net[]` — PLAIN (non-saturating) adds, because
// order-freedom for SIGNED integers requires plain wraparound arithmetic:
// saturating signed adds are order-DEPENDENT (ruling A1.7), which would break
// the CPU<->CUDA tol-0 contract the moment two rays hit one cell in a different
// order. Written through unsigned so the (documented, out-of-band) overflow
// case wraps deterministically instead of being C++ UB — which is also exactly
// what the device's atomicAdd(int*) does, so the two backends agree even there.
//
// NO-OVERFLOW BOUND — RE-DERIVED AT P-F1a (the P-R4 derivation is VOID: it
// leaned on E° saturating at INT32_MAX above T_game ≈ 1766 to cap |ΔE|, and
// that saturation is exactly the ceiling L2-B3 removed by widening the table to
// int64). The bound is now carried entirely by the FLUX LIMITER, which is the
// honest place for it:
//
//   PER TERM. Every pair term and the sky term are clamped, so
//       |x| <= (|ΔT| << his) >> RAD_LIM_SHIFT <= 16000·65536·2^5 / 2^4
//            = 2.097e9  <  2^31 − 1 = 2.147e9
//   at the worst case (a full-scale T_MAX_PHYS gap against steel's his = 5).
//   In the OPERATING band (T ≈ 443 game, furniture his = 3) the T⁴ term itself
//   binds long before the rail: E° ≈ 1.9e7 counts ⇒ |x| ≲ 1.2e6.
//
//   PER CELL. rad_net[i] accumulates the terms of every ray that touches i plus
//   every ray i itself casts. Two structural facts keep that sum bounded:
//   contact termination (rule 3) means a tile inside an assembly is on almost
//   no sightlines, and the 1/r ray density means no cell is on more than a
//   handful of a firestorm's sightlines. At the shipped dials a 600-emitter
//   firestorm sums to ~1e7 per cell — two orders below the rail, three below
//   2^31. In the pathological regime (every term railed, thousands of rays on
//   one cell) the accumulation WRAPS, deterministically and identically on both
//   backends — that is the documented, out-of-band contract of this function,
//   not UB, and it is exactly what the device's atomicAdd(int*) does.
inline void rad_signed_add(int32_t* cell, int32_t delta) {
    *cell = (int32_t)((uint32_t)*cell + (uint32_t)delta);
}

struct LightSource {
    float x, y;              // tile coordinates
    float max_range  = 20;
    int   ray_count  = 0;    // 0 = auto-compute from range + spread
    float angle_center = 0;
    float angle_spread = 2.0f * 3.14159265f;
    float intensity  = 1.0f;
    float heat       = 0.0f;
    float jitter     = 0.0f;
    float color[3]   = {1.0f, 1.0f, 1.0f};   // RGB tint, default white
    Falloff falloff  = Falloff::UNIFORM;

    int get_ray_count() const {
        if (ray_count > 0) return ray_count;
        float full_circle = std::ceil(2.0f * 3.14159265f * max_range);
        float fraction = angle_spread / (2.0f * 3.14159265f);
        return std::max(8, static_cast<int>(std::ceil(full_circle * fraction)));
    }
};

// ---- P-R4: the per-source radiation payload (ruling A1.8) -----------------
//
// The P-R1 builder used to hand the march ONE scalar per source: the painter's
// `heat = k_fire_heat · I`. The net-exchange law needs the emitter's STATE
// instead — its temperature (for the E° lookup), its absorptivity (== its
// emissivity) and its own thermal-mass shift (for the limiter's budget) — plus
// the cell index to DEBIT. One RadSource rides alongside each LightSource, in
// the same row-major order.
struct RadSource {
    int     idx   = -1;   // source cell index (row*w + col) — the debit target
    int32_t T_q   = 0;    // source temperature, Q16.16 (pass-entry snapshot)
    int64_t E_s   = 0;    // E°[e_bucket_of(T_q)] — baked table lookup (int64)
    float   a_s   = 0.0f; // absorptivity == emissivity == heat_atten[idx]
    int     his_s = 0;    // heat_inv_shift[idx] (limiter budget, source end)
    // D3 / v7.1 item 4 (M4): the LEGACY reach — `range_base +
    // range_per_intensity · I`, the range this source's rays used to march to.
    // The emission ray now runs to RADIATION_RANGE, but the rad_flux SENSOR
    // write stays behind this deterministic distance guard so unit radiant
    // damage is UNCHANGED and far-field damage bursts never ship. Books
    // untouched: rad_flux is not part of the ledger.
    float   damage_range = 0.0f;
};

// The per-RAY radiation constants (the RadSource folded with 1/ray_count),
// i.e. the exact analogue of the old `ray_heat = src.heat * atten * inv_n`.
struct RadRay {
    int     src_idx = -1;
    int32_t T_q     = 0;
    int64_t E_s     = 0;
    int     his_s   = 0;
    float   coef    = 0.0f;   // a_s * (1/ray_count)  — the PINNED first factor
    float   damage_range = 0.0f;   // D3's legacy reach guard (see RadSource)
};

// The planes the exchange reads/writes. nullptr `rad_net` == radiation OFF for
// this cast (every non-fire caller: lamps, muzzle flashes, the render pass).
struct RadCtx {
    const int64_t* e_table        = nullptr;   // E_TABLE_SIZE entries (int64)
    const int32_t* temperature    = nullptr;   // Q16.16 (h,w)
    const int32_t* heat_inv_shift = nullptr;   // (h,w)
    // ---- v7.1 item 13: THE EMITTER MASK PLANE -----------------------------
    // `burning ∪ (thermal_solid && T >= quantize(T_emit_gate))`, computed ONCE
    // PER TICK from the same temperature snapshot the E° lookups read, as a
    // single INTEGER THRESHOLD compare — and then only LOOKED UP by the march.
    // Rule 2 (the half-weight mutual branch) keys on THIS plane, so CPU and
    // CUDA can never disagree about who is an emitter: they read the same
    // bytes rather than each re-evaluating a predicate against a float dial.
    // 1 == emitter, 0 == not.
    const uint8_t* emit_mask      = nullptr;   // (h,w)
    int32_t*       rad_net        = nullptr;   // Q16.16 (h,w) signed accumulator
    // ---- rule 4: THE AMBIENT (SKY) LEDGER ---------------------------------
    // The ONLY place energy leaves the tile books. Per-tile int32 plane with
    // PLAIN adds, keyed by the EMITTER's cell index — deliberately NOT a single
    // global atomic: a per-tile plane keeps the accumulation order-free and
    // contention-free on the device, and the host reduces it to a uint64 total
    // once per tick. Gate (ii) checks Σ rad_net + Σ rad_amb == 0 PRE-FOLD.
    // Entries are non-negative (E°[T_s] >= E°[0] for every bucket), so the
    // host-side reduction into uint64 is exact.
    int32_t*       rad_amb        = nullptr;   // (h,w) sky ledger
    // ---- D3: the RADIANT-FLUX SENSOR plane (amendment 5, Erik's ruling) ----
    // *** THIS IS NOT PART OF THE ENERGY LEDGER. ***  Read that again before
    // touching it: `rad_flux` is a DAMAGE SENSOR, not a transport term. It is
    // written at AIR cells only, it moves no energy, it changes no
    // temperature, nothing is debited anywhere to pay for it, and no solver
    // reads it — its ONE consumer is unit heat damage
    // (simulation/exchange.apply_environmental_damage), which used to sample
    // the PAINTER's air deposit at the unit's footprint. Radiation lands only
    // on solids (air has a == 0 and neither absorbs nor emits — Kirchhoff), so
    // without this plane a fire could not burn a marine standing next to it.
    // It carries the same occlusion (τ) and the same 1/r ray-density falloff
    // the painter did, so the damage sampler sees the physically right
    // incident flux — and because it is positive-only it keeps `heat[]`'s
    // ORDER-FREE saturating-add contract (unlike rad_net, which must be signed
    // and therefore plain).
    int32_t*       rad_flux       = nullptr;   // Q16.16 (h,w) positive-only
    bool active() const { return rad_net != nullptr && e_table != nullptr
                              && temperature != nullptr && heat_inv_shift != nullptr
                              && emit_mask != nullptr; }
};

class Raycaster {
public:
    // ---- P-R4 radiation dials (ruling A1) ---------------------------------
    //
    // `rad_scale` — the EMISSION calibration constant: heat counts per K⁴, with
    // σ, the 0.833 m² tile face, the per-tick dt and the game↔Kelvin mapping all
    // folded into it at bake time (ruling A1.3). Derivation of the shipped value
    // is in config.toml [physics.fire] rad_scale and in bake_emissive_table().
    // Changing it re-bakes the table (see bake_emissive_table).
    double rad_scale = 1.0e-5;
    // `T_emit_gate` — the temperature (game units) at or above which a NON-
    // burning thermal solid also CASTS (ruling A1.8, Erik's 180 = 653 K). The
    // gate decides who can radiatively LOSE heat; RECEIVERS ARE FREE (a cold
    // crate is heated correctly on the flame's own rays whatever this is).
    double T_emit_gate = 180.0;
    // ---- P-F1a / v7 rule 4: RADIATION_RANGE ------------------------------
    //
    // The reach of an EMISSION ray, in tiles. A STABILITY-CLASS CONSTANT, NOT A
    // FEEL DIAL: it must be >= the GRID DIAGONAL of the largest shipping level
    // so that reach-termination can NEVER precede the world edge. That is what
    // makes "the ray genuinely escaped" mean "the ray left the world" —
    // map-independent, and it makes the corridor leak (a ray dying mid-room and
    // charging nobody, so a long room silently loses less than a short one)
    // STRUCTURALLY IMPOSSIBLE rather than tuned away.
    //
    // FLOOR: the largest shipping level is 128 x 256 ⇒ diagonal
    // sqrt(128² + 256²) = 286.22 ⇒ the floor is 287. The shipped 320 carries
    // ~12% headroom for a larger level without a law change.
    //
    // COST: air cells cost a march STEP and no deposit work — which is why the
    // PURE-RADIATION FAST PATH (march_ray_radiation) exists and is mandatory:
    // an emission ray does no RGB, no gas optics, no `exp`. Fire's visible
    // light is a SEPARATE short cast on the legacy range formula.
    //
    // `range_base` / `range_per_intensity` return to render/legacy duty and to
    // D3's damage_range guard; they do NOT bound an emission ray.
    static constexpr double RADIATION_RANGE_MIN = 287.0;
    double radiation_range = 320.0;

    // Bake (or re-bake) the E° table from the CURRENT `rad_scale`. Idempotent
    // and a pure function of `rad_scale` — cast_from_fire_plane / the ray-list
    // builders call it lazily when `rad_scale` has moved since the last bake,
    // so a caller that only sets the dial can never march against a stale table.
    void bake_emissive_table() const;
    // The baked table (E_TABLE_SIZE int64 entries). Bakes on first use.
    const int64_t* emissive_table() const;

    // ---- Smoke optics (ch.05 §6.1 §6 — decoupled per-channel absorption vs glow) ----
    //
    // Two INDEPENDENT per-channel budgets, NOT constrained to absorb + glow = 1:
    //
    //   1. Per-channel transmission (Beer-Lambert):
    //        tau_c   = smoke_absorption[c] * smoke_density * smoke_absorb_scale
    //        trans_c = exp(-tau_c)               // never reaches 0 -> beam survives deep smoke
    //        survival[c] *= trans_c              // (engine/08: occlusion decays SURVIVAL)
    //      smoke_absorb_scale is the global "beam reach" dial: LOW = long beam
    //      (flashlight travels far through smoke and still glows), HIGH = beam
    //      dies fast. Per-channel absorption is the (future) gas COLOUR.
    //
    //   2. Separate additive scatter/glow (god-rays, smoke_glow buffer):
    //        smoke_glow[c] += deposited_light[c] * smoke_scatter_albedo[c] * smoke_density
    //      This is the light the smoke SCATTERS BACK toward the viewer. It is
    //      independent of (and may exceed) absorption -> "barely absorbs, glows
    //      brightly" gases (steam) are expressible.
    //
    // Legacy scalar `smoke_absorption` (kept for the scalar march_ray /
    // cast_source path which has no per-channel notion).
    float smoke_absorption = 0.8f;

    // SUPERSEDED (engine/05 §6.2, M2): the single-field per-channel coefficients.
    // The directional march now reads the per-GAS `absorption`/`scatter_albedo`
    // tables passed per-cast (from GasTable), summed density-weighted across the
    // (N,h,w) gas array — these two struct members no longer drive the directional
    // look. They are kept INERT (still bound) so any non-gas caller compiles; the
    // active dial for the gas path is `smoke_absorb_scale` below.
    float smoke_absorption_rgb[3] = {1.0f, 1.0f, 1.0f};
    float smoke_scatter_albedo[3] = {1.0f, 1.0f, 1.0f};
    // Global beam-reach dial (STILL ACTIVE for the gas path): scales the summed
    // per-gas tau. LOW = long beam (flashlight travels far). Default 1.4.
    float smoke_absorb_scale = 1.4f;

    // ---- Propagation-model cull thresholds (engine/08 §The march, §Falloff is
    // density) — the per-channel SURVIVAL floors of the pure-density model ----
    //
    // A ray carries fixed per-channel energy and a survival ∈ [0,1] that decays
    // ONLY by occlusion (never by distance — the 1/r falloff is ray DENSITY, not
    // a per-ray multiplier). A channel keeps depositing while its own survival is
    // above its threshold; the ray terminates when EVERY emitting channel is
    // below its threshold (or max_range). Because survival decays only by
    // occlusion, in open air it stays 1.0 and the ray runs to max_range — the
    // cull only bites BEHIND occluders (≈99% absorbed at 0.01).
    //   light_cull : ε_rgb — the RGB render channels' floor.
    //   heat_cull  : ε_heat — the heat (gameplay/damage) channel's floor; its
    //                own dial so a heat-shield/low-E-glass material can diverge
    //                from light. Heat deposits gate on heat_survival > heat_cull,
    //                which is what DECOUPLES heat from the float light path
    //                (engine/08 §Determinism: heat is decoupled from light).
    float light_cull = 0.01f;
    float heat_cull  = 0.01f;

    // ---- Legacy API (intensity only) ----
    //
    // (P-R1, 2026-07-31: update_from_fire + its coarse_cluster dial were
    // deleted here — no production caller, legacy intensity-only signature,
    // an RNG jitter land-mine, and clustering is incompatible with the
    // planned radiation law (a merged pseudo-source has no well-defined T_s).
    // See docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.2.)

    // Cast a single source (for flashlights, muzzle flashes, etc.)
    void cast_source(
        const LightSource& src,
        float* light_map,
        const float* smoke_field,
        const bool* is_wall,
        int h, int w
    ) const;

    // ---- Directional API (RGB light + dominant light direction) ----
    //
    // Outputs three fields:
    //   light_rgb[i*3 + c] = accumulated RGB light arriving at tile i (c=0..2)
    //   light_dx[i]        = x component of the (unit) light direction at tile i
    //   light_dy[i]        = y component of the (unit) light direction at tile i
    //
    // light_rgb is interleaved (R,G,B per tile), shape (h, w, 3) in row-major.
    // PURE-DENSITY model (engine/08 §Falloff is density): each ray carries fixed
    // energy = total_power / N (the cast divides by ray_count), and a per-channel
    // SURVIVAL ∈[0,1] that starts at 1 and decays ONLY by occlusion. The deposit
    // is energy·color[c]·survival[c] — there is NO per-ray distance falloff; the
    // 1/r intensity law emerges from ray density (N cancels).
    //
    // Occlusion is PER-CHANNEL material attenuation (ch.03 §the march): the
    // per-tile `light_atten` input (h, w, 3, interleaved) is the material
    // table's static attenuation. After depositing into a tile each channel's
    // survival is multiplied by (1 - mat_atten[c]) then the gas transmission;
    // opaque tiles ([1,1,1]) drive every channel to 0 == the old wall
    // hard-stop, glass ([0.1,..]) transmits dimmed, an unequal triple tints.
    // There is NO binary wall stop. Each channel STOPS DEPOSITING once its own
    // survival drops below its cull floor (light_cull for RGB, heat_cull for
    // heat); the ray marches to the AGGREGATE range — it continues while ANY
    // channel survives (no per-channel early-out, CUDA-divergence rule).
    //
    // light_dx/light_dy are unit-normalized after all rays are cast (vector
    // magnitude, not by intensity — see expert review notes in
    // docs/patch_level_pipeline_v1.md). At tiles where opposing rays cancel
    // (or no rays arrive), direction is (0,0) — the shader must handle that.

    // ---- Multi-gas optics (engine/05 §6.2 — coloured N-gas summation) ----
    //
    // The directional march generalises the single `smoke` scalar to N gas
    // density fields (gmap.gas, shape (N,h,w)), each with its OWN per-channel
    // `absorption` (N,3) and `scatter_albedo` (N,3) row from GasTable. Per tile,
    // per channel c, the two decoupled budgets above are SUMMED density-weighted
    // across all gases (engine/05 §6.2 — "mixing falls out of the sum"):
    //
    //   transmission:  tau_c = smoke_absorb_scale * Σ_g ( gas[g][tile] * absorption[g][c] )
    //                  trans_c = exp(-tau_c);  survival[c] *= trans_c
    //   scatter/glow:  smoke_glow[c] += dep_c * Σ_g ( gas[g][tile] * scatter_albedo[g][c] )
    //
    // `smoke_absorb_scale` stays the global beam-reach dial. A single populated
    // gas reproduces exactly what the old single-`smoke` path did for that gas's
    // coefficients (with absorption/scatter = that gas's row). Heat is untouched
    // (smoke/gas does not attenuate the heat channel).

    // Cast a single source and accumulate RGB light + direction, plus the two
    // Slice-4 outputs:
    //   heat       : Q16.16 fixed-point int32, shape (h,w). Deposited where the
    //                source emits heat (src.heat > 0). Heat is the INDEPENDENT
    //                4th ray channel (engine/06 §1): it carries its OWN scalar
    //                survival, attenuated per tile by `heat_atten` exactly as
    //                each RGB channel is attenuated by `light_atten[c]`. The
    //                deposit is (src.heat/N) * heat_survival (NO distance
    //                falloff), GATED on heat_survival > heat_cull, quantized +
    //                SATURATING-added — independent of the RGB survival, so a
    //                heat-shield (light-clear, heat-opaque) blocks heat while
    //                passing light, and smoked glass (light-opaque, heat-clear)
    //                does the converse. May be nullptr to skip.
    //   smoke_glow : f32 RGB, shape (h,w,3), interleaved. God-ray glow (ch.03
    //                C16): the light each tile's SMOKE ABSORBS is deposited here
    //                per channel — energy-conserving by construction (the energy
    //                the smoke removed from the ray). May be nullptr to skip.
    //
    // heat_atten : per-tile scalar heat-ray attenuation (h,w), the heat analogue
    //              of light_atten (air 0, walls 1.0, glass 0.3). May be nullptr,
    //              in which case heat is NOT attenuated (the pre-S6 behaviour:
    //              heat survival stays 1.0 the whole march).
    //
    // Caller is responsible for zeroing the output buffers before casting the
    // frame's sources. Normalization of light_dx/dy is NOT performed here —
    // call normalize_directions() once after all sources have been cast.
    void cast_source_directional(
        const LightSource& src,
        float* light_rgb,
        float* light_dx,
        float* light_dy,
        int32_t* heat,              // Q16.16 fixed-point, (h,w) or nullptr
        float* smoke_glow,          // RGB god-ray glow, (h,w,3) or nullptr
        const float* gas_field,     // (n_gases, h, w) contiguous gas densities
        const float* gas_absorption,// (n_gases, 3) per-gas per-channel absorption
        const float* gas_scatter,   // (n_gases, 3) per-gas per-channel scatter
        int n_gases,
        const float* light_atten,   // per-tile static material atten (h,w,3)
        const float* heat_atten,    // per-tile heat atten (h,w) or nullptr
        int h, int w,
        // ---- P-R4: the net-T⁴ radiation exchange (ruling A1) --------------
        // `rad` carries the planes, `rs` the emitter's state. BOTH default to
        // "off" so every non-fire caller (lamps, muzzle flashes, the render
        // pass, the legacy bound API) compiles and behaves EXACTLY as before —
        // the exchange is a strictly additive channel on top of the march.
        const RadCtx* rad = nullptr,
        const RadSource* rs = nullptr
    ) const;

    // ---- P-R1: whole-fire-plane cast (source build moved into C++) ----
    //
    // docs/radiation_raycaster_extinction_ruling_2026-07-31.md A4.1-A4.2.
    // Replaces PhysicsRunner.cast_fire_heat's old per-tile Python loop
    // (one bp.LightSource() + ~10 pybind attribute writes PER BURNING TILE,
    // PER TICK): enumerates every burning tile (fire[i] > 0, Q16.16) in
    // ROW-MAJOR order via build_fire_sources() and casts each resulting
    // source immediately with cast_source_directional — i.e. the SAME
    // per-source CPU cast the old Python loop drove, just built natively.
    // `heat` is byte-identical to that old loop (same sources, same march,
    // same order-free saturating add) — this is a mechanical relocation,
    // no law/behavior change (P-R1's byte-identity gate).
    //
    // `fire` is the Q16.16 int32 fire plane (h, w); the dial parameters
    // mirror the old Python-side runner attributes EXACTLY (see
    // build_fire_sources for the float-parity contract on why they are
    // `double`, not `float`). `jitter` stays fixed at 0.0 by the caller
    // (fire heat is sim-affecting — no dither; S8c item 1's RNG-coupling
    // guard).
    //
    // *** P-R4 (ruling A1): THE PAINTER IS GONE. ***  This cast no longer
    // takes `k_fire_heat` and no longer takes the `heat` plane: a fire does
    // not PAINT one-way energy into every cell its rays cross. It now runs the
    // antisymmetric net-T⁴ EXCHANGE into `rad_net` (signed), reading
    // `temperature` for both ends' E° and `heat_inv_shift` for the limiter.
    // The emitter set also widens (ruling A1.8): burning tiles ∪ thermal
    // solids at or above `T_emit_gate`, still row-major.
    //
    // *** P-F1a: TWO CASTS PER SOURCE. ***  The single combined march is split:
    //
    //   1. THE EMISSION CAST (march_ray_radiation) — the PURE-RADIATION FAST
    //      PATH, mandatory under v7 rule 4. Marches to `radiation_range` (>=
    //      the grid diagonal) doing ONLY the integer exchange terms and the
    //      heat_atten survival. No RGB, no direction accumulation, no gas
    //      optics, no `exp`. This is what pays for the long rays.
    //   2. THE VISIBLE-LIGHT CAST (cast_source_directional with radiation OFF)
    //      — the OLD range formula (`range_base + range_per_intensity·I`) and
    //      the OLD light machinery, unchanged. It is GOLDEN-NEUTRAL FOR RENDER:
    //      identical LightSource, identical angles, identical march ⇒ the
    //      light_rgb / light_dx / light_dy this function writes are bit-for-bit
    //      what the combined cast wrote before. It is SKIPPED ENTIRELY when
    //      `light_rgb == nullptr`, which is the zero-cost route for a caller
    //      that only wants the books.
    //
    //      (NOTE for the reader: the live sim path — PhysicsRunner.
    //      cast_fire_heat — passes scratch light buffers and DISCARDS them; the
    //      renderer draws fire light from its own blackbody selector,
    //      renderer/fire_lights.py. The light cast is kept here so this C++ API
    //      keeps its contract and any light-consuming caller is unaffected.)
    // RETURNS: the number of emission rays that terminated on a CONTACT FACE
    // (rule 3) this cast. A pure diagnostic — contact directions are
    // non-participating BY DESIGN — but the equivalence gate needs it to derive
    // its a = 0.5 tolerance instead of guessing one, and it is the honest
    // measure of how much of a given geometry's telescoping identity is
    // legitimately not covered.
    int64_t cast_from_fire_plane(
        const int32_t* fire, int h, int w,
        int fire_ray_count,
        double range_base, double range_per_intensity,
        double intensity_base, double intensity_per_intensity,
        const float color[3],
        float* light_rgb,
        float* light_dx,
        float* light_dy,
        float* smoke_glow,          // RGB god-ray glow, (h,w,3) or nullptr
        const float* gas_field,     // (n_gases, h, w) contiguous gas densities
        const float* gas_absorption,// (n_gases, 3) per-gas per-channel absorption
        const float* gas_scatter,   // (n_gases, 3) per-gas per-channel scatter
        int n_gases,
        const float* light_atten,   // per-tile static material atten (h,w,3)
        const float* heat_atten,    // per-tile heat atten (h,w) — a_x, REQUIRED
        // ---- P-R4 radiation planes ---------------------------------------
        const int32_t* temperature,     // Q16.16 (h,w) — both ends' E° source
        const int32_t* heat_inv_shift,  // (h,w) — the limiter's per-end budget
        const bool* thermal_solid,      // (h,w) — the warm-emitter mask
        int32_t* rad_net,               // Q16.16 (h,w) — SIGNED accumulator
        int32_t* rad_amb,               // (h,w) — the SKY ledger (rule 4)
        int32_t* rad_flux,              // D3: (h,w) positive-only damage sensor
        // D4 (amendment 5): the SIM TICK, as a plain integer. The per-source
        // fan phase rotates with it, so the discrete view factor time-averages
        // and no tile pair is permanently disconnected (see build_fire_sources).
        int tick,
        double jitter = 0.0
    ) const;

    // ---- CUDA-S2 gate: host ray-list builder (shared CPU/GPU angle math) ----
    //
    // Replicates cast_source_directional's per-ray loop EXACTLY — same
    // get_ray_count(), the same (i+0.5)/N angle sweep, the same jitter RNG
    // (mt19937 seeded (unsigned)(src.x*1000+src.y), uniform_real(-1,1)*jitter),
    // the same falloff angular_atten, the same inv_n normalisation — and folds
    // angle->(cos,sin), angular_atten, color and /N into each RayHD's
    // (dx,dy,e_r,e_g,e_b,heat_emit). Rays with angular_atten<=0 are SKIPPED, just
    // as the CPU cast skips them. Because this runs in THIS /fp:strict TU (the one
    // that already owns the identical angle math in cast_source_directional), the
    // GPU march's host-precomputed dx=cos(angle)/dy=sin(angle) are bit-identical
    // to what the CPU march_ray_directional computes internally from `angle` —
    // which is the contract that makes the DDA tile path (hence heat) match.
    // P-R4: `rs` (nullable) folds the emitter's radiation payload into every
    // RayHD alongside the light/heat budgets — the device march then runs the
    // identical exchange with no extra per-source lookup.
    std::vector<breach_cuda::RayHD> build_ray_list(
        const LightSource& src, const RadSource* rs = nullptr) const;

    // ---- P-R1: CUDA twin of cast_from_fire_plane ----
    //
    // The SAME enumeration + per-source parameter construction as
    // cast_from_fire_plane (build_fire_sources, shared — the float-parity-
    // critical code path runs exactly once for both backends), folded into
    // RayHD via build_ray_list and concatenated in row-major source order —
    // IDENTICAL to how cuda_raycaster_cast_batch (bindings.cpp, S8c item 1)
    // concatenates a Python-supplied source list, except the source list is
    // now built FROM THE FIRE PLANE here instead of supplied by Python. The
    // caller feeds the result straight into the existing
    // breach_cuda::raycaster_cast_directional batched march — no new device
    // code, no march/law change.
    // P-F1a: `emit_mask_out` (nullable) receives the once-per-tick emitter mask
    // plane so the caller can upload it alongside the rays — rule 2's branch
    // reads it on the device exactly as the CPU march does.
    //
    // `light_rays_out` (nullable) receives the SECOND, SHORT cast: the same
    // sources' VISIBLE-LIGHT rays on the legacy `range_base +
    // range_per_intensity·I` formula, carrying no radiation payload
    // (`rad_src_idx < 0`). The device runs them through the UNCHANGED
    // raycaster_cast_directional; the returned (primary) list is the pure
    // emission set, for the fast-path kernel.
    std::vector<breach_cuda::RayHD> build_fire_ray_list(
        const int32_t* fire, int h, int w,
        int fire_ray_count,
        double range_base, double range_per_intensity,
        double intensity_base, double intensity_per_intensity,
        const float color[3],
        // P-R4: the same three planes the CPU builder reads (temperature for
        // T_s/E_s, heat_atten for a_s, heat_inv_shift for the limiter) plus
        // the warm-emitter mask. Every RayHD comes back carrying its emitter's
        // payload, so the device march needs no source table.
        const int32_t* temperature,
        const float* heat_atten,
        const int32_t* heat_inv_shift,
        const bool* thermal_solid,
        int tick,                       // D4: the fan's per-tick phase rotation
        double jitter = 0.0,
        std::vector<uint8_t>* emit_mask_out = nullptr,
        std::vector<breach_cuda::RayHD>* light_rays_out = nullptr
    ) const;

    // Normalize direction vectors in place: (dx, dy) /= length(dx, dy).
    // Tiles with zero-length direction stay (0, 0).
    static void normalize_directions(
        float* light_dx, float* light_dy,
        int h, int w
    );

private:
    void march_ray(
        float sx, float sy, float angle,
        float ray_intensity, float max_range,
        float* light_map,
        const float* smoke_field,
        const bool* is_wall,
        int h, int w
    ) const;

    void march_ray_directional(
        float sx, float sy, float angle,
        float ray_intensity, float max_range,
        const float color[3],
        float heat_emit,            // src.heat: 0 = no heat deposit on this ray
        float* light_rgb,
        float* light_dx, float* light_dy,
        int32_t* heat,              // Q16.16 fixed-point, (h,w) or nullptr
        float* smoke_glow,          // RGB god-ray glow, (h,w,3) or nullptr
        const float* gas_field,     // (n_gases, h, w) contiguous gas densities
        const float* gas_absorption,// (n_gases, 3) per-gas per-channel absorption
        const float* gas_scatter,   // (n_gases, 3) per-gas per-channel scatter
        int n_gases,
        const float* light_atten,   // per-tile static material atten (h,w,3)
        const float* heat_atten,    // per-tile heat atten (h,w) or nullptr
        int h, int w,
        const RadCtx* rad,          // P-R4: nullptr / inactive == exchange off
        const RadRay* rr            // P-R4: this ray's emitter payload
    ) const;

    // ---- P-F1a: THE PURE-RADIATION FAST PATH (v7 rule 4 / round-3.6 MAJOR-3)
    //
    // The emission ray. Same DDA, same det_cos/det_sin directions, same
    // `heat_atten` survival and `heat_cull` floor as march_ray_directional —
    // and NOTHING else. No light_rgb, no light_dx/dy, no smoke_glow, no gas
    // absorption/scatter loop, no `exp`. That is what makes a >= 287-tile ray
    // affordable: an air cell costs one DDA step, two float multiplies and a
    // bounds test.
    //
    // It is also where the three structural rules live that the combined march
    // never had: the explicit distance-0 SELF-CELL exclusion, rule 3's CONTACT
    // TERMINATION, and rule 4's SKY charge at the grid edge.
    //
    // `max_range` is RADIATION_RANGE (the caller's `radiation_range`), NOT the
    // per-intensity formula. D3's rad_flux write keeps the legacy reach through
    // `rr->damage_range`.
    // Returns 1 if this ray terminated on a CONTACT FACE (rule 3), else 0 — the
    // contact-termination counter the equivalence gate's a = 0.5 variant needs
    // to DERIVE its tolerance rather than guess one (v7.1 item 5, M5). Contact
    // directions are non-participating by design, so their residual is charged
    // to nobody; the count is how much of the telescoping identity a given
    // geometry legitimately does not cover.
    int march_ray_radiation(
        float sx, float sy, float angle, float max_range,
        const float* heat_atten,    // a_x — REQUIRED (Kirchhoff)
        int h, int w,
        const RadCtx* rad,
        const RadRay* rr
    ) const;

    // ---- P-R1: the shared fire-plane source enumerator ----
    //
    // Enumerates fire[row*w+col] > 0 in ROW-MAJOR order (row outer, col
    // inner — the same order np.nonzero(fire > 0) yielded to the old Python
    // loop) and builds ONE LightSource per burning tile, reproducing
    // PhysicsRunner.cast_fire_heat's old per-tile Python math EXACTLY:
    //   x = col + 0.5, y = row + 0.5
    //   max_range  = range_base + range_per_intensity * I
    //   angle_center = ((col*7 + row*13) % ray_count) * (2*pi/ray_count)
    //   intensity  = intensity_base + intensity_per_intensity * I
    //   heat       = k_fire_heat * I
    //   jitter = 0, angle_spread = 2*pi (omni), ray_count = fire_ray_count
    // where I = float(fire_q) / 65536 (Q16.16 dequant, fire_fixed.FP_ONE).
    //
    // FLOAT-PARITY CONTRACT (the reason this function exists rather than
    // just inlining floats): the old Python loop computed every expression
    // above in DOUBLE (Python floats are C doubles; math.pi is a double)
    // and narrowed to float32 ONLY at the final `src.field = <python
    // float>` pybind attribute set. To land the identical float32 bits,
    // every expression here is DOUBLE arithmetic, with the same operator
    // shapes/order as the Python source, cast to `float` ONLY at the point
    // that mirrors that pybind narrowing. The dial parameters are `double`
    // in this signature for the same reason: the Python runner's attributes
    // (self.k_fire_heat etc.) stay double for their whole lifetime and are
    // never pre-narrowed to float32 before this multiply — accepting them
    // as `float` here would round a tick early and could flip the final
    // float32 bit. `color` has no arithmetic before it lands on the source
    // (a pure passthrough constant either way), so it is safely `float`.
    //
    // P-R4 (ruling A1.8): the enumeration widens to `burning ∪ (thermal_solid
    // && T >= T_emit_gate)` and every source also yields a RadSource into
    // `rad_out` (same index, same order). `heat` is GONE from the payload —
    // `k_fire_heat` no longer exists — and a WARM (non-burning) emitter uses
    // I = 0 in the range/intensity formulas, i.e. max_range = range_base: the
    // documented interim choice (short reach for a merely-warm surface; the
    // ruling defers a per-emitter reach model).
    //
    // D4 — THE PER-TICK FAN PHASE ROTATION (amendment 5, Erik's ruling).
    // The shipped phase `((x*7 + y*13) mod N) * (2*pi/N)` is a NO-OP for a
    // full-circle fan: rotating N evenly-spaced rays by a multiple of their own
    // spacing maps the set onto itself, so every source in the world casts the
    // SAME N directions and some tile pairs are NEVER connected (measured: no
    // ray reaches the (+2, 0) axis neighbour, at any intensity — pre-existing
    // painter aliasing that became load-bearing once spread depended on it).
    // The fix adds a sub-spacing rotation that advances with the tick:
    //     angle_center = (hash mod N)*(2*pi/N) + (tick mod N)*(2*pi/(N*N))
    // Over N consecutive ticks the fan sweeps one full ray spacing, so EVERY
    // direction is sampled and the discrete view factor time-averages to the
    // continuous one. Deterministic and a pure function of (x, y, tick) — the
    // tick arrives as a plain integer so CPU and CUDA build identical fans.
    std::vector<LightSource> build_fire_sources(
        const int32_t* fire, int h, int w,
        int fire_ray_count,
        double range_base, double range_per_intensity,
        double intensity_base, double intensity_per_intensity,
        const float color[3], double jitter,
        const int32_t* temperature, const float* heat_atten,
        const int32_t* heat_inv_shift, const bool* thermal_solid,
        int tick,
        std::vector<RadSource>* rad_out,
        // P-F1a / v7.1 item 13: the once-per-tick EMITTER MASK PLANE, filled
        // here (h*w bytes) from the SAME temperature snapshot the E° lookups
        // read, by the SAME integer threshold the caster predicate uses. Both
        // backends get it from this one function, so rule 2's half-weight
        // branch can never key on a different set on GPU than on CPU.
        std::vector<uint8_t>* emit_mask_out = nullptr
    ) const;

    // P-R4: the baked E° table + the `rad_scale` it was baked at. `mutable` so
    // the const cast entry points can lazily (re-)bake — the bake is a PURE
    // function of `rad_scale`, so this is a cache, not hidden state.
    mutable std::vector<int64_t> e_table_;
    mutable double e_table_scale_ = 0.0;   // 0 == never baked
};
