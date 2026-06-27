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
//   Conduction (§2.2, gather + double-buffer):
//       acc = Σ_{dir∈N,S,E,W}  (temp[n] - temp[i]) >> face_shift[i][dir]
//       temp_new[i] = temp[i] + acc            (then swap temp_new -> temp)
//   The DIFFERENCE is shifted (not the neighbour), so equal neighbours produce
//   EXACTLY 0 change (no drift) and the flux is conservative-shaped. A face is
//   skipped when face_shift == NO_FACE (grid edge, or κ==0 on either side), so
//   air (all faces NO_FACE) is a structural no-op (Σr = 0 -> unchanged). The
//   per-tile face_shift cache is baked at LOAD from the harmonic-mean face table
//   (all log2/division at load, in float); the runtime is a PURE signed add +
//   arithmetic right shift -> order-independent (gather over a frozen buffer),
//   bit-identical cross-machine. With SHIFT_MIN==2 (max face rate ¼) and 4
//   neighbours, Σr ≤ 1, so the update is a convex combination of {T_i, T_n} —
//   the discrete maximum principle holds (no new extremum ever created),
//   unconditionally stable for all time (proposal §2.6).
//
//   Ambient cooling (§3, gather over the geometric 4-neighbours):
//       shift = exposed ? cool_shift_vacuum : cool_shift
//       T    -= (T < 0) ? -((-T) >> shift) : (T >> shift)
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
    // o2_vacuum_thresh — atmosphere value below which a neighbour counts as
    //   vacuum for the exposure test (in the same REAL units as gmap.atmosphere,
    //   i.e. the pre-quantize pressure). It is a config dial (bound from Python as
    //   a real value); S3c quantizes it ONCE per step to a Q16.16 count and the
    //   exposure test is then a pure integer compare on the int32 atmosphere field.
    //   Kept as a float member because it is a config/boundary value, not synced
    //   per-cell state (the documented boundary exception, like fire's `dt`).
    int   cool_shift = 5;
    int   cool_shift_vacuum = 3;
    float o2_vacuum_thresh = 0.3f;

    void  set_cool_shift(int v) { cool_shift = v; }
    int   get_cool_shift() const { return cool_shift; }
    void  set_cool_shift_vacuum(int v) { cool_shift_vacuum = v; }
    int   get_cool_shift_vacuum() const { return cool_shift_vacuum; }
    void  set_o2_vacuum_thresh(float v) { o2_vacuum_thresh = v; }
    float get_o2_vacuum_thresh() const { return o2_vacuum_thresh; }

    // One tick of thermal work.
    //   Pass 1 — heat -> temperature conversion (§1.2), solids only.
    //   Pass 2 — conduction relaxation (§2.2), gather + double-buffered.
    //   Pass 3 — ambient cooling (§3), solids only, vacuum-exposure 1-bit.
    //
    //   temperature : Q16.16 int32, (h, w). Persistent field (ΔT above ambient;
    //                 T_ambient == 0, proposal §3.1). Mutated in place.
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
    //   solid       : bool, (h, w). The physics solid mask. Conversion and
    //                 cooling run on solids only; air is skipped (stays 0).
    //                 (Conduction needs no solid branch — air faces are all
    //                 NO_FACE.)
    //   is_vacuum   : bool, (h, w). The physics vacuum mask. A solid tile cools
    //                 at cool_shift_vacuum if ANY in-bounds 4-neighbour is vacuum
    //                 (§3.3). Same field the atmosphere/smoke solvers read.
    //   atmosphere  : int32 Q16.16, (h, w). The atmosphere field (S2c). A neighbour
    //                 with atmosphere < quantize(o2_vacuum_thresh) also counts as
    //                 vacuum-exposed — a pure integer compare (S3c: no float).
    void step(
        int32_t* temperature,
        const int32_t* heat,
        const int32_t* heat_inv_shift,
        const int32_t* face_shift,
        const bool* solid,
        const bool* is_vacuum,
        const int32_t* atmosphere,
        int h, int w
    ) const;

private:
    // Double-buffer scratch for the conduction gather (temp -> temp_new). Owned
    // by the solver, resized on demand; reused across ticks (no per-tick alloc).
    // `mutable` so the const step() can use it as pure scratch.
    mutable std::vector<int32_t> scratch_;
};
