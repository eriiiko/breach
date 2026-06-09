#pragma once
// Temperature solver — turns the per-tick `heat` deposit into a persistent
// `temperature` field on solids (engine/06 §1), then spreads it by CONDUCTION
// (engine/06 §2; temperature_design_proposal §2).
//
// STEP A scope: the heat -> temperature CONVERSION pass (shipped).
// STEP B scope (this file): the CONDUCTION RELAXATION pass, run AFTER the
// conversion, per the proposal §6 order. Ambient cooling (§3) and unit damage
// (§4) are LATER steps and will be added to step() as further passes.
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

#include <cstdint>
#include <vector>

class TemperatureSolver {
public:
    // Sentinel face shift: grid edge or κ==0 on either side -> no conduction.
    // MUST match config [physics.thermal].NO_FACE (bound via set_no_face).
    int no_face = 63;

    void set_no_face(int v) { no_face = v; }
    int  get_no_face() const { return no_face; }

    // One tick of thermal work.
    //   Pass 1 — heat -> temperature conversion (§1.2), solids only.
    //   Pass 2 — conduction relaxation (§2.2), gather + double-buffered.
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
    //   solid       : bool, (h, w). The physics solid mask. Conversion runs on
    //                 solids only; air is skipped (stays 0). (Conduction needs no
    //                 solid branch — air faces are all NO_FACE.)
    void step(
        int32_t* temperature,
        const int32_t* heat,
        const int32_t* heat_inv_shift,
        const int32_t* face_shift,
        const bool* solid,
        int h, int w
    ) const;

private:
    // Double-buffer scratch for the conduction gather (temp -> temp_new). Owned
    // by the solver, resized on demand; reused across ticks (no per-tick alloc).
    // `mutable` so the const step() can use it as pure scratch.
    mutable std::vector<int32_t> scratch_;
};
