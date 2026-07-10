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
    // n_floor_heat — floor on the per-tile N divisor (read from `atmosphere`,
    //   the P2 density proxy) for the SAME deposit divide, INDEPENDENT of any
    //   other floor in the system (design §4.3 / decisions.md item 7).
    float gas_advection_rate = 900.0f;
    float c_v = 1.0f;
    float n_floor_heat = 0.05f;

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
    //   Pass 2 — conduction relaxation (§2.2), gather + double-buffered.
    //            UNCHANGED CODE: air's newly-nonzero `conductivity` (config)
    //            makes this whole-grid pass do air<->air and solid<->air for
    //            free (P2 §4).
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
    //   solid       : bool, (h, w). The physics solid mask. Conversion and
    //                 cooling run on solids only; the P2 gas rules run on the
    //                 complementary open-air mask (!solid && !is_vacuum).
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
        float dt
    ) const;

private:
    // Double-buffer scratch for the conduction gather (temp -> temp_new). Owned
    // by the solver, resized on demand; reused across ticks (no per-tick alloc).
    // `mutable` so the const step() can use it as pure scratch.
    mutable std::vector<int32_t> scratch_;
    // P2: separate pre-advection snapshot scratch for the gas-T semi-Lagrangian
    // pass (Pass 0) — kept distinct from `scratch_` (the conduction double
    // buffer) so the two passes never alias each other's live data.
    mutable std::vector<int32_t> gas_scratch_;
};
