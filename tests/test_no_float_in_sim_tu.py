"""S0 scaffold — the "no float in a sim TU" ratchet (a BASELINE, not yet a gate).

PURPOSE / END STATE
-------------------
The fixed-point migration drives the simulation solver translation units to
INTEGER arithmetic, TU by TU, so that lockstep determinism becomes BIT-IDENTICAL
across machines (today it is only same-machine bit-identical under /fp:precise).
The end state is a HARD gate: each sim solver TU contains ZERO ``float`` /
``double`` / ``/fp:fast`` once it has gone integer.

We are not there yet — the solvers are still float. So this scaffold is a
RATCHET, not an enforced zero-gate:

  * It records the CURRENT per-TU baseline counts of ``float`` / ``double`` /
    ``fp:fast`` occurrences (below, in BASELINE).
  * It PASSES as long as no count EXCEEDS its baseline — tolerating the existing
    float while preventing NEW float from creeping in.
  * It FAILS if a count goes UP (new float added to a TU that is supposed to be
    shrinking toward integer).
  * As each solver is migrated to integer, drop its baseline numbers toward 0;
    when a TU reaches its integer-end-to-end floor it becomes a HARD "no NEW float
    here" gate for that TU. As of S3c, fire_simulation.cpp + temperature_solver.cpp
    are MIGRATED (integer end-to-end) — their residual counts are the DOCUMENTED
    EXCEPTIONS the plan allows (the `float dt` boundary cast, FireParams float
    members, the load-time `quantize`/`make_recip` double precompute, render/glow),
    pinned by test_migrated_tus_at_documented_floor (a real-float regression trips
    it). The whole sim FIELD path is now integer after S3c.

It is a GUARD MECHANISM, deliberately simple and robust — a whole-word line
scan, NOT a C++ parser. It counts LINES that mention the token (a line with two
``float``s counts once), which is all a drift-detecting ratchet needs. Comments
that say "float" count too; that is fine and conservative (the migration removes
the comments along with the code).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_no_float_in_sim_tu.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CPP_SRC = ROOT / "cpp" / "src"

# The simulation solver translation units the migration must drive to integer.
# (Render-only TUs like raycaster.cpp and the pure-glue bindings.cpp are NOT in
# scope — they are not part of the synced lockstep state.)
SIM_TUS = (
    "atmosphere_solver.cpp",
    "smoke_dynamics.cpp",
    "fire_simulation.cpp",
    "water_solver.cpp",
    "temperature_solver.cpp",
    "physics_engine.cpp",
)

# Whole-word ``float`` / ``double`` (so "floating" / "doubled" in prose don't
# match) and the literal ``/fp:fast`` build pragma anywhere on a line.
_FLOAT_RE = re.compile(r"\bfloat\b")
_DOUBLE_RE = re.compile(r"\bdouble\b")
_FP_FAST_RE = re.compile(r"fp:fast")

# Per-TU baseline: number of LINES containing each token, recorded 2026-06-24 on
# the s0-prereq-gate branch (the code is still float — these are the numbers the
# migration will drive to zero, TU by TU). The ratchet fails if any count rises
# above its baseline here; LOWER the numbers as each solver goes integer.
# S1 (water -> Q16.16, 2026-06-24) re-baselined water_solver.cpp + physics_engine
# .cpp. The water TRANSPORT core is now integer (float lines 55 -> 32: the
# remainder is the render-only step_ripple + the gated head-term FLOAT BRIDGE +
# the constant-precompute signatures), but `double` ROSE 0 -> 23 because the
# load-time constants (max_dt_q's correctly-rounded sqrt, the per-step g*dt /
# damp*dt / dt-over-dx / tilt-position quantize) are computed ONCE in double then
# quantized — the LOCKED S1 decision (IEEE double is bit-identical cross-machine
# for these scalar/constant computations; no per-cell float). physics_engine.cpp
# `float` rose 64 -> 68: the four W5/W3 dequantize FLOAT BRIDGES (atmosphere/gas/
# dyn_permeability are still float until S2). These doubles/floats are at
# load-time/boundary scope, NOT per-cell transport — when S2 lands the bridges
# collapse to integer. The remaining counts shrink further as S2/S3 migrate.
# S2a (wave -> Q16.16, 2026-06-25) re-baselined atmosphere_solver.cpp +
# physics_engine.cpp. The explicit WAVE core (wave_substep + mean_wp) is now
# integer: atmosphere_solver.cpp `float` fell 68 -> 57 (the wave kick/feed/absorb/
# transfer + the sponge wave decays went integer; the remainder is the still-float
# S2c body — the GS diffusion, the atmosphere sponge, the residual hook, the wind
# term — plus the permeability/wave_absorb FLOAT BRIDGES on the per-face weights
# and the atmosphere-transfer/wind dequantize bridges). `double` ROSE 0 -> 17: the
# per-substep Q16.16 constant precompute (c_sq*dt, damp*dt, dt, transfer*dt,
# feed_rate*dt, the absorb/sponge decays — computed ONCE in double then quantized,
# the LOCKED S1 idiom: IEEE double is bit-identical cross-machine for scalar
# constants, no per-cell float). `fp:fast` 2 -> 1 (a stale comment removed).
# physics_engine.cpp `float` rose 68 -> 71: the wave_p dequantize FLOAT BRIDGE
# (step_tail ripple + step_water head term read wave_p as float, both already
# float bridges). All of these collapse to integer when S2c lands. The counts
# shrink further as S2b/S2c migrate. (water_solver.cpp `double` is 22 now, < its
# 23 baseline — a stale-low, non-failing; left for a water-side tidy.)
# NOTE the atmosphere `float` 57 -> 62 vs the naive integer-only path: the
# wave->atmosphere TRANSFER deposit is done in float at the bridge (dequantize the
# integer anomaly, multiply by the real transfer*dt) ON PURPOSE — a TRUNCATING
# integer mul_q16 deposit DC-leaks (every cell loses 1 LSB toward -inf -> a
# systematic ~percent/tick sink into the conserved atmosphere). The float deposit
# is unbiased + minimises divergence from the float build; it becomes a
# round-to-nearest/conservative INTEGER deposit when atmosphere goes integer (S2c).
# S2b (smoke/gas -> Q16.16, 2026-06-25) re-baselined smoke_dynamics.cpp +
# fire_simulation.cpp + physics_engine.cpp. The smoke/gas TRANSPORT core (the
# integer-SL: DDA march + integer bilinear + reciprocal_q16 renorm + the integer
# wind-coupled diffusion) is now integer:
#   * smoke_dynamics.cpp `float` FELL 47 -> 25 and `fp:fast` 1 -> 0 (the whole
#     advect/diffuse/clamp went integer Q16.16); `double` ROSE 1 -> 15 because the
#     FLOAT BRIDGE constants are computed ONCE per cell in double then quantized:
#     the wind*dt_adv displacement (-wind*dt_adv -> Q16.16) and the wind-coupled
#     d_eff*dt diffusion coefficient (|wind|² is still float until S2c). The
#     permeability per-face weight is also a float-bridge quantize. These are the
#     wind/permeability FLOAT BRIDGES that collapse to integer when S2c lands.
#   * fire_simulation.cpp `float` 29 -> 31 (the smoke param is int32 now, +a couple
#     of comment lines), `double` 0 -> 2: the smoke EMISSION delta
#     (smoke_emission*dt*I) is computed in double then quantized to a Q16.16 add
#     (order-free, deterministic) — a FLOAT BRIDGE the brief leaves open (the fire
#     system migrates later).
#   * physics_engine.cpp `float` 71 -> 66 (the gas/steam float bridges shrank — the
#     gas planes + the W5 steam puff are int32 now, so fewer `float* gas` lines),
#     `double` 24 -> 25 (the steam-puff quantize). The remaining bridges collapse
#     when the fire/steam systems migrate.
# S2c (atmosphere/wind -> Q16.16 + COLLAPSE every S2 float bridge, 2026-06-26)
# re-baselined atmosphere_solver.cpp + smoke_dynamics.cpp + physics_engine.cpp.
# This is the CLOSER of the S2 group — atmosphere + wind are now integer, so the
# whole atmosphere/wave/wind/smoke/gas group is cross-GPU deterministic (only the
# downstream FIRE bridge remains, S3).
#   * atmosphere_solver.cpp `float` FELL 62 -> 32: the RB-GS diffusion, the wind
#     gradient, the sponge/vac atmosphere scales, AND the wave->atmosphere transfer
#     all went integer (the wind's wave_p dequantize bridge + the float transfer
#     deposit are GONE). `double` ROSE 17 -> 30: the per-tick/per-substep Q16.16
#     CONSTANT precompute folds (mu = d_atm*dt, eta + the three sponge factors, the
#     residual-ratio readout) — computed ONCE in double then quantized, the LOCKED
#     S1 idiom (IEEE double is bit-identical cross-machine for scalar constants; no
#     per-cell float). The remaining `float` 32 is the still-float permeability
#     per-face weight quantize (permeability is a structural cache, not yet
#     migrated) + the const signatures + comments. `fp:fast` 1 is a stale comment.
#   * smoke_dynamics.cpp `float` 25 -> 24, `double` 15 -> 13: the wind FLOAT BRIDGE
#     collapsed (the advection displacement + the |wind|² diffusion read are integer
#     now — wind is Q16.16); the remaining `double` is the dt_adv / d_eff*dt scalar
#     folds + the permeability per-face quantize.
#   * physics_engine.cpp `float` 66 -> 65, `double` 25 -> 31: the W5/W3 atmosphere
#     reads/scales went int<->int (boil threshold compare, the P*V mul_q16), and the
#     FIRE BRIDGE (the ONE float bridge S2 leaves open) dequantizes atmosphere/wind
#     to float scratch for the float fire+temperature then re-quantizes the plume —
#     a load/boundary double fold, not per-cell transport. The water-head bridge
#     also dequantizes atmosphere now (k_p != 0). All collapse when fire migrates.
# Per-TU baseline: number of LINES containing each token (see the header). The
# ratchet fails if any count rises above its baseline; LOWER as each solver
# migrates. (Counts below recorded 2026-06-26 on s2-atmosphere-fixedpoint.)
# S3a (fire FIELD -> Q16.16 + the Python ignition twin, 2026-06-27) re-baselined
# physics_engine.cpp ONLY. fire is now int32 Q16.16, but the C++ FireSimulation
# logistic stays FLOAT for this commit (S3a flips the representation + the Python
# ignition O2 mean, not the C++ math). So step_tail gains a TEMPORARY internal
# FIRE FIELD BRIDGE (the S2 internal-bridge discipline): dequantize the int32 fire
# into the reused fire_f_ scratch, run the still-float fire.step on it, re-quantize
# back. physics_engine.cpp `float` 65 -> 68 (the three bridge comment lines) and
# `double` 31 -> 32 (the `quantize((double)fire_f_[i])` re-quantize fold — a
# load/boundary double, not per-cell transport). fire_simulation.cpp is UNCHANGED
# (still 31/2 — its logistic is float until S3b). Both bridge counts collapse in
# S3b (the C++ logistic goes integer) / S3c (the bridge buffers + the atm/wind
# bridges are deleted, and fire_simulation.cpp + temperature_solver.cpp join the
# hard 0/0/0 gate per the plan §S3c CI ratchet).
# S3b (the C++ fire LOGISTIC -> integer Q16.16 + sqrt_q16, 2026-06-27) re-baselined
# fire_simulation.cpp + physics_engine.cpp. The fire logistic is now INTEGER
# end-to-end (fire/wall_hp int32; it reads atmosphere/wind/temperature int32 and the
# new sqrt_q16 floor-isqrt for W):
#   * fire_simulation.cpp `float` FELL 31 -> 6 (only the `float dt` step-arg signature
#     + the FireParams float members' comment lines remain; the whole logistic +
#     gates + deposits went integer Q16.16). `double` ROSE 2 -> 18: the LOAD-TIME
#     CONSTANT precompute — every config param/threshold/reciprocal is computed ONCE
#     in double then quantized (quantize((double)p.x), make_recip((double)p.x)), the
#     LOCKED S1 idiom (IEEE double is bit-identical cross-machine for scalar
#     constants; NO per-cell float). fire_simulation.cpp is now a candidate for the
#     hard 0/0/0 gate once the `float dt` arg is also retired (S3c per the plan).
#   * physics_engine.cpp `float` 68 -> 66, `double` 32 -> 30: the S3a fire-field
#     bridge + the S2c atm/wind float bridges that fed the fire are GONE (fire reads
#     int32 directly). The ONLY float bridge left in step_tail is the temperature
#     pass's atmosphere read (atm_f_, dequantized POST-plume) — S3c retires it when
#     temperature goes integer, then fire_simulation.cpp + temperature_solver.cpp
#     join the hard 0/0/0 gate (plan §S3c CI ratchet).
# S3c (collapse the fire bridge — the CLOSER of S3, 2026-06-27) re-baselined
# temperature_solver.cpp + physics_engine.cpp, and MIGRATED fire + temperature
# into the ratchet at their DOCUMENTED-EXCEPTION FLOOR (plan §S3c, watch item #3).
# After S3c there is NO float bridge left inside the sim FIELD path (water + S2
# group + fire + temperature are all integer); only the Q2-fenced Python combat HP
# math + render/cosmetic + the documented boundary casts remain.
#   * temperature_solver.cpp `float` 2 -> 3, `double` 1 -> 2: the atmosphere arg
#     went `float* -> const int32_t*` and the vacuum-exposure threshold is now a
#     Q16.16 INTEGER compare (atmosphere[n] < quantize(o2_vacuum_thresh)). This TU
#     is now FULLY INTEGER — every `float`/`double` line that remains is either a
#     COMMENT ("was float", "double-buffered" prose) or the ONE documented boundary
#     cast `quantize((double)o2_vacuum_thresh)` (a load/boundary scalar cast, the
#     LOCKED S1 idiom — no per-cell float). The count rose only because S3c ADDED
#     explanatory comments; there is no new arithmetic. This is its MIGRATED FLOOR.
#   * physics_engine.cpp `float` 65 -> 62, `double` 30 -> 30: step_tail's atm_f_
#     dequantize bridge for the temperature pass is GONE (temperature reads int32
#     atmosphere directly), and the dead wind_x_f_/wind_y_f_/fire_f_ scratch decls
#     were deleted. The surviving `float`/atm_f_ use is the SEPARATE, documented
#     S2c WATER-HEAD bridge inside step_water (k_p·(atm+wave_p) reads atmosphere as
#     float) — NOT a fire/temperature bridge; it retires with a later water unify.
# MIGRATED-FLOOR TUs (plan §S3c CI ratchet): fire_simulation.cpp +
# temperature_solver.cpp are now INTEGER end-to-end. Their residual float/double
# counts are NOT a "still float" baseline — they are the DOCUMENTED EXCEPTIONS the
# plan explicitly allows: the `float dt` step-arg boundary cast, the FireParams
# float-member declarations/comments, the load-time `quantize((double)param)` /
# `make_recip((double)param)` constant precompute (the LOCKED S1 idiom), and the
# render/glow boundary. A later patch that adds REAL per-cell float arithmetic to
# either TU pushes the count above this floor and TRIPS the ratchet — that is the
# gate. See MIGRATED_FLOOR_TUS below.
# BEDROCK CLIFF-PATCH (integerize the atmosphere/wave + smoke substep cliffs, 2026-06-27)
# re-baselined atmosphere_solver.cpp ONLY. The atmosphere/wave `n` and smoke `n_smoke`
# substep COUNTS moved from `double`+std::ceil to integer `fixedpoint::ceil_div` (like
# water already was). atmosphere_solver.cpp gains `max_dt_q()` — the wave-CFL bound as a
# Q16.16 CONSTANT (computed ONCE in double, `0.5/c`, an exact IEEE divide -> bit-identical
# cross-machine, then quantized: the LOCKED S1 "load-time const is free" idiom, NOT per-cell
# float). `double` 30 -> 32 (the max_dt_q bake). physics_engine.cpp `float` 62 -> 64: the
# n_smoke CFL now reads `d_smoke_max` from the config diffusion table in its native float
# (a read-only config scalar, immediately quantized once) + the substep-length `dt_smoke`
# boundary cast passed to .step() + the cliff comment lines; its cliff `double`s (the old
# dequantize + d_eff/dt_stable + double ceil) are GONE. `double` 30 -> 28: removing the two
# cliff doubles drops the count, so the baseline is TIGHTENED to its new EXACT value 28 (was
# 30, which left 2 of slack — now pinned at 28 so any new double in this TU trips immediately).
# The COUNTS themselves are now INTEGER: n = ceil_div(sim_time_q, max_dt_q) and
# n_smoke = fixedpoint::smoke_cliff_count(...) (128-bit-rational integer ceil). This
# COMPLETES the integer foundation: no `double` remains in the determinism-critical path.
#
# CUDA-S4a: physics_engine.cpp `float` 64 -> 65. The GPU smoke dispatch added in the
# gas loop passes the per-gas diffusion `(float)gas_diffusion[gi]` + the solver dials
# `this->smoke.wind_diffusion_scale` / `advection_rate` to the FREE-function
# breach_cuda::smoke_step (a free function takes the solver's scalar dials explicitly,
# exactly as the S3 water dispatch passes this->water.g/damping/... to water_step).
# This is NOT new per-cell sim float — it is the same config scalar the CPU branch
# already casts onto this->smoke.d_smoke; the GPU branch just names it once more. The
# whole dispatch is guarded by `#ifdef BREACH_HAS_CUDA` (the CPU-only build never sees
# it) and the smoke path is bit-identical CPU vs GPU (tol 0, tests/cuda_s4a_check.py).
#
# CUDA-S4b: physics_engine.cpp `float` 65 -> 66. The GPU sink_hop dispatch added in the
# K-hop sink loop passes the solver dial `this->smoke.sink_strength` (a float) to the
# FREE-function breach_cuda::smoke_sink_hop — exactly the same shape as the S4a step
# dispatch above (a free function takes the solver scalar explicitly). It is NOT new
# per-cell sim float; it is the same config scalar the CPU branch already reads off
# this->smoke. The dispatch is guarded by `#ifdef BREACH_HAS_CUDA` and the sink_hop is
# bit-identical CPU vs GPU (tol 0, tests/cuda_s4b_check.py).
# CUDA-S5: physics_engine.cpp `float` 66 -> 67. The GPU wave dispatch added in the
# n_wave loop passes `(float)dt_actual` to the FREE-function breach_cuda::
# wave_substep_gpu (the same boundary `dt` cast the CPU branch already makes — the
# new line carries the token). The solver dials it also forwards
# (this->atmos.c / damping / absorb_strength / transfer / feed_rate /
# max_source_per_step) are NOT new float — they are the same config scalars the CPU
# wave_substep already reads off this->atmos, and they sit on lines without the bare
# `float` token. It is NOT new per-cell sim float; the dispatch is guarded by
# `#ifdef BREACH_HAS_CUDA` and the wave substep is bit-identical CPU vs GPU (tol 0,
# tests/cuda_s5_check.py), incl. the mean_wp int64 reduction.
# CUDA-S7: physics_engine.cpp `float` 67 -> 68. The GPU diffuse_solve dispatch added
# in step 2 of run_substeps carries ONE comment line bearing the `float` token (it
# explains that last_gs_residual — a non-synced FLOAT diagnostic — is not recomputed
# on the GPU path). The solver dials the dispatch forwards to the FREE-function
# breach_cuda::diffuse_solve_gpu (this->atmos.d_atm / breach_rate / gs_iters) are NOT
# new float — they are the same config scalars the CPU diffuse_solve already reads off
# this->atmos, and they sit on lines without the bare `float` token. It is NOT new
# per-cell sim float; the dispatch is guarded by `#ifdef BREACH_HAS_CUDA` and the
# diffuse_solve (RB-GS + vacuum sponge + wind) is bit-identical CPU vs GPU (tol 0,
# tests/cuda_s7_check.py — all SIX synced fields, incl. the drift-free GS check).
# EOS refactor P2 (docs/eos_refactor_design.md §4, §8 patch P2): temperature_
# solver.cpp `float` 3 -> 4, `double` 2 -> 5. The unified-temperature gas-T
# rules add exactly the documented-exception categories this TU's floor
# already allows, same shape as fire's existing `float dt` arg:
#   float 3->4: ONE new line, the `float dt` step() arg (tick's elapsed
#     seconds) — the SAME boundary-cast category fire_simulation.cpp's `dt`
#     already carries (this TU simply gains its own now that gas-T advection
#     needs a tick length).
#   double 2->5: THREE new lines, all ONCE-PER-STEP scalar boundary casts
#     feeding the LOCKED S1 quantize/make_recip idiom (never per-cell):
#     `dt_adv = gas_advection_rate*dt` (the advection displacement scale,
#     mirrors SmokeDynamics' own dt_adv precompute), `c_v_safe`/`(double)c_v`
#     (feeds make_recip(c_v) once per step), and `n_floor_q =
#     quantize((double)n_floor_heat)` — the exact same shape as this file's
#     pre-existing `thresh_q = fixedpoint::quantize((double)o2_vacuum_thresh)`
#     line (already in the pre-P2 baseline). No per-cell float/double was
#     added; the gas-T hot loops (advection backtrace, radiation deposit) are
#     pure Q16.16 integer throughout (reciprocal_q16, mul_q16, recip_mul).
BASELINE = {
    "atmosphere_solver.cpp":  {"float": 32, "double": 32, "fp:fast": 1},
    "smoke_dynamics.cpp":     {"float": 24, "double": 13, "fp:fast": 0},
    "fire_simulation.cpp":    {"float": 6,  "double": 19, "fp:fast": 0},
    "water_solver.cpp":       {"float": 32, "double": 22, "fp:fast": 1},
    "temperature_solver.cpp": {"float": 4,  "double": 5,  "fp:fast": 0},
    "physics_engine.cpp":     {"float": 68, "double": 28, "fp:fast": 1},
}

# The TUs that have been MIGRATED to integer end-to-end (S3c). For these, the
# ratchet is a HARD gate AT THE DOCUMENTED-EXCEPTION FLOOR: the recorded BASELINE
# counts are not "remaining float to migrate" but the documented boundary casts +
# comments the plan §S3c explicitly allows (the `float dt` arg, FireParams float
# members, the load-time `quantize`/`make_recip` double precompute, render/glow).
# A separate test (test_migrated_tus_at_documented_floor) asserts these stay at or
# below their floor, so a later patch cannot smuggle a new float into migrated
# fire/temperature without tripping CI.
MIGRATED_FLOOR_TUS = ("fire_simulation.cpp", "temperature_solver.cpp")


def _count_tokens(path: Path) -> dict:
    """Count LINES mentioning each tracked token. Missing file -> all zero (the
    water solver could be renamed; a missing TU should not crash the guard)."""
    counts = {"float": 0, "double": 0, "fp:fast": 0}
    if not path.exists():
        return counts
    text = path.read_text(encoding="utf-8", errors="replace")
    for line in text.splitlines():
        if _FLOAT_RE.search(line):
            counts["float"] += 1
        if _DOUBLE_RE.search(line):
            counts["double"] += 1
        if _FP_FAST_RE.search(line):
            counts["fp:fast"] += 1
    return counts


def current_counts() -> dict:
    """Per-TU current token counts — usable as a small report from the CLI."""
    return {tu: _count_tokens(CPP_SRC / tu) for tu in SIM_TUS}


def test_sim_tus_exist():
    """The scan must actually be looking at the solver TUs (catch a rename/move
    that would silently make the ratchet vacuous)."""
    missing = [tu for tu in SIM_TUS if not (CPP_SRC / tu).exists()]
    assert not missing, (
        f"sim TU(s) not found under {CPP_SRC} — did a solver get renamed? "
        f"Update SIM_TUS/BASELINE: {missing}")


def test_no_new_float_in_sim_tus():
    """RATCHET: no tracked token count may EXCEED its recorded baseline.

    Tolerates the existing float (the migration is mid-flight); fails the instant
    NEW float/double/fp:fast is added to a sim solver TU. As each TU goes integer,
    lower its BASELINE — when it hits 0/0/0 this is a hard 'no float here' gate."""
    regressions = []
    for tu in SIM_TUS:
        cur = _count_tokens(CPP_SRC / tu)
        base = BASELINE[tu]
        for tok in ("float", "double", "fp:fast"):
            if cur[tok] > base[tok]:
                regressions.append(
                    f"{tu}: '{tok}' rose to {cur[tok]} (baseline {base[tok]}) — "
                    f"new float crept into a sim TU; either revert it or, if this "
                    f"TU is intentionally still float, justify and bump BASELINE")
    assert not regressions, (
        "no-float ratchet tripped (NEW float in a sim solver TU):\n  "
        + "\n  ".join(regressions))


def test_migrated_tus_at_documented_floor():
    """HARD GATE for the MIGRATED (integer end-to-end) TUs — fire + temperature.

    Per plan §S3c, fire_simulation.cpp and temperature_solver.cpp are now integer
    end-to-end. Their residual float/double counts are NOT a "still float" baseline
    to drive down — they are the DOCUMENTED EXCEPTIONS the plan allows (the
    `float dt` step-arg boundary cast, the FireParams float-member declarations, the
    load-time `quantize((double)param)` / `make_recip((double)param)` constant
    precompute — the LOCKED S1 idiom — plus comment lines + the render/glow
    boundary). This test pins those TUs AT OR BELOW their recorded floor: a later
    patch that adds REAL per-cell float arithmetic (a new `float` local, a fast-math
    pragma, a float field read) pushes a count above the floor and FAILS here — the
    point of making fire/temperature join the no-float check.

    (This is distinct from test_no_new_float_in_sim_tus, which is a soft per-TU
    ratchet across ALL sim TUs; this one is the SHARP, named gate for the two TUs
    the plan declares migrated, so a regression is reported as a migration breach.)"""
    breaches = []
    for tu in MIGRATED_FLOOR_TUS:
        cur = _count_tokens(CPP_SRC / tu)
        floor = BASELINE[tu]
        for tok in ("float", "double", "fp:fast"):
            if cur[tok] > floor[tok]:
                breaches.append(
                    f"{tu}: '{tok}' rose to {cur[tok]} (documented floor {floor[tok]}) "
                    f"— a NEW float crept into a MIGRATED (integer end-to-end) TU. "
                    f"The only float allowed here is the documented `dt` boundary "
                    f"cast / load-time quantize precompute / render boundary. If this "
                    f"is genuinely one of those, justify and raise the floor; "
                    f"otherwise it is a determinism regression — revert it.")
    assert not breaches, (
        "MIGRATED-TU float floor breached (plan §S3c — fire/temperature must stay "
        "integer end-to-end):\n  " + "\n  ".join(breaches))


def test_baseline_is_not_stale_low():
    """If a TU's real count drops BELOW its baseline (good — migration progress),
    the baseline is stale and should be tightened so the ratchet stays sharp.

    This is a SOFT nudge: it does not fail, it just reports. (Kept as a passing
    test with an informative message via ``pytest -rA`` / -s so progress is
    visible without blocking CI.)"""
    stale = []
    for tu in SIM_TUS:
        cur = _count_tokens(CPP_SRC / tu)
        base = BASELINE[tu]
        for tok in ("float", "double", "fp:fast"):
            if cur[tok] < base[tok]:
                stale.append(f"{tu}: '{tok}' now {cur[tok]} < baseline {base[tok]} "
                             f"(tighten BASELINE to lock in the win)")
    if stale:
        print("\n[no-float ratchet] baseline can be tightened:\n  "
              + "\n  ".join(stale))
    # Intentionally always passes — this is a nudge, not a gate.
    assert True


if __name__ == "__main__":
    import json
    print(json.dumps(current_counts(), indent=2))
