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
    when a TU reaches 0/0/0 this becomes a true "no float here" gate for that TU.

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
# Per-TU baseline: number of LINES containing each token (see the header). The
# ratchet fails if any count rises above its baseline; LOWER as each solver
# migrates. (Counts below recorded 2026-06-25 on s2-atmosphere-fixedpoint.)
BASELINE = {
    "atmosphere_solver.cpp":  {"float": 62, "double": 17, "fp:fast": 1},
    "smoke_dynamics.cpp":     {"float": 25, "double": 15, "fp:fast": 0},
    "fire_simulation.cpp":    {"float": 31, "double": 2,  "fp:fast": 0},
    "water_solver.cpp":       {"float": 32, "double": 23, "fp:fast": 1},
    "temperature_solver.cpp": {"float": 2,  "double": 1,  "fp:fast": 0},
    "physics_engine.cpp":     {"float": 66, "double": 25, "fp:fast": 1},
}


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
