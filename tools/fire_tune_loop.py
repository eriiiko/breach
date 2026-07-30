#!/usr/bin/env python
"""Erik's manual fire-tuning loop (Fable, 2026-07-25).

Edit the TUNE block, run, read the scorecard, repeat until satisfied.
Reference: docs/fire_tuning_plan_2026-07-22.md §9 (the fire chain + the
tuning order + the targets). This wrapper drives tools/fire_timing_harness.py
via its CLI (--set overrides) — config.toml is NEVER touched; when a combo
is blessed, its values go to config in the close-out commit, not here.

Run (from a worktree root):
    conda run -n data python tools/fire_tune_loop.py

Tuning order (§9.3 — one group at a time, everything else frozen):
    1. STRUCTURE (once)   fire_T_ext=250 < ignition_temp, span 100
    2. THERMAL            k_fire_heat vs cool_shift -> flame T 400-500
    3. RAMP               k_grow / k_die -> peak ~0.5 @ ~3 min
    4. LIFETIME           wall_damage -> fire death 6-8 min
    5. VERIFY anchors     (burn_rate / X_ext / fuel_per_o2 / tau -- no tuning)

--------------------------------------------------------------------------
DEFAULTS REBASED 2026-07-30 by the thermal-mass-axis arc (P3 close).
Full measurement report: docs/thermal_mass_axis_bench_report_2026-07-30.md
--------------------------------------------------------------------------
The crate is now a THERMAL SOLID. The old TUNE block described the gas
regime (crate temperature = hot gas the plume advected away) — that was the
§9.5 regression, and it is FIXED (f5e9aa3 + 312e984 + 6f57762). What the
crate looks like now:

  * medium is chosen by `thermal_mass > 0`, not by `permeability <= 0`;
  * `temperature[]` on the crate is OWNED by the TemperatureSolver — the EOS
    pass no longer touches it (measured: 0.000 game/tick, every run);
  * the ambient decay is therefore the crate's ONE loss channel, and a REAL
    dial.

The equilibrium is exact (measured to +/-1% at equilibrium, 3 operating
points), so you can predict a move before you make it:

    T*(I) = k_fire_heat * I * 2^(cool_shift - heat_inv_shift)

    heat_inv_shift = log2(thermal_mass); furniture thermal_mass = 8 -> 3.
    So at cool_shift=5:  T* = 4 * k_fire_heat * I   (design §2.5's form).

--------------------------------------------------------------------------
DIAL MOVED 2026-07-30 by the COOL-SHIFT AXIS (the loss-side twin of
thermal_mass). READ THIS IF YOU HAVE THE OLD KEY IN MUSCLE MEMORY.
--------------------------------------------------------------------------
The decay shift is now a PER-MATERIAL column:

    --set materials.furniture.cool_shift=12      <-- the crate dial NOW
    --set physics.thermal.COOL_SHIFT=12          <-- NO LONGER MOVES THE CRATE

`[physics.thermal] COOL_SHIFT` is kept and still has two jobs — it is the
DEFAULT for a material row that omits the column, and (with COOL_SHIFT_VACUUM)
it defines the vacuum-exposure OFFSET — but every row in config.toml now
authors `cool_shift` explicitly, so the column WINS and overriding the global
alone changes nothing on the crate. Warning 1 below used to say this dial was
global; that WAS the problem, and it is what the axis fixed.

Two warnings before you turn a dial:

  1. cool_shift is PER MATERIAL now — that is the point. Move the crate
     alone with `materials.furniture.cool_shift` and hull/steel/glass/wood/
     doors keep their 1.3 s e-fold, so the big feel change (and the goldens
     it would move) stays confined to the thing you are tuning. Legal range
     [SHIFT_MIN=2, 20]; e-fold = 2^shift / 24 s (5 -> 1.3 s, 12 -> 171 s).
     A vacuum-exposed tile takes `max(2, cool_shift - 2)` — the 4x space
     discount stays one global rule, so each material keeps ONE dial.

  2. THE I_crit CLIFF (bench report §4). Deposit is linear in I and loss is
     linear in T, so T* is linear in I — and the hot gate opens at
     fire_T_ext. Below

         I_crit = I_peak * fire_T_ext / T_flame

     the gate closes and the fire self-collapses. At §9.3's own targets
     (T_flame 450 @ I_peak 0.5, fire_T_ext 250) that is I_crit = 0.278,
     nearly 3x ignition_seed = 0.1 — so the fire can neither ignite from
     the seed nor burn down gracefully. This is why design §2.5's
     k_fire_heat = 225 @ cool_shift = 5 is arithmetically right but
     DYNAMICALLY DEAD: measured, it snaps out at tick 1.

     The four levers are measured in bench report §4.3. Three of them
     (cool_shift ~12; ignition_seed into the band; fire_T_ext below
     T*(I_seed)) relocate the cliff; only the fourth — making the deposit
     non-linear in I — removes it, and that is a MODEL change, not a dial.
     Erik's call, at the joint re-tune.

THE TUNE BLOCK BELOW WAS RE-DERIVED 2026-07-30 for Erik's tuning session
(session seed `docs/fire_tuning_session_seed_2026-07-30.md`). Its values are
DERIVED FROM THE ANALYTICS, not measured — each carries its provenance inline.
The previous, measured cool_shift-12 set is preserved as ALT_MEASURED_CS12
below; ALT_* are all swap-in-wholesale (`TUNE.update(ALT_...)`).

--------------------------------------------------------------------------
REQUIRES — read this if the scorecard says n/a or every run reads I = 0
--------------------------------------------------------------------------
This file was committed to the repo for the first time by the P3 close-out
(2026-07-30). It had lived untracked in the fire-o2-integration worktree,
and it depends on TWO other files that are STILL untracked there:

  * tools/fire_timing_harness.py — the WARM-SEED build. Erik's copy adds
    `gmap.temperature[crate] = 280` at setup (a tile only ignites in-engine
    BECAUSE its T crossed ignition_temp; a cold seed is a bootstrap race the
    game never runs) plus three CSV columns this scorecard reads:
    `hot`, `Tfar_game`, `X_local`. The committed harness on this branch has
    NEITHER. Without the warm seed the crate starts at ambient and every run
    reads I = 0 — the scorecard will tell you so, loudly, rather than
    crashing.

  * tools/fire_tune_plot.py — the auto Kelvin plot. Absent, the run still
    completes and prints the scorecard; only the plot is skipped.

Both should be committed from that worktree — they are real tools, they are
referenced by docs/fire_tuning_plan_2026-07-22.md, and a fresh clone loses
them. P3 did not copy them here: they are another live session's uncommitted
work, and the project's worktree-hygiene rule says do not reach into one.

All measurements in the bench report were taken with standalone scratchpad
scripts that apply the warm seed themselves, so the numbers do not depend on
which harness build you have.
"""
from __future__ import annotations

import csv
import statistics
import subprocess
import sys
from pathlib import Path

# ===========================================================================
# THE TUNE BLOCK — edit me, run me.           (bare key -> [physics.fire];
#                                              dotted  -> that config section)
# ===========================================================================
TUNE = {
    # =====================================================================
    # STRUCTURE FACTS (re-measured 2026-07-30, live path, post-arc):
    #  * The crate is a THERMAL SOLID (thermal_mass 8 -> heat_inv_shift 3).
    #    The ambient decay applies to it; furniture kappa=0, so
    #    `materials.furniture.cool_shift` is its ONLY loss channel
    #    (ruling A5 — deliberate, one clean dial, now per material).
    #  * The EOS pass does NOT advect its temperature away any more.
    #    Measured EOS delta on the crate: 0.000 game/tick, all 70 runs.
    #  * fire_T_ext can go back to PHYSICAL values (§9.2's below-ignition
    #    rule still applies: ignition_temp furniture = 280).
    # =====================================================================

    # -- 1. STRUCTURE (§9.2 / §9.3 step 1). One-time; blessed values. --
    "fire_T_ext": 250.0,    # below ignition_temp (280) — ember-sustain region
    "fire_T_span": 100.0,   # hot = 1 above T = 350

    # -- 2. THERMAL operating point (§9.3 step 2). THE dial pair.
    #       T* = k_fire_heat * I * 2^(cool_shift - 3)   [+/-1% at equilibrium]
    #       Raising k_fire_heat raises the plateau AND the far-field rise (they
    #       scale together — far rise is the binding constraint, not the flame
    #       temperature). Since the COOL-SHIFT AXIS the shift is the FURNITURE
    #       row only: every other material keeps its 1.3 s e-fold, so a big
    #       number here does not follow you onto the hull.
    #
    #   DERIVED START, 2026-07-30 (session-seed handoff). NOT measured — this
    #   run is the first measurement. Provenance of each number:
    #
    #   k_fire_heat 33: from the verified analytic
    #       T* = k_fire_heat * I * 2^(cool_shift - heat_inv_shift),
    #   heat_inv_shift = log2(furniture thermal_mass 8) = 3, so at cool_shift 9
    #       T* = k * I * 2^6 = 64 k I;  at Erik's I anchor 0.21 -> T* = 13.4 k.
    #   The §9.3 flame band is 400-500 game, so k ~= 33.5 lands T* ~= 450
    #       (= 1193 K). 33 is that, rounded.
    "k_fire_heat": 33.0,
    #   cool_shift 9: e-fold 2^9/24 = 21.3 s — a physically plausible wood-
    #   surface time constant (12 = 171 s was the cliff-clearing crutch, not a
    #   wood number). Integer shift. NOTE THE KEY: the old
    #   `physics.thermal.COOL_SHIFT` no longer moves the crate (see the header).
    "materials.furniture.cool_shift": 9,

    # -- 3. RAMP (§9.3 step 3 — YOURS). Ratio sets peak, magnitude sets
    #       speed.
    #   k_die/k_grow = 0.080 is ERIK'S EXPLICIT CHOICE (2026-07-30). The
    #   logistic (fire_simulation.cpp:209-224) is
    #       dI/dt = I * [ k_grow*a*(1-I) - k_die*(1-a) ],  a = F * o2f * hot
    #   so the fixed point is  I_eq = 1 - (k_die/k_grow) * (1-a)/a  and the
    #   fire only sustains at all while  k_die/k_grow < a/(1-a).
    #   Since the full-response split (b340bba) o2f = (X-0.13)/(1.0-0.13), so
    #   ambient air gives o2f = 0.092. At F = 1, hot = 1 that is a = 0.092,
    #   a/(1-a) = 0.1013, and 0.080 lands I_eq = 0.210 — exactly Erik's anchor
    #   (0.49 @ X=0.25, 0.67 @ X=0.30, 1.00 at pure O2: normal air is not a
    #   maximum, which is the point).
    #   ** CAVEAT, measured 2026-07-30: F = clamp01(wall_hp/fuel_ref) and
    #   fuel_ref below is 40 while furniture hp is 30, so a FULL-HEALTH crate
    #   has F = 0.75, not 1. That drops a to 0.069, a/(1-a) to 0.0741 — BELOW
    #   0.080 — so the anchor's own arithmetic predicts a fire that cannot
    #   sustain on this crate at any intensity. fuel_ref is Erik's call, not
    #   this bench's; left at 40 and reported. **
    #   Realised as 0.35 / 0.028: the RATIO sets peak height, the MAGNITUDE
    #   sets ramp speed (magnitude carried from the previous measured point,
    #   which had its peak time in band).
    "k_grow": 0.35,
    "k_die": 0.028,

    # -- 4. LIFETIME (§9.3 step 4 — YOURS). NOT re-derived for the 2026-07-30
    #       start: carried over from the cool_shift-12 point (where it measured
    #       death at 332.7 s with wall_hp 4.5 left). Lifetime is stage 4 —
    #       settle steps 2-3 first, then move this. --
    "wall_damage": 0.083,

    # -- ANCHORED — verify, don't tune (see §9.3 for the paper trail) --
    "physics.combustion.burn_rate": 0.02,     # Huggett 1980 — THE O2-draw dial
    "physics.combustion.fuel_per_o2": 0.7,    # wood stoich (0.045 was stale)
    "o2_frac_ext": 0.13,                      # Peatross-Beyler 1997 extinction
    "o2_frac_full": 1.0,                      # full-response reference = PURE O2
                                              # (2026-07-30 split; NOT ambient —
                                              # ambient air now gives o2f 0.092,
                                              # so k_die/k_grow moves ~10x with it)
    "fuel_ref": 40.0,
    "k_wind_strip": 0.0,                      # plume self-blow-out off (2026-07-23)
}

# --- Other measured branches (bench report §4.3). Swap one in wholesale by
# --- doing  TUNE.update(ALT_...)  right here, then re-run.

# The P3 close-out's best MEASURED set (the one the 2026-07-30 re-derivation
# replaced above): 6 of 9 §9.3 targets passed — peak I 0.331 @ 143.9 s, plateau
# 414 game (1120 K), death 332.7 s with wall_hp 4.5 left. cool_shift 12 is a
# 171 s e-fold: it clears the I_crit cliff, but it is a crutch, not a wood
# number. Kept as the fallback if the derived start does not sustain.
ALT_MEASURED_CS12 = {
    "k_fire_heat": 2.2,
    "materials.furniture.cool_shift": 12,
    "k_grow": 0.35, "k_die": 0.06,
    "fire_T_ext": 250.0, "fire_T_span": 100.0,
}

# Keeps the decay shift at Erik's stated preference (6-7) and the blessed
# fire_T_ext — but the flame runs at 1007 game (2307 K), ~2x too hot, and
# lowering k_fire_heat from here kills the fire outright (floor is 175).
ALT_PREFERRED_COOL_SHIFT = {
    "materials.furniture.cool_shift": 7,
    "k_fire_heat": 175.0,
    "fire_T_ext": 250.0, "fire_T_span": 100.0,
}

# Lands the flame at 450 game (1193 K) and lives 208 s with the fuel actually
# consumed — but fire_T_ext = 90 is 473 K / 200 C, not a defensible
# flame-extinction temperature, and it fails gate (c) monotonicity (dip -46).
ALT_LOW_T_EXT = {
    "materials.furniture.cool_shift": 7,
    "k_fire_heat": 56.25,
    "fire_T_ext": 90.0, "fire_T_span": 100.0,
}

# Design §2.5's arithmetic point, for the record. Plateau 449 (dead-centre)
# ONLY if ignition_seed is raised to 0.4 — and then it dies at 48.5 s with
# 24.6 wall_hp left (the cliff moves to the decay end). With the stock
# ignition_seed = 0.1 this set snaps out at tick 1.
ALT_DESIGN_2_5 = {
    "materials.furniture.cool_shift": 5,
    "k_fire_heat": 225.0,
    "fire_T_ext": 250.0, "fire_T_span": 100.0,
    "ignition_seed": 0.4,
}

BENCH = {
    "--max-seconds": 900,      # 15 min window (death target is 6-8 min)
    "--tail-seconds": 60,
    "--sky-tau-s": 60,
    "--sponge-width": 8,
    # bench geometry: harness defaults are already sponge-safe (84x40, crate
    # deep at x=12); override here if needed: "--interior-w": 84, ...
}

# Targets for the scorecard (Erik 2026-07-24/25; §9.3).
T = {
    "peak_lo": 0.40, "peak_aim": 0.50, "peak_hi": 0.60,
    "peak_t_lo_s": 120.0, "peak_t_aim_s": 180.0, "peak_t_hi_s": 300.0,
    "death_lo_s": 360.0, "death_hi_s": 480.0,
    "flameT_lo": 400.0, "flameT_hi": 500.0,
    "far_rise_max": 20.0,
    "ntot_min": 0.90,
    "farX_min": 0.19,
    "I_min_snap": 0.02,
}

# ===========================================================================
ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = ROOT / "_fire_tuning_artifacts" / "tune_loop_last.csv"


def run_harness():
    cmd = [sys.executable, str(ROOT / "tools" / "fire_timing_harness.py"),
           "--csv", str(CSV_PATH)]
    for k, v in BENCH.items():
        cmd += [k, str(v)]
    for k, v in TUNE.items():
        # shift counts are INTEGERS (the loader rejects a fractional value)
        int_key = ("COOL" in k) or k.endswith("cool_shift")
        vs = str(int(v)) if float(v).is_integer() and int_key else str(v)
        cmd += ["--set", f"{k}={vs}"]
    CSV_PATH.parent.mkdir(exist_ok=True)
    print("[cmd]", " ".join(cmd), "\n")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stdout.write(r.stdout)
        sys.stderr.write(r.stderr)
        sys.exit(f"harness failed (exit {r.returncode})")
    return r.stdout


def load_series():
    rows = []
    with open(CSV_PATH, newline="") as f:
        for row in csv.reader(f):
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "t_s":
                hdr = row
                continue
            rows.append({h: float(x) for h, x in zip(hdr, row)})
    return rows


# Columns this scorecard wants that ONLY the warm-seed harness build writes.
# See the REQUIRES note in the module docstring: without them the run is still
# valid, the scorecard just reports n/a for those rows instead of crashing.
OPTIONAL_COLS = ("hot", "Tfar_game", "X_local")


def _col(s, name, reduce_fn, default=float("nan")):
    """reduce_fn over column `name`, or `default` if the harness didn't write it."""
    if not s or name not in s[0]:
        return default
    return reduce_fn([r[name] for r in s])


def metrics(s):
    m = {}
    m["missing_cols"] = [c for c in OPTIONAL_COLS if s and c not in s[0]]
    peak = max(s, key=lambda r: r["I"])
    m["peak_I"], m["peak_t"] = peak["I"], peak["t_s"]
    i_pk = s.index(peak)
    # fire death: first snap-to-zero AFTER the peak (edge-trigger world:
    # it must STAY dead — take the last transition into I<=snap).
    death = None
    for r in s[i_pk:]:
        if r["I"] <= T["I_min_snap"]:
            death = r["t_s"]
            break
    m["death_t"] = death
    # flame plateau T: median T_game while I >= 60% of peak (the burning core)
    core = [r["T_game"] for r in s if r["I"] >= 0.6 * m["peak_I"]]
    m["flame_T"] = statistics.median(core) if core else float("nan")
    core_rows = [r for r in s if r["I"] >= 0.6 * m["peak_I"]]
    m["hot_at_plateau"] = _col(core_rows, "hot", statistics.median, 0.0) if core else 0.0
    m["far_rise"] = _col(s, "Tfar_game", max)
    m["ntot_min"] = _col(s, "Ntot_room", min)
    m["farX_min"] = _col(s, "O2far_X", min)
    m["Xlocal_min"] = _col(s, "X_local", min)
    m["hp_end"] = s[-1]["wall_hp"]
    m["end_t"] = s[-1]["t_s"]
    return m


def verdict(ok, value=None):
    """PASS/MISS, or 'n/a ' when the harness didn't write the column."""
    import math
    if value is not None and isinstance(value, float) and math.isnan(value):
        return "n/a "
    return "PASS" if ok else "MISS"


def predicted_T_star(peak_I):
    """The measured equilibrium law, so the scorecard can say what the dials
    PREDICT as well as what the run did (bench report §3.1, +/-1%).
        T* = k_fire_heat * I * 2^(cool_shift - heat_inv_shift)
    heat_inv_shift = log2(furniture thermal_mass = 8) = 3. COOL-SHIFT AXIS: the
    shift is the FURNITURE row's `cool_shift`, not the global any more."""
    c = int(TUNE.get("materials.furniture.cool_shift",
                     TUNE.get("physics.thermal.COOL_SHIFT", 5)))
    return float(TUNE["k_fire_heat"]) * peak_I * (2.0 ** (c - 3))


def scorecard(m):
    if m.get("missing_cols"):
        print("!" * 74)
        print("[harness] this harness build does NOT write: "
              + ", ".join(m["missing_cols"]))
        print("[harness] those scorecard rows will read n/a, AND — more "
              "importantly — this")
        print("[harness] build almost certainly lacks the WARM SEED "
              "(gmap.temperature[crate] = 280).")
        print("[harness] Without it the crate starts at ambient, never "
              "reaches ignition_temp,")
        print("[harness] and every run reads I = 0. See the REQUIRES note in "
              "this file's docstring.")
        print("!" * 74)
    print("=" * 74)
    print(f"{'metric':<26}{'value':>16}   target                    verdict")
    print("-" * 74)
    print(f"{'peak I':<26}{m['peak_I']:>16.3f}   "
          f"{T['peak_lo']}-{T['peak_hi']} (aim {T['peak_aim']})        "
          f"{verdict(T['peak_lo'] <= m['peak_I'] <= T['peak_hi'])}")
    print(f"{'peak time':<26}{m['peak_t']/60:>13.2f} min   "
          f"~{T['peak_t_aim_s']/60:.0f} min ({T['peak_t_lo_s']/60:.0f}-"
          f"{T['peak_t_hi_s']/60:.0f} ok)      "
          f"{verdict(T['peak_t_lo_s'] <= m['peak_t'] <= T['peak_t_hi_s'])}")
    d = m["death_t"]
    print(f"{'fire death':<26}"
          f"{(d/60 if d else float('nan')):>13.2f} min   "
          f"{T['death_lo_s']/60:.0f}-{T['death_hi_s']/60:.0f} min             "
          f"     {verdict(d is not None and T['death_lo_s'] <= d <= T['death_hi_s'])}")
    print(f"{'flame plateau T (game)':<26}{m['flame_T']:>16.0f}   "
          f"{T['flameT_lo']:.0f}-{T['flameT_hi']:.0f}  "
          f"(= {293+2*T['flameT_lo']:.0f}-{293+2*T['flameT_hi']:.0f} K)  "
          f"{verdict(T['flameT_lo'] <= m['flame_T'] <= T['flameT_hi'], m['flame_T'])}")
    print(f"{'  (hot gate at plateau)':<26}{m['hot_at_plateau']:>16.2f}   "
          f"should be ~1.0 while ablaze")
    # what the dials PREDICT at the observed peak I (the §2.5 analytic)
    ts = predicted_T_star(m["peak_I"])
    print(f"{'  T* predicted @ peak I':<26}{ts:>16.0f}   "
          f"= k_fire_heat*I*2^(cool_shift-3)")
    # the cliff: below this intensity the hot gate closes and the fire dies
    if ts > 0:
        i_crit = m["peak_I"] * float(TUNE["fire_T_ext"]) / ts
        print(f"{'  I_crit (self-collapse)':<26}{i_crit:>16.3f}   "
              f"fire cannot live below this I")
    print(f"{'far-field T rise (game)':<26}{m['far_rise']:>16.1f}   "
          f"<= {T['far_rise_max']:.0f}                     "
          f"{verdict(m['far_rise'] <= T['far_rise_max'], m['far_rise'])}")
    print(f"{'room N_total min':<26}{m['ntot_min']:>16.3f}   "
          f">= {T['ntot_min']}                   "
          f"{verdict(m['ntot_min'] >= T['ntot_min'], m['ntot_min'])}")
    print(f"{'far-field X min':<26}{m['farX_min']:>16.4f}   "
          f">= {T['farX_min']} (O2 fix alive)    "
          f"{verdict(m['farX_min'] >= T['farX_min'], m['farX_min'])}")
    print(f"{'local X min (vitiation)':<26}{m['Xlocal_min']:>16.4f}   "
          f"info: should DIP near flame")
    print(f"{'wall_hp at end':<26}{m['hp_end']:>16.2f}   "
          f"info: >0 = charred remains (ok)")
    print("=" * 74)
    print(f"[csv] {CSV_PATH}")


if __name__ == "__main__":
    import os
    out = run_harness()
    tail = [ln for ln in out.splitlines() if ln.strip()][-6:]
    print("\n".join("[harness] " + ln for ln in tail), "\n")
    scorecard(metrics(load_series()))
    # Auto-plot (I / T / room-X vs time). The window blocks until closed —
    # that's the loop rhythm: look, close, edit TUNE, re-run.
    # Disable the window (PNG still written) with:  set FIRE_TUNE_SHOW=0
    show = os.environ.get("FIRE_TUNE_SHOW", "1") != "0"
    manual = (f'    conda run -n data python '
              f'"{ROOT / "tools" / "fire_tune_plot.py"}" "{CSV_PATH}" --show')
    try:
        # tools/ is sys.path[0] when this file is run as a script.
        from fire_tune_plot import make_plot
        make_plot(CSV_PATH, show=show, targets=T)
    except Exception as e:      # matplotlib missing, no display, CSV drift, ...
        # NEVER take the run down with the plot: the scorecard above is the
        # deliverable. Print the exact command to reproduce the plot by hand.
        print(f"[plot] skipped ({type(e).__name__}: {e}). Plot it manually:")
        print(manual)
