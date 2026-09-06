"""tools/fire_tuning_lab.py — Erik's sit-with-it fire tuning lab (session #12).

Edit the TUNING PANEL below, hit run, get a plot. Nothing here changes the
sim or config.toml on disk: every dial is a runtime CFG patch through
tools/fire_timing_harness.py's own apply_overrides/restore_overrides seam
(the canonical bench seam — never a parallel one), applied BEFORE the
Simulation is built and restored afterwards.

Unlike the harness (synthetic room), this loads a REAL shipped level via
level_loader.load — the map matters. Default: levels/fire_tuning, igniting
the station-3 furniture sample (see IGNITE_TILES for the station catalogue
and the wood-bonfire finding). NOTE this level is boundary="space"
(hull-sealed hall, vacuum ring, NO sky-exchange refill): the hall is one
fixed O2 inventory, which is honest ship physics — watch the x_room curve.

Run (always the conda `data` python):
    C:/Users/steen/anaconda3/python.exe tools/fire_tuning_lab.py

Outputs (untracked): tests/_fire_lab/<tag>.png + <tag>.csv, then plt.show().
Deterministic: fixed seed, still air, same panel -> same numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ===========================================================================
# ======================        TUNING PANEL        ========================
# ===========================================================================
# Every entry in DIALS is applied as a CFG override for this run only.
# COMMENT A LINE OUT to fall back to whatever config.toml currently says.
# The values written here are config.toml's values as of 2026-09-06 (the R3
# landing), so an untouched panel reproduces the shipped behaviour.

LEVEL = "fire_tuning"
SIM_SECONDS = 2400.0            # 60-120 for growth-phase tuning; 1800 for the full arc
RUN_TAG = "fire_lab"           # output filename stem (tests/_fire_lab/<tag>.png/.csv)

# What burns ((x, y) tiles; the FIRST tile is the PROBE the plots follow).
# fire_tuning station catalogue (tools/make_fire_tuning_level.py):
#   station 1 bonfire (2x2 WOOD):      [(8, 8), (9, 8), (8, 9), (9, 9)]
#   station 3 wood sample:             [(40, 8)]
#   station 3 furniture sample:        [(46, 8)]
#   station 3 kindling sample:         [(52, 8)]
#   station 4 sealed-chamber crate:    [(9, 31)]
#   station 5 door-room crate:         [(27, 31)]
# DEFAULT = the furniture sample: it reproduces the m1 bench (peak I 0.850
# @ 13.1 s vs the bench's 0.849 @ 13.1 s — cross-validated 2026-09-06).
# FINDING (2026-09-06): the WOOD bonfire cannot sustain fire under any of
# the current dials — wood's conductivity (0.15, vs furniture's 0.0 = "no
# conduction face") drains the ignition heat into the surroundings with a
# ~3 s e-fold, `hot` hits 0 by t≈5 s and the fire starves cold. A real
# material/map finding, not a lab artifact; needs its own ruling.
IGNITE_TILES = [(46, 8)]

# Extra heat above the material's ignition point at seed time (game units).
# 0.0 = the engine-faithful bootstrap: the exact state a tile has the tick
# its temperature crosses ignition_temp (T = ignition, I = ignition_seed).
# Raise it to ask "what if ignition delivered more of a heat punch?" — e.g.
# 100-200 to ride out the young fire's cold-start sag.
IGNITE_T_MARGIN = 55.0

DIALS = {
    # --- intensity ODE (the I ramp: TEMPO / SIZE / death wall) -------------
    "physics.fire.k_grow":            0.382,      # TEMPO — logistic growth gain (1/s)
    "physics.fire.k_die":             0.007,    # death wall when starved/cold (1/s)
    "physics.fire.I_cap_per_avail":   0.95,     # SIZE — I_cap = c * avail * hot

    # --- hot-burns-faster, R3 (the hotf ramp + the two rate dials) ---------
    "physics.fire.hotf_cap":          10.0,     # ceiling on the uncapped hotf ramp
    # NOTE these two rates re-size WITH delta (neutral-at-ignition anchor):
    #   burn_rate = 0.02 * span/delta      wall_damage = 0.03 * span/delta
    #   delta 200 -> 0.018 / 0.027    delta 150 -> 0.024 / 0.036
    #   delta 120 -> 0.030 / 0.045    delta 100 -> 0.036 / 0.054
    # If you move delta, move BOTH of these too, or young fires get a
    # silently biased O2 draw (the exact R3 anchor bug in miniature).
    "physics.fire.wall_damage":       0.045,    # fuel/hp drain rate  (re-anchored, delta 120)
    "physics.combustion.burn_rate":   0.030,    # O2 demand rate      (re-anchored, delta 120)

    # --- knee geometry (shared by hot AND hotf; per-material foot) ---------
    "physics.fire.ignition_to_ext_delta": 120.0,  # fire_T_ext[mat] = ignition[mat] - Δ — THE survival-edge dial (smaller Δ = death line closer to ignition = grazing ignitions die); co-move burn_rate/wall_damage above!
    "physics.fire.fire_T_span":           180.0,  # ramp width above the foot

    # --- oxygen gates (R1-renormalized sustain law) -------------------------
    "physics.fire.o2_frac_ext":       0.13,     # X_ext — flame extinction mole fraction
    "physics.fire.o2_frac_amb":       0.21,     # X_amb — o2f == 1 at ordinary air
    "physics.fire.o2f_cap":           5.0,      # enrichment-flare ceiling

    # --- heat deposit (H_bed — the Phase-4 temperature lever) --------------
    "physics.combustion.H_BED_M":     18125.0,  # H_bed = H_BED_M * 2^H_BED_SHIFT
    "physics.combustion.H_BED_SHIFT": 4,
    "physics.combustion.o2_potency":  1.0,      # multiplier on H_fuel + H_bed
    "physics.combustion.fuel_per_o2": 0.7,      # fuel drained per O2 drawn

    # --- material rows (baked at level load via MaterialTable.from_config —
    #     overrides here ARE applied before the build, so they land) --------
    # "materials.wood.cool_shift":      13,     # e-fold 2^cs/24 s
    # "materials.wood.conductivity":    0.15,   # 0.0 = no conduction face (furniture)
    # "materials.furniture.cool_shift": 13,
}
# ===========================================================================
# ==================     end of panel — machinery below     ================
# ===========================================================================

import numpy as np                                          # noqa: E402
import matplotlib
import matplotlib.pyplot as plt                             # noqa: E402

from fire_timing_harness import (apply_overrides, restore_overrides,  # noqa: E402
                                 _open_neighbors, _hot_gate)
import breach_physics as bp                                 # noqa: E402
from config import CFG                                      # noqa: E402
from level_loader import load as load_level                 # noqa: E402
from simulation import Simulation, fire_fixed               # noqa: E402
from simulation.gases import O2, INERT_N2                   # noqa: E402
import temperature_scale                                    # noqa: E402

FP_ONE = 1 << 16
OUT = ROOT / "tests" / "_fire_lab"
TS = temperature_scale.load()


def _clamp(v, lo, hi):
    return min(hi, max(lo, v))


def run(sim_seconds=None):
    restore = apply_overrides(DIALS)
    try:
        return _run_inner(sim_seconds if sim_seconds is not None else SIM_SECONDS)
    finally:
        restore_overrides(restore)


def _run_inner(sim_seconds):
    level = load_level(LEVEL)
    sim = Simulation(level, seed=12345, breach_physics=bp, enable_recorder=False)
    gmap = sim.gmap

    px, py = IGNITE_TILES[0]                      # the probe tile
    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    n_ticks = int(round(sim_seconds * tps))

    # Panel dials the lab mirrors lab-side (read back AFTER the override so
    # the mirror always matches what the solver was actually given).
    span = float(CFG.physics.fire.fire_T_span)
    delta = float(CFG.physics.fire.ignition_to_ext_delta)
    hotf_cap = float(CFG.physics.fire.hotf_cap)
    x_ext = float(CFG.physics.fire.o2_frac_ext)
    x_amb = float(CFG.physics.fire.o2_frac_amb)
    o2f_cap = float(CFG.physics.fire.o2f_cap)
    c_cap = float(CFG.physics.fire.I_cap_per_avail)
    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.12))

    # Game-faithful ignition (harness precedent): a tile ignites BECAUSE its T
    # crossed ignition_temp, so seed each tile at its OWN material's ignition
    # point (= fire_T_ext_plane + Δ — the plane is the per-material foot the
    # solver itself subtracts) with fire = ignition_seed. Solid tiles: a
    # direct temperature write is the sanctioned bench bootstrap (the gas
    # mirror rule concerns GAS cells; these are wood).
    for (ix, iy) in IGNITE_TILES:
        t_ext_i = int(gmap.fire_T_ext_plane[iy, ix]) / FP_ONE
        gmap.temperature[iy, ix] = fire_fixed.quantize_scalar(
            t_ext_i + delta + IGNITE_T_MARGIN)
        gmap.fire[iy, ix] = fire_fixed.quantize_scalar(seed_i)

    t_ext_probe = int(gmap.fire_T_ext_plane[py, px]) / FP_ONE
    ign_probe = t_ext_probe + delta
    hp0 = int(gmap.wall_hp[py, px]) / FP_ONE
    nbrs = _open_neighbors(gmap, py, px)
    room = (~gmap.solid) & (~gmap.is_vacuum) & (~gmap.is_ambient)

    cols = ("t", "I", "I_max", "T", "hp", "F", "x_local", "x_room",
            "o2f", "hot", "hotf", "avail", "I_cap")
    rec = {k: [] for k in cols}

    for k in range(1, n_ticks + 1):
        sim.set_paused(False)
        sim.step()
        t = k * dt
        I = int(gmap.fire[py, px]) / FP_ONE
        I_max = float(max(int(gmap.fire[iy, ix]) for (ix, iy) in IGNITE_TILES)) / FP_ONE
        T = int(gmap.temperature[py, px]) / FP_ONE
        hp = int(gmap.wall_hp[py, px]) / FP_ONE
        F = hp / hp0 if hp0 > 0 else 0.0
        # X over the probe's open 4-neighbours — fraction of SUMS, mirroring
        # fire_simulation.cpp's own read (harness x_local convention).
        if nbrs:
            o2_loc = float(sum(int(gmap.gas[O2, ny, nx]) for (ny, nx) in nbrs))
            tot_loc = float(sum(int(gmap.gas[O2, ny, nx]) + int(gmap.gas[INERT_N2, ny, nx])
                                for (ny, nx) in nbrs))
            x_local = o2_loc / max(1.0, tot_loc)
        else:
            x_local = float("nan")
        o2m = gmap.gas[O2][room].astype(np.float64)
        ntm = o2m + gmap.gas[INERT_N2][room].astype(np.float64)
        x_room = float(np.divide(o2m, ntm, out=np.zeros_like(o2m),
                                 where=ntm > 0).mean())
        hot = _hot_gate(T, t_ext_probe, span)
        hotf = _clamp((T - t_ext_probe) / span, 0.0, hotf_cap) if span > 0 else 0.0
        o2f = _clamp((x_local - x_ext) / max(1e-9, x_amb - x_ext), 0.0, o2f_cap)
        avail = F * o2f
        i_cap = c_cap * avail * hot
        for key, val in zip(cols, (t, I, I_max, T, hp, F, x_local, x_room,
                                   o2f, hot, hotf, avail, i_cap)):
            rec[key].append(val)

    for key in rec:
        rec[key] = np.asarray(rec[key], dtype=np.float64)
    return dict(rec=rec, t_ext=t_ext_probe, ign=ign_probe, hp0=hp0,
                hotf_cap=hotf_cap, x_ext=x_ext, x_amb=x_amb, dt=dt)


# ---------------------------------------------------------------------------
# Plot + CSV
# ---------------------------------------------------------------------------
def _summary(m):
    rec = m["rec"]
    I, t, T = rec["I_max"], rec["t"], rec["T"]
    peak_I = float(I.max())
    peak_t = float(t[int(np.argmax(I))])
    hit = np.nonzero(I >= 0.9 * peak_I)[0]
    t90 = float(t[hit[0]]) if hit.size else float("nan")
    half = T[len(T) // 2:]
    # Death detection + CAUSE (so a heat-collapse is never misread as a
    # fuel burnout): fire is dead when I has dropped to ~0 after burning.
    on = np.nonzero(I > 0.05)[0]
    if on.size and on[-1] < len(t) - 1:
        i_d = int(on[-1])
        F_d, T_d = float(rec["F"][i_d]), float(T[i_d])
        if F_d <= 0.02:
            cause = "FUEL burnout (hp -> 0)"
        elif T_d <= m["t_ext"] + 5:
            cause = f"HEAT-COLLAPSE (T fell through T_ext={m['t_ext']:.0f}, fuel left {F_d*100:.0f}%)"
        else:
            cause = f"O2/other (F={F_d:.2f}, T={T_d:.0f} at death)"
        death = f"DIED at {t[i_d]:.0f} s -- {cause}"
    else:
        death = f"alive at end of run ({t[-1]:.0f} s)"
    lines = [
        death,
        f"peak I (cluster max) = {peak_I:.3f} @ {peak_t:.1f} s   "
        f"(90% of peak at {t90:.1f} s)",
        f"probe T end = {T[-1]:.1f} game = {TS.to_kelvin(T[-1]):.0f} K; "
        f"median over last half = {np.median(half):.1f} game "
        f"= {TS.to_kelvin(float(np.median(half))):.0f} K",
        f"fuel left (probe) = {rec['F'][-1] * 100:.1f}%   "
        f"room X mean end = {rec['x_room'][-1]:.4f}",
    ]
    return lines


def plot(m):
    rec = m["rec"]
    t = rec["t"]
    minutes = t[-1] > 300.0
    x = t / 60.0 if minutes else t
    xlabel = "time [min]" if minutes else "time [s]"

    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(11, 12))
    fig.subplots_adjust(hspace=0.12, top=0.91, right=0.86)

    # 1 — intensity
    ax = axes[0]
    ax.plot(x, rec["I"], color="tab:red", lw=1.5, label="I (probe)")
    ax.plot(x, rec["I_max"], color="tab:red", lw=1.0, ls="--", alpha=0.6,
            label="I (cluster max)")
    ax.plot(x, rec["I_cap"], color="tab:gray", lw=1.0, ls=":",
            label="I_cap = c·avail·hot")
    ax.set_ylabel("intensity")
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.25)

    # 2 — temperature (Kelvin left, Celsius right; same curve)
    ax = axes[1]
    K = np.array([TS.to_kelvin(v) for v in rec["T"]])
    ax.plot(x, K, color="tab:orange", lw=1.5, label="T probe")
    for T_ref, name, c in ((m["ign"], "ignition", "tab:green"),
                           (m["t_ext"], "T_ext", "tab:blue")):
        ax.axhline(TS.to_kelvin(T_ref), color=c, lw=0.8, ls="--", alpha=0.7)
        ax.text(x[-1], TS.to_kelvin(T_ref), f" {name}", color=c, fontsize=8,
                va="bottom", ha="right")
    ax.set_ylabel("T [K]")
    sec = ax.secondary_yaxis("right", functions=(lambda k: k - 273.15,
                                                 lambda c: c + 273.15))
    sec.set_ylabel("T [°C]")
    ax.grid(alpha=0.25)

    # 3 — the gates (0..~1 left; hotf on its own right axis, 0..cap)
    ax = axes[2]
    ax.plot(x, rec["hot"], color="tab:blue", lw=1.2, label="hot (sustain gate)")
    ax.plot(x, rec["o2f"], color="tab:cyan", lw=1.2, label="o2f")
    ax.plot(x, rec["F"], color="tab:brown", lw=1.2, label="F (fuel fraction)")
    ax.plot(x, rec["avail"], color="tab:gray", lw=1.0, ls=":", label="avail = F·o2f")
    ax.set_ylabel("gates [0..1]")
    ax.set_ylim(bottom=0)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.25)
    axr = ax.twinx()
    axr.plot(x, rec["hotf"], color="tab:purple", lw=1.4, ls="--", label="hotf")
    axr.set_ylabel("hotf [0..cap]", color="tab:purple")
    axr.tick_params(axis="y", labelcolor="tab:purple")
    axr.set_ylim(0, m["hotf_cap"] * 1.05)
    axr.legend(loc="upper right", fontsize=8)

    # 4 — oxygen
    ax = axes[3]
    ax.plot(x, rec["x_local"], color="tab:green", lw=1.5, label="X local (flame)")
    ax.plot(x, rec["x_room"], color="tab:olive", lw=1.2, label="X room mean")
    ax.axhline(m["x_ext"], color="tab:red", lw=0.8, ls="--", alpha=0.7)
    ax.axhline(m["x_amb"], color="tab:gray", lw=0.8, ls="--", alpha=0.7)
    ax.set_ylabel("O2 mole fraction")
    ax.set_xlabel(xlabel)
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.25)

    items = [f"{k.split('.')[-1]}={v}" for k, v in DIALS.items()]
    dial_txt = "\n".join("   ".join(items[i:i + 5])
                         for i in range(0, len(items), 5))
    margin = f", ignite +{IGNITE_T_MARGIN:g}" if IGNITE_T_MARGIN else ""
    fig.suptitle(f"fire_tuning_lab — level={LEVEL}, probe {IGNITE_TILES[0]}"
                 f"{margin}, {t[-1]:.0f} s\n{dial_txt}", fontsize=8)
    return fig


def write_csv(m, path):
    rec = m["rec"]
    keys = list(rec.keys())
    arr = np.column_stack([rec[k] for k in keys])
    header = ",".join(keys)
    np.savetxt(path, arr, delimiter=",", header=header, comments="",
               fmt="%.6g")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = run()
    for line in _summary(m):
        print("  " + line)
    fig = plot(m)
    png = OUT / f"{RUN_TAG}.png"
    csv = OUT / f"{RUN_TAG}.csv"
    fig.savefig(png, dpi=130)
    write_csv(m, csv)
    print(f"  wrote {png}")
    print(f"  wrote {csv}")
    plt.show()


if __name__ == "__main__":
    main()
