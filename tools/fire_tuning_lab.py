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


# ===========================================================================
# ===============   THE REACH BENCH  (`--reach`; P2b, issue #12)  ===========
# ===========================================================================
# "How far does this fire reach?", measured on the radiation SWEEP's shadow
# planes and judged by the engine's OWN temperature-ignition criterion.
#
# WHAT IT MEASURES. `combat.py::apply_temperature_ignition` lights a flammable
# tile when `temperature >= ignition_temp`. A receiver heated by radiation alone
# settles where what it absorbs equals what it emits, i.e. at `E°⁻¹(Φ)` — the
# inverse of the emissive table applied to the sweep's own fluence plane
# `rad_fluence`. So the reach of a fire is the distance at which `E°⁻¹(Φ)`
# falls through the receiving material's `ignition_temp`. That is exactly the
# convention the P1/P2a isotropy gate already uses ("the E°[280] ignition
# crossing"), read here as a radial profile instead of a radius.
#
# It is an UPPER BOUND on reach, deliberately and by construction: it is the
# radiative equilibrium, with no `cool_shift` ambient decay and no conduction
# competing for the tile. Section 5/10 of report_p2b.md quantifies how much the
# cool_shift channel takes back.
#
# WHY A FREE-FIELD SCENE for the curve. Probes are AIR (`a = 0`): they neither
# absorb nor shadow, so Φ at a probe is the fluence a receiver placed there
# would see before it starts shielding the cells behind it. A real level's
# walls would fold geometry into a number that is meant to be about the law.
# The real level IS measured, as the end-to-end cross-check below, which reads
# `gmap.rad_fluence` after a whole `Simulation.step` and so also proves the
# live P2b binding (`[physics.radiation] rad_scale_derived` -> the engine's
# emissive table) is what the sweep actually ran on.
#
# THE TWO SCALES. `emissive.rad_scale` is settable from Python, so fitted vs
# derived is two bakes, not two builds.
#
# Run:  C:/Users/steen/anaconda3/python.exe tools/fire_tuning_lab.py --reach
# Out:  tests/_fire_lab/reach.png + reach.csv (untracked). Deterministic: the
#       sweep is pure integer and the scene is built, not sampled.

REACH_TAG = "reach"
REACH_GRID = 145            # odd: the source patch is centred; nothing clips
REACH_MAX_D = 56            # probe distance in tiles
REACH_T_SRC = 1263.0        # game — the measured crate plateau (survey 9.3's 1556 K)
# TWO source rows on purpose. `furniture` is the row Erik asked to tune first
# (design row 6). `wood` is the row where the two SCALES actually differ: at the
# fitted scale its Fleck factor is 0.679 at the plateau (undamped only through
# 1068 game, P0b 0.8a), at the derived scale it is 1.000 (undamped over the
# whole table, report_p2b 4). Measuring only furniture would hide that.
REACH_SRC_MATS = ("furniture", "wood")
REACH_RECV_MATS = ("furniture", "wood")   # whose ignition_temp the reach is judged against
REACH_SIZES = (1, 2, 4)     # source patch edge in tiles (1, 2x2, 4x4)
REACH_TRANSPORT = "shear"   # the ruled heat transport (design 2.4)
REACH_N_ORD = 16            # S16
REACH_SIGMA = 5.670374419e-8   # W/m2/K4 — for the physical-irradiance column only

# Applied through the harness's own seam, like DIALS above. Empty by default so
# an untouched panel measures config.toml. The interesting entries are the two
# emission scales and the per-row heat capacities -- which is the whole point of
# P2b: `thermal_mass` is the tuning lever and it now has units (J/K).
REACH_DIALS: dict = {
    # "materials.furniture.density": 555.0,      # thermal_mass is DERIVED (R14)
    # "materials.furniture.specific_heat": 1620.0,
    # "physics.radiation.rad_scale_derived": 2.125632e-08,
}


def _reach_scene(n, size, T_src_game, a_src_q, his_src):
    """A free field of air with an `size x size` patch of the source material at
    its centre, held at `T_src_game`. Returns the int32/bool planes the sweep's
    `.noconvert()` binding requires."""
    T = np.zeros((n, n), dtype=np.int32)
    a = np.zeros((n, n), dtype=np.int32)
    his = np.zeros((n, n), dtype=np.int32)
    ts = np.zeros((n, n), dtype=bool)
    c = n // 2
    lo, hi = c, c + size                      # patch grows +x/+y from the centre
    T[lo:hi, lo:hi] = int(round(T_src_game)) << 16
    a[lo:hi, lo:hi] = a_src_q
    his[lo:hi, lo:hi] = his_src
    ts[lo:hi, lo:hi] = True
    d = a.copy()                              # no stamped bodies: d == a
    return T, a, d, his, ts, c, hi


def _reach_sweep(T, a, d, his, ts, table, k_leak):
    """One sweep on the scene. Returns (rad_fluence, fleck_plane)."""
    n = T.shape[0]
    out = [np.zeros((n, n), dtype=np.int64) for _ in range(4)]
    sweep = bp.RadiationSweep()
    k_q = int(round(k_leak * FP_ONE))
    sweep.run(np.ascontiguousarray(T), np.ascontiguousarray(a),
              np.ascontiguousarray(d), np.ascontiguousarray(his),
              np.ascontiguousarray(ts), table, None,
              int(TS.kelvin_ambient) << 16, k_q,
              getattr(bp.RadiationSweep, REACH_TRANSPORT.upper()),
              REACH_N_ORD, *out, fleck_enabled=True)
    return out[3], np.asarray(sweep.fleck_plane(), dtype=np.int64)


def _baked_table(scale):
    tbl = bp.EmissiveTable()
    tbl.rad_scale = float(scale)
    tbl.kelvin_ambient = float(TS.kelvin_ambient)
    tbl.k_temp_to_kelvin = float(TS.k_temp_to_kelvin)
    tbl.bake()
    return tbl


def _mat(name):
    """(heat_atten, thermal_mass, ignition_temp) for a material row.

    Read from a freshly built MaterialTable, not from the raw CFG row, because
    since R14 `thermal_mass` is DERIVED from `density * specific_heat` and no
    longer exists as a config key (design v2 2026-09-19). The table is rebuilt
    per call so a REACH_DIALS override on any of the three still lands.
    """
    from simulation.materials import MaterialTable, MATERIAL_NAMES
    tbl = MaterialTable.from_config(CFG)
    i = [k for k, v in MATERIAL_NAMES.items() if v == name][0]
    return (float(tbl.heat_atten[i]), int(tbl.thermal_mass[i]),
            float(tbl.ignition_temp[i]))


def reach_run():
    restore = apply_overrides(REACH_DIALS)
    try:
        return _reach_run_inner()
    finally:
        restore_overrides(restore)


def _reach_run_inner():
    scales = {"fitted": float(CFG.physics.fire.rad_scale),
              "derived": float(CFG.physics.radiation.rad_scale_derived)}
    dist = np.arange(1, REACH_MAX_D + 1)
    curves, meta, src_rows = {}, {}, {}

    for mat in REACH_SRC_MATS:
        ha, tm, _ = _mat(mat)
        src_rows[mat] = (ha, tm)
        a_src_q = int(round(ha * FP_ONE))
        his_src = int(tm).bit_length() - 1      # log2(thermal_mass), the engine's own
        for sname, scale in scales.items():
            tbl = _baked_table(scale)
            e0 = int(np.asarray(tbl.table())[0])
            for size in REACH_SIZES:
                T, a, d, his, ts, c, hi = _reach_scene(
                    REACH_GRID, size, REACH_T_SRC, a_src_q, his_src)
                for k_leak in ((0.0, 0.10) if size == 1 else (0.0,)):
                    phi, fpl = _reach_sweep(T, a, d, his, ts, tbl, k_leak)
                    # probe along +x from the patch's far edge, on its first row
                    cols = hi - 1 + dist
                    p = phi[c, cols].astype(np.int64)
                    t_cap = np.array([tbl.e_inv_q(int(v)) for v in p],
                                     dtype=np.int64) / FP_ONE
                    # The PHYSICAL irradiance the scheme is delivering. Because
                    # rad_scale = sigma*A_rad*dt / J_per_count by construction,
                    # counts -> W/m2 is q = sigma * Phi / rad_scale, and the
                    # rad_scale in it CANCELS: the irradiance a receiver sees is
                    # independent of the calibration (only the Fleck factor
                    # carries a scale dependence). Reported so that claim can be
                    # read off the CSV instead of taken on trust.
                    q = REACH_SIGMA * (p - e0) / scale        # W/m2, excess over ambient
                    key = (mat, sname, size, k_leak)
                    curves[key] = dict(phi=p, t_cap=t_cap, q=q)
                    meta[key] = dict(
                        f_src=float(fpl[c, c]) / float(bp.RadiationSweep.F_ONE),
                        e0=e0, phi_src=int(phi[c, c]), scale=scale,
                        a=ha, tm=tm)

    ign = {m: _mat(m)[2] for m in REACH_RECV_MATS}
    return dict(dist=dist, curves=curves, meta=meta, ign=ign, scales=scales,
                src_rows=src_rows)


def reach_crossings(m):
    """For each curve, the last distance at which E°⁻¹(Φ) is still at or above
    each receiver's ignition_temp (0 = never reaches)."""
    out = {}
    for key, cur in m["curves"].items():
        out[key] = {}
        for name, thr in m["ign"].items():
            ok = np.nonzero(cur["t_cap"] >= thr)[0]
            out[key][name] = int(m["dist"][ok[-1]]) if ok.size else 0
    return out


def reach_plot(m, live=None):
    dist = m["dist"]
    cross = reach_crossings(m)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(hspace=0.26, wspace=0.16, top=0.87, bottom=0.07)
    styles = {"fitted": dict(ls="--", lw=2.4, alpha=0.55),
              "derived": dict(ls="-", lw=1.4)}
    colours = {1: "tab:red", 2: "tab:orange", 4: "tab:purple"}

    # (a), (b) — the profile per source material, both scales over each other.
    for ax, mat in zip(axes[0], REACH_SRC_MATS):
        for sname in m["scales"]:
            for size in REACH_SIZES:
                cur = m["curves"][(mat, sname, size, 0.0)]
                ax.plot(dist, cur["t_cap"], color=colours[size], **styles[sname],
                        label=f"{size}x{size}, {sname}")
        cur = m["curves"].get((mat, "derived", 1, 0.10))
        if cur is not None:
            ax.plot(dist, cur["t_cap"], color="tab:blue", ls=":", lw=1.6,
                    label="1 tile, k_leak = 0.10 (dormant)")
        for name, thr in m["ign"].items():
            ax.axhline(thr, color="tab:green", lw=0.8, ls="-.", alpha=0.7)
            ax.text(dist[-1], thr, f"{name} ignites ({thr:.0f}) ", color="tab:green",
                    fontsize=7, va="bottom", ha="right")
        ha, tm = m["src_rows"][mat]
        ff = m["meta"][(mat, "fitted", 1, 0.0)]["f_src"]
        fd = m["meta"][(mat, "derived", 1, 0.0)]["f_src"]
        ax.set_title(f"source = {mat}  (a={ha}, thermal_mass={tm})\n"
                     f"Fleck f at the source: fitted {ff:.4f}   derived {fd:.4f}",
                     fontsize=9)
        ax.set_xlabel("distance from the source edge [tiles]")
        ax.set_ylabel("E$^{-1}$($\\Phi$) — radiative equilibrium T [game = K above 293]")
        ax.set_xlim(0, dist[-1])
        ax.set_ylim(0, None)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper right", fontsize=7, ncol=2)

    # (c) — where the two scales differ at all. Flat 1.0 == they do not.
    # Plotted in ABSOLUTE temperature (game + 293). In game units the same ratio
    # blows up in the tail, where both curves approach 0 game but the physics is
    # approaching the ambient BATH, not zero: the interesting quantity is the
    # absolute-T ratio, whose ceiling is exactly (1/f)^(1/4) (T_abs^4 is linear
    # in the fluence, and the Fleck factor is the only scale-dependent term).
    ax = axes[1][0]
    K = float(TS.kelvin_ambient)
    for mat in REACH_SRC_MATS:
        for size in REACH_SIZES:
            f = m["curves"][(mat, "fitted", size, 0.0)]["t_cap"] + K
            d_ = m["curves"][(mat, "derived", size, 0.0)]["t_cap"] + K
            ax.plot(dist, d_ / f, color=colours[size], lw=1.7,
                    ls="-" if mat == "wood" else "--",
                    label=f"{mat} {size}x{size}")
    ax.axhline(1.0, color="k", lw=0.9, alpha=0.6)
    f_w = m["meta"][("wood", "fitted", 1, 0.0)]["f_src"]
    ax.axhline(f_w ** -0.25, color="tab:brown", lw=0.9, ls=":",
               label=f"$(1/f)^{{1/4}}$ = {f_w ** -0.25:.4f} (wood's ceiling)")
    ax.set_title("derived / fitted, same scene, in ABSOLUTE T\n"
                 "1.0 = the calibration does not move the reach at all", fontsize=9)
    ax.set_xlabel("distance from the source edge [tiles]")
    ax.set_ylabel("$T_{abs}$ ratio (derived / fitted)")
    ax.set_xlim(0, dist[-1])
    ax.grid(alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)

    # (d) — the live level's own shadow plane against the free field.
    ax = axes[1][1]
    fff = m["curves"][("furniture", "derived", 1, 0.0)]
    ax.plot(dist, fff["t_cap"], color="tab:red", lw=2.6, alpha=0.5,
            label="free field, 1 tile furniture, derived")
    if live is not None:
        ax.plot(live["dist"], live["t_cap"], color="k", lw=1.2, marker="o", ms=3,
                label=f"LIVE {LEVEL}: gmap.rad_fluence after Simulation.step")
        eq = int(np.sum(live["t_cap"] ==
                        fff["t_cap"][:len(live["t_cap"])]))
        ax.set_title(f"end-to-end: the engine's own shadow plane\n"
                     f"{eq}/{len(live['t_cap'])} probes EQUAL the free field, "
                     f"integer for integer", fontsize=9)
    for name, thr in m["ign"].items():
        ax.axhline(thr, color="tab:green", lw=0.8, ls="-.", alpha=0.7)
    ax.set_xlabel("distance from the source [tiles]")
    ax.set_ylabel("E$^{-1}$($\\Phi$) [game]")
    ax.set_xlim(0, 31)
    ax.set_ylim(0, None)
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)

    txt = " | ".join(
        f"{mat[:4]} {s[:4]} " + "/".join(str(cross[(mat, s, sz, 0.0)]["furniture"])
                                         for sz in REACH_SIZES)
        for mat in REACH_SRC_MATS for s in m["scales"])
    fig.suptitle(
        f"fire_tuning_lab --reach (P2b, issue #12) — free field {REACH_GRID}², "
        f"{REACH_TRANSPORT} S{REACH_N_ORD}, k_leak = 0, source held at "
        f"{REACH_T_SRC:.0f} game = {TS.to_kelvin(REACH_T_SRC):.0f} K\n"
        f"reach to furniture ignition, tiles, 1x1/2x2/4x4:   {txt}", fontsize=9)
    return fig


def reach_real_level(scale_name="derived"):
    """END-TO-END: the same measurement on a real level, read off the LIVE
    shadow plane `gmap.rad_fluence` after a whole `Simulation.step`.

    This is the half that proves the P2b binding: the number the engine's own
    conductor produced, on the table PhysicsRunner baked from
    `[physics.radiation] rad_scale_derived`, not on a table this bench baked.
    """
    level = load_level(LEVEL)
    sim = Simulation(level, seed=12345, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    sx, sy = 46, 8                                # station 3 furniture sample
    # CLAUDE.md "Starting a fire": deliver HEAT as well as lighting the tile.
    g.temperature[sy, sx] = fire_fixed.quantize_scalar(REACH_T_SRC)
    g.fire[sy, sx] = fire_fixed.quantize_scalar(0.8)
    sim.set_paused(False)
    sim.step()
    eng = sim.physics_runner.engine
    tbl = eng.emissive
    phi = np.asarray(g.rad_fluence)
    ys = np.arange(sy + 1, sy + 1 + 30)            # straight down the open hall
    p = phi[ys, sx].astype(np.int64)
    t_cap = np.array([tbl.e_inv_q(int(v)) for v in p], dtype=np.int64) / FP_ONE
    return dict(dist=ys - sy, phi=p, t_cap=t_cap,
                scale=float(tbl.rad_scale),
                T_src=int(g.temperature[sy, sx]) / FP_ONE,
                e0=int(np.asarray(tbl.table())[0]))


def reach_main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = reach_run()
    cross = reach_crossings(m)
    keys = sorted(m["curves"])
    print(f"\n  source held at {REACH_T_SRC:.0f} game = {TS.to_kelvin(REACH_T_SRC):.0f} K, "
          f"{REACH_TRANSPORT} S{REACH_N_ORD}, free field {REACH_GRID}x{REACH_GRID}")
    print(f"  {'src':<10}{'scale':<9}{'size':<6}{'k_leak':<8}{'rad_scale':<12}"
          f"{'f_src':<10}{'E0':>10}  {'reach (last tile >= ignition)':<34}"
          "q at d=1,2,3 [kW/m2]")
    for key in keys:
        mat, sname, size, k = key
        md, c, cur = m["meta"][key], cross[key], m["curves"][key]
        q = "  ".join(f"{cur['q'][i] / 1e3:7.2f}" for i in (0, 1, 2))
        print(f"  {mat:<10}{sname:<9}{size}x{size:<4}{k:<8.2f}{md['scale']:<12.4e}"
              f"{md['f_src']:<10.6f}{md['e0']:>10}  " +
              "  ".join(f"{n} {v:>2}t" for n, v in c.items()).ljust(34) + q)
    r = reach_real_level()
    ff = m["curves"][("furniture", "derived", 1, 0.0)]
    n_eq = int(np.sum(r["t_cap"] == ff["t_cap"][:len(r["t_cap"])]))
    print(f"\n  LIVE {LEVEL} — gmap.rad_fluence after a whole Simulation.step, on the "
          f"table PhysicsRunner baked ({r['scale']:.4e}):")
    print(f"    the source tile reads {r['T_src']:.1f} game AFTER the tick (the sweep is "
          f"step 2b, before the fold, so it saw {REACH_T_SRC:.0f})")
    for i in (0, 1, 2, 4, 9, 19, 29):
        print(f"    d={r['dist'][i]:>3}  Phi={r['phi'][i]:>14d}  "
              f"E_inv(Phi)={r['t_cap'][i]:8.1f} game   free field "
              f"{ff['t_cap'][i]:8.1f}")
    print(f"    -> {n_eq}/{len(r['t_cap'])} probes EQUAL the free field, integer for integer")

    fig = reach_plot(m, live=r)
    png = OUT / f"{REACH_TAG}.png"
    fig.savefig(png, dpi=130)
    header = ["distance_tiles"]
    cols = [m["dist"].astype(np.float64)]
    for key in keys:
        mat, sname, size, k = key
        stem = f"{mat}_{sname}_{size}x{size}_kleak{k:g}"
        header += [f"{stem}_phi", f"{stem}_t_cap", f"{stem}_q_Wm2"]
        cols += [m["curves"][key]["phi"].astype(np.float64),
                 m["curves"][key]["t_cap"], m["curves"][key]["q"]]
    csv = OUT / f"{REACH_TAG}.csv"
    np.savetxt(csv, np.column_stack(cols), delimiter=",",
               header=",".join(header), comments="", fmt="%.6g")
    print(f"\n  wrote {png}\n  wrote {csv}")
    plt.show()


if __name__ == "__main__":
    if "--reach" in sys.argv:
        reach_main()
    else:
        main()
