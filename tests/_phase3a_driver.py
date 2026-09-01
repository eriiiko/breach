"""Fire session #12, Phase 3a — thin driver reusing tools/fire_timing_harness.py
and the temperature_scale module. MEASUREMENT ONLY: no sim-code or config.toml
edits; every dial change here is a runtime CFG patch through the harness's own
``apply_overrides``/``run_one`` seam (never a parallel bench).

Produces raw artifacts under tests/_phase3a_artifacts/ (untracked) that
docs/fire_phase3a_measurements_2026-09-01.md is written from.

Run:
    C:/Users/steen/anaconda3/python.exe tests/_phase3a_driver.py [--m1] [--m5] [--m7] [--all]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

from tools.fire_timing_harness import (run_one, write_timeseries_csv,  # noqa: E402
                                        apply_overrides, restore_overrides)
from config import CFG  # noqa: E402
import temperature_scale  # noqa: E402

ART = ROOT / "tests" / "_phase3a_artifacts"
ART.mkdir(parents=True, exist_ok=True)

TS = temperature_scale.load()


def kelvin(t_game: float) -> float:
    return TS.to_kelvin(t_game)


COMMON = dict(interior_w=84, interior_h=40, crate_xy=(12, 21), tile_size_m=0.333)


def _plateau_window(m):
    """Mirror run_one's own steady_T window: [time_to_peak, fuel-out]."""
    rec = m["rec"]
    t_arr = rec["t"]
    end_i = None
    # fuel_out tick isn't directly exposed; recover it from burnout_time.
    if not np.isnan(m["burnout_time"]):
        end_i = int(np.searchsorted(t_arr, m["burnout_time"])) + 1
    else:
        end_i = len(t_arr)
    if np.isnan(m["time_to_peak"]):
        start_i = 0
    else:
        start_i = int(np.searchsorted(t_arr, m["time_to_peak"]))
    if end_i <= start_i:
        start_i, end_i = 0, len(t_arr)
    return start_i, end_i


def _metrics_summary(m):
    start_i, end_i = _plateau_window(m)
    rec = m["rec"]
    I_win = rec["I"][start_i:end_i]
    X_win = rec["x_local"][start_i:end_i]
    I_plateau = float(np.median(I_win)) if I_win.size else float("nan")
    X_plateau = float(np.nanmedian(X_win)) if X_win.size else float("nan")
    hp0 = m["crate_hp0"]
    hp_end = m["hp_end"]
    # cause of death: did I snap to 0 while hp still > 0 (heat/O2-governed) or
    # did hp hit 0 first (fuel-governed)?
    fuel_gone = (not np.isnan(m["burnout_time"]))
    snapped = (not np.isnan(m["snap_time"]))
    if snapped and fuel_gone:
        cause = ("fuel-governed (hp->0 first)" if m["burnout_time"] <= m["snap_time"]
                 else "heat/O2-governed (I->0 with fuel left)")
    elif fuel_gone:
        cause = "fuel-governed (hp->0, fire still nonzero at run end / never snapped)"
    elif snapped:
        cause = "heat/O2-governed (I->0 with fuel left, hp never hit 0)"
    else:
        cause = "NEITHER within run window (hit tick cap)"
    return dict(
        k_grow=m["k_grow"], k_die=m["k_die"],
        peak_I=m["peak_I"], peak_time_s=m["peak_time"],
        time_to_90pct_peak_s=m["time_to_peak"],
        I_plateau=I_plateau, X_plateau=X_plateau,
        T_plateau_game=m["steady_T"], T_plateau_K=kelvin(m["steady_T"]),
        burnout_time_s=m["burnout_time"], snap_time_s=m["snap_time"],
        stalled=bool(m["stalled"]), collapsed=bool(m["collapsed"]),
        hp0=hp0, hp_end=hp_end,
        fuel_unburned_frac=(hp_end / hp0) if hp0 else float("nan"),
        cause_of_death=cause,
        nbrs=m["nbrs"], fire_T_ext=m["fire_T_ext"], fire_T_span=m["fire_T_span"],
        o2far_x_min=m["o2far_x_min"], o2room_x_min=m["o2room_x_min"],
        n_ticks=m["n_ticks"], hit_cap=(m["n_ticks"] * m["dt"] >= COMMON.get("_max_seconds", 1e9) - 1e-6),
    )


def run_m1(max_seconds=1500.0, tail_seconds=30.0):
    print(f"[M1] reference single crate, stock dials, cap={max_seconds}s ...")
    t0 = time.time()
    m = run_one(0.0, max_seconds=max_seconds, tail_seconds=tail_seconds,
                verbose=True, **COMMON)
    print(f"[M1] wall time {time.time()-t0:.1f}s, {m['n_ticks']} ticks")
    write_timeseries_csv(m, ART / "m1_reference.csv")
    summ = _metrics_summary(m)
    summ["hit_cap"] = bool(m["n_ticks"] * m["dt"] >= max_seconds - m["dt"])
    summ["max_seconds_cap"] = max_seconds
    (ART / "m1_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=2))
    return m, summ


def run_m5(k_grow_values=(0.125, 0.25, 0.5, 1.0, 2.0), max_seconds=1200.0, tail_seconds=30.0):
    rows = []
    for kg in k_grow_values:
        print(f"[M5] k_grow={kg} ...")
        t0 = time.time()
        m = run_one(0.0, overrides={"physics.fire.k_grow": kg},
                    max_seconds=max_seconds, tail_seconds=tail_seconds,
                    verbose=False, **COMMON)
        print(f"[M5]   wall time {time.time()-t0:.1f}s, {m['n_ticks']} ticks, "
              f"peak_I={m['peak_I']:.3f} stalled={m['stalled']} "
              f"burnout={m['burnout_time']} snap={m['snap_time']}")
        write_timeseries_csv(m, ART / f"m5_kgrow_{kg}.csv")
        summ = _metrics_summary(m)
        summ["hit_cap"] = bool(m["n_ticks"] * m["dt"] >= max_seconds - m["dt"])
        summ["max_seconds_cap"] = max_seconds
        rows.append(summ)
    (ART / "m5_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return rows


def run_m7(max_seconds=1800.0):
    """Sealed-chamber O2 death (station 4 of levels/fire_tuning): ignite the
    wood crate, run until the fire snaps out from O2 starvation or hp hits 0,
    or the tick cap is hit. Pure measurement: builds the Simulation from the
    already-authored level, writes gmap.fire/gmap.temperature via the same
    allowed debug seed path the harness itself uses (fire_fixed quantize).
    """
    print(f"[M7] sealed chamber O2 death, cap={max_seconds}s ...")
    import breach_physics as bp
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed
    from simulation.gases import O2, INERT_N2
    from simulation.materials import MAT_WOOD

    FP_ONE = 1 << 16
    lvl = load_level("fire_tuning", levels_dir=str(ROOT / "levels"))
    sim = Simulation(lvl, seed=12345, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    cx, cy = 9, 31  # S4_CRATE from tools/make_fire_tuning_level.py
    assert int(g.material[cy, cx]) == MAT_WOOD, (
        f"expected MAT_WOOD at station-4 crate tile, got {int(g.material[cy, cx])}")

    ignition_temp_wood = 300.0
    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.12))
    g.fire[cy, cx] = fire_fixed.quantize_scalar(seed_i)
    g.temperature[cy, cx] = fire_fixed.quantize_scalar(ignition_temp_wood)

    def _open_nbrs():
        h, w = g.fire.shape
        out = []
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not bool(g.solid[ny, nx]) \
                    and not bool(g.is_vacuum[ny, nx]):
                out.append((ny, nx))
        return out

    nbrs = _open_nbrs()
    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    n_max = int(round(max_seconds * tps))
    hp0 = int(g.wall_hp[cy, cx]) / FP_ONE
    hp_mat = hp0  # wall_hp at ignition == full material hp (fresh, unburned crate)

    had_fire = False
    death_tick = None
    fuel_out_tick = None
    rec_t, rec_I, rec_T, rec_X, rec_hp = [], [], [], [], []
    for k in range(1, n_max + 1):
        sim.set_paused(False)
        sim.step()
        t = k * dt
        I = int(g.fire[cy, cx]) / FP_ONE
        Tg = int(g.temperature[cy, cx]) / FP_ONE
        hp = int(g.wall_hp[cy, cx]) / FP_ONE
        if nbrs:
            o2_loc = float(sum(int(g.gas[O2, ny, nx]) for (ny, nx) in nbrs))
            tot_loc = float(sum(int(g.gas[O2, ny, nx]) + int(g.gas[INERT_N2, ny, nx])
                                for (ny, nx) in nbrs))
            X = o2_loc / max(1.0, tot_loc)
        else:
            X = float("nan")
        rec_t.append(t); rec_I.append(I); rec_T.append(Tg); rec_X.append(X); rec_hp.append(hp)
        if I > 0.05:
            had_fire = True
        if had_fire and hp <= 0.0 and fuel_out_tick is None:
            fuel_out_tick = k
        if had_fire and I == 0.0 and death_tick is None:
            death_tick = k
        if death_tick is not None or fuel_out_tick is not None:
            # tail a little then stop
            if t > (death_tick or fuel_out_tick) * dt + 5.0:
                break

    import csv
    with open(ART / "m7_sealed.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "I", "T_game", "T_kelvin", "X_local", "wall_hp"])
        for i in range(len(rec_t)):
            w.writerow([f"{rec_t[i]:.4f}", f"{rec_I[i]:.6f}", f"{rec_T[i]:.3f}",
                        f"{kelvin(rec_T[i]):.3f}", f"{rec_X[i]:.6f}", f"{rec_hp[i]:.6f}"])

    death_time = death_tick * dt if death_tick is not None else float("nan")
    fuel_out_time = fuel_out_tick * dt if fuel_out_tick is not None else float("nan")
    X_at_death = rec_X[death_tick - 1] if death_tick is not None else float("nan")
    hp_at_death = rec_hp[death_tick - 1] if death_tick is not None else float("nan")
    summ = dict(
        hp0=hp0, hp_mat=hp_mat, had_fire=had_fire,
        death_time_s=death_time, fuel_out_time_s=fuel_out_time,
        X_at_death=X_at_death, hp_at_death=hp_at_death,
        fuel_unburned_frac_at_death=(hp_at_death / hp0) if (hp0 and not np.isnan(hp_at_death)) else float("nan"),
        n_ticks=len(rec_t), hit_cap=(death_tick is None and fuel_out_tick is None),
        max_seconds_cap=max_seconds, nbrs=len(nbrs),
        peak_I=float(max(rec_I)) if rec_I else float("nan"),
        peak_T_game=float(max(rec_T)) if rec_T else float("nan"),
        peak_T_kelvin=kelvin(float(max(rec_T))) if rec_T else float("nan"),
    )
    print(json.dumps(summ, indent=2))
    (ART / "m7_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def run_m7_sealed_generic(material_id, material_name, hp0_expected, ignition_temp,
                          max_seconds, tag):
    """Generic sealed-box single-crate burn (a THIN harness variant, per the
    task brief's fallback: "or a harness variant with a sealed box"). Builds a
    fully hull-sealed chamber in memory (SPACE ring -> vacuum outside since
    boundary != 'ambient', HULL ring, AIR interior) -- the SAME shape
    tools/make_fire_tuning_level.py's station-4 scaffold uses, just not tied to
    that one authored level, so the crate MATERIAL is a free parameter. Used
    to separate the sealed-chamber death MODE (O2 starvation vs. thermal
    collapse) from a material confound (station 4 ships MAT_WOOD, which has
    nonzero conductivity=0.15 -- unlike M1's MAT_FURNITURE reference material,
    conductivity=0.0 -- so a wood crate loses heat via conduction into the
    surrounding air on top of cool_shift decay, a materially different loss
    channel than the M1 reference fire)."""
    print(f"[M7-{tag}] sealed chamber ({material_name}), cap={max_seconds}s ...")
    import breach_physics as bp
    from level_loader import LevelData
    from simulation import Simulation, fire_fixed
    from simulation.gases import O2, INERT_N2

    FP_ONE = 1 << 16
    SPACE_CODE = 9
    W, H = 20, 20
    x0, y0, x1, y1 = 3, 3, 16, 16
    cx, cy = 9, 9
    tm = np.full((H, W), 0, dtype=np.int32)  # 0 = MAT_AIR
    tm[0, :] = tm[-1, :] = SPACE_CODE
    tm[:, 0] = tm[:, -1] = SPACE_CODE
    tm[1, 1:W - 1] = tm[H - 2, 1:W - 1] = 1     # 1 = MAT_HULL
    tm[1:H - 1, 1] = tm[1:H - 1, W - 2] = 1
    tm[y0, x0:x1 + 1] = tm[y1, x0:x1 + 1] = 1
    tm[y0:y1 + 1, x0] = tm[y0:y1 + 1, x1] = 1
    tm[cy, cx] = material_id
    lvl = LevelData(name="phase3a_sealed", version="2", path=Path("."),
                    tilemap=tm, tile_size_m=0.333, diffuse_path=Path("."),
                    boundary="space")   # NOT "ambient" -> SPACE ring is vacuum,
                                        # HULL ring seals the interior for real.
    sim = Simulation(lvl, seed=12345, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    assert int(g.material[cy, cx]) == material_id

    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.12))
    g.fire[cy, cx] = fire_fixed.quantize_scalar(seed_i)
    g.temperature[cy, cx] = fire_fixed.quantize_scalar(ignition_temp)

    def _open_nbrs():
        h, w = g.fire.shape
        out = []
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not bool(g.solid[ny, nx]) \
                    and not bool(g.is_vacuum[ny, nx]):
                out.append((ny, nx))
        return out

    nbrs = _open_nbrs()
    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    n_max = int(round(max_seconds * tps))
    hp0 = int(g.wall_hp[cy, cx]) / FP_ONE
    assert abs(hp0 - hp0_expected) < 1e-3, (hp0, hp0_expected)

    had_fire = False
    death_tick = fuel_out_tick = None
    rec_t, rec_I, rec_T, rec_X, rec_hp = [], [], [], [], []
    for k in range(1, n_max + 1):
        sim.set_paused(False)
        sim.step()
        t = k * dt
        I = int(g.fire[cy, cx]) / FP_ONE
        Tg = int(g.temperature[cy, cx]) / FP_ONE
        hp = int(g.wall_hp[cy, cx]) / FP_ONE
        if nbrs:
            o2_loc = float(sum(int(g.gas[O2, ny, nx]) for (ny, nx) in nbrs))
            tot_loc = float(sum(int(g.gas[O2, ny, nx]) + int(g.gas[INERT_N2, ny, nx])
                                for (ny, nx) in nbrs))
            X = o2_loc / max(1.0, tot_loc)
        else:
            X = float("nan")
        rec_t.append(t); rec_I.append(I); rec_T.append(Tg); rec_X.append(X); rec_hp.append(hp)
        if I > 0.05:
            had_fire = True
        if had_fire and hp <= 0.0 and fuel_out_tick is None:
            fuel_out_tick = k
        if had_fire and I == 0.0 and death_tick is None:
            death_tick = k
        if death_tick is not None or fuel_out_tick is not None:
            if t > (death_tick or fuel_out_tick) * dt + 5.0:
                break

    import csv
    with open(ART / f"m7_sealed_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "I", "T_game", "T_kelvin", "X_local", "wall_hp"])
        for i in range(len(rec_t)):
            w.writerow([f"{rec_t[i]:.4f}", f"{rec_I[i]:.6f}", f"{rec_T[i]:.3f}",
                        f"{kelvin(rec_T[i]):.3f}", f"{rec_X[i]:.6f}", f"{rec_hp[i]:.6f}"])

    death_time = death_tick * dt if death_tick is not None else float("nan")
    fuel_out_time = fuel_out_tick * dt if fuel_out_tick is not None else float("nan")
    X_at_death = rec_X[death_tick - 1] if death_tick is not None else float("nan")
    hp_at_death = rec_hp[death_tick - 1] if death_tick is not None else float("nan")
    summ = dict(
        material=material_name, hp0=hp0, had_fire=had_fire,
        death_time_s=death_time, fuel_out_time_s=fuel_out_time,
        X_at_death=X_at_death, hp_at_death=hp_at_death,
        fuel_unburned_frac_at_death=(hp_at_death / hp0) if (hp0 and not np.isnan(hp_at_death)) else float("nan"),
        n_ticks=len(rec_t), hit_cap=(death_tick is None and fuel_out_tick is None),
        max_seconds_cap=max_seconds, nbrs=len(nbrs),
        peak_I=float(max(rec_I)) if rec_I else float("nan"),
        peak_T_game=float(max(rec_T)) if rec_T else float("nan"),
        peak_T_kelvin=kelvin(float(max(rec_T))) if rec_T else float("nan"),
    )
    print(json.dumps(summ, indent=2))
    (ART / f"m7_summary_{tag}.json").write_text(json.dumps(summ, indent=2))
    return summ


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", action="store_true")
    ap.add_argument("--m5", action="store_true")
    ap.add_argument("--m7", action="store_true")
    ap.add_argument("--m7-furniture", action="store_true",
                    help="supplementary sealed-box check with MAT_FURNITURE "
                         "(the M1 reference material) instead of station-4's MAT_WOOD")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--m1-max-seconds", type=float, default=1500.0)
    ap.add_argument("--m5-max-seconds", type=float, default=1200.0)
    ap.add_argument("--m7-max-seconds", type=float, default=1800.0)
    args = ap.parse_args(argv)
    if not (args.m1 or args.m5 or args.m7 or args.m7_furniture or args.all):
        args.all = True
    if args.m1 or args.all:
        run_m1(max_seconds=args.m1_max_seconds)
    if args.m5 or args.all:
        run_m5(max_seconds=args.m5_max_seconds)
    if args.m7 or args.all:
        run_m7(max_seconds=args.m7_max_seconds)
    if args.m7_furniture:
        from simulation.materials import MAT_FURNITURE
        run_m7_sealed_generic(MAT_FURNITURE, "furniture", 30.0, 280.0,
                              args.m7_max_seconds, "furniture")


if __name__ == "__main__":
    main()
