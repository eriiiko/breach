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
                                        apply_overrides, restore_overrides,
                                        build_level)
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


def _sealed_box_level(material_id):
    """20x20 fully-sealed hull chamber (SPACE ring -> vacuum outside since
    boundary != "ambient", HULL ring, AIR interior), ONE crate at (9,9).
    Factored out of the original M7 sealed-box scene builder (fire session
    #12, Phase 3c pre-benches, 2026-09-01) so Bench 1's infinite-fuel sealed
    variant reuses EXACTLY the M7-run-2 scenario shape instead of a second
    hand-rolled tilemap. Returns (LevelData, cx, cy)."""
    from level_loader import LevelData
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
    return lvl, cx, cy


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
    from simulation import Simulation, fire_fixed
    from simulation.gases import O2, INERT_N2

    FP_ONE = 1 << 16
    lvl, cx, cy = _sealed_box_level(material_id)
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


def _hot_gate(T, fire_T_ext, fire_T_span):
    if fire_T_span <= 0:
        return 1.0 if T >= fire_T_ext else 0.0
    return min(1.0, max(0.0, (T - fire_T_ext) / fire_T_span))


def _run_infinite_fuel(level, crate_xy, material_id, ignition_temp, max_seconds, tag,
                       checkpoint_every_s=300.0):
    """Bench 1 (fire session #12, Phase 3c pre-benches, 2026-09-01) -- Erik's M7
    follow-up: pin the crate's ``wall_hp`` back to full EVERY TICK (a debug
    field write, before AND after ``sim.step()`` so no tick, in or out, ever
    sees depleted fuel) to isolate the O2/heat death channel from fuel
    exhaustion. This is a MEASUREMENT INSTRUMENT write (allowed per the task
    brief), not a sim-code or config change -- ``avail = F*o2f`` always sees
    F=1. Shared by BOTH bench-1 legs: the sealed chamber (does infinite fuel
    ever hit the o2_frac_ext=0.13 wall?) and the open sky-fed M1 control
    (does it burn forever at a genuine steady state?). Runs a custom per-tick
    loop (not ``run_one``) because ``run_one`` has no per-tick hook to pin a
    field; reuses the harness's OWN gate math (``_hot_gate`` mirrors
    ``tools.fire_timing_harness._hot_gate`` exactly) and its OWN X_local
    definition (Sigma n_o2 / Sigma n_total over the open 4-neighbours) so
    readouts are directly comparable to 3a's numbers."""
    print(f"[B1-{tag}] infinite-fuel burn, cap={max_seconds}s ...")
    import breach_physics as bp
    from simulation import Simulation, fire_fixed
    from simulation.gases import O2, INERT_N2

    FP_ONE = 1 << 16
    cx, cy = crate_xy
    sim = Simulation(level, seed=12345, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    assert int(g.material[cy, cx]) == material_id

    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
    g.fire[cy, cx] = fire_fixed.quantize_scalar(seed_i)
    g.temperature[cy, cx] = fire_fixed.quantize_scalar(ignition_temp)
    hp0_q = int(g.wall_hp[cy, cx])
    hp0 = hp0_q / FP_ONE

    def _open_nbrs():
        h, w = g.fire.shape
        out = []
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < h and 0 <= nx < w and not bool(g.solid[ny, nx]) \
                    and not bool(g.is_vacuum[ny, nx]):
                out.append((ny, nx))
        return out

    nbrs = _open_nbrs()
    fire_t_ext = int(g.fire_T_ext_plane[cy, cx]) / FP_ONE
    fire_t_span = float(getattr(CFG.physics.fire, "fire_T_span", 150.0))
    o2_frac_ext = float(getattr(CFG.physics.fire, "o2_frac_ext", 0.13))
    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    n_max = int(round(max_seconds * tps))

    had_fire = False
    death_tick = None
    rec_t, rec_I, rec_T, rec_X, rec_hot, rec_hp = [], [], [], [], [], []
    # N_total per sample (R1 verification bench, fire session #12, 2026-09-01):
    # the same Sigma(O2+INERT_N2) over the open neighbours used for X_local's
    # denominator, logged as its own raw-count column -- cheap (already
    # computed as tot_loc below), and lets a reader see whether the O2 gate's
    # denominator itself is moving (thermal expansion/contraction) independent
    # of the numerator, without re-deriving it from X_local*something.
    rec_ntotal = []
    checkpoints = []
    next_ckpt = checkpoint_every_s
    for k in range(1, n_max + 1):
        g.wall_hp[cy, cx] = hp0_q          # PIN pre-step: infinite fuel (instrument write)
        sim.set_paused(False)
        sim.step()
        g.wall_hp[cy, cx] = hp0_q          # PIN post-step (so the recorded hp is always hp0)
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
            tot_loc = float("nan")
            X = float("nan")
        hot = _hot_gate(Tg, fire_t_ext, fire_t_span)
        rec_t.append(t); rec_I.append(I); rec_T.append(Tg); rec_X.append(X)
        rec_hot.append(hot); rec_hp.append(hp); rec_ntotal.append(tot_loc)
        if I > 0.05:
            had_fire = True
        if had_fire and I == 0.0 and death_tick is None:
            death_tick = k
        if t >= next_ckpt:
            checkpoints.append(dict(t_s=t, I=I, T_game=Tg, T_kelvin=kelvin(Tg), X_local=X, hot=hot,
                                    N_total=tot_loc))
            next_ckpt += checkpoint_every_s
        if death_tick is not None and t > death_tick * dt + 15.0:
            break

    import csv
    with open(ART / f"b1_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "I", "T_game", "T_kelvin", "X_local", "hot", "wall_hp",
                    "N_total"])
        for i in range(len(rec_t)):
            w.writerow([f"{rec_t[i]:.4f}", f"{rec_I[i]:.6f}", f"{rec_T[i]:.3f}",
                        f"{kelvin(rec_T[i]):.3f}", f"{rec_X[i]:.6f}", f"{rec_hot[i]:.4f}",
                        f"{rec_hp[i]:.6f}", f"{rec_ntotal[i]:.6f}"])

    death_time = death_tick * dt if death_tick is not None else float("nan")
    X_at_death = rec_X[death_tick - 1] if death_tick is not None else float("nan")
    hot_at_death = rec_hot[death_tick - 1] if death_tick is not None else float("nan")
    T_at_death = rec_T[death_tick - 1] if death_tick is not None else float("nan")
    Ntotal_at_death = rec_ntotal[death_tick - 1] if death_tick is not None else float("nan")
    # Plateau window: last quasi-steady stretch before death (or before the cap
    # if it never dies) -- last 20% of the run, mirroring 3a's near-peak-window
    # convention but adapted for a run that may never decay.
    end_i = (death_tick - 1) if death_tick is not None else len(rec_t)
    start_i = max(0, int(end_i * 0.8))
    I_plateau = float(np.median(rec_I[start_i:end_i])) if end_i > start_i else float("nan")
    X_plateau = float(np.nanmedian(rec_X[start_i:end_i])) if end_i > start_i else float("nan")
    T_plateau = float(np.median(rec_T[start_i:end_i])) if end_i > start_i else float("nan")
    Ntotal_plateau = (float(np.nanmedian(rec_ntotal[start_i:end_i]))
                      if end_i > start_i else float("nan"))
    # X trajectory shape: slope over the last checkpoint_every_s window (game
    # seconds) -- near-zero => flat/asymptote, still-negative => linear burn-down.
    tail_n = max(2, int(checkpoint_every_s * tps))
    X_tail = np.asarray(rec_X[-tail_n:], dtype=np.float64)
    t_tail = np.asarray(rec_t[-tail_n:], dtype=np.float64)
    X_tail_slope_per_min = (float(np.polyfit(t_tail, X_tail, 1)[0]) * 60.0
                            if len(t_tail) >= 2 and not np.all(np.isnan(X_tail)) else float("nan")
                            )
    cause = "NEITHER (hit tick cap, never died)"
    if death_tick is not None:
        if X_at_death <= o2_frac_ext + 0.015:
            cause = "O2-gate (X_at_death near o2_frac_ext)"
        else:
            cause = "T-gate (hot dropped with X still well above o2_frac_ext)"

    summ = dict(
        tag=tag, hp0=hp0, had_fire=had_fire, death_time_s=death_time,
        cause_of_death=cause, X_at_death=X_at_death, hot_at_death=hot_at_death,
        T_at_death_game=T_at_death, T_at_death_kelvin=kelvin(T_at_death) if not np.isnan(T_at_death) else float("nan"),
        Ntotal_at_death=Ntotal_at_death,
        o2_frac_ext=o2_frac_ext,
        I_plateau=I_plateau, X_plateau=X_plateau, T_plateau_game=T_plateau,
        T_plateau_kelvin=kelvin(T_plateau) if not np.isnan(T_plateau) else float("nan"),
        Ntotal_plateau=Ntotal_plateau,
        X_final=rec_X[-1] if rec_X else float("nan"),
        X_tail_slope_per_min=X_tail_slope_per_min,
        peak_I=float(max(rec_I)) if rec_I else float("nan"),
        peak_T_game=float(max(rec_T)) if rec_T else float("nan"),
        peak_T_kelvin=kelvin(float(max(rec_T))) if rec_T else float("nan"),
        n_ticks=len(rec_t), hit_cap=(death_tick is None), max_seconds_cap=max_seconds,
        nbrs=len(nbrs), fire_T_ext=fire_t_ext, fire_T_span=fire_t_span,
        checkpoints=checkpoints,
    )
    print(json.dumps(summ, indent=2))
    (ART / f"b1_{tag}_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def run_b1(sealed_max_seconds=4200.0, open_max_seconds=3600.0):
    """Bench 1 -- infinite fuel, sealed (isolate the O2 death channel) +
    infinite fuel, open M1 control (the closest thing to a true plateau this
    engine has). See ``_run_infinite_fuel`` for the pin mechanism."""
    from simulation.materials import MAT_FURNITURE
    print("[B1] sealed leg (M7-run-2 shape, infinite fuel) ...")
    lvl_sealed, cx_s, cy_s = _sealed_box_level(MAT_FURNITURE)
    summ_sealed = _run_infinite_fuel(lvl_sealed, (cx_s, cy_s), MAT_FURNITURE, 280.0,
                                     sealed_max_seconds, "sealed_inf_fuel")

    print("[B1] open M1-control leg (infinite fuel) ...")
    lvl_open = build_level(COMMON["interior_w"], COMMON["interior_h"], COMMON["crate_xy"],
                           COMMON["tile_size_m"], sky_tau_s=60.0, sponge_width=8)
    summ_open = _run_infinite_fuel(lvl_open, COMMON["crate_xy"], MAT_FURNITURE, 280.0,
                                   open_max_seconds, "open_inf_fuel")
    return summ_sealed, summ_open


def _build_cluster_level(interior_w, interior_h, origin_xy, offsets, tile_size_m,
                         sky_tau_s=60.0, sponge_width=8):
    """Same open PLANETSIDE arena as ``tools.fire_timing_harness.build_level``
    (Bench 2, fire session #12 Phase 3c pre-benches, 2026-09-01), except it
    paints a CLUSTER of furniture tiles (``offsets`` from ``origin_xy``)
    instead of one, to measure mutual net-T^4 radiation coupling between
    burning tiles. Mirrors build_level's ring/ambient setup exactly; only the
    crate placement differs. Returns (LevelData, [(cx,cy), ...])."""
    from level_loader import LevelData
    from simulation.ambient import derive_ambient
    from simulation.materials import MAT_FURNITURE
    AIR, SPACE = 0, 9
    h, w = interior_h + 2, interior_w + 2
    tm = np.full((h, w), AIR, dtype=np.int32)
    tm[0, :] = SPACE
    tm[h - 1, :] = SPACE
    tm[:, 0] = SPACE
    tm[:, w - 1] = SPACE
    ox, oy = origin_xy
    cells = []
    for dx, dy in offsets:
        cx, cy = ox + dx, oy + dy
        tm[cy, cx] = MAT_FURNITURE
        cells.append((cx, cy))
    ambient = derive_ambient(sky_tau_s=float(sky_tau_s), sponge_width=int(sponge_width))
    lvl = LevelData(
        name="phase3a_cluster", version="2", path=Path("."),
        tilemap=tm, tile_size_m=float(tile_size_m), diffuse_path=Path("."),
        boundary="ambient", ambient=ambient)
    return lvl, cells


def run_cluster(tag, offsets, max_seconds=1800.0, origin_xy=None):
    """One cluster-coupling variant (Bench 2): ignite every tile in ``offsets``
    simultaneously in the M1 open scenario and track per-tile I/T/hp
    aggregates -- peak/plateau of the hottest tile AND the cluster mean, a
    'burning tiles only' mean intensity, ramp time, burnout/death time, and
    fuel-unburned fraction at death. Custom loop (not ``run_one``, which
    assumes a single crate tile)."""
    origin_xy = origin_xy or COMMON["crate_xy"]
    print(f"[B2-{tag}] cluster offsets={offsets}, cap={max_seconds}s ...")
    import breach_physics as bp
    from simulation import Simulation, fire_fixed

    FP_ONE = 1 << 16
    lvl, cells = _build_cluster_level(COMMON["interior_w"], COMMON["interior_h"], origin_xy,
                                      offsets, COMMON["tile_size_m"])
    sim = Simulation(lvl, seed=12345, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
    for cx, cy in cells:
        g.fire[cy, cx] = fire_fixed.quantize_scalar(seed_i)
        g.temperature[cy, cx] = fire_fixed.quantize_scalar(280.0)
    hp0 = int(g.wall_hp[cells[0][1], cells[0][0]]) / FP_ONE
    n_cells = len(cells)

    tps = float(CFG.clock.ticks_per_second)
    dt = 1.0 / tps
    n_max = int(round(max_seconds * tps))

    had_fire = False
    death_tick = None
    fuel_out_tick = None
    rec_t, rec_Iall, rec_Iburn, rec_Tmax, rec_Tmean, rec_hpfrac = [], [], [], [], [], []
    for k in range(1, n_max + 1):
        sim.set_paused(False)
        sim.step()
        t = k * dt
        Is = [int(g.fire[cy, cx]) / FP_ONE for cx, cy in cells]
        Ts = [int(g.temperature[cy, cx]) / FP_ONE for cx, cy in cells]
        hps = [int(g.wall_hp[cy, cx]) / FP_ONE for cx, cy in cells]
        Iall = sum(Is) / n_cells
        burning = [i for i in Is if i > 0.05]
        Iburn = (sum(burning) / len(burning)) if burning else 0.0
        Tmax = max(Ts)
        Tmean = sum(Ts) / n_cells
        hp_frac = (sum(hps) / (n_cells * hp0)) if hp0 else float("nan")
        rec_t.append(t); rec_Iall.append(Iall); rec_Iburn.append(Iburn)
        rec_Tmax.append(Tmax); rec_Tmean.append(Tmean); rec_hpfrac.append(hp_frac)
        if Iall > 0.05:
            had_fire = True
        if had_fire and all(i == 0.0 for i in Is) and death_tick is None:
            death_tick = k
        if had_fire and all(h <= 0.0 for h in hps) and fuel_out_tick is None:
            fuel_out_tick = k
        done = death_tick if death_tick is not None else fuel_out_tick
        if done is not None and t > done * dt + 15.0:
            break

    import csv
    with open(ART / f"b2_{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "I_all_mean", "I_burning_mean", "T_hottest_game", "T_hottest_kelvin",
                    "T_cluster_mean_game", "T_cluster_mean_kelvin", "hp_frac_mean"])
        for i in range(len(rec_t)):
            w.writerow([f"{rec_t[i]:.4f}", f"{rec_Iall[i]:.6f}", f"{rec_Iburn[i]:.6f}",
                        f"{rec_Tmax[i]:.3f}", f"{kelvin(rec_Tmax[i]):.3f}",
                        f"{rec_Tmean[i]:.3f}", f"{kelvin(rec_Tmean[i]):.3f}",
                        f"{rec_hpfrac[i]:.6f}"])

    rec_Iburn_arr = np.asarray(rec_Iburn)
    peak_Iburn = float(rec_Iburn_arr.max()) if rec_Iburn_arr.size else float("nan")
    peak_idx = int(np.argmax(rec_Iburn_arr)) if rec_Iburn_arr.size else 0
    peak_time = float(rec_t[peak_idx]) if rec_t else float("nan")
    ramp_time = float("nan")
    if peak_Iburn > 0:
        hit = np.nonzero(rec_Iburn_arr >= 0.9 * peak_Iburn)[0]
        if hit.size:
            ramp_time = float(rec_t[hit[0]])
    burnout_time = float(fuel_out_tick * dt) if fuel_out_tick is not None else float("nan")
    death_time = float(death_tick * dt) if death_tick is not None else float("nan")
    end_i = fuel_out_tick if fuel_out_tick is not None else len(rec_t)
    start_i = int(np.searchsorted(np.asarray(rec_t), ramp_time)) if not np.isnan(ramp_time) else 0
    win = slice(start_i, end_i) if end_i > start_i else slice(0, len(rec_t))
    Tmax_plateau = float(np.median(np.asarray(rec_Tmax)[win])) if rec_Tmax else float("nan")
    Tmean_plateau = float(np.median(np.asarray(rec_Tmean)[win])) if rec_Tmean else float("nan")
    fuel_unburned_at_death = (rec_hpfrac[death_tick - 1] if death_tick is not None
                              else (rec_hpfrac[-1] if rec_hpfrac else float("nan")))

    summ = dict(
        tag=tag, n_cells=n_cells, offsets=offsets, hp0=hp0,
        peak_I_burning_mean=peak_Iburn, peak_time_s=peak_time, ramp_time_s=ramp_time,
        peak_T_hottest_game=float(max(rec_Tmax)) if rec_Tmax else float("nan"),
        peak_T_hottest_kelvin=kelvin(float(max(rec_Tmax))) if rec_Tmax else float("nan"),
        T_hottest_plateau_game=Tmax_plateau,
        T_hottest_plateau_kelvin=kelvin(Tmax_plateau) if not np.isnan(Tmax_plateau) else float("nan"),
        T_cluster_mean_plateau_game=Tmean_plateau,
        T_cluster_mean_plateau_kelvin=kelvin(Tmean_plateau) if not np.isnan(Tmean_plateau) else float("nan"),
        burnout_time_s=burnout_time, death_time_s=death_time,
        fuel_unburned_frac_at_death=fuel_unburned_at_death,
        n_ticks=len(rec_t), hit_cap=(death_tick is None and fuel_out_tick is None),
        max_seconds_cap=max_seconds,
    )
    print(json.dumps(summ, indent=2))
    (ART / f"b2_{tag}_summary.json").write_text(json.dumps(summ, indent=2))
    return summ


def run_b2(max_seconds=1800.0):
    """Bench 2 -- cluster coupling: single crate vs 1x2 pair vs 2x2 block, same
    dials, stock, M1 open scenario. Measures how much mutual net-T^4 radiation
    between burning tiles raises the cluster's temperature and extends/shortens
    its life vs the lone crate."""
    rows = []
    rows.append(run_cluster("single", [(0, 0)], max_seconds=max_seconds))
    rows.append(run_cluster("pair_1x2", [(0, 0), (1, 0)], max_seconds=max_seconds))
    rows.append(run_cluster("block_2x2", [(0, 0), (1, 0), (0, 1), (1, 1)], max_seconds=max_seconds))
    (ART / "b2_summary.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--m1", action="store_true")
    ap.add_argument("--m5", action="store_true")
    ap.add_argument("--m7", action="store_true")
    ap.add_argument("--m7-furniture", action="store_true",
                    help="supplementary sealed-box check with MAT_FURNITURE "
                         "(the M1 reference material) instead of station-4's MAT_WOOD")
    ap.add_argument("--b1", action="store_true",
                    help="Phase 3c pre-bench 1: infinite-fuel sealed + open-control burns "
                         "(isolate the O2 death channel, Erik's M7 follow-up)")
    ap.add_argument("--b2", action="store_true",
                    help="Phase 3c pre-bench 2: cluster coupling (1 vs 1x2 vs 2x2 furniture crates)")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--m1-max-seconds", type=float, default=1500.0)
    ap.add_argument("--m5-max-seconds", type=float, default=1200.0)
    ap.add_argument("--m7-max-seconds", type=float, default=1800.0)
    ap.add_argument("--b1-sealed-max-seconds", type=float, default=4200.0)
    ap.add_argument("--b1-open-max-seconds", type=float, default=3600.0)
    ap.add_argument("--b2-max-seconds", type=float, default=1800.0)
    args = ap.parse_args(argv)
    if not (args.m1 or args.m5 or args.m7 or args.m7_furniture or args.b1 or args.b2 or args.all):
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
    if args.b1:
        run_b1(sealed_max_seconds=args.b1_sealed_max_seconds,
              open_max_seconds=args.b1_open_max_seconds)
    if args.b2:
        run_b2(max_seconds=args.b2_max_seconds)


if __name__ == "__main__":
    main()
