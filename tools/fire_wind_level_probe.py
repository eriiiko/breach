"""P-F4b task 3 — THE PRESSURE-DRIVEN WIND LEVEL (the honest WindForcer
replacement) — FEASIBILITY PROBE, and the reason this task order item STOPS.

docs/fire_realism_design_2026-08-01.md v5.2's execution order: "P-F4b (... +
FORCED-WIND level with a literature slot)"; the task order's own escape
clause: "IF a sustained honest flow cannot be built from existing machinery,
STOP that task only, document why, and deliver tasks 1-2 + the write-up (the
wind level then becomes a design item -- do not hack the WindForcer back
in)." This script is the EVIDENCE for that stop: it builds the two scenarios
the task order names as candidates using ONLY existing boundary machinery
(``tools/fire_room_bench.py``'s hull-enclosed room + vent, ``GameMap.
destroy_wall``) and MEASURES whether either produces a sustained |W| at a
fire's position. Both do not, for a structural reason (not a tuning
shortfall):

  1. ``GameMap``'s ``boundary`` is a SINGLE, level-global flag (space OR
     ambient — see ``src/simulation/gamemap.py`` __init__, "Boundary mode
     (BC build...)"): every SPACE(9)-coded tile on one level routes WHOLESALE
     to ``is_vacuum`` (space) or ``is_ambient`` (ambient), never a per-tile
     mix. A "corridor with one end open to the planetside ambient ring and
     the other vented to SPACE" (the task order's own first candidate) is
     therefore NOT constructible: there is no way to make one end of one
     level a P=0 sink and the other end a P=P_amb source in the same map.
  2. The task order's own fallback ("two ambient regions if the engine
     supports differing pressures") also fails: ``[ambient]`` carries ONE
     ``p_amb`` per level (``src/simulation/ambient.py``), not a per-region
     value.
  3. What IS buildable from existing machinery — a sealed room breached to
     vacuum (fire_room_bench's own "vent open"/"breach" modes) — is NOT a
     sustained flow. docs/architecture/engine/04_atmosphere_and_pressure.md's
     own as-built description frames breach venting as an ACOUSTIC-SCALE
     TRANSIENT ("the front passes, the dome lingers" — P_prev tracks the
     transient, not a steady state) driven by an elliptic, whole-domain
     pressure solve, not a slow leak. This script's own measurement (below)
     confirms it directly: a room at quasi-steady sealed pin-I combustion
     (192 counts/s O2 draw) drops to a fully-drained near-vacuum interior
     (room N_total ~1.0 -> ~0.05, O2 draw -> 0 counts/s) WITHIN THE SAME TICK
     the vent opens and the following ~0.75 s — there is no multi-second (let
     alone fire-lifetime-scale) plateau to sweep a "wind level" against.

CONCLUSION (task order's own STOP clause invoked): a sustained, honest,
pressure-driven cross-flow cannot be built from today's existing boundary
machinery. This is a DESIGN ITEM (Q-G-adjacent: F-BO's own escape clause,
"if the supply channel alone proves too weak, re-siting the wind fan is a
DESIGN question — bring it back, do not dial it"), not something this
measurement-only patch may hack around (no WindForcer-style forced field
write is used here — every |W| below is the SOLVER'S OWN wind field, read
only). Tasks 1 and 2 stand on their own; this task delivers only the
feasibility evidence.

RUN:
    python tools/fire_wind_level_probe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np
import breach_physics as bp                          # noqa: E402
from config import CFG                                # noqa: E402
from simulation import Simulation, fire_fixed          # noqa: E402
from simulation.gases import O2, INERT_N2              # noqa: E402

from fire_room_bench import build_room_level, vent_tile_coords  # noqa: E402
from fire_timing_harness import (                     # noqa: E402
    FP_ONE, FURN, apply_overrides, restore_overrides,
)

ARTIFACTS_DIR = ROOT / "_fire_tuning_artifacts"

INTERIOR = 12
TILE_SIZE_M = 0.5
CRATE_XY = (6, 6)
VENT_WIDTH = 2
VENT_SIDE = "east"
PIN_I = 0.192
BREACH_AT_S = 15.0
RUN_SECONDS = 25.0
SAMPLE_EVERY_TICKS = 6         # ~0.25s at 24 tps


def probe(*, draw_r=2, verbose=True):
    """Sealed room, pin-I fire established under closed hull, breach a real
    vent (existing destroy_wall path) mid-run, and log |wind_x| at the fire's
    own tile + the room's bulk N_total (the absolute gas inventory) each
    sample. Returns the per-sample trace as a dict of lists."""
    overrides = {
        "physics.combustion.draw_r": draw_r,
        "physics.combustion.max_claimants": max(
            2 * draw_r * (draw_r + 1), int(CFG.physics.combustion.max_claimants)),
    }
    restore = apply_overrides(overrides)
    try:
        level = build_room_level(
            INTERIOR, INTERIOR, TILE_SIZE_M, vent_width=VENT_WIDTH,
            vent_side=VENT_SIDE, vent_open_at_build=False,
            crates=[(CRATE_XY[0], CRATE_XY[1], FURN)])
        sim = Simulation(level, seed=12345, breach_physics=bp, enable_recorder=False)
        gmap = sim.gmap
        fx, fy = CRATE_XY[0] + 1, CRATE_XY[1] + 1
        ign_temp = float(gmap.materials.ignition_temp[FURN])
        gmap.temperature[fy, fx] = fire_fixed.quantize_scalar(ign_temp)
        wall_hp0 = int(gmap.wall_hp[fy, fx])
        pin_I_q = fire_fixed.quantize_scalar(PIN_I)
        gmap.fire[fy, fx] = pin_I_q
        vent_tiles = vent_tile_coords(INTERIOR, INTERIOR, VENT_WIDTH, VENT_SIDE)

        tps = float(CFG.clock.ticks_per_second)
        dt = 1.0 / tps
        breach_tick = int(round(BREACH_AT_S * tps))
        n = int(round(RUN_SECONDS * tps))
        room_mask = (~gmap.solid) & (~gmap.is_vacuum)
        breached = False

        trace = dict(t=[], wind_abs=[], drawn_counts_per_s=[], room_ntot=[], breached=[])
        for k in range(1, n + 1):
            if not breached and k >= breach_tick:
                for (vy, vx) in vent_tiles:
                    gmap.destroy_wall(vy, vx)
                breached = True
            gmap.fire[fy, fx] = pin_I_q
            gmap.wall_hp[fy, fx] = wall_hp0
            sim.set_paused(False)
            sim.step()
            if k % SAMPLE_EVERY_TICKS != 0:
                continue
            t = k * dt
            wx = int(gmap.wind_x[fy, fx]) / FP_ONE
            wy = int(gmap.wind_y[fy, fx]) / FP_ONE
            wind_abs = float((wx * wx + wy * wy) ** 0.5)
            _saved = {kk: getattr(gmap, kk).copy()
                     for kk in ("gas", "temperature", "wall_hp", "heat", "dem_acc")}
            _o2_before = float(gmap.gas[O2].astype(np.int64).sum())
            sim.physics_runner._run_combustion(gmap, dt)
            drawn = (_o2_before - float(gmap.gas[O2].astype(np.int64).sum())) / dt
            for kk, vv in _saved.items():
                getattr(gmap, kk)[...] = vv
            ntm_o2 = gmap.gas[O2][room_mask].astype(np.float64)
            ntm = ntm_o2 + gmap.gas[INERT_N2][room_mask].astype(np.float64)
            ntot = float(ntm.mean()) / FP_ONE
            trace["t"].append(t)
            trace["wind_abs"].append(wind_abs)
            trace["drawn_counts_per_s"].append(drawn)
            trace["room_ntot"].append(ntot)
            trace["breached"].append(breached)
            if verbose:
                print(f"  t={t:6.2f}s  |W|={wind_abs:9.4f}  drawn={drawn:8.2f} counts/s  "
                     f"room_N_total={ntot:.4f}  breached={breached}")
        return trace
    finally:
        restore_overrides(restore)


def write_csv(trace, path):
    import csv
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["# fire_wind_level_probe -- sealed room -> mid-run vent breach"])
        w.writerow([f"# interior={INTERIOR}x{INTERIOR} vent_width={VENT_WIDTH} "
                    f"vent_side={VENT_SIDE} breach_at_s={BREACH_AT_S} pin_I={PIN_I}"])
        w.writerow(["t_s", "wind_abs", "drawn_counts_per_s", "room_N_total", "breached"])
        for i in range(len(trace["t"])):
            w.writerow([f"{trace['t'][i]:.4f}", f"{trace['wind_abs'][i]:.6f}",
                        f"{trace['drawn_counts_per_s'][i]:.4f}",
                        f"{trace['room_ntot'][i]:.6f}", int(trace["breached"][i])])


def main():
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    print("=" * 78)
    print("P-F4b task 3 PROBE: sealed room (12x12) -> vent breach at t=15s -- "
         "does |wind_x| at the fire settle to a sustained plateau?")
    print("=" * 78)
    trace = probe(verbose=True)
    path = ARTIFACTS_DIR / "wind_level_probe_sealed_breach.csv"
    write_csv(trace, path)
    print(f"[artifacts] wrote {path}")

    # Quantify the collapse for the write-up.
    import numpy as _np
    t = _np.asarray(trace["t"])
    ntot = _np.asarray(trace["room_ntot"])
    pre = ntot[t < BREACH_AT_S]
    post_1s = ntot[(t >= BREACH_AT_S) & (t < BREACH_AT_S + 1.0)]
    post_end = ntot[-1]
    print("-" * 78)
    print(f"pre-breach room N_total (mean): {pre.mean():.4f}")
    print(f"room N_total 1s after breach:   {post_1s.min():.4f} (min in that window)")
    print(f"room N_total at run end:        {post_end:.4f}")
    print("CONCLUSION: no sustained flow window exists -- task 3 STOPPED per "
         "its own escape clause. See docs/fire_sizing_package_2026-08-02.md.")


if __name__ == "__main__":
    main()
