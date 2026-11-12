"""tools/obs_body_imprint.py -- the bodies' imprint in the observation's
physics planes (#195 1g (b); obs spec v1 section 3.2 -- an ACCEPTED GAP
since Erik's ruling of 2026-11-12, "on the wind question, its OK! We accept
it!"; the measurement's table is in the spec).

``GameMap.stamp_units`` writes every living body into ``dyn_permeability``,
``dyn_wave_absorb`` and ``dyn_heat_atten_q``, which the gas solver and the
radiation sweep read -- so the eight physics planes of the observation are
not a function of public state alone: a hidden body leaves a wake. This
tool MEASURES that wake (it measured it for Erik's ruling; it stays the
instrument should the gap ever be revisited): the ``test_wind_doorway_e2e`` scene (ship_demo, the zombie room
over-pressured by +0.44 atm venting through its two west doors: a
half-second JET, then a lasting draft), played twice in lock-step -- once
with ONE body (standing in the draft in front of the upper door, or walking
out through the lower door) and once with none -- and per tick, per plane,
the largest absolute difference of the SQUASHED, float16-rounded plane
values (the observation's own numbers, the reference marine's denominators)
inside the 25 x 25 window around the body against the same statistic far
from it (every tile beyond 50 tiles, Chebyshev); near and far windows both.

    C:/Users/steen/anaconda3/envs/data/python.exe tools/obs_body_imprint.py
        [--ticks 240] [--json OUT]

Prints one table per case (standing, walking) and per phase (the jet:
ticks 1..24; the draft: the rest): per plane the max over ticks of the near
statistic, of the far statistic, and their ratio. Measurement only: writes
nothing into the repo.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release",
           ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402

JET_TICKS = 24
FAR_TILES = 50


def _scene(bp, body):
    """The e2e scene with ``body`` ("standing", "walking" or None)."""
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.field_edit import EditMode, FieldEdit, Region
    from simulation.ruleset import ContinuousRealtime
    from simulation.unit import Unit
    from test_wind_doorway_e2e import (LOWER_DOOR, OVERPRESSURE_ATM, ROOM_COLS,
                                       ROOM_ROWS, UPPER_DOOR, WEST_WALL)
    from simulation.materials import MAT_DOOR
    level = load_level("ship_demo", levels_dir=str(ROOT / "levels"))
    sim = Simulation(level, seed=1, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    g = sim.gmap
    sim.edit_queue.enqueue(FieldEdit(
        "atmosphere", Region.RECT,
        (ROOM_ROWS[0], ROOM_COLS[0], ROOM_ROWS[1], ROOM_COLS[1]),
        OVERPRESSURE_ATM, EditMode.ADD))
    sim.set_paused(False)
    for _ in range(30):
        sim.step()
    for (r, c) in np.argwhere(g.material == MAT_DOOR).tolist():
        on_room = (ROOM_ROWS[0] - 1 <= r <= ROOM_ROWS[1] + 1
                   and ROOM_COLS[0] - 1 <= c <= ROOM_COLS[1] + 1)
        if not on_room:
            g.destroy_wall(r, c)
    for _ in range(24):
        sim.step()
    unit = None
    if body == "standing":
        unit = Unit("body", x=37.0, y=20.0, team=0)
        sim.add_unit(unit)
    elif body == "walking":
        unit = Unit("body", x=42.0, y=29.0, team=0)
        sim.add_unit(unit)
    for r in list(UPPER_DOOR) + list(LOWER_DOOR):
        g.destroy_wall(r, WEST_WALL)
    return sim, unit, level


def _planes(sim, ctx, ref_unit):
    """The eight squashed float16 planes, near (tiles) and far (blocks),
    for the reference body's denominators."""
    from simulation import obs_spec_v1 as O
    P = O.FieldPlanes(sim.gmap, tuple(sim._doors), sim.impulse_table)
    near = np.stack([O.squash_array(O.physics_value(n, P.raw[n], ref_unit, ctx))
                     for n in O.PHYSICS_PLANES]).astype(np.float16)
    far = np.stack([O.squash_array(O.physics_value(n, P.far_raw[n], ref_unit,
                                                   ctx))
                    for n in O.PHYSICS_PLANES]).astype(np.float16)
    return near, far


def _stats(diff, cy, cx, half, far_tiles):
    """Per plane: the max |diff| within ``half`` of (cy, cx) and beyond
    ``far_tiles`` (Chebyshev)."""
    h, w = diff.shape[1:]
    yy, xx = np.mgrid[0:h, 0:w]
    cheb = np.maximum(np.abs(yy - cy), np.abs(xx - cx))
    near = (cheb <= half)
    far = (cheb > far_tiles)
    d = np.abs(diff.astype(np.float64))
    n = d[:, near].max(axis=1) if near.any() else np.zeros(d.shape[0])
    f = d[:, far].max(axis=1) if far.any() else np.zeros(d.shape[0])
    return n, f


def measure(bp, body, ticks, out=print):
    from dataclasses import replace
    from simulation import obs_spec_v1 as O
    from simulation.unit import Unit
    sim_a, unit, level = _scene(bp, body)
    sim_b, _, _ = _scene(bp, None)
    ref = Unit("ref", x=1.0, y=1.0, team=0)       # the denominators only
    spec = replace(level.match, teams=tuple(replace(t, controller="idle")
                                             for t in level.match.teams))
    ctx = O.ObsContext.build(sim_a, spec, 4)
    from control_source import quantize_stick_direction
    dx_q, dy_q, _ = quantize_stick_direction(-1.0, 0.0, 0.0)
    rows = []
    for t in range(1, ticks + 1):
        if body == "walking":
            sim_a.set_move_dir(unit.id, dx_q, dy_q, 0)
        sim_a.step()
        sim_b.step()
        na, fa = _planes(sim_a, ctx, ref)
        nb, fb = _planes(sim_b, ctx, ref)
        cy, cx = unit.center_tile_y(), unit.center_tile_x()
        n_near, n_far = _stats(na.astype(np.float64) - nb.astype(np.float64),
                               cy, cx, O.HALF, FAR_TILES)
        f_near, f_far = _stats(fa.astype(np.float64) - fb.astype(np.float64),
                               cy // O.BLOCK, cx // O.BLOCK, O.HALF,
                               FAR_TILES // O.BLOCK)
        rows.append({"tick": t, "cy": int(cy), "cx": int(cx),
                     "near_window": {"near": n_near.tolist(),
                                     "far": n_far.tolist()},
                     "far_window": {"near": f_near.tolist(),
                                    "far": f_far.tolist()}})
    return rows


def _table(rows, window, lo, hi, names, out=print):
    sel = [r for r in rows if lo <= r["tick"] <= hi]
    near = np.array([r[window]["near"] for r in sel])
    far = np.array([r[window]["far"] for r in sel])
    out(f"  {'plane':12s} {'near max':>10s} {'far max':>10s} {'ratio':>8s} "
        f"{'near mean':>10s} {'far mean':>10s}")
    for k, name in enumerate(names):
        nm, fm = float(near[:, k].max()), float(far[:, k].max())
        ratio = (nm / fm) if fm > 0 else (float("inf") if nm > 0 else 1.0)
        out(f"  {name:12s} {nm:10.4f} {fm:10.4f} {ratio:8.2f} "
            f"{float(near[:, k].mean()):10.4f} {float(far[:, k].mean()):10.4f}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="obs_body_imprint")
    ap.add_argument("--ticks", type=int, default=240)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    import breach_physics as bp
    from simulation import obs_spec_v1 as O
    names = list(O.PHYSICS_PLANES)
    result = {}
    for body in ("standing", "walking"):
        rows = measure(bp, body, args.ticks)
        result[body] = rows
        print(f"== {body} body, {args.ticks} ticks (the jet 1..{JET_TICKS}, "
              f"the draft after) ==")
        for window in ("near_window", "far_window"):
            for label, lo, hi in (("jet", 1, JET_TICKS),
                                  ("draft", JET_TICKS + 1, args.ticks)):
                print(f" {window} / {label}: max |squashed f16 diff| within "
                      f"12 {'tiles' if window == 'near_window' else 'blocks'} "
                      f"of the body vs beyond {FAR_TILES} tiles")
                _table(rows, window, lo, hi, names)
    if args.json:
        Path(args.json).write_text(json.dumps(result), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
