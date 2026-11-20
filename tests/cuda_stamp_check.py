"""#29 P4b — THE STAMP KERNEL GATE: the five stamp planes, device vs the CPU
reference, tol 0, on every tick of four scenes.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_stamp.py). The CPU stamp (``GameMap.stamp_units`` ->
``PhysicsEngine.stamp_units``) is THE oracle; the device twin is
``stamp_launch_resident`` (cpp/src/cuda_stamp.cu), driven through
``PhysicsEngine.stamp_units_resident`` from the rows the host stamp built
(``GameMap.take_stamp_rows``). Per tick the script:

  1. moves the scene's units and runs the host stamp (the oracle's planes);
  2. takes the stamp rows (the device's ONLY unit input);
  3. POISONS the five device planes -- a plane the kernel did not write stays
     poisoned and cannot equal the host's, so a kernel that quietly does
     nothing (or a stale buffer that happens to equal) cannot pass;
  4. runs the device stamp and compares obstacles, dyn_permeability,
     dyn_wave_absorb, dyn_heat_atten_q and dyn_light_atten_q to the host's,
     tol 0 (a bool / float / int array_equal).

Scenes (each unit set DIFFERS in size, permeability, wave_absorb, heat_atten
and light_atten, and the units overlap one another AND walls, die and are born,
and cross the map edge -- the cases where the combine direction and the host's
row order matter):

  empty   no units at all (the reset alone, rows = 0);
  crowd   eleven units of footprint 1 / 3 / 5, overlapping pairs whose
          permeabilities differ (the host WRITES `min(up, sp)` per row, so the
          LAST row's value stands where bodies overlap -- not a running min),
          moving every tick over a wood segment and the hull, one dying at
          tick 5 and one born at tick 8, one partly off the map;
  edits   the crowd plus a wall destroyed mid-run (the static planes move; the
          gate re-uploads them as the tick's entry upload does) so the
          baselines and the obstacle diff change under the units;
  dense   many overlapping unit rows on a few tiles (the atomic path's worst
          case: hundreds of rows contending for one cell).

PROPERTY: the device's five planes equal the CPU stamp's, byte for byte, on
every tick, from the host's rows alone. BREAKS IF the kernel's combine
direction flips (a MIN where the host MAXes), its row order (the last row must
win the permeability write), a reset is skipped, a footprint is off by one, or
a channel of the light triple is dropped. Validated by breaking it (see the
run log of the patch).

Prints ``STAMP_RESULT: PASS`` / ``FAIL`` and exits 0 / 1.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PLANES = ("obstacles", "dyn_permeability", "dyn_wave_absorb",
          "dyn_heat_atten_q", "dyn_light_atten_q")
STATIC = ("permeability", "wave_absorb", "heat_atten_q", "light_atten_q")
_POISON = {"obstacles": True, "dyn_permeability": 7.0, "dyn_wave_absorb": 7.0,
           "dyn_heat_atten_q": -12345, "dyn_light_atten_q": -12345}

H, W = 48, 64


def _world():
    from level_loader import LevelData
    from simulation.gamemap import GameMap
    from simulation.physics_runner import PhysicsRunner

    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:H - 2, 2:W - 2] = 1                 # hull ring
    tm[3:H - 3, 3:W - 3] = 4                 # interior air
    tm[20:30, 30] = 2                        # a wood wall segment
    tm[10, 10:24] = 2                        # another, horizontal
    level = LevelData(name="stamp_gate", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
    g = GameMap(level)
    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    g.stamp_units([])
    g.enable_residency()
    return g, runner


def _unit(name, x, y, fp, perm=None, wabs=None, heat=None, light=None):
    from simulation.unit import Unit
    u = Unit(name, x, y, team=0, footprint=fp)
    if perm is not None:
        u.permeability = perm
    if wabs is not None:
        u.wave_absorb = wabs
    if heat is not None:
        u.heat_atten = heat
    if light is not None:
        u.light_atten = light
    return u


def _crowd():
    return [
        _unit("a", 12.0, 12.0, 3, perm=0.2, wabs=0.9, heat=0.4, light=(0.9, 0.2, 0.5)),
        _unit("b", 13.0, 12.0, 3, perm=0.9, wabs=0.1, heat=0.8, light=(0.1, 0.8, 0.3)),
        _unit("c", 14.0, 13.0, 1, perm=0.6, wabs=0.3, heat=0.2, light=(0.5, 0.5, 0.9)),
        _unit("d", 29.0, 24.0, 5, perm=0.3),                   # over the wood
        _unit("e", 30.0, 25.0, 3, wabs=0.99),                  # overlaps d and wood
        _unit("f", 40.0, 30.0, 3),                              # all defaults
        _unit("g", 41.0, 30.0, 3, perm=0.05, heat=0.05, light=(0.0, 0.0, 0.0)),
        _unit("h", 3.0, 3.0, 5, perm=0.7),                      # hull corner, off-grid tiles
        _unit("i", 60.0, 44.0, 3, wabs=0.4),                    # against the far hull
        _unit("j", 20.0, 40.0, 1),
        _unit("k", 20.0, 40.0, 1, perm=0.15, heat=0.95),        # exactly on j
    ]


def _move(units, t, scene):
    """Deterministic motion: every unit shifts each tick (tiles freed and
    refilled), some bounce off the grid edge; one dies, one is born."""
    for i, u in enumerate(units):
        dx = ((i * 7 + t * 3) % 5) - 2
        dy = ((i * 5 + t * 2) % 5) - 2
        u.set_pos(float(min(W + 1, max(-1, u.x + dx * 0.9))),
                  float(min(H + 1, max(-1, u.y + dy * 0.9))))
    if scene in ("crowd", "edits"):
        if t == 5:
            units[2].alive = False
        if t == 8:
            units[2].alive = True
            units[2].set_pos(15.0, 14.0)


def _dense():
    # 300 unit-rows contending for a 4 x 4 patch: every cell is touched by a
    # long run of rows whose permeability, absorb, heat and light differ.
    units = []
    for i in range(34):
        units.append(_unit(f"d{i}", 22.0 + (i % 3), 22.0 + (i % 2), 3,
                           perm=(i * 37 % 100) / 100.0,
                           wabs=(i * 53 % 100) / 100.0,
                           heat=(i * 29 % 100) / 100.0,
                           light=((i * 11 % 100) / 100.0, (i * 17 % 100) / 100.0,
                                  (i * 23 % 100) / 100.0)))
    return units


def _compare(g, tag, t):
    bad = 0
    for name in PLANES:
        dev = g._dev[name].get()
        host = getattr(g, name)
        if dev.shape != host.shape or not np.array_equal(dev, host):
            bad += 1
            n = int(np.count_nonzero(dev != host)) if dev.shape == host.shape else -1
            print(f"  FAIL {tag} tick {t}: device {name} != the CPU stamp ({n} cells)")
    return bad


def _run_scene(tag, units_fn, n_ticks, scene):
    import cupy as cp
    g, runner = _world()
    units = units_fn()
    eng = runner.engine
    bad = 0
    rows_seen = 0
    c0 = dict(bp.resident_stage_counters())["stamp"]
    for t in range(n_ticks):
        if scene == "edits" and t == 6:
            g.destroy_wall(25, 30)               # a wood tile: baselines move
        if scene == "edits" and t == 9:
            g.destroy_wall(10, 15)
        _move(units, t, scene)
        g.stamp_units(units)                      # the oracle
        rows = g.take_stamp_rows()
        if rows is None:
            print(f"  FAIL {tag} tick {t}: the host stamp left no rows")
            bad += 1
            continue
        rows_seen += int(rows.shape[0])
        g.from_host(list(STATIC))                 # the entry upload's static set
        for name in PLANES:
            g._dev[name].fill(_POISON[name])
        eng.stamp_units_resident(g.device_ptrs(), H, W, rows)
        cp.cuda.Stream.null.synchronize()
        bad += _compare(g, tag, t)
        if bad >= 6:
            print(f"  {tag}: aborting after {bad} divergences")
            break
    ran = dict(bp.resident_stage_counters())["stamp"] - c0
    ok = (bad == 0) and ran >= n_ticks
    if ran < n_ticks:
        print(f"  FAIL {tag}: the device stamp ran {ran}/{n_ticks} ticks")
    print(f"  {tag}: {n_ticks} ticks, {rows_seen} footprint rows, device stamp ran "
          f"{ran}x, five planes {'tol 0' if bad == 0 else f'{bad} DIVERGENCES'}")
    return ok, rows_seen


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("STAMP_RESULT: FAIL (no CUDA build / device)")
        return 1
    sys.path.insert(0, os.path.join(_ROOT, "tools"))
    ok = True
    scenes = (("empty", lambda: [], 3, "empty"),
              ("crowd", _crowd, 14, "crowd"),
              ("edits", _crowd, 14, "edits"),
              ("dense", _dense, 4, "dense"))
    rows_by = {}
    for tag, fn, n, scene in scenes:
        s_ok, rows = _run_scene(tag, fn, n, scene)
        ok = ok and s_ok
        rows_by[tag] = rows
    # The scenes must actually exercise what they claim (a vacuous scene is a
    # silent hole): the empty scene has no rows, the crowd many, the dense
    # scene hundreds of rows over a few cells.
    if rows_by["empty"] != 0:
        ok = False
        print("  FAIL: the empty scene produced stamp rows")
    if rows_by["crowd"] < 100 or rows_by["dense"] < 300:
        ok = False
        print(f"  FAIL: the scenes were too thin ({rows_by})")
    print("STAMP_RESULT: PASS" if ok else "STAMP_RESULT: FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
