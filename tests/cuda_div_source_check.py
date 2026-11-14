"""#31 Patch B gate (in the subprocess): the blast's divergence source on the GPU.

Two independently built worlds, the same scripted blasts on both, run through the
REAL per-tick path (PhysicsRunner.step): the CPU against the per-call GPU EOS
(cuda_eos_step.cu's host mid-stage), then the CPU against the resident EOS
(K_div_source). Every synced field, the wind included, the resident telemetry and
the trace books are compared per tick at tol 0 (cuda_s8a_check._compare_tick).

The script fires, on one map: a strength 1 blast on open floor, a strength 12 blast
in a one-tile pocket (a source saturated at S_MAX), and a blast on a wall tile with
no known side. Vacuousness guards: the dispatch counters advanced, the pocket
really saturated, and a third CPU world with the source muted differs from the
others, so a GPU path that dropped the source could not pass.

Prints ``DIV_SOURCE_RESULT: PASS``/``FAIL`` and exits 0/1.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from simulation.draws import MatchDraws  # #29 P3b: the keyed draws

# Import the CUDA build FIRST so it is the cached `breach_physics`.
import breach_physics as bp

import cuda_s8a_check as s8

N_TICKS = 24
OPEN_TICK, POCKET_TICK, WALL_TICK = 3, 7, 11
H, W = 40, 60


def _rows():
    from simulation.weapons import PayloadTable
    base = dict(radius=2, wall_damage=0, unit_damage=0, explosive="comp_b",
                emit_blast_smoke=True, clear_smoke=True)
    return PayloadTable({"one": dict(base, strength=1.0),
                         "twelve": dict(base, strength=12.0)}).by_name


def _build():
    from config import CFG
    from level_loader import LevelData
    from simulation.field_edit import EditQueue
    from simulation.gamemap import GameMap
    from simulation.materials import MAT_AIR, MAT_HULL
    from simulation.physics_runner import PhysicsRunner

    tm = np.full((H, W), MAT_HULL, dtype=np.int32)
    tm[2:H - 2, 2:W - 12] = MAT_AIR
    tm[20, W - 6] = MAT_AIR                 # the one-tile pocket, sealed
    level = LevelData(name="div_source", version="2", path=Path("."),
                      tilemap=tm, tile_size_m=0.333, diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    runner = PhysicsRunner(bp)
    g.bind_physics_engine(runner.engine)
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    return runner, g, dt, EditQueue(), MatchDraws(5)


def _tick(world, t, rows, mute=False):
    from simulation.payloads import execute_payload
    runner, g, dt, q, rng = world
    if t == OPEN_TICK:
        execute_payload(g, q, [], 20, 20, rows["one"], rng)
    elif t == POCKET_TICK:
        execute_payload(g, q, [], 20, W - 6, rows["twelve"], rng)
    elif t == WALL_TICK:
        execute_payload(g, q, [], 1, 20, rows["one"], rng)
    if mute:
        g.div_source[...] = 0
    q.flush(g, rng)
    g.stamp_units([])
    destroyed = runner.step(g, dt)
    for (yy, xx) in destroyed:
        g.destroy_wall(yy, xx)
    g.heat.fill(0)


def _run(label, residency, calls):
    rows = _rows()
    s8._residency(False); s8._set_backends(False)
    cpu, gpu = _build(), _build()
    calls0 = int(calls())
    bad = 0
    for t in range(N_TICKS):
        s8._residency(False); s8._set_backends(False)
        _tick(cpu, t, rows)
        s8._residency(residency); s8._set_backends(True)
        _tick(gpu, t, rows)
        s8._residency(False); s8._set_backends(False)
        # #29 P4a: every GameMap array and every book (cuda_s8a_check's
        # comparison since the resident tail moved to the device).
        bad += s8._compare_all(label, t, cpu[1], gpu[1], cpu[0], gpu[0])
        if bad >= 8:
            print(f"  {label}: aborting after 8 divergences")
            break
    fired = int(calls()) - calls0
    if fired < N_TICKS and bad < 8:
        print(f"  {label}: the GPU path ran only {fired}/{N_TICKS} ticks: vacuous")
        return False
    if cpu[1].div_source_sat_hits < 1:
        print(f"  {label}: the pocket never saturated: vacuous")
        return False
    print(f"  {label}: {N_TICKS} ticks, {bad} divergences, "
          f"{cpu[1].div_source_sat_hits} saturated source(s), tol 0")
    return bad == 0


def _source_matters():
    """A world with the source muted must not match the world with it."""
    rows = _rows()
    s8._residency(False); s8._set_backends(False)
    a, b = _build(), _build()
    for t in range(OPEN_TICK + 3):
        _tick(a, t, rows)
        _tick(b, t, rows, mute=True)
    if np.array_equal(a[1].wind_x, b[1].wind_x):
        print("  the source moved no wind: the comparison is vacuous")
        return False
    return True


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("DIV_SOURCE_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    ok = _source_matters()
    ok = _run("per-call", False, bp.eos_step_cuda_calls) and ok
    ok = _run("resident", True, bp.eos_resident_calls) and ok
    print("DIV_SOURCE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
