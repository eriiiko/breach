"""P-G2 ad-hoc AB probe (implementer verification, not a committed gate).

Runs the SAME level through PhysicsRunner twice -- once on the CPU EOS/bulk/
temperature backends (the default), once with every relevant CUDA backend
flag flipped on (sl_advection, bulk_flux, mg_solve, kick_compression,
temperature) -- and diffs gas_energy/temperature/wind_x/wind_y/atmosphere at
tol 0 every tick. Must run inside the CUDA python process (cuda_harness).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import breach_physics as bp

ROOT = Path(__file__).resolve().parent.parent


def _make_runner(level_name):
    from level_loader import load as load_level
    from simulation.gamemap import GameMap
    from simulation.physics_runner import PhysicsRunner

    level = load_level(level_name)
    g = GameMap(level)
    g.stamp_units([])
    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    return g, runner


def _seed_scenario(g):
    from simulation import atmosphere_fixed
    from simulation.gases import O2
    q = atmosphere_fixed.quantize_scalar
    h, w = g.temperature.shape
    cy, cx = h // 2, w // 2
    r = max(2, min(h, w) // 8)
    y0, y1 = max(0, cy - r), min(h, cy + r)
    x0, x1 = max(0, cx - r), min(w, cx + r)
    g.temperature[y0:y1, x0:x1] += q(1200.0)
    g.gas[O2, y0:y1, x0:x1] += q(1.0)
    # a cooler, denser pocket too, to exercise the compression/expansion range
    y2, y3 = max(0, cy + 2 * r), min(h, cy + 4 * r)
    if y2 < y3:
        g.temperature[y2:y3, x0:x1] -= q(50.0)


def _counters(runner):
    """arc #54 P-G1d: EVERY int-valued ledger/rail counter the EOS and the
    thermal solver expose, gathered by NAME so a counter added later joins the
    parity gate automatically instead of silently escaping it. Returned as one
    int64 vector in sorted-name order (with the names, for the report)."""
    names, vals = [], []
    for owner, prefix in ((runner.eos, "eos."),
                          (runner.engine.temperature, "temp.")):
        for a in sorted(dir(owner)):
            if a.startswith("_") or not (a.endswith("_sum")
                                         or a.endswith("_hits")
                                         or a.endswith("_calls")):
                continue
            try:
                v = getattr(owner, a)
            except Exception:  # noqa: BLE001 — a property that needs live state
                continue
            if isinstance(v, bool) or not isinstance(v, int):
                continue
            names.append(prefix + a)
            vals.append(int(v))
    return names, np.array(vals, dtype=object)


def run_trajectory(g, runner, n_steps, dt):
    from config import CFG
    from simulation.physics_runner import PhysicsRunner  # noqa: F401
    # arc #54 P-G1d: the GAS PLANES join the field list (they are the mass side
    # of every energy face; a transport twin can diverge on gas alone and leave
    # gas_energy briefly identical), and the counters join as a pseudo-field so
    # every rail and ledger is compared tick by tick, not just at the end.
    fields = ["gas_energy", "temperature", "wind_x", "wind_y", "atmosphere",
              "gas"]
    traj = {f: [] for f in fields}
    traj["_counters"] = []
    names = None
    for _ in range(n_steps):
        runner.step(g, dt)
        for f in fields:
            traj[f].append(np.array(getattr(g, f)).copy())
        names, vals = _counters(runner)
        traj["_counters"].append(vals)
    traj["_counter_names"] = names
    return traj


def diff(tag, a, b):
    ok = True
    names = a.get("_counter_names") or []
    for f in a:
        if f == "_counter_names":
            continue
        for t, (fa, fb) in enumerate(zip(a[f], b[f])):
            if not np.array_equal(fa, fb):
                mism = int(np.count_nonzero(fa != fb))
                idx = int(np.argmax(fa != fb))
                who = (f" ({names[idx]})" if f == "_counters" and idx < len(names)
                       else "")
                print(f"  [{tag}] tick {t} field {f}: {mism} entries differ, "
                      f"first @ {idx}{who} cpu={fa.flat[idx]} "
                      f"cuda={fb.flat[idx]}")
                ok = False
                break
        if not ok:
            break
    return ok


def probe_level(level_name, n_steps=20):
    from config import CFG
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    print(f"=== {level_name} ===")
    g_cpu, r_cpu = _make_runner(level_name)
    _seed_scenario(g_cpu)
    traj_cpu = run_trajectory(g_cpu, r_cpu, n_steps, dt)

    bp.set_sl_advection_backend(True)
    bp.set_bulk_flux_backend(True)
    bp.set_mg_solve_backend(True)
    bp.set_kick_compression_backend(True)
    bp.set_temperature_backend(True)
    try:
        g_gpu, r_gpu = _make_runner(level_name)
        _seed_scenario(g_gpu)
        traj_gpu = run_trajectory(g_gpu, r_gpu, n_steps, dt)
    finally:
        bp.set_sl_advection_backend(False)
        bp.set_bulk_flux_backend(False)
        bp.set_mg_solve_backend(False)
        bp.set_kick_compression_backend(False)
        bp.set_temperature_backend(False)

    ok = diff(level_name, traj_cpu, traj_gpu)
    if ok:
        shown = [k for k in traj_cpu if k != "_counter_names"]
        print(f"  PASS: {n_steps} ticks bit-identical on {shown} "
              f"({len(traj_cpu.get('_counter_names') or [])} counters)")
    return ok


def main() -> int:
    if not getattr(bp, "HAS_CUDA", False) or not bp.cuda_available():
        print("PG2_AB_RESULT: FAIL (no CUDA build / device)")
        return 1
    print("device:", bp.cuda_device_info())
    levels = sys.argv[1:] or ["playground"]
    ok = True
    for lvl in levels:
        try:
            ok &= probe_level(lvl)
        except Exception as e:  # noqa: BLE001
            import traceback
            traceback.print_exc()
            print(f"  EXCEPTION on {lvl}: {e}")
            ok = False
    print("PG2_AB_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
