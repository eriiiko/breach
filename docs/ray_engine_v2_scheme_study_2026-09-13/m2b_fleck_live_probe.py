"""m2b_fleck_live_probe.py — E2E: what f does the LIVE sim compute, in a fire?

The synthetic probe (m2b_fleck_probe.py) drives RadiationSweep directly. This
one runs the real `Simulation` on the shipped `fire_tuning` level with a real
ignition and reads `engine.radiation.fleck_plane()` — the very plane the tile
inspector shows — tick by tick.

It answers ONE question end-to-end: does the Fleck damping ever engage on the
live path, on the thin rows M2 authored?

CONTROL (non-vacuity): the same run with the engine's E-table forced to the
retired FITTED scale, where M2's 0.013 came from. If the probe reports f == 1
in both, the probe is blind. It does not.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release", ROOT / "tools"):
    sys.path.insert(0, str(p))

import breach_physics as bp                    # noqa: E402
from config import CFG                         # noqa: E402
from level_loader import load as load_level    # noqa: E402
from simulation import Simulation, fire_fixed  # noqa: E402

ONE = 1 << 16
F_ONE = 1 << 24
LEVEL = "fire_tuning"
IGNITE = [(46, 8)]
SECONDS = 20.0


def run(force_scale=None, label=""):
    level = load_level(LEVEL)
    sim = Simulation(level, seed=12345, breach_physics=bp, enable_recorder=False)
    gmap = sim.gmap
    eng = sim.physics_runner.engine
    if force_scale is not None:
        eng.emissive.rad_scale = float(force_scale)
        eng.emissive.bake()
    scale = eng.emissive.rad_scale

    delta = float(CFG.physics.fire.ignition_to_ext_delta)
    for (ix, iy) in IGNITE:
        t_ext = int(gmap.fire_T_ext_plane[iy, ix]) / ONE
        gmap.temperature[iy, ix] = fire_fixed.quantize_scalar(t_ext + delta + 5.0)
        gmap.fire[iy, ix] = fire_fixed.quantize_scalar(0.12)

    tps = float(CFG.clock.ticks_per_second)
    n = int(SECONDS * tps)
    worst_f = 1.0
    worst_at = None
    n_damped_ticks = 0
    peak_T = 0.0
    for k in range(n):
        sim.set_paused(False)
        sim.step()
        f = np.asarray(eng.radiation.fleck_plane(), dtype=np.int64)
        if f.size == 0:
            continue
        # only where the sweep's solid branch can produce L > 0 at all
        mask = (gmap.heat_atten_q > 0) & gmap.thermal_solid
        if not mask.any():
            continue
        fm = f[mask]
        mn = int(fm.min()) / F_ONE
        if mn < 1.0:
            n_damped_ticks += 1
        if mn < worst_f:
            worst_f = mn
            j = int(np.argmin(fm))
            ys, xs = np.nonzero(mask)
            worst_at = (int(ys[j]), int(xs[j]),
                        int(gmap.temperature[ys[j], xs[j]]) / ONE, k + 1)
        peak_T = max(peak_T, float(gmap.temperature[mask].max()) / ONE)
    print(f"  {label:<44} rad_scale={scale:<12.6g} "
          f"min f over {n} ticks = {worst_f:.6f}  "
          f"ticks with any damping = {n_damped_ticks}/{n}  "
          f"peak solid T = {peak_T:.1f} game")
    if worst_at:
        print(f"      worst cell (y={worst_at[0]}, x={worst_at[1]}) "
              f"T={worst_at[2]:.1f} game at tick {worst_at[3]}")
    return worst_f


if __name__ == "__main__":
    print("E2E — the LIVE fleck_plane on a real burning level "
          f"({LEVEL}, {SECONDS:g} s)")
    a = run(None, "AS SHIPPED (rad_scale_derived)")
    b = run(5.1427e-5, "CONTROL: forced to the FITTED scale")
    print(f"\n  => the probe {'CAN' if b < 1.0 else 'CANNOT'} observe damping; "
          f"as shipped it reports {'NO damping' if a >= 1.0 else f'min f = {a:.4f}'}.")
