"""tests/_drag2_sweep_bench.py -- drag-law v2 (docs/drag_law_v2_design_
2026-08-23.md §6/§8 P2 row) k2 SIZING INSTRUMENT for P3.

Standalone, NOT pytest-collected (``_`` prefix, tests/ convention). Sweeps
``k_drag2`` over a dial list on the SAME transcribed blast + 4-tile
breach-to-vacuum scenario the venting gate (tests/test_drag2_venting_gate.py)
uses -- itself transcribed from tools/tabs_pw2_venting_capture.py (design
§6's Benches-reuse rule: do not invent new geometry). Kept as a second,
independent transcription (not an import from the test module or from
tools/) so this harness stays runnable standalone with no test-module
coupling, matching this codebase's established convention (every existing
`tabs_pw2_venting_capture.py`-derived script re-transcribes rather than
imports across the tools/tests boundary).

Prints, per k2: the 50%-equalization ratio vs the k2=0 baseline (same exact
definition as the venting gate -- see that module's docstring for the full
derivation and the "why mass, not pressure" finding), the worst (max)
single-tick |u| reached anywhere in the open interior, the mean per-tick
``e_drag_deposit`` (plus its run-total for context), the rail counters
(``e_drag_rail_clipped`` energy-sum and ``t_max_phys_hits`` hit-count), and
the design §5 int64 ledger-headroom margin (worst single-tick
``ke_drag_removed`` vs the 2^31 threshold).

This is the sizing table design §1(c) and P3 need in front of Erik with the
dials -- it does NOT gate anything (no asserts): P3 re-runs/extends it at
each candidate dial.

Usage:
    C:/Users/steen/anaconda3/python.exe tests/_drag2_sweep_bench.py
    C:/Users/steen/anaconda3/python.exe tests/_drag2_sweep_bench.py --ticks 200 --k2 0 0.1 0.5 1 10
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                             # noqa: E402
from config import CFG                                   # noqa: E402
from level_loader import LevelData                       # noqa: E402
from simulation import atmosphere_fixed                  # noqa: E402
from simulation.gamemap import GameMap                    # noqa: E402
from simulation.gases import O2, INERT_N2                 # noqa: E402
from simulation.physics_runner import PhysicsRunner       # noqa: E402

H = W = 48
FP_ONE = 65536.0
DEFAULT_TICKS = 120
# Spans below/at/above the venting gate's tested {0.25,0.5,1.0} set and the
# empirically-found ~0.15-0.2 crossover (see test_drag2_venting_gate.py's
# leg-2 xfail reason), up through the k2=10 negative control.
DEFAULT_K2_DIALS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0)


def build_scenario():
    """TRANSCRIBED verbatim from tools/tabs_pw2_venting_capture.py::
    build_scenario -- see this file's module docstring."""
    tm = np.zeros((H, W), dtype=np.int32)
    tm[2:46, 2:46] = 1
    tm[3:45, 3:45] = 4
    tm[22:26, 45] = 4          # the breach: hull ring opened to the vacuum band
    level = LevelData(name="eos_p64_blast_vent", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    assert g.is_vacuum.any(), "scenario must have vacuum to vent into"

    q = atmosphere_fixed.quantize_scalar
    # gas-energy conservation arc #54, design §2.7 last row (P-G0): both
    # patches are open interior air (material 4, inside the [3:45,3:45]
    # carve), so their temperature seeds go through the seam primitive that
    # keeps gas_energy in sync — not a raw `temperature[...] =` write.
    g.seed_gas_temperature((slice(10, 16), slice(10, 16)),
                           g.temperature[10:16, 10:16] + q(5000.0))
    g.gas[O2, 11:14, 11:14] += q(4.0)
    g.seed_gas_temperature((slice(30, 36), slice(30, 36)),
                           g.temperature[30:36, 30:36] + q(15500.0))
    return g


def run_leg(ticks, k_drag2):
    g = build_scenario()
    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    runner.eos.k_drag2 = float(k_drag2)
    dt = 1.0 / float(CFG.clock.ticks_per_second)

    open_mask = ~g.solid & ~g.is_vacuum

    def n_total():
        return float((g.gas[O2][open_mask].astype(np.int64)
                      + g.gas[INERT_N2][open_mask].astype(np.int64)).sum()) / FP_ONE

    n_trace = np.empty(ticks + 1, dtype=np.float64)
    n_trace[0] = n_total()
    e_dep_sum = e_rail_sum = 0
    worst_ke = 0
    worst_speed = 0.0

    for k in range(1, ticks + 1):
        runner.step(g, dt)
        n_trace[k] = n_total()
        ke = int(runner.eos.ke_drag_removed)
        e_dep_sum += int(runner.eos.e_drag_deposit)
        e_rail_sum += int(runner.eos.e_drag_rail_clipped)
        worst_ke = max(worst_ke, ke)
        rad = (g.wind_x[open_mask].astype(np.int64) ** 2
               + g.wind_y[open_mask].astype(np.int64) ** 2)
        speed = float(np.sqrt(rad.max())) / FP_ONE if rad.size else 0.0
        worst_speed = max(worst_speed, speed)

    return dict(n_trace=n_trace, e_dep_sum=e_dep_sum, e_rail_sum=e_rail_sum,
                worst_ke=worst_ke, worst_speed=worst_speed,
                t_max_phys_hits=int(runner.eos.t_max_phys_hits))


def tick_50(n_trace, n_half):
    below = np.where(n_trace <= n_half)[0]
    return int(below[0]) if len(below) else -1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ticks", type=int, default=DEFAULT_TICKS)
    ap.add_argument("--k2", type=float, nargs="+", default=list(DEFAULT_K2_DIALS))
    args = ap.parse_args(argv)
    ticks = args.ticks
    k2_dials = args.k2

    print(f"drag2 sweep bench: {H}x{W} blast+vent scenario "
          f"(tabs_pw2_venting_capture geometry), {ticks} ticks, dials={k2_dials}\n")

    base = run_leg(ticks, 0.0)
    N0 = base["n_trace"][0]
    Nfinal = base["n_trace"][-1]
    Nhalf = N0 - 0.5 * (N0 - Nfinal)
    t50_base = tick_50(base["n_trace"], Nhalf)
    print(f"baseline (k2=0): N0={N0:.2f} Nfinal(t={ticks})={Nfinal:.2f} "
          f"Nhalf={Nhalf:.2f} tick50={t50_base}\n")

    header = (f"{'k2':>6} | {'ratio':>8} | {'worst|u|':>9} | "
              f"{'mean e_dep/tick':>16} | {'e_dep_sum':>14} | "
              f"{'e_rail_sum':>12} | {'t_max_phys':>10} | {'headroom_x':>11}")
    print(header)
    print("-" * len(header))

    def _row(k2, r):
        t50 = tick_50(r["n_trace"], Nhalf)
        ratio = (t50 / t50_base) if (t50 > 0 and t50_base > 0) else float("inf")
        mean_dep = r["e_dep_sum"] / ticks
        worst_real = r["worst_ke"] / (2.0 ** 32)
        margin = (2.0 ** 31) / worst_real if worst_real > 0 else float("inf")
        print(f"{k2:6.3f} | {ratio:8.3f} | {r['worst_speed']:9.2f} | "
              f"{mean_dep:16.4e} | {r['e_dep_sum']:14d} | "
              f"{r['e_rail_sum']:12d} | {r['t_max_phys_hits']:10d} | {margin:11.2f}")

    _row(0.0, base)
    for k2 in k2_dials:
        if k2 == 0.0:
            continue
        r = run_leg(ticks, k2)
        _row(k2, r)

    print("\nratio = 50%-equalization tick / baseline tick (bound 1.5 per "
          "design §6 -- see test_drag2_venting_gate.py for the gate + the "
          "measured k2~=0.15-0.2 empirical crossover at THIS neck)")
    print("worst|u| = max wind speed (m/s) reached anywhere in the open "
          "interior over the whole run")
    print("mean e_dep/tick, e_dep_sum = drag heat DEPOSIT (e_drag_deposit), "
          "per-tick mean and run total (raw Q16.16^2 energy-sum units)")
    print("e_rail_sum = e_drag_rail_clipped run total (energy-sum units); "
          "t_max_phys = t_max_phys_hits (cumulative hit COUNT)")
    print("headroom_x = design §5's 2^31 int64 ledger bound / worst single-"
          "tick Sigma_cells N*du^2 (real units) -- bigger is safer")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
