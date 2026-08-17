#!/usr/bin/env python
"""P-E1 boundary instrument — flame-cell n_bulk histogram on the two-room bench.

Design `energy_transport_design_2026-08-16.md` v2.1 §6 (P-E1 row): the
MacCormack-fallback decision at this boundary needs to see WHERE the flame
cells sit on the mass axis, because the whole law change is "T = e / n_bulk"
— a flame cell's recovered temperature is only as trustworthy as the mass it
divides by. The anchor scorecard says how the fire FEELS; this says what the
denominator is doing underneath it.

Sampling: the committed `bench_two_room` fixture (the same scenario
`tools/bench_two_room.py` digests), P-F1b dials, seed 12345. Every tick, every
FIRE-ROOM cell (the left room, the one holding the crate) whose fire intensity
exceeds `--i-min` (default 0.1 — the "flame, not ember" cut) contributes one
sample of n_bulk = (O2 + inert_N2) dequantized. Output is the pooled
distribution over all (tick, cell) samples.

Run (from a worktree root):
    conda run -n data python tools/e1_flame_nbulk_histogram.py
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (ROOT, os.path.join(ROOT, "src"), os.path.join(ROOT, "tools"),
           os.path.join(ROOT, "cpp", "build", "Release")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import breach_physics as bp                          # noqa: E402
import storm_probe as sp                             # noqa: E402
from config import CFG                               # noqa: E402
from simulation import Simulation                    # noqa: E402
from simulation import fire_fixed                    # noqa: E402
from simulation import gases                         # noqa: E402
from fire_timing_harness import (                    # noqa: E402
    FP_ONE, apply_overrides, restore_overrides,
)
from bench_two_room import CRATE_XY, load_fixture    # noqa: E402

# Percentiles the report table carries (the pocket-N reading idiom P-E0 used
# for the cold-rail window: min / quartiles / max, not a fitted shape).
PCTS = (0, 1, 5, 25, 50, 75, 95, 99, 100)
# Fixed bin edges so a pre/post pair is comparable row-for-row even when the
# ranges differ (the whole point of the boundary measurement).
BINS = (0.0, 0.01, 0.05, 0.1, 0.15, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 1e9)


def collect(ticks=4800, i_min=0.1, dials=None):
    overrides = dict(dials or {})
    restore = apply_overrides(overrides) if overrides else []
    try:
        level = load_fixture()
        sim = Simulation(level, seed=12345, breach_physics=bp,
                         enable_recorder=False)
        gmap = sim.gmap
        h, w = gmap.solid.shape

        # The FIRE ROOM = the half of the map holding the crate. The fixture is
        # two rooms split by a vertical partition at mid-width with a door, so
        # "same side as the crate" is the room membership test (cheap, and it
        # cannot silently follow the partition if the fixture ever moves — the
        # fixture equivalence itself is asserted by load_fixture()).
        fx, fy = CRATE_XY
        room = np.zeros((h, w), dtype=bool)
        if fx < w // 2:
            room[:, :w // 2] = True
        else:
            room[:, w // 2:] = True
        room &= ~gmap.solid

        # The BULK pair — the `gas_conservative` planes the flux + the new
        # energy books ride (gases.py: O2 / INERT_N2 are the only two).
        cons = [gases.O2, gases.INERT_N2]

        seed_i = float(getattr(CFG.physics.fire, "ignition_seed", 0.1))
        gmap.fire[fy, fx] = fire_fixed.quantize_scalar(seed_i)
        gmap.temperature[fy, fx] = fire_fixed.quantize_scalar(280.0)

        i_min_q = i_min * FP_ONE
        samples = []
        flame_cells_per_tick = []
        for _ in range(ticks):
            sim.set_paused(False)
            sim.step()
            flame = room & (gmap.fire > i_min_q)
            ncell = int(flame.sum())
            flame_cells_per_tick.append(ncell)
            if ncell:
                nb = np.zeros((h, w), dtype=np.int64)
                for gi in cons:
                    nb += gmap.gas[gi].astype(np.int64)
                samples.append(nb[flame].astype(np.float64) / FP_ONE)
        return (np.concatenate(samples) if samples else np.zeros(0),
                np.asarray(flame_cells_per_tick))
    finally:
        restore_overrides(restore)


def report(vals, per_tick, i_min):
    print("=" * 74)
    print(f"flame-cell n_bulk histogram — fire room, intensity > {i_min}")
    print("=" * 74)
    print(f"{'samples (tick x cell)':<34}{vals.size:>12}")
    print(f"{'ticks with >=1 flame cell':<34}{int((per_tick > 0).sum()):>12}"
          f"  / {per_tick.size}")
    print(f"{'peak flame cells in one tick':<34}{int(per_tick.max()):>12}")
    if vals.size == 0:
        print("no flame samples — scenario produced no cell above the cut")
        print("=" * 74)
        return
    print(f"{'mean n_bulk':<34}{vals.mean():>12.4f}")
    print("-" * 74)
    print(f"{'percentile':<34}{'n_bulk':>12}")
    for p in PCTS:
        print(f"{'  p' + str(p):<34}{np.percentile(vals, p):>12.4f}")
    print("-" * 74)
    print(f"{'bin':<34}{'count':>12}{'share':>12}")
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        c = int(((vals >= lo) & (vals < hi)).sum())
        label = f"  [{lo:g}, {hi:g})" if hi < 1e8 else f"  [{lo:g}, inf)"
        print(f"{label:<34}{c:>12}{c / vals.size:>12.4f}")
    print("=" * 74)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ticks", type=int, default=4800, help="24 tps; 4800 = 200 s")
    ap.add_argument("--i-min", type=float, default=0.1,
                    help="flame-cell intensity cut (default 0.1)")
    ap.add_argument("--pf1b", action="store_true", default=True,
                    help="apply the P-F1b dials (default on — the bench's set)")
    ap.add_argument("--no-pf1b", dest="pf1b", action="store_false")
    a = ap.parse_args(argv)

    dials = dict(sp.PF1B) if a.pf1b else {}
    vals, per_tick = collect(a.ticks, a.i_min, dials)
    print(f"[scenario] bench_two_room, seed 12345, ticks={a.ticks}, "
          f"pf1b={a.pf1b}")
    report(vals, per_tick, a.i_min)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
