"""repro_destroy_wall_mint.py — does breaking a wall CREATE bulk gas? (P-M0)

The mass-books arc's decisive experiment
(docs/mass_books_arc_kickoff_2026-08-18.md §1.2). The event catalogue from
`debug_blowup_20260818_040647.npz` showed 87.7% of a session's 2.201x mass mint
arriving on snaps where a wall broke, with a payload that never repeats — the
signature of something proportional to local state rather than a fired weapon
payload. But in that dump every wall-break is also an explosive detonation, so
the dump alone cannot separate "the weapon deposits" from "the destruction
mints".

This isolates it. No weapon, no explosion, no solver step: build the sealed
two-room fixture, sum bulk N, call `destroy_wall` on one tile, sum again.
Any difference is minted by the destruction path itself.

ANALYSIS ONLY: drives the shipped engine; nothing in cpp/ or src/ changes.

Usage:
    conda run -n data python tools/repro_destroy_wall_mint.py
    conda run -n data python tools/repro_destroy_wall_mint.py --pressurize 20
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release",
           ROOT / "cpp" / "build_cuda" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                          # noqa: E402
import level_loader                                  # noqa: E402
from simulation import Simulation                    # noqa: E402
from simulation.gases import O2, INERT_N2            # noqa: E402

FP_ONE = 65536.0
FIXTURE = "bench_two_room"


def bulk_n(gmap):
    """Total bulk N over the map, exact int64 in raw Q16.16.

    Ambient air is o2 + n2 = 0.21 + 0.79 = 1.0, so one ambient cell is FP_ONE
    and `bulk_n / FP_ONE` reads directly as cell-equivalents of ambient air.
    """
    return (int(gmap.gas[O2].astype(np.int64).sum())
            + int(gmap.gas[INERT_N2].astype(np.int64).sum()))


def partition_wall_tiles(gmap):
    """Interior partition tiles of the two-room fixture (the mid column),
    excluding the door gap and the outer hull rows."""
    h, w = gmap.solid.shape
    mid = w // 2
    return [(y, mid) for y in range(1, h - 1) if gmap.solid[y, mid]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pressurize", type=float, default=1.0,
                    help="scale bulk gas by this factor before breaking the "
                         "wall (the mint should scale with it)")
    ap.add_argument("--walls", type=int, default=1, help="tiles to destroy")
    ap.add_argument("--step", action="store_true",
                    help="also step one tick afterwards")
    ap.add_argument("--breach", action="store_true",
                    help="break an edge hull tile into vacuum instead of an "
                         "interior wall (the is_vacuum + seed interaction)")
    a = ap.parse_args(argv)

    level = level_loader.load(FIXTURE)
    sim = Simulation(level, seed=12345, breach_physics=bp,
                     enable_recorder=False)
    gmap = sim.gmap

    if a.pressurize != 1.0:
        for g in (O2, INERT_N2):
            gmap.gas[g][:] = (gmap.gas[g].astype(np.int64)
                              * a.pressurize).astype(np.int32)

    if a.breach:
        # Force the outside to be the vacuum boundary, then break an edge hull
        # tile: destroy_wall's `exposes` rule turns it into a breach cell
        # (is_vacuum), and bulk_transport zeroes N on vacuum every pass. Does
        # the neighbour-mean seed survive, or is it minted and then deleted?
        # An EDGE HULL tile (row 0) — `on_edge_hull` fires there, so the tile
        # joins the vacuum boundary. No mask stamping needed.
        h, w = gmap.solid.shape
        tiles = [(0, w // 4)]
    else:
        tiles = partition_wall_tiles(gmap)[:a.walls]
    if not tiles:
        raise SystemExit("no interior partition wall found in the fixture")

    n0 = bulk_n(gmap)
    print(f"fixture {FIXTURE}  grid {gmap.solid.shape}  "
          f"gas cells {int((~gmap.solid).sum())}")
    print(f"pressurize x{a.pressurize:g}")
    print(f"\n  Sum N before        {n0:>18,d} raw = {n0 / FP_ONE:10.3f} cell-eq")

    total = 0
    for (fy, fx) in tiles:
        before = bulk_n(gmap)
        nb = [gmap.gas[O2][fy + dy, fx + dx] + gmap.gas[INERT_N2][fy + dy, fx + dx]
              for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
              if not gmap.solid[fy + dy, fx + dx]]
        gmap.destroy_wall(fy, fx)
        d = bulk_n(gmap) - before
        total += d
        print(f"  destroy_wall({fy:2d},{fx:2d})  dN = {d:>+14,d} raw "
              f"= {d / FP_ONE:+8.3f} cell-eq   "
              f"(open neighbours: {len(nb)}, mean {np.mean(nb) / FP_ONE if nb else 0:.3f})")

    n1 = bulk_n(gmap)
    print(f"\n  Sum N after         {n1:>18,d} raw = {n1 / FP_ONE:10.3f} cell-eq")
    print(f"  MOVED BY DESTRUCTION  {total:>+16,d} raw = {total / FP_ONE:+8.3f} "
          f"cell-eq  ({100.0 * total / n0:+.4f}% of the map's air)")
    # P-M3: the seed is now a NAMED channel, not an anonymous mint. The tool's
    # own bracket and the engine's must agree to the LSB.
    booked = int(getattr(gmap, "n_destruction_seed_sum", 0))
    print(f"  booked to n_destruction_seed_sum "
          f"{booked:>+14,d} raw = {booked / FP_ONE:+8.3f} cell-eq"
          f"   [{'MATCHES' if booked == total else 'MISMATCH'} the measured Sum N move]")

    if a.step:
        sim.set_paused(False)
        sim.step()
        n2 = bulk_n(gmap)
        print(f"\n  after one solver tick {n2:>16,d} raw = {n2 / FP_ONE:10.3f} "
              f"cell-eq   (tick alone: {(n2 - n1) / FP_ONE:+.3f} cell-eq)")

    # The DEFECT was never "total > 0" — a bounded, booked ambient seed is the
    # sanctioned behaviour (design §2). The defect was that the seed scaled with
    # local density. Run --pressurize 1 / 10 / 100 and compare: a constant
    # per-tile move is healthy, a proportional one is the amplifier.
    per_tile = total / len(tiles) / FP_ONE
    print(f"\nVERDICT: {per_tile:+.3f} cell-eq per destroyed tile. "
          f"Re-run at --pressurize 10/100: this number must NOT move.")
    # Always exit 0: this is a measurement tool, not the gate. The gate lives in
    # tests/test_destroy_wall_conserves_mass.py.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
