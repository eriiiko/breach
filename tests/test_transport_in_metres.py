"""#217 patch U -- wind is m/s, so the EOS step-1 transport moves air in METRES.

Erik's ruling (2026-11-20): the solver's velocity planes are metres per
second. Step 1 of ``EOSSolver::step`` turns a velocity into a displacement in
TILES through one scalar, ``eos_transport_step_q(dt, n_sub, dx)`` = dt_s/dx
(docs/eos_checkerboard_design_2026-11-20.md §7). Before patch U the scalar was
dt_s alone, so at 0.333 m tiles air and smoke moved a third of the distance
the wind said (``tools/wind_drift_probe.py``: plume drift 0.34 x |u|).

THE PROPERTY: the same wind field, described in metres, moves the same air
the same number of METRES per second at any tile size. It is measured through
the LIVE tick (``PhysicsRunner.step``, then the solver's own solve-input
caches ``dbg_mg_inputs``, written after step 1 and before the kick), never
through a reference entry, so it covers both live uses of the scalar:

  * the MASS leg -- step 1d's donor-cell face coefficient: the mass crossing a
    mid-room face line per metre of line per second equals N * U;
  * the SL leg -- step 1a's semi-Lagrangian backtrace in ``step()`` itself: a
    small front carried on a uniform wind moves U * dt metres, located as the
    centroid of the solver's own ``div_u`` cache.

WHAT MUST BREAK IT: dropping (or doubling) the 1/dx at EITHER live site --
the coefficient cache or ``step()``'s own SL backtrace -- or a second,
tile-unit scalar beside it. At 0.5 m tiles the broken leg then moves half (or
twice) the metres it moves at 1.0 m. It pins no count and no hash.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from level_loader import LevelData  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402
from simulation.physics_runner import PhysicsRunner  # noqa: E402

Q = 65536
DT = 1.0 / 24.0                 # one game tick
ROOM_W_M = 24.0                 # sealed room, metres (interior)
ROOM_H_M = 6.0
TILES = (1.0, 0.5)              # two tile sizes in metres
U = 6.0                         # the uniform wind, m/s, along +x
TOL = 0.05                      # relative: Q16 rounding + the substep split


def _room(tile):
    """A sealed hull box of ROOM_W_M x ROOM_H_M metres of air at ``tile``."""
    w = int(round(ROOM_W_M / tile)) + 2
    h = int(round(ROOM_H_M / tile)) + 2
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    lvl = LevelData(name=f"transport_{tile}", version="3", path=Path("."),
                    tilemap=tm, tile_size_m=tile, diffuse_path=Path("."))
    g = GameMap(lvl)
    g.stamp_units([])
    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    return g, runner


def _x_m(g):
    """Cell-centre x in metres from the room's west inner face."""
    h, w = g.solid.shape
    return (np.arange(w) - 1 + 0.5) * float(g.tile_size_m)


def _n_total(g):
    cons = np.flatnonzero(g.gases.conservative)
    return sum(g.gas[i].astype(np.int64) for i in cons)


def _mass_flux_per_metre(tile):
    """Mass crossing the mid-room line per metre of line per second, in
    ambient-density units times m/s (so N * U for a uniform state)."""
    g, runner = _room(tile)
    live = ~(g.solid | g.is_vacuum)
    g.wind_x[live] = int(round(U * Q))
    n0 = _n_total(g)
    runner.step(g, DT)
    h, w = g.solid.shape
    n1 = np.asarray(runner.eos.dbg_mg_inputs()[2], dtype=np.int64).reshape(h, w)
    xm = _x_m(g)
    down = live & (xm[None, :] > ROOM_W_M / 2.0)
    gained = float((n1 - n0)[down].sum()) / Q          # density x tiles
    rows = int(live[:, w // 2].sum())
    line_m = rows * tile
    n_amb = float(n0[live].mean()) / Q
    flux = gained * tile * tile / (line_m * DT)
    return flux, n_amb, int(runner.eos.dbg_last_n_sub)


def _front_shift_m(tile):
    """Metres the SL step moves a small front carried on the uniform wind."""
    g, runner = _room(tile)
    live = ~(g.solid | g.is_vacuum)
    xm = _x_m(g)
    x0, width, a = 8.0, 1.0, 0.2                          # m, m, m/s
    ux = U + a * 0.5 * (1.0 + np.tanh((xm - x0) / width))
    ux_q = np.round(ux * Q).astype(np.int32)
    g.wind_x[:, :] = np.where(live, ux_q[None, :], 0)
    u0 = g.wind_x.astype(np.float64)
    runner.step(g, DT)
    h, w = g.solid.shape
    div1 = np.asarray(runner.eos.dbg_mg_inputs()[1], dtype=np.float64).reshape(h, w)
    # Before the tick, the same interior stencil the cache uses (the face form
    # equals the central difference where all four neighbours are open).
    div0 = np.zeros_like(u0)
    div0[:, 1:-1] = u0[:, 2:] - u0[:, :-2]
    win = (xm > 3.0) & (xm < 16.0)                        # around the front,
    rows = live[:, w // 2]                                # away from the walls

    def centroid(div):
        d = div[rows][:, win].sum(axis=0)
        return float((d * xm[win]).sum() / d.sum())

    return centroid(div1) - centroid(div0), int(runner.eos.dbg_last_n_sub)


def test_the_mass_a_wind_moves_per_metre_is_the_same_at_every_tile_size():
    """Mass leg (step 1d's coefficient): Sigma_downstream(N1 - N0) * dx^2 /
    (L_m * dt) == N * U at 1.0 m and at 0.5 m tiles. Breaks if the 1/dx is
    dropped or doubled in the face coefficient: the flux then scales with dx."""
    got = {}
    for tile in TILES:
        flux, n_amb, n_sub = _mass_flux_per_metre(tile)
        assert n_amb > 0.5, "vacuous: no air in the room"
        got[tile] = flux / (n_amb * U)
    for tile, ratio in got.items():
        assert ratio == pytest.approx(1.0, rel=TOL), (
            f"at {tile} m tiles the wind moved {ratio:.3f} x the mass N*U "
            f"per metre per second (all sizes: {got}); 1/dx missing or "
            "doubled in the step-1d coefficient")


def test_the_sl_step_moves_a_front_the_same_metres_at_every_tile_size():
    """SL leg (``step()``'s own backtrace): a 0.2 m/s front on a 6 m/s wind
    moves (U + a/2) * dt metres in one tick at 1.0 m and at 0.5 m tiles,
    located in the solver's ``div_u`` cache. Breaks if the 1/dx is dropped or
    doubled at the live backtrace: the front then moves U * dt TILES."""
    expect = (U + 0.1) * DT
    got = {}
    for tile in TILES:
        shift, n_sub = _front_shift_m(tile)
        got[tile] = shift
    for tile, shift in got.items():
        assert shift == pytest.approx(expect, rel=2 * TOL), (
            f"at {tile} m tiles the SL step moved the front {shift:.4f} m, "
            f"expected {expect:.4f} m (all sizes: {got}); 1/dx missing or "
            "doubled in step()'s backtrace")
    assert got[TILES[0]] == pytest.approx(got[TILES[1]], rel=2 * TOL)
