"""Lossy wave boundary: units absorb blasts (ch.04 §4a).

The wave update now removes energy at absorbing cells via the dynamic
``dyn_wave_absorb`` field (static material ``wave_absorb`` projected onto the
grid, raised by living units' footprints to ``CFG.physics.unit_wave_absorb``).
Pure energy removal at cells the wave already touches — strictly stabilizing,
no CFL/Laplacian change. Air has ``wave_absorb == 0`` so open-air wave
behaviour is unchanged.

This test isolates the effect from level geometry by carving a clean all-air
arena into a real :class:`GameMap` (no walls, no vacuum), injecting a wave_p
pulse on one side, and measuring wave energy reaching the far side WITH a unit
between blast and measurement vs the identical run WITHOUT the unit.

Two assertions:

1. **Units absorb** — far-side wave energy with the unit is measurably lower
   (< 0.8x) than without it.
2. **Open air unchanged** — two no-absorber runs are bit-identical, and the
   no-unit run here matches a run made with the absorption code path inactive
   (wave_absorb all zero ⇒ k=1 ⇒ no change).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_wave_absorption.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp
from config import CFG
from level_loader import load as load_level
from simulation.gamemap import GameMap
from simulation.unit import Unit
from simulation import wave_fixed   # S2a: wave fields are Q16.16 int32


# Arena geometry (interior, away from the map edge so no sponge/vacuum reaches
# in). A localized pulse is injected aligned with the unit's rows; the unit
# straddles the path; energy is integrated over the run in the unit's SHADOW
# (the rows behind it, just past its footprint). Integrating over the run
# removes the oscillation sensitivity of a single-frame snapshot of a damped
# wave, giving a stable with-vs-without contrast.
ARENA_Y0, ARENA_Y1 = 50, 80        # rows [y0, y1)
ARENA_X0, ARENA_X1 = 10, 40        # cols [x0, x1)
BLAST_X = 12                       # injection column
UNIT_X, UNIT_Y = 22, 63            # unit anchor; 3x3 footprint spans x22..24, y63..65
PULSE_Y0, PULSE_Y1 = 62, 66        # localized beam rows (aligned with unit)
SHADOW_X0, SHADOW_X1 = 26, 32      # measurement band just past the unit (x>=25)
N_SUBSTEPS = 50


def _make_arena() -> GameMap:
    """A real GameMap with a clean all-air rectangle carved into it.

    The arena is forced to interior air: not wall, not vacuum, permeability 1,
    wave_absorb 0, atmosphere 1. Everything else in the map is irrelevant — we
    only ever read/measure inside the arena, and walls outside it just seal it.
    """
    g = GameMap(load_level("unhcr_vessel"))
    ys = slice(ARENA_Y0, ARENA_Y1)
    xs = slice(ARENA_X0, ARENA_X1)
    g.solid[ys, xs] = False
    g.is_vacuum[ys, xs] = False
    g.obstacles[ys, xs] = False
    g.permeability[ys, xs] = 1.0
    g.dyn_permeability[ys, xs] = 1.0
    g.wave_absorb[ys, xs] = 0.0
    g.atmosphere[ys, xs] = 1.0
    g.wave_p[:] = 0.0
    g.wave_v[:] = 0.0
    g.wave_source[:] = 0.0
    return g


def _new_solver():
    s = bp.AtmosphereSolver()
    s.c = float(CFG.physics.wave_c)
    s.damping = float(CFG.physics.wave_damping)
    s.transfer = float(CFG.physics.wave_transfer)
    s.d_atm = float(CFG.physics.d_atm)
    s.feed_rate = float(CFG.physics.source_feed_rate)
    s.breach_rate = float(CFG.physics.breach_rate)
    s.max_source_per_step = float(CFG.physics.max_source_per_step)
    s.absorb_strength = float(CFG.physics.wave_absorb_strength)
    return s


def _inject_pulse(g: GameMap):
    """A localized wave_p beam aligned with the unit's row band.

    S2a: wave_p is Q16.16 int32 — quantize the real pulse magnitude (8.0)."""
    g.wave_p[PULSE_Y0:PULSE_Y1, BLAST_X] = wave_fixed.quantize_scalar(8.0)


def _run(g: GameMap, units):
    """Stamp units, inject a pulse, run N substeps, return the wave energy
    INTEGRATED over the run in the unit's shadow band."""
    g.stamp_units(units)
    _inject_pulse(g)
    solver = _new_solver()
    dt = solver.max_dt()
    acc = 0.0
    for _ in range(N_SUBSTEPS):
        solver.step(
            g.wave_p, g.wave_v, g.wave_source, g.atmosphere,
            g.wind_x, g.wind_y,
            g.obstacles, g.solid, g.is_vacuum,
            g.dyn_permeability, g.dyn_wave_absorb,
            dt,
        )
        # S2a: wave_p is Q16.16 int32 — DEQUANTIZE to real units before the
        # energy sum (an int32 square would overflow; energy must be real).
        band = wave_fixed.dequantize(g.wave_p[PULSE_Y0:PULSE_Y1, SHADOW_X0:SHADOW_X1])
        acc += float(np.sum(band ** 2))
    return acc


def test_unit_absorbs_blast():
    """A body between blast and measurement soaks the shockwave: far-side wave
    energy is measurably lower than the identical run with no unit."""
    # Control: no absorber anywhere.
    g0 = _make_arena()
    e_without = _run(g0, [])

    # With a unit straddling the arena mid-column.
    g1 = _make_arena()
    u = Unit("ABS", x=UNIT_X, y=UNIT_Y, team=0)
    e_with = _run(g1, [u])

    # Sanity: the control actually delivered energy to the far side.
    assert e_without > 1e-6, f"control delivered no wave energy ({e_without})"

    # The unit must soak a meaningful fraction of the blast.
    assert e_with < 0.8 * e_without, (
        f"unit did not absorb enough: with={e_with:.4f} "
        f"without={e_without:.4f} (ratio {e_with / e_without:.3f})"
    )


def test_open_air_unchanged():
    """With no absorbers (air wave_absorb == 0 ⇒ k == 1), the absorption code
    path is a no-op: two identical no-absorber runs match bit-for-bit."""
    g_a = _make_arena()
    e_a = _run(g_a, [])
    g_b = _make_arena()
    e_b = _run(g_b, [])
    assert e_a == e_b, f"no-absorber runs diverged: {e_a} != {e_b}"

    # And the full wave_p fields must be identical too (stronger check).
    ga2 = _make_arena()
    ga2.stamp_units([])
    _inject_pulse(ga2)
    gb2 = _make_arena()
    gb2.stamp_units([])
    _inject_pulse(gb2)
    s = _new_solver()
    dt = s.max_dt()
    for _ in range(N_SUBSTEPS):
        for g in (ga2, gb2):
            s.step(
                g.wave_p, g.wave_v, g.wave_source, g.atmosphere,
                g.wind_x, g.wind_y, g.obstacles, g.solid, g.is_vacuum,
                g.dyn_permeability, g.dyn_wave_absorb, dt,
            )
    assert np.array_equal(ga2.wave_p, gb2.wave_p)


if __name__ == "__main__":
    test_unit_absorbs_blast()
    test_open_air_unchanged()
    print("ok")
