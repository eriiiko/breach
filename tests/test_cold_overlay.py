"""Headless unit tests for renderer/cold_overlay.py (P-W2, arc
`tabs-compression-work`, design D-7's cold-tier render instrument).

The mapping function (``pack_cold_rgba``) is pyray-free by construction
(same isolation pattern as ``renderer.blackbody.pack_emissive_rgba`` /
``renderer.hover_readout.pack_hover_readout``), so it is imported directly —
no GL, no renderer/__init__.

Two things this gate must show:
  1. The mapping is correct in isolation: T_rel >= 0 packs fully
     transparent; T_rel < 0 packs a nonzero, monotonically-deepening alpha,
     clamped at the deepest stop.
  2. "Visually-by-numbers" (design D-7): a real venting-adjacent scenario
     (the P-W0/P-W2 quiet-room recipe, a handful of ticks — cheap and
     deterministic) produces sub-ambient interior cells, and the cold pass
     would receive nonzero alpha on them — i.e. the overlay is wired to a
     scenario the T_abs law actually makes cold, not just a synthetic array.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer.cold_overlay import COLD_STOPS, TEMP_SCALE, pack_cold_rgba


def _q(t_game):
    return np.round(np.asarray(t_game) * TEMP_SCALE).astype(np.int32)


def test_ambient_and_warm_cells_are_fully_transparent():
    t = _q([[0.0, 5.0, 300.0]])
    rgba = pack_cold_rgba(t)
    assert np.all(rgba[..., 3] == 0)
    assert np.all(rgba == 0)  # premultiplied: alpha 0 -> rgb 0 too


def test_cold_cells_pack_nonzero_monotonic_alpha():
    t = _q([[-1.0, -10.0, -50.0, -150.0, -500.0]])
    rgba = pack_cold_rgba(t)
    alpha = rgba[0, :, 3].astype(np.int64)
    assert alpha[0] > 0, "even a 1-degree chill must be visible (nonzero alpha)"
    # Monotonically non-decreasing as T_rel drops (colder reads more opaque).
    assert np.all(np.diff(alpha) >= 0)
    # Clamped at the deepest stop: -150 and -500 read identically (the ramp
    # bottoms out at COLD_STOPS' most negative entry).
    assert alpha[3] == alpha[4] == int(COLD_STOPS[-1, 4])


def test_stop_table_is_ordered_ambient_to_deep_cold():
    # COLD_STOPS[:, 0] must run 0 -> increasingly negative (pack_cold_rgba's
    # np.interp flip assumes this order) and alpha must increase monotonically.
    assert np.all(np.diff(COLD_STOPS[:, 0]) < 0)
    assert np.all(np.diff(COLD_STOPS[:, 4]) >= 0)


def test_venting_adjacent_scenario_produces_nonzero_cold_pass():
    """Visually-by-numbers (design D-7): the P-W0/P-W2 quiet-room recipe (a
    +0.1 atm Gaussian bump in a sealed ambient box) rings through an
    expansion phase within a handful of ticks under the T_abs law -- assert
    the cold pass actually lights up on it, not just on a synthetic array."""
    pytest.importorskip("breach_physics", reason="needs the compiled breach_physics")
    import quiet_room_drift as qrd

    g = qrd._ambient_gmap(qrd.H, qrd.W, qrd.derive_ambient())
    runner = qrd.PhysicsRunner(qrd.bp)
    interior = (~g.solid) & (~g.is_ambient)
    qrd._seed_pressure_bump(g, interior)

    for _ in range(5):
        runner.step(g, qrd.DT_TICK)

    assert (g.temperature[interior] < 0).any(), (
        "scenario setup regressed: expected sub-ambient interior cells "
        "within 5 ticks of the pressure-bump ring")

    rgba = pack_cold_rgba(g.temperature)
    cold_mask = g.temperature < 0
    assert (rgba[..., 3][cold_mask] > 0).any(), (
        "cold pass received sub-ambient cells but packed zero alpha "
        "everywhere -- the overlay would show nothing on a real scenario")
