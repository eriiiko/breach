"""The dynamic per-channel LIGHT extinction plane ``dyn_light_atten_q``.

``stamp_units`` rebuilds, every tick, the integer plane the sweep's light
channels read: the static material extinction ``light_atten_q`` (Q16, (h, w,
3)) combined via per-channel MAX with each living unit's own ``light_atten``
triple stamped over its footprint (default [1, 1, 1] -> ONE = an opaque body,
the unit's shadow).

Rewritten at ray-engine-v2 P6c (design v3 §11.2 row "test_dyn_light_atten.py:
4 survive on the integer twin, 1 rewritten"): until P6c these four stamp
properties were held on the FLOAT plane ``dyn_light_atten`` the old render
march read; that plane is deleted with the march, and the properties hold on
its integer twin. The fifth test -- a C++ march ray blocked downstream of a
stamped unit -- is held on the sweep by
tests/test_radiation_sweep_light.py::test_a_stamped_marine_blocks_light_and_does_not_glow
(P6a). The GameMaps here have no engine bound, so they exercise the Python
reference stamp; tests/test_stamp_units_cpp_ab.py holds the C++ stamp equal to
it.

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_dyn_light_atten.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from level_loader import load as load_level  # noqa: E402
from simulation import optics_fixed  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.unit import Unit  # noqa: E402

ONE = optics_fixed.FP_ONE


def _clear_air_anchor(g: GameMap, footprint: int = 3):
    """An interior anchor (tile_y, tile_x) whose footprint-sized block is fully
    passable air (so the static extinction there is 0 on every channel)."""
    h, w = g.material.shape
    for y in range(2, h - footprint - 2):
        for x in range(2, w - footprint - 2):
            if g.is_passable_block(y, x, footprint):
                if not g.light_atten_q[y:y + footprint, x:x + footprint].any():
                    return y, x
    raise AssertionError("no clear-air footprint found in level")


def test_a_unit_stamps_an_opaque_shadow_and_nothing_else_moves():
    """PROPERTY: a living unit with the default opacity raises
    dyn_light_atten_q to ONE on every channel over its footprint; everywhere
    else the dynamic plane equals the static one, and the static plane itself
    is not mutated.

    BREAKS IF: the stamp misses footprint tiles, leaks outside the footprint,
    writes the static plane, or quantizes the default opacity below ONE.
    """
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    u = Unit("U1", x=ax, y=ay, team=0)
    static_before = g.light_atten_q.copy()
    g.stamp_units([u])
    occ = u.occupied_tiles()
    for (tx, ty) in occ:
        assert np.all(g.dyn_light_atten_q[ty, tx] == ONE), \
            f"unit tile ({tx},{ty}) not opaque: {g.dyn_light_atten_q[ty, tx]}"
    mask = np.ones((g._h, g._w), dtype=bool)
    for (tx, ty) in occ:
        mask[ty, tx] = False
    assert np.array_equal(g.dyn_light_atten_q[mask], g.light_atten_q[mask]), \
        "dyn diverged from static outside the unit footprint"
    assert np.array_equal(g.light_atten_q, static_before), "static light_atten_q mutated"


def test_the_plane_is_filled_in_place_never_reallocated():
    """PROPERTY: dyn_light_atten_q is written IN PLACE every tick -- the same
    buffer object across stamps -- so a C++ view or a residency device buffer
    of it never goes stale.

    BREAKS IF: a stamp path reassigns the plane (``self.dyn_light_atten_q =
    ...``) instead of filling it.
    """
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    buf_id = id(g.dyn_light_atten_q)
    u = Unit("U1", x=ax, y=ay, team=0)
    g.stamp_units([u])
    assert id(g.dyn_light_atten_q) == buf_id, "dyn_light_atten_q was reassigned"
    g.stamp_units([u])
    assert id(g.dyn_light_atten_q) == buf_id, "dyn_light_atten_q reassigned on second tick"


def test_the_per_channel_max_only_adds_opacity():
    """PROPERTY: the stamp is a per-channel MAX -- a partial, per-colour unit
    opacity over a partial material keeps, channel by channel, the larger of
    the two (never their sum, never the unit's alone), and over clear air it is
    exactly the unit's triple, quantized once.

    BREAKS IF: the stamp sums, overwrites the material, or collapses the
    channels to one value.
    """
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    half = optics_fixed.quantize_scalar(0.5)
    g.light_atten_q[ay, ax] = [half, half, half]
    u = Unit("U1", x=ax, y=ay, team=0)
    u.light_atten = (0.2, 0.0, 0.8)
    q = [optics_fixed.quantize_scalar(v) for v in u.light_atten]
    g.stamp_units([u])
    assert list(g.dyn_light_atten_q[ay, ax]) == [half, half, q[2]], \
        f"per-channel max wrong: {g.dyn_light_atten_q[ay, ax]}"
    assert list(g.dyn_light_atten_q[ay + 2, ax + 2]) == q


def test_a_dead_unit_casts_no_shadow():
    """PROPERTY: a dead unit stamps nothing -- the dynamic plane equals the
    static one.

    BREAKS IF: the stamp stops filtering on ``u.alive``.
    """
    g = GameMap(load_level("unhcr_vessel"))
    ay, ax = _clear_air_anchor(g)
    u = Unit("U1", x=ax, y=ay, team=0)
    u.alive = False
    g.stamp_units([u])
    assert np.array_equal(g.dyn_light_atten_q, g.light_atten_q), \
        "dead unit must not alter the dynamic plane"
