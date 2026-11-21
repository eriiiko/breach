"""The lamps' own glints on the water (#97, Erik 2026-11-21: "I'd like
scatter, Snell, lamp glint") -- properties of the twin
(``renderer/water_surface.py``: ``lamp_table``, ``lamp_select``,
``lamp_glint``) and of how THE one light assembly feeds the water pass. The
shader is held to the twin by ``tests/gl_water_caustic_check.py``
(``lamp_glint_matches_twin``).

Each test names the property it protects and the change that must break it.
No GL, no sim.
"""
from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import numpy as np
import pytest

from level_lights import LightSpec
from renderer import water_surface as W

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def settings():
    from config import CFG
    return W.water_settings(CFG)


def _lamp(x, y, inten=1.0, **kw):
    return LightSpec(x=x, y=y, color=(1.0, 1.0, 1.0), intensity=inten, **kw)


def test_the_glint_is_on_by_default(settings):
    """Property: the shipped look carries the lamps' glints (Erik's call,
    rendered at 1 on the reassessment page). Breaks if the default turns
    them off again."""
    assert settings.lamp_glint > 0.0


def test_flat_water_mirrors_a_lamp_straight_under_itself(settings):
    """Property: on flat water seen from straight above, a lamp's glint is
    brightest exactly under the lamp, falls off with the distance from that
    point, and is gone a few lamp heights away -- the mirror image of a
    ceiling lamp (what the field's one elevation, light_z, cannot give).
    Breaks if the lamp's direction is taken from the field (one elevation:
    no peak) or the lamp's height is dropped."""
    s = dataclasses.replace(settings, lamp_glint=1.0)
    lamps = W.lamp_table([_lamp(5.0, 5.0)], 1.0)
    cx = cy = 5.5                            # the lamp's cell centre, m
    xs = np.linspace(cx, cx + 3 * s.lamp_height_m, 61)
    p = np.stack([xs, np.full_like(xs, cy)], -1)
    N = np.tile([0.0, 0.0, 1.0], (xs.size, 1))
    g = W.lamp_glint(p, N, lamps, s, s.roughness_base)
    assert g[0] == g.max() and g[0] > 1.0
    assert np.all(np.diff(g) <= 1e-12)
    assert g[-1] < 1e-3 * g[0]


def test_a_tilted_ripple_throws_the_lamp_off_its_nadir(settings):
    """Property: a ripple tilted toward the lamp by half the lamp's angle
    from vertical mirrors the lamp where flat water would not (the glitter
    out to ~2 x height x slope). Breaks if the normal does not enter the
    glint (a flat-water disc)."""
    s = dataclasses.replace(settings, lamp_glint=1.0)
    lamps = W.lamp_table([_lamp(0.0, 0.0)], 1.0)          # centre (0.5, 0.5)
    d = 0.6
    half = math.atan(d / s.lamp_height_m) / 2.0
    p = np.array([[0.5 + d, 0.5]])
    tilted = np.array([[-math.sin(half), 0.0, math.cos(half)]])
    flat = np.array([[0.0, 0.0, 1.0]])
    gt = W.lamp_glint(p, tilted, lamps, s, s.roughness_base)[0]
    gf = W.lamp_glint(p, flat, lamps, s, s.roughness_base)[0]
    assert gt > 10.0 * gf


def test_a_beam_lamp_glints_only_in_its_wedge(settings):
    """Property: a spot or beacon lights only its wedge -- the bearing from
    the lamp inside ``angle_spread`` about ``angle_center`` (the specs'
    screen convention) -- so its glint is there and nowhere behind it, and
    its share carries ``2 pi / spread`` (the sweep puts a beam's power into
    its spread). Breaks if a beam is taken as an omni lamp, its centre's
    sign or convention flips, or its share ignores the spread."""
    s = dataclasses.replace(settings, lamp_glint=1.0)
    spot = _lamp(10.0, 10.0, 2.0, angle_center=0.5 * math.pi,   # +y (south)
                 angle_spread=math.radians(60.0), kind="spot")
    t = W.lamp_table([spot], 1.0)
    assert t.shape == (1, 6)
    assert t[0, 2] == pytest.approx(2.0 * 360.0 / 60.0)
    up = np.array([[0.0, 0.0, 1.0]])
    south = np.array([[10.5, 10.5 + 0.4]])
    north = np.array([[10.5, 10.5 - 0.4]])
    east = np.array([[10.5 + 0.4, 10.5]])
    assert W.lamp_glint(south, up, t, s, 0.1)[0] > 0.0
    assert W.lamp_glint(north, up, t, s, 0.1)[0] == 0.0
    assert W.lamp_glint(east, up, t, s, 0.1)[0] == 0.0


def test_no_glint_when_off_or_without_mounted_lamps(settings):
    """Property: the dial 0 or no mounted lamp gives exactly 0; only the
    level's MOUNTED kinds (lamp, spot, beacon) enter -- the flashlight, the
    cursor lamp and the transients have no height in their spec and keep
    the field's own glint (the ACCEPTED GAP). Breaks if the term leaks past
    its dial, or a flashlight or a muzzle flash is taken as a ceiling
    lamp."""
    p = np.zeros((4, 2))
    N = np.tile([0.0, 0.0, 1.0], (4, 1))
    lamps = W.lamp_table([_lamp(0.0, 0.0)], 0.5)
    off = dataclasses.replace(settings, lamp_glint=0.0)
    assert np.all(W.lamp_glint(p, N, lamps, off, 0.08) == 0.0)
    on = dataclasses.replace(settings, lamp_glint=1.0)
    assert np.all(W.lamp_glint(p, N, np.zeros((0, 6)), on, 0.08) == 0.0)
    specs = [_lamp(3.2, 4.7, 2.0, kind="lamp"),
             _lamp(1.0, 1.0, 5.0, kind="flashlight", angle_spread=0.8),
             _lamp(2.0, 2.0, 5.0, kind="cursor"),
             _lamp(6.0, 6.0, 5.0, kind="transient")]
    rows = W.lamp_table(specs, 0.5)
    assert rows.shape == (1, 6)
    assert np.allclose(rows[0, :3], [3.5 * 0.5, 4.5 * 0.5, 2.0])
    assert set(W.LAMP_KINDS) >= {"lamp", "spot", "beacon"}


def test_the_lamps_sent_are_the_nearest_to_the_drawn_water():
    """Property: when there are more lamps than the shader takes, the ones
    nearest the drawn rect (0 inside it) are kept, so the lamps over the
    water are never dropped for lamps over dry floor. Breaks if the
    selection keeps the table's order or the view's centre."""
    far = [_lamp(100.0 + i, 0.0) for i in range(W.LAMP_GLINT_MAX)]
    near = [_lamp(5.0, 5.0), _lamp(7.0, 6.0)]
    t = W.lamp_table(far + near, 1.0)
    sel = W.lamp_select(t, (4.0, 4.0, 8.0, 8.0))
    assert sel.shape[0] == W.LAMP_GLINT_MAX
    kept = {tuple(r[:2]) for r in sel}
    assert (5.5, 5.5) in kept and (7.5, 6.5) in kept


def test_the_game_and_the_demo_hand_the_water_pass_the_assembly_lists():
    """Property: the water pass's lamps come from THE one light assembly in
    both drivers -- main.py hands it the same two spec lists its cone rows
    are built from, the lighting demo (and so the look shots) the list
    ``demo_light_rows`` builds its rows from, at every call site -- never a
    second light list. Breaks if a driver builds its own lamp list for the
    water, or a demo path stops handing the renderer over."""
    main = (ROOT / "main.py").read_text(encoding="utf-8")
    assert "renderer.water_pass.set_lamp_specs(static_specs, dyn)" in main
    assert "frame_lights.cone_rows(dyn," in main
    demo = (ROOT / "tools" / "lighting_demo.py").read_text(encoding="utf-8")
    assert "renderer.water_pass.set_lamp_specs(specs)" in demo
    assert "return frame_lights.cone_rows(specs," in demo
    import re
    calls = re.findall(r"(?<!def )demo_light_rows\(lights_in,[^)]*\)", demo,
                       flags=re.S)
    assert calls and all("renderer=renderer" in c for c in calls), calls
