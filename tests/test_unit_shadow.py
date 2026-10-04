"""The 3D units' sharp projected shadows (#33, tier 1) -- renderer/unit_shadow.py.

Headless: the pure light -> shadow rule, the flattening matrix, and the light
read on a real lit scene. No GL context is opened (the painter is exercised by
the game itself and the in-game screenshots).

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_shadow.py -q
"""
from __future__ import annotations

import copy
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer.unit_shadow import (UnitLight, UnitShadowSettings,  # noqa: E402
                                  project_point, projection_matrix,
                                  read_unit_light, shadow_cast, smooth_shadow)
from simulation.light_field import LightView  # noqa: E402

ON = UnitShadowSettings(enabled=True, length=0.75, strength=0.35,
                        min_brightness=0.05, min_directionality=0.2)
BRIGHT_ONE_SIDED = dict(brightness=1.0, directionality=0.9)


def _light(dx, dy, brightness=1.0, directionality=0.9):
    n = math.hypot(dx, dy)
    return UnitLight(brightness, directionality, dx / n, dy / n)


def test_the_shadow_falls_away_from_the_light_and_its_length_follows_the_setting():
    """PROPERTY: for a light travelling in any horizontal direction d, the
    flattening matrix lands every point of the unit on the floor (Y' constant),
    leaves the feet where they stand, and slides a point at height H by exactly
    ``length * H`` along +d -- AWAY from the light, never toward it; doubling
    ``length`` doubles the slide.

    BREAKS IF: the direction's sign is flipped (the shadow falls toward the
    lamp -- e.g. reading the packed texture's toward-the-light convention as the
    travel direction), the x/y axes are swapped, ``length`` is ignored or scaled
    by anything but the point's height, or the matrix is laid out in the wrong
    (row- vs column-major) order.
    """
    H = 240.0
    for ang in np.linspace(0.0, 2.0 * math.pi, 13)[:-1]:
        d = (math.cos(ang), math.sin(ang))
        for length in (0.5, 0.75, 1.5):
            s = UnitShadowSettings(True, length, 0.35, 0.05, 0.2)
            cast = shadow_cast(_light(*d), s)
            m = projection_matrix(cast.vx, cast.vy, plane_y=0.0)
            fx, fy, fz = project_point(m, 100.0, 0.0, 50.0)
            assert (fx, fy, fz) == pytest.approx((100.0, 0.0, 50.0))
            hx, hy, hz = project_point(m, 100.0, H, 50.0)
            assert hy == pytest.approx(0.0)
            slide = (hx - 100.0, hz - 50.0)
            assert math.hypot(*slide) == pytest.approx(length * H, rel=1e-6)
            # away from the light: along the travel direction
            assert slide[0] * d[0] + slide[1] * d[1] == pytest.approx(length * H, rel=1e-6)


def test_no_shadow_when_off_dark_or_directionless_and_never_above_strength():
    """PROPERTY: the shadow's darkness is 0 when ``enabled`` is false, when the
    unit's patch is dark, and when the light has no clear direction
    (directionality 0, or two equal opposite lamps cancelling on a real
    LightView read); over the whole (brightness, directionality) plane it never
    exceeds ``strength``, and the per-unit smoothing keeps it there too.

    BREAKS IF: the fade is dropped or made additive (a shadow on a dark tile or
    under balanced light, whose direction is noise -- the flicker the fade
    exists to prevent), ``enabled`` stops gating, the fade overshoots 1, or the
    smoother extrapolates instead of blending.
    """
    off = UnitShadowSettings(False, 0.75, 0.35, 0.05, 0.2)
    assert shadow_cast(_light(1, 0), off).alpha == 0.0
    assert shadow_cast(_light(1, 0, brightness=0.0), ON).alpha == 0.0
    assert shadow_cast(_light(1, 0, directionality=0.0), ON).alpha == 0.0
    assert shadow_cast(UnitLight(1.0, 0.9, 0.0, 0.0), ON).alpha == 0.0
    assert shadow_cast(_light(1, 0), ON).alpha > 0.0          # non-vacuous

    prev = None
    for b in np.linspace(0.0, 5.0, 21):
        for dd in np.linspace(0.0, 1.0, 21):
            for ang in (0.0, 2.0, 4.0):
                c = shadow_cast(_light(math.cos(ang), math.sin(ang), b, dd), ON)
                assert 0.0 <= c.alpha <= ON.strength + 1e-12
                prev = smooth_shadow(prev, c, 1.0 / 60.0, 0.12)
                assert prev.alpha <= ON.strength + 1e-9

    # Two equal lamps on opposite sides of a unit, as the accessor reports them:
    # every cell one-sided, but the patch's net flux cancels.
    h, w = 8, 8
    rgb = np.full((h, w, 3), 0.5, np.float32)
    flux = np.zeros((h, w, 2), np.float32)
    flux[:, :4, 0] = 1.0                 # left half: light travels +x
    flux[:, 4:, 0] = -1.0                # right half: light travels -x
    view = LightView(rgb=rgb, flux_dir=flux, glow=np.zeros_like(rgb),
                     directionality=np.full((h, w), 0.9, np.float32))
    balanced = read_unit_light(view, x0=2, y0=2, footprint=4, exposure=1.0)
    assert balanced.directionality == pytest.approx(0.0, abs=1e-9)
    assert shadow_cast(balanced, ON).alpha == 0.0
    one_sided = read_unit_light(view, x0=0, y0=2, footprint=4, exposure=1.0)
    assert shadow_cast(one_sided, ON).alpha > 0.0


def _lit_playground_with_a_lamp_left_of_a_unit():
    """The playground with light requested and ONE omni lamp four tiles to the
    left of a marine's footprint, ticked a few times."""
    import breach_physics as bp
    from level_loader import load as load_level
    from level_lights import LightSpec
    from renderer import frame_lights as fl
    from simulation import Simulation
    level = load_level("playground")
    sim = Simulation(level, seed=5, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    h, w = g.solid.shape
    from simulation.unit import Unit
    # A marine where main.py spawns one on the playground (open floor), the
    # lamp on the open floor four tiles to his left.
    sim.add_unit(Unit("M1", x=13, y=42, team=0))
    unit = sim.units[-1]
    fp = int(getattr(unit, "footprint", 3))
    assert not g.solid[42:42 + fp, 8:13 + fp].any()
    lamp = LightSpec(x=float(unit.x) - 4 + 0.5, y=float(unit.y) + fp / 2.0,
                     color=(1.0, 1.0, 0.95), intensity=2.5, angle_center=0.0,
                     angle_spread=2 * math.pi, kind="cursor", source="render")
    specs = fl.frame_light_specs([], total_tick=0, sim_time_per_tick=1 / 24.0,
                                 extra=[lamp]).specs
    sim.set_light(True, fl.cone_rows(specs, w, h, int(bp.L_FINE_BITS)),
                  np.zeros((16, 3), dtype=np.int64))
    for _ in range(3):
        sim.set_paused(False)
        sim.step()
    return sim, unit


def test_reading_the_light_for_shadows_changes_nothing():
    """PROPERTY: the shadow's light read (the accessor's ``read_light`` plus
    ``read_unit_light`` for every unit) leaves every unit's attributes and
    every GameMap array -- the light planes included -- exactly as they were;
    and on that real scene a lamp to a unit's left gives a one-sided read whose
    shadow falls to the right, with directionality inside [0, 1] everywhere.

    BREAKS IF: the read writes anything back (a cached field stored on the
    GameMap or a Unit, an in-place normalisation of ``light_flux_q``, a
    renderer-side attribute set on a ``Unit`` -- which would enter the digest),
    or the one-sidedness stops measuring the sweep's net flux against its
    irradiance (the lamp's read would not be one-sided, or not to the right).
    """
    from simulation import light_field
    sim, unit = _lit_playground_with_a_lamp_left_of_a_unit()
    g = sim.gmap
    arrays = {k: v.copy() for k, v in vars(g).items() if isinstance(v, np.ndarray)}
    units = [copy.deepcopy(vars(u)) for u in sim.units]
    assert len(arrays) > 20 and g.light_q.any()

    view = light_field.read_light(g)
    reads = [read_unit_light(view, int(u.x), int(u.y),
                             int(getattr(u, "footprint", 3)), 1.0) for u in sim.units]

    for k, before in arrays.items():
        assert np.array_equal(getattr(g, k), before), f"GameMap.{k} changed"
    for u, before in zip(sim.units, units):
        assert vars(u).keys() == before.keys(), f"unit {u.id} gained/lost attributes"
        for k, v in before.items():
            now = vars(u)[k]
            same = (np.array_equal(now, v) if isinstance(v, np.ndarray) else now == v)
            assert same, f"unit {u.id}.{k} changed"

    assert np.all((view.directionality >= 0.0) & (view.directionality <= 1.0))
    lit = reads[sim.units.index(unit)]
    assert lit.directionality > 0.5 and lit.dir_x > 0.9, lit
    assert shadow_cast(UnitLight(1.0, lit.directionality, lit.dir_x, lit.dir_y),
                       ON).vx > 0.0
