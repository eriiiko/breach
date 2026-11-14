"""Lit 3D units, tier 1 (#114) -- the BODY's light from the unit's one light read.

A unit's 3x3 footprint is stamped opaque into the light sweep (its floor
shadow), so the field holds light only in the footprint row facing the lamp.
``renderer/unit_shadow.py::read_unit_light`` -- the read the projected shadow
already takes -- also yields the body's irradiance (``dir_rgb`` / ``iso_rgb``),
which the lit unit shader shades the whole body with
(``[render.unit_shading] body_light = "unit"``).
Spec: docs/lit_3d_units_tier1_2026-10-27.md §5.

Headless, on the real sweep (playground, one marine, omni lamps), no GL.
Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_unit_body_light.py -q
"""
from __future__ import annotations

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
                                  body_irradiance, read_unit_light,
                                  shadow_cast)

# The marine's footprint: x 13..15, y 42..44 (where main.py spawns Bravo on
# the playground; open floor for many tiles around).
X0, Y0, FP = 13, 42, 3
ON = UnitShadowSettings(enabled=True, length=0.75, strength=0.35,
                        min_brightness=0.05, min_directionality=0.2)


def _lit_scene(lamps):
    """The playground, light requested, ONE marine at (X0, Y0) and an omni
    lamp (intensity 2.5) at each tile centre in ``lamps``, ticked twice --
    before a standing unit's float position starts to drift below its
    integer (from tick 3), so the stamped footprint is exactly X0..X0+2,
    Y0..Y0+2. Returns (light view, unit)."""
    import breach_physics as bp
    from level_loader import load as load_level
    from level_lights import LightSpec
    from renderer import frame_lights as fl
    from simulation import Simulation, light_field
    from simulation.unit import Unit
    sim = Simulation(load_level("playground"), seed=5, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    h, w = g.solid.shape
    assert not g.solid[Y0 - 6:Y0 + FP + 6, X0 - 6:X0 + FP + 6].any()
    sim.add_unit(Unit("M1", x=X0, y=Y0, team=0))
    specs = [LightSpec(x=x + 0.5, y=y + 0.5, color=(1.0, 1.0, 0.95),
                       intensity=2.5, angle_center=0.0, angle_spread=2 * math.pi,
                       kind="cursor", source="render") for x, y in lamps]
    specs = fl.frame_light_specs([], total_tick=0, sim_time_per_tick=1 / 24.0,
                                 extra=specs).specs
    sim.set_light(True, fl.cone_rows(specs, w, h, int(bp.L_FINE_BITS)),
                  np.zeros((16, 3), dtype=np.int64))
    for _ in range(2):
        sim.set_paused(False)
        sim.step()
    unit = sim.units[-1]
    assert (unit.tile_x, unit.tile_y) == (X0, Y0)
    return light_field.read_light(g), unit


def _lamp(side: str, gap: int):
    """The lamp tile ``gap`` floor tiles beyond the footprint's middle on
    ``side`` (N, S, E, W)."""
    mid_x, mid_y = X0 + FP // 2, Y0 + FP // 2
    return {"N": (mid_x, Y0 - gap - 1), "S": (mid_x, Y0 + FP + gap),
            "E": (X0 + FP + gap, mid_y), "W": (X0 - gap - 1, mid_y)}[side]


def _read(view):
    return read_unit_light(view, X0, Y0, FP, 1.0)


def _total(light: UnitLight) -> np.ndarray:
    return np.asarray(light.dir_rgb) + np.asarray(light.iso_rgb)


def test_extending_the_read_leaves_the_shadow_untouched():
    """PROPERTY: on a real lit scene, the shadow's part of the unit's read --
    brightness (the footprint's brightest channel per tile), directionality
    (|net flux| / irradiance) and the travel direction -- is exactly what the
    accessor's planes give over the footprint, and the shadow it casts is the
    same with or without the body's irradiance fields (the shadow never reads
    them).

    BREAKS IF: the #114 extension changes the shadow's inputs (a body-only
    estimate written into ``brightness`` or ``directionality``, a re-normalised
    direction), or ``shadow_cast`` starts reading ``dir_rgb`` / ``iso_rgb``.
    """
    view, _ = _lit_scene([_lamp("W", 3)])
    exposure = 4.0          # the game's is sweep_gain x master_gain
    light = read_unit_light(view, X0, Y0, FP, exposure)

    rgb = view.rgb[Y0:Y0 + FP, X0:X0 + FP].astype(np.float64)
    s = rgb.sum(axis=2)
    mag = view.directionality[Y0:Y0 + FP, X0:X0 + FP].astype(np.float64) * s
    fx = float((view.flux_dir[Y0:Y0 + FP, X0:X0 + FP, 0] * mag).sum())
    fy = float((view.flux_dir[Y0:Y0 + FP, X0:X0 + FP, 1] * mag).sum())
    norm = math.hypot(fx, fy)
    assert light.brightness == pytest.approx(rgb.sum(axis=(0, 1)).max() / FP ** 2 * exposure,
                                              rel=1e-12)
    assert light.directionality == pytest.approx(min(1.0, norm / s.sum()), rel=1e-12)
    assert (light.dir_x, light.dir_y) == pytest.approx((fx / norm, fy / norm), rel=1e-12)
    assert light.dir_x > 0.9                          # non-vacuous: lamp to the west

    bare = UnitLight(light.brightness, light.directionality, light.dir_x, light.dir_y)
    assert any(c > 0.0 for c in light.dir_rgb)        # the body fields are filled
    assert shadow_cast(light, ON) == shadow_cast(bare, ON)
    assert shadow_cast(light, ON).alpha > 0.0


def test_the_lit_side_is_as_bright_as_the_floor_beside_it():
    """PROPERTY: a unit lit from one side by an omni lamp reads, as its body's
    irradiance, about the light on the floor row just outside its footprint
    on that side (the three floor tiles along the lit face): within
    [0.7, 1.2] of it, per channel, with the lamp four floor tiles beyond the
    face. The body sits one tile further from the lamp than that row, so at
    this distance the honest value is ~0.85 of it (the light's own fall-off
    over one tile); nearer lamps widen that gap the same way the floor does.

    BREAKS IF: the dark interior dilutes the estimate (the footprint sum over
    its 9 tiles, i.e. today's ``brightness``, reads ~0.3 of the floor), the
    lit row is counted twice or the sum is divided by the lit row's length
    only for one axis (~1.7), or the one-sided and balanced shares overlap.
    """
    view, _ = _lit_scene([_lamp("N", 4)])
    light = _read(view)
    floor = view.rgb[Y0 - 1, X0:X0 + FP].astype(np.float64).mean(axis=0)
    ratio = _total(light) / floor
    assert np.all((ratio >= 0.7) & (ratio <= 1.2)), (ratio, light)
    diluted = view.rgb[Y0:Y0 + FP, X0:X0 + FP].astype(np.float64).sum(axis=(0, 1)) / FP ** 2
    assert np.all(diluted / floor < 0.5)              # the trap the property guards
    assert light.directionality > 0.8                 # one-sided, not balanced


@pytest.mark.parametrize("gap", [1, 3])
def test_a_lamp_north_south_east_or_west_lights_the_body_alike(gap):
    """PROPERTY: a lamp N, S, E or W of the unit at the same distance gives the
    same body irradiance (both shares, every channel, to float noise) and a
    direction that points AT the lamp each time (the shader lights the side
    of the body facing ``-(dir_x, dir_y)``).

    BREAKS IF: a direction's sign flips (the body lit on the side away from
    the lamp), x and y are read from each other's plane, or the width the
    footprint presents to the beam is taken along the wrong axis for one
    direction only. (An axis swap of that width on a SQUARE footprint is
    invisible here; the conservation test below holds it on non-square ones.)
    """
    cx, cy = X0 + FP / 2.0, Y0 + FP / 2.0
    reads = {}
    for side in "NSEW":
        lx, ly = _lamp(side, gap)
        view, _ = _lit_scene([(lx, ly)])
        light = _read(view)
        reads[side] = light
        to_lamp = np.array([lx + 0.5 - cx, ly + 0.5 - cy])
        to_lamp /= np.linalg.norm(to_lamp)
        assert float(np.dot((-light.dir_x, -light.dir_y), to_lamp)) > 0.999, (side, light)
    ref = reads["N"]
    assert sum(ref.dir_rgb) > 0.0
    for side, light in reads.items():
        assert light.dir_rgb == pytest.approx(ref.dir_rgb, rel=1e-5), side
        assert light.iso_rgb == pytest.approx(ref.iso_rgb, rel=1e-5, abs=1e-12), side
        assert light.directionality == pytest.approx(ref.directionality, rel=1e-5), side


def test_balanced_light_lights_the_body_evenly_not_black():
    """PROPERTY: two equal lamps on opposite sides of the unit read as
    balanced light -- directionality ~0, no one-sided share -- and the body
    still gets a balanced irradiance, at least as much light as ONE of those
    lamps alone gives it: adding an equal opposite lamp never darkens the
    body, so the shader lights every side of it instead of leaving it black.

    BREAKS IF: the body is lit from the one-sided share only (balanced light
    then leaves it black on both sides), or the balanced share is dropped
    when the net flux cancels.
    """
    view, _ = _lit_scene([_lamp("E", 2), _lamp("W", 2)])
    light = _read(view)
    assert light.directionality < 0.02
    assert sum(light.dir_rgb) == pytest.approx(0.0, abs=0.02 * sum(light.iso_rgb))
    assert all(c > 0.0 for c in light.iso_rgb)
    one_view, _ = _lit_scene([_lamp("E", 2)])
    assert np.all(_total(light) >= _total(_read(one_view)))


def test_a_unit_in_a_uniform_sky_is_as_bright_as_the_floor_around_it():
    """PROPERTY: an opaque footprint (2x2, 3x3, 4x4) standing in a uniform,
    balanced field -- the sweep's own uniform sky on an open grid -- reads as
    balanced light (directionality ~0) with a body irradiance within
    [0.85, 1.15] of the light on the open floor away from it, per channel.
    (The 3x3 block measures 1.02: the light it intercepts over its mean
    projected width, 2 (w + h) / pi.)

    BREAKS IF: the balanced share is divided by the footprint's area (the
    dark interior dilutes it: ~0.43 for 3x3), by its perimeter or by the lit
    row's length only (~0.6 to 1.3, depending on the size), or the read
    stops treating cancelling flux as balanced light.
    """
    from types import SimpleNamespace

    import _radiation_sweep_harness as H
    from simulation import light_field

    n = 41
    zero = [[0] * n for _ in range(n)]
    temp = [[H.T_AMB_Q] * n for _ in range(n)]
    sky = [[1 << 30] * 3 for _ in range(16)]
    for fp in (2, 3, 4):
        x0 = y0 = n // 2 - fp // 2
        ld = [[H.ONE if (y0 <= y < y0 + fp and x0 <= x < x0 + fp) else 0
               for x in range(n)] for y in range(n)]
        out = H.cpp_sweep(zero, zero, 0, temp, 0, H.ts_from_a(zero),
                          light=dict(la=[zero] * 3, ld=[ld] * 3, sky=sky,
                                     transport="shear"))[6]
        gmap = SimpleNamespace(
            light_q=np.ascontiguousarray(np.moveaxis(out["light_q"], 0, -1)),
            light_flux_q=np.ascontiguousarray(np.moveaxis(out["light_flux"], 0, -1)),
            light_glow=np.ascontiguousarray(np.moveaxis(out["light_glow"], 0, -1)))
        view = light_field.read_light(gmap)
        light = read_unit_light(view, x0, y0, fp, 1.0)
        floor = view.rgb[4, 4].astype(np.float64)       # open floor, far off
        assert light.directionality < 1e-3, fp
        ratio = _total(light) / floor
        assert np.all((ratio >= 0.85) & (ratio <= 1.15)), (fp, ratio)


def test_the_body_irradiance_split_conserves_the_light_the_footprint_holds():
    """PROPERTY (pure): the one-sided share times the width the footprint
    presents to the beam, plus the balanced share times its mean projected
    width, gives back the footprint's summed light, per channel, for any
    direction and any footprint shape (the two shares split the light, they
    never create or drop any); with no direction everything is balanced.

    BREAKS IF: the shares overlap or leave a gap (``d`` and ``1 - d`` not
    complementary), the beam's width ignores the footprint's shape, or a zero
    net flux divides by a zero width.
    """
    rgb = (3.0, 2.0, 1.0)
    for w, h in ((3, 3), (2, 3), (3, 1)):
        w_iso = 2.0 * (w + h) / math.pi
        for ang in np.linspace(0.0, 2.0 * math.pi, 9):
            dx, dy = math.cos(ang), math.sin(ang)
            for d in (0.0, 0.4, 1.0):
                dir_rgb, iso_rgb = body_irradiance(rgb, d, dx, dy, w, h, 2.0)
                w_dir = w * abs(dy) + h * abs(dx)
                back = [a * w_dir + b * w_iso for a, b in zip(dir_rgb, iso_rgb)]
                assert back == pytest.approx([2.0 * c for c in rgb], rel=1e-12)
        dir_rgb, iso_rgb = body_irradiance(rgb, 0.7, 0.0, 0.0, w, h, 1.0)
        assert dir_rgb == (0.0, 0.0, 0.0)
        assert iso_rgb == pytest.approx([c / w_iso for c in rgb], rel=1e-12)
