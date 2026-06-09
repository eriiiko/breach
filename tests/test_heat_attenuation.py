"""heat_atten as the independent 4th channel of the ray march (engine/06 §1).

The directional march carries R, G, B, AND heat, each attenuated by its OWN
per-tile material coefficient: the RGB channels by ``light_atten`` (per-channel),
the heat channel by ``heat_atten`` (scalar). This file pins the heat-channel half
of that contract:

  * heat deposited PAST an obstacle tile scales with that tile's ``heat_atten``:
    air (0.0) -> full heat downrange, glass (0.3) -> partial, wall (1.0) -> ~none;
  * heat attenuation is INDEPENDENT of light: a tile that is light-opaque but
    heat-transparent passes heat while blocking light, and the converse;
  * the ray marches until ALL FOUR channels are extinguished, so a heat-opaque /
    light-clear tile does not prematurely kill the surviving light (and vice
    versa);
  * determinism: the same scene yields a bit-identical heat buffer.

Headless C++ on 1-row grids (a pure +x march so each downrange tile sees the ray
exactly once — no diagonal aliasing), mirroring test_heat_smoke_glow.py.
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

# Q16.16 scale — must match cpp/src/raycaster.h HEAT_SCALE.
HEAT_SCALE = 65536


def _make_source(color=(1.0, 1.0, 1.0), heat=1.0, intensity=1.0, w=20):
    s = bp.LightSource()
    s.x, s.y = 0.0, 0.0
    s.max_range = float(w * 2)
    s.intensity = intensity
    s.angle_center = 0.0          # +x
    s.angle_spread = 0.05         # thin pencil beam along +x
    s.ray_count = 1
    s.color = color
    s.heat = heat
    return s


def _cast(h=1, w=20, color=(1.0, 1.0, 1.0), heat_emit=1.0,
          light_atten=None, heat_atten=None, intensity=1.0):
    """Cast one +x heat-emitting beam; return (rgb, heat).

    ``light_atten`` is (h,w,3) f32 (default all-clear); ``heat_atten`` is
    (h,w) f32 (default None -> no heat attenuation, the pre-S6 path).
    """
    rc = bp.Raycaster()
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    # No gas in this file (heat_atten focus): an empty single-gas array — gases
    # never attenuate the heat channel anyway (engine/05 §6.2 / engine/06 §1).
    gas = np.zeros((1, h, w), np.float32)
    gas_absorption = np.ones((1, 3), np.float32)
    gas_scatter = np.ones((1, 3), np.float32)
    if light_atten is None:
        light_atten = np.zeros((h, w, 3), np.float32)
    heat = np.zeros((h, w), np.int32)
    glow = np.zeros((h, w, 3), np.float32)
    s = _make_source(color=color, heat=heat_emit, intensity=intensity, w=w)
    rc.cast_source_directional(s, rgb, dx, dy,
                               gas, gas_absorption, gas_scatter, light_atten,
                               heat=heat, smoke_glow=glow,
                               heat_atten=heat_atten)
    return rgb, heat


def _heat_atten_row(w, col, value):
    """A (1,w) heat-atten field that is `value` at `col`, 0 (air) elsewhere."""
    ha = np.zeros((1, w), np.float32)
    ha[0, col] = value
    return ha


def _light_atten_row(w, col, triple):
    """A (1,w,3) light-atten field that is `triple` at `col`, clear elsewhere."""
    la = np.zeros((1, w, 3), np.float32)
    la[0, col] = triple
    return la


# --------------------------------------------------- heat scales with heat_atten


def test_heat_passes_freely_through_air():
    # heat_atten 0 everywhere -> heat survival never drops; downrange heat equals
    # the no-attenuation control exactly (bit-identical, same falloff).
    w = 20
    _, heat_air = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 0.0))
    _, heat_ctrl = _cast(w=w, heat_atten=None)
    assert np.array_equal(heat_air, heat_ctrl), \
        "air (heat_atten 0) must not attenuate heat at all"
    # And there IS heat well past col 5 (the obstacle tile).
    assert heat_air[0, 10] > 0, "heat should reach downrange through air"


def test_glass_partially_attenuates_heat():
    # Glass heat_atten 0.3 -> heat PAST it is ~0.7x the air control (the deposit
    # at col 6 = survival_after_col5 * falloff(6); survival drops by (1-0.3)).
    w = 20
    _, heat_glass = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 0.3))
    _, heat_air = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 0.0))
    past = 6  # first tile AFTER the obstacle at col 5
    assert heat_air[0, past] > 0
    ratio = heat_glass[0, past] / heat_air[0, past]
    assert np.isclose(ratio, 0.7, rtol=2e-3), \
        f"glass should transmit (1-0.3)=0.7 of the heat, got ratio {ratio}"
    # Some heat passes; it is strictly less than air but strictly positive.
    assert 0 < heat_glass[0, past] < heat_air[0, past]


def test_wall_blocks_heat_past_it():
    # Wall heat_atten 1.0 -> heat survival driven to 0; ~no heat past the wall.
    w = 20
    _, heat_wall = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 1.0))
    # Heat lands ON the wall tile (deposit uses survival BEFORE this tile's
    # attenuation), but every tile strictly past it is zero.
    assert heat_wall[0, 5] > 0, "heat is deposited up to and on the wall tile"
    assert np.all(heat_wall[0, 6:] == 0), \
        f"no heat past a heat-opaque wall: {heat_wall[0, 6:]}"


def test_heat_ordering_air_glass_wall():
    # Monotone: at the first tile past the obstacle, air > glass > wall.
    w = 20
    past = 6
    _, h_air = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 0.0))
    _, h_glass = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 0.3))
    _, h_wall = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 1.0))
    assert h_air[0, past] > h_glass[0, past] > h_wall[0, past], (
        f"air {h_air[0,past]} > glass {h_glass[0,past]} > "
        f"wall {h_wall[0,past]} expected")
    assert h_wall[0, past] == 0


# ------------------------------------------ independence of heat and light atten


def test_heat_transparent_light_opaque_tile():
    # A material LIGHT-opaque ([1,1,1]) but HEAT-transparent (heat_atten 0):
    # light dies at col 5, heat sails through. This is "smoked glass".
    w = 20
    la = _light_atten_row(w, 5, (1.0, 1.0, 1.0))   # opaque to light
    ha = _heat_atten_row(w, 5, 0.0)                 # clear to heat
    rgb, heat = _cast(w=w, light_atten=la, heat_atten=ha)
    past = 6
    # Light is blocked past the opaque tile...
    assert rgb[0, past].sum() == 0.0, "light must be blocked past a light-opaque tile"
    # ...but heat passes unimpeded (== a no-obstacle heat control).
    _, heat_ctrl = _cast(w=w, heat_atten=_heat_atten_row(w, 5, 0.0))
    assert heat[0, past] > 0, "heat must pass a heat-transparent tile"
    assert np.array_equal(heat, heat_ctrl), \
        "heat through a light-opaque/heat-clear tile must equal the clear control"


def test_light_transparent_heat_opaque_tile():
    # The converse: LIGHT-transparent ([0,0,0]) but HEAT-opaque (heat_atten 1.0):
    # a heat-shield. Light sails through, heat dies at the tile.
    w = 20
    la = _light_atten_row(w, 5, (0.0, 0.0, 0.0))   # clear to light
    ha = _heat_atten_row(w, 5, 1.0)                 # opaque to heat
    rgb, heat = _cast(w=w, light_atten=la, heat_atten=ha)
    past = 6
    # Light passes unimpeded (== a fully-clear light control).
    rgb_ctrl, _ = _cast(w=w, light_atten=np.zeros((1, w, 3), np.float32),
                        heat_atten=ha)
    assert rgb[0, past].sum() > 0.0, "light must pass a light-transparent tile"
    assert np.allclose(rgb, rgb_ctrl), \
        "light through a heat-opaque/light-clear tile is unaffected by heat_atten"
    # Heat is blocked past the heat-opaque tile.
    assert np.all(heat[0, 6:] == 0), \
        f"no heat past a heat-opaque tile: {heat[0, 6:]}"


def test_light_only_source_unaffected_by_heat_atten():
    # A pure light source (heat_emit 0) deposits NO heat regardless of heat_atten,
    # and a heat-opaque tile must not shorten the light ray (4-channel cull: the
    # ray keeps going while the RGB channels survive).
    w = 20
    ha = _heat_atten_row(w, 5, 1.0)   # heat-opaque, but the source emits no heat
    rgb, heat = _cast(w=w, heat_emit=0.0, heat_atten=ha)
    assert np.all(heat == 0), "a heat=0 source deposits no heat"
    # Light still reaches far downrange — the heat-opaque tile is light-clear and
    # must not cull the (still-alive) light channels early.
    assert rgb[0, 15].sum() > 0.0, \
        "a heat-opaque/light-clear tile must not shorten the light ray"


def test_heat_only_survives_when_light_extinguished():
    # The 4-channel cull from the heat side: a tile that is LIGHT-opaque but
    # HEAT-clear must let the ray keep marching to deposit heat downrange even
    # though every RGB channel is already dead. (Pre-S6, an RGB-only cull would
    # have stopped the ray at the opaque tile and starved the heat deposit.)
    w = 20
    la = _light_atten_row(w, 3, (1.0, 1.0, 1.0))   # light dies at col 3
    ha = np.zeros((1, w), np.float32)              # heat clear everywhere
    rgb, heat = _cast(w=w, light_atten=la, heat_atten=ha)
    assert rgb[0, 4].sum() == 0.0, "light is dead past col 3"
    # Heat must still be deposited well past the light-extinction point.
    assert heat[0, 12] > 0, \
        "heat channel must survive past a light-opaque/heat-clear tile"


# --------------------------------------------------------------- determinism


def test_heat_buffer_is_bit_identical_across_casts():
    # Same scene, two independent casts -> bit-identical heat buffers (pure
    # integer saturating-add, no RNG on this path).
    w = 24
    ha = _heat_atten_row(w, 7, 0.3)
    la = _light_atten_row(w, 4, (1.0, 1.0, 1.0))
    _, heat1 = _cast(w=w, light_atten=la, heat_atten=ha)
    _, heat2 = _cast(w=w, light_atten=la, heat_atten=ha)
    assert np.array_equal(heat1, heat2), "heat buffer must be deterministic"
    assert heat1.dtype == np.int32


def test_heat_atten_none_matches_zero_field():
    # heat_atten=None (skip the attenuation, pre-S6 path) must be bit-identical
    # to an all-zero (all-air) heat_atten field.
    w = 20
    _, heat_none = _cast(w=w, heat_atten=None)
    _, heat_zero = _cast(w=w, heat_atten=np.zeros((1, w), np.float32))
    assert np.array_equal(heat_none, heat_zero), \
        "None heat_atten must equal an all-air (0.0) field"
