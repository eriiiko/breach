"""Per-channel material attenuation in the directional ray march (Slice 2).

Headless C++ tests on small synthetic gmaps. Verifies:
  - opaque [1,1,1] tile fully blocks the ray downstream (== old is_wall stop),
  - glass [0.1,0.1,0.1] transmits ~90% per channel,
  - an asymmetric atten triple tints the surviving light per channel,
  - aggregate (not per-channel) termination: a colour with a clear channel
    keeps the ray alive for the other channels too (no per-channel early-out).
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


def _cast_along_x(light_atten, color=(1.0, 1.0, 1.0), h=1, w=20, sy=0, sx=0):
    """Cast a single +x ray from (sx, sy) and return the RGB field.

    A 1-row grid forces a pure +x march so each downstream tile sees the ray
    exactly once (deterministic, no diagonal aliasing).
    """
    rc = bp.Raycaster()
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    # Empty single-gas field (no smoke) — multi-gas march (engine/05 §6.2).
    gas = np.zeros((1, h, w), np.float32)
    gas_absorption = np.ones((1, 3), np.float32)
    gas_scatter = np.ones((1, 3), np.float32)
    s = bp.LightSource()
    s.x, s.y = float(sx), float(sy)
    s.max_range = float(w * 2)
    s.intensity = 1.0
    s.angle_center = 0.0          # +x
    s.angle_spread = 0.05         # a thin pencil beam along +x
    s.ray_count = 1
    # Falloff defaults to UNIFORM (no per-angle attenuation) — good for a beam.
    s.color = color
    rc.cast_source_directional(s, rgb, dx, dy,
                               gas, gas_absorption, gas_scatter, light_atten)
    return rgb


def test_opaque_tile_blocks_like_old_wall():
    # Air everywhere except an opaque [1,1,1] wall at x=5.
    h, w = 1, 20
    atten = np.zeros((h, w, 3), np.float32)
    atten[0, 5] = [1.0, 1.0, 1.0]
    rgb = _cast_along_x(atten, h=h, w=w)

    # Light reaches the wall tile (deposit happens before attenuation) ...
    assert rgb[0, 5].sum() > 0.0
    # ... but every tile BEYOND the opaque wall is dark (full block == old stop).
    assert np.all(rgb[0, 6:] == 0.0), f"light leaked past opaque wall: {rgb[0, 6:]}"


def test_glass_transmits_about_90_percent():
    # Single glass [0.1,0.1,0.1] tile at x=5; compare survivor just after it to
    # the same march with clear air in that tile.
    h, w = 1, 20
    glass = np.zeros((h, w, 3), np.float32)
    glass[0, 5] = [0.1, 0.1, 0.1]
    air = np.zeros((h, w, 3), np.float32)

    rgb_glass = _cast_along_x(glass, h=h, w=w)
    rgb_air = _cast_along_x(air, h=h, w=w)

    # Tile right after the glass: glass survivor / air survivor ~= (1 - 0.1).
    after = 6
    ratio = rgb_glass[0, after, 0] / rgb_air[0, after, 0]
    assert np.isclose(ratio, 0.9, atol=1e-3), f"glass transmission {ratio} != 0.9"
    # Glass is not opaque: light continues well past it.
    assert rgb_glass[0, 10].sum() > 0.0


def test_asymmetric_atten_tints_surviving_light():
    # Tinted window [0.9, 0.9, 0.1] at x=5 with white light: after it the blue
    # channel survives ~9x more than red/green ( (1-0.1)=0.9 vs (1-0.9)=0.1 ).
    h, w = 1, 20
    atten = np.zeros((h, w, 3), np.float32)
    atten[0, 5] = [0.9, 0.9, 0.1]
    rgb = _cast_along_x(atten, color=(1.0, 1.0, 1.0), h=h, w=w)

    after = 6
    r, g, b = rgb[0, after]
    assert b > r and b > g, f"blue channel should dominate after tint: {rgb[0, after]}"
    # Red survivor / blue survivor == (1-0.9)/(1-0.1) = 0.1/0.9 ~= 0.1111.
    assert np.isclose(r / b, 0.1 / 0.9, atol=1e-3)
    assert np.isclose(g / b, 0.1 / 0.9, atol=1e-3)


def test_aggregate_termination_no_per_channel_early_out():
    # A red light hitting a tile that kills only red [1,0,0] must still let the
    # (untouched) green/blue energy continue — but with a RED source those
    # channels are zero, so use a white source through a red-killing tile and
    # confirm the surviving (green/blue) light keeps marching past it. This
    # proves termination is on aggregate energy, not the dead red channel.
    h, w = 1, 20
    atten = np.zeros((h, w, 3), np.float32)
    atten[0, 5] = [1.0, 0.0, 0.0]   # kills red only
    rgb = _cast_along_x(atten, color=(1.0, 1.0, 1.0), h=h, w=w)

    after = 8
    r, g, b = rgb[0, after]
    assert r == 0.0, "red must be fully blocked"
    assert g > 0.0 and b > 0.0, "green/blue must survive and keep marching"
