"""Verify the RGB light deposit + 16F pack math (Slice 1, RGB light).

Headless: exercises the C++ raycaster's per-channel RGB deposit and the
renderer's pack layout (Texture A = light_rgb + dir.x, Texture B = dir.y)
without opening a GL window. Visual colour correctness is a separate human
spot-check.
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


def _cast(color):
    rc = bp.Raycaster()
    h, w = 30, 30
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    smoke = np.zeros((h, w), np.float32)
    # Per-channel static material attenuation (all air -> no occlusion).
    light_atten = np.zeros((h, w, 3), np.float32)
    s = bp.LightSource()
    s.x, s.y = 15, 15
    s.max_range = 10
    s.intensity = 1.0
    s.angle_spread = 6.283
    s.color = color
    rc.cast_source_directional(s, rgb, dx, dy, smoke, light_atten)
    bp.Raycaster.normalize_directions(dx, dy)
    return rgb, dx, dy


def test_default_color_is_white():
    assert bp.LightSource().color == (1.0, 1.0, 1.0)


def test_per_channel_deposit_matches_color_ratio():
    rgb, _, _ = _cast((1.0, 0.6, 0.2))
    cy, cx = 15, 15
    r, g, b = rgb[cy, cx]
    assert r > g > b
    # Ratio of channels equals the source colour ratio (deposit is scalar*color).
    assert np.isclose(g / r, 0.6, atol=1e-4)
    assert np.isclose(b / r, 0.2, atol=1e-4)


def test_pack_layout_signed_direction_float16():
    rgb, dx, dy = _cast((1.0, 0.6, 0.2))
    h, w, _ = rgb.shape
    packed_a = np.zeros((h, w, 4), np.float16)
    packed_b = np.zeros((h, w, 4), np.float16)
    packed_a[..., 0:3] = rgb
    packed_a[..., 3] = dx
    packed_b[..., 0:3] = 0.0  # smoke_glow reserved this slice
    packed_b[..., 3] = dy

    assert packed_a.dtype == np.float16 and packed_b.dtype == np.float16
    # A tile to the +x side of the source: light arrives from -x, so the
    # stored (signed) direction.x must be negative — confirms we are NOT
    # using the old 0.5-centered encode.
    ty, tx = 15, 20
    assert float(packed_a[ty, tx, 3]) < 0.0
    # smoke_glow slot is zero this slice.
    assert np.all(packed_b[..., 0:3] == 0.0)
