"""Unit tests for the W2b water-depth debug overlay (renderer/overlays.py).

``WaterFieldOverlay`` is the render-time view of ``gmap.water_depth`` (blue
tint scaled by depth, toggled with O). These tests verify the packing
contract WITHOUT a window: the only GL touchpoints (texture create/upload in
``renderer.core``) are monkeypatched away, so the overlay packs into its
CPU-side ``packed`` buffer headlessly.

Covered:
  - zero depth -> fully transparent (alpha AND premultiplied RGB all zero);
  - alpha scales monotonically with depth; display max -> max_alpha;
  - depths above ``depth_display_max`` clamp (no wrap/overflow);
  - the tint is blue-dominant and premultiplied (channel <= tint * a/255);
  - render-only: ``update`` never mutates the depth field it is handed.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import renderer.core as rcore  # noqa: E402
from renderer.overlays import WaterFieldOverlay  # noqa: E402


@pytest.fixture
def overlay(monkeypatch):
    """A 4x4 WaterFieldOverlay with the GL calls stubbed out (headless)."""
    monkeypatch.setattr(rcore, "create_dynamic_rgba_texture",
                        lambda w, h: None)
    monkeypatch.setattr(rcore, "update_rgba_texture",
                        lambda tex, packed: None)
    return WaterFieldOverlay(4, 4, depth_display_max=1.0)


def test_zero_depth_is_fully_transparent(overlay):
    overlay.update(np.zeros((4, 4), dtype=np.float32))
    # Premultiplied pack: alpha 0 forces RGB to 0 too — the whole texel is 0.
    assert overlay.packed.max() == 0


def test_alpha_scales_with_depth(overlay):
    depth = np.zeros((4, 4), dtype=np.float32)
    depth[0, 0], depth[0, 1], depth[0, 2] = 0.1, 0.5, 1.0
    overlay.update(depth)
    a = overlay.packed[..., 3].astype(np.int32)
    assert 0 < a[0, 0] < a[0, 1] < a[0, 2], "alpha must rise with depth"
    assert a[0, 2] == overlay.max_alpha, "display max must hit max_alpha"
    assert a[1, 1] == 0, "dry tile must stay transparent"


def test_depth_above_display_max_clamps(overlay):
    # 2.5 m == the U-pour ceiling; display max is 1.0 m -> clamp, no overflow.
    overlay.update(np.full((4, 4), 2.5, dtype=np.float32))
    assert (overlay.packed[..., 3] == overlay.max_alpha).all()


def test_tint_is_premultiplied_blue(overlay):
    overlay.update(np.full((4, 4), 0.5, dtype=np.float32))
    r, g, b, a = (overlay.packed[..., i].astype(np.float64) for i in range(4))
    assert (b > r).all() and (b > g).all(), "water must read blue"
    # Premultiplied alpha: each channel <= tint * (alpha/255), up to uint8
    # truncation slack — required for the BLEND_ALPHA_PREMULTIPLY draw.
    for ch, tint in ((r, overlay.tint_r), (g, overlay.tint_g),
                     (b, overlay.tint_b)):
        assert (ch <= tint * a / 255.0 + 1.0).all()


def test_update_never_mutates_the_field(overlay):
    depth = np.full((4, 4), 0.3, dtype=np.float32)
    before = depth.copy()
    overlay.update(depth)
    assert np.array_equal(depth, before), "render-only contract violated"
