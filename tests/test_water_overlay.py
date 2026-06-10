"""Unit tests for the water overlay (renderer/overlays.py) — W2b tint + W6b v2.

``WaterFieldOverlay`` is the render-time view of the sim's water fields
(blue depth tint + ripple shading + foam + ambient sines, toggled with O).
These tests verify the packing contract WITHOUT a window: the only GL
touchpoints (texture create/upload in ``renderer.core``) are monkeypatched
away, so the overlay packs into its CPU-side ``packed`` buffer headlessly.

Covered (W2b legacy path — ``update(depth)`` with no ripple):
  - zero depth -> fully transparent (alpha AND premultiplied RGB all zero);
  - alpha scales monotonically with depth; display max -> max_alpha;
  - depths above ``depth_display_max`` clamp (no wrap/overflow);
  - the tint is blue-dominant and premultiplied (channel <= tint * a/255);
  - render-only: ``update`` never mutates the depth field it is handed.

Covered (W6b v2 path — ripple/ripple_v/flow/t threaded in):
  - foam whitens + lifts alpha above ``foam_thresh`` and not below;
  - ripple shading changes texels vs a flat ripple (non-vacuity) and is
    directional (positive d/dx slope brightens, negative darkens);
  - ambient sines are nonzero on standing water, animate with t, and their
    amplitude grows with local ripple energy (|ripple| + |ripple_v|);
  - premultiplied-valid under the wild combo (every channel <= alpha);
  - render-only for ALL five input fields; zero water -> zero texels even
    with junk ripple fields; zero-water fast path skips the GPU upload once
    the texture is clear (and never before — it starts opaque black).
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


# ---------------------------------------------------------------------------
# W6b v2: ripple shading + foam + ambient sines
# ---------------------------------------------------------------------------

N = 16  # v2 tests use a bigger grid so half-grid means average the sines out


@pytest.fixture
def overlay16(monkeypatch):
    """A 16x16 v2 overlay, GL stubbed, shipped knob defaults pinned."""
    monkeypatch.setattr(rcore, "create_dynamic_rgba_texture",
                        lambda w, h: None)
    monkeypatch.setattr(rcore, "update_rgba_texture",
                        lambda tex, packed: None)
    return WaterFieldOverlay(N, N, depth_display_max=1.0,
                             ripple_shade=0.35, foam_thresh=0.03,
                             ambient_base=0.06)


def _flat_depth(value: float = 0.5) -> np.ndarray:
    return np.full((N, N), value, dtype=np.float32)


def _slope_plane(slope: float) -> np.ndarray:
    """ripple[y, x] = slope * x — d/dx == slope exactly, everywhere
    (np.gradient is exact on a linear ramp, edges included)."""
    xs = np.arange(N, dtype=np.float32)[None, :]
    return np.broadcast_to(slope * xs, (N, N)).astype(np.float32)


def _ridge(slope: float) -> np.ndarray:
    """A tent ridge peaking mid-grid: d/dx == +slope on the left half,
    -slope on the right half."""
    xs = np.arange(N, dtype=np.float32)[None, :]
    return np.broadcast_to(
        slope * np.minimum(xs, (N - 1) - xs), (N, N)).astype(np.float32)


_Z = np.zeros((N, N), dtype=np.float32)


def test_foam_above_thresh_not_below(overlay16):
    """foam_thresh = 0.03 m/tile: a settled-ish slope (0.015) never foams; a
    fresh-splash slope (0.075, the plan's 0.05 m over ~2 tiles) fully foams
    (blend to white + alpha lift toward opaque froth)."""
    depth = _flat_depth(0.5)   # alpha = sqrt(0.5)*200 = 141.42 -> uint8 141

    overlay16.update(depth, ripple=_slope_plane(0.015), ripple_v=_Z)
    r, b = (overlay16.packed[..., 0].astype(np.float64),
            overlay16.packed[..., 2].astype(np.float64))
    a = overlay16.packed[..., 3]
    assert (a == 141).all(), "no foam -> alpha is the plain depth ramp"
    assert (b > 0).all()
    assert (r / b < 0.30).all(), \
        "below thresh the tint must stay blue-dominant (no whitening)"

    overlay16.update(depth, ripple=_slope_plane(0.075), ripple_v=_Z)
    r, b = (overlay16.packed[..., 0].astype(np.float64),
            overlay16.packed[..., 2].astype(np.float64))
    a = overlay16.packed[..., 3]
    assert (a > 160).all(), "full foam must lift alpha toward opaque froth"
    assert (r / b > 0.95).all(), "full foam must read white (r ~ b)"


def test_ripple_shading_changes_texels(overlay16):
    """Non-vacuity: a sloped ripple must change the packed texels vs a flat
    one (same depth, same t)."""
    depth = _flat_depth(0.5)
    overlay16.update(depth, ripple=_Z, ripple_v=_Z, t=0.0)
    flat = overlay16.packed.copy()
    overlay16.update(depth, ripple=_ridge(0.01), ripple_v=_Z, t=0.0)
    assert not np.array_equal(overlay16.packed, flat), \
        "ripple shading had no effect on the texels"


def test_ripple_shading_is_directional(monkeypatch):
    """Positive d(ripple)/dx brightens, negative darkens. Ambient is zeroed
    (base knob + energy gain) to isolate the shading term; slope 0.01 stays
    below both the foam thresh (0.03) and the 255-clip."""
    monkeypatch.setattr(rcore, "create_dynamic_rgba_texture",
                        lambda w, h: None)
    monkeypatch.setattr(rcore, "update_rgba_texture",
                        lambda tex, packed: None)
    ov = WaterFieldOverlay(N, N, ambient_base=0.0)
    ov._AMB_ENERGY_GAIN = 0.0   # instance shadow: kill the energy term too
    ov.update(_flat_depth(0.5), ripple=_ridge(0.01), ripple_v=_Z)
    b = ov.packed[..., 2].astype(np.float64)
    left = b[:, :N // 2 - 1].mean()    # +slope side
    right = b[:, N // 2 + 1:].mean()   # -slope side
    assert left > right + 10, \
        f"shading not directional: left {left:.1f} vs right {right:.1f}"


def test_ambient_nonzero_and_animates_on_standing_water(overlay16):
    """Dead-calm pool (zero ripple AND ripple_v): the ambient sines alone must
    texture the surface (spatial variation) and animate with t."""
    depth = _flat_depth(0.5)
    overlay16.update(depth, ripple=_Z, ripple_v=_Z, t=0.0)
    b0 = overlay16.packed[..., 2].copy()
    # With shade == foam == 0 and flat depth, ONLY the ambient sines can
    # make texels differ from one another.
    assert b0.std() > 0, "ambient term is dead on standing water"
    overlay16.update(depth, ripple=_Z, ripple_v=_Z, t=1.7)
    assert not np.array_equal(overlay16.packed[..., 2], b0), \
        "ambient sines do not animate with t"


def test_ambient_amplitude_grows_with_ripple_energy(overlay16):
    """Canon §6 modulation rule: amplitude = base + local ripple energy. An
    agitated pool (|ripple_v| = 2 m/s) must shimmer harder than a calm one."""
    depth = _flat_depth(0.5)
    overlay16.update(depth, ripple=_Z, ripple_v=_Z, t=0.4)
    std_calm = overlay16.packed[..., 2].astype(np.float64).std()
    overlay16.update(depth, ripple=_Z,
                     ripple_v=np.full((N, N), 2.0, dtype=np.float32), t=0.4)
    std_agitated = overlay16.packed[..., 2].astype(np.float64).std()
    assert std_agitated > 2.0 * std_calm, \
        f"energy must boost the shimmer: {std_agitated:.2f} vs {std_calm:.2f}"


def test_v2_premultiplied_valid_under_wild_combo(overlay16):
    """Foam + hard shading + saturated ambient at once: every RGB channel
    must stay <= alpha per texel (BLEND_ALPHA_PREMULTIPLY validity)."""
    depth = _flat_depth(0.5)
    depth[:, 3] = 0.0                      # a dry stripe (wet/dry edges)
    depth[:, 9] = 2.5                      # over display max
    ripple = _ridge(0.08)                  # steep: foams AND shades hard
    rv = np.full((N, N), 3.0, dtype=np.float32)   # ambient amp at the cap
    flow = np.full((N, N), 1.0, dtype=np.float32)  # front foam triggers
    overlay16.update(depth, ripple=ripple, ripple_v=rv,
                     flow_vx=flow, flow_vy=flow, t=5.0)
    a = overlay16.packed[..., 3]
    for ch in range(3):
        assert (overlay16.packed[..., ch] <= a).all(), \
            f"channel {ch} exceeds alpha — premultiplied contract broken"
    assert (overlay16.packed[:, 3, :] == 0).all(), \
        "dry stripe must stay fully transparent"


def test_v2_render_only_all_fields(overlay16):
    """The v2 update must not mutate ANY of the five input fields."""
    depth = _flat_depth(0.4)
    ripple = _ridge(0.05)
    rv = np.full((N, N), 0.7, dtype=np.float32)
    fvx = np.full((N, N), 0.3, dtype=np.float32)
    fvy = np.full((N, N), -0.2, dtype=np.float32)
    keep = [x.copy() for x in (depth, ripple, rv, fvx, fvy)]
    overlay16.update(depth, ripple=ripple, ripple_v=rv,
                     flow_vx=fvx, flow_vy=fvy, t=1.0)
    for arr, before in zip((depth, ripple, rv, fvx, fvy), keep):
        assert np.array_equal(arr, before), "render-only contract violated"


def test_v2_zero_water_zero_texels(overlay16):
    """Zero depth + junk ripple/flow fields -> every texel stays 0 (ambient,
    shading and foam are all masked to wet tiles)."""
    junk = _slope_plane(0.075)             # would fully foam if wet
    ones = np.ones((N, N), dtype=np.float32)
    overlay16.update(np.zeros((N, N), dtype=np.float32),
                     ripple=junk, ripple_v=ones,
                     flow_vx=ones, flow_vy=ones, t=2.0)
    assert overlay16.packed.max() == 0


def test_zero_water_skips_upload_once_texture_clear(monkeypatch):
    """Performance guard: a dry ship skips the pack+upload — but only AFTER
    one clearing upload (the GPU texture starts opaque black)."""
    uploads = []
    monkeypatch.setattr(rcore, "create_dynamic_rgba_texture",
                        lambda w, h: None)
    monkeypatch.setattr(rcore, "update_rgba_texture",
                        lambda tex, packed: uploads.append(1))
    ov = WaterFieldOverlay(8, 8)
    zeros = np.zeros((8, 8), dtype=np.float32)
    ov.update(zeros, ripple=zeros, ripple_v=zeros)
    assert len(uploads) == 1, "first dry update must clear the black texture"
    ov.update(zeros, ripple=zeros, ripple_v=zeros)
    assert len(uploads) == 1, "second dry update must be skipped"
    wet = zeros.copy()
    wet[4, 4] = 0.3
    ov.update(wet, ripple=zeros, ripple_v=zeros)
    assert len(uploads) == 2, "water arriving must re-upload"
    ov.update(zeros)            # drained — legacy single-arg path
    assert len(uploads) == 3, "draining needs one clearing upload"
    ov.update(zeros)
    assert len(uploads) == 3, "dry again -> skipped on the legacy path too"
