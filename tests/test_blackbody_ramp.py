"""Unit tests for renderer/blackbody.py — the B1 temperature->colour primitive.

Headless / pure-numpy: imports the module in ISOLATION (importlib from file)
so it never touches ``renderer/__init__.py`` -> pyray (the raylib pytest
tripwire). No GL, no breach_physics. Covers the design's P1 gate
(docs/fire_b1_blackbody_fire_lights_design_2026-07-21.md §5):
  - LUT endpoints red -> white,
  - chroma monotonically desaturating with K,
  - intensity monotone in T,
  - vectorized chroma_intensity == scalar light_color on samples.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

# Load renderer/blackbody.py directly, bypassing the renderer package __init__
# (which imports pyray and would fail / open a window headless).
_BB_PATH = Path(__file__).resolve().parent.parent / "renderer" / "blackbody.py"
_spec = importlib.util.spec_from_file_location("_blackbody_isolated", _BB_PATH)
blackbody = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(blackbody)

BlackbodyRamp = blackbody.BlackbodyRamp
TEMP_SCALE = blackbody.TEMP_SCALE


def _saturation(rgb):
    """HSV-style saturation of a normalized-chroma row: (max - min) / max."""
    mx = np.max(rgb, axis=-1)
    mn = np.min(rgb, axis=-1)
    return np.where(mx > 1e-6, (mx - mn) / mx, 0.0)


# ---- construction / validation -----------------------------------------

def test_lut_shapes_and_defaults():
    r = BlackbodyRamp()
    assert r._chroma_lut.shape == (256, 3)
    assert r._inten_lut.shape == (256,)
    assert r._chroma_lut.dtype == np.float32


def test_invalid_params_raise():
    for kw in (dict(lut_size=1), dict(kelvin_floor=5000, kelvin_ceil=800),
               dict(kelvin_glow_min=3000, kelvin_ref=1000)):
        try:
            BlackbodyRamp(**kw)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kw}")


# ---- LUT endpoints: red -> white ---------------------------------------

def test_endpoints_red_to_white():
    r = BlackbodyRamp()
    lo = r._chroma_lut[0]    # ~800 K
    hi = r._chroma_lut[-1]   # ~10000 K
    # Normalized chroma: max channel is 1 everywhere.
    assert np.isclose(lo.max(), 1.0)
    assert np.isclose(hi.max(), 1.0)
    # Cool end (800 K) is red-dominant: red is the peak, blue is near zero.
    assert lo[0] >= lo[2]
    assert lo[2] < 0.15
    # Hot end (10000 K) is desaturated toward white: all channels high.
    assert _saturation(hi) < _saturation(lo)
    assert hi.min() > 0.5


# ---- chroma desaturates monotonically with K ---------------------------

def test_chroma_desaturates_with_temperature():
    r = BlackbodyRamp()
    # Physical claim (research Q1): blackbody chroma desaturates monotonically
    # toward white as T rises through the warm range. Above ~6600 K it tips
    # slightly blue-white, so check monotonicity over the design-relevant warm
    # span (floor -> the LUT entry nearest 6500 K).
    hot_idx = int(np.argmin(np.abs(r._kelvins - 6500.0)))
    sat = _saturation(r._chroma_lut[: hot_idx + 1])
    diffs = np.diff(sat)
    assert np.all(diffs <= 1e-6), f"saturation not non-increasing: max +{diffs.max():.2e}"
    # And it genuinely falls a lot, not a flat line.
    assert sat[0] - sat[-1] > 0.5


# ---- intensity monotone in T, floored below glow_min, capped -----------

def test_intensity_monotone_and_bounded():
    r = BlackbodyRamp()
    inten = r._inten_lut
    assert np.all(np.diff(inten) >= -1e-6), "intensity must be non-decreasing in K"
    assert inten.max() <= r.intensity_max + 1e-6
    assert inten.min() >= 0.0
    # At the reference temperature the ramp passes through ~1.0 (unit exposure).
    (_rgb, i_ref) = r.light_color((r.kelvin_ref - r.kelvin_ambient)
                                  / r.k_temp_to_kelvin)
    assert np.isclose(i_ref, 1.0, atol=0.05)


def test_cold_tile_has_no_glow():
    r = BlackbodyRamp()
    # T_game = 0 -> kelvin = ambient (293) < glow_min (800) -> zero intensity.
    _rgb, i = r.light_color(0.0)
    assert i == 0.0
    # A whole cold field: intensity all zero.
    field = np.zeros((8, 8), dtype=np.int32)
    _chroma, inten = r.chroma_intensity(field)
    assert np.all(inten == 0.0)


def test_hot_extreme_saturates_intensity():
    r = BlackbodyRamp()
    # T_MAX_PHYS = 16000 game units -> kelvin ~= 32293, well past kelvin_ref;
    # intensity clamps to intensity_max.
    _rgb, i = r.light_color(16000.0)
    assert np.isclose(i, r.intensity_max)


# ---- the load-bearing invariant: vectorized == scalar, bit-identical ---

def test_vectorized_equals_scalar():
    r = BlackbodyRamp()
    # Spread samples across cold, warm wood-fire, and extreme regimes. Choose
    # integer game-units so quantize/dequantize is exact in Q16.16.
    samples = [0, 50, 150, 300, 600, 1200, 3000, 5000, 9000, 16000]
    field = np.array(samples, dtype=np.int64).reshape(2, 5) * int(TEMP_SCALE)
    field = field.astype(np.int32)
    chroma, inten = r.chroma_intensity(field)
    for i, tg in enumerate(samples):
        row, col = divmod(i, 5)
        # Feed light_color the exact dequantized value chroma_intensity sees.
        t_game = float(np.float64(field[row, col]) / TEMP_SCALE)
        (lr, lg, lb), li = r.light_color(t_game)
        assert (lr, lg, lb) == tuple(chroma[row, col]), f"chroma mismatch at T={tg}"
        assert li == inten[row, col], f"intensity mismatch at T={tg}"


# ---- render-only: never mutate the field -------------------------------

def test_field_not_mutated():
    r = BlackbodyRamp()
    field = (np.arange(64, dtype=np.int32).reshape(8, 8) * 5000)
    before = field.copy()
    r.chroma_intensity(field)
    assert np.array_equal(field, before)


# ---- config binding -----------------------------------------------------

def test_from_config_reads_section():
    class _NS:
        pass
    bb = _NS()
    bb.k_temp_to_kelvin = 3.0
    bb.intensity_max = 4.0
    render = _NS()
    render.blackbody = bb
    cfg = _NS()
    cfg.render = render
    r = BlackbodyRamp.from_config(cfg)
    assert r.k_temp_to_kelvin == 3.0
    assert r.intensity_max == 4.0
    # Unspecified keys fall back to the design defaults.
    assert r.kelvin_ref == 3000.0


def test_from_config_missing_section_uses_defaults():
    class _NS:
        pass
    r = BlackbodyRamp.from_config(_NS())
    assert r.k_temp_to_kelvin == 2.0
    assert r.lut_size == 256
