"""Slice-4 ray outputs: the Q16.16 `heat` deposit + the RGB `smoke_glow`
god-ray glow (ch.03 §the march, ch.04 §Fixed-point format, ch.05 §God-rays).

Headless C++ tests on small synthetic gmaps (1-row grids force a pure +x march
so each downstream tile sees the ray exactly once — deterministic, no diagonal
aliasing). Verifies:
  - heat deposits as a POSITIVE int where the source emits heat, zero where it
    does not (src.heat == 0);
  - heat SATURATES (a huge deposit clamps at INT32_MAX, never wraps negative);
  - heat is the Q16.16 quantization of the aggregate light energy (scale check);
  - smoke_glow is non-zero only where smoke > 0 along a lit ray, and ~equals the
    light the smoke ABSORBED (energy-conserving);
  - a no-smoke scene leaves smoke_glow exactly zero.
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
INT32_MAX = 2**31 - 1


def _make_source(color=(1.0, 1.0, 1.0), heat=0.0, intensity=1.0, w=20):
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


def _cast(h=1, w=20, color=(1.0, 1.0, 1.0), heat_emit=0.0,
          smoke=None, atten=None, intensity=1.0,
          want_heat=True, want_glow=True, smoke_absorption=0.8):
    """Cast one +x beam; return (rgb, heat, smoke_glow)."""
    rc = bp.Raycaster()
    rc.smoke_absorption = smoke_absorption
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    if smoke is None:
        smoke = np.zeros((h, w), np.float32)
    if atten is None:
        atten = np.zeros((h, w, 3), np.float32)
    heat = np.zeros((h, w), np.int32) if want_heat else None
    glow = np.zeros((h, w, 3), np.float32) if want_glow else None
    s = _make_source(color=color, heat=heat_emit, intensity=intensity, w=w)
    rc.cast_source_directional(s, rgb, dx, dy, smoke, atten,
                               heat=heat, smoke_glow=glow)
    return rgb, heat, glow


# ---------------------------------------------------------------- heat deposit


def test_heat_zero_when_source_emits_no_heat():
    # src.heat == 0 -> a pure light source deposits NO heat anywhere.
    rgb, heat, _ = _cast(heat_emit=0.0)
    assert rgb.sum() > 0.0, "sanity: light did deposit"
    assert heat.dtype == np.int32
    assert np.all(heat == 0), f"heat leaked from a heat=0 source: {heat[heat != 0]}"


def test_heat_deposits_positive_int_where_source_emits():
    rgb, heat, _ = _cast(heat_emit=1.0)
    # Heat appears along the lit beam, as positive ints.
    lit = rgb.sum(axis=2) > 0.0
    assert np.all(heat[lit] > 0), "every lit tile should have positive heat"
    assert np.all(heat >= 0), "heat must never be negative (no wrap)"


def test_heat_equals_q16_16_of_aggregate_energy():
    # With heat_emit=1.0 the deposit is round(aggregate_light_energy * SCALE),
    # where aggregate = sum of the RGB deposit at that tile (the same energy the
    # light buffer saw). Check the source tile (dist_atten == 1).
    rgb, heat, _ = _cast(heat_emit=1.0, color=(1.0, 1.0, 1.0))
    agg = float(rgb[0, 0].sum())            # aggregate light energy at source tile
    expected = round(agg * HEAT_SCALE)
    assert heat[0, 0] == expected, f"heat {heat[0,0]} != Q16.16({agg}) = {expected}"


def test_heat_scales_with_emit_multiplier():
    _, heat1, _ = _cast(heat_emit=1.0)
    _, heat2, _ = _cast(heat_emit=2.0)
    # Twice the heat multiplier -> ~twice the deposit (quantization aside).
    assert heat2[0, 0] == 2 * heat1[0, 0] or abs(heat2[0, 0] - 2 * heat1[0, 0]) <= 1


def test_heat_saturates_does_not_wrap():
    # A huge heat multiplier would overflow int32 if added naively. The
    # saturating add must clamp at INT32_MAX and NEVER produce a negative value.
    rgb, heat, _ = _cast(heat_emit=1e12)
    assert np.all(heat >= 0), f"heat wrapped negative: min={heat.min()}"
    # The lit tiles should be pinned at the int32 ceiling, not garbage.
    lit = rgb.sum(axis=2) > 0.0
    assert np.all(heat[lit] == INT32_MAX), "saturated heat should clamp at INT32_MAX"


def test_heat_accumulates_across_two_sources_saturating():
    # Two near-max deposits into the same cell must clamp, not overflow.
    rc = bp.Raycaster()
    h, w = 1, 20
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    smoke = np.zeros((h, w), np.float32)
    atten = np.zeros((h, w, 3), np.float32)
    heat = np.zeros((h, w), np.int32)
    glow = np.zeros((h, w, 3), np.float32)
    s = _make_source(heat=1e9, w=w)
    rc.cast_source_directional(s, rgb, dx, dy, smoke, atten,
                               heat=heat, smoke_glow=glow)
    before = heat.copy()
    assert np.all(before >= 0)
    # Cast the same source AGAIN into the same buffer (accumulate).
    rc.cast_source_directional(s, rgb, dx, dy, smoke, atten,
                               heat=heat, smoke_glow=glow)
    assert np.all(heat >= 0), "second deposit must not wrap"
    assert np.all(heat >= before), "saturating add never decreases a cell"


# --------------------------------------------------------------- smoke_glow


def test_smoke_glow_zero_without_smoke():
    rgb, _, glow = _cast(heat_emit=0.0)  # no smoke field
    assert rgb.sum() > 0.0, "sanity: beam is lit"
    assert np.all(glow == 0.0), f"smoke_glow non-zero with no smoke: {glow[glow != 0]}"


def test_smoke_glow_only_where_smoke_present():
    h, w = 1, 20
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = 0.5
    rgb, _, glow = _cast(h=h, w=w, smoke=smoke)
    # Glow only at the smoke tile (and nowhere else).
    nz = np.where(glow.sum(axis=2) > 0.0)
    assert list(nz[1]) == [5], f"glow appeared off the smoke tile: cols {nz[1]}"


def test_smoke_glow_equals_absorbed_energy():
    # The glow deposited at a smoke tile == the light that tile's smoke removed
    # from the ray: dep[c] * (smoke_density * smoke_absorption). Reconstruct the
    # pre-absorption deposit `dep[c]` from the light buffer (which deposits
    # BEFORE attenuation), then check the glow matches the absorbed fraction.
    h, w = 1, 20
    sd, absorp = 0.5, 0.8
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = sd
    rgb, _, glow = _cast(h=h, w=w, smoke=smoke, smoke_absorption=absorp,
                         color=(1.0, 0.0, 0.0))  # red beam -> red shaft
    dep_r = rgb[0, 5, 0]                  # red light deposited at the smoke tile
    absorbed = dep_r * (sd * absorp)
    assert np.isclose(glow[0, 5, 0], absorbed, rtol=1e-4), (
        f"glow {glow[0,5,0]} != absorbed {absorbed}")
    # Red beam -> only the red channel of the shaft glows.
    assert glow[0, 5, 1] == 0.0 and glow[0, 5, 2] == 0.0, "non-red shaft leaked"


def test_smoke_glow_rgb_preserves_beam_colour():
    # A green beam through smoke casts a green shaft (RGB-preserving, ch.03 C16).
    h, w = 1, 20
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = 0.6
    _, _, glow = _cast(h=h, w=w, smoke=smoke, color=(0.0, 1.0, 0.0))
    g = glow[0, 5]
    assert g[1] > 0.0, "green shaft should glow"
    assert g[0] == 0.0 and g[2] == 0.0, "only green should glow for a green beam"
