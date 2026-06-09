"""Slice-4 ray outputs: the Q16.16 `heat` deposit + the RGB `smoke_glow`
god-ray glow (ch.03 §the march, ch.04 §Fixed-point format, ch.05 §God-rays).

Headless C++ tests on small synthetic gmaps (1-row grids force a pure +x march
so each downstream tile sees the ray exactly once — deterministic, no diagonal
aliasing). Verifies:
  - heat deposits as a POSITIVE int where the source emits heat, zero where it
    does not (src.heat == 0);
  - heat SATURATES (a huge deposit clamps at INT32_MAX, never wraps negative);
  - heat is the Q16.16 quantization of heat_emit * heat_survival * falloff — the
    INDEPENDENT 4th channel, decoupled from the RGB light energy (engine/06 §1);
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
          want_heat=True, want_glow=True,
          absorption_rgb=(1.0, 1.0, 1.0),
          scatter_albedo=(1.0, 1.0, 1.0),
          absorb_scale=1.4):
    """Cast one +x beam; return (rgb, heat, smoke_glow).

    Smoke optics are the decoupled per-channel model (ch.05 §6.1 §6):
      transmission  trans_c = exp(-absorption_rgb[c] * density * absorb_scale)
      scatter/glow  smoke_glow[c] += local_light[c] * scatter_albedo[c] * density
    """
    rc = bp.Raycaster()
    rc.smoke_absorb_scale = absorb_scale
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    if smoke is None:
        smoke = np.zeros((h, w), np.float32)
    if atten is None:
        atten = np.zeros((h, w, 3), np.float32)
    heat = np.zeros((h, w), np.int32) if want_heat else None
    glow = np.zeros((h, w, 3), np.float32) if want_glow else None
    # Multi-gas march (engine/05 §6.2): drive the single `smoke` field as ONE gas
    # whose per-channel absorption/scatter rows ARE the old scalar coefficients.
    # A single populated gas reproduces exactly what the old single-smoke path
    # did for those coefficients, so every assertion below is preserved.
    gas = smoke[np.newaxis, :, :].astype(np.float32)
    gas_absorption = np.array([absorption_rgb], np.float32)
    gas_scatter = np.array([scatter_albedo], np.float32)
    s = _make_source(color=color, heat=heat_emit, intensity=intensity, w=w)
    rc.cast_source_directional(s, rgb, dx, dy,
                               gas, gas_absorption, gas_scatter, atten,
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


def test_heat_equals_q16_16_of_emit_independent_of_colour():
    # Heat is the INDEPENDENT 4th channel (engine/06 §1): the deposit is
    # heat_emit * heat_survival * dist_atten, NOT the RGB aggregate. At the source
    # tile (heat_survival == 1, dist_atten == 1) it is exactly round(emit * SCALE)
    # regardless of the source colour — a dim colour no longer dims the heat.
    for color in [(1.0, 1.0, 1.0), (1.0, 0.0, 0.0), (0.2, 0.2, 0.2)]:
        _, heat, _ = _cast(heat_emit=1.0, color=color)
        expected = round(1.0 * HEAT_SCALE)
        assert heat[0, 0] == expected, (
            f"heat {heat[0,0]} != Q16.16(emit=1.0) = {expected} for colour {color} "
            f"(heat must be decoupled from RGB)")


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
    gas = np.zeros((1, h, w), np.float32)
    gas_absorption = np.ones((1, 3), np.float32)
    gas_scatter = np.ones((1, 3), np.float32)
    atten = np.zeros((h, w, 3), np.float32)
    heat = np.zeros((h, w), np.int32)
    glow = np.zeros((h, w, 3), np.float32)
    s = _make_source(heat=1e9, w=w)
    rc.cast_source_directional(s, rgb, dx, dy,
                               gas, gas_absorption, gas_scatter, atten,
                               heat=heat, smoke_glow=glow)
    before = heat.copy()
    assert np.all(before >= 0)
    # Cast the same source AGAIN into the same buffer (accumulate).
    rc.cast_source_directional(s, rgb, dx, dy,
                               gas, gas_absorption, gas_scatter, atten,
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


def test_smoke_glow_equals_scatter_deposit():
    # Decoupled scatter model (ch.05 §6.1 6b): the glow deposited at a smoke tile
    # == local_light[c] * scatter_albedo[c] * density. It is a SEPARATE additive
    # budget, NOT the absorbed amount. Reconstruct the deposit `dep[c]` from the
    # light buffer (which deposits BEFORE attenuation) and check the scatter.
    h, w = 1, 20
    sd, albedo = 0.5, 0.7
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = sd
    rgb, _, glow = _cast(h=h, w=w, smoke=smoke,
                         scatter_albedo=(albedo, 0.0, 0.0),
                         color=(1.0, 0.0, 0.0))  # red beam -> red shaft
    dep_r = rgb[0, 5, 0]                  # red light deposited at the smoke tile
    expected = dep_r * albedo * sd
    assert np.isclose(glow[0, 5, 0], expected, rtol=1e-4), (
        f"glow {glow[0,5,0]} != scatter {expected}")
    # Red beam -> only the red channel of the shaft glows.
    assert glow[0, 5, 1] == 0.0 and glow[0, 5, 2] == 0.0, "non-red shaft leaked"


# ---------------------------------------------- decoupled per-channel optics


def test_transmission_follows_beer_lambert():
    # The surviving beam past a smoke tile follows exp(-absorption*density*scale).
    # Compare the light deposited at the tile AFTER the smoke tile (col 6) for a
    # neutral white beam, against the analytic transmission of the col-5 tile.
    h, w = 1, 20
    sd, absorp, scale = 0.5, 1.0, 1.4
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = sd
    rgb, _, _ = _cast(h=h, w=w, smoke=smoke,
                      absorption_rgb=(absorp, absorp, absorp),
                      absorb_scale=scale)
    # dep at col5 is pre-attenuation; dep at col6 is col5's deposit * trans *
    # (dist falloff ratio). Easiest invariant: the RATIO of surviving energy
    # across the smoke tile equals exp(-tau) up to the distance-falloff factor.
    # Cross-check directly against a no-smoke control (same geometry, no smoke):
    rgb0, _, _ = _cast(h=h, w=w, absorption_rgb=(absorp, absorp, absorp),
                       absorb_scale=scale)
    trans = float(np.exp(-absorp * sd * scale))
    # col6 with smoke / col6 without smoke == transmission of the col5 tile.
    ratio = rgb[0, 6, 0] / rgb0[0, 6, 0]
    assert np.isclose(ratio, trans, rtol=1e-3), (
        f"transmission ratio {ratio} != exp(-tau) {trans}")
    # exp(-tau) NEVER reaches zero -> the beam survives deep smoke.
    assert rgb[0, 6, 0] > 0.0, "beam must survive a smoke tile (exp never hits 0)"


def test_higher_absorption_channel_is_dimmed_more():
    # Per-channel: a channel with higher absorption is attenuated more. Blue
    # absorbed hardest -> least blue survives past the smoke tile.
    h, w = 1, 20
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = 0.6
    rgb, _, _ = _cast(h=h, w=w, smoke=smoke, color=(1.0, 1.0, 1.0),
                      absorption_rgb=(0.2, 0.5, 1.0), absorb_scale=1.4)
    r, g, b = rgb[0, 6, 0], rgb[0, 6, 1], rgb[0, 6, 2]
    assert r > g > b > 0.0, (
        f"higher absorption must dim more: r={r} g={g} b={b}")


def test_lower_absorb_scale_increases_beam_reach():
    # absorb_scale is the beam-reach dial: LOWER -> more light survives (longer
    # beam). Same smoke, two scales; the low scale transmits strictly more.
    h, w = 1, 30
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = 0.7
    rgb_far, _, _ = _cast(h=h, w=w, smoke=smoke, absorb_scale=0.3)
    rgb_near, _, _ = _cast(h=h, w=w, smoke=smoke, absorb_scale=3.0)
    # Light deposited well past the smoke tile (col 15).
    assert rgb_far[0, 15, 0] > rgb_near[0, 15, 0], (
        "lower absorb_scale must transmit MORE light (longer beam reach)")


def test_scatter_is_independent_of_absorption():
    # Decoupling: glow depends ONLY on scatter_albedo, not on absorption. Two
    # casts with very different absorption but the same scatter_albedo deposit
    # the same glow at the smoke tile (the glow is the SEPARATE additive budget,
    # not the absorbed amount). This is the "barely absorbs, glows brightly"
    # property (steam).
    h, w = 1, 20
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 3] = 0.5
    glow_lo = _cast(h=h, w=w, smoke=smoke, absorb_scale=0.1,
                    scatter_albedo=(1.0, 1.0, 1.0))[2]
    glow_hi = _cast(h=h, w=w, smoke=smoke, absorb_scale=5.0,
                    scatter_albedo=(1.0, 1.0, 1.0))[2]
    # The deposit at the smoke tile happens BEFORE attenuation, so the local
    # light (and thus the scatter) is identical regardless of absorb_scale.
    assert np.isclose(glow_lo[0, 3, 0], glow_hi[0, 3, 0], rtol=1e-5), (
        f"scatter must not depend on absorption: {glow_lo[0,3,0]} vs {glow_hi[0,3,0]}")
    # And glow can EXCEED what absorption alone removes: with tiny absorption,
    # albedo>0 still produces a bright deposit.
    assert glow_lo[0, 3, 0] > 0.0, "glow present even when absorption is tiny"


def test_smoke_glow_rgb_preserves_beam_colour():
    # A green beam through smoke casts a green shaft (RGB-preserving, ch.03 C16).
    h, w = 1, 20
    smoke = np.zeros((h, w), np.float32)
    smoke[0, 5] = 0.6
    _, _, glow = _cast(h=h, w=w, smoke=smoke, color=(0.0, 1.0, 0.0))
    g = glow[0, 5]
    assert g[1] > 0.0, "green shaft should glow"
    assert g[0] == 0.0 and g[2] == 0.0, "only green should glow for a green beam"
