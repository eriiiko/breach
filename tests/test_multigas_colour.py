"""Multi-gas coloured ray march (engine/05 §6.2, M2).

The directional ray march now sums ALL gases density-weighted per channel,
each with its own per-channel `absorption` / `scatter_albedo` row from
`GasTable`:

    transmission:  tau_c = absorb_scale * Σ_g ( gas[g][tile] * absorption[g][c] )
                   trans_c = exp(-tau_c);  remaining[c] *= trans_c
    scatter/glow:  smoke_glow[c] += dep_c * Σ_g ( gas[g][tile] * scatter_albedo[g][c] )

This file pins the colour model on small headless gmaps (1-row grids force a
pure +x march so each downrange tile sees the ray exactly once — deterministic,
no diagonal aliasing), mirroring test_heat_smoke_glow.py. It verifies, against
the CANON gas table in config.toml:

  * poison tints the beam GREEN downrange (G survives > R > B);
  * smoke dims strongly and ~neutrally (R≈G≈B, all far below poison's G);
  * steam barely dims but its smoke_glow brightens (scatter-dominated);
  * mixing falls out of the SUM — poison+black in the same tile attenuates as
    the product of the two transmissions (== summing their tau), per channel;
  * a single populated gas reproduces the documented single-smoke Beer-Lambert
    path for that gas's coefficients;
  * determinism (bit-identical across casts).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402
from simulation.gases import (  # noqa: E402
    GasTable, N_GASES,
    STEAM, SMOKE, POISON,
)


# ----------------------------------------------------------------- canon table


def _canon_table():
    """Load the real per-gas absorption/scatter tables from config.toml.

    Loaded headless via tomllib (no pyray / full config machinery) so the test
    asserts against the SAME research-approved coefficients the game runs with.
    """
    import tomllib
    cfg = tomllib.load(open(ROOT / "config.toml", "rb"))
    gt = GasTable(cfg["gases"])
    return gt.absorption.astype(np.float32), gt.scatter_albedo.astype(np.float32)


# --------------------------------------------------------------------- casting


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


def _cast(gas, gas_absorption, gas_scatter, *, color=(1.0, 1.0, 1.0),
          absorb_scale=1.0, w=None, want_glow=True):
    """Cast one +x white-by-default beam through `gas` (N,h,w); return (rgb, glow).

    `gas_absorption` / `gas_scatter` are (N,3) per-gas per-channel tables.
    """
    n, h, w_ = gas.shape
    if w is None:
        w = w_
    rc = bp.Raycaster()
    rc.smoke_absorb_scale = absorb_scale
    rgb = np.zeros((h, w, 3), np.float32)
    dx = np.zeros((h, w), np.float32)
    dy = np.zeros((h, w), np.float32)
    glow = np.zeros((h, w, 3), np.float32) if want_glow else None
    s = _make_source(color=color, w=w)
    rc.cast_source_directional(s, rgb, dx, dy,
                               np.ascontiguousarray(gas, np.float32),
                               np.ascontiguousarray(gas_absorption, np.float32),
                               np.ascontiguousarray(gas_scatter, np.float32),
                               np.zeros((h, w, 3), np.float32),
                               smoke_glow=glow)
    return rgb, glow


def _single_gas_field(gas_id, n, col, density, h=1, w=20):
    """An (n,h,w) gas array that is `density` for `gas_id` at `col`, else 0."""
    gas = np.zeros((n, h, w), np.float32)
    gas[gas_id, 0, col] = density
    return gas


# ------------------------------------------------------------------ poison green


def test_poison_tints_beam_green_downrange():
    # White beam through ONE poison tile (canon absorption [0.45,0.10,0.80]):
    # G is absorbed least -> most green survives; B absorbed hardest -> least.
    absorp, scatter = _canon_table()
    w = 20
    smoke_col = 5
    gas = _single_gas_field(POISON, N_GASES, smoke_col, 0.6, w=w)
    rgb, _ = _cast(gas, absorp, scatter, absorb_scale=1.4, w=w)
    past = smoke_col + 1
    r, g, b = rgb[0, past, 0], rgb[0, past, 1], rgb[0, past, 2]
    # Yellow-green: green transmits most, then red, then blue.
    assert g > r > b > 0.0, (
        f"poison should green the beam (G>R>B), got r={r} g={g} b={b}")
    # The beam SURVIVES (exp(-tau) never hits 0) — coloured, not killed.
    assert g > 0.0


# --------------------------------------------------------------- black neutral


def test_black_smoke_dims_strongly_and_neutrally():
    # Black smoke (absorption [0.88,0.90,0.93]) dims hard and ~neutrally: the
    # three surviving channels stay close to each other (no strong hue) and all
    # sit far below poison's green-survival for the same density.
    absorp, scatter = _canon_table()
    w = 20
    smoke_col = 5
    gas = _single_gas_field(SMOKE, N_GASES, smoke_col, 0.6, w=w)
    rgb, _ = _cast(gas, absorp, scatter, absorb_scale=1.4, w=w)
    past = smoke_col + 1
    r, g, b = rgb[0, past, 0], rgb[0, past, 1], rgb[0, past, 2]
    ctrl, _ = _cast(np.zeros((N_GASES, 1, w), np.float32), absorp, scatter,
                    absorb_scale=1.4, w=w)
    # Strong dimming: the survivor is a small fraction of the no-smoke control.
    assert rgb[0, past].sum() < 0.5 * ctrl[0, past].sum(), "black smoke must dim strongly"
    # ~Neutral: channels within ~12% of one another (slight blue tilt is fine).
    mx, mn = max(r, g, b), min(r, g, b)
    assert (mx - mn) / mx < 0.12, f"black smoke should be ~neutral, got r={r} g={g} b={b}"
    # And it dims FAR more than poison greens: black's brightest survivor is
    # below poison's green survivor at the same density.
    poison_gas = _single_gas_field(POISON, N_GASES, smoke_col, 0.6, w=w)
    rgb_p, _ = _cast(poison_gas, absorp, scatter, absorb_scale=1.4, w=w)
    assert mx < rgb_p[0, past, 1], "black smoke survivor should be below poison's green"


# ------------------------------------------------------- white scatter-dominated


def test_white_smoke_barely_dims_but_glows():
    # White smoke: absorption [0.10,...] (tiny) but scatter_albedo [0.92,...]
    # (large). Transmission stays HIGH (beam barely dims) while smoke_glow is
    # BRIGHT — the scatter-dominated steam signature (engine/05 §6.2).
    absorp, scatter = _canon_table()
    w = 20
    smoke_col = 5
    sd = 0.6
    gas = _single_gas_field(STEAM, N_GASES, smoke_col, sd, w=w)
    rgb, glow = _cast(gas, absorp, scatter, absorb_scale=1.4, w=w)
    ctrl, _ = _cast(np.zeros((N_GASES, 1, w), np.float32), absorp, scatter,
                    absorb_scale=1.4, w=w)
    past = smoke_col + 1
    # Barely dims: >80% of the control light still reaches downrange.
    assert rgb[0, past].sum() > 0.8 * ctrl[0, past].sum(), "white smoke must barely dim"
    # But the glow is bright: the scatter deposit at the smoke tile is large
    # (dep * scatter_albedo * density, ~0.92*sd of the local light).
    glow_tile = glow[0, smoke_col]
    assert np.all(glow_tile > 0.0), "white smoke must glow (scatter)"
    # White smoke glows FAR brighter than black smoke for the same density.
    black = _single_gas_field(SMOKE, N_GASES, smoke_col, sd, w=w)
    _, glow_black = _cast(black, absorp, scatter, absorb_scale=1.4, w=w)
    assert glow_tile.sum() > 5.0 * glow_black[0, smoke_col].sum(), (
        "white smoke (scatter 0.92) must out-glow black smoke (scatter 0.04)")


# -------------------------------------------------------------- mixing = Σ tau


def test_mixing_is_density_weighted_sum_of_tau():
    # Two gases sharing a tile attenuate as exp(-scale*(tau_a + tau_b)) per
    # channel == trans_a * trans_b. Verify the combined transmission equals the
    # PRODUCT of the two single-gas transmissions, per channel (mixing == Σ tau).
    absorp, scatter = _canon_table()
    w = 20
    smoke_col = 5
    scale = 1.4
    da, db = 0.5, 0.4  # poison density, black density
    # Single-gas casts (control transmissions).
    g_pois = _single_gas_field(POISON, N_GASES, smoke_col, da, w=w)
    g_blk = _single_gas_field(SMOKE, N_GASES, smoke_col, db, w=w)
    rgb_p, _ = _cast(g_pois, absorp, scatter, absorb_scale=scale, w=w)
    rgb_b, _ = _cast(g_blk, absorp, scatter, absorb_scale=scale, w=w)
    ctrl, _ = _cast(np.zeros((N_GASES, 1, w), np.float32), absorp, scatter,
                    absorb_scale=scale, w=w)
    # Mixed cast: BOTH gases in the same tile.
    g_mix = g_pois + g_blk
    rgb_m, _ = _cast(g_mix, absorp, scatter, absorb_scale=scale, w=w)
    past = smoke_col + 1
    for c in range(3):
        trans_p = rgb_p[0, past, c] / ctrl[0, past, c]
        trans_b = rgb_b[0, past, c] / ctrl[0, past, c]
        trans_m = rgb_m[0, past, c] / ctrl[0, past, c]
        # exp(-scale*(da*ab_p + db*ab_b)) == exp(-scale*da*ab_p)*exp(-scale*db*ab_b)
        assert np.isclose(trans_m, trans_p * trans_b, rtol=1e-4), (
            f"channel {c}: mixed transmission {trans_m} != product of singles "
            f"{trans_p * trans_b} (mixing must be Σ tau)")
    # Cross-check against the analytic summed-tau transmission for one channel.
    tau_b_blue = scale * (da * absorp[POISON, 2] + db * absorp[SMOKE, 2])
    trans_blue_analytic = float(np.exp(-tau_b_blue))
    trans_blue_measured = rgb_m[0, past, 2] / ctrl[0, past, 2]
    assert np.isclose(trans_blue_measured, trans_blue_analytic, rtol=1e-3), (
        f"blue mixed transmission {trans_blue_measured} != analytic "
        f"{trans_blue_analytic}")


def test_mixing_scatter_glow_is_summed():
    # The scatter/glow deposit is ALSO a density-weighted sum: poison+black glow
    # == poison-glow + black-glow at the shared tile (additive, per channel).
    absorp, scatter = _canon_table()
    w = 20
    smoke_col = 3
    da, db = 0.5, 0.4
    g_pois = _single_gas_field(POISON, N_GASES, smoke_col, da, w=w)
    g_blk = _single_gas_field(SMOKE, N_GASES, smoke_col, db, w=w)
    _, glow_p = _cast(g_pois, absorp, scatter, absorb_scale=1.4, w=w)
    _, glow_b = _cast(g_blk, absorp, scatter, absorb_scale=1.4, w=w)
    _, glow_m = _cast(g_pois + g_blk, absorp, scatter, absorb_scale=1.4, w=w)
    # Glow deposit happens BEFORE attenuation (same local light), so it sums.
    assert np.allclose(glow_m[0, smoke_col],
                       glow_p[0, smoke_col] + glow_b[0, smoke_col], rtol=1e-4), (
        f"mixed glow {glow_m[0,smoke_col]} != sum of singles "
        f"{glow_p[0,smoke_col] + glow_b[0,smoke_col]}")


# ------------------------------------------- single gas == old single-smoke path


def test_single_gas_reproduces_beer_lambert():
    # A single populated gas with neutral coefficients [a,a,a] reproduces the
    # documented single-smoke Beer-Lambert transmission exp(-a*sd*scale) for the
    # tile it occupies — the M1 "behaviour-preserving" property.
    w = 20
    smoke_col = 5
    sd, a, scale = 0.5, 0.7, 1.4
    # One-gas table (N=1) with neutral grey coefficients.
    gas = np.zeros((1, 1, w), np.float32)
    gas[0, 0, smoke_col] = sd
    absorp = np.array([[a, a, a]], np.float32)
    scatter = np.zeros((1, 3), np.float32)
    rgb, _ = _cast(gas, absorp, scatter, absorb_scale=scale, w=w)
    ctrl, _ = _cast(np.zeros((1, 1, w), np.float32), absorp, scatter,
                    absorb_scale=scale, w=w)
    past = smoke_col + 1
    trans = rgb[0, past, 0] / ctrl[0, past, 0]
    expected = float(np.exp(-a * sd * scale))
    assert np.isclose(trans, expected, rtol=1e-3), (
        f"single-gas transmission {trans} != Beer-Lambert {expected}")
    assert rgb[0, past, 0] > 0.0, "beam survives (exp never hits 0)"


def test_empty_gas_leaves_light_untouched_and_no_glow():
    # No gas density anywhere -> the march deposits zero glow and the light is
    # the plain no-occlusion beam (gases only act where density > 0).
    absorp, scatter = _canon_table()
    w = 20
    gas = np.zeros((N_GASES, 1, w), np.float32)
    rgb, glow = _cast(gas, absorp, scatter, absorb_scale=1.4, w=w)
    assert rgb.sum() > 0.0, "sanity: the beam is lit"
    assert np.all(glow == 0.0), f"no gas must leave glow zero: {glow[glow != 0]}"


# --------------------------------------------------------------- determinism


def test_multigas_march_is_bit_identical():
    # Same multi-gas scene, two casts -> bit-identical light AND glow buffers
    # (no RNG on a single fixed ray; the inner gas sum is a deterministic fold).
    absorp, scatter = _canon_table()
    w = 24
    gas = np.zeros((N_GASES, 1, w), np.float32)
    gas[POISON, 0, 5] = 0.5
    gas[SMOKE, 0, 5] = 0.4
    gas[STEAM, 0, 8] = 0.3
    rgb1, glow1 = _cast(gas, absorp, scatter, absorb_scale=1.4, w=w)
    rgb2, glow2 = _cast(gas, absorp, scatter, absorb_scale=1.4, w=w)
    assert np.array_equal(rgb1, rgb2), "light buffer must be deterministic"
    assert np.array_equal(glow1, glow2), "glow buffer must be deterministic"
