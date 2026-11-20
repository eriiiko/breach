"""Water W1 (#97): the fine surface that flows -- properties of the law's
numpy twin (``renderer/water_surface.py``), of its extension of the shared
advection crossfade (``renderer/advected_noise.py``) and of the ripple bake
(``renderer/noise_bake.py``). The shader is held to the twin by
``tests/gl_water_flow_check.py`` (wrapper ``tests/test_gl_water_flow.py``).

Each test names the property it protects and the change that must break it.
No GL, no sim.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from renderer import advected_noise as AN
from renderer import water_surface as W
from renderer.noise_bake import (RIPPLE_CELLS, bake_ripple_rgba,
                                 ripple_field)


@pytest.fixture(scope="module")
def tex():
    return bake_ripple_rgba(256)


@pytest.fixture(scope="module")
def settings():
    from config import CFG
    return W.water_settings(CFG)


def _frame(s, t=12.34, tps=24.0, zoom=64.0):
    return W.flow_frame(t, tps, zoom, s)


# ---------------------------------------------------------------------------
# The shared crossfade: extended, and bit-identical for its old callers
# ---------------------------------------------------------------------------

def _frozen_advection_phase(tick, cycle_seconds, tps):
    """advection_phase as it stood before W1 (76bd422e), frozen."""
    tau = float(cycle_seconds) * float(tps)
    if tau <= 0.0:
        return (0.0, 0.0)
    return (float(np.mod(float(tick) / tau, 1.0)), tau)


def _frozen_layer_ages_weight(phase):
    p = float(phase) % 1.0
    return p, (p + 0.5) % 1.0, float(0.5 - 0.5 * np.cos(2.0 * np.pi * p))


def test_the_crossfade_is_bit_identical_for_the_tick_stepped_callers():
    """Property: with no ``alpha`` and no ``offset`` the extended
    ``advection_phase`` / ``layer_ages_weight`` return exactly (==, not
    close) what they returned before W1, so the gas detail, the gas relief
    and the speckle draw bit-identically. Breaks if the extension changes
    the default arithmetic (e.g. ``(tick + alpha) / tau`` rewritten as
    ``tick / tau + alpha / tau``)."""
    rng = np.random.default_rng(97)
    for tick in list(range(0, 200)) + list(rng.integers(0, 10**9, 300)):
        for cyc, tps in ((6.0, 24.0), (0.45, 24.0), (2.5, 30.0), (0.0, 24.0)):
            got = AN.advection_phase(int(tick), cyc, tps)
            assert (got.phase, got.tau_ticks) == _frozen_advection_phase(
                int(tick), cyc, tps)
            assert AN.layer_ages_weight(got.phase) == \
                _frozen_layer_ages_weight(got.phase)


def test_the_sub_tick_alpha_moves_the_phase_between_ticks():
    """Property: ``alpha`` advances the phase continuously between two
    ticks (phase(t + a) = phase(t) + a / tau), so the water flows smoothly
    at any frame rate. Breaks if alpha is ignored or mis-scaled."""
    a0 = AN.advection_phase(100, 1.0, 24.0)
    a1 = AN.advection_phase(100, 1.0, 24.0, alpha=0.5)
    assert a1.phase == pytest.approx(a0.phase + 0.5 / 24.0, abs=1e-12)


def test_the_field_form_equals_the_scalar_form():
    """Property: ``layer_ages_weight_field`` is the scalar law at every
    point (the twin and the scalar can never drift apart). Breaks if one is
    edited without the other."""
    rng = np.random.default_rng(1)
    off = rng.uniform(-0.5, 0.5, 200)
    for ph in (0.0, 0.123, 0.5, 0.999):
        a0, a1, w0 = AN.layer_ages_weight_field(ph, off)
        for i in range(off.size):
            s0, s1, sw = AN.layer_ages_weight(ph, off[i])
            assert a0[i] == pytest.approx(s0, abs=1e-12)
            assert a1[i] == pytest.approx(s1, abs=1e-12)
            assert w0[i] == pytest.approx(sw, abs=1e-12)


def test_the_phase_offset_is_smooth_and_spread_over_a_cycle():
    """Property: the per-point phase offset spans (nearly) a whole cycle
    across a few of its lengths -- so the surface never resets all at once
    -- and changes slowly between neighbours (no tearing). Breaks if it is
    constant (a whole-surface pulse) or high-frequency."""
    x, y = np.meshgrid(np.linspace(0, 30, 301), np.linspace(0, 30, 301))
    o = AN.phase_offset_field(x, y, 3.0)
    assert o.max() - o.min() > 0.8
    assert np.abs(o).max() <= 0.5 + 1e-12
    step = np.abs(np.diff(o, axis=1)).max() / 0.1      # per metre
    assert step < 0.5 / 3.0 * 2 * np.pi * 1.2


# ---------------------------------------------------------------------------
# The ripple bake
# ---------------------------------------------------------------------------

def test_the_ripple_texture_tiles_exactly_and_has_unit_rms_slope(tex):
    """Property: the ripple field is exactly periodic (integer wave vectors:
    no seam where the texture repeats) and the bake's slopes have RMS length
    1 (so ``detail_strength_*`` IS the RMS slope). Breaks with a non-integer
    wave vector or a changed normalisation."""
    rng = np.random.default_rng(3)
    c = rng.uniform(-2, 2, (500, 2))
    for shift in ((1.0, 0.0), (0.0, 1.0), (3.0, -2.0)):
        a = ripple_field(c)
        b = ripple_field(c + np.array(shift))
        for u, v in zip(a, b):
            assert np.allclose(u, v, atol=1e-9)
    rms = np.sqrt(np.mean(tex[..., 0].astype(np.float64) ** 2
                          + tex[..., 1].astype(np.float64) ** 2))
    assert rms == pytest.approx(1.0, rel=1e-5)
    assert np.array_equal(tex, bake_ripple_rgba(256)), "bake not deterministic"


# ---------------------------------------------------------------------------
# The settings door
# ---------------------------------------------------------------------------

def _cfg(**over):
    from config import CFG
    d = dict(vars(CFG.graphics.water))
    d.update(over)
    return SimpleNamespace(graphics=SimpleNamespace(
        water=SimpleNamespace(**{k: v for k, v in d.items()
                                 if v is not None})))


def test_the_door_refuses_missing_mistyped_and_out_of_range_dials():
    """Property: ``water_settings`` refuses a missing section or key, a
    mistyped value and an out-of-range one, by name. Breaks if a dial falls
    back to a default (a typo would hide while tuning)."""
    for name in W.config_keys():
        with pytest.raises(ValueError, match=name):
            W.water_settings(_cfg(**{name: None}))
    with pytest.raises(ValueError, match="debug_view"):
        W.water_settings(_cfg(debug_view=1.0))
    with pytest.raises(ValueError, match="flow_cycle_s"):
        W.water_settings(_cfg(flow_cycle_s="slow"))
    with pytest.raises(ValueError, match="water_color"):
        W.water_settings(_cfg(water_color=[0.1, 0.2]))
    with pytest.raises(ValueError, match="detail_wavelength_m"):
        W.water_settings(_cfg(detail_wavelength_m=0.0))
    with pytest.raises(ValueError, match="detail_zoom_full"):
        W.water_settings(_cfg(detail_zoom_zero=20.0, detail_zoom_full=10.0))
    with pytest.raises(ValueError, match=r"\[graphics.water\]"):
        W.water_settings(SimpleNamespace(graphics=SimpleNamespace()))


def test_the_config_section_holds_exactly_the_doors_dials():
    """Property: ``[graphics.water]`` holds exactly the keys the door reads
    -- no dead dial (``wave_scale`` and ``ambient_amp`` went with the
    ambient sines) and none the door would miss. Breaks if a key is added
    to one side only."""
    from config import CFG
    assert set(vars(CFG.graphics.water)) == set(W.config_keys())


# ---------------------------------------------------------------------------
# The law
# ---------------------------------------------------------------------------

def _grid(n=48, span=4.0, x0=2.0, y0=1.0):
    y, x = np.meshgrid(np.linspace(y0, y0 + span, n),
                       np.linspace(x0, x0 + span, n), indexing="ij")
    return x, y


def test_still_water_is_time_steady_but_for_the_drift(tex, settings):
    """Property: with zero flow the fine surface at time t2 is the surface
    at t1 moved by the drift (and with no drift it does not change at all):
    still water does not breathe or pulse with the crossfade. Breaks if the
    two layers read different points at zero flow (e.g. a per-layer grain
    offset) or the crossfade weights do not sum to one."""
    s = settings
    x, y = _grid()
    v = np.zeros(x.shape + (2,))
    depth = np.full(x.shape, 0.3)
    f1, f2 = _frame(s, t=10.0), _frame(s, t=10.37)
    d1 = W.detail_slope(x, y, v, depth, 0.0, f1, tex, s)
    dx = s.period_m * (f2.drift_u - f1.drift_u)
    dy = s.period_m * (f2.drift_v - f1.drift_v)
    assert abs(dx) + abs(dy) > 1e-3, "no drift to test against"
    d2 = W.detail_slope(x + dx, y + dy, v, depth, 0.0, f2, tex, s)
    assert np.allclose(d1, d2, atol=1e-9)
    s0 = dataclasses.replace(s, detail_drift_mps=0.0)
    a = W.detail_slope(x, y, v, depth, 0.0, _frame(s0, t=3.0), tex, s0)
    b = W.detail_slope(x, y, v, depth, 0.0, _frame(s0, t=3.29), tex, s0)
    assert np.allclose(a, b, atol=1e-12)
    assert np.abs(a).max() > 0.0


def test_reversing_the_flow_reverses_the_slide(settings):
    """Property: each layer's texture coordinates move by exactly the
    opposite amount when the flow is reversed (the surface streams with the
    water, downstream). Breaks if a sign or an axis is crossed."""
    s = settings
    x, y = _grid(16)
    depth = np.full(x.shape, 0.3)
    rng = np.random.default_rng(5)
    v = rng.uniform(-0.8, 0.8, x.shape + (2,))
    f = _frame(s, t=4.2)
    q0s, q1s, _, _ = W.layer_coords(x, y, 0 * v, depth, f, s)
    q0p, q1p, _, _ = W.layer_coords(x, y, v, depth, f, s)
    q0m, q1m, _, _ = W.layer_coords(x, y, -v, depth, f, s)
    for qs, qp, qm in ((q0s, q0p, q0m), (q1s, q1p, q1m)):
        assert np.allclose(qp - qs, -(qm - qs), atol=1e-12)
    # downstream: the texture coordinate moves AGAINST v (content moves with v)
    a0, _, _ = AN.layer_ages_weight_field(f.phase, AN.phase_offset_field(
        x, y, s.flow_phase_noise_m))
    moving = a0 > 0.05
    dot = ((q0p - q0s) * v).sum(axis=-1)
    assert np.all(dot[moving] < 0.0)


def test_the_slide_cap_holds(settings):
    """Property (critique M7): no layer ever slides further than
    ``flow_max_offset_m``, however fast the flood. Breaks if the clamp goes
    or is applied per axis."""
    s = settings
    rng = np.random.default_rng(7)
    v = rng.uniform(-30.0, 30.0, (5000, 2))
    age = rng.uniform(0.0, 5.0, 5000)
    d = W.flow_slide(v, age, s.flow_max_offset_m)
    assert np.sqrt((d ** 2).sum(-1)).max() <= s.flow_max_offset_m * (1 + 1e-12)
    x, y = _grid(24)
    vv = rng.uniform(-30.0, 30.0, x.shape + (2,))
    depth = np.full(x.shape, 0.5)
    for t in np.linspace(0.0, 2.0, 17):
        f = _frame(s, t=t)
        q0s, q1s, _, _ = W.layer_coords(x, y, 0 * vv, depth, f, s)
        q0, q1, _, _ = W.layer_coords(x, y, vv, depth, f, s)
        for a, b in ((q0, q0s), (q1, q1s)):
            slide = np.sqrt((((a - b) * s.period_m) ** 2).sum(-1))
            assert slide.max() <= s.flow_max_offset_m * (1 + 1e-9)
    # small velocities are NOT capped (the cap is a ceiling, not a scale)
    small = W.flow_slide(np.array([[0.1, 0.0]]), np.array([1.0]),
                         s.flow_max_offset_m)
    assert small[0, 0] == pytest.approx(0.1)


def test_the_flow_fades_to_nothing_at_the_wet_dry_edge(settings):
    """Property (M7): at zero depth the flow slides nothing, from
    ``flow_edge_depth_m`` on it slides fully. Breaks if the depth weight is
    dropped (the shoreline would tear)."""
    s = settings
    x, y = _grid(8)
    v = np.full(x.shape + (2,), 0.7)
    f = _frame(s, t=1.1)
    for depth, k in ((0.0, 0.0), (s.flow_edge_depth_m, 1.0), (0.5, 1.0)):
        _, _, _, ve = W.layer_coords(x, y, v, np.full(x.shape, depth), f, s)
        assert np.allclose(ve, v * k)


def test_the_strength_rises_with_flow_and_agitation_and_fades_zoomed_out(
        settings):
    """Property: still water has ``detail_strength_still``, a flow at or
    above ``flow_speed_ref`` (or agitation at ``agitation_ref``) the full
    ``detail_strength_flow``, monotone between; zoomed out below
    ``detail_zoom_zero`` the fine surface is gone (critique M8). Breaks if
    the strength ignores the flow or the zoom fade is lost."""
    s = settings
    f = _frame(s, zoom=64.0)
    sp = np.linspace(0.0, 2.0 * s.flow_speed_ref, 41)
    v = np.stack([sp, 0 * sp], -1)
    k = W.strength(v, 0.0, f, s)
    assert k[0] == pytest.approx(s.detail_strength_still)
    assert k[-1] == pytest.approx(s.detail_strength_flow)
    assert np.all(np.diff(k) >= -1e-15)
    ka = W.strength(np.zeros((1, 2)), s.agitation_ref, f, s)
    assert ka[0] == pytest.approx(s.detail_strength_flow)
    f_out = _frame(s, zoom=s.detail_zoom_zero)
    assert np.all(W.strength(v, 0.0, f_out, s) == 0.0)


def _phase_reset_change(tex, s, dt=2e-4, n_t=240, offsets=True):
    """Max |change| of the surface over ``dt`` at ``n_t`` times through one
    cycle, over a grid with a uniform flow."""
    if not offsets:
        s = dataclasses.replace(s, flow_phase_noise_m=0.0)
    x, y = _grid(24, span=6.0)
    v = np.zeros(x.shape + (2,))
    v[..., 0], v[..., 1] = 0.6, -0.25
    depth = np.full(x.shape, 0.3)
    worst = 0.0
    cyc = s.flow_cycle_s
    for t in 50.0 * cyc + np.linspace(0.0, cyc, n_t, endpoint=False):
        a = W.detail_slope(x, y, v, depth, 0.0, _frame(s, t=t), tex, s)
        b = W.detail_slope(x, y, v, depth, 0.0, _frame(s, t=t + dt), tex, s)
        worst = max(worst, float(np.abs(b - a).max()))
    return worst


def test_no_pulse_across_a_phase_reset(tex, settings, monkeypatch):
    """Property (critique minor): the surface changes boundedly over a small
    time step at EVERY moment of the cycle -- including the instants a layer
    resets, which sit under zero crossfade weight -- with and without the
    per-point phase offset. The test is not vacuous: a crossfade frozen at
    half weight (the reset in plain view) fails it. Breaks if the reset is
    no longer hidden (weights, ages or offsets out of step)."""
    s = settings
    k = s.detail_strength_flow
    for offsets in (True, False):
        worst = _phase_reset_change(tex, s, offsets=offsets)
        assert worst < 0.05 * k, (offsets, worst)

    def frozen(phase, offset):
        a0, a1, _w = AN.layer_ages_weight_field(phase, offset)
        return a0, a1, np.full(np.shape(a0), 0.5)
    monkeypatch.setattr(W, "layer_ages_weight_field", frozen)
    assert _phase_reset_change(tex, s, n_t=60) > 0.2 * k


def _row_spectrum(field_rows):
    f = field_rows - field_rows.mean(axis=1, keepdims=True)
    return (np.abs(np.fft.rfft(f, axis=1)) ** 2).mean(axis=0)


def test_no_spectral_peak_at_the_tile_frequency(tex, settings):
    """Property (critique minor, replacing "offset by a whole tile"): with a
    flow that changes tile by tile (bilinear, as the GPU reads it) the
    surface's spectrum has no peak at the tile frequency -- no tile-locked
    pattern. The detector is not vacuous: a pattern locked to the tiles (the
    old ambient sines' kind) shows a peak. Breaks if the surface is sized or
    driven per tile."""
    from renderer.heat_shimmer import bilinear_tiles
    s = settings
    tile_m = 1.0 / 3.0
    spt = 16                                   # samples per tile
    n_tiles, n_rows = 64, 24
    rng = np.random.default_rng(11)
    vx = rng.uniform(-1.0, 1.0, (n_rows + 2, n_tiles + 2))
    vy = rng.uniform(-1.0, 1.0, (n_rows + 2, n_tiles + 2))
    xt = (np.arange(n_tiles * spt) + 0.5) / spt
    yt = np.arange(n_rows) + 1.5
    XT, YT = np.meshgrid(xt, yt)
    v = np.stack([bilinear_tiles(vx, XT, YT), bilinear_tiles(vy, XT, YT)], -1)
    d = W.detail_slope(XT * tile_m, YT * tile_m, v, np.full(XT.shape, 0.3),
                       0.0, _frame(s, t=7.7), tex, s)
    k_tile = n_tiles                         # bin of 1 cycle per tile

    def peak_ratio(rows):
        p = _row_spectrum(rows)
        near = np.r_[p[k_tile - 12:k_tile - 1], p[k_tile + 2:k_tile + 13]]
        return p[k_tile] / np.median(near)
    for c in (0, 1):
        assert peak_ratio(d[..., c]) < 4.0
    locked = d[..., 0] + 0.5 * np.abs(d[..., 0]).mean() * np.sin(
        2 * np.pi * XT)
    assert peak_ratio(locked) > 20.0


def test_no_spectral_peak_at_the_ripple_textures_own_period(tex, settings):
    """Property (Erik, 2026-11-20: the ripple must not show repetition):
    sampled over a whole number M of texture periods, a plain lookup puts
    all its power on the comb of the texture's own period (every M-th bin);
    the hex-tiled surface spreads it, so the comb holds about its share of
    the bins and no more. Breaks if the hex-tiling is bypassed, its offsets
    or rotations stop varying per vertex, or the lattice repeats with the
    texture."""
    s = settings
    M, spp = 8, 128                            # periods, samples per period
    x = (np.arange(M * spp) + 0.5) / spp * s.period_m
    ys = np.linspace(0.0, 3.0 * s.period_m, 24)
    X, Y = np.meshgrid(x, ys)
    v = np.zeros(X.shape + (2,))
    f = _frame(s, t=2.0)
    comb = np.zeros(M * spp // 2 + 1, bool)
    comb[::M] = True
    comb[0] = False

    def comb_share(lookup):
        d = W.detail_slope(X, Y, v, np.full(X.shape, 0.3), 0.0, f, tex, s,
                           lookup=lookup)
        share = []
        for c in (0, 1):
            p = _row_spectrum(d[..., c])
            share.append(p[comb].sum() / p[1:].sum())
        return max(share)
    assert comb_share(W.plain_slope) > 0.9      # the repetition is real
    assert comb_share(W.hex_slope) < 3.0 / M


def test_the_hex_lookup_keeps_the_contrast(tex):
    """Property: the hex-tiled lookup keeps most of the texture's slope
    variance (Mikkelsen's luminance x w^7 weights blend near-single taps,
    not a washed-out average). Breaks if the weights flatten (e.g. plain
    barycentric weights)."""
    yy, xx = np.mgrid[0:96, 0:96]
    st = np.stack([xx / 37.0, yy / 37.0], -1)
    h = W.hex_slope(tex, st)
    p = W.plain_slope(tex, st)
    assert h.std() > 0.8 * p.std()


def test_wet_box_reads_the_int_plane():
    """Property (critique M4): the wet box is None on a dry int plane and
    the wet tiles' box dilated by one tile otherwise. Breaks if a dry map
    reports a box (it would be drawn)."""
    q = np.zeros((10, 12), np.int32)
    assert W.wet_box(q) is None
    q[3:5, 4:7] = 1
    assert W.wet_box(q) == (3, 2, 8, 6)
    q[0, 0] = 5
    assert W.wet_box(q) == (0, 0, 8, 6)
