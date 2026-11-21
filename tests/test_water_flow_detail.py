"""Water W1 + W1b (#97): the fine surface that flows -- properties of the
law's numpy twin (``renderer/water_surface.py``: since W1b procedural
travelling, sharpened sines in two layers on a hex lattice) and of its
extension of the shared advection crossfade (``renderer/advected_noise.py``).
The shader is held to the twin by ``tests/gl_water_flow_check.py`` (wrapper
``tests/test_gl_water_flow.py``).

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
# The wave set (W1b): travelling, sharpened sines
# ---------------------------------------------------------------------------

def _q(n=60, span=1.3, x0=0.4, y0=0.7):
    y, x = np.meshgrid(np.linspace(y0, y0 + span, n),
                       np.linspace(x0, x0 + span, n), indexing="ij")
    return np.stack([x, y], -1)


@pytest.mark.parametrize("K", [1.0, 1.6, 3.0])
def test_the_wave_derivatives_are_exact(settings, K):
    """Property (W1b): ``wave_sum``'s slope IS the gradient of its height
    -- central differences agree to 1e-6 of their scale, for the plain sine
    (K = 1) and sharpened crests. Breaks if Finch's ``g'`` is mis-derived, a
    wave vector or an amplitude enters one and not the other."""
    for layer in W.LAYERS:
        tab = _frame(settings).waves[layer]
        q = _q()
        Kq = np.full(q.shape[:-1], K)
        h, sl = W.wave_sum(tab, q, Kq)
        e = 1e-6

        def at(dx, dy):
            return W.wave_sum(tab, q + [dx, dy], Kq)
        hx = (at(e, 0)[0] - at(-e, 0)[0]) / (2 * e)
        hy = (at(0, e)[0] - at(0, -e)[0]) / (2 * e)
        scale = np.abs(sl).max()
        assert np.abs(hx - sl[..., 0]).max() < 1e-6 * scale + 1e-9
        assert np.abs(hy - sl[..., 1]).max() < 1e-6 * scale + 1e-9


def test_sharpness_one_is_a_plain_sine_sum(settings):
    """Property (W1b): at ``K = 1`` Finch's shape is the identity -- the
    set is ``sum a_i sin(k_i . q + phi_i)``, its slope ``sum a_i cos(.)
    k_i`` bit for bit; above 1 the crests are sharper than the troughs
    (the height skews positive; ten waves dilute it). Breaks if the shape leaks into K = 1 or
    sharpens nothing."""
    tab = _frame(settings).waves[W.LAYER_DETAIL]
    q = _q(80, span=3.0)
    h, sl = W.wave_sum(tab, q, np.ones(q.shape[:-1]))
    href = np.zeros(q.shape[:-1])
    sref = np.zeros(q.shape)
    for (kx, ky), ph, am in zip(tab.k, tab.phase, tab.amp):
        a = am / np.hypot(kx, ky)
        psi = kx * q[..., 0] + ky * q[..., 1] + ph
        href += a * np.sin(psi)
        sref[..., 0] += a * np.cos(psi) * kx
        sref[..., 1] += a * np.cos(psi) * ky
    assert np.allclose(h, href, atol=1e-12)
    assert np.allclose(sl, sref, atol=1e-12)

    def skew(a):
        a = a - a.mean()
        return float((a ** 3).mean() / (a ** 2).mean() ** 1.5)
    assert abs(skew(h)) < 0.15
    hs, _s = W.wave_sum(tab, q, np.full(q.shape[:-1], 3.0))
    assert skew(hs) > 0.2 and skew(hs) > abs(skew(h)) + 0.15


def test_the_wave_set_is_incommensurate_and_runs_at_waters_speed(settings):
    """Property (W1b): within a layer no two wavelengths are in a small
    rational ratio (p/q, q <= 12, to 1e-6: the sum would repeat on a short
    period), every wave runs at the capillary-gravity dispersion speed of
    its own wave number, and of two waves the longer is faster (all are
    past the 1.7 cm capillary minimum). Breaks if the wavelengths are drawn
    on a lattice (the W1 bake's integer wave vectors) or the speeds are one
    constant (the pattern would slide, not morph)."""
    for layer in W.LAYERS:
        ws = W.wave_set(layer, settings)
        k = np.hypot(ws.k[:, 0], ws.k[:, 1])
        n = len(k)
        assert n == (settings.wave_count if layer == W.LAYER_DETAIL
                     else settings.swell_wave_count)
        for i in range(n):
            for j in range(i + 1, n):
                r = k[i] / k[j]
                for qd in range(1, 13):
                    assert abs(r * qd - round(r * qd)) > 1e-6 * qd, \
                        (layer, i, j, r)
        om = W.dispersion_omega(k, settings.wave_speed)
        assert np.allclose(ws.omega, om)
        assert np.allclose(om, np.sqrt(W.G_ACCEL * k
                                       + W.SIGMA_OVER_RHO * k ** 3))
        c = om / k
        order = np.argsort(-k)                  # short to long
        assert np.all(np.diff(c[order]) > 0)


def test_each_layer_has_its_strength_as_rms_slope(settings):
    """Property (W1b): on still water, over a few cells, a layer alone has
    RMS slope ``*_strength_still`` (within 15 %: the hex blend and the
    finite window) -- the set's amplitudes are normalised at the base
    sharpness, so a strength IS the RMS slope whatever the count or the
    shape. Breaks if a layer's normalisation (``finch_rms``, the sum of
    amplitudes) is lost."""
    y, x = np.mgrid[0:200, 0:200] * 0.05
    v = np.zeros(x.shape + (2,))
    d = np.full(x.shape, 0.3)
    for k_d, k_s in ((0.05, 0.0), (0.0, 0.07)):
        s = dataclasses.replace(settings, detail_strength_still=k_d,
                                detail_strength_flow=k_d,
                                swell_strength_still=k_s,
                                swell_strength_flow=k_s)
        sl = W.detail_slope(x + 1.0, y + 2.0, v, d, 0.0, _frame(s), s)
        rms = float(np.sqrt((sl ** 2).sum(-1).mean()))
        assert rms == pytest.approx(k_d + k_s, rel=0.15)
    assert W.finch_rms(1.0) == pytest.approx(1.0)
    ph = np.linspace(0, 2 * np.pi, 20001)[:-1]
    for K in (1.6, 3.0):
        u = (np.sin(ph) + 1) / 2
        g1 = K * u ** (K - 1) * np.cos(ph)
        assert W.finch_rms(K) == pytest.approx(
            np.sqrt(2 * np.mean(g1 ** 2)), rel=1e-6)


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
    with pytest.raises(ValueError, match="wave_count"):
        W.water_settings(_cfg(wave_count=W.MAX_WAVES + 1))
    with pytest.raises(ValueError, match="swell_wave_count"):
        W.water_settings(_cfg(swell_wave_count=0))
    with pytest.raises(ValueError, match="wave_sharpness"):
        W.water_settings(_cfg(wave_sharpness=0.5))
    with pytest.raises(ValueError, match="wave_cell_m"):
        W.water_settings(_cfg(wave_cell_m=0.0))
    with pytest.raises(ValueError, match="swell_spread_deg"):
        W.water_settings(_cfg(swell_spread_deg=200.0))
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


def _best_shift_corr(a, b):
    """The largest correlation of slope field ``b`` with ``a`` over EVERY
    circular shift (an FFT cross-correlation, both components): 1 when
    ``b`` is ``a`` moved rigidly, low when the pattern changed shape."""
    num = 0.0
    for c in (0, 1):
        fa = np.fft.fft2(a[..., c] - a[..., c].mean())
        fb = np.fft.fft2(b[..., c] - b[..., c].mean())
        num = num + np.real(np.fft.ifft2(fa * np.conj(fb)))
    den = np.sqrt(((a - a.mean((0, 1))) ** 2).sum()
                  * ((b - b.mean((0, 1))) ** 2).sum())
    return float(num.max() / den)


def test_still_water_morphs_and_is_smooth_in_time(settings):
    """Property (W1b; Erik: "sines in x and y morph differently all the
    time"): on still water the surface at t + 2 s is NOT the surface at t
    moved anywhere -- its best correlation over every shift falls below 0.5
    -- while over one frame (1/60 s) it barely changes (> 0.95 in place).
    The detector is not vacuous: a rigidly moved field scores ~1 (W1's
    sliding texture, whose drift was its only motion). Breaks if the waves
    stop travelling at their own speeds (one speed for all: a rigid slide)
    or jump between frames."""
    s = settings
    n, span = 128, 2.56
    y, x = np.mgrid[0:n, 0:n] * (span / n)
    x, y = x + 3.0, y + 3.0
    v = np.zeros(x.shape + (2,))
    d = np.full(x.shape, 0.3)
    a = W.detail_slope(x, y, v, d, 0.0, _frame(s, t=20.0), s)
    b = W.detail_slope(x, y, v, d, 0.0, _frame(s, t=22.0), s)
    c = W.detail_slope(x, y, v, d, 0.0, _frame(s, t=20.0 + 1 / 60), s)
    assert _best_shift_corr(a, b) < 0.5
    inplace = float((a * c).sum() / np.sqrt((a * a).sum() * (c * c).sum()))
    assert inplace > 0.95
    rigid = np.roll(a, (7, -11), axis=(0, 1))
    assert _best_shift_corr(a, rigid) > 0.99


def test_reversing_the_flow_reverses_the_slide(settings):
    """Property: each layer's advected point moves by exactly the opposite
    amount when the flow is reversed (the surface streams with the water,
    downstream). Breaks if a sign or an axis is crossed."""
    s = settings
    x, y = _grid(16)
    depth = np.full(x.shape, 0.3)
    rng = np.random.default_rng(5)
    v = rng.uniform(-0.8, 0.8, x.shape + (2,))
    f = _frame(s, t=4.2)
    q0s, q1s, _, _ = W.layer_points(x, y, 0 * v, depth, f, s)
    q0p, q1p, _, _ = W.layer_points(x, y, v, depth, f, s)
    q0m, q1m, _, _ = W.layer_points(x, y, -v, depth, f, s)
    for qs, qp, qm in ((q0s, q0p, q0m), (q1s, q1p, q1m)):
        assert np.allclose(qp - qs, -(qm - qs), atol=1e-12)
    # downstream: the evaluated point moves AGAINST v (content moves with v)
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
        q0s, q1s, _, _ = W.layer_points(x, y, 0 * vv, depth, f, s)
        q0, q1, _, _ = W.layer_points(x, y, vv, depth, f, s)
        for a, b in ((q0, q0s), (q1, q1s)):
            slide = np.sqrt(((a - b) ** 2).sum(-1))
            assert slide.max() <= s.flow_max_offset_m * (1 + 1e-9)
    small = W.flow_slide(np.array([[0.1, 0.0]]), np.array([1.0]),
                         s.flow_max_offset_m)
    assert small[0, 0] == pytest.approx(0.1)


def test_slow_currents_leave_the_waves_to_their_own_speed(settings):
    """Property (W1b): the waves ride the flow from ``flow_still_mps`` on;
    below half of it their slides are EXACTLY 0 (both flow layers read the
    same point, evaluated once -- the sim's still rooms keep ~1 cm/s
    currents everywhere), and the fade is monotone between. Breaks if a
    slow current makes every still pixel pay for two evaluations or a fast
    one stops carrying the waves."""
    s = settings
    sp = np.linspace(0.0, 2.0 * s.flow_still_mps, 41)
    v = np.stack([sp, 0 * sp], -1)
    d = np.stack([sp, sp], -1) * 0.3
    w0, w1 = W.wave_slides(d, 2 * d, v, s)
    slow = sp < 0.5 * s.flow_still_mps
    assert not w0[slow].any() and not w1[slow].any()
    fast = sp >= s.flow_still_mps
    assert np.array_equal(w0[fast], d[fast])
    frac = w0[:, 0] / np.maximum(d[:, 0], 1e-30)
    assert np.all(np.diff(frac) >= -1e-12)


def test_the_flow_fades_to_nothing_at_the_wet_dry_edge(settings):
    """Property (M7): at zero depth the flow slides nothing, from
    ``flow_edge_depth_m`` on it slides fully. Breaks if the depth weight is
    dropped (the shoreline would tear)."""
    s = settings
    x, y = _grid(8)
    v = np.full(x.shape + (2,), 0.7)
    f = _frame(s, t=1.1)
    for depth, k in ((0.0, 0.0), (s.flow_edge_depth_m, 1.0), (0.5, 1.0)):
        _, _, _, ve = W.layer_points(x, y, v, np.full(x.shape, depth), f, s)
        assert np.allclose(ve, v * k)


def test_the_strength_rises_with_flow_and_agitation_and_fades_zoomed_out(
        settings):
    """Property: still water has each layer's ``*_strength_still``, a flow
    at or above ``flow_speed_ref`` (or agitation at ``agitation_ref``) its
    full ``*_strength_flow``, monotone between, and the crests sharpen
    with it (``wave_sharpness`` toward ``wave_sharpness_flow``); zoomed out
    below ``detail_zoom_zero`` the fine surface is gone (critique M8).
    Breaks if the strength ignores the flow or the zoom fade is lost."""
    s = settings
    f = _frame(s, zoom=64.0)
    sp = np.linspace(0.0, 2.0 * s.flow_speed_ref, 41)
    v = np.stack([sp, 0 * sp], -1)
    for layer, lo, hi in ((W.LAYER_DETAIL, s.detail_strength_still,
                           s.detail_strength_flow),
                          (W.LAYER_SWELL, s.swell_strength_still,
                           s.swell_strength_flow)):
        k = W.strength(v, 0.0, f, s, layer)
        assert k[0] == pytest.approx(lo)
        assert k[-1] == pytest.approx(hi)
        assert np.all(np.diff(k) >= -1e-15)
        ka = W.strength(np.zeros((1, 2)), s.agitation_ref, f, s, layer)
        assert ka[0] == pytest.approx(hi)
        f_out = _frame(s, zoom=s.detail_zoom_zero)
        assert np.all(W.strength(v, 0.0, f_out, s, layer) == 0.0)
    K = W.sharpness(v, 0.0, s)
    assert K[0] == pytest.approx(s.wave_sharpness)
    assert K[-1] == pytest.approx(s.wave_sharpness_flow)
    assert np.all(np.diff(K) >= -1e-15)


def test_a_room_heaving_as_one_has_the_still_strength(settings):
    """Property (W2's finding 1, 2026-11-20): the strength reads the RELATIVE
    agitation (``relative_agitation``: ripple_v against its wet neighbours',
    the packed ``u_water`` A), so a still room whose ripple_v is uniform --
    the sim's ripple field after a blast anywhere: ~0.84 m/s, then a frozen
    ~0.04 -- keeps ``detail_strength_still`` everywhere, up to its walls,
    while a tile moving against its neighbours is driven. Breaks if the
    strength reads raw |ripple_v| again (still rooms at ~0.75 of full) or a
    wall counts as a neighbour."""
    from renderer.heat_shimmer import bilinear_tiles
    s = settings
    f = _frame(s, zoom=64.0)
    h, w = 12, 16
    wet = np.ones((h, w), bool)
    wet[0, :] = wet[-1, :] = wet[:, 0] = wet[:, -1] = False   # the walls
    for heave in (0.84, 0.0375):
        rv = np.where(wet, heave, 0.0)
        a = W.relative_agitation(rv, wet)
        y, x = np.mgrid[1.0:h - 1.0:0.25, 1.0:w - 1.0:0.25]
        k = W.strength(np.zeros(x.shape + (2,)), bilinear_tiles(a, x, y),
                       f, s)
        assert np.allclose(k, s.detail_strength_still, atol=1e-9)
        raw = W.strength(np.zeros((1, 2)), heave, f, s)
        assert raw[0] > s.detail_strength_still + 0.3 * (
            s.detail_strength_flow - s.detail_strength_still)  # not vacuous
    rv = np.where(wet, 0.84, 0.0)
    rv[6, 8] += 2.0 * s.agitation_ref
    a = W.relative_agitation(rv, wet)
    k = W.strength(np.zeros((1, 2)), np.array([a[6, 8]]), f, s)
    assert k[0] == pytest.approx(s.detail_strength_flow)


def test_the_swell_follows_the_flow(settings):
    """Property (W1b): the swell's direction is ``swell_dir_deg`` on still
    water and the flow's own direction from ``flow_speed_ref`` on, turning
    continuously between; at a lattice vertex its set is turned that way,
    so a fast uniform flow turns the swell's slope toward the flow (its
    slope variance lies along the flow, not across). Breaks if the swell
    ignores the flow or turns the wrong way (the sign of the tap's turn)."""
    s = dataclasses.replace(settings, detail_strength_still=0.0,
                            detail_strength_flow=0.0, swell_spread_deg=0.0)
    a0 = np.radians(s.swell_dir_deg)
    assert W.swell_direction(np.zeros((1, 2)), s)[0] == pytest.approx(a0)
    for ang in (0.0, 2.0, -1.0):
        vv = np.array([[np.cos(ang), np.sin(ang)]]) * 2.0 * s.flow_speed_ref
        assert W.swell_direction(vv, s)[0] == pytest.approx(ang)
    y, x = np.mgrid[0:100, 0:100] * 0.04
    d = np.full(x.shape, 0.3)
    for ang in (0.3, 1.9):
        flow = np.array([np.cos(ang), np.sin(ang)]) * 1.5 * s.flow_speed_ref

        def fn(xm, ym):
            return np.broadcast_to(flow, np.shape(xm) + (2,))
        # the waves' own position, still (no slide): only the direction
        sl = W.detail_slope(x + 4, y + 4, np.zeros(x.shape + (2,)), d, 0.0,
                            _frame(s), s, flow_fn=fn)
        along = sl[..., 0] * np.cos(ang) + sl[..., 1] * np.sin(ang)
        across = -sl[..., 0] * np.sin(ang) + sl[..., 1] * np.cos(ang)
        assert (along ** 2).mean() > 4.0 * (across ** 2).mean(), ang


def _phase_reset_change(s, dt=2e-4, n_t=120, offsets=True):
    """Max |change| of the surface over ``dt`` at ``n_t`` times through one
    cycle, over a grid with a uniform flow."""
    if not offsets:
        s = dataclasses.replace(s, flow_phase_noise_m=0.0)
    x, y = _grid(20, span=4.0)
    v = np.zeros(x.shape + (2,))
    v[..., 0], v[..., 1] = 0.6, -0.25
    depth = np.full(x.shape, 0.3)
    worst = 0.0
    cyc = s.flow_cycle_s
    for t in 50.0 * cyc + np.linspace(0.0, cyc, n_t, endpoint=False):
        a = W.detail_slope(x, y, v, depth, 0.0, _frame(s, t=t), s)
        b = W.detail_slope(x, y, v, depth, 0.0, _frame(s, t=t + dt), s)
        worst = max(worst, float(np.abs(b - a).max()))
    return worst


def test_no_pulse_across_a_phase_reset(settings, monkeypatch):
    """Property (critique minor): the surface changes boundedly over a small
    time step at EVERY moment of the cycle -- including the instants a layer
    resets, which sit under zero crossfade weight -- with and without the
    per-point phase offset. The test is not vacuous: a crossfade frozen at
    half weight (the reset in plain view) fails it. Breaks if the reset is
    no longer hidden (weights, ages or offsets out of step)."""
    s = settings
    k = s.detail_strength_flow + s.swell_strength_flow
    for offsets in (True, False):
        worst = _phase_reset_change(s, offsets=offsets)
        assert worst < 0.05 * k, (offsets, worst)

    def frozen(phase, offset):
        a0, a1, _w = AN.layer_ages_weight_field(phase, offset)
        return a0, a1, np.full(np.shape(a0), 0.5)
    monkeypatch.setattr(W, "layer_ages_weight_field", frozen)
    assert _phase_reset_change(s, n_t=40) > 0.2 * k


def _row_spectrum(field_rows):
    f = field_rows - field_rows.mean(axis=1, keepdims=True)
    return (np.abs(np.fft.rfft(f, axis=1)) ** 2).mean(axis=0)


def test_no_spectral_peak_at_the_tile_frequency(settings):
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
                       0.0, _frame(s, t=7.7), s)
    k_tile = n_tiles                         # bin of 1 cycle per tile

    def peak_ratio(rows):
        p = _row_spectrum(rows)
        near = np.r_[p[k_tile - 12:k_tile - 1], p[k_tile + 2:k_tile + 13]]
        return p[k_tile] / np.median(near)
    for c in (0, 1):
        assert peak_ratio(d[..., c]) < 4.0
    locked = d[..., 0] + 0.5 * np.abs(d[..., 0]).mean() * np.sin(
        2 * np.pi * XT)
    assert peak_ratio(locked) > 10.0


def _max_lag_corr(sig, min_lag):
    """The largest normalised autocorrelation of a long 1D signal at any
    lag of at least ``min_lag`` samples (up to half its length)."""
    a = sig - sig.mean()
    n = len(a)
    f = np.fft.rfft(a, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / ac[0]
    return float(ac[min_lag:n // 2].max())


def test_no_exact_period_over_a_large_window(settings):
    """Property (W1b): along a 40 m line of still water the surface never
    repeats -- its autocorrelation stays under 0.5 at every lag from 1 m to
    20 m (the incommensurate wavelengths, and the lattice's per-cell turns
    and shifts). The detector is not vacuous: the same field made periodic
    (repeated every 1.9 m, W1's texture period) scores ~1. Breaks if the
    wave set becomes commensurate or the lattice stops varying the cells."""
    s = settings
    spm = 50                                       # samples per metre
    x = np.arange(40 * spm) / spm + 0.3
    y = np.full(x.shape, 5.7)
    v = np.zeros(x.shape + (2,))
    sl = W.detail_slope(x, y, v, np.full(x.shape, 0.3), 0.0, _frame(s), s)
    for c in (0, 1):
        assert _max_lag_corr(sl[:, c], spm) < 0.5
    per = int(1.92 * spm)
    tiled = np.tile(sl[:per, 0], len(x) // per + 1)[:len(x)]
    assert _max_lag_corr(tiled, spm) > 0.95


def test_the_lattice_cells_decorrelate_distant_patches(settings, monkeypatch):
    """Property (W1b, Mikkelsen on the wave set): two patches one lattice
    vector apart -- the same place in their cells -- are unrelated (their
    slopes correlate under 0.3), because each cell turns and shifts the sets
    by its own hashes. Not vacuous: with every hash 0 the lattice is a
    plain periodic copy and the two patches are the same field (> 0.99).
    Breaks if the turn and the shift stop varying per vertex."""
    s = settings
    cell = s.wave_cell_m
    n = 60
    y, x = np.mgrid[0:n, 0:n] * (cell * 0.5 / n)
    x, y = x + 3 * cell, y + 3 * cell
    lat = np.array([cell, 0.0]) * 4.0          # four vertices along x
    v = np.zeros(x.shape + (2,))
    d = np.full(x.shape, 0.3)

    def corr():
        a = W.detail_slope(x, y, v, d, 0.0, _frame(s), s)
        b = W.detail_slope(x + lat[0], y + lat[1], v, d, 0.0, _frame(s), s)
        return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    assert abs(corr()) < 0.3
    monkeypatch.setattr(W, "hex_hash01",
                        lambda vx, vy, salt: np.zeros(np.shape(vx)))
    assert corr() > 0.99


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
