"""Water W4 (#97): the sharp caustic -- properties of the law's numpy twin
(``renderer/water_surface.py``: ``hex_surface``, ``detail_surface``,
``caustic_det``, ``caustic_gain``, ``caustic_term``) and of the ripple's
curvature bake (``renderer/noise_bake.py::bake_ripple_curvature_rgba``).
The shader is held to the twin by ``tests/gl_water_caustic_check.py``
(wrapper ``tests/test_gl_water_caustic.py``).

Each test names the property it protects and the change that must break it.
No GL, no sim.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from renderer import water_surface as W
from renderer.noise_bake import (RIPPLE_CELLS, bake_ripple_curvature_rgba,
                                 bake_ripple_rgba, ripple_field)

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def tex():
    return bake_ripple_rgba(256)


@pytest.fixture(scope="module")
def curv():
    return bake_ripple_curvature_rgba(256)


@pytest.fixture(scope="module")
def settings():
    from config import CFG
    return W.water_settings(CFG)


def _frame(s, t=12.34):
    return W.flow_frame(t, 24.0, 64.0, s)


def _still(s, k):
    """Settings whose fine surface has RMS slope ``k`` everywhere."""
    return dataclasses.replace(s, detail_strength_still=k,
                               detail_strength_flow=k)


def _period_hessian(curv, s, k):
    """The Hessian (1/m) of ONE full period of the un-tiled surface at RMS
    slope ``k``: the curvature bake's texels (the exact periodic sums of
    the waves) scaled by k / P, the layer law's scale."""
    return curv[..., :3].astype(np.float64) * (k / s.period_m)


# ---------------------------------------------------------------------------
# The bake
# ---------------------------------------------------------------------------

def test_the_curvature_bake_is_the_slope_bakes_exact_derivative():
    """Property: the curvature channels are the analytic second derivatives
    of the SAME waves, in the slope bake's scaling: a central difference of
    the slope field equals them, and the bake's texels equal the field's
    values divided by the slope RMS. Breaks if the bake differences a
    texture, uses another seed / band, or another normalisation."""
    rng = np.random.default_rng(4)
    st = rng.random((500, 2))
    e = 1e-5
    _h, hu, hv, huu, huv, hvv = ripple_field(st, second=True)
    _, hu_p, hv_p = ripple_field(st + [e, 0.0])
    _, hu_m, hv_m = ripple_field(st - [e, 0.0])
    _, _, hv_pv = ripple_field(st + [0.0, e])
    _, _, hv_mv = ripple_field(st - [0.0, e])
    scale = np.abs(huu).max()
    assert np.abs((hu_p - hu_m) / (2 * e) - huu).max() < 1e-4 * scale
    assert np.abs((hv_p - hv_m) / (2 * e) - huv).max() < 1e-4 * scale
    assert np.abs((hv_pv - hv_mv) / (2 * e) - hvv).max() < 1e-4 * scale
    n = 64
    slope = bake_ripple_rgba(n).astype(np.float64)
    cb = bake_ripple_curvature_rgba(n).astype(np.float64)
    yy, xx = np.mgrid[0:n, 0:n]
    uv = np.stack([xx / n, yy / n], -1)       # the bake's texel coords
    _h, hu, hv, huu, huv, hvv = ripple_field(uv, second=True)
    rms = np.sqrt(np.mean(hu * hu + hv * hv))
    assert np.allclose(slope[..., 0], hu / rms, atol=1e-5)
    for ch, ref in enumerate((huu, huv, hvv)):
        assert np.allclose(cb[..., ch], ref / rms, rtol=1e-5,
                           atol=1e-4), ch
    assert not cb[..., 3].any()


# ---------------------------------------------------------------------------
# The derivatives through the hex blend and the flow map
# ---------------------------------------------------------------------------

def _dominant_points(n, rng, wmin=0.93):
    """Texture points where one hex tap's barycentric weight is >= wmin, so
    its ``w^7`` weight carries the blend (the weights' own gradient, which
    the law drops on purpose, vanishes there)."""
    pts = []
    while len(pts) < n:
        st = rng.random((4 * n, 2)) * 3.0 + 1.0
        w, _ = W.triangle_grid(st)
        pts.extend(st[w.max(-1) >= wmin])
    return np.array(pts[:n])


def _numeric_hessian(lookup, st, e):
    """The derivative of a slope lookup by central differences ->
    (hxx, hxy, hyy) with hxy averaged over its two orders."""
    sx_p = lookup(st + [e, 0.0])
    sx_m = lookup(st - [e, 0.0])
    sy_p = lookup(st + [0.0, e])
    sy_m = lookup(st - [0.0, e])
    hxx = (sx_p[..., 0] - sx_m[..., 0]) / (2 * e)
    hyy = (sy_p[..., 1] - sy_m[..., 1]) / (2 * e)
    hxy = 0.5 * ((sx_p[..., 1] - sx_m[..., 1])
                 + (sy_p[..., 0] - sy_m[..., 0])) / (2 * e)
    return np.stack([hxx, hxy, hyy], -1)


def test_the_hex_taps_rotate_the_tensor_back():
    """Property: where one hex tap carries the blend, ``hex_surface``'s
    Hessian is the derivative of ``hex_slope`` -- each tap's baked tensor
    rotated back on both indices (``R^T Hf R``). Measured on a fine bake
    (1024 texels, so the bilinear slope differences well); the unrotated
    tensor is measured too and must be far worse (the test is not vacuous).
    Breaks if the tensor is not rotated, rotated one way only, or read at
    another point than its slope tap."""
    n = 1024
    tex = bake_ripple_rgba(n)
    curv = bake_ripple_curvature_rgba(n)
    st = _dominant_points(300, np.random.default_rng(7))
    _s, hess = W.hex_surface(tex, curv, st)
    num = _numeric_hessian(lambda q: W.hex_slope(tex, q), st, 2.0 / n)
    ref = np.sqrt((num ** 2).sum(-1))
    err = np.sqrt(((hess - num) ** 2).sum(-1)) / np.maximum(ref, 1e-9)
    assert float(np.median(err)) < 0.05, np.median(err)
    # the same blend with the tensor NOT rotated back
    w, verts = W.triangle_grid(st)
    i = np.argmax(w, -1)
    v = np.take_along_axis(np.stack(verts, -2), i[:, None, None], -2)[:, 0]
    cen = W.hex_center(v)
    th = 2.0 * np.pi * (W.hex_hash01(v[:, 0], v[:, 1], 3) - 0.5)
    c, sn = np.cos(th), np.sin(th)
    d = st - cen
    q = np.stack([c * d[:, 0] - sn * d[:, 1] + cen[:, 0]
                  + W.hex_hash01(v[:, 0], v[:, 1], 1),
                  sn * d[:, 0] + c * d[:, 1] + cen[:, 1]
                  + W.hex_hash01(v[:, 0], v[:, 1], 2)], -1)
    raw = W.sample_bilinear_repeat(curv, q)[..., :3]
    bad = np.sqrt(((raw - num) ** 2).sum(-1)) / np.maximum(ref, 1e-9)
    assert float(np.median(bad)) > 0.3, np.median(bad)


def test_the_surface_hessian_is_the_slopes_derivative_in_metres(
        settings):
    """Property: ``detail_surface`` returns ``detail_slope``'s slope bit for
    bit (the caustic reads the ONE fine surface), and its Hessian is that
    slope's derivative in METRES (``k / P`` x the texture's), where one hex
    tap carries the blend (fine 1024-texel bakes, so the bilinear slope
    differences well over two texels). Breaks if the Hessian forgets the
    period or the strength, or the caustic grows a surface of its own."""
    n = 1024
    tex = bake_ripple_rgba(n)
    curv = bake_ripple_curvature_rgba(n)
    s = _still(settings, 0.2)
    fr = _frame(s)
    rng = np.random.default_rng(11)
    # dominant points in the layer coordinates of a still surface: the
    # still layers read q = p / P - drift (no flow, no slide)
    st = _dominant_points(200, rng)
    x = (st[:, 0] + fr.drift_u) * s.period_m
    y = (st[:, 1] + fr.drift_v) * s.period_m
    vel = np.zeros(x.shape + (2,))
    depth = np.full(x.shape, 0.3)
    ag = np.zeros(x.shape)
    slope, hess = W.detail_surface(x, y, vel, depth, ag, fr, tex, curv, s)
    assert np.array_equal(slope, W.detail_slope(x, y, vel, depth, ag, fr,
                                                tex, s))
    e = 2.0 / n * s.period_m

    def lookup(p):
        return W.detail_slope(p[..., 0], p[..., 1], vel, depth, ag, fr,
                              tex, s)
    num = _numeric_hessian(lookup, np.stack([x, y], -1), e)
    ref = np.sqrt((num ** 2).sum(-1))
    err = np.sqrt(((hess - num) ** 2).sum(-1)) / np.maximum(ref, 1e-9)
    assert float(np.median(err)) < 0.05, np.median(err)
    # the Hessian in texture units (no k / P) would be far off
    raw = hess * s.period_m / 0.2
    bad = np.sqrt(((raw - num) ** 2).sum(-1)) / np.maximum(ref, 1e-9)
    assert float(np.median(bad)) > 0.5, np.median(bad)


# ---------------------------------------------------------------------------
# The caustic
# ---------------------------------------------------------------------------

def test_a_flat_surface_makes_exactly_no_caustic(tex, curv, settings):
    """Property: a flat fine surface (zero slope: both strengths 0, or
    zoomed out) gives det J == 1 and a caustic of EXACTLY 0 (==, not
    close), whatever the light and the depth. Breaks if the term carries a
    bias, a time term or a read of the tile ripple."""
    s = _still(settings, 0.0)
    y, x = np.mgrid[0:40, 0:40] * 0.05
    vel = np.zeros(x.shape + (2,))
    _sl, hess = W.detail_surface(x, y, vel, np.full(x.shape, 0.8),
                                 np.zeros(x.shape), _frame(s), tex, curv, s)
    assert not hess.any()
    det = W.caustic_det(hess, np.full(x.shape, 0.8))
    assert np.all(det == 1.0)
    light = np.full(x.shape + (3,), 3.7)
    assert np.all(W.caustic_term(light, det, s) == 0.0)


def test_no_light_no_caustic(curv, settings):
    """Property: the caustic is the floor's light, focused or spread -- with
    no light there is none, however rough the surface. Breaks if the term
    adds light of its own or modulates the ambient."""
    H = _period_hessian(curv, settings, 0.3)
    det = W.caustic_det(H, 0.3)
    assert np.abs(W.caustic_gain(det, settings)).max() > 0.5   # not vacuous
    assert np.all(W.caustic_term(np.zeros(det.shape + (3,)), det,
                                 settings) == 0.0)


def test_the_signed_term_redistributes_to_second_order(curv, settings):
    """Property: over ONE full period of a small-amplitude surface the
    signed caustic's mean is ~0 -- it moves light, it does not make it.
    TOLERANCE AND WHY: the gather form evaluates 1/det at the surface point,
    and ``1/det - 1 = -a tr H + a^2 (tr H^2 - det H) + O(a^3)``; over a
    period ``<tr H> = 0`` and ``<det H> = 0`` exactly (the bake's texels are
    the exact periodic sums), so the mean is ``a^2 <tr H^2>`` -- SECOND order,
    the square of the term's own RMS (first order). Asserted: |mean| <=
    1.5 x rms^2, and halving the amplitude quarters the mean (3..5 x).
    Breaks if the term is one-signed (the old ``max(lap, 0)``), carries a
    first-order bias, or loses its sign."""
    s = dataclasses.replace(settings, caustic_strength=1.0,
                            caustic_max=50.0)
    out = []
    for k in (0.02, 0.01):
        g = W.caustic_gain(W.caustic_det(_period_hessian(curv, s, k), 0.3),
                           s)
        mean, rms = float(g.mean()), float(g.std())
        assert (g < 0).mean() > 0.3 and (g > 0).mean() > 0.3
        assert abs(mean) <= 1.5 * rms * rms, (k, mean, rms)
        assert abs(mean) < 0.1 * rms, (k, mean, rms)
        out.append(mean)
    assert 3.0 <= out[0] / out[1] <= 5.0, out


def test_the_term_never_exceeds_its_cap(curv, settings):
    """Property: the caustic never brightens the floor's light by more than
    ``(caustic_max - 1) x caustic_strength`` and never takes more than all of
    it, however hard the surface focuses (folds, deep water). Breaks if the
    clamp goes or is applied to det instead of 1/|det|."""
    s = dataclasses.replace(settings, caustic_strength=1.0, caustic_max=3.0)
    for k, depth in ((0.3, 0.3), (0.3, 1.2), (0.6, 2.0)):
        det = W.caustic_det(_period_hessian(curv, s, k), depth)
        assert (det < 0).any() and (np.abs(det) < 1.0 / 3.0).any()
        g = W.caustic_gain(det, s)
        assert g.max() <= 2.0 + 1e-12 and g.min() >= -1.0
        light = np.full(det.shape + (3,), 2.5)
        t = W.caustic_term(light, det, s)
        assert t.max() <= 2.0 * 2.5 + 1e-9


def test_past_a_focus_the_light_is_inverted_not_dark(settings):
    """Property: the gain reads |det|: past a focus (det < 0) the beam is
    inverted and carries E / |det|, so a fold is a bright line and the
    floor beyond it keeps its light. Breaks if ``clamp(1/det, 0, max)``
    blacks out every folded patch."""
    s = dataclasses.replace(settings, caustic_strength=1.0, caustic_max=4.0)
    d = np.array([0.1, 0.5, 1.0, 2.0, 5.0])
    assert np.array_equal(W.caustic_gain(-d, s), W.caustic_gain(d, s))
    assert W.caustic_gain(np.array([0.0]), s)[0] == pytest.approx(3.0)


def test_deeper_water_focuses_more(curv, settings):
    """Property: for the same small slope the caustic's contrast grows with
    depth (depth enters only through the Jacobian's ``a``), until the clamp
    holds it. Breaks if depth stops entering the Jacobian, or enters it
    twice (an extra "scaled by depth" would also grow it, but then the
    flat-surface and redistribution properties are what pin the form)."""
    s = dataclasses.replace(settings, caustic_strength=1.0, caustic_max=3.0)
    H = _period_hessian(curv, s, 0.05)
    amp = [float(np.abs(W.caustic_gain(W.caustic_det(H, dp), s)).mean())
           for dp in (0.02, 0.05, 0.1, 0.2, 0.4)]
    assert all(b > a for a, b in zip(amp, amp[1:])), amp
    # first order while small: the contrast is proportional to the depth
    assert 2.5 * 0.8 < amp[1] / amp[0] < 2.5 * 1.2, amp
    deep = W.caustic_gain(W.caustic_det(H, 50.0), s)
    assert deep.max() <= 2.0 + 1e-12 and deep.min() >= -1.0


def test_the_door_holds_the_caustic_dials(settings):
    """Property: ``caustic_max`` must exceed 1 and ``caustic_strength``
    must not be negative, refused by name; ``caustic_scale`` (the old
    procedural detail) is gone from the config, the door and the shader,
    and with it the shader's last ``u_time``. Breaks if a dead dial comes
    back or a bad one passes."""
    from types import SimpleNamespace
    from config import CFG
    base = dict(vars(CFG.graphics.water))
    for over, name in (({"caustic_max": 1.0}, "caustic_max"),
                       ({"caustic_strength": -0.1}, "caustic_strength")):
        d = dict(base, **over)
        with pytest.raises(ValueError, match=name):
            W.water_settings(SimpleNamespace(graphics=SimpleNamespace(
                water=SimpleNamespace(**d))))
    assert "caustic_scale" not in base
    assert "caustic_scale" not in W.config_keys()
    fs = (ROOT / "shaders" / "water.fs").read_text(encoding="utf-8")
    code = "\n".join(ln.split("//")[0] for ln in fs.splitlines())
    assert "u_time" not in code and "u_caustic_scale" not in code
    assert "u_caustic" in code and "u_ripple_curv" in code
