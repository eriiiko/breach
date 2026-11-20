"""Water W4 (#97): the sharp caustic -- properties of the law's numpy twin
(``renderer/water_surface.py``: ``hex_waves``, ``detail_surface``,
``caustic_det``, ``caustic_gain``, ``caustic_term``). Since W1b the
surface's Hessian is the EXACT second derivative of the procedural sines
(no curvature bake); the waves' own derivative identities are
``tests/test_water_flow_detail.py``'s. The shader is held to the twin by
``tests/gl_water_caustic_check.py`` (wrapper
``tests/test_gl_water_caustic.py``).

Each test names the property it protects and the change that must break it.
No GL, no sim.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from renderer import water_surface as W

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def settings():
    from config import CFG
    return W.water_settings(CFG)


def _frame(s, t=12.34):
    return W.flow_frame(t, 24.0, 64.0, s)


def _still(s, k, swell=0.0):
    """Settings whose ripples have RMS slope ``k`` everywhere (and the
    swell ``swell``)."""
    return dataclasses.replace(s, detail_strength_still=k,
                               detail_strength_flow=k,
                               swell_strength_still=swell,
                               swell_strength_flow=swell)


def _set_hessian(s, k, n=500, span=10.0):
    """The Hessian (1/m) of ONE wave set (the ripples, unblended, at RMS
    slope ``k``) over a ``span`` m square: the sum the lattice's taps read,
    away from any seam."""
    tab = _frame(s).waves[W.LAYER_DETAIL]
    y, x = np.mgrid[0:n, 0:n] * (span / n)
    q = np.stack([x, y], -1)
    _h, _sl, hs = W.wave_sum(tab, q, np.full(x.shape, s.wave_sharpness))
    return hs * k


# ---------------------------------------------------------------------------
# The derivatives through the hex blend and the flow map
# ---------------------------------------------------------------------------

def _dominant_points(n, rng, cell, wmin=0.93):
    """Points (metres) where one hex tap's barycentric weight is >= wmin, so
    it alone carries the blend (the others under ``HEX_SKIP``; the weights'
    own gradient, which the law drops on purpose, vanishes there)."""
    pts = []
    while len(pts) < n:
        p = rng.random((4 * n, 2)) * 3.0 * cell + cell
        w, _ = W.triangle_grid(p / (cell * W.HEX_SCALE))
        pts.extend(p[w.max(-1) >= wmin])
    return np.array(pts[:n])


def _numeric_hessian(lookup, p, e):
    """The derivative of a slope by central differences -> (hxx, hxy, hyy)
    with hxy averaged over its two orders."""
    sx_p = lookup(p + [e, 0.0])
    sx_m = lookup(p - [e, 0.0])
    sy_p = lookup(p + [0.0, e])
    sy_m = lookup(p - [0.0, e])
    hxx = (sx_p[..., 0] - sx_m[..., 0]) / (2 * e)
    hyy = (sy_p[..., 1] - sy_m[..., 1]) / (2 * e)
    hxy = 0.5 * ((sx_p[..., 1] - sx_m[..., 1])
                 + (sy_p[..., 0] - sy_m[..., 0])) / (2 * e)
    return np.stack([hxx, hxy, hyy], -1)


def test_the_surface_hessian_is_the_slopes_derivative_in_metres(settings):
    """Property: ``detail_surface`` returns ``detail_slope``'s slope bit for
    bit (the caustic reads the ONE fine surface), and its Hessian -- both
    layers, each tap's tensor turned back (``R H R^T``) and x ``k_L`` -- is
    that slope's derivative in metres where one hex tap carries the blend.
    The tensor left in the tap's own frame is far off, so the turn is what
    is held. Breaks if the Hessian forgets a layer, a strength or the turn,
    or the caustic grows a surface of its own."""
    s = _still(settings, 0.15, swell=0.12)
    fr = _frame(s)
    rng = np.random.default_rng(11)
    p = _dominant_points(300, rng, s.wave_cell_m)
    x, y = p[:, 0], p[:, 1]
    vel = np.zeros(x.shape + (2,))
    depth = np.full(x.shape, 0.3)
    ag = np.zeros(x.shape)
    slope, hess = W.detail_surface(x, y, vel, depth, ag, fr, s)
    assert np.array_equal(slope, W.detail_slope(x, y, vel, depth, ag, fr, s))

    def lookup(q):
        return W.detail_slope(q[..., 0], q[..., 1], vel, depth, ag, fr, s)
    num = _numeric_hessian(lookup, p, 1e-5)
    ref = np.sqrt((num ** 2).sum(-1))
    err = np.sqrt(((hess - num) ** 2).sum(-1)) / np.maximum(ref, 1e-9)
    assert float(np.median(err)) < 1e-3, np.median(err)
    assert float(np.percentile(err, 95)) < 1e-2

    # the same Hessian left unturned (each tap's own frame) is far off
    from renderer import water_surface as WS
    orig = WS.turn_hessian
    try:
        WS.turn_hessian = lambda hs, c, sn: hs
        _s, raw = W.detail_surface(x, y, vel, depth, ag, fr, s)
    finally:
        WS.turn_hessian = orig
    bad = np.sqrt(((raw - num) ** 2).sum(-1)) / np.maximum(ref, 1e-9)
    assert float(np.median(bad)) > 0.3, np.median(bad)


# ---------------------------------------------------------------------------
# The caustic
# ---------------------------------------------------------------------------

def test_a_flat_surface_makes_exactly_no_caustic(settings):
    """Property: a flat fine surface (every strength 0, or zoomed out) gives
    zero slope, det J == 1 and a caustic of EXACTLY 0 (==, not close),
    whatever the light and the depth. Breaks if the term carries a bias, a
    time term, a read of the tile ripple, or a layer that ignores its
    strength."""
    for s, fr in ((_still(settings, 0.0), None),
                  (settings, W.flow_frame(12.34, 24.0,
                                          settings.detail_zoom_zero,
                                          settings))):
        fr = fr or _frame(s)
        y, x = np.mgrid[0:40, 0:40] * 0.05
        vel = np.zeros(x.shape + (2,))
        sl, hess = W.detail_surface(x, y, vel, np.full(x.shape, 0.8),
                                    np.zeros(x.shape), fr, s)
        assert not sl.any() and not hess.any()
        det = W.caustic_det(hess, np.full(x.shape, 0.8))
        assert np.all(det == 1.0)
        light = np.full(x.shape + (3,), 3.7)
        assert np.all(W.caustic_term(light, det, s) == 0.0)


def test_no_light_no_caustic(settings):
    """Property: the caustic is the floor's light, focused or spread -- with
    no light there is none, however rough the surface. Breaks if the term
    adds light of its own or modulates the ambient."""
    H = _set_hessian(settings, 0.3, n=200)
    det = W.caustic_det(H, 0.3)
    assert np.abs(W.caustic_gain(det, settings)).max() > 0.5   # not vacuous
    assert np.all(W.caustic_term(np.zeros(det.shape + (3,)), det,
                                 settings) == 0.0)


def test_the_signed_term_redistributes_to_second_order(settings):
    """Property: over a large window of a small-amplitude surface the signed
    caustic's mean is ~0 -- it moves light, it does not make it. TOLERANCE
    AND WHY: the gather form evaluates 1/det at the surface point, and
    ``1/det - 1 = -a tr H + a^2 (tr H^2 - det H) + O(a^3)``; ``<tr H>`` is 0
    (each wave's ``g''`` averages to 0 over its period) and ``<det H>`` is 0
    (one wave's Hessian has rank 1; the cross terms of two waves average
    out), so the mean is ``a^2 <tr H^2>`` -- SECOND order, the square of the
    term's own RMS. Over a 10 m window of incommensurate waves those means
    are 0 to a few per cent of their RMS, so: |mean| <= 1.5 x rms^2 + 2 %
    of rms, and halving the amplitude quarters it (2.5..5.5 x). Breaks if
    the term is one-signed (the old ``max(lap, 0)``), carries a first-order
    bias, or loses its sign."""
    s = dataclasses.replace(settings, caustic_strength=1.0,
                            caustic_max=50.0, wave_sharpness=1.0)
    out = []
    for k in (0.02, 0.01):
        g = W.caustic_gain(W.caustic_det(_set_hessian(s, k), 0.3), s)
        mean, rms = float(g.mean()), float(g.std())
        assert (g < 0).mean() > 0.3 and (g > 0).mean() > 0.3
        assert abs(mean) <= 1.5 * rms * rms + 0.02 * rms, (k, mean, rms)
        out.append(mean)
    assert 2.5 <= out[0] / out[1] <= 5.5, out


def test_the_term_never_exceeds_its_cap(settings):
    """Property: the caustic never brightens the floor's light by more than
    ``(caustic_max - 1) x caustic_strength`` and never takes more than all of
    it, however hard the surface focuses (folds, deep water). Breaks if the
    clamp goes or is applied to det instead of 1/|det|."""
    s = dataclasses.replace(settings, caustic_strength=1.0, caustic_max=3.0)
    for k, depth in ((0.3, 0.3), (0.3, 1.2), (0.6, 2.0)):
        det = W.caustic_det(_set_hessian(s, k, n=200), depth)
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


def test_deeper_water_focuses_more(settings):
    """Property: for the same small slope the caustic's contrast grows with
    depth (depth enters only through the Jacobian's ``a``), until the clamp
    holds it. Breaks if depth stops entering the Jacobian, or enters it
    twice (an extra "scaled by depth" would also grow it, but then the
    flat-surface and redistribution properties are what pin the form)."""
    s = dataclasses.replace(settings, caustic_strength=1.0, caustic_max=3.0)
    H = _set_hessian(s, 0.05, n=200)
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
    and with it the shader's last ``u_time``; since W1b the shader reads no
    curvature texture (one wave law: the Hessian is the sines' own). Breaks
    if a dead dial or a second wave source comes back or a bad one
    passes."""
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
    assert "u_caustic" in code and "u_waves" in code
    assert "u_ripple_curv" not in code and "u_ripple;" not in code
