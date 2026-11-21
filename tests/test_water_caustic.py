"""Water W4 (#97), landed form of 2026-11-21 (Erik: "I'd like scatter,
Snell, lamp glint"): THE REFRACTION MAP's two terms -- the caustic in
Wallace's scatter form and the Snell floor bend -- as properties of the
law's numpy twin (``renderer/water_surface.py``: ``ripple_slope``,
``floor_offset``, ``scatter_light``, ``scatter_disp``, ``caustic_gain``,
``caustic_term``) and of the settings door. The shader and the scatter
passes are held to the twin by ``tests/gl_water_caustic_check.py`` (wrapper
``tests/test_gl_water_caustic.py``).

Each test names the property it protects and the change that must break it.
No GL, no sim.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from renderer import water_surface as W

ROOT = Path(__file__).resolve().parent.parent
TILE_M = 1.0 / 3.0
CELL_M = TILE_M / W.SCATTER_CELLS_PER_TILE


@pytest.fixture(scope="module")
def settings():
    from config import CFG
    return W.water_settings(CFG)


def _periodic_surface(n, amp):
    """One period of a smooth surface on an n x n grid of cells, as the
    refraction map's move in CELLS (the gradient of a potential, so it is a
    real surface's map) and its exact det J at the same samples."""
    x = np.arange(n) + W.SCATTER_OFFSET
    X, Y = np.meshgrid(x, x)
    k1, k2 = 2 * np.pi * 3 / n, 2 * np.pi * 2 / n
    p1, p2 = k1 * X + 0.3 * k2 * Y, k2 * Y + 1.0
    dx = amp * np.cos(p1)
    dy = amp * 0.3 * k2 / k1 * np.cos(p1) + amp * np.cos(p2)
    hxx = -amp * k1 * np.sin(p1)
    hxy = -amp * 0.3 * k2 * np.sin(p1)
    hyy = -amp * (0.3 * k2) ** 2 / k1 * np.sin(p1) - amp * k2 * np.sin(p2)
    det = (1 + hxx) * (1 + hyy) - hxy * hxy
    return np.stack([dx, dy], -1), det


def _shipped_periodic_slope(s, still: bool, period_m=2.5, t=10.0):
    """The shipped surface made exactly periodic: both layers' wave sets
    (their slope amplitudes, phases and Finch sharpness, at the shipped
    still or flow strengths) with each wave vector snapped to the period's
    lattice, sampled at the scatter's own grid (``SCATTER_OFFSET`` in each
    cell of ``CELL_M``-ish size) -> (slope (n, n, 2), cell m)."""
    n = int(round(period_m / CELL_M))
    cell = period_m / n
    x = (np.arange(n) + W.SCATTER_OFFSET) * cell
    X, Y = np.meshgrid(x, x)
    q = np.stack([X, Y], -1)
    K = s.wave_sharpness if still else s.wave_sharpness_flow
    ks = ((s.detail_strength_still, s.swell_strength_still) if still
          else (s.detail_strength_flow, s.swell_strength_flow))
    sl = np.zeros(q.shape)
    for L, k in zip(W.LAYERS, ks):
        tab = W.wave_tables(s, t)[L]
        kk = np.round(tab.k * period_m / (2 * np.pi)) * (2 * np.pi / period_m)
        _h, g = W.wave_sum(W.WaveTable(kk, tab.phase, tab.amp), q,
                           np.full(X.shape, K))
        sl += g * k
    return sl, cell


# ---------------------------------------------------------------------------
# The caustic (Wallace's scatter)
# ---------------------------------------------------------------------------

def test_a_flat_surface_scatters_exactly_the_incident_light(settings):
    """Property: with no slope every cell receives exactly the incident light
    (1.0, bit for bit), the gain is exactly 0 and the term exactly 0 under
    any light -- the scatter leaves a flat pool's floor untouched. Breaks if
    the sample sits at the cell centre or the quad is split through it (a
    centre on a shared edge counted twice or never), or a triangle's ratio
    is not exactly 1 when nothing moved."""
    for periodic, shape in ((True, (32, 32)), (False, (9, 13))):
        img = W.scatter_light(np.zeros(shape + (2,)), periodic=periodic)
        assert np.all(img == 1.0)
        assert np.all(W.caustic_gain(img, settings) == 0.0)
        light = np.full(shape + (3,), 3.7)
        assert np.all(W.caustic_term(light, img, settings) == 0.0)


@pytest.mark.parametrize("still", [True, False], ids=["still", "flow"])
def test_the_shipped_surface_conserves_light_in_the_deepest_shot_water(
        settings, still):
    """Property: over one period of the shipped surface (its wave sets at
    the shipped still or full-flow strengths, made periodic), at the
    deepest water the shots use (the tank's 1.2 m), the scattered light's
    mean is the incident light within 2 % -- the caustic moves light, it
    makes or loses none (measured 1.004 still, 1.006 flow; the tolerance is
    the rasteriser's point-sampling of the fold slivers, ~1 %). The surface
    focuses well past the first fold there (cells above 20x), so the
    property is not vacuous; the old per-cell cap at 3 lost 18 % (still) of
    the same light. Breaks if a cap on the light image comes back, a folded
    triangle is culled, the ratio is inverted, light is deposited twice, or
    SCATTER_RATIO_MAX is lowered into the folds' range."""
    sl, cell = _shipped_periodic_slope(settings, still)
    disp = W.scatter_disp(sl, np.full(sl.shape[:-1], 1.2), cell)
    img = W.scatter_light(disp, periodic=True)
    assert abs(img.mean() - 1.0) < 0.02, img.mean()
    assert img.max() > 20.0, img.max()                  # past the first fold
    s = dataclasses.replace(settings, caustic_strength=1.0)
    g = W.caustic_gain(img, s)
    assert abs(g.mean()) < 0.02 and g.min() >= -1.0
    if still:
        assert np.minimum(img, 3.0).mean() < 0.9        # the old cap's loss


def _periodic_bilinear(img, u, v):
    """Bilinear read of a periodic image at cell coordinates (centres at
    k + 1/2)."""
    n_y, n_x = img.shape
    u = (u - 0.5) % n_x
    v = (v - 0.5) % n_y
    i0 = np.floor(u).astype(int)
    j0 = np.floor(v).astype(int)
    fu, fv = u - i0, v - j0
    i1, j1 = (i0 + 1) % n_x, (j0 + 1) % n_y
    return (img[j0, i0] * (1 - fu) * (1 - fv) + img[j0, i1] * fu * (1 - fv)
            + img[j1, i0] * (1 - fu) * fv + img[j1, i1] * fu * fv)


def test_the_picture_keeps_the_light_only_read_at_the_pixel(settings):
    """Property: the caustic is read at the pixel's OWN floor point -- there
    the picture's mean is the light image's (the incident light, within
    2 %) -- and never through the floor bend: read where the bend sends the
    view ray, the view looks through the very lens that made the pattern
    (the scatter's map is the vertical light's path) and the picture's mean
    is the gather's bias again (measured 1.97 at the tank's 1.2 m, still
    water). Breaks (with the GL check's floor read) if the draw reads the
    image through the bend, or the scatter stops conserving."""
    sl, cell = _shipped_periodic_slope(settings, True)
    disp = W.scatter_disp(sl, np.full(sl.shape[:-1], 1.2), cell)
    img = W.scatter_light(disp, periodic=True)
    n = img.shape[0]
    jj, ii = np.mgrid[0:n, 0:n] + W.SCATTER_OFFSET
    at_pixel = _periodic_bilinear(img, ii, jj)
    through_bend = _periodic_bilinear(img, ii + disp[..., 0],
                                      jj + disp[..., 1])
    assert abs(at_pixel.mean() - 1.0) < 0.02, at_pixel.mean()
    assert through_bend.mean() > 1.5, through_bend.mean()


def test_the_scatter_conserves_light_over_a_period_below_and_past_a_fold():
    """Property: over one period of a synthetic surface the scattered light's
    mean is the incident light, below the first focus (within 2 %) and past
    it (within 5 %: the folded triangles carry 20-46 % of the light here)
    -- where the gather's per-sample ``1/|det J|`` was biased bright (W4
    finding 1). Breaks if a turned-over (folded) triangle is dropped (a
    back-face cull), the ratio inverted (after / before), or light
    deposited twice."""
    n = 128
    for amp, folds in ((1.0, False), (4.0, False), (8.0, True), (16.0, True)):
        move, det = _periodic_surface(n, amp)
        assert bool((det < 0).any()) == folds
        img = W.scatter_light(move, periodic=True)
        tol = 0.05 if folds else 0.02
        assert abs(img.mean() - 1.0) < tol, (amp, img.mean())
        if folds:
            assert np.mean(1.0 / np.abs(det)) > 1.2     # the gather's bias
            assert img.max() > 10.0                     # a bright fold line


def test_the_scatter_brightens_where_the_map_focuses():
    """Property: the scatter puts light where the refraction map squeezes it
    -- below the first focus its image follows ``1/det J`` sample for sample
    (correlation > 0.95). Breaks if the move's sign flips (light pushed off
    the focus, the crests dark)."""
    move, det = _periodic_surface(128, 2.0)
    assert det.min() > 0.5
    img = W.scatter_light(move, periodic=True)
    assert np.corrcoef(img.ravel(), (1.0 / det).ravel())[0, 1] > 0.95
    flipped = W.scatter_light(-move, periodic=True)
    assert np.corrcoef(flipped.ravel(), (1.0 / det).ravel())[0, 1] < 0.0


def test_deeper_water_focuses_more(settings):
    """Property: for the same surface the caustic's contrast grows with the
    depth (depth enters only through the map's ``a = depth x CAUSTIC_K``),
    proportionally while small. Breaks if depth stops entering the map or
    enters it twice."""
    sl, cell = _shipped_periodic_slope(settings, True)
    amp = []
    for d in (0.02, 0.05, 0.1):
        img = W.scatter_light(W.scatter_disp(sl, np.full(sl.shape[:-1], d),
                                             cell), periodic=True)
        amp.append(float(np.abs(img - 1.0).mean()))
    assert amp[0] < amp[1] < amp[2], amp
    assert 2.5 * 0.75 < amp[1] / amp[0] < 2.5 * 1.25, amp


def test_no_light_no_caustic(settings):
    """Property: the caustic is the floor's light, focused or spread -- with
    no light there is none, however strongly the surface focuses. Breaks if
    the term adds light of its own or modulates the ambient."""
    move, _det = _periodic_surface(64, 8.0)
    img = W.scatter_light(move, periodic=True)
    assert np.abs(W.caustic_gain(img, settings)).max() > 0.5   # not vacuous
    assert np.all(W.caustic_term(np.zeros(img.shape + (3,)), img,
                                 settings) == 0.0)


def test_light_moved_off_the_image_is_lost_not_folded_back():
    """Property: on a bounded image (the drawn water) light the map moves
    past the edge is lost, never wrapped onto the far side. Breaks if the
    non-periodic image indexes modulo its size."""
    move = np.zeros((16, 16, 2))
    move[:, :4, 0] = -6.0                # the first four columns leave west
    img = W.scatter_light(move, periodic=False)
    assert img.sum() < 16 * 16 - 16 * 3
    assert np.all(img[:, 8:] <= 1.0 + 1e-12)


# ---------------------------------------------------------------------------
# The floor bend (Snell) and the one map
# ---------------------------------------------------------------------------

def test_the_floor_bend_and_the_caustic_move_by_one_map():
    """Property: the floor bend and the scatter's move are ONE map, ``depth
    (1 - 1/n) grad h`` ALONG the slope (Snell's vector law at small slopes:
    a vertical ray leaves toward the uphill side): the floor seen through a
    point is where that point's light lands, deeper water moves it
    proportionally, no depth or no slope none. Breaks if the bend takes the
    legacy sign (``-N.xy``), a map-UV length, or another factor than
    CAUSTIC_K, or the scatter another map than the bend."""
    rng = np.random.default_rng(7)
    slope = rng.normal(0.0, 0.1, (20, 2))
    depth = rng.uniform(0.0, 1.5, 20)
    b = W.floor_offset(slope, depth)
    assert np.allclose(b, slope * depth[:, None] * (1.0 - 1.0 / 1.333))
    assert np.allclose(W.scatter_disp(slope, depth, CELL_M) * CELL_M, b)
    assert np.all((b * slope).sum(-1) >= 0.0)            # along the slope
    assert np.all(W.floor_offset(slope, np.zeros(20)) == 0.0)
    assert np.all(W.floor_offset(np.zeros((20, 2)), depth) == 0.0)


def test_the_ripple_slope_is_the_normals_central_difference(settings):
    """Property: the tile ripple's share of the slope is the shader's normal
    term -- the bilinear ripple one tile either side, halved, x
    ``ripple_scale`` -- so on a plane ripple it is the plane's gradient per
    tile x ``ripple_scale`` (exact in the interior), and 0 on a still one.
    Breaks if the slope the bend and the caustic read leaves the tile
    ripple out or scales it otherwise than the normal does."""
    h, w = 8, 10
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    gx, gy = 0.003, -0.002
    plane = gx * xx + gy * yy
    x = np.array([3.2, 5.7, 4.0])
    y = np.array([2.5, 4.1, 3.3])
    sl = W.ripple_slope(plane, x, y, settings)
    assert np.allclose(sl, [[gx * settings.ripple_scale,
                             gy * settings.ripple_scale]] * 3)
    assert np.all(W.ripple_slope(np.zeros((h, w)), x, y, settings) == 0.0)


# ---------------------------------------------------------------------------
# The door and the deletions
# ---------------------------------------------------------------------------

def test_the_door_holds_the_caustic_dial_and_the_losing_forms_are_gone(
        settings):
    """Property: ``caustic_strength`` must not be negative, refused by name;
    the losing forms' dials (``refract_strength``, ``caustic_max``,
    ``caustic_form``, ``floor_bend``) are gone from the config, the door and
    the shader, and so is the gather's surface Hessian (no curvature in the
    wave law, the shader or the twin) -- one caustic, one bend, no parallel
    look path. Breaks if a dead dial or a second form comes back."""
    from config import CFG
    base = dict(vars(CFG.graphics.water))
    d = dict(base, caustic_strength=-0.1)
    with pytest.raises(ValueError, match="caustic_strength"):
        W.water_settings(SimpleNamespace(graphics=SimpleNamespace(
            water=SimpleNamespace(**d))))
    gone = ("refract_strength", "caustic_max", "caustic_form", "floor_bend",
            "caustic_scale")
    for k in gone:
        assert k not in base and k not in W.config_keys()
    for name in ("caustic_det", "turn_hessian"):
        assert not hasattr(W, name)
    fs = (ROOT / "shaders" / "water.fs").read_text(encoding="utf-8")
    code = "\n".join(ln.split("//")[0] for ln in fs.splitlines())
    for token in ("u_refract_strength", "u_floor_bend", "causticDet",
                  "hess", "u_time", "u_caustic_scale"):
        assert token not in code, token
    assert "u_caustic_img" in code and "u_waves" in code
    assert "caustic_strength" in base
    assert settings.caustic_strength >= 0.0
