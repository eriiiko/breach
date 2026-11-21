"""PROTOTYPE (#97 reassessment, branch 97-reassess): the caustic in SCATTER
form and the Snell floor bend -- properties of the twin
(``renderer/water_surface.py``: ``scatter_light``, ``scatter_disp``,
``scatter_gain``, ``floor_offset_snell``) and the settings door's two string
dials. The GL face (``shaders/water_scatter.vs/.fs``) mirrors
``scatter_light``'s grid, offset, split and cap.

Each test names the property it protects and the change that must break it.
No GL, no sim.
"""
from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import numpy as np
import pytest

from renderer import water_surface as W


@pytest.fixture(scope="module")
def settings():
    from config import CFG
    return W.water_settings(CFG)


def _periodic_surface(n, amp):
    """One period of a smooth surface on an n x n grid of cells, as the
    refraction map's move in CELLS (the gradient of a potential, so it is a
    real surface's map) and the gather's exact det J at the same samples."""
    x = np.arange(n) + W.SCATTER_OFFSET
    X, Y = np.meshgrid(x, x)
    k1, k2 = 2 * np.pi * 3 / n, 2 * np.pi * 2 / n
    p1, p2 = k1 * X + 0.3 * k2 * Y, k2 * Y + 1.0
    # potential phi = amp/k1 sin(p1) + amp/k2 sin(p2): move = grad phi
    dx = amp * np.cos(p1)
    dy = amp * 0.3 * k2 / k1 * np.cos(p1) + amp * np.cos(p2)
    hxx = -amp * k1 * np.sin(p1)
    hxy = -amp * 0.3 * k2 * np.sin(p1)
    hyy = -amp * (0.3 * k2) ** 2 / k1 * np.sin(p1) - amp * k2 * np.sin(p2)
    det = (1 + hxx) * (1 + hyy) - hxy * hxy
    return np.stack([dx, dy], -1), det


def test_a_flat_surface_scatters_exactly_the_incident_light(settings):
    """Property: with no slope every cell receives exactly the incident light
    (1.0, bit for bit) and the gain is exactly 0 -- the scatter leaves a
    flat pool's floor untouched. Breaks if the sample sits at the cell
    centre or the quad is split through it (a centre on a shared edge is
    counted twice or never), or if a triangle's area ratio is not exactly 1
    when nothing moved."""
    for periodic, shape in ((True, (32, 32)), (False, (9, 13))):
        img = W.scatter_light(np.zeros(shape + (2,)), periodic=periodic)
        assert np.all(img == 1.0)
        assert np.all(W.scatter_gain(img, settings) == 0.0)


def test_the_scatter_conserves_light_over_a_period():
    """Property: over one period of a surface the scattered light's mean is
    the incident light, below the first focus (within 2 %) and past it
    (within 5 %: the rasteriser point-samples the bright fold lines; the
    folded triangles carry 20-46 % of the light here) -- where the gather's
    per-sample 1/|det J| is biased bright (W4 finding 1). Breaks if a
    turned-over (folded) triangle is dropped (a back-face cull), if the
    ratio is inverted (after / before), or if light is deposited twice."""
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


def test_the_scatter_brightens_where_the_gather_map_focuses():
    """Property: the scatter puts light where the refraction map squeezes it
    -- below the first focus its image follows the gather's 1/det J sample
    for sample (correlation > 0.95). Breaks if the move's sign flips (light
    pushed off the crests, the crests dark) or the move ignores depth."""
    move, det = _periodic_surface(128, 2.0)
    assert det.min() > 0.5
    img = W.scatter_light(move, periodic=True)
    c = np.corrcoef(img.ravel(), (1.0 / det).ravel())[0, 1]
    assert c > 0.95, c
    flipped = W.scatter_light(-move, periodic=True)
    assert np.corrcoef(flipped.ravel(), (1.0 / det).ravel())[0, 1] < 0.0


def test_the_scatter_moves_by_the_same_map_as_the_gather_and_the_floor():
    """Property: the scatter's move, the gather's map and the Snell floor
    bend are ONE map, ``depth (1 - 1/n) grad h``: the floor seen through a
    point is where that point's light lands, and deeper water moves it
    proportionally. Breaks if the floor bend or the scatter takes another
    sign or factor than CAUSTIC_K."""
    rng = np.random.default_rng(7)
    slope = rng.normal(0.0, 0.1, (20, 2))
    depth = rng.uniform(0.0, 1.5, 20)
    cell = 1.0 / 72.0
    assert np.allclose(W.scatter_disp(slope, depth, cell) * cell,
                       W.floor_offset_snell(slope, depth))
    assert np.allclose(W.floor_offset_snell(slope, depth),
                       slope * depth[:, None] * W.CAUSTIC_K)


def test_light_moved_off_the_image_is_lost_not_folded_back():
    """Property: on a bounded image (the view's water) light the map moves
    past the edge is lost, never wrapped onto the far side. Breaks if the
    non-periodic image indexes modulo its size."""
    move = np.zeros((16, 16, 2))
    move[:, :4, 0] = -6.0                # the first four columns leave west
    img = W.scatter_light(move, periodic=False)
    assert img.sum() < 16 * 16 - 16 * 3
    assert np.all(img[:, 8:] <= 1.0 + 1e-12)


def test_the_door_holds_the_prototype_dials(settings):
    """Property: ``caustic_form`` and ``floor_bend`` take only their named
    values and a wrong one is refused BY NAME (the one refusing door).
    Breaks if a mistyped form falls back to a default silently."""
    base = {k: getattr(settings, k) for k in W.config_keys()}
    for name, bad in (("caustic_form", "scattered"), ("caustic_form", 1),
                      ("floor_bend", "Snell"), ("floor_bend", None)):
        d = dict(base, **{name: bad})
        with pytest.raises(ValueError, match=name):
            W.water_settings(SimpleNamespace(graphics=SimpleNamespace(
                water=SimpleNamespace(**d))))
    for form in W.CAUSTIC_FORMS:
        for bend in W.FLOOR_BENDS:
            d = dict(base, caustic_form=form, floor_bend=bend)
            s = W.water_settings(SimpleNamespace(graphics=SimpleNamespace(
                water=SimpleNamespace(**d))))
            assert (s.caustic_form, s.floor_bend) == (form, bend)
    assert dataclasses.replace(settings, caustic_form="scatter")


# ---------------------------------------------------------------------------
# PROTOTYPE (#97 reassessment, item 3): the lamps' own mirror glints
# ---------------------------------------------------------------------------

def test_flat_water_mirrors_a_lamp_straight_under_itself(settings):
    """Property: on flat water seen from straight above, a lamp's glint is
    brightest exactly under the lamp and falls off with the distance from
    that point, and is gone a few lamp heights away -- the mirror image of
    a ceiling lamp (what the field's one elevation, light_z, cannot give).
    Breaks if the lamp's direction is taken from the field (one elevation:
    no peak), or the lamp's height is dropped."""
    s = dataclasses.replace(settings, lamp_glint=1.0)
    lamp = np.array([[5.0, 5.0, 1.0]])
    xs = np.linspace(5.0, 5.0 + 3 * s.lamp_height_m, 61)
    p = np.stack([xs, np.full_like(xs, 5.0)], -1)
    N = np.tile([0.0, 0.0, 1.0], (xs.size, 1))
    g = W.lamp_glint(p, N, lamp, s, s.roughness_base)
    assert g[0] == g.max() and g[0] > 1.0
    assert np.all(np.diff(g) <= 1e-12)
    assert g[-1] < 1e-3 * g[0]


def test_a_tilted_ripple_throws_the_lamp_off_its_nadir(settings):
    """Property: a ripple tilted toward the lamp by half the lamp's angle
    from vertical mirrors the lamp where flat water would not (the glitter
    out to ~2 x height x slope). Breaks if the normal does not enter the
    glint (a flat-water disc)."""
    s = dataclasses.replace(settings, lamp_glint=1.0)
    lamp = np.array([[0.0, 0.0, 1.0]])
    d = 0.6
    half = np.arctan(d / s.lamp_height_m) / 2.0
    p = np.array([[d, 0.0]])
    tilted = np.array([[-np.sin(half), 0.0, np.cos(half)]])
    flat = np.array([[0.0, 0.0, 1.0]])
    gt = W.lamp_glint(p, tilted, lamp, s, s.roughness_base)[0]
    gf = W.lamp_glint(p, flat, lamp, s, s.roughness_base)[0]
    assert gt > 10.0 * gf


def test_no_lamp_glint_when_off_or_without_lamps(settings):
    """Property: the dial 0 (the shipped look) or no omni lamp gives exactly
    0, and only omni LAMPS enter (a beam would need its cone). Breaks if the
    term leaks into the default look, or a beacon or the flashlight is taken
    as an omni lamp."""
    p = np.zeros((4, 2))
    N = np.tile([0.0, 0.0, 1.0], (4, 1))
    off = dataclasses.replace(settings, lamp_glint=0.0)
    assert np.all(W.lamp_glint(p, N, [[0.0, 0.0, 1.0]], off, 0.08) == 0.0)
    on = dataclasses.replace(settings, lamp_glint=1.0)
    assert np.all(W.lamp_glint(p, N, np.zeros((0, 3)), on, 0.08) == 0.0)
    specs = [SimpleNamespace(x=3.2, y=4.7, intensity=2.0, kind="lamp"),
             SimpleNamespace(x=1.0, y=1.0, intensity=5.0, kind="beacon"),
             SimpleNamespace(x=2.0, y=2.0, intensity=5.0, kind="flashlight")]
    rows = W.lamp_rows(specs, None, 0.5)
    assert rows.shape == (1, 3)
    assert np.allclose(rows[0], [3.5 * 0.5, 4.5 * 0.5, 2.0])
