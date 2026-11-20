"""Water W2 (#97, docs/water_look_design_2026-11-19.md W2 + critique M9-M11):
the foam's law, asserted on its numpy twin ``renderer/water_foam.py``.

Each test names the property it protects and the change that must break it.
The shaders are held to the twin by ``tests/gl_water_foam_check.py``.
"""
from __future__ import annotations

import dataclasses
import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from renderer import water_foam as WF  # noqa: E402
from renderer import water_surface as W  # noqa: E402

TILE_M = 1.0 / 3.0


def _settings(**kw):
    from config import CFG
    return dataclasses.replace(W.water_settings(CFG), **kw)


def _quiet(**kw):
    """No source and no decay: only the carrying is left."""
    base = dict(foam_gain_froude=0.0, foam_gain_steep=0.0,
                foam_gain_edge=0.0, foam_gain_wall=0.0,
                foam_decay_s=math.inf)
    base.update(kw)
    return _settings(**base)


def _channel(h=8, w=24, depth=0.3, vx=0.0, vy=0.0, walls=True):
    """Packed (water, flow) float16 textures of a channel: solid rows top
    and bottom (if ``walls``), uniform depth and flow, still depth."""
    water = np.zeros((h, w, 4), np.float16)
    flow = np.zeros((h, w, 4), np.float16)
    water[..., 2] = depth
    flow[..., 0] = vx
    flow[..., 1] = vy
    flow[..., 3] = depth
    if walls:
        for r in (0, h - 1):
            water[r, :, 2] = 0.0
            flow[r, :, :2] = 0.0
            flow[r, :, 2] = 1.0
            flow[r, :, 3] = 0.0
    return water, flow


def _run(F, water, flow, s, steps, m=4):
    inp = WF.cell_inputs(water, flow, m, s)
    vmax = WF.flow_max(flow)
    for _ in range(steps):
        F = WF.step(F, inp, vmax, m, TILE_M, s)
    return F, inp


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------

def test_with_no_source_the_maximum_never_rises():
    """Property (critique M9): with every source off the foam's maximum
    never rises and no cell goes negative, step after step, over a flow that
    changes cell by cell around walls and a dry patch -- MacCormack's
    correction is reverted to the forward value where it leaves the taps'
    range, and decay, clamp and kill only lower. Breaks if the revert is
    dropped (a sharp step overshoots), if a blocked tap reads anything but a
    cell value, or if a source leaks in."""
    rng = np.random.default_rng(7)
    h, w, m = 10, 16, 4
    water, flow = _channel(h, w)
    yy, xx = np.mgrid[0:h, 0:w]
    flow[..., 0] = np.where(flow[..., 2] > 0, 0.0,
                            4.0 * np.sin(0.7 * xx + 0.3 * yy))
    flow[..., 1] = np.where(flow[..., 2] > 0, 0.0,
                            2.5 * np.cos(0.5 * yy - 0.4 * xx))
    water[4:6, 6:8, 2] = 0.0                  # a dry island
    s = _quiet(foam_decay_s=50.0)
    inp = WF.cell_inputs(water, flow, m, s)
    F = (rng.random((h * m, w * m)) > 0.6).astype(np.float64)  # sharp steps
    F = np.where(inp.blocked, 0.0, F) * 0.8
    top = F.max()
    vmax = WF.flow_max(flow)
    assert WF.substeps(vmax, s.foam_step_s, m, TILE_M) > 1   # not vacuous
    for _ in range(40):
        F = WF.step(F, inp, vmax, m, TILE_M, s)
        assert F.max() <= top + 1e-12
        assert F.min() >= 0.0
        top = F.max()
    assert top > 0.1, "vacuous: the foam vanished"


def test_the_revert_is_what_holds_the_maximum():
    """Property: the plain MacCormack correction on the same sharp field DOES
    overshoot (so the test above measures the clamp, not a smooth field).
    Breaks if the field gets too smooth to test anything."""
    rng = np.random.default_rng(7)
    F = (rng.random((40, 64)) > 0.6).astype(np.float64)
    d = np.zeros((40, 64, 2))
    d[..., 0] = 0.37
    d[..., 1] = -0.21
    blocked = np.zeros_like(F, bool)
    h, w = F.shape
    jj, ii = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    fwd, _, _ = WF._traced(F, blocked, jj - d[..., 0], ii - d[..., 1], F)
    back, _, _ = WF._traced(fwd, blocked, jj + d[..., 0], ii + d[..., 1],
                            fwd)
    raw = fwd + 0.5 * (F - back)
    assert raw.max() > 1.0 + 1e-3
    assert WF.advect(F, d, blocked).max() <= 1.0 + 1e-12


def test_zero_flow_and_zero_decay_keep_the_image_steady():
    """Property: still water (no flow, depth as last tick, the calm pool's
    residual ripple_v) with decay off holds the image EXACTLY -- the traces
    land on the cell itself and no source fires. Breaks if a source fires on
    still water (the steepness dead zone, the edge's square) or the trace
    drifts."""
    h, w, m = 8, 12, 4
    water, flow = _channel(h, w)
    water[..., 1] = np.where(flow[..., 2] > 0, 0.0, 0.01)   # W6a's drift
    s = _settings(foam_decay_s=math.inf)
    inp = WF.cell_inputs(water, flow, m, s)
    assert float(np.abs(inp.source).max()) == 0.0
    rng = np.random.default_rng(3)
    F0 = np.where(inp.blocked, 0.0, rng.random((h * m, w * m)))
    F, _ = _run(F0.copy(), water, flow, s, 30, m)
    assert np.array_equal(F, F0)


@pytest.mark.parametrize("u", [0.6, -0.6])
def test_the_amount_moves_downstream_at_the_flow_speed(u):
    """Property: a patch of foam in a uniform channel flow is carried at the
    flow's speed (its centroid within 5 % of ``u t``) and the right way
    (reversing the flow reverses the motion), and keeps most of itself
    (MacCormack's low diffusion). Breaks if the trace is scaled wrongly (m,
    the tile size, the substeps), signed wrongly, or the scheme smears."""
    h, w, m = 6, 60, 4
    water, flow = _channel(h, w, vx=u, walls=False)
    s = _quiet()
    yy, xx = np.mgrid[0:h * m, 0:w * m].astype(float)
    x0 = (w * m) / 2.0
    F0 = np.exp(-((xx - x0) / 6.0) ** 2)
    steps = 96                                     # 2 s
    F, _ = _run(F0.copy(), water, flow, s, steps, m)
    t = steps * s.foam_step_s
    cx0 = (F0 * xx).sum() / F0.sum()
    cx = (F * xx).sum() / F.sum()
    moved_m = (cx - cx0) / m * TILE_M
    assert abs(moved_m - u * t) < 0.05 * abs(u) * t, (moved_m, u * t)
    assert F.sum() > 0.97 * F0.sum()
    assert F.max() > 0.85


def test_a_substep_moves_at_most_one_cell():
    """Property (critique M10): the substep count covers the fastest flow
    at one cell a substep up to the cap, and a substep's displacement never
    exceeds one cell even past the cap. Breaks if the count or the clamp is
    lost (a fast flood would jump cells and tear the foam)."""
    m, dt = 4, 1.0 / 48.0
    for v in (0.0, 0.5, 2.0, 3.9, 8.0, 50.0):
        n = WF.substeps(v, dt, m, TILE_M)
        assert 1 <= n <= WF.FOAM_SUBSTEP_CAP
        if v * dt * m / TILE_M <= WF.FOAM_SUBSTEP_CAP:
            assert v * (dt / n) * m / TILE_M <= 1.0 + 1e-9
        d = WF.cell_step(np.array([[v, 0.3 * v]]), dt / n, m, TILE_M)
        assert float(np.hypot(*d[0])) <= 1.0 + 1e-12


def test_foam_on_a_cell_that_dries_is_killed():
    """Property: when a cell's tile dries out (depth 0) or turns solid its
    foam is gone at the next step. Breaks if the wet mask is not applied."""
    h, w, m = 6, 10, 4
    water, flow = _channel(h, w, walls=False)
    s = _quiet()
    F = np.ones((h * m, w * m))
    water[2, 3, 2] = 0.0
    flow[4, 7, 2] = 1.0
    F, inp = _run(F, water, flow, s, 1, m)
    assert (F[2 * m:3 * m, 3 * m:4 * m] == 0).all()
    assert (F[4 * m:5 * m, 7 * m:8 * m] == 0).all()
    assert (F[~inp.blocked] > 0.99).all()


def test_a_wall_neither_eats_nor_takes_foam():
    """Property: foam carried along a wall stays as much as it was (the
    backtrace's taps on the wall read the cell's own value) and none enters
    the wall. Breaks if a blocked tap reads 0 (the wall would eat the foam
    beside it) or a wall cell is not killed."""
    h, w, m = 8, 20, 4
    water, flow = _channel(h, w, vx=0.8, vy=0.0)
    flow[1:-1, :, 1] = -0.2                       # toward the north wall
    s = _quiet()
    inp = WF.cell_inputs(water, flow, m, s)
    F = np.where(inp.blocked, 0.0, 1.0)
    F, inp = _run(F, water, flow, s, 30, m)
    assert (F[inp.blocked] == 0).all()
    assert F[~inp.blocked].min() > 0.999


def test_the_sources_fire_where_they_should():
    """Property (critique M11): whitewater by Froude number (fast and
    shallow foams, slow and deep does not); the advancing edge (new water
    foams, standing water does not); walls (fast water beside a wall foams,
    the middle of the channel does not); steepness (strong agitation foams,
    the calm pool's residual does not). Breaks if a source is wired to the
    wrong channel or loses its threshold."""
    s = _settings()
    z = np.zeros(1)
    fast = np.array([[2.5, 0.0]])
    slow = np.array([[0.2, 0.0]])
    fr, *_ = WF.sources(fast, np.array([0.1]), np.array([0.1]), z, z, s,
                        split=True)
    assert fr[0] > 0
    fr, *_ = WF.sources(slow, np.array([0.5]), np.array([0.5]), z, z, s,
                        split=True)
    assert fr[0] == 0
    _, _, edge, _ = WF.sources(slow, np.array([0.2]), np.array([0.0]), z, z,
                               s, split=True)
    assert edge[0] == pytest.approx(s.foam_gain_edge)
    _, _, edge, _ = WF.sources(slow, np.array([0.3]), np.array([0.299]), z,
                               z, s, split=True)
    assert edge[0] < 1e-4 * s.foam_gain_edge
    _, _, _, wall = WF.sources(fast, np.array([0.5]), np.array([0.5]),
                               np.array([0.375]), z, s, split=True)
    assert wall[0] > 0
    _, _, _, wall = WF.sources(fast, np.array([0.5]), np.array([0.5]), z, z,
                               s, split=True)
    assert wall[0] == 0
    _, st, _, _ = WF.sources(np.zeros((1, 2)), np.array([0.3]),
                             np.array([0.3]), z, np.array([0.2]), s,
                             split=True)
    assert st[0] == pytest.approx(s.foam_gain_steep)
    _, st, _, _ = WF.sources(np.zeros((1, 2)), np.array([0.3]),
                             np.array([0.3]), z, np.array([0.01]), s,
                             split=True)
    assert st[0] == 0


def test_a_flood_front_makes_foam_that_travels_with_it():
    """Property (the flooding_corridor brief): water arriving on dry floor
    makes foam at its front, and the foam is carried downstream after it.
    Breaks if the edge source or the carrying is lost."""
    h, w, m = 6, 40, 4
    water, flow = _channel(h, w, vx=1.5, walls=False)
    water[:, 20:, 2] = 0.0
    flow[:, 20:, 3] = 0.0
    flow[:, 20:, :2] = 0.0
    flow[:, 15:20, 3] = 0.0                      # it just arrived here
    s = _settings()
    F = np.zeros((h * m, w * m))
    F, inp = _run(F, water, flow, s, 4, m)
    front = F[:, 15 * m:20 * m].mean()
    behind = F[:, 2 * m:10 * m].mean()
    assert front > 0.2 and front > 5 * behind
    # now the front has passed: the foam is carried on, downstream
    flow[:, 15:20, 3] = 0.3
    s2 = _quiet()
    cx0 = (F * np.arange(w * m)).sum() / F.sum()
    F, _ = _run(F, water, flow, s2, 8, m)
    cx = (F * np.arange(w * m)).sum() / max(F.sum(), 1e-12)
    assert cx > cx0


# ---------------------------------------------------------------------------
# The box (the ping-pong's partial passes)
# ---------------------------------------------------------------------------

def test_leaving_rects_are_exactly_the_old_box_outside_the_new():
    """Property: the rects a shrinking or moving box clears cover exactly
    the old box minus the new one, without overlap. Breaks if a stale cell
    is missed (old foam would reappear when the water returns) or a live one
    is cleared."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        a = sorted(rng.integers(0, 30, 2)), sorted(rng.integers(0, 30, 2))
        b = sorted(rng.integers(0, 30, 2)), sorted(rng.integers(0, 30, 2))
        old = (a[0][0], a[1][0], a[0][1] + 1, a[1][1] + 1)
        new = (b[0][0], b[1][0], b[0][1] + 1, b[1][1] + 1)
        want = np.zeros((32, 32), int)
        want[old[1]:old[3], old[0]:old[2]] = 1
        want[new[1]:new[3], new[0]:new[2]] = 0
        got = np.zeros((32, 32), int)
        for x0, y0, x1, y1 in WF.leaving_rects(old, new):
            got[y0:y1, x0:x1] += 1
        assert np.array_equal(got, want)
    assert WF.leaving_rects(None, (0, 0, 2, 2)) == []
    assert WF.leaving_rects((0, 0, 2, 2), None) == [(0, 0, 2, 2)]


# ---------------------------------------------------------------------------
# The draw
# ---------------------------------------------------------------------------

def _bubbles():
    from renderer.noise_bake import bake_foam_rgba
    return bake_foam_rgba(128)


def test_the_bubble_bake_is_uniform_periodic_and_deterministic():
    """Property: the bubble field is uniform on [0, 1] (the lace threshold's
    closed form depends on it), exactly periodic (it tiles) and the same bits
    every bake. Breaks if the rank equalisation, the wrap of the feature
    points or the fixed seed is lost."""
    from renderer.noise_bake import worley_f1_f2
    a = _bubbles()
    b = _bubbles()
    assert np.array_equal(a, b)
    r = a[..., 0]
    hist, _ = np.histogram(r, bins=10, range=(0.0, 1.0))
    assert hist.min() > 0.95 * r.size / 10
    xy = np.random.default_rng(5).random((400, 2))
    f1, f2, n = worley_f1_f2(xy, 8, np.random.default_rng(1))
    g1, g2, k = worley_f1_f2(xy + 1.0, 8, np.random.default_rng(1))
    assert np.allclose(f1, g1) and np.allclose(f2, g2)
    assert np.array_equal(n, k)


def _cover_grid(amount, s, fade=1.0, n=220):
    span = 6.0
    y, x = np.meshgrid(np.linspace(0.3, span, n), np.linspace(0.2, span, n),
                       indexing="ij")
    frame = W.FlowFrame(0.3, s.flow_cycle_s, 0.0, 0.0, fade)
    vel = np.zeros(x.shape + (2,))
    vel[..., 0] = 0.4
    depth = np.full(x.shape, 0.3)
    return WF.draw_cover(x, y, vel, depth, np.full(x.shape, amount), frame,
                         _bubbles(), s)


@pytest.mark.parametrize("f", [0.1, 0.3, 0.5, 0.7, 0.9])
def test_the_lace_covers_the_amounts_share(f):
    """Property (Parberry): the foam breaks into lace, it does not fade --
    the drawn cover is mostly 0 or 1 (few half-covered pixels), and the
    share it covers is the amount's ``f = F / foam_cover`` within 0.06.
    Breaks if the threshold's closed form or the bake's equalisation is
    changed, or if the cover becomes a fade (a uniform grey of f)."""
    s = _settings(foam_bubble_m=(0.1, 0.4), foam_cover=1.0)
    c = _cover_grid(f, s)
    assert abs(c.mean() - f) < 0.06, (c.mean(), f)
    mid = ((c > 0.1) & (c < 0.9)).mean()
    assert mid < 0.35, f"too much half-cover ({mid:.2f}): a fade, not lace"


def test_the_sheet_closes_and_no_foam_draws_nothing():
    """Property: an amount at ``foam_cover`` or above is a closed sheet, an
    amount of 0 draws exactly nothing (the water below unchanged). Breaks if
    the threshold's widening at the ends is lost."""
    s = _settings(foam_bubble_m=(0.1, 0.4))
    assert _cover_grid(s.foam_cover, s).min() > 0.999
    assert _cover_grid(1.5 * s.foam_cover, s).min() > 0.999
    assert _cover_grid(0.0, s).max() == 0.0


def test_zoomed_out_the_lace_is_its_mean():
    """Property (critique M8's zoom fade): with the fine detail faded out
    the foam is its mean cover ``f`` everywhere -- no minified lace
    shimmer. Breaks if the fade is not applied to the foam."""
    s = _settings(foam_bubble_m=(0.1, 0.4), foam_cover=1.0)
    c = _cover_grid(0.4, s, fade=0.0)
    assert np.allclose(c, 0.4)


def test_the_bubbles_ride_the_one_flow_map():
    """Property: the bubbles slide with W1's flow map -- their layer
    coordinates are the position minus ``water_surface.layer_slides``'s
    slide and the drift, so still water's two layers read the same point and
    a flow moves them downstream. Breaks if the foam gets an advector of its
    own."""
    s = _settings(foam_bubble_m=(0.1, 0.4))
    x = np.array([1.0, 2.0])
    y = np.array([0.5, 0.7])
    frame = W.FlowFrame(0.3, s.flow_cycle_s, 0.0, 0.0, 1.0)
    still = np.zeros((2, 2))
    d0, d1, _w, _v = W.layer_slides(x, y, still, np.full(2, 0.3), frame, s)
    a = WF.bubble_coords(x, y, d0, frame, s)
    b = WF.bubble_coords(x, y, d1, frame, s)
    assert all(np.array_equal(p, q) for p, q in zip(a, b))
    vel = np.tile([[0.5, 0.0]], (2, 1))
    d0, _d1, _w, _v = W.layer_slides(x, y, vel, np.full(2, 0.3), frame, s)
    q = WF.bubble_coords(x, y, d0, frame, s)
    assert (q[0][:, 0] < a[0][:, 0]).all()     # the texture slid downstream


def test_the_shadow_darkens_the_floor_by_the_cover():
    """Property: beneath a closed sheet the floor keeps ``1 - foam_shadow``
    of its light, beneath no foam all of it. Breaks if the shadow's weight
    stops being the amount's share."""
    s = _settings(foam_shadow=0.5)
    assert WF.shadow_factor(0.0, s) == 1.0
    assert WF.shadow_factor(s.foam_cover, s) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The door
# ---------------------------------------------------------------------------

def _cfg(**over):
    from types import SimpleNamespace
    from config import CFG
    d = dict(vars(CFG.graphics.water))
    d.update(over)
    return SimpleNamespace(graphics=SimpleNamespace(
        water=SimpleNamespace(**d)))


@pytest.mark.parametrize("over,name", [
    ({"foam_cells_per_tile": 0}, "foam_cells_per_tile"),
    ({"foam_cells_per_tile": 9}, "foam_cells_per_tile"),
    ({"foam_cells_per_tile": 4.0}, "foam_cells_per_tile"),
    ({"foam_step_s": 0.0}, "foam_step_s"),
    ({"foam_step_s": 0.5}, "foam_step_s"),
    ({"foam_decay_s": 0.0}, "foam_decay_s"),
    ({"foam_cover": -1.0}, "foam_cover"),
    ({"foam_gain_edge": -1.0}, "foam_gain_edge"),
    ({"foam_bubble_m": [0.1]}, "foam_bubble_m"),
    ({"foam_bubble_m": [0.1, 0.0]}, "foam_bubble_m"),
    ({"foam_shadow": 1.5}, "foam_shadow"),
    ({"debug_view": 3}, "debug_view"),
])
def test_the_door_refuses_bad_foam_dials(over, name):
    """Property (critique M6): the one door refuses a foam dial out of its
    range or of the wrong shape, by name. Breaks if a foam dial bypasses the
    door or falls back to a default."""
    with pytest.raises(ValueError, match=name):
        W.water_settings(_cfg(**over))


def test_the_old_gradient_foam_dials_are_gone():
    """Property: ``foam_threshold`` / ``foam_intensity`` (the gradient foam)
    and the dead overlay's ``[display] water_*`` keys left with what they
    drove. Breaks if one comes back as a dead dial."""
    from config import CFG
    keys = set(vars(CFG.graphics.water))
    assert not {"foam_threshold", "foam_intensity"} & keys
    disp = set(vars(CFG.display))
    assert not {k for k in disp if k.startswith("water_")}
