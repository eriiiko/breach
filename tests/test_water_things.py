"""Water W3 (#97) -- things in the water: the WATERLINE (critique M12,
``water_surface.waterline_tint``, mirrored by ``lit3d._WATERLINE_GLSL``),
the RING and the WAKE (M13, ``renderer/water_wake.py`` -> W2's foam through
``water_foam.splat_rate`` / ``frame_splats``), the SPRAY (the particle
primitive's fifth consumer, ``particles.SpraySystem``) and its landings'
foam dabs, the splash at a wet wall-stop.

Properties on the numpy twins only (never a snapshot); every test names the
property it protects and the change that must break it. The GL face is held
to these twins by ``tests/gl_water_things_check.py``.
"""
from __future__ import annotations

import dataclasses
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from renderer import particles as P  # noqa: E402
from renderer import water_foam as WF  # noqa: E402
from renderer import water_wake as WW  # noqa: E402
from renderer import water_surface as WS  # noqa: E402

TILE_M = 1.0 / 3.0
TPS = 24.0


def _ws(**kw):
    from config import CFG
    return dataclasses.replace(WS.water_settings(CFG), **kw)


def _spray(**kw):
    base = dict(enabled=True, rate=30.0, speed_mps=2.5, up_mps=3.0,
                spread=0.6, size_tiles=0.05, albedo=(0.7, 0.78, 0.85),
                alpha=0.85, seed=4513, cap=1024, splash_drops=6,
                splash_speed_mps=1.0, splash_up_mps=1.5)
    base.update(kw)
    return P.SpraySettings(**base)


def _blast(pos, strength=3.0, tick0=192):
    return {"kind": "blast", "pos": pos, "strength": strength,
            "approach": None, "tick0": tick0}


def _room(h=40, w=40):
    blocked = np.ones((h, w), bool)
    blocked[1:-1, 1:-1] = False
    return blocked


def _seeds_fn(open_mask):
    from simulation.blast_expansion import walk_seeds
    return lambda fy, fx, approach: walk_seeds(open_mask, fy, fx, approach)


# ---------------------------------------------------------------------------
# The waterline
# ---------------------------------------------------------------------------

def test_a_dry_tile_or_a_fragment_above_the_surface_is_untouched():
    """PROPERTY: a fragment on a DRY tile (depth 0), at or above the surface
    (height >= depth), or with ``waterline_strength`` 0 comes out of the
    waterline BIT-IDENTICAL to its lit colour.

    BREAKS IF: the law tints without water over the fragment (e.g. reads a
    blurred depth, or adds the water colour whatever the column), or the
    strength 0 still mixes."""
    s = _ws()
    rng = np.random.default_rng(1)
    lit = rng.random((500, 3)) * 3.0
    h = rng.random(500) * 2.0
    for depth in (np.zeros(500), h * rng.random(500)):      # dry; below h
        out = WS.waterline_tint(lit, depth, h, s)
        assert np.array_equal(out, lit)
    out = WS.waterline_tint(lit, h + 0.1, h, _ws(waterline_strength=0.0))
    assert np.array_equal(out, lit)


def test_the_tint_grows_with_the_water_column_above_the_fragment():
    """PROPERTY: the further below the surface a fragment is, the further
    its colour is taken from its lit colour toward the water's volume
    colour (monotone in the column), 0 at the surface, and never past the
    water pass's own alpha ceiling of the way to ``water_color``.

    BREAKS IF: the column is measured from the floor instead of the surface
    (the top of a submerged shin would be tinted most), the sign of the
    fade flips, or the weight leaves the water pass's alpha ramp."""
    s = _ws()
    lit = np.array([2.0, 1.5, 1.0])
    depth = 0.3
    hs = np.linspace(0.3, 0.0, 61)                 # surface down to floor
    out = WS.waterline_tint(np.tile(lit, (61, 1)), np.full(61, depth), hs, s)
    dist = np.linalg.norm(out - lit, axis=1)  # blas-exempt: test only
    assert dist[0] == 0.0
    assert np.all(np.diff(dist) >= -1e-12) and dist[-1] > 0.3
    wc = np.asarray(s.water_color)
    full = np.linalg.norm(wc - lit)  # blas-exempt: test only
    assert dist.max() <= s.alpha_max * full + 1e-9


def test_the_waterline_is_the_floor_law_of_the_water_pass():
    """PROPERTY: a body fragment AT the floor (height 0) under ``d`` of water
    takes exactly what the water pass's composite does to the floor there
    (``alpha x mix(water_color, floor, 2^(-fog d)) + (1 - alpha) floor``,
    the same dials) -- one law, one source of truth.

    BREAKS IF: the waterline grows its own colour, fog or alpha dial."""
    s = _ws()
    lit = np.array([[0.4, 0.5, 0.6]])
    for d in (0.05, 0.15, 0.3, 1.2):
        a = np.clip(d * s.alpha_scale, s.alpha_min, s.alpha_max)
        a *= float(WS.smoothstep(0.0, max(s.height_edge, 1e-4), d))
        fog = 2.0 ** (-s.fog_density * d)
        refr = np.asarray(s.water_color) * (1 - fog) + lit * fog
        want = a * refr + (1 - a) * lit
        got = WS.waterline_tint(lit, np.array([d]), np.array([0.0]), s)
        assert np.allclose(got, want, atol=1e-12)


# ---------------------------------------------------------------------------
# The splats: ring, wake, dab
# ---------------------------------------------------------------------------

def test_a_splat_is_zero_beyond_its_band_and_peaks_on_its_ring():
    """PROPERTY: a ring splat's rate is its peak on the ring (distance r_c
    from the segment), falls linearly to 0 at r_c +- hw, and is exactly 0
    outside; a zero-length capsule is a circle; no splat, no rate.

    BREAKS IF: the tent leaks beyond its band (foam in the whole room) or
    the distance is taken to the end point instead of the segment."""
    sp = np.array([[10.0, 10.0, 10.0, 10.0, 1.2, 0.25, 2.0]])
    ang = np.linspace(0, 2 * np.pi, 33)
    for r, want in ((1.2, 2.0), (1.2 + 0.125, 1.0), (1.2 + 0.25, 0.0),
                    (0.5, 0.0), (3.0, 0.0)):
        got = WF.splat_rate(10 + r * np.cos(ang), 10 + r * np.sin(ang), sp)
        assert np.allclose(got, want, atol=1e-9), (r, got)
    seg = np.array([[0.0, 0.0, 10.0, 0.0, 1.0, 0.2, 1.0]])
    assert np.allclose(WF.splat_rate(np.linspace(0, 10, 11), 1.0, seg), 1.0)
    assert WF.splat_rate(np.array([5.0]), np.array([5.0]),
                         np.zeros((0, 7))).max() == 0.0


def test_every_dab_adds_its_amount_once_and_none_is_dropped():
    """PROPERTY: ``frame_splats`` hands the frame's rate splats as they are
    and each queued DAB as a rate that, over the frame's steps, adds its
    amount exactly once; at most SPLAT_MAX splats per frame, the dabs that
    do not fit waiting (never dropped); a frame that runs no step keeps
    every dab.

    BREAKS IF: a dab's amount is applied per step (n times), dropped at
    the cap, or spent by a frame without steps."""
    rates = np.tile([1, 1, 1, 1, 0.5, 0.2, 0.7], (5, 1)).astype(float)
    dabs = np.tile([2, 2, 2, 2, 0.0, 0.3, 0.5], (40, 1)).astype(float)
    sp, left = WF.frame_splats(rates, dabs, 0.0)
    assert sp.shape[0] == 0 and left.shape[0] == 40
    total = 0
    queue = dabs
    steps_s = 3 * (1 / 48)
    while queue.shape[0]:
        sp, queue = WF.frame_splats(rates, queue, steps_s)
        assert sp.shape[0] <= WF.SPLAT_MAX
        assert np.array_equal(sp[:5], rates)
        got = sp[5:]
        assert np.allclose(got[:, 6] * steps_s, 0.5)
        total += got.shape[0]
    assert total == 40


def test_a_still_unit_makes_a_ring_and_no_wake():
    """PROPERTY: a unit standing still in water hands ONE splat, a circle
    (a == b) of the ring's rate alone -- the wake's share is exactly 0.

    BREAKS IF: speed leaks in from anything but the centre's per-tick
    displacement (a pose, a facing), or a still unit gets a wake."""
    s = _ws()
    depth = np.full((20, 20), 0.15)
    wt = WW.WakeTracker()
    for tick in range(10):
        sp = wt.splats([(7, 10.0, 10.0)], tick, depth, TILE_M, TPS, s)
        assert sp.shape == (1, 7)
        assert sp[0, 0] == sp[0, 2] and sp[0, 1] == sp[0, 3]
        assert sp[0, 6] == pytest.approx(s.wake_ring_rate)
    assert float(WW.wake_rate(0.0, s)) == 0.0


def test_the_wake_grows_with_speed_and_sweeps_the_ticks_path():
    """PROPERTY: a walking unit's splat is the capsule from its centre at the
    previous tick to its centre now, its rate the ring's plus a share that
    grows with the speed (m/s from the per-tick displacement x tile x tps).

    BREAKS IF: the speed is not the displacement per sim tick (per frame,
    say: frames between ticks would read 0), or the wake stops growing."""
    s = _ws()
    depth = np.full((40, 40), 0.15)
    prev_rate = None
    for v_mps in (0.3, 0.8, 1.4, 4.5):
        wt = WW.WakeTracker()
        step = v_mps / TPS / TILE_M
        x = 5.0
        for tick in range(5):
            for _frame in range(3):                 # frames between ticks
                sp = wt.splats([(1, x, 20.0)], tick, depth, TILE_M, TPS, s)
            x += step
        assert sp[0, 2] - sp[0, 0] == pytest.approx(step)
        assert sp[0, 6] == pytest.approx(float(WW.ring_rate(v_mps, s)))
        if prev_rate is not None:
            assert sp[0, 6] > prev_rate
        prev_rate = sp[0, 6]


def test_nothing_on_dry_tiles_and_a_seek_forgets():
    """PROPERTY: a unit on a dry tile (or on a dry map) hands no splat; a
    sim tick that goes BACK (a seek) forgets every path, so the first
    sighting after it is a standing start; at most SPLAT_MAX splats a frame.

    BREAKS IF: the wake foams dry floor, a seek reads a huge speed from the
    old position, or a crowd overflows the shader's uniform arrays."""
    s = _ws()
    depth = np.zeros((20, 20))
    depth[:, 10:] = 0.2
    wt = WW.WakeTracker()
    assert wt.splats([(1, 5.0, 5.0)], 0, depth, TILE_M, TPS, s).shape[0] == 0
    assert wt.splats([(1, 5.0, 5.0)], 1, None, TILE_M, TPS, s).shape[0] == 0
    wt.splats([(2, 12.0, 5.0)], 50, depth, TILE_M, TPS, s)
    sp = wt.splats([(2, 16.0, 9.0)], 20, depth, TILE_M, TPS, s)
    assert sp[0, 6] == pytest.approx(s.wake_ring_rate)
    many = [(k, 10.5 + (k % 9), 1.0 + k // 9) for k in range(100)]
    assert wt.splats(many, 21, depth, TILE_M, TPS, s).shape[0] <= WF.SPLAT_MAX


def test_a_ring_makes_foam_only_round_the_unit_in_the_foam_step():
    """PROPERTY: through W2's own step, a standing unit's ring foams the
    band round it and nothing else in a still pool; without the splat the
    pool stays empty (the splat is the only cause).

    BREAKS IF: ``cell_inputs`` drops the splats, or the splat lands in the
    wrong frame (cell vs tile coordinates)."""
    s = _ws()
    h, w, m = 16, 16, s.foam_cells_per_tile
    water = np.zeros((h, w, 4), np.float16)
    water[..., 2] = 0.15
    flow = np.zeros((h, w, 4), np.float16)
    flow[..., 3] = 0.15
    sp = WW.WakeTracker().splats([(1, 8.0, 8.0)], 0, water[..., 2]
                                 .astype(float), TILE_M, TPS, s)
    for splats, expect_foam in ((None, False), (sp, True)):
        inp = WF.cell_inputs(water, flow, m, s, splats=splats)
        F = np.zeros((h * m, w * m))
        for _ in range(48):
            F = WF.step(F, inp, 0.0, m, TILE_M, s)
        assert (F.max() > 0.05) == expect_foam
    xt, yt = WF.cell_centres(h, w, m)
    r = np.hypot(xt - 8.0, yt - 8.0)
    rc, hw = s.wake_body_m / TILE_M, s.wake_band_m / TILE_M
    assert F[np.abs(r - rc) > hw + 1.0 / m].max() == 0.0
    assert F[np.abs(r - rc) < 0.5 * hw].min() > 0.0


# ---------------------------------------------------------------------------
# The spray
# ---------------------------------------------------------------------------

def _depth(h=40, w=40, wet=True):
    d = np.zeros((h, w))
    if wet:
        d[1:-1, 1:-1] = 0.3
    return d


def test_a_blast_on_a_dry_room_throws_no_spray():
    """PROPERTY: a detonation whose walk's seeds are all dry (or a dry map:
    no depth) throws no droplet, and its effect is still marked seen (it
    never sprays later when water arrives).

    BREAKS IF: the count ignores the wet share, or the seen mark is set
    only when droplets are thrown."""
    blocked = _room()
    seeds = _seeds_fn(~blocked)
    for depth in (_depth(wet=False), None):
        sys_ = P.SpraySystem(_spray())
        fx = _blast((20, 20))
        sys_.update(t=0.0, effects=[fx], seeds_fn=seeds, sim_tick=192,
                    depth=depth, tile_m=TILE_M, blocked=blocked)
        assert sys_.p.n == 0 and fx["spray_seen"]
        sys_.update(t=0.1, effects=[fx], seeds_fn=seeds, sim_tick=194,
                    depth=_depth(), tile_m=TILE_M, blocked=blocked)
        assert sys_.p.n == 0


def test_the_spray_scales_with_strength_and_the_wet_share():
    """PROPERTY: the droplet count is round(rate x strength x wet share),
    the wet share the seeds standing in water; droplets leave only from
    WET seeds.

    BREAKS IF: a half-dry blast throws a full spray, or a droplet leaves a
    dry seed."""
    assert P.spray_count(3.0, 30.0, 1.0) == 90
    assert P.spray_count(3.0, 30.0, 0.5) == 45
    assert P.spray_count(0.0, 30.0, 1.0) == 0
    blocked = _room()
    blocked[20, 20] = True                  # a pillar: the seeds are its
    depth = _depth()                        # open neighbours
    depth[:, :20] = 0.0                     # the west half dry
    sys_ = P.SpraySystem(_spray())
    seeds_fn = _seeds_fn(~blocked)
    seeds = seeds_fn(20, 20, None)
    wet = sum(depth[sy, sx] > 0 for sy, sx in seeds) / len(seeds)
    assert 0.0 < wet < 1.0
    sys_.update(t=0.0, effects=[_blast((20, 20))], seeds_fn=seeds_fn,
                sim_tick=192, depth=depth, tile_m=TILE_M, blocked=blocked)
    assert sys_.p.n == P.spray_count(3.0, 30.0, wet)
    assert np.all(depth[np.floor(sys_.p.temp).astype(int),
                        np.floor(sys_.p.t0).astype(int)] > 0)


def _fly(sys_, t_end, dt=1 / 60):
    t, lands = 0.0, []
    n0 = sys_.p.n
    starts = sys_.p.d[:, [P.T0, P.TEMP]].copy()
    while t < t_end:
        t += dt
        lands.append(sys_.update(t=t, effects=[], seeds_fn=None,
                                 sim_tick=0, depth=_depth(), tile_m=TILE_M))
    return n0, starts, np.concatenate(lands)


def test_spray_lands_within_its_reach_and_every_landing_is_one_dab():
    """PROPERTY: every droplet lands, exactly once, at most the dials' reach
    (``spray_reach_tiles``) from where it left, never inside a wall; each
    landing becomes exactly one foam dab of ``splash_foam``; nothing in the
    air outlives its flight.

    BREAKS IF: the ground track is integrated past its flight time, a wall
    is crossed, a droplet lands twice or never, or a landing hands no (or
    two) foam sources."""
    s = _spray()
    blocked = _room(30, 30)
    sys_ = P.SpraySystem(s)
    sys_.update(t=0.0, effects=[_blast((5, 15))], seeds_fn=_seeds_fn(~blocked),
                sim_tick=192, depth=_depth(30, 30), tile_m=TILE_M,
                blocked=blocked)
    n0 = sys_.p.n
    assert n0 > 50
    starts = sys_.p.d[:, [P.T0, P.TEMP]].copy()
    life_max = float(sys_.p.life.max())
    t, lands = 0.0, []
    while t < life_max + 0.1:
        t += 1 / 60
        lands.append(sys_.update(t=t, effects=[], seeds_fn=None, sim_tick=0,
                                 depth=_depth(30, 30), tile_m=TILE_M))
        assert np.all(sys_.p.age < sys_.p.life)
    land = np.concatenate(lands)
    assert sys_.p.n == 0 and land.shape[0] == n0
    reach = P.spray_reach_tiles(s.speed_mps, s.up_mps, s.spread, TILE_M)
    # each landing is within reach of SOME launch point (one seed per drop)
    d = np.hypot(land[:, None, 0] - starts[None, :, 0],
                 land[:, None, 1] - starts[None, :, 1]).min(axis=1)
    assert d.max() <= reach + 1e-9
    assert not blocked[np.floor(land[:, 1]).astype(int),
                       np.floor(land[:, 0]).astype(int)].any()
    dabs = WW.landing_dabs(land, TILE_M, _ws())
    assert dabs.shape == (n0, 7)
    assert np.allclose(dabs[:, 6], _ws().splash_foam)


def test_the_rise_and_fall_is_analytic():
    """PROPERTY: a droplet's height is 0 at launch and at landing, highest
    at mid-flight, and ``life = 2 up / g`` (the ballistic flight).

    BREAKS IF: the height law drifts from the parabola or the flight time
    from the launch speed."""
    life = 2 * 3.0 / P.G_MPS2
    g = P.G_MPS2 / TILE_M
    a = np.linspace(0, life, 101)
    z = P.spray_height(a, life, g)
    assert z[0] == 0.0 and abs(z[-1]) < 1e-12
    assert np.argmax(z) == 50
    assert z[50] == pytest.approx(3.0 ** 2 / (2 * P.G_MPS2) / TILE_M)


def test_a_splash_flies_back_off_the_wall():
    """PROPERTY: a splash's droplets leave within 60 deg of the direction
    handed in (back along the round's flight), a few of them.

    BREAKS IF: the heading is ignored (droplets fly into the wall)."""
    sys_ = P.SpraySystem(_spray())
    k = sys_.splash(10.0, 10.0, -1.0, 0.0, TILE_M, np.random.default_rng(3))
    assert k == 6
    ang = np.arctan2(sys_.p.vy, sys_.p.vx)
    assert np.all(np.abs(np.abs(ang) - np.pi) <= np.radians(60) + 1e-9)
    assert WW.wet_near(_depth(), 10.0, 10.0)
    assert not WW.wet_near(_depth(wet=False), 10.0, 10.0)
    assert not WW.wet_near(None, 10.0, 10.0)


def test_the_cap_holds():
    """PROPERTY: never more droplets in the air than ``cap``.

    BREAKS IF: the add path ignores the cap."""
    blocked = _room()
    sys_ = P.SpraySystem(_spray(cap=40))
    sys_.update(t=0.0, effects=[_blast((20, 20), strength=12.0)],
                seeds_fn=_seeds_fn(~blocked), sim_tick=1, depth=_depth(),
                tile_m=TILE_M, blocked=blocked)
    assert sys_.p.n == 40


# ---------------------------------------------------------------------------
# The doors
# ---------------------------------------------------------------------------

def test_the_doors_refuse():
    """PROPERTY: the W3 water dials and the spray dials pass refusing doors:
    an out-of-range or missing value raises naming it.

    BREAKS IF: a dial gets a silent default or the range checks go."""
    from config import CFG
    with pytest.raises(ValueError, match="waterline_strength"):
        WS.water_settings(SimpleNamespace(graphics=SimpleNamespace(
            water=SimpleNamespace(**{**vars(CFG.graphics.water),
                                     "waterline_strength": 1.5}))))
    with pytest.raises(ValueError, match="wake_band_m"):
        WS.water_settings(SimpleNamespace(graphics=SimpleNamespace(
            water=SimpleNamespace(**{**vars(CFG.graphics.water),
                                     "wake_band_m": 0.0}))))
    sec = dict(vars(CFG.render.spray))
    sec.pop("splash_drops")
    with pytest.raises(ValueError, match="splash_drops"):
        P.spray_settings_from_config(SimpleNamespace(
            render=SimpleNamespace(spray=SimpleNamespace(**sec))))
    assert P.spray_settings_from_config(CFG).enabled in (True, False)


def test_the_renderer_consumes_blasts_with_its_own_flag():
    """PROPERTY: the spray marks blasts ``spray_seen`` -- its own flag, never
    another consumer's (sparks / debris / shock keep theirs).

    BREAKS IF: the spray reuses another consumer's seen flag (a blast would
    then be thrown by only one of them)."""
    src = inspect.getsource(P.SpraySystem.consume)
    assert '"spray_seen"' in src
    for other in ("sparks_seen", "debris_seen", "shock_seen"):
        assert other not in src
