"""THE FIRE'S ROOM-LIGHT FLICKER -- graphics G3 F4b (renderer/fire_flicker.py,
the door renderer/frame_lights.py emit_mult_rows, the sweep's emission
multiplier). Design docs/g3_fire_design_2026-10-23.md section 6.3 (RULED D)
and Erik's two worries in section 1 item 6.

Headless: the flicker is pure numpy (no GL); the game-level gates run a real
Simulation with the light requested and the flicker rows handed through
Simulation.set_light, exactly as main.py hands them.

Every test's docstring names its property and the change that breaks it.
"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tests", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402

from renderer import fire_flicker as ff  # noqa: E402
from renderer import fire_vfx_ref as ref  # noqa: E402
from renderer import frame_lights as fl  # noqa: E402

TPS = 24.0
DIALS = ref.FireVFXDials()
S = ff.FlickerSettings()
MAP_W = 64


def _sources(burn_mask, tile_size_m=0.333):
    """THE source law's FlameSources for a map whose burning tiles are
    ``burn_mask`` (permeable fuel at 1200 K, I = 0.6)."""
    fire = np.where(burn_mask, 0.6, 0.0)
    tk = np.where(burn_mask, 1200.0, 300.0)
    z = np.zeros(burn_mask.shape, bool)
    return ref.flame_sources(fire, tk, z, burn_mask, z, t_amb_k=293.0,
                             glow_min_k=800.0, tile_size_m=tile_size_m, dials=DIALS)


def _studio_like_layout():
    """A big fire and separate small ones: a 2 x 9 block (one cluster -- the
    studio's furniture row, doubled), a 3 x 3 block, and five LONE tiles of
    the same size (equal fires: the case most likely to lock in step)."""
    m = np.zeros((40, MAP_W), bool)
    m[10:12, 8:17] = True
    m[25:28, 30:33] = True
    for (y, x) in ((5, 40), (5, 50), (18, 45), (33, 12), (33, 55)):
        m[y, x] = True
    return m


def _series(burn_mask, seconds, settings=S, t0=37.0):
    src = _sources(burn_mask)
    ys, xs, hz, ph = ff.burning_tiles(src, (0, 0, burn_mask.shape[1], burn_mask.shape[0]))
    ts = t0 + np.arange(int(seconds * TPS)) / TPS
    M = np.stack([ff.multipliers(ys, xs, hz, ph, t, settings, DIALS, MAP_W) for t in ts])
    return ys, xs, M


# ---------------------------------------------------------------------------
# 1. ERIK'S WORRY: NO SYNC
# ---------------------------------------------------------------------------
def test_fires_and_tiles_three_apart_do_not_flicker_in_sync():
    """PROPERTY (design 6.3 "no sync", Erik's first worry): over 30 s of ticks
    (>= the gate's 10 s) every pair of burning tiles three or more tiles apart
    -- inside one fire and between separate, equal-sized fires -- has a
    multiplier series correlation |rho| < 0.3, and the mean |rho| over the
    pairs stays near the shared puffing's small share (< 0.15).

    BREAKS IF: the flicker runs on a shared clock or phase (one sinusoid for
    every tile, a per-cluster-only multiplier, a puff_share near 1), or the
    per-tile hash maps nearby tiles to the same frequencies (the weak
    fire_vfx_ref._hash01 did: tiles 572 / 574 drew 0.3282 / 0.3276).
    """
    ys, xs, M = _series(_studio_like_layout(), 30.0)
    C = np.corrcoef(M.T)
    far = []
    for i in range(len(ys)):
        for j in range(i + 1, len(ys)):
            if max(abs(ys[i] - ys[j]), abs(xs[i] - xs[j])) >= 3:
                far.append(abs(C[i, j]))
    far = np.asarray(far)
    assert len(far) > 300
    assert far.max() < 0.3, f"max |rho| {far.max():.3f}"
    assert far.mean() < 0.15, f"mean |rho| {far.mean():.3f}"


def test_a_shared_clock_would_fail_the_no_sync_gate():
    """PROPERTY (the gate is not vacuous): the same layout with puff_share = 1
    -- every tile of a fire on its fire's puffing alone, a shared clock --
    has tiles of one fire at |rho| ~ 1. So the no-sync test above measures
    something the construction could break.

    BREAKS IF: the measure stops seeing a shared clock (e.g. the series were
    computed per cluster and broadcast after the correlation).
    """
    ys, xs, M = _series(_studio_like_layout(), 10.0, settings=ff.FlickerSettings(puff_share=1.0))
    C = np.corrcoef(M.T)
    same_fire = [abs(C[i, j]) for i in range(len(ys)) for j in range(i + 1, len(ys))
                 if 10 <= ys[i] <= 11 and 10 <= ys[j] <= 11
                 and abs(xs[i] - xs[j]) >= 3]
    assert min(same_fire) > 0.95


# ---------------------------------------------------------------------------
# 2. MEAN-PRESERVING, BOUNDED, NEVER A STROBE
# ---------------------------------------------------------------------------
def test_the_flicker_keeps_the_fires_average_light():
    """PROPERTY (design 6.3 "clamp: mean-preserving over a second"): for every
    tile, the multiplier's mean over every 1 s window stays within a third of
    the amplitude of 1 (a second cannot average out a 0.8 Hz breath, but it
    never holds the fire far off its light), its window means average to
    within amplitude / 6 of 1, and over 60 s the mean is 1 within 0.02 -- so
    Erik's tuned brightness (the physics' light) is the flicker's average.

    BREAKS IF: the flicker biases the light (a one-sided law such as
    1 + a|z|, an asymmetric clip, a puff term with a non-zero mean) or its
    band reaches down to slow drifts that hold a fire bright or dim for seconds.
    """
    a = S.amplitude
    _ys, _xs, M = _series(_studio_like_layout(), 60.0)
    w = int(TPS)
    win = M[: (len(M) // w) * w].reshape(-1, w, M.shape[1]).mean(axis=1)
    assert np.abs(win - 1.0).max() < a / 3.0, np.abs(win - 1.0).max()
    assert np.abs(win - 1.0).mean() < a / 6.0
    assert np.abs(M.mean(axis=0) - 1.0).max() < 0.02


def test_the_flicker_is_bounded_and_never_strobes():
    """PROPERTY (the look: "breathe unevenly, never strobe"): the multiplier
    never leaves [1 - amplitude, 1 + amplitude]; under 5 % of each tile's
    variance sits above 6 Hz (half the 24 Hz tick's Nyquist rate) -- the
    flicker is slow enough that the tick-rate sampling shows a breath and a
    flutter, never a tick-to-tick blink.

    BREAKS IF: the band is raised toward the tick rate, the clip is removed
    (an unbounded factor would also breach the sweep door's 2 ONE), or a
    per-tick random draw replaces the smooth oscillators.
    """
    _ys, _xs, M = _series(_studio_like_layout(), 30.0)
    assert M.min() >= 1.0 - S.amplitude - 1e-12
    assert M.max() <= 1.0 + S.amplitude + 1e-12
    X = M - M.mean(axis=0)
    P = np.abs(np.fft.rfft(X, axis=0)) ** 2
    f = np.fft.rfftfreq(len(M), 1.0 / TPS)
    hi = P[f > 6.0].sum(axis=0) / P[1:].sum(axis=0)
    assert hi.max() < 0.05, hi.max()


def test_the_settings_door_refuses_a_strobe_band_and_a_negative_light():
    """PROPERTY (the config door): [render.flicker] refuses, by name, an
    amplitude outside [0, 1] (above 1 the light would go negative, and 1 + a
    is the sweep door's 2 ONE), a puff_share outside [0, 1], and a band that
    is empty or reaches past 10 Hz (toward the 24 Hz tick's 12 Hz Nyquist
    rate); a missing section is the schema defaults.

    BREAKS IF: a typo'd dial silently produces a strobe or a negative
    emission instead of an error.
    """
    def cfg(**kw):
        return SimpleNamespace(render=SimpleNamespace(flicker=SimpleNamespace(**kw)))
    assert ff.settings_from_config(SimpleNamespace()) == ff.FlickerSettings()
    for bad in (dict(amplitude=1.2), dict(amplitude=-0.1), dict(puff_share=1.5),
                dict(band_lo_hz=3.0, band_hi_hz=2.0), dict(band_hi_hz=14.0),
                dict(band_lo_hz=0.0)):
        with pytest.raises(ValueError, match=r"\[render.flicker\]"):
            ff.settings_from_config(cfg(**bad))
    from config import CFG
    live = ff.settings_from_config(CFG)
    assert 0.0 < live.amplitude <= 1.0 and live.enabled


# ---------------------------------------------------------------------------
# 3. WHERE IT APPLIES: BURNING TILES ONLY, THROUGH THE ONE DOOR
# ---------------------------------------------------------------------------
class _Pack:
    def __init__(self, src, window):
        self.sources, self.window = src, window


def test_rows_name_only_burning_tiles_and_cross_the_door_in_q16():
    """PROPERTY ("multiplier = exactly 1 on non-burning cells"): the rows the
    renderer hands the sweep name exactly the flame's BURNING tiles (in map
    coordinates, a cropped window included) -- a hot tile below the flame
    floor, a cold tile, a wall get no row, i.e. the sweep's identity ONE --
    with mult_q = round(m * 2^16) inside the sweep door [0, 2 ONE], one row
    per tile; disabled, or with no pack, there are no rows at all.

    BREAKS IF: the rows come from a temperature threshold instead of the
    flame's source law (a hot wall or a lamp's cell would flicker), the
    window offset is dropped, or the door rounds or clamps differently from
    the sweep's bound.
    """
    H, W = 30, MAP_W
    fire = np.zeros((H, W))
    fire[12, 20:25] = 0.6
    fire[20, 40] = 0.005                      # below i_min: no flame, no row
    tk = np.full((H, W), 300.0)
    tk[fire > 0] = 1200.0
    tk[5, 5] = 1500.0                         # hot, not burning: no row
    z = np.zeros((H, W), bool)
    win = (16, 8, 48, 26)                     # (x0, y0, x1, y1): a cropped pack
    sl = (slice(win[1], win[3]), slice(win[0], win[2]))
    src = ref.flame_sources(fire[sl], tk[sl], z[sl], (fire > 0)[sl], z[sl],
                            t_amb_k=293.0, glow_min_k=800.0, tile_size_m=0.333,
                            dials=DIALS, origin=(win[1], win[0]), map_w=W)
    flick = ff.FireFlicker(S)
    rows = flick.rows(_Pack(src, win), 12.3, DIALS, W, H)
    assert rows.dtype == np.int64 and rows.shape == (5, 3)
    assert sorted(map(tuple, rows[:, :2])) == [(12, x) for x in range(20, 25)]
    ys, xs, hz, ph = ff.burning_tiles(src, win)
    m = ff.multipliers(ys, xs, hz, ph, 12.3, S, DIALS, W)
    assert np.array_equal(rows[:, 2], np.rint(m * 65536).astype(np.int64))
    assert fl.LIGHT_EMIT_MULT_MAX == int(bp.RadiationSweep.LIGHT_EMIT_MULT_MAX)
    q = fl.emit_mult_rows([0, 0, 1, 99], [0, 0, 1, 1], [5.0, 0.5, -1.0, 1.0], W, H)
    assert q.tolist() == [[0, 0, fl.LIGHT_EMIT_MULT_MAX], [1, 1, 0]]
    flick.toggle()
    assert not flick.enabled and flick.rows(_Pack(src, win), 12.3, DIALS, W, H) is None
    flick.toggle()
    assert flick.rows(None, 12.3, DIALS, W, H) is None


# ---------------------------------------------------------------------------
# 4. THE GAME: LIGHT ONLY, NEVER HEAT; DISABLED IS THE PRE-F4b LIGHT
# ---------------------------------------------------------------------------
def _flicker_rows_for(sim, t):
    """The game's rows for this tick: the flame's burning tiles of the CURRENT
    sim state (the renderer's pack, built headless) at VFX time t."""
    from simulation import fire_fixed
    from temperature_scale import load as load_tscale
    from renderer.blackbody import TEMP_SCALE, BlackbodyRamp
    from config import CFG
    g = sim.gmap
    scales = ref.CanonicalScales(tscale=load_tscale(CFG), ramp=BlackbodyRamp.from_config(CFG))
    tk = scales.to_kelvin(np.asarray(g.temperature, dtype=np.float64) / TEMP_SCALE)
    src = ref.flame_sources(fire_fixed.dequantize_f32(g.fire), tk, g.solid, g.flammable,
                            g.is_vacuum, t_amb_k=scales.t_amb_k,
                            glow_min_k=scales.glow_min_k, tile_size_m=g.tile_size_m,
                            dials=DIALS)
    h, w = g.solid.shape
    return ff.FireFlicker(S).rows(_Pack(src, (0, 0, w, h)), t, DIALS, w, h)


def test_heat_and_the_golden_are_untouched_by_the_flicker():
    """PROPERTY (design 6.3 "no heat, no digest, no golden"): the canonical
    golden scenario with the game's light requested AND emission-multiplier
    rows handed through Simulation.set_light every tick -- on EVERY interior
    cell, factors spread over the whole door [0, 2 ONE] and changing per tick
    (a harsher flicker than the game's, which names only burning tiles) --
    produces the committed GOLDEN_AGGREGATE. Non-vacuous: the engine held the
    rows on every tick. (The scenario's small fire never reaches the visible
    glow floor in 30 ticks, so the light-moves half is the playground test
    below.)

    BREAKS IF: the emission multiplier reaches any synced integer -- the heat
    planes, temperature, fire, gas, a unit -- anywhere in the 30 ticks.
    """
    from field_ab_harness import capture_trajectory, default_scenario_sim
    from field_digest import trajectory_digest
    from _xarch_perfield_digest import GOLDEN_AGGREGATE, N_STEPS
    held = []
    sky = np.full((16, 3), 1 << 30, dtype=np.int64)
    rng = np.random.default_rng(93)

    def make_sim():
        sim = default_scenario_sim()
        g = sim.gmap
        cells = np.argwhere(~g.solid & ~g.is_vacuum)
        step = sim.step

        def step_with_flicker():
            m = rng.uniform(0.0, 2.0, len(cells))
            h, w = g.solid.shape
            rows = fl.emit_mult_rows(cells[:, 0], cells[:, 1], m, w, h)
            sim.set_light(True, None, sky, rows)
            step()
            held.append(sim.physics_runner.light_emit_mult is not None
                        and len(sim.physics_runner.light_emit_mult) == len(cells))
        sim.step = step_with_flicker
        return sim

    traj = capture_trajectory(make_sim=make_sim, n_steps=N_STEPS)
    assert trajectory_digest(traj) == GOLDEN_AGGREGATE
    assert all(held) and len(held) == N_STEPS


def test_disabled_flicker_is_the_pre_f4b_light_and_heat_is_identical_on_off():
    """PROPERTY ("disabled -> light_q bit-identical to before F4b"; "heat
    identical on/off"): on a lit burning playground, ticking with the flicker
    rows ON vs with None (what a disabled flicker hands over) vs with the
    pre-F4b call (set_light without the argument): the None and pre-F4b runs
    are identical in EVERY GameMap array including light_q, and the ON run
    differs from them in the light planes only -- temperature, gas_energy,
    fire, every heat plane equal.

    BREAKS IF: None is not the no-op (a default multiplier other than ONE, a
    stale row kept across ticks), or the flicker leaks into any non-light array.
    """
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_WOOD
    from level_loader import load as load_level
    level = load_level("playground")

    def mk():
        sim = Simulation(level, seed=5, breach_physics=bp, enable_recorder=False)
        g = sim.gmap
        ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
        for k in range(min(6, len(ys))):
            g.temperature[ys[k], xs[k]] = 1263 << 16
            g.fire[ys[k], xs[k]] = fire_fixed.quantize_scalar(0.8)
        return sim

    on, off, pre = mk(), mk(), mk()
    sky = np.full((16, 3), 3 << 28, dtype=np.int64)
    n_rows = 0
    for t in range(8):
        rows = _flicker_rows_for(on, 5.0 + t / TPS)
        n_rows += 0 if rows is None else len(rows)
        on.set_light(True, None, sky, rows)
        off.set_light(True, None, sky, None)
        pre.set_light(True, None, sky)
        for s in (on, off, pre):
            s.set_paused(False)
            s.step()
    assert n_rows > 0, "nothing burned visibly: vacuous"
    light = {"light_q", "light_flux_q", "light_glow"}
    moved = False
    for k, v in vars(off.gmap).items():
        if not isinstance(v, np.ndarray):
            continue
        assert np.array_equal(v, getattr(pre.gmap, k), equal_nan=True), k
        same = np.array_equal(v, getattr(on.gmap, k), equal_nan=True)
        if k in light:
            moved = moved or not same
        else:
            assert same, f"the flicker moved gmap.{k}"
    assert moved, "the flicker never moved the light: vacuous"
