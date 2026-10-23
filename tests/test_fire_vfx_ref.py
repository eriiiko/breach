"""Property tests for the flame VFX's numpy spec twin (graphics arc G3, patch
F0b; docs/g3_fire_design_2026-10-23.md §8 and §13).

The twin (``renderer/fire_vfx_ref.py``) is the spec F1/F2's GL loop is checked
against, so its invariants are pinned here as PROPERTIES: each test names the
property and the change that must break it. Nothing pins an exact image,
count or set. Pure numpy, no GL, no breach_physics; small grids (M = 4 where
the property does not depend on resolution) keep the file at seconds.

Run:
    C:/Users/steen/anaconda3/envs/data/python.exe -m pytest tests/test_fire_vfx_ref.py -q
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from renderer import fire_vfx_ref as fv  # noqa: E402

DUMP = ROOT / "tests" / "data" / "fire_smoke_studio_flame_inputs.npz"
SRC = ROOT / "renderer" / "fire_vfx_ref.py"


@pytest.fixture(scope="module")
def scales():
    return fv.load_scales()


def _tick(scales, fire, t_k, solid=None, flammable=None, vacuum=None,
          wind=None, dials=None, tps=24.0):
    h, w = fire.shape
    z = np.zeros((h, w), dtype=bool)
    solid = z if solid is None else solid
    flammable = (fire > 0) if flammable is None else flammable
    vacuum = z if vacuum is None else vacuum
    wind = np.zeros((h, w, 2)) if wind is None else wind
    return fv.prepare_tick(fire, scales.tscale.from_kelvin(t_k), solid,
                           flammable, vacuum, wind, tps=tps, tile_size_m=0.333,
                           scales=scales, dials=dials or fv.FireVFXDials())


def _lone_fire(n=8, at=(4, 3), i=0.45, t_fire=1300.0, t_gas=450.0):
    fire = np.zeros((n, n))
    fire[at] = i
    tk = np.full((n, n), t_gas)
    tk[at] = t_fire
    return fire, tk


def _run(tick, dials, n_sub, state=None, each=None):
    modes = fv.curl_noise_modes(dials)
    st = state or fv.FireVFXState.zeros(*tick.s.shape)
    for k in range(n_sub):
        st = fv.substep(st, tick, dials, modes)
        if each is not None:
            each(k, st)
    return st


# ---------------------------------------------------------------------------
# The loop's invariants (§8 "spec twin", §13.5, §13.6)
# ---------------------------------------------------------------------------

def _hostile_scene(scales, dials):
    """Two fuels at different temperatures, a wood wall, vacuum, and a
    strong real-shaped wind: the case where advection could make new
    extrema or carry flame into walls."""
    n = 10
    fire = np.zeros((n, n)); tk = np.full((n, n), 400.0)
    fire[5, 3] = 0.6; tk[5, 3] = 1450.0           # hot furniture
    fire[6, 5] = 0.3; tk[6, 5] = 950.0            # cool furniture
    solid = np.zeros((n, n), bool); solid[3, :] = True    # wood wall row 3
    fl = fire > 0
    fl[3, :] = True
    fire[3, 4] = 0.5; tk[3, 4] = 1200.0           # a burning wood tile
    vac = np.zeros((n, n), bool); vac[:, 8:] = True       # vacuum band east
    rng = np.random.default_rng(3)
    wind = np.zeros((n, n, 2))
    wind[..., 0] = 0.12 + 0.05 * rng.standard_normal((n, n))   # ~3 tiles/s
    wind[..., 1] = 0.05 * rng.standard_normal((n, n))
    tick = _tick(scales, fire, tk, solid, fl, vac, wind, dials)
    tick.tile_blocked = solid | vac            # the test's own record of the masks
    return tick


def test_flame_density_never_negative_and_never_hotter_than_its_source(scales):
    """Property: after EVERY substep, R >= 0 and no flame cell is hotter than
    the hottest source temperature (T_vfx <= T_src), in a scene with
    sources at different temperatures and a strong, uneven wind. Breaks if
    the temperature limiter on R*T is removed (R and R*T are each bounded
    by their own stencil, their ratio is not -- with Selle's revert alone
    the ratio still overshoots), or if injection adds heat without flame
    (s*T_src without s)."""
    d = fv.FireVFXDials(m=4)
    tick = _hostile_scene(scales, d)
    t_src = float(tick.t_src_k.max())
    worst = {"rmin": 0.0, "tmax": 0.0}

    def check(k, st):
        worst["rmin"] = min(worst["rmin"], float(st.R.min()))
        live = st.R > 1e-4
        if live.any():
            worst["tmax"] = max(worst["tmax"], float((st.RT[live] / st.R[live]).max()))

    _run(tick, d, 360, each=check)
    assert worst["rmin"] >= 0.0
    assert worst["tmax"] <= t_src * (1.0 + 1e-9), (worst["tmax"], t_src)


def test_no_flame_in_solids_or_vacuum_after_any_substep(scales):
    """Property: R and R*T are exactly 0 in solid and vacuum cells after
    every substep (§3.4 step 5, §13.6), even with a fire burning against a
    wall and a wind blowing toward vacuum. The blocked cells are derived
    here from the scene's own tile masks, not from the twin's. Breaks if
    the blocked-cell clear is dropped or vacuum is left out of it."""
    d = fv.FireVFXDials(m=4)
    tick = _hostile_scene(scales, d)
    blocked = fv._upsample_nearest(tick.tile_blocked, d.m)
    leaks = []

    def check(k, st):
        if st.R[blocked].any() or st.RT[blocked].any():
            leaks.append(k)

    st = _run(tick, d, 240, each=check)
    assert not leaks, f"flame inside solid/vacuum at substeps {leaks[:5]}"
    assert st.R[~blocked].max() > 0.1, "scene made no flame -- test is vacuous"


def test_cooling_never_heats(scales):
    """Property (§13.5): the decay/cool stage never raises a flame cell's
    temperature, even where the local game gas is HOTTER than the flame
    (a cool flame drifting over a hot tile). Breaks if cooling becomes a
    plain relaxation toward the gas temperature (``T <- relax(T, T_gas)``
    without the ``min``)."""
    d = fv.FireVFXDials(m=4)
    fire, tk = _lone_fire(t_gas=1400.0, t_fire=1300.0)
    tick = _tick(scales, fire, tk, dials=d)
    rng = np.random.default_rng(0)
    R = rng.uniform(0.2, 2.0, tick.s.shape)
    T = rng.uniform(850.0, 1350.0, tick.s.shape)
    R2, RT2 = fv.decay_cool(R, R * T, tick, d.dt_s, d)
    live = R2 > 0
    assert live.any()
    assert (RT2[live] / R2[live] <= T[live] * (1 + 1e-12)).all()


def test_no_fire_means_no_flame_and_a_dead_fire_leaves_nothing(scales):
    """Property: with no fire, R stays identically 0; and once a fire's
    sources stop, R is identically 0 everywhere within
    ``extinction_time_s`` (no stale flame lingers). Breaks if anything
    injects without a source, or if the r_cut clear is removed (exponential
    decay alone never reaches 0)."""
    d = fv.FireVFXDials(m=4)
    fire, tk = _lone_fire()
    cold = _tick(scales, np.zeros_like(fire), tk, dials=d)
    st = _run(cold, d, 60)
    assert not st.R.any() and not st.RT.any()

    hot = _tick(scales, fire, tk, dials=d)
    st = _run(hot, d, 120)
    r_peak = float(st.R.max())
    assert r_peak > 0.1
    n_ext = int(np.ceil(fv.extinction_time_s(d, r_peak) / d.dt_s)) + 1
    st = _run(cold, d, n_ext, state=st)
    assert not st.R.any() and not st.RT.any()


def test_sealed_room_pressure_stays_bounded(scales):
    """Property (§13.1, the F0b gate): a fire inside a closed 6x6 room keeps
    max|p| bounded over 10^4 substeps -- the screened solve has a unique
    bounded solution although no flow can leave the room and sum(q) > 0.
    Checked as: finite, and no growth between the middle and the end of the
    run. Breaks if the screening term is removed (pure Neumann + sum q > 0:
    warm-started Jacobi grows p without limit) -- the second half of the
    test runs exactly that change and requires the growth to show."""
    d = fv.FireVFXDials(m=2, n_jacobi=4, curl_modes=2)       # cheap: 10^4 substeps
    n = 10
    solid = np.zeros((n, n), bool)
    solid[1, 1:9] = solid[8, 1:9] = solid[1:9, 1] = solid[1:9, 8] = True
    fire, tk = _lone_fire(n=n, at=(4, 4))
    tick = _tick(scales, fire, tk, solid=solid, flammable=fire > 0, dials=d)
    assert tick.q.sum() > 0

    def max_p_series(dials, n_sub):
        """The velocity/pressure half of ``substep`` (stages 1-3, the real
        stage functions in its order): the flame scalars never feed back
        into u or p, so they are skipped to keep 10^4 substeps cheap."""
        modes = fv.curl_noise_modes(dials)
        u = np.zeros(tick.s.shape + (2,)); p = np.zeros(tick.s.shape)
        out, t = [], 0.0
        for _ in range(n_sub):
            u = fv.advect_velocity(u, dials.dt_s, dials.m)
            u = fv.follow_wind(u, tick.u_wind, dials.k_follow, dials.dt_s)
            u = fv.confine(u, tick.i_local, tick.solid, t, dials.dt_s, dials, modes)
            rhs = fv.pressure_rhs(u, tick, t, dials)
            for _ in range(dials.n_jacobi):
                p = fv.jacobi_sweep(p, rhs, tick.solid, tick.vacuum, dials.m,
                                    dials.screen_len_tiles)
            u = fv.project(u, p, tick.solid, tick.vacuum, dials.m)
            t += dials.dt_s
            out.append(float(np.abs(p).max()))
        return np.array(out)

    pm = max_p_series(d, 10_000)
    assert np.isfinite(pm).all()
    mid, end = pm[3000:6000].max(), pm[7000:].max()
    assert end <= 1.25 * mid + 1e-6, (mid, end)

    # the breaking change, executed: no screening -> p keeps growing
    pm0 = max_p_series(d.with_(screen_len_tiles=1e9), 2000)
    assert pm0[-1] > 2.0 * pm0[600], "unscreened room did not grow: test is blind"


def test_expansion_survives_the_projection(scales):
    """Property (§3.3 B3): the prescribed expansion q is IN the pressure
    solve, so the projected velocity diverges outward over the burning
    tile (mean div u > 0 there). Breaks if q is applied as a velocity kick
    the projection then removes, or dropped from the rhs."""
    d = fv.FireVFXDials(m=4, curl_amp=0.0, vort_eps=0.0)
    fire, tk = _lone_fire()
    tick = _tick(scales, fire, tk, dials=d)
    st = _run(tick, d, 60)
    div = fv.divergence(st.u, tick.solid, d.m)
    src = tick.s > 0
    assert div[src].mean() > 0.25 * tick.q[src].mean()


def test_flame_leans_downwind(scales):
    """Property (§13.2, the F0b look gate in numbers): in a steady wind the
    flame's tongues (emission outside the source cells) sit downwind of
    where they sit in still air. Breaks if the wind does not enter the
    flame velocity (k_follow = 0, or tiles/tick not scaled by tps)."""
    d = fv.FireVFXDials(m=4, seed=11)
    fire, tk = _lone_fire(n=10, at=(5, 4))

    def tongue_x(wind_tiles_per_s):
        wind = np.zeros((10, 10, 2))
        wind[..., 0] = wind_tiles_per_s / 24.0           # TAMED is tiles/tick
        tick = _tick(scales, fire, tk, wind=wind, dials=d)
        xs = []

        def acc(k, st):
            if k >= 120:
                Le, E, _ = fv.flame_emission(st, scales.ramp, "B", d)
                w = Le.sum(-1) * E * (tick.s <= 0)
                xx = np.mgrid[0:w.shape[0], 0:w.shape[1]][1] / d.m
                xs.append((w * xx).sum() / max(w.sum(), 1e-12))
        _run(tick, d, 300, each=acc)
        return float(np.mean(xs))

    assert tongue_x(3.0) - tongue_x(0.0) > 0.1


def test_flame_tiles_flicker_out_of_step(scales):
    """Property (§6.3, Erik's "no sync" worry for flicker D): two equal
    burning tiles four apart have per-tile flicker series (sum of R*Le over
    the tile's 3x3 neighbourhood, per tick) with |correlation| < 0.3.
    Breaks if the puffing becomes one shared clock -- the same phase,
    frequency and wobble for every cluster."""
    d = fv.FireVFXDials(m=4, seed=5)
    n = 12
    fire = np.zeros((6, n)); tk = np.full((6, n), 450.0)
    for x in (3, 7):
        fire[3, x] = 0.45; tk[3, x] = 1300.0
    tick = _tick(scales, fire, tk, dials=d)
    series = {3: [], 7: []}
    m = d.m

    def acc(k, st):
        if k >= 120 and k % 5 == 0:                        # per tick, after 1 s
            _, inten = scales.ramp.emission_at_kelvin(st.temperature(d.r_eps))
            lum = st.R * np.where(st.R > d.r_eps, inten, 0.0)
            for x in series:
                series[x].append(lum[2 * m:5 * m, (x - 1) * m:(x + 2) * m].sum())
    _run(tick, d, 120 + 5 * 140, each=acc)
    a, b = (np.asarray(series[x]) for x in (3, 7))
    assert a.std() > 0 and b.std() > 0
    rho = np.corrcoef(a, b)[0, 1]
    assert abs(rho) < 0.3, rho


def test_a_and_b_emission_differ(scales):
    """Property (§3.5, minor 9): the A ("temperature only") and B
    ("temperature + thickness") opacities differ visibly on a burning
    tile's flame with the default dials. Breaks if r0/r1/kappa are retuned
    so the A/B Erik judges becomes meaningless."""
    d = fv.FireVFXDials(m=4)
    fire, tk = _lone_fire()
    st = _run(_tick(scales, fire, tk, dials=d), d, 240)
    Le, Ea, _ = fv.flame_emission(st, scales.ramp, "A", d)
    _, Eb, _ = fv.flame_emission(st, scales.ramp, "B", d)
    lit = Le.sum(-1) > 0.05
    assert np.abs(Ea - Eb)[lit].mean() > 0.05


def test_stages_never_write_their_inputs(scales):
    """Property (render-only; F2's per-operator parity): every stage reads
    its inputs and returns new arrays -- the tick inputs (from read-only
    sim copies) and the previous state are never written. Breaks if any
    stage updates an array in place (it raises on the read-only views)."""
    d = fv.FireVFXDials(m=4)
    fire, tk = _lone_fire()
    ro = lambda a: np.lib.stride_tricks.as_strided(a, writeable=False)  # noqa: E731
    h, w = fire.shape
    tick = fv.prepare_tick(ro(fire), ro(scales.tscale.from_kelvin(tk)),
                           ro(np.zeros((h, w), bool)), ro(fire > 0),
                           ro(np.zeros((h, w), bool)), ro(np.zeros((h, w, 2))),
                           tps=24.0, tile_size_m=0.333, scales=scales, dials=d)
    st = _run(tick, d, 30)
    for name in ("s", "q", "t_src_k", "puff_hz", "puff_phase", "solid",
                 "blocked", "vacuum", "u_wind", "t_gas_k", "i_local"):
        getattr(tick, name).flags.writeable = False
    for name in ("u", "p", "R", "RT"):
        getattr(st, name).flags.writeable = False
    fv.substep(st, tick, d)                                   # must not raise


def test_module_is_render_only():
    """Property (§8 render-only, structurally): the twin imports no pyray,
    no breach_physics and no sim writer, and never assigns into a GameMap.
    Breaks if VFX code starts touching sim state or GL."""
    code = SRC.read_text(encoding="utf-8")
    assert not re.search(r"^\s*(import|from)\s+(pyray|raylib|breach_physics)", code, re.M)
    assert not re.search(r"^\s*from\s+simulation", code, re.M)
    assert not re.search(r"\bgmap\.\w+\s*(\[[^\]]*\])?\s*[+\-*/]?=(?!=)", code)


def test_canonical_scales_not_literals(scales):
    """Property: the glow floor and the ambient come from the ONE ramp and
    the ONE temperature map, never from literals in the twin. Breaks if a
    Kelvin constant (the glow floor, 293 K ambient) is hardcoded there or
    the scales stop following ``[render.blackbody]`` /
    ``[physics.temperature_scale]``."""
    from config import CFG
    from renderer.blackbody import BlackbodyRamp
    from temperature_scale import load
    assert scales.glow_min_k == BlackbodyRamp.from_config(CFG).kelvin_glow_min
    assert scales.t_amb_k == load(CFG).kelvin_ambient
    code = re.sub(r'"""[\s\S]*?"""', "", SRC.read_text(encoding="utf-8"))
    code = re.sub(r"#[^\n]*", "", code)
    for lit in (str(int(scales.glow_min_k)), str(int(scales.t_amb_k))):
        assert not re.search(rf"\b{lit}(\.0*)?\b", code), lit


# ---------------------------------------------------------------------------
# The source law (§3.3, §13.4) on randomised tile maps
# ---------------------------------------------------------------------------

def _random_map(rng, h=14, w=14):
    solid = rng.random((h, w)) < 0.25
    vacuum = (rng.random((h, w)) < 0.08) & ~solid
    flammable = rng.random((h, w)) < 0.5
    fire = np.where(flammable & (rng.random((h, w)) < 0.6), rng.random((h, w)), 0.0)
    tk = rng.uniform(300.0, 1500.0, (h, w))
    return fire, tk, solid, flammable, vacuum


def _law(scales, fire, tk, solid, fl, vac, dials=fv.FireVFXDials()):
    return fv.flame_sources(fire, tk, solid, fl, vac, t_amb_k=scales.t_amb_k,
                            glow_min_k=scales.glow_min_k, tile_size_m=0.333,
                            dials=dials)


@pytest.mark.parametrize("seed", range(6))
def test_sources_sit_on_or_beside_burning_fuel_by_class(scales, seed):
    """Property: every source tile is fed by a burning tile that is (a)
    itself, for permeable fuel (whole tile), or (b) a 4-neighbour of solid
    fuel, with the strip on the face they share; no source is in a solid or
    vacuum tile; T_src is the feeding tile's own temperature; nothing burns
    below the glow floor or in vacuum. Breaks if injection leaks past the
    fuel classes, lands in walls/vacuum, detaches its colour from the tile,
    or ignores §13.4's floor."""
    fire, tk, solid, fl, vac = _random_map(np.random.default_rng(seed))
    src = _law(scales, fire, tk, solid, fl, vac)
    h, w = fire.shape
    face_toward = {(-1, 0): fv.FACE_N, (1, 0): fv.FACE_S,
                   (0, -1): fv.FACE_W, (0, 1): fv.FACE_E}
    assert not src.cover[solid | vac].any()
    assert not src.burning[(tk < scales.glow_min_k) | vac | ~fl].any()
    ys, xs = np.nonzero(src.cover)
    assert len(ys) > 0
    for y, x in zip(ys, xs):
        fy, fx = divmod(int(src.feeder[y, x]), w)
        assert src.burning[fy, fx]
        assert src.t_src_k[y, x] == tk[fy, fx]
        if (fy, fx) == (y, x):
            assert not solid[fy, fx] and src.cover[y, x] == fv.COVER_FULL
        else:
            assert solid[fy, fx] and (abs(fy - y) + abs(fx - x)) == 1
            if src.cover[y, x] != fv.COVER_FULL:
                assert src.cover[y, x] & face_toward[(fy - y, fx - x)]
    # every burning permeable tile covers itself; every burning solid fuel
    # tile with an open neighbour feeds a strip there
    open_air = ~solid & ~vac
    for y, x in zip(*np.nonzero(src.burning)):
        if not solid[y, x]:
            assert src.cover[y, x] == fv.COVER_FULL
        else:
            for (dy, dx), bit in face_toward.items():
                ny, nx = y - dy, x - dx
                if 0 <= ny < h and 0 <= nx < w and open_air[ny, nx]:
                    assert src.cover[ny, nx] & (bit | fv.COVER_FULL)
    # and per cell: no injection or expansion inside a solid or vacuum
    s, q, *_ = fv.rasterize_sources(src, 4, 0.375)
    blocked = fv._upsample_nearest(solid | vac, 4)
    assert not s[blocked].any() and not q[blocked].any()


def test_source_rate_and_expansion_are_monotone_in_intensity(scales):
    """Property: raising a burning tile's fire intensity never lowers its
    source rate s or expansion q, and strictly raises s (a dying fire
    shrinks for free). Breaks if s or q stop scaling with I."""
    tk = np.full((3, 3), 400.0); tk[1, 1] = 1200.0
    z = np.zeros((3, 3), bool)
    fl = z.copy(); fl[1, 1] = True
    prev_s, prev_q = -1.0, -1.0
    for i in np.linspace(0.05, 1.0, 12):
        fire = np.zeros((3, 3)); fire[1, 1] = i
        src = _law(scales, fire, tk, z, fl, z)
        assert src.s[1, 1] > prev_s and src.q[1, 1] >= prev_q
        prev_s, prev_q = src.s[1, 1], src.q[1, 1]


def test_cold_fire_makes_no_flame(scales):
    """Property (§13.4): a tile whose temperature is below the ramp's glow
    floor makes no source however strongly it 'burns' (the ~300 K look-shot
    air fires). Breaks if the glow-floor gate is dropped."""
    z = np.zeros((3, 3), bool)
    fl = z.copy(); fl[1, 1] = True
    fire = np.zeros((3, 3)); fire[1, 1] = 1.0
    tk = np.full((3, 3), scales.glow_min_k - 1.0)
    assert not _law(scales, fire, tk, z, fl, z).cover.any()
    tk[1, 1] = scales.glow_min_k + 1.0
    assert _law(scales, fire, tk, z, fl, z).cover[1, 1] == fv.COVER_FULL


def test_real_studio_fires_are_under_their_own_flame(scales):
    """Property, on the real dump (``tools/fire_smoke_probe.py --dump``):
    every burning furniture tile of fire_smoke_studio sources its own
    flame, at its own temperature, and nothing is sourced in the hull.
    Breaks if the source law drifts from the sim's real fields (masks,
    the dequantized I, the canonical Kelvin map)."""
    d = np.load(DUMP)
    for k in range(len(d["ticks"])):
        tk = scales.to_kelvin(d["t_game"][k])
        src = _law(scales, d["fire"][k], tk, d["solid"], d["flammable"], d["is_vacuum"])
        hot = (d["fire"][k] > 0.05) & (tk >= scales.glow_min_k)
        assert hot.any()
        assert (src.cover[hot] == fv.COVER_FULL).all()
        assert np.array_equal(src.t_src_k[hot], tk[hot])
        assert not src.cover[d["solid"]].any()
