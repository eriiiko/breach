"""THE LIGHT CHANNELS on the radiation sweep, CPU (ray-engine-v2 P6a).

docs/ray_engine_v2_p6a_light_channels_brief_2026-09-25.md section 3; design v3
sections 4, 5, 6.3, 7, 8. The integer reference (sweep_ref_q.py's LightGroup,
gate 18) is the spec; this file holds the ENGINE to it (gate 0 for light) and
to the brief's properties on the engine's own output. The CUDA twin is held to
this CPU sweep by tests/test_cuda_radiation_sweep_light.py.

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_light.py -q
"""
from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import (  # noqa: E402
    ONE, R, T_AMB_Q, TRANSPORTS, as_i32, bp, cpp_sweep, engine_currency,
    light_arrays, light_out, live_table, random_gas, random_scene, ref_sweep,
    reference_table)

Q = R.quant
K_LEAK = Q(0.10)


# ---------------------------------------------------------------------------
# scene vocabulary
# ---------------------------------------------------------------------------
def light_scene(rng, h, w):
    """Random light extinction (channel-first lists): clear air, the shipped
    glass (0.1) and crate (0.55), opaque walls, two asymmetric tints, and
    bodies (d = ONE) stamped on some cells."""
    tints = [(0, 0, 0), (0, 0, 0), (Q(0.1),) * 3, (Q(0.55),) * 3, (ONE,) * 3,
             (Q(0.2), Q(0.6), Q(0.9)), (Q(0.9), Q(0.1), Q(0.4))]
    la = [[[0] * w for _ in range(h)] for _ in range(3)]
    ld = [[[0] * w for _ in range(h)] for _ in range(3)]
    for y in range(h):
        for x in range(w):
            t = rng.choice(tints)
            body = rng.random() < 0.15
            for c in range(3):
                la[c][y][x] = t[c]
                ld[c][y][x] = ONE if body else t[c]
    return la, ld


def light_cols(rng, n):
    lab = [[rng.choice([0, Q(0.14), Q(1.26), Q(2.5)]) for _ in range(3)] for _ in range(n)]
    lgl = [[rng.choice([0, Q(0.04), Q(0.92)]) for _ in range(3)] for _ in range(n)]
    return lab, lgl


def clear_room(n, sources=(), a_src=ONE):
    """An n x n transparent room; `sources` = [((y, x), T_game)] light-opaque
    (a = d = a_src) emitters. Returns (heat scene tuple, light dict)."""
    a = [[0] * n for _ in range(n)]
    d = [[0] * n for _ in range(n)]
    T = [[0] * n for _ in range(n)]
    his = [[3] * n for _ in range(n)]
    ts = [[0] * n for _ in range(n)]
    la = [[[0] * n for _ in range(n)] for _ in range(3)]
    ld = [[[0] * n for _ in range(n)] for _ in range(3)]
    for (y, x), tg in sources:
        T[y][x] = tg << 16
        for c in range(3):
            la[c][y][x] = ld[c][y][x] = a_src
    return (a, d, T, his, ts), dict(la=la, ld=ld)


def engine(sc, light, *, table=None, transport="shear", n_ord=16, k_q=0, **kw):
    """The C++ sweep with the light group; returns the harness 7-tuple."""
    a, d, T, his, ts = sc
    return cpp_sweep(a, d, k_q, T, his, ts, transport=transport, n_ord=n_ord,
                     table=table if table is not None else live_table(),
                     light=light, **kw)


# ---------------------------------------------------------------------------
# 1. GATE 0 FOR LIGHT
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("htr", ["shear", "step"])
@pytest.mark.parametrize("ltr", ["step", "shear"])
@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("tname", ["live", "resolving"])
def test_cpp_light_channels_reproduce_the_reference_bit_for_bit(seed, n_ord, htr, ltr,
                                                                smoke, tname):
    """PROPERTY (brief 3.1, gate 0 for light): the C++ sweep's light_q,
    light_flux_q and light_glow, its twelve light books and its light-stream
    telemetry EQUAL sweep_ref_q's LightGroup, integer for integer -- and so do
    its four heat planes, Fleck plane and heat telemetry, on the same call --
    over gate 0's randomised scenes with light tints, glass, bodies and the
    whole temperature range, S16 and S12, heat shear/step x light step/shear,
    the smoke term on and off, the live fine table and the resolving one.
    Non-vacuous: light_q and the flux non-zero, glow non-zero with smoke.

    BREAKS IF: any light term is transcribed differently in C++ -- the split's
    remainder, the dark ring, the stamped absorption, the per-ordinate emission,
    the symmetric flux shift, the glow product, the smoke term's per-channel
    law, the light upwind pair of either transport, the direction-cosine
    literals. Validated by breaking radiation_sweep.cpp once (see the report).
    """
    rng = random.Random(20260925 + seed)
    h, w = rng.choice([(7, 9), (9, 11), (12, 8)])
    a, d, T, his, ts = random_scene(rng, h, w)
    if tname == "live":
        his = [[rng.choice([-3, -2, 3, 5]) for _ in range(w)] for _ in range(h)]
    la, ld = light_scene(rng, h, w)
    light = dict(la=la, ld=ld, transport=ltr)
    kw = {}
    if smoke:
        gas, hq, nb = random_gas(rng, h, w)
        lab, lgl = light_cols(rng, len(gas))
        light.update(light_absorb_q=lab, light_glow_q=lgl)
        kw = dict(gas=gas, hq=hq, n_bulk=nb)
    table, rtab = (live_table(), R.E_LIVE) if tname == "live" else (reference_table(), R.E)
    got = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, n_ord=n_ord, table=table,
                    light=light, **kw)
    exp = ref_sweep(a, d, K_LEAK, T, his, transport=htr, n_ord=n_ord, ts=ts,
                    table=rtab, light=light, **kw)
    for name, g, e in zip(("rad_net", "rad_flux", "rad_amb", "rad_fluence", "fleck"),
                          got[:5], exp[:5]):
        assert np.array_equal(np.asarray(g, dtype=np.int64), e), name
    assert (got[5].min_stream, got[5].max_stream) == (exp[5].min_stream, exp[5].max_stream)
    gl, el = got[6], exp[6]
    for k in ("light_q", "light_flux", "light_glow"):
        if not np.array_equal(gl[k], el[k]):
            bad = np.argwhere(gl[k] != el[k])
            pytest.fail(f"{k} differs at {len(bad)} cells; first {tuple(bad[0])} "
                        f"cpp={int(gl[k][tuple(bad[0])])} ref={int(el[k][tuple(bad[0])])}")
    assert dict(gl["books"]) == el["books"]
    assert np.any(gl["light_q"] != 0) and np.any(gl["light_flux"] != 0)
    if smoke:
        assert np.any(gl["light_glow"] != 0), "no glow with smoke: vacuous"
        # the smoke term reached the loop: some gas cell reads a > its material a
        la_eff = np.moveaxis(np.asarray(got[5].light_a_eff_plane()), -1, 0)
        assert np.any(la_eff > np.asarray(la)), "no gas cell took the light smoke term"


def test_cpp_light_reproduces_the_reference_on_the_optics_scenes():
    """PROPERTY (brief 3.1, the light-specific scenes): gate 0 on the scenes the
    optics properties are made of -- a hot source column behind a wall with a
    gap, behind a glass column, behind poison and soot columns at several
    densities, a body column, and the table top everywhere (the over-driven
    headroom scene: an int64 wrap in C++ would show here first).

    BREAKS IF: C++ and the reference part on any of these (the same breaks as
    the matrix above, and any int64 overflow at the table top).
    """
    n = 11
    xw = n // 2
    top = R.T_TABLE_TOP_GAME
    scenes = []
    for kind in ("gap", "glass", "body", "poison", "soot", "top"):
        sc, light = clear_room(n)
        a, d, T, his, ts = sc
        for y in range(n):
            T[y][0] = (3000 if kind in ("poison", "soot") else 1263) << 16
            for c in range(3):
                light["la"][c][y][0] = light["ld"][c][y][0] = ONE
        gas = None
        if kind == "gap":
            for y in range(n):
                if y != n // 2:
                    for c in range(3):
                        light["la"][c][y][xw] = light["ld"][c][y][xw] = ONE
        elif kind == "glass":
            for y in range(n):
                a[y][xw] = d[y][xw] = Q(0.3)
                ts[y][xw] = 1
                for c in range(3):
                    light["la"][c][y][xw] = light["ld"][c][y][xw] = Q(0.1)
        elif kind == "body":
            for y in range(n):
                T[y][xw] = 3000 << 16
                for c in range(3):
                    light["ld"][c][y][xw] = ONE
        elif kind in ("poison", "soot"):
            for y in range(n):
                ts[y][0] = 1
            dens = Q(0.4)
            smoke = [[dens if x == xw else 0 for x in range(n)] for _ in range(n)]
            o2 = [[13763] * n for _ in range(n)]
            n2 = [[51773] * n for _ in range(n)]
            nb = [[o2[y][x] + n2[y][x] for x in range(n)] for y in range(n)]
            gas = ([smoke, o2, n2], [0, 0, 0], nb)
            coef = ([Q(0.45 * 1.4), Q(0.10 * 1.4), Q(0.80 * 1.4)] if kind == "poison"
                    else [Q(0.88 * 1.4), Q(0.90 * 1.4), Q(0.93 * 1.4)])
            light.update(light_absorb_q=[coef, [0] * 3, [0] * 3],
                         light_glow_q=[[Q(0.5)] * 3, [0] * 3, [0] * 3])
        elif kind == "top":
            for y in range(n):
                for x in range(n):
                    T[y][x] = top << 16
                    a[y][x] = d[y][x] = ONE
                    for c in range(3):
                        light["la"][c][y][x] = light["ld"][c][y][x] = ONE
        scenes.append((kind, sc, light, gas))
    for kind, sc, light, gas in scenes:
        a, d, T, his, ts = sc
        kw = {} if gas is None else dict(gas=gas[0], hq=gas[1], n_bulk=gas[2])
        for htr in ("shear", "step"):
            got = cpp_sweep(a, d, 0, T, his, ts, transport=htr, table=live_table(),
                            light=light, **kw)
            exp = ref_sweep(a, d, 0, T, his, transport=htr, ts=ts, table=R.E_LIVE,
                            light=light, **kw)
            for k in ("light_q", "light_flux", "light_glow"):
                assert np.array_equal(got[6][k], exp[6][k]), (kind, htr, k)
            assert dict(got[6]["books"]) == exp[6]["books"], (kind, htr)
            for g, e in zip(got[:4], exp[:4]):
                assert np.array_equal(g, e), (kind, htr)
        if kind == "top":
            assert int(got[6]["light_q"].max()) < 2 ** 46


# ---------------------------------------------------------------------------
# 2. HEAT IS UNTOUCHED
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [4, 5])
@pytest.mark.parametrize("htr", ["shear", "step"])
@pytest.mark.parametrize("n_ord", [16, 12])
def test_heat_is_bit_identical_with_light_on_and_off(seed, htr, n_ord):
    """PROPERTY (brief 3.3): the four heat planes, the Fleck plane, the heat
    stream telemetry and the effective heat extinction planes from ONE call with
    the light group equal those from the same call without it, integer for
    integer -- with smoke, bodies, thin rows and the live fine table. Light
    reads nothing heat writes and writes nothing heat reads.

    BREAKS IF: the light code writes a heat plane or a heat book, reuses heat's
    scratch (the outflow store, the effective planes), or changes the walk.
    Validated: booking the light stream into rad_fluence (a one-line leak)
    turns this red.
    """
    rng = random.Random(90 + seed)
    h, w = 9, 12
    a, d, T, _his, ts = random_scene(rng, h, w)
    his = [[rng.choice([-3, 3, 5]) for _ in range(w)] for _ in range(h)]
    la, ld = light_scene(rng, h, w)
    gas, hq, nb = random_gas(rng, h, w)
    lab, lgl = light_cols(rng, len(gas))
    base = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, n_ord=n_ord,
                     table=live_table(), gas=gas, hq=hq, n_bulk=nb)
    lit = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, n_ord=n_ord,
                    table=live_table(), gas=gas, hq=hq, n_bulk=nb,
                    light=dict(la=la, ld=ld, light_absorb_q=lab, light_glow_q=lgl))
    for name, x, y in zip(("rad_net", "rad_flux", "rad_amb", "rad_fluence", "fleck"),
                          base[:5], lit[:5]):
        assert np.array_equal(x, y), f"light moved heat's {name}"
    assert (base[5].min_stream, base[5].max_stream) == (lit[5].min_stream, lit[5].max_stream)
    assert np.array_equal(np.asarray(base[5].a_eff_plane()), np.asarray(lit[5].a_eff_plane()))
    assert np.any(lit[6]["light_q"] != 0) and np.any(base[0] != 0)


# ---------------------------------------------------------------------------
# 3. CONSERVATION PER CHANNEL
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [6, 7, 8])
@pytest.mark.parametrize("ltr", ["step", "shear"])
def test_the_light_books_close_exactly_per_channel(seed, ltr):
    """PROPERTY (brief 3.4): on the light channel's own books, per channel,
    exactly in int64: emitted + entered-from-the-ring == absorbed +
    left-to-the-ring. Entered-from-the-ring is 0 (the ring is dark until P6b's
    sky); emitted, absorbed and left-to-the-ring are each non-zero. The emitted
    book also equals the planes' own arithmetic: n_ord x the per-ordinate
    emission, summed over cells.

    BREAKS IF: a split loses its remainder (two shifts), a ring share is booked
    on one side only, the body's absorption leaves the books, or the emission
    is booked per cell instead of per ordinate. Validated: the light split with
    a second shift in radiation_sweep.cpp turns this red.
    """
    rng = random.Random(300 + seed)
    h, w = 10, 13
    a, d, T, his, ts = random_scene(rng, h, w)
    la, ld = light_scene(rng, h, w)
    gas, hq, nb = random_gas(rng, h, w)
    lab, lgl = light_cols(rng, len(gas))
    for n_ord in (16, 12):
        got = cpp_sweep(a, d, K_LEAK, T, his, ts, n_ord=n_ord, table=live_table(),
                        gas=gas, hq=hq, n_bulk=nb,
                        light=dict(la=la, ld=ld, transport=ltr, light_absorb_q=lab,
                                   light_glow_q=lgl))
        bk = got[6]["books"]
        for c in range(3):
            assert bk["emit"][c] + bk["ring_in"][c] == bk["absorb"][c] + bk["ring_out"][c], c
            assert bk["ring_in"][c] == 0
            assert bk["emit"][c] > 0 and bk["absorb"][c] > 0 and bk["ring_out"][c] > 0
        assert bk["min_stream"] >= 0


# ---------------------------------------------------------------------------
# 4. A DARK WORLD IS EXACTLY DARK
# ---------------------------------------------------------------------------
def test_nothing_hot_means_exactly_no_light():
    """PROPERTY (brief 3.5): with no cell above the glow floor (the room from
    sub-ambient to 480 game, smoke, bodies, glass, tints present) and the ring
    at 0, light_q, the flux and the glow are exactly 0 everywhere and the
    emission book is 0. PAIR: one burning cell (1263 game) lights the room.

    BREAKS IF: the ring returns anything but 0, the table's bucket 0 glows (a
    room-temperature body would emit), a body re-emits anything, or the
    emission reads E° instead of L°.
    """
    rng = random.Random(44)
    h, w = 11, 13
    a, d, _T, his, ts = random_scene(rng, h, w)
    T = [[rng.choice([-(200 << 16), 0, 300 << 16, 480 << 16]) for _ in range(w)]
         for _ in range(h)]
    la, ld = light_scene(rng, h, w)
    gas, hq, nb = random_gas(rng, h, w)
    lab, lgl = light_cols(rng, len(gas))
    light = dict(la=la, ld=ld, light_absorb_q=lab, light_glow_q=lgl)
    got = cpp_sweep(a, d, 0, T, his, ts, table=live_table(), gas=gas, hq=hq,
                    n_bulk=nb, light=light)
    lo = got[6]
    assert not np.any(lo["light_q"]) and not np.any(lo["light_flux"])
    assert not np.any(lo["light_glow"])
    assert lo["books"]["emit"] == (0, 0, 0)
    T[5][6] = 1263 << 16
    for c in range(3):
        la[c][5][6] = ld[c][5][6] = ONE
    lit = cpp_sweep(a, d, 0, T, his, ts, table=live_table(), gas=gas, hq=hq,
                    n_bulk=nb, light=light)
    assert np.count_nonzero(lit[6]["light_q"][0]) > 10


# ---------------------------------------------------------------------------
# 5. OPTICS
# ---------------------------------------------------------------------------
def _source_column(n, T_game=1263):
    sc, light = clear_room(n)
    for y in range(n):
        sc[2][y][0] = T_game << 16
        for c in range(3):
            light["la"][c][y][0] = light["ld"][c][y][0] = ONE
    return sc, light


def test_an_opaque_wall_leaves_the_far_side_exactly_dark():
    """PROPERTY (brief 3.6): a wall of a = ONE (every channel) spanning the grid
    between a burning column and the rest of the room leaves every cell beyond
    it at exactly 0 light on every channel, while the near side is lit. PAIR: a
    one-tile gap in the wall lets light through.

    BREAKS IF: the absorption stops taking the whole stamped share, the split
    leaks past an opaque cell, or the ring feeds the far side.
    """
    n, xw = 13, 6
    for gap in (None, n // 2):
        sc, light = _source_column(n)
        for y in range(n):
            if y != gap:
                for c in range(3):
                    light["la"][c][y][xw] = light["ld"][c][y][xw] = ONE
        lq = engine(sc, light)[6]["light_q"]
        far = lq[:, :, xw + 1:]
        if gap is None:
            assert not np.any(far), "light crossed an opaque wall"
            assert np.all(lq[0, :, 1:xw] > 0)
        else:
            assert np.any(far[0] > 0), "PAIR: the gap let no light through"


def test_glass_reads_its_light_row_for_light_and_its_heat_row_for_heat():
    """PROPERTY (brief 3.6, design §5 "glass light-clear / heat-opaque"): on the
    SHIPPED glass row (config light_atten 0.1, heat_atten 0.3, through the
    materials door) the light channel's planes depend on light_atten_q and not
    on heat_atten_q, and the heat channel's on heat_atten_q and not on
    light_atten_q: moving the OTHER channel's coefficient changes nothing,
    moving a channel's own changes it. Light behind one glass column keeps
    more of its source than heat keeps of its excess.

    BREAKS IF: the light channel reads heat_atten_q (or the heat channel
    light_atten_q), or the materials door quantizes one row into the other's
    column.
    """
    from simulation.materials import MaterialTable, MAT_GLASS
    tbl = MaterialTable.from_config()
    g_light = [int(v) for v in tbl.light_atten_q16[MAT_GLASS]]
    g_heat = int(tbl.heat_atten_q16[MAT_GLASS])
    assert g_light == [Q(0.1)] * 3 and g_heat == Q(0.3)
    n, xw = 11, 5

    def run(light_c, heat_c):
        sc, light = _source_column(n)
        a, d, T, his, ts = sc
        for y in range(n):
            a[y][0] = d[y][0] = ONE
            ts[y][0] = 1
            a[y][xw] = d[y][xw] = heat_c
            ts[y][xw] = 1
            for c in range(3):
                light["la"][c][y][xw] = light["ld"][c][y][xw] = light_c
        got = engine(sc, light)
        return got[6]["light_q"], got[3]

    lq, phi = run(g_light[0], g_heat)
    lq_h, phi_h = run(g_light[0], Q(0.9))      # heat's coefficient moves
    lq_l, phi_l = run(Q(0.9), g_heat)          # light's coefficient moves
    assert np.array_equal(lq, lq_h) and not np.array_equal(lq, lq_l)
    assert np.array_equal(phi, phi_l) and not np.array_equal(phi, phi_h)
    lq0, phi0 = run(0, 0)
    amb = 16 * ((int(np.asarray(live_table().table())[0]) * 4096) >> 16)
    yc = n // 2
    t_light = lq[0, yc, n - 1] / lq0[0, yc, n - 1]
    t_heat = (phi[yc, n - 1] - amb) / (phi0[yc, n - 1] - amb)
    assert t_light > t_heat > 0, (t_light, t_heat)


def test_a_stamped_marine_blocks_light_and_does_not_glow():
    """PROPERTY (brief 3.6): a marine stamped by GameMap.stamp_units (the MAX
    stamp, the unit's default opaque light_atten) raises dyn_light_atten_q to
    ONE over its footprint -- never below a material's -- and a burning column's
    light does not reach the cells in the shadow of a marine wall, while the
    marine itself, however hot its tile, emits no light (a body re-emits the
    dark ambient). PAIR: without the marines the same cells are lit.

    BREAKS IF: the stamp sums instead of MAXing, skips the light twin, the sweep
    reads the static plane instead of the stamped one, or a body emits.
    """
    from level_loader import LevelData
    from simulation import Simulation
    from simulation.unit import Unit
    h = w = 16
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:15, 1:15] = 4                               # hull room, air inside
    lvl = LevelData(name="light_stamp", version="1", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."))
    sim = Simulation(lvl, seed=3, breach_physics=bp, enable_recorder=False)
    g = sim.gmap
    xs = 7
    for y in (1, 4, 7, 10, 12):          # 3x3 footprints anchored top-left: rows 1..14
        sim.add_unit(Unit(f"M{y}", x=xs, y=y, team=0))
    g.stamp_units(sim.units)
    fp = np.zeros((h, w), dtype=bool)
    for u in sim.units:
        for (tx, ty) in u.occupied_tiles():
            if 0 <= ty < h and 0 <= tx < w:
                fp[ty, tx] = True
    assert np.all(fp[1:15, xs]), "the marine wall has a gap"
    assert np.all(g.dyn_light_atten_q[fp] == ONE)
    assert np.all(g.dyn_light_atten_q >= g.light_atten_q)
    assert np.array_equal(g.dyn_light_atten_q[~fp], g.light_atten_q[~fp])
    # sweep: a burning column at x = 1, the marine wall, receivers beyond it
    to_cf = lambda p: np.moveaxis(p, -1, 0).tolist()          # noqa: E731
    T = [[0] * w for _ in range(h)]
    for y in range(1, 15):
        T[y][1] = 1263 << 16
    la = to_cf(g.light_atten_q)
    for y in range(1, 15):
        for c in range(3):
            la[c][y][1] = ONE
    for y in range(h):
        for x in range(w):
            if fp[y, x]:
                T[y][x] = 3000 << 16                         # a HOT marine tile
    ld_stamped = to_cf(g.dyn_light_atten_q)
    for y in range(1, 15):
        for c in range(3):
            ld_stamped[c][y][1] = ONE
    ld_free = [[row[:] for row in p] for p in la]
    zeros = [[0] * w for _ in range(h)]
    ts = [[0] * w for _ in range(h)]
    his = [[3] * w for _ in range(h)]
    shadow = np.zeros((h, w), dtype=bool)
    shadow[1:15, int(np.max(np.nonzero(fp)[1])) + 1:15] = True   # beyond the wall
    assert shadow.sum() > 20
    blocked = engine((zeros, zeros, T, his, ts), dict(la=la, ld=ld_stamped))[6]["light_q"]
    free = engine((zeros, zeros, T, his, ts), dict(la=la, ld=ld_free))[6]["light_q"]
    assert np.all(free[0][shadow] > 0), "PAIR: the shadow cells are not lit without marines"
    assert not np.any(blocked[:, shadow]), "light passed through a wall of marines"


def test_smoke_tints_by_its_rgb_absorption_and_dims_with_density():
    """PROPERTY (brief 3.6): with the SHIPPED gas table's light columns
    (GasTable.light_absorb_q16: RGB absorption x the [smoke] dials), a column of
    POISON (absorbs B hardest, G least) between a hot source and a receiver
    raises the receiver's G/R and lowers its B/R; a column of SMOKE (soot) dims
    every channel monotonically as its density rises, strictly until the column
    is opaque and exactly 0 beyond.

    BREAKS IF: the smoke term drops a channel (all three read one coefficient:
    no tint), reads heat_absorb instead of the light column, is summed with the
    material instead of MAXed, or loses the density law.
    """
    from simulation.gases import GasTable, SMOKE, POISON, N_GASES, O2, INERT_N2
    tbl = GasTable.from_config()
    lab = tbl.light_absorb_q16.tolist()
    lgl = tbl.light_glow_q16.tolist()
    n, xw = 11, 5
    yc = n // 2

    def run(gas_id, dens):
        sc, light = _source_column(n, 3000)
        a, d, T, his, ts = sc
        for y in range(n):
            ts[y][0] = 1
        planes = [[[0] * n for _ in range(n)] for _ in range(N_GASES)]
        for y in range(n):
            planes[gas_id][y][xw] = dens
            for x in range(n):
                planes[O2][y][x] = 13763
                planes[INERT_N2][y][x] = 51773
        nb = [[planes[O2][y][x] + planes[INERT_N2][y][x] for x in range(n)] for y in range(n)]
        light.update(light_absorb_q=lab, light_glow_q=lgl)
        return engine(sc, light, gas=planes, hq=[0] * N_GASES, n_bulk=nb)[6]["light_q"][:, yc, n - 1]

    clear = run(POISON, 0).astype(np.float64)
    pois = run(POISON, Q(0.5)).astype(np.float64)
    assert pois[1] / pois[0] > clear[1] / clear[0]
    assert pois[2] / pois[0] < clear[2] / clear[0]
    curve = np.array([run(SMOKE, Q(v / 20)) for v in range(21)])
    assert np.all(np.diff(curve, axis=0) <= 0), "denser smoke let more light through"
    for c in range(3):
        zero_at = int(np.argmax(curve[:, c] == 0)) if np.any(curve[:, c] == 0) else 21
        assert zero_at < 21, "a full-density soot column never went opaque"
        assert np.all(np.diff(curve[:zero_at, c]) < 0)
        assert np.all(curve[zero_at:, c] == 0)


def test_no_gas_no_glow_and_glow_is_light_times_albedo_density():
    """PROPERTY (brief 3.6, decision 6): with no gas the glow plane is exactly 0
    everywhere (light present); with a steam column (the shipped albedo) the
    glow is exactly mul128_shr(light_q, g, 16) on the gas cells, 0 on every
    other cell and every thermal solid, and it doubles with the density at a
    fixed light (per unit light_q, within two Q16 counts).

    BREAKS IF: the glow is computed on non-gas cells, reads the absorption
    instead of the albedo column, drops the density, or is booked into the
    stream (it is render-only).
    """
    from simulation.gases import GasTable, STEAM, N_GASES, O2, INERT_N2
    tbl = GasTable.from_config()
    n, xw = 11, 5
    sc, light = _source_column(n, 3000)      # 3000 game: all three channels lit
    lo = engine(sc, light)[6]
    assert np.any(lo["light_q"]) and not np.any(lo["light_glow"])

    def run(dens):
        sc2, light2 = _source_column(n, 3000)
        for y in range(n):
            sc2[4][y][0] = 1
        planes = [[[0] * n for _ in range(n)] for _ in range(N_GASES)]
        for y in range(n):
            planes[STEAM][y][xw] = dens
            for x in range(n):
                planes[O2][y][x] = 13763
                planes[INERT_N2][y][x] = 51773
        nb = [[planes[O2][y][x] + planes[INERT_N2][y][x] for x in range(n)] for y in range(n)]
        light2.update(light_absorb_q=tbl.light_absorb_q16.tolist(),
                      light_glow_q=tbl.light_glow_q16.tolist())
        got = engine(sc2, light2, gas=planes, hq=[0] * N_GASES, n_bulk=nb)
        return got[6], np.moveaxis(np.asarray(got[5].light_gcoef_plane()), -1, 0)

    lo1, g1 = run(Q(0.25))
    lo2, g2 = run(Q(0.5))
    lq, gl = lo1["light_q"], lo1["light_glow"]
    want = (lq.astype(object) * g1.astype(object)) >> 16
    assert np.array_equal(gl, want.astype(np.int64))
    off = np.ones((n, n), dtype=bool)
    off[:, xw] = False
    assert not np.any(gl[:, off]) and np.all(gl[:, :, xw] > 0)
    r1 = lo1["light_glow"][:, :, xw] / lo1["light_q"][:, :, xw]
    r2 = lo2["light_glow"][:, :, xw] / lo2["light_q"][:, :, xw]
    assert np.all(np.abs(r2 - 2.0 * r1) <= 2.0 / ONE)


# ---------------------------------------------------------------------------
# 6. COLOUR FOLLOWS TEMPERATURE
# ---------------------------------------------------------------------------
def test_a_hot_cell_emits_blackbody_py_s_chroma_and_hotter_is_brighter():
    """PROPERTY (brief 3.7): a lone opaque cell at T emits, per ordinate and per
    channel, exactly ((L°_c[T] * w_m) >> 16) -- the emission book is n_ord x
    that -- and the chroma of what it emits equals renderer/blackbody.py's at
    that temperature (BlackbodyRamp.emission_at_kelvin at the bucket midpoint)
    within the currency's quantization: two counts per ordinate against a peak
    of >= 2^8. Hotter is brighter: over a sweep of temperatures through the
    ramp's rising range the peak channel's emission strictly rises, and it
    never falls anywhere.

    BREAKS IF: the emission bypasses the checked-in table, swaps a channel, is
    split other than by w_m, or the table stops following the colour map.
    """
    from config import CFG
    from renderer.blackbody import BlackbodyRamp
    ramp = BlackbodyRamp.from_config(CFG)
    lt = np.asarray(bp.LightEmissionTable().table(), dtype=np.int64)
    prev_peak = -1
    rising = []
    for tg in list(range(540, 4300, 180)) + [6000, 12000, R.T_TABLE_TOP_GAME]:
        sc, light = clear_room(5, sources=[((2, 2), tg)])
        for n_ord in (16, 12):
            bk = engine(sc, light, n_ord=n_ord)[6]["books"]
            b = (tg << 16) >> 18
            wm = ONE // n_ord
            per_ord = [int(lt[c, b] * wm) >> 16 for c in range(3)]
            assert bk["emit"] == tuple(n_ord * v for v in per_ord), (tg, n_ord)
        per_ord16 = np.array([int(lt[c, b] * 4096) >> 16 for c in range(3)])
        peak = int(per_ord16.max())
        if peak >= 256:
            chroma, _i = ramp.emission_at_kelvin([ramp.kelvin_ambient + 4 * b + 2])
            assert np.all(np.abs(per_ord16 - chroma[0] * peak) <= 2), (tg, per_ord16, chroma)
        assert peak >= prev_peak, "a hotter cell emitted less in its peak channel"
        if tg < 4200:
            rising.append(peak)
        prev_peak = peak
    assert all(b > a for a, b in zip(rising, rising[1:])), "flat inside the T^4 ramp"


# ---------------------------------------------------------------------------
# 7. HEADROOM AND THE RESOLUTION FLOOR
# ---------------------------------------------------------------------------
def test_headroom_holds_over_driven_and_the_floor_holds_sixteen_tiles_away():
    """PROPERTY (brief 3.8, decision 4): (a) OVER-DRIVEN -- every cell at the
    table top, light-opaque, S16 and S12: the engine equals the reference bit
    for bit (an int64 wrap in C++ would part them; the reference computes in
    unbounded ints) and light_q stays below 2^46; (b) THE FLOOR -- one burning
    tile of the shipped furniture row (1263 game) in a clear 41 x 41 room: at
    EVERY cell 16 tiles away the red light is >= 2^8 counts (the reference's
    G18 shows the brightest single ordinate there carries >= 2^21.8; light_q is
    at least that ordinate).

    BREAKS IF: the light currency grows past the headroom (the products wrap),
    or shrinks below the floor (L_FINE_BITS < 24 fails (b); the faintest-glow
    floor, tests/test_light_table.py, is the tighter one).
    """
    top = R.T_TABLE_TOP_GAME
    for n_ord in (16, 12):
        sc, light = clear_room(9)
        a, d, T, his, ts = sc
        for y in range(9):
            for x in range(9):
                T[y][x] = top << 16
                for c in range(3):
                    light["la"][c][y][x] = light["ld"][c][y][x] = ONE
        got = engine(sc, light, n_ord=n_ord)
        exp = ref_sweep(a, d, 0, T, his, n_ord=n_ord, ts=ts, table=R.E_LIVE, light=light)
        assert np.array_equal(got[6]["light_q"], exp[6]["light_q"])
        assert np.array_equal(got[6]["light_flux"], exp[6]["light_flux"])
        assert int(got[6]["light_q"].max()) < 2 ** 46
    from simulation.materials import MaterialTable, MAT_FURNITURE
    fur = [int(v) for v in MaterialTable.from_config().light_atten_q16[MAT_FURNITURE]]
    n, cy = 41, 20
    sc, light = clear_room(n, sources=[((cy, cy), 1263)])
    for c in range(3):
        light["la"][c][cy][cy] = light["ld"][c][cy][cy] = fur[c]
    lq = engine(sc, light)[6]["light_q"][0]
    ring = [(y, x) for y in range(n) for x in range(n)
            if 15.5 <= math.hypot(y - cy, x - cy) < 16.5]
    assert len(ring) > 100
    assert min(int(lq[y, x]) for (y, x) in ring) >= 2 ** 8


# ---------------------------------------------------------------------------
# 8. THE FLUX VECTOR
# ---------------------------------------------------------------------------
def test_the_flux_vector_points_away_from_a_lone_source():
    """PROPERTY (brief 3.9): around a lone burning cell in a clear room, the net
    flux Σ_m I_m ŝ_m points AWAY from it on each of its four sides -- flux_x > 0
    to its right, < 0 to its left; flux_y > 0 below it (+y is the increasing
    row), < 0 above -- and a mirrored cell books the exactly mirrored vector
    (the symmetric shift).

    BREAKS IF: the flux sums the outflow with the wrong sign convention, uses a
    cosine table out of step with the ordinates, or floors instead of the
    symmetric shift (the mirror breaks by a count).
    """
    n = 13
    c0 = n // 2
    sc, light = clear_room(n, sources=[((c0, c0), 1263)])
    f = engine(sc, light)[6]["light_flux"]
    fx, fy = f[0], f[1]
    assert np.all(fx[c0, c0 + 1:] > 0) and np.all(fx[c0, :c0] < 0)
    assert np.all(fy[c0 + 1:, c0] > 0) and np.all(fy[:c0, c0] < 0)
    assert np.array_equal(fx[:, c0 + 1:], -fx[:, :c0][:, ::-1])
    assert np.array_equal(fy[:, c0 + 1:], fy[:, :c0][:, ::-1])


# ---------------------------------------------------------------------------
# 9. THE REQUEST
# ---------------------------------------------------------------------------
def _burning_room():
    """The playground with one WOOD tile beside air heated to the 1263-game
    plateau and lit -- a fire started by delivering heat (CLAUDE.md "Starting a
    fire"); wood's light_atten is 1.0, so it is a light source through L°."""
    from level_loader import load as load_level
    from simulation import Simulation, fire_fixed
    from simulation.materials import MAT_WOOD
    sim = Simulation(load_level("playground"), seed=5, breach_physics=bp,
                     enable_recorder=False)
    g = sim.gmap
    ys, xs = np.where((g.material == MAT_WOOD) & g.thermal_solid)
    for y, x in zip(ys, xs):
        if 0 < y < g.material.shape[0] - 1 and 0 < x < g.material.shape[1] - 1 \
                and (not g.thermal_solid[y, x + 1] or not g.thermal_solid[y + 1, x]):
            g.temperature[y, x] = 1263 << 16          # a THERMAL SOLID: its T is its own
            g.fire[y, x] = fire_fixed.quantize_scalar(0.8)
            return sim
    raise AssertionError("the playground carries no wood tile beside air")


def test_light_is_computed_only_when_requested():
    """PROPERTY (brief decision 2, design §7.1): the engine's light is OFF by
    default -- a full Simulation.step leaves the light planes exactly as it
    found them (a sentinel survives) and computes the same heat as ever -- and
    ON when PhysicsEngine.light_requested is set: the sweep then OVERWRITES the
    light planes (the sentinel is gone, the hull is lit by the fire) while every
    heat-side GameMap array stays identical to the unrequested run, tick after
    tick. Requesting light without handing the planes is refused loudly.

    BREAKS IF: light is computed by default (headless training would pay for
    it and the golden would see its planes), the request is ignored, requesting
    light moves any heat integer, or step_tail silently computes nothing when
    the planes are missing.
    """
    s_off, s_on = _burning_room(), _burning_room()
    assert s_off.physics_runner.engine.light_requested is False
    SENT = -12345
    for s in (s_off, s_on):
        s.gmap.light_q[...] = SENT
    s_on.physics_runner.engine.light_requested = True
    heat_fields = ("temperature", "gas_energy", "rad_net_sweep", "rad_flux_sweep",
                   "rad_amb_sweep", "rad_fluence", "fire", "gas", "atmosphere")
    for _t in range(6):
        for s in (s_off, s_on):
            s.set_paused(False)
            s.step()
        for k in heat_fields:
            assert np.array_equal(getattr(s_off.gmap, k), getattr(s_on.gmap, k)), k
    assert np.all(s_off.gmap.light_q == SENT), "light was computed without a request"
    assert not np.any(s_on.gmap.light_q == SENT)
    assert np.any(s_on.gmap.light_q > 0), "a requested light run lit nothing"
    eng = bp.PhysicsEngine()
    eng.light_requested = True
    g = s_on.gmap
    with pytest.raises(ValueError):
        eng.step_tail(
            g.ripple, g.ripple_v, g.water_depth, g.wave_p, g.solid, g.fire.copy(),
            g.atmosphere.copy(), g.smoke.copy(), g.wall_hp.copy(), g.temperature.copy(),
            g.wind_x, g.wind_y, g.is_vacuum, g.flammable, g.heat, g.heat_inv_shift,
            g.face_shift, g.thermal_solid, g.fuel_recip, g.fire_T_ext_plane,
            g.gas.copy(), g.gases.conservative, 5, 0.0,
            heat_atten_q=g.heat_atten_q, dyn_heat_atten_q=g.dyn_heat_atten_q,
            rad_net_sweep=g.rad_net_sweep.copy(), rad_flux_sweep=g.rad_flux_sweep.copy(),
            rad_amb_sweep=g.rad_amb_sweep.copy(), rad_fluence=g.rad_fluence.copy(),
            gas_heat_absorb_q16=g.gases.heat_absorb_q16, k_leak_q=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
