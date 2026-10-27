"""CONE EMITTERS and THE SKY BOUNDARY on the radiation sweep, CPU (ray-engine-v2 P6b).

docs/ray_engine_v2_p6b_lit_world_brief_2026-09-25.md sections 1.3-1.4 and 3.2-3.3;
design v3 sections 2.7, 4.1, 7.2. The integer reference (sweep_ref_q.py's
cone_overlaps / cone_emission / LightGroup.cones / .sky, gate 19) is the spec;
this file holds the ENGINE to it -- gate 0 for the cones and the sky -- and pins
the brief's properties on the engine's own output. The CUDA twin is held to this
CPU sweep at tol 0 by tests/test_cuda_radiation_sweep_cones.py.

Every test's docstring names its property and the change that breaks it.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_cones_sky.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "tests") not in sys.path:
    sys.path.insert(0, str(ROOT / "tests"))

from _radiation_sweep_harness import (  # noqa: E402
    ONE, R, bp, cpp_sweep, live_table, random_gas, random_scene, ref_sweep)

Q = R.quant
K_LEAK = Q(0.10)
TURN = R.CONE_TURN


def light_scene(rng, h, w):
    """Random light extinction, channel-first: clear air, glass, a crate,
    opaque, two tints; bodies (d = ONE) on some cells."""
    tints = [(0, 0, 0), (0, 0, 0), (0, 0, 0), (Q(0.1),) * 3, (Q(0.55),) * 3, (ONE,) * 3,
             (Q(0.2), Q(0.6), Q(0.9)), (Q(0.9), Q(0.1), Q(0.4))]
    la = [[[0] * w for _ in range(h)] for _ in range(3)]
    ld = [[[0] * w for _ in range(h)] for _ in range(3)]
    for y in range(h):
        for x in range(w):
            t = rng.choice(tints)
            body = rng.random() < 0.1
            for c in range(3):
                la[c][y][x] = t[c]
                ld[c][y][x] = ONE if body else t[c]
    return la, ld


def random_cones(rng, h, w, n):
    """n cones: every beam kind -- zero spread, narrow, wide, one straddling
    angle 0, omni -- two of them on one cell (the per-cell sum)."""
    beams = [(rng.randrange(TURN), 0), (rng.randrange(TURN), rng.randrange(1, 3000)),
             (rng.randrange(TURN), rng.randrange(3000, 40000)), (65000, 3000),
             (rng.randrange(TURN), TURN), (0, 10923)]
    out = []
    for i in range(n):
        cq, sq = beams[i % len(beams)]
        y, x = (rng.randrange(h), rng.randrange(w)) if i != 1 else (out[0][0], out[0][1])
        out.append((y, x, tuple(rng.randrange(1 << 39) for _ in range(3)), cq, sq))
    return out


def random_sky(rng, n_ord, kind):
    if kind == "overcast":
        return [[5 << 30, 5 << 30, 6 << 30]] * n_ord
    if kind == "sun":
        return R.cone_emission([3 << 34] * 3, rng.randrange(TURN), 4000, n_ord)
    return [[rng.randrange(1 << 36) for _ in range(3)] for _ in range(n_ord)]


# ---------------------------------------------------------------------------
# 1. THE PROJECTION: one implementation, the engine's, equal to the spec
# ---------------------------------------------------------------------------
def test_the_engine_projection_is_the_reference_one():
    """PROPERTY (brief 1.3, gate 19 (a) on the engine): RadiationSweep.
    cone_emission -- the projection the sweep, its CUDA twin and the Python sky
    door all read -- equals sweep_ref_q.cone_emission integer for integer over
    every beam kind (zero spread, narrow, wide, straddling angle 0, omni, an odd
    spread) at S16 and S12.

    BREAKS IF: the C++ bin edge floors instead of ceiling, the unwrap drops a
    bin's second copy, the zero-spread or omni branch picks a different bin or
    width, or the division rounds differently.
    """
    rng = random.Random(29)
    beams = [(0, 0), (4095, 0), (65535, 0), (0, 10923), (65000, 3000), (123, TURN),
             (32768, 1), (40000, 65535), (1000, 21846)]
    beams += [(rng.randrange(TURN), rng.randrange(TURN + 1)) for _ in range(40)]
    for n_ord in (16, 12):
        for (cq, sq) in beams:
            rgb = [rng.randrange(1 << 44) for _ in range(3)]
            got = np.asarray(bp.RadiationSweep.cone_emission(rgb, cq, sq, n_ord))
            exp = np.asarray(R.cone_emission(rgb, cq, sq, n_ord), dtype=np.int64)
            assert np.array_equal(got, exp), (n_ord, cq, sq)
    assert int(bp.RadiationSweep.CONE_TURN) == R.CONE_TURN
    assert int(bp.RadiationSweep.LIGHT_CONE_BUDGET) == R.LIGHT_CONE_BUDGET
    assert int(bp.RadiationSweep.LIGHT_SKY_MAX) == R.LIGHT_SKY_MAX


# ---------------------------------------------------------------------------
# 2. GATE 0 FOR CONES AND SKY
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("htr", ["shear", "step"])
@pytest.mark.parametrize("ltr", ["step", "shear"])
@pytest.mark.parametrize("smoke", [False, True])
@pytest.mark.parametrize("sky_kind", ["overcast", "sun", "random", None])
def test_cpp_cones_and_sky_reproduce_the_reference_bit_for_bit(seed, n_ord, htr, ltr,
                                                               smoke, sky_kind):
    """PROPERTY (brief 3.2 / 3.3, gate 0 for the cones and the sky): with cone
    emitters (every beam kind, two sharing a cell) and a sky (overcast, a sun,
    random, none) riding the light group, the C++ sweep's light_q, light_flux_q,
    light_glow, its twelve light books and its light telemetry EQUAL the
    reference's, integer for integer -- and its heat planes too -- over S16/S12 x
    heat shear/step x light step/shear x smoke on/off, on the live table.
    Non-vacuous: the ring book is non-zero when there is a sky, and the emitter
    cells are lit.

    BREAKS IF: the C++ injects a cone after the cell's absorption, reads the sky
    for only one of the two upwind shares, books the injection outside `emit`,
    sums two cones in a cell differently, or indexes a slot's ordinate wrong.
    """
    rng = random.Random(20260929 + seed * 7 + n_ord)
    h, w = rng.choice([(7, 9), (9, 11), (12, 8)])
    a, d, T, his, ts = random_scene(rng, h, w)
    la, ld = light_scene(rng, h, w)
    cones = random_cones(rng, h, w, 6)
    sky = random_sky(rng, n_ord, sky_kind) if sky_kind else None
    light = dict(la=la, ld=ld, transport=ltr, cones=cones, sky=sky)
    kw = {}
    if smoke:
        gas, hq, nb = random_gas(rng, h, w)
        lab = [[rng.choice([0, Q(0.14), Q(1.26)]) for _ in range(3)] for _ in range(len(gas))]
        lgl = [[rng.choice([0, Q(0.04), Q(0.92)]) for _ in range(3)] for _ in range(len(gas))]
        light.update(light_absorb_q=lab, light_glow_q=lgl)
        kw = dict(gas=gas, hq=hq, n_bulk=nb)
    got = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, n_ord=n_ord,
                    table=live_table(), light=light, **kw)
    exp = ref_sweep(a, d, K_LEAK, T, his, transport=htr, n_ord=n_ord, ts=ts,
                    table=R.E_LIVE, light=light, **kw)
    for name, g, e in zip(("rad_net", "rad_flux", "rad_amb", "rad_fluence", "fleck"),
                          got[:5], exp[:5]):
        assert np.array_equal(np.asarray(g, dtype=np.int64), e), name
    gl, el = got[6], exp[6]
    for k in ("light_q", "light_flux", "light_glow"):
        assert np.array_equal(gl[k], el[k]), k
    gb, eb = dict(gl["books"]), el["books"]
    assert gb == eb
    for c in range(3):
        assert gb["emit"][c] + gb["ring_in"][c] == gb["absorb"][c] + gb["ring_out"][c]
    if sky_kind in ("overcast", "random"):
        assert all(v > 0 for v in gb["ring_in"])
    assert any(gl["light_q"][0][y][x] > 0 for (y, x, *_r) in cones)


def test_heat_is_bit_identical_with_cones_and_sky():
    """PROPERTY (brief 3.5, in the sweep): the four heat planes, the Fleck plane
    and the heat telemetry are identical with the light group -- cones and sky
    included -- on and off, on randomised scenes at both heat transports.

    BREAKS IF: the cone injection, the sky or the per-cell slot lookup writes a
    heat integer or reorders a heat visit.
    """
    rng = random.Random(3)
    for htr in ("shear", "step"):
        h, w = 9, 11
        a, d, T, his, ts = random_scene(rng, h, w)
        la, ld = light_scene(rng, h, w)
        off = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, table=live_table())
        on = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, table=live_table(),
                       light=dict(la=la, ld=ld, cones=random_cones(rng, h, w, 5),
                                  sky=random_sky(rng, 16, "random")))
        for k in range(5):
            assert np.array_equal(np.asarray(off[k]), np.asarray(on[k])), k
        assert (off[5].min_stream, off[5].max_stream) == (on[5].min_stream, on[5].max_stream)


# ---------------------------------------------------------------------------
# 3. THE BRIEF'S PROPERTIES ON THE ENGINE
# ---------------------------------------------------------------------------
def _room(n, *, opaque=(), body=()):
    a = [[0] * n for _ in range(n)]
    T = [[0] * n for _ in range(n)]
    his = [[3] * n for _ in range(n)]
    ts = [[0] * n for _ in range(n)]
    la = [[[0] * n for _ in range(n)] for _ in range(3)]
    ld = [[[0] * n for _ in range(n)] for _ in range(3)]
    for (y, x) in opaque:
        for c in range(3):
            la[c][y][x] = ld[c][y][x] = ONE
    for (y, x) in body:
        for c in range(3):
            ld[c][y][x] = ONE
    return (a, [r[:] for r in a], T, his, ts), la, ld


def _lit(sc, la, ld, **light):
    a, d, T, his, ts = sc
    return cpp_sweep(a, d, 0, T, his, ts, table=live_table(),
                     light=dict(la=la, ld=ld, **light))[6]


def test_a_cone_lights_only_inside_its_beam():
    """PROPERTY (brief 3.2): a cone lights only inside its beam. In a clear room
    a narrow beam pointing +x (east) lights the cells east of the emitter and
    leaves every cell WEST of it exactly dark; rotated to point west it does the
    mirror; a zero-spread cone lights a strictly thinner fan than a 60-degree
    one. On a BODY cell the same cone lights its own cell and nothing else.

    BREAKS IF: the projection leaks power into ordinates outside the beam (the
    west half lights up), the beam's angle convention flips (east and west
    swap), a zero-spread cone becomes omni, or the injection bypasses the
    emitter cell's own absorption (the body leaks light).
    """
    n = 15
    c0 = (7, 7)
    sc, la, ld = _room(n)
    rgb = (1 << 37,) * 3
    east = _lit(sc, la, ld, cones=[(c0[0], c0[1], rgb, 0, TURN // 6)])["light_q"][0]
    west = _lit(sc, la, ld, cones=[(c0[0], c0[1], rgb, TURN // 2, TURN // 6)])["light_q"][0]
    assert np.all(east[:, :c0[1]] == 0) and np.all(east[c0[0], c0[1] + 1:] > 0)
    assert np.all(west[:, c0[1] + 1:] == 0) and np.all(west[c0[0], :c0[1]] > 0)
    assert np.array_equal(east, west[:, ::-1])          # the mirror, exactly
    thin = _lit(sc, la, ld, cones=[(c0[0], c0[1], rgb, 0, 0)])["light_q"][0]
    assert 0 < np.count_nonzero(thin) < np.count_nonzero(east)
    sc_b, la_b, ld_b = _room(n, body=[c0])
    blocked = _lit(sc_b, la_b, ld_b, cones=[(c0[0], c0[1], rgb, 0, TURN // 6)])["light_q"]
    mask = np.ones((n, n), dtype=bool)
    mask[c0] = False
    assert np.all(blocked[:, mask] == 0) and np.all(blocked[:, c0[0], c0[1]] > 0)


def test_a_sealed_room_is_exactly_dark_and_an_opening_lets_the_sky_in():
    """PROPERTY (brief 3.3): a sealed room (a ring of light-opaque walls) under a
    uniform sky has light_q EXACTLY 0 inside while the outside reads the sky,
    so the eye there sees only the floor dial; open one wall cell and the sky
    comes in, decaying with distance from the opening (the step transport).
    A uniform sky carries exactly n_ord * S into every clear cell.

    BREAKS IF: the ring stops reading the sky, the sky leaks through an opaque
    wall (a transport or absorption bug), or the opening's light does not decay
    (a transport that stopped spreading).
    """
    n = 17
    lo, hi = 3, 13
    wall = [(y, x) for y in range(lo, hi + 1) for x in range(lo, hi + 1)
            if y in (lo, hi) or x in (lo, hi)]
    S = 3 << 32
    sky = [[S, S, S]] * 16
    sc, la, ld = _room(n, opaque=wall)
    q = _lit(sc, la, ld, sky=sky)["light_q"]
    inside = q[:, lo + 1:hi, lo + 1:hi]
    assert np.all(inside == 0)
    outside = np.ones((n, n), dtype=bool)
    outside[lo:hi + 1, lo:hi + 1] = False
    assert np.all(q[:, outside] > 0)                  # the box shadows, never blacks out
    ymid = (lo + hi) // 2
    sc_o, la_o, ld_o = _room(n, opaque=[p for p in wall if p != (ymid, lo)])
    qo = _lit(sc_o, la_o, ld_o, sky=sky)["light_q"][0]
    axis = qo[ymid, lo + 1:hi]
    assert np.all(axis > 0) and np.all(np.diff(axis) < 0)
    sc_c, la_c, ld_c = _room(n)
    assert np.all(_lit(sc_c, la_c, ld_c, sky=sky)["light_q"] == 16 * S)


def test_the_cone_and_sky_door_refuses_illegal_input():
    """PROPERTY (the int64 argument's door, radiation_sweep.h): the C++ sweep
    refuses -- before touching any output -- a cone outside the grid, a
    negative rgb, a centre outside [0, TURN), a spread above TURN, a summed rgb
    one count over LIGHT_CONE_BUDGET and a sky value outside [0, LIGHT_SKY_MAX];
    it accepts exactly the budget and exactly the maximum.

    BREAKS IF: a check is dropped or off by one (a budget + 1 accepted would
    let a stream past the int64 argument), or a refusal writes the planes first.
    """
    n = 5
    sc, la, ld = _room(n)
    B = R.LIGHT_CONE_BUDGET
    bad = [dict(cones=[(n, 0, (1, 1, 1), 0, 0)]), dict(cones=[(0, 0, (-1, 0, 0), 0, 0)]),
           dict(cones=[(0, 0, (1, 1, 1), TURN, 0)]), dict(cones=[(0, 0, (1, 1, 1), 0, TURN + 1)]),
           dict(cones=[(0, 0, (B, 0, 0), 0, 0), (1, 1, (1, 0, 0), 0, 0)]),
           dict(sky=[[R.LIGHT_SKY_MAX + 1, 0, 0]] + [[0] * 3] * 15),
           dict(sky=[[-1, 0, 0]] + [[0] * 3] * 15)]
    a, d, T, his, ts = sc
    for kw in bad:
        with pytest.raises(ValueError):
            cpp_sweep(a, d, 0, T, his, ts, table=live_table(), light=dict(la=la, ld=ld, **kw))
    ok = _lit(sc, la, ld, cones=[(0, 0, (B, B, B), 0, 0)], sky=[[R.LIGHT_SKY_MAX] * 3] * 16)
    assert ok["light_q"].max() > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
