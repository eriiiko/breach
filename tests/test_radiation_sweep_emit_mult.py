"""THE FIRE'S LIGHT FLICKER on the radiation sweep, CPU (graphics G3 F4b).

docs/g3_fire_design_2026-10-23.md section 6.3 (RULED D): a per-cell Q16
EMISSION MULTIPLIER row (y, x, mult_q) scales that cell's THERMAL light
emission, every ordinate and channel alike, and nothing else. The integer
reference (sweep_ref_q.py's LightGroup.emit_mult / validate_emit_mult) is the
spec; this file holds the engine to it bit for bit and pins the properties the
design rests on: light moves no heat, ONE is the identity, no rows is the
pre-F4b sweep, the books close, lamps (cones) are never scaled, the door
refuses illegal rows, and the headroom holds at the door's maximum. The CUDA
twin is held to this CPU sweep at tol 0 by tests/test_cuda_radiation_sweep_
emit_mult.py.

Every test's docstring names its property and the change that breaks it.
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
from test_radiation_sweep_cones_sky import (  # noqa: E402
    light_scene, random_cones, random_sky)

Q = R.quant
K_LEAK = Q(0.10)
MAX = R.LIGHT_EMIT_MULT_MAX


def hot_scene(rng, h, w):
    """random_scene with a few cells raised into the visible range (a burning
    row), so the light channels carry thermal emission to scale."""
    a, d, T, his, ts = random_scene(rng, h, w)
    hot = [(rng.randrange(h), rng.randrange(w)) for _ in range(5)]
    for (y, x) in hot:
        T[y][x] = rng.choice([1100, 1263, 1500, 2200]) << 16
    return (a, d, T, his, ts), hot


def random_mults(rng, cells, extra, h, w):
    """One row per distinct cell: the hot cells plus some cold ones, factors
    across the door's range including both ends and ONE."""
    seen, rows = set(), []
    pool = list(cells) + [(rng.randrange(h), rng.randrange(w)) for _ in range(extra)]
    vals = [0, ONE, MAX, Q(0.7), Q(1.3), Q(0.85), Q(1.15)]
    for i, (y, x) in enumerate(pool):
        if (y, x) in seen:
            continue
        seen.add((y, x))
        rows.append((y, x, vals[i % len(vals)] if i < len(vals)
                     else rng.randrange(MAX + 1)))
    return rows


# ---------------------------------------------------------------------------
# 1. GATE 0 FOR THE MULTIPLIER
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("ltr", ["step", "shear"])
@pytest.mark.parametrize("smoke", [False, True])
def test_cpp_emit_mult_reproduces_the_reference_bit_for_bit(seed, n_ord, ltr, smoke):
    """PROPERTY (gate 0 for the flicker): with emission-multiplier rows (every
    factor kind -- 0, ONE, the 2 ONE maximum, fractions -- on hot and cold cells)
    riding the light group beside cones and a sky, the C++ sweep's light_q,
    light_flux_q, light_glow, its twelve light books and its heat planes EQUAL
    the reference's, integer for integer, at S16/S12 x light step/shear x smoke
    on/off. Non-vacuous: a scaled hot cell's light differs from the unscaled run.

    BREAKS IF: the C++ scales the emission before the a_c product instead of
    after it (a different floor), scales the cone injection or the absorbed
    share, books the unscaled emission, or reads a row's cell transposed.
    """
    rng = random.Random(20261023 + seed * 11 + n_ord)
    h, w = rng.choice([(7, 9), (9, 11), (12, 8)])
    (a, d, T, his, ts), hot = hot_scene(rng, h, w)
    la, ld = light_scene(rng, h, w)
    for (y, x) in hot:                         # emitters that actually emit
        for c in range(3):
            la[c][y][x] = ld[c][y][x] = Q(0.8)
    rows = random_mults(rng, hot, 4, h, w)
    light = dict(la=la, ld=ld, transport=ltr, cones=random_cones(rng, h, w, 3),
                 sky=random_sky(rng, n_ord, "random"), emit_mult=rows)
    kw = {}
    if smoke:
        gas, hq, nb = random_gas(rng, h, w)
        lab = [[rng.choice([0, Q(0.14), Q(1.26)]) for _ in range(3)] for _ in range(len(gas))]
        lgl = [[rng.choice([0, Q(0.04), Q(0.92)]) for _ in range(3)] for _ in range(len(gas))]
        light.update(light_absorb_q=lab, light_glow_q=lgl)
        kw = dict(gas=gas, hq=hq, n_bulk=nb)
    got = cpp_sweep(a, d, K_LEAK, T, his, ts, n_ord=n_ord, table=live_table(),
                    light=light, **kw)
    exp = ref_sweep(a, d, K_LEAK, T, his, n_ord=n_ord, ts=ts, table=R.E_LIVE,
                    light=light, **kw)
    for k in range(5):
        assert np.array_equal(np.asarray(got[k], dtype=np.int64), exp[k]), k
    gl, el = got[6], exp[6]
    for k in ("light_q", "light_flux", "light_glow"):
        assert np.array_equal(gl[k], el[k]), k
    assert dict(gl["books"]) == el["books"]
    plain = dict(light)
    plain.pop("emit_mult")
    base = cpp_sweep(a, d, K_LEAK, T, his, ts, n_ord=n_ord, table=live_table(),
                     light=plain, **kw)[6]
    assert not np.array_equal(base["light_q"], gl["light_q"]), "the rows moved nothing"


# ---------------------------------------------------------------------------
# 2. THE INVARIANTS F4b RESTS ON
# ---------------------------------------------------------------------------
def test_heat_is_bit_identical_with_and_without_the_flicker():
    """PROPERTY (design 6.3 "no heat"): the four heat planes, the Fleck plane
    and the heat stream telemetry are identical with the emission multipliers
    on (every factor kind, on the hot cells) and off, at both heat transports.

    BREAKS IF: the multiplier reaches the heat emission (the E-table source) or
    the Fleck pre-pass, i.e. the flicker would move the sim's heat -- and so
    the digest and the golden.
    """
    rng = random.Random(7)
    for htr in ("shear", "step"):
        h, w = 9, 11
        (a, d, T, his, ts), hot = hot_scene(rng, h, w)
        la, ld = light_scene(rng, h, w)
        light = dict(la=la, ld=ld)
        off = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, table=live_table(),
                        light=light)
        on = cpp_sweep(a, d, K_LEAK, T, his, ts, transport=htr, table=live_table(),
                       light=dict(light, emit_mult=random_mults(rng, hot, 3, h, w)))
        for k in range(5):
            assert np.array_equal(np.asarray(off[k]), np.asarray(on[k])), k
        assert (off[5].min_stream, off[5].max_stream) == (on[5].min_stream,
                                                          on[5].max_stream)


def test_one_is_the_identity_and_no_rows_is_the_pre_f4b_sweep():
    """PROPERTY ("disabled -> light_q bit-identical to before F4b"): a row of
    exactly ONE on every cell of the grid gives the SAME light_q, flux, glow and
    books as no rows at all -- and no rows is the code path that existed before
    F4b (nullptr, nothing scaled).

    BREAKS IF: the scaling rounds (ONE must be exact: (e * 2^16) >> 16 == e for
    e >= 0), or the multiplier plane defaults to anything but ONE on a cell
    without a row.
    """
    rng = random.Random(11)
    h, w = 9, 11
    (a, d, T, his, ts), _hot = hot_scene(rng, h, w)
    la, ld = light_scene(rng, h, w)
    light = dict(la=la, ld=ld, cones=random_cones(rng, h, w, 3))
    none = cpp_sweep(a, d, K_LEAK, T, his, ts, table=live_table(), light=light)[6]
    ones = cpp_sweep(a, d, K_LEAK, T, his, ts, table=live_table(),
                     light=dict(light, emit_mult=[(y, x, ONE) for y in range(h)
                                                  for x in range(w)]))[6]
    for k in ("light_q", "light_flux", "light_glow"):
        assert np.array_equal(none[k], ones[k]), k
    assert dict(none["books"]) == dict(ones["books"])


def test_the_flicker_scales_the_emission_and_the_books_close():
    """PROPERTY (isotropic, emission-only): on one burning cell in a clear
    room, a factor m scales that cell's emission book by m (within the floor of
    one count per ordinate and channel), the light it sends EVERY way scales
    alike (the irradiance one tile away on all four sides moves by the same
    ratio), and the books close exactly -- emit + ring_in == absorb + ring_out.

    BREAKS IF: the factor is applied per ordinate unevenly (directional, D2 --
    out of scope), applied twice, or left out of the emission book.
    """
    n, c0 = 11, 5
    a = [[0] * n for _ in range(n)]
    T = [[0] * n for _ in range(n)]
    his = [[3] * n for _ in range(n)]
    ts = [[0] * n for _ in range(n)]
    T[c0][c0] = 1263 << 16
    la = [[[0] * n for _ in range(n)] for _ in range(3)]
    for c in range(3):
        la[c][c0][c0] = Q(0.8)
    ld = [[r[:] for r in p] for p in la]
    base = cpp_sweep(a, [r[:] for r in a], 0, T, his, ts, table=live_table(),
                     light=dict(la=la, ld=ld))[6]
    for m in (Q(0.6), Q(1.4), MAX, 0):
        got = cpp_sweep(a, [r[:] for r in a], 0, T, his, ts, table=live_table(),
                        light=dict(la=la, ld=ld, emit_mult=[(c0, c0, m)]))[6]
        b = dict(got["books"])
        for c in range(3):
            e0 = dict(base["books"])["emit"][c]
            assert abs(b["emit"][c] - (e0 * m >> 16)) <= 16, (m, c)
            assert b["emit"][c] + b["ring_in"][c] == b["absorb"][c] + b["ring_out"][c]
        nb = [(c0, c0 + 1), (c0, c0 - 1), (c0 + 1, c0), (c0 - 1, c0)]
        ratios = [int(got["light_q"][0][y][x]) / int(base["light_q"][0][y][x]) for (y, x) in nb]
        assert max(ratios) - min(ratios) < 1e-6, ratios
        assert abs(ratios[0] - m / ONE) < 1e-3, (m, ratios)


def test_a_lamp_in_a_flickering_cell_is_not_scaled():
    """PROPERTY ("lamps untouched"): a cone emitter sitting in a cell whose
    thermal emission is multiplied keeps its injection -- on a COLD cell (no
    thermal emission at all) any factor leaves the whole light field exactly
    as without it.

    BREAKS IF: the multiplier scales the cone injection (a lamp next to a fire
    would flicker -- Erik's option B, not D).
    """
    n, c0 = 9, 4
    a = [[0] * n for _ in range(n)]
    T = [[0] * n for _ in range(n)]
    his = [[3] * n for _ in range(n)]
    ts = [[0] * n for _ in range(n)]
    la = [[[0] * n for _ in range(n)] for _ in range(3)]
    ld = [[[0] * n for _ in range(n)] for _ in range(3)]
    cones = [(c0, c0, (1 << 38, 1 << 38, 1 << 38), 0, R.CONE_TURN)]
    base = cpp_sweep(a, [r[:] for r in a], 0, T, his, ts, table=live_table(),
                     light=dict(la=la, ld=ld, cones=cones))[6]
    assert int(base["light_q"].sum()) > 0
    for m in (0, Q(0.5), MAX):
        got = cpp_sweep(a, [r[:] for r in a], 0, T, his, ts, table=live_table(),
                        light=dict(la=la, ld=ld, cones=cones,
                                   emit_mult=[(c0, c0, m)]))[6]
        assert np.array_equal(base["light_q"], got["light_q"]), m
        assert dict(base["books"]) == dict(got["books"]), m


def test_the_emit_mult_door_refuses_illegal_rows():
    """PROPERTY (the door, both homes): a row off the grid, a negative factor,
    one above LIGHT_EMIT_MULT_MAX, or two rows for one cell is refused by name
    (ValueError) before any output is touched -- by the engine AND by the
    reference -- and the bound is the same number in both homes and the
    binding.

    BREAKS IF: the door drops a clause (an above-max factor would breach the
    int64 headroom argument; a duplicate would make "which row wins" depend on
    the renderer's emission order) or the homes drift apart.
    """
    assert int(bp.RadiationSweep.LIGHT_EMIT_MULT_MAX) == R.LIGHT_EMIT_MULT_MAX == 2 * ONE
    rng = random.Random(5)
    h, w = 6, 7
    (a, d, T, his, ts), _hot = hot_scene(rng, h, w)
    la, ld = light_scene(rng, h, w)
    for bad in ([(h, 0, ONE)], [(0, -1, ONE)], [(0, 0, -1)], [(0, 0, MAX + 1)],
                [(1, 1, ONE), (1, 1, Q(0.5))]):
        sentinel = None
        with pytest.raises(ValueError):
            sentinel = cpp_sweep(a, d, K_LEAK, T, his, ts, table=live_table(),
                                 light=dict(la=la, ld=ld, emit_mult=bad))
        assert sentinel is None
        with pytest.raises(ValueError):
            R.validate_emit_mult(bad, h, w)


def test_headroom_holds_at_the_door_maximum_over_driven():
    """PROPERTY (the int64 argument): every cell at the table top, light-opaque,
    every cell's emission at the door's maximum 2 ONE -- the engine still equals
    the reference bit for bit (an int64 wrap in C++ would part them; the
    reference computes in unbounded ints) and light_q stays below 2^47 (one bit
    above P6a's 2^46 over-driven bound).

    BREAKS IF: LIGHT_EMIT_MULT_MAX grows past the headroom, or the scaling is
    formed in a way that overflows before the shift.
    """
    top = R.T_TABLE_TOP_GAME
    n = 9
    a = [[0] * n for _ in range(n)]
    T = [[top << 16] * n for _ in range(n)]
    his = [[3] * n for _ in range(n)]
    ts = [[0] * n for _ in range(n)]
    la = [[[ONE] * n for _ in range(n)] for _ in range(3)]
    light = dict(la=la, ld=[[r[:] for r in p] for p in la],
                 emit_mult=[(y, x, MAX) for y in range(n) for x in range(n)])
    for n_ord in (16, 12):
        got = cpp_sweep(a, [r[:] for r in a], 0, T, his, ts, n_ord=n_ord,
                        table=live_table(), light=light)
        exp = ref_sweep(a, [r[:] for r in a], 0, T, his, n_ord=n_ord, ts=ts,
                        table=R.E_LIVE, light=light)
        assert np.array_equal(got[6]["light_q"], exp[6]["light_q"])
        assert np.array_equal(got[6]["light_flux"], exp[6]["light_flux"])
        assert int(got[6]["light_q"].max()) < 2 ** 47
