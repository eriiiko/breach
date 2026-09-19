"""GATE 0 — THE REFERENCE IS THE SPEC (ray-engine-v2 P1, design v3 §2.9 gate 0).

The C++ sweep (cpp/src/radiation_sweep.cpp) must reproduce the integer
reference `docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py` BIT FOR
BIT — all four planes AND the Fleck pre-pass — on randomised grids, both
transport steps, leak on and off, bodies present, S16 and S12, temperatures
from sub-ambient to above the table top. Any change to the arithmetic is made
in the reference first, its gates re-run, and the C++ written against it —
never the other way round (CLAUDE.md "Integer reference").

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_radiation_sweep_reference.py -q
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
    F_ONE, ONE, R, cpp_sweep, random_scene, ref_sweep, reference_table, ts_from_a)

Q = R.quant
K_LEAK = Q(0.10)

# The AMBIENT axis (thermal model v2 R3). `None` is the door's uniform E°[0] —
# the only ambient the scalar era could express. The other two are non-uniform,
# and they are what makes gate 0 able to catch a PARTIAL hoist: one site left
# reading a global while the others went per-cell moves no integer on a uniform
# ambient, so without these the bit-for-bit gate would be blind to it.
AMBIENTS = ("uniform", "cold-half", "random")


def _ambient(kind, rng, h, w):
    if kind == "uniform":
        return None
    e0 = R.E0
    if kind == "cold-half":
        return [[0 if x < w // 2 else e0 for x in range(w)] for _ in range(h)]
    return [[rng.choice([0, 1, e0 // 4, e0 // 2, e0 - 1, e0]) for _ in range(w)]
            for _ in range(h)]


@pytest.mark.parametrize("seed,h,w", [(1, 7, 9), (2, 9, 11), (3, 12, 8), (4, 5, 5)])
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("k_q", [0, K_LEAK])
@pytest.mark.parametrize("transport", ["shear", "step"])
@pytest.mark.parametrize("ambient", AMBIENTS)
def test_cpp_sweep_reproduces_the_reference_bit_for_bit(seed, h, w, n_ord, k_q,
                                                        transport, ambient):
    """PROPERTY: rad_net, rad_flux, rad_amb, rad_fluence and the Fleck plane from
    the C++ sweep are EQUAL, integer for integer, to sweep_ref_q on the same
    scene — and the scene is non-trivial (every plane is non-zero somewhere,
    some Fleck factor is below 2^24, some body share is present).

    BREAKS IF: any term of design §2.3 is transcribed differently in C++ — the
    remainder split, the gather offsets, the ring's two books, the body's
    ambient re-emission, the Q24 Fleck product, the pre-pass's L_q chain, the
    per-ordinate constants, or the traversal order failing to be topological.
    With the ambient axis, ALSO: any one of the four ambient-derived sites (the
    cell's emission floor, its ceiling return `ret`, its body's re-emission, the
    virtual ring) reading a different cell's ambient than the reference does.
    """
    rng = random.Random(20260916 + seed)
    a, d, T, his, ts = random_scene(rng, h, w)
    amb = _ambient(ambient, rng, h, w)
    got = cpp_sweep(a, d, k_q, T, his, ts, transport=transport, n_ord=n_ord, amb=amb)
    exp = ref_sweep(a, d, k_q, T, his, transport=transport, n_ord=n_ord, amb=amb)
    names = ("rad_net", "rad_flux", "rad_amb", "rad_fluence", "fleck")
    for name, g, e in zip(names, got[:5], exp[:5]):
        g64 = np.asarray(g, dtype=np.int64)
        assert g64.shape == e.shape
        if not np.array_equal(g64, e):
            bad = np.argwhere(g64 != e)
            y, x = bad[0]
            pytest.fail(f"{transport} S{n_ord} k={k_q} seed={seed} amb={ambient}: "
                        f"{name} differs at {len(bad)} cells; first ({y},{x}) "
                        f"cpp={g64[y, x]} ref={e[y, x]}")
    rn, rf, ra, rl, fl = (np.asarray(p, dtype=np.int64) for p in got[:5])
    assert np.any(rn != 0) and np.any(rf != 0) and np.any(ra != 0) and np.any(rl != 0)
    assert np.any(fl < F_ONE) and np.any(fl == F_ONE)
    assert any(d[y][x] > a[y][x] for y in range(h) for x in range(w))
    # the telemetry agrees with the reference's own measurement
    assert got[5].min_stream == exp[5].min_stream
    assert got[5].max_stream == exp[5].max_stream


def test_fleck_prepass_matches_the_reference_cell_by_cell_on_the_shipped_rows():
    """PROPERTY: the pre-pass value at every bucket of the table, for every
    shipped (a, his) pair, equals sweep_ref_q.fleck_f_solid_q — the same
    excess-form Q24 factor, not merely the same sweep totals.

    BREAKS IF: L_q's shr_round0 / shift order changes, the denominator moves
    off the absolute temperature, or the division stops being the kit floor.
    """
    tbl = reference_table()
    rows = R.shipped_absorbing_rows()
    assert rows, "config.toml ships no absorbing material?"
    t_amb_q = R.K_AMB << 16
    for _name, a_q, his, _atten, _tm in rows:
        for b in range(0, R.E_TABLE_SIZE, 13):
            T_q = (4 * b) << 16
            got = bp_fleck(tbl, T_q, a_q, his, t_amb_q)
            exp, _L = R.fleck_f_solid_q(T_q, a_q, his)
            assert got == exp, (_name, b, got, exp)
    # sub-ambient and the top of the int32 field
    for T_q in (-(292 << 16), -(1 << 16), (32767 << 16) + 65535):
        assert bp_fleck(tbl, T_q, ONE, 3, t_amb_q) == R.fleck_f_solid_q(T_q, ONE, 3)[0]


def bp_fleck(tbl, T_q, a_q, his, t_amb_q):
    import breach_physics as bp
    return bp.RadiationSweep.fleck_f_solid_q24(tbl, int(T_q), int(a_q), int(his), int(t_amb_q))


def test_illegal_scenes_are_rejected_at_the_engine_door_too():
    """PROPERTY (gate 3's rejections at the C++ door): the sweep refuses a > d,
    d > ONE, k outside [0, ONE] and an unsupported ordinate count — the same
    scenes the reference's validate_planes raises on — with a Python exception,
    never a silently measured illegal scene.

    BREAKS IF: the pre-pass check is dropped.
    """
    import breach_physics as bp
    h = w = 3
    a = R.plane(h, w, Q(0.5))
    d = R.plane(h, w, ONE)
    T = R.plane(h, w, 0)
    his = R.plane(h, w, 3)
    ts = ts_from_a(a)
    cpp_sweep(a, d, 0, T, his, ts)                                   # legal: no raise
    with pytest.raises(ValueError):
        cpp_sweep(R.plane(h, w, Q(0.9)), R.plane(h, w, Q(0.5)), 0, T, his, ts)   # a > d
    with pytest.raises(ValueError):
        cpp_sweep(a, R.plane(h, w, ONE + 1), 0, T, his, ts)         # d > ONE
    with pytest.raises(ValueError):
        cpp_sweep(a, d, ONE + 1, T, his, ts)                        # k > ONE
    with pytest.raises(ValueError):
        cpp_sweep(a, d, -1, T, his, ts)                             # k < 0
    with pytest.raises(ValueError):
        cpp_sweep(a, d, 0, T, his, ts, n_ord=8)                     # not S16/S12
    # the ambient invariant (thermal v2 R3), at the same door
    cpp_sweep(a, d, 0, T, his, ts, amb=R.E0)                        # legal: no raise
    cpp_sweep(a, d, 0, T, his, ts, amb=0)                           # legal: cold sky
    with pytest.raises(ValueError):
        cpp_sweep(a, d, 0, T, his, ts, amb=R.E0 + 1)                # hotter than ambient
    with pytest.raises(ValueError):
        cpp_sweep(a, d, 0, T, his, ts, amb=-1)                      # negative level
    # a stale int32 output plane is a TypeError, not a silently discarded copy
    sweep = bp.RadiationSweep()
    tbl = reference_table()
    z32 = np.zeros((h, w), dtype=np.int32)
    z64 = np.zeros((h, w), dtype=np.int64)
    with pytest.raises(TypeError):
        sweep.run(z32, z32, np.full((h, w), ONE, dtype=np.int32), np.full((h, w), 3, np.int32),
                  np.ones((h, w), dtype=bool), tbl, None, 293 << 16, 0,
                  bp.RadiationSweep.SHEAR, 16, z32, z64, z64, z64)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
