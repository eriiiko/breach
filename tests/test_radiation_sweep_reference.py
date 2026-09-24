"""GATE 0 — THE REFERENCE IS THE SPEC (ray-engine-v2 P1, design v3 §2.9 gate 0).

The C++ sweep (cpp/src/radiation_sweep.cpp) must reproduce the integer
reference `docs/ray_engine_v2_scheme_study_2026-09-13/sweep_ref_q.py` BIT FOR
BIT — all four planes AND the Fleck pre-pass — on randomised grids, both
transport steps, leak on and off, bodies present, S16 and S12, temperatures
from sub-ambient to above the table top. Any change to the arithmetic is made
in the reference first, its gates re-run, and the C++ written against it —
never the other way round (CLAUDE.md "Integer reference").

P5a (design v3 §6.3) adds the SMOKE axis: gas planes, a per-gas heat_absorb
table and the bulk count, compared bit for bit INCLUDING the two effective
extinction planes the sweep read — over gate 0's own random scenes, on one scene
carrying every feature of the term, and at the engine's door (the rejections).
Validated by breaking the C++ once: shifting each gas term before summing (the
design sums, then shifts once) turned the smoke matrix red.

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
    F_ONE, ONE, R, cpp_sweep, random_gas, random_scene, ref_sweep, reference_table,
    smoke_feature_scene, ts_from_a)

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


_PLANES = ("rad_net", "rad_flux", "rad_amb", "rad_fluence", "fleck")


def _assert_bit_for_bit(tag, got, exp, *, effective=False):
    """The five planes (and, with the smoke term, the two EFFECTIVE extinction
    planes the sweep read) equal integer for integer, plus the telemetry."""
    for name, g, e in zip(_PLANES, got[:5], exp[:5]):
        g64 = np.asarray(g, dtype=np.int64)
        assert g64.shape == e.shape
        if not np.array_equal(g64, e):
            bad = np.argwhere(g64 != e)
            y, x = bad[0]
            pytest.fail(f"{tag}: {name} differs at {len(bad)} cells; first ({y},{x}) "
                        f"cpp={g64[y, x]} ref={e[y, x]}")
    if effective:
        for name, g, e in (("a_eff", got[5].a_eff_plane(), exp[5].a_eff),
                           ("d_eff", got[5].d_eff_plane(), exp[5].d_eff)):
            g64 = np.asarray(g, dtype=np.int64)
            e64 = np.asarray(e, dtype=np.int64)
            if not np.array_equal(g64, e64):
                bad = np.argwhere(g64 != e64)
                y, x = bad[0]
                pytest.fail(f"{tag}: {name} differs at {len(bad)} cells; first "
                            f"({y},{x}) cpp={g64[y, x]} ref={e64[y, x]}")
    assert got[5].min_stream == exp[5].min_stream
    assert got[5].max_stream == exp[5].max_stream


@pytest.mark.parametrize("seed,h,w", [(11, 7, 9), (12, 9, 11), (13, 12, 8), (14, 5, 5)])
@pytest.mark.parametrize("n_ord", [16, 12])
@pytest.mark.parametrize("k_q", [0, K_LEAK])
@pytest.mark.parametrize("transport", ["shear", "step"])
@pytest.mark.parametrize("ambient", ("uniform", "random"))
def test_cpp_sweep_reproduces_the_reference_with_smoke_bit_for_bit(seed, h, w, n_ord,
                                                                   k_q, transport, ambient):
    """PROPERTY (P5a, design v3 §6.3): with the SMOKE TERM live -- gas planes,
    a per-gas heat_absorb table, the bulk count -- the C++ sweep's four planes,
    its Fleck plane AND the two effective extinction planes it read (a with the
    smoke term, d = max(d, a)) EQUAL the reference's, integer for integer, over
    gate 0's own randomised scenes (thermal solids, bodies, hot and sub-ambient
    cells), S16 and S12, both transports, leak on and off, a non-uniform
    ambient. Non-vacuous per case: some GAS cell absorbs through the smoke term
    and the smoke moves rad_net against the same scene without it.

    BREAKS IF: radiation_sweep.cpp transcribes the density law differently
    (a second density factor, a per-gas shift instead of one after the sum),
    the N_EPS floor keys on anything but the bulk count or at anything but
    N_EPS_RAW, a thermal solid takes the smoke term, the stamped total is summed
    instead of MAXed, a negative density absorbs, the zero-coefficient
    compaction drops a live gas, or the loop reads the input planes instead of
    the effective ones.
    """
    rng = random.Random(20260924 + seed)
    a, d, T, his, ts = random_scene(rng, h, w)
    amb = _ambient(ambient, rng, h, w)
    gas, hq, n_bulk = random_gas(rng, h, w)
    tag = f"{transport} S{n_ord} k={k_q} seed={seed} amb={ambient} SMOKE"
    got = cpp_sweep(a, d, k_q, T, his, ts, transport=transport, n_ord=n_ord, amb=amb,
                    gas=gas, hq=hq, n_bulk=n_bulk)
    exp = ref_sweep(a, d, k_q, T, his, transport=transport, n_ord=n_ord, amb=amb,
                    ts=ts, gas=gas, hq=hq, n_bulk=n_bulk)
    _assert_bit_for_bit(tag, got, exp, effective=True)
    a_eff = np.asarray(exp[5].a_eff)
    ts_a = np.asarray(ts) != 0
    assert np.any((a_eff > np.asarray(a)) & ~ts_a), "no gas cell absorbed: vacuous"
    clear = ref_sweep(a, d, k_q, T, his, transport=transport, n_ord=n_ord, amb=amb, ts=ts)
    assert not np.array_equal(clear[0], exp[0]), "the smoke moved nothing: vacuous"


@pytest.mark.parametrize("transport", ["shear", "step"])
@pytest.mark.parametrize("k_q", [0, K_LEAK])
def test_cpp_sweep_reproduces_the_reference_on_every_feature_of_the_smoke_term(transport,
                                                                              k_q):
    """PROPERTY (P5a): on one scene that carries EVERY feature of the smoke term
    at once -- a thin absorber, a saturated one, hot smoke, smoke below and
    exactly at the N_EPS floor, bodies standing in thin and opaque smoke, a
    thermal solid full of gas, a negative density, a zero-coefficient gas with
    density everywhere -- the C++ sweep equals the reference bit for bit, AND
    each feature does what the design says (read off the reference's effective
    planes, so the comparison is not vacuous about any of them).

    BREAKS IF: any single feature is transcribed differently in C++; or the
    reference itself stops meaning what design 6.3 says (the feature asserts).
    """
    a, d, T, ts, gas, hq, n_bulk, cells = smoke_feature_scene()
    got = cpp_sweep(a, d, k_q, T, 3, ts, transport=transport, gas=gas, hq=hq,
                    n_bulk=n_bulk)
    exp = ref_sweep(a, d, k_q, T, 3, transport=transport, ts=ts, gas=gas, hq=hq,
                    n_bulk=n_bulk)
    _assert_bit_for_bit(f"{transport} k={k_q} FEATURES", got, exp, effective=True)
    ae, de, rn = exp[5].a_eff, exp[5].d_eff, exp[0]
    Q = R.quant
    at = lambda key: cells[key]                                          # noqa: E731
    assert 0 < ae[at("thin")[0]][at("thin")[1]] < ONE                   # thin: the law
    assert ae[at("thin")[0]][at("thin")[1]] == (Q(5.0) * Q(0.06)) >> 16
    assert ae[at("saturated")[0]][at("saturated")[1]] == ONE            # the ONE cap
    assert rn[at("hot")] < 0                                            # hot smoke radiates
    assert ae[at("floored")[0]][at("floored")[1]] == 0 and rn[at("floored")] == 0
    assert ae[at("at_floor")[0]][at("at_floor")[1]] == ONE              # N_EPS_RAW absorbs
    y, x = at("body_thin")
    assert de[y][x] == ONE and 0 < ae[y][x] < ONE and exp[1][y, x] != 0
    y, x = at("body_opaque")
    assert de[y][x] == ONE and ae[y][x] == ONE and exp[1][y, x] == 0   # a MAX, not a sum
    assert ae[5][6] == Q(0.5)                                           # ts ignores its gas
    assert ae[at("negative")[0]][at("negative")[1]] == 0                # absorbs nothing


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


def test_illegal_gas_inputs_are_rejected_at_the_engine_door():
    """PROPERTY (P5a, gate 3's rejections for the smoke term): the C++ sweep
    refuses exactly what the reference's validate_gas refuses -- a heat_absorb
    coefficient below 0 or above 2^28 (the gas door's [0, 4096]), more gas
    planes than the headroom argument covers (16), a partial gas group -- with a
    ValueError, and a gas plane of the wrong dtype with a TypeError (checked,
    never converted), leaving the caller's output planes untouched. The legal
    edges (0 and 2^28, 16 planes) are accepted.

    BREAKS IF: any of the checks is dropped from radiation_sweep.cpp or from the
    binding's gas_group_args, or a rejected scene touches the planes.
    """
    import breach_physics as bp
    h = w = 4
    a = R.plane(h, w, 0)
    d = R.plane(h, w, 0)
    T = R.plane(h, w, 0)
    ts = R.plane(h, w, 0)
    nb = R.plane(h, w, ONE)
    g2 = [R.plane(h, w, ONE), R.plane(h, w, ONE)]
    cpp_sweep(a, d, 0, T, 3, ts, gas=g2, hq=[0, R.HEAT_ABSORB_Q_MAX], n_bulk=nb)  # legal
    sixteen = [R.plane(h, w, ONE)] * R.N_GAS_PLANES_MAX
    cpp_sweep(a, d, 0, T, 3, ts, gas=sixteen, hq=[ONE] * R.N_GAS_PLANES_MAX, n_bulk=nb)
    for name, gas, hq in (("heat_absorb < 0", g2, [-1, 0]),
                          ("heat_absorb > 2^28", g2, [R.HEAT_ABSORB_Q_MAX + 1, 0]),
                          ("17 gas planes", sixteen + [R.plane(h, w, 0)],
                           [0] * (R.N_GAS_PLANES_MAX + 1))):
        with pytest.raises(ValueError):
            cpp_sweep(a, d, 0, T, 3, ts, gas=gas, hq=hq, n_bulk=nb)
    # the binding's own door: a partial group, a table/plane mismatch, a dtype
    sweep = bp.RadiationSweep()
    tbl = reference_table()
    z32 = np.zeros((h, w), dtype=np.int32)
    outs = [np.full((h, w), 777, dtype=np.int64) for _ in range(4)]
    base = (z32, z32, z32, np.full((h, w), 3, np.int32), np.zeros((h, w), dtype=bool),
            tbl, None, 293 << 16, 0, bp.RadiationSweep.SHEAR, 16, *outs)
    gas_a = np.ones((2, h, w), dtype=np.int32)
    hq_a = np.zeros(2, dtype=np.int32)
    nb_a = np.full((h, w), ONE, dtype=np.int32)
    with pytest.raises(ValueError):
        sweep.run(*base, gas=gas_a, heat_absorb_q16=hq_a)             # no n_bulk
    with pytest.raises(ValueError):
        sweep.run(*base, gas=gas_a, heat_absorb_q16=np.zeros(3, np.int32), n_bulk=nb_a)
    with pytest.raises(TypeError):
        sweep.run(*base, gas=gas_a.astype(np.int64), heat_absorb_q16=hq_a, n_bulk=nb_a)
    assert all(np.all(p == 777) for p in outs), "a rejected scene touched the planes"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
