"""The emissive table E°(T), its owner and its inverse (ray-engine-v2 P1,
design v3 §2.6).

The bake body moved from raycaster.cpp into the strict TU emissive_table.cpp
and now has TWO callers — Raycaster::bake_emissive_table (the old march, until
P3) and PhysicsEngine.emissive (the sweep's and the clamp's owner). One
implementation, two owners: these tests assert the two tables are identical
entry for entry and equal to the integer reference's own bake, and that the
new inverse E°⁻¹ is the reference's e_inv_q -- and, since P5d, that the
clamp's ceiling e_ceiling_q is the reference's too.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_emissive_table.py -q
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp  # noqa: E402
from config import CFG  # noqa: E402
import temperature_scale  # noqa: E402

_STUDY = ROOT / "docs" / "ray_engine_v2_scheme_study_2026-09-13"
if str(_STUDY) not in sys.path:
    sys.path.insert(0, str(_STUDY))
import sweep_ref_q as R  # noqa: E402


def _dials():
    """The three dials exactly as PhysicsRunner assigns them.

    T6 (issue #12): rebased on `[physics.radiation] rad_scale_derived` --
    the sweep's emission calibration is now the only one there is.
    `[physics.fire] rad_scale` (the old cast's FITTED key) is deleted with
    the cast; this test's own property (the E° tables are identical to each
    other and to the reference bake) survives unchanged at the new key.
    """
    ts = temperature_scale.load(CFG)
    return (float(getattr(CFG.physics.radiation, "rad_scale_derived", 1.0e-5)),
            float(ts.kelvin_ambient), float(ts.k_temp_to_kelvin))


def _engine_with_dials():
    eng = bp.PhysicsEngine()
    scale, amb, slope = _dials()
    eng.raycaster.rad_scale = scale
    eng.raycaster.kelvin_ambient = amb
    eng.raycaster.k_temp_to_kelvin = slope
    eng.raycaster.bake_emissive_table()
    eng.emissive.rad_scale = scale
    eng.emissive.kelvin_ambient = amb
    eng.emissive.k_temp_to_kelvin = slope
    eng.emissive.bake()
    return eng


def test_raycaster_and_engine_tables_are_identical_and_equal_the_reference_bake():
    """PROPERTY: Raycaster's E° table == PhysicsEngine.emissive's table, entry
    for entry, == sweep_ref_q.bake_e_table() at the config dials.

    T6 (issue #12): `ref` now bakes at `_dials()`'s OWN scale explicitly,
    rather than at `R.bake_e_table()`'s no-arg default (`R.RAD_SCALE`, a
    fixed "resolving" literal unrelated to any config key -- report_m3.md
    §6 -- which only equalled `_dials()`'s old `[physics.fire] rad_scale`
    read by historical coincidence: both were 5.1427e-5). Rebasing `_dials()`
    onto `[physics.radiation] rad_scale_derived` broke that coincidence; this
    keeps the actual property (two owners agree with EACH OTHER and with the
    reference, at whatever scale config configures) rather than pinning to
    the reference's unrelated default.

    BREAKS IF: a second bake appears (one owner drifts), the bake stops being
    the exact int64 K⁴ chain with one boundary multiply, or config.toml's dials
    move under the reference (the reference test guards that separately).
    """
    ok, detail = R.config_dials_match()
    assert ok, f"config.toml dials drifted under the reference: {detail}"
    eng = _engine_with_dials()
    scale, _amb, _slope = _dials()
    ray = np.asarray(eng.raycaster.emissive_table(), dtype=np.int64)
    own = np.asarray(eng.emissive.table(), dtype=np.int64)
    # #78: the engine's table is baked in its FINE currency (E_FINE_BITS, the
    # reference's FINE_BITS); the vestigial Raycaster copy follows the same bake,
    # so the identity is at equal dials AND equal currency.
    assert int(eng.emissive.fine_bits) == bp.E_FINE_BITS == R.FINE_BITS
    ref = np.asarray(R.bake_e_table(rad_scale=scale, fine_bits=int(eng.emissive.fine_bits)),
                     dtype=np.int64)
    assert ray.shape == own.shape == ref.shape == (bp.E_TABLE_SIZE,)
    assert np.array_equal(ray, own), "Raycaster and engine tables differ"
    assert np.array_equal(own, ref), "engine table differs from the reference bake"
    # non-vacuity: the table is monotone and spans 8 orders of magnitude
    assert np.all(np.diff(own) > 0) and own[0] > 0 and own[-1] > own[0] * 10 ** 6


# T6 (issue #12) deletes test_live_runner_bakes_each_owner_at_its_own_
# configured_scale. Property it protected: during the P2b-P3c transition
# window the two E table owners were DELIBERATELY bound to two DIFFERENT
# config keys (the old cast's fitted [physics.fire] rad_scale, the sweep's
# derived [physics.radiation] rad_scale_derived), so the live game and every
# golden stayed put while the sweep ran shadow physics. That property no
# longer exists to protect: the old cast and its config key are deleted, so
# PhysicsRunner now binds ONE key to both owners and the "each owner its own
# key, and they must differ" assertions this test made would themselves be
# wrong going forward. The surviving property -- one bake implementation,
# two owners producing IDENTICAL tables -- is
# test_raycaster_and_engine_tables_are_identical_and_equal_the_reference_bake
# above, now exercised at the one remaining key via the rebased _dials().


def test_lazy_rebake_on_a_dial_change():
    """PROPERTY: table() re-bakes when rad_scale moves (the Raycaster contract,
    kept), and the re-baked table equals the reference at the new scale -- and
    (#78) when the CURRENCY moves: a table whose fine_bits changed is re-baked in
    the new currency, never served from the old cache.

    BREAKS IF: the cache key drops a dial or the currency. Validated (#78): the
    cache key without the currency serves the fine table to a coarse request.
    """
    eng = _engine_with_dials()
    fb = int(eng.emissive.fine_bits)
    before = np.asarray(eng.emissive.table(), dtype=np.int64)
    eng.emissive.rad_scale = eng.emissive.rad_scale * 2.0
    after = np.asarray(eng.emissive.table(), dtype=np.int64)
    assert not np.array_equal(before, after)
    ref = np.asarray(R.bake_e_table(rad_scale=eng.emissive.rad_scale, fine_bits=fb),
                     dtype=np.int64)
    assert np.array_equal(after, ref)
    eng.emissive.fine_bits = 0
    coarse = np.asarray(eng.emissive.table(), dtype=np.int64)
    assert np.array_equal(coarse, np.asarray(R.bake_e_table(rad_scale=eng.emissive.rad_scale),
                                             dtype=np.int64))
    assert not np.array_equal(coarse, after)


def test_bake_precondition_rejects_a_non_integer_kelvin_dial():
    """PROPERTY: the exact-int64 bake refuses a fractional Kelvin dial (RuntimeError)
    instead of silently baking a table that CPU and CUDA would disagree on.

    BREAKS IF: the integrality check is dropped from the moved bake.
    """
    tbl = bp.EmissiveTable()
    tbl.rad_scale = R.RAD_SCALE
    tbl.kelvin_ambient = 293.5
    tbl.k_temp_to_kelvin = 1.0
    with pytest.raises(RuntimeError):
        tbl.bake()


def test_e_bucket_of_matches_the_reference_including_sub_ambient_and_saturation():
    """PROPERTY: e_bucket_of(T_q) == the reference's for T below 0 (bucket 0),
    across the table and above its top (bucket 3999).

    BREAKS IF: the lookup moves off the 18-bit shift or loses either clamp.
    """
    tbl = bp.EmissiveTable()
    rng = random.Random(3)
    probes = [-(300 << 16), -1, 0, 1, (4 << 16) - 1, 4 << 16, 15996 << 16,
              (16000 << 16) - 1, 16000 << 16, 2 ** 31 - 1]
    probes += [rng.randint(-(2 ** 31), 2 ** 31 - 1) for _ in range(2000)]
    for T_q in probes:
        assert tbl.e_bucket_of(T_q) == R.e_bucket_of(T_q), T_q


def test_e_inv_q_is_the_reference_inverse_with_both_edge_cases():
    """PROPERTY: e_inv_q(Φ) == sweep_ref_q.e_inv_q(Φ) on boundary probes and
    random Φ over the table's range; Φ < E°[0] -> 0; Φ >= E°[3999] -> 15996 game
    (below T_MAX_PHYS); idempotent (E°[e_bucket_of(E°⁻¹(Φ))] <= Φ); exact at
    every bucket edge.

    BREAKS IF: the 12-trip lifting returns a bucket's high edge, stops covering
    4000 buckets, or the sub-E°[0] case is left undefined.
    """
    tbl = bp.EmissiveTable()
    tbl.rad_scale = R.RAD_SCALE
    tbl.kelvin_ambient = float(R.K_AMB)
    tbl.k_temp_to_kelvin = float(R.K_SLOPE)
    tbl.fine_bits = R.fine_bits_of(R.E)       # #78: the resolving table is coarse
    tbl.bake()
    E = R.E
    assert tbl.e_inv_q(E[0] - 1) == 0
    assert tbl.e_inv_q(0) == 0
    assert tbl.e_inv_q(E[3999] * 3) == (bp.E_INV_TOP_GAME << 16) == (15996 << 16)
    assert bp.E_INV_TOP_GAME < 16000
    rng = random.Random(11)
    probes = [E[0], E[0] + 1, E[17] - 1, E[17], E[2000] + 12345, E[3999] - 1, E[3999]]
    probes += [rng.randint(0, E[3999] * 2) for _ in range(3000)]
    for phi in probes:
        got = tbl.e_inv_q(phi)
        assert got == R.e_inv_q(phi), phi
        assert E[R.e_bucket_of(got)] <= phi or phi < E[0]
    for b in range(0, R.E_TABLE_SIZE, 7):
        assert tbl.e_inv_q(E[b]) == (4 * b) << 16


def _baked(scale, fine_bits):
    tbl = bp.EmissiveTable()
    tbl.rad_scale = scale
    tbl.kelvin_ambient = float(R.K_AMB)
    tbl.k_temp_to_kelvin = float(R.K_SLOPE)
    tbl.fine_bits = fine_bits
    tbl.bake()
    return tbl


@pytest.mark.parametrize("scale_name", ["resolving", "live"])
def test_e_ceiling_q_is_the_top_of_the_first_bucket_out_emitting_phi(scale_name):
    """PROPERTY (P5d, Erik's ruling of 2026-09-24 -- the Pass-1 clamp's ceiling):
    over EVERY bucket of the table (each bucket's two edges and its midpoint in
    Phi), the engine's EmissiveTable.e_ceiling_q(Phi) equals the reference's
    sweep_ref_q.e_ceiling_q(Phi), and its value c is the LAST Q16 value of its
    bucket, whose E° is the first to exceed Phi: E°[bucket(c) - 1] <= Phi <
    E°[bucket(c)]. It is >= e_inv_q(Phi) (the radiation temperature, unchanged)
    and monotone in Phi; Phi < E°[0] -> 0 (design row 31); Phi >= E°[3999] ->
    E_CEILING_TOP_Q, one LSB below 16000 game, below T_MAX_PHYS. Both the
    resolving table (the gates') and the live one (the game's).

    BREAKS IF: the clamp's ceiling is reverted to e_inv_q (its bucket is then
    Phi's own, whose E° does NOT exceed Phi -- every probe fails), moves to any
    other edge of the staircase, loses the sub-E°[0] case, or saturates at or
    above T_MAX_PHYS; or the binding stops calling the one FP_HD definition the
    folds call (it drifts from the reference).
    """
    # #78: each table in its own currency (the resolving one coarse, the live
    # one fine) -- the ceiling compares Phi and E° in one unit, so it is the
    # same function in either
    E = R.E if scale_name == "resolving" else R.E_LIVE
    scale = R.RAD_SCALE if scale_name == "resolving" else R.RAD_SCALE_LIVE
    tbl = _baked(scale, R.fine_bits_of(E))
    assert np.array_equal(np.asarray(tbl.table(), dtype=np.int64), np.asarray(E))
    last, n = -1, 0
    for b in range(R.E_TABLE_SIZE - 1):
        for phi in sorted({E[b], E[b] + (E[b + 1] - E[b]) // 2, E[b + 1] - 1}):
            c = tbl.e_ceiling_q(phi)
            assert c == R.e_ceiling_q(phi, E), (b, phi)
            bc = c >> R.E_INDEX_SHIFT
            assert ((c + 1) >> R.E_INDEX_SHIFT) == bc + 1, (b, phi, c)   # its bucket's top
            assert E[bc - 1] <= phi < E[bc], (b, phi, bc)                # first above Phi
            assert c >= tbl.e_inv_q(phi)
            assert c >= last
            last = c
            n += 1
    assert tbl.e_ceiling_q(0) == 0 and tbl.e_ceiling_q(E[0] - 1) == 0
    for phi in (E[3999], E[3999] + 1, 3 * E[3999]):
        assert tbl.e_ceiling_q(phi) == bp.E_CEILING_TOP_Q == (16000 << 16) - 1
    assert bp.E_CEILING_TOP_Q < 16000 << 16
    assert n >= R.E_TABLE_SIZE - 1                # every bucket probed at least once


def test_the_fine_currency_is_the_same_bake_at_a_power_of_two_scale():
    """PROPERTY (#78): a table in the currency k (2^k integers per heat count)
    is THE SAME BAKE at the scale rad_scale * 2^k -- a power-of-two multiply,
    exact in the bake's own arithmetic, so the one rounding boundary is
    unchanged -- entry for entry equal to the reference's bake_e_table(..., k),
    at every currency the engine admits, and the engine's own table is in
    E_FINE_BITS. It is a genuinely FINER table, not the coarse one shifted: its
    entries are not all multiples of 2^k.

    BREAKS IF: the currency is applied AFTER the bake's rounding (a shifted
    coarse table: every entry a multiple of 2^k, and it differs from the bake
    at rad_scale * 2^k), a second rounding enters the scale, or the engine's
    default currency drifts from the one constant. Validated: the bake with the
    scale left unscaled and each rounded entry shifted left by k -- red at k = 1.
    """
    assert int(bp.EmissiveTable().fine_bits) == bp.E_FINE_BITS == R.FINE_BITS
    for k in (0, 1, 5, bp.E_FINE_BITS):
        tbl = _baked(R.RAD_SCALE_LIVE, k)
        got = np.asarray(tbl.table(), dtype=np.int64)
        same = _baked(R.RAD_SCALE_LIVE * float(1 << k), 0)
        ref = np.asarray(R.bake_e_table(rad_scale=R.RAD_SCALE_LIVE, fine_bits=k),
                         dtype=np.int64)
        assert np.array_equal(got, np.asarray(same.table(), dtype=np.int64)), k
        assert np.array_equal(got, ref), k
        if k > 0:
            assert np.any(got % (1 << k) != 0), "a shifted coarse table, not a finer bake"
    live = np.asarray(_baked(R.RAD_SCALE_LIVE, bp.E_FINE_BITS).table(), dtype=np.int64)
    assert np.array_equal(live, np.asarray(R.E_LIVE, dtype=np.int64))


def test_the_bake_refuses_a_currency_outside_its_headroom():
    """PROPERTY (#78): the bake admits exactly the currencies the sweep's int64
    headroom argument covers, [0, E_FINE_BITS] -- coarse and the engine's own --
    and refuses anything else with a ValueError (the reference refuses outside
    its measuring range too), so a table the arithmetic was never proven for
    cannot be baked silently.

    BREAKS IF: the bake's currency door is dropped or widened. Validated: the
    door replaced by `if (false)` -- the bake at k = -1 and k = 12 did not raise.
    """
    for k in (0, bp.E_FINE_BITS):
        _baked(R.RAD_SCALE_LIVE, k)                          # the legal edges
    for k in (-1, bp.E_FINE_BITS + 1):
        tbl = bp.EmissiveTable()
        tbl.rad_scale = R.RAD_SCALE_LIVE
        tbl.fine_bits = k
        with pytest.raises(ValueError):
            tbl.bake()
    with pytest.raises(ValueError):
        R.bake_e_table(rad_scale=R.RAD_SCALE_LIVE, fine_bits=R.FINE_BITS_MAX + 1)


def test_the_bake_refuses_a_table_outside_the_sweeps_int64_headroom():
    """PROPERTY (#78): a currency makes a table 2^k larger, so the bake is the
    door where a scale x currency pair outside the sweep's int64 headroom is
    refused: a table whose top entry reaches E_TABLE_TOP_MAX (2^44 -- every plain
    int64 product of the sweep, at most 2^16 x the top, stays below 2^60) raises
    a ValueError and LEAVES THE CACHE AS IT WAS. The shipped tables are inside
    with room: the fine live table (2^41.1) and the coarse resolving one
    (2^41.7); the resolving scale baked FINE (2^52.7) is refused.

    BREAKS IF: the door is dropped (the resolving scale bakes fine and a sweep
    on it silently wraps int64), moved below a shipped table, or a refused bake
    leaves its entries in the cache under the previous dials. Validated: with the
    door commented out the fine resolving bake goes through and this is red.
    """
    assert bp.E_TABLE_TOP_MAX == 1 << 44
    for scale, k in ((R.RAD_SCALE_LIVE, bp.E_FINE_BITS), (R.RAD_SCALE, 0)):
        top = int(np.asarray(_baked(scale, k).table(), dtype=np.int64)[-1])
        assert top < bp.E_TABLE_TOP_MAX >> 2, (scale, k, top)   # >= 2^2 of room
    tbl = _baked(R.RAD_SCALE_LIVE, bp.E_FINE_BITS)
    good = np.asarray(tbl.table(), dtype=np.int64)
    tbl.rad_scale = R.RAD_SCALE                               # the resolving scale, fine
    with pytest.raises(ValueError):
        tbl.table()
    tbl.rad_scale = R.RAD_SCALE_LIVE                          # back: the cache is intact
    assert np.array_equal(np.asarray(tbl.table(), dtype=np.int64), good)
    tbl.rad_scale = R.RAD_SCALE
    tbl.fine_bits = 0                                         # coarse: inside the door
    assert np.array_equal(np.asarray(tbl.table(), dtype=np.int64), np.asarray(R.E))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
