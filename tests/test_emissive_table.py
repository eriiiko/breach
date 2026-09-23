"""The emissive table E°(T), its owner and its inverse (ray-engine-v2 P1,
design v3 §2.6).

The bake body moved from raycaster.cpp into the strict TU emissive_table.cpp
and now has TWO callers — Raycaster::bake_emissive_table (the old march, until
P3) and PhysicsEngine.emissive (the sweep's and the clamp's owner). One
implementation, two owners: these tests assert the two tables are identical
entry for entry and equal to the integer reference's own bake, and that the
new inverse E°⁻¹ is the reference's e_inv_q.

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
    ref = np.asarray(R.bake_e_table(rad_scale=scale), dtype=np.int64)
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
    kept), and the re-baked table equals the reference at the new scale.

    BREAKS IF: the cache key drops a dial.
    """
    eng = _engine_with_dials()
    before = np.asarray(eng.emissive.table(), dtype=np.int64)
    eng.emissive.rad_scale = eng.emissive.rad_scale * 2.0
    after = np.asarray(eng.emissive.table(), dtype=np.int64)
    assert not np.array_equal(before, after)
    ref = np.asarray(R.bake_e_table(rad_scale=eng.emissive.rad_scale), dtype=np.int64)
    assert np.array_equal(after, ref)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
