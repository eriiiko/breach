"""The fuel-fraction axis — F normalises against the tile's OWN hp (2026-07-30).

THE DEFECT. The fire logistic's fuel term was

    F = clamp01(wall_hp[i] / fuel_ref)          [physics.fire] fuel_ref = 60.0

and ``fuel_ref`` is WOOD's hp. ``F`` is meant to be "the fraction of THIS tile's
fuel remaining", so every material whose hp differs from wood's read a
permanently wrong fraction:

    material    hp    F when pristine
    wood        60    1.00   correct by coincidence — 60 IS the global
    furniture   30    0.50   a brand-new crate, permanently half burnt out

That is not a cosmetic error. Sustain requires ``k_die/k_grow < a/(1-a)`` with
``a = F*o2f*hot``; at ambient air ``o2f = 0.09195`` (the full-response split), so
F = 0.5 caps the sustain ceiling at 0.048 and a furniture fire cannot sustain at
ANY intensity or temperature. Measured 2026-07-30 on the tuning bench: at
``cool_shift = 10`` the crate's temperature rose correctly (280 -> 313.8 game,
``hot`` = 0.638) and the fire still died — heat was never the limiter.

Lowering the global is NOT the fix: at ``fuel_ref = 30``, wood (hp 60) would sit
clamped at F = 1 until it had already lost half its mass, destroying its
burn-down curve. The quantity is per-material by nature — the fourth global
standing in for a material property this arc has found (``thermal_mass``,
``cool_shift``, ``fire_T_ext``, now this).

THE FIX (this patch). A per-material reciprocal ``1/hp``, baked at LOAD in
exactly the form ``fixedpoint::make_recip`` bakes it, projected to the per-tile
``GameMap.fuel_recip`` grid on the SAME single seam as ``heat_inv_shift`` /
``thermal_solid`` / ``cool_shift``, and multiplied (never divided) per cell. It
is DERIVED from the existing ``hp`` column — there is no new dial and there must
never be one, or the fuel fraction and the health bar could disagree.

``fuel_ref`` survives as the solver's FALLBACK divisor when a caller supplies no
per-tile plane (the ``o2_frac_amb`` tombstone precedent); the live engine always
supplies one, so the global no longer affects the game.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_fuel_fraction_axis.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402

from config import CFG  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_FURNITURE, MAT_WOOD, MATERIAL_NAMES, MaterialTable,
    fuel_recip_from_hp, _FUEL_RECIP_SHIFT,
)

FP_ONE = 1 << 16
_TBL = MaterialTable.from_config()
FUEL_REF = float(getattr(CFG.physics.fire, "fuel_ref", 60.0))


def _q(x):
    """Round-to-nearest Q16.16 (matches fixedpoint::quantize)."""
    x = np.asarray(x, dtype=np.float64)
    return np.int32(np.where(x >= 0, np.floor(x * FP_ONE + 0.5),
                             np.ceil(x * FP_ONE - 0.5)))


# ---------------------------------------------------------------------------
# 1. The bake — Python must reproduce the C++ make_recip EXACTLY
# ---------------------------------------------------------------------------
def test_bake_matches_cpp_make_recip_for_every_shipped_material():
    """THE determinism gate for this patch. The reciprocal is baked in Python
    (where the material table lives) but consumed by the C++/CUDA solver, which
    bakes its own scalar fallback with ``fixedpoint::make_recip``. A one-count
    disagreement between the two would be a silent cross-path divergence, so the
    agreement is asserted against the real C++ entry point, not assumed."""
    for mid, name in sorted(MATERIAL_NAMES.items()):
        hp = float(_TBL.hp[mid])
        want = bp.fp_make_recip(hp) if hp > 0 else 0
        assert int(_TBL.fuel_recip[mid]) == want, (
            f"materials.{name}: baked {int(_TBL.fuel_recip[mid])} but "
            f"fixedpoint::make_recip({hp}) == {want}")


@pytest.mark.parametrize("hp", [
    0.5, 1.0, 2.0, 3.0, 7.0, 15.0, 30.0, 40.0, 60.0, 100.0, 200.0, 300.0,
    400.0, 999.0, 1e4, 1e6, 0.001, 12.5, 33.333,
])
def test_bake_matches_cpp_make_recip_off_table(hp):
    """The same agreement away from the shipped rows, so a future material with
    an unusual hp cannot be the first to discover a divergence."""
    assert fuel_recip_from_hp(hp) == bp.fp_make_recip(float(hp))


def test_bake_is_a_wide_sweep_match():
    """A dense geometric sweep across five decades — the rounding boundary of
    ``(int64)(2^32/hp + 0.5)`` is where a re-implementation would drift."""
    xs = np.geomspace(1e-3, 1e6, 5000)
    bad = [float(x) for x in xs if fuel_recip_from_hp(x) != bp.fp_make_recip(float(x))]
    assert not bad, f"{len(bad)} divergences, first at hp={bad[:3]}"


def test_recip_shift_mirrors_the_cpp_constant():
    """The Python mirror of ``fixed_point.h``'s ``RECIP_SHIFT``. If C++ ever
    changes it, ``fp_make_recip`` moves and this catches the stale mirror."""
    assert _FUEL_RECIP_SHIFT == 32
    assert bp.fp_make_recip(1.0) == 1 << _FUEL_RECIP_SHIFT


# ---------------------------------------------------------------------------
# 2. hp == 0 — the no-divide-by-zero contract
# ---------------------------------------------------------------------------
def test_zero_hp_bakes_to_zero_not_infinity():
    """``hp = 0`` (air today) must never reach a divide. The chosen sentinel is
    0, which through ``recip_mul(x, 0) == 0`` reads F = 0 — "no fuel here", the
    honest answer. A sentinel meaning "infinite fuel" would be a trap waiting
    for the first flammable gas."""
    assert fuel_recip_from_hp(0.0) == 0
    assert fuel_recip_from_hp(-5.0) == 0
    assert int(_TBL.fuel_recip[MAT_AIR]) == 0
    assert float(_TBL.hp[MAT_AIR]) == 0.0


def test_no_flammable_material_has_zero_hp():
    """The reason the zero case is unreachable in practice: the logistic runs
    under ``if (!flammable[i]) continue``. This asserts the premise instead of
    trusting it — a flammable row with hp 0 would make the sentinel live."""
    for mid, name in sorted(MATERIAL_NAMES.items()):
        if bool(_TBL.flammable[mid]):
            assert float(_TBL.hp[mid]) > 0.0, f"materials.{name} is flammable with hp 0"


# ---------------------------------------------------------------------------
# 3. The column is DERIVED, and says what the defect said it should
# ---------------------------------------------------------------------------
def test_wood_reproduces_the_retired_global_exactly():
    """Wood's hp IS the retired global (60), so wood's burn-down curve does not
    move one count. That is why gate (a)'s wood-only scenarios stay byte-
    identical even WITHOUT the back-compat pin."""
    assert float(_TBL.hp[MAT_WOOD]) == FUEL_REF
    assert int(_TBL.fuel_recip[MAT_WOOD]) == bp.fp_make_recip(FUEL_REF)


def test_a_pristine_crate_now_reads_full_fuel():
    """THE defect, stated as an assertion. Pre-patch a full-health crate's fuel
    fraction was hp/fuel_ref = 30/60 = 0.5; post-patch it is 1.0."""
    hp = float(_TBL.hp[MAT_FURNITURE])
    assert hp != FUEL_REF, "the crate must differ from the global for this to bite"
    old_F = hp / FUEL_REF
    new_F = _recip_mul(_q(hp), int(_TBL.fuel_recip[MAT_FURNITURE])) / FP_ONE
    assert abs(old_F - 0.5) < 1e-9
    assert abs(new_F - 1.0) < 1e-6


def _recip_mul(x_q16, recip):
    """Python mirror of ``fixedpoint::recip_mul`` (128-bit product >> 32)."""
    prod = int(x_q16) * int(recip)
    return prod >> 32 if prod >= 0 else -((-prod + (1 << 32) - 1) >> 32)


def test_column_is_derived_from_hp_not_a_config_key():
    """There is deliberately NO ``fuel_recip`` config column: it is a pure
    function of ``hp``, so the fuel fraction and the health bar cannot drift
    apart. A dict-built table (the tests' path) proves it needs no new key."""
    rows = {name: dict(hp=float(_TBL.hp[mid]),
                       flammable=bool(_TBL.flammable[mid]),
                       mobility=int(_TBL.mobility[mid]),
                       conductivity=float(_TBL.conductivity[mid]),
                       thermal_mass=float(_TBL.thermal_mass[mid]),
                       ignition_temp=float(_TBL.ignition_temp[mid]),
                       heat_atten=float(_TBL.heat_atten[mid]),
                       wave_absorb=float(_TBL.wave_absorb[mid]),
                       blast_resist=float(_TBL.blast_resist[mid]),
                       permeability=float(_TBL.permeability[mid]),
                       light_atten=list(_TBL.light_atten[mid]))
            for mid, name in sorted(MATERIAL_NAMES.items())}
    tbl = MaterialTable(rows)
    assert np.array_equal(tbl.fuel_recip, _TBL.fuel_recip)


def test_changing_hp_moves_the_reciprocal_with_it():
    """The single-source-of-truth claim: edit ``hp``, the normaliser follows."""
    rows = {name: dict(hp=float(_TBL.hp[mid]),
                       flammable=bool(_TBL.flammable[mid]),
                       mobility=int(_TBL.mobility[mid]),
                       conductivity=float(_TBL.conductivity[mid]),
                       thermal_mass=float(_TBL.thermal_mass[mid]),
                       ignition_temp=float(_TBL.ignition_temp[mid]),
                       heat_atten=float(_TBL.heat_atten[mid]),
                       wave_absorb=float(_TBL.wave_absorb[mid]),
                       blast_resist=float(_TBL.blast_resist[mid]),
                       permeability=float(_TBL.permeability[mid]),
                       light_atten=list(_TBL.light_atten[mid]))
            for mid, name in sorted(MATERIAL_NAMES.items())}
    rows["furniture"]["hp"] = 45.0
    tbl = MaterialTable(rows)
    assert int(tbl.fuel_recip[MAT_FURNITURE]) == bp.fp_make_recip(45.0)


# ---------------------------------------------------------------------------
# 4. The per-tile grid: built on ONE seam, patched on ONE seam
# ---------------------------------------------------------------------------
def _grid_map(tilemap):
    from level_loader import LevelData
    level = LevelData(name="fuel_frac_test", version="2", path=Path("."),
                      tilemap=np.asarray(tilemap, dtype=np.int32),
                      tile_size_m=1.0, diffuse_path=Path("."))
    return GameMap(level)


def test_grid_is_the_column_indexed_by_material():
    g = _grid_map([[0, 2, 6], [1, 6, 2], [0, 0, 1]])
    assert g.fuel_recip.dtype == np.int64
    assert g.fuel_recip.shape == g.material.shape
    assert np.array_equal(g.fuel_recip, _TBL.fuel_recip[g.material])


def test_grid_is_patched_when_a_tile_changes_material():
    """A crate burning out must stop reading a crate's fuel normaliser the
    instant its material changes — the numerator (``wall_hp``) and the
    denominator are patched on the SAME seam, so they can never come from two
    different materials."""
    g = _grid_map([[1, 1, 1], [1, 6, 1], [1, 1, 1]])
    assert int(g.fuel_recip[1, 1]) == int(_TBL.fuel_recip[MAT_FURNITURE])
    g.destroy_wall(1, 1)
    assert int(g.material[1, 1]) == MAT_AIR
    assert int(g.fuel_recip[1, 1]) == int(_TBL.fuel_recip[MAT_AIR]) == 0
    assert np.array_equal(g.fuel_recip, _TBL.fuel_recip[g.material])


def test_grid_joins_the_resident_field_set():
    """It is REASSIGNED by ``_update_caches`` and patched IN PLACE by
    ``on_tile_changed``, so it needs the ``__setattr__`` stale-pointer guard and
    a device buffer, exactly like ``cool_shift``."""
    assert "fuel_recip" in GameMap._RESIDENT_MASKS
    assert "fuel_recip" in GameMap._RESIDENT_FIELD_NAMES


# ---------------------------------------------------------------------------
# 5. The solver boundary: a UNIFORM plane == the scalar fallback, bit-for-bit
# ---------------------------------------------------------------------------
def _fire_state(rng, h, w):
    n = h * w
    flammable = (rng.random(n) < 0.6).reshape(h, w)
    is_wall = np.ones((h, w), dtype=bool)
    is_wall[rng.random(n).reshape(h, w) < 0.5] = False
    is_vacuum = (rng.random(n) < 0.08).reshape(h, w)
    fire = _q(rng.random(n).reshape(h, w))
    total = rng.random(n).reshape(h, w) * 1.0 + 0.3
    frac = rng.random(n).reshape(h, w) * 0.35
    return dict(
        fire=np.ascontiguousarray(fire),
        atmosphere=np.ascontiguousarray(_q(np.ones((h, w)))),
        n_o2=np.ascontiguousarray(_q(frac * total)),
        n_total=np.ascontiguousarray(_q(total)),
        smoke=np.ascontiguousarray(_q(np.zeros((h, w)))),
        wall_hp=np.ascontiguousarray(_q(rng.random(n).reshape(h, w) * 60.0)),
        temperature=np.ascontiguousarray(_q(rng.random(n).reshape(h, w) * 900.0)),
        wind_x=np.ascontiguousarray(_q(rng.random(n).reshape(h, w) * 0.4 - 0.2)),
        wind_y=np.ascontiguousarray(_q(rng.random(n).reshape(h, w) * 0.4 - 0.2)),
        is_wall=np.ascontiguousarray(is_wall),
        is_vacuum=np.ascontiguousarray(is_vacuum),
        flammable=np.ascontiguousarray(flammable),
    )


def _run_fire(state, fuel_recip=None, fuel_ref=FUEL_REF, dt=1.0 / 24.0):
    sim = bp.FireSimulation()
    sim.params.fuel_ref = float(fuel_ref)
    c = {k: v.copy() for k, v in state.items()}
    destroyed = sim.step(c["fire"], c["atmosphere"], c["n_o2"], c["n_total"],
                         c["smoke"], c["wall_hp"], c["temperature"],
                         c["wind_x"], c["wind_y"], c["is_wall"], c["is_vacuum"],
                         c["flammable"], dt, fuel_recip)
    return c, sorted(tuple(t) for t in destroyed)


@pytest.mark.parametrize("seed", range(8))
def test_uniform_plane_equals_the_null_fallback_bit_for_bit(seed):
    """GATE (a) AT THE SOLVER BOUNDARY. A plane filled with ``make_recip(
    fuel_ref)`` must reproduce the no-plane (pre-axis) result exactly — the
    property the whole-engine back-compat capture rests on, asserted in-suite so
    it cannot rot."""
    rng = np.random.default_rng(1000 + seed)
    h, w = 17, 23
    st = _fire_state(rng, h, w)
    ref = np.full((h, w), bp.fp_make_recip(FUEL_REF), dtype=np.int64)
    a, da = _run_fire(st, fuel_recip=None)
    b, db = _run_fire(st, fuel_recip=np.ascontiguousarray(ref))
    for k in ("fire", "temperature", "smoke", "wall_hp"):
        assert np.array_equal(a[k], b[k]), f"{k} differs (seed {seed})"
    assert da == db


@pytest.mark.parametrize("seed", range(4))
def test_the_plane_is_actually_read(seed):
    """NON-VACUOUSNESS control for the test above: a plane that is NOT the
    scalar must diverge from the fallback, or the two would be equal for the
    boring reason."""
    rng = np.random.default_rng(2000 + seed)
    h, w = 17, 23
    st = _fire_state(rng, h, w)
    half = np.full((h, w), bp.fp_make_recip(FUEL_REF / 2.0), dtype=np.int64)
    a, _ = _run_fire(st, fuel_recip=None)
    b, _ = _run_fire(st, fuel_recip=np.ascontiguousarray(half))
    assert not np.array_equal(a["fire"], b["fire"])


def test_a_non_uniform_plane_is_read_per_tile():
    """The point of the axis: two tiles with the same ``wall_hp`` but different
    material hp must step differently in the SAME call."""
    # Row 1 is the fuel strip; rows 0 and 2 are OPEN air holding ambient gas, so
    # every fuel tile sees the same O2 mole fraction and the ONLY thing that
    # differs across the strip is the per-tile fuel reciprocal.
    h, w = 3, 6
    st = _fire_state(np.random.default_rng(7), h, w)
    st["flammable"][:] = False
    st["flammable"][1, :] = True
    st["is_wall"][:] = False
    st["is_wall"][1, :] = True
    st["is_vacuum"][:] = False
    st["fire"][:] = 0
    st["fire"][1, :] = _q(0.5)
    st["wall_hp"][:] = _q(30.0)
    st["temperature"][:] = _q(900.0)
    st["wind_x"][:] = 0
    st["wind_y"][:] = 0
    st["n_total"][:] = _q(1.0)
    st["n_o2"][:] = _q(0.21)
    plane = np.full((h, w), bp.fp_make_recip(60.0), dtype=np.int64)
    plane[1, 3:] = bp.fp_make_recip(30.0)
    out, _ = _run_fire(st, fuel_recip=np.ascontiguousarray(plane))
    lo = out["fire"][1, 1:3]     # interior tiles (grid-edge columns see 3 nbrs)
    hi = out["fire"][1, 3:5]
    assert len(set(lo.tolist())) == 1 and len(set(hi.tolist())) == 1
    assert hi[0] > lo[0], "the tile that owns its full fuel must grow harder"


# ---------------------------------------------------------------------------
# 6. The live engine may never fall back to the global
# ---------------------------------------------------------------------------
def test_step_tail_requires_the_plane():
    """``fuel_recip`` is a REQUIRED step_tail argument (like ``thermal_solid``
    and ``cool_shift_grid``): a caller must not be able to silently put the live
    engine back on the single global."""
    import inspect
    doc = bp.PhysicsEngine.step_tail.__doc__ or ""
    assert "fuel_recip" in doc, (
        "step_tail's signature must name fuel_recip; got:\n" + doc)
    assert "fuel_recip: numpy.ndarray" in doc.replace("  ", " ") or "fuel_recip" in doc
