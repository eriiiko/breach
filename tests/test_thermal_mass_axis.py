"""The thermal-mass axis — furniture burns like an object (P1, CPU).

Design: ``docs/thermal_mass_axis_design_2026-07-25.md`` (Fable, blessed
2026-07-25) + ``docs/thermal_mass_axis_build_addendum_2026-07-30.md``.

THE DEFECT. The unified thermal pass selected its medium (gas vs solid) by
asking ``solid[i]``, which is ``permeability <= 0`` — a **flow** property.
Furniture (``permeability = 0.5``, the deliberate "shield but not seal" soft
body) therefore fell into the GAS thermal regime, so a burning crate's
temperature was hot gas the fire's own plume advected away rather than an
object temperature governed by COOL_SHIFT. ``thermal_mass = 8`` was already
correct in config; nothing read it as a **medium selector**.

THE FIX (this patch). A derived ``thermal_solid`` mask (``thermal_mass > 0``)
that replaces ``solid`` at the six medium-test sites in the thermal solver and
nowhere else. ``permeability`` / ``solid`` / ``dyn_permeability`` / mobility /
LoS are UNTOUCHED — gas and water still seep past a crate.

The load-bearing property this module pins is the patch's own zero-tolerance
gate: **furniture is the ONLY material that is permeable AND thermally solid**,
so on any furniture-free map ``thermal_solid == solid`` elementwise and every
thermal path is byte-identical to before the patch.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_thermal_mass_axis.py -q
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
from level_loader import load as load_level  # noqa: E402
from simulation.gamemap import GameMap  # noqa: E402
from simulation.materials import (  # noqa: E402
    MAT_AIR, MAT_FURNITURE, MAT_HULL, MAT_WOOD, MATERIAL_NAMES, MaterialTable,
)

FP_ONE = 1 << 16
_TBL = MaterialTable.from_config()
NO_FACE = int(_TBL.no_face)
COOL_SHIFT = int(getattr(CFG.physics.thermal, "COOL_SHIFT", 5))


# ---------------------------------------------------------------------------
# 1. The material column + the derived per-id mask
# ---------------------------------------------------------------------------
def test_air_is_thermal_mass_zero_and_the_only_gas_row():
    """Air carries ``thermal_mass = 0`` == the GAS thermal regime (addendum D2).

    The blessed predicate is ``thermal_mass > 0``; it is unsatisfiable while air
    is 1 (air would become a thermal solid and the WHOLE grid would take the
    solid regime), so 0 had to become legal.
    """
    tbl = MaterialTable.from_config(CFG)
    assert tbl.thermal_mass[MAT_AIR] == 0.0
    assert bool(tbl.thermal_solid[MAT_AIR]) is False
    # Every other shipped row is a thermal solid.
    for mid in sorted(MATERIAL_NAMES):
        if mid == MAT_AIR:
            continue
        assert bool(tbl.thermal_solid[mid]) is True, MATERIAL_NAMES[mid]


def test_existing_solid_materials_keep_their_tuned_thermal_mass():
    """Addendum D1: hull/steel 32, glass 16, wood/door/door_closed/furniture 8.

    These are LIVE TUNED physics (per-tile ``heat >> log2(thermal_mass)``);
    flattening them to a single 8 would move every heat->T convert on metal and
    glass and blow the byte-identity gate.
    """
    tbl = MaterialTable.from_config(CFG)
    expected = {"air": 0, "hull": 32, "wood": 8, "door": 8, "steel": 32,
                "glass": 16, "furniture": 8, "door_closed": 8}
    got = {name: int(round(float(v)))
           for name, v in zip(tbl.names, tbl.thermal_mass.tolist())}
    assert got == expected


def test_thermal_solid_is_derived_from_thermal_mass_not_permeability():
    """``thermal_solid`` is the THERMAL axis; ``solid`` is the FLOW axis.

    Furniture is the ONE row where they disagree — and that divergence is the
    entire point of the patch (addendum D4).
    """
    tbl = MaterialTable.from_config(CFG)
    thermal = tbl.thermal_solid
    flow = tbl.permeability <= 0.0
    divergent = [name for name, a, b in zip(tbl.names, flow.tolist(),
                                            thermal.tolist()) if a != b]
    assert divergent == ["furniture"], (
        "furniture must be the ONLY permeable thermal solid — the byte-identity "
        f"gate rests on it; got {divergent}")
    assert bool(thermal[MAT_FURNITURE]) is True
    assert bool(flow[MAT_FURNITURE]) is False


def test_per_tile_shift_matches_log2_thermal_mass():
    tbl = MaterialTable.from_config(CFG)
    for name, tm, shift, ts in zip(tbl.names, tbl.thermal_mass.tolist(),
                                   tbl.heat_inv_shift.tolist(),
                                   tbl.thermal_solid.tolist()):
        tm_int = int(round(float(tm)))
        if tm_int == 0:
            # A never-read placeholder: the mask routes gas tiles away from the
            # shift path entirely.
            assert shift == 0 and ts is False, name
        else:
            assert (1 << shift) == tm_int, name
            assert ts is True, name


# ---------------------------------------------------------------------------
# 2. Loader validation (addendum D2)
# ---------------------------------------------------------------------------
def _row(**over):
    base = dict(hp=10.0, flammable=False, mobility=1000, conductivity=1.0,
                thermal_mass=8, ignition_temp=0.0, heat_atten=0.0,
                wave_reflect=0.0, wave_absorb=0.0, blast_resist=0.0,
                light_atten=[0.0, 0.0, 0.0])
    base.update(over)
    return base


def _table(thermal_mass_by_name):
    cfg = {name: _row(thermal_mass=thermal_mass_by_name.get(name, 8))
           for name in MATERIAL_NAMES.values()}
    return MaterialTable(cfg)


def test_loader_accepts_thermal_mass_zero():
    tbl = _table({"air": 0})
    assert bool(tbl.thermal_solid[MAT_AIR]) is False
    assert int(tbl.heat_inv_shift[MAT_AIR]) == 0


@pytest.mark.parametrize("bad", [3, 6, 12, 20, 100])
def test_loader_still_rejects_non_power_of_two_above_one(bad):
    """Only 0 is exempt: everything >= 1 keeps today's power-of-two contract
    (the convert is a free arithmetic right shift, no divide)."""
    with pytest.raises(ValueError, match="thermal_mass"):
        _table({"wood": bad})


@pytest.mark.parametrize("good", [1, 2, 4, 8, 16, 32, 64])
def test_loader_accepts_powers_of_two(good):
    tbl = _table({"wood": good})
    assert (1 << int(tbl.heat_inv_shift[MAT_WOOD])) == good
    assert bool(tbl.thermal_solid[MAT_WOOD]) is True


# ---------------------------------------------------------------------------
# 3. The derived GRID (addendum D3 — ONE build seam, ONE patch seam)
# ---------------------------------------------------------------------------
def test_grid_mask_is_the_table_column_projected():
    g = GameMap(load_level("unhcr_vessel"))
    assert g.thermal_solid.dtype == np.bool_
    assert g.thermal_solid.shape == g.solid.shape
    assert np.array_equal(g.thermal_solid,
                          g.materials.thermal_solid[g.material])


def test_on_tile_changed_patches_the_mask_both_ways():
    """The mask rides the SAME structural-edit seam as ``heat_inv_shift``, so a
    crate joins/leaves the solid thermal regime the instant its material
    changes — and NOTHING else on the grid moves."""
    g = GameMap(load_level("unhcr_vessel"))
    ys, xs = np.where((g.material == MAT_AIR) & ~g.is_vacuum)
    y, x = int(ys[len(ys) // 2]), int(xs[len(ys) // 2])

    before = g.thermal_solid.copy()
    assert bool(g.thermal_solid[y, x]) is False       # air

    g.material[y, x] = MAT_FURNITURE
    g.on_tile_changed(y, x)
    assert bool(g.thermal_solid[y, x]) is True        # crate: thermal solid
    assert bool(g.solid[y, x]) is False               # ...but NOT flow-solid
    assert g.permeability[y, x] == np.float32(0.5)    # flow axis untouched
    touched = before != g.thermal_solid
    assert touched.sum() == 1 and touched[y, x], "patch must be O(1), one tile"

    g.material[y, x] = MAT_AIR
    g.on_tile_changed(y, x)
    assert bool(g.thermal_solid[y, x]) is False
    assert np.array_equal(g.thermal_solid, before)


def test_destroying_a_crate_leaves_the_solid_thermal_regime():
    g = GameMap(load_level("unhcr_vessel"))
    ys, xs = np.where((g.material == MAT_AIR) & ~g.is_vacuum)
    y, x = int(ys[len(ys) // 2]), int(xs[len(ys) // 2])
    g.material[y, x] = MAT_FURNITURE
    g.on_tile_changed(y, x)
    assert bool(g.thermal_solid[y, x]) is True
    g.destroy_wall(y, x)
    assert int(g.material[y, x]) == MAT_AIR
    assert bool(g.thermal_solid[y, x]) is False


# ---------------------------------------------------------------------------
# 4. THE GATE: furniture-free identity
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("level", ["unhcr_vessel", "planetside_demo",
                                   "airlock_demo", "door_test"])
def test_furniture_free_level_mask_equals_solid(level):
    """The patch's zero-tolerance gate, at the mask level: with no furniture on
    the map the new medium mask IS the old one, elementwise — so every thermal
    path is byte-identical on those maps by construction."""
    g = GameMap(load_level(level))
    if (g.material == MAT_FURNITURE).any():
        pytest.skip(f"{level} carries furniture")
    assert np.array_equal(g.thermal_solid, g.solid)


def test_furniture_bearing_level_differs_exactly_on_furniture():
    g = GameMap(load_level("playground"))
    furn = (g.material == MAT_FURNITURE)
    assert furn.any(), "playground is expected to carry furniture"
    assert np.array_equal(g.thermal_solid != g.solid, furn)


# ---------------------------------------------------------------------------
# 5. The SOLVER: the medium test is thermal_solid, and only that
# ---------------------------------------------------------------------------
def _solver():
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.cool_shift = COOL_SHIFT
    s.cool_shift_vacuum = int(getattr(CFG.physics.thermal,
                                      "COOL_SHIFT_VACUUM", 3))
    s.o2_vacuum_thresh = float(getattr(CFG.physics.thermal,
                                       "o2_vacuum_thresh", 0.3))
    s.c_v = float(getattr(CFG.physics.thermal, "c_v", 1.0))
    s.n_floor_heat = float(getattr(CFG.physics.thermal, "n_floor_heat", 0.05))
    return s


def _grid(h=5, w=5):
    """A tiny room: hull ring (solid + thermal solid), air interior, conduction
    disabled (all-NO_FACE) so the convert + cooling passes act in isolation."""
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    return dict(
        temperature=np.zeros((h, w), dtype=np.int32),
        heat=np.zeros((h, w), dtype=np.int32),
        heat_inv_shift=np.full((h, w), 3, dtype=np.int32),   # thermal_mass 8
        face_shift=np.full((h, w, 4), NO_FACE, dtype=np.int32),
        solid=solid,
        is_vacuum=np.zeros((h, w), dtype=bool),
        atmosphere=np.full((h, w), FP_ONE, dtype=np.int32),
    )


def test_permeable_thermal_solid_takes_the_SHIFT_convert_not_the_gas_deposit():
    """The crate tile: ``solid`` False, ``thermal_solid`` True.

    With the thermal mask supplied it takes the solid branch (``heat >> 3``);
    with the mask omitted (the nullptr fallback == pre-patch behaviour) the SAME
    tile takes the gas branch (``deposit / (N*c_v)``, N == 1 here). One deposit,
    two regimes — the whole defect in one assertion.
    """
    s = _solver()
    deposit = 8 * FP_ONE            # 8.0 game units of heat energy
    for use_mask in (True, False):
        gr = _grid()
        gr["heat"][2, 2] = deposit
        ts = gr["solid"].copy()
        ts[2, 2] = True             # permeable, but a thermal SOLID
        kw = {"thermal_solid": ts} if use_mask else {}
        s.step(gr["temperature"], gr["heat"], gr["heat_inv_shift"],
               gr["face_shift"], gr["solid"], gr["is_vacuum"],
               gr["atmosphere"], **kw)
        got = int(gr["temperature"][2, 2])
        if use_mask:
            # solid regime: heat >> 3, then COOL_SHIFT ambient decay.
            gain = deposit >> 3
            assert got == gain - (gain >> COOL_SHIFT)
        else:
            # gas regime: full deposit / (N * c_v), NO ambient decay.
            assert got == deposit


def test_thermal_solid_blocks_gas_T_advection_across_the_tile():
    """Medium sites 2-4: with the crate in the thermal regime, the gas-T
    semi-Lagrangian pass neither advects the crate's own T nor samples through
    it (it is an occluder / sealed corner for the ray-walk)."""
    s = _solver()
    gr = _grid(5, 7)
    # Hot gas to the left of the crate, wind blowing +x.
    gr["temperature"][2, 1] = 400 * FP_ONE
    wind_x = np.zeros((5, 7), dtype=np.int32)
    wind_y = np.zeros((5, 7), dtype=np.int32)
    wind_x[:, :] = FP_ONE // 64
    crate = (2, 3)

    results = {}
    for use_mask in (True, False):
        t = gr["temperature"].copy()
        ts = gr["solid"].copy()
        ts[crate] = True
        kw = {"thermal_solid": ts} if use_mask else {}
        s.step(t, gr["heat"], gr["heat_inv_shift"], gr["face_shift"],
               gr["solid"], gr["is_vacuum"], gr["atmosphere"],
               wind_x=wind_x, wind_y=wind_y, dt=1.0, **kw)
        results[use_mask] = t
    # Pre-patch: the crate tile is open air, so the advection pass rewrites its
    # T from the upwind sample.  Post-patch: it is a thermal solid, so the
    # advection pass skips it entirely (only convert/cool can touch it, and
    # there is no deposit here).
    assert int(results[True][crate]) == 0
    assert results[False][crate] != results[True][crate]


def test_furniture_free_grid_is_byte_identical_with_and_without_the_mask():
    """Gate (a) at the solver boundary: when ``thermal_solid == solid``
    elementwise (any furniture-free map), supplying the mask changes NOTHING —
    every pass, every cell, tol 0."""
    rng = np.random.default_rng(20260730)
    s = _solver()
    for _ in range(8):
        gr = _grid(9, 11)
        h, w = gr["solid"].shape
        # A random interior wall set (still furniture-free: thermal == flow).
        interior = np.zeros((h, w), dtype=bool)
        interior[1:-1, 1:-1] = rng.random((h - 2, w - 2)) < 0.25
        gr["solid"] |= interior
        gr["is_vacuum"][:, -1] = True
        gr["heat"][:] = (rng.random((h, w)) * 4 * FP_ONE).astype(np.int32)
        gr["temperature"][:] = (rng.integers(-200, 900, (h, w)) * FP_ONE
                                ).astype(np.int32)
        gr["atmosphere"][:] = (rng.random((h, w)) * FP_ONE).astype(np.int32)
        wind_x = (rng.integers(-FP_ONE // 32, FP_ONE // 32, (h, w))
                  ).astype(np.int32)
        wind_y = (rng.integers(-FP_ONE // 32, FP_ONE // 32, (h, w))
                  ).astype(np.int32)

        a = gr["temperature"].copy()
        b = gr["temperature"].copy()
        common = (gr["heat"], gr["heat_inv_shift"], gr["face_shift"],
                  gr["solid"], gr["is_vacuum"], gr["atmosphere"])
        for _tick in range(4):
            s.step(a, *common, wind_x=wind_x, wind_y=wind_y, dt=0.04)
            s.step(b, *common, wind_x=wind_x, wind_y=wind_y, dt=0.04,
                   thermal_solid=gr["solid"])
        assert np.array_equal(a, b), "furniture-free identity broken (gate a)"


# ---------------------------------------------------------------------------
# 6. Addendum D5 — the structural-edit temperature seed
# ---------------------------------------------------------------------------
def test_seal_close_t_seeds_from_thermal_solid_neighbours():
    """``seal_tiles`` seeds a newly-sealed tile's T from the integer mean of its
    PRE-call THERMAL-solid 4-neighbours (a burning crate can now warm the door
    panel that closes beside it).  A no-op wherever thermal_solid == solid."""
    g = GameMap(load_level("unhcr_vessel"))
    # Find an interior air tile whose 4-neighbours are all open air.
    h, w = g.material.shape
    target = None
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if g.solid[y, x] or g.is_vacuum[y, x]:
                continue
            nbrs = [(y - 1, x), (y + 1, x), (y, x + 1), (y, x - 1)]
            if all(not g.solid[p] and not g.is_vacuum[p] for p in nbrs) \
                    and not g.water_depth[y, x]:
                target = (y, x, nbrs)
                break
        if target:
            break
    assert target is not None, "no all-open interior tile found"
    y, x, nbrs = target

    # Make ONE neighbour a hot crate: permeable (not `solid`) but thermal solid.
    ny, nx = nbrs[0]
    g.material[ny, nx] = MAT_FURNITURE
    g.on_tile_changed(ny, nx)
    g.temperature[ny, nx] = 500 * FP_ONE

    g.seal_tiles([(y, x)], MAT_HULL)
    assert int(g.temperature[y, x]) == 500 * FP_ONE, (
        "the crate is a thermal solid and must be able to seed close-T")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
