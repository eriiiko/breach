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
from simulation import gas_fixed  # noqa: E402
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
    # Subset check (not full-dict equality): the test's OWN name/intent is
    # "EXISTING materials keep their tuned value" — a later row (P-F4a's
    # kindling, thermal_mass=8 per its own locked spec) must not force an
    # edit here just to be listed; it is covered by its own material-row
    # tests instead.
    assert expected.items() <= got.items(), (
        f"an EXISTING material's thermal_mass moved: expected {expected}, "
        f"got {got}")


def test_thermal_solid_is_derived_from_thermal_mass_not_permeability():
    """``thermal_solid`` is the THERMAL axis; ``solid`` is the FLOW axis.

    Furniture was the ONE row where they disagreed when this axis landed —
    that divergence was the entire point of the patch (addendum D4). P-F4a's
    kindling (a real material row, cellulosic-copied from furniture per its
    own locked spec) shares the SAME shape by construction: permeability 0.5
    (flow-open) + thermal_mass 8 (a thermal solid) — a second, expected
    member of this set, not a regression.
    """
    tbl = MaterialTable.from_config(CFG)
    thermal = tbl.thermal_solid
    flow = tbl.permeability <= 0.0
    divergent = [name for name, a, b in zip(tbl.names, flow.tolist(),
                                            thermal.tolist()) if a != b]
    assert divergent == ["furniture", "kindling"], (
        "furniture + kindling must be the ONLY permeable thermal solids — "
        f"the byte-identity gate rests on this set; got {divergent}")
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


def test_temperature_solver_gas_T_advection_is_retired():
    """P-E1 (energy-books design SS2.1.1; round-1 finding L3-5) — REPLACES
    `test_thermal_solid_blocks_gas_T_advection_across_the_tile`.

    TemperatureSolver Pass 0b was the engine's SECOND semi-Lagrangian T-copier
    (medium sites 2/6, 3/6 and 4/6 lived there). It was dormant in the live
    engine only because `step_tail` passes null winds — "one plumbing change
    must not silently re-open the mint" — so the arc DELETES it on both
    backends. The guarantee is now the strongest available one: supplying wind
    changes nothing, with the mask or without it."""
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
        for windy in (True, False):
            t = gr["temperature"].copy()
            ts = gr["solid"].copy()
            ts[crate] = True
            kw = {"thermal_solid": ts} if use_mask else {}
            if windy:
                kw.update(wind_x=wind_x, wind_y=wind_y, dt=1.0)
            s.step(t, gr["heat"], gr["heat_inv_shift"], gr["face_shift"],
                   gr["solid"], gr["is_vacuum"], gr["atmosphere"], **kw)
            results[(use_mask, windy)] = t
    for use_mask in (True, False):
        assert np.array_equal(results[(use_mask, True)],
                              results[(use_mask, False)]), (
            "TemperatureSolver Pass 0b is back: supplying wind moved gas T "
            "(mask=%s)" % ("on" if use_mask else "off"))
    # The hot upwind gas cell simply STAYS where it is (nothing advected).
    assert int(results[(True, True)][2, 1]) != 0


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


# ---------------------------------------------------------------------------
# 7. P2 — the resident half (CPU-side; the GPU lockstep is
#    tests/test_cuda_thermal_mass.py, which skips without a CUDA build)
# ---------------------------------------------------------------------------
def test_thermal_solid_is_in_the_resident_mask_set():
    """P2: the medium mask must be a RESIDENT field — it gets a device buffer +
    one upload at ``enable_residency``, and (the load-bearing half on a
    CUDA-less machine) the ``__setattr__`` stale-pointer guard covers it, so it
    can never be silently REASSIGNED out from under a device pointer the way a
    non-resident field could."""
    assert "thermal_solid" in GameMap._RESIDENT_MASKS
    assert "thermal_solid" in GameMap._RESIDENT_FIELD_NAMES
    # It rides with `solid`: same structural seam, same mutation pattern.
    assert "solid" in GameMap._RESIDENT_MASKS


def test_thermal_solid_grid_is_a_plain_bool_grid():
    """Determinism contract: the mask crosses to C++/CUDA as a bool plane — no
    floats, no packing, C-contiguous, exactly (h, w)."""
    g = GameMap(load_level("playground"))
    assert g.thermal_solid.dtype == np.bool_
    assert g.thermal_solid.shape == g.solid.shape
    assert g.thermal_solid.flags["C_CONTIGUOUS"]


# ---------------------------------------------------------------------------
# 8. P-EOS — the thermal medium inside the EOS pass
#    (docs/thermal_mass_eos_ruling_2026-07-30.md; the ruling that closes P1's
#    escalation). THE GOVERNING RULE: on a thermal_solid tile `temperature[]` is
#    OWNED by the TemperatureSolver — the EOS reads T (for p* = C·N·T) and never
#    writes it there. These tests drive the two documented CPU replay entries
#    (eos_sl_advect_ref / eos_kick_compression_ref), which call the SAME
#    file-local routines EOSSolver::step calls, so they pin the live arithmetic.
#    The CPU<->CUDA lockstep half is tests/test_cuda_thermal_mass_eos.py.
# ---------------------------------------------------------------------------
def _eos_world(h=12, w=16, crate=((5, 8), (7, 11))):
    """A hull-shelled air box with a FURNITURE block: the only place where the
    thermal medium (thermal_mass > 0) diverges from the flow medium
    (permeability <= 0). Returns (solid, is_vacuum, thermal_solid, perm, furn)."""
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    is_vacuum = np.zeros((h, w), dtype=bool)
    thermal_solid = solid.copy()
    perm = np.where(solid, 0.0, 1.0).astype(np.float32)
    furn = np.zeros((h, w), dtype=bool)
    (y0, y1), (x0, x1) = crate
    furn[y0:y1, x0:x1] = True
    furn &= ~solid
    thermal_solid |= furn
    perm[furn] = 0.5          # shield, NOT seal — permeability is untouched
    return solid, is_vacuum, thermal_solid, perm, furn


def _q32(a):
    return np.ascontiguousarray(a, dtype=np.int32)


def test_eos_sl_advect_is_u_only_and_never_writes_temperature():
    """P-E1 (energy-books design SS2.1.1) — REPLACES the two step-1b T-sample
    tests (`..._does_not_write_temperature_on_a_thermal_solid` and
    `..._treats_a_thermal_solid_as_a_backtrace_occluder`), whose premise WAS
    the semi-Lagrangian T sample.

    That sample is RETIRED: it was T-WRITE SITE 1/2 and the measured mint (a
    temperature COPY onto mass that never paid for it). Ruling A1's guarantee
    therefore no longer needs the `thermal_solid` mask to hold at this site —
    it holds STRUCTURALLY, for every cell, because SL advection writes no
    temperature at all. That is strictly stronger than the two tests it
    replaces, so it is asserted directly, with the mask and without it, with
    the velocity field asserted to still move (non-vacuity).

    (Ruling A2's job — a hot crate must not heat downwind gas for free — is
    now carried by ts-face rule (d) in the energy books, gated at the engine
    level by `test_eos_energy_transport_never_heats_gas_from_a_crate` below.)
    """
    h, w = 12, 16
    solid, vac, tsol, perm, furn = _eos_world(h, w)
    T0 = _q32(np.where(furn, 900 * FP_ONE, 100 * FP_ONE))
    wx0 = _q32(np.full((h, w), 6 * FP_ONE))
    wy0 = _q32(np.zeros((h, w)))
    wx0[solid] = 0

    for kw in ({"thermal_solid": tsol}, {}):
        T = T0.copy()
        wx, wy = wx0.copy(), wy0.copy()
        bp.eos_sl_advect_ref(wx, wy, T, solid, vac, perm,
                             dt=1.0 / 24.0, n_sub=4, **kw)
        assert np.array_equal(T, T0), (
            "the retired SL T-copy is back: eos_sl_advect_ref wrote "
            "temperature (mask=%s)" % ("on" if kw else "off"))
        # Non-vacuity: the pass really ran and really advected VELOCITY.
        assert not np.array_equal(wx, wx0), "vacuous: u did not move"


def test_eos_energy_transport_never_heats_gas_from_a_crate():
    """P-E1 ts-face rule (d) (design SS2.1.4), at the ENGINE level — the
    structural replacement for ruling A2's retired backtrace occluder.

    Relative energy never crosses a face touching a thermal_solid tile: mass
    still moves through a permeable crate, but it arrives carrying ZERO
    relative energy (ts->air), and gas leaving into the crate is debited at
    its OWN temperature into the counted `e_ts_residual` (air->ts). So a
    1300-deg crate parked in a windy room can NEVER warm the gas — the
    free-energy channel the occluder mask used to patch is gone at the root.
    """
    from level_loader import LevelData
    from simulation.physics_runner import PhysicsRunner

    # v1 tilemap vocabulary: 1 = hull wall, 4 = interior air. The crate is
    # stamped through `material` + on_tile_changed (this module's idiom), the
    # only place the THERMAL medium diverges from the FLOW medium.
    H = W = 16
    tm = np.full((H, W), 1, dtype=np.int32)
    tm[1:-1, 1:-1] = 4
    level = LevelData(name="e1_crate_rule_d", version="1", path=Path("."),
                      tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    g = GameMap(level)
    g.stamp_units([])
    for y in range(6, 9):
        for x in range(5, 8):
            g.material[y, x] = MAT_FURNITURE
            g.on_tile_changed(y, x)
    furn = g.thermal_solid & ~g.solid
    assert furn.any(), "fixture must carry a permeable thermal_solid crate"

    # The crate is hot, and a HOT AIR POCKET sits upwind of it; a steady wind
    # drags that air across the crate for the whole run. Both directions of
    # rule (d) are therefore exercised: air->ts (the hot pocket sheds its
    # relative energy into the counted `e_ts_residual` as it enters the crate)
    # and ts->air (the crate's own 1300 must never ride out the far side).
    # Ambient air alone would make the air->ts leg VACUOUS — at T_rel = 0 it
    # carries exactly zero relative energy, so nothing is there to destroy.
    g.temperature[:] = 0
    g.temperature[furn] = 1300 * FP_ONE
    g.temperature[6:9, 2:5] = 800 * FP_ONE          # the upwind hot pocket
    g.wind_x[~g.solid] = 5 * FP_ONE

    runner = PhysicsRunner(bp)
    runner.eos.dx = float(g.tile_size_m)
    inert_n2_idx = int(g.gases.name_to_id["inert_n2"])
    dt = 1.0 / float(CFG.clock.ticks_per_second)
    gas_cells = ~g.solid & ~g.thermal_solid
    assert gas_cells.any()
    eos = runner.engine.eos

    # The observable is the EOS TRANSPORT BRACKET, not the raw gas T: the
    # engine's OTHER thermal channels (Pass-2 conduction across the crate's
    # faces above all) are honest, separate and deliberately still live at
    # this rung — P-E2a owns them. What rule (d) claims is exactly what the
    # bracket measures: the transport pass itself may only ever DESTROY
    # relative energy at a ts face, never deliver it.
    ts_debits = 0
    for _ in range(30):
        runner.engine.run_substeps(
            g.wave_p, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
            g.obstacles, g.solid, g.is_vacuum,
            g.dyn_permeability, g.dyn_wave_absorb,
            g.gas, g.gases.diffusion, g.gases.conservative,
            g.gases.decay, inert_n2_idx, dt,
            thermal_solid=g.thermal_solid,   # the axis under test
        )
        assert int(eos.eth_transport_delta) <= 0, (
            "the transport pass CREATED %d raw of gas book-energy beside a "
            "hot crate" % int(eos.eth_transport_delta))
        ts_debits += int(eos.e_ts_residual)
    # Non-vacuity: air really did transit the crate, so rule (d) really fired.
    assert ts_debits > 0, (
        "vacuous: no air->ts face traffic, so rule (d) never engaged")
    # The crate itself keeps its object temperature (T-WRITE guard, ruling A1).
    assert int(g.temperature[furn].min()) > 0, (
        "the EOS stripped the crate's own temperature")


def test_eos_step1b_mask_never_moves_velocity():
    """Ruling §4 item 4: `cmask` is UNTOUCHED, so the velocity self-advection —
    and through it the pressure solve and the gas flow — is identical with and
    without the thermal mask. The T occlusion rides a SEPARATE, T-only mask."""
    h, w = 12, 16
    solid, vac, tsol, perm, furn = _eos_world(h, w)
    rng = np.random.default_rng(4242)
    T0 = _q32(rng.integers(-3 * FP_ONE, 900 * FP_ONE, size=(h, w)))
    wx0 = _q32(rng.integers(-5 * FP_ONE, 5 * FP_ONE, size=(h, w)))
    wy0 = _q32(rng.integers(-5 * FP_ONE, 5 * FP_ONE, size=(h, w)))
    wx0[solid] = 0
    wy0[solid] = 0
    a = (wx0.copy(), wy0.copy(), T0.copy())
    b = (wx0.copy(), wy0.copy(), T0.copy())
    bp.eos_sl_advect_ref(a[0], a[1], a[2], solid, vac, perm,
                         dt=1.0 / 24.0, n_sub=3, thermal_solid=tsol)
    bp.eos_sl_advect_ref(b[0], b[1], b[2], solid, vac, perm,
                         dt=1.0 / 24.0, n_sub=3)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (
        "the thermal mask changed the velocity field — cmask must be untouched")
    # P-E1: the second leg INVERTS. It used to read "vacuous: T did not change
    # either" — i.e. the mask HAD to change the T sample. The T sample is
    # retired (design SS2.1.1), so the mask must now change nothing at this
    # entry at all, temperature included.
    assert np.array_equal(a[2], b[2]) and np.array_equal(a[2], T0), (
        "the SL entry is u-only now — neither run may write temperature")


def test_eos_step4c_does_not_write_temperature_on_a_thermal_solid():
    """Ruling A1 / T-WRITE SITE 2/2: compression work is work done ON GAS BY
    COMPRESSION — an object does not compress, so step 4c may not touch a
    thermal_solid tile's T. The momentum kick (which writes u, never T) must be
    bit-identical with and without the mask."""
    h, w = 12, 16
    solid, vac, tsol, perm, furn = _eos_world(h, w)
    rng = np.random.default_rng(99)
    T0 = _q32(np.where(furn, 900 * FP_ONE, 120 * FP_ONE))
    wx0 = _q32(rng.integers(-4 * FP_ONE, 4 * FP_ONE, size=(h, w)))
    wy0 = _q32(rng.integers(-4 * FP_ONE, 4 * FP_ONE, size=(h, w)))
    wx0[solid] = 0
    wy0[solid] = 0
    p_new = _q32(rng.integers(-FP_ONE // 4, FP_ONE // 2, size=(h, w)))
    gas = _q32(rng.integers(0, FP_ONE, size=(3, h, w)))
    cons = np.array([True, True, False], dtype=bool)
    wabs = np.ascontiguousarray(np.zeros((h, w), dtype=np.float32))
    args = dict(dt=1.0 / 24.0, c_local_q=300 * FP_ONE, c_max=300.0, dx=1.0 / 3.0,
                adiabatic_index=1.4, absorb_strength=8.0, n_floor_solver=1e-3,
                t_min=-289.0, t_work_clamp=0.5, t_max_phys=16000.0,
                u_max=1000.0)   # trace_mass_scale arg RETIRED (P-T0)

    a = (wx0.copy(), wy0.copy(), T0.copy())
    b = (wx0.copy(), wy0.copy(), T0.copy())
    bp.eos_kick_compression_ref(a[0], a[1], a[2], p_new, gas, cons, solid, vac,
                                wabs, thermal_solid=tsol, **args)
    bp.eos_kick_compression_ref(b[0], b[1], b[2], p_new, gas, cons, solid, vac,
                                wabs, **args)
    assert np.array_equal(a[2][furn], T0[furn]), (
        "step 4c wrote temperature on a thermal_solid tile")
    assert not np.array_equal(b[2][furn], T0[furn]), (
        "control failed: the pre-patch path must do compression work there")
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]), (
        "the thermal mask changed the momentum kick — it must not")


@pytest.mark.parametrize("entry", ["sl", "kick"])
def test_eos_furniture_free_identity_at_the_replay_boundary(entry):
    """Gate (a), at the EOS boundary: where ``thermal_solid == solid`` (any
    furniture-free map) passing the mask must be BYTE-IDENTICAL to passing
    nothing — the nullable fallback is not a second code path."""
    h, w = 12, 16
    solid, vac, _tsol, perm, _furn = _eos_world(h, w, crate=((0, 0), (0, 0)))
    tsol = solid.copy()                        # furniture-free => identical
    assert np.array_equal(tsol, solid)
    rng = np.random.default_rng(7)
    T0 = _q32(rng.integers(-2 * FP_ONE, 800 * FP_ONE, size=(h, w)))
    wx0 = _q32(rng.integers(-5 * FP_ONE, 5 * FP_ONE, size=(h, w)))
    wy0 = _q32(rng.integers(-5 * FP_ONE, 5 * FP_ONE, size=(h, w)))
    wx0[solid] = 0
    wy0[solid] = 0
    if entry == "sl":
        a = (wx0.copy(), wy0.copy(), T0.copy())
        b = (wx0.copy(), wy0.copy(), T0.copy())
        da = bp.eos_sl_advect_ref(a[0], a[1], a[2], solid, vac, perm,
                                  dt=1.0 / 24.0, n_sub=3, thermal_solid=tsol)
        db = bp.eos_sl_advect_ref(b[0], b[1], b[2], solid, vac, perm,
                                  dt=1.0 / 24.0, n_sub=3)
        assert da == db
    else:
        p_new = _q32(rng.integers(-FP_ONE // 4, FP_ONE // 2, size=(h, w)))
        gas = _q32(rng.integers(0, FP_ONE, size=(3, h, w)))
        cons = np.array([True, True, False], dtype=bool)
        wabs = np.ascontiguousarray(np.zeros((h, w), dtype=np.float32))
        args = dict(dt=1.0 / 24.0, c_local_q=300 * FP_ONE, c_max=300.0,
                    dx=1.0 / 3.0, adiabatic_index=1.4, absorb_strength=8.0,
                    n_floor_solver=1e-3, t_min=-289.0, t_work_clamp=0.5,
                    t_max_phys=16000.0, u_max=1000.0)  # trace_mass_scale RETIRED (P-T0)
        a = (wx0.copy(), wy0.copy(), T0.copy())
        b = (wx0.copy(), wy0.copy(), T0.copy())
        da = bp.eos_kick_compression_ref(a[0], a[1], a[2], p_new, gas, cons,
                                         solid, vac, wabs, thermal_solid=tsol,
                                         **args)
        db = bp.eos_kick_compression_ref(b[0], b[1], b[2], p_new, gas, cons,
                                         solid, vac, wabs, **args)
        assert tuple(da) == tuple(db)
    for x, y in zip(a, b):
        assert np.array_equal(x, y)


def test_combustion_deposit_converts_via_heat_inv_shift_on_a_thermal_solid():
    """Ruling §2 site 3: a FURNITURE tile is an open, gas-holding burn site but
    thermally an OBJECT, so its aggregate deposit must convert through the tile's
    own ``heat_inv_shift`` (``deposit >> log2(thermal_mass)``) instead of the thin
    pore gas's N divisor. Pinned to the LSB against the object formula."""
    h = w = 9
    solid = np.zeros((h, w), dtype=bool)
    solid[0, :] = solid[-1, :] = solid[:, 0] = solid[:, -1] = True
    vac = np.zeros((h, w), dtype=bool)
    furn = np.zeros((h, w), dtype=bool)
    furn[4, 4] = True                       # ONE crate tile == the burn site
    tsol = solid | furn
    shift = _q32(np.where(tsol, 3, 0))      # furniture thermal_mass 8 -> >> 3

    flam = np.zeros((h, w), dtype=bool)
    flam[4, 3] = True                       # a burning WOOD source beside it
    solid[4, 3] = True
    tsol[4, 3] = True
    shift[4, 3] = 3

    o2, n2, soot = 0, 1, 2
    # quantize_scalar, NOT int(x * FP_ONE) (audit Patch A / A8, 2026-08-04).
    # int(0.21 * FP_ONE) TRUNCATES to 13762 where the suite's convention
    # (round-half-away) gives 13763. 13762 + 51773 = 65535, so this fixture was
    # one count short of ambient and silently violated the exact
    # `N_amb == FP_ONE` invariant that test_eos_p1_calibration.py:55-62 exists
    # to pin. Every other O2 fixture in the suite already uses quantize_scalar.
    # This is the "a divergent copy makes a GATE quietly wrong rather than red"
    # hazard in its concrete form.
    gas0 = np.stack([np.full((h, w), gas_fixed.quantize_scalar(0.21)),
                     np.full((h, w), gas_fixed.quantize_scalar(0.79)),
                     np.zeros((h, w))]).astype(np.int32)
    fire = _q32(np.where(flam, int(0.8 * FP_ONE), 0))
    wall_hp = _q32(np.where(flam, 30 * FP_ONE, 0))
    ign = _q32(np.where(flam, 280 * FP_ONE, 0))
    T0 = _q32(np.full((h, w), 400 * FP_ONE))

    solver = bp.CombustionSolver()
    out = {}
    for tag, mask in (("object", tsol), ("gas", None)):
        gas = np.ascontiguousarray(gas0.copy())
        T = _q32(T0.copy())
        whp = _q32(wall_hp.copy())
        solver.step(gas, o2, n2, soot, T, whp, fire, flam, solid, vac, ign,
                    dt=1.0 / 24.0, c_v=1.0, n_floor_heat=0.05,
                    thermal_solid=mask,
                    heat_inv_shift=(np.ascontiguousarray(shift)
                                    if mask is not None else None))
        out[tag] = (T, gas)

    burn = int(gas0[o2][4, 4]) - int(out["object"][1][o2][4, 4])
    assert burn > 0, "the crate tile must actually burn (vacuous otherwise)"
    # OBJECT path: dT == (burn*H_fuel) >> shift, exactly.
    h_fuel_q = int(round(float(solver.H_fuel) * FP_ONE))
    deposit = (burn * h_fuel_q) >> 16                    # mul_q16, truncating
    expect = int(T0[4, 4]) + (deposit >> 3)
    assert int(out["object"][0][4, 4]) == expect, (
        f"object deposit != deposit>>heat_inv_shift "
        f"(got {int(out['object'][0][4, 4])}, expected {expect})")
    # The GAS path (the pre-patch behaviour) divides by the thin pore N instead,
    # so it lands somewhere ELSE — the branch is load-bearing, not cosmetic.
    assert int(out["gas"][0][4, 4]) != expect
    # Same energy IN either way: the O2 drawn and the fuel paid are identical.
    assert np.array_equal(out["object"][1], out["gas"][1]), (
        "the deposit's CONVERSION moved; the gas bookkeeping must not")


def test_run_substeps_thermal_solid_is_nullable_and_solid_equivalent():
    """The plumbing contract: ``run_substeps(thermal_solid=None)`` is the legacy
    path, and on a furniture-free map it is byte-identical to passing the real
    mask (which equals ``solid`` there). Driven through the real engine entry."""
    from level_loader import LevelData
    from simulation.physics_runner import PhysicsRunner

    def _furniture_free_map():
        h = w = 16
        tm = np.full((h, w), MAT_HULL, dtype=np.int32)
        tm[1:h - 1, 1:w - 1] = MAT_AIR
        tm[5:9, 6] = MAT_WOOD                 # a thermal solid that IS flow-solid
        lvl = LevelData(name="peos_ff", version="2", path=Path("."),
                        tilemap=tm, tile_size_m=1.0 / 3.0, diffuse_path=Path("."))
        return GameMap(lvl)

    g1, g2 = _furniture_free_map(), _furniture_free_map()
    assert np.array_equal(g1.thermal_solid, g1.solid), \
        "the identity scenario must be furniture-free (addendum D4)"
    assert g1.thermal_solid.any()
    outs = []
    for g, mask in ((g1, None), (g2, g2.thermal_solid)):
        r = PhysicsRunner(bp)
        g.bind_physics_engine(r.engine)
        g.stamp_units([])
        g.temperature[g.solid] = 300 * FP_ONE
        r.eos.dx = float(g.tile_size_m)
        r.engine.run_substeps(
            g.wave_p, g.atmosphere, g.wind_x, g.wind_y, g.temperature,
            g.obstacles, g.solid, g.is_vacuum,
            g.dyn_permeability, g.dyn_wave_absorb,
            g.gas, g.gases.diffusion, g.gases.conservative,
            g.gases.decay, int(g.gases.name_to_id["inert_n2"]),
            1.0 / 24.0, thermal_solid=mask)
        outs.append((g.temperature.copy(), g.atmosphere.copy(),
                     g.wind_x.copy(), g.wind_y.copy(), g.gas.copy()))
    for a, b in zip(*outs):
        assert np.array_equal(a, b)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
