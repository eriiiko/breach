"""The cool-shift axis — the ambient-decay shift is per MATERIAL (2026-07-30).

THE DEFECT. ``COOL_SHIFT`` was ONE global integer in ``[physics.thermal]``
governing the Newtonian ambient decay ``T -= T >> COOL_SHIFT`` on every thermal
solid, plus ``COOL_SHIFT_VACUUM`` for tiles whose 4-neighbourhood touches
vacuum. The thermal-mass arc then made furniture a thermal solid whose ONLY loss
channel is that decay (furniture ``conductivity = 0`` -> NO_FACE both ways -> no
conduction face at all). At 24 Hz, ``COOL_SHIFT = 5`` is an e-fold of
``2^5/24 == 1.3 s`` — right for thin hull plate, absurd for a wooden crate,
while a value slow enough for wood (12 -> 171 s) is absurd for plate. One global
number cannot serve both, so the tuning loop was forced onto a compromise wrong
for most materials.

THE FIX (this patch). A per-material ``cool_shift`` column, projected to the
per-tile ``GameMap.cool_shift`` grid on the SAME single seam as
``heat_inv_shift`` / ``thermal_solid``, and read by the cooling pass. It is the
LOSS-side twin of the ``thermal_mass`` (gain-side) axis.

THE VACUUM DECISION. The space-exposed rate is NOT a second per-material column.
It is the same per-material shift with a single GLOBAL OFFSET applied:

    exposed -> max(SHIFT_MIN, cool_shift - (COOL_SHIFT - COOL_SHIFT_VACUUM))

"space sheds 4x faster" is a property of the BOUNDARY, not of the material, so
it stays one rule and every material keeps exactly ONE dial. With every row
seeded at the old global 5 this reproduces the shipped interior-5 / exposed-3
pair bit-exactly — the patch's zero-tolerance gate (a).

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_cool_shift_axis.py -q
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
    _COOL_SHIFT_MAX,
)

FP_ONE = 1 << 16
_THERMAL = CFG.physics.thermal
_TBL = MaterialTable.from_config()
NO_FACE = int(_TBL.no_face)
COOL_SHIFT = int(getattr(_THERMAL, "COOL_SHIFT", 5))
COOL_SHIFT_VACUUM = int(getattr(_THERMAL, "COOL_SHIFT_VACUUM", 3))
SHIFT_MIN = int(getattr(_THERMAL, "SHIFT_MIN", 2))
VAC_OFFSET = COOL_SHIFT - COOL_SHIFT_VACUUM
TICKS_PER_SECOND = float(CFG.clock.ticks_per_second)


def _expected_exposed(base):
    """The vacuum-offset rule, in one place, as the source of truth for the
    tests below (deliberately re-derived from config, not copied from C++)."""
    return max(SHIFT_MIN, base - VAC_OFFSET)


# ---------------------------------------------------------------------------
# 1. The material column
# ---------------------------------------------------------------------------
def test_every_material_carries_the_column_seeded_at_the_old_global():
    """Gate (a)'s precondition, asserted in the suite: every material that
    EXISTED when this axis landed ships at the value the single global used
    to impose, so the engine is byte-identical on arrival. Moving any of
    THOSE rows is a deliberate, HUMAN-TESTED feel change — and this test is
    what will notice.

    P-F4a's kindling is a brand-new bench-only material row (no shipped
    level paints it, so nothing EXISTING changes feel); its cool_shift=9 is
    a locked value from its own spec
    (docs/fire_realism_design_2026-08-01.md v5.2), not a re-tune of a
    material this axis already shipped — excluded from the seeded-default
    check by name, not by weakening the check for everyone else.
    """
    tbl = MaterialTable.from_config(CFG)
    assert tbl.cool_shift.dtype == np.int32
    assert tbl.cool_shift.shape == (tbl.n,)
    NEW_MATERIALS_WITH_LOCKED_VALUES = {"kindling"}
    for name, cs in zip(tbl.names, tbl.cool_shift.tolist()):
        if name in NEW_MATERIALS_WITH_LOCKED_VALUES:
            continue
        assert cs == COOL_SHIFT, (
            f"materials.{name}.cool_shift is {cs}, expected the seeded "
            f"{COOL_SHIFT}. If this is an intended re-tune, gate (a) "
            f"byte-identity no longer holds and this test must be updated "
            f"together with a HUMAN-TEST play session.")


def test_the_globals_are_kept_and_still_have_jobs():
    """COOL_SHIFT is the omitted-column DEFAULT and the solver's own fallback;
    the PAIR defines the vacuum offset. Neither was removed."""
    assert COOL_SHIFT == 5 and COOL_SHIFT_VACUUM == 3
    assert VAC_OFFSET == 2
    assert (1 << COOL_SHIFT) // (1 << COOL_SHIFT_VACUUM) == 4   # the shipped 4x


def test_efold_seconds_are_the_documented_powers_of_two():
    """The dial's user-facing meaning: e-fold ~ 2^shift / tick_rate seconds."""
    assert TICKS_PER_SECOND == 24.0
    assert abs((1 << 5) / TICKS_PER_SECOND - 1.333) < 0.01     # crate today
    assert abs((1 << 12) / TICKS_PER_SECOND - 170.7) < 0.1     # crate at 12


# ---------------------------------------------------------------------------
# 2. Loader validation
# ---------------------------------------------------------------------------
def _row(**over):
    base = dict(hp=10.0, flammable=False, mobility=1000, conductivity=1.0,
                thermal_mass=8, ignition_temp=0.0, heat_atten=0.0,
                wave_reflect=0.0, wave_absorb=0.0, blast_resist=0.0,
                light_atten=[0.0, 0.0, 0.0])
    base.update(over)
    return base


def _table(cool_shift_by_name=None, thermal_cfg=None):
    cool_shift_by_name = cool_shift_by_name or {}
    cfg = {}
    for name in MATERIAL_NAMES.values():
        over = {}
        if name in cool_shift_by_name:
            over["cool_shift"] = cool_shift_by_name[name]
        cfg[name] = _row(**over)
    return MaterialTable(cfg, thermal_cfg)


def test_column_is_optional_and_defaults_to_the_global():
    """A row (or a whole dict-built test table) that omits the column inherits
    ``[physics.thermal] COOL_SHIFT`` — exactly the pre-axis behaviour, which is
    why the global is kept rather than retired."""
    tbl = _table()                                   # nobody sets the column
    assert list(tbl.cool_shift) == [5] * tbl.n       # _THERMAL_DEFAULTS
    tbl2 = _table(thermal_cfg={"COOL_SHIFT": 9})
    assert list(tbl2.cool_shift) == [9] * tbl2.n


@pytest.mark.parametrize("good", [SHIFT_MIN, 3, 5, 8, 12, 14, _COOL_SHIFT_MAX])
def test_loader_accepts_the_legal_range(good):
    tbl = _table({"furniture": good})
    assert int(tbl.cool_shift[MAT_FURNITURE]) == good


def test_loader_rejects_shift_zero_the_instant_total_wipe():
    """Shift 0 means ``T -= T``: the whole temperature field is annihilated
    every tick. It is not a fast dial, it is a broken one."""
    with pytest.raises(ValueError, match="cool_shift"):
        _table({"furniture": 0})


@pytest.mark.parametrize("bad", [-1, 0, 1])
def test_loader_rejects_below_the_floor(bad):
    with pytest.raises(ValueError, match="SHIFT_MIN"):
        _table({"wood": bad})


@pytest.mark.parametrize("bad", [_COOL_SHIFT_MAX + 1, 31, 64, 1000])
def test_loader_rejects_above_the_ceiling(bad):
    with pytest.raises(ValueError, match="cool_shift"):
        _table({"hull": bad})


@pytest.mark.parametrize("bad", [5.5, 4.25, "5", True])
def test_loader_rejects_non_integers(bad):
    """A shift count is consumed by a C++ arithmetic right shift; a float here
    would be a silent truncation, and determinism forbids floats in the sim
    path anyway. (``None`` is deliberately NOT in this list: for an OPTIONAL
    column it is indistinguishable from "omitted" and means the global default
    — TOML cannot express a null value at all.)"""
    with pytest.raises(ValueError, match="cool_shift"):
        _table({"steel": bad})


def test_loader_accepts_an_integer_valued_float():
    """TOML/dict authors sometimes write 8.0; that is unambiguous, so accept it
    and freeze the int — only a FRACTIONAL value is a mistake."""
    tbl = _table({"glass": 8.0})
    assert int(tbl.cool_shift[5]) == 8


# ---------------------------------------------------------------------------
# 3. The derived per-tile GRID (ONE build seam, ONE patch seam)
# ---------------------------------------------------------------------------
def test_grid_is_the_table_column_projected():
    g = GameMap(load_level("playground"))
    assert g.cool_shift.dtype == np.int32
    assert g.cool_shift.shape == g.solid.shape
    assert np.array_equal(g.cool_shift, g.materials.cool_shift[g.material])


def test_grid_is_built_in_the_same_function_as_heat_inv_shift():
    """D3: ONE seam. ``_update_caches`` rebuilds every table-derived cache, so
    a table swap must move cool_shift exactly as it moves heat_inv_shift."""
    g = GameMap(load_level("playground"))
    tbl = g.materials
    tbl.cool_shift = tbl.cool_shift.copy()
    tbl.cool_shift[MAT_WOOD] = 11
    tbl.heat_inv_shift = tbl.heat_inv_shift.copy()
    tbl.heat_inv_shift[MAT_WOOD] = 4
    g._update_caches()
    wood = (g.material == MAT_WOOD)
    assert wood.any()
    assert (g.cool_shift[wood] == 11).all()
    assert (g.heat_inv_shift[wood] == 4).all()


def test_on_tile_changed_patches_the_grid_both_ways_and_is_O1():
    g = GameMap(load_level("unhcr_vessel"))
    tbl = g.materials
    tbl.cool_shift = tbl.cool_shift.copy()
    tbl.cool_shift[MAT_FURNITURE] = 12          # a slow, wood-like crate
    g._update_caches()

    ys, xs = np.where((g.material == MAT_AIR) & ~g.is_vacuum)
    y, x = int(ys[len(ys) // 2]), int(xs[len(ys) // 2])
    before = g.cool_shift.copy()

    g.material[y, x] = MAT_FURNITURE
    g.on_tile_changed(y, x)
    assert int(g.cool_shift[y, x]) == 12
    touched = before != g.cool_shift
    assert touched.sum() == 1 and touched[y, x], "patch must be O(1), one tile"

    g.material[y, x] = MAT_AIR
    g.on_tile_changed(y, x)
    assert np.array_equal(g.cool_shift, before)


def test_burning_out_a_crate_patches_the_grid():
    g = GameMap(load_level("unhcr_vessel"))
    ys, xs = np.where((g.material == MAT_AIR) & ~g.is_vacuum)
    y, x = int(ys[len(ys) // 2]), int(xs[len(ys) // 2])
    g.material[y, x] = MAT_FURNITURE
    g.on_tile_changed(y, x)
    g.destroy_wall(y, x)
    assert int(g.material[y, x]) == MAT_AIR
    assert int(g.cool_shift[y, x]) == int(g.materials.cool_shift[MAT_AIR])


def test_grid_joins_the_resident_mask_set():
    """The device-side seam: a resident buffer + the __setattr__ stale-pointer
    guard, exactly like heat_inv_shift / thermal_solid."""
    assert "cool_shift" in GameMap._RESIDENT_MASKS
    assert "cool_shift" in GameMap._RESIDENT_FIELD_NAMES


# ---------------------------------------------------------------------------
# 4. The SOLVER — the per-tile shift, and the vacuum-offset rule
# ---------------------------------------------------------------------------
def _solver():
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.cool_shift = COOL_SHIFT
    s.cool_shift_vacuum = COOL_SHIFT_VACUUM
    s.cool_shift_floor = SHIFT_MIN
    s.o2_vacuum_thresh = float(getattr(_THERMAL, "o2_vacuum_thresh", 0.3))
    s.c_v = float(getattr(_THERMAL, "c_v", 1.0))
    s.n_floor_heat = float(getattr(_THERMAL, "n_floor_heat", 0.05))
    return s


def _grid(h=5, w=5):
    """A tiny all-thermal-solid slab with conduction disabled (all-NO_FACE) and
    no vacuum, so Pass 3 acts in isolation on each cell."""
    return dict(
        temperature=np.zeros((h, w), dtype=np.int32),
        heat=np.zeros((h, w), dtype=np.int32),
        heat_inv_shift=np.full((h, w), 3, dtype=np.int32),
        face_shift=np.full((h, w, 4), NO_FACE, dtype=np.int32),
        solid=np.ones((h, w), dtype=bool),
        is_vacuum=np.zeros((h, w), dtype=bool),
        atmosphere=np.full((h, w), FP_ONE, dtype=np.int32),
    )


def _cool_once(shift_grid=None, is_vacuum=None, t0=1000 * FP_ONE, h=5, w=5):
    s = _solver()
    gr = _grid(h, w)
    if is_vacuum is not None:
        gr["is_vacuum"] = is_vacuum
    gr["temperature"][:] = t0
    kw = {}
    if shift_grid is not None:
        kw["cool_shift_grid"] = shift_grid
    s.step(gr["temperature"], gr["heat"], gr["heat_inv_shift"],
           gr["face_shift"], gr["solid"], gr["is_vacuum"], gr["atmosphere"],
           thermal_solid=gr["solid"], **kw)
    return gr["temperature"]


def test_uniform_grid_at_the_seeded_value_equals_the_null_fallback():
    """Gate (a) in one assertion at the solver boundary: a per-tile grid
    uniformly == the global is BIT-IDENTICAL to passing no grid at all."""
    t0 = 1234 * FP_ONE
    without = _cool_once(None, t0=t0)
    with_grid = _cool_once(np.full((5, 5), COOL_SHIFT, dtype=np.int32), t0=t0)
    assert np.array_equal(without, with_grid)
    assert int(without[2, 2]) == t0 - (t0 >> COOL_SHIFT)


def test_per_tile_shift_is_honoured_cell_by_cell():
    t0 = 1 << 24
    shifts = np.array([[2, 3, 4, 5, 6],
                       [7, 8, 9, 10, 11],
                       [12, 13, 14, 15, 16],
                       [17, 18, 19, 20, 2],
                       [5, 5, 5, 5, 5]], dtype=np.int32)
    out = _cool_once(shifts, t0=t0)
    for y in range(5):
        for x in range(5):
            s = int(shifts[y, x])
            assert int(out[y, x]) == t0 - (t0 >> s), (y, x, s)


def test_negative_temperatures_still_round_toward_zero_per_tile():
    """The pinned symmetric shift (``x<0 ? -((-x)>>s) : x>>s``) must survive the
    per-tile lookup — it is the determinism contract, not an optimisation."""
    t0 = -(1 << 22)
    shifts = np.full((5, 5), 7, dtype=np.int32)
    out = _cool_once(shifts, t0=t0)
    assert int(out[2, 2]) == t0 + ((-t0) >> 7)


# --- the vacuum-offset rule ------------------------------------------------
def _vacuum_column(h=5, w=5):
    """A vacuum cell in column 0, so every tile in column 1 is 4-adjacent to
    vacuum (space-exposed) and columns >= 2 are not."""
    v = np.zeros((h, w), dtype=bool)
    v[:, 0] = True
    return v


def test_vacuum_offset_reproduces_the_old_pair_at_the_seeded_value():
    """THE decision, gated: with every material at 5, an exposed tile cools at
    5 - (5-3) == 3 == the old COOL_SHIFT_VACUUM, bit-exactly."""
    t0 = 1000 * FP_ONE
    out = _cool_once(np.full((5, 5), COOL_SHIFT, dtype=np.int32),
                     is_vacuum=_vacuum_column(), t0=t0)
    assert int(out[2, 1]) == t0 - (t0 >> COOL_SHIFT_VACUUM)   # exposed
    assert int(out[2, 3]) == t0 - (t0 >> COOL_SHIFT)          # interior


@pytest.mark.parametrize("base", [SHIFT_MIN, 3, 4, 5, 8, 12, 16, _COOL_SHIFT_MAX])
def test_vacuum_shift_is_the_base_minus_the_global_offset_floored(base):
    t0 = 1 << 26
    out = _cool_once(np.full((5, 5), base, dtype=np.int32),
                     is_vacuum=_vacuum_column(), t0=t0)
    exposed = _expected_exposed(base)
    assert int(out[2, 1]) == t0 - (t0 >> exposed), f"exposed base={base}"
    assert int(out[2, 3]) == t0 - (t0 >> base), f"interior base={base}"


@pytest.mark.parametrize("base", [SHIFT_MIN, SHIFT_MIN + 1])
def test_the_floor_binds_and_never_lets_the_exposed_shift_reach_zero(base):
    """Load-bearing: without the clamp a material legally sitting at the floor
    would derive an exposed shift of ``0`` == ``T -= T`` — an instant total wipe
    of a space-facing tile. The clamp is why the loader's floor is safe."""
    t0 = 1000 * FP_ONE
    out = _cool_once(np.full((5, 5), base, dtype=np.int32),
                     is_vacuum=_vacuum_column(), t0=t0)
    assert base - VAC_OFFSET <= SHIFT_MIN, "this case must actually clamp"
    assert int(out[2, 1]) == t0 - (t0 >> SHIFT_MIN)
    assert int(out[2, 1]) != 0


def test_a_vacuum_exposed_tile_keeps_ONE_dial_per_material():
    """Two materials, two shifts, ONE offset rule: the exposed/interior ratio
    is the same 4x for both — the reason the vacuum rate is not a second
    column."""
    t0 = 1 << 26
    shifts = np.full((5, 5), 6, dtype=np.int32)
    shifts[:, 1] = 12                     # a slow material, in the exposed column
    shifts[:, 2] = 12                     # ...and in the interior
    out = _cool_once(shifts, is_vacuum=_vacuum_column(), t0=t0)
    loss_exposed = t0 - int(out[2, 1])
    loss_interior = t0 - int(out[2, 2])
    assert loss_exposed == t0 >> (12 - VAC_OFFSET)
    assert loss_interior == t0 >> 12
    assert loss_exposed == 4 * loss_interior


# ---------------------------------------------------------------------------
# 5. THE POINT: the dial actually changes how long an object stays hot
# ---------------------------------------------------------------------------
def _decay_ticks_to_half(shift, t0=1000 * FP_ONE, limit=40000):
    """Drive the real solver until an isolated thermal solid drops below t0/2,
    with conduction disabled — i.e. measure the dial's actual half-life."""
    s = _solver()
    gr = _grid(3, 3)
    gr["temperature"][1, 1] = t0
    grid = np.full((3, 3), shift, dtype=np.int32)
    for n in range(1, limit + 1):
        s.step(gr["temperature"], gr["heat"], gr["heat_inv_shift"],
               gr["face_shift"], gr["solid"], gr["is_vacuum"],
               gr["atmosphere"], thermal_solid=gr["solid"],
               cool_shift_grid=grid)
        if int(gr["temperature"][1, 1]) <= t0 // 2:
            return n
    raise AssertionError(f"shift {shift} did not halve within {limit} ticks")


def test_two_materials_with_different_cool_shift_decay_at_different_rates():
    """THE dial test. A crate at 5 vs the same crate at 12: the half-life
    scales with 2^shift, so the e-fold goes from ~1.3 s to ~171 s at 24 Hz."""
    n5 = _decay_ticks_to_half(5)
    n12 = _decay_ticks_to_half(12)
    assert n5 < n12
    # ln2 * 2^shift ticks, to within a tick of quantization.
    assert abs(n5 - 0.693 * (1 << 5)) <= 2, n5
    assert abs(n12 - 0.693 * (1 << 12)) <= 8, n12
    # ...and the ratio is the power of two, not something incidental.
    assert abs(n12 / n5 - (1 << 7)) < 4.0
    # Reported in seconds, the numbers Erik tunes against.
    assert abs(n5 / TICKS_PER_SECOND - 0.92) < 0.05      # ~0.9 s half-life
    assert abs(n12 / TICKS_PER_SECOND - 118.3) < 1.0     # ~2 minutes


def test_two_materials_in_ONE_grid_diverge_in_ONE_step():
    """The same assertion inside a single solver call: adjacent cells with
    different per-material shifts must not be able to share a rate."""
    t0 = 1 << 26
    shifts = np.full((5, 5), 5, dtype=np.int32)
    shifts[2, 2] = 12                    # a slow crate among fast plate
    out = _cool_once(shifts, t0=t0)
    assert int(out[2, 2]) == t0 - (t0 >> 12)
    assert int(out[2, 1]) == t0 - (t0 >> 5)
    assert int(out[2, 2]) > int(out[2, 1]), "the slow tile must stay hotter"


def test_a_crate_grid_from_config_is_uniform_today_but_addressable():
    """The shipped state: one value everywhere (gate (a)), but the grid really
    is keyed by material — flip furniture's row and only crates move."""
    g = GameMap(load_level("playground"))
    assert (g.cool_shift == COOL_SHIFT).all()
    tbl = g.materials
    tbl.cool_shift = tbl.cool_shift.copy()
    tbl.cool_shift[MAT_FURNITURE] = 12
    g._update_caches()
    furn = (g.material == MAT_FURNITURE)
    assert furn.any()
    assert (g.cool_shift[furn] == 12).all()
    assert (g.cool_shift[~furn] == COOL_SHIFT).all()
