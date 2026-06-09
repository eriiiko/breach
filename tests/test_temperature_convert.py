"""Heat -> temperature conversion (engine/06 §1.2, proposal §1).

STEP A of the temperature substrate: the C++ ``TemperatureSolver`` turns the
per-tick Q16.16 ``heat`` deposit into the persistent Q16.16 ``temperature``
field on SOLID tiles via a pure arithmetic right shift:

    temperature[i] = sat_add( temperature[i], heat[i] >> heat_inv_shift[i] )

Verifies, on small synthetic grids (no renderer, no ray pass):
  - on a solid tile, temperature == heat >> log2(thermal_mass) for the tile's
    material (wood >>3, hull >>5 with the shipped thermal_mass values);
  - a solid ACCUMULATES across two ticks (saturating add);
  - an AIR tile stays EXACTLY 0 (conversion skips non-solids);
  - the saturating add PINS at INT32_MAX rather than wrapping negative;
  - determinism: same inputs -> bit-identical temperature.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_temperature_convert.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp

from simulation.materials import (
    MAT_AIR, MAT_HULL, MAT_WOOD,
    MaterialTable,
)

# Q16.16 scale — must match cpp/src/raycaster.h HEAT_SCALE and the temperature
# field (TEMP_SCALE == HEAT_SCALE).
HEAT_SCALE = 65536
INT32_MAX = 2**31 - 1

# Per-material log2(thermal_mass) from the shipped table (hull/steel 32 -> >>5,
# glass 16 -> >>4, wood/door 8 -> >>3, air 1 -> >>0). Read from the live config
# so the test tracks the table rather than a hardcoded copy.
_TBL = MaterialTable.from_config()
SHIFT_WOOD = int(_TBL.heat_inv_shift[MAT_WOOD])
SHIFT_HULL = int(_TBL.heat_inv_shift[MAT_HULL])


# NO_FACE sentinel — must match config [physics.thermal].NO_FACE and the
# materials table. These conversion-only tests use an all-NO_FACE face cache so
# the conduction pass (engine/06 §2) is a no-op and only the §1 conversion is
# exercised (conduction has its own test module).
NO_FACE = int(_TBL.no_face)


def _grid(material_ids):
    """Build (temperature, heat, heat_inv_shift, face_shift, solid, is_vacuum,
    atmosphere) for a 1-row grid whose tiles have the given material ids.
    temperature starts at 0; heat starts at 0; the shift/solid caches are derived
    from the material table. The face_shift cache is all-NO_FACE (conduction
    disabled — this module tests the conversion pass only). is_vacuum is all
    False and atmosphere is all 1.0 (sealed interior), so the cooling pass uses
    the interior COOL_SHIFT everywhere; cooling is exercised in its own module."""
    mats = np.asarray(material_ids, dtype=np.int8).reshape(1, -1)
    w = mats.shape[1]
    temperature = np.zeros((1, w), dtype=np.int32)
    heat = np.zeros((1, w), dtype=np.int32)
    shift = _TBL.heat_inv_shift[mats].astype(np.int32)
    # Solid mask: a tile is solid iff impermeable. For this synthetic grid we
    # use the table's permeability the same way GameMap._update_caches does.
    solid = (_TBL.permeability[mats] <= 0.0)
    face_shift = np.full((1, w, 4), NO_FACE, dtype=np.int32)
    is_vacuum = np.zeros((1, w), dtype=bool)
    atmosphere = np.ones((1, w), dtype=np.float32)
    return (temperature, np.ascontiguousarray(heat),
            np.ascontiguousarray(shift), np.ascontiguousarray(face_shift),
            np.ascontiguousarray(solid), np.ascontiguousarray(is_vacuum),
            np.ascontiguousarray(atmosphere))


def _solver():
    """A solver with conduction AND cooling disabled, so these tests exercise the
    §1 conversion pass in ISOLATION. Conduction is off via the all-NO_FACE face
    cache (from _grid); cooling is off by pinning both cool shifts huge, so the
    arithmetic right shift yields 0 for every test value (the dead-band swallows
    it) and `temperature` is left exactly at the conversion result. Cooling has
    its own module (test_temperature_cooling.py)."""
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.cool_shift = 31           # T >> 31 == 0 for all in-range values -> no cooling
    s.cool_shift_vacuum = 31
    return s


def _step(solver, temp, heat, shift, face_shift, solid, is_vacuum, atmosphere):
    """solver.step with the cooling fields threaded (signature is the same for
    every caller, so wrap it once)."""
    solver.step(temp, heat, shift, face_shift, solid, is_vacuum, atmosphere)


def test_shipped_shifts():
    # Guard the table values STEP A is anchored to: wood >>3, hull >>5.
    assert SHIFT_WOOD == 3, f"wood thermal_mass should be 8 (>>3), got >>{SHIFT_WOOD}"
    assert SHIFT_HULL == 5, f"hull thermal_mass should be 32 (>>5), got >>{SHIFT_HULL}"


def test_solid_conversion_equals_heat_shifted():
    # One solid wood tile and one solid hull tile. After one tick, temperature
    # is exactly heat >> shift for each material.
    temp, heat, shift, face_shift, solid, vac, atm = _grid([MAT_WOOD, MAT_HULL])
    heat[0, 0] = 800 * HEAT_SCALE     # a big, clean Q16.16 heat value (wood tile)
    heat[0, 1] = 800 * HEAT_SCALE     # same deposit on the hull tile
    solver = _solver()
    _step(solver, temp, heat, shift, face_shift, solid, vac, atm)
    assert temp[0, 0] == heat[0, 0] >> SHIFT_WOOD, "wood: temp != heat >> 3"
    assert temp[0, 1] == heat[0, 1] >> SHIFT_HULL, "hull: temp != heat >> 5"
    # Sanity: metal soaks more energy/degree -> lower temperature for equal heat.
    assert temp[0, 1] < temp[0, 0]


def test_air_tile_stays_exactly_zero():
    # An air (non-solid) tile is skipped by the conversion: even with a huge heat
    # deposit, its temperature stays bit-exactly 0.
    temp, heat, shift, face_shift, solid, vac, atm = _grid([MAT_AIR])
    assert not solid[0, 0], "sanity: air must be non-solid"
    # Air's shift is 0 (>>0), so if it were wrongly converted the full deposit
    # would land. Use the int32 ceiling to make any leak unmistakable.
    heat[0, 0] = INT32_MAX
    solver = _solver()
    _step(solver, temp, heat, shift, face_shift, solid, vac, atm)
    assert temp[0, 0] == 0, f"air tile gained temperature: {temp[0, 0]}"


def test_accumulates_over_two_ticks():
    # The conversion is a SATURATING ADD onto the persistent field: a solid tile
    # under a steady deposit accumulates across ticks.
    temp, heat, shift, face_shift, solid, vac, atm = _grid([MAT_WOOD])
    heat[0, 0] = 100 * HEAT_SCALE
    solver = _solver()
    per_tick = heat[0, 0] >> SHIFT_WOOD
    _step(solver, temp, heat, shift, face_shift, solid, vac, atm)   # tick 1 (heat NOT cleared)
    assert temp[0, 0] == per_tick
    _step(solver, temp, heat, shift, face_shift, solid, vac, atm)   # tick 2: same deposit
    assert temp[0, 0] == 2 * per_tick, "two ticks must accumulate"


def test_saturating_add_pins_at_int32_max():
    # Pre-load temperature near the ceiling; a further deposit must clamp at
    # INT32_MAX, never wrap negative.
    temp, heat, shift, face_shift, solid, vac, atm = _grid([MAT_WOOD])
    temp[0, 0] = INT32_MAX - 10
    heat[0, 0] = 1000 * HEAT_SCALE        # deposit >> 3 is far more than 10
    solver = _solver()
    _step(solver, temp, heat, shift, face_shift, solid, vac, atm)
    assert temp[0, 0] == INT32_MAX, f"saturating add did not pin: {temp[0, 0]}"
    assert temp[0, 0] >= 0, "must never wrap negative"


def test_deterministic_same_inputs_bit_identical():
    # Same seed/inputs -> bit-identical temperature buffers across two runs.
    rng = np.random.default_rng(1234)
    w = 64
    mats = rng.integers(0, 6, size=w).astype(np.int8).reshape(1, w)

    def run():
        temp = np.zeros((1, w), dtype=np.int32)
        heat = (rng_local := np.random.default_rng(99)).integers(
            0, 500 * HEAT_SCALE, size=w, dtype=np.int64).astype(np.int32).reshape(1, w)
        shift = np.ascontiguousarray(_TBL.heat_inv_shift[mats].astype(np.int32))
        solid = np.ascontiguousarray(_TBL.permeability[mats] <= 0.0)
        heat = np.ascontiguousarray(heat)
        face_shift = np.full((1, w, 4), NO_FACE, dtype=np.int32)
        is_vacuum = np.ascontiguousarray(np.zeros((1, w), dtype=bool))
        atmosphere = np.ascontiguousarray(np.ones((1, w), dtype=np.float32))
        solver = _solver()
        _step(solver, temp, heat, shift, np.ascontiguousarray(face_shift),
              solid, is_vacuum, atmosphere)
        return temp

    a = run()
    b = run()
    assert np.array_equal(a, b), "conversion is not deterministic"
