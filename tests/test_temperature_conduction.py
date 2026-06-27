"""Conduction relaxation (engine/06 §2, proposal §2) — STEP B.

After the §1 heat -> temperature conversion, the C++ ``TemperatureSolver`` runs
ONE gather-stencil conduction relaxation pass (double-buffered) that spreads
temperature along the harmonic-mean face shifts:

    acc = Σ_{dir∈N,S,E,W}  (temp[n] - temp[i]) >> face_shift[i][dir]
    temp_new[i] = temp[i] + acc        (NO_FACE faces skipped)

Verifies, on small synthetic grids (no renderer, no ray pass):
  - a hot solid tile spreads heat to conductive solid neighbours over ticks;
  - METAL (hull, low face shift) spreads FASTER than WOOD (high face shift);
  - the DISCRETE MAXIMUM PRINCIPLE holds: from an arbitrary solid temperature
    field, over many ticks the global max never increases, the global min never
    decreases, and no value ever exceeds the initial max (convex update §2.6);
  - AIR tiles stay BIT-EXACTLY 0 (every air face is NO_FACE -> structural no-op);
  - EQUAL-temperature neighbours produce ZERO change (the difference is shifted,
    so equal neighbours -> exactly 0, no drift);
  - a WOOD<->METAL face conducts at ~the WOOD (slow) rate (harmonic mean);
  - determinism: same field -> bit-identical after N ticks.

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_temperature_conduction.py -q
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

_TBL = MaterialTable.from_config()
NO_FACE = int(_TBL.no_face)

# Fixed face-direction order N,S,E,W (must match the C++ DIR_* / DY,DX and the
# GameMap bake).
_FACE_DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))


def _build_caches(material_grid):
    """Build (heat_inv_shift, face_shift, solid) for a 2D material-id grid,
    exactly as GameMap._update_caches / _rebuild_face_shift would (so the test
    exercises the same cache layout the engine feeds the solver).

    Returns C-contiguous arrays ready for the solver.
    """
    m = np.asarray(material_grid, dtype=np.int8)
    h, w = m.shape
    shift = _TBL.heat_inv_shift[m].astype(np.int32)
    solid = (_TBL.permeability[m] <= 0.0)
    face_tbl = _TBL.face_shift_table

    face = np.full((h, w, 4), NO_FACE, dtype=np.int32)
    for d, (dy, dx) in enumerate(_FACE_DIRS):
        ty0, ty1 = max(0, -dy), h - max(0, dy)
        tx0, tx1 = max(0, -dx), w - max(0, dx)
        mi = m[ty0:ty1, tx0:tx1]
        mn = m[ty0 + dy:ty1 + dy, tx0 + dx:tx1 + dx]
        face[ty0:ty1, tx0:tx1, d] = face_tbl[mi, mn]

    return (np.ascontiguousarray(shift),
            np.ascontiguousarray(face),
            np.ascontiguousarray(solid))


def _solver():
    # Cooling disabled (both shifts pinned huge -> T >> 31 == 0 for every test
    # value, swallowed by the dead-band) so this module exercises the §2
    # CONDUCTION pass in ISOLATION, the way it uses zero heat to isolate it from
    # the §1 conversion. Ambient cooling has its own module
    # (test_temperature_cooling.py).
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.cool_shift = 31
    s.cool_shift_vacuum = 31
    return s


def _zero_heat(shape):
    return np.ascontiguousarray(np.zeros(shape, dtype=np.int32))


def _cooling_fields(shape):
    """Sealed-interior vacuum/atmosphere fields for the cooling pass. Cooling is
    disabled in this module's _solver (shift 31), so the values are immaterial —
    but valid arrays must still be passed."""
    is_vacuum = np.ascontiguousarray(np.zeros(shape, dtype=bool))
    # S3c: atmosphere is int32 Q16.16 (1.0 real == FP_ONE == 65536 counts).
    atmosphere = np.ascontiguousarray(np.full(shape, 1 << 16, dtype=np.int32))
    return is_vacuum, atmosphere


def _run(temp, shift, face, solid, n_ticks, heat=None):
    """Run n_ticks of solver.step with NO fresh heat (conduction-only), mutating
    temp in place."""
    solver = _solver()
    if heat is None:
        heat = _zero_heat(temp.shape)
    is_vacuum, atmosphere = _cooling_fields(temp.shape)
    for _ in range(n_ticks):
        solver.step(temp, heat, shift, face, solid, is_vacuum, atmosphere)
    return temp


# Per-material face self-shift (homogeneous block), for reference in assertions.
SHIFT_WOOD = int(_TBL.face_shift_table[MAT_WOOD, MAT_WOOD])   # 8
SHIFT_HULL = int(_TBL.face_shift_table[MAT_HULL, MAT_HULL])   # 2
SHIFT_WOOD_HULL = int(_TBL.face_shift_table[MAT_WOOD, MAT_HULL])


def test_face_table_anchor_values():
    # Guard the load-time table STEP B is anchored to (engine/06 §2.4–§2.5).
    assert SHIFT_HULL == 2, f"hull-hull face should be shift 2, got {SHIFT_HULL}"
    assert SHIFT_WOOD == 8, f"wood-wood face should be shift 8, got {SHIFT_WOOD}"
    # Wood<->metal conducts at ~the WOOD (slow) rate, NOT the metal rate
    # (harmonic mean): its shift sits near wood, far from hull.
    assert SHIFT_WOOD_HULL >= SHIFT_WOOD - 1, "wood<->hull must be ~wood-slow"
    assert SHIFT_WOOD_HULL > SHIFT_HULL + 2, "wood<->hull must NOT be metal-fast"
    # Symmetric table -> symmetric flux.
    assert (_TBL.face_shift_table == _TBL.face_shift_table.T).all()


def test_hot_tile_spreads_to_neighbours():
    # A 1x5 hull strip, hot in the centre. Conduction must warm the immediate
    # neighbours over a few ticks while the hot centre cools toward them.
    mats = np.full((1, 5), MAT_HULL, dtype=np.int8)
    shift, face, solid = _build_caches(mats)
    temp = np.zeros((1, 5), dtype=np.int32)
    temp[0, 2] = 1 << 20          # hot centre (Q16.16)
    centre0 = int(temp[0, 2])
    _run(temp, shift, face, solid, 8)
    assert temp[0, 1] > 0 and temp[0, 3] > 0, "heat did not reach neighbours"
    assert temp[0, 2] < centre0, "hot centre did not cool toward neighbours"
    # Symmetric layout -> symmetric spread.
    assert temp[0, 1] == temp[0, 3]
    assert temp[0, 0] == temp[0, 4]


def test_metal_spreads_faster_than_wood():
    # Two identical strips (hot centre), one hull one wood. After the same
    # number of ticks the hull neighbour is hotter (lower face shift = faster).
    def strip(mat):
        mats = np.full((1, 5), mat, dtype=np.int8)
        shift, face, solid = _build_caches(mats)
        temp = np.zeros((1, 5), dtype=np.int32)
        temp[0, 2] = 1 << 24
        _run(temp, shift, face, solid, 4)
        return temp

    hull = strip(MAT_HULL)
    wood = strip(MAT_WOOD)
    assert hull[0, 1] > wood[0, 1], (
        f"metal must spread faster: hull nbr {hull[0,1]} <= wood nbr {wood[0,1]}"
    )
    # And the hull centre has shed more (cooled further) than the wood centre.
    assert hull[0, 2] < wood[0, 2], "metal centre should cool faster than wood"


def test_air_tiles_stay_bit_exactly_zero():
    # A hull tile flanked by AIR. The hull may hold heat, but the air tiles have
    # every face NO_FACE (kappa==0) -> they never gain a single count.
    mats = np.array([[MAT_AIR, MAT_HULL, MAT_AIR]], dtype=np.int8)
    shift, face, solid = _build_caches(mats)
    temp = np.zeros((1, 3), dtype=np.int32)
    temp[0, 1] = 1 << 24          # hot hull, air on both sides
    _run(temp, shift, face, solid, 50)
    assert temp[0, 0] == 0, f"air gained temperature: {temp[0, 0]}"
    assert temp[0, 2] == 0, f"air gained temperature: {temp[0, 2]}"
    # The hull, isolated by air faces (all NO_FACE), conducts to nothing -> it
    # is itself unchanged (Σr == 0).
    assert temp[0, 1] == (1 << 24), "isolated hull should be a no-op"


def test_equal_neighbours_zero_change():
    # A uniform hull block at a constant nonzero temperature: every difference
    # is 0, so (T_n - T_i) >> s == 0 on every face -> NO drift, exactly stable.
    mats = np.full((4, 4), MAT_HULL, dtype=np.int8)
    shift, face, solid = _build_caches(mats)
    temp = np.full((4, 4), 12345 << 4, dtype=np.int32)
    before = temp.copy()
    _run(temp, shift, face, solid, 100)
    assert np.array_equal(temp, before), "equal-temp field drifted (should be exact)"


def test_discrete_maximum_principle():
    # From an arbitrary solid (hull) temperature field, the convex update must
    # never create a new extremum: global max non-increasing, global min
    # non-decreasing, no value above the initial max, none below the initial min.
    rng = np.random.default_rng(20260609)
    h, w = 12, 12
    mats = np.full((h, w), MAT_HULL, dtype=np.int8)
    shift, face, solid = _build_caches(mats)
    temp = rng.integers(-(1 << 22), 1 << 22, size=(h, w), dtype=np.int64).astype(np.int32)
    temp = np.ascontiguousarray(temp)

    init_max = int(temp.max())
    init_min = int(temp.min())
    solver = _solver()
    heat = _zero_heat(temp.shape)
    is_vacuum, atmosphere = _cooling_fields(temp.shape)
    prev_max, prev_min = init_max, init_min
    for _ in range(200):
        solver.step(temp, heat, shift, face, solid, is_vacuum, atmosphere)
        cur_max, cur_min = int(temp.max()), int(temp.min())
        assert cur_max <= prev_max, "global max increased (extremum created)"
        assert cur_min >= prev_min, "global min decreased (extremum created)"
        assert cur_max <= init_max, "value exceeded the initial maximum"
        assert cur_min >= init_min, "value fell below the initial minimum"
        prev_max, prev_min = cur_max, cur_min


def test_wood_metal_face_conducts_at_wood_rate():
    # A hull tile and a wood tile share ONE face. The flux across that face uses
    # the harmonic-mean shift (~the wood, slow, rate), NOT the hull-fast rate.
    # Compare the per-tick flux to a pure wood-wood face under the same drop.
    mats_mix = np.array([[MAT_HULL, MAT_WOOD]], dtype=np.int8)
    sh_mix, fc_mix, so_mix = _build_caches(mats_mix)
    mats_wood = np.array([[MAT_WOOD, MAT_WOOD]], dtype=np.int8)
    sh_w, fc_w, so_w = _build_caches(mats_wood)

    DROP = 1 << 24
    # one tick, hot left tile, cold right tile, in each pair
    def one_tick_gain(shift, face, solid):
        temp = np.zeros((1, 2), dtype=np.int32)
        temp[0, 0] = DROP
        _run(temp, shift, face, solid, 1)
        return int(temp[0, 1])   # how much the cold tile gained

    gain_mix = one_tick_gain(sh_mix, fc_mix, so_mix)
    gain_wood = one_tick_gain(sh_w, fc_w, so_w)
    # The mixed face's flux is at the wood-ish (slow) scale: within a factor of
    # 2 of the wood-wood flux (one shift bucket), and FAR below the hull-fast
    # flux (DROP >> 2). It must NOT conduct like metal.
    hull_fast = DROP >> SHIFT_HULL
    assert gain_mix <= 2 * gain_wood, "wood<->metal face conducts too fast (not ~wood)"
    assert gain_mix < hull_fast // 4, "wood<->metal face conducts like metal (wrong)"
    # And the actual face shift used is the wood-ish bucket.
    assert int(fc_mix[0, 0, 2]) == SHIFT_WOOD_HULL   # E face of left tile


def test_deterministic_bit_identical():
    # Same field + same caches -> bit-identical after N ticks, two independent
    # runs (gather + double-buffer is order-independent, pure integer).
    rng = np.random.default_rng(7)
    h, w = 10, 10
    mats = rng.integers(0, 6, size=(h, w)).astype(np.int8)
    shift, face, solid = _build_caches(mats)
    base = rng.integers(0, 1 << 24, size=(h, w), dtype=np.int64).astype(np.int32)

    def run():
        temp = np.ascontiguousarray(base.copy())
        _run(temp, shift, face, solid, 64)
        return temp

    a = run()
    b = run()
    assert np.array_equal(a, b), "conduction is not deterministic"
