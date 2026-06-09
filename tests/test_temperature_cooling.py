"""Ambient cooling (engine/06 §3, proposal §3) — STEP C.

After the §1 heat -> temperature conversion and the §2 conduction relaxation,
the C++ ``TemperatureSolver`` runs ONE ambient-cooling pass (the LAST thermal
pass, §3.5). Temperature stores ΔT above ambient, so T_ambient == 0 and cooling
relaxes toward 0 with no subtraction:

    shift = exposed ? COOL_SHIFT_VACUUM : COOL_SHIFT
    T    -= (T < 0) ? -((-T) >> shift) : (T >> shift)          # round toward 0

`exposed` is true when ANY in-bounds 4-neighbour is vacuum (is_vacuum) OR has
atmosphere < o2_vacuum_thresh — read from the SAME atmosphere/vacuum fields the
rest of the physics uses (no new field/buffer), reusing the geometric N,S,E,W
gather the conduction pass walks. Cooling runs on SOLID tiles only.

Verifies, on small synthetic grids (no renderer, no ray pass, conduction
disabled via an all-NO_FACE face cache so cooling is exercised in isolation):
  - a hot solid tile RELAXES toward 0 over ticks, monotone, and never crosses
    below ambient for a single isolated tile;
  - a VACUUM-EXPOSED tile (a vacuum neighbour) cools ~4× faster than an interior
    tile with the same start (COOL_SHIFT 5 -> COOL_SHIFT_VACUUM 3);
  - the atmosphere threshold also flips a tile to the fast shift (a neighbour
    with atmosphere < o2_vacuum_thresh counts as vacuum-exposed);
  - the DEAD-BAND: a small value < (1<<COOL_SHIFT) settles to an EXACT rest and
    stops — no jitter, no overshoot below ambient;
  - AIR stays BIT-EXACTLY 0 (non-solid tiles are skipped by cooling);
  - a negative ΔT (below ambient) relaxes UP toward 0, symmetric, never crosses;
  - determinism: same field -> bit-identical after N ticks;
  - integration sanity: inject heat once (conduction ON), then with no further
    deposit the tile conducts + cools back down toward ambient (the burn-out
    precondition).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_temperature_cooling.py -q
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

from config import CFG
from simulation.materials import (
    MAT_AIR, MAT_HULL, MAT_WOOD,
    MaterialTable,
)

_TBL = MaterialTable.from_config()
NO_FACE = int(_TBL.no_face)

# Cooling dials read from the live config so the test tracks the [physics.thermal]
# dials rather than a hardcoded copy.
_THERMAL = CFG.physics.thermal
COOL_SHIFT = int(getattr(_THERMAL, "COOL_SHIFT", 5))
COOL_SHIFT_VACUUM = int(getattr(_THERMAL, "COOL_SHIFT_VACUUM", 3))
O2_VAC_THRESH = float(getattr(_THERMAL, "o2_vacuum_thresh", 0.3))


def test_shipped_cooling_dials():
    # Guard the dials STEP C is anchored to: interior >>5, vacuum >>3 (4× faster),
    # and a sensible atmosphere threshold for the vacuum-exposure test.
    assert COOL_SHIFT == 5, f"COOL_SHIFT should be 5, got {COOL_SHIFT}"
    assert COOL_SHIFT_VACUUM == 3, f"COOL_SHIFT_VACUUM should be 3, got {COOL_SHIFT_VACUUM}"
    # 2^(5-3) == 4 -> vacuum sheds 4× the interior fraction per tick.
    assert (1 << COOL_SHIFT) // (1 << COOL_SHIFT_VACUUM) == 4
    assert 0.0 < O2_VAC_THRESH < 1.0, "o2_vacuum_thresh must sit between vacuum and full atm"


def _solver():
    """A solver with the SHIPPED cooling dials and conduction left to the caller
    (this module always passes an all-NO_FACE face cache, so conduction is a
    no-op and only the §3 cooling pass acts)."""
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.cool_shift = COOL_SHIFT
    s.cool_shift_vacuum = COOL_SHIFT_VACUUM
    s.o2_vacuum_thresh = O2_VAC_THRESH
    return s


def _grid(material_ids, *, vacuum=None, atmosphere=None, face=None):
    """Build the solver inputs for a 1-row grid of the given material ids.

    By default: conduction OFF (all-NO_FACE face cache), no vacuum, full
    atmosphere (1.0) -> every solid tile cools at the interior COOL_SHIFT. Pass
    ``vacuum`` (bool list) / ``atmosphere`` (float list) to drive the
    vacuum-exposure test, or ``face`` to enable conduction for the integration
    sanity check.
    """
    mats = np.asarray(material_ids, dtype=np.int8).reshape(1, -1)
    w = mats.shape[1]
    temperature = np.zeros((1, w), dtype=np.int32)
    heat = np.ascontiguousarray(np.zeros((1, w), dtype=np.int32))
    shift = np.ascontiguousarray(_TBL.heat_inv_shift[mats].astype(np.int32))
    solid = np.ascontiguousarray(_TBL.permeability[mats] <= 0.0)
    if face is None:
        face = np.full((1, w, 4), NO_FACE, dtype=np.int32)
    face = np.ascontiguousarray(face.astype(np.int32))
    if vacuum is None:
        is_vacuum = np.zeros((1, w), dtype=bool)
    else:
        is_vacuum = np.asarray(vacuum, dtype=bool).reshape(1, w)
    if atmosphere is None:
        atm = np.ones((1, w), dtype=np.float32)
    else:
        atm = np.asarray(atmosphere, dtype=np.float32).reshape(1, w)
    return (temperature,
            heat, shift, face,
            np.ascontiguousarray(solid),
            np.ascontiguousarray(is_vacuum),
            np.ascontiguousarray(atm))


def _run(solver, temp, heat, shift, face, solid, is_vacuum, atm, n_ticks):
    for _ in range(n_ticks):
        solver.step(temp, heat, shift, face, solid, is_vacuum, atm)
    return temp


def test_hot_tile_relaxes_toward_zero_monotone():
    # A single isolated hot hull tile (conduction off). Cooling must drive it
    # toward 0, strictly decreasing while above the dead-band, and it must NEVER
    # cross below ambient (0).
    temp, heat, shift, face, solid, vac, atm = _grid([MAT_HULL])
    temp[0, 0] = 1 << 24
    solver = _solver()
    prev = int(temp[0, 0])
    # Enough ticks to cross the full ~16.7M -> dead-band span at ~3.1%/tick
    # (~13 e-folds × ~32 ticks, plus the slowdown approaching the band).
    for _ in range(1500):
        solver.step(temp, heat, shift, face, solid, vac, atm)
        cur = int(temp[0, 0])
        assert cur <= prev, "temperature increased during cooling (not monotone)"
        assert cur >= 0, "single tile crossed BELOW ambient (overshoot)"
        prev = cur
    # After many e-folds it has settled into the dead-band near 0 and stopped.
    assert temp[0, 0] < (1 << COOL_SHIFT), "hot tile did not relax into the rest band"
    before = int(temp[0, 0])
    _run(solver, temp, heat, shift, face, solid, vac, atm, 100)
    assert int(temp[0, 0]) == before, "rest band is not an exact fixed point"


def test_vacuum_exposed_cools_about_4x_faster():
    # Two identical hot tiles starting equal. The INTERIOR tile (all neighbours
    # full atmosphere) cools at COOL_SHIFT; the VACUUM-EXPOSED tile (a vacuum
    # 4-neighbour) cools at COOL_SHIFT_VACUUM == 4× the per-tick fraction. After
    # ONE tick the exposed tile has shed ~4× as much.
    T0 = 1 << 24

    # Interior: a single hull tile, no vacuum, full atmosphere.
    ti, hi, si, fi, soli, vi, ai = _grid([MAT_HULL])
    ti[0, 0] = T0
    _run(_solver(), ti, hi, si, fi, soli, vi, ai, 1)
    interior_loss = T0 - int(ti[0, 0])

    # Exposed: a hull tile with a vacuum tile as its E neighbour. The vacuum cell
    # is non-solid so it never cools itself; it only flags the hull as exposed.
    te, he, se, fe, sole, ve, ae = _grid([MAT_HULL, MAT_AIR], vacuum=[False, True])
    te[0, 0] = T0
    _run(_solver(), te, he, se, fe, sole, ve, ae, 1)
    exposed_loss = T0 - int(te[0, 0])

    assert interior_loss == T0 >> COOL_SHIFT
    assert exposed_loss == T0 >> COOL_SHIFT_VACUUM
    ratio = exposed_loss / interior_loss
    assert abs(ratio - 4.0) < 1e-6, (
        f"vacuum-exposed should shed ~4× the interior: ratio {ratio:.3f} "
        f"(interior_loss {interior_loss}, exposed_loss {exposed_loss})"
    )


def test_low_atmosphere_neighbour_counts_as_exposed():
    # A neighbour BELOW o2_vacuum_thresh (but not flagged is_vacuum) must also
    # flip the tile to the fast shift — the atmosphere half of the exposure OR.
    T0 = 1 << 24
    low = float(O2_VAC_THRESH) * 0.5      # safely below the threshold
    t, h, s, f, sol, v, a = _grid([MAT_HULL, MAT_AIR], atmosphere=[1.0, low])
    t[0, 0] = T0
    _run(_solver(), t, h, s, f, sol, v, a, 1)
    assert (T0 - int(t[0, 0])) == T0 >> COOL_SHIFT_VACUUM, (
        "low-atmosphere neighbour did not trigger fast (vacuum) cooling"
    )

    # And a neighbour ABOVE the threshold stays on the slow interior shift.
    t2, h2, s2, f2, sol2, v2, a2 = _grid(
        [MAT_HULL, MAT_AIR], atmosphere=[1.0, float(O2_VAC_THRESH) + 0.5])
    t2[0, 0] = T0
    _run(_solver(), t2, h2, s2, f2, sol2, v2, a2, 1)
    assert (T0 - int(t2[0, 0])) == T0 >> COOL_SHIFT, (
        "above-threshold neighbour wrongly triggered fast cooling"
    )


def test_dead_band_settles_to_exact_rest():
    # A value strictly below (1<<COOL_SHIFT) shifts to a loss of 0 -> it never
    # decays: an exact, jitter-free resting state at the value (no "+1 nudge").
    rest = (1 << COOL_SHIFT) - 1            # largest value with loss == 0
    t, h, s, f, sol, v, a = _grid([MAT_HULL])
    t[0, 0] = rest
    solver = _solver()
    _run(solver, t, h, s, f, sol, v, a, 200)
    assert int(t[0, 0]) == rest, (
        f"dead-band value jittered/decayed: {int(t[0,0])} != {rest}"
    )

    # One above the band sheds exactly once into the band, then rests.
    t[0, 0] = 1 << COOL_SHIFT               # loss == 1
    _run(solver, t, h, s, f, sol, v, a, 1)
    assert int(t[0, 0]) == (1 << COOL_SHIFT) - 1
    _run(solver, t, h, s, f, sol, v, a, 100)
    assert int(t[0, 0]) == (1 << COOL_SHIFT) - 1, "did not settle into the rest band"


def test_air_stays_bit_exactly_zero():
    # Cooling runs on SOLID tiles only. An air (non-solid) tile is skipped and
    # stays bit-exactly 0 even sitting next to a hot, vacuum-exposed wall.
    t, h, s, f, sol, v, a = _grid([MAT_AIR, MAT_HULL, MAT_AIR],
                                  vacuum=[True, False, True])
    assert not sol[0, 0] and not sol[0, 2], "sanity: air must be non-solid"
    t[0, 1] = 1 << 24
    _run(_solver(), t, h, s, f, sol, v, a, 100)
    assert int(t[0, 0]) == 0, f"air gained temperature: {int(t[0,0])}"
    assert int(t[0, 2]) == 0, f"air gained temperature: {int(t[0,2])}"


def test_negative_delta_relaxes_up_toward_zero():
    # A tile BELOW ambient (negative ΔT) relaxes UP toward 0, symmetric to the
    # hot case, and never crosses above ambient. The signed shift is pinned to
    # round toward 0 (`x<0 ? -((-x)>>s) : x>>s`).
    t, h, s, f, sol, v, a = _grid([MAT_HULL])
    t[0, 0] = -(1 << 24)
    solver = _solver()
    prev = int(t[0, 0])
    for _ in range(1500):
        solver.step(t, h, s, f, sol, v, a)
        cur = int(t[0, 0])
        assert cur >= prev, "below-ambient tile cooled further (wrong direction)"
        assert cur <= 0, "below-ambient tile crossed ABOVE ambient (overshoot)"
        prev = cur
    assert -int(t[0, 0]) < (1 << COOL_SHIFT), "did not relax into the rest band"
    # Symmetry: the magnitude one tick from a symmetric start matches the hot side.
    th, *_rest = _grid([MAT_HULL])
    th[0, 0] = 1 << 24
    _run(solver, th, h, s, f, sol, v, a, 1)
    tc, *_rest2 = _grid([MAT_HULL])
    tc[0, 0] = -(1 << 24)
    _run(solver, tc, h, s, f, sol, v, a, 1)
    assert int(th[0, 0]) == -int(tc[0, 0]), "cooling is not symmetric about ambient"


def test_deterministic_bit_identical():
    # Same field + same dials -> bit-identical after N ticks (pure signed add +
    # arithmetic right shift; gather over a frozen field).
    rng = np.random.default_rng(20260609)
    h_, w = 8, 8
    mats = np.full((h_, w), MAT_HULL, dtype=np.int8)
    base = rng.integers(-(1 << 22), 1 << 22, size=(h_, w), dtype=np.int64).astype(np.int32)
    # A random vacuum mask so both shifts are exercised.
    vmask = rng.integers(0, 2, size=(h_, w)).astype(bool)
    atm = np.ascontiguousarray(np.ones((h_, w), dtype=np.float32))
    shift = np.ascontiguousarray(_TBL.heat_inv_shift[mats].astype(np.int32))
    solid = np.ascontiguousarray(_TBL.permeability[mats] <= 0.0)
    face = np.ascontiguousarray(np.full((h_, w, 4), NO_FACE, dtype=np.int32))
    heat = np.ascontiguousarray(np.zeros((h_, w), dtype=np.int32))

    def run():
        temp = np.ascontiguousarray(base.copy())
        solver = _solver()
        for _ in range(64):
            solver.step(temp, heat, shift, face, solid,
                        np.ascontiguousarray(vmask.copy()), atm)
        return temp

    a = run()
    b = run()
    assert np.array_equal(a, b), "cooling is not deterministic"


def test_integration_inject_then_burn_out():
    # Burn-out precondition: inject heat ONCE into a hull strip (conduction ON),
    # then with NO further deposit the tile conducts the heat along the metal AND
    # cooling sheds it, so the whole strip relaxes back toward ambient over time.
    from simulation.materials import MAT_HULL as H
    mats = np.full((1, 5), H, dtype=np.int8)
    w = mats.shape[1]

    # Build a REAL conduction face cache (engine/06 §2 layout) so conduction acts.
    face_tbl = _TBL.face_shift_table
    face = np.full((1, w, 4), NO_FACE, dtype=np.int32)
    dirs = ((-1, 0), (1, 0), (0, 1), (0, -1))   # N,S,E,W
    for d, (dy, dx) in enumerate(dirs):
        ty0, ty1 = max(0, -dy), 1 - max(0, dy)
        tx0, tx1 = max(0, -dx), w - max(0, dx)
        if ty0 >= ty1 or tx0 >= tx1:
            continue
        mi = mats[ty0:ty1, tx0:tx1]
        mn = mats[ty0 + dy:ty1 + dy, tx0 + dx:tx1 + dx]
        face[ty0:ty1, tx0:tx1, d] = face_tbl[mi, mn]

    t, h, s, _f, sol, v, a = _grid([H, H, H, H, H], face=face)
    t[0, 2] = 1 << 26             # one big injection at the centre, then nothing
    total0 = int(t.astype(np.int64).sum())

    solver = _solver()
    # No fresh heat (h is all 0): conduction spreads, cooling sheds. Enough ticks
    # to cross the full span down into the dead-band at ~3.1%/tick.
    _run(solver, t, h, s, face, sol, v, a, 2000)

    total_end = int(t.astype(np.int64).sum())
    assert total_end < total0, "strip did not shed heat (no burn-out)"
    assert total_end >= 0, "strip overshot below ambient"
    # Settled near ambient: every tile is within the rest band.
    assert int(t.max()) < (1 << COOL_SHIFT), (
        f"strip did not cool back toward ambient: max {int(t.max())}"
    )
