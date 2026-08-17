"""Conduction relaxation — STEP B, rewritten in ENERGY form at P-E2a.

The C++ ``TemperatureSolver`` runs ONE gather-stencil conduction pass
(double-buffered). Since P-E2a (energy-books arc, design
``energy_transport_design_2026-08-16.md`` §2.3) that pass is denominated in
ENERGY, not in temperature:

    g     = |T_j - T_i| ;  C_min = min(C_i, C_j)
    ΔE    = ±((g·C_min) >> s),  clamped to ±((g·C_min) >> LIM_SHIFT)
    ΔT_i  = floordiv(Σ_faces ΔE, C_i)          (NO_FACE faces skipped)

where C is the cell's real heat capacity (object: thermal_mass; gas: N·c_v
floored by ``n_floor_heat``). WHAT CHANGED, AND WHY THIS MODULE'S OLD METRIC
DIED: the pre-P-E2a law exchanged ΔT, so across a solid<->air face — where the
capacities differ by ~32× — the energy the wall gained was 32× the energy the
gas lost. The flat sum ``temperature.sum()`` was therefore NOT a conserved
quantity of the physics; it merely looked like one because the old law moved
equal ΔT to both ends. The conserved quantity is the CAPACITY-WEIGHTED sum
Σ C_i·T_i, and this module now measures that.

Verifies, on small synthetic grids (no renderer, no ray pass):
  - a hot solid tile spreads heat to conductive solid neighbours over ticks;
  - METAL (hull, low face shift) spreads FASTER than WOOD (high face shift);
  - **FACE ANTISYMMETRY, EXACTLY** (design §2.3 constraint 1): an independent
    Python transcription of the law reproduces the C++ field bit-for-bit, and
    its per-face ΔE array sums to EXACTLY 0 over every face pair, in int64;
  - **the energy books CLOSE to an identity** (design §7): Σ_cells ΔT_i·C_i
    equals the two counted residuals (``e_cond_trunc_sum`` +
    ``e_cond_cap_sum``) and nothing else — conduction's global drift IS the
    counted floor terms;
  - **the LIMITER-BOUNDED property** replaces the old discrete-maximum
    principle (authorized rewrite, Appendix A P-E2a): no endpoint may pass the
    donor across a face, and no new extremum appears beyond the ≤1-raw-count
    slack the endpoint ``floordiv`` (toward −∞) can add on a losing cell;
  - AIR tiles stay BIT-EXACTLY 0 where every air face is NO_FACE;
  - EQUAL-temperature neighbours produce ZERO change (g == 0 ⇒ ΔE == 0);
  - a WOOD<->METAL face conducts at ~the WOOD (slow) rate (harmonic mean);
  - determinism: same field -> bit-identical after N ticks.

Run:
    conda run -n data python -m pytest tests/test_temperature_conduction.py -q
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


# ---------------------------------------------------------------------------
# P-E2a: an INDEPENDENT Python transcription of the energy law (design §2.3).
# It exists so the face-antisymmetry gate is a real reimplementation, not a
# restatement of the C++ arithmetic — if the two agree bit-for-bit AND this
# one's per-face ΔE array sums to exactly 0, constraint 1 holds.
# ---------------------------------------------------------------------------
FP_ONE = 65536
CAP_SHIFT_MAX = 12      # conduction::CAP_SHIFT_MAX
LIM_SHIFT = 1           # conduction::LIM_SHIFT
_OPP = (1, 0, 3, 2)     # N<->S, E<->W


def _quantize(v):
    """fixedpoint::quantize — round half away from zero."""
    return int(np.floor(v * FP_ONE + 0.5) if v >= 0 else np.ceil(v * FP_ONE - 0.5))


def _capacities(ts, heat_inv_shift, n_raw, n_floor_heat=0.05, c_v=1.0):
    """conduction::cell_capacity_q, vectorized. Returns (cap_used, cap_real)."""
    n_floor_q = _quantize(n_floor_heat)
    c_v_q = _quantize(c_v)
    his = np.maximum(heat_inv_shift.astype(np.int64), 0)
    ceiling = np.int64(1) << (CAP_SHIFT_MAX + 16)

    used = np.zeros(ts.shape, dtype=np.int64)
    real = np.zeros(ts.shape, dtype=np.int64)
    used[ts] = np.int64(1) << np.minimum(his[ts], CAP_SHIFT_MAX)
    used[ts] <<= 16
    real[ts] = np.int64(1) << np.minimum(his[ts], 30)
    real[ts] <<= 16

    nr = np.maximum(n_raw.astype(np.int64), 0)
    nu = np.maximum(nr, np.int64(n_floor_q))
    cu = np.minimum((nu * c_v_q) >> 16, ceiling)
    cr = np.minimum((nr * c_v_q) >> 16, ceiling)
    used[~ts] = np.maximum(cu[~ts], 1)
    real[~ts] = np.maximum(cr[~ts], 0)
    return used, real


def _mirror_conduction(temp, face, cap_used, cap_real):
    """One conduction pass, transcribed from design §2.3.

    Returns (temp_new, de_face, trunc_sum, cap_sum, limit_hits) where
    ``de_face[y, x, d]`` is the energy that flowed INTO (y, x) across face d —
    the array whose antisymmetry constraint 1 is about.
    """
    h, w = temp.shape
    T = temp.astype(np.int64)
    de_face = np.zeros((h, w, 4), dtype=np.int64)
    limit_hits = 0
    for y in range(h):
        for x in range(w):
            for d, (dy, dx) in enumerate(_FACE_DIRS):
                s_i = int(face[y, x, d])
                if s_i == NO_FACE:
                    continue
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                s_j = int(face[ny, nx, _OPP[d]])
                if s_j == NO_FACE:
                    continue
                s = max(s_i, s_j)
                diff = int(T[ny, nx]) - int(T[y, x])
                g = abs(diff)
                cmin = min(int(cap_used[y, x]), int(cap_used[ny, nx]))
                full = g * cmin
                q = full >> s
                lim = full >> LIM_SHIFT
                if q > lim:
                    q = lim
                    limit_hits += 1
                de_face[y, x, d] = q if diff >= 0 else -q
    de = de_face.sum(axis=2)
    dT = np.zeros((h, w), dtype=np.int64)
    trunc_sum = 0
    cap_sum = 0
    for y in range(h):
        for x in range(w):
            if de[y, x] == 0:
                continue
            cu = int(cap_used[y, x])
            q = int(de[y, x]) // cu          # Python // IS floor division
            dT[y, x] = q
            trunc_sum += q * cu - int(de[y, x])
            cap_sum += q * (int(cap_real[y, x]) - cu)
    return (T + dT).astype(np.int32), de_face, trunc_sum, cap_sum, limit_hits


def _face_pairs_are_antisymmetric(de_face, face):
    """Every shared face's two directed entries must sum to EXACTLY 0."""
    h, w, _ = de_face.shape
    worst = 0
    for y in range(h):
        for x in range(w):
            for d, (dy, dx) in enumerate(_FACE_DIRS):
                ny, nx = y + dy, x + dx
                if not (0 <= ny < h and 0 <= nx < w):
                    continue
                s = int(de_face[y, x, d]) + int(de_face[ny, nx, _OPP[d]])
                worst = max(worst, abs(s))
    return worst


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


def test_air_conducts_with_solids():
    # EOS P2 (unified temperature, design §4): air now has a small nonzero
    # conductivity, so the hull<->air interface is a REAL face — the sealed-room
    # energy sink the unified-field decision banks on. The OLD doctrine (air
    # faces are NO_FACE; a hull flanked by air is a no-op) was retired by locked
    # decision 7; golden re-blessed at the P1+P2 merge, 2026-07-10.
    mats = np.array([[MAT_AIR, MAT_HULL, MAT_AIR]], dtype=np.int8)
    shift, face, solid = _build_caches(mats)
    temp = np.zeros((1, 3), dtype=np.int32)
    temp[0, 1] = 1 << 24          # hot hull, air on both sides
    _ts0 = solid
    _n0 = np.full(temp.shape, FP_ONE, dtype=np.int32)
    before_energy = int((temp.astype(np.int64) * _capacities(_ts0, shift, _n0)[1]).sum())
    _run(temp, shift, face, solid, 50)
    assert temp[0, 0] > 0, "air did not receive interface conduction"
    assert temp[0, 2] > 0, "air did not receive interface conduction"
    assert temp[0, 1] < (1 << 24), "hull did not lose heat to the air"
    # P-E2a: the conserved quantity is the CAPACITY-WEIGHTED sum, not the flat
    # one. The old `temperature.sum() <= before` assert measured a quantity the
    # honest law does not conserve (and MUST not: the hull is ~32× heavier than
    # the gas, so a hull losing 1 degree must warm the air by ~32, and the flat
    # sum RISES). Weighted, conduction is one-way: it can only lose, via the
    # counted endpoint-truncation term.
    ts = solid                       # no thermal_solid passed -> solver uses `solid`
    n_raw = np.full(temp.shape, FP_ONE, dtype=np.int32)   # atmosphere proxy
    _, cap_real = _capacities(ts, shift, n_raw)
    assert int((temp.astype(np.int64) * cap_real).sum()) <= before_energy, (
        "conduction invented ENERGY (capacity-weighted books)")


def test_equal_neighbours_zero_change():
    # A uniform hull block at a constant nonzero temperature: every difference
    # is 0, so (T_n - T_i) >> s == 0 on every face -> NO drift, exactly stable.
    mats = np.full((4, 4), MAT_HULL, dtype=np.int8)
    shift, face, solid = _build_caches(mats)
    temp = np.full((4, 4), 12345 << 4, dtype=np.int32)
    before = temp.copy()
    _run(temp, shift, face, solid, 100)
    assert np.array_equal(temp, before), "equal-temp field drifted (should be exact)"


def _mixed_field(seed, h=12, w=12):
    """A HETEROGENEOUS grid — mixed capacities are the whole point of the
    energy form, and a homogeneous hull block cannot exercise it (there
    C_min == C_i == C_j and the law reduces to the old one)."""
    rng = np.random.default_rng(seed)
    mats = rng.choice([MAT_AIR, MAT_HULL, MAT_WOOD], size=(h, w)).astype(np.int8)
    shift, face, solid = _build_caches(mats)
    temp = np.ascontiguousarray(
        rng.integers(-(1 << 22), 1 << 22, size=(h, w), dtype=np.int64).astype(np.int32))
    # A wide spread of gas densities, so the thin/dense capacity contrast (and
    # the n_floor_heat floor) is actually exercised.
    n_bulk = np.ascontiguousarray(
        rng.integers(0, 3 * FP_ONE, size=(h, w), dtype=np.int64).astype(np.int32))
    return mats, shift, face, solid, temp, n_bulk


def test_limiter_bounded_no_endpoint_passes_the_donor():
    """P-E2a's REPLACEMENT for the old discrete-maximum-principle test
    (authorized rewrite, Appendix A). The convex bound the ΔT form had for
    free is now assembled from two pieces, and this asserts both:

      (a) PER FACE (design §2.3 constraint 4): moving ΔE across a face changes
          the gap by ΔE/C_i + ΔE/C_j ≥ 2·ΔE/C_min, so capping |ΔE| at
          (g·C_min) >> LIM_SHIFT == half the gap-closing energy means the gap
          can never INVERT — no endpoint passes the donor.
      (b) AGGREGATE (SHIFT_MIN == 2, unchanged): each face moves at most g/4 of
          ΔT because C_min ≤ C_i, so the four-face update is still a convex
          combination and no NEW extremum appears — up to the ≤1-raw-count
          slack `floordiv_q` (toward −∞) can add on a cell that is LOSING
          energy. That slack is the price of the one-way rounding R3 requires,
          and it is asserted as a bound, not waved at.
    """
    LSB_SLACK = 1        # the floordiv toward−∞ allowance, per cell per tick
    for seed in (20260609, 424242, 7):
        mats, shift, face, solid, temp, n_bulk = _mixed_field(seed)
        solver = _solver()
        heat = _zero_heat(temp.shape)
        is_vacuum, atmosphere = _cooling_fields(temp.shape)
        cap_used, _ = _capacities(solid, shift, n_bulk)
        for _ in range(200):
            before = temp.copy()
            solver.step(temp, heat, shift, face, solid, is_vacuum, atmosphere,
                        None, None, 0.0, n_bulk)
            lo, hi = int(before.min()), int(before.max())
            assert int(temp.max()) <= hi + LSB_SLACK, (
                "a new global MAXIMUM appeared beyond the floordiv slack")
            assert int(temp.min()) >= lo - LSB_SLACK, (
                "a new global MINIMUM appeared beyond the floordiv slack")
        assert np.any(temp != before), "scenario went inert (vacuous gate)"


def test_single_face_gap_never_inverts_at_any_capacity_ratio():
    """(a) of the limiter property, in ISOLATION — one face, two cells, so the
    face's own quantum is the ONLY thing acting.

    This is the assertion the limiter directly buys, and it is deliberately
    separate from the aggregate one above: with four live faces a cell can of
    course end up past ONE particular neighbour (three hot neighbours outvote a
    barely-hotter fourth) — that is ordinary diffusion and the pre-P-E2a law
    did it too. What must never happen is a SINGLE face's exchange overshooting
    its own gap, which is precisely the failure mode a floored thin endpoint
    would have caused: without the limiter it would close 4× the gap per tick.

    Swept across the extreme capacity ratios the map really contains — a hull
    tile (C = 32) against near-vacuum gas (C floored at 0.05·c_v), a 640×
    contrast in both directions.
    """
    pairs = ((MAT_HULL, MAT_AIR), (MAT_AIR, MAT_HULL), (MAT_WOOD, MAT_AIR),
             (MAT_HULL, MAT_WOOD), (MAT_AIR, MAT_AIR), (MAT_HULL, MAT_HULL))
    n_levels = (0, 1, _quantize(0.01), _quantize(0.05), _quantize(0.5),
                FP_ONE, 4 * FP_ONE)
    rng = np.random.default_rng(5150)
    checked = 0
    for (ma, mb) in pairs:
        mats = np.array([[ma, mb]], dtype=np.int8)
        shift, face, solid = _build_caches(mats)
        if (face[0, 0, 2] == NO_FACE):
            continue                      # no live face for this pair
        for na in n_levels:
            for nb_ in n_levels:
                n_bulk = np.ascontiguousarray(
                    np.array([[na, nb_]], dtype=np.int32))
                for _ in range(8):
                    t0 = int(rng.integers(-(1 << 23), 1 << 23))
                    t1 = int(rng.integers(-(1 << 23), 1 << 23))
                    temp = np.ascontiguousarray(
                        np.array([[t0, t1]], dtype=np.int32))
                    g0 = t1 - t0
                    solver = _solver()
                    heat = _zero_heat(temp.shape)
                    is_vacuum, atmosphere = _cooling_fields(temp.shape)
                    solver.step(temp, heat, shift, face, solid, is_vacuum,
                                atmosphere, None, None, 0.0, n_bulk)
                    g1 = int(temp[0, 1]) - int(temp[0, 0])
                    checked += 1
                    # The gap may shrink to 0 or stay put; it may NOT flip sign
                    # by more than the two endpoints' floordiv slack.
                    assert not (g0 * g1 < 0 and abs(g1) > 2), (
                        f"single-face gap inverted: {g0} -> {g1} "
                        f"(mats {ma},{mb}; N {na},{nb_})")
                    # And it may not GROW: a face only ever relaxes.
                    assert abs(g1) <= abs(g0) + 2, (
                        f"single-face gap GREW: {g0} -> {g1}")
    assert checked > 500, "sweep did not actually run"


def test_conduction_face_energy_is_exactly_antisymmetric():
    """GATE (design §2.3 constraint 1; the patch's headline property).

    An INDEPENDENT Python transcription of the law is run alongside the C++
    solver. Two things must hold, both EXACTLY, in int64:
      1. every shared face's two directed ΔE entries sum to 0 — what leaves i
         enters j, bit for bit;
      2. the mirror's resulting field is bit-identical to the C++ solver's, so
         (1) is a statement about the shipped law and not about the mirror.
    """
    mats, shift, face, solid, temp, n_bulk = _mixed_field(31337, h=10, w=10)
    solver = _solver()
    heat = _zero_heat(temp.shape)
    is_vacuum, atmosphere = _cooling_fields(temp.shape)
    cap_used, cap_real = _capacities(solid, shift, n_bulk)

    moved = 0
    for _ in range(12):
        expect, de_face, _, _, _ = _mirror_conduction(temp, face, cap_used, cap_real)
        assert _face_pairs_are_antisymmetric(de_face, face) == 0, (
            "face energy is NOT antisymmetric — energy is created or destroyed "
            "at a face")
        solver.step(temp, heat, shift, face, solid, is_vacuum, atmosphere,
                    None, None, 0.0, n_bulk)
        assert np.array_equal(temp, expect), (
            "the C++ conduction pass does not match the independent "
            "transcription of design §2.3")
        moved += int(np.abs(de_face).sum())
    assert moved > 0, "no energy moved at all (vacuous gate)"


def test_conduction_energy_books_close():
    """GATE (design §7): conduction's global energy drift IS the counted floor
    terms — nothing else. Because Σ_cells ΔE_i == 0 exactly (antisymmetry),

        Σ_cells ΔT_i · C_real_i  ==  e_cond_trunc_sum + e_cond_cap_sum

    holds every tick, as an identity in int64, not as a bound.
    """
    mats, shift, face, solid, temp, n_bulk = _mixed_field(20260817, h=14, w=14)
    solver = _solver()               # cooling disabled -> conduction in isolation
    heat = _zero_heat(temp.shape)
    is_vacuum, atmosphere = _cooling_fields(temp.shape)
    _, cap_real = _capacities(solid, shift, n_bulk)

    prev = (0, 0)
    for tick in range(150):
        before = temp.copy()
        solver.step(temp, heat, shift, face, solid, is_vacuum, atmosphere,
                    None, None, 0.0, n_bulk)
        now = (int(solver.e_cond_trunc_sum), int(solver.e_cond_cap_sum))
        d_trunc, d_cap = now[0] - prev[0], now[1] - prev[1]
        prev = now
        lhs = int(((temp.astype(np.int64) - before.astype(np.int64)) * cap_real).sum())
        assert lhs == d_trunc + d_cap, (
            f"tick {tick}: conduction books do NOT close: "
            f"ΣΔT·C = {lhs}, counted = {d_trunc + d_cap}")
        # R3: the endpoint truncation is ONE-WAY — it may only destroy.
        assert d_trunc <= 0, "the endpoint floordiv residual CREATED energy"
    assert prev[0] < 0, "no truncation at all (vacuous gate)"
    # Pass 3 and the Pass-0 wipes are inert in this scenario (cooling disabled,
    # no vacuum, no ring), so their SIGNED channels must read exactly 0.
    assert int(solver.e_cool_sum) == 0
    assert int(solver.e_vac_wipe_sum) == 0
    assert int(solver.e_ring_pin_sum) == 0


def test_conduction_limiter_is_inert_at_shipped_face_shifts():
    """The per-face limiter (constraint 4) is a STRUCTURAL guarantee, not an
    operating mechanism: with SHIFT_MIN == 2 every face already moves at most a
    QUARTER of the gap-closing energy, so the ≤½ clamp never binds at any
    shipped face shift. Measured here so the "worst-case overshoot fraction"
    the patch reports is a number, not an assumption."""
    mats, shift, face, solid, temp, n_bulk = _mixed_field(999, h=10, w=10)
    solver = _solver()
    heat = _zero_heat(temp.shape)
    is_vacuum, atmosphere = _cooling_fields(temp.shape)
    cap_used, cap_real = _capacities(solid, shift, n_bulk)

    worst_frac = 0.0
    for _ in range(30):
        _, de_face, _, _, limit_hits = _mirror_conduction(temp, face, cap_used, cap_real)
        assert limit_hits == 0, "the limiter engaged at a SHIPPED face shift"
        for d, (dy, dx) in enumerate(_FACE_DIRS):
            h_, w_ = temp.shape
            for y in range(h_):
                for x in range(w_):
                    ny, nx = y + dy, x + dx
                    if not (0 <= ny < h_ and 0 <= nx < w_):
                        continue
                    if int(face[y, x, d]) == NO_FACE:
                        continue
                    g = abs(int(temp[ny, nx]) - int(temp[y, x]))
                    cmin = min(int(cap_used[y, x]), int(cap_used[ny, nx]))
                    if g * cmin == 0:
                        continue
                    worst_frac = max(worst_frac,
                                     abs(int(de_face[y, x, d])) / (g * cmin))
        solver.step(temp, heat, shift, face, solid, is_vacuum, atmosphere,
                    None, None, 0.0, n_bulk)
    # <= 1/4 at SHIFT_MIN == 2; the limiter's own ceiling is 1/2.
    assert worst_frac <= 0.25 + 1e-12, f"face fraction {worst_frac} exceeded 1/4"
    assert int(solver.cond_limit_hits) == 0, (
        "the C++ limiter counter engaged at shipped face shifts")


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
