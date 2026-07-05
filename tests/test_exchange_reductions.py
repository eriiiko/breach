"""The exchange-layer reduction vocabulary + coupling table (mechanics/05 §1).

Exercises ``simulation.exchange`` — the physics↔unit coupling-table module
introduced by the P1 behaviour-preserving refactor:

  - the v1 reduction vocabulary ``center | max | mean | sum | grad`` as pure
    integer functions over footprint tiles on an int32 field, on small
    hand-built arrays;
  - the in-bounds guard (mirroring combat.py's footprint loop): off-grid
    tiles are skipped, a fully off-grid footprint reduces to the zero
    element, and nothing ever raises;
  - ``mean``'s single round-half-away-from-zero divide — the sign-symmetric
    ``fixed_point.h::mean_round`` twin (no DC bias, exact on the half);
  - ``center``'s bounding-box middle == ``Unit.center_tile_x/y()`` for a real
    unit footprint;
  - the COUPLING_TABLE registration of the two shipped rows (heat → radiant
    damage, wave_p → blast damage): order, reduction names, and that the
    response callables ARE the shipped implementations (identity, also via
    the combat.py compatibility re-exports).

Run:
    C:/Users/steen/anaconda3/python.exe -m pytest tests/test_exchange_reductions.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

from simulation.exchange import (  # noqa: E402
    REDUCTIONS, CouplingRow,
    reduce_center, reduce_grad, reduce_max, reduce_mean, reduce_sum,
)
from simulation.unit import Unit  # noqa: E402


# ---------------------------------------------------------------------------
# Hand-built fields. Convention throughout: field[ty, tx], tiles are (tx, ty).
# ---------------------------------------------------------------------------
def _coord_field(h, w):
    """field[ty, tx] = 100*ty + tx — every tile value spells its coordinates."""
    ty, tx = np.mgrid[0:h, 0:w]
    return (100 * ty + tx).astype(np.int32)


def _square(anchor_x, anchor_y, size):
    """Row-major size×size footprint tiles, anchored top-left — exactly the
    shape/order of the species default offsets (species._default_3x3_offsets)."""
    return [(anchor_x + dx, anchor_y + dy)
            for dy in range(size) for dx in range(size)]


# ---------------------------------------------------------------------------
# center
# ---------------------------------------------------------------------------
def test_center_odd_footprint_is_middle_tile():
    f = _coord_field(4, 5)
    tiles = _square(1, 0, 3)           # bbox x:1..3, y:0..2 -> centre (2, 1)
    assert reduce_center(f, tiles) == 102


def test_center_is_order_independent():
    """The centre comes from the bounding box, not any list position."""
    f = _coord_field(4, 5)
    tiles = _square(1, 0, 3)
    assert reduce_center(f, list(reversed(tiles))) == reduce_center(f, tiles)


def test_center_even_footprint_matches_unit_convention():
    # 2x2 at (0,0): (lo+hi+1)//2 = 1 on both axes == anchor + footprint//2.
    f = _coord_field(4, 5)
    assert reduce_center(f, _square(0, 0, 2)) == 101


def test_center_single_tile():
    f = _coord_field(4, 5)
    assert reduce_center(f, [(4, 3)]) == 304


def test_center_matches_unit_center_tile():
    """On a real Unit footprint, reduce_center samples exactly the tile
    Unit.center_tile_x/y() names."""
    f = _coord_field(20, 20)
    u = Unit("M1", x=10, y=10, team=0)
    assert reduce_center(f, u.occupied_tiles()) == \
        int(f[u.center_tile_y(), u.center_tile_x()])


def test_center_empty_and_offgrid():
    f = _coord_field(4, 5)
    assert reduce_center(f, []) == 0
    # 3x3 anchored at (4,0) on w=5: centre tx=5 is off-grid -> 0.
    assert reduce_center(f, _square(4, 0, 3)) == 0
    # Straddling the edge but the centre tile itself still in bounds: the
    # geometry uses ALL tiles (position, not readability), sample is guarded.
    assert reduce_center(f, _square(3, 0, 3)) == 104   # centre (4, 1)


# ---------------------------------------------------------------------------
# max
# ---------------------------------------------------------------------------
def test_max_picks_hottest_tile():
    f = np.array([[5, -7], [3, 9]], dtype=np.int32)
    assert reduce_max(f, _square(0, 0, 2)) == 9


def test_max_ignores_offgrid_tiles():
    f = np.array([[5, -7], [3, 9]], dtype=np.int32)
    tiles = [(0, 0), (1, 1), (5, 5), (-1, 0)]   # two off-grid intruders
    assert reduce_max(f, tiles) == 9


def test_max_is_true_max_on_signed_fields():
    """Unlike the heat row's 0-floored inline peak (safe there: heat >= 0),
    the vocabulary max is a TRUE max — a signed field may reduce negative."""
    f = np.array([[-5, -3], [-9, -1]], dtype=np.int32)
    assert reduce_max(f, _square(0, 0, 2)) == -1


def test_max_empty_and_fully_offgrid():
    f = _coord_field(2, 2)
    assert reduce_max(f, []) == 0
    assert reduce_max(f, [(7, 7), (-2, -3)]) == 0


# ---------------------------------------------------------------------------
# mean — one integer sum + ONE round-half-away divide (mean_round twin)
# ---------------------------------------------------------------------------
def test_mean_exact_integer_mean():
    f = np.array([[10, 20, 31]], dtype=np.int32)
    tiles = [(0, 0), (1, 0), (2, 0)]
    assert reduce_mean(f, tiles) == 20        # (61 + 1) // 3


def test_mean_rounds_half_away_from_zero():
    f = np.array([[3, 4]], dtype=np.int32)
    assert reduce_mean(f, [(0, 0), (1, 0)]) == 4          # 3.5 -> 4
    g = np.array([[-3, -4]], dtype=np.int32)
    assert reduce_mean(g, [(0, 0), (1, 0)]) == -4         # -3.5 -> -4


def test_mean_sign_symmetric_no_dc_bias():
    """mean(-values) == -mean(values) — the fixed_point.h::mean_round contract
    (a truncating divide would bias each mean toward zero by sign(sum))."""
    vals = np.array([[7, 8, 10]], dtype=np.int32)
    tiles = [(0, 0), (1, 0), (2, 0)]
    assert reduce_mean(-vals, tiles) == -reduce_mean(vals, tiles)


def test_mean_excludes_offgrid_from_sum_and_count():
    f = np.array([[10]], dtype=np.int32)
    assert reduce_mean(f, [(0, 0), (9, 9)]) == 10   # NOT 5


def test_mean_empty_and_fully_offgrid():
    f = _coord_field(2, 2)
    assert reduce_mean(f, []) == 0
    assert reduce_mean(f, [(-1, -1)]) == 0


# ---------------------------------------------------------------------------
# sum
# ---------------------------------------------------------------------------
def test_sum_exact():
    f = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert reduce_sum(f, _square(0, 0, 2)) == 10


def test_sum_excludes_offgrid():
    f = np.array([[1, 2], [3, 4]], dtype=np.int32)
    assert reduce_sum(f, [(0, 0), (1, 1), (2, 0)]) == 5


def test_sum_no_int32_overflow():
    """Python-int accumulation: two INT32_MAX tiles sum exactly."""
    big = np.iinfo(np.int32).max
    f = np.array([[big, big]], dtype=np.int32)
    assert reduce_sum(f, [(0, 0), (1, 0)]) == 2 * int(big)


def test_sum_empty():
    assert reduce_sum(_coord_field(2, 2), []) == 0


# ---------------------------------------------------------------------------
# grad — difference of the extreme edge-line means, per axis
# ---------------------------------------------------------------------------
def test_grad_x_ramp():
    ty, tx = np.mgrid[0:3, 0:3]
    f = (10 * tx).astype(np.int32)             # columns 0, 10, 20
    assert reduce_grad(f, _square(0, 0, 3)) == (20, 0)


def test_grad_y_ramp():
    ty, tx = np.mgrid[0:3, 0:3]
    f = (100 * ty).astype(np.int32)            # rows 0, 100, 200
    assert reduce_grad(f, _square(0, 0, 3)) == (0, 200)


def test_grad_mixed_ramp_points_uphill():
    ty, tx = np.mgrid[0:3, 0:3]
    f = (10 * tx + 100 * ty).astype(np.int32)
    assert reduce_grad(f, _square(0, 0, 3)) == (20, 200)


def test_grad_flat_field_is_zero():
    f = np.full((3, 3), 42, dtype=np.int32)
    assert reduce_grad(f, _square(0, 0, 3)) == (0, 0)


def test_grad_single_tile_and_empty():
    f = _coord_field(3, 3)
    assert reduce_grad(f, [(1, 1)]) == (0, 0)
    assert reduce_grad(f, []) == (0, 0)
    assert reduce_grad(f, [(9, 9)]) == (0, 0)   # fully off-grid


def test_grad_clipped_footprint_uses_inbounds_extremes():
    """3x3 anchored at x=-1 on a w=3 x-ramp: in-bounds columns are x=0,1 ->
    gx = mean(col 1) - mean(col 0) = 10; rows all flat in x-ramp -> gy = 0."""
    ty, tx = np.mgrid[0:3, 0:3]
    f = (10 * tx).astype(np.int32)
    assert reduce_grad(f, _square(-1, 0, 3)) == (10, 0)


# ---------------------------------------------------------------------------
# The vocabulary registry + guard hygiene
# ---------------------------------------------------------------------------
def test_vocabulary_is_the_designed_five():
    assert set(REDUCTIONS) == {"center", "max", "mean", "sum", "grad"}
    assert REDUCTIONS["center"] is reduce_center
    assert REDUCTIONS["max"] is reduce_max
    assert REDUCTIONS["mean"] is reduce_mean
    assert REDUCTIONS["sum"] is reduce_sum
    assert REDUCTIONS["grad"] is reduce_grad


def test_no_reduction_raises_on_wild_footprints():
    """The in-bounds guard (combat.py style) absorbs any off-grid tile —
    negative, past-the-edge, or absurd — without raising."""
    f = _coord_field(3, 3)
    wild = [(-100, -100), (0, 0), (1000000, 3), (2, -1)]
    for name, fn in REDUCTIONS.items():
        fn(f, wild)   # must not raise
        fn(f, [])     # must not raise


# ---------------------------------------------------------------------------
# The coupling table — the shipped rows, registered (mechanics/05 §1).
# P1 registered heat + blast; P4 grew the table with the wave_p|grad
# impulse-push row; weapons W3 grew it again with the gas[teargas] and
# gas[poison] rows (the table GROWING is the design's point).
# ---------------------------------------------------------------------------
def test_coupling_table_registers_the_shipped_rows_in_order():
    """Registration: heat, blast, push, teargas, poison (the chapter's row
    order — the P0 'couplings in table order' execution order once the named
    READ slot lands). Rows are plain frozen data."""
    from simulation.exchange import COUPLING_TABLE
    assert isinstance(COUPLING_TABLE, tuple)
    assert [row.field for row in COUPLING_TABLE] == [
        "heat", "wave_p", "wave_p", "gas[teargas]", "gas[poison]"]
    assert all(isinstance(row, CouplingRow) for row in COUPLING_TABLE)


def test_coupling_table_reductions_name_the_vocabulary():
    """A row's reduction is a vocabulary name (or None for a shipped response
    that predates the field read — the blast row's documented state)."""
    from simulation.exchange import COUPLING_TABLE
    heat_row, blast_row, push_row, tear_row, poison_row = COUPLING_TABLE
    assert heat_row.reduction == "max" and heat_row.reduction in REDUCTIONS
    assert blast_row.reduction is None
    assert blast_row.note      # the predates-the-field-read status is written
    assert push_row.reduction == "grad" and push_row.reduction in REDUCTIONS
    # The W3 gas rows read like the heat row: footprint max (the densest gas
    # tile on the body is the exposure that matters).
    assert tear_row.reduction == "max" and tear_row.reduction in REDUCTIONS
    assert poison_row.reduction == "max" and poison_row.reduction in REDUCTIONS


def test_coupling_table_responses_are_the_shipped_implementations():
    """Behaviour preservation at the identity level: the registered response
    callables ARE the shipped functions, and the combat.py compatibility
    re-exports resolve to the very same objects (legacy imports unchanged)."""
    from simulation import combat, exchange
    heat_row, blast_row, push_row, tear_row, poison_row = exchange.COUPLING_TABLE
    assert heat_row.response is exchange.apply_environmental_damage
    assert blast_row.response is exchange.apply_blast_damage
    assert push_row.response is exchange.apply_wave_push
    assert tear_row.response is exchange.apply_teargas_blind
    assert poison_row.response is exchange.apply_poison_dose
    assert combat.apply_environmental_damage is exchange.apply_environmental_damage
    assert combat.apply_blast_damage is exchange.apply_blast_damage
    assert combat.HEAT_SCALE == exchange.HEAT_SCALE == 65536
