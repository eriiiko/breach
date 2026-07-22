"""tools/level_edit_common.py — the Arc C5 wand primitives: enclosure fill
+ same-code select + the hull-leak border check.

Pins:
  - same_code_region: 4-connected flood over VALUE equality (boundary = a
    value change), works on any integer grid (materials, zones.npy);
  - enclosure_fill_region: 4-connected flood over a caller-supplied boolean
    open_mask (boundary = wherever the mask is False), generic over
    materials/zones/air's own boundary definitions;
  - region_touches_border: the hull-leak test — does a region reach the
    outermost ring of the grid;
  - both flood primitives return None on an out-of-bounds/blocked start
    tile, never raise.

Run:
    python -m pytest tests/test_wand_paint.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_edit_common as lec  # noqa: E402


# ---------------------------------------------------------------------------
# same_code_region
# ---------------------------------------------------------------------------

def test_same_code_region_selects_contiguous_equal_run():
    g = np.array([[1, 1, 2],
                  [1, 1, 2],
                  [3, 3, 2]], dtype=np.int32)
    region = lec.same_code_region(g, 0, 0)
    assert region == frozenset({(0, 0), (1, 0), (0, 1), (1, 1)})


def test_same_code_region_stops_at_a_value_change():
    g = np.array([[5, 5, 9]], dtype=np.int32)
    region = lec.same_code_region(g, 0, 0)
    assert (2, 0) not in region
    assert region == frozenset({(0, 0), (1, 0)})


def test_same_code_region_single_isolated_cell():
    g = np.array([[1, 2, 1]], dtype=np.int32)
    assert lec.same_code_region(g, 1, 0) == frozenset({(1, 0)})


def test_same_code_region_out_of_bounds_is_none():
    g = np.zeros((3, 3), dtype=np.int32)
    assert lec.same_code_region(g, -1, 0) is None
    assert lec.same_code_region(g, 3, 0) is None
    assert lec.same_code_region(g, 0, 3) is None


def test_same_code_region_works_on_zone_id_grids():
    zones = np.array([[1, 1, 0],
                      [1, 2, 0],
                      [0, 0, 0]], dtype=np.uint8)
    assert lec.same_code_region(zones, 0, 0) == frozenset(
        {(0, 0), (1, 0), (0, 1)})
    assert lec.same_code_region(zones, 1, 1) == frozenset({(1, 1)})


# ---------------------------------------------------------------------------
# enclosure_fill_region
# ---------------------------------------------------------------------------

def _room_mask(w=6, h=6):
    """An open 4x4 interior ringed by a 1-tile closed border (mask False)."""
    m = np.zeros((h, w), dtype=bool)
    m[1:h - 1, 1:w - 1] = True
    return m


def test_enclosure_fill_region_bounded_by_the_mask():
    m = _room_mask()
    region = lec.enclosure_fill_region(m, 2, 2)
    expected = frozenset((x, y) for y in range(1, 5) for x in range(1, 5))
    assert region == expected
    # never spills onto the False ring
    assert not any(x in (0, 5) or y in (0, 5) for x, y in region)


def test_enclosure_fill_region_start_on_false_is_none():
    m = _room_mask()
    assert lec.enclosure_fill_region(m, 0, 0) is None


def test_enclosure_fill_region_out_of_bounds_is_none():
    m = _room_mask()
    assert lec.enclosure_fill_region(m, -1, 2) is None
    assert lec.enclosure_fill_region(m, 6, 2) is None


def test_enclosure_fill_region_two_separate_rooms_dont_leak_into_each_other():
    m = np.zeros((5, 11), dtype=bool)
    m[1:4, 1:4] = True     # room A: x 1..3
    m[1:4, 7:10] = True    # room B: x 7..9  (x 4,5,6 stay a solid wall)
    region_a = lec.enclosure_fill_region(m, 2, 2)
    region_b = lec.enclosure_fill_region(m, 8, 2)
    assert region_a.isdisjoint(region_b)
    assert len(region_a) == 9 and len(region_b) == 9


def test_enclosure_fill_region_leak_extends_through_a_one_tile_gap():
    """A one-tile gap in the ring connects the interior straight through to
    whatever lies beyond it — enclosure fill has no notion of "the nearest
    room", only connectivity (exactly why the hull-leak validator checks
    the BORDER reachability, not just "did it stay inside the rect I
    expected")."""
    m = _room_mask(8, 8)
    m[0, 3] = True          # breach the top wall at one tile
    region = lec.enclosure_fill_region(m, 4, 4)
    interior = frozenset((x, y) for y in range(1, 7) for x in range(1, 7))
    assert interior <= region                # the whole sealed interior
    assert (3, 0) in region                  # PLUS the breached border tile


# ---------------------------------------------------------------------------
# region_touches_border — the hull-leak test
# ---------------------------------------------------------------------------

def test_region_touches_border_true_when_a_tile_sits_on_the_edge():
    assert lec.region_touches_border(frozenset({(0, 3)}), 10, 10)
    assert lec.region_touches_border(frozenset({(9, 3)}), 10, 10)
    assert lec.region_touches_border(frozenset({(3, 0)}), 10, 10)
    assert lec.region_touches_border(frozenset({(3, 9)}), 10, 10)


def test_region_touches_border_false_for_a_fully_interior_region():
    region = frozenset((x, y) for y in range(1, 5) for x in range(1, 5))
    assert not lec.region_touches_border(region, 6, 6)


def test_region_touches_border_empty_region_is_false():
    assert not lec.region_touches_border(frozenset(), 6, 6)


def test_sealed_room_enclosure_fill_never_touches_border():
    m = _room_mask(8, 8)
    region = lec.enclosure_fill_region(m, 4, 4)
    assert not lec.region_touches_border(region, 8, 8)


def test_leaky_room_enclosure_fill_touches_border():
    m = _room_mask(8, 8)
    m[0, 3] = True           # breach straight to the top edge
    region = lec.enclosure_fill_region(m, 4, 4)
    assert lec.region_touches_border(region, 8, 8)
