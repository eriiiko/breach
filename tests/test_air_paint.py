"""tools/air_paint.py — the AIR wand tool's hull-leak-gated enclosure fill
(Arc C5, editor doc §7).

Pins:
  - default_ambient_grid: FP_ONE (1.0 atm) EVERYWHERE — a brand-new
    air_init.npy must seed the WHOLE map to the engine's own pre-override
    default, or painting one room would silently zero-pressure the rest;
  - quantize_atm: round-half-up float atm -> Q16.16 int, clamped >= 0;
  - plan_air_fill: a SEALED fill (bounded by solid, never reaching the
    border) returns its region; a LEAKY fill (reaches the border, walked
    THROUGH vacuum/SPACE — the tools/level_airtight.py connectivity)
    returns None and paints NOTHING (the caller never receives a region to
    paint from a refused fill); a solid start tile also refuses.

Run:
    python -m pytest tests/test_air_paint.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import air_paint  # noqa: E402
from simulation.materials import MAT_AIR, MAT_HULL  # noqa: E402

SPACE_CODE = 9
SOLID = frozenset({MAT_HULL})


def _sealed_room(w=8, h=8):
    """A hull-ringed room: MAT_HULL border, MAT_AIR interior — no vacuum
    anywhere, so any interior fill is sealed."""
    g = np.full((h, w), MAT_AIR, dtype=np.int32)
    g[0, :] = g[-1, :] = MAT_HULL
    g[:, 0] = g[:, -1] = MAT_HULL
    return g


def _leaky_room(w=8, h=8):
    """The same room with a one-tile breach straight to a SPACE ring one
    tile further out (space beyond the hull) — the fill must walk THROUGH
    the breach and the vacuum tile to reach the true grid edge."""
    g = np.full((h, w), MAT_AIR, dtype=np.int32)
    g[1, :] = g[-2, :] = MAT_HULL
    g[:, 1] = g[:, -2] = MAT_HULL
    g[0, :] = g[-1, :] = SPACE_CODE
    g[:, 0] = g[:, -1] = SPACE_CODE
    g[1, w // 2] = MAT_AIR              # breach: punch the top hull ring
    return g


# ---------------------------------------------------------------------------
# default_ambient_grid / quantize_atm
# ---------------------------------------------------------------------------

def test_default_ambient_grid_is_fp_one_everywhere():
    g = air_paint.default_ambient_grid((5, 7))
    assert g.shape == (5, 7)
    assert g.dtype == np.int32
    assert (g == air_paint.FP_ONE).all()


def test_quantize_atm_round_half_up_and_clamped_nonnegative():
    assert air_paint.quantize_atm(1.0) == air_paint.FP_ONE
    assert air_paint.quantize_atm(0.0) == 0
    assert air_paint.quantize_atm(-3.0) == 0            # clamped, no negatives
    assert air_paint.quantize_atm(2.0) == 2 * air_paint.FP_ONE


# ---------------------------------------------------------------------------
# plan_air_fill — the hull-leak validator
# ---------------------------------------------------------------------------

def test_plan_air_fill_sealed_room_returns_the_interior_region():
    g = _sealed_room()
    region, why = air_paint.plan_air_fill(g, SOLID, 4, 4)
    assert why == "ok"
    assert region is not None
    # every returned tile is open interior air, none of it hull
    for tx, ty in region:
        assert int(g[ty, tx]) not in SOLID
    assert (1, 1) in region and (6, 6) in region


def test_plan_air_fill_leaky_room_refuses_and_returns_no_region():
    g = _leaky_room()
    region, why = air_paint.plan_air_fill(g, SOLID, 4, 4)
    assert region is None
    assert "border" in why


def test_plan_air_fill_start_on_solid_is_refused():
    g = _sealed_room()
    region, why = air_paint.plan_air_fill(g, SOLID, 0, 0)   # a hull tile
    assert region is None
    assert "solid" in why


def test_plan_air_fill_start_out_of_bounds_is_refused():
    g = _sealed_room()
    region, why = air_paint.plan_air_fill(g, SOLID, -1, 0)
    assert region is None


def test_plan_air_fill_never_mutates_the_grid():
    g = _leaky_room()
    before = g.copy()
    air_paint.plan_air_fill(g, SOLID, 4, 4)
    assert np.array_equal(g, before)     # a pure query — nothing painted
