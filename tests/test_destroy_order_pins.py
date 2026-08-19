"""P-M4c — Gate 8 / 8b: ``destroy_wall``'s ORDER dependence, pinned.

Design: ``docs/mass_books_pm3_destroy_wall_seed_design_2026-08-18.md`` §3.7,
§6 gates 8 and 8b. The value/booking half of P-M3 is gated in
``tests/test_destroy_wall_conserves_mass.py``; this file gates the half that
determinism actually rests on.

**Why this file exists.** ``destroy_wall`` is order-dependent in two ways:

*Decision-level.* It writes ``breach_mask[fy, fx] = True`` and the NEXT tile's
``exposes`` test reads the LIVE mask — so whether tile B is seeded at all
depends on whether adjacent tile A was destroyed first. A step function per
tile.

*Value-level.* ``on_tile_changed`` clears ``solid`` BEFORE the seed runs, so
B's donor set can include A's freshly-seeded composition. Same total,
different O2/N2 split, different combustion.

Determinism is a HARD requirement (multiplayer, distributed training), and it
holds today only because four independent callers each PIN their iteration
order. Historical stakes: ``cpp/src/cuda_fire.cu:58-70`` records that an
UNSORTED burn-through list was a measured CPU != GPU *and* GPU != GPU
divergence; the repair is the host-side sort at ``cuda_fire.cu:482``.

The four pins, and where each is gated:

===================================  =========================================
pin                                  gate
===================================  =========================================
CUDA burn-through list sort          ``tests/cuda_fire_check.py`` (pre-existing)
``find_burst_walls`` descending sort **here** (Gate 8a)
``physics.apply_explosion`` dy/dx    **here** (Gate 8b)
``door_system`` row-major span       **here** (Gate 8c; the span's OWN sort at
                                     ``entities/door.py:164`` is gated by
                                     ``test_a6_doors.py::
                                     test_span_orientation_and_res_replication``
                                     — what was ungated is the EMISSION order)
===================================  =========================================

Every gate here asserts the ORDER, not the set, and every one carries an
in-test proof of its own discriminating power: alongside the expected
sequence it asserts that the sequence a plausible REGRESSION would produce
(row-major instead of sorted-by-spread; dx-outer instead of dy-outer;
reversed span) is genuinely DIFFERENT for that scene. A gate whose expected
value coincides with the regression's cannot go red, and this arc has already
produced three of those.

Gate 8b (§6) is the per-tile split under reordering. It compares the per-tile
``(gas[O2], gas[INERT_N2])`` PAIR and never ``Sum N`` — the seeded total is a
constant per tile, so ``Sum N`` is order-invariant BY CONSTRUCTION and an
assertion on it cannot fail. See ``test_gate8b_*`` for the measured answer.

**Mutation record (P-M4c, run at ae7f0cb).** Every gate here was shown red by
temporarily breaking the pin it guards, then reverted:

===================================================  =========================
mutation applied to production                       tests that went RED
===================================================  =========================
``gamemap.py``: drop ``failing.sort(...)``           8a worst-first, 8a cap
``gamemap.py``: drop the sort's ``reverse=True``     8a worst-first, 8a cap
``gamemap.py``: ``np.where(solid.T)`` (transposed)   8a equal-spreads
``physics.py``: swap the ``dy`` / ``dx`` nesting     8b explosion row-major
``door_system.py``: ``reversed(d.span)``             8c horizontal, 8c vertical
``gamemap.py``: drop the donor inheritance           8b pair, 8b L-group
``gamemap.py``: ``exposes`` edge-hull-only           8b decision-level
===================================================  =========================

Run:
    conda run -n data python -m pytest tests/test_destroy_order_pins.py -q
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                                    # noqa: E402
from level_loader import EntityInstance, LevelData             # noqa: E402
from simulation import Simulation                              # noqa: E402
from simulation import atmosphere_fixed as afx                 # noqa: E402
from simulation.door_system import sweep_doors                 # noqa: E402
from simulation.entities import door as door_mod               # noqa: E402
from simulation.events import DoorDestroyedEvent               # noqa: E402
from simulation.field_edit import EditQueue                    # noqa: E402
from simulation.gamemap import (                               # noqa: E402
    GameMap, MAT_AIR, MAT_WOOD,
)
from simulation.gases import O2, INERT_N2                      # noqa: E402
from simulation.physics import apply_explosion                 # noqa: E402

SEED = 20260818
SPACE_CODE = 9          # v2 tilemap: outer space (MAT_AIR + vacuum)
HULL_CODE = 1


# ---------------------------------------------------------------------------
# Recording harness — the ONE way this file observes a caller's emission order
# ---------------------------------------------------------------------------
def record_destructions(gmap):
    """Patch ``gmap.destroy_wall`` to log every call, in call order, and
    still do the real work. Returns the (live) list."""
    seen: list[tuple[int, int]] = []
    inner = gmap.destroy_wall

    def _spy(fy, fx):
        seen.append((int(fy), int(fx)))
        return inner(fy, fx)

    gmap.destroy_wall = _spy
    return seen


def row_major(tiles):
    return sorted(tiles)


def col_major(tiles):
    return sorted(tiles, key=lambda t: (t[1], t[0]))


# ---------------------------------------------------------------------------
# Level fixtures
# ---------------------------------------------------------------------------
def _v2_sealed_box(h=12, w=12, name="pins_box") -> LevelData:
    """v2 vocabulary, NO vacuum anywhere: hull ring, interior air."""
    tm = np.full((h, w), HULL_CODE, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = MAT_AIR
    return LevelData(name=name, version="2", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."))


def _v2_hull_box(h=14, w=14, edits=(), name="pins_hull_box") -> LevelData:
    """A SPACE map: vacuum ring, hull box inside it, interior air.
    (Same shape as ``test_destroy_wall_conserves_mass._hull_box_level``.)"""
    tm = np.full((h, w), SPACE_CODE, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = HULL_CODE
    tm[2:h - 2, 2:w - 2] = MAT_AIR
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name=name, version="2", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."))


def _all_hull(h=14, w=14, name="pins_all_hull") -> LevelData:
    """v1 vocabulary, every tile hull — the blast scene, so the disc's
    membership is pure geometry and never a material accident."""
    return LevelData(name=name, version="1", path=Path("."),
                     tilemap=np.ones((h, w), dtype=np.int32),
                     tile_size_m=1.0, diffuse_path=Path("."))


def _gmap(level):
    return GameMap(level)


def _sim(level, physics=bp):
    return Simulation(level, seed=SEED, breach_physics=physics,
                      enable_recorder=False)


# ===========================================================================
# GATE 8a — find_burst_walls: descending spread over a row-major scan
# ===========================================================================
#
# The scene: a 1-tile-thick vertical WOOD partition at x = 6 (wood ships
# burst_threshold = 2.0; hull ships 0.0 == never-bursts, so hull cannot be
# used here). Each partition tile's spread is `left_room_P - 1.0` for its own
# row, so the ten tiles carry ten different differentials and the sort has
# something to do.
_BURST_X = 6
_BURST_LEFT_ATM = {1: 3.0, 2: 8.0, 3: 4.0, 4: 9.0, 5: 1.2,
                   6: 6.0, 7: 5.0, 8: 7.0, 9: 3.5, 10: 2.0}
_BURST_RIGHT_ATM = 1.0
# row:      1     2    3    4    5     6    7    8    9    10
# spread:   2.0*  7.0  3.0  8.0  0.2*  5.0  4.0  6.0  2.5  1.0*
# (* not strictly ABOVE the 2.0 threshold -> excluded. The test asserts
#  membership too, so a threshold regression is visible, but the POINT of the
#  gate is the sequence.)


def _burst_scene(left_atm, h=12, w=12):
    gmap = _gmap(_v2_sealed_box(h, w, name="pins_burst"))
    for y in range(1, h - 1):
        gmap.material[y, _BURST_X] = MAT_WOOD
        gmap.on_tile_changed(y, _BURST_X)
        gmap.atmosphere[y, _BURST_X] = 0        # a solid tile's P is a hard 0
    assert gmap.solid[5, _BURST_X], "the partition must be solid wood"
    assert not gmap.is_vacuum.any(), "no vacuum: every side is a real pressure"
    for y in range(1, h - 1):
        for x in range(1, _BURST_X):
            gmap.atmosphere[y, x] = afx.quantize(np.float32(left_atm[y]))
        for x in range(_BURST_X + 1, w - 1):
            gmap.atmosphere[y, x] = afx.quantize(np.float32(_BURST_RIGHT_ATM))
    return gmap


def test_gate8a_burst_walls_emit_worst_differential_first():
    """``find_burst_walls`` returns its tiles in DESCENDING spread order.

    This is the pin ``destroy_wall``'s decision-level coupling rests on for
    slot 9b: the burst valve destroys the returned list in the returned
    order, so a wall that pops FIRST can flip its neighbour's ``exposes``
    test and its donor composition.
    """
    gmap = _burst_scene(_BURST_LEFT_ATM)
    got = gmap.find_burst_walls()

    expected = [(4, 6), (2, 6), (8, 6), (6, 6), (7, 6), (3, 6), (9, 6)]
    assert got == expected

    # --- proof the assertion can go red -------------------------------------
    # If the `failing.sort(key=..., reverse=True)` at gamemap.py were dropped,
    # the list would come out in the `np.where(solid)` row-major scan order;
    # if `reverse=True` were dropped it would come out ascending. BOTH differ
    # from `expected` for this scene, so the assertion above discriminates.
    assert got != row_major(got), "scene must not be sort-order-degenerate"
    assert got != list(reversed(got)), "ascending must differ from descending"
    assert got == sorted(
        got, key=lambda t: -_BURST_LEFT_ATM[t[0]]), "descending by spread"


def test_gate8a_burst_cap_takes_the_WORST_not_the_first_scanned():
    """``max_pops`` slices AFTER the sort, so the per-tick cap keeps the
    worst differentials — not whichever tiles the row-major scan reached
    first. Slicing before the sort would be a silent determinism change with
    identical set semantics at the uncapped call."""
    gmap = _burst_scene(_BURST_LEFT_ATM)
    assert gmap.find_burst_walls(max_pops=3) == [(4, 6), (2, 6), (8, 6)]
    # The scene's discriminating power, again in-test: a cap applied to the
    # SCAN order would have returned the three lowest rows instead.
    assert row_major(gmap.find_burst_walls())[:3] == [(2, 6), (3, 6), (4, 6)]


def _tie_scene(h=12, w=12):
    """TWO parallel wood partitions (x = 4 and x = 8) carrying the IDENTICAL
    4.0 spread, so the returned order is decided purely by the scan.

    Two columns rather than one on purpose: with a single column the
    row-major and column-major scans coincide and the tie leg would be
    unfalsifiable against a transposed scan."""
    gmap = _gmap(_v2_sealed_box(h, w, name="pins_burst_tie"))
    for x in (4, 8):
        for y in range(1, h - 1):
            gmap.material[y, x] = MAT_WOOD
            gmap.on_tile_changed(y, x)
            gmap.atmosphere[y, x] = 0
    # rooms: x 1..3 = 5 atm, x 5..7 = 1 atm, x 9..10 = 5 atm -> spread 4.0 on
    # BOTH partitions, every row.
    for y in range(1, h - 1):
        for x, p in ((1, 5.0), (2, 5.0), (3, 5.0), (5, 1.0), (6, 1.0),
                     (7, 1.0), (9, 5.0), (10, 5.0)):
            gmap.atmosphere[y, x] = afx.quantize(np.float32(p))
    return gmap


def test_gate8a_equal_spreads_keep_the_row_major_scan_order():
    """Ties resolve to the ``np.where(solid)`` row-major scan order — the
    sort is STABLE and the scan is row-major, and both halves are load
    bearing. (Equal spreads are the common real case: a straight run of wall
    between two rooms.)

    This leg cannot catch a DROPPED sort — with all keys equal a stable sort
    is the identity, and that is exactly why the distinct-spread test above
    exists. It catches the other half: a scan that stops being row-major
    (``np.where(solid.T)``, a set, a dict over materials)."""
    gmap = _tie_scene()
    got = gmap.find_burst_walls()

    expected = [(y, x) for y in range(1, 11) for x in (4, 8)]
    assert got == expected
    assert got == row_major(got)

    # --- proof the assertion can go red -------------------------------------
    # The scene spans two columns, so a transposed scan is a different
    # sequence over the identical set.
    assert got != col_major(got)
    assert got != list(reversed(got))
    # The cap then takes the first-SCANNED tiles, deterministically.
    assert gmap.find_burst_walls(max_pops=4) == [(1, 4), (1, 8), (2, 4), (2, 8)]


# ===========================================================================
# GATE 8b(caller) — physics.apply_explosion: dy OUTER, dx INNER (row-major)
# ===========================================================================
def test_gate8b_explosion_destroys_in_row_major_dy_outer_dx_inner():
    """``apply_explosion``'s nested ``for dy: for dx:`` loop emits its
    ``destroy_wall`` calls in strict row-major order over the blast disc.

    Both loops ascend from ``-radius``, so the emission order is exactly
    ``sorted()``. Swapping the nesting (``for dx: for dy:``) keeps the SET
    identical and every ``Sum N`` identical — and changes which tile's
    ``breach_mask`` write and freshly-seeded composition the next tile sees.
    That is precisely the class of change no existing gate could see."""
    gmap = _gmap(_all_hull())
    gmap.wall_hp[:] = 1            # any positive damage destroys
    seen = record_destructions(gmap)

    apply_explosion(gmap, EditQueue(), 7, 7, radius=3, pressure=1.0,
                    wall_damage=100.0)

    expected = [(y, x) for y in range(5, 10) for x in range(5, 10)]
    assert seen == expected
    assert len(set(seen)) == len(seen), "no tile destroyed twice"

    # --- proof the assertion can go red -------------------------------------
    # The scene is 2-D and wider than one row, so the transposed (dx-outer)
    # enumeration of the SAME set is a genuinely different sequence.
    assert seen == row_major(seen)
    assert seen != col_major(seen), "dx-outer nesting must be distinguishable"
    assert seen != list(reversed(seen))


def test_gate8b_explosion_order_is_stable_across_identical_calls():
    """Two identical blasts emit the identical sequence — the loop carries no
    iteration-order hazard (no set, no dict, no argsort tie)."""
    def run():
        gmap = _gmap(_all_hull())
        gmap.wall_hp[:] = 1
        seen = record_destructions(gmap)
        apply_explosion(gmap, EditQueue(), 6, 8, radius=4, pressure=1.0,
                        wall_damage=100.0)
        return seen

    a, b = run(), run()
    assert a == b and len(a) > 20


# ===========================================================================
# GATE 8c — door_system.sweep_doors: the row-major span, in span order
# ===========================================================================
def _door_inst(eid, ordinal, x, y, orientation="h", length_m=1.0,
               initial_state="closed"):
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation=orientation, length_m=length_m,
                  initial_state=initial_state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal,
                          fields=fields)


def _wide_door_level(h=12, w=12):
    """v1 vocabulary (hull ring 1, interior air 4) with a horizontal interior
    hull wall at y = 6 and a 4-tile doorway at x = 3..6."""
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    tm[6, 1:w - 1] = 1
    tm[6, 3:7] = 4
    return LevelData(name="pins_door", version="1", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."),
                     entities=[_door_inst("d", 0, x=3, y=6, orientation="h",
                                          length_m=4.0)])


def test_gate8c_door_assembly_death_destroys_in_row_major_span_order():
    """Slot 9e's whole-door rule destroys the REMAINING intact span tiles in
    ``d.span`` order, which ``entities/door.py:164`` pins to row-major.

    A 4-tile door with its THIRD tile already destroyed externally is the
    discriminating case: the emitted sequence is neither the full span nor a
    contiguous run, so a reversed or set-derived iteration is visible."""
    sim = _sim(_wide_door_level(), physics=None)
    (door,) = sim._doors
    gmap = sim.gmap
    assert door.span == [(6, 3), (6, 4), (6, 5), (6, 6)], "row-major span"

    gmap.destroy_wall(6, 5)                  # the external destruction (§8)
    seen = record_destructions(gmap)         # spy AFTER, so only 9e is logged
    sweep_doors(sim)

    assert not door.alive and door.hp == [0, 0, 0, 0]
    expected = [(6, 3), (6, 4), (6, 6)]
    assert seen == expected

    # --- proof the assertion can go red -------------------------------------
    assert seen == row_major(seen)
    assert seen != list(reversed(seen)), "reverse iteration must differ"
    assert seen != door.span, "the pre-destroyed tile is not re-emitted"

    # The event stream carries the same order (the S3 contract's 9e half).
    positions = [e.pos for e in sim.tick_events
                 if isinstance(e, DoorDestroyedEvent)]
    assert positions == expected


def test_gate8c_vertical_door_span_order_is_top_to_bottom():
    """The same pin on the other orientation — ``base_span`` walks +y for
    "v", so a vertical door dies top-to-bottom, never bottom-up."""
    h = w = 12
    tm = np.ones((h, w), dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = 4
    tm[1:h - 1, 6] = 1
    tm[4:8, 6] = 4
    lvl = LevelData(name="pins_vdoor", version="1", path=Path("."), tilemap=tm,
                    tile_size_m=1.0, diffuse_path=Path("."),
                    entities=[_door_inst("d", 0, x=6, y=4, orientation="v",
                                         length_m=4.0)])
    sim = _sim(lvl, physics=None)
    (door,) = sim._doors
    gmap = sim.gmap
    assert door.span == [(4, 6), (5, 6), (6, 6), (7, 6)]

    gmap.destroy_wall(6, 6)
    seen = record_destructions(gmap)
    sweep_doors(sim)

    expected = [(4, 6), (5, 6), (7, 6)]
    assert seen == expected
    assert seen == row_major(seen)
    assert seen != list(reversed(seen))


# ===========================================================================
# GATE 8b (§6) — the PER-TILE SPLIT under reordering
# ===========================================================================
#
# MEASURED ANSWER (P-M4c): the splits DO differ, exactly as §3.7's value-level
# coupling predicts — but ONLY when the donor rooms carry different
# compositions. In a room of UNIFORM composition the inherited fraction is a
# FIXED POINT of the seed (a tile seeded from fraction f holds fraction f, so
# it is an indistinguishable donor), and every order gives the identical
# split. That is why this gate builds a deliberately non-uniform scene: a
# uniform-composition test of the same proposition would be the fourth
# unfalsifiable gate of this arc.
#
# The gate PINS the production (row-major) order's result and records the
# reordered result as a documented, measured difference. Production does not
# have order-independence here and does not need it — it has FOUR pinned
# caller orders instead, gated above.

_LEFT_O2, _LEFT_N2 = 26214, 39322        # 40 / 60 of FP_ONE
_RIGHT_O2, _RIGHT_N2 = 6554, 58982       # 10 / 90 of FP_ONE
_SPLIT_COL = 7


def _two_composition_scene(wall_tiles):
    """Hull box whose open air is 40/60 O2 left of column 7 and 10/90 right
    of it, with ``wall_tiles`` stamped as destructible hull."""
    edits = [(y, x, HULL_CODE) for (y, x) in wall_tiles]
    gmap = _sim(_v2_hull_box(edits=edits, name="pins_split")).gmap
    open_air = ~(gmap.solid | gmap.is_vacuum)
    cols = np.broadcast_to(np.arange(gmap.solid.shape[1]), gmap.solid.shape)
    left = open_air & (cols < _SPLIT_COL)
    right = open_air & (cols >= _SPLIT_COL)
    gmap.gas[O2][:] = np.where(left, _LEFT_O2, np.where(right, _RIGHT_O2, 0))
    gmap.gas[INERT_N2][:] = np.where(left, _LEFT_N2,
                                     np.where(right, _RIGHT_N2, 0))
    for (y, x) in wall_tiles:
        assert gmap.solid[y, x], "the pair must start as solid wall"
        assert not gmap.is_vacuum[y, x]
    return gmap


def _pair(gmap, t):
    return int(gmap.gas[O2][t]), int(gmap.gas[INERT_N2][t])


def _bulk_n(gmap):
    return (int(gmap.gas[O2].astype(np.int64).sum())
            + int(gmap.gas[INERT_N2].astype(np.int64).sum()))


def _destroy_in(order, wall_tiles):
    gmap = _two_composition_scene(wall_tiles)
    n_before = _bulk_n(gmap)
    for t in order:
        gmap.destroy_wall(*t)
    return ({t: _pair(gmap, t) for t in wall_tiles},
            _bulk_n(gmap) - n_before,
            int(gmap.n_destruction_seed_sum))


_A, _B = (6, 6), (6, 7)


def test_gate8b_adjacent_pair_split_DIFFERS_under_reordering():
    """The measured value-level coupling, at the per-tile ``(O2, N2)`` pair.

    A = (6, 6) sits in the 40/60 room; B = (6, 7) in the 10/90 room. Whichever
    is destroyed FIRST inherits purely from its own room's donors; the second
    then inherits from three of its own donors PLUS the first tile, which is
    no longer solid and now holds a full ambient cell of the OTHER room's
    composition. Exact arithmetic (``o2_q = (65536*sum_o2 + sum_n//2)//sum_n``):

        A then B:  A = 26214/39322 (3 left donors)
                   B = 11469/54067 (3 right donors + A)
        B then A:  B =  6554/58982 (3 right donors)
                   A = 21299/44237 (3 left donors + B)

    Production order for the row-major callers (blast, door) is A then B, and
    that is what this gate PINS."""
    prod, d_prod, chan_prod = _destroy_in([_A, _B], [_A, _B])
    rev, d_rev, chan_rev = _destroy_in([_B, _A], [_A, _B])

    # The production order's result — the pin.
    assert prod == {_A: (26214, 39322), _B: (11469, 54067)}
    # The reordered result — measured, documented, DIFFERENT.
    assert rev == {_A: (21299, 44237), _B: (6554, 58982)}
    assert prod[_A] != rev[_A] and prod[_B] != rev[_B]

    # ...and the reason a Sum-N gate could never have caught it: the total is
    # a constant per seeded tile, so it is order-invariant BY CONSTRUCTION.
    assert d_prod == d_rev == 2 * 65536
    assert chan_prod == chan_rev == d_prod
    for pair in (*prod.values(), *rev.values()):
        assert sum(pair) == 65536, "the constant total, on every ordering"
        assert pair[0] >= 0 and pair[1] >= 0


def test_gate8b_uniform_composition_is_the_degenerate_case():
    """Why the gate above needs two compositions.

    With one composition everywhere, a seeded tile holds exactly the donor
    fraction, so it is an INDISTINGUISHABLE donor for its neighbour and every
    order gives the identical split. An order-sensitivity gate written on a
    uniform scene would be unfalsifiable — it would stay green against any
    reordering of any caller. Pinned here so nobody 'simplifies' the scene
    above back into one."""
    def run(order):
        gmap = _sim(_v2_hull_box(edits=[(y, x, HULL_CODE) for (y, x)
                                        in (_A, _B)],
                                 name="pins_uniform")).gmap
        # Non-vacuity: the donors must actually HOLD gas, or this would be
        # green because the `sum_n == 0` fallback fired, not because the
        # inherited fraction is a fixed point.
        assert _pair(gmap, (5, 6))[0] > 0 and _pair(gmap, (6, 5))[0] > 0
        for t in order:
            gmap.destroy_wall(*t)
        return {t: _pair(gmap, t) for t in (_A, _B)}

    assert run([_A, _B]) == run([_B, _A])
    assert run([_A, _B]) == {_A: (13763, 51773), _B: (13763, 51773)}


_L_GROUP = [(6, 6), (6, 7), (7, 7)]


def test_gate8b_L_group_has_three_distinct_outcomes_over_six_orders():
    """The L-shaped group: 6 permutations, 3 distinct per-tile outcomes, ONE
    Sum N. The coupling is not a two-tile curiosity — it compounds through a
    connected destruction front, which is exactly what a burst run, a blast
    disc and a door span all are.

    The row-major (production) outcome is pinned; the count of distinct
    outcomes is what makes the proposition falsifiable — a change that broke
    the donor inheritance into an order-free form would collapse it to 1."""
    outcomes = {}
    totals = set()
    for order in itertools.permutations(_L_GROUP):
        pairs, delta, chan = _destroy_in(list(order), _L_GROUP)
        outcomes.setdefault(tuple(pairs[t] for t in _L_GROUP), []).append(order)
        totals.add((delta, chan))

    assert len(outcomes) == 3, f"expected 3 distinct splits, got {outcomes}"
    # Sum N cannot tell any of them apart.
    assert totals == {(3 * 65536, 3 * 65536)}

    prod, _d, _c = _destroy_in(row_major(_L_GROUP), _L_GROUP)
    assert prod == {(6, 6): (26214, 39322),
                    (6, 7): (13107, 52429),
                    (7, 7): (13107, 52429)}
    rev, _d, _c = _destroy_in(list(reversed(row_major(_L_GROUP))), _L_GROUP)
    assert rev == {(6, 6): (21845, 43691),
                   (6, 7): (8738, 56798),
                   (7, 7): (13107, 52429)}
    assert prod != rev


def test_gate8b_decision_level_reordering_moves_even_the_TOTAL():
    """§3.7's decision-level coupling — the strictly larger sensitivity.

    V = (1, 6) is a hull-ring tile whose (0, 6) neighbour is the vacuum ring;
    W = (2, 6) is a hull tile behind it with no vacuum of its own. Destroying
    V first sets ``is_vacuum[V]``, and W's ``exposes`` test then reads the
    LIVE mask and finds one — so W joins the boundary and is seeded with
    NOTHING. Destroy W first and it is an interior tile: a full ambient cell.

    A step function per tile, and unlike the value-level coupling it moves
    ``Sum N`` too (by a whole seed), so it is booked differently as well.
    The row-major production order (V then W) is what this gate pins."""
    V, W = (1, 6), (2, 6)

    def run(order):
        gmap = _sim(_v2_hull_box(edits=[(W[0], W[1], HULL_CODE)],
                                 name="pins_decision")).gmap
        assert gmap.solid[V] and gmap.solid[W] and gmap.is_vacuum[0, 6]
        n_before = _bulk_n(gmap)
        for t in order:
            gmap.destroy_wall(*t)
        return (_pair(gmap, V), _pair(gmap, W),
                bool(gmap.is_vacuum[V]), bool(gmap.is_vacuum[W]),
                _bulk_n(gmap) - n_before, int(gmap.n_destruction_seed_sum))

    prod = run([V, W])           # production: row-major
    rev = run([W, V])

    assert prod == ((0, 0), (0, 0), True, True, 0, 0)
    assert rev == ((0, 0), (13763, 51773), True, False, 65536, 65536)
    assert prod != rev
    # Sum N DOES move here — the decision-level coupling is the bigger one.
    assert prod[4] != rev[4]


def test_gate8b_a_pinned_order_is_bit_reproducible():
    """Closing the loop: with the order pinned, the whole destruction is
    bit-reproducible run to run — which is the property the four caller pins
    exist to buy."""
    def run():
        gmap = _two_composition_scene(_L_GROUP)
        for t in row_major(_L_GROUP):
            gmap.destroy_wall(*t)
        return (gmap.gas[O2].copy(), gmap.gas[INERT_N2].copy(),
                int(gmap.n_destruction_seed_sum))

    a_o2, a_n2, a_ch = run()
    b_o2, b_n2, b_ch = run()
    assert np.array_equal(a_o2, b_o2) and np.array_equal(a_n2, b_n2)
    assert a_ch == b_ch
