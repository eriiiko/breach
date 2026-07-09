"""EOS refactor P2 — the sealed-room energy E2E (docs/eos_refactor_design.md
§4, §8 patch P2's DEFINING gate).

P2 unifies `temperature` across gas + solid (decisions.md item 7): gas-T lives
in the SAME Q16.16 array, on the SAME ambient-relative scale, and the ONLY
new physical link between the two is the EXISTING whole-grid conduction pass
now doing real work at the solid<->air interface (air gets a small nonzero
`conductivity`, engine/06 §2). This module is the "does it actually behave
like one connected thermal system" proof:

  (a) SEALED ROOM (no vacuum adjacency): seed a hot gas pocket via the NEW
      radiation-deposit path (heat -> gas ΔT, §4.3), then run hundreds of
      ticks with NO further heat input. Total thermal content (== the plain
      sum of the unified `temperature` field over gas + solid — see "why a
      plain sum" below) must be conserved up to a SMALL, PROVEN, BOUNDED
      integer-truncation drift, and the walls adjacent to the pocket must
      visibly warm up (temperature rises from 0).
  (b) Same room, ONE hull tile additionally exposed to vacuum (a genuine
      open vacuum cell placed just outside that one tile, mirroring
      test_temperature_cooling.py's own vacuum-exposure convention — the
      hull tile ITSELF stays solid+non-vacuum; a real vacuum neighbour cell
      is what flags it "exposed"). Total energy must be MONOTONICALLY
      non-increasing every tick (the hull radiates to space via
      cool_shift_vacuum) and must show a REAL, non-negligible net drop over
      the run — i.e. gas + interior-solid energy actually drains through
      that one wall, not just numerical noise.

ISOLATING WHAT P2 ACTUALLY CHANGED (documented, load-bearing test design
choice): the pre-existing SOLID interior Newtonian decay (`cool_shift`,
`T -= T >> cool_shift`, unconditional on EVERY solid tile regardless of
vacuum exposure) is explicitly OUT OF P2's SCOPE — the patch instructions
pin it "unchanged for solids" (only gas cells are newly excluded from
cooling). That decay is a real, sizeable, INTENTIONAL legacy mechanism (the
"burn-out" feature), not integer-truncation noise, so a conservation gate
that left it active would be swamped by an orthogonal, pre-existing effect
and would prove nothing about P2. Both scenarios below PIN `cool_shift` to a
huge value (31, the exact "T >> 31 == 0 for any in-range T" idiom already
used throughout test_temperature_conduction.py / test_temperature_cooling.py
to disable a pass under test) so the gate isolates and directly measures the
NEW P2 mechanisms: gas advection (proven below to be an EXACT identity at
wind==0, so it contributes zero drift here — the test still calls step()
with real wind/dt arrays, exercising the actual production code path), the
NEW gas radiation deposit, and the UNIFIED conduction pass (including its
newly-live air<->air / air<->solid faces). Scenario (b) leaves
`cool_shift_vacuum` at its real, fast, shipped-scale value — THAT is the one
mechanism under test there.

WHY A PLAIN SUM IS THE RIGHT "TOTAL ENERGY" METRIC: the conduction pass
(temperature_solver.cpp Pass 2, UNCHANGED CODE) exchanges `(T_j - T_i) >> s`
between cells i/j sharing a face — a temperature-difference relaxation, not a
heat-capacity-weighted one (this scheme was already shipped for solid<->solid
faces; P2 does not change its shape, only widens which faces are non-NO_FACE).
Since both gas and solid cells now live in the exact same array/scale, "total
thermal content" is naturally the flat sum `temperature.sum()` — exactly the
metric test_temperature_conduction.py's discrete-maximum-principle test and
test_temperature_cooling.py's test_integration_inject_then_burn_out already
use for solids alone; P2 extends it to the unified field for free.

WHY THE EPSILON BOUND IS PROVABLE, NOT A GUESS: for a conducting face pair
(i, j) sharing shift s, cell i gains `floor(d / 2^s)` (d = T_j - T_i, C++'s
arithmetic right shift == floor for signed ints) and cell j gains
`floor(-d / 2^s)`. The identity `floor(x) + floor(-x) == 0` if x is an
integer, else `-1`, means EVERY conducting pair contributes EITHER 0 or
EXACTLY -1 to the grid-wide sum each tick — NEVER positive. So Sigma(T) is
PROVABLY monotonically non-increasing under conduction alone, and the total
per-tick loss is bounded by the number of live (non-NO_FACE) conducting face
PAIRS. epsilon_bound = pairs * n_ticks is therefore a real, derived upper
bound on the drift, not an empirical tolerance — see `_epsilon_bound` below.

Run:
    C:/Users/steen/miniconda3/envs/data/python.exe -m pytest tests/test_eos_p2_sealed_room_energy.py -q
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build" / "Release"))

import breach_physics as bp  # noqa: E402

from simulation.materials import MAT_AIR, MAT_HULL, MaterialTable  # noqa: E402

HEAT_SCALE = 65536
FP_ONE = 65536

_TBL = MaterialTable.from_config()
NO_FACE = int(_TBL.no_face)
_FACE_DIRS = ((-1, 0), (1, 0), (0, 1), (0, -1))   # N,S,E,W — must match the C++ DY,DX


def _build_caches(material_grid):
    """(heat_inv_shift, face_shift, solid) for a 2D material-id grid — the
    SAME derivation GameMap._update_caches uses (ported from
    test_temperature_conduction.py's own helper; kept local so this module
    has no cross-test-file coupling)."""
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


def _epsilon_bound(face, n_ticks):
    """Provable upper bound on |Sigma(T) drift| from conduction's per-pair
    arithmetic-shift truncation over n_ticks (see module docstring). Each
    live (non-NO_FACE) DIRECTED face entry has a symmetric partner (the
    harmonic-mean face table is symmetric — test_temperature_conduction.py's
    own test_face_table_anchor_values pins this); directed_count // 2 is the
    number of undirected conducting PAIRS, each losing at most 1 count/tick."""
    directed_count = int(np.sum(face != NO_FACE))
    pairs = directed_count // 2
    return pairs * n_ticks


def _room_8x8():
    """An 8x8 sealed room: a 1-tile MAT_HULL ring around a 6x6 MAT_AIR
    interior. No vacuum anywhere (scenario a)."""
    h = w = 8
    mats = np.full((h, w), MAT_HULL, dtype=np.int8)
    mats[1:h - 1, 1:w - 1] = MAT_AIR
    shift, face, solid = _build_caches(mats)
    is_vacuum = np.zeros((h, w), dtype=bool)
    atmosphere = np.full((h, w), FP_ONE, dtype=np.int32)   # ambient N proxy everywhere
    return mats, shift, face, solid, is_vacuum, atmosphere


def _room_9x8_one_face_exposed():
    """The SAME 8x8 room (rows 0..7) as `_room_8x8`, PLUS one extra row
    (row 8) representing space just below the room's bottom hull wall: row 8
    is filled with MAT_HULL (harmless extra thermal mass, conducts normally)
    EXCEPT column 3, which is a genuine open, non-solid, is_vacuum=True cell
    — the SAME "a real vacuum neighbour cell flags the wall exposed"
    convention test_temperature_cooling.py's own
    test_vacuum_exposed_cools_about_4x_faster uses. This makes EXACTLY ONE
    hull tile (row 7, col 3 — part of the room's bottom wall) "exposed"
    (cool_shift_vacuum) via the ordinary 4-neighbour exposure gather; every
    other wall tile keeps its normal (here: disabled) interior shift."""
    h, w = 9, 8
    mats = np.full((h, w), MAT_HULL, dtype=np.int8)
    mats[1:8 - 1, 1:w - 1] = MAT_AIR          # rows 1..6, cols 1..6: interior air
    mats[8, 3] = MAT_AIR                      # the one breach/space cell
    shift, face, solid = _build_caches(mats)
    is_vacuum = np.zeros((h, w), dtype=bool)
    is_vacuum[8, 3] = True
    atmosphere = np.full((h, w), FP_ONE, dtype=np.int32)
    return mats, shift, face, solid, is_vacuum, atmosphere


def _zero_wind(shape):
    return (np.ascontiguousarray(np.zeros(shape, dtype=np.int32)),
            np.ascontiguousarray(np.zeros(shape, dtype=np.int32)))


def _solver(cool_shift_vacuum=3):
    """cool_shift PINNED huge (interior decay disabled — see module
    docstring); cool_shift_vacuum left at a real, fast value (the shipped
    default 3) so scenario (b)'s one exposed tile actually radiates. gas_*
    dials are left at their shipped C++ defaults (gas_advection_rate=900,
    c_v=1.0, n_floor_heat=0.05 — the SAME values config.toml now ships)."""
    s = bp.TemperatureSolver()
    s.no_face = NO_FACE
    s.cool_shift = 31
    s.cool_shift_vacuum = cool_shift_vacuum
    return s


def test_sealed_room_energy_conserved_and_walls_warm():
    """Scenario (a): no vacuum adjacency anywhere."""
    mats, shift, face, solid, is_vacuum, atmosphere = _room_8x8()
    h, w = mats.shape
    temperature = np.zeros((h, w), dtype=np.int32)
    heat = np.zeros((h, w), dtype=np.int32)
    wind_x, wind_y = _zero_wind((h, w))
    solver = _solver()

    # --- Seed a hot gas pocket: every interior AIR cell gets a heat deposit ---
    DEPOSIT = 20 * HEAT_SCALE   # 20 "degrees" worth of energy per interior tile
    interior_mask = (mats == MAT_AIR)
    heat[interior_mask] = DEPOSIT
    solver.step(temperature, heat, shift, face, solid, is_vacuum, atmosphere,
                wind_x, wind_y, 1.0 / 24.0)

    # Exact-value sanity: at ambient N (atmosphere==FP_ONE) and c_v==1.0, the
    # reciprocal chain is provably lossless (reciprocal_q16(FP_ONE) == FP_ONE
    # exactly, make_recip(1.0) == 2**32 exactly) — deposit -> gas ΔT is EXACT,
    # not just "close". This pins the §4.3 reciprocal composition correctness
    # as a side effect of the energy gate. NOTE: this single step() call also
    # runs Pass 2 (conduction) AFTER Pass 1's deposit, in the SAME tick — so
    # only a DEEP interior cell (every 4-neighbour also interior, hence equal
    # T, hence zero conduction flux — the shipped "equal neighbours -> exactly
    # 0" invariant) stays at the exact deposit value; interior cells touching
    # the (still-0) hull already shed a first sliver to their cold neighbour
    # within this same tick — that is Pass 2 doing its job, not a bug.
    assert temperature[3, 3] == DEPOSIT, (
        "a deep-interior cell (no cold neighbour) moved off the exact deposit value")
    # Edge-middle hull tiles (e.g. (0,3)) touch an interior cell directly and
    # already picked up a first sliver this same tick; CORNER hull tiles
    # (e.g. (0,0)) touch only other hull tiles and need a second tick — both
    # are correct conduction behaviour, checked precisely here.
    assert temperature[0, 3] > 0, "an edge-adjacent hull tile did not conduct within one tick"
    assert temperature[0, 0] == 0, "a corner hull tile (no direct interior neighbour) got heat in one tick"

    total0 = int(temperature.astype(np.int64).sum())
    eps_one_tick = _epsilon_bound(face, 1)
    assert 0 <= DEPOSIT * int(interior_mask.sum()) - total0 <= eps_one_tick, (
        "post-seed total drifted from the exact deposit total by more than one "
        "tick's proven conduction-truncation bound"
    )

    # --- No further heat: pure advection(=identity, wind==0) + conduction + cooling ---
    heat[:, :] = 0
    N_TICKS = 400
    eps = _epsilon_bound(face, N_TICKS)
    prev_total = total0
    for _ in range(N_TICKS):
        solver.step(temperature, heat, shift, face, solid, is_vacuum, atmosphere,
                    wind_x, wind_y, 1.0 / 24.0)
        cur_total = int(temperature.astype(np.int64).sum())
        # Provable invariant (module docstring): conduction alone can only
        # hold or LOSE total(T), never gain — monotonic non-increase, every
        # tick, not just net.
        assert cur_total <= prev_total, (
            f"total thermal content increased tick-over-tick: {prev_total} -> {cur_total}")
        prev_total = cur_total

    total_end = prev_total
    drift = total0 - total_end
    assert 0 <= drift <= eps, (
        f"sealed-room drift {drift} exceeds the proven epsilon bound {eps} "
        f"(total0={total0}, total_end={total_end})")

    # Heat visibly flowed gas -> walls: every wall tile borders a hot interior
    # cell in this room, so the WHOLE hull ring must have warmed from 0.
    hull_mask = (mats == MAT_HULL)
    assert np.all(temperature[hull_mask] > 0), "hull ring did not warm from the hot gas pocket"
    # And a specific, named tile (top wall, middle) for a concrete assertion.
    assert temperature[0, 3] > 0, "top-wall tile did not warm"


def test_sealed_room_with_one_hull_face_exposed_drains_monotonically():
    """Scenario (b): one hull tile (row 7, col 3) additionally exposed to a
    real vacuum neighbour cell (row 8, col 3) — the hull radiates to space
    (cool_shift_vacuum) and total energy must monotonically drain."""
    mats, shift, face, solid, is_vacuum, atmosphere = _room_9x8_one_face_exposed()
    h, w = mats.shape
    temperature = np.zeros((h, w), dtype=np.int32)
    heat = np.zeros((h, w), dtype=np.int32)
    wind_x, wind_y = _zero_wind((h, w))
    solver = _solver(cool_shift_vacuum=3)

    DEPOSIT = 20 * HEAT_SCALE
    interior_mask = (mats == MAT_AIR) & ~is_vacuum
    heat[interior_mask] = DEPOSIT
    solver.step(temperature, heat, shift, face, solid, is_vacuum, atmosphere,
                wind_x, wind_y, 1.0 / 24.0)

    total0 = int(temperature.astype(np.int64).sum())
    assert total0 > 0

    heat[:, :] = 0
    N_TICKS = 400
    prev_total = total0
    for _ in range(N_TICKS):
        solver.step(temperature, heat, shift, face, solid, is_vacuum, atmosphere,
                    wind_x, wind_y, 1.0 / 24.0)
        cur_total = int(temperature.astype(np.int64).sum())
        assert cur_total <= prev_total, (
            f"total energy increased with a hull face exposed to vacuum: "
            f"{prev_total} -> {cur_total}")
        prev_total = cur_total

    total_end = prev_total
    # A REAL, non-negligible drain (not just truncation noise): the exposed
    # tile's cool_shift_vacuum (>>3, ~12.5%/tick once it's warmed) must visibly
    # out-pace the sealed room's proven near-zero epsilon drift.
    eps_conduction_only = _epsilon_bound(face, N_TICKS)
    drop = total0 - total_end
    assert drop > 10 * eps_conduction_only, (
        f"drop {drop} is not meaningfully larger than pure-conduction noise "
        f"{eps_conduction_only} — the exposed face does not appear to radiate")
    # The vacuum cell NEVER ACCUMULATES T across ticks (Pass 0 zeroes it at the
    # START of every tick — "energy leaves with the gas"): it cannot exceed
    # what a SINGLE tick's conduction from its one hot neighbour (7,3) could
    # push into it before the NEXT tick's Pass 0 clears it again. It is not
    # exactly 0 at arbitrary sampling time (Pass 2 runs AFTER Pass 0 within the
    # same tick), so bound it instead of asserting a hard zero.
    assert 0 <= temperature[8, 3] < DEPOSIT, (
        "the vacuum breach cell is accumulating energy across ticks (should "
        "only ever hold at most one tick's worth of fresh conduction)")
    # The exposed wall tile is part of the connected hull and still finite/sane.
    assert temperature[7, 3] >= 0


def test_solid_and_vacuum_hull_tile_is_not_wiped_by_pass0():
    """Regression guard for the Pass-0 bug this module's design caught: a
    tile that is BOTH solid AND is_vacuum (the intact-hull convention,
    gamemap.py: "an intact hull is vacuum AND solid") must KEEP its own
    solid-path temperature across ticks — Pass 0's zero-at-vacuum invariant
    is for OPEN (non-solid) breach cells only."""
    h, w = 3, 3
    mats = np.full((h, w), MAT_HULL, dtype=np.int8)
    shift, _face, solid = _build_caches(mats)
    # Conduction DISABLED (all-NO_FACE, the same idiom test_temperature_
    # convert.py's _grid uses) so this test isolates Pass 0's decision in
    # ISOLATION — a real conducting hull would legitimately shed a fully-
    # surrounded centre tile's heat to its 4 (cold) neighbours in one tick
    # (shift 2 means each face carries 1/4 of the delta — 4 faces exactly
    # empties an isolated hot tile whose neighbours are all equal-cold, which
    # is correct physics, not a Pass-0 bug; that is NOT what this test checks).
    face = np.full((h, w, 4), NO_FACE, dtype=np.int32)
    is_vacuum = np.ones((h, w), dtype=bool)     # every hull tile is space-facing
    atmosphere = np.full((h, w), FP_ONE, dtype=np.int32)
    temperature = np.zeros((h, w), dtype=np.int32)
    temperature[1, 1] = 1 << 24                 # pre-existing solid heat
    heat = np.zeros((h, w), dtype=np.int32)
    wind_x, wind_y = _zero_wind((h, w))

    solver = bp.TemperatureSolver()
    solver.no_face = NO_FACE
    solver.cool_shift = 31             # isolate: disable cooling entirely here
    solver.cool_shift_vacuum = 31

    solver.step(temperature, heat, shift, face, solid, is_vacuum, atmosphere,
                wind_x, wind_y, 1.0 / 24.0)
    assert temperature[1, 1] != 0, (
        "a solid+is_vacuum tile's temperature was wiped by the gas zero-at-vacuum pass")


def test_solid_solid_face_shift_unaffected_by_air_conductivity():
    """GATE (P2 item 2): "the solid<->solid faces must be COMPLETELY
    unaffected (assert: face_shift table entries between solid materials
    unchanged)". Recomputed INDEPENDENTLY here from each pair's own
    conductivity (never touching AIR's kappa in the formula at all) —
    mathematically, an entry that never reads air's conductivity cannot have
    moved when only air's config row changed. Also pins the concrete shift
    values so a future accidental edit to a SOLID material's own conductivity
    (or KAPPA_REF/SHIFT_AT_REF/SHIFT_MIN) trips this test too."""
    from simulation.materials import MAT_DOOR, MAT_GLASS, MAT_STEEL, MAT_WOOD

    SOLID_MATS = (MAT_HULL, MAT_WOOD, MAT_DOOR, MAT_STEEL, MAT_GLASS)
    assert all(_TBL.permeability[m] <= 0.0 for m in SOLID_MATS), (
        "sanity: all five must still be solid materials")

    kappa_ref = 50.0        # config.toml [physics.thermal].KAPPA_REF
    shift_min = 2           # config.toml [physics.thermal].SHIFT_MIN
    no_face = NO_FACE

    def independent_face_shift(ka, kb):
        hm = 2.0 * ka * kb / (ka + kb)
        s = int(round(-math.log2(hm / kappa_ref)))
        return max(shift_min, min(s, no_face))

    for a in SOLID_MATS:
        for b in SOLID_MATS:
            ka = float(_TBL.conductivity[a])
            kb = float(_TBL.conductivity[b])
            expected = independent_face_shift(ka, kb)
            actual = int(_TBL.face_shift_table[a, b])
            assert actual == expected, (
                f"face_shift_table[{a},{b}] == {actual}, expected {expected} "
                f"(independent recompute from solid-only conductivities — "
                f"air's conductivity change must not reach here)")

    # And the concrete, historically-known-good self-face values (the same
    # ones test_temperature_conduction.py::test_face_table_anchor_values
    # pins) stay put — a second, redundant confirmation.
    assert int(_TBL.face_shift_table[MAT_HULL, MAT_HULL]) == 2
    assert int(_TBL.face_shift_table[MAT_WOOD, MAT_WOOD]) == 8
