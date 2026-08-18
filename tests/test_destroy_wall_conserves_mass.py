"""P-M3 — ``destroy_wall``'s seed: constant, inherited, booked.

The asserting twin of ``tools/repro_destroy_wall_mint.py`` (which stays a
measurement CLI). The tool established the defect; this file locks the fix.

**The defect.** ``destroy_wall`` used to seed a newly-opened tile with the
neighbour MEAN of its bulk gas, withdrawing nothing from the donors — so every
destroyed tile created one neighbour-mean cell of air out of nothing. Because
``find_burst_walls`` fires on a pressure DIFFERENTIAL, a bursting wall is
high-pressure by definition and the mint scaled with exactly that pressure: the
emergent pressure-relief valve was a pressure AMPLIFIER. One recorded session
put 87.7% of a 2.201x total mass growth through this path.

**The fix** (docs/mass_books_pm3_destroy_wall_seed_design_2026-08-18.md): seed a
CONSTANT total at the map's ambient (``GameMap.ambient_seed()``), inherit the
composition from the open donors by an exact int64 form, skip the seed on a
breach tile AND evacuate what it already held, and BOOK every one of those
changes to the signed ``n_destruction_seed_sum`` channel. The load-bearing
property is the constant TOTAL — it is what breaks the feedback loop — not the
value and not the composition.

Gates here are §6's 1-5. Deliberately NOT here: 6 (energy books — needs a C++
binding for ``eth_books_sum``), 10 (CPU<->GPU), 11 (HUMAN-TEST). Each is its
own patch. Gates 8 / 8b (the caller ORDER pins and the per-tile split under
reordering) landed as P-M4c in ``tests/test_destroy_order_pins.py``.

Every gate below is written so it can go RED. In particular Gate 1 asserts the
PREDICTED value ``k * ambient_seed().n_total_q``, never ``Delta(Sum N) ==
n_destruction_seed_sum`` — with a MEASURED channel the latter is ``A == A`` and
stays green against the unfixed mint.

Run:
    conda run -n data python -m pytest tests/test_destroy_wall_conserves_mass.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "tools",
           ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                              # noqa: E402
import level_loader                                      # noqa: E402
from level_loader import LevelData                       # noqa: E402
from simulation import Simulation                        # noqa: E402
from simulation.gases import O2, INERT_N2                # noqa: E402
from simulation.materials import MAT_AIR, MAT_FURNITURE  # noqa: E402

SEED = 20260818
FIXTURE = "bench_two_room"          # the tool's fixture — same scene, asserted
SPACE_CODE = 9                      # v2 tilemap: outer space (MAT_AIR + vacuum)
HULL_CODE = 1


# ---------------------------------------------------------------------------
# Helpers (shared with tools/repro_destroy_wall_mint.py by intent, not import —
# the tool must keep running standalone)
# ---------------------------------------------------------------------------
def bulk_n(gmap):
    """Exact int64 total of the conservative bulk species, raw Q16.16."""
    return (int(gmap.gas[O2].astype(np.int64).sum())
            + int(gmap.gas[INERT_N2].astype(np.int64).sum()))


def tile_n(gmap, fy, fx):
    return int(gmap.gas[O2][fy, fx]) + int(gmap.gas[INERT_N2][fy, fx])


def _sim(level):
    return Simulation(level, seed=SEED, breach_physics=bp,
                      enable_recorder=False)


def _fixture_sim():
    return _sim(level_loader.load(FIXTURE))


def partition_wall_tiles(gmap):
    """Interior partition tiles of the two-room fixture (the mid column),
    excluding the door gap and the outer hull rows."""
    h, w = gmap.solid.shape
    mid = w // 2
    return [(y, mid) for y in range(1, h - 1) if gmap.solid[y, mid]]


def _scale_all_gas(gmap, factor):
    """Scale the bulk gas AND ``atmosphere`` by ``factor`` (Gate 2's sweep).

    The tool scales ``gas`` only, which would leave an ``atmosphere``-sourced
    seed invisible to the sweep — so the gate scales both."""
    for g in (O2, INERT_N2):
        gmap.gas[g][:] = (gmap.gas[g].astype(np.int64) * factor).astype(np.int32)
    gmap.atmosphere[:] = (gmap.atmosphere.astype(np.int64)
                          * factor).astype(np.int32)


# ---------------------------------------------------------------------------
# Scenes for the furniture / breach gates.
#
# Furniture ships permeability = 0.5 -> NOT solid -> it is in the open-air mask
# and therefore ALREADY HOLDS bulk N. That is what makes the destruction delta
# signed, and what opens the unbooked sink the breach path has to close.
# ---------------------------------------------------------------------------
def _hull_box_level(h=14, w=14, edits=(), name="pm3_box"):
    """A SPACE map: a vacuum ring, a hull box inside it, interior air."""
    tm = np.full((h, w), SPACE_CODE, dtype=np.int32)
    tm[1:h - 1, 1:w - 1] = HULL_CODE
    tm[2:h - 2, 2:w - 2] = MAT_AIR
    for (y, x, code) in edits:
        tm[y, x] = code
    return LevelData(name=name, version="2", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."))


# ===========================================================================
# Gate 0 (support) — the accessor itself
# ===========================================================================
def test_ambient_seed_is_one_accessor_and_exact():
    """``ambient_seed()`` is the ONE derivation of the map's air constant.

    On a space map ``_ambient`` is None (it is populated only under
    ``boundary == "ambient"``), so the fallback branch is the one every ship
    map takes — and its split must sum to the total EXACTLY, or the seed
    leaks an LSB per destroyed tile."""
    gmap = _fixture_sim().gmap
    n_total_q, o2_q, n2_q, pin_q = gmap.ambient_seed()
    assert o2_q + n2_q == n_total_q, "the split must be an exact complement"
    assert (n_total_q, o2_q, n2_q) == (65536, 13763, 51773)
    assert pin_q == 65536
    assert all(isinstance(v, int) for v in (n_total_q, o2_q, n2_q, pin_q))


# ===========================================================================
# GATE 1 — solid path, PREDICTED value
# ===========================================================================
@pytest.mark.parametrize("room_density", (1, 3))
def test_gate1_solid_destruction_seeds_exactly_k_ambient_cells(room_density):
    """``Delta(Sum N)`` over k destroyed SOLID tiles is EXACTLY
    ``k * ambient_seed().n_total_q``, and each seeded tile holds
    ``o2 + n2 == n_total_q``.

    This is the predicted value, not the measured channel: asserting
    ``Delta(Sum N) == n_destruction_seed_sum`` would be ``A == A`` (the channel
    IS that measurement) and would stay green against the old mint.

    **The density parameter is load-bearing, not decoration.** At x1 every cell
    of this fixture already holds exactly ``n_total_q``, so the OLD
    neighbour-mean seed satisfies the proposition too — measured: the x1 leg
    alone stays green against the unfixed code. The x3 leg is what makes the
    gate falsifiable: the neighbour mean there is ``3 * n_total_q`` per tile.
    (Verified by reverting the seed in a scratch copy: the x1 leg passes, the
    x3 leg goes red.)"""
    gmap = _fixture_sim().gmap
    if room_density != 1:
        _scale_all_gas(gmap, room_density)
    n_total_q, _o2, _n2, _pin = gmap.ambient_seed()
    tiles = partition_wall_tiles(gmap)[:4]
    assert len(tiles) == 4, "fixture must offer 4 interior partition tiles"

    n_before = bulk_n(gmap)
    for (fy, fx) in tiles:
        assert gmap.solid[fy, fx], "Gate 1 is the SOLID path"
        assert tile_n(gmap, fy, fx) == 0, "a solid tile holds no bulk N"
        gmap.destroy_wall(fy, fx)

    assert bulk_n(gmap) - n_before == len(tiles) * n_total_q

    for (fy, fx) in tiles:
        # Non-vacuity: these must really have taken the SEED path, not the
        # breach path (which seeds nothing and would make the sum above wrong
        # for a different reason).
        assert not gmap.is_vacuum[fy, fx] and not gmap.is_ambient[fy, fx]
        assert not gmap.solid[fy, fx]
        assert tile_n(gmap, fy, fx) == n_total_q
        assert int(gmap.material[fy, fx]) == MAT_AIR


def test_gate1_seed_is_constant_not_proportional_to_local_pressure():
    """The load-bearing property, stated directly: the seeded total does not
    move when the local gas does.

    This is the amplifier's death certificate. Under the neighbour mean the
    same destruction at x100 minted 100x as much; the burst valve fires on a
    differential, so the mint scaled with exactly the pressure that triggered
    it."""
    seen = set()
    for scale in (1, 10, 100):
        gmap = _fixture_sim().gmap
        _scale_all_gas(gmap, scale)
        n_total_q, _o2, _n2, _pin = gmap.ambient_seed()
        (fy, fx) = partition_wall_tiles(gmap)[1]
        n_before = bulk_n(gmap)
        gmap.destroy_wall(fy, fx)
        seen.add(bulk_n(gmap) - n_before)
        assert bulk_n(gmap) - n_before == n_total_q
    assert seen == {65536}, f"seed must not scale with local density: {seen}"


# ===========================================================================
# GATE 2 — density independence AND composition
# ===========================================================================
@pytest.mark.parametrize("scale", (1, 10, 100))
def test_gate2_seeded_o2_n2_pair_is_identical_at_every_density(scale):
    """The x1 / x10 / x100 sweep, with ``atmosphere`` scaled ALONGSIDE ``gas``,
    asserting the seeded ``(O2, N2)`` PAIR — not merely the total.

    A total-only assertion has zero discriminating power over composition: a
    split implemented as ``o2 := sum(O2)/4`` with ``n2 := n_total - o2`` gives
    the right total at every scale while injecting density-scaled oxidizer (and
    a NEGATIVE n2, which bulk_transport then silently clamps to 0 — an unbooked
    mint one substep after the books recorded a smaller delta).

    The donors are given a deliberately NON-ambient 40/60 composition so the
    inherited value is distinguishable from the fallback constant."""
    gmap = _fixture_sim().gmap
    n_total_q, o2_amb_q, _n2_amb, _pin = gmap.ambient_seed()

    # A distinct, exactly-representable donor composition: 0.40 / 0.60 of
    # FP_ONE (26214 + 39322 == 65536), then the whole map scaled uniformly.
    donor_o2, donor_n2 = 26214, 39322
    assert donor_o2 + donor_n2 == n_total_q
    assert donor_o2 != o2_amb_q, "must differ from the fallback, or the gate " \
                                 "cannot tell inheritance from the constant"
    open_air = ~(gmap.solid | gmap.is_vacuum)
    gmap.gas[O2][:] = np.where(open_air, donor_o2, 0)
    gmap.gas[INERT_N2][:] = np.where(open_air, donor_n2, 0)
    _scale_all_gas(gmap, scale)

    (fy, fx) = partition_wall_tiles(gmap)[1]
    n_before = bulk_n(gmap)
    gmap.destroy_wall(fy, fx)

    # The PAIR, identical at every scale — the parametrization is the sweep.
    assert (int(gmap.gas[O2][fy, fx]),
            int(gmap.gas[INERT_N2][fy, fx])) == (donor_o2, donor_n2)
    # ...and the total is still the constant, and still exactly booked.
    assert tile_n(gmap, fy, fx) == n_total_q
    assert bulk_n(gmap) - n_before == n_total_q
    # No negative plane can reach bulk_transport's silent clamp.
    assert int(gmap.gas[O2][fy, fx]) >= 0
    assert int(gmap.gas[INERT_N2][fy, fx]) >= 0


def test_gate2_composition_fallback_is_ambient_when_no_donor_holds_gas():
    """The ``sum_n == 0`` fallback is NOT an edge case — it fires on the
    interior tile of a >=2-thick slab (all four neighbours solid), i.e.
    ordinary blast geometry.

    Erik's ruling: fall back to the map's AMBIENT composition, not pure N2.
    Pure N2 is safer for fire but breaks the cave case that motivates the
    ambient seed in the first place — digging out a cave must not fill it with
    nitrogen and suffocate the player. ACCEPTED GAP: blasting a burning slab
    therefore briefly feeds the fire."""
    # A 3x3 hull block floating in the room: its CENTRE is the interior tile
    # of a >=2-thick slab, so all four donors are solid.
    slab = [(y, x, HULL_CODE) for y in (5, 6, 7) for x in (5, 6, 7)]
    gmap = _sim(_hull_box_level(edits=slab, name="pm3_thick_slab")).gmap
    n_total_q, o2_amb_q, n2_amb_q, _pin = gmap.ambient_seed()
    fy, fx = 6, 6
    for (dy, dx) in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        assert gmap.solid[fy + dy, fx + dx], "every donor must be solid"

    gmap.destroy_wall(fy, fx)
    assert (int(gmap.gas[O2][fy, fx]),
            int(gmap.gas[INERT_N2][fy, fx])) == (o2_amb_q, n2_amb_q)
    assert tile_n(gmap, fy, fx) == n_total_q


# ===========================================================================
# GATE 3 — breach path, SPACE map
# ===========================================================================
def test_gate3_breach_tile_ends_with_zero_gas_and_no_compensating_wipe():
    """On a SPACE map a destroyed tile that joins the vacuum boundary is
    Dirichlet-pinned: it is seeded with NOTHING, and there is no
    mint-then-delete round trip for the next tick to undo.

    Measured before the fix: +10 at destroy, -10 one tick later. The delete
    half went through ``bulk_transport``'s ``solid || is_vacuum -> N = 0``
    clamp, which carries no ``boundary_flux`` credit — so the round trip was
    unbooked on BOTH sides of the seam.

    (Stated per map class deliberately: on an AMBIENT map the same skip is
    correct for the opposite reason — the rail FILLS the tile to N_amb each
    substep and books the difference to ``boundary_flux``.)"""
    sim = _fixture_sim()
    gmap = sim.gmap
    assert gmap._boundary == "space", "this gate is the space-map class"
    h, w = gmap.solid.shape
    fy, fx = 0, w // 4                       # an EDGE HULL tile: on_edge_hull
    assert gmap.solid[fy, fx] and tile_n(gmap, fy, fx) == 0

    n_before = bulk_n(gmap)
    gmap.destroy_wall(fy, fx)

    # It really took the breach path (non-vacuity — otherwise the zeros below
    # would be green for the wrong reason).
    assert gmap.is_vacuum[fy, fx], "edge hull must join the vacuum boundary"
    assert int(gmap.gas[O2][fy, fx]) == 0
    assert int(gmap.gas[INERT_N2][fy, fx]) == 0
    # Nothing minted, so nothing for the wipe to compensate.
    assert bulk_n(gmap) - n_before == 0
    assert gmap.n_destruction_seed_sum == 0

    n_after_destroy = bulk_n(gmap)
    sim.set_paused(False)
    sim.step()
    # The tile is still empty a tick later, and the tick did NOT have to remove
    # a seed: any Sum N change belongs to legitimate venting elsewhere, and the
    # destroyed tile contributes exactly 0 to it.
    assert tile_n(gmap, fy, fx) == 0
    assert bulk_n(gmap) <= n_after_destroy      # venting only ever removes
    assert gmap.n_destruction_seed_sum == 0     # the tick books nothing here


# ===========================================================================
# GATE 4 — breach x furniture (the sink a bare skip would open)
# ===========================================================================
def test_gate4_breached_furniture_books_its_prior_gas_as_a_negative_delta():
    """Chew a FURNITURE tile adjacent to vacuum.

    Furniture ships ``permeability = 0.5`` -> NOT solid -> it is in the
    open-air mask and already holds bulk N. ``destroy_wall``'s gate is
    ``material != MAT_AIR``, so it reaches here. With a bare breach SKIP its
    existing gas would never be booked, and the next transport pass would zero
    it through the no-credit clamp: mass vanishing with a channel on NEITHER
    side of the seam.

    So the breach path EVACUATES explicitly, at destroy time, inside the
    measured bracket. The prior N shows up as a NEGATIVE delta and nothing is
    left for the clamp to take."""
    # Furniture replacing one hull tile of the box wall, so its outward
    # neighbour is the (non-solid) vacuum ring.
    fy, fx = 1, 6
    sim = _sim(_hull_box_level(edits=[(fy, fx, MAT_FURNITURE)],
                               name="pm3_breach_furniture"))
    gmap = sim.gmap
    assert int(gmap.material[fy, fx]) == MAT_FURNITURE
    assert not gmap.solid[fy, fx], "furniture is permeable -> it holds gas"
    assert gmap.is_vacuum[fy - 1, fx] and not gmap.solid[fy - 1, fx]

    prior = tile_n(gmap, fy, fx)
    assert prior > 0, "the scene must actually put gas in the crate"

    n_before = bulk_n(gmap)
    gmap.destroy_wall(fy, fx)

    assert gmap.is_vacuum[fy, fx], "it must have joined the boundary"
    # Evacuated at destroy time — nothing left for the transport clamp.
    assert int(gmap.gas[O2][fy, fx]) == 0
    assert int(gmap.gas[INERT_N2][fy, fx]) == 0
    # ...and the removal is booked, signed NEGATIVE, at the predicted value.
    assert bulk_n(gmap) - n_before == -prior
    assert gmap.n_destruction_seed_sum == -prior
    assert gmap.n_destruction_seed_sum < 0

    # The next transport pass finds nothing to take unbooked at that tile.
    booked = gmap.n_destruction_seed_sum
    sim.set_paused(False)
    sim.step()
    assert tile_n(gmap, fy, fx) == 0
    assert gmap.n_destruction_seed_sum == booked


# ===========================================================================
# GATE 5 — furniture in a pressurised room
# ===========================================================================
def test_gate5_furniture_in_a_pressurised_room_is_a_signed_negative_delta():
    """Writing the CONSTANT ambient seed into a tile that already held gas is
    ``seed - prior`` — negative whenever the room is above ambient. Chewing a
    crate at 5 atm DELETES about 4 cell-equivalents.

    That is why the channel is measured and signed rather than the
    ``ambient_N x tiles_seeded`` formula an earlier draft specified. The gate
    asserts the PREDICTED value (``n_total_q - prior``), so it is not a
    restatement of the measurement."""
    fy, fx = 5, 5                                   # interior, no vacuum near
    sim = _sim(_hull_box_level(edits=[(fy, fx, MAT_FURNITURE)],
                               name="pm3_pressurised_furniture"))
    gmap = sim.gmap
    assert not gmap.solid[fy, fx]
    _scale_all_gas(gmap, 5)

    n_total_q, _o2, _n2, _pin = gmap.ambient_seed()
    prior = tile_n(gmap, fy, fx)
    assert prior == 5 * n_total_q

    n_before = bulk_n(gmap)
    gmap.destroy_wall(fy, fx)

    assert not gmap.is_vacuum[fy, fx], "Gate 5 is the NON-breach path"
    assert tile_n(gmap, fy, fx) == n_total_q         # the constant, not 5x it
    # Signed, negative, and exactly the predicted value.
    delta = bulk_n(gmap) - n_before
    assert delta == n_total_q - prior
    assert delta == -4 * n_total_q
    assert delta < 0
    # The books close: the channel names the whole of the observed change.
    assert gmap.n_destruction_seed_sum == delta


def test_gate5_the_channel_accumulates_across_mixed_events():
    """The channel is a running signed total, not a per-call value: a solid
    destruction (+n_total) and a pressurised-furniture destruction
    (-(4 x n_total)) net out on the same counter, and the counter still equals
    the map's whole observed change across both."""
    fy, fx = 5, 5
    py, px = 9, 9                    # a free-standing hull pillar, no vacuum
    sim = _sim(_hull_box_level(edits=[(fy, fx, MAT_FURNITURE),
                                      (py, px, HULL_CODE)],
                               name="pm3_mixed_events"))
    gmap = sim.gmap
    _scale_all_gas(gmap, 5)
    n_total_q, _o2, _n2, _pin = gmap.ambient_seed()
    assert gmap.solid[py, px] and not gmap.solid[fy, fx]

    n_before = bulk_n(gmap)
    gmap.destroy_wall(7, 7)          # interior AIR tile -> gated out entirely
    assert gmap.n_destruction_seed_sum == 0, "MAT_AIR is not destructible"

    gmap.destroy_wall(py, px)        # the solid pillar:  +n_total
    gmap.destroy_wall(fy, fx)        # the 5 atm crate:   -(4 * n_total)
    assert gmap.n_destruction_seed_sum == bulk_n(gmap) - n_before
    assert gmap.n_destruction_seed_sum == n_total_q - 4 * n_total_q


# ===========================================================================
# Determinism (Gate 7's Python half): the seed is exact integer state
# ===========================================================================
def test_seed_writes_are_exact_integers_and_bit_reproducible():
    """Two independent runs of the same destruction produce bit-identical gas
    planes. The seed introduces no float and no new quantisation path — the
    only float in the old code was ``_neighbor_mean``'s ``total / count``,
    which this replaces."""
    def run():
        gmap = _fixture_sim().gmap
        _scale_all_gas(gmap, 7)
        for (fy, fx) in partition_wall_tiles(gmap)[:3]:
            gmap.destroy_wall(fy, fx)
        return (gmap.gas[O2].copy(), gmap.gas[INERT_N2].copy(),
                gmap.atmosphere.copy(), gmap.n_destruction_seed_sum)

    a_o2, a_n2, a_atm, a_ch = run()
    b_o2, b_n2, b_atm, b_ch = run()
    assert np.array_equal(a_o2, b_o2) and np.array_equal(a_n2, b_n2)
    assert np.array_equal(a_atm, b_atm)
    assert a_ch == b_ch
    assert a_o2.dtype == np.int32 and a_n2.dtype == np.int32


# ===========================================================================
# The same-tick companions of the seed (design §3.1.2 / §3.1 / §3.3)
# ===========================================================================
def test_atmosphere_temperature_and_fire_are_written_on_every_path():
    """``atmosphere`` MUST be written, including on a breach tile.

    A solid tile's ``atmosphere`` is a hard 0, and the two callers that run
    AFTER the physics step (fire burn-through, the burst valve) are refilled by
    nothing that tick. ``find_burst_walls`` then reads the hole's 0 as a real
    side: a wall between a 2.1 atm room and a 1.0 atm corridor sees sides
    {2.1, 1.0, 0.0} -> spread 2.1 > threshold 2.0 -> its neighbours pop.
    Dropping the write would remove the amplifier at the mass end and install
    one at the burst end, same code path, same tick.

    ``temperature := 0`` keeps the energy books closed (they sum
    ``n_bulk * T_game`` with no offset, so a cell joining at 0 contributes
    exactly 0); ``fire := 0`` clears the stale display/sensor value the
    non-burn-through callers leave behind."""
    sim = _fixture_sim()
    gmap = sim.gmap
    _n_total_q, _o2, _n2, pin_q = gmap.ambient_seed()
    h, w = gmap.solid.shape

    interior = partition_wall_tiles(gmap)[1]
    breach = (0, w // 4)
    for (fy, fx) in (interior, breach):
        gmap.temperature[fy, fx] = 900
        gmap.fire[fy, fx] = 40000
        gmap.destroy_wall(fy, fx)
        assert int(gmap.atmosphere[fy, fx]) == pin_q
        assert int(gmap.temperature[fy, fx]) == 0
        assert int(gmap.fire[fy, fx]) == 0
    # The breach tile really was one (the write is not skipped there).
    assert gmap.is_vacuum[breach]
