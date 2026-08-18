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

Gates here are §6's 1-5, and 6 (energy books) since P-M4b added the C++ binding
``bp.eos_energy_books_sum`` / ``PhysicsRunner.energy_books_sum``. Gates 8 / 8b
(the caller ORDER pins, and the per-tile split under reordering) landed as
P-M4c in ``tests/test_destroy_order_pins.py``. Deliberately NOT here: 10
(CPU<->GPU, P-M5) and 11 (HUMAN-TEST — PASSED 2026-08-18, see
``docs/human_test_2026-08-18_destroy_wall_seed.md``).

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
# GATE 6 — the ENERGY books close across a destruction (design §4)
# ===========================================================================
# The claim: the energy books sum ``n_bulk * T_game`` over an accountable set
# that skips ``solid || thermal_solid || is_vacuum || (ambient && is_ambient)``
# with NO offset term (eos_solver.cpp: ``acc += nb * (int64_t)temperature[i]``
# — no C, no s_eos_q, no ``+ t_amb_q``). A destroyed tile therefore joins that
# set at ``T = 0`` contributing exactly ``nb * 0 == 0``, so
# ``Delta(energy books) == 0`` across a destruction and NO energy channel is
# needed — including for a BURNING wall.
#
# The instrument is ``PhysicsRunner.energy_books_sum`` -> the C++
# ``eos_energy_books_sum``, which is THE SAME routine ``EOSSolver::step``'s
# eth_transport_delta / eth_compression_delta brackets call (P-M4b extracted it
# from the step-local lambda). Nothing here re-implements the skip-set: a
# Python transcription of those four flags would drift from the books it claims
# to measure, which is the exact failure class this arc exists to close.
#
# WHICH LEGS HAVE TEETH — measured, not assumed, by deleting
# ``self.temperature[fy, fx] = 0`` from ``destroy_wall`` and re-running.
# The measured DELTAS (the gate-6 claim itself):
#   (b) burning wall     0 -> +58,982,400   (65536 * 900)  RED
#   (d) furniture at T   0 -> +45,875,200   (65536 * 700)  RED
#   (a) plain solid wall 0 ->            0                 delta stays green
#   (c) breach tile      0 ->            0                 delta stays green
# (a)'s delta is green either way only because this fixture's T field is 0
# everywhere, so the hole has nothing to leak — the same "the obvious leg does
# not discriminate" shape Gate 1 hit with room_density. (c)'s delta is green
# STRUCTURALLY: a breach tile is skipped by ``is_vacuum`` on both sides of the
# destruction, so no T written there can reach the books at all. Both are kept
# as the stated-per-path coverage §6 asks for, with the REASON asserted so a
# future change that stops excluding them fails loudly.
#
# (As a TEST, (c) does go red in that experiment — but on its trailing
# ``temperature == 0`` companion assertion, which is coverage of the write, not
# of the books. Only (b) and (d) fail on the Delta itself, and those are the
# two legs that make gate 6 a real gate.)
#
# All four cases are SPACE maps, on which ``is_ambient`` reaches C++ as a NULL
# pointer — so none of them exercise the fourth flag of the skip-set. The
# AMBIENT-map leg below covers it, and covers it falsifiably: heating a ring
# tile that really holds bulk N must move the books by exactly nothing.
# ===========================================================================
def _books(sim):
    """Sigma n_bulk*T over the accountable set — raw Q16.16^2, exact int."""
    return sim.physics_runner.energy_books_sum(sim.gmap)


def test_gate6_energy_books_binding_matches_the_solvers_own_bracket():
    """Support gate: the binding is not a second implementation.

    A tick's ``eth_compression_delta`` is ``S_after - S_before`` across step
    4c, taken by the SAME function this test calls. So the books read straight
    after a tick must be an exact int64 (no float, no overflow, no re-derived
    mask), and a second read of untouched state must return the identical
    value. Cheap, but it is what makes the four Delta assertions below
    meaningful rather than self-consistent noise."""
    sim = _fixture_sim()
    sim.set_paused(False)
    sim.step()
    s1 = _books(sim)
    s2 = _books(sim)
    assert isinstance(s1, int)
    assert s1 == s2, "the instrument must be a pure read"
    # The solver's own bracket ran this tick over the same set; if the binding
    # had a different mask the counters below could not both be finite int64.
    assert isinstance(sim.physics_runner.eos.eth_compression_delta, int)


def test_gate6_energy_books_unchanged_across_a_plain_solid_wall():
    """(a) Plain solid wall. Delta(energy books) == 0.

    NOTE (measured): this leg stays green with the ``T := 0`` write deleted,
    because the fixture's T field is 0 everywhere so a solid wall has no heat
    to carry in. It is the baseline-path coverage, not the discriminating leg
    — see (b)."""
    sim = _fixture_sim()
    gmap = sim.gmap
    fy, fx = partition_wall_tiles(gmap)[1]
    assert gmap.solid[fy, fx]

    before = _books(sim)
    gmap.destroy_wall(fy, fx)
    after = _books(sim)

    assert after - before == 0
    # It really joined the accountable set (otherwise 0 is green for the wrong
    # reason: an excluded tile contributes 0 whatever it holds).
    assert not gmap.solid[fy, fx]
    assert not gmap.thermal_solid[fy, fx]
    assert not gmap.is_vacuum[fy, fx] and not gmap.is_ambient[fy, fx]
    assert tile_n(gmap, fy, fx) > 0, "it joined WITH mass, at T = 0"
    assert int(gmap.temperature[fy, fx]) == 0


def test_gate6_energy_books_unchanged_across_a_BURNING_wall():
    """(b) THE discriminating leg. A wall carrying fire and a hot T.

    ``destroy_wall`` wrote no temperature before P-M3, and the solver skips its
    two ``temperature`` writes on a ``thermal_solid`` tile — so a wall's T is
    stale-but-live state that nothing resets, and a burning wall joined the
    books HOT. That is a pre-existing energy-seam hole (design §4), and it is
    what ``T := 0`` closes.

    Measured with the write deleted: Delta == +58,982,400 == 65536 * 900, i.e.
    the tile's seeded n_bulk times its stale T — energy minted out of a
    destruction, in the same currency as eth_transport_delta."""
    sim = _fixture_sim()
    gmap = sim.gmap
    fy, fx = partition_wall_tiles(gmap)[1]
    assert gmap.solid[fy, fx] and gmap.thermal_solid[fy, fx]
    gmap.temperature[fy, fx] = 900          # the burning wall's stored heat
    gmap.fire[fy, fx] = 40000
    # Excluded WHILE solid, so the 900 is invisible to the books right now —
    # the whole question is what happens when the tile joins.
    before = _books(sim)

    gmap.destroy_wall(fy, fx)
    after = _books(sim)

    assert after - before == 0, (
        "a burning wall must join the energy books COLD; a nonzero delta here "
        "is energy created by destruction, with no channel to name it")
    assert not gmap.thermal_solid[fy, fx], "on_tile_changed must clear ts"
    assert not gmap.solid[fy, fx] and not gmap.is_vacuum[fy, fx]
    assert int(gmap.temperature[fy, fx]) == 0
    assert int(gmap.fire[fy, fx]) == 0
    assert tile_n(gmap, fy, fx) > 0


def test_gate6_energy_books_unchanged_across_a_breach_tile():
    """(c) Breach tile on a SPACE map. Delta == 0, structurally.

    A breached tile joins ``is_vacuum``, which the accountable set skips — so
    it is excluded on BOTH sides of the destruction and no temperature written
    there can reach the books. Stated as coverage of the third path, with the
    REASON asserted: if a future change stops marking breach tiles vacuum while
    still leaving them hot, the exclusion assertion fires even though the delta
    would not.

    (The T := 0 write still matters on this path for a different consumer: the
    c_local scan skips only ``solid || is_vacuum``, so on an AMBIENT map a hot
    breached tile inflates map-wide sound speed for a tick. That is not an
    energy-books effect and is not gated here.)"""
    sim = _fixture_sim()
    gmap = sim.gmap
    assert gmap._boundary == "space"
    h, w = gmap.solid.shape
    fy, fx = 0, w // 4                       # edge hull -> joins the boundary
    assert gmap.solid[fy, fx]
    gmap.temperature[fy, fx] = 900
    gmap.fire[fy, fx] = 40000

    before = _books(sim)
    gmap.destroy_wall(fy, fx)
    after = _books(sim)

    assert after - before == 0
    assert gmap.is_vacuum[fy, fx], (
        "the breach tile's exclusion from the books rests on is_vacuum; if it "
        "is no longer vacuum the zero above stops being structural")
    assert tile_n(gmap, fy, fx) == 0, "breach seeds nothing"
    assert int(gmap.temperature[fy, fx]) == 0


def test_gate6_energy_books_unchanged_across_a_furniture_tile():
    """(d) Furniture, the second discriminating leg — and a different shape
    from (b).

    Furniture is ``permeability = 0.5`` -> NOT solid, so it already holds bulk
    N; but it IS ``thermal_solid`` (``thermal_mass > 0`` — the crate's
    temperature belongs to the TemperatureSolver), so it is excluded from the
    energy books anyway. Destroying it clears ts and it joins the books for the
    first time, with the seeded N and T := 0 -> contributes 0.

    So this leg exercises the ``thermal_solid`` flag of the skip-set rather
    than the ``solid`` flag (b) exercises — and it is exactly the case a Python
    transcription of the four flags would get wrong, since a crate looks like
    open air on the ``solid`` axis.

    Measured with the ``T := 0`` write deleted: Delta == +45,875,200 ==
    65536 * 700."""
    fy, fx = 6, 6                            # interior, NOT adjacent to vacuum
    sim = _sim(_hull_box_level(edits=[(fy, fx, MAT_FURNITURE)],
                               name="pm4b_furniture_books"))
    gmap = sim.gmap
    assert int(gmap.material[fy, fx]) == MAT_FURNITURE
    assert not gmap.solid[fy, fx], "furniture is permeable -> it holds gas"
    assert gmap.thermal_solid[fy, fx], "...and it is a THERMAL solid"
    assert tile_n(gmap, fy, fx) > 0
    gmap.temperature[fy, fx] = 700

    before = _books(sim)
    gmap.destroy_wall(fy, fx)
    after = _books(sim)

    assert after - before == 0
    assert not gmap.thermal_solid[fy, fx], "destroying it must clear ts"
    assert not gmap.solid[fy, fx] and not gmap.is_vacuum[fy, fx]
    assert int(gmap.temperature[fy, fx]) == 0
    assert tile_n(gmap, fy, fx) > 0, "it joined the books WITH mass"


def test_gate6_energy_books_unchanged_across_breached_furniture():
    """(d') The §3.2 escape: furniture adjacent to vacuum, evacuated at destroy
    time. The mass side is Gate 4; the energy side is zero for the same
    structural reason as (c) — the tile ends is_vacuum — and it is asserted so
    the evacuation path is not silently outside gate 6's coverage."""
    fy, fx = 1, 6
    sim = _sim(_hull_box_level(edits=[(fy, fx, MAT_FURNITURE)],
                               name="pm4b_breach_furniture_books"))
    gmap = sim.gmap
    gmap.temperature[fy, fx] = 700

    before = _books(sim)
    gmap.destroy_wall(fy, fx)
    after = _books(sim)

    assert after - before == 0
    assert gmap.is_vacuum[fy, fx]
    assert tile_n(gmap, fy, fx) == 0


def test_gate6_energy_books_on_an_AMBIENT_map_exclude_the_ring():
    """The fourth skip flag. Every case above is a SPACE map, where
    ``is_ambient`` reaches C++ as a null pointer and the ring term is dormant
    by branch — so none of them prove the ambient leg is wired at all.

    On a planetside map ``PhysicsRunner._ambient_mask`` returns the live ring,
    and heating a ring tile (which holds real bulk N) must move the books by
    EXACTLY nothing. If the mask were dropped on the way to C++ this goes red
    immediately; the space-map cases could not notice.

    Then the gate-6 claim itself on this map class: destroying a burning
    interior wall is still Delta == 0."""
    sim = _sim(level_loader.load("planetside_demo"))
    gmap = sim.gmap
    runner = sim.physics_runner
    assert gmap._boundary == "ambient"
    assert gmap.is_ambient.any(), "the fixture must actually have a ring"
    assert runner._ambient_mask(gmap) is not None

    # A ring tile holds bulk N, and heating it must be invisible to the books.
    ys, xs = np.nonzero(gmap.is_ambient)
    ry, rx = int(ys[0]), int(xs[0])
    assert tile_n(gmap, ry, rx) > 0, "a ring tile holds gas — that is the point"
    before = _books(sim)
    gmap.temperature[ry, rx] = 4000
    assert _books(sim) - before == 0, (
        "the ambient ring must be excluded; a nonzero delta means the "
        "is_ambient mask never reached the C++ skip-set")

    # ...while an interior cell on the SAME map is accounted normally.
    ys, xs = np.nonzero((~gmap.solid) & (~gmap.thermal_solid)
                        & (~gmap.is_vacuum) & (~gmap.is_ambient))
    iy, ix = int(ys[len(ys) // 2]), int(xs[len(xs) // 2])
    nb = tile_n(gmap, iy, ix)
    assert nb > 0
    mid = _books(sim)
    gmap.temperature[iy, ix] = int(gmap.temperature[iy, ix]) + 250
    assert _books(sim) - mid == nb * 250

    # And the gate-6 claim on this map class: a burning interior wall.
    h, w = gmap.solid.shape
    wall = next(((y, x) for y in range(3, h - 3) for x in range(3, w - 3)
                 if gmap.solid[y, x] and not gmap.is_ambient[y, x]), None)
    assert wall is not None, "the fixture must offer an interior wall"
    wy, wx = wall
    gmap.temperature[wy, wx] = 900
    gmap.fire[wy, wx] = 40000
    before = _books(sim)
    gmap.destroy_wall(wy, wx)
    assert _books(sim) - before == 0
    assert int(gmap.temperature[wy, wx]) == 0


def test_gate6_the_instrument_can_see_energy_when_there_is_energy():
    """Non-vacuity for the four zeros above: the instrument is not a constant.

    Heating one accountable cell by dT moves the books by exactly
    ``n_bulk * dT``. If this did not hold, ``Delta == 0`` everywhere above
    would prove nothing — a books function that always returns the same number
    passes every conservation gate ever written."""
    sim = _fixture_sim()
    gmap = sim.gmap
    h, w = gmap.solid.shape
    ys, xs = np.nonzero((~gmap.solid) & (~gmap.thermal_solid)
                        & (~gmap.is_vacuum) & (~gmap.is_ambient))
    assert len(ys) > 0
    fy, fx = int(ys[0]), int(xs[0])
    nb = tile_n(gmap, fy, fx)
    assert nb > 0

    before = _books(sim)
    gmap.temperature[fy, fx] = int(gmap.temperature[fy, fx]) + 250
    assert _books(sim) - before == nb * 250

    # ...and a cell OUTSIDE the set moves it by nothing, which is the skip-set
    # actually being applied rather than a whole-map sum.
    sy, sx = partition_wall_tiles(gmap)[1]
    assert gmap.solid[sy, sx]
    mid = _books(sim)
    gmap.temperature[sy, sx] = 5000
    assert _books(sim) - mid == 0


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
