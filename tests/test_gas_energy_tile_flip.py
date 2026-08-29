"""arc #54 P-G1b — THE TILE-FLIP PROPERTY GATE (design §2.7, last row).

Under a stored energy field, a tile changing MEDIUM is an energy event. A gas
cell that becomes solid or thermal_solid must RETIRE its stored energy through
a named channel; a solid or thermal_solid that becomes gas must be BORN AT
AMBIENT, mirror included. Neither may move `Σ_accountable gas_energy` by one
raw count more or less than the channel says it did.

The gate is the one the design names: *flip a tile both ways, books move by
exactly the named amount* — with `tests/test_destroy_wall_conserves_mass.py`
gate 6 as the precedent for how to bracket it (a pure-read instrument either
side of a structural edit, no solver step in between, so nothing but the edit
can move the number).

WHY THIS IS ITS OWN FILE rather than more legs of gate 6: gate 6 measures the
MASS-seam's energy consequence for `destroy_wall` alone. This measures the
MEMBERSHIP seam — `on_tile_changed`, which every structural edit in the engine
funnels through (seal, unseal, destroy, burst walls, bullet cover-chew,
furniture flips) — and it measures it in BOTH directions, which is the half
that catches a stale hot crate leaving its temperature behind on the air tile
that replaces it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import breach_physics as bp                              # noqa: E402
from level_loader import LevelData                        # noqa: E402
from simulation import Simulation                          # noqa: E402
from simulation import atmosphere_fixed                    # noqa: E402
from simulation import materials                           # noqa: E402


def _room_sim(h=16, w=16):
    """A sealed hull room with an air interior — the same synthetic shape the
    books-identity gate uses, big enough that a flipped tile has open
    neighbours on every side."""
    tm = np.ones((h, w), dtype=np.int32)      # all hull
    tm[2:h - 2, 2:w - 2] = 4                   # carve interior air
    level = LevelData(name="gas_energy_tile_flip", version="1",
                      path=Path("."), tilemap=tm, tile_size_m=1.0 / 3.0,
                      diffuse_path=Path("."))
    return Simulation(level, seed=7, breach_physics=bp, enable_recorder=False)


def _e_acct(g) -> int:
    """`Σ_accountable gas_energy`, as a Python int (design §2.2 forbids an
    absolute int64 sum — the books are always relative sums in production, and
    a test must not be the thing that wraps)."""
    acct = g._gas_energy_accountable()
    return int(g.gas_energy[acct].astype(object).sum())


def _books_net(g) -> int:
    return int(g.gas_energy_seam_net())


def _seed_hot(g, sel, deg):
    """Seed a gas patch through the ONE sanctioned seeding seam."""
    g.seed_gas_temperature(sel, g.temperature[sel]
                           + atmosphere_fixed.quantize_scalar(float(deg)))


# ---------------------------------------------------------------------------
# Direction 1 — gas -> solid: the stored energy RETIRES, exactly.
# ---------------------------------------------------------------------------
def test_gas_to_solid_retires_exactly_the_tiles_energy():
    """Seal one hot interior tile. `Σ_accountable gas_energy` must fall by
    exactly (its energy that did not go to a receiver) and the seam books must
    account for every raw count of the difference.

    `seal_tiles` is TWO named events at once — the evacuated mass is MOVED to
    the receivers carrying the sealed tile's own T_abs, and the sub-count
    remainder the mirror could not represent RETIRES — so the assertion is on
    the seam net, which is the sum of both."""
    sim = _room_sim()
    g = sim.gmap
    _seed_hot(g, np.s_[3:12, 3:12], 400.0)
    fy, fx = 7, 7
    assert g.gas_is_accountable(fy, fx)
    # Plant a SUB-COUNT residue so the retire is genuinely non-zero. Straight
    # after `seed_gas_temperature` the field is exactly `N·T_abs`, so the whole
    # of it moves to the receivers and the retire is 0 — correct, but vacuous
    # as a test of the retire path. `E = N·T_abs + r` with 0 < r < N is the
    # state the recovery's floor divide actually leaves in play.
    residue = 7
    g.gas_energy[fy, fx] += residue
    e_tile = int(g.gas_energy[fy, fx])
    assert e_tile > 0, "vacuous: the tile carried no energy to retire"

    before, books_before = _e_acct(g), _books_net(g)
    g.seal_tiles([(fy, fx)], materials.MAT_HULL)
    after, books_after = _e_acct(g), _books_net(g)

    assert not g.gas_is_accountable(fy, fx), "the tile did not actually seal"
    assert int(g.gas_energy[fy, fx]) == 0, (
        "a sealed tile must hold no stored gas energy")
    assert (after - before) == (books_after - books_before), (
        f"the books moved by {books_after - books_before} but "
        f"Σ_accountable gas_energy moved by {after - before}")
    # The retire is exactly the residue: everything the mirror COULD represent
    # went to the receivers as a MOVE, and only the sub-count remainder left
    # the books.
    assert (after - before) == -residue, (
        f"expected the seal to retire exactly the {residue}-count residue, "
        f"but Σ_accountable gas_energy moved by {after - before}")


def test_seal_moves_the_energy_to_the_receivers_at_the_sealed_T_abs():
    """The MOVED half, on its own: what the receivers gained is exactly
    `ΔN · T_abs(sealed tile)`, priced at the SEALED tile's temperature and not
    at the receivers' own — that is the difference between a conservative move
    and a mint whose size is set by whichever room happened to be hotter."""
    sim = _room_sim()
    g = sim.gmap
    fy, fx = 7, 7
    # Make the sealed tile MUCH hotter than its neighbours, so pricing at the
    # wrong end is a large, unmissable error rather than a rounding term.
    _seed_hot(g, np.s_[fy:fy + 1, fx:fx + 1], 5000.0)
    t_abs_seal = g.gas_t_abs_at(fy, fx)
    n_seal = g.gas_bulk_n_at(fy, fx)
    assert n_seal > 0

    e_recv_before = 0
    recv = [(fy - 1, fx), (fy + 1, fx), (fy, fx - 1), (fy, fx + 1)]
    for r in recv:
        e_recv_before += int(g.gas_energy[r])
    g.seal_tiles([(fy, fx)], materials.MAT_HULL)
    e_recv_after = sum(int(g.gas_energy[r]) for r in recv)

    assert e_recv_after - e_recv_before == n_seal * t_abs_seal, (
        f"receivers gained {e_recv_after - e_recv_before}, expected "
        f"ΔN·T_abs(sealed) = {n_seal * t_abs_seal}")


# ---------------------------------------------------------------------------
# Direction 2 — solid -> gas: BORN AT AMBIENT, mirror included.
# ---------------------------------------------------------------------------
def test_destroyed_wall_is_born_at_ambient_not_at_the_rooms_temperature():
    """`destroy_wall`'s seed is the arc's ONE true mint (design §2.7). It must
    be born at ambient — `E = N · T_AMB_raw`, mirror 0 — even when the room it
    opens into is an inferno. Inheriting the room's T here would make every
    blown wall a free heat source scaled by how hot the room already was, which
    is #54's entire family of bug."""
    sim = _room_sim()
    g = sim.gmap
    _seed_hot(g, np.s_[3:12, 3:12], 6000.0)
    # An interior hull tile with open neighbours: the ring at row 1 is the
    # outer hull, so take a tile on the inner face of the wall.
    fy, fx = 2, 7
    g.material[fy, fx] = materials.MAT_HULL
    g.on_tile_changed(fy, fx)
    assert g.solid[fy, fx]

    g.destroy_wall(fy, fx)

    assert g.gas_is_accountable(fy, fx), "the tile did not join the gas set"
    n = g.gas_bulk_n_at(fy, fx)
    assert n > 0, "vacuous: the seed placed no mass"
    assert int(g.temperature[fy, fx]) == 0, (
        f"the seed's mirror is {int(g.temperature[fy, fx])}, not ambient — it "
        "inherited the burning room's temperature")
    assert int(g.gas_energy[fy, fx]) == n * g.gas_t_amb_raw(), (
        "the seed's stored energy is not N·T_AMB_raw")


def test_thermal_solid_to_gas_leaves_no_stale_object_temperature():
    """A burnt-out crate becoming air must NOT leave its 1300 K sitting on the
    tile. The object's temperature was earned on the SOLIDS side of the ledger
    (D2: thermal solids stay on T) and stays there; the air that replaces it is
    born at ambient, stored energy and mirror together."""
    sim = _room_sim()
    g = sim.gmap
    fy, fx = 8, 8
    # Make the tile furniture (a thermal solid that still holds gas), heat the
    # OBJECT hard, then burn it away to air.
    g.material[fy, fx] = materials.MAT_FURNITURE
    g.on_tile_changed(fy, fx)
    assert g.thermal_solid[fy, fx], "fixture: the tile must be a thermal solid"
    assert not g.gas_is_accountable(fy, fx)
    g.temperature[fy, fx] = atmosphere_fixed.quantize_scalar(1300.0)

    before, books_before = _e_acct(g), _books_net(g)
    g.destroy_wall(fy, fx)
    after, books_after = _e_acct(g), _books_net(g)

    assert g.gas_is_accountable(fy, fx)
    assert int(g.temperature[fy, fx]) == 0, (
        f"stale object temperature {int(g.temperature[fy, fx])} survived onto "
        "the air tile")
    n = g.gas_bulk_n_at(fy, fx)
    assert int(g.gas_energy[fy, fx]) == n * g.gas_t_amb_raw()
    assert (after - before) == (books_after - books_before), (
        "the ts->gas flip moved the books by an unbooked amount")


# ---------------------------------------------------------------------------
# Both ways — the round trip.
# ---------------------------------------------------------------------------
def test_flip_both_ways_books_move_by_exactly_the_named_amount():
    """THE gate the design names. Seal a tile, then unseal it, with a hot room
    around it — and require that at EVERY step

        Δ(Σ_accountable gas_energy)  ==  Δ(the seam's own books)

    to the raw count. No solver step runs in between, so nothing but the
    structural edits can move either side: any drift is a seam that changed
    the field without booking it (or booked what it did not change)."""
    sim = _room_sim()
    g = sim.gmap
    _seed_hot(g, np.s_[3:12, 3:12], 900.0)
    fy, fx = 6, 6

    for leg, action in (("seal", lambda: g.seal_tiles([(fy, fx)],
                                                      materials.MAT_HULL)),
                        ("unseal", lambda: g.unseal_tiles([(fy, fx)]))):
        before, books_before = _e_acct(g), _books_net(g)
        action()
        after, books_after = _e_acct(g), _books_net(g)
        assert (after - before) == (books_after - books_before), (
            f"[{leg}] Σ_accountable gas_energy moved by {after - before} "
            f"but the seam booked {books_after - books_before}")

    assert g.gas_is_accountable(fy, fx), "the round trip left the tile sealed"


def test_unseal_is_a_withdrawal_priced_at_each_donors_own_T_abs():
    """`unseal_tiles` is a conservative WITHDRAWAL, not a mint (design §2.7,
    seam finding 3): each donor's share carries THAT DONOR's `T_abs`, so the
    grid total is unchanged to the raw count.

    Born-at-ambient would be wrong here in a way that shows up on every door
    cycle: it would mint or destroy `ΔN·(T_AMB − T_abs,donor)` each time, which
    in a burning or a vented room is not a rounding term. The test therefore
    sets the donors well away from ambient and requires the TOTAL to hold."""
    sim = _room_sim()
    g = sim.gmap
    _seed_hot(g, np.s_[3:12, 3:12], 2000.0)
    fy, fx = 6, 6
    g.seal_tiles([(fy, fx)], materials.MAT_HULL)

    before, books_before = _e_acct(g), _books_net(g)
    g.unseal_tiles([(fy, fx)])
    after, books_after = _e_acct(g), _books_net(g)

    # The withdrawal itself is internal to the accountable set: donors lose
    # exactly what the opened tile gains, so the seam's NET is the born-at-
    # ambient event of the membership flip and nothing else.
    assert (after - before) == (books_after - books_before)
    n = g.gas_bulk_n_at(fy, fx)
    assert n > 0, "vacuous: the opened tile was seeded with no mass"
    # Priced at the donors' T_abs, not at ambient: the opened tile is HOT.
    assert int(g.temperature[fy, fx]) > 0, (
        "the opened tile came out at ambient — the withdrawal was priced at "
        "T_AMB instead of at each donor's own T_abs")


def test_seam_books_channels_are_named_not_anonymous():
    """Support gate: every flip above must have gone through a NAMED channel.
    An empty books dict would make the equality assertions above green for the
    worst possible reason — nothing moved, and nothing was booked."""
    sim = _room_sim()
    g = sim.gmap
    _seed_hot(g, np.s_[3:12, 3:12], 900.0)
    g.seal_tiles([(6, 6)], materials.MAT_HULL)
    g.unseal_tiles([(6, 6)])
    assert g.gas_energy_books, "no seam channel was booked at all"
    assert any(k.startswith("seal") for k in g.gas_energy_books)
    assert any(k.startswith("unseal") for k in g.gas_energy_books)
