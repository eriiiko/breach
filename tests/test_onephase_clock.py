"""P1 — the OnePhaseWEGO round clock (onephase_wego design §2/§3/§13).

What these pin, in the design's own terms:

- ONE phase, ~4 s rounds off ``CFG.clock.round_duration_seconds`` (§2).
- The clock is FREE-RUNNING: ``sim.tick`` never rewinds, and within-round
  position is a modulo (the kickoff doc's §2.1 — a rewound clock cannot carry
  a cooldown across a seam, which §13 requires).
- **Invisible seams** (§13): crossing a round boundary changes NOTHING about
  the world. Positions are not snapped (§4 removes that outright), orders are
  not cleared, AP is not refilled, obstacles are not reset, in-flight
  projectiles survive, and every carried tick-deadline still means what it
  meant one tick earlier.
- AP is dead (§3): the cost policy charges nothing.
- ``TwoPhaseWEGO`` is untouched by all of the above.
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

from config import CFG  # noqa: E402
from level_loader import LevelData
from simulation.combat import Projectile
from simulation.orders import (  # noqa: E402
    ORDER_GRENADE, ORDER_MOVE, ORDER_MOVE_ATTACK, ORDER_MOVE_COVER,
    ORDER_SPRINT, Order,
)
from simulation.ruleset import OnePhaseWEGO, TwoPhaseWEGO
from simulation.simulation import Simulation
from simulation.unit import Unit


# ---------------------------------------------------------------------------
# A minimal walled room: no physics module, so the tick body reduces to the
# unit/order path and the round clock — exactly what P1 is about.
# ---------------------------------------------------------------------------
def _level(h=24, w=24):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    return LevelData(name="onephase_clock", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(ruleset=None):
    sim = Simulation(_level(), seed=42, breach_physics=None,
                     enable_recorder=False,
                     ruleset=ruleset if ruleset is not None else OnePhaseWEGO())
    sim.set_paused(False)
    return sim


def _marine(sim, x=4.0, y=4.0, name="m"):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    return u


def _run(sim, n):
    for _ in range(n):
        sim.step()
        sim.set_paused(False)      # step through the planning pause


# ---------------------------------------------------------------------------
# Round geometry (§2)
# ---------------------------------------------------------------------------
def test_round_is_one_phase_of_round_duration_seconds():
    """96 ticks @ 24 Hz from the 4.0 s dial — and it is NOT the two-phase
    round length, which stays what it always was."""
    sim = _sim()
    tps = CFG.clock.ticks_per_second
    assert sim.ticks_per_round == round(CFG.clock.round_duration_seconds * tps)
    assert sim.ticks_per_round == 96
    # The two-phase round (240 ticks) is untouched and still available.
    assert CFG.clock.ticks_per_round == 240
    assert sim.ticks_per_round != CFG.clock.ticks_per_round


def test_round_tick_and_index_are_modulo_of_a_free_running_tick():
    sim = _sim()
    tpr = sim.ticks_per_round
    _run(sim, tpr + 5)
    # THE structural pin: the absolute tick kept counting past the boundary.
    assert sim.tick == tpr + 5
    assert sim.round_tick == 5
    assert sim.round_index == 1
    assert sim.round_start_tick() == tpr


def test_two_phase_wego_still_rewinds_its_tick():
    """The sibling ruleset is untouched — its clock still wraps to 0, which is
    what every shipped golden was recorded against."""
    sim = _sim(TwoPhaseWEGO())
    _run(sim, CFG.clock.ticks_per_round)
    assert sim.tick == 0
    assert sim.turn_number == 2


# ---------------------------------------------------------------------------
# The pause is the ONLY thing the boundary does to the player (§2)
# ---------------------------------------------------------------------------
def test_boundary_pauses_exactly_once_per_round():
    sim = _sim()
    tpr = sim.ticks_per_round
    for _ in range(tpr - 1):
        sim.step()
        assert not sim.is_paused()
    sim.step()                       # the tick that completes the round
    assert sim.is_paused()
    assert sim.round_tick == 0
    assert sim.round_index == 1


# ---------------------------------------------------------------------------
# Invisible seams (§13) — the heart of P1
# ---------------------------------------------------------------------------
def test_seam_does_not_snap_positions():
    """§4: the end-of-round integer-tile snap is REMOVED. A unit mid-tile at
    the boundary stays exactly where it was."""
    sim = _sim()
    u = _marine(sim, x=4.37, y=6.82)
    _run(sim, sim.ticks_per_round)
    assert u.x == pytest.approx(4.37)
    assert u.y == pytest.approx(6.82)


def test_two_phase_wego_still_snaps_positions():
    """Contrast: the shipped ruleset's teardown is unchanged."""
    sim = _sim(TwoPhaseWEGO())
    u = _marine(sim, x=4.37, y=6.82)
    _run(sim, CFG.clock.ticks_per_round)
    assert u.x == 4.0
    assert u.y == 7.0


def test_seam_does_not_clear_orders():
    """§13 + §5: an unfinished plan continues across the seam; only the
    player issuing NEW orders replaces a queue (new orders interrupt by
    default). The move is long enough to still be running at the boundary."""
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    assert sim.apply_action(u.id, Order(ORDER_MOVE, 4, 20, 0))
    assert len(u.orders) == 1
    _run(sim, sim.ticks_per_round)
    assert sim.round_index == 1
    assert len(u.orders) == 1, "the seam ate a queued order"


def test_legacy_order_types_are_refused_by_this_ruleset():
    """OnePhaseWEGO has its own vocabulary (§5): Sprint and Move-w/-Cover are
    removed as separate orders, and Move&Attack's auto-attack-while-moving is
    replaced by the explicit Move & Shoot. A legacy id has no registry row, so
    it is rejected rather than silently reinterpreted."""
    sim = _sim()
    u = _marine(sim)
    for legacy in (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT):
        assert sim.apply_action(u.id, Order(legacy, 9, 9, 0)) is False
    assert u.orders == []


def test_seam_does_not_refill_ap_or_reset_fire_timers():
    """Every carried tick-deadline still means what it meant one tick before
    the seam — the whole reason the clock is monotonic."""
    sim = _sim()
    u = _marine(sim)
    u.ap = [0, 0]
    u.last_fire_tick = 3
    u.reload_done_tick = 7
    u.current_mag = 2
    _run(sim, sim.ticks_per_round)
    assert u.ap == [0, 0], "AP is dead — nothing should refill it"
    assert u.last_fire_tick == 3
    assert u.reload_done_tick == 7
    assert u.current_mag == 2, "the seam topped a magazine off"


def test_seam_never_runs_the_two_phase_teardown():
    """``_end_round`` — the snap / clear / AP-refill / obstacle-reset / tick-
    rewind bundle — is never reached under this ruleset.

    Stated as a direct assertion rather than through its side effects because
    one of them is genuinely unobservable: the obstacle reset only undid what
    the per-tick ``stamp_units`` (step slot 6) already rebuilds from the living
    units every tick, ruleset-independently. The other four ARE observable and
    are pinned by the tests around this one.
    """
    sim = _sim()
    _marine(sim)

    def _boom():
        raise AssertionError("_end_round ran under OnePhaseWEGO")

    sim._end_round = _boom
    _run(sim, sim.ticks_per_round * 2 + 3)
    assert sim.round_index == 2


def test_seam_keeps_undetonated_projectiles_and_prunes_spent_ones():
    """In-flight projectiles persist (§13). Already-detonated ones are inert
    bookkeeping and get pruned — an unobservable housekeeping step."""
    sim = _sim()
    live = Projectile(ORDER_GRENADE, 2, 2, 8, 8, fuse_seconds=999.0,
                      thrown_tick=0)
    spent = Projectile(ORDER_GRENADE, 3, 3, 9, 9, fuse_seconds=999.0,
                       thrown_tick=0)
    spent.detonated = True
    sim.projectiles.extend([live, spent])
    _run(sim, sim.ticks_per_round)
    assert live in sim.projectiles
    assert spent not in sim.projectiles


def test_a_deadline_set_before_the_seam_still_expires_after_it():
    """The point of the monotonic clock, stated as behaviour: a cooldown set
    near the end of a round is still pending early in the next one, and
    expires on the tick it was always going to expire on."""
    sim = _sim()
    tpr = sim.ticks_per_round
    _run(sim, tpr - 4)
    deadline = sim.tick + 10          # 6 ticks into the NEXT round
    _run(sim, 4)
    assert sim.round_index == 1
    assert sim.tick < deadline, "the seam swallowed a live deadline"
    _run(sim, 6)
    assert sim.tick == deadline


# ---------------------------------------------------------------------------
# AP is dead (§3)
# ---------------------------------------------------------------------------
def test_cost_policy_charges_nothing():
    sim = _sim()
    u = _marine(sim)
    u.ap = [0, 0]
    order = Order(ORDER_MOVE_ATTACK, 9, 9, 0)
    order.ap_cost = 99
    assert sim.ruleset.validate_and_cost(sim, u, order) is True
    assert u.ap == [0, 0]
    sim.ruleset.refund(sim, u, order)
    assert u.ap == [0, 0]


def test_orders_are_accepted_with_zero_ap():
    """The AP gate that would reject this under TwoPhaseWEGO is simply gone."""
    sim = _sim()
    u = _marine(sim)
    u.ap = [0, 0]
    u.has_grenade = 1
    assert sim.apply_action(u.id, Order(ORDER_GRENADE, 9, 9, 0,
                                        grenade_fuse=2.0)) is True
    sim_two = _sim(TwoPhaseWEGO())
    u2 = _marine(sim_two)
    u2.ap = [0, 0]
    u2.has_grenade = 1
    assert sim_two.apply_action(u2.id, Order(ORDER_GRENADE, 9, 9, 0,
                                             grenade_fuse=2.0)) is False


# ---------------------------------------------------------------------------
# Episode boundary (§1 — the RL contract)
# ---------------------------------------------------------------------------
def test_is_terminal_is_one_side_eliminated_not_round_complete():
    sim = _sim()
    _marine(sim)
    z = Unit("z", x=12.0, y=12.0, team=1)
    sim.add_unit(z)
    _run(sim, sim.ticks_per_round + 1)
    assert not sim.is_terminal(), "a completed round is not an episode end"
    z.alive = False
    assert sim.is_terminal()
