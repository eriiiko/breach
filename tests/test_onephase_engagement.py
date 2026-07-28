"""P6 — Overwatch, Ambush, marking, idle stance (design §9/§10/§11/§13).

The three ways a unit shoots without a shoot order being the thing it is doing
this instant, plus the marking that steers all of them.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from level_loader import LevelData  # noqa: E402
from simulation import engagement as E  # noqa: E402
from simulation import orders as O  # noqa: E402
from simulation.ruleset import OnePhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

EAST = 0.0
NORTH = math.pi / 2
WEST = math.pi
SOUTH = -math.pi / 2


def _level(h=48, w=48):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    return LevelData(name="onephase_engagement", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim():
    sim = Simulation(_level(), seed=3, breach_physics=None,
                     enable_recorder=False, ruleset=OnePhaseWEGO())
    sim.set_paused(False)
    return sim


def _marine(sim, x, y, name="m", facing=EAST):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    u.facing = facing
    return u


def _zombie(sim, x, y, name="z", hp=10_000):
    u = Unit(name, x=x, y=y, team=1)
    sim.add_unit(u)
    u.current_hp = hp
    u.zombie_activated = False
    return u


def _run(sim, n):
    for _ in range(n):
        sim.step()
        sim.set_paused(False)


def _overwatch(sim, u, tx, ty, cone=None):
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, tx, ty, 0,
                                   cone_half_deg=cone))
    _run(sim, 8)                       # let the 0.25 s establish step finish


# ---------------------------------------------------------------------------
# Overwatch (§9)
# ---------------------------------------------------------------------------
def test_overwatch_engages_a_target_entering_the_cone():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    z = _zombie(sim, 24.0, 24.0)
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp < hp0


def test_overwatch_ignores_a_target_outside_the_cone():
    """§9: the cone's PRIMARY purpose is target control — targets outside it
    are ignored, so narrowing is indirect target selection."""
    sim = _sim()
    m = _marine(sim, 24.0, 24.0)
    _overwatch(sim, m, 44, 24, cone=15.0)      # a narrow cone pointing east
    z = _zombie(sim, 24.0, 40.0)               # due south of the marine
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp == hp0


def test_narrowing_the_cone_is_target_selection():
    wide = _sim()
    mw = _marine(wide, 24.0, 24.0)
    _overwatch(wide, mw, 44, 24, cone=110.0)
    zw = _zombie(wide, 34.0, 34.0)             # off to the south-east
    hp0 = zw.current_hp
    _run(wide, 24)
    assert zw.current_hp < hp0

    narrow = _sim()
    mn = _marine(narrow, 24.0, 24.0)
    _overwatch(narrow, mn, 44, 24, cone=10.0)
    zn = _zombie(narrow, 34.0, 34.0)
    hp1 = zn.current_hp
    _run(narrow, 24)
    assert zn.current_hp == hp1


def test_overwatch_engages_continuously_not_once():
    """§9: continuous engagement (normal weapon behaviour while targets are in
    the cone), not one reaction shot."""
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    z = _zombie(sim, 24.0, 24.0)
    _run(sim, 12)
    after_first = z.current_hp
    _run(sim, 24)
    assert z.current_hp < after_first


def test_overwatch_survives_the_round_seam():
    """§9: overwatch is a STATE — it persists across rounds until replaced."""
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    _run(sim, sim.ticks_per_round)
    assert sim.round_index >= 1
    assert m.overwatch_facing is not None
    z = _zombie(sim, 24.0, 24.0)
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp < hp0, "overwatch stopped working after the seam"


def test_an_active_order_supersedes_standing_watch():
    """Issuing an order takes the unit off watch for the duration of the
    order; the watch resumes when it completes."""
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    z = _zombie(sim, 24.0, 24.0)
    sim.apply_action(m.id, O.Order(O.ORDER_MOVE, 10, 40, 0))
    hp0 = z.current_hp
    _run(sim, 20)
    assert z.current_hp == hp0, "the unit shot while executing a move order"


def test_overwatch_prefers_a_marked_target():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 40, 24, cone=110.0)
    near = _zombie(sim, 20.0, 24.0, name="near")
    far = _zombie(sim, 30.0, 24.0, name="far")
    sim.marks.setdefault(0, set()).add(far.id)
    assert E.overwatch_target(sim, m) is far


def test_target_ranking_falls_back_to_exposure_then_cone_centre():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 40, 24, cone=110.0)
    centred = _zombie(sim, 30.0, 24.0, name="centred")
    offset = _zombie(sim, 30.0, 34.0, name="offset")
    ranked = E.rank_targets(sim, m, [offset, centred])
    assert ranked[0] is centred, "the cone-centre tie-break did not apply"


# ---------------------------------------------------------------------------
# Ambush (§10)
# ---------------------------------------------------------------------------
def _ambush(sim, u, target, pre_move=None):
    if pre_move is not None:
        sim.apply_action(u.id, O.Order(O.ORDER_MOVE, pre_move[0], pre_move[1],
                                       0))
    sim.apply_action(u.id, O.Order(O.ORDER_AMBUSH, target.tile_x,
                                   target.tile_y, 0, target_unit_id=target.id))


def test_nobody_fires_until_every_member_is_ready():
    """§10's fire condition: the instant ALL group members on that target are
    ready, all fire simultaneously — and not one tick before."""
    sim = _sim()
    ready = _marine(sim, 20.0, 20.0, name="ready")
    walker = _marine(sim, 20.0, 40.0, name="walker")
    z = _zombie(sim, 30.0, 20.0)
    _ambush(sim, ready, z)
    _ambush(sim, walker, z, pre_move=(20, 30))   # has to walk first
    hp0 = z.current_hp
    _run(sim, 6)
    assert z.current_hp == hp0, "a lone ready member opened fire early"
    assert z.id not in sim.ambush_released


def test_the_group_fires_together_once_the_last_member_arrives():
    sim = _sim()
    ready = _marine(sim, 20.0, 20.0, name="ready")
    walker = _marine(sim, 20.0, 26.0, name="walker")
    z = _zombie(sim, 30.0, 20.0)
    _ambush(sim, ready, z)
    _ambush(sim, walker, z, pre_move=(20, 23))
    hp0 = z.current_hp
    _run(sim, 60)
    assert z.id in sim.ambush_released
    assert z.current_hp < hp0


def test_readiness_is_only_reaching_the_order():
    """Erik's simplification: no LOS condition, no other semantics."""
    sim = _sim()
    m = _marine(sim, 20.0, 20.0)
    z = _zombie(sim, 30.0, 20.0)
    _ambush(sim, m, z)
    _run(sim, 2)
    assert E.is_ready(sim, m) is True


def test_a_sprung_ambush_releases_the_ready_members():
    """§10: if ANY group member is fired upon, all ready members open fire
    immediately; not-yet-ready members continue their queues."""
    sim = _sim()
    ready = _marine(sim, 20.0, 20.0, name="ready")
    walker = _marine(sim, 20.0, 44.0, name="walker")
    z = _zombie(sim, 30.0, 20.0)
    _ambush(sim, ready, z)
    _ambush(sim, walker, z, pre_move=(20, 30))
    _run(sim, 4)
    assert z.id not in sim.ambush_released
    ready.recent_attackers[z.id] = 1            # somebody shot at us
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.id in sim.ambush_released
    assert z.current_hp < hp0


def test_a_dead_member_does_not_deadlock_the_group():
    sim = _sim()
    ready = _marine(sim, 20.0, 20.0, name="ready")
    doomed = _marine(sim, 20.0, 44.0, name="doomed")
    z = _zombie(sim, 30.0, 20.0)
    _ambush(sim, ready, z)
    _ambush(sim, doomed, z, pre_move=(20, 30))
    _run(sim, 4)
    assert z.id not in sim.ambush_released
    doomed.alive = False
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.id in sim.ambush_released
    assert z.current_hp < hp0


def test_an_unready_group_reverts_to_idle_at_round_end():
    """§10's timeout backstop — no infinite holds."""
    sim = _sim()
    ready = _marine(sim, 20.0, 20.0, name="ready")
    walker = _marine(sim, 20.0, 44.0, name="walker")
    # Far enough away that it cannot close and SPRING the ambush during the
    # round — this test is about the timeout, not the sprung path.
    z = _zombie(sim, 42.0, 42.0)
    _ambush(sim, ready, z)
    # A 40-tile walk cannot finish inside a 96-tick round, so this group can
    # never become ready — exactly the case the backstop exists for.
    _ambush(sim, walker, z, pre_move=(20, 4))
    _run(sim, sim.ticks_per_round + 1)
    assert z.id not in sim.ambush_released
    assert not any(o.order_type == O.ORDER_AMBUSH for o in ready.orders), \
        "the stale ambush order survived the round-end timeout"


def test_a_released_ambush_is_not_dropped_at_round_end():
    sim = _sim()
    m = _marine(sim, 20.0, 20.0)
    z = _zombie(sim, 30.0, 20.0)
    _ambush(sim, m, z)
    _run(sim, 4)
    assert z.id in sim.ambush_released
    _run(sim, sim.ticks_per_round)
    assert any(o.order_type == O.ORDER_AMBUSH for o in m.orders)


def test_the_breach_choreography_composes():
    """§10's worked example: the breacher's queue [move -> detonate -> Ambush]
    means everyone else, already waiting on the barrier, fires the moment the
    breacher arrives at its Ambush step."""
    sim = _sim()
    a = _marine(sim, 20.0, 18.0, name="stacked_a")
    b = _marine(sim, 20.0, 22.0, name="stacked_b")
    breacher = _marine(sim, 20.0, 30.0, name="breacher")
    z = _zombie(sim, 32.0, 20.0)
    _ambush(sim, a, z)
    _ambush(sim, b, z)
    _ambush(sim, breacher, z, pre_move=(20, 26))
    hp0 = z.current_hp
    _run(sim, 8)
    assert z.current_hp == hp0, "the stack fired before the breacher arrived"
    _run(sim, 80)
    assert z.id in sim.ambush_released
    assert z.current_hp < hp0


def test_ambush_needs_no_signal_bus():
    """§10: "No SignalBus needed in v1 — Ambush is a per-group readiness
    counter in the sim"."""
    sim = _sim()
    m = _marine(sim, 20.0, 20.0)
    z = _zombie(sim, 30.0, 20.0)
    _ambush(sim, m, z)
    _run(sim, 4)
    assert sim._signal_bus is None
    assert sim.ambush_released == {z.id: pytest.approx(sim.ambush_released[z.id])}


# ---------------------------------------------------------------------------
# Marking (§11)
# ---------------------------------------------------------------------------
def test_marks_are_per_team_and_persist():
    sim = _sim()
    m = _marine(sim, 20.0, 20.0)
    z = _zombie(sim, 30.0, 20.0)
    sim.apply_action(m.id, O.Order(O.ORDER_MARK, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 2)
    assert z.id in sim.marks[0]
    _run(sim, sim.ticks_per_round + 5)
    assert z.id in sim.marks[0], "marks persist until unmarked or death"


def test_v1_has_a_single_focus_level():
    """§11: graded priorities 1-5 are a future refinement — v1 is a set."""
    sim = _sim()
    assert isinstance(sim.marks, dict)
    sim.marks.setdefault(0, set()).add(7)
    assert isinstance(sim.marks[0], set)


# ---------------------------------------------------------------------------
# Idle stance (§13)
# ---------------------------------------------------------------------------
def test_an_idle_unit_returns_fire_at_its_attacker():
    sim = _sim()
    m = _marine(sim, 20.0, 24.0)
    z = _zombie(sim, 30.0, 24.0)
    m.recent_attackers[z.id] = 1
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp < hp0


def test_an_idle_unit_does_not_free_fire():
    """§13: "returns fire at attackers … but does not free-fire at everything
    it sees. Return fire is a floor, not an AI"."""
    sim = _sim()
    m = _marine(sim, 20.0, 24.0)
    z = _zombie(sim, 30.0, 24.0)
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp == hp0


def test_idle_return_fire_prefers_a_marked_attacker():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    near = _zombie(sim, 20.0, 24.0, name="near")
    far = _zombie(sim, 30.0, 24.0, name="far")
    m.recent_attackers[near.id] = 1
    m.recent_attackers[far.id] = 1
    sim.marks.setdefault(0, set()).add(far.id)
    assert E.idle_target(sim, m) is far


def test_the_attacker_memory_clears_at_the_round_boundary():
    sim = _sim()
    m = _marine(sim, 20.0, 24.0)
    z = _zombie(sim, 30.0, 24.0)
    m.recent_attackers[z.id] = 1
    _run(sim, sim.ticks_per_round)
    assert m.recent_attackers == {}


def test_idle_return_fire_can_be_turned_off(monkeypatch):
    """§20 item 5: return-fire vs do-nothing is a dial."""
    monkeypatch.setattr(CFG.onephase, "idle_return_fire", False)
    sim = _sim()
    m = _marine(sim, 20.0, 24.0)
    z = _zombie(sim, 30.0, 24.0)
    m.recent_attackers[z.id] = 1
    assert E.idle_target(sim, m) is None


def test_an_idle_unit_will_not_shoot_what_it_cannot_see():
    sim = _sim()
    m = _marine(sim, 20.0, 24.0, facing=WEST)     # facing away
    z = _zombie(sim, 34.0, 24.0)
    m.recent_attackers[z.id] = 1
    assert E.idle_target(sim, m) is None


# ---------------------------------------------------------------------------
# Attribution — the hook the idle stance rides on
# ---------------------------------------------------------------------------
def test_being_shot_records_the_shooter():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    z = _zombie(sim, 20.0, 24.0)
    sim.apply_action(m.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 24)
    assert m.id in z.recent_attackers


def test_environmental_damage_is_not_an_attacker():
    """A source_id of -1 (heat, an unattributed blast) must not become
    somebody to return fire at."""
    from simulation.damage import DamagePacket, KINETIC, apply_packet
    sim = _sim()
    m = _marine(sim, 20.0, 24.0)
    apply_packet(m, DamagePacket(amount=1, dtype=KINETIC, source_id=-1),
                 None, source="heat")
    assert m.recent_attackers == {}


# ---------------------------------------------------------------------------
# Overwatch ends when the posture is REPLACED (Erik, 3rd play session: the
# cones "stay even after a unit leaves overwatch, which they shouldn't")
# ---------------------------------------------------------------------------
def test_a_new_plan_ends_the_overwatch_posture():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    assert m.overwatch_facing is not None
    sim.begin_new_plan(m.id)
    assert m.overwatch_facing is None, "the watch outlived its replacement"
    assert m.overwatch_half_deg is None


def test_an_ordinary_order_ends_it_too():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    _run(sim, sim.ticks_per_round)          # into the next round
    assert m.overwatch_facing is not None, "it must survive a quiet seam"
    sim.apply_action(m.id, O.Order(O.ORDER_MOVE, 10, 30, 0))
    assert m.overwatch_facing is None


def test_a_replaced_watch_stops_engaging():
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    sim.begin_new_plan(m.id)
    z = _zombie(sim, 24.0, 24.0)
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp == hp0
    assert E.on_overwatch(m) is False


def test_submitting_nothing_keeps_the_watch_standing():
    """§9's actual rule: it persists across rounds until REPLACED — and doing
    nothing is not a replacement."""
    sim = _sim()
    m = _marine(sim, 10.0, 24.0)
    _overwatch(sim, m, 30, 24)
    _run(sim, sim.ticks_per_round * 2)
    assert m.overwatch_facing is not None
    z = _zombie(sim, 24.0, 24.0)
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp < hp0
