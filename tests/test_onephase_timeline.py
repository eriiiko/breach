"""P3 — the compiled timeline (onephase_wego design §3/§5/§6/§7/§13/§14).

"A unit's plan is a timeline — (move to A, arrive 1.4 s) -> (GCD to 1.9 s) ->
(fire to 3.0 s)." These pin that sentence: the cursor walk, the GCD and
cooldown gates, aim-relative speeds, situational spread, the interrupt
semantics, and the two invariants the whole arc leans on —

  1. the schedule is authoritative for TIME, execution may under-deliver in
     SPACE (a blocked path costs ground, never the following steps' times);
  2. every tick in a plan is ABSOLUTE, so a plan crossing the round seam
     needs no fixups.
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
from simulation import orders as O  # noqa: E402
from simulation import timeline as T  # noqa: E402
from simulation.ruleset import OnePhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

TPS = CFG.clock.ticks_per_second
GCD = CFG.onephase.gcd_ticks


def _level(h=32, w=32, walls=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x) in walls:
        tm[y, x] = 1
    return LevelData(name="onephase_timeline", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(walls=()):
    sim = Simulation(_level(walls=walls), seed=7, breach_physics=None,
                     enable_recorder=False, ruleset=OnePhaseWEGO())
    sim.set_paused(False)
    return sim


def _marine(sim, x=4.0, y=4.0, name="m"):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    return u


def _zombie(sim, x=10.0, y=4.0, name="z"):
    u = Unit(name, x=x, y=y, team=1)
    sim.add_unit(u)
    return u


def _run(sim, n):
    for _ in range(n):
        sim.step()
        sim.set_paused(False)


def _move(tx, ty, **kw):
    return O.Order(O.ORDER_MOVE, tx, ty, phase=0, **kw)


# ---------------------------------------------------------------------------
# The cursor walk (§3)
# ---------------------------------------------------------------------------
def test_a_plan_is_a_sequence_of_scheduled_steps():
    sim = _sim()
    u = _marine(sim)
    assert sim.apply_action(u.id, _move(4, 10))
    plan = u.plan
    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.action.name == "move"
    assert step.start_tick == sim.tick
    assert step.end_tick == step.start_tick + len(step.path)
    assert len(step.path) > 0


def test_steps_run_back_to_back_in_queue_order():
    sim = _sim()
    u = _marine(sim)
    sim.apply_action(u.id, _move(4, 8))
    sim.apply_action(u.id, _move(8, 8))
    a, b = u.plan.steps
    assert a.end_tick <= b.start_tick
    assert b.start_tick == a.end_tick, "no unexplained gap between moves"


def test_movement_never_charges_the_gcd():
    """§3: the GCD is triggered by actions, never by movement."""
    sim = _sim()
    u = _marine(sim)
    sim.apply_action(u.id, _move(4, 6))
    sim.apply_action(u.id, _move(4, 8))
    a, b = u.plan.steps
    assert b.start_tick == a.end_tick


def test_an_action_after_an_action_waits_out_the_gcd():
    sim = _sim()
    u = _marine(sim)
    z = _zombie(sim)
    sim.apply_action(u.id, O.Order(O.ORDER_MARK, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 10, 4, 0))
    mark, ow = u.plan.steps
    # Mark itself does not trigger the GCD (§11 — a command, not a physical
    # act), so overwatch starts immediately after it...
    assert ow.start_tick == mark.end_tick
    # ...but overwatch DOES, so anything after it waits out the GCD.
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 4, 10, 0))
    ow1, ow2 = u.plan.steps[1], u.plan.steps[2]
    assert ow2.start_tick >= ow1.start_tick + GCD


def test_gcd_is_charged_from_the_start_not_the_end():
    """§3: "the GCD gates *changing* action". For a step longer than the GCD
    the duration dominates; for a short one the GCD is the floor."""
    sim = _sim()
    u = _marine(sim)
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 10, 4, 0))
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 4, 10, 0))
    a, b = u.plan.steps
    assert a.duration_ticks() == sim.actions_table.get(
        "overwatch").duration_ticks
    assert b.start_tick == a.start_tick + GCD


def test_weapon_swap_has_its_own_non_global_cooldown():
    """§3: swapping is free except a 0.75 s cooldown that is NOT the GCD — so
    a swap does not delay a following action by the GCD, but a SECOND swap
    waits out the swap cooldown."""
    sim = _sim()
    u = _marine(sim)
    u.loadout = ["k5_carbine", "k5_carbine"]
    sim.apply_action(u.id, O.Order(O.ORDER_SWAP, 0, 0, 0))
    sim.apply_action(u.id, O.Order(O.ORDER_SWAP, 0, 0, 0))
    a, b = u.plan.steps
    assert b.start_tick == a.end_tick + CFG.onephase.weapon_swap_ticks


def test_hold_until_t_waits_for_its_moment():
    """§5's sequencing verb: the wait IS the step."""
    sim = _sim()
    u = _marine(sim)
    release = sim.tick + 40
    sim.apply_action(u.id, O.Order(O.ORDER_HOLD, 0, 0, 0, start_tick=release))
    sim.apply_action(u.id, _move(4, 8))
    hold, move = u.plan.steps
    assert hold.end_tick == release
    assert move.start_tick == release


def test_a_channeled_action_occupies_its_full_duration():
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 1
    sim.apply_action(u.id, O.Order(O.ORDER_EXPLOSIVE, 6, 4, 0,
                                   action_name="plant_charge"))
    step = u.plan.steps[0]
    assert step.action.interruptible is False
    assert step.duration_ticks() == TPS      # 1.0 s


# ---------------------------------------------------------------------------
# Sustained steps (§9's "persists until replaced")
# ---------------------------------------------------------------------------
def test_a_sustained_tail_runs_indefinitely():
    sim = _sim()
    u = _marine(sim)
    z = _zombie(sim)
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    step = u.plan.steps[0]
    assert step.end_tick is T.INDEFINITE
    assert step.contains(sim.tick + 10_000)


def test_a_sustained_step_is_closed_by_whatever_follows_it():
    sim = _sim()
    u = _marine(sim)
    z = _zombie(sim)
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    sim.apply_action(u.id, _move(4, 8))
    shoot, move = u.plan.steps
    assert shoot.end_tick == move.start_tick
    assert move.start_tick >= shoot.start_tick + GCD, "GCD floors the salvo"


# ---------------------------------------------------------------------------
# Aim-relative speed (§6)
# ---------------------------------------------------------------------------
def test_speed_table_matches_the_design():
    assert T.speed_pct(O.ORDER_MOVE, False) == 1.0
    assert T.speed_pct(O.ORDER_MOVE_SHOOT, False) == \
        CFG.onephase.move_shoot_speed_pct
    assert T.speed_pct(O.ORDER_MOVE_SHOOT, True) == \
        CFG.onephase.move_shoot_reverse_speed_pct


def test_plain_move_is_full_speed_and_move_shoot_is_slower():
    sim = _sim()
    a = _marine(sim, x=4.0, y=4.0, name="a")
    b = _marine(sim, x=4.0, y=12.0, name="b")
    z = _zombie(sim, x=4.0, y=2.0)          # ahead of b's travel? see below
    sim.apply_action(a.id, _move(4, 20))
    sim.apply_action(b.id, O.Order(O.ORDER_MOVE_SHOOT, 4, 20, 0,
                                   target_unit_id=z.id))
    fast = len(a.plan.steps[0].path)
    slow = len(b.plan.steps[0].path)
    assert slow > fast, "move & shoot must be slower than move"


def test_reversing_is_slower_than_advancing():
    """Backpedaling out of a room with your gun on the door is deliberately
    slow (§6) — same path, opposite aim."""
    sim = _sim()
    fwd = _marine(sim, x=4.0, y=10.0, name="fwd")
    rev = _marine(sim, x=10.0, y=10.0, name="rev")
    # Both walk +y; fwd aims where it is going, rev aims back where it came.
    sim.apply_action(fwd.id, O.Order(O.ORDER_MOVE_SHOOT, 4, 20, 0,
                                     aim_anchor=(4, 26)))
    sim.apply_action(rev.id, O.Order(O.ORDER_MOVE_SHOOT, 10, 20, 0,
                                     aim_anchor=(10, 2)))
    assert len(rev.plan.steps[0].path) > len(fwd.plan.steps[0].path)


def test_is_reversing_is_a_dot_product_against_the_dial():
    assert T.is_reversing(0, 1, 0, -1) is True       # 180 deg apart
    assert T.is_reversing(0, 1, 0, 1) is False       # aligned
    assert T.is_reversing(0, 1, 1, 0) is True        # 90 deg == the dial
    assert T.is_reversing(0, 0, 0, 1) is False       # no travel, no opinion


def test_terrain_and_aim_multipliers_compose():
    """Terrain first (the shipped mobility average), then the §6 fraction."""
    sim = _sim()
    u = _marine(sim)
    full = T.tile_cadence(sim.gmap, u, 8, 4, O.ORDER_MOVE, False)
    half = T.tile_cadence(sim.gmap, u, 8, 4, O.ORDER_MOVE_SHOOT, True)
    assert full == CFG.movement.marine_sprint_ticks_per_tile
    assert half == int(full / CFG.onephase.move_shoot_reverse_speed_pct + 0.5)


# ---------------------------------------------------------------------------
# Accuracy is spread (§7) — no to-hit roll anywhere
# ---------------------------------------------------------------------------
def test_spread_opens_on_the_move_and_further_when_reversing():
    sim = _sim()
    w = sim.weapons_tables.weapons.by_name["k5_carbine"]
    stand = T.spread_deg_for(w, O.ORDER_SHOOT)
    moving = T.spread_deg_for(w, O.ORDER_MOVE_SHOOT)
    backing = T.spread_deg_for(w, O.ORDER_MOVE_SHOOT, reversing=True)
    assert stand == w.spread_deg
    assert moving == pytest.approx(stand * CFG.onephase.spread_move_shoot_mult)
    assert backing == pytest.approx(stand * CFG.onephase.spread_reverse_mult)
    assert backing > moving > stand


def test_overwatch_tightens_the_cone():
    sim = _sim()
    w = sim.weapons_tables.weapons.by_name["k5_carbine"]
    assert T.spread_deg_for(w, O.ORDER_SHOOT, overwatch=True) < \
        T.spread_deg_for(w, O.ORDER_SHOOT)


def test_suppressed_aim_still_falls_back_to_the_snap_cone():
    sim = _sim()
    w = sim.weapons_tables.weapons.by_name["k5_carbine"]
    assert T.spread_deg_for(w, O.ORDER_SHOOT, can_aim=False) == \
        w.spread_snap_deg


# ---------------------------------------------------------------------------
# Execution — invariant 1: time is authoritative, space may under-deliver
# ---------------------------------------------------------------------------
def test_a_unit_walks_its_compiled_path():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 9))
    n = len(u.plan.steps[0].path)
    _run(sim, n)
    assert u.tile_y == 9 and u.tile_x == 4


def test_a_blocked_path_halts_in_place_and_does_not_shift_later_steps():
    """§14's v1 behaviour: halt in place, continue remaining non-move orders,
    no auto-repath, replan next round. The critical half is the SCHEDULE —
    the following step still starts on the tick the player was shown."""
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 14))
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 10, 14, 0))
    move, ow = u.plan.steps
    promised_ow_start = ow.start_tick

    _run(sim, 6)
    y_before = u.y
    # Drop a wall across the corridor mid-move.
    sim.gmap.material[int(u.y) + 3, :] = 1
    sim.gmap.solid[int(u.y) + 3, :] = True
    sim.gmap.obstacles[int(u.y) + 3, :] = True
    _run(sim, 30)

    assert move.blocked is True
    assert u.y < 14, "the unit should not have reached a walled-off target"
    assert u.y >= y_before
    assert ow.start_tick == promised_ow_start, "a block moved the schedule"


def test_no_auto_repath_after_a_block():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 14))
    _run(sim, 4)
    sim.gmap.material[8, :] = 1
    sim.gmap.solid[8, :] = True
    sim.gmap.obstacles[8, :] = True
    path_before = list(u.plan.steps[0].path)
    _run(sim, 40)
    assert u.plan.steps[0].path == path_before, "the plan repathed itself"


# ---------------------------------------------------------------------------
# Execution — invariant 2: absolute ticks cross the seam untouched (§13)
# ---------------------------------------------------------------------------
def test_a_plan_spanning_the_round_boundary_is_not_disturbed():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    tpr = sim.ticks_per_round
    _run(sim, tpr - 8)                       # 8 ticks left in this round
    sim.apply_action(u.id, _move(4, 20))
    step = u.plan.steps[0]
    assert step.end_tick > sim.round_start_tick() + tpr, "test needs a spill"
    promised_end = step.end_tick
    _run(sim, 20)
    assert sim.round_index == 1
    assert step.end_tick == promised_end
    assert step.blocked is False


def test_a_sustained_order_keeps_running_into_the_next_round():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    # Burn most of the round FIRST, with no enemy on the map, so the run-up
    # cannot itself change the outcome; then set up the shot near the seam.
    _run(sim, sim.ticks_per_round - 5)
    z = _zombie(sim, x=9.0, y=4.0)
    z.current_hp = 10_000
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 30)
    assert sim.round_index == 1
    assert u.plan.step_at(sim.tick) is not None
    assert u.plan.steps[0].fired_ticks > 0


# ---------------------------------------------------------------------------
# Shooting (§5: aim tracks the target during execution)
# ---------------------------------------------------------------------------
def test_a_shoot_order_puts_rounds_into_its_target():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    z = _zombie(sim, x=9.0, y=4.0)
    hp0 = z.current_hp
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 24)
    assert z.current_hp < hp0


def test_aim_tracks_a_moving_target():
    """The order names a UNIT, not a tile — so a target that walks away is
    still shot at from its new position (§5)."""
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    z = _zombie(sim, x=9.0, y=4.0)
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 6)
    z.x, z.y = 9.0, 12.0                      # teleport the target
    hp0 = z.current_hp
    _run(sim, 24)
    assert z.current_hp < hp0, "the shooter kept firing at the old tile"


def test_shooting_respects_the_weapon_cadence():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    z = _zombie(sim, x=9.0, y=4.0)
    z.current_hp = 10_000                     # survive the burst
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    w = sim.weapons_tables.weapons.by_name[u.weapon_id]
    _run(sim, 24)
    step = u.plan.steps[0]
    assert step.fired_ticks <= 24 // max(1, w.rof_interval_ticks) + 1


def test_walls_block_the_shoot_order():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    z = _zombie(sim, x=12.0, y=4.0)
    sim.gmap.material[:, 8] = 1
    sim.gmap.solid[:, 8] = True
    hp0 = z.current_hp
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 24)
    assert z.current_hp == hp0


# ---------------------------------------------------------------------------
# Interrupt semantics (§13)
# ---------------------------------------------------------------------------
def test_new_orders_replace_the_remaining_queue_next_round():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 20))
    sim.apply_action(u.id, _move(10, 20))
    assert len(u.orders) == 2
    _run(sim, sim.ticks_per_round)            # cross into the next round
    sim.apply_action(u.id, _move(4, 6))
    assert len(u.orders) == 1, "the first order of a round replaces the queue"


def test_further_orders_in_the_same_round_append():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 8))
    sim.apply_action(u.id, _move(8, 8))
    sim.apply_action(u.id, _move(8, 4))
    assert len(u.orders) == 3


def test_a_channeled_action_in_progress_survives_the_replacement():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    u.has_explosive = 1
    sim.apply_action(u.id, O.Order(O.ORDER_EXPLOSIVE, 6, 4, 0,
                                   action_name="plant_charge"))
    _run(sim, 4)                              # mid-channel
    _run(sim, sim.ticks_per_round - 4)        # into the next round
    # ... but the channel is still running: force the situation directly.
    sim2 = _sim()
    v = _marine(sim2, x=4.0, y=4.0)
    v.has_explosive = 1
    sim2.apply_action(v.id, O.Order(O.ORDER_EXPLOSIVE, 6, 4, 0,
                                    action_name="plant_charge"))
    v.plan_round = -1                         # pretend a new round began
    kept = list(v.orders)
    sim2.apply_action(v.id, _move(4, 8))
    assert kept[0] in v.orders, "a channeled action was interrupted"
    assert len(v.orders) == 2


def test_undo_pops_and_recompiles_and_refunds_the_item():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    grenades = u.has_grenade = 2
    sim.apply_action(u.id, O.Order(O.ORDER_GRENADE, 8, 4, 0,
                                   action_name="use_hand_grenade",
                                   grenade_fuse=2.0))
    assert u.has_grenade == grenades - 1
    assert len(u.plan.steps) == 1
    assert sim.undo_last_order(u.id) is True
    assert u.has_grenade == grenades
    assert u.plan.steps == []


def test_an_order_without_the_item_is_refused():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    u.has_grenade = 0
    assert sim.apply_action(u.id, O.Order(
        O.ORDER_GRENADE, 8, 4, 0, action_name="use_hand_grenade",
        grenade_fuse=2.0)) is False


def test_a_move_onto_a_wall_is_refused():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    assert sim.apply_action(u.id, _move(0, 0)) is False


# ---------------------------------------------------------------------------
# Retirement — a recompile can never re-run a finished action
# ---------------------------------------------------------------------------
def test_finished_steps_retire_their_orders():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 6))
    n = len(u.plan.steps[0].path)
    _run(sim, n + 2)
    assert u.orders == [], "a completed order stayed in the queue"
    assert u.plan.steps[0].retired is True


def test_recompiling_mid_round_does_not_replay_a_finished_step():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 6))
    n = len(u.plan.steps[0].path)
    _run(sim, n + 1)
    y_after = u.y
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 10, 6, 0))
    assert all(s.action.name != "move" for s in u.plan.steps)
    _run(sim, 10)
    assert u.y == pytest.approx(y_after)


# ---------------------------------------------------------------------------
# Weapon swap (§15) and overwatch state (§9) written by the plan
# ---------------------------------------------------------------------------
def test_swapping_flips_the_active_slot_and_resets_coupled_state():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    u.loadout = ["k5_carbine", "gl6_revolver"]
    u.active_slot = 0
    u.weapon_id = "k5_carbine"
    u.current_mag = 1
    sim.apply_action(u.id, O.Order(O.ORDER_SWAP, 0, 0, 0))
    _run(sim, 2)
    assert u.active_slot == 1
    assert u.weapon_id == "gl6_revolver"
    assert u.current_mag is None, "a swapped weapon arrives with a fresh mag"
    assert u.swap_cd_until_tick > sim.tick


def test_overwatch_order_writes_persistent_state():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 20, 5, 0,
                                   cone_half_deg=30.0))
    _run(sim, 2)
    assert u.overwatch_half_deg == 30.0
    assert u.overwatch_facing is not None
    _run(sim, sim.ticks_per_round + 5)
    assert u.overwatch_half_deg == 30.0, "overwatch must survive the seam"


def test_overwatch_cone_is_clamped_to_the_dialled_bounds():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 20, 5, 0,
                                   cone_half_deg=999.0))
    _run(sim, 2)
    assert u.overwatch_half_deg == CFG.onephase.overwatch_cone_max_half_deg


def test_mark_writes_the_team_table():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    z = _zombie(sim, x=9.0, y=4.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MARK, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 2)
    assert z.id in sim.marks.get(0, set())


# ---------------------------------------------------------------------------
# The UI's queries (§16)
# ---------------------------------------------------------------------------
def test_arrival_time_is_reported_in_seconds_into_the_round():
    sim = _sim()
    u = _marine(sim, x=4.0, y=4.0)
    sim.apply_action(u.id, _move(4, 9))
    step = u.plan.steps[0]
    secs = u.plan.seconds_into_round(step.end_tick, sim.round_start_tick())
    assert secs == pytest.approx(len(step.path) / TPS)
    assert 0 < secs < 10


# ---------------------------------------------------------------------------
# The legacy path is untouched
# ---------------------------------------------------------------------------
def test_two_phase_wego_does_not_use_the_timeline():
    from simulation.ruleset import TwoPhaseWEGO
    sim = Simulation(_level(), seed=7, breach_physics=None,
                     enable_recorder=False, ruleset=TwoPhaseWEGO())
    sim.set_paused(False)
    u = _marine(sim, x=4.0, y=4.0)
    assert sim.ruleset.drives_units is False
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE_ATTACK, 4, 9, 0))
    _run(sim, 20)
    assert u.plan is None, "the legacy path compiled a timeline"
    assert len(u.move_path) > 0, "the legacy path stopped precomputing"
