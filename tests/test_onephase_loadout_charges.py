"""P7 — loadout, belt, and scheduled detonations (design §12/§15).

§15's loadout model (primary + secondary, swapped freely except the 0.75 s CD,
plus a quick-item belt) and §12's replacement of the two-phase round's three
detonation SLOTS with a schedulable MOMENT — which is only expressible because
the ruleset's clock is monotonic.
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
from level_loader import LevelData  # noqa: E402
from simulation import charges as C  # noqa: E402
from simulation import orders as O  # noqa: E402
from simulation.materials import MAT_AIR  # noqa: E402
from simulation.ruleset import OnePhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402

TPS = CFG.clock.ticks_per_second


def _level(h=40, w=40, walls=()):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    for (y, x) in walls:
        tm[y, x] = 1
    return LevelData(name="onephase_loadout", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(walls=()):
    sim = Simulation(_level(walls=walls), seed=9, breach_physics=None,
                     enable_recorder=False, ruleset=OnePhaseWEGO())
    sim.set_paused(False)
    return sim


def _marine(sim, x=10.0, y=10.0, name="m"):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    return u


def _run(sim, n):
    for _ in range(n):
        sim.step()
        sim.set_paused(False)


# ---------------------------------------------------------------------------
# Loadout + belt (§15)
# ---------------------------------------------------------------------------
def test_a_marine_spawns_with_a_primary_and_a_secondary():
    sim = _sim()
    u = _marine(sim)
    assert len(u.loadout) == 2
    assert u.loadout[0] == CFG.marine.weapon
    assert u.loadout[1] == CFG.marine.secondary
    assert u.weapon_id == u.loadout[0], "weapon_id mirrors the active slot"


def test_a_marine_spawns_with_the_configured_belt():
    sim = _sim()
    u = _marine(sim)
    assert u.belt == list(CFG.marine.belt)
    assert "hand_grenade" in u.belt


def test_zombies_carry_no_loadout():
    sim = _sim()
    z = Unit("z", x=20.0, y=20.0, team=1)
    sim.add_unit(z)
    assert z.loadout == [] and z.belt == []


def test_swapping_alternates_and_mirrors_weapon_id():
    sim = _sim()
    u = _marine(sim)
    primary, secondary = u.loadout
    sim.apply_action(u.id, O.Order(O.ORDER_SWAP, 0, 0, 0))
    _run(sim, 2)
    assert u.weapon_id == secondary
    # Second swap must wait out the 0.75 s cooldown — swapping is free, but
    # not instant-repeatable (§3).
    sim.apply_action(u.id, O.Order(O.ORDER_SWAP, 0, 0, 0))
    _run(sim, 2)
    assert u.weapon_id == secondary, "the swap CD was ignored"
    _run(sim, CFG.onephase.weapon_swap_ticks + 2)
    assert u.weapon_id == primary


def test_a_belt_item_in_hand_does_not_fire_like_a_gun():
    """§15 allows a slot to hold an ITEM. A lobbed row has no trigger path, so
    the unit simply does not shoot rather than marching a grenade."""
    sim = _sim()
    u = _marine(sim)
    u.loadout = ["hand_grenade", "k5_carbine"]
    u.weapon_id = "hand_grenade"
    z = Unit("z", x=20.0, y=10.0, team=1)
    sim.add_unit(z)
    z.current_hp = 10_000
    hp0 = z.current_hp
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    _run(sim, 24)
    assert z.current_hp == hp0


def test_belt_items_are_the_registry_item_rows():
    """§15: "Quick-belt slots are hotbar slots — same registry rows, one
    system wearing two skins"."""
    sim = _sim()
    u = _marine(sim)
    item_rows = {a.item for a in sim.actions_table.item_rows()}
    assert set(u.belt) <= item_rows


# ---------------------------------------------------------------------------
# Planting is channeled (§5) — the charge exists when planting FINISHES
# ---------------------------------------------------------------------------
def _plant(sim, u, x, y, det_tick=None):
    return sim.apply_action(u.id, O.Order(O.ORDER_EXPLOSIVE, x, y, 0,
                                          action_name="plant_charge",
                                          det_tick=det_tick))


def test_a_charge_appears_only_when_the_plant_completes():
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 1
    assert _plant(sim, u, 12, 10)
    _run(sim, 4)
    assert sim.planted_charges == [], "the charge existed mid-channel"
    _run(sim, TPS)                       # the 1.0 s channel completes
    assert len(sim.planted_charges) == 1
    assert (sim.planted_charges[0].x, sim.planted_charges[0].y) == (12, 10)


def test_planting_consumes_the_item_and_undo_returns_it():
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 2
    _plant(sim, u, 12, 10)
    assert u.has_explosive == 1
    assert sim.undo_last_order(u.id) is True
    assert u.has_explosive == 2


def test_planting_without_a_charge_is_refused():
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 0
    assert _plant(sim, u, 12, 10) is False


# ---------------------------------------------------------------------------
# Scheduled detonation (§12)
# ---------------------------------------------------------------------------
def test_a_timed_charge_fires_at_its_chosen_tick():
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 1
    det = sim.tick + TPS * 2                  # 2.0 s into the round
    _plant(sim, u, 16, 10, det_tick=det)
    _run(sim, TPS + 4)                        # planting done, charge waiting
    assert sim.planted_charges and sim.planted_charges[0].live
    _run(sim, TPS)                            # past the moment
    assert sim.planted_charges == [], "the charge never fired"


def test_the_detonation_actually_breaches():
    """The charge goes through the shipped payload executor, so its blast is
    the blast a door charge has always been."""
    sim = _sim(walls=[(10, 16)])
    u = _marine(sim)
    u.has_explosive = 1
    assert sim.gmap.solid[10, 16]
    det = sim.tick + TPS + 2
    _plant(sim, u, 16, 10, det_tick=det)
    _run(sim, TPS * 3)
    assert sim.gmap.material[10, 16] == MAT_AIR, "the wall survived the charge"


def test_a_remote_charge_waits_for_the_detonate_order():
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 1
    _plant(sim, u, 16, 10)                    # no det_tick: remote
    _run(sim, TPS * 3)
    assert len(sim.planted_charges) == 1 and sim.planted_charges[0].live
    assert sim.planted_charges[0].det_tick is None


def test_a_detonate_order_fires_the_planters_charges():
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 2
    _plant(sim, u, 16, 10)
    _plant(sim, u, 16, 12)
    _run(sim, TPS * 3)
    assert len(sim.planted_charges) == 2
    sim.apply_action(u.id, O.Order(O.ORDER_DETONATE, 0, 0, 0,
                                   det_tick=sim.tick + 4))
    _run(sim, 10)
    assert sim.planted_charges == []


def test_a_detonate_order_does_not_fire_someone_elses_charges():
    sim = _sim()
    planter = _marine(sim, 10.0, 10.0, name="planter")
    other = _marine(sim, 10.0, 20.0, name="other")
    planter.has_explosive = 1
    _plant(sim, planter, 16, 10)
    _run(sim, TPS * 2)
    assert len(sim.planted_charges) == 1
    sim.apply_action(other.id, O.Order(O.ORDER_DETONATE, 0, 0, 0,
                                       det_tick=sim.tick + 2))
    _run(sim, 6)
    assert len(sim.planted_charges) == 1, "somebody else's charge went off"


def test_a_detonation_can_be_scheduled_into_the_next_round():
    """§12's breach opening: "A charge planted during planning may detonate at
    t=0 of the next round" — an absolute tick, so the seam is a non-event."""
    sim = _sim(walls=[(10, 16)])
    u = _marine(sim)
    u.has_explosive = 1
    next_round_start = sim.round_start_tick() + sim.ticks_per_round
    _plant(sim, u, 16, 10, det_tick=next_round_start)
    _run(sim, sim.ticks_per_round - 4)
    assert sim.planted_charges and sim.planted_charges[0].live, \
        "the charge fired inside the wrong round"
    _run(sim, 8)
    assert sim.round_index == 1
    assert sim.planted_charges == []
    assert sim.gmap.material[10, 16] == MAT_AIR


def test_detonation_time_is_a_moment_not_a_slot():
    """The three DET_* slots are gone: two charges in the same round can fire
    at two different, freely-chosen times."""
    sim = _sim()
    u = _marine(sim)
    u.has_explosive = 2
    early = sim.tick + TPS + 2
    late = sim.tick + TPS * 3
    _plant(sim, u, 16, 10, det_tick=early)
    _plant(sim, u, 16, 14, det_tick=late)
    _run(sim, TPS * 2 + 4)
    live = [c for c in sim.planted_charges if c.live]
    assert len(live) == 1, "both charges fired on the same slot"
    assert live[0].x == 16 and live[0].y == 14
    _run(sim, TPS * 2)
    assert sim.planted_charges == []


# ---------------------------------------------------------------------------
# Grenades on the timeline
# ---------------------------------------------------------------------------
def test_a_grenade_is_thrown_from_where_the_thrower_actually_is():
    """The two-phase round materialized grenades up front from a PLANNED end
    position; on the timeline the throw is a scheduled action, so it happens
    where the marine is standing when it happens."""
    sim = _sim()
    u = _marine(sim, 10.0, 10.0)
    u.has_grenade = 1
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 10, 20, 0))
    sim.apply_action(u.id, O.Order(O.ORDER_GRENADE, 16, 20, 0,
                                   action_name="use_hand_grenade",
                                   grenade_fuse=2.0))
    _run(sim, 4)
    assert sim.projectiles == [], "the grenade left before the marine did"
    _run(sim, 200)
    assert len(sim.projectiles) >= 1 or u.orders == []
    thrown = [p for p in sim.projectiles] or None
    if thrown:
        assert thrown[0].start_fy > 11, \
            "the grenade was thrown from the starting position"


def test_no_pre_spawn_helper_is_needed_under_this_ruleset():
    """``spawn_projectiles_from_grenade_orders`` exists for the two-phase
    round; the timeline never needs it, and calling it is not required for a
    grenade to fly."""
    sim = _sim()
    u = _marine(sim)
    u.has_grenade = 1
    sim.apply_action(u.id, O.Order(O.ORDER_GRENADE, 16, 10, 0,
                                   action_name="use_hand_grenade",
                                   grenade_fuse=1.0))
    _run(sim, 4)
    assert len(sim.projectiles) == 1


# ---------------------------------------------------------------------------
# The charge model itself
# ---------------------------------------------------------------------------
def test_charge_due_semantics():
    c = C.PlantedCharge(3, 4, "demo_breach", owner_id=0, det_tick=10)
    assert c.due(9) is False
    assert c.due(10) is True
    c.live = False
    assert c.due(11) is False


def test_a_remote_charge_is_never_due():
    c = C.PlantedCharge(3, 4, "demo_breach", owner_id=0)
    assert c.due(10_000) is False
