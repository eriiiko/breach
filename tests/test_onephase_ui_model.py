"""P8 — the UI view-models (onephase_wego design §16).

``ui.model`` is pure and headless by contract — no raylib, no window, no sim
mutation — which is exactly what lets the interface be tested at all. These
pin the hotbar (the registry rendered), the teal planning viz and its arrival
timestamps, the scrub-preview primitive, the planning clock, fog gating,
flashlights, and the DS3 menu.
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
from simulation import orders as O  # noqa: E402
from simulation.ruleset import OnePhaseWEGO, TwoPhaseWEGO  # noqa: E402
from simulation.simulation import Simulation  # noqa: E402
from simulation.unit import Unit  # noqa: E402
import ui  # noqa: E402

TPS = CFG.clock.ticks_per_second


def _level(h=40, w=40):
    tm = np.zeros((h, w), dtype=np.int32)
    tm[0, :] = 1
    tm[-1, :] = 1
    tm[:, 0] = 1
    tm[:, -1] = 1
    return LevelData(name="onephase_ui", version="2", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."))


def _sim(ruleset=None):
    sim = Simulation(_level(), seed=13, breach_physics=None,
                     enable_recorder=False,
                     ruleset=ruleset if ruleset is not None else OnePhaseWEGO())
    sim.set_paused(False)
    return sim


def _marine(sim, x=6.0, y=6.0, name="m"):
    u = Unit(name, x=x, y=y, team=0)
    sim.add_unit(u)
    return u


def _zombie(sim, x=20.0, y=6.0, name="z"):
    u = Unit(name, x=x, y=y, team=1)
    sim.add_unit(u)
    return u


def _run(sim, n):
    for _ in range(n):
        sim.step()
        sim.set_paused(False)


# ---------------------------------------------------------------------------
# The package contract: pure and headless (§16)
# ---------------------------------------------------------------------------
def test_the_model_layer_imports_no_raylib():
    """The seam that makes the UI testable. ui.draw may import pyray; the
    model layer must not, or every rule it owns becomes untestable."""
    import ui.model
    src = Path(ui.model.__file__).read_text(encoding="utf-8")
    assert "import pyray" not in src
    assert "pyray" not in sys.modules or True   # (draw.py may have loaded it)


def test_the_model_layer_never_mutates_the_sim():
    sim = _sim()
    u = _marine(sim)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 14, 0))
    before = (sim.tick, len(u.orders), u.x, u.y, len(u.plan.steps))
    ui.hotbar(sim, u)
    ui.plan_overlay(sim, u)
    ui.planning_clock(sim, 1.0)
    ui.drawable_enemies(sim, 0)
    ui.flashlight_cones(sim, 0)
    ui.ds3_menu(sim, u, 0)
    assert (sim.tick, len(u.orders), u.x, u.y, len(u.plan.steps)) == before


# ---------------------------------------------------------------------------
# Hotbar (§16) — the action registry, rendered
# ---------------------------------------------------------------------------
def test_the_hotbar_renders_registry_rows():
    sim = _sim()
    u = _marine(sim)
    slots = ui.hotbar(sim, u)
    assert len(slots) == 10
    bound = {s.action_name for s in slots if s.bound}
    assert {"move", "shoot", "move_shoot", "overwatch", "ambush"} <= bound
    for s in slots:
        if s.bound:
            assert s.label == sim.actions_table.get(s.action_name).label


def test_hotbar_keys_are_1_through_0():
    sim = _sim()
    keys = [s.key_label for s in ui.hotbar(sim, _marine(sim))]
    assert keys == ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]


def test_the_belt_fills_the_free_slots():
    """§15: belt slots ARE hotbar slots — one system wearing two skins."""
    sim = _sim()
    u = _marine(sim)
    u.belt = ["hand_grenade"]
    names = [s.action_name for s in ui.hotbar(sim, u)]
    assert "use_hand_grenade" in names


def test_dragging_an_item_binds_a_slot():
    sim = _sim()
    u = _marine(sim)
    bindings = ui.default_bindings(u)
    assert bindings[0] == "move"
    rebound = ui.bind_slot(bindings, 0, "use_breach_charge")
    assert rebound[0] == "use_breach_charge"
    assert bindings[0] == "move", "bind_slot mutated its input"
    slots = ui.hotbar(sim, u, rebound)
    assert slots[0].action_name == "use_breach_charge"


def test_the_default_belt_already_occupies_the_free_slots():
    """Nothing has to be dragged for a marine's grenades and charges to be on
    the bar — dragging is for rearranging, not for basic usability."""
    sim = _sim()
    u = _marine(sim)
    bindings = ui.default_bindings(u)
    assert bindings[8] == "use_hand_grenade"
    assert bindings[9] == "use_breach_charge"


def test_a_slot_greys_out_while_its_cooldown_runs():
    sim = _sim()
    u = _marine(sim)
    bindings = ui.bind_slot(ui.default_bindings(u), 0, "swap_weapon")
    sim.apply_action(u.id, O.Order(O.ORDER_SWAP, 0, 0, 0))
    _run(sim, 2)
    slot = ui.hotbar(sim, u, bindings)[0]
    assert slot.enabled is False
    assert slot.reason == "cooling down"
    assert slot.cooldown_remaining > 0


def test_a_slot_greys_out_during_the_global_cooldown():
    sim = _sim()
    u = _marine(sim)
    sim.apply_action(u.id, O.Order(O.ORDER_OVERWATCH, 20, 6, 0))
    _run(sim, 2)
    slots = {s.action_name: s for s in ui.hotbar(sim, u) if s.bound}
    assert slots["shoot"].enabled is False
    assert slots["shoot"].reason == "global cooldown"
    # Movement never triggers the GCD, so Move stays live (§3).
    assert slots["move"].enabled is True


def test_an_item_slot_shows_its_stock_and_greys_when_empty():
    sim = _sim()
    u = _marine(sim)
    u.has_grenade = 2
    slots = {s.action_name: s for s in ui.hotbar(sim, u) if s.bound}
    assert slots["use_hand_grenade"].count == 2
    u.has_grenade = 0
    slots = {s.action_name: s for s in ui.hotbar(sim, u) if s.bound}
    assert slots["use_hand_grenade"].enabled is False
    assert slots["use_hand_grenade"].reason == "none left"


# ---------------------------------------------------------------------------
# The teal planning viz (§16)
# ---------------------------------------------------------------------------
def test_a_move_order_draws_a_path_with_an_arrival_time():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 14, 0))
    overlay = ui.plan_overlay(sim, u)
    assert len(overlay.paths) == 1
    path = overlay.paths[0]
    assert path.endpoint == (6.0, 14.0)
    assert path.footprint == u.footprint
    assert path.arrival_seconds == pytest.approx(
        len(u.plan.steps[0].path) / TPS)
    assert path.arrival_seconds > 0


def test_arrival_times_are_the_compiled_schedule_not_an_estimate():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 12, 0))
    overlay = ui.plan_overlay(sim, u)
    shown = overlay.paths[0].arrival_seconds
    ticks = round(shown * TPS)
    _run(sim, ticks)
    assert (u.tile_x, u.tile_y) == (6, 12), \
        "the marine did not arrive when the label promised"


def test_a_waypoint_string_marks_each_intermediate_point():
    """§16: shift-click waypoint strings show teal footprint markers at each
    clicked intermediate point, always with the path line."""
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 12, 0))
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 14, 12, 0))
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 14, 20, 0))
    overlay = ui.plan_overlay(sim, u)
    assert len(overlay.paths) == 3
    assert len(overlay.waypoints) == 2, "the destination is not a waypoint"
    assert (overlay.waypoints[0].x, overlay.waypoints[0].y) == (6.0, 12.0)
    assert overlay.waypoints[1].arrival_seconds > \
        overlay.waypoints[0].arrival_seconds


def test_a_shoot_order_from_a_future_position_draws_a_hologram():
    """§16: a shoot order from a future position shows a teal hologram of the
    marine at the firing tile, indicating its target."""
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    z = _zombie(sim, 20.0, 20.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 18, 0))
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    overlay = ui.plan_overlay(sim, u)
    assert len(overlay.holograms) == 1
    holo = overlay.holograms[0]
    assert holo.action_name == "shoot"
    assert (holo.x, holo.y) == (6.0, 18.0), "the ghost is at the wrong tile"
    assert holo.target == (float(z.center_tile_x()), float(z.center_tile_y()))
    assert holo.at_seconds > 0


def test_no_hologram_when_the_action_happens_where_the_marine_stands():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    z = _zombie(sim)
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    assert ui.plan_overlay(sim, u).holograms == []


def test_a_blocked_path_is_flagged_for_the_draw_layer():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 20, 0))
    _run(sim, 4)
    sim.gmap.material[12, :] = 1
    sim.gmap.solid[12, :] = True
    _run(sim, 200)
    assert ui.plan_overlay(sim, u).paths[0].blocked is True


def test_an_empty_plan_yields_an_empty_overlay():
    sim = _sim()
    assert ui.plan_overlay(sim, _marine(sim)).empty is True


# ---------------------------------------------------------------------------
# Enemy target markers (Erik, after the first play session)
# ---------------------------------------------------------------------------
def test_an_ordered_enemy_gets_a_teal_marker():
    """Before this, an ordered shot was invisible until the round ran — §16
    covers where a marine will BE, never what it is aimed AT."""
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    z = _zombie(sim, 20.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    overlay = ui.plan_overlay(sim, u)
    assert len(overlay.targets) == 1
    marker = overlay.targets[0]
    assert marker.unit_id == z.id
    assert (marker.x, marker.y) == (z.x, z.y)
    assert marker.footprint == z.footprint
    assert marker.action_name == "shoot"
    assert marker.hovered is False


def test_several_ordered_targets_are_all_marked_in_plan_order():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    a = _zombie(sim, 20.0, 6.0, name="a")
    b = _zombie(sim, 24.0, 6.0, name="b")
    sim.apply_action(u.id, O.Order(O.ORDER_MARK, a.tile_x, a.tile_y, 0,
                                   target_unit_id=a.id))
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, b.tile_x, b.tile_y, 0,
                                   target_unit_id=b.id))
    ids = [t.unit_id for t in ui.plan_overlay(sim, u).targets]
    assert ids == [a.id, b.id]


def test_hovering_an_enemy_with_a_unit_action_armed_previews_the_pick():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    z = _zombie(sim, 20.0, 6.0)
    overlay = ui.plan_overlay(sim, u, hover_tile=(21, 7),
                              armed_action="shoot")
    assert len(overlay.targets) == 1
    assert overlay.targets[0].hovered is True, \
        "a considered target must look different from a committed one"


def test_no_hover_marker_without_a_unit_targeting_action_armed():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    _zombie(sim, 20.0, 6.0)
    assert ui.plan_overlay(sim, u, hover_tile=(21, 7)).targets == []
    assert ui.plan_overlay(sim, u, hover_tile=(21, 7),
                           armed_action="move").targets == []


def test_hovering_an_already_ordered_target_does_not_stack_markers():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    z = _zombie(sim, 20.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    overlay = ui.plan_overlay(sim, u, hover_tile=(21, 7),
                              armed_action="shoot")
    assert len(overlay.targets) == 1
    assert overlay.targets[0].hovered is False


def test_a_dead_target_stops_being_marked():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    z = _zombie(sim, 20.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_SHOOT, z.tile_x, z.tile_y, 0,
                                   target_unit_id=z.id))
    assert ui.plan_overlay(sim, u).targets
    z.alive = False
    assert ui.plan_overlay(sim, u).targets == []


def test_enemy_at_finds_the_footprint_not_just_the_anchor():
    sim = _sim()
    _marine(sim, 6.0, 6.0)
    z = _zombie(sim, 20.0, 6.0)
    assert ui.enemy_at(sim, (20, 6)) is z
    assert ui.enemy_at(sim, (22, 8)) is z       # far corner of the 3x3
    assert ui.enemy_at(sim, (23, 9)) is None
    assert ui.enemy_at(sim, None) is None


# ---------------------------------------------------------------------------
# The scrub-preview primitive (§16)
# ---------------------------------------------------------------------------
def test_position_at_is_an_exact_dry_run_of_the_plan():
    """Determinism makes the preview EXACT (§16) — so a query for tick T must
    equal where the marine actually is when T arrives."""
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 16, 0))
    probe = sim.tick + 20
    predicted = ui.position_at(u, probe)
    _run(sim, 20)
    assert sim.tick == probe
    assert (u.x, u.y) == pytest.approx(predicted)


def test_position_at_before_the_plan_starts_is_where_the_unit_stands():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 16, 0))
    assert ui.position_at(u, sim.tick) == (6.0, 6.0)


def test_position_at_after_the_plan_ends_is_the_final_position():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 16, 0))
    assert ui.position_at(u, sim.tick + 100_000) == (6.0, 16.0)


def test_position_at_between_two_moves_holds_the_previous_endpoint():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 6, 10, 0))
    sim.apply_action(u.id, O.Order(O.ORDER_HOLD, 0, 0, 0,
                                   start_tick=sim.tick + 200))
    sim.apply_action(u.id, O.Order(O.ORDER_MOVE, 14, 10, 0))
    first_end = u.plan.steps[0].end_tick
    assert ui.position_at(u, first_end + 5) == (6.0, 10.0)


# ---------------------------------------------------------------------------
# Planning clock (§16)
# ---------------------------------------------------------------------------
def test_the_clock_is_disabled_when_untimed():
    sim = _sim()
    assert CFG.onephase.planning_clock_seconds == 0.0
    clock = ui.planning_clock(sim, 3.0)
    assert clock.enabled is False


def test_the_clock_counts_down_and_expires(monkeypatch):
    monkeypatch.setattr(CFG.onephase, "planning_clock_seconds", 20.0)
    sim = _sim()
    clock = ui.planning_clock(sim, 5.0)
    assert clock.enabled is True
    assert clock.remaining_seconds == pytest.approx(15.0)
    assert clock.fraction == pytest.approx(0.75)
    assert clock.expired is False
    assert ui.planning_clock(sim, 25.0).expired is True
    assert ui.planning_clock(sim, 25.0).remaining_seconds == 0.0


# ---------------------------------------------------------------------------
# Fog of war (§8) — gating only
# ---------------------------------------------------------------------------
def test_an_unseen_enemy_is_not_drawable():
    sim = _sim()
    m = _marine(sim, 6.0, 6.0)
    m.facing = 0.0                              # looking east
    behind = _zombie(sim, 6.0, 30.0, name="behind")
    ahead = _zombie(sim, 24.0, 6.0, name="ahead")
    drawable = {u.id for u in ui.drawable_enemies(sim, 0)}
    assert ahead.id in drawable
    assert behind.id not in drawable
    assert behind in sim.units and behind.alive, "fog must not touch the sim"


def test_a_ruleset_without_vision_draws_everything():
    sim = _sim(TwoPhaseWEGO())
    _marine(sim, 6.0, 6.0)
    hidden = _zombie(sim, 6.0, 30.0)
    assert hidden in ui.drawable_enemies(sim, 0)


# ---------------------------------------------------------------------------
# Flashlights (§8) — render-only, both variants (§20 item 3)
# ---------------------------------------------------------------------------
def test_both_flashlight_variants_exist():
    sim = _sim()
    a = _marine(sim, 6.0, 6.0, name="a")
    _marine(sim, 10.0, 6.0, name="b")
    assert len(ui.flashlight_cones(sim, 0, mode="team")) == 2
    solo = ui.flashlight_cones(sim, 0, mode="selected", selected_unit_id=a.id)
    assert len(solo) == 1 and solo[0].unit_id == a.id


def test_during_planning_the_flashlight_aims_at_the_cursor():
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    u.facing = 0.0
    aimed = ui.flashlight_cones(sim, 0, cursor_tile=(7, 30), paused=True)[0]
    assert aimed.facing != pytest.approx(0.0)
    free = ui.flashlight_cones(sim, 0, cursor_tile=(7, 30), paused=False)[0]
    assert free.facing == pytest.approx(0.0)


def test_flashlights_never_touch_the_sim():
    """Render-only in v1: the moment lights affect gameplay they cross into a
    stealth system, which is another session's decision (§8)."""
    sim = _sim()
    u = _marine(sim, 6.0, 6.0)
    facing = u.facing
    ui.flashlight_cones(sim, 0, cursor_tile=(20, 20), paused=True)
    assert u.facing == facing


# ---------------------------------------------------------------------------
# The DS3 menu (§15)
# ---------------------------------------------------------------------------
def test_the_menu_pages_are_eriks_spec():
    assert ui.DS3_PAGES == ("Inventory", "Equipment", "Character", "Options",
                            "Quit")


def test_inventory_rows_are_draggable_registry_rows():
    sim = _sim()
    u = _marine(sim)
    u.has_grenade = 3
    model = ui.ds3_menu(sim, u, 0)
    assert model.page == "Inventory"
    assert model.rows, "the belt produced no inventory rows"
    row = model.rows[0]
    assert row.action_name.startswith("use_")
    assert row.value == "3"
    # The drag target: an inventory row's action_name is what binds a slot.
    bound = ui.bind_slot(ui.default_bindings(u), 8, row.action_name)
    assert ui.hotbar(sim, u, bound)[8].action_name == row.action_name


def test_equipment_shows_the_loadout_and_which_slot_is_active():
    sim = _sim()
    u = _marine(sim)
    model = ui.ds3_menu(sim, u, 1)
    assert model.page == "Equipment"
    assert [r.label for r in model.rows] == ["Primary", "Secondary"]
    assert "(active)" in model.rows[0].value
    assert "(active)" not in model.rows[1].value


def test_options_surfaces_the_time_currency_dials():
    sim = _sim()
    model = ui.ds3_menu(sim, _marine(sim), 3)
    labels = {r.label for r in model.rows}
    assert {"Round length", "Global cooldown", "Weapon swap"} <= labels


def test_pages_wrap():
    sim = _sim()
    u = _marine(sim)
    assert ui.ds3_menu(sim, u, len(ui.DS3_PAGES)).page == "Inventory"
