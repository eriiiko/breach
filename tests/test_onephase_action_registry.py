"""P2 — the action registry (onephase_wego design §5).

Pins the registry AS the extensibility backbone the design asks for: the
closed targeting/start-condition vocabularies, the seconds -> ticks
derivation (time is the only currency, §3), the GCD rules, channeled actions,
item-generated rows, class gating, and the config-override discipline.

Also pins the two properties the rest of the arc leans on: adding a verb is a
row (not a schema change), and nothing here disturbs the legacy order
vocabulary TwoPhaseWEGO runs on.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for _p in (ROOT, ROOT / "src", ROOT / "cpp" / "build" / "Release"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from config import CFG  # noqa: E402
from simulation.action_registry import (  # noqa: E402
    ActionDef, ActionTable, START_CONDITIONS, TARGETING_MODES, get_table,
    rebuild_table,
)
from simulation import orders as O  # noqa: E402
from simulation.weapons import get_tables as weapon_tables  # noqa: E402

TPS = CFG.clock.ticks_per_second


def _table():
    return rebuild_table()


# ---------------------------------------------------------------------------
# The v1 verb set (§5's table)
# ---------------------------------------------------------------------------
def test_v1_verb_set_is_present():
    t = _table()
    for name in ("move", "shoot", "move_shoot", "overwatch", "ambush", "hold",
                 "swap_weapon", "mark", "plant_charge", "detonate"):
        assert name in t.by_name, f"design §5 verb {name!r} missing"


def test_every_row_declares_a_closed_targeting_and_start_condition():
    for a in _table().by_name.values():
        assert a.targeting in TARGETING_MODES
        assert a.start_condition in START_CONDITIONS


def test_rows_carry_their_order_type_and_reverse_lookup():
    t = _table()
    assert t.get("move").order_type == O.ORDER_MOVE
    assert t.get("shoot").order_type == O.ORDER_SHOOT
    assert t.get("hold").order_type == O.ORDER_HOLD
    assert t.for_order_type(O.ORDER_OVERWATCH).name == "overwatch"


# ---------------------------------------------------------------------------
# Time is the only currency (§3): seconds -> integer ticks
# ---------------------------------------------------------------------------
def test_durations_and_cooldowns_derive_to_integer_ticks():
    t = _table()
    ow = t.get("overwatch")
    assert ow.duration_seconds == 0.25
    assert ow.duration_ticks == round(0.25 * TPS) == 6


def test_weapon_swap_cooldown_comes_off_the_dial():
    """§3: 0.75 s, its own NON-global cooldown — so swapping neither triggers
    the GCD nor is gated by it."""
    swap = _table().get("swap_weapon")
    assert swap.cooldown_seconds == CFG.onephase.weapon_swap_seconds == 0.75
    assert swap.cooldown_ticks == round(0.75 * TPS) == 18
    assert swap.triggers_gcd is False


def test_zero_duration_stays_zero_not_one_tick():
    """"Instantaneous" is a distinct meaning from "at least one tick" — the
    max(1, ...) floor must not silently promote a 0."""
    t = _table()
    assert t.get("move").duration_ticks == 0
    assert t.get("mark").duration_ticks == 0
    assert t.get("shoot").cooldown_ticks == 0


def test_there_is_no_ap_column():
    """AP is dead (§3). The registry must not grow one by accident."""
    assert not any("ap" in s for s in ActionDef.__slots__)


# ---------------------------------------------------------------------------
# The GCD rules (§3)
# ---------------------------------------------------------------------------
def test_movement_never_triggers_the_gcd():
    assert _table().get("move").triggers_gcd is False


def test_actions_trigger_the_gcd():
    t = _table()
    for name in ("shoot", "move_shoot", "overwatch", "ambush", "plant_charge",
                 "detonate"):
        assert t.get(name).triggers_gcd is True, name


def test_weapons_are_salvo_exempt():
    """§3: within a weapon's own salvo, successive rounds do NOT re-trigger
    the GCD — an SMG burst is one action; the GCD gates CHANGING action."""
    t = _table()
    assert t.get("shoot").gcd_exempt_within_salvo is True
    assert t.get("move_shoot").gcd_exempt_within_salvo is True
    assert t.get("overwatch").gcd_exempt_within_salvo is False


def test_a_move_row_that_triggers_the_gcd_is_a_load_error():
    with pytest.raises(ValueError, match="never by movement"):
        ActionTable(TPS, actions_cfg={"move": {"triggers_gcd": True}})


# ---------------------------------------------------------------------------
# Channeled actions (§5)
# ---------------------------------------------------------------------------
def test_planting_a_charge_is_channeled_and_shooting_is_not():
    t = _table()
    assert t.get("plant_charge").interruptible is False
    assert t.get("shoot").interruptible is True, "mid-salvo is interruptible"
    assert t.get("move").interruptible is True


# ---------------------------------------------------------------------------
# Item-generated rows (§5) + the hotbar's data source (§16)
# ---------------------------------------------------------------------------
def test_item_rows_are_generated_from_the_lobbed_and_placed_weapon_rows():
    t = _table()
    wt = weapon_tables()
    expected = {f"use_{n}" for n, w in wt.weapons.by_name.items()
                if w.archetype in ("lobbed", "placed")}
    assert expected, "the armory has no item-archetype rows to generate from"
    assert expected <= set(t.by_name)
    for name in expected:
        assert t.get(name).item, "a generated row must name its item"


def test_generated_placed_rows_are_channeled_and_lobbed_ones_are_not():
    t = _table()
    wt = weapon_tables()
    for n, w in wt.weapons.by_name.items():
        if w.archetype == "placed":
            assert t.get(f"use_{n}").interruptible is False
        elif w.archetype == "lobbed":
            assert t.get(f"use_{n}").interruptible is True


def test_item_rows_query_returns_only_items():
    t = _table()
    assert all(a.item for a in t.item_rows())
    assert "move" not in {a.name for a in t.item_rows()}


def test_hotbar_order_is_deterministic():
    a = [x.name for x in _table().hotbar_rows()]
    b = [x.name for x in _table().hotbar_rows()]
    assert a == b
    assert a[:3] == ["move", "shoot", "move_shoot"], "declaration order first"


def test_class_gating_filters_the_hotbar():
    t = _table()
    t.by_name["breacher_only"] = ActionDef("breacher_only", O.ORDER_MOVE,
                                           classes=("breacher",))
    names = {x.name for x in t.hotbar_rows(unit_class="rifleman")}
    assert "breacher_only" not in names
    assert "breacher_only" in {x.name for x in t.hotbar_rows("breacher")}


# ---------------------------------------------------------------------------
# "Adding a verb = adding a row" (§5)
# ---------------------------------------------------------------------------
def test_a_new_verb_needs_no_schema_change():
    t = _table()
    before = len(t.names)
    t._add(ActionDef("suppress", O.ORDER_SHOOT, targeting="tile",
                     duration_seconds=0.5, triggers_gcd=True))
    assert len(t.by_name) == before + 1
    assert t.get("suppress").targeting == "tile"


def test_duplicate_row_names_are_a_load_error():
    t = _table()
    with pytest.raises(ValueError, match="duplicate row"):
        t._add(ActionDef("move", O.ORDER_MOVE))


# ---------------------------------------------------------------------------
# Config-override discipline (the entities.toml split)
# ---------------------------------------------------------------------------
def test_numeric_overrides_land_in_the_derived_ticks():
    t = ActionTable(TPS, actions_cfg={"overwatch": {"duration_seconds": 1.0}})
    assert t.get("overwatch").duration_ticks == TPS


def test_unknown_row_or_column_is_a_hard_error():
    with pytest.raises(ValueError, match="does not exist"):
        ActionTable(TPS, actions_cfg={"teleport": {"duration_seconds": 1.0}})
    with pytest.raises(ValueError, match="cannot override"):
        ActionTable(TPS, actions_cfg={"shoot": {"targeting": "tile"}})


def test_negative_or_non_numeric_override_is_a_hard_error():
    with pytest.raises(ValueError, match=">= 0"):
        ActionTable(TPS, actions_cfg={"shoot": {"duration_seconds": -1.0}})
    with pytest.raises(ValueError, match="must be a number"):
        ActionTable(TPS, actions_cfg={"shoot": {"duration_seconds": "fast"}})


def test_a_built_table_never_mutates_the_module_templates():
    """A config override in one table must not leak into the next one built
    in the same process (the bug conftest's weapon-table reset exists for)."""
    ActionTable(TPS, actions_cfg={"overwatch": {"duration_seconds": 9.0}})
    assert _table().get("overwatch").duration_seconds == 0.25


# ---------------------------------------------------------------------------
# The legacy vocabulary is untouched
# ---------------------------------------------------------------------------
def test_legacy_order_ids_and_move_tuple_are_unchanged():
    assert (O.ORDER_MOVE_ATTACK, O.ORDER_MOVE_COVER, O.ORDER_SPRINT,
            O.ORDER_GRENADE, O.ORDER_EXPLOSIVE, O.ORDER_FIRE) == (0, 1, 2, 3,
                                                                  4, 5)
    assert O.MOVE_ORDER_TYPES == (O.ORDER_MOVE_ATTACK, O.ORDER_MOVE_COVER,
                                  O.ORDER_SPRINT)
    # The new movement orders are deliberately NOT in the legacy tuple.
    assert O.ORDER_MOVE not in O.MOVE_ORDER_TYPES
    assert O.ORDER_MOVE_SHOOT not in O.MOVE_ORDER_TYPES


def test_new_order_ids_do_not_collide():
    ids = [O.ORDER_MOVE_ATTACK, O.ORDER_MOVE_COVER, O.ORDER_SPRINT,
           O.ORDER_GRENADE, O.ORDER_EXPLOSIVE, O.ORDER_FIRE, O.ORDER_OVERWATCH,
           O.ORDER_AMBUSH, O.ORDER_HOLD, O.ORDER_SWAP, O.ORDER_MOVE_SHOOT,
           O.ORDER_MARK, O.ORDER_MOVE, O.ORDER_SHOOT, O.ORDER_DETONATE]
    assert len(ids) == len(set(ids))
    assert set(ids) <= set(O.ORDER_NAMES)


def test_a_legacy_order_carries_no_onephase_payload():
    o = O.Order(O.ORDER_MOVE_ATTACK, 3, 4, phase=0)
    assert (o.action_name, o.target_unit_id, o.start_tick, o.det_tick,
            o.cone_half_deg, o.aim_anchor) == (None,) * 6


def test_module_table_is_shared_and_rebuildable():
    rebuild_table()
    assert get_table() is get_table()
