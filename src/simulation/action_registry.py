"""The ACTION REGISTRY — OnePhaseWEGO's extensibility backbone (design §5).

``docs/onephase_wego_design_2026-07-28.md`` §5: "A data-driven table, same
pattern as the material table. Each action is a row. Adding a new verb =
adding a row (+ its resolution branch)." This module is that table.

It is the single place that answers, for every verb the player can issue:

- how long does it take (``duration_ticks``) and what does it deny afterwards
  (``cooldown_ticks``, ``triggers_gcd``)?
- can a new order cut it short (``interruptible``) — channeled actions
  (operating a terminal, planting a charge, objective interactions) cannot;
- what does it point at (``targeting``) and when does it start
  (``start_condition``)?
- which unit classes may take it (``classes``), and is it an ITEM's action
  rather than a standing verb (``item``)?

The hotbar RENDERS this table (§16) — dragging an item to a hotbar slot binds
the slot to a row — so the registry is also the UI's data source, not just the
executor's. Nothing else in the engine may hard-code the verb list.

**Time is the only currency (§3).** Every duration/cooldown here is authored
in SECONDS in ``config.toml`` and derived once into integer ticks at table
build (the ``ticks_from_seconds`` convention every other table uses), so the
whole economy is tick-rate independent. There is no AP column and never will
be — the round's seconds ARE the budget.

**Determinism.** The table is config-static data built at Simulation
construction (the ``WeaponsTables`` precedent — Ctrl+R alone does not rebuild
it), holds only ints/bools/strings, and draws no RNG. Changing a number in it
is a deliberate behavioral change, exactly like a weapon row.

Dormancy: nothing outside the OnePhaseWEGO ruleset and its systems reads this
module, so its existence cannot move a shipped golden.
"""
from __future__ import annotations

from config import CFG, ticks_from_seconds
from simulation.orders import (
    ORDER_AMBUSH, ORDER_DETONATE, ORDER_EXPLOSIVE, ORDER_GRENADE, ORDER_HOLD,
    ORDER_MARK, ORDER_MOVE, ORDER_MOVE_SHOOT, ORDER_OVERWATCH, ORDER_SHOOT,
    ORDER_SWAP,
)
from simulation.weapons import get_tables as weapon_tables

# ---------------------------------------------------------------------------
# Closed vocabularies (§5). Both sets are CLOSED like the weapon archetypes:
# a row naming something outside them is a config bug, loud at load.
# ---------------------------------------------------------------------------
#: What an action points at. ``time`` is Hold-until-t's "target" — the player
#: picks a moment on the timeline rather than a place.
TARGETING_MODES = frozenset({"unit", "tile", "direction", "time", "none"})

#: When an action begins. ``signal`` is the §10 slot deliberately kept open
#: for level logic (Arc B wiring) to trigger squad actions — v1 has no
#: SignalBus start-condition, but the FIELD exists so adding one later is a
#: row edit rather than a schema change.
START_CONDITIONS = frozenset({"immediate", "at_time", "ambush_barrier",
                              "signal"})


class ActionDef:
    """One registry row (§5).

    ``order_type`` is the :mod:`simulation.orders` id whose resolution branch
    executes this row — the seam between "what the player picked" (a registry
    row, rendered on the hotbar) and "how the sim runs it" (an order type).
    Several rows may share one order type: every item-generated USE row rides
    ``ORDER_GRENADE`` / ``ORDER_EXPLOSIVE`` with a different ``item``.
    """

    __slots__ = ("name", "icon", "order_type", "duration_seconds",
                 "cooldown_seconds", "duration_ticks", "cooldown_ticks",
                 "triggers_gcd", "gcd_exempt_within_salvo", "interruptible",
                 "targeting", "start_condition", "classes", "item", "label",
                 "sustained")

    def __init__(self, name, order_type, *, icon="", duration_seconds=0.0,
                 cooldown_seconds=0.0, triggers_gcd=False,
                 gcd_exempt_within_salvo=False, interruptible=True,
                 targeting="none", start_condition="immediate", classes=(),
                 item="", label="", sustained=False):
        self.name = name
        self.order_type = int(order_type)
        self.icon = str(icon)
        self.duration_seconds = float(duration_seconds)
        self.cooldown_seconds = float(cooldown_seconds)
        self.duration_ticks = 0          # derived at table build
        self.cooldown_ticks = 0          # derived at table build
        self.triggers_gcd = bool(triggers_gcd)
        self.gcd_exempt_within_salvo = bool(gcd_exempt_within_salvo)
        self.interruptible = bool(interruptible)
        self.targeting = str(targeting)
        self.start_condition = str(start_condition)
        self.classes = tuple(classes)
        self.item = str(item)
        self.label = str(label or name.replace("_", " ").title())
        # SUSTAINED actions occupy the timeline until the NEXT step starts
        # (or indefinitely, if they are the plan's tail) rather than for a
        # fixed duration: "stand and shoot at that unit" is a standing order,
        # not a 0.3 s animation. A sustained tail is precisely what §9 means by
        # "Overwatch is a state — persists across rounds until replaced", and
        # what makes a shoot order keep shooting after the round seam.
        self.sustained = bool(sustained)

    def allows_class(self, unit_class: str) -> bool:
        """Class gating (§5): an empty ``classes`` tuple means every unit may
        take the action; otherwise the unit's class must be named."""
        return not self.classes or unit_class in self.classes

    def __repr__(self):
        return (f"ActionDef({self.name!r}, dur={self.duration_ticks}t, "
                f"cd={self.cooldown_ticks}t, gcd={self.triggers_gcd}, "
                f"targeting={self.targeting!r})")


# ---------------------------------------------------------------------------
# The v1 verb set (§5's table, verbatim). Authored in code rather than
# config.toml because the ROWS are vocabulary (like the weapon archetypes) —
# what belongs in config is the NUMBERS, and `[actions.*]` overrides them
# below, exactly the entities.toml split ("schema in code, tuning in TOML").
# ---------------------------------------------------------------------------
_V1_ROWS = (
    # Move is THE primary order (§5): full speed — the old Sprint is folded in
    # — no auto-attack, and NO GCD, because the GCD is never triggered by
    # movement (§3). Shift-click waypoint strings ride one order each.
    ActionDef("move", ORDER_MOVE, icon="move", targeting="tile",
              triggers_gcd=False, label="Move"),
    # Shoot: the secondary-most-important order; default is stand and shoot,
    # aim tracks the target through execution (§5).
    ActionDef("shoot", ORDER_SHOOT, icon="shoot", targeting="unit",
              triggers_gcd=True, gcd_exempt_within_salvo=True, sustained=True,
              label="Shoot"),
    # Move & Shoot exists, at reduced accuracy (§7) and reduced speed (§6).
    # Not sustained: its length is its PATH's length.
    ActionDef("move_shoot", ORDER_MOVE_SHOOT, icon="move_shoot",
              targeting="unit", triggers_gcd=True,
              gcd_exempt_within_salvo=True, label="Move & Shoot"),
    # Overwatch is a STATE that persists across rounds (§9); the order that
    # establishes it is a short action pointing a cone in a direction.
    ActionDef("overwatch", ORDER_OVERWATCH, icon="overwatch",
              duration_seconds=0.25, targeting="direction", triggers_gcd=True,
              label="Overwatch"),
    # Ambush waits on the group barrier (§10) — its start condition is the
    # readiness counter, not a clock.
    ActionDef("ambush", ORDER_AMBUSH, icon="ambush", targeting="unit",
              start_condition="ambush_barrier", triggers_gcd=True,
              sustained=True, label="Ambush"),
    # Hold (until t): the sequencing verb that makes choreography composable
    # (§5). Its "target" is a moment.
    ActionDef("hold", ORDER_HOLD, icon="hold", targeting="time",
              start_condition="at_time", triggers_gcd=False, label="Hold"),
    # Weapon swap: free except its OWN non-global cooldown (§3) — so it does
    # not trigger the GCD, and the GCD does not gate it.
    ActionDef("swap_weapon", ORDER_SWAP, icon="swap",
              cooldown_seconds=CFG.onephase.weapon_swap_seconds,
              targeting="none", triggers_gcd=False, label="Swap Weapon"),
    # Marking is a command, not a physical act (§11): no duration, no GCD —
    # "the real cost is UI".
    ActionDef("mark", ORDER_MARK, icon="mark", targeting="unit",
              triggers_gcd=False, label="Mark Target"),
    # Planting a charge is CHANNELED (§5): interruptible = False.
    ActionDef("plant_charge", ORDER_EXPLOSIVE, icon="charge",
              duration_seconds=1.0, targeting="tile", triggers_gcd=True,
              interruptible=False, item="breach_charge", label="Plant Charge"),
    # Detonation time is schedulable anywhere in the round (§12).
    ActionDef("detonate", ORDER_DETONATE, icon="detonate", targeting="time",
              start_condition="at_time", triggers_gcd=True, label="Detonate"),
)

#: Weapon archetypes whose rows become USE-item actions (§5: "registry rows
#: generated from usable inventory items"). LOBBED = thrown (grenades),
#: PLACED = planted (charges) — exactly the two that ride an order flow rather
#: than a trigger, which is why they are items on the hotbar and not weapons.
ITEM_ARCHETYPES = ("lobbed", "placed")

#: The order type each item archetype's generated row executes through.
_ITEM_ORDER_TYPE = {"lobbed": ORDER_GRENADE, "placed": ORDER_EXPLOSIVE}


class ActionTable:
    """The built registry: rows keyed by name, in a stable order.

    Order of ``self.names`` is the hotbar's default left-to-right order and is
    deterministic: the v1 verbs in declaration order, then item-generated rows
    in weapon-table (config) order. Nothing sorts by a hash or a set.
    """

    def __init__(self, ticks_per_second, actions_cfg=None, weapons_tables=None):
        self.by_name: dict[str, ActionDef] = {}
        for row in _V1_ROWS:
            self._add(_clone(row))
        # Item-generated rows (§5). Built from the SAME weapon table the rest
        # of the sim reads, so an armory edit shows up on the hotbar with no
        # second source of truth.
        tables = weapons_tables if weapons_tables is not None else weapon_tables()
        for name, w in tables.weapons.by_name.items():
            if w.archetype not in ITEM_ARCHETYPES:
                continue
            self._add(ActionDef(
                f"use_{name}", _ITEM_ORDER_TYPE[w.archetype], icon=name,
                targeting="tile", triggers_gcd=True,
                # A thrown item is a quick motion; a planted one is channeled
                # (§5) — the same rule the hand-authored plant_charge row
                # states, applied to every generated PLACED row.
                duration_seconds=(0.0 if w.archetype == "lobbed" else 1.0),
                interruptible=(w.archetype == "lobbed"),
                item=name, label=f"Use {name.replace('_', ' ').title()}"))

        # `[actions.*]` numeric overrides (the entities.toml split: schema in
        # code, tuning in TOML). Applied BEFORE the seconds -> ticks
        # derivation so an override lands in the derived integers too.
        cfg = actions_cfg if actions_cfg is not None else getattr(
            CFG, "actions", None)
        if cfg is not None:
            self._apply_overrides(cfg)

        for a in self.by_name.values():
            self._validate(a)
            # Seconds -> integer ticks. A ZERO duration/cooldown must stay
            # zero — it means "instantaneous" / "no cooldown", which is a
            # distinct meaning from "at least one tick", so the max(1, ...)
            # helper is applied only to positive values (the rof_interval_ticks
            # precedent in WeaponTable).
            a.duration_ticks = (ticks_from_seconds(a.duration_seconds,
                                                   ticks_per_second)
                                if a.duration_seconds > 0 else 0)
            a.cooldown_ticks = (ticks_from_seconds(a.cooldown_seconds,
                                                   ticks_per_second)
                                if a.cooldown_seconds > 0 else 0)
        self.names = list(self.by_name)

    # -- build helpers -------------------------------------------------
    def _add(self, action: ActionDef) -> None:
        if action.name in self.by_name:
            raise ValueError(
                f"action registry: duplicate row '{action.name}' — the table "
                f"is keyed by name (design §5)")
        self.by_name[action.name] = action

    def _apply_overrides(self, cfg) -> None:
        """Apply ``[actions.<row>] key = number`` overrides from config.

        Same discipline as ``entities.toml`` (registry.py): an unknown row or
        an unknown/non-numeric field is a HARD ERROR, never a silent skip — a
        typo in a tuning dial must not quietly do nothing.
        """
        rows = (cfg.items() if isinstance(cfg, dict) else vars(cfg).items())
        tunable = {"duration_seconds", "cooldown_seconds", "triggers_gcd",
                   "gcd_exempt_within_salvo", "interruptible"}
        for row_name, row in rows:
            if row_name not in self.by_name:
                raise ValueError(
                    f"[actions.{row_name}] overrides an action row that does "
                    f"not exist; rows: {sorted(self.by_name)}")
            action = self.by_name[row_name]
            fields = (row.items() if isinstance(row, dict)
                      else vars(row).items())
            for key, value in fields:
                if key not in tunable:
                    raise ValueError(
                        f"[actions.{row_name}] cannot override {key!r} — "
                        f"tunable columns are {sorted(tunable)} (the rest is "
                        f"vocabulary, and lives in code)")
                if key in ("triggers_gcd", "gcd_exempt_within_salvo",
                           "interruptible"):
                    setattr(action, key, bool(value))
                else:
                    if isinstance(value, bool) or not isinstance(
                            value, (int, float)):
                        raise ValueError(
                            f"[actions.{row_name}].{key} must be a number, "
                            f"got {value!r}")
                    if value < 0:
                        raise ValueError(
                            f"[actions.{row_name}].{key} must be >= 0, got "
                            f"{value!r}")
                    setattr(action, key, float(value))

    @staticmethod
    def _validate(a: ActionDef) -> None:
        if a.targeting not in TARGETING_MODES:
            raise ValueError(
                f"actions.{a.name}.targeting {a.targeting!r} is not one of "
                f"{sorted(TARGETING_MODES)} (design §5 — the set is closed)")
        if a.start_condition not in START_CONDITIONS:
            raise ValueError(
                f"actions.{a.name}.start_condition {a.start_condition!r} is "
                f"not one of {sorted(START_CONDITIONS)} (design §5 — the set "
                f"is closed)")
        # The GCD is never triggered by movement (§3). `move` is the only row
        # that is pure movement; move_shoot triggers it through the SHOT.
        if a.name == "move" and a.triggers_gcd:
            raise ValueError(
                "actions.move.triggers_gcd must stay false — the GCD is "
                "triggered by actions, never by movement (design §3)")

    # -- queries -------------------------------------------------------
    def get(self, name: str) -> ActionDef:
        try:
            return self.by_name[name]
        except KeyError:
            raise KeyError(
                f"unknown action {name!r}; rows: {sorted(self.by_name)}"
            ) from None

    def for_order_type(self, order_type: int) -> ActionDef:
        """First row executing through ``order_type`` — the reverse lookup the
        executor uses when an order arrives without an explicit row name (a
        bare ``Order`` from a test or a legacy call site)."""
        for a in self.by_name.values():
            if a.order_type == int(order_type):
                return a
        raise KeyError(f"no action row for order_type {order_type!r}")

    def hotbar_rows(self, unit_class: str = "") -> list:
        """The rows a hotbar should offer this unit, in table order (§16).
        Class-gated rows the unit cannot take are omitted."""
        return [a for a in self.by_name.values() if a.allows_class(unit_class)]

    def item_rows(self) -> list:
        """Only the item-generated rows — what the inventory pane lists as
        draggable onto a hotbar slot (§16)."""
        return [a for a in self.by_name.values() if a.item]


def _clone(a: ActionDef) -> ActionDef:
    """Fresh copy of a declared row, so a built table never mutates the
    module-level ``_V1_ROWS`` templates (a config override or a tick-rate
    change must not leak into the next table built in the same process — the
    bug the conftest weapon-table reset exists to prevent)."""
    return ActionDef(
        a.name, a.order_type, icon=a.icon,
        duration_seconds=a.duration_seconds,
        cooldown_seconds=a.cooldown_seconds, triggers_gcd=a.triggers_gcd,
        gcd_exempt_within_salvo=a.gcd_exempt_within_salvo,
        interruptible=a.interruptible, targeting=a.targeting,
        start_condition=a.start_condition, classes=a.classes, item=a.item,
        label=a.label, sustained=a.sustained)


# ---------------------------------------------------------------------------
# Module-level table (the weapons.get_tables/rebuild_tables precedent)
# ---------------------------------------------------------------------------
_TABLE: ActionTable | None = None


def get_table() -> ActionTable:
    """The shared action table, built on first use."""
    global _TABLE
    if _TABLE is None:
        _TABLE = ActionTable(CFG.clock.ticks_per_second)
    return _TABLE


def rebuild_table(weapons_tables=None) -> ActionTable:
    """Rebuild from the live CFG — called at every Simulation construction /
    reset, exactly as ``rebuild_weapon_tables`` is (config-static data,
    construction-bound)."""
    global _TABLE
    _TABLE = ActionTable(CFG.clock.ticks_per_second,
                         weapons_tables=weapons_tables)
    return _TABLE


__all__ = [
    "ActionDef", "ActionTable", "TARGETING_MODES", "START_CONDITIONS",
    "ITEM_ARCHETYPES", "get_table", "rebuild_table",
]
