"""Inventory — runtime item container stub, and InventoryProfile on SpeciesDef.

Real item system is a future task. The stub exists so every Unit carries
an ``inventory`` field with the correct interface. Actual ammo / weapon
booleans (has_grenade, has_explosive) stay on Unit until they migrate here.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Placeholder until the item system is built.
ItemId = int


@dataclass
class Inventory:
    """Stub inventory. Real item list integration is a future task.

    The field exists on every Unit so consumers can call
    ``unit.inventory.current_load()`` without an AttributeError, and
    so the carry-capacity / encumbrance system has a stable attachment
    point when it is built.
    """
    equipped: list = field(default_factory=list)
    carried:  list = field(default_factory=list)

    def current_load(self) -> float:
        """Sum of all item masses. Returns 0 until items are wired in."""
        return 0.0


@dataclass(frozen=True)
class InventoryProfile:
    """Per-species carry rules, on SpeciesDef. Data only (spec §9)."""
    has_inventory:       bool  = True
    carry_capacity_base: float = 30.0   # kg; robots override with higher values


__all__ = ["ItemId", "Inventory", "InventoryProfile"]
