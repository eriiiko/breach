"""Order types and constants used by units / planning / execution.

Step 2 lift: just the order type IDs (needed by :class:`simulation.unit.Unit`
for things like ``current_order_type`` and ``get_fire_order_in_phase``). The
:class:`Order` data class itself is added in step 3 of the migration.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Order type IDs (matches the legacy values in game.py:85-99 exactly)
# ---------------------------------------------------------------------------
ORDER_MOVE_ATTACK = 0
ORDER_MOVE_COVER  = 1
ORDER_SPRINT      = 2
ORDER_GRENADE     = 3
ORDER_EXPLOSIVE   = 4
ORDER_FIRE        = 5

ORDER_NAMES = {
    ORDER_MOVE_ATTACK: "Move & Attack",
    ORDER_MOVE_COVER:  "Move w/ Cover",
    ORDER_SPRINT:      "Sprint",
    ORDER_GRENADE:     "Grenade",
    ORDER_EXPLOSIVE:   "Explosive",
    ORDER_FIRE:        "Fire",
}

# Movement-type orders (the ones with a marine-speed table behind them).
MOVE_ORDER_TYPES = (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT)

# ---------------------------------------------------------------------------
# Detonation slots for door explosives (game.py:132-139)
# ---------------------------------------------------------------------------
DET_START_PHASE1   = 0
DET_BETWEEN_PHASES = 1
DET_END_PHASE2     = 2

DET_SLOT_NAMES = {
    DET_START_PHASE1:   "Start P1",
    DET_BETWEEN_PHASES: "Between P1/P2",
    DET_END_PHASE2:     "End P2",
}
