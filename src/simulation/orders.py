"""Order types and the :class:`Order` data class.

Lifted from ``game.py:Order`` (lines 1089-1098) plus the order-type and
detonation-slot constant blocks (game.py:85-99 and 132-139). Per the
locked design decisions, :class:`Order` stays a SINGLE class with an
``order_type`` discriminator + optional payload fields (``grenade_fuse``,
``det_slot``) — subclassing into ``MoveOrder`` / ``FireOrder`` / etc. is
explicitly an anti-goal for this patch.
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


# ---------------------------------------------------------------------------
# Order data class
# ---------------------------------------------------------------------------
class Order:
    """A queued action: move / fire / grenade / explosive.

    Fields beyond the basics are payload conditional on ``order_type``:

    - ``grenade_fuse`` (seconds) — set when ``order_type == ORDER_GRENADE``.
    - ``det_slot`` (one of ``DET_START_PHASE1`` / ``DET_BETWEEN_PHASES`` /
      ``DET_END_PHASE2``) — set when ``order_type == ORDER_EXPLOSIVE``.
    - ``ammo_name`` (W3) — which ``[ammo.*]`` row the order delivers, for the
      LOBBED / PLACED archetypes: ``None`` = the shipped defaults
      (``"grenade_frag"`` for grenades, ``"demo_breach"`` for explosives —
      the UI path is unchanged; per-type loadout selection is W6). Tests
      exercise smoke/tear/poison grenades and C4 by naming the round here.

    ``ap_cost`` defaults to 1; movement orders get it patched to 0 by
    the placement code, since walking doesn't cost AP in v1.
    """

    def __init__(self, order_type, target_fx, target_fy, phase,
                 grenade_fuse=None, det_slot=None, ammo_name=None):
        self.order_type = order_type
        self.target_fx = target_fx
        self.target_fy = target_fy
        self.phase = phase                  # 0 = Phase 1, 1 = Phase 2
        self.grenade_fuse = grenade_fuse    # seconds; grenades only
        self.det_slot = det_slot            # slot ID; door explosives only
        self.ammo_name = ammo_name          # W3: LOBBED/PLACED round (None = default)
        self.ap_cost = 1                    # caller may overwrite
