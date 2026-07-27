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

# ---------------------------------------------------------------------------
# OnePhaseWEGO's added vocabulary (onephase_wego design §5). Ids >= 6 so every
# legacy value above keeps its exact meaning — TwoPhaseWEGO never sees these,
# and no shipped golden can move because an id was appended.
#
# Note ORDER_MOVE is NEW and DISTINCT from ORDER_MOVE_ATTACK: v1's Move does
# not auto-attack and runs at full speed (the old Sprint is folded into it),
# while Sprint and Move-w/-Cover are removed as separate orders (§5). Likewise
# ORDER_SHOOT is distinct from ORDER_FIRE: it targets a UNIT and its aim
# tracks that unit during execution, where ORDER_FIRE targets a tile.
# ---------------------------------------------------------------------------
ORDER_OVERWATCH  = 6
ORDER_AMBUSH     = 7
ORDER_HOLD       = 8
ORDER_SWAP       = 9
ORDER_MOVE_SHOOT = 10
ORDER_MARK       = 11
ORDER_MOVE       = 12
ORDER_SHOOT      = 13
ORDER_DETONATE   = 14

ORDER_NAMES = {
    ORDER_MOVE_ATTACK: "Move & Attack",
    ORDER_MOVE_COVER:  "Move w/ Cover",
    ORDER_SPRINT:      "Sprint",
    ORDER_GRENADE:     "Grenade",
    ORDER_EXPLOSIVE:   "Explosive",
    ORDER_FIRE:        "Fire",
    ORDER_OVERWATCH:   "Overwatch",
    ORDER_AMBUSH:      "Ambush",
    ORDER_HOLD:        "Hold",
    ORDER_SWAP:        "Weapon Swap",
    ORDER_MOVE_SHOOT:  "Move & Shoot",
    ORDER_MARK:        "Mark Target",
    ORDER_MOVE:        "Move",
    ORDER_SHOOT:       "Shoot",
    ORDER_DETONATE:    "Detonate",
}

# Movement-type orders (the ones with a marine-speed table behind them).
# UNCHANGED — this tuple is the TwoPhaseWEGO speed-table gate and is read all
# over the legacy path; OnePhaseWEGO's movement orders live in their own
# tuple below so widening one can never alter the other.
MOVE_ORDER_TYPES = (ORDER_MOVE_ATTACK, ORDER_MOVE_COVER, ORDER_SPRINT)

# The OnePhaseWEGO orders that displace the unit (§6 — aim-relative speeds).
ONEPHASE_MOVE_ORDER_TYPES = (ORDER_MOVE, ORDER_MOVE_SHOOT)

# The OnePhaseWEGO orders that put rounds downrange (§7 — spread selection).
ONEPHASE_SHOOT_ORDER_TYPES = (ORDER_SHOOT, ORDER_MOVE_SHOOT, ORDER_AMBUSH)

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
                 grenade_fuse=None, det_slot=None, ammo_name=None,
                 action_name=None, target_unit_id=None, start_tick=None,
                 det_tick=None, cone_half_deg=None, aim_anchor=None):
        self.order_type = order_type
        self.target_fx = target_fx
        self.target_fy = target_fy
        self.phase = phase                  # 0 = Phase 1, 1 = Phase 2
        self.grenade_fuse = grenade_fuse    # seconds; grenades only
        self.det_slot = det_slot            # slot ID; door explosives only
        self.ammo_name = ammo_name          # W3: LOBBED/PLACED round (None = default)
        self.ap_cost = 1                    # caller may overwrite

        # ---- OnePhaseWEGO payload (design §5) --------------------------
        # All default to None and are read ONLY by the OnePhaseWEGO executor,
        # so a legacy Order is byte-identical in behavior to its pre-P2 self.
        #
        # action_name    the action-registry row this order executes through
        #                (§5). None = derive from order_type — the reverse
        #                lookup ActionTable.for_order_type does.
        # target_unit_id the UNIT this order points at (§5: Shoot targets a
        #                unit and aim tracks it during execution; Ambush and
        #                Mark likewise), where target_fx/fy stay the TILE.
        # start_tick     ABSOLUTE tick this order may begin at — the
        #                "at_time" start condition (Hold-until-t, §5). The
        #                clock is monotonic under this ruleset, so an absolute
        #                tick survives the round seam unambiguously.
        # det_tick       ABSOLUTE detonation tick (§12) — replaces det_slot's
        #                start/between/end trichotomy with a moment the player
        #                picks anywhere in the round.
        # cone_half_deg  player-set overwatch cone half-angle (§9). Its
        #                PRIMARY purpose is target control: targets outside
        #                the cone are ignored.
        # aim_anchor     (x, y) the unit keeps its gun on while moving —
        #                the second half of Move & Shoot's two-part gesture
        #                (§6), and what makes "reversing" computable.
        self.action_name = action_name
        self.target_unit_id = target_unit_id
        self.start_tick = start_tick
        self.det_tick = det_tick
        self.cone_half_deg = cone_half_deg
        self.aim_anchor = aim_anchor
