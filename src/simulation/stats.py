"""Unit stat types -- the CHARACTER SHEET's vocabulary (StatId, BaseStats)
and the player-visibility rule.

``stats.py`` is the sheet's vocabulary; nothing composes here and no law
reads a stat (#29 U1, ``docs/unit_design_2026-11-14.md`` D2): a stat acts
on the game only through the compile step (``unit_traits.compile_unit``,
the hooks) and the accessors of the hooked values. The old
``compute_effective_stats`` / ``effective_*`` seam is deleted (one compose
point: the accessors); the full health is ``unit_traits.max_hp_q``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class StatId(Enum):
    STRENGTH        = "strength"
    AGILITY         = "agility"
    ENDURANCE       = "endurance"
    VITALITY        = "vitality"
    INTELLIGENCE    = "intelligence"
    WILL_STRENGTH   = "will_strength"
    IMAGINATION     = "imagination"
    WILL_ORIENTATION = "will_orientation"
    MASS            = "mass"
    BASE_SPEED      = "base_speed"


@dataclass(frozen=True)
class BaseStats:
    strength:         float
    agility:          float
    endurance:        float
    vitality:         float   # sheet input (#199); the full health is unit_traits.max_hp_q
    intelligence:     float
    will_strength:    float
    imagination:      float   # hidden until unit.awakened
    will_orientation: float   # hidden permanently; [-1, +1]


def is_stat_player_visible(stat: StatId, unit) -> bool:
    """UI policy for which stats are shown on the character sheet (spec §3.3).

    - will_orientation: never visible.
    - imagination: visible only if unit.awakened.
    - All others: always visible.
    """
    if stat is StatId.WILL_ORIENTATION:
        return False
    if stat is StatId.IMAGINATION:
        return getattr(unit, "awakened", False)
    return True


__all__ = ["StatId", "BaseStats", "is_stat_player_visible"]
