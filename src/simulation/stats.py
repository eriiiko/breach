"""Unit stat types: StatId enum, BaseStats/EffectiveStats dataclasses,
per-stat accessor functions, and the player-visibility rule.

Foundation pass: compute_effective_stats returns base unchanged. The
modifier system (spec §1.2, §13) slots in here later.
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
    vitality:         float   # max HP pool
    intelligence:     float
    will_strength:    float
    imagination:      float   # hidden until unit.awakened
    will_orientation: float   # hidden permanently; [-1, +1]


# EffectiveStats has the same shape as BaseStats — alias for clarity.
# Modifier system will produce actual derived values here later.
EffectiveStats = BaseStats


def compute_effective_stats(unit) -> EffectiveStats:
    """Foundation pass: returns base unchanged.

    Modifier system (spec §1.2, §13) slots in here later — wounds,
    zombification, encumbrance, fear, buffs all compose here.
    """
    return unit.base_stats


# Per-stat accessors — gameplay code calls these, never reads BaseStats directly.
# This ensures the modifier system can be slotted in transparently (spec §3.2).

def effective_vitality(unit) -> float:
    return compute_effective_stats(unit).vitality

def effective_strength(unit) -> float:
    return compute_effective_stats(unit).strength

def effective_agility(unit) -> float:
    return compute_effective_stats(unit).agility

def effective_endurance(unit) -> float:
    return compute_effective_stats(unit).endurance

def effective_intelligence(unit) -> float:
    return compute_effective_stats(unit).intelligence

def effective_will_strength(unit) -> float:
    return compute_effective_stats(unit).will_strength

def effective_imagination(unit) -> float:
    return compute_effective_stats(unit).imagination

def effective_will_orientation(unit) -> float:
    return compute_effective_stats(unit).will_orientation


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


__all__ = [
    "StatId", "BaseStats", "EffectiveStats",
    "compute_effective_stats",
    "effective_vitality", "effective_strength", "effective_agility",
    "effective_endurance", "effective_intelligence", "effective_will_strength",
    "effective_imagination", "effective_will_orientation",
    "is_stat_player_visible",
]
