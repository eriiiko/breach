"""EnvironmentProfile — species/unit survivability data (spec §7).

Data-only for the foundation pass. Behaviour (O2 reserve drain,
pressure damage, temperature damage, submersion) is deferred per
spec §13. The profile is attached to SpeciesDef as the base; effective
profile = base + equipment modifiers (modifier system deferred).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SubmersionRule(Enum):
    DROWNS         = "drowns"
    UNAFFECTED     = "unaffected"
    REQUIRES_WATER = "requires_water"


@dataclass(frozen=True)
class EnvironmentProfile:
    # --- Respiration ---
    breathes:          bool  = True
    can_breathe_air:   bool  = True
    can_breathe_water: bool  = False
    o2_reserve_max:    float = 60.0    # ticks of survival without an O2 source

    # --- Pressure (1.0 = standard atmosphere) ---
    pressure_min: float = 0.4
    pressure_max: float = 2.5

    # --- O2 partial pressure / concentration in the breathable medium ---
    o2_level_min: float = 0.15
    o2_level_max: float = 1.0

    # --- Temperature tolerance band (arbitrary scalar for now) ---
    temperature_min: float = -20.0
    temperature_max: float =  60.0

    submersion: SubmersionRule = SubmersionRule.DROWNS

    # Damage per tick while outside any tolerance — data only;
    # not applied by any tick handler yet (deferred per spec §13).
    environmental_damage_rate: float = 1.0


# The default field values ARE the human baseline.
HUMAN_ENVIRONMENT = EnvironmentProfile()


__all__ = ["SubmersionRule", "EnvironmentProfile", "HUMAN_ENVIRONMENT"]
