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

    # --- Stability (mechanics/06 §4, KNOCKED_DOWN) ---
    # Knockdown susceptibility multiplier: a blast knocks the unit down when
    # its per-tick |Δv| >= knockdown_dv_threshold * stability (the wave_p
    # push row, mechanics/05 §1). 1.0 = human baseline; < 1 topples easier
    # (shamblers), > 1 resists toppling (a low four-legged robot). This is
    # the ONE non-physical knockdown knob (mass and footprint are Newtonian;
    # the resistance table mitigates damage only — mechanics/06 §4 division
    # of labor). Door-2 data: authored values must sit on the Q16.16 grid
    # (1.0 is exact; non-dyadic values snap at definition — see
    # species.ZOMBIE_STABILITY for the pattern).
    stability: float = 1.0

    submersion: SubmersionRule = SubmersionRule.DROWNS

    # Damage per tick while outside any tolerance — data only;
    # not applied by any tick handler yet (deferred per spec §13).
    environmental_damage_rate: float = 1.0


# The default field values ARE the human baseline.
HUMAN_ENVIRONMENT = EnvironmentProfile()


__all__ = ["SubmersionRule", "EnvironmentProfile", "HUMAN_ENVIRONMENT"]
