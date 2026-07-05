"""Species definitions — data-driven unit generator parameters (spec §4, §11).

Foundation pass: one species only ("human"), covering marines and zombies
(is_zombie is a runtime state, not a separate species). No "ogryn" or
"gray" in this pass (locked decision #4).

Stat distribution decisions (agent):
  - vitality mean = 100.0 (= CFG.marine.hp, the current HP baseline)
  - vitality stddev = 15.0 — so ±1 stddev spans 85–115, ±2 spans 70–130
  - base_speed mean = 1.0 (1.0 = nominal cadence; derived movement logic
    maps this to ticks-per-tile separately). stddev = 0.1 (tight variance).
  - Human stats on a 1–10 scale for player-visible stats; mass in kg.
  - Correlations: mass<->strength +0.6, mass<->agility -0.3,
    mass<->base_speed -0.2 (heavier humans tend to be stronger but slower).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from config import CFG
from simulation import unit_fixed
from simulation.damage import (
    HEAT, MitigationProfile, NEUTRAL_MITIGATION, build_mitigation,
)
from simulation.environment import EnvironmentProfile, HUMAN_ENVIRONMENT
from simulation.inventory import InventoryProfile


SpeciesId = str   # "human", "gray", "ogryn", ... — just a string for now

N_GENERATED_STATS = 10

# Order MUST match BaseStats field order (strength…will_orientation)
# followed by mass and base_speed. Used by generation.py to map vector
# components back to named attributes.
GENERATED_STAT_NAMES = (
    "strength", "agility", "endurance", "vitality", "intelligence",
    "will_strength", "imagination", "will_orientation",
    "mass", "base_speed",
)


@dataclass(frozen=True)
class StatDistribution:
    """Multivariate-normal parameters for stat generation (spec §11.1).

    mean / stddev shape: (N_GENERATED_STATS,)
    correlation: (N, N) symmetric, unit diagonal, positive semi-definite
    clamp_min / clamp_max: hard bounds applied after sampling
    """
    mean:        np.ndarray   # shape (N_GENERATED_STATS,)
    stddev:      np.ndarray   # shape (N_GENERATED_STATS,)
    correlation: np.ndarray   # shape (N, N)
    clamp_min:   np.ndarray
    clamp_max:   np.ndarray

    def covariance(self) -> np.ndarray:
        """Compose covariance from stddev + correlation (spec §11.1)."""
        s = self.stddev
        return np.outer(s, s) * self.correlation


@dataclass(frozen=True)
class SpeciesDef:
    """Static species record — one per species, data-driven (spec §4).

    Designers add species by adding data here; no subclasses.
    """
    id:                  SpeciesId
    name:                str
    stat_dist:           StatDistribution
    default_offsets:     tuple              # tuple[tuple[int, int], ...]
    environment:         EnvironmentProfile = field(default_factory=lambda: HUMAN_ENVIRONMENT)
    inventory_profile:   InventoryProfile   = field(default_factory=InventoryProfile)
    # Mitigation tables (mechanics/06 §3) — flat armor[dtype] + resist_mult
    # [dtype], door-2 data. Default = neutral (armor 0, resist 1.0). A future
    # robot species ships resist_mult[POISON] = 0 here — profile data, no code.
    mitigation:          MitigationProfile  = field(default_factory=lambda: NEUTRAL_MITIGATION)
    can_become_zombie:   bool = True
    nn_intelligence_tier: int = 0           # data-only; NN tier logic deferred


# ---------------------------------------------------------------------------
# Human species definition
# ---------------------------------------------------------------------------

def _default_3x3_offsets() -> tuple:
    """Standard human footprint: 3×3 tile square, anchored at top-left."""
    return tuple((dx, dy) for dy in range(3) for dx in range(3))


def _human_stat_distribution() -> StatDistribution:
    """Human stat distribution parameters.

    Agent tuning decisions:
    - vitality mean = 100 → reproduces CFG.marine.hp exactly
    - base_speed mean = 1.0 → nominal; movement cadence is derived elsewhere
    - Player-visible stats (strength, agility, etc.) on a 1–10 scale;
      stddev = 1.0 for most, 1.5 for intelligence/imagination (more spread)
    - will_orientation centred at 0, small stddev (most humans near neutral)
    - mass mean = 80 kg (healthy adult); stddev = 10 kg
    - Correlations per spec §11.1: mass<->strength, mass<->agility,
      mass<->base_speed
    """
    mean = np.array([
        5.0,    # strength
        5.0,    # agility
        5.0,    # endurance
        100.0,  # vitality (= CFG.marine.hp baseline)
        5.0,    # intelligence
        5.0,    # will_strength
        5.0,    # imagination
        0.0,    # will_orientation (centred; [-1, +1])
        80.0,   # mass kg
        1.0,    # base_speed (1.0 = nominal)
    ], dtype=np.float64)

    stddev = np.array([
        1.0,    # strength
        1.0,    # agility
        1.0,    # endurance
        15.0,   # vitality: ±1σ ≈ ±15 HP, ±2σ spans 70–130 HP
        1.5,    # intelligence
        1.0,    # will_strength
        1.5,    # imagination
        0.2,    # will_orientation (small; most humans near neutral)
        10.0,   # mass kg
        0.1,    # base_speed (tight: most humans near nominal speed)
    ], dtype=np.float64)

    n = N_GENERATED_STATS
    corr = np.eye(n, dtype=np.float64)
    idx = {name: i for i, name in enumerate(GENERATED_STAT_NAMES)}

    def _link(a: str, b: str, r: float) -> None:
        i, j = idx[a], idx[b]
        corr[i, j] = corr[j, i] = r

    _link("mass", "strength",   0.6)
    _link("mass", "agility",   -0.3)
    _link("mass", "base_speed", -0.2)

    clamp_min = np.array([
        1.0,    # strength
        1.0,    # agility
        1.0,    # endurance
        20.0,   # vitality (never below 20 — minimum survivable)
        1.0,    # intelligence
        1.0,    # will_strength
        1.0,    # imagination
       -1.0,    # will_orientation
        30.0,   # mass kg (extreme minimum: very small human)
        0.3,    # base_speed
    ], dtype=np.float64)

    clamp_max = np.array([
        10.0,   # strength
        10.0,   # agility
        10.0,   # endurance
        300.0,  # vitality (extreme upper bound; not a tank)
        10.0,   # intelligence
        10.0,   # will_strength
        10.0,   # imagination
         1.0,   # will_orientation
        200.0,  # mass kg (extreme upper bound)
        2.0,    # base_speed
    ], dtype=np.float64)

    return StatDistribution(
        mean=mean, stddev=stddev, correlation=corr,
        clamp_min=clamp_min, clamp_max=clamp_max,
    )


HUMAN = SpeciesDef(
    id="human",
    name="Human",
    stat_dist=_human_stat_distribution(),
    default_offsets=_default_3x3_offsets(),
    environment=HUMAN_ENVIRONMENT,
    mitigation=NEUTRAL_MITIGATION,   # unarmored, unresisting baseline
    can_become_zombie=True,
    nn_intelligence_tier=2,
)

# ---------------------------------------------------------------------------
# The ZOMBIE mitigation profile (mechanics/06 §2 "proof of shape").
# ---------------------------------------------------------------------------
# Zombie-ness is runtime STATE on the human species (the one-species
# foundation decision above), so this is the STATE'S profile overlay rather
# than a species field: damage.mitigation_for selects it on ``unit.is_zombie``
# at damage time — exactly the predicate the dissolved special case used, so
# construction with team=1, end-of-round conversion, and direct flag flips
# all behave as before. resist_mult[HEAT] REPLACES the old
# ``if u.is_zombie: dmg *= CFG.zombie.fire_damage_multiplier`` branch in the
# heat response (a vulnerability is just a resistance above 1); the shipped
# 4.0 is an exact binary scale, so the swap was bit-identical.
#
# STANDARD VALUE (mechanics/06 §8): since P5 the value is SOURCED from
# ``[zombie] fire_damage_multiplier`` in config.toml (feel comment + range
# there), read ONCE here at module import and Q16.16-snapped by
# build_mitigation (door 2) -> RESTART to apply a config edit, not Ctrl+R
# (this constant + the profile pointers on live units don't re-bind). The
# heat tests read the same key as the expected damage ratio, so sim + tests
# move together. All other slots neutral: bullets keep their site-side
# ``bullet_damage_multiplier`` math (an int-truncating pre-mitigation amount
# rule, not a resistance).
ZOMBIE_MITIGATION = build_mitigation(resist_mult={
    HEAT: float(getattr(CFG.zombie, "fire_damage_multiplier", 4.0)),
})

# Zombie knockdown stability (mechanics/06 §4) — the same state-overlay
# pattern as ZOMBIE_MITIGATION: zombie-ness is runtime state on the human
# species, so the zombie's `stability` is selected on ``unit.is_zombie`` at
# exchange time (exchange._stability_for) instead of living on a species
# EnvironmentProfile. 0.9 = shamblers topple a little easier than a braced
# marine (effective knockdown threshold = knockdown_dv_threshold * 0.9).
# STANDARD VALUE for Erik to tune (mechanics/06 §8) — since P5 SOURCED from
# ``[zombie] stability`` in config.toml (feel comment + range there), read
# ONCE at module import, snapped onto the Q16.16 grid at definition (door 2;
# 0.9 is not dyadic) -> RESTART to apply, not Ctrl+R.
ZOMBIE_STABILITY = unit_fixed.dequantize_scalar(unit_fixed.quantize_scalar(
    float(getattr(CFG.zombie, "stability", 0.9))))

# Registry: all known species indexed by SpeciesId string.
SPECIES_REGISTRY: dict[str, SpeciesDef] = {HUMAN.id: HUMAN}


def get_species(species_id: SpeciesId) -> SpeciesDef:
    """Look up a species by id. Raises KeyError for unknown ids."""
    return SPECIES_REGISTRY[species_id]


__all__ = [
    "SpeciesId", "N_GENERATED_STATS", "GENERATED_STAT_NAMES",
    "StatDistribution", "SpeciesDef",
    "HUMAN", "ZOMBIE_MITIGATION", "ZOMBIE_STABILITY",
    "SPECIES_REGISTRY", "get_species",
]
