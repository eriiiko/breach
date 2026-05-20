"""Unit stat generation — sample a stat vector from a SpeciesDef (spec §11).

Called at Unit construction. Named/important characters can supply overrides
(spec §11.2) to hand-set specific stats while sampling the rest from the
species distribution.
"""
from __future__ import annotations

import numpy as np

from simulation.species import SpeciesDef, GENERATED_STAT_NAMES
from simulation.stats import BaseStats


def sample_unit_attributes(
    species: SpeciesDef,
    rng: np.random.Generator | None = None,
    overrides: dict | None = None,
) -> tuple[BaseStats, float, float]:
    """Sample stats from ``species.stat_dist`` and return (base_stats, mass, base_speed).

    The 10-dim stat vector is drawn from the species' multivariate normal,
    then each component is clamped to [clamp_min, clamp_max]. The first 8
    components map to BaseStats fields (strength…will_orientation); the last
    two (mass, base_speed) are returned separately because they live directly
    on Unit rather than in BaseStats (spec §3.2, §5, §8).

    Parameters
    ----------
    species:
        The SpeciesDef to sample from.
    rng:
        numpy random Generator. If None, a fresh unseeded RNG is created so
        unit tests can construct units without booting a Simulation.
    overrides:
        Optional {stat_name: value} applied AFTER clamping (spec §11.2).
        Used for named characters whose key stats are hand-set.
    """
    if rng is None:
        rng = np.random.default_rng()

    sd = species.stat_dist
    vec = rng.multivariate_normal(sd.mean, sd.covariance())
    vec = np.clip(vec, sd.clamp_min, sd.clamp_max)

    if overrides:
        for name, value in overrides.items():
            i = GENERATED_STAT_NAMES.index(name)
            vec[i] = float(value)

    # First 8 → BaseStats, last 2 → mass + base_speed.
    base = BaseStats(
        strength=float(vec[0]),
        agility=float(vec[1]),
        endurance=float(vec[2]),
        vitality=float(vec[3]),
        intelligence=float(vec[4]),
        will_strength=float(vec[5]),
        imagination=float(vec[6]),
        will_orientation=float(vec[7]),
    )
    mass       = float(vec[8])
    base_speed = float(vec[9])
    return base, mass, base_speed


__all__ = ["sample_unit_attributes"]
