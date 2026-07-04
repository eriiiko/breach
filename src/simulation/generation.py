"""Unit stat generation — deterministic predefined attributes (spec §11).

Called at Unit construction. Named/important characters can supply overrides
(spec §11.2) to hand-set specific stats.

INGRESS RULE (2026-07-04, docs/lenovo_dev_setup.md §8b): spawn stats are SYNCED
state (current_hp = vitality; mass/base_speed drive movement), so every number
here must enter through an approved door. This module uses door 2 — config
constants quantized ONCE onto the Q16.16 grid at the boundary.

The draft sampler this replaces (``sample_unit_attributes``, drawing the stat
vector via ``rng.multivariate_normal``) was cross-machine NON-deterministic:
numpy's MVN factorizes the covariance through LAPACK (SVD), whose result
depends on the CPU-dispatched BLAS kernel — and with the species covariance's
repeated variances + correlations the differences are O(sigma), not ULPs
(mass 96 -> 64 under a forced kernel flip). It caused the tick-0 __unit_hp__
cross-machine digest divergence found by the 2026-07-04 Ada confirm run.

Stat VARIATION remains a design goal: it returns with the units/stats
redesign as a deterministic sampler (seeded integer RNG stream -> pure
algebraic transform -> Q16.16 snap). Git history holds the draft.
"""
from __future__ import annotations

from simulation.species import SpeciesDef, GENERATED_STAT_NAMES
from simulation.stats import BaseStats

Q16_ONE = 65536


def _q16_snap(v: float) -> float:
    """Quantize a config-authored float onto the Q16.16 grid and return the
    EXACT dyadic float n/65536 (ingress door 2). Round half away from zero,
    matching the C++ kit's quantize (cpp/src/fixed_point.h)."""
    n = int(abs(v) * Q16_ONE + 0.5)
    return (-n if v < 0.0 else n) / Q16_ONE


def predefined_unit_attributes(
    species: SpeciesDef,
    overrides: dict | None = None,
) -> tuple[BaseStats, float, float]:
    """Deterministic spawn attributes; returns (base_stats, mass, base_speed).

    The 10-dim stat vector is the species' MEAN vector, snapped onto the
    Q16.16 grid and clamped to [clamp_min, clamp_max]. No RNG is involved —
    every unit of a species spawns identical until the deterministic sampler
    lands with the stats redesign. The first 8 components map to BaseStats
    fields (strength…will_orientation); the last two (mass, base_speed) are
    returned separately because they live directly on Unit rather than in
    BaseStats (spec §3.2, §5, §8).

    Parameters
    ----------
    species:
        The SpeciesDef whose distribution MEANS define the stat vector.
    overrides:
        Optional {stat_name: value} for named characters (spec §11.2).
        Overrides are Q16.16-snapped too — they are config-authored numbers
        entering synced state (door 2) — and applied AFTER clamping, like the
        draft sampler did (a named character may exceed species clamps).
    """
    sd = species.stat_dist
    vec = [
        min(max(_q16_snap(float(m)), float(l)), float(h))
        for m, l, h in zip(sd.mean.tolist(),
                           sd.clamp_min.tolist(), sd.clamp_max.tolist())
    ]

    if overrides:
        for name, value in overrides.items():
            i = GENERATED_STAT_NAMES.index(name)
            vec[i] = _q16_snap(float(value))

    # First 8 → BaseStats, last 2 → mass + base_speed.
    base = BaseStats(
        strength=vec[0],
        agility=vec[1],
        endurance=vec[2],
        vitality=vec[3],
        intelligence=vec[4],
        will_strength=vec[5],
        imagination=vec[6],
        will_orientation=vec[7],
    )
    mass       = vec[8]
    base_speed = vec[9]
    return base, mass, base_speed


__all__ = ["predefined_unit_attributes"]
