"""The speed table (gameplay pack v1, G1p-c; ``docs/bots_zombies_speeds_design_2026-11-10.md`` §1).

Every unit speed on the continuous path is a ``[speeds]`` row in METRES PER
SECOND (Erik's ruling, 2026-11-10: the tiles-per-second dials were leftovers).
This module is the table's two doors:

* :func:`speed_settings` -- the CONFIG door: the six keys, each a number,
  refused by name when missing or mistyped, plus Erik's rules on them, so a
  later retune cannot break a rule silently:

  - the zombie's sprint is within 70 % .. 90 % of the marine's walk
    (:data:`SPRINT_WALK_LO` / :data:`SPRINT_WALK_HI`), and
  - it is ABOVE the marine's backward walk (``backward_frac x walk``): a
    sprinting zombie catches a marine backing away;

  and the rules the speed law needs to be monotone: ``0 < walk <= full``,
  ``0 < backward_frac <= sideways_frac <= 1``.

* :func:`speed_table` -- the MAP door: the settings quantized ONCE, at sim
  construction, with the map's own ``tile_size_m`` and ``[clock]
  ticks_per_second``. The marine's two speeds become COSTS in Q16.16 ticks
  per tile (``quantize(tile_size_m * tps / mps)``) -- the cadence currency
  :func:`movement.default_speed` composes terrain and wind in, so the scaling
  is that function's law unchanged, at a finer currency; the direction
  factors become Q16 fractions; the zombie's shamble becomes the INTEGER
  cadence its walk law uses (``ticks_from_seconds(tile_size_m / mps, tps)``,
  the old ``[zombie] seconds_per_tile`` derivation). A map on which a unit
  would cover a whole tile or more in one tick is refused.

The speed LAW (direction and throttle) is ``movement.move_speed_tiles_per_
tick``; this module holds numbers only. The only floats are the doors' own
quantization of the config dials, once (the turn-rate door's precedent,
``turning.turn_rate_q16``).

The Q16 COST is chosen over a Q16 SPEED (tiles per tick) on purpose: a
cadence the old dials gave, 1/18 tile a tick, has no Q16 tiles-per-tick value
but is exactly 18 x ONE ticks per tile, so the law reproduces the old walk bit
for bit when the table is set to it (the patch's proof of cause), and the
terrain / wind reduction is literally ``default_speed`` (one law, never a
second copy of it).
"""
from __future__ import annotations

from dataclasses import dataclass

from simulation.unit_fixed import FP_ONE, quantize_scalar

#: Erik's rule (2026-11-10): a sprinting zombie reaches 70 % .. 90 % of a
#: marine's walk.
SPRINT_WALK_LO = 0.70
SPRINT_WALK_HI = 0.90

#: The six keys of ``[speeds]``, in the order the door checks them.
SPEED_KEYS = ("marine_walk_mps", "marine_full_mps", "zombie_shamble_mps",
              "zombie_sprint_mps", "sideways_frac", "backward_frac")


@dataclass(frozen=True)
class SpeedSettings:
    """``[speeds]`` as read by the door, in m/s and plain fractions."""
    marine_walk_mps: float
    marine_full_mps: float
    zombie_shamble_mps: float
    zombie_sprint_mps: float
    sideways_frac: float
    backward_frac: float


@dataclass(frozen=True)
class SpeedTable:
    """The settings quantized for ONE map (:func:`speed_table`).

    ``walk_cost_q`` / ``full_cost_q``: the marine's walk (throttle 0) and full
    speed (throttle ONE) as Q16.16 TICKS PER TILE on open floor in calm air
    (``full_cost_q > ONE``: never a whole tile in a tick). ``side_q`` /
    ``back_q``: the direction factors in Q16. ``zombie_shamble_ticks``: the
    zombie's integer walk cadence (ticks per tile). ``zombie_sprint_cost_q``:
    the zombie sprint in Q16 ticks per tile -- read and checked now, used by
    nothing until the zombie patch (G9p)."""
    walk_cost_q: int
    full_cost_q: int
    side_q: int
    back_q: int
    zombie_shamble_ticks: int
    zombie_sprint_cost_q: int
    tile_size_m: float
    ticks_per_second: int

    def realised_mps(self, cost_q: int) -> float:
        """A Q16 cost back in m/s (reports, tests): ``tile * tps * ONE / cost``."""
        return self.tile_size_m * self.ticks_per_second * FP_ONE / int(cost_q)

    @property
    def zombie_shamble_realised_mps(self) -> float:
        """The shamble the zombie actually walks at, after its cadence
        quantization (an integer number of ticks per tile)."""
        return self.tile_size_m * self.ticks_per_second / self.zombie_shamble_ticks


def _number(section, key):
    if section is None or not hasattr(section, key):
        raise ValueError(f"[speeds] is missing {key!r}")
    v = getattr(section, key)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"[speeds] {key} must be a number, got {v!r}")
    return float(v)


def speed_settings(cfg) -> SpeedSettings:
    """``[speeds]`` of a loaded config (``CFG``) through the refusing door:
    every key of :data:`SPEED_KEYS` present and a number, and Erik's rules
    (module doc). ValueError naming the key or the rule otherwise."""
    section = getattr(cfg, "speeds", None)
    if section is None:
        raise ValueError("config has no [speeds] table")
    s = SpeedSettings(**{k: _number(section, k) for k in SPEED_KEYS})
    for key in ("marine_walk_mps", "marine_full_mps", "zombie_shamble_mps",
                "zombie_sprint_mps", "sideways_frac", "backward_frac"):
        if not getattr(s, key) > 0:
            raise ValueError(f"[speeds] {key} must be > 0, "
                             f"got {getattr(s, key)!r}")
    if s.marine_full_mps < s.marine_walk_mps:
        raise ValueError(
            f"[speeds] marine_full_mps ({s.marine_full_mps}) is below "
            f"marine_walk_mps ({s.marine_walk_mps}): more throttle would be "
            f"slower")
    if not s.backward_frac <= s.sideways_frac <= 1.0:
        raise ValueError(
            f"[speeds] needs backward_frac <= sideways_frac <= 1 (the walk "
            f"slows monotonically as it turns away from the facing), got "
            f"sideways_frac {s.sideways_frac}, backward_frac {s.backward_frac}")
    lo = SPRINT_WALK_LO * s.marine_walk_mps
    hi = SPRINT_WALK_HI * s.marine_walk_mps
    if not lo <= s.zombie_sprint_mps <= hi:
        raise ValueError(
            f"[speeds] zombie_sprint_mps ({s.zombie_sprint_mps}) must be "
            f"within {SPRINT_WALK_LO:.0%} .. {SPRINT_WALK_HI:.0%} of "
            f"marine_walk_mps ({s.marine_walk_mps}): {lo:.4g} .. {hi:.4g} "
            f"m/s (Erik's rule, 2026-11-10)")
    back = s.backward_frac * s.marine_walk_mps
    if not s.zombie_sprint_mps > back:
        raise ValueError(
            f"[speeds] zombie_sprint_mps ({s.zombie_sprint_mps}) must be "
            f"above the marine's backward walk, backward_frac x "
            f"marine_walk_mps = {back:.4g} m/s: a sprinting zombie catches a "
            f"marine walking backwards (Erik's rule, 2026-11-10)")
    return s


def _cost_q(mps: float, tile_size_m: float, tps: int, key: str) -> int:
    cost = quantize_scalar(tile_size_m * tps / mps)
    if cost <= FP_ONE:
        raise ValueError(
            f"[speeds] {key} = {mps} m/s covers a whole tile or more per "
            f"tick on a {tile_size_m} m tile at {tps} tps (cost "
            f"{cost / FP_ONE:.4g} ticks per tile): a step would skip tiles")
    return cost


def speed_table(cfg, tile_size_m: float) -> SpeedTable:
    """The ``[speeds]`` table quantized ONCE for a map of ``tile_size_m``
    metres per tile (``GameMap.tile_size_m``) at ``[clock]
    ticks_per_second``: see the module doc. Refuses what
    :func:`speed_settings` refuses, a non-positive tile, and a speed that
    would cover a whole tile or more in one tick."""
    from config import ticks_from_seconds
    s = speed_settings(cfg)
    tile = float(tile_size_m)
    if not tile > 0:
        raise ValueError(f"tile_size_m must be > 0, got {tile_size_m!r}")
    tps = int(cfg.clock.ticks_per_second)
    walk = _cost_q(s.marine_walk_mps, tile, tps, "marine_walk_mps")
    full = _cost_q(s.marine_full_mps, tile, tps, "marine_full_mps")
    sprint = _cost_q(s.zombie_sprint_mps, tile, tps, "zombie_sprint_mps")
    _cost_q(s.zombie_shamble_mps, tile, tps, "zombie_shamble_mps")
    return SpeedTable(
        walk_cost_q=walk,
        full_cost_q=full,
        side_q=quantize_scalar(s.sideways_frac),
        back_q=quantize_scalar(s.backward_frac),
        zombie_shamble_ticks=ticks_from_seconds(tile / s.zombie_shamble_mps,
                                                tps),
        zombie_sprint_cost_q=sprint,
        tile_size_m=tile,
        ticks_per_second=tps,
    )


__all__ = ["SPEED_KEYS", "SPRINT_WALK_LO", "SPRINT_WALK_HI", "SpeedSettings",
           "SpeedTable", "speed_settings", "speed_table"]
