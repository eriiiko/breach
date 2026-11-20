"""Noise (gameplay pack v1 G9p; ``docs/bots_zombies_speeds_design_2026-11-10.md``
§3, the hearing row; Erik's ruling of 2026-11-10: zombies hear by distance
alone -- walls ignored, no sound propagation).

THE RULE: what can be heard is a NOISE -- a ``(tile y, tile x, radius in
tiles)`` -- appended to ``GameMap.noise_log`` by its source through
:func:`emit`; hearing reads noises, never the sources. The sources:

- every shot -- each trigger pull of a ranged marcher
  (``combat._dispatch_trigger``: projectile and hitscan, every path that
  fires one) and each spray burst that starts, at the shooter's centre tile,
  with the weapon row's ``loudness`` (``[weapons.*] loudness``, METRES of
  hearing radius, quantized once per map to whole tiles,
  ``WeaponDef.loudness_tiles``; REQUIRED on every row that fires);
- every detonation -- ``payloads.execute_payload``, at the detonation tile,
  with the payload row's ``loudness`` (``[payloads.*] loudness``, metres,
  optional, 0 = silent; ``PayloadDef.loudness_tiles``);
- every door bash -- the horde's strike on a closed door
  (``ai_zombie``), at the struck tile, ``[zombie] bash_loudness_m``.

The log is a plain per-tick list (never digested, never recorded, never a
field: the ``blast_log`` precedent), DRAINED once per tick at the zombie
slot (``Simulation._run_zombie_slot`` -> :func:`take`), so a noise is heard
at the first zombie slot after it was made: a shot or a grenade of slot 2 / 4
the same tick, a timed charge of slot 9e and a bash of slot 5 itself the
next. ACCEPTED GAP: the log crosses that tick boundary unsynced (a save /
load between a slot-9e detonation and the next slot 5 would lose it).
"""
from __future__ import annotations


def radius_tiles(metres: float, tile_size_m: float) -> int:
    """A hearing radius in metres as whole tiles on a map of
    ``tile_size_m`` (the weapon range's rule: one divide, round half up);
    0 stays 0 (silent)."""
    m = float(metres)
    if m <= 0.0:
        return 0
    return int(m / float(tile_size_m) + 0.5)


def emit(gmap, fy: int, fx: int, radius: int) -> None:
    """Record a noise at tile ``(fy, fx)`` heard within ``radius`` tiles.
    A radius of 0 is silence (nothing recorded); a map double without a
    log (test stubs) records nothing."""
    if radius <= 0:
        return
    log = getattr(gmap, "noise_log", None)
    if log is not None:
        log.append((int(fy), int(fx), int(radius)))


def take(gmap) -> list:
    """Drain the log: the noises made since the last zombie slot, in the
    order they were made; the log is left empty."""
    log = getattr(gmap, "noise_log", None)
    if not log:
        return []
    out = list(log)
    log.clear()
    return out


__all__ = ["radius_tiles", "emit", "take"]
