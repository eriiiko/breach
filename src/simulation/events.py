"""Tick events — transient signals emitted by the simulation each tick.

A *tick event* is something that happened during this tick that has **no
persistent state in the simulation** but the renderer should react to visually
(a tracer line, a screen shake, a blood splatter). Compare with
:class:`simulation.combat.Projectile`, which IS persistent simulation state
(it ticks across many frames, has a fuse and position, can collide).

Architectural split (from the patch plan):

- **Projectiles** = full simulation entities. Live on
  ``Simulation.projectiles``, ticked every frame, read directly by the
  renderer. Examples: grenades, plasma bolts, thrown items.

- **Tick events** = one-shot signals. Live on ``Simulation.tick_events``,
  **cleared at the start of every** :meth:`Simulation.step`. The sim does
  not own decay / fade timers — the renderer copies events out and manages
  its own short-lived effect queue (with fade timers). This keeps the sim
  state pure (deterministic, serializable, AI-friendly) while letting the
  renderer be expressive.

Event classes are plain dataclasses, no methods. The renderer
``consume_events(events)`` API matches on ``isinstance`` (or attribute
sniffing); each event carries only the data the renderer needs to spawn a
matching visual effect.

Adding a new event type: define a dataclass here, emit it from the
appropriate ``Simulation`` step, teach the renderer to recognise it in
``consume_events``. Do not store fade state on the event itself — that
belongs in the renderer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ShotFiredEvent:
    """A bullet was fired from one tile toward another.

    Emitted by :func:`simulation.combat.fire_burst` once per bullet. The
    renderer draws a fading tracer line from ``from_tile`` to ``to_tile``.

    ``hit_target_id`` is the ID of the unit hit (or ``None`` for a miss /
    wall-stop). Useful for the renderer to attach a hit spark at the
    target end vs. just fading the tracer.
    """
    unit_id: int                    # the shooter's unit id
    from_tile: tuple                # (fx, fy) — bullet origin
    to_tile: tuple                  # (fx, fy) — where the bullet ended up
    hit_target_id: Optional[int] = None


@dataclass
class ExplosionEvent:
    """A pressure / fire event that should produce a flash + screen shake.

    Emitted by grenade detonations and door explosives. ``kind`` lets the
    renderer pick a flavour ('grenade' vs 'door_explosive').
    """
    pos: tuple                      # (fx, fy)
    radius: int
    kind: str                       # "grenade" | "door_explosive"


@dataclass
class UnitHitEvent:
    """A unit took damage this tick.

    For blood-splatter sprites, damage numbers, screen flash on player hit.
    ``source`` is a short string identifying what hit them: "bullet",
    "explosion", "fire", "melee" — keep stable; the renderer may key
    effects off it.
    """
    unit_id: int
    damage: int
    source: str


@dataclass
class UnitKilledEvent:
    """A unit died this tick. Renderer plays death animation."""
    unit_id: int
    killed_by: str                  # "bullet" / "explosion" / "fire" / "melee" / ...


@dataclass
class DoorDestroyedEvent:
    """A door tile was destroyed (typically by a charge). For sound + dust."""
    pos: tuple                      # (fy, fx) in tile coords (note: matches gmap convention)


@dataclass
class WallDestroyedEvent:
    """A wall tile was destroyed (explosion or burn-through)."""
    pos: tuple                      # (fy, fx)


__all__ = [
    "ShotFiredEvent",
    "ExplosionEvent",
    "UnitHitEvent",
    "UnitKilledEvent",
    "DoorDestroyedEvent",
    "WallDestroyedEvent",
]
