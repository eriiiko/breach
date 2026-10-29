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
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Credited events (#123) — the closed CAUSE vocabulary + the attacker id.
# ---------------------------------------------------------------------------
class Cause(str, Enum):
    """What hurt a unit — the closed vocabulary of ``UnitHitEvent.source`` and
    ``UnitKilledEvent.killed_by`` (#123, rl-ready scope §3 item 4).

    The VALUES are the strings the producers emitted before #123, unchanged,
    so every consumer comparing against a literal (``ev.source == "heat"``)
    and the per-tick unit digest's hashing of the string stay byte-identical;
    the events themselves carry the plain string value (:func:`cause_value`),
    never the member, so a ``repr`` of an event record is what it was.

    One row per PRODUCER that exists today (derived from every
    ``apply_packet`` site, 2026-10-29):

    ==============  ==========================================================
    ``bullet``      a kinetic round's hit (``BulletInFlight`` — rifles, the
                    slow rounds)
    ``laser``       a hitscan beam (``fire_beam``)
    ``melee``       a melee strike (``resolve_melee``) and a zombie's bite
    ``explosion``   a payload's blast (``exchange.apply_blast_damage`` —
                    grenades, shells, placed and timed charges)
    ``heat``        the ``heat | max`` coupling row: fire, a flame jet, a
                    plasma splash (``exchange.apply_environmental_damage``)
    ``poison_gas``  the ``gas[poison]`` coupling row (``apply_poison_dose``)
    ``burning`` / ``poisoned`` / ``suffocating`` / ``regen``
                    the DoT/HoT status rows (``status.STATUS_REGISTRY``'s
                    names; ``status.py`` asserts every emitting row is here)
    ==============  ==========================================================

    NOT here because no damage path exists yet (``environment.py``: pressure,
    O2 and submersion damage are deferred): vacuum, pressure, o2, water,
    fall. A new damage source adds its row here in the same patch that adds
    the producer — an unknown string is refused at the packet door
    (``damage.apply_packet``) and by the events' own constructors.
    """
    BULLET      = "bullet"
    LASER       = "laser"
    MELEE       = "melee"
    EXPLOSION   = "explosion"
    HEAT        = "heat"
    POISON_GAS  = "poison_gas"
    BURNING     = "burning"
    POISONED    = "poisoned"
    SUFFOCATING = "suffocating"
    REGEN       = "regen"


#: Every cause string, in declaration order.
CAUSE_NAMES = tuple(c.value for c in Cause)

#: ``attacker_id`` of a hit nobody is credited with: the environment.
NO_ATTACKER = -1


def cause_value(cause) -> str:
    """The plain string of a cause (a :class:`Cause` member or its string),
    or ``ValueError`` for anything outside the closed vocabulary."""
    try:
        return Cause(cause).value
    except ValueError:
        raise ValueError(
            f"unknown damage cause {cause!r}: the closed vocabulary is "
            f"{CAUSE_NAMES} (simulation.events.Cause) -- a new damage "
            f"source adds its row there") from None


def attacker_value(attacker_id) -> int:
    """Normalize an attacker id: ``None`` and negatives are the environment
    (:data:`NO_ATTACKER`), anything else the unit id as a plain int."""
    if attacker_id is None:
        return NO_ATTACKER
    a = int(attacker_id)
    return a if a >= 0 else NO_ATTACKER


@dataclass
class ShotFiredEvent:
    """A bullet was fired from one tile toward another.

    Emitted by :func:`simulation.combat.fire_burst` once per bullet. The
    renderer draws a fading tracer line from ``from_tile`` to ``to_tile``.

    ``hit_target_id`` is the ID of the unit hit (or ``None`` for a miss /
    wall-stop). Useful for the renderer to attach a hit spark at the
    target end vs. just fading the tracer.

    ``launch`` is True on the segment a round LEAVES THE BARREL on (its first
    advance); a slow marching round (plasma bolt, 40 mm) emits one event per
    tick it flies, and only the first is a shot at the muzzle. The renderer's
    muzzle flash (#33) keys on it. Render-only, like the whole event -- not in
    the synced event digest.
    """
    unit_id: int                    # the shooter's unit id
    from_tile: tuple                # (fx, fy) — bullet origin
    to_tile: tuple                  # (fx, fy) — where the bullet ended up
    hit_target_id: Optional[int] = None
    launch: bool = True             # this segment starts at the muzzle


@dataclass
class LaserFiredEvent:
    """A hitscan beam was fired (mechanics/03 §5, W2 — the Lance family).

    Emitted by :func:`simulation.combat.fire_beam` once per trigger. The
    renderer draws a distinct beam line (brighter, redder, thicker than a
    bullet tracer). ``kind`` names the beam flavour for future variants
    (heavy laser, cutting beam). Render-only — NOT part of the synced event
    digest (the damage it caused rides UnitHitEvent as usual).

    The beam-as-light-source hookup (its glow feeding the light sweep as a
    transient cone emitter, renderer/frame_lights.py) is explicitly DEFERRED to the explosion-light pass
    (mechanics/03 §8 status note).
    """
    unit_id: int                    # the shooter's unit id
    from_tile: tuple                # (fx, fy) — beam origin
    to_tile: tuple                  # (fx, fy) — where the beam terminated
    kind: str = "laser"


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
    """A unit took damage this tick — a CREDITED event (#123).

    For blood-splatter sprites, damage numbers, screen flash on player hit,
    and (later) reward. ``source`` is the cause, a :class:`Cause` value (the
    plain string — "bullet", "explosion", "heat", "melee", ...); an unknown
    string is refused here. ``attacker_id`` is the unit credited with the
    hit, :data:`NO_ATTACKER` (-1) for the environment — see
    ``damage.apply_packet`` for who is credited with what.

    SYNCED: hashed into the per-tick unit digest with all four fields
    (``tests/field_ab_harness._SYNCED_EVENT_TYPES``).
    """
    unit_id: int
    damage: int
    source: str
    attacker_id: int = NO_ATTACKER

    def __post_init__(self):
        self.source = cause_value(self.source)
        self.attacker_id = attacker_value(self.attacker_id)


@dataclass
class UnitKilledEvent:
    """A unit died this tick — a CREDITED event (#123). Renderer plays the
    death animation. ``killed_by`` is the killing hit's cause (a
    :class:`Cause` value), ``attacker_id`` the unit credited with the kill,
    :data:`NO_ATTACKER` (-1) for the environment. SYNCED, like the hit."""
    unit_id: int
    killed_by: str
    attacker_id: int = NO_ATTACKER

    def __post_init__(self):
        self.killed_by = cause_value(self.killed_by)
        self.attacker_id = attacker_value(self.attacker_id)


@dataclass
class SprayJetEvent:
    """One tick of a live SPRAY burst (W6 — the flame-jet visual; the
    LaserFiredEvent precedent applied to the W4 archetype).

    Emitted by :func:`simulation.combat.process_sprays` on every tick a
    burst actually deposits (an interrupted / finished burst emits
    nothing), carrying the cone the deposits were aimed with: apex = the
    sprayer's centre tile, ``to_tile`` = the captured burst target,
    ``range_tiles`` + ``cone_half_angle_degrees`` = the weapon row's cone.
    The renderer draws a translucent jet fan + a transient light from it
    (warm for ``kind="flame"`` — the Dragon family; a fainter sickly
    green for ``kind="miasma"`` — the poison projector) and NEVER writes
    back. Emission is a pure function of already-synced state (unit
    position, the captured order target, weapon-row constants) and the
    determinism digest hashes only UnitHit/UnitKilled events
    (field_ab_harness._SYNCED_EVENT_TYPES), so the jet moves no digest.
    """
    unit_id: int                    # the sprayer's unit id
    from_tile: tuple                # (fx, fy) — cone apex (sprayer centre)
    to_tile: tuple                  # (fx, fy) — the burst's captured target
    range_tiles: int                # cone length (already tile-derived, W6)
    cone_half_angle_degrees: float  # the config half-angle convention
    kind: str = "flame"             # "flame" | "miasma"


@dataclass
class ProjectileGlowEvent:
    """An in-flight glowing round's position this tick (W6 — the plasma
    bolt visual; the LaserFiredEvent precedent for slow projectiles).

    Emitted by :class:`simulation.combat.BulletInFlight` each tick it
    advances while its round authors a nonempty ``glow`` profile
    (``[ammo.*] glow = "plasma"``). The renderer draws a glowing orb at
    ``pos`` + a transient light; the detonation flash itself rides the
    ordinary :class:`ExplosionEvent` (kind "shell"). Render-only — not
    part of the synced event digest.
    """
    pos: tuple                      # (fx, fy) — the bolt's position now
    kind: str = "plasma"            # glow profile name from the ammo row


@dataclass
class DoorDestroyedEvent:
    """A door tile was destroyed (typically by a charge). For sound + dust."""
    pos: tuple                      # (fy, fx) in tile coords (note: matches gmap convention)


@dataclass
class WallDestroyedEvent:
    """A wall tile was destroyed (explosion or burn-through)."""
    pos: tuple                      # (fy, fx)


__all__ = [
    "Cause",
    "CAUSE_NAMES",
    "NO_ATTACKER",
    "cause_value",
    "attacker_value",
    "ShotFiredEvent",
    "LaserFiredEvent",
    "SprayJetEvent",
    "ProjectileGlowEvent",
    "ExplosionEvent",
    "UnitHitEvent",
    "UnitKilledEvent",
    "DoorDestroyedEvent",
    "WallDestroyedEvent",
]
