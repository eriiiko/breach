"""Simultaneous resolution of a tick's attacks (#169).

THE RULE (RL-ready scope §5 item 5, "simultaneous actions on one tick"): no
unit's ability to act in a tick may depend on its id relative to another
unit's. The attack slots walk their units (and in-flight rounds) in a fixed
order; if each hit were applied the moment it is resolved, a unit killed or
stunned earlier in that walk would never fire, and the lower id would win
every duel.

So an attack slot runs in TWO PHASES:

- **Phase A (decide)** -- the slot's ordinary walk. Every shooter's
  ELIGIBILITY (alive, ``composed_flags(u).can_act``, cadence, magazine,
  order / trigger) and every shot's GEOMETRY (which tile, wall, crate or
  unit a round or beam reaches, crit and exposure rolls) are decided while
  nothing any attack DOES has happened yet: the walk records its effects
  here instead of applying them, so every decision reads the slot-start
  state. RNG is drawn only in this phase, in the walk's fixed order -- the
  order it has always been drawn in.
- **Phase B (apply)** -- :meth:`SimultaneousEffects.resolve` applies the
  recorded effects in the order they were recorded (shooter by shooter in
  the slot's walk order, round by round within a shooter): damage packets
  through the one door (``damage.apply_packet``), attack statuses, facing
  snaps, wall chew, payload detonations, spray deposits. The hits' EXPOSURE
  raises (G4p, :mod:`simulation.exposure`) land together after the last
  effect, so every hit of the slot reads its target's slot-start exposure.
  (A packet a ``world`` effect applies -- a detonation's blast -- goes
  through the door directly and raises at once; no such packet carries an
  ``expose`` in v1.)

Consequences that hold by construction: two units can kill each other on
one tick; a unit killed this tick by another's round still got its own shot
off; a round stops at the first unit on its line even if another round
"already" killed that unit this tick (the body takes the round; its packet
still lands, the door emits its UnitHit but no second UnitKilled -- the
packet that takes hp to zero is the kill, credited to its attacker).

When no recorded effect changes anything another decision of the same slot
reads -- no hit on a unit that also acts, no wall chewed that a later round
crosses -- phase B applies exactly what the old one-pass walk applied, in
the same order, with the same RNG draws: such a tick is bit-identical to
the pre-#169 slot (``tests/test_simultaneous_shots.py`` gates it in lockstep).

:data:`IMMEDIATE` is the same interface applying every effect at once --
the pre-#169 behaviour, for direct callers of the combat helpers that own
no slot (tests, tools, OnePhaseWEGO's timeline).

Pure bookkeeping: no float, no RNG, no new synced state. The pending list
lives one slot and is never digested; its effect is (hp, statuses, facing,
walls and the hashed hit / kill events).
"""
from __future__ import annotations

from simulation.damage import apply_packet
from simulation.exposure import ExposureLanding
from simulation.status import apply_status

_HIT, _STATUS, _FACE, _WORLD = 0, 1, 2, 3


class SimultaneousEffects:
    """The effects an attack slot's phase A recorded, applied by
    :meth:`resolve` (phase B) in recording order. ``deferred=False`` is
    :data:`IMMEDIATE`: each effect applies when recorded.

    The four effect kinds are the whole interface; an attack path that
    changes the world or a unit records through one of them -- never a
    direct write inside phase A, which would let a later decision of the
    same slot read it:

    - :meth:`hit`    -- a damage packet through ``apply_packet``;
    - :meth:`status` -- an attack's delivery-site status (the baton's stun),
      applied only if the unit is still alive at its turn;
    - :meth:`face`   -- the shooter's facing snap (a target's facing is what
      a crit or a facing-defense roll of another shooter reads);
    - :meth:`world`  -- any other world write (wall chew, a payload
      detonation, a spray's deposit): a call run at its turn.
    """

    __slots__ = ("deferred", "_pending")

    def __init__(self, deferred: bool = True):
        self.deferred = bool(deferred)
        self._pending = []

    # -- recording ----------------------------------------------------------
    def hit(self, unit, packet, events, *, source: str) -> None:
        """``apply_packet(unit, packet, events, source=source)`` at its turn.
        The caller decided the hit against the slot-start state (the unit
        was alive then); a unit an earlier effect killed still takes the
        packet -- the door emits no second UnitKilled."""
        if self.deferred:
            self._pending.append((_HIT, unit, packet, events, source))
        else:
            apply_packet(unit, packet, events, source=source)

    def status(self, unit, kind: int, *, magnitude, duration_ticks: int,
               source_id) -> None:
        """``apply_status`` at its turn, if ``unit`` is alive then (statuses
        freeze on corpses; a killing blow stuns no corpse)."""
        if self.deferred:
            self._pending.append((_STATUS, unit, kind, magnitude,
                                  duration_ticks, source_id))
        else:
            _status_if_alive(unit, kind, magnitude, duration_ticks, source_id)

    def face(self, unit, facing) -> None:
        """``unit.facing = facing`` at its turn."""
        if self.deferred:
            self._pending.append((_FACE, unit, facing))
        else:
            unit.facing = facing

    def world(self, fn, *args, **kwargs) -> None:
        """``fn(*args, **kwargs)`` at its turn -- a world write (the wall
        chew, a payload detonation, a spray deposit). ``fn`` must draw no
        RNG: phase B draws nothing, so the draw order stays phase A's."""
        if self.deferred:
            self._pending.append((_WORLD, fn, args, kwargs))
        else:
            fn(*args, **kwargs)

    # -- phase B ------------------------------------------------------------
    def resolve(self) -> None:
        """Apply every recorded effect in recording order and empty the list.
        A no-op for :data:`IMMEDIATE` (it never records).

        The slot's EXPOSURE (G4p) lands at the end: every recorded hit reads
        its target's exposure as it stood when the slot began, and the raises
        (and the zeroing of a unit the slot killed) of all of them land
        together after the last effect (an
        :class:`~simulation.exposure.ExposureLanding`), so two shooters on one
        tick take the same damage per round whatever their ids."""
        pending, self._pending = self._pending, []
        landing = ExposureLanding()
        for eff in pending:
            kind = eff[0]
            if kind == _HIT:
                _, unit, packet, events, source = eff
                apply_packet(unit, packet, events, source=source,
                             exposure_landing=landing)
            elif kind == _STATUS:
                _, unit, skind, magnitude, duration, source_id = eff
                _status_if_alive(unit, skind, magnitude, duration, source_id)
            elif kind == _FACE:
                _, unit, facing = eff
                unit.facing = facing
            else:
                _, fn, args, kwargs = eff
                fn(*args, **kwargs)
        landing.land()

    def __len__(self) -> int:
        return len(self._pending)


def _status_if_alive(unit, kind, magnitude, duration_ticks, source_id):
    if unit.alive:
        apply_status(unit, kind, magnitude=magnitude,
                     duration_ticks=duration_ticks, source_id=source_id)


#: Apply-at-once: the pre-#169 one-pass behaviour, for callers that own no
#: slot. Stateless (it never records), so one shared instance serves all.
IMMEDIATE = SimultaneousEffects(deferred=False)


def effects_or_immediate(effects):
    """``effects`` if given, else :data:`IMMEDIATE`."""
    return IMMEDIATE if effects is None else effects


__all__ = ["SimultaneousEffects", "IMMEDIATE", "effects_or_immediate"]
