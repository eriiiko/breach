"""Faction types — foundation pass (spec §10.1).

FactionId is a plain int alias. The FactionRelationshipTable (dynamic
per-mission friend/foe table) is deferred per spec §13. For now combat
code reads unit.team directly; faction_id is an alias for it.

Team semantics (#119 P1): ``team`` is ANY int and hostility is "a different
team" — marine teams are counted from 0. Zombies live on their own team,
:data:`ZOMBIE_TEAM`, chosen far outside any marine roster count so the two can
never collide; ``unit.is_zombie`` (state, set from the team at construction and
by conversion) is the ONLY zombie check — never a team literal. Import-light
(stdlib only): the level loader and the editor read it.
"""
from __future__ import annotations

from enum import Enum


# Foundation pass: just a type alias. Full relationship table comes later.
FactionId = int

#: The zombies' team. Not 1: marine teams are 0, 1, 2, ... (#119).
ZOMBIE_TEAM: FactionId = 255

#: The zombie code in LEVEL DATA written before #119 (``[[spawn]] team = 1``,
#: a motion sensor's ``faction_filter = 1``). Level data only — the loader
#: maps it to :data:`ZOMBIE_TEAM`; no runtime code reads it.
LEGACY_ZOMBIE_TEAM: FactionId = 1


def team_from_level(team: int) -> FactionId:
    """A team int as authored in level data -> the runtime team: the legacy
    zombie code 1 -> :data:`ZOMBIE_TEAM`; every other value (0, ZOMBIE_TEAM
    itself, -1 = "any" in a filter) passes through. Called only at the level
    doors (``level_loader`` for ``[[spawn]]``, ``sensor_system`` for
    ``faction_filter``)."""
    team = int(team)
    return ZOMBIE_TEAM if team == LEGACY_ZOMBIE_TEAM else team


class Stance(Enum):
    """Defined for completeness — not yet consulted by any code (spec §10.1)."""
    ALLIED   = "allied"
    FRIENDLY = "friendly"
    NEUTRAL  = "neutral"
    HOSTILE  = "hostile"


__all__ = ["FactionId", "Stance", "ZOMBIE_TEAM", "LEGACY_ZOMBIE_TEAM",
           "team_from_level"]
