"""Faction types — foundation pass (spec §10.1).

FactionId is a plain int alias. The FactionRelationshipTable (dynamic
per-mission friend/foe table) is deferred per spec §13. For now combat
code reads unit.team directly; faction_id is an alias for it.
"""
from __future__ import annotations

from enum import Enum


# Foundation pass: just a type alias. Full relationship table comes later.
FactionId = int

# No faction (arc #63 P2): the swarm-unit default; v1 never targets swarm
# units. Team ints in use are non-negative, so this never aliases a team.
FACTION_NONE: FactionId = -1


class Stance(Enum):
    """Defined for completeness — not yet consulted by any code (spec §10.1)."""
    ALLIED   = "allied"
    FRIENDLY = "friendly"
    NEUTRAL  = "neutral"
    HOSTILE  = "hostile"


__all__ = ["FactionId", "Stance", "FACTION_NONE"]
