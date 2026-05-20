"""Faction types — foundation pass (spec §10.1).

FactionId is a plain int alias. The FactionRelationshipTable (dynamic
per-mission friend/foe table) is deferred per spec §13. For now combat
code reads unit.team directly; faction_id is an alias for it.
"""
from __future__ import annotations

from enum import Enum


# Foundation pass: just a type alias. Full relationship table comes later.
FactionId = int


class Stance(Enum):
    """Defined for completeness — not yet consulted by any code (spec §10.1)."""
    ALLIED   = "allied"
    FRIENDLY = "friendly"
    NEUTRAL  = "neutral"
    HOSTILE  = "hostile"


__all__ = ["FactionId", "Stance"]
