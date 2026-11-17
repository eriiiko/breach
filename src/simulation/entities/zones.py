"""Zone entity classes — ``breach_site`` + ``extraction_zone`` (Arc A, A8),
and ``swarm_brood`` (arc #63 P4: where swarm units start).

Level editor v3 design §5 (LOCKED 2026-07-18): TWO zone classes; the swarm
arc's brood (engine/17 §11) is the third, in the same idiom. Zones
are matter-first — the painted mask lives in ``zones.npy`` (uint8 paint-id
grid, shape == tilemap, 0 = unpainted, discovered by presence next to
level.toml); each zone is an ``[[entity]]`` instance carrying ``zone_id`` =
its integer paint id. The grid holds paint ids; the instance holds
everything else. The binding validators (every painted id has exactly one
instance; duplicate ``zone_id`` = load error; orphaned paint / zero-tile
instance = warning) are the loader's (level_loader._validate_zone_binding).

Both classes are intangible (design §5) and carry NO runtime behavior and no
runtime digest rows: a zone never occupies the grid, never steps, never
emits beyond the free ``alive``. ``zone_id``/``faction`` are KIND_INT and so
enter the ENTITY_SECT_V1 presence digest (A4) like any synced field —
entity-free levels stay bit-identical (dormancy).

``breach_site.roster`` is ``[[unit_type, count], ...]`` where ``unit_type``
is UNIT-system vocabulary — units are NOT entities (entity design §3e), so
no registry validation exists beyond it being a non-empty string. Spawn
realization (positions randomized inside the zone from the level seed — the
ML variation hook) is stack-2's, not Arc A's.
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, KIND_INT, KIND_ROSTER, KIND_STR, register,
)

# The zone paint-id namespace is ONE space across both classes: zones.npy is
# uint8, 0 = unpainted, so valid paint ids are 1..255.
ZONE_ID_MIN = 1
ZONE_ID_MAX = 255


@register
class breach_site(Entity):
    """Where a faction's force enters the map (editor design §5)."""

    INTANGIBLE = True   # a zone never occupies the grid (design §5)

    FIELDS = (
        Field("zone_id", KIND_INT, default=None,
              minimum=ZONE_ID_MIN, maximum=ZONE_ID_MAX,
              doc="REQUIRED integer paint id — the value painted in "
                  "zones.npy (uint8; 0 = unpainted, so ids are 1..255). "
                  "One instance per painted id, one id per instance."),
        Field("faction", KIND_INT, default=0, minimum=0,
              doc="owning faction/team int (editor design §5 instance "
                  "tuple; same integer vocabulary as [[spawn]] team)"),
        Field("roster", KIND_ROSTER, default=(),
              doc="[[unit_type, count], ...] — unit_type is UNIT-system "
                  "vocabulary (units are not entities, design §3e; never "
                  "registry-validated); spawn realization is stack-2's"),
    )


@register
class extraction_zone(Entity):
    """Where a faction's units must reach to extract (editor design §5)."""

    INTANGIBLE = True   # a zone never occupies the grid (design §5)

    FIELDS = (
        Field("zone_id", KIND_INT, default=None,
              minimum=ZONE_ID_MIN, maximum=ZONE_ID_MAX,
              doc="REQUIRED integer paint id — the value painted in "
                  "zones.npy (uint8; 0 = unpainted, so ids are 1..255). "
                  "One instance per painted id, one id per instance."),
        Field("faction", KIND_INT, default=0, minimum=0,
              doc="faction/team int that extracts here (editor design §5 "
                  "instance tuple; same integer vocabulary as [[spawn]] "
                  "team)"),
    )


# The default brood size (arc #63 P4): 200 larvae, about two per tile over a
# 10 x 10 room -- enough to read as a swarm, a fortieth of the default store.
BROOD_DEFAULT_COUNT = 200


@register
class swarm_brood(Entity):
    """Where swarm units start (arc #63 P4, engine/17 §11): a painted zone a
    level fills with ``count`` units of ``species`` at load, before tick 0.

    A third zone class in the same paint-id namespace (the binding
    validators are the loader's, unchanged). It never ticks and keeps no
    runtime state: the scatter is drawn from the match draws
    (``simulation/swarm_brood.py``), never from a seed column (Erik's
    ruling: one source of draws)."""

    INTANGIBLE = True   # a zone never occupies the grid

    FIELDS = (
        Field("zone_id", KIND_INT, default=None,
              minimum=ZONE_ID_MIN, maximum=ZONE_ID_MAX,
              doc="REQUIRED integer paint id -- the value painted in "
                  "zones.npy (uint8; 0 = unpainted, so ids are 1..255). "
                  "One instance per painted id, one id per instance."),
        Field("species", KIND_STR, default="larva",
              doc="the swarm species the brood spawns (a name of "
                  "simulation.swarm.SWARM_SPECIES; an unknown one is refused "
                  "at load by name)"),
        Field("count", KIND_INT, default=BROOD_DEFAULT_COUNT, minimum=1,
              doc="how many units the brood scatters over the zone's free "
                  "(non-solid) tiles at load; past the species' capacity the "
                  "load is refused (SwarmFull), never clamped"),
    )
