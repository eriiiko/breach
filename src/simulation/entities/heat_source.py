"""The `heat_source` entity -- a thing that is hot every tick (levelgen
vocabulary E.2 row 3, docs/levelgen_zone_vocabulary_2026-10-29.md; Erik's
ruling 2026-10-31 accepts the new slot-9e tick logic, RL-batch habit §A).

An engine, a galley stove, a furnace: a level-authored source that deposits
``heat_per_s`` into the ``heat`` ingress plane every tick through THE heat
deposit path (:func:`simulation.payloads.deposit_heat` -- a ``heat``
FieldEdit the temperature solver converts and books in its own group, never
a ``temperature[...] =``). When an explosion's disc takes its ``hp`` to 0 it
dies and executes its ``death_payload`` row once through THE payload
executor -- the engine that delivers IGNITE_HEAT when it goes (CLAUDE.md
"Starting a fire": the payload row carries the heat).

This module is the SCHEMA only and stays IMPORT-LIGHT (stdlib, the entity
design §3b rule, CI-tested by tests/test_entities_import_light.py). The
sim-side runtime (validation by name, the once-only quantization, the 9e
sweep, the blast hook) lives in :mod:`simulation.heat_source_system` -- the
charge.py / timed_charge_system.py split.

Field kinds: ``x``/``y``/``radius``/``hp``/``enabled`` are SYNCED kinds
(moving, resizing, hardening or disabling a source moves the entity digest).
``heat_per_s`` and ``max_temp`` are ``KIND_LENGTH_M``-style authoring
numbers (the ``tau_s`` / ``trigger_temp`` precedent: quantized ONCE at load,
never used raw in the sim path). ``death_payload`` is ``KIND_STR`` -- a
config.toml row name, validated at load. The runtime's synced consequences
are the ``hp`` runtime row (Q16.16, the ``wall_hp`` currency) and the free
``alive`` row (0 once the source has died).
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, KIND_BOOL, KIND_INT, KIND_LENGTH_M, KIND_STR, register,
)

# The largest disc radius a source may author (runtime tiles). A bound, not
# physics: a furnace that heats a 33-tile disc is an authoring slip.
MAX_RADIUS = 16


@register
class heat_source(Entity):
    """One placed heat source (engine, stove, furnace) hot every tick."""

    INTANGIBLE = False    # sits on a floor tile (it occupies nothing)

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="source tile COL at base resolution -- REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="source tile ROW at base resolution -- REQUIRED"),
        Field("heat_per_s", KIND_LENGTH_M, default=None, minimum=0.0,
              doc="heat delivered per second at the source tile, in the "
                  "`heat` plane's real unit (the [payloads.*] heat_amount "
                  "unit; a tile's dT = heat / thermal_mass) -- REQUIRED > 0; "
                  "divided by the tick rate and quantized ONCE at load to a "
                  "Q16.16 per-tick deposit (heat_source_system)"),
        Field("radius", KIND_INT, default=0, minimum=0, maximum=MAX_RADIUS,
              doc="disc radius in runtime tiles beyond the source tile "
                  "(0 = its own tile only); the deposit falls off linearly, "
                  "full at the source, 1/(radius+1) at the rim"),
        Field("enabled", KIND_BOOL, default=True,
              doc="false = the source deposits nothing (it can still be "
                  "destroyed) [synced]"),
        Field("hp", KIND_INT, default=200, minimum=1,
              doc="structural hit points in the [materials.*] hp unit "
                  "(steel 200): a blast disc subtracts its wall_damage with "
                  "the wall law's linear falloff; at 0 the source dies "
                  "[synced; the live value is the `hp` runtime row]"),
        Field("death_payload", KIND_STR, default="",
              doc="the [payloads.<name>] row executed once when hp reaches "
                  "0 (\"\" = none) -- validated at load by name [not synced "
                  "-- a config name; its effect is the fields it writes]"),
        Field("max_temp", KIND_LENGTH_M, default=0.0, minimum=0.0,
              doc="cap in GAME DEGREES above ambient: the source skips its "
                  "deposit while its own tile's temperature reads at or above "
                  "it (overshoot at most one deposit); 0 = no cap -- "
                  "quantized ONCE at load to Q16.16"),
    )
    SIGNALS = ()
    INPUTS = ()

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """The synced runtime row: ``hp`` -- the source's live hit points in
        Q16.16 (the ``wall_hp`` currency, ``wall_fixed``), 0 once dead (the
        free ``alive`` row then reads 0 too). Read off the
        :class:`simulation.heat_source_system.HeatSourceRuntime` wrapper (a
        bare EntityInstance has none -- loud AttributeError; digests only
        come from constructed sims, the timed_charge precedent)."""
        return (("hp", int(entity.hp_q16)),)
