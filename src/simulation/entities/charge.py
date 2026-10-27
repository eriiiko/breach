"""The `timed_charge` entity — a scripted or periodic detonation (issue #31).

Design: docs/prep_patches_handoff_2026-09-30.md P5 + §3 ("Timed charge").
A level-authored charge that executes one ``[payloads.*]`` row through THE
payload executor (:func:`simulation.payloads.execute_payload`) on the sim
clock: first at ``first_at_s``, then every ``period_s`` (0 = once). Built for
the explosion studio, reusable for benches and RL scenarios — never a
scheduler in ``main.py``, a test-only hook, or a second payload path.

This module is the SCHEMA only and stays IMPORT-LIGHT (stdlib, the entity
design §3b rule, CI-tested by tests/test_entities_import_light.py). The
sim-side runtime (the payload-name validation, the seconds -> ticks
quantization, the 9e sweep) lives in :mod:`simulation.timed_charge_system`,
the vents.py / vent_system.py split.

Not to be confused with :mod:`simulation.charges` (a UNIT-planted charge, an
ammo row, fired by a Detonate order or its own det tick): this one is placed
by the level author and names a payload row directly.

Field kinds: ``x``/``y``/``enabled`` are SYNCED kinds (moving or disabling a
charge moves the entity digest). ``first_at_s``/``period_s`` are
``KIND_LENGTH_M``-style authoring numbers (the ``tau_s`` precedent in
nodes.py: seconds, quantized ONCE at load into ticks — never used raw in the
sim path); their synced consequence is the runtime countdown row below.
``payload`` is ``KIND_STR`` — a config.toml row name, validated at load
against the PayloadTable (an unknown name is a loud load error).
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, KIND_BOOL, KIND_INT, KIND_LENGTH_M, KIND_STR, register,
)


@register
class timed_charge(Entity):
    """One placed charge that detonates a payload row on the sim clock."""

    INTANGIBLE = False    # sits on a tile (it occupies nothing; the blast site)

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="detonation tile COL at base resolution — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="detonation tile ROW at base resolution — REQUIRED"),
        Field("payload", KIND_STR, default=None,
              doc="the [payloads.<name>] config row it executes — REQUIRED; "
                  "validated at load (timed_charge_system.build_timed_"
                  "charges): an unknown name is a load error [not synced — "
                  "a config name; its effect is the fields it writes]"),
        Field("first_at_s", KIND_LENGTH_M, default=0.0, minimum=0.0,
              doc="seconds of sim clock before the first detonation — "
                  "quantized once at load via config.ticks_from_seconds "
                  "(so at least 1 tick)"),
        Field("period_s", KIND_LENGTH_M, default=0.0, minimum=0.0,
              doc="seconds between repeats; 0 = fire once — quantized once "
                  "at load via config.ticks_from_seconds"),
        Field("enabled", KIND_BOOL, default=True,
              doc="false = the charge never fires [synced]"),
    )
    SIGNALS = ()
    INPUTS = ()

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """The one synced runtime row: ``ticks_to_fire`` — sim steps left
        before the next detonation (0 = fires in this step's 9e sweep; -1 =
        spent or disabled). A COUNTDOWN, not an absolute ``next_fire_tick``:
        TwoPhaseWEGO rewinds ``sim.tick`` to 0 every round, so an absolute
        deadline would re-fire each round. Read off the
        :class:`simulation.timed_charge_system.TimedChargeRuntime` wrapper
        (a bare EntityInstance has none — loud AttributeError; digests only
        come from constructed sims, the door/pump/vent precedent)."""
        return (("ticks_to_fire", int(entity.ticks_to_fire)),)
