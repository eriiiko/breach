"""The `duct` + `vent` entities — vent system PATCH 1 (mechanism).

Design: docs/vent_system_design_2026-08-23.md (v2, post-adversarial-critique),
§2/§7 patch 1 scope. This module carries the SCHEMA (fields / signals / the
duct's runtime-row NAMES) for the two registry rows the design's §2 splits
out: the ``duct`` is the plenum entity (intangible — logic, not a tile), the
``vent`` is the physical aperture that feeds it. The sim-side runtime (the
9e(d) circulation sweep, the extended gas-N primitives it drives, the filter
lookup) lives in :mod:`simulation.vent_system`; this package stays
IMPORT-LIGHT (stdlib only — entity design §3b, CI-tested in
tests/test_entities_import_light.py) exactly like door.py / nodes.py /
sensors.py.

Patch-1 scope (design §7 item 1, orchestrator-pinned): duct + vent entities,
the circulation term, the plenum ledger, the filter table, the extended gas-N
primitives, the 9e(d) sweep, runtime guards, digest rows. EXPLICITLY OUT OF
SCOPE here (patches 2-3): the makeup P-controller, EMA/P_ctrl SENSING (no
pressure sensing at all in patch 1 — the design's §3 "Sensing" open question
is moot until patch 3 needs it), reserve-depletion gameplay, any tuning.
``q_makeup_max`` and ``state`` are declared now (schema-reserved, per §2) but
UNUSED — the patch-1 runtime never reads them; every vent behaves as if
``state == "open"`` (§2: "v1: always open. Reserved so #49 adds
closed/damaged-open/welded-shut without migration").
"""
from __future__ import annotations

from simulation.entities.schema import (
    Entity, Field, KIND_ENTITY_REF, KIND_ENUM, KIND_INT, KIND_LENGTH_M,
    KIND_STR, register,
)

# ---------------------------------------------------------------------------
# The aperture model (§2) — the single mechanism every vent, however mounted,
# reduces to: one gas-mass-delta tile. Floor mount: the vent's own tile (the
# physically-correct 2D projection of an out-of-plane ceiling/floor jet, §2).
# Wall mount: the OPEN tile in front of the wall face, addressed by a compass
# `facing` (never a raw dx/dy — the field is a digest-token enum like every
# other synced choice, §2's `facing` column).
# ---------------------------------------------------------------------------
FACINGS = ("n", "e", "s", "w")

# (dy, dx) the facing points INTO — e.g. "n" faces the tile one row above the
# mount (the wall's north face opens onto the tile north of it).
FACING_OFFSETS = {"n": (-1, 0), "e": (0, 1), "s": (1, 0), "w": (0, -1)}

MOUNTS = ("floor", "wall")
ROLES = ("supply", "return")
# Reserved for #49 (§2/§8) — v1 forces "open" behavior regardless of the
# authored value; the enum exists now so #49 never needs a schema migration.
VENT_STATES = ("open", "closed", "damaged_open", "welded_shut")


def resolve_aperture_base(fields: dict) -> tuple:
    """The BASE-resolution aperture tile ``(fy, fx)`` (§2) — floor mount: the
    vent's own anchor tile; wall mount: the open tile in front of ``facing``.
    Pure; the sim-side runtime (vent_system.build_vents) scales this by
    ``res_factor`` to the gmap grid, mirroring the door/sensor/pump pattern.
    """
    x, y = int(fields["x"]), int(fields["y"])
    if fields["mount"] == "floor":
        return (y, x)
    dy, dx = FACING_OFFSETS[fields["facing"]]
    return (y + dy, x + dx)


@register
class duct(Entity):
    """The plenum entity (§2, ⟨crit⟩ "two registry rows, not one"): a
    logic-only row (no tile — the pattern of ``breach_site`` / the logic
    nodes) that owns the CIRCULATING BULK RESERVE its member vents share.

    Authoring fields only; the plenum STATE (the bulk pair, the trace
    composition, the energy ledger, the counted sinks) is runtime-only — it
    rides ``runtime_digest_rows``, never a schema ``Field`` (design §5).
    """

    INTANGIBLE = True     # logic, not a tile — the breach_site/node pattern

    FIELDS = (
        Field("reserve_size", KIND_LENGTH_M, default=1.0, minimum=0.0,
              doc="reserve capacity in real units, quantized once at load "
                  "(door-2/pump `rate` idiom). SCHEMA-RESERVED in patch 1 — "
                  "the reserve-depletion gameplay is patch 3 (design §7 item "
                  "3); the patch-1 plenum ledger is unbounded by this field."),
        Field("filter", KIND_STR, default="derelict",
              doc="the [filters.<name>] config row this duct's intake runs "
                  "through (§4) — loader-validated at build_vents (a "
                  "config-integrity error, not the generic dangling-entity-"
                  "ref warning: this name addresses config.toml, not "
                  "another [[entity]])."),
    )
    SIGNALS = ()
    INPUTS = ()

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """§5: the plenum ledger — bulk reserve pair, trace composition
        vector, the energy currency, the counted filter sinks + the energy
        wipe channel, and the T-rail hit counters (§4's "counter-tracked"
        clamp). Read off the :class:`simulation.vent_system.DuctRuntime`
        wrapper (a bare EntityInstance has none of these — loud
        AttributeError, digests only come from constructed sims, the door/
        pump precedent).
        """
        rows = [
            ("o2_raw", int(entity.o2_raw)),
            ("n2_raw", int(entity.n2_raw)),
            ("e_plenum", int(entity.e_plenum)),
            ("e_wipe", int(entity.e_wipe)),
            ("rail_lo_hits", int(entity.rail_lo_hits)),
            ("rail_hi_hits", int(entity.rail_hi_hits)),
        ]
        # trace_i / sink_i index by TRACE gas id (simulation.gases: STEAM=0,
        # SMOKE=1, POISON=2, TEARGAS=3, FUEL_GAS=4 — N_TRACE_GASES order).
        # Not imported here (import-light, §3b — gases.py pulls numpy); the
        # mapping lives sim-side in vent_system.py where it's authoritative.
        for i, v in enumerate(entity.trace_raw):
            rows.append((f"trace_{i}", int(v)))
        for i, v in enumerate(entity.sink):
            rows.append((f"sink_{i}", int(v)))
        return tuple(rows)


@register
class vent(Entity):
    """One aperture feeding a duct's plenum (§2) — mass-only circulation in
    patch 1 (no makeup term, no pressure sensing — see the module doc)."""

    INTANGIBLE = False    # a placed aperture with a mount tile (like a sensor)

    FIELDS = (
        Field("x", KIND_INT, default=None, minimum=0,
              doc="mount tile COL at base resolution — REQUIRED"),
        Field("y", KIND_INT, default=None, minimum=0,
              doc="mount tile ROW at base resolution — REQUIRED"),
        Field("mount", KIND_ENUM, default="floor", choices=MOUNTS,
              doc="floor = the vent's own tile is the aperture (canon: "
                  "ceiling/floor — the top-down sim can't tell, §2); wall = "
                  "the open tile in front of `facing` is the aperture"),
        Field("facing", KIND_ENUM, default="n", choices=FACINGS,
              doc="wall mounts only: the compass face the aperture opens "
                  "onto (n/e/s/w — ignored for a floor mount)"),
        Field("duct", KIND_ENTITY_REF, default="",
              doc="the `duct` entity this vent's plenum is (§2 ruling 2). "
                  "Empty/dangling = unwired: the vent builds but never "
                  "sweeps (fail-safe, the bus-free pump precedent)."),
        Field("role", KIND_ENUM, default="supply", choices=ROLES,
              doc="supply = deposits from the plenum; return = extracts "
                  "into the plenum. Editor-assigned, never self-negotiated "
                  "(§2 ruling 3)."),
        Field("q_circ", KIND_LENGTH_M, default=0.0, minimum=0.0,
              doc="circulation throughput, real units/s — quantized ONCE at "
                  "load into a raw-per-second Bresenham accrual rate (§3 "
                  "Quantization; door-2/pump `rate` idiom). 0 = inert vent."),
        Field("q_makeup_max", KIND_LENGTH_M, default=0.0, minimum=0.0,
              doc="rate limit on the makeup term. SCHEMA-RESERVED — the "
                  "makeup controller is patch 3 (design §7 item 3); the "
                  "patch-1 runtime never reads this field."),
        Field("state", KIND_ENUM, default="open", choices=VENT_STATES,
              doc="SCHEMA-RESERVED (§2/§8, for #49's damage/console states "
                  "without a migration) — v1 ignores this field: every vent "
                  "behaves as if it were `open`, regardless of the authored "
                  "value."),
    )
    SIGNALS = ()
    INPUTS = ()

    @classmethod
    def runtime_digest_rows(cls, entity) -> tuple:
        """§3 Quantization / §5: the per-vent Bresenham flux-accrual
        accumulator — the ONLY vent runtime row in patch 1 (no `P_ctrl`: the
        design's §5 EMA row is patch 3's, out of scope here — see the module
        doc). Read off the :class:`simulation.vent_system.VentRuntime`
        wrapper."""
        return (("accum", int(entity.accum)),)
