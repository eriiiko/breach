"""Timed-charge RUNTIME — the slot-9e detonation sweep (issue #31, P5).

Design: docs/prep_patches_handoff_2026-09-30.md P5 + §3 ("Timed charge"). The
sim-side runtime of the :class:`simulation.entities.charge.timed_charge`
entity (the schema stays import-light there; everything that touches the
gmap / the payload table lives HERE — the vents.py / vent_system.py split).

A charge executes ONE ``[payloads.*]`` row through THE payload executor
(:func:`simulation.payloads.execute_payload`) — the same call, argument order
and ``rng`` hand-off as the grenade fuse-out (``Simulation._update_
projectiles``) and the door charge (``combat.process_door_explosives``). The
executor never draws from ``rng`` and neither does this sweep.

Clock: seconds are quantized ONCE at load (:func:`config.ticks_from_seconds`,
so at least one tick). The synced state is a per-charge COUNTDOWN,
``ticks_to_fire`` (steps left before the next detonation; -1 = spent or
disabled) rather than an absolute ``next_fire_tick``: ``TwoPhaseWEGO`` rewinds
``sim.tick`` to 0 at every round boundary, and an absolute deadline compared
against a rewound clock would re-fire every round. The countdown advances once
per ``Simulation.step()`` actually taken (a paused sim does not step), so a
charge fires in step ``first + k*period`` counted from load under EVERY
ruleset.

Effects: ``apply_explosion``'s structural wall damage is immediate; every
FieldEdit the executor enqueues here (slot 9e, after 6b's flush) lands at the
NEXT step's 6b flush — the same one-tick latency the door charge (slot 8) has.

Dormancy: a charge-free level builds an empty list, the 9e call is a single
``if self._timed_charges`` check, and no ENTITY_SECT record exists for a
charge that does not exist — so every charge-free digest is byte-identical.
"""
from __future__ import annotations

from config import ticks_from_seconds
from simulation.payloads import execute_payload

# The ExplosionEvent flavour (the renderer draws every ExplosionEvent alike;
# kind only names the source).
EVENT_KIND = "timed_charge"


class TimedChargeError(ValueError):
    """A timed_charge row that cannot be built (unknown payload, off-grid)."""


class TimedChargeRuntime:
    """Sim-side runtime object for one ``timed_charge``.

    Doubles as the SERIALIZER runtime object (duck-typed ordinal / id /
    class_name / fields + ``alive``, the VentRuntime pattern), so the class's
    ``runtime_digest_rows`` reads ``ticks_to_fire`` straight off the sim's
    entity list.
    """

    __slots__ = ("inst", "fy", "fx", "payload", "period_ticks",
                 "ticks_to_fire", "alive")

    def __init__(self, inst, fy, fx, payload, first_ticks, period_ticks,
                 enabled):
        self.inst = inst
        self.fy = int(fy)
        self.fx = int(fx)
        self.payload = payload                 # the resolved PayloadDef
        self.period_ticks = int(period_ticks)  # 0 = fire once
        self.ticks_to_fire = int(first_ticks) if enabled else -1
        self.alive = True

    # --- serializer duck-type -------------------------------------------
    @property
    def ordinal(self):
        return self.inst.ordinal

    @property
    def id(self):
        return self.inst.id

    @property
    def class_name(self):
        return self.inst.class_name

    @property
    def fields(self):
        return self.inst.fields


def build_timed_charges(sim) -> list:
    """Build the ordinal-ordered runtime list and REPLACE each
    ``timed_charge`` instance in ``sim.entities`` with its runtime wrapper.

    Must run after ``sim.weapons_tables`` exists: each ``payload`` name is
    validated against the PayloadTable HERE (load time) — an unknown name
    raises :class:`TimedChargeError` naming the charge and the known rows.
    ``x``/``y`` are base-resolution tiles scaled by ``res_factor`` (the
    vent/door/pump pattern) and bounds-checked hard.
    """
    insts = sorted((e for e in sim.entities if e.class_name == "timed_charge"),
                   key=lambda e: int(e.ordinal))
    if not insts:
        return []
    rf = int(getattr(sim.level, "res_factor", 1) or 1)
    tps = int(sim._tps)
    h, w = sim.gmap.solid.shape
    by_name = sim.weapons_tables.payloads.by_name
    charges = []
    for e in insts:
        f = e.fields
        name = f["payload"]
        if name not in by_name:
            raise TimedChargeError(
                f"timed_charge '{e.id}': payload {name!r} names no "
                f"[payloads.*] row in config.toml (known: {sorted(by_name)})")
        fy, fx = rf * int(f["y"]), rf * int(f["x"])
        if not (0 <= fy < h and 0 <= fx < w):
            raise TimedChargeError(
                f"timed_charge '{e.id}': tile ({fy}, {fx}) is out of the "
                f"{h}x{w} grid (base tiles x={f['x']}, y={f['y']}, scaled "
                f"by res_factor {rf})")
        first = ticks_from_seconds(float(f["first_at_s"]), tps)
        period_s = float(f["period_s"])
        period = ticks_from_seconds(period_s, tps) if period_s > 0 else 0
        rt = TimedChargeRuntime(e, fy, fx, by_name[name], first, period,
                                bool(f["enabled"]))
        charges.append(rt)
        _replace_entity(sim.entities, e, rt)
    return charges


def sweep_timed_charges(sim) -> None:
    """Slot 9e: fire every charge whose countdown reached 0, in ordinal
    order, then advance the others one step. Integer-only, no RNG drawn."""
    for c in sim._timed_charges:
        if c.ticks_to_fire < 0:
            continue
        if c.ticks_to_fire == 0:
            execute_payload(sim.gmap, sim.edit_queue, sim.units, c.fy, c.fx,
                            c.payload, sim.rng, events=sim.tick_events,
                            kind=EVENT_KIND)
            c.ticks_to_fire = c.period_ticks - 1 if c.period_ticks > 0 else -1
        else:
            c.ticks_to_fire -= 1


def _replace_entity(entities, old, new) -> None:
    """Swap ``old`` for its runtime wrapper, preserving ordinal position —
    mirrors :func:`simulation.vent_system._replace_entity`."""
    for i, e in enumerate(entities):
        if e is old:
            entities[i] = new
            return
    raise ValueError(                     # pragma: no cover - defensive
        f"timed_charge {getattr(old, 'id', old)!r} not found in the sim "
        f"entity list")


__all__ = ["EVENT_KIND", "TimedChargeError", "TimedChargeRuntime",
           "build_timed_charges", "sweep_timed_charges"]
