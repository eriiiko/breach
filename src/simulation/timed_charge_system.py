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

Heat trigger (levelgen vocabulary E.2 row 1, Erik's ruling 2026-10-31): a
``trigger = "heat"`` charge waits, ``ticks_to_fire = -1`` and its synced
``fired`` latch 0, until the sweep READS ``gmap.temperature`` on its own tile
at or above ``trigger_temp_q16`` (a plain integer compare on the Q16.16
mirror -- the sweep writes no temperature). That tick it latches ``fired = 1``
and loads its fuse (``first_at_s``; 0 = no fuse), and the countdown below
runs from the SAME sweep, so a zero fuse detonates in the crossing tick. A
heat charge fires once, ever: ``period_s`` is refused at load. Not a reuse of
``combat.apply_temperature_ignition`` (vocabulary critique 7: that is a
per-material threshold over a per-tile armed plane that only lights fires).

Dormancy: a charge-free level builds an empty list, the 9e call is a single
``if self._timed_charges`` check, and no ENTITY_SECT record exists for a
charge that does not exist — so every charge-free digest is byte-identical.
"""
from __future__ import annotations

from config import ticks_from_seconds
from simulation.gas_fixed import quantize_q16_scalar
from simulation.payloads import execute_payload

# The ExplosionEvent flavour (the renderer draws every ExplosionEvent alike;
# kind only names the source).
EVENT_KIND = "timed_charge"

# The closed set of triggers (the door refuses anything else by name).
TRIGGER_TIMER = "timer"
TRIGGER_HEAT = "heat"
TRIGGERS = (TRIGGER_TIMER, TRIGGER_HEAT)


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
                 "ticks_to_fire", "alive", "heat", "trigger_temp_q16",
                 "fuse_ticks", "enabled", "fired")

    def __init__(self, inst, fy, fx, payload, first_ticks, period_ticks,
                 enabled, trigger_temp_q16=None):
        self.inst = inst
        self.fy = int(fy)
        self.fx = int(fx)
        self.payload = payload                 # the resolved PayloadDef
        self.period_ticks = int(period_ticks)  # 0 = fire once
        self.enabled = bool(enabled)
        # Heat trigger: None = a timer charge. A heat charge waits (-1) with
        # its fuse held aside until the latch closes.
        self.heat = trigger_temp_q16 is not None
        self.trigger_temp_q16 = (int(trigger_temp_q16) if self.heat else 0)
        self.fuse_ticks = int(first_ticks)
        self.fired = 0                         # the heat latch (synced row)
        if self.heat:
            self.ticks_to_fire = -1
        else:
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
        trigger = f.get("trigger", TRIGGER_TIMER)
        if trigger not in TRIGGERS:
            raise TimedChargeError(
                f"timed_charge '{e.id}': trigger {trigger!r} is not a "
                f"trigger (known: {list(TRIGGERS)})")
        period_s = float(f["period_s"])
        thr_q16 = None
        if trigger == TRIGGER_HEAT:
            t_game = float(f.get("trigger_temp", 0.0))
            if not t_game > 0.0:
                raise TimedChargeError(
                    f"timed_charge '{e.id}': trigger = \"heat\" needs "
                    f"trigger_temp > 0 (game degrees above ambient), got "
                    f"{t_game!r}")
            if period_s > 0:
                raise TimedChargeError(
                    f"timed_charge '{e.id}': a heat charge fires once -- "
                    f"period_s must be 0, got {period_s!r}")
            thr_q16 = quantize_q16_scalar(t_game)
            # The fuse: 0 s = the crossing tick itself (no ticks_from_
            # seconds floor of one tick -- that floor is for a timer's
            # "first", where 0 would mean "before the level exists").
            first_s = float(f["first_at_s"])
            first = ticks_from_seconds(first_s, tps) if first_s > 0 else 0
        else:
            first = ticks_from_seconds(float(f["first_at_s"]), tps)
        period = ticks_from_seconds(period_s, tps) if period_s > 0 else 0
        rt = TimedChargeRuntime(e, fy, fx, by_name[name], first, period,
                                bool(f["enabled"]), thr_q16)
        charges.append(rt)
        _replace_entity(sim.entities, e, rt)
    return charges


def sweep_timed_charges(sim) -> None:
    """Slot 9e: fire every charge whose countdown reached 0, in ordinal
    order, then advance the others one step. A waiting heat charge first
    READS its tile's Q16.16 temperature; at or above its threshold it
    latches and loads its fuse, which this same sweep then counts (a zero
    fuse fires now). Integer-only, no RNG drawn."""
    temperature = sim.gmap.temperature
    for c in sim._timed_charges:
        if c.heat and not c.fired:
            if not c.enabled:
                continue
            if int(temperature[c.fy, c.fx]) < c.trigger_temp_q16:
                continue
            c.fired = 1
            c.ticks_to_fire = c.fuse_ticks
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


__all__ = ["EVENT_KIND", "TRIGGERS", "TRIGGER_HEAT", "TRIGGER_TIMER",
           "TimedChargeError", "TimedChargeRuntime",
           "build_timed_charges", "sweep_timed_charges"]
