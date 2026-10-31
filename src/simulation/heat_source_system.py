"""Heat-source RUNTIME -- the slot-9e "hot every tick" sweep (levelgen
vocabulary E.2 row 3, docs/levelgen_zone_vocabulary_2026-10-29.md; Erik's
ruling 2026-10-31 accepts the new 9e tick logic against RL-batch habit §A).

The sim-side runtime of :class:`simulation.entities.heat_source.heat_source`
(the schema stays import-light there; everything touching the gmap, the
payload table or the clock lives HERE -- the charge.py /
timed_charge_system.py split).

DEPOSIT. Each tick every living, enabled source hands its per-tick heat to
THE heat deposit path, :func:`simulation.payloads.deposit_heat` -- one
``heat`` DISC ADD FieldEdit (LINEAR falloff, no skip-mask: heat lands on
solids too, as a plasma splash's does), which the next step's 6b flush
writes into the ``heat`` ingress plane with a Q16.16 saturating add and the
temperature solver converts and BOOKS in its own group (``e_gas_deposit_sum``
/ ``e_solid_deposit_sum``) -- so the energy closure identity is untouched:
no new channel, no ``temperature[...] =``. ``heat_per_s`` is divided by the
tick rate and quantized ONCE at build (:func:`gas_fixed.quantize_q16_scalar`)
and handed to the edit as its exact dequantized value, so the centre tile
receives exactly that Q16 count every tick. The disc's ``heat_radius`` is
``radius + 1``: the FieldEdit DISC is strict (``dist < r``), so radius 0
reaches exactly the source tile at full weight.

MAX_TEMP. A source with ``max_temp > 0`` READS its own tile's Q16.16
temperature mirror (a plain integer compare, the heat-trigger precedent --
the sweep writes no temperature) and skips this tick's deposit at or above
the cap. The deposit enqueued at 9e lands at the next step, after which the
solver may still raise the tile, so the tile overshoots the cap by at most
one deposit (plus what neighbours conduct in). A thermostat-free way to hold
an engine room at a working temperature: the deleted Pass-3 thermostat
relaxed toward ambient; this only withholds its own heat.

DEATH (the blast hook). Entities have no hp path in the damage system
(``damage.py`` targets units, ``apply_explosion`` walls), so a source reads
the BLAST LOG: ``payloads.execute_payload`` appends every explosion disc
``(fy, fx, radius, wall_damage)`` to ``gmap.blast_log`` (every detonation in
the game -- grenade, shell, planted or door charge, timed charge -- goes
through it), and this sweep drains it. A disc covering the source's tile
subtracts ``wall_damage * (1 - dist/radius)`` (``dist <= radius``),
quantized through ``wall_fixed`` -- the wall law of ``apply_explosion``
exactly, in the ``wall_hp`` currency. At ``hp <= 0`` the source dies:
``alive = False``, ``enabled = False``, ``hp = 0``, and its
``death_payload`` row (if any) is executed ONCE through THE payload
executor; a death payload's own blast joins the discs still to be read in
this sweep, so a chain of engines goes off in one tick, deterministically
(discs in log order, sources in ordinal order per disc).

ORDER IN 9e. LAST of the actuator sub-slot, after the timed-charge sweep:
every blast of the tick (slot 2 grenades and shells, slot 8 charges, 9e
timed charges) is in the log when the sources read it, so a source killed
this tick deposits nothing this tick and its death payload's edits land at
the next 6b flush (the timed charge's one-tick latency).

Dormancy: a source-free level builds an empty list; the 9e call is a single
``if self._heat_sources`` check (else the log is cleared unread); no
ENTITY_SECT record exists for a source that does not exist -- so every
source-free digest is byte-identical. Integer state, no RNG drawn.
"""
from __future__ import annotations

import math

from simulation import wall_fixed
from simulation.entities.heat_source import MAX_RADIUS
from simulation.gas_fixed import dequantize_q16, quantize_q16_scalar
from simulation.payloads import deposit_heat, execute_payload

# The ExplosionEvent flavour of a death payload.
EVENT_KIND = "heat_source"


class HeatSourceError(ValueError):
    """A heat_source row that cannot be built (unknown payload, bad heat)."""


class HeatSourceRuntime:
    """Sim-side runtime object for one ``heat_source``.

    Doubles as the SERIALIZER runtime object (duck-typed ordinal / id /
    class_name / fields + ``alive``), so the class's ``runtime_digest_rows``
    reads ``hp_q16`` straight off the sim's entity list.
    """

    __slots__ = ("inst", "fy", "fx", "heat_q16", "heat_radius", "enabled",
                 "hp_q16", "death_payload", "max_temp_q16", "alive")

    def __init__(self, inst, fy, fx, heat_q16, radius, enabled, hp_q16,
                 death_payload, max_temp_q16):
        self.inst = inst
        self.fy = int(fy)
        self.fx = int(fx)
        self.heat_q16 = int(heat_q16)          # per-tick centre deposit
        self.heat_radius = int(radius) + 1     # the strict DISC's radius
        self.enabled = bool(enabled)
        self.hp_q16 = int(hp_q16)
        self.death_payload = death_payload     # PayloadDef or None
        self.max_temp_q16 = int(max_temp_q16)  # 0 = no cap
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


def build_heat_sources(sim) -> list:
    """Build the ordinal-ordered runtime list and REPLACE each
    ``heat_source`` instance in ``sim.entities`` with its runtime wrapper.

    Must run after ``sim.weapons_tables`` exists: ``death_payload`` is
    validated against the PayloadTable HERE (load time). ``x``/``y`` are
    base-resolution tiles scaled by ``res_factor`` (the timed-charge
    pattern) and bounds-checked hard; ``radius`` is in runtime tiles (the
    [payloads.*] heat_radius convention). Refused by name: an unknown
    payload, ``heat_per_s <= 0`` or below one Q16 count per tick, a radius
    outside [0, MAX_RADIUS], ``hp < 1``, a negative ``max_temp``.
    """
    insts = sorted((e for e in sim.entities if e.class_name == "heat_source"),
                   key=lambda e: int(e.ordinal))
    if not insts:
        return []
    rf = int(getattr(sim.level, "res_factor", 1) or 1)
    tps = int(sim._tps)
    h, w = sim.gmap.solid.shape
    by_name = sim.weapons_tables.payloads.by_name
    sources = []
    for e in insts:
        f = e.fields
        name = f.get("death_payload", "") or ""
        if name and name not in by_name:
            raise HeatSourceError(
                f"heat_source '{e.id}': death_payload {name!r} names no "
                f"[payloads.*] row in config.toml (known: {sorted(by_name)})")
        fy, fx = rf * int(f["y"]), rf * int(f["x"])
        if not (0 <= fy < h and 0 <= fx < w):
            raise HeatSourceError(
                f"heat_source '{e.id}': tile ({fy}, {fx}) is out of the "
                f"{h}x{w} grid (base tiles x={f['x']}, y={f['y']}, scaled "
                f"by res_factor {rf})")
        heat_per_s = float(f["heat_per_s"])
        if not heat_per_s > 0.0:
            raise HeatSourceError(
                f"heat_source '{e.id}': heat_per_s must be > 0 (use "
                f"enabled = false for a cold source), got {heat_per_s!r}")
        heat_q16 = quantize_q16_scalar(heat_per_s / tps)
        if heat_q16 < 1:
            raise HeatSourceError(
                f"heat_source '{e.id}': heat_per_s {heat_per_s!r} is below "
                f"one Q16.16 count per tick at {tps} ticks/s")
        radius = int(f.get("radius", 0))
        if not 0 <= radius <= MAX_RADIUS:
            raise HeatSourceError(
                f"heat_source '{e.id}': radius {radius!r} is outside "
                f"[0, {MAX_RADIUS}]")
        hp = int(f.get("hp", 200))
        if hp < 1:
            raise HeatSourceError(
                f"heat_source '{e.id}': hp must be >= 1, got {hp!r}")
        max_temp = float(f.get("max_temp", 0.0))
        if max_temp < 0.0:
            raise HeatSourceError(
                f"heat_source '{e.id}': max_temp must be >= 0 (0 = no "
                f"cap), got {max_temp!r}")
        rt = HeatSourceRuntime(
            e, fy, fx, heat_q16, radius, bool(f.get("enabled", True)),
            wall_fixed.quantize_scalar(float(hp)),
            by_name[name] if name else None,
            quantize_q16_scalar(max_temp) if max_temp > 0.0 else 0)
        sources.append(rt)
        _replace_entity(sim.entities, e, rt)
    return sources


def _blast_damage_q16(src, fy, fx, radius, wall_damage) -> int:
    """The wall law of ``apply_explosion`` at the source's tile: 0 outside
    ``dist <= radius``, else ``wall_damage * (1 - dist/radius)`` quantized
    once into the wall_hp currency."""
    if radius <= 0:
        return 0
    dy, dx = src.fy - fy, src.fx - fx
    dist = math.sqrt(dy * dy + dx * dx)
    if dist > radius:
        return 0
    falloff = 1.0 - (dist / radius)
    return wall_fixed.quantize_scalar(float(wall_damage) * float(falloff))


def _die(sim, src) -> None:
    """hp reached 0: the source is dead, cold, and executes its death
    payload once (which carries the heat that starts the fire)."""
    src.alive = False
    src.enabled = False
    src.hp_q16 = 0
    if src.death_payload is not None:
        execute_payload(sim.gmap, sim.edit_queue, sim.units, src.fy, src.fx,
                        src.death_payload, sim.rng, events=sim.tick_events,
                        kind=EVENT_KIND)


def sweep_heat_sources(sim) -> None:
    """Slot 9e, after the timed charges: drain the blast log into the
    sources (deaths and death payloads, chained), then every living,
    enabled, uncapped source enqueues its per-tick heat. Integer state,
    ordinal order, no RNG drawn."""
    gmap = sim.gmap
    log = gmap.blast_log
    discs = list(log)
    log.clear()
    sources = sim._heat_sources
    i = 0
    while i < len(discs):                 # a death payload extends `discs`
        fy, fx, radius, wall_damage = discs[i]
        i += 1
        for s in sources:
            if not s.alive:
                continue
            dmg = _blast_damage_q16(s, fy, fx, radius, wall_damage)
            if dmg <= 0:
                continue
            s.hp_q16 -= dmg
            if s.hp_q16 <= 0:
                _die(sim, s)
                discs.extend(log)         # its own blast, read this sweep
                log.clear()
    temperature = gmap.temperature
    for s in sources:
        if not (s.alive and s.enabled):
            continue
        if s.max_temp_q16 and int(temperature[s.fy, s.fx]) >= s.max_temp_q16:
            continue
        deposit_heat(gmap, sim.edit_queue, s.fy, s.fx,
                     float(dequantize_q16(s.heat_q16)), s.heat_radius)


def _replace_entity(entities, old, new) -> None:
    """Swap ``old`` for its runtime wrapper, preserving ordinal position --
    mirrors :func:`simulation.timed_charge_system._replace_entity`."""
    for i, e in enumerate(entities):
        if e is old:
            entities[i] = new
            return
    raise ValueError(                     # pragma: no cover - defensive
        f"heat_source {getattr(old, 'id', old)!r} not found in the sim "
        f"entity list")


__all__ = ["EVENT_KIND", "HeatSourceError", "HeatSourceRuntime",
           "build_heat_sources", "sweep_heat_sources"]
