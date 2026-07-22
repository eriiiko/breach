"""Pump RUNTIME — the slot-9e step-(d) actuator sweep (Arc B patch B4).

Design: docs/arc_b_impl_2026-07-21.md §6 (v2, D10/D11). This is the sim-side
runtime of the ``pump`` N-feed actuator: at 9e(d), BEFORE the door structural
sweep, it resolves ``inject``/``extract`` from ``pub`` (via the shared
``aggregate_input`` helper, §2d), edits the PORT tile's gas mass through the new
GameMap integer primitive (``inject_gas_n``/``extract_gas_n`` — §6/D10), and
publishes ``at_target`` (a latched, anti-chatter Schmitt band, §6). The SCHEMA
(fields / signals / inputs + the ``PUMP`` marker) lives in the import-light
:mod:`simulation.entities.nodes`; everything that touches the gmap / the
SignalBus lives HERE, exactly as the door + node + sensor runtimes do.

The per-tick quantum ΔN and the D11 band-skip assert are computed ONCE at load
(:func:`build_pumps`), the door-2 rule (quantize-once, no float in the synced
path). Post-EOS the pump is a gas-MASS feed: at standard temperature the EOS
calibration ``p* = C·N·T`` gives ``C·T_amb == 1`` (config echoes the pinned
DEFAULT_C = 1/290, DEFAULT_T_AMB_K = 290), so a pressure quantum ΔP atm and its
gas-mass quantum ΔN coincide numerically — but they are derived separately so a
future non-unit C·T stays honest.

Determinism (§9): integer-only (Q16.16); ordinal-order sweep; ΔN quantized once
at load; no RNG, no float, no dequantize in the sweep (the port pressure and the
target are compared as raw Q16.16 ints). The edit lands at 9e(d) and reaches the
solver NEXT tick via the step-6 restamp (the 2-tick field-effect contract, §2c).
"""
from __future__ import annotations

from simulation.ambient import DEFAULT_C, DEFAULT_T_AMB_K
from simulation.entities import REGISTRY
from simulation.entities.schema import INPUT_HELD
from simulation.logic_nodes import aggregate_input


def is_pump(class_name) -> bool:
    """True iff ``class_name`` is a registered ``pump`` actuator (the §6 marker,
    parallel to ``is_logic_node`` / ``is_sensor``)."""
    cls = REGISTRY.get(class_name)
    return bool(cls is not None and getattr(cls, "PUMP", False))


class PumpRuntime:
    """Sim-side runtime object for one ``pump`` (§6).

    Doubles as the SERIALIZER runtime object (duck-typed
    ordinal/id/class_name/fields + ``alive``, mirroring
    :class:`simulation.logic_nodes.FilterRuntime` /
    :class:`simulation.door_system.DoorRuntime`) so the ``pump`` class's
    ``runtime_digest_rows`` reads ``self.at_target`` — the latched hysteresis
    state — straight off the sim's entity list (§8). Carries the per-tick edit
    (:meth:`sweep`).

    ``at_target`` is a LATCHED Schmitt band (anti-chatter, §6): it turns ON when
    the port pressure enters ``±band`` of ``target_atm`` and only turns OFF once
    the pressure leaves the wider ``±(2·band)`` release band — so a sweep past
    the setpoint gives ONE clean transition, never bang-bang toggling. The D11
    load assert (``ΔN_per_tick_atm < 2·band``) guarantees a single ΔN step
    cannot jump the full 2·band entry window, so the ON-latch is reachable.
    """

    __slots__ = ("inst", "at_target_slot", "inject_slots", "extract_slots",
                 "port_y", "port_x", "target_atm", "band", "release",
                 "delta_n_q", "at_target", "alive")

    def __init__(self, inst, at_target_slot, inject_slots, extract_slots,
                 port_y, port_x, target_atm, band, delta_n_q):
        self.inst = inst
        self.at_target_slot = (None if at_target_slot is None
                               else int(at_target_slot))
        self.inject_slots = list(inject_slots)
        self.extract_slots = list(extract_slots)
        self.port_y = int(port_y)
        self.port_x = int(port_x)
        self.target_atm = int(target_atm)
        self.band = int(band)
        self.release = 2 * int(band)      # the wider Schmitt release band
        self.delta_n_q = int(delta_n_q)
        self.at_target = 0                 # the latched synced state (§8 row)
        self.alive = True

    # --- serializer duck-type (serialize.py entity_records) ------------
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

    # --- actuator sweep (9e d) -----------------------------------------
    def sweep(self, sim) -> None:
        """9e(d): apply the held inject/extract edit at the port tile, then
        publish the latched ``at_target``. EXTRACT BEATS INJECT (the pinned
        both-active rule, §6 — the safe depressurize state). A dead pump
        (``alive == 0``) neither edits nor publishes a fresh at_target (its
        latch is frozen — fail-passive)."""
        if not self.alive:
            return
        bus = sim._signal_bus
        gmap = sim.gmap
        inject = aggregate_input(bus, self.inject_slots, INPUT_HELD) != 0
        extract = aggregate_input(bus, self.extract_slots, INPUT_HELD) != 0
        if extract:                        # extract beats inject (safe state)
            gmap.extract_gas_n(self.port_y, self.port_x, self.delta_n_q)
        elif inject:
            gmap.inject_gas_n(self.port_y, self.port_x, self.delta_n_q)

        # at_target: the port pressure vs target, latched Schmitt (§6). The
        # this-tick edit does not materialize until NEXT tick (the 2-tick field
        # contract), so `atmosphere` here is the pre-edit pressure — the honest
        # reading. Raw Q16.16 ints, no dequantize.
        p = int(gmap.atmosphere[self.port_y, self.port_x])
        err = p - self.target_atm
        if err < 0:
            err = -err
        if self.at_target:
            if err > self.release:
                self.at_target = 0
        else:
            if err <= self.band:
                self.at_target = 1
        if self.at_target_slot is not None:
            bus.set_pub(self.at_target_slot, self.at_target)


def _delta_n_quantum(rate_atm_per_s, tps):
    """The per-tick pressure quantum (Q16.16 atm) and gas-mass quantum (Q16.16),
    computed once at load (§6, door-2 rule). ``ΔP_atm = rate/tps``; ``ΔN_mass =
    ΔP_atm / (C·T_amb)`` — the two coincide at the calibrated C·T_amb == 1 but
    are derived separately for honesty. Returns ``(delta_p_atm_q, delta_n_q)``.
    """
    from simulation import atmosphere_fixed as _atm_fx
    from simulation import gas_fixed as _gas_fx
    delta_p_atm = float(rate_atm_per_s) / float(tps)       # atm per tick
    ct = float(DEFAULT_C) * float(DEFAULT_T_AMB_K)         # == 1.0 at defaults
    delta_n_mass = delta_p_atm / ct if ct > 0 else delta_p_atm
    return (_atm_fx.quantize_scalar(delta_p_atm),
            _gas_fx.quantize_scalar(delta_n_mass))


def build_pumps(sim) -> list:
    """Build the ordinal-ordered pump runtime list for the 9e(d) sweep and
    REPLACE each ``pump`` instance in ``sim.entities`` with its
    :class:`PumpRuntime` (so the latched ``at_target`` is serialized, §8).

    Computes each pump's per-tick ΔN quantum ONCE here and asserts the D11
    band-skip guard (``ΔN_per_tick_atm < 2·hysteresis_band``) — a hard
    ``ValueError`` at load if violated, since a step that jumps the full band
    would leave ``at_target`` unlatchable (airlock deadlock). The PORT tile is
    resolved from ``x``/``y`` + ``port_dx``/``port_dy`` at base resolution,
    scaled to the gmap grid by ``res_factor`` (the sensor/door pattern), and
    validated in-bounds. ``sim._signal_bus`` must exist (the caller gates on
    it); ``sim.entities`` is patched in place — the caller rebuilds
    ``_entity_by_ordinal`` from the patched list.
    """
    bus = sim._signal_bus
    gmap = sim.gmap
    h, w = gmap.solid.shape
    level = sim.level
    rf = int(getattr(level, "res_factor", 1) or 1)
    tps = sim._tps
    wires = getattr(level, "wires", None) or []

    # (target_ordinal, input) -> [pub slot, ...] in a pinned order (target,
    # input, source, signal) — commutative HELD aggregation, order immaterial to
    # the result but pinned for a deterministic build (the logic_nodes idiom).
    input_slots: dict = {}
    for wire in sorted(wires, key=lambda x: (int(x.target_ordinal), x.input,
                                             int(x.source_ordinal), x.signal)):
        if wire.input not in ("inject", "extract"):
            continue
        input_slots.setdefault((int(wire.target_ordinal), wire.input), []).append(
            bus.slot(wire.source_ordinal, wire.signal))

    pump_insts = sorted((e for e in sim.entities if is_pump(e.class_name)),
                        key=lambda e: int(e.ordinal))
    pumps: list = []
    for e in pump_insts:
        ordinal = int(e.ordinal)
        bx, by = int(e.fields["x"]), int(e.fields["y"])
        pdx, pdy = int(e.fields["port_dx"]), int(e.fields["port_dy"])
        port_y = rf * (by + pdy)
        port_x = rf * (bx + pdx)
        if not (0 <= port_y < h and 0 <= port_x < w):
            raise ValueError(
                f"pump '{e.id}': port tile ({port_y}, {port_x}) is out of the "
                f"{h}x{w} grid — check x/y + port_dx/port_dy (base tiles, "
                f"scaled by res_factor {rf})")
        band = int(e.fields["hysteresis_band"])
        target_atm = int(e.fields["target_atm"])
        delta_p_atm_q, delta_n_q = _delta_n_quantum(e.fields["rate"], tps)
        # D11: a per-tick pressure step that spans the FULL 2·band entry window
        # can jump the port tile from below-band to above-band, so at_target
        # never latches -> the airlock's EQUALIZE/REPRESSURIZE waits forever.
        if delta_p_atm_q >= 2 * band:
            raise ValueError(
                f"pump '{e.id}': ΔN_per_tick == {delta_p_atm_q} Q16.16 atm "
                f"(rate {e.fields['rate']} atm/s / {tps} tps) is >= 2·band "
                f"({2 * band}) — the port tile can jump the hysteresis band in "
                f"one tick and at_target never latches (airlock deadlock, D11). "
                f"Lower `rate` or widen `hysteresis_band`.")
        at_slot = (bus.slot(ordinal, "at_target")
                   if bus.has(ordinal, "at_target") else None)
        rt = PumpRuntime(
            e, at_slot,
            input_slots.get((ordinal, "inject"), []),
            input_slots.get((ordinal, "extract"), []),
            port_y, port_x, target_atm, band, delta_n_q)
        _replace_entity(sim.entities, e, rt)
        pumps.append(rt)
    return pumps


def _replace_entity(entities, old, new) -> None:
    """Swap ``old`` for its runtime wrapper ``new`` in the sim entity list,
    preserving ordinal position (identity match — one object per instance).
    Mirrors :func:`simulation.logic_nodes._replace_entity`."""
    for i, e in enumerate(entities):
        if e is old:
            entities[i] = new
            return
    raise ValueError(                     # pragma: no cover - defensive
        f"pump {getattr(old, 'id', old)!r} not found in the sim entity list")


def sweep_pumps(sim) -> None:
    """9e(d): sweep every pump in ORDINAL order — apply the port-tile gas edit
    and publish ``at_target`` (§2b step d, BEFORE the door structural sweep)."""
    for p in sim._pumps:
        p.sweep(sim)
