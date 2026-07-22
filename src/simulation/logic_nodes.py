"""Logic-node RUNTIME — the slot-9e step-(b) sweep (Arc B patch B2).

Design: docs/arc_b_impl_2026-07-21.md §2b/§2d/§5 (v2). This is the sim-side,
gmap-FREE evaluation of the L0 logic nodes: pure SignalBus I/O (§4). The
import-light ``simulation.entities`` package holds the node SCHEMA + the exact
``k``-snap; everything that touches the SignalBus / the sim runtime list lives
HERE, exactly as the door sweep lives in :mod:`simulation.door_system`.

THE order-independence invariant (§2b, the proof): every node reads ``pub``
(this tick's sensor / alive / is_open values + LAST tick's node outputs) and
writes ``stg`` — a node NEVER reads ``stg``, so it cannot observe another
node's THIS-tick output. The 9e(b) sweep therefore gives the same result in any
order; we still sweep in ORDINAL order (determinism ledger §9). Node outputs are
swapped ``pub <- stg`` at 9e(e) → one tick per hop (§2c).

Determinism (§9): integer-only (Q16.16); ordinal-order sweep; no RNG, no float
(the filter ``k`` is snapped by exact Fraction arithmetic at load, in
:func:`simulation.entities.nodes.snap_filter_k`); no dict-order dependence
(driver slot lists are built in a pinned order and aggregation is commutative).
"""
from __future__ import annotations

from simulation.entities import REGISTRY
from simulation.entities.actuators import (
    AIRLOCK_CLOSING, AIRLOCK_EQUALIZE, AIRLOCK_FAULT, AIRLOCK_IDLE,
    AIRLOCK_OPEN_FAR, AIRLOCK_REPRESSURIZE, AIRLOCK_RESEAL,
)
from simulation.entities.nodes import snap_filter_k
from simulation.entities.schema import (
    INPUT_AND, INPUT_EDGE, INPUT_HELD, INPUT_SINGLE,
)

# The comparator token -> integer predicate (decider, §5). Both operands are
# plain Python ints (Q16.16), so these are exact integer comparisons.
_COMPARATORS = {
    "gt": lambda a, b: a > b,
    "ge": lambda a, b: a >= b,
    "lt": lambda a, b: a < b,
    "le": lambda a, b: a <= b,
    "eq": lambda a, b: a == b,
    "ne": lambda a, b: a != b,
}


# ---------------------------------------------------------------------------
# THE shared input-aggregation helper (§2d) — one rule for the door input
# resolve (9e c) and the node sweep (9e b).
# ---------------------------------------------------------------------------

def aggregate_input(bus, driver_slots, mode) -> int:
    """Combine an input's driving wires from ``pub`` per its aggregation mode
    (§2d). ``driver_slots`` is the (possibly empty) list of dense ``pub`` slot
    indices feeding one input. Reads ``pub`` ONLY (never ``stg`` — the
    order-independence invariant).

    - ``INPUT_HELD`` (OR): 1 if ANY driver != 0, else 0.
    - ``INPUT_AND``:       1 iff EVERY driver != 0; empty => 0.
    - ``INPUT_SINGLE``:    the one driver's integer value verbatim; empty => 0
      (arity 1 is guaranteed by the §1b loader check).
    - ``INPUT_EDGE``:      reserved (§2d) — no B2 node uses it; needs a synced
      prev row, deferred to a later patch.
    """
    pub = bus.pub
    if mode == INPUT_SINGLE:
        return int(pub[driver_slots[0]]) if driver_slots else 0
    if mode == INPUT_AND:
        if not driver_slots:
            return 0
        return 1 if all(int(pub[i]) != 0 for i in driver_slots) else 0
    if mode == INPUT_HELD:
        return 1 if any(int(pub[i]) != 0 for i in driver_slots) else 0
    if mode == INPUT_EDGE:
        raise NotImplementedError(
            "INPUT_EDGE aggregation is reserved (Arc B §2d) — no B2 node uses "
            "it; it needs a synced prev row (a later patch).")
    raise ValueError(f"unknown aggregation mode {mode!r}")


# ---------------------------------------------------------------------------
# Node evaluators — one per node instance, prev-read (pub) / next-write (stg).
# Stateless nodes (decider / gates) are plain evaluators; the `filter` carries
# runtime EMA state and doubles as the serializer runtime object (below).
# ---------------------------------------------------------------------------

class _DeciderEval:
    """``out`` = cmp(in, threshold) AND (require_alive => source alive)."""

    __slots__ = ("out_slot", "in_slots", "predicate", "threshold",
                 "require_alive", "alive_slot")

    def __init__(self, out_slot, in_slots, comparator, threshold,
                 require_alive, alive_slot):
        self.out_slot = int(out_slot)
        self.in_slots = list(in_slots)
        self.predicate = _COMPARATORS[comparator]
        self.threshold = int(threshold)
        self.require_alive = bool(require_alive)
        # The source entity's `alive` pub slot (present iff require_alive AND
        # the `in` wire is connected — build_signal_bus adds it, D4). None when
        # require_alive is off or the input is unwired.
        self.alive_slot = None if alive_slot is None else int(alive_slot)

    def evaluate(self, bus) -> None:
        val = aggregate_input(bus, self.in_slots, INPUT_SINGLE)
        res = 1 if self.predicate(val, self.threshold) else 0
        if self.require_alive:
            # Fail-passive (§2d): a dead / disconnected source zeroes the
            # decider this tick. pub[alive] is current-tick (emitted at 9e a).
            if self.alive_slot is None or int(bus.pub[self.alive_slot]) == 0:
                res = 0
        bus.stg[self.out_slot] = res


class _GateEval:
    """``out`` (0/1) = the AND / OR reduction of the driving wires (§5)."""

    __slots__ = ("out_slot", "in_slots", "mode")

    def __init__(self, out_slot, in_slots, mode):
        self.out_slot = int(out_slot)
        self.in_slots = list(in_slots)
        self.mode = mode                  # INPUT_AND (gate_and) | INPUT_HELD (or)

    def evaluate(self, bus) -> None:
        bus.stg[self.out_slot] = aggregate_input(bus, self.in_slots, self.mode)


class _NotEval:
    """``out`` = 1 - (in != 0) — logical negation of a single input (§5)."""

    __slots__ = ("out_slot", "in_slots")

    def __init__(self, out_slot, in_slots):
        self.out_slot = int(out_slot)
        self.in_slots = list(in_slots)

    def evaluate(self, bus) -> None:
        val = aggregate_input(bus, self.in_slots, INPUT_SINGLE)
        bus.stg[self.out_slot] = 0 if val != 0 else 1


class FilterRuntime:
    """Sim-side runtime object for one ``filter`` node (§5, D5).

    Doubles as the SERIALIZER runtime object (the duck-type
    ordinal/id/class_name/fields + ``alive``, mirroring
    :class:`simulation.door_system.DoorRuntime`) so the ``filter`` class's
    ``runtime_digest_rows`` reads ``self.ema`` straight off the sim's entity
    list. It also carries the node evaluator (:meth:`evaluate`).

    Integer EMA with ``k`` guard bits (§5): the accumulator ``ema`` holds the
    running average scaled by ``2^k`` (``k`` FRACTIONAL guard bits below the
    Q16.16 value). Each tick it moves toward ``in`` and rounds-to-nearest
    BEFORE the shift, so a sub-LSB difference accumulates in the guard bits
    instead of truncating to zero and parking (the design's truncation-park
    fix). ``k == 0`` degenerates to ``out == in`` (no smoothing).
    """

    __slots__ = ("inst", "out_slot", "in_slot", "k", "_half", "ema", "alive")

    def __init__(self, inst, out_slot, in_slot, k):
        self.inst = inst
        self.out_slot = int(out_slot)
        # `in` is SINGLE (arity 1); None when the input is unwired => reads 0.
        self.in_slot = None if in_slot is None else int(in_slot)
        self.k = int(k)
        self._half = (1 << (self.k - 1)) if self.k >= 1 else 0
        self.ema = 0                      # the guard-bit accumulator (synced row)
        self.alive = True

    # --- serializer duck-type (serialize.py:121-132) -------------------
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

    # --- node evaluator (9e b) -----------------------------------------
    def evaluate(self, bus) -> None:
        x = int(bus.pub[self.in_slot]) if self.in_slot is not None else 0
        # Round-to-nearest current value, then accumulate the residual toward x
        # in the guard bits (no truncation park). k == 0 => value == ema == x.
        value = (self.ema + self._half) >> self.k
        self.ema += x - value
        bus.stg[self.out_slot] = (self.ema + self._half) >> self.k


class AirlockControllerRuntime:
    """Sim-side runtime for one ``airlock_controller`` state machine (§7, D12).

    A LOGIC-NODE evaluator (swept at 9e(b)): it reads its wired inputs from
    ``pub`` (the presence plate value + each door's is_open/alive + the pump's
    at_target), advances the state machine, and writes its command signals to
    ``stg`` — swapped into ``pub`` at 9e(e), so each command reaches its door /
    pump ONE tick later (the node-hop latency, §2c). It touches NO entity
    directly — every read and write is a SignalBus slot — so its runtime list
    position is never observable (§9).

    Doubles as the SERIALIZER runtime object (the ordinal/id/class_name/fields
    + ``alive`` duck-type, mirroring :class:`FilterRuntime` /
    :class:`simulation.pump_system.PumpRuntime`) so the ``airlock_controller``
    class's ``runtime_digest_rows`` reads ``self.state`` / ``self.dwell``
    straight off the sim's entity list (§8). ``state`` is the integer enum
    (:mod:`simulation.entities.actuators`); ``dwell`` counts ticks held in the
    current state (0 on the tick a transition lands). Both are synced rows.

    The transition function is a Moore machine (§7): read inputs → compute the
    next state → drive the command outputs from the (new) state. Integer-only,
    no RNG / float / dict-order; the outputs are a pure function of the state,
    so the sweep is order-independent (nodes never read ``stg``, the invariant).

    Bidirectional note (B7 — Erik's Option 2): ``equalize_cmd`` is the pump
    drive direction for EQUALIZE (+1 inject toward a higher far target, -1
    extract toward a lower one, 0 when the two targets coincide); REPRESSURIZE
    drives the opposite. The two legs gate on INDEPENDENT sensed inputs — a
    chamber pressure sensor through two deciders: EQUALIZE waits on ``at_far``
    (chamber ≤ far target), REPRESSURIZE waits on ``at_near`` (chamber ≥ near
    target). The pump runs open-loop; the sensor closes the loop, so
    REPRESSURIZE genuinely refills (the B5 single-pump gap is gone).
    """

    __slots__ = ("inst", "far_is_outer", "equalize_cmd",
                 "i_presence", "i_inner_open", "i_outer_open",
                 "i_inner_alive", "i_outer_alive", "i_at_far", "i_at_near",
                 "o_inner_close", "o_inner_open", "o_outer_close",
                 "o_outer_open", "o_inject", "o_extract", "o_busy",
                 "state", "dwell", "alive")

    def __init__(self, inst, far_is_outer, equalize_cmd, in_slots, out_slots):
        self.inst = inst
        self.far_is_outer = bool(far_is_outer)
        self.equalize_cmd = int(equalize_cmd)   # +1 inject / -1 extract / 0
        # Input slots (None => unwired => reads 0). Named to avoid colliding
        # with the same-named command OUTPUT (inner_open vs inner_open_cmd).
        self.i_presence = in_slots["presence"]
        self.i_inner_open = in_slots["inner_open"]
        self.i_outer_open = in_slots["outer_open"]
        self.i_inner_alive = in_slots["inner_alive"]
        self.i_outer_alive = in_slots["outer_alive"]
        self.i_at_far = in_slots["at_far"]     # chamber ≤ far target (decider)
        self.i_at_near = in_slots["at_near"]   # chamber ≥ near target (decider)
        # Output slots (all present — a LOGIC_NODE's SIGNALS all get a slot).
        self.o_inner_close = int(out_slots["inner_close"])
        self.o_inner_open = int(out_slots["inner_open_cmd"])
        self.o_outer_close = int(out_slots["outer_close"])
        self.o_outer_open = int(out_slots["outer_open_cmd"])
        self.o_inject = int(out_slots["pump_inject"])
        self.o_extract = int(out_slots["pump_extract"])
        self.o_busy = int(out_slots["busy"])
        self.state = AIRLOCK_IDLE          # synced runtime row (§8)
        self.dwell = 0                     # ticks in the current state (synced)
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

    # --- the transition function (9e b) --------------------------------
    @staticmethod
    def _pump_dir(cmd):
        """(inject, extract) for a signed pump command (+1 / -1 / 0)."""
        if cmd > 0:
            return 1, 0
        if cmd < 0:
            return 0, 1
        return 0, 0

    def _next_state(self, presence, inner_open, outer_open, inner_alive,
                    outer_alive, at_far, at_near, far_open, far_alive):
        """The §7 transition: the state to hold this tick. Integer-only.

        B7: EQUALIZE gates on ``at_far`` and REPRESSURIZE on ``at_near`` — two
        independent chamber-pressure-sensed inputs (Erik's Option 2)."""
        s = self.state
        if s == AIRLOCK_IDLE:
            return AIRLOCK_CLOSING if presence >= 1 else AIRLOCK_IDLE
        if s == AIRLOCK_CLOSING:
            # D12 breach-abort: a door reading alive==0 is a venting HOLE, not a
            # seal (a destroyed door also reads is_open==0). Never pump into it.
            if inner_alive == 0 or outer_alive == 0:
                return AIRLOCK_FAULT
            if inner_open == 0 and outer_open == 0:
                return AIRLOCK_EQUALIZE
            return AIRLOCK_CLOSING           # occupancy stall lives here (§7)
        if s == AIRLOCK_EQUALIZE:
            return AIRLOCK_OPEN_FAR if at_far != 0 else AIRLOCK_EQUALIZE
        if s == AIRLOCK_OPEN_FAR:
            return AIRLOCK_RESEAL if presence == 0 else AIRLOCK_OPEN_FAR
        if s == AIRLOCK_RESEAL:
            # D12 again: a far door that reads alive==0 mid-reseal is a breach —
            # abort before REPRESSURIZE would pump into it.
            if far_alive == 0:
                return AIRLOCK_FAULT
            return AIRLOCK_REPRESSURIZE if far_open == 0 else AIRLOCK_RESEAL
        if s == AIRLOCK_REPRESSURIZE:
            return AIRLOCK_IDLE if at_near != 0 else AIRLOCK_REPRESSURIZE
        return AIRLOCK_FAULT                 # FAULT is latched until reset (§7)

    def evaluate(self, bus) -> None:
        pub = bus.pub

        def rd(slot):
            return int(pub[slot]) if slot is not None else 0

        presence = rd(self.i_presence)
        inner_open = rd(self.i_inner_open)
        outer_open = rd(self.i_outer_open)
        inner_alive = rd(self.i_inner_alive)
        outer_alive = rd(self.i_outer_alive)
        at_far = rd(self.i_at_far)
        at_near = rd(self.i_at_near)
        # Far / near mapping (which physical door is the far side).
        if self.far_is_outer:
            far_open, far_alive = outer_open, outer_alive
        else:
            far_open, far_alive = inner_open, inner_alive

        prev = self.state
        nxt = self._next_state(presence, inner_open, outer_open, inner_alive,
                               outer_alive, at_far, at_near, far_open, far_alive)
        self.dwell = self.dwell + 1 if nxt == prev else 0
        self.state = nxt

        # Moore outputs: a pure function of the (new) state. far/near are the
        # ABSTRACT roles; mapped to inner/outer by far_door below.
        far_open_cmd = far_close = near_open_cmd = near_close = 0
        inject = extract = busy = 0
        st = self.state
        if st == AIRLOCK_IDLE:
            near_open_cmd = 1               # near door open for entry
            far_close = 1                  # far door held shut
        elif st == AIRLOCK_CLOSING:
            near_close = far_close = busy = 1
        elif st == AIRLOCK_EQUALIZE:
            near_close = far_close = busy = 1
            inject, extract = self._pump_dir(self.equalize_cmd)
        elif st == AIRLOCK_OPEN_FAR:
            near_close = far_open_cmd = busy = 1
        elif st == AIRLOCK_RESEAL:
            near_close = far_close = busy = 1
        elif st == AIRLOCK_REPRESSURIZE:
            near_close = far_close = busy = 1
            inject, extract = self._pump_dir(-self.equalize_cmd)
        # AIRLOCK_FAULT: everything 0 — doors released to manual, busy 0 (§7).

        # Map the far/near roles onto the physical inner/outer command slots.
        if self.far_is_outer:
            inner_open_o, inner_close_o = near_open_cmd, near_close
            outer_open_o, outer_close_o = far_open_cmd, far_close
        else:
            inner_open_o, inner_close_o = far_open_cmd, far_close
            outer_open_o, outer_close_o = near_open_cmd, near_close

        stg = bus.stg
        stg[self.o_inner_close] = inner_close_o
        stg[self.o_inner_open] = inner_open_o
        stg[self.o_outer_close] = outer_close_o
        stg[self.o_outer_open] = outer_open_o
        stg[self.o_inject] = inject
        stg[self.o_extract] = extract
        stg[self.o_busy] = busy


# ---------------------------------------------------------------------------
# Build + sweep — wired into simulation.py's slot-9e block.
# ---------------------------------------------------------------------------

def is_logic_node(class_name) -> bool:
    """True iff ``class_name`` is a registered LOGIC-NODE class (§2a marker)."""
    cls = REGISTRY.get(class_name)
    return bool(cls is not None and getattr(cls, "LOGIC_NODE", False))


def build_logic_nodes(sim) -> list:
    """Build the ordinal-ordered node evaluator list for the slot-9e(b) sweep
    (§2b) and REPLACE each ``filter`` instance in ``sim.entities`` with its
    :class:`FilterRuntime` (so the EMA accumulator is serialized, §5/§8).

    Precomputes each node input's driving ``pub`` slot indices from the
    resolved ``sim.level.wires``, in a pinned order (target, input, source,
    signal) — commutative aggregation makes the order immaterial to the result,
    but pinning it keeps the build deterministic. ``sim._signal_bus`` must
    exist (the caller gates on it). ``sim.entities`` is patched in place; the
    caller rebuilds ``_entity_by_ordinal`` from the patched list.
    """
    bus = sim._signal_bus
    wires = getattr(sim.level, "wires", None) or []

    # (target_ordinal, input) -> [pub slot index, ...] in pinned order.
    input_slots: dict = {}
    # (target_ordinal, input) -> source_ordinal (for require_alive; SINGLE => 1).
    in_source: dict = {}
    for w in sorted(wires, key=lambda w: (int(w.target_ordinal), w.input,
                                          int(w.source_ordinal), w.signal)):
        key = (int(w.target_ordinal), w.input)
        input_slots.setdefault(key, []).append(
            bus.slot(w.source_ordinal, w.signal))
        in_source[key] = int(w.source_ordinal)

    tps = sim._tps
    node_insts = sorted((e for e in sim.entities if is_logic_node(e.class_name)),
                        key=lambda e: int(e.ordinal))
    evaluators: list = []
    for e in node_insts:
        ordinal = int(e.ordinal)
        drivers = input_slots.get((ordinal, "in"), [])
        cn = e.class_name
        # Single-output nodes (decider / gate_* / filter) emit exactly `out`;
        # the airlock_controller emits SEVERAL command signals (§7), so `out`
        # is resolved per-branch — a node with no `out` never touches it.
        if cn == "decider":
            out_slot = bus.slot(ordinal, "out")
            require_alive = bool(e.fields.get("require_alive"))
            alive_slot = None
            if require_alive:
                src = in_source.get((ordinal, "in"))
                if src is not None:
                    alive_slot = bus.slot(src, "alive")
            evaluators.append(_DeciderEval(
                out_slot, drivers, e.fields["comparator"],
                int(e.fields["threshold"]), require_alive, alive_slot))
        elif cn == "gate_and":
            evaluators.append(_GateEval(bus.slot(ordinal, "out"), drivers,
                                        INPUT_AND))
        elif cn == "gate_or":
            evaluators.append(_GateEval(bus.slot(ordinal, "out"), drivers,
                                        INPUT_HELD))
        elif cn == "gate_not":
            evaluators.append(_NotEval(bus.slot(ordinal, "out"), drivers))
        elif cn == "filter":
            k = snap_filter_k(e.fields["tau_s"], tps)
            frt = FilterRuntime(e, bus.slot(ordinal, "out"),
                                drivers[0] if drivers else None, k)
            _replace_entity(sim.entities, e, frt)
            evaluators.append(frt)
        elif cn == "airlock_controller":
            # A MULTI-OUTPUT node (§7): one SINGLE input each, seven command
            # signals. Resolve every declared input's driving slot (SINGLE →
            # the one slot or None) and every output signal's slot.
            def _single(name, _ord=ordinal):
                lst = input_slots.get((_ord, name), [])
                return lst[0] if lst else None
            in_slots = {n: _single(n) for n in (
                "presence", "inner_open", "outer_open", "inner_alive",
                "outer_alive", "at_far", "at_near")}
            out_slots = {sig.name: bus.slot(ordinal, sig.name)
                         for sig in REGISTRY[cn].SIGNALS}
            far = int(e.fields["target_far_atm"])
            near = int(e.fields["target_near_atm"])
            equalize_cmd = 1 if far > near else (-1 if far < near else 0)
            far_is_outer = (e.fields["far_door"] == "outer")
            art = AirlockControllerRuntime(
                e, far_is_outer, equalize_cmd, in_slots, out_slots)
            _replace_entity(sim.entities, e, art)
            evaluators.append(art)
        else:                             # pragma: no cover - registry drift
            raise ValueError(
                f"logic node class {cn!r} has no B-arc evaluator — the registry "
                f"and logic_nodes.build_logic_nodes disagree")
    return evaluators


def _replace_entity(entities, old, new) -> None:
    """Swap ``old`` for its runtime wrapper ``new`` in the sim entity list,
    preserving ordinal position (identity match — the list carries one object
    per instance)."""
    for i, e in enumerate(entities):
        if e is old:
            entities[i] = new
            return
    raise ValueError(                     # pragma: no cover - defensive
        f"entity {getattr(old, 'id', old)!r} not found in the sim entity list")


def sweep_logic_nodes(sim) -> None:
    """9e(b): evaluate every node in ORDINAL order — read ``pub``, write
    ``stg[out]`` (§2b). Order-independent by construction (nodes never read
    ``stg``); ordinal order is pinned for the determinism ledger."""
    bus = sim._signal_bus
    for node in sim._logic_nodes:
        node.evaluate(bus)
