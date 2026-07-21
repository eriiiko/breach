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
        out_slot = bus.slot(ordinal, "out")
        drivers = input_slots.get((ordinal, "in"), [])
        cn = e.class_name
        if cn == "decider":
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
            evaluators.append(_GateEval(out_slot, drivers, INPUT_AND))
        elif cn == "gate_or":
            evaluators.append(_GateEval(out_slot, drivers, INPUT_HELD))
        elif cn == "gate_not":
            evaluators.append(_NotEval(out_slot, drivers))
        elif cn == "filter":
            k = snap_filter_k(e.fields["tau_s"], tps)
            frt = FilterRuntime(e, out_slot, drivers[0] if drivers else None, k)
            _replace_entity(sim.entities, e, frt)
            evaluators.append(frt)
        else:                             # pragma: no cover - registry drift
            raise ValueError(
                f"logic node class {cn!r} has no B2 evaluator — the registry "
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
