"""The SignalBus — Arc B's integer dataflow substrate (impl doc §2).

Design: docs/arc_b_impl_2026-07-21.md §2 (v2, 3-lens critique folded). This is
the sim-side dense signal store the slot-9e logic block reads and writes. It is
NOT in the import-light ``simulation.entities`` package (it holds numpy buffers
and is stepped by the sim loop); the schema/registry stay stdlib-only.

The two invariants that make it deterministic and dormancy-safe:

- **Built ONLY when logic exists (D1).** :func:`build_signal_bus` returns
  ``None`` unless the level declares wires (in B1 the union
  ``sensors ∪ nodes ∪ wires`` reduces to ``wires``). A door-only, wire-free
  level carries NO bus → ``__signals__`` stays empty → its digest is
  byte-identical to Arc A (the dormancy guarantee, §8).
- **The slot table enumerates only wire-referenced / sensor-node-emitted
  signals (D1)** — never "every entity's every signal". In B1 that is exactly
  the set of distinct wire SOURCES (a wired door's ``is_open``, a wired
  entity's free ``alive``). Slots are ordered ``(ordinal, name)`` — identical
  to ``serialize_signal_state``'s sort (serialize.py) — so the dense buffers
  digest with no dict-order dependence.

The bus holds two dense ``int64`` buffers over that frozen slot table:

- ``pub`` — values readable THIS tick (published).
- ``stg`` — where node/actuator writes for NEXT tick land (staged).

Node-output slots (none in B1 — no nodes yet) are swapped ``pub ← stg`` at the
end of slot 9e; sensor / ``alive`` / ``is_open`` slots are refreshed every tick
at 9e(a) and are never swapped. See :mod:`simulation.simulation`'s slot-9e block
for the sub-order that drives this store.

Determinism (§9): integer-only (0/1 flags and Q16.16 later), ordinal slot order,
no RNG, no float, no dict-order dependence.
"""
from __future__ import annotations

import numpy as np

# The free `alive` signal is hashed ONLY as the __entity__ alive row (A4
# critique 7); serialize_signal_state REFUSES it. So a slot named `alive` may
# exist in the bus (a wire can source `<id>.alive` for require_alive / the
# airlock's breach check) but it never enters __signals__.
ALIVE_NAME = "alive"


class SignalBus:
    """Dense integer signal store over a frozen ``(ordinal, name)`` slot table.

    Constructed with the slot table already sorted ``(ordinal, name)``.
    ``node_slots`` is the subset of slot indices whose value is a NODE output
    (staged into ``stg`` at 9e(b), swapped into ``pub`` at 9e(e)); it is empty
    in B1 (no node classes yet), making :meth:`swap_node_signals` a no-op.
    """

    def __init__(self, slots, node_slots=()):
        # slots: iterable of (ordinal:int, name:str), pre-sorted (ordinal, name).
        self.slots = tuple((int(o), str(n)) for o, n in slots)
        self.index = {slot: i for i, slot in enumerate(self.slots)}
        if len(self.index) != len(self.slots):
            raise ValueError("SignalBus slot table has duplicate (ordinal, "
                             "name) entries")
        n = len(self.slots)
        self.pub = np.zeros(n, dtype=np.int64)
        self.stg = np.zeros(n, dtype=np.int64)
        # Node-output slot indices (empty in B1) — the only slots swapped.
        self._node_slots = tuple(sorted(int(i) for i in node_slots))

    def __len__(self) -> int:
        return len(self.slots)

    def slot(self, ordinal: int, name: str) -> int:
        """The dense slot index for ``(ordinal, name)`` — KeyError if absent."""
        return self.index[(int(ordinal), str(name))]

    def has(self, ordinal: int, name: str) -> bool:
        return (int(ordinal), str(name)) in self.index

    def read(self, ordinal: int, name: str) -> int:
        """This tick's published value for a slot (9e reads go through pub)."""
        return int(self.pub[self.index[(int(ordinal), str(name))]])

    def set_pub(self, idx: int, value: int) -> None:
        self.pub[idx] = int(value)

    def swap_node_signals(self) -> None:
        """9e(e): ``pub[node-slots] ← stg[node-slots]``. No-op in B1 (no node
        slots); sensor / alive / is_open slots are refreshed at 9e(a), never
        swapped."""
        for i in self._node_slots:
            self.pub[i] = self.stg[i]

    def digest_rows(self) -> tuple:
        """The ``__signals__`` payload: ``(ordinal, name, pub_value)`` for
        every NON-``alive`` slot, in slot (ordinal, name) order — exactly what
        ``serialize_signal_state`` expects. ``alive`` slots are excluded (they
        ride the __entity__ alive row only, A4 critique 7)."""
        return tuple((o, n, int(self.pub[i]))
                     for i, (o, n) in enumerate(self.slots)
                     if n != ALIVE_NAME)


def build_signal_bus(level_data):
    """Build the SignalBus for a level, or ``None`` when no logic exists (D1).

    The "logic exists" union is ``sensors ∪ nodes ∪ wires`` (§2a); B2 grows the
    B1 wires-only gate to ``nodes ∪ wires`` (sensor emitters join in B3). The
    bus is built iff the level declares at least one resolved wire OR one logic
    node. The slot table (§2a) enumerates only signals that are

    - **wire-referenced** — distinct wire SOURCES ``(source_ordinal, signal)``;
    - **node-emitted** — every LOGIC-NODE instance's ``out`` (its
      prev-read/next-write output, swapped at 9e(e)); AND
    - **require_alive sources** — the ``alive`` slot of the source feeding a
      ``require_alive`` decider's ``in`` wire (D4), so 9e(a) can emit it and
      the decider read it current-tick. ``alive`` slots never enter
      ``__signals__`` (``digest_rows`` excludes them, A4 c7).

    — never "every entity's every signal". Slots are ordered ``(ordinal,
    name)`` (identical to ``serialize_signal_state``'s sort). Node-output slot
    indices are passed as ``node_slots`` so :meth:`SignalBus.swap_node_signals`
    swaps exactly those (sensor / alive / is_open slots are refreshed at 9e(a),
    never swapped).
    """
    from simulation.entities import REGISTRY

    wires = getattr(level_data, "wires", None) or []
    entities = getattr(level_data, "entities", None) or []
    node_ents = [e for e in entities
                 if getattr(REGISTRY.get(e.class_name), "LOGIC_NODE", False)]
    if not wires and not node_ents:
        return None                       # no wires, no nodes ⇒ dormant (D1)

    slots: set = set()
    node_keys: set = set()                # the swapped (node-output) slots
    for w in wires:
        slots.add((int(w.source_ordinal), str(w.signal)))
    for e in node_ents:
        cls = REGISTRY[e.class_name]
        for sig in cls.SIGNALS:           # a node's `out` (prev-read/next-write)
            key = (int(e.ordinal), str(sig.name))
            slots.add(key)
            node_keys.add(key)
    # require_alive decider sources need an `alive` slot to read at 9e(b) (D4).
    for e in node_ents:
        if e.class_name == "decider" and e.fields.get("require_alive"):
            src = _decider_in_source(wires, int(e.ordinal))
            if src is not None:
                slots.add((src, ALIVE_NAME))

    ordered = sorted(slots, key=lambda s: (s[0], s[1]))
    node_slots = tuple(i for i, s in enumerate(ordered) if s in node_keys)
    return SignalBus(ordered, node_slots=node_slots)


def _decider_in_source(wires, decider_ordinal):
    """The source ordinal of the ``in`` wire feeding one decider (its `in` is
    SINGLE — at most one wire, §1b). ``None`` when the input is unwired."""
    for w in wires:
        if int(w.target_ordinal) == int(decider_ordinal) and w.input == "in":
            return int(w.source_ordinal)
    return None
