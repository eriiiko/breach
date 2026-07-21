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

    In B1 the "logic exists" union ``sensors ∪ nodes ∪ wires`` reduces to
    ``wires``: the bus is built iff the level declares at least one resolved
    wire. The slot table is the set of distinct wire SOURCES
    ``(source_ordinal, signal)`` — the only signals a consumer reads this arc —
    sorted ``(ordinal, name)``. Sensor/node emitters add slots in later
    patches; none exist yet, so there are no node-output slots to swap.
    """
    wires = getattr(level_data, "wires", None) or []
    if not wires:
        return None                       # no wires ⇒ no bus ⇒ dormant (D1)
    sources = {(int(w.source_ordinal), str(w.signal)) for w in wires}
    slots = sorted(sources, key=lambda s: (s[0], s[1]))
    return SignalBus(slots)
