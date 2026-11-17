"""THE EVENT RING of the device unit kernels (#29 P6.3; residency design §4.5,
"the event ring", and §4.4b).

What a device slot kernel emits -- today the synced ``UnitHitEvent`` /
``UnitKilledEvent`` of ``damage.apply_packet`` -- is a RECORD of
:data:`FIELDS` int64 in a per-env ring ``(N, E_max, F)``, kept per tick, in
the reference's EMISSION ORDER (per slot, id order; within a unit, the order
its law emits). A record carries exactly what
``tests/field_ab_harness._event_record`` hashes -- the kind, the unit, the
damage (Q16: ``damage.apply_packet``'s applied ``dmg_q``, whose float view the
host event carries), the cause (its index in ``events.CAUSE_NAMES``), the
credited attacker -- and the render-only direction the host event carries
(``dir_x_q`` / ``dir_y_q``, set by the producer in integers). So the integer
record is lossless: :func:`event_of` rebuilds the host event bit for bit, and
:func:`record_of` reads one.

The FIELD LAYOUT is this module's ONE table; C++ reads its offsets (and the
vocabulary a kernel needs by name: the event kinds, the causes, the damage
types, the credit channel of the ``heat`` buffer) ONLY from the GENERATED,
checked-in ``cpp/src/unit_events_layout.h`` (:func:`header_text`, written by
``tools/gen_unit_layout.py --write`` beside the store's and the tables'
headers; ``tests/test_unit_events.py`` refuses a stale copy and
:class:`EventRing` refuses a compiled module whose fingerprint is not this
table's). A new event kind, field or cause: regenerate, commit, rebuild
``cpp/build_cuda``.

ORDER (how a kernel appends, ``cpp/src/cuda_unit_events.{h,cu}``): a
per-row kernel writes each row's events into that row's STAGING area (``(N,
U, ROW_CAP, F)``, at most :func:`row_cap` events a row per launch -- 2b's worst
case: one hit per status row and one kill), and one gather per env then
appends the rows' staged events to the ring in ROW order (= id order). The
order is the reference's whatever the thread schedule. A serial per-env
producer (P6.4's applier) appends to the ring directly through the same sink.

CAPACITY: ``E_max`` per env (default :data:`E_MAX`). An event past it -- or
past a row's staging capacity -- is never written: the env's DROPPED count
rises (a counted per-env overflow, never a silent drop) and the kernels'
Python face (``simulation.unit_kernels``) refuses the run, naming the kernel
and the env (:class:`UnitEventOverflow`).

NOT PART OF THE PLAYED TICK: Class T (training only), like the store.
Import-light: numpy and the event vocabulary; ``breach_physics`` only inside
:class:`EventRing`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from simulation import unit_fixed

#: The generated header (C++'s copy of the ring layout).
HEADER_REL = "cpp/src/unit_events_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"

#: A record's fields, in order. ``damage_q``: the APPLIED damage in Q16 (a
#: hit; 0 for a kill); ``cause``: the index in ``events.CAUSE_NAMES``.
FIELDS = ("kind", "unit_id", "damage_q", "cause", "attacker_id", "dir_x_q",
          "dir_y_q")
F = len(FIELDS)
#: The kinds a record holds (code = position + 1; 0 = no record).
KINDS = ("UnitHitEvent", "UnitKilledEvent")
KIND_HIT, KIND_KILLED = 1, 2
#: The default ring capacity per env per tick.
E_MAX = 1024
#: The credit channel code of the ``heat`` buffer in the claims table (a gas
#: deposit's channel is its ``gmap.gas`` slice index, >= 0).
CREDIT_HEAT = -1
#: A claims table's padding row's channel (never matches).
CREDIT_PAD = -2


class UnitEventOverflow(RuntimeError):
    """A device kernel's events did not fit the ring (or a row's staging)."""


def causes() -> tuple:
    from simulation.events import CAUSE_NAMES
    return tuple(CAUSE_NAMES)


def damage_types() -> tuple:
    from simulation.damage import DAMAGE_TYPE_NAMES, N_DAMAGE_TYPES
    return tuple(DAMAGE_TYPE_NAMES[i] for i in range(N_DAMAGE_TYPES))


def row_cap() -> int:
    """The most events one row emits in one per-row kernel launch: one hit
    per status row (slot 2b's damage over time) and one kill."""
    from simulation.unit_schema import status_capacity
    return int(status_capacity()) + 1


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    fields: tuple
    kinds: tuple
    causes: tuple
    damage_types: tuple
    row_cap: int
    fingerprint: str

    @property
    def width(self) -> int:
        return len(self.fields)

    def field(self, name: str) -> int:
        return self.fields.index(name)


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    key = (FIELDS, KINDS, causes(), damage_types(), row_cap())
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    text = "".join(f"field:{i}:{n}\n" for i, n in enumerate(FIELDS))
    text += "".join(f"kind:{i + 1}:{n}\n" for i, n in enumerate(KINDS))
    text += "".join(f"cause:{i}:{n}\n" for i, n in enumerate(key[2]))
    text += "".join(f"dtype:{i}:{n}\n" for i, n in enumerate(key[3]))
    text += f"row_cap:{key[4]}\ncredit_heat:{CREDIT_HEAT}\ncredit_pad:{CREDIT_PAD}\n"
    lay = Layout(fields=FIELDS, kinds=KINDS, causes=key[2],
                 damage_types=key[3], row_cap=key[4],
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())
    _LAYOUT_CACHE[key] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/unit_events_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/unit_events.py",
        "// (#29 P6.3, residency design 4.5 / 4.4b) -- NEVER EDIT BY HAND.",
        "// The event ring of the device unit kernels: per env (E_max, UNIT_EVENT_F)",
        "// int64 records in the reference's emission order, each the fields",
        "// field_ab_harness._event_record hashes plus the render-only direction;",
        "// and the vocabulary a kernel names: the event kinds, the causes",
        "// (events.CAUSE_NAMES order), the damage types (damage.py order), the",
        "// credit channel of the heat buffer. A new kind / field / cause: run",
        f"// `{GENERATOR}` and commit; tests/test_unit_events.py refuses a",
        "// stale copy.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace unit_events {",
        "",
        f'constexpr const char* UNIT_EVENTS_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int UNIT_EVENT_F = {lay.width};",
        f"constexpr int UNIT_EVENT_ROW_CAP = {lay.row_cap};",
        "",
        "// A record's fields.",
    ]
    for i, n in enumerate(lay.fields):
        lines.append(f"constexpr int UNIT_EV_{n.upper()} = {i};")
    lines += ["", "// The record kinds (0 = no record)."]
    for i, n in enumerate(lay.kinds):
        short = {"UnitHitEvent": "HIT", "UnitKilledEvent": "KILLED"}[n]
        lines.append(f"constexpr int64_t UNIT_EVKIND_{short} = {i + 1};")
    lines += ["", "// The causes (events.CAUSE_NAMES order)."]
    for i, n in enumerate(lay.causes):
        lines.append(f"constexpr int64_t UNIT_CAUSE_{n.upper()} = {i};")
    lines.append(f"constexpr int UNIT_N_CAUSES = {len(lay.causes)};")
    lines += ["", "// The damage types (damage.py order)."]
    for i, n in enumerate(lay.damage_types):
        lines.append(f"constexpr int UNIT_DTYPE_{n.upper()} = {i};")
    lines.append(f"constexpr int UNIT_N_DAMAGE_TYPES = {len(lay.damage_types)};")
    lines += ["", "// The claims table's channel codes (a gas slice is its index).",
              f"constexpr int64_t UNIT_CREDIT_HEAT = {CREDIT_HEAT};",
              f"constexpr int64_t UNIT_CREDIT_PAD = {CREDIT_PAD};",
              "", "}  // namespace unit_events", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Records <-> host events
# ---------------------------------------------------------------------------
def is_synced(ev) -> bool:
    """#29 P7a.1: whether a host event is a SYNCED one -- a kind the ring
    holds (``field_ab_harness._SYNCED_EVENT_TYPES``: the unit hit / kill the
    digest hashes). The render-only events (MeleeEvent, ShotFired, SprayJet,
    the destroyed-wall signals, ...) have no device form; the shadow harness
    drops them before it compares a ring."""
    return type(ev).__name__ in KINDS


def record_of(ev) -> tuple:
    """A host event as its ring record (a tuple of :data:`F` ints). Refused
    (``ValueError``) for an event the ring does not hold, naming its type."""
    kind = type(ev).__name__
    if kind not in KINDS:
        raise ValueError(f"unit events: the ring holds {KINDS}, not {kind}")
    cause_list = causes()
    if kind == "UnitHitEvent":
        dmg_q = unit_fixed.quantize_scalar(float(ev.damage))
        if unit_fixed.dequantize_scalar(dmg_q) != float(ev.damage):
            raise ValueError(f"unit events: a hit's damage {ev.damage!r} is "
                             f"not on the Q16 grid")
        return (KIND_HIT, int(ev.unit_id), dmg_q, cause_list.index(ev.source),
                int(ev.attacker_id), int(ev.dir_x_q), int(ev.dir_y_q))
    return (KIND_KILLED, int(ev.unit_id), 0, cause_list.index(ev.killed_by),
            int(ev.attacker_id), 0, 0)


def event_of(rec):
    """A ring record as the host event ``damage.apply_packet`` appends (the
    damage its float view, the cause its string): equal field for field."""
    from simulation.events import UnitHitEvent, UnitKilledEvent
    r = [int(x) for x in rec]
    cause = causes()[r[3]]
    if r[0] == KIND_HIT:
        return UnitHitEvent(unit_id=r[1],
                            damage=unit_fixed.dequantize_scalar(r[2]),
                            source=cause, attacker_id=r[4], dir_x_q=r[5],
                            dir_y_q=r[6])
    if r[0] == KIND_KILLED:
        return UnitKilledEvent(unit_id=r[1], killed_by=cause, attacker_id=r[4])
    raise ValueError(f"unit events: no record kind {r[0]}")


# ---------------------------------------------------------------------------
# The device ring
# ---------------------------------------------------------------------------
class EventRing:
    """THE per-env device event ring ``(N, E_max, F)`` int64 beside a
    ``DeviceStore`` of ``(N, U)`` rows (the staging area is per row), on the
    tick stream (``breach_physics.UnitEventRingBuffer``, the CUDA build
    only). :meth:`begin` empties every env's ring (the start of a tick); a
    kernel appends; :meth:`download` returns the records, the counts and the
    per-env DROPPED counts. Refuses a compiled module whose events layout is
    not this table's."""

    def __init__(self, n_env: int, n_units: int, e_max: int = E_MAX, bp=None):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("unit events: the device ring needs the CUDA "
                               "build (breach_physics.HAS_CUDA)")
        lay = layout()
        compiled = bp.unit_events_layout()
        if compiled["fingerprint"] != lay.fingerprint \
                or int(compiled["width"]) != lay.width \
                or int(compiled["row_cap"]) != lay.row_cap:
            raise ValueError(
                f"unit events: the compiled ring layout ({compiled['fingerprint'][:12]}) "
                f"is not simulation.unit_events' ({lay.fingerprint[:12]}): run "
                f"`{GENERATOR}` and rebuild cpp/build_cuda")
        if int(e_max) < 1:
            raise ValueError(f"unit events: e_max {e_max} must be >= 1")
        self.bp = bp
        self.shape = (int(n_env), int(e_max), lay.width)
        self.n_units = int(n_units)
        self._buf = bp.UnitEventRingBuffer(int(n_env), int(n_units), int(e_max))

    @property
    def buffer(self):
        return self._buf

    def begin(self) -> None:
        """Empty every env's ring (counts and dropped to 0)."""
        self._buf.begin()

    def download(self):
        """``(records (N, E_max, F), counts (N,), dropped (N,))`` int64."""
        return self._buf.download()

    def records(self) -> list:
        """Per env, its records as a list of int tuples (emission order)."""
        recs, counts, _dropped = self.download()
        return [[tuple(int(x) for x in recs[e, i]) for i in range(int(counts[e]))]
                for e in range(recs.shape[0])]


__all__ = ["HEADER_REL", "GENERATOR", "FIELDS", "F", "KINDS", "KIND_HIT",
           "KIND_KILLED", "E_MAX", "CREDIT_HEAT", "CREDIT_PAD",
           "UnitEventOverflow", "causes", "damage_types", "row_cap", "Layout",
           "layout", "header_text", "record_of", "event_of", "EventRing",
           "is_synced"]
