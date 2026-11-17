"""THE REQUEST LIST of the device unit kernels (#29 P6.4a; residency design
§4.3 row "1 events, 1b intents", §4.4b, §4.5).

What a device unit kernel does that is NOT a write to a unit row -- today
slot 1b's three: a thrown grenade leaving the hand (a ``combat.Projectile``
appended to ``sim.projectiles``), a planted charge laid down (a
``charges.PlantedCharge`` through ``charges.place``), a door's ``want_open``
latch flipped (``Simulation._direct_use``) -- is a REQUEST RECORD of
:data:`FIELDS` int64 in a per-env list ``(N, R_max, F)``, kept per tick,
carrying every value the reference computes at that point. Since #29 P7b a
throw and an accepted plant are TABLE WRITES: ``unit_kernels.table_appends``
appends them to the device grenade and charge tables (``unit_schema.
GRENADE_TABLE`` / ``CHARGE_TABLE``) in list order, and their records here are
the transport; the door's table arrives with P6b. A record is what the
harness compares with what the host did, and what the CHAINED form of slot
1b alone turns back into the host's objects (:func:`apply_to_host`; a chain
that lands the tables never does so for a throw or a plant).

ORDER is the reference's and is load-bearing: slot 1b walks the units in id
order and, within a unit, throws, then plants, then uses; slot 2 detonates
grenades in ``sim.projectiles`` order, and ``charges.place``'s capacity
refuses by count in id order. A per-row kernel writes each row's requests
into that row's staging (at most :data:`ROW_CAP` a launch: one of each kind)
and the ONE gather of the event ring's mechanism (``cpp/src/cuda_unit_events
.{h,cu}``: ``UnitRecordList``, which the event ring is an instance of)
appends them per env in row (= id) order. The one ORDER-DEPENDENT decision --
which plant a full charge table refuses -- is then made by a per-env pass in
list order against the env's live-charge count (a per-tick input,
:func:`charges_live`): an accepted plant is marked ``accepted`` 1 and spends
the planter's charge (``has_explosive``), a refused one stays 0 and is
counted (``refused``, == the host's ``sim.charges_refused`` delta).

The FIELD LAYOUT is this module's ONE table; C++ reads its offsets and kind
codes ONLY from the GENERATED, checked-in ``cpp/src/unit_requests_layout.h``
(:func:`header_text`, written by ``tools/gen_unit_layout.py --write``;
``tests/test_unit_requests.py`` refuses a stale copy, and
:class:`RequestList` refuses a compiled module whose fingerprint is not this
table's). A new request kind or field: regenerate, commit, rebuild
``cpp/build_cuda``.

CAPACITY: ``R_max`` per env (default :data:`R_MAX`). A request past it -- or
past a row's staging -- is never written: the env's DROPPED count rises and
the kernels' Python face refuses the run, naming the kernel and the env
(:class:`UnitRequestOverflow`).

NOT PART OF THE PLAYED TICK: Class T (training only), like the store.
Import-light: numpy; ``breach_physics`` only inside :class:`RequestList`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from simulation import unit_fixed

#: The generated header (C++'s copy of the request layout).
HEADER_REL = "cpp/src/unit_requests_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"

#: A record's fields, in order. Per kind (unused fields 0, ``door`` -1):
#:   throw: ``unit_id`` the thrower (the projectile's owner), ``tick`` the
#:          thrown tick, ``tile_x`` / ``tile_y`` the start tile (the
#:          thrower's centre tile), ``target_x_q`` / ``target_y_q`` the Q16
#:          target, ``ticks`` the fuse (clamped to the hand grenade's range),
#:          ``item`` the throw kind (the hand-grenade row's index);
#:   plant: ``unit_id`` the planter (the charge's owner), ``tick`` the slot's
#:          tick, ``tile_x`` / ``tile_y`` the charge's tile (the planter's
#:          centre tile), ``ticks`` the det_tick, ``item`` the charge's
#:          ``[ammo.*]`` row index (``unit_schema.table_row("ammo", ...)``),
#:          ``approach_y`` / ``approach_x`` the approach tile, ``accepted``
#:          1 placed / 0 refused at the charge table's capacity;
#:   use:   ``unit_id`` the user, ``tick`` the slot's tick, ``door`` the
#:          flipped door's ordinal in ``sim._doors``.
FIELDS = ("kind", "unit_id", "tick", "tile_x", "tile_y", "target_x_q",
          "target_y_q", "ticks", "item", "approach_y", "approach_x", "door",
          "accepted")
F = len(FIELDS)
#: The kinds a record holds (code = position + 1; 0 = no record).
KINDS = ("throw", "plant", "use")
KIND_THROW, KIND_PLANT, KIND_USE = 1, 2, 3
#: The most requests one row makes in one launch of slot 1b: one of each kind
#: (a unit releases at most one grenade a tick -- a 0-tick wind-up releases at
#: once and leaves no wind-up to advance --, places at most one charge, flips
#: at most one door).
ROW_CAP = len(KINDS)
#: The default list capacity per env per tick.
R_MAX = 256
#: The door-span table's row: (door ordinal, ty, tx); a padding row has
#: ordinal -1.
DOOR_SPAN_WIDTH = 3
NO_DOOR = -1


class UnitRequestOverflow(RuntimeError):
    """A device kernel's requests did not fit the list (or a row's staging)."""


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    fields: tuple
    kinds: tuple
    row_cap: int
    fingerprint: str

    @property
    def width(self) -> int:
        return len(self.fields)

    def field(self, name: str) -> int:
        return self.fields.index(name)


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    key = (FIELDS, KINDS, ROW_CAP)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    text = "".join(f"field:{i}:{n}\n" for i, n in enumerate(FIELDS))
    text += "".join(f"kind:{i + 1}:{n}\n" for i, n in enumerate(KINDS))
    text += f"row_cap:{ROW_CAP}\nno_door:{NO_DOOR}\n"
    lay = Layout(fields=FIELDS, kinds=KINDS, row_cap=ROW_CAP,
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest())
    _LAYOUT_CACHE[key] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/unit_requests_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/unit_requests.py",
        "// (#29 P6.4a, residency design 4.3 / 4.4b) -- NEVER EDIT BY HAND.",
        "// The request list of the device unit kernels: per env (R_max,",
        "// UNIT_REQ_F) int64 records -- what a kernel does that is not a write",
        "// to a unit row (a grenade thrown, a charge planted, a door flipped) --",
        "// in the reference's order. A new kind / field: run",
        f"// `{GENERATOR}` and commit; tests/test_unit_requests.py",
        "// refuses a stale copy.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace unit_requests {",
        "",
        f'constexpr const char* UNIT_REQUESTS_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int UNIT_REQ_F = {lay.width};",
        f"constexpr int UNIT_REQ_ROW_CAP = {lay.row_cap};",
        f"constexpr int64_t UNIT_REQ_NO_DOOR = {NO_DOOR};",
        "",
        "// A record's fields.",
    ]
    for i, n in enumerate(lay.fields):
        lines.append(f"constexpr int UNIT_RQ_{n.upper()} = {i};")
    lines += ["", "// The record kinds (0 = no record)."]
    for i, n in enumerate(lay.kinds):
        lines.append(f"constexpr int64_t UNIT_RQKIND_{n.upper()} = {i + 1};")
    lines += ["", "}  // namespace unit_requests", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The per-tick inputs of slot 1b
# ---------------------------------------------------------------------------
def door_spans(sim) -> np.ndarray:
    """The LIVE doors' span tiles as ``(R, 3)`` int64 rows ``(ordinal, ty,
    tx)`` -- the ordinal the door's index in ``sim._doors``, in door order and
    each door's span order: what ``verb_gates.use_door`` scans (a dead door
    is skipped there and absent here)."""
    rows = []
    for k, d in enumerate(getattr(sim, "_doors", ()) or ()):
        if not d.alive:
            continue
        for (ty, tx) in d.span:
            rows.append((k, int(ty), int(tx)))
    return np.array(rows, dtype=np.int64).reshape(len(rows), DOOR_SPAN_WIDTH)


def pad_door_spans(per_env) -> np.ndarray:
    """N envs' span rows as one ``(N, max(1, R), 3)`` array, the short envs
    padded with ordinal -1 (never in reach)."""
    per_env = [np.asarray(r, dtype=np.int64).reshape(-1, DOOR_SPAN_WIDTH)
               for r in per_env]
    R = max([1] + [r.shape[0] for r in per_env])
    out = np.zeros((len(per_env), R, DOOR_SPAN_WIDTH), dtype=np.int64)
    out[:, :, 0] = NO_DOOR
    for e, r in enumerate(per_env):
        out[e, :r.shape[0]] = r
    return out


def charges_live(sim) -> int:
    """The live planted charges the sim holds (what ``charges.place`` counts
    against ``unit_schema.CHARGE_CAPACITY``)."""
    return sum(1 for c in getattr(sim, "planted_charges", ()) if c.live)


# ---------------------------------------------------------------------------
# Records <-> host objects
# ---------------------------------------------------------------------------
def _blank(kind, unit_id, tick) -> list:
    r = [0] * F
    r[0], r[1], r[2] = int(kind), int(unit_id), int(tick)
    r[FIELDS.index("door")] = NO_DOOR
    return r


def throw_record(sim, proj) -> tuple:
    """The record of a ``Projectile`` slot 1b appended (its start tile, Q16
    target, fuse, thrown tick, kind and owner)."""
    tables = sim.weapons_tables
    kinds = tables.hand_grenade_rows
    if proj.ammo_name not in kinds:
        raise ValueError(f"unit requests: {proj.ammo_name!r} is no hand-grenade kind")
    r = _blank(KIND_THROW, proj.owner_id, proj.thrown_tick)
    sx, sy = int(proj.start_x_q), int(proj.start_y_q)
    if sx & 0xFFFF or sy & 0xFFFF:
        raise ValueError("unit requests: a thrown grenade's start is not a tile")
    r[3], r[4] = sx >> 16, sy >> 16
    r[5], r[6] = int(proj.target_x_q), int(proj.target_y_q)
    r[7] = int(proj.fuse_ticks)
    r[8] = kinds.index(proj.ammo_name)
    r[12] = 1
    return tuple(r)


def plant_record(charge, tick, accepted=True) -> tuple:
    """The record of a ``PlantedCharge`` slot 1b laid down at ``tick``."""
    from simulation.unit_schema import table_row
    r = _blank(KIND_PLANT, charge.owner_id, tick)
    r[3], r[4] = int(charge.x), int(charge.y)
    r[7] = int(charge.det_tick)
    r[8] = table_row("ammo", charge.ammo_name)
    r[9], r[10] = (int(v) for v in charge.approach)
    r[12] = 1 if accepted else 0
    return tuple(r)


def use_record(unit_id, tick, door_ordinal) -> tuple:
    """The record of a door flip by ``unit_id``."""
    r = _blank(KIND_USE, unit_id, tick)
    r[11] = int(door_ordinal)
    r[12] = 1
    return tuple(r)


def apply_to_host(sim, records) -> dict:
    """Turn ``records`` (one env's, in list order) into what slot 1b's host
    code would have done: each throw a ``Projectile`` appended to
    ``sim.projectiles``, each plant offered to ``charges.place`` (THE door --
    it must agree with the record's ``accepted``, else ``ValueError``), each
    use a ``want_open`` flip. Returns the counts by kind."""
    from simulation import charges
    from simulation.combat import Projectile
    from simulation.orders import ORDER_GRENADE
    from simulation.unit_schema import table_name
    n = {"throw": 0, "plant": 0, "refused": 0, "use": 0}
    tables = sim.weapons_tables
    for rec in records:
        r = [int(x) for x in rec]
        kind = r[0]
        if kind == KIND_THROW:
            sim.projectiles.append(Projectile(
                ORDER_GRENADE, r[3], r[4],
                unit_fixed.dequantize_scalar(r[5]),
                unit_fixed.dequantize_scalar(r[6]),
                fuse_seconds=0.0, thrown_tick=r[2],
                ammo_name=tables.hand_grenade_kind(r[8]), owner_id=r[1],
                fuse_ticks=r[7]))
            n["throw"] += 1
        elif kind == KIND_PLANT:
            placed = charges.place(sim, charges.PlantedCharge(
                r[3], r[4], table_name("ammo", r[8]), r[1], det_tick=r[7],
                approach=(r[9], r[10])))
            if bool(placed) != bool(r[12]):
                raise ValueError(f"unit requests: the device {'placed' if r[12] else 'refused'} "
                                 f"a charge charges.place {'refused' if r[12] else 'placed'}")
            n["plant" if placed else "refused"] += 1
        elif kind == KIND_USE:
            d = sim._doors[r[11]]
            d.want_open = not d.want_open
            n["use"] += 1
        else:
            raise ValueError(f"unit requests: no record kind {kind}")
    return n


# ---------------------------------------------------------------------------
# The device list
# ---------------------------------------------------------------------------
class RequestList:
    """THE per-env device request list ``(N, R_max, F)`` int64 beside a
    ``DeviceStore`` of ``(N, U)`` rows (the staging area is per row), on the
    tick stream (``breach_physics.UnitRecordListBuffer`` at this table's
    width and row cap, the CUDA build only: the event ring's own record-list
    class). :meth:`begin` empties every env's list; a kernel appends;
    :meth:`download` returns the records, the counts and the per-env DROPPED
    counts. Refuses a compiled module whose request layout is not this
    table's."""

    def __init__(self, n_env: int, n_units: int, r_max: int = R_MAX, bp=None):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("unit requests: the device list needs the CUDA "
                               "build (breach_physics.HAS_CUDA)")
        lay = layout()
        compiled = bp.unit_requests_layout()
        if compiled["fingerprint"] != lay.fingerprint \
                or int(compiled["width"]) != lay.width \
                or int(compiled["row_cap"]) != lay.row_cap:
            raise ValueError(
                f"unit requests: the compiled request layout "
                f"({compiled['fingerprint'][:12]}) is not simulation."
                f"unit_requests' ({lay.fingerprint[:12]}): run `{GENERATOR}` "
                f"and rebuild cpp/build_cuda")
        if int(r_max) < 1:
            raise ValueError(f"unit requests: r_max {r_max} must be >= 1")
        self.bp = bp
        self.shape = (int(n_env), int(r_max), lay.width)
        self.n_units = int(n_units)
        self._buf = bp.UnitRecordListBuffer(int(n_env), int(n_units),
                                            int(r_max), lay.width,
                                            lay.row_cap)

    @property
    def buffer(self):
        return self._buf

    def begin(self) -> None:
        """Empty every env's list (counts and dropped to 0)."""
        self._buf.begin()

    def download(self):
        """``(records (N, R_max, F), counts (N,), dropped (N,))`` int64."""
        return self._buf.download()

    def records(self) -> list:
        """Per env, its records as a list of int tuples (list order)."""
        recs, counts, _dropped = self.download()
        return [[tuple(int(x) for x in recs[e, i]) for i in range(int(counts[e]))]
                for e in range(recs.shape[0])]


__all__ = ["HEADER_REL", "GENERATOR", "FIELDS", "F", "KINDS", "KIND_THROW",
           "KIND_PLANT", "KIND_USE", "ROW_CAP", "R_MAX", "DOOR_SPAN_WIDTH",
           "NO_DOOR", "UnitRequestOverflow", "Layout", "layout", "header_text",
           "door_spans", "pad_door_spans", "charges_live", "throw_record",
           "plant_record", "use_record", "apply_to_host", "RequestList"]
