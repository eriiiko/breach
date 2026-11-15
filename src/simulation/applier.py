"""THE PER-ENV SERIAL APPLIER's one table (#29 P6.4b; residency design §4.3,
"The per-env serial applier", and §4.4b's P6.4 row).

The applier is THE device form of every order-dependent law that writes the
map: ONE BLOCK per env (``blockIdx.y`` the env, the ``bgrid`` convention),
thread 0 walking that env's recorded EFFECTS in the reference's order, each
effect a DEVICE FUNCTION (``cpp/src/cuda_topology.cuh``: the topology seam --
``on_tile_changed``, ``destroy_wall``) whose book writes land in per-env BOOK
channels. This module is its ONE Python table, with generated offsets
(``cpp/src/applier_layout.h``, written by ``tools/gen_unit_layout.py --write``
beside the unit store's, the tables' and the event ring's; fingerprint-checked
by :class:`DeviceApplier`):

- :data:`FIELDS` / :data:`KINDS` -- an effect RECORD (int64 ``kind, y, x,
  arg``) and its kinds. Today ``DESTROY_WALL`` (slot 9's burn-through; P7's
  chew and blasts and P6b's door deaths append the same kind). A per-env list
  ``(N, cap, F)`` with a count per env: an effect past ``cap`` is never
  applied, the env's ``dropped`` book rises and the Python face refuses the
  run naming the kernel and the env (:class:`EffectOverflow`); the first
  ``cap`` are applied.
- :data:`BOOKS` -- the per-env book channels the effects write, each with the
  HOST quantity it twins (``gmap.<attr>`` or ``gmap.<dict>:<key>``), so the
  shadow harness compares a book with the host's delta over the slot and the
  chained harness lands it on the host.
- :data:`PLANES` -- the planes the topology seam reads and writes, by
  ``GameMap`` field name, each in its own dtype and per-cell width: what the
  applier's launch takes as device pointers to ``(N, ...)`` arrays. Most are
  resident fields (``GameMap._RESIDENT_FIELD_NAMES``: the arrays
  ``ResidentBatch.buffers`` holds); :func:`non_resident_planes` names the rest
  (P8b must hold them on the device).
- :data:`CLEAR_PLANES` -- the per-tick planes the tick-end clears zero.

A NEW EFFECT KIND, BOOK OR PLANE is a row here, ``tools/gen_unit_layout.py
--write``, commit, rebuild ``cpp/build_cuda``.

The fan-out of an effect's tile set over the block's threads with a barrier
between effects (§4.3: a blast's deposits, a reach walk's frontier) arrives
with its first user (P7c's deposits); the launch already has the block shape
so it can be added without moving the kernel.

NOT PART OF THE PLAYED TICK: Class T (training only), like the unit store.
Import-light: numpy; ``breach_physics`` only inside :class:`DeviceApplier`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

#: The generated header (C++'s copy of this table).
HEADER_REL = "cpp/src/applier_layout.h"
GENERATOR = "tools/gen_unit_layout.py --write"

#: #29 P7a.1 -- THE EFFECT LIST (residency design 4.4c, decision 1): the
#: record is the kind, then ARG_SLOTS int64 whose meaning is the kind's own
#: (:data:`KIND_FIELDS`); C++ reads a kind's field as ``APPLIER_<KIND>_<FIELD>``
#: (its column), never a bare slot number.
ARG_SLOTS = 10
#: An effect record's fields, in order.
FIELDS = ("kind",) + tuple(f"a{i}" for i in range(ARG_SLOTS))
F = len(FIELDS)
#: The effect kinds (code = position + 1; 0 = no record) and each kind's
#: fields, in the order they occupy the arg slots. Every value is what the
#: reference's call receives at that point, already in its integer currency:
#:
#: - DESTROY_WALL ``GameMap.destroy_wall(y, x)`` (slot 9's burn-through; P6b's
#:   door deaths);
#: - HIT ``damage.apply_packet(units[row], packet, sim.tick_events,
#:   source=cause)``: the packet at its door (``DamagePacket.q()``, the AP
#:   quantized as ``mitigate_q`` quantizes it, the attacker normalized, the
#:   cause's index in ``events.CAUSE_NAMES``); in a stage that LANDS
#:   (:data:`STAGES`) with the slot's exposure landing;
#: - STATUS ``status.apply_status`` if the unit is alive at its turn (the
#:   magnitude quantized as ``apply_status`` quantizes it; source -1 = None);
#: - PUSH ``unit_impulse.add(sim.impulse_table, units[row], ...)`` if alive;
#: - CHEW ``combat.chew_wall(gmap, y, x, wall_damage)``: ``live`` 0 when
#:   ``wall_damage <= 0`` (a no-op), ``damage_q`` the per-call
#:   ``wall_fixed.quantize_scalar(float(wall_damage))``;
#: - PAYLOAD ``payloads.execute_payload(gmap, queue, units, y, x, payload,
#:   rng, events=sim.tick_events, kind=..., owner_id=owner, approach=...)``:
#:   a RECORD ONLY -- the applier refuses to apply it, by name, until P7c.1
#:   gives it its function (``payload`` the row's index in
#:   ``sim.weapons_tables.payloads``, ``kind`` its index in
#:   :data:`PAYLOAD_KINDS`, ``has_approach`` 0 for ``approach=None``).
#:
#: - SPRAY_CELLS (#29 P7a.3) ``combat._enqueue_spray_cells(gmap, queue,
#:   owner, heat_q, gas_id, gas_q, cells)``: the call's scalars as given
#:   (``gas_id`` -1 for ``None``) and the cells' SPAN in the env's CELL BUFFER
#:   (``simulation.attack_lists``: rows ``(y, x, div)``, filled by slot 4b's
#:   phase A against the walls standing at the slot's start; the walk never
#:   recomputes a cone).
#:
#: There is no FACE kind: its only producers are slot 4's WEGO branches
#: (design 4.4c scope); a host FACE effect is refused by name.
KIND_FIELDS: tuple = (
    ("DESTROY_WALL", ("y", "x")),
    ("HIT", ("row", "amount_q", "dtype", "attacker", "indirect", "expose",
             "dir_x_q", "dir_y_q", "ap_q", "cause")),
    ("STATUS", ("row", "status", "magnitude_q", "duration", "source")),
    ("PUSH", ("row", "jx_q", "jy_q", "lever_q", "twist_q16")),
    ("CHEW", ("y", "x", "damage_q", "live")),
    ("PAYLOAD", ("y", "x", "payload", "owner", "approach_y", "approach_x",
                 "has_approach", "kind")),
    ("SPRAY_CELLS", ("owner", "heat_q", "gas_id", "gas_q", "start", "count")),
)
KINDS = tuple(k for k, _f in KIND_FIELDS)
KIND_DESTROY_WALL, KIND_HIT, KIND_STATUS, KIND_PUSH, KIND_CHEW, \
    KIND_PAYLOAD, KIND_SPRAY_CELLS = range(1, 8)
#: The kinds the applier refuses to APPLY (counted in the ``refused`` book,
#: the run refused by name): a record with no device function yet.
UNAPPLIED_KINDS = ("PAYLOAD",)
#: ``execute_payload``'s ``kind`` values (its render-only ExplosionEvent
#: label), as a PAYLOAD record's ``kind`` index; another is refused by name.
PAYLOAD_KINDS = ("explosion", "shell", "grenade", "door_explosive",
                 "timed_charge", "heat_source")
#: The STAGES of the one applier kernel (decision 1: one walker, several
#: stages): (name, lands) -- ``lands`` True when a HIT in that stage holds
#: its exposure raises and zeroings for the walk's end (decision 2: the mode
#: is per (stage, kind); every other packet lands at once).
#:   TOPOLOGY  slot 9's burn-through list (DESTROY_WALL only today);
#:   RESOLVE   slot 4c, ``SimultaneousEffects.resolve`` of the attack slot.
STAGES: tuple = (("TOPOLOGY", False), ("RESOLVE", True))
STAGE_NAMES = tuple(s for s, _l in STAGES)
STAGE_TOPOLOGY, STAGE_RESOLVE = 0, 1
#: The most effects ONE ROW of a per-row producer stages in one launch: slot
#: 4a's blow is a HIT, a STATUS and a PUSH. A later producer that stages more
#: raises it (a row here, regenerate).
ROW_CAP = 3
#: The default list capacity per env per launch.
E_CAP = 256


assert all(len(fs) <= ARG_SLOTS for _k, fs in KIND_FIELDS),     "applier: a kind has more fields than the record's arg slots"


def kind_field(kind: str, name: str) -> int:
    """The record column of ``kind``'s field ``name``."""
    for k, fs in KIND_FIELDS:
        if k == kind:
            return 1 + fs.index(name)
    raise KeyError(f"applier: no kind {kind!r}")


def make_record(kind: str, /, **values) -> tuple:
    """One record of ``kind`` from its named fields (every field required)."""
    fs = dict(KIND_FIELDS)[kind]
    if set(values) != set(fs):
        raise ValueError(f"applier: a {kind} record has the fields {fs}, not "
                         f"{tuple(sorted(values))}")
    rec = [0] * F
    rec[0] = KINDS.index(kind) + 1
    for i, n in enumerate(fs):
        rec[1 + i] = int(values[n])
    return tuple(rec)


@dataclass(frozen=True)
class BookRow:
    """One per-env book channel: ``host`` names the host quantity it twins
    (``attr`` or ``attr:key`` on the GameMap; None = the applier's own
    count, with no host attribute)."""
    name: str
    host: str | None
    note: str = ""


BOOKS: tuple = (
    BookRow("n_destruction_seed", "n_destruction_seed_sum",
            "destroy_wall's measured bulk delta (design §5.10: the destroyed "
            "tile's own bulk delta)"),
    BookRow("ge_retire", "gas_energy_books:retire",
            "the gas energy seam's 'retire' channel (on_tile_changed's "
            "membership retire; a breach's evacuation)"),
    BookRow("ge_mint", "gas_energy_books:mint",
            "the 'mint' channel (on_tile_changed's born-at-ambient)"),
    BookRow("ge_destroy_seed", "gas_energy_books:destroy_seed",
            "destroy_wall's seed: the retire, then the one true mint"),
    BookRow("solid_destroy_wall", "solid_energy_books:destroy_wall",
            "the solid books' destroy_wall channel (its _solid_books_sink "
            "is the host's: the chained harness books it there)"),
    BookRow("tiles_changed", "_tile_serial",
            "on_tile_changed calls (the host's _tile_serial bump)"),
    BookRow("applied", None, "effects that acted (a destroy of a non-air "
            "tile in bounds)"),
    BookRow("dropped", None, "effects past the list's capacity (refused)"),
    BookRow("faults", None, "effects whose operands left the proved range "
            "(nothing written; refused)"),
    BookRow("refused", None, "#29 P7a.1: records of a kind with no device "
            "function yet (UNAPPLIED_KINDS: PAYLOAD until P7c.1), never "
            "applied; the run is refused by name"),
)
BOOK_NAMES = tuple(b.name for b in BOOKS)
N_BOOKS = len(BOOKS)

#: The topology seam's planes: (GameMap field, numpy dtype, per-cell width;
#: "G" = the gas planes' count, the leading axis). Float planes are read and
#: written as their exact 32-bit pattern (never a float on the device).
PLANES: tuple = (
    ("material", "int8", 1),
    ("wall_hp", "int32", 1),
    ("gas", "int32", "G"),
    ("gas_energy", "int64", 1),
    ("temperature", "int32", 1),
    ("fire", "int32", 1),
    ("atmosphere", "int32", 1),
    ("solid", "bool", 1),
    ("is_vacuum", "bool", 1),
    ("is_ambient", "bool", 1),
    ("thermal_solid", "bool", 1),
    ("flammable", "bool", 1),
    ("sight_screen", "bool", 1),
    ("heat_inv_shift", "int32", 1),
    ("face_shift", "int32", 4),
    ("conductivity", "float32", 1),
    ("permeability", "float32", 1),
    ("wave_absorb", "float32", 1),
    ("heat_atten", "float32", 1),
    ("heat_atten_q", "int32", 1),
    ("dyn_heat_atten_q", "int32", 1),
    ("light_atten", "float32", 3),
    ("light_atten_q", "int32", 3),
    ("dyn_light_atten_q", "int32", 3),
    ("fuel_recip", "int64", 1),
    ("fire_T_ext_plane", "int32", 1),
    ("fuel_per_o2_plane", "int32", 1),
)
PLANE_NAMES = tuple(p[0] for p in PLANES)
N_PLANES = len(PLANES)

#: The tick-end clears' planes (Simulation._clear_tick_buffers, with a
#: physics runner): (GameMap field, dtype). The credit claims' clear is P7's
#: (its device form is the claims table); the shot tracers' expiry is render.
CLEAR_PLANES: tuple = (
    ("heat", "int32"),
    ("rad_net", "int64"),
    ("rad_amb", "int64"),
    ("rad_flux", "int64"),
)
CLEAR_NAMES = tuple(p[0] for p in CLEAR_PLANES)


class EffectOverflow(RuntimeError):
    """An env's effects did not fit the applier's list (its capacity)."""


class ApplierFault(RuntimeError):
    """An effect's operands left the range the applier was proved for."""


class EffectRefused(RuntimeError):
    """#29 P7a.1: an effect of a kind the applier cannot apply yet (a PAYLOAD
    record until P7c.1), or a host effect with no device kind (FACE, a cover
    chew, a spray's cells, any other world call)."""


def non_resident_planes() -> tuple:
    """The planes the topology seam writes that are NOT resident fields today
    (P8b must hold them on the device)."""
    from simulation.gamemap import GameMap
    res = set(GameMap._RESIDENT_FIELD_NAMES)
    return tuple(n for n in PLANE_NAMES + CLEAR_NAMES if n not in res)


# ---------------------------------------------------------------------------
# The layout
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    fields: tuple
    kinds: tuple
    books: tuple
    planes: tuple
    clears: tuple
    fingerprint: str
    kind_fields: tuple = ()
    stages: tuple = ()
    row_cap: int = ROW_CAP

    @property
    def width(self) -> int:
        return len(self.fields)


_LAYOUT_CACHE: dict = {}


def layout() -> Layout:
    key = (FIELDS, KINDS, BOOK_NAMES, PLANES, CLEAR_PLANES)
    hit = _LAYOUT_CACHE.get(key)
    if hit is not None:
        return hit
    text = "".join(f"field:{i}:{n}\n" for i, n in enumerate(FIELDS))
    text += "".join(f"kind:{i + 1}:{n}\n" for i, n in enumerate(KINDS))
    text += "".join(f"book:{i}:{n}\n" for i, n in enumerate(BOOK_NAMES))
    text += "".join(f"plane:{i}:{n}:{d}:{c}\n"
                    for i, (n, d, c) in enumerate(PLANES))
    text += "".join(f"clear:{i}:{n}:{d}\n"
                    for i, (n, d) in enumerate(CLEAR_PLANES))
    lay = Layout(fields=FIELDS, kinds=KINDS, books=BOOK_NAMES, planes=PLANES,
                 clears=CLEAR_PLANES,
                 fingerprint=hashlib.sha256(text.encode("ascii")).hexdigest(),
                 kind_fields=KIND_FIELDS, stages=STAGES, row_cap=ROW_CAP)
    _LAYOUT_CACHE[key] = lay
    return lay


def header_text(lay: Layout | None = None) -> str:
    """The generated C++ header's full text (``cpp/src/applier_layout.h``)."""
    lay = lay or layout()
    lines = [
        "#pragma once",
        "// ==========================================================================",
        "// GENERATED by tools/gen_unit_layout.py from src/simulation/applier.py",
        "// (#29 P6.4b / P7a.1, residency design 4.3 / 4.4b / 4.4c) -- NEVER EDIT",
        "// BY HAND. The per-env serial applier: an effect record of APPLIER_F",
        "// int64 (the kind, then the kind's own fields: APPLIER_<KIND>_<FIELD> is",
        "// a field's column), its kinds, its stages; the per-env book channels;",
        "// the topology seam's planes",
        "// (the order of the launch's pointer table) and the tick-end clears'",
        "// planes. A new kind / book / plane: a row of simulation/applier.py,",
        f"// then `{GENERATOR}` and commit; tests/test_applier_layout.py",
        "// refuses a stale copy.",
        "// ==========================================================================",
        "#include <cstdint>",
        "",
        "namespace applier {",
        "",
        f'constexpr const char* APPLIER_FINGERPRINT = "{lay.fingerprint}";',
        f"constexpr int APPLIER_F = {lay.width};",
        f"constexpr int APPLIER_ROW_CAP = {lay.row_cap};",
        "",
        "// An effect record's fields (the kind, then the arg slots).",
    ]
    for i, n in enumerate(lay.fields):
        lines.append(f"constexpr int APPLIER_FX_{n.upper()} = {i};")
    lines += ["", "// The effect kinds (0 = no record)."]
    for i, n in enumerate(lay.kinds):
        lines.append(f"constexpr int64_t APPLIER_KIND_{n.upper()} = {i + 1};")
    lines.append(f"constexpr int APPLIER_N_KINDS = {len(lay.kinds)};")
    lines += ["", "// Each kind's fields: their record column."]
    for n, fs in lay.kind_fields:
        for i, f in enumerate(fs):
            lines.append(f"constexpr int APPLIER_{n.upper()}_{f.upper()} = {1 + i};")
    lines += ["", "// The stages of the one walker; _LANDS: a HIT in the stage holds its",
              "// exposure raises and zeroings for the walk's end (the landing)."]
    for i, (n, l) in enumerate(lay.stages):
        lines.append(f"constexpr int APPLIER_STAGE_{n.upper()} = {i};")
        lines.append(f"constexpr bool APPLIER_STAGE_{n.upper()}_LANDS = "
                     f"{'true' if l else 'false'};")
    lines.append(f"constexpr int APPLIER_N_STAGES = {len(lay.stages)};")
    lines += ["", "// execute_payload's kind values (a PAYLOAD record's kind index)."]
    for i, n in enumerate(PAYLOAD_KINDS):
        lines.append(f"constexpr int64_t APPLIER_PAYLOAD_KIND_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_PAYLOAD_KINDS = {len(PAYLOAD_KINDS)};")
    lines += ["", "// The per-env book channels."]
    for i, n in enumerate(lay.books):
        lines.append(f"constexpr int APPLIER_BOOK_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_BOOKS = {len(lay.books)};")
    lines += ["", "// The topology seam's planes: the launch's pointer table order."]
    for i, (n, _d, _c) in enumerate(lay.planes):
        lines.append(f"constexpr int APPLIER_PLANE_{n.upper()} = {i};")
    lines.append(f"constexpr int APPLIER_N_PLANES = {len(lay.planes)};")
    lines += ["", "// The tick-end clears' planes (their element size in bytes)."]
    for i, (n, d) in enumerate(lay.clears):
        lines.append(f"constexpr int APPLIER_CLEAR_{n.upper()} = {i};")
        lines.append(f"constexpr int APPLIER_CLEAR_{n.upper()}_BYTES = "
                     f"{np.dtype(d).itemsize};")
    lines.append(f"constexpr int APPLIER_N_CLEARS = {len(lay.clears)};")
    lines += ["", "}  // namespace applier", ""]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Effect lists
# ---------------------------------------------------------------------------
def destroy_records(tiles) -> np.ndarray:
    """A list of ``(y, x)`` tiles as DESTROY_WALL records ``(R, F)``."""
    t = np.asarray(list(tiles), dtype=np.int64).reshape(-1, 2)
    out = np.zeros((t.shape[0], F), dtype=np.int64)
    out[:, 0] = KIND_DESTROY_WALL
    out[:, 1:3] = t
    return out


def pack(per_env, cap: int = E_CAP):
    """N envs' ``(R_e, F)`` records as the list the device walks: ``recs``
    ``(N, cap, F)`` (the first ``cap`` of each env) and ``counts`` ``(N,)``
    -- each env's REQUESTED count, which may exceed ``cap`` (the device
    applies ``cap`` and counts the rest in its ``dropped`` book)."""
    per_env = [np.asarray(r, dtype=np.int64).reshape(-1, F) for r in per_env]
    cap = int(cap)
    if cap < 1:
        raise ValueError(f"applier: cap {cap} must be >= 1")
    recs = np.zeros((len(per_env), cap, F), dtype=np.int64)
    counts = np.zeros(len(per_env), dtype=np.int64)
    for e, r in enumerate(per_env):
        k = min(cap, r.shape[0])
        recs[e, :k] = r[:k]
        counts[e] = r.shape[0]
    return recs, counts


# ---------------------------------------------------------------------------
# #29 P7a.1: the host's effects as records (the effect list's ONE record
# function, decision 8: what the shadow harness compares and feeds)
# ---------------------------------------------------------------------------
def _unwrap(fn):
    while hasattr(fn, "__wrapped__"):
        fn = fn.__wrapped__
    return fn


def _row_of(sim, unit) -> int:
    for r, u in enumerate(sim.units):
        if u is unit:
            return r
    raise EffectRefused(f"applier: an effect names a unit that is not in "
                        f"sim.units ({getattr(unit, 'id', None)!r})")


def payload_index(sim, payload) -> int:
    """A ``[payloads.*]`` row's index in ``sim.weapons_tables.payloads``."""
    names = list(sim.weapons_tables.payloads.by_name)
    name = getattr(payload, "name", None)
    if name not in names or sim.weapons_tables.payloads.by_name[name] is not payload:
        raise EffectRefused(f"applier: payload {name!r} is not a row of the "
                            f"sim's payload table")
    return names.index(name)


def record_of(sim, eff, cells=None) -> tuple:
    """ONE host effect of ``simulation.simultaneous.SimultaneousEffects``'s
    pending list as its device record (:data:`KIND_FIELDS`). A ``_WORLD``
    call is recognised by its function: ``combat.chew_wall`` is CHEW,
    ``payloads.execute_payload`` is PAYLOAD, ``combat._enqueue_spray_cells``
    is SPRAY_CELLS (#29 P7a.3: its cells appended to ``cells``, the CELL
    BUFFER's host list, the record holding their span; refused without it);
    FACE (WEGO-only), a cover chew and any other call are refused by name
    (:class:`EffectRefused`), as is an effect whose objects are not ``sim``'s
    own (another map, another impulse table, another event list)."""
    from simulation import combat, payloads, simultaneous, unit_fixed, wall_fixed
    from simulation.events import (CAUSE_NAMES, NO_ATTACKER, attacker_value,
                                   cause_value)
    kind = eff[0]
    if kind == simultaneous._HIT:
        _, unit, packet, events, source = eff
        if events is not sim.tick_events:
            raise EffectRefused("applier: a HIT whose events are not the sim's "
                                "tick_events has no device form")
        return make_record(
            "HIT", row=_row_of(sim, unit), amount_q=int(packet.q()),
            dtype=int(packet.dtype),
            attacker=attacker_value(getattr(packet, "attacker_id", NO_ATTACKER)),
            indirect=int(bool(getattr(packet, "indirect", False))),
            expose=int(getattr(packet, "expose", 0) or 0),
            dir_x_q=int(getattr(packet, "dir_x_q", 0)),
            dir_y_q=int(getattr(packet, "dir_y_q", 0)),
            ap_q=unit_fixed.quantize_scalar(float(packet.ap)),   # p3e-door: mitigate_q's
            cause=CAUSE_NAMES.index(cause_value(source)))
    if kind == simultaneous._STATUS:
        _, unit, skind, magnitude, duration, source_id = eff
        return make_record(
            "STATUS", row=_row_of(sim, unit), status=int(skind),
            magnitude_q=unit_fixed.quantize_scalar(float(magnitude)),  # p3e-door: apply_status's
            duration=int(duration),
            source=-1 if source_id is None else int(source_id))
    if kind == simultaneous._PUSH:
        _, table, unit, jx, jy, lever, twist = eff
        if table is not sim.impulse_table:
            raise EffectRefused("applier: a PUSH through another impulse table "
                                "than the sim's has no device form")
        return make_record("PUSH", row=_row_of(sim, unit), jx_q=jx, jy_q=jy,
                           lever_q=lever, twist_q16=twist)
    if kind == simultaneous._FACE:
        raise EffectRefused("applier: FACE has no device kind (its producers "
                            "are slot 4's WEGO branches, design 4.4c scope)")
    if kind != simultaneous._WORLD:
        raise EffectRefused(f"applier: no effect kind {kind!r}")
    _, fn, args, kwargs = eff
    f = _unwrap(fn)
    if f is _unwrap(combat.chew_wall):
        gmap, iy, ix, wall_damage = args
        if kwargs or gmap is not sim.gmap:
            raise EffectRefused("applier: a chew_wall on another map / with "
                                "keywords has no device form")
        live = wall_damage > 0
        return make_record(
            "CHEW", y=int(iy), x=int(ix),
            damage_q=(wall_fixed.quantize_scalar(float(wall_damage))  # p3e-door: chew_wall's
                      if live else 0),
            live=int(live))
    if f is _unwrap(payloads.execute_payload):
        gmap, queue, units, fy, fx, payload = args[:6]
        if gmap is not sim.gmap or queue is not sim.edit_queue or units is not sim.units \
                or kwargs.get("events", None) is not sim.tick_events:
            raise EffectRefused("applier: an execute_payload on another sim's "
                                "map / queue / units / events has no device form")
        pk = kwargs.get("kind", "explosion")
        if pk not in PAYLOAD_KINDS:
            raise EffectRefused(f"applier: payload kind {pk!r} is not one of "
                                f"{PAYLOAD_KINDS}")
        appr = kwargs.get("approach", None)
        return make_record(
            "PAYLOAD", y=int(fy), x=int(fx), payload=payload_index(sim, payload),
            owner=int(kwargs.get("owner_id", NO_ATTACKER)),
            approach_y=-1 if appr is None else int(appr[0]),
            approach_x=-1 if appr is None else int(appr[1]),
            has_approach=int(appr is not None), kind=PAYLOAD_KINDS.index(pk))
    if f is _unwrap(combat._enqueue_spray_cells):
        gmap, queue, owner, heat_q, gas_id, gas_q, cl = args
        if kwargs or gmap is not sim.gmap or queue is not sim.edit_queue:
            raise EffectRefused("applier: a spray's cells on another sim's map / "
                                "queue (or with keywords) have no device form")
        if cells is None:
            raise EffectRefused("applier: a spray's cells need the cell buffer "
                                "(records_and_cells_of)")
        start = len(cells)
        cells.extend((int(y), int(x), int(d)) for (y, x, d) in cl)
        return make_record(
            "SPRAY_CELLS", owner=int(owner), heat_q=int(heat_q),
            gas_id=-1 if gas_id is None else int(gas_id), gas_q=int(gas_q),
            start=start, count=len(cells) - start)
    raise EffectRefused(f"applier: the world call {getattr(f, '__qualname__', f)!r} "
                        f"has no device kind (CHEW is combat.chew_wall, PAYLOAD "
                        f"payloads.execute_payload, SPRAY_CELLS "
                        f"combat._enqueue_spray_cells)")


def records_and_cells_of(sim, pending, cell_base: int = 0):
    """#29 P7a.3: a host pending list as ``(R, F)`` int64 records in its
    order AND the cells its SPRAY_CELLS records point into (``(C, 3)`` int64
    rows ``(y, x, div)``), the spans counted from ``cell_base`` (the cells
    the env's buffer already holds)."""
    cells: list = [None] * int(cell_base)
    recs = [record_of(sim, e, cells) for e in pending]
    out = np.array(cells[int(cell_base):], dtype=np.int64).reshape(-1, 3)
    return np.array(recs, dtype=np.int64).reshape(len(recs), F), out


def records_of(sim, pending) -> np.ndarray:
    """A host pending list as ``(R, F)`` int64 records, in its order (a
    SPRAY_CELLS record's span counts from the list's first spray:
    :func:`records_and_cells_of` gives the cells)."""
    return records_and_cells_of(sim, pending)[0]


def effects_of(args, kwargs):
    """The ``SimultaneousEffects`` a slot callable was handed: ``self`` of
    ``resolve`` or the ``effects=`` of an attack slot's walk."""
    from simulation.simultaneous import SimultaneousEffects
    for a in tuple(args) + tuple(kwargs.values()):
        if isinstance(a, SimultaneousEffects):
            return a
    raise EffectRefused("applier: the slot callable was handed no "
                        "SimultaneousEffects")


# ---------------------------------------------------------------------------
# The device applier
# ---------------------------------------------------------------------------
class DeviceApplier:
    """THE per-env EFFECT LIST ``(N, cap, F)`` -- an instance of the record
    list (``UnitRecordList``: per-row staging and ONE gather per env, so a
    per-row producer's records land in row order; #29 P7a.1) -- and the book
    channels ``(N, N_BOOKS)``, on the tick stream
    (``breach_physics.FieldApplierBuffer``, the CUDA build only).

    Producers APPEND: :meth:`stage` sets every env's list from host records
    (the first ``cap`` kept, the rest counted as dropped: the host staging the
    gates use, and a chained run's host-made records); a device producer
    (slot 4a, ``simulation.unit_kernels.melee``) stages per row and gathers
    after whatever the list holds, so several producers continue one list
    before it is walked. :meth:`begin` empties it. The walk
    (``unit_kernels.apply_effects`` / ``resolve``, one kernel, a STAGE
    argument) zeroes the books, walks every env's list and downloads them.
    ``n_units`` is the store's row count (the staging's), ``row_cap`` a row's
    staging. Refuses a compiled module whose applier layout is not this
    table's."""

    def __init__(self, n_env: int, cap: int = E_CAP, bp=None, n_units: int = 1,
                 row_cap: int = ROW_CAP):
        if bp is None:
            import breach_physics as bp
        if not getattr(bp, "HAS_CUDA", False):
            raise RuntimeError("applier: the device applier needs the CUDA "
                               "build (breach_physics.HAS_CUDA)")
        lay = layout()
        compiled = bp.applier_layout()
        if compiled["fingerprint"] != lay.fingerprint                 or int(compiled["f"]) != lay.width                 or int(compiled["n_books"]) != N_BOOKS                 or int(compiled["n_planes"]) != N_PLANES:
            raise ValueError(
                f"applier: the compiled layout ({compiled['fingerprint'][:12]}) "
                f"is not simulation.applier's ({lay.fingerprint[:12]}): run "
                f"`{GENERATOR}` and rebuild cpp/build_cuda")
        self.bp = bp
        self.n_env = int(n_env)
        self.cap = int(cap)
        self.n_units = int(n_units)
        self.row_cap = int(row_cap)
        self._buf = bp.FieldApplierBuffer(self.n_env, self.cap, self.n_units,
                                          self.row_cap)

    @property
    def buffer(self):
        return self._buf

    @property
    def shape(self):
        return (self.n_env, self.cap, F)

    def begin(self) -> None:
        """Empty every env's list (counts, dropped, the staging)."""
        self._buf.begin()

    def stage(self, recs, counts) -> None:
        """Set every env's list from a packed host list (:func:`pack`):
        ``counts`` the REQUESTED counts (past ``cap`` counted as dropped)."""
        recs = np.ascontiguousarray(recs, dtype=np.int64)
        counts = np.ascontiguousarray(counts, dtype=np.int64).reshape(-1)
        if recs.shape != (self.n_env, self.cap, F) or counts.shape != (self.n_env,):
            raise ValueError(f"applier: a list must be ({self.n_env}, {self.cap}, "
                             f"{F}) with ({self.n_env},) counts, not "
                             f"{recs.shape} / {counts.shape}")
        self._buf.stage(recs, counts)

    def stage_lists(self, per_env) -> None:
        """:meth:`stage` of N envs' ``(R_e, F)`` record arrays."""
        self.stage(*pack(per_env, self.cap))

    def download(self):
        """``(records (N, cap, F), counts (N,), dropped (N,))`` int64."""
        return self._buf.download()

    def records(self) -> list:
        """Per env, its records as a list of int tuples (list order)."""
        recs, counts, _dropped = self.download()
        return [[tuple(int(x) for x in recs[e, i]) for i in range(int(counts[e]))]
                for e in range(recs.shape[0])]


def books_dict(books) -> dict:
    """``(N, N_BOOKS)`` as ``{name: (N,) int64}``."""
    b = np.asarray(books, dtype=np.int64)
    return {n: b[:, i].copy() for i, n in enumerate(BOOK_NAMES)}


def host_book(gmap, ref: str) -> int:
    """The host quantity a book twins (``attr`` or ``attr:key``)."""
    if ":" in ref:
        attr, key = ref.split(":", 1)
        return int(getattr(gmap, attr).get(key, 0))
    return int(getattr(gmap, ref))


__all__ = ["HEADER_REL", "GENERATOR", "FIELDS", "F", "KINDS",
           "KIND_DESTROY_WALL", "E_CAP", "BookRow", "BOOKS", "BOOK_NAMES",
           "N_BOOKS", "PLANES", "PLANE_NAMES", "N_PLANES", "CLEAR_PLANES",
           "CLEAR_NAMES", "EffectOverflow", "ApplierFault", "EffectRefused",
           "ARG_SLOTS", "KIND_FIELDS", "KIND_HIT", "KIND_STATUS", "KIND_PUSH",
           "KIND_CHEW", "KIND_PAYLOAD", "UNAPPLIED_KINDS", "PAYLOAD_KINDS",
           "STAGES", "STAGE_NAMES", "STAGE_TOPOLOGY", "STAGE_RESOLVE", "ROW_CAP",
           "kind_field", "make_record", "record_of", "records_of", "effects_of",
           "payload_index", "KIND_SPRAY_CELLS", "records_and_cells_of",
           "non_resident_planes", "Layout", "layout", "header_text",
           "destroy_records", "pack", "DeviceApplier", "books_dict",
           "host_book"]
