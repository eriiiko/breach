"""#29 P6 -- THE SHADOW-LOCKSTEP HARNESS (design §4.4b decision 4): a device
slot kernel checked against the reference slot, row for row, on live matches;
and the CHAINED form (decision 4's third), adjacent device slots replacing the
host's in a twin match that must play on equal to an untouched reference.

SHADOW LOCKSTEP (:class:`ShadowLockstep`). The reference plays every tick in
Python as always; the device NEVER drives the game. Around a named slot (a
``tools/tick_stage_profile.py`` label -- the twin manifest's own vocabulary)
the harness, per env and per tick:

1. packs the env's units (``simulation.unit_store.pack``) when the slot is
   entered -- the rows the slot starts from -- and captures the slot's
   declared INPUTS at that moment (:data:`INPUTS`: the tick, a per-tick host
   table such as the order requests, field planes);
2. lets the host run the slot;
3. packs again when the slot returns -- the rows the slot ended with;

and once every env has ticked, uploads the batch's start rows to the
``DeviceStore`` (built with the envs' device tables,
``simulation.unit_tables.values_batch``), runs the slot's device kernel over
all N envs with the inputs, downloads and compares with the host's end rows at
tol 0, reporting the first difference as (tick, env, unit id, column). It also
runs each env's start rows ALONE through an N = 1 store (``check_alone``: that
env's tables and inputs) and requires the batch's env i to equal it (env i of
N == env i alone), and it counts the rows the host changed, so a kernel's
coverage counter can be held to it.

Hooking only READS the units and the inputs around the original callable, so
a hooked match plays exactly as an unhooked one.

THE API the later P6 patches extend::

    slot = DeviceSlot(label="1c exposure decay", run=unit_kernels.exposure_decay)
    slot = DeviceSlot(label="1b' orders",
                      run=lambda store, inp: unit_kernels.apply_orders(
                          store, inp.tables["order_requests"]),
                      inputs=("order_requests",))
    slot = DeviceSlot(label="9c2 wind on units",
                      run=lambda store, inp: unit_kernels.wind_impulse(
                          store, inp.fields),
                      inputs=("field:gas", "field:wind_x", "field:wind_y",
                              "field:solid"))
    with ShadowLockstep(drivers, slot) as ls:      # drivers: .sim + .tick()
        for _ in range(T):
            rep = ls.tick()                        # a TickReport
    ls.reports, ls.divergences

    with ShadowLockstep(drivers, [slot_a, slot_b]) as ls:   # several slots:
        reps = ls.tick()                           # {label: TickReport}

``run(store)`` when the slot declares no inputs (P6.1's form), else
``run(store, inputs)`` with a :class:`SlotInputs`: ``inputs.tick`` ``(N,)``
int64 (each env's ``sim.tick`` at the slot's entry), ``inputs.tables[name]``
``(N, R, C)`` int64 (a per-tick host table; its capture function pads the
short envs), ``inputs.fields[name]`` a DEVICE array ``(N, ...)`` in the
resident field's own dtype (a CuPy array of the env's ``GameMap.<name>`` at the
slot's entry, stacked -- the shape ``ResidentBatch.buffers`` holds, so the
device backend passes its own). A new input kind is a row of :data:`INPUTS`
(a capture function of the sim) or a ``field:<GameMap attribute>``.

REQUESTS (#29 P6.4a): a slot whose kernel does something that is not a row
write (``DeviceSlot(..., requests=True)``: slot 1b -- a grenade released, a
charge laid, a door flipped) is handed a ``simulation.unit_requests.
RequestList`` the harness begins before the kernel (after the ring in the
call: ``run(store[, inputs][, ring], requests)``); while the host slot runs the
harness records, in the host's order, each grenade ``Simulation.
_release_grenade`` appended, each charge offered to ``charges.place`` with its
answer and each door ``Simulation._direct_use`` flipped (read through the
verb's own ``verb_gates.use_door``), as request records, and compares them
with the device's list record for record (the first difference as (tick, env,
request index, field)); the alone run's list must equal the batch's env i.
In the CHAINED form the list is turned into the host's objects
(``unit_requests.apply_to_host``: the projectiles appended, each charge
offered to ``charges.place`` -- whose answer must be the record's -- and the
doors flipped) right after the unpack. Inputs ``door_spans`` (a table) and
``charges_live`` (a SCALAR input: ``SlotInputs.scalars[name]``, ``(N,)``).

EVENTS (#29 P6.3): a slot whose kernel emits (``DeviceSlot(...,
events=True)``: the slots that hurt, over the device ``apply_packet``) is
handed an :class:`simulation.unit_events.EventRing` the harness begins before
the kernel -- ``run(store, ring)``, or ``run(store, inputs, ring)`` with inputs
-- and the harness compares the RING with the events the host appended to
``sim.tick_events`` during the slot (from its first entry to its last exit),
record for record, in order (``unit_events.record_of``: the kind, the unit,
the Q16 damage, the cause, the attacker, the direction): the first difference
is named as (tick, env, event index, field); the alone run's ring must equal
the batch's env i. ``TickReport.events`` / ``host_events`` are the two counts.
The claims of the tick's credit (``damage.TickCredit``) are the input
``credit_claims`` (``unit_kernels.claims_of`` / ``pad_claims``).

LISTS (#29 P7a.1, design 4.4c decision 8): a host LIST a slot reads or writes
(today the attack slot's effects, ``SimultaneousEffects``' pending list) is
declared like a field -- ``inputs=("list:effects",)`` and / or
``outputs=("list:effects",)`` -- through ONE record function per list
(:data:`LISTS`: ``simulation.applier.records_of`` for the effects). The
harness owns the device list (a ``simulation.applier.DeviceApplier`` per
name, the alone run's beside it) and hands the slot ``lists`` (a dict, the
call's last argument): an INPUT list is staged from the host's records at
the slot's entry (so slot 4c is fed the host's list), an output-only list
begun empty; an OUTPUT list is downloaded after the kernel and compared with
the host's records at the slot's exit, record for record and field for
field (the first difference as (tick, env, effect index, kind.field)); the
alone run's list must equal the batch's env i. A host effect with no device
kind (FACE, a cover chew, a spray's cells) is refused by name at the record
function. ``DeviceSlot.select(sim, args, kwargs, caller)`` keeps only the
calls of the label's callables that belong to the slot (``caller`` the
calling function's name: slot 4c's ``resolve`` is the conductor's, not slot
2's landing). The CHAINED form stages a chain's input lists at its entry,
empties the host's list once the device has applied it, and refuses by name
a later chained callable that finds the host appended to it in between.

#29 P7b: the OBJECT TABLES are lists too (``grenades``: ``sim.projectiles``,
``charges``: ``sim.planted_charges``, each through ``object_tables.records``;
the device list a ``DeviceTable``): staged as inputs, compared as outputs and,
in the chained form, LANDED back onto the host list in place
(``ListDef.land``). ``payloads`` is a LOGGED list: the ``execute_payload``
calls at the P7b call sites (:data:`PAYLOAD_SITES`, a tap installed while a
slot names it), emptied at the slot's first entry (``ListDef.mark``),
recorded through ``applier.record_of``; the chained form replays the device's
records through the host's call (:func:`payload_call`). Output lists are kept
per name. ``DeviceSlot(skip_ok=True)``: a slot the conductor gates off
(``charges.tick`` while no charge lies) is compared, in an env where it did
not run, as a run on the state the tick ended with that must change nothing.

EVENTS keep only SYNCED events: the host's render-only events (MeleeEvent,
ShotFired, SprayJet, ...: outside every digest, no device form) are dropped
before the ring is compared (``unit_events.is_synced``).

``DeviceSlot.columns`` limits the compare to named columns (default: the
whole row). A label naming several callables is hooked on each: the first
entry in a tick packs the start rows and captures the inputs, every exit
repacks the end rows. The slot must run once in every env every tick (a slot
gated off in an env raises, naming it).

CHAINED (:class:`ChainedSlots`): in ONE twin sim, the host callables of
adjacent slots (``DeviceSlot`` s in slot order) are replaced -- the first
slot's by: capture its inputs, pack, upload, run every slot's kernel in a row
(no host pack between), download, ``unpack`` onto the units, then the
caller's ``host_side(sim, pre_rows, post_rows)`` (what the skipped host slots
did that is not unit state: clearing the request table, a converted unit's
name); every other slot's by nothing. When a chained slot emits, the chain's
ring records are appended to ``sim.tick_events`` as host events
(``unit_events.event_of``) right after the unpack -- where the first slot's
host callable would have appended them -- so the match's per-tick unit digest
(which hashes them) stays on the reference's path. Any other sim (the reference match in
the same process) runs the original callables. The match plays on; the gate
compares its per-tick unit hash and field digest with an untouched reference.

Not a test module (no ``test_`` prefix): imported by the ``cuda_unit_*_check``
scripts inside the CUDA subprocess.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from simulation import unit_events, unit_kernels, unit_store, unit_tables


# ---------------------------------------------------------------------------
# Slot inputs
# ---------------------------------------------------------------------------
FIELD_PREFIX = "field:"


def _capture_tick(sim):
    return int(sim.tick)


def _capture_requests(sim):
    return unit_kernels.requests_of(sim)


def _capture_claims(sim):
    return unit_kernels.claims_of(sim)


def _capture_door_spans(sim):                      # #29 P6.4a
    from simulation import unit_requests
    return unit_requests.door_spans(sim)


def _capture_charges_live(sim):                    # #29 P6.4a
    from simulation import unit_requests
    return unit_requests.charges_live(sim)


def _capture_draw_key(sim):                        # #29 P7a.2
    from simulation import unit_draws
    return unit_draws.draw_key(sim)


def _pad_door_spans(vals):
    from simulation import unit_requests
    return unit_requests.pad_door_spans(vals)


#: name -> (kind, capture function of the sim at the slot's entry). kind
#: "tick" gathers to ``SlotInputs.tick``; "table" to ``SlotInputs.tables``
#: (an ``(R, C)`` int64 per env, padded across envs by ``pad``); "scalar"
#: (#29 P6.4a) to ``SlotInputs.scalars[name]`` (an ``(N,)`` int64).
INPUTS: dict = {
    "tick": ("tick", _capture_tick),
    "order_requests": ("table", _capture_requests),
    "credit_claims": ("table", _capture_claims),     # #29 P6.3
    "door_spans": ("table", _capture_door_spans),    # #29 P6.4a
    "charges_live": ("scalar", _capture_charges_live),   # #29 P6.4a
    "draw_key": ("table", _capture_draw_key),        # #29 P7a.2: (N, 3)
}
#: How a table input pads across envs (default: zeros).
PADS: dict = {"order_requests": unit_kernels.pad_requests,
              "credit_claims": unit_kernels.pad_claims,
              "door_spans": _pad_door_spans}


# #29 P6.4b: inputs read from the slot callable's ARGUMENTS (a slot whose
# input is handed to it by the conductor, not held on the sim).
def _capture_walls(sim, args, kw):
    # Simulation._burn_through_walls(self, destroyed): slot 7's list, in order
    destroyed = args[1] if len(args) > 1 else kw.get("destroyed", ())
    return unit_kernels.walls_of(destroyed)


#: name -> (kind, capture function of (sim, args, kwargs) at the slot's entry).
ARG_INPUTS: dict = {
    "burn_walls": ("table", _capture_walls),
}
PADS["burn_walls"] = unit_kernels.pad_walls

# ---------------------------------------------------------------------------
# #29 P7a.1: host LISTS (decision 8), declared like a field
# ---------------------------------------------------------------------------
LIST_PREFIX = "list:"


@dataclass(frozen=True)
class ListDef:
    """One host list the harness knows: ``host(sim, args, kw)`` the host
    object a slot callable was handed; ``records(sim, obj)`` its ONE record
    function -- the object's current content as ``(R, F)`` int64 device
    records, in its order; ``consume(obj)`` empties it (the chained form,
    once the device applied it); ``device(n_env, n_units, cap, bp)`` the
    device list; ``fields(rec)`` names a record's fields for a report.
    #29 P7b: ``mark(sim)`` (optional) runs at the slot's first entry each
    tick, before the host slot (a log of what the slot does, such as the
    PAYLOAD calls, starts empty there); ``land(sim, obj, recs)`` (optional)
    is the chained form's write-back of the device's records onto the host
    in place of ``consume`` (an OBJECT TABLE's list rebuilt from the
    device's rows; a PAYLOAD record replayed through the host's call)."""
    host: Callable
    records: Callable
    consume: Callable
    device: Callable
    fields: Callable
    mark: Callable | None = None
    land: Callable | None = None


def _effects_host(sim, args, kw):
    from simulation import applier
    return applier.effects_of(args, kw or {})


def _effects_records(sim, obj):
    from simulation import applier
    return applier.records_of(sim, obj._pending)


def _effects_consume(obj):
    obj._pending.clear()


def _effects_device(n_env, n_units, cap, bp):
    from simulation import applier
    return applier.DeviceApplier(n_env, cap, bp=bp, n_units=n_units)


def _effects_fields(rec):
    from simulation import applier
    k = int(rec[0])
    if not 1 <= k <= len(applier.KINDS):
        return f"kind {k}", ("kind",) + tuple(f"a{i}" for i in range(applier.ARG_SLOTS))
    name, fs = applier.KIND_FIELDS[k - 1]
    return name, ("kind",) + fs


# #29 P7a.2: the rounds in flight (Simulation.bullets) and the noise log
# (GameMap.noise_log), each an object table (simulation.object_tables)
def _rounds_host(sim, args, kw):
    return sim


def _rounds_records(sim, obj):
    return unit_kernels.rounds_of(sim)


def _rounds_device(n_env, n_units, cap, bp):
    from simulation import object_tables, unit_schema
    return object_tables.DeviceTable(unit_schema.ROUND_TABLE, n_env, n_units=n_units,
                                     row_cap=unit_kernels.round_row_cap(), bp=bp)


def _rounds_land(sim, obj, recs):
    sim.bullets = unit_kernels.rounds_to_host(sim, recs)


def _table_fields(name):
    def f(rec):
        from simulation import object_tables
        lay = object_tables.table_layout(name)
        return name, tuple(c.name for c in lay.columns)
    return f


def _noise_host(sim, args, kw):
    return sim


def _noise_records(sim, obj):
    return unit_kernels.noise_of(sim)


def _noise_device(n_env, n_units, cap, bp):
    from simulation import object_tables, unit_schema
    return object_tables.DeviceTable(unit_schema.NOISE_TABLE, n_env, n_units=n_units,
                                     row_cap=1, bp=bp)


def _noise_land(sim, obj, recs):
    sim.gmap.noise_log[:] = [(int(r[0]), int(r[1]), int(r[2])) for r in recs]


def _no_consume(obj):
    pass


# -- #29 P7b: the OBJECT TABLES' host lists (``Simulation.projectiles``,
# ``Simulation.planted_charges``) and the PAYLOAD calls of the grenade and
# charge slots (``execute_payload`` at its two P7b call sites, recorded by a
# tap while a slot naming ``list:payloads`` runs) --------------------------
def _table_list(attr, table_name):
    def _schema():
        from simulation import unit_schema
        return {t.name: t for t in unit_schema.OBJECT_TABLES}[table_name]

    def host(sim, args, kw):
        return getattr(sim, attr)

    def records(sim, obj):
        from simulation import object_tables
        t = _schema()
        w = object_tables.table_layout(t).width
        return np.array(object_tables.records(t, obj),
                        dtype=np.int64).reshape(-1, w)

    def consume(obj):
        raise RuntimeError(f"lockstep: the {table_name!r} table is landed, "
                           f"never consumed")

    def device(n_env, n_units, cap, bp):
        from simulation import object_tables
        return object_tables.DeviceTable(_schema(), n_env, bp=bp)

    def fields(rec):
        from simulation import object_tables
        return table_name, tuple(
            c.name if c.width == 1 else f"{c.name}[{j}]"
            for c in object_tables.table_layout(_schema()).columns
            for j in range(c.width))

    def land(sim, obj, recs):
        # the host list becomes the device's rows, IN PLACE (one list object
        # for the sim's life, as the slot-10 prune keeps it)
        from simulation import object_tables
        t = _schema()
        w = object_tables.table_layout(t).width
        rows = np.array(recs, dtype=np.int64).reshape(-1, w)
        getattr(sim, attr)[:] = object_tables.unpack(t, rows, rows.shape[0])

    return ListDef(host, records, consume, device, fields, None, land)


#: The PAYLOAD tap's log: id(gmap) -> the execute_payload calls (as
#: SimultaneousEffects world effects) since the slot's mark.
_PAYLOAD_LOG: dict = {}
#: The P7b call sites of execute_payload: the modules whose global the tap
#: wraps (slot 2's grenades; charges.fire, slot 9e(d')).
PAYLOAD_SITES = ("simulation.simulation", "simulation.charges",
                 # #29 P6b.4: the timed charges' sweep and a heat source's death
                 "simulation.timed_charge_system", "simulation.heat_source_system")
#: The LOGGED lists of PAYLOAD calls (the tap is installed while a slot names
#: one): P7b's slot-local log, #29 P6b.4 the TIMED_CHARGES / HEAT_SOURCES
#: stages' own lists of what they applied.
PAYLOAD_LOGS = ("payloads", "timed_payloads", "heat_payloads")


def install_payload_tap() -> list:
    """Wrap ``execute_payload`` at :data:`PAYLOAD_SITES`: a call on a marked
    map is logged (then made). Returns ``(module, attr, original)`` rows for
    the uninstall. Reads only."""
    from simulation import simultaneous
    out = []
    for name in PAYLOAD_SITES:
        mod = importlib.import_module(name)
        orig = mod.execute_payload

        def tapped(*a, __o=orig, **kw):
            log = _PAYLOAD_LOG.get(id(a[0])) if a else None
            if log is not None:
                log.append((simultaneous._WORLD, __o, tuple(a), dict(kw)))
            return __o(*a, **kw)
        tapped.__wrapped__ = orig
        mod.execute_payload = tapped
        out.append((mod, "execute_payload", orig))
    return out


def _payloads_host(sim, args, kw):
    return _PAYLOAD_LOG.setdefault(id(sim.gmap), [])


def _payloads_mark(sim):
    _PAYLOAD_LOG[id(sim.gmap)] = []


def _payloads_records(sim, obj):
    from simulation import applier
    return applier.records_of(sim, obj)


def applier_width() -> int:
    from simulation import applier
    return applier.F


def payload_call(sim, rec):
    """A PAYLOAD record as the host call it records: ``(args, kwargs)`` of
    ``payloads.execute_payload`` on ``sim`` (the inverse of
    ``applier.record_of``'s PAYLOAD branch)."""
    from simulation import applier
    r = [int(x) for x in rec]
    if r[0] != applier.KIND_PAYLOAD:
        raise RuntimeError(f"lockstep: a kind {r[0]} record on the PAYLOAD list")
    f = {n: r[applier.kind_field("PAYLOAD", n)]
         for n in dict(applier.KIND_FIELDS)["PAYLOAD"]}
    names = list(sim.weapons_tables.payloads.by_name)
    args = (sim.gmap, sim.edit_queue, sim.units, f["y"], f["x"],
            sim.weapons_tables.payloads.by_name[names[f["payload"]]], sim.draws)
    kw = dict(events=sim.tick_events, kind=applier.PAYLOAD_KINDS[f["kind"]],
              owner_id=f["owner"],
              approach=((f["approach_y"], f["approach_x"]) if f["has_approach"]
                        else None))
    return args, kw


def _payloads_land(sim, obj, recs):
    """The device's PAYLOAD records replayed IN ORDER through the host's
    ``execute_payload`` (the request list's ``apply_to_host`` pattern): each
    call rebuilt from its record, which must reproduce the record."""
    from simulation import applier, payloads, simultaneous
    for r in recs:
        args, kw = payload_call(sim, r)
        again = applier.record_of(sim, (simultaneous._WORLD,
                                        payloads.execute_payload, args, kw))
        if tuple(again) != tuple(int(x) for x in r):
            raise RuntimeError(f"lockstep: a replayed PAYLOAD {again} is not "
                               f"its record {tuple(r)}")
        payloads.execute_payload(*args, **kw)


#: name -> ListDef.
LISTS: dict = {
    "effects": ListDef(_effects_host, _effects_records, _effects_consume,
                       _effects_device, _effects_fields),
    "rounds": ListDef(_rounds_host, _rounds_records, _no_consume, _rounds_device,
                      _table_fields("round"), None, _rounds_land),
    "noise": ListDef(_noise_host, _noise_records, _no_consume, _noise_device,
                     _table_fields("noise"), None, _noise_land),
    "grenades": _table_list("projectiles", "grenade"),          # #29 P7b
    "charges": _table_list("planted_charges", "charge"),        # #29 P7b
    "payloads": ListDef(_payloads_host, _payloads_records,      # #29 P7b
                        lambda obj: obj.clear(), _effects_device,
                        _effects_fields, _payloads_mark, _payloads_land),
}


# -- #29 P7a.3: THE CELL BUFFER (the spray cells the effect list's SPRAY_CELLS
# spans point into), THE DEVICE EDIT DOOR's rows (the edits the host's
# producer enqueued in the slot, gated at the door: decision 7) and THE CLAIMS
# LIST (every TickCredit.claim of the slot, recorded by a tap) ---------------
def _cells_records(sim, obj):
    from simulation import attack_lists
    return attack_lists.cells_of(sim, obj._pending)


def _cells_device(n_env, n_units, cap, bp):
    from simulation import attack_lists
    return attack_lists.cell_buffer(n_env, bp=bp)


#: id(gmap) -> the queue's length at the slot's mark (the producer's first edit).
_EDIT_MARK: dict = {}
#: The spray producer's edit sources (combat._SRC_SPRAY_HEAT / _GAS): an edit
#: of another source inside the walk is an interleave, refused by name.
DOOR_SOURCES = None


def _door_sources():
    """The device producers' edit sources: the spray's (combat._SRC_SPRAY_*)
    and, #29 P7c.1, the payload function's (physics._SRC_EXPLOSION,
    payloads._SRC_PAYLOAD_GAS / _IGNITE / _HEAT; #29 P7c.2 the blast smoke,
    physics._SRC_EXPLOSION_SMOKE). An edit of any other source inside a gated
    walk is an interleave, refused by name."""
    from simulation import combat, payloads, physics
    return {int(combat._SRC_SPRAY_HEAT), int(combat._SRC_SPRAY_GAS),
            int(physics._SRC_EXPLOSION), int(payloads._SRC_PAYLOAD_GAS),
            int(payloads._SRC_PAYLOAD_IGNITE), int(payloads._SRC_PAYLOAD_HEAT),
            int(physics._SRC_EXPLOSION_SMOKE)}


#: The tags of a combined door record (#29 P7c.1: a DISC edit has many tile
#: rows, so each edit is its EDIT record followed by one record per tile row).
DOOR_TAG_EDIT, DOOR_TAG_TILE = 0, 1


def door_combined(edits, tiles):
    """The door's rows as ONE record list: per edit, in order, its EDIT row
    (tag 0, the row) then each of its tile rows in their order (tag 1, the
    TILE row, zero-padded to the EDIT row's width)."""
    from simulation import edit_door
    E, T = edit_door.EDIT_COLS, edit_door.TILE_COLS
    out = []
    edits = np.asarray(edits, dtype=np.int64).reshape(-1, E)
    tiles = np.asarray(tiles, dtype=np.int64).reshape(-1, T)
    for k in range(edits.shape[0]):
        out.append((DOOR_TAG_EDIT,) + tuple(int(x) for x in edits[k]))
        for t in tiles[tiles[:, 0] == k]:
            out.append((DOOR_TAG_TILE,) + tuple(int(x) for x in t) + (0,) * (E - T))
    return np.array(out, dtype=np.int64).reshape(len(out), 1 + E)


def door_split(recs):
    """The inverse of :func:`door_combined`: ``(edits, tiles)``."""
    from simulation import edit_door
    E, T = edit_door.EDIT_COLS, edit_door.TILE_COLS
    r = np.asarray(recs, dtype=np.int64).reshape(-1, 1 + E)
    edits = r[r[:, 0] == DOOR_TAG_EDIT][:, 1:]
    tiles = r[r[:, 0] == DOOR_TAG_TILE][:, 1:1 + T]
    return edits.copy(), tiles.copy()


def _edits_mark(sim):
    _EDIT_MARK[id(sim.gmap)] = len(sim.edit_queue._edits)


def _edits_records(sim, obj):
    from simulation import edit_door
    n0 = _EDIT_MARK.get(id(sim.gmap), 0)
    e, t = edit_door.door_rows(sim.gmap, obj._edits[n0:], _door_sources())
    return door_combined(e, t)


class DoorList:
    """The device edit door as a harness list (its rows combined per edit)."""

    def __init__(self, n_env, bp=None):
        from simulation import edit_door
        self.door = edit_door.DeviceEditDoor(n_env, bp=bp)
        self.n_env = n_env

    @property
    def buffer(self):
        return self.door.buffer

    def begin(self):
        self.door.begin()

    def set_seq(self, seq):
        self.door.set_seq(seq)

    def check(self, what=""):
        self.door.check(what)

    def records(self):
        return [[tuple(int(x) for x in r) for r in door_combined(e, t)]
                for e, t in self.door.rows()]


def _edits_device(n_env, n_units, cap, bp):
    return DoorList(n_env, bp=bp)


def _edits_land(sim, obj, recs):
    from simulation import edit_door
    e, t = door_split(recs)
    edit_door.replay(sim, e, t)


def _edits_fields(rec):
    from simulation import edit_door, field_edit as fe
    if int(rec[0]) == DOOR_TAG_TILE:
        return "tile", ("tag", "t.edit", "t.tile", "t.weight", "t.scale") + tuple(
            f"pad{i}" for i in range(edit_door.EDIT_COLS - edit_door.TILE_COLS))
    names = ("env", "field", "slice", "mode", "amount_q", "den", "noisy", "low",
             "key", "clamp_flags", "lo", "hi", "o2_lo", "o2_hi", "n2_lo", "n2_hi")
    assert len(names) == fe.EDIT_COLS
    return "edit", ("tag",) + names + ("source_id", "region", "arg0", "arg1", "arg2",
                                       "arg3")


#: id(TickCredit) -> the claims of the marked slot, every call in order.
_CLAIM_LOG: dict = {}


def install_claims_tap() -> list:
    """Wrap ``damage.TickCredit.claim``: a claim on a marked TickCredit is
    logged per tile, as the device's claims list holds it ((channel, y, x,
    attacker); the environment's never), then made. Reads only."""
    from simulation import damage
    from simulation.events import NO_ATTACKER, attacker_value
    from simulation.unit_events import CREDIT_HEAT
    orig = damage.TickCredit.claim

    def claim(self, channel, tiles, attacker_id, __o=orig):
        # #29 P7c.1: the tiles may be a GENERATOR (payloads._claim_disc): read
        # once into a list, which both the log and the claim then walk (the
        # tap consumed it before and the host claimed nothing: a hooked match
        # that differed from an unhooked one)
        tiles = list(tiles)
        log = _CLAIM_LOG.get(id(self))
        if log is not None:
            a = attacker_value(attacker_id)
            if a != NO_ATTACKER:
                ch = CREDIT_HEAT if channel == damage.CREDIT_HEAT else int(channel)
                for (y, x) in tiles:
                    log.append((ch, int(y), int(x), int(a)))
        return __o(self, channel, tiles, attacker_id)
    claim.__wrapped__ = orig
    damage.TickCredit.claim = claim
    return [(damage.TickCredit, "claim", orig)]


def _claims_host(sim, args, kw):
    from simulation.damage import tick_credit
    return _CLAIM_LOG.setdefault(id(tick_credit(sim.gmap)), [])


def _claims_mark(sim):
    from simulation.damage import tick_credit
    _CLAIM_LOG[id(tick_credit(sim.gmap))] = []


def _claims_records(sim, obj):
    return np.array(obj, dtype=np.int64).reshape(len(obj), 4)


def _claims_device(n_env, n_units, cap, bp):
    from simulation import attack_lists
    return attack_lists.claims_list(n_env, bp=bp)


def _claims_land(sim, obj, recs):
    from simulation import attack_lists
    attack_lists.replay_claims(sim.gmap, recs)


LISTS["spray_cells"] = ListDef(_effects_host, _cells_records, lambda obj: None,
                               _cells_device, lambda rec: ("cell", ("y", "x", "div")))
LISTS["edits"] = ListDef(lambda sim, a, k: sim.edit_queue, _edits_records,
                         lambda obj: None, _edits_device, _edits_fields,
                         _edits_mark, _edits_land)
LISTS["claims"] = ListDef(_claims_host, _claims_records, lambda obj: obj.clear(),
                          _claims_device,
                          lambda rec: ("claim", ("channel", "y", "x", "attacker")),
                          _claims_mark, _claims_land)
INPUTS["edit_seq"] = ("scalar", lambda sim: int(sim.edit_queue._seq))


# -- #29 P7c.1: THE BLAST LOG (GameMap.blast_log, an object table) and the
# DETONATIONS (the PAYLOAD calls of slot 2 / 9e(d') as the device APPLIES
# them: the payloads list's records, consumed -- never replayed -- by the
# chained form, so the host's execute_payload runs 0 times there) ---------
def _blast_records(sim, obj):
    return unit_kernels.blast_log_of(sim)


def _blast_device(n_env, n_units, cap, bp):
    from simulation import object_tables, unit_schema
    return object_tables.DeviceTable(unit_schema.BLAST_LOG_TABLE, n_env, n_units=n_units,
                                     row_cap=1, bp=bp)


def _blast_land(sim, obj, recs):
    sim.gmap.blast_log[:] = [tuple(int(x) for x in r[:4]) for r in recs]


LISTS["blast_log"] = ListDef(_noise_host, _blast_records, _no_consume, _blast_device,
                             _table_fields("blast"), None, _blast_land)
#: A chained run's PAYLOAD records of refused rows (applier.PAYLOAD_REFUSALS)
#: replayed on the host (the ship's frag): set by a gate, empty otherwise.
REPLAY_REFUSED: dict = {"on": False}


def _detonations_land(sim, obj, recs):
    """The chained form's landing of the detonations: the device applied
    every record of a row it ports, so nothing is replayed; a record of a
    REFUSED row is replayed through the host's call only when the gate asks
    (``REPLAY_REFUSED``) and only alone in its walk (a refused row between
    applied ones would land out of order: refused by name)."""
    from simulation import applier
    obj.clear()
    if not len(recs):
        return
    pc = applier.kind_field("PAYLOAD", "payload")
    names = list(sim.weapons_tables.payloads.by_name)
    codes = [applier.payload_refusal(sim.weapons_tables.payloads.by_name[names[int(r[pc])]])
             for r in recs]
    if not any(codes):
        return
    if not REPLAY_REFUSED["on"]:
        raise RuntimeError(f"lockstep: a refused PAYLOAD row in a chained walk: "
                           f"{[names[int(r[pc])] for r, c in zip(recs, codes) if c]}")
    if not all(codes):
        raise RuntimeError("lockstep: refused and applied PAYLOAD records in one walk "
                           "(the replay would land out of order)")
    _payloads_land(sim, obj, recs)


LISTS["detonations"] = ListDef(_payloads_host, _payloads_records, lambda obj: obj.clear(),
                               _effects_device, _effects_fields, _payloads_mark,
                               _detonations_land)

# ---------------------------------------------------------------------------
# #29 P6b.1: THE DOOR TABLE and its tile side table as lists (the first
# entity table: ``Simulation._doors`` through ``unit_kernels.doors_of`` /
# ``door_tiles_of``). Staged as inputs, compared as outputs and, in the
# chained form, LANDED back onto the sim's own door runtimes in place
# (``unit_kernels.doors_to_host``): the tile table lands first (a stash), the
# door table then lands both (a chain naming only the doors -- slot 1b's use
# write -- lands them over the host's own tiles).
# ---------------------------------------------------------------------------
_DOOR_TILE_STASH: dict = {}


def _sim_host(sim, args, kw):
    return sim


def _door_device(table_name):
    def device(n_env, n_units, cap, bp):
        from simulation import object_tables, unit_schema
        t = {x.name: x for x in unit_schema.OBJECT_TABLES}[table_name]
        return object_tables.DeviceTable(t, n_env, bp=bp)
    return device


def _doors_records(sim, obj):
    return unit_kernels.doors_of(sim)


def _door_tiles_records(sim, obj):
    return unit_kernels.door_tiles_of(sim)


def _door_tiles_land(sim, obj, recs):
    _DOOR_TILE_STASH[id(sim)] = [tuple(int(x) for x in r) for r in recs]


def _doors_land(sim, obj, recs):
    tiles = _DOOR_TILE_STASH.pop(id(sim), None)
    if tiles is None:
        tiles = unit_kernels.door_tiles_of(sim)
    unit_kernels.doors_to_host(sim, recs, tiles)


def _door_consume(obj):
    raise RuntimeError("lockstep: the door tables are landed, never consumed")


LISTS["doors"] = ListDef(_sim_host, _doors_records, _door_consume, _door_device("door"),
                         _table_fields("door"), None, _doors_land)
LISTS["door_tiles"] = ListDef(_sim_host, _door_tiles_records, _door_consume,
                              _door_device("door_tile"), _table_fields("door_tile"), None,
                              _door_tiles_land)

# #29 P6b.2: THE DUCT TABLE and THE VENT TABLE as lists (``Simulation._ducts``
# / ``_vents`` through ``unit_kernels.ducts_of`` / ``vents_of``): staged as
# inputs, compared as outputs and, in the chained form, LANDED back onto the
# sim's own runtimes in place (``unit_kernels.ducts_to_host`` /
# ``vents_to_host``).
def _ducts_records(sim, obj):
    return unit_kernels.ducts_of(sim)


def _vents_records(sim, obj):
    return unit_kernels.vents_of(sim)


def _ducts_land(sim, obj, recs):
    unit_kernels.ducts_to_host(sim, recs)


def _vents_land(sim, obj, recs):
    unit_kernels.vents_to_host(sim, recs)


def _vent_consume(obj):
    raise RuntimeError("lockstep: the duct / vent tables are landed, never consumed")


def _wide_fields(name):
    # a multi-wide column's ints named one by one (trace_raw[0] ...)
    def f(rec):
        from simulation import object_tables
        return name, tuple(c.name if c.width == 1 else f"{c.name}[{j}]"
                           for c in object_tables.table_layout(name).columns
                           for j in range(c.width))
    return f


# #29 P7c.2: THE BLAST-SOURCE TABLE and its tile side table as lists
# (``GameMap.expansions`` through ``unit_kernels.sources_of`` /
# ``source_tiles_of``): staged as inputs, compared as outputs and, in the
# chained form, LANDED back onto the map's own list in place
# (``unit_kernels.sources_to_host``): the tile table lands first (a stash),
# the source table then lands both.
_SOURCE_TILE_STASH: dict = {}


def _sources_records(sim, obj):
    return unit_kernels.sources_of(sim)


def _source_tiles_records(sim, obj):
    return unit_kernels.source_tiles_of(sim)


def _source_tiles_land(sim, obj, recs):
    _SOURCE_TILE_STASH[id(sim)] = [tuple(int(x) for x in r) for r in recs]


def _sources_land(sim, obj, recs):
    tiles = _SOURCE_TILE_STASH.pop(id(sim), None)
    if tiles is None:
        tiles = unit_kernels.source_tiles_of(sim)
    unit_kernels.sources_to_host(sim, recs, tiles)


def _source_consume(obj):
    raise RuntimeError("lockstep: the blast-source tables are landed, never consumed")


LISTS["sources"] = ListDef(_sim_host, _sources_records, _source_consume,
                           _door_device("source"), _table_fields("source"), None,
                           _sources_land)
LISTS["source_tiles"] = ListDef(_sim_host, _source_tiles_records, _source_consume,
                                _door_device("source_tile"), _table_fields("source_tile"),
                                None, _source_tiles_land)

# #29 P6b.3: THE SIGNAL TABLES as lists (the bus, the sensors and their disc
# tiles, the nodes and their inputs, the pumps and their inputs, the door
# drives: ``unit_kernels.SIGNAL_RECORDS``). Staged as inputs, compared as
# outputs and, in the chained form, LANDED back onto the sim's own bus and
# runtimes (``signals_to_host``, ``nodes_to_host``, ``pumps_to_host``); the
# load-constant tables (sensors, disc tiles, inputs, drives) land by being
# held equal to the sim's own.
def _signal_records(list_name):
    def rec(sim, obj):
        return unit_kernels.SIGNAL_RECORDS[list_name](sim)
    return rec


def _same_land(list_name):
    def land(sim, obj, recs):
        unit_kernels._same_rows(list_name, recs, unit_kernels.SIGNAL_RECORDS[list_name](sim))
    return land


def _signal_consume(obj):
    raise RuntimeError("lockstep: the signal tables are landed, never consumed")


_SIGNAL_LANDS = {
    "signals": lambda sim, obj, recs: unit_kernels.signals_to_host(sim, recs),
    "nodes": lambda sim, obj, recs: unit_kernels.nodes_to_host(sim, recs),
    "pumps": lambda sim, obj, recs: unit_kernels.pumps_to_host(sim, recs),
}
for _ln, _tn in zip(unit_kernels.SIGNAL_LISTS, unit_kernels._SIGNAL_TABLE_NAMES):
    LISTS[_ln] = ListDef(_sim_host, _signal_records(_ln), _signal_consume, _door_device(_tn),
                         _wide_fields(_tn), None, _SIGNAL_LANDS.get(_ln, _same_land(_ln)))

# #29 P7d: THE HORDE'S FIELD CACHE (``sim.horde_state``, cross-tick device
# state: horde_cache.CacheRecords -- a header and one record per field, each
# with its plane's digest, the planes riding along so the cache is staged and
# landed whole). ``horde`` is staged from the host every call (the shadow);
# ``horde_live`` is never restaged after its first begin (the CACHE gate: the
# device's own cache across ticks, compared with the host's every tick).
def _horde_records(sim, obj):
    from simulation import horde_cache
    return horde_cache.cache_records(sim)


def _horde_device(persistent):
    def device(n_env, n_units, cap, bp):
        from simulation import horde_cache
        return horde_cache.DeviceHordeCache(n_env, n_units, persistent=persistent)
    return device


def _horde_land(sim, obj, recs):
    from simulation import horde_cache
    horde_cache.to_host(sim, recs)


def _horde_fields(rec):
    if int(rec[0]) == 0:
        return "horde header", ("tag", "mask_built", "has_serial", "serial", "floods",
                                "n_fields", "masks_digest")
    return "horde field", ("tag", "kind", "fp", "sy", "sx", "slot", "dist_digest")


def _horde_consume(obj):
    raise RuntimeError("lockstep: the horde cache is landed, never consumed")


LISTS["horde"] = ListDef(_sim_host, _horde_records, _horde_consume, _horde_device(False),
                         _horde_fields, None, _horde_land)
LISTS["horde_live"] = ListDef(_sim_host, _horde_records, _horde_consume,
                              _horde_device(True), _horde_fields, None, _horde_land)


def _capture_horde_latch(sim):
    from simulation import horde_cache
    return horde_cache.latch_of(sim, len(sim.units))


def _pad_horde_latch(vals):
    w = max(len(v) for v in vals)
    out = np.zeros((len(vals), w), dtype=np.int64)
    for e, v in enumerate(vals):
        out[e, :len(v)] = v
    return out


INPUTS["horde_latch"] = ("table", _capture_horde_latch)
PADS["horde_latch"] = _pad_horde_latch

LISTS["ducts"] = ListDef(_sim_host, _ducts_records, _vent_consume, _door_device("duct"),
                         _wide_fields("duct"), None, _ducts_land)
LISTS["vents"] = ListDef(_sim_host, _vents_records, _vent_consume, _door_device("vent"),
                         _wide_fields("vent"), None, _vents_land)


# ---------------------------------------------------------------------------
# #29 P6.4b: field outputs, books, the write-set gate
# ---------------------------------------------------------------------------
BOOK_PREFIX = "book:"


def book_ref(name: str) -> str:
    """The host quantity a book output twins (``simulation.applier.BOOKS``)."""
    from simulation import applier
    for b in applier.BOOKS:
        if b.name == name:
            if b.host is None:
                raise ValueError(f"lockstep: book {name!r} has no host twin")
            return b.host
    raise ValueError(f"lockstep: no book {name!r} (simulation.applier.BOOKS)")


def host_books(sim, names) -> dict:
    from simulation import applier
    return {n: applier.host_book(sim.gmap, book_ref(n)) for n in names}


_SCALARS = (bool, int, float, str, type(None), np.integer, np.floating, np.bool_)
#: GameMap attributes that hold objects, probed by a function of the object
#: (its synced content), so the write-set gate sees them.
OBJECT_PROBES = {
    "damage_credit": lambda o: dict(getattr(o, "_owner", {}) or {}),
}


def gmap_state(gmap) -> dict:
    """What the WRITE-SET gate compares across a host slot: every ndarray
    attribute of the GameMap (a copy), every scalar, every dict (per key, its
    scalar values), set, list and tuple (a copy), and the object probes. Not
    traced: other objects (the material / gas tables, the engine binding, the
    level), which no structural seam writes."""
    out = {}
    ptrs: dict = {}
    for k, v in vars(gmap).items():
        if k == "_dev":            # residency's device mirror (not host state)
            continue
        if isinstance(v, np.ndarray):
            out[k] = v.copy()
            ptrs.setdefault((v.__array_interface__["data"][0], v.shape,
                             v.dtype.str), []).append(k)
        elif isinstance(v, _SCALARS):
            out[k] = v
        elif isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, _SCALARS + (tuple,)):
                    out[f"{k}:{kk}"] = vv
        elif isinstance(v, (set, frozenset)):
            out[k] = frozenset(v)
        elif isinstance(v, (list, tuple)):
            out[k] = list(v)
        elif k in OBJECT_PROBES:
            out[k] = OBJECT_PROBES[k](v)
    # ALIASES: attributes that are the SAME array (e.g. `obstacles` is
    # `solid` until the first stamp_units): a write to one is a write to all
    # of them, covered when any of them is declared.
    out[ALIASES] = {n: frozenset(g) - {n} for g in ptrs.values() if len(g) > 1
                    for n in g}
    return out


ALIASES = "__aliases__"


def changed_attrs(pre: dict, post: dict) -> list:
    """The names whose value differs between two :func:`gmap_state` s
    (an added or removed key included)."""
    out = []
    for k in sorted((set(pre) | set(post)) - {ALIASES}):
        a, b = pre.get(k, None), post.get(k, None)
        if (k in pre) != (k in post):
            out.append(k)
        elif isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)
                    and a.shape == b.shape and a.dtype == b.dtype
                    and np.array_equal(a, b)):
                out.append(k)
        elif a != b:
            out.append(k)
    return out


def write_set_violations(slot, changed, aliases=None) -> list:
    """The changed names that are neither an output of ``slot`` nor host-only
    (a dict key ``attr:key`` is covered by ``attr:key`` or by ``attr``; an
    array by any attribute that is the same array, ``aliases`` from
    :func:`gmap_state`)."""
    allowed = set(slot.field_outputs)
    allowed |= {book_ref(b) for b in slot.book_outputs}
    allowed |= {h for h, _why in slot.host_only}
    aliases = aliases or {}
    out = []
    for c in changed:
        if c in allowed or c.split(":", 1)[0] in allowed:
            continue
        # #29 P6b.1: a LIST attribute whose element books are declared
        # (``trace_clip_books:<g>``) is covered by them (each compared)
        if any(a.startswith(c + ":") for a in allowed):
            continue
        if aliases.get(c, frozenset()) & allowed:
            continue
        out.append(c)
    return out


def _num(v):
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def split_result(res):
    """A slot run's result: ``coverage`` or ``(coverage, books)``."""
    if isinstance(res, tuple) and len(res) == 2 and isinstance(res[1], dict):
        return np.asarray(res[0], dtype=np.int64), res[1]
    return np.asarray(res, dtype=np.int64), None


def first_field_diff(name, got, want):
    """The first differing element of two ``(N, ...)`` arrays as ``(env,
    index tuple, host, device)`` or None."""
    if got.shape != want.shape or got.dtype != want.dtype:
        return (0, ("shape/dtype",), f"{want.shape} {want.dtype}",
                f"{got.shape} {got.dtype}")
    if got.dtype.kind == "f":        # bit for bit (a -0.0 is not a 0.0)
        got = got.view(np.dtype(f"u{got.dtype.itemsize}"))
        want = want.view(np.dtype(f"u{want.dtype.itemsize}"))
    d = got != want
    if not d.any():
        return None
    idx = tuple(int(x) for x in np.argwhere(d)[0])
    return (idx[0], idx[1:], want[idx], got[idx])


@dataclass
class SlotInputs:
    """What a slot reads besides the rows, gathered over the batch."""
    tick: np.ndarray | None = None              # (N,) int64
    tables: dict = field(default_factory=dict)  # name -> (N, R, C) int64
    fields: dict = field(default_factory=dict)  # name -> device (N, ...)
    _host_fields: dict = field(default_factory=dict)   # name -> np (N, ...)
    scalars: dict = field(default_factory=dict)  # name -> (N,) int64 (P6.4a)
    lists: dict = field(default_factory=dict)    # name -> [N x (R_e, F)] (P7a.1)

    def env(self, e: int) -> "SlotInputs":
        """Env ``e``'s inputs alone (an N = 1 batch)."""
        import cupy as cp
        return SlotInputs(
            tick=None if self.tick is None else self.tick[e:e + 1].copy(),
            tables={k: np.ascontiguousarray(v[e:e + 1])
                    for k, v in self.tables.items()},
            fields={k: cp.ascontiguousarray(cp.asarray(v[e:e + 1]))
                    for k, v in self._host_fields.items()},
            _host_fields={k: v[e:e + 1] for k, v in self._host_fields.items()},
            scalars={k: v[e:e + 1].copy() for k, v in self.scalars.items()},
            lists={k: v[e:e + 1] for k, v in self.lists.items()})


def capture(names, sim, args=(), kw=None) -> dict:
    """One env's raw inputs at this moment: ``{name: value}`` (``args`` /
    ``kw``: the slot callable's arguments, for an :data:`ARG_INPUTS` row)."""
    out = {}
    for n in names:
        if n.startswith(FIELD_PREFIX):
            out[n] = np.array(getattr(sim.gmap, n[len(FIELD_PREFIX):]),
                              copy=True)
        elif n.startswith(LIST_PREFIX):
            ld = LISTS[n[len(LIST_PREFIX):]]
            out[n] = ld.records(sim, ld.host(sim, tuple(args), dict(kw or {})))
        elif n in ARG_INPUTS:
            out[n] = ARG_INPUTS[n][1](sim, tuple(args), dict(kw or {}))
        else:
            if n not in INPUTS:
                raise ValueError(f"lockstep: no slot input {n!r} (known: "
                                 f"{sorted(INPUTS) + sorted(ARG_INPUTS)} or "
                                 f"field:<name>)")
            out[n] = INPUTS[n][1](sim)
    return out


def gather(names, per_env) -> SlotInputs:
    """N envs' raw inputs (:func:`capture`) as one :class:`SlotInputs`."""
    inp = SlotInputs()
    for n in names:
        vals = [p[n] for p in per_env]
        if n.startswith(FIELD_PREFIX):
            import cupy as cp
            host = np.ascontiguousarray(np.stack(vals))
            key = n[len(FIELD_PREFIX):]
            inp._host_fields[key] = host
            inp.fields[key] = cp.asarray(host)
        elif n.startswith(LIST_PREFIX):
            inp.lists[n[len(LIST_PREFIX):]] = list(vals)
        elif n in INPUTS and INPUTS[n][0] == "tick":
            inp.tick = np.array(vals, dtype=np.int64)
        elif n in INPUTS and INPUTS[n][0] == "scalar":
            inp.scalars[n] = np.array(vals, dtype=np.int64)
        else:
            pad = PADS.get(n)
            if pad is not None:
                inp.tables[n] = pad(vals)
            else:
                inp.tables[n] = np.ascontiguousarray(np.stack(vals),
                                                     dtype=np.int64)
    return inp


@dataclass(frozen=True)
class DeviceSlot:
    """One ported slot: its profiler ``label`` (the callables hooked are
    that label's ``SLOT_WRAPS`` rows), ``run(store[, inputs]) -> coverage
    (N,)`` -- the device kernel over the uploaded store, returning per env
    the rows it changed --, the compared ``columns`` (default all) and the
    ``inputs`` it reads (names of :data:`INPUTS` or ``field:<name>``);
    ``events``: the kernel appends to an event ring, handed as the last
    argument and compared with the host's events (#29 P6.3); ``requests``:
    the kernel appends to a request list (``unit_requests.RequestList``),
    handed after the ring and compared with what the host did in the slot --
    the grenades released, the charges offered to ``charges.place`` (placed
    or refused), the doors flipped -- record for record, in order (#29
    P6.4a)."""
    label: str
    run: Callable
    columns: tuple = ()
    inputs: tuple = ()
    events: bool = False
    requests: bool = False
    # #29 P6.4b -- FIELD OUTPUTS: what the slot writes besides unit rows:
    # "field:<GameMap attribute>" (the kernel writes the input's DEVICE array
    # in place; captured at the entry like an input) and "book:<name>" (a
    # simulation.applier.BOOKS row: the kernel returns its per-env value,
    # compared with the host's delta over the slot). `host_only`: (GameMap
    # attribute or "attr:key", reason) pairs the slot writes on the host with
    # no device form. With outputs declared, the WRITE-SET gate runs: every
    # GameMap attribute the host slot changed must be an output or host-only.
    outputs: tuple = ()
    host_only: tuple = ()
    # #29 P7a.1: which calls of the label's callables are this slot's
    # (sim, args, kwargs, caller function name) -> bool; None = every call.
    select: Callable | None = None
    # #29 P7a.2: a NAME when two slots share a label (slot 4c's resolve and
    # slot 2's landing both hook SimultaneousEffects.resolve); reports and
    # divergences are keyed by it.
    name: str = ""

    @property
    def key(self) -> str:
        return self.name or self.label
    # #29 P7b: the conductor may gate the slot off (charges.tick runs only
    # while a charge is planted): an env where it did not run is compared as
    # the slot run on the state the tick ended with, which must change
    # nothing (its inputs, rows, lists, books captured then; its device run
    # must leave every output as it found it). False: not running raises.
    skip_ok: bool = False

    def call(self, store, inputs: SlotInputs | None, ring=None, req=None,
             lists=None):
        args = [store]
        if self.inputs or self.outputs:
            args.append(inputs)
        if self.events:
            args.append(ring)
        if self.requests:
            args.append(req)
        if self.list_names:
            args.append(lists)
        return self.run(*args)

    @property
    def list_inputs(self) -> tuple:
        return tuple(i[len(LIST_PREFIX):] for i in self.inputs
                     if i.startswith(LIST_PREFIX))

    @property
    def list_outputs(self) -> tuple:
        return tuple(o[len(LIST_PREFIX):] for o in self.outputs
                     if o.startswith(LIST_PREFIX))

    @property
    def list_names(self) -> tuple:
        return tuple(dict.fromkeys(self.list_inputs + self.list_outputs))

    def selects(self, sim, args, kw, caller) -> bool:
        return self.select is None or bool(self.select(sim, args, kw, caller))

    @property
    def field_outputs(self) -> tuple:
        return tuple(o[len(FIELD_PREFIX):] for o in self.outputs
                     if o.startswith(FIELD_PREFIX))

    @property
    def book_outputs(self) -> tuple:
        return tuple(o[len(BOOK_PREFIX):] for o in self.outputs
                     if o.startswith(BOOK_PREFIX))

    @property
    def capture_names(self) -> tuple:
        """The inputs, then the output fields not already inputs."""
        names = list(self.inputs)
        for f in self.field_outputs:
            if FIELD_PREFIX + f not in names:
                names.append(FIELD_PREFIX + f)
        return tuple(names)


@dataclass
class TickReport:
    tick: int                      # the sims' tick after the step
    coverage: np.ndarray           # (N,) the kernel's changed rows
    host_changed: np.ndarray       # (N,) rows the host slot changed
    equal: bool
    first_diff: tuple | None = None   # (tick, env, unit id, column)
    label: str = ""                   # the slot's label
    events: np.ndarray | None = None       # (N,) the ring's records (events slots)
    host_events: np.ndarray | None = None  # (N,) the host's events in the slot
    requests: list | None = None           # per env: the device's request records
    host_requests: list | None = None      # per env: the host's (P6.4a)
    books: dict | None = None              # #29 P6.4b: the device's {name: (N,)}
    host_books: dict | None = None         # the host's delta over the slot
    write_set: list | None = None          # per env: the attributes the host changed
    lists: dict | None = None              # #29 P7a.1: {name: per env device records}
    host_lists: dict | None = None         # {name: per env host records at the exit}


@dataclass
class Divergence:
    tick: int
    env: int
    unit_id: int
    column: str
    host: int
    device: int
    label: str = ""

    def __str__(self):
        return (f"[{self.label}] tick {self.tick} env {self.env} "
                f"unit {self.unit_id} "
                f"column {self.column}: host {self.host} device "
                f"{self.device}")


def slot_callables(label: str) -> list:
    """``(container, attr)`` of every callable ``tick_stage_profile`` wraps
    under ``label`` (a class only where it defines the method)."""
    import tick_stage_profile as tsp
    out = []
    for mod, qual, lab in tsp.SLOT_WRAPS:
        if lab != label:
            continue
        try:
            cont, attr = tsp._resolve(importlib.import_module(mod), qual)
            getattr(cont, attr)
        except AttributeError:
            continue
        if not isinstance(cont, type) or attr in cont.__dict__:
            out.append((cont, attr))
    if not out:
        raise ValueError(f"lockstep: no tick_stage_profile slot labelled "
                         f"{label!r}")
    return out


def build_store(sims, n_units, bp=None) -> unit_store.DeviceStore:
    """A ``DeviceStore`` for ``sims`` at ``n_units`` rows, their device
    tables uploaded."""
    return unit_store.DeviceStore(len(sims), n_units, bp=bp,
                                  tables=unit_tables.values_batch(sims))


class ShadowLockstep:
    """The harness over ``drivers`` (each with ``.sim`` and ``.tick()``: a
    ``MatchSession``, or any loop that steps one sim once) and ONE slot or a
    sequence of slots (each hooked on its own callables; one match run
    shadows them all). U is the largest env's unit count at construction (a
    match spawns once; a unit is never removed).

    :meth:`tick` returns the slot's :class:`TickReport` -- a dict ``label ->
    TickReport`` when several slots are shadowed. ``reports`` holds every
    report (each with its ``label``), ``divergences`` every difference."""

    def __init__(self, drivers, slot, bp=None,
                 check_alone: bool = True, n_units: int | None = None,
                 e_max: int = unit_events.E_MAX, r_max: int | None = None,
                 list_cap: int | None = None):
        if r_max is None:
            from simulation.unit_requests import R_MAX as r_max
        self.drivers = list(drivers)
        self.slots = list(slot) if isinstance(slot, (list, tuple)) else [slot]
        self.multi = isinstance(slot, (list, tuple))
        labels = [s.key for s in self.slots]
        if len(set(labels)) != len(labels):
            raise ValueError(f"lockstep: a slot is named twice ({labels})")
        self.slot = self.slots[0]
        self.N = len(self.drivers)
        self.U = int(n_units) if n_units is not None else max(
            len(d.sim.units) for d in self.drivers)
        sims = [d.sim for d in self.drivers]
        self.tables = unit_tables.values_batch(sims)
        self.store = unit_store.DeviceStore(self.N, self.U, bp=bp,
                                            tables=self.tables)
        self.alone = (unit_store.DeviceStore(1, self.U, bp=bp,
                                             tables=self.tables[:1])
                      if check_alone and self.N > 1 else None)
        emits = any(s.events for s in self.slots)
        self.ring = (unit_events.EventRing(self.N, self.U, e_max, bp=bp)
                     if emits else None)
        self.alone_ring = (unit_events.EventRing(1, self.U, e_max, bp=bp)
                           if emits and self.alone is not None else None)
        # #29 P6.4a: a slot that makes requests gets a request list (and the
        # host's own requests are captured while it runs: _install_requests)
        from simulation import unit_requests
        asks = any(s.requests for s in self.slots)
        self.req = (unit_requests.RequestList(self.N, self.U, r_max, bp=bp)
                    if asks else None)
        self.alone_req = (unit_requests.RequestList(1, self.U, r_max, bp=bp)
                          if asks and self.alone is not None else None)
        # #29 P7a.1: the device lists (one per declared name, the alone run's)
        from simulation import applier
        cap = int(list_cap) if list_cap is not None else applier.E_CAP
        names = dict.fromkeys(n for s in self.slots for n in s.list_names)
        self.dev_lists = {n: LISTS[n].device(self.N, self.U, cap, bp) for n in names}
        self.alone_lists = ({n: LISTS[n].device(1, self.U, cap, bp) for n in names}
                            if self.alone is not None else {})
        self.layout = unit_store.layout()
        self._masks = [self._column_mask(s.columns) for s in self.slots]
        self._orig = []
        self._env = None
        self._reset_tick()
        self.reports: list = []
        self.divergences: list = []
        self.last_inputs: dict = {}

    def _reset_tick(self):
        k = len(self.slots)
        self._pre = [[None] * self.N for _ in range(k)]
        self._post = [[None] * self.N for _ in range(k)]
        self._inp = [[None] * self.N for _ in range(k)]
        self._ev0 = [[None] * self.N for _ in range(k)]
        self._evs = [[None] * self.N for _ in range(k)]
        self._reqs = [[[] for _ in range(self.N)] for _ in range(k)]   # P6.4a
        self._req_k = None          # the requests slot running now (its index)
        # #29 P6.4b: the field outputs' host values at the exit, the books
        # at the entry / exit, the write-set snapshots
        self._fpost = [[None] * self.N for _ in range(k)]
        self._bk0 = [[None] * self.N for _ in range(k)]
        self._bk1 = [[None] * self.N for _ in range(k)]
        self._ws0 = [[None] * self.N for _ in range(k)]
        self._ws1 = [[None] * self.N for _ in range(k)]
        # #29 P7a.1: the output lists' host records at the exit
        self._lout = [[None] * self.N for _ in range(k)]

    # -- the hook ------------------------------------------------------------
    def _column_mask(self, names):
        if not names:
            return None
        m = np.zeros(self.layout.width, dtype=bool)
        for n in names:
            c = self.layout.column(n)
            m[c.offset:c.offset + c.width] = True
        return m

    def install(self):
        ls = self
        for k, slot in enumerate(self.slots):
            for cont, attr in slot_callables(slot.label):
                orig = getattr(cont, attr)

                def hooked(*a, __orig=orig, __k=k, **kw):
                    e = ls._env
                    if e is None:
                        return __orig(*a, **kw)
                    sim = ls.drivers[e].sim
                    sl = ls.slots[__k]
                    if not sl.selects(sim, a, kw, _caller()):
                        return __orig(*a, **kw)
                    if ls._pre[__k][e] is None:
                        ls._pre[__k][e] = unit_store.pack(sim.units, ls.U)
                        ls._inp[__k][e] = capture(sl.capture_names, sim, a, kw)
                        ls._ev0[__k][e] = (sim.tick_events,
                                           len(sim.tick_events))
                        if sl.outputs:
                            ls._bk0[__k][e] = host_books(sim, sl.book_outputs)
                            ls._ws0[__k][e] = gmap_state(sim.gmap)
                        for ln in sl.list_names:            # #29 P7b
                            if LISTS[ln].mark is not None:
                                LISTS[ln].mark(sim)
                    if ls.slots[__k].requests:
                        ls._req_k = __k
                    try:
                        return __orig(*a, **kw)
                    finally:
                        ls._req_k = None
                        ls._post[__k][e] = unit_store.pack(sim.units, ls.U)
                        evl, n0 = ls._ev0[__k][e]
                        ls._evs[__k][e] = list(evl[n0:])
                        lout = {}
                        for ln in sl.list_outputs:          # #29 P7b: per name
                            ld = LISTS[ln]
                            lout[ln] = ld.records(sim, ld.host(sim, a, kw))
                        if lout:
                            ls._lout[__k][e] = lout
                        if sl.outputs:
                            ls._fpost[__k][e] = {
                                f: np.array(getattr(sim.gmap, f), copy=True)
                                for f in sl.field_outputs}
                            ls._bk1[__k][e] = host_books(sim, sl.book_outputs)
                            ls._ws1[__k][e] = gmap_state(sim.gmap)
                hooked.__wrapped__ = orig
                setattr(cont, attr, hooked)
                self._orig.append((cont, attr, orig))
        if any(s.requests for s in self.slots):
            self._install_requests()
        if any(n in PAYLOAD_LOGS for s in self.slots for n in s.list_names):   # #29 P7b
            self._orig.extend(install_payload_tap())
        if any("claims" in s.list_names for s in self.slots):     # #29 P7a.3
            self._orig.extend(install_claims_tap())
        return self

    def _install_requests(self):
        """#29 P6.4a: while a requests slot runs, record what the HOST does
        that a request record stands for, in the order it does it: each
        grenade ``Simulation._release_grenade`` appends, each charge offered
        to ``charges.place`` (with its answer), each door
        ``Simulation._direct_use`` flips (the door read first through the
        verb's own pure scan, ``verb_gates.use_door``). Reads only."""
        from simulation import charges, unit_requests, verb_gates
        from simulation.simulation import Simulation
        ls = self

        def _sink():
            if ls._env is None or ls._req_k is None:
                return None
            return ls._reqs[ls._req_k][ls._env]

        orig_rel = Simulation._release_grenade

        def release(sim, u, *a, __o=orig_rel, **kw):
            n0 = len(sim.projectiles)
            out = __o(sim, u, *a, **kw)
            s = _sink()
            if s is not None and len(sim.projectiles) > n0:
                s.append(unit_requests.throw_record(sim, sim.projectiles[-1]))
            return out
        release.__wrapped__ = orig_rel
        Simulation._release_grenade = release
        self._orig.append((Simulation, "_release_grenade", orig_rel))

        orig_place = charges.place

        def place(sim, charge, __o=orig_place):
            ok = __o(sim, charge)
            s = _sink()
            if s is not None:
                s.append(unit_requests.plant_record(charge, sim.tick, ok))
            return ok
        place.__wrapped__ = orig_place
        charges.place = place
        self._orig.append((charges, "place", orig_place))

        orig_use = Simulation._direct_use

        def use(sim, u, __o=orig_use):
            s = _sink()
            door = verb_gates.use_door(sim, u) if s is not None else None
            out = __o(sim, u)
            if s is not None and door is not None:
                s.append(unit_requests.use_record(u.id, sim.tick,
                                                  sim._doors.index(door)))
            return out
        use.__wrapped__ = orig_use
        Simulation._direct_use = use
        self._orig.append((Simulation, "_direct_use", orig_use))

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    # -- one tick --------------------------------------------------------------
    def tick(self):
        """Tick every driver once (the slots hooked), then each device slot
        over the batch, then the compares. Returns the tick's report (a dict
        by label when several slots are shadowed; also appended to
        ``reports``; a difference to ``divergences``)."""
        self._reset_tick()
        for e, d in enumerate(self.drivers):
            self._env = e
            try:
                d.tick()
            finally:
                self._env = None
            for k, slot in enumerate(self.slots):
                if self._pre[k][e] is None and slot.skip_ok:     # #29 P7b
                    self._fill_skipped(k, slot, e, d.sim)
                if self._pre[k][e] is None:
                    raise RuntimeError(
                        f"lockstep: slot {slot.label!r} did not run in env "
                        f"{e} at tick {d.sim.tick}")
        tick = int(self.drivers[0].sim.tick)
        out = {}
        for k, slot in enumerate(self.slots):
            out[slot.key] = self._compare(k, slot, tick)
        return out if self.multi else out[self.slot.key]

    def _fill_skipped(self, k, slot, e, sim):
        """#29 P7b: env ``e`` did not run the gated slot ``k`` this tick: its
        entry and exit are the state the tick ended with (a run that must
        change nothing)."""
        rows = unit_store.pack(sim.units, self.U)
        self._pre[k][e] = rows
        self._post[k][e] = rows.copy()
        self._inp[k][e] = capture(slot.capture_names, sim)
        self._evs[k][e] = []
        lout = {ln: LISTS[ln].records(sim, LISTS[ln].host(sim, (), {}))
                for ln in slot.list_outputs if LISTS[ln].mark is None}
        lout.update({ln: np.zeros((0, applier_width()), dtype=np.int64)
                     for ln in slot.list_outputs if LISTS[ln].mark is not None})
        if lout:
            self._lout[k][e] = lout
        if slot.outputs:
            self._fpost[k][e] = {f: np.array(getattr(sim.gmap, f), copy=True)
                                 for f in slot.field_outputs}
            self._bk0[k][e] = self._bk1[k][e] = host_books(sim, slot.book_outputs)
            self._ws0[k][e] = self._ws1[k][e] = gmap_state(sim.gmap)

    def _compare(self, k, slot, tick) -> TickReport:
        pre = np.stack(self._pre[k])
        post = np.stack(self._post[k])
        names = slot.capture_names
        inputs = gather(names, self._inp[k]) if names else None
        self.last_inputs[slot.key] = inputs
        self.store.upload(pre)
        if slot.events:
            self.ring.begin()
        if slot.requests:
            self.req.begin()
        self._stage_lists(slot, self.dev_lists, inputs)
        coverage, books = split_result(slot.call(self.store, inputs, self.ring,
                                                 self.req, self.dev_lists))
        got = self.store.download()
        host_changed = np.array([int((pre[e] != post[e]).any(axis=1).sum())
                                 for e in range(self.N)], dtype=np.int64)
        first = self._first_diff(tick, got, post, self._masks[k])
        ring_recs = host_n = None
        if slot.events:
            ring_recs = self.ring.records()
            host_n = np.array([len([v for v in self._evs[k][e]
                                    if unit_events.is_synced(v)])
                               for e in range(self.N)], dtype=np.int64)
            if first is None:
                first = self._first_event_diff(tick, k, ring_recs)
        req_recs = None
        if slot.requests:
            req_recs = self.req.records()
            if first is None:
                first = self._first_request_diff(tick, k, req_recs)
        # #29 P7a.1: the output lists, record for record
        dev_lists = host_lists = None
        if slot.list_outputs:
            dev_lists = {n: self.dev_lists[n].records() for n in slot.list_outputs}
            host_lists = {n: [[tuple(int(x) for x in r) for r in self._lout[k][e][n]]
                              for e in range(self.N)] for n in slot.list_outputs}
            if first is None:
                first = self._first_list_diff(tick, slot, dev_lists, host_lists)
        # #29 P6.4b: the field outputs, the books, the write set
        dev_fields = hb = ws = None
        if slot.outputs:
            dev_fields = {f: inputs.fields[f].get() for f in slot.field_outputs}
            hb = {b: np.array([self._bk1[k][e][b] - self._bk0[k][e][b]
                               for e in range(self.N)], dtype=np.int64)
                  for b in slot.book_outputs}
            ws = [changed_attrs(self._ws0[k][e], self._ws1[k][e])
                  for e in range(self.N)]
            if first is None:
                first = self._first_output_diff(tick, k, slot, dev_fields,
                                                books, hb, ws)
        if first is None and self.alone is not None:
            for e in range(self.N):
                self.alone.upload_tables(self.tables[e:e + 1])
                self.alone.upload(pre[e:e + 1])
                if slot.events:
                    self.alone_ring.begin()
                if slot.requests:
                    self.alone_req.begin()
                ie = None if inputs is None else inputs.env(e)
                self._stage_lists(slot, self.alone_lists, ie)
                _cov1, books1 = split_result(slot.call(self.alone, ie,
                                                       self.alone_ring,
                                                       self.alone_req,
                                                       self.alone_lists))
                for n in slot.list_outputs:
                    if self.alone_lists[n].records()[0] != dev_lists[n][e]:
                        first = Divergence(tick, e, -1, f"ALONE list {n}",
                                           len(self.alone_lists[n].records()[0]),
                                           len(dev_lists[n][e]))
                        break
                if first is not None:
                    break
                if slot.requests and self.alone_req.records()[0] != req_recs[e]:
                    first = Divergence(tick, e, -1, "ALONE request list",
                                       len(self.alone_req.records()[0]),
                                       len(req_recs[e]))
                    break
                if slot.outputs:
                    for f in slot.field_outputs:
                        fd = first_field_diff(f, ie.fields[f].get(),
                                              dev_fields[f][e:e + 1])
                        if fd is not None:
                            first = Divergence(tick, e, -1, f"ALONE field {f}"
                                               f"{tuple(fd[1])}", _num(fd[2]),
                                               _num(fd[3]))
                            break
                    if first is not None:
                        break
                    for b in slot.book_outputs:
                        if int(books1[b][0]) != int(books[b][e]):
                            first = Divergence(tick, e, -1, f"ALONE book {b}",
                                               int(books[b][e]),
                                               int(books1[b][0]))
                            break
                    if first is not None:
                        break
                one = self.alone.download()[0]
                if not np.array_equal(one, got[e]):
                    r, col = np.argwhere(one != got[e])[0]
                    first = Divergence(tick, e,
                                       int(pre[e, r, self.layout.column("id").offset]),
                                       "ALONE " + self.layout.column_at(int(col)),
                                       int(one[r, col]), int(got[e, r, col]))
                    break
                if slot.events:
                    alone_recs = self.alone_ring.records()[0]
                    if alone_recs != ring_recs[e]:
                        first = Divergence(tick, e, -1, "ALONE event ring",
                                           len(alone_recs), len(ring_recs[e]))
                        break
        if first is not None:
            first.label = slot.key
            self.divergences.append(first)
        rep = TickReport(tick=tick, coverage=coverage,
                         host_changed=host_changed, equal=first is None,
                         first_diff=None if first is None else (
                             first.tick, first.env, first.unit_id,
                             first.column),
                         label=slot.key,
                         events=None if ring_recs is None else np.array(
                             [len(r) for r in ring_recs], dtype=np.int64),
                         host_events=host_n,
                         requests=None if req_recs is None else req_recs,
                         host_requests=None if req_recs is None else [
                             list(self._reqs[k][e]) for e in range(self.N)],
                         books=books, host_books=hb, write_set=ws,
                         lists=dev_lists, host_lists=host_lists)
        self.reports.append(rep)
        return rep

    @staticmethod
    def _stage_lists(slot, devs, inputs):
        """#29 P7a.1: every list the slot names, on the device before its
        kernel: an input list staged from the host's records at the entry,
        an output-only list begun empty."""
        for n in slot.list_names:
            if n in slot.list_inputs:
                devs[n].stage_lists(inputs.lists[n])
            else:
                devs[n].begin()

    def _first_list_diff(self, tick, slot, dev_lists, host_lists):
        """#29 P7a.1: the first difference between the host's output list at
        the slot's exit and the device's, per env, record for record: (tick,
        env, index, kind.field) as a :class:`Divergence`."""
        for n in slot.list_outputs:
            ld = LISTS[n]
            for e in range(self.N):
                host, dev = host_lists[n][e], dev_lists[n][e]
                for i in range(max(len(host), len(dev))):
                    if i >= len(host):
                        return Divergence(tick, e, -1, f"{n}[{i}] (extra on the "
                                          f"device: {ld.fields(dev[i])[0]})", 0, dev[i][0])
                    if i >= len(dev):
                        return Divergence(tick, e, -1, f"{n}[{i}] (missing on the "
                                          f"device: {ld.fields(host[i])[0]})", host[i][0], 0)
                    if host[i] != dev[i]:
                        kname, names = ld.fields(host[i])
                        for f, (a, b) in enumerate(zip(host[i], dev[i])):
                            if a != b:
                                fname = names[f] if f < len(names) else f"a{f - 1}"
                                return Divergence(tick, e, -1, f"{n}[{i}].{kname}."
                                                  f"{fname}", a, b)
        return None

    def _first_request_diff(self, tick, k, req_recs):
        """#29 P6.4a: the first difference between the host's requests of
        slot ``k`` (captured in the order the host made them) and the
        device's list, per env, record for record: (tick, env, request
        index, field) as a :class:`Divergence`."""
        from simulation import unit_requests
        for e in range(self.N):
            host = self._reqs[k][e]
            dev = req_recs[e]
            for i in range(max(len(host), len(dev))):
                if i >= len(host):
                    return Divergence(tick, e, dev[i][1], f"request[{i}] (extra on "
                                      f"the device)", 0, dev[i][0])
                if i >= len(dev):
                    return Divergence(tick, e, host[i][1], f"request[{i}] (missing "
                                      f"on the device)", host[i][0], 0)
                for f, name in enumerate(unit_requests.FIELDS):
                    if host[i][f] != dev[i][f]:
                        return Divergence(tick, e, host[i][1],
                                          f"request[{i}].{name}", host[i][f],
                                          dev[i][f])
        return None

    def _first_output_diff(self, tick, k, slot, dev_fields, books, hb, ws):
        """#29 P6.4b: the first difference in a slot's field outputs (tick,
        env, field, tile), its books (the device's value against the host's
        delta over the slot) or its write set (a GameMap attribute the host
        changed that is neither an output nor host-only)."""
        for f in slot.field_outputs:
            want = np.stack([self._fpost[k][e][f] for e in range(self.N)])
            fd = first_field_diff(f, dev_fields[f], want)
            if fd is not None:
                return Divergence(tick, fd[0], -1, f"field {f} {tuple(fd[1])}",
                                  _num(fd[2]), _num(fd[3]))
        if slot.book_outputs and books is None:
            return Divergence(tick, 0, -1, "books (the kernel returned none)",
                              0, 0)
        for b in slot.book_outputs:
            for e in range(self.N):
                if int(books[b][e]) != int(hb[b][e]):
                    return Divergence(tick, e, -1, f"book {b}", int(hb[b][e]),
                                      int(books[b][e]))
        for e in range(self.N):
            bad = write_set_violations(slot, ws[e], self._ws0[k][e].get(ALIASES))
            if bad:
                return Divergence(tick, e, -1, f"write-set: {bad} changed on "
                                  f"the host, neither an output nor host-only",
                                  0, 0)
        return None

    def _first_event_diff(self, tick, k, ring_recs):
        """The first difference between the host's events of slot ``k`` and
        the ring, per env, record for record in order: (tick, env, event
        index, field) as a :class:`Divergence` (unit_id the host's event's)."""
        for e in range(self.N):
            # #29 P7a.1: the ring keeps only synced events; the host's
            # render-only ones (MeleeEvent, ShotFired, ...) are dropped here
            host = [v for v in self._evs[k][e] if unit_events.is_synced(v)]
            dev = ring_recs[e]
            for i in range(max(len(host), len(dev))):
                if i >= len(host):
                    return Divergence(tick, e, dev[i][1], f"event[{i}] (extra "
                                      f"on the device)", 0, dev[i][0])
                try:
                    h = unit_events.record_of(host[i])
                except ValueError as ex:
                    return Divergence(tick, e, getattr(host[i], "unit_id", -1),
                                      f"event[{i}] {ex}", 0, 0)
                if i >= len(dev):
                    return Divergence(tick, e, h[1], f"event[{i}] (missing on "
                                      f"the device)", h[0], 0)
                for f, name in enumerate(unit_events.FIELDS):
                    if h[f] != dev[i][f]:
                        return Divergence(tick, e, h[1], f"event[{i}].{name}",
                                          h[f], dev[i][f])
        return None

    def _first_diff(self, tick, got, want, mask):
        diff = got != want
        if mask is not None:
            diff &= mask
        if not diff.any():
            return None
        e, r, col = (int(x) for x in np.argwhere(diff)[0])
        return Divergence(tick, e, int(want[e, r, self.layout.column("id").offset]),
                          self.layout.column_at(col), int(want[e, r, col]),
                          int(got[e, r, col]))


# ---------------------------------------------------------------------------
# CHAINED
# ---------------------------------------------------------------------------
def _caller() -> str:
    """#29 P7a.2: the name of the function that called a hooked slot
    callable, skipping the harness's own wrappers (two slots hooking one
    callable nest their wrappers: slot 4c's resolve and slot 2's landing)."""
    f = sys._getframe(2)
    while f is not None and f.f_code.co_filename == __file__ \
            and f.f_code.co_name in ("hooked", "replaced"):
        f = f.f_back
    return f.f_code.co_name if f is not None else ""


def _is_for(sim, args) -> bool:
    """Whether a slot callable's arguments name ``sim`` (the sim itself, its
    unit list, or -- #29 P7.0a, slot 9d's ``apply_temperature_ignition(gmap,
    ...)`` -- its GameMap)."""
    bus = getattr(sim, "_signal_bus", None)
    for a in args:
        if a is sim or a is getattr(sim, "units", None) \
                or a is getattr(sim, "gmap", None):
            return True
        # #29 P6b.3: slot 9e(e)'s SignalBus.swap_node_signals(bus)
        if bus is not None and a is bus:
            return True
    return False


class ChainedSlots:
    """Adjacent device slots replacing the host's in ONE sim (``driver.sim``)
    -- see the module docstring. ``slots`` are in slot order; the first
    slot's entry runs them all. ``host_side(sim, pre, post)`` does the
    skipped slots' host-only side after the unpack (``pre`` / ``post`` the
    ``(U, W)`` rows before and after the kernels). ``coverage`` collects
    each tick's per-slot coverage."""

    def __init__(self, driver, slots, host_side=None, bp=None,
                 e_max: int = unit_events.E_MAX, list_cap: int | None = None):
        self.driver = driver
        self.slots = list(slots)
        self.host_side = host_side
        self.U = max(1, len(driver.sim.units))   # #29 P6b.4: a unit-free level pads one row
        self.store = build_store([driver.sim], self.U, bp=bp)
        self.ring = (unit_events.EventRing(1, self.U, e_max, bp=bp)
                     if any(s.events for s in self.slots) else None)
        from simulation import unit_requests
        self.req = (unit_requests.RequestList(1, self.U, bp=bp)
                    if any(s.requests for s in self.slots) else None)
        # #29 P7a.1: the chain's device lists and the host objects it consumed
        from simulation import applier
        cap = int(list_cap) if list_cap is not None else applier.E_CAP
        names = dict.fromkeys(n for s in self.slots for n in s.list_names)
        self.dev_lists = {n: LISTS[n].device(1, self.U, cap, bp) for n in names}
        self.lists: list = []             # per run: {name: the device's records}
        self._consumed: list = []         # host list objects this tick's chain took
        self._orig = []
        self.coverage: list = []          # per run: [per-slot (1,) arrays]
        self.events: list = []            # per run: the ring's records
        self.requests: list = []          # per run: the request list (P6.4a)
        self.runs = 0
        self.books: dict = {}             # #29 P6.4b: the last run's books
        self.inputs = None                # the last run's captured inputs

    def _run_chain(self, args=(), kw=None):
        sim = self.driver.sim
        names = []
        for s in self.slots:
            names.extend(n for n in s.capture_names if n not in names)
        inputs = (gather(names, [capture(names, sim, args, kw)]) if names
                  else None)
        pre = unit_store.pack(sim.units, self.U)
        self.store.upload(pre[None])
        if self.ring is not None:
            self.ring.begin()
        if self.req is not None:
            self.req.begin()
        # #29 P7a.1: the chain's lists -- an input list staged from the host's
        # records at the chain's entry (once: a later slot continues the
        # device's list), an output-only one begun empty
        hosts = {}
        for n in self.dev_lists:
            ins = any(n in s.list_inputs for s in self.slots)
            if ins:
                self.dev_lists[n].stage_lists(inputs.lists[n])
            else:
                self.dev_lists[n].begin()
            hosts[n] = LISTS[n].host(sim, tuple(args), dict(kw or {}))
        covs, books = [], {}
        for s in self.slots:
            c, b = split_result(s.call(self.store, inputs, self.ring, self.req,
                                       self.dev_lists))
            covs.append(c)
            if b is not None:
                # #29 P6b.4: the books of several slots of one chain SUM (each
                # launch zeroes its books: a slot's are its own deltas; a whole
                # 9e chain has several stages booking the same channel)
                for n_, v in b.items():
                    books[n_] = books.get(n_, 0) + int(v[0])
        post = self.store.download()[0]
        unit_store.unpack(post, sim.units)
        # #29 P6.4b: the device's field outputs written back into the
        # GameMap IN PLACE (the books are the host side's to land:
        # ``self.books``; the entry's inputs are ``self.inputs``)
        for s in self.slots:
            for f in s.field_outputs:
                getattr(sim.gmap, f)[...] = inputs.fields[f].get()[0]
        self.books, self.inputs = books, inputs
        if self.ring is not None:
            recs = self.ring.records()[0]
            sim.tick_events.extend(unit_events.event_of(r) for r in recs)
            self.events.append(recs)
        if self.req is not None:          # #29 P6.4a: the host's objects
            from simulation import unit_requests
            rq = self.req.records()[0]
            unit_requests.apply_to_host(sim, rq)
            self.requests.append(rq)
        # #29 P7a.1: the device applied the host's list: empty it, and refuse
        # a later chained callable that finds the host appended in between
        self._consumed = []
        landed = {n: d.records()[0] for n, d in self.dev_lists.items()}
        for n, obj in hosts.items():
            if LISTS[n].land is not None:     # #29 P7b: the device's rows written back
                LISTS[n].land(sim, obj, landed[n])
                continue
            LISTS[n].consume(obj)
            self._consumed.append((n, obj))
        if self.dev_lists:
            self.lists.append(landed)
        if self.host_side is not None:
            self.host_side(sim, pre, post)
        self.coverage.append(covs)
        self.runs += 1

    def install(self):
        ch = self
        sim = self.driver.sim
        for i, s in enumerate(self.slots):
            for j, (cont, attr) in enumerate(slot_callables(s.label)):
                orig = getattr(cont, attr)

                # the chain runs at the FIRST callable of the first slot (a
                # label may name several, #29 P6.3's "9c3 gas rows")
                def replaced(*a, __orig=orig, __first=(i == 0 and j == 0),
                             __s=s, **k):
                    held = [(n, o) for n, o in ch._consumed
                            if any(x is o for x in tuple(a) + tuple(k.values()))]
                    if not (_is_for(sim, a) or held):
                        return __orig(*a, **k)
                    if not __s.selects(sim, a, k, _caller()):
                        return __orig(*a, **k)
                    for n, o in held:
                        recs = LISTS[n].records(sim, o)
                        if len(recs):
                            raise RuntimeError(
                                f"chained: the host appended {len(recs)} record(s) "
                                f"to list {n!r} between the chain and "
                                f"{__s.label!r}: they would never be applied")
                    if __first:
                        ch._run_chain(a, k)
                    return None
                replaced.__wrapped__ = orig
                setattr(cont, attr, replaced)
                self._orig.append((cont, attr, orig))
        return self

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False


__all__ = ["FIELD_PREFIX", "INPUTS", "PADS", "SlotInputs", "capture",
           "gather", "DeviceSlot", "TickReport", "Divergence",
           "slot_callables", "build_store", "ShadowLockstep", "ChainedSlots"]
__all__ += ["LIST_PREFIX", "ListDef", "LISTS"]
__all__ += ["PAYLOAD_SITES", "install_payload_tap", "payload_call"]
__all__ += ["install_claims_tap", "door_combined", "door_split", "DoorList",
            "DOOR_TAG_EDIT", "DOOR_TAG_TILE", "REPLAY_REFUSED"]
__all__ += ["ARG_INPUTS", "BOOK_PREFIX", "ALIASES", "book_ref", "host_books",
            "OBJECT_PROBES", "gmap_state", "changed_attrs",
            "write_set_violations", "split_result", "first_field_diff"]


# ---------------------------------------------------------------------------
# #29 P6b.4: THE TIMED-CHARGE and THE HEAT-SOURCE TABLES as lists
# (``Simulation._timed_charges`` / ``_heat_sources`` through
# ``unit_kernels.timed_charges_of`` / ``heat_sources_of``): staged as inputs,
# compared as outputs and, in the chained form, LANDED back onto the sim's own
# runtimes (``timed_charges_to_host`` / ``heat_sources_to_host``). The
# TIMED_CHARGES / HEAT_SOURCES stages' own PAYLOAD lists (what each applied):
# the host side is the PAYLOAD tap's log of the slot's calls, the chained form
# consumes them (the device applied them all; never replayed).
# ---------------------------------------------------------------------------
def _hazard_consume(obj):
    raise RuntimeError("lockstep: the timed-charge / heat-source tables are landed, "
                       "never consumed")


LISTS["timed_charges"] = ListDef(
    _sim_host, lambda sim, obj: unit_kernels.timed_charges_of(sim), _hazard_consume,
    _door_device("timed_charge"), _wide_fields("timed_charge"), None,
    lambda sim, obj, recs: unit_kernels.timed_charges_to_host(sim, recs))
LISTS["heat_sources"] = ListDef(
    _sim_host, lambda sim, obj: unit_kernels.heat_sources_of(sim), _hazard_consume,
    _door_device("heat_source"), _wide_fields("heat_source"), None,
    lambda sim, obj, recs: unit_kernels.heat_sources_to_host(sim, recs))
for _ln in ("timed_payloads", "heat_payloads"):
    LISTS[_ln] = ListDef(_payloads_host, _payloads_records, lambda obj: obj.clear(),
                         _effects_device, _effects_fields, _payloads_mark,
                         _detonations_land)
__all__ += ["PAYLOAD_LOGS"]
