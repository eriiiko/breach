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
}
#: How a table input pads across envs (default: zeros).
PADS: dict = {"order_requests": unit_kernels.pad_requests,
              "credit_claims": unit_kernels.pad_claims,
              "door_spans": _pad_door_spans}


@dataclass
class SlotInputs:
    """What a slot reads besides the rows, gathered over the batch."""
    tick: np.ndarray | None = None              # (N,) int64
    tables: dict = field(default_factory=dict)  # name -> (N, R, C) int64
    fields: dict = field(default_factory=dict)  # name -> device (N, ...)
    _host_fields: dict = field(default_factory=dict)   # name -> np (N, ...)
    scalars: dict = field(default_factory=dict)  # name -> (N,) int64 (P6.4a)

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
            scalars={k: v[e:e + 1].copy() for k, v in self.scalars.items()})


def capture(names, sim) -> dict:
    """One env's raw inputs at this moment: ``{name: value}``."""
    out = {}
    for n in names:
        if n.startswith(FIELD_PREFIX):
            out[n] = np.array(getattr(sim.gmap, n[len(FIELD_PREFIX):]),
                              copy=True)
        else:
            if n not in INPUTS:
                raise ValueError(f"lockstep: no slot input {n!r} (known: "
                                 f"{sorted(INPUTS)} or field:<name>)")
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
        elif INPUTS[n][0] == "tick":
            inp.tick = np.array(vals, dtype=np.int64)
        elif INPUTS[n][0] == "scalar":
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

    def call(self, store, inputs: SlotInputs | None, ring=None, req=None):
        args = [store]
        if self.inputs:
            args.append(inputs)
        if self.events:
            args.append(ring)
        if self.requests:
            args.append(req)
        return self.run(*args)


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
                 e_max: int = unit_events.E_MAX, r_max: int | None = None):
        if r_max is None:
            from simulation.unit_requests import R_MAX as r_max
        self.drivers = list(drivers)
        self.slots = list(slot) if isinstance(slot, (list, tuple)) else [slot]
        self.multi = isinstance(slot, (list, tuple))
        labels = [s.label for s in self.slots]
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
                    if ls._pre[__k][e] is None:
                        ls._pre[__k][e] = unit_store.pack(sim.units, ls.U)
                        ls._inp[__k][e] = capture(ls.slots[__k].inputs, sim)
                        ls._ev0[__k][e] = (sim.tick_events,
                                           len(sim.tick_events))
                    if ls.slots[__k].requests:
                        ls._req_k = __k
                    try:
                        return __orig(*a, **kw)
                    finally:
                        ls._req_k = None
                        ls._post[__k][e] = unit_store.pack(sim.units, ls.U)
                        evl, n0 = ls._ev0[__k][e]
                        ls._evs[__k][e] = list(evl[n0:])
                hooked.__wrapped__ = orig
                setattr(cont, attr, hooked)
                self._orig.append((cont, attr, orig))
        if any(s.requests for s in self.slots):
            self._install_requests()
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
                if self._pre[k][e] is None:
                    raise RuntimeError(
                        f"lockstep: slot {slot.label!r} did not run in env "
                        f"{e} at tick {d.sim.tick}")
        tick = int(self.drivers[0].sim.tick)
        out = {}
        for k, slot in enumerate(self.slots):
            out[slot.label] = self._compare(k, slot, tick)
        return out if self.multi else out[self.slot.label]

    def _compare(self, k, slot, tick) -> TickReport:
        pre = np.stack(self._pre[k])
        post = np.stack(self._post[k])
        inputs = gather(slot.inputs, self._inp[k]) if slot.inputs else None
        self.last_inputs[slot.label] = inputs
        self.store.upload(pre)
        if slot.events:
            self.ring.begin()
        if slot.requests:
            self.req.begin()
        coverage = np.asarray(slot.call(self.store, inputs, self.ring, self.req),
                              dtype=np.int64)
        got = self.store.download()
        host_changed = np.array([int((pre[e] != post[e]).any(axis=1).sum())
                                 for e in range(self.N)], dtype=np.int64)
        first = self._first_diff(tick, got, post, self._masks[k])
        ring_recs = host_n = None
        if slot.events:
            ring_recs = self.ring.records()
            host_n = np.array([len(self._evs[k][e]) for e in range(self.N)],
                              dtype=np.int64)
            if first is None:
                first = self._first_event_diff(tick, k, ring_recs)
        req_recs = None
        if slot.requests:
            req_recs = self.req.records()
            if first is None:
                first = self._first_request_diff(tick, k, req_recs)
        if first is None and self.alone is not None:
            for e in range(self.N):
                self.alone.upload_tables(self.tables[e:e + 1])
                self.alone.upload(pre[e:e + 1])
                if slot.events:
                    self.alone_ring.begin()
                if slot.requests:
                    self.alone_req.begin()
                slot.call(self.alone, None if inputs is None else inputs.env(e),
                          self.alone_ring, self.alone_req)
                if slot.requests and self.alone_req.records()[0] != req_recs[e]:
                    first = Divergence(tick, e, -1, "ALONE request list",
                                       len(self.alone_req.records()[0]),
                                       len(req_recs[e]))
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
            first.label = slot.label
            self.divergences.append(first)
        rep = TickReport(tick=tick, coverage=coverage,
                         host_changed=host_changed, equal=first is None,
                         first_diff=None if first is None else (
                             first.tick, first.env, first.unit_id,
                             first.column),
                         label=slot.label,
                         events=None if ring_recs is None else np.array(
                             [len(r) for r in ring_recs], dtype=np.int64),
                         host_events=host_n,
                         requests=None if req_recs is None else req_recs,
                         host_requests=None if req_recs is None else [
                             list(self._reqs[k][e]) for e in range(self.N)])
        self.reports.append(rep)
        return rep

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

    def _first_event_diff(self, tick, k, ring_recs):
        """The first difference between the host's events of slot ``k`` and
        the ring, per env, record for record in order: (tick, env, event
        index, field) as a :class:`Divergence` (unit_id the host's event's)."""
        for e in range(self.N):
            host = self._evs[k][e]
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
def _is_for(sim, args) -> bool:
    """Whether a slot callable's arguments name ``sim`` (the sim itself, or
    its unit list)."""
    for a in args:
        if a is sim or a is getattr(sim, "units", None):
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
                 e_max: int = unit_events.E_MAX):
        self.driver = driver
        self.slots = list(slots)
        self.host_side = host_side
        self.U = len(driver.sim.units)
        self.store = build_store([driver.sim], self.U, bp=bp)
        self.ring = (unit_events.EventRing(1, self.U, e_max, bp=bp)
                     if any(s.events for s in self.slots) else None)
        from simulation import unit_requests
        self.req = (unit_requests.RequestList(1, self.U, bp=bp)
                    if any(s.requests for s in self.slots) else None)
        self._orig = []
        self.coverage: list = []          # per run: [per-slot (1,) arrays]
        self.events: list = []            # per run: the ring's records
        self.requests: list = []          # per run: the request list (P6.4a)
        self.runs = 0

    def _run_chain(self):
        sim = self.driver.sim
        names = []
        for s in self.slots:
            names.extend(n for n in s.inputs if n not in names)
        inputs = gather(names, [capture(names, sim)]) if names else None
        pre = unit_store.pack(sim.units, self.U)
        self.store.upload(pre[None])
        if self.ring is not None:
            self.ring.begin()
        if self.req is not None:
            self.req.begin()
        covs = [np.asarray(s.call(self.store, inputs, self.ring, self.req),
                           dtype=np.int64) for s in self.slots]
        post = self.store.download()[0]
        unit_store.unpack(post, sim.units)
        if self.ring is not None:
            recs = self.ring.records()[0]
            sim.tick_events.extend(unit_events.event_of(r) for r in recs)
            self.events.append(recs)
        if self.req is not None:          # #29 P6.4a: the host's objects
            from simulation import unit_requests
            rq = self.req.records()[0]
            unit_requests.apply_to_host(sim, rq)
            self.requests.append(rq)
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
                def replaced(*a, __orig=orig, __first=(i == 0 and j == 0), **k):
                    if not _is_for(sim, a):
                        return __orig(*a, **k)
                    if __first:
                        ch._run_chain()
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
