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


#: name -> (kind, capture function of the sim at the slot's entry). kind
#: "tick" gathers to ``SlotInputs.tick``; "table" to ``SlotInputs.tables``
#: (an ``(R, C)`` int64 per env, padded across envs by ``pad``).
INPUTS: dict = {
    "tick": ("tick", _capture_tick),
    "order_requests": ("table", _capture_requests),
    "credit_claims": ("table", _capture_claims),     # #29 P6.3
}
#: How a table input pads across envs (default: zeros).
PADS: dict = {"order_requests": unit_kernels.pad_requests,
              "credit_claims": unit_kernels.pad_claims}


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

    def env(self, e: int) -> "SlotInputs":
        """Env ``e``'s inputs alone (an N = 1 batch)."""
        import cupy as cp
        return SlotInputs(
            tick=None if self.tick is None else self.tick[e:e + 1].copy(),
            tables={k: np.ascontiguousarray(v[e:e + 1])
                    for k, v in self.tables.items()},
            fields={k: cp.ascontiguousarray(cp.asarray(v[e:e + 1]))
                    for k, v in self._host_fields.items()},
            _host_fields={k: v[e:e + 1] for k, v in self._host_fields.items()})


def capture(names, sim, args=(), kw=None) -> dict:
    """One env's raw inputs at this moment: ``{name: value}`` (``args`` /
    ``kw``: the slot callable's arguments, for an :data:`ARG_INPUTS` row)."""
    out = {}
    for n in names:
        if n.startswith(FIELD_PREFIX):
            out[n] = np.array(getattr(sim.gmap, n[len(FIELD_PREFIX):]),
                              copy=True)
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
        elif n in INPUTS and INPUTS[n][0] == "tick":
            inp.tick = np.array(vals, dtype=np.int64)
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
    argument and compared with the host's events (#29 P6.3)."""
    label: str
    run: Callable
    columns: tuple = ()
    inputs: tuple = ()
    events: bool = False
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

    def call(self, store, inputs: SlotInputs | None, ring=None):
        args = [store]
        if self.inputs or self.outputs:
            args.append(inputs)
        if self.events:
            args.append(ring)
        return self.run(*args)

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
    books: dict | None = None              # #29 P6.4b: the device's {name: (N,)}
    host_books: dict | None = None         # the host's delta over the slot
    write_set: list | None = None          # per env: the attributes the host changed


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
                 e_max: int = unit_events.E_MAX):
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
        # #29 P6.4b: the field outputs' host values at the exit, the books
        # at the entry / exit, the write-set snapshots
        self._fpost = [[None] * self.N for _ in range(k)]
        self._bk0 = [[None] * self.N for _ in range(k)]
        self._bk1 = [[None] * self.N for _ in range(k)]
        self._ws0 = [[None] * self.N for _ in range(k)]
        self._ws1 = [[None] * self.N for _ in range(k)]

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
                    if ls._pre[__k][e] is None:
                        ls._pre[__k][e] = unit_store.pack(sim.units, ls.U)
                        ls._inp[__k][e] = capture(sl.capture_names, sim, a, kw)
                        ls._ev0[__k][e] = (sim.tick_events,
                                           len(sim.tick_events))
                        if sl.outputs:
                            ls._bk0[__k][e] = host_books(sim, sl.book_outputs)
                            ls._ws0[__k][e] = gmap_state(sim.gmap)
                    try:
                        return __orig(*a, **kw)
                    finally:
                        ls._post[__k][e] = unit_store.pack(sim.units, ls.U)
                        evl, n0 = ls._ev0[__k][e]
                        ls._evs[__k][e] = list(evl[n0:])
                        if sl.outputs:
                            ls._fpost[__k][e] = {
                                f: np.array(getattr(sim.gmap, f), copy=True)
                                for f in sl.field_outputs}
                            ls._bk1[__k][e] = host_books(sim, sl.book_outputs)
                            ls._ws1[__k][e] = gmap_state(sim.gmap)
                hooked.__wrapped__ = orig
                setattr(cont, attr, hooked)
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
        names = slot.capture_names
        inputs = gather(names, self._inp[k]) if names else None
        self.last_inputs[slot.label] = inputs
        self.store.upload(pre)
        if slot.events:
            self.ring.begin()
        coverage, books = split_result(slot.call(self.store, inputs, self.ring))
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
                ie = None if inputs is None else inputs.env(e)
                _cov1, books1 = split_result(slot.call(self.alone, ie,
                                                       self.alone_ring))
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
                         host_events=host_n, books=books, host_books=hb,
                         write_set=ws)
        self.reports.append(rep)
        return rep

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
        self._orig = []
        self.coverage: list = []          # per run: [per-slot (1,) arrays]
        self.events: list = []            # per run: the ring's records
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
        covs, books = [], {}
        for s in self.slots:
            c, b = split_result(s.call(self.store, inputs, self.ring))
            covs.append(c)
            if b is not None:
                books.update({n: int(v[0]) for n, v in b.items()})
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
__all__ += ["ARG_INPUTS", "BOOK_PREFIX", "ALIASES", "book_ref", "host_books",
            "OBJECT_PROBES", "gmap_state", "changed_attrs",
            "write_set_violations", "split_result", "first_field_diff"]
