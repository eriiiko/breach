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
name); every other slot's by nothing. Any other sim (the reference match in
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

from simulation import unit_kernels, unit_store, unit_tables


# ---------------------------------------------------------------------------
# Slot inputs
# ---------------------------------------------------------------------------
FIELD_PREFIX = "field:"


def _capture_tick(sim):
    return int(sim.tick)


def _capture_requests(sim):
    return unit_kernels.requests_of(sim)


#: name -> (kind, capture function of the sim at the slot's entry). kind
#: "tick" gathers to ``SlotInputs.tick``; "table" to ``SlotInputs.tables``
#: (an ``(R, C)`` int64 per env, padded across envs by ``pad``).
INPUTS: dict = {
    "tick": ("tick", _capture_tick),
    "order_requests": ("table", _capture_requests),
}
#: How a table input pads across envs (default: zeros).
PADS: dict = {"order_requests": unit_kernels.pad_requests}


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
    ``inputs`` it reads (names of :data:`INPUTS` or ``field:<name>``)."""
    label: str
    run: Callable
    columns: tuple = ()
    inputs: tuple = ()

    def call(self, store, inputs: SlotInputs | None):
        if self.inputs:
            return self.run(store, inputs)
        return self.run(store)


@dataclass
class TickReport:
    tick: int                      # the sims' tick after the step
    coverage: np.ndarray           # (N,) the kernel's changed rows
    host_changed: np.ndarray       # (N,) rows the host slot changed
    equal: bool
    first_diff: tuple | None = None   # (tick, env, unit id, column)
    label: str = ""                   # the slot's label


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
                 check_alone: bool = True, n_units: int | None = None):
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
                    try:
                        return __orig(*a, **kw)
                    finally:
                        ls._post[__k][e] = unit_store.pack(sim.units, ls.U)
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
        inputs = gather(slot.inputs, self._inp[k]) if slot.inputs else None
        self.last_inputs[slot.label] = inputs
        self.store.upload(pre)
        coverage = np.asarray(slot.call(self.store, inputs), dtype=np.int64)
        got = self.store.download()
        host_changed = np.array([int((pre[e] != post[e]).any(axis=1).sum())
                                 for e in range(self.N)], dtype=np.int64)
        first = self._first_diff(tick, got, post, self._masks[k])
        if first is None and self.alone is not None:
            for e in range(self.N):
                self.alone.upload_tables(self.tables[e:e + 1])
                self.alone.upload(pre[e:e + 1])
                slot.call(self.alone, None if inputs is None else inputs.env(e))
                one = self.alone.download()[0]
                if not np.array_equal(one, got[e]):
                    r, col = np.argwhere(one != got[e])[0]
                    first = Divergence(tick, e,
                                       int(pre[e, r, self.layout.column("id").offset]),
                                       "ALONE " + self.layout.column_at(int(col)),
                                       int(one[r, col]), int(got[e, r, col]))
                    break
        if first is not None:
            first.label = slot.label
            self.divergences.append(first)
        rep = TickReport(tick=tick, coverage=coverage,
                         host_changed=host_changed, equal=first is None,
                         first_diff=None if first is None else (
                             first.tick, first.env, first.unit_id,
                             first.column),
                         label=slot.label)
        self.reports.append(rep)
        return rep

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

    def __init__(self, driver, slots, host_side=None, bp=None):
        self.driver = driver
        self.slots = list(slots)
        self.host_side = host_side
        self.U = len(driver.sim.units)
        self.store = build_store([driver.sim], self.U, bp=bp)
        self._orig = []
        self.coverage: list = []          # per run: [per-slot (1,) arrays]
        self.runs = 0

    def _run_chain(self):
        sim = self.driver.sim
        names = []
        for s in self.slots:
            names.extend(n for n in s.inputs if n not in names)
        inputs = gather(names, [capture(names, sim)]) if names else None
        pre = unit_store.pack(sim.units, self.U)
        self.store.upload(pre[None])
        covs = [np.asarray(s.call(self.store, inputs), dtype=np.int64)
                for s in self.slots]
        post = self.store.download()[0]
        unit_store.unpack(post, sim.units)
        if self.host_side is not None:
            self.host_side(sim, pre, post)
        self.coverage.append(covs)
        self.runs += 1

    def install(self):
        ch = self
        sim = self.driver.sim
        for i, s in enumerate(self.slots):
            for cont, attr in slot_callables(s.label):
                orig = getattr(cont, attr)

                def replaced(*a, __orig=orig, __first=(i == 0), **k):
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
