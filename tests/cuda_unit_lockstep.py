"""#29 P6 -- THE SHADOW-LOCKSTEP HARNESS (design §4.4b decision 4): a device
slot kernel checked against the reference slot, row for row, on live matches.

The reference plays every tick in Python as always; the device NEVER drives
the game. Around a named slot (a ``tools/tick_stage_profile.py`` label -- the
twin manifest's own vocabulary) the harness, per env and per tick:

1. packs the env's units (``simulation.unit_store.pack``) when the slot is
   entered -- the rows the slot starts from;
2. lets the host run the slot;
3. packs again when the slot returns -- the rows the slot ended with;

and once every env has ticked, uploads the batch's start rows to the
``DeviceStore``, runs the slot's device kernel over all N envs, downloads
and compares with the host's end rows at tol 0, reporting the first
difference as (tick, env, unit id, column). It also runs each env's start
rows ALONE through an N = 1 store (``check_alone``) and requires the batch's
env i to equal it (env i of N == env i alone), and it counts the rows the
host changed, so a kernel's coverage counter can be held to it.

Hooking only READS the units (two packs) around the original callable, so a
hooked match plays exactly as an unhooked one (``cuda_unit_store_check``
proves it: equal per-tick unit hashes and field digests).

THE API the later P6 patches extend:

    slot = DeviceSlot(label="1c exposure decay",
                      run=lambda store: store.exposure_decay(decay_q))
    with ShadowLockstep(drivers, slot) as ls:      # drivers: .sim + .tick()
        for _ in range(T):
            rep = ls.tick()                        # a TickReport
    ls.reports, ls.divergences

``DeviceSlot.columns`` limits the compare to named columns (default: the
whole row). A label naming several callables (``tick_stage_profile`` sums
them) is hooked on each: the first entry in a tick packs the start rows,
every exit repacks the end rows. The slot must run once in every env every
tick (a slot gated off in an env raises, naming it -- a later patch whose
slot is gated extends ``tick``).

Not a test module (no ``test_`` prefix): imported by
``tests/cuda_unit_store_check.py`` inside the CUDA subprocess.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from simulation import unit_store


@dataclass(frozen=True)
class DeviceSlot:
    """One ported slot: its profiler ``label`` (the callables hooked are
    that label's ``SLOT_WRAPS`` rows) and ``run(store) -> coverage (N,)``,
    the device kernel over the uploaded store, returning per env the rows it
    changed."""
    label: str
    run: Callable
    columns: tuple = ()


@dataclass
class TickReport:
    tick: int                      # the sims' tick after the step
    coverage: np.ndarray           # (N,) the kernel's changed rows
    host_changed: np.ndarray       # (N,) rows the host slot changed
    equal: bool
    first_diff: tuple | None = None   # (tick, env, unit id, column)


@dataclass
class Divergence:
    tick: int
    env: int
    unit_id: int
    column: str
    host: int
    device: int

    def __str__(self):
        return (f"tick {self.tick} env {self.env} unit {self.unit_id} "
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


class ShadowLockstep:
    """The harness over ``drivers`` (each with ``.sim`` and ``.tick()``: a
    ``MatchSession``, or any loop that steps one sim once). U is the
    largest env's unit count at construction (a match spawns once; a unit
    is never removed)."""

    def __init__(self, drivers, slot: DeviceSlot, bp=None,
                 check_alone: bool = True, n_units: int | None = None):
        self.drivers = list(drivers)
        self.slot = slot
        self.N = len(self.drivers)
        self.U = int(n_units) if n_units is not None else max(
            len(d.sim.units) for d in self.drivers)
        self.store = unit_store.DeviceStore(self.N, self.U, bp=bp)
        self.alone = (unit_store.DeviceStore(1, self.U, bp=bp)
                      if check_alone and self.N > 1 else None)
        self.layout = unit_store.layout()
        self._mask = self._column_mask(slot.columns)
        self._orig = []
        self._env = None
        self._pre = [None] * self.N
        self._post = [None] * self.N
        self.reports: list = []
        self.divergences: list = []

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
        for cont, attr in slot_callables(self.slot.label):
            orig = getattr(cont, attr)

            def hooked(*a, __orig=orig, **k):
                e = ls._env
                if e is None:
                    return __orig(*a, **k)
                sim = ls.drivers[e].sim
                if ls._pre[e] is None:
                    ls._pre[e] = unit_store.pack(sim.units, ls.U)
                try:
                    return __orig(*a, **k)
                finally:
                    ls._post[e] = unit_store.pack(sim.units, ls.U)
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
    def tick(self) -> TickReport:
        """Tick every driver once (the slot hooked), then the device slot
        over the batch, then the compare. Returns the tick's report (also
        appended to ``reports``; a difference to ``divergences``)."""
        self._pre = [None] * self.N
        self._post = [None] * self.N
        for e, d in enumerate(self.drivers):
            self._env = e
            try:
                d.tick()
            finally:
                self._env = None
            if self._pre[e] is None:
                raise RuntimeError(
                    f"lockstep: slot {self.slot.label!r} did not run in env "
                    f"{e} at tick {d.sim.tick}")
        pre = np.stack(self._pre)
        post = np.stack(self._post)
        self.store.upload(pre)
        coverage = np.asarray(self.slot.run(self.store), dtype=np.int64)
        got = self.store.download()
        host_changed = np.array([int((pre[e] != post[e]).any(axis=1).sum())
                                 for e in range(self.N)], dtype=np.int64)
        tick = int(self.drivers[0].sim.tick)
        first = self._first_diff(tick, got, post)
        if first is None and self.alone is not None:
            for e in range(self.N):
                self.alone.upload(pre[e:e + 1])
                self.slot.run(self.alone)
                one = self.alone.download()[0]
                if not np.array_equal(one, got[e]):
                    r, col = np.argwhere(one != got[e])[0]
                    first = Divergence(tick, e,
                                       int(pre[e, r, self.layout.column("id").offset]),
                                       "ALONE " + self.layout.column_at(int(col)),
                                       int(one[r, col]), int(got[e, r, col]))
                    break
        if first is not None:
            self.divergences.append(first)
        rep = TickReport(tick=tick, coverage=coverage,
                         host_changed=host_changed, equal=first is None,
                         first_diff=None if first is None else (
                             first.tick, first.env, first.unit_id,
                             first.column))
        self.reports.append(rep)
        return rep

    def _first_diff(self, tick, got, want):
        diff = got != want
        if self._mask is not None:
            diff &= self._mask
        if not diff.any():
            return None
        e, r, col = (int(x) for x in np.argwhere(diff)[0])
        return Divergence(tick, e, int(want[e, r, self.layout.column("id").offset]),
                          self.layout.column_at(col), int(want[e, r, col]),
                          int(got[e, r, col]))


__all__ = ["DeviceSlot", "TickReport", "Divergence", "slot_callables",
           "ShadowLockstep"]
