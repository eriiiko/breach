"""THE UNIT SLOT KERNELS' Python face (#29 P6.2; residency design §4.4b,
"Re-check at P6.2's start", item 4).

One function per ported slot, over a :class:`simulation.unit_store.DeviceStore`
built with its device tables (``DeviceStore(n, U, tables=unit_tables.
values_batch(sims))``) and the slot's inputs; each calls the slot's C++ free
function (``cpp/src/cuda_unit_kernels.h``: one TU per slot family over the
device unit library ``cpp/src/cuda_unit_lib.cuh``), returns the per-env
COVERAGE (the rows the kernel changed, ``(N,)`` int64), keeps the per-env
status applications refused at the stack cap in ``store.last_refused``, and
REFUSES the result when a row's operands left the range the kernel was
proved for (:class:`UnitKernelFault`, naming the kernel and the env: the
Python law would have computed numbers the device's int64 cannot hold).

=====================  ====================================  ===============
function               slot (the reference law)              inputs
=====================  ====================================  ===============
:func:`exposure_decay` 1c   ``exposure.decay_units``         --
:func:`integrate_turn` 1a   ``unit_impulse.integrate_all``   --
:func:`wind_impulse`   9c2  ``exchange.apply_wind_force``    field planes
:func:`apply_orders`   1b'  ``unit_orders.apply_requests``   the requests
:func:`expire_orders`  10'  ``unit_orders.expire``           the ticks
:func:`convert_zombies` 10 (ruleset tail) ``ai_zombie.       --
                       convert_marines_to_zombies``
=====================  ====================================  ===============

A NEW SLOT KERNEL is: its law's table values as rows of
``simulation.unit_tables`` (regenerate the header), a free function in its
slot family's ``cpp/src/cuda_unit_<family>.cu`` (declared in
``cuda_unit_kernels.h``, the TU on CMakeLists' CUDA list), a binding in
``bindings.cpp`` returning the three counts, a function here, a Class-T row
in ``tests/twin_manifest.toml``, and its shadow lockstep
(``tests/cuda_unit_lockstep.py``'s ``DeviceSlot``).

NOT PART OF THE PLAYED TICK: Class T, training only.
"""
from __future__ import annotations

import numpy as np

#: The order requests' row: (uid, what, where, until, target); uid -1 = none.
REQUEST_WIDTH = 5
NO_REQUEST = -1


class UnitKernelFault(RuntimeError):
    """A slot kernel refused rows whose operands left its proved range."""


def _done(store, name: str, counts) -> np.ndarray:
    cov, flt, ref = (np.asarray(c, dtype=np.int64) for c in counts)
    store.last_refused = ref
    if flt.any():
        envs = [int(e) for e in np.nonzero(flt)[0]]
        raise UnitKernelFault(
            f"unit kernel {name}: {int(flt.sum())} row(s) in env(s) {envs} "
            f"left the range the kernel was proved for (the rows were left "
            f"as they were)")
    return cov


def exposure_decay(store) -> np.ndarray:
    """Slot 1c: ``exposure.decay_step`` on every valid row's exposure block
    (the tables' ``exposure_decay_q``)."""
    return _done(store, "exposure_decay",
                 store.bp.unit_exposure_decay(store.buffer, store.tables))


def integrate_turn(store) -> np.ndarray:
    """Slot 1a: ``unit_impulse.integrate`` then ``turning.step`` on every
    valid living row."""
    return _done(store, "integrate_turn",
                 store.bp.unit_integrate_turn(store.buffer, store.tables))


#: The planes slot 9c2 reads, by ``GameMap`` field name.
WIND_FIELDS = ("gas", "wind_x", "wind_y", "solid")


def wind_impulse(store, fields: dict) -> np.ndarray:
    """Slot 9c2: the wind's impulse on every valid living row. ``fields``
    maps each of :data:`WIND_FIELDS` to a DEVICE array of ``(N, ...)`` in
    the resident field's own dtype (a CuPy array, or anything with
    ``.data.ptr``, ``.shape`` and ``.dtype``): ``gas`` ``(N, G, h, w)``
    int32, ``wind_x`` / ``wind_y`` ``(N, h, w)`` int32, ``solid``
    ``(N, h, w)`` bool -- ``ResidentBatch.buffers`` as they are."""
    n = store.shape[0]
    want = {"gas": np.int32, "wind_x": np.int32, "wind_y": np.int32,
            "solid": np.bool_}
    for k, dt in want.items():
        a = fields.get(k)
        if a is None:
            raise ValueError(f"unit kernel wind_impulse: field {k!r} missing")
        if np.dtype(a.dtype) != np.dtype(dt):
            raise ValueError(f"unit kernel wind_impulse: field {k!r} is "
                             f"{a.dtype}, the resident field is {np.dtype(dt)}")
        if int(a.shape[0]) != n:
            raise ValueError(f"unit kernel wind_impulse: field {k!r} has "
                             f"{a.shape[0]} envs, the store {n}")
    g = fields["gas"]
    _, n_gases, h, w = (int(x) for x in g.shape)
    for k in ("wind_x", "wind_y", "solid"):
        if tuple(int(x) for x in fields[k].shape) != (n, h, w):
            raise ValueError(f"unit kernel wind_impulse: field {k!r} is "
                             f"{fields[k].shape}, not ({n}, {h}, {w})")
    return _done(store, "wind_impulse", store.bp.unit_wind_impulse(
        store.buffer, store.tables, int(g.data.ptr),
        int(fields["wind_x"].data.ptr), int(fields["wind_y"].data.ptr),
        int(fields["solid"].data.ptr), n_gases, h, w))


def requests_of(sim) -> np.ndarray:
    """``sim._order_requests`` as ``(R, 5)`` int64 rows sorted by unit id
    (the order ``apply_requests`` walks): ``(uid, what, where, until,
    target)``."""
    req = sim._order_requests
    rows = [(int(uid),) + tuple(int(v) for v in req[uid]) for uid in sorted(req)]
    return np.array(rows, dtype=np.int64).reshape(len(rows), REQUEST_WIDTH)


def pad_requests(per_env) -> np.ndarray:
    """N envs' ``(R_e, 5)`` request rows as one ``(N, max(1, R), 5)`` array,
    the short envs padded with uid -1."""
    per_env = [np.asarray(r, dtype=np.int64).reshape(-1, REQUEST_WIDTH)
               for r in per_env]
    R = max([1] + [r.shape[0] for r in per_env])
    out = np.full((len(per_env), R, REQUEST_WIDTH), 0, dtype=np.int64)
    out[:, :, 0] = NO_REQUEST
    for e, r in enumerate(per_env):
        out[e, :r.shape[0]] = r
    return out


def apply_orders(store, requests) -> np.ndarray:
    """Slot 1b': ``requests`` ``(N, R, 5)`` int64 (:func:`pad_requests`)
    written onto their living, non-zombie rows."""
    a = np.ascontiguousarray(requests, dtype=np.int64)
    return _done(store, "apply_orders",
                 store.bp.unit_apply_orders(store.buffer, store.tables, a))


def expire_orders(store, ticks) -> np.ndarray:
    """Slot 10': each env's orders whose tick has come (``ticks`` ``(N,)``,
    each env's ``sim.tick`` after the advance) cleared."""
    a = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    return _done(store, "expire_orders",
                 store.bp.unit_expire_orders(store.buffer, store.tables, a))


def convert_zombies(store) -> np.ndarray:
    """Slot 10's ruleset tail: every row killed by a zombie rises as one.
    The host side (the name's "Z-" prefix, the momentum books) is the
    caller's."""
    return _done(store, "convert_zombies",
                 store.bp.unit_convert_zombies(store.buffer, store.tables))


def lib_probe(store, op: int, a0=0, a1=0, a2=0, a3=0):
    """TEST-ONLY: the device unit library on the store's rows (see
    ``cuda_unit_kernels.h``); returns ``(out (N, U, K), coverage, faults,
    refused)``."""
    return store.bp.unit_lib_probe(store.buffer, store.tables, int(op),
                                   int(a0), int(a1), int(a2), int(a3))


__all__ = ["REQUEST_WIDTH", "NO_REQUEST", "UnitKernelFault",
           "exposure_decay", "integrate_turn", "WIND_FIELDS", "wind_impulse",
           "requests_of", "pad_requests", "apply_orders", "expire_orders",
           "convert_zombies", "lib_probe"]
