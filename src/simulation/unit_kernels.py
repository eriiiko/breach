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
:func:`movement`       3    ``Simulation._update_player_      the material
                       movement`` (continuous path)          plane
:func:`status_tick`    2b   ``status.tick_statuses``         the ring
:func:`heat_damage`    9c   ``exchange.apply_environmental_  the ring, planes,
                       damage``                              the claims
:func:`gas_rows`       9c3  ``apply_teargas_blind``,         the ring, planes,
                       ``apply_poison_dose``, ``poison.tick`` the claims
=====================  ====================================  ===============

#29 P6.3: the slots that hurt run the device ``damage.apply_packet``
(``cpp/src/cuda_unit_damage.cuh``) and append their UnitHit / UnitKilled
records to an :class:`simulation.unit_events.EventRing` (begun by the caller
at the tick's start) in the reference's emission order; each REFUSES a run
whose events the ring could not hold (:class:`simulation.unit_events.
UnitEventOverflow`, naming the kernel and the env) as it refuses a fault. The
credit claims (``damage.TickCredit``) reach them as a per-tick table
(:func:`claims_of`, :func:`pad_claims`).

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

from simulation.unit_events import CREDIT_HEAT, CREDIT_PAD, UnitEventOverflow

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
    The host side (the name's "Z-" prefix, the momentum books, and
    ``ContinuousRealtime.on_tick_end``'s prune of detonated grenades from
    ``sim.projectiles``) is the caller's."""
    return _done(store, "convert_zombies",
                 store.bp.unit_convert_zombies(store.buffer, store.tables))


#: The planes slot 3 reads, by ``GameMap`` field name (#29 P6.2m).
MOVEMENT_FIELDS = ("material",)


def movement(store, fields: dict) -> np.ndarray:
    """Slot 3 (#29 P6.2m), the CONTINUOUS path's ``Simulation.
    _update_player_movement``: every valid living row's MOVE_DIR walk and
    pushed velocity as one displacement (the slide rule against
    ``GameMap.is_passable_block``), a held-no-aim walker's facing target, a
    pushed row by its pushed velocity alone. ``fields["material"]`` is a
    DEVICE array ``(N, h, w)`` int8 -- the resident ``GameMap.material``
    planes, as ``ResidentBatch.buffers`` holds them; the mobility column is
    the tables' ``material_mobility`` row. The host-only side (the momentum
    books' WALL entries, the legacy WEGO path step's ``path_tick_offset``)
    is the caller's."""
    n = store.shape[0]
    a = fields.get("material")
    if a is None:
        raise ValueError("unit kernel movement: field 'material' missing")
    if np.dtype(a.dtype) != np.dtype(np.int8):
        raise ValueError(f"unit kernel movement: field 'material' is {a.dtype}, "
                         f"the resident field is int8")
    if a.ndim != 3 or int(a.shape[0]) != n:
        raise ValueError(f"unit kernel movement: field 'material' is {a.shape}, "
                         f"not ({n}, h, w)")
    _, h, w = (int(x) for x in a.shape)
    return _done(store, "movement", store.bp.unit_movement(
        store.buffer, store.tables, int(a.data.ptr), h, w))


def movement_probe(bp, op: int, args, material=None, mobility=None) -> np.ndarray:
    """TEST-ONLY: the walk's library functions on synthetic rows (see
    ``cuda_unit_kernels.h``, ``unit_movement_probe``): ``args`` ``(M, 6)``
    int64 -> ``(M, 4)`` int64, column 3 the bad flag."""
    args = np.ascontiguousarray(args, dtype=np.int64).reshape(-1, 6)
    mat = np.ascontiguousarray(
        material if material is not None else np.zeros((1, 1)), dtype=np.int8)
    mob = np.ascontiguousarray(
        mobility if mobility is not None else np.zeros(1), dtype=np.int64)
    return bp.unit_movement_probe(int(op), args, mat, mob)
# ---------------------------------------------------------------------------
# #29 P6.3: the slots that hurt
# ---------------------------------------------------------------------------
#: A claims row: (channel, y, x, attacker); channel CREDIT_HEAT for the heat
#: buffer, a gas slice's index for a gas deposit.
CLAIM_WIDTH = 4


def claims_of(sim) -> np.ndarray:
    """``sim.gmap.damage_credit`` (``damage.TickCredit``: ``(channel, y, x)
    -> attacker``, every key once -- the first claim of the tick) as ``(R,
    4)`` int64 rows ``(channel, y, x, attacker)``, sorted; the heat channel
    as :data:`simulation.unit_events.CREDIT_HEAT`."""
    from simulation.damage import CREDIT_HEAT as HOST_HEAT
    tc = getattr(sim.gmap, "damage_credit", None)
    owner = getattr(tc, "_owner", None) or {}
    rows = []
    for (ch, y, x), a in owner.items():
        if ch == HOST_HEAT:
            code = CREDIT_HEAT
        else:
            code = int(ch)
            if code < 0:
                raise ValueError(f"unit kernels: a credit channel {ch!r} the "
                                 f"claims table cannot hold")
        rows.append((code, int(y), int(x), int(a)))
    rows.sort()
    return np.array(rows, dtype=np.int64).reshape(len(rows), CLAIM_WIDTH)


def pad_claims(per_env) -> np.ndarray:
    """N envs' ``(R_e, 4)`` claims as one ``(N, max(1, R), 4)`` array, the
    short envs padded with rows that match no tile (channel CREDIT_PAD)."""
    per_env = [np.asarray(r, dtype=np.int64).reshape(-1, CLAIM_WIDTH)
               for r in per_env]
    R = max([1] + [r.shape[0] for r in per_env])
    out = np.full((len(per_env), R, CLAIM_WIDTH), -1, dtype=np.int64)
    out[:, :, 0] = CREDIT_PAD
    for e, r in enumerate(per_env):
        out[e, :r.shape[0]] = r
    return out


def _done_events(store, name: str, counts) -> np.ndarray:
    cov, flt, ref, drp = (np.asarray(c, dtype=np.int64) for c in counts)
    cov = _done(store, name, (cov, flt, ref))
    store.last_dropped = drp
    if drp.any():
        envs = [int(e) for e in np.nonzero(drp)[0]]
        raise UnitEventOverflow(
            f"unit kernel {name}: {int(drp.sum())} event(s) in env(s) {envs} "
            f"did not fit the event ring (its E_max, or a row's staging): "
            f"nothing was written past the capacity")
    return cov


def _check_planes(name, store, fields, want):
    n = store.shape[0]
    for k, dt in want.items():
        a = fields.get(k)
        if a is None:
            raise ValueError(f"unit kernel {name}: field {k!r} missing")
        if np.dtype(a.dtype) != np.dtype(dt):
            raise ValueError(f"unit kernel {name}: field {k!r} is {a.dtype}, "
                             f"the resident field is {np.dtype(dt)}")
        if int(a.shape[0]) != n:
            raise ValueError(f"unit kernel {name}: field {k!r} has "
                             f"{a.shape[0]} envs, the store {n}")


def _check_ring(name, store, ring):
    if ring is None:
        raise ValueError(f"unit kernel {name}: an EventRing is required")
    if ring.shape[0] != store.shape[0] or ring.n_units != store.shape[1]:
        raise ValueError(f"unit kernel {name}: the ring is for ({ring.shape[0]} "
                         f"envs, {ring.n_units} rows), the store "
                         f"{store.shape[:2]}")


def status_tick(store, ring) -> np.ndarray:
    """Slot 2b: ``status.tick_statuses`` on every valid living row (the
    sweep, each damage-over-time status's packet, the countdown); events to
    ``ring``."""
    _check_ring("status_tick", store, ring)
    return _done_events(store, "status_tick", store.bp.unit_status_tick(
        store.buffer, store.tables, ring.buffer))


#: The planes slot 9c reads, by ``GameMap`` field name.
HEAT_FIELDS = ("heat", "rad_flux_sweep")
#: The planes slot 9c3 reads.
GAS_FIELDS = ("gas",)


def heat_damage(store, ring, fields: dict, claims) -> np.ndarray:
    """Slot 9c: the heat row on every valid living row. ``fields``: ``heat``
    ``(N, h, w)`` int32 and ``rad_flux_sweep`` ``(N, h, w)`` int64 DEVICE
    arrays (``ResidentBatch.buffers`` as they are); ``claims`` ``(N, R, 4)``
    (:func:`pad_claims`)."""
    _check_ring("heat_damage", store, ring)
    _check_planes("heat_damage", store, fields,
                  {"heat": np.int32, "rad_flux_sweep": np.int64})
    _, h, w = (int(x) for x in fields["heat"].shape)
    if tuple(int(x) for x in fields["rad_flux_sweep"].shape) != \
            (store.shape[0], h, w):
        raise ValueError("unit kernel heat_damage: rad_flux_sweep's shape is "
                         "not heat's")
    c = np.ascontiguousarray(claims, dtype=np.int64)
    return _done_events(store, "heat_damage", store.bp.unit_heat_damage(
        store.buffer, store.tables, ring.buffer, int(fields["heat"].data.ptr),
        int(fields["rad_flux_sweep"].data.ptr), h, w, c))


def gas_rows(store, ring, fields: dict, claims) -> np.ndarray:
    """Slot 9c3: the teargas row, the poison row's dose and the poison law,
    fused per row (each unit's three steps read only the planes and its own
    row). ``fields``: ``gas`` ``(N, G, h, w)`` int32 DEVICE array;
    ``claims`` as for :func:`heat_damage`. ``store.last_refused`` holds the
    doses clipped at the body's store (poison.SATURATION's hits)."""
    _check_ring("gas_rows", store, ring)
    _check_planes("gas_rows", store, fields, {"gas": np.int32})
    _, n_gases, h, w = (int(x) for x in fields["gas"].shape)
    c = np.ascontiguousarray(claims, dtype=np.int64)
    return _done_events(store, "gas_rows", store.bp.unit_gas_rows(
        store.buffer, store.tables, ring.buffer, int(fields["gas"].data.ptr),
        n_gases, h, w, c))


#: A probe packet row (TEST-ONLY :func:`damage_probe`).
PROBE_PACKET = ("amount_q", "dtype", "attacker", "indirect", "expose",
                "dir_x_q", "dir_y_q", "ap_q", "cause", "mark_killed_by_zombie")


def damage_probe(store, ring, packets):
    """TEST-ONLY: the device ``damage.apply_packet`` on the store's rows, one
    packet per row (``(N, U, 10)`` int64, :data:`PROBE_PACKET`); returns
    ``(out (N, U, 2): applied damage, killed; coverage, faults, refused,
    dropped)`` -- unrefused, so a gate can count faults."""
    _check_ring("damage_probe", store, ring)
    a = np.ascontiguousarray(packets, dtype=np.int64)
    return store.bp.unit_damage_probe(store.buffer, store.tables, ring.buffer, a)


def lib_probe(store, op: int, a0=0, a1=0, a2=0, a3=0):
    """TEST-ONLY: the device unit library on the store's rows (see
    ``cuda_unit_kernels.h``); returns ``(out (N, U, K), coverage, faults,
    refused)``."""
    return store.bp.unit_lib_probe(store.buffer, store.tables, int(op),
                                   int(a0), int(a1), int(a2), int(a3))


__all__ = ["REQUEST_WIDTH", "NO_REQUEST", "UnitKernelFault",
           "exposure_decay", "integrate_turn", "WIND_FIELDS", "wind_impulse",
           "requests_of", "pad_requests", "apply_orders", "expire_orders",
           "convert_zombies", "lib_probe", "MOVEMENT_FIELDS", "movement",
           "movement_probe", "CLAIM_WIDTH", "claims_of",
           "pad_claims", "status_tick", "HEAT_FIELDS", "GAS_FIELDS",
           "heat_damage", "gas_rows", "PROBE_PACKET", "damage_probe"]
