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


# ---------------------------------------------------------------------------
# #29 P6.4a: slot 1b's hand verbs and the done kernel
# ---------------------------------------------------------------------------
def _check_requests(name, store, requests):
    if requests is None:
        raise ValueError(f"unit kernel {name}: a RequestList is required")
    if requests.shape[0] != store.shape[0] or requests.n_units != store.shape[1]:
        raise ValueError(f"unit kernel {name}: the request list is for "
                         f"({requests.shape[0]} envs, {requests.n_units} rows), the "
                         f"store {store.shape[:2]}")


def verbs(store, requests, ticks, door_spans, charges_live) -> np.ndarray:
    """Slot 1b (``Simulation._consume_direct_intents``, the continuous path):
    a dead or turned row drops its trigger, wind-up and plant; every living
    marine consumes its own latches (throw, plant, use, reload, shove) and
    advances its wind-up and plant. ``requests`` a begun
    :class:`simulation.unit_requests.RequestList` (the grenades released, the
    charges laid -- accepted or refused at the capacity --, the doors flipped,
    in id order); ``ticks`` ``(N,)`` each env's ``sim.tick``;
    ``door_spans`` ``(N, R, 3)`` (``unit_requests.pad_door_spans``);
    ``charges_live`` ``(N,)`` the live planted charges. ``store.last_refused``
    holds the plants refused at ``CHARGE_CAPACITY``. REFUSES a run whose
    requests the list could not hold (:class:`simulation.unit_requests.
    UnitRequestOverflow`, naming the kernel and the env) as it refuses a
    fault."""
    from simulation.unit_requests import UnitRequestOverflow
    _check_requests("verbs", store, requests)
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    s = np.ascontiguousarray(door_spans, dtype=np.int64)
    lv = np.ascontiguousarray(charges_live, dtype=np.int64).reshape(-1)
    cov, flt, ref, drp = (np.asarray(c, dtype=np.int64) for c in store.bp.unit_verbs(
        store.buffer, store.tables, requests.buffer, t, s, lv))
    cov = _done(store, "verbs", (cov, flt, ref))
    store.last_dropped = drp
    if drp.any():
        envs = [int(e) for e in np.nonzero(drp)[0]]
        raise UnitRequestOverflow(
            f"unit kernel verbs: {int(drp.sum())} request(s) in env(s) {envs} "
            f"did not fit the request list (its R_max, or a row's staging): "
            f"nothing was written past the capacity")
    return cov


#: The done kernel's columns.
DONE_COLUMNS = ("terminal", "outcome", "winner", "living_teams")
DONE_NONE, DONE_WIN, DONE_DRAW = 0, 1, 2


def match_done(store, ticks) -> np.ndarray:
    """The done kernel (``MatchRules.is_terminal`` / ``living_teams`` /
    ``result``) on every env: ``(N, 4)`` int64 :data:`DONE_COLUMNS` -- the
    outcome :data:`DONE_NONE` while not terminal, :data:`DONE_WIN` (the
    winner the last marine team standing) or :data:`DONE_DRAW`; ``ticks``
    ``(N,)`` each env's ``sim.tick`` after the step; the limit is the tables'
    ``match_limit_ticks`` (0: the sim plays no match, no limit)."""
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    return np.asarray(store.bp.unit_match_done(store.buffer, store.tables, t))


def verbs_probe(store, kind=0, tick=0, base=0) -> np.ndarray:
    """TEST-ONLY: the hand verbs' gates on the store's rows (see
    ``cuda_unit_kernels.h``, ``unit_verbs_probe``) -> ``(N, U, 16)``."""
    return np.asarray(store.bp.unit_verbs_probe(store.buffer, store.tables,
                                                int(kind), int(tick), int(base)))


__all__ = ["REQUEST_WIDTH", "NO_REQUEST", "UnitKernelFault",
           "verbs", "DONE_COLUMNS", "DONE_NONE", "DONE_WIN", "DONE_DRAW",
           "match_done", "verbs_probe",
           "exposure_decay", "integrate_turn", "WIND_FIELDS", "wind_impulse",
           "requests_of", "pad_requests", "apply_orders", "expire_orders",
           "convert_zombies", "lib_probe", "MOVEMENT_FIELDS", "movement",
           "movement_probe", "CLAIM_WIDTH", "claims_of",
           "pad_claims", "status_tick", "HEAT_FIELDS", "GAS_FIELDS",
           "heat_damage", "gas_rows", "PROBE_PACKET", "damage_probe"]


# ---------------------------------------------------------------------------
# #29 P6.4b: the per-env serial applier (slot 9's burn-through) and the
# tick-end clears -- the FIELD side: their outputs are planes and books
# ---------------------------------------------------------------------------
#: A burn-through list's row: (y, x, valid); a padding row is valid 0.
WALL_WIDTH = 3


def walls_of(destroyed) -> np.ndarray:
    """Slot 9's input -- slot 7's burn-through list, in its order -- as
    ``(R, 3)`` int64 rows ``(y, x, 1)``."""
    rows = [(int(y), int(x), 1) for (y, x) in destroyed]
    return np.array(rows, dtype=np.int64).reshape(len(rows), WALL_WIDTH)


def pad_walls(per_env) -> np.ndarray:
    """N envs' ``(R_e, 3)`` wall rows as one ``(N, max(1, R), 3)`` array, the
    short envs padded with ``valid = 0`` rows."""
    per_env = [np.asarray(r, dtype=np.int64).reshape(-1, WALL_WIDTH)
               for r in per_env]
    R = max([1] + [r.shape[0] for r in per_env])
    out = np.zeros((len(per_env), R, WALL_WIDTH), dtype=np.int64)
    for e, r in enumerate(per_env):
        out[e, :r.shape[0]] = r
    return out


def _plane_ptrs(name, n_env, fields):
    """The topology planes' device addresses in ``applier.PLANES`` order,
    checked (dtype, env count, one (h, w), the per-cell width), and the
    shape ``(n_gases, h, w)``."""
    from simulation import applier
    g = fields.get("gas")
    if g is None:
        raise ValueError(f"unit kernel {name}: field 'gas' missing")
    if g.ndim != 4:
        raise ValueError(f"unit kernel {name}: field 'gas' is {g.shape}, "
                         f"not (N, G, h, w)")
    _, n_gases, h, w = (int(x) for x in g.shape)
    ptrs = np.zeros(applier.N_PLANES, dtype=np.uint64)
    for k, (pname, dt, cw) in enumerate(applier.PLANES):
        a = fields.get(pname)
        if a is None:
            raise ValueError(f"unit kernel {name}: field {pname!r} missing")
        if np.dtype(a.dtype) != np.dtype(dt):
            raise ValueError(f"unit kernel {name}: field {pname!r} is {a.dtype}, "
                             f"the GameMap field is {np.dtype(dt)}")
        want = ((n_env, n_gases, h, w) if cw == "G" else
                (n_env, h, w) if cw == 1 else (n_env, h, w, int(cw)))
        if tuple(int(x) for x in a.shape) != want:
            raise ValueError(f"unit kernel {name}: field {pname!r} is {a.shape}, "
                             f"not {want}")
        if not a.flags.c_contiguous:
            raise ValueError(f"unit kernel {name}: field {pname!r} is not "
                             f"C-contiguous")
        ptrs[k] = int(a.data.ptr)
    return ptrs, n_gases, h, w


def walk_effects(store, dev_applier, fields: dict, stage: int, ring=None):
    """THE PER-ENV SERIAL APPLIER's walk (#29 P6.4b; P7a.1: one kernel, a
    STAGE argument): every env's effect list (``dev_applier``, a
    :class:`simulation.applier.DeviceApplier` -- staged from the host, or
    appended by a device producer, or both) walked in order, each effect a
    device function: DESTROY_WALL / CHEW over the topology seam on the env's
    planes -- ``fields`` maps every ``applier.PLANES`` name to a DEVICE array
    ``(N, ...)`` in the GameMap field's dtype (``ResidentBatch.buffers`` as
    they are), written IN PLACE --, HIT / STATUS / PUSH on the store's rows
    (a HIT's events into ``ring``, a :class:`simulation.unit_events.
    EventRing`; in a stage that lands, with the exposure landing). Returns
    ``(coverage, books)``: the rows the walk changed per env and the books
    ``{name: (N,) int64}``. Refuses (after the run: the first ``cap`` effects
    ARE applied) a run that dropped effects (:class:`simulation.applier.
    EffectOverflow`), dropped events (``UnitEventOverflow``), met a record it
    may not apply (:class:`simulation.applier.EffectRefused`: a PAYLOAD until
    P7c.1) or faulted (:class:`simulation.applier.ApplierFault`), naming the
    kernel and the envs."""
    from simulation import applier
    n = store.shape[0]
    name = f"applier[{applier.STAGE_NAMES[stage]}]"
    if dev_applier.n_env != n:
        raise ValueError(f"unit kernel {name}: the applier is for "
                         f"{dev_applier.n_env} envs, the store {n}")
    if ring is not None:
        _check_ring(name, store, ring)
    ptrs, n_gases, h, w = _plane_ptrs(name, n, fields)
    b, cov, rdrop = store.bp.applier_run(
        dev_applier.buffer, store.buffer, store.tables, ptrs, n_gases, h, w,
        int(stage), None if ring is None else ring.buffer)
    books = applier.books_dict(b)
    cov = np.asarray(cov, dtype=np.int64)
    store.last_books = books
    store.last_coverage = cov
    if books["dropped"].any():
        envs = [int(e) for e in np.nonzero(books["dropped"])[0]]
        raise applier.EffectOverflow(
            f"unit kernel {name}: {int(books['dropped'].sum())} effect(s) "
            f"in env(s) {envs} did not fit the applier's list (cap "
            f"{dev_applier.cap}, or a row's staging): the first {dev_applier.cap} "
            f"were applied")
    if np.asarray(rdrop).any():
        envs = [int(e) for e in np.nonzero(np.asarray(rdrop))[0]]
        raise UnitEventOverflow(
            f"unit kernel {name}: {int(np.asarray(rdrop).sum())} event(s) in "
            f"env(s) {envs} did not fit the event ring")
    if books["refused"].any():
        envs = [int(e) for e in np.nonzero(books["refused"])[0]]
        raise applier.EffectRefused(
            f"unit kernel {name}: {int(books['refused'].sum())} record(s) in "
            f"env(s) {envs} of a kind with no device function yet "
            f"({', '.join(applier.UNAPPLIED_KINDS)}: execute_payload arrives with "
            f"P7c.1): not applied")
    if books["faults"].any():
        envs = [int(e) for e in np.nonzero(books["faults"])[0]]
        raise applier.ApplierFault(
            f"unit kernel {name}: {int(books['faults'].sum())} effect(s) "
            f"in env(s) {envs} left the range the applier was proved for "
            f"(nothing written for them)")
    return cov, books


def apply_effects(store, dev_applier, fields: dict, per_env,
                  stage: int | None = None, ring=None) -> dict:
    """The effect lists ``per_env`` (a list of ``(R_e, applier.F)`` int64,
    e.g. :func:`simulation.applier.destroy_records`) STAGED from the host
    into ``dev_applier`` and walked (:func:`walk_effects`, default stage
    TOPOLOGY). Returns the books (``store.last_coverage`` the rows changed)."""
    from simulation import applier
    dev_applier.stage_lists(per_env)
    _cov, books = walk_effects(store, dev_applier, fields,
                               applier.STAGE_TOPOLOGY if stage is None else stage,
                               ring)
    return books


def resolve(store, dev_applier, fields: dict, ring) -> tuple:
    """Slot 4c (``SimultaneousEffects.resolve`` of the attack slot): the
    effect list as the producers left it, walked in the RESOLVE stage (the
    HITs with the exposure landing). Returns ``(coverage, books)``."""
    from simulation import applier
    return walk_effects(store, dev_applier, fields, applier.STAGE_RESOLVE, ring)


def melee(store, dev_applier, ticks) -> np.ndarray:
    """Slot 4a (``combat.process_melee_intents``, the melee verb): every
    valid row's latch consumed; a striker that can (``verb_gates.can_melee``)
    and finds a target in its arc writes ``last_blow_tick`` and APPENDS its
    blow (HIT, STATUS, PUSH) to ``dev_applier``'s effect list, gathered per
    env in row order after what the list holds. ``ticks`` ``(N,)`` each
    env's ``sim.tick``. Refuses a run whose records did not fit (the list's
    cap or a row's staging: :class:`simulation.applier.EffectOverflow`) as it
    refuses a fault."""
    from simulation import applier
    if dev_applier.n_env != store.shape[0] or dev_applier.n_units != store.shape[1]:
        raise ValueError(f"unit kernel melee: the effect list is for "
                         f"({dev_applier.n_env} envs, {dev_applier.n_units} rows), "
                         f"the store {store.shape[:2]}")
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    cov, flt, ref, drp = (np.asarray(c, dtype=np.int64) for c in store.bp.unit_melee(
        store.buffer, store.tables, dev_applier.buffer, t))
    cov = _done(store, "melee", (cov, flt, ref))
    store.last_dropped = drp
    if drp.any():
        envs = [int(e) for e in np.nonzero(drp)[0]]
        raise applier.EffectOverflow(
            f"unit kernel melee: {int(drp.sum())} effect record(s) in env(s) "
            f"{envs} did not fit the effect list (its cap, or a row's staging "
            f"of {dev_applier.row_cap}): nothing was written past the capacity")
    return cov


def burn_through(store, dev_applier, fields: dict, walls) -> tuple:
    """Slot 9 (``Simulation._burn_through_walls``): each env's burn-through
    list (``walls`` ``(N, R, 3)``, :func:`pad_walls`) as DESTROY_WALL
    effects through :func:`apply_effects`. Returns ``(coverage, books)``:
    coverage the destroys that acted per env. The host-only side (the render
    marks, the sky mask's flag, ``_tile_serial``, the solid books' sink, the
    render-only Wall / DoorDestroyedEvent) is the caller's."""
    from simulation import applier
    w = np.asarray(walls, dtype=np.int64)
    per_env = [applier.destroy_records(w[e][w[e, :, 2] != 0][:, :2])
               for e in range(w.shape[0])]
    books = apply_effects(store, dev_applier, fields, per_env)
    return books["applied"].copy(), books


def tick_end_clears(bp, fields: dict, active=None) -> None:
    """The tick-end clears (``Simulation._clear_tick_buffers``' planes,
    ``applier.CLEAR_PLANES``) zeroed in place on every env whose ``active``
    byte is set (default: all; the reference clears only with a physics
    runner). ``fields`` maps each clear plane to a DEVICE ``(N, h, w)``
    array; ``active`` a DEVICE ``(N,)`` uint8 array or None."""
    from simulation import applier
    ptrs = np.zeros(len(applier.CLEAR_PLANES), dtype=np.uint64)
    shape = None
    for k, (pname, dt) in enumerate(applier.CLEAR_PLANES):
        a = fields.get(pname)
        if a is None:
            raise ValueError(f"unit kernel tick_end_clears: field {pname!r} missing")
        if np.dtype(a.dtype) != np.dtype(dt) or a.ndim != 3:
            raise ValueError(f"unit kernel tick_end_clears: field {pname!r} is "
                             f"{a.dtype} {a.shape}, not (N, h, w) {np.dtype(dt)}")
        if shape is None:
            shape = tuple(int(x) for x in a.shape)
        elif tuple(int(x) for x in a.shape) != shape:
            raise ValueError("unit kernel tick_end_clears: the planes' shapes differ")
        ptrs[k] = int(a.data.ptr)
    n, h, w = shape
    act = 0
    if active is not None:
        if np.dtype(active.dtype) != np.dtype(np.uint8) or tuple(active.shape) != (n,):
            raise ValueError("unit kernel tick_end_clears: active must be (N,) uint8")
        act = int(active.data.ptr)
    bp.tick_end_clears(n, h, w, ptrs, act)


def topology_probe(store, dev_applier, fields: dict, per_env) -> dict:
    """TEST-ONLY: ``material[y, x] = m; on_tile_changed(y, x)`` on the device
    for each env's ``(y, x, m)`` rows (``cpp/src/cuda_topology_probe.cu``),
    on the planes in place; returns the books, unrefused."""
    from simulation import applier
    n = store.shape[0]
    ptrs, n_gases, h, w = _plane_ptrs("topology_probe", n, fields)
    recs_in = []
    for r in per_env:
        r = np.asarray(r, dtype=np.int64).reshape(-1, 3)
        out = np.zeros((r.shape[0], applier.F), dtype=np.int64)
        out[:, 1:4] = r          # the first three arg slots: (y, x, m)
        recs_in.append(out)
    recs, counts = applier.pack(recs_in, dev_applier.cap)
    dev_applier.stage(recs, counts)
    return applier.books_dict(store.bp.topology_probe_tile_changed(
        dev_applier.buffer, store.tables, ptrs, n_gases, h, w))


__all__ += ["walk_effects", "resolve", "melee"]
__all__ += ["WALL_WIDTH", "walls_of", "pad_walls", "apply_effects",
            "burn_through", "tick_end_clears", "topology_probe"]
