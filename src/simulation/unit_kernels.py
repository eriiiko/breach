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


def walk_effects(store, dev_applier, fields: dict, stage: int, ring=None,
                 lists=None, refused_ok: bool = False):
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
    kernel and the envs. #29 P7a.3: ``lists`` (optional) the walk's further
    reach, a :class:`WalkLists` (the cell buffer a SPRAY_CELLS span points
    into, the device edit door, the claims list); a SPRAY_CELLS record with
    none faults. Their overflows are refused by name after the walk
    (``attack_lists.ListOverflow``, ``edit_door.DoorOverflow``). #29 P7c.1:
    a PAYLOAD record is applied (``cpp/src/cuda_unit_payload.cuh``) and needs
    every list of ``lists`` (the door, the claims, ``noise``, ``blast_log``)
    and the ring, or it faults; a PAYLOAD of a refused payload row
    (``applier.PAYLOAD_REFUSALS``) writes nothing and is refused by name
    unless ``refused_ok`` (a chained run that replays those rows on the
    host)."""
    from simulation import applier
    n = store.shape[0]
    name = f"applier[{applier.STAGE_NAMES[stage]}]"
    if dev_applier.n_env != n:
        raise ValueError(f"unit kernel {name}: the applier is for "
                         f"{dev_applier.n_env} envs, the store {n}")
    if ring is not None:
        _check_ring(name, store, ring)
    ptrs, n_gases, h, w = _plane_ptrs(name, n, fields)
    wl = lists if lists is not None else WalkLists()
    for nm, obj in (("cells", wl.cells), ("door", wl.door), ("claims", wl.claims),
                    ("noise", wl.noise), ("blast_log", wl.blast_log)):
        ne = None if obj is None else getattr(obj, "n_env", None)
        if obj is not None and ne is None:
            ne = obj.shape[0]                    # an object table (#29 P7c.1)
        if obj is not None and ne != n:
            raise ValueError(f"unit kernel {name}: the {nm} list is for {ne} "
                             f"envs, the store {n}")
    b, cov, rdrop = store.bp.applier_run(
        dev_applier.buffer, store.buffer, store.tables, ptrs, n_gases, h, w,
        int(stage), None if ring is None else ring.buffer,
        None if wl.cells is None else wl.cells.buffer,
        None if wl.door is None else wl.door.buffer,
        None if wl.claims is None else wl.claims.buffer,
        None if wl.noise is None else wl.noise.buffer,
        None if wl.blast_log is None else wl.blast_log.buffer)
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
    if books["refused"].any() and not refused_ok:
        envs = [int(e) for e in np.nonzero(books["refused"])[0]]
        raise applier.EffectRefused(
            f"unit kernel {name}: {int(books['refused'].sum())} record(s) in "
            f"env(s) {envs} refused, not applied: "
            + payload_refusals(store, dev_applier, envs))
    if books["faults"].any():
        envs = [int(e) for e in np.nonzero(books["faults"])[0]]
        raise applier.ApplierFault(
            f"unit kernel {name}: {int(books['faults'].sum())} effect(s) "
            f"in env(s) {envs} left the range the applier was proved for "
            f"(nothing written for them)")
    if wl.door is not None:
        wl.door.check(name)
    if wl.claims is not None:
        wl.claims.check(name)
    for t in (wl.noise, wl.blast_log):          # #29 P7c.1
        if t is not None:
            t.check(name)
    return cov, books


def payload_refusals(store, dev_applier, envs) -> str:
    """#29 P7c.1: the refused PAYLOAD records of ``envs`` named (the payload
    row and why: ``applier.PAYLOAD_REFUSALS`` through the env's
    ``payload_refuse`` table row)."""
    from simulation import applier, unit_tables
    recs, counts, _d = dev_applier.download()
    tab = np.asarray(store.tables.download(), dtype=np.int64)
    col = unit_tables.layout().column("payload_refuse")
    from simulation.weapons import get_tables
    names = list(get_tables().payloads.by_name)
    pc = applier.kind_field("PAYLOAD", "payload")
    out = []
    for e in envs:
        for i in range(min(int(counts[e]), recs.shape[1])):
            r = recs[e, i]
            if int(r[0]) != applier.KIND_PAYLOAD:
                continue
            p = int(r[pc])
            if not 0 <= p < col.width:
                continue
            code = int(tab[e, col.offset + p])
            if code:
                out.append(f"env {e}: PAYLOAD payloads.{names[p]} -- "
                           f"{applier.payload_refusal_reason(code)}")
    return "; ".join(out) if out else "PAYLOAD records of refused rows"


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


def resolve(store, dev_applier, fields: dict, ring, lists=None) -> tuple:
    """Slot 4c (``SimultaneousEffects.resolve`` of the attack slot): the
    effect list as the producers left it, walked in the RESOLVE stage (the
    HITs with the exposure landing; #29 P7a.3: a SPRAY_CELLS record's edits
    through ``lists.door``, its claims onto ``lists.claims``). Returns
    ``(coverage, books)``."""
    from simulation import applier
    return walk_effects(store, dev_applier, fields, applier.STAGE_RESOLVE, ring,
                        lists)


def detonate(store, dev_applier, fields: dict, ring, lists, refused_ok: bool = False):
    """#29 P7c.1: the PAYLOAD records slot 2's grenades / slot 9e(d')'s
    charges appended (:func:`grenades` / :func:`charges_tick`), walked in the
    DETONATE stage right after their kernel -- where the reference calls
    ``execute_payload`` at once: ``lists`` a :class:`WalkLists` with the
    door, the claims, the noise list and the blast log. Returns
    ``(coverage, books)``."""
    from simulation import applier
    return walk_effects(store, dev_applier, fields, applier.STAGE_DETONATE, ring,
                        lists, refused_ok=refused_ok)


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


# ---------------------------------------------------------------------------
# #29 P7.0a: slot 9d (ignition from temperature) and the kit's probes
# ---------------------------------------------------------------------------
#: The planes slot 9d reads and writes, by GameMap field name, with their
#: dtypes (``fire`` and ``ignition_armed`` written in place).
IGNITION_FIELDS = (("temperature", np.int32), ("material", np.int8),
                   ("flammable", np.bool_), ("wall_hp", np.int32),
                   ("solid", np.bool_), ("is_vacuum", np.bool_),
                   ("gas", np.int32), ("fire", np.int32),
                   ("ignition_armed", np.bool_))


def ignition(store, fields: dict) -> np.ndarray:
    """Slot 9d (``combat.apply_temperature_ignition``): the per-cell law on
    every env's planes -- a flammable tile below its threshold re-arms; a
    hot, fuelled, armed, unlit one whose open neighbours' O2 fraction
    exceeds ``o2_frac_ext`` is seeded; a hot burning one disarms.
    ``fields`` maps each :data:`IGNITION_FIELDS` name to a DEVICE array
    ``(N, h, w)`` (``gas`` ``(N, G, h, w)``) in its dtype, C-contiguous
    (``ResidentBatch.buffers`` as they are); ``fire`` and ``ignition_armed``
    are written IN PLACE. The dials are the store's tables (the ``ign_*``
    rows). Returns the cells changed per env; refuses a run in which a
    cell's material or a gas index left the tables
    (:class:`UnitKernelFault`)."""
    n = store.shape[0]
    _check_planes("ignition", store, fields, dict(IGNITION_FIELDS))
    g = fields["gas"]
    if g.ndim != 4:
        raise ValueError(f"unit kernel ignition: field 'gas' is {g.shape}, not (N, G, h, w)")
    _, n_gases, h, w = (int(x) for x in g.shape)
    ptrs = np.zeros(len(IGNITION_FIELDS), dtype=np.uint64)
    for k, (name, _dt) in enumerate(IGNITION_FIELDS):
        a = fields[name]
        want = (n, n_gases, h, w) if name == "gas" else (n, h, w)
        if tuple(int(x) for x in a.shape) != want:
            raise ValueError(f"unit kernel ignition: field {name!r} is {a.shape}, "
                             f"not {want}")
        if not a.flags.c_contiguous:
            raise ValueError(f"unit kernel ignition: field {name!r} is not C-contiguous")
        ptrs[k] = int(a.data.ptr)
    return _done(store, "ignition",
                 store.bp.unit_ignition(store.buffer, store.tables, ptrs, n_gases, h, w))


#: The draws probe's argument row and output columns (TEST-ONLY).
KIT_DRAW_ARGS = ("salt", "ident", "index", "cone_q", "p_q", "salt2")
KIT_DRAW_OUT = ("word", "cone", "roll", "bad", "a0", "b0", "a1", "b1", "a2", "b2")


def kit_draws_probe(bp, keys, args) -> np.ndarray:
    """TEST-ONLY: the device draws (``cpp/src/cuda_unit_draws.cuh``) --
    ``keys`` ``(N, 3)`` (``unit_draws.draw_keys``), ``args`` ``(N, R, 6)``
    (:data:`KIT_DRAW_ARGS`) -> ``(N, R, 10)`` (:data:`KIT_DRAW_OUT`)."""
    from simulation import unit_draws
    unit_draws.check_compiled(bp)
    return np.asarray(bp.unit_kit_draws_probe(
        np.ascontiguousarray(keys, dtype=np.int64),
        np.ascontiguousarray(args, dtype=np.int64)))


def kit_sight_probe(bp, solid, screen, pairs) -> np.ndarray:
    """TEST-ONLY: the device ``has_los`` (``cpp/src/cuda_unit_sight.cuh``)
    over DEVICE ``(N, h, w)`` bool planes ``solid`` / ``screen`` (the
    ``sight_screen``) for ``pairs`` ``(N, R, 4)`` (fy1, fx1, fy2, fx2) ->
    ``(N, R, 3)``: sight, physical, bad."""
    for nm, a in (("solid", solid), ("screen", screen)):
        if np.dtype(a.dtype) != np.dtype(np.bool_) or a.ndim != 3 \
                or not a.flags.c_contiguous:
            raise ValueError(f"kit_sight_probe: {nm} must be a C-contiguous "
                             f"(N, h, w) bool device array")
    if tuple(solid.shape) != tuple(screen.shape):
        raise ValueError("kit_sight_probe: solid and screen differ in shape")
    n, h, w = (int(x) for x in solid.shape)
    return np.asarray(bp.unit_kit_sight_probe(
        int(solid.data.ptr), int(screen.data.ptr), n, h, w,
        np.ascontiguousarray(pairs, dtype=np.int64)))


def kit_falloff_probe(bp, args) -> np.ndarray:
    """TEST-ONLY: ``falloff_floor_q`` / ``falloff_round_q`` on the device for
    ``args`` ``(n, 3)`` (v, d2, r2) -> ``(n, 2)`` (-1 outside the proved
    range)."""
    return np.asarray(bp.unit_kit_falloff_probe(
        np.ascontiguousarray(args, dtype=np.int64).reshape(-1, 3)))


__all__ += ["IGNITION_FIELDS", "ignition", "KIT_DRAW_ARGS", "KIT_DRAW_OUT",
            "kit_draws_probe", "kit_sight_probe", "kit_falloff_probe"]


__all__ += ["walk_effects", "resolve", "melee"]
__all__ += ["WALL_WIDTH", "walls_of", "pad_walls", "apply_effects",
            "burn_through", "tick_end_clears", "topology_probe"]


# ---------------------------------------------------------------------------
# #29 P7a.2: slot 4's shooting and slot 2''s rounds (cpp/src/cuda_unit_shooting.cu
# over THE MARCH, cpp/src/cuda_unit_march.cuh)
# ---------------------------------------------------------------------------
#: The planes the march reads (GameMap fields, in their own dtypes).
SHOOT_FIELDS = ("solid", "material")


class ShotRefused(RuntimeError):
    """#29 P7a.2: a state the device shooting does not port (named): a hitscan
    trigger (the beam), a ruleset that is not continuous, a unit holding WEGO
    orders, a WEGO draws tick base, a round under physical cover."""


def shooting_door(sim) -> None:
    """THE DOOR of the device shooting (design 4.4c scope: the continuous path
    only): refuses by name a sim whose ruleset is not ``ContinuousRealtime``
    (or drives the units: OnePhaseWEGO), a unit holding WEGO ``orders`` (the
    targeted / auto-fire branches), and a draws tick base (the legacy WEGO
    rewind: the device's spawn tick is ``sim.tick``). A round under physical
    cover is refused by the round table's encoder."""
    from simulation.ruleset import ContinuousRealtime
    rs = getattr(sim, "ruleset", None)
    if not isinstance(rs, ContinuousRealtime) or getattr(rs, "drives_units", False):
        raise ShotRefused(f"device shooting: the ruleset {type(rs).__name__} is not "
                          f"the continuous path")
    for u in sim.units:
        if getattr(u, "orders", None):
            raise ShotRefused(f"device shooting: unit {u.id} holds WEGO orders (the "
                              f"targeted / auto-fire branches are not ported)")
    if int(getattr(sim.draws, "tick_base", 0)) != 0:
        raise ShotRefused("device shooting: the draws carry a WEGO tick base")


def _shoot_planes(name, store, fields):
    n = store.shape[0]
    s, m = fields.get("solid"), fields.get("material")
    if s is None or m is None:
        raise ValueError(f"unit kernel {name}: fields 'solid' and 'material' needed")
    if np.dtype(s.dtype) != np.bool_ or np.dtype(m.dtype) != np.int8:
        raise ValueError(f"unit kernel {name}: solid must be bool, material int8")
    if s.ndim != 3 or tuple(s.shape) != tuple(m.shape) or int(s.shape[0]) != n:
        raise ValueError(f"unit kernel {name}: solid / material must be one "
                         f"(N, h, w), got {s.shape} / {m.shape}")
    if not (s.flags.c_contiguous and m.flags.c_contiguous):
        raise ValueError(f"unit kernel {name}: a plane is not C-contiguous")
    return int(s.data.ptr), int(m.data.ptr), int(s.shape[1]), int(s.shape[2])


def _shoot_refusals(name, store, effects, drp, tables):
    from simulation import applier
    from simulation.object_tables import ObjectTableOverflow
    store.last_dropped = drp
    if np.asarray(drp).any():
        envs = [int(e) for e in np.nonzero(drp)[0]]
        raise applier.EffectOverflow(
            f"unit kernel {name}: {int(np.asarray(drp).sum())} effect record(s) in "
            f"env(s) {envs} did not fit the effect list (its cap, or a row's "
            f"staging of {effects.row_cap})")
    for tname, lost in tables:
        lost = np.asarray(lost, dtype=np.int64)
        if lost.any():
            e = int(np.nonzero(lost)[0][0])
            raise ObjectTableOverflow(
                f"unit kernel {name}: object table {tname!r}: env {e} lost "
                f"{int(lost[e])} row(s) past its capacity")


def shooting(store, effects, rounds, noise, ticks, keys, fields) -> np.ndarray:
    """Slot 4's shooting (``combat.process_shooting`` on the continuous path:
    ``_directional_fire``, ``fire_burst``, ``start_spray_burst_directional``):
    every valid row holding the trigger that its gates let fire writes its
    own row (the magazine, ``last_fire_tick``, a spray's burst) and APPENDS,
    gathered per env in row order after what each holds, its records to
    ``effects`` (a :class:`simulation.applier.DeviceApplier`: per round the
    recoil PUSH, then THE MARCH's records), each round still flying to
    ``rounds`` and its noise to ``noise`` (:class:`simulation.object_tables.
    DeviceTable` s of ``unit_schema.ROUND_TABLE`` / ``NOISE_TABLE``).
    ``ticks`` ``(N,)`` each env's ``sim.tick``, ``keys`` ``(N, 3)`` its
    ``unit_draws.draw_key``; ``fields`` the :data:`SHOOT_FIELDS` planes as
    device arrays ``(N, h, w)``. Refuses a fault (:class:`UnitKernelFault`), a
    hitscan trigger (:class:`ShotRefused`: the beam is not ported), an
    overflow of any list (``EffectOverflow`` / ``ObjectTableOverflow``)."""
    solid, mat, h, w = _shoot_planes("shooting", store, fields)
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    k = np.ascontiguousarray(keys, dtype=np.int64).reshape(-1, 3)
    cov, flt, ref, drp, rl, nl = (np.asarray(c, dtype=np.int64) for c in
                                  store.bp.unit_shooting(
                                      store.buffer, store.tables, effects.buffer,
                                      rounds.buffer, noise.buffer, t, k, solid, mat,
                                      h, w))
    cov = _done(store, "shooting", (cov, flt, ref))
    if ref.any():
        envs = [int(e) for e in np.nonzero(ref)[0]]
        raise ShotRefused(f"unit kernel shooting: {int(ref.sum())} hitscan trigger(s) "
                          f"in env(s) {envs}: the beam (combat.fire_beam) has no device "
                          f"form (P7a.2 did not build it)")
    _shoot_refusals("shooting", store, effects, drp,
                    (("round", rl), ("noise", nl)))
    return cov


def rounds_march(store, effects, rounds, ticks, keys, fields, compact=True) -> np.ndarray:
    """Slot 2''s phase A (``Simulation._march_rounds``): per env every round
    of ``rounds`` marched in table order against the rows and planes as they
    stand, its records APPENDED to ``effects``' list (a serial producer),
    a finished round's live column 0 and, with ``compact``, dropped (the
    stable compaction: the survivors in order). The landing is
    :func:`resolve` on the same list. Returns the coverage (always 0: the
    march writes no row). Refuses a fault, an overflow."""
    solid, mat, h, w = _shoot_planes("rounds", store, fields)
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    k = np.ascontiguousarray(keys, dtype=np.int64).reshape(-1, 3)
    cov, flt, ref, drp, rl = (np.asarray(c, dtype=np.int64) for c in
                              store.bp.unit_rounds(
                                  store.buffer, store.tables, effects.buffer,
                                  rounds.buffer, t, k, solid, mat, h, w,
                                  bool(compact)))
    cov = _done(store, "rounds", (cov, flt, ref))
    _shoot_refusals("rounds", store, effects, drp, (("round", rl),))
    return cov


def rounds_of(sim) -> np.ndarray:
    """The round table's ONE record function over ``sim.bullets`` (``(R, W)``
    int64), behind :func:`shooting_door`."""
    from simulation import object_tables, unit_schema
    shooting_door(sim)
    recs = object_tables.records(unit_schema.ROUND_TABLE, sim.bullets)
    return np.array(recs, dtype=np.int64).reshape(
        len(recs), object_tables.table_layout(unit_schema.ROUND_TABLE).width)


def rounds_to_host(sim, records) -> list:
    """The round table's rows back as ``BulletInFlight`` s (the schema's
    decode and constructor), each round's ``shooter`` bound by its id to the
    unit of ``sim.units`` (a converted shooter is the same object)."""
    from simulation import object_tables, unit_schema
    recs = np.asarray(records, dtype=np.int64).reshape(
        -1, object_tables.table_layout(unit_schema.ROUND_TABLE).width)
    out = object_tables.unpack(unit_schema.ROUND_TABLE, recs, recs.shape[0])
    by_id = {u.id: u for u in sim.units}
    for b in out:
        b.shooter = by_id.get(b.shooter_id)
    return out


def blast_log_of(sim) -> np.ndarray:
    """#29 P7c.1: the blast log's ONE record function over
    ``sim.gmap.blast_log``."""
    from simulation import object_tables, unit_schema
    recs = object_tables.records(unit_schema.BLAST_LOG_TABLE, sim.gmap.blast_log)
    return np.array(recs, dtype=np.int64).reshape(
        len(recs), object_tables.table_layout(unit_schema.BLAST_LOG_TABLE).width)


def noise_of(sim) -> np.ndarray:
    """The noise list's ONE record function over ``sim.gmap.noise_log``."""
    from simulation import object_tables, unit_schema
    recs = object_tables.records(unit_schema.NOISE_TABLE, sim.gmap.noise_log)
    return np.array(recs, dtype=np.int64).reshape(
        len(recs), object_tables.table_layout(unit_schema.NOISE_TABLE).width)


__all__ += ["SHOOT_FIELDS", "ShotRefused", "shooting_door", "shooting",
            "rounds_march", "rounds_of", "rounds_to_host", "noise_of",
            "blast_log_of", "detonate", "payload_refusals"]


def round_row_cap() -> int:
    """The round table's per-row staging for slot 4 (a shooter appends at
    most its burst's rounds): the largest ``shots_per_trigger`` of the
    weapon table (at least one)."""
    from simulation.weapons import get_tables
    return max([1] + [int(w.shots_per_trigger)
                      for w in get_tables().weapons.by_name.values()])


__all__ += ["round_row_cap"]


# ---------------------------------------------------------------------------
# #29 P7b: the grenades and the planted charges on their object tables
# (``simulation.object_tables.DeviceTable`` over ``unit_schema.GRENADE_TABLE``
# / ``CHARGE_TABLE``); what they detonate is a PAYLOAD record in the effect
# list (``simulation.applier.DeviceApplier``), applied by nobody until P7c.1
# ---------------------------------------------------------------------------
class ObjectSlotRefused(RuntimeError):
    """#29 P7b: what the grenade / charge slots' device door refuses by name:
    the WEGO forms (a grenade of the order path, whose fuse is float
    ``fuse_seconds``; ``charges.detonate_owned``, the Detonate order of a
    ruleset that ``drives_units``)."""


def refuse_wego(sims) -> None:
    """THE DEVICE DOOR of the grenade and charge slots (design 4.4c scope:
    the continuous path only): refuses by name a sim whose ruleset
    ``drives_units`` (OnePhaseWEGO: its timeline drives ``charges.tick`` and
    ``charges.detonate_owned``, the Detonate order, which has no device
    form) and a grenade of the WEGO order path (no integer fuse)."""
    for e, sim in enumerate(sims):
        rules = getattr(sim, "ruleset", None)
        if getattr(rules, "drives_units", False):
            raise ObjectSlotRefused(
                f"object slots: env {e}'s ruleset {type(rules).__name__} drives "
                f"the units (charges.detonate_owned, the WEGO Detonate order, "
                f"and its timeline's charges.tick have no device form)")
        for p in getattr(sim, "projectiles", ()):
            if p.fuse_ticks is None:
                raise ObjectSlotRefused(
                    f"object slots: env {e} holds a grenade of the WEGO order "
                    f"path (fuse_seconds {p.fuse_seconds!r}, no integer fuse)")


def _object_done(name, counts, tables=()):
    ch, flt, ref, drp = (np.asarray(c, dtype=np.int64) for c in counts)
    if flt.any():
        envs = [int(e) for e in np.nonzero(flt)[0]]
        raise UnitKernelFault(
            f"unit kernel {name}: {int(flt.sum())} row(s) in env(s) {envs} left "
            f"the range the kernel was proved for (left as they were, nothing "
            f"recorded for them)")
    if ref.any():
        envs = [int(e) for e in np.nonzero(ref)[0]]
        raise ObjectSlotRefused(
            f"unit kernel {name}: {int(ref.sum())} grenade(s) of the WEGO order "
            f"path in env(s) {envs} (a float fuse_seconds, no integer fuse): "
            f"not flown")
    if drp.any():
        from simulation import applier
        envs = [int(e) for e in np.nonzero(drp)[0]]
        raise applier.EffectOverflow(
            f"unit kernel {name}: {int(drp.sum())} PAYLOAD record(s) in env(s) "
            f"{envs} did not fit the effect list: nothing was written past its "
            f"capacity")
    for t in tables:
        t.check(name)
    return ch


def _check_tables(name, store, *tables):
    for t in tables:
        if t.shape[0] != store.shape[0]:
            raise ValueError(f"unit kernel {name}: the {t.name!r} table is for "
                             f"{t.shape[0]} envs, the store {store.shape[0]}")


def table_appends(store, requests, grenades, charges) -> np.ndarray:
    """Slot 1b's table writes (after :func:`verbs` filled ``requests``): each
    THROW request appended to ``grenades`` (the ``Projectile``
    ``Simulation._release_grenade`` appends), each ACCEPTED plant to
    ``charges`` (the ``PlantedCharge`` ``charges.place`` appends), per env in
    request-list order (= the host's: id order). Returns the rows appended per
    env; refuses a fault and a table overflow (``ObjectTableOverflow``, naming
    the table and the env)."""
    _check_tables("table_appends", store, grenades, charges)
    return _object_done("table_appends", store.bp.unit_table_appends(
        store.tables, requests.buffer, grenades.buffer, charges.buffer),
        (grenades, charges))


def grenades(store, table, dev_applier, ticks) -> np.ndarray:
    """Slot 2's grenades (``Simulation._update_grenades``): every live row
    flies (``Projectile.update_position``, wide), a row whose fuse ran out is
    marked detonated and -- its proj_type ORDER_GRENADE -- its detonation
    APPENDED to ``dev_applier``'s effect list as a PAYLOAD record
    (``execute_payload``'s call, ``applier.record_of``'s fields), in row
    order after what the list holds. ``ticks`` ``(N,)`` each env's
    ``sim.tick``. Returns the rows changed per env; refuses a fault, a WEGO
    grenade (:class:`ObjectSlotRefused`) and an effect-list overflow."""
    _check_tables("grenades", store, table)
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    return _object_done("grenades", store.bp.unit_grenades(
        store.tables, table.buffer, dev_applier.buffer, t))


def charges_tick(store, table, dev_applier, ticks) -> np.ndarray:
    """Slot 9e(d') ``charges.tick``: every live charge whose det_tick has
    come is marked dead and its detonation APPENDED to ``dev_applier``'s
    effect list as a PAYLOAD record (kind ``door_explosive``) in row order;
    then the table's stable compaction (the prune). Returns the charges fired
    per env; refuses a fault and an effect-list overflow."""
    _check_tables("charges_tick", store, table)
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    return _object_done("charges_tick", store.bp.unit_charges_tick(
        store.tables, table.buffer, dev_applier.buffer, t))


def prune_grenades(table) -> None:
    """Slot 10's grenade prune (``ContinuousRealtime.on_tick_end``): the
    detonated grenades leave the table -- its stable compaction."""
    table.compact()


__all__ += ["ObjectSlotRefused", "refuse_wego", "table_appends", "grenades",
            "charges_tick", "prune_grenades"]


# ---------------------------------------------------------------------------
# #29 P7a.3: slot 4b (the sprays), the walk's further reach
# ---------------------------------------------------------------------------
class WalkLists:
    """What the applier's walk reaches besides the planes, the rows and the
    ring (each optional): ``cells`` THE CELL BUFFER
    (``attack_lists.cell_buffer``), ``door`` THE DEVICE EDIT DOOR
    (``edit_door.DeviceEditDoor``), ``claims`` THE CLAIMS LIST
    (``attack_lists.claims_list``)."""

    def __init__(self, cells=None, door=None, claims=None, noise=None,
                 blast_log=None):
        self.cells, self.door, self.claims = cells, door, claims
        # #29 P7c.1: THE NOISE LIST and THE BLAST LOG (object tables) a
        # PAYLOAD record appends to
        self.noise, self.blast_log = noise, blast_log


class SpraySlotRefused(RuntimeError):
    """#29 P7a.3: what slot 4b's device form refuses by name: a burst of the
    WEGO ``start_spray_burst`` (no aim angle: the tile-derived bearing, out
    of the continuous scope)."""


#: The planes slot 4b reads (the PHYSICAL sight's), by GameMap field name.
SPRAY_FIELDS = (("solid", np.bool_),)


def spray_box_max(store, h: int, w: int) -> int:
    """The largest cone box any env's tables allow on an ``(h, w)`` grid:
    ``(2 r + 1)^2`` of the largest ``spray_range_tiles`` row, never more than
    the grid (the box is clipped to it) -- the launch's scratch per env."""
    from simulation import unit_tables
    c = unit_tables.layout().column("spray_range_tiles")
    tab = np.asarray(store.tables.download(), dtype=np.int64)
    r = int(tab[:, c.offset:c.offset + c.width].max()) if tab.size else 0
    return min((2 * r + 1) ** 2, int(h) * int(w))


def sprays(store, dev_applier, cells, fields: dict, box_max: int | None = None):
    """Slot 4b (``combat.process_sprays``, the continuous path): every valid
    row with a burst in progress, in row order -- a dead or interrupted burst
    cleared, a live one's cone (``combat.spray_cone_tiles`` over the PHYSICAL
    sight of ``fields["solid"]``, a DEVICE ``(N, h, w)`` bool array) appended
    to ``cells`` (``attack_lists.cell_buffer``) and ONE SPRAY_CELLS record to
    ``dev_applier``'s effect list after what it holds; then the countdown.
    Returns the rows changed per env; refuses a fault
    (:class:`UnitKernelFault`), a WEGO burst (:class:`SpraySlotRefused`) and
    an overflow of the effect list (``applier.EffectOverflow``) or the cell
    buffer (``attack_lists.ListOverflow``), naming the envs."""
    from simulation import applier
    n = store.shape[0]
    _check_planes("sprays", store, fields, dict(SPRAY_FIELDS))
    s = fields["solid"]
    if s.ndim != 3 or not s.flags.c_contiguous:
        raise ValueError("unit kernel sprays: solid must be a C-contiguous (N, h, w) array")
    if dev_applier.n_env != n or cells.n_env != n:
        raise ValueError("unit kernel sprays: the effect list / cell buffer env count "
                         "is not the store's")
    _, h, w = (int(x) for x in s.shape)
    bm = int(box_max) if box_max is not None else spray_box_max(store, h, w)
    cov, flt, ref, drp = (np.asarray(c, dtype=np.int64) for c in store.bp.unit_sprays(
        store.buffer, store.tables, dev_applier.buffer, cells.buffer, int(s.data.ptr),
        h, w, max(1, bm)))
    store.last_refused = ref
    store.last_dropped = drp
    if flt.any():
        envs = [int(e) for e in np.nonzero(flt)[0]]
        raise UnitKernelFault(
            f"unit kernel sprays: {int(flt.sum())} row(s) in env(s) {envs} left the "
            f"range the kernel was proved for (left as they were, nothing recorded)")
    if ref.any():
        envs = [int(e) for e in np.nonzero(ref)[0]]
        raise SpraySlotRefused(
            f"unit kernel sprays: {int(ref.sum())} burst(s) of the WEGO "
            f"start_spray_burst in env(s) {envs} (no aim angle): not deposited")
    if drp.any():
        _r, _c, fdrop = dev_applier.download()
        if np.asarray(fdrop).any():
            envs = [int(e) for e in np.nonzero(np.asarray(fdrop))[0]]
            raise applier.EffectOverflow(
                f"unit kernel sprays: SPRAY_CELLS record(s) in env(s) {envs} did not "
                f"fit the effect list (cap {dev_applier.cap})")
        cells.check("sprays")
    return cov


def spray_cone_probe(bp, solid, args, box_max: int, max_out: int):
    """TEST-ONLY: the cone (``cpp/src/cuda_unit_spray.cuh``'s decide / emit) on
    synthetic cones over the DEVICE ``(N, h, w)`` bool plane ``solid``;
    ``args`` ``(N, R, 8)`` (ay, ax, angle_q, range, half_q, ex_x, ex_y, fp) ->
    ``(cells (N, R, max_out, 3), meta (N, R, 2): count, bad)``."""
    if np.dtype(solid.dtype) != np.dtype(np.bool_) or solid.ndim != 3 \
            or not solid.flags.c_contiguous:
        raise ValueError("spray_cone_probe: solid must be a C-contiguous (N, h, w) bool "
                         "device array")
    n, h, w = (int(x) for x in solid.shape)
    out, meta = bp.unit_spray_cone_probe(int(solid.data.ptr), n, h, w,
                                         np.ascontiguousarray(args, dtype=np.int64),
                                         int(box_max), int(max_out))
    return np.asarray(out), np.asarray(meta)


__all__ += ["WalkLists", "SpraySlotRefused", "SPRAY_FIELDS", "spray_box_max",
            "sprays", "spray_cone_probe"]
# #29 P6b.1: THE DOORS -- the door table (unit_schema.DOOR_TABLE, the first
# entity table) and its tile side table, slot 1b's USE as a table write, slot
# 9e(d)'s sweep as the applier's DOORS stage, and the seam's TEST-ONLY probe
# (seal / unseal / can_seal, the gas-N primitives, _aperture_blocked)
# ---------------------------------------------------------------------------
class DoorTableMismatch(RuntimeError):
    """Door rows that are not the sim's doors (another count, ordinal or span)."""


def _door_rows(recs, width) -> np.ndarray:
    return np.asarray(recs, dtype=np.int64).reshape(-1, width)


def doors_of(sim) -> np.ndarray:
    """THE record function of the door table: ``sim._doors`` (ordinal order)
    as ``(D, W)`` int64 rows (``object_tables.records``)."""
    from simulation import object_tables, unit_schema
    t = unit_schema.DOOR_TABLE
    return _door_rows(object_tables.records(t, getattr(sim, "_doors", ()) or ()),
                      object_tables.table_layout(t).width)


def door_tiles_of(sim) -> np.ndarray:
    """THE record function of the door tile side table: every door's span
    tiles with their hp, in door order, each row-major, as ``(S, W)`` int64."""
    from simulation import object_tables, unit_schema
    return _door_rows(object_tables.side_records(
        unit_schema.DOOR_TABLE, getattr(sim, "_doors", ()) or (), "tiles"),
        object_tables.table_layout(unit_schema.DOOR_TILE_TABLE).width)


def door_objects(door_recs, tile_recs) -> list:
    """The door table's rows (and its side table's) as ``unit_schema.DoorRow``
    objects (``object_tables.unpack`` over the schema)."""
    from simulation import object_tables, unit_schema
    d = _door_rows(door_recs, object_tables.table_layout(unit_schema.DOOR_TABLE).width)
    s = _door_rows(tile_recs, object_tables.table_layout(unit_schema.DOOR_TILE_TABLE).width)
    return object_tables.unpack(unit_schema.DOOR_TABLE, d, d.shape[0],
                                sides={"tiles": (s, s.shape[0])})


def doors_to_host(sim, door_recs, tile_recs) -> None:
    """The device's door rows LANDED on the sim's own runtimes in place (the
    chained form): each ``sim._doors[k]`` takes row k's ``alive``,
    ``state``, ``want_open`` and ``hp``; a row whose ordinal or span is not
    that door's -- or another door count -- is refused by name
    (:class:`DoorTableMismatch`)."""
    rows = door_objects(door_recs, tile_recs)
    doors = list(getattr(sim, "_doors", ()) or ())
    if len(rows) != len(doors):
        raise DoorTableMismatch(f"doors: {len(rows)} device rows for {len(doors)} doors")
    for k, (d, r) in enumerate(zip(doors, rows)):
        if int(d.ordinal) != r.ordinal or [tuple(t) for t in d.span] != r.span:
            raise DoorTableMismatch(f"doors: row {k} (ordinal {r.ordinal}) is not door "
                                    f"{d.id!r} (ordinal {d.ordinal})")
    for d, r in zip(doors, rows):
        d.alive, d.state, d.want_open = r.alive, int(r.state), r.want_open
        d.hp = [int(v) for v in r.hp]


def door_entity_rows(door_rec, tile_recs) -> tuple:
    """A door row's part of ENTITY_SECT_V1 (``entities/serialize``): the free
    ``alive`` row, then ``door.runtime_digest_rows`` -- ``state``,
    ``want_open``, ``hp_<i>`` per span tile -- as ``(name, int)`` pairs, read
    from the table (``tile_recs`` the env's whole tile side table)."""
    from simulation import object_tables, unit_schema
    tl = object_tables.table_layout(unit_schema.DOOR_TILE_TABLE)
    v = object_tables.row_values(unit_schema.DOOR_TABLE, door_rec)
    tiles = _door_rows(tile_recs, tl.width)
    a, n = int(v["tile_start"]), int(v["tile_count"])
    hp_col = tl.column("hp").offset
    rows = [("alive", int(v["alive"])), ("state", int(v["state"])),
            ("want_open", int(v["want_open"]))]
    rows.extend((f"hp_{i}", int(tiles[a + i, hp_col])) for i in range(n))
    return tuple(rows)


def door_tables(n_env: int, bp=None):
    """The device door table and its tile side table for ``n_env`` envs
    (``object_tables.DeviceTable`` each)."""
    from simulation import object_tables, unit_schema
    return (object_tables.DeviceTable(unit_schema.DOOR_TABLE, n_env, bp=bp),
            object_tables.DeviceTable(unit_schema.DOOR_TILE_TABLE, n_env, bp=bp))


def door_uses(store, requests, doors) -> np.ndarray:
    """Slot 1b's USE as a TABLE WRITE (after :func:`verbs` filled
    ``requests``): each USE record flips ``want_open`` on the door table's row
    ``door`` (``Simulation._direct_use`` / ``unit_requests.apply_to_host``),
    per env in request-list order. Returns the flips per env; refuses a
    record naming a row outside the env's table (:class:`UnitKernelFault`)."""
    _check_tables("door_uses", store, doors)
    ch, flt, _ref, _drp = (np.asarray(c, dtype=np.int64) for c in store.bp.unit_door_uses(
        store.tables, requests.buffer, doors.buffer))
    if flt.any():
        envs = [int(e) for e in np.nonzero(flt)[0]]
        raise UnitKernelFault(
            f"unit kernel door_uses: {int(flt.sum())} USE record(s) in env(s) {envs} "
            f"name a door outside the env's door table (nothing written for them)")
    return ch


def sweep_doors(store, dev_applier, fields: dict, doors, tiles) -> tuple:
    """Slot 9e(d)'s door sweep (``door_system.sweep_doors``) as the applier's
    DOORS stage: every env's door table walked in row (= ordinal) order on its
    planes (``fields`` as :func:`walk_effects`' -- every ``applier.PLANES``
    name, written IN PLACE) and its unit rows (the occupancy), the door and
    tile tables written in place. Returns ``(coverage, books)``: coverage the
    door transitions per env (open, close, death: the ``applied`` book).
    Refuses a fault by name (:class:`simulation.applier.ApplierFault`: the
    reference would have raised, or an operand left the proved range)."""
    from simulation import applier
    n = store.shape[0]
    name = "applier[DOORS]"
    if dev_applier.n_env != n:
        raise ValueError(f"unit kernel {name}: the applier is for {dev_applier.n_env} "
                         f"envs, the store {n}")
    _check_tables(name, store, doors, tiles)
    ptrs, n_gases, h, w = _plane_ptrs(name, n, fields)
    b, _cov, _rdrop = store.bp.applier_run(
        dev_applier.buffer, store.buffer, store.tables, ptrs, n_gases, h, w,
        int(applier.STAGE_DOORS), None, door_table=doors.buffer,
        door_tiles=tiles.buffer)
    books = applier.books_dict(b)
    store.last_books = books
    if books["faults"].any():
        envs = [int(e) for e in np.nonzero(books["faults"])[0]]
        raise applier.ApplierFault(
            f"unit kernel {name}: {int(books['faults'].sum())} door(s) in env(s) {envs} "
            f"faulted (a seal / unseal the reference would raise on, a span "
            f"outside its table or the grid, or an operand past the proved range)")
    return books["applied"].copy(), books


#: TEST-ONLY seam probe op codes (cpp/src/cuda_topology_probe.cu).
SEAM_OPS = {"seal": 1, "unseal": 2, "can_seal": 3, "inject": 4, "extract": 5,
            "transfer": 6, "aperture": 7,
            # #29 P6b.2: inject_gas_n_vec and the kit's floor division / product
            "inject_vec": 8, "floordiv": 9, "muldiv": 10}
#: An op row's width (the op code, then its arguments; shorter rows are padded).
SEAM_OP_W = 16


def seam_probe(store, fields: dict, per_env_ops, per_env_tiles) -> tuple:
    """TEST-ONLY: per env its ops (rows of 8 int64: the op code of
    :data:`SEAM_OPS` then its arguments) walked serially on the planes in
    place (``cpp/src/cuda_topology_probe.cu``; rows up to :data:`SEAM_OP_W`
    wide, padded with zeros); a span is ``(start, count)``
    rows of the env's tile list (``per_env_tiles``, ``(T, 2)`` each).
    Returns ``(results, books)``: per env a ``(K_e, 9)`` array (refusal code,
    value, the per-slice withdrawal) and the books ``{name: (N,)}``,
    unrefused."""
    from simulation import applier
    n = store.shape[0]
    ptrs, n_gases, h, w = _plane_ptrs("seam_probe", n, fields)
    ops = []
    for o in per_env_ops:
        a = np.asarray(o, dtype=np.int64)
        a = a.reshape(-1, a.shape[-1]) if a.size else np.zeros((0, SEAM_OP_W), np.int64)
        if a.shape[1] > SEAM_OP_W:
            raise ValueError(f"seam_probe: an op row is at most {SEAM_OP_W} wide")
        ops.append(np.pad(a, ((0, 0), (0, SEAM_OP_W - a.shape[1]))))
    tl = [np.asarray(t, dtype=np.int64).reshape(-1, 2) for t in per_env_tiles]
    K = max([1] + [o.shape[0] for o in ops])
    T = max([1] + [t.shape[0] for t in tl])
    O = np.zeros((n, K, SEAM_OP_W), dtype=np.int64)
    TT = np.zeros((n, T, 2), dtype=np.int64)
    cnt = np.zeros(n, dtype=np.int64)
    for e in range(n):
        O[e, :ops[e].shape[0]] = ops[e]
        TT[e, :tl[e].shape[0]] = tl[e]
        cnt[e] = ops[e].shape[0]
    res, books = store.bp.topology_probe_seam(store.tables, ptrs, n_gases, h, w, O, cnt, TT)
    res = np.asarray(res)
    return [res[e, :cnt[e]].copy() for e in range(n)], applier.books_dict(books)


__all__ += ["DoorTableMismatch", "doors_of", "door_tiles_of", "door_objects",
            "doors_to_host", "door_entity_rows", "door_tables", "door_uses",
            "sweep_doors", "SEAM_OPS", "seam_probe", "SEAM_OP_W"]


# ---------------------------------------------------------------------------
# #29 P6b.2: THE VENTS -- the duct table and the vent table (unit_schema.
# DUCT_TABLE / VENT_TABLE, the second and third entity tables) and slot
# 9e(d)'s circulation as the applier's VENTS stage
# ---------------------------------------------------------------------------
class VentTableMismatch(RuntimeError):
    """Duct / vent rows that are not the sim's (another count, ordinal,
    filter, duct, role, aperture or load constant)."""


def ducts_of(sim) -> np.ndarray:
    """THE record function of the duct table: ``sim._ducts`` (ordinal order)
    as ``(D, W)`` int64 rows (``object_tables.records``)."""
    from simulation import object_tables, unit_schema
    t = unit_schema.DUCT_TABLE
    return _door_rows(object_tables.records(t, getattr(sim, "_ducts", ()) or ()),
                      object_tables.table_layout(t).width)


def vents_of(sim) -> np.ndarray:
    """THE record function of the vent table: ``sim._vents`` (ordinal order)
    as ``(V, W)`` int64 rows; a vent's duct is its duct's entity ordinal."""
    from simulation import object_tables, unit_schema
    t = unit_schema.VENT_TABLE
    return _door_rows(object_tables.records(t, getattr(sim, "_vents", ()) or ()),
                      object_tables.table_layout(t).width)


def duct_objects(recs) -> list:
    """The duct table's rows as ``unit_schema.DuctRow`` objects (``unpack``)."""
    from simulation import object_tables, unit_schema
    d = _door_rows(recs, object_tables.table_layout(unit_schema.DUCT_TABLE).width)
    return object_tables.unpack(unit_schema.DUCT_TABLE, d, d.shape[0])


def vent_objects(recs) -> list:
    """The vent table's rows as ``unit_schema.VentRow`` objects (``unpack``)."""
    from simulation import object_tables, unit_schema
    v = _door_rows(recs, object_tables.table_layout(unit_schema.VENT_TABLE).width)
    return object_tables.unpack(unit_schema.VENT_TABLE, v, v.shape[0])


_DUCT_STATE = ("o2_raw", "n2_raw", "e_plenum", "e_wipe", "rail_lo_hits", "rail_hi_hits")
_VENT_CONST = ("aperture_y", "aperture_x", "rate_raw_per_s", "tps")


def ducts_to_host(sim, recs) -> None:
    """The device's duct rows LANDED on the sim's own ``DuctRuntime``s in
    place (the chained form): the ledger and ``alive``; a row whose ordinal or
    filter is not that duct's, or another duct count, is refused by name
    (:class:`VentTableMismatch`)."""
    rows = duct_objects(recs)
    ducts = list(getattr(sim, "_ducts", ()) or ())
    if len(rows) != len(ducts):
        raise VentTableMismatch(f"ducts: {len(rows)} device rows for {len(ducts)} ducts")
    for k, (d, r) in enumerate(zip(ducts, rows)):
        if int(d.ordinal) != r.ordinal or d.inst.fields["filter"] != r.filter:
            raise VentTableMismatch(f"ducts: row {k} (ordinal {r.ordinal}) is not duct "
                                    f"{d.id!r} (ordinal {d.ordinal})")
    for d, r in zip(ducts, rows):
        for a in _DUCT_STATE:
            setattr(d, a, int(getattr(r, a)))
        d.trace_raw = [int(x) for x in r.trace_raw]
        d.sink = [int(x) for x in r.sink]
        d.alive = bool(r.alive)


def vents_to_host(sim, recs) -> None:
    """The device's vent rows LANDED on the sim's own ``VentRuntime``s in
    place: ``accum``, ``guard_skips`` and ``alive``; a row whose ordinal,
    duct, role or load constants are not that vent's is refused by name."""
    from simulation import unit_schema
    rows = vent_objects(recs)
    vents = list(getattr(sim, "_vents", ()) or ())
    if len(rows) != len(vents):
        raise VentTableMismatch(f"vents: {len(rows)} device rows for {len(vents)} vents")
    for k, (v, r) in enumerate(zip(vents, rows)):
        mine = unit_schema._vent_duct_ordinal(v)
        role = "return" if v.role == "return" else "supply"
        if int(v.ordinal) != r.ordinal or mine != r.duct_ordinal or role != r.role or \
                any(int(getattr(v, a)) != int(getattr(r, a)) for a in _VENT_CONST):
            raise VentTableMismatch(f"vents: row {k} (ordinal {r.ordinal}) is not vent "
                                    f"{v.id!r} (ordinal {v.ordinal})")
    for v, r in zip(vents, rows):
        v.accum, v.guard_skips, v.alive = int(r.accum), int(r.guard_skips), bool(r.alive)


def duct_entity_rows(rec) -> tuple:
    """A duct row's part of ENTITY_SECT_V1 (``entities/serialize``): the free
    ``alive`` row, then ``duct.runtime_digest_rows`` -- the bulk pair, the
    energy, the wipe, the rail hits, ``trace_<i>``, ``sink_<i>`` -- read from
    the table (P6b.4's export)."""
    from simulation import object_tables, unit_schema
    v = object_tables.row_values(unit_schema.DUCT_TABLE, rec)
    rows = [("alive", int(v["alive"]))]
    rows += [(a, int(v[a])) for a in _DUCT_STATE]
    rows += [(f"trace_{i}", int(x)) for i, x in enumerate(v["trace_raw"])]
    rows += [(f"sink_{i}", int(x)) for i, x in enumerate(v["sink"])]
    return tuple(rows)


def vent_entity_rows(rec) -> tuple:
    """A vent row's part of ENTITY_SECT_V1: the free ``alive`` row, then
    ``vent.runtime_digest_rows`` (``accum``), read from the table."""
    from simulation import object_tables, unit_schema
    v = object_tables.row_values(unit_schema.VENT_TABLE, rec)
    return (("alive", int(v["alive"])), ("accum", int(v["accum"])))


def vent_tables(n_env: int, bp=None):
    """The device duct table and vent table for ``n_env`` envs
    (``object_tables.DeviceTable`` each)."""
    from simulation import object_tables, unit_schema
    return (object_tables.DeviceTable(unit_schema.DUCT_TABLE, n_env, bp=bp),
            object_tables.DeviceTable(unit_schema.VENT_TABLE, n_env, bp=bp))


#: The ledgers of a duct row the range check reads (cpp/src/cuda_unit_vents.cuh).
VENT_LEDGERS = ("o2_raw", "n2_raw", "trace_raw", "sink", "e_plenum", "e_wipe",
                "rail_lo_hits", "rail_hi_hits")
#: The proved range's bounds (cpp/src/cuda_unit_vents.cuh's L_*, TILE_*).
VENT_L_BULK, VENT_L_TRACE, VENT_L_SINK = 1 << 61, 1 << 54, 1 << 62
VENT_L_E, VENT_L_WIPE, VENT_L_COUNT = 1 << 61, 1 << 62, 1 << 62
VENT_TILE_BULK, VENT_TILE_TRACE, VENT_TILE_T_SHIFT = 1 << 32, 1 << 31, 29


def vent_ledger_range(duct_rec, vent_recs) -> list:
    """The ledgers of ONE duct row outside the range its sweep was proved for
    (``cpp/src/cuda_unit_vents.cuh``: the bounds depend on the duct's own
    returns' quanta and its return / supply counts), as ``(name, value, lo,
    hi)``; empty when the duct may run. ``vent_recs`` the env's whole vent
    table. The host twin of the device's head check (it names the refusal)."""
    from simulation import object_tables, unit_schema
    d = object_tables.row_values(unit_schema.DUCT_TABLE, duct_rec)
    b_n, r, s, out = 0, 0, 0, []
    for vr in vent_recs:
        v = object_tables.row_values(unit_schema.VENT_TABLE, vr)
        if v["duct_ordinal"] != d["ordinal"]:
            continue
        if not 0 <= v["guard_skips"] <= VENT_L_COUNT - 1:
            out.append(("guard_skips", v["guard_skips"], 0, VENT_L_COUNT - 1))
        if v["role"] == 1:
            q = (v["accum"] + v["rate_raw_per_s"]) // max(1, v["tps"])
            b_n += min(q, VENT_TILE_BULK)
            r += 1
        else:
            s += 1
    if b_n > (VENT_L_E >> VENT_TILE_T_SHIFT):
        out.append(("returns' quanta (B_N)", b_n, 0, VENT_L_E >> VENT_TILE_T_SHIFT))
        return out
    b_e = b_n << VENT_TILE_T_SHIFT
    lim = {"o2_raw": (0, VENT_L_BULK - b_n), "n2_raw": (0, VENT_L_BULK - b_n),
           "trace_raw": (0, VENT_L_TRACE - r * VENT_TILE_TRACE),
           "sink": (0, VENT_L_SINK - r * VENT_TILE_TRACE),
           "e_plenum": (-(VENT_L_E - b_e), VENT_L_E - b_e),
           "e_wipe": (-VENT_L_WIPE, VENT_L_WIPE),
           "rail_lo_hits": (0, VENT_L_COUNT - s), "rail_hi_hits": (0, VENT_L_COUNT - s)}
    for name in VENT_LEDGERS:
        lo, hi = lim[name]
        vals = d[name] if isinstance(d[name], tuple) else (d[name],)
        for x in vals:
            if not lo <= x <= hi:
                out.append((name, int(x), lo, hi))
    return out


def sweep_vents(store, dev_applier, fields: dict, ducts, vents) -> tuple:
    """Slot 9e(d)'s vent circulation (``vent_system.sweep_vents``) as the
    applier's VENTS stage: every env's duct table walked in row (= ordinal)
    order, within a duct its returns then its supplies in vent-row order, on
    the env's planes (``fields`` as :func:`walk_effects`' -- every
    ``applier.PLANES`` name, written IN PLACE), the duct and vent tables
    written in place. Returns ``(coverage, books)``: coverage the extractions
    that took gas plus the deposits that landed per env (the ``applied``
    book). Refuses by name a duct whose ledgers left the proved range
    (:class:`simulation.applier.LedgerRangeRefused`, naming the env, the duct
    and the ledger: nothing of that duct was written) and a fault
    (:class:`simulation.applier.ApplierFault`)."""
    from simulation import applier
    n = store.shape[0]
    name = "applier[VENTS]"
    if dev_applier.n_env != n:
        raise ValueError(f"unit kernel {name}: the applier is for {dev_applier.n_env} "
                         f"envs, the store {n}")
    _check_tables(name, store, ducts, vents)
    ptrs, n_gases, h, w = _plane_ptrs(name, n, fields)
    b, _cov, _rdrop = store.bp.applier_run(
        dev_applier.buffer, store.buffer, store.tables, ptrs, n_gases, h, w,
        int(applier.STAGE_VENTS), None, duct_table=ducts.buffer, vent_table=vents.buffer)
    books = applier.books_dict(b)
    store.last_books = books
    if books["ledger_refused"].any():
        drows, dcount, _ = ducts.download()
        vrows, vcount, _ = vents.download()
        named = []
        for e in np.nonzero(books["ledger_refused"])[0]:
            for k in range(int(dcount[e])):
                bad = vent_ledger_range(drows[e, k], vrows[e, :int(vcount[e])])
                if bad:
                    named.append(f"env {int(e)} duct row {k} (ordinal "
                                 f"{int(drows[e, k, 0])}): " + ", ".join(
                                     f"{nm} = {x} outside [{lo}, {hi}]"
                                     for nm, x, lo, hi in bad))
        raise applier.LedgerRangeRefused(
            f"unit kernel {name}: {int(books['ledger_refused'].sum())} duct(s) refused at "
            f"the head of their sweep, a ledger outside the proved range (nothing of "
            f"them written): " + "; ".join(named))
    if books["faults"].any():
        envs = [int(e) for e in np.nonzero(books["faults"])[0]]
        raise applier.ApplierFault(
            f"unit kernel {name}: {int(books['faults'].sum())} fault(s) in env(s) {envs} "
            f"(a vent row outside its constants, a filter outside the table, or a seam "
            f"call the reference would raise on / past its proved range)")
    return books["applied"].copy(), books


__all__ += ["VentTableMismatch", "ducts_of", "vents_of", "duct_objects",
            "vent_objects", "ducts_to_host", "vents_to_host", "duct_entity_rows",
            "vent_entity_rows", "vent_tables", "VENT_LEDGERS", "vent_ledger_range",
            "sweep_vents"]
