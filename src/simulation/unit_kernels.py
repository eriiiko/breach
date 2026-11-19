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
#: #215: and the body temperature's air read (atmosphere, temperature, solid).
HEAT_FIELDS = ("heat", "rad_flux_sweep", "atmosphere", "temperature", "solid")
#: The planes slot 9c3 reads (arc #63 P5a: + the swarm's bite-dose plane,
#: the swarm_bite_dose row's input; P5b: + its count plane, the corpse row's).
GAS_FIELDS = ("gas", "swarm_bite_dose", "swarm_count")


def heat_damage(store, ring, fields: dict, claims) -> np.ndarray:
    """Slot 9c: the heat row on every valid living row (#215: the body
    temperature's law, the band on the body). ``fields``: ``heat`` ``(N, h,
    w)`` int32, ``rad_flux_sweep`` ``(N, h, w)`` int64, ``atmosphere`` and
    ``temperature`` ``(N, h, w)`` int32 and ``solid`` ``(N, h, w)`` bool
    DEVICE arrays (``ResidentBatch.buffers`` as they are); ``claims`` ``(N,
    R, 4)`` (:func:`pad_claims`)."""
    _check_ring("heat_damage", store, ring)
    _check_planes("heat_damage", store, fields,
                  {"heat": np.int32, "rad_flux_sweep": np.int64,
                   "atmosphere": np.int32, "temperature": np.int32,
                   "solid": np.bool_})
    _, h, w = (int(x) for x in fields["heat"].shape)
    for k in ("rad_flux_sweep", "atmosphere", "temperature", "solid"):
        if tuple(int(x) for x in fields[k].shape) != (store.shape[0], h, w):
            raise ValueError(f"unit kernel heat_damage: {k}'s shape is not "
                             f"heat's")
    c = np.ascontiguousarray(claims, dtype=np.int64)
    return _done_events(store, "heat_damage", store.bp.unit_heat_damage(
        store.buffer, store.tables, ring.buffer, int(fields["heat"].data.ptr),
        int(fields["rad_flux_sweep"].data.ptr),
        int(fields["atmosphere"].data.ptr), int(fields["temperature"].data.ptr),
        int(fields["solid"].data.ptr), h, w, c))


def gas_rows(store, ring, fields: dict, claims) -> np.ndarray:
    """Slot 9c3: the teargas row, the poison row's dose, the larvae's bite
    dose (arc #63 P5a), the corpses' infestation (P5b), the poison law and
    (#215) the stress tick, fused per row (each unit's steps read only the
    planes and its own row). ``fields``: ``gas`` ``(N, G, h, w)``,
    ``swarm_bite_dose`` and ``swarm_count`` ``(N, h, w)`` int32 DEVICE
    arrays;
    ``claims`` as for :func:`heat_damage`. ``store.last_refused`` holds the
    doses clipped at the body's store (poison.SATURATION's hits)."""
    _check_ring("gas_rows", store, ring)
    _check_planes("gas_rows", store, fields,
                  {"gas": np.int32, "swarm_bite_dose": np.int32,
                   "swarm_count": np.int32})
    _, n_gases, h, w = (int(x) for x in fields["gas"].shape)
    for k in ("swarm_bite_dose", "swarm_count"):
        if tuple(int(x) for x in fields[k].shape) != (store.shape[0], h, w):
            raise ValueError(f"unit kernel gas_rows: {k}'s shape is not the "
                             f"gas planes'")
    c = np.ascontiguousarray(claims, dtype=np.int64)
    return _done_events(store, "gas_rows", store.bp.unit_gas_rows(
        store.buffer, store.tables, ring.buffer, int(fields["gas"].data.ptr),
        n_gases, h, w, c, int(fields["swarm_bite_dose"].data.ptr),
        int(fields["swarm_count"].data.ptr)))


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


def _flight_plane(name, store, fields):
    """#215: ``fields["flight"]``, the DEVICE ``(N, h, w)`` bool flight plane
    (``combat.flight_plane``), checked; -> (pointer, h, w)."""
    _check_planes(name, store, fields, dict(GRENADE_FIELDS))
    s = fields["flight"]
    if s.ndim != 3 or not s.flags.c_contiguous:
        raise ValueError(f"unit kernel {name}: the flight plane must be a "
                         f"C-contiguous (N, h, w) array")
    return int(s.data.ptr), int(s.shape[1]), int(s.shape[2])


def verbs(store, requests, ticks, door_spans, charges_live, fields) -> np.ndarray:
    """Slot 1b (``Simulation._consume_direct_intents``, the continuous path):
    a dead or turned row drops its trigger, wind-up and plant; every living
    marine consumes its own latches (throw, plant, use, reload, shove) and
    advances its wind-up and plant. ``requests`` a begun
    :class:`simulation.unit_requests.RequestList` (the grenades released, the
    charges laid -- accepted or refused at the capacity --, the doors flipped,
    in id order); ``ticks`` ``(N,)`` each env's ``sim.tick``;
    ``door_spans`` ``(N, R, 3)`` (``unit_requests.pad_door_spans``);
    ``charges_live`` ``(N,)`` the live planted charges; ``fields["flight"]``
    (#215) the flight plane a throw's release point is walked over (see
    :func:`grenades`). ``store.last_refused``
    holds the plants refused at ``CHARGE_CAPACITY``. REFUSES a run whose
    requests the list could not hold (:class:`simulation.unit_requests.
    UnitRequestOverflow`, naming the kernel and the env) as it refuses a
    fault."""
    from simulation.unit_requests import UnitRequestOverflow
    _check_requests("verbs", store, requests)
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    s = np.ascontiguousarray(door_spans, dtype=np.int64)
    lv = np.ascontiguousarray(charges_live, dtype=np.int64).reshape(-1)
    ptr, fh, fw = _flight_plane("verbs", store, fields)
    cov, flt, ref, drp = (np.asarray(c, dtype=np.int64) for c in store.bp.unit_verbs(
        store.buffer, store.tables, requests.buffer, t, s, lv, ptr, fh, fw))
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
                 lists=None, refused_ok: bool = False, source_out=None,
                 timed_table=None, heat_table=None):
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
                    ("noise", wl.noise), ("blast_log", wl.blast_log),
                    ("sources", wl.sources), ("source_tiles", wl.source_tiles)):
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
        None if wl.blast_log is None else wl.blast_log.buffer,
        source_table=None if wl.sources is None else wl.sources.buffer,
        source_tiles=None if wl.source_tiles is None else wl.source_tiles.buffer,
        source_out=_source_out_ptr(name, n, h, w, source_out),
        timed_table=None if timed_table is None else timed_table.buffer,   # #29 P6b.4
        heat_table=None if heat_table is None else heat_table.buffer)
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
    if books["bulk_refused"].any():           # #29 P7c.2
        envs = [int(e) for e in np.nonzero(books["bulk_refused"])[0]]
        raise applier.BulkGasOverflow(
            f"unit kernel {name}: {int(books['bulk_refused'].sum())} bulk-gas "
            f"deposit(s) in env(s) {envs} refused: a tile's bulk slice would pass "
            f"int32 (inject_bulk_gas raises OverflowError; nothing of the deposit "
            f"written)")
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
    for t in (wl.noise, wl.blast_log, wl.sources, wl.source_tiles):   # #29 P7c.1 / .2
        if t is not None:
            t.check(name)
    return cov, books


def _source_out_ptr(name, n, h, w, out) -> int:
    """#29 P7c.2: the SOURCES stage's hand-out plane, a device ``(N, h, w)``
    int32 array (or None: 0)."""
    if out is None:
        return 0
    if (tuple(int(x) for x in out.shape) != (n, h, w) or np.dtype(out.dtype) != np.int32
            or not out.flags.c_contiguous):
        raise ValueError(f"unit kernel {name}: the source hand-out must be a C-contiguous "
                         f"({n}, {h}, {w}) int32 device array, not {out.shape} {out.dtype}")
    return int(out.data.ptr)


def blast_probe(store, dev_applier, fields: dict, queries, max_tiles: int) -> np.ndarray:
    """#29 P7c.2, TEST-ONLY: the device walk and air read
    (``cpp/src/cuda_unit_blast.cuh``) per query -- ``queries`` (N, K, 7)
    int64 ``(fy, fx, r2, has_approach, ay, ax, mode)`` (mode 0 the reach walk
    over the open mask, 1 ``blast_air.disc_tiles`` of radius r2) -- on the
    env's planes. Returns (N, K, 3 + 2 M): the count, the heat factor,
    all_one, the cells (row-major index), their smoke factors."""
    ptrs, n_gases, h, w = _plane_ptrs("blast_probe", store.shape[0], fields)
    q = np.ascontiguousarray(queries, dtype=np.int64)
    return np.asarray(store.bp.applier_blast_probe(dev_applier.buffer, store.tables, ptrs,
                                                   n_gases, h, w, q, int(max_tiles)))


def take_sources(store, dev_applier, fields: dict, lists, source_out=None) -> tuple:
    """#29 P7c.2: slot 7's head on the device, ``blast_expansion.
    take_source_plane`` for every env -- the SOURCES stage of the one applier:
    each env's pending ``div_source`` plane (``fields``) handed out into
    ``source_out`` (a device ``(N, h, w)`` int32 array, or None) and zeroed,
    then the blast-source table (``lists.sources`` / ``lists.source_tiles``)
    added in row order, counted down and compacted. Returns ``(taken (N,)
    bool: the reference returned a plane, not None; books)``."""
    from simulation import applier
    _cov, books = walk_effects(store, dev_applier, fields, applier.STAGE_SOURCES, None,
                               lists, source_out=source_out)
    return books["source_taken"].astype(bool), books


def sources_of(sim) -> np.ndarray:
    """#29 P7c.2: the blast-source table's ONE record function over
    ``sim.gmap.expansions``."""
    from simulation import object_tables, unit_schema
    recs = object_tables.records(unit_schema.SOURCE_TABLE, sim.gmap.expansions)
    return np.array(recs, dtype=np.int64).reshape(
        len(recs), object_tables.table_layout(unit_schema.SOURCE_TABLE).width)


def source_tiles_of(sim) -> np.ndarray:
    """#29 P7c.2: the source tile side table's ONE record function (every
    source's walked tiles in list order)."""
    from simulation import object_tables, unit_schema
    recs = object_tables.side_records(unit_schema.SOURCE_TABLE, sim.gmap.expansions, "tiles")
    return np.array(recs, dtype=np.int64).reshape(
        len(recs), object_tables.table_layout(unit_schema.SOURCE_TILE_TABLE).width)


def sources_to_host(sim, recs, tile_recs) -> None:
    """#29 P7c.2: the device's blast-source rows as ``sim.gmap.expansions``
    (IN PLACE: one list object for the map's life), through the schema's
    decode (``object_tables.unpack`` with the side table)."""
    from simulation import object_tables, unit_schema
    lay = object_tables.table_layout(unit_schema.SOURCE_TABLE)
    tlay = object_tables.table_layout(unit_schema.SOURCE_TILE_TABLE)
    rows = np.asarray(recs, dtype=np.int64).reshape(-1, lay.width)
    trows = np.asarray(tile_recs, dtype=np.int64).reshape(-1, tlay.width)
    sim.gmap.expansions[:] = object_tables.unpack(
        unit_schema.SOURCE_TABLE, rows, rows.shape[0],
        sides={"tiles": (trows, trows.shape[0])})


def source_tables(n_env: int, bp=None):
    """#29 P7c.2: an env batch's blast-source table and its tile side table
    (device)."""
    from simulation import object_tables, unit_schema
    return (object_tables.DeviceTable(unit_schema.SOURCE_TABLE, n_env, bp=bp),
            object_tables.DeviceTable(unit_schema.SOURCE_TILE_TABLE, n_env, bp=bp))


__all__ += ["take_sources", "sources_of", "source_tiles_of", "sources_to_host",
            "source_tables"]


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


#: The plane slots 1b and 2 read for a thrown grenade (#215: the wall walk's):
#: "flight" = ``combat.flight_plane(gmap)`` (off the grid is blocked; solid or
#: a ``blocks_thrown`` material), never a GameMap attribute of its own.
GRENADE_FIELDS = (("flight", np.bool_),)


def grenades(store, table, dev_applier, ticks, fields: dict) -> np.ndarray:
    """Slot 2's grenades (``Simulation._update_grenades``): every live row
    flies (``Projectile.fly``, wide: #215 the current leg, the wall walk
    over ``fields["flight"]`` -- a DEVICE ``(N, h, w)`` bool array,
    ``combat.flight_plane`` as the slot found it -- and a bounce; on a
    fuse-out ``settle``), a row
    whose fuse ran out is marked detonated and -- its proj_type
    ORDER_GRENADE -- its detonation APPENDED to ``dev_applier``'s effect list
    as a PAYLOAD record (``execute_payload``'s call at
    ``combat.detonation_tile``, ``applier.record_of``'s fields), in row
    order after what the list holds. ``ticks`` ``(N,)`` each env's
    ``sim.tick``. Returns the rows changed per env; refuses a fault, a WEGO
    grenade (:class:`ObjectSlotRefused`) and an effect-list overflow."""
    _check_tables("grenades", store, table)
    ptr, h, w = _flight_plane("grenades", store, fields)
    t = np.ascontiguousarray(ticks, dtype=np.int64).reshape(-1)
    return _object_done("grenades", store.bp.unit_grenades(
        store.tables, table.buffer, dev_applier.buffer, t, ptr, h, w))


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
            "GRENADE_FIELDS", "charges_tick", "prune_grenades"]


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
                 blast_log=None, sources=None, source_tiles=None):
        self.cells, self.door, self.claims = cells, door, claims
        # #29 P7c.1: THE NOISE LIST and THE BLAST LOG (object tables) a
        # PAYLOAD record appends to
        self.noise, self.blast_log = noise, blast_log
        # #29 P7c.2: THE BLAST-SOURCE TABLE and its tile side table (object
        # tables, ``unit_schema.SOURCE_TABLE`` / ``SOURCE_TILE_TABLE``): a
        # comparison row's later ticks (start_expansion), walked by the
        # SOURCES stage (take_source_plane)
        self.sources, self.source_tiles = sources, source_tiles


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


# ---------------------------------------------------------------------------
# #29 P6b.3: THE SIGNALS -- the bus, the sensors, the logic nodes, the pumps
# and the doors' wire drives as entity tables (unit_schema.SIGNAL_TABLE,
# SENSOR_TABLE + SENSOR_TILE_TABLE, NODE_TABLE + NODE_IN_TABLE, PUMP_TABLE +
# PUMP_IN_TABLE, DOOR_DRIVE_TABLE) and slot 9e's (a), (b), (c), (d) pumps and
# (e) as the applier's EMIT, LOGIC, DOOR_INPUTS, PUMPS and SWAP stages
# (cpp/src/cuda_unit_signals.cuh)
# ---------------------------------------------------------------------------
class SignalTableMismatch(RuntimeError):
    """Signal-table rows that are not the sim's (another count, slot,
    ordinal, kind or load constant)."""


class SignalTableRefused(RuntimeError):
    """A bus slot the device's emit cannot serve, by name: an ``is_open`` of a
    non-door entity that carries a ``state`` (its state moves in the tick),
    or an ``alive`` of a class whose alive moves in the tick on the host only
    (a heat source, a cover row: P6b.4's / not ported)."""


#: The signal tables' list names in the applier's ``signal_tables`` order.
SIGNAL_LISTS = ("signals", "sensors", "sensor_tiles", "nodes", "node_ins", "pumps",
                "pump_ins", "door_drives")
_SIGNAL_TABLE_NAMES = ("signal", "sensor", "sensor_tile", "node", "node_in", "pump",
                       "pump_in", "door_drive")
#: Classes whose ``alive`` changes inside a tick on the host only. #29 P6b.4:
#: the heat source left this list -- the bus reads ``Simulation.
#: _entity_by_ordinal``, built BEFORE ``build_heat_sources`` replaced the
#: source's EntityInstance with its runtime, so the alive the reference emits
#: is the instance's (which has none: 1, always) -- a constant; were the map
#: ever to hold the runtime, :func:`signal_rows` refuses it by name.
_ALIVE_MOVES = ("cover",)


def _rows(recs, table) -> np.ndarray:
    from simulation import object_tables
    return np.asarray(recs, dtype=np.int64).reshape(
        -1, object_tables.table_layout(table).width)


def signal_rows(sim) -> list:
    """The bus (``sim._signal_bus``, slot order) as ``unit_schema.SignalRow``
    objects: each slot's pub / stg, node flag and its 9e(a) emit (a door's
    is_open / alive by its door-table row, a constant alive by its value);
    empty without a bus. Refuses an emit the device cannot serve by name
    (:class:`SignalTableRefused`)."""
    from simulation import unit_schema as us
    from simulation.signal_bus import ALIVE_NAME
    bus = getattr(sim, "_signal_bus", None)
    if bus is None:
        return []
    door_row = {int(d.ordinal): k for k, d in enumerate(getattr(sim, "_doors", ()) or ())}
    by_ord = getattr(sim, "_entity_by_ordinal", {}) or {}
    node = set(int(i) for i in bus._node_slots)
    out = []
    for idx, (o, n) in enumerate(bus.slots):
        ent = by_ord.get(int(o))
        emit, ref = "none", 0
        if ent is not None and n in ("is_open", ALIVE_NAME):
            if int(o) in door_row and ent is sim._doors[door_row[int(o)]]:
                emit, ref = ("door_open" if n == "is_open" else "door_alive"), door_row[int(o)]
            elif n == "is_open":
                if hasattr(ent, "state"):
                    raise SignalTableRefused(
                        f"signals: slot {idx} ({o}, 'is_open') reads the state of "
                        f"{ent.class_name!r} entity {ent.id!r}, which is not a door")
                emit, ref = "constant", 0      # getattr(ent, "state", None) != DOOR_OPEN
            else:
                if ent.class_name == "heat_source" and hasattr(ent, "hp_q16"):
                    raise SignalTableRefused(
                        f"signals: slot {idx} ({o}, 'alive') reads the alive of heat "
                        f"source {ent.id!r}'s RUNTIME, which moves in the tick (the "
                        f"device emits a constant: the reference's map holds the "
                        f"instance)")
                if ent.class_name in _ALIVE_MOVES:
                    raise SignalTableRefused(
                        f"signals: slot {idx} ({o}, 'alive') reads the alive of "
                        f"{ent.class_name!r} entity {ent.id!r}, which moves on the host "
                        f"only")
                emit, ref = "constant", 1 if getattr(ent, "alive", True) else 0
        out.append(us.SignalRow(ordinal=int(o), name=str(n), pub=int(bus.pub[idx]),
                                stg=int(bus.stg[idx]), node=idx in node, emit=emit,
                                emit_ref=int(ref)))
    return out


def sensor_rows(sim) -> list:
    """``sim._sensors`` (ordinal order) as ``unit_schema.SensorRow`` objects."""
    from simulation import sensor_system, unit_schema as us
    from simulation.gases import O2, SMOKE
    chans = us.sensor_channels()
    out = []
    for s in getattr(sim, "_sensors", ()) or ():
        kw = dict(ordinal=int(s.inst.ordinal), alive=bool(getattr(s.inst, "alive", True)),
                  slot=int(s.value_slot), channel=-1, gas=-1, sy=0, sx=0,
                  has_disc=False, disc=[], period=0, r2=0, needs_los=False, faction=0,
                  min_footprint=0)
        if isinstance(s, sensor_system._FieldSensorRuntime):
            ch = s.channel.value
            kw.update(kind="field", channel=chans.index(ch),
                      gas=SMOKE if ch == "smoke" else (O2 if ch == "o2" else -1),
                      sy=s.sample_tile[0], sx=s.sample_tile[1],
                      has_disc=s.disc is not None,
                      disc=[(int(y), int(x)) for y, x in (s.disc or ())])
        elif isinstance(s, sensor_system._ClockSensorRuntime):
            kw.update(kind="clock", period=int(s.period))
        elif isinstance(s, sensor_system._MotionSensorRuntime):
            kw.update(kind="motion", sy=s.sy, sx=s.sx, r2=s.r2, needs_los=s.needs_los,
                      faction=s.faction, min_footprint=s.min_footprint)
        else:                              # pragma: no cover - a new sensor runtime
            raise SignalTableRefused(f"signals: sensor runtime {type(s).__name__} has no "
                                     f"device row")
        out.append(us.SensorRow(**kw))
    return out


def node_rows(sim) -> list:
    """``sim._logic_nodes`` (ordinal order) as ``unit_schema.NodeRow`` objects:
    the predicate as its comparator CODE (never the lambda)."""
    from simulation import logic_nodes as ln, unit_schema as us
    from simulation.entities.schema import INPUT_AND
    insts = sorted((e for e in sim.entities if ln.is_logic_node(e.class_name)),
                   key=lambda e: int(e.ordinal))
    nodes = list(getattr(sim, "_logic_nodes", ()) or ())
    if len(insts) != len(nodes):
        raise SignalTableMismatch(f"nodes: {len(nodes)} evaluators for {len(insts)} "
                                  f"logic-node entities")
    out = []
    for e, n in zip(insts, nodes):
        kw = dict(ordinal=int(e.ordinal), out=None, ins=[], cmp=0, threshold=0,
                  require_alive=False, alive_slot=None, k=0, ema=0,
                  a_in=[None] * len(us.AIRLOCK_INPUTS),
                  a_out=[None] * len(us.AIRLOCK_OUTPUTS), far_is_outer=False,
                  equalize_cmd=0, interior=False, band=0, state=0, dwell=0)
        if isinstance(n, ln._DeciderEval):
            kw.update(kind="decider", out=n.out_slot, ins=list(n.in_slots),
                      cmp=us.comparators().index(n.comparator), threshold=n.threshold,
                      require_alive=n.require_alive, alive_slot=n.alive_slot)
        elif isinstance(n, ln._GateEval):
            kw.update(kind="gate_and" if n.mode == INPUT_AND else "gate_or",
                      out=n.out_slot, ins=list(n.in_slots))
        elif isinstance(n, ln._NotEval):
            kw.update(kind="gate_not", out=n.out_slot, ins=list(n.in_slots))
        elif isinstance(n, ln.FilterRuntime):
            kw.update(kind="filter", out=n.out_slot,
                      ins=[] if n.in_slot is None else [n.in_slot], k=n.k, ema=n.ema)
        elif isinstance(n, ln.AirlockControllerRuntime):
            kw.update(kind="airlock_controller",
                      a_in=[getattr(n, a) for a in us._AIRLOCK_IN_ATTR],
                      a_out=[getattr(n, a) for a in us._AIRLOCK_OUT_ATTR],
                      far_is_outer=n.far_is_outer, equalize_cmd=n.equalize_cmd,
                      interior=n.interior, band=n.band, state=n.state, dwell=n.dwell)
        else:                              # pragma: no cover - a new node class
            raise SignalTableRefused(f"signals: node evaluator {type(n).__name__} has no "
                                     f"device row")
        out.append(us.NodeRow(**kw))
    return out


def pump_rows(sim) -> list:
    """``sim._pumps`` (ordinal order) as ``unit_schema.PumpRow`` objects."""
    from simulation import unit_schema as us
    out = []
    for p in getattr(sim, "_pumps", ()) or ():
        src = p.source if p.source is not None else (0, 0)
        out.append(us.PumpRow(
            ordinal=int(p.ordinal), alive=bool(p.alive), at_target=int(p.at_target),
            at_slot=p.at_target_slot, port_y=p.port_y, port_x=p.port_x,
            has_source=p.source is not None, src_y=int(src[0]), src_x=int(src[1]),
            target=p.target_atm, band=p.band, release=p.release, delta_n_q=p.delta_n_q,
            ins=[(int(s), False) for s in p.inject_slots] +
                [(int(s), True) for s in p.extract_slots]))
    return out


def drive_rows(sim) -> list:
    """``sim._door_drives`` as ``unit_schema.DriveRow`` objects: door (= door
    table row) order, each door's open wires then its close wires."""
    from simulation import unit_schema as us
    drives = getattr(sim, "_door_drives", {}) or {}
    out = []
    for k, d in enumerate(getattr(sim, "_doors", ()) or ()):
        drv = drives.get(int(d.ordinal))
        if drv is None:
            continue
        out += [us.DriveRow(door=k, close=False, slot=int(s)) for s in drv["open"]]
        out += [us.DriveRow(door=k, close=True, slot=int(s)) for s in drv["close"]]
    return out


def _records_of(table, objs) -> np.ndarray:
    from simulation import object_tables
    return _rows(object_tables.records(table, objs), table)


def _side_of(table, objs, span) -> np.ndarray:
    from simulation import object_tables, unit_schema as us
    side = {s.name: s.side for s in table.spans}[span]
    return _rows(object_tables.side_records(table, objs, span),
                 {t.name: t for t in us.OBJECT_TABLES}[side])


def signals_of(sim) -> np.ndarray:
    """THE record function of the bus table: ``(S, W)`` int64, slot order."""
    from simulation import unit_schema as us
    return _records_of(us.SIGNAL_TABLE, signal_rows(sim))


def sensors_of(sim) -> np.ndarray:
    """THE record function of the sensor table (ordinal order)."""
    from simulation import unit_schema as us
    return _records_of(us.SENSOR_TABLE, sensor_rows(sim))


def sensor_tiles_of(sim) -> np.ndarray:
    """THE record function of the sensor disc side table."""
    from simulation import unit_schema as us
    return _side_of(us.SENSOR_TABLE, sensor_rows(sim), "disc")


def nodes_of(sim) -> np.ndarray:
    """THE record function of the node table (ordinal order)."""
    from simulation import unit_schema as us
    return _records_of(us.NODE_TABLE, node_rows(sim))


def node_ins_of(sim) -> np.ndarray:
    """THE record function of the node input side table."""
    from simulation import unit_schema as us
    return _side_of(us.NODE_TABLE, node_rows(sim), "ins")


def pumps_of(sim) -> np.ndarray:
    """THE record function of the pump table (ordinal order)."""
    from simulation import unit_schema as us
    return _records_of(us.PUMP_TABLE, pump_rows(sim))


def pump_ins_of(sim) -> np.ndarray:
    """THE record function of the pump input side table."""
    from simulation import unit_schema as us
    return _side_of(us.PUMP_TABLE, pump_rows(sim), "ins")


def door_drives_of(sim) -> np.ndarray:
    """THE record function of the door drive table."""
    from simulation import unit_schema as us
    return _records_of(us.DOOR_DRIVE_TABLE, drive_rows(sim))


#: list name -> its record function.
SIGNAL_RECORDS = {"signals": signals_of, "sensors": sensors_of,
                  "sensor_tiles": sensor_tiles_of, "nodes": nodes_of,
                  "node_ins": node_ins_of, "pumps": pumps_of, "pump_ins": pump_ins_of,
                  "door_drives": door_drives_of}


def _unpack(table, recs, sides=None) -> list:
    from simulation import object_tables
    r = _rows(recs, table)
    return object_tables.unpack(table, r, r.shape[0], sides=sides)


def signal_objects(recs) -> list:
    from simulation import unit_schema as us
    return _unpack(us.SIGNAL_TABLE, recs)


def sensor_objects(recs, tile_recs) -> list:
    from simulation import unit_schema as us
    t = _rows(tile_recs, us.SENSOR_TILE_TABLE)
    return _unpack(us.SENSOR_TABLE, recs, {"disc": (t, t.shape[0])})


def node_objects(recs, in_recs) -> list:
    from simulation import unit_schema as us
    t = _rows(in_recs, us.NODE_IN_TABLE)
    return _unpack(us.NODE_TABLE, recs, {"ins": (t, t.shape[0])})


def pump_objects(recs, in_recs) -> list:
    from simulation import unit_schema as us
    t = _rows(in_recs, us.PUMP_IN_TABLE)
    return _unpack(us.PUMP_TABLE, recs, {"ins": (t, t.shape[0])})


def drive_objects(recs) -> list:
    from simulation import unit_schema as us
    return _unpack(us.DOOR_DRIVE_TABLE, recs)


def signals_to_host(sim, recs) -> None:
    """The device's bus rows LANDED on the sim's own bus in place (pub and
    stg); another slot table is refused by name (:class:`SignalTableMismatch`)."""
    rows = signal_objects(recs)
    bus = getattr(sim, "_signal_bus", None)
    slots = () if bus is None else bus.slots
    if [(r.ordinal, r.name) for r in rows] != [(int(o), str(n)) for o, n in slots]:
        raise SignalTableMismatch(f"signals: {len(rows)} device rows are not the bus's "
                                  f"{len(slots)} slots")
    for i, r in enumerate(rows):
        bus.pub[i] = int(r.pub)
        bus.stg[i] = int(r.stg)


def _same_rows(what, got, want) -> None:
    if np.asarray(got, dtype=np.int64).tolist() != np.asarray(want, dtype=np.int64).tolist():
        raise SignalTableMismatch(f"{what}: the device rows are not the sim's (load "
                                  f"constants only: nothing lands)")


def sensors_to_host(sim, recs, tile_recs=None) -> None:
    """The sensor table holds load constants only: its rows must be the
    sim's (refused by name otherwise); nothing lands."""
    _same_rows("sensors", recs, sensors_of(sim))
    if tile_recs is not None:
        _same_rows("sensor tiles", tile_recs, sensor_tiles_of(sim))


def nodes_to_host(sim, recs, in_recs=None) -> None:
    """The device's node rows LANDED on the sim's own evaluators: a filter's
    ``ema``, an airlock controller's ``state`` / ``dwell``; a row whose
    ordinal, kind or wiring is not that node's is refused by name."""
    from simulation import logic_nodes as ln
    in_recs = node_ins_of(sim) if in_recs is None else in_recs
    rows = node_objects(recs, in_recs)
    mine = node_rows(sim)
    if len(rows) != len(mine):
        raise SignalTableMismatch(f"nodes: {len(rows)} device rows for {len(mine)} nodes")
    for k, (r, m) in enumerate(zip(rows, mine)):
        m.ema, m.state, m.dwell = r.ema, r.state, r.dwell
        if r != m:
            raise SignalTableMismatch(f"nodes: row {k} (ordinal {r.ordinal}) is not the "
                                      f"sim's node (ordinal {m.ordinal})")
    for r, n in zip(rows, sim._logic_nodes):
        if isinstance(n, ln.FilterRuntime):
            n.ema = int(r.ema)
        elif isinstance(n, ln.AirlockControllerRuntime):
            n.state, n.dwell = int(r.state), int(r.dwell)


def pumps_to_host(sim, recs, in_recs=None) -> None:
    """The device's pump rows LANDED on the sim's own ``PumpRuntime``s: the
    latch ``at_target``; another row is refused by name."""
    in_recs = pump_ins_of(sim) if in_recs is None else in_recs
    rows = pump_objects(recs, in_recs)
    mine = pump_rows(sim)
    if len(rows) != len(mine):
        raise SignalTableMismatch(f"pumps: {len(rows)} device rows for {len(mine)} pumps")
    for k, (r, m) in enumerate(zip(rows, mine)):
        m.at_target = r.at_target
        if r != m:
            raise SignalTableMismatch(f"pumps: row {k} (ordinal {r.ordinal}) is not the "
                                      f"sim's pump (ordinal {m.ordinal})")
    for r, p in zip(rows, sim._pumps):
        p.at_target = int(r.at_target)


def drives_to_host(sim, recs) -> None:
    """The door drives are load constants: the rows must be the sim's."""
    _same_rows("door drives", recs, door_drives_of(sim))


def signal_digest_rows(recs) -> tuple:
    """The ``__signals__`` payload read from the bus table: ``(ordinal, name,
    pub)`` for every non-``alive`` slot in slot order (P6b.4's export), held
    equal to ``SignalBus.digest_rows``."""
    return tuple((r.ordinal, r.name, int(r.pub)) for r in signal_objects(recs)
                 if r.name != "alive")


def node_entity_rows(rec) -> tuple:
    """A node row's runtime part of ENTITY_SECT_V1 after the free ``alive``
    row (always 1: nothing kills a node): a filter's ``ema``, an airlock
    controller's ``state`` / ``dwell``, nothing for the stateless nodes."""
    from simulation import object_tables, unit_schema as us
    v = object_tables.row_values(us.NODE_TABLE, rec)
    kind = us.NODE_KINDS[int(v["kind"])]
    rows = [("alive", 1)]
    if kind == "filter":
        rows.append(("ema", int(v["ema"])))
    elif kind == "airlock_controller":
        rows += [("state", int(v["state"])), ("dwell", int(v["dwell"]))]
    return tuple(rows)


def pump_entity_rows(rec) -> tuple:
    """A pump row's part of ENTITY_SECT_V1: the free ``alive`` row, then
    ``pump.runtime_digest_rows`` (``at_target``)."""
    from simulation import object_tables, unit_schema as us
    v = object_tables.row_values(us.PUMP_TABLE, rec)
    return (("alive", int(v["alive"])), ("at_target", int(v["at_target"])))


def signal_tables(n_env: int, bp=None) -> dict:
    """The eight device signal tables for ``n_env`` envs, by list name
    (:data:`SIGNAL_LISTS`; ``object_tables.DeviceTable`` each)."""
    from simulation import object_tables, unit_schema as us
    by = {t.name: t for t in us.OBJECT_TABLES}
    return {ln: object_tables.DeviceTable(by[tn], n_env, bp=bp)
            for ln, tn in zip(SIGNAL_LISTS, _SIGNAL_TABLE_NAMES)}


#: The filter's proved range (cpp/src/cuda_unit_signals.cuh).
FILTER_LIM, FILTER_K_MAX = 1 << 60, 62


def filter_range(node_rec, in_recs, bus_recs) -> list:
    """The operands of ONE node row outside the range the LOGIC stage was
    proved for, as ``(name, value, lo, hi)`` (empty: the node may run): a
    filter's ``k``, ``ema`` and input. The host twin of the stage's head."""
    from simulation import object_tables, unit_schema as us
    v = object_tables.row_values(us.NODE_TABLE, node_rec)
    if us.NODE_KINDS[int(v["kind"])] != "filter":
        return []
    out = []
    x = 0
    if int(v["in_count"]) > 0:
        slot = int(np.asarray(in_recs)[int(v["in_start"]), 0])
        x = int(object_tables.row_values(us.SIGNAL_TABLE, bus_recs[slot])["pub"])
    for name, val, lo, hi in (("k", int(v["k"]), 0, FILTER_K_MAX),
                              ("ema", int(v["ema"]), -FILTER_LIM, FILTER_LIM),
                              ("in", x, -FILTER_LIM, FILTER_LIM)):
        if not lo <= val <= hi:
            out.append((name, val, lo, hi))
    return out


def _signal_run(name, store, dev_applier, fields, stage, lists, ticks=None):
    from simulation import applier
    n = store.shape[0]
    if dev_applier.n_env != n:
        raise ValueError(f"unit kernel {name}: the applier is for {dev_applier.n_env} "
                         f"envs, the store {n}")
    tabs = [lists.get(ln) for ln in SIGNAL_LISTS]
    _check_tables(name, store, *[t for t in tabs if t is not None])
    doors = lists.get("doors")
    if fields is None:                    # a stage that reads no plane
        ptrs, n_gases, h, w = np.zeros(applier.N_PLANES, dtype=np.uint64), 1, 1, 1
    else:
        ptrs, n_gases, h, w = _plane_ptrs(name, n, fields)
    tk = None if ticks is None else np.ascontiguousarray(ticks, dtype=np.int64)
    b, _cov, _rdrop = store.bp.applier_run(
        dev_applier.buffer, store.buffer, store.tables, ptrs, n_gases, h, w, int(stage),
        None, door_table=None if doors is None else doors.buffer,
        signal_tables=[None if t is None else t.buffer for t in tabs], ticks=tk)
    books = applier.books_dict(b)
    store.last_books = books
    if books["faults"].any():
        envs = [int(e) for e in np.nonzero(books["faults"])[0]]
        raise applier.ApplierFault(
            f"unit kernel {name}: {int(books['faults'].sum())} fault(s) in env(s) {envs} "
            f"(a slot, door row or span outside its table, a sample tile off the grid, a "
            f"clock period < 1, a sight end past its range, or a pump's seam call the "
            f"reference would raise on / past its proved range)")
    return books["signal_writes"].copy(), books


def signals_emit(store, dev_applier, fields: dict, lists: dict, ticks) -> tuple:
    """Slot 9e(a) (``Simulation._signal_emit`` then ``sample_sensors``) as the
    applier's EMIT stage: per env the bus's emits in slot order (the door
    table's state / alive), then every sensor in ordinal order on the env's
    planes (``fields`` as :func:`walk_effects'`), unit rows (the motion
    plates) and tick (``ticks`` (N,), the clock). Writes the bus in place.
    Returns ``(coverage, books)``: coverage the bus writes that changed a
    value (``signal_writes``)."""
    from simulation import applier
    return _signal_run("applier[EMIT]", store, dev_applier, fields, applier.STAGE_EMIT,
                       lists, ticks)


def signals_logic(store, dev_applier, lists: dict) -> tuple:
    """Slot 9e(b) (``sweep_logic_nodes``) as the applier's LOGIC stage: every
    node in ordinal order reads pub and writes stg (the node table's ema /
    state / dwell in place). Refuses by name an env whose filter left the
    proved range (:class:`simulation.applier.SignalRangeRefused`, naming the
    env, the node and the operand: nothing of the env's nodes written)."""
    from simulation import applier
    name = "applier[LOGIC]"
    cov, books = _signal_run(name, store, dev_applier, None, applier.STAGE_LOGIC, lists)
    if books["signal_refused"].any():
        nr, nc, _ = lists["nodes"].download()
        ir, ic, _ = lists["node_ins"].download()
        br, bc, _ = lists["signals"].download()
        named = []
        for e in np.nonzero(books["signal_refused"])[0]:
            for k in range(int(nc[e])):
                bad = filter_range(nr[e, k], ir[e, :int(ic[e])], br[e, :int(bc[e])])
                if bad:
                    named.append(f"env {int(e)} node row {k} (ordinal {int(nr[e, k, 0])}): "
                                 + ", ".join(f"{nm} = {x} outside [{lo}, {hi}]"
                                             for nm, x, lo, hi in bad))
        raise applier.SignalRangeRefused(
            f"unit kernel {name}: {int(books['signal_refused'].sum())} env(s) refused at "
            f"the head of the node sweep, a filter outside the proved range (nothing of "
            f"them written): " + "; ".join(named))
    return cov, books


def signals_door_inputs(store, dev_applier, lists: dict) -> tuple:
    """Slot 9e(c) (``Simulation._resolve_door_inputs``) as the applier's
    DOOR_INPUTS stage: each wired door's ``want_open`` on the door table
    (close beats open; neither keeps the latch)."""
    from simulation import applier
    return _signal_run("applier[DOOR_INPUTS]", store, dev_applier, None,
                       applier.STAGE_DOOR_INPUTS, lists)


def signals_pumps(store, dev_applier, fields: dict, lists: dict) -> tuple:
    """Slot 9e(d)'s pumps (``sweep_pumps``) as the applier's PUMPS stage:
    every pump in ordinal order, its flux through the topology seam on the
    env's planes (in place, booked), then its latch and its slot."""
    from simulation import applier
    return _signal_run("applier[PUMPS]", store, dev_applier, fields, applier.STAGE_PUMPS,
                       lists)


def signals_swap(store, dev_applier, lists: dict) -> tuple:
    """Slot 9e(e) (``SignalBus.swap_node_signals``) as the applier's SWAP
    stage: pub <- stg on the node slots."""
    from simulation import applier
    return _signal_run("applier[SWAP]", store, dev_applier, None, applier.STAGE_SWAP,
                       lists)


__all__ += ["SignalTableMismatch", "SignalTableRefused", "SIGNAL_LISTS", "signal_rows",
            "sensor_rows", "node_rows", "pump_rows", "drive_rows", "signals_of",
            "sensors_of", "sensor_tiles_of", "nodes_of", "node_ins_of", "pumps_of",
            "pump_ins_of", "door_drives_of", "SIGNAL_RECORDS", "signal_objects",
            "sensor_objects", "node_objects", "pump_objects", "drive_objects",
            "signals_to_host", "sensors_to_host", "nodes_to_host", "pumps_to_host",
            "drives_to_host", "signal_digest_rows", "node_entity_rows",
            "pump_entity_rows", "signal_tables", "FILTER_LIM", "FILTER_K_MAX",
            "filter_range", "signals_emit", "signals_logic", "signals_door_inputs",
            "signals_pumps", "signals_swap"]


# ---------------------------------------------------------------------------
# #29 P7d -- SLOT 5, THE HORDE (ai_zombie.update_zombies_tick through
# Simulation._run_zombie_slot), the applier's HORDE stage
# ---------------------------------------------------------------------------
def horde(store, dev_applier, fields: dict, ring, noise, doors, tiles, cache,
          latch) -> tuple:
    """Slot 5 as the applier's HORDE stage (``cpp/src/cuda_unit_horde.cuh``):
    per env, the noise list ``noise`` DRAINED at the head (``noise.take``),
    then -- unless the latch's request set is empty or no prey lives (both
    before ``begin_tick``: the cache stays standing) -- the cache's refresh
    rule, passes 1 and 2 wide, the hearing, and pass 3 serial in row order:
    the bite (``ring`` gets its events), the walk on the cached fields (a field
    the walk needs first is FLOODED by the block from the live planes
    ``fields``, written in place by a door bash through the topology seam,
    its noise appended to ``noise``) and the wander. ``doors`` / ``tiles`` the
    door tables (the bashable mask); ``cache`` a
    :class:`simulation.horde_cache.DeviceHordeCache` (its planes allocated
    for the map on first use); ``latch`` the per-env LATCH rows the host
    writes (``horde_cache.latch_of``: the tick, the topology serial, the
    zombie request set). Returns ``(coverage, books)``. Refuses by name a fault
    (:class:`simulation.applier.ApplierFault`), a field past the cache's
    capacity (:class:`simulation.applier.HordeCacheOverflow`), a ring or noise
    list that overflowed."""
    from simulation import applier
    n = store.shape[0]
    name = "applier[HORDE]"
    if dev_applier.n_env != n or cache.n_env != n:
        raise ValueError(f"unit kernel {name}: the applier / cache are for "
                         f"{dev_applier.n_env} / {cache.n_env} envs, the store {n}")
    if cache.n_units < store.shape[1]:
        raise ValueError(f"unit kernel {name}: the cache's latch holds {cache.n_units} "
                         f"rows, the store {store.shape[1]}")
    _check_ring(name, store, ring)
    _check_tables(name, store, doors, tiles, noise)
    ptrs, n_gases, h, w = _plane_ptrs(name, n, fields)
    cache.ensure(h, w)
    cache.set_latch(latch)
    b, cov, rdrop = store.bp.applier_run(
        dev_applier.buffer, store.buffer, store.tables, ptrs, n_gases, h, w,
        int(applier.STAGE_HORDE), ring.buffer, noise=noise.buffer,
        door_table=doors.buffer, door_tiles=tiles.buffer, horde=cache.reach())
    books = applier.books_dict(b)
    cov = np.asarray(cov, dtype=np.int64)
    store.last_books = books
    store.last_coverage = cov
    if np.asarray(rdrop).any():
        envs = [int(e) for e in np.nonzero(np.asarray(rdrop))[0]]
        raise UnitEventOverflow(f"unit kernel {name}: events in env(s) {envs} did not fit "
                                f"the event ring")
    if books["horde_refused"].any():
        envs = [int(e) for e in np.nonzero(books["horde_refused"])[0]]
        raise applier.HordeCacheOverflow(
            f"unit kernel {name}: env(s) {envs} needed a field past the horde cache's "
            f"capacity {cache.fcap}")
    if books["faults"].any():
        envs = [int(e) for e in np.nonzero(books["faults"])[0]]
        raise applier.ApplierFault(
            f"unit kernel {name}: env(s) {envs} left the range the HORDE stage was "
            f"proved for (or a slot input was missing)")
    noise.check(name)
    return cov, books


def horde_flood_probe(bp, tile_ok, fp, seeds) -> np.ndarray:
    """#29 P7d, TEST-ONLY: the HORDE stage's flood on given masks. ``tile_ok``
    a device (N, h, w) uint8 array, ``fp`` (N,), ``seeds`` a list of N
    ``(k_e, 2)`` (y, x) arrays. Returns (N, h, w) int32 (host)."""
    import cupy as cp
    ok = cp.ascontiguousarray(tile_ok, dtype=cp.uint8)
    n, h, w = (int(x) for x in ok.shape)
    k = max(1, max(len(s) for s in seeds))
    sd = np.zeros((n, k, 2), dtype=np.int64)
    ns = np.zeros(n, dtype=np.int64)
    for e, s in enumerate(seeds):
        s = np.asarray(s, dtype=np.int64).reshape(-1, 2)
        sd[e, :len(s)] = s
        ns[e] = len(s)
    out = cp.zeros((n, h, w), dtype=cp.int32)
    bp.horde_flood_probe(n, h, w, int(ok.data.ptr),
                         np.ascontiguousarray(fp, dtype=np.int64), sd, ns, int(out.data.ptr))
    return out.get()


__all__ += ["horde", "horde_flood_probe"]


# ---------------------------------------------------------------------------
# #29 P6b.4: THE TIMED CHARGES and THE HEAT SOURCES on their entity tables
# (``unit_schema.TIMED_CHARGE_TABLE`` / ``HEAT_SOURCE_TABLE``), slot 9e(d')
# and 9e(d'') as the applier's TIMED_CHARGES and HEAT_SOURCES stages
# (cpp/src/cuda_unit_hazards.cuh), and THE EXPORT of the entity and signal
# sections of the digest from the device tables
# ---------------------------------------------------------------------------
class HazardTableMismatch(RuntimeError):
    """Timed-charge / heat-source rows that are not the sim's (another count,
    ordinal, tile or load constant), or an export missing a table."""


def timed_charges_of(sim) -> np.ndarray:
    """THE record function of the timed-charge table: ``sim._timed_charges``
    (ordinal order) as ``(C, W)`` int64 rows."""
    from simulation import unit_schema as us
    return _records_of(us.TIMED_CHARGE_TABLE, list(getattr(sim, "_timed_charges", ()) or ()))


def heat_sources_of(sim) -> np.ndarray:
    """THE record function of the heat-source table: ``sim._heat_sources``
    (ordinal order) as ``(S, W)`` int64 rows."""
    from simulation import unit_schema as us
    return _records_of(us.HEAT_SOURCE_TABLE, list(getattr(sim, "_heat_sources", ()) or ()))


def timed_objects(recs) -> list:
    """The timed-charge table's rows as ``unit_schema.TimedRow`` objects."""
    from simulation import unit_schema as us
    return _unpack(us.TIMED_CHARGE_TABLE, recs)


def heat_objects(recs) -> list:
    """The heat-source table's rows as ``unit_schema.HeatRow`` objects."""
    from simulation import unit_schema as us
    return _unpack(us.HEAT_SOURCE_TABLE, recs)


#: The columns that move after the build (the sweeps move all but a timed
#: charge's enabled, which only the host -- an editor, a fixture -- writes).
TIMED_STATE = ("ticks_to_fire", "fired", "enabled")
HEAT_STATE = ("hp_q16", "alive", "enabled")


def _land(what, table, recs, runtimes, state) -> None:
    """The device's rows LANDED on the sim's own runtimes in place: each
    runtime takes its row's ``state`` columns; a row whose other columns
    (ordinal, tile, load constants) are not that runtime's -- or another
    count -- is refused by name (nothing landed)."""
    from simulation import object_tables
    lay = object_tables.table_layout(table)
    rows = np.asarray(recs, dtype=np.int64).reshape(-1, lay.width)
    mine = _records_of(table, runtimes)
    if rows.shape[0] != mine.shape[0]:
        raise HazardTableMismatch(f"{what}: {rows.shape[0]} device rows for "
                                  f"{mine.shape[0]} runtimes")
    keep = np.ones(lay.width, dtype=bool)
    for c in state:
        col = lay.column(c)
        keep[col.offset:col.offset + col.width] = False
    for k in range(rows.shape[0]):
        if not np.array_equal(rows[k, keep], mine[k, keep]):
            raise HazardTableMismatch(f"{what}: row {k} (ordinal {int(rows[k, 0])}) is not "
                                      f"the sim's (ordinal {int(mine[k, 0])}): only "
                                      f"{state} may move")
    for rt, rec in zip(runtimes, rows):
        v = object_tables.row_values(table, rec)
        for c in state:
            cur = getattr(rt, c)
            setattr(rt, c, bool(v[c]) if isinstance(cur, bool) else int(v[c]))


def timed_charges_to_host(sim, recs) -> None:
    """The device's timed-charge rows LANDED on ``sim._timed_charges``: the
    countdown and the heat latch (refused by name otherwise)."""
    from simulation import unit_schema as us
    _land("timed charges", us.TIMED_CHARGE_TABLE, recs,
          list(getattr(sim, "_timed_charges", ()) or ()), TIMED_STATE)


def heat_sources_to_host(sim, recs) -> None:
    """The device's heat-source rows LANDED on ``sim._heat_sources``: hp,
    alive, enabled (refused by name otherwise)."""
    from simulation import unit_schema as us
    _land("heat sources", us.HEAT_SOURCE_TABLE, recs,
          list(getattr(sim, "_heat_sources", ()) or ()), HEAT_STATE)


def timed_entity_rows(rec) -> tuple:
    """A timed-charge row's part of ENTITY_SECT_V1: the free ``alive`` row,
    then ``timed_charge.runtime_digest_rows`` -- ``ticks_to_fire``, and
    ``fired`` on a heat charge only -- read from the table."""
    from simulation import object_tables, unit_schema as us
    v = object_tables.row_values(us.TIMED_CHARGE_TABLE, rec)
    rows = (("alive", int(v["alive"])), ("ticks_to_fire", int(v["ticks_to_fire"])))
    if int(v["heat"]):
        rows += (("fired", int(v["fired"])),)
    return rows


def heat_entity_rows(rec) -> tuple:
    """A heat-source row's part of ENTITY_SECT_V1: the free ``alive`` row,
    then ``heat_source.runtime_digest_rows`` (``hp``), read from the table."""
    from simulation import object_tables, unit_schema as us
    v = object_tables.row_values(us.HEAT_SOURCE_TABLE, rec)
    return (("alive", int(v["alive"])), ("hp", int(v["hp_q16"])))


def sensor_entity_rows(rec) -> tuple:
    """A sensor row's part of ENTITY_SECT_V1: the free ``alive`` row (no
    sensor class carries runtime rows), read from the table."""
    from simulation import object_tables, unit_schema as us
    v = object_tables.row_values(us.SENSOR_TABLE, rec)
    return (("alive", int(v["alive"])),)


def hazard_tables(n_env: int, bp=None):
    """The device timed-charge table and heat-source table for ``n_env``
    envs (``object_tables.DeviceTable`` each)."""
    from simulation import object_tables, unit_schema as us
    return (object_tables.DeviceTable(us.TIMED_CHARGE_TABLE, n_env, bp=bp),
            object_tables.DeviceTable(us.HEAT_SOURCE_TABLE, n_env, bp=bp))


def heat_chain_bound(heat_recs, n_blast: int) -> int:
    """The HEAT_SOURCES stage's proved bound on the discs its chain reads (the
    host twin of the stage's head): the log's ``n_blast`` discs at the head
    plus every LIVING source whose death payload's row has ``radius > 0`` --
    each source dies at most once, a death applies one payload, and only a
    row with ``radius > 0`` appends a disc (``execute_payload``)."""
    n = int(n_blast)
    for h in heat_objects(heat_recs):
        if h.alive and h.death_payload is not None and int(h.death_payload.radius) > 0:
            n += 1
    return n


def timed_charges(store, dev_applier, fields: dict, ring, lists, table) -> tuple:
    """Slot 9e(d') (``timed_charge_system.sweep_timed_charges``) as the
    applier's TIMED_CHARGES stage: every env's timed-charge table (``table``)
    walked in row (= ordinal) order on its planes (``fields`` as
    :func:`walk_effects`), each heat trigger read from the temperature mirror
    and each detonation applied AT ONCE through the PAYLOAD function
    (``lists`` a :class:`WalkLists` with the door, the claims, the noise
    list, the blast log and the blast-source tables). Each applied PAYLOAD
    record is appended to ``dev_applier``'s list (what the stage applied:
    compared with the host's ``execute_payload`` calls, never walked).
    Returns ``(coverage, books)``; refuses what :func:`walk_effects`
    refuses."""
    from simulation import applier
    _check_tables("applier[TIMED_CHARGES]", store, table)
    return walk_effects(store, dev_applier, fields, applier.STAGE_TIMED_CHARGES, ring,
                        lists, timed_table=table)


def heat_sources(store, dev_applier, fields: dict, ring, lists, table) -> tuple:
    """Slot 9e(d'') (``heat_source_system.sweep_heat_sources``) as the
    applier's HEAT_SOURCES stage: per env the blast log (``lists.blast_log``:
    the tick's discs so far) read as a GROWING list into the heat-source
    table (``table``, row order) -- a source a disc takes to hp 0 dies and
    its death payload is applied AT ONCE, its disc read by the same loop --,
    the log emptied, then every living, enabled, uncapped source's deposit
    through the edit door. The applied PAYLOAD records are appended to
    ``dev_applier``'s list. Refuses by name an env whose chain's proved bound
    (:func:`heat_chain_bound`) passes the blast log's capacity
    (:class:`simulation.applier.HeatChainRefused`, nothing of the env
    written), and what :func:`walk_effects` refuses."""
    from simulation import applier
    name = "applier[HEAT_SOURCES]"
    _check_tables(name, store, table)
    if lists is None or lists.blast_log is None:
        raise ValueError(f"unit kernel {name}: the sweep reads the blast log "
                         f"(lists.blast_log)")
    blast_before = lists.blast_log.download()[1].copy()
    cov, books = walk_effects(store, dev_applier, fields, applier.STAGE_HEAT_SOURCES, ring,
                              lists, heat_table=table)
    if books["heat_refused"].any():
        envs = [int(e) for e in np.nonzero(books["heat_refused"])[0]]
        hr, hc, _ = table.download()
        named = [f"env {e}: bound {heat_chain_bound(hr[e, :int(hc[e])], int(blast_before[e]))}"
                 for e in envs]
        raise applier.HeatChainRefused(
            f"unit kernel {name}: {len(envs)} env(s) refused at the head of the heat-"
            f"source sweep: the chain's proved bound (the blast log's discs plus the living "
            f"sources whose death payload appends a disc) passes the blast log's capacity "
            f"{lists.blast_log.shape[1]} ({'; '.join(named)}; nothing of the env written)")
    return cov, books


__all__ += ["HazardTableMismatch", "timed_charges_of", "heat_sources_of", "timed_objects",
            "heat_objects", "TIMED_STATE", "HEAT_STATE", "timed_charges_to_host",
            "heat_sources_to_host", "timed_entity_rows", "heat_entity_rows",
            "sensor_entity_rows", "hazard_tables", "heat_chain_bound", "timed_charges",
            "heat_sources"]


#: The list names of the tables :class:`EntityExport` reads.
EXPORT_LISTS = ("doors", "door_tiles", "ducts", "vents", "nodes", "pumps", "sensors",
                "timed_charges", "heat_sources")


def export_records(sim) -> dict:
    """The host's own records of every table :class:`EntityExport` reads (the
    record functions), by list name."""
    return {"doors": doors_of(sim), "door_tiles": door_tiles_of(sim), "ducts": ducts_of(sim),
            "vents": vents_of(sim), "nodes": nodes_of(sim), "pumps": pumps_of(sim),
            "sensors": sensors_of(sim), "timed_charges": timed_charges_of(sim),
            "heat_sources": heat_sources_of(sim)}


class EntityExport:
    """THE EXPORT of the digest's entity and signal sections from the device
    tables (#29 P6b.4): ``ENTITY_SECT_V1`` / ``SIGNAL_SECT_V1`` bytes equal to
    the host serializer's (``entities/serialize``), built once per sim.

    Each entity's record is ``serialize.record_head`` (the header and the
    declared synced fields: load constants, made once here) followed by
    ``serialize.record_tail`` of its moving rows, read from its class's
    table row: a door from the door table (its hp from the tile table), a
    duct, vent, logic node, pump, sensor, timed charge or heat source from
    theirs, by ordinal. A class with NO device table (button, terminal,
    light, prop, the zones, cover) has no row that moves on the continuous
    path: its tail is taken once, here, from the host (a load constant; the
    gates hold the export equal to the serializer every tick). The signal
    section is ``serialize_signal_state`` of the bus table's
    :func:`signal_digest_rows` (empty without a bus)."""

    def __init__(self, sim):
        from simulation import logic_nodes as ln
        from simulation.entities import REGISTRY
        from simulation.entities.serialize import record_head, record_tail
        ents = sorted(sim.entities, key=lambda e: int(e.ordinal))
        ordinals = {e.id: int(e.ordinal) for e in ents}
        door_ords = [int(d.ordinal) for d in (getattr(sim, "_doors", ()) or ())]
        sensor_ords = {int(s.inst.ordinal) for s in (getattr(sim, "_sensors", ()) or ())}
        self._tail = record_tail
        self.has_bus = getattr(sim, "_signal_bus", None) is not None
        self.items = []                      # (head, source, key, entity id)
        for e in ents:
            o = int(e.ordinal)
            head = record_head(e, REGISTRY, ordinals)
            c = e.class_name
            if o in door_ords:
                src, key = "doors", door_ords.index(o)
            elif c in ("duct", "vent", "pump", "timed_charge", "heat_source"):
                src, key = {"duct": "ducts", "vent": "vents", "pump": "pumps",
                            "timed_charge": "timed_charges",
                            "heat_source": "heat_sources"}[c], o
            elif ln.is_logic_node(c):
                src, key = "nodes", o
            elif o in sensor_ords:
                src, key = "sensors", o
            else:                            # no device table: a load constant
                alive = 1 if getattr(e, "alive", True) else 0
                src = "static"
                key = record_tail(e.id, (("alive", alive),)
                                  + tuple(REGISTRY[c].runtime_digest_rows(e)))
            self.items.append((head, src, key, e.id))
        self.sources = sorted({s for _h, s, _k, _i in self.items if s != "static"})
        if "doors" in self.sources:
            self.sources.append("door_tiles")

    _ROWS = {"ducts": "duct_entity_rows", "vents": "vent_entity_rows",
             "nodes": "node_entity_rows", "pumps": "pump_entity_rows",
             "sensors": "sensor_entity_rows", "timed_charges": "timed_entity_rows",
             "heat_sources": "heat_entity_rows"}

    def entity_section(self, tables: dict) -> bytes:
        """ONE env's ``ENTITY_SECT_V1`` bytes from its tables: ``tables`` maps
        a list name (:data:`EXPORT_LISTS`) to that env's records; a class the
        sim holds whose table is missing, or an entity with no row, is refused
        by name (:class:`HazardTableMismatch`)."""
        from simulation.entities.serialize import ENTITY_SECT_PREAMBLE
        missing = [s for s in self.sources if s not in tables]
        if missing:
            raise HazardTableMismatch(f"entity export: the tables {missing} are missing")
        idx = {s: {int(r[0]): r for r in np.asarray(tables[s], dtype=np.int64)}
               for s in self.sources if s not in ("doors", "door_tiles")}
        g = globals()
        parts = [ENTITY_SECT_PREAMBLE]
        for head, src, key, eid in self.items:
            if src == "static":
                parts.append(head + key)
                continue
            if src == "doors":
                doors = np.asarray(tables["doors"], dtype=np.int64)
                if key >= doors.shape[0]:
                    raise HazardTableMismatch(f"entity export: door {eid!r} has no row "
                                              f"{key} in the door table")
                rows = door_entity_rows(doors[key], tables["door_tiles"])
            else:
                rec = idx[src].get(key)
                if rec is None:
                    raise HazardTableMismatch(f"entity export: entity {eid!r} (ordinal "
                                              f"{key}) has no row in the {src} table")
                rows = g[self._ROWS[src]](rec)
            parts.append(head + self._tail(eid, rows))
        return b"".join(parts)

    def signal_section(self, signal_recs=None) -> bytes:
        """ONE env's ``SIGNAL_SECT_V1`` bytes from its bus table's records."""
        from simulation.entities.serialize import serialize_signal_state
        if not self.has_bus:
            return serialize_signal_state(())
        return serialize_signal_state(signal_digest_rows(signal_recs))

    @staticmethod
    def digest(section: bytes) -> str:
        """A section's digest as ``tests/field_digest.tick_digest`` folds it."""
        import hashlib
        return hashlib.blake2b(section, digest_size=32).hexdigest()


__all__ += ["EXPORT_LISTS", "export_records", "EntityExport"]
