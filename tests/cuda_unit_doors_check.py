"""#29 P6b.1 -- THE DOORS GATE: the door table (the first entity table, with
its tile side table), ``seal_tiles`` / ``unseal_tiles`` / ``can_seal_tiles``,
the gas-N primitives and ``_aperture_blocked`` in the Device topology seam,
slot 9e(d)'s door sweep as the applier's DOORS stage and slot 1b's USE as a
door-table write, each against its Python law at tol 0 (design 4.4c, the cut
table's P6b.1 row).

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_doors.py). The tables are ``simulation.object_tables.
DeviceTable`` over ``unit_schema.DOOR_TABLE`` / ``DOOR_TILE_TABLE`` (the
harness lists ``doors`` / ``door_tiles``, record functions
``unit_kernels.doors_of`` / ``door_tiles_of``); the kernels
``unit_kernels.sweep_doors`` (``cpp/src/cuda_applier.cu`` over
``cpp/src/cuda_unit_doors.cuh`` and ``cpp/src/cuda_topology.cuh``),
``unit_kernels.door_uses`` (``cpp/src/cuda_unit_doors.cu``) and the TEST-ONLY
``unit_kernels.seam_probe`` (``cpp/src/cuda_topology_probe.cu``).

PART TABLES -- every P6b.1 device table row equals an independent read of its
reference value; the door tables' fingerprint is compiled in.
PART PROBE -- seal / unseal / can_seal on real maps with real gas (test_level,
test_arena, door_test, airlock_demo, wind_studio, both door scenes; every
door's span, constructed horizontal and vertical runs of 1 to 6 tiles, runs
beside vacuum and beside the ambient ring, with smoke near the trace
saturation), each op run by the device on the map's planes and by the host
on the map itself: every plane bit for bit, every book, the energy / mass /
solid closure on both sides; EVERY refusal reason taken (bounds, already
solid, unknown material, a material that is not solid, standing water, a
sealed pocket, an int32 overflow, a duplicate tile, unseal over an open tile;
and the device-only span_too_long and unsorted span) with nothing written on
either side. The gas-N primitives (inject / extract / transfer) and
_aperture_blocked on synthetic and real states over their range (no air,
negative holdings, a vacuum tile, a trace slice near its saturation, a
destination at the int32 edge, out of bounds), results, planes and books equal.
PART SCENE -- the SHADOW LOCKSTEP of slot 1b (+ the use write) and slot 9e(d)
on the DOOR SCENE (door ENTITIES: ship_demo has none): every tick the door
table, the tile table, every plane, the books and the write set equal; the
table's IDENTITY (unpack(pack(doors)) is the doors; its entity rows are the
serializer's) every tick; a CENSUS of the sweep's branches, each required,
the unreachable ones named; the hooked match equal to an unhooked twin.
PART BATCH -- N = 2 and 8 of envs that differ in layout, door count, seed,
boundary and fields (both door-scene variants, test_arena's bots match,
airlock_demo, test_level, door_test): env i == env i alone every tick.
PART PADDING -- garbage rows past each table's count neither read nor written.
PART OVERFLOW -- the door table, the tile table and one door's span past
their capacities refused by name.
PART CHAINED -- slot 1b (+ the use write) and slot 9e(d) replaced by the
device's in a twin match on the door scene (the tables landed back on the
door runtimes, the books and the host-only side landed): the unit hash, the
field digest, every plane, the books, the door table and the entity section
equal an untouched reference every tick; the host's sweep ran 0 times.
PART SHIP -- ship_demo's bots match, 500 ticks, both slots shadowed (the sweep
idle: no door entity).

PROPERTY: the door table is the sim's doors; the device seal / unseal /
can_seal and the gas-N primitives are GameMap's integer for integer, refusals
included (nothing written); the device sweep is ``door_system.sweep_doors``
on the table (ordinal order, the occupancy test, external destruction, the hp
copy and restamp) and the USE write is ``_direct_use``'s flip; in a batch as
alone. BREAKS IF any plane family a seal / unseal writes, the trace
saturation, an energy move, the close-T mean, a refusal, a primitive's split,
the occupancy test, the ordinal order, the whole-door rule, the hp copy or
restamp, the use write or the batch indexing differs from the law.

Prints per-part lines and ``UNIT_DOORS_RESULT: PASS`` / ``FAIL``; exits 0 /
1. ``bench()`` prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import (ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep, first_field_diff)
from simulation import (applier, object_tables as ot, unit_kernels,  # noqa: E402
                        unit_requests, unit_schema, unit_store, unit_tables)
from simulation.gases import INERT_N2, N_GASES, O2, SMOKE  # noqa: E402
from simulation.materials import (MAT_AIR, MAT_DOOR_CLOSED, MAT_HULL,  # noqa: E402
                                  MATERIAL_NAMES)

FAILS: list = []
ONE = 65536
DT = unit_schema.DOOR_TABLE
TT = unit_schema.DOOR_TILE_TABLE
DLAY = ot.table_layout(DT)
TLAY = ot.table_layout(TT)
SCENE_TICKS = int(os.environ.get("UNIT_DOORS_SCENE_TICKS", "110"))
BATCH_TICKS = int(os.environ.get("UNIT_DOORS_BATCH_TICKS", "80"))
SHIP_TICKS = int(os.environ.get("UNIT_DOORS_SHIP_TICKS", "500"))
CHAIN_TICKS = int(os.environ.get("UNIT_DOORS_CHAIN_TICKS", "110"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_DOORS_BATCH_NS", "2,8").split(",")]
SPACE = level_loader.SPACE_CODE
CODE = {n: i + 1 for i, n in enumerate(applier.SEAM_REFUSAL_NAMES)}


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
PLANE_OUTPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)
DOOR_BOOK_NAMES = ("n_destruction_seed", "ge_retire", "ge_mint", "ge_destroy_seed",
                   "solid_destroy_wall", "tiles_changed", "ge_seal",
                   "ge_diag_seal_from", "ge_diag_seal_to", "ge_unseal",
                   "ge_diag_unseal_from", "ge_diag_unseal_to", "solid_seal",
                   "solid_unseal") + tuple(f"trace_clip_{g}" for g in range(N_GASES))
DOOR_BOOKS = tuple("book:" + b for b in DOOR_BOOK_NAMES)
HOST_ONLY = (
    ("render_dirty", "the wall seam's render mark (the renderer drains it; no "
     "device form)"),
    ("_sky_mask_dirty", "the sky exchange's lazy-rebuild flag (no device form)"),
    ("_gas_t_amb_raw_cache", "a lazily filled cache of a config constant (the "
     "device reads the table row topo_t_amb_q)"),
    # #29 P6b.2: the door scene never sealed a smoky tile; the vent scene does
    ("smoke", "GameMap.smoke IS a view of gas[SMOKE] (the same memory): compared "
     "through field:gas"),
)
_APPLIERS: dict = {}


def _applier_for(store):
    n = store.shape[0]
    if n not in _APPLIERS:
        _APPLIERS[n] = applier.DeviceApplier(n, 4, bp=bp)
    return _APPLIERS[n]


def _run_doors(st, inp, lists):
    return unit_kernels.sweep_doors(st, _applier_for(st), inp.fields, lists["doors"],
                                    lists["door_tiles"])


def _run_1b(st, inp, req, lists):
    from cuda_unit_lockstep import flight_fields       # #215: the flight plane
    cov = unit_kernels.verbs(st, req, inp.tick_row, inp.tables["door_spans"],
                             inp.scalars["charges_live"], flight_fields(inp, st))
    unit_kernels.door_uses(st, req, lists["doors"])
    return cov


# The sweep writes no unit row (columns=("id",)); the conductor runs it only
# on a level with doors (skip_ok).
SLOT_9ED = DeviceSlot(label="9e(d) doors", run=_run_doors, columns=("id",), skip_ok=True,
                      inputs=("list:door_tiles", "list:doors"),
                      outputs=PLANE_OUTPUTS + DOOR_BOOKS + ("list:door_tiles", "list:doors"),
                      host_only=HOST_ONLY)
SLOT_1B = DeviceSlot(label="1b intents", run=_run_1b,
                     inputs=("tick", "door_spans", "charges_live", "flight_plane", "list:doors"),
                     requests=True, outputs=("list:doors",))
SLOTS = [SLOT_1B, SLOT_9ED]


# ---------------------------------------------------------------------------
# The reference values of the P6b.1 table rows (cuda_unit_kernels_check's
# every-row gate reads this too)
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """An INDEPENDENT read of every P6b.1 table row: the materials module's
    id, the map's own materials and gas tables, the door schema's codes."""
    from simulation.entities import door as door_schema
    g = sim.gmap
    nm = len(MATERIAL_NAMES)
    return {
        "door_mat_closed": int(MAT_DOOR_CLOSED),
        "seal_mat_ok": tuple(0 if float(p) > 0.0 else 1 for p in g.materials.permeability[:nm]),
        "gas_trace": tuple(0 if bool(c) else 1 for c in g.gases.conservative),
        "inject_o2_frac_q": int(round(0.21 * ONE)),
        "door_states": (door_schema.DOOR_CLOSED, door_schema.DOOR_OPEN,
                        door_schema.DOOR_DESTROYED),
    }


# ---------------------------------------------------------------------------
# The door scene
# ---------------------------------------------------------------------------
def _door_entity(eid, ordinal, x, y, orient, length, state):
    from level_loader import EntityInstance
    from simulation.entities import door as door_mod
    fields = {f.name: f.default for f in door_mod.door.FIELDS}
    fields.update(x=x, y=y, orientation=orient, length_m=float(length), initial_state=state)
    return EntityInstance(id=eid, class_name="door", ordinal=ordinal, fields=fields)


def _door_level(variant):
    """1 m tiles, 40 x 64, space past the east hull (x 63). Rooms A (x 1..15)
    | D0 in the wall x 16 | B (x 17..29) | D1 in the wall x 30 | the 1-wide
    corridor x 31 | D2 in the wall x 32 | C (x 33..61) | D3 in the east hull
    x 62 (to space). D4 / D5 in the wall row y 30 under A; D6 an OPEN door in
    a one-tile pocket (no receiver: its close is refused), D7 a CLOSED door
    in another (it opens empty). Odd variants: planetside (the ambient
    ring), D0 moved, D5 dropped; variants 2, 3, 6, 7: a wall across C with
    D8 in it (another layout, another door count)."""
    from level_loader import LevelData
    h, w = 40, 64
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:h - 1, 1:62] = MAT_AIR
    tm[:, 63] = SPACE
    d0y = 4 + 6 * (variant % 2)
    tm[1:h - 1, 16] = MAT_HULL
    tm[d0y:d0y + 3, 16] = MAT_AIR
    tm[1:h - 1, 30] = MAT_HULL
    tm[1:h - 1, 32] = MAT_HULL
    tm[10:13, 30] = MAT_AIR
    tm[10:13, 32] = MAT_AIR
    tm[20:23, 62] = MAT_AIR
    tm[30, 1:16] = MAT_HULL
    tm[30, 5:8] = MAT_AIR
    tm[30, 10:12] = MAT_AIR
    for (py, px) in ((35, 40), (35, 50)):
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            tm[py + dy, px + dx] = MAT_HULL
    ents = [_door_entity("d3", 3, 62, 20, "v", 3, "closed"),
            _door_entity("d1", 1, 30, 10, "v", 3, "open"),
            _door_entity("d0", 0, 16, d0y, "v", 3, "closed"),
            _door_entity("d2", 2, 32, 10, "v", 3, "closed"),
            _door_entity("d4", 4, 5, 30, "h", 3, "closed"),
            _door_entity("d6", 6, 40, 35, "v", 1, "open"),
            _door_entity("d7", 7, 50, 35, "v", 1, "closed")]
    if variant % 2 == 0:
        ents.append(_door_entity("d5", 5, 10, 30, "h", 2, "closed"))
    if (variant // 2) % 2 == 1:                 # another layout: a wall across C, D8 in it
        tm[15, 33:62] = MAT_HULL
        tm[15, 45:48] = MAT_AIR
        ents.append(_door_entity("d8", 8, 45, 15, "h", 3, "open"))
    return LevelData(name=f"p6b1_doors{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."), entities=ents,
                     boundary="ambient" if variant % 2 else "vacuum"), d0y


class DoorScene:
    """The door scene: units beside every door, driven by scripted USE
    intents through the intent facade; fixtures between ticks (a teleport
    into a doorway and out, a death in a doorway, a wall destroyed under a
    closed door -- as a blast or a chew would -- and under an open one).
    ``variant`` shifts the schedule, the seed, the layout and the boundary."""

    def __init__(self, variant=0):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.v, self.k = variant, 0
        level, self.d0y = _door_level(variant)
        sim = Simulation(level, seed=61100 + variant, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        y0 = self.d0y
        self.u4 = self._add("U4", 13, y0)            # west of D0
        self.u5 = self._add("U5", 17, y0 + 3)        # east of D0, below the doorway
        self.u1 = self._add("U1", 27, 10)            # west of D1
        self.u2 = self._add("U2", 33, 10)            # east of D2
        self.u3 = self._add("U3", 59, 20)            # inside of D3
        self.u6 = self._add("U6", 41, 36)            # by the pocket door D6
        self.u7 = self._add("U7", 51, 36)            # by the pocket door D7
        self.u8 = (self._add("U8", 45, 16)           # under D8
                   if (variant // 2) % 2 == 1 else None)

    def _add(self, name, x, y):
        from simulation.unit import Unit
        u = Unit(name, x=float(x), y=float(y), team=0)
        self.sim.add_unit(u)
        u.current_hp = 1.0e6
        return u

    def _use(self, *units):
        for u in units:
            self.sim.use_intent(u.id)

    def _kill(self, u):
        from simulation.damage import DamagePacket, apply_packet
        apply_packet(u, DamagePacket(amount=1.0e7, dtype=0, attacker_id=-1), [],
                     source="melee")

    def tick(self):
        sim, k = self.sim, self.k - self.v
        y0 = self.d0y
        if k == 3:
            self._use(self.u4)                      # D0 opens
        if k == 6:
            self.u4.set_pos_q(15 * ONE, y0 * ONE)   # into the doorway
            self._use(self.u4)                      # the close is blocked
        if k == 12:
            self.u4.set_pos_q(13 * ONE, y0 * ONE)   # out: the retried close succeeds
        if k == 16:
            self._use(self.u4, self.u5)             # two uses of D0 in one tick
        if k == 18:                                 # a round chews the closed D2: its
            from simulation import combat           # hp copied at the open, restamped
            hp = int(sim.gmap.wall_hp[11, 32])      # at the close
            combat.chew_wall(sim.gmap, 11, 32, hp / ONE / 4)
        if k == 20:
            self._use(self.u1, self.u2)             # D1 closes, D2 opens: one tick
        if k == 25:
            self._use(self.u3)                      # D3 opens to space / the sky
        if k == 30:
            sim.gmap.destroy_wall(30, 6)            # a blast through D4's middle
        if k == 31 and self.v % 2 == 0:
            sim.gmap.destroy_wall(30, 10)           # D5: both tiles
            sim.gmap.destroy_wall(30, 11)
        if k == 33 and self.u8 is not None:
            self._use(self.u8)                      # D8 closes
        if k == 35:
            self._use(self.u6)                      # D6: the close refused (a pocket)
        if k == 40:
            self._use(self.u3)                      # D3 closes against the boundary
        if k == 45:
            self._use(self.u7)                      # D7 opens empty
        if k == 50:
            self._use(self.u7)                      # D7 closes, gas-free, no receiver
        if k == 52:
            self._use(self.u1, self.u2)             # D1 opens, D2 closes: one tick
        if k == 55:
            self._use(self.u4)                      # D0 opens
        if k == 58:
            sim.gmap.destroy_wall(y0 + 1, 16)       # a "blast" on the open D0: air, nothing
        if k == 62:
            self._use(self.u4)                      # D0 closes
        if k == 64:
            self._use(self.u5)                      # D0 opens
        if k == 66:
            self.u5.set_pos_q(15 * ONE, y0 * ONE)   # a corpse in the doorway
            self._kill(self.u5)
        if k == 68:
            self._use(self.u4)                      # D0 closes over the corpse
        if k == 75:
            sim.gmap.destroy_wall(y0 + 2, 16)       # a blast through the closed D0
        sim.step()
        sim.set_paused(False)
        self.k += 1


class LevelDriver:
    """A shipped level's sim (its doors, its entities), stepped; with
    ``match`` a MatchSession (its bots and zombies)."""

    def __init__(self, name, seed=1, match=False):
        level = level_loader.load(name, str(_ROOT / "levels"))
        if match:
            from simulation.match_session import MatchSession
            self.session = MatchSession(level, seed=seed, breach_physics=bp)
            self.sim = self.session.sim
        else:
            from simulation import Simulation
            from simulation.ruleset import ContinuousRealtime
            self.session = None
            self.sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False,
                                  ruleset=ContinuousRealtime())
            self.sim.set_paused(False)

    def tick(self):
        if self.session is not None:
            self.session.tick()
        else:
            self.sim.step()
            self.sim.set_paused(False)

    def is_terminal(self):
        return self.session is not None and self.session.is_terminal()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _planes(gmap) -> dict:
    return {n: np.array(getattr(gmap, n), copy=True) for n in applier.PLANE_NAMES}


def _dev_planes(per_env: list) -> dict:
    import cupy as cp
    return {n: cp.asarray(np.ascontiguousarray(np.stack([p[n] for p in per_env])))
            for n in applier.PLANE_NAMES}


def _books(gmap) -> dict:
    return {b.name: applier.host_book(gmap, b.host) for b in applier.BOOKS
            if b.host is not None}


def _store(sims, U=None):
    U = U or max([1] + [len(s.units) for s in sims])
    return unit_store.DeviceStore(len(sims), U, bp=bp, tables=unit_tables.values_batch(sims))


def _cap_real(shift):
    s = np.clip(shift.astype(np.int64), -16, 30)
    return np.left_shift(np.int64(1), s + 16)


def _closure(tag, pre, post, books, e=0, mass=True):
    """The books close on ``post`` (a plane dict) against ``pre``: the gas
    energy (every cell's: a retired cell is 0) moves by the non-diagnostic
    channels; the bulk mass by the destruction seed (seal / unseal and the
    transfer conserve it); each trace slice by minus its clip; the solid
    books by their channels."""
    def e_all(P):
        return int(P["gas_energy"].astype(object).sum())

    def s_sum(P):
        ts = P["thermal_solid"]
        return int((_cap_real(P["heat_inv_shift"][ts]).astype(object)
                    * P["temperature"][ts].astype(object)).sum())

    seam = sum(int(books[b.name][e]) for b in applier.BOOKS
               if b.name.startswith("ge_") and "diag" not in b.name)
    de = e_all(post) - e_all(pre)
    if de != seam:
        fail(f"{tag}: d sum gas_energy {de} != the seam's channels {seam}")
        return False
    ds = s_sum(post) - s_sum(pre)
    sb = sum(int(books[n][e]) for n in ("solid_seal", "solid_unseal", "solid_destroy_wall"))
    if ds != sb:
        fail(f"{tag}: d sum solid books {ds} != the solid channels {sb}")
        return False
    if mass:                       # seal / unseal / transfer conserve every slice
        for g in range(N_GASES):
            dn = int(post["gas"][g].astype(np.int64).sum() - pre["gas"][g].astype(np.int64).sum())
            want = 0 if g in (O2, INERT_N2) else -int(books[f"trace_clip_{g}"][e])
            if dn != want:
                fail(f"{tag}: gas slice {g} moved by {dn}, want {want} (minus its clip)")
                return False
    return True


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def part_tables():
    sims = [DoorScene(0).sim, DoorScene(1).sim]
    st = _store(sims)
    got = st.tables.download()
    n = 0
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    if bp.object_tables_layout()["fingerprint"] != ot.layout().fingerprint:
        fail("tables: the compiled object tables are not the schema's")
    print(f"tables: {n} P6b.1 values over 2 envs (space, planetside) equal the "
          f"reference; the door table (W {DLAY.width}, cap {DLAY.capacity}) and the "
          f"tile table (W {TLAY.width}, cap {TLAY.capacity}) compiled in")


# ---------------------------------------------------------------------------
# PART PROBE
# ---------------------------------------------------------------------------
def _host_code(ex, fn):
    from simulation.gamemap import SealBlocked
    m = str(ex)
    if isinstance(ex, OverflowError):
        return CODE["overflow"]
    if "duplicate" in m:
        return CODE["duplicate_tile"]
    if "out of bounds" in m:
        return CODE["out_of_bounds"]
    if "already solid" in m:
        return CODE["already_solid"]
    if "unknown material" in m:
        return CODE["unknown_material"]
    if "permeability > 0" in m:
        return CODE["material_not_solid"]
    if isinstance(ex, SealBlocked) and "standing water" in m:
        return CODE["standing_water"]
    if isinstance(ex, SealBlocked) and "sealed pocket" in m:
        return CODE["sealed_pocket"]
    if fn == "unseal" and "is not solid" in m:
        return CODE["not_solid"]
    raise AssertionError(f"probe: an unmapped host refusal {type(ex).__name__}: {m}")


class Probe:
    """One map's probe: each op run by the device on the map's planes at that
    moment, then by the host on the map itself; compared and counted."""

    def __init__(self, name, sim, seen):
        self.name, self.sim, self.seen = name, sim, seen
        self.st = _store([sim], 1)
        self.n = 0

    def op(self, op, *args, tiles=(), host_only_ok=False, tag=""):
        """``op`` a SEAM_OPS name; ``args`` its arguments (a span op: (start,
        count[, mid])); returns the host's result (None on a refusal)."""
        g = self.sim.gmap
        pre = _planes(g)
        bk0 = _books(g)
        row = [unit_kernels.SEAM_OPS[op]] + [int(a) for a in args]
        row += [0] * (8 - len(row))
        dev = _dev_planes([pre])
        tl = np.asarray(tiles, dtype=np.int64).reshape(-1, 2)
        res, books = unit_kernels.seam_probe(self.st, dev, [[row]], [tl])
        r = res[0][0]
        span = [tuple(int(v) for v in t) for t in tl[args[0]:args[0] + args[1]]] \
            if op in ("seal", "unseal", "can_seal") else None
        hcode, hval, hrem = 0, 0, None
        try:
            if op == "seal":
                g.seal_tiles(span, args[2])
            elif op == "unseal":
                g.unseal_tiles(span)
            elif op == "can_seal":
                hval = int(g.can_seal_tiles(span))
                if not hval:
                    err, _ = g._seal_blockers(g._normalize_span(span))
                    hcode = _host_code(err, "seal")
            elif op == "inject":
                g.inject_gas_n(*args[:3])
            elif op == "extract":
                hrem = [int(x) for x in g.extract_gas_n_vec(*args[:3])]
                hval = hrem[O2] + hrem[INERT_N2]
            elif op == "transfer":
                hval = int(g.transfer_gas_n(*args[:5]))
            elif op == "aperture":
                from simulation.vent_system import _aperture_blocked
                if not (0 <= args[0] < g._h and 0 <= args[1] < g._w):
                    raise ValueError("aperture: out of bounds")
                hval = int(_aperture_blocked(g, args[0], args[1]))
        except (ValueError, OverflowError) as ex:
            if op == "aperture":
                hcode = CODE["out_of_bounds"]
            else:
                hcode = _host_code(ex, op)
        post = _planes(g)
        bk1 = _books(g)
        self.n += 1
        where = f"probe {self.name} #{self.n} {op}{tuple(args)} {tag}"
        if int(r[0]) != hcode:
            fail(f"{where}: device code {int(r[0])} != host {hcode}")
            return None
        if op == "transfer" and hcode == CODE["overflow"]:
            # the reference drains the source and THEN raises (non-atomic, a
            # finding): the device refuses before any write; the host's map
            # is put back to the pre-state
            got = {n: dev[n].get()[0] for n in applier.PLANE_NAMES}
            if any(not np.array_equal(got[n], pre[n]) for n in applier.PLANE_NAMES) or                     any(int(v[0]) for v in books.values()):
                fail(f"{where}: the device's refused transfer wrote something")
            for n in applier.PLANE_NAMES:
                getattr(g, n)[...] = pre[n]
            self.seen["transfer: overflow (the reference drains first)"] += 1
            return None
        name = (applier.SEAM_REFUSAL_NAMES[hcode - 1] if hcode else "done")
        self.seen[f"{op}: {name}"] += 1
        if op in ("can_seal", "extract", "transfer", "aperture") and int(r[1]) != hval:
            fail(f"{where}: device value {int(r[1])} != host {hval}")
            return None
        if hrem is not None and [int(x) for x in r[2:2 + N_GASES]] != hrem:
            fail(f"{where}: device withdrawal {r[2:2 + N_GASES].tolist()} != host {hrem}")
            return None
        got = {n: dev[n].get()[0] for n in applier.PLANE_NAMES}
        for n in applier.PLANE_NAMES:
            fd = first_field_diff(n, got[n][None], post[n][None])
            if fd is not None:
                fail(f"{where}: plane {n} at {fd[1]}: host {fd[2]} device {fd[3]}")
                return None
        for b in applier.BOOKS:
            if b.host is None:
                continue
            if int(books[b.name][0]) != bk1[b.name] - bk0[b.name]:
                fail(f"{where}: book {b.name}: host {bk1[b.name] - bk0[b.name]} device "
                     f"{int(books[b.name][0])}")
                return None
        if hcode:
            if any(not np.array_equal(pre[n], post[n]) for n in applier.PLANE_NAMES):
                fail(f"{where}: a refused op wrote the host's planes")
            return None
        if op in ("seal", "unseal", "transfer"):
            _closure(where + " [device]", pre, got, books, mass=True)
            _closure(where + " [host]", pre, post, books, mass=True)
        elif op in ("inject", "extract"):
            _closure(where + " [device]", pre, got, books, mass=False)
        for b in ("ge_diag_seal_to", "ge_diag_seal_from", "ge_diag_unseal_from",
                  "ge_diag_unseal_to", "ge_retire", "ge_mint", "solid_seal", "solid_unseal",
                  "ge_diag_pump_transfer_to", "ge_diag_pump_transfer_from") + tuple(
                      f"trace_clip_{g}" for g in range(N_GASES)):
            if int(books[b][0]):
                self.seen[f"book {b} moved"] += 1
        return True

    def device_only(self, op, args, tiles, want):
        """An op with no host twin (span_too_long, an unsorted span): the
        device's refusal ``want`` and nothing written."""
        g = self.sim.gmap
        pre = _planes(g)
        dev = _dev_planes([pre])
        row = [unit_kernels.SEAM_OPS[op]] + [int(a) for a in args]
        row += [0] * (8 - len(row))
        res, books = unit_kernels.seam_probe(self.st, dev, [[row]],
                                             [np.asarray(tiles, dtype=np.int64).reshape(-1, 2)])
        if int(res[0][0][0]) != CODE[want]:
            fail(f"probe {self.name}: device-only {op} gave {int(res[0][0][0])}, want {want}")
        if any(not np.array_equal(dev[n].get()[0], pre[n]) for n in applier.PLANE_NAMES) or \
                any(int(v[0]) for v in books.values()):
            fail(f"probe {self.name}: a device-only refusal ({want}) wrote something")
        self.seen[f"{op}: {want} (device only)"] += 1


def _runs(g, rng, n, maxlen):
    """``n`` straight runs of open tiles (h and v, 1..maxlen), row-major."""
    h, w = g._h, g._w
    open_ = ~g.solid
    out = []
    tries = 0
    while len(out) < n and tries < 20000:
        tries += 1
        L = int(rng.integers(1, maxlen + 1))
        vert = bool(rng.integers(2))
        y, x = int(rng.integers(1, h - 1)), int(rng.integers(1, w - 1))
        span = [(y + i, x) if vert else (y, x + i) for i in range(L)]
        if all(0 <= a < h and 0 <= b < w and open_[a, b] for a, b in span):
            out.append(span)
    return out


def _boundary_runs(g, rng, n):
    """Runs of open tiles that touch the breach ring (vacuum or ambient)."""
    ring = g.is_ambient if g._boundary == "ambient" else g.is_vacuum
    h, w = g._h, g._w
    cand = []
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            if g.solid[y, x] or ring[y, x]:
                continue
            if any(ring[y + dy, x + dx] and not g.solid[y + dy, x + dx]
                   for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                cand.append((y, x))
    rng.shuffle(cand)
    return [[c] for c in cand[:n]]


def _pocket_span(g):
    """A vertical (or horizontal) 3-run whose middle tile has its other two
    neighbours solid: the middle has no receiver -- a sealed pocket."""
    h, w = g._h, g._w
    for y in range(2, h - 2):
        for x in range(2, w - 2):
            if g.solid[y, x]:
                continue
            if g.solid[y, x - 1] and g.solid[y, x + 1] and not g.solid[y - 1, x] \
                    and not g.solid[y + 1, x]:
                return [(y - 1, x), (y, x), (y + 1, x)]
            if g.solid[y - 1, x] and g.solid[y + 1, x] and not g.solid[y, x - 1] \
                    and not g.solid[y, x + 1]:
                return [(y, x - 1), (y, x), (y, x + 1)]
    return None


def _set_gas(g, y, x, slice_, value):
    """A fixture write of one gas cell, the energy reseeded on it (the
    sanctioned way a scenario builder writes bulk gas)."""
    g.gas[slice_][y, x] = int(value)
    sel = (np.array([y]), np.array([x]))
    g.reseed_gas_energy(sel)


def _probe_map(name, sim, seen, rng, ticks=30):
    for _ in range(ticks):
        sim.step()
        sim.set_paused(False)
    P = Probe(name, sim, seen)
    g = sim.gmap
    longest = max([1] + [len(d.span) for d in sim._doors])
    # every door: unseal a closed one / seal an open one, can_seal, then back
    for d in sim._doors:
        span = [tuple(t) for t in d.span]
        tl = span
        if g.solid[span[0]]:
            P.op("unseal", 0, len(span), tiles=tl, tag=f"door {d.id}")
            P.op("can_seal", 0, len(span), tiles=tl)
            P.op("seal", 0, len(span), MAT_DOOR_CLOSED, tiles=tl, tag=f"door {d.id} back")
        else:
            P.op("can_seal", 0, len(span), tiles=tl)
            P.op("seal", 0, len(span), MAT_DOOR_CLOSED, tiles=tl, tag=f"door {d.id}")
            P.op("unseal", 0, len(span), tiles=tl, tag=f"door {d.id} back")
    # constructed runs (with smoke on some, one near the trace saturation)
    for i, span in enumerate(_runs(g, rng, 10, max(6, longest))):
        if i % 3 == 0:
            y, x = span[0]
            _set_gas(g, y, x, SMOKE, int(rng.integers(1, 1 << 24)))
        if i == 1:
            y, x = span[-1]
            _set_gas(g, y, x, SMOKE, (1 << 31) - 5)
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if not g.solid[ny, nx] and (ny, nx) not in span:
                    _set_gas(g, ny, nx, SMOKE, (1 << 31) - 7)
        P.op("can_seal", 0, len(span), tiles=span)
        if P.op("seal", 0, len(span), MAT_DOOR_CLOSED, tiles=span, tag="run"):
            P.op("unseal", 0, len(span), tiles=span, tag="run back")
    for span in _boundary_runs(g, rng, 3):
        if P.op("seal", 0, len(span), MAT_DOOR_CLOSED, tiles=span, tag="by the ring"):
            P.op("unseal", 0, len(span), tiles=span, tag="by the ring back")
    # every refusal reason
    h, w = g._h, g._w
    open_t = [tuple(int(v) for v in t) for t in np.argwhere(~g.solid & ~g.is_vacuum
                                                            & ~g.is_ambient)[:50]]
    solid_t = [tuple(int(v) for v in t) for t in np.argwhere(g.solid)[:5]]
    a = open_t[len(open_t) // 2]
    P.op("seal", 0, 1, MAT_DOOR_CLOSED, tiles=[(h, 2)])
    P.op("can_seal", 0, 1, tiles=[(-1, 2)])
    P.op("seal", 0, 1, MAT_DOOR_CLOSED, tiles=[solid_t[0]])
    P.op("seal", 0, 1, 99, tiles=[a])
    P.op("seal", 0, 1, MAT_AIR, tiles=[a])
    P.op("seal", 0, 2, MAT_DOOR_CLOSED, tiles=[a, a])
    old_w = int(g.water_depth[a])
    g.water_depth[a] = 1 << 14
    P.op("seal", 0, 1, MAT_DOOR_CLOSED, tiles=[a])
    P.op("can_seal", 0, 1, tiles=[a])
    P.op("aperture", a[0], a[1])
    g.water_depth[a] = old_w
    pk = _pocket_span(g)
    if pk is not None:
        P.op("seal", 0, 3, MAT_DOOR_CLOSED, tiles=pk, tag="pocket")
        P.op("can_seal", 0, 3, tiles=pk)
    # an int32 overflow at a receiver
    for (y, x) in open_t:
        nb = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
              if not g.solid[y + dy, x + dx]]
        if nb and int(g.gas[O2][y, x]) > 0:
            r = nb[0]
            keep = int(g.gas[O2][r])
            _set_gas(g, r[0], r[1], O2, (1 << 31) - 1 - int(g.gas[O2][y, x]) + 1)
            P.op("seal", 0, 1, MAT_DOOR_CLOSED, tiles=[(y, x)], tag="overflow")
            P.op("can_seal", 0, 1, tiles=[(y, x)])
            P.op("transfer", y, x, r[0], r[1], 1 << 30, tag="transfer overflow")
            _set_gas(g, r[0], r[1], O2, keep)
            break
    P.op("unseal", 0, 1, tiles=[(-3, 4)])
    P.op("unseal", 0, 1, tiles=[a])
    P.op("unseal", 0, 2, tiles=[solid_t[0], solid_t[0]])
    P.device_only("seal", (0, unit_schema.DOOR_TILE_CAPACITY + 1, MAT_DOOR_CLOSED),
                  [open_t[i % len(open_t)] for i in range(unit_schema.DOOR_TILE_CAPACITY + 1)],
                  "span_too_long")
    if len(open_t) >= 2:
        P.device_only("unseal", (0, 2), [open_t[1], open_t[0]], "fault")
    # the gas-N primitives on the real map
    cells = open_t + [tuple(int(v) for v in t) for t in np.argwhere(g.is_vacuum)[:4]] \
        + [tuple(int(v) for v in t) for t in np.argwhere(g.is_ambient)[:4]] + solid_t[:2]
    for i, (y, x) in enumerate(cells[:40]):
        # the edges, and (every other cell) any amount: the O2 share's
        # round-half-up is seen only on an amount whose 0.21 share has a
        # fraction >= 1/2 (no power of two has one)
        dn = (int(rng.choice([0, -5, 1, 3, 77, 65536, 1 << 20, 1 << 26, 1 << 33])) if i % 2
              else int(rng.integers(1, 1 << 22)))
        P.op("inject", y, x, dn)
        P.op("extract", y, x, int(rng.choice([0, -1, 1, 999, 1 << 18, 1 << 31])))
        y2, x2 = cells[(i * 7 + 3) % len(cells)]
        P.op("transfer", y, x, y2, x2, int(rng.choice([0, 3, 1 << 16, 1 << 24])))
        P.op("aperture", y, x)
    P.op("inject", a[0], a[1], 1 << 33, tag="overflow")
    P.op("inject", -1, 0, 5)
    P.op("extract", 0, w, 5)
    P.op("transfer", 1, 1, h, 1, 5)
    P.op("transfer", a[0], a[1], a[0], a[1], 5, tag="same tile")
    P.op("aperture", h, 0)
    # synthetic: no air (trace only), negative holdings, a smoky tile
    y, x = a
    keep = [int(g.gas[s][y, x]) for s in range(N_GASES)]
    _set_gas(g, y, x, O2, 0)
    _set_gas(g, y, x, INERT_N2, 0)
    _set_gas(g, y, x, SMOKE, 12345)
    P.op("extract", y, x, 100, tag="no air")
    P.op("transfer", y, x, *open_t[0], 100, tag="no air")
    _set_gas(g, y, x, O2, -40)
    _set_gas(g, y, x, INERT_N2, 5000)
    P.op("extract", y, x, 3000, tag="negative O2")
    for s in range(N_GASES):
        _set_gas(g, y, x, s, keep[s])
    return P.n


def part_probe():
    seen = Counter()
    rng = np.random.default_rng(6101)
    n = 0
    for name in ("test_level", "test_arena", "door_test", "airlock_demo", "wind_studio"):
        n += _probe_map(name, LevelDriver(name).sim, seen, rng)
    n += _probe_map("door scene", DoorScene(0).sim, seen, rng, ticks=5)
    n += _probe_map("door scene (planetside)", DoorScene(1).sim, seen, rng, ticks=5)
    need = [f"seal: {r}" for r in ("done", "out_of_bounds", "already_solid", "unknown_material",
                                   "material_not_solid", "standing_water", "sealed_pocket",
                                   "overflow", "duplicate_tile")]
    need += [f"unseal: {r}" for r in ("done", "out_of_bounds", "not_solid", "duplicate_tile")]
    need += ["can_seal: done", "can_seal: out_of_bounds", "can_seal: standing_water",
             "can_seal: sealed_pocket", "can_seal: overflow",
             "inject: done", "inject: overflow", "inject: out_of_bounds", "extract: done",
             "extract: out_of_bounds", "transfer: done",
             "transfer: out_of_bounds", "aperture: done", "aperture: out_of_bounds",
             "transfer: overflow (the reference drains first)",
             "seal: span_too_long (device only)", "unseal: fault (device only)",
             "book ge_diag_seal_to moved", "book ge_retire moved",
             "book solid_seal moved", "book solid_unseal moved",
             f"book trace_clip_{SMOKE} moved"]
    for k in need:
        if not seen[k]:
            fail(f"probe: the case {k!r} was never taken")
    print(f"probe: {n} ops on 7 maps, device == host (planes, books, closure) on every "
          f"one; cases: " + ", ".join(f"{k} {v}" for k, v in sorted(seen.items())))


# ---------------------------------------------------------------------------
# The census of the sweep's branches (the host's own decisions)
# ---------------------------------------------------------------------------
class Census:
    """Wraps (reads only) ``door_system.sweep_doors``, the seam's seal /
    unseal and ``Simulation._direct_use`` and counts each branch taken."""

    def __init__(self):
        self.c = Counter()
        self._orig = []

    def install(self):
        from simulation import door_system
        from simulation import simulation as simmod
        from simulation.gamemap import GameMap
        from simulation.simulation import Simulation
        c = self.c
        o_sw = simmod.sweep_doors

        def sw(sim, __o=o_sw):
            g = sim.gmap
            changes = []
            for d in sim._doors:
                if not d.alive:
                    c["sweep: a dead door skipped"] += 1
                    continue
                if d.state == door_system.DOOR_CLOSED:
                    bad = [t for t in d.span if int(g.material[t]) != MAT_DOOR_CLOSED]
                    if bad:
                        rest = len(d.span) - len(bad)
                        c["sweep: a closed door destroyed (the whole-door rule)"] += 1
                        c["sweep: its remaining tiles destroyed" if rest
                          else "sweep: every tile already gone"] += 1
                        changes.append(d)
                        continue
                    if d.want_open:
                        c["sweep: closed + want_open: opens"] += 1
                        changes.append(d)
                        if [int(g.wall_hp[t]) for t in d.span] != list(d.hp):
                            c["sweep: an open copies an hp the door did not hold"] += 1
                    else:
                        c["sweep: closed, latch closed: idle"] += 1
                elif d.want_open:
                    c["sweep: open, latch open: idle"] += 1
                else:
                    clear = door_system._occupancy_clear(sim.units, d._span_set)
                    if not clear:
                        c["sweep: close blocked by a living unit in the span"] += 1
                    elif not g.can_seal_tiles(d.span):
                        c["sweep: close refused by can_seal"] += 1
                    else:
                        c["sweep: open + latch closed: closes"] += 1
                        changes.append(d)
                        from simulation import wall_fixed
                        full = int(wall_fixed.quantize_scalar(
                            float(g.materials.hp[MAT_DOOR_CLOSED])))
                        if any(int(v) != full for v in d.hp):
                            c["sweep: a close restamps an hp unlike the table's"] += 1
                        if any(not u.alive and any((ty, tx) in d._span_set
                                                   for tx, ty in u.occupied_tiles())
                               for u in sim.units):
                            c["sweep: closes over a corpse"] += 1
            if len(changes) >= 2:
                c["sweep: two doors change on one tick"] += 1
            return __o(sim)
        simmod.sweep_doors = sw
        self._orig.append((simmod, "sweep_doors", o_sw))

        o_seal, o_unseal = GameMap.seal_tiles, GameMap.unseal_tiles

        def seal(g, tiles, mid, __o=o_seal):
            span = g._normalize_span(tiles)
            for t in span:
                rs = g._seal_receivers(t[0], t[1], set(span))
                if not rs:
                    c["seal: a tile with no receiver (gas-free)"] += 1
                if any(g.is_vacuum[r] or g.is_ambient[r] for r in rs):
                    c["seal: a receiver on the breach ring"] += 1
                if any(g.thermal_solid[t[0] + dy, t[1] + dx]
                       for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))):
                    c["seal: a tile takes the close-T mean"] += 1
                else:
                    c["seal: a tile keeps its air T (no thermal neighbour)"] += 1
            return __o(g, tiles, mid)

        def unseal(g, tiles, __o=o_unseal):
            span = g._normalize_span(tiles)
            ring = g.is_ambient if g._boundary == "ambient" else g.is_vacuum
            for t in span:
                nb = [(t[0] + dy, t[1] + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))]
                if ring[t] or any(ring[n] and not g.solid[n] for n in nb):
                    c["unseal: a tile joins the boundary"] += 1
                elif not any(not g.solid[n] and not g.is_vacuum[n] and n not in span
                             for n in nb):
                    c["unseal: a tile opens empty (no donor)"] += 1
                else:
                    c["unseal: a tile seeded from its donors"] += 1
            return __o(g, tiles)
        GameMap.seal_tiles, GameMap.unseal_tiles = seal, unseal
        self._orig += [(GameMap, "seal_tiles", o_seal), (GameMap, "unseal_tiles", o_unseal)]

        o_use = Simulation._direct_use
        per_tick = {}

        def use(sim, u, __o=o_use):
            from simulation import verb_gates
            d = verb_gates.use_door(sim, u)
            if d is not None:
                key = (id(sim), sim.tick, id(d))
                per_tick[key] = per_tick.get(key, 0) + 1
                c["use: a door's latch flipped"] += 1
                if per_tick[key] == 2:
                    c["use: one door used twice in one tick"] += 1
            return __o(sim, u)
        Simulation._direct_use = use
        self._orig.append((Simulation, "_direct_use", o_use))
        return self

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()


NEED = (
    "sweep: a dead door skipped", "sweep: a closed door destroyed (the whole-door rule)",
    "sweep: its remaining tiles destroyed", "sweep: every tile already gone",
    "sweep: closed + want_open: opens", "sweep: closed, latch closed: idle",
    "sweep: open, latch open: idle", "sweep: close blocked by a living unit in the span",
    "sweep: close refused by can_seal", "sweep: open + latch closed: closes",
    "sweep: closes over a corpse", "sweep: two doors change on one tick",
    "sweep: an open copies an hp the door did not hold",
    "sweep: a close restamps an hp unlike the table's",
    "seal: a tile with no receiver (gas-free)", "seal: a receiver on the breach ring",
    "seal: a tile takes the close-T mean",
    "seal: a tile keeps its air T (no thermal neighbour)",
    "unseal: a tile joins the boundary", "unseal: a tile opens empty (no donor)",
    "unseal: a tile seeded from its donors",
    "use: a door's latch flipped", "use: one door used twice in one tick",
)
UNREACHABLE = {
    "sweep: a seal / unseal raising inside the sweep":
        "the sweep composes can_seal_tiles before seal_tiles and unseals only a "
        "closed door's own span (solid by construction); PART PROBE takes every "
        "refusal of the primitives",
    "sweep: a destroy on an open door's tile reconciled":
        "the whole-door rule observes CLOSED doors only: a destroy_wall on an "
        "open door's air tile is a no-op (the scene does one at k 58)",
}


# ---------------------------------------------------------------------------
# PART SCENE
# ---------------------------------------------------------------------------
def _identity(sim, where):
    from simulation.entities import door as door_schema
    recs = unit_kernels.doors_of(sim)
    tiles = unit_kernels.door_tiles_of(sim)
    back = unit_kernels.door_objects(recs, tiles)
    if len(back) != len(sim._doors):
        fail(f"identity {where}: {len(back)} rows for {len(sim._doors)} doors")
        return False
    for k, (d, b) in enumerate(zip(sim._doors, back)):
        a = (int(d.ordinal), bool(d.alive), int(d.state), bool(d.want_open),
             [tuple(t) for t in d.span], [int(v) for v in d.hp])
        if a != (b.ordinal, b.alive, b.state, b.want_open, b.span, b.hp):
            fail(f"identity {where}: door {d.id}: unpack(pack) differs")
            return False
        want = (("alive", 1 if d.alive else 0),) + door_schema.door.runtime_digest_rows(d)
        if unit_kernels.door_entity_rows(recs[k], tiles) != want:
            fail(f"identity {where}: door {d.id}: the table's entity rows are not the "
                 f"serializer's")
            return False
    return True


def run_shadow(name, drivers, ticks, census=None, twins=None, slots=None, n_units=None):
    slots = slots or SLOTS
    acts = Counter()
    n = 0
    t0 = time.perf_counter()
    with ShadowLockstep(drivers, list(slots), n_units=n_units) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            reps = ls.tick()
            n += 1
            for lab, rep in reps.items():
                if not rep.equal:
                    fail(f"{name}: {ls.divergences[-1]}")
                    return acts, n
            r9 = reps.get(SLOT_9ED.key)
            for e in range(len(drivers)):
                if r9 is not None and int(r9.books["applied"][e]):
                    acts[SLOT_9ED.key] += 1
                r1 = reps.get(SLOT_1B.key)
                if r1 is not None and any(r[0] == unit_requests.KIND_USE
                                          for r in r1.host_requests[e]):
                    acts[SLOT_1B.key] += 1
            for e, d in enumerate(drivers):
                if not _identity(d.sim, f"{name} tick {d.sim.tick} env {e}"):
                    return acts, n
            if twins is not None:
                from cuda_unit_verbs_check import _same
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked match diverged from the unhooked one "
                             f"at tick {d.sim.tick}")
                        return acts, n
    print(f"{name}: N={len(drivers)}, {n} ticks, both slots equal every tick (tables, "
          f"planes, books, write set){' (env i == env i alone)' if len(drivers) > 1 else ''}"
          f"{' (the unhooked twin equal)' if twins is not None else ''}; env-ticks acted: "
          f"{dict(acts)} ({time.perf_counter() - t0:.0f} s)")
    return acts, n


def part_scene():
    census = Census().install()
    try:
        acts, _ = run_shadow("scene", [DoorScene(0)], SCENE_TICKS, twins=[DoorScene(0)])
        run_shadow("scene planetside", [DoorScene(1)], SCENE_TICKS)
    finally:
        census.uninstall()
    c = census.c
    for b in NEED:
        if not c[b]:
            fail(f"scene: the census branch {b!r} was never taken")
    for s in SLOTS:
        if not acts[s.key]:
            fail(f"scene: slot {s.key!r} never acted (vacuous)")
    print("scene census: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    print("scene unreachable: " + "; ".join(f"{k} -- {v}" for k, v in UNREACHABLE.items()))


# ---------------------------------------------------------------------------
# PART BATCH
# ---------------------------------------------------------------------------
def _batch_drivers(n):
    """A batch shares one grid (the planes are (N, h, w)): the door scene's
    variants -- space and planetside, two layouts (D8's wall), 7 to 9 doors,
    their own seeds and schedules (so their fields differ)."""
    if n == 2:
        return [DoorScene(1), DoorScene(2)]
    return [DoorScene(v) for v in range(n)]


def part_batch():
    for n in BATCH_NS:
        drivers = _batch_drivers(n)
        acts, _ = run_shadow(f"batch N={n}", drivers, BATCH_TICKS)
        for s in SLOTS:
            if not acts[s.key]:
                fail(f"batch N={n}: slot {s.key!r} never acted (vacuous)")
        counts = sorted({len(d.sim._doors) for d in drivers})
        if len(counts) < 2:
            fail(f"batch N={n}: every env holds {counts} doors (vacuous)")


def part_levels():
    """The shipped levels with door entities, each its own N = 1 shadow
    (their grids differ): test_arena's bots match (the bots' use, the
    zombies' bashing), airlock_demo (the signal bus drives its doors' latch
    at 9e(c), read here as the host left it), test_level, door_test."""
    for name, match in (("test_arena", True), ("airlock_demo", False),
                        ("test_level", False), ("door_test", False)):
        d = LevelDriver(name, seed=1, match=match)
        run_shadow(f"level {name} ({len(d.sim._doors)} doors, {len(d.sim.units)} units)",
                   [d], BATCH_TICKS, n_units=max(1, len(d.sim.units)))


# ---------------------------------------------------------------------------
# PART PADDING
# ---------------------------------------------------------------------------
def part_padding():
    scs = [DoorScene(0), DoorScene(1)]
    for sc in scs:
        for _ in range(3):
            sc.tick()
    for d in scs[0].sim._doors[:3]:
        d.want_open = not d.want_open          # a change on the run
    sims = [s.sim for s in scs]
    st = _store(sims)

    def run(dirty):
        dt, tt = unit_kernels.door_tables(2)
        rd = np.zeros(dt.shape, dtype=np.int64)
        rt = np.zeros(tt.shape, dtype=np.int64)
        nd, nt = [], []
        for e, sim in enumerate(sims):
            a, b = unit_kernels.doors_of(sim), unit_kernels.door_tiles_of(sim)
            rd[e, :len(a)] = a
            rt[e, :len(b)] = b
            nd.append(len(a))
            nt.append(len(b))
            if dirty:
                junk = a[0].copy()
                junk[DLAY.column("want_open").offset] = 1 - junk[DLAY.column("want_open").offset]
                junk[DLAY.column("alive").offset] = 1
                rd[e, len(a):] = junk
                rt[e, len(b):] = b[0]
        dt.upload(rd, np.array(nd))
        tt.upload(rt, np.array(nt))
        fields = _dev_planes([_planes(s.gmap) for s in sims])
        unit_kernels.sweep_doors(st, _applier_for(st), fields, dt, tt)
        return dt.download(), tt.download(), {k: v.get() for k, v in fields.items()}, nd, nt, rd, rt
    clean, dirty = run(False), run(True)
    for e in range(2):
        nd, nt = clean[3][e], clean[4][e]
        if not np.array_equal(clean[0][0][e, :nd], dirty[0][0][e, :nd]) or \
                not np.array_equal(clean[1][0][e, :nt], dirty[1][0][e, :nt]):
            fail(f"padding env {e}: garbage past a table's count changed the valid rows")
        if not np.array_equal(dirty[0][0][e, nd:], dirty[5][e, nd:]) or \
                not np.array_equal(dirty[1][0][e, nt:], dirty[6][e, nt:]):
            fail(f"padding env {e}: a padding row was written")
        for k in applier.PLANE_NAMES:
            if not np.array_equal(clean[2][k][e], dirty[2][k][e]):
                fail(f"padding env {e}: garbage rows changed plane {k}")
                break
    if clean[3][0] == clean[3][1]:
        fail("padding: the two envs hold the same door count (vacuous)")
    print(f"padding: garbage door / tile rows past each table's count (door counts "
          f"{clean[3]}, tile counts {clean[4]}) neither read nor written")


# ---------------------------------------------------------------------------
# PART OVERFLOW
# ---------------------------------------------------------------------------
def part_overflow():
    sim = DoorScene(0).sim
    seen = Counter()
    recs, tiles = unit_kernels.doors_of(sim), unit_kernels.door_tiles_of(sim)
    for name, table, rows in (("door", DT, recs), ("door_tile", TT, tiles)):
        small = ot.DeviceTable(table, 2, cap=len(rows) - 1)
        try:
            small.stage_lists([rows[:1], rows])
            fail(f"overflow: a full {name} table took {len(rows)} rows")
        except ot.ObjectTableOverflow as ex:
            if f"'{name}'" not in str(ex) or "env 1" not in str(ex):
                fail(f"overflow: the {name} refusal does not name the table and the env: {ex}")
            seen[name] += 1
    many = [type("D", (), {"ordinal": i, "alive": True, "state": 0, "want_open": False,
                           "span": [(1, j) for j in range(9)], "hp": [ONE] * 9})()
            for i in range(unit_schema.DOOR_CAPACITY + 1)]
    try:
        ot.pack(DT, many)
        fail("overflow: a door list past the table's capacity was packed")
    except ot.ObjectTableOverflow as ex:
        if "'door'" not in str(ex):
            fail(f"overflow: the door table's refusal does not name it: {ex}")
        seen["door list"] += 1
    try:
        ot.pack_sides(DT, many[:unit_schema.DOOR_CAPACITY])
        fail("overflow: a tile side table past its capacity was packed")
    except ot.ObjectTableOverflow as ex:
        if "'door_tile'" not in str(ex):
            fail(f"overflow: the tile table's refusal does not name it: {ex}")
        seen["tile list"] += 1
    big = many[0]
    big.span = [(1, j) for j in range(unit_schema.DOOR_TILE_CAPACITY + 1)]
    big.hp = [ONE] * len(big.span)
    try:
        ot.records(DT, [big])
        fail("overflow: a door span past its bound was encoded")
    except ValueError as ex:
        if "tiles" not in str(ex):
            fail(f"overflow: the span refusal does not name the member: {ex}")
        seen["span"] += 1
    print(f"overflow: the door table, the tile side table and one door's span past "
          f"their capacities refused by name {dict(seen)}")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
_CH_REQ = {}


def _run_1b_chained(st, inp, lists):
    """Slot 1b + the use write in a chain: the request list is the chain's own
    (a USE lands as a door-table write, never through apply_to_host); a throw
    or a plant has no landing here (the scene makes none) and is refused."""
    key = st.shape
    if key not in _CH_REQ:
        _CH_REQ[key] = unit_requests.RequestList(st.shape[0], st.shape[1])
    req = _CH_REQ[key]
    req.begin()
    from cuda_unit_lockstep import flight_fields       # #215: the flight plane
    cov = unit_kernels.verbs(st, req, inp.tick_row, inp.tables["door_spans"],
                             inp.scalars["charges_live"], flight_fields(inp, st))
    if any(r[0] != unit_requests.KIND_USE for r in req.records()[0]):
        raise RuntimeError("chained: a throw / plant in the door chain (no landing)")
    unit_kernels.door_uses(st, req, lists["doors"])
    return cov


SLOT_1B_CH = DeviceSlot(label="1b intents", run=_run_1b_chained,
                        inputs=SLOT_1B.inputs, outputs=SLOT_1B.outputs)


def _host_side_9ed(ch):
    """Slot 9e(d)'s host-only side after the device ran it: the books landed
    (the solid books through _solid_energy_book, so the solver's sink moves),
    the render marks and the sky flag of every tile whose material changed,
    the serial, and the render-only DoorDestroyedEvents of the doors that
    died (each tile still closed at the entry, door then span order)."""
    from simulation.events import DoorDestroyedEvent
    ge = {"ge_retire": "retire", "ge_mint": "mint", "ge_destroy_seed": "destroy_seed",
          "ge_seal": "seal", "ge_diag_seal_from": "diag_seal_from_nonacct",
          "ge_diag_seal_to": "diag_seal_to_nonacct", "ge_unseal": "unseal",
          "ge_diag_unseal_from": "diag_unseal_from_nonacct",
          "ge_diag_unseal_to": "diag_unseal_to_nonacct"}

    def hs(sim, _pre, _post):
        g = sim.gmap
        b = ch.books
        g.n_destruction_seed_sum += b["n_destruction_seed"]
        for name, key in ge.items():
            g._gas_energy_book(key, b[name])
        for name, key in (("solid_seal", "seal"), ("solid_unseal", "unseal"),
                          ("solid_destroy_wall", "destroy_wall")):
            g._solid_energy_book(key, b[name])
        for s in range(N_GASES):
            g.trace_clip_books[s] += b[f"trace_clip_{s}"]
        mat0 = ch.inputs._host_fields["material"][0]
        changed = np.argwhere(mat0 != g.material)
        for (y, x) in changed.tolist():
            g.render_dirty.add((int(y), int(x)))
        if len(changed):
            g._sky_mask_dirty = True
        g._tile_serial += b["tiles_changed"]
        doors0 = ch.inputs.lists["doors"][0]
        tiles0 = ch.inputs.lists["door_tiles"][0]
        alive = DLAY.column("alive").offset
        for k, d in enumerate(sim._doors):
            if int(doors0[k][alive]) and not d.alive:
                a = int(doors0[k][DLAY.column("tile_start").offset])
                for i in range(int(doors0[k][DLAY.column("tile_count").offset])):
                    y, x = int(tiles0[a + i][0]), int(tiles0[a + i][1])
                    if int(mat0[y, x]) == MAT_DOOR_CLOSED:
                        sim.tick_events.append(DoorDestroyedEvent(pos=(y, x)))
    return hs


def part_chained():
    from cuda_unit_topology_check import _same_twin
    from simulation.entities.serialize import entity_records
    from cuda_unit_verbs_check import _books_host_side
    from simulation import simulation as simmod
    from simulation.simulation import Simulation
    for v in (0, 1):
        twin, ref = DoorScene(v), DoorScene(v)
        host_calls = Counter()
        o_sw, o_use = simmod.sweep_doors, Simulation._direct_use

        def sw_count(sim, __o=o_sw):
            host_calls["twin sweep" if sim is twin.sim else "reference sweep"] += 1
            return __o(sim)

        def use_count(sim, u, __o=o_use):
            host_calls["twin use" if sim is twin.sim else "reference use"] += 1
            return __o(sim, u)
        simmod.sweep_doors, Simulation._direct_use = sw_count, use_count
        t0 = time.perf_counter()
        n = acted = 0
        holder = {}
        ch1 = ChainedSlots(twin, [SLOT_1B_CH], host_side=_books_host_side, bp=bp)
        ch9 = ChainedSlots(twin, [SLOT_9ED], bp=bp,
                           host_side=lambda s, a, b: holder["hs"](s, a, b))
        holder["hs"] = _host_side_9ed(ch9)
        with ch1, ch9:
            for _ in range(CHAIN_TICKS):
                twin.tick()
                ref.tick()
                n += 1
                why = _same_twin(twin.sim, ref.sim)
                if why is None and unit_kernels.doors_of(twin.sim).tolist() != \
                        unit_kernels.doors_of(ref.sim).tolist():
                    why = "the door table"
                if why is None and unit_kernels.door_tiles_of(twin.sim).tolist() != \
                        unit_kernels.door_tiles_of(ref.sim).tolist():
                    why = "the tile table"
                if why is None and entity_records(twin.sim.entities) != \
                        entity_records(ref.sim.entities):
                    why = "the entity section"
                if why is not None:
                    fail(f"chained v{v}: the twin left the reference at tick "
                         f"{twin.sim.tick} ({why})")
                    break
                acted += int(ch9.books.get("applied", 0) > 0) if ch9.books else 0
        simmod.sweep_doors, Simulation._direct_use = o_sw, o_use
        if host_calls["twin sweep"] or host_calls["twin use"] or                 not host_calls["reference sweep"]:
            fail(f"chained v{v}: the host law ran in the twin {dict(host_calls)}")
        print(f"chained v{v} (1b + use write; 9e(d)): {n} ticks, the twin's unit hash, "
              f"field digest, every plane, the books, the door and tile tables and the "
              f"entity section equal the reference every tick ({ch1.runs} + {ch9.runs} "
              f"chain runs, the sweep acted on {acted} ticks; host law calls "
              f"{dict(host_calls)}; "
              f"{time.perf_counter() - t0:.0f} s)")
        if ch1.runs != n or ch9.runs != n or acted == 0:
            fail(f"chained v{v}: {ch1.runs} / {ch9.runs} runs in {n} ticks, {acted} acted "
                 f"(vacuous or skipped)")


# ---------------------------------------------------------------------------
# PART SHIP
# ---------------------------------------------------------------------------
def part_ship():
    from cuda_unit_objects_check import ShipDriver
    acts, n = run_shadow("ship", [ShipDriver()], SHIP_TICKS, twins=[ShipDriver()])
    print(f"ship: the slots acted on {dict(acts)} of {n} ticks (ship_demo has no "
          f"door entity: the sweep is gated off, compared as a run that changes nothing)")


# ---------------------------------------------------------------------------
PARTS = ("tables", "probe", "scene", "batch", "levels", "padding", "overflow", "chained",
         "ship")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_DOORS_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_DOORS_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "probe": part_probe, "scene": part_scene,
             "batch": part_batch, "levels": part_levels, "padding": part_padding, "overflow": part_overflow,
             "chained": part_chained, "ship": part_ship}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
        sys.stdout.flush()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_DOORS_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table (median ms over ``reps``, N = 1 / 8 / 64 copies of the
    door scene): the sweep idle (every latch agrees with its door), the sweep
    with doors changing (four doors flip: two open, two close), and one seal +
    one unseal of a 3-tile span through the probe; each call's launch, its
    books' D2H and the Python face (each rep re-stages, not timed)."""
    sc = DoorScene(0)
    for _ in range(4):
        sc.tick()
    sim = sc.sim
    print("TIMING (median ms over %d reps)" % reps)
    print("| op | N=1 | N=8 | N=64 |")
    res = {}
    for n in (1, 8, 64):
        sims = [sim] * n
        st = _store(sims)
        dt, tt = unit_kernels.door_tables(n)
        planes = _planes(sim.gmap)
        d0, t0_ = unit_kernels.doors_of(sim), unit_kernels.door_tiles_of(sim)
        flip = d0.copy()
        wo = DLAY.column("want_open").offset
        for k in (0, 1, 2, 3):
            flip[k, wo] = 1 - flip[k, wo]
        out = []
        for rows in (d0, flip):
            ts = []
            for _ in range(reps):
                dt.stage_lists([rows] * n)
                tt.stage_lists([t0_] * n)
                fields = _dev_planes([planes] * n)
                s = time.perf_counter()
                unit_kernels.sweep_doors(st, _applier_for(st), fields, dt, tt)
                ts.append(time.perf_counter() - s)
            out.append(float(np.median(ts)))
        span = [tuple(t) for t in sim._doors[1].span]          # D1: open at the start
        ops = [[unit_kernels.SEAM_OPS["seal"], 0, 3, MAT_DOOR_CLOSED, 0, 0, 0, 0],
               [unit_kernels.SEAM_OPS["unseal"], 0, 3, 0, 0, 0, 0, 0]]
        ts = []
        for _ in range(reps):
            fields = _dev_planes([planes] * n)
            s = time.perf_counter()
            unit_kernels.seam_probe(st, fields, [ops] * n, [span] * n)
            ts.append(time.perf_counter() - s)
        out.append(float(np.median(ts)))
        res[n] = out
    for i, name in enumerate(("9e(d) sweep, idle (8 doors)", "9e(d) sweep, 4 doors change",
                              "seal + unseal of a 3-tile span (probe)")):
        print(f"| {name} | " + " | ".join(f"{res[n][i] * 1e3:.3f}" for n in (1, 8, 64)) + " |")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        sys.exit(bench())
    sys.exit(main())
