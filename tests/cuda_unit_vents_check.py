"""#29 P6b.2 -- THE VENTS GATE: the duct table and the vent table (the second
and third entity tables), ``inject_gas_n_vec`` in the Device topology seam, the
kit's floor division and floor product, and slot 9e(d)'s vent circulation as
the applier's VENTS stage, each against its Python law at tol 0 (design 4.4c,
the cut table's P6b.2 row).

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_vents.py). The tables are ``simulation.object_tables.
DeviceTable`` over ``unit_schema.DUCT_TABLE`` / ``VENT_TABLE`` (the harness
lists ``ducts`` / ``vents``, record functions ``unit_kernels.ducts_of`` /
``vents_of``); the kernel ``unit_kernels.sweep_vents`` (``cpp/src/
cuda_applier.cu`` over ``cpp/src/cuda_unit_vents.cuh`` and ``cpp/src/
cuda_topology.cuh``) and the TEST-ONLY ``unit_kernels.seam_probe`` (ops
``inject_vec``, ``floordiv``, ``muldiv``).

THE GATE'S CONFIG: the ``[filters.derelict]`` row is given fractional
efficiencies for the whole process (``main``), a config value both sides read
(the duct runtime's ``filter_eff_q`` and the device tables' row), so the
scrub's truncation is taken: the shipped rows are all 0.0 / 1.0.

PART TABLES -- every P6b.2 device table row equals an independent read of its
reference value; the duct / vent tables' fingerprint is compiled in.
PART KIT -- ``floordiv_q`` (negative, zero, positive operands of both signs)
and ``muldiv_floor_q`` (products past 2^64, refusals) on the device against
Python's ``//``.
PART PROBE -- ``inject_gas_n_vec`` on real maps with real gas (ship_demo,
playground, fire_smoke_studio, test_level, both vent-scene boundaries) and on
synthetic states: random vectors and temperatures (both sides of every
floor), each rail, accountable and non-accountable targets, a bulk-free
result, a zero vector, the trace saturation, every refusal (bounds, overflow;
device-only faults) with nothing written; planes, books and the energy / mass
closure equal.
PART SCENE -- the SHADOW LOCKSTEP of slot 1b, 9e(d) vents and 9e(d) doors on
the VENT SCENE (six ducts, sixteen vents, a door over a vent): every tick the
tables, every plane, the books and the write set equal; the tables' IDENTITY
and entity rows every tick; a CENSUS of the sweep's branches, each required,
the unreachable ones named; the hooked match equal to an unhooked twin; the
ledgers' growth printed.
PART LEVELS -- ship_demo's bots match (its duct runs), playground,
fire_smoke_studio, airlock_demo (no duct: the sweep gated off).
PART REFUSAL -- each ledger family one past its proved range (both sides of
a signed one): refused by name, nothing of the duct written; at the bound:
run and equal to the host.
PART BATCH -- N = 2 and 8 of vent-scene variants that differ in duct count,
vent count, layout, boundary, seed and fields: env i == env i alone.
PART PADDING / OVERFLOW -- garbage past each table's count neither read nor
written; each table past its capacity refused by name.
PART CHAINED -- 9e(d) vents and 9e(d) doors replaced by the device's in twin
matches (the vent scene, both boundaries; the door scene), the tables landed
back on the runtimes: unit hash, field digest, planes, books, tables, entity
section equal every tick, the host laws run 0 times in the twin; ship_demo's
bots match, 500 ticks, with the device vent sweep.

PROPERTY: the duct / vent tables are the sim's ducts and vents; the device
inject_gas_n_vec is GameMap's integer for integer, refusals included; the
device sweep is ``vent_system.sweep_vents`` on the tables (duct order, returns
then supplies, the guard, the accrual, the filter, the wipe, the split, the
deposit and its rails, the debit); a duct past the proved range is refused by
name with nothing written; in a batch as alone. BREAKS IF any of those differs
from the law.

Prints per-part lines and ``UNIT_VENTS_RESULT: PASS`` / ``FAIL``; exits 0 /
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
                        unit_schema, unit_store, unit_tables)
from simulation.gases import (FUEL_GAS, INERT_N2, N_GASES, O2, POISON,  # noqa: E402
                              SMOKE, STEAM, TEARGAS)
from simulation.materials import MAT_AIR, MAT_FOLIAGE, MAT_HULL  # noqa: E402

FAILS: list = []
ONE = 65536
DT = unit_schema.DUCT_TABLE
VT = unit_schema.VENT_TABLE
DLAY = ot.table_layout(DT)
VLAY = ot.table_layout(VT)
SCENE_TICKS = int(os.environ.get("UNIT_VENTS_SCENE_TICKS", "130"))
BATCH_TICKS = int(os.environ.get("UNIT_VENTS_BATCH_TICKS", "90"))
LEVEL_TICKS = int(os.environ.get("UNIT_VENTS_LEVEL_TICKS", "120"))
SHIP_TICKS = int(os.environ.get("UNIT_VENTS_SHIP_TICKS", "500"))
CHAIN_TICKS = int(os.environ.get("UNIT_VENTS_CHAIN_TICKS", "130"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_VENTS_BATCH_NS", "2,8").split(",")]
SPACE = level_loader.SPACE_CODE
CODE = {n: i + 1 for i, n in enumerate(applier.SEAM_REFUSAL_NAMES)}
TRACE_CELL_MAX = (1 << 31) - 1
#: The gate's fractional [filters.derelict] (see the module doc).
GATE_DERELICT = {"steam": 0.5, "smoke": 0.37, "poison": 0.999, "teargas": 0.25,
                 "fuel_gas": 0.0}


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


def gate_config():
    """THE GATE'S CONFIG: fractional [filters.derelict] for the process (before
    any sim is built; both sides read it)."""
    from config import CFG
    for k, v in GATE_DERELICT.items():
        setattr(CFG.filters.derelict, k, v)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
PLANE_OUTPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)
VENT_BOOK_NAMES = ("ge_pump_extract", "ge_pump_inject_vec", "ge_pump_rail") + tuple(
    f"trace_clip_{g}" for g in range(N_GASES))
VENT_BOOKS = tuple("book:" + b for b in VENT_BOOK_NAMES)
HOST_ONLY = (
    ("_gas_t_amb_raw_cache", "a lazily filled cache of a config constant (the "
     "device reads the table row topo_t_amb_q)"),
    ("smoke", "GameMap.smoke IS a view of gas[SMOKE] (the same memory): compared "
     "through field:gas"),
)
_APPLIERS: dict = {}


def _applier_for(store):
    n = store.shape[0]
    if n not in _APPLIERS:
        _APPLIERS[n] = applier.DeviceApplier(n, 4, bp=bp)
    return _APPLIERS[n]


def _run_vents(st, inp, lists):
    return unit_kernels.sweep_vents(st, _applier_for(st), inp.fields, lists["ducts"],
                                    lists["vents"])


SLOT_VENTS = DeviceSlot(label="9e(d) vents", run=_run_vents, columns=("id",), skip_ok=True,
                        inputs=("list:ducts", "list:vents"),
                        outputs=PLANE_OUTPUTS + VENT_BOOKS + ("list:ducts", "list:vents"),
                        host_only=HOST_ONLY)


def _door_slots():
    from cuda_unit_doors_check import SLOT_1B, SLOT_9ED
    return SLOT_1B, SLOT_9ED


def _slots():
    s1b, s9 = _door_slots()
    return [s1b, SLOT_VENTS, s9]


# ---------------------------------------------------------------------------
# The reference values of the P6b.2 table rows (cuda_unit_kernels_check's
# every-row gate reads this too)
# ---------------------------------------------------------------------------
def _q16(x):
    """An independent round-half-away of x * 2^16."""
    import math
    y = float(x) * ONE
    return int(math.copysign(math.floor(abs(y) + 0.5), y))


def table_reference(sim) -> dict:
    """An INDEPENDENT read of every P6b.2 table row: config.toml's
    [filters.*] (sorted by name), the gas ids, the T rails' config, N_EPS."""
    from config import CFG
    from simulation import vent_system
    names = sorted(vars(CFG.filters).keys())
    eff = tuple(_q16(getattr(getattr(CFG.filters, n), g))
                for n in names for g in ("steam", "smoke", "poison", "teargas", "fuel_gas"))
    return {
        "vent_filter_eff_q": eff,
        "vent_trace_ids": (STEAM, SMOKE, POISON, TEARGAS, FUEL_GAS),
        "vent_t_min_q": _q16(CFG.physics.eos.T_MIN),
        "vent_t_max_q": _q16(CFG.physics.thermal.T_MAX_PHYS),
        "vent_n_eps": int(vent_system.N_EPS),
    }


# ---------------------------------------------------------------------------
# The vent scene
# ---------------------------------------------------------------------------
def _entity(eid, cls, ordinal, **fields):
    from level_loader import EntityInstance
    from simulation.entities import door as door_mod, vents as vent_mod
    klass = {"door": door_mod.door, "duct": vent_mod.duct, "vent": vent_mod.vent}[cls]
    f = {x.name: x.default for x in klass.FIELDS}
    f.update(fields)
    return EntityInstance(id=eid, class_name=cls, ordinal=ordinal, fields=f)


def _vent(eid, ordinal, x, y, duct, role, q=2.0, mount="floor", facing="n"):
    return _entity(eid, "vent", ordinal, x=x, y=y, duct=duct, role=role, q_circ=float(q),
                   mount=mount, facing=facing)


#: Pocket tiles (x, y): one-tile rooms walled by hull.
POCKETS = ((15, 35), (25, 35), (5, 35))


def _vent_level(variant):
    """1 m tiles, 40 x 64, space past the east hull (x 63). Rooms A (x 1..20,
    y 1..17) | the wall x 21 with the door DV (y 8..10) | B (x 22..40) | the
    hull x 41 | C (x 42..61); under the hull rows y 18..19, D (x 1..30, y
    20..38) | the hull x 31 | E (x 32..61). Three one-tile pockets in D, a
    foliage tile in D. Ducts (ordinal != row): d1 (10, derelict: C -> A,
    its supply on d0's return tile), d3 (11, hepa: returns only, one through
    the hull x 41), d0 (12, hepa: A -> B and C and the door's middle tile),
    d4 (13: supplies only, into E and pocket 2), d2 (14, derelict: D's cold
    air, the space-facing wall vent, pocket 1 -> E and the foliage tile), d5
    (15: one supply into pocket 3). Two unwired vents. Odd variants:
    planetside; variants 2, 3, 6, 7: d5 dropped and a wall across E (another
    duct / vent count, another layout)."""
    from level_loader import LevelData
    h, w = 40, 64
    tm = np.full((h, w), MAT_HULL, dtype=np.int32)
    tm[1:18, 1:21] = MAT_AIR
    tm[8:11, 21] = MAT_AIR                       # the door DV's span
    tm[1:18, 22:41] = MAT_AIR
    tm[1:18, 42:62] = MAT_AIR
    tm[20:39, 1:31] = MAT_AIR
    tm[20:39, 32:62] = MAT_AIR
    tm[:, 63] = SPACE
    for (px, py) in POCKETS:
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)):
            tm[py + dy, px + dx] = MAT_HULL
        tm[py, px] = MAT_AIR
    tm[30, 10] = MAT_FOLIAGE
    alt = (variant // 2) % 2 == 1
    if alt:
        tm[28, 33:46] = MAT_HULL
    ents = [
        _entity("dv", "door", 3, x=21, y=8, orientation="v", length_m=3.0,
                initial_state="open"),
        _entity("d0", "duct", 12, filter="hepa_basic"),
        _entity("d1", "duct", 10, filter="derelict"),
        _entity("d2", "duct", 14, filter="derelict"),
        _entity("d3", "duct", 11, filter="hepa_basic"),
        _entity("d4", "duct", 13, filter="derelict"),
        # d0: two returns in A, supplies in B, C and on the door's middle tile
        _vent("r0", 21, 5, 5, "d0", "return"),
        _vent("r1", 27, 15, 12, "d0", "return", q=1.5),
        _vent("s0", 22, 25, 5, "d0", "supply"),
        _vent("s1", 30, 35, 12, "d0", "supply", q=1.0),
        _vent("s2", 24, 50, 10, "d0", "supply", q=0.75),
        _vent("sv", 25, 21, 9, "d0", "supply", q=0.5),
        # d1: C's smoke -> A, onto r0's tile
        _vent("r2", 23, 55, 5, "d1", "return", q=3.0),
        _vent("s3", 29, 5, 5, "d1", "supply", q=2.5),
        # d2: D (cold), the wall vent facing space, pocket 1 -> E, the foliage
        _vent("r3", 31, 10, 25, "d2", "return"),
        _vent("r4", 33, 62, 22, "d2", "return", mount="wall", facing="e"),
        _vent("r5", 35, POCKETS[0][0], POCKETS[0][1], "d2", "return"),
        _vent("s4", 26, 45, 30, "d2", "supply"),
        _vent("s5", 37, 10, 30, "d2", "supply"),
        # d3: returns only (one through the hull x 41 until it is destroyed)
        _vent("r6", 28, 40, 25, "d3", "return", q=4.0),
        _vent("r7", 39, 41, 5, "d3", "return"),
        # d4: supplies only (its plenum set by the fixtures)
        _vent("s6", 34, 50, 35, "d4", "supply", q=2.0),
        _vent("s8", 36, POCKETS[1][0], POCKETS[1][1], "d4", "supply", q=1.0),
        _vent("s7", 38, 55, 33, "d4", "supply", q=0.0002),
        # unwired: no duct, a dangling duct
        _vent("u0", 40, 30, 30, "", "supply"),
        _vent("u1", 41, 35, 30, "nope", "return"),
    ]
    if not alt:
        ents += [_entity("d5", "duct", 15, filter="hepa_basic"),
                 _vent("s9", 42, POCKETS[2][0], POCKETS[2][1], "d5", "supply", q=1.0)]
    rng = np.random.default_rng(900 + variant)
    order = list(rng.permutation(len(ents)))
    ents = [ents[i] for i in order]              # the list order is not the ordinal order
    return LevelData(name=f"p6b2_vents{variant}", version="3", path=Path("."),
                     tilemap=tm, tile_size_m=1.0, diffuse_path=Path("."), entities=ents,
                     boundary="ambient" if variant % 2 else "vacuum")


def _set_gas(g, y, x, slice_, value):
    """A fixture write of one gas cell, the energy reseeded on it."""
    g.gas[slice_][y, x] = int(value)
    g.reseed_gas_energy((np.array([y]), np.array([x])))


class VentScene:
    """The vent scene, driven by fixtures between ticks (smoke and steam in
    the rooms, D seeded cold, the pockets emptied; a door closing over a
    vent, a hull destroyed under one, a flood, two vents killed for a while;
    a hot and a cold plenum set on d4 / d5 for the rails and the wipe; a trace
    slice near its saturation at a supply). ``variant`` shifts the schedule,
    the seed, the layout and the boundary."""

    def __init__(self, variant=0):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.v, self.k = variant, 0
        sim = Simulation(_vent_level(variant), seed=62200 + variant, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim = sim
        from simulation.unit import Unit
        u = Unit("U", x=18.0, y=9.0, team=0)
        sim.add_unit(u)
        u.current_hp = 1.0e6
        self.u = u
        g = sim.gmap
        rng = np.random.default_rng(77 + variant)
        # smoke in A (hepa scrubs it to the sink), steam in A (hepa passes it);
        # smoke, poison, teargas, fuel gas in C (the fractional derelict)
        # raw trace counts: scale-free (#88) -- the gate compares device with host
        g.gas[SMOKE][1:18, 1:21] = rng.integers(0, 1 << 26, (17, 20)).astype(np.int32)
        # raw trace counts: scale-free (#88) -- the gate compares device with host
        g.gas[STEAM][1:18, 1:21] = rng.integers(0, 1 << 24, (17, 20)).astype(np.int32)
        for s, hi in ((SMOKE, 1 << 25), (POISON, 1 << 23), (TEARGAS, 1 << 22),
                      (FUEL_GAS, 1 << 21)):
            g.gas[s][1:18, 42:62] = rng.integers(0, hi, (17, 20)).astype(np.int32)
        # D cold (e_plenum below zero at the divisions)
        g.seed_gas_temperature((slice(20, 39), slice(1, 31)), -40 * ONE - 12345 * variant)
        for (px, py) in POCKETS:                # the pockets airless
            for s in range(N_GASES):
                _set_gas(g, py, px, s, 0)
        self.byid = {e.id: e for e in sim.entities}

    def _use(self):
        self.sim.use_intent(self.u.id)

    def tick(self):
        sim, k = self.sim, self.k - self.v
        g = sim.gmap
        V, D = self.byid, self.byid
        if k == 10:
            self._use()                          # DV closes over sv's tile
        if k == 15:
            V["r1"].alive = False                # two vents dead for a while
            V["s1"].alive = False
        if k == 20:
            V["r1"].alive = True
            V["s1"].alive = True
            g.water_depth[30, 45] = 1 << 14      # s4's aperture flooded
        if k == 24:
            g.water_depth[30, 45] = 0
        if k == 26:
            for yy in range(3, 8):               # s0's tile (and its block, so the
                for xx in range(23, 28):         # tick's diffusion keeps it there)
                    _set_gas(g, yy, xx, STEAM, TRACE_CELL_MAX - 3)   # near saturation
        if k == 30:
            self._use()                          # DV opens
        if k == 40:                              # a HOT plenum on d4: the high rail
            d = D["d4"]
            d.o2_raw, d.n2_raw = 9000, 21000
            d.e_plenum = (d.o2_raw + d.n2_raw) * (int(sim._vent_t_max_phys_q) + (1 << 20)) + 777
            d.trace_raw[1] = 10 ** 9             # smoke in the plenum, into pocket 2
        if k == 41:                              # pocket 2 (air in it now: an airless
            px, py = POCKETS[1]                  # cell's trace is wiped) near its
            _set_gas(g, py, px, SMOKE, TRACE_CELL_MAX - 3)   # saturation: a clip
        if k == 50:
            g.destroy_wall(5, 41)                # r7's hull destroyed: it starts to draw
        if k == 60:
            self._use()                          # DV closes again
        if k == 70 and "d5" in D:                # a COLD plenum on d5: the low rail
            d = D["d5"]
            d.o2_raw, d.n2_raw = 4000, 9000
            d.e_plenum = (d.o2_raw + d.n2_raw) * (int(sim._vent_t_min_q) - (1 << 20)) - 555
        if k == 75:
            g.destroy_wall(9, 21)                # DV dies: sv's tile is air again
        if k == 90:                              # a near-empty plenum holding energy
            d = D["d4"]
            d.o2_raw, d.n2_raw, d.e_plenum = 10, 20, 123456789
        sim.step()
        sim.set_paused(False)
        self.k += 1


class LevelDriver:
    """A shipped level's sim, stepped; with ``match`` a MatchSession."""

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


def _tables_for(sims):
    dt, vt = unit_kernels.vent_tables(len(sims), bp=bp)
    dt.stage_lists([unit_kernels.ducts_of(s) for s in sims])
    vt.stage_lists([unit_kernels.vents_of(s) for s in sims])
    return dt, vt


def _step_both(sims, tag):
    """ONE sweep: the device's on the sims' planes and tables, then the host's
    ``sweep_vents`` on the sims themselves; every plane, the tables and the
    books compared. Returns the device books (None on a difference)."""
    from simulation import vent_system
    pre = [_planes(s.gmap) for s in sims]
    bk0 = [_books(s.gmap) for s in sims]
    st = _store(sims)
    dt, vt = _tables_for(sims)
    dev = _dev_planes(pre)
    _cov, books = unit_kernels.sweep_vents(st, _applier_for(st), dev, dt, vt)
    for s in sims:
        vent_system.sweep_vents(s)
    drows, dcnt, _ = dt.download()
    vrows, vcnt, _ = vt.download()
    for e, s in enumerate(sims):
        got = {n: dev[n].get()[e] for n in applier.PLANE_NAMES}
        post = _planes(s.gmap)
        for n in applier.PLANE_NAMES:
            fd = first_field_diff(n, got[n][None], post[n][None])
            if fd is not None:
                fail(f"{tag} env {e}: plane {n} at {fd[1]}: host {fd[2]} device {fd[3]}")
                return None
        if drows[e, :dcnt[e]].tolist() != unit_kernels.ducts_of(s).tolist():
            fail(f"{tag} env {e}: the duct table differs from the host's ducts")
            return None
        if vrows[e, :vcnt[e]].tolist() != unit_kernels.vents_of(s).tolist():
            fail(f"{tag} env {e}: the vent table differs from the host's vents")
            return None
        bk1 = _books(s.gmap)
        for b in applier.BOOKS:
            if b.host is not None and int(books[b.name][e]) != bk1[b.name] - bk0[e][b.name]:
                fail(f"{tag} env {e}: book {b.name}: host {bk1[b.name] - bk0[e][b.name]} "
                     f"device {int(books[b.name][e])}")
                return None
    return books


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def part_tables():
    sims = [VentScene(0).sim, VentScene(1).sim]
    st = _store(sims)
    got = st.tables.download()
    n = 0
    as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
    for e, sim in enumerate(sims):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    for e, sim in enumerate(sims):            # the duct runtimes read the same row
        names = unit_schema.filter_names()
        effs = unit_tables.read(got[e], "vent_filter_eff_q")
        for d in sim._ducts:
            i = names.index(d.inst.fields["filter"])
            if list(effs[i * 5:(i + 1) * 5]) != list(d.filter_eff_q):
                fail(f"tables env {e}: duct {d.id}'s filter_eff_q is not its table row")
    if bp.object_tables_layout()["fingerprint"] != ot.layout().fingerprint:
        fail("tables: the compiled object tables are not the schema's")
    print(f"tables: {n} P6b.2 values over 2 envs (space, planetside) equal the reference "
          f"(derelict fractional in the gate: {GATE_DERELICT}); the duct table (W "
          f"{DLAY.width}, cap {DLAY.capacity}) and the vent table (W {VLAY.width}, cap "
          f"{VLAY.capacity}) compiled in")


# ---------------------------------------------------------------------------
# PART KIT
# ---------------------------------------------------------------------------
def part_kit():
    sim = VentScene(0).sim
    st = _store([sim], 1)
    dev = _dev_planes([_planes(sim.gmap)])
    rng = np.random.default_rng(6204)
    fd, md = [], []
    for n in (-7, -6, -1, 0, 1, 6, 7, -(1 << 62), 1 << 62):
        for d in (1, 2, 3, 7, 64, -1, -3, (1 << 40) + 7):
            fd.append((n, d))
    for _ in range(3000):
        d = int(rng.integers(1, 1 << int(rng.integers(1, 63))))
        fd.append((int(rng.integers(-(1 << 62), 1 << 62)), d if rng.random() < 0.8 else -d))
    for _ in range(3000):
        c = int(rng.integers(1, 1 << int(rng.integers(1, 63))))
        md.append((int(rng.integers(0, 1 << int(rng.integers(1, 63)))),
                   int(rng.integers(0, c + 1)), c))
    md += [((1 << 62), (1 << 62), 1), (5, 7, 0), (-1, 3, 4), (3, -1, 4), (3, 4, -2),
           ((1 << 63) - 1, (1 << 63) - 1, (1 << 63) - 1)]
    ops = [[unit_kernels.SEAM_OPS["floordiv"], n, d, 0] for n, d in fd] + \
          [[unit_kernels.SEAM_OPS["muldiv"], a, b, c] for a, b, c in md]
    res, _books_ = unit_kernels.seam_probe(st, dev, [ops], [np.zeros((0, 2))])
    r = res[0]
    neg = wide = refused = 0
    for i, (n, d) in enumerate(fd):
        if int(r[i][1]) != n // d:
            fail(f"kit: device floordiv({n}, {d}) = {int(r[i][1])}, Python {n // d}")
            break
        neg += (n % d != 0) and ((n < 0) != (d < 0))
    for j, (a, b, c) in enumerate(md):
        want = (a * b) // c if (a >= 0 and b >= 0 and c > 0 and (a * b) // c < (1 << 63)) else -1
        got = int(r[len(fd) + j][1])
        if got != want:
            fail(f"kit: device muldiv({a}, {b}, {c}) = {got}, want {want}")
            break
        wide += a >= 0 and b >= 0 and a * b >= (1 << 64)
        refused += want == -1
    if neg < 100 or wide < 100 or refused < 4:
        fail(f"kit: vacuous (truncating floors {neg}, 128-bit products {wide}, refusals {refused})")
    print(f"kit: {len(fd)} floor divisions ({neg} on a negative remainder) and {len(md)} floor "
          f"products ({wide} past 2^64, {refused} refused) on the device == Python")


# ---------------------------------------------------------------------------
# PART PROBE (inject_gas_n_vec)
# ---------------------------------------------------------------------------
def _closure_inject(tag, pre, post, books, adds, clip_ok=True):
    """The books close on the device's planes: sum gas_energy moves by the
    inject channels; the bulk mass by the vector's bulk; each trace slice by
    its add minus its clip."""
    de = int(post["gas_energy"].astype(object).sum() - pre["gas_energy"].astype(object).sum())
    seam = int(books["ge_pump_inject_vec"][0]) + int(books["ge_pump_rail"][0])
    if de != seam:
        fail(f"{tag}: d sum gas_energy {de} != the inject channels {seam}")
        return False
    for g in range(N_GASES):
        dn = int(post["gas"][g].astype(np.int64).sum() - pre["gas"][g].astype(np.int64).sum())
        want = int(adds[g]) - (0 if g in (O2, INERT_N2) else int(books[f"trace_clip_{g}"][0]))
        if dn != want:
            fail(f"{tag}: gas slice {g} moved by {dn}, want {want}")
            return False
    return True


class Probe:
    def __init__(self, name, sim, seen):
        self.name, self.sim, self.seen = name, sim, seen
        self.st = _store([sim], 1)
        self.n = 0

    def inject(self, y, x, add, t_dep, t_min=None, t_max=None, tag=""):
        """One inject_gas_n_vec by the device on the map's planes, then by the
        host on the map; compared and counted."""
        g, sim = self.sim.gmap, self.sim
        from simulation import vent_system
        lo, hi = vent_system._t_rails_q()
        t_min = lo if t_min is None else t_min
        t_max = hi if t_max is None else t_max
        add = [int(a) for a in add]
        pre = _planes(g)
        bk0 = _books(g)
        row = [unit_kernels.SEAM_OPS["inject_vec"], y, x, t_dep, t_min, t_max] + add
        dev = _dev_planes([pre])
        res, books = unit_kernels.seam_probe(self.st, dev, [[row]], [np.zeros((0, 2))])
        r = res[0][0]
        hcode, ht, hr = 0, 0, 0
        acct = 0 <= y < g._h and 0 <= x < g._w and g.gas_is_accountable(y, x)
        try:
            ht, hr = g.inject_gas_n_vec(y, x, add, t_dep, t_min, t_max)
        except ValueError:
            hcode = CODE["out_of_bounds"]
        except OverflowError:
            hcode = CODE["overflow"]
        post = _planes(g)
        bk1 = _books(g)
        self.n += 1
        where = f"probe {self.name} #{self.n} inject_vec({y}, {x}, t_dep {t_dep}) {tag}"
        if int(r[0]) != hcode:
            fail(f"{where}: device code {int(r[0])} != host {hcode}")
            return None
        if not hcode and (int(r[1]) != int(ht) or int(r[2]) != int(hr)):
            fail(f"{where}: device (t_new {int(r[1])}, rail {int(r[2])}) != host ({ht}, {hr})")
            return None
        got = {n: dev[n].get()[0] for n in applier.PLANE_NAMES}
        for n in applier.PLANE_NAMES:
            fd = first_field_diff(n, got[n][None], post[n][None])
            if fd is not None:
                fail(f"{where}: plane {n} at {fd[1]}: host {fd[2]} device {fd[3]}")
                return None
        for b in applier.BOOKS:
            if b.host is not None and int(books[b.name][0]) != bk1[b.name] - bk0[b.name]:
                fail(f"{where}: book {b.name}: host {bk1[b.name] - bk0[b.name]} device "
                     f"{int(books[b.name][0])}")
                return None
        if hcode:
            if any(not np.array_equal(pre[n], post[n]) for n in applier.PLANE_NAMES):
                fail(f"{where}: a refused op wrote the host's planes")
            self.seen[f"inject: {applier.SEAM_REFUSAL_NAMES[hcode - 1]}"] += 1
            return None
        if not _closure_inject(where, pre, got, books, add):
            return None
        s = self.seen
        if sum(add) == 0:
            s["inject: a zero vector (nothing written)"] += 1
        elif add[O2] + add[INERT_N2] + int(pre["gas"][O2][y, x]) + int(pre["gas"][INERT_N2][y, x]) == 0:
            s["inject: a bulk-free result (T untouched)"] += 1
        elif acct:
            s["inject: accountable (the conversion seam)"] += 1
            e1 = int(pre["gas_energy"][y, x]) + (add[O2] + add[INERT_N2]) * (
                t_dep + g.gas_t_amb_raw())
            n1 = int(pre["gas"][O2][y, x]) + int(pre["gas"][INERT_N2][y, x]) + add[O2] + add[INERT_N2]
            if e1 % n1 != 0 and e1 < 0 and e1 // n1 - g.gas_t_amb_raw() >= t_min:
                s["inject: a negative energy's floor, unrailed"] += 1
            if e1 % n1 != 0 and e1 < 0:
                s["inject: a negative energy's floor"] += 1
            elif e1 % n1 != 0:
                s["inject: a positive energy's floor"] += 1
        else:
            s["inject: non-accountable (the T-mix)"] += 1
            nn = int(pre["gas"][O2][y, x]) + int(pre["gas"][INERT_N2][y, x])
            num = nn * int(pre["temperature"][y, x]) + (add[O2] + add[INERT_N2]) * t_dep
            if num % (nn + add[O2] + add[INERT_N2]) != 0:
                s["inject: the T-mix's floor (" + ("neg" if num < 0 else "pos") + ")"] += 1
        if hr < 0:
            s["inject: the low rail"] += 1
        if hr > 0:
            s["inject: the high rail"] += 1
        if any(int(books[f"trace_clip_{q}"][0]) for q in range(N_GASES)):
            s["inject: a trace clip"] += 1
        return True

    def device_only(self, y, x, add, t_dep, want, tag):
        g = self.sim.gmap
        pre = _planes(g)
        dev = _dev_planes([pre])
        from simulation import vent_system
        lo, hi = vent_system._t_rails_q()
        row = [unit_kernels.SEAM_OPS["inject_vec"], y, x, t_dep, lo, hi] + [int(a) for a in add]
        res, books = unit_kernels.seam_probe(self.st, dev, [[row]], [np.zeros((0, 2))])
        if int(res[0][0][0]) != CODE[want]:
            fail(f"probe {self.name}: device-only inject ({tag}) gave {int(res[0][0][0])}, "
                 f"want {want}")
        if any(not np.array_equal(dev[n].get()[0], pre[n]) for n in applier.PLANE_NAMES) or \
                any(int(v[0]) for v in books.values()):
            fail(f"probe {self.name}: a device-only refusal ({tag}) wrote something")
        self.seen[f"inject: {want} (device only: {tag})"] += 1


def _vec(rng, bulk_hi=1 << 20, trace_p=0.5, trace_hi=1 << 24):
    a = [0] * N_GASES
    a[O2] = int(rng.integers(0, bulk_hi))
    a[INERT_N2] = int(rng.integers(0, bulk_hi))
    for g in range(N_GASES):
        if g not in (O2, INERT_N2) and rng.random() < trace_p:
            a[g] = int(rng.integers(0, trace_hi))
    return a


def _probe_map(name, sim, seen, rng, ticks=20):
    from simulation import vent_system
    for _ in range(ticks):
        sim.step()
        sim.set_paused(False)
    P = Probe(name, sim, seen)
    g = sim.gmap
    lo, hi = vent_system._t_rails_q()
    acct = [tuple(int(v) for v in t) for t in np.argwhere(~g.solid & ~g.thermal_solid
                                                          & ~g.is_vacuum & ~g.is_ambient)]
    ring = [tuple(int(v) for v in t) for t in np.argwhere(g.is_vacuum | g.is_ambient)]
    rng.shuffle(acct)
    rng.shuffle(ring)
    for i, (y, x) in enumerate(acct[:30]):
        P.inject(y, x, _vec(rng), int(rng.integers(-(1 << 26), 1 << 26)), tag="random")
        P.inject(y, x, _vec(rng, 1 << 12), int(rng.integers(-(1 << 29), 1 << 29)), tag="any T")
    for (y, x) in ring[:8]:
        P.inject(y, x, _vec(rng), int(rng.integers(-(1 << 26), 1 << 26)), tag="ring")
    # the rails: a deposit dominating its tile, hotter than T_MAX / colder than T_MIN
    for (y, x) in acct[30:34]:
        P.inject(y, x, [0, 0] + [0] * (N_GASES - 2) if False else
                 [((1 << 28) if q in (O2, INERT_N2) else 0) for q in range(N_GASES)],
                 hi + int(rng.integers(1, 1 << 20)), tag="hot")
    for (y, x) in acct[34:38]:
        P.inject(y, x, [((1 << 28) if q in (O2, INERT_N2) else 0) for q in range(N_GASES)],
                 lo - int(rng.integers(1, 1 << 20)), tag="cold")
    for (y, x) in ring[8:10]:
        P.inject(y, x, [((1 << 28) if q in (O2, INERT_N2) else 0) for q in range(N_GASES)],
                 hi + 5, tag="ring hot")
        P.inject(y, x, [((1 << 28) if q in (O2, INERT_N2) else 0) for q in range(N_GASES)],
                 lo - 5, tag="ring cold")
    # the mirror's floor on a NEGATIVE energy: with the shipped rails (T_MIN >
    # -T_AMB) a negative energy always rails low, so the floor is seen only
    # under a rail below -T_AMB (the primitive takes any rails)
    t_amb = g.gas_t_amb_raw()
    for (y, x) in acct[38:40]:
        P.inject(y, x, [((1 << 26) + int(rng.integers(0, 999)) if q in (O2, INERT_N2) else 0)
                        for q in range(N_GASES)],
                 -t_amb - int(rng.integers(1, 1 << 16)), -(1 << 29), hi, tag="below -T_AMB")
    a = acct[40]
    P.inject(a[0], a[1], [0] * N_GASES, 1234, tag="zero vector")
    P.inject(a[0], a[1], _vec(rng), 0, tag="t_dep 0")
    # a bulk-free result: trace only into an airless tile
    keep = [int(g.gas[s][a]) for s in range(N_GASES)]
    _set_gas(g, a[0], a[1], O2, 0)
    _set_gas(g, a[0], a[1], INERT_N2, 0)
    tv = [0] * N_GASES
    tv[SMOKE] = 4321
    P.inject(a[0], a[1], tv, 999, tag="trace only, airless")
    for s in range(N_GASES):
        _set_gas(g, a[0], a[1], s, keep[s])
    # the trace saturation
    b = acct[41]
    _set_gas(g, b[0], b[1], STEAM, TRACE_CELL_MAX - 9)
    tv = [0] * N_GASES
    tv[STEAM], tv[O2] = 1 << 20, 50
    P.inject(b[0], b[1], tv, 100, tag="saturation")
    # the refusals
    c = acct[42]
    big = [0] * N_GASES
    big[O2] = (1 << 31) - int(g.gas[O2][c])
    P.inject(c[0], c[1], big, 0, tag="overflow")
    P.inject(-1, 3, _vec(rng), 0, tag="oob")
    P.inject(3, g._w, _vec(rng), 0, tag="oob")
    neg = _vec(rng)
    neg[INERT_N2] = -1
    P.device_only(c[0], c[1], neg, 0, "fault", "a negative slice")
    P.device_only(c[0], c[1], _vec(rng), 1 << 30, "fault", "t_dep past 2^30")
    huge = [0] * N_GASES
    huge[SMOKE] = 1 << 54
    P.device_only(c[0], c[1], huge, 0, "fault", "a slice past 2^54")
    return P.n


def part_probe():
    seen = Counter()
    rng = np.random.default_rng(6205)
    n = 0
    for name in ("ship_demo", "playground", "fire_smoke_studio", "test_level"):
        n += _probe_map(name, LevelDriver(name).sim, seen, rng)
    n += _probe_map("vent scene", VentScene(0).sim, seen, rng, ticks=5)
    n += _probe_map("vent scene (planetside)", VentScene(1).sim, seen, rng, ticks=5)
    need = ("inject: accountable (the conversion seam)", "inject: non-accountable (the T-mix)",
            "inject: a negative energy's floor", "inject: a positive energy's floor",
            "inject: a negative energy's floor, unrailed",
            "inject: the T-mix's floor (neg)", "inject: the T-mix's floor (pos)",
            "inject: the low rail", "inject: the high rail", "inject: a trace clip",
            "inject: a zero vector (nothing written)", "inject: a bulk-free result (T untouched)",
            "inject: overflow", "inject: out_of_bounds",
            "inject: fault (device only: a negative slice)",
            "inject: fault (device only: t_dep past 2^30)",
            "inject: fault (device only: a slice past 2^54)")
    for k in need:
        if not seen[k]:
            fail(f"probe: the case {k!r} was never taken")
    print(f"probe: {n} inject_gas_n_vec ops on 6 maps, device == host (results, planes, "
          f"books, closure) on every one; cases: "
          + ", ".join(f"{k} {v}" for k, v in sorted(seen.items())))


# ---------------------------------------------------------------------------
# The census of the sweep's branches (the host's own decisions)
# ---------------------------------------------------------------------------
_GUARDS = ("solid", "thermal_solid", "vacuum", "ambient", "flooded")


def _guard_reason(g, y, x):
    if g.solid[y, x]:
        return "solid"
    if g.thermal_solid[y, x]:
        return "thermal_solid"
    if g.is_vacuum[y, x]:
        return "vacuum"
    if g.is_ambient[y, x]:
        return "ambient"
    if int(g.water_depth[y, x]) != 0:
        return "flooded"
    return None


class Census:
    """Wraps (reads only) ``vent_system._duct_sweep`` and, inside it,
    ``GameMap.extract_gas_n_vec`` / ``inject_gas_n_vec``, and counts each
    branch the law takes: the decisions are re-derived from the state at the
    duct's entry (guards and quanta do not depend on the returns' edits) and
    the primitives' own calls."""

    def __init__(self):
        self.c = Counter()
        self._orig = []
        self.ctx = None
        self.tiles = Counter()

    def install(self):
        from simulation import vent_system
        from simulation.gamemap import GameMap
        c = self.c
        cen = self
        o_ds = vent_system._duct_sweep

        def ds(gmap, duct, ret, sup, t_min_q, t_max_phys_q, __o=o_ds):
            tiles = Counter((v.aperture_y, v.aperture_x) for v in ret + sup)
            if any(n > 1 for n in tiles.values()):
                pass
            for v in ret:
                cen.tiles[(v.aperture_y, v.aperture_x, id(duct))] += 1
            for v in sup:
                cen.tiles[(v.aperture_y, v.aperture_x, id(duct))] += 1
            pre = dict(o2=duct.o2_raw, n2=duct.n2_raw, e=duct.e_plenum, wipe=duct.e_wipe,
                       lo=duct.rail_lo_hits, hi=duct.rail_hi_hits, sink=list(duct.sink),
                       tr=list(duct.trace_raw))
            ready = []
            for v in sup:
                if not v.alive:
                    c["supply: a dead vent skipped"] += 1
                    continue
                why = _guard_reason(gmap, v.aperture_y, v.aperture_x)
                if why:
                    c[f"supply: guarded ({why})"] += 1
                    continue
                w_i = (v.accum + v.rate_raw_per_s) // v.tps
                if w_i > 0:
                    ready.append(w_i)
                else:
                    c["supply: a quantum of 0 (accrued)"] += 1
            for v in ret:
                if not v.alive:
                    c["return: a dead vent skipped"] += 1
                    continue
                why = _guard_reason(gmap, v.aperture_y, v.aperture_x)
                if why:
                    c[f"return: guarded ({why})"] += 1
                    continue
                if (v.accum + v.rate_raw_per_s) // v.tps <= 0:
                    c["return: a quantum of 0 (accrued)"] += 1
            cen.ctx = dict(duct=duct, gmap=gmap, extracts=[], injects=[], pre=pre)
            try:
                return __o(gmap, duct, ret, sup, t_min_q, t_max_phys_q)
            finally:
                ctx, cen.ctx = cen.ctx, None
                cen._post(ctx, ready)
        vent_system._duct_sweep = ds
        self._orig.append((vent_system, "_duct_sweep", o_ds))

        o_ex, o_in = GameMap.extract_gas_n_vec, GameMap.inject_gas_n_vec

        def ex(g, fy, fx, dn, __o=o_ex):
            t = int(g.temperature[fy, fx])
            out = __o(g, fy, fx, dn)
            if cen.ctx is not None and g is cen.ctx["gmap"]:
                cen.ctx["extracts"].append((list(out), t))
            return out

        def inj(g, fy, fx, add, t_dep, t_min_q, t_max_phys_q, *a, __o=o_in, **k):
            clip0 = list(g.trace_clip_books)
            out = __o(g, fy, fx, add, t_dep, t_min_q, t_max_phys_q, *a, **k)
            if cen.ctx is not None and g is cen.ctx["gmap"]:
                cen.ctx["injects"].append((list(add), int(t_dep), out,
                                           list(g.trace_clip_books) != clip0))
            return out
        GameMap.extract_gas_n_vec, GameMap.inject_gas_n_vec = ex, inj
        self._orig += [(GameMap, "extract_gas_n_vec", o_ex), (GameMap, "inject_gas_n_vec", o_in)]
        return self

    def _post(self, ctx, ready):
        from simulation import vent_system
        c, d, pre = self.c, ctx["duct"], ctx["pre"]
        e = pre["e"]
        o2, n2 = pre["o2"], pre["n2"]
        filt = d.inst.fields["filter"]
        for rem, t in ctx["extracts"]:
            if sum(rem) == 0:
                c["return: the aperture empty (nothing credited)"] += 1
                continue
            c["return: extracted"] += 1
            o2 += rem[O2]
            n2 += rem[INERT_N2]
            e += (rem[O2] + rem[INERT_N2]) * t
            for i, gid in enumerate(vent_system._TRACE_IDS):
                r = rem[gid]
                if r == 0:
                    continue
                eff = d.filter_eff_q[i]
                c[f"filter {filt}: trace drawn"] += 1
                if (r * eff) >> 16:
                    c["return: trace scrubbed into the sink"] += 1
                if r - ((r * eff) >> 16):
                    c["return: trace passed into the plenum"] += 1
                if (r * eff) & 0xFFFF:
                    c["return: the scrub's truncation"] += 1
            if t < 0:
                c["return: a cold tile's energy credit (below ambient)"] += 1
        n_bulk = o2 + n2
        if n_bulk < vent_system.N_EPS:
            c["plenum: near-empty, energy wiped" if e != 0 else
              "plenum: near-empty, nothing to wipe"] += 1
            return
        sum_w = sum(ready)
        total = min(n_bulk, sum_w)
        if total <= 0:
            c["plenum: no ready supply"] += 1
            return
        c["plenum: drain-limited (N_bulk < sum of weights)" if n_bulk < sum_w
          else "plenum: rate-limited (sum of weights <= N_bulk)"] += 1
        t_dep = e // n_bulk
        if e < 0 and e % n_bulk:
            c["plenum: a negative e_plenum's floor at the division"] += 1
        elif e > 0 and e % n_bulk:
            c["plenum: a positive e_plenum's floor at the division"] += 1
        if len(ctx["injects"]) < len(ready):
            c["plenum: a share floored to 0"] += 1
        o2_rem, n2_rem = o2, n2
        for add, td, out, clipped in ctx["injects"]:
            if td != t_dep:
                c["CENSUS ERROR: t_dep"] += 1
            share = add[O2] + add[INERT_N2]
            if o2_rem + n2_rem > 0 and ((o2_rem * share) // (o2_rem + n2_rem)
                                         + (n2_rem * share) // (o2_rem + n2_rem)) < share:
                c["deposit: the split's shortfall cascaded"] += 1
            o2_rem -= add[O2]
            n2_rem -= add[INERT_N2]
            c["deposit: landed"] += 1
            if any(add[g] for g in range(N_GASES) if g not in (O2, INERT_N2)):
                c["deposit: trace rides along"] += 1
            if out[1] < 0:
                c["deposit: the low rail"] += 1
            if out[1] > 0:
                c["deposit: the high rail"] += 1
            if clipped:
                c["deposit: a trace clip"] += 1

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()


NEED = (
    "return: a dead vent skipped", "return: guarded (solid)",
    "return: guarded (vacuum)", "return: extracted",
    "return: the aperture empty (nothing credited)", "return: trace scrubbed into the sink",
    "return: trace passed into the plenum", "return: the scrub's truncation",
    "return: a cold tile's energy credit (below ambient)",
    "supply: a dead vent skipped", "supply: guarded (solid)",
    "supply: guarded (thermal_solid)", "supply: guarded (flooded)",
    "supply: a quantum of 0 (accrued)",
    "plenum: near-empty, energy wiped", "plenum: near-empty, nothing to wipe",
    "plenum: no ready supply", "plenum: drain-limited (N_bulk < sum of weights)",
    "plenum: rate-limited (sum of weights <= N_bulk)",
    "plenum: a negative e_plenum's floor at the division",
    "plenum: a positive e_plenum's floor at the division",
    "plenum: a share floored to 0",
    "deposit: landed", "deposit: the split's shortfall cascaded", "deposit: trace rides along",
    "deposit: the low rail", "deposit: the high rail", "deposit: a trace clip",
    "filter hepa_basic: trace drawn", "filter derelict: trace drawn",
)
NEED_PLANETSIDE = ("return: guarded (ambient)",)
UNREACHABLE = {
    "plenum: every share floored to 0 (nothing moves)":
        "with N_bulk >= N_EPS = 64 and at most VENT_CAPACITY = 64 ready supplies, the "
        "largest share floor(min(N_bulk, sum w) * w_max / sum w) is >= 1; the reference "
        "(no vent bound) reaches it only past 64 ready supplies on one duct",
    "inject: the non-accountable branch / a bulk-free result inside the sweep":
        "the aperture guards exclude every non-accountable tile and a deposit's share is "
        "> 0 bulk; PART PROBE takes both",
    "return: a quantum of 0 on a duct's return":
        "taken only by a return slower than one count per tick; every return of the scene "
        "accrues >= 1 (counted when it happens, not required)",
}


# ---------------------------------------------------------------------------
# PART SCENE
# ---------------------------------------------------------------------------
def _identity(sim, where):
    from simulation.entities import vents as vent_schema
    drec, vrec = unit_kernels.ducts_of(sim), unit_kernels.vents_of(sim)
    ducts, vents = unit_kernels.duct_objects(drec), unit_kernels.vent_objects(vrec)
    if len(ducts) != len(sim._ducts) or len(vents) != len(sim._vents):
        fail(f"identity {where}: row counts differ")
        return False
    for k, (d, r) in enumerate(zip(sim._ducts, ducts)):
        a = (int(d.ordinal), d.inst.fields["filter"], d.o2_raw, d.n2_raw, list(d.trace_raw),
             d.e_plenum, list(d.sink), d.e_wipe, d.rail_lo_hits, d.rail_hi_hits, bool(d.alive))
        b = (r.ordinal, r.filter, r.o2_raw, r.n2_raw, r.trace_raw, r.e_plenum, r.sink, r.e_wipe,
             r.rail_lo_hits, r.rail_hi_hits, r.alive)
        if a != b:
            fail(f"identity {where}: duct {d.id}: unpack(pack) differs")
            return False
        want = (("alive", 1 if d.alive else 0),) + vent_schema.duct.runtime_digest_rows(d)
        if unit_kernels.duct_entity_rows(drec[k]) != want:
            fail(f"identity {where}: duct {d.id}: the table's entity rows are not the "
                 f"serializer's")
            return False
    for k, (v, r) in enumerate(zip(sim._vents, vents)):
        a = (int(v.ordinal), unit_schema._vent_duct_ordinal(v),
             "return" if v.role == "return" else "supply", v.aperture_y, v.aperture_x,
             v.rate_raw_per_s, v.tps, v.accum, v.guard_skips, bool(v.alive))
        b = (r.ordinal, r.duct_ordinal, r.role, r.aperture_y, r.aperture_x, r.rate_raw_per_s,
             r.tps, r.accum, r.guard_skips, r.alive)
        if a != b:
            fail(f"identity {where}: vent {v.id}: unpack(pack) differs")
            return False
        want = (("alive", 1 if v.alive else 0),) + vent_schema.vent.runtime_digest_rows(v)
        if unit_kernels.vent_entity_rows(vrec[k]) != want:
            fail(f"identity {where}: vent {v.id}: the table's entity rows are not the "
                 f"serializer's")
            return False
    return True


def run_shadow(name, drivers, ticks, twins=None, slots=None, n_units=None):
    slots = slots or _slots()
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
            rv = reps.get(SLOT_VENTS.key)
            for e in range(len(drivers)):
                if rv is not None and int(rv.books["applied"][e]):
                    acts[SLOT_VENTS.key] += 1
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
    print(f"{name}: N={len(drivers)}, {n} ticks, the slots equal every tick (tables, "
          f"planes, books, write set){' (env i == env i alone)' if len(drivers) > 1 else ''}"
          f"{' (the unhooked twin equal)' if twins is not None else ''}; env-ticks the vent "
          f"sweep moved gas: {acts[SLOT_VENTS.key]} ({time.perf_counter() - t0:.0f} s)")
    return acts, n


def _ledger_span(sim):
    return {d.id: (d.o2_raw + d.n2_raw, d.e_plenum, sum(d.sink), d.e_wipe, d.rail_lo_hits,
                   d.rail_hi_hits) for d in sim._ducts}


def part_scene():
    census = Census().install()
    try:
        sc = VentScene(0)
        start = _ledger_span(sc.sim)
        acts, _ = run_shadow("scene", [sc], SCENE_TICKS, twins=[VentScene(0)])
        end = _ledger_span(sc.sim)
        sc1 = VentScene(1)
        run_shadow("scene planetside", [sc1], SCENE_TICKS)
    finally:
        census.uninstall()
    c = census.c
    for b in NEED + NEED_PLANETSIDE:
        if not c[b]:
            fail(f"scene: the census branch {b!r} was never taken")
    if c["CENSUS ERROR: t_dep"]:
        fail("scene: the census's t_dep differs from the law's")
    shared = Counter()
    for (y, x, _d), n in census.tiles.items():
        shared[(y, x)] += 1
    if not any(n > 1 for n in shared.values()):
        fail("scene: no tile is shared by vents of two ducts (vacuous)")
    if not acts[SLOT_VENTS.key]:
        fail("scene: the vent sweep never moved gas (vacuous)")
    grew = {k: (start[k], end[k]) for k in start}
    big = [k for k, (a, b) in grew.items()
           if abs(b[1]) >= 1000 * max(1, abs(a[1])) or abs(b[2]) >= 1000 * max(1, abs(a[2]))]
    if len(big) < 2:
        fail(f"scene: the ledgers did not leave their start values by orders of magnitude {grew}")
    print("scene census: " + ", ".join(f"{k} {v}" for k, v in sorted(c.items())))
    print("scene unreachable: " + "; ".join(f"{k} -- {v}" for k, v in UNREACHABLE.items()))
    print("scene ledgers (bulk, e_plenum, sum sink, e_wipe, rail lo, rail hi) start -> end: "
          + "; ".join(f"{k}: {a} -> {b}" for k, (a, b) in grew.items()))


# ---------------------------------------------------------------------------
# PART LEVELS
# ---------------------------------------------------------------------------
def part_levels():
    from cuda_unit_objects_check import ShipDriver
    d = ShipDriver()
    acts, n = run_shadow("level ship_demo (bots match)", [d], LEVEL_TICKS,
                         n_units=max(1, len(d.sim.units)), slots=[SLOT_VENTS])
    if not acts[SLOT_VENTS.key]:
        fail("levels: ship_demo's duct never moved gas (vacuous)")
    for name in ("playground", "fire_smoke_studio", "airlock_demo"):
        d = LevelDriver(name, seed=1)
        acts, n = run_shadow(f"level {name} ({len(d.sim._ducts)} ducts, "
                             f"{len(d.sim._vents)} vents)", [d], LEVEL_TICKS,
                             n_units=max(1, len(d.sim.units)), slots=[SLOT_VENTS])
        if d.sim._ducts and not acts[SLOT_VENTS.key]:
            fail(f"levels: {name}'s ducts never moved gas (vacuous)")


# ---------------------------------------------------------------------------
# PART REFUSAL
# ---------------------------------------------------------------------------
def _duct_bounds(sim, k):
    """The proved range of duct k (vent_ledger_range's bounds)."""
    drec, vrec = unit_kernels.ducts_of(sim), unit_kernels.vents_of(sim)
    ordk = int(drec[k, 0])
    b_n = r = s = 0
    for v in unit_kernels.vent_objects(vrec):
        if v.duct_ordinal != ordk:
            continue
        if v.role == "return":
            b_n += min((v.accum + v.rate_raw_per_s) // v.tps, 1 << 32)
            r += 1
        else:
            s += 1
    return b_n, r, s


def part_refusal():
    from simulation import vent_system
    sc = VentScene(0)
    for _ in range(12):
        sc.tick()
    sim = sc.sim
    k = [i for i, d in enumerate(sim._ducts) if d.id == "d0"][0]
    d = sim._ducts[k]
    gv = [j for j, v in enumerate(sim._vents) if v.duct is d][0]
    # name -> (setter, the edge from the duct's (B_N, R, S) at that moment, the
    # step that leaves the range)
    cases = {
        "o2_raw": (lambda x: setattr(d, "o2_raw", x), lambda b, r, s: (1 << 61) - b, 1),
        "n2_raw": (lambda x: setattr(d, "n2_raw", x), lambda b, r, s: (1 << 61) - b, 1),
        "trace_raw": (lambda x: d.trace_raw.__setitem__(2, x),
                      lambda b, r, s: (1 << 54) - r * (1 << 31), 1),
        "sink": (lambda x: d.sink.__setitem__(4, x),
                 lambda b, r, s: (1 << 62) - r * (1 << 31), 1),
        "e_plenum": (lambda x: setattr(d, "e_plenum", x),
                     lambda b, r, s: (1 << 61) - (b << 29), 1),
        "e_plenum (negative)": (lambda x: setattr(d, "e_plenum", x),
                                lambda b, r, s: -((1 << 61) - (b << 29)), -1),
        "e_wipe": (lambda x: setattr(d, "e_wipe", x), lambda b, r, s: 1 << 62, 1),
        "e_wipe (negative)": (lambda x: setattr(d, "e_wipe", x),
                              lambda b, r, s: -(1 << 62), -1),
        "rail_lo_hits": (lambda x: setattr(d, "rail_lo_hits", x),
                         lambda b, r, s: (1 << 62) - s, 1),
        "rail_hi_hits": (lambda x: setattr(d, "rail_hi_hits", x),
                         lambda b, r, s: (1 << 62) - s, 1),
        "o2_raw (negative)": (lambda x: setattr(d, "o2_raw", x), lambda b, r, s: 0, -1),
        "guard_skips": (lambda x: setattr(sim._vents[gv], "guard_skips", x),
                        lambda b, r, s: (1 << 62) - 1, 1),
    }
    snap = (d.o2_raw, d.n2_raw, list(d.trace_raw), list(d.sink), d.e_plenum, d.e_wipe,
            d.rail_lo_hits, d.rail_hi_hits, [v.guard_skips for v in sim._vents])
    import copy

    def restore():
        (d.o2_raw, d.n2_raw, tr, sk, d.e_plenum, d.e_wipe, d.rail_lo_hits,
         d.rail_hi_hits, gs) = copy.deepcopy(snap)
        d.trace_raw[:] = tr
        d.sink[:] = sk
        for v, x in zip(sim._vents, gs):
            v.guard_skips = x
    n_ok = n_ref = 0
    for name, (setter, edge_of, step) in cases.items():
        # one step past: refused by name, nothing of the duct written
        restore()
        b_n, r, s = _duct_bounds(sim, k)
        edge = edge_of(b_n, r, s)
        setter(edge + step)
        pre = _planes(sim.gmap)
        st = _store([sim])
        dt, vt = _tables_for([sim])
        d0, v0 = dt.download()[0][0].copy(), vt.download()[0][0].copy()
        dev = _dev_planes([pre])
        try:
            unit_kernels.sweep_vents(st, _applier_for(st), dev, dt, vt)
            fail(f"refusal {name}: a ledger one past its range ({edge + step}) was not refused")
        except applier.LedgerRangeRefused as ex:
            msg = str(ex)
            key = name.split(" ")[0]
            if f"ordinal {int(d.ordinal)}" not in msg or key not in msg:
                fail(f"refusal {name}: the refusal does not name the duct and the ledger: {msg}")
            n_ref += 1
        d1, v1 = dt.download()[0][0], vt.download()[0][0]
        if not np.array_equal(d1[k], d0[k]):
            fail(f"refusal {name}: the refused duct's row was written")
        mine = [j for j, v in enumerate(sim._vents) if v.duct is d]
        if not np.array_equal(v1[mine], v0[mine]):
            fail(f"refusal {name}: the refused duct's vents were written")
        # the other ducts ran: their planes must equal a host run with this
        # duct's vents removed is not what we compare; the refused duct's own
        # apertures are untouched instead
        got = dev["gas"].get()[0]
        for v in sim._vents:
            if v.duct is d and v.duct is not None:
                y, x = v.aperture_y, v.aperture_x
                others = [w for w in sim._vents if w.duct is not d and
                          (w.aperture_y, w.aperture_x) == (y, x)]
                if not others and not np.array_equal(got[:, y, x], pre["gas"][:, y, x]):
                    fail(f"refusal {name}: the refused duct's aperture ({y}, {x}) was written")
        # at the bound: run, and equal to the host
        restore()
        b_n, r, s = _duct_bounds(sim, k)
        edge = edge_of(b_n, r, s)
        setter(edge)
        if name.startswith("e_plenum"):
            d.o2_raw = 1 << 40                   # t_dep inside the deposit's range
        if _step_both([sim], f"refusal at the bound ({name})") is not None:
            n_ok += 1
        restore()
    print(f"refusal: {n_ref} ledgers one past their proved range refused by name (nothing of "
          f"the duct written); {n_ok} at the bound run and equal the host "
          f"(B_N {b_n}, returns {r}, supplies {s})")
    if n_ref != len(cases) or n_ok != len(cases):
        fail(f"refusal: {n_ref} refused / {n_ok} at the bound of {len(cases)} cases")


# ---------------------------------------------------------------------------
# PART BATCH
# ---------------------------------------------------------------------------
def part_batch():
    for n in BATCH_NS:
        # env 0 holds FEWER vents and ducts than the next (an env reading
        # another's counts would walk past its own rows or stop short)
        drivers = ([VentScene(2), VentScene(1)] if n == 2 else
                   [VentScene(v) for v in (2, 3, 0, 1, 6, 7, 4, 5)[:n]])
        acts, _ = run_shadow(f"batch N={n}", drivers, BATCH_TICKS)
        if not acts[SLOT_VENTS.key]:
            fail(f"batch N={n}: the vent sweep never moved gas (vacuous)")
        if len({len(d.sim._ducts) for d in drivers}) < 2 or \
                len({len(d.sim._vents) for d in drivers}) < 2:
            fail(f"batch N={n}: every env holds the same duct / vent counts (vacuous)")


# ---------------------------------------------------------------------------
# PART PADDING
# ---------------------------------------------------------------------------
def part_padding():
    scs = [VentScene(0), VentScene(2)]
    for sc in scs:
        for _ in range(4):
            sc.tick()
    sims = [s.sim for s in scs]
    st = _store(sims)

    def run(dirty):
        dt, vt = unit_kernels.vent_tables(2, bp=bp)
        rd = np.zeros(dt.shape, dtype=np.int64)
        rv = np.zeros(vt.shape, dtype=np.int64)
        nd, nv = [], []
        for e, sim in enumerate(sims):
            a, b = unit_kernels.ducts_of(sim), unit_kernels.vents_of(sim)
            rd[e, :len(a)] = a
            rv[e, :len(b)] = b
            nd.append(len(a))
            nv.append(len(b))
            if dirty:
                rd[e, len(a):] = a[0]            # garbage: copies of live rows
                rv[e, len(b):] = b[0]
        dt.upload(rd, np.array(nd))
        vt.upload(rv, np.array(nv))
        fields = _dev_planes([_planes(s.gmap) for s in sims])
        unit_kernels.sweep_vents(st, _applier_for(st), fields, dt, vt)
        return (dt.download()[0], vt.download()[0], {k: v.get() for k, v in fields.items()},
                nd, nv, rd, rv)
    clean, dirty = run(False), run(True)
    for e in range(2):
        nd, nv = clean[3][e], clean[4][e]
        if not np.array_equal(clean[0][e, :nd], dirty[0][e, :nd]) or \
                not np.array_equal(clean[1][e, :nv], dirty[1][e, :nv]):
            fail(f"padding env {e}: garbage past a table's count changed the valid rows")
        if not np.array_equal(dirty[0][e, nd:], dirty[5][e, nd:]) or \
                not np.array_equal(dirty[1][e, nv:], dirty[6][e, nv:]):
            fail(f"padding env {e}: a padding row was written")
        for k in applier.PLANE_NAMES:
            if not np.array_equal(clean[2][k][e], dirty[2][k][e]):
                fail(f"padding env {e}: garbage rows changed plane {k}")
                break
    if clean[3][0] == clean[3][1]:
        fail("padding: the two envs hold the same duct count (vacuous)")
    print(f"padding: garbage duct / vent rows past each table's count (duct counts "
          f"{clean[3]}, vent counts {clean[4]}) neither read nor written")


# ---------------------------------------------------------------------------
# PART OVERFLOW
# ---------------------------------------------------------------------------
def part_overflow():
    sim = VentScene(0).sim
    seen = Counter()
    for name, table, rows in (("duct", DT, unit_kernels.ducts_of(sim)),
                              ("vent", VT, unit_kernels.vents_of(sim))):
        small = ot.DeviceTable(table, 2, cap=len(rows) - 1, bp=bp)
        try:
            small.stage_lists([rows[:1], rows])
            fail(f"overflow: a full {name} table took {len(rows)} rows")
        except ot.ObjectTableOverflow as ex:
            if f"'{name}'" not in str(ex) or "env 1" not in str(ex):
                fail(f"overflow: the {name} refusal does not name the table and the env: {ex}")
            seen[name] += 1
        many = list(sim._ducts if name == "duct" else sim._vents)
        many = many * (table.capacity // len(many) + 1)
        try:
            ot.pack(table, many[:table.capacity + 1])
            fail(f"overflow: a {name} list past the capacity was packed")
        except ot.ObjectTableOverflow as ex:
            if f"'{name}'" not in str(ex):
                fail(f"overflow: the {name} list's refusal does not name it: {ex}")
            seen[f"{name} list"] += 1
    print(f"overflow: the duct and vent tables past their capacities refused by name "
          f"{dict(seen)}")


# ---------------------------------------------------------------------------
# PART CHAINED
# ---------------------------------------------------------------------------
_VENT_GE = {"ge_pump_extract": "pump_extract", "ge_pump_inject_vec": "pump_inject_vec",
            "ge_pump_rail": "pump_rail"}


def _host_side_vents(ch):
    """Slot 9e(d) vents' host-only side after the device ran it: the books
    landed (the sweep changes no tile: nothing else)."""
    def hs(sim, _pre, _post):
        g = sim.gmap
        for name, key in _VENT_GE.items():
            g._gas_energy_book(key, ch.books.get(name, 0))
        for s in range(N_GASES):
            g.trace_clip_books[s] += ch.books.get(f"trace_clip_{s}", 0)
    return hs


def _chain(name, make, ticks, with_doors=True):
    from cuda_unit_doors_check import SLOT_9ED, _host_side_9ed
    from cuda_unit_topology_check import _same_twin
    from simulation import simulation as simmod
    from simulation.entities.serialize import entity_records
    twin, ref = make(), make()
    calls = Counter()
    o_v, o_d = simmod.sweep_vents, simmod.sweep_doors

    def v_count(sim, __o=o_v):
        calls["twin vents" if sim is twin.sim else "reference vents"] += 1
        return __o(sim)

    def d_count(sim, __o=o_d):
        calls["twin doors" if sim is twin.sim else "reference doors"] += 1
        return __o(sim)
    simmod.sweep_vents, simmod.sweep_doors = v_count, d_count
    t0 = time.perf_counter()
    n = acted = 0
    holder = {}
    chv = ChainedSlots(twin, [SLOT_VENTS], bp=bp, host_side=lambda s, a, b: holder["v"](s, a, b))
    holder["v"] = _host_side_vents(chv)
    chs = [chv]
    if with_doors:
        chd = ChainedSlots(twin, [SLOT_9ED], bp=bp,
                           host_side=lambda s, a, b: holder["d"](s, a, b))
        holder["d"] = _host_side_9ed(chd)
        chs.append(chd)
    try:
        for c in chs:
            c.install()
        for _ in range(ticks):
            if getattr(twin, "is_terminal", lambda: False)():
                break
            twin.tick()
            ref.tick()
            n += 1
            why = _same_twin(twin.sim, ref.sim)
            for nm, f in (("the duct table", unit_kernels.ducts_of),
                          ("the vent table", unit_kernels.vents_of),
                          ("the door table", unit_kernels.doors_of)):
                if why is None and f(twin.sim).tolist() != f(ref.sim).tolist():
                    why = nm
            if why is None and entity_records(twin.sim.entities) != \
                    entity_records(ref.sim.entities):
                why = "the entity section"
            if why is not None:
                fail(f"chained {name}: the twin left the reference at tick {twin.sim.tick} "
                     f"({why})")
                break
            acted += int(chv.books.get("applied", 0) > 0) if chv.books else 0
            chv.books = {}
    finally:
        for c in reversed(chs):
            c.uninstall()
        simmod.sweep_vents, simmod.sweep_doors = o_v, o_d
    if calls["twin vents"] or calls["twin doors"]:
        fail(f"chained {name}: the host law ran in the twin {dict(calls)}")
    print(f"chained {name}: {n} ticks, the twin's unit hash, field digest, planes, books, "
          f"duct / vent / door tables and entity section equal the reference every tick "
          f"({' + '.join(str(c.runs) for c in chs)} chain runs, the vent sweep moved gas on "
          f"{acted} ticks; host law calls {dict(calls)}; {time.perf_counter() - t0:.0f} s)")
    return chs, n, acted, calls


def part_chained():
    from cuda_unit_doors_check import DoorScene
    from cuda_unit_objects_check import ShipDriver
    for v in (0, 1):
        chs, n, acted, calls = _chain(f"vent scene v{v} (9e(d) vents + doors)",
                                      lambda: VentScene(v), CHAIN_TICKS)
        if chs[0].runs != n or chs[1].runs != n or acted == 0 or \
                not calls["reference vents"]:
            fail(f"chained vent scene v{v}: {chs[0].runs} / {chs[1].runs} runs in {n} ticks, "
                 f"{acted} acted (vacuous or skipped)")
    chs, n, acted, calls = _chain("door scene (9e(d) vents + doors; no duct)",
                                  lambda: DoorScene(0), 80)
    if chs[1].runs != n or chs[0].runs != 0:
        fail(f"chained door scene: {chs[0].runs} vent / {chs[1].runs} door runs in {n} ticks")
    chs, n, acted, calls = _chain("ship_demo bots match (9e(d) vents)", ShipDriver, SHIP_TICKS,
                                  with_doors=False)
    if chs[0].runs != n or acted == 0 or calls["twin vents"]:
        fail(f"chained ship: {chs[0].runs} runs in {n} ticks, {acted} acted")


# ---------------------------------------------------------------------------
PARTS = ("tables", "kit", "probe", "scene", "levels", "refusal", "batch", "padding",
         "overflow", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_VENTS_RESULT: FAIL (not the CUDA build)")
        return 1
    gate_config()
    parts = parts or os.environ.get("UNIT_VENTS_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "kit": part_kit, "probe": part_probe,
             "scene": part_scene, "levels": part_levels, "refusal": part_refusal,
             "batch": part_batch, "padding": part_padding, "overflow": part_overflow,
             "chained": part_chained}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
        sys.stdout.flush()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_VENTS_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table (median ms over ``reps``): the 9e(d) vent sweep for N =
    1 / 8 / 64 copies of ship_demo (1 duct, 2 vents) and of the vent scene
    (6 ducts, 16 vents) -- the launch, its books' D2H and the Python face
    (each rep re-stages the tables and the planes, not timed)."""
    from cuda_unit_objects_check import ShipDriver
    gate_config()
    ship = ShipDriver()
    for _ in range(30):
        ship.tick()
    sc = VentScene(0)
    for _ in range(30):
        sc.tick()
    print("TIMING (median ms over %d reps)" % reps)
    print("| op | N=1 | N=8 | N=64 |")
    for label, sim in (("9e(d) vents, ship_demo (1 duct, 2 vents)", ship.sim),
                       ("9e(d) vents, the vent scene (6 ducts, 16 vents)", sc.sim)):
        row = []
        for n in (1, 8, 64):
            sims = [sim] * n
            st = _store(sims)
            planes = _planes(sim.gmap)
            dt, vt = unit_kernels.vent_tables(n, bp=bp)
            dr, vr = unit_kernels.ducts_of(sim), unit_kernels.vents_of(sim)
            ts = []
            for _ in range(reps):
                dt.stage_lists([dr] * n)
                vt.stage_lists([vr] * n)
                fields = _dev_planes([planes] * n)
                s = time.perf_counter()
                unit_kernels.sweep_vents(st, _applier_for(st), fields, dt, vt)
                ts.append(time.perf_counter() - s)
            row.append(float(np.median(ts)))
        print(f"| {label} | " + " | ".join(f"{x * 1e3:.3f}" for x in row) + " |")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        sys.exit(bench())
    sys.exit(main())
