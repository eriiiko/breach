"""#29 P6.4b -- THE TOPOLOGY GATE: the per-env serial applier, the topology
seam as device functions (``on_tile_changed``, ``destroy_wall``), slot 9 (the
fire's burn-through walls) and the tick-end clears, each against its Python
law at tol 0 on every plane it writes AND every book.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_topology.py). The applier is ``simulation.applier``'s
``DeviceApplier`` (C++ ``cpp/src/cuda_applier.cu`` over
``cpp/src/cuda_topology.cuh``), its Python face ``simulation.unit_kernels.
apply_effects`` / ``burn_through`` / ``tick_end_clears``; the harness
``tests/cuda_unit_lockstep.py`` with the FIELD OUTPUTS of #29 P6.4b
(``DeviceSlot(..., outputs=, host_only=)``: every declared plane compared tile
for tile, every book against the host's delta over the slot, and the
WRITE-SET gate -- every GameMap attribute the host slot changed is an output
or host-only with its reason).

PART TABLES -- every P6.4b table row read back from the device equals an
independent read of the env's own GameMap (its materials table, its ambient
seed, its boundary), per env of a batch of two maps (space and planetside,
two tile sizes).

PART SEAM -- the device functions on FIXTURE maps against the Python, plane
for plane and book for book, one destroy list per map (space at 1/3 m,
planetside at 1/2 m, space at 1 m), forward and REVERSED: an interior slab's
centre (the ambient fallback), a wall between donors of unequal composition
(the inherited fraction and its rounding), edge hulls, a tile exposing the
boundary through a neighbour, a CHAIN (a plug exposed only through the hull
tile destroyed before it: the two orders give different maps, both equal on
the device), furniture in a compressed hot pocket (a negative seed booking)
and beside a breach (the evacuation), a burning wood tile, both door
materials, every material row, out-of-bounds and air no-ops; the census of
every branch must be non-zero. After each run the energy books close on the
DEVICE's planes as on the host's (the device twin of
tests/test_gas_energy_tile_flip.py, design §5.7): Δ Σ_accountable gas_energy
== the seam's channels, Δ Σ bulk N == n_destruction_seed, Δ Σ cap_real·T ==
solid_destroy_wall. And ``on_tile_changed`` alone (a TEST-ONLY probe:
``material = m; on_tile_changed``): the membership retire (gas -> hull), the
born at ambient (hull -> air), no change (furniture -> wood), every material
as a target.

PART SCENES -- the SHADOW LOCKSTEP of slot 9 and the tick-end clears on a
fire scene built for the gate (walls burning through on many ticks, a
two-tile hull plug breaching to space in one tick, adjacent tiles in one
tick, a door of each material in slot 7's list -- the shipped doors are not
flammable, so the scene appends them, said here), the destroyed count equal
to the host's and > 0; plus ship_demo's bots match (seed 1) for the clears
(its burn-through count printed).

PART BATCH -- N = 2 and 8 fire scenes that differ in layout, tile size,
boundary, seed, ignition and schedule (topology events on ticks where other
envs have none): env i of the batch == env i alone, planes and books.

PART PADDING -- an env with an empty list beside full ones is untouched (the
ticks counted); the clears' env mask leaves a masked env untouched.

PART OVERFLOW -- more effects than the list holds: the first ``cap`` applied
(the device's planes equal a host that destroyed only those), the rest
counted in ``dropped``, the Python face refuses the run by name.

PART CHAINED -- slot 9 replaced by the device's in a twin fire match (the
planes written back, the books landed on the host, the host-only side done
by the caller: the render marks, the sky flag, the serial, the events), and
the clears alone the same way: per-tick field digest, unit hash, every
topology plane and every book equal an untouched reference, long enough that
physics ran on the patched planes.

PROPERTY: the topology seam's device functions are GameMap.on_tile_changed
and destroy_wall integer for integer on every plane and book, serially in the
list's order, per env; slot 9 and the tick-end clears equal their Python
slots. BREAKS IF the seed's rounding, the donor predicate, the breach rule's
read of an earlier destroy, the atmosphere pin, T or fire's clear, a
neighbour's facing face entry, the membership retire, a booking's sign, any
cache patch, the list's order, the env rebase or a cleared plane differs from
the reference. Each validated by breaking it (the patch's report).

Prints per-part lines and ``UNIT_TOPOLOGY_RESULT: PASS`` / ``FAIL``; exits
0 / 1. ``bench()`` prints the TIMING table.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT / "tools") not in sys.path:
    sys.path.append(str(_ROOT / "tools"))   # appended: never ahead of the CUDA build

import level_loader  # noqa: E402
from cuda_unit_lockstep import (ALIASES, ChainedSlots, DeviceSlot,  # noqa: E402
                                ShadowLockstep, changed_attrs, gmap_state,
                                write_set_violations)
from simulation import (applier, unit_kernels, unit_store,  # noqa: E402
                        unit_tables)
from simulation.materials import (MAT_AIR, MAT_DOOR, MAT_DOOR_CLOSED,  # noqa: E402
                                  MAT_FURNITURE, MAT_GRILLE, MAT_HULL,
                                  MAT_WOOD, MATERIAL_NAMES)

FAILS: list = []
CAP = int(os.environ.get("UNIT_TOPOLOGY_CAP", "256"))
FIRE_TICKS = int(os.environ.get("UNIT_TOPOLOGY_FIRE_TICKS", "90"))
SHIP_TICKS = int(os.environ.get("UNIT_TOPOLOGY_SHIP_TICKS", "500"))
BATCH_TICKS = int(os.environ.get("UNIT_TOPOLOGY_BATCH_TICKS", "70"))
CHAIN_TICKS = int(os.environ.get("UNIT_TOPOLOGY_CHAIN_TICKS", "160"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_TOPOLOGY_BATCH_NS", "2,8").split(",")]


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
PLANE_OUTPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)
SLOT9_BOOKS = tuple("book:" + b for b in (
    "n_destruction_seed", "ge_retire", "ge_mint", "ge_destroy_seed",
    "solid_destroy_wall", "tiles_changed"))
SLOT9_HOST_ONLY = (
    ("render_dirty", "the wall seam's render mark (the renderer drains it; "
     "no device form)"),
    ("_sky_mask_dirty", "the sky exchange's lazy-rebuild flag (ambient maps; "
     "the sky bracket is outside the device backend, §4.3)"),
    ("_gas_t_amb_raw_cache", "a lazily filled cache of a config constant "
     "(the device reads the table row topo_t_amb_q)"),
)
CLEAR_OUTPUTS = tuple("field:" + n for n in applier.CLEAR_NAMES)
CLEARS_HOST_ONLY = (
    ("damage_credit", "the credit claims' clear: their device form is the "
     "claims table (P7)"),
)

_APPLIERS: dict = {}


def _applier_for(store):
    n = store.shape[0]
    a = _APPLIERS.get(n)
    if a is None:
        a = _APPLIERS[n] = applier.DeviceApplier(n, CAP, bp=bp)
    return a


def _run9(st, inp):
    return unit_kernels.burn_through(st, _applier_for(st), inp.fields,
                                     inp.tables["burn_walls"])


def _run_clears(st, inp):
    import cupy as cp
    n = st.shape[0]
    cov = np.array([sum(int(cp.count_nonzero(inp.fields[c][e]))
                        for c in applier.CLEAR_NAMES) for e in range(n)],
                   dtype=np.int64)
    unit_kernels.tick_end_clears(bp, {c: inp.fields[c] for c in applier.CLEAR_NAMES})
    return cov


SLOT_9 = DeviceSlot(label="9 burn-through walls", run=_run9,
                    inputs=("burn_walls",), outputs=PLANE_OUTPUTS + SLOT9_BOOKS,
                    host_only=SLOT9_HOST_ONLY)
SLOT_CLEARS = DeviceSlot(label="tick-end clears", run=_run_clears,
                         outputs=CLEAR_OUTPUTS, host_only=CLEARS_HOST_ONLY)


# ---------------------------------------------------------------------------
# Helpers: levels, planes, the host's books
# ---------------------------------------------------------------------------
SPACE = level_loader.SPACE_CODE


def _level(tm, name, ts, boundary):
    from level_loader import LevelData
    return LevelData(name=name, version="3", path=Path("."), tilemap=tm,
                     tile_size_m=ts, diffuse_path=Path("."), boundary=boundary)


def _sim(level, seed=7):
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    sim = Simulation(level, seed=seed, breach_physics=bp, enable_recorder=False,
                     ruleset=ContinuousRealtime())
    sim.set_paused(False)
    return sim


def _add_unit(sim, x, y, name="U0"):
    from simulation.unit import Unit
    u = Unit(name, x=float(x), y=float(y), team=0)
    sim.add_unit(u)
    return u


def _planes(gmap) -> dict:
    return {n: np.array(getattr(gmap, n), copy=True) for n in applier.PLANE_NAMES}


def _dev_planes(per_env: list) -> dict:
    import cupy as cp
    return {n: cp.asarray(np.ascontiguousarray(np.stack([p[n] for p in per_env])))
            for n in applier.PLANE_NAMES}


def _books(gmap) -> dict:
    return {b.name: applier.host_book(gmap, b.host) for b in applier.BOOKS
            if b.host is not None}


def _compare_planes(tag, dev: dict, want: list) -> bool:
    """Device ``(N, ...)`` planes against N host plane dicts, bit for bit."""
    from cuda_unit_lockstep import first_field_diff
    for n in applier.PLANE_NAMES:
        got = dev[n].get()
        w = np.stack([p[n] for p in want])
        fd = first_field_diff(n, got, w)
        if fd is not None:
            fail(f"{tag}: plane {n} env {fd[0]} at {fd[1]}: host {fd[2]} device {fd[3]}")
            return False
    return True


def _cap_real(shift):
    s = np.clip(shift.astype(np.int64), -16, 30)
    return np.left_shift(np.int64(1), s + 16)


def _closure(tag, pre: dict, post: dict, books: dict, e: int = 0):
    """The energy and mass books close on ``post`` (a plane dict, the
    device's or the host's) against ``pre``: the device twin of
    test_gas_energy_tile_flip's identity (design §5.7)."""
    def acct(P):
        return ~P["solid"] & ~P["thermal_solid"] & ~P["is_vacuum"] & ~P["is_ambient"]

    def e_sum(P):
        return int(P["gas_energy"][acct(P)].astype(object).sum())

    def n_sum(P):
        from simulation.gases import INERT_N2, O2
        return int(P["gas"][O2].astype(np.int64).sum()
                   + P["gas"][INERT_N2].astype(np.int64).sum())

    def s_sum(P):
        ts = P["thermal_solid"]
        return int((_cap_real(P["heat_inv_shift"][ts]).astype(object)
                    * P["temperature"][ts].astype(object)).sum())

    de = e_sum(post) - e_sum(pre)
    seam = int(books["ge_retire"][e] + books["ge_mint"][e] + books["ge_destroy_seed"][e])
    ok = True
    if de != seam:
        fail(f"{tag}: d sum_accountable gas_energy {de} != the seam's channels {seam}")
        ok = False
    dn = n_sum(post) - n_sum(pre)
    if dn != int(books["n_destruction_seed"][e]):
        fail(f"{tag}: d sum bulk N {dn} != n_destruction_seed "
             f"{int(books['n_destruction_seed'][e])}")
        ok = False
    ds = s_sum(post) - s_sum(pre)
    if ds != int(books["solid_destroy_wall"][e]):
        fail(f"{tag}: d sum solid books {ds} != solid_destroy_wall "
             f"{int(books['solid_destroy_wall'][e])}")
        ok = False
    return ok


def _store_for(sims):
    return unit_store.DeviceStore(len(sims), max([1] + [len(s.units) for s in sims]),
                                  bp=bp, tables=unit_tables.values_batch(sims))


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def table_reference(sim) -> dict:
    """An INDEPENDENT read of every P6.4b table row from the env's GameMap
    (numpy's own float32 bit views, the materials module's ids), for
    cuda_unit_kernels_check's every-row TABLES gate too."""
    from simulation import materials as M
    from simulation import wall_fixed
    g = sim.gmap
    t = g.materials
    nm = len(MATERIAL_NAMES)
    bits = lambda a: tuple(int(x) for x in  # noqa: E731
                           np.asarray(a, dtype=np.float32).reshape(-1).view(np.int32))
    n_total, o2, _n2, pin = g.ambient_seed()
    return {
        "topo_mat_air": M.MAT_AIR, "topo_mat_hull": M.MAT_HULL,
        "topo_boundary_ambient": 1 if g._boundary == "ambient" else 0,
        "topo_amb_n_total_q": int(n_total), "topo_amb_o2_q": int(o2),
        "topo_amb_pin_q": int(pin), "topo_t_amb_q": int(g.gas_t_amb_raw()),
        "mat_light_atten_bits": bits(t.light_atten[:nm]),
        "mat_heat_atten_bits": bits(t.heat_atten[:nm]),
        "mat_heat_atten_q": tuple(int(x) for x in t.heat_atten_q16[:nm]),
        "mat_light_atten_q": tuple(int(x) for x in np.asarray(t.light_atten_q16[:nm]).reshape(-1)),
        "mat_flammable": tuple(int(x) for x in t.flammable[:nm]),
        "mat_wall_hp_q": tuple(int(wall_fixed.quantize(np.asarray([float(h)]))[0])
                               for h in t.hp[:nm]),
        "mat_conductivity_bits": bits(t.conductivity[:nm]),
        "mat_heat_inv_shift": tuple(int(x) for x in t.heat_inv_shift[:nm]),
        "mat_thermal_solid": tuple(int(x) for x in t.thermal_solid[:nm]),
        "mat_fuel_recip": tuple(int(x) for x in t.fuel_recip[:nm]),
        "mat_fire_t_ext_q": tuple(int(x) for x in t.fire_T_ext_q16[:nm]),
        "mat_fuel_per_o2_q": tuple(int(x) for x in t.fuel_per_o2_q16[:nm]),
        "mat_face_shift": tuple(int(x) for x in np.asarray(t.face_shift_table)[:nm, :nm].reshape(-1)),
        "mat_no_face": int(t.no_face),
        "mat_permeability_bits": bits(t.permeability[:nm]),
        "mat_wave_absorb_bits": bits(t.wave_absorb[:nm]),
        "mat_solid": tuple(int(x) for x in (np.asarray(t.permeability[:nm], dtype=np.float32) <= 0)),
        "mat_blocks_sight": tuple(int(x) for x in t.blocks_sight[:nm]),
    }


def part_tables():
    a = _sim(seam_level("space", 1.0 / 3.0))
    b = _sim(seam_level("ambient", 0.5))
    st = _store_for([a, b])
    got = st.tables.download()
    n = 0
    for e, sim in enumerate((a, b)):
        for k, v in table_reference(sim).items():
            dv = unit_tables.read(got[e], k)
            n += 1
            as_t = lambda x: tuple(x) if isinstance(x, (tuple, list)) else (x,)  # noqa: E731
            if as_t(dv) != as_t(v):
                fail(f"tables env {e} row {k}: device {dv} != reference {v}")
    if unit_tables.read(got[0], "topo_boundary_ambient") == \
            unit_tables.read(got[1], "topo_boundary_ambient"):
        fail("tables: a space map and a planetside map gave one boundary row")
    print(f"tables: {n} P6.4b rows over 2 envs (space 1/3 m, planetside 1/2 m) equal "
          f"the reference (T {unit_tables.layout().width})")


# ---------------------------------------------------------------------------
# PART SEAM
# ---------------------------------------------------------------------------
SH, SW = 24, 36


def seam_level(boundary, ts):
    """The seam's fixture map: a SPACE ring on three sides and an EDGE HULL row
    at the bottom (the map's last row), a hull box, interior air, and the
    features of :func:`seam_list`."""
    tm = np.full((SH, SW), SPACE, dtype=np.int32)
    tm[1:SH, 1:SW - 1] = MAT_HULL
    tm[2:SH - 2, 2:SW - 3 + 1] = MAT_AIR
    tm[SH - 1, :] = MAT_HULL                     # the edge hull row
    tm[4:7, 4:7] = MAT_HULL                      # an interior slab
    tm[4, 10] = MAT_HULL                         # a wall between donors
    tm[2, 24] = MAT_HULL                         # the chain's plug under (1, 24)
    tm[9, 5] = MAT_FURNITURE                     # in a compressed hot pocket
    tm[1, 28] = SPACE                            # a breach in the top hull
    tm[2, 28] = MAT_FURNITURE                    # beside it
    tm[12, 5] = MAT_WOOD                         # burning
    tm[12, 9] = MAT_DOOR
    tm[12, 11] = MAT_DOOR_CLOSED
    for i, m in enumerate(sorted(MATERIAL_NAMES)):   # every material row
        if m != MAT_AIR:
            tm[16, 3 + 2 * i] = m
    tm[19, 14:17] = MAT_GRILLE                   # a gas tile among gas tiles
    tm[1, 31] = MAT_GRILLE                       # a hot gas tile in the hull, by space
    return _level(tm, f"p64b_seam_{boundary}", ts, boundary)


def seam_list():
    """The destroy list (row, col), in order."""
    every = [(16, 3 + 2 * i) for i, m in enumerate(sorted(MATERIAL_NAMES)) if m != MAT_AIR]
    return ([(5, 5), (4, 10), (SH - 1, 15), (SH - 1, 16), (1, 20), (1, 24), (2, 24),
             (9, 5), (-1, 3), (2, 28), (12, 5), (12, 9), (12, 11), (SH, 2), (8, 20)]
            + every + [(19, 15), (3, SW), (1, 31), (5, 5)])


def seam_prep(sim, variant=0):
    """The fixtures: a hot interior, a compressed hot pocket around the
    furniture, donors of unequal composition, a burning hot wood tile, hot
    thermal solids (non-zero solid books)."""
    from simulation import atmosphere_fixed as afx
    from simulation.gases import INERT_N2, O2
    g = sim.gmap
    gas_sel = g._gas_energy_accountable()
    g.seed_gas_temperature(gas_sel, g.temperature[gas_sel]
                           + afx.quantize_scalar(150.0 + 40.0 * variant))
    # the compressed pocket (bulk written directly, then the energy reseeded)
    pock = np.s_[8:11, 4:7]
    g.gas[O2][pock] = (g.gas[O2][pock].astype(np.int64) * 5 // 2).astype(np.int32)
    g.gas[INERT_N2][pock] = (g.gas[INERT_N2][pock].astype(np.int64) * 5 // 2).astype(np.int32)
    g.gas[O2][9, 5] = 30000 + 777 * variant
    g.gas[INERT_N2][9, 5] = 100000
    g.seed_gas_temperature(np.s_[8:11, 4:7], g.temperature[8:11, 4:7]
                           + afx.quantize_scalar(400.0))
    g.reseed_gas_energy(np.s_[8:11, 4:7])
    # donors of unequal composition around (4, 10)
    for (y, x), (o2, n2) in zip(((3, 10), (5, 10), (4, 9), (4, 11)),
                                ((9000, 70001), (21000, 40000), (2, 99999), (65000, 1))):
        g.gas[O2][y, x] = o2 + variant
        g.gas[INERT_N2][y, x] = n2
    g.reseed_gas_energy(np.s_[3:6, 9:12])
    # stray gas on a SOLID neighbour of the slab's centre: the donor predicate
    # must not read it (a non-solid vacuum / ambient neighbour would make the
    # tile a breach, so a solid tile holding gas is the one way the predicate
    # shows); it is outside the books and no destroy touches it
    g.gas[O2][4, 5] = 5000
    g.gas[INERT_N2][4, 5] = 20000
    # the furniture by the breach holds gas
    g.gas[O2][2, 28] = 14000
    g.gas[INERT_N2][2, 28] = 52000
    # a burning wood tile; hot thermal solids
    g.fire[12, 5] = 30000
    for (y, x) in zip(*np.nonzero(g.thermal_solid)):
        g.temperature[y, x] = afx.quantize_scalar(50.0 + (3 * y + x) % 17 * 13.0)
    g.temperature[12, 5] = afx.quantize_scalar(650.0)


def census_one(g, t, done, counts):
    """The branch flags of ONE host destroy (read before it), counted."""
    y, x = t
    h, w = g.material.shape
    if not (0 <= y < h and 0 <= x < w):
        counts["noop_oob"] += 1
        return
    m = int(g.material[y, x])
    if m == MAT_AIR:
        counts["noop_air"] += 1
        return
    counts[f"mat_{m}"] += 1
    bm = g.is_ambient if g._boundary == "ambient" else g.is_vacuum
    nbs = [(y + dy, x + dx) for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1))
           if 0 <= y + dy < h and 0 <= x + dx < w]
    edge = m == MAT_HULL and (y < 1 or y >= h - 1 or x < 1 or x >= w - 1)
    exposes = any(bm[n] and not g.solid[n] for n in nbs)
    breach = bool(bm[y, x]) or edge or exposes
    counts["edge_hull"] += int(edge)
    counts["exposes"] += int(exposes and not edge)
    counts["breach"] += int(breach)
    if any(n in done for n in nbs):
        counts["chain"] += 1
    if int(g.fire[y, x]) > 0:
        counts["burning"] += 1
    if g.thermal_solid[y, x] and int(g.temperature[y, x]) != 0:
        counts["thermal_hot"] += 1
    from simulation.gases import INERT_N2, O2
    if breach:
        if int(g.gas[O2][y, x]) + int(g.gas[INERT_N2][y, x]) > 0:
            counts["evacuation"] += 1
        if g.gas_is_accountable(y, x) and int(g.temperature[y, x]) != 0:
            counts["breach_hot_gas"] += 1      # only the explicit T = 0 cools it
        return
    don = [n for n in nbs if not g.solid[n] and not g.is_vacuum[n]]
    so = sum(int(g.gas[O2][n]) for n in don)
    sn = sum(int(g.gas[O2][n]) + int(g.gas[INERT_N2][n]) for n in don)
    if sn == 0:
        counts["fallback"] += 1
        return
    fr = {(int(g.gas[O2][n]), int(g.gas[O2][n]) + int(g.gas[INERT_N2][n])) for n in don}
    fr = {(a, b) for a, b in fr if b > 0}
    if len({a * 10 ** 12 // b for a, b in fr}) > 1:
        counts["unequal_donors"] += 1
    if (g.ambient_seed()[0] * so) % sn:
        counts["rounding"] += 1


def _seam_run(tag, boundary, ts, order, counts, variant=0):
    """One destroy list on a fixture map: host and device, compared."""
    sim = _sim(seam_level(boundary, ts), seed=11 + variant)
    _add_unit(sim, 20.0, 9.0)
    seam_prep(sim, variant)
    g = sim.gmap
    tiles = seam_list() if order == "forward" else list(reversed(seam_list()))
    pre = _planes(g)
    b0 = _books(g)
    ws0 = gmap_state(g)
    st = _store_for([sim])
    dev = _dev_planes([pre])
    done = set()
    for t in tiles:
        n0 = g.n_destruction_seed_sum
        census_one(g, t, done, counts)
        g.destroy_wall(*t)
        if g.n_destruction_seed_sum - n0 < 0:
            counts["negative_seed"] += 1
        done.add(t)
    post = _planes(g)
    b1 = _books(g)
    ws = changed_attrs(ws0, gmap_state(g))
    books = unit_kernels.apply_effects(st, _applier_for(st), dev,
                                       [applier.destroy_records(tiles)])
    ok = _compare_planes(tag, dev, [post])
    for k in b0:
        if int(books[k][0]) != b1[k] - b0[k]:
            fail(f"{tag}: book {k} device {int(books[k][0])} != host {b1[k] - b0[k]}")
            ok = False
    if int(books["applied"][0]) != b1["tiles_changed"] - b0["tiles_changed"]:
        fail(f"{tag}: applied {int(books['applied'][0])} != the host's destroys")
        ok = False
    bad = write_set_violations(SLOT_9, ws, ws0[ALIASES])
    if bad:
        fail(f"{tag}: write-set: {bad} changed on the host, neither an output nor "
             f"host-only")
        ok = False
    dev_post = {n: dev[n].get()[0] for n in applier.PLANE_NAMES}
    ok &= _closure(tag + " (device)", pre, dev_post, books)
    hb = {k: np.array([b1[k] - b0[k]]) for k in b0}
    ok &= _closure(tag + " (host)", pre, post, hb)
    counts[f"map_{boundary}"] += 1
    counts[f"ts_{ts:.3f}"] += 1
    return ok, post


def _probe_run(tag, boundary, ts, counts):
    """on_tile_changed alone: material = m; on_tile_changed, host vs device."""
    sim = _sim(seam_level(boundary, ts), seed=5)
    _add_unit(sim, 20.0, 9.0)
    seam_prep(sim)
    g = sim.gmap
    recs = [(3, 3, MAT_HULL), (3, 3, MAT_AIR), (9, 5, MAT_WOOD), (3, 14, MAT_GRILLE),
            (3, 14, MAT_AIR), (7, 20, MAT_FURNITURE), (7, 20, MAT_AIR)]
    for i, m in enumerate(sorted(MATERIAL_NAMES)):
        recs.append((10, 12 + i, m))
        recs.append((11, 12 + i, (m + 3) % len(MATERIAL_NAMES)))
    pre = _planes(g)
    b0 = _books(g)
    st = _store_for([sim])
    dev = _dev_planes([pre])
    for (y, x, m) in recs:
        was = g.gas_is_accountable(y, x)
        g.material[y, x] = m
        g.on_tile_changed(y, x)
        now = g.gas_is_accountable(y, x)
        counts["retire" if was and not now else "born" if now and not was else
               "no_membership_change"] += 1
    post = _planes(g)
    b1 = _books(g)
    books = unit_kernels.topology_probe(st, _applier_for(st), dev, [np.array(recs)])
    ok = _compare_planes(tag, dev, [post])
    for k in ("ge_retire", "ge_mint", "tiles_changed"):
        if int(books[k][0]) != b1[k] - b0[k]:
            fail(f"{tag}: book {k} device {int(books[k][0])} != host {b1[k] - b0[k]}")
            ok = False
    if int(books["faults"][0]) or int(books["applied"][0]) != len(recs):
        fail(f"{tag}: probe applied {int(books['applied'][0])} of {len(recs)}, "
             f"faults {int(books['faults'][0])}")
        ok = False
    return ok


SEAM_MAPS = (("space", 1.0 / 3.0), ("ambient", 0.5), ("space", 1.0))
SEAM_NEED = ("fallback", "unequal_donors", "rounding", "edge_hull", "exposes",
             "chain", "negative_seed", "evacuation", "breach_hot_gas", "burning",
             "thermal_hot",
             "noop_oob", "noop_air", "map_space", "map_ambient", "retire", "born",
             "no_membership_change", "ts_0.333", "ts_0.500", "ts_1.000")


def part_seam():
    from collections import Counter
    counts = Counter()
    n_ok = 0
    for boundary, ts in SEAM_MAPS:
        fw = rv = None
        for order in ("forward", "reversed"):
            ok, post = _seam_run(f"seam {boundary} {ts:.3f} {order}", boundary, ts,
                                 order, counts)
            n_ok += int(ok)
            if order == "forward":
                fw = post
            else:
                rv = post
        if all(np.array_equal(fw[n], rv[n]) for n in applier.PLANE_NAMES):
            fail(f"seam {boundary} {ts:.3f}: the two orders gave one map (the chain "
                 f"case is vacuous)")
        n_ok += int(_probe_run(f"seam probe {boundary} {ts:.3f}", boundary, ts, counts))
    need = list(SEAM_NEED) + [f"mat_{m}" for m in sorted(MATERIAL_NAMES) if m != MAT_AIR]
    zero = [k for k in need if counts[k] == 0]
    if zero:
        fail(f"seam: branches never reached (vacuous): {zero}")
    print(f"seam: {n_ok} runs equal plane for plane and book for book, books "
          f"closed on the device; census {dict(sorted(counts.items()))}")


# ---------------------------------------------------------------------------
# The fire scene (slot 9)
# ---------------------------------------------------------------------------
FH, FW = 30, 40


class FireScene:
    """Walls burning through on many ticks: two wood partitions (a door of
    each material in them), a wood run in the top hull with a plug under it
    (the plug is exposed only through the hull tile above it, so the two burn
    in one tick and the second breaches through the first), random wood
    health (the natural burn), a schedule of tiles set to burn through on
    chosen ticks (adjacent tiles in one tick), and the two doors appended to
    slot 7's list on chosen ticks (no shipped door material is flammable).
    ``variant`` moves the layout, the seed, the ignition and the schedule;
    ``boundary`` / ``ts`` the map. Fixtures for the clears: rad_net / rad_amb /
    rad_flux written non-zero before each tick (nothing writes them since T6),
    so a dropped clear shows."""

    def __init__(self, variant=0, boundary="space", ts=1.0 / 3.0, extra_units=0):
        from simulation import atmosphere_fixed as afx
        from simulation import wall_fixed
        v = int(variant)
        self.v = v
        c1, c2 = 12 + v % 3, 25 + (v // 3) % 3
        tm = np.full((FH, FW), SPACE, dtype=np.int32)
        tm[1:FH - 1, 1:FW - 1] = MAT_HULL
        tm[2:FH - 2, 2:FW - 2] = MAT_AIR
        tm[2:FH - 2, c1] = MAT_WOOD
        tm[2:FH - 2, c2] = MAT_WOOD
        tm[1, 5:9] = MAT_WOOD
        tm[2, 6] = MAT_WOOD                      # the plug under (1, 6)
        self.doors = [(8 + v % 4, c1), (20 - v % 4, c2)]
        tm[self.doors[0]] = MAT_DOOR_CLOSED
        tm[self.doors[1]] = MAT_DOOR
        self.sim = _sim(_level(tm, f"p64b_fire_{v}", ts, boundary), seed=900 + v)
        for i in range(1 + extra_units):
            _add_unit(self.sim, 4.0 + 2 * i, 20.0, name=f"U{i}")
        g = self.sim.gmap
        rng = np.random.default_rng(31 + v)
        T = afx.quantize_scalar(700.0 - 25.0 * (v % 4))
        wood = list(zip(*np.nonzero(g.material == MAT_WOOD)))
        for (y, x) in wood:
            g.temperature[y, x] = T
            g.fire[y, x] = 40000
            g.wall_hp[y, x] = int(wall_fixed.quantize_scalar(
                float(rng.uniform(0.02, 3.5 + v % 3))))
        t0 = 3 + v % 5
        self.schedule = {
            t0: [(1, 6), (2, 6)],                       # the breach chain
            t0 + 4: [(13, c1), (14, c1), (15, c1)],    # adjacent, one tick
            t0 + 9: [(5, c2), (6, c2)],
        }
        for tiles in self.schedule.values():
            for t in tiles:
                g.wall_hp[t] = int(wall_fixed.quantize_scalar(200.0))
        self.appends = {t0 + 2: [self.doors[0]], t0 + 7: [self.doors[1]]}
        sim = self.sim
        orig = sim._step_physics

        def step_physics():
            out = list(orig())
            return out + [d for d in self.appends.get(sim.tick, ())
                          if int(sim.gmap.material[d]) != MAT_AIR]
        sim._step_physics = step_physics

    def tick(self):
        sim, g = self.sim, self.sim.gmap
        for t in self.schedule.get(sim.tick, ()):
            if int(g.material[t]) == MAT_WOOD:
                g.wall_hp[t] = 1
        k = sim.tick + 1 + self.v
        g.rad_net[k % FH, (3 * k) % FW] = 1000 + k
        g.rad_amb[(k + 5) % FH, k % FW] = -(2000 + k)
        g.rad_flux[(2 * k) % FH, (k + 7) % FW] = 3000 + k
        sim.step()


class ShipDriver:
    def __init__(self, seed=1):
        from simulation.match_session import MatchSession
        level = level_loader.load("ship_demo", str(_ROOT / "levels"))
        self.session = MatchSession(level, seed=seed, breach_physics=bp)
        self.sim = self.session.sim

    def tick(self):
        self.session.tick()

    def is_terminal(self):
        return self.session.is_terminal()


def _events_of(sim):
    from simulation.events import DoorDestroyedEvent, WallDestroyedEvent
    d = sum(1 for ev in sim.tick_events if isinstance(ev, DoorDestroyedEvent))
    w = sum(1 for ev in sim.tick_events if isinstance(ev, WallDestroyedEvent))
    return d, w


def _shadow(name, drivers, ticks, need_destroys=True, need_clears=True):
    """Slot 9 and the clears shadowed together; returns the tallies."""
    N = len(drivers)
    tally = dict(ticks=0, destroyed=0, ticks_destroying=0, multi=0, breach=0,
                 doors=0, clears=0, empty_beside_full=0)
    t0 = time.perf_counter()
    with ShadowLockstep(drivers, [SLOT_9, SLOT_CLEARS], bp=bp) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            reps = ls.tick()
            r9, rc = reps[SLOT_9.label], reps[SLOT_CLEARS.label]
            tally["ticks"] += 1
            if not (r9.equal and rc.equal):
                fail(f"{name}: {ls.divergences[-1]}")
                break
            cov = r9.coverage
            host = r9.host_books["tiles_changed"]
            if not np.array_equal(cov, host):
                fail(f"{name}: tick {r9.tick}: applied {cov.tolist()} != the host's "
                     f"destroys {host.tolist()}")
                break
            tally["destroyed"] += int(cov.sum())
            tally["ticks_destroying"] += int((cov > 0).any())
            tally["multi"] += int((cov > 1).any())
            tally["clears"] += int((rc.coverage > 0).any())
            if (cov > 0).any() and (cov == 0).any():
                tally["empty_beside_full"] += 1
            for e in range(N):
                ws = r9.write_set[e]
                if "is_vacuum" in ws or "is_ambient" in ws:
                    tally["breach"] += 1
                tally["doors"] += _events_of(drivers[e].sim)[0]
    tally["seconds"] = round(time.perf_counter() - t0)
    print(f"{name}: N {N}, {tally}")
    if need_destroys and (tally["destroyed"] == 0 or tally["multi"] == 0
                          or tally["breach"] == 0 or tally["doors"] == 0):
        fail(f"{name}: a slot-9 branch never acted (vacuous): {tally}")
    if need_clears and tally["clears"] == 0:
        fail(f"{name}: the clears never cleared anything (vacuous)")
    return tally


# ---------------------------------------------------------------------------
# PART SCENES / BATCH / PADDING / OVERFLOW / CHAINED
# ---------------------------------------------------------------------------
def part_scenes():
    _shadow("scene fire (space, 1/3 m)", [FireScene(0)], FIRE_TICKS)
    _shadow("scene fire (planetside, 1/2 m)", [FireScene(1, "ambient", 0.5)], FIRE_TICKS)
    t = _shadow("scene ship (bots, seed 1)", [ShipDriver(1)], SHIP_TICKS,
                need_destroys=False, need_clears=False)
    print(f"scene ship: {t['destroyed']} burn-throughs in {t['ticks']} ticks "
          f"(the clears acted on {t['clears']} ticks)")


def _batch_envs(n):
    out = []
    for i in range(n):
        boundary = "ambient" if i % 2 else "space"
        ts = (1.0 / 3.0, 0.5, 1.0)[i % 3]
        out.append(FireScene(i + 2, boundary, ts, extra_units=i % 3))
    return out


def part_batch():
    for n in BATCH_NS:
        t = _shadow(f"batch fire N={n}", _batch_envs(n), BATCH_TICKS)
        if t["empty_beside_full"] == 0:
            fail(f"batch N={n}: no tick had an env with no topology event beside one "
                 f"with some (§5.2)")


def part_padding():
    import cupy as cp
    # (1) an empty list beside full ones: counted in the batch scenes
    t = _shadow("padding fire N=3", [FireScene(7), FireScene(8, "ambient", 0.5),
                                     FireScene(9, "space", 1.0)], 40)
    if t["empty_beside_full"] == 0:
        fail("padding: no tick had an empty env beside a full one")
    # (2) the clears' env mask: a masked env untouched
    rng = np.random.default_rng(3)
    host = {c: np.ascontiguousarray(rng.integers(-1000, 1000, size=(3, 9, 11)).astype(d))
            for c, d in applier.CLEAR_PLANES}
    dev = {c: cp.asarray(v) for c, v in host.items()}
    active = cp.asarray(np.array([1, 0, 1], dtype=np.uint8))
    unit_kernels.tick_end_clears(bp, dev, active)
    for c in applier.CLEAR_NAMES:
        got = dev[c].get()
        if got[0].any() or got[2].any():
            fail(f"padding: the clears left plane {c} non-zero on an active env")
        if not np.array_equal(got[1], host[c][1]):
            fail(f"padding: the clears touched plane {c} on a masked env")
    print("padding: an empty env beside full ones untouched on "
          f"{t['empty_beside_full']} ticks; the clears' mask honoured")


def part_overflow():
    sim = _sim(seam_level("space", 1.0 / 3.0), seed=3)
    _add_unit(sim, 20.0, 9.0)
    seam_prep(sim)
    g = sim.gmap
    tiles = seam_list()
    cap = 6
    pre = _planes(g)
    for t in tiles[:cap]:
        g.destroy_wall(*t)
    want = _planes(g)
    st = _store_for([sim])
    ap = applier.DeviceApplier(1, cap, bp=bp)
    dev = _dev_planes([pre])
    try:
        unit_kernels.apply_effects(st, ap, dev, [applier.destroy_records(tiles)])
        fail("overflow: the Python face did not refuse a list past its capacity")
    except applier.EffectOverflow as ex:
        msg = str(ex)
        if "env(s) [0]" not in msg:
            fail(f"overflow: the refusal does not name the env: {msg}")
    books = st.last_books
    if int(books["dropped"][0]) != len(tiles) - cap:
        fail(f"overflow: dropped {int(books['dropped'][0])} != {len(tiles) - cap}")
    _compare_planes("overflow (the first cap applied)", dev, [want])
    print(f"overflow: {len(tiles)} effects into a list of {cap}: the first {cap} "
          f"applied, {int(books['dropped'][0])} counted, the run refused by name")


def _same_twin(a, b):
    from cuda_unit_damage_check import _field_digest, _unit_hash
    if _unit_hash(a) != _unit_hash(b):
        return "unit hash"
    if _field_digest(a) != _field_digest(b):
        return "field digest"
    for n in applier.PLANE_NAMES + applier.CLEAR_NAMES:
        if not np.array_equal(getattr(a.gmap, n), getattr(b.gmap, n)):
            return f"plane {n}"
    ba, bb = _books(a.gmap), _books(b.gmap)
    if ba != bb:
        return f"books {ba} != {bb}"
    if a.gmap.gas_energy_seam_net() != b.gmap.gas_energy_seam_net() or \
            a.gmap.solid_energy_seam_net() != b.gmap.solid_energy_seam_net():
        return "seam nets"
    if sorted(a.gmap.render_dirty) != sorted(b.gmap.render_dirty):
        return "render_dirty"
    return None


def _host_side_9(ch):
    """Slot 9's host-only side after the device ran it: the books landed
    (the solid book through _solid_energy_book, so the solver's sink moves
    too), the render marks, the sky flag, the serial, the events."""
    from simulation.events import DoorDestroyedEvent, WallDestroyedEvent

    def hs(sim, _pre, _post):
        g = sim.gmap
        b = ch.books
        g.n_destruction_seed_sum += b["n_destruction_seed"]
        for name, key in (("ge_retire", "retire"), ("ge_mint", "mint"),
                          ("ge_destroy_seed", "destroy_seed")):
            g._gas_energy_book(key, b[name])
        g._solid_energy_book("destroy_wall", b["solid_destroy_wall"])
        walls = ch.inputs.tables["burn_walls"][0]
        mat = ch.inputs._host_fields["material"][0].copy()
        h, w = mat.shape
        n = 0
        for (y, x, valid) in walls.tolist():
            if not valid:
                continue
            inb = 0 <= y < h and 0 <= x < w
            m = int(mat[y, x]) if inb else -1
            if inb and m != MAT_AIR:
                g._tile_serial += 1
                g._sky_mask_dirty = True
                g.render_dirty.add((int(y), int(x)))
                mat[y, x] = MAT_AIR
                n += 1
            sim.tick_events.append(DoorDestroyedEvent(pos=(y, x))
                                   if m in (MAT_DOOR, MAT_DOOR_CLOSED)
                                   else WallDestroyedEvent(pos=(y, x)))
        if n != b["tiles_changed"]:
            fail(f"chained 9: the host side counted {n} destroys, the device "
                 f"{b['tiles_changed']}")
    return hs


def _host_side_clears(sim, _pre, _post):
    from simulation.damage import tick_credit
    tick_credit(sim.gmap).clear()
    if sim.shots:
        sim.shots = [s for s in sim.shots if sim.real_time - s.time < s.duration]


def _chained(name, make, slot, host_side_of, ticks):
    twin, ref = make(), make()
    t0 = time.perf_counter()
    n = acted = 0
    holder = {}
    with ChainedSlots(twin, [slot], bp=bp,
                      host_side=lambda s, a, b: holder["hs"](s, a, b)) as ch:
        holder["hs"] = host_side_of(ch)
        for _ in range(ticks):
            twin.tick()
            ref.tick()
            n += 1
            why = _same_twin(twin.sim, ref.sim)
            if why is not None:
                fail(f"{name}: the chained match left the reference at tick "
                     f"{twin.sim.tick} ({why})")
                break
            acted += int(int(ch.coverage[-1][0][0]) > 0)
    print(f"{name}: {n} ticks, the chained twin equal to the reference every tick "
          f"(digests, every topology plane, the books; [{slot.label}] acted on "
          f"{acted} ticks; {time.perf_counter() - t0:.0f} s)")
    if ch.runs != n:
        fail(f"{name}: the chain ran {ch.runs} times in {n} ticks")
    if acted == 0:
        fail(f"{name}: [{slot.label}] never acted (vacuous)")


def part_chained():
    _chained("chained 9 (fire, space)", lambda: FireScene(4), SLOT_9, _host_side_9,
             CHAIN_TICKS)
    _chained("chained 9 (fire, planetside)", lambda: FireScene(5, "ambient", 0.5),
             SLOT_9, _host_side_9, CHAIN_TICKS)
    _chained("chained clears (fire)", lambda: FireScene(6), SLOT_CLEARS,
             lambda ch: _host_side_clears, CHAIN_TICKS)


# ---------------------------------------------------------------------------
# main / bench
# ---------------------------------------------------------------------------
PARTS = ("tables", "seam", "scenes", "batch", "padding", "overflow", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_TOPOLOGY_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("UNIT_TOPOLOGY_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "seam": part_seam, "scenes": part_scenes,
             "batch": part_batch, "padding": part_padding,
             "overflow": part_overflow, "chained": part_chained}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_TOPOLOGY_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 100) -> int:
    """The TIMING table: the applier (stage + the walk + the books' D2H) for
    1, 8 and 64 destroys per env at N = 1, 8, 64 envs, and the tick-end clears
    at N = 1, 8, 64; medians over ``reps`` (the planes re-copied from a
    pristine device copy before each rep, outside the timing)."""
    import cupy as cp
    scene = FireScene(0)
    sim = scene.sim
    g = sim.gmap
    wood = [tuple(int(v) for v in t) for t in zip(*np.nonzero(g.material == MAT_WOOD))]
    hull = [tuple(int(v) for v in t) for t in zip(*np.nonzero(g.material == MAT_HULL))]
    pool = wood + hull
    pre = _planes(g)
    print("TIMING (median ms): applier = stage + walk + books D2H; clears = launch")
    print(f"{'N':>4} {'destroys/env':>13} {'applier ms':>11}")
    for n in (1, 8, 64):
        st = unit_store.DeviceStore(n, 1, bp=bp,
                                    tables=np.tile(unit_tables.values(sim), (n, 1)))
        ap = applier.DeviceApplier(n, 64, bp=bp)
        pristine = _dev_planes([pre] * n)
        work = {k: v.copy() for k, v in pristine.items()}
        for k_d in (1, 8, 64):
            lists = [applier.destroy_records(pool[:k_d])] * n
            ts = []
            for _ in range(reps):
                for k in work:
                    work[k][...] = pristine[k]
                cp.cuda.runtime.deviceSynchronize()
                t0 = time.perf_counter()
                unit_kernels.apply_effects(st, ap, work, lists)
                ts.append(time.perf_counter() - t0)
            print(f"{n:>4} {k_d:>13} {1e3 * float(np.median(ts)):>11.3f}")
        clear = {c: cp.zeros((n, FH, FW), dtype=d) for c, d in applier.CLEAR_PLANES}
        ts = []
        for _ in range(reps):
            cp.cuda.runtime.deviceSynchronize()
            t0 = time.perf_counter()
            unit_kernels.tick_end_clears(bp, clear)
            cp.cuda.runtime.deviceSynchronize()
            ts.append(time.perf_counter() - t0)
        print(f"{n:>4} {'clears':>13} {1e3 * float(np.median(ts)):>11.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
