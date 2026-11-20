"""#29 P7e.2 -- THE DEVICE SORTS AND 9b: slot 9 fed from a burn-through list
sorted ON THE DEVICE, slot 9b's scan and top-K on the device, and slot 6's
stamp rows built from the unit store, each against its Python law at tol 0
on every plane it writes and every book.

docs/residency_batching_design_2026-11-07.md, the paragraph "P7e RE-CUT
(2026-11-19)" and its P7e.2 row. Runs inside the GPU subprocess
(tests/cuda_harness.run_cuda_script; wrapper tests/test_cuda_unit_sorts.py).
Faces: ``simulation.unit_kernels.burn_through_device`` / ``burst_walls`` /
``stamp_rows`` / ``stamp`` over ``cpp/src/cuda_unit_sorts.cu``. The shadows
are ``tests/cuda_unit_lockstep.py``'s (field outputs, books, the write-set
gate, env i of N == env i alone).

PART BURN -- the SHADOW LOCKSTEP of slot 9 with its input sorted on the
device, on the topology gate's fire scenes (walls burning through on many
ticks, several in one tick, a hull plug breaching only through the tile
destroyed before it: the order matters) at N = 1 and N = 2. The device is
handed the host's list SCRAMBLED (each tick a different permutation, never
the row-major one when a tick holds two or more walls) and nothing else.
PROPERTY: the device slot ranks the list into the host's row-major order
itself. BREAKS IF the sort is skipped or unstable, the rank reads another
env's list, or the face falls back to a host-built list (it is handed none).

PART RESIDENT -- the same sort fed in place from the LAST resident tick's
own burn-through list (``bp.resident_burn_list``: the fire's counter and its
atomic arrival-order buffer, never downloaded), a ResidentBatch of two fire
scenes: every tick the device-sorted records equal the list the played tick
handed slot 9 (sorted on the host). PROPERTY: slot 7's list reaches slot 9's
records with no host round trip. BREAKS IF the address is not the fire's, or
the rank differs from the host's sort.

PART BURST -- the SHADOW LOCKSTEP of slot 9b (``Simulation._burst_walls``) on
a scene BUILT FOR THE GATE (``burst_enabled`` on and the wood membrane's
threshold lowered for the scene: the shipped config has 9b off): a long
one-tile wood membrane between an air room and a vacuum pocket, so the first
scan sees more candidates than the cap, all with the SAME differential (the
cap cuts inside a tie: the tie order decides which walls fall), and later
ticks see distinct ones as the pocket fills. N = 1 and N = 2. PROPERTY: the
device scan, top-K and destroys are the host's, ties included. BREAKS IF the
compare, the side rule, the ambient / vacuum side values, the descending
order, the tie order or the cap differ.

PART STAMP -- the SHADOW LOCKSTEP of slot 6 from the unit store (the five
planes and the refilled atmosphere) on ship_demo's bots match (units walking
across tiles) and on a fire scene with units (walls burning through between
stamps: the refill acts); and every tick the device rows equal the host C++
stamp's rows (``GameMap._stamp_units_cpp``'s), which must change over the
walk. PROPERTY: slot 6's rows are built on the device from the store in the
host's order and the planes follow them. BREAKS IF the order, a footprint
tile, the alive / valid filter, a default value or the refill's order or
mean differ, or the rows come from anywhere but the store.

Prints per-part lines and ``UNIT_SORTS_RESULT: PASS`` / ``FAIL``.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

import cuda_unit_topology_check as T   # noqa: E402  (FireScene, ShipDriver, the slot-9 outputs)
from cuda_unit_lockstep import DeviceSlot, ShadowLockstep  # noqa: E402
from simulation import applier, unit_kernels  # noqa: E402

FAILS: list = []
BURN_TICKS = int(os.environ.get("UNIT_SORTS_BURN_TICKS", "90"))
BURST_TICKS = int(os.environ.get("UNIT_SORTS_BURST_TICKS", "40"))
STAMP_TICKS = int(os.environ.get("UNIT_SORTS_STAMP_TICKS", "200"))
RESIDENT_TICKS = int(os.environ.get("UNIT_SORTS_RESIDENT_TICKS", "40"))


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


# ---------------------------------------------------------------------------
# PART BURN
# ---------------------------------------------------------------------------
class _Scramble:
    """Hands the device the host's list in a non-row-major order: reversed
    and rotated by a per-call amount (two or more walls are never in the
    host's order)."""

    def __init__(self):
        self.calls = 0
        self.scrambled = 0

    def lists(self, walls):
        import cupy as cp
        w = np.asarray(walls, dtype=np.int64)
        n, R = int(w.shape[0]), max(1, int(w.shape[1]))
        self.calls += 1
        counts = np.zeros(n, dtype=np.int32)
        lists = np.zeros((n, R), dtype=np.int32)
        return counts, lists, w

    def build(self, walls, width):
        import cupy as cp
        counts, lists, w = self.lists(walls)
        for e in range(w.shape[0]):
            rows = w[e][w[e, :, 2] != 0]
            idx = [int(y) * width + int(x) for (y, x) in rows[:, :2]]
            k = len(idx)
            if k >= 2:
                idx = idx[::-1]
                r = self.calls % k
                idx = idx[r:] + idx[:r]
                if idx == sorted(idx):          # a rotation back onto the sort
                    idx = idx[1:] + idx[:1]
                self.scrambled += 1
            counts[e] = k
            lists[e, :k] = idx
        return unit_kernels.BurnList.of_arrays(cp.asarray(counts), cp.asarray(lists))


_SCR = _Scramble()


def _run9_sorted(st, inp):
    w = int(inp.fields["material"].shape[2])
    burn = _SCR.build(inp.tables["burn_walls"], w)
    return unit_kernels.burn_through_device(st, T._applier_for(st), inp.fields, burn)


SLOT_9_SORTED = DeviceSlot(label="9 burn-through walls", name="9 burn-through (device sort)",
                           run=_run9_sorted, inputs=("burn_walls",),
                           outputs=T.PLANE_OUTPUTS + T.SLOT9_BOOKS,
                           host_only=T.SLOT9_HOST_ONLY)


def _shadow(name, slot, drivers, ticks, tally_of):
    tally = dict(ticks=0, applied=0, ticks_acting=0, multi=0)
    t0 = time.perf_counter()
    with ShadowLockstep(drivers, slot, bp=bp) as ls:
        for _ in range(ticks):
            if any(getattr(d, "is_terminal", lambda: False)() for d in drivers):
                break
            rep = ls.tick()
            tally["ticks"] += 1
            if not rep.equal:
                fail(f"{name}: {ls.divergences[-1]}")
                break
            tally_of(rep, tally)
    tally["seconds"] = round(time.perf_counter() - t0)
    print(f"{name}: N {len(drivers)}, {tally}")
    return tally


def _tally_topology(rep, tally):
    cov = rep.coverage
    host = rep.host_books["tiles_changed"]
    if not np.array_equal(cov, host):
        fail(f"tick {rep.tick}: applied {cov.tolist()} != the host's destroys {host.tolist()}")
    tally["applied"] += int(cov.sum())
    tally["ticks_acting"] += int((cov > 0).any())
    tally["multi"] += int((cov > 1).any())


def part_burn():
    before = _SCR.scrambled
    for name, drivers in (("burn fire (space)", [T.FireScene(0)]),
                          ("burn fire (planetside)", [T.FireScene(1, "ambient", 0.5)]),
                          ("burn fire batch", [T.FireScene(2), T.FireScene(5)])):
        t = _shadow(name, SLOT_9_SORTED, drivers, BURN_TICKS, _tally_topology)
        if t["applied"] == 0 or t["multi"] == 0:
            fail(f"{name}: no tick destroyed two walls at once (the order is untested)")
    if _SCR.scrambled == before:
        fail("burn: the device was never handed a scrambled list (vacuous)")
    print(f"burn: {_SCR.scrambled - before} scrambled lists sorted on the device")


# ---------------------------------------------------------------------------
# PART RESIDENT
# ---------------------------------------------------------------------------
def _dev_ints(ptr, k):
    """``k`` int32 read from a device address (a gate's download)."""
    import cupy as cp
    mem = cp.cuda.UnownedMemory(int(ptr), 4 * int(k), None)
    return cp.ndarray((int(k),), dtype=cp.int32, memptr=cp.cuda.MemoryPointer(mem, 0)).get()


def part_resident():
    import cuda_s8a_check as s8a
    from simulation import physics_runner as pr  # noqa: F401
    from simulation.resident_batch import ResidentBatch
    from simulation.simulation import Simulation
    scenes = [T.FireScene(0), T.FireScene(3)]
    sims = [s.sim for s in scenes]
    for s in sims:
        s.physics_runner.headless = True
    got: dict = {}
    orig = Simulation._burn_through_walls

    def hook(self, destroyed, __o=orig):
        got[id(self)] = [(int(y), int(x)) for (y, x) in destroyed]
        return __o(self, destroyed)
    seen = dict(ticks=0, walls=0, multi=0, unsorted_arrival=0)
    s8a._residency(True)
    s8a._set_backends(True)
    Simulation._burn_through_walls = hook
    try:
        batch = ResidentBatch(sims)
        h, w = sims[0].gmap.solid.shape
        st = T._store_for(sims)
        app = applier.DeviceApplier(len(sims), T.CAP, bp=bp)
        for _ in range(RESIDENT_TICKS):
            for sc in scenes:                    # the scene's own schedule
                g = sc.sim.gmap
                for t in sc.schedule.get(sc.sim.tick, ()):
                    if int(g.material[t]) == T.MAT_WOOD:
                        g.wall_hp[t] = 1          # the entry upload carries wall_hp
            got.clear()
            batch.step()
            burn = unit_kernels.BurnList.resident(bp, len(sims), h, w)
            app.begin()
            unit_kernels._done(st, "burn_sort", bp.unit_burn_sort(
                st.buffer, app.buffer, burn.counts, burn.lists, burn.stride, h, w))
            recs = app.records()
            cnt = _dev_ints(burn.counts, len(sims))
            for e, sim in enumerate(sims):
                host = got.get(id(sim), [])
                dev = [(int(r[1]), int(r[2])) for r in recs[e]]
                if dev != host:
                    fail(f"resident: tick {sim.tick} env {e}: device-sorted {dev} != the "
                         f"played tick's list {host}")
                seen["walls"] += len(host)
                seen["multi"] += int(len(host) > 1)
                k = int(cnt[e])
                if k > 1:
                    arr = list(_dev_ints(burn.lists + 4 * e * burn.stride, k))
                    seen["unsorted_arrival"] += int(arr != sorted(arr))
            seen["ticks"] += 1
    finally:
        Simulation._burn_through_walls = orig
        s8a._residency(False)
        s8a._set_backends(False)
    if seen["walls"] == 0 or seen["multi"] == 0:
        fail(f"resident: the fire never burned two walls in one tick (vacuous): {seen}")
    print(f"resident: N=2 fire scenes, {seen} (the device sort read the resident fire's "
          f"own counter and arrival list in place)")


# ---------------------------------------------------------------------------
# PART BURST
# ---------------------------------------------------------------------------
BH, BW = 40, 30


class BurstScene:
    """A one-tile wood membrane (column ``c``) between an air room and a
    vacuum pocket inside the hull; the scene sets the wood row's burst
    threshold low (the shipped wood holds 2.0) and turns 9b on (the shipped
    config has it off). ``variant`` moves the membrane, the pocket's width
    and the seed."""

    THRESH = 0.25

    def __init__(self, variant=0):
        from simulation.materials import MAT_AIR, MAT_HULL, MAT_WOOD
        v = int(variant)
        c = 12 + v % 4
        tm = np.full((BH, BW), T.SPACE, dtype=np.int32)
        tm[1:BH - 1, 1:BW - 1] = MAT_HULL
        tm[2:BH - 2, 2:c] = MAT_AIR
        tm[2:BH - 2, c] = MAT_WOOD
        tm[2:BH - 2, c + 1:BW - 3 - v % 2] = T.SPACE      # the vacuum pocket
        tm[5 + v, 2:c] = MAT_WOOD                         # a wood partition in the room
        self.sim = T._sim(T._level(tm, f"p7e2_burst_{v}", 1.0 / 3.0, "space"), seed=1300 + v)
        T._add_unit(self.sim, 4.0, 20.0)

    def tick(self):
        self.sim.step()


def _run9b(st, inp):
    return unit_kernels.burst_walls(st, T._applier_for(st), inp.fields)


SLOT_9B = DeviceSlot(label="9b over-pressure walls", run=_run9b,
                     outputs=T.PLANE_OUTPUTS + T.SLOT9_BOOKS, host_only=T.SLOT9_HOST_ONLY)


def _burst_probe():
    """Wrap find_burst_walls to tally, per scan, the candidates beyond the cap
    and whether the cap cut inside a tie (a pure scan: a second call changes
    nothing)."""
    from simulation.gamemap import GameMap
    orig = GameMap.find_burst_walls
    stats = dict(scans=0, over_cap=0, tie_at_cap=0, max_cand=0, distinct=0)

    def spread(g, y, x):
        vals = []
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if not (0 <= ny < g.solid.shape[0] and 0 <= nx < g.solid.shape[1]):
                continue
            if g.solid[ny, nx]:
                continue
            vals.append(0 if g.is_vacuum[ny, nx] else int(g.atmosphere[ny, nx]))
        return max(vals) - min(vals)

    def probe(self, max_pops=None, __o=orig):
        out = __o(self, max_pops)
        full = __o(self, None)
        stats["scans"] += 1
        stats["max_cand"] = max(stats["max_cand"], len(full))
        if len(out) >= 2 and len({spread(self, *t) for t in out}) > 1:
            stats["distinct"] += 1                # an order decided by the spread
        if max_pops is not None and len(full) > max_pops > 0:
            stats["over_cap"] += 1
            a, b = full[max_pops - 1], full[max_pops]
            if spread(self, *a) == spread(self, *b):
                stats["tie_at_cap"] += 1
        return out
    return orig, probe, stats


def part_burst():
    from config import CFG
    from simulation.gamemap import GameMap
    from simulation.materials import MAT_WOOD
    orig, probe, stats = _burst_probe()
    phys = CFG.physics
    was = getattr(phys, "burst_enabled", True)
    setattr(phys, "burst_enabled", True)
    GameMap.find_burst_walls = probe
    saved = []
    try:
        for name, mk in (("burst membrane", lambda: [BurstScene(0)]),
                         ("burst membrane batch", lambda: [BurstScene(1), BurstScene(2)])):
            drivers = mk()
            for d in drivers:
                tbl = d.sim.gmap.materials
                saved.append((tbl, float(tbl.burst_threshold[MAT_WOOD])))
                tbl.burst_threshold[MAT_WOOD] = BurstScene.THRESH
            t = _shadow(name, SLOT_9B, drivers, BURST_TICKS, _tally_topology)
            if t["applied"] == 0 or t["multi"] == 0:
                fail(f"{name}: the scan never destroyed two walls in one tick (vacuous)")
    finally:
        GameMap.find_burst_walls = orig
        setattr(phys, "burst_enabled", was)
        for tbl, v in reversed(saved):
            tbl.burst_threshold[MAT_WOOD] = v
    if stats["over_cap"] == 0 or stats["tie_at_cap"] == 0 or stats["distinct"] == 0:
        fail(f"burst: no scan held more candidates than the cap with a tie across it, "
             f"or none destroyed walls of different spreads (the top-K, its tie order "
             f"or the descending order is untested): {stats}")
    print(f"burst: scans {stats}")


# ---------------------------------------------------------------------------
# PART STAMP
# ---------------------------------------------------------------------------
STAMP_IN = tuple("field:" + n for n in ("permeability", "wave_absorb", "heat_atten_q",
                                        "light_atten_q", "solid", "is_vacuum"))
STAMP_OUT = tuple("field:" + n for n in ("obstacles", "dyn_permeability", "dyn_wave_absorb",
                                         "dyn_heat_atten_q", "dyn_light_atten_q",
                                         "atmosphere"))
STAMP_HOST_ONLY = (
    ("_stamp_pending", "the host C++ stamp's kept rows for the PLAYED resident tick's device "
     "stamp (take_stamp_rows); the device slot builds its own from the store"),
)


class _StampRun:
    def __init__(self):
        self.scratch = None
        self.drivers = []
        self.host_rows: dict = {}
        self.compared = 0
        self.changed = 0
        self.prev = None
        self.refilled = 0
        self.cpp = True

    def __call__(self, st, inp):
        f = inp.fields
        cov, self.scratch = unit_kernels.stamp(st, f, self.scratch)
        self.refilled += int(np.asarray(cov).sum())
        if st.shape[0] == len(self.drivers) and self.cpp:
            h, w = int(f["atmosphere"].shape[1]), int(f["atmosphere"].shape[2])
            rows, total, self.scratch = unit_kernels.stamp_rows(st, h, w, self.scratch)
            dev = rows[:total].get()
            want = np.concatenate([self.host_rows[id(d.sim.gmap)] for d in self.drivers])
            if dev.shape != want.shape or not np.array_equal(dev, want):
                fail(f"stamp rows: tick {self.drivers[0].sim.tick}: the device's "
                     f"{dev.shape[0]} rows != the host stamp's {want.shape[0]}")
            self.compared += 1
            if self.prev is not None and not np.array_equal(self.prev, want):
                self.changed += 1
            self.prev = want
        return cov


def _host_rows_hook(run):
    from simulation.gamemap import GameMap
    orig = GameMap._stamp_units_cpp

    def hook(self, units, __o=orig):
        out = __o(self, units)
        ys, xs, perm, wabs, heat_q, light_q, _serial = self._stamp_pending
        e = next(i for i, d in enumerate(run.drivers) if d.sim.gmap is self)
        rows = np.empty((ys.shape[0], 9), dtype=np.int32)
        rows[:, 0] = e
        rows[:, 1] = ys
        rows[:, 2] = xs
        rows[:, 3] = perm.view(np.int32)
        rows[:, 4] = wabs.view(np.int32)
        rows[:, 5] = heat_q
        rows[:, 6:9] = light_q
        run.host_rows[id(self)] = rows
        return out
    return orig, hook


def part_stamp():
    """The C++ stamp path (every game path with an engine) keeps ``obstacles``
    the SAME array as ``solid`` (``_update_caches`` aliases them; the C++
    stamp writes in place), so destroy_wall opens the wall in ``obstacles``
    too and the refill's ``prev_obstacles & ~obstacles`` is never true there
    (a reference finding, not changed). The refill is reachable on the
    Python stamp path (``use_cpp_stamp = False``: its first stamp gives
    ``obstacles`` an array of its own), byte for byte the C++ path's planes
    otherwise: the fire scenes run it there to exercise the device refill."""
    from simulation.gamemap import GameMap
    for name, mk, ticks, cpp in (
            ("stamp ship (bots, seed 1)", lambda: [T.ShipDriver(1)], STAMP_TICKS, True),
            ("stamp fire (units)", lambda: [T.FireScene(0, extra_units=2)], 60, True),
            ("stamp fire (units, python stamp: the refill)",
             lambda: [T.FireScene(0, extra_units=2)], 60, False),
            ("stamp fire batch (python stamp: the refill)",
             lambda: [T.FireScene(2, extra_units=1), T.FireScene(5, extra_units=1)], 60,
             False)):
        run = _StampRun()
        run.drivers = mk()
        run.cpp = cpp
        if not cpp:
            for d in run.drivers:
                d.sim.gmap.use_cpp_stamp = False
        slot = DeviceSlot(label="6 stamp_units", run=run, inputs=STAMP_IN, outputs=STAMP_OUT,
                          host_only=STAMP_HOST_ONLY)
        orig, hook = _host_rows_hook(run)
        GameMap._stamp_units_cpp = hook
        try:
            _shadow(name, slot, run.drivers, ticks, lambda rep, t: None)
        finally:
            GameMap._stamp_units_cpp = orig
        print(f"{name}: rows compared on {run.compared} ticks, changed on {run.changed}; "
              f"the refill wrote {run.refilled} tiles")
        if cpp and (run.compared == 0 or run.changed == 0):
            fail(f"{name}: the stamp rows never changed (the units did not cross tiles)")
        if not cpp and run.refilled == 0:
            fail(f"{name}: the refill never acted (no wall opened between stamps)")


def main(parts=None) -> int:
    print(f"P7e.2 the device sorts and 9b, HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}")
    t0 = time.perf_counter()
    for name, f in (("burn", part_burn), ("resident", part_resident),
                    ("burst", part_burst), ("stamp", part_stamp)):
        if parts and name not in parts:
            continue
        try:
            f()
        except Exception as ex:          # a crash is a failure, named
            import traceback
            traceback.print_exc()
            fail(f"{name}: {type(ex).__name__}: {ex}")
    print(f"({time.perf_counter() - t0:.0f} s)")
    print("UNIT_SORTS_RESULT:", "FAIL" if FAILS else "PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
