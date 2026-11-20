"""#29 P7e.1 -- PLANES AND LATCHES RESIDENT: the device unit slots read the
batch's resident planes, the device-built flight plane, the per-env TICK ROW
and the device claims list -- never a host-staged copy.

docs/residency_batching_design_2026-11-07.md, the paragraph "P7e RE-CUT
(2026-11-19)" and its P7e.1 row. Runs inside the GPU subprocess
(tests/cuda_harness.run_cuda_script; wrapper tests/test_cuda_unit_resident.py).
Every existing shadow and chained device check stays the gate of each slot's
law (tol 0); this check holds the four things P7e.1 moved onto the device:

PART REFRESH -- a ResidentBatch of ship_demo Simulations (N = 2, residency
on): after every entry upload of the played resident tick, the batch's
buffers of the unit planes (``GameMap._RESIDENT_UNIT_PLANES``: sight_screen,
heat_atten, light_atten, ignition_armed, div_source) equal the host mirror;
fixtures between ticks move each of them (grille and furniture tiles through
``on_tile_changed``, re-armed / disarmed ignition tiles, a blast source with
later ticks). PROPERTY: while host gameplay is in the loop a device slot
handed ``ResidentBatch.buffers`` reads what the host holds. BREAKS IF a unit
plane leaves the entry upload (``PhysicsRunner._RESIDENT_ENTRY_*``) or the
resident set.

PART INPLACE -- device slots handed ``ResidentBatch.buffers`` AS THEY ARE
write the batch's own buffers in place, equal to the host law on the host
map: slot 9's burn-through (TOPOLOGY, destroy_wall over the seam: every
resident plane incl. heat_atten / light_atten / div_source untouched or
written as the host), the seam's tile probe on grille and furniture tiles (sight_screen,
heat_atten, light_atten written in place), slot 9d's ignition (fire and
ignition_armed in place). The burn-through's TICK ROW count moves by the
host's ``_tile_serial`` delta. PROPERTY: the device path ran on the resident
buffers. BREAKS IF a face copies a field before its kernel (the buffer does
not move), a plane the seam writes is not resident, or the seam's bump is
lost.

PART FLIGHT -- ``unit_kernels.flight_plane`` (the device builder) equals
``combat.flight_plane`` on random material maps over the whole table (the
grille's blocks_thrown row set), N envs; a material outside the table faults.
PROPERTY: the flight plane the grenade slots read is the law's, built on the
device. BREAKS IF the builder drops the blocks_thrown term, reads another
env's tables or planes, or the row is not the sim's own column.

PART TICKROW -- every unit kernel face refuses per-launch host ticks (a
TypeError: the tick row is the only tick); a device burn-through bumps the
row's tiles_changed in place, and the HORDE stage run next on the same row
reads the bumped serial: its cache header's serial equals the host's
``HordeState.serial`` after the host's destroy_wall and slot 5. PROPERTY:
the topology serial is device state the later kernel sees. BREAKS IF the
seam's bump is lost or the HORDE stage reads the serial from anywhere but
the row.

PART CLAIMS -- slot 9c on the fire lanes (a flamethrower's credited heat)
shadowed with the device claims list staged (a) as the TickCredit's first
claims, (b) with a decoy claim of the same key AFTER each (the device must
still credit the first: equal) and (c) with the decoy BEFORE each (the
device must credit the decoy: the shadow must diverge on a credited hit).
PROPERTY: 9c / 9c3 read the claims list in place, first match first. BREAKS
IF the kernel reads a host table, ignores the list order or its count.
"""
from __future__ import annotations

import dataclasses
import sys
import time

import numpy as np

import breach_physics as bp

import cuda_s8a_check as s8a
from simulation import applier, unit_draws, unit_kernels, unit_store, unit_tables  # noqa: E402

FAILS: list = []
TICKS = 12


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", msg)


def _ship_sim(seed):
    from cuda_batch_invariance_check import _ship_sim as mk
    return mk(seed)


def _store(sims):
    U = max(1, max(len(s.units) for s in sims))
    st = unit_store.DeviceStore(len(sims), U, tables=unit_tables.values_batch(sims))
    st.upload(np.stack([unit_store.pack(s.units, U) for s in sims]))
    return st


def _open_tiles(g, k, seed):
    ys, xs = np.nonzero(~g.solid & ~g.is_vacuum & (g.material == 0))
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(ys), size=k, replace=False)
    return [(int(ys[i]), int(xs[i])) for i in pick]


def _walls(g, k, seed):
    """Interior wall tiles (solid, four solid-or-open non-vacuum neighbours)."""
    h, w = g.solid.shape
    ys, xs = np.nonzero(g.solid[1:-1, 1:-1] & (g.wall_hp[1:-1, 1:-1] > 0))
    cand = []
    for y, x in zip(ys + 1, xs + 1):
        nb = [(y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)]
        if not any(g.is_vacuum[p] for p in nb) and any(not g.solid[p] for p in nb):
            cand.append((int(y), int(x)))
    rng = np.random.default_rng(seed)
    return [cand[i] for i in rng.choice(len(cand), size=k, replace=False)]


# ---------------------------------------------------------------------------
# PART REFRESH
# ---------------------------------------------------------------------------
def part_refresh():
    from simulation import physics_runner as pr
    from simulation.gamemap import GameMap
    from simulation.materials import MAT_FURNITURE, MAT_GRILLE
    from simulation.resident_batch import ResidentBatch
    names = GameMap._RESIDENT_UNIT_PLANES
    sims = [_ship_sim(e + 1) for e in range(2)]
    seen = {"compared": 0, "moved": {n: 0 for n in names}}
    prev = {}
    orig = pr.PhysicsRunner._resident_upload

    def upload(self, gmap, ctx, upload_stamp, __o=orig):
        out = __o(self, gmap, ctx, upload_stamp)
        e = next(i for i, s in enumerate(sims) if s.gmap is gmap)
        for n in names:
            dev = gmap._dev[n].get()
            host = np.asarray(getattr(gmap, n))
            if not np.array_equal(dev, host):
                fail(f"refresh: env {e} plane {n} on the device != the host mirror after "
                     f"the entry upload ({int((dev != host).sum())} cells)")
            if (e, n) in prev and not np.array_equal(prev[(e, n)], host):
                seen["moved"][n] += 1
            prev[(e, n)] = host.copy()
        seen["compared"] += 1
        return out
    s8a._residency(True)
    s8a._set_backends(True)
    pr.PhysicsRunner._resident_upload = upload
    try:
        batch = ResidentBatch(sims)
        for t in range(TICKS):
            for e, sim in enumerate(sims):
                g = sim.gmap
                if t % 3 == 1:                       # sight_screen, heat / light atten
                    for k, (y, x) in enumerate(_open_tiles(g, 4, 100 * e + t)):
                        g.material[y, x] = MAT_GRILLE if k % 2 else MAT_FURNITURE
                        g.on_tile_changed(y, x)
                if t % 2 == 0:                       # the edge trigger, both ways
                    for (y, x) in _open_tiles(g, 5, 200 * e + t):
                        g.ignition_armed[y, x] = not g.ignition_armed[y, x]
                if t == 3:                           # a source with later ticks
                    tiles = _open_tiles(g, 6, 300 + e)
                    g.expansions.append([tiles, 1 << 16, 5])
            batch.step()
    finally:
        pr.PhysicsRunner._resident_upload = orig
        s8a._residency(False)
        s8a._set_backends(False)
    still = [n for n, k in seen["moved"].items() if k == 0]
    if still:
        fail(f"refresh: the fixtures never moved {still} at an entry (vacuous)")
    print(f"refresh: N=2 ship_demo, {TICKS} ticks, {seen['compared']} entry uploads; the "
          f"unit planes' buffers equal the host mirror at every entry; ticks each plane "
          f"moved: {seen['moved']}")


# ---------------------------------------------------------------------------
# PART INPLACE and the tick row's bump
# ---------------------------------------------------------------------------
def _resident_fields(batch, sims):
    """Every applier PLANE as the batch holds it (the non-resident ripple pair
    from the host: applier.non_resident_planes())."""
    import cupy as cp
    out = {}
    for name in applier.PLANE_NAMES:
        if name in batch.buffers:
            out[name] = batch.buffers[name]
        else:
            out[name] = cp.asarray(np.stack([np.asarray(getattr(s.gmap, name)) for s in sims]))
    return out


def part_inplace():
    import cupy as cp
    from simulation import combat
    from simulation.combat import ignition_params
    from config import CFG
    from simulation.materials import MAT_FURNITURE, MAT_GRILLE, MAT_WOOD
    from simulation.resident_batch import ResidentBatch
    nr = set(applier.non_resident_planes())
    sims = [_ship_sim(e + 11) for e in range(2)]
    s8a._residency(True)
    s8a._set_backends(True)
    try:
        batch = ResidentBatch(sims)
        for _ in range(3):
            batch.step()
        for s in sims:
            s.gmap.from_host()
        res_planes = [n for n in applier.PLANE_NAMES if n not in nr]
        for n in res_planes:
            if n not in batch.buffers:
                fail(f"inplace: plane {n} is outside non_resident_planes() but has no batch "
                     f"buffer")
        fields = _resident_fields(batch, sims)
        st = _store(sims)
        trow = unit_draws.TickRow.of_sims(sims)
        serial0 = [int(s.gmap._tile_serial) for s in sims]
        # slot 9's burn-through on the batch's own buffers
        walls = [_walls(s.gmap, 3, 7 + e) for e, s in enumerate(sims)]
        before = {n: batch.buffers[n].get() for n in res_planes}
        unit_kernels.burn_through(st, applier.DeviceApplier(2),
                                  fields, unit_kernels.pad_walls(
                                      [unit_kernels.walls_of(w) for w in walls]),
                                  tick_row=trow)
        for s, w in zip(sims, walls):
            for (y, x) in w:
                s.gmap.destroy_wall(y, x)
        moved = []
        for n in res_planes:
            dev = batch.buffers[n].get()
            for e, s in enumerate(sims):
                if not np.array_equal(dev[e], np.asarray(getattr(s.gmap, n))):
                    fail(f"inplace: after the burn-through, env {e}'s resident {n} != the host")
            if not np.array_equal(dev, before[n]):
                moved.append(n)
        for want in ("material", "solid", "wall_hp"):
            if want not in moved:
                fail(f"inplace: the burn-through did not move the batch's {want} buffer "
                     f"(the kernel did not write the resident buffers)")
        tiles = trow.tiles()
        burn_delta = [int(t) - s0 for t, s0 in zip(tiles, serial0)]
        for e, s in enumerate(sims):
            d_host = int(s.gmap._tile_serial) - serial0[e]
            if int(tiles[e]) - serial0[e] != d_host or d_host <= 0:
                fail(f"inplace: env {e}'s tick row moved by {int(tiles[e]) - serial0[e]}, the "
                     f"host's _tile_serial by {d_host}")
        # the seam's tile probe on a grille: sight_screen and the float atten pair in place
        probe = [[(y, x, MAT_GRILLE if k % 2 else MAT_FURNITURE)
                  for k, (y, x) in enumerate(_open_tiles(s.gmap, 4, 50 + e))]
                 for e, s in enumerate(sims)]
        b0 = {n: batch.buffers[n].get() for n in ("sight_screen", "heat_atten", "light_atten")}
        unit_kernels.topology_probe(st, applier.DeviceApplier(2), fields, probe)
        for s, rows in zip(sims, probe):
            for (y, x, m) in rows:
                s.gmap.material[y, x] = m
                s.gmap.on_tile_changed(y, x)
        for n in ("sight_screen", "heat_atten", "light_atten"):
            dev = batch.buffers[n].get()
            if np.array_equal(dev, b0[n]):
                fail(f"inplace: the grille probe did not move the batch's {n} buffer")
            for e, s in enumerate(sims):
                if not np.array_equal(dev[e], np.asarray(getattr(s.gmap, n))):
                    fail(f"inplace: after the grille probe, env {e}'s {n} != the host")
        # slot 9d on the batch's own buffers: hot armed wood seeds, a disarm
        for e, s in enumerate(sims):
            g = s.gmap
            ys, xs = np.nonzero((g.material == MAT_WOOD) & g.flammable)
            if len(ys) == 0:
                for (y, x) in _walls(g, 4, 90 + e):
                    g.material[y, x] = MAT_WOOD
                    g.on_tile_changed(y, x)
                ys, xs = np.nonzero(g.material == MAT_WOOD)
            T = int(g.materials.ignition_temp_q16[MAT_WOOD]) + (1 << 20)
            for k, (y, x) in enumerate(zip(ys[:6], xs[:6])):
                g.temperature[y, x] = T              # a solid wood tile, never a gas cell
                g.ignition_armed[y, x] = k % 2 == 0
            g.from_host()
        fields = _resident_fields(batch, sims)
        ign = {n: batch.buffers[n] for n, _d in unit_kernels.IGNITION_FIELDS}
        f0 = batch.buffers["fire"].get()
        a0 = batch.buffers["ignition_armed"].get()
        o2_frac_ext, seed = ignition_params(CFG)
        unit_kernels.ignition(st, ign)
        for s in sims:
            combat.apply_temperature_ignition(s.gmap, o2_frac_ext, seed)
        f1 = batch.buffers["fire"].get()
        a1 = batch.buffers["ignition_armed"].get()
        for e, s in enumerate(sims):
            if not (np.array_equal(f1[e], s.gmap.fire)
                    and np.array_equal(a1[e], s.gmap.ignition_armed)):
                fail(f"inplace: slot 9d on the batch's buffers != the host law in env {e}")
        if np.array_equal(f1, f0) and np.array_equal(a1, a0):
            fail("inplace: slot 9d moved neither fire nor ignition_armed in the batch's "
                 "buffers (vacuous, or the kernel wrote a copy)")
        # the ripple pair is the only survivor of non_resident_planes(): render only
        print(f"inplace: the burn-through, the grille probe and slot 9d wrote the batch's "
              f"own buffers equal to the host (moved {sorted(moved)}); the tick row's "
              f"tiles_changed moved by the host's _tile_serial delta {burn_delta}; "
              f"non_resident_planes() = {sorted(nr)}")
        del cp
    finally:
        s8a._residency(False)
        s8a._set_backends(False)


# ---------------------------------------------------------------------------
# PART FLIGHT
# ---------------------------------------------------------------------------
def part_flight(trials=6):
    import cupy as cp
    from types import SimpleNamespace
    from simulation import combat
    from simulation.materials import MAT_GRILLE
    sims = [_ship_sim(e + 21) for e in range(3)]
    st = _store(sims)
    tbl = sims[0].gmap.materials
    nm = len(tbl.blocks_thrown)
    if not bool(tbl.blocks_thrown[MAT_GRILLE]):
        fail("flight: the grille's blocks_thrown row is not set (the term is vacuous)")
    rng = np.random.default_rng(5)
    hit = 0
    for t in range(trials):
        h, w = int(rng.integers(5, 60)), int(rng.integers(5, 60))
        mat = rng.integers(0, nm, size=(3, h, w)).astype(np.int8)
        solid = rng.random((3, h, w)) < 0.3
        dev = unit_kernels.flight_plane(st, {"solid": cp.asarray(solid),
                                             "material": cp.asarray(mat)}).get()
        for e, s in enumerate(sims):
            stand = SimpleNamespace(solid=solid[e], material=mat[e], materials=s.gmap.materials)
            want = combat.flight_plane(stand)
            if not np.array_equal(dev[e], want):
                fail(f"flight: trial {t} env {e}: the device plane != combat.flight_plane")
            hit += int((want & ~solid[e]).sum())
    if hit == 0:
        fail("flight: no open tile was blocked by its material (blocks_thrown never read)")
    bad = np.zeros((3, 4, 4), dtype=np.int8)
    bad[1, 2, 2] = nm
    try:
        unit_kernels.flight_plane(st, {"solid": cp.zeros((3, 4, 4), dtype=cp.bool_),
                                       "material": cp.asarray(bad)})
        fail("flight: a material outside the table was not refused")
    except unit_kernels.UnitKernelFault as ex:
        if "[1]" not in str(ex):
            fail(f"flight: the refusal does not name the env: {ex}")
    print(f"flight: {trials} random maps x 3 envs equal combat.flight_plane ({hit} open "
          f"tiles blocked by their material's blocks_thrown); a material outside the "
          f"table refused by name")


# ---------------------------------------------------------------------------
# PART TICKROW
# ---------------------------------------------------------------------------
def part_tickrow():
    import cupy as cp
    from simulation import horde_cache, object_tables as ot, unit_schema
    from simulation.unit_events import EventRing
    from cuda_unit_horde_check import HordeScene, _applier_for
    sims = [_ship_sim(31)]
    st = _store(sims)
    for name, call in (
            ("expire_orders", lambda: unit_kernels.expire_orders(st, np.array([0]))),
            ("match_done", lambda: unit_kernels.match_done(st, np.array([0]))),
            ("melee", lambda: unit_kernels.melee(st, applier.DeviceApplier(1),
                                                 np.array([0])))):
        try:
            call()
            fail(f"tickrow: {name} took host ticks (the tick row is the only tick)")
        except TypeError:
            pass
    # the bump, read by the HORDE stage next on the same row
    checked = 0
    sc = HordeScene(0)
    for t in range(40):
        sim = sc.sim
        g = sim.gmap
        U = len(sim.units)
        st = unit_store.DeviceStore(1, U, tables=unit_tables.values_batch([sim]))
        st.upload(unit_store.pack(sim.units, U)[None])
        ring = EventRing(1, U)
        ring.begin()
        noise = ot.DeviceTable(unit_schema.NOISE_TABLE, 1)
        noise.stage_lists([unit_kernels.noise_of(sim)])
        doors = ot.DeviceTable(unit_schema.DOOR_TABLE, 1)
        tiles = ot.DeviceTable(unit_schema.DOOR_TILE_TABLE, 1)
        doors.stage_lists([unit_kernels.doors_of(sim)])
        tiles.stage_lists([unit_kernels.door_tiles_of(sim)])
        cache = horde_cache.DeviceHordeCache(1, U)
        cache.stage_lists([horde_cache.cache_records(sim)])
        fields = {n: cp.asarray(np.array(getattr(g, n))[None]) for n in applier.PLANE_NAMES}
        trow = unit_draws.TickRow.of_sims([sim])
        latch = [horde_cache.latch_of(sim, U)]
        if t % 4 == 1:
            wall = _walls(g, 1, t)
            unit_kernels.burn_through(st, _applier_for(st), fields,
                                      unit_kernels.pad_walls([unit_kernels.walls_of(wall)]),
                                      tick_row=trow)
            unit_kernels.horde(st, _applier_for(st), fields, ring, noise, doors, tiles, cache,
                               latch, trow)
            # the host: the same destroy, then slot 5 (twin state discarded after)
            for (y, x) in wall:
                g.destroy_wall(y, x)
            sim._run_zombie_slot()
            got = cache.records()[0].header
            hdr = horde_cache._HDR
            want = sim.horde_state.serial
            if want is not None and int(got[hdr["has_serial"]]) and \
                    int(got[hdr["serial"]]) != int(want):
                fail(f"tickrow: tick {sim.tick}: the HORDE stage read serial "
                     f"{int(got[hdr['serial']])}, the host's slot 5 {want} (the tick row's "
                     f"bump was not seen)")
            elif want is not None:
                checked += 1
            if int(trow.tiles()[0]) != int(g._tile_serial):
                fail(f"tickrow: tick {sim.tick}: the row's tiles_changed "
                     f"{int(trow.tiles()[0])} != the host's {int(g._tile_serial)}")
        sc.tick()
    if checked == 0:
        fail("tickrow: the HORDE stage never read a serial after a device bump (vacuous)")
    print(f"tickrow: the faces refuse host ticks; a device burn-through bumped the row and "
          f"the HORDE stage read the bumped serial == the host's on {checked} ticks")


# ---------------------------------------------------------------------------
# PART CLAIMS
# ---------------------------------------------------------------------------
def part_claims(ticks=60):
    import cuda_unit_lockstep as LS
    from cuda_unit_damage_check import SLOT_9C, LaneDriver
    from simulation.unit_events import CREDIT_HEAT
    orig = LS.LISTS["credit"]
    base = orig.records

    def decoys(before):
        def rec(sim, obj):
            r = base(sim, obj)
            out = []
            for row in r:
                d = row.copy()
                d[3] = int(row[3]) + 1000          # a claimant that is no unit
                out.extend([d, row] if before else [row, d])
            return np.array(out, dtype=np.int64).reshape(-1, 4)
        return rec

    res = {}
    for label, rec in (("first claims", base), ("decoys after", decoys(False)),
                       ("decoys before", decoys(True))):
        LS.LISTS["credit"] = dataclasses.replace(orig, records=rec)
        credited = 0
        try:
            drv = LaneDriver("fire")
            with LS.ShadowLockstep([drv], SLOT_9C) as ls:
                for _ in range(ticks):
                    ls.tick()
                    credited += sum(1 for ev in drv.sim.tick_events
                                    if type(ev).__name__ == "UnitHitEvent"
                                    and getattr(ev, "attacker_id", None) not in (None, -1))
            res[label] = (len(ls.divergences), ls.divergences[:1], credited)
        finally:
            LS.LISTS["credit"] = orig
    n0, d0, c0 = res["first claims"]
    n1, d1, _c1 = res["decoys after"]
    n2, d2, _c2 = res["decoys before"]
    if c0 == 0:
        fail("claims: the fire lanes credited no hit (vacuous)")
    if n0:
        fail(f"claims: the first-claims list diverged: {d0[0]}")
    if n1:
        fail(f"claims: a decoy AFTER each claim changed the credit (first match first is "
             f"not the reader's rule): {d1[0]}")
    if n2 == 0:
        fail("claims: a decoy BEFORE each claim changed nothing: 9c does not read the "
             "staged claims list (or the lanes credit no heat)")
    elif "attacker" not in d2[0].column:
        fail(f"claims: the decoys-before run diverged elsewhere than the attacker: {d2[0]}")
    print(f"claims: 9c on the fire lanes {ticks} ticks: the first-claims list equal, decoys "
          f"after equal, decoys before diverge at {d2[0] if d2 else None}; {c0} credited "
          f"hits (heat channel {CREDIT_HEAT})")


def main(parts=None) -> int:
    print(f"P7e.1 planes and latches resident, HAS_CUDA={getattr(bp, 'HAS_CUDA', False)}")
    t0 = time.perf_counter()
    for name, f in (("refresh", part_refresh), ("inplace", part_inplace),
                    ("flight", part_flight), ("tickrow", part_tickrow),
                    ("claims", part_claims)):
        if parts and name not in parts:
            continue
        try:
            f()
        except Exception as ex:          # a crash is a failure, named
            import traceback
            traceback.print_exc()
            fail(f"{name}: {type(ex).__name__}: {ex}")
    print(f"({time.perf_counter() - t0:.0f} s)")
    print("UNIT_RESIDENT_RESULT:", "FAIL" if FAILS else "PASS")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
