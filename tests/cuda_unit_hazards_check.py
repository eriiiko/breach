"""#29 P6b.4 -- THE HAZARDS GATE: the timed charges and the heat sources on
the device (residency design 4.4c, the cut table's P6b.4 row;
cpp/src/cuda_unit_hazards.cuh, the applier's TIMED_CHARGES and HEAT_SOURCES
stages), slot 9e CHAINED whole, and the digest's entity and signal sections
exported from the device tables.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_unit_hazards.py; CPU side tests/test_hazard_tables.py).

PART TABLES -- the two tables ARE the sims' (unpack(records) re-encodes, the
entity rows are the serializer's) on the scene and fire_smoke_studio; P6b.4
adds no Device tables row (the every-row gate has nothing new to read).
PART SCENE -- the SHADOW LOCKSTEP of 9e(d') timed charges, 9e(d') planted
charges and 9e(d'') heat sources on the hazard scene (tests/_hazard_scene.py,
both boundaries) and on fire_smoke_studio: both tables, the blast log, the
PAYLOAD records, every plane, the books, the edits, the claims, the noise,
the ring equal each tick, the tables' IDENTITY each tick, the hooked match
equal to an unhooked twin; a census of both laws' branches, each required.
PART BOUND -- a chain exactly at its proved bound (the blast log one disc and
five sources whose death payloads chain) runs and equals the host; one past
it is refused by name with nothing written.
PART BATCH -- N = 2 / 8 scene variants (entity counts, layout, seed, fields;
env 0 not the largest; an env with no charge and no heat source).
PART PADDING -- garbage past both tables' counts neither read nor written.
PART OVERFLOW -- each table past its capacity, a stage's own PAYLOAD list
past its capacity: refused by name.
PART CHAINED -- slot 9e WHOLE ((a) emit + sensors, (b) logic, (c) door
inputs, (d) pumps, vents, doors, timed charges, planted charges, heat
sources, (e) swap) on the device with no host law between, in a running sim
(the scene both ways, airlock_demo, fire_smoke_studio, ship_demo's bots 500
ticks): unit hash, field digest, planes, books, every table, the edit queue,
the noise and blast logs equal an untouched reference every tick; the entity
and signal sections EXPORTED from the device's tables equal the reference's
serializer bytes (and their digests) every tick; every host law of 9e run 0
times in the twin.

BREAKS IF any of those differs from the law. Prints per-part lines and
``UNIT_HAZARDS_RESULT: PASS`` / ``FAIL``; ``bench()`` prints the TIMING table.
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
import cuda_unit_lockstep as LS  # noqa: E402
from cuda_unit_lockstep import ChainedSlots, DeviceSlot, ShadowLockstep  # noqa: E402
from _hazard_scene import HazardScene, hazard_level  # noqa: E402
from simulation import (applier, attack_lists, edit_door, object_tables as ot,  # noqa: E402
                        unit_events, unit_kernels as uk, unit_schema as us, unit_store,
                        unit_tables)

FAILS: list = []
ONE = 65536
SCENE_TICKS = int(os.environ.get("UNIT_HAZARDS_SCENE_TICKS", "160"))
LEVEL_TICKS = int(os.environ.get("UNIT_HAZARDS_LEVEL_TICKS", "120"))
BATCH_TICKS = int(os.environ.get("UNIT_HAZARDS_BATCH_TICKS", "90"))
BATCH_NS = [int(x) for x in os.environ.get("UNIT_HAZARDS_BATCH_NS", "2,8").split(",")]
CHAIN_TICKS = int(os.environ.get("UNIT_HAZARDS_CHAIN_TICKS", "160"))
SHIP_TICKS = int(os.environ.get("UNIT_HAZARDS_SHIP_TICKS", "500"))


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", str(msg).encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# The slots
# ---------------------------------------------------------------------------
def _pg():
    import cuda_unit_payloads_check as pg
    return pg


PLANE_OUTPUTS = tuple("field:" + n for n in applier.PLANE_NAMES)


def _walk_lists(lists):
    return uk.WalkLists(door=lists["edits"].door, claims=lists["claims"], noise=lists["noise"],
                        blast_log=lists["blast_log"], sources=lists.get("sources"),
                        source_tiles=lists.get("source_tiles"))


def _run_tc(seed=True):
    def run(st, inp, ring, lists):
        if seed:
            lists["edits"].set_seq(inp.scalars["edit_seq"])
        return uk.timed_charges(st, lists["timed_payloads"], inp.fields, ring, _walk_lists(lists),
                                lists["timed_charges"])
    return run


def _run_pl(seed=True, log="payloads"):
    def run(st, inp, ring, lists):
        uk.charges_tick(st, lists["charges"], lists[log], inp.tick)
        if seed:
            lists["edits"].set_seq(inp.scalars["edit_seq"])
        return uk.detonate(st, lists[log], inp.fields, ring, _walk_lists(lists))
    return run


def _run_hs(seed=True):
    def run(st, inp, ring, lists):
        if seed:
            lists["edits"].set_seq(inp.scalars["edit_seq"])
        return uk.heat_sources(st, lists["heat_payloads"], inp.fields, ring, _walk_lists(lists),
                               lists["heat_sources"])
    return run


def _book_outputs():
    return _pg().BOOK_OUTPUTS


def _list_outputs():
    return _pg().LIST_OUTPUTS


def _host_only():
    return _pg().HOST_ONLY + (
        ("smoke", "GameMap.smoke IS a view of gas[SMOKE] (the same memory): compared "
                  "through field:gas"),)


DET_IN = ("tick", "edit_seq", "list:noise", "list:blast_log", "list:source_tiles",
          "list:sources")   # the tiles land first


def slot_tc(seed=True):
    return DeviceSlot(label="9e(d') timed charges", run=_run_tc(seed), events=True,
                      skip_ok=True, inputs=DET_IN + ("list:timed_charges",),
                      outputs=PLANE_OUTPUTS + _book_outputs() + _list_outputs()
                      + ("list:timed_charges", "list:timed_payloads"),
                      host_only=_host_only())


def slot_pl(seed=True, log="payloads"):
    return DeviceSlot(label="9e(d') planted charges", run=_run_pl(seed, log), events=True,
                      skip_ok=True, inputs=DET_IN + ("list:charges",),
                      outputs=PLANE_OUTPUTS + _book_outputs() + _list_outputs()
                      + ("list:charges", f"list:{log}"), host_only=_host_only())


def slot_hs(seed=True):
    return DeviceSlot(label="9e(d'') heat sources", run=_run_hs(seed), events=True,
                      skip_ok=True, inputs=DET_IN + ("list:heat_sources",),
                      outputs=PLANE_OUTPUTS + _book_outputs() + _list_outputs()
                      + ("list:heat_sources", "list:heat_payloads"),
                      host_only=_host_only())


def share(sims):
    """Every sim on the process's weapon tables (cuda_unit_payloads_check.
    share_tables), and each timed charge's and heat source's payload rebound
    to the shared table's row of its name (a runtime holds its build's row
    object; ``applier.payload_index`` names a row of the sim's table by
    identity)."""
    t = _pg().share_tables(sims)
    for s in sims:
        for c in s._timed_charges:
            c.payload = t.payloads.by_name[c.payload.name]
        for h in s._heat_sources:
            if h.death_payload is not None:
                h.death_payload = t.payloads.by_name[h.death_payload.name]
    return t


class Scene(HazardScene):
    def __init__(self, variant=0):
        super().__init__(variant, bp=bp)


class Level:
    """A shipped level's sim (``match``: its MatchSession)."""

    def __init__(self, name, match=False, seed=1):
        from cuda_unit_vents_check import LevelDriver
        self._d = LevelDriver(name, seed=seed, match=match)
        self.sim = self._d.sim

    def tick(self):
        self._d.tick()

    def is_terminal(self):
        return self._d.is_terminal()


# ---------------------------------------------------------------------------
# Identity
# ---------------------------------------------------------------------------
def _identity(sim, where):
    from simulation.entities import REGISTRY
    by_ord = {int(e.ordinal): e for e in sim.entities}
    for fn, table, objs, rows_fn, cls in (
            (uk.timed_charges_of, us.TIMED_CHARGE_TABLE, uk.timed_objects, uk.timed_entity_rows,
             "timed_charge"),
            (uk.heat_sources_of, us.HEAT_SOURCE_TABLE, uk.heat_objects, uk.heat_entity_rows,
             "heat_source")):
        recs = fn(sim)
        if uk._records_of(table, objs(recs)).tolist() != recs.tolist():
            fail(f"identity {where}: the {table.name} table: unpack(records) re-encodes "
                 f"differently")
            return False
        for r in recs:
            e = by_ord[int(r[0])]
            want = (("alive", 1 if getattr(e, "alive", True) else 0),) + tuple(
                REGISTRY[cls].runtime_digest_rows(e))
            if rows_fn(r) != want:
                fail(f"identity {where}: {e.id}: the table's entity rows are not the "
                     f"serializer's")
                return False
    return True


# ---------------------------------------------------------------------------
# The census: both laws' branches, counted on the host's own decisions
# ---------------------------------------------------------------------------
class Census:
    def __init__(self):
        self.c = Counter()
        self._orig = []
        self.origin = {}          # (fy, fx, radius, wall_q) -> who appended the disc
        self.initial = set()
        self._last = None
        self._sim = None
        self._deaths = 0
        self._timed_fired_tick = None

    def _wrap(self, cont, attr, fn):
        orig = getattr(cont, attr)
        self._orig.append((cont, attr, orig))

        def w(*a, **k):
            return fn(orig, *a, **k)
        w.__wrapped__ = orig             # applier.record_of recognises the law
        setattr(cont, attr, w)

    def __enter__(self):
        return self.install()

    def __exit__(self, *exc):
        self.uninstall()
        return False

    def after_tick(self, sim):
        pass

    def install(self):
        import importlib
        from simulation import heat_source_system as hss
        from simulation import simulation as simmod
        c = self.c
        cen = self
        for modname, who in (("simulation.simulation", "a grenade"),
                             ("simulation.timed_charge_system", "a timed charge"),
                             ("simulation.charges", "a planted charge"),
                             ("simulation.heat_source_system", "a heat source's death")):
            mod = importlib.import_module(modname)

            def ep(orig, gmap, *a, __who=who, **k):
                n0 = len(gmap.blast_log)
                out = orig(gmap, *a, **k)
                for d in gmap.blast_log[n0:]:
                    cen.origin[(d[0], d[1], d[2], d[3])] = __who
                payload = a[4]
                c["fired: a strength row" if payload.strength > 0 else
                  ("fired: an explosive disc row" if payload.explosive else
                   "fired: a disc row")] += 1
                return out
            self._wrap(mod, "execute_payload", ep)

        def tc(orig, sim):
            g = sim.gmap
            pre = [(ch, ch.ticks_to_fire, ch.fired, ch.enabled, int(g.temperature[ch.fy, ch.fx]),
                    bool(g.solid[ch.fy, ch.fx])) for ch in sim._timed_charges]
            out = orig(sim)
            if any(t0 == 0 or (ch.heat and not f0 and ch.fired and ch.fuse_ticks == 0)
                   for ch, t0, f0, _e, _T, _s in pre):
                cen._timed_fired_tick = sim.tick
            for ch, t0, f0, en, T0, sol0 in pre:
                if ch.heat and not f0:
                    if not en:
                        c["heat: a disabled charge waits"] += 1
                    elif ch.fired:
                        c["heat: the latch closes" + (", a zero fuse fires in the crossing sweep"
                                                      if ch.fuse_ticks == 0 else
                                                      ", its fuse loads")] += 1
                    else:
                        c["heat: below its threshold"] += 1
                        if sol0 and not g.solid[ch.fy, ch.fx] and T0 >= ch.trigger_temp_q16:
                            c["heat: its hot tile destroyed by an earlier charge of the same "
                              "sweep (reads it cold)"] += 1
                    continue
                if t0 < 0:
                    c["spent / disabled / armed-and-spent (ticks_to_fire -1)"] += 1
                elif t0 > 0:
                    c["armed heat charge counts its fuse" if ch.heat else "timer counts down"] += 1
                elif ch.period_ticks > 0:
                    c["timer fires and reloads its period"] += 1
                else:
                    c[("armed heat charge fires" if ch.heat else "timer fires once")] += 1
            return out
        self._wrap(simmod, "sweep_timed_charges", tc)

        def ct(orig, sim):
            n = sum(1 for q in sim.planted_charges if q.due(sim.tick))
            out = orig(sim)
            if n and cen._timed_fired_tick == sim.tick:
                c["a timed and a planted charge fire in one tick"] += 1
            return out
        from simulation import charges as chmod
        self._wrap(chmod, "tick", ct)

        def bd(orig, src, fy, fx, radius, wall_q):
            out = orig(src, fy, fx, radius, wall_q)
            d2 = (src.fy - fy) ** 2 + (src.fx - fx) ** 2
            if d2 > radius * radius:
                c["heat: a disc misses (d2 > r^2)"] += 1
            elif out == 0:
                c["heat: a disc's rim (falloff 0)"] += 1
            else:
                c["heat: a disc hits"] += 1
                if not cen._sim.gmap.has_los(fy, fx, src.fy, src.fx, physical=True):
                    c["heat: damaged across a wall (the disc law reads no wall)"] += 1
            cen._last = (fy, fx, radius, wall_q)
            return out
        self._wrap(hss, "_blast_damage_q16", bd)

        def die(orig, sim, src):
            key = cen._last
            if key in cen.initial:
                c[f"heat: killed by {cen.origin.get(key, 'a blast of slots 2 / 4c')}"] += 1
            else:
                c["heat: killed by a disc its chain appended this sweep"] += 1
            c["heat: dies, its death payload applied" if src.death_payload is not None else
              "heat: dies, no death payload"] += 1
            cen._deaths += 1
            return orig(sim, src)
        self._wrap(hss, "_die", die)

        def hs(orig, sim):
            cen._sim = sim
            cen.initial = set(tuple(d) for d in sim.gmap.blast_log)
            cen._deaths = 0
            if any(not s.alive for s in sim._heat_sources) and sim.gmap.blast_log:
                c["heat: a dead source skipped by a disc"] += 1
            out = orig(sim)
            if cen._deaths >= 3:
                c["heat: a chain of three or more deaths in one sweep"] += 1
            g = sim.gmap
            for s in sim._heat_sources:
                if not s.alive:
                    c["deposit: skipped, dead"] += 1
                elif not s.enabled:
                    c["deposit: skipped, disabled"] += 1
                elif s.max_temp_q16 and int(g.temperature[s.fy, s.fx]) >= s.max_temp_q16:
                    c["deposit: skipped, at its cap"] += 1
                else:
                    c["deposit: enqueued" + (", capped source below its cap"
                                             if s.max_temp_q16 else "")] += 1
            bus = getattr(sim, "_signal_bus", None)
            if bus is not None:
                for i, (o, n) in enumerate(bus.slots):
                    e = sim._entity_by_ordinal.get(o)
                    if n == "alive" and e is not None and e.class_name == "heat_source":
                        rt = [h for h in sim._heat_sources if int(h.ordinal) == int(o)][0]
                        c["signal: a wired heat-source alive read " + str(int(bus.pub[i]))
                          + (" after its death" if not rt.alive else "")] += 1
            return out
        self._wrap(simmod, "sweep_heat_sources", hs)

        return self

    def uninstall(self):
        for cont, attr, orig in reversed(self._orig):
            setattr(cont, attr, orig)
        self._orig.clear()


NEED = (
    "heat: a disabled charge waits", "heat: below its threshold",
    "heat: the latch closes, a zero fuse fires in the crossing sweep",
    "heat: the latch closes, its fuse loads", "armed heat charge counts its fuse",
    "armed heat charge fires", "timer counts down", "timer fires once",
    "timer fires and reloads its period",
    "spent / disabled / armed-and-spent (ticks_to_fire -1)",
    "heat: its hot tile destroyed by an earlier charge of the same sweep (reads it cold)",
    "fired: a strength row", "fired: a disc row", "fired: an explosive disc row",
    "a timed and a planted charge fire in one tick",
    "heat: a disc misses (d2 > r^2)", "heat: a disc hits",
    "heat: damaged across a wall (the disc law reads no wall)",
    "heat: killed by a timed charge", "heat: killed by a grenade",
    "heat: killed by a planted charge", "heat: killed by a disc its chain appended this sweep",
    "heat: dies, its death payload applied", "heat: dies, no death payload",
    "heat: a chain of three or more deaths in one sweep",
    "heat: a dead source skipped by a disc",
    "deposit: skipped, dead", "deposit: skipped, disabled", "deposit: skipped, at its cap",
    "deposit: enqueued", "deposit: enqueued, capped source below its cap",
    "signal: a wired heat-source alive read 1 after its death",
)
UNREACHABLE = {
    "heat: a disc of radius <= 0": "execute_payload appends a disc to the blast log only when "
                                   "radius > 0, so _blast_damage_q16's r <= 0 branch is read by "
                                   "no log entry",
    "heat: the chain past the blast log's capacity": "the reference's list has no bound; the "
                                                     "device's is PART BOUND's",
    "signal: a wired heat-source alive read 0": "the bus reads the source's EntityInstance "
                                                "(_entity_by_ordinal is built before "
                                                "build_heat_sources): a reported finding",
}


# ---------------------------------------------------------------------------
# PART TABLES
# ---------------------------------------------------------------------------
def part_tables():
    sc = Scene(0)
    lv = Level("fire_smoke_studio")
    for _ in range(80):
        sc.tick()
        lv.tick()
        for nm, sim in (("scene", sc.sim), ("fire_smoke_studio", lv.sim)):
            if not _identity(sim, f"{nm} tick {sim.tick}"):
                return
    n = (len(uk.timed_charges_of(sc.sim)), len(uk.heat_sources_of(sc.sim)),
         len(uk.timed_charges_of(lv.sim)))
    if min(n) < 1:
        fail(f"tables: vacuous {n}")
    print(f"tables: the timed-charge and heat-source tables are the sims' (re-encode, entity "
          f"rows) 80 ticks on the scene and fire_smoke_studio (rows {n}); no new Device "
          f"tables row")


# ---------------------------------------------------------------------------
# PART SCENE / BATCH
# ---------------------------------------------------------------------------
SLOTS = None


def _slots():
    return [slot_tc(), slot_pl(), slot_hs()]


def run_lockstep(name, drivers, ticks, census=None, twins=None):
    t0 = time.perf_counter()
    tot = Counter()
    n = 0
    ctx = census if census is not None else _pg()._Null()
    U = max([1] + [len(d.sim.units) for d in drivers])
    with ctx, ShadowLockstep(drivers, _slots(), n_units=U) as ls:
        for _ in range(ticks):
            reps = ls.tick()
            n += 1
            for lab, rep in reps.items():
                if not rep.equal:
                    fail(f"{name}: {ls.divergences[-1]}")
                    return tot
                if not np.array_equal(rep.coverage, rep.host_changed):
                    fail(f"{name} [{lab}] tick {rep.tick}: coverage {rep.coverage.tolist()} "
                         f"!= host-changed {rep.host_changed.tolist()}")
                    return tot
                for e in range(len(drivers)):
                    for ln in ("timed_payloads", "payloads", "heat_payloads"):
                        if rep.lists and ln in rep.lists:
                            tot[f"PAYLOAD [{lab}]"] += len(rep.lists[ln][e])
                    tot["door rows"] += len(rep.lists["edits"][e])
                    tot["blast log"] += len(rep.lists["blast_log"][e])
                tot[f"events [{lab}]"] += int(rep.events.sum())
            for e, d in enumerate(drivers):
                if not _identity(d.sim, f"{name} tick {d.sim.tick} env {e}"):
                    return tot
            if twins is not None:
                for d, tw in zip(drivers, twins):
                    tw.tick()
                    if not _same(d.sim, tw.sim):
                        fail(f"{name}: the hooked run left the unhooked one at tick "
                             f"{d.sim.tick}")
                        return tot
    print(f"{name}: N={len(drivers)}, {n} ticks, the three slots equal every tick (tables, "
          f"blast log, PAYLOAD records, planes, books, edits, claims, noise, ring, write set)"
          f"{', env i == env i alone' if len(drivers) > 1 else ''}"
          f"{', the unhooked twin equal' if twins else ''}; compared {dict(tot)} "
          f"({time.perf_counter() - t0:.0f} s)")
    return tot


def _same(a, b):
    from cuda_unit_verbs_check import _same as s
    return s(a, b)


def part_scene():
    census = Census()
    sc, tw = Scene(0), Scene(0)
    share([sc.sim, tw.sim])
    tot = run_lockstep("scene", [sc], SCENE_TICKS, census, twins=[tw])
    if FAILS:
        return
    s1 = Scene(1)
    share([s1.sim])
    run_lockstep("scene planetside", [s1], SCENE_TICKS, census)
    lv = Level("fire_smoke_studio")
    share([lv.sim])
    run_lockstep("level fire_smoke_studio", [lv], LEVEL_TICKS, census)
    c = census.c
    miss = [b for b in NEED if not c[b]]
    if miss:
        fail(f"scene: census branches never taken {miss}")
    if not tot.get("PAYLOAD [9e(d'') heat sources]"):
        fail("scene: vacuous, no heat-source PAYLOAD compared")
    print("scene census: " + "; ".join(f"{k} = {v}" for k, v in sorted(c.items())))
    print("scene unreachable: " + "; ".join(f"{k} -- {v}" for k, v in UNREACHABLE.items()))


def part_batch():
    for n in BATCH_NS:
        vs = [2, 0] if n == 2 else [2, -1, 0, 1, 3, 4, 5, 6][:n]
        ds = [Scene(v) for v in vs]
        share([d.sim for d in ds])
        counts = [len(d.sim._timed_charges) + len(d.sim._heat_sources) for d in ds]
        if counts[0] >= max(counts):
            fail(f"batch N={n}: env 0 is the largest ({counts})")
        if n == 8 and min(counts) != 0:
            fail("batch N=8: no env without a charge and a heat source")
        run_lockstep(f"batch N={n} (variants {vs}, entity counts {counts})", ds, BATCH_TICKS)
        if FAILS:
            return


# ---------------------------------------------------------------------------
# PART BOUND
# ---------------------------------------------------------------------------
def _bound_level(n_src):
    from level_loader import LevelData
    from _hazard_scene import _tc, _hs
    from simulation.materials import MAT_AIR, MAT_HULL
    tm = np.full((12, 40), MAT_HULL, dtype=np.int32)
    tm[1:11, 1:39] = MAT_AIR
    ents = [_tc("t0", 3, 5, "frag_standard", first=0.5)]
    ents += [_hs(f"s{i}", 4 + 2 * i, 5, 10.0, hp=10, death="plasma_splash_large")
             for i in range(n_src)]
    for i, e in enumerate(ents):
        e.ordinal = 200 + i
    return LevelData(name="p6b4_bound", version="3", path=Path("."), tilemap=tm,
                     tile_size_m=1.0, diffuse_path=Path("."), entities=ents[::-1], wires=[],
                     boundary="vacuum")


class BoundScene:
    def __init__(self, n_src=5):
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        self.sim = Simulation(_bound_level(n_src), seed=5, breach_physics=bp,
                              enable_recorder=False, ruleset=ContinuousRealtime())
        self.sim.set_paused(False)

    def is_terminal(self):
        return False

    def tick(self):
        self.sim.step()
        self.sim.set_paused(False)


def _blast_cap(cap):
    """The lockstep's blast-log list with a capacity of ``cap`` (a context)."""
    old = LS.LISTS["blast_log"]

    def device(n_env, n_units, c, bp_):
        return ot.DeviceTable(us.BLAST_LOG_TABLE, n_env, n_units=n_units, row_cap=1, bp=bp_,
                              cap=cap)

    class Ctx:
        def __enter__(self):
            LS.LISTS["blast_log"] = LS.ListDef(old.host, old.records, old.consume, device,
                                               old.fields, old.mark, old.land)

        def __exit__(self, *exc):
            LS.LISTS["blast_log"] = old
            return False
    return Ctx()


def part_bound():
    n_src = 5
    bound = 1 + n_src                     # the frag's disc + five sources that append
    d = BoundScene(n_src)
    share([d.sim])
    with _blast_cap(bound):
        tot = run_lockstep(f"bound: a chain AT its proved bound ({bound} discs, the log's "
                           f"capacity {bound})", [d], 20)
    if FAILS:
        return
    if sum(1 for s in d.sim._heat_sources if not s.alive) != n_src:
        fail("bound: vacuous, the chain did not run to its bound")
        return
    _ = tot
    d = BoundScene(n_src)
    share([d.sim])
    refused = None
    with _blast_cap(bound - 1), ShadowLockstep([d], [slot_tc(), slot_hs()], n_units=1) as ls:
        for _ in range(20):
            try:
                ls.tick()
            except applier.HeatChainRefused as ex:
                refused = ex
                break
        if refused is None:
            fail("bound: one past the bound was not refused")
            return
        msg = str(refused)
        if "env 0: bound 6" not in msg or f"capacity {bound - 1}" not in msg:
            fail(f"bound: the refusal does not name the env, the bound and the capacity: {msg}")
        inp = ls.last_inputs["9e(d'') heat sources"]
        for f in inp.fields:
            if not np.array_equal(inp.fields[f].get(), inp._host_fields[f]):
                fail(f"bound: the refused stage wrote plane {f}")
        got = ls.dev_lists["heat_sources"].records()[0]
        if [tuple(r) for r in got] != [tuple(int(x) for x in r) for r in
                                       inp.lists["heat_sources"][0]]:
            fail("bound: the refused stage wrote the heat-source table")
        if ls.dev_lists["blast_log"].records()[0] != [
                tuple(int(x) for x in r) for r in inp.lists["blast_log"][0]]:
            fail("bound: the refused stage wrote the blast log")
        if ls.dev_lists["edits"].records()[0]:
            fail("bound: the refused stage enqueued edits")
    print(f"bound: a chain at its proved bound ({bound}) equal to the host; one past "
          f"(capacity {bound - 1}) refused by name, nothing written: {msg[:160]}...")


# ---------------------------------------------------------------------------
# PART PADDING / OVERFLOW
# ---------------------------------------------------------------------------
def _device_stage(sims, stage, tables, garbage=False, own_cap=applier.E_CAP):
    """One TIMED_CHARGES / HEAT_SOURCES launch over ``sims`` from their host
    state (``tables``: (timed, heat) DeviceTables staged by the caller)."""
    pg = _pg()
    N = len(sims)
    U = max(len(s.units) for s in sims)
    st = unit_store.DeviceStore(N, U, tables=unit_tables.values_batch(sims))
    st.upload(np.stack([unit_store.pack(s.units, U) for s in sims]))
    ring = unit_events.EventRing(N, U, unit_events.E_MAX)
    ring.begin()
    own = applier.DeviceApplier(N, own_cap, n_units=U)
    own.begin()
    door = edit_door.DeviceEditDoor(N)
    door.begin()
    door.set_seq(np.array([s.edit_queue._seq for s in sims], dtype=np.int64))
    claims = attack_lists.claims_list(N)
    claims.begin()
    noise = ot.DeviceTable(us.NOISE_TABLE, N)
    noise.stage_lists([uk.noise_of(s) for s in sims])
    blast = ot.DeviceTable(us.BLAST_LOG_TABLE, N)
    blast.stage_lists([uk.blast_log_of(s) for s in sims])
    srcs, stl = uk.source_tables(N)
    srcs.stage_lists([uk.sources_of(s) for s in sims])
    stl.stage_lists([uk.source_tiles_of(s) for s in sims])
    fields = pg._dev_planes([pg._planes(s.gmap) for s in sims])
    wl = uk.WalkLists(door=door, claims=claims, noise=noise, blast_log=blast, sources=srcs,
                      source_tiles=stl)
    timed, heat = tables
    if stage == "timed":
        uk.timed_charges(st, own, fields, ring, wl, timed)
    else:
        uk.heat_sources(st, own, fields, ring, wl, heat)
    return {k: v.get() for k, v in fields.items()}, own


def _stage_tables(sims, garbage):
    timed, heat = uk.hazard_tables(len(sims))
    staged = {}
    for name, t, fn in (("timed", timed, uk.timed_charges_of), ("heat", heat, uk.heat_sources_of)):
        rows = np.zeros(t.shape, dtype=np.int64)
        cnt = []
        for e, s in enumerate(sims):
            a = fn(s)
            rows[e, :len(a)] = a
            cnt.append(len(a))
            if garbage and len(a):
                rows[e, len(a):] = a[0]           # live-looking junk: a firing charge, a source
        t.upload(rows, np.array(cnt))
        staged[name] = (rows.copy(), cnt)
    return (timed, heat), staged


def part_padding():
    scs = [Scene(2), Scene(0)]
    share([s.sim for s in scs])
    for sc in scs:
        for _ in range(23):                       # t_frag fires on the next sweep
            sc.tick()
    sims = [s.sim for s in scs]
    for stage in ("timed", "heat"):
        tabs_c, sc_ = _stage_tables(sims, False)
        planes_c, _ = _device_stage(sims, stage, tabs_c)
        tabs_d, sd_ = _stage_tables(sims, True)
        planes_d, _ = _device_stage(sims, stage, tabs_d)
        for i, name in enumerate(("timed", "heat")):
            rc = tabs_c[i].download()[0]
            rd = tabs_d[i].download()[0]
            for e in range(2):
                n = sc_[name][1][e]
                if not np.array_equal(rc[e, :n], rd[e, :n]):
                    fail(f"padding [{stage}] env {e}: garbage past the {name} table's count "
                         f"changed its valid rows")
                if not np.array_equal(rd[e, n:], sd_[name][0][e, n:]):
                    fail(f"padding [{stage}] env {e}: a padding row of the {name} table was "
                         f"written")
        for k in planes_c:
            if not np.array_equal(planes_c[k], planes_d[k]):
                fail(f"padding [{stage}]: garbage rows changed plane {k}")
    counts = [len(s._timed_charges) for s in sims]
    if counts[0] == counts[1]:
        fail("padding: the two envs hold the same charge count (vacuous)")
    print(f"padding: garbage rows past both tables' counts (charges {counts}) neither read "
          f"nor written, through both stages")


def part_overflow():
    sc = Scene(0)
    share([sc.sim])
    seen = Counter()
    for tab, objs in ((us.TIMED_CHARGE_TABLE, sc.sim._timed_charges),
                      (us.HEAT_SOURCE_TABLE, sc.sim._heat_sources)):
        many = (list(objs) * (tab.capacity // len(objs) + 1))[:tab.capacity + 1]
        try:
            ot.pack(tab, many)
            fail(f"overflow: a {tab.name} list past the capacity was packed")
        except ot.ObjectTableOverflow as ex:
            if f"'{tab.name}'" not in str(ex):
                fail(f"overflow: the {tab.name} refusal does not name it: {ex}")
            seen[tab.name] += 1
        rows = uk._records_of(tab, list(objs))
        small = ot.DeviceTable(tab, 2, cap=len(rows) - 1, bp=bp)
        try:
            small.stage_lists([rows[:1], rows])
            fail(f"overflow: a full {tab.name} device table took more rows")
        except ot.ObjectTableOverflow as ex:
            if f"'{tab.name}'" not in str(ex) or "env 1" not in str(ex):
                fail(f"overflow: the {tab.name} refusal does not name the table and the env")
            seen[f"{tab.name} device"] += 1
    # a stage's own PAYLOAD list one short: two timer charges due on the next
    # sweep (a fixture sets their countdowns), the list holding one record
    for _ in range(3):
        sc.tick()
    due = [c for c in sc.sim._timed_charges if not c.heat and c.enabled][:2]
    for c in due:
        c.ticks_to_fire = 0
    tabs, _ = _stage_tables([sc.sim], False)
    try:
        _device_stage([sc.sim], "timed", tabs, own_cap=1)
        fail("overflow: the timed stage's PAYLOAD list took more than its capacity")
    except applier.EffectOverflow as ex:
        if "TIMED_CHARGES" not in str(ex) or "env(s) [0]" not in str(ex):
            fail(f"overflow: the own-list refusal does not name the stage and the env: {ex}")
        seen["the stage's own PAYLOAD list"] += 1
    print(f"overflow: every table past its capacity refused by name {dict(seen)}")


# ---------------------------------------------------------------------------
# PART CHAINED: slot 9e WHOLE
# ---------------------------------------------------------------------------
def _land_books(g, b):
    """Every book of the chain landed once on the host quantity it twins
    (``applier.BOOKS``' host refs, through the seams' own book functions)."""
    for row in applier.BOOKS:
        v = int(b.get(row.name, 0))
        if row.host is None or not v:
            continue
        if row.host.startswith("gas_energy_books:"):
            g._gas_energy_book(row.host.split(":", 1)[1], v)
        elif row.host.startswith("solid_energy_books:"):
            g._solid_energy_book(row.host.split(":", 1)[1], v)
        elif row.host.startswith("trace_clip_books:"):
            g.trace_clip_books[int(row.host.split(":", 1)[1])] += v
        else:
            setattr(g, row.host, getattr(g, row.host) + v)


def _host_side_9e(ch):
    """Slot 9e's host-only side after the WHOLE chain ran on the device: the
    books landed once, the render marks and the sky flag of every tile whose
    material changed, and the render-only DoorDestroyedEvents of the doors
    that died, at their place before the chain's ring events (the door sweep
    runs before the charges)."""
    from cuda_unit_doors_check import DLAY
    from simulation.events import DoorDestroyedEvent
    from simulation.materials import MAT_DOOR_CLOSED

    def hs(sim, _pre, _post):
        g = sim.gmap
        _land_books(g, ch.books)
        mat0 = ch.inputs._host_fields["material"][0]
        changed = np.argwhere(mat0 != g.material)
        for (y, x) in changed.tolist():
            g.render_dirty.add((int(y), int(x)))
        if len(changed):
            g._sky_mask_dirty = True
        if "doors" not in ch.inputs.lists:
            return
        doors0 = ch.inputs.lists["doors"][0]
        tiles0 = ch.inputs.lists["door_tiles"][0]
        alive = DLAY.column("alive").offset
        evs = []
        for k, d in enumerate(sim._doors):
            if int(doors0[k][alive]) and not d.alive:
                a = int(doors0[k][DLAY.column("tile_start").offset])
                for i in range(int(doors0[k][DLAY.column("tile_count").offset])):
                    y, x = int(tiles0[a + i][0]), int(tiles0[a + i][1])
                    if int(mat0[y, x]) == MAT_DOOR_CLOSED:
                        evs.append(DoorDestroyedEvent(pos=(y, x)))
        if evs:
            n_ring = len(ch.events[-1]) if ch.events else 0
            at = len(sim.tick_events) - n_ring
            sim.tick_events[at:at] = evs
    return hs


def _plan(sim):
    """Slot 9e's slots in the conductor's order, from the first one whose host
    callable runs on this level (the chain runs at its entry; the conductor
    gates the earlier ones off: no bus)."""
    import cuda_unit_signals_check as sg
    from cuda_unit_doors_check import SLOT_9ED
    from cuda_unit_vents_check import SLOT_VENTS
    plan = [("bus", sg.SLOT_A), ("bus", sg.SLOT_B), ("bus", sg.SLOT_C), ("bus", sg.SLOT_P),
            ("ducts", SLOT_VENTS), ("doors", SLOT_9ED),
            ("timed", slot_tc(seed=True)), ("planted", slot_pl(seed=False, log="detonations")),
            ("heat", slot_hs(seed=False)), ("bus", sg.SLOT_E)]
    has = {"bus": sim._signal_bus is not None, "ducts": bool(sim._ducts),
           "doors": bool(sim._doors), "timed": bool(sim._timed_charges)}
    first = next(i for i, (k, _s) in enumerate(plan) if has.get(k))
    out = [s for _k, s in plan[first:]]
    # the edit door's seq is seeded by the chain's FIRST producer of an edit
    tc = [i for i, s in enumerate(out) if s.label == "9e(d') timed charges"]
    _ = tc
    return out


_HOST_LAWS = (("simulation.simulation", "Simulation._signal_emit"),
              ("simulation.simulation", "sample_sensors"),
              ("simulation.simulation", "sweep_logic_nodes"),
              ("simulation.simulation", "Simulation._resolve_door_inputs"),
              ("simulation.simulation", "sweep_pumps"),
              ("simulation.simulation", "sweep_vents"),
              ("simulation.simulation", "sweep_doors"),
              ("simulation.simulation", "sweep_timed_charges"),
              ("simulation.charges", "tick"),
              ("simulation.simulation", "sweep_heat_sources"),
              ("simulation.signal_bus", "SignalBus.swap_node_signals"),
              ("simulation.timed_charge_system", "execute_payload"),
              ("simulation.charges", "execute_payload"),
              ("simulation.heat_source_system", "execute_payload"),
              ("simulation.heat_source_system", "deposit_heat"))


def _tables_of(sim):
    out = {n: f(sim) for n, f in uk.SIGNAL_RECORDS.items()}
    out.update(uk.export_records(sim))
    return out


def chained(name, make, ticks):
    import importlib
    from cuda_unit_topology_check import _same_twin
    from simulation.entities.serialize import serialize_entity_state, serialize_signal_state
    pg = _pg()
    twin, ref = make(), make()
    share([twin.sim, ref.sim])
    calls = Counter()
    saved = []
    for mod, qual in _HOST_LAWS:
        m = importlib.import_module(mod)
        cont, attr = (m, qual) if "." not in qual else (getattr(m, qual.split(".")[0]),
                                                        qual.split(".")[1])
        orig = getattr(cont, attr)
        saved.append((cont, attr, orig))

        def counted(*a, __o=orig, __q=f"{mod}.{qual}", **k):
            who = "twin" if any(x is twin.sim or x is getattr(twin.sim, "_signal_bus", 0)
                                or x is twin.sim.gmap for x in a) else "reference"
            calls[f"{who} {__q}"] += 1
            return __o(*a, **k)
        setattr(cont, attr, counted)
    plan = _plan(twin.sim)
    holder = {}
    ch = ChainedSlots(twin, plan, bp=bp, host_side=lambda s, a, b: holder["hs"](s, a, b))
    holder["hs"] = _host_side_9e(ch)
    ex = uk.EntityExport(twin.sim)
    n = 0
    t0 = time.perf_counter()
    exported = 0
    try:
        with ch:
            for _ in range(ticks):
                if any(getattr(x, "is_terminal", lambda: False)() for x in (twin, ref)):
                    break
                runs0 = ch.runs
                twin.tick()
                ref.tick()
                n += 1
                why = _same_twin(twin.sim, ref.sim)
                if why is None and pg._queue_state(twin.sim) != pg._queue_state(ref.sim):
                    why = "the edit queue"
                if why is None and (list(twin.sim.gmap.noise_log) != list(ref.sim.gmap.noise_log)
                                    or list(twin.sim.gmap.blast_log) !=
                                    list(ref.sim.gmap.blast_log)):
                    why = "the noise / blast log"
                if why is None:
                    a, b = _tables_of(twin.sim), _tables_of(ref.sim)
                    for k in a:
                        if np.asarray(a[k]).tolist() != np.asarray(b[k]).tolist():
                            why = f"the {k} table"
                            break
                if why is None and ch.runs > runs0:
                    dev = ch.lists[-1]
                    tabs = {k: dev[k] for k in uk.EXPORT_LISTS if k in dev}
                    for k in uk.EXPORT_LISTS:       # a table no slot of the plan holds
                        tabs.setdefault(k, uk.export_records(ref.sim)[k])
                    want_e = serialize_entity_state(ref.sim.entities)
                    want_s = serialize_signal_state(ref.sim._digest_signals())
                    got_e = ex.entity_section(tabs)
                    got_s = ex.signal_section(dev.get("signals"))
                    if got_e != want_e or ex.digest(got_e) != ex.digest(want_e):
                        why = "the entity section exported from the device tables"
                    elif got_s != want_s or ex.digest(got_s) != ex.digest(want_s):
                        why = "the signal section exported from the device tables"
                    else:
                        exported += 1
                if why is not None:
                    fail(f"chained {name}: the twin left the reference at tick "
                         f"{twin.sim.tick} ({why})")
                    break
    finally:
        for cont, attr, orig in reversed(saved):
            setattr(cont, attr, orig)
    twin_calls = {k: v for k, v in calls.items() if k.startswith("twin")}
    if twin_calls:
        fail(f"chained {name}: the host laws ran in the twin {twin_calls}")
    if ch.runs != n:
        fail(f"chained {name}: the chain ran {ch.runs} times in {n} ticks")
    if exported != n:
        fail(f"chained {name}: the export was compared on {exported} of {n} ticks")
    recs = Counter()
    for x in ch.lists:
        for nm in ("timed_payloads", "detonations", "heat_payloads"):
            recs[nm] += len(x.get(nm, ()))
    print(f"chained {name}: {n} ticks, slot 9e WHOLE on the device ({len(plan)} slots: "
          f"{[s.label for s in plan]}): unit hash, field digest, planes, books, every table, "
          f"edit queue, noise and blast logs equal the reference every tick; the entity and "
          f"signal sections exported from the device tables equal the serializer's (and their "
          f"digests) on {exported} ticks; PAYLOADs applied {dict(recs)}; host 9e law calls in "
          f"the twin {sum(twin_calls.values()) if twin_calls else 0}, in the reference "
          f"{sum(v for k, v in calls.items() if k.startswith('reference'))} "
          f"({time.perf_counter() - t0:.0f} s)")
    return n, recs


def part_chained():
    from cuda_unit_signals_check import AirlockDriver
    for v in (0, 1):
        n, recs = chained(f"hazard scene v{v}", lambda v=v: Scene(v), CHAIN_TICKS)
        if FAILS:
            return
        if not recs["heat_payloads"] or not recs["timed_payloads"]:
            fail(f"chained hazard scene v{v}: vacuous {dict(recs)}")
    chained("airlock_demo", AirlockDriver, CHAIN_TICKS)
    chained("fire_smoke_studio", lambda: Level("fire_smoke_studio"), CHAIN_TICKS)
    chained("ship_demo bots match", lambda: Level("ship_demo", match=True), SHIP_TICKS)


# ---------------------------------------------------------------------------
PARTS = ("tables", "scene", "bound", "batch", "padding", "overflow", "chained")


def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("UNIT_HAZARDS_RESULT: FAIL (not the CUDA build)")
        return 1
    from cuda_unit_vents_check import gate_config
    gate_config()
    parts = parts or os.environ.get("UNIT_HAZARDS_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            globals()[f"part_{p}"]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
        sys.stdout.flush()
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("UNIT_HAZARDS_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
