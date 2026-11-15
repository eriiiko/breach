"""#29 P7.0b -- THE OBJECT TABLES' device gate (design 4.4c decision 3), with
the planted-charge table as the pilot. Run in the CUDA build's subprocess
(``tests/cuda_harness.py``; wrapper ``tests/test_cuda_object_table.py``).

Parts (``OBJECT_TABLE_PARTS``, default all):

  tables   -- the compiled layout's fingerprint is the schema's, and a
              ``DeviceTable`` refuses a module whose fingerprint differs.
  trip     -- a device round trip of random rows and counts, bit for bit
              (padding garbage included: upload copies, never interprets).
  append   -- appends keep their call order, continue what the table holds,
              differ per env; past the capacity nothing is written, the
              env's overflow is counted and the Python face refuses naming
              the table and the env.
  compact  -- the stable compaction against the host reference: dead rows in
              the middle, at both ends, all, none, per env in one batch;
              garbage with live = 1 past the count never read; the freed rows
              zeroed.
  scene    -- IDENTITY on live scenes that plant charges (P6.4a's verb scene,
              variants 0 and 3 -- the latter's table full --, and a charge
              scene whose timers make dead rows in every position): every
              tick, ``unpack(pack(list))`` equal attribute for attribute; the
              device round trip bit for bit; the tick's APPEND on the device
              (the table at the tick's head + the charges the tick placed:
              the scene's fixtures through ``charges.place`` and slot 1b's
              plants)
              equal to the host list at ``charges.tick``'s entry; the
              COMPACTION on the device (that list with this tick's fires
              marked) equal to the host list after the prune; in a batch of
              differing envs, env i == env i alone (an N = 1 table). A census
              of the branches the scenes must take.

Prints per-part lines and ``OBJECT_TABLE_RESULT: PASS`` / ``FAIL``; exits
0 / 1. ``bench()`` prints the TIMING table (append + compaction).
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

from simulation import charges, object_tables as ot, unit_schema  # noqa: E402

FAILS: list = []
T = unit_schema.CHARGE_TABLE
LAY = ot.table_layout(T)
W = LAY.width
CAP = LAY.capacity
SCENE_TICKS = int(os.environ.get("OBJECT_TABLE_SCENE_TICKS", "160"))
PARTS = ("tables", "trip", "append", "compact", "scene")


def fail(msg):
    FAILS.append(msg)
    print("  FAIL:", msg)


def _eq(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


# ---------------------------------------------------------------------------
def part_tables():
    compiled = bp.object_tables_layout()
    if compiled["fingerprint"] != ot.layout().fingerprint:
        fail("the compiled object tables fingerprint is not the schema's")

    class Fake:
        HAS_CUDA = True

        @staticmethod
        def object_tables_layout():
            return {"fingerprint": "0" * 64}

        @staticmethod
        def ObjectTableBuffer(*a):         # what a refusal must never reach
            return object()
    try:
        ot.DeviceTable(T, 1, bp=Fake)
        fail("a DeviceTable accepted a module with another fingerprint")
    except ot.LayoutMismatch:
        pass
    print(f"tables: fingerprint {compiled['fingerprint'][:12]} == schema; a "
          f"mismatched module refused")


def part_trip(n_env=5, trials=4):
    rng = np.random.default_rng(7)
    for t in range(trials):
        tab = ot.DeviceTable(T, n_env)
        rows = rng.integers(-2 ** 62, 2 ** 62, size=(n_env, CAP, W), dtype=np.int64)
        counts = rng.integers(0, CAP + 1, size=n_env).astype(np.int64)
        tab.upload(rows, counts)
        r, c, o = tab.download()
        if not (_eq(r, rows) and _eq(c, counts) and not o.any()):
            fail(f"trip {t}: the device round trip is not bit for bit")
    try:
        tab.upload(rows, np.full(n_env, CAP + 1))
        fail("an upload with a count past the capacity was accepted")
    except ValueError:
        pass
    print(f"trip: {trials} random batches of {n_env} envs bit for bit; a count "
          f"past the capacity refused")


def _rec(v, live=1):
    r = np.zeros(W, dtype=np.int64)
    r[0] = v
    r[LAY.live] = live
    return r


def part_append(n_env=4):
    tab = ot.DeviceTable(T, n_env)
    base = np.zeros((n_env, CAP, W), dtype=np.int64)
    counts = np.array([0, 3, CAP - 2, 1], dtype=np.int64)
    for e in range(n_env):
        for i in range(counts[e]):
            base[e, i] = _rec(1000 * e + i)
    tab.upload(base, counts)
    want = [base[e].copy() for e in range(n_env)]
    want_n = list(counts)
    k = 2
    for step in range(3):                       # three appends in a row
        recs = np.zeros((n_env, k, W), dtype=np.int64)
        n = np.array([2, 1, 0 if step else 2, 2], dtype=np.int64)
        for e in range(n_env):
            for j in range(n[e]):
                recs[e, j] = _rec(10 ** 6 + 100 * e + 10 * step + j)
            want[e], want_n[e], lost = ot.append_rows(T, want[e], want_n[e],
                                                     list(recs[e, :n[e]]))
            assert lost == 0
        tab.append(recs, n)
    r, c, o = tab.download()
    for e in range(n_env):
        if c[e] != want_n[e] or not _eq(r[e], want[e]):
            fail(f"append: env {e} differs from the ordered host append")
    if o.any():
        fail("append: an overflow counted below the capacity")
    # past the capacity: env 2 is full (CAP); one more is refused by name
    recs = np.zeros((n_env, 3, W), dtype=np.int64)
    recs[2, :3] = [_rec(-1), _rec(-2), _rec(-3)]
    before = tab.download()
    try:
        tab.append(recs, np.array([0, 0, 3, 0]))
        fail("append past the capacity was not refused")
    except ot.ObjectTableOverflow as ex:
        if "'charge'" not in str(ex) or "env 2" not in str(ex) or "3 row" not in str(ex):
            fail(f"the overflow refusal does not name the table, env and count: {ex}")
    r, c, o = tab.download()
    if not (_eq(r, before[0]) and _eq(c, before[1])) or int(o[2]) != 3 \
            or o[[0, 1, 3]].any():
        fail("append past the capacity wrote a row or miscounted the overflow")
    print(f"append: three ordered appends over counts {list(counts)} equal the "
          f"host's; 3 rows past a full table refused, counted, nothing written")


PATTERNS = ([1, 0, 1, 0, 1], [0, 1, 1, 1, 0], [0, 0, 0], [1, 1, 1], [],
            [0] * CAP, [1] * CAP, [i % 3 == 1 for i in range(CAP)])


def part_compact():
    n_env = len(PATTERNS)
    rows = np.zeros((n_env, CAP, W), dtype=np.int64)
    counts = np.zeros(n_env, dtype=np.int64)
    for e, live in enumerate(PATTERNS):
        for i, l in enumerate(live):
            rows[e, i] = _rec(100 * e + i, int(l))
            rows[e, i, 1] = 7 * i + e           # a second column carried
        counts[e] = len(live)
        if len(live) < CAP:                     # garbage past the count
            rows[e, len(live):, LAY.live] = 1
            rows[e, len(live):, 0] = -9
    tab = ot.DeviceTable(T, n_env)
    tab.upload(rows, counts)
    tab.compact()
    r, c, o = tab.download()
    for e in range(n_env):
        want, wn = ot.compact_rows(T, rows[e], counts[e])
        if c[e] != wn or not _eq(r[e], want):
            fail(f"compact: pattern {PATTERNS[e]} differs from the host's")
    # compacting again changes nothing (no dead row is left)
    tab.compact()
    r2, c2, _ = tab.download()
    if not (_eq(r2, r) and _eq(c2, c)):
        fail("compact: a second compaction moved a row")
    print(f"compact: {n_env} patterns (dead in the middle, at the ends, all, "
          f"none, full tables) equal the host's stable compaction; garbage "
          f"past the count untouched; idempotent")


# ---------------------------------------------------------------------------
# The scenes
# ---------------------------------------------------------------------------
class ChargeScene:
    """A room of planters; real plants (the verb, its clamped fuse) and
    charges laid through THE door ``charges.place`` (a fixture, like the
    verb scene's ``_top_up``) on timers that make this tick's dead rows the
    first, a middle one, the last, all and none, plus a remote charge that
    never fires. ``variant`` shifts every time (envs that differ); variant
    0 alone holds the remote charge, the real plants (their clamped fuse
    outlasts the scene) and a far timer, so variant 1's table empties: all
    dead on one tick."""

    def __init__(self, variant=0):
        from cuda_unit_verbs_check import _verb_level
        from simulation import Simulation
        from simulation.ruleset import ContinuousRealtime
        from simulation.turning import place
        from simulation.unit import Unit
        level, _ = _verb_level(variant)
        sim = Simulation(level, seed=777 + variant, breach_physics=bp,
                         enable_recorder=False, ruleset=ContinuousRealtime())
        sim.set_paused(False)
        self.sim, self.v, self.k = sim, variant, 0
        self.planters = []
        for i in range(3):
            u = sim.get_unit(sim.add_unit(Unit(f"P{i}", x=40.0 + 6 * i,
                                               y=30.0, team=0)))
            place(u, 0)
            u.current_hp = 10000.0
            self.planters.append(u)

    def _lay(self, det, x):
        charges.place(self.sim, charges.PlantedCharge(
            x, 40, "demo_c4", -1, det_tick=det, approach=(39, x)))

    def tick(self):
        sim, k, v = self.sim, self.k, self.v
        t = sim.tick
        if k == 2 + v:                            # dead: first, middle, last
            for j, d in enumerate((10, 30, 10, 30, 10)):
                self._lay(t + d + 1, 34 + 4 * j)
        if k == 3 and v == 0:
            self._lay(None, 80)                   # remote: never fires
        if k == 50 + v:                           # all of these at once
            for j in range(3):
                self._lay(t + 6, 40 + 4 * j)
        if k == 5 and v == 0:                     # real plants, the clamped fuse
            for u in self.planters[:2 + v % 2]:
                sim.plant_charge_intent(u.id, 0)
        if k == 70 and v == 0:
            self._lay(10 ** 6, 60)
        sim.step()
        sim.set_paused(False)
        self.k += 1


def _drivers():
    from cuda_unit_verbs_check import VerbDriver
    return [VerbDriver(0), VerbDriver(3), ChargeScene(0), ChargeScene(1)]


def part_scene(ticks=SCENE_TICKS):
    from simulation.simulation import Simulation
    drivers = _drivers()
    n_env = len(drivers)
    cen = Counter()
    hold = {}
    orig_step, orig_tick = Simulation.step, charges.tick

    def step(sim, *a, **kw):
        hold.setdefault(id(sim), {})
        r = orig_step(sim, *a, **kw)
        h = hold[id(sim)]
        if "pre" not in h:                        # charges.tick did not run
            h["pre"] = h["marked"] = h["post"] = None
            h["mid"] = list(sim.planted_charges)
        return r

    def tick(sim, __o=orig_tick):
        h = hold[id(sim)]
        objs = list(sim.planted_charges)
        h["mid"] = objs
        h["pre"] = ot.records(T, objs)            # before this tick's fires
        r = __o(sim)
        h["marked"] = ot.records(T, objs)         # the same objects, fires marked
        h["post"] = list(sim.planted_charges)
        return r

    Simulation.step = step
    charges.tick = tick
    tab = ot.DeviceTable(T, n_env)
    alone = ot.DeviceTable(T, 1)
    try:
        for t in range(ticks):
            heads, mids, marks, posts, apps = [], [], [], [], []
            for e, d in enumerate(drivers):
                head = list(d.sim.planted_charges)   # the table at the tick's head
                hold[id(d.sim)] = {}
                d.tick()                          # its fixtures, then the step
                sim = d.sim
                h = hold[id(sim)]
                mid = h["mid"]
                # the tick only appended before charges.tick (place appends)
                if mid[:len(head)] != head or any(a is not b for a, b in zip(head, mid)):
                    fail(f"scene t{t} env {e}: the list at the 9e entry does "
                         f"not start with the list at the step's head")
                    return
                add = mid[len(head):]
                heads.append(head)
                apps.append(ot.records(T, add))
                mids.append(mid)
                marks.append(h["marked"])
                posts.append(sim.planted_charges)
                # IDENTITY: unpack(pack(list)) attribute for attribute
                rows, n = ot.pack(T, sim.planted_charges)
                back = ot.unpack(T, rows, n)
                for x, y in zip(sim.planted_charges, back):
                    if any(getattr(x, s) != getattr(y, s)
                           for s in charges.PlantedCharge.__slots__):
                        fail(f"scene t{t} env {e}: unpack(pack) differs: {x} / {y}")
                        return
                cen["ticks with live charges"] += bool(n)
                cen["max count"] = max(cen["max count"], n)
                if n == CAP:
                    cen["a full table"] += 1
                if add:
                    cen["ticks with appends"] += 1
                    if head:
                        cen["an append after held rows"] += 1
                    if len(add) > 1:
                        cen["several rows appended on one tick"] += 1
                if h["marked"] is not None:
                    live = [r[LAY.live] for r in h["marked"]]
                    if 0 in live:
                        cen["ticks with fires"] += 1
                        dead = [i for i, l in enumerate(live) if not l]
                        if 0 in dead and len(dead) < len(live):
                            cen["dead first"] += 1
                        if len(live) - 1 in dead and len(dead) < len(live):
                            cen["dead last"] += 1
                        if any(0 < i < len(live) - 1 for i in dead):
                            cen["dead in the middle"] += 1
                        if len(dead) == len(live):
                            cen["all dead"] += 1
                    else:
                        cen["9e with none dead"] += 1
            # the batch on the device: head + append == the list at 9e
            hr, hc = ot.pack_batch(T, heads)
            k = max(1, max(len(a) for a in apps))
            ar = np.zeros((n_env, k, W), dtype=np.int64)
            an = np.zeros(n_env, dtype=np.int64)
            for e, a in enumerate(apps):
                if a:
                    ar[e, :len(a)] = np.array(a, dtype=np.int64)
                an[e] = len(a)
            tab.upload(hr, hc)
            r0, c0, _ = tab.download()
            if not (_eq(r0, hr) and _eq(c0, hc)):
                fail(f"scene t{t}: the device round trip is not bit for bit")
                return
            tab.append(ar, an)
            mr, mc = ot.pack_batch(T, mids)
            r1, c1, _ = tab.download()
            if not (_eq(r1, mr) and _eq(c1, mc)):
                e = int(np.nonzero((c1 != mc) | (r1 != mr).any(axis=(1, 2)))[0][0])
                fail(f"scene t{t} env {e}: the device append differs from the "
                     f"host list at charges.tick's entry")
                return
            # the compaction: the marked list -> the pruned list
            kr = mr.copy()
            for e, m in enumerate(marks):
                if m is not None:
                    kr[e, :len(m)] = np.array(m, dtype=np.int64) if m else kr[e, :0]
            tab.upload(kr, mc)
            tab.compact()
            pr, pc = ot.pack_batch(T, posts)
            r2, c2, _ = tab.download()
            if not (_eq(r2, pr) and _eq(c2, pc)):
                e = int(np.nonzero((c2 != pc) | (r2 != pr).any(axis=(1, 2)))[0][0])
                fail(f"scene t{t} env {e}: the device compaction differs from "
                     f"the host's prune")
                return
            # env i of N == env i alone (an N = 1 table)
            for e in range(n_env):
                alone.upload(hr[e:e + 1], hc[e:e + 1])
                alone.append(ar[e:e + 1], an[e:e + 1])
                a1 = alone.download()
                alone.upload(kr[e:e + 1], mc[e:e + 1])
                alone.compact()
                a2 = alone.download()
                if not (_eq(a1[0][0], r1[e]) and a1[1][0] == c1[e]
                        and _eq(a2[0][0], r2[e]) and a2[1][0] == c2[e]):
                    fail(f"scene t{t} env {e}: env {e} of {n_env} != env {e} alone")
                    return
    finally:
        Simulation.step = orig_step
        charges.tick = orig_tick
    need = ("ticks with live charges", "a full table", "ticks with appends",
            "an append after held rows", "several rows appended on one tick",
            "ticks with fires", "dead first",
            "dead last", "dead in the middle", "all dead", "9e with none dead")
    for b in need:
        if not cen[b]:
            fail(f"scene: the census branch {b!r} was never taken")
    print(f"scene: {n_env} envs x {ticks} ticks, identity / round trip / "
          f"append / compaction / alone equal every tick; census "
          f"{dict(sorted(cen.items()))}")


# ---------------------------------------------------------------------------
def main(parts=None) -> int:
    if not getattr(bp, "HAS_CUDA", False):
        print("OBJECT_TABLE_RESULT: FAIL (not the CUDA build)")
        return 1
    parts = parts or os.environ.get("OBJECT_TABLE_PARTS", ",".join(PARTS)).split(",")
    t0 = time.perf_counter()
    for p in parts:
        try:
            {"tables": part_tables, "trip": part_trip, "append": part_append,
             "compact": part_compact, "scene": part_scene}[p]()
        except Exception as ex:          # a part's crash fails it, the rest run
            import traceback
            traceback.print_exc()
            fail(f"part {p} raised {type(ex).__name__}: {ex}")
    print(f"total {time.perf_counter() - t0:.0f} s")
    ok = not FAILS
    print("OBJECT_TABLE_RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def bench(reps: int = 200) -> int:
    """The TIMING table: the charge table's append (half the capacity, one
    call) + the stable compaction (every other row dead), medians over
    ``reps``, at N = 1, 8, 64; each rep re-uploads the start state (not
    timed)."""
    print("TIMING (median ms over %d reps)" % reps)
    print("| op | N=1 | N=8 | N=64 |")
    res = {}
    for n in (1, 8, 64):
        tab = ot.DeviceTable(T, n)
        rows = np.zeros((n, CAP, W), dtype=np.int64)
        counts = np.full(n, CAP // 2, dtype=np.int64)
        for i in range(CAP // 2):
            rows[:, i] = _rec(i)
        k = CAP // 2
        recs = np.zeros((n, k, W), dtype=np.int64)
        for j in range(k):
            recs[:, j] = _rec(100 + j, j % 2)
        nn = np.full(n, k, dtype=np.int64)
        ta, tc, tb = [], [], []
        for _ in range(reps):
            tab.upload(rows, counts)
            s = time.perf_counter()
            tab.append(recs, nn)
            m = time.perf_counter()
            tab.compact()
            tab.download()
            e = time.perf_counter()
            ta.append(m - s)
            tc.append(e - m)
            tb.append(e - s)
        res[n] = (np.median(ta), np.median(tc), np.median(tb))
    for i, name in enumerate(("append (H2D + kernel + overflow check)",
                              "compact (kernel + D2H)", "append + compact")):
        print(f"| {name} | " + " | ".join(f"{res[n][i] * 1e3:.3f}" for n in (1, 8, 64)) + " |")
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "bench":
        sys.exit(bench())
    sys.exit(main())
