"""#29 P7e.4 -- THE DEVICE CONDUCTOR AT N = 1 (``simulation.conductor``): a
whole reference tick's unit side run ON THE DEVICE, every piece of state
resident across ticks, chained with the resident physics tick, against a
HOST RE-SIMULATION run beside it -- ``(field_digest, tick_digest)`` equal at
EVERY tick, tol 0.

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_conductor.py).

THE GATE (:func:`run_gate`): a REFERENCE sim plays on the host (the CPU
path, the oracle) and a TWIN sim, built identically, is handed to a
``DeviceConductor``. Each tick the reference's intents -- taken from its own
intent tap (``Simulation.attach_intent_log``) for a hand scene, or from a
recording's intent rows for a recorded match (``match_replay``) -- are the
conductor's latches; the reference steps, the conductor ticks, the device
state is downloaded onto the twin (the gate's download, the only one) and the
two sims' digests (``match_replay.match_digests``) are compared, then the
state in NO digest (rounds, grenades, noise list, blast sources, the edit
door's window, the horde cache) against the reference's record functions, so
a divergence there is caught at its tick, not when it reaches a digested
plane. A refusal raised by the conductor is a REFUSED tick: the run stops
there with the name (never compared, never continued), and the gate reports
how many ticks ran equal before it.

PART SCENES (S1) -- the hand scenes of the existing device checks, their
between-tick fixtures (host writes the conductor does not model by design)
hoisted into the setup or dropped, driven by intents alone: payloads (every
hand-grenade kind, payload rounds in their firing tick and on landing,
charges of every applied payload row), hazards (timed charges, the heat
source chain, a bus driving a pump, vents, a door), sprays (overlapping
cones, a wall chewed to one count), the horde (bashing zombies, a fire ring,
the controller's request sets), signals (sensors, nodes, pumps, airlocks,
USE on a bus-driven door), fire (walls and doors burning through) and
walking (the movement box: throttles, aims, hooks). Every tick equal.
PART RESIDENT -- the same gate with a RESTAGE PROBE: every host-to-device
restaging face (the store, its tables, an object table, the horde cache,
the map's from_host) counted while the conductor ticks: zero after the
build. ``CONDUCTOR_BREAK=restage`` re-uploads the store from the twin every
tick (the probe must fail: shown red once).
PART MATCH (S2) -- a short ship_demo match recorded through the game's own
recorder (``tools/run_match.run_match(record=True)``, the bots driving),
replayed: the reference by the ``replay`` controller (verifying the
recording's checkpoints), the conductor from the recording's intent rows.
PART PINNED (S3) -- the two pinned ship_demo matches (``bots``, ``bots_v1``,
``tests/_pinned_matches``) recorded and fed to the conductor to their end,
whatever tick that is; the recording play itself is the reference
(:func:`live_gate`: each match played ONCE on the host).
PART SYNCS (S4) -- the host syncs of a conductor tick counted (see
:func:`count_syncs`).

PROPERTY: the device conductor is the reference tick: every digested plane,
unit row, synced event, entity and swarm section, and the cross-tick state in
no digest, equal at every tick; the state lives on the device across ticks.
BREAKS IF a slot runs out of order, a face reads a host input instead of a
device one, a latch writes a column the intent does not, the tick row does
not advance on the device, a table is restaged from the host between ticks,
or any face's law differs from the reference's.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src", _ROOT / "tools", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import level_loader  # noqa: E402
from simulation import conductor as cd  # noqa: E402

FAILS: list = []
ONE = 65536
SCENE_TICKS = int(os.environ.get("CONDUCTOR_SCENE_TICKS", "160"))
MATCH_TICKS = int(os.environ.get("CONDUCTOR_MATCH_TICKS", "300"))
BREAK = os.environ.get("CONDUCTOR_BREAK", "")


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", str(msg).encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# The intent tap and the digests
# ---------------------------------------------------------------------------
class Tap:
    """The reference's intent sink: this tick's (kind, args), issue order."""

    def __init__(self):
        self.now = []

    def record(self, tick, kind, args):
        if kind == "drive_zombies":
            args = (list(args[0]),)
        self.now.append((kind, tuple(args)))

    def take(self):
        out, self.now = self.now, []
        return out


def digests(sim):
    import match_replay
    return match_replay.match_digests(sim)


def locate(ref, twin) -> str:
    """The first difference between the two sims' snapshots, by name."""
    from field_ab_harness import _diff_entity_state, _diff_unit_state, capture_snapshot
    from field_digest import DIGEST_FIELDS, DIGEST_SECTIONS
    a, b = capture_snapshot(ref), capture_snapshot(twin)
    for name, _dt in DIGEST_FIELDS:
        x, y = np.asarray(a[name]), np.asarray(b[name])
        if x.shape != y.shape or not np.array_equal(x, y):
            idx = np.argwhere(x != y) if x.shape == y.shape else []
            where = tuple(int(v) for v in idx[0]) if len(idx) else "shape"
            return (f"field {name}: {len(idx)} cell(s) differ, first {where}: ref "
                    f"{x[tuple(idx[0])] if len(idx) else x.shape} device "
                    f"{y[tuple(idx[0])] if len(idx) else y.shape}")
    for name, _dt, _p in DIGEST_SECTIONS:
        x, y = a.get(name), b.get(name)
        if (x is None) != (y is None) or (x is not None and not np.array_equal(x, y)):
            return f"section {name} differs"
    from field_ab_harness import ENTITY_DIGEST_KEY, UNIT_DIGEST_KEY
    d = _diff_unit_state(ref.tick, a[UNIT_DIGEST_KEY], b[UNIT_DIGEST_KEY])
    if d:
        return "; ".join(d[:2])[:600]
    d = _diff_entity_state(ref.tick, a[ENTITY_DIGEST_KEY], b[ENTITY_DIGEST_KEY])
    if d:
        return "; ".join(d[:2])[:600]
    return "the digests differ but no field, unit or entity was located (swarm?)"


def nondigest_diff(conductor, ref) -> str:
    got = conductor.nondigest_records()
    want = cd.reference_records(ref)
    for k in ("rounds", "grenades", "noise", "sources", "source_tiles"):
        g = [tuple(int(x) for x in r) for r in got[k]]
        if g != list(want[k]):
            return f"{k}: device {g[:3]}... ({len(g)}) != reference {list(want[k])[:3]}... " \
                   f"({len(want[k])})"
    if got["door"] != want["door"]:
        return (f"the edit door's window: device {len(got['door'][0])} edits != reference "
                f"{len(want['door'][0])}")
    if got["horde"] != want["horde"]:
        return "the horde cache differs"
    return ""


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
class RestageProbe:
    """Counts every host-to-device restaging face while armed (PART RESIDENT)."""

    def __init__(self):
        from simulation import horde_cache, object_tables, unit_store
        from simulation.gamemap import GameMap
        self.n = 0
        self.on = False
        self._saved = []
        for owner, attr in ((unit_store.DeviceStore, "upload"),
                            (unit_store.DeviceStore, "upload_tables"),
                            (object_tables.DeviceTable, "upload"),
                            (object_tables.DeviceTable, "stage_lists"),
                            (horde_cache.DeviceHordeCache, "stage_lists"),
                            (GameMap, "from_host")):
            orig = getattr(owner, attr)

            def wrapped(*a, __o=orig, __n=f"{owner.__name__}.{attr}", **k):
                if self.on:
                    self.n += 1
                    self.last = __n
                return __o(*a, **k)
            setattr(owner, attr, wrapped)
            self._saved.append((owner, attr, orig))

    def close(self):
        for owner, attr, orig in self._saved:
            setattr(owner, attr, orig)


def run_gate(name, ref, twin, ticks, intents_of=None, step_ref=None, probe=None,
             compare_every=1, census=None):
    """Play ``ref`` and the conductor on ``twin`` for ``ticks`` ticks (or until
    ``step_ref`` reports the match over); compare every tick. ``intents_of(k)``
    the tick's intents (default: the reference's tap); ``step_ref()`` steps
    the reference (default ``ref.step()``) and returns True when it ended.
    Returns (ticks equal, refused name or None)."""
    tap = None
    if intents_of is None:
        tap = Tap()
        ref.attach_intent_log(tap)
    if probe is not None:
        probe.n = 0
    if BREAK == "slot":                  # the S1 red: slot 2b (the statuses) left out
        from simulation import unit_kernels
        unit_kernels.status_tick = lambda store, ring: np.zeros(store.shape[0], np.int64)
    t0 = time.perf_counter()
    c = cd.DeviceConductor(twin, bp=bp)
    t_build = time.perf_counter() - t0
    d0 = digests(ref), digests(twin)
    if d0[0] != d0[1]:
        fail(f"{name}: the twin differs from the reference before tick 0: {locate(ref, twin)}")
        return 0, None
    n_equal = 0
    t_tick = 0.0
    for k in range(ticks):
        tick = int(ref.tick)
        if step_ref is None:
            if hasattr(ref, "_conductor_before"):
                ref._conductor_before(k)
            ref.step()
            ended = False
        else:
            ended = step_ref(k)
        intents = tap.take() if tap is not None else intents_of(tick)
        try:
            ts = time.perf_counter()
            if probe is not None:
                probe.on = True
            if BREAK == "restage":
                from simulation import unit_store
                c.store.upload(unit_store.pack(twin.units, c.U)[None])
            c.latch(intents)
            c.tick()
            if probe is not None:
                probe.on = False
            t_tick += time.perf_counter() - ts
        except Exception as e:                        # a REFUSED tick: never compared
            if probe is not None:
                probe.on = False
            name_ = type(e).__name__
            if os.environ.get("CONDUCTOR_TRACE"):
                import traceback
                traceback.print_exc()
            print(f"  {name}: tick {tick} REFUSED by the conductor ({name_}: {e}) after "
                  f"{n_equal} equal ticks")
            return n_equal, f"{name_}: {e}"
        if probe is not None and probe.n:
            fail(f"{name} tick {tick}: the conductor restaged {probe.n} time(s) from the host "
                 f"between ticks (last {probe.last})")
            return n_equal, None
        if (k + 1) % compare_every == 0 or ended:
            c.download()
            a, b = digests(ref), digests(twin)
            if a != b:
                fail(f"{name} tick {tick}: digests differ: {locate(ref, twin)}")
                return n_equal, None
            nd = nondigest_diff(c, ref)
            if nd:
                fail(f"{name} tick {tick}: {nd}")
                return n_equal, None
        if census is not None:
            census(ref, c)
        n_equal += 1
        if ended:
            break
    if tap is not None:
        ref.attach_intent_log(None)
    print(f"  {name}: {n_equal} ticks equal (build {t_build:.2f} s, conductor "
          f"{1000 * t_tick / max(1, n_equal):.1f} ms/tick incl. latch)")
    return n_equal, None


# ---------------------------------------------------------------------------
# PART SCENES: the hand scenes, fixtures hoisted or dropped, intents only
# ---------------------------------------------------------------------------
def _payloads():
    import cuda_unit_payloads_check as pg

    def make():
        ps = pg.PayloadScene(0)
        sim = ps.sim
        s = 2
        g = sim.gmap
        from simulation.gases import INERT_N2, O2
        sl = (slice(21, 25), slice(86, 96))
        o2 = g.gas[O2][sl].astype(np.int64)
        mv = (o2 * 9) // 10
        g.gas[O2][sl] = (o2 - mv).astype(np.int32)
        g.gas[INERT_N2][sl] = (g.gas[INERT_N2][sl].astype(np.int64) + mv).astype(np.int32)
        from simulation.physics import _BLAST_WALL_MATERIALS
        for y, x in np.argwhere(np.isin(g.material, _BLAST_WALL_MATERIALS)):
            if 3 <= y < pg.SH - 3 and g.wall_hp[y, x] > 30 * ONE:
                g.wall_hp[y, x] = 30 * ONE
        for j, am in enumerate(ps.charge_ammo()[:12]):
            ps._lay(am, 60 + 3 * (j % 12), 38 + (j // 12) * 2, s + 8 + (j % 3),
                    ps.who[f"T{j % ps.n_kinds}"].id if j % 4 else -1)

        def before(k):
            w = ps.who
            if k == s:
                for i in range(ps.n_kinds):
                    ps._throw(w[f"T{i}"], i, 18, 12)
            if k == s + 1:
                for i in range(0, ps.n_kinds, 2):
                    ps._throw(w[f"U{i}"], i, 17, 13)
            if k == s + 30:
                for i in range(ps.n_kinds):
                    ps._throw(w[f"T{i}"], (i + 3) % ps.n_kinds, 14, 12)
            for nm in ("S0", "S1", "S2", "F0", "F1", "F2", "K", "P"):
                sim.set_trigger(w[nm].id, k in (s, s + 1, s + 40) and w[nm].alive)
        sim._conductor_before = before
        return sim
    return make


def _hazards(variant):
    def make():
        from _hazard_scene import HazardScene
        hs = HazardScene(variant, bp=bp)
        return hs.sim
    return make


def _sprays():
    import cuda_unit_sprays_check as sc

    def make():
        d = sc.SprayDriver(0)
        sim, w = d.sim, d.who
        for y in range(29, 34):
            if sim.gmap.wall_hp[y, 27] > 0:
                sim.gmap.wall_hp[y, 27] = 1

        def before(k):
            for u in d.hosing:
                if u.alive:
                    sim.set_trigger(u.id, True)
            sim.set_trigger(w["G"].id, k == 0)
            if w["M"].alive:
                sim.melee_intent(w["M"].id)
        sim._conductor_before = before
        return sim
    return make


def _horde(controller):
    import cuda_unit_horde_check as hc

    def make():
        orig = hc._bare_sim
        hc._bare_sim = lambda tm, tile=1.0, entities=(), physics=True: orig(
            tm, tile, entities, physics=True)
        try:
            hs = hc.HordeScene(0, controller=controller)
        finally:
            hc._bare_sim = orig
        sim = hs.sim

        def before(k):
            if hs.controller is not None:
                ids = [u.id for u in sim.units if u.alive and u.is_zombie]
                if k % 9 != 8:
                    if k % 5 == 3:
                        ids = ids[1:]
                    if ids:
                        sim.drive_zombies(ids)
        sim._conductor_before = before
        return sim
    return make


def _signals():
    import cuda_unit_signals_check as sg

    def make():
        sc_ = sg.SignalScene(0)
        sim = sc_.sim

        def before(k):
            if k in (8, 70, 140):
                sim.use_intent(sc_.u4.id)
        sim._conductor_before = before
        return sim
    return make


def _fire(variant):
    import cuda_unit_topology_check as tc

    def make():
        fs = tc.FireScene(variant)
        sim = fs.sim
        del sim._step_physics                   # the scene's slot-7 appends: a fixture
        return sim
    return make


def _walking():
    import cuda_unit_movement_check as mc

    def make():
        d = mc.BoxDriver(0)
        sim = d.sim
        quarter = mc.TWO_PI_Q16 // 16

        def before(k):
            for i, u in enumerate(d.team0):
                if not u.alive:
                    continue
                dd = (k // 45 * 5 + i * 3 + d.seed) % 16
                if i == 3:
                    dd = 12 if k % 160 < 80 else 4
                dx, dy = mc._dir(dd * quarter)
                t = (0, ONE // 3, ONE // 2 + 77, ONE)[(k // 40 + i) % 4]
                sim.set_move_dir(u.id, dx, dy, t)
                if i % 3 == 0:
                    sim.set_aim(u.id, ONE, 0)
                elif i % 3 == 1:
                    ax, ay = mc._dir((k * 3000 + i * 50000) % mc.TWO_PI_Q16)
                    sim.set_aim(u.id, ax, ay)
            for j, u in enumerate(d.team1):
                if u.alive and j % 2 == 0:
                    dx, dy = mc._dir(((k // 25 + j) % 16) * quarter)
                    sim.set_move_dir(u.id, dx, dy, ONE // 4)
        sim._conductor_before = before
        return sim
    return make


SCENES = {
    "payloads": (_payloads(), None),
    "hazards0": (_hazards(0), None),
    "hazards2": (_hazards(2), None),
    "sprays": (_sprays(), None),
    "horde": (_horde(None), None),
    "horde_ctrl": (_horde("zombies"), None),
    "signals": (_signals(), None),
    "fire0": (_fire(0), None),
    "fire1": (_fire(1), None),
    "walking": (_walking(), None),
}


def part_scenes(names=None, probe=False):
    pr = RestageProbe() if probe else None
    try:
        for name in (names or list(SCENES)):
            make, _ = SCENES[name]
            ref, twin = make(), make()
            n, refused = run_gate(name, ref, twin, SCENE_TICKS, probe=pr)
            if refused:
                fail(f"{name}: refused at tick {int(ref.tick)}: {refused}")
            elif n < SCENE_TICKS and not FAILS:
                fail(f"{name}: ran {n} of {SCENE_TICKS} ticks")
    finally:
        if pr is not None:
            pr.close()


def part_resident():
    part_scenes(["payloads", "hazards0", "walking"], probe=True)


# ---------------------------------------------------------------------------
# PARTS MATCH / PINNED: recorded matches replayed through the conductor
# ---------------------------------------------------------------------------
def record_match(which=None, seed=None, max_ticks=None):
    """A ship_demo match recorded through the game's own recorder path
    (``run_match(record=True)``: the bots drive, ``match_replay`` taps)."""
    from run_match import run_match
    import _pinned_matches as pm
    level = pm.ship_demo()
    spec = pm.spec_of(level, which) if which else None
    if max_ticks is not None:
        from dataclasses import replace
        sp = spec if spec is not None else level.match
        spec = replace(sp, time_limit_s=max_ticks / 24.0)
    t0 = time.perf_counter()
    out = run_match(level, spec, seed=seed, record=True)
    print(f"  recorded {which or 'bots'}: {out.recording.end_tick} ticks, "
          f"{out.recording.n_intents()} intents ({time.perf_counter() - t0:.1f} s)")
    return level, out.recording


def replay_gate(name, level, rec, ticks=None):
    """The reference replays ``rec`` by the replay controller (its own
    checkpoints verified); the conductor's twin is a fresh session of the same
    spec and seed, fed the recording's intent rows."""
    from simulation.match_session import MatchSession
    ref_s = MatchSession(level, rec.spec, seed=rec.seed, replay=rec, breach_physics=bp)
    twin_s = MatchSession(level, rec.spec, seed=rec.seed, replay=rec, breach_physics=bp)
    ref, twin = ref_s.sim, twin_s.sim
    end = rec.end_tick if ticks is None else min(rec.end_tick, ticks)

    def step_ref(k):
        ref_s.tick()
        return ref_s.is_terminal()

    n, refused = run_gate(name, ref, twin, end, intents_of=lambda t: rec.intents.get(t, ()),
                          step_ref=step_ref)
    if refused:
        print(f"  {name}: REFUSED after {n} equal ticks: {refused}")
    return n, refused, end


def part_match():
    level, rec = record_match(max_ticks=MATCH_TICKS)
    n, refused, end = replay_gate("match", level, rec)
    if refused:
        fail(f"match: refused after {n} equal ticks: {refused}")
    elif n < end:
        fail(f"match: ran {n} of {end} ticks")


PINNED_RESULT: dict = {}


def live_gate(name, which=None, seed=None):
    """A ship_demo match RECORDED through the game's recorder path
    (``MatchSession.record``, the bots driving, as ``run_match(record=True)``
    records it) and fed to the conductor from the recording's intent rows AS
    THEY ARE WRITTEN: the recording play itself is the reference, so the
    match is played ONCE on the host (DEVICE CHECK PLAYS; part MATCH keeps
    the other reference, the replay controller re-simulating a recording).
    Runs until the match ends, however long it is (the clock bounds it).
    Returns (ticks equal, refused or None, the ending tick or None)."""
    import _pinned_matches as pm
    from simulation.match_session import MatchSession
    level = pm.ship_demo()
    spec = pm.spec_of(level, which) if which else None
    t0 = time.perf_counter()
    ref_s = MatchSession(level, spec, seed=seed, breach_physics=bp)
    twin_s = MatchSession(level, spec, seed=seed, breach_physics=bp)
    ref_s.record()
    rec = ref_s._recorder.recording
    limit = int(round(float(ref_s.spec.time_limit_s) * int(ref_s.sim._tps)))

    def step_ref(k):
        ref_s.tick()
        return ref_s.is_terminal()

    # the bound is a hang guard past the clock, never the match's length
    n, refused = run_gate(name, ref_s.sim, twin_s.sim, limit + 2,
                          intents_of=lambda t: rec.intents.get(t, ()), step_ref=step_ref)
    end = int(ref_s.sim.tick) if ref_s.is_terminal() else None
    if end is not None:
        ref_s.finish()
    print(f"  {name}: recorded and gated live, {n} ticks, ended at {end} "
          f"({time.perf_counter() - t0:.1f} s, one host play)")
    return n, refused, end


def part_pinned(which=("bots", "bots_v1")):
    """The pinned matches to their end, whatever tick that is: every tick
    equal, the match ENDED (its ending tick the reference's), or refused by
    name."""
    for w in which:
        n, refused, end = live_gate(w, None if w == "bots" else w)
        PINNED_RESULT[w] = (n, refused, end)
        if refused:
            fail(f"pinned {w}: REFUSED after {n} equal ticks: {refused}")
        elif end is None:
            if not FAILS:
                fail(f"pinned {w}: the match did not end within its clock ({n} ticks)")
        elif n < end:
            fail(f"pinned {w}: ran {n} of {end} ticks")
        else:
            print(f"  pinned {w}: every one of {end} ticks equal")



# ---------------------------------------------------------------------------
# PART SYNCS (S4): the host syncs of a conductor tick
# ---------------------------------------------------------------------------
#: The named syncs a conductor tick may make: the physics' two (the EOS
#: stage's max n_sub, the tail's books D2H) and THE ONE BOOKS READ; and, on a
#: tick whose order latch holds an order, the order table's staging (the
#: latch's H2D waits for its pinned buffer).
NAMED_SYNCS = ("max_n_sub", "books", "books_read")
ORDER_SYNC = "1b' orders"


class SyncCount:
    """Counts a conductor tick's host syncs: the C++ unit side and the
    resident tick (``bp.host_sync_log_begin`` / ``_end``: every counted_sync,
    counted_event_sync, tick_sync_point and the books read, labelled by the
    conductor's call line) and every CuPy download (``cupy.ndarray.get``,
    ``cupy.asnumpy``) made inside the tick."""

    def __init__(self):
        import cupy as cp
        self.cp = cp
        self.py = []
        self._get, self._asnp = cp.ndarray.get, cp.asnumpy
        me = self

        def get(a, *k, **kw):
            me.py.append("cupy.get " + bp.__dict__.get("_label", ""))
            return me._get(a, *k, **kw)

        def asnumpy(a, *k, **kw):
            me.py.append("cupy.asnumpy")
            return me._asnp(a, *k, **kw)
        cp.ndarray.get = get
        cp.asnumpy = asnumpy

    def close(self):
        self.cp.ndarray.get = self._get
        self.cp.asnumpy = self._asnp


def count_syncs(name, ref, twin, ticks, intents_of=None, step_ref=None):
    """Play the gate's way for ``ticks`` ticks and log each conductor tick's
    syncs (the gate's download outside the count). Returns the per-tick logs."""
    tap = None
    if intents_of is None:
        tap = Tap()
        ref.attach_intent_log(tap)
    c = cd.DeviceConductor(twin, bp=bp)
    if BREAK == "sync":                  # the S4 red: a face's download inside the tick
        walk = c._walk

        def broken(L):
            c.noise.check("break")
            c.noise.download()
            walk(L)
        c._walk = broken
    sc = SyncCount()
    logs = []
    try:
        for k in range(ticks):
            tick = int(ref.tick)
            if step_ref is None:
                if hasattr(ref, "_conductor_before"):
                    ref._conductor_before(k)
                ref.step()
                ended = False
            else:
                ended = step_ref(k)
            intents = tap.take() if tap is not None else intents_of(tick)
            c.latch(intents)
            had_orders = any(c._orders)
            sc.py = []
            bp.host_sync_log_begin()
            try:
                c.tick()
            finally:
                cx = list(bp.host_sync_log_end())
            logs.append((tick, cx + sc.py, had_orders))
            if ended:
                break
    finally:
        sc.close()
    return logs


def judge_syncs(name, logs):
    from collections import Counter
    per = Counter()
    bad = []
    # the first conductor tick grows the faces' scratch (an allocation, a
    # sync, once per process): warm-up, reported, never steady state
    warm = logs[0][1] if logs else []
    print(f"  {name}: warm-up tick {logs[0][0] if logs else '-'}: {len(warm)} syncs")
    logs = logs[1:]
    for tick, log, had_orders in logs:
        for entry in log:
            label, _, what = entry.partition(" | ")
            per[entry] += 1
            if what in NAMED_SYNCS:
                continue
            if had_orders and label == ORDER_SYNC:
                continue
            bad.append((tick, entry))
    n = len(logs)
    tot = sum(len(lg) for _t, lg, _o in logs)
    print(f"  {name}: {n} ticks, {tot / max(1, n):.2f} host syncs per conductor tick")
    for entry, k in sorted(per.items(), key=lambda kv: -kv[1]):
        print(f"    {k / max(1, n):6.2f} / tick  {entry}")
    if bad:
        fail(f"{name}: {len(bad)} sync(s) outside the named ones, first at tick {bad[0][0]}: "
             f"{bad[0][1]}")


def part_syncs():
    for name in ("payloads", "hazards0"):
        make, _ = SCENES[name]
        judge_syncs(name, count_syncs(name, make(), make(), 60))
    level, rec = record_match(max_ticks=200)
    from simulation.match_session import MatchSession
    ref_s = MatchSession(level, rec.spec, seed=rec.seed, replay=rec, breach_physics=bp)
    twin_s = MatchSession(level, rec.spec, seed=rec.seed, replay=rec, breach_physics=bp)

    def step_ref(k):
        ref_s.tick()
        return ref_s.is_terminal()
    logs = count_syncs("ship_demo", ref_s.sim, twin_s.sim, rec.end_tick,
                       intents_of=lambda t: rec.intents.get(t, ()), step_ref=step_ref)
    judge_syncs("ship_demo", logs)

PARTS = {"scenes": part_scenes, "resident": part_resident, "match": part_match,
         "pinned": part_pinned, "syncs": part_syncs}


def main(parts=None) -> int:
    """Run ``parts`` (default: all) in ONE process, each judged alone
    (``_check_parts.run_parts``: its own FAILS, a crash failing that part
    only); ``scene:<name>`` runs one hand scene."""
    import _check_parts
    from simulation import physics_runner as pr
    pr.set_residency(False)               # the reference runs the CPU path (the oracle)
    t0 = time.perf_counter()
    names = list(parts or list(PARTS))
    table = dict(PARTS)
    for name in names:
        if isinstance(name, str) and name.startswith("scene:"):
            table[name] = (lambda _n=name[6:]: part_scenes([_n]))
    _check_parts.run_parts(names, table, FAILS, PINNED_RESULT.clear)
    print(f"{time.perf_counter() - t0:.1f} s")
    print("CONDUCTOR_RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
