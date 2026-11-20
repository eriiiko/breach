"""#29 P7e.5 -- THE CONDUCTOR'S BATCH (``simulation.conductor.DeviceConductor``
over N envs): env i of a batch of N EQUAL to env i run alone by the same
conductor at N = 1, at EVERY tick -- ``(field_digest, tick_digest)`` and the
cross-tick state in no digest (``nondigest_records``) -- tol 0. Env i alone is
itself held equal to the CPU reference by ``tests/cuda_conductor_check.py``
(P7e.4's gate, unchanged); PART SWARM here also holds the swarm-bearing solo
run equal to the CPU reference (its launch words now come from the device tick
row).

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_conductor_batch.py).

THE GATE (:func:`batch_gate`): each env's intents per tick are fixed first
(a hand scene's from a HOST reference's intent tap -- the scene's own
schedule, delayed by a per-env shift so the envs' intents differ; a recorded
match's from its recording's intent rows; none for a level without units);
every env is played ALONE through a conductor at N = 1, its per-tick
fingerprint kept; then the envs are played together in ONE conductor and each
env's fingerprint compared with its solo run at every tick. An env whose
intents end (its match ended) keeps ticking with no intents, alone and in the
batch alike, to the batch's last tick. A refusal raised by a conductor stops
that run with its name (never compared past it).

PART SCENES -- the hand scenes, N = 2 and N = 4, the envs' intents different
(a per-env delay of the scene's schedule; where two scenes share a level --
the hazard variants, the horde with and without its controller, the fire
variants -- both in one batch).
PART MATCHES -- ship_demo matches recorded through the game's recorder with
DIFFERENT seeds per env (``bots`` and ``bots_v1`` specs), replayed to their
end in one batch (envs end at different ticks), N = 2 and N = 4.
PART SWARM -- larva_studio (two broods) with different seeds per env (the
broods' scatter differs), swarm units alive, N = 2 and N = 4; env 0 alone
against the CPU reference too.
PART SWAP -- an env's state never depends on its batch index or its
neighbours: the same envs in another order, each env's fingerprints
unchanged. ``CONDUCTOR_BATCH_BREAK=index`` (the env index leaks into its
draw key) must turn it red.
PART SYNCS -- the host syncs of a steady batch tick do not depend on N
(N = 1, 2, 4 of one scene; the named reads only: ``max_n_sub``, the books
read). ``CONDUCTOR_BATCH_BREAK=sync`` (a download per env inside the walk)
must turn it red.
PART FAULT -- a fault in env k names env k: a beam (hitscan) trigger in env 2
of 4 is refused by name (``ShotRefused``) with ``.env == 2`` and the batch is
refused. ``CONDUCTOR_BATCH_BREAK=layout`` (the count block read env-major)
must turn it red.

PROPERTY: the batch is N independent conductors in one walk. BREAKS IF a face
reads or writes another env's rows, planes, tables or lists, a latch lands on
the wrong env, an env's draws or tick row depend on its index, a refusal
names the wrong env, or a sync per env enters the walk.
"""
from __future__ import annotations

import hashlib
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

import cuda_conductor_check as cc  # noqa: E402
from simulation import conductor as cd  # noqa: E402

FAILS: list = []
SCENE_TICKS = int(os.environ.get("CONDUCTOR_SCENE_TICKS", "160"))
SWARM_TICKS = int(os.environ.get("CONDUCTOR_SWARM_TICKS", "200"))
BREAK = os.environ.get("CONDUCTOR_BATCH_BREAK", "")


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", str(msg).encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# The fingerprint of one env after a tick (the gate's download first)
# ---------------------------------------------------------------------------
def _feed(m, x) -> None:
    if isinstance(x, np.ndarray):
        m.update(f"{x.dtype}{x.shape}".encode())
        m.update(np.ascontiguousarray(x).tobytes())
    elif isinstance(x, (list, tuple)):
        m.update(f"[{len(x)}".encode())
        for y in x:
            _feed(m, y)
        m.update(b"]")
    elif hasattr(x, "header") and hasattr(x, "dist"):     # horde_cache.CacheRecords
        for a in (x.header, x.keys, x.dist, x.masks):
            _feed(m, a)
    else:
        m.update(repr(x).encode())


def _h(x) -> str:
    """A content hash (every byte of every array, never a summarised repr)."""
    m = hashlib.sha1()
    _feed(m, x)
    return m.hexdigest()[:16]


def fingerprint(c, e) -> tuple:
    """Env ``e``'s (field_digest, tick_digest), its no-digest state by key,
    and its swarm death EVENTS (a death row's env column is its batch index,
    the event carries none)."""
    from simulation.swarm import death_events
    sim = c.sims[e]
    d = cc.digests(sim)
    nd = c.nondigest_records(e)
    rows = c.swarm_death_rows[e]
    ev = [tuple(vars(x).values()) for x in death_events(rows)] if rows is not None else []
    return (tuple(d), tuple((k, _h(v)) for k, v in sorted(nd.items())), _h(ev))


def _what_differs(a, b) -> str:
    if a[0] != b[0]:
        return "the digests (field / tick)"
    for (k, x), (_k, y) in zip(a[1], b[1]):
        if x != y:
            return f"the no-digest state {k!r}"
    return "the swarm death events"


# ---------------------------------------------------------------------------
# Running a conductor over envs
# ---------------------------------------------------------------------------
class Env:
    """One env of a gate: ``make()`` builds its twin sim (a fresh one per
    call); ``intents(t, sim_tick)`` its intents at the run's t-th tick."""

    def __init__(self, name, make, intents):
        self.name, self.make, self.intents = name, make, intents


def play(envs, ticks, label, after_build=None, walk_wrap=None):
    """One conductor over ``envs`` for ``ticks`` ticks; the gate's download
    and every env's fingerprint after each tick. Returns (per-env lists of
    fingerprints, refusal or None)."""
    sims = [e.make() for e in envs]
    c = cd.DeviceConductor(sims, bp=bp)
    if after_build is not None:
        after_build(c)
    if walk_wrap is not None:
        walk_wrap(c)
    out = [[] for _ in envs]
    t0 = time.perf_counter()
    for t in range(ticks):
        for k, env in enumerate(envs):
            c.latch(env.intents(t, int(sims[k].tick)), env=k)
        try:
            c.tick()
        except Exception as ex:                      # a REFUSED tick: never compared
            print(f"  {label}: tick {t} REFUSED ({type(ex).__name__}: {ex})")
            return out, ex
        c.download()
        for k in range(len(envs)):
            out[k].append(fingerprint(c, k))
    dt = time.perf_counter() - t0
    print(f"  {label}: {ticks} ticks, N = {len(envs)} ({1000 * dt / max(1, ticks):.1f} ms per "
          f"tick incl. the gate's download)")
    return out, None


_SOLO: dict = {}


def solo(env, ticks):
    key = (env.name, ticks)
    if key not in _SOLO:
        fps, ex = play([env], ticks, f"{env.name} alone")
        if ex is not None:
            fail(f"{env.name} alone: refused at tick {len(fps[0])}: {ex}")
        _SOLO[key] = fps[0]
    return _SOLO[key]


def batch_gate(name, envs, ticks, order=None, after_build=None) -> bool:
    """Every env alone, then all of them in ONE conductor (in ``order``, a
    permutation of the envs; default as given): each env equal to itself
    alone at every tick. Returns True when equal."""
    alone = [solo(e, ticks) for e in envs]
    order = list(range(len(envs))) if order is None else list(order)
    fps, ex = play([envs[i] for i in order], ticks, f"{name} batch {order}",
                   after_build=after_build)
    if ex is not None:
        fail(f"{name}: the batch was refused at tick {len(fps[0])}: {ex}")
        return False
    for pos, i in enumerate(order):
        a, b = alone[i], fps[pos]
        for t in range(min(len(a), len(b))):
            if a[t] != b[t]:
                fail(f"{name}: env {envs[i].name} (batch index {pos} of {len(envs)}) differs "
                     f"from itself alone at tick {t}: {_what_differs(a[t], b[t])}")
                return False
        if len(a) != len(b):
            fail(f"{name}: env {envs[i].name}: {len(b)} ticks in the batch, {len(a)} alone")
            return False
    print(f"  {name}: every env == itself alone, every one of {ticks} ticks "
          f"(N = {len(envs)}, order {order})")
    return True


# ---------------------------------------------------------------------------
# PART SCENES
# ---------------------------------------------------------------------------
_HOST_INTENTS: dict = {}


def _shifted(make, shift):
    def m():
        sim = make()
        b = getattr(sim, "_conductor_before", None)
        if b is not None and shift:
            sim._conductor_before = (lambda k, _b=b: _b(k - shift) if k >= shift else None)
        return sim
    return m


def host_intents(scene, shift, ticks) -> list:
    """The scene's intents per tick, from a HOST reference (the CPU path)
    playing its schedule delayed by ``shift`` ticks (the reference's tap)."""
    key = (scene, shift, ticks)
    if key not in _HOST_INTENTS:
        make, _ = cc.SCENES[scene]
        ref = _shifted(make, shift)()
        tap = cc.Tap()
        ref.attach_intent_log(tap)
        out = []
        for k in range(ticks):
            if hasattr(ref, "_conductor_before"):
                ref._conductor_before(k)
            ref.step()
            out.append(tap.take())
        ref.attach_intent_log(None)
        _HOST_INTENTS[key] = out
    return _HOST_INTENTS[key]


def scene_env(scene, shift, ticks) -> Env:
    make, _ = cc.SCENES[scene]
    ints = host_intents(scene, shift, ticks)
    return Env(f"{scene}+{shift}", make,
               lambda t, _tick, _i=ints: _i[t] if t < len(_i) else ())


#: Per group, the envs (scene, delay): N = 2 takes the first two, N = 4 all.
SCENE_GROUPS = {
    "payloads": [("payloads", 0), ("payloads", 7), ("payloads", 3), ("payloads", 12)],
    "hazards": [("hazards0", 0), ("hazards2", 0), ("hazards2", 0), ("hazards0", 0)],
    "sprays": [("sprays", 0), ("sprays", 9), ("sprays", 4), ("sprays", 15)],
    "horde": [("horde", 0), ("horde_ctrl", 0), ("horde_ctrl", 6), ("horde", 0)],
    "signals": [("signals", 0), ("signals", 30), ("signals", 11), ("signals", 50)],
    "fire": [("fire0", 0), ("fire1", 0), ("fire1", 0), ("fire0", 0)],
    "walking": [("walking", 0), ("walking", 20), ("walking", 7), ("walking", 41)],
}


def part_scenes(groups=None):
    for g in (groups or list(SCENE_GROUPS)):
        spec = SCENE_GROUPS[g]
        envs = [scene_env(s, d, SCENE_TICKS) for s, d in spec]
        for n in (2, 4):
            batch_gate(f"{g} N={n}", envs[:n], SCENE_TICKS)
        _SOLO.clear()


# ---------------------------------------------------------------------------
# PART MATCHES
# ---------------------------------------------------------------------------
#: (spec, seed) per env: two specs of the one level, different seeds.
MATCH_ENVS = [(None, 11), ("bots_v1", 12), (None, 13), ("bots_v1", 14)]


def match_env(which, seed) -> tuple:
    from simulation.match_session import MatchSession
    level, rec = cc.record_match(which, seed=seed)

    def make():
        return MatchSession(level, rec.spec, seed=rec.seed, replay=rec, breach_physics=bp).sim
    return Env(f"{which or 'bots'}@{seed}", make,
               lambda t, tick, _r=rec: _r.intents.get(tick, ()) if tick < _r.end_tick else ()
               ), int(rec.end_tick)


def part_matches(n_list=(2, 4)):
    envs, ends = [], []
    for which, seed in MATCH_ENVS[:max(n_list)]:
        e, end = match_env(which, seed)
        envs.append(e)
        ends.append(end)
    print(f"  match ends per env: {ends}")
    if len(set(ends)) < 2:
        fail("matches: every env ends at the same tick (the gate needs envs that end apart)")
    for n in n_list:
        batch_gate(f"matches N={n}", envs[:n], max(ends[:n]))
    _SOLO.clear()


# ---------------------------------------------------------------------------
# PART SWARM
# ---------------------------------------------------------------------------
def _larva_sim(seed):
    from level_loader import load as load_level
    from simulation import Simulation
    from simulation.ruleset import ContinuousRealtime
    sim = Simulation(load_level("larva_studio"), seed=seed, breach_physics=bp,
                     enable_recorder=False, ruleset=ContinuousRealtime())
    sim.physics_runner.headless = True
    sim.set_paused(False)
    return sim


def part_swarm(n_list=(2, 4)):
    from simulation import physics_runner as pr
    pr.set_residency(False)
    seeds = [1, 2, 3, 4]
    # the broods differ by seed (the scatter is keyed on the seed)
    starts = []
    for s in seeds[:2]:
        g = _larva_sim(s).gmap
        starts.append(g.swarm_larva_pos_x[:int(g.swarm_larva_high_water)].tobytes())
    if starts[0] == starts[1]:
        fail("swarm: the broods of two seeds start alike (the gate would be vacuous)")
    # env 0 alone against the CPU reference (the swarm's launch words from the
    # device tick row)
    n, refused = cc.run_gate("larva_studio alone vs CPU", _larva_sim(1), _larva_sim(1),
                             SWARM_TICKS)
    cc_fails = list(cc.FAILS)
    cc.FAILS.clear()
    for m in cc_fails:
        fail(m)
    if refused:
        fail(f"swarm: env alone refused vs the CPU reference: {refused}")
    envs = [Env(f"larva@{s}", (lambda _s=s: _larva_sim(_s)), lambda t, tick: ())
            for s in seeds]
    for k in n_list:
        batch_gate(f"swarm N={k}", envs[:k], SWARM_TICKS)
    # the swarm was alive in the batch (not vacuous): the last tick's units
    sims = [e.make() for e in envs[:2]]
    c = cd.DeviceConductor(sims, bp=bp)
    deaths = [0, 0]
    for t in range(SWARM_TICKS):
        c.tick()
        for k in range(2):
            r = c.swarm_death_rows[k]
            deaths[k] += 0 if r is None else int(len(r))
    c.download()
    alive = [int((s.gmap.swarm_larva_valid != 0).sum()) for s in sims]
    print(f"  swarm: live larvae after {SWARM_TICKS} ticks {alive}, deaths {deaths}")
    if min(alive) == 0 and sum(deaths) == 0:
        fail("swarm: no larva alive and none died (vacuous)")
    _SOLO.clear()


# ---------------------------------------------------------------------------
# PART SWAP: an env's state never depends on its index or its neighbours
# ---------------------------------------------------------------------------
def _index_leak(c):
    """THE BREAK: the env index leaks into its draw key (row seed lo)."""
    import cupy as cp
    from simulation.unit_draws import TICK_ROW_KEY
    c.trow.buf[:, TICK_ROW_KEY] += cp.arange(c.N, dtype=cp.int64)


def part_swap():
    envs = [scene_env(s, d, SCENE_TICKS) for s, d in SCENE_GROUPS["payloads"][:3]] + \
        [scene_env(s, d, SCENE_TICKS) for s, d in SCENE_GROUPS["walking"][:1]]
    if len({e.make().gmap.solid.shape for e in envs}) != 1:
        envs = envs[:3]
    ab = after = _index_leak if BREAK == "index" else None
    batch_gate("swap order", envs, SCENE_TICKS, order=list(reversed(range(len(envs)))),
               after_build=ab)
    batch_gate("swap rotated", envs, SCENE_TICKS, order=[1, 2, 0] + list(range(3, len(envs))),
               after_build=after)
    _SOLO.clear()


# ---------------------------------------------------------------------------
# PART SYNCS: the sync count of a steady tick does not depend on N
# ---------------------------------------------------------------------------
def _per_env_download(c):
    """THE BREAK: a face's download per env inside the walk."""
    walk = c._walk

    def broken(L):
        for _e in range(c.N):
            c.noise.check("break")
            c.noise.download()
        walk(L)
    c._walk = broken


def sync_counts(envs, ticks) -> list:
    sims = [e.make() for e in envs]
    c = cd.DeviceConductor(sims, bp=bp)
    if BREAK == "sync":
        _per_env_download(c)
    sc = cc.SyncCount()
    per = []
    try:
        for t in range(ticks):
            for k, env in enumerate(envs):
                c.latch(env.intents(t, int(sims[k].tick)), env=k)
            had_orders = any(c._orders)
            sc.py = []
            bp.host_sync_log_begin()
            try:
                c.tick()
            finally:
                log = list(bp.host_sync_log_end()) + sc.py
            per.append((t, log, had_orders))
    finally:
        sc.close()
    return per


def part_syncs():
    for scene in ("payloads", "walking"):
        env = scene_env(scene, 0, 60)
        counts = {}
        for n in (1, 2, 4):
            logs = sync_counts([env] * n, 60)
            steady = [(t, lg) for t, lg, o in logs[1:] if not o]
            names = sorted({x.partition(" | ")[2] for _t, lg in steady for x in lg})
            counts[n] = [len(lg) for _t, lg in steady]
            print(f"  {scene} N={n}: {np.mean(counts[n]):.2f} host syncs per steady tick "
                  f"({len(steady)} ticks), named {names}")
            bad = [x for x in names if x not in ("max_n_sub", "books_read")]
            if bad:
                fail(f"syncs {scene} N={n}: a sync outside the named reads: {bad}")
        if not (counts[1] == counts[2] == counts[4]):
            fail(f"syncs {scene}: the per-tick sync count depends on N "
                 f"(N=1 {counts[1][:6]}..., N=2 {counts[2][:6]}..., N=4 {counts[4][:6]}...)")
        else:
            print(f"  {scene}: the per-tick sync count is the same at N = 1, 2, 4 "
                  f"({sorted(set(counts[1]))} per tick)")


# ---------------------------------------------------------------------------
# PART FAULT: a fault in env k names env k
# ---------------------------------------------------------------------------
def part_fault(k=2, n=4):
    from simulation import unit_kernels
    base = scene_env("walking", 0, 60)
    make0 = base.make
    holder = {}

    def make_beam():
        sim = make0()
        u = sim.units[0]
        u.weapon_id = "lance_3"
        holder["uid"] = int(u.id)
        return sim

    def beam_intents(t, tick):
        return list(base.intents(t, tick)) + [("set_trigger", (holder["uid"], True))]

    envs = [base] * n
    envs[k] = Env("walking+beam", make_beam, beam_intents)
    if BREAK == "layout":
        cd.counts_by_env = lambda v, N: np.asarray(v, dtype=np.int64).reshape(int(N), 3).T
    fps, ex = play(envs, 60, f"fault in env {k} of {n}")
    if ex is None:
        fail(f"fault: the beam trigger in env {k} was never refused (vacuous)")
        return
    got = getattr(ex, "env", None)
    if not isinstance(ex, unit_kernels.ShotRefused) or got != k or f"env {k}" not in str(ex):
        fail(f"fault: the refusal does not name env {k}: {type(ex).__name__} env={got}: {ex}")
    else:
        print(f"  fault: {type(ex).__name__} names env {got} at tick {len(fps[0])}; the batch "
              f"is refused")


PARTS = {"scenes": part_scenes, "matches": part_matches, "swarm": part_swarm,
         "swap": part_swap, "syncs": part_syncs, "fault": part_fault}


def main(parts=None) -> int:
    from simulation import physics_runner as pr
    pr.set_residency(False)               # the host references run the CPU path
    t0 = time.perf_counter()
    for name in (parts or list(PARTS)):
        if isinstance(name, str) and name.startswith("group:"):
            print(f"PART scenes {name[6:]}")
            part_scenes([name[6:]])
            continue
        print(f"PART {name}")
        PARTS[name]()
    print(f"{time.perf_counter() - t0:.1f} s")
    print("CONDUCTOR_BATCH_RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
