"""#29 P8 -- THE DEVICE OBSERVATION'S SHARED CHECK (``simulation.device_obs``):
the observation built on the card inside the device conductor's tick, held
EQUAL, as raw bits, to the HOST observation (``obs_spec_v1.observe_rows`` and
``rl_env.EnvRunner.row_flags``, THE reference) at every decision tick. Every
P8 patch extends this one script (a part per patch, never a new script).

Runs inside the GPU subprocess (tests/cuda_harness.run_cuda_script; wrapper
tests/test_cuda_obs.py).

THE CORPUS, played ONCE per subprocess (:func:`play`): a REFERENCE match runs
on the host CPU path with its own controllers (the bots), its intents taken
from its own tap per tick (``Simulation.attach_intent_log``), and at every
decision tick (tick 0, then every :data:`DECIMATION` ticks) the host builds
the whole observation the env would serve (an ``ObsContext`` per episode,
the rows' units, the team picture the session keeps). Kept: the tick's
intents and the OWNED arrays of each decision. (a) the pinned ship_demo
matches ``bots`` and ``bots_v1`` (``tests/_pinned_matches``); (b) a training
room (``tools/gen_training_rooms.py``'s ``range_door``, written to a temp
folder: a door, cover, ``all_round``); and, for the batch, more seeds of both.

OWNERSHIP (``tests/obs_column_owners.toml``, the column-to-patch table): a
gate compares exactly the columns / planes owned by :data:`PATCH` and every
earlier patch, and counts per owned column how many cells over the corpus
are not the default (0); an owned column holding one value over its whole
corpus FAILS (a gate on a column that never moved is vacuous).

THE GATE (:func:`gate`): the conductor on fresh twins of the played matches
(N envs, one level), fed each env's tapped intents per tick (an env whose
match ended ticks on with none); a ``DeviceObservation`` attached; at every
decision tick of every env its owned device arrays compared with the
reference's, raw bits (``view`` as unsigned ints: -0.0 and NaN count); the
first difference printed (tick, env, key, plane, index) and FAILS.

PART GATE (P8.0) -- corpus (a) and (b) at N = 1.
PART BATCH (P8.0) -- N = 2 and 4 of ship_demo and of the training room,
different seeds per env (envs end apart): every env equal to ITS reference
at every one of its decision ticks (so equal to itself alone).
PART HANDOVER (P8.0) -- the tensors torch gets (``DeviceObservation.torch``,
``team_rows``) share the CuPy buffers' memory (data pointer identity); the
stream ordering the hand-over rests on, read from the runtime: the tick
stream is BLOCKING, CuPy and torch work on the legacy default stream; a
torch read straight after a tick (no explicit sync) equals the synced value.
PART SYNCS (P8.0) -- a conductor tick with the observation attached makes
only the conductor's named syncs (``max_n_sub``, the books read), the same
count at N = 1 and 2, and the blocks were built during the count.
PART ENDED (P8.0) -- an env whose ended latch is set keeps its buffers (the
blocks skip it) while its neighbour is rebuilt.

THE BREAKS (``OBS_BREAK``), each must turn a part red: ``anchor`` (the
blocks read the units one tile off: the window's anchor wrong), ``stale``
(the blocks are not rebuilt after tick 0), ``constant`` (an owned column
forced constant on both sides: the constancy guard).

PROPERTY: the device observation IS the host observation, value for value,
on every owned column, at every decision tick, in a batch as alone; the
torch tensors are the buffers; the observation adds no host sync. BREAKS IF
a block reads the wrong anchor, a row map entry, another env's slice, a
stale state, or a plane outside the reference's fill rules.
"""
from __future__ import annotations

import os
import sys
import tempfile
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import breach_physics as bp   # the CUDA build (cuda_harness bootstrap): FIRST

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "src", _ROOT / "tools", _ROOT / "tests"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import cuda_conductor_check as cc  # noqa: E402
from simulation import conductor as cd  # noqa: E402
from simulation import device_obs as dobs  # noqa: E402

FAILS: list = []
BREAK = os.environ.get("OBS_BREAK", "")
#: The patch this script's gates are at: every column owned by it or an
#: earlier patch of the version-2 table is compared.
PATCH = "P8.0"
#: The decision cadence of the corpus (design 13.1's decimation, a run file's
#: ``[run] decimation``; the device observation takes it as its argument).
DECIMATION = 4
#: The seeds of the batch's envs (per corpus: different seeds, envs end apart).
BATCH_SHIP = [(None, 11), ("bots_v1", 12), (None, 13), ("bots_v1", 14)]
BATCH_ROOM = [21, 22, 23, 24]
ROOM = "range_door"
OWNERS = _ROOT / "tests" / "obs_column_owners.toml"


def fail(msg):
    FAILS.append(msg)
    print("FAIL:", str(msg).encode("ascii", "replace").decode("ascii"))


# ---------------------------------------------------------------------------
# Ownership (the column-to-patch table)
# ---------------------------------------------------------------------------
def owned(patch: str = PATCH) -> dict:
    """``key -> tuple of plane / column names`` (None: the whole key) owned
    by ``patch`` and every earlier patch of the table."""
    t = tomllib.loads(OWNERS.read_text(encoding="utf-8"))
    order = t["patches"]
    upto = set(order[:order.index(patch) + 1])
    out = {}
    for key, who in t["keys"].items():
        if who in upto:
            out[key] = None
    for key, cols in t["columns"].items():
        names = tuple(n for n, who in cols.items() if who in upto)
        if names:
            out[key] = names
    return out


def _plane_index(key: str, name: str) -> int:
    from simulation import obs_spec_v1 as o
    planes = {"near_f": o.NEAR_F_PLANES, "near_u": o.NEAR_U_PLANES,
              "far_f": o.FAR_F_PLANES, "far_u": o.FAR_U_PLANES}
    if key in planes:
        return planes[key].index(name)
    cols = {"vector": o.VECTOR_COLUMNS, "list": o.ENTRY_COLUMNS}[key]
    return next(f.pos for f in cols if f.name == name)


def owned_parts(arrs: dict, own: dict, env_axis: bool) -> dict:
    """The owned slices of an observation dict, ``(key, name) -> array``
    (the row axis first; ``env_axis``: drop a leading env axis first is the
    caller's)."""
    out = {}
    for key, names in own.items():
        a = arrs[key]
        if names is None:
            out[(key, None)] = np.array(a, copy=True)
            continue
        for n in names:
            i = _plane_index(key, n)
            out[(key, n)] = np.array(a[:, i] if key not in ("vector", "list")
                                     else a[..., i], copy=True)
    return out


def raw(a) -> np.ndarray:
    a = np.ascontiguousarray(a)
    return a.view({1: np.uint8, 2: np.uint16, 4: np.uint32, 8: np.uint64}[a.dtype.itemsize])


# ---------------------------------------------------------------------------
# THE CORPUS: each match played ONCE on the host (the reference)
# ---------------------------------------------------------------------------
class Played:
    """One reference match: its level, spec, seed, (T, S), per tick its
    intents, per decision tick the owned arrays of the host observation."""

    def __init__(self, name, level, spec, seed, T, S):
        self.name, self.level, self.spec, self.seed = name, level, spec, seed
        self.T, self.S = T, S
        self.intents = []                     # per tick k: [(kind, args)]
        self.obs = {}                         # tick -> {(key, name): array}
        self.end = 0


_PLAYED: dict = {}
_ROOMS_DIR = None


def ship_level():
    import _pinned_matches as pm
    return pm.ship_demo()


def room_level():
    global _ROOMS_DIR
    import gen_training_rooms as TR
    import level_loader
    if _ROOMS_DIR is None:
        _ROOMS_DIR = tempfile.mkdtemp(prefix="obs_rooms_")
        for d in TR.drafts():
            if d.name == ROOM:
                TR.write(d, _ROOMS_DIR)
    return level_loader.load(ROOM, _ROOMS_DIR)


def host_observation(session, ctx, T, S) -> dict:
    """The env's served observation of ``session`` now, built by THE
    reference (``EnvRunner.row_units`` / ``row_flags`` over the session,
    ``obs_spec_v1.observe_rows``), one env's ``(U, ...)`` arrays."""
    import rl_env
    from simulation import obs_spec_v1 as o
    me = SimpleNamespace(session=session, layout=SimpleNamespace(n_slots=S, n_rows=T * S))
    shapes = o.shapes(1, T * S, S)
    out = {k: np.zeros(sh[1:], dtype=dt) for k, (sh, dt) in shapes.items()}
    o.observe_rows(session.sim, ctx, rl_env.EnvRunner.row_units(me), out)
    valid, in_match = rl_env.EnvRunner.row_flags(me)
    out["valid"][...] = valid
    out["in_match"][...] = in_match
    return out


def play(name, level, spec, seed) -> Played:
    """Play the match on the host once (cached): its intents per tick, the
    owned host observation at every decision tick."""
    key = (name, seed)
    if key in _PLAYED:
        return _PLAYED[key]
    import rl_env
    from simulation.match_session import MatchSession
    from simulation.obs_spec_v1 import ObsContext
    t0 = time.perf_counter()
    s = MatchSession(level, spec, seed=seed, breach_physics=bp)
    sp = s.spec
    T, S = len(sp.teams), max(rl_env._team_slots(level))
    p = Played(name, level, spec, seed, T, S)
    ctx = ObsContext.build(s.sim, sp, S, n_teams=T)
    own = owned()
    tap = cc.Tap()
    s.sim.attach_intent_log(tap)
    t_obs = 0.0

    def observe():
        nonlocal t_obs
        ts = time.perf_counter()
        p.obs[int(s.sim.tick)] = owned_parts(host_observation(s, ctx, T, S), own, False)
        t_obs += time.perf_counter() - ts

    observe()
    while not s.is_terminal():
        s.tick()
        p.intents.append(tap.take())
        if int(s.sim.tick) % DECIMATION == 0:
            observe()
    s.sim.attach_intent_log(None)
    p.end = int(s.sim.tick)
    _PLAYED[key] = p
    print(f"  played {name}@{seed}: {p.end} ticks, {len(p.obs)} decisions "
          f"({time.perf_counter() - t0:.1f} s, host observation {1000 * t_obs / len(p.obs):.1f} "
          f"ms per decision)")
    return p


def twin(p: Played):
    from simulation.match_session import MatchSession
    return MatchSession(p.level, p.spec, seed=p.seed, breach_physics=bp)


# ---------------------------------------------------------------------------
# THE GATE
# ---------------------------------------------------------------------------
class Census:
    """Per owned (key, name): cells over the corpus not the default, the
    distinct values seen (capped)."""

    def __init__(self):
        self.nondefault: dict = {}
        self.values: dict = {}

    def add(self, parts: dict, valid) -> None:
        rows = np.asarray(valid, dtype=bool)
        for k, a in parts.items():
            a = np.asarray(a)
            sel = a if k[1] is None else a[rows]
            r = raw(sel).ravel()
            self.nondefault[k] = self.nondefault.get(k, 0) + int(np.count_nonzero(r))
            vs = self.values.setdefault(k, set())
            if len(vs) < 4:
                vs.update(int(x) for x in np.unique(r)[:4])

    def judge(self, label: str, strict: bool = True) -> None:
        for k in sorted(self.nondefault, key=str):
            n, vs = self.nondefault[k], self.values.get(k, set())
            print(f"    {label} {k[0]}{'' if k[1] is None else '/' + k[1]}: {n} non-default "
                  f"cells, {len(vs)}{'+' if len(vs) >= 4 else ''} distinct value(s)")
            if strict and len(vs) <= 1:
                fail(f"{label}: the owned column {k} is constant over its corpus "
                     f"(the gate on it would be vacuous)")


def _anchor_break(obs):
    """THE BREAK ``anchor``: the blocks read every unit one tile off."""
    from simulation import unit_store
    build = obs.build
    xq = unit_store.layout().column("x_q").offset

    def broken(store_ptr=None):
        rows = obs.c._rows.copy()
        rows[:, :, xq] += 1 << 16
        obs._break_rows = rows                     # alive until the launch ran
        build(store_ptr=int(rows.data.ptr))
    obs.build = broken


def gate(label, played: list, census: Census | None = None) -> bool:
    """The conductor over twins of ``played`` (one level), each env fed its
    tapped intents; every env's owned device arrays equal to its reference
    at each of its decision ticks. Returns True when equal."""
    own = owned()
    sessions = [twin(p) for p in played]
    c = cd.DeviceConductor([s.sim for s in sessions], bp=bp)
    T = max(p.T for p in played)
    S = max(p.S for p in played)
    obs = dobs.DeviceObservation(c, [s.spec for s in sessions], S, DECIMATION, n_teams=T)
    if BREAK == "anchor":                          # from the first decision after tick 0
        _anchor_break(obs)
    if BREAK == "stale":                           # nothing rebuilt after tick 0
        obs.skip_build = True
    n_cmp = 0
    end = max(p.end for p in played)
    t0 = time.perf_counter()
    for k in range(end + 1):
        tick = int(c.sims[0].tick)
        if tick % DECIMATION == 0:
            got = obs.host(own)
            for e, p in enumerate(played):
                if tick > p.end:
                    continue
                want = p.obs[tick]
                mine = {kk: (got[kk[0]][e] if kk[1] is None else
                             got[kk[0]][e][:, _plane_index(kk[0], kk[1])])
                        for kk in want}
                if BREAK == "constant":
                    for kk in want:
                        if kk == ("near_u", "wall"):
                            want[kk] = np.zeros_like(want[kk])
                            mine[kk] = np.zeros_like(mine[kk])
                for kk, w in want.items():
                    g = mine[kk]
                    if w.shape != g.shape or not np.array_equal(raw(w), raw(g)):
                        idx = np.argwhere(raw(w) != raw(g)) if w.shape == g.shape else []
                        at = tuple(int(x) for x in idx[0]) if len(idx) else "shape"
                        fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}), {kk[0]}"
                             f"{'' if kk[1] is None else ' plane ' + kk[1]}: {len(idx)} value(s) "
                             f"differ, first at (row, ...) {at}: reference "
                             f"{w[tuple(idx[0])] if len(idx) else w.shape} device "
                             f"{g[tuple(idx[0])] if len(idx) else g.shape}")
                        return False
                if census is not None:
                    census.add(want, want[("valid", None)] if ("valid", None) in want
                               else np.ones(p.T * p.S, bool))
                n_cmp += 1
        if k == end:
            break
        for e, p in enumerate(played):
            c.latch(p.intents[k] if k < len(p.intents) else (), env=e)
        c.tick()
    print(f"  {label}: {n_cmp} env-decisions equal (N = {len(played)}, {end} ticks, "
          f"{obs.builds} block launches, {1000 * (time.perf_counter() - t0) / max(1, end):.1f} "
          f"ms per tick incl. the gate's downloads)")
    return True


def corpus_a():
    lv = ship_level()
    import _pinned_matches as pm
    return [play(w, lv, pm.spec_of(lv, w), None) for w in pm.WHICH]


def corpus_b():
    lv = room_level()
    return [play(ROOM, lv, None, None)]


def part_gate():
    ca, cb = Census(), Census()
    for p in corpus_a():
        gate(f"(a) {p.name}", [p], ca)
    for p in corpus_b():
        gate(f"(b) {p.name}", [p], cb)
    ca.judge("census (a)", strict=False)
    cb.judge("census (b)", strict=False)
    both = Census()
    for src in (ca, cb):
        for k, n in src.nondefault.items():
            both.nondefault[k] = both.nondefault.get(k, 0) + n
            both.values.setdefault(k, set()).update(src.values.get(k, set()))
    both.judge("census (a)+(b)")


def part_batch(n_list=(2, 4)):
    lv = ship_level()
    import _pinned_matches as pm
    ships = [play(w or "bots", lv, pm.spec_of(lv, w or "bots"), s) for w, s in BATCH_SHIP]
    rl = room_level()
    rooms = [play(ROOM, rl, None, s) for s in BATCH_ROOM]
    for group, ps in (("ship_demo", ships), ("room", rooms)):
        ends = [p.end for p in ps]
        print(f"  {group} ends per env: {ends}")
        if len(set(ends[:max(n_list)])) < 2:
            print(f"  NOTE: every {group} env ends at the same tick")
        for n in n_list:
            gate(f"batch {group} N={n}", ps[:n])


def part_handover():
    import cupy as cp
    import torch
    p = corpus_b()[0]
    s = twin(p)
    c = cd.DeviceConductor(s.sim, bp=bp)
    obs = dobs.DeviceObservation(c, [s.spec], p.S, DECIMATION, n_teams=p.T)
    ts = obs.torch()
    bad = [k for k, v in obs.buffers.items() if ts[k].data_ptr() != int(v.data.ptr)
           or str(ts[k].dtype).split(".")[-1] != str(np.dtype(v.dtype)) or tuple(ts[k].shape) != tuple(v.shape)]
    if bad:
        fail(f"handover: torch tensors that are not their CuPy buffer: {bad}")
    else:
        print(f"  handover: {len(ts)} torch tensors share their CuPy buffers' pointers, "
              f"dtypes and shapes")
    for t in range(p.T):
        rows = obs.team_rows(t)
        for k, v in rows.items():
            base = obs.buffers[k]
            want = int(base.data.ptr) + t * p.S * int(base.strides[1])
            if v.data_ptr() != want or v.shape[1] != p.S:
                fail(f"handover: team_rows({t})[{k!r}] is not a view of the buffer")
                break
    so = dobs.stream_ordering(bp)
    print(f"  stream ordering: {so}")
    if so["tick_stream_flags"] != 0:
        fail(f"handover: the tick stream is not blocking (flags {so['tick_stream_flags']})")
    if so["cupy_stream"] != 0 or so["torch_stream"] != 0:
        fail(f"handover: CuPy / torch are not on the legacy default stream: {so}")
    # a torch read straight after the tick that builds a decision, no sync
    for k in range(DECIMATION):
        c.latch(p.intents[k], env=0)
        c.tick()
    tw = ts["near_u"].to(torch.int64).sum().item()
    cp.cuda.runtime.deviceSynchronize()
    ref = int(obs.buffers["near_u"].astype(cp.int64).sum().get())
    if tw != ref or ref == 0:
        fail(f"handover: torch read {tw} after the tick, the synced buffer holds {ref}")
    else:
        print(f"  handover: torch's unsynced read after the tick == the synced buffer ({ref})")


def part_syncs():
    p = corpus_b()[0]
    counts = {}
    for n in (1, 2):
        sessions = [twin(p) for _ in range(n)]
        c = cd.DeviceConductor([x.sim for x in sessions], bp=bp)
        obs = dobs.DeviceObservation(c, [x.spec for x in sessions], p.S, DECIMATION,
                                     n_teams=p.T)
        sc = cc.SyncCount()
        per = []
        b0 = obs.builds
        try:
            for k in range(min(60, len(p.intents))):
                for e in range(n):
                    c.latch(p.intents[k], env=e)
                had = any(c._orders)
                sc.py = []
                bp.host_sync_log_begin()
                try:
                    c.tick()
                finally:
                    log = list(bp.host_sync_log_end()) + sc.py
                per.append((k, log, had))
        finally:
            sc.close()
        steady = [lg for k, lg, o in per[1:] if not o]
        names = sorted({x.partition(" | ")[2] for lg in steady for x in lg})
        counts[n] = [len(lg) for lg in steady]
        print(f"  N={n}: {np.mean(counts[n]):.2f} host syncs per steady tick, named {names}, "
              f"{obs.builds - b0} block launches during the count")
        if obs.builds - b0 == 0:
            fail(f"syncs N={n}: no block was built during the count (vacuous)")
        extra = [x for x in names if x not in ("max_n_sub", "books_read")]
        if extra:
            fail(f"syncs N={n}: a sync outside the conductor's named reads: {extra}")
    if counts[1] != counts[2]:
        fail(f"syncs: the per-tick count depends on N ({counts[1][:6]} vs {counts[2][:6]})")


def part_ended():
    import cupy as cp
    p = corpus_b()[0]
    sessions = [twin(p), twin(p)]
    c = cd.DeviceConductor([x.sim for x in sessions], bp=bp)
    obs = dobs.DeviceObservation(c, [x.spec for x in sessions], p.S, DECIMATION, n_teams=p.T)
    keep = {k: obs.buffers[k][1].copy() for k in dobs.BUILT}
    obs.ended[1] = 1
    moved = False
    n_dec = 0
    for k in range(min(len(p.intents), 12 * DECIMATION)):
        for e in range(2):
            c.latch(p.intents[k], env=e)
        c.tick()
        tick = int(c.sims[0].tick)
        if tick % DECIMATION == 0:
            n_dec += 1
            w = p.obs[tick]
            if not np.array_equal(raw(obs.buffers["near_u"][0, :, 0].get()),
                                  raw(w[("near_u", "wall")])):
                fail(f"ended: env 0 (latch clear) is not its reference at tick {tick}")
                return
            if any(not bool(cp.array_equal(obs.buffers[k_][1], keep[k_])) for k_ in keep):
                moved = True
    if moved:
        fail("ended: the env whose ended latch is set was rebuilt")
    else:
        print(f"  ended: env 1 (latch set) kept its buffers over {n_dec} decisions; env 0 "
              f"equal to its reference")
    ref_moved = any(not np.array_equal(p.obs[0][kk], p.obs[max(p.obs)][kk])
                    for kk in p.obs[0] if kk[0] in ("near_u", "far_f", "far_u"))
    if not ref_moved:
        print("  NOTE: the reference planes did not move over the match")


PARTS = {"gate": part_gate, "batch": part_batch, "handover": part_handover,
         "syncs": part_syncs, "ended": part_ended}


def main(parts=None) -> int:
    from simulation import physics_runner as pr
    pr.set_residency(False)               # the reference runs the CPU path (the oracle)
    t0 = time.perf_counter()
    print(f"OBS check at {PATCH}: owned {owned()}")
    for name in (parts or list(PARTS)):
        print(f"PART {name}")
        PARTS[name]()
    print(f"{time.perf_counter() - t0:.1f} s")
    print("OBS_RESULT:", "PASS" if not FAILS else f"FAIL ({len(FAILS)})")
    return 0 if not FAILS else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or None))
