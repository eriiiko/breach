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
folder: a door, ``all_round``; no cover: the crates are ``range_hall``'s,
P8.2's :func:`corpus_cover` plays it as it is and with a crate moved onto
the line of fire); (d) the hand match (P8.1: a ship_demo
bots match cut at :data:`HAND_TICKS` with a hand state at tick 0 and hand
intents, :func:`hand_setup` / :func:`hand_script`, for the owned columns (a)
and (b) never move); and, for the batch, more seeds of (a) and (b).

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

PART GATE (P8.0) -- corpus (a), (b) and (d) at N = 1.
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

PART FLOATKIT (P8.1) -- the observation's float kit on the card
(``cpp/src/cuda_obs_float.cuh``): double -> float16 exhaustively equal to
numpy, double -> float32 equal to numpy, a*b + c never contracted, the
squash's served float32 equal to the host's (CUDA's log; the float64
difference count is printed).

P8.2 (no served column: the sight every later block reads) --
- inside every GATE / BATCH run: who sees whom (``DeviceObservation.vision``:
  ``seen_by``, ``visible_by_team``) equal to the reference's VisionState
  (``vision.state_for``, the team picture's) at EVERY tick, both sight rules,
  the per-rule census (pair-ticks, pairs ever seen / not seen / both) judged
  by part GATE; the line of fire (``lof_pairs``, ``line_of_fire.solid_in_way``
  over the cover table) on every (unit, unit) pair at every decision tick,
  the pairs a cover decided counted (> 0 over the corpus, else FAIL); a
  cover's alive unchanged over every match (the table is staged once);
- PART SIGHT: the bulk forms against ``rays_clear`` / ``window_clear``;
- PART WALL: the wall's first device gate (critique finding 22).

THE BREAKS (``OBS_BREAK``), each must turn a part red: ``cone`` / ``early``
/ ``cover`` (P8.2, part GATE: the cone a degree wider, who sees whom before
the conversion, every cover box one tile off), ``forms`` (P8.2, part SIGHT:
the two has_los forms swapped); ``anchor`` (the
blocks read the units one tile off: the window's anchor wrong), ``stale``
(the blocks are not rebuilt after tick 0), ``constant`` (an owned column
forced constant on both sides: the constancy guard), ``divisor`` (P8.1: the
obs table's weapon block shifted one row, so ``rounds`` / ``range`` / ...
divide by the neighbouring row), ``half_f32`` (P8.1: double -> float16 via
float32, double rounding), ``fused`` (P8.1: the kit's a*b + c as an fma).

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
PATCH = "P8.2"
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

    def __init__(self, name, level, spec, seed, T, S, setup=None):
        self.name, self.level, self.spec, self.seed = name, level, spec, seed
        self.T, self.S = T, S
        self.setup = setup                    # corpus (d): the hand state at tick 0
        self.intents = []                     # per tick k: [(kind, args)]
        self.obs = {}                         # tick -> {(key, name): array}
        self.end = 0
        # P8.2: the reference's VisionState every tick, its line of fire over
        # every (unit, unit) pair at every decision tick, the sight rule
        self.vision = {}                      # tick -> (seen_by, visible_by_team)
        self.lof = {}                         # tick -> (U, U) uint8 (sim.units order)
        self.lof_cover = {}                   # tick -> pairs a cover decided (walk clear)
        self.rule = "cone"
        self.uids = []
        self.cover_alive = None               # (at tick 0, at the end)


_PLAYED: dict = {}
_ROOMS_DIR = None


def ship_level():
    import _pinned_matches as pm
    return pm.ship_demo()


def room_level(name: str = ROOM):
    global _ROOMS_DIR
    import gen_training_rooms as TR
    import level_loader
    if _ROOMS_DIR is None:
        _ROOMS_DIR = tempfile.mkdtemp(prefix="obs_rooms_")
    if not (Path(_ROOMS_DIR) / name / "level.toml").exists():
        for d in TR.drafts():
            if d.name == name:
                TR.write(d, _ROOMS_DIR)
    return level_loader.load(name, _ROOMS_DIR)


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


def host_lof(sim):
    """THE reference line of fire (``line_of_fire.solid_in_way_many``) from
    every unit's centre tile to every unit's (``sim.units`` order, dead and
    turned included): ``(U, U)`` uint8, and the count of pairs a COVER decided
    (the physical walk clear, a live cover crossing the segment)."""
    from simulation import line_of_fire, vision
    cs = [(int(u.center_tile_x()), int(u.center_tile_y())) for u in sim.units]
    n = len(cs)
    out = np.zeros((n, n), dtype=np.uint8)
    pairs = np.array([(ay, ax, by, bx) for ax, ay in cs for bx, by in cs],
                     dtype=np.int32).reshape(-1, 4)
    walk = vision.rays_clear(sim, pairs, physical=True).reshape(n, n) if n else None
    for a in range(n):
        out[a] = np.array(line_of_fire.solid_in_way_many(sim, cs[a], cs), dtype=np.uint8)
    cover = int(np.count_nonzero(out.astype(bool) & walk)) if n else 0
    return out, cover


def play(name, level, spec, seed, setup=None, script=None, max_ticks=None) -> Played:
    """Play the match on the host once (cached): its intents per tick, the
    owned host observation at every decision tick. Corpus (d): ``setup``
    (session) puts the hand state in at tick 0 (the twin gets the same);
    ``script`` (sim) issues hand intents before each tick (the tap records
    them with the controllers', so the twin is fed them); ``max_ticks`` ends
    the play early."""
    key = (name, seed)
    if key in _PLAYED:
        return _PLAYED[key]
    import rl_env
    from simulation.match_session import MatchSession
    from simulation.obs_spec_v1 import ObsContext
    t0 = time.perf_counter()
    s = MatchSession(level, spec, seed=seed, breach_physics=bp)
    if setup is not None:
        setup(s)
    sp = s.spec
    T, S = len(sp.teams), max(rl_env._team_slots(level))
    p = Played(name, level, spec, seed, T, S, setup=setup)
    ctx = ObsContext.build(s.sim, sp, S, n_teams=T)
    own = owned()
    tap = cc.Tap()
    s.sim.attach_intent_log(tap)
    t_obs = 0.0

    def observe():
        nonlocal t_obs
        ts = time.perf_counter()
        p.obs[int(s.sim.tick)] = owned_parts(host_observation(s, ctx, T, S), own, False)
        p.lof[int(s.sim.tick)], p.lof_cover[int(s.sim.tick)] = host_lof(s.sim)
        t_obs += time.perf_counter() - ts

    def see():
        from simulation import vision
        vs = vision.state_for(s.sim)          # the team picture's (slot 11) cached state
        p.vision[int(s.sim.tick)] = (dict(vs.seen_by), dict(vs.visible_by_team))

    p.rule = str(getattr(s.sim, "sight", "cone"))
    p.uids = [int(u.id) for u in s.sim.units]
    alive0 = [bool(c.alive) for c in (s.sim.cover or ())]
    see()
    observe()
    tap.take()                                # the setup's own calls are no intents
    while not s.is_terminal() and (max_ticks is None or int(s.sim.tick) < max_ticks):
        if script is not None:
            script(s.sim)
        s.tick()
        p.intents.append(tap.take())
        see()
        if int(s.sim.tick) % DECIMATION == 0:
            observe()
    s.sim.attach_intent_log(None)
    p.cover_alive = (alive0, [bool(c.alive) for c in (s.sim.cover or ())])
    p.end = int(s.sim.tick)
    _PLAYED[key] = p
    print(f"  played {name}@{seed}: {p.end} ticks, {len(p.obs)} decisions "
          f"({time.perf_counter() - t0:.1f} s, host observation {1000 * t_obs / len(p.obs):.1f} "
          f"ms per decision)")
    return p


def twin(p: Played):
    from simulation.match_session import MatchSession
    s = MatchSession(p.level, p.spec, seed=p.seed, breach_physics=bp)
    if p.setup is not None:
        p.setup(s)
    return s


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


def _divisor_break(obs):
    """THE BREAK ``divisor`` (P8.1): the obs table's weapon block shifted one
    row, so a column's divisor (``rounds`` / ``mag_size``, ``range``, ...)
    is read from the NEIGHBOURING weapon row; must turn the gate red."""
    import cupy as cp
    off = {n: o for n, o, _ in dobs.table_layout()}
    ww = len(dobs.WEAPON_FIELDS)
    nw = int(bp_layout("OBS_TAB_N_WEAPONS"))
    t = obs.table_host.copy()
    blk = t[:, off["WEAPON"]:off["WEAPON"] + nw * ww].reshape(len(t), nw, ww)
    blk[:] = np.roll(blk, -1, axis=1)
    obs.table = cp.asarray(t)


def bp_layout(name: str) -> int:
    """A generated constant of the observation's layout (device_obs's own list)."""
    return dict(dobs._constants())[name]


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
    if BREAK == "divisor":                         # rounds over the NEXT weapon row's magazine
        _divisor_break(obs)
    _sight_breaks(obs)
    n_cmp = 0
    end = max(p.end for p in played)
    t0 = time.perf_counter()
    for k in range(end + 1):
        tick = int(c.sims[0].tick)
        if not vision_equal(label, obs, played, tick):
            return False
        if tick % DECIMATION == 0 and not lof_equal(label, obs, played, tick):
            return False
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
                             f"{'' if kk[1] is None else ' column/plane ' + kk[1]}: {len(idx)} "
                             f"value(s) differ, first at (row, ...) {at}: reference "
                             f"{w[tuple(idx[0])] if len(idx) else w.shape} device "
                             f"{g[tuple(idx[0])] if len(idx) else g.shape}"
                             + (f" (bits 0x{int(raw(w)[tuple(idx[0])]):x} vs "
                                f"0x{int(raw(g)[tuple(idx[0])]):x})" if len(idx) else ""))
                        if np.issubdtype(w.dtype, np.floating):
                            nan_w, nan_g = int(np.isnan(w).sum()), int(np.isnan(g).sum())
                            if nan_w or nan_g:
                                print(f"    NaNs: reference {nan_w}, device {nan_g}")
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


# ---------------------------------------------------------------------------
# P8.2: who sees whom, every tick; the line of fire over the cover table
# ---------------------------------------------------------------------------
#: Per sight rule: pair-ticks compared, the (match, observer, target) pairs
#: ever seen / ever not seen (o != t); per gate label the line-of-fire pairs
#: and those a cover decided. Filled by every gate; judged by part GATE.
VSTATS: dict = {}
LOFSTATS: dict = {}


def _sight_breaks(obs):
    """THE P8.2 BREAKS: ``cone`` (the cone's half-angle one degree wider in
    the obs table), ``early`` (who sees whom computed one slot too early:
    before the ruleset's tick end, the conversion), ``cover`` (every cover
    box shifted one tile in x)."""
    import cupy as cp
    if BREAK == "cone":
        from config import CFG
        from simulation import vision
        off = {n: o for n, o, _ in dobs.table_layout()}["CONE_COS_BITS"]
        t = obs.table.get()
        t[:, off] = dobs._f64_bits(vision.cos_threshold(
            float(CFG.onephase.vision_cone_half_deg) + 1.0))
        obs.table = cp.asarray(t)
        obs.vision()                               # tick 0 again, broken
    if BREAK == "early":
        obs.early_vision = True
    if BREAK == "cover":
        cov = obs.cover_host.copy()
        for f in ("X0_BITS", "X1_BITS"):
            i = dobs.COVER_FIELDS.index(f)
            vals = np.ascontiguousarray(cov[..., i]).view(np.float64) + 1.0
            cov[..., i] = vals.view(np.int64)
        obs.cover = cp.asarray(cov)


def _teams_of_slot(obs):
    from simulation.factions import ZOMBIE_TEAM
    return list(range(obs.T)) + [int(ZOMBIE_TEAM)]


def vision_equal(label, obs, played, tick) -> bool:
    """Every env's device who-sees-whom equal to its reference's VisionState
    at ``tick`` (``seen_by`` and ``visible_by_team``, every pair), no fault;
    the per-rule census filled."""
    sees = obs.sees.get()
    vis = obs.visible.get()
    faults = obs.vision_faults.get()
    slots = _teams_of_slot(obs)
    for e, p in enumerate(played):
        if tick > p.end:
            continue
        if int(faults[e]):
            fail(f"{label}: tick {tick}, env {e}: vision fault bits {int(faults[e])}")
            return False
        uids = obs.c.uids[e]
        if uids != p.uids:
            fail(f"{label}: env {e}: the store's rows are not the reference's units")
            return False
        want_seen, want_vis = p.vision[tick]
        bad_teams = [t for t in want_vis if t not in slots]
        if bad_teams:
            fail(f"{label}: tick {tick}, env {e}: reference teams {bad_teams} have no slot")
            return False
        got_seen, got_vis = dobs.vision_state(sees[e], vis[e], uids, slots)
        n = len(uids)
        ref = np.zeros((n, n), dtype=bool)
        row = {u: r for r, u in enumerate(uids)}
        for tid, obsv in want_seen.items():
            for oid in obsv:
                ref[row[oid], row[tid]] = True
        dev = sees[e][:n, :n].astype(bool)
        if got_seen != want_seen or got_vis != want_vis or not np.array_equal(ref, dev):
            idx = np.argwhere(ref != dev)
            at = (f"pair (observer {uids[idx[0][0]]}, target {uids[idx[0][1]]}): reference "
                  f"{bool(ref[tuple(idx[0])])} device {bool(dev[tuple(idx[0])])}"
                  if len(idx) else "the visible sets")
            fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}, rule {p.rule}): who sees "
                 f"whom differs on {len(idx)} pair(s), first {at}; visible reference "
                 f"{want_vis} device {got_vis}")
            return False
        st = VSTATS.setdefault(p.rule, {"pair_ticks": 0, "true": set(), "false": set(),
                                        "seen_ticks": 0})
        st["pair_ticks"] += n * (n - 1)
        st["seen_ticks"] += int(np.count_nonzero(ref))
        key = (p.name, p.seed, e)
        off = ~np.eye(n, dtype=bool)
        st["true"].update((key, int(a), int(b)) for a, b in np.argwhere(ref & off))
        st["false"].update((key, int(a), int(b)) for a, b in np.argwhere(~ref & off))
    return True


def lof_equal(label, obs, played, tick) -> bool:
    """``line_of_fire.solid_in_way`` on the card over every (unit, unit)
    pair equal to the reference at a decision tick; the pairs a cover decided
    counted."""
    out, faults = obs.lof_pairs()
    out, faults = out.get(), faults.get()
    for e, p in enumerate(played):
        if tick > p.end:
            continue
        if int(faults[e]):
            fail(f"{label}: tick {tick}, env {e}: line-of-fire fault bits {int(faults[e])}")
            return False
        want = p.lof[tick]
        n = want.shape[0]
        got = out[e][:n, :n]
        if not np.array_equal(want, got):
            idx = np.argwhere(want != got)
            a, b = (int(x) for x in idx[0])
            fail(f"{label}: tick {tick}, env {e} ({p.name}@{p.seed}): solid_in_way differs on "
                 f"{len(idx)} pair(s), first (unit {p.uids[a]} -> unit {p.uids[b]}): reference "
                 f"{int(want[a, b])} device {int(got[a, b])}")
            return False
        st = LOFSTATS.setdefault(label, {"pairs": 0, "blocked": 0, "cover": 0, "covers": 0})
        st["pairs"] += n * n
        st["blocked"] += int(np.count_nonzero(want))
        st["cover"] += int(p.lof_cover[tick])
        st["covers"] = int(obs.cover_count_host[e])
    return True


def judge_vision() -> None:
    """The census of who sees whom per rule and of the line of fire: a rule
    whose pairs were never both seen and unseen, or a corpus where no cover
    ever decided a pair, FAILS (a vacuous gate)."""
    for rule, st in sorted(VSTATS.items()):
        both = len(st["true"] & st["false"])
        print(f"    vision rule {rule}: {st['pair_ticks']} pair-ticks compared, "
              f"{st['seen_ticks']} seen; {len(st['true'])} pairs ever seen, "
              f"{len(st['false'])} ever not seen, {both} both")
        if not st["true"] or not st["false"] or both == 0:
            fail(f"vision rule {rule}: no pair was both seen and unseen over the corpus "
                 f"(the gate would be vacuous)")
    for rule in ("cone", "all_round"):
        if rule not in VSTATS:
            fail(f"vision: no corpus match runs the {rule!r} rule")
    for label, st in sorted(LOFSTATS.items()):
        print(f"    line of fire {label}: {st['pairs']} pairs compared, {st['blocked']} "
              f"blocked, {st['cover']} by a cover ({st['covers']} cover entities)")
    if not any(st["cover"] > 0 for st in LOFSTATS.values()):
        fail("line of fire: no cover decided any pair over the corpus (vacuous)")


def corpus_a():
    lv = ship_level()
    import _pinned_matches as pm
    return [play(w, lv, pm.spec_of(lv, w), None) for w in pm.WHICH]


def corpus_b():
    lv = room_level()
    return [play(ROOM, lv, None, None)]


# -- corpus (d): the smallest hand match for the owned columns (a) and (b)
# never move (P8.1: a second weapon, no loadout, no melee row, every status
# kind, the hand-grenade kinds the shipped loadout lacks, a plant, a blow's
# cooldown). A ship_demo bots match, cut short, with a hand state at tick 0
# (both the reference and the twin get it) and hand intents (tapped, so the
# twin is fed them).
HAND_SEED = 31
HAND_TICKS = 320
#: The hand loadout: the [marine] loadout row with every hand-grenade kind of
#: the weapon table's family (these counts, in the family's order).
HAND_GRENADES = (2, 1, 2, 1, 1, 3)


def hand_kit():
    """Corpus (d)'s loadout: the reference loadout row replaced IN THE LIVE
    TABLES by one carrying every hand-grenade kind (a frozen row: a
    ``dataclasses.replace``). ``Simulation`` construction rebuilds the
    loadout tables from the config (``loadouts.rebuild_tables``), so the
    row is put in after each build (the reference's and the twin's) and is
    gone at the next one: the host builders and the obs table read it while
    the session plays."""
    from dataclasses import replace
    from config import CFG
    from simulation.loadouts import get_tables
    from simulation.weapons import get_tables as wt
    by_name = get_tables().loadouts.by_name
    name = str(CFG.marine.loadout)
    rows = wt().hand_grenade_rows
    by_name[name] = replace(by_name[name],
                            grenades={r: HAND_GRENADES[k] for k, r in enumerate(rows)})


def _marines(sim):
    return sorted((u for u in sim.units if not u.is_zombie), key=lambda u: int(u.id))


def hand_setup(session):
    """Corpus (d)'s state at tick 0: the hand loadout (:func:`hand_kit`) and
    its grenades in every marine's hands; marine 0 without a loadout or a
    melee row (class -1, no charge kind, no grenade kinds, no melee); marine
    1 with a second weapon (its magazines belong to the loadout's weapon:
    spares untracked) and a blow just struck (the melee cooldown); marine 4
    with a third; every status kind 1..8 on a marine of its own, durations
    apart (DoTs gentle)."""
    from simulation import status as stt
    hand_kit()
    m = _marines(session.sim)
    for u in m:
        u.grenade_counts = list(HAND_GRENADES)
    m[0].loadout_name = None
    m[0].melee_id = ""
    m[1].weapon_id = "mp11_pdw"
    m[1].last_blow_tick = 0
    m[4].weapon_id = "lr50"
    for k in range(1, stt.N_STATUS_KINDS):
        stt.apply_status(m[k % len(m)], k, 0.05, 40 + 30 * k)


def hand_script(sim):
    """Corpus (d)'s hand intents: a plant, throws of every kind but the frag,
    a reload, a shove."""
    from simulation import unit_fixed
    m = _marines(sim)
    t = int(sim.tick)
    one = unit_fixed.FP_ONE
    alive = lambda u: u.alive and not u.is_zombie  # noqa: E731
    if t == 2 and alive(m[6]):
        sim.plant_charge_intent(int(m[6].id), 10 ** 6)
    for tk, who, kind in ((6, 7, 2), (10, 5, 3), (14, 4, 5), (30, 7, 1), (44, 5, 4)):
        if t == tk and alive(m[who]):
            sim.throw_grenade_intent(int(m[who].id), one, 0, 60, kind=kind)
    if t == 20 and alive(m[5]):
        sim.reload_intent(int(m[5].id))
    if t == 24 and alive(m[7]):
        sim.shove_intent(int(m[7].id))


#: P8.2's cover hand match: the training room whose crates are cover
#: (``range_hall``: two crates, ``blocks_los`` false, placed OFF the line of
#: fire), with crate 0 moved at tick 0 onto the line between the two spawns
#: (the reference and the twin alike), so the learner bot's line of fire to
#: the target crosses it (the room's own crates never cross a pair: counted).
COVER_ROOM = "range_hall"


def hall_setup(session):
    """The cover hand state at tick 0: crate 0 of ``sim.cover`` (a 3 x 3
    tile box) centred on the segment between the two units' centre tiles."""
    sim = session.sim
    a, b = sorted(sim.units, key=lambda u: int(u.id))[:2]
    ax, ay = a.center_tile_x(), a.center_tile_y()
    bx, by = b.center_tile_x(), b.center_tile_y()
    mx, my = (ax + bx) // 2, (ay + by) // 2
    c = sim.cover[0]
    c.x0, c.x1 = float(mx - 1), float(mx + 2)
    c.y0, c.y1 = float(my - 1), float(my + 2)


def corpus_d():
    lv = ship_level()
    import _pinned_matches as pm
    return [play("hand", lv, pm.spec_of(lv, "bots"), HAND_SEED, setup=hand_setup,
                 script=hand_script, max_ticks=HAND_TICKS)]


def corpus_cover():
    """P8.2: the training room with cover as it is, and the cover hand match."""
    lv = room_level(COVER_ROOM)
    return [play(COVER_ROOM, lv, None, None),
            play("hall_cover", lv, None, 51, setup=hall_setup)]


#: P8.2: the all_round rule over many units (the training rooms are 1 v 1, so
#: no team there ever has a seen AND an unseen enemy): a ship_demo bots match
#: whose spec names ``sight = "all_round"``, cut at this many ticks.
ROUND_TICKS = 480


def corpus_round():
    from dataclasses import replace

    import _pinned_matches as pm
    lv = ship_level()
    spec = replace(lv.match, sight="all_round")       # the "bots" match is the level's own
    return [play("ship_all_round", lv, spec, 61, max_ticks=ROUND_TICKS)]


def part_gate():
    ca, cb, cdn = Census(), Census(), Census()
    for p in corpus_a():
        gate(f"(a) {p.name}", [p], ca)
    for p in corpus_b():
        gate(f"(b) {p.name}", [p], cb)
    for p in corpus_d():
        gate(f"(d) {p.name}", [p], cdn)
    for p in corpus_cover():
        gate(f"(b/d cover) {p.name}", [p], cb)
    for p in corpus_round():
        gate(f"(d all_round) {p.name}", [p], cdn)
    ca.judge("census (a)", strict=False)
    cb.judge("census (b)", strict=False)
    cdn.judge("census (d)", strict=False)
    both = Census()
    for src in (ca, cb, cdn):
        for k, n in src.nondefault.items():
            both.nondefault[k] = both.nondefault.get(k, 0) + n
            both.values.setdefault(k, set()).update(src.values.get(k, set()))
    both.judge("census (a)+(b)+(d)")
    judge_vision()
    for p in corpus_a() + corpus_b() + corpus_d() + corpus_cover() + corpus_round():
        a0, a1 = p.cover_alive
        if a0 != a1:
            fail(f"cover: a cover's alive changed on the continuous path in {p.name} "
                 f"({a0} -> {a1}): the staged cover table would be stale")
        else:
            print(f"    cover {p.name}@{p.seed}: {len(a0)} cover entities, alive unchanged "
                  f"over {p.end} ticks")


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


def half_corpus() -> np.ndarray:
    """EVERY float16 rounding case as float64: every float16 value (all 2^16
    bit patterns, inf and NaN included), every midpoint of two neighbouring
    finite halves and its two float64 neighbours, the overflow boundary
    (65520 and its neighbours), values past it, the underflow boundary
    (2^-25 and its neighbours, 2^-26, float64 subnormals), signed zeros,
    NaN payloads; both signs."""
    h = np.arange(0x7c00, dtype=np.uint16).view(np.float16).astype(np.float64)
    mids = (h[:-1] + h[1:]) / 2.0                         # exact in float64
    edge = np.array([65504.0, 65520.0, 65536.0, 1e5, 1e300, np.inf, 2.0 ** -24,
                     2.0 ** -25, 2.0 ** -26, 3 * 2.0 ** -26, 5e-324, 2.2e-308, 0.0])
    pos = np.concatenate([h, mids, edge])
    pos = np.concatenate([pos, np.nextafter(pos, np.inf), np.nextafter(pos, -np.inf)])
    pos = pos[pos >= 0.0]
    allh = np.arange(1 << 16, dtype=np.uint32).astype(np.uint16).view(np.float16).astype(
        np.float64)
    nans = np.array([0x7ff8000000000000, 0x7ff0000000000001, 0x7ff4000000000000,
                     0x7ff00400_00000000, 0xfff8000000000000, 0x7fffffffffffffff],
                    dtype=np.uint64).view(np.float64)
    return np.concatenate([pos, -pos, allh, nans])


def part_floatkit():
    """The observation's float kit on the card (``cpp/src/cuda_obs_float.cuh``):
    the hand double -> float16 conversion EXHAUSTIVELY equal to numpy's direct
    conversion; the double -> float32 cast equal to numpy's; a*b + c never
    contracted (numpy rounds twice); the squash's served float32 equal to the
    host's ``obs_spec_v1.squash`` on a dense sweep (CUDA's log against the
    host's libm: the count is reported). Breaks: ``half_f32`` (the conversion
    through float32, double rounding), ``fused`` (an explicit fma, what
    contraction would build)."""
    from simulation import obs_spec_v1 as o
    rng = np.random.default_rng(81)
    # -- float16 --
    xs = half_corpus()
    with np.errstate(over="ignore", invalid="ignore"):
        want = xs.astype(np.float16).view(np.uint16)
    got = np.asarray(bp.obs_probe_half(xs, via_f32=(BREAK == "half_f32")))
    bad = np.flatnonzero(got != want)
    print(f"  float16: {len(xs)} values (every half, every midpoint +- 1 ulp, the edges, "
          f"NaNs): {len(bad)} differ from numpy")
    if len(bad):
        i = bad[0]
        fail(f"floatkit: double -> float16 differs on {len(bad)} of {len(xs)}: first "
             f"{xs[i]!r} (0x{int(np.float64(xs[i]).view(np.uint64)):016x}) numpy "
             f"0x{int(want[i]):04x} device 0x{int(got[i]):04x}")
    # -- float32 --
    rnd = np.concatenate([rng.uniform(-70000, 70000, 200000),
                          np.exp(rng.uniform(-750, 700, 200000)) * rng.choice([-1, 1], 200000)])
    xf = np.concatenate([xs, rnd, np.nextafter(rnd, np.inf)])
    with np.errstate(over="ignore", invalid="ignore"):
        wf = xf.astype(np.float32).view(np.uint32)
    gf = np.asarray(bp.obs_probe_f32(xf)).view(np.uint32)
    badf = np.flatnonzero(gf != wf)
    print(f"  float32: {len(xf)} values: {len(badf)} differ from numpy")
    if len(badf):
        i = badf[0]
        fail(f"floatkit: double -> float32 differs on {len(badf)}: first {xf[i]!r} numpy "
             f"0x{int(wf[i]):08x} device 0x{int(gf[i]):08x}")
    # -- no contraction: c = -(a*b rounded), so a*b + c is 0 unrounded-twice and
    # the product's rounding error when fused --
    a = rng.uniform(0.5, 2.0, 100000)
    b = rng.uniform(0.5, 2.0, 100000)
    c = -(a * b)
    wm = (a * b) + c                                       # numpy: two roundings
    gm = np.asarray(bp.obs_probe_muladd(a, b, c, fused=(BREAK == "fused")))
    fused = np.asarray(bp.obs_probe_muladd(a, b, c, fused=True))
    badm = int(np.count_nonzero(gm.view(np.uint64) != wm.view(np.uint64)))
    disc = int(np.count_nonzero(fused.view(np.uint64) != wm.view(np.uint64)))
    print(f"  a*b + c: {len(a)} triples, {badm} differ from numpy's two roundings; the "
          f"fused form differs on {disc} (the inputs discriminate)")
    if badm:
        fail(f"floatkit: a*b + c is contracted on {badm} of {len(a)} triples")
    if disc < len(a) // 2:
        fail(f"floatkit: the contraction probe's inputs do not discriminate ({disc})")
    # -- the squash: CUDA's log against the host's (the served float32) --
    sx = np.concatenate([np.exp(rng.uniform(0.0, 3.2, 300000)),
                         -np.exp(rng.uniform(0.0, 3.2, 100000)),
                         rng.uniform(-1.0, 1.0, 10000),
                         np.array([1.0, -1.0, np.e ** 3, np.nextafter(np.e ** 3, 0), 1e9]),
                         np.arange(1, 4097, dtype=np.float64) / 16.0])
    g64, g32 = bp.obs_probe_squash(sx)
    w64 = np.array([o.squash(x) for x in sx], dtype=np.float64)
    w32 = w64.astype(np.float32)
    d64 = int(np.count_nonzero(np.asarray(g64).view(np.uint64) != w64.view(np.uint64)))
    d32 = int(np.count_nonzero(np.asarray(g32).view(np.uint32) != w32.view(np.uint32)))
    print(f"  squash: {len(sx)} values: float64 differs from the host's on {d64}, the "
          f"served float32 on {d32}")
    if d32:
        fail(f"floatkit: the squash's served float32 differs from the host's on {d32} of "
             f"{len(sx)} values (CUDA's log vs the host's: the shared-module trigger)")


# ---------------------------------------------------------------------------
# P8.2: PART SIGHT -- the bulk forms against rays_clear / window_clear
# ---------------------------------------------------------------------------
SIGHT_PAIRS = 30000
SIGHT_WINDOWS = 120
WIN = 25


def _ship_states():
    """Three ship_demo states of one shape (one launch): the level as it
    starts (every door shut), half the doors OPENED (``want_open`` and the
    door sweep, on a scratch session), and grille tiles (open to gas, closed
    to sight: the sight screen set) stamped on 3% of the open tiles of a
    scratch copy. Returns (sims, label)."""
    import _pinned_matches as pm
    from simulation import door_system
    from simulation.entities.door import DOOR_OPEN
    from simulation.match_session import MatchSession
    lv = ship_level()
    spec = pm.spec_of(lv, "bots")
    a, b, g = (MatchSession(lv, spec, seed=41, breach_physics=bp) for _ in range(3))
    _toggle_doors(b.sim, every=2)
    opened = sum(int(d.state == DOOR_OPEN) for d in b.sim._doors)
    rng = np.random.default_rng(82)
    open_tiles = np.argwhere(~g.sim.gmap.solid)
    pick = open_tiles[rng.choice(len(open_tiles), max(1, len(open_tiles) // 33), replace=False)]
    g.sim.gmap.sight_screen[pick[:, 0], pick[:, 1]] = True
    print(f"  ship_demo states: shut doors / {opened} of {len(b.sim._doors)} doors opened / "
          f"{len(pick)} grille tiles")
    return [a.sim, b.sim, g.sim], "ship_demo"


def _toggle_doors(sim, every=1) -> int:
    """Flip every ``every``-th door (an open one shut, a shut one open)
    through its latch and the door sweep (a scratch session); returns how
    many flipped."""
    from simulation import door_system
    from simulation.entities.door import DOOR_OPEN
    doors = list(sim._doors)[::every]
    before = [d.state for d in doors]
    for d in doors:
        d.want_open = d.state != DOOR_OPEN
    door_system.sweep_doors(sim)
    return sum(int(d.state != s) for d, s in zip(doors, before))


def _room_states():
    """The training room with a door (as it starts: open; and shut) and the
    one with cover crates (``blocks_los`` false: rays_clear ignores them)."""
    from simulation.match_session import MatchSession
    lv = room_level()
    a, b = (MatchSession(lv, None, seed=42, breach_physics=bp) for _ in range(2))
    flipped = _toggle_doors(b.sim)
    print(f"  {ROOM} states: as it starts / {flipped} of {len(b.sim._doors)} door(s) flipped")
    return [a.sim, b.sim], ROOM


def _hall_states():
    from simulation.match_session import MatchSession
    lv = room_level(COVER_ROOM)
    a = MatchSession(lv, None, seed=43, breach_physics=bp)
    print(f"  {COVER_ROOM} state: {len(a.sim.cover)} cover entities")
    return [a.sim], COVER_ROOM


def _hot_points(sims):
    """Tiles where a line's answer turns: door span tiles and grille tiles."""
    pts = []
    for s in sims:
        for d in s._doors:
            pts += [tuple(t) for t in d.span]
        pts += [tuple(t) for t in np.argwhere(s.gmap.sight_screen)]
    return np.array(sorted(set(pts)), dtype=np.int64).reshape(-1, 2)


def _pairs(rng, h, w, hot, n):
    """``n`` (y0, x0, y1, x1) pairs: uniform over the map and a 3-tile ring
    outside it (out of bounds), lines across the hot tiles, and pairs on one
    tile."""
    k = n // 3
    uni = np.stack([rng.integers(-3, h + 3, k), rng.integers(-3, w + 3, k),
                    rng.integers(-3, h + 3, k), rng.integers(-3, w + 3, k)], axis=1)
    if len(hot):
        c = hot[rng.integers(0, len(hot), k)]
        near = np.concatenate([c + rng.integers(-10, 11, (k, 2)),
                               c + rng.integers(-10, 11, (k, 2))], axis=1)
    else:
        near = uni.copy()
    m = n - 2 * k
    one = rng.integers(-2, max(h, w) + 2, (m, 2))
    same = np.concatenate([one, one], axis=1)
    return np.concatenate([uni, near, same]).astype(np.int32)


def _sight_group(sims, label, rng) -> None:
    import cupy as cp
    from simulation import vision
    h, w = sims[0].gmap.solid.shape
    solid = cp.asarray(np.stack([np.ascontiguousarray(s.gmap.solid) for s in sims]))
    screen = cp.asarray(np.stack([np.ascontiguousarray(s.gmap.sight_screen) for s in sims]))
    hot = _hot_points(sims)
    pairs = _pairs(rng, h, w, hot, SIGHT_PAIRS)
    host = {}
    for physical in (False, True):
        dev_phys = (not physical) if BREAK == "forms" else physical
        stack = cp.asarray(np.stack([pairs] * len(sims)))
        got, faults = dobs.rays_clear_device(bp, solid, screen, stack, dev_phys)
        got, faults = got.get(), faults.get()
        for e, s in enumerate(sims):
            want = vision.rays_clear(s, pairs, physical=physical).astype(np.uint8)
            host[(e, physical)] = want
            if int(faults[e]) or not np.array_equal(want, got[e]):
                bad = np.flatnonzero(want != got[e])
                i = int(bad[0]) if len(bad) else -1
                fail(f"sight {label}: env {e} rays_clear(physical={physical}) differs on "
                     f"{len(bad)} of {len(pairs)} pairs (faults {int(faults[e])})"
                     + (f", first {tuple(int(x) for x in pairs[i])}: reference {want[i]} "
                        f"device {got[e][i]}" if i >= 0 else ""))
                return
    oob = int(np.count_nonzero((pairs[:, [0, 2]] < 0).any(1) | (pairs[:, [0, 2]] >= h).any(1)
                               | (pairs[:, [1, 3]] < 0).any(1) | (pairs[:, [1, 3]] >= w).any(1)))
    same = int(np.count_nonzero((pairs[:, 0] == pairs[:, 2]) & (pairs[:, 1] == pairs[:, 3])))
    forms = sum(int(np.count_nonzero(host[(e, False)] != host[(e, True)]))
                for e in range(len(sims)))
    doors = (int(np.count_nonzero(host[(0, True)] != host[(1, True)]))
             if len(sims) > 1 else None)
    print(f"  sight {label}: {len(sims)} envs x {len(pairs)} pairs equal in both forms "
          f"({oob} with an end outside the map, {same} on one tile; {doors} pairs the "
          f"opened doors turn, {forms} env-pairs the two forms answer apart)")
    if label == "ship_demo" and forms == 0:
        fail("sight ship_demo: no pair where the two forms differ (the grille case is vacuous)")
    if doors == 0:
        fail(f"sight {label}: no pair turned by flipping the doors (vacuous)")
    # -- windows --
    n = SIGHT_WINDOWS
    cy, cx = rng.integers(-2, h + 2, n), rng.integers(-2, w + 2, n)
    q = np.stack([cy, cx, cy - WIN // 2 + rng.integers(-15, 16, n),
                  cx - WIN // 2 + rng.integers(-15, 16, n)], axis=1).astype(np.int32)
    if len(hot):                                    # half of them centred near a hot tile
        hh = hot[rng.integers(0, len(hot), n // 2)] + rng.integers(-6, 7, (n // 2, 2))
        q[: n // 2, 0], q[: n // 2, 1] = hh[:, 0], hh[:, 1]
        q[: n // 2, 2], q[: n // 2, 3] = hh[:, 0] - WIN // 2, hh[:, 1] - WIN // 2
    cells = 0
    for to_window in (False, True):
        for physical in (False, True):
            dev_phys = (not physical) if BREAK == "forms" else physical
            got, faults = dobs.window_clear_device(bp, solid, screen,
                                                   cp.asarray(np.stack([q] * len(sims))),
                                                   WIN, WIN, to_window, dev_phys)
            got, faults = got.get(), faults.get()
            for e, s in enumerate(sims):
                for j, (y, x, wy, wx) in enumerate(q.tolist()):
                    want = vision.window_clear(s, y, x, wy, wx, WIN, WIN, to_window=to_window,
                                               physical=physical)
                    cells += want.size
                    if int(faults[e]) or not np.array_equal(want.astype(np.uint8), got[e, j]):
                        bad = np.argwhere(want.astype(np.uint8) != got[e, j])
                        fail(f"sight {label}: env {e} window_clear(({y}, {x}), ({wy}, {wx}), "
                             f"to_window={to_window}, physical={physical}) differs on "
                             f"{len(bad)} cells, first {tuple(int(v) for v in bad[0])}"
                             if len(bad) else f"sight {label}: env {e} window faults")
                        return
    print(f"  sight {label}: {len(sims)} envs x {n} windows x 4 (both directions, both forms) "
          f"equal ({cells} cells)")


def part_sight():
    """vision.rays_clear / window_clear on the card (P8.2) equal to the host's
    on dense random pairs and windows over real maps: ship_demo with its
    doors shut, half opened, and grille tiles; the training room (crates
    that do not block sight) shut and opened. Break ``forms``: the two
    has_los forms swapped."""
    rng = np.random.default_rng(8202)
    for states in (_ship_states, _room_states, _hall_states):
        sims, label = states()
        _sight_group(sims, label, rng)


# ---------------------------------------------------------------------------
# P8.2: PART WALL -- the wall's first device gate (critique finding 22)
# ---------------------------------------------------------------------------
def part_wall():
    """Who a team sees depends on an enemy's row only through that enemy
    standing in view: perturbing the store row (facing, hp, turn rate,
    effort, exposure) of an enemy NO member of team t sees leaves t's
    visible set unchanged; the same perturbation of a SEEN enemy leaves it
    unchanged too, and killing the seen enemy removes it. Corpus (a) bots
    (cone) and the all_round ship match, at decision ticks."""
    import cupy as cp
    from simulation import turning
    from simulation import unit_store
    lay = unit_store.layout()
    col = {n: lay.column(n).offset for n in ("team", "life_state", "facing_q",
                                            "current_hp", "omega_q", "effort", "exposure")}
    col["valid"] = lay.valid
    dead = 1                                         # unit_schema._LIFE_CHOICES.index("dead")
    total = {}
    for p in corpus_a()[:1] + corpus_round():
        s = twin(p)
        c = cd.DeviceConductor(s.sim, bp=bp)
        obs = dobs.DeviceObservation(c, [s.spec], p.S, DECIMATION, n_teams=p.T)
        done = 0
        for k in range(len(p.intents)):
            c.latch(p.intents[k], env=0)
            c.tick()
            if int(c.sims[0].tick) % DECIMATION or done >= 6:
                continue
            rows = c._rows.get()[0]
            vis = obs.visible.get()[0]
            for t in range(p.T):
                enemy = [r for r in range(len(p.uids)) if rows[r, col["valid"]]
                         and rows[r, col["life_state"]] == 0 and rows[r, col["team"]] != t]
                seen = [r for r in enemy if vis[t, r]]
                unseen = [r for r in enemy if not vis[t, r]]
                if not seen or not unseen:
                    continue

                def run(edit):
                    pert = c._rows.copy()
                    edit(pert)
                    obs.vision(store_ptr=int(pert.data.ptr))
                    out = obs.visible.get()[0, t].copy()
                    del pert
                    return out

                def perturb(r):
                    def f(a):
                        f0 = int(a[0, r, col["facing_q"]])
                        f1 = f0 + turning.PI_Q16 // 2       # a quarter turn, in (-PI, PI]
                        a[0, r, col["facing_q"]] = f1 - 2 * turning.PI_Q16 \
                            if f1 > turning.PI_Q16 else f1
                        a[0, r, col["current_hp"]] = max(1, int(a[0, r, col["current_hp"]]) // 3)
                        a[0, r, col["omega_q"]] += 12345
                        a[0, r, col["effort"]] += 1 << 15
                        a[0, r, col["exposure"]] += 7
                    return f

                u, sv = unseen[0], seen[0]
                if not np.array_equal(run(perturb(u)), vis[t]):
                    fail(f"wall {p.name}: tick {int(c.sims[0].tick)}, team {t}: perturbing "
                         f"unseen enemy {p.uids[u]}'s row changed the team's visible set")
                    return
                if not np.array_equal(run(perturb(sv)), vis[t]):
                    fail(f"wall {p.name}: tick {int(c.sims[0].tick)}, team {t}: perturbing seen "
                         f"enemy {p.uids[sv]}'s facing / hp changed whether it is seen")
                    return

                def kill(a):
                    a[0, sv, col["life_state"]] = dead
                killed = run(kill)
                if killed[sv] or np.array_equal(killed, vis[t]):
                    fail(f"wall {p.name}: tick {int(c.sims[0].tick)}, team {t}: killing seen "
                         f"enemy {p.uids[sv]} did not remove it from the visible set")
                    return
                done += 1
            obs.vision()                             # the real store again
        total[p.rule] = total.get(p.rule, 0) + done
        print(f"  wall {p.name} ({p.rule}): {done} (team, tick) cases with a seen and an "
              f"unseen enemy: unseen perturbed -> unchanged, seen perturbed -> unchanged, "
              f"seen killed -> removed")
    for rule in ("cone", "all_round"):
        if not total.get(rule):
            fail(f"wall: no case under the {rule!r} rule (vacuous)")


PARTS = {"gate": part_gate, "batch": part_batch, "handover": part_handover,
         "syncs": part_syncs, "ended": part_ended, "floatkit": part_floatkit,
         "sight": part_sight, "wall": part_wall}


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
