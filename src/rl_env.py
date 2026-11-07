"""THE batched environment (#29 P1a, residency design §4.1 / §4.5 / §13 b.1).

``BatchedEnv(run, backend="cpu_farm")`` advances N matches per call and
returns observations, rewards and terminals with a leading N axis. A trainer
touches nothing else: never a ``Simulation``, a controller or a kernel. One
API, two backends; P1a builds ``cpu_farm`` only (the ``device`` backend is
P5..P9) and refuses any other name.

THE RUN FILE (``run.toml``, kept OUTSIDE the repo tree, any path; its sha256
is :attr:`BatchedEnv.run_sha256` for the later run manifest). Nothing about a
training run lives in ``config.toml``. Refused BY NAME on an unknown or
missing key or a mistyped value (:func:`load_run`)::

    [run]
    seed = 1                      # the run seed, 0 <= seed < 2**64
    n_envs = 8
    decimation = 1                # ticks per step()
    levels = ["ship_demo"]        # level folder names (each has a match.toml)
    policy = "policy"             # "policy" for every team, or a controller
                                  # name per team: ["policy", "bots"]
    levels_dir = "..."            # OPTIONAL: the levels root (absolute, or
                                  # relative to the run file; default the
                                  # repo's levels/)
    [obs]
    channels = ["pressure", "smoke", "temperature", "solid", "material",
                "units_own", "units_enemy", "units_zombie"]
    crop_radius = 8               # the crop is (2r+1) x (2r+1) tiles
    vector = ["hp", "team", "alive", "facing", "x", "y"]
    [reward]                      # a weight per counter name (absent = 0)
    kills = 1.0
    deaths_bullet = -1.0

EPISODES. Episode ``k`` of env ``e`` plays level ``levels[i]`` with match
seed ``s``, ``(s, i) = episode_draw(run_seed, e, k, len(levels))``: ONE
Philox-4x32-10 block keyed by the run seed, counter ``(e, k, EPISODE_SALT,
0)`` -- a pure function of its arguments, so an env's episode sequence never
depends on N or on when other envs end. The match's own draws stay
``sim.rng`` seeded by ``s`` (P3b moves them to Philox). The match is the
level's match.toml (``LevelData.match``) with the seed replaced and every
team's controller set by ``[run] policy``; the horde keeps its
``zombie_controller``.

THE BACKEND ``cpu_farm``: ``n_workers`` processes (``multiprocessing``,
spawn: the GIL), each owning a contiguous block of envs, each env one
:class:`EnvRunner` over a ``simulation.match_session.MatchSession`` -- the
match exactly as ``tools/run_match.py`` plays it. Actions, observations,
rewards and terminals live in ``multiprocessing.shared_memory`` blocks;
commands and the small per-env info go over a pipe. A worker imports nothing
from ``renderer/`` and only the CPU build of ``breach_physics``
(``cpp/build/Release``; it refuses any other), so it never creates a CUDA
context.

THE TENSORS (U = T x S rows, row ``t * S + s`` = team ``t``'s roster slot
``s``; T = the most marine teams of any level, S = the largest
``max_per_team`` of any marine team's spawn room; a row is ``valid`` while
its unit is alive and still on its team -- fixed roster slots, team size is a
mask):

- ``actions`` ``(N, U, A)`` int32: the ``policy`` controller's action rows
  (``control_policy.action_schema()``, generated from
  ``match_replay.INTENT_KINDS``); rows of teams not on ``policy`` are
  ignored. Decimation: the latch kinds (throw, use) go out on the step's
  first tick, move / aim / the trigger hold on every tick of it.
- ``obs["crop"]`` ``(N, U, C, 2r+1, 2r+1)`` float32: the sensor accessor's
  ``crop`` (the frozen channels + ``material`` + unit occupancy by team),
  ``obs["vector"]`` ``(N, U, V)`` float32 (:data:`VECTOR_FIELDS`),
  ``obs["valid"]`` ``(N, U)`` bool. Placeholder rows (agent era).
- ``reward`` ``(N, U)`` float32: the step's per-unit counters
  (:data:`COUNTER_NAMES`, from the ticks' credited ``UnitHitEvent`` /
  ``UnitKilledEvent``) weighted by ``[reward]``. PLACEHOLDER.
- ``terminated`` ``(N,)`` bool: the ``MatchRules`` law (fewer than two marine
  teams alive). ``truncated`` ``(N,)`` bool: the time limit, a DRAW today --
  PLACEHOLDER until the agent era.
- ``info``: ``tick``, ``episode``, ``episode_seed``, ``level``, ``ticks``
  (simulated this step), ``ended``, ``final_obs`` (the terminal observation
  of the envs that ended; they were reset inside the step and ``obs`` is
  their episode k+1's tick 0), ``result``, ``counters`` and
  ``checkpoints``: per env the ``(episode, tick, field_digest,
  tick_digest)`` taken at the ``[replay] checkpoint_every_ticks`` cadence,
  at tick 0 and at an episode's last tick -- the recording's checkpoints, so
  the equality gates can compare them with ``run_match``.
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import sys
import time
import tomllib
import traceback
from dataclasses import dataclass, field
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CPU_BUILD = ROOT / "cpp" / "build" / "Release"
DEFAULT_LEVELS_DIR = ROOT / "levels"

BACKENDS = ("cpu_farm",)
#: The Philox stream salt of the episode draw ("EPIS"); P3b's salt enum
#: takes it over.
EPISODE_SALT = 0x45504953

#: The per-unit vector fields a run file may name, each read off the Unit.
VECTOR_FIELDS = {
    "hp": lambda u: float(u.current_hp),
    "team": lambda u: float(u.team),
    "alive": lambda u: 1.0 if u.alive else 0.0,
    "facing": lambda u: float(u.facing),
    "x": lambda u: float(u.x),
    "y": lambda u: float(u.y),
}


def _counter_names() -> tuple:
    from simulation.events import CAUSE_NAMES
    return (("hits_dealt", "damage_dealt", "team_hits", "kills", "team_kills",
             "hits_taken", "damage_taken")
            + tuple(f"deaths_{c}" for c in CAUSE_NAMES))


#: The per-unit reward counters, generated from ``events.Cause`` (a new
#: cause row adds its ``deaths_<cause>``). Per step, for the unit of a row
#: while it was on the row's team at the tick's start: ``hits_dealt`` /
#: ``damage_dealt`` -- damaging hits it is credited with on a unit of
#: another team (marines or the horde); ``team_hits`` -- on another unit of
#: its own team; ``kills`` / ``team_kills`` likewise for kills; its own
#: ``hits_taken`` / ``damage_taken`` (any source, damage > 0: a heal is not a
#: hit) and ``deaths_<cause>``. Self-hits count only as taken.
COUNTER_NAMES = _counter_names()


# ---------------------------------------------------------------------------
# The episode draw
# ---------------------------------------------------------------------------
def episode_draw(run_seed: int, env: int, k: int, n_levels: int) -> tuple:
    """``(match_seed, level_index)`` of episode ``k`` of env ``e``: one
    Philox-4x32-10 block, key ``(run_seed lo, run_seed hi)``, counter
    ``(env, k, EPISODE_SALT, 0)``; the seed is the block's first two words
    (64 bits, the width of P3b's Philox key), the level ``below(w2,
    n_levels)`` (Lemire multiply-shift, bias < n / 2**32). Pure."""
    from simulation.philox32 import philox32_4x32_10, philox32_below
    run_seed = int(run_seed)
    if not 0 <= run_seed < 2 ** 64:
        raise ValueError(f"run seed must be in [0, 2**64), got {run_seed}")
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    w = philox32_4x32_10((env, k, EPISODE_SALT, 0),
                         (run_seed & 0xFFFFFFFF, run_seed >> 32))
    return w[0] | (w[1] << 32), philox32_below(w[2], n_levels)


# ---------------------------------------------------------------------------
# The run file door
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RunConfig:
    """A parsed, validated run file (:func:`load_run`)."""
    seed: int
    n_envs: int
    decimation: int
    levels: tuple
    policy: object              # str (every team) or tuple (per team)
    levels_dir: str
    channels: tuple
    crop_radius: int
    vector: tuple
    reward: tuple               # ((counter, weight), ...) in COUNTER_NAMES order
    sha256: str = ""
    path: str = ""

    def reward_weights(self) -> np.ndarray:
        w = dict(self.reward)
        return np.array([float(w.get(n, 0.0)) for n in COUNTER_NAMES],
                        dtype=np.float32)

    def team_controllers(self, n_teams: int, where: str = "") -> tuple:
        if isinstance(self.policy, str):
            return (self.policy,) * n_teams
        if len(self.policy) != n_teams:
            raise ValueError(f"[run] policy names {len(self.policy)} "
                             f"controller(s), level {where} has {n_teams} "
                             f"teams")
        return tuple(self.policy)


_RUN_KEYS = {"seed": True, "n_envs": True, "decimation": True,
             "levels": True, "policy": True, "levels_dir": False}
_OBS_KEYS = {"channels": True, "crop_radius": True, "vector": True}


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _table(raw, name, keys, where):
    t = raw.get(name)
    if not isinstance(t, dict):
        raise ValueError(f"{where}: missing table [{name}]")
    bad = sorted(set(t) - set(keys))
    if bad:
        raise ValueError(f"{where}: unknown key(s) {bad} in [{name}] (known: "
                         f"{sorted(keys)})")
    for k, required in keys.items():
        if required and k not in t:
            raise ValueError(f"{where}: missing key [{name}] {k}")
    return t


def _int_key(t, table, key, where, lo, hi=None):
    v = t[key]
    if not _is_int(v) or v < lo or (hi is not None and v >= hi):
        rng = f">= {lo}" + (f" and < {hi}" if hi is not None else "")
        raise ValueError(f"{where}: [{table}] {key} must be an integer {rng}, "
                         f"got {v!r}")
    return int(v)


def _names_key(t, table, key, where, *, allow_empty=False):
    v = t[key]
    if not isinstance(v, list) or not all(isinstance(x, str) and x
                                          for x in v):
        raise ValueError(f"{where}: [{table}] {key} must be a list of "
                         f"names, got {v!r}")
    if not v and not allow_empty:
        raise ValueError(f"{where}: [{table}] {key} is empty")
    return tuple(v)


def parse_run(raw: dict, where: str = "run file",
              base_dir: Path | None = None) -> RunConfig:
    """The run-file door: a raw TOML dict -> a :class:`RunConfig`, refusing
    by name an unknown table or key, a missing one, a mistyped value, an
    unknown controller, crop channel (``light`` included, until #81),
    vector field or counter."""
    from control_source import match_file_controllers
    from simulation.sensor_accessor import resolve_crop_channels
    bad = sorted(set(raw) - {"run", "obs", "reward"})
    if bad:
        raise ValueError(f"{where}: unknown table(s) {bad} (a run file "
                         f"carries [run], [obs], [reward])")
    run = _table(raw, "run", _RUN_KEYS, where)
    obs = _table(raw, "obs", _OBS_KEYS, where)
    if not isinstance(raw.get("reward"), dict):
        raise ValueError(f"{where}: missing table [reward]")
    seed = _int_key(run, "run", "seed", where, 0, 2 ** 64)
    n_envs = _int_key(run, "run", "n_envs", where, 1)
    decimation = _int_key(run, "run", "decimation", where, 1)
    levels = _names_key(run, "run", "levels", where)
    known_ctrl = set(match_file_controllers()) | {"policy"}
    pol = run["policy"]
    if isinstance(pol, str):
        names = (pol,)
    elif isinstance(pol, list) and pol and all(isinstance(x, str)
                                               for x in pol):
        names = tuple(pol)
        pol = tuple(pol)
    else:
        raise ValueError(f"{where}: [run] policy must be a controller name or "
                         f"a list of names per team, got {pol!r}")
    for n in names:
        if n not in known_ctrl:
            raise ValueError(f"{where}: [run] policy names {n!r}, not a "
                             f"controller (known: {sorted(known_ctrl)})")
    if "levels_dir" in run:
        ld = run["levels_dir"]
        if not isinstance(ld, str) or not ld:
            raise ValueError(f"{where}: [run] levels_dir must be a path "
                             f"string, got {ld!r}")
        p = Path(ld)
        if not p.is_absolute():
            p = (base_dir or Path.cwd()) / p
        levels_dir = str(p.resolve())
    else:
        levels_dir = str(DEFAULT_LEVELS_DIR)
    channels = _names_key(obs, "obs", "channels", where)
    try:
        channels = resolve_crop_channels(channels)
    except ValueError as e:
        raise ValueError(f"{where}: [obs] channels: {e}") from None
    crop_radius = _int_key(obs, "obs", "crop_radius", where, 0)
    vector = _names_key(obs, "obs", "vector", where, allow_empty=True)
    for v in vector:
        if v not in VECTOR_FIELDS:
            raise ValueError(f"{where}: [obs] vector names {v!r} (known: "
                             f"{sorted(VECTOR_FIELDS)})")
    if len(set(vector)) != len(vector):
        raise ValueError(f"{where}: [obs] vector names a field twice")
    weights = {}
    for k, v in raw["reward"].items():
        if k not in COUNTER_NAMES:
            raise ValueError(f"{where}: [reward] {k!r} is not a counter "
                             f"(known: {list(COUNTER_NAMES)})")
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{where}: [reward] {k} must be a number, got "
                             f"{v!r}")
        weights[k] = float(v)
    reward = tuple((n, weights[n]) for n in COUNTER_NAMES if n in weights)
    return RunConfig(seed=seed, n_envs=n_envs, decimation=decimation,
                     levels=levels, policy=pol, levels_dir=levels_dir,
                     channels=channels, crop_radius=crop_radius,
                     vector=vector, reward=reward)


def load_run(path) -> RunConfig:
    """Read and door-check the run file at ``path``; the result carries the
    file's sha256 (of its exact bytes) and path."""
    path = Path(path)
    data = path.read_bytes()
    try:
        raw = tomllib.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise ValueError(f"{path}: not a TOML run file ({e})") from None
    cfg = parse_run(raw, str(path), base_dir=path.parent)
    return _with(cfg, sha256=hashlib.sha256(data).hexdigest(),
                 path=str(path.resolve()))


def _with(cfg: RunConfig, **kw) -> RunConfig:
    from dataclasses import replace
    return replace(cfg, **kw)


# ---------------------------------------------------------------------------
# The layout (shapes, rows) -- computed from the levels, the same everywhere
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Layout:
    """The tensor shapes of a run over its levels."""
    n_envs: int
    n_teams: int                # T
    n_slots: int                # S
    n_actions: int              # A
    channels: tuple
    crop_radius: int
    vector: tuple
    counters: tuple = field(default=COUNTER_NAMES)

    @property
    def n_rows(self) -> int:
        return self.n_teams * self.n_slots

    @property
    def crop_side(self) -> int:
        return 2 * self.crop_radius + 1

    def shapes(self) -> dict:
        n, u = self.n_envs, self.n_rows
        k = self.crop_side
        c, v, kk = len(self.channels), len(self.vector), len(self.counters)
        return {
            "actions": ((n, u, self.n_actions), np.int32),
            "crop": ((n, u, c, k, k), np.float32),
            "vector": ((n, u, v), np.float32),
            "valid": ((n, u), np.uint8),
            "final_crop": ((n, u, c, k, k), np.float32),
            "final_vector": ((n, u, v), np.float32),
            "final_valid": ((n, u), np.uint8),
            "reward": ((n, u), np.float32),
            "counters": ((n, u, kk), np.float32),
            "terminated": ((n,), np.uint8),
            "truncated": ((n,), np.uint8),
            "tick": ((n,), np.int64),
            "episode": ((n,), np.int64),
            "episode_seed": ((n,), np.uint64),
            "level": ((n,), np.int64),
            "ticks": ((n,), np.int64),
        }


def _team_slots(level) -> list:
    """The marine teams' slot budgets of a level (its spawn rooms'
    ``max_per_team``), team order."""
    spec = level.match
    rooms = {r.team: r.max_per_team for r in level.spawn_rooms}
    return [rooms[t] for t in range(len(spec.teams))]


def make_layout(run: RunConfig, levels: list) -> Layout:
    from control_policy import action_schema
    n_teams = max(len(lv.match.teams) for lv in levels)
    n_slots = max(max(_team_slots(lv)) for lv in levels)
    return Layout(n_envs=run.n_envs, n_teams=n_teams, n_slots=n_slots,
                  n_actions=len(action_schema()), channels=run.channels,
                  crop_radius=run.crop_radius, vector=run.vector)


def load_levels(run: RunConfig) -> list:
    """Load every level of the run (refusing one without a match.toml or
    whose team count disagrees with a per-team ``policy`` list)."""
    import level_loader
    out = []
    for name in run.levels:
        lv = level_loader.load(name, run.levels_dir)
        if lv.match is None:
            raise ValueError(f"run level {name!r} has no "
                             f"{level_loader.MATCH_FILENAME}")
        run.team_controllers(len(lv.match.teams), repr(name))
        out.append(lv)
    return out


# ---------------------------------------------------------------------------
# One env (the worker's unit of work; also usable in-process by a gate)
# ---------------------------------------------------------------------------
class EnvRunner:
    """One env of the farm: its episodes, each a ``MatchSession`` with the
    run's controllers, stepped ``decimation`` ticks per :meth:`step`, its
    observation, counters and terminals. ``record=True`` records every
    episode through the sim's intent tap (``MatchSession.record``);
    :attr:`last_recording` holds the last finished one."""

    def __init__(self, env: int, run: RunConfig, layout: Layout, levels,
                 *, breach_physics=None, checkpoint_every_ticks=None,
                 record: bool = False):
        from config import CFG
        import match_replay
        from simulation.events import UnitHitEvent, UnitKilledEvent
        self.env = int(env)
        self.run = run
        self.layout = layout
        self.levels = list(levels)
        self.bp = breach_physics
        self.every = (match_replay.replay_settings(CFG)
                      if checkpoint_every_ticks is None
                      else int(checkpoint_every_ticks))
        self.record = bool(record)
        self.last_recording = None
        self._hit, self._kill = UnitHitEvent, UnitKilledEvent
        self._vec = [VECTOR_FIELDS[v] for v in layout.vector]
        self._ci = {n: i for i, n in enumerate(layout.counters)}
        self.k = -1
        self.session = None
        self.seed = None
        self.level_index = None

    # -- episodes -------------------------------------------------------
    def reset(self) -> list:
        """Start the env's next episode (k + 1): draw its level and seed,
        build the match. Returns its checkpoints so far (tick 0)."""
        from dataclasses import replace
        from control_policy import AgentPolicy
        from simulation.match_session import MatchSession
        from simulation.sensor_accessor import EntityFieldAccessor
        self.k += 1
        seed, li = episode_draw(self.run.seed, self.env, self.k,
                                len(self.levels))
        level = self.levels[li]
        base = level.match
        ctrls = self.run.team_controllers(len(base.teams), level.name)
        spec = replace(base, seed=seed, teams=tuple(
            replace(t, controller=c) for t, c in zip(base.teams, ctrls)))
        self.seed, self.level_index = seed, li
        self.session = MatchSession(level, spec, seed=seed,
                                    breach_physics=self.bp)
        if self.record:
            self.session.record(self.every)
        sim = self.session.sim
        self.policies = {c.team: c for c in self.session.controllers
                         if isinstance(c, AgentPolicy)}
        self.accessor = EntityFieldAccessor(sim.gmap)
        S = self.layout.n_slots
        self._id_row = {}
        for t in range(len(spec.teams)):
            for u in sim.roster(t):
                self._id_row[u.id] = (t, t * S + u.slot)
        self._limit = self.session.rules.limit_ticks(sim)
        return [(self.k, 0, *self.session.digests())]

    def step(self, actions) -> dict:
        """One decimation step on ``actions`` ``(U, A)``: each policy team's
        rows, then up to ``decimation`` ticks (fewer if the match ends).
        Returns the step's counters, ticks, checkpoints and -- when the match
        ended -- terminated / truncated and its result (the session is then
        finished; call :meth:`reset` for the next episode)."""
        s = self.session
        sim = s.sim
        S = self.layout.n_slots
        for team, pol in self.policies.items():
            pol.set_rows(actions[team * S:(team + 1) * S])
        counters = np.zeros((self.layout.n_rows, len(self.layout.counters)),
                            dtype=np.float32)
        cps = []
        ticks = 0
        for _ in range(self.run.decimation):
            team_before = {u.id: int(u.team) for u in sim.units}
            s.tick()
            ticks += 1
            self._count(sim.tick_events, team_before, counters)
            if sim.tick % self.every == 0:
                cps.append((self.k, int(sim.tick), *s.digests()))
            if s.is_terminal():
                break
        out = {"counters": counters, "ticks": ticks, "checkpoints": cps,
               "ended": False, "terminated": False, "truncated": False,
               "result": None}
        if s.is_terminal():
            if not cps or cps[-1][1] != int(sim.tick):
                cps.append((self.k, int(sim.tick), *s.digests()))
            res = s.result()
            out["ended"] = True
            out["terminated"] = len(s.rules.living_teams(sim)) < 2
            out["truncated"] = not out["terminated"]
            out["result"] = {"outcome": res.outcome, "team": res.team,
                             "ticks": res.ticks}
            rec = s.finish()
            if rec is not None:
                self.last_recording = rec
        return out

    def _row(self, uid: int, team_before: dict):
        tr = self._id_row.get(uid)
        if tr is None or team_before.get(uid) != tr[0]:
            return None
        return tr[1]

    def _count(self, events, team_before, counters) -> None:
        ci = self._ci
        for ev in events:
            if isinstance(ev, self._hit):
                dmg = float(ev.damage)
                if dmg <= 0.0:
                    continue
                v, a = int(ev.unit_id), int(ev.attacker_id)
                vr = self._row(v, team_before)
                if vr is not None:
                    counters[vr, ci["hits_taken"]] += 1.0
                    counters[vr, ci["damage_taken"]] += dmg
                if a >= 0 and a != v:
                    ar = self._row(a, team_before)
                    if ar is not None:
                        if team_before.get(v) == team_before.get(a):
                            counters[ar, ci["team_hits"]] += 1.0
                        else:
                            counters[ar, ci["hits_dealt"]] += 1.0
                            counters[ar, ci["damage_dealt"]] += dmg
            elif isinstance(ev, self._kill):
                v, a = int(ev.unit_id), int(ev.attacker_id)
                vr = self._row(v, team_before)
                if vr is not None:
                    counters[vr, ci[f"deaths_{ev.killed_by}"]] += 1.0
                if a >= 0 and a != v:
                    ar = self._row(a, team_before)
                    if ar is not None:
                        key = ("team_kills" if team_before.get(v)
                               == team_before.get(a) else "kills")
                        counters[ar, ci[key]] += 1.0

    # -- observation ----------------------------------------------------
    def row_units(self) -> list:
        """The unit of every row (``None`` where the row is not valid)."""
        sim = self.session.sim
        S = self.layout.n_slots
        out = [None] * self.layout.n_rows
        for t in range(len(self.session.spec.teams)):
            for u in sim.roster(t):
                if u.alive and u.team == t:
                    out[t * S + u.slot] = u
        return out

    def observe(self, crop_out, vector_out, valid_out) -> None:
        """Write the observation into the given ``(U, ...)`` views."""
        units = self.row_units()
        crop_out[...] = self.accessor.crop(
            units, self.layout.channels, self.layout.crop_radius,
            all_units=self.session.sim.units)
        vector_out[...] = 0.0
        for i, u in enumerate(units):
            valid_out[i] = u is not None
            if u is not None:
                for j, f in enumerate(self._vec):
                    vector_out[i, j] = f(u)


# ---------------------------------------------------------------------------
# The worker process
# ---------------------------------------------------------------------------
def _worker_paths() -> None:
    sys.path[:] = [p for p in sys.path if "build_cuda" not in str(p)]
    for p in (ROOT, ROOT / "src", CPU_BUILD, ROOT / "tests"):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _import_cpu_build():
    """``breach_physics`` from the CPU build, refusing any other (the CUDA
    build must never load in a farm worker)."""
    import breach_physics as bp
    got = Path(bp.__file__).resolve().parent
    if got != CPU_BUILD.resolve():
        raise RuntimeError(f"farm worker imported breach_physics from {got}, "
                           f"not the CPU build {CPU_BUILD}")
    return bp


def _attach(names: dict, layout: Layout) -> tuple:
    shms, arrays = {}, {}
    for key, (shape, dtype) in layout.shapes().items():
        shm = shared_memory.SharedMemory(name=names[key])
        shms[key] = shm
        arrays[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
    return shms, arrays


def _worker_main(conn, task: dict) -> None:
    """A farm worker: its envs' runners, driven by commands on ``conn``."""
    try:
        _worker_paths()
        bp = _import_cpu_build()
        run, layout = task["run"], task["layout"]
        levels = load_levels(run)
        shms, a = _attach(task["shm"], layout)
        runners = {e: EnvRunner(e, run, layout, levels, breach_physics=bp)
                   for e in task["envs"]}

        def publish(e, r):
            a["tick"][e] = r.session.tick_count
            a["episode"][e] = r.k
            a["episode_seed"][e] = r.seed
            a["level"][e] = r.level_index
            r.observe(a["crop"][e], a["vector"][e], a["valid"][e])

        conn.send(("ready", os.getpid()))
        while True:
            cmd, arg = conn.recv()
            if cmd == "close":
                break
            if cmd == "reset":
                info = {}
                for e in arg:
                    cps = runners[e].reset()
                    publish(e, runners[e])
                    info[e] = {"checkpoints": cps}
                conn.send(("ok", info))
            elif cmd == "step":
                info = {}
                for e, r in runners.items():
                    out = r.step(a["actions"][e])
                    a["counters"][e] = out["counters"]
                    a["reward"][e] = (out["counters"]
                                      * task["weights"]).sum(axis=1)
                    a["ticks"][e] = out["ticks"]
                    a["terminated"][e] = out["terminated"]
                    a["truncated"][e] = out["truncated"]
                    cps = out["checkpoints"]
                    if out["ended"]:
                        r.observe(a["final_crop"][e], a["final_vector"][e],
                                  a["final_valid"][e])
                        cps = cps + r.reset()
                    publish(e, r)
                    info[e] = {"checkpoints": cps, "ended": out["ended"],
                               "result": out["result"]}
                conn.send(("ok", info))
            elif cmd == "probe":
                conn.send(("ok", {
                    "pid": os.getpid(),
                    "breach_physics": str(Path(bp.__file__).resolve()),
                    "modules": sorted(m for m in sys.modules
                                      if m.split(".")[0] in
                                      ("renderer", "pyray", "raylib", "cupy")
                                      or "cuda" in m.lower())}))
            else:
                raise RuntimeError(f"unknown farm command {cmd!r}")
        for shm in shms.values():
            shm.close()
    except Exception:   # report, then die: the driver raises it
        try:
            conn.send(("error", traceback.format_exc()))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# The driver
# ---------------------------------------------------------------------------
class BatchedEnv:
    """N matches stepped together (see the module doc). ``run`` is the run
    file's path or a :class:`RunConfig`; it is the one source of the levels,
    ``n_envs``, the decimation, the policy, the observation and the reward
    rows (so its sha256 describes the run). ``backend`` must be
    ``"cpu_farm"`` in P1a. ``n_workers`` (default ``min(n_envs,
    os.cpu_count() - 1)``) processes share the envs in contiguous blocks --
    a machine property, not the run's: results do not depend on it."""

    def __init__(self, run, backend: str = "cpu_farm", *,
                 n_workers: int | None = None):
        if backend not in BACKENDS:
            raise ValueError(f"backend {backend!r} is not built (P1a builds "
                             f"{list(BACKENDS)}; the 'device' backend is "
                             f"P5..P9)")
        self.run = run if isinstance(run, RunConfig) else load_run(run)
        self.backend = backend
        self.levels = load_levels(self.run)
        self.layout = make_layout(self.run, self.levels)
        n = self.run.n_envs
        if n_workers is None:
            n_workers = max(1, min(n, (os.cpu_count() or 2) - 1))
        if not 1 <= int(n_workers) <= n:
            raise ValueError(f"n_workers must be in [1, n_envs={n}], got "
                             f"{n_workers}")
        self.n_workers = int(n_workers)
        self._shm, self._a = {}, {}
        self._procs, self._conns = [], []
        self._owner = [None] * n
        self._started = np.zeros(n, dtype=bool)
        self._closed = False
        try:
            for key, (shape, dtype) in self.layout.shapes().items():
                nbytes = max(1, int(np.prod(shape)) * np.dtype(dtype).itemsize)
                shm = shared_memory.SharedMemory(create=True, size=nbytes)
                self._shm[key] = shm
                self._a[key] = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
                self._a[key][...] = 0
            blocks = np.array_split(np.arange(n), self.n_workers)
            ctx = mp.get_context("spawn")
            weights = self.run.reward_weights()
            for w, envs in enumerate(blocks):
                envs = [int(e) for e in envs]
                for e in envs:
                    self._owner[e] = w
                parent, child = ctx.Pipe()
                task = {"run": self.run, "layout": self.layout, "envs": envs,
                        "shm": {k: s.name for k, s in self._shm.items()},
                        "weights": weights}
                p = ctx.Process(target=_worker_main, args=(child, task),
                                daemon=True, name=f"breach-farm-{w}")
                p.start()
                child.close()
                self._procs.append(p)
                self._conns.append(parent)
            for c in self._conns:
                self._recv(c)
        except BaseException:
            self.close()
            raise

    # -- plumbing ---------------------------------------------------------
    @property
    def run_sha256(self) -> str:
        """The run file's sha256 (empty for a RunConfig built in memory)."""
        return self.run.sha256

    @property
    def n_envs(self) -> int:
        return self.run.n_envs

    def worker_pids(self) -> list:
        return [p.pid for p in self._procs]

    def _recv(self, conn):
        kind, payload = conn.recv()
        if kind == "error":
            self.close()
            raise RuntimeError(f"farm worker failed:\n{payload}")
        return payload

    def _broadcast(self, per_worker: dict) -> dict:
        for w, msg in per_worker.items():
            self._conns[w].send(msg)
        info = {}
        for w in per_worker:
            info.update(self._recv(self._conns[w]))
        return info

    def probe(self) -> list:
        """Each worker's pid, the breach_physics file it loaded and any
        renderer / CUDA module it imported (a gate reads it)."""
        return [self._broadcast({w: ("probe", None)})
                for w in range(self.n_workers)]

    def _obs(self, prefix="") -> dict:
        a = self._a
        return {"crop": a[prefix + "crop"].copy(),
                "vector": a[prefix + "vector"].copy(),
                "valid": a[prefix + "valid"].astype(bool)}

    def _info_arrays(self) -> dict:
        a = self._a
        return {"tick": a["tick"].copy(), "episode": a["episode"].copy(),
                "episode_seed": a["episode_seed"].copy(),
                "level": a["level"].copy()}

    # -- the API ----------------------------------------------------------
    def reset(self, mask=None) -> tuple:
        """Start the next episode of the masked envs (all when ``mask`` is
        None; a mask is ``(N,)`` bool -- never an id list). Returns ``(obs,
        info)``; ``info["checkpoints"]`` holds each reset env's tick-0
        checkpoint."""
        self._check_open()
        n = self.n_envs
        if mask is None:
            mask = np.ones(n, dtype=bool)
        mask = np.asarray(mask)
        if mask.shape != (n,) or mask.dtype != bool:
            raise ValueError(f"reset mask must be a ({n},) bool array, got "
                             f"{mask.shape} {mask.dtype}")
        per = {}
        for e in np.flatnonzero(mask):
            per.setdefault(self._owner[int(e)], []).append(int(e))
        got = self._broadcast({w: ("reset", envs)
                               for w, envs in per.items()})
        self._started |= mask
        info = self._info_arrays()
        info["checkpoints"] = [got[e]["checkpoints"] if e in got else []
                               for e in range(n)]
        return self._obs(), info

    def step(self, actions) -> tuple:
        """Advance every env ``decimation`` ticks on ``actions`` ``(N, U,
        A)`` int; returns ``(obs, reward, terminated, truncated, info)``
        (see the module doc). Envs whose match ended were reset inside the
        call."""
        self._check_open()
        if not self._started.all():
            raise RuntimeError("step() before reset(): envs "
                               f"{np.flatnonzero(~self._started).tolist()} "
                               f"have no episode")
        actions = np.asarray(actions)
        want = self._a["actions"].shape
        if actions.shape != want or not np.issubdtype(actions.dtype,
                                                      np.integer):
            raise ValueError(f"actions must be an integer {want} array, got "
                             f"{actions.shape} {actions.dtype}")
        self._a["actions"][...] = actions
        got = self._broadcast({w: ("step", None)
                               for w in range(self.n_workers)})
        a = self._a
        n = self.n_envs
        ended = np.array([got[e]["ended"] for e in range(n)], dtype=bool)
        info = self._info_arrays()
        info["ticks"] = a["ticks"].copy()
        info["ended"] = ended
        final = self._obs("final_")
        for v in final.values():        # only the envs that ended carry one
            v[~ended] = 0
        info["final_obs"] = final
        info["result"] = [got[e]["result"] for e in range(n)]
        info["counters"] = a["counters"].copy()
        info["checkpoints"] = [got[e]["checkpoints"] for e in range(n)]
        return (self._obs(), a["reward"].copy(),
                a["terminated"].astype(bool), a["truncated"].astype(bool),
                info)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("this BatchedEnv is closed")

    def close(self) -> None:
        """Stop the workers and free the shared memory. Idempotent."""
        if self._closed:
            return
        self._closed = True
        for c in self._conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        deadline = time.monotonic() + 10.0
        for p in self._procs:
            p.join(max(0.0, deadline - time.monotonic()))
            if p.is_alive():
                p.terminate()
                p.join(2.0)
        for c in self._conns:
            c.close()
        self._a = {}
        for shm in self._shm.values():
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass
        self._shm = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


__all__ = [
    "BatchedEnv", "EnvRunner", "RunConfig", "Layout", "load_run",
    "parse_run", "make_layout", "load_levels", "episode_draw",
    "COUNTER_NAMES", "VECTOR_FIELDS", "BACKENDS", "EPISODE_SALT",
]
