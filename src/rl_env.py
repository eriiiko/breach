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
    out_dir = "..."               # OPTIONAL: the run's output directory
                                  # (absolute, or relative to the run file;
                                  # refused INSIDE the checkout): the run
                                  # manifest goes there
    [obs]
    spec = "crop"                 # OPTIONAL: the observation spec (#195 1g):
                                  # "crop" (default, the placeholder rows
                                  # below) or "v1" (simulation.obs_spec_v1;
                                  # the three crop keys are then refused)
    full_view = true              # OPTIONAL, "v1" only: serve the value
                                  # network's full view under info (1g (c));
                                  # false = none (a farm reading without it)
    channels = ["pressure", "smoke", "temperature", "solid", "material",
                "units_own", "units_enemy", "units_zombie"]
    crop_radius = 8               # the crop is (2r+1) x (2r+1) tiles
    vector = ["hp", "team", "alive", "facing", "x", "y"]
    [reward]                      # a weight per counter name (absent = 0)
    kills = 1.0
    deaths_bullet = -1.0

THE OBSERVATION (#195 1g). ``[obs] spec`` names the layout the sensor
accessor's ``observe`` door serves (``sensor_accessor.OBS_SPECS``). Under
``"v1"`` the keys are ``docs/rl_agent_obs_spec_v1_2026-11-12.md`` 4.1's:
``vector`` ``(N, U, 97)`` float32, ``list`` ``(N, U, E, 67)`` float32 with
``E = S + 33``, ``list_mask`` ``(N, U, E)`` uint8, ``list_uid`` ``(N, U, E)``
int32 (an EQUALITY key for the decoder, never a network input), the two
windows (1g (b), spec 3.9: ``near_f`` ``(N, U, 8, 25, 25)`` float16 and
``near_u`` ``(N, U, 5, 25, 25)`` uint8 at tile resolution, ``far_f``
``(N, U, 10, 25, 25)`` float16 and ``far_u`` ``(N, U, 5, 25, 25)`` uint8 in
4 x 4-tile blocks, both centred on the soldier, never rotated), and the two
row flags ``valid`` (the unit acts: alive and still on its team) and
``in_match`` (the roster slot holds a unit, dead or turned included) --
``(N, U)`` uint8 each; a row with ``valid`` 0 is all zeros. A run on ``"v1"``
refuses a level without a room graph at the map's shape (the compass reads
it). WHO MAY READ WHICH ROWS: a policy acting for team ``t`` reads only rows
``t*S .. t*S + S - 1`` of every key, through :meth:`BatchedEnv.team_rows`
(views of the env-owned arrays), and nothing of ``info``, ``reward`` or
``counters``; the trainer reads those for its losses.

THE FULL VIEW (#195 1g (c), spec section 5): what the training-only VALUE
network is shown, ``obs_spec_v1.FULL_VIEW_NAME`` = ``"full_v1"``, served by
the same accessor door beside ``"v1"`` (``[obs] full_view``, default true)
into the farm's OWN buffers under ``info["full_view"]`` of ``step`` /
``reset`` -- NEVER a key of ``obs`` --: ``full_list`` ``(N, U, 64, 85)``
float32 (every living unit of every side where it truly is, the soldier
itself first, every live grenade and charge, every living door),
``full_mask`` / ``full_uid`` ``(N, U, 64)``, ``full_match`` ``(N, U, 2T + 3)``
float32 and ``full_watched`` ``(N, U, 25, 25)`` uint8; built for every row
with ``in_match`` 1, a dead or turned soldier's centred on its last place
(spec 5.6). The terminal full view of an env that ended inside the step is
``info["final_full_view"]`` (the same five keys, zeros on every other row)
-- never a ``final_full_*`` key inside ``info["final_obs"]``. Under
``"crop"`` both are empty dicts.

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

THE OBSERVATION BUFFERS ARE THE ENV'S. ``obs["crop"]``, ``obs["vector"]``,
the ``info["final_obs"]`` arrays (and under ``"v1"`` every observation key,
the ``info["full_view"]`` and ``info["final_full_view"]`` arrays alike) are
the same arrays on every call,
overwritten by the next :meth:`BatchedEnv.step` / :meth:`BatchedEnv.reset`: a
caller that keeps one across a step copies it (a trainer's write into its
rollout buffer is that copy). A fresh ~10 MB array per step cost the driver
1.8 to 2.3 ms at 64 envs in page faults, with every worker idle meanwhile
(#29, 2026-11-11). Every other returned array is a fresh copy.

SNAPSHOTS AND THE RUN MANIFEST (#29 P1b, residency design §6). A training run
on a snapshot calls :func:`require_snapshot` (``BatchedEnv(...,
snapshot=tag)`` does): an ANNOTATED tag whose commit is HEAD, a tree clean
INCLUDING untracked files under ``src/``, ``levels/``, ``config.toml`` and
``cpp/``, ``simulation`` and ``breach_physics`` imported from this checkout's
own ``src/`` and ``cpp/build*/``, the module's build-time ``build_id()``
equal to HEAD, and the run file's sha256 recorded; the ``device`` backend
also needs an empty twin debt ledger (``tools/twin_debt.py``). Each check is
its own function taking the repo path. The farm needs no ledger and trains
at any tag. With ``[run] out_dir`` the env writes :data:`RUN_MANIFEST`
there (:func:`write_run_manifest`: commit, tag, build id, run and bank
sha256, python / numpy / torch / CuPy / CUDA runtime and driver versions,
the GPU, the host) and a later env on the same ``out_dir`` -- a resume --
refuses any change of it (:func:`check_run_manifest`, naming the field).
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing as mp
import os
import platform
import socket
import subprocess
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
#: The environment every farm worker starts with: ONE BLAS / OpenMP thread.
#: Nothing per tick contracts through BLAS (the no-BLAS rule), and OpenBLAS
#: otherwise commits a buffer per hardware thread in every process: measured
#: 2026-11-07, ~1.9 GB of private bytes per worker on the 16-thread Home
#: Desktop (19 MB with one thread) -- 8 workers would commit the machine's
#: RAM. Set before the worker starts (spawn re-imports numpy at once).
WORKER_ENV = {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1",
              "MKL_NUM_THREADS": "1"}

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
    out_dir: str = ""           # [run] out_dir, resolved; "" = none
    sight: str | None = None    # [run] sight (#195 1a): replaces every level's
                                # match.toml sight; None = the level's own
    spec: str = "crop"          # [obs] spec (#195 1g): the observation layout
    full_view: bool = True      # [obs] full_view (#195 1g (c)): serve the
                                # value network's full view under info ("v1")
    action_spec: str = "raw"    # [action] spec (#195 1h): the action layout

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
             "levels": True, "policy": True, "levels_dir": False,
             "out_dir": False, "sight": False}
#: The ``[obs]`` keys per spec (#195 1g): the three crop keys belong to the
#: ``"crop"`` placeholder alone; ``"v1"`` takes none (its layout is the spec).
_OBS_KEYS = {"crop": {"spec": False, "channels": True, "crop_radius": True,
                      "vector": True},
             "v1": {"spec": False, "full_view": False}}


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
    from simulation.sensor_accessor import OBS_SPECS, resolve_crop_channels
    bad = sorted(set(raw) - {"run", "obs", "reward", "action"})
    if bad:
        raise ValueError(f"{where}: unknown table(s) {bad} (a run file "
                         f"carries [run], [obs], [reward], [action])")
    run = _table(raw, "run", _RUN_KEYS, where)
    if not isinstance(raw.get("obs"), dict):
        raise ValueError(f"{where}: missing table [obs]")
    spec = raw["obs"].get("spec", "crop")
    if not isinstance(spec, str) or spec not in OBS_SPECS:
        raise ValueError(f"{where}: [obs] spec must be one of "
                         f"{list(OBS_SPECS)}, got {spec!r}")
    try:
        obs = _table(raw, "obs", _OBS_KEYS[spec], where)
    except ValueError as e:
        if spec != "crop" and "unknown key" in str(e):
            raise ValueError(f"{where}: {e} -- the crop keys (channels, "
                             f"crop_radius, vector) belong to the 'crop' "
                             f"spec; spec {spec!r} takes none") from None
        raise
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
    out_dir = ""
    if "out_dir" in run:
        od = run["out_dir"]
        if not isinstance(od, str) or not od:
            raise ValueError(f"{where}: [run] out_dir must be a path "
                             f"string, got {od!r}")
        p = Path(od)
        if not p.is_absolute():
            p = (base_dir or Path.cwd()) / p
        p = p.resolve()
        if p == ROOT.resolve() or ROOT.resolve() in p.parents:
            raise ValueError(f"{where}: [run] out_dir {str(p)!r} lies inside "
                             f"the checkout {str(ROOT)!r}: run outputs go "
                             f"outside it (a snapshot's tree stays clean)")
        out_dir = str(p)
    sight = None
    if "sight" in run:
        from simulation.match import check_sight
        sight = check_sight(run["sight"], f"{where}: [run] sight")
    channels, crop_radius, vector = (), 0, ()
    full_view = True
    if spec == "v1" and "full_view" in obs:
        full_view = obs["full_view"]
        if not isinstance(full_view, bool):
            raise ValueError(f"{where}: [obs] full_view must be true or "
                             f"false, got {full_view!r}")
    if spec == "crop":
        channels = _names_key(obs, "obs", "channels", where)
        try:
            channels = resolve_crop_channels(channels)
        except ValueError as e:
            raise ValueError(f"{where}: [obs] channels: {e}") from None
        crop_radius = _int_key(obs, "obs", "crop_radius", where, 0)
        vector = _names_key(obs, "obs", "vector", where, allow_empty=True)
        for v in vector:
            if v not in VECTOR_FIELDS:
                raise ValueError(f"{where}: [obs] vector names {v!r} "
                                 f"(known: {sorted(VECTOR_FIELDS)})")
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
    action_spec = "raw"
    if "action" in raw:
        from simulation.action_spec_v1 import ACTION_SPECS
        if not isinstance(raw["action"], dict):
            raise ValueError(f"{where}: [action] must be a table")
        act = _table(raw, "action", {"spec": False}, where)
        action_spec = act.get("spec", "raw")
        if (not isinstance(action_spec, str)
                or action_spec not in ACTION_SPECS):
            raise ValueError(f"{where}: [action] spec must be one of "
                             f"{list(ACTION_SPECS)}, got {action_spec!r}")
        if action_spec == "v1" and spec != "v1":
            raise ValueError(f"{where}: [action] spec 'v1' needs [obs] spec "
                             f"'v1' (its aim pointer indexes the v1 list), "
                             f"got [obs] spec {spec!r}")
    return RunConfig(seed=seed, n_envs=n_envs, decimation=decimation,
                     levels=levels, policy=pol, levels_dir=levels_dir,
                     channels=channels, crop_radius=crop_radius,
                     vector=vector, reward=reward, out_dir=out_dir,
                     sight=sight, spec=spec, full_view=full_view,
                     action_spec=action_spec)


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
    spec: str = "crop"          # the observation spec (#195 1g)
    full_view: bool = False     # the value network's full view served (1g (c))
    action_spec: str = "raw"    # the action spec (#195 1h)
    n_kinds: int = 0            # K: the hand-grenade rows ("v1" actions)

    @property
    def n_rows(self) -> int:
        return self.n_teams * self.n_slots

    @property
    def full_keys(self) -> tuple:
        """The full view's keys (``info["full_view"]``), in order (each has
        a ``final_`` twin); empty when it is not served."""
        if not self.full_view:
            return ()
        from simulation.obs_spec_v1 import FULL_KEYS
        return FULL_KEYS

    def full_shapes(self) -> dict:
        if not self.full_view:
            return {}
        from simulation.obs_spec_v1 import full_shapes
        return full_shapes(self.n_envs, self.n_rows, self.n_teams)

    @property
    def crop_side(self) -> int:
        return 2 * self.crop_radius + 1

    @property
    def n_entries(self) -> int:
        """``E`` of the ``"v1"`` list: ``S + 33`` (spec 2.1)."""
        from simulation.obs_spec_v1 import ENTRIES_BEYOND_SLOTS
        return self.n_slots + ENTRIES_BEYOND_SLOTS

    @property
    def head_sizes(self) -> tuple:
        """The five heads' sizes under ``[action] spec = "v1"`` (empty
        under ``"raw"``)."""
        if self.action_spec != "v1":
            return ()
        from simulation.action_spec_v1 import head_sizes
        return head_sizes(self.n_entries, self.n_kinds)

    @property
    def mask_keys(self) -> tuple:
        """The action masks served in ``obs`` (``"v1"`` actions only)."""
        if self.action_spec != "v1":
            return ()
        from simulation.action_spec_v1 import MASK_KEYS
        return MASK_KEYS

    @property
    def obs_keys(self) -> tuple:
        """The observation keys of the spec, in order (each has a
        ``final_`` twin), and -- under ``[action] spec = "v1"`` -- the
        action masks after them (#195 1h: what a policy is FED)."""
        if self.spec == "crop":
            return ("crop", "vector", "valid")
        from simulation.obs_spec_v1 import OBS_KEYS
        return OBS_KEYS + ("valid", "in_match") + self.mask_keys

    def obs_shapes(self) -> dict:
        n, u = self.n_envs, self.n_rows
        if self.spec == "crop":
            k = self.crop_side
            c, v = len(self.channels), len(self.vector)
            return {"crop": ((n, u, c, k, k), np.float32),
                    "vector": ((n, u, v), np.float32),
                    "valid": ((n, u), np.uint8)}
        from simulation.obs_spec_v1 import shapes
        out = shapes(n, u, self.n_slots)
        if self.action_spec == "v1":
            from simulation.action_spec_v1 import mask_shapes
            out.update(mask_shapes(n, u, self.n_entries, self.n_kinds))
        return out

    def shapes(self) -> dict:
        n, u = self.n_envs, self.n_rows
        kk = len(self.counters)
        out = {"actions": ((n, u, self.n_actions), np.int32)}
        obs = self.obs_shapes()
        out.update(obs)
        out.update({"final_" + k: v for k, v in obs.items()})
        full = self.full_shapes()
        out.update(full)
        out.update({"final_" + k: v for k, v in full.items()})
        out.update({
            "reward": ((n, u), np.float32),
            "counters": ((n, u, kk), np.float32),
            "terminated": ((n,), np.uint8),
            "truncated": ((n,), np.uint8),
            "tick": ((n,), np.int64),
            "episode": ((n,), np.int64),
            "episode_seed": ((n,), np.uint64),
            "level": ((n,), np.int64),
            "ticks": ((n,), np.int64),
        })
        return out


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
    n_kinds, n_actions = 0, len(action_schema())
    if run.action_spec == "v1":
        from simulation.action_spec_v1 import HEADS
        from simulation.weapons import get_tables
        n_kinds, n_actions = len(get_tables().hand_grenade_rows), len(HEADS)
    return Layout(n_envs=run.n_envs, n_teams=n_teams, n_slots=n_slots,
                  n_actions=n_actions, channels=run.channels,
                  crop_radius=run.crop_radius, vector=run.vector,
                  spec=run.spec,
                  full_view=bool(run.spec == "v1" and run.full_view),
                  action_spec=run.action_spec, n_kinds=n_kinds)


def load_levels(run: RunConfig) -> list:
    """Load every level of the run (refusing one without a match.toml,
    whose team count disagrees with a per-team ``policy`` list, or -- under
    an observation spec that reads the room graph, ``"v1"`` -- without a
    room graph at the map's shape, #195 1g)."""
    import level_loader
    out = []
    for name in run.levels:
        lv = level_loader.load(name, run.levels_dir)
        if lv.match is None:
            raise ValueError(f"run level {name!r} has no "
                             f"{level_loader.MATCH_FILENAME}")
        run.team_controllers(len(lv.match.teams), repr(name))
        if run.spec == "v1":
            graph = lv.rooms
            if graph is None or (tuple(graph.grid.shape)
                                 != tuple(lv.tilemap.shape)):
                raise ValueError(
                    f"run level {name!r} has no room graph at the map's "
                    f"shape ({level_loader.ROOMS_FILENAME}): a run whose "
                    f"observation spec ({run.spec!r}) reads the room graph "
                    f"refuses a level without one")
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
        self.ctx = None             # the "v1" spec's per-episode context

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
        if self.run.sight is not None:      # #195 1a: the run file's rule wins;
            spec = replace(spec, sight=self.run.sight)   # the recorder keeps it
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
        if self.layout.spec == "v1":
            from simulation.obs_spec_v1 import ObsContext
            self.ctx = ObsContext.build(sim, spec, S,
                                        n_teams=self.layout.n_teams)
        self.act_tables = None
        self._list_uid = None
        if self.layout.action_spec == "v1":
            # #195 1h: the decoder, set before any row is handed over
            from simulation.action_spec_v1 import action_tables
            self.act_tables = action_tables(sim, S)
            for pol in self.policies.values():
                pol.configure("v1", self.act_tables)
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
        if self.layout.action_spec == "v1":
            # #195 1h: each policy team's (S, 5) block with the list_uid of
            # the observation it was chosen on (this runner's last publish)
            lu = self._list_uid
            if lu is None:
                raise RuntimeError(
                    f"env {self.env}: a 'v1' action step needs the "
                    f"observation it was chosen on -- observe() after "
                    f"reset() and after every step()")
            self._list_uid = None
            for team, pol in self.policies.items():
                pol.set_decisions(actions[team * S:(team + 1) * S],
                                  lu[team * S:(team + 1) * S])
        else:
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

    def roster_units(self) -> list:
        """The ROSTER unit of every row (dead and turned ones included; the
        full view's rows, spec 5.6), ``None`` at an empty roster place."""
        sim = self.session.sim
        S = self.layout.n_slots
        out = [None] * self.layout.n_rows
        for t in range(len(self.session.spec.teams)):
            for u in sim.roster(t):
                if u is not None:
                    out[t * S + u.slot] = u
        return out

    def row_flags(self) -> tuple:
        """``(valid, in_match)`` per row (spec 4.2): ``in_match`` while the
        roster slot holds a unit (dead or turned included), ``valid`` while
        that unit is alive and still on the row's team; an empty roster
        place reads 0, 0."""
        sim = self.session.sim
        S = self.layout.n_slots
        n = self.layout.n_rows
        valid = np.zeros(n, dtype=np.uint8)
        in_match = np.zeros(n, dtype=np.uint8)
        for t in range(len(self.session.spec.teams)):
            for u in sim.roster(t):
                if u is None:
                    continue
                in_match[t * S + u.slot] = 1
                if u.alive and u.team == t:
                    valid[t * S + u.slot] = 1
        return valid, in_match

    def observe(self, out: dict) -> None:
        """Write the observation into ``out``'s ``(U, ...)`` views (the
        spec's keys, ``layout.obs_keys``) through the sensor accessor's
        ``observe`` door, and the row flags beside it -- and, when ``out``
        carries the full view's keys (``layout.full_keys``), the full view
        through the same door for the rows' ROSTER units (1g (c))."""
        sim = self.session.sim
        units = self.row_units()
        if self.layout.spec == "crop":
            self.accessor.observe("crop", units, out,
                                  channels=self.layout.channels,
                                  r=self.layout.crop_radius,
                                  all_units=sim.units, vector_fields=self._vec)
            out["valid"][...] = [u is not None for u in units]
            return
        aims = None
        if self.layout.action_spec == "v1":
            aims = {}
            for pol in self.policies.values():
                aims.update(pol.aim_view())
        self.accessor.observe("v1", units, out, sim=sim, context=self.ctx,
                              aims=aims)
        valid, in_match = self.row_flags()
        out["valid"][...] = valid
        out["in_match"][...] = in_match
        if self.layout.action_spec == "v1":
            # #195 1h: the masks, at the start of the decision tick, from
            # the list just built (one gate call per hand bit)
            from simulation.action_spec_v1 import build_masks
            lst = out["list"]
            build_masks(sim, units, lst[:, :, 0], out["list_mask"],
                        out["list_uid"], self.act_tables,
                        self.run.decimation, out)
            self._list_uid = np.array(out["list_uid"], copy=True)
        if self.layout.full_keys and self.layout.full_keys[0] in out:
            from simulation.sensor_accessor import FULL_VIEW_SPEC
            self.accessor.observe(FULL_VIEW_SPEC, self.roster_units(), out,
                                  sim=sim, context=self.ctx)


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

        keys = layout.obs_keys + layout.full_keys

        def publish(e, r):
            a["tick"][e] = r.session.tick_count
            a["episode"][e] = r.k
            a["episode_seed"][e] = r.seed
            a["level"][e] = r.level_index
            r.observe({k: a[k][e] for k in keys})

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
                        r.observe({k: a["final_" + k][e] for k in keys})
                        cps = cps + r.reset()
                    publish(e, r)
                    info[e] = {"checkpoints": cps, "ended": out["ended"],
                               "result": out["result"]}
                conn.send(("ok", info))
            elif cmd == "probe":
                import simulation
                bid = getattr(bp, "build_id", None)
                conn.send(("ok", {
                    "pid": os.getpid(),
                    "breach_physics": str(Path(bp.__file__).resolve()),
                    "simulation": str(Path(simulation.__file__).resolve()),
                    "build_id": bid() if bid is not None else None,
                    "env": {k: os.environ.get(k) for k in WORKER_ENV},
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
# Snapshots (#29 P1b, residency design §6)
# ---------------------------------------------------------------------------
#: The backends require_snapshot knows ("device" refuses until P8b exists).
SNAPSHOT_BACKENDS = ("cpu_farm", "device")
#: Where an UNTRACKED file makes a snapshot dirty (anywhere else -- art,
#: notes -- it may stay). A modified tracked file is dirty anywhere.
SNAPSHOT_PATHS = ("src/", "levels/", "config.toml", "cpp/")
#: The run manifest's file name inside ``[run] out_dir``.
RUN_MANIFEST = "run_manifest.json"
RUN_MANIFEST_FORMAT = 1
#: Recorded, never compared on a resume: a resumed Berzelius job lands on
#: another node of the same image (its GPU and driver fields are compared).
RUN_MANIFEST_INFORMATIONAL = ("hostname",)


class SnapshotRefused(RuntimeError):
    """A :func:`require_snapshot` check failed; the message names it."""


class RunManifestChanged(RuntimeError):
    """A resume found the run manifest changed; the message names the
    field(s)."""


def _git(repo, *args, strip=True) -> tuple:
    """``(returncode, stdout)`` of ``git -C repo <args>`` (stripped unless
    ``strip`` is False: porcelain status lines start with a space)."""
    try:
        r = subprocess.run(["git", "-C", str(repo), *args],
                           capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return 1, str(e)
    if r.returncode != 0:
        return r.returncode, ""
    return 0, r.stdout.strip() if strip else r.stdout


def _refuse(check: str, msg: str):
    raise SnapshotRefused(f"require_snapshot: {check}: {msg}")


def _head(repo) -> str:
    rc, sha = _git(repo, "rev-parse", "HEAD")
    if rc != 0 or not sha:
        _refuse("head", f"{repo} is not a git checkout with a HEAD")
    return sha


def check_annotated_tag(repo, tag: str) -> str:
    """The tag's commit SHA; refuses unless ``refs/tags/<tag>`` exists and
    is ANNOTATED (``git cat-file -t`` says ``tag``; a lightweight tag points
    straight at a commit and records no tagger)."""
    rc, kind = _git(repo, "cat-file", "-t", f"refs/tags/{tag}")
    if rc != 0:
        _refuse("annotated tag", f"no tag {tag!r} in {repo}")
    if kind != "tag":
        _refuse("annotated tag", f"tag {tag!r} is lightweight (git cat-file "
                f"-t says {kind!r}); a training snapshot is an annotated tag "
                f"(git tag -a)")
    rc, sha = _git(repo, "rev-parse", f"refs/tags/{tag}^{{}}")
    if rc != 0 or not sha:
        _refuse("annotated tag", f"tag {tag!r} does not peel to a commit")
    return sha


def check_head_at_tag(repo, tag: str) -> str:
    """HEAD's SHA; refuses unless it equals ``git rev-parse <tag>^{}``."""
    head = _head(repo)
    rc, sha = _git(repo, "rev-parse", f"refs/tags/{tag}^{{}}")
    if head != sha:
        _refuse("head at tag", f"HEAD is {head}, tag {tag!r} is {sha}: "
                f"check the tag out detached in its own folder")
    return head


def check_clean_tree(repo) -> None:
    """Refuses a modified tracked file anywhere, or an untracked file under
    :data:`SNAPSHOT_PATHS` (ignored files are not untracked: the build dirs
    and ``__pycache__`` stay out of it)."""
    rc, out = _git(repo, "status", "--porcelain", "-z",
                   "--untracked-files=all", strip=False)
    if rc != 0:
        _refuse("clean tree", f"git status failed in {repo}")
    entries = out.split("\0")
    bad, i = [], 0
    while i < len(entries):
        e = entries[i]
        i += 1
        if not e:
            continue
        xy, path = e[:2], e[3:]
        if xy[0] in "RC":            # a rename / copy: its source follows
            i += 1
        if xy == "??":
            if any(path == p or path.startswith(p) for p in SNAPSHOT_PATHS):
                bad.append(f"untracked {path}")
        else:
            bad.append(f"modified {path} ({xy.strip()})")
    if bad:
        shown = "; ".join(bad[:8]) + ("; ..." if len(bad) > 8 else "")
        _refuse("clean tree", f"{len(bad)} change(s) in {repo}: {shown}")


def _under(path, root) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def check_module_paths(repo, simulation_file, bp_file) -> None:
    """Refuses unless ``simulation`` was imported from ``<repo>/src/`` and
    ``breach_physics`` from a ``<repo>/cpp/build*/`` directory -- this
    checkout's own code and build, never another worktree's."""
    repo = Path(repo)
    if not simulation_file or not _under(simulation_file, repo / "src"):
        _refuse("module paths", f"simulation was imported from "
                f"{simulation_file!r}, not {repo / 'src'}")
    ok = False
    if bp_file and _under(bp_file, repo / "cpp"):
        rel = Path(bp_file).resolve().relative_to((repo / "cpp").resolve())
        ok = len(rel.parts) > 1 and rel.parts[0].startswith("build")
    if not ok:
        _refuse("module paths", f"breach_physics was imported from "
                f"{bp_file!r}, not a {repo / 'cpp'}/build*/ directory")


def check_build_id(repo, build_id) -> None:
    """Refuses unless the module's build-time ``build_id()`` is HEAD's SHA
    (a module built from another commit, or from a dirty tree, is not this
    snapshot's)."""
    head = _head(repo)
    if not build_id:
        _refuse("build id", "breach_physics has no build_id() (built before "
                "#29 P1b): rebuild it in this checkout")
    if build_id != head:
        why = ("from a dirty tree" if build_id.startswith(head)
               else f"from {build_id}")
        _refuse("build id", f"breach_physics was built {why}, HEAD is "
                f"{head}: rebuild it in this checkout")


def check_run_recorded(repo, run) -> str:
    """The run file's sha256; refuses a run without one (a RunConfig built
    in memory) or whose file lies inside the checkout."""
    cfg = run if isinstance(run, RunConfig) else load_run(run)
    if not cfg.sha256 or not cfg.path:
        _refuse("run file", "the run file's sha256 is not recorded (a "
                "RunConfig built in memory has none: load the run from its "
                "file)")
    if _under(cfg.path, repo):
        _refuse("run file", f"the run file {cfg.path} lies inside the "
                f"checkout {repo} (it lives outside the snapshot tree)")
    return cfg.sha256


def _twin_debt_module():
    import importlib.util
    path = ROOT / "tools" / "twin_debt.py"
    spec = importlib.util.spec_from_file_location("_breach_twin_debt", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def check_twin_debt(repo) -> None:
    """Refuses while ``tools/twin_debt.py`` reports debt for any twin."""
    rows = _twin_debt_module().debt(repo)
    if rows:
        _refuse("twin debt", "the device backend needs an empty ledger; in "
                "debt: " + "; ".join(f"{n} ({r})" for n, r in rows))


def module_probe() -> dict:
    """``simulation`` / ``breach_physics`` as imported in THIS process: their
    files and the module's build id (the default probe)."""
    import breach_physics as bp
    import simulation
    bid = getattr(bp, "build_id", None)
    return {"simulation_file": simulation.__file__, "bp_file": bp.__file__,
            "build_id": bid() if bid is not None else None}


def require_snapshot(tag: str, backend: str, run, *, repo=None,
                     probe: dict | None = None) -> dict:
    """THE training door (residency design §6): refuses
    (:class:`SnapshotRefused`, naming the failed check) unless ``tag`` is an
    annotated tag, HEAD is its commit, the tree is clean (untracked files
    included under :data:`SNAPSHOT_PATHS`), ``simulation`` and
    ``breach_physics`` come from this checkout, the module's build id is
    HEAD and ``run`` (a run file path or a :class:`RunConfig` loaded from
    one) has its sha256 recorded. ``backend="device"`` also needs an empty
    twin debt ledger, then refuses: no device backend exists yet (P8b).

    ``repo`` defaults to this checkout; ``probe`` (``simulation_file``,
    ``bp_file``, ``build_id``) to :func:`module_probe` -- the farm passes its
    workers'. Returns ``{tag, commit, build_id, run_sha256, backend}``."""
    if backend not in SNAPSHOT_BACKENDS:
        _refuse("backend", f"{backend!r} is not a backend (known: "
                f"{list(SNAPSHOT_BACKENDS)})")
    repo = Path(repo) if repo is not None else ROOT
    probe = module_probe() if probe is None else probe
    commit = check_annotated_tag(repo, tag)
    check_head_at_tag(repo, tag)
    check_clean_tree(repo)
    check_module_paths(repo, probe.get("simulation_file"),
                       probe.get("bp_file"))
    check_build_id(repo, probe.get("build_id"))
    run_sha = check_run_recorded(repo, run)
    if backend == "device":
        check_twin_debt(repo)
        _refuse("backend", "no device backend yet (P5..P9; P8b makes it)")
    return {"tag": tag, "commit": commit, "build_id": probe.get("build_id"),
            "run_sha256": run_sha, "backend": backend}


# ---------------------------------------------------------------------------
# The run manifest (#29 P1b, residency design §6)
# ---------------------------------------------------------------------------
def _dist(*names):
    """``name==version`` of the first installed distribution, read from its
    metadata -- never imported."""
    from importlib import metadata
    for n in names:
        try:
            return f"{n}=={metadata.version(n)}"
        except metadata.PackageNotFoundError:
            continue
    return None


def _nvidia_smi():
    """``(gpu names, driver version)`` from ``nvidia-smi`` (a subprocess: no
    CUDA context in this process), or ``(None, None)``."""
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                            "--format=csv,noheader"], capture_output=True,
                           text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return None, None
    rows = [ln.split(",") for ln in r.stdout.splitlines() if "," in ln]
    if r.returncode != 0 or not rows:
        return None, None
    return ("; ".join(x[0].strip() for x in rows), rows[0][1].strip())


def environment_versions() -> dict:
    """The run's environment, read WITHOUT importing torch or CuPy (a farm
    process never creates a CUDA context): python and numpy, torch and CuPy
    from their distribution metadata, the CUDA runtime torch was built
    against (its ``+cuNNN`` local version), the GPU(s) and driver from
    ``nvidia-smi``, the host name."""
    torch = _dist("torch")
    cuda_rt = None
    if torch and "+cu" in torch:
        tag = torch.rsplit("+cu", 1)[1]
        if tag.isdigit() and len(tag) >= 3:
            cuda_rt = f"{int(tag[:-1])}.{tag[-1]}"
    gpu, driver = _nvidia_smi()
    return {"python": platform.python_version(), "numpy": np.__version__,
            "torch": torch,
            "cupy": _dist("cupy-cuda12x", "cupy-cuda11x", "cupy"),
            "cuda_runtime": cuda_rt, "cuda_driver": driver, "gpu": gpu,
            "hostname": socket.gethostname()}


def run_manifest_record(*, run_sha256: str, build_id, backend: str,
                        tag: str | None = None, bank_sha256=None,
                        repo=None) -> dict:
    """The manifest's content: format, backend, the snapshot tag (None off a
    snapshot) and the commit run (HEAD; the tag's commit on a snapshot), the
    module's build id, the run file's and the bank's sha256 (the bank is
    P8a's: None until then), and :func:`environment_versions`."""
    repo = Path(repo) if repo is not None else ROOT
    rc, commit = _git(repo, "rev-parse", "HEAD")
    rec = {"format": RUN_MANIFEST_FORMAT, "backend": backend, "tag": tag,
           "commit": commit if rc == 0 and commit else None,
           "build_id": build_id, "run_sha256": run_sha256,
           "bank_sha256": bank_sha256}
    rec.update(environment_versions())
    return rec


def write_run_manifest(path, **kw) -> dict:
    """Write :func:`run_manifest_record` (``kw``) as JSON at ``path``
    (parents created); returns the record."""
    rec = run_manifest_record(**kw)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8")
    return rec


def check_run_manifest(path, **kw) -> dict:
    """A resume's door: the manifest at ``path`` against the record ``kw``
    describes now; :class:`RunManifestChanged` naming every changed field
    (all but :data:`RUN_MANIFEST_INFORMATIONAL`). Returns the stored
    record."""
    path = Path(path)
    try:
        old = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        raise RunManifestChanged(f"{path}: not a run manifest ({e})") from None
    new = run_manifest_record(**kw)
    changed = [f"{k}: {old.get(k)!r} -> {new.get(k)!r}"
               for k in sorted(set(old) | set(new))
               if k not in RUN_MANIFEST_INFORMATIONAL
               and old.get(k) != new.get(k)]
    if changed:
        raise RunManifestChanged(f"{path}: the run changed since it was "
                                 f"written -- " + "; ".join(changed))
    return old


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
    a machine property, not the run's: results do not depend on it. Each
    worker starts with :data:`WORKER_ENV` (one BLAS thread).

    ``snapshot`` (a tag name): :func:`require_snapshot` over the modules the
    WORKERS imported, refusing before the first reset. With ``[run]
    out_dir`` the env writes the run manifest there, or -- when one is there
    already, a resume -- refuses any change of it (:func:`check_run_manifest`).
    """

    def __init__(self, run, backend: str = "cpu_farm", *,
                 n_workers: int | None = None, snapshot: str | None = None):
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
            # The env-owned observation buffers (see the module doc): one
            # per observation key of the spec and its ``final_`` twin. The
            # "crop" spec's ``valid`` keeps its bool copy (as it was).
            self._buf = {}
            for k in self.layout.obs_keys + self.layout.full_keys:
                if not (self.layout.spec == "crop" and k == "valid"):
                    self._buf[k] = np.zeros_like(self._a[k])
                self._buf["final_" + k] = (
                    np.zeros(self._a["final_" + k].shape, dtype=bool)
                    if self.layout.spec == "crop" and k == "valid"
                    else np.zeros_like(self._a["final_" + k]))
            self._final_rows = np.zeros(n, dtype=bool)
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
                saved = {k: os.environ.get(k) for k in WORKER_ENV}
                os.environ.update(WORKER_ENV)
                try:
                    p.start()
                finally:
                    for k, v in saved.items():
                        if v is None:
                            os.environ.pop(k, None)
                        else:
                            os.environ[k] = v
                child.close()
                self._procs.append(p)
                self._conns.append(parent)
            for c in self._conns:
                self._recv(c)
            self.snapshot = None
            self.manifest_path = None
            if snapshot is not None or self.run.out_dir:
                probe = self._module_probe()
                if snapshot is not None:
                    self.snapshot = require_snapshot(
                        snapshot, backend, self.run, probe=probe)
                if self.run.out_dir:
                    kw = dict(run_sha256=self.run.sha256,
                              build_id=probe["build_id"], backend=backend,
                              tag=snapshot)
                    path = Path(self.run.out_dir) / RUN_MANIFEST
                    if path.exists():
                        check_run_manifest(path, **kw)
                    else:
                        write_run_manifest(path, **kw)
                    self.manifest_path = str(path)
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
        """Each worker's pid, the breach_physics and simulation files it
        loaded, the module's build id and any renderer / CUDA module it
        imported (a gate reads it)."""
        return [self._broadcast({w: ("probe", None)})
                for w in range(self.n_workers)]

    def _module_probe(self) -> dict:
        """The workers' ``simulation`` / ``breach_physics`` files and build
        id as :func:`require_snapshot`'s probe; every worker must agree."""
        seen = {(p["simulation"], p["breach_physics"], p["build_id"])
                for p in self.probe()}
        if len(seen) != 1:
            raise RuntimeError(f"farm workers loaded different modules: "
                               f"{sorted(seen)}")
        sim, bp_file, bid = seen.pop()
        return {"simulation_file": sim, "bp_file": bp_file, "build_id": bid}

    def _obs(self) -> dict:
        """The observation, in the env's own buffers (valid until the next
        step / reset; see the module doc): the spec's keys."""
        a, b = self._a, self._buf
        out = {}
        for k in self.layout.obs_keys:
            if k not in b:                      # "crop"'s valid: a bool copy
                out[k] = a[k].astype(bool)
                continue
            np.copyto(b[k], a[k])
            out[k] = b[k]
        return out

    def _final(self, keys, ended) -> dict:
        """The terminal arrays ``keys`` of the envs that ended, zeros on
        every other row, in the env's own buffers: last step's rows
        (``_final_rows``, the caller moves it) are wiped and only this
        step's are copied."""
        a, b = self._a, self._buf
        out = {k: b["final_" + k] for k in keys}
        if self._final_rows.any():
            for v in out.values():
                v[self._final_rows] = 0
        if ended.any():
            for key, v in out.items():
                v[ended] = a["final_" + key][ended]
        return out

    def _final_obs(self, ended) -> dict:
        """The terminal observation of the envs that ended (the spec's
        keys)."""
        return self._final(self.layout.obs_keys, ended)

    def _full_view(self) -> dict:
        """The full view (#195 1g (c), spec 5), in the env's own buffers:
        the layout's full keys (empty under ``"crop"`` or ``full_view =
        false``)."""
        a, b = self._a, self._buf
        out = {}
        for k in self.layout.full_keys:
            np.copyto(b[k], a[k])
            out[k] = b[k]
        return out

    def _final_full_view(self, ended) -> dict:
        """The terminal full view of the envs that ended (spec 5.2's
        ``info["final_full_view"]``)."""
        return self._final(self.layout.full_keys, ended)

    def team_rows(self, obs: dict, team: int) -> dict:
        """THE one path a trainer takes to a team's observation (#195 1g,
        spec 4.1): every key of ``obs`` (an ``obs`` or ``final_obs`` dict
        this env returned) as the VIEW ``arr[:, t*S : (t+1)*S]`` of team
        ``team``'s rows -- never a copy (the env-owned rule). A policy
        acting for a team is fed only through this."""
        S = self.layout.n_slots
        t = int(team)
        if not 0 <= t < self.layout.n_teams:
            raise ValueError(f"team {team} is not a row block of this run "
                             f"(T = {self.layout.n_teams})")
        return {k: v[:, t * S:(t + 1) * S] for k, v in obs.items()}

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
        info["full_view"] = self._full_view()
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
        masked = None
        if self.layout.action_spec == "v1":
            # #195 1h: an index outside its head is a trainer bug, refused;
            # a masked in-range one is decoded by its fallback and COUNTED,
            # against the masks served with the observation it answers
            from simulation.action_spec_v1 import check_actions, masked_choices
            check_actions(actions, self.layout.head_sizes)
            masked = masked_choices(actions, {k: self._a[k]
                                              for k in self.layout.mask_keys})
        self._a["actions"][...] = actions
        got = self._broadcast({w: ("step", None)
                               for w in range(self.n_workers)})
        a = self._a
        n = self.n_envs
        ended = np.array([got[e]["ended"] for e in range(n)], dtype=bool)
        info = self._info_arrays()
        info["ticks"] = a["ticks"].copy()
        info["ended"] = ended
        info["final_obs"] = self._final_obs(ended)
        info["final_full_view"] = self._final_full_view(ended)
        self._final_rows = ended.copy()
        info["full_view"] = self._full_view()
        info["result"] = [got[e]["result"] for e in range(n)]
        info["counters"] = a["counters"].copy()
        info["checkpoints"] = [got[e]["checkpoints"] for e in range(n)]
        if masked is not None:
            info["masked_choices"] = masked
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
    "require_snapshot", "SnapshotRefused", "check_annotated_tag",
    "check_head_at_tag", "check_clean_tree", "check_module_paths",
    "check_build_id", "check_run_recorded", "check_twin_debt",
    "module_probe", "write_run_manifest", "check_run_manifest",
    "run_manifest_record", "environment_versions", "RunManifestChanged",
    "RUN_MANIFEST", "SNAPSHOT_PATHS", "SNAPSHOT_BACKENDS",
]
