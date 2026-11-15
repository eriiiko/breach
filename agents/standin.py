"""agents/standin.py -- a stand-in for ``rl_env.BatchedEnv`` (rung 2 note
2b, 8; design 8): the made-up tasks run against it, so the trainer under
test is the one that will meet the farm. The tasks of patch 2d use it too.

WHAT THE TRAINER READS of ``BatchedEnv``, and so what this serves under the
SAME names (read off ``src/rl_env.py`` at main's head; a test holds every
name to the game's source): the attributes ``run`` (a real
``rl_env.RunConfig``: ``team_controllers``, ``reward_profile``) and
``layout`` (a real ``rl_env.Layout``: the game's own shape tables) and
``n_envs``; the methods ``reset(mask=None) -> (obs, info)``,
``step(actions) -> (obs, reward, terminated, truncated, info)``,
``team_rows(obs, team)`` and ``close()``; the ``info`` keys
``full_view``, ``ended``, ``final_obs``, ``final_full_view``,
``rows_ended``, ``masked_choices`` (and ``tick``, ``episode``, ``result``).
Not served, because the trainer never reads them: ``counters``, the
``team_*`` books, ``checkpoints``, ``episode_seed``, ``level``, ``ticks``.

ITS BEHAVIOURS (the note's list):

1. the observation arrays, the full view and the ``final`` arrays are the
   SAME arrays on every call, overwritten in place;
2. a match that ends inside a step is restarted inside the same call: the
   returned view is the new match's first, the ended match's last view is
   under ``info["final_obs"]`` / ``info["final_full_view"]`` (zeros on
   every other row, wiped the step after), ``info["ended"]`` says so, and
   the step's rewards are the ended match's;
3. the reward is ``(N, U, 5)`` in the game's group order, zeros for an
   empty roster place and for a row whose records ended in an earlier
   step; ``info["rows_ended"]`` (every row with open records at a match's
   end, and a row whose team the task ends early);
4. every team's rows are observed, masked and paid, the non-learning ones
   too; ``team_rows`` gives views of one team's rows;
5. a row with ``valid`` 0 is all zeros with the default masks
   (``action_spec_v1.default_masks``);
6. actions are checked as the game checks them (``check_actions``: shape,
   range) and a masked choice is counted in ``info["masked_choices"]``
   (``masked_choices``, against the masks served with the observation the
   actions answer);
7. the arrays have the game's names, number formats and shapes, and values
   in the game's real ranges: a fixed made-up scenery per row from
   :mod:`agents.fake` (kinds inside their tables, counts small), which a
   task overwrites where its signal lives.

A TASK (:class:`Task`) decides what a match is: who stands in the roster,
what each living row is shown and allowed (written over the scenery and the
game-like masks), what each step pays, who dies, whose records end, when
the match is over. A match also ends on its clock (``Task.match_len``
decisions): an ending, as under every named profile.

The stand-in also keeps its OWN books of what the learning teams (the teams
on ``policy``) were paid, per group, over the decisions of their living
rows (:attr:`StandIn.paid`, :attr:`StandIn.paid_decisions`): a task's
score is read there, never from the trainer's batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from agents import _game  # noqa: F401
from agents import fake
from agents.sizes import sizes_for
from simulation import action_spec_v1 as A
from simulation import obs_spec_v1 as O
from simulation.reward_v1 import PROFILES, REWARD_GROUPS
import rl_env

#: The ``info`` keys the stand-in serves (a test finds each in the game's
#: ``BatchedEnv`` source).
INFO_KEYS = ("tick", "episode", "ended", "final_obs", "final_full_view",
             "full_view", "result", "rows_ended", "masked_choices")
#: The attributes and methods the trainer reads.
ATTRIBUTES = ("run", "layout", "n_envs", "reset", "step", "team_rows",
              "close")


@dataclass
class Task:
    """What a match of the stand-in is (subclass and override). ``name``,
    ``n_teams`` marine teams of ``n_slots`` roster places, the controller
    of each team (``"policy"`` = a learning team), the decisions a match
    lasts, and :attr:`best`, the best possible score per decision."""
    name: str = "task"
    n_teams: int = 1
    n_slots: int = 1
    controllers: tuple = ("policy",)
    match_len: int = 8
    best: float = 1.0

    def roster(self, e: int, rng) -> np.ndarray:
        """``(U,)`` bool: which roster places hold a unit this match."""
        return np.ones(self.n_teams * self.n_slots, dtype=bool)

    def start(self, e: int, rng):
        """A new match's state."""
        return {}

    def observe(self, e: int, state, obs: dict, full: dict, valid, rng):
        """Write the task's signal and masks over env ``e``'s ``(U, ...)``
        rows (living rows: ``valid``)."""

    def pay(self, e: int, state, actions, valid, rng):
        """One step on ``actions`` ``(U, 5)``: returns ``(reward (U, G),
        deaths (U,) bool, rows_end (U,) bool, over bool)``."""
        u = self.n_teams * self.n_slots
        return (np.zeros((u, len(REWARD_GROUPS))), np.zeros(u, bool),
                np.zeros(u, bool), False)


class StandIn:
    """Behaves as ``rl_env.BatchedEnv`` does (see the module text)."""

    def __init__(self, task: Task, n_envs: int, seed: int = 0,
                 profile: str = "fireteam"):
        from simulation.weapons import get_tables
        self.task = task
        n, T, S = int(n_envs), int(task.n_teams), int(task.n_slots)
        K = len(get_tables().hand_grenade_rows)
        self.layout = rl_env.Layout(
            n_envs=n, n_teams=T, n_slots=S, n_actions=len(A.HEADS),
            channels=(), crop_radius=0, vector=(), spec="v1",
            full_view=True, action_spec="v1", n_kinds=K,
            reward_profile=profile)
        terms = (tuple(sorted(PROFILES[profile].items()))
                 if profile in PROFILES else ())
        self.run = rl_env.RunConfig(
            seed=int(seed), n_envs=n, decimation=1, levels=(task.name,),
            policy=tuple(task.controllers), levels_dir="", channels=(),
            crop_radius=0, vector=(), reward=(), spec="v1", full_view=True,
            action_spec="v1", reward_profile=profile, reward_terms=terms)
        self.sizes = sizes_for(S, T, K)
        self.rng = np.random.default_rng(int(seed))
        U = self.layout.n_rows
        self.U = U
        self._closed = False
        # the env-owned buffers (behaviour 1), from the game's shape tables
        shapes = self.layout.shapes()
        self._obs_keys = self.layout.obs_keys
        self._full_keys = self.layout.full_keys
        self._buf = {}
        for k in self._obs_keys + self._full_keys:
            for key in (k, "final_" + k):
                shape, dt = shapes[key]
                self._buf[key] = np.zeros(shape, dtype=dt)
        self._reward = np.zeros(shapes["reward"][0], dtype=np.float32)
        # behaviour 7: a fixed made-up scenery per row, in range
        bg = fake.make_obs(self.sizes, n * U, self.rng, p_invalid=0.0,
                           p_windup=0.0)
        self._bg_obs = {k: bg[k].reshape((n, U) + bg[k].shape[1:])
                        for k in O.OBS_KEYS}
        fv = fake.make_full(self.sizes, n * U, self.rng)
        self._bg_full = {k: fv[k].reshape((n, U) + fv[k].shape[1:])
                         for k in self._full_keys}
        # per env: the match's roster, life, open records, state, clock
        self.in_match = np.zeros((n, U), dtype=bool)
        self.alive = np.zeros((n, U), dtype=bool)
        self.open = np.zeros((n, U), dtype=bool)
        self.state = [None] * n
        self.clock = np.zeros(n, dtype=np.int64)
        self.episode = np.full(n, -1, dtype=np.int64)
        self.tick = np.zeros(n, dtype=np.int64)
        self._final_rows = np.zeros(n, dtype=bool)
        self._started = np.zeros(n, dtype=bool)
        ctrls = self.run.team_controllers(T)
        self.learning = np.zeros(U, dtype=bool)
        for t, c in enumerate(ctrls):
            if c == "policy":
                self.learning[t * S:(t + 1) * S] = True
        self.paid = np.zeros(len(REWARD_GROUPS))
        self.paid_decisions = 0

    # -- plumbing ---------------------------------------------------------
    @property
    def n_envs(self) -> int:
        return self.layout.n_envs

    def team_rows(self, obs: dict, team: int) -> dict:
        """Views of team ``team``'s rows of every key (as the game's)."""
        S = self.layout.n_slots
        t = int(team)
        if not 0 <= t < self.layout.n_teams:
            raise ValueError(f"team {team} is not a row block of this run "
                             f"(T = {self.layout.n_teams})")
        return {k: v[:, t * S:(t + 1) * S] for k, v in obs.items()}

    def close(self) -> None:
        self._closed = True

    # -- a match ----------------------------------------------------------
    def _new_match(self, e: int) -> None:
        self.episode[e] += 1
        self.tick[e] = 0
        self.clock[e] = 0
        self.in_match[e] = self.task.roster(e, self.rng)
        self.alive[e] = self.in_match[e]
        self.open[e] = self.in_match[e]
        self.state[e] = self.task.start(e, self.rng)

    def _observe(self, e: int, prefix: str = "") -> None:
        """Env ``e``'s rows of the observation (or with ``prefix``
        ``"final_"`` its terminal one) and full view."""
        b = self._buf
        obs = {k: b[prefix + k][e] for k in self._obs_keys}
        full = {k: b[prefix + k][e] for k in self._full_keys}
        valid = self.alive[e] & self.in_match[e]
        for k, v in self._bg_obs.items():
            obs[k][...] = v[e]
        obs["valid"][...] = valid
        obs["in_match"][...] = self.in_match[e]
        # game-like masks for the living rows (behaviour 7)
        obs["mask_feet"][...] = 1
        obs["mask_trigger"][...] = 1
        aim = obs["mask_aim"]
        aim[...] = 0
        aim[:, :A.AIM_KEEP] = 1
        tp = fake._type_pos()
        aimable = (obs["list_mask"] != 0) & np.isin(
            obs["list"][..., tp].astype(np.int64), A.AIMABLE_TYPES)
        aim[:, A.AIM_TOKENS:] = aimable
        hand = obs["mask_hand"]
        hand[...] = 0
        hand[:, A.HAND_NOTHING] = 1
        hand[:, A.HAND_RELOAD] = 1
        for k, v in self._bg_full.items():
            full[k][...] = np.where(
                self.in_match[e].reshape((-1,) + (1,) * (v.ndim - 2)),
                v[e], 0)
        self.task.observe(e, self.state[e], obs, full, valid, self.rng)
        # the full view is built for the rows in the match only
        for k in self._full_keys:
            full[k][~self.in_match[e]] = 0
        # behaviour 5: a row with valid 0 is all zeros, the default masks
        for i in np.flatnonzero(~valid):
            for k in self._obs_keys:
                obs[k][i] = 0
            obs["list_uid"][i] = -1
            obs["in_match"][i] = self.in_match[e, i]
            A.default_masks(obs, int(i))

    # -- the API ----------------------------------------------------------
    def _obs(self) -> dict:
        return {k: self._buf[k] for k in self._obs_keys}

    def _full(self) -> dict:
        return {k: self._buf[k] for k in self._full_keys}

    def reset(self, mask=None):
        """Start the next match of the masked envs (all when None)."""
        if self._closed:
            raise RuntimeError("this stand-in is closed")
        n = self.n_envs
        mask = np.ones(n, dtype=bool) if mask is None else np.asarray(mask)
        if mask.shape != (n,) or mask.dtype != bool:
            raise ValueError(f"reset mask must be a ({n},) bool array")
        for e in np.flatnonzero(mask):
            self._new_match(int(e))
            self._observe(int(e))
        self._started |= mask
        info = {"tick": self.tick.copy(), "episode": self.episode.copy(),
                "full_view": self._full()}
        return self._obs(), info

    def step(self, actions):
        """One decision of every env (see the module text)."""
        if self._closed:
            raise RuntimeError("this stand-in is closed")
        if not self._started.all():
            raise RuntimeError("step() before reset()")
        actions = np.asarray(actions)
        want = (self.n_envs, self.U, len(A.HEADS))
        if actions.shape != want or not np.issubdtype(actions.dtype,
                                                      np.integer):
            raise ValueError(f"actions must be an integer {want} array, got "
                             f"{actions.shape} {actions.dtype}")
        A.check_actions(actions, self.layout.head_sizes)
        masked = A.masked_choices(actions, {k: self._buf[k]
                                            for k in A.MASK_KEYS})
        n, U = self.n_envs, self.U
        w_valid = self._buf["valid"].astype(bool)          # at the decision
        rows_ended = np.zeros((n, U), dtype=bool)
        ended = np.zeros(n, dtype=bool)
        result = [None] * n
        # behaviour 2: last step's final rows are wiped
        b = self._buf
        if self._final_rows.any():
            for k in self._obs_keys + self._full_keys:
                b["final_" + k][self._final_rows] = 0
        for e in range(n):
            r, deaths, rend, over = self.task.pay(
                e, self.state[e], actions[e], w_valid[e], self.rng)
            r = np.asarray(r, dtype=np.float64)
            # behaviour 3: nothing for an empty place or a closed record
            r = np.where((self.in_match[e] & self.open[e])[:, None], r, 0.0)
            self._reward[e] = r
            learn = self.learning & w_valid[e]
            self.paid += r[learn].sum(axis=0)
            self.paid_decisions += int(learn.sum())
            self.alive[e] &= ~np.asarray(deaths, dtype=bool)
            self.clock[e] += 1
            self.tick[e] += 1
            over = bool(over) or self.clock[e] >= self.task.match_len
            rend = np.asarray(rend, dtype=bool) & self.open[e]
            if over:
                rend = rend | self.open[e]
            rows_ended[e] = rend
            self.open[e] &= ~rend
            if over:
                ended[e] = True
                result[e] = {"outcome": "over", "team": -1,
                             "ticks": int(self.tick[e])}
                self._observe(e, "final_")
                self._new_match(e)
            self._observe(e)
        self._final_rows = ended.copy()
        info = {"tick": self.tick.copy(), "episode": self.episode.copy(),
                "ended": ended,
                "final_obs": {k: b["final_" + k] for k in self._obs_keys},
                "final_full_view": {k: b["final_" + k]
                                    for k in self._full_keys},
                "full_view": self._full(), "result": result,
                "rows_ended": rows_ended, "masked_choices": masked}
        return (self._obs(), self._reward.copy(), ended.copy(),
                np.zeros(n, dtype=bool), info)


__all__ = ["INFO_KEYS", "ATTRIBUTES", "Task", "StandIn"]
