"""agents/collector.py -- the collector (rung 2 note 2b, 3; design 4.1,
4.2): takes the soldier's weights, plays T decisions in each of the N
matches, returns the batch.

Per step, for all learning rows in one pass:

1. the learning rows of the served arrays (through ``env.team_rows``; the
   full view from ``info``, the same rows) are COPIED into the batch's slot
   for this step -- the environment overwrites its arrays at its next step;
2. the soldier's network reads that slot (without gradients) and
   ``choosing.draw`` draws the five choices under the slot's masks: the
   network acts on exactly the arrays that are kept for learning;
3. every row the trainer does not drive is sent the legal default (stand,
   aim where I walk -- or "keep" while a throw is wound up, the one legal
   aim then --, no trigger, hand nothing, 0); a learning row that is dead
   or empty carries one legal option per head, so its draw IS the default;
4. the step; ``info["masked_choices"]`` on the driven living rows must be
   0, anything else stops the run naming the env and the row;
5. the step's rewards and endings are stored, and the row bookkeeping
   moves: a row's story ends when its match ends or its records end
   (``rows_ended``); a match that ended was restarted inside the step, so
   the next decision is that row's ``first``.

Between rounds nothing is reset: the next round goes on from the matches as
they stand.

THE MEMORY (patch 2d; design 4.5), with the soldier's memory on: every
learning row's memory is carried from step to step AND from round to round
(a match does not end with the round); it is zero only at the run's first
reset and, inside the network, where ``first`` says a match starts -- never
at a round's or a stretch's edge. At step ``t`` the memory carried in is
stored as ``batch.memory[t]`` (BEFORE the decision, before the network
zeroes it) and the network acts from that stored copy with that step's
``first`` flags; after the last step the memory carried out is stored as
``batch.memory[T]``. A dead or empty row is stepped like the rest (its
memory is whatever the network makes of the served zeros) and is never
learnt from. After the last step the view the value network reads is copied
once (``last_*``) for the value after the last stored decision.

The collector holds its OWN copy of the soldier's network; the learner
hands it weights (:meth:`Collector.set_weights`) and it hands back a batch:
one hand-over each way. When a round's collecting starts it writes the
batch's header (patch 2d.2): the version of the weights it was handed and
the run's names, which the learner compares with its own.

WHAT A ROUND OF COLLECTING MEASURED (patch 2c's per-round line; read by
``agents/run.py``, never a second walk over the environment):
:attr:`Collector.seconds` -- the wall seconds of the round split into the
environment's step (waiting for the workers), the network's answers (the
forward pass, the draw and the answers' copy off the card) and the copying
(the served rows into the batch, the defaults, the step's rewards and the
bookkeeping); :attr:`Collector.log` -- the matches that ended in the round
(:class:`EndedMatch`: per env and learning team, the level it was played on,
how it ended read off the books' counters of the team's rows, and the
reward per group its rows were paid over the whole match, rounds before
included), the books' counters summed over the learning rows
(``env.layout.counters`` order) and the masked choices counted on driven
living rows.
"""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass

import numpy as np
import torch

from agents import _game  # noqa: F401
from agents.batch import (FULL_STORED, LAST_FULL, LAST_OBS, OBS_STORED,
                          Batch, LearningRows, run_ids)
from agents.choosing import draw, masks_of
from agents.network import SOLDIER_KEYS, as_tensors
from simulation import action_spec_v1 as A


class MaskedChoiceSent(RuntimeError):
    """The environment counted a masked choice on a row the trainer drives
    (it must never: the draw is under the served masks). ``count``: the
    masked choices counted on the driven living rows in that step."""

    def __init__(self, msg: str, count: int = 0):
        super().__init__(msg)
        self.count = int(count)


#: How a match ended for a learning team, read off the books' counters of
#: its rows at the match's last step (``reward_v1.BOOK_COUNTERS``), in this
#: order of precedence; "other" when none was counted (the stand-in, which
#: keeps no counters).
OUTCOMES = ("won", "clock", "eliminated", "other")


@dataclass
class EndedMatch:
    """One match that ended, for one learning team: the env, the team, the
    level index it was played on, how it ended (:data:`OUTCOMES`) and the
    reward per group its rows were paid over the whole match."""
    env: int
    team: int
    level: int
    outcome: str
    reward: np.ndarray


def _outcome(counters, names) -> str:
    """How a match ended for one team, from its rows' summed counters."""
    for o in OUTCOMES[:-1]:
        if o in names and counters[names.index(o)] > 0:
            return o
    return OUTCOMES[-1]


def default_actions(masks: dict, n_heads: int) -> np.ndarray:
    """The legal default for every row of the served ``masks`` (``(N, U,
    size)`` each): stand, aim where I walk (or "keep" where that is the one
    legal aim), no trigger, hand nothing, argument 0."""
    aim = masks["mask_aim"]
    out = np.zeros(aim.shape[:-1] + (n_heads,), dtype=np.int64)
    out[..., 0] = A.FEET_STAND
    out[..., 1] = np.where(aim[..., A.AIM_WALK] != 0, A.AIM_WALK, A.AIM_KEEP)
    out[..., 2] = 0
    out[..., 3] = A.HAND_NOTHING
    out[..., 4] = 0
    return out


class Collector:
    """Plays the learning rows of ``env`` (a ``BatchedEnv`` or the
    stand-in) with its own copy of ``soldier``."""

    def __init__(self, env, soldier, rows: LearningRows, device="cpu",
                 seed: int = 0, ids: dict | None = None):
        self.env = env
        #: the run's names written into every batch's header, and the
        #: version of the weights it plays with (set at each hand-over)
        self.ids = run_ids(ids)
        self.version = 0
        self.rows = rows
        self.device = torch.device(device)
        self.soldier = copy.deepcopy(soldier).to(self.device).eval()
        for p in self.soldier.parameters():
            p.requires_grad_(False)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(seed))
        self.env_of_row = rows.env_of_row()
        self._obs = self._full = None
        self._memory = None           # (R, core) tensor; None: no memory
        R = rows.n_rows
        self._first = np.ones(R, dtype=bool)
        self._story = np.ones(R, dtype=bool)
        self.decisions = 0            # learning decisions stored, all rounds
        self._level = None            # each env's level at the last view
        self._paid = None             # (R, groups): set at start()
        self._names = tuple(getattr(env.layout, "counters", ()))
        self.seconds = {"env_step": 0.0, "answers": 0.0, "copy": 0.0}
        self.log = {"ended": [], "counters": np.zeros(len(self._names)),
                    "masked": 0}

    # -- the hand-over: weights in --------------------------------------
    def set_weights(self, weights: dict, version: int) -> None:
        """The soldier's weights the next round is played with, and their
        version (written into the batch's header)."""
        self.soldier.load_state_dict(weights)
        self.version = int(version)

    # -- the matches ----------------------------------------------------
    def start(self) -> None:
        """Start every match (the run's first reset). Called once; between
        rounds nothing is reset."""
        obs, info = self.env.reset()
        self._obs, self._full = obs, info["full_view"]
        self._level = np.asarray(info["level"]).copy()
        self._first[...] = True
        self._story[...] = True
        self._memory = self.soldier.initial_state(self.rows.n_rows,
                                                  self.device)
        self._paid = np.zeros((self.rows.n_rows,
                               self.env.layout.reward_shape_tail[0]))

    def _slot(self, batch: Batch, t: int):
        """The batch's arrays for step ``t`` by served key (the soldier's
        input, the masks, the two flags) and the full view's."""
        obs = {k: batch.obs[k][t] for k in OBS_STORED}
        obs["valid"] = batch.valid[t]
        obs["in_match"] = batch.in_match[t]
        full = {k: batch.full[k][t] for k in FULL_STORED}
        return obs, full

    def _gather_step(self, batch: Batch, t: int):
        """COPY the learning rows of the served arrays into step ``t``'s
        slot; return the slot (what the network acts on)."""
        obs, full = self._slot(batch, t)
        self.rows.gather(self._obs, obs, tuple(obs))
        self.rows.gather(self._full, full, tuple(full))
        return obs, full

    @torch.no_grad()
    def _act(self, obs: dict, state=None, first=None) -> dict:
        """The soldier's answers to ``obs`` as numpy arrays (copied off the
        card: the round's "answers" seconds include that wait); with the
        memory on, from the memory ``state`` with the ``first`` flags, and
        the memory after the decision under ``"memory"``."""
        x = as_tensors({k: obs[k] for k in SOLDIER_KEYS}, self.device)
        masks = {k: v.to(self.device)
                 for k, v in masks_of({k: obs[k] for k in A.MASK_KEYS}).items()}
        out, new = self.soldier(x, state, first)
        d = draw(out, masks, self.generator)
        got = {k: getattr(d, k).cpu().numpy()
               for k in ("actions", "used", "logp_heads", "logp",
                         "arg_mask")}
        got["memory"] = new
        return got

    def _act_step(self, batch: Batch, t: int, obs: dict) -> dict:
        """Step ``t``'s decision: with the memory on, store the memory
        carried in as ``batch.memory[t]`` and act from that stored copy with
        the step's ``first`` flags; carry the memory after it."""
        if self._memory is None:
            return self._act(obs)
        batch.memory[t] = self._memory.cpu().numpy()
        state = torch.as_tensor(batch.memory[t], device=self.device)
        first = torch.as_tensor(batch.first[t] != 0, device=self.device)
        d = self._act(obs, state, first)
        self._memory = d["memory"]
        return d

    def _begin_round(self) -> None:
        for k in self.seconds:
            self.seconds[k] = 0.0
        self.log = {"ended": [], "counters": np.zeros(len(self._names)),
                    "masked": 0}

    def _log_step(self, info: dict, ended, reward) -> None:
        """The round log's share of one step (module text): the counters
        over the learning rows; the reward paid so far in each row's match;
        for every env that ended, one :class:`EndedMatch` per learning team
        (its level is the one the ended match was shown on, from the view
        before the step: ``info["level"]`` is already the next match's)."""
        rows, S = self.rows, self.rows.n_slots
        L = len(rows.teams)
        cnt = None
        if self._names:
            got = {"c": np.zeros((rows.n_rows, len(self._names)),
                                 dtype=np.float64)}
            rows.gather({"c": info["counters"]}, got, ("c",))
            cnt = got["c"]
            self.log["counters"] += cnt.sum(axis=0)
        self._paid += reward
        for e in np.flatnonzero(ended):
            for li, t in enumerate(rows.teams):
                r0 = (int(e) * L + li) * S
                block = slice(r0, r0 + S)
                c = (cnt[block].sum(axis=0) if cnt is not None
                     else np.zeros(0))
                self.log["ended"].append(EndedMatch(
                    env=int(e), team=int(t), level=int(self._level[e]),
                    outcome=_outcome(c, self._names),
                    reward=self._paid[block].sum(axis=0).copy()))
                self._paid[block] = 0.0
        self._level = np.asarray(info["level"]).copy()

    def collect(self, batch: Batch) -> Batch:
        """Play ``batch.T`` decisions in every match; fill and return
        ``batch``."""
        if self._obs is None:
            raise RuntimeError("collect() before start()")
        if (self._memory is None) != (batch.memory is None):
            raise ValueError("the batch must keep the memory exactly when "
                             "the soldier has one (Batch(..., core=))")
        env, rows = self.env, self.rows
        batch.set_header(version=self.version, **self.ids)
        self._begin_round()
        sec = self.seconds
        clock = time.perf_counter
        for t in range(batch.T):
            t0 = clock()
            obs, _ = self._gather_step(batch, t)
            batch.first[t] = self._first
            batch.value_row[t] = (batch.in_match[t] != 0) & self._story
            t1 = clock()
            d = self._act_step(batch, t, obs)
            t2 = clock()
            batch.actions[t] = d["actions"]
            batch.used[t] = d["used"]
            batch.logp_heads[t] = d["logp_heads"]
            batch.logp[t] = d["logp"]
            batch.arg_mask[t] = d["arg_mask"]
            sent = default_actions({k: self._obs[k] for k in A.MASK_KEYS},
                                   len(A.HEADS))
            rows.scatter(batch.actions[t], sent)
            t3 = clock()
            new_obs, reward, _, _, info = env.step(sent)
            t4 = clock()
            self._check_masked(info, batch.valid[t])
            got = {"reward": np.zeros_like(batch.reward[t]),
                   "rows_ended": np.zeros(rows.n_rows, dtype=bool)}
            rows.gather({"reward": reward, "rows_ended": info["rows_ended"]},
                        got, ("reward", "rows_ended"))
            batch.reward[t] = got["reward"]
            ended = np.asarray(info["ended"], dtype=bool)[self.env_of_row]
            done = ended | got["rows_ended"]
            batch.done[t] = done
            # the bookkeeping: a story ends at its ending; a match that ended
            # was restarted inside the step (its next decision is a first)
            self._story &= ~done
            self._story |= ended
            self._first = ended.copy()
            self._obs, self._full = new_obs, info["full_view"]
            self.decisions += int(batch.valid[t].sum())
            self._log_step(info, np.asarray(info["ended"], dtype=bool),
                           got["reward"])
            t5 = clock()
            sec["copy"] += (t1 - t0) + (t3 - t2) + (t5 - t4)
            sec["answers"] += t2 - t1
            sec["env_step"] += t4 - t3
        t6 = clock()
        if self._memory is not None:
            batch.memory[batch.T] = self._memory.cpu().numpy()
        self._gather_last(batch)
        sec["copy"] += clock() - t6
        return batch

    def _gather_last(self, batch: Batch) -> None:
        """COPY the view after the last step (the value network's keys)."""
        self.rows.gather(self._obs, batch.last_obs, LAST_OBS)
        self.rows.gather(self._full, batch.last_full, LAST_FULL)

    def _check_masked(self, info: dict, valid) -> None:
        if "masked_choices" not in info:
            raise RuntimeError("the environment served no 'masked_choices' "
                               "(an action spec other than 'v1'?)")
        got = {"m": np.zeros(self.rows.n_rows, dtype=np.int64)}
        self.rows.gather({"m": info["masked_choices"]}, got, ("m",))
        bad = np.flatnonzero((got["m"] != 0) & (np.asarray(valid) != 0))
        n = int(got["m"][bad].sum()) if len(bad) else 0
        self.log["masked"] += n
        if len(bad):
            r = int(bad[0])
            raise MaskedChoiceSent(
                f"{int(got['m'][r])} masked choice(s) sent on a driven "
                f"living row ({self.rows.name(r)}); {len(bad)} row(s) in all",
                count=n)


__all__ = ["MaskedChoiceSent", "OUTCOMES", "EndedMatch", "default_actions",
           "Collector"]
